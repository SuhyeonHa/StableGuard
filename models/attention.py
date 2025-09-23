from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads



# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32 if in_channels > 32 else 8, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
        #       f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        assert not torch.any(torch.isnan(q))
        assert not torch.any(torch.isnan(k))
        assert not torch.any(torch.isnan(v))        
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        assert not torch.any(torch.isnan(out))
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, checkpoint=True):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        return checkpoint(self._forward, (x, ), self.parameters(), self.checkpoint)

    def _forward(self, x):
        x = self.attn(self.norm1(x)) + x
        assert not torch.any(torch.isnan(x))
        x = self.ff(self.norm2(x)) + x
        return x



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head, kernel_size=3,
                 depth=1, dropout=0., use_checkpoint=True, local_attention=False, window_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.local_attention = local_attention
        self.window_size = window_size
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels,inner_dim,
                                 kernel_size=kernel_size, stride=1, padding=1 if kernel_size > 1 else 0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        self.proj_out = nn.Conv2d(inner_dim, in_channels,
                                  kernel_size=kernel_size, stride=1, padding=1 if kernel_size > 1 else 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        assert not torch.any(torch.isnan(x)) 
        if not self.local_attention:
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        else:
            num_h = h // self.window_size
            num_w = w // self.window_size
            x = rearrange(x, 'b c (num_h ws_h) (num_w ws_w) -> (b num_h num_w) (ws_h ws_w) c', 
                          num_h=num_h, num_w=num_w, ws_h=self.window_size, ws_w=self.window_size).contiguous()
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
        assert not torch.any(torch.isnan(x))
        if not self.local_attention:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        else:
            x = rearrange(x, '(b h w) (ws_h ws_w) c -> b c (h ws_h) (w ws_w)',
                          ws_h=self.window_size, ws_w=self.window_size, h=num_h, w=num_w).contiguous()
        x = self.proj_out(x)
        assert not torch.any(torch.isnan(x))
        return x + x_in


class FrequencyTransformer(nn.Module):
    """
    Transformer block for frequency - domain data.
    Applies Fourier transform to input, processes in frequency domain,
    and then applies inverse Fourier transform to restore spatial domain.
    """
    def __init__(self, in_channels, n_heads, d_head, kernel_size=3,
                 depth=1, dropout=0., use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm_mag = Normalize(in_channels)
        self.norm_phase = Normalize(in_channels)
        self.proj_in_mag = nn.Conv2d(in_channels, inner_dim,
                                     kernel_size=kernel_size, stride=1, padding=1 if kernel_size > 1 else 0)
        self.proj_in_phase = nn.Conv2d(in_channels, inner_dim,
                                       kernel_size=kernel_size, stride=1, padding=1 if kernel_size > 1 else 0)

        self.transformer_blocks_mag = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, checkpoint=use_checkpoint)
             for d in range(depth)]
        )
        self.transformer_blocks_phase = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, checkpoint=use_checkpoint)
             for d in range(depth)]
        )

        self.proj_out_mag = nn.Conv2d(inner_dim, in_channels,
                                      kernel_size=kernel_size, stride=1, padding=1 if kernel_size > 1 else 0)
        self.proj_out_phase = nn.Conv2d(inner_dim, in_channels,
                                        kernel_size=kernel_size, stride=1, padding=1 if kernel_size > 1 else 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        ori_dtype = x.dtype
        # Step 1: Convert to frequency domain
        x_freq = torch.fft.fft2(x.float())
        x_mag = x_freq.abs().to(ori_dtype)  # Magnitude
        x_phase = x_freq.angle().to(ori_dtype)  # Phase

        # Step 2: Apply transformer processing to magnitude and phase separately
        x_mag = self.norm_mag(x_mag)
        x_mag = self.proj_in_mag(x_mag)
        x_mag = rearrange(x_mag, 'b c h w -> b (h w) c').contiguous()
        for i, block in enumerate(self.transformer_blocks_mag):
            x_mag = block(x_mag)
        x_mag = rearrange(x_mag, 'b (h w) c -> b c h w', h = h, w = w).contiguous()
        x_mag = self.proj_out_mag(x_mag)

        x_phase = self.norm_phase(x_phase)
        x_phase = self.proj_in_phase(x_phase)
        x_phase = rearrange(x_phase, 'b c h w -> b (h w) c').contiguous()
        for i, block in enumerate(self.transformer_blocks_phase):
            x_phase = block(x_phase)
        x_phase = rearrange(x_phase, 'b (h w) c -> b c h w', h = h, w = w).contiguous()
        x_phase = self.proj_out_phase(x_phase)

        # Step 3: Combine with phase and inverse transform
        x_freq = torch.polar(x_mag.float(), x_phase.float())  # Restore complex tensor
        x = torch.fft.ifft2(x_freq).real  # Inverse FFT to return to spatial domain
        x = x.to(ori_dtype)
        return x + x_in

