from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.unets.unet_2d_blocks import (
    AutoencoderTinyBlock,
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
    

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs
    

# class MsgAdapter(nn.Module):
#     def __init__(self, in_channels, num_bits):
#         super(MsgAdapter, self).__init__()
#         self.num_bits = num_bits
        
#         self.secret_dense1 = Dense(self.num_bits, 32 * 32, activation='relu') 
#         self.secret_dense2 = Dense(32 * 32, 8 * 32 * 32, activation='relu') 
#         self.conv1 = Conv2D(in_channels+8, in_channels+8, 3, activation='relu')
#         self.conv2 = zero_module(Conv2D(in_channels+8, in_channels, 3, activation=None))
    
#     def forward(self, img_feature, secrect):
#         secrect = 2 * (secrect - .5)
#         secrect = self.secret_dense1(secrect)  
#         secrect = self.secret_dense2(secrect)  
#         secrect = secrect.reshape(-1, 8, 32, 32) 
#         scale_factor = img_feature.size(-1) // 32
#         secrect_enlarged = nn.Upsample(scale_factor=(scale_factor, scale_factor))(secrect)
#         inputs = torch.cat([secrect_enlarged, img_feature], dim=1) 
#         conv1 = self.conv1(inputs) 
#         conv2 = self.conv2(conv1) 
        
#         return conv2 + img_feature

# class WatermarkGenerator(nn.Module):
#     def __init__(self, vector_dim: int, latent_h: int, latent_w: int):
#         super().__init__()
#         self.learnable_vector = nn.Parameter(torch.randn(1, vector_dim, 1, 1))
#         self.target_h = latent_h
#         self.target_w = latent_w

#     def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#         B = input_tensor.shape[0]
#         # [1, vector_dim, 1, 1] -> [B, vector_dim, H, W]
#         watermark = self.learnable_vector.repeat(
#             B, 1, self.target_h, self.target_w
#         )
#         return watermark
    
# class DualAdapter(nn.Module):
#     def __init__(self, in_channels, num_bits):
#         super(DualAdapter, self).__init__()
#         self.num_bits = num_bits
        
#         # convert the secret message into a feature map with 8 channels
#         # self.secret_dense1 = Dense(self.num_bits, 32 * 32, activation='relu') 
#         # self.secret_dense2 = Dense(32 * 32, 8 * 32 * 32, activation='relu')
#         self.secret_conv1 = Conv2D(num_bits, 8, 1, activation='relu')


#         # spatial branch
#         self.conv1 = Conv2D(in_channels+8, in_channels+8, 3, activation='relu')
#         self.conv2 = zero_module(Conv2D(in_channels+8, in_channels, 3, activation=None))

#         # frequency branch
#         self.freq_conv1 = Conv2D(in_channels+8, in_channels+8, 3, activation='relu')
#         self.freq_conv2 = zero_module(Conv2D(in_channels+8, in_channels, 3, activation=None))
#         self.freq_conv3 = Conv2D(in_channels+8, in_channels+8, 3, activation='relu')
#         self.freq_conv4 = zero_module(Conv2D(in_channels+8, in_channels, 3, activation=None))

#     def forward(self, img_feature, secret):
#         # secret = 2 * (secret - .5)
#         # secret = self.secret_dense1(secret)  
#         # secret = self.secret_dense2(secret)  
#         # secret = secret.reshape(-1, 8, 32, 32) 

#         secret  = self.secret_conv1(secret)
#         scale_factor = img_feature.size(-1) // 32
#         secret_enlarged = nn.Upsample(scale_factor=(scale_factor, scale_factor))(secret)
        
#         ## two branches: spatial and frequency
#         # spatial branch
#         inputs = torch.cat([secret_enlarged, img_feature], dim=1)
#         conv1 = self.conv1(inputs) 
#         x_spatial = self.conv2(conv1)

#         # frequency branch
#         img_freq = torch.fft.rfft2(img_feature.float())
#         wm_freq = torch.fft.rfft2(secret_enlarged.float())

#         img_amp = torch.abs(img_freq)
#         img_phase = torch.angle(img_freq)
#         wm_amp = torch.abs(wm_freq)
#         wm_phase = torch.angle(wm_freq)

#         # [B, in_channels + 8, H, W//2+1]
#         inputs = torch.cat([wm_phase, img_phase], dim=1)
#         x_phase = self.freq_conv1(inputs)
#         x_phase = self.freq_conv2(x_phase)

#         inputs = torch.cat([wm_amp, img_amp], dim=1)
#         x_amp = self.freq_conv3(inputs)
#         x_amp = self.freq_conv4(x_amp)

#         x_freq = torch.polar(x_amp.float(), x_phase.float())
#         x_freq = torch.fft.irfft2(x_freq)  # [B, in_channels, H, W]

#         return img_feature + x_spatial + x_freq
    

class FreqAdapter(nn.Module):
    def __init__(self, c, h, w, num_bits):
        super(FreqAdapter, self).__init__()
        # self.watermark = nn.Parameter(torch.randn(1, c, h, (w // 2) + 1))
        self.num_bits = num_bits
        self.watermark = nn.Parameter(torch.randn(1, c, h, w))
        # self.secret_dense1 = Dense(self.num_bits, h * w, activation='relu') 
        # self.secret_dense2 = Dense(h * w, c * h * w, activation='relu')

    def forward(self, img_feature):
        # img_freq = torch.fft.rfft2(img_feature.float())
        # watermark = self.secret_dense1(msgs)
        # watermark = self.secret_dense2(watermark)
        # watermark = watermark.reshape(-1, img_feature.size(1), img_feature.size(2), img_feature.size(3))
        watermarked = img_feature + self.watermark
        # watermarked = torch.fft.irfft2(wm_freq, dim=(-2, -1)).real
        return watermarked

class MultiplexingWatermarkVAEDecoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"),
        block_out_channels: Tuple[int, ...] = (128,256,512,512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        num_bits=64,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.msg_adapters = nn.ModuleList([])
        self.num_bits = num_bits
        # self.watermark_generator = WatermarkGenerator(vector_dim=self.num_bits, latent_h=32, latent_w=32)

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )
        
        latent_h, latent_w = 32, 32
        
        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            if latent_h == 32 or latent_h == 64:
                # apply FreqAdapter for 32x32 and 64x64 resolutions
                self.msg_adapters.append(FreqAdapter(c=prev_output_channel, h=latent_h, w=latent_w, num_bits=num_bits))
            else:
                # no watermarking for other resolutions
                self.msg_adapters.append(nn.Identity())

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

            # double the latent size after each upsampling
            if not is_final_block:
                latent_h *= 2
                latent_w *= 2

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False


    def forward(
        self,
        input: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(input)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        
        # middle
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)

        # up
        for up_block, adapter in zip(self.up_blocks, self.msg_adapters): # add watermark
        # for up_block in self.up_blocks:
            sample = adapter(sample)
            sample = up_block(sample, latent_embeds)
        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample, sample-input