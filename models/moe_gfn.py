from torch import nn
from typing import Optional, Tuple
import torch
from torch.nn import functional as F

from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.unets.unet_2d_blocks import (
    AutoencoderTinyBlock,
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)
from einops import rearrange, repeat
from .attention import SpatialTransformer, FrequencyTransformer

import torch
import numpy as np
import matplotlib.pyplot as plt

import timm

import timm

class MoEGuidedForensicNet(nn.Module):
    """
    Fuses the 2nd and 3rd feature maps (features[1], features[2]) from ConvNext
    and produces a full-resolution, pixel-wise prediction.
    """
    def __init__(self, model_name='convnext_tiny', in_chans=3, out_chans=1, dec_dim=32):
        super().__init__()

        # 1. Load a pre-trained ConvNext backbone (frozen).
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            in_chans=in_chans
        )
        for param in self.model.parameters():
            param.requires_grad = False

        feature_info = self.model.feature_info.channels()
        f1_channels = feature_info[1] # e.g., 192 for convnext_tiny
        f2_channels = feature_info[2] # e.g., 384 for convnext_tiny

        # 2. Layer to process features[2] to match features[1].
        # Upsamples f2 by 2x (e.g., H/16 -> H/8) and matches channel count of f1.
        self.fuse_layer = nn.ConvTranspose2d(f2_channels, f1_channels, kernel_size=2, stride=2)

        # Bottleneck layer to fuse concatenated features (f1 + upsampled f2).
        self.bottleneck_conv = nn.Conv2d(f1_channels * 2, f1_channels, kernel_size=1, bias=False)

        # 3. Upsampler module redesigned for 8x upsampling (H/8 -> H).
        self.decoder = nn.Sequential(
            # Input: [B, f1_channels, H/8, W/8]
            nn.Conv2d(f1_channels, f1_channels, kernel_size=3, padding=1),
            nn.GELU(),
            
            # Upsample 2x: H/8 -> H/4
            nn.ConvTranspose2d(f1_channels, f1_channels // 2, kernel_size=2, stride=2),
            nn.GELU(),
            
            # Upsample 2x: H/4 -> H/2
            nn.ConvTranspose2d(f1_channels // 2, f1_channels // 4, kernel_size=2, stride=2),
            nn.GELU(),

            # Upsample 2x: H/2 -> H
            nn.ConvTranspose2d(f1_channels // 4, dec_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        
        # 4. Final 1x1 Conv layer for pixel-wise prediction.
        self.last_layer = nn.Conv2d(dec_dim, out_chans, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        original_size = x.shape[-2:]

        features = self.model(x)
        f1 = features[1] # Mid-level features (e.g., [B, 192, H/8, W/8])
        f2 = features[2] # High-level features (e.g., [B, 384, H/16, W/16])
        # ---------------------------------------------------

        f2_upsampled = self.fuse_layer(f2)
        
        concatenated_features = torch.cat([f1, f2_upsampled], dim=1)

        fused_features = self.bottleneck_conv(concatenated_features)

        upsampled_features = self.decoder(fused_features)

        prediction = self.last_layer(upsampled_features)
        
        prediction = F.interpolate(prediction, size=original_size, mode='bilinear', align_corners=False)
        
        return prediction


# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))
    
# class Stem(nn.Module):
#     """
#     StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
#     Modified StemBlock to dynamically adjust downsampling rate.
#     """

#     def __init__(self, c1, cm, c2):
#         """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
#         super().__init__()
#         self.stem1 = Conv(c1, cm, 3, 2)
#         self.stem2a = Conv(cm, cm // 2, 2, 1, 0)
#         self.stem2b = Conv(cm // 2, cm, 2, 1, 0)
#         self.stem3 = Conv(cm * 2, cm, 3, 2)
#         self.stem4 = Conv(cm, c2, 1, 1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

#     def forward(self, x):
#         """Forward pass with dynamic downsampling."""
#         # Determine input size dynamically
#         input_size = x.shape[-1]  # Assume square input (H == W)
        
#         # Original stem pipeline
#         x = self.stem1(x)
#         x = F.pad(x, [0, 1, 0, 1])
#         x2 = self.stem2a(x)
#         x2 = F.pad(x2, [0, 1, 0, 1])
#         x2 = self.stem2b(x2)
#         x1 = self.pool(x)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.stem3(x)
#         x = self.stem4(x)

#         return x


# class UpSample(nn.Module):
#     """
#     UpSampleBlock designed to upsample the input features back to the original size.
#     The module uses a combination of transposed convolutions and interpolation.
#     """

#     def __init__(self, c1, cm, c2):
#         """
#         Args:
#             c1 (int): Number of input channels.
#             cm (int): Intermediate channel size for processing.
#             c2 (int): Number of output channels (matching the original size).
#         """
#         super().__init__()
#         self.up1 = nn.ConvTranspose2d(c1, cm, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(cm)
#         self.act1 = nn.SiLU()

#         self.up2a = Conv(cm, cm // 2, k=1, s=1)
#         self.up2b = Conv(cm // 2, cm, k=1, s=1)

#         self.up3 = nn.ConvTranspose2d(cm, cm, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(cm)
#         self.act3 = nn.SiLU()

#         self.up4 = Conv(cm, c2, k=1, s=1)

#     def forward(self, x):
#         """
#         Forward pass of the upsampling module.

#         Args:
#             x (Tensor): Input feature map of shape (B, C, H, W).

#         Returns:
#             Tensor: Upsampled feature map of shape (B, c2, H, W).
#         """
#         # First upsampling stage
#         x = self.act1(self.bn1(self.up1(x)))  # Transposed convolution to upsample by 2x

#         # Additional processing with intermediate convolutions
#         x2 = self.up2a(x)
#         x2 = self.up2b(x2)

#         # Second upsampling stage
#         x = self.act3(self.bn3(self.up3(x + x2)))  # Combine with intermediate features and upsample again

#         # Final 1x1 convolution for channel adjustment
#         x = self.up4(x)

#         return x


# class MoFEBlock(nn.Module):
#     '''
#     Mixture-of-Forensic-Experts Block
#     '''
#     def __init__(self, input_channels, num_experts=3):
#         super().__init__()
#         # three forensic expert
#         self.watermark_expert = SpatialTransformer(in_channels=input_channels, n_heads=8, d_head=input_channels // 8)
#         self.tampering_expert = SpatialTransformer(in_channels=input_channels, n_heads=8, d_head=input_channels // 8, local_attention=True)
#         self.boundary_expert = FrequencyTransformer(in_channels=input_channels, n_heads=8, d_head=input_channels // 8)

#         # projection net
#         self.watermark_proj = nn.Sequential(
#             nn.Linear(input_channels, input_channels),
#             nn.LayerNorm(input_channels),
#             nn.SiLU(),
#             nn.Linear(input_channels, input_channels)
#         )
#         self.tampering_proj = nn.Sequential(
#             nn.Linear(input_channels, input_channels),
#             nn.LayerNorm(input_channels),
#             nn.SiLU(),
#             nn.Linear(input_channels, input_channels)
#         )
#         self.boundary_proj = nn.Sequential(
#             nn.Linear(input_channels, input_channels),
#             nn.LayerNorm(input_channels),
#             nn.SiLU(),
#             nn.Linear(input_channels, input_channels)
#         )

#         # gate net
#         self.gate = nn.Sequential(
#             nn.Linear(input_channels, input_channels),
#             nn.SiLU(),
#             nn.Linear(input_channels, num_experts)
#         )
#         self.norm = nn.LayerNorm(input_channels)
    
#     def forward(self, input_feat): 
#         b, c, h, w = input_feat.size()  
#         skip_res = input_feat   

#         # expert output
#         watermark_feat = self.watermark_expert(input_feat)
#         tampering_feat = self.tampering_expert(input_feat)
#         boundary_feat = self.boundary_expert(input_feat)

#         # change shape
#         input_feat = rearrange(input_feat, 'b c h w -> b (h w) c').contiguous()
#         watermark_feat = rearrange(watermark_feat, 'b c h w -> b (h w) c').contiguous()
#         tampering_feat = rearrange(tampering_feat, 'b c h w -> b (h w) c').contiguous()   
#         boundary_feat = rearrange(boundary_feat, 'b c h w -> b (h w) c').contiguous() 

#         gate_weights = F.softmax(self.gate(input_feat), dim=-1)  

#         # proj
#         watermark_feat = self.watermark_proj(watermark_feat)
#         tampering_feat = self.tampering_proj(tampering_feat)
#         boundary_feat = self.boundary_proj(boundary_feat)
         
#         # fuse
#         fused_output = self.norm(
#             gate_weights[:, :, 0:1] * watermark_feat +  
#             gate_weights[:, :, 1:2] * tampering_feat +
#             gate_weights[:, :, 2:3] * boundary_feat            
#         )

#         fused_output = rearrange(fused_output, 'b (h w) c -> b c h w', h=h, w=w).contiguous()  

#         return fused_output + skip_res
    

# class Encoder(nn.Module):
#     '''
#     Mixture-of-Experts Guided Forensic Network (MoE-GFN) Encoder
#     '''
#     def __init__(
#         self,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
#         block_out_channels: Tuple[int, ...] = (64,),
#         layers_per_block: int = 2,
#         norm_num_groups: int = 16,
#         act_fn: str = "silu",
#     ):
#         super().__init__()

#         self.excoder = FeatureExtractor(model_name='convnext_tiny', in_chans=in_channels)

#         # mid
#         self.mid_block = UNetMidBlock2D(
#             in_channels=block_out_channels[-1],
#             resnet_eps=1e-6,
#             resnet_act_fn=act_fn,
#             output_scale_factor=1,
#             resnet_time_scale_shift="default",
#             attention_head_dim=block_out_channels[-1],
#             resnet_groups=norm_num_groups,
#             temb_channels=None,
#             add_attention=False,
#         )

#         # out for decoder
#         self.out = nn.Sequential(nn.Conv2d(block_out_channels[-1], block_out_channels[-1], 3, padding=1),
#                                  nn.SiLU(),
#                                  nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1))

#         # self.layers_per_block = layers_per_block

#         # self.conv_in = nn.Conv2d(
#         #     in_channels,
#         #     block_out_channels[0],
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1,
#         # )

#         # self.mid_block = None
#         # self.down_blocks = nn.ModuleList([])

#         # # down
#         # output_channel = block_out_channels[0]
#         # for i, down_block_type in enumerate(down_block_types):
#         #     input_channel = output_channel
#         #     output_channel = block_out_channels[i]
#         #     is_final_block = i == len(block_out_channels) - 1

#         #     down_block = get_down_block(
#         #         down_block_type,
#         #         num_layers=self.layers_per_block,
#         #         in_channels=input_channel,
#         #         out_channels=output_channel,
#         #         add_downsample=not is_final_block,
#         #         resnet_eps=1e-6,
#         #         downsample_padding=0,
#         #         resnet_act_fn=act_fn,
#         #         resnet_groups=norm_num_groups,
#         #         attention_head_dim=output_channel,
#         #         temb_channels=None,
#         #     )
#         #     self.down_blocks.append(down_block)


#     def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
#         r"""The forward method of the `Encoder` class."""
#         res = [] # residual
#         sample = self.conv_in(sample)
#         res.append(sample)
#         # down
#         for down_block in self.down_blocks:
#             sample = down_block(sample)
#             res.append(sample)

#         # middle
#         sample = self.mid_block(sample)
#         # for decoder
#         latent = self.out(sample)

#         return latent, res


# class Decoder(nn.Module):
#     '''
#     Mixture-of-Experts Guided Forensic Network (MoE-GFN) Decoder
#     '''
#     def __init__(
#         self,
#         in_channels: int = 3,
#         up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
#         block_out_channels: Tuple[int, ...] = (64,),
#         layers_per_block: int = 2,
#         norm_num_groups: int = 16,
#         act_fn: str = "silu",
#         norm_type: str = "group",  # group, spatial
#         num_bits: int = 64,
#     ):
#         super().__init__()
#         self.layers_per_block = layers_per_block

#         self.conv_in = nn.Conv2d(
#             in_channels,
#             block_out_channels[-1],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#         self.up_blocks = nn.ModuleList([])
#         self.mofe_blocks = nn.ModuleList([])

#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))

#         output_channel = reversed_block_out_channels[0]
#         for i, up_block_type in enumerate(up_block_types):
#             prev_output_channel = output_channel
#             output_channel = reversed_block_out_channels[i+1] if (i+1) < len(reversed_block_out_channels) else reversed_block_out_channels[i]

#             if_first_block = i == 0
#             up_block = get_up_block(
#                 up_block_type,
#                 num_layers=self.layers_per_block + 1,
#                 in_channels=prev_output_channel*2,
#                 out_channels=output_channel,
#                 prev_output_channel=None,
#                 add_upsample=not if_first_block,
#                 resnet_eps=1e-6,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 attention_head_dim=output_channel,
#                 temb_channels=None,
#                 resnet_time_scale_shift=norm_type,
#             )

#             self.up_blocks.append(up_block)
#             self.mofe_blocks.append(MoFEBlock(input_channels=output_channel))
#             prev_output_channel = output_channel

#         # mid
#         self.mid_block = UNetMidBlock2D(
#             in_channels=block_out_channels[0]*2,
#             resnet_eps=1e-6,
#             resnet_act_fn=act_fn,
#             output_scale_factor=1,
#             resnet_time_scale_shift="default",
#             attention_head_dim=block_out_channels[0],
#             resnet_groups=norm_num_groups,
#             temb_channels=None,
#             add_attention=False,
#         )

#         self.final_mofe_block = MoFEBlock(input_channels=block_out_channels[0]*2)

#         # mask
#         self.mask = nn.Sequential(UpSample(block_out_channels[0]*2, block_out_channels[0], block_out_channels[0]),
#                                   nn.Conv2d(block_out_channels[0], 3, 3, stride=1, padding=1),
#                                   nn.GroupNorm(num_groups=3, num_channels=3, affine=True),
#                                   nn.SiLU(),
#                                   nn.Conv2d(3, 1, 3,  stride=1, padding=1))
        
#         # msg
#         self.msg = nn.Sequential(nn.Conv2d(block_out_channels[0]*2, 32, 3, stride=1, padding=1),
#                                  nn.GroupNorm(num_groups=16, num_channels=32, affine=True),
#                                  nn.SiLU(),
#                                  nn.Conv2d(32, 4, 3,  stride=1, padding=1),
#                                  nn.AdaptiveAvgPool2d(64),
#                                  nn.Flatten(1),
#                                  nn.Linear(64*64*4, num_bits)
#                                  )

#     def forward(
#         self,
#         sample: torch.FloatTensor,
#         res: Optional[torch.FloatTensor] = None,  
#     ) -> torch.FloatTensor:                       
#         r"""The forward method of the `Decoder` class."""
#         res = list(reversed(res))
#         sample = self.conv_in(sample)

#         # up
#         for up_block, mofe_block, res_feat in \
#                 zip(self.up_blocks, self.mofe_blocks, res):
#             sample = up_block(torch.cat([sample, res_feat], dim=1))
#             sample = mofe_block(sample)

#         sample = self.mid_block(torch.cat([sample, res[-1]], dim=1))
#         sample = self.final_mofe_block(sample)

#         # post-process
#         mask = self.mask(sample)
#         msg = self.msg(sample)

#         return mask, msg


# class MoEGuidedForensicNet(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = 3,
#         out_channels: int = 1,
#         num_bits: int=64,
#         down_block_types: Tuple[str] = ("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
#         up_block_types: Tuple[str] = ("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"),
#         block_out_channels: Tuple[int] = (96,192,384,768),
#         layers_per_block: int = 2,
#         act_fn: str = "silu",
#         norm_num_groups: int = 16,
#     ):
#         super().__init__()
#         self.encoder = FeatureExtractor(model_name='convnext_tiny', in_chans=in_channels)

#         # feature map shapes
#         # [B, 3, 224, 224] -> [B, 96, 56, 56], [B, 192, 28, 28], [B, 384, 14, 14], [B, 768, 7, 7]
#         # [B, 3, 384, 384] -> [B, 96, 96, 96], [B, 192, 48, 48], [B, 384, 24, 24], [B, 768, 12, 12]

#         # mid
#         # self.mid_block = UNetMidBlock2D(
#         #     in_channels=block_out_channels[-1],
#         #     resnet_eps=1e-6,
#         #     resnet_act_fn=act_fn,
#         #     output_scale_factor=1,
#         #     resnet_time_scale_shift="default",
#         #     attention_head_dim=block_out_channels[-1],
#         #     resnet_groups=norm_num_groups,
#         #     temb_channels=None,
#         #     add_attention=False,
#         # )

#         # out for decoder
#         self.decoder = nn.Sequential(nn.Conv2d(block_out_channels[1], block_out_channels[0], 3, padding=1),
#                                  nn.SiLU(),
#                                  nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1))

#     def forward(self, x):
#         latent, _ = self.encoder(x)
#         mask = self.decoder(latent)
#         return mask
    
#     # def encoder_forward(self, x):
#     #     latent, residual = self.encoder(x)
#     #     return latent, residual
    
#     # def decoder_forward(self, latent, residual):
#     #     mask, msg = self.decoder(latent, residual)
#     #     return mask, msg
