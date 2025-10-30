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

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel as UNet
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput, logger
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from tuner import Tuner

# class Dense(nn.Module):
#     def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
#         super(Dense, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.activation = activation
#         self.kernel_initializer = kernel_initializer

#         self.linear = nn.Linear(in_features, out_features)
#         # initialization
#         if kernel_initializer == 'he_normal':
#             nn.init.kaiming_normal_(self.linear.weight)
#         else:
#             raise NotImplementedError

#     def forward(self, inputs):
#         outputs = self.linear(inputs)
#         if self.activation is not None:
#             if self.activation == 'relu':
#                 outputs = nn.ReLU(inplace=True)(outputs)
#         return outputs


# class Conv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
#         super(Conv2D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.activation = activation
#         self.strides = strides

#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
#         # default: using he_normal as the kernel initializer
#         nn.init.kaiming_normal_(self.conv.weight)

#     def forward(self, inputs):
#         outputs = self.conv(inputs)
#         if self.activation is not None:
#             if self.activation == 'relu':
#                 outputs = nn.ReLU(inplace=True)(outputs)
#             else:
#                 raise NotImplementedError
#         return outputs
    

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



# class MultiplexingWatermarkVAEDecoder(nn.Module):
#     r"""p
#     The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output samle.

#     Args:
#         in_channels (`int`, *optional*, defaults to 3):
#             The number of input channels.
#         out_channels (`int`, *optional*, defaults to 3):
#             The number of output channels.
#         up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
#             The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
#         block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
#             The number of output channels for each block.
#         layers_per_block (`int`, *optional*, defaults to 2):
#             The number of layers per block.
#         norm_num_groups (`int`, *optional*, defaults to 32):
#             The number of groups for normalization.
#         act_fn (`str`, *optional*, defaults to `"silu"`):
#             The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
#         norm_type (`str`, *optional*, defaults to `"group"`):
#             The normalization type to use. Can be either `"group"` or `"spatial"`.
#     """

#     def __init__(
#         self,
#         in_channels: int = 4,
#         out_channels: int = 3,
#         up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"),
#         block_out_channels: Tuple[int, ...] = (128,256,512,512),
#         layers_per_block: int = 2,
#         norm_num_groups: int = 32,
#         act_fn: str = "silu",
#         norm_type: str = "group",  # group, spatial
#         mid_block_add_attention=True,
#         num_bits=64,
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

#         self.mid_block = None
#         self.up_blocks = nn.ModuleList([])
#         self.msg_adapters = nn.ModuleList([])
#         self.num_bits = num_bits
#         self.watermark_generator = WatermarkGenerator(vector_dim=self.num_bits, latent_h=32, latent_w=32)

#         temb_channels = in_channels if norm_type == "spatial" else None

#         # mid
#         self.mid_block = UNetMidBlock2D(
#             in_channels=block_out_channels[-1],
#             resnet_eps=1e-6,
#             resnet_act_fn=act_fn,
#             output_scale_factor=1,
#             resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
#             attention_head_dim=block_out_channels[-1],
#             resnet_groups=norm_num_groups,
#             temb_channels=temb_channels,
#             add_attention=mid_block_add_attention,
#         )

#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))
#         output_channel = reversed_block_out_channels[0]
#         for i, up_block_type in enumerate(up_block_types):
#             prev_output_channel = output_channel
#             output_channel = reversed_block_out_channels[i]

#             self.msg_adapters.append(DualAdapter(in_channels=prev_output_channel, num_bits=num_bits)) # add msg adapter to each layer

#             is_final_block = i == len(block_out_channels) - 1

#             up_block = get_up_block(
#                 up_block_type,
#                 num_layers=self.layers_per_block + 1,
#                 in_channels=prev_output_channel,
#                 out_channels=output_channel,
#                 prev_output_channel=None,
#                 add_upsample=not is_final_block,
#                 resnet_eps=1e-6,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 attention_head_dim=output_channel,
#                 temb_channels=temb_channels,
#                 resnet_time_scale_shift=norm_type,
#             )
#             self.up_blocks.append(up_block)
#             prev_output_channel = output_channel


#         # out
#         if norm_type == "spatial":
#             self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
#         else:
#             self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
#         self.conv_act = nn.SiLU()
#         self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

#         self.gradient_checkpointing = False


#     def forward(
#         self,
#         sample: torch.FloatTensor,
#         latent_embeds: Optional[torch.FloatTensor] = None,
#         msgs: torch.FloatTensor = None,
#     ) -> torch.FloatTensor:
#         r"""The forward method of the `Decoder` class."""

#         sample = self.conv_in(sample)
#         upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        
#         # middle
#         sample = self.mid_block(sample, latent_embeds)
#         sample = sample.to(upscale_dtype)

#         watermark = self.watermark_generator(sample)

#         # up
#         for up_block, adaptor in zip(self.up_blocks, self.msg_adapters): # add watermark
#             sample = adaptor(sample, watermark)
#             sample = up_block(sample, latent_embeds)
#         # post-process
#         if latent_embeds is None:
#             sample = self.conv_norm_out(sample)
#         else:
#             sample = self.conv_norm_out(sample, latent_embeds)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)
#         return sample
    

class Embedder(UNet): 
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        down_block_types: Tuple[str] = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        mid_block_type: str = "UNetMidBlock2DCrossAttn",

        # --- [추가] 어텐션 관련 중요 인수들 ---
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False, # SD 2.1은 False일 수 있음
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None, # attention_head_dim으로 대체될 수 있음
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # ---------------------------------

        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        upcast_attention: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,

        # (Class/Addition embedding 관련 인수들)
        projection_class_embeddings_input_dim: Optional[int] = None,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        # (기타 필요한 인수들...)
        dropout: float = 0.0,
        resnet_time_scale_shift: str = "default",


        # --- Embedder-specific arguments ---
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        # 1. 부모 클래스 (UNet) 초기화
        # [수정] 누락되었을 수 있는 어텐션 관련 인수 추가
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            mid_block_type=mid_block_type,

            only_cross_attention=only_cross_attention,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            num_attention_heads=num_attention_heads,
            transformer_layers_per_block=transformer_layers_per_block,

            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            upcast_attention=upcast_attention,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,

            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            dropout=dropout,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
        
        self.layers_per_block = layers_per_block
        
        self.tuners = nn.ModuleList()
        
        # (U-Net 1.5 스킵 채널: 3x320, 3x640, 3x1280, 3x1280)
        # skip_channels_list = []
        # for i in range(len(block_out_channels)):
        #     for _ in range(layers_per_block + 1): # (2 ResNet + 1 Downsampler)
        #         skip_channels_list.append(block_out_channels[i])
        
        # # (SD 1.5는 conv_in 스킵 피처를 사용하지 않고, 
        # #  down_blocks에서 4x3=12개의 스킵 피처를 생성합니다)
        # if len(skip_channels_list) != 12:
        #     # (Hugging Face diffusers의 UNet 1.5 로직 기준)
        #     skip_channels_list = []
        #     for ch in block_out_channels:
        #         for _ in range(layers_per_block + 1): # 3
        #             skip_channels_list.append(ch)
        
        # 조건 인코더의 출력 채널 (4개 레벨)
        # cond_channels_list = []
        # for ch in block_out_channels:
        #      for _ in range(layers_per_block + 1):
        #         cond_channels_list.append(ch)

        # 12개의 튜너 생성
        # self.tuners = nn.ModuleList()
        # for skip_ch, cond_ch in zip(skip_channels_list, cond_channels_list):
        #     self.tuners.append(Tuner(skip_ch, cond_ch, hidden_dim=tuner_hidden_dim))
            
        # # 4. Mid Block Tuner 정의
        # self.mid_tuner = Tuner(
        #     block_out_channels[-1], 
        #     block_out_channels[-1],
        #     hidden_dim=tuner_hidden_dim
        # )

        self.localization_skip_params = nn.ParameterList()
        
        current_height = 64
        current_width = 64
        
        for i, ch in enumerate(block_out_channels): # 4 레벨
            # (1, C, H, W) trainable watermarks
            param = nn.Parameter(torch.zeros(1, ch, current_height, current_width))
            self.localization_skip_params.append(param)

            if i < len(block_out_channels) - 1:
                current_height //= 2
                current_width //= 2
                
        mid_ch = block_out_channels[-1]
        self.localization_mid_param = nn.Parameter(
            torch.zeros(1, mid_ch, current_height, current_width)
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,        
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True
        
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float): dtype = torch.float32 if is_mps else torch.float64
            else: dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        
        aug_emb = None
        if self.class_embedding is not None:
            if class_labels is None: raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)
            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
            if self.config.class_embeddings_concat: emb = torch.cat([emb, class_emb], dim=-1)
            else: emb = emb + class_emb
        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_time":
            pass 

        emb = emb + aug_emb if aug_emb is not None else emb
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)
        
        ####### Modification starts
        assert len(self.localization_skip_params) == len(self.down_blocks) # watermark for each level
        down_block_additional_residuals = list(self.localization_skip_params)
        mid_block_additional_residual = self.localization_mid_param

        is_controlnet = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # (unet.py의 3. down 후반부 로직 시작 - 잔차 주입)
        if is_controlnet:
            new_down_block_res_samples = ()
            
            # 1. conv_in
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_samples[0],)
            
            # 2. 12개의 스킵 피처를 순회하며 워터마크 더하기
            skip_index = 0 # down_block_additional_residuals (워터마크) 인덱스
            level_feature_count = 0 # 현재 레벨에서 처리한 스킵 피처 수
            
            # down_block_res_samples[1:] -> 12개 스킵 피처
            for i in range(len(down_block_res_samples) - 1):
                res_sample = down_block_res_samples[i + 1] # 원본 스킵 피처
                
                # 현재 스킵 피처에 맞는 워터마크 파라미터 가져오기
                watermark_param = down_block_additional_residuals[skip_index]
                
                # skip: same resolution * 2 and down-sample * 1
                if res_sample.shape[-2:] != watermark_param.shape[-2:]:
                    watermark_resized = F.interpolate(watermark_param, size=res_sample.shape[-2:], mode='nearest')
                else:
                    watermark_resized = watermark_param
                    
                modified_res_sample = res_sample + watermark_resized
                new_down_block_res_samples = new_down_block_res_samples + (modified_res_sample,)
                
                # 다음 레벨 워터마크로 넘어갈지 결정
                level_feature_count += 1
                if level_feature_count == (self.layers_per_block + 1):
                    if skip_index < len(down_block_additional_residuals) - 1:
                         skip_index += 1
                    level_feature_count = 0 # 카운터 리셋

            down_block_res_samples = new_down_block_res_samples
            
            # (Mid block 잔차 더하기는 기존과 동일)
            # sample = sample + mid_block_additional_residual

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        
        if is_controlnet:
            if sample.shape[-2:] != mid_block_additional_residual.shape[-2:]:
                mid_residual_resized = F.interpolate(mid_block_additional_residual, size=sample.shape[-2:], mode='nearest')
            else:
                mid_residual_resized = mid_block_additional_residual
            sample = sample + mid_residual_resized

        # 5. up (unet.py와 동일 - 편집된 스킵 피처가 `down_block_res_samples`에 들어있음)
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process (unet.py와 동일)
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
    
def get_aug_embed(self, emb, added_cond_kwargs, B):
        if self.config.addition_embed_type == "text":
            return self.add_embedding(added_cond_kwargs.get("text_embeds").repeat_interleave(B // added_cond_kwargs.get("text_embeds").shape[0], dim=0))
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            text_embeds = added_cond_kwargs.get("text_embeds")
            image_embeds = added_cond_kwargs.get("image_embeds")
            text_embeds = text_embeds.repeat_interleave(B // text_embeds.shape[0], dim=0)
            image_embeds = image_embeds.repeat_interleave(B // image_embeds.shape[0], dim=0)
            return self.add_embedding(text_embeds, image_embeds)
        elif self.config.addition_embed_type == "text_time":
             # SDXL - style
            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")
            text_embeds = text_embeds.repeat_interleave(B // text_embeds.shape[0], dim=0)
            time_ids = time_ids.repeat_interleave(B // time_ids.shape[0], dim=0)
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            return self.add_embedding(add_embeds)
        # Add other addition_embed_type conditions if needed
        return None

def handle_encoder_hid_proj(self, encoder_hidden_states, added_cond_kwargs, B):
    if self.config.encoder_hid_dim_type == "text_proj":
        return self.encoder_hid_proj(encoder_hidden_states.repeat_interleave(B // encoder_hidden_states.shape[0], dim=0))
    elif self.config.encoder_hid_dim_type == "text_image_proj":
        # Kadinsky 2.1 - style
        image_embeds = added_cond_kwargs.get("image_embeds")
        image_embeds = image_embeds.repeat_interleave(B // image_embeds.shape[0], dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(B // encoder_hidden_states.shape[0], dim=0)
        return self.encoder_hid_proj(encoder_hidden_states, image_embeds)
    # Add other encoder_hid_dim_type conditions if needed
    return encoder_hidden_states.repeat_interleave(B // encoder_hidden_states.shape[0], dim=0)

if __name__ == "__main__":
    """
    이 파일을 직접 실행할 때, Embedder (LocalizationUNet)를 초기화하고
    사전 학습된 가중치를 복사하며, 더미 데이터로 forward pass를 테스트합니다.
    """
    from diffusers import UNet2DConditionModel, AutoencoderKL # VAE 로드 추가

    # 0. 설정
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_h = 64 # 512x512 이미지 기준
    latent_w = 64

    print(f"Loading original U-Net '{pretrained_model_name_or_path}' to CPU...")
    original_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="unet",
        cache_dir="/mnt/nas5/suhyeon/caches"
    ).to('cpu')
    
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="vae",
        cache_dir="/mnt/nas5/suhyeon/caches"
    ).to(device)
    vae.eval()
    # --------------------------------------------------------

    # 2. 커스텀 Embedder(LocalizationUNet)를 GPU(VRAM)에 생성
    print(f"Initializing Embedder (LocalizationUNet) on {device}...")
    config = original_unet.config # 원본 U-Net 설정을 가져옴
    
    print(f"Initializing Embedder (LocalizationUNet) on {device} using SD 2.1 config...")

    init_kwargs = {
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "block_out_channels": config.block_out_channels,
        "layers_per_block": config.layers_per_block,
        "cross_attention_dim": config.cross_attention_dim,
        "attention_head_dim": config.attention_head_dim,
        "down_block_types": config.down_block_types,
        "up_block_types": config.up_block_types,
        "mid_block_type": config.mid_block_type,

        # .get()으로 선택적 인수 추가 (기본값 설정)
        "only_cross_attention": config.get("only_cross_attention", False),
        "dual_cross_attention": config.get("dual_cross_attention", False),
        "use_linear_projection": config.get("use_linear_projection", False),
        "num_attention_heads": config.get("num_attention_heads"), # None일 수 있음
        "transformer_layers_per_block": config.get("transformer_layers_per_block", 1),

        "norm_num_groups": config.get("norm_num_groups", 32),
        "norm_eps": config.get("norm_eps", 1e-5),
        "upcast_attention": config.get("upcast_attention", False),
        "time_embedding_type": config.get("time_embedding_type", "positional"),
        "time_embedding_dim": config.get("time_embedding_dim"),

        "projection_class_embeddings_input_dim": config.get("projection_class_embeddings_input_dim"),
        "class_embed_type": config.get("class_embed_type"),
        "addition_embed_type": config.get("addition_embed_type"),
        "addition_time_embed_dim": config.get("addition_time_embed_dim"),

        "dropout": config.get("dropout", 0.0),
        "resnet_time_scale_shift": config.get("resnet_time_scale_shift", "default"),
        
        # Embedder 고유 인수
        "latent_height": latent_h,
        "latent_width": latent_w,
    }

    embedder_model = Embedder(**init_kwargs).to(device)

    # 3. 'for' 루프를 사용해 가중치 복사 (CPU에서 작업)
    print("Copying pre-trained weights from CPU to GPU model...")
    original_weights = original_unet.state_dict()
    custom_state_dict = embedder_model.state_dict()

    copied_keys_count = 0
    trainable_keys_count = 0
    trainable_params_list = []

    for name, param in custom_state_dict.items():
        if name in original_weights:
            # 원본 U-Net에 있는 가중치 복사
            try:
                param.copy_(original_weights[name].detach().clone())
                param.requires_grad_(False) # 동결
                copied_keys_count += 1
            except RuntimeError as e:
                print(f"  Error copying {name}: {e}") # Shape 불일치 등
        else:
            # 새로 추가된 가중치 (localization_*)
            param.requires_grad_(True) # 학습 대상
            trainable_keys_count += 1
            trainable_params_list.append(name)
            # print(f"  Trainable Param Map: {name} (Shape: {param.shape})") # 너무 길어서 주석 처리

    print(f"--- Initialization Complete ---")
    print(f"Copied {copied_keys_count} matching U-Net layers (now frozen).")
    print(f"Found {trainable_keys_count} new trainable localization maps.")
    if trainable_keys_count > 0:
         print(f"  Example trainable param: {trainable_params_list[0]}")
    
    # 4. CPU에 있던 원본 U-Net은 메모리에서 삭제
    del original_unet
    del original_weights
    
    print(f"Embedder (LocalizationUNet) is ready on {device}.")
    embedder_model.eval() # 테스트를 위해 eval 모드로 설정

    # 5. 테스트 실행 (가짜 입력 데이터)
    try:
        B = 2 # 배치 사이즈
        
        # 실제 사용 시 Scheduler에서 샘플링된 노이즈 사용
        noise_latent = torch.randn(B, config.in_channels, latent_h, latent_w).to(device)
        timestep = torch.tensor([999] * B).to(device) # 높은 timestep 예시
        
        # 실제 사용 시 CLIP 등 텍스트 인코더 출력 사용
        text_embeds = torch.randn(B, 77, config.cross_attention_dim).to(device) 
        
        print("\nTesting forward pass...")
        with torch.no_grad(): # 테스트이므로 그래디언트 계산 비활성화
            # Embedder 모델 실행
            output = embedder_model(
                sample=noise_latent,
                timestep=timestep,
                encoder_hidden_states=text_embeds,
                # (별도 조건 입력이 필요 없음)
            )
            predicted_noise = output.sample
            
        print(f"Forward pass successful. Predicted noise shape: {predicted_noise.shape}")
        assert predicted_noise.shape == noise_latent.shape # 입력과 출력 shape 확인

        # --- [추가] VAE 디코딩 테스트 ---
        # (실제 이미지 생성을 모방하여 VAE 호환성 확인)
        print("\nTesting VAE decoding...")
        # predicted_noise를 사용하여 latent 업데이트 (간단한 예시)
        # 실제로는 scheduler.step() 사용
        denoised_latent = noise_latent - predicted_noise # 매우 단순화된 예측
        
        # VAE 디코더로 이미지 복원
        denoised_latent = denoised_latent / vae.config.scaling_factor
        with torch.no_grad():
             image = vae.decode(denoised_latent).sample
             
        print(f"VAE decoding successful. Output image shape: {image.shape}")
        # 예상 shape: (B, 3, latent_h*8, latent_w*8)
        assert image.shape == (B, 3, latent_h * 8, latent_w * 8)

    except Exception as e:
        print(f"\nForward pass or VAE test failed. Error: {e}")
        import traceback
        traceback.print_exc()