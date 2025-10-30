import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

# --- diffusers에서 필요한 모듈 임포트 ---
# (unet.py가 있는 폴더에서 실행하거나, unet.py를 PYTHONPATH에 추가해야 함)
try:
    from .unet import UNet
except ImportError:
    from unet import UNet

from diffusers.models.unet_2d_condition import UNet2DConditionOutput, logger
from diffusers.models.unet_2d_blocks import (
    get_down_block, 
    CrossAttnDownBlock2D, 
    DownBlock2D
)
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

#
# --- 1. 헬퍼 모듈 (Helper Modules) ---
#

class ConditionEncoder(nn.Module):
    """
    (c) Dense Conv  (조건 인코더)
    조건 이미지를 U-Net의 각 해상도 레벨에 맞는
    피처 맵 리스트로 인코딩합니다. (총 4개 레벨)
    """
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # U-Net의 DownEncoderBlock과 동일한 구조 사용
            block = get_down_block(
                "DownEncoderBlock2D",
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                # (SD 1.5 U-Net은 CrossAttn이 없으므로 DownEncoderBlock2D 사용)
            )
            self.down_blocks.append(block)

    def forward(self, condition_image: torch.FloatTensor) -> List[torch.FloatTensor]:
        """
        조건 이미지를 4개 레벨의 피처로 인코딩합니다.
        (예: 320, 640, 1280, 1280 채널)
        """
        features = []
        h = self.conv_in(condition_image)
        for i, block in enumerate(self.down_blocks):
            h = block(h, temb=None)
            if i < len(self.down_blocks):
                 features.append(h) # 각 레벨의 다운샘플링 *전* 피처 저장
        
        # (총 4개의 피처맵 반환)
        return features

class SCETuner(nn.Module):
    """
    (a) SC-Tuner  / (b) CSC-Tuner [cite: 909-911]
    논문의 Eq. (5) [cite: 896]와 Eq. (6) [cite: 902-903] (Adapter OP)을 구현합니다.
    Tuner(x, c) + x
    """
    def __init__(self, skip_channels: int, condition_channels: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            # hidden_dim을 더 작게 설정하여 파라미터 절약 가능
            hidden_dim = skip_channels
        
        # W_down: 입력을 hidden_dim으로 (1x1 conv)
        # (논문에서는 x+c를 입력받지만, x와 c를 concat하는 것이 더 일반적)
        self.W_down = nn.Conv2d(skip_channels + condition_channels, hidden_dim, kernel_size=1)
        # phi (Activation)
        self.activation = nn.GELU() # [cite: 902-903]
        # W_up: hidden_dim을 출력으로 (1x1 conv)
        self.W_up = nn.Conv2d(hidden_dim, skip_channels, kernel_size=1)
        
        # ControlNet의 zero-convolution처럼 W_up을 0으로 초기화
        nn.init.zeros_(self.W_up.weight)
        nn.init.zeros_(self.W_up.bias)

    def forward(self, skip_feat: torch.FloatTensor, cond_feat: torch.FloatTensor) -> torch.FloatTensor:
        """
        Eq. (5) [cite: 896]와 Eq. (6) [cite: 902-903]을 구현
        Output = Tuner(x, c) + x
        """
        # 1. 튜너의 입력 (x 와 c 를 결합)
        tuner_input = torch.cat([skip_feat, cond_feat], dim=1)
        
        # 2. T_j(x, c) = W_up(phi(W_down(input))) [cite: 902-903]
        residual = self.W_up(self.activation(self.W_down(tuner_input)))
        
        # 3. O_j = T_j(x, c) + x (Eq. 5의 잔차 학습) [cite: 896]
        return skip_feat + residual

#
# --- 2. 메인 SCEditUnet 클래스 ---
#

class SCEditUnet(UNet):
    """
    unet.py의 UNet을 상속받아 SCEdit 로직을 추가한 클래스.
    """
    def __init__(
        self,
        # --- UNet2DConditionModel의 기본 파라미터 (SD 1.5 기준) ---
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        cross_attention_dim: int = 768,
        # --- SCEdit에 필요한 추가 파라미터 ---
        condition_image_channels: int = 3,
        tuner_hidden_dim: Optional[int] = None, # 튜너의 hidden dim
        **kwargs,
    ):
        # 1. 부모 클래스 (UNet) 초기화
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            **kwargs,
        )
        
        self.layers_per_block = layers_per_block

        # 2. (c) Dense Conv (조건 인코더) 정의 
        self.condition_encoder = ConditionEncoder(
            in_channels=condition_image_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block
        )
        
        # 3. (a) SC-Tuners 정의 
        #    U-Net의 스킵 커넥션 개수(12개)만큼 생성합니다.
        self.tuners = nn.ModuleList()
        
        # (U-Net 1.5 스킵 채널: 3x320, 3x640, 3x1280, 3x1280)
        skip_channels_list = []
        for i in range(len(block_out_channels)):
            for _ in range(layers_per_block + 1): # (2 ResNet + 1 Downsampler)
                skip_channels_list.append(block_out_channels[i])
        
        # (SD 1.5는 conv_in 스킵 피처를 사용하지 않고, 
        #  down_blocks에서 4x3=12개의 스킵 피처를 생성합니다)
        if len(skip_channels_list) != 12:
            # (Hugging Face diffusers의 UNet 1.5 로직 기준)
            skip_channels_list = []
            for ch in block_out_channels:
                for _ in range(layers_per_block + 1): # 3
                    skip_channels_list.append(ch)
        
        # 조건 인코더의 출력 채널 (4개 레벨)
        cond_channels_list = []
        for ch in block_out_channels:
             for _ in range(layers_per_block + 1):
                cond_channels_list.append(ch)

        # 12개의 튜너 생성
        self.tuners = nn.ModuleList()
        for skip_ch, cond_ch in zip(skip_channels_list, cond_channels_list):
            self.tuners.append(SCETuner(skip_ch, cond_ch, hidden_dim=tuner_hidden_dim))
            
        # 4. Mid Block Tuner 정의
        self.mid_tuner = SCETuner(
            block_out_channels[-1], 
            block_out_channels[-1],
            hidden_dim=tuner_hidden_dim
        )

    # 
    # === 3. Forward 메서드 (unet.py에서 복사 후 수정) ===
    #
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        
        # --- [수정] SCEdit을 위한 새 인수 추가 ---
        condition_image: torch.FloatTensor,
        control_weight: float = 1.0, # 제어 강도 (alpha) [cite: 909-911]
        
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        
        # --- [수정] 이 인수들은 내부에서 생성되므로 제거 ---
        # down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        # mid_block_additional_residual: Optional[torch.Tensor] = None,
        
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        
        # (unet.py의 0, 1, 2 섹션 코드를 그대로 복사)
        # 0. 기본 설정
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
        
        # (aug_emb, class_emb 등 나머지 임베딩 로직 ... unet.py와 동일)
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
             # (SDXL 로직 ... 생략)
            pass 
        # (aug_emb, class_emb 등 나머지 임베딩 로직 끝)

        emb = emb + aug_emb if aug_emb is not None else emb
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)
        
        
        # === [수정] SCEdit 로직 1: 조건 인코딩 ===
        # (c) Dense Conv  실행
        # condition_feats: [level 0 feat, level 1 feat, level 2 feat, level 3 feat]
        condition_feats = self.condition_encoder(condition_image)
        # ======================================


        # 3. down (unet.py와 동일 + SCEdit 로직 주입)
        down_block_res_samples = (sample,)
        
        # --- [수정] SCEdit 잔차(Residuals)를 저장할 리스트 ---
        down_block_additional_residuals = []
        tuner_idx = 0
        cond_level_idx = 0
        # ------------------------------------------------
        
        for downsample_block in self.down_blocks:
            
            # --- [수정] 현재 레벨의 조건 피처 가져오기 ---
            current_cond_feat = condition_feats[cond_level_idx]
            # ----------------------------------------

            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # (CrossAttnDownBlock2D의 경우)
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                # (DownBlock2D의 경우)
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            # === [수정] SCEdit 로직 2: 스킵 피처 편집 (튜닝) ===
            edited_res_samples = []
            for res_sample in res_samples:
                tuner = self.tuners[tuner_idx]
                
                # 조건 피처의 해상도를 스킵 피처와 맞춤
                cond_feat_resized = F.interpolate(current_cond_feat, size=res_sample.shape[-2:], mode="nearest")
                
                # 튜너 실행: O_j = T(x, c) + x [cite: 896]
                edited_sample = tuner(res_sample, cond_feat_resized)
                
                # ControlNet 로직을 위해 '잔차(residual)'만 계산: T(x, c)
                residual = (edited_sample - res_sample) * control_weight
                
                down_block_additional_residuals.append(residual)
                edited_res_samples.append(edited_sample) # (튜닝된 피처)
                
                tuner_idx += 1
            
            # (ControlNet과 달리, SCEdit은 원본 스킵 피처가 아닌
            #  '편집된' 스킵 피처를 다음 업샘플링에 전달해야 함)
            # down_block_res_samples += tuple(edited_res_samples) # -> 이 방식 대신 ControlNet 방식 사용
            down_block_res_samples += res_samples # 원본 U-Net과 동일하게 원본 스킵 저장
            # ----------------------------------------------------
            
            cond_level_idx += 1
            
        # --- [수정] ControlNet 플래그 활성화 ---
        is_controlnet = True 
        
        # (unet.py의 3. down 후반부 로직 시작 - 잔차 주입)
        if is_controlnet:
            new_down_block_res_samples = ()
            
            # `down_block_res_samples` (원본 스킵)와 
            # `down_block_additional_residuals` (튜너 잔차)를 더함
            # (conv_in 스킵은 튜닝하지 않았으므로 1부터 시작)
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_samples[0],)
            
            for i in range(len(down_block_additional_residuals)):
                down_block_res_sample = down_block_res_samples[i + 1]
                down_block_additional_residual = down_block_additional_residuals[i]
                
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
        # (unet.py의 3. down 후반부 로직 끝)


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
        
        # === [수정] Mid Block 튜닝 ===
        cond_feat_mid = F.interpolate(condition_feats[-1], size=sample.shape[-2:], mode="nearest")
        edited_mid_sample = self.mid_tuner(sample, cond_feat_mid)
        mid_block_additional_residual = (edited_mid_sample - sample) * control_weight
        # =============================
        
        if is_controlnet:
            sample = sample + mid_block_additional_residual

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


#
# --- 4. 메인 실행 (초기화 예제) ---
#
if __name__ == "__main__":
    """
    이 파일을 직접 실행할 때, SCEditUnet을 초기화하고
    사전 학습된 가중치를 복사하는 예제를 보여줍니다.
    """
    from diffusers import UNet2DConditionModel

    # 0. 설정
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 원본 U-Net을 CPU에 로드 (VRAM 절약)
    print(f"Loading original U-Net '{pretrained_model_name_or_path}' to CPU...")
    original_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="unet"
    ).to('cpu')
    
    # 2. 커스텀 SCEditUnet을 GPU(VRAM)에 생성
    #    (이 모델의 모든 가중치는 현재 무작위 상태입니다)
    print(f"Initializing SCEditUnet on {device}...")
    
    # 원본 U-Net의 설정값을 그대로 사용
    config = original_unet.config
    scedit_unet = SCEditUnet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        cross_attention_dim=config.cross_attention_dim,
        # ... (필요한 다른 config 값들) ...
        # (간단한 예시를 위해 주요 파라미터만 전달)
    ).to(device)

    # 3. 'for' 루프를 사용해 가중치 복사 (CPU에서 작업)
    print("Copying pre-trained weights from CPU to GPU model...")
    original_weights = original_unet.state_dict()
    custom_state_dict = scedit_unet.state_dict()

    copied_keys = 0
    trainable_keys = 0

    for name, param in custom_state_dict.items():
        if name in original_weights:
            # 원본 U-Net에 있는 가중치 복사
            param.copy_(original_weights[name].detach().clone())
            param.requires_grad_(False) # 동결
            copied_keys += 1
        else:
            # 새로 추가된 가중치 (condition_encoder, tuners)
            param.requires_grad_(True) # 학습 대상
            trainable_keys += 1
            print(f"  Trainable key: {name}")

    print(f"--- Initialization Complete ---")
    print(f"Copied {copied_keys} matching U-Net layers (now frozen).")
    print(f"Found {trainable_keys} new trainable layers (in tuners/encoder).")
    
    # 4. CPU에 있던 원본 U-Net은 메모리에서 삭제
    del original_unet
    del original_weights
    
    print(f"SCEditUnet is ready on {device}.")

    # 5. 테스트 실행 (가짜 입력 데이터)
    try:
        B = 1
        H_latent = 64
        W_latent = 64
        H_cond = 512
        W_cond = 512
        
        noise_latent = torch.randn(B, 4, H_latent, W_latent).to(device)
        timestep = torch.tensor(100).to(device)
        text_embeds = torch.randn(B, 77, 768).to(device)
        condition_image = torch.randn(B, 3, H_cond, W_cond).to(device)
        
        print("\nTesting forward pass...")
        output = scedit_unet(
            sample=noise_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            condition_image=condition_image
        )
        print(f"Forward pass successful. Output shape: {output.sample.shape}")

    except Exception as e:
        print(f"\nForward pass failed. This might be due to simplified __init__ params.")
        print(f"Error: {e}")
        print("Please ensure all config parameters from UNet2DConditionModel are passed to SCEditUnet.")