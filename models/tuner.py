import torch
import torch.nn as nn
from typing import List, Tuple

from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput, logger
from diffusers.models.unets.unet_2d_blocks import (
    get_down_block, 
    CrossAttnDownBlock2D, 
    DownBlock2D
)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Tuner(nn.Module):
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
        self.activation = nn.GELU()
        # W_up: hidden_dim을 출력으로 (1x1 conv)
        self.W_up = zero_module(nn.Conv2d(hidden_dim, skip_channels, kernel_size=1))

    def forward(self, skip_feat: torch.FloatTensor, cond_feat: torch.FloatTensor) -> torch.FloatTensor:
        """
        Eq. (5) [cite: 896]와 Eq. (6) [cite: 902-903]을 구현
        Output = Tuner(x, c) + x
        """
        tuner_input = torch.cat([skip_feat, cond_feat], dim=1)
        residual = self.W_up(self.activation(self.W_down(tuner_input)))
        return skip_feat + residual