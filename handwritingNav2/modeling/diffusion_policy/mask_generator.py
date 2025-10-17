#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Tuple


class LowdimMaskGenerator(nn.Module):
    """低维掩码生成器"""
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int,
        fix_obs_steps: bool = True,
        action_visible: bool = False
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible
    
    def forward(self, shape: Tuple[int, ...], device=None) -> torch.Tensor:
        """
        生成掩码
        
        Args:
            shape: 输入张量形状 (B, T, D)
            device: 目标设备，如果为None则尝试从参数推断
            
        Returns:
            mask: 布尔掩码，True表示需要条件化的位置
        """
        B, T, D = shape
        # 安全地获取设备
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        
        # 创建掩码
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        
        if self.fix_obs_steps:
            # 固定观察步数
            n_obs_steps = self.max_n_obs_steps
        else:
            # 随机观察步数
            n_obs_steps = torch.randint(
                1, self.max_n_obs_steps + 1, (B,), device=device
            )
        
        # 设置观察掩码
        if self.obs_dim > 0:
            for i in range(B):
                if self.fix_obs_steps:
                    obs_steps = n_obs_steps
                else:
                    obs_steps = n_obs_steps[i].item()
                
                # 观察部分需要条件化
                mask[i, :obs_steps, self.action_dim:] = True
        
        # 设置动作掩码
        if self.action_visible:
            # 动作可见，不需要条件化
            pass
        else:
            # 动作不可见，需要条件化
            mask[:, :, :self.action_dim] = True
        
        return mask


