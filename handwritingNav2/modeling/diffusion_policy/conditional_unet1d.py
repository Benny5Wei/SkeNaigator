#!/usr/bin/env python3
"""
条件UNet 1D网络

用于扩散策略的1D条件UNet网络，专门处理动作序列生成任务。
该网络基于UNet架构，支持时间步嵌入和全局条件，用于学习动作序列的分布。

主要特性:
- 1D卷积架构: 专门处理序列数据
- 时间步嵌入: 将扩散时间步编码为特征
- 全局条件: 支持观察条件输入
- 跳跃连接: 保持细节信息
- 组归一化: 提高训练稳定性

作者: AI Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.block2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.gnorm1 = nn.GroupNorm(groups, in_channels)
        self.gnorm2 = nn.GroupNorm(groups, out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        h = self.gnorm1(x)
        h = F.relu(h)
        h = self.block1(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None]
        
        h = self.gnorm2(h)
        h = F.relu(h)
        h = self.block2(h)
        
        return h + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """条件UNet 1D网络"""
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        down_dims: list = [256, 512, 1024],
        diffusion_step_embed_dim: int = 256,
        down_step_sizes: list = [1, 1, 1],
        kernel_size: int = 5,
        n_groups: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.down_dims = down_dims
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim)
        )
        
        # 全局条件投影
        if global_cond_dim > 0:
            self.global_cond_mlp = nn.Sequential(
                nn.Linear(global_cond_dim, diffusion_step_embed_dim),
                nn.Mish(),
                nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim)
            )
        
        # 输入投影
        self.input_proj = nn.Conv1d(input_dim, down_dims[0], 1)
        
        # 下采样层
        self.down_layers = nn.ModuleList()
        in_channels = down_dims[0]
        
        for i, (out_channels, step_size) in enumerate(zip(down_dims, down_step_sizes)):
            self.down_layers.append(
                ResidualBlock(
                    in_channels, out_channels, 
                    diffusion_step_embed_dim, n_groups
                )
            )
            if step_size > 1:
                self.down_layers.append(
                    nn.Conv1d(out_channels, out_channels, step_size, stride=step_size)
                )
            in_channels = out_channels
        
        # 中间层
        self.mid_layers = nn.ModuleList([
            ResidualBlock(
                down_dims[-1], down_dims[-1], 
                diffusion_step_embed_dim, n_groups
            ),
            ResidualBlock(
                down_dims[-1], down_dims[-1], 
                diffusion_step_embed_dim, n_groups
            )
        ])
        
        # 上采样层
        self.up_layers = nn.ModuleList()
        in_channels = down_dims[-1]
        
        for i, (out_channels, step_size) in enumerate(zip(reversed(down_dims), reversed(down_step_sizes))):
            if step_size > 1:
                self.up_layers.append(
                    nn.ConvTranspose1d(
                        in_channels, in_channels, step_size, stride=step_size
                    )
                )
            self.up_layers.append(
                ResidualBlock(
                    in_channels + out_channels, out_channels,
                    diffusion_step_embed_dim, n_groups
                )
            )
            in_channels = out_channels
        
        # 输出投影
        self.output_proj = nn.Conv1d(down_dims[0], input_dim, 1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, T, D]
            timesteps: 时间步 [B]
            global_cond: 全局条件 [B, global_cond_dim]
        """
        # 转换维度: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        
        # 时间步嵌入
        time_emb = self.time_mlp(timesteps)
        
        # 全局条件嵌入
        if global_cond is not None and self.global_cond_dim > 0:
            global_emb = self.global_cond_mlp(global_cond)
            time_emb = time_emb + global_emb
        
        # 输入投影
        h = self.input_proj(x)
        
        # 存储跳跃连接
        skip_connections = []
        
        # 下采样
        for layer in self.down_layers:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
                # 保存ResidualBlock的输出作为skip connection
                skip_connections.append(h)
            else:
                # 下采样层（Conv1d）
                h = layer(h)
        
        # 中间层
        for layer in self.mid_layers:
            h = layer(h, time_emb)
        
        # 上采样
        for layer in self.up_layers:
            if isinstance(layer, ResidualBlock):
                skip_h = skip_connections.pop()
                h = torch.cat([h, skip_h], dim=1)
                h = layer(h, time_emb)
            else:
                h = layer(h)
        
        # 输出投影
        h = self.output_proj(h)
        
        # 转换回原始维度: [B, D, T] -> [B, T, D]
        h = h.transpose(1, 2)
        
        return h
