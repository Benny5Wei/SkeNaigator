#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np


class LinearNormalizer(nn.Module):
    """线性归一化器"""
    
    def __init__(self):
        super().__init__()
        self.normalizers = nn.ModuleDict()
    
    def add_normalizer(self, key: str, normalizer: 'Normalizer'):
        """添加归一化器"""
        self.normalizers[key] = normalizer
    
    def normalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """归一化数据"""
        normalized = {}
        for key, value in data.items():
            if key in self.normalizers:
                normalized[key] = self.normalizers[key].normalize(value)
            else:
                normalized[key] = value
        return normalized
    
    def unnormalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """反归一化数据"""
        unnormalized = {}
        for key, value in data.items():
            if key in self.normalizers:
                unnormalized[key] = self.normalizers[key].unnormalize(value)
            else:
                unnormalized[key] = value
        return unnormalized
    
    def __getitem__(self, key: str) -> 'Normalizer':
        """获取归一化器"""
        return self.normalizers[key]
    
    def __setitem__(self, key: str, normalizer: 'Normalizer'):
        """设置归一化器"""
        self.normalizers[key] = normalizer


class Normalizer(nn.Module):
    """基础归一化器"""
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化"""
        return (x - self.mean) / (self.std + 1e-8)
    
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化"""
        return x * self.std + self.mean
    
    @classmethod
    def from_data(cls, data: torch.Tensor) -> 'Normalizer':
        """从数据创建归一化器"""
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        return cls(mean, std)
    
    @classmethod
    def from_stats(cls, mean: np.ndarray, std: np.ndarray) -> 'Normalizer':
        """从统计量创建归一化器"""
        mean_tensor = torch.from_numpy(mean).float()
        std_tensor = torch.from_numpy(std).float()
        return cls(mean_tensor, std_tensor)


