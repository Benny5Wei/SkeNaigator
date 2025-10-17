#!/usr/bin/env python3
"""
扩散策略数据集加载器

从离线专家演示数据中加载序列数据用于扩散策略训练
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import cv2
import logging

logger = logging.getLogger(__name__)


class DiffusionNavigationDataset(Dataset):
    """扩散策略导航数据集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        horizon: int = 16,
        n_obs_steps: int = 3,
        n_action_steps: int = 4,
        use_rgb: bool = True,
        use_depth: bool = True,
        rgb_size: Tuple[int, int] = (256, 256),
        depth_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train' 或 'test'
            horizon: 序列长度
            n_obs_steps: 观察步数
            n_action_steps: 动作步数
            use_rgb: 是否使用RGB
            use_depth: 是否使用Depth
            rgb_size: RGB图像大小
            depth_size: Depth图像大小
        """
        self.data_dir = data_dir
        self.split = split
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.rgb_size = rgb_size
        self.depth_size = depth_size
        
        # 收集所有episode路径
        self.episodes = []
        split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(split_dir):
            logger.warning(f"数据集目录不存在: {split_dir}")
            return
        
        # 遍历所有场景
        for scene_name in os.listdir(split_dir):
            scene_dir = os.path.join(split_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            
            # 遍历该场景的所有episode
            for episode_name in os.listdir(scene_dir):
                episode_dir = os.path.join(scene_dir, episode_name)
                if not os.path.isdir(episode_dir) or not episode_name.startswith('episode_'):
                    continue
                
                # 检查必要的文件是否存在
                actions_path = os.path.join(episode_dir, 'actions.npy')
                if os.path.exists(actions_path):
                    self.episodes.append(episode_dir)
        
        logger.info(f"加载 {split} 集: 找到 {len(self.episodes)} 个episodes")
        
        # 预处理：为每个episode创建可用的序列索引
        self.valid_sequences = []
        for ep_idx, episode_dir in enumerate(self.episodes):
            try:
                actions = np.load(os.path.join(episode_dir, 'actions.npy'))
                traj_length = len(actions)
                
                # 计算该episode可以生成多少个有效序列
                # 每个序列需要 n_obs_steps 个观察 + n_action_steps 个动作
                min_length = max(n_obs_steps, n_action_steps)
                
                if traj_length >= min_length:
                    # 可以从这个episode中提取多个序列
                    for start_idx in range(traj_length - min_length + 1):
                        self.valid_sequences.append((ep_idx, start_idx))
            except Exception as e:
                logger.warning(f"处理episode {episode_dir} 时出错: {e}")
        
        logger.info(f"总共生成 {len(self.valid_sequences)} 个有效序列")
    
    def __len__(self) -> int:
        return len(self.valid_sequences)
    
    def _load_rgb(self, episode_dir: str, frame_idx: int) -> np.ndarray:
        """加载RGB图像，返回 [H, W, 3]"""
        rgb_path = os.path.join(episode_dir, f"rgb_{frame_idx:05d}.png")
        if not os.path.exists(rgb_path):
            # 返回黑色图像作为fallback
            return np.zeros((*self.rgb_size, 3), dtype=np.float32)
        
        img = cv2.imread(rgb_path)
        if img is None:
            return np.zeros((*self.rgb_size, 3), dtype=np.float32)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.rgb_size)
        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 确保是 [H, W, 3]
        if len(img.shape) != 3 or img.shape[2] != 3:
            logger.warning(f"RGB图像维度异常: {img.shape}, 路径: {rgb_path}")
            return np.zeros((*self.rgb_size, 3), dtype=np.float32)
        
        return img
    
    def _load_depth(self, episode_dir: str, frame_idx: int) -> np.ndarray:
        """加载Depth图像，返回 [H, W]"""
        depth_path = os.path.join(episode_dir, f"depth_{frame_idx:05d}.npy")
        if not os.path.exists(depth_path):
            # 返回零深度作为fallback
            return np.zeros(self.depth_size, dtype=np.float32)
        
        depth = np.load(depth_path)
        original_shape = depth.shape
        
        # 确保深度图是2D的 - 移除所有大小为1的维度
        depth = np.squeeze(depth)
        
        # 如果squeeze后变成了0维或1维，说明数据有问题
        if len(depth.shape) < 2:
            logger.warning(f"深度图维度异常: 原始{original_shape} -> squeeze后{depth.shape}, 路径: {depth_path}")
            return np.zeros(self.depth_size, dtype=np.float32)
        
        # 如果squeeze后还是超过2维，只取前两维
        if len(depth.shape) > 2:
            logger.warning(f"深度图有多余维度: {depth.shape}, 只使用前两维, 路径: {depth_path}")
            depth = depth[..., 0]  # 只取第一个通道
        
        # 调整大小
        if depth.shape[:2] != self.depth_size:
            depth = cv2.resize(depth, self.depth_size)
        
        # 再次确保是2D数组
        assert len(depth.shape) == 2, f"深度图必须是2D的，当前: {depth.shape}"
        
        # 归一化 (假设深度范围 0-10m)
        depth = np.clip(depth, 0, 10.0) / 10.0
        return depth.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个序列样本"""
        ep_idx, start_idx = self.valid_sequences[idx]
        episode_dir = self.episodes[ep_idx]
        
        # 加载动作
        actions = np.load(os.path.join(episode_dir, 'actions.npy'))
        
        # 提取观察序列 (n_obs_steps 帧)
        obs_list = []
        for i in range(self.n_obs_steps):
            frame_idx = min(start_idx + i, len(actions) - 1)
            
            obs_features = []
            
            # 加载RGB（如果启用）
            if self.use_rgb:
                rgb = self._load_rgb(episode_dir, frame_idx)
                obs_features.append(rgb)
            
            # 加载Depth（优先使用）
            if self.use_depth:
                depth = self._load_depth(episode_dir, frame_idx)
                # 扩展维度以匹配RGB格式
                depth = np.expand_dims(depth, axis=-1)
                obs_features.append(depth)
            
            # 拼接所有特征
            if len(obs_features) > 0:
                obs = np.concatenate(obs_features, axis=-1)
            else:
                # 如果都没有，使用零填充
                obs = np.zeros((self.depth_size[0], self.depth_size[1], 1), dtype=np.float32)
            
            obs_list.append(obs)
        
        # 堆叠观察 [n_obs_steps, H, W, C]
        obs_seq = np.stack(obs_list, axis=0)
        
        # 调试：检查维度
        if len(obs_seq.shape) != 4:
            logger.error(f"观察序列维度错误! 期望4维[n_obs_steps, H, W, C]，实际: {obs_seq.shape}")
            logger.error(f"obs_list长度: {len(obs_list)}, 第一个obs形状: {obs_list[0].shape if obs_list else 'empty'}")
            # 尝试修复：如果有额外维度，进行squeeze
            while len(obs_seq.shape) > 4:
                obs_seq = obs_seq.squeeze()
            # 如果还是不对，返回一个合理的默认值
            if len(obs_seq.shape) != 4:
                logger.error(f"无法修复维度，使用默认零填充")
                obs_channels = 1 if self.use_depth and not self.use_rgb else 3
                if self.use_rgb and self.use_depth:
                    obs_channels = 4
                obs_seq = np.zeros((self.n_obs_steps, *self.depth_size, obs_channels), dtype=np.float32)
        
        # 提取动作序列 (n_action_steps)
        action_indices = []
        for i in range(self.n_action_steps):
            action_idx = min(start_idx + i, len(actions) - 1)
            action_indices.append(action_idx)
        
        action_seq = actions[action_indices]
        
        # 转换为torch张量
        # 观察: [n_obs_steps, C, H, W] (CHW格式)
        obs_seq = torch.from_numpy(obs_seq).permute(0, 3, 1, 2).float()
        # 动作: [n_action_steps]
        action_seq = torch.from_numpy(action_seq).long()
        
        return {
            'obs': obs_seq,
            'action': action_seq,
        }


def create_diffusion_dataloader(
    data_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    horizon: int = 16,
    n_obs_steps: int = 3,
    n_action_steps: int = 4,
    distributed: bool = False,
    **kwargs
) -> DataLoader:
    """创建数据加载器的便捷函数（支持DDP）"""
    from torch.utils.data.distributed import DistributedSampler
    
    dataset = DiffusionNavigationDataset(
        data_dir=data_dir,
        split=split,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        **kwargs
    )
    
    # 如果是分布式训练，使用DistributedSampler
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle
        )
        shuffle = False  # 有sampler时不能设置shuffle=True
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if split == 'train' else False,
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    logging.basicConfig(level=logging.INFO)
    
    data_dir = "/mnt_data/skenav2/handwritingNav2/data/diffusion_dataset"
    
    # 创建训练数据加载器
    train_loader = create_diffusion_dataloader(
        data_dir=data_dir,
        split='train',
        batch_size=4,
        num_workers=0,  # 用于调试
        shuffle=True,
    )
    
    logger.info(f"数据加载器创建成功，batch数: {len(train_loader)}")
    
    # 测试加载一个batch
    for batch in train_loader:
        logger.info(f"观察形状: {batch['obs'].shape}")  # [B, n_obs_steps, C, H, W]
        logger.info(f"动作形状: {batch['action'].shape}")  # [B, n_action_steps]
        logger.info(f"观察范围: [{batch['obs'].min():.3f}, {batch['obs'].max():.3f}]")
        logger.info(f"动作值: {batch['action'][0]}")
        break
    
    logger.info("数据加载器测试通过！")

