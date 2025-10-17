#!/usr/bin/env python3

import os
import time
import logging
from collections import deque
from typing import Dict, List, Optional
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from habitat import Config, logger

from ..common.base_trainer import BaseRLTrainer
from ..common.tensorboard_utils import TensorboardWriter
from ..common.utils import linear_decay
from .diffusion_nav_policy import DiffusionNavPolicy
from .diffusion_dataset import create_diffusion_dataloader


class HabitatDiffusionTrainer(BaseRLTrainer):
    """使用离线数据集的扩散策略训练器（支持DataParallel多GPU训练）"""
    
    def __init__(self, config=None, local_rank=-1):
        super().__init__(config)
        self.policy = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.local_rank = local_rank
        self.is_distributed = False  # 不使用DDP
        
    def _setup_policy(self, config: Config) -> None:
        """设置扩散策略"""
        logger.add_filehandler(self.config.LOG_FILE)
        
        # 从配置中获取参数
        use_pointnav = self.config.USE_POINTNAV if hasattr(self.config, 'USE_POINTNAV') else False
        predict_goal = self.config.PREDICT_GOAL if hasattr(self.config, 'PREDICT_GOAL') else True
        
        # 构造观察空间（用于策略初始化）
        from gym import spaces
        
        # 根据配置确定观察维度
        obs_channels = 0
        if self.config.EXTRA_RGB:
            obs_channels += 3  # RGB
        if self.config.EXTRA_DEPTH:
            obs_channels += 1  # Depth
        
        # 创建简化的观察空间
        observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(256, 256, obs_channels), dtype=np.uint8)
        })
        
        # 创建动作空间
        action_space = spaces.Discrete(config.RL.DIFFUSION.action_dim)
        
        self.policy = DiffusionNavPolicy(
            observation_space=observation_space,
            action_space=action_space,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID if hasattr(self.config, 'TASK_CONFIG') else "pointgoal_with_gps_compass",
            hidden_size=config.RL.DIFFUSION.hidden_size,
            horizon=config.RL.DIFFUSION.horizon,
            n_action_steps=config.RL.DIFFUSION.n_action_steps,
            n_obs_steps=config.RL.DIFFUSION.n_obs_steps,
            obs_dim=config.RL.DIFFUSION.obs_dim,
            action_dim=config.RL.DIFFUSION.action_dim,
            num_inference_steps=config.RL.DIFFUSION.num_inference_steps,
            extra_rgb=self.config.EXTRA_RGB,
            extra_depth=self.config.EXTRA_DEPTH,
            slam=self.config.SLAM,
            use_vae=self.config.USE_VAE,
            use_pointnav=use_pointnav,
            predict_goal=predict_goal,
            obs_as_global_cond=config.RL.DIFFUSION.obs_as_global_cond
        )
        self.policy.to(self.device)
        
        # 初始化归一化器（使用恒等归一化，即不进行归一化）
        from .normalizer import Normalizer
        # 为obs创建归一化器（不做实际归一化，mean=0, std=1）
        obs_normalizer = Normalizer(
            mean=torch.zeros(1).to(self.device),
            std=torch.ones(1).to(self.device)
        )
        # 为action创建归一化器（不做实际归一化）
        action_normalizer = Normalizer(
            mean=torch.zeros(1).to(self.device),
            std=torch.ones(1).to(self.device)
        )
        self.policy.normalizer['obs'] = obs_normalizer
        self.policy.normalizer['action'] = action_normalizer
        
        # 多GPU训练包装（DataParallel）
        if torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行DataParallel训练")
            self.policy = nn.DataParallel(self.policy)
            self._is_data_parallel = True
        else:
            self._is_data_parallel = False
        
        # 设置优化器
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.RL.DIFFUSION.lr,
            weight_decay=config.RL.DIFFUSION.weight_decay
        )
        
        # 设置学习率调度器
        total_steps = config.NUM_UPDATES
        self.scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: linear_decay(x, total_steps)
        )
        
    def _setup_dataloaders(self) -> None:
        """设置数据加载器"""
        data_dir = self.config.DIFFUSION_DATA_DIR if hasattr(self.config, 'DIFFUSION_DATA_DIR') else \
                   "/mnt_data/skenav2/handwritingNav2/data/diffusion_dataset"
        
        logger.info(f"从 {data_dir} 加载数据集")
        
        # 创建训练数据加载器
        self.train_loader = create_diffusion_dataloader(
            data_dir=data_dir,
            split='train',
            batch_size=self.config.RL.DIFFUSION.batch_size if hasattr(self.config.RL.DIFFUSION, 'batch_size') else 32,
            num_workers=self.config.RL.DIFFUSION.num_workers if hasattr(self.config.RL.DIFFUSION, 'num_workers') else 4,
            shuffle=True,
            horizon=self.config.RL.DIFFUSION.horizon,
            n_obs_steps=self.config.RL.DIFFUSION.n_obs_steps,
            n_action_steps=self.config.RL.DIFFUSION.n_action_steps,
            use_rgb=self.config.EXTRA_RGB,
            use_depth=self.config.EXTRA_DEPTH,
            distributed=False,
        )
        
        # 创建验证数据加载器
        self.val_loader = create_diffusion_dataloader(
            data_dir=data_dir,
            split='test',
            batch_size=self.config.RL.DIFFUSION.batch_size if hasattr(self.config.RL.DIFFUSION, 'batch_size') else 32,
            num_workers=self.config.RL.DIFFUSION.num_workers if hasattr(self.config.RL.DIFFUSION, 'num_workers') else 2,
            shuffle=False,
            horizon=self.config.RL.DIFFUSION.horizon,
            n_obs_steps=self.config.RL.DIFFUSION.n_obs_steps,
            n_action_steps=self.config.RL.DIFFUSION.n_action_steps,
            use_rgb=self.config.EXTRA_RGB,
            use_depth=self.config.EXTRA_DEPTH,
            distributed=False,
        )
        
        logger.info(f"训练集: {len(self.train_loader)} batches")
        logger.info(f"验证集: {len(self.val_loader)} batches")
    
    def save_checkpoint(self, file_name: str) -> None:
        """保存检查点"""
        # 如果使用DataParallel，保存原始模型
        policy_to_save = self.policy.module if self._is_data_parallel else self.policy
        
        checkpoint = {
            "policy_state_dict": policy_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "update": self.update_count if hasattr(self, "update_count") else 0,
        }
        checkpoint_path = os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存检查点到: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, *args, **kwargs)
        return checkpoint
    
    def train(self, checkpoint_path: Optional[str] = None) -> None:
        """训练扩散策略 - 使用离线数据集"""
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        
        # 设置设备
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        logger.info(f"使用设备: {self.device}")
        
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        
        # 设置策略
        self._setup_policy(self.config)
        
        # 设置数据加载器
        self._setup_dataloaders()
        
        # 检查是否需要从现有检查点恢复训练
        start_update = 0
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.info(f"加载检查点: {checkpoint_path}")
            checkpoint = self.load_checkpoint(checkpoint_path, map_location=self.device)
            
            # 恢复模型参数
            if "policy_state_dict" in checkpoint:
                self.policy.load_state_dict(checkpoint["policy_state_dict"])
                logger.info(f"成功加载策略参数")
            
            # 恢复优化器状态
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info(f"成功加载优化器状态")
            
            # 恢复学习率调度器状态
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info(f"成功加载学习率调度器状态")
            
            # 恢复训练进度
            if "update" in checkpoint:
                start_update = checkpoint["update"]
                logger.info(f"从更新步骤 {start_update} 继续训练")
        
        logger.info(f"config: {self.config}")
        logger.warning(f"====== 开始扩散策略离线训练 ======")
        logger.info(
            "策略参数数量: {}".format(
                sum(param.numel() for param in self.policy.parameters())
            )
        )
        
        # 训练循环
        t_start = time.time()
        self.update_count = start_update
        
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            
            # 计算每个epoch的iterations数
            iterations_per_epoch = len(self.train_loader)
            total_epochs = (self.config.NUM_UPDATES + iterations_per_epoch - 1) // iterations_per_epoch
            
            logger.info(f"总epochs: {total_epochs}, 每epoch {iterations_per_epoch} iterations")
            
            for epoch in range(total_epochs):
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch+1}/{total_epochs}")
                logger.info(f"{'='*60}")
                
                # 训练阶段
                self.policy.train()
                epoch_losses = []
                
                train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
                
                for batch_idx, batch in enumerate(train_iter):
                    # 如果已经达到目标更新次数，停止训练
                    if self.update_count >= self.config.NUM_UPDATES:
                        logger.info(f"达到目标更新次数 {self.config.NUM_UPDATES}，停止训练")
                        break
                    
                    # 将数据移到设备
                    obs = batch['obs'].to(self.device)  # [B, n_obs_steps, C, H, W]
                    actions = batch['action'].to(self.device)  # [B, n_action_steps]
                    
                    # 清零梯度
                    self.optimizer.zero_grad()
                    
                    # 前向传播（计算损失）
                    # 如果使用DataParallel，需要通过.module访问
                    policy_model = self.policy.module if self._is_data_parallel else self.policy
                    loss = policy_model.compute_loss({
                        'obs': obs,
                        'action': actions
                    })
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    
                    # 更新参数
                    self.optimizer.step()
                    
                    # 更新学习率
                    if self.config.RL.DIFFUSION.use_linear_lr_decay:
                        self.scheduler.step()
                    
                    epoch_losses.append(loss.item())
                    
                    # 记录日志
                    if self.update_count % self.config.LOG_INTERVAL == 0:
                        avg_loss = np.mean(epoch_losses[-100:])  # 最近100个batch的平均loss
                        logger.info(
                            f"Update: {self.update_count}, Loss: {loss.item():.6f}, "
                            f"Avg Loss: {avg_loss:.6f}, LR: {self.scheduler.get_last_lr()[0]:.2e}, "
                            f"Time: {time.time() - t_start:.1f}s"
                        )
                        
                        # 记录到tensorboard
                        writer.add_scalar('Training/Loss', loss.item(), self.update_count)
                        writer.add_scalar('Training/Avg_Loss', avg_loss, self.update_count)
                        writer.add_scalar('Training/Learning_Rate', self.scheduler.get_last_lr()[0], self.update_count)
                    
                    # 保存检查点
                    if (self.update_count + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(f"ckpt.{self.update_count}.pth")
                    
                    self.update_count += 1
                
                # Epoch结束，计算验证集loss
                if len(self.val_loader) > 0:
                    val_loss = self._validate()
                    logger.info(f"Epoch {epoch+1} - 训练Loss: {np.mean(epoch_losses):.6f}, 验证Loss: {val_loss:.6f}")
                    writer.add_scalar('Validation/Loss', val_loss, epoch)
                else:
                    logger.info(f"Epoch {epoch+1} - 训练Loss: {np.mean(epoch_losses):.6f}")
                
                # 每个epoch保存一次
                self.save_checkpoint(f"ckpt.epoch_{epoch}.pth")
        
        # 保存最终模型
        self.save_checkpoint("final_model.pth")
        logger.info("训练完成！")
    
    def _validate(self) -> float:
        """在验证集上评估"""
        self.policy.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                obs = batch['obs'].to(self.device)
                actions = batch['action'].to(self.device)
                
                try:
                    # 如果使用DataParallel，需要通过.module访问
                    policy_model = self.policy.module if self._is_data_parallel else self.policy
                    loss = policy_model.compute_loss({
                        'obs': obs,
                        'action': actions
                    })
                    val_losses.append(loss.item())
                except:
                    continue
        
        self.policy.train()
        return np.mean(val_losses) if val_losses else 0.0
    
    def eval(self, eval_interval: int = 1, prev_ckpt_ind: int = -1, use_last_ckpt: bool = False) -> None:
        """评估扩散策略"""
        logger.info("评估功能待实现")
        # TODO: 实现评估逻辑，需要创建环境来测试策略性能
        pass
