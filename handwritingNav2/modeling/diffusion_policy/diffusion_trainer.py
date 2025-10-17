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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from gym import spaces

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image

from ..common.base_trainer import BaseRLTrainer
from ..common.env_utils import construct_envs
from ..common.tensorboard_utils import TensorboardWriter
from ..common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation
)
from .diffusion_nav_policy import DiffusionNavPolicy
from .normalizer import LinearNormalizer, Normalizer
from ..ppo.policy import HandWritingNavPolicy


class DiffusionTrainer(BaseRLTrainer):
    """基于扩散策略的导航训练器"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.envs = None
        self.optimizer = None
        self.scheduler = None
        
        # 设置是否使用专家动作作为辅助
        self.use_expert_actions = config.USE_EXPERT_ACTIONS if hasattr(config, 'USE_EXPERT_ACTIONS') else False
        
        # 成功距离，用于ShortestPathFollower判断何时停止
        self.success_distance = config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE if hasattr(config, 'TASK_CONFIG') and hasattr(config.TASK_CONFIG, 'TASK') and hasattr(config.TASK_CONFIG.TASK, 'SUCCESS') and hasattr(config.TASK_CONFIG.TASK.SUCCESS, 'SUCCESS_DISTANCE') else 1.0
        
    def _setup_policy(self, config: Config, observation_space=None) -> None:
        """设置扩散策略"""
        logger.add_filehandler(self.config.LOG_FILE)
        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]
            
        # 从配置中获取参数
        use_pointnav = self.config.USE_POINTNAV if hasattr(self.config, 'USE_POINTNAV') else False
        predict_goal = self.config.PREDICT_GOAL if hasattr(self.config, 'PREDICT_GOAL') else True
        
        self.policy = DiffusionNavPolicy(
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            hidden_size=config.RL.PPO.hidden_size,
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
        
        # 设置优化器
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.RL.DIFFUSION.lr,
            weight_decay=config.RL.DIFFUSION.weight_decay
        )
        
        # 设置学习率调度器
        self.scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES)
        )
        
    def save_checkpoint(self, file_name: str) -> None:
        """保存检查点"""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "update": self.update_count if hasattr(self, "update_count") else 0,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )
    
    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, *args, **kwargs)
        return checkpoint
    
    def _collect_trajectory_data(self, num_episodes: int = 100) -> Dict[str, torch.Tensor]:
        """收集轨迹数据用于训练"""
        observations = self.envs.reset()
        batch = batch_obs(observations)
        
        # 存储轨迹数据
        obs_sequences = []
        action_sequences = []
        
        episode_count = 0
        current_obs_seq = []
        current_action_seq = []
        
        while episode_count < num_episodes:
            # 编码观察
            with torch.no_grad():
                obs_features = self.policy.encode_observations(batch)
                current_obs_seq.append(obs_features)
            
            # 随机动作（用于数据收集）
            actions = [self.envs.action_spaces[0].sample() for _ in range(self.envs.num_envs)]
            current_action_seq.append(torch.tensor(actions, dtype=torch.long))
            
            # 执行动作
            outputs = self.envs.step(actions)
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            
            # 检查是否有episode结束
            for i, done in enumerate(dones):
                if done:
                    if len(current_obs_seq) >= self.policy.n_obs_steps and len(current_action_seq) >= self.policy.n_action_steps:
                        # 保存轨迹
                        obs_seq = torch.stack(current_obs_seq[-self.policy.n_obs_steps:])
                        action_seq = torch.stack(current_action_seq[-self.policy.n_action_steps:])
                        
                        obs_sequences.append(obs_seq)
                        action_sequences.append(action_seq)
                        episode_count += 1
                    
                    # 重置序列
                    current_obs_seq = []
                    current_action_seq = []
            
            # 更新观察
            batch = batch_obs(observations)
        
        # 转换为张量
        obs_tensor = torch.stack(obs_sequences)  # [num_episodes, n_obs_steps, obs_dim]
        action_tensor = torch.stack(action_sequences)  # [num_episodes, n_action_steps, action_dim]
        
        return {
            'obs': obs_tensor,
            'action': action_tensor
        }
    
    def _setup_normalizer(self, data: Dict[str, torch.Tensor]) -> None:
        """设置归一化器"""
        # 为观察数据创建归一化器
        obs_normalizer = Normalizer.from_data(data['obs'])
        self.policy.normalizer.add_normalizer('obs', obs_normalizer)
        
        # 为动作数据创建归一化器
        action_normalizer = Normalizer.from_data(data['action'].float())
        self.policy.normalizer.add_normalizer('action', action_normalizer)
    
    def train(self, checkpoint_path: Optional[str] = None) -> None:
        """训练扩散策略"""
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        
        # 创建环境
        from habitat.core.environments import HandWritingNavRLEnv
        self.envs = construct_envs(self.config, HandWritingNavRLEnv)
        
        # 设置设备
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        
        # 设置策略
        self._setup_policy(self.config)
        
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
        logger.warning(f"====== 开始扩散策略训练 ======")
        logger.info(
            "策略参数数量: {}".format(
                sum(param.numel() for param in self.policy.parameters())
            )
        )
        
        # 收集初始数据并设置归一化器
        logger.info("收集初始数据用于设置归一化器...")
        initial_data = self._collect_trajectory_data(num_episodes=50)
        self._setup_normalizer(initial_data)
        logger.info("归一化器设置完成")
        
        # 训练循环
        t_start = time.time()
        self.update_count = start_update
        
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                # 更新学习率
                if self.config.RL.DIFFUSION.use_linear_lr_decay:
                    self.scheduler.step()
                
                # 收集新的训练数据
                train_data = self._collect_trajectory_data(num_episodes=10)
                
                # 训练步骤
                self.policy.train()
                self.optimizer.zero_grad()
                
                # 计算损失
                loss = self.policy.compute_loss(train_data)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                
                # 记录日志
                if update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tloss: {:.6f}\ttime: {:.3f}s".format(
                            update, loss.item(), time.time() - t_start
                        )
                    )
                    
                    # 记录到tensorboard
                    writer.add_scalar('Training/Loss', loss.item(), update)
                    writer.add_scalar('Training/Learning_Rate', self.scheduler.get_lr()[0], update)
                
                # 保存检查点
                if (update + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{update}.pth")
                
                self.update_count = update
        
        self.envs.close()
    
    def eval(self, eval_interval: int = 1, prev_ckpt_ind: int = -1, use_last_ckpt: bool = False) -> None:
        """评估扩散策略"""
        # 这里可以实现评估逻辑
        # 类似于PPO训练器中的_eval_checkpoint方法
        pass


