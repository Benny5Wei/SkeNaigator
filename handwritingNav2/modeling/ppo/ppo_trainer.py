#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import math
import logging
from collections import deque
from typing import Dict, List, Optional
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical
from tqdm import tqdm
from numpy.linalg import norm
from gym import spaces
from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
import sys
# 动态获取项目根目录，添加 utils_fmm 到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils_fmm"))
from mapper import Mapper
from habitat.core.environments import HandWritingNavRLEnv
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from ..common.base_trainer import BaseRLTrainer
from ..common.baseline_registry import baseline_registry
from ..common.env_utils import construct_envs
from ..common.rollout_storage import RolloutStorage
from ..common.tensorboard_utils import TensorboardWriter
from ..common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation
)
from .policy import HandWritingNavPolicy
from .ppo import PPO

# @baseline_registry.register_trainer(name="HWNavTrainer")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        
        # 设置是否使用专家动作作为辅助
        self.use_expert_actions = config.USE_EXPERT_ACTIONS if hasattr(config, 'USE_EXPERT_ACTIONS') else False
        
        # 成功距离，用于ShortestPathFollower判断何时停止
        self.success_distance = config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE if hasattr(config, 'TASK_CONFIG') and hasattr(config.TASK_CONFIG, 'TASK') and hasattr(config.TASK_CONFIG.TASK, 'SUCCESS') and hasattr(config.TASK_CONFIG.TASK.SUCCESS, 'SUCCESS_DISTANCE') else 1.0
        
    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)
        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]
        # 从配置中获取参数，如果未设置则设置默认值
        use_pointnav = self.config.USE_POINTNAV if hasattr(self.config, 'USE_POINTNAV') else False
        predict_goal = self.config.PREDICT_GOAL if hasattr(self.config, 'PREDICT_GOAL') else True
        
        self.actor_critic = HandWritingNavPolicy(
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            extra_rgb=self.config.EXTRA_RGB,
            extra_depth=self.config.EXTRA_DEPTH,
            slam = self.config.SLAM,
            use_vae = self.config.USE_VAE,
            use_pointnav = use_pointnav,  # 传递PointNav参数
            predict_goal = predict_goal   # 传递目标预测参数
        )
        self.actor_critic.to(self.device)

        # 为Goal Predictor创建一个单独的优化器，确保其参数更新与PPO训练独立
        # 当启用目标预测时使用
        self.goal_predictor_optimizer = None
        
        # 直接访问目标预测器模块，而不是通过参数名称过滤
        if hasattr(self.config, 'PREDICT_GOAL') and self.config.PREDICT_GOAL:
            # 检查模型是否已启用目标预测
            if hasattr(self.actor_critic.net, '_predict_goal') and self.actor_critic.net._predict_goal:
                # 检查目标预测器是否存在
                if hasattr(self.actor_critic.net, 'goal_predictor'):
                    # 直接获取目标预测器的参数
                    goal_predictor_params = self.actor_critic.net.goal_predictor.parameters()
                    
                    # 创建高级目标预测器的优化器
                    # 为高级目标预测器设置特定的学习率
                    goal_pred_lr = ppo_cfg.advanced_goal_pred_lr if hasattr(ppo_cfg, 'advanced_goal_pred_lr') else \
                                  (ppo_cfg.goal_pred_lr if hasattr(ppo_cfg, 'goal_pred_lr') else ppo_cfg.lr * 0.5)
                    print(f"初始化高级目标预测器优化器，学习率: {goal_pred_lr}")
                    
                    # 为高级目标预测器添加权重衰减(weight decay)来防止过拟合
                    weight_decay = ppo_cfg.weight_decay if hasattr(ppo_cfg, 'weight_decay') else 1e-4
                    self.goal_predictor_optimizer = torch.optim.AdamW(
                        goal_predictor_params,
                        lr=goal_pred_lr,
                        eps=ppo_cfg.eps,
                        weight_decay=weight_decay
                    )

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )
        
        # 设置是否使用专家动作引导
        self.use_expert_actions = self.config.USE_EXPERT_ACTIONS
        
        # 如果启用专家动作，记录成功距离用于ShortestPathFollower
        if self.use_expert_actions:
            # 在训练过程中初始化path followers
            # VectorEnv不直接暴露内部环境，所以在_collect_rollout_step中动态创建
            pass  # 缩进块需要至少一条语句

    def save_checkpoint(self, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
            "update": self.update_count if hasattr(self, "update_count") else 0,
            "optimizer": self.agent.optimizer.state_dict() if hasattr(self.agent, "optimizer") else None,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        checkpoint = torch.load(checkpoint_path, *args, **kwargs)
        return checkpoint


    def _collect_rollout_step(
        self, rollouts, current_episode_reward, current_episode_step, episode_rewards,
            episode_spls, episode_counts, episode_steps
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )
            
            # 先初始化一个空白专家动作占位符，之后会更新它
            empty_expert_actions = torch.zeros_like(actions)
            rollouts.expert_actions[rollouts.step].copy_(empty_expert_actions)

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        action_per_env = [a[0].item() for a in actions]
        # breakpoint()
        outputs = self.envs.step(action_per_env)
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations)
        # 当有任一环境在此步结束时，重置 SLAM / Free 空间地图，避免在新 episode 中继续累积上一次的地图信息。
        if any(dones):
            rollouts.reset_map(batch)
        batch = rollouts.update_map(batch, action_per_env)
        
        # 添加专家动作奖励 - 如果智能体的动作与专家一致，给予额外奖励
        if hasattr(self, 'use_expert_actions') and self.use_expert_actions:
            expert_reward_bonus = self.config.RL.PPO.expert_reward_coef if hasattr(self.config.RL.PPO, 'expert_reward_coef') else 0.1
            
            for i, info in enumerate(infos):
                if "expert_action" in info:
                    expert_action = info["expert_action"]
                    agent_action = action_per_env[i]
                    
                    # 处理专家动作（可能是one-hot格式）
                    if isinstance(expert_action, np.ndarray):
                        # 从one-hot数组找出非零元素的索引
                        nonzero_indices = np.nonzero(expert_action)[0]
                        if len(nonzero_indices) > 0:
                            expert_action_idx = nonzero_indices[0]  # 取第一个非零元素的索引
                        else:
                            expert_action_idx = 0  # 默认STOP动作
                            
                        # 映射到实际的动作空间 (0-3)
                        if expert_action_idx < 4:
                            expert_action = expert_action_idx
                        else:
                            expert_action = 0  # 超出范围使用STOP动作
                    
                    # 如果智能体动作与专家动作匹配，增加奖励
                    if expert_action == agent_action:
                        rewards[i] += expert_reward_bonus
                        # logging.debug(f"智能体动作[{agent_action}]与专家动作[{expert_action}]匹配，获得额外奖励{expert_reward_bonus}")
        
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )
        spls = torch.tensor(
            [[info['spl']] for info in infos]
        )
        
        # 从环境提供的info字典中获取专家动作
        if hasattr(self, 'use_expert_actions') and self.use_expert_actions:
            expert_actions = []
            
            # 从最近一步的info字典中获取专家动作
            for i, info in enumerate(infos):
                # 如果环境实现了专家动作生成，并在info中提供了expert_action
                if "expert_action" in info:
                    expert_action = info["expert_action"]
                else:
                    # 如果环境没有提供专家动作，使用原PPO动作
                    expert_action = actions[i][0].item()
                    logging.warning(f"环境没有提供专家动作，使用原PPO动作")
                
                expert_actions.append(expert_action)
            
            # 创建tensor存储专家动作
            num_envs = len(expert_actions)
            expert_actions_tensor = torch.zeros(num_envs, 1, dtype=torch.long, device=self.device)
            
            # 处理专家动作（one-hot数组转换为索引）
            for i, action_array in enumerate(expert_actions):
                # 处理7维one-hot数组格式的专家动作
                if isinstance(action_array, np.ndarray):
                    # 从one-hot数组找出非零元素的索引
                    nonzero_indices = np.nonzero(action_array)[0]
                    if len(nonzero_indices) > 0:
                        action_idx = nonzero_indices[0]  # 取第一个非零元素的索引
                    else:
                        action_idx = 0  # 默认STOP动作
                    
                    # 映射到实际的动作空间 (0-3)
                    if action_idx < 4:
                        expert_actions_tensor[i, 0] = action_idx
                    else:
                        expert_actions_tensor[i, 0] = 0  # 超出范围使用STOP动作
                else:
                    # 如果是其他格式，尝试转换为整数
                    action_val = int(action_array) if isinstance(action_array, (int, float)) else 0
                    if 0 <= action_val < 4:
                        expert_actions_tensor[i, 0] = action_val
                    else:
                        expert_actions_tensor[i, 0] = 0  # 使用STOP动作
                    
            # 确保形状匹配
            if expert_actions_tensor.shape != rollouts.expert_actions[rollouts.step].shape:
                # 如果形状不匹配，创建正确形状的副本并复制值
                correct_shape_tensor = torch.zeros_like(rollouts.expert_actions[rollouts.step])
                for i in range(min(expert_actions_tensor.size(0), correct_shape_tensor.size(0))):
                    correct_shape_tensor[i, 0] = expert_actions_tensor[i, 0]
                expert_actions_tensor = correct_shape_tensor

                
            # 更新已初始化的专家动作张量
            rollouts.expert_actions[rollouts.step].copy_(expert_actions_tensor)

        current_episode_reward += rewards
        current_episode_step += 1
        # current_episode_reward is accumulating rewards across multiple updates,
        # as long as the current episode is not finished
        # the current episode reward is added to the episode rewards only if the current episode is done
        # the episode count will also increase by 1
        episode_rewards += (1 - masks) * current_episode_reward
        episode_spls += (1 - masks) * spls
        episode_steps += (1 - masks) * current_episode_step
        episode_counts += 1 - masks
        current_episode_reward *= masks
        current_episode_step *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        
        # 获取最新观测数据
        last_observation = {k: v[-1] for k, v in rollouts.observations.items()}
        
        # --------------- 分开处理前向传播 -----------------
        # 首先不带梯度地获取next_value
        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.prev_actions[-1],
                rollouts.masks[-1],
            ).detach()
        
        # 计算returns
        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )
        
        # --------------- PPO 主更新 -----------------
        ppo_start = time.time()
        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
        print(f"[DEBUG] PPO update time: {time.time()-ppo_start:.3f}s")
        
        # --------------- 目标预测器独立优化 -----------------
        goal_pred_loss = 0.0  # 用于日志
        if (
            hasattr(self.config, "PREDICT_GOAL")
            and self.config.PREDICT_GOAL
            and self.goal_predictor_optimizer is not None
        ):
            # 重新进行前向传播，为目标预测器创建独立的计算图
            # PPO更新完成后才进行这一步，避免inplace操作影响
            _, _ = self.actor_critic.net(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.prev_actions[-1],
                rollouts.masks[-1],
            )
            
            # 现在我们获取一个全新的目标预测损失引用
            loss_tensor = self.actor_critic.net.goal_prediction_loss
            
            if loss_tensor is not None:
                goal_pred_loss = loss_tensor.item()
                if loss_tensor.requires_grad:
                    self.goal_predictor_optimizer.zero_grad()
                    loss_tensor.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.net.goal_predictor.parameters(), 1.0
                    )
                    self.goal_predictor_optimizer.step()
                    print(
                        f"[GOAL PREDICT] 优化完成 - loss={goal_pred_loss:.4f}, has_grad={loss_tensor.requires_grad}"
                    )
                else:
                    print(
                        "[GOAL PREDICT] Loss 不需要梯度，可能是前向传播被包裹在 no_grad 中"
                    )
            else:
                print("[GOAL PREDICT] 未找到损失张量，跳过优化")
        

        rollouts.after_update()
        delta_ppo_update = time.time() - t_update_model
        
        # 返回更新时间、各项损失包括目标预测损失
        return (
            delta_ppo_update, 
            value_loss, 
            action_loss, 
            dist_entropy, 
            goal_pred_loss
        )

    def train(self, checkpoint_path: Optional[str] = None) -> None:
        r"""Main method for training PPO.
        
        Args:
            checkpoint_path: Optional path to a checkpoint to resume training from

        Returns:
            None
        """

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        self.envs = construct_envs(self.config, HandWritingNavRLEnv)
        
        # Move logging statements after environment initialization
        # 已移除调试代码，确认了动作空间结构
        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        
        # 检查是否需要从现有检查点恢复训练
        start_update = 0
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.info(f"加载检查点: {checkpoint_path}")
            checkpoint = self.load_checkpoint(checkpoint_path, map_location=self.device)
            
            # 恢复模型参数
            if "state_dict" in checkpoint:
                self.agent.load_state_dict(checkpoint["state_dict"])
                logger.info(f"成功加载模型参数")
            
            # 如果检查点中存储了训练进度信息，则恢复它
            if "update" in checkpoint:
                start_update = checkpoint["update"]
                logger.info(f"从更新步骤 {start_update} 继续训练")
            
            # 如果检查点中存储了优化器状态，则恢复它
            if "optimizer" in checkpoint and hasattr(self.agent, "optimizer"):
                # 检查模型参数是否有变化
                missing_keys = set(self.agent.state_dict().keys()) - set(checkpoint["state_dict"].keys())
                if missing_keys:
                    # 如果模型结构发生变化，不加载优化器状态
                    logger.info(f"模型结构发生变化，跳过加载优化器状态")
                else:
                    try:
                        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])
                        logger.info(f"成功加载优化器状态")
                    except ValueError as e:
                        logger.warning(f"加载优化器状态失败: {e}")
                        logger.info("使用新初始化的优化器继续训练")
        

        logger.info(f"config: {self.config}")
        logger.warning(f"====== 开始训练 ======")
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )
        # 打印动作空间信息用于调试
        # logging.warning(f"动作空间类型: {type(self.envs.action_spaces[0])}")
        # logging.warning(f"动作空间结构: {self.envs.action_spaces[0]}")
        # logging.warning(f"动作空间大小: {self.envs.action_spaces[0].n if hasattr(self.envs.action_spaces[0], 'n') else 'unknown'}")
        # logging.warning(f"动作空间形状: {self.envs.action_spaces[0].shape if hasattr(self.envs.action_spaces[0], 'shape') else 'unknown'}")
        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            self.config,
            self.device,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)
        rollouts.reset(batch)

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        # episode_rewards and episode_counts accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_spls = torch.zeros(self.envs.num_envs, 1)
        episode_steps = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_step = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_spl = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )
#
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in tqdm(range(ppo_cfg.num_steps)):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        current_episode_step,
                        episode_rewards,
                        episode_spls,
                        episode_counts,
                        episode_steps
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                # 更新PPO智能体和目标预测器
                delta_pth_time, value_loss, action_loss, dist_entropy, goal_pred_loss = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_spl.append(episode_spls.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())

                losses = [value_loss, action_loss, dist_entropy, goal_pred_loss]
                stats = zip(
                    ["count", "reward", "step", 'spl'],
                    [window_episode_counts, window_episode_reward, window_episode_step, window_episode_spl],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                # this reward is averaged over all the episodes happened during window_size updates
                # approximately number of steps is window_size * num_steps
                if update % 10 == 0:
                    writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/SPL", deltas["spl"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)
                    writer.add_scalar('Policy/Value_Loss', value_loss, count_steps)
                    writer.add_scalar('Policy/Action_Loss', action_loss, count_steps)
                    writer.add_scalar('Policy/Entropy', dist_entropy, count_steps)
                    # 记录目标预测损失
                    if isinstance(goal_pred_loss, torch.Tensor):
                        writer.add_scalar('Policy/Goal_Prediction_Loss', goal_pred_loss.item(), count_steps)
                    else:
                        writer.add_scalar('Policy/Goal_Prediction_Loss', goal_pred_loss, count_steps)
                    writer.add_scalar('Policy/Learning_Rate', lr_scheduler.get_lr()[0], count_steps)

                # log stats
                if update > 0 and (update + 1) % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )
                    
                    # 添加SPL相关日志输出
                    logger.info(
                        "Current training stats - frames: {}\tSPL: {:.4f}".format(
                            count_steps,
                            (deltas["spl"] / deltas["count"]) if deltas["count"] > 0 else 0.0,
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_spls = (
                        window_episode_spl[-1] - window_episode_spl[0]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}, SPL: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                                (window_spls / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if (update + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        
        # 创建eval_{ckpt_index}.log文件用于记录评估进度和指标
        ckpt_filename = os.path.basename(checkpoint_path)  # 获取文件名，如ckpt.5.pth
        ckpt_index = ckpt_filename.split('.')[-2]  # 提取检查点索引号，如"5"
        # 将日志文件保存到data目录的上一级
        parent_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        os.makedirs(os.path.join(parent_dir,'eval'), exist_ok=True)
        eval_log_path = os.path.join(parent_dir,'eval',f'eval_{ckpt_index}.log')
        with open(eval_log_path, 'w') as f:
            f.write(f"开始评估checkpoint: {checkpoint_path}\n")
            f.write(f"检查点索引: {ckpt_index}\n")
            f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("===================================\n")
            f.write("评估进度 | SPL | 成功率 | 距离\n")
            f.write("===================================\n")

            
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        
        # 不需要预设总样本数，使用已评估的样本数显示进度

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        # 保存原始checkpoint的传感器尺寸用于创建正确的模型架构
        # 这确保了模型架构与训练时使用的相匹配
        checkpoint_depth_width = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        checkpoint_depth_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        model_resolution = checkpoint_depth_width
        
        # 如果需要不同的显示分辨率，仅修改显示维度，同时保持正确的宽高比
        if self.config.DISPLAY_RESOLUTION != checkpoint_depth_width:
            # 计算原始宽高比
            aspect_ratio = checkpoint_depth_height / checkpoint_depth_width
            
            # 根据宽高比设置新的高度
            new_height = int(self.config.DISPLAY_RESOLUTION * aspect_ratio)
            
            # 设置RGB和深度传感器尺寸，保持宽高比
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = self.config.DISPLAY_RESOLUTION
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = new_height
            config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = self.config.DISPLAY_RESOLUTION
            config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = new_height
            
            print(f"设置传感器分辨率：{self.config.DISPLAY_RESOLUTION}x{new_height}，保持原始宽高比 {aspect_ratio:.3f}")
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        #logger.info(f"env config: {config}")
        # self.envs = construct_envs(
        #     config, get_env_class(config.ENV_NAME)
        # )
        self.envs = construct_envs(config, HandWritingNavRLEnv)

        # Create observation space with the SAME dimensions as used during training
        # This is critical to match the model architecture
        observation_space = self.envs.observation_spaces[0]
        observation_space.spaces['depth'] = spaces.Box(low=0, high=1, shape=(checkpoint_depth_height,
                                                      checkpoint_depth_width, 1), dtype=np.uint8)
        observation_space.spaces['rgb'] = spaces.Box(low=0, high=1, shape=(checkpoint_depth_height,
                                                    checkpoint_depth_width, 3), dtype=np.uint8)
        self._setup_actor_critic_agent(ppo_cfg, observation_space)

        if config.FOLLOW_SHORTEST_PATH:
            follower = ShortestPathFollower(
                self.envs.workers[0]._env.habitat_env.sim, 0.5, False
            )

        # Load the state dict - now the model architecture should match the checkpoint
        try:
            # 使用strict=False允许加载部分参数，适用于模型结构有变化的情况
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
            print("Successfully loaded checkpoint weights (strict=False)")
            # 打印未加载的参数信息
            missing_keys = set(self.agent.state_dict().keys()) - set(ckpt_dict["state_dict"].keys())
            if missing_keys:
                print(f"新增参数 (将被随机初始化): {missing_keys}")
                        # 如果模型结构有变化且存在优化器状态，则跳过加载优化器状态
            if "optimizer" in ckpt_dict and hasattr(self.agent, "optimizer") and missing_keys:
                print(f"模型结构发生变化，跳过加载优化器状态")
            elif "optimizer" in ckpt_dict and hasattr(self.agent, "optimizer"):
                try:
                    self.agent.optimizer.load_state_dict(ckpt_dict["optimizer"])
                    print(f"成功加载优化器状态")
                except ValueError as e:
                    print(f"加载优化器状态失败: {e}")
                    print("使用新初始化的优化器继续训练")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            # Print actual shapes for debugging
            if 'actor_critic.net.depth_encoder.cnn.6.weight' in ckpt_dict["state_dict"]:
                ckpt_shape = ckpt_dict["state_dict"]["actor_critic.net.depth_encoder.cnn.6.weight"].shape
                print(f"Checkpoint shape: {ckpt_shape}")
            if hasattr(self.agent, "actor_critic") and hasattr(self.agent.actor_critic, "net") and \
               hasattr(self.agent.actor_critic.net, "depth_encoder") and \
               hasattr(self.agent.actor_critic.net.depth_encoder, "cnn") and \
               len(self.agent.actor_critic.net.depth_encoder.cnn) > 6 and \
               hasattr(self.agent.actor_critic.net.depth_encoder.cnn[6], "weight"):
                model_shape = self.agent.actor_critic.net.depth_encoder.cnn[6].weight.shape
                print(f"Model shape: {model_shape}")
            raise
        self.actor_critic = self.agent.actor_critic
        # 将模型设置为评估模式以确保所有子模块（包括goal predictor）正确运行
        self.actor_critic.eval()
        # 确保目标预测器也被设置为评估模式
        if hasattr(self.actor_critic.net, '_predict_goal') and self.actor_critic.net._predict_goal:
            if hasattr(self.actor_critic.net, 'goal_predictor'):
                self.actor_critic.net.goal_predictor.eval()
                print("高级目标预测器已设置为评估模式")

        self.metric_uuids = []
        # get name of performance metric, e.g. "spl"
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                metric_cfg.TYPE
            )
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)
            
        # 为每个环境初始化独立的SLAM地图生成器
        self.mappers = [Mapper(self.config, self.device) for _ in range(self.envs.num_envs)]
        
        # 为每个环境添加SLAM观测
        for i in range(self.envs.num_envs):
            # 确保所需的传感器数据可用
            if 'depth' in observations[i]:
                # 每个环境使用自己的mapper实例
                self.mappers[i].reset()
                
                # 处理第一帧信息生成初始SLAM地图
                # 创建observations字典传递给mapper
                # 确保所有数据都是PyTorch张量
                depth = observations[i]['depth']
                gps = observations[i].get('gps', torch.zeros(2))
                compass = observations[i].get('compass', torch.zeros(1))
                
                # 确保数据是张量并添加批量维度
                if not isinstance(depth, torch.Tensor):
                    depth = torch.from_numpy(depth).unsqueeze(0)
                elif depth.dim() == 2:
                    depth = depth.unsqueeze(0)
                
                if not isinstance(gps, torch.Tensor):
                    gps = torch.from_numpy(gps).unsqueeze(0).float()
                else:
                    gps = gps.unsqueeze(0).float()
                
                if not isinstance(compass, torch.Tensor):
                    compass = torch.tensor([[compass]], dtype=torch.float32)
                else:
                    compass = compass.reshape(1, 1).float()
                
                mapper_obs = {
                    'depth': depth,
                    'gps': gps,
                    'compass': compass
                }
                
                # 调用mapper的forward方法生成SLAM地图
                # 需要传递action_per_env参数
                action_per_env = [0]  # 初始化时使用0作为默认动作
                slam_map = self.mappers[i].forward(
                    mapper_obs, 
                    action_per_env,
                    show_obstacle=True,
                    show_visited_area=True,  # 显示已访问区域
                    show_frontier=True       # 显示探索边界
                )
                
                # 添加到观测中
                # 去除多余的维度
                if 'slam' not in observations[i]:
                    # 如果是5维或者前导维度为1，则去掉该维度
                    if slam_map.dim() > 3 and slam_map.size(0) == 1:
                        slam_map = slam_map.squeeze(0)
                    observations[i]['slam'] = slam_map
        
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        # Create tensorboard videos if needed
        rgb_frames = [[] for _ in range(self.envs.num_envs)]  # type: List[List[np.ndarray]]
        audios = [[] for _ in range(self.envs.num_envs)]
        
        # 用于每隔一定步数记录评估进度和指标的变量
        eval_log_interval = 20  # 每隔多少步记录一次日志
        current_spl_values = []
        current_success_values = []
        current_distance_values = []
        last_log_step = 0
        
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False
                )

                prev_actions.copy_(actions)

            if config.FOLLOW_SHORTEST_PATH:
                actions = [follower.get_next_action(
                    self.envs.workers[0]._env.habitat_env.current_episode.goals[0].view_points[0].agent_state.position)]
                outputs = self.envs.step(actions)
            else:
                outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            
            # 首先检查是否有环境完成episode，如果有，先重置其mapper
            for i in range(self.envs.num_envs):
                if dones[i]:
                    # 在环境返回done=True时立即重置mapper
                    if i < len(self.mappers):
                        self.mappers[i].reset()
                        print(f"\n[调试] 环境{i} 的episode完成，重置Mapper")
            
            # 然后再更新每个环境的SLAM地图
            for i in range(self.envs.num_envs):
                if 'depth' in observations[i]:
                    # 创建observations字典传递给mapper
                    # 确保所有数据都是PyTorch张量
                    depth = observations[i]['depth']
                    gps = observations[i].get('gps', torch.zeros(2))
                    compass = observations[i].get('compass', torch.zeros(1))
                    
                    # 确保数据是张量并添加批量维度
                    if not isinstance(depth, torch.Tensor):
                        depth = torch.from_numpy(depth).unsqueeze(0)
                    elif depth.dim() == 2:
                        depth = depth.unsqueeze(0)
                    
                    if not isinstance(gps, torch.Tensor):
                        gps = torch.from_numpy(gps).unsqueeze(0).float()
                    else:
                        gps = gps.unsqueeze(0).float()
                    
                    if not isinstance(compass, torch.Tensor):
                        compass = torch.tensor([[compass]], dtype=torch.float32)
                    else:
                        compass = compass.reshape(1, 1).float()
                    
                    mapper_obs = {
                        'depth': depth,
                        'gps': gps,
                        'compass': compass
                    }
                    
                    # 使用对应环境的mapper更新SLAM地图
                    # 需要传递action_per_env参数
                    # 在这里使用默认动作，因为forward调用可能发生在actions生成之前
                    action_per_env = [0]
                    slam_map = self.mappers[i].forward(
                        mapper_obs, 
                        action_per_env,
                        show_obstacle=True,
                        show_visited_area=True,  # 显示已访问区域
                        show_frontier=True       # 显示探索边界
                    )
                    
                    # 更新观测中的SLAM数据
                    # 去除多余的维度
                    if slam_map.dim() > 3 and slam_map.size(0) == 1:
                        slam_map = slam_map.squeeze(0)
                    observations[i]['slam'] = slam_map
            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observations[i]:
                        for observation in observations[i]['intermediate']:
                            frame = observations_to_image(observation, infos[i])
                            rgb_frames[i].append(frame)
                        del observations[i]['intermediate']

                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                           self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
                    audios[i].append(observations[i]['audiogoal'])

            if config.DISPLAY_RESOLUTION != model_resolution:
                resize_observation(observations, model_resolution)
            batch = batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                # pause envs which runs out of episodes
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    # 使用对应环境的mapper实例重置该环境的SLAM地图
                    if i < len(self.mappers):
                        self.mappers[i].reset()
                    
                    episode_stats = {}
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["{}_reward".format(config.EVAL.SPLIT)] = current_episode_reward[i].item()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    current_episodes[i].stats = episode_stats
                    # 使用(scene_id, episode_id)元组作为键
                    stats_episodes[(current_episodes[i].scene_id, current_episodes[i].episode_id)] = episode_stats
                    
                    # 收集当前SPL、成功率和距离指标
                    if "spl" in infos[i]:
                        current_spl_values.append(infos[i]["spl"])
                    if "success" in infos[i]:
                        current_success_values.append(infos[i]["success"])
                    if "distance_to_goal" in infos[i]:
                        current_distance_values.append(infos[i]["distance_to_goal"])
                    
                    # 每隔一定步数记录评估进度和指标
                    episodes_done = len(stats_episodes)
                    if episodes_done - last_log_step >= eval_log_interval:
                        # 计算当前平均指标
                        avg_spl = sum(current_spl_values) / len(current_spl_values) if current_spl_values else 0
                        avg_success = sum(current_success_values) / len(current_success_values) if current_success_values else 0
                        avg_distance = sum(current_distance_values) / len(current_distance_values) if current_distance_values else 0
                        
                        # 记录到日志文件
                        with open(eval_log_path, 'a') as f:
                            f.write(f"已评估 {episodes_done} 个样本 | SPL: {avg_spl:.4f} | 成功率: {avg_success:.4f} | 距离: {avg_distance:.4f}\n")
                        
                        # 重置计数器和收集器
                        last_log_step = episodes_done
                        current_spl_values = []
                        current_success_values = []
                        current_distance_values = []

                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    t.update()

                    if len(self.config.VIDEO_OPTION) > 0:
                        fps = int(1 / self.config.TASK_CONFIG.SIMULATOR.STEP_TIME)
                        if 'sound' in current_episodes[i].info:
                            sound = current_episodes[i].info['sound']
                        else:
                            sound = current_episodes[i].sound_id.split('/')[1][:-4]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i][:-1],
                            scene_name=current_episodes[i].scene_id.split('/')[3],
                            sound=sound,
                            sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=audios[i][:-1],
                            fps=fps
                        )

                        # observations has been reset but info has not
                        # to be consistent, do not use the last frame
                        rgb_frames[i] = []
                        audios[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        top_down_map = plot_top_down_map(infos[i],
                                                         dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET)
                        scene = current_episodes[i].scene_id.split('/')[3]
                        writer.add_image('{}_{}_{}/{}'.format(config.EVAL.SPLIT, scene, current_episodes[i].episode_id,
                                                              config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5]),
                                         top_down_map,
                                         dataformats='WHC')

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )
            
            # 每隔一定步数记录评估进度和指标
            episodes_done = len(stats_episodes)
            if episodes_done - last_log_step >= eval_log_interval:
                # 计算当前平均指标
                avg_spl = sum(current_spl_values) / len(current_spl_values) if current_spl_values else 0
                avg_success = sum(current_success_values) / len(current_success_values) if current_success_values else 0
                avg_distance = sum(current_distance_values) / len(current_distance_values) if current_distance_values else 0
                
                # 记录到日志文件
                with open(eval_log_path, 'a') as f:
                    f.write(f"已评估 {episodes_done} 个样本 | SPL: {avg_spl:.4f} | 成功率: {avg_success:.4f} | 距离: {avg_distance:.4f}\n")
                
                # 重置计数器和收集器
                last_log_step = episodes_done
                current_spl_values = []
                current_success_values = []
                current_distance_values = []

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(config.TENSORBOARD_DIR, '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        # 确保元组中的所有元素都是字符串类型，然后再进行join操作
        new_stats_episodes = {','.join(map(str, key)): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes


            
        # 将最终评估结果写入日志文件
        with open(eval_log_path, 'a') as f:
            f.write("\n===================================\n")
            f.write("最终评估结果:\n")
            f.write(f"评估完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总样本数: {num_episodes}\n")
            f.write(f"平均奖励: {episode_reward_mean:.6f}\n")
            for metric_uuid in self.metric_uuids:
                f.write(f"平均 {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}\n")
            f.write("===================================\n")
            

        if not config.EVAL.SPLIT.startswith('test'):
            writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
            for metric_uuid in self.metric_uuids:
                writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid],
                                  checkpoint_index)

        self.envs.close()

        result = {
            'episode_reward_mean': episode_reward_mean
        }
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        return result
