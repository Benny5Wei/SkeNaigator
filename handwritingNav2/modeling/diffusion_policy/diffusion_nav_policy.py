#!/usr/bin/env python3
"""
扩散策略导航模型

基于扩散概率模型(DDPM)的导航策略，用于手绘地图导航任务。
该模型通过学习动作序列的分布来生成平滑、一致的导航行为。

主要特性:
- 多模态感知: 支持RGB、深度、手绘地图、SLAM等多种传感器输入
- 目标预测: 从手绘地图预测目标位置
- 序列建模: 直接学习动作序列分布，生成更平滑的行为
- Habitat兼容: 完全兼容Habitat仿真器环境

作者: AI Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .conditional_unet1d import ConditionalUnet1D
from .mask_generator import LowdimMaskGenerator
from .normalizer import LinearNormalizer
from ..models.visual_cnn import VisualCNN
from ..models.advanced_goal_predictor import AdvancedGoalPredictor
from ..models.rnn_state_encoder import RNNStateEncoder


class DiffusionNavPolicy(nn.Module):
    """
    基于扩散策略的导航策略，用于手绘地图导航任务
    结合了视觉编码、目标预测和扩散去噪过程
    """
    
    def __init__(
        self,
        observation_space,  # Habitat观察空间，包含各种传感器数据
        action_space,       # Habitat动作空间，通常是离散的4个动作
        goal_sensor_uuid: str,  # 目标传感器UUID，用于识别手绘地图
        hidden_size: int = 512,  # 隐藏层大小，用于特征编码
        horizon: int = 16,  # 动作序列长度，扩散模型生成的动作序列长度
        n_action_steps: int = 4,  # 实际执行的动作步数，从序列中取前N步执行
        n_obs_steps: int = 3,  # 观察步数，用于序列建模的历史观察数量
        obs_dim: int = 512,  # 观察特征维度，编码后的观察特征大小
        action_dim: int = 4,  # 动作维度 (前进、左转、右转、停止)
        num_inference_steps: int = 20,  # 扩散推理步数，去噪过程的步数
        extra_rgb: bool = False,  # 是否使用额外的RGB传感器
        extra_depth: bool = True,  # 是否使用深度传感器
        slam: bool = False,  # 是否使用SLAM地图
        use_vae: bool = False,  # 是否使用VAE编码器
        use_pointnav: bool = True,  # 是否使用PointNav传感器
        predict_goal: bool = True,  # 是否启用目标预测
        obs_as_global_cond: bool = True,  # 是否使用全局条件（观察作为全局条件）
        **kwargs  # 其他参数
    ):
        super().__init__()
        
        # 存储基本参数
        self.goal_sensor_uuid = goal_sensor_uuid  # 目标传感器标识符
        self.hidden_size = hidden_size  # 隐藏层维度
        self.horizon = horizon  # 动作序列长度
        self.n_action_steps = n_action_steps  # 实际执行的动作步数
        self.n_obs_steps = n_obs_steps  # 观察步数
        self.obs_dim = obs_dim  # 观察特征维度
        self.action_dim = action_dim  # 动作维度
        self.obs_as_global_cond = obs_as_global_cond  # 是否使用全局条件
        self.predict_goal = predict_goal  # 是否启用目标预测
        
        # ==================== 视觉编码器模块 ====================
        # 手绘地图编码器：处理手绘地图输入，提取空间特征
        self.map_encoder = VisualCNN(
            observation_space, hidden_size, 
            extra_map=True, goal_id=self.goal_sensor_uuid
        )
        
        # RGB图像编码器：处理RGB图像输入，提取视觉特征
        if extra_rgb:
            self.visual_encoder = VisualCNN(
                observation_space, hidden_size, extra_rgb=True
            )
        
        # 深度图像编码器：处理深度图像输入，提取3D空间特征
        if extra_depth:
            self.depth_encoder = VisualCNN(
                observation_space, hidden_size, extra_depth=True
            )
        
        # SLAM地图编码器：处理SLAM构建的地图，提取环境结构特征
        if slam:
            self.slam_encoder = VisualCNN(
                observation_space, hidden_size, slam=True
            )
        
        # VAE编码器：使用变分自编码器处理手绘地图，学习潜在表示
        if use_vae:
            from ..models.vae import VAE
            self.vae_encoder = VAE(self.goal_sensor_uuid, hidden_size)
        
        # ==================== 目标预测模块 ====================
        # 高级目标预测器：从手绘地图预测目标位置
        # 使用Transformer架构和射线特征进行空间推理
        if self.predict_goal:
            self.goal_predictor = AdvancedGoalPredictor(
                k_points=25, in_dim=10  # 5x5关键点网格，8射线+2坐标特征
            )
        
        # ==================== 扩散模型模块 ====================
        # 条件UNet 1D：核心扩散模型，用于生成动作序列
        # 基于观察条件生成平滑的动作序列分布
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,  # 输入维度（动作维度）
            global_cond_dim=obs_dim,  # 全局条件维度（观察特征）
            down_dims=[256, 512, 1024],  # 下采样层维度
            diffusion_step_embed_dim=256,  # 扩散步嵌入维度
            down_step_sizes=[1, 1, 1],  # 下采样步长
            kernel_size=5,  # 卷积核大小
            n_groups=8  # 组归一化组数
        )
        
        # 动作嵌入层：将离散动作转换为连续表示
        # 用于处理Habitat的离散动作空间（0-3）
        self.action_embedding = nn.Embedding(action_space.n, action_dim)
        
        # DDPM噪声调度器：控制扩散过程中的噪声水平
        # 定义从纯噪声到清晰动作序列的去噪过程
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,  # 训练时间步数
            beta_start=0.0001,  # 起始噪声水平
            beta_end=0.02,  # 结束噪声水平
            beta_schedule="linear",  # 噪声调度类型
            prediction_type="epsilon"  # 预测类型（噪声）
        )
        
        # ==================== 辅助模块 ====================
        # 掩码生成器：生成扩散过程中的条件掩码
        # 用于控制哪些位置需要条件化（观察数据）
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,  # 动作维度
            obs_dim=0 if obs_as_global_cond else obs_dim,  # 观察维度
            max_n_obs_steps=n_obs_steps,  # 最大观察步数
            fix_obs_steps=True,  # 固定观察步数
            action_visible=True  # 动作可见（不mask动作，对所有位置计算损失）
        )
        
        # 数据归一化器：对观察和动作数据进行归一化
        # 确保训练稳定性和收敛性
        self.normalizer = LinearNormalizer()
        
        # 特征融合网络：将多模态特征融合为统一表示
        # 用于将不同传感器的特征映射到统一的观察空间
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size, obs_dim),  # 降维映射
            nn.ReLU(),  # 非线性激活
            nn.Linear(obs_dim, obs_dim)  # 最终特征维度
        )
        
        # 使用EfficientNet作为图像编码器（参考flona_vint模型）
        # 输入: [B, C, H, W]，输出: [B, obs_dim]
        obs_channels = 1 if extra_depth and not extra_rgb else 3
        if extra_rgb and extra_depth:
            obs_channels = 4
        
        # 导入并初始化EfficientNet
        try:
            from efficientnet_pytorch import EfficientNet
            self.image_backbone = EfficientNet.from_name('efficientnet-b0', in_channels=obs_channels)
            # 替换BatchNorm为GroupNorm以提高稳定性
            self.image_backbone = self._replace_bn_with_gn(self.image_backbone)
            num_features = self.image_backbone._fc.in_features
            
            # 压缩层：将EfficientNet特征映射到obs_dim
            if num_features != obs_dim:
                self.compress_image_features = nn.Linear(num_features, obs_dim)
            else:
                self.compress_image_features = nn.Identity()
        except ImportError:
            # 如果没有efficientnet_pytorch，使用简单的CNN
            print("警告: 无法导入efficientnet_pytorch，使用简单CNN作为后备方案")
            self.image_backbone = nn.Sequential(
                nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.compress_image_features = nn.Linear(64, obs_dim)
        
        self.num_inference_steps = num_inference_steps
        self.kwargs = kwargs
    
    def _replace_bn_with_gn(self, module: nn.Module, features_per_group: int = 16) -> nn.Module:
        """将BatchNorm替换为GroupNorm以提高训练稳定性"""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_groups = max(1, child.num_features // features_per_group)
                new_module = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features)
                setattr(module, name, new_module)
            else:
                self._replace_bn_with_gn(child, features_per_group)
        return module
        
    def encode_observations(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        编码观察数据为特征向量 - 与Habitat环境兼容
        
        该方法将多模态观察数据（RGB、深度、手绘地图等）编码为统一的特征表示，
        用于后续的扩散模型条件生成。
        
        Args:
            observations: Habitat环境返回的观察字典，包含各种传感器数据
                - 'rgb': RGB图像 [H, W, 3]
                - 'depth': 深度图像 [H, W, 1] 
                - 'handwriting_instr': 手绘地图 [H, W, 3]
                - 'pointgoal': 目标位置 [2]
                - 'gps': GPS坐标 [2]
                - 'compass': 朝向角度 [1]
        
        Returns:
            torch.Tensor: 编码后的观察特征 [batch_size, obs_dim]
        """
        features = []
        
        # 处理Habitat的观察格式
        if isinstance(observations, dict):
            # 确保所有观察都是张量格式
            processed_obs = {}
            for key, value in observations.items():
                if isinstance(value, np.ndarray):
                    processed_obs[key] = torch.from_numpy(value).float()
                elif isinstance(value, torch.Tensor):
                    processed_obs[key] = value
                else:
                    processed_obs[key] = torch.tensor(value).float()
        else:
            processed_obs = observations
        
        # 编码手绘地图
        if self.goal_sensor_uuid in processed_obs:
            map_features = self.map_encoder(processed_obs)
            features.append(map_features)
        
        # 编码其他传感器数据
        if hasattr(self, 'visual_encoder') and 'rgb' in processed_obs:
            rgb_features = self.visual_encoder(processed_obs)
            features.append(rgb_features)
            
        if hasattr(self, 'depth_encoder') and 'depth' in processed_obs:
            depth_features = self.depth_encoder(processed_obs)
            features.append(depth_features)
            
        if hasattr(self, 'slam_encoder') and 'slam' in processed_obs:
            slam_features = self.slam_encoder(processed_obs)
            features.append(slam_features)
            
        if hasattr(self, 'vae_encoder') and self.goal_sensor_uuid in processed_obs:
            vae_features = self.vae_encoder(processed_obs)
            features.append(vae_features)
        
        # 目标预测
        if self.predict_goal and 'slam' in processed_obs and self.goal_sensor_uuid in processed_obs:
            goal_features = self._predict_goal_position(processed_obs)
            features.append(goal_features)
        elif hasattr(self, 'use_pointnav') and 'pointgoal' in processed_obs:
            # 使用PointNav数据
            pointgoal_features = processed_obs['pointgoal']
            if len(pointgoal_features.shape) == 1:
                pointgoal_features = pointgoal_features.unsqueeze(0)
            features.append(pointgoal_features)
        
        # 融合所有特征
        if features:
            combined_features = torch.cat(features, dim=1)
            encoded_features = self.feature_fusion(combined_features)
        else:
            # 如果没有特征，创建零向量
            if isinstance(processed_obs, dict) and processed_obs:
                batch_size = next(iter(processed_obs.values())).shape[0]
                device = next(iter(processed_obs.values())).device
            else:
                batch_size = 1
                device = torch.device('cpu')
            encoded_features = torch.zeros(batch_size, self.obs_dim, device=device)
        
        return encoded_features
    
    def _predict_goal_position(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从手绘地图预测目标位置
        """
        # 这里简化实现，实际应该使用AdvancedGoalPredictor的完整逻辑
        # 返回目标预测的极坐标表示
        batch_size = observations[self.goal_sensor_uuid].shape[0]
        device = observations[self.goal_sensor_uuid].device
        
        # 简化的目标预测：返回零向量作为占位符
        # 实际实现应该使用AdvancedGoalPredictor的完整逻辑
        goal_features = torch.zeros(batch_size, 2, device=device)  # [distance, angle]
        return goal_features
    
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        条件采样生成动作序列
        
        这是扩散策略的核心方法，通过逐步去噪过程生成动作序列。
        从纯噪声开始，在观察条件的指导下，逐步生成清晰的动作序列。
        
        Args:
            condition_data: 条件数据，包含观察信息 [batch_size, horizon, obs_dim]
            condition_mask: 条件掩码，标记哪些位置需要条件化 [batch_size, horizon, obs_dim]
            global_cond: 全局条件，观察特征 [batch_size, obs_dim]
            generator: 随机数生成器，用于可重现的采样
            **kwargs: 其他参数传递给噪声调度器
        
        Returns:
            torch.Tensor: 生成的动作序列 [batch_size, horizon, action_dim]
        """
        # 初始化随机轨迹
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )
        
        # 设置时间步
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        # 扩散去噪过程
        for t in self.noise_scheduler.timesteps:
            # 应用条件
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # 预测噪声
            model_output = self.unet(
                trajectory, t, global_cond=global_cond
            )
            
            # 计算前一步
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample
        
        # 确保条件被强制执行
        trajectory[condition_mask] = condition_data[condition_mask]
        
        return trajectory
    
    def predict_action(
        self, 
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        预测动作序列
        """
        assert 'obs' in obs_dict
        
        # 归一化观察数据
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, To, Do = nobs.shape
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        
        device = self.device
        dtype = self.dtype
        
        # 构建条件
        if self.obs_as_global_cond:
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # 通过inpainting进行条件
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True
            global_cond = None
        
        # 运行采样
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
            **self.kwargs
        )
        
        # 反归一化预测
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        # 获取动作
        start = To
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        
        if not self.obs_as_global_cond:
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
            
        return result
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算扩散损失
        
        Args:
            batch: 包含'obs'和'action'的字典
                   obs: [B, n_obs_steps, C, H, W] 原始图像
                   action: [B, n_action_steps] 动作序列
        """
        # 归一化输入
        nbatch = self.normalizer.normalize(batch)
        raw_obs = nbatch['obs']  # [B, n_obs_steps, C, H, W]
        action = nbatch['action']  # [B, n_action_steps]
        
        # 编码观察数据: 将图像编码为特征向量
        # 需要将 [B, n_obs_steps, C, H, W] 处理成 [B, n_obs_steps, obs_dim]
        B = raw_obs.shape[0]
        n_obs = raw_obs.shape[1]
        C, H, W = raw_obs.shape[2:]
        
        # 重塑为 [B*n_obs_steps, C, H, W] 以便批量编码
        raw_obs_flat = raw_obs.reshape(B * n_obs, C, H, W)
        
        # 使用EfficientNet编码器处理图像（参考flona_vint）
        # [B*n_obs_steps, C, H, W] -> [B*n_obs_steps, obs_dim]
        try:
            # EfficientNet编码流程
            features = self.image_backbone.extract_features(raw_obs_flat)
            features = self.image_backbone._avg_pooling(features)
            if self.image_backbone._global_params.include_top:
                features = features.flatten(start_dim=1)
                features = self.image_backbone._dropout(features)
            encoded_obs_flat = self.compress_image_features(features)
        except AttributeError:
            # 如果使用简单CNN（后备方案）
            features = self.image_backbone(raw_obs_flat)
            encoded_obs_flat = self.compress_image_features(features)
        
        # 重塑回 [B, n_obs_steps, obs_dim]
        obs = encoded_obs_flat.reshape(B, n_obs, self.obs_dim)
        
        # 处理动作：将离散动作嵌入到连续空间
        # action: [B, n_action_steps] -> [B, n_action_steps, action_dim]
        # 使用嵌入层而不是one-hot，这样可以学习更好的表示，并且值域更大
        action_embedded = self.action_embedding(action.long())  # [B, n_action_steps, action_dim]
        
        # 归一化到[-1, 1]范围，增大损失的动态范围
        action_embedded = torch.tanh(action_embedded)  # [B, n_action_steps, action_dim]
        
        # 扩展动作序列到horizon长度
        # 通过重复最后一个动作来填充
        if action_embedded.shape[1] < self.horizon:
            # 需要填充到horizon长度
            last_action = action_embedded[:, -1:, :]  # [B, 1, action_dim]
            pad_size = self.horizon - action_embedded.shape[1]
            padding = last_action.repeat(1, pad_size, 1)  # [B, pad_size, action_dim]
            action_trajectory = torch.cat([action_embedded, padding], dim=1)  # [B, horizon, action_dim]
        else:
            # 如果超过horizon，截断
            action_trajectory = action_embedded[:, :self.horizon, :]
        
        # 处理观察条件
        if self.obs_as_global_cond:
            # 使用观察作为全局条件
            # 对多个观察步进行平均池化，得到 [B, obs_dim]
            # 参考 flona_vint: obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)
            obs_for_cond = obs[:, :self.n_obs_steps, :]  # [B, n_obs_steps, obs_dim]
            global_cond = torch.mean(obs_for_cond, dim=1)  # [B, obs_dim]
            # 轨迹只包含动作
            trajectory = action_trajectory  # [B, horizon, action_dim]
        else:
            # 观察和动作都包含在轨迹中
            # 需要扩展观察序列到horizon
            if obs.shape[1] < self.horizon:
                last_obs = obs[:, -1:, :]
                pad_size = self.horizon - obs.shape[1]
                obs_padding = last_obs.repeat(1, pad_size, 1)
                obs_trajectory = torch.cat([obs, obs_padding], dim=1)
            else:
                obs_trajectory = obs[:, :self.horizon, :]
            
            trajectory = torch.cat([action_trajectory, obs_trajectory], dim=-1)
            global_cond = None
        
        # 生成inpainting掩码
        condition_mask = self.mask_generator(trajectory.shape, device=trajectory.device)
        
        # 采样噪声
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        
        # 采样随机时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        
        # 添加噪声
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        
        # 计算损失掩码
        loss_mask = ~condition_mask
        
        # 应用条件
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # 预测噪声残差
        pred = self.unet(
            noisy_trajectory, timesteps,
            global_cond=global_cond
        )
        
        # 计算损失
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        """设置归一化器"""
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def reset(self):
        """重置策略状态"""
        pass
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
