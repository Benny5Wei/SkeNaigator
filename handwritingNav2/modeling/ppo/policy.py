import abc

import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary

from ..common.utils import CategoricalNet
from ..models.rnn_state_encoder import RNNStateEncoder
from ..models.visual_cnn import VisualCNN
from ..models.audio_cnn import AudioCNN
from ..models.vae import VAE
from ..models.advanced_goal_predictor import (
    AdvancedGoalPredictor,
    generate_explore_keypoints,
    ray_features_pytorch,
    extract_occupancy_grid,
    extract_sketch_occupancy_grid,
)

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)
        
    def get_action_distribution(self, observations):
        """获取当前状态的动作分布，用于行为克隆损失计算"""
        # 像 act 方法一样处理当前观测并返回分布
        if isinstance(observations, dict):
            features, _ = self.net(
                observations, None, None, None
            )
        else:
            features = observations
        # 返回动作分布
        return self.action_distribution(features)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        # print('Features: ', features.cpu().numpy())
        distribution = self.action_distribution(features)
        # print('Distribution: ', distribution.logits.cpu().numpy())
        value = self.critic(features)
        # print('Value: ', value.item())

        if deterministic:
            action = distribution.mode()
            # print('Deterministic action: ', action.item())
        else:
            action = distribution.sample()
            # print('Sample action: ', action.item())

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class HandWritingNavPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False,
        extra_depth = True,
        slam = False,
        use_vae = False,
        use_pointnav = True,  # 默认启用PointNav传感器
        predict_goal = True   # 默认启用目标预测
    ):

        super().__init__(
            HandWritingNavNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb,
                extra_depth=extra_depth,
                slam=slam,
                use_vae=use_vae,
                use_pointnav=use_pointnav,  # 传递PointNav参数
                predict_goal=predict_goal    # 传递目标预测参数
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class HandWritingNavNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    
    如果启用predict_goal，则使用手绘地图预测目标位置，而不是直接使用PointNav传感器。
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb = False, extra_depth = False, slam = False, use_vae = False, use_pointnav = False, predict_goal = True):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        # self._audiogoal = False # True
        self._use_pointnav = use_pointnav  # 启用PointNav传感器
        self._predict_goal = predict_goal  # 从手绘地图预测目标位置
        self.extra_rgb = extra_rgb
        self.extra_depth = extra_depth
        self.slam = slam
        #self.slam = False
        self.use_vae = use_vae

        self.map_encoder = VisualCNN(observation_space, hidden_size, extra_map = True, goal_id = self.goal_sensor_uuid)  # handwriting map
        
        # 参数用于射线特征 - 减少数量以提高性能
        self.k_rows = 5  # 从9减少到5
        self.k_cols = 5  # 从9减少到5
        self.n_rays = 8   # 从16减少到8
        self.k_points = self.k_rows * self.k_cols  # 25个点而不是81个
        self.in_dim = self.n_rays + 2

        # 目标预测器
        if self._predict_goal:
            # 直接使用高级目标预测器（Transformer + 射线特征）
            self.goal_predictor = AdvancedGoalPredictor(k_points=self.k_points, in_dim=self.in_dim)
            print("Using AdvancedGoalPredictor with Transformer architecture")
        
        if self.extra_rgb:
            self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb = True) # RGB
        if self.extra_depth:
            self.depth_encoder = VisualCNN(observation_space, hidden_size, extra_depth = True) # depth
        if self.slam:
            self.slam_encoder = VisualCNN(observation_space, hidden_size, slam = True) # slam map
        if self.use_vae:
            self.vae_encoder = VAE(self.goal_sensor_uuid, hidden_size)
            vae_model_path = "/data/xhj/handwritingNav/modeling/models/vae_model_final.pth"
            
            vae_state_dict = torch.load(vae_model_path, map_location='cpu')
            vae_model_dict = self.vae_encoder.state_dict()

            filtered_vae_state_dict = {k: v for k, v in vae_state_dict.items() if
                            k.startswith('encoder') or 
                            k.startswith('mean_linear') or 
                            k.startswith('var_linear')}

            vae_model_dict.update(filtered_vae_state_dict)
            self.vae_encoder.load_state_dict(vae_model_dict)

            for name, param in self.vae_encoder.named_parameters():
                if name.startswith('encoder') or name.startswith('mean_linear') or name.startswith('var_linear'):
                    param.requires_grad = False

            for name, param in self.vae_encoder.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

        # 为PointNav添加2维输入(或与您在配置中设置的维度相匹配)
        pointnav_size = 2 if self._use_pointnav else 0  # PointNav传感器维度
        rnn_input_size = (self._hidden_size if extra_rgb else 0) + \
                (self._hidden_size if extra_depth else 0) + \
                (self._hidden_size if self.use_vae else 0) + \
                (self._hidden_size if self.slam else 0) + \
                pointnav_size + \
                (self._hidden_size)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        # if 'rgb' in observation_space.spaces and not extra_rgb:
        #     rgb_shape = observation_space.spaces['rgb'].shape
        #     summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        # if 'depth' in observation_space.spaces:
        #     depth_shape = observation_space.spaces['depth'].shape
        #     summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        # print(self.visual_encoder.cnn)
        # if self._audiogoal:
        #     audio_shape = observation_space.spaces[audiogoal_sensor].shape
        #     summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if self.extra_rgb:
            rgb_result = self.visual_encoder(observations)
            x.append(rgb_result)

        if self.extra_depth:
            depth_result = self.depth_encoder(observations)
            x.append(depth_result)

        if self.slam:
            slam_result = self.slam_encoder(observations)
            x.append(slam_result)
        
        if self.use_vae:
            vae_result = self.vae_encoder(observations)
            x.append(vae_result)
        
        # 处理手绘地图
        map_result = self.map_encoder(observations)
        
        # 从手绘地图预测目标位置
        if self._predict_goal:
            # 获取智能体的位置和朝向信息（如果有）
            has_gps = 'gps' in observations
            has_compass = 'compass' in observations

            # -------- 构造高级目标预测器输入 - 优化批处理版本 --------
            slam_imgs = observations["slam"]  # [B, H, W, 3]
            sketch_imgs = observations[self.goal_sensor_uuid]  # [B, H, W, C]

            # At this point slam_imgs is guaranteed to be a Tensor
            B, H, W, _ = slam_imgs.shape
            device = slam_imgs.device
            
            # 使用内存缓冲区预分配存储空间
            explore_feat_tensor = torch.zeros(B, self.k_points, self.in_dim, device=device)
            sketch_feat_tensor = torch.zeros(B, self.k_points, self.in_dim, device=device)
            keypoints_tensor = torch.zeros(B, self.k_points, 2, device=device)
            ske_keypoints_tensor = torch.zeros(B, self.k_points, 2, device=device)
            
            # 设置半精度计算条件
            use_half = hasattr(self.goal_predictor, 'use_half_precision') and \
                      self.goal_predictor.use_half_precision and \
                      device.type == 'cuda'
                      
            # 提前创建角度数组，避免重复计算
            angles = np.linspace(0, 2*np.pi, self.n_rays, endpoint=False)
            # 从草图中提取目标坐标（紫色点）
            
            goal_coords = None
            if B > 0:  # 确保有数据
                goal_coords = extract_goal_coordinates(sketch_imgs[0], device=str(device))
                if goal_coords is not None:
                    print(f"从草图中提取到目标坐标: {goal_coords}")

            # 批处理：一次性提取占用栅格 - 使用PyTorch操作代替循环
            with torch.cuda.amp.autocast(enabled=use_half):
                # 将输入图像重新排列并转换为连续张量
                slam_imgs_contiguous = slam_imgs.contiguous()
                sketch_imgs_contiguous = sketch_imgs.contiguous()
                
                # 使用并行方法批量处理数据
                for b in range(B):
                    # 从SLAM图中提取占用栅格地图
                    occ = extract_occupancy_grid(slam_imgs_contiguous[b])
                    # 从草图中提取占用栅格
                    sketch_occ = extract_sketch_occupancy_grid(sketch_imgs_contiguous[b])
                    
                    # 生成关键点
                    kps_tensor = generate_explore_keypoints(occ, k_rows=self.k_rows, k_cols=self.k_cols)
                    ske_kps_tensor = generate_explore_keypoints(sketch_occ, k_rows=self.k_rows, k_cols=self.k_cols)
                    # 提取射线特征 - 使用相同的关键点同时处理两种地图
                    # 传递预计算的角度数组和PyTorch张量版本的关键点，避免重复计算和类型转换
                    max_dist = float(max(H, W))  # 使用地图尺寸作为最大距离
                    exp_feat = ray_features_pytorch(occ, kps_tensor, angles, max_dist, device=str(device))
                    sk_feat = ray_features_pytorch(sketch_occ, ske_kps_tensor, angles, max_dist, device=str(device))
                    
                    # 将特征与坐标结合并直接存入预分配的张量
                    explore_feat_tensor[b, :, :self.n_rays] = exp_feat  # 现在exp_feat已经是PyTorch张量
                    explore_feat_tensor[b, :, self.n_rays:] = kps_tensor
                    
                    sketch_feat_tensor[b, :, :self.n_rays] = sk_feat  # 现在sk_feat已经是PyTorch张量
                    sketch_feat_tensor[b, :, self.n_rays:] = ske_kps_tensor
                    
                    keypoints_tensor[b] = kps_tensor
            
            # 使用高级目标预测器 - 返回的是地图像素坐标和注意力权重
            predicted_goal_absolute, attention_weights = self.goal_predictor(
                explore_feat_tensor, sketch_feat_tensor, keypoints_tensor
            )
            
            # 坐标转换处理流程
            # 1. predicted_goal_absolute：预测的地图像素坐标（绝对坐标，范围约为 [0, 512]）
            # 2. 确定agent在地图中的位置和朝向
            # 3. 计算目标相对于agent的向量（先在像素坐标系中，然后转换为米）
            # 4. 转换为极坐标格式（与pointgoal传感器一致）：[距离(米), 相对角度(弧度)]
            
            # 设置地图相关参数（与mapper.py中保持一致）
            # 参数定义 - 地图相关常量
            map_size_cm = 5120         # 地图物理尺寸，单位厘米
            map_resolution = 10        # 分辨率：10cm/像素
            meters_per_pixel = map_resolution / 100.0  # 转换为米/像素
            map_size_px = map_size_cm / map_resolution  # 地图尺寸，单位像素 (512x512)
            map_center_px = map_size_px / 2   # 地图中心点坐标（像素），通常是 (256, 256)
            
            # 预测的目标是地图上的绝对像素坐标
            predicted_goal_px = predicted_goal_absolute  # [B, 2] - 地图像素坐标
            B = predicted_goal_px.shape[0]
            
            # 获取agent的当前世界坐标和朝向
            if 'gps' in observations and 'compass' in observations:
                current_gps = observations['gps']  # [B, 2] - 全局米制坐标
                agent_ori_rad = observations['compass']  # [B, 1] - 全局朝向（弧度）
                
                # 确保维度正确
                if len(current_gps.shape) == 1:
                    current_gps = current_gps.unsqueeze(0)  # [1, 2]
                if len(agent_ori_rad.shape) == 1:
                    agent_ori_rad = agent_ori_rad.unsqueeze(0)  # [1, 1] 
                elif len(agent_ori_rad.shape) == 2 and agent_ori_rad.shape[1] != 1:
                    agent_ori_rad = agent_ori_rad.unsqueeze(1)  # [B, 1]
            else:
                # 如果没有位置信息，使用默认值
                current_gps = torch.zeros(B, 2, device=predicted_goal_px.device)
                agent_ori_rad = torch.zeros(B, 1, device=predicted_goal_px.device)  # 朝北方向(角度为0)
            
            # 1. 获取目标在世界坐标系中的绝对坐标
            if 'agent_initial_pose' in observations:
                # 使用初始位置作为参考
                # print("=== Debug agent_initial_pose ===")
                # print(f"Found agent_initial_pose in observations!")
                # print(f"agent_initial_pose shape: {observations['agent_initial_pose'].shape}")
                # print(f"agent_initial_pose values: {observations['agent_initial_pose']}")
                
                initial_pose = observations['agent_initial_pose']  # [B, 3] - 初始位置和朝向
                initial_position = initial_pose[:, 0:2]  # 初始位置
                initial_orientation = initial_pose[:, 2]  # 初始朝向角度（弧度） [B]
                # print(f"initial_position: {initial_position}")
                # print(f"initial_orientation: {initial_orientation}")
                # print("=== End Debug ===")
                
                # 计算预测目标相对于地图中心的像素偏移
                # 注意：在像素坐标系中，y轴向下为正方向
                goal_px_offset_from_center_x = predicted_goal_px[:, 0] - map_center_px  # 相对于地图中心的x偏移（像素）
                goal_px_offset_from_center_y = predicted_goal_px[:, 1] - map_center_px  # 相对于地图中心的y偏移（像素）
                
                # 将像素偏移转换为米制偏移（考虑y轴方向反转）
                goal_m_offset_from_center_x = goal_px_offset_from_center_x * meters_per_pixel  # x轴（米） [B]
                goal_m_offset_from_center_y = -goal_px_offset_from_center_y * meters_per_pixel  # y轴（米）（反转y轴方向） [B]
                
                # 将米制偏移从地图坐标系（以初始方向为参考）旋转回全局坐标系
                # 需要使用初始朝向的逆旋转矩阵：R^-1 = [[cos(-θ), sin(-θ)], [-sin(-θ), cos(-θ)]] = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
                goal_m_offset_global_x = goal_m_offset_from_center_x * torch.cos(initial_orientation) - \
                                       goal_m_offset_from_center_y * torch.sin(initial_orientation)
                goal_m_offset_global_y = goal_m_offset_from_center_x * torch.sin(initial_orientation) + \
                                       goal_m_offset_from_center_y * torch.cos(initial_orientation)
                
                # 将偏移添加到初始位置，得到目标的全局坐标
                goal_global_position = torch.zeros_like(current_gps, device=current_gps.device)
                goal_global_position[:, 0] = initial_position[:, 0] + goal_m_offset_global_x
                goal_global_position[:, 1] = initial_position[:, 1] + goal_m_offset_global_y
            
            else:
                # 如果没有初始位置信息，我们假设地图坐标系与世界坐标系对齐（初始朝向为北方）
                # 此时只需将像素坐标转换为米制坐标，并加上一个适当的偏移
                goal_px_offset_from_center_x = predicted_goal_px[:, 0] - map_center_px
                goal_px_offset_from_center_y = predicted_goal_px[:, 1] - map_center_px
                
                # 将像素偏移转换为米制偏移（考虑y轴方向反转）
                goal_m_offset_x = goal_px_offset_from_center_x * meters_per_pixel
                goal_m_offset_y = -goal_px_offset_from_center_y * meters_per_pixel  # 反转y轴方向
                
                # 由于缺少初始位置参考，这种情况下计算的全局坐标可能不准确
                # 我们只能假设地图中心对应于某个参考点（如(0,0)或当前GPS减去地图中心对应的米制偏移）
                goal_global_position = torch.zeros_like(current_gps, device=current_gps.device)
                goal_global_position[:, 0] = goal_m_offset_x
                goal_global_position[:, 1] = goal_m_offset_y
            
            # 2. 计算目标相对于agent当前位置的向量（笛卡尔坐标）
            goal_vector_m = goal_global_position - current_gps  # [B, 2] - 目标相对于当前位置的向量（米）
            
            # 3. 将笛卡尔坐标转换为极坐标
            # 计算欧氏距离（米）
            distance = torch.norm(goal_vector_m, dim=1, keepdim=True)  # [B, 1]
            
            # 计算目标相对于agent当前位置的全局角度
            global_angle = torch.atan2(goal_vector_m[:, 1:2], goal_vector_m[:, 0:1])  # [B, 1]
            
            # 计算相对于agent当前朝向的角度
            # 相对角度 = 全局角度 - agent朝向
            relative_angle = global_angle - agent_ori_rad  # [B, 1]
            
            # 将角度规范化到[-π, π]区间
            relative_angle = torch.atan2(torch.sin(relative_angle), torch.cos(relative_angle))
            
            # 组合成最终的极坐标格式（与pointgoal传感器一致）
            # 极坐标系的是为了输入给RNN，而笛卡尔坐标系的是为了计算损失
            predicted_goal = torch.cat([distance, relative_angle], dim=1)  # [B, 2] - [距离(米), 相对角度(弧度)]
            
            # 预测损失计算（如果有真实目标）
            self.goal_prediction_loss = None
            if self._use_pointnav and 'pointgoal' in observations:
                # 使用PointNav数据作为真实值进行训练
                true_goal = observations['pointgoal']  # 这是相对极坐标 [distance, angle]
                
                # 将预测和真实的极坐标都转换为笛卡尔坐标计算损失
                # 这样可以避免角度在-π和π附近时的收敛问题
                
                # 将真实极坐标转换为笛卡尔坐标
                true_distance = true_goal[:, 0:1]  # 距离
                true_angle = true_goal[:, 1:2]  # 角度
                
                # 检查NaN
                if torch.isnan(true_distance).any() or torch.isnan(true_angle).any():
                    print(f"[WARNING] 检测到真实目标中的NaN值: distance={true_distance}, angle={true_angle}")
                    # 替换NaN值
                    true_distance = torch.where(torch.isnan(true_distance), torch.tensor(1.0, device=true_distance.device), true_distance)
                    true_angle = torch.where(torch.isnan(true_angle), torch.tensor(0.0, device=true_angle.device), true_angle)
                    
                # 将真实极坐标转换为笛卡尔坐标
                true_x = true_distance * torch.cos(true_angle)
                true_y = true_distance * torch.sin(true_angle)
                true_cartesian = torch.cat([true_x, true_y], dim=1)
                
                # 从原始预测直接计算笛卡尔坐标（保留梯度流）
                # 应用裁剪以增强数值稳定性
                pred_distance = predicted_goal[:, 0:1]
                pred_angle = predicted_goal[:, 1:2]
                
                # 检查NaN
                if torch.isnan(pred_distance).any() or torch.isnan(pred_angle).any():
                    print(f"[WARNING] 检测到预测目标中的NaN值: distance={pred_distance}, angle={pred_angle}")
                    # 替换NaN值
                    pred_distance = torch.where(torch.isnan(pred_distance), torch.tensor(1.0, device=pred_distance.device), pred_distance)
                    pred_angle = torch.where(torch.isnan(pred_angle), torch.tensor(0.0, device=pred_angle.device), pred_angle)
                
                pred_x = pred_distance * torch.cos(pred_angle)
                pred_y = pred_distance * torch.sin(pred_angle)
                pred_cartesian = torch.cat([pred_x, pred_y], dim=1)
                
                # 设置目标格式（笛卡尔坐标）
                if hasattr(self.goal_predictor, '_is_polar_target'):
                    self.goal_predictor._is_polar_target = False
                
                # 检查是否有NaN值
                if torch.isnan(pred_cartesian).any() or torch.isnan(true_cartesian).any():
                    print(f"[ERROR] 在计算损失前检测到NaN: pred_cart={pred_cartesian}, true_cart={true_cartesian}")
                    # 安全替换为合理值避免传播NaN
                    pred_cartesian = torch.where(torch.isnan(pred_cartesian), torch.tensor(0.0, device=pred_cartesian.device), pred_cartesian)
                    true_cartesian = torch.where(torch.isnan(true_cartesian), torch.tensor(0.0, device=true_cartesian.device), true_cartesian)
                
                # 为了确保梯度流正确传递到目标预测器，我们使用非分离的原始预测值
                self.goal_prediction_loss = self.goal_predictor.compute_loss(pred_cartesian, true_cartesian)
                
                # 在调试中一直记录这个值，确保它不为空
                if torch.isnan(self.goal_prediction_loss):
                    print("[ERROR] 损失计算结果为NaN，将替换为一个小的常数值避免传播")
                    self.goal_prediction_loss = torch.tensor(0.1, device=self.goal_prediction_loss.device, requires_grad=True)
                
                print(f"[POLICY DEBUG] 目标预测损失值: {self.goal_prediction_loss.item():.6f}, requires_grad: {self.goal_prediction_loss.requires_grad}")
                
            # 使用预测的目标替代或补充真实目标
            x.append(predicted_goal)
        # 如果不预测目标但有PointNav数据，则直接使用
        elif self._use_pointnav and 'pointgoal' in observations:
            # 直接使用pointgoal向量(极坐标形式[distance, angle])
            pointgoal_embedding = observations['pointgoal']
            x.append(pointgoal_embedding)
            
        # 添加手绘地图特征
        x.append(map_result)
        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        if torch.isnan(x2).any().item():
            for key in observations:
                print(key, torch.isnan(observations[key]).any().item())
            print('rnn_old', torch.isnan(rnn_hidden_states).any().item())
            print('rnn_new', torch.isnan(rnn_hidden_states1).any().item())
        #     print('mask', torch.isnan(masks).any().item())
        #     assert True
        return x2, rnn_hidden_states1
