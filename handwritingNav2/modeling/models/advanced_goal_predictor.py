"""
advanced_goal_predictor.py
-------------------------
A more sophisticated goal–prediction network that follows the design proposed by the
user.  The network works in two stages:
1. Pre-processing (outside the nn.Module):
   • Given an exploration map (occupancy grid) and a hand-drawn sketch map, we first
     sample K = k_rows × k_cols keypoints on a regular grid (default 9 × 9 → 81 points).
   • For every keypoint we cast N equally-spaced rays until they hit an obstacle or the
     map boundary, producing an N-dim feature vector of normalised distances.  We then
     concatenate the (x, y) coordinates of the keypoint to obtain a (N + 2)-d vector.
   • We create two tensors of shape (B, K, N + 2): one for the exploration map and one
     for the hand-drawn sketch.  These are fed to the neural network defined below.

2. Neural network architecture (inside AdvancedGoalPredictor):
   • Two independent Transformer encoders (self-attention) produce contextualised
     features for each keypoint in the exploration map and the sketch respectively.
   • A cross-attention layer (MultiHeadAttention) uses exploration features as Query
     and sketch features as Key/Value, fusing sketch information into every
     exploration keypoint.
   • A lightweight MLP produces a logit for each keypoint and a softmax converts them
     into weights.  We obtain the predicted goal position by a weighted average of the
     keypoint coordinates.
   • Training loss: mean-squared error (MSE) between the predicted coordinates and the
     ground-truth GPS coordinates.

Utilities for keypoint sampling and ray-based feature extraction are also provided so
that the network can be used end-to-end without external dependencies (only PyTorch &
NumPy/Scipy stack).  If `scikit-image` is available we use its Bresenham implementation
for speed; otherwise we fall back to a simple Python loop.
"""
from __future__ import annotations

from typing import Tuple, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# GPU版本
import cv2
# -----------------------------------------------------------------------------
# Utility functions (can be used in the data-loading pipeline)
# -----------------------------------------------------------------------------

def extract_occupancy_grid(slam_rgb, use_gpu=True, device='cuda'):
    """
    从SLAM RGB地图提取占用栅格。
    我们将:
      - 灰色区域 (障碍物) 设置为255
      - 浅绿色区域 (已探索自由空间) 设置为0
      - 白色区域 (未探索) 设置为255
    
    参数:
        slam_rgb: RGB格式的SLAM地图 [H, W, 3], float 0~1
        use_gpu: 是否使用GPU加速
        device: GPU设备名称
        return_tensor: 是否返回PyTorch张量（True）或NumPy数组（False）
        
    返回:
        占用栅格地图:
        - 0: 自由空间
        - 255: 障碍物或未探索区域
    """
    start_time = time.time()
    
    # 检查输入是否已经是torch张量
    if not isinstance(slam_rgb, torch.Tensor):
        # 将numpy数组转换为torch张量并移动到GPU
        slam_tensor = torch.from_numpy(slam_rgb.astype(np.float32)).to(device)
    else:
        # 已经是张量，确保在正确的设备上
        slam_tensor = slam_rgb.to(device)
    
    # 计算灰度图
    gray_map = 0.299 * slam_tensor[:,:,0] + 0.587 * slam_tensor[:,:,1] + 0.114 * slam_tensor[:,:,2]
    
    # 创建掩码 - 全部在GPU上操作
    obstacle_mask = (gray_map > 0.3) & (gray_map < 0.5)
    free_mask = (slam_tensor[:,:,1] > slam_tensor[:,:,0] + 0.3) & (slam_tensor[:,:,1] > slam_tensor[:,:,2] + 0.3)
    unexplored_mask = (slam_tensor[:,:,0] > 0.9) & (slam_tensor[:,:,1] > 0.9) & (slam_tensor[:,:,2] > 0.9)
    
    # 创建占用栅格 - 初始化为零张量
    occupancy_grid = torch.zeros_like(gray_map, device=device)
    
    # 设置障碍物和未探索区域
    occupancy_grid = torch.where(obstacle_mask | unexplored_mask, 
                                torch.tensor(255.0, device=device), 
                                occupancy_grid)
    
    # 设置未知区域为障碍物
    unknown_mask = ~(free_mask | obstacle_mask | unexplored_mask)
    occupancy_grid = torch.where(unknown_mask, 
                                torch.tensor(255.0, device=device), 
                                occupancy_grid)
    
    # 转换为byte格式
    occupancy_grid = occupancy_grid.byte()
    
    end_time = time.time()
    if end_time - start_time > 0.01:
        print(f"[占用栅格提取] 耗时: {end_time - start_time:.4f}秒")
    
    return occupancy_grid


def extract_sketch_occupancy_grid(sketch_rgb, use_gpu=True, device='cuda'):
    """
    从手绘地图提取二维占用栅格地图
    
    手绘地图特点:
    - 蓝色点是起点
    - 紫点是终点
    - 红色是轨迹线
    - 黑色线是地图布局（障碍物/墙壁）
    
    参数:
        sketch_rgb: RGB格式的手绘图 [H, W, 3], float 0~1
        use_gpu: 是否使用GPU加速
        device: GPU设备名称
        
    返回:
        occupancy_grid: 二维占用栅格地图 [H, W]
        - 0: 自由空间
        - 255: 障碍物
        
    注意:
        即使返回张量，也会在CPU上进行形态学处理（用OpenCV）然后再移回GPU
        这是因为PyTorch形态学操作支持不如OpenCV完善
    """
    start_time = time.time()
    
    # 检查输入是否已经是torch张量
    if not isinstance(sketch_rgb, torch.Tensor):
        # 将numpy数组转换为torch张量并移动到GPU
        sketch_tensor = torch.from_numpy(sketch_rgb.astype(np.float32)).to(device)
    else:
        # 已经是张量，确保在正确的设备上
        sketch_tensor = sketch_rgb.to(device)
    
    # 在GPU上计算灰度图
    gray_map = 0.299 * sketch_tensor[:,:,0] + 0.587 * sketch_tensor[:,:,1] + 0.114 * sketch_tensor[:,:,2]
    
    # 在GPU上创建所有掩码
    black_mask = gray_map < 0.2
    red_mask = (sketch_tensor[:,:,0] > 0.7) & (sketch_tensor[:,:,1] < 0.3) & (sketch_tensor[:,:,2] < 0.3)
    blue_mask = (sketch_tensor[:,:,0] < 0.3) & (sketch_tensor[:,:,1] < 0.3) & (sketch_tensor[:,:,2] > 0.7)
    purple_mask = (sketch_tensor[:,:,0] > 0.5) & (sketch_tensor[:,:,1] < 0.3) & (sketch_tensor[:,:,2] > 0.5)
    
    # 定义障碍物掩码 - 黑色线条但排除红色轨迹和起终点
    wall_mask = black_mask & ~red_mask & ~blue_mask & ~purple_mask
    
    # 创建占用栅格地图
    occupancy_grid = torch.zeros_like(gray_map, device=device)
    occupancy_grid = torch.where(wall_mask, torch.tensor(255.0, device=device), occupancy_grid)
    
    # 将张量转回NumPy进行形态学处理
    # 形态学操作在CPU上更容易使用OpenCV
    np_occupancy = occupancy_grid.byte().cpu().numpy()
    kernel = np.ones((2, 2), np.uint8)
    np_occupancy = cv2.dilate(np_occupancy, kernel, iterations=1)
    
    end_time = time.time()
    if end_time - start_time > 0.01:
        print(f"[手绘图占用栅格提取] 耗时: {end_time - start_time:.4f}秒")
    
    return torch.from_numpy(np_occupancy).to(device)
            

def generate_regular_keypoints(
    h: int,
    w: int,
    k_rows: int = 9,
    k_cols: int = 9,
    use_gpu: bool = True,
    device: str = 'cuda',
) -> np.ndarray:
    """Return (K, 2) array of integer pixel coordinates on a regular grid.
    
    参数:
        h, w: 地图高度和宽度
        k_rows, k_cols: 网格的行列数
        use_gpu: 是否使用GPU加速
        device: GPU设备名称
    
    返回:
        关键点坐标数组，形状为(K, 2)
    """

            
    # 使用PyTorch在GPU上生成关键点
    ys = torch.linspace(0, h - 1, k_rows, device=device)
    xs = torch.linspace(0, w - 1, k_cols, device=device)
    
    # 创建网格
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    
    # 扁平化并堆叠坐标
    keypoints = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (K, 2): (x, y)
    
    # 直接返回张量
    return keypoints


def generate_explore_keypoints(
    occ_map: np.ndarray,
    k_rows: int = 9,
    k_cols: int = 9,
    explore_value_threshold: int = 254,
    padding: int = 5,
    use_gpu: bool = True,
    device: str = 'cuda',
) -> np.ndarray:
    """
    在占用地图的*已探索*区域生成集中分布的关键点。
    期望占用地图使用255（或大于explore_value_threshold的值）来表示未知/障碍物。
    我们首先找到值小于阈值（即已探索）的像素的边界框，在所有边上扩大*pading*像素，
    然后在这个框内布置一个规则的k_rows×k_cols网格。
    如果已探索区域太小，不足以容纳所需的网格，我们将退回到在整个地图上布置规则网格
    
    参数:
        use_gpu: 是否使用GPU加速
        device: GPU设备名称
    """
    assert occ_map.ndim == 2, "occ_map must be H×W gray / binary image"
 
    start_time = time.time()
    
    # 检查输入是否是PyTorch张量，如果不是则转换
    if not isinstance(occ_map, torch.Tensor):
        occ_map_tensor = torch.from_numpy(occ_map).to(device)
    else:
        occ_map_tensor = occ_map.to(device)
        
    # 创建掩码
    mask = occ_map_tensor < explore_value_threshold
    
    # 如果没有已探索区域，则在整个地图上生成规则网格
    if not torch.any(mask):
        return generate_regular_keypoints(
            occ_map.shape[0], occ_map.shape[1], 
            k_rows, k_cols, use_gpu=use_gpu, device=device
        )
    
    # 找到掩码中非零元素的索引（已探索区域）
    ys, xs = torch.where(mask)
    
    # 获取边界框
    y0, y1 = ys.min().item(), ys.max().item()
    x0, x1 = xs.min().item(), xs.max().item()
    
    # 添加填充
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(occ_map_tensor.shape[0] - 1, y1 + padding)
    x1 = min(occ_map_tensor.shape[1] - 1, x1 + padding)
    
    # 检查边界框是否足够大
    h_box = y1 - y0 + 1
    w_box = x1 - x0 + 1
    if h_box < k_rows or w_box < k_cols:
        # 探索区域不够大，退回到在整个地图上布置规则网格
        return generate_regular_keypoints(
            occ_map.shape[0], occ_map.shape[1], 
            k_rows, k_cols, use_gpu=use_gpu, device=device
        )
    
    # 在边界框内创建网格
    ys = torch.linspace(float(y0), float(y1), k_rows, device=device)
    xs = torch.linspace(float(x0), float(x1), k_cols, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    
    # 扁平化并堆叠坐标
    keypoints = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (K, 2): (x, y)
    
    # # 转换为numpy数组并确保数据类型正确
    # result = keypoints.cpu().numpy().astype(np.int32)
    
    end_time = time.time()
    #print(f"\n[关键点生成] GPU版本耗时: {end_time - start_time:.4f}秒")
    
    return keypoints


def extract_goal_coordinates(sketch_rgb, use_gpu=True, device='cuda'):
    """
    从手绘地图中提取终点（紫色点）的坐标
    
    参数:
        sketch_rgb: RGB格式的手绘图 [H, W, 3], float 0~1
        use_gpu: 是否使用GPU加速
        device: GPU设备名称
        
    返回:
        goal_coords: 终点坐标 [y, x]，如果没有找到终点则返回None
    """
    start_time = time.time()
    
    # 检查输入是否已经是torch张量
    if not isinstance(sketch_rgb, torch.Tensor):
        # 将numpy数组转换为torch张量并移动到GPU
        sketch_tensor = torch.from_numpy(sketch_rgb.astype(np.float32)).to(device)
    else:
        # 已经是张量，确保在正确的设备上
        sketch_tensor = sketch_rgb.to(device)
    
    # 创建紫色掩码 (终点)
    # 紫色: 高R值, 低G值, 高B值
    purple_mask = (sketch_tensor[:,:,0] > 0.5) & (sketch_tensor[:,:,1] < 0.3) & (sketch_tensor[:,:,2] > 0.5)
    
    # 找到紫色点的坐标
    purple_coords = torch.nonzero(purple_mask)
    
    # 如果没有找到紫色点，返回None
    if len(purple_coords) == 0:
        return None
    
    # 计算紫色区域的中心点作为终点坐标
    goal_y = purple_coords[:, 0].float().mean().item()
    goal_x = purple_coords[:, 1].float().mean().item()
    
    end_time = time.time()
    if end_time - start_time > 0.01:
        print(f"[终点坐标提取] 耗时: {end_time - start_time:.4f}秒")
    
    return (goal_y, goal_x)




def ray_features_pytorch(occ_map, keypoints, angles, max_dist, device='cuda'):
    """使用PyTorch在GPU上计算射线特征 - 优化的并行化GPU实现
    
    参数:
        occ_map: 输入的障碍物地图 (H, W)
        keypoints: 关键点坐标 (K, 2)
        angles: 射线角度列表
        max_dist: 最大距离
        device: 设备（'cuda'或'cpu'）
        
    返回:
        射线特征张量 (K, len(angles))
    """
    
    # 记录开始时间
    start_time = time.time()
    
    # 高效地将数据转换为张量并移动到设备上
    if not isinstance(occ_map, torch.Tensor):
        occ_map_tensor = torch.from_numpy(occ_map.astype(np.float32)).to(device)
    else:
        # 确保类型是float32，避免与后续float32类型的occ_values不匹配
        occ_map_tensor = occ_map.float().to(device)
    
    if not isinstance(keypoints, torch.Tensor):
        keypoints_tensor = torch.from_numpy(keypoints.astype(np.float32)).to(device)
    else:
        keypoints_tensor = keypoints.to(device)
    
    K = keypoints_tensor.shape[0]
    n_rays = len(angles)
    
    # 一次性计算所有角度的方向向量
    angles_tensor = torch.tensor(angles, dtype=torch.float32, device=device)
    dx = torch.cos(angles_tensor)  # [n_rays]
    dy = torch.sin(angles_tensor)  # [n_rays]
    
    # 优化步长和最大步数
    step_size = 3.0  # 增加步长可减少迭代次数
    max_steps = int(max_dist / step_size)
    
    # 创建起点张量
    y0 = keypoints_tensor[:, 1].unsqueeze(1).repeat(1, n_rays)  # [K, n_rays]
    x0 = keypoints_tensor[:, 0].unsqueeze(1).repeat(1, n_rays)  # [K, n_rays]
    
    # 预分配结果张量
    max_distances = torch.zeros((K, n_rays), dtype=torch.float32, device=device)
    hit_mask = torch.zeros((K, n_rays), dtype=torch.bool, device=device)
    
    # 地图大小
    h, w = occ_map_tensor.shape
    
    # 创建所有步骤的位移矩阵 - 这样可以避免在循环中重复计算
    steps = torch.arange(0, max_steps, device=device).float() * step_size  # [max_steps]
    
    # 计算所有步骤的x和y坐标增量
    dx_steps = dx.unsqueeze(0) * steps.unsqueeze(1)  # [max_steps, n_rays]
    dy_steps = dy.unsqueeze(0) * steps.unsqueeze(1)  # [max_steps, n_rays]
    
    # 批处理多个步骤以减少循环迭代次数
    batch_size = 50  # 每次处理的步骤数
    
    for step_start in range(0, max_steps, batch_size):
        step_end = min(step_start + batch_size, max_steps)
        batch_steps = slice(step_start, step_end)
        
        # 为当前批次中的所有步骤计算位置
        x_batched = x0.unsqueeze(2) + dx.unsqueeze(0).unsqueeze(2) * steps[batch_steps].unsqueeze(0).unsqueeze(1)  # [K, n_rays, batch_size]
        y_batched = y0.unsqueeze(2) + dy.unsqueeze(0).unsqueeze(2) * steps[batch_steps].unsqueeze(0).unsqueeze(1)  # [K, n_rays, batch_size]
        
        # 重塑以便于处理
        K_batch = K * n_rays * (step_end - step_start)
        x_flat = x_batched.reshape(-1)  # [K * n_rays * batch_size]
        y_flat = y_batched.reshape(-1)  # [K * n_rays * batch_size]
        
        # 转换为整数并限制在边界内
        x_int = x_flat.long().clamp(0, w-1)
        y_int = y_flat.long().clamp(0, h-1)
        
        # 创建有效边界掩码
        valid_mask_flat = (x_flat >= 0) & (x_flat < w) & (y_flat >= 0) & (y_flat < h)
        
        # 检索占用值
        occ_values = torch.zeros_like(x_flat, dtype=torch.float32)
        valid_indices = torch.nonzero(valid_mask_flat).squeeze(-1)
        if len(valid_indices) > 0:
            occ_values[valid_indices] = occ_map_tensor[y_int[valid_indices], x_int[valid_indices]]
        
        # 重塑回原始形状
        occ_values = occ_values.reshape(K, n_rays, step_end - step_start)
        valid_mask = valid_mask_flat.reshape(K, n_rays, step_end - step_start)
        
        # 检测碰撞
        hit_detected = (occ_values > 0) & valid_mask
        
        # 沿着射线方向压缩 - 对每条射线检测是否有任何点击中障碍物
        any_hit = hit_detected.any(dim=2)  # [K, n_rays]
        
        # 找到第一次碰撞的索引
        first_hit_idx = torch.argmax(hit_detected.float(), dim=2)  # [K, n_rays]
        
        # 计算碰撞距离
        hit_distances = steps[batch_steps][first_hit_idx]  # [K, n_rays]
        
        # 更新最大距离和碰撞掩码
        # 仅更新之前未碰撞但现在碰撞的射线
        new_hits = any_hit & (~hit_mask)
        
        # 更新最大距离
        max_distances = torch.where(new_hits, hit_distances, max_distances)
        
        # 更新碰撞掩码
        hit_mask = hit_mask | any_hit
        
        # 如果所有射线都碰到障碍物，提前退出
        if hit_mask.all():
            break
    
    # 将距离归一化为0-1范围并保持为PyTorch张量
    result = max_distances / max_dist
    
    # 记录结束时间并打印
    end_time = time.time()
    #print(f"\n[射线特征提取-优化版] GPU版本耗时: {end_time - start_time:.4f}秒, 处理{K}个关键点, 每个{n_rays}条射线")
    
    # 返回PyTorch张量，留在原设备上，避免不必要的CPU-GPU转换
    return result

# -----------------------------------------------------------------------------
# Neural network – AdvancedGoalPredictor
# -----------------------------------------------------------------------------
def find_closest_keypoint(keypoints, goal_coords):
    """
    找到距离目标坐标最近的关键点索引
    
    参数:
        keypoints: 关键点坐标张量 [K, 2] (x, y)
        goal_coords: 目标坐标 (y, x)
        
    返回:
        closest_idx: 最近关键点的索引
    """
    if goal_coords is None:
        return None
    
    # 将目标坐标转换为与关键点相同的格式 (x, y)
    goal_y, goal_x = goal_coords
    goal_tensor = torch.tensor([goal_x, goal_y], device=keypoints.device)
    
    # 计算每个关键点到目标的欧氏距离
    distances = torch.norm(keypoints - goal_tensor, dim=1)
    
    # 找到距离最小的关键点索引
    closest_idx = torch.argmin(distances).item()
    
    return closest_idx

class AdvancedGoalPredictor(nn.Module):
    """优化的Transformer-based目标预测器，支持半精度计算加速。

    Parameters
    ----------
    k_points : int
        每个地图的关键点数量(默认25，对应于5×5网格)。
    in_dim : int
        每个关键点输入向量的维度(射线特征+xy)(默认10)。
    d_model : int
        Transformer/attention隐藏层大小。
    nhead : int
        注意力头的数量。
    num_layers : int
        自注意力编码器中的层数。
    dropout : float
        整个网络中的dropout率。
    use_half_precision : bool
        是否使用FP16半精度计算加速训练。
    """

    def __init__(
        self,
        k_points: int = 25,  # 默认5x5=25个点
        in_dim: int = 10,   # 默认8+2=10维
        d_model: int = 128, 
        nhead: int = 4,     # 减少注意力头
        num_layers: int = 1, # 减少层数
        dropout: float = 0.1,
        use_half_precision: bool = False, # 默认使用半精度
    ) -> None:
        super().__init__()
        self._is_polar_target = False
        self.k_points = k_points
        self.use_half_precision = use_half_precision
        
        self.explore_proj = nn.Linear(in_dim, d_model)
        self.sketch_proj = nn.Linear(in_dim, d_model)


        # 优化的Transformer编码层配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            # 添加一些优化参数
            norm_first=True,  # 先规范化再注意力，可以提高训练稳定性
        )
        self.explore_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sketch_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention: using PyTorch's MultiheadAttention (batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # 简化的融合层标准化
        self.fuse_norm = nn.LayerNorm(d_model)
        
        # 简化的MLP预测器
        self.logit_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),  # per-keypoint logit
        )

        # 初始化权重
        self._init_weights()

    # ------------------------------------------------------------------ utility
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        explore_feat: torch.Tensor,  # shape (B, K, in_dim)
        sketch_feat: torch.Tensor,   # shape (B, K, in_dim)
        keypoint_xy: torch.Tensor,   # shape (B, K, 2)  absolute coordinates (x, y)
        goal_coords=None,            # Optional: 目标坐标 (y, x) 或 None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pred_xy, weight) where
        • pred_xy: (B, 2) predicted coordinates
        • weight : (B, K) per-keypoint weights (sum to 1)
        
        优化版本: 使用半精度计算加速，共享编码器，并优化计算流程
                
        参数:
            explore_feat: 探索图特征 [B, K, in_dim]
            sketch_feat: 手绘图特征 [B, K, in_dim]
            keypoint_xy: 关键点坐标 [B, K, 2]
            goal_coords: 可选，目标坐标 (y, x) 或 None  
        """
        # 检查输入形状
        B, K, D_in = explore_feat.shape
        assert K == self.k_points, f"Expected {self.k_points} keypoints, got {K}"

        # 如果提供了目标坐标，为最近的关键点添加特殊token embedding
        if goal_coords is not None:
            # 找到距离目标最近的关键点
            closest_idx = find_closest_keypoint(keypoint_xy[0], goal_coords)
            
            if closest_idx is not None:
                # 为每个批次中的最近点添加特殊token embedding
                for b in range(B):
                    sketch_feat[b, closest_idx] = sketch_feat[b, closest_idx] + self.special_token_embedding
                print(f"添加特殊token到索引 {closest_idx} 的关键点")

        # 如果启用了半精度计算
        if self.use_half_precision and explore_feat.device.type == 'cuda':
            # 转换为半精度
            with torch.cuda.amp.autocast():
                # 共享投影层以减少参数
                explore = self.explore_proj(explore_feat)  # (B, K, d_model)
                sketch = self.sketch_proj(sketch_feat)    # (B, K, d_model)
                
                # 使用编码器对两种特征进行编码，减少计算量
                explore_enc = self.explore_encoder(explore)  # (B, K, d_model)
                sketch_enc = self.sketch_encoder(sketch)   # (B, K, d_model)
                
                # 优化的交叉注意力: explore作为查询，sketch作为键和值
                # 减少内存使用量，使用更紧凑的表示
                fused, _ = self.cross_attn(query=explore_enc, key=sketch_enc, value=sketch_enc)
                # 残差连接和层归一化
                fused = self.fuse_norm(fused + explore_enc)
                
                # 轻量级MLP预测logits -> weights
                logits = self.logit_mlp(fused).squeeze(-1)  # (B, K)
                # 使用softmax计算权重
                weight = F.softmax(logits, dim=-1)  # (B, K)
                
                # 计算坐标的加权平均值
                # 转回全精度进行最终计算，避免精度损失
                weight_fp32 = weight.float()
                keypoint_xy_fp32 = keypoint_xy.float()
                pred_xy = torch.sum(weight_fp32.unsqueeze(-1) * keypoint_xy_fp32, dim=1)  # (B, 2)
        else:
            # 全精度版本 - 使用编码器但不使用半精度
            explore = self.explore_proj(explore_feat)  # (B, K, d_model)
            sketch = self.sketch_proj(sketch_feat)    # (B, K, d_model)
            
            # 使用编码器对两种特征进行编码，减少计算量
            explore_enc = self.explore_encoder(explore)  # (B, K, d_model)
            sketch_enc = self.sketch_encoder(sketch)   # (B, K, d_model)
            
            # 交叉注意力
            fused, _ = self.cross_attn(query=explore_enc, key=sketch_enc, value=sketch_enc)
            # 残差连接和层归一化
            fused = self.fuse_norm(fused + explore_enc)
            
            # 预测logits和权重
            logits = self.logit_mlp(fused).squeeze(-1)  # (B, K)
            weight = F.softmax(logits, dim=-1)         # (B, K)
            
            # 计算坐标的加权平均值
            pred_xy = torch.sum(weight.unsqueeze(-1) * keypoint_xy, dim=1)  # (B, 2)
            
        return pred_xy, weight

    # ------------------------------------------------------------------ loss
    def compute_loss(self, predictions, targets):
        """
        计算预测目标与真实目标之间的损失
        
        predictions: 预测的目标位置 [batch_size, 2] - 笛卡尔坐标 [x, y]
        targets: 输入可能是极坐标 [distance, angle] 或笛卡尔坐标 [x, y]

        """
        # 检查NaN值并报告
        if torch.isnan(predictions).any():
            print(f"[WARNING] 进入compute_loss前预测值中存在NaN: {predictions}")
            # 替换NaN值为0，避免传播
            predictions = torch.nan_to_num(predictions, nan=0.0)

        if torch.isnan(targets).any():
            print(f"[WARNING] 进入compute_loss前目标值中存在NaN: {targets}")
            # 替换NaN值为0，避免传播
            targets = torch.nan_to_num(targets, nan=0.0)

        # 确保使用全精度进行损失计算，提高数值稳定性
        predictions = predictions.float()
        targets = targets.float()

        # 如果目标是极坐标格式，先转换为笛卡尔坐标
        if hasattr(self, '_is_polar_target') and self._is_polar_target:
            # 将极坐标转换为笛卡尔坐标 [x, y]
            distance, angle = targets[:, 0:1], targets[:, 1:2]
            cartesian_targets = torch.cat([
                distance * torch.cos(angle),
                distance * torch.sin(angle)
            ], dim=1)
        else:
            cartesian_targets = targets

        # 直接使用MSE损失函数计算所有坐标分量的损失
        return F.mse_loss(predictions, cartesian_targets)
    
    def set_polar_target(self, is_polar=True):
        """
        设置目标是否为极坐标格式
        """
        self._is_polar_target = is_polar
