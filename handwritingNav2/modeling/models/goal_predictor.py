import torch
import torch.nn as nn
import torch.nn.functional as F

class GoalPredictor(nn.Module):
    """
    模型用于从手绘地图预测目标位置（笛卡尔坐标形式）
    输入：手绘地图的CNN特征向量
    输出：预测的目标位置（x坐标和y坐标）
    """
    def __init__(self, input_size=512, hidden_size=256):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 2)  # 输出2维：[x, y]
        
        # 初始化权重
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, x):
        """
        x: 手绘地图的CNN特征向量 [batch_size, input_size]
        返回：预测的目标位置 [batch_size, 2] - 笛卡尔坐标 [x, y]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        goal_pred = self.fc3(x)  # 直接输出笛卡尔坐标 [x, y]
        
        return goal_pred
    
    def compute_loss(self, predictions, targets):
        """
        计算预测目标与真实目标之间的损失
        
        predictions: 预测的目标位置 [batch_size, 2] - 笛卡尔坐标 [x, y]
        targets: 输入可能是极坐标 [distance, angle] 或笛卡尔坐标 [x, y]
        """
        # 如果目标是极坐标格式，先转换为笛卡尔坐标
        # 假设targets格式为[distance, angle]
        if hasattr(self, '_is_polar_target') and self._is_polar_target:
            # 将极坐标转换为笛卡尔坐标
            distance = targets[:, 0:1]  # 距离
            angle = targets[:, 1:2]  # 角度
            
            # 转换为笛卡尔坐标 [x, y]
            target_x = distance * torch.cos(angle)
            target_y = distance * torch.sin(angle)
            
            # 组合成笛卡尔坐标
            cartesian_targets = torch.cat([target_x, target_y], dim=1)
        else:
            # 已经是笛卡尔坐标格式
            cartesian_targets = targets
        
        # 使用均方误差计算x和y坐标的损失
        x_loss = F.mse_loss(predictions[:, 0], cartesian_targets[:, 0])
        y_loss = F.mse_loss(predictions[:, 1], cartesian_targets[:, 1])
        
        # 总损失
        total_loss = x_loss + y_loss
        
        return total_loss
    
    def set_polar_target(self, is_polar=True):
        """
        设置目标是否为极坐标格式
        """
        self._is_polar_target = is_polar
