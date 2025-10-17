import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.utils import Flatten


def conv_output_dim(dimension, padding, dilation, kernel_size, stride
):
    r"""Calculates the output height and width based on the input
    height and width to the convolution layer.

    ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    """
    assert len(dimension) == 2
    out_dimension = []
    for i in range(len(dimension)):
        out_dimension.append(
            int(
                np.floor(
                    (
                            (
                                    dimension[i]
                                    + 2 * padding[i]
                                    - dilation[i] * (kernel_size[i] - 1)
                                    - 1
                            )
                            / stride[i]
                    )
                    + 1
                )
            )
        )
    return tuple(out_dimension)


def layer_init(cnn):
    for layer in cnn:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                layer.weight, nn.init.calculate_gain("relu")
            )
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)


class VisualCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, extra_map = False, goal_id = None, extra_rgb = False, extra_depth = False, slam = False):
        super().__init__()
        # if "rgb" in observation_space.spaces and not extra_rgb:
        if "rgb" in observation_space.spaces and extra_rgb:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces and extra_depth:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        if slam:
            self._n_input_slam = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_slam = 0
        
        if extra_map and goal_id is not None and goal_id in observation_space.spaces:
            self._n_input_map = observation_space.spaces[goal_id].shape[2] # the same shape
            self.goal_id = goal_id
        else:
            self._n_input_map = 0
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]


        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]
        
        # 记录期望的输入尺寸 - 用于训练和推理时的尺寸匹配
        # 68096 = 64 * h * w, 其中经过多次卷积下采后必须得到这个数字
        # 根据错误信息，我们需要确保最终的flattened大小是68096
        self.expected_height = 640  # 训练时使用的高度
        self.expected_width = 480   # 训练时使用的宽度

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        if self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )
        if self._n_input_slam > 0:
            cnn_dims = np.array((512, 512), dtype=np.float32)
        if self._n_input_map > 0:
            cnn_dims = np.array(
                observation_space.spaces[goal_id].shape[:2], dtype=np.float32
            )

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth + self._n_input_slam + self._n_input_map,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                Flatten(),
                nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )

        layer_init(self.cnn)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth + self._n_input_map + self._n_input_slam == 0
    
    def forward(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            
            # 检查尺寸并进行调整以匹配训练时的尺寸
            current_shape = rgb_observations.shape
            if current_shape[2] != self.expected_height or current_shape[3] != self.expected_width:
                # print(f"Resizing RGB observations from {current_shape[2]}x{current_shape[3]} to {self.expected_height}x{self.expected_width}")
                # 确保调整到正确的尺寸，使得最终的flatten向量长度为68096
                rgb_observations = F.interpolate(
                    rgb_observations, 
                    size=(self.expected_height, self.expected_width),
                    mode='bilinear',
                    align_corners=False
                )
            
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            
            # 检查尺寸并进行调整以匹配训练时的尺寸
            current_shape = depth_observations.shape
            if current_shape[2] != self.expected_height or current_shape[3] != self.expected_width:
                # print(f"Resizing depth observations from {current_shape[2]}x{current_shape[3]} to {self.expected_height}x{self.expected_width}")
                # 确保调整到正确的尺寸，使得最终的flatten向量长度为68096
                depth_observations = F.interpolate(
                    depth_observations, 
                    size=(self.expected_height, self.expected_width),
                    mode='bilinear',
                    align_corners=False
                )
            
            cnn_input.append(depth_observations)

        if self._n_input_slam > 0:
            slam_observations = observations["slam"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            slam_observations = slam_observations.permute(0, 3, 1, 2)
            slam_observations = slam_observations / 255.0  # normalize slam
            cnn_input.append(slam_observations)
        
        if self._n_input_map > 0:
            goal_observations = observations[self.goal_id]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            goal_observations = goal_observations.permute(0, 3, 1, 2)
            goal_observations = goal_observations / 255.0  # normalize map
            cnn_input.append(goal_observations)


    
        cnn_input = torch.cat(cnn_input, dim=1)
        
        # 打印输入张量尺寸信息以进行调试
        # print(f"CNN input shape: {cnn_input.shape}")
        
        # 运行模型并返回结果
        return self.cnn(cnn_input)
