#!/usr/bin/env python3

"""
测试扩散策略模型的简单脚本
"""

import torch
import torch.nn as nn
from gym import spaces
import numpy as np

# 模拟观察空间
def create_mock_observation_space():
    """创建模拟的观察空间"""
    obs_space = spaces.Dict({
        'rgb': spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
        'depth': spaces.Box(low=0, high=1, shape=(128, 128, 1), dtype=np.float32),
        'handwriting_goal': spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
        'pointgoal': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
        'gps': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
        'compass': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
    })
    return obs_space

def create_mock_action_space():
    """创建模拟的动作空间"""
    return spaces.Discrete(4)  # 4个动作：前进、左转、右转、停止

def test_diffusion_policy():
    """测试扩散策略模型"""
    print("开始测试扩散策略模型...")
    
    # 创建模拟空间
    obs_space = create_mock_observation_space()
    action_space = create_mock_action_space()
    
    # 导入扩散策略
    try:
        from diffusion_policy.diffusion_nav_policy import DiffusionNavPolicy
        print("✓ 成功导入DiffusionNavPolicy")
    except ImportError as e:
        print(f"✗ 导入DiffusionNavPolicy失败: {e}")
        return False
    
    # 创建策略实例
    try:
        policy = DiffusionNavPolicy(
            observation_space=obs_space,
            action_space=action_space,
            goal_sensor_uuid='handwriting_goal',
            hidden_size=256,  # 使用较小的隐藏层大小进行测试
            horizon=8,  # 使用较短的时间范围
            n_action_steps=2,
            n_obs_steps=2,
            obs_dim=256,
            action_dim=4,
            num_inference_steps=10,
            extra_rgb=True,
            extra_depth=True,
            slam=False,
            use_vae=False,
            use_pointnav=True,
            predict_goal=False,  # 暂时禁用目标预测以简化测试
            obs_as_global_cond=True
        )
        print("✓ 成功创建DiffusionNavPolicy实例")
    except Exception as e:
        print(f"✗ 创建DiffusionNavPolicy失败: {e}")
        return False
    
    # 创建模拟观察数据
    batch_size = 2
    obs_steps = 2
    
    mock_obs = {
        'rgb': torch.randn(batch_size, obs_steps, 128, 128, 3),
        'depth': torch.randn(batch_size, obs_steps, 128, 128, 1),
        'handwriting_goal': torch.randn(batch_size, obs_steps, 128, 128, 3),
        'pointgoal': torch.randn(batch_size, obs_steps, 2),
        'gps': torch.randn(batch_size, obs_steps, 2),
        'compass': torch.randn(batch_size, obs_steps, 1),
    }
    
    # 设置归一化器
    from diffusion_policy.normalizer import LinearNormalizer, Normalizer
    
    # 为观察数据创建归一化器
    obs_data = torch.randn(batch_size, obs_steps, 256)  # 模拟编码后的观察数据
    obs_normalizer = Normalizer.from_data(obs_data)
    policy.normalizer.add_normalizer('obs', obs_normalizer)
    
    # 为动作数据创建归一化器
    action_data = torch.randn(batch_size, 2, 4)  # 模拟动作数据
    action_normalizer = Normalizer.from_data(action_data)
    policy.normalizer.add_normalizer('action', action_normalizer)
    
    print("✓ 成功设置归一化器")
    
    # 测试观察编码
    try:
        with torch.no_grad():
            encoded_obs = policy.encode_observations(mock_obs)
            print(f"✓ 观察编码成功，输出形状: {encoded_obs.shape}")
    except Exception as e:
        print(f"✗ 观察编码失败: {e}")
        return False
    
    # 测试动作预测
    try:
        obs_dict = {'obs': encoded_obs.unsqueeze(1)}  # 添加时间维度
        with torch.no_grad():
            action_result = policy.predict_action(obs_dict)
            print(f"✓ 动作预测成功，动作形状: {action_result['action'].shape}")
    except Exception as e:
        print(f"✗ 动作预测失败: {e}")
        return False
    
    # 测试损失计算
    try:
        batch_data = {
            'obs': encoded_obs.unsqueeze(1),
            'action': torch.randn(batch_size, 2, 4)
        }
        with torch.no_grad():
            loss = policy.compute_loss(batch_data)
            print(f"✓ 损失计算成功，损失值: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")
        return False
    
    print("✓ 所有测试通过！扩散策略模型工作正常。")
    return True

def test_conditional_unet():
    """测试条件UNet模型"""
    print("\n开始测试条件UNet模型...")
    
    try:
        from diffusion_policy.conditional_unet1d import ConditionalUnet1D
        print("✓ 成功导入ConditionalUnet1D")
    except ImportError as e:
        print(f"✗ 导入ConditionalUnet1D失败: {e}")
        return False
    
    try:
        unet = ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=256,
            down_dims=[128, 256],
            diffusion_step_embed_dim=128,
            down_step_sizes=[1, 1],
            kernel_size=3,
            n_groups=4
        )
        print("✓ 成功创建ConditionalUnet1D实例")
    except Exception as e:
        print(f"✗ 创建ConditionalUnet1D失败: {e}")
        return False
    
    # 测试前向传播
    try:
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, 4)
        timesteps = torch.randint(0, 1000, (batch_size,))
        global_cond = torch.randn(batch_size, 256)
        
        with torch.no_grad():
            output = unet(x, timesteps, global_cond)
            print(f"✓ UNet前向传播成功，输出形状: {output.shape}")
    except Exception as e:
        print(f"✗ UNet前向传播失败: {e}")
        return False
    
    print("✓ 条件UNet模型测试通过！")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("扩散策略模型测试")
    print("=" * 50)
    
    # 测试条件UNet
    unet_success = test_conditional_unet()
    
    # 测试扩散策略
    policy_success = test_diffusion_policy()
    
    print("\n" + "=" * 50)
    if unet_success and policy_success:
        print("✓ 所有测试通过！扩散策略模型已准备就绪。")
    else:
        print("✗ 部分测试失败，请检查错误信息。")
    print("=" * 50)


