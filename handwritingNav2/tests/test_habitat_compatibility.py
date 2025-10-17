#!/usr/bin/env python3

"""
测试扩散策略与Habitat环境的兼容性
"""

import sys
import os
import torch
import numpy as np
from gym import spaces

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_habitat_observation_format():
    """测试Habitat观察格式兼容性"""
    print("测试Habitat观察格式兼容性...")
    
    try:
        from diffusion_policy.diffusion_nav_policy import DiffusionNavPolicy
        
        # 创建模拟的Habitat观察空间
        obs_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=1, shape=(480, 640, 1), dtype=np.float32),
            'handwriting_instr': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            'pointgoal': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            'gps': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            'compass': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        })
        
        action_space = spaces.Discrete(4)  # Habitat的离散动作空间
        
        # 创建策略
        policy = DiffusionNavPolicy(
            observation_space=obs_space,
            action_space=action_space,
            goal_sensor_uuid='handwriting_instr',
            hidden_size=256,
            horizon=8,
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
            predict_goal=False,
            obs_as_global_cond=True
        )
        print("✓ 策略创建成功")
        
        # 创建模拟的Habitat观察数据
        batch_size = 2
        habitat_obs = {
            'rgb': np.random.randint(0, 255, (batch_size, 480, 640, 3), dtype=np.uint8),
            'depth': np.random.rand(batch_size, 480, 640, 1).astype(np.float32),
            'handwriting_instr': np.random.randint(0, 255, (batch_size, 480, 640, 3), dtype=np.uint8),
            'pointgoal': np.random.randn(batch_size, 2).astype(np.float32),
            'gps': np.random.randn(batch_size, 2).astype(np.float32),
            'compass': np.random.randn(batch_size, 1).astype(np.float32),
        }
        
        # 测试观察编码
        with torch.no_grad():
            encoded_obs = policy.encode_observations(habitat_obs)
            print(f"✓ 观察编码成功，输出形状: {encoded_obs.shape}")
            
        # 测试动作预测
        obs_dict = {'obs': encoded_obs.unsqueeze(1)}  # 添加时间维度
        with torch.no_grad():
            action_result = policy.predict_action(obs_dict)
            print(f"✓ 动作预测成功，动作形状: {action_result['action'].shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Habitat观察格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_space_compatibility():
    """测试动作空间兼容性"""
    print("\n测试动作空间兼容性...")
    
    try:
        from diffusion_policy.diffusion_nav_policy import DiffusionNavPolicy
        
        # 测试离散动作空间
        action_space = spaces.Discrete(4)
        obs_space = spaces.Dict({
            'handwriting_instr': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
        })
        
        policy = DiffusionNavPolicy(
            observation_space=obs_space,
            action_space=action_space,
            goal_sensor_uuid='handwriting_instr',
            hidden_size=128,
            horizon=4,
            n_action_steps=2,
            n_obs_steps=2,
            obs_dim=128,
            action_dim=4,
            num_inference_steps=5,
            extra_rgb=False,
            extra_depth=False,
            slam=False,
            use_vae=False,
            use_pointnav=False,
            predict_goal=False,
            obs_as_global_cond=True
        )
        
        print("✓ 离散动作空间策略创建成功")
        
        # 测试动作嵌入
        test_actions = torch.tensor([0, 1, 2, 3])
        embedded_actions = policy.action_embedding(test_actions)
        print(f"✓ 动作嵌入成功，形状: {embedded_actions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 动作空间兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_compatibility():
    """测试训练兼容性"""
    print("\n测试训练兼容性...")
    
    try:
        from diffusion_policy.habitat_diffusion_trainer import HabitatDiffusionTrainer
        from config.default import get_config
        
        # 创建模拟配置
        config = get_config('config/train_diffusion_hwnav.yaml')
        config.NUM_UPDATES = 1  # 只测试1个更新
        config.LOG_INTERVAL = 1
        config.CHECKPOINT_INTERVAL = 1
        
        # 创建训练器
        trainer = HabitatDiffusionTrainer(config)
        print("✓ Habitat扩散训练器创建成功")
        
        # 测试配置解析
        print(f"✓ 配置解析成功，使用扩散策略: {hasattr(config, 'USE_DIFFUSION_POLICY')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Habitat扩散策略兼容性测试")
    print("=" * 60)
    
    # 测试观察格式兼容性
    obs_success = test_habitat_observation_format()
    
    # 测试动作空间兼容性
    action_success = test_action_space_compatibility()
    
    # 测试训练兼容性
    training_success = test_training_compatibility()
    
    print("\n" + "=" * 60)
    if obs_success and action_success and training_success:
        print("✓ 所有兼容性测试通过！扩散策略与Habitat环境完全兼容。")
        print("\n下一步:")
        print("1. 运行训练: python run.py --run-type train --exp-config config/train_diffusion_hwnav.yaml")
        print("2. 检查日志确保训练正常进行")
    else:
        print("✗ 部分兼容性测试失败，请检查错误信息。")
        print("\n建议:")
        print("1. 检查Habitat环境是否正确安装")
        print("2. 确保所有依赖库已安装")
        print("3. 检查配置文件路径是否正确")
    print("=" * 60)

if __name__ == "__main__":
    main()


