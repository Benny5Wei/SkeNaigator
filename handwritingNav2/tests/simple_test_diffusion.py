#!/usr/bin/env python3

"""
简化的扩散策略测试脚本，不依赖外部库
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入"""
    print("测试导入...")
    
    try:
        import torch
        print("✓ PyTorch导入成功")
    except ImportError:
        print("✗ PyTorch导入失败")
        return False
    
    try:
        import torch.nn as nn
        print("✓ torch.nn导入成功")
    except ImportError:
        print("✗ torch.nn导入失败")
        return False
    
    try:
        from diffusion_policy.conditional_unet1d import ConditionalUnet1D
        print("✓ ConditionalUnet1D导入成功")
    except ImportError as e:
        print(f"✗ ConditionalUnet1D导入失败: {e}")
        return False
    
    try:
        from diffusion_policy.normalizer import LinearNormalizer, Normalizer
        print("✓ 归一化器导入成功")
    except ImportError as e:
        print(f"✗ 归一化器导入失败: {e}")
        return False
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        import torch
        from diffusion_policy.conditional_unet1d import ConditionalUnet1D
        from diffusion_policy.normalizer import Normalizer
        
        # 测试UNet创建
        unet = ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=256,
            down_dims=[128, 256],
            diffusion_step_embed_dim=128,
            down_step_sizes=[1, 1],
            kernel_size=3,
            n_groups=4
        )
        print("✓ UNet创建成功")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, 4)
        timesteps = torch.randint(0, 1000, (batch_size,))
        global_cond = torch.randn(batch_size, 256)
        
        with torch.no_grad():
            output = unet(x, timesteps, global_cond)
            print(f"✓ UNet前向传播成功，输出形状: {output.shape}")
        
        # 测试归一化器
        data = torch.randn(10, 4)
        normalizer = Normalizer.from_data(data)
        normalized = normalizer.normalize(data)
        unnormalized = normalizer.unnormalize(normalized)
        
        print(f"✓ 归一化器测试成功，数据形状: {data.shape}")
        print(f"  归一化后形状: {normalized.shape}")
        print(f"  反归一化后形状: {unnormalized.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("扩散策略模型简化测试")
    print("=" * 50)
    
    # 测试导入
    import_success = test_imports()
    
    if not import_success:
        print("\n✗ 导入测试失败，请检查依赖")
        return
    
    # 测试基本功能
    functionality_success = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if import_success and functionality_success:
        print("✓ 所有测试通过！扩散策略核心组件工作正常。")
        print("\n下一步:")
        print("1. 安装完整依赖: pip install diffusers einops")
        print("2. 运行完整测试: python test_diffusion_policy.py")
        print("3. 开始训练: python run.py --run-type train --exp-config config/train_diffusion_hwnav.yaml")
    else:
        print("✗ 部分测试失败，请检查错误信息。")
    print("=" * 50)

if __name__ == "__main__":
    main()


