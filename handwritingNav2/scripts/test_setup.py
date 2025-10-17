#!/usr/bin/env python3
"""
测试脚本 - 验证环境和配置是否正确
"""

import os
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 添加自定义 habitat-lab 到 Python 路径
HABITAT_LAB_PATH = os.path.join(PROJECT_ROOT, 'habitat-lab')
sys.path.insert(0, HABITAT_LAB_PATH)

def test_imports():
    """测试必要的模块是否可以导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False
    
    try:
        import habitat
        print(f"✓ Habitat 版本: {habitat.__version__}")
    except ImportError as e:
        print(f"✗ Habitat 导入失败: {e}")
        return False
    
    try:
        import habitat_sim
        print(f"✓ Habitat-Sim 版本: {habitat_sim.__version__}")
    except ImportError as e:
        print(f"✗ Habitat-Sim 导入失败: {e}")
        return False
    
    try:
        import diffusers
        print(f"✓ Diffusers 版本: {diffusers.__version__}")
    except ImportError as e:
        print(f"✗ Diffusers 导入失败: {e}")
        return False
    
    try:
        import einops
        print(f"✓ Einops 已安装")
    except ImportError as e:
        print(f"✗ Einops 导入失败: {e}")
        return False
    
    print()
    return True


def test_project_structure():
    """测试项目结构是否完整"""
    print("=" * 60)
    print("测试项目结构...")
    print("=" * 60)
    
    required_dirs = [
        'modeling',
        'modeling/config',
        'modeling/ppo',
        'modeling/diffusion_policy',
        'habitat-lab',
        'utils_fmm',
        'scripts',
        'docs',
    ]
    
    required_files = [
        'modeling/config/train_diffusion_hwnav.yaml',
        'modeling/config/train_hwnav.yaml',
        'modeling/config/hwnav_base.yaml',
        'scripts/train.py',
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = os.path.join(PROJECT_ROOT, dir_path)
        if os.path.isdir(full_path):
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"✗ 目录缺失: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        full_path = os.path.join(PROJECT_ROOT, file_path)
        if os.path.isfile(full_path):
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"✗ 文件缺失: {file_path}")
            all_good = False
    
    print()
    return all_good


def test_data_paths():
    """测试数据路径是否正确"""
    print("=" * 60)
    print("测试数据路径...")
    print("=" * 60)
    
    data_paths = [
        '/mnt_data/skenav2/data/big_train_1.json',
        '/mnt_data/skenav2/data/scene_datasets',
        '/mnt_data/skenav2/data/handwriting_instr',
    ]
    
    all_good = True
    
    for path in data_paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024 * 1024)  # MB
                print(f"✓ 文件存在: {path} ({size:.2f} MB)")
            else:
                print(f"✓ 目录存在: {path}")
        else:
            print(f"✗ 路径不存在: {path}")
            all_good = False
    
    print()
    return all_good


def test_config_loading():
    """测试配置文件是否可以正确加载"""
    print("=" * 60)
    print("测试配置加载...")
    print("=" * 60)
    
    try:
        from modeling.config.default import get_config
        config_path = os.path.join(PROJECT_ROOT, 'modeling/config/train_diffusion_hwnav.yaml')
        config = get_config(config_path, None, None, 'train', False)
        print(f"✓ 配置加载成功")
        print(f"  NUM_PROCESSES: {config.NUM_PROCESSES}")
        print(f"  USE_DIFFUSION_POLICY: {config.USE_DIFFUSION_POLICY}")
        print(f"  NUM_UPDATES: {config.NUM_UPDATES}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_vars():
    """测试环境变量设置"""
    print("=" * 60)
    print("检查环境变量...")
    print("=" * 60)
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'MAGNUM_DEVICE',
        'EGL_PLATFORM',
        'LIBGL_ALWAYS_SOFTWARE',
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        print(f"  {var}: {value}")
    
    print()
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("HandWriting Navigation - 环境测试")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("模块导入", test_imports()))
    results.append(("项目结构", test_project_structure()))
    results.append(("数据路径", test_data_paths()))
    results.append(("配置加载", test_config_loading()))
    results.append(("环境变量", test_environment_vars()))
    
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ 所有测试通过！可以开始训练。")
        print("\n运行训练:")
        print("  cd /mnt_data/skenav2/handwritingNav2/scripts")
        print("  python train.py")
    else:
        print("✗ 某些测试失败，请检查上述错误信息。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

