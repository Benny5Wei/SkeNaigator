#!/usr/bin/env python3
"""
测试扩散策略数据加载器

用于验证数据集是否正确生成和加载
"""

import os
import sys

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import logging
import numpy as np
import torch

from modeling.diffusion_policy.diffusion_dataset import create_diffusion_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dataloader():
    """测试数据加载器"""
    
    data_dir = "/mnt_data/skenav2/handwritingNav2/data/diffusion_dataset"
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        logger.error("请先运行 generate_diffusion_dataset.py 生成数据")
        return False
    
    # 检查训练集和测试集
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        logger.error(f"训练集目录不存在: {train_dir}")
        return False
    
    logger.info(f"找到数据目录: {data_dir}")
    
    # 创建训练数据加载器
    logger.info("\n=== 创建训练数据加载器 ===")
    try:
        train_loader = create_diffusion_dataloader(
            data_dir=data_dir,
            split='train',
            batch_size=4,
            num_workers=0,  # 使用0避免多进程问题
            shuffle=True,
            horizon=16,
            n_obs_steps=3,
            n_action_steps=4,
            use_rgb=True,
            use_depth=True,
        )
        logger.info(f"✅ 训练数据加载器创建成功")
        logger.info(f"   - Batch数: {len(train_loader)}")
        logger.info(f"   - 总样本数: {len(train_loader.dataset)}")
    except Exception as e:
        logger.error(f"❌ 创建训练数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试加载一个batch
    logger.info("\n=== 测试加载数据 ===")
    try:
        for batch_idx, batch in enumerate(train_loader):
            logger.info(f"成功加载第 {batch_idx} 个batch")
            logger.info(f"观察形状: {batch['obs'].shape}")  # [B, n_obs_steps, C, H, W]
            logger.info(f"动作形状: {batch['action'].shape}")  # [B, n_action_steps]
            logger.info(f"观察数据类型: {batch['obs'].dtype}")
            logger.info(f"动作数据类型: {batch['action'].dtype}")
            logger.info(f"观察值范围: [{batch['obs'].min():.3f}, {batch['obs'].max():.3f}]")
            logger.info(f"动作值范围: [{batch['action'].min()}, {batch['action'].max()}]")
            logger.info(f"第一个样本的动作: {batch['action'][0]}")
            
            # 只测试第一个batch
            break
        
        logger.info("✅ 数据加载测试通过")
    except Exception as e:
        logger.error(f"❌ 加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建测试数据加载器
    if os.path.exists(test_dir):
        logger.info("\n=== 创建测试数据加载器 ===")
        try:
            test_loader = create_diffusion_dataloader(
                data_dir=data_dir,
                split='test',
                batch_size=4,
                num_workers=0,
                shuffle=False,
                horizon=16,
                n_obs_steps=3,
                n_action_steps=4,
            )
            logger.info(f"✅ 测试数据加载器创建成功")
            logger.info(f"   - Batch数: {len(test_loader)}")
            logger.info(f"   - 总样本数: {len(test_loader.dataset)}")
        except Exception as e:
            logger.error(f"❌ 创建测试数据加载器失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    logger.info("\n" + "="*60)
    logger.info("✅ 所有测试通过！数据加载器工作正常")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    success = test_dataloader()
    sys.exit(0 if success else 1)






