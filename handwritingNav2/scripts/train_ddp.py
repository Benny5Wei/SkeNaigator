#!/usr/bin/env python3
"""
DDP分布式训练脚本 - HandWriting Navigation

使用方法:
    torchrun --nproc_per_node=4 scripts/train_ddp.py
    或
    python -m torch.distributed.launch --nproc_per_node=4 scripts/train_ddp.py
"""

import argparse
import logging
import os
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 添加自定义 habitat-lab 到 Python 路径
HABITAT_LAB_PATH = os.path.join(PROJECT_ROOT, 'habitat-lab')
sys.path.insert(0, HABITAT_LAB_PATH)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.distributed as dist

from modeling.config.default import get_config
from modeling.diffusion_policy.habitat_diffusion_trainer import HabitatDiffusionTrainer


def init_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('未检测到分布式环境变量，使用单GPU模式')
        return -1
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    print(f'初始化分布式训练: rank={rank}, world_size={world_size}, local_rank={local_rank}')
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=False,
        default=None,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()
    
    # 初始化分布式
    local_rank = init_distributed()
    
    # 如果未指定配置文件，使用默认配置
    if args.exp_config is None:
        if args.run_type == 'train':
            args.exp_config = os.path.join(PROJECT_ROOT, 'modeling/config/train_diffusion_hwnav.yaml')
        else:
            args.exp_config = os.path.join(PROJECT_ROOT, 'modeling/config/val_hwnav.yaml')
        if local_rank <= 0:
            print(f"使用默认配置文件: {args.exp_config}")
    
    # 加载配置
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, False)
    
    # 创建trainer（传入local_rank）
    trainer = HabitatDiffusionTrainer(config, local_rank=local_rank)
    
    torch.set_num_threads(1)
    
    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    
    if args.run_type == "train":
        if args.resume_from is not None:
            if local_rank <= 0:
                print(f"从checkpoint恢复训练: {args.resume_from}")
            trainer.train(checkpoint_path=args.resume_from)
        else:
            trainer.train()
    elif args.run_type == "eval":
        trainer.eval()
    
    # 清理分布式环境
    if local_rank >= 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


