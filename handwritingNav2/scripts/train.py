#!/usr/bin/env python3
"""
训练脚本 - HandWriting Navigation

支持PPO和扩散策略两种训练方式，支持多GPU分布式训练
"""

import argparse
import logging
import os
import sys

# 设置环境变量以解决 EGL 渲染问题（必须在导入其他库之前设置）
os.environ['MAGNUM_DEVICE'] = '0'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # 使用软件渲染作为后备
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['GLOG_minloglevel'] = '2'

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 添加自定义 habitat-lab 到 Python 路径（确保使用项目内的版本）
HABITAT_LAB_PATH = os.path.join(PROJECT_ROOT, 'habitat-lab')
sys.path.insert(0, HABITAT_LAB_PATH)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
# import tensorflow as tf  # 不需要 tensorflow，代码使用 PyTorch

from modeling.common.baseline_registry import baseline_registry
from modeling.config.default import get_config
from modeling.ppo.ppo_trainer import PPOTrainer
from modeling.diffusion_policy.habitat_diffusion_trainer import HabitatDiffusionTrainer
# import habitat_sim.logger as logger 
# logger.setLevel(logger.DEBUG)
# def find_best_ckpt_idx(event_dir_path, min_step=-1, max_step=10000):
#     events = os.listdir(event_dir_path)

#     max_value = 0
#     max_index = -1
#     for event in events:
#         if "events" not in event:
#             continue
#         iterator = tf.compat.v1.train.summary_iterator(os.path.join(event_dir_path, event))
#         for e in iterator:
#             if len(e.summary.value) == 0:
#                 continue
#             # 查找环境SPL指标
#             if 'Environment/SPL' not in e.summary.value[0].tag:
#                 continue
#             if not min_step <= e.step <= max_step:
#                 continue
#             if len(e.summary.value) > 0 and e.summary.value[0].simple_value > max_value:
#                 max_value = e.summary.value[0].simple_value
#                 max_index = e.step

#     if max_index == -1:
#         print('No max index is found in {}'.format(event_dir_path))
#     else:
#         print('The best index in {} is {}'.format(event_dir_path, max_index))

#     return max_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        # required=True,
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
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--eval-best",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for training (default: use all available)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')",
    )
    args = parser.parse_args()
    
    # 如果未指定配置文件，使用默认配置
    if args.exp_config is None:
        if args.run_type == 'train':
            args.exp_config = os.path.join(PROJECT_ROOT, 'modeling/config/train_diffusion_hwnav.yaml')
        else:
            args.exp_config = os.path.join(PROJECT_ROOT, 'modeling/config/val_hwnav.yaml')
        print(f"使用默认配置文件: {args.exp_config}")
    
    # if args.eval_best:
    #     best_ckpt_idx = find_best_ckpt_idx(os.path.join(args.model_dir, 'tb'))
    #     best_ckpt_path = os.path.join(args.model_dir, 'data', f'ckpt.{best_ckpt_idx}.pth')
    #     print(f'Evaluating the best checkpoint: {best_ckpt_path}')
    #     args.opts += ['EVAL_CKPT_PATH_DIR', best_ckpt_path]

    # run exp
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)
    
    # 设置 GPU 配置
    # 优先级: 命令行参数 > 配置文件 > 默认使用所有GPU
    gpu_ids_from_config = getattr(config, 'GPU_IDS', None)
    num_gpus_from_config = getattr(config, 'NUM_GPUS', None)
    
    if args.gpu_ids is not None:
        # 1. 命令行指定了GPU ID (最高优先级)
        gpu_ids = args.gpu_ids
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        num_gpus = len(gpu_ids.split(','))
        print(f"使用命令行指定的GPU: {gpu_ids} (共{num_gpus}个GPU)")
    elif args.num_gpus is not None:
        # 2. 命令行指定了GPU数量
        num_gpus = args.num_gpus
        gpu_ids = ','.join(map(str, range(num_gpus)))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"使用命令行指定的前{num_gpus}个GPU: {gpu_ids}")
    elif gpu_ids_from_config is not None:
        # 3. 配置文件指定了GPU ID
        # 处理配置文件中的GPU_IDS，可能是字符串、列表或元组
        if isinstance(gpu_ids_from_config, (list, tuple)):
            gpu_ids = ','.join(map(str, gpu_ids_from_config))
        else:
            gpu_ids = str(gpu_ids_from_config)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        num_gpus = len(gpu_ids.split(','))
        print(f"使用配置文件指定的GPU: {gpu_ids} (共{num_gpus}个GPU)")
    elif num_gpus_from_config is not None:
        # 4. 配置文件指定了GPU数量
        num_gpus = int(num_gpus_from_config)
        gpu_ids = ','.join(map(str, range(num_gpus)))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"使用配置文件指定的前{num_gpus}个GPU: {gpu_ids}")
    else:
        # 5. 默认使用所有可用的GPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_ids = ','.join(map(str, range(num_gpus)))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            print(f"使用所有可用GPU: {gpu_ids} (共{num_gpus}个GPU)")
        else:
            num_gpus = 0
            print("警告: 未检测到可用GPU，将使用CPU")
    
    # 根据配置选择训练器
    if hasattr(config, 'USE_DIFFUSION_POLICY') and config.USE_DIFFUSION_POLICY:
        trainer = HabitatDiffusionTrainer(config, local_rank=-1)  # -1表示不使用DDP
        print("使用Habitat兼容扩散策略训练器（DataParallel模式）")
    else:
        trainer = PPOTrainer(config)
        print("使用PPO训练器")
    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    if args.run_type == "train":
        # 如果指定了resume-from参数，则从该检查点继续训练
        if args.resume_from is not None:
            print(f"Resuming training from checkpoint: {args.resume_from}")
            trainer.train(checkpoint_path=args.resume_from)
        # 如果指定了eval-best参数，则从最佳检查点继续训练
        # elif args.eval_best and best_ckpt_path is not None:
        #     print(f"Resuming training from best checkpoint: {best_ckpt_path}")
        #     trainer.train(checkpoint_path=best_ckpt_path)
        # 否则从头开始训练
        else:
            trainer.train()
    elif args.run_type == "eval":
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, config.USE_LAST_CKPT)


if __name__ == "__main__":
    main()
