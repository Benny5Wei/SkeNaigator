#!/usr/bin/env python3

import argparse
import logging
import os

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import tensorflow as tf
from .common.baseline_registry import baseline_registry
from .config.default import get_config
from .ppo.ppo_trainer import PPOTrainer
from .diffusion_policy.habitat_diffusion_trainer import HabitatDiffusionTrainer
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
        # required=True,
        default='/data/xhj/handwritingNav/modeling/config/train_hwnav.yaml',
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
    args = parser.parse_args()

    # if args.eval_best:
    #     best_ckpt_idx = find_best_ckpt_idx(os.path.join(args.model_dir, 'tb'))
    #     best_ckpt_path = os.path.join(args.model_dir, 'data', f'ckpt.{best_ckpt_idx}.pth')
    #     print(f'Evaluating the best checkpoint: {best_ckpt_path}')
    #     args.opts += ['EVAL_CKPT_PATH_DIR', best_ckpt_path]

    # run exp
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)
    
    # 根据配置选择训练器
    if hasattr(config, 'USE_DIFFUSION_POLICY') and config.USE_DIFFUSION_POLICY:
        trainer = HabitatDiffusionTrainer(config)
        print("使用Habitat兼容扩散策略训练器")
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
        elif args.eval_best and best_ckpt_path is not None:
            print(f"Resuming training from best checkpoint: {best_ckpt_path}")
            trainer.train(checkpoint_path=best_ckpt_path)
        # 否则从头开始训练
        else:
            trainer.train()
    elif args.run_type == "eval":
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, config.USE_LAST_CKPT)


if __name__ == "__main__":
    main()
