#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
import os
import logging
import shutil

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN
import habitat

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0
# 使用相对路径，在运行时会被实际路径替换
_C.BASE_TASK_CONFIG_PATH = "modeling/config/hwnav_base.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "HWNavTrainer"
_C.ENV_NAME = "HandWritingNavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.VISUALIZATION_OPTION = ["top_down_map"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"# 默认值，实际运行时会被命令行输入覆盖
_C.NUM_PROCESSES = 8
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]# 默认值，实际运行时会被命令行输入覆盖
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"# 默认值，实际运行时会被命令行输入覆盖
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.EXTRA_RGB = False
_C.EXTRA_DEPTH = True
_C.SLAM = False
_C.USE_VAE = False
_C.DEBUG = False
_C.USE_LAST_CKPT = False
_C.DISPLAY_RESOLUTION = 128
_C.CONTINUOUS = False
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
#
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.WITH_TIME_PENALTY = True
_C.RL.WITH_DISTANCE_REWARD = True
_C.RL.DISTANCE_REWARD_SCALE = 3.0
_C.RL.TIME_DIFF = False
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
# 专家动作配置
_C.RL.PPO.expert_reward_coef = 0.1    # 专家动作奖励系数 - 当代理动作与专家一致时的额外奖励

# 以下参数保留但已不使用(为保证兼容性)
_C.RL.PPO.use_behavior_cloning = False  # 是否使用行为克隆损失
_C.RL.PPO.behavior_cloning_coef = 0.0  # 行为克隆损失的权重系数
_C.USE_EXPERT_ACTIONS = False  # 是否使用Habitat的专家动作引导

# -----------------------------------------------------------------------------
# DIFFUSION POLICY
# -----------------------------------------------------------------------------
_C.RL.DIFFUSION = CN()
_C.RL.DIFFUSION.horizon = 16  # 动作序列长度
_C.RL.DIFFUSION.n_action_steps = 4  # 实际执行的动作步数
_C.RL.DIFFUSION.n_obs_steps = 3  # 观察步数
_C.RL.DIFFUSION.obs_dim = 512  # 观察特征维度
_C.RL.DIFFUSION.action_dim = 4  # 动作维度
_C.RL.DIFFUSION.num_inference_steps = 20  # 推理步数
_C.RL.DIFFUSION.obs_as_global_cond = True  # 使用全局条件
_C.RL.DIFFUSION.lr = 1e-4  # 学习率
_C.RL.DIFFUSION.weight_decay = 1e-4  # 权重衰减
_C.RL.DIFFUSION.use_linear_lr_decay = True  # 使用线性学习率衰减
_C.RL.DIFFUSION.num_train_timesteps = 1000  # 训练时间步数
_C.RL.DIFFUSION.beta_start = 0.0001  # 噪声调度器起始beta
_C.RL.DIFFUSION.beta_end = 0.02  # 噪声调度器结束beta
_C.RL.DIFFUSION.beta_schedule = "linear"  # 噪声调度器类型
_C.RL.DIFFUSION.prediction_type = "epsilon"  # 预测类型
# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------
_TC = habitat.get_config()
_TC.defrost()
# -----------------------------------------------------------------------------
# HANDWRITINGGOAL_SENSOR
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# HANDWRITINGGOAL_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.HANDWRITINGGOAL_SENSOR= CN()
_TC.TASK.HANDWRITINGGOAL_SENSOR.TYPE = "HandWritingGoalSensor"
_TC.TASK.HANDWRITINGINSTR_SENSOR= CN()
_TC.TASK.HANDWRITINGINSTR_SENSOR.TYPE = "HandWritingInstrSensor"
# -----------------------------------------------------------------------------
# HandWriting Navigation
# -----------------------------------------------------------------------------
_TC.SIMULATOR.GRID_SIZE = 0.5
_TC.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False
_TC.SIMULATOR.VIEW_CHANGE_FPS = 10
_TC.SIMULATOR.SCENE_DATASET = 'replica'
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = True
_TC.SIMULATOR.SCENE_OBSERVATION_DIR = 'data/scene_observations'
_TC.SIMULATOR.STEP_TIME = 1.0
# -----------------------------------------------------------------------------
# DistanceToGoal Measure
# -----------------------------------------------------------------------------
_TC.TASK.DISTANCE_TO_GOAL = CN()
_TC.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_TC.DATASET.VERSION = 'v1'
_TC.DATASET.CONTINUOUS = False


def merge_from_path(config, config_paths):
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
    return config


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    model_dir: Optional[str] = None,
    run_type: Optional[str] = None,
    overwrite: bool = False
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
        model_dir: suffix for output dirs
        run_type: either train or eval
    """
    config = merge_from_path(_C.clone(), config_paths)
    #print(f"BASE_TASK_CONFIG_PATH: {_C.BASE_TASK_CONFIG_PATH}")  # 
    config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)
    #print(f"BASE_TASK_CONFIG_PATH: {_C.BASE_TASK_CONFIG_PATH}")  # 
    # config_name = os.path.basename(config_paths).split('.')[0]
    if model_dir is None:
        model_dir = 'data/models/output'
    #从命令行的输入覆盖掉了上面的默认配置
    config.TENSORBOARD_DIR = os.path.join(model_dir, 'tb')
    config.CHECKPOINT_FOLDER = os.path.join(model_dir, 'data')
    config.VIDEO_DIR = os.path.join(model_dir, 'video_dir')
    config.LOG_FILE = os.path.join(model_dir, 'train.log')
    config.EVAL_CKPT_PATH_DIR = os.path.join(model_dir, 'data')

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    dirs = [config.VIDEO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
    # if run_type == 'train':
    #     # check dirs
    #     if any([os.path.exists(d) for d in dirs]):
    #         for d in dirs:
    #             if os.path.exists(d):
    #                 print('{} exists'.format(d))
    #         if overwrite or input('Output directory already exists! Overwrite the folder? (y/n)') == 'y':
    #             for d in dirs:
    #                 if os.path.exists(d):
    #                     shutil.rmtree(d)

    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV

    #一般不开连续，下面的配置不常用
    if config.CONTINUOUS:
        config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
        config.TASK_CONFIG.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
        config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
        config.TASK_CONFIG.SIMULATOR.STEP_TIME = 0.25
        config.TASK_CONFIG.SIMULATOR.AUDIO.CROSSFADE = True
        config.TASK_CONFIG.DATASET.CONTINUOUS = True
        config.RL.DISTANCE_REWARD_SCALE = 1.0
        assert False

        # config.TASK_CONFIG.SIMULATOR.STEP_TIME = 1.0
        # config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 1.0
        # config.TASK_CONFIG.SIMULATOR.TURN_ANGLE = 90
    else:
        config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = config.TASK_CONFIG.SIMULATOR.GRID_SIZE # forward distance ?
    config.TASK_CONFIG.freeze()
    config.freeze()
    return config


def get_task_config(
        config_paths: Optional[Union[List[str], str]] = None,
        opts: Optional[list] = None
) -> habitat.Config:
    config = _TC.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        # 获取项目根目录（modeling/config/default.py 的上上级目录）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        for config_path in config_paths:
            # 如果是相对路径，则相对于项目根目录解析
            if not os.path.isabs(config_path):
                config_path = os.path.join(project_root, config_path)
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
