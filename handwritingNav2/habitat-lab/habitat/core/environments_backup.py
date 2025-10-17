#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@habitat.registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging
import numpy as np

import habitat
from habitat import Config, Dataset
import math

def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return habitat.registry.get_env(env_name)


@habitat.registry.register_env(name="RearrangeRLEnv")
class RearrangeRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE

    def reset(self):
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward = self.config.TASK.SLACK_REWARD

        reward += current_measure

        if self._episode_success():
            reward += self.config.TASK.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        if self.config.TASK.END_ON_SUCCESS and self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@habitat.registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE

        self._previous_measure: Optional[float] = None

    def reset(self):
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self.config.TASK.SLACK_REWARD - 1.0,
            self.config.TASK.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self.config.TASK.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self.config.TASK.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()



@habitat.registry.register_env(name="HandWritingNavRLEnv")
class HandWritingNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._continuous = config.CONTINUOUS

        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS.SUCCESS_DISTANCE
        
        # 检查是否使用专家动作（从配置中获取）
        self._use_expert_actions = False
        if hasattr(config, 'USE_EXPERT_ACTIONS'):
            self._use_expert_actions = config.USE_EXPERT_ACTIONS
        
        # ShortestPathFollower 实例，用于生成专家动作
        self._path_follower = None
        
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        logging.debug(super().current_episode)

        if self._continuous:
            self._previous_target_distance = self._distance_target()
        else:
            self._previous_target_distance = self.habitat_env.current_episode.info[
                "geodesic_distance"
            ]
            
        # 初始化 ShortestPathFollower（在reset时初始化以确保sim已准备好）
        if self._use_expert_actions:
            try:
                from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
                self._path_follower = ShortestPathFollower(
                    self._env.sim,
                    self._success_distance
                )
            except Exception as e:
                logging.warning(f"无法初始化ShortestPathFollower: {e}")
                self._path_follower = None
                
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            # current_target_distance = self._distance_target()
            # print(current_target_distance)
            current_target_distance = observations["handwriting_goal"][0]
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        assert not math.isnan(reward)

        return reward

    def _distance_target(self):
        return self._env.get_metrics()['distance_to_goal']

    def _episode_success(self):
        if self._env.task.is_stop_called and \
                (self._distance_target() < self._success_distance):
                #  (not self._continuous and self._env.sim.reaching_goal)):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id