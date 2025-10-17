#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import os
from collections import defaultdict
from typing import Dict, Optional
from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from habitat.core.logging import logger
import habitat_sim.agent
import json

class Benchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote: bool = False
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths)
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-lab repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        # 定义一个函数，用于将实体打包为grpc可以接受的格式
        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        # 定义一个函数，用于将grpc接受的格式解包为实体
        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        # 定义一个函数，用于获取远程episode是否结束
        def remote_ep_over(stub):
            # 调用stub的episode_over方法，获取SerializedEntity
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            # 返回episode_over的值
            return res_env["episode_over"]

        # 获取环境地址和端口
        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        # 创建grpc通道
        channel = grpc.insecure_channel(env_address_port)
        # 创建stub
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        # 获取远程episode的数量
        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        # 初始化聚合指标
        agg_metrics: Dict = defaultdict(float)

        # 初始化episode计数器
        count_episodes = 0

        # 循环执行episode
        while count_episodes < num_episodes:
            # 重置agent
            agent.reset()
            # 获取环境状态
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            # 循环执行action
            while not remote_ep_over(stub):
                # 获取observation
                obs = res_env["observations"]
                # 执行action
                action = agent.act(obs)

                # 获取环境状态
                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            # 获取指标
            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            # 更新聚合指标
            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            # 更新episode计数器
            count_episodes += 1

        # 计算平均指标
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        # 更新提交
        stub.evalai_update_submission(evaluation_pb2.Package())

        # 返回平均指标
        return avg_metrics

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        # 如果num_episodes为None，则将其设置为环境中的episode数量
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            # 如果num_episodes大于环境中的episode数量，则抛出异常
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        # 如果num_episodes小于等于0，则抛出异常
        assert num_episodes > 0, "num_episodes should be greater than 0"
        print(num_episodes)
        # 初始化一个字典，用于存储每个episode的指标
        agg_metrics: Dict = defaultdict(float)

        # 记录已经完成的episode数量
        count_episodes = 0
        # 存储所有episode的指标
        all_metrics = []
        # 存储所有episode的初始指标
        all_metrics_0 = []
        # 记录成功的episode数量
        count_success = 0
        # 当已经完成的episode数量小于num_episodes时，继续循环
        while count_episodes < num_episodes:
            # 重置环境
            observations = self._env.reset()
            # 重置agent
            agent.reset()
            # 获取环境的初始指标
            metrics = self._env.get_metrics()
            # 将初始指标添加到all_metrics_0中
            all_metrics_0.append(metrics)

            # 当环境没有结束时，继续循环
            while not self._env.episode_over:
                # agent执行动作
                action = agent.act(observations)
                #if count_episodes < 14:#779 802
                #    action = 0
                # 如果agent的总步数等于500，则获取环境的指标
                if agent.total_steps == 500:
                    metrics = self._env.get_metrics()
                # 环境执行动作，并返回新的观测值
                observations = self._env.step(action)

            # 获取环境的指标
            metrics = self._env.get_metrics()
            # 将指标添加到all_metrics中
            all_metrics.append(metrics)
            print(count_episodes, metrics)
            # 如果指标中的success为1，则将count_success加1
            if metrics['success'] == 1:
                count_success += 1
            # 遍历指标的键值对
            for m, v in metrics.items():
                # 如果值是字典类型，则遍历字典的键值对
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        # 将子指标的值添加到agg_metrics中
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    # 将指标的值添加到agg_metrics中
                    agg_metrics[m] += v
            # 将已经完成的episode数量加1
            count_episodes += 1
            # 计算平均指标
            avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
            # 遍历平均指标的键值对，并打印
            for k,v in avg_metrics.items():
                logger.info("{}: {}".format(k, v))
            
            # if metrics['success'] == 1:
            #     with open("output/FBE_PSL_oh_b_v2/f_r_count_s.txt", "a") as file_object:
            #         file_object.write(str(agent.fronter_this_ex) +' '+ str(agent.random_this_ex)+ '\n')
            # else:
            #     with open("output/FBE_PSL_oh_b_v2/f_r_count_f.txt", "a") as file_object:
            #         file_object.write(str(agent.fronter_this_ex) + ' '+str(agent.random_this_ex)+ '\n')
                    
            # 将all_metrics写入文件
            # with open('output_new/FBE_PSL_oh_b/results.txt', 'w') as fp:
            with open(f'{agent.args.result_folder}/results.txt', 'w') as fp:
                for item in all_metrics:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                    
            # 将all_metrics_0写入文件
            with open(f'{agent.args.result_folder}/results_0.txt', 'w') as fp:
                for item in all_metrics_0:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            
            
        # 计算平均指标
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        # 返回平均指标
        return avg_metrics

    def evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)
