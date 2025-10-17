#!/usr/bin/env python3

import os
import sys
import logging
import numpy as np

import habitat
from habitat.config import Config
from habitat_sim.utils.common import quat_from_angle_axis
from habitat.tasks.nav.handwriting_nav_task import HWNavEpisode
from habitat.core.environments import HandWritingNavRLEnv

from modeling.config.default import get_config

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

# 设置输出所有调试日志
logging.getLogger().setLevel(logging.DEBUG)

def main():
    # 获取配置
    model_dir = '/data/xhj/handwritingNav/exps/20250521_debug'
    os.makedirs(model_dir, exist_ok=True)
    
    config = get_config(None, ['NUM_PROCESSES', '1', 'USE_EXPERT_ACTIONS', 'True'], model_dir, 'train')
    
    # 创建单个环境而不是向量化环境
    logging.info("创建HandWritingNavRLEnv环境")
    try:
        env = HandWritingNavRLEnv(config)
        
        # 重置环境
        logging.info("重置环境")
        obs = env.reset()
        logging.info(f"成功获取初始观察: {list(obs.keys())}")
        
        # 获取info以测试专家动作
        logging.info("获取环境信息")
        info = env.get_info(obs)
        
        # 检查专家动作是否生成成功
        if "expert_action" in info:
            logging.info(f"生成的专家动作: {info['expert_action']}")
        else:
            logging.warning("未找到专家动作")
            
        # 尝试执行几个步骤
        for i in range(5):
            # 使用专家动作（如果有）或默认动作
            expert_action = info.get("expert_action", 0)
            
            # 如果专家动作是数组，选择非零的最高值索引
            if isinstance(expert_action, (list, np.ndarray)):
                # 找到expert_action中值最大的索引
                action = np.argmax(expert_action)
                logging.info(f"专家动作数组: {expert_action}, 选取动作: {action}")
            else:
                action = int(expert_action)
            
            logging.info(f"执行动作: {action}")
            
            # 必须将动作作为关键字参数传递
            obs, reward, done, info = env.step(action=action)
            logging.info(f"步骤 {i+1}: 奖励={reward}, 完成={done}")
            
            if done:
                logging.info("环境完成，重置")
                obs = env.reset()
        
        logging.info("测试完成")
        
    except Exception as e:
        logging.error(f"环境初始化或运行出错: {e}", exc_info=True)
    
    logging.info("调试结束")

if __name__ == "__main__":
    main()
