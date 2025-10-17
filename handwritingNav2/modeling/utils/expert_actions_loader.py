import os
import numpy as np
import json
import logging

class ExpertActionsLoader:
    """
    加载预计算的专家动作数据集
    支持两种格式:
    1. 多个JSON文件: 每个episode一个文件，命名为{episode_id}.json
    2. 单个JSON文件: 包含所有episodes的专家动作的单个文件
    """
    def __init__(self, expert_actions_path):
        """
        初始化加载器
        Args:
            expert_actions_path: 存储专家动作的路径（文件或目录）
        """
        self.expert_actions_path = expert_actions_path
        self.expert_actions_cache = {}
        self._load_expert_actions()
    
    def _load_expert_actions(self):
        """
        加载专家动作数据
        支持两种格式:
        1. 目录模式: 每个episode一个独立的JSON文件
        2. 文件模式: 单个JSON文件包含所有episodes的专家动作
        """
        if not os.path.exists(self.expert_actions_path):
            logging.warning(f"专家动作路径不存在: {self.expert_actions_path}")
            return
        
        try:
            # 判断是目录还是文件
            if os.path.isdir(self.expert_actions_path):
                self._load_from_directory()
            else:
                self._load_from_file()
                
            logging.info(f"成功加载 {len(self.expert_actions_cache)} 个episode的专家动作")
        except Exception as e:
            logging.error(f"加载专家动作失败: {e}")
    
    def _load_from_directory(self):
        """
        从目录加载多个JSON文件，每个episode一个文件
        文件命名格式: {episode_id}.json
        """
        files = os.listdir(self.expert_actions_path)
        loaded_count = 0
        
        for file in files:
            if file.endswith('.json'):
                episode_id = file.split('.')[0]
                file_path = os.path.join(self.expert_actions_path, file)
                try:
                    with open(file_path, 'r') as f:
                        actions = json.load(f)
                    self.expert_actions_cache[episode_id] = actions
                    loaded_count += 1
                except Exception as e:
                    logging.error(f"加载文件 {file} 失败: {e}")
        
        logging.info(f"从目录模式加载了 {loaded_count} 个episode的专家动作")
    
    def _load_from_file(self):
        """
        从单个JSON文件加载所有episodes的专家动作
        预期的文件格式:
        {
            "episodes": [
                {
                    "episode_id": 123,
                    ... (其他episode信息) ...
                    "expert_actions": [0, 1, 2, 3, ...] 或 {"actions": [0, 1, 2, ...]}
                },
                ...
            ]
        }
        """
        try:
            with open(self.expert_actions_path, 'r') as f:
                data = json.load(f)
            
            # 处理数据格式
            if 'episodes' in data:
                # 标准格式，提取每个episode的专家动作
                for episode in data['episodes']:
                    if 'episode_id' in episode:
                        episode_id = str(episode['episode_id'])
                        
                        # 检查不同字段名的专家动作
                        if 'expert_actions' in episode:
                            self.expert_actions_cache[episode_id] = episode['expert_actions']
                        elif 'actions' in episode:
                            self.expert_actions_cache[episode_id] = episode['actions']
                        else:
                            # 如果没有专家动作字段，记录警告
                            logging.warning(f"Episode {episode_id} 缺少专家动作数据")
            else:
                # 尝试其他可能的格式
                # 假设数据是 {episode_id: actions} 格式
                for episode_id, actions in data.items():
                    self.expert_actions_cache[str(episode_id)] = actions
            
            logging.info(f"从文件模式加载了 {len(self.expert_actions_cache)} 个episode的专家动作")
        except Exception as e:
            logging.error(f"从文件加载专家动作失败: {e}")
    
    def get_expert_actions(self, episode_id):
        """
        获取指定episode的专家动作序列
        Args:
            episode_id: episode ID
        Returns:
            专家动作列表，如果不存在则返回None
        """
        return self.expert_actions_cache.get(str(episode_id), None)
    
    def get_expert_action(self, episode_id, step_idx):
        """
        获取指定episode和步骤的专家动作
        Args:
            episode_id: episode ID
            step_idx: 步骤索引
        Returns:
            专家动作，如果不存在则返回None
        """
        actions = self.get_expert_actions(episode_id)
        if actions is None or step_idx >= len(actions):
            return None
        return actions[step_idx]
    
    def get_available_episodes(self):
        """
        获取所有可用的episode ID
        Returns:
            所有有专家动作的episode ID列表
        """
        return list(self.expert_actions_cache.keys())
