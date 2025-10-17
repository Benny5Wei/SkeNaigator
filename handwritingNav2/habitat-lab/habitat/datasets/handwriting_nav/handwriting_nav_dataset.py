# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import logging
from typing import List, Optional

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationGoal,
    ShortestPathPoint,
)
from habitat.tasks.nav.handwriting_nav_task import HWNavEpisode


ALL_SCENES_MASK = "*"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"


@registry.register_dataset(name="HandWritingNav")
class HandWritingNavDataset(Dataset):
    r"""Class inherited from Dataset that loads HandWriting Nav dataset.
    """

    episodes: List[HWNavEpisode]

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        r"""Return list of scene directories in handwriting_instr folder.
        """
        assert HandWritingNavDataset.check_config_paths_exist(config), \
            (config.DATA_PATH.format(split=config.SPLIT), config.SCENES_DIR)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(split=config.SPLIT)
        )
        
        return HandWritingNavDataset._get_scenes_from_folder(dataset_dir)

    @staticmethod
    def _get_scenes_from_folder(dataset_dir):
        scenes = []
        # 直接从handwriting_instr目录获取可用场景
        handwriting_dir = os.path.join(dataset_dir, 'handwriting_instr')
        if os.path.exists(handwriting_dir):
            for entry in os.listdir(handwriting_dir):
                entry_path = os.path.join(handwriting_dir, entry)
                if os.path.isdir(entry_path):
                    scenes.append(entry)
            logging.info(f"Found {len(scenes)} scene directories in {handwriting_dir}")
        else:
            logging.warning(f"Directory not found: {handwriting_dir}")
        
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self._config = config

        if config is None:
            return

        # 只加载主数据文件
        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=datasetfile_path)
        
        logging.info(f"Loaded {len(self.episodes)} episodes from {datasetfile_path}")
        
        # 可以获取场景列表供其他功能使用，但不再尝试加载场景文件
        if config.CONTENT_SCENES and ALL_SCENES_MASK in config.CONTENT_SCENES:
            dataset_dir = os.path.dirname(datasetfile_path)
            scenes = self._get_scenes_from_folder(dataset_dir)
            # 如果需要使用场景列表，可以在这里进行其他操作

    def filter_by_ids(self, scene_ids):
        episodes_to_keep = list()

        for episode in self.episodes:
            for scene_id in scene_ids:
                scene, ep_id = scene_id.split(',')
                if scene in episode.scene_id and ep_id == episode.episode_id:
                    episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    # filter by scenes for data collection
    def filter_by_scenes(self, scene):
        episodes_to_keep = list()

        for episode in self.episodes:
            episode_scene = episode.scene_id.split("/")[3]
            if scene == episode_scene:
                episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, scene_filename: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode["start_room"] = None
            #episode["instruction"] = None
            episode = HWNavEpisode(**episode)
            # a temporal workaround to set scene_dataset_config attribute
            episode.scene_dataset_config = self._config.SCENES_DIR.split('/')[-1]

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)

            if episode.reference_path is not None:
                for p_index, point in enumerate(episode.reference_path):
                    new_point = {
                        "action": None,
                        "rotation": None,
                        "position": point,
                    }

                    episode.reference_path[p_index] = ShortestPathPoint(**new_point)

            self.episodes.append(episode)
