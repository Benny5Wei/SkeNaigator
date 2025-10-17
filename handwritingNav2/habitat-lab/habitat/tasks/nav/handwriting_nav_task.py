from typing import Any, Dict, List, Optional, Type, Union

import attr
from gym import spaces
import numpy as np
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask
from habitat.core.dataset import Episode
import cv2
@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[int]] = None
    
def preprocess_image(path, target_size=512):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    h, w = img.shape[:2]
    max_dim = max(h, w)

    square_img = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255  

    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2

    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
    resized_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return resized_img

@attr.s(auto_attribs=True, kw_only=True)
class HWNavEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation
    of agent, goal specifications, instruction specifications, reference path,
    and optional shortest paths.

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: list of goals specifications
        reference_path: List of (x, y, z) positions which gives the reference
            path to the goal that aligns with the hand writing map.
        trajectory_id: id of ground truth trajectory path.
    """
    reference_path: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: InstructionData = attr.ib(
        default=None
    )
    trajectory_id: int = attr.ib(default=None, validator=not_none_validator)
    handwriting_map: np.array = attr.ib(init=False)

    def __attrs_post_init__(self):
        # 使用绝对路径确保可以找到手绘地图
        # 根据训练/验证阶段使用不同的目录
        instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
        
        # 其他数据集的路径（如果需要，可以取消注释）
        #instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/train_clipasso/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
        #instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/train_smooth/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
        #instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/val_seen_clipasso/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
        #instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/val_seen_smooth/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
        #instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/val_unseen_clipasso/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
        #instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/val_unseen_smooth/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
        
        self.handwriting_map = preprocess_image(instr_path)
        
        # Override the start_rotation with default rotation (use simulator default)
        # Default quaternion [0, 0, 0, 1] represents no rotation
        #self.start_rotation = [0.0, 0.0, 0.0, 1.0]
        


@registry.register_sensor(name="HandWritingGoalSensor")
class HandWritingGoalSensor(Sensor):

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "HandWritingNav",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        self.cls_uuid = "handwriting_goal"
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def project_to_segment(self, p, a, b):
        """
        Project point p onto segment ab.
        Return the projected point and t in [0,1].
        """
        ap = p - a
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq == 0:
            return a, 0.0  # a == b, return endpoint
        t = np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0)
        proj = a + t * ab
        return proj, t

    def compute_hierarchical_distance(self, agent_pos, reference_path, lamda):
        path = np.array(reference_path)
        agent_pos = np.array(agent_pos)

        min_dist = float("inf")
        best_proj_point = None
        best_segment_idx = 0

        # Search best projection on any segment
        for i in range(len(path) - 1):
            proj, _ = self.project_to_segment(agent_pos, path[i], path[i+1])
            dist = np.linalg.norm(agent_pos - proj)
            if dist < min_dist:
                min_dist = dist
                best_proj_point = proj
                best_segment_idx = i

        d_off_path = min_dist

        # Compute remaining path distance from projection point to goal
        d_remaining = 0.0
        # from projected point to end of segment
        d_remaining += np.linalg.norm(path[best_segment_idx+1] - best_proj_point)
        # rest of the path
        if best_segment_idx + 1 < len(path) - 1:
            d_remaining += np.sum(
                np.linalg.norm(np.diff(path[best_segment_idx+1:], axis=0), axis=1)
            )

        hierarchical_distance = lamda * d_remaining + (1 - lamda) * d_off_path

        return np.array([hierarchical_distance], dtype=np.float32)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: HWNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        current_position = self._sim.get_agent_state().position
        points = [point.position for point in episode.reference_path]
        return self.compute_hierarchical_distance(current_position, points, lamda=0.8)



@registry.register_sensor(name="HandWritingInstrSensor")
class HandWritingInstrSensor(Sensor):

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "HandWritingNav",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        self.cls_uuid = "handwriting_instr"
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (512, 512, 3)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: HWNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:

        return episode.handwriting_map



@registry.register_task(name="HandWritingNav")
class HandWritingNavTask(NavigationTask):
    r"HandwritingNavTask"
#     def overwrite_sim_config(
#         self, sim_config: Any, episode: Type[Episode]
#     ) -> Any:
#         return merge_sim_episode_config(sim_config, episode)


# def merge_sim_episode_config(
#     sim_config: Config, episode: Type[Episode]
# ) -> Any:
#     sim_config.defrost()
#     # here's where the scene update happens, extract the scene name out of the path
#     sim_config.SCENE = episode.scene_id
#     sim_config.freeze()
#     if (
#         episode.start_position is not None
#         and episode.start_rotation is not None
#     ):
#         agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
#         agent_cfg = getattr(sim_config, agent_name)
#         agent_cfg.defrost()
#         agent_cfg.START_POSITION = episode.start_position
#         agent_cfg.START_ROTATION = episode.start_rotation
#         agent_cfg.GOAL_POSITION = episode.goals[0].position
#         agent_cfg.IS_SET_START_STATE = True
#         agent_cfg.freeze()
#     return sim_config