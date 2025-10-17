import habitat
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import observations_to_image
from habitat.utils.visualizations import maps
import os
import cv2
import numpy as np
from habitat.config.default import get_config
from typing import Union, List, Dict, Any, cast
import datetime
from yacs.config import CfgNode as CN
import math

def get_yaw_and_pitch(rotation):
    """
    Extract yaw and pitch from quaternion
    """
    w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
    
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    sin_pitch = 2.0 * (w * x - y * z)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    
    return yaw, pitch

class ShortestPathFollowerAgent(Agent):
    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast(HabitatSim, env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )
        # Initialize trajectory and visualization
        self.trajectory = []
        self.current_map = None
        self.world_to_map_func = None
        self.scene_bounds = None
        self.map_resolution = (1024, 1024)
        self.base_maps = {}

    def reset(self) -> None:
        self.trajectory = []
        # Reset or initialize the map for the current scene
        scene_id = self.env.current_episode.scene_id
        if scene_id not in self.base_maps:
            self._generate_base_map(scene_id)
        self.current_map = self.base_maps[scene_id].copy()
        self.world_to_map_func = self._get_world_to_map_func(scene_id)

    def _generate_base_map(self, scene_id: str) -> None:
        try:
            # Get agent's initial height from current position
            agent_state = self.env.sim.get_agent_state()
            agent_height = agent_state.position[1] if agent_state else 0.5
            print(f"Using agent height {agent_height} for map generation")
            top_down_map = maps.get_topdown_map(
                self.env.sim.pathfinder,
                height=agent_height,
                map_resolution=1024
            )
            top_down_map = maps.colorize_topdown_map(top_down_map)
            self.base_maps[scene_id] = top_down_map
            # Get scene bounds for coordinate conversion
            self.scene_bounds = self.env.sim.pathfinder.get_bounds()
            print(f"Scene bounds for {scene_id}: lower={self.scene_bounds[0]}, upper={self.scene_bounds[1]}")
        except Exception as e:
            print(f"Error generating base map for {scene_id}: {e}")
            self.base_maps[scene_id] = np.zeros((self.map_resolution[0], self.map_resolution[1], 3), dtype=np.uint8)

    def _get_world_to_map_func(self, scene_id: str):
        if scene_id not in self.base_maps:
            return None
        lower_bound, upper_bound = self.scene_bounds
        h, w = self.base_maps[scene_id].shape[0], self.base_maps[scene_id].shape[1]
        world_width = upper_bound[0] - lower_bound[0]
        world_height = upper_bound[2] - lower_bound[2]

        def world_to_map(world_x, world_z):
            norm_x = (world_x - lower_bound[0]) / world_width
            norm_z = (world_z - lower_bound[2]) / world_height
            map_x = int(np.clip(norm_x * w, 0, w-1))
            map_y = int(np.clip(norm_z * h, 0, h-1))  
            print(f"World coords ({world_x:.2f}, {world_z:.2f}) -> Map coords ({map_x}, {map_y})")
            return map_y, map_x

        return world_to_map

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        # Record current position and orientation
        agent_state = self.env.sim.get_agent_state()
        position = agent_state.position
        rotation = agent_state.rotation
        yaw, pitch = get_yaw_and_pitch(rotation)
        
        # Add to trajectory if it's the first position or after a step
        if len(self.trajectory) == 0:
            self.trajectory.append({
                'position': position,
                'orientation': {'yaw': yaw, 'pitch': pitch}
            })
            # Draw start marker
            if self.current_map is not None and self.world_to_map_func is not None:
                start_map_y, start_map_x = self.world_to_map_func(position[0], position[2])
                cv2.circle(self.current_map, (start_map_x, start_map_y), 10, (0, 255, 0), -1)
                cv2.putText(self.current_map, "S", (start_map_x-10, start_map_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        next_action = self.shortest_path_follower.get_next_action(
            self.env.current_episode.goals[0].position
        )
        if next_action is None:
            current_pos = self.env.sim.get_agent_state().position
            goal_pos = self.env.current_episode.goals[0].position
            distance = self.env.sim.geodesic_distance(current_pos, goal_pos)
            if isinstance(distance, np.ndarray):
                distance = distance.item()
            if distance <= self.shortest_path_follower.goal_radius:
                return HabitatSimActions.STOP
            else:
                print(f"No valid path to goal position {goal_pos}")
                return HabitatSimActions.STOP
        return next_action

    def update_trajectory_after_step(self, observations):
        # After taking a step, get the new position and draw the trajectory segment
        agent_state = self.env.sim.get_agent_state()
        position = agent_state.position
        rotation = agent_state.rotation
        yaw, pitch = get_yaw_and_pitch(rotation)
        
        self.trajectory.append({
            'position': position,
            'orientation': {'yaw': yaw, 'pitch': pitch}
        })
        
        # Draw trajectory segment if we have at least 2 points
        if len(self.trajectory) >= 2 and self.current_map is not None and self.world_to_map_func is not None:
            prev_pos = self.trajectory[-2]['position']
            curr_pos = self.trajectory[-1]['position']
            start_map_y, start_map_x = self.world_to_map_func(prev_pos[0], prev_pos[2])
            end_map_y, end_map_x = self.world_to_map_func(curr_pos[0], curr_pos[2])
            cv2.line(
                self.current_map,
                (start_map_x, start_map_y),
                (end_map_x, end_map_y),
                color=(0, 0, 255),
                thickness=5
            )
            print(f"Drew trajectory segment from ({prev_pos[0]:.2f}, {prev_pos[2]:.2f}) to ({curr_pos[0]:.2f}, {curr_pos[2]:.2f})")

    def finalize_trajectory(self):
        # Draw end marker on the map if trajectory exists
        if len(self.trajectory) > 0 and self.current_map is not None and self.world_to_map_func is not None:
            end_pos = self.trajectory[-1]['position']
            end_map_y, end_map_x = self.world_to_map_func(end_pos[0], end_pos[2])
            cv2.circle(self.current_map, (end_map_x, end_map_y), 10, (255, 0, 0), -1)
            cv2.putText(self.current_map, "E", (end_map_x-10, end_map_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def get_trajectory_map(self) -> np.ndarray:
        return self.current_map if self.current_map is not None else np.zeros((self.map_resolution[0], self.map_resolution[1], 3), dtype=np.uint8)

    def get_trajectory(self) -> List[Dict[str, Any]]:
        return self.trajectory

class VisualizationAgent:
    """Agent for generating top-down maps and trajectory visualizations"""
    def __init__(self, sim):
        self.sim = sim
        # 增加地图分辨率以获得更清晰的图像，可以按需调整
        self.map_resolution = (1024, 1024)
    
    def generate_top_down_map(self, trajectory: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate top-down map for the scene and correctly draw the trajectory.
        
        Args:
            trajectory: Optional, if provided draw trajectory on the map
            
        Returns:
            np.ndarray: Top-down map image
        """
        try:
            # 1. 获取基础俯视图
            # height 参数表示地图是基于哪个高度平面生成的，可以根据场景调整
            top_down_map = maps.get_topdown_map(
                self.sim.pathfinder,
                height=0.3,
                map_resolution=1024
            )

            # 转换为彩色图像
            top_down_map = maps.colorize_topdown_map(top_down_map)
            return top_down_map
            
        except Exception as e:
            print(f"Error generating top-down map: {e}")
            import traceback
            traceback.print_exc()  # 打印详细的错误堆栈
            # 返回一个空白图像作为备用
            map_image = np.zeros((self.map_resolution[0], self.map_resolution[1], 3), dtype=np.uint8)
            cv2.putText(map_image, "Map generation failed", (20, self.map_resolution[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return map_image

def setup_log_directory(base_dir: str) -> str:
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_trajectory_data(
    output_dir: str,
    episode_id: int,
    scene_id: str,
    trajectory: List[Dict[str, Any]],
    observations_list: List[np.ndarray],
    viz_agent: VisualizationAgent,
    episode_counts: Dict[str, int],
    episode_totals: Dict[str, int],
    goal_position: np.ndarray,
    trajectory_map: np.ndarray,
    split_ratio: float = 0.9
) -> None:
    scene_name = os.path.splitext(os.path.basename(scene_id))[0]  # Remove .glb
    scene_datasets_dir = os.path.join(output_dir, "scene_datasets")
    
    # Determine train/test split
    episode_count = episode_counts.get(scene_name, 0)
    episode_counts[scene_name] = episode_count + 1
    total_episodes = episode_totals.get(scene_name, 1)
    split_idx = math.floor(split_ratio * total_episodes)
    split = "train" if episode_count < split_idx or total_episodes <= 1 else "test"
    print(f"Scene {scene_name}, episode {episode_id}, count {episode_count}, total episodes {total_episodes}, split index {split_idx}, assigned to {split}")
    
    scene_dir = os.path.join(scene_datasets_dir, split, scene_name)
    traj_dir = os.path.join(scene_dir, f"traj_{episode_id}")
    os.makedirs(traj_dir, exist_ok=True)

    # Generate and save top-down map (only once per scene)
    floor_plan_path = os.path.join(scene_dir, "floorplan.png")
    if not os.path.exists(floor_plan_path):
        try:
            map_image = viz_agent.generate_top_down_map()
            cv2.imwrite(floor_plan_path, map_image)
            print(f"Saved top-down map to: {floor_plan_path}")
        except Exception as e:
            print(f"Error saving top-down map for {scene_name}: {e}")

    # Save trajectory data
    positions = np.array([step['position'] for step in trajectory])
    yaws = np.array([step['orientation']['yaw'] for step in trajectory])
    pitches = np.array([step['orientation']['pitch'] for step in trajectory])
    print(f"Positions shape: {positions.shape}, Yaws shape: {yaws.shape}, Pitches shape: {pitches.shape}")
    
    trajectory_array = np.column_stack((positions[:, [0, 2]], pitches, yaws))
    np.save(os.path.join(traj_dir, f"traj_{episode_id}.npy"), trajectory_array)

    with open(os.path.join(traj_dir, f"traj_{episode_id}.txt"), "w") as f:
        for pos, pitch, yaw in zip(positions, pitches, yaws):
            f.write(f"{pos[0]:.6f} {pos[2]:.6f} {pitch:.3f} {yaw:.3f}\n")

    # Save trajectory visualization from agent's map
    try:
        cv2.imwrite(os.path.join(traj_dir, f"traj_{episode_id}.png"), trajectory_map)
        print(f"Saved trajectory visualization to: {os.path.join(traj_dir, f'traj_{episode_id}.png')}")
    except Exception as e:
        print(f"Error saving trajectory visualization for {scene_name}: {e}")


    # Save observation images
    for i, obs in enumerate(observations_list):
        try:
            if i >= 100000:
                print(f"Warning: Scene {scene_name}, trajectory {episode_id} has over 100k frames, skipping")
                break
            obs_image = observations_to_image(obs, {})
            cv2.imwrite(os.path.join(traj_dir, f"{i:05d}.png"), obs_image[:, :, ::-1])
        except Exception as e:
            print(f"Error saving observation image {i:05d}.png for {scene_name}: {e}")

def main():
    config = habitat.get_config("/data/xhj/handwritingNav/flona_dataset/pointnav.yaml")
    config.defrost()
    config.DATASET.DATA_PATH = "/data/xhj/handwritingNav/flona_dataset/episodes_1k_filtered.json.gz"
    config.freeze()

    output_dir = "/data/xhj/handwritingNav/flona_dataset/dataset"
    os.makedirs(output_dir, exist_ok=True)

    try:
        import habitat_sim
        print(f"Habitat-Sim version: {habitat_sim.__version__}")
        with habitat.Env(config=config) as env:
            goal_radius = env.episodes[0].goals[0].radius or 3
            agent = ShortestPathFollowerAgent(env=env, goal_radius=goal_radius)
            
            # Create visualization agent (for floor plan generation only)
            viz_agent = VisualizationAgent(env.sim)

            # Pre-calculate total episodes per scene
            episode_totals = {}
            for episode in env.episodes:
                scene_name = os.path.splitext(os.path.basename(episode.scene_id))[0]
                episode_totals[scene_name] = episode_totals.get(scene_name, 0) + 1
            print(f"Total episodes per scene: {episode_totals}")
        
            episode_counts = {}  # Track processed episode counts

            for episode_id in range(len(env.episodes)):
                try:
                    observations = env.reset()
                    print(f"Processing episode {episode_id}, scene: {env.current_episode.scene_id}")
                    agent.reset()  # Reset trajectory and map for new episode
                    done = False
                    observations_list = []
                    
                    # Get goal position for this episode
                    goal_position = env.current_episode.goals[0].position

                    while not done:
                        observations_list.append(observations)
                        action = agent.act(observations)
                        if action is None:
                            break
                        observations = env.step(action)
                        agent.update_trajectory_after_step(observations)
                        done = env.episode_over
                    print(f"Episode {episode_id} finished")
                    agent.finalize_trajectory()

                    save_trajectory_data(
                        output_dir,
                        episode_id,
                        env.current_episode.scene_id,
                        agent.get_trajectory(),
                        observations_list,
                        viz_agent,
                        episode_counts,
                        episode_totals,
                        goal_position,
                        agent.get_trajectory_map(),
                        split_ratio=0.9
                    )

                except Exception as e:
                    print(f"Error processing episode {episode_id}, scene {env.current_episode.scene_id}: {e}")
                    continue

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()