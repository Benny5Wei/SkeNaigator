import habitat
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import observations_to_image
from habitat.utils.visualizations import maps
import os
import cv2
from typing import Union, cast, List, Dict, Any
import numpy as np
from habitat.config.default import get_config
from habitat.tasks.nav.nav import NavigationEpisode
from yacs.config import CfgNode as CN
import datetime
import json
import time
import math


class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref:`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast(HabitatSim, env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        next_action = self.shortest_path_follower.get_next_action(
            self.env.current_episode.goals[0].position
        )
        if next_action is None:
            return HabitatSimActions.STOP
        return next_action

    def reset(self) -> None:
        pass

def setup_visualization_directory() -> str:
    """Create and return the path to the visualization directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join("/data/xhj/handwritingNav/visualizations", timestamp)
    os.makedirs(vis_dir, exist_ok=True)
    
    return vis_dir

def visualize_depth(depth_obs):
    """Convert depth observation to a colored visualization"""
    # Normalize depth for better visualization
    depth = depth_obs.copy()
    normalized_depth = depth / np.max(depth) if np.max(depth) > 0 else depth
    # Apply colormap for better visualization (jet is good for depth)
    #colored_depth = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return (normalized_depth * 255).astype(np.uint8)

def create_top_down_map(env, observations):
    """Create a top-down map visualization from the environment"""
    # Get raw occupancy map (2D, values: 0 = free, 1 = obstacle, 2 = unknown)
    top_down_map = maps.get_topdown_map_from_sim(
        env.sim, map_resolution=1024
    )
    # Convert to RGB so that subsequent drawings are preserved
    top_down_map = maps.colorize_topdown_map(top_down_map)
    
    h, w = top_down_map.shape[0], top_down_map.shape[1]
    print("Map shape:", top_down_map.shape)
    
    # 自定义坐标转换函数
    def world_to_map(world_x, world_z):
        # 获取场景边界
        scene_bounds = env.sim.pathfinder.get_bounds()
        lower_bound = scene_bounds[0]
        upper_bound = scene_bounds[1]
        
        # 计算场景宽度和高度
        world_width = upper_bound[0] - lower_bound[0]
        world_height = upper_bound[2] - lower_bound[2]
        
        # 归一化坐标
        norm_x = (world_x - lower_bound[0]) / world_width
        norm_z = (world_z - lower_bound[2]) / world_height
        
        # 转换为地图坐标并确保在有效范围内
        map_x = int(np.clip(norm_x * w, 0, w-1))
        map_y = int(np.clip(norm_z * h, 0, h-1))  # 反转y坐标以匹配地图方向
        
        print(f"World coords ({world_x:.2f}, {world_z:.2f}) -> Map coords ({map_x}, {map_y})")
        return map_x, map_y
    
    # Get agent position and heading
    agent_state = env.sim.get_agent_state()
    agent_position = agent_state.position
    
    # 使用自定义函数转换坐标
    agent_map_x, agent_map_y = world_to_map(agent_position[0], agent_position[2])
    # 绘制agent位置
    agent_radius = min(top_down_map.shape[0:2]) // 80  # Increased size for visibility
    
    # 在agent位置画一个大的绿色圆点
    cv2.circle(
        top_down_map,
        (agent_map_x, agent_map_y),
        agent_radius,
        (0, 255, 0),  # 绿色 (BGR)
        -1  # 填充圆圈
    )

    goal_position = env.current_episode.goals[0].position
    # 使用自定义坐标转换函数计算目标点位置
    goal_map_x, goal_map_y = world_to_map(goal_position[0], goal_position[2])
    # 绘制目标点为红色圆点
    point_padding = min(top_down_map.shape[0:2]) // 80
    cv2.circle(
        top_down_map,
        (goal_map_x, goal_map_y),
        point_padding,
        (0, 0, 255),  # 红色 (BGR格式)
        -1  # 填充圆圈
    )
    return top_down_map

def main():
    # Set up directory to save visualizations
    vis_dir = setup_visualization_directory()
    print(f"Visualizations will be saved to: {vis_dir}")
    
    config = habitat.get_config("/data/xhj/handwritingNav/test/pointnav.yaml")
    config.defrost()
    config.DATASET.DATA_PATH = "/data/xhj/handwritingNav/flona_dataset/train_15.json.gz"

    # Ensure DEPTH_SENSOR is enabled in the simulator
    if "DEPTH_SENSOR" not in config.SIMULATOR.AGENT_0.SENSORS:
        config.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_SENSOR")
    config.freeze()

    with habitat.Env(config=config) as env:
        # Get goal radius from the first episode
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = 3

        agent = ShortestPathFollowerAgent(env=env, goal_radius=goal_radius)

        # Iterate over all episodes defined in the JSON dataset
        for idx in range(len(env.episodes)):
            try:
                # Try to load and reset the environment to episode idx
                observations = env.reset()
            except Exception as e:
                # If env.reset() fails, log the episode_id and skip this episode
                real_id_load_error = env.episodes[idx].episode_id
                print(f"Real Episode ID {real_id_load_error}: environment reset failed, skipping. Error: {e}")
                continue
            
            # Reset succeeded, retrieve the true episode_id from the current episode
            real_id = env.current_episode.episode_id
            print(f"Starting Episode ID: {real_id}")

            # Create episode directory for visualizations
            episode_dir = os.path.join(vis_dir, f"episode_{real_id}")
            os.makedirs(episode_dir, exist_ok=True)

            done = False
            step_count = 0
            while not done:
                # Get current state observations
                rgb_img = observations["rgb"]
                
                # Save RGB image
                rgb_filename = os.path.join(episode_dir, f"step_{step_count:04d}_rgb.png")
                cv2.imwrite(rgb_filename, rgb_img[:, :, ::-1])  # Convert RGB to BGR for saving
                
                # Print available observation keys
                #print(f"Available observation keys: {list(observations.keys())}")
                
                # Process and save depth image if available
                if "depth" in observations:
                    depth_img = observations["depth"]
                    # Visualize the depth image (convert from single-channel float to visible format)
                    depth_vis = visualize_depth(depth_img)
                    depth_filename = os.path.join(episode_dir, f"step_{step_count:04d}_depth.png")
                    cv2.imwrite(depth_filename, depth_vis)
                else:
                    print("Depth observation not available")
                
                # Create and save top-down map
                top_down_map = create_top_down_map(env, observations)
                topdown_filename = os.path.join(episode_dir, f"step_{step_count:04d}_topdown.png")
                cv2.imwrite(topdown_filename, top_down_map)  # Already in BGR format
                
                # Get action and step environment
                action = agent.act(observations)
                if action is None:
                    print(f"Real Episode ID {real_id}: No path to goal found.")
                    break
                
                # Print current action for debugging
                action_map = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
                #print(f"Step {step_count}: Action: {action_map.get(action, action)}")
                
                observations = env.step(action)
                done = env.episode_over
                step_count += 1
                
                # Optional delay to prevent overwhelming disk I/O
                time.sleep(0.01)

            # Compute metrics after episode
            metrics = env.get_metrics()
            spl = metrics.get("spl", 0)
            success = metrics.get("success", 0)

            print(f"Real Episode ID {real_id} finished. SPL: {spl:.3f}, Success: {success}")
            
            print(f"Episode {real_id} visualizations saved to: {episode_dir}")


if __name__ == "__main__":
    main()
