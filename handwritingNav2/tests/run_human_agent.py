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
import sys
import torch
import random

# Add the parent directory to path so utils_fmm can be imported as a package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils_fmm"))
from mapper import Mapper


class HumanAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    accepts human input through terminal for controlling the agent.
    Supports multi-step commands like "10w", "5a", "3d".
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.goal_radius = goal_radius
        # Define action mapping for user input
        self.action_map = {
            'w': HabitatSimActions.MOVE_FORWARD,
            'a': HabitatSimActions.TURN_LEFT,
            'd': HabitatSimActions.TURN_RIGHT,
            's': HabitatSimActions.STOP
        }
        # Provide instructions to the user
        print("\n==== Human Agent Controls ====")
        print("w: Move Forward (e.g., 10w = move forward 10 steps)")
        print("a: Turn Left (e.g., 5a = turn left 5 times)")
        print("d: Turn Right (e.g., 3d = turn right 3 times)")
        print("s: Stop (End Episode)")
        print("============================\n")
        
        # Store the last action chosen to return multiple of the same action
        self.last_action = None
        self.remaining_actions = 0

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        print("Geodesic dist:", self.env.get_metrics()["distance_to_goal"])
        
        # If there are remaining actions from a multi-step command, return the next one
        if self.remaining_actions > 0:
            self.remaining_actions -= 1
            return self.last_action
            
        # Get user input for action
        while True:
            user_input = input("Enter action (w/a/d/s or [number][w/a/d]): ").lower().strip()
            
            # Check for single character commands
            if user_input in self.action_map:
                self.last_action = self.action_map[user_input]
                return self.last_action
                
            # Check for multi-step commands like "10w", "5a", "3d"
            if len(user_input) > 1:
                # Try to extract number and action
                num_steps = ""
                for char in user_input[:-1]:  # All characters except the last one
                    if char.isdigit():
                        num_steps += char
                    else:
                        break  # Stop at the first non-digit character
                        
                action_char = user_input[-1]  # Last character is the action
                
                # Check if we have a valid multi-step command
                if num_steps and action_char in self.action_map:
                    try:
                        steps = int(num_steps)
                        if steps > 0:
                            print(f"Executing {steps} '{action_char}' actions")
                            self.last_action = self.action_map[action_char]
                            self.remaining_actions = steps - 1  # -1 because we're returning the first one now
                            return self.last_action
                    except ValueError:
                        pass  # If num_steps can't be converted to int, continue to error message
            
            print("Invalid input. Use 'w', 'a', 'd', 's' or a number followed by w/a/d (e.g., '5w')")

    def reset(self) -> None:
        pass

def setup_visualization_directory() -> str:
    """Create and return the path to the visualization directory"""
    sub_dir = "human_2b"
    #加时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = f"{sub_dir}_{timestamp}"
    vis_dir = os.path.join("/data/xhj/handwritingNav/visualizations", sub_dir)
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

def setup_metrics_logging(vis_dir):
    """Setup metrics logging and return log file path"""
    log_file_path = os.path.join(vis_dir, "metrics_log.json")
    
    # Initialize the metrics log with an empty list of episodes
    metrics_log = {
        "episodes": [],
        "summary": {
            "total_episodes": 0,
            "success_rate": 0.0,
            "avg_spl": 0.0,
            "avg_soft_spl": 0.0,
            "avg_distance_to_goal": 0.0,
        }
    }
    
    # Write initial log
    with open(log_file_path, "w") as f:
        json.dump(metrics_log, f, indent=4)
    
    return log_file_path

def update_metrics_log(log_file_path, episode_id, metrics):
    """Update metrics log with new episode data and recalculate summary"""
    # Extract metrics
    spl = metrics.get("spl", 0)
    success = metrics.get("success", 0)
    soft_spl = metrics.get("soft_spl", 0)
    distance_to_goal = metrics.get("distance_to_goal", 0)
    
    # Read existing log
    with open(log_file_path, "r") as f:
        metrics_log = json.load(f)
    
    # Add new episode data
    episode_data = {
        "episode_id": episode_id,
        "spl": spl,
        "success": success,
        "soft_spl": soft_spl,
        "distance_to_goal": distance_to_goal
    }
    metrics_log["episodes"].append(episode_data)
    
    # Recalculate summary
    num_episodes = len(metrics_log["episodes"])
    total_spl = sum(ep["spl"] for ep in metrics_log["episodes"])
    total_success = sum(ep["success"] for ep in metrics_log["episodes"])
    total_soft_spl = sum(ep["soft_spl"] for ep in metrics_log["episodes"])
    total_distance = sum(ep["distance_to_goal"] for ep in metrics_log["episodes"])
    
    # Update summary
    metrics_log["summary"] = {
        "total_episodes": num_episodes,
        "success_rate": total_success / num_episodes if num_episodes > 0 else 0,
        "avg_spl": total_spl / num_episodes if num_episodes > 0 else 0,
        "avg_soft_spl": total_soft_spl / num_episodes if num_episodes > 0 else 0,
        "avg_distance_to_goal": total_distance / num_episodes if num_episodes > 0 else 0,
    }
    
    # Write updated log
    with open(log_file_path, "w") as f:
        json.dump(metrics_log, f, indent=4)
    
    return metrics_log["summary"]

def main():
    # Set up directory to save visualizations
    vis_dir = setup_visualization_directory()
    print(f"Visualizations will be saved to: {vis_dir}")
    
    # Set up metrics logging
    metrics_log_path = setup_metrics_logging(vis_dir)
    print(f"Metrics will be logged to: {metrics_log_path}")
    
    config = habitat.get_config("/data/xhj/handwritingNav/test/pointnav2.yaml")
    config.defrost()
    config.DATASET.DATA_PATH = "/data/xhj/handwritingNav/human/val_2b.json.gz"
    
    # Configure for headless mode (CPU rendering)
    if not hasattr(config.SIMULATOR, "HABITAT_SIM_V0"):
        config.SIMULATOR.HABITAT_SIM_V0 = CN()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = -1  # Use CPU for rendering
    # 确保深度传感器在模拟器中启用

    # 只在模拟器中强制开启 DEPTH_SENSOR。GPS_SENSOR 和 COMPASS_SENSOR 是任务层 (TASK) 传感器，
    # 不应加入 SIMULATOR.AGENT_0.SENSORS，否则 HabitatSim 会在低层找不到相应的定义而报错。
    if "DEPTH_SENSOR" not in config.SIMULATOR.AGENT_0.SENSORS:
        config.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_SENSOR")
    print(config.SIMULATOR.AGENT_0.SENSORS)

    # 为mapper创建所需的完整配置结构
    if not hasattr(config, "TASK_CONFIG"):
        config.TASK_CONFIG = CN()
    if not hasattr(config.TASK_CONFIG, "SIMULATOR"):
        config.TASK_CONFIG.SIMULATOR = CN()
    if not hasattr(config.TASK_CONFIG.SIMULATOR, "DEPTH_SENSOR"):
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR = CN()
    if not hasattr(config.TASK_CONFIG.SIMULATOR, "AGENT_0"):
        config.TASK_CONFIG.SIMULATOR.AGENT_0 = CN()
    
    print(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)

    # 从现有配置中直接获取需要的配置参数
    depth_height = config.SIMULATOR.DEPTH_SENSOR.HEIGHT
    depth_width = config.SIMULATOR.DEPTH_SENSOR.WIDTH
    depth_hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV  # 水平视场角
    agent_height = config.SIMULATOR.AGENT_0.HEIGHT if hasattr(config.SIMULATOR.AGENT_0, "HEIGHT") else 1.5
    
    print(depth_height, depth_width, depth_hfov, agent_height)

    # 设置mapper所需的配置值 - 深度传感器
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = depth_height
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = depth_width
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV = depth_hfov  # 水平视场角
    
    # 设置mapper所需的配置值 - 代理相关
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = agent_height  # 以米为单位
    config.freeze()
    
    print(f"Configured depth sensor resolution: {depth_width}x{depth_height}")
    
    # Initialize mapper for exploration visualization
    mapper = Mapper(config, "cpu")

    with habitat.Env(config=config) as env:
        goal_radius = 3.0
        print(f"Goal radius: {goal_radius}")
        
        # Create a human-controlled agent instead of an automated one
        agent = HumanAgent(env=env, goal_radius=goal_radius)

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
            print(f"\n===============================")
            print(f"Starting Episode ID: {real_id}")
            print(f"===============================\n")

            # Create episode directory for visualizations
            episode_dir = os.path.join(vis_dir, f"episode_{real_id}")
            os.makedirs(episode_dir, exist_ok=True)
            
            # Reset mapper for new episode
            mapper.reset()

            done = False
            step_count = 0
            while not done:
                # Get current state observations
                rgb_img = observations["rgb"]
                
                # Save RGB image
                rgb_filename = os.path.join(episode_dir, f"step_{step_count:04d}_rgb.png")
                cv2.imwrite(rgb_filename, rgb_img[:, :, ::-1])  # Convert RGB to BGR for saving
                
                # Process and save depth image if available
                if "depth" in observations:
                    depth_img = observations["depth"]
                    # Visualize the depth image (convert from single-channel float to visible format)
                    depth_vis = visualize_depth(depth_img)
                    depth_filename = os.path.join(episode_dir, f"step_{step_count:04d}_depth.png")
                    cv2.imwrite(depth_filename, depth_vis)
                
                # Create and save top-down map (habitat全局地图)
                top_down_map = create_top_down_map(env, observations)
                topdown_filename = os.path.join(episode_dir, f"step_{step_count:04d}_topdown.png")
                cv2.imwrite(topdown_filename, top_down_map)  # Already in BGR format
                
                # Create and save exploration map using mapper (agent局部探索地图)
                # 准备mapper需要的观测数据
                agent_state = env.sim.get_agent_state()
                gps_obs = observations["gps"]      # (2,)
                compass_obs = observations["compass"]  # ()
                
                mapper_observations = {
                    # mapping.py会自动添加一个维度，所以这里只需要添加一个维度
                    "depth": torch.from_numpy(observations["depth"]).unsqueeze(0),  # [1, H, W]
                    "gps": torch.from_numpy(gps_obs).unsqueeze(0).float(),
                    "compass": torch.tensor([[compass_obs]], dtype=torch.float32)  
                }
                
                # 更新地图，显示agent的实时和历史探索区域
                action_per_env = [action] if step_count > 0 else [0]  # 第一步使用0作为默认动作
                exploration_map = mapper.forward(
                    mapper_observations, 
                    action_per_env, 
                    show_obstacle=True,      # 显示障碍物
                    show_visited_area=True,  # 显示已访问区域
                    show_frontier=True       # 显示探索边界
                )
                
                # 将torch张量转换为numpy数组并保存
                exploration_map_np = (exploration_map[0].cpu().numpy() * 255).astype(np.uint8)
                exploration_filename = os.path.join(episode_dir, f"step_{step_count:04d}_exploration_map.png")
                cv2.imwrite(exploration_filename, exploration_map_np)

                # Display step number
                print(f"\nStep: {step_count}")
                
                # Get human action
                action = agent.act(observations)
                
                # Print current action
                action_map = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
                print(f"Action: {action_map.get(action, action)}")

                # Take a step in the environment
                observations = env.step(action)
                done = env.episode_over
                step_count += 1

            # Compute metrics after episode
            metrics = env.get_metrics()
            spl = metrics.get("spl", 0)
            success = metrics.get("success", 0)
            soft_spl = metrics.get("soft_spl", 0)
            distance_to_goal = metrics.get("distance_to_goal", 0)

            # Display episode metrics
            print(f"\n==== Episode {real_id} Results ====")
            print(f"SPL: {spl:.3f}")
            print(f"Success: {success}")
            print(f"Soft SPL: {soft_spl:.3f}")
            print(f"Distance to Goal: {distance_to_goal:.3f}")
            print(f"Total Steps: {step_count}")
            print(f"============================\n")
            
            # Update metrics log
            summary = update_metrics_log(metrics_log_path, real_id, metrics)
            
            # Display cumulative metrics
            print(f"==== Overall Progress ====")
            print(f"Episodes Completed: {summary['total_episodes']}")
            print(f"Average SPL: {summary['avg_spl']:.3f}")
            print(f"Success Rate: {summary['success_rate']:.3f}")
            print(f"Average Soft SPL: {summary['avg_soft_spl']:.3f}")
            print(f"Average Distance to Goal: {summary['avg_distance_to_goal']:.3f}")
            print(f"============================\n")
            
            print(f"Episode {real_id} visualizations saved to: {episode_dir}")
            
            # Ask if user wants to continue to the next episode
            if idx < len(env.episodes) - 1:
                response = input("Continue to next episode? (y/n): ").strip().lower()
                if response != 'y':
                    print("Exiting...")
                    break


if __name__ == "__main__":
    main()
