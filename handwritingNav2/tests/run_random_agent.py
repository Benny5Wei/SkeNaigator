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
import random
import sys

class RandomAgent(Agent):
    r"""Implementation of a random agent that selects actions randomly
    from the list of possible actions. Used to test SPL metric.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self._POSSIBLE_ACTIONS = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT
        ]
        self.total_steps = 0
        
    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        # Randomly select an action from possible actions
        # We increase the probability of STOP slightly to ensure episodes eventually end
        # If total_steps gets very large, increase likelihood of STOP to prevent infinite episodes
        self.total_steps += 1
        
        if self.total_steps > 500:  # Set a maximum step limit to avoid endless wandering
            return HabitatSimActions.STOP
            
        # 10% chance to STOP, 90% chance to choose a random movement action
        if random.random() < 0.1:
            return HabitatSimActions.STOP
        else:
            # Choose randomly from movement actions (excluding STOP)
            movement_actions = [a for a in self._POSSIBLE_ACTIONS if a != HabitatSimActions.STOP]
            return random.choice(movement_actions)
            
    def reset(self) -> None:
        self.total_steps = 0


def setup_log_directory() -> str:
    """Create and return the path to the log directory"""
    log_dir = os.path.join("/data/xhj/handwritingNav/logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def calculate_soft_spl(start_distance: float, final_distance: float, path_length: float) -> float:
    """Calculate Soft SPL manually.
    
    Args:
        start_distance: Initial geodesic distance from start to goal
        final_distance: Final geodesic distance from agent to goal
        path_length: Total path length traversed by the agent
    
    Returns:
        float: Soft SPL value between 0 and 1
    """
    # Calculate soft success as progress toward the goal
    # 1 - (final distance / start distance) gives a value between 0 and 1
    # If agent moved away from goal, clamp to 0
    soft_success = max(0, 1 - (final_distance / start_distance)) if start_distance > 0 else 0
    
    # Path efficiency: direct_path_length / actual_path_length
    # This is the same weighting factor as in regular SPL
    path_efficiency = start_distance / max(start_distance, path_length)
    
    # Soft SPL = soft_success * path_efficiency
    return soft_success * path_efficiency

def log_non_perfect_episodes(episodes_data: List[Dict[str, Any]], log_dir: str) -> None:
    """Log episodes where SPL is not 1 to a file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"non_perfect_spl_{timestamp}.log")
    
    with open(log_filename, "w") as log_file:
        log_file.write(f"=== Episodes with SPL < 1 (Logged at {timestamp}) ===\n\n")
        log_file.write(f"Total non-perfect episodes: {len(episodes_data)}\n\n")
        
        for episode_data in episodes_data:
            episode_id = episode_data["episode_id"]
            spl = episode_data["spl"]
            success = episode_data["success"]
            
            log_file.write(f"Episode ID: {episode_id}\n")
            log_file.write(f"SPL: {spl:.3f}\n")
            log_file.write(f"Success: {success}\n")
            
            # Add scene information if available
            if "scene_id" in episode_data:
                log_file.write(f"Scene ID: {episode_data['scene_id']}\n")
            
            # Add goal information if available
            if "goal_position" in episode_data:
                log_file.write(f"Goal Position: {episode_data['goal_position']}\n")
                
            log_file.write("---\n")
    
    print(f"\nLogged {len(episodes_data)} non-perfect episodes to: {log_filename}")

def main():
    config = habitat.get_config("/data/xhj/handwritingNav/test/pointnav2.yaml")
    config.defrost()
    #config.DATASET.DATA_PATH = f"/data/xhj/handwritingNav/data/mp3d_hwnav/train/gTV8FGcVJC9.json.json.gz"
    config.DATASET.DATA_PATH = f"/data/xhj/handwritingNav/test/val_1.json.gz"
    # Add all necessary measurements for PointNav evaluation
    # 使用Habitat标准测量指标
    config.TASK.MEASUREMENTS = []  # 清空现有测量指标以避免重复
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    
    # 添加标准的PointNav指标 - 注意顺序很重要
    
    # 1. 首先添加 DISTANCE_TO_GOAL 指标
    if not hasattr(config.TASK, "DISTANCE_TO_GOAL"):
        config.TASK.DISTANCE_TO_GOAL = CN()
    config.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
    config.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
    
    # 2. 然后添加 SUCCESS 指标 (依赖于 DISTANCE_TO_GOAL)
    if not hasattr(config.TASK, "SUCCESS"):
        config.TASK.SUCCESS = CN()
    config.TASK.SUCCESS.TYPE = "Success"
    config.TASK.SUCCESS.SUCCESS_DISTANCE = 3  # 设置成功判定的距离阈值
    config.TASK.MEASUREMENTS.append("SUCCESS")
    
    # 3. 最后添加 SPL 指标 (依赖于 SUCCESS)
    if not hasattr(config.TASK, "SPL"):
        config.TASK.SPL = CN()
    config.TASK.SPL.TYPE = "SPL"
    config.TASK.MEASUREMENTS.append("SPL")
    config.freeze() 
    
    with habitat.Env(config=config) as env:
        # Get goal radius from episode or config if not specified
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = 3.0  # Default value
            
        agent = RandomAgent(env=env, goal_radius=goal_radius)

        # Track SPL (Success weighted by Path Length) and distance metrics
        total_success = 0
        total_spl = 0
        total_distance = 0
        
        # List to store data about episodes where SPL is not 1
        non_perfect_episodes = []

        for episode_id in range(len(env.episodes)):
            observations = env.reset()
            done = False
            
            # Step through the episode
            while not done:
                # Get action from the shortest path follower
                action = agent.act(observations)
                #print(action)
                # Check if we got a valid action
                if action is None:
                    print(f"Episode {episode_id}: No path to goal found.")
                    break
                    
                # Take step in environment
                observations = env.step(action)
                done = env.episode_over
                
                # 打印测地线距离和成功状态
                # print("Geodesic distance: ", env.sim.geodesic_distance(
                #     env.sim.get_agent_state().position, 
                #     env.current_episode.goals[0].position
                # ))
                # print("Is success: ", env.get_metrics().get("success", "N/A"))
            
            # Calculate metrics after episode completion
            metrics = env.get_metrics()
            print(f"Episode {episode_id} finished.")
            
            spl = metrics.get("spl", 0)
            success = metrics.get("success", 0)
            distance = metrics.get("distance_to_goal", 0)
            
            print("spl: ", spl)
            print("success: ", success)
            print("distance_to_goal: ", distance)
            
            # Track episodes where SPL is not 1
            if spl < 1.0:
                episode_data = {
                    "episode_id": episode_id,
                    "spl": spl,
                    "success": success,
                    "distance_to_goal": distance,
                    "scene_id": env.current_episode.scene_id,
                    "goal_position": env.current_episode.goals[0].position
                }
                non_perfect_episodes.append(episode_data)
            
            if "success" in metrics:
                total_success += metrics["success"]
            if "spl" in metrics:
                total_spl += metrics["spl"]
            if "distance_to_goal" in metrics:
                total_distance += metrics["distance_to_goal"]
        
        # Report overall statistics
        num_episodes = min(len(env.episodes), 500)  # Adjust based on actual episodes run
        print(f"\nOverall success rate: {total_success/num_episodes:.2f}")
        print(f"Overall SPL: {total_spl/num_episodes:.3f}")
        print(f"Average distance to goal: {total_distance/num_episodes:.3f}")
        #print(f"Episodes with SPL < 1: {len(non_perfect_episodes)} out of {num_episodes}")
        
        # Log episodes with SPL < 1 to a file
        if non_perfect_episodes:
            log_dir = setup_log_directory()
            log_non_perfect_episodes(non_perfect_episodes, log_dir)



if __name__ == "__main__":
    main()
    