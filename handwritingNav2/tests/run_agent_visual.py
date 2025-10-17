import os
import cv2
import datetime
import json
import numpy as np
import habitat
from typing import Union, List, Dict, Any, cast
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import observations_to_image
from habitat.tasks.nav.nav import NavigationEpisode
from yacs.config import CfgNode as CN


class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the habitat.core.agent.Agent interface that
    uses ShortestPathFollower to get the next action along the shortest path
    to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

        # Initialize the ShortestPathFollower with the simulator and goal radius
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast(HabitatSim, env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        # Get the next action toward the goal’s position
        next_action = self.shortest_path_follower.get_next_action(
            self.env.current_episode.goals[0].position
        )
        # If there is no valid next action, issue a STOP
        if next_action is None:
            return HabitatSimActions.STOP
        return next_action

    def reset(self) -> None:
        # No special reset behavior needed here
        pass


def setup_log_directory() -> str:
    """Create and return a directory path for logging."""
    log_dir = os.path.join("/data/xhj/handwritingNav/logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def log_non_perfect_episodes(episodes_data: List[Dict[str, Any]], log_dir: str) -> None:
    """Write episodes with SPL < 1 to a log file."""
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

            if "scene_id" in episode_data:
                log_file.write(f"Scene ID: {episode_data['scene_id']}\n")

            if "goal_position" in episode_data:
                log_file.write(f"Goal Position: {episode_data['goal_position']}\n")

            log_file.write("---\n")

    print(f"\nLogged {len(episodes_data)} non-perfect episodes to: {log_filename}")


def main():
    # Load the PointNav YAML configuration
    config = habitat.get_config("/data/xhj/handwritingNav/test/pointnav.yaml")
    config.defrost()
    # Point to our JSON-GZ dataset that contains multiple episodes
    config.DATASET.DATA_PATH = "/data/xhj/handwritingNav/none_perfect_rxr.json.gz"

    # Add necessary measurements for PointNav evaluation
    config.TASK.MEASUREMENTS = []
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")

    if not hasattr(config.TASK, "DISTANCE_TO_GOAL"):
        config.TASK.DISTANCE_TO_GOAL = CN()
    config.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
    config.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")

    if not hasattr(config.TASK, "SUCCESS"):
        config.TASK.SUCCESS = CN()
    config.TASK.SUCCESS.TYPE = "Success"
    config.TASK.SUCCESS.SUCCESS_DISTANCE = 3
    config.TASK.MEASUREMENTS.append("SUCCESS")

    if not hasattr(config.TASK, "SPL"):
        config.TASK.SPL = CN()
    config.TASK.SPL.TYPE = "SPL"
    config.TASK.MEASUREMENTS.append("SPL")

    config.freeze()

    # ────────────────────────────────────────────────────────────────
    # Create a `dataset` object from the JSON-GZ file. This tells Habitat to
    # read each episode’s "scene_id" from the JSON and load scenes accordingly.
    dataset = habitat.make_dataset(id_dataset=config.DATASET.TYPE, config=config.DATASET)

    # Pass both `config` and `dataset` into the Env constructor. Habitat will
    # automatically load each episode’s scene based on the JSON “scene_id”.
    with habitat.Env(config=config, dataset=dataset) as env:
        # Retrieve the goal radius from the first episode (default to 3 if None)
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = 3

        agent = ShortestPathFollowerAgent(env=env, goal_radius=goal_radius)

        total_success = 0
        total_spl = 0
        non_perfect_episodes: List[Dict[str, Any]] = []

        # Prepare two log files:
        #  - failed_nav_file:   episodes where success == 0
        #  - failed_load_file:  episodes where env.reset() itself fails
        failed_nav_file = "/data/xhj/handwritingNav/test/failed_episodes.txt"
        failed_load_file = "/data/xhj/handwritingNav/test/failed_load_episodes.txt"
        open(failed_nav_file, "w").close()
        open(failed_load_file, "w").close()

        # Create a folder to save each episode’s “final-frame” PNG
        frame_dir = os.path.join(os.path.dirname(failed_nav_file), "final_frames")
        os.makedirs(frame_dir, exist_ok=True)

        # Iterate over all episodes loaded in `env.episodes`
        for idx in range(len(env.episodes)):
            try:
                # Attempt to reset the environment to the idx-th episode
                observations = env.reset()
            except Exception as e:
                # If env.reset() fails, log that episode_id and skip
                real_id_load_error = env.episodes[idx].episode_id
                print(
                    f"Real Episode ID {real_id_load_error}: environment reset failed, skipping. Error: {e}"
                )
                with open(failed_load_file, "a") as f_load:
                    f_load.write(f"{real_id_load_error}\n")
                continue

            # Reset succeeded; get the true episode_id from the current episode
            real_id = env.current_episode.episode_id

            # Let the agent navigate until episode_over is True
            done = False
            while not done:
                action = agent.act(observations)
                if action is None:
                    print(f"Real Episode ID {real_id}: No path to goal found.")
                    break
                observations = env.step(action)
                done = env.episode_over

            # ─── Save the final-frame image for this episode ───────────────────────
            # At this point, `observations` corresponds to the last step’s observation.
            info = env.get_metrics()
            # Convert observation to an RGB numpy array
            frame = observations_to_image(observations, info)
            # Convert RGB to BGR for OpenCV saving
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Build filename: "{sceneName}_{episodeID}_final.png"
            scene_name = os.path.basename(env.current_episode.scene_id)
            filename = f"{scene_name}_{real_id}_final.png"
            save_path = os.path.join(frame_dir, filename)

            # Write image to disk immediately
            cv2.imwrite(save_path, frame_bgr)
            print(f"Saved final-frame image: {save_path}")
            # ─────────────────────────────────────────────────────────────────────

            # ─── Compute and log navigation metrics after the episode ────────────
            metrics = env.get_metrics()
            spl = metrics.get("spl", 0.0)
            success = metrics.get("success", 0.0)

            print(f"Real Episode ID {real_id} finished. SPL: {spl:.3f}, Success: {success}")

            # If navigation failed (success == 0), record in failed_nav_file
            if success == 0:
                with open(failed_nav_file, "a") as f_nav:
                    f_nav.write(f"{real_id}\n")

            # If SPL < 1, collect the episode info for later logging
            if spl < 1.0:
                episode_data = {
                    "episode_id": real_id,
                    "spl": spl,
                    "success": success,
                    "scene_id": env.current_episode.scene_id,
                    "goal_position": env.current_episode.goals[0].position,
                }
                non_perfect_episodes.append(episode_data)

            total_success += success
            total_spl += spl
            # ─────────────────────────────────────────────────────────────────────

        # ─── After all episodes, print overall stats and log non-perfect ones ───
        num_episodes = min(len(env.episodes), 500)
        print(f"\nOverall success rate: {total_success / num_episodes:.2f}")
        print(f"Overall SPL: {total_spl / num_episodes:.3f}")
        print(f"Episodes with SPL < 1: {len(non_perfect_episodes)} out of {num_episodes}")

        if non_perfect_episodes:
            log_dir = setup_log_directory()
            log_non_perfect_episodes(non_perfect_episodes, log_dir)


if __name__ == "__main__":
    main()
