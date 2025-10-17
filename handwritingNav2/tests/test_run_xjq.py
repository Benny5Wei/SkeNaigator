import os
import cv2
import habitat
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import observations_to_image
from typing import Union, cast
import numpy as np
from habitat.config.default import get_config
from yacs.config import CfgNode as CN

class ShortestPathFollowerAgent(Agent):
    r"""Uses Habitat's ShortestPathFollower to follow the shortest path to the goal."""

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast(HabitatSim, env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def reset(self) -> None:
        pass

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        next_action = self.shortest_path_follower.get_next_action(
            self.env.current_episode.goals[0].position
        )
        if next_action is None:
            return HabitatSimActions.STOP
        return next_action

def main():
    config = habitat.get_config("/data/xhj/handwritingNav/test/pointnav.yaml")
    config.defrost()

    config.DATASET.DATA_PATH = "/data/xhj/handwritingNav/train_filter2_no_trajectory_id.json.gz"


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


    save_dir = "saved_observations"
    os.makedirs(save_dir, exist_ok=True)

    with habitat.Env(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = 3

        agent = ShortestPathFollowerAgent(env=env, goal_radius=goal_radius)

        total_success = 0
        total_spl = 0

        for episode_id in range(len(env.episodes)):
            observations = env.reset()
            agent.reset()
            done = False
            step_id = 0


            episode_dir = os.path.join(save_dir, f"episode_{episode_id:03d}")
            os.makedirs(episode_dir, exist_ok=True)

            while not done:
                action = agent.act(observations)
                if action is None:
                    print(f"Episode {episode_id}: No path to goal found.")
                    break


                vis = observations_to_image(observations)
                img_path = os.path.join(episode_dir, f"step_{step_id:04d}.png")
                cv2.imwrite(img_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

                observations = env.step(action)
                done = env.episode_over
                step_id += 1

            metrics = env.get_metrics()
            print(f"Episode {episode_id} finished. spl: {metrics['spl']}, success: {metrics['success']}")
            total_success += metrics.get("success", 0)
            total_spl += metrics.get("spl", 0)


            if episode_id >= 499:
                break

        num_episodes = min(len(env.episodes), 500)
        print(f"\nOverall success rate: {total_success/num_episodes:.2f}")
        print(f"Overall SPL: {total_spl/num_episodes:.3f}")

if __name__ == "__main__":
    main()
