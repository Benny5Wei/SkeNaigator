#!/usr/bin/env python3
"""
为扩散策略生成专家演示数据集

使用ShortestPathFollower收集专家轨迹，保存到磁盘供后续离线训练使用。
数据将保存到 /mnt_data/skenav2/handwritingNav2/data/diffusion_dataset/
"""

import os
import sys

# 必须在导入其他库之前设置环境变量（尝试使用GPU加速）
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['GLOG_minloglevel'] = '2'

# 尝试GPU渲染配置（提高速度）
# 如果失败会自动回退到CPU
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.5'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '450'

# 设置EGL设备（尝试使用GPU 0）
os.environ['EGL_DEVICE_ID'] = '0'
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'i965'

import cv2
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Union, cast
import datetime
import logging

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
HABITAT_LAB_PATH = os.path.join(PROJECT_ROOT, 'habitat-lab')
sys.path.insert(0, HABITAT_LAB_PATH)

import habitat
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExpertDemonstrationAgent(Agent):
    """使用ShortestPathFollower收集专家演示的Agent"""
    
    def __init__(self, env: habitat.Env, goal_radius: float = 0.2, max_steps: int = 500):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast(HabitatSim, env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )
        self.max_steps = max_steps
        self.step_count = 0
        
        # 存储轨迹数据
        self.observations_seq = []
        self.actions_seq = []
        self.positions_seq = []
        self.rotations_seq = []
        
    def reset(self) -> None:
        """重置agent状态"""
        self.step_count = 0
        self.observations_seq = []
        self.actions_seq = []
        self.positions_seq = []
        self.rotations_seq = []
        
    def act(self, observations: Dict) -> Union[int, None]:
        """执行专家动作"""
        # 检查最大步数
        if self.step_count >= self.max_steps:
            logger.warning(f"达到最大步数 {self.max_steps}，停止导航")
            return HabitatSimActions.STOP
        
        # 获取专家动作
        next_action = self.shortest_path_follower.get_next_action(
            self.env.current_episode.goals[0].position
        )
        
        if next_action is None:
            # 检查是否已到达目标
            current_pos = self.env.sim.get_agent_state().position
            goal_pos = self.env.current_episode.goals[0].position
            distance = self.env.sim.geodesic_distance(current_pos, goal_pos)
            if isinstance(distance, np.ndarray):
                distance = distance.item()
            
            if distance <= self.shortest_path_follower.goal_radius:
                return HabitatSimActions.STOP
            else:
                logger.warning(f"没有到目标位置的有效路径，距离: {distance:.2f}m")
                return HabitatSimActions.STOP
        
        return next_action
    
    def store_step(self, observations: Dict, action: int) -> None:
        """存储当前步的数据"""
        agent_state = self.env.sim.get_agent_state()
        
        self.observations_seq.append(observations)
        self.actions_seq.append(action)
        self.positions_seq.append(agent_state.position.copy())
        self.rotations_seq.append(agent_state.rotation)
        
        self.step_count += 1
        
    def get_trajectory_data(self) -> Dict[str, Any]:
        """获取完整的轨迹数据"""
        return {
            'observations': self.observations_seq,
            'actions': self.actions_seq,
            'positions': self.positions_seq,
            'rotations': self.rotations_seq,
            'length': len(self.actions_seq)
        }


def save_episode_data(
    output_dir: str,
    episode_id: int,
    scene_id: str,
    trajectory_data: Dict[str, Any],
    split: str = 'train',
) -> bool:
    """保存单个episode的数据"""
    try:
        scene_name = os.path.splitext(os.path.basename(scene_id))[0]
        scene_dir = os.path.join(output_dir, split, scene_name)
        traj_dir = os.path.join(scene_dir, f"episode_{episode_id}")
        os.makedirs(traj_dir, exist_ok=True)
        
        # 验证轨迹质量
        positions = np.array(trajectory_data['positions'])
        if len(positions) < 2:
            logger.warning(f"轨迹太短 ({len(positions)} 步)，跳过")
            return False
        
        # 检查位置是否有实际变化
        position_changes = np.diff(positions, axis=0)
        max_change = np.max(np.linalg.norm(position_changes, axis=1))
        total_distance = np.sum(np.linalg.norm(position_changes, axis=1))
        
        if max_change < 0.01:
            logger.warning(f"轨迹变化太小 (max: {max_change:.4f}m)，跳过")
            return False
        
        if total_distance < 0.1:
            logger.warning(f"轨迹总距离太短 ({total_distance:.4f}m)，跳过")
            return False
        
        logger.info(f"轨迹统计 - 长度: {len(positions)}, 最大变化: {max_change:.3f}m, 总距离: {total_distance:.3f}m")
        
        # 只保存深度图（节省存储空间，提高加载速度）
        observations = trajectory_data['observations']
        for t, obs in enumerate(observations):
            # 保存Depth
            if 'depth' in obs:
                depth_path = os.path.join(traj_dir, f"depth_{t:05d}.npy")
                np.save(depth_path, obs['depth'])
                
                # 保存深度图可视化（便于检查数据质量）
                if t % 10 == 0:  # 每10帧保存一个可视化
                    depth = obs['depth'].squeeze()
                    # 归一化到0-255用于可视化
                    depth_norm = np.clip(depth, 0.5, 5.0)
                    depth_vis = ((depth_norm - 0.5) / 4.5 * 255).astype(np.uint8)
                    # 使用伪彩色映射
                    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
                    vis_path = os.path.join(traj_dir, f"depth_vis_{t:05d}.png")
                    cv2.imwrite(vis_path, depth_color)
            
            # 不保存RGB（节省存储空间）
            # if 'rgb' in obs:
            #     rgb_path = os.path.join(traj_dir, f"rgb_{t:05d}.png")
            #     cv2.imwrite(rgb_path, cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
        
        # 保存动作序列
        actions = np.array(trajectory_data['actions'], dtype=np.int32)
        actions_path = os.path.join(traj_dir, "actions.npy")
        np.save(actions_path, actions)
        
        # 保存位置序列
        positions_path = os.path.join(traj_dir, "positions.npy")
        np.save(positions_path, positions)
        
        # 保存元数据
        metadata = {
            'episode_id': episode_id,
            'scene_id': scene_id,
            'trajectory_length': len(actions),
            'total_distance': float(total_distance),
            'split': split,
        }
        metadata_path = os.path.join(traj_dir, "metadata.npy")
        np.save(metadata_path, metadata)
        
        logger.info(f"✅ 成功保存轨迹到: {traj_dir}")
        return True
        
    except Exception as e:
        logger.error(f"保存episode {episode_id} 数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数：收集专家演示数据"""
    
    # 配置输出目录
    output_dir = "/mnt_data/skenav2/handwritingNav2/data/diffusion_dataset"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"数据将保存到: {output_dir}")
    
    # 使用专门的PointNav配置（不加载HandWritingNav，速度快）
    config_path = os.path.join(PROJECT_ROOT, "modeling/config/pointnav_datagen.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    logger.info(f"加载配置文件: {config_path}")
    config = habitat.get_config(config_path)
    config.defrost()
    
    # 使用CPU渲染（Docker环境中GPU EGL不可用）
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = -1  # -1 = CPU渲染
    logger.info("使用CPU渲染模式（稳定，适合Docker环境）")
    logger.info("注意：CPU渲染较慢，但不会出错。如果需要GPU加速，需要配置EGL设备。")
    
    logger.info(f"使用数据集: {config.DATASET.DATA_PATH}")
    
    # 配置已经包含所有必要设置，直接冻结
    config.freeze()
    
    # 创建环境
    logger.info("创建Habitat环境...")
    with habitat.Env(config=config) as env:
        # 获取目标半径
        goal_radius = 0.2
        if len(env.episodes) > 0 and hasattr(env.episodes[0].goals[0], 'radius'):
            goal_radius = env.episodes[0].goals[0].radius or 0.2
        
        logger.info(f"目标半径: {goal_radius}m")
        logger.info(f"总episode数: {len(env.episodes)}")
        
        # 创建专家agent
        agent = ExpertDemonstrationAgent(env, goal_radius=goal_radius)
        
        # 统计变量
        total_episodes = len(env.episodes)
        
        # 从配置文件读取数据生成参数
        max_episodes = None
        split_ratio = 0.9
        
        if hasattr(config, 'DATA_GENERATION'):
            # 读取最大episode数量
            if hasattr(config.DATA_GENERATION, 'MAX_EPISODES'):
                max_eps = config.DATA_GENERATION.MAX_EPISODES
                if max_eps > 0:  # -1或0表示处理全部
                    max_episodes = max_eps
            
            # 读取训练集/测试集划分比例
            if hasattr(config.DATA_GENERATION, 'SPLIT_RATIO'):
                split_ratio = config.DATA_GENERATION.SPLIT_RATIO
        
        # 应用episode数量限制
        if max_episodes is not None and max_episodes < total_episodes:
            logger.info(f"⚠️  限制处理前 {max_episodes} 个episodes（总共{len(env.episodes)}个）")
            total_episodes = max_episodes
        else:
            logger.info(f"将处理全部 {total_episodes} 个episodes")
        
        saved_episodes = 0
        skipped_episodes = 0
        split_idx = int(split_ratio * total_episodes)
        
        logger.info(f"训练集/测试集划分比例: {split_ratio:.1%} / {1-split_ratio:.1%}")
        
        # 遍历episodes
        for episode_idx in range(total_episodes):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"处理 Episode {episode_idx}/{total_episodes}")
                logger.info(f"{'='*60}")
                
                # 重置环境和agent
                observations = env.reset()
                agent.reset()
                
                scene_id = env.current_episode.scene_id
                logger.info(f"场景: {os.path.basename(scene_id)}")
                logger.info(f"起始位置: {env.sim.get_agent_state().position}")
                logger.info(f"目标位置: {env.current_episode.goals[0].position}")
                
                done = False
                step_count = 0
                
                # 收集轨迹
                while not done:
                    # 获取专家动作
                    action = agent.act(observations)
                    
                    if action is None or action == HabitatSimActions.STOP:
                        logger.info(f"收到停止动作，episode结束")
                        break
                    
                    # 存储数据
                    agent.store_step(observations, action)
                    
                    # 执行动作
                    observations = env.step(action)
                    done = env.episode_over
                    step_count += 1
                    
                    if step_count % 50 == 0:
                        current_pos = env.sim.get_agent_state().position
                        logger.info(f"步骤 {step_count}: 位置 {current_pos}")
                
                # 获取指标
                metrics = env.get_metrics()
                spl = metrics.get("spl", 0)
                success = metrics.get("success", 0)
                
                if isinstance(spl, np.ndarray):
                    spl = spl.item()
                if isinstance(success, np.ndarray):
                    success = success.item()
                
                logger.info(f"Episode {episode_idx} 完成 - SPL: {spl:.3f}, Success: {success}, 步数: {step_count}")
                
                # 确定split
                split = "train" if episode_idx < split_idx else "test"
                
                # 保存数据
                trajectory_data = agent.get_trajectory_data()
                if save_episode_data(output_dir, episode_idx, scene_id, trajectory_data, split):
                    saved_episodes += 1
                else:
                    skipped_episodes += 1
                    
            except Exception as e:
                logger.error(f"处理episode {episode_idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                skipped_episodes += 1
                continue
        
        # 打印总结
        logger.info(f"\n{'='*60}")
        logger.info(f"数据收集完成！")
        logger.info(f"{'='*60}")
        logger.info(f"处理episode数: {total_episodes} / {len(env.episodes)}")
        logger.info(f"成功保存: {saved_episodes}")
        logger.info(f"跳过: {skipped_episodes}")
        logger.info(f"训练集: ~{int(saved_episodes * split_ratio)} episodes")
        logger.info(f"测试集: ~{saved_episodes - int(saved_episodes * split_ratio)} episodes")
        logger.info(f"数据保存位置: {output_dir}")
        
        if max_episodes is not None and max_episodes < len(env.episodes):
            logger.info(f"\n💡 提示: 当前只处理了前 {max_episodes} 个episodes")
            logger.info(f"   如需处理全部，请在配置文件中设置 DATA_GENERATION.MAX_EPISODES: -1")


if __name__ == "__main__":
    main()

