#!/usr/bin/env python3
"""
数据设置脚本 - 自动配置训练数据
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def print_status(message, status="INFO"):
    """打印带颜色的状态信息"""
    colors = {
        "INFO": "\033[94m",    # 蓝色
        "SUCCESS": "\033[92m", # 绿色
        "WARNING": "\033[93m", # 黄色
        "ERROR": "\033[91m",   # 红色
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{status}: {message}{reset}")

def check_file_exists(path, description):
    """检查文件或目录是否存在"""
    if os.path.exists(path):
        print_status(f"{description} 存在: {path}", "SUCCESS")
        return True
    else:
        print_status(f"{description} 不存在: {path}", "WARNING")
        return False

def analyze_json_data(json_path):
    """分析JSON数据文件"""
    print_status(f"分析JSON文件: {json_path}", "INFO")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        episodes = data.get('episodes', [])
        print_status(f"总Episodes数: {len(episodes)}", "SUCCESS")
        
        # 提取场景列表
        scenes = set()
        for ep in episodes:
            scene_id = ep.get('scene_id', '')
            if scene_id:
                scene_name = scene_id.split('/')[-2]
                scenes.add(scene_name)
        
        print_status(f"涉及的场景数: {len(scenes)}", "SUCCESS")
        print_status(f"场景列表: {sorted(list(scenes))[:5]}... (显示前5个)", "INFO")
        
        return episodes, sorted(list(scenes))
    
    except Exception as e:
        print_status(f"读取JSON失败: {e}", "ERROR")
        return [], []

def check_handwriting_maps(map_dir, episodes):
    """检查手绘地图文件"""
    print_status(f"\n检查手绘地图: {map_dir}", "INFO")
    
    if not os.path.exists(map_dir):
        print_status(f"手绘地图目录不存在", "ERROR")
        return 0, 0
    
    # 统计PNG文件
    png_files = list(Path(map_dir).glob("*.png"))
    print_status(f"找到PNG文件数: {len(png_files)}", "SUCCESS")
    
    # 检查映射关系
    missing_maps = []
    for ep in episodes[:100]:  # 只检查前100个
        scene_name = ep['scene_id'].split('/')[-2]
        episode_id = ep['episode_id']
        expected_file = f"{scene_name}_epi{episode_id}_cut1.png"
        expected_path = os.path.join(map_dir, expected_file)
        
        if not os.path.exists(expected_path):
            missing_maps.append(expected_file)
    
    if missing_maps:
        print_status(f"前100个episodes中缺少{len(missing_maps)}个地图文件", "WARNING")
        if len(missing_maps) <= 5:
            for m in missing_maps:
                print_status(f"  缺少: {m}", "WARNING")
    else:
        print_status(f"前100个episodes的地图文件都存在", "SUCCESS")
    
    return len(png_files), len(missing_maps)

def check_scene_datasets(scenes_dir, scene_list):
    """检查3D场景数据"""
    print_status(f"\n检查3D场景数据: {scenes_dir}", "INFO")
    
    if not os.path.exists(scenes_dir):
        print_status(f"场景目录不存在", "ERROR")
        return []
    
    missing_scenes = []
    found_scenes = []
    
    for scene in scene_list[:10]:  # 检查前10个
        scene_path = os.path.join(scenes_dir, scene)
        glb_file = os.path.join(scene_path, f"{scene}.glb")
        
        if os.path.exists(glb_file):
            found_scenes.append(scene)
        else:
            missing_scenes.append(scene)
    
    print_status(f"检查了{len(scene_list[:10])}个场景", "INFO")
    print_status(f"找到: {len(found_scenes)}", "SUCCESS")
    print_status(f"缺失: {len(missing_scenes)}", "WARNING")
    
    if missing_scenes:
        print_status(f"缺失的场景: {missing_scenes[:3]}...", "WARNING")
    
    return missing_scenes

def create_symlinks(source_dir, target_dir):
    """创建符号链接"""
    print_status(f"\n创建符号链接", "INFO")
    print_status(f"源: {source_dir}", "INFO")
    print_status(f"目标: {target_dir}", "INFO")
    
    try:
        if os.path.exists(target_dir):
            if os.path.islink(target_dir):
                print_status(f"链接已存在，删除旧链接", "WARNING")
                os.unlink(target_dir)
            else:
                print_status(f"目标路径已存在且不是链接，跳过", "WARNING")
                return False
        
        # 创建父目录
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        
        # 创建链接
        os.symlink(source_dir, target_dir)
        print_status(f"链接创建成功", "SUCCESS")
        return True
    
    except Exception as e:
        print_status(f"创建链接失败: {e}", "ERROR")
        return False

def update_config_file(config_path, json_path, scenes_dir, map_dir):
    """更新配置文件中的路径"""
    print_status(f"\n配置文件路径建议", "INFO")
    print_status(f"DATASET.SCENES_DIR: {scenes_dir}", "INFO")
    print_status(f"DATASET.DATA_PATH: {json_path}", "INFO")
    print_status(f"手绘地图目录: {map_dir}", "INFO")

def main():
    parser = argparse.ArgumentParser(description="数据设置和验证脚本")
    parser.add_argument("--json", default="/mnt_data/skenav2/data/big_train_1.json",
                        help="Episode JSON文件路径")
    parser.add_argument("--maps", default="/mnt_data/skenav2/data/mp3d_hwnav/big_train_1",
                        help="手绘地图目录")
    parser.add_argument("--scenes", default="/mnt_data/skenav2/data/scene_datasets/mp3d",
                        help="3D场景目录")
    parser.add_argument("--create-links", action="store_true",
                        help="创建符号链接到项目data目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HandWriting Navigation - 数据设置脚本")
    print("=" * 60)
    
    # 1. 检查JSON文件
    print_status("\n[1/4] 检查Episode数据", "INFO")
    json_exists = check_file_exists(args.json, "JSON文件")
    
    episodes = []
    scenes = []
    if json_exists:
        episodes, scenes = analyze_json_data(args.json)
    
    # 2. 检查手绘地图
    print_status("\n[2/4] 检查手绘地图", "INFO")
    maps_exist = check_file_exists(args.maps, "地图目录")
    
    if maps_exist and episodes:
        total_maps, missing_maps = check_handwriting_maps(args.maps, episodes)
    
    # 3. 检查3D场景
    print_status("\n[3/4] 检查3D场景数据", "INFO")
    scenes_exist = check_file_exists(args.scenes, "场景目录")
    
    if scenes_exist and scenes:
        missing_scenes = check_scene_datasets(args.scenes, scenes)
    
    # 4. 创建链接（可选）
    if args.create_links:
        print_status("\n[4/4] 创建符号链接", "INFO")
        
        project_data_dir = os.path.join(PROJECT_ROOT, "data")
        
        # 链接JSON
        if json_exists:
            json_link = os.path.join(project_data_dir, "mp3d_hwnav", "big_train_1.json")
            create_symlinks(args.json, json_link)
        
        # 链接地图目录
        if maps_exist:
            maps_link = os.path.join(project_data_dir, "mp3d_hwnav", "big_train_1")
            create_symlinks(args.maps, maps_link)
        
        # 链接场景目录
        if scenes_exist:
            scenes_link = os.path.join(project_data_dir, "scene_datasets", "mp3d")
            create_symlinks(args.scenes, scenes_link)
    
    # 总结
    print("\n" + "=" * 60)
    print_status("数据状态总结", "INFO")
    print("=" * 60)
    
    print(f"✓ Episode JSON: {'✅' if json_exists else '❌'}")
    print(f"✓ 手绘地图: {'✅' if maps_exist else '❌'}")
    print(f"✓ 3D场景: {'✅' if scenes_exist else '❌'}")
    
    if json_exists and maps_exist:
        print_status("\n数据基本完整，可以开始训练！", "SUCCESS")
        print_status("注意：需要在代码中修改硬编码的地图路径", "WARNING")
        print_status("  文件: habitat-lab/habitat/tasks/nav/handwriting_nav_task.py", "INFO")
        print_status("  第63行: 改为 'big_train_1' 目录", "INFO")
    else:
        print_status("\n数据不完整，请先准备数据", "ERROR")
    
    print("\n详细文档: docs/DATA_REQUIREMENTS.md")
    print("=" * 60)

if __name__ == "__main__":
    main()

