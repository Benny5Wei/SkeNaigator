import os
import json
import re

def extract_scene_and_epi_numbers_from_filenames(directory):
    # 存储 (scene_id, episode_id) 组合
    valid_pairs = set()
    epi_pattern = re.compile(r'/([^/]+)_epi(\d+)_')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)
                match = epi_pattern.search(full_path)
                if match:
                    scene_id = match.group(1)
                    episode_id = int(match.group(2))
                    valid_pairs.add((scene_id, episode_id))

    return valid_pairs

def filter_json_by_scene_and_epi(json_data, valid_pairs):
    filtered_episodes = []

    for ep in json_data["episodes"]:
        episode_id = ep["episode_id"]
        scene_id_full = ep["scene_id"]  # e.g. "mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
        
        # 提取 scene_id 简写部分
        parts = scene_id_full.split('/')
        if len(parts) >= 2:
            scene_simple = parts[1]  # 取出中间的部分，比如 "1LXtFkjw3qL"

            # 检查是否在 valid_pairs 中
            if (scene_simple, episode_id) in valid_pairs:
                filtered_episodes.append(ep)

    return {
        "episodes": filtered_episodes
    }

def main():
    input_json_path = "/data/xhj/handwritingNav/data/mp3d_hwnav/r2r/val.json"         # 替换为你的输入 JSON 文件路径
    input_dir_path = "/data/xhj/handwritingNav/data/mp3d_hwnav/r2r/r2r_val/"  # 图片所在目录
    output_json_path = "/data/xhj/handwritingNav/data/mp3d_hwnav/r2r/val_1.json"  # 输出文件路径

    # 提取有效的 (scene_id, episode_id) 对
    valid_pairs = extract_scene_and_epi_numbers_from_filenames(input_dir_path)
    print(f"Found {len(valid_pairs)} valid (scene_id, episode_id) pairs.")

    # 读取原始 JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # 过滤 JSON
    filtered_data = filter_json_by_scene_and_epi(data, valid_pairs)
    
    # 写入新 JSON
    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Filtered JSON saved to {output_json_path}")

if __name__ == "__main__":
    main()