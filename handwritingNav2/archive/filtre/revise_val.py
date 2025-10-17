#!/usr/bin/env python3

import json
import os
from tqdm import tqdm

# 输入和输出文件路径
input_file = '/data/xhj/handwritingNav/data/mp3d_hwnav/r2r/val_1.json'
output_file = '//data/xhj/handwritingNav/data/mp3d_hwnav/r2r/val_2.json'

def process_json():
    print(f"正在处理文件: {input_file}")
    
    # 检查文件大小，以便设置进度条
    file_size = os.path.getsize(input_file)
    print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
    
    # 读取JSON文件
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 获取episodes数组长度用于进度显示
    total_episodes = len(data['episodes'])
    print(f"共有 {total_episodes} 个episodes需要处理")
    
    # 移除每个episode中的trajectory_id和instruction和reference_path
    for episode in tqdm(data['episodes'], desc="处理进度"):
        episode['trajectory_id'] = 1
        episode['start_rotation']= [0.0, 0.0, 0.0, 1.0]
        episode['info'] = {
            "geodesic_distance": 0
          }
    
    # 写入新文件
    print(f"正在写入到新文件: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"处理完成! 结果已保存到: {output_file}")

if __name__ == "__main__":
    process_json()
