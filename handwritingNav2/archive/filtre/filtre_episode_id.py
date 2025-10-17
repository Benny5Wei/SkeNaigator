import json

# ===== 配置路径 =====
json_path = "./none_perfect_rxr.json"          # 原始 JSON 文件路径
txt_path = "./test/failed_episodes.txt"       # 存有待删除 episode_id 的 TXT 文件路径
output_path = "./data/mp3d_hwnav/train/train_rxr_mtradius_filtered.json"     # 输出的 JSON 文件路径

# ===== 读取要删除的 episode_id 列表 =====
with open(txt_path, "r") as f:
    remove_ids = set(line.strip() for line in f if line.strip())

# 尝试将字符串变成整数（如果原始 json 是 int 类型的 episode_id）
try:
    remove_ids = set(int(x) for x in remove_ids)
except ValueError:
    pass  # 如果不能转 int，就说明原始 episode_id 是字符串，保持原样

# ===== 加载原始 JSON =====
with open(json_path, "r") as f:
    data = json.load(f)

# ===== 过滤 episodes =====
original_count = len(data["episodes"])
data["episodes"] = [
    ep for ep in data["episodes"]
    if ep["episode_id"] not in remove_ids
]

# ===== 输出结果 =====
new_count = len(data["episodes"])
print(f"原始 episodes 数量: {original_count}")
print(f"删除后剩余数量: {new_count}")

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"已保存到: {output_path}")
