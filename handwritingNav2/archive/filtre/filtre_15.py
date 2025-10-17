import json
import pathlib

# 假设THRESHOLD是5米
THRESHOLD = 5.0

# 加载数据集
data_path = "/data/xhj/handwritingNav/filtre/episodes_1k_filtered.json"
output_path = "/data/xhj/handwritingNav/filtre/train_15.json"

# 注意：如果文件是gzip压缩格式，需要先解压才能正确读取。
# 这里假设文件已经被解压或者不是以.gz结尾的压缩文件。
with open(data_path, 'r') as file:
    data = json.load(file)

before = len(data["episodes"])

# 筛选geodesic_distance大于5米的episode
filtered = [ep for ep in data["episodes"] if ep.get("info", {}).get("geodesic_distance", 0) > THRESHOLD]

after = len(filtered)

print(f"Total episodes before filtering: {before}")
print(f"Total episodes after >{THRESHOLD} m filter: {after}")

# 输出筛选后的数据到新文件
with open(output_path, 'w') as outfile:
    json.dump({"episodes": filtered}, outfile, indent=2)