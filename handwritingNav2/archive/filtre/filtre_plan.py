import json, math, pathlib

THRESHOLD = 0.30               # 你的同层阈值（米）

data = json.loads(pathlib.Path("/data/xhj/handwritingNav/data/mp3d_hwnav/train/train_clipasso.json").read_text())
before = len(data["episodes"])

filtered = [ep for ep in data["episodes"]
            if abs(ep["start_position"][1] - ep["goals"][0]["position"][1]) <= THRESHOLD]

after = len(filtered)

print(f"Total episodes before filtering: {before}")
print(f"Total episodes after  ≤{THRESHOLD} m  filter: {after}")

pathlib.Path("/data/xhj/handwritingNav/data/mp3d_hwnav/train/train_clipasso_same_floor.json").write_text(
    json.dumps({"episodes": filtered}, indent=2)
)
