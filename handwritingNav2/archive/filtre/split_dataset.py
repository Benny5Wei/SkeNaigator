import json
import os

def split_episodes(input_path: str, output_dir: str):
   
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    total = len(episodes)
    print(f"原始 JSON 中共 {total} 条 episodes。")

   
    splits = [
        (1000, "episodes_1k_filtered.json")
    ]

   
    os.makedirs(output_dir, exist_ok=True)

   
    for count, filename in splits:
        subset = episodes[:min(total, count)]
        out_data = {"episodes": subset}

        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(out_data, out_f, ensure_ascii=False, indent=2)

        actual = len(subset)
        print(f"已生成 {filename}，共包含 {actual} 条 episodes。")

if __name__ == "__main__":
   
    INPUT_JSON = "/data/xhj/handwritingNav/data/mp3d_hwnav/train/train_rxr_filtered.json"     
    OUTPUT_DIR = "/data/ww/4test/generate_dataset/"        

    split_episodes(INPUT_JSON, OUTPUT_DIR)
    print("所有文件已生成完毕。")