import json

def count_unique_scene_ids(json_file):
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract all scene_ids
    episodes = data.get("episodes", [])
    scene_ids = []
    for item in episodes:
        if 'scene_id' in item:
            scene_ids.append(item['scene_id'])
    
    # Count unique scene_ids
    unique_scene_ids = set(scene_ids)
    
    print(f"Total entries: {len(data)}")
    print(f"Total unique scene_ids: {len(unique_scene_ids)}")
    print("Unique scene_ids:")
    for scene_id in sorted(unique_scene_ids):
        print(f"- {scene_id}")
    
    return unique_scene_ids

if __name__ == "__main__":
    json_file = "/data/xhj/handwritingNav/data/mp3d_hwnav/train/episodes_1k.json"
    count_unique_scene_ids(json_file)
