import json
import os
import shutil

# Path to the JSON file
json_file_path = '/data/xhj/handwritingNav/data/mp3d_hwnav/test/test_filter2.json'

# Create a backup of the original file
backup_file_path = json_file_path + '.backup'
shutil.copy2(json_file_path, backup_file_path)
print(f"Created backup at {backup_file_path}")

# Read the JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Count the number of episodes that will be modified
count = 0

# Modify the radius value in all episode goals
for episode in data['episodes']:
    for goal in episode['goals']:
        if 'radius' in goal:
            if goal['radius'] != 3.0:  # Only change if not already 3.0
                goal['radius'] = 3.0
                count += 1

print(f"Modified radius for {count} goals in {len(data['episodes'])} episodes")

# Save the modified data back to the file
with open(json_file_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Successfully updated {json_file_path}")
