# 数据需求说明

本文档详细说明HandWriting Navigation项目所需的数据格式和组织方式。

## 📊 数据概览

训练该项目需要以下数据：

1. **3D场景数据** (Matterport3D)
2. **导航Episode数据** (JSON格式)
3. **手绘地图** (PNG图像)

## 🗂️ 数据组织结构

```
/mnt_data/skenav2/
├── data/
│   ├── scene_datasets/              # 3D场景数据
│   │   └── mp3d/                   # Matterport3D场景
│   │       ├── 17DRP5sb8fy/
│   │       │   └── 17DRP5sb8fy.glb
│   │       ├── 1LXtFkjw3qL/
│   │       │   └── 1LXtFkjw3qL.glb
│   │       └── ...
│   │
│   └── mp3d_hwnav/                 # 导航数据
│       ├── big_train_1.json        # Episode定义文件
│       └── big_train_1/            # 手绘地图目录
│           ├── 17DRP5sb8fy_epi11265_cut1.png
│           ├── 1LXtFkjw3qL_epi29_cut1.png
│           └── ...
│
└── handwritingNav2/                # 项目代码
    └── modeling/
        └── config/
            └── hwnav_base.yaml     # 配置文件
```

## 📋 数据详细说明

### 1. Episode数据 (JSON文件)

**位置**: `/mnt_data/skenav2/data/big_train_1.json`

**格式**:
```json
{
  "episodes": [
    {
      "episode_id": 1,
      "trajectory_id": 4,
      "scene_id": "mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb",
      "start_position": [-16.267, 0.152, 0.721],
      "start_rotation": [0.0, 0.707, 0.0, -0.707],
      "info": {
        "geodesic_distance": 6.425
      },
      "goals": [
        {
          "position": [-12.337, 0.152, 4.214],
          "radius": 3.0
        }
      ],
      "reference_path": [
        [-16.267, 0.152, 0.721],
        [-16.284, 0.152, 2.412],
        ...
      ]
    }
  ]
}
```

**字段说明**:
- `episode_id`: Episode唯一标识符
- `trajectory_id`: 轨迹ID
- `scene_id`: 场景文件路径
- `start_position`: 起始位置 [x, y, z]
- `start_rotation`: 起始旋转 (四元数)
- `goals`: 目标位置列表
- `reference_path`: 参考路径点列表

**您的数据**:
- ✅ 已有: `/mnt_data/skenav2/data/big_train_1.json`
- ✅ 大小: 约340万行 (包含大量episodes)

### 2. 手绘地图 (PNG图像)

**位置**: `/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/`

**命名格式**: `{scene_id}_epi{episode_id}_cut1.png`

**示例**:
- `17DRP5sb8fy_epi11265_cut1.png`
- `1LXtFkjw3qL_epi29_cut1.png`

**要求**:
- 格式: PNG
- 建议分辨率: 任意 (会被预处理为512x512)
- 内容: 从起点到终点的手绘路径
- 背景: 白色 (255, 255, 255)
- 路径: 黑色线条

**您的数据**:
- ✅ 已有: `/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/`
- ✅ 数量: 47,296 张图像
- ✅ 总大小: 1.7 GB

### 3. 3D场景数据 (Matterport3D)

**位置**: 应该在 `data/scene_datasets/mp3d/`

**格式**:
```
mp3d/
├── 17DRP5sb8fy/
│   ├── 17DRP5sb8fy.glb          # 3D场景文件
│   ├── 17DRP5sb8fy.house        # 房屋结构
│   └── 17DRP5sb8fy.navmesh      # 导航网格
├── 1LXtFkjw3qL/
│   └── ...
```

**获取方式**:
1. 访问 [Matterport3D网站](https://niessner.github.io/Matterport/)
2. 填写数据使用协议
3. 下载场景数据

**场景列表** (从您的数据推断):
根据JSON文件中的`scene_id`，需要以下场景：
- 17DRP5sb8fy
- 1LXtFkjw3qL
- 7y3sRwLe3Va
- jh4fc5c5qoQ
- ... (更多场景)

## 🔗 数据映射关系

### Episode → 手绘地图

每个Episode通过以下规则映射到对应的手绘地图：

```python
scene_name = episode.scene_id.split('/')[-2]  # 例如: "17DRP5sb8fy"
episode_id = episode.episode_id              # 例如: 11265
image_path = f"data/mp3d_hwnav/big_train_1/{scene_name}_epi{episode_id}_cut1.png"
```

**示例**:
- Episode: `episode_id=11265`, `scene_id="mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"`
- 对应图像: `17DRP5sb8fy_epi11265_cut1.png`

### Episode → 3D场景

```python
scene_path = "data/scene_datasets/" + episode.scene_id
# 例如: "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
```

## ⚙️ 配置数据路径

### 方法1: 修改配置文件（推荐）

编辑 `modeling/config/hwnav_base.yaml`:

```yaml
DATASET:
  TYPE: "HandWritingNav"
  SPLIT: "train"
  SCENES_DIR: "data/scene_datasets"  # 场景目录
  DATA_PATH: "data/mp3d_hwnav/big_train_1.json"  # Episode文件
```

### 方法2: 使用软链接

如果数据在其他位置，可以创建软链接：

```bash
cd /mnt_data/skenav2/handwritingNav2

# 链接Episode数据
ln -s /mnt_data/skenav2/data/mp3d_hwnav data/mp3d_hwnav

# 链接场景数据（如果有）
ln -s /path/to/mp3d data/scene_datasets/mp3d
```

### 方法3: 修改代码中的路径

编辑 `habitat-lab/habitat/tasks/nav/handwriting_nav_task.py`:

```python
# 第63行，修改为您的实际路径
def __attrs_post_init__(self):
    instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
    self.handwriting_map = preprocess_image(instr_path)
```

## 🔍 验证数据完整性

### 1. 检查JSON文件

```bash
# 查看文件大小
ls -lh /mnt_data/skenav2/data/big_train_1.json

# 查看episode数量
python -c "
import json
with open('/mnt_data/skenav2/data/big_train_1.json') as f:
    data = json.load(f)
    print(f'总Episodes数: {len(data[\"episodes\"])}')
"
```

### 2. 检查手绘地图

```bash
# 统计图像数量
ls /mnt_data/skenav2/data/mp3d_hwnav/big_train_1/*.png | wc -l

# 检查文件大小
du -sh /mnt_data/skenav2/data/mp3d_hwnav/big_train_1/

# 查看示例图像
ls /mnt_data/skenav2/data/mp3d_hwnav/big_train_1/*.png | head -5
```

### 3. 验证映射关系

```python
import json

# 加载JSON
with open('/mnt_data/skenav2/data/big_train_1.json') as f:
    data = json.load(f)

# 检查第一个episode
episode = data['episodes'][0]
scene_name = episode['scene_id'].split('/')[-2]
episode_id = episode['episode_id']

# 对应的图像路径
image_path = f"/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/{scene_name}_epi{episode_id}_cut1.png"

# 验证文件存在
import os
print(f"Episode ID: {episode_id}")
print(f"Scene: {scene_name}")
print(f"Image exists: {os.path.exists(image_path)}")
```

## 📝 当前数据状态

### ✅ 已准备好的数据

1. **Episode数据**: 
   - 路径: `/mnt_data/skenav2/data/big_train_1.json`
   - 状态: ✅ 完整

2. **手绘地图**: 
   - 路径: `/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/`
   - 数量: 47,296 张
   - 大小: 1.7 GB
   - 状态: ✅ 完整

### ⚠️ 可能缺少的数据

3. **3D场景数据**: 
   - 期望路径: `/mnt_data/skenav2/data/scene_datasets/mp3d/`
   - 状态: ❓ 需要确认

## 🚀 开始训练的准备工作

### 1. 确认数据位置

```bash
# 检查当前数据位置
ls -la /mnt_data/skenav2/data/
ls -la /mnt_data/skenav2/data/mp3d_hwnav/

# 检查项目期望的数据位置
ls -la /mnt_data/skenav2/handwritingNav2/data/
```

### 2. 创建数据链接或移动数据

**选项A: 创建符号链接（推荐）**
```bash
cd /mnt_data/skenav2/handwritingNav2

# 链接到项目data目录
ln -s /mnt_data/skenav2/data/mp3d_hwnav data/mp3d_hwnav
ln -s /mnt_data/skenav2/data/big_train_1.json data/mp3d_hwnav/big_train_1.json
```

**选项B: 修改配置文件使用绝对路径**
```yaml
# modeling/config/hwnav_base.yaml
DATASET:
  SCENES_DIR: "/mnt_data/skenav2/data/scene_datasets"
  DATA_PATH: "/mnt_data/skenav2/data/big_train_1.json"
```

### 3. 修改代码中的硬编码路径

编辑 `habitat-lab/habitat/tasks/nav/handwriting_nav_task.py` 第63行：

```python
# 原代码（需要修改）
instr_path = f"data/mp3d_hwnav/train_clipasso/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"

# 改为
instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
```

### 4. 更新配置文件

确保 `modeling/config/hwnav_base.yaml` 指向正确的路径：

```yaml
DATASET:
  TYPE: "HandWritingNav"
  SPLIT: "train"
  CONTENT_SCENES: ["*"]
  VERSION: 'v1'
  SCENES_DIR: "/mnt_data/skenav2/data/scene_datasets"
  DATA_PATH: "/mnt_data/skenav2/data/big_train_1.json"
```

## 🎯 数据需求总结

| 数据类型 | 路径 | 状态 | 说明 |
|---------|------|------|------|
| Episode JSON | `/mnt_data/skenav2/data/big_train_1.json` | ✅ 已有 | 340万行 |
| 手绘地图 | `/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/` | ✅ 已有 | 47,296张PNG |
| 3D场景 | `/mnt_data/skenav2/data/scene_datasets/mp3d/` | ❓ 待确认 | Matterport3D |

## ⚠️ 重要注意事项

1. **手绘地图位置**: 代码中硬编码了 `data/mp3d_hwnav/train_clipasso/` 路径，但您的数据在 `big_train_1/`，需要修改
2. **3D场景数据**: 训练需要实际的3D场景文件，如果缺少可能导致仿真器无法加载
3. **路径一致性**: 确保配置文件、代码和实际数据路径保持一致

## 🔧 下一步

1. **确认3D场景数据**是否存在
2. **修改硬编码路径**指向 `big_train_1/`
3. **创建数据链接**或更新配置文件
4. **运行验证测试**确保数据可以正常加载

---

**最后更新**: 2024

