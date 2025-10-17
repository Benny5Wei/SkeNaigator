# 数据生成指南

## 问题解决

### 原始问题
运行 `generate_diffusion_dataset.py` 时卡在 "Initializing dataset HandWritingNav"

**原因**: HandWritingNav 数据集在初始化时需要加载每个episode的手写地图图片，如果有上千个episode，会非常慢（每个episode读取一张图片并预处理）。

### 解决方案
使用PointNav配置替代HandWritingNav，避免加载手写地图图片。

## 修改内容

### 1. 创建了新配置文件
**文件**: `modeling/config/pointnav_datagen.yaml`

这个配置使用PointNav任务（TYPE: Nav-v0），不需要加载手写地图：
- 数据集类型: PointNav-v1 
- 任务类型: Nav-v0
- 传感器: RGB + Depth + PointGoal
- 成功距离: 3.0m

### 2. 修改了数据生成脚本
**文件**: `scripts/generate_diffusion_dataset.py`

- 使用 `pointnav_datagen.yaml` 配置
- 不再使用 `hwnav_base.yaml`（会触发HandWritingNav加载）
- 简化了配置代码

## 运行步骤

### 1. 激活Habitat环境

```bash
conda activate habitat
```

### 2. 进入scripts目录

```bash
cd /mnt_data/skenav2/handwritingNav2/scripts
```

### 3. 运行数据生成脚本

```bash
python generate_diffusion_dataset.py
```

## 预期输出

```
2025-10-13 XX:XX:XX,XXX - INFO - 数据将保存到: /mnt_data/skenav2/handwritingNav2/data/diffusion_dataset
2025-10-13 XX:XX:XX,XXX - INFO - 加载配置文件: /mnt_data/skenav2/handwritingNav2/modeling/config/pointnav_datagen.yaml
2025-10-13 XX:XX:XX,XXX - INFO - 使用数据集: /mnt_data/skenav2/data/big_train_1.json
2025-10-13 XX:XX:XX,XXX - INFO - 创建Habitat环境...
Initializing dataset PointNav  # 注意：现在是 PointNav，不是 HandWritingNav
2025-10-13 XX:XX:XX,XXX - INFO - 目标半径: 3.0m
2025-10-13 XX:XX:XX,XXX - INFO - 总episode数: XXX

============================================================
处理 Episode 0/XXX
============================================================
场景: XXXXXXXXXXXX
...
```

**关键点**: 现在应该显示 "Initializing dataset PointNav"，不再卡住！

## 配置文件对比

### 旧配置 (hwnav_base.yaml) ❌
```yaml
TASK:
  TYPE: HandWritingNav  # 会加载手写地图
DATASET:
  TYPE: "HandWritingNav"  # 触发加载所有手写图片
```

### 新配置 (pointnav_datagen.yaml) ✅
```yaml
TASK:
  TYPE: Nav-v0  # 简单的PointNav
DATASET:
  TYPE: PointNav-v1  # 只需要点目标，无需图片
```

## 技术细节

### 为什么HandWritingNav慢？

查看 `habitat-lab/habitat/tasks/nav/handwriting_nav_task.py` 第75行：

```python
def __attrs_post_init__(self):
    instr_path = f"/mnt_data/skenav2/data/mp3d_hwnav/big_train_1/{self.scene_id.split('/')[-2]}_epi{self.episode_id}_cut1.png"
    self.handwriting_map = preprocess_image(instr_path)  # 每个episode都要读取并处理图片
```

如果有1000个episode，就需要：
- 读取1000张图片
- 每张图片resize到512x512
- 占用大量内存

### PointNav的优势

PointNav只需要目标的3D坐标，不需要读取任何图片文件：
- 快速加载（秒级）
- 内存占用小
- 适合数据生成

### 数据集兼容性

虽然使用PointNav配置，但可以复用HandWritingNav的数据集文件（big_train_1.json），因为：
- 两者都包含起点、终点、场景ID
- PointNav忽略手写地图信息
- episode数量和路径信息完全相同

## 故障排除

### 问题1: 配置文件不存在

```
ERROR - 配置文件不存在: .../pointnav_datagen.yaml
```

**解决**: 确保已创建配置文件，路径应为：
```
/mnt_data/skenav2/handwritingNav2/modeling/config/pointnav_datagen.yaml
```

### 问题2: 数据集文件不存在

```
ERROR - 数据集文件不存在: /mnt_data/skenav2/data/big_train_1.json
```

**解决**: 检查数据集路径，或修改配置文件中的 `DATASET.DATA_PATH`

### 问题3: 场景文件不存在

```
ERROR - 场景文件未找到
```

**解决**: 确保 MP3D 场景文件在：
```
/mnt_data/skenav2/data/scene_datasets/mp3d/
```

## 完成后

数据将保存到：
```
/mnt_data/skenav2/handwritingNav2/data/diffusion_dataset/
├── train/
│   └── scene_name/
│       └── episode_X/
│           ├── rgb_00000.png
│           ├── depth_00000.npy
│           ├── actions.npy
│           ├── positions.npy
│           └── metadata.npy
└── test/
    └── ...
```

然后可以开始训练：
```bash
python train.py --run-type train --exp-config modeling/config/train_diffusion_hwnav.yaml
```

---

**更新时间**: 2024-10-13






