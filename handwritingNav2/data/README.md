# 扩散策略数据集

本目录用于存储扩散策略训练所需的离线专家演示数据。

## 数据集结构

```
diffusion_dataset/
├── train/                    # 训练集
│   ├── scene_name_1/        # 场景1
│   │   ├── episode_0/       # Episode 0
│   │   │   ├── rgb_00000.png
│   │   │   ├── rgb_00001.png
│   │   │   ├── ...
│   │   │   ├── depth_00000.npy
│   │   │   ├── depth_00001.npy
│   │   │   ├── ...
│   │   │   ├── actions.npy
│   │   │   ├── positions.npy
│   │   │   └── metadata.npy
│   │   ├── episode_1/
│   │   └── ...
│   ├── scene_name_2/
│   └── ...
└── test/                     # 测试集
    └── ...

```

## 生成数据集

运行以下命令生成专家演示数据：

```bash
cd /mnt_data/skenav2/handwritingNav2/scripts
python generate_diffusion_dataset.py
```

数据将自动保存到 `/mnt_data/skenav2/handwritingNav2/data/diffusion_dataset/`

## 数据格式说明

### Episode数据

每个episode目录包含：

- **rgb_XXXXX.png**: RGB观察图像序列
- **depth_XXXXX.npy**: Depth观察序列（如果启用）
- **actions.npy**: 动作序列 (shape: [T], dtype: int32)
  - 0: STOP
  - 1: MOVE_FORWARD
  - 2: TURN_LEFT
  - 3: TURN_RIGHT
- **positions.npy**: 位置序列 (shape: [T, 3], dtype: float32)
- **metadata.npy**: 元数据字典

## 训练

使用离线数据集训练扩散策略：

```bash
cd /mnt_data/skenav2/handwritingNav2/scripts
python train.py --run-type train --exp-config modeling/config/train_diffusion_hwnav.yaml
```

## 注意事项

1. 数据生成可能需要较长时间，取决于episode数量
2. 确保有足够的磁盘空间（约10-50GB）
3. 训练时不需要多进程环境，只需要单个GPU
4. 数据加载使用PyTorch DataLoader，支持多线程加载

