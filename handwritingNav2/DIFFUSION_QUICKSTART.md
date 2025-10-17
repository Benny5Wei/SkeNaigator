# 扩散策略训练快速开始指南

本指南将帮助您从零开始使用扩散策略训练导航模型。

## 概述

扩散策略使用**离线学习**方式，不需要多进程环境。训练流程分为两步：

1. **数据收集阶段**：使用专家（ShortestPathFollower）收集演示数据
2. **训练阶段**：从收集的数据中学习策略

## 步骤1：生成专家演示数据

### 1.1 配置数据生成

数据生成脚本会自动使用项目配置：
- 配置文件: `modeling/config/hwnav_base.yaml`
- 输出目录: `/mnt_data/skenav2/handwritingNav2/data/diffusion_dataset/`

### 1.2 运行数据生成

```bash
cd /mnt_data/skenav2/handwritingNav2/scripts
python generate_diffusion_dataset.py
```

**预期输出：**
- 训练集和测试集自动按9:1划分
- 每个episode包含RGB图像、深度图、动作序列等
- 低质量轨迹会被自动过滤

**预计时间：** 取决于episode数量，通常需要1-3小时

### 1.3 验证数据集

数据生成完成后，检查数据结构：

```bash
ls /mnt_data/skenav2/handwritingNav2/data/diffusion_dataset/train/
ls /mnt_data/skenav2/handwritingNav2/data/diffusion_dataset/test/
```

应该看到多个场景目录，每个场景包含多个episode目录。

## 步骤2：训练扩散策略

### 2.1 配置检查

确保配置文件 `modeling/config/train_diffusion_hwnav.yaml` 包含：

```yaml
# 关键配置
USE_DIFFUSION_POLICY: True
NUM_PROCESSES: 1  # 离线训练不需要多进程
USE_VECENV: False
DIFFUSION_DATA_DIR: "/mnt_data/skenav2/handwritingNav2/data/diffusion_dataset"

RL:
  DIFFUSION:
    batch_size: 32
    num_workers: 4
    lr: 1e-4
```

### 2.2 开始训练

```bash
cd /mnt_data/skenav2/handwritingNav2/scripts
python train.py --run-type train --exp-config modeling/config/train_diffusion_hwnav.yaml
```

或者使用GPU指定：

```bash
python train.py \
    --run-type train \
    --exp-config modeling/config/train_diffusion_hwnav.yaml \
    --gpu-ids 0
```

### 2.3 监控训练

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir=/mnt_data/skenav2/handwritingNav2/tb/
```

访问 `http://localhost:6006` 查看：
- Training/Loss: 训练损失
- Training/Avg_Loss: 平均损失
- Training/Learning_Rate: 学习率
- Validation/Loss: 验证损失

### 2.4 恢复训练

如果训练中断，可以从检查点恢复：

```bash
python train.py \
    --run-type train \
    --exp-config modeling/config/train_diffusion_hwnav.yaml \
    --resume-from data/checkpoints/ckpt.1000.pth
```

## 步骤3：评估和部署

### 3.1 评估模型（待实现）

```bash
python train.py --run-type eval --model-dir data/checkpoints/
```

### 3.2 使用模型进行推理

训练好的模型可以用于导航任务。模型会：
1. 接收观察序列（RGB/Depth）
2. 生成未来的动作序列
3. 执行前n个动作

## 故障排除

### 问题1：数据生成时"ConnectionResetError"

**原因：** 这是多进程环境问题，数据生成脚本不应该出现这个错误。

**解决：** 数据生成脚本使用单进程环境，不应该出现此问题。

### 问题2：训练时"No module found"

**解决：**
```bash
export PYTHONPATH=/mnt_data/skenav2/handwritingNav2:$PYTHONPATH
```

### 问题3：GPU内存不足

**解决：** 减小batch_size：

```yaml
RL:
  DIFFUSION:
    batch_size: 16  # 从32减少到16
```

### 问题4：数据加载慢

**解决：** 增加num_workers：

```yaml
RL:
  DIFFUSION:
    num_workers: 8  # 增加数据加载线程
```

## 关键配置参数说明

### 数据生成参数

在 `generate_diffusion_dataset.py` 中：
- `goal_radius`: 目标半径，默认0.2m
- `max_steps`: 每个episode最大步数，默认500

### 训练参数

在 `train_diffusion_hwnav.yaml` 中：

- `horizon`: 预测未来多少步（16）
- `n_obs_steps`: 使用多少历史观察（3）
- `n_action_steps`: 执行多少步动作（4）
- `batch_size`: 批次大小（32）
- `lr`: 学习率（1e-4）
- `NUM_UPDATES`: 总训练步数（100000）

## 与PPO训练的对比

| 特性 | PPO（在线学习） | 扩散策略（离线学习） |
|------|----------------|---------------------|
| 数据收集 | 实时交互 | 预先收集 |
| 多进程 | 需要（NUM_PROCESSES=12） | 不需要（NUM_PROCESSES=1） |
| GPU使用 | 多GPU并行环境 | 单GPU训练 |
| 内存需求 | 高（多环境） | 低（只需数据加载） |
| 训练稳定性 | 需要调参 | 相对稳定 |
| 数据复用 | 低 | 高 |

## 常见问题

**Q: 需要多少数据？**
A: 建议至少1000个episode，每个episode平均50-200步。

**Q: 训练多久可以收敛？**
A: 取决于数据量和任务复杂度，通常需要10000-50000个updates。

**Q: 可以使用多GPU训练吗？**
A: 当前实现是单GPU训练，可以通过DataParallel扩展到多GPU。

**Q: 如何提高性能？**
A: 1) 收集更多高质量数据  2) 调整网络架构  3) 使用数据增强

## 下一步

1. 收集更多场景的数据
2. 实验不同的超参数
3. 实现评估流程
4. 尝试不同的观察编码器
5. 添加数据增强

## 参考资料

- Diffusion Policy论文: https://arxiv.org/abs/2303.04137
- Habitat文档: https://aihabitat.org/docs/habitat-lab/






