# 多GPU训练指南

## 概述

本项目已配置支持多GPU并行训练，可以充分利用所有可用的GPU来加速训练过程。

## 环境配置

### 自动EGL渲染修复

训练脚本已内置EGL渲染问题的解决方案，会自动设置以下环境变量：
- `MAGNUM_DEVICE=0`
- `EGL_PLATFORM=surfaceless`
- `LIBGL_ALWAYS_SOFTWARE=1` (软件渲染后备)
- `HABITAT_SIM_LOG=quiet`

无需手动设置或使用shell脚本。

## 多GPU训练配置

### 方法 1: 使用所有可用GPU（默认）

```bash
cd /mnt_data/skenav2/handwritingNav2/scripts
python train.py
```

这将自动检测并使用所有可用的GPU。

### 方法 2: 指定GPU数量

```bash
python train.py --num-gpus 3
```

使用前3个GPU（GPU 0, 1, 2）。

### 方法 3: 指定特定的GPU ID

```bash
python train.py --gpu-ids "0,2"
```

只使用GPU 0和GPU 2进行训练。

```bash
python train.py --gpu-ids "1,2,3"
```

使用GPU 1, 2, 3进行训练。

### 方法 4: 单GPU训练

```bash
python train.py --num-gpus 1
```

或者：

```bash
python train.py --gpu-ids "0"
```

## 配置文件调整

### NUM_PROCESSES 参数

在配置文件中（如 `train_diffusion_hwnav.yaml`），`NUM_PROCESSES` 控制并行环境的数量：

```yaml
# 建议设置: num_gpus * 4
# 例如: 3个GPU -> NUM_PROCESSES: 12
NUM_PROCESSES: 12
```

**调整建议：**
- **高内存GPU (24GB+)**: `num_gpus * 4` 到 `num_gpus * 8`
- **中等内存GPU (12GB)**: `num_gpus * 2` 到 `num_gpus * 4`
- **低内存GPU (8GB)**: `num_gpus * 1` 到 `num_gpus * 2`

### 批量大小调整

对于扩散策略训练，可以调整：

```yaml
RL:
  DIFFUSION:
    # 学习率随GPU数量缩放
    lr: 1e-4  # 基础学习率
```

对于PPO训练：

```yaml
RL:
  PPO:
    num_mini_batch: 4  # 可以根据GPU数量增加
    num_steps: 150
```

## 性能优化

### 1. 环境数量优化

根据GPU内存调整 `NUM_PROCESSES`：

```python
# 检查GPU内存使用
nvidia-smi
```

如果内存不足，减少 `NUM_PROCESSES`。

### 2. 数据加载优化

确保数据集路径正确：

```yaml
DATASET:
  DATA_PATH: "/mnt_data/skenav2/data/big_train_1.json"
  SCENES_DIR: "/mnt_data/skenav2/data/scene_datasets"
```

### 3. 检查点保存

多GPU训练时，检查点会定期保存：

```yaml
CHECKPOINT_INTERVAL: 500  # 每500次更新保存一次
```

## 训练示例

### 示例 1: 扩散策略多GPU训练

```bash
# 使用所有3个GPU，扩散策略
cd /mnt_data/skenav2/handwritingNav2/scripts
python train.py --exp-config ../modeling/config/train_diffusion_hwnav.yaml
```

### 示例 2: PPO多GPU训练

```bash
# 使用前2个GPU，PPO策略
python train.py --num-gpus 2 --exp-config ../modeling/config/train_hwnav.yaml
```

### 示例 3: 指定输出目录

```bash
# 使用特定GPU并指定输出目录
python train.py --gpu-ids "0,1" --model-dir /mnt_data/skenav2/experiments/run1
```

### 示例 4: 从检查点恢复训练

```bash
# 从之前的检查点恢复训练
python train.py --resume-from /path/to/checkpoint.pth
```

## 监控训练

### TensorBoard

启动TensorBoard查看训练进度：

```bash
# 在另一个终端中运行
tensorboard --logdir=/mnt_data/skenav2/experiments --port=6006
```

### 训练日志

日志文件保存在：
```
<model_dir>/train.log
```

## 常见问题

### Q1: 显存不足 (Out of Memory)

**解决方案：**
1. 减少 `NUM_PROCESSES`
2. 减少并行环境数量
3. 使用更少的GPU

```bash
# 示例：减少到每GPU 2个进程
# 修改 train_diffusion_hwnav.yaml
NUM_PROCESSES: 6  # 3 GPUs * 2
```

### Q2: 训练速度慢

**可能原因：**
- 使用了软件渲染（`LIBGL_ALWAYS_SOFTWARE=1`）

**解决方案：**
尝试重新启动Docker容器并启用GPU渲染：

```bash
docker run --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
  -v /mnt_data:/mnt_data \
  -it your_container
```

### Q3: EGL 错误

**当前配置已自动处理：**
脚本会自动使用软件渲染作为后备，虽然较慢但可以运行。

### Q4: 进程间通信错误

**解决方案：**
尝试设置：

```yaml
USE_SYNC_VECENV: True  # 使用同步向量环境
```

## 性能基准

### 预期训练速度（参考）

使用3个RTX 3090：
- **GPU渲染**: ~500-1000 FPS (frames per second)
- **软件渲染**: ~50-100 FPS

使用1个RTX 3090：
- **GPU渲染**: ~200-400 FPS
- **软件渲染**: ~20-40 FPS

## 调试模式

### 单进程调试

```bash
# 设置为1个进程方便调试
python train.py --num-gpus 1
```

然后在配置文件中设置：
```yaml
NUM_PROCESSES: 1
```

### 详细日志

修改 `train.py` 中的日志级别：

```python
# 在配置中设置
DEBUG: True
```

## 更多资源

- [Habitat-Lab 文档](https://github.com/facebookresearch/habitat-lab)
- [扩散策略论文](https://diffusion-policy.cs.columbia.edu/)
- [PPO算法说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

## 需要帮助？

如果遇到问题，请查看：
1. `docs/TROUBLESHOOTING_EGL.md` - EGL渲染问题
2. `docs/DATA_REQUIREMENTS.md` - 数据集要求
3. 项目日志文件和TensorBoard

