# 快速启动指南

## 前提条件

✅ 已完成的准备工作：
- 安装了 Python 3.7+ 
- 安装了 habitat-sim 和相关依赖
- 准备了训练数据集
- 有可用的 NVIDIA GPU

## 快速开始训练

### 1. 激活环境

```bash
conda activate habitat
```

### 2. 进入脚本目录

```bash
cd /mnt_data/skenav2/handwritingNav2/scripts
```

### 3. 开始训练

#### 使用所有可用GPU（推荐）

```bash
python train.py
```

#### 使用特定数量的GPU

```bash
# 使用3个GPU
python train.py --num-gpus 3

# 使用2个GPU
python train.py --num-gpus 2

# 单GPU训练
python train.py --num-gpus 1
```

#### 使用特定的GPU

```bash
# 只使用GPU 0和1
python train.py --gpu-ids "0,1"

# 只使用GPU 2
python train.py --gpu-ids "2"
```

## 训练模式

### 扩散策略训练（默认）

```bash
python train.py
# 自动使用 modeling/config/train_diffusion_hwnav.yaml
```

### PPO训练

```bash
python train.py --exp-config ../modeling/config/train_hwnav.yaml
```

## 从检查点恢复训练

```bash
python train.py --resume-from /path/to/checkpoint.pth
```

## 监控训练进度

### 方法1: 查看日志

```bash
tail -f /path/to/model_dir/train.log
```

### 方法2: TensorBoard

在另一个终端运行：

```bash
tensorboard --logdir=/path/to/model_dir/tb --port=6006
```

然后在浏览器访问: http://localhost:6006

## 评估模型

```bash
python train.py --run-type eval --model-dir /path/to/trained/model
```

## 配置说明

### 关键配置参数

在 `modeling/config/train_diffusion_hwnav.yaml`:

```yaml
# 并行环境数量（建议: num_gpus * 4）
NUM_PROCESSES: 12

# 训练更新次数
NUM_UPDATES: 100000

# 检查点保存间隔
CHECKPOINT_INTERVAL: 500

# 日志输出间隔
LOG_INTERVAL: 10
```

### 调整NUM_PROCESSES

根据GPU内存选择：
- **3个24GB GPU**: `NUM_PROCESSES: 12-24`
- **2个24GB GPU**: `NUM_PROCESSES: 8-16`
- **1个24GB GPU**: `NUM_PROCESSES: 4-8`

## 常用命令示例

### 完整训练流程

```bash
# 1. 激活环境
conda activate habitat

# 2. 进入目录
cd /mnt_data/skenav2/handwritingNav2/scripts

# 3. 使用所有GPU开始训练
python train.py

# 4. 在另一个终端监控（可选）
tensorboard --logdir=../data/models/output/tb
```

### 自定义配置训练

```bash
# 使用自定义配置和输出目录
python train.py \
  --exp-config ../modeling/config/train_diffusion_hwnav.yaml \
  --model-dir /mnt_data/skenav2/experiments/exp1 \
  --num-gpus 3
```

### 调试模式（单GPU单进程）

修改配置文件设置 `NUM_PROCESSES: 1`，然后：

```bash
python train.py --num-gpus 1
```

## 预期结果

### 训练输出

```
使用所有可用GPU: 0,1,2 (共3个GPU)
使用默认配置文件: .../train_diffusion_hwnav.yaml
使用Habitat兼容扩散策略训练器
Loaded 47296 episodes from /mnt_data/skenav2/data/big_train_1.json
初始化环境...
开始训练...
```

### 训练速度

- **3 GPU (软件渲染)**: ~50-100 FPS
- **3 GPU (GPU渲染)**: ~500-1000 FPS
- **1 GPU (软件渲染)**: ~20-40 FPS

注：当前配置使用软件渲染作为后备，速度较慢但稳定。

## 常见问题快速解决

### 问题1: 显存不足

```bash
# 减少并行进程数
# 编辑 modeling/config/train_diffusion_hwnav.yaml
NUM_PROCESSES: 6  # 减少到6
```

### 问题2: EGL渲染错误

✅ 已自动处理，使用软件渲染

### 问题3: 找不到数据集

确认数据路径正确：
```bash
ls /mnt_data/skenav2/data/big_train_1.json
ls /mnt_data/skenav2/data/scene_datasets/
```

### 问题4: 模块导入错误

```bash
# 确保在正确的环境
conda activate habitat

# 检查依赖
pip list | grep -E "torch|habitat|diffusers"
```

## 下一步

- 📖 阅读 [多GPU训练详细指南](MULTI_GPU_TRAINING.md)
- 🔧 查看 [EGL问题故障排除](TROUBLESHOOTING_EGL.md)
- 📊 了解 [数据集要求](DATA_REQUIREMENTS.md)

## 获取帮助

遇到问题时：
1. 检查 `train.log` 查看详细错误信息
2. 运行 `nvidia-smi` 检查GPU状态
3. 查看文档目录中的故障排除指南
4. 确保所有依赖已正确安装

祝训练顺利！🚀

