# 详细安装和配置指南

本文档提供详细的安装步骤和常见问题解决方案。

## 系统要求

### 硬件要求
- CPU: 4核心以上推荐
- 内存: 16GB以上推荐
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 2070或更高)
- 存储: 100GB+ 可用空间（包括数据集）

### 软件要求
- 操作系统: Ubuntu 18.04/20.04 或 macOS (仅CPU)
- Python: 3.8, 3.9, 或 3.10
- CUDA: 11.3 或更高版本（GPU训练）
- GCC: 7.5 或更高版本

## 详细安装步骤

### 1. 准备环境

```bash
# 更新系统包
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    cmake \
    curl \
    wget \
    vim

# 安装CUDA（如果使用GPU）
# 请访问 https://developer.nvidia.com/cuda-downloads
```

### 2. 安装Conda/Miniconda

```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 重新加载shell
source ~/.bashrc
```

### 3. 创建Python环境

```bash
# 创建虚拟环境
conda create -n hwnav python=3.8 -y
conda activate hwnav

# 验证Python版本
python --version  # 应该显示 Python 3.8.x
```

### 4. 安装PyTorch

根据您的CUDA版本选择：

```bash
# CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 11.7
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117

# CPU only (不推荐用于训练)
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu

# 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 5. 安装Habitat-Sim

**方法1: 使用Conda（推荐）**

```bash
conda install habitat-sim -c conda-forge -c aihabitat
```

**方法2: 从源码编译**

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless  # 无显示器
# 或
python setup.py install --with-cuda  # 带CUDA支持
```

### 6. 安装项目依赖

```bash
cd /path/to/handwritingNav2

# 安装主要依赖
pip install -r requirements.txt

# 安装Habitat-Lab
cd habitat-lab
pip install -e .
cd ..
```

### 7. 验证安装

```bash
# 运行简单测试
python tests/simple_test_diffusion.py

# 如果看到 "✓ 所有测试通过！" 说明安装成功
```

## 数据集准备

### Matterport3D数据集

1. **获取访问权限**
   - 访问 [Matterport3D](https://niessner.github.io/Matterport/)
   - 填写协议并申请访问

2. **下载场景数据**
```bash
# 创建数据目录
mkdir -p data/scene_datasets/mp3d

# 下载脚本（需要提供访问凭证）
python scripts/download_mp.py --output-dir data/scene_datasets/mp3d
```

3. **下载导航数据集**
```bash
mkdir -p data/datasets/mp3d_hwnav

# 数据集应包含：
# - train episodes (训练集)
# - val_seen episodes (验证集-已见场景)
# - val_unseen episodes (验证集-未见场景)
```

### 数据集结构

```
data/
├── scene_datasets/
│   └── mp3d/
│       ├── 17DRP5sb8fy/
│       ├── 1LXtFkjw3qL/
│       └── ...
└── datasets/
    └── mp3d_hwnav/
        ├── train.json.gz
        ├── val_seen.json.gz
        └── val_unseen.json.gz
```

## 配置修改

### 修改数据路径

编辑 `modeling/config/hwnav_base.yaml`:

```yaml
DATASET:
  SCENES_DIR: "data/scene_datasets"  # 场景目录
  DATA_PATH: "data/datasets/mp3d_hwnav/train.json.gz"  # 训练数据
```

### GPU设置

编辑配置文件：

```yaml
SIMULATOR_GPU_ID: 0  # 仿真器使用的GPU
TORCH_GPU_ID: 0      # PyTorch使用的GPU
```

如果有多个GPU：

```bash
# 方法1: 环境变量
export CUDA_VISIBLE_DEVICES=0,1

# 方法2: 在配置文件中指定
SIMULATOR_GPU_ID: 0
TORCH_GPU_ID: 1
```

### 内存优化

如果内存不足，修改：

```yaml
RL:
  DIFFUSION:
    horizon: 8              # 从16减少到8
    n_action_steps: 2       # 从4减少到2
    
NUM_PROCESSES: 1            # 减少并行进程数
```

## 常见问题解决

### 问题1: ImportError: libGL.so.1

```bash
# Ubuntu/Debian
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install -y mesa-libGL
```

### 问题2: CUDA out of memory

**解决方案:**
1. 减小批处理大小
2. 减少 `horizon` 参数
3. 使用梯度累积
4. 使用混合精度训练

```python
# 在训练脚本中启用混合精度
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### 问题3: Habitat场景加载失败

```bash
# 检查场景文件
ls data/scene_datasets/mp3d/

# 验证文件完整性
python -c "
import habitat
from habitat.config import Config
config = Config()
# 测试加载
"
```

### 问题4: 多进程训练出错

```yaml
# 暂时禁用多进程
NUM_PROCESSES: 1
USE_VECENV: False
```

### 问题5: 路径相关错误

确保所有路径使用相对路径或正确的绝对路径：

```bash
# 检查当前工作目录
pwd

# 应该在项目根目录
cd /path/to/handwritingNav2

# 运行脚本
python scripts/train.py
```

## 性能优化

### 训练加速

1. **使用混合精度**
```yaml
USE_MIXED_PRECISION: True
```

2. **数据预加载**
```yaml
NUM_WORKERS: 4  # 数据加载线程数
```

3. **批处理优化**
```yaml
NUM_PROCESSES: 4  # 增加并行环境
```

### 内存优化

1. **梯度检查点**
```python
# 在模型中启用
model.gradient_checkpointing_enable()
```

2. **清理缓存**
```python
import torch
torch.cuda.empty_cache()
```

## 开发工具

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir=outputs/diffusion_exp1/tb --port=6006

# 在浏览器中访问
# http://localhost:6006
```

### 调试模式

```yaml
DEBUG: True  # 启用详细日志
LOG_INTERVAL: 1  # 更频繁的日志
```

### 代码检查

```bash
# 安装开发工具
pip install black flake8 pytest

# 格式化代码
black modeling/

# 检查代码风格
flake8 modeling/

# 运行测试
pytest tests/
```

## 下一步

安装完成后，请查看：
- [README.md](../README.md) - 快速开始指南
- [DIFFUSION_POLICY.md](DIFFUSION_POLICY.md) - 扩散策略详细说明
- [tests/](../tests/) - 测试和示例代码

## 获取帮助

如遇到问题：
1. 查看本文档的常见问题部分
2. 检查GitHub Issues
3. 联系项目维护者

---

**最后更新**: 2024

