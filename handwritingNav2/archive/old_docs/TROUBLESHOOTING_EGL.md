# EGL 渲染问题故障排除

## 问题描述

在 Docker 容器中运行训练时，可能会遇到以下错误：
```
Platform::WindowlessEglApplication::tryCreateContext(): unable to find EGL device for CUDA device 0
WindowlessContext: Unable to create windowless context
```

这是因为 Habitat-Sim 需要 EGL（OpenGL ES）来进行无头渲染，但 Docker 容器可能没有正确配置 EGL 设备访问。

## 解决方案

### 方案 1: 使用正确的 Docker 启动参数（推荐）

启动 Docker 容器时，需要添加以下参数来启用 GPU 和 EGL 访问：

```bash
docker run --gpus all \
    --device=/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -it your_container_name
```

如果没有 `/dev/dri`，尝试：
```bash
docker run --gpus all \
    --privileged \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    -it your_container_name
```

### 方案 2: 使用软件渲染（较慢但可靠）

已经在 `scripts/run_train.sh` 中配置：
```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

这会使用 CPU 进行渲染，速度较慢但不需要 EGL。

### 方案 3: 安装并使用 xvfb（虚拟显示）

```bash
apt-get update
apt-get install -y xvfb mesa-utils

# 启动虚拟显示
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# 然后运行训练
python train.py
```

### 方案 4: 使用预编译的 habitat-sim with headless 支持

重新安装 habitat-sim with headless rendering：
```bash
conda install habitat-sim headless -c conda-forge -c aihabitat
```

### 方案 5: 修改代码使用 CPU 模式（测试用）

在配置文件 `modeling/config/hwnav_base.yaml` 中：
```yaml
SIMULATOR:
  HABITAT_SIM_V0:
    GPU_DEVICE_ID: -1  # 使用 CPU
```

## 当前状态

项目已经做了以下修复：
1. ✅ 创建了 `/mnt_data/skenav2/data/handwriting_instr` 目录
2. ✅ 创建了 `scripts/run_train.sh` 脚本，设置了必要的环境变量
3. ✅ 配置了软件渲染作为后备方案

## 推荐的操作步骤

1. **首选方案**：重新启动 Docker 容器并使用方案 1 的参数
2. **临时方案**：使用 `scripts/run_train.sh` 运行训练（会使用软件渲染）
3. **如果需要快速测试**：暂时禁用渲染器或使用方案 5

## 运行训练

使用提供的脚本：
```bash
cd /mnt_data/skenav2/handwritingNav2/scripts
bash run_train.sh
```

或直接运行（需要手动设置环境变量）：
```bash
export LIBGL_ALWAYS_SOFTWARE=1
export MAGNUM_DEVICE=0
python train.py
```

## 性能影响

- **GPU 渲染（EGL）**：最快，推荐用于训练
- **软件渲染**：慢 10-50 倍，但可以运行
- **CPU 模式**：更慢，仅用于调试

## 需要帮助？

如果上述方法都不起作用，请检查：
1. Docker 容器是否以 `--gpus all` 启动
2. `nvidia-docker` 是否正确安装
3. NVIDIA 驱动版本是否兼容
4. habitat-sim 版本是否支持当前的 CUDA 版本

