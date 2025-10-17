# HandWriting Navigation - 基于扩散策略的手绘地图导航

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于扩散策略(Diffusion Policy)的手绘地图导航系统，用于在3D仿真环境中根据手绘地图进行智能导航。

## 📋 目录

- [特性](#特性)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [参考项目](#参考项目)
- [常见问题](#常见问题)

## ✨ 特性

- **🌊 扩散策略**: 基于DDPM的动作序列生成，产生平滑连续的导航行为
- **👁️ 多模态感知**: 支持RGB、深度、手绘地图、GPS、指南针等多种传感器
- **🎯 目标预测**: 智能从手绘地图中预测目标位置
- **🏃 PPO支持**: 同时支持传统的PPO强化学习方法
- **🔧 高度可配置**: 灵活的配置系统，支持命令行和YAML配置
- **🎮 Habitat集成**: 完全兼容Habitat仿真器环境

## 📂 项目结构

```
handwritingNav2/
├── modeling/                      # 核心建模代码
│   ├── common/                    # 通用工具和基类
│   ├── config/                    # 配置文件
│   │   ├── hwnav_base.yaml       # 基础配置
│   │   ├── train_diffusion_hwnav.yaml  # 扩散策略训练配置
│   │   └── train_hwnav.yaml      # PPO训练配置
│   ├── diffusion_policy/          # 扩散策略实现
│   │   ├── diffusion_nav_policy.py      # 主策略类
│   │   ├── conditional_unet1d.py        # 条件UNet
│   │   ├── habitat_diffusion_trainer.py # 训练器
│   │   └── normalizer.py                # 数据归一化
│   ├── models/                    # 神经网络模型
│   │   ├── visual_cnn.py         # 视觉编码器
│   │   ├── advanced_goal_predictor.py   # 目标预测器
│   │   └── rnn_state_encoder.py  # 状态编码器
│   ├── ppo/                       # PPO实现
│   └── utils/                     # 工具函数
│
├── scripts/                       # 可执行脚本
│   ├── train.py                   # 主训练脚本
│   ├── debug_env.py               # 环境调试
│   ├── download_mp.py             # 数据下载
│   └── count.py                   # 数据统计
│
├── tests/                         # 测试文件
│   ├── test_diffusion_policy.py
│   ├── test_habitat_compatibility.py
│   └── simple_test_diffusion.py
│
├── docs/                          # 文档
│   └── DIFFUSION_POLICY.md       # 扩散策略详细文档
│
├── references/                    # 参考项目
│   └── flodiff/                   # FloNa参考实现
│
├── habitat-lab/                   # Habitat仿真器
├── data/                          # 数据目录（不包含在git中）
├── requirements.txt               # Python依赖
└── README.md                      # 本文件
```

## 🚀 安装

### 1. 环境要求

- Python 3.8+
- CUDA 11.3+ (推荐用于GPU加速)
- 8GB+ GPU内存（训练时）

### 2. 安装步骤

```bash
# 克隆项目
cd /path/to/handwritingNav2

# 创建虚拟环境（推荐）
conda create -n hwnav python=3.8
conda activate hwnav

# 安装PyTorch（根据您的CUDA版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装Habitat-Sim（推荐使用conda）
conda install habitat-sim -c conda-forge -c aihabitat

# 安装其他依赖
pip install -r requirements.txt

# 安装Habitat-Lab（项目已包含）
cd habitat-lab
pip install -e .
cd ..
```

### 3. 数据准备

```bash
# 下载场景数据（Matterport3D）
# 请参考 https://niessner.github.io/Matterport/ 获取访问权限

# 数据应放置在以下位置：
# data/scene_datasets/mp3d/
# data/datasets/mp3d_hwnav/
```

## 🎯 快速开始

### 训练扩散策略模型

```bash
# 使用默认配置训练
python scripts/train.py --run-type train

# 指定配置文件
python scripts/train.py \
    --run-type train \
    --exp-config modeling/config/train_diffusion_hwnav.yaml \
    --model-dir outputs/diffusion_exp1

# 从检查点恢复训练
python scripts/train.py \
    --run-type train \
    --resume-from outputs/diffusion_exp1/data/ckpt.1000.pth
```

### 训练PPO模型

```bash
python scripts/train.py \
    --run-type train \
    --exp-config modeling/config/train_hwnav.yaml \
    --model-dir outputs/ppo_exp1
```

### 评估模型

```bash
python scripts/train.py \
    --run-type eval \
    --exp-config modeling/config/val_hwnav.yaml \
    --model-dir outputs/diffusion_exp1
```

### 测试安装

```bash
# 简单测试
python tests/simple_test_diffusion.py

# 完整测试
python tests/test_diffusion_policy.py

# Habitat兼容性测试
python tests/test_habitat_compatibility.py
```

## ⚙️ 配置说明

### 扩散策略关键参数

在 `modeling/config/train_diffusion_hwnav.yaml` 中：

```yaml
RL:
  DIFFUSION:
    horizon: 16              # 动作序列长度
    n_action_steps: 4        # 每次执行的动作步数
    n_obs_steps: 3           # 观察历史步数
    obs_dim: 512             # 观察特征维度
    action_dim: 4            # 动作维度（前进、左转、右转、停止）
    num_inference_steps: 20  # 推理时的去噪步数
    lr: 1e-4                 # 学习率
    weight_decay: 1e-4       # 权重衰减
```

### 训练参数

```yaml
NUM_PROCESSES: 1             # 并行环境数量
NUM_UPDATES: 100000          # 总更新步数
LOG_INTERVAL: 10             # 日志间隔
CHECKPOINT_INTERVAL: 500     # 检查点保存间隔
```

### 传感器配置

```yaml
SENSORS: ["DEPTH_SENSOR", 'RGB_SENSOR']
EXTRA_RGB: True              # 额外RGB传感器
EXTRA_DEPTH: True            # 深度传感器
PREDICT_GOAL: True           # 启用目标预测
```

## 🔍 参考项目

本项目参考了以下优秀工作：

- **FloNa** ([references/flodiff](references/flodiff/)): Floor Plan Guided Embodied Visual Navigation
  - Paper: [arXiv:2412.18335](https://arxiv.org/pdf/2412.18335)
  - Project: [https://gauleejx.github.io/flona/](https://gauleejx.github.io/flona/)

## 📊 算法对比

| 特性 | PPO | 扩散策略 |
|------|-----|----------|
| 动作生成 | 单步决策 | 序列生成 |
| 行为平滑性 | 可能不连续 | 平滑一致 |
| 训练稳定性 | 需要仔细调参 | 相对稳定 |
| 计算复杂度 | 较低 | 较高 |
| 序列建模能力 | 有限 | 强大 |
| 推理速度 | 快 | 较慢 |

## 🐛 常见问题

### Q: 导入错误 - "No module named 'modeling'"

**A**: 确保从项目根目录运行脚本，或使用提供的 `scripts/train.py`（已自动设置路径）。

### Q: Habitat环境初始化失败

**A**: 
1. 检查数据集路径是否正确
2. 确保Habitat-Sim正确安装
3. 检查GPU驱动和CUDA版本

### Q: 训练时内存不足

**A**: 
- 减少 `horizon` 参数（如从16改为8）
- 减少批处理大小
- 使用更小的网络架构

### Q: 配置文件路径错误

**A**: 所有配置文件现在使用相对路径。如果仍有问题，请检查：
- 是否从项目根目录运行
- 配置文件是否存在于 `modeling/config/` 目录

## 📝 开发指南

### 添加新的传感器

1. 在 `habitat-lab/habitat/` 中定义传感器
2. 在配置文件中添加传感器配置
3. 在 `modeling/models/` 中添加对应的编码器

### 修改网络架构

主要文件：
- `modeling/diffusion_policy/diffusion_nav_policy.py` - 主策略网络
- `modeling/diffusion_policy/conditional_unet1d.py` - UNet架构
- `modeling/models/visual_cnn.py` - 视觉编码器

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Habitat](https://aihabitat.org/) - 3D仿真环境
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - 扩散策略思想
- [FloNa](https://gauleejx.github.io/flona/) - 地图导航参考实现

## 📧 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**更新日期**: 2024
**版本**: 0.1.0
