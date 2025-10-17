# 项目结构说明

本文档详细说明项目的目录组织和文件职责。

## 📁 顶层目录

```
handwritingNav2/
├── modeling/           # 核心建模代码（主要开发区域）
├── scripts/            # 可执行脚本
├── tests/              # 测试代码
├── docs/               # 文档
├── references/         # 参考项目（只读）
├── habitat-lab/        # Habitat仿真器（第三方库）
├── data/               # 数据存储（gitignore）
├── utils/              # 项目级工具
├── requirements.txt    # Python依赖
├── setup.py            # 安装配置
├── README.md           # 项目说明
└── .gitignore          # Git忽略规则
```

## 🔧 modeling/ - 核心建模代码

这是项目的核心部分，包含所有机器学习模型和训练逻辑。

### modeling/common/ - 通用工具
```
common/
├── __init__.py
├── base_trainer.py           # 训练器基类
├── baseline_registry.py      # 基线模型注册
├── benchmark.py              # 性能测试
├── env_utils.py              # 环境工具
├── rollout_storage.py        # 轨迹存储
├── shortest_path_follower.py # 最短路径跟随器
├── simple_agents.py          # 简单智能体
├── sync_vector_env.py        # 同步向量环境
├── tensorboard_utils.py      # TensorBoard工具
└── utils.py                  # 通用工具函数
```

### modeling/config/ - 配置文件
```
config/
├── default.py                    # 默认配置（Python）
├── hwnav_base.yaml              # 基础配置
├── train_diffusion_hwnav.yaml   # 扩散策略训练配置
├── train_hwnav.yaml             # PPO训练配置
├── test_hwnav.yaml              # 测试配置
└── val_hwnav.yaml               # 验证配置
```

**配置文件优先级**: 命令行参数 > YAML配置 > default.py

### modeling/diffusion_policy/ - 扩散策略
```
diffusion_policy/
├── __init__.py
├── diffusion_nav_policy.py       # 主策略网络（核心）
├── conditional_unet1d.py         # 1D条件UNet
├── habitat_diffusion_trainer.py  # Habitat训练器（推荐）
├── diffusion_trainer.py          # 原始训练器
├── mask_generator.py             # 掩码生成器
└── normalizer.py                 # 数据归一化器
```

**关键文件**:
- `diffusion_nav_policy.py`: 实现扩散策略的主要逻辑
- `habitat_diffusion_trainer.py`: 完全兼容Habitat的训练器

### modeling/models/ - 神经网络模型
```
models/
├── __init__.py
├── advanced_goal_predictor.py   # 高级目标预测器（Transformer）
├── goal_predictor.py            # 简单目标预测器
├── visual_cnn.py                # 视觉编码器（ResNet）
├── audio_cnn.py                 # 音频编码器
├── rnn_state_encoder.py         # RNN状态编码器
├── vae.py                       # 变分自编码器
└── vae_model_final.pth          # 预训练VAE权重
```

### modeling/ppo/ - PPO强化学习
```
ppo/
├── __init__.py
├── ppo.py           # PPO算法实现
├── policy.py        # PPO策略网络
└── ppo_trainer.py   # PPO训练器
```

### modeling/utils/ - 建模工具
```
utils/
└── expert_actions_loader.py   # 专家动作加载器
```

## 🚀 scripts/ - 可执行脚本

用户直接运行的脚本文件。

```
scripts/
├── __init__.py
├── train.py        # 主训练脚本（重要）
├── debug_env.py    # 环境调试
├── download_mp.py  # 数据下载
└── count.py        # 数据统计
```

**使用方式**:
```bash
python scripts/train.py --run-type train
```

## 🧪 tests/ - 测试代码

所有测试和验证脚本。

```
tests/
├── __init__.py
├── test_diffusion_policy.py        # 扩散策略测试
├── test_habitat_compatibility.py   # Habitat兼容性测试
└── simple_test_diffusion.py        # 简单功能测试
```

**运行测试**:
```bash
# 单个测试
python tests/simple_test_diffusion.py

# 所有测试
pytest tests/
```

## 📚 docs/ - 文档

项目文档和说明。

```
docs/
├── DIFFUSION_POLICY.md   # 扩散策略详细文档
└── SETUP.md              # 安装配置指南
```

## 🔍 references/ - 参考项目

外部参考项目，只读，不修改。

```
references/
└── flodiff/              # FloNa项目参考实现
    ├── README.md
    ├── model/
    ├── training/
    └── ...
```

**注意**: 这个目录的代码不应该被直接使用，只作为参考。

## 🎮 habitat-lab/ - Habitat仿真器

第三方库，通常不需要修改。

```
habitat-lab/
├── habitat/              # 核心库
│   ├── core/
│   ├── tasks/
│   │   └── nav/
│   │       └── handwriting_nav_task.py  # 自定义任务
│   └── datasets/
│       └── handwriting_nav/             # 自定义数据集
├── habitat_baselines/    # 基线实现
└── ...
```

**自定义部分**:
- `habitat/tasks/nav/handwriting_nav_task.py`: 手绘导航任务定义
- `habitat/datasets/handwriting_nav/`: 数据集加载器

## 💾 data/ - 数据目录

存储所有数据，不纳入版本控制。

```
data/
├── scene_datasets/       # 场景数据
│   └── mp3d/            # Matterport3D场景
│       ├── 17DRP5sb8fy/
│       ├── 1LXtFkjw3qL/
│       └── ...
├── datasets/            # Episode数据集
│   └── mp3d_hwnav/
│       ├── train.json.gz
│       ├── val_seen.json.gz
│       └── val_unseen.json.gz
└── models/              # 保存的模型
    └── outputs/
```

## 🔧 utils/ - 项目级工具

项目级的工具函数。

```
utils/
├── filtre/              # 过滤工具
└── utils_fmm/           # FMM相关工具
```

## 📝 配置文件

### requirements.txt
Python依赖包列表，用于 `pip install -r requirements.txt`

### setup.py
项目安装配置，用于 `pip install -e .`

### .gitignore
Git版本控制忽略规则：
- Python编译文件 (`__pycache__/`, `*.pyc`)
- 数据文件 (`data/`, `*.pth`)
- 日志文件 (`*.log`, `tb/`)
- IDE配置 (`.vscode/`, `.idea/`)

## 🎯 开发工作流

### 1. 添加新功能

```
1. 在 modeling/ 中实现核心逻辑
2. 在 tests/ 中添加测试
3. 在 docs/ 中更新文档
4. 更新 README.md
```

### 2. 训练新模型

```
1. 在 modeling/config/ 创建配置文件
2. 使用 scripts/train.py 训练
3. 模型保存在 data/models/
4. 日志记录在 TensorBoard
```

### 3. 调试问题

```
1. 使用 scripts/debug_env.py 检查环境
2. 运行 tests/ 中的测试
3. 启用 DEBUG 模式
4. 查看日志文件
```

## 📊 文件数量统计

```bash
# 统计各目录文件数
find modeling/ -name "*.py" | wc -l      # Python文件
find tests/ -name "*.py" | wc -l         # 测试文件
find modeling/config/ -name "*.yaml" | wc -l  # 配置文件
```

## 🔗 重要文件快速索引

| 类别 | 文件 | 说明 |
|------|------|------|
| 入口 | `scripts/train.py` | 主训练脚本 |
| 配置 | `modeling/config/train_diffusion_hwnav.yaml` | 扩散策略配置 |
| 核心 | `modeling/diffusion_policy/diffusion_nav_policy.py` | 扩散策略主类 |
| 训练 | `modeling/diffusion_policy/habitat_diffusion_trainer.py` | 训练器 |
| 模型 | `modeling/models/visual_cnn.py` | 视觉编码器 |
| 测试 | `tests/simple_test_diffusion.py` | 快速测试 |
| 文档 | `README.md` | 项目说明 |

## 📌 注意事项

1. **不要修改 references/** - 这是参考代码，保持原样
2. **数据文件很大** - 不要提交到Git
3. **配置使用相对路径** - 确保可移植性
4. **测试先行** - 修改代码前运行测试
5. **文档同步** - 修改代码时更新文档

---

**最后更新**: 2024
**维护者**: HandWriting Nav Team

