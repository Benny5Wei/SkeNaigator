# 扩散策略导航模型

本项目已成功将原有的PPO导航模型替换为基于扩散策略的方法。扩散策略是一种新的强化学习方法，通过学习动作序列的分布来生成更平滑、更一致的行为。

## 主要特性

### 1. 扩散策略核心组件

- **DiffusionNavPolicy**: 主要的扩散策略类，整合了视觉编码、目标预测和扩散去噪过程
- **ConditionalUnet1D**: 1D条件UNet网络，用于噪声预测
- **DiffusionTrainer**: 扩散策略训练器，替换了原有的PPO训练器
- **LinearNormalizer**: 数据归一化器，确保训练稳定性

### 2. 关键优势

- **序列建模**: 直接学习动作序列的分布，而不是单步动作
- **平滑行为**: 生成的动作序列更加平滑和一致
- **条件生成**: 基于观察条件生成动作，支持多模态输入
- **可扩展性**: 易于添加新的传感器或修改网络架构

## 文件结构

```
diffusion_policy/
├── __init__.py                    # 模块初始化
├── diffusion_nav_policy.py        # 主要扩散策略类
├── conditional_unet1d.py          # 1D条件UNet网络
├── diffusion_trainer.py           # 原始扩散策略训练器
├── habitat_diffusion_trainer.py   # Habitat兼容扩散策略训练器
├── mask_generator.py              # 掩码生成器
└── normalizer.py                  # 数据归一化器

config/
├── train_diffusion_hwnav.yaml     # 扩散策略训练配置
└── default.py                     # 更新了扩散策略配置

test_diffusion_policy.py           # 基础测试脚本
test_habitat_compatibility.py      # Habitat兼容性测试脚本
```

## 使用方法

### 1. 训练扩散策略模型

```bash
python run.py --run-type train --exp-config config/train_diffusion_hwnav.yaml
```

### 2. 评估模型

```bash
python run.py --run-type eval --exp-config config/train_diffusion_hwnav.yaml --model-dir path/to/model
```

### 3. 测试模型

```bash
# 基础功能测试
python test_diffusion_policy.py

# Habitat兼容性测试
python test_habitat_compatibility.py
```

## 配置参数

### 扩散策略参数 (config/train_diffusion_hwnav.yaml)

```yaml
RL:
  DIFFUSION:
    horizon: 16              # 动作序列长度
    n_action_steps: 4        # 实际执行的动作步数
    n_obs_steps: 3           # 观察步数
    obs_dim: 512             # 观察特征维度
    action_dim: 4            # 动作维度
    num_inference_steps: 20  # 推理步数
    obs_as_global_cond: True # 使用全局条件
    lr: 1e-4                 # 学习率
    weight_decay: 1e-4       # 权重衰减
```

## 核心算法

### 1. 扩散过程

扩散策略基于去噪扩散概率模型(DDPM)：

1. **前向过程**: 向动作序列添加高斯噪声
2. **反向过程**: 学习预测并去除噪声
3. **条件生成**: 基于观察条件生成动作序列

### 2. 网络架构

- **视觉编码器**: 处理RGB、深度、手绘地图等多模态输入
- **目标预测器**: 从手绘地图预测目标位置
- **条件UNet**: 基于观察条件生成动作序列
- **噪声调度器**: 控制扩散过程中的噪声水平

### 3. 训练过程

1. **数据收集**: 从环境中收集观察-动作轨迹
2. **归一化**: 对观察和动作数据进行归一化
3. **扩散训练**: 训练UNet预测噪声
4. **条件采样**: 基于观察条件生成动作序列

## 与PPO的对比

| 特性 | PPO | 扩散策略 |
|------|-----|----------|
| 动作生成 | 单步决策 | 序列生成 |
| 行为平滑性 | 可能不连续 | 平滑一致 |
| 训练稳定性 | 需要调参 | 相对稳定 |
| 计算复杂度 | 较低 | 较高 |
| 序列建模 | 有限 | 强大 |

## 注意事项

1. **内存使用**: 扩散策略需要更多内存来存储动作序列
2. **训练时间**: 由于需要处理序列数据，训练时间可能更长
3. **超参数**: 需要仔细调整扩散相关的超参数
4. **数据质量**: 需要高质量的轨迹数据用于训练

## 故障排除

### 常见问题

1. **内存不足**: 减少`horizon`或`batch_size`
2. **训练不稳定**: 调整学习率和权重衰减
3. **动作不连续**: 检查归一化设置
4. **收敛慢**: 增加训练步数或调整网络架构

### 调试建议

1. 使用`test_diffusion_policy.py`验证模型
2. 检查数据归一化是否正确
3. 监控训练损失和生成的动作质量
4. 使用tensorboard可视化训练过程

## 未来改进

1. **多模态融合**: 改进多传感器数据的融合方式
2. **目标预测**: 集成更先进的目标预测算法
3. **在线学习**: 支持在线学习和适应
4. **效率优化**: 优化推理速度和内存使用

## 参考文献

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Handwriting Navigation with Diffusion Policy](https://github.com/your-repo/flodiff)
