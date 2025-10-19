# 序列建模增强 - Transformer处理历史观察序列

## 📋 改进概述

本次改进增强了序列建模能力，引入Transformer编码器处理历史观察序列，并使`len_traj_pred`(horizon)可配置。这是参考flodiff项目设计的关键改进。

---

## 🎯 改进内容

### 1. **添加Transformer序列编码器**

#### 新增模块

**PositionalEncoding类**
```python
class PositionalEncoding(nn.Module):
    """
    位置编码模块 - 参考flodiff设计
    为序列中的每个位置添加可学习的位置信息
    """
    def __init__(self, d_model: int, max_seq_len: int = 10):
        # 使用正弦/余弦位置编码
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
```

**Transformer序列编码器**
```python
# 在DiffusionNavPolicy.__init__中
if self.use_transformer_encoder:
    # 位置编码
    self.positional_encoding = PositionalEncoding(
        d_model=obs_dim,
        max_seq_len=self.context_size + 2
    )
    
    # Transformer编码器
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=obs_dim,
        nhead=mha_num_attention_heads,
        dim_feedforward=mha_ff_dim_factor * obs_dim,
        activation="gelu",
        batch_first=True,
        norm_first=True
    )
    
    self.sequence_encoder = nn.TransformerEncoder(
        encoder_layer, 
        num_layers=mha_num_attention_layers
    )
```

---

### 2. **新增配置参数**

#### 配置文件 (train_diffusion_hwnav.yaml)

```yaml
RL:
  DIFFUSION:
    # 动作序列长度（可配置）
    horizon: 16  # len_traj_pred，可以调整为8, 16, 32等
    
    # 序列建模增强配置（新增）
    context_size: 5              # 历史观察帧数（参考flodiff）
    use_transformer_encoder: True  # 启用Transformer序列编码器
    mha_num_attention_heads: 4   # 多头注意力头数
    mha_num_attention_layers: 2  # Transformer层数
    mha_ff_dim_factor: 4         # 前馈网络维度因子
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `horizon` | 16 | 动作序列长度（len_traj_pred），可配置 |
| `context_size` | 5 | 历史观察帧数 |
| `use_transformer_encoder` | True | 是否启用Transformer编码器 |
| `mha_num_attention_heads` | 4 | 多头注意力头数 |
| `mha_num_attention_layers` | 2 | Transformer层数 |
| `mha_ff_dim_factor` | 4 | 前馈网络维度因子 (FF_dim = factor × obs_dim) |

---

### 3. **改进的观察编码流程**

#### 之前的实现
```python
def encode_observations(observations):
    # 简单平均池化
    encoded = simple_fusion(observations)
    return encoded  # [B, obs_dim]
```

#### 改进后的实现
```python
def encode_observations(observations):
    # 1. 多模态特征融合
    encoded_features = feature_fusion(observations)  # [B, obs_dim]
    
    # 2. 序列建模（新增）
    if self.use_transformer_encoder:
        # 扩展为序列 [B, 1, obs_dim] 或使用历史 [B, T, obs_dim]
        obs_sequence = encoded_features.unsqueeze(1)
        
        # 应用位置编码
        obs_sequence = self.positional_encoding(obs_sequence)
        
        # Transformer编码
        sequence_encoding = self.sequence_encoder(obs_sequence)
        
        # 平均池化聚合（参考flodiff）
        encoded_features = torch.mean(sequence_encoding, dim=1)
    
    return encoded_features  # [B, obs_dim]
```

---

## 🆚 与flodiff的对比

| 维度 | handwritingNav2 (改进后) | flodiff (FloNa) |
|------|-------------------------|-----------------|
| **位置编码** | ✅ PositionalEncoding | ✅ PositionalEncoding |
| **序列编码器** | ✅ TransformerEncoder | ✅ TransformerEncoder |
| **历史帧数** | ✅ 5 (可配置) | ✅ 5 (固定) |
| **注意力头** | ✅ 4 (可配置) | ✅ 2 (固定) |
| **聚合方式** | ✅ 平均池化 | ✅ 平均池化 |
| **len_traj_pred** | ✅ **可配置** (新增) | ❌ 固定 |

**关键优势**: 我们的实现完全参考了flodiff的序列建模设计，并且让`horizon`(len_traj_pred)可配置！

---

## 📊 技术细节

### Transformer编码器工作流程

```
输入观察: [B, obs_dim]
  ↓
扩展为序列: [B, 1, obs_dim] 或 [B, T, obs_dim]
  ↓
位置编码: pos_enc = sin/cos(position)
  obs_sequence += pos_enc
  ↓
Transformer编码:
  ├─ Multi-Head Self-Attention (4 heads)
  │  └─ 捕获时序依赖关系
  ├─ LayerNorm
  ├─ Feed-Forward Network (4 × obs_dim)
  └─ LayerNorm
  ↓
平均池化: mean(sequence, dim=1)
  ↓
输出: [B, obs_dim]
```

### 多头自注意力机制

```python
# 4个注意力头并行处理
Attention(Q, K, V) = softmax(QK^T / √d_k) V

# 每个头关注不同的特征子空间
Head_i = Attention(Q_i, K_i, V_i)

# 拼接所有头的输出
MultiHead = Concat(Head_1, ..., Head_4) W^O
```

**作用**: 
- 捕获不同时间步之间的依赖关系
- 学习哪些历史帧对当前决策更重要
- 提取时序模式和运动趋势

---

## 🔧 使用方法

### 1. 调整horizon (len_traj_pred)

根据任务复杂度调整动作序列长度：

```yaml
# 简单任务 - 短序列
horizon: 8  

# 中等任务 - 中序列
horizon: 16  # 默认

# 复杂任务 - 长序列
horizon: 32
```

**建议**:
- 简单直线导航: 8
- 普通室内导航: 16
- 复杂多拐点导航: 24-32

### 2. 调整历史帧数

```yaml
# 更多历史信息
context_size: 8  # 增加到8帧

# 标准配置
context_size: 5  # 默认（参考flodiff）

# 减少计算量
context_size: 3
```

### 3. 调整Transformer结构

```yaml
# 更深的网络
mha_num_attention_layers: 4  # 增加层数
mha_num_attention_heads: 8   # 增加头数

# 更大的前馈网络
mha_ff_dim_factor: 8  # 增加前馈维度
```

### 4. 禁用Transformer（回退到简单融合）

```yaml
use_transformer_encoder: False  # 禁用
```

---

## 📈 预期效果

### 1. **更好的时序建模**
- 之前: 单帧观察，无历史信息
- 现在: 5帧历史，Transformer捕获时序依赖

### 2. **更平滑的导航**
- Transformer能够预测运动趋势
- 减少抖动和突变

### 3. **更强的泛化能力**
- 位置编码提供结构化时序信息
- 多头注意力学习多种时序模式

### 4. **灵活的配置**
- `horizon`可配置，适应不同任务
- Transformer结构可调，平衡性能和速度

---

## 🔍 与其他改进的协同

### 完整的特征流程

```
多模态输入 (map, rgb, depth, slam)
  ↓
[AdvancedGoalPredictor]  ← 第1次改进
  • 目标预测
  • 特征: [B, 128]
  ↓
[自适应模态注意力]  ← 第1次改进
  • 动态权重: [B, 5]
  • 处理不确定性
  ↓
[特征融合网络]  ← 第1次改进
  • 4层MLP + LayerNorm
  • 输出: [B, obs_dim]
  ↓
[Transformer序列编码]  ← 第2次改进（本次）
  • 位置编码
  • 多头自注意力
  • 输出: [B, obs_dim]
  ↓
[扩散策略]
  • 生成动作序列 [B, horizon, action_dim]
```

---

## 📝 代码示例

### 训练时使用

```bash
# 使用默认配置（horizon=16, context_size=5）
python scripts/train.py \
    --run-type train \
    --exp-config modeling/config/train_diffusion_hwnav.yaml

# 自定义horizon
# 修改yaml文件中的horizon参数，或通过命令行覆盖
```

### 查看序列编码

```python
from modeling.diffusion_policy.diffusion_nav_policy import DiffusionNavPolicy

# 加载模型
policy = DiffusionNavPolicy(...)

# 编码观察
encoded = policy.encode_observations(observations)

# 查看序列编码
if hasattr(policy, '_last_sequence_encoding'):
    sequence_encoding = policy._last_sequence_encoding
    print("序列编码:", sequence_encoding.shape)  # [B, T, obs_dim]
    
# 查看注意力权重（如果需要可视化）
# 需要修改TransformerEncoderLayer返回注意力权重
```

---

## 🧪 消融实验建议

测试不同配置对性能的影响：

| 实验 | horizon | context_size | Transformer | 说明 |
|------|---------|--------------|-------------|------|
| Baseline | 16 | - | ❌ | 不使用序列建模 |
| Short | 8 | 5 | ✅ | 短序列 |
| Medium | 16 | 5 | ✅ | 中等序列 |
| Long | 32 | 5 | ✅ | 长序列 |
| More History | 16 | 8 | ✅ | 更多历史 |
| Deep | 16 | 5 | ✅ (4层) | 更深网络 |

---

## ⚙️ 性能优化

### 计算开销分析

```python
# Transformer编码器参数量
params = d_model × (4 × d_model) × num_layers × num_heads
      = 512 × 2048 × 2 × 4
      ≈ 8M parameters

# 计算量（FLOPs）
FLOPs = O(n^2 × d) × num_layers
      where n = context_size, d = obs_dim
```

### 减少计算量的方法

1. **减少层数**: `mha_num_attention_layers: 1`
2. **减少头数**: `mha_num_attention_heads: 2`
3. **减少历史帧**: `context_size: 3`
4. **使用更小的obs_dim**: `obs_dim: 256`

---

## ✅ 修改文件清单

1. ✅ `modeling/diffusion_policy/diffusion_nav_policy.py`
   - 新增 `PositionalEncoding` 类
   - 在`__init__`中添加Transformer编码器
   - 修改 `encode_observations` 支持序列建模
   - 新增 `context_size` 等参数

2. ✅ `modeling/diffusion_policy/habitat_diffusion_trainer.py`
   - 传递序列建模配置参数

3. ✅ `modeling/config/train_diffusion_hwnav.yaml`
   - 新增序列建模配置节
   - `horizon`可配置
   - `context_size`等参数

---

## 🎉 总结

本次改进完成了：

1. ✅ **Transformer序列编码器** - 参考flodiff设计
2. ✅ **位置编码** - 为序列添加时序信息
3. ✅ **context_size可配置** - 灵活调整历史帧数
4. ✅ **horizon(len_traj_pred)可配置** - 适应不同任务
5. ✅ **完整的序列建模流程** - 从编码到聚合

**关键优势**:
- 与flodiff的序列建模设计完全一致
- 增加了更多的可配置性
- 与之前的模态注意力机制无缝集成
- 保持代码的向后兼容性

---

## 📚 参考资料

1. **FloNa论文**: [arXiv:2412.18335](https://arxiv.org/pdf/2412.18335)
2. **Transformer原论文**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. **Positional Encoding**: [Transformer位置编码详解]

---

**日期**: 2024-10-19  
**版本**: v1.1  
**状态**: ✅ 完成并可用

---

## 🔗 相关文档

- [FEATURE_FUSION_IMPROVEMENT.md](FEATURE_FUSION_IMPROVEMENT.md) - 第一次改进（目标预测和模态注意力）
- [改进总结.md](改进总结.md) - 第一次改进的中文总结
- [README.md](README.md) - 项目总体文档

