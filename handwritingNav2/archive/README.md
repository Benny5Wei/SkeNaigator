# Archive 归档目录

本目录存放项目开发过程中产生的旧文件、已废弃的代码和过时的文档。

## 归档时间

2024年10月13日

## 归档内容

### 旧文档（Old Documentation）

#### 根目录旧文档
- `CHECKLIST.md` - 旧的检查清单
- `REORGANIZATION_SUMMARY.md` - 项目重组总结（已过时）
- `PROJECT_STRUCTURE.md` - 旧的项目结构说明
- `STRUCTURE_TREE.txt` - 旧的目录树
- `DATA_STATUS.md` - 旧的数据状态文档

#### old_docs/ 子目录
- `DATA_REQUIREMENTS.md` - 旧的数据要求文档
- `DIFFUSION_POLICY.md` - 旧的扩散策略说明
- `MULTI_GPU_TRAINING.md` - 旧的多GPU训练指南
- `QUICK_START.md` - 旧的快速开始指南
- `SETUP.md` - 旧的设置指南
- `TROUBLESHOOTING_EGL.md` - EGL问题排查指南

### 旧工具和脚本（Old Tools & Scripts）

#### filtre/
数据集过滤和处理脚本（已废弃）
- `count_scene_ids.py`
- `filtre_15.py`
- `filtre_correspond.py`
- `filtre_episode_id.py`
- `filtre_plan.py`
- `revise_val.py`
- `split_dataset.py`
- `update_radius.py`

#### flona_dataset/
旧的数据集生成脚本（已被新的生成脚本替代）
- `4generating.py`
- `generate_realtime.py`
- `run_agent_depth.py`

#### utils_fmm/
FMM（Fast Marching Method）规划工具
- `control_helper.py`
- `depth_utils.py`
- `fmm_planner.py`
- `mapper.py`
- `mapping.py`
- `model.py`
- `pose_utils.py`
- `rotation_utils.py`

## 当前项目使用的文档

项目根目录下的最新文档：

- **`README.md`** - 项目主文档
- **`DIFFUSION_QUICKSTART.md`** - 扩散策略快速开始指南 ⭐
- **`EXECUTION_CHECKLIST.md`** - 执行清单 ⭐
- **`CHANGES_SUMMARY.md`** - 最新修改总结

## 注意事项

1. 本目录中的文件仅作为历史参考，不应在当前项目中使用
2. 如需恢复某些功能，请先评估是否与当前架构兼容
3. 建议定期清理过期的归档内容

## 归档原因

为了保持项目结构清晰，减少混淆，将以下内容归档：
- 已被新文档替代的旧文档
- 不再使用的工具和脚本
- 与当前离线学习架构不兼容的在线学习相关代码

项目已从**在线强化学习（PPO）**架构迁移到**离线学习（扩散策略）**架构，因此许多旧的工具和文档已不再适用。






