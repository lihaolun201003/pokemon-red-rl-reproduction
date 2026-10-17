# 训练进度（Pokémon Red PPO，实验 seed_0）— 已完结

> 本文件由 RL Training Supervisor 维护。新会话接管顺序：**本文件 → `docs/reports/training_handoff.md` → `artifacts/training/seed_0/training_state.json` → `docs/reports/training_final_report.md`**，并检查真实进程/日志/checkpoint/磁盘后，再决定是否有用户授权的新任务。
> 最后更新：2026-10-18 01:40（Asia/Shanghai）

## 1. 总任务状态：本轮已完成

| 项目 | 最终值 |
|---|---|
| 总任务完成度 | **100%**（训练 1M 步完成、Checkpoint 管理完成、独立评估完成、监督/交接/最终报告完成） |
| 累计真实训练环境步 | **1,000,000 / 1,000,000（预算用尽，未超出）** |
| 评估环境步（独立统计，非训练预算） | 307,328（protocol v1 + v1-ext + CUDA 冒烟） |
| 训练时间线 | 启动 2026-10-18 00:23:31 → 完成 00:50:43（墙钟 **27.2 分钟**；纯计算 1,539.98 s） |
| 平均吞吐 | 649.4 steps/s（时间加权）/ 675.3（逐 update 均值），范围 322.9–873.9 |
| 阶段 | A（10k）→ B（100k）→ C（300k）→ D（1M）全部自然完成 `stop_reason=step_target_reached` |
| 12 小时上限 | 远未触及（训练仅 27 分钟） |
| 主推最佳模型 | `artifacts/training/seed_0/stage_d_1m/checkpoint_001000000.pt`（v1-ext n=5 最佳：reward 3.83±2.77、coords 205.4±25.6、maps 4.0） |
| 次优模型 | `artifacts/training/seed_0/stage_c_300k/checkpoint_000300000.pt`（v1 n=3 最高：4.18±2.47 / 169±35 / 3.67；v1-ext 次高 3.09 / 197.6） |
| 训练进程 | 已全部退出，无残留；GPU 空闲；未运行任何后台训练 |
| Git | 本地提交（不 push）；ROM/权重/视频/artifacts 均本地，未上传 |

## 2. 最终结论（诚实版）

1. 训练管线稳定、可复现：真实 ROM + 官方初始存档 + PyBoy + CUDA PPO，1M 步全程 0 NaN/Inf、0 次 checkpoint 失败、0 条 supervisor 告警；仅 Stage A 首次启动发生一次 `nvcuda64.dll` 原生崩溃，重试成功（详见最终报告 §11）。
2. 学习信号真实但有限：训练 episode 奖励 B 阶段均值 0.74 → D 阶段 5.33；探索坐标 episode 级最大 430；队伍最大 6 只、等级和最大 20、出现升级/捕捉/濒死。**但没有获得任何徽章，未到 Mt Moon/Cerulean**；在独立评估的 8192 步视界内最远到 Pallet Town / Route 1（训练 episode 中偶发到 Viridian 一带）。
3. 与基线对比：v1-ext 中 1M/300k 探索坐标 205/198、地图 4.0，优于 untrained（134 / 2.2）与随机动作基线（73 / 2.0）；训练模型 5/5 episode 能离开出生房间，untrained 仅 2/5。但 n=3–5 样本小、方差大，不做显著性宣称。
4. 重要发现：所有 checkpoint 的确定性 argmax 轨迹退化为单键粘滞（coords=1），确定性指标无区分度；探索行为只在随机采样下出现。500k checkpoint 在 v1 与 v1-ext 中一致地明显回落（研究观察，未改算法）。
5. 建议：经用户授权后可扩大训练规模（从 1M checkpoint 继续，10M→30M→100M 增量 + 阶段评估）；当前 1M 步仅约论文完整配置（~3.2×10⁹ 步）的 0.03%。

## 3. 阶段运行表（真实数据）

| 阶段 | 累计目标 | 实际结束 | updates | 计算时间 | 平均吞吐 | 完成 episode | 里程碑 Checkpoint（已验证 loadable） |
|---|---:|---:|---:|---:|---:|---:|---|
| A 稳定性 | 10,000 | 10,000 | 5 | 18.3 s | 552.5 steps/s | 0 | `stage_a_10k/checkpoint_000010000.pt` |
| B | 100,000 | 100,000 | 45 | 196.6 s | 475.6 steps/s | 8 | `stage_b_100k/checkpoint_000100000.pt` |
| C | 300,000 | 300,000 | 100 | 343.2 s | 617.6 steps/s | 10 | `stage_c_300k/checkpoint_000300000.pt` |
| D | 1,000,000 | 1,000,000 | 350 | 981.9 s | 719.2 steps/s | 48 | `stage_d_1m/checkpoint_000500000.pt`（额外评估点）、`checkpoint_001000000.pt` |
| 合计 | 1,000,000 | 1,000,000 | 500 | 1,539.98 s | 649.4 steps/s | 66 | 5 个里程碑已 verify（verify_summary.json 5/5 ok） |

## 4. Checkpoint 管理

- 每 5 updates（10,000 步）保存；tmp + 原子 replace；保存后立即读回校验模型/优化器/步数；每阶段最后一步必存。
- 5 个里程碑全部 CPU/官方模型重建 strict load 验证通过（`artifacts/training/seed_0/monitor/verify_summary.json`）。
- 旧 checkpoint 已移动到各 stage 的 `_archive/`（stage_c 10 个、stage_d 59 个；**移动不删除**，保留最近 10 个 + 里程碑）。
- 所有 checkpoint 在本地 `artifacts/`（gitignored），未上传。

## 5. 异常记录（摘要）

| 时间 | 事件 | 处理/结果 |
|---|---|---|
| 00:15–00:19 | 20-worker 基准内存过载、nvidia-smi 短暂失联 | 终止变体，固定 8 worker |
| 00:22:11 | Stage A 首次启动 nvcuda64.dll 原生崩溃（BEX64 0xc0000409） | 归档 `_failed/stage_a_10k_attempt1/`，确认 CUDA 健康后重试成功 |
| 全程 | NaN/Inf、死进程、checkpoint 失败、OOM、磁盘不足 | 未发生；supervisor 139 快照 0 告警 |
| 评估期 | CUDA 冒烟 + 11 次评估串行 | 无驱动异常；deterministic 退化是策略性质，已单列 |

## 6. 下一步（需要用户明确授权）

1. 若用户授权扩大预算：从 `checkpoint_001000000.pt` 继续，建议保持 `configs/train_seed0.yaml`（8×250、论文超参）分阶段 10M → 30M → 100M，每阶段固定协议评估（`scripts/eval_checkpoint.py`）；用 `scripts/supervisor.py` 监督。
2. 若用户只要当前结果：本轮已交付完成，读取 `docs/reports/training_final_report.md` 即可；观看命令见报告 §10。
3. **未经用户同意不要自动继续训练或扩大预算**；不修改 PPO/奖励/模型/观测/动作；不删除既有 checkpoint；不 push GitHub。
