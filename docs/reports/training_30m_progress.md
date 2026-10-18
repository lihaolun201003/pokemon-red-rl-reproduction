# 30M 步连续训练进度（实验 seed_0 — Stage E / Stage F）

> 本文件由训练监督 Agent 持续维护，会话中断后以 `artifacts/training/seed_0/training_state.json` + 本文件 + `docs/reports/training_handoff.md` 恢复。
> 任务：在已有 1,000,000 步基础上，**累计扩至 30,000,000 真实环境步**（Stage E 1M→10M，Stage F 10M→30M），PPO/奖励/网络/动作/配置不变，不并行训练。

## 1. 任务状态

| 项目 | 值 |
|---|---|
| 授权 | 用户明确授权：训练预算 1,000,000 → 30,000,000 环境步 |
| 起始 Checkpoint | `artifacts/training/seed_0/stage_d_1m/checkpoint_001000000.pt`（已验证：phase1_format=1、next_update=500、env_steps=1,000,000、model strict load OK、optimizer 非空、config resume 兼容） |
| 训练配置 | `artifacts/training/seed_0/stage_d_1m/config.yaml`（8 workers × 250 steps = 2,000 步/update；epochs 3；minibatches 8；lr 3e-4；gamma 0.997；lambda 0.95；clip 0.2；beta 0；seed 0；CUDA） |
| 检查点间隔 | 每 5 updates（10,000 步）原子保存 + 读回校验 |
| 预算上限 | 30,000,000 步（不允许超出） |
| 里程碑保留 | 1M、10M、20M、30M |
| Stage E | 1M → 10M，out=`artifacts/training/seed_0/stage_e_10m` |
| Stage F | 10M → 30M，out=`artifacts/training/seed_0/stage_f_30m`（Stage E 验收成功后自动开始，不问用户） |
| 当前阶段 | **Stage E 运行中**（启动 2026-10-18 01:49:20，PID 16136） |

## 2. 启动前检查（2026-10-18 01:49，全部通过）

| 检查 | 结果 |
|---|---|
| GPU | RTX 4060 Laptop 8 GB，空闲显存 6.25 GiB，温度 50 °C |
| RAM | 15.7 GB 总内存，可用 5.3 GiB；swap 1.67 GiB |
| 磁盘 | C: 剩余 619.4 GB（3000 个 checkpoint ≈ 76 GB，充足） |
| 遗留训练/评估进程 | 无 |
| 1M checkpoint 可加载 | 通过（model strict load、2,086,600 参数、finite、optimizer 28 组、config resume_compatible=true） |
| 配置与 checkpoint 兼容 | 通过（唯一 benign 差异 `reset_params/monitor_more_stats`，环境 wrapper 运行时注入） |
| STOP 文件 | 不存在（如存在则不得忽略用户停止请求） |
| 新阶段输出目录 | `stage_e_10m`、`stage_f_30m` 均不存在（不会覆盖旧实验） |
| 代码/日志路径 | 正常；日志写入 `artifacts/training/seed_0/logs/stage_{e_10m,f_30m}.{stdout,stderr}.log` |

## 3. 阶段进度

| 阶段 | 起始累计 | 目标累计 | 新增步数 | 预计耗时* | 状态 |
|---|---:|---:|---:|---:|---|
| Stage E | 1,000,000 | 10,000,000 | 9,000,000 | 约 3.5–4.5 h | 🔄 运行中 |
| Stage F | 10,000,000 | 30,000,000 | 20,000,000 | 约 8–9 h | 待启动（E 验收后自动） |

\* 按上一轮实测吞吐 649–750 steps/s 估算，实际以监控为准。

## 4. 监控方式与指标

- 进程/阶段：`scripts/supervisor.py status` / `watch`（每 2 分钟快照，`artifacts/training/seed_0/monitor/status.jsonl`）
- 指标：`metrics.jsonl`（PPO loss/value loss/entropy/KL/clip/lr/reward、game/*、actions/share_*、吞吐）、`episodes.jsonl`、`result.json`
- 重点关注（沿用 1M 轮次经验）：
  1. reward 是持续改善还是波动；
  2. 探索坐标是否停滞；
  3. 动作熵是否持续下降/异常波动；
  4. 是否反复卡墙/菜单循环/单键粘滞；
  5. 是否真正推进剧情（badges/map/events），而不是只刷探索奖励；
  6. 500k 式性能回落是否在 10M/30M 再次出现；
  7. NaN/Inf、OOM、GPU 温度（>85 °C 暂停）、RAM 可用（<1.5 GiB 告警）、checkpoint 完整性、吞吐异常下降。
- 评估：10M、30M 各做一次 **v1-ext** 固定协议（8192 步 × 5 stochastic + 1 deterministic，seed 100000，每 episode 新建 PyBoy），与 untrained/300k/500k/1M/10M/30M 对比；长 episode 协议单独标注版本，不与 v1-ext 混比。

## 5. 异常与处理记录

| 时间 | 事件 | 处理 | 结果 |
|---|---|---|---|
| 2026-10-18 01:49 | 启动前检查 | 全部通过 | Stage E 正常启动 |
| （待记录） | | | |

## 6. 下一步

1. Stage E 运行中：定期读取 supervisor/日志，记录异常；发现异常按「保留最近完整 checkpoint → 新目录 resume」恢复，不重新初始化模型。
2. 达到 10,000,000 步：确认 `result.json.stop_reason=step_target_reached`、`environment_steps=10000000`；验证 `checkpoint_010000000.pt`；v1-ext 独立评估；写 10M 阶段报告；**自动启动 Stage F**。
3. 达到 30,000,000 步：同样验收（保留 20M/30M 里程碑）；v1-ext 评估；生成 `docs/reports/training_30m_final_report.md`；停止，不超出预算。
4. 最终：更新 `training_state.json`、本文件、`training_handoff.md`；本地 git 提交（**不 push**）。
