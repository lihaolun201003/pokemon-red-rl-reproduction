# 训练监督 Phase 1：训练前环境审查（Pokémon Red PPO，seed_0）

- 审查时间：2026-10-18 00:10–00:25（Asia/Shanghai，UTC+8）
- 审查执行：RL Training Supervisor（本会话）
- 项目：`C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`
- 结论：**基础环境、真实 ROM/初始存档、CUDA、Checkpoint/续训与日志链路均可用；已按本机内存约束完成并行规模基准测试，可以启动正式训练实验 `artifacts/training/seed_0/`。**

本报告只记录实际执行的检查与实测数据；未重新搭建环境、未重新下载 ROM、未修改 PPO 数学逻辑、奖励函数、模型架构、观测或动作定义。

## 1. 环境快照

| 项目 | 实测值 | 验证方式 |
|---|---|---|
| 操作系统 | Windows 11 家庭版中文版 10.0.26200 | `Get-CimInstance Win32_OperatingSystem` |
| CPU | 13th Gen Intel Core i9-13900HX，24 核 / 32 逻辑线程 | `Win32_Processor` |
| 内存 | 15.7 GiB 物理内存；审查时可用 3.5–7.7 GiB（多应用占用，随系统波动） | psutil / `Win32_OperatingSystem` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB 显存，驱动 592.82 | `nvidia-smi` |
| GPU 空闲显存 | 约 6.3–6.4 GiB（桌面应用已占约 1.5 GiB） | `nvidia-smi` |
| 磁盘 C: | 剩余约 622 GB | `Get-PSDrive` |
| Python | 3.10.11（项目内 `.venv\Scripts\python.exe`） | `check_cuda.py` |
| PyTorch | 2.5.1+cu121，`torch.cuda.is_available() == True` | `scripts/check_cuda.py` |
| PyBoy / Gymnasium | 2.4.0 / 0.29.0 | `doctor.py` / pip |
| TensorBoard | 2.18.0（`torch.utils.tensorboard` 可用） | import 测试 |
| 其他依赖 | opencv 4.10.0、psutil 6.1.1、matplotlib 3.10.9、mediapy、Pillow 12.3.0 | import 测试 |
| 当前 Git | `5f1cb4a`（Phase1 验收提交）；本轮新增配置/日志补丁在启动实验前提交 | `git log` |

## 2. 训练前审查清单（10 项）

| # | 检查项 | 结果 | 证据 |
|---|---|---|---|
| 1 | ROM 与初始存档存在、兼容 | **通过**。`local\PokemonRed.gb` 1,048,576 B，SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`（与官方英文版一致）；`local\has_pokedex_nballs.state` SHA256 `0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e` | `Get-FileHash`；`scripts/train.py` 启动前 `validate_inputs` 强制校验 |
| 2 | CUDA 仍可用 | **通过**。CUDA True，CUDA 版本 12.1；CPU/CUDA 官方网络前后向均 passed（2,086,600 参数） | `scripts/check_cuda.py` |
| 3 | 是否已有同项目训练进程 | **无**。审查时与启动前均无 python/pyboy 训练进程在跑 | `Get-Process` 多次检查 |
| 4 | GPU 显存、CPU 核心、内存 | 8 GiB 显存（空闲约 6.4 GiB）；24 核/32 线程；15.7 GiB 物理内存，可用余量紧张（见第 4 节基准） | `nvidia-smi`、psutil |
| 5 | 剩余磁盘空间 | C: 剩余约 622 GB；单个 checkpoint 约 25 MB，1M 步按每 10k 步保存约 100 个 ≈ 2.5 GB，余量充足 | `Get-PSDrive` |
| 6 | Checkpoint 保存策略 | `scripts/train.py` 每 `checkpoint_interval` 个 update 或最终步保存；**临时文件 + 原子 replace**（`reproduction/runtime.py:save_checkpoint`）；保存后立即 `read_checkpoint` 读回并逐参数比对模型、校验 `optimizer` 状态与 `environment_steps` | 源码审查 + 既有 smoke 日志 |
| 7 | TensorBoard 日志目录 | 每个 run 输出 `out/tensorboard/`；`metrics.jsonl` 同步 JSONL 记录 | 源码审查 + `test_tensorboard_roundtrip` |
| 8 | 是否支持断点续训 | **支持**。`--resume` 恢复模型、AdamW 优化器、主进程 RNG、`next_update`、`environment_steps`；严格校验 `model/sampler/trainer/environment` 配置段一致；调度器按恢复 update 继续。恢复会重开 episode（不恢复模拟器 RAM/worker RNG/未结束 episode），已在文档中说明 | `reproduction/runtime.py:restore_checkpoint`、Phase1 验收 |
| 9 | 中断后能否安全恢复 | **可以**。训练循环只在 update 边界保存；中断最多丢失 1 个 checkpoint 间隔（本配置 5 update = 10,000 步 ≈ 20 s）；恢复命令 `--config <run>/config.yaml --resume <run>/checkpoint_*.pt --out <新目录>`，不会覆盖旧目录 | 源码审查 + Phase1 resumed 验收 |
| 10 | 上一次 64 步训练报告与日志 | **通过**。首次 update：loss 0.4105、参数最大变化 6.0e-4、梯度范数 0.5；恢复后第 2 次 update：loss 0.1888、参数最大变化 6.0e-4；两模型分别为 `artifacts/smoke-gpu/checkpoint_000000032.pt`、`artifacts/smoke-gpu-resumed/checkpoint_000000064.pt`，仅用于管线验证，标注 Untrained Agent | `docs/reports/phase1_validation.md`、两份 `metrics.jsonl` |

补充自动测试：本轮在日志补丁后重跑 `pytest tests\phase1 -q` → **11 passed**（5.35 s）。

## 3. 训练脚本语义确认（避免超训/提前停止）

- `--total-steps` 是**累计绝对目标步数**（包含 resume 前的步数），不是本次增量；必须为 `n_workers × worker_steps` 的整数倍，脚本不会自动补齐或扩展。
- 每个 update 恰好消耗 `n_workers × worker_steps` 个真实环境步；训练循环 `while steps < total_steps` 以完整 update 边界停止。
- Checkpoint 保存条件：`update % checkpoint_interval == 0` 或 `steps == total_steps`；新增的 supervisor `--stop-file` 只在收到停止请求时**保保存当前 update checkpoint 后再退出**，不改变 PPO 行为。
- 恢复：从 checkpoint 的 `environment_steps` 继续；`--total-steps` 必须大于该值。阶段推进采用「每阶段一个新输出目录 + resume 上一阶段最后一个 checkpoint」，不会覆盖旧实验。

## 4. 并行规模基准测试（本机实测，不套用论文 32 worker）

论文完整配置为 **32 workers × 2048 steps = 65,536 步/update**；其 screen rollout 仅观测就约 4.22 GiB 显存/内存，加上 16 GB 物理内存（且系统已有多个应用占用）不可行。按任务要求先做真实基准，再选保守并行规模。

基准方法：用正式配置 `configs/train_seed0.yaml` 的副本，仅改 `sampler.n_workers/worker_steps`，各跑 1 个完整 PPO update（论文超参不变：3 epochs、8 minibatches），从 `metrics.jsonl` 的 `time/update_seconds`（采样+训练端到端）计算吞吐；以 PowerShell 0.5 s 间隔采样所有 python 进程 WorkingSet 之和与 `nvidia-smi`（桌面应用基线约 1.5 GiB 显存放于同一读数内）。

| 变体 | batch | update 耗时 | 端到端吞吐 | python 进程 RSS 峰值 | GPU 峰值温度 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 8 × 250 | 2,000 | 3.90 s | **513.1 steps/s** | 5,684 MB | 59 °C | **采用**：吞吐已足够，内存余量最大 |
| 12 × 170 | 2,040 | 3.90 s | 523.6 steps/s | 7,709 MB | 60 °C | +2% 吞吐、+2 GB 内存，不划算 |
| 16 × 125 | 2,000 | 3.66 s | 546.2 steps/s | 9,270 MB | 60 °C | +6% 吞吐、+3.6 GB 内存，内存风险高 |
| 20 × 100 | 2,000 | 未完成 | — | 约 10 GB+ | — | **放弃**：与系统其他应用叠加后内存过载，出现页面交换与 `nvidia-smi` 短暂无法通信；已终止该变体并清理进程 |

**决策**：`n_workers = 8`、`worker_steps = 250`，batch = 2,000。
理由：8 个 worker 已提供约 513 环境步/秒（1M 步纯训练约 33–40 分钟）；继续加 worker 只换来 2–6% 吞吐却显著抬高内存风险。本机可用内存 3.5–7.7 GiB 波动，8 worker 的 5.7 GB RSS 峰值仍偏紧但可运行；GPU 峰值温度约 60 °C、显存占用约 1.9 GB（含桌面基线），安全。

**batch=2,000 的额外好处**：整除所有阶段目标 10,000 / 100,000 / 300,000 / 1,000,000（分别为 5 / 50 / 150 / 500 个 update），因此每个阶段都能在精确的 PPO update 边界结束，不需要超出目标步数。

## 5. 正式配置与阶段计划

- 配置：`configs/train_seed0.yaml`（论文 PPO 超参不变：gamma 0.997、lambda 0.95、clip 0.2、value coef 0.5、max grad norm 0.5、epochs 3、minibatches 8、AdamW lr 3e-4 常数、entropy beta 0；环境奖励与论文 baseline 一致）。
- 与 `configs/paper_baseline.yaml` 的唯一差异：`sampler.n_workers=8`、`worker_steps=250`、`model.checkpoint_interval=5`；不覆盖论文原配置。
- 实验目录：`artifacts/training/seed_0/`；每个阶段单独 run 目录、单独 `config.yaml`/`environment.json`/`metrics.jsonl`/`tensorboard/`/checkpoint，全部与 Phase1 smoke 产物分开。
- 固定随机种子 `seed=0`；`environment.json` 记录 Python/PyTorch/CUDA/GPU、Git SHA、ROM SHA1、初始状态 SHA256、seed、device、resume 来源与目标。
- 阶段目标（累计环境步，均为 batch=2000 的整数倍）：

| 阶段 | 累计目标 | 新增步数 | 预计纯训练耗时（513 steps/s） |
|---|---:|---:|---:|
| A（稳定性） | 10,000 | 10,000 | 约 20 s |
| B | 100,000 | 90,000 | 约 3 min |
| C | 300,000 | 200,000 | 约 7 min |
| D | 1,000,000 | 700,000 | 约 23 min |

- 预算上限：**累计训练 1,000,000 环境步或 12 小时（先到者为准）**，不自动扩大预算。预计纯训练约 35–40 分钟，加上阶段启动/评估/分析，总时长远低于 12 小时；若实际吞吐明显下降则按剩余预算调整阶段 D 目标并在 progress 中记录。
- 不覆盖、不删除既有 `artifacts/smoke-gpu*`、`artifacts/english-*`、`artifacts/synthetic-*` 等验收产物。

## 6. 监督、Checkpoint 与恢复机制（本轮新增）

1. **独立进程训练**：`scripts/train.py` 由 `Start-Process` 以独立后台进程启动，stdout/stderr 重定向到 `artifacts/training/seed_0/logs/`；训练循环完全由本地 Python + GPU 执行，不逐步调用 LLM。
2. **持久化状态**：`artifacts/training/seed_0/training_state.json` 记录当前阶段、PID、日志路径、目标/实际步数、最近 checkpoint、累计训练秒数、停止文件路径、预算与截止时间；新会话先读该文件 + `docs/reports/training_progress.md` + `docs/reports/training_handoff.md`。
3. **PID 复用防护**：监督脚本用 psutil 校验 PID 存活且命令行包含 `scripts\train.py` 和阶段输出目录，不能仅凭 PID 判断运行正常。
4. **轻量监督**：`scripts/supervisor.py status/watch/verify-checkpoint/prune-checkpoints/stop-request`（由 teammate 开发，Lead 验收）：每 ~60 s 采集训练步数、吞吐、loss/熵/KL/clip、reward、游戏进度、CPU/GPU/内存/磁盘/温度，异常时退出码非 0，避免 LLM 高频轮询。
5. **优雅停止**：`--stop-file` 哨兵文件；达到 1M 步或 12h 预算时由 supervisor 创建停止文件，训练在**下一个 update 边界**保存 checkpoint 后干净退出，`result.json` 记录 `stop_reason`。
6. **Checkpoint 管理**：每 5 update（10,000 步）保存；保存即原子替换并读回校验；阶段里程碑 checkpoint 永久保留；其余保留最近若 N 个，旧 checkpoint 移至 `_archive/`（移动而非删除）；最佳模型另存评估目录，选择依据剧情里程碑 + 独立评估，不只看累计 reward。
7. **异常处理**：可自动处理单次进程异常退出（从最后验证通过的 checkpoint 恢复）、临时文件问题、资源不足时适度降 worker；同类异常连续两次不自动重试；不修改奖励/PPO/模型/观测/动作，不删除现有 checkpoint，不清空日志，不无限重启。

## 7. 日志增强（非侵入式，不改变 PPO 数学）

为满足 Phase 4 监督指标（游戏进度、动作重复、吞吐），本轮对项目代码做了两处**只增加日志采集**的修改，未改变训练行为：

1. `neroRL/sampler/trajectory_sampler.py`：把 worker 每步**本来就会返回**的 `info` 字典保留为 `latest_infos`，并把本次 update 完成的 episode 列表暴露为 `last_episode_infos`（原先只在 episode 结束时短暂使用）。不改变采样、动作、奖励、GAE 或 PPO。
2. `scripts/train.py`：
   - 新增 `episodes.jsonl`：每次 episode 截断时记录完整游戏进度 info（reward/length/coord_count/badges/levels_sum/party_size/deaths/max_map_progress/mt_moon/cerulean 等）。
   - 每个 update 的 `metrics.jsonl` 新增：`game/coord_count`、`game/max_map_progress`、`game/num_badges`、`game/levels_sum`、`game/party_size`、`game/deaths`、`game/max_event_rew`、`game/event_reward`、`game/healr`、`game/mt_moon`、`game/cerulean`、`actions/share_0..6`（动作分布，用于检测单一动作刷屏）、`rollout/reward_sum`、`time/unix`、`time/steps_per_second`、`episode/completed_this_update`；TensorBoard 同步记录。
   - 新增 `--stop-file` 优雅停止（见上）。
   - 新增 `--stop-file` 不改变其余 CLI 语义；`result.json` 增加 `stop_reason` 字段。
3. 回归验证：`pytest tests\phase1 -q` → 11 passed；真实游戏 32 步 smoke 重跑成功（loss 0.4105 与上一轮一致，新增指标正常，`game/coord_count=3`、动作分布 7 项、`time/steps_per_second≈52.8`）。

## 8. 风险与应对

| 风险 | 实测/评估 | 应对 |
|---|---|---|
| 物理内存紧张（16 GB，多应用占用） | 8 worker 峰值 5.7 GB RSS；20 worker 时曾出现内存过载与 nvidia-smi 短暂失联 | 固定 8 worker；监督脚本监控 available RAM 与页面交换；异常时降 worker；不并行跑重评估 |
| GPU 温度/功耗 | 基准 59–60 °C，功耗低；RTX 4060 Laptop | 监督每 ~60 s 读取温度；>85 °C 暂停并报告，不做速度优先 |
| 长时间无新环境步数 | 吞吐基准稳定；每 update 约 4 s | supervisor 检测最近更新 >600 s 或连续两次无新步数即告警 |
| Loss NaN/Inf、参数不更新 | train.py 本身已有 finite 检查与“参数必须变化”断言 | 直接失败退出，从最近验证通过的 checkpoint 恢复 |
| Checkpoint 损坏 | 保存后立即读回校验；原子替换 | 恢复时 verify-checkpoint 校验并保留最近若干份 |
| Reward hacking（刷治疗/升级/探索不推进剧情） | 论文讨论的风险；奖励与论文 baseline 一致 | 记录 game/* 进度与 actions/share_*、episodes.jsonl；阶段评估同时看剧情里程碑（badges/map/events），不只看 reward，也不把 reward 上升当作“学会游戏” |
| 训练吞吐低于预期 | 当前约 513 steps/s | 阶段推进时用剩余预算重算 D 目标并记录；12h 硬截止 |

## 9. 启动前最终状态

- 无正在运行的训练进程；GPU 温度 50 °C、显存占用约 1.5 GiB（桌面基线）。
- 正式配置与日志补丁已就绪；实验目录、状态文件、启动命令已准备。
- **批准启动 Stage A（累计 10,000 步）→ B（100,000）→ C（300,000）→ D（1,000,000）。**
