# 训练交接说明（RL Training Supervisor 轮次，实验 seed_0）— 本轮完成

> 给下一个接管本项目的 Agent。**先按顺序读**：本文件 → `docs/reports/training_progress.md` → `docs/reports/training_final_report.md` → `artifacts/training/seed_0/training_state.json` → 运行 `python scripts/supervisor.py status` 并检查真实进程/日志/checkpoint/磁盘。
> 最后更新：2026-10-18 01:40（Asia/Shanghai）

## 0. 一句话现状

本轮正式长训实验 `seed_0` **已完成**：累计 **1,000,000 真实环境步**（A 10k → B 100k → C 300k → D 1M，500 次 PPO update，seed=0），训练进程已全部退出，无残留。独立评估（protocol v1 + v1-ext）已完成，主推最佳模型 `stage_d_1m/checkpoint_001000000.pt`，次优 `stage_c_300k/checkpoint_000300000.pt`。**没有获得徽章/剧情里程碑**（详见最终报告）。未经用户明确授权，不要继续训练或扩大预算。

## 1. 项目与不可变事实

- 工作目录：`C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`；解释器 `.venv\Scripts\python.exe`（Python 3.10.11 / torch 2.5.1+cu121 / CUDA / RTX 4060 Laptop 8 GB / 16 GB RAM）。
- ROM：`local/PokemonRed.gb`，SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`；初始状态 `local/has_pokedex_nballs.state`，SHA256 `0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e`。**不要替换/重新下载 ROM，不要上传 ROM/权重/视频。**
- PPO、GAE、网络、奖励、观测、动作全部来自 neroRL `poke_red`（见 `UPSTREAM.json`）；**禁止修改 PPO 数学、奖励权重、模型架构、观测/动作定义**。只允许必要的日志/运行兼容修改。
- 正式配置：`configs/train_seed0.yaml`（8 workers × 250 steps = 2,000 步/update；论文超参不变；checkpoint 每 10,000 步；seed=0）。论文配置 `configs/paper_baseline.yaml` 未被覆盖。
- 本轮唯一的非侵入式代码修改（Git 提交 `ed4fcb2` 及后续）：`neroRL/sampler/trajectory_sampler.py` 保留每步 info（日志用）；`scripts/train.py` 增加 `episodes.jsonl`、更多 metrics、`--stop-file` 优雅停止、`result.json.stop_reason`；新增监督/评估/分析脚本（`scripts/supervisor.py`、`scripts/eval_checkpoint.py`、`scripts/summarize_eval.py`、`scripts/plot_training_curves.py`、`scripts/plot_eval_comparison.py`）。未改变训练行为。
- 不要 Push GitHub；artifacts/ 全部本地保存、已被 gitignore；ROM/状态/权重/视频不得上传。

## 2. 已完成的工作与产物

| 产物 | 路径 |
|---|---|
| 训练 run A/B/C/D | `artifacts/training/seed_0/stage_{a_10k,b_100k,c_300k,d_1m}/`（config.yaml、environment.json、metrics.jsonl、episodes.jsonl、tensorboard/、checkpoint_*.pt、result.json） |
| 状态 | `artifacts/training/seed_0/training_state.json`（status=complete；含 stages、best_checkpoint、eval_runs、checkpoint_management） |
| 监督/验证/剪枝 | `artifacts/training/seed_0/monitor/`（status.jsonl 139 条、verify_summary.json、prune_actual.json、阶段完成快照） |
| 独立评估 | `artifacts/training/seed_0/eval/<run>/results.json`、`eval_summary.json`、`videos/` |
| 报告 | `docs/reports/training_final_report.md`（最终）、`training_progress.md`、本文件、`training_supervisor.md`（Phase1 审查）、`training_curves.md`、`eval_protocol.md`、`eval_results.md`、`supervisor_tooling.md` |
| 图表 | `docs/reports/figures/*.png`（7 张训练曲线 + 评估对比图） |
| 分析数据 | `artifacts/training/seed_0/analysis/`（training_summary.json/.md、metrics_10k.csv、run_log.json） |

## 3. 如用户授权继续训练（未授权则不要做）

1. 先 `python scripts/supervisor.py status` 确认无训练进程、GPU 空闲、state 正常。
2. 建议从 1M checkpoint 继续（新目录，不覆盖）：
   ```
   .\.venv\Scripts\python.exe scripts\train.py --config artifacts\training\seed_0\stage_d_1m\config.yaml \
     --resume artifacts\training\seed_0\stage_d_1m\checkpoint_001000000.pt --total-steps <新累计目标，如 10000000> \
     --device cuda --out artifacts\training\seed_0\stage_e_10m --stop-file artifacts\training\seed_0\STOP
   ```
   目标必须是 2,000 的整数倍；阶段推进用 `--resume` 链式从上一阶段最后 checkpoint 继续；固定新预算并记录。
3. 训练运行时**禁止其他 CUDA 进程**（本机曾因并发 GPU + 内存压力出现 nvcuda64.dll 原生崩溃）；建议保持 8 workers（16GB RAM 限制），如要加 worker 必须重跑基准并记录。
4. 每阶段结束用 `scripts/supervisor.py watch`/`verify-checkpoint`/`prune-checkpoints`，并用 `scripts/eval_checkpoint.py` 做固定协议评估（对比基线 untrained/random-actions，协议 v1：`--steps 4096 --episodes 3 --seed 100000`，或 v1-ext：`--steps 8192 --episodes 5`）。
5. 达到新预算或用户指定时间后停止（自然到目标或创建 `artifacts/training/seed_0/STOP`），保存 checkpoint 并更新报告。

## 4. 已知风险/注意事项

- 物理内存紧张（16 GB，多应用占用）：训练时可用 RAM 曾低至 2.0 GiB；**训练期间禁止其他 CUDA 进程/重负载**。
- 恢复会重开 episode（不恢复模拟器 RAM/worker RNG），已文档化；恢复后不能期望逐位一致轨迹。
- 不要把训练 shaped reward 上升当作“学会游戏”；结论以固定协议独立评估 + 剧情里程碑（badges/地图/事件）为准。
- 评估消耗的真实环境步数单独统计（本轮 307,328），不计入训练预算；继续训练时仍需在报告中单列。
- 确定性 argmax 推理在本轮所有 checkpoint 上退化（单键粘滞）；如需真实部署/演示，用 `scripts/watch.py`（随机采样）或论文作者确认的评测方式。
- 详细限制与诚实声明见 `docs/reports/training_final_report.md` §13。
