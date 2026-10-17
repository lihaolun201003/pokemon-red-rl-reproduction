# Pokémon Red RL — 独立复现与长训监督实验（PPO / neroRL `poke_red`）

本仓库是对论文 **[Pokémon Red via Reinforcement Learning](https://arxiv.org/abs/2502.19920)** 官方代码 [neroRL `poke_red` 分支](https://github.com/MarcoMeter/neroRL/tree/poke_red) 的**独立复现工程**：在 Windows + RTX 4060 Laptop + Python 3.10 上跑通真实 Game Boy 模拟器（PyBoy）+ PPO + CUDA 训练管线，并完成一次 **1,000,000 真实环境步** 的监督式训练实验（实验 `seed_0`）与**固定协议的独立评估**。

> ⚠️ **合规声明（务必先读）**：本仓库**不包含**任何游戏 ROM、模拟器存档、模型权重或游戏录像。运行需要你**自行合法拥有**的英文 Pokémon Red ROM 与初始状态文件；代码中的 SHA1/SHA256 校验只用于确认输入兼容性，不提供下载。Pokémon 及相关游戏内容版权归其权利人所有。详见 [NOTICE.md](NOTICE.md)。

## 当前状态与结果摘要（真实数据，非论文成绩）

- 已完成 Phase 1 基础验收：真实英文 ROM + 官方初始存档 + PyBoy + 真实 CUDA PPO 更新 + Checkpoint 保存/恢复 + TensorBoard + 观看/录像；自动测试通过。
- 已完成实验 `seed_0` 正式训练：**累计 1,000,000 环境步**（A 10k → B 100k → C 300k → D 1M，500 次 PPO update，固定 seed=0），纯计算约 **1,540 s（25.7 分钟）**，平均 **649.4 steps/s**（8 workers × 250 steps，本机 16 GB RAM 下的保守并行规模）。
- 已用固定协议（checkpoint 配置权威、每 episode 新建 PyBoy 实例、4096/8192 步 × 3/5 个随机策略 episode + 确定性 episode）评估 10k/100k/300k/500k/1M 五个里程碑 checkpoint，并与**未训练随机初始化**和**均匀随机动作**基线对比。
- **诚实结论**：训练后策略学会了离开出生房间（Oak's Lab）并在 Pallet Town/Route 1 一带探索（扩展评估 n=5：探索坐标 205 vs 未训练 134），但**未获得任何道馆徽章，未到达 Mt Moon 或 Cerulean**；确定性 argmax 推理在所有 checkpoint 都退化为“单键粘滞”，真实探索只在随机采样下出现；500k checkpoint 在两种协议中一致地明显回落。1M 步仅约为论文完整配置（约 3.2×10⁹ 步）的 **0.03%**，因此**不能声称学会通关或复现论文成绩**。
- 最佳模型（按扩展评估）：`checkpoint_001000000.pt`（1M 步）；次优：`checkpoint_000300000.pt`（300k 步）。**模型权重不包含在本仓库中**（体积/合规原因），需本地训练生成。

完整报告（中文）：
[最终报告](docs/reports/training_final_report.md) ·
[训练曲线](docs/reports/training_curves.md) ·
[评估结果](docs/reports/eval_results.md) ·
[评估协议](docs/reports/eval_protocol.md) ·
[Phase 1 验收](docs/reports/phase1_validation.md) ·
[接管说明](docs/HANDOFF.md) / [训练交接](docs/reports/training_handoff.md)

## 仓库结构

```
configs/           # 训练配置：paper_baseline.yaml（论文原始参数，未改）、train_seed0.yaml（本轮正式长训）、smoke.yaml
scripts/           # 入口脚本：train / watch / eval_checkpoint / supervisor / plot_* / doctor / check_* / probe / build_red
neroRL/            # 固定上游 neroRL（poke_red 分支）源码，MIT，见 LICENSE / UPSTREAM.json
reproduction/      # 本仓库的运行时封装（配置加载、输入校验、Checkpoint 原子保存/恢复、设备选择）
tests/phase1/      # 不依赖 ROM 的自动测试（11 项，含合成 PPO/Checkpoint/TensorBoard）
docs/              # 上游文档、复现报告、训练监督/评估报告与图表
notebooks/ result/ # 上游附带文件
configs/ 与 docs/ 的说明见各文件
```

## 环境要求与安装（Windows 验证环境）

- Windows 11，Python 3.10.11，独立虚拟环境 `.venv`
- PyTorch 2.5.1+cu121（CUDA 12.1），PyBoy 2.4.0，Gymnasium 0.29.0，TensorBoard 2.18.0
- 已验证硬件：i9-13900HX（24C/32T）、16 GB RAM、NVIDIA RTX 4060 Laptop 8 GB（驱动 592.82）
- 其它平台未验证；Windows 是唯一实测目标。

```powershell
cd C:\path\to\pokemon-red-rl-reproduction
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-cuda.txt   # CUDA 环境（已验证）
# 纯 CPU 环境可选 requirements-lock-cpu.txt
```

自检：

```powershell
.\.venv\Scripts\python.exe scripts\check_cuda.py     # CPU/CUDA 张量与官方网络前后向
.\.venv\Scripts\python.exe scripts\doctor.py         # 模拟器/依赖探针
.\.venv\Scripts\python.exe -m pytest tests\phase1 -q # 11 passed（不依赖 ROM）
```

## 合法 ROM 与初始状态（必须自备）

本仓库不分发 ROM/存档。支持并强制校验：

- ROM：英文 Pokémon Red（pret/pokered 构建），1 MiB，**SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`**
- 初始状态：官方 `has_pokedex_nballs.state`，**SHA256 `0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e`**
- 放置为 `local/PokemonRed.gb` 和 `local/has_pokedex_nballs.state`，或通过 `--rom/--state` 显式指定；`local/` 已被 gitignore。
- 构建说明见 [Phase 1 报告](docs/reports/phase1_validation.md) 与 `scripts/build_red.ps1`（需要你自行准备 pret/pokered 源码与 RGBDS 工具；脚本不下载 ROM）。

真实游戏冒烟检查：

```powershell
.\.venv\Scripts\python.exe scripts\check_game.py --steps 128
```

## 训练

```powershell
# 冒烟：2 workers × 16 steps，累计 32 步，输出目录必须不存在
.\.venv\Scripts\python.exe scripts\train.py --config configs\smoke.yaml --seed 0 --total-steps 32 --device cuda --out artifacts\smoke

# 本仓库的正式长训配置（8 workers × 250 steps = 2,000 步/update；论文 PPO 超参不变）
.\.venv\Scripts\python.exe scripts\train.py --config configs\train_seed0.yaml --seed 0 --total-steps 1000000 --device cuda --out artifacts\my_run

# 断点续训：输出必须是新目录；--total-steps 是累计目标且必须是 2,000 的整数倍
.\.venv\Scripts\python.exe scripts\train.py --config artifacts\my_run\config.yaml --resume artifacts\my_run\checkpoint_000100000.pt --total-steps 300000 --device cuda --out artifacts\my_run_resumed
```

- 每个 run 输出 `config.yaml`、`environment.json`（Git SHA/ROM/状态哈希/seed/device）、`metrics.jsonl`、`episodes.jsonl`、`tensorboard/`、原子保存并读回校验的 `checkpoint_*.pt`、`result.json`。
- `--stop-file <path>`：监督者创建该文件后，训练在**下一个 update 边界**保存 checkpoint 并干净退出（供外层监督使用）。
- Checkpoint 每 `checkpoint_interval` 个 update 保存；`reproduction/runtime.py` 使用 tmp + 原子 replace，并在保存后立即读回校验模型/优化器/步数。
- 恢复会重开 episode（不恢复模拟器中途 RAM、worker RNG），不保证逐位一致轨迹；详见 [docs/HANDOFF.md](docs/HANDOFF.md)。

TensorBoard：

```powershell
.\.venv\Scripts\tensorboard.exe --logdir artifacts --port 6006
```

## 观看模型 / 独立评估 / 监督工具

```powershell
# 观看（随机策略推理；Q/Esc/Ctrl+C 退出；--speed 1 ≈ 实时）
.\.venv\Scripts\python.exe scripts\watch.py --checkpoint artifacts\my_run\checkpoint_001000000.pt --device cuda --speed 1 --steps 300

# 固定协议独立评估（checkpoint 配置权威；每 episode 新建 PyBoy；3 随机 + 1 确定性 episode × 4096 步）
.\.venv\Scripts\python.exe scripts\eval_checkpoint.py --checkpoint artifacts\my_run\checkpoint_001000000.pt --out artifacts\eval\run_1m --device cuda --steps 4096 --episodes 3 --seed 100000 --video artifacts\eval\demo.mp4
# 基线
.\.venv\Scripts\python.exe scripts\eval_checkpoint.py --untrained --config configs\train_seed0.yaml --out artifacts\eval\untrained --device cuda --steps 4096 --episodes 3
.\.venv\Scripts\python.exe scripts\eval_checkpoint.py --random-actions --config configs\train_seed0.yaml --out artifacts\eval\random --device cuda --steps 4096 --episodes 3

# 汇总评估结果 → eval_summary.json + docs/reports/eval_results.md
.\.venv\Scripts\python.exe scripts\summarize_eval.py artifacts\eval

# 训练监督（status/watch/verify-checkpoint/prune-checkpoints/stop-request；只读，不启动训练）
.\.venv\Scripts\python.exe scripts\supervisor.py status
.\.venv\Scripts\python.exe scripts\supervisor.py watch --timeout 3600
.\.venv\Scripts\python.exe scripts\supervisor.py verify-checkpoint --path artifacts\my_run\checkpoint_001000000.pt --out artifacts\my_run
.\.venv\Scripts\python.exe scripts\supervisor.py prune-checkpoints --out artifacts\my_run --keep-last 10 --milestones 100000,300000,1000000 --dry-run

# 曲线
.\.venv\Scripts\python.exe scripts\plot_training_curves.py
.\.venv\Scripts\python.exe scripts\plot_eval_comparison.py
```

`scripts/supervisor.py` 的 `watch` 退出码：0=阶段目标完成、1=异常/进程退出无 result、2=超时、4=提前停止；`status` 输出吞吐/ETA/资源/游戏进度/告警，并追加 `monitor/status.jsonl` 快照。

## 训练实验 seed_0 的核心数字（实测）

| 项目 | 值 |
|---|---|
| 训练环境步 / PPO updates | 1,000,000 / 500 |
| 计算时间 / 墙钟 | 1,539.98 s（25.7 min）/ 1,632 s（27.2 min，含阶段切换） |
| 平均吞吐 | 649.4 steps/s（时间加权；区间 322.9–873.9） |
| GPU | RTX 4060 Laptop 8 GB；显存 1.8–2.24 GiB；温度 56–64 °C |
| 资源告警 | 139 次监督快照 0 条告警；无 NaN/Inf、无 checkpoint 失败 |
| 完成 episode / 奖励 | 66 个；episode reward 均值 4.723±3.644，最大 12.827 |
| 探索 | 已探索坐标 episode 级最大 430；关键地图推进最大 3/14（Viridian 一带） |
| 剧情 | **徽章 0；Mt Moon 0；Cerulean 0** |
| 独立评估（扩展，n=5/8192 步） | 1M：reward 3.83±2.77、coords 205.4±25.6、maps 4.0；300k：3.09±2.37、197.6、4.0；untrained：1.47±1.50、134.4、2.2 |

> 口径：训练奖励是官方 shaped reward，不是通关奖励；评估步数不属于训练预算（本轮 315,520 评估/演示步单独统计）。所有数字来自本地真实 `metrics.jsonl` / `episodes.jsonl` / `results.json`，未做人工修饰。

## 复现边界

- 本仓库是**独立复现工程**，不是论文作者官方发布；论文与上游/本地实现之间的已知差异（等级/治疗奖励归一化、事件 budget、终止条件、seed 语义等）见 [docs/baseline_comparison.md](docs/baseline_comparison.md)。
- 论文参数保留在 `configs/paper_baseline.yaml`（32 workers × 2048 steps 等），本机 16 GB RAM 无法直接运行该规模；本轮使用经基准测试后的 `configs/train_seed0.yaml`（8 × 250），PPO/奖励/网络超参不变。
- 训练结果仅代表 seed=0 的单次实验；评估样本量小（n=3–5），不做显著性结论。

## 许可与致谢

- 本仓库代码沿用上游 neroRL 的 **MIT License**（Copyright © 2019 Marco Pleines），见 [LICENSE](LICENSE)；本地修改与来源说明见 [NOTICE.md](NOTICE.md) 与 [UPSTREAM.json](UPSTREAM.json)。
- 环境源自 [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)（MIT，许可保留在 `docs/upstream/`）；地图代码致谢 `pokemonred_puffer`。
- 论文：Marco Pleines et al., *Pokémon Red via Reinforcement Learning*, [arXiv:2502.19920](https://arxiv.org/abs/2502.19920)。
- ROM 由用户自行合法获取；本仓库不包含任何游戏 ROM、存档、模型权重、录像或大型日志。

## 不随仓库分发的内容

以下均已在 `.gitignore` 中排除，需本地自行生成或合法准备：`local/`（ROM、初始状态）、`artifacts/`（checkpoint、TensorBoard、评估结果、视频）、`.venv/`、`work/`（ROM 构建源码/工具链）、`session/`、`*.gb`、`*.state`、`*.pt`、`*.mp4`、`*.log` 等。请勿用 `git add -f` 把它们加入仓库。
