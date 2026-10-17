# Pokémon Red RL — 第一阶段独立复现

复用 [论文](https://arxiv.org/abs/2502.19920) 作者的 [neroRL `poke_red` 分支](https://github.com/MarcoMeter/neroRL/tree/poke_red)，固定 SHA `1d5c2db1f47f29d501ec9eacdc9c7f6480ea6217`。保留 MIT 许可证及原作者署名，详见 `NOTICE.md`、`UPSTREAM.json`。

**状态：CPU 与 RTX 4060 CUDA 环境均已验证，自动测试 11 项全部通过。用户提供的 2 MiB 中文修改版 ROM 可启动，但与官方初始状态不兼容，真实游戏训练验收仍阻塞。未进行真实 Pokémon 训练，未复现论文成绩，未发布到 GitHub。** 合成 RAM 测试的 Checkpoint 不能算真实训练成果。

## 安装（Windows / Python 3.10）

本地目录 `C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`，已建独立 `.venv`。VS Code 打开目录或 `.code-workspace`，选择 `.venv\Scripts\python.exe`。

```powershell
cd C:\Users\lihao\Desktop\pokemon-red-rl-reproduction
# 已有环境无需重装；新建环境可使用 cuda（或改为 cpu）
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Device cuda
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe scripts\check_cuda.py
.\.venv\Scripts\python.exe -m pip check
```

验证版本：Python 3.10.11、PyTorch 2.5.1+cu121、PyBoy 2.4.0、Gymnasium 0.29.0。当前依赖锁为 `requirements-lock-cuda.txt`；此前 CPU 依赖锁保留为 `requirements-lock-cpu.txt`。上游完整依赖另存 `docs/upstream/requirements.txt`。未安装 Unity/Procgen 等无关环境。

RTX 4060 Laptop 8 GB、驱动 592.82。已从官方源下载并核对 SHA-256 后，仅替换项目 `.venv` 中的 PyTorch。`torch.cuda.is_available()` 为 True，CPU/CUDA Tensor backward、完整官方网络前后向、CUDA 合成 PPO 更新与 Checkpoint 恢复均通过。重建 CUDA 环境的命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Device cuda
```

该命令下载约 2.45 GB 的 `torch==2.5.1+cu121`，不会把 CPU 构建误认为 CUDA 构建。当前环境已经安装，无需重复执行。系统 Python 没有安装或修改 torch。

## 合法 ROM 与初始状态

项目不提供或下载游戏 ROM。自行合法取得英文 Pokémon Red，放到 `local\PokemonRed.gb`。支持 SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`，来源见 [辅助项目说明](https://github.com/PWhiddy/PokemonRedExperiments#running-the-pretrained-model-interactively)。不支持其他语言、改版或不同内存布局。

本次提供的 `local\PokemonRed.gbc` 为 2 MiB，SHA1 `944ff0b461fb570d2c8aa99d819e8c8df0bdc446`。PyBoy 实际显示中文标题、菜单和剧情，Enter 对应 Start，启动和按键可用。官方 `has_pokedex_nballs.state` 已按用户要求从固定版本复制到 `local`，但载入两帧后队伍数量变 57、Level 0、HP 57/14592，程序停在 `0x0038`，第一步奖励异常为 1258.005。**判定为官方初始状态执行不兼容；未修改内存地址、ROM、存档或奖励来绕过。**

仅诊断未知 ROM 的可重复命令（不批准训练；此输入退出码 2）：

```powershell
.\.venv\Scripts\python.exe scripts\probe_rom.py --rom local\PokemonRed.gbc
```

详细失败原因及证据路径见 `docs/reports/phase1_validation.md`。训练/观看入口继续拒绝这个不匹配的 ROM；不要重命名扩展名或创建白名单强行通过。

初始状态放到 `local\has_pokedex_nballs.state`，必须合法提供且兼容 PyBoy 2.4.0；`.sav` 电池存档不等于 `.state` 模拟器状态。论文 Baseline 在选择杰尼龟、交付 Oak's Parcel、取得 Pokédex / Poké Balls 后开始。可在自己的合法 ROM 中手动完成，再保存私有状态：

```powershell
.\.venv\Scripts\python.exe scripts\capture_state.py
```

入口打开 PyBoy SDL 窗口；达到起点后在启动终端按 Ctrl+C 保存；关闭窗口则不保存。**真实游戏入口未验收。** 上游 Git 附带的初始状态不能代替 ROM，本项目不将其作为可再分发素材；原始文件仅留在被忽略的本地参考文件中。

入口支持 `--rom C:\path\PokemonRed.gb --state C:\path\initial.state`。缺文件会在启动 worker 前失败，不会自动替换为模拟数据。

## 最小测试与启动训练

```powershell
# 不依赖 ROM：官方完整网络、合成 RAM、PPO 更新/恢复、PyBoy 原创诊断卡带、TensorBoard
.\.venv\Scripts\python.exe -m pytest tests\phase1 -q

# 以下需要合法 ROM + 初始状态
.\.venv\Scripts\python.exe scripts\check_game.py --steps 128
.\.venv\Scripts\python.exe scripts\train.py --config configs\smoke.yaml --seed 0 --total-steps 32 --device cpu --out artifacts\smoke
```

Smoke 为 2 workers × 16 steps，一轮 32 个环境步；累计硬上限 256。`--total-steps` 是累计目标，必须是完整 rollout 的倍数，不自动补齐或扩容。每次更新检查 Loss/梯度/参数有限值及实际参数变化。

## 恢复训练

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config artifacts\smoke\config.yaml --resume artifacts\smoke\checkpoint_000000032.pt --total-steps 64 --device cpu --out artifacts\smoke-resumed
```

恢复模型、AdamW、随机数状态、next_update 和 environment_steps；多项式调度器以恢复的 update 继续计算。严格校验配置。目标 64 表示从 32 再跑 32 步。输出目录必须全新。

不保存中途模拟器 RAM、worker RNG 或未结束 episode；恢复从初始状态重开 episode，不承诺逐位一致的轨迹。仅加载可信本地 `.pt`，官方格式含 pickle/Gym spaces。

## 观看 AI 玩游戏

```powershell
# 明确标记：Untrained Agent — 仅用于运行测试
.\.venv\Scripts\python.exe scripts\watch.py --untrained --speed 1 --steps 1000 --device cpu

# 将来真实 smoke 完成后加载指定权重；Smoke 权重仍不代表学会游戏
.\.venv\Scripts\python.exe scripts\watch.py --checkpoint artifacts\smoke\checkpoint_000000032.pt --speed 2 --steps 1000 --device cpu

# 可选视频
.\.venv\Scripts\python.exe scripts\watch.py --untrained --speed 1 --steps 100 --video artifacts\watch.mp4
```

OpenCV 窗口显示 PyBoy 原始游戏画面，模型自动采样动作，不调用 LLM。`--speed 1` 约 60 Game Boy 帧/秒（2.5 动作/秒）。Q、Esc、关闭窗口或终端 Ctrl+C 退出。输出步数、奖励、事件、动作、坐标。无兼容预训练权重随项目提供。真实 AI 观看/录像因当前 ROM 与官方状态不兼容未验收；原始 PyBoy SDL 游戏画面已实际确认。

## 后续训练与日志

入口支持 `--config`、`--seed`、`--total-steps`、`--device cpu/cuda/auto`、`--out`、`--resume`。独立论文参数为 `configs/paper_baseline.yaml`，原 `poke_red_v2.yaml` 保留。论文与源代码的奖励差异见 `docs/baseline_comparison.md`。

论文批次 32×2048，单份 float32 screen rollout 约 4.22 GiB；还需要批次副本、模型、优化器及其他模态。不要在 16 GB RAM 本机直接启动论文规模；先完成真实 Smoke 再评估资源。本阶段未运行该配置。

每次训练输出 `config.yaml`、`environment.json`（硬件、依赖、Git SHA、输入 hash、seed）、`metrics.jsonl`、`tensorboard/`、`checkpoint_*.pt`、`result.json`。记录 Reward、Loss、梯度、Learning Rate、Environment Steps 和 checkpoint 路径。

```powershell
.\.venv\Scripts\tensorboard.exe --logdir artifacts --port 6006
```

访问 `http://localhost:6006`。本阶段验证了事件文件写入后可由 TensorBoard EventAccumulator 读取。

## 实际验收状态

- CPU/CUDA 环境、官方网络推理、合成 PPO 更新/保存/加载/恢复、TensorBoard：通过，pytest 11 passed。
- 当前 ROM 的启动、中文游戏画面、Enter/Start 响应：通过原始模拟器验证。
- 官方状态文件 API load 可成功，但继续执行失败；真实 rollout/PPO 更新、真实 checkpoint 和 AI watch：阻塞，未执行训练。
- 通关、城市里程碑、论文成功率、长时间训练：未开展，无成绩声明。

报告 `docs/reports/phase1_validation.md`；进度 `docs/reports/progress.md`；交接 `docs/HANDOFF.md`。本地 Git 是不含 ROM/状态的独立历史；原始上游 Git 归档 `local/upstream.git` 含上游状态历史，已忽略，禁止推送。不要用 `git add -f` 纳入本地素材。
