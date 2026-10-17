# Pokémon Red RL — 第一阶段独立复现

复用 [论文](https://arxiv.org/abs/2502.19920) 作者的 [neroRL `poke_red` 分支](https://github.com/MarcoMeter/neroRL/tree/poke_red)，固定 SHA `1d5c2db1f47f29d501ec9eacdc9c7f6480ea6217`。保留 MIT 及原作者署名，见 `LICENSE`、`NOTICE.md`、`UPSTREAM.json`。

**基础运行验收通过**：标准英文 ROM + 官方初始状态正常，RTX 4060 完成 32 步真实 PPO 更新，再恢复到累计 64 步；CPU/CUDA 观看均显示模型操作游戏，GPU 视频可解码。自动测试 **11 passed / 0 failed / 0 skipped**。已停止训练，未复现论文成绩，未发布 GitHub。

Checkpoint 仅用于管线验证，明确标记 **Untrained Agent — 仅用于运行测试**；不能声称学会通关。

## 本机环境与安装

项目 `C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`，独立解释器 `.venv\Scripts\python.exe`。VS Code 打开根目录的 `.code-workspace` 即可选择该环境。现有环境可直接使用，无需重装。

验证版本：Python 3.10.11、PyTorch 2.5.1+cu121、PyBoy 2.4.0、Gymnasium 0.29.0；RTX 4060 Laptop 8 GB、驱动 592.82、系统 RAM 约 16 GB。CPU 与 CUDA 张量/网络前后向均通过，系统 Python 没有安装或修改 torch。

```powershell
cd C:\Users\lihao\Desktop\pokemon-red-rl-reproduction
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe scripts\check_cuda.py
.\.venv\Scripts\python.exe -m pip check
```

只有新建环境时才运行：

```powershell
powershell -File scripts\setup.ps1 -Device cuda
# CPU 环境改用 -Device cpu
```

实际依赖锁为 `requirements-lock-cuda.txt`，先前 CPU 依赖锁为 `requirements-lock-cpu.txt`。上游完整依赖另存 `docs/upstream/requirements.txt`；不需要安装 Unity/Procgen 等无关环境。

## 合法 ROM 与官方初始状态

ROM、游戏状态、截图、录像及模型均不随代码分发。必须自行具备合法使用权。支持的英文原版 SHA1：`ea9bcae617fdf159b045185467ae58b2e4a48b9a`。

本机用户已确认合法使用权，使用 [pret/pokered](https://github.com/pret/pokered) 源码构建了 1 MiB 英文版，固定 SHA `af519899719f0754965776faac0e836a3b906e6d`、RGBDS 1.0.4，运行 `make red` 后哈希完全一致。仅复制并改文件名，未修改 ROM 字节：

```text
local\PokemonRed.gb
local\has_pokedex_nballs.state
```

初始状态来自固定版本 neroRL，SHA256 `0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e`。它在英文原版中加载及继续运行均正常；初始位置 `(5,3,map 40)`、1 只杰尼龟、Level 6、HP 22/22。`.sav` 电池存档不能代替 `.state` 模拟器状态。

构建源码与工具保留在忽略的 `work/` 下；已有工具的本机可重复执行：

```powershell
.\scripts\build_red.ps1
```

脚本校验源码版本、RGBDS 版本及输出 SHA1，只有完全匹配才复制到 `local`。构建细节、便携工具下载源及校验见 `docs/reports/phase1_validation.md` 和 [上游安装说明](https://github.com/pret/pokered/blob/master/INSTALL.md)。脚本不包含源码素材，也不会下载预编译 ROM。

之前的 `local/PokemonRed.gbc` 为中文修改版，与官方状态不兼容，已停止适配；不要把它改名冒充英文原版。历史失败证据保留在 `docs/reports/rom_cuda_validation_previous.md`。所有生产入口仍严格检查 ROM hash，并支持显式 `--rom`、`--state` 路径。

需要自行创建状态时，可用 `scripts/capture_state.py` 打开自己的合法 ROM，达到起点后在终端 Ctrl+C 保存；这个手动捕获入口未单独验收。现有官方状态已可用，不必重新创建。

## 测试与启动训练

```powershell
# 不依赖 ROM 的官方网络、合成 PPO、Checkpoint、TensorBoard 等测试
.\.venv\Scripts\python.exe -m pytest tests\phase1 -q
# 真实游戏：连续操作、全部动作、观测、render、预算结束与 reset/close
.\.venv\Scripts\python.exe scripts\check_game.py --steps 128
# 真实游戏状态语义及 32 步兼容性诊断
.\.venv\Scripts\python.exe scripts\probe_rom.py --rom local\PokemonRed.gb --out artifacts\english-rom-check-new
# 新的 32 步真实 CUDA 训练；输出目录必须不存在
.\.venv\Scripts\python.exe scripts\train.py --config configs\smoke.yaml --seed 0 --total-steps 32 --device cuda --out artifacts\smoke-new
```

Smoke 为 2 workers × 16 steps，一轮 32 个训练采样步，累计硬上限 256。每次检查 Loss/梯度/参数有限值及实际参数变化。`--total-steps` 是累计目标，必须整批；不自动补齐、扩容或后台训练。CPU 可用 `--device cpu`；本阶段真实优化器更新在 CUDA 上执行。

已验收产物：`artifacts/smoke-gpu/checkpoint_000000032.pt` 和 `artifacts/smoke-gpu-resumed/checkpoint_000000064.pt`。不要重新用这些已存在目录启动新运行。

## 恢复训练

以下从已验收的 32 步权重继续到累计 64 步，输出使用新目录：

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config artifacts\smoke-gpu\config.yaml --resume artifacts\smoke-gpu\checkpoint_000000032.pt --total-steps 64 --device cuda --out artifacts\resume-new
```

恢复模型、AdamW、主进程 RNG、next_update 和 environment_steps；多项式调度器按恢复的 update 继续计算。严格校验配置。恢复从初始状态重开 episode，不恢复中途模拟器 RAM、worker RNG 或未结束 episode，不承诺逐位一致的轨迹。仅加载可信本地 `.pt`。

## 观看 AI 玩游戏

```powershell
# 使用已完成真实 Smoke 和恢复验证的权重
.\.venv\Scripts\python.exe scripts\watch.py --checkpoint artifacts\smoke-gpu-resumed\checkpoint_000000064.pt --device cuda --speed 1 --steps 300
# 同一权重可在 CPU 推理
.\.venv\Scripts\python.exe scripts\watch.py --checkpoint artifacts\smoke-gpu-resumed\checkpoint_000000064.pt --device cpu --speed 2 --steps 300
# 可选视频：在观看命令追加 --video artifacts\watch-new.mp4
```

`--speed 1` 约 60 Game Boy 帧/秒（2.5 动作/秒）。窗口和终端明确标记未训练用途。入口实现 Q/Esc、关闭窗口、Ctrl+C 退出和达到 `--steps` 自动结束；本轮实测了画面及步数上限正常结束，交互退出键没有完成独立验收（Esc 停止了 Computer Use）。不调用 LLM。

本机已保存 `artifacts/watch-gpu.mp4`：150 帧，160×144，读取解码通过；它是本地游戏素材，不上传。未提供论文兼容预训练权重。

## 日志、配置与结果边界

训练支持 `--config`、`--seed`、`--total-steps`、`--device`、`--out`、`--resume`；输出 `config.yaml`、`environment.json`（Git SHA、输入 hash、seed、硬件）、`metrics.jsonl`、`tensorboard/`、`checkpoint_*.pt`、`result.json`。

```powershell
.\.venv\Scripts\tensorboard.exe --logdir artifacts --port 6006
```

TensorBoard 可读真实训练的 Reward、Loss、Learning Rate、Environment Steps、参数变化和 Checkpoint 路径。

论文参数独立保留在 `configs/paper_baseline.yaml`，原 `poke_red_v2.yaml` 未覆盖。代码和论文的奖励差异见 `docs/baseline_comparison.md`。32×2048 的论文批次仅 screen rollout 就约 4.22 GiB，还需其他内存；本阶段未运行此配置。

**完成的是基础运行复现，不是论文实验成绩复现。** 没有通关、城市里程碑成功率、大规模实验或长时间训练。验收见 `docs/reports/phase1_validation.md`，进度见 `docs/reports/progress.md`，接管见 `docs/HANDOFF.md`。ROM、状态、权重、构建源码/工具、日志和视频均被 Git 忽略；不得用 `git add -f` 纳入。
