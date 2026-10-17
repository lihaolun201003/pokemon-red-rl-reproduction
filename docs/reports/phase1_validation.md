# 第一阶段最终验收：英文原版 + 真实 CUDA PPO

实测日志日期：2026-10-17，Asia/Shanghai。随后仅整理证据与文档，没有追加训练。

**结论：基础运行目标通过。标准英文 ROM、论文官方初始状态、真实环境采样、RTX 4060 PPO 更新、Checkpoint 保存/加载/恢复、CPU/CUDA 观看及 TensorBoard 均已实际验证。训练采样总量仅 64 步（首次 32，恢复后再 32），没有开展论文成绩实验或长时间训练。**

## 1. 本地项目及固定版本

- 项目：`C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`
- 解释器：`.venv\Scripts\python.exe`，Python 3.10.11。
- neroRL：官方 `poke_red` 分支，SHA `1d5c2db1f47f29d501ec9eacdc9c7f6480ea6217`。
- PyTorch `2.5.1+cu121`、PyBoy `2.4.0`、Gymnasium `0.29.0`。
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GB，驱动 592.82；CUDA available=True。
- CPU：i9-13900HX，32 逻辑线程；RAM 约 16 GB。
- 沿用已有仓库、venv 和算法，没有重装 Python、改变奖励或改写内存地址。

旧的缺 ROM / 中文修改版失败记录分别保留在 `phase1_validation_initial.md`、`rom_cuda_validation_previous.md`，不能把这些历史状态当作当前结论。中文修改版没有继续适配。

## 2. 从源码构建英文原版：通过

用户已明确确认具备在本机从源码构建并使用英文 ROM 的合法使用权。

官方来源：[pret/pokered](https://github.com/pret/pokered)，安装说明：[INSTALL.md](https://github.com/pret/pokered/blob/master/INSTALL.md)。固定源码 SHA：

`af519899719f0754965776faac0e836a3b906e6d`

工具检查结果：Git 2.55 正常；系统存在 wsl.exe，但 WSL 本身未安装，Make/RGBDS 不在 PATH。Cygwin 的项目内静默安装尝试被自动审批审查拒绝（仅返回 blocked by policy），未重试该安装器。改为无需安装、无需系统配置的便携工具链，成功完成构建。

| 工具 | 版本 | 来源与 SHA256 |
|---|---|---|
| RGBDS | 1.0.4，与 `.rgbds-version` 一致 | [gbdev/rgbds v1.0.4](https://github.com/gbdev/rgbds/releases/tag/v1.0.4)，win64 zip：`bd1d7e386f290afd57f4ea9a7353bfcd4ae10884a94c5c1f2fc42a75db7f97e6` |
| w64devkit | 2.10.1，GNU Make 4.4.1 / GCC 16.2.0 | [官方发布](https://github.com/skeeto/w64devkit/releases/tag/v2.10.1)，x64 archive：`add1edf984ce4d473cfeb7c52e0ab38db73a855bba6c7f564d5e16648eb2896a` |

工具下载后均核对官方 release 提供的 hash；Windows tar 解压便携工具，不执行安装器、不设置全局 PATH。源码和工具只在忽略的 `work/` 下。

实际构建运行 `make -j2 red`，通过 `RGBDS=` 指定本地 1.0.4 工具；构建、版本与复制检查封装为 `scripts/build_red.ps1`。

| 文件 | 大小 | SHA1 |
|---|---:|---|
| `work/pokered/pokered.gbc` | 1,048,576 bytes | `ea9bcae617fdf159b045185467ae58b2e4a48b9a` |
| `local/PokemonRed.gb` | 1,048,576 bytes | `ea9bcae617fdf159b045185467ae58b2e4a48b9a` |

哈希与目标完全一致，仅复制改文件名，未修改 ROM 字节。没有下载任何预编译游戏 ROM。构建日志与 manifest：`artifacts/build_red.log`、`artifacts/rom_build.json`。

## 3. 官方初始状态与游戏数据：通过

状态：`local/has_pokedex_nballs.state`，来自固定版本 neroRL；SHA256：

`0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e`

直接 PyBoy 加载、两帧继续执行、官方 wrapper reset、32 步逐步语义检查和额外 128 步连续运行均正常。曾在中文修改版出现的 RAM 异常未再出现。

| 数据 | 实测 |
|---|---|
| 初始坐标 X / Y / map | 5 / 3 / 40 |
| 队伍数量 / 首只 species | 1 / 177（Squirtle） |
| Level / HP / max HP | 6 / 22 / 22 |
| badges | 0 |
| 非零事件字节 / masked event flags | 4 / 11 |
| 连续 32 步奖励总和 | 0.015 |
| 连续 128 步奖励总和 | 0.015 |
| reset / render / close | 正常；重置后恢复初始数据 |
| 时间预算耗尽逻辑 | 通过临时缩短测试实例剩余预算触发 truncation，再 reset；没有改训练配置或奖励 |

动作下标保持官方 `DOWN / LEFT / RIGHT / UP / A / B / START`，7 个动作全部执行；每次按 8 帧、松 16 帧。

| 观测模态 | 模型输入 shape | dtype / 范围 |
|---|---|---|
| screens | `(3,72,80)` | float32，[0,1]，当前和前两帧 |
| map | `(1,48,48)` | float32，[0,1] |
| events | `(827,)` | float32，0/1 |
| game_state | `(12,)` | float32，声明 [-1,1]，6 个 HP 比例 + 6 个 level/100 |

证据：`artifacts/english-rom-compatibility/report.json`、`artifacts/english_game_check.json`。没有写入游戏 RAM、伪造事件或修改地址。

## 4. 真实 PPO 更新和恢复：通过

配置 `configs/smoke.yaml`：2 workers × 16 steps、1 epoch、2 minibatches，完整官方 2,086,600 参数模型，优化器 AdamW；采样在 CPU，优化在 CUDA。未改变 PPO 损失、GAE 或奖励函数。

| 运行 | 累计训练步 / update | Loss | 最大参数变化 | 梯度范数 |
|---|---|---:|---:|---:|
| 首次真实更新 | 32 / 1 | 0.4105313420 | 0.0006017238 | 0.49999991 |
| 加载后再次更新 | 64 / 2 | 0.1887938976 | 0.0006023832 | 0.49999988 |

Loss、梯度和参数均有限，参数确实改变。每份 Checkpoint 保存后重新读取，模型权重一致；恢复载入模型、28 组 AdamW 参数状态、RNG、next_update 和 environment_steps；调度器按恢复 update 继续，Learning Rate 为 0.0003。

真实 Checkpoint：

- `artifacts/smoke-gpu/checkpoint_000000032.pt`
- `artifacts/smoke-gpu-resumed/checkpoint_000000064.pt`

两份 provenance 都记录 `device=cuda`、标准 ROM hash、初始状态 hash、seed=0、配置与来源。运行时 Git SHA 为 `9c26fc5...`，dirty=true（本轮文档/构建脚本准备中）；未改动 RL 模型、采样器或 PPO 源码。

恢复会新建 episode，不恢复运行中的模拟器 RAM、worker RNG 或未结束 episode，因此不是轨迹逐位恢复。环境检查、启动前 7 个动作检查和观看中的游戏动作不计入这 64 个训练采样步。

## 5. 观看、视频与正常结束

`watch.py` 实际加载了真实 Checkpoint，模型自动输出动作。已用 computer-use 查看 CUDA 和 CPU 窗口，见到玩家移动、英文菜单及有效 HP/Level；没有用 LLM 控制动作。

- CUDA：150 步正常结束，累计奖励 0.045，日志为 `artifacts/watch-gpu.log`。
- CPU：100 步正常结束，累计奖励 0.030，日志为 `artifacts/watch-cpu.log`。
- GPU 视频 `artifacts/watch-gpu.mp4`：150 帧，160×144，OpenCV 解码通过。
- 额外退出检查中的物理 Esc 停止了 Computer Use；没有把此事件当作游戏退出键验收。Q/Esc/关闭窗口逻辑已实现，但独立交互退出键验证未完成；达到 `--steps` 上限后的正常清理和结束已通过。

所有 Smoke 权重和窗口均明确标记 **Untrained Agent — 仅用于运行测试**。观看正常不能代表模型学会了游戏。

## 6. 自动测试与 TensorBoard

`pytest tests/phase1`：**11 passed，0 failed，0 skipped，4.61 秒**。仅有 SDL2 binaries 来源提示。

涵盖 CPU/CUDA 模型与合成 PPO 更新、保存/恢复、Observation/Action/Reward fixture、TensorBoard、launcher 产物和错误 RAM 拒绝等。真实验收由上述独立命令执行，未混入合成测试数量。

重新读取了真实 TensorBoard 文件：Loss、Reward、Learning Rate、environment_steps、参数变化在 32/64 步均存在且与 JSONL 一致。证据 `artifacts/real_tensorboard_readback.json`、`artifacts/final_acceptance_readback.json`、`artifacts/phase1-english-junit.xml`。

## 7. 实际执行的主要命令

在项目目录执行（构建工具下载 URL/hash 见上表）：

```powershell
git clone --depth 1 https://github.com/pret/pokered.git work/pokered
git -C work/pokered rev-parse HEAD
Get-Content work/pokered/.rgbds-version
tar.exe -xf work/w64devkit-x64-2.10.1.7z.exe -C work
.\scripts\build_red.ps1
.\.venv\Scripts\python.exe scripts/probe_rom.py --rom local/PokemonRed.gb --out artifacts/english-rom-compatibility
.\.venv\Scripts\python.exe scripts/check_game.py --steps 128 --out artifacts/english_game_check.json
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke.yaml --seed 0 --total-steps 32 --device cuda --out artifacts/smoke-gpu
.\.venv\Scripts\python.exe scripts/train.py --config artifacts/smoke-gpu/config.yaml --resume artifacts/smoke-gpu/checkpoint_000000032.pt --total-steps 64 --device cuda --out artifacts/smoke-gpu-resumed
.\.venv\Scripts\python.exe -m pytest tests/phase1 -q --junitxml=artifacts/phase1-english-junit.xml
.\.venv\Scripts\python.exe scripts/watch.py --checkpoint artifacts/smoke-gpu/checkpoint_000000032.pt --device cuda --speed 1 --steps 150 --video artifacts/watch-gpu.mp4
.\.venv\Scripts\python.exe scripts/watch.py --checkpoint artifacts/smoke-gpu-resumed/checkpoint_000000064.pt --device cpu --speed 1 --steps 100
```

已有输出目录不要覆盖。用户观看命令：

```powershell
.\.venv\Scripts\python.exe scripts/watch.py --checkpoint artifacts/smoke-gpu-resumed/checkpoint_000000064.pt --device cuda --speed 1 --steps 300
```

## 8. 范围、Git 与后续工作

本阶段只验证训练管线。没有通关或城市里程碑成功率，没有大规模实验、百万步训练或训练监督系统；论文与固定源码的奖励差异仍见 `docs/baseline_comparison.md`，不能宣称严格数值复现。

ROM、状态、模型、日志、录像及 pret 构建源码/工具均保留在 Git 忽略目录。只提交脚本、说明、配置和文本报告；保留 MIT/NOTICE。本次完成本地整理，未发布 GitHub Public 仓库。

后续 Agent 可直接使用当前环境与累计 64 步的真实 Checkpoint。长训应在新任务授权、资源评估和独立终端中安排；Smoke 配置累计上限仍是 256，不会自动扩容。当前没有后台训练。
