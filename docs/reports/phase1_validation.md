# 第一阶段验收记录

日期：2026-10-17，Asia/Shanghai。

**结论：CPU 基础环境及不依赖 ROM 的管线联调通过；真实 Pokémon 游戏、真实 PPO 更新、真实 Checkpoint 恢复和观看仍阻塞。用户明确确认没有 ROM。没有进行真实游戏训练，没有复现论文成绩，没有发布 GitHub。**

## 路径与来源

- 项目：`C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`
- Python：`.venv\Scripts\python.exe`，Python 3.10.11。
- 论文：[arXiv 2502.19920 v2](https://arxiv.org/abs/2502.19920)。
- 上游：[MarcoMeter/neroRL `poke_red`](https://github.com/MarcoMeter/neroRL/tree/poke_red)，SHA `1d5c2db1f47f29d501ec9eacdc9c7f6480ea6217`。
- 本地实现提交：`966d44245a6cfcc31e2c396a3f9a3aab13d0be83`；验收文档另有后续提交。
- 本地独立 Git 历史不含上游 `.state`。原始 `.git` 只归档到忽略目录 `local/upstream.git`；`UPSTREAM.json` 保留版本来源。MIT 和 Peter Whidden 的 MIT 许可证均保留。

## 实际环境

| 项目 | 结果 |
|---|---|
| CPU | Intel Core i9-13900HX，24 核 / 32 线程 |
| 系统内存 | 16,858,472,448 bytes，约 15.7 GiB |
| GPU | NVIDIA RTX 4060 Laptop，8,188 MiB；驱动 592.82；nvidia-smi 正常 |
| 当前 PyTorch | 2.5.1+cpu，CPU 张量乘法及 backward 通过 |
| CUDA | 驱动报告 CUDA 13.1，但当前 torch 为 CPU 构建，CUDA 张量运算未验证 |
| PyBoy / Gymnasium | 2.4.0 / 0.29.0 |
| neroRL | 1.0.0，本地 editable 安装 |
| Git | 2.55.0.windows.5 |
| 其他 | NumPy 1.26.4、OpenCV 4.10.0.84、TensorBoard 2.18.0 |

CUDA 2.45 GB wheel 下载耗时且没有及时进度，终止后改用同版本 205.5 MB CPU wheel；没有反复重装，也没有等待训练。已清除停止下载的部分 CUDA wheel。切换 CUDA 的命令保留在 README，未执行验证。

## 实际执行的环境命令

在 PowerShell 中，工作目录为上述项目目录，主要命令如下（安装顺序有局部依赖补齐）：

```powershell
py -0p
git --version
nvidia-smi
Get-CimInstance Win32_Processor
Get-CimInstance Win32_ComputerSystem
git clone --branch poke_red --single-branch --depth 1 https://github.com/MarcoMeter/neroRL.git C:\Users\lihao\Desktop\pokemon-red-rl-reproduction
git rev-parse HEAD
python -m venv .venv

# 此 CUDA 安装被终止，没有报告成功
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# 实际成功的 CPU 安装
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu --timeout 30 --retries 1
# 实际固定安装的最小依赖
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 gymnasium==0.29.0 pyboy==2.4.0 einops==0.8.0 ruamel.yaml==0.18.6 scikit-image==0.24.0 mediapy==1.2.2 psutil==6.1.1 tblib==3.0.0 tensorboard==2.18.0 docopt==0.6.2 pytest==8.3.5 opencv-python==4.10.0.84 reprint==0.6.0
.\.venv\Scripts\python.exe -m pip install wheel==0.45.1
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.\.venv\Scripts\python.exe -m pip install windows-curses==2.4.1
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe -m compileall -q reproduction scripts tests/phase1
.\.venv\Scripts\python.exe -m pytest tests/phase1 -q --junitxml=artifacts/phase1-junit.xml
```

`pip check` 最初发现上游 Windows curses 依赖缺失，安装精确版本后输出 `No broken requirements found.`。首轮 7 passed / 1 skipped；补充实际 launcher 输出与恢复测试后，最终为 **8 passed / 1 skipped，4.82 秒**。唯一跳过项是 CUDA PPO 测试。SDL2 提示使用 pysdl2-dll 为普通警告。

一键重建脚本 `scripts/setup.ps1` 已编写；本阶段按上述分步安装并验证，没有另外销毁 venv 重跑一次安装。完整实际 CPU 依赖见 `requirements-lock-cpu.txt`。

## 通过的测试及证据

| 检查 | 已执行结果 | 范围 |
|---|---|---|
| 依赖导入、CPU matmul/backward | passed | doctor，真实 CPU torch |
| PyBoy 创建/tick/save_state/load_state/close | passed | 临时生成的原创诊断卡带，不含 Pokémon 内容 |
| Observation / Action / Reward / budget / close | passed | 确定性合成 RAM fixture，不是真实游戏 |
| 官方完整共享 Actor-Critic 前向 | passed | 2,086,600 参数，输出 [2,7] 概率及 [2] Value |
| PPO 更新、梯度及参数变化 | passed | 原版 PPOTrainer + 原版 sampler/buffer/GAE；同步测试 transport + 合成 RAM |
| Checkpoint 写入、严格加载、AdamW 状态一致 | passed | 合成数据 checkpoint |
| 恢复后的再次更新 | passed | 继续 next_update=1，参数再次变化，Loss 有限 |
| 实际 train.main 输出与 resume | passed | 仅测试内替换 ROM 检查与模拟器；生成配置、环境信息、JSONL、TensorBoard、checkpoint |
| TensorBoard 读写 | passed | SummaryWriter 写入，EventAccumulator 读取 Loss 与 environment_steps |
| 配置隔离 | passed | 官方 YAML 原样保存，论文参数与 Smoke 分开 |
| OpenCV 窗口打开/关闭 | passed | 纯色诊断画面，不是游戏；未通过肉眼观察游戏 |
| MP4 encode/decode | passed | 3 帧 160×144 诊断视频；不是游戏录像 |

合成 CPU 测试结果：首次 loss `0.037116777151823044`，恢复后 loss `0.2045879364013672`，均有限。每段测试只做必要更新，合成测试一共两轮；补充 launcher 测试也只做两轮，没有真正的游戏步数。

本地证据（均忽略，不上传）：

- `artifacts/doctor.json`
- `artifacts/phase1-junit.xml`
- `artifacts/synthetic-cpu/result.json`（明确 `synthetic: true`, `real_game: false`）
- `artifacts/synthetic-cpu/checkpoint.pt`（仅单元测试，不能当真实训练成果）
- `artifacts/synthetic-cpu/tensorboard/`
- `artifacts/display-probe.mp4`（纯色测试，不是游戏录像）
- `artifacts/missing-rom-checks.json`

额外窗口验证实际调用了 `cv2.namedWindow/imshow/waitKey/getWindowProperty/destroyAllWindows`；用 `VideoWriter(mp4v)` 写入 3 帧，`VideoCapture` 解码检查为 `(144,160,3)`。

## 缺 ROM 的真实入口验证

以下三个命令实际执行，均在启动 worker 或训练前报 `Missing lawful Pokemon Red ROM`，退出码非零，属于**缺输入保护通过**，不属于游戏运行通过：

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\smoke.yaml --total-steps 32 --device cpu --out artifacts\smoke-missing-rom
.\.venv\Scripts\python.exe scripts\watch.py --untrained --steps 7 --device cpu
.\.venv\Scripts\python.exe scripts\check_game.py --steps 7
```

没有下载 ROM。项目、桌面/下载/本次 Codex 工作区的 .gb/.gbc 文件检查未找到 ROM；用户也确认没有。`local/PokemonRed.gb` 和 `local/has_pokedex_nballs.state` 目前均不存在。上游已有的状态仅留作忽略文件，未作为公开内容或已验收输入。

## Observation / Action / Reward 合约

| 模态 | 底层 | 模型输入 | dtype / 范围 |
|---|---|---|---|
| screens | 72×80×3，当前帧及前两帧 | 3×72×80 | float32，[0,1] |
| map | 48×48×1，探索 crop | 1×48×48 | float32，[0,1] |
| events | 上游 mask 选中 827 位 | 827 | float32，0/1 |
| game_state | 6 个 HP 比例 + 6 个 level/100 | 12 | float32，声明 [-1,1] |

动作下标保持 `DOWN / LEFT / RIGHT / UP / A / B / START`，不是按用户列举顺序重新编码。每个 action 执行 24 模拟帧（按 8、松 16）。预算用尽产生 truncation，经 neroRL 合并为 done。

论文参数配置的 event 增量 2、coordinate 增量 .005、level 系数 .5、heal 系数 2.5，均保留原版奖励函数。代码和论文在 level 偏移、heal 聚合/过滤和事件延长 budget 等方面有实质差异，已独立记录在 `docs/baseline_comparison.md`，不能称作严格数值复现。

## 实现与限制

- 局部修复：wrapper 观测 float32、底层 health/level/events dtype；原空 close 正常停止 PyBoy，且可重复关闭；headless 不限速；worker rank/worker_id/record_video 参数传递和退出清理；PPO 非有限 loss/梯度在 optimizer.step 前拒绝。
- 不改网络、PPO clipped objective、GAE、AdamW 或原奖励算法；没有引入 LLM、训练监督平台或多 Agent 系统。
- Checkpoint 附带配置、seed、模型、AdamW、进度、主进程 RNG、环境来源；调度从 next_update 继续。恢复新建 episode，不保存中途模拟器与 worker RNG，因此不是轨迹逐位恢复。
- Smoke 为 2×16，1 epoch / 2 minibatches，累计硬上限 256；默认单次 32 步。论文配置 32×2048 没有运行，当前资源不宜直接执行。
- capture_state、真实游戏 OpenCV 观看/速度/退出/录像及实际 multiprocessing Pokémon workers 尚待 ROM 验收；不能从合成同步 transport 测试推断它们全部通过。
- 上游其他可选环境的测试未执行，范围仅 `tests/phase1`。

## 后续必做（拿到合法输入后）

1. 配置 `local/PokemonRed.gb` 和兼容 `local/has_pokedex_nballs.state`，确认 Baseline 起点。
2. `scripts/check_game.py --steps 128`：真实 reset、连续 step、RAM、render、budget 截断和 close。
3. `scripts/train.py --config configs/smoke.yaml --seed 0 --total-steps 32 --device cpu --out artifacts/smoke`。
4. `scripts/train.py --config artifacts/smoke/config.yaml --resume artifacts/smoke/checkpoint_000000032.pt --total-steps 64 --device cpu --out artifacts/smoke-resumed`。
5. `scripts/watch.py --checkpoint artifacts/smoke/checkpoint_000000032.pt --speed 1 --steps 100 --device cpu`，确认肉眼可见游戏和 Q/Esc/窗口退出；仍明确标记未训练用途。
6. 更新本报告与 progress；若需要 CUDA，在独立终端安装并复测。上述最小验证后即停，不启动长训。

核心真实游戏验收未完成，因此按用户条件保留本地仓库，未创建 GitHub Public 仓库，也未因公开发布去获取或上传游戏素材。
