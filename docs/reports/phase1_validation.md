# 第一阶段续验：ROM 兼容性与 RTX 4060 CUDA

日期：2026-10-17（Asia/Shanghai）。沿用现有仓库和 `.venv`，没有重建项目。

**结论：CUDA 安装及 CPU/GPU PPO 管线联调成功；当前中文修改版 ROM 可以启动和显示，但与官方初始状态不兼容。没有执行真实 32 步 PPO 训练，没有真实游戏 Checkpoint，没有把合成训练结果当作游戏复现。**

## 路径和来源

- 项目：`C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`
- Python：`.venv\Scripts\python.exe`，3.10.11。
- 官方分支：`poke_red`，固定 SHA `1d5c2db1f47f29d501ec9eacdc9c7f6480ea6217`。
- 本轮诊断/检查脚本提交：`8bd598875224c2848c04f7a2082fb0bd9147b528`；报告与文档另有后续提交。
- 先前 CPU/缺 ROM 阶段记录保留为 `docs/reports/phase1_validation_initial.md`，当前状态以本文件为准。

## 1. ROM 与官方状态：不兼容

| 检查 | 实际结果 |
|---|---|
| 用户 ROM | `local/PokemonRed.gbc`，2,097,152 bytes（2 MiB） |
| ROM SHA1 | `944ff0b461fb570d2c8aa99d819e8c8df0bdc446` |
| ROM SHA256 | `f4092c50b04a0c471c8e1cda4af1551ac451c754be8ae03f86430c9c0dc0e073` |
| 内部标题 / header checksum | `POKEMON RED` / 通过 |
| ROM header | CGB flag=0，mapper=0x13，ROM size code=0x06，RAM size code=0x03 |
| 原始启动 | PyBoy 2.4.0 显示启动动画、中文标题和菜单；SDL 原生窗口可见中文大木博士剧情 |
| 按键 | Enter 对应 Start；实际进入游戏；固定 8 帧 press、120 帧 release 也进入中文菜单 |
| 论文预期 ROM SHA1 | `ea9bcae617fdf159b045185467ae58b2e4a48b9a`；当前输入不匹配 |

`.gbc` 扩展名和英文内部标题不能证明是论文使用的英文原版。实际画面为中文修改版，ROM 大小也不同于预期版本。没有裁剪、修补或重命名 ROM 来伪装原版。

按用户授权，将固定上游已检出的 `neroRL/environments/poke_red/has_pokedex_nballs.state` 原样复制至 `local/has_pokedex_nballs.state`。两份 SHA256 相同：

`0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e`

`PyBoy.load_state` 没有抛异常，wrapper reset 后初始观测也符合数值 shape/range；但继续执行失败：

| 官方 RAM 地址含义 | 刚 load | 两个模拟帧后 |
|---|---:|---:|
| 位置 X / Y（D362 / D361） | 5 / 3 | 0 / 57 |
| 地图（D35E） | 40 | 0 |
| 队伍数量（D163） | 1 | 57（非法） |
| 首只宝可梦 species（D164） | 177（Squirtle） | 0 |
| HP / max HP（D16C / D18D） | 22 / 22 | 57 / 14592（非法） |
| Level（D18C） | 6 | 0（非法） |
| 非零事件字节数（D747..D886） | 4 | 160 |
| CPU PC | 0x20B3 | 0x0038，后续多帧仍停留此处 |

官方环境第一步 DOWN 复现同样 RAM 异常，观测事件和变成 151，奖励异常跳到 `1258.005`。持续运行后画面成为灰屏。该现象在直接 PyBoy 状态加载和官方 wrapper reset/step 中均可复现。

**判定：状态文件可反序列化，但这份 ROM 与该官方状态的继续执行不兼容。** 很可能与修改版代码/布局不同有关；没有原版对照，无法进一步断定具体补丁位置。未把“没有崩溃”“shape 正确”或“奖励有限”当作兼容通过。

新增只读诊断 `scripts/probe_rom.py`：允许针对未知 hash 做诊断，不放宽训练入口；检查每一步 party/HP/Level 语义，失败即停止并返回 2。ROM、状态、内存地址、奖励函数均未修改。reset 的初始观测正常不代表 step 可用。

初始观测：screens float32 `(3,72,80)`、map `(1,48,48)`、events `(827,)`、game_state `(12,)`；动作仍为官方 `DOWN/LEFT/RIGHT/UP/A/B/START`。后续 step 的数值虽然仍可能在 Box 范围内，游戏语义已无效，故整体观察/奖励验收失败。

## 2. CUDA 安装与验证：通过

硬件：i9-13900HX，约 16 GB RAM，NVIDIA GeForce RTX 4060 Laptop GPU（8 GB），驱动 592.82，`nvidia-smi` 正常。

从 [PyTorch 官方 cu121 索引](https://download.pytorch.org/whl/cu121/torch/) 下载 `torch-2.5.1+cu121-cp310-cp310-win_amd64.whl`，使用断点续传、连接超时 20 秒、低速超时 60 秒、总超时 900 秒及最多 2 次重试。本次一次下载完成，约 7 分 16 秒。

下载大小 `2,449,372,784` bytes；SHA256 与官方索引一致：

`9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f`

校验后才在项目 `.venv` 中执行本地 wheel 安装，未删除 venv，未更改系统 Python。系统 Python 的 `find_spec('torch')` 仍为 None。

| 实测项 | 结果 |
|---|---|
| torch version | `2.5.1+cu121` |
| torch.version.cuda | `12.1` |
| torch.cuda.is_available | True |
| GPU name | NVIDIA GeForce RTX 4060 Laptop GPU |
| CPU 与 CUDA Tensor matmul/backward | 通过，梯度有限 |
| 完整官方 Actor-Critic CPU/CUDA 前后向 | 通过，2,086,600 参数 |
| GPU 官方 PPO 更新及恢复 | 合成 RAM 数据通过；真实游戏不兼容，未执行 |
| pip check | No broken requirements found |

CUDA PPO 单元测试第一轮 loss `0.03746623545885086`、恢复后 loss `0.19939640164375305`；检查所有 loss/梯度有限、参数实际改变、模型和 AdamW 状态一致、恢复后参数继续变化。数据明确为 `synthetic: true, real_game: false`。

首次 CUDA DLL 导入较慢，最终成功；不是仅凭安装命令成功或 nvidia-smi 就判为通过。当前安装支持 `--device cpu` 和 `--device cuda`，两者均实测。依赖锁 `requirements-lock-cuda.txt` 已去除本地 wheel 绝对 URL，并保留精确版本及官方索引。

## 3. 实际执行命令

以下均在现有项目目录中执行：

```powershell
nvidia-smi
Copy-Item neroRL/environments/poke_red/has_pokedex_nballs.state local/has_pokedex_nballs.state
Get-FileHash -Algorithm SHA256 neroRL/environments/poke_red/has_pokedex_nballs.state,local/has_pokedex_nballs.state
.\.venv\Scripts\python.exe scripts/probe_rom.py --rom local/PokemonRed.gbc
# 预期退出码 2：兼容性拒绝；报告在 artifacts/rom-compatibility/

curl.exe --fail --location --continue-at - --retry 2 --retry-delay 2 --connect-timeout 20 --speed-time 60 --speed-limit 1024 --max-time 900 --output "work/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl" "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl"
.\.venv\Scripts\python.exe work/verify_wheel.py
.\.venv\Scripts\python.exe -m pip install --no-deps --timeout 30 --retries 2 "work/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
.\.venv\Scripts\python.exe scripts/check_cuda.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/phase1 -q --junitxml=artifacts/phase1-cuda-junit.xml
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe -m compileall -q reproduction scripts tests/phase1
```

另外运行了忽略目录中的 `work/state_diag.py`，逐帧读取 PC/SP 和原 RAM 地址，没有写入状态。通过 `work/preview_rom.py` 打开真实 PyBoy SDL 窗口，用 computer-use 检查实际画面并按 Enter，观察到中文剧情；窗口有自动结束期限，没有留后台模拟或训练。

生产入口的 `validate_inputs(..., 'local/PokemonRed.gbc', ...)` 仍明确报 `Unsupported ROM SHA1`；未添加白名单或静默绕过。用户拿到匹配版本后可直接使用原 README 的 32 步训练、恢复和 watch 命令。

## 4. 测试数量与验收边界

**自动测试：11 passed，0 failed，0 skipped，5.53 秒。** 一条 SDL2 binaries 来源提示，不是错误。比之前增加了 CUDA PPO 用例实际执行和两项 RAM 语义回归检查。

通过内容：PyBoy 原创诊断卡带、官方 Observation/Action/Reward 的合成 fixture、CPU/CUDA 官方 PPO 更新及恢复、模型输出、TensorBoard、launcher 产物、论文配置分离、非法 RAM 语义拒绝。

**真实输入验收：ROM+官方状态兼容性 1 项失败。** 原始 ROM 启动/画面/Enter 通过；状态恢复后的游戏语义失败。自动测试通过不等于此真实验收通过。

因用户要求“ROM 兼容性成功后才训练”，没有启动真实 32 步 PPO，没有生成真实 Checkpoint，没有执行模型控制游戏的 `watch.py`。原始模拟器启动画面可见，不应表述成 AI watch 已成功。此前短程固定动作诊断也不属于 PPO 训练。

## 5. 本地证据和下一步

- `artifacts/cuda-wheel-verification.json`：下载来源、大小和校验。
- `artifacts/cuda-validation.json`、`artifacts/doctor.json`：GPU、CPU/CUDA 张量与网络结果。
- `artifacts/phase1-cuda-junit.xml`：11 项测试。
- `artifacts/synthetic-cuda/result.json`、`checkpoint.pt`：仅合成测试。
- `artifacts/rom-compatibility/report.json`、`probe.log`、`cpu_trace.json`：真实失败及逐帧 RAM/PC。
- 同目录 PNG：启动、中文标题/菜单、官方状态载入和失败后画面。均为忽略本地素材，不提交。

后续需要合法、匹配的英文原版 ROM。无需重装环境。先运行 check_game，再做 32 步训练、恢复至累计 64 步、watch；最小验收后停止。不能仅给当前修改版另存一个状态就宣称论文环境复现成功。

本轮保留本地仓库，更新 README、progress、HANDOFF。核心真实验收失败，故未发布 GitHub Public 仓库。ROM、游戏状态、截图、模型、wheel 均不在 Git 文件或提交历史中；未修改论文算法，未进行长训。
