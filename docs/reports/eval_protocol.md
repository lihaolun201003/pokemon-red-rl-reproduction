# 独立评估协议 v1（`scripts/eval_checkpoint.py`）

本文件定义 Pokemon Red PPO 复现项目在训练里程碑 checkpoint 上使用的**固定评估协议 v1**，并记录脚本实现、自测结果与已知限制。评估脚本只做推理，不做任何训练：

- 不更新权重、不创建/不加载 optimizer、不写训练 run 目录、不改动 `neroRL/`、`configs/`、`local/`。
- 结果全部为实际运行测量值，不做筛选、平滑或手工修饰。
- 写权限范围：`scripts/eval_checkpoint.py`、本文件、`artifacts/training/eval_smoke/`。正式训练的评估输出目录由 Lead 指定。

---

## 1. 命令

解释器：`.venv\Scripts\python.exe`；工作目录：仓库根目录（脚本内部会把环境的 scratch `./session` 目录重定向到输出目录，不污染仓库根）。

### 1.1 checkpoint 评估（正式协议，默认 3 stochastic + 1 deterministic）

```powershell
.venv\Scripts\python.exe scripts\eval_checkpoint.py `
    --checkpoint <checkpoint.pt> --out <out_dir>
```

### 1.2 未训练基线

```powershell
.venv\Scripts\python.exe scripts\eval_checkpoint.py `
    --untrained --config configs\train_seed0.yaml --out <out_dir>
```

### 1.3 均匀随机动作基线

```powershell
.venv\Scripts\python.exe scripts\eval_checkpoint.py `
    --random-actions --config configs\train_seed0.yaml --out <out_dir>
```

### 1.4 常规定制（示例）

```powershell
# 只跑确定性 episode（--episodes 控制个数）
.venv\Scripts\python.exe scripts\eval_checkpoint.py --checkpoint <ckpt.pt> `
    --steps 2048 --episodes 1 --no-stochastic --out <out_dir>

# 录视频 + 25%/50%/100% 截图（对 deterministic episode）
.venv\Scripts\python.exe scripts\eval_checkpoint.py --checkpoint <ckpt.pt> `
    --video <out_dir>\deterministic.mp4 --out <out_dir>

# 指定设备 / 种子 / ROM / 初始存档（--rom/--state 覆盖配置中的路径）
.venv\Scripts\python.exe scripts\eval_checkpoint.py --checkpoint <ckpt.pt> `
    --device cpu --seed 100000 --rom .\local\PokemonRed.gb --state .\local\has_pokedex_nballs.state --out <out_dir>
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--checkpoint PATH` | 必选（三选一） | 训练 checkpoint；其 `configs` 为权威配置 |
| `--untrained` | — | 不加载权重，按配置随机初始化模型 |
| `--random-actions` | — | 均匀随机动作，不构建模型 |
| `--config PATH` | `configs/train_seed0.yaml` | 仅 `--untrained`/`--random-actions` 使用；`--checkpoint` 时忽略并提示 |
| `--out DIR` | 必选 | 输出目录 |
| `--episodes N` | 3 | stochastic episode 数（`--no-stochastic` 时改为确定性 episode 数） |
| `--steps N` | 2048 | 每 episode 最多步数 |
| `--seed N` | 100000 | 基准种子；episode i 用 `seed+i` |
| `--device auto|cpu|cuda` | auto | 模型推理设备（环境始终 CPU） |
| `--video PATH` | 无 | 录制最后一个 episode 为 mp4（默认即 deterministic episode）并保存截图 |
| `--rom/--state` | 配置值 | 覆盖 ROM / 初始存档路径 |
| `--no-stochastic` | 关 | 只跑 argmax 确定性 episode |
| `--overwrite` | 关 | 允许覆盖已存在的 `results.json` / 视频（默认拒绝，防误覆盖历史评估） |

**GPU 协调规则（Lead 要求）**：正式训练运行期间，所有冒烟测试只用 `--device cpu`，不运行任何 CUDA 测试；只有 Lead 明确通知“训练已暂停，可用 GPU 评估”的窗口才用 `--device cuda` 做正式评估。CPU/GPU 两种设备下的协议完全相同，仅推理速度与 `protocol.device` 元数据不同。

---

## 2. 协议定义

### 2.1 配置来源

- `--checkpoint`：使用 `checkpoint["configs"]` 原样配置（环境、奖励、模型结构、PPO 超参全部与训练一致），**不覆盖任何 PPO/奖励/环境参数**；只对 ROM/初始存档路径做绝对路径归一化与 SHA1/SHA256 校验。`--config` 被忽略并打印提示。
- `--untrained` / `--random-actions`：使用 `--config`（默认 `configs/train_seed0.yaml`）。
- 仅支持 `PokeRedV2` 环境与 feed-forward 模型；配置里出现 `recurrence` 或 `transformer` 时直接报错退出。
- 动作空间必须是 `Discrete(7)`；动作顺序固定为官方顺序：0 DOWN、1 LEFT、2 RIGHT、3 UP、4 A、5 B、6 START。

### 2.2 Episode 计划与种子

| 模式 | 默认 episode 计划 |
|---|---|
| `--checkpoint` / `--untrained` | N 个 **stochastic**（seed = base+i, i=0..N-1）+ 1 个 **deterministic**（seed = base+N） |
| 加 `--no-stochastic` | N 个 deterministic（seed = base+i） |
| `--random-actions` | N 个随机动作 episode（seed = base+i） |

- 默认 `--episodes 3 --seed 100000`：stochastic 种子 100000/100001/100002，deterministic 种子 100003。
- 每个 episode 都先 `seed_all(episode_seed)`（Python/NumPy/torch/CUDA），env reset 消耗的 Python RNG 也因此可复现。
- `results.json` 同时记录协议种子 `seed` 与环境内部实际使用的 `env_seed`（wrapper 由 `reset_params.start-seed/num-seeds` 采样得到；默认配置下恒为 0）。初始存档固定 Game Boy RNG，本协议不重播种游戏内 RNG。
- **每个 episode 都新建一个 PyBoy 环境实例，reset 官方初始存档**。原因：`load_state` 恢复的是模拟器 RAM，但复用同一个 PyBoy 实例时屏幕渲染缓冲/渲染历史不会完全恢复，实测同一实例上第二个确定性 episode 从第 3 个观测起 `screens` 像素发生变化，argmax 动作在第 5、8 步翻转（RAM 指标 events/game_state/map 完全相同）；改为每 episode 新实例后，两个确定性 episode 的 actions/coords/rewards/info/obs_events_sum 完全一致。新实例成本约 1–2 s/episode，相对 2048 步的 episode 可忽略。

### 2.3 推理与动作选择

- deterministic：`policy.probs.argmax()`（无随机性）。
- stochastic：从 categorical policy 分布采样（`policy.sample()`），每 episode 用 `seed+i` 重新播种 torch（含 CUDA RNG），因此动作序列由种子唯一确定。
- `--random-actions`：Python `random.randrange(7)` 均匀随机，同样按 episode 播种。
- 每步都做 `check_observation` 校验（观测在官方 space 内、全 finite）。

### 2.4 Episode 终止

- 最多 `--steps` 步。
- `terminated_by`：
  - `step_cap`：达到本脚本的 `--steps` 上限（协议预算）；
  - `env_truncation`：达到官方环境自身的 episode 步数预算（默认 `max_steps: 10240`，新事件可 +2048），此时 wrapper 会在 `info` 中给出 `reward/length/events_sum`，脚本记录 `env_reported_reward`、`env_reported_length` 并用 `reward_consistency` 校验与逐步 rewards 之和一致。
- 官方环境只有步数截断（truncation），没有黑屏/全灭/到城市等 terminal。

### 2.5 视频与截图

- `--video PATH` 时录制**最后一个 episode**（默认协议下即 deterministic episode；`--random-actions` 时是最后一个随机 episode）：
  - 160×144 mp4，`cv2.VideoWriter` + `mp4v`，fps = 2.5（1 帧/决策），每步写入一帧，颜色由 PyBoy RGB 转 BGR；
  - 录制后从视频中抽取 **25% / 50% / 100%** 三帧保存 PNG（帧号按实际 episode 长度四舍五入；100% 为最后一帧）。小 episode 下比例会有舍入（如 16 步时 25% 落在第 4 帧=0.267），`results.json` 同时记录 `requested_fraction` 与 `actual_fraction`。

---

## 3. 输出

### 3.1 `out/results.json`

顶层结构：

| 字段 | 内容 |
|---|---|
| `protocol` | 协议元数据：`name`、`schema`、`created_utc`、`mode`、`config_source`、`episodes_requested`、`steps_per_episode`、`seed`、`device`、`seeds`、`stochastic_evaluation`、`action_selection`、`video`、`episode_order`、`totals`、`reward_params`、`rom_sha1`、`state_sha256`、`command`、`cwd`、`environment`（python/torch/numpy/cv2/git_sha/git_dirty/platform/GPU/RAM）、`fresh_environment_per_episode`、`environment_instances`、`limitations` |
| `model` | `source`（checkpoint/untrained/random_actions）、`parameters`、`finite`、`device`、`loaded_from_checkpoint` |
| `checkpoint` | `path`、`sha256`、`size_bytes`、`environment_steps`、`agent_label`、`phase1_format`、`next_update`、`update`、`seed`、`run_id`、`model_parameters`；`--untrained`/`--random-actions` 时为 `null` |
| `episodes` | 每个 episode 的完整记录（见下） |
| `summary` | 按组（stochastic / deterministic / random_actions）的 mean/std/min/max；不同组不混合 |
| `limitations` | 已知限制列表 |

每个 episode 记录的字段：

- 标识：`index`、`mode`、`seed`、`env_seed`、`steps_requested`；
- 结果：`length`、`terminated_by`、`episode_reward`（逐步 reward 之和）、`env_reported_reward`、`env_reported_length`、`reward_consistency`；
- 性能：`reset_seconds`、`stepping_seconds`、`inference_seconds`、`loop_seconds`、`steps_per_second`（episode 步数 / 该 episode 墙钟）；
- 游戏指标：`coord_count`、`unique_map_count`、`maps_first_step`（map → 首次到达步号）、`final_coords`、`badges`（badge_1+badge_2）、`badge_1/2`、`badge_1_steps/badge_2_steps`、`levels_sum`、`party_size`、`deaths`、`max_map_progress`、`mt_moon`、`cerulean`、`max_foe_level`、`max_event_rew`、`event_reward`、`healr`、`obs_events_sum`；
- 动作：`action_counts`、`action_shares`（7 个官方动作的计数/占比）；
- 视频/截图：`video`、`screenshots`（路径相对于 `out`）；
- `trace`：**每步**记录 `actions`、`rewards`、`coords`（`[x, y, map]`）、`obs_events_sum`（`obs["events"].sum()`）以及 `info` 的 13 个指标序列（`deaths / max_foe_level / max_event_rew / party_size / levels_sum / mt_moon / cerulean / event_reward / healr / coord_count / max_map_progress / badge_1 / badge_2`）。

### 3.2 其他输出

- `out/config_snapshot.yaml`：本次实际使用的完整配置快照（checkpoint 模式为 checkpoint 内配置；含环境、模型、trainer、sampler）。
- `out/<prefix>_25pct.png`、`_50pct.png`、`_100pct.png`：`--video` 时的截图（`prefix` 为 `deterministic` 或 `random_actions`）。
- 视频文件位于 `--video` 指定路径（建议放在 `out` 内，`results.json` 记录相对路径）。

---

## 4. 指标含义与口径

| 指标 | 口径 |
|---|---|
| `episode_reward` | 逐步环境 reward 之和（奖励定义来自配置，原样未改：event/level/heal/explore 组合） |
| `coord_count` | 官方 info 的 `coord_count` = episode 内走过的唯一 (x,y,map) 坐标数 |
| `unique_map_count` | 从 `get_game_coords()` 的 map 字节统计的不同地图数 |
| `badges` | `badge_1 + badge_2`（官方 env 只跟踪前两个徽章） |
| `levels_sum` | 官方 info 的 `levels_sum`（队伍等级和，含官方等级奖励的裁剪口径） |
| `party_size` / `deaths` | 官方 info：队伍人数 / 全灭次数 |
| `max_map_progress` | 官方 essential-map 进度（0..14，越大越深入） |
| `mt_moon` / `cerulean` | 是否到过月见山（map 59）/ 华蓝市（map 3） |
| `max_foe_level` / `max_event_rew` / `event_reward` / `healr` | 官方奖励组成部分的累计值 |
| `obs_events_sum` | 最后一步观测中事件 flag 数量（`obs["events"].sum()`） |
| `steps_per_second`（episode） | episode 步数 / 该 episode 墙钟（含模型推理与 env step） |
| `steps_per_second`（totals） | 总评估步数 / 协议总墙钟（含环境构建、reset、视频编码），整体吞吐 |

---

## 5. 与训练预算的严格区分

- 评估步数只统计在 `protocol.totals.eval_steps`，**不进入任何训练计数器**：不加载 optimizer、不调用 `save_checkpoint`、不写 `artifacts/training/seed_0/...` 等训练目录。
- checkpoint 的 `environment_steps` 只作为评估起点记录（`checkpoint.environment_steps`），与本次评估消耗的步数无关。
- 一个 checkpoint 的完整默认协议 = 4 个 episode × 2048 步 = **8192 评估步**（不是 8192 训练步）。
- 评估吞吐（环境 1 进程 + 小模型推理）与训练吞吐（8 workers 并行采样 + PPO 更新）不是同一口径，不可直接比较。
- 评估不改变 PPO/奖励/环境的任何参数；checkpoint 模式使用 checkpoint 内配置快照，避免“评估配置漂移”。

---

## 6. 校验与失败行为（实测）

- `read_checkpoint` 读取（`weights_only=False`，仅信任本地文件）；checkpoint 损坏/截断时报清晰错误并退出码 1。
- 模型构建后 `load_state_dict(..., strict=True)`，结构不匹配时报错；随后逐参数 `torch.isfinite` 审计。
- 缺失 ROM / 初始存档 / SHA1 不匹配时报错；`--rom/--state` 可覆盖。
- 输出目录已有 `results.json`（或视频已存在）且未给 `--overwrite` 时拒绝覆盖。
- 配置含 recurrence/transformer、环境类型非 `PokeRedV2`、动作空间非 `Discrete(7)`、`--no-stochastic` 与 `--random-actions` 同时使用等均报错。
- 每步观测做官方 space + finite 校验；每 episode 从官方初始存档 reset。

---

## 7. 自测记录（真实运行，CPU，与正式训练并发）

2026-10-18，训练运行期间按 Lead 的 GPU 协调规则全部使用 `--device cpu`、小步数。

### 7.1 checkpoint 冒烟（必测项）

```powershell
.venv\Scripts\python.exe scripts\eval_checkpoint.py `
  --checkpoint artifacts\smoke-gpu-resumed\checkpoint_000000064.pt `
  --steps 64 --episodes 1 --device cpu `
  --out artifacts\training\eval_smoke\smoke_ckpt
```

实际结果（`results.json`）：

- checkpoint 元数据：`environment_steps=64`、`next_update=2`、`phase1_format=1`、`agent_label="Untrained Agent — 仅用于运行测试"`、模型 2,086,600 参数、`load_state_dict(strict=True)` 与 finite 校验通过；
- stochastic episode（seed=100000, env_seed=0）：length=64（step_cap），reward=0.0200，coord_count=4，unique_map_count=1，badges=0，levels_sum=6，party_size=1，deaths=0，max_map_progress=0；
- deterministic episode（seed=100001, env_seed=0）：length=64（step_cap），reward=0.0050，coord_count=1，badges=0；
- 总计 128 评估步，协议墙钟 2.87 s，整体吞吐 44.7 steps/s（含 3 次环境实例构建）。

### 7.2 untrained 冒烟（必测项）

```powershell
.venv\Scripts\python.exe scripts\eval_checkpoint.py `
  --untrained --config configs\train_seed0.yaml `
  --steps 128 --episodes 1 --device cpu `
  --out artifacts\training\eval_smoke\untrained_smoke
```

实际结果：

- 随机初始化模型 2,086,600 参数（未加载任何权重）；
- stochastic episode（seed=100000）：length=128（step_cap），reward=0.0650，coord_count=13，badges=0；
- deterministic episode（seed=100001）：length=128（step_cap），reward=0.0050，coord_count=1，动作直方图 `[0,0,0,128,0,0,0]`（恒按 DOWN，符合未训练网络行为），badges=0；
- 总计 256 评估步，协议墙钟 4.36 s，整体吞吐 58.7 steps/s；
- **未训练模型未出现任何徽章/明显通关**（badges=0），符合预期。

### 7.3 其他实现自测

- 视频/截图：`--no-stochastic --steps 16 --video ...` → mp4 可读、16 帧、160×144、fps=2.5；截图 25%=第 4 帧、50%=第 8 帧、100%=第 15 帧（PNG 144×160×3，非空白）。
- 随机动作基线：`--random-actions --steps 32 --episodes 2` → 2 个 episode 各 32 步，reward 0.0050/0.0200，badges=0，无模型构建。
- 确定性复现：`--no-stochastic --episodes 2 --steps 16` 两个 episode 的 actions/coords/rewards/info/obs_events_sum **完全一致**（每 episode 新环境实例）。
- 损坏 checkpoint（截断为 4096 字节）：报 `[eval] ERROR: could not read checkpoint ...: RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory`，退出码 1。
- 覆盖保护：对已有 `results.json` 的目录重跑且不给 `--overwrite` → 清晰报错退出码 1。
- 输出结构：`results.json` 含 protocol/model/checkpoint/episodes/summary/limitations，trace 每步长度与 episode length 一致；`config_snapshot.yaml` 完整。

### 7.4 尚未验证

- `--device cuda` 路径尚未在本轮实测（Lead 要求训练期间禁用 GPU）。脚本的模型/设备创建与推理调用与 `scripts/watch.py` 的已验证 GPU 路径一致（`create_actor_critic_model(..., device)`，`torch.no_grad()` + `.sample()/argmax`），待 Lead 给出 GPU 暂停窗口后，用正式评估前的小 checkpoint 冒烟补验。

---

## 8. 已知限制

1. **没有胜利条件**：官方环境只有步数截断，因此评估只能比较进度指标（exp/event/level/badge…），不能给出“是否通关”。
2. **确定性评估方差恒为 0**：初始存档固定 Game Boy RNG + argmax 策略 → 同一模型、同一初始存档的确定性轨迹唯一；协议因此只跑 1 个 deterministic episode。**方差来自 stochastic 策略评估（3 个种子）**，而不是环境随机性。
3. stochastic 的“随机”指策略采样随机；协议种子只控制 torch/动作采样，不改写游戏内 RNG。
4. 每 episode 重建 PyBoy 实例（约 1–2 s/回合）以保证 episode 独立性；不复用实例是刻意选择，原因见 §2.2。
5. 徽章只有前两个（官方 env 限制）；事件/故事进度来自官方 info，未做额外战斗/剧情校验。
6. 仅支持 feed-forward 模型；recurrence/transformer checkpoint 会直接报错。
7. 不做 mid-episode RAM 保存/恢复；每回合都从官方初始存档开始。
8. `--video` 只录最后一个 episode（默认 deterministic），fps=2.5 仅用于人眼检视，不用于训练/评测计算。
9. 评估吞吐与训练吞吐不可直接比较（见 §5）。
10. 默认协议 4×2048 步在实际 50–70 steps/s 量级下约需 2–3 分钟/checkpoint（含环境构建），具体以实测 `protocol.totals` 为准。

---

## 9. 阶段评估 SOP（等 Lead 指定目录后执行）

1. 只在 Lead 通知的 GPU 暂停窗口内用 `--device cuda` 跑正式评估；训练运行期间如需预跑，使用 `--device cpu`（协议相同，仅速度/设备元数据不同）。
2. 每个里程碑 checkpoint 的完整评估协议（seed_0 正式评估预算：`--steps 4096 --episodes 3 --seed 100000`，即 3 stochastic + 1 deterministic = 16,384 评估步）：

```powershell
.venv\Scripts\python.exe scripts\eval_checkpoint.py `
    --checkpoint <stage_dir>\checkpoint_XXXXXXXXX.pt `
    --out artifacts\training\seed_0\eval\eval_XXXXXXXXX `
    --video artifacts\training\seed_0\eval\videos\eval_XXXXXXXXX.mp4 `
    --device cuda --steps 4096 --episodes 3 --seed 100000
```

3. 同配置基线的评估预算必须一致（同为 `--steps 4096 --episodes 3 --seed 100000`）：

```powershell
.venv\Scripts\python.exe scripts\eval_checkpoint.py --untrained `
    --config configs\train_seed0.yaml --out artifacts\training\seed_0\eval\baseline_untrained `
    --video artifacts\training\seed_0\eval\videos\baseline_untrained.mp4 `
    --device cuda --steps 4096 --episodes 3 --seed 100000
.venv\Scripts\python.exe scripts\eval_checkpoint.py --random-actions `
    --config configs\train_seed0.yaml --out artifacts\training\seed_0\eval\baseline_random_actions `
    --device cuda --steps 4096 --episodes 3 --seed 100000
```

4. 汇报时引用 `results.json` 的 `protocol.totals.eval_steps`、各组 `summary.mean/std`、`checkpoint.environment_steps`，并注明设备与是否在 GPU 窗口运行。
5. 不主动评估未指定的正式训练 checkpoint，不擅自扩大步数预算。
6. 扩展检查（v1-ext）：当 n=3 的统计功效不足、需要更好的最佳模型判断时，可对候选 checkpoint（如 300k / 500k / 1M）与 untrained 基线追加 `--steps 8192 --episodes 5 --seed 100000`（5 stochastic + 1 deterministic，不录视频），输出目录命名 `ext_<name>`。扩展评估的步数单列统计、不属于训练预算，且**不改变** v1 正式协议定义。

---

## 10. 汇总工具（`scripts/summarize_eval.py`）

纯读文件的聚合工具：输入若干 `results.json`（或包含结果目录的父目录，脚本自动发现），输出：

- `artifacts/training/seed_0/eval/eval_summary.json`：结构化汇总（每个 run 的协议元数据、checkpoint 元数据、stochastic/deterministic/random_actions 各组 mean/std/min/max、逐 episode 记录、动作分布与熵、全局警告）；
- `docs/reports/eval_results.md`：Markdown 对比表。**行 = checkpoint / 基线**，列为训练步数、episode reward mean±std、coord_count、maps visited、badges、levels_sum、party_size、deaths、max_map_progress、mt_moon、cerulean、动作分布熵/最大动作占比、eval steps、eval 秒数、eval steps/s；**deterministic 轨迹单独一张表**（含 terminated_by、reward、视频路径），random-actions 基线单独一张表；另含数据来源（results.json 路径与 SHA256、checkpoint SHA256、创建时间）与异常/警告。

```powershell
.venv\Scripts\python.exe scripts\summarize_eval.py artifacts\training\seed_0\eval `
    --exclude _cuda_smoke --exclude _selftest --overwrite
```

报告强制包含以下口径说明（不因数据好看与否而改动）：

- **确定性评估方差为 0**：每个 run 只有 1 个 argmax deterministic episode，初始存档固定 Game Boy RNG 且每 episode 新建环境实例，轨迹唯一；该数值不是统计估计，不参与显著性比较。
- **stochastic 的 mean±std 是样本统计**（v1：n=3，seed = base seed + 0..2；v1-ext：n=5，seed = base seed + 0..4），n 很小，std 只作波动参考，不应据此做显著性结论。
- **eval steps 不属于训练预算**：训练步数取自 `checkpoint.environment_steps`（基线为 0），评估步数只是推理消耗，不推进训练。
- eval 秒数包含环境构建/reset/视频编码，跨设备与负载不可直接横向比较；动作熵为 7 动作上的 Shannon 熵（bit，均匀最大 ≈2.807），最大动作占比为最高频动作比例。
- 若输入之间 `steps_per_episode`/设备/seed 不一致，会在“异常与警告”中如实标出（例如把 64 步的 CUDA 冒烟误混入正式对比表）。
- **v1 与 v1-ext 自动分表**：目录名前缀默认 `ext_`（可用 `--ext-prefix` 覆盖）判定为 v1-ext；报告包含 v1 的 stochastic/deterministic/random-actions 三张表，以及独立的“Extended check（v1-ext）”小节（stochastic 与 deterministic 各一张表）。两种协议的 episode 预算不同（4096×3 vs 8192×5），绝对值不可跨协议直接比较，扩展评估步数单列统计、不并入 v1 表。

自测（用现有 CPU 冒烟结果）：目录自动发现、`--exclude`、覆盖保护、混合协议告警均正常；汇总 JSON/Markdown 结构完整，数值可直接追溯到各 `results.json`。真实评估数据生成后再用上述命令产出正式 `eval_summary.json` / `eval_results.md`。
