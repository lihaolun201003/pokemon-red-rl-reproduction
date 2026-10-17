# 训练监督工具 `scripts/supervisor.py`（用法与退出码）

- 开发：training monitor engineer（teammate `monitor-engineer`）
- 适用：Pokémon Red PPO 正式训练实验 `artifacts/training/seed_0/`
- 定位：**轻量、只读**的外部监督脚本。用一次阻塞调用（DSH job）等待阶段结束或异常，避免高频 LLM 轮询。
  - 不启动训练、不停止/不 kill 训练进程、不修改任何 run 目录；
  - 只读 `training_state.json`、`metrics.jsonl`、`result.json`、checkpoint 文件的元数据（checkpoint 内容只有 `verify-checkpoint` 会读）；
  - 只写自己的输出：`artifacts/training/seed_0/monitor/status.jsonl`。
- 依赖：Python 标准库 + 项目已安装的 `psutil`（`status`/`watch` 不 import torch）；`verify-checkpoint` 需要 torch/numpy/gymnasium/neroRL。
- 运行方式（工作目录 = 项目根目录）：

```powershell
.venv\Scripts\python.exe scripts\supervisor.py status
.venv\Scripts\python.exe scripts\supervisor.py status --json
.venv\Scripts\python.exe scripts\supervisor.py watch --timeout 3600
```

> 训练期间执行 `status/watch/prune --dry-run/launch --dry-run` 均为只读，对训练零干扰。
> `verify-checkpoint` 会 import torch 并在 CPU 上重建模型（约 300 MB RSS 的短暂内存峰值），
> 建议在阶段边界或训练结束后执行；不要在训练运行中与训练抢内存。

## 1. 子命令

### `status [--state PATH] [--json] [--out DIR]`

一次性快照：读 state + 当前 stage 的 `metrics.jsonl` 最后一行 + checkpoint 列表 + `result.json`，
采集系统指标并给出 `alerts`，追加一条快照到 monitor/status.jsonl。

- 默认 `--state artifacts/training/seed_0/training_state.json`；找不到 state 时自动探测
  `artifacts/training/seed_0/stage_*/` 中最新的 run 目录（`missing` 里会标记 `training_state`），不崩溃。
- `--json` 输出完整机器可读快照（schema `supervisor_status/1`）。
- `--out` 可覆盖 stage 目录（用于检查归档目录/旧 run，例如 `_failed/stage_a_10k_attempt1`）。
- 实测采集耗时约 **1.1 s**（含 `psutil.cpu_percent(interval=1)` 的 1 s 采样），满足 <5 s 要求。

人类可读输出包含：stage/out_dir、进程（实际 PID、state PID、存活与 cmdline 校验）、
步数/进度百分比/update/最近更新时间/近期吞吐、累计步数、stage ETA 与 1M ETA、
checkpoint 数量与最新文件、result.json 状态、CPU/RAM/磁盘/GPU 指标、loss/熵/KL/clip/lr、
game/* 游戏进度、actions/share_* 动作分布、alerts/missing。

关键实现细节：

- **PID 复用防护**：不只信 state 里的 PID。扫描所有 `python*.exe` 进程的 cmdline，
  只把 `--out` 参数指向当前 run 目录的进程当作训练进程（不能只按目录名子串匹配：后续阶段的
  `--resume .../stage_c_300k/checkpoint_...` 也包含上一阶段目录名，会造成误匹配）。
  state PID 过期时会自动找到真实训练进程并给出 `stale_state_pid` 警告；PID 被复用为无关进程时
  cmdline 不匹配，不会被误认为训练进程。本机实测训练进程是一个两进程链：
  `.venv\Scripts\python.exe`（Start-Process 目标）+ 其启动的
  `Python310\python.exe`（实际父进程），两者 cmdline 相同，supervisor 都识别为同一 run。
- `--out DIR` 用于检查非当前 stage（归档目录、已完成阶段）：此时 target/start_steps/PID/status
  全部从 `DIR/environment.json`、`DIR/metrics.jsonl`、`DIR/result.json` 读取，不与 state 中
  当前 stage 的字段混用（`stage.configured_status="cli-override"`），便于在 D 运行期间回看 C 的完成状态。
- **resume 阶段的 checkpoint 期望**：checkpoint 写入发生在全局 update 的 `checkpoint_interval` 倍数处
  （即环境步数 unit=batch×interval 的倍数）。恢复阶段不会重写起始 checkpoint，因此期望值从
  `start_steps` 之后第一个 unit 倍数算起（`start_steps` 取自 state、environment.json 的
  `resumed_from` 文件名或首个 metrics 行），避免把上一阶段的 checkpoint 误报为"缺失"。
- **吞吐**：最近 5 条 metrics 的 `time/steps_per_second` 中位数（不足时退化为 tail 窗口内
  Δ步数/Δ时间）；ETA = 剩余步数 / 吞吐。
- **累计步数**：取 state.total_training_env_steps、当前 stage metrics 的 `progress/environment_steps`、
  最新 checkpoint 步数、result.json 步数的最大值（resume 后 metrics 记录的是累计绝对步数）。
- **GPU**：`nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw
  --format=csv,noheader,nounits`；失败时记录 `gpu_error` 字符串并自动重试 1 次，不影响其他字段。
- **RAM/swap**：输出 `ram.available_bytes/percent` 与 `swap.used/free/percent`（本机 swap 约 27–29 GiB，
  训练时典型已用约 2 GiB/7%）；available RAM <1.5 GiB 触发 critical `low_ram`。
- 异常检测项与阈值见第 4 节。

### `watch [--state PATH] [--interval 60] [--timeout 7200] [--target-steps N] [--stall-seconds 900] [--json]`

循环执行 status 逻辑（每轮追加快照），阻塞等待下列终止条件之一：

| 条件 | 行为 | 退出码 |
|---|---|---|
| (a) 当前 stage 的 `result.json` 存在且 `environment_steps >= target` | 打印最终 JSON，`outcome="completed"` | **0** |
| (b) 严重异常（进程退出但无 result.json；metrics NaN/Inf；参数不更新；checkpoint 损坏；资源持续告警；长时间无进度） | 打印诊断 JSON，`outcome="anomaly"` | **1** |
| (c) 达到 `--timeout` 秒仍未终止 | 打印诊断 JSON，`outcome="timeout"` | **2** |
| (d) `result.json` 存在但 `environment_steps < target`（提前干净停止，如 stop_file） | 打印 JSON，`outcome="stopped_before_target"` | **4** |

- target 缺省从 `state.stage.target_steps` 读取（每轮重读 state，Lead 在阶段边界更新 state 后自动切换到新 stage）；
  可用 `--target-steps` 覆盖。
- 立即触发退出码 1 的 critical：`process_exited_without_result`、`metrics_nonfinite`、
  `parameter_delta_nonpositive`、`checkpoint_too_small`。
- 连续两轮出现才触发退出码 1 的 critical：`stale_metrics`、`checkpoint_expected_missing`、
  `state_checkpoint_missing`、`low_disk`、`gpu_hot`、`low_ram`（避免瞬时误报）。
- 无进度超时：进程存活但最近一次 update 距今 > `--stall-seconds`（默认 900 s，且 stage 已运行超过该时长）
  → 退出码 1。
- `--json` 压制每轮人类可读行，只在最后打印 JSON；无论是否 `--json`，最终都会打印一份 JSON
  （含 `outcome`、`exit_code`、`reason`、`rounds`、`elapsed_seconds`、`final_snapshot`）。
- 每轮 `[watch] round=... t=+... steps=.../... last_update=... throughput=... alerts=...` 行输出到 stdout 并 flush，
  便于 DSH job 日志观察；job 被取消（Ctrl+C/kill）时脚本本身不写任何训练目录。
- 训练进行的轮次之间用 `time.sleep` 间隔 `--interval`；可被 DSH 后台 job 取消。

### `verify-checkpoint --path PT [--out DIR | --config YAML]`

只读校验一个 checkpoint（不写入、不修改文件；脚本用 mtime/size 前后对比记录 `readonly`）：

1. 文件存在、大小 ≥ 1 MiB（否则 warning）；
2. `torch.load`（经 `reproduction.runtime.read_checkpoint`，与训练/恢复完全同语义）；
3. `phase1_format == 1`、`next_update > 0`、`environment_steps > 0`；
4. `model`/`optimizer` state 非空，所有模型参数与优化器浮点张量 finite；
5. `observation_space` / `action_space` 存在且类型正确；
6. 用 checkpoint 自带的 `observation_space`/`action_space` + checkpoint `configs["model"]`
   在 CPU 上重建官方 actor-critic 模型，`load_state_dict(..., strict=True)` 严格加载验证；
7. 若给 `--out DIR`：比较 checkpoint `configs` 与 `DIR/config.yaml` 的
   `environment/sampler/trainer/model` 四个 section；若给 `--config`：与指定 YAML 比较。
   - 由于 `PokeRedV2Wrapper` 在 config.yaml 写出后会向 `environment.reset_params` 注入运行时默认键
     `monitor_more_stats=False`，checkpoint configs 会比 config.yaml 多这一个键。
     该键属于已知 benign 差异：记录到 `benign_differences`，不判为不可续训；
     其余任何差异记为 substantive，`resume_compatible=false`（`restore_checkpoint` 会拒绝这样的恢复）。

输出 JSON（schema `supervisor_verify/1`），字段：`checks[]`、`warnings[]`、`errors[]`、
`model_rebuild`、`config`（含每 section 差异明细）、`readonly`。

退出码：**0** 通过；**1** 校验失败或 config 实质不匹配；**2** 用法/IO 错误（文件不存在等）。

### `prune-checkpoints --out DIR [--keep-last 10] [--milestones 10000,100000,300000,1000000] [--dry-run] [--json] [--state PATH]`

列出 `DIR/checkpoint_*.pt`（只匹配非递归的正式 checkpoint 文件名），保留：

- 最近 N 个（按步数）；
- 步数命中里程碑集合的；
- `training_state.json` 中 `latest_checkpoint` / `latest_valid_checkpoint` 指向的文件（防误移）；
- 60 秒内修改过的文件（防与训练保存竞争）。

其余 **移动**（`shutil.move`，不删除）到 `DIR/_archive/`。安全措施：

- `--out`、每个源文件、`_archive` 目标都做 `Path.resolve()` 校验，必须位于项目目录内、且目标在 `--out` 内；
- 目标已存在则跳过（`already_archived`），不覆盖；
- `--dry-run` 只打印计划，不做任何写入（不创建 `_archive/`）。

输出：人类可读的计划/结果行；`--json` 额外输出 JSON 摘要（schema `supervisor_prune/1`，含
`kept/planned_moves/moved/skipped/milestones_matched/milestones_unmatched/errors`）。
退出码：0 正常（含 dry-run）；1 有文件移动错误。

### `stop-request [--state PATH]`

只 `touch` state 中 `stop_file`（缺省 `artifacts/training/seed_0/STOP`），打印说明。
`scripts/train.py` 在**当前 update 结束、原子保存并读回校验 checkpoint 后**干净退出，
`result.json.stop_reason = "stop_file"`；**不 kill 进程**，最多多跑一个 update（本配置 2000 步 ≈ 4 s）。
state 无 `stop_file` 字段时回退到实验目录下 `STOP` 并给出 warning。退出码 0/2。

### `launch --dry-run`

只打印将要执行的训练命令与 `Start-Process` PowerShell 片段（含 stdout/stderr 日志建议路径），
**不会启动任何进程**；真启动始终由 Lead 用 `Start-Process` 完成，避免与 Lead 争控制权。
未带 `--dry-run` 时同样只打印、不启动。

## 2. 状态文件与快照 schema

### `artifacts/training/seed_0/training_state.json`（Lead 维护，supervisor 只读）

当前实际 schema（supervisor 全部容错，缺字段只记入 `missing`/警告，不崩溃）：

```jsonc
{
  "experiment": "seed_0", "config": "configs/train_seed0.yaml", "seed": 0, "device": "cuda",
  "batch_per_update": 2000, "n_workers": 8, "worker_steps": 250,
  "checkpoint_interval_updates": 5, "stop_file": "artifacts/training/seed_0/STOP",
  "status": "running",
  "stage": {"name": "stage_b_100k", "target_steps": 100000, "start_steps": 10000,
            "out_dir": "artifacts/training/seed_0/stage_b_100k", "resume_from": "…checkpoint_000010000.pt",
            "pid": 47120, "status": "running", "started_at": "2026-10-18T00:24:16+08:00",
            "stdout_log": "…", "stderr_log": "…"},
  "stages_completed": [{"name": "stage_a_10k", "target_steps": 10000, "end_steps": 10000, "…": "…"}],
  "latest_checkpoint": {"path": "…", "steps": 10000, "update": 5, "bytes": 25331916},
  "latest_valid_checkpoint": {"path": "…", "steps": 10000, "update": 5, "bytes": 25331916},
  "total_training_env_steps": 10000, "cumulative_training_seconds": 18.32,
  "budget": {"max_training_env_steps": 1000000, "max_wall_hours": 12,
             "budget_started_at": "…", "deadline_at": "…"},
  "notes": ["…"]
}
```

兼容别名：`budget.max_steps`→`max_training_env_steps`、`budget.max_hours`→`max_wall_hours`；
`latest_checkpoint` 也可以是纯路径字符串。

### `artifacts/training/seed_0/monitor/status.jsonl`（supervisor 追加）

每行一条 JSON 快照，`kind` 为 `status` / `watch` / `stop_request`。快照结构（`supervisor_status/1`）：

```
paths{root,state,experiment_dir,stage_out,monitor_dir,status_log}
state{path,exists,error,status,stage_status,latest_checkpoint,latest_valid_checkpoint,
      total_training_env_steps,cumulative_training_seconds,budget,stop_file,stop_file_exists}
stage{name,out_dir,target_steps,configured_status,pid,started_at,stage_age_seconds,
      resume_from,source,auto_detected,newer_stage_detected}
process{state_pid,state_pid_alive,state_pid_cmdline_match,live_matching_pids,matched_cmdlines,
        running,pid_mismatch}
progress{stage_env_steps,total_training_env_steps,update,target_steps,target_remaining_steps,
         target_reached,last_update_unix,last_update_age_seconds,throughput_steps_per_second,
         eta_stage_seconds,eta_project_1m_seconds,eta_budget_steps_seconds,
         budget_wall_remaining_seconds,batch_per_update,checkpoint_interval_updates}
metrics{path,exists,size_bytes,rows_seen,last_update_age_seconds,nonfinite_keys,latest{…}}
checkpoints{count,total_size_bytes,latest{name,steps,size_bytes,size_mb,path},recent[],error}
result{path,exists,environment_steps,updates_completed,stop_reason,checkpoint}
system{cpu_percent,ram{total,available,used,percent},disk{free,percent},gpu{utilization_percent,
       memory_used_mib,memory_free_mib,memory_total_mib,temperature_c,power_w},gpu_error}
alerts[{code,severity,message,…}], has_critical, missing[], timing{collect_seconds}
```

监控目录里另有本工具生成的验收/后处理产物（不属训练产物、不参与训练）：
`stage_c_completion.status.json`、`stage_d_completion.status.json`（阶段自然完成快照）、
`final_status.json`（state 更新为 training_complete 后的总快照）、
`verify_{10000,100000,300000,500000,1000000}.json` + `verify_summary.json`、
`verify_smoke_gpu_resumed.json`、`verify_smoke_vs_seed0config.json`、
`prune_{dryrun,actual}_{stage}.log` + `prune_dryrun.json` + `prune_actual.json`，
以及 `selftest/`（stop-request 隔离自测，未触及正式 STOP）。

## 3. alerts 一览

| code | severity | 触发条件 |
|---|---|---|
| `metrics_nonfinite` | critical | 最近一条 metrics 存在 NaN/Inf |
| `parameter_delta_nonpositive` | critical | `validation/parameter_max_delta <= 0` |
| `stale_metrics` | critical | 进程存活但最近 update 距今 >600 s |
| `process_exited_without_result` | critical | 无匹配训练进程且无 `result.json`（且 stage 已开始 >60 s） |
| `checkpoint_too_small` | critical | 最新 checkpoint / state 指向文件 <1 MiB |
| `state_checkpoint_missing` | critical | state 指向的 checkpoint 文件不存在 |
| `checkpoint_expected_missing` | critical | 按 `batch × checkpoint_interval` 推算应已保存的 checkpoint 不存在 |
| `low_disk` | critical | 项目盘剩余 <10 GiB |
| `gpu_hot` | critical | GPU 温度 >85 °C |
| `low_ram` | critical | 可用 RAM <1.5 GiB |
| `low_gpu_memory` | warning | GPU 空闲显存 <1 GiB |
| `no_new_steps` | warning | 连续两次 supervisor 快照步数不变且间隔 ≥120 s |
| `stale_state_pid` | warning | state PID 与实际匹配进程不一致（PID 复用/launcher 差异） |
| `multiple_train_processes` | warning | 同一 run 目录匹配到 >2 个训练 python 进程 |
| `state_stale` | warning | 进程已结束且有 result.json，但 state 仍标记 running |
| `stopped_before_target` | warning | result.json 的步数小于目标（如 stop_file 提前停止） |
| `metrics_missing` | warning | 运行 >120 s 仍无 metrics.jsonl |
| `gpu_query_failed` | warning | nvidia-smi 调用失败（已重试 1 次，记录 `gpu_error`） |
| `newer_stage_detected` | info | 磁盘上存在比 state 更新的 run 目录（提示 Lead 更新 state） |

## 4. 退出码总表

| 命令 | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| `status` | 快照已打印（即使有 alerts） | — | 用法/路径安全错误 | — |
| `watch` | 到达目标并产出 result.json | 严重异常 | 超时 | 有 result.json 但未达目标 |
| `verify-checkpoint` | 校验通过 | 校验失败/config 实质不匹配 | 用法/IO 错误 | — |
| `prune-checkpoints` | 正常（dry-run 恒 0） | 有移动错误 | 用法/路径错误 | — |
| `stop-request` | stop 文件已创建/已存在 | — | state/IO 错误 | — |
| `launch --dry-run` | 命令已打印 | — | 用法/路径错误 | — |

## 5. 实测记录（2026-10-18，真实游戏训练）

| 测试 | 结果 |
|---|---|
| `status`（Stage B @88k 步运行中） | 采集 1.09 s；进程 running=True、PID 校验通过；吞吐 409 steps/s；ETA stage 29.3 s / 1M 37m09s；checkpoint 7 个；RAM 2.4 GiB free、GPU 57 °C；alerts=none |
| `status`（Stage D @314k 步运行中） | 采集 1.31 s；running=True pids=[14232,28628]、state_pid=28628 匹配；吞吐 624 steps/s；ETA stage/1M = 18m19s；checkpoint 310k；alerts=none；RAM 2.3 GiB free、swap 2.0/27.1 GiB、GPU 59 °C |
| `status --json` | schema `supervisor_status/1` 完整输出，字段可机器解析 |
| `status --out stage_c_300k`（D 运行期间回看 C） | 正确识别 C 的 result.json/20 个 checkpoint（含 300k 里程碑）、`running=False`、target=300000、start_steps=100000；无 critical；快照存 `monitor/stage_c_completion.status.json` |
| `status --out …_failed/stage_a_10k_attempt1` | 正确报 `[critical] process_exited_without_result`（该目录是 00:22 nvcuda64.dll 原生崩溃后的归档，无 result.json） |
| `watch --timeout 120 --interval 5`（Stage B 90k→100k） | 25.6 s 后退出码 **0**，`outcome="completed"`，`reason="result.json present and target reached"` |
| `watch --timeout 120 --interval 10`（Stage C 运行中） | 121.2 s 后退出码 **2**，`outcome="timeout"`（Stage C 需 ~7 min > 120 s），期间无误报 critical |
| `watch --out …_failed/…` | 1.2 s 后退出码 **1**，`reason="immediate critical: process_exited_without_result"` |
| `watch --out stage_a_10k --target-steps 200000 --timeout 30` | 1.3 s 后退出码 **4**，`outcome="stopped_before_target"` |
| `status` 三次计时 | 1.38 s / 1.37 s / 1.39 s（均 <5 s 要求） |
| 路径安全 | `status --out C:\Windows` → 退出码 2（"outside the project directory"），不读取项目外文件 |
| 缺失 state 降级 | `--state` 指向不存在文件 → 退出码 0，`missing=[training_state, state.stage]`，不崩溃 |
| `stop-request`（隔离自测 state，stop_file 在 monitor/selftest 内） | 退出码 0，创建 stop file 并打印语义说明（未触碰正式训练 STOP） |
| `prune-checkpoints --out stage_b_100k --keep-last 1 --milestones 10000 --dry-run` | 只读计划：8 个 checkpoint 中计划移动 5 个旧的到 `_archive`，保留最新，跳过 60 s 内新文件；无实际移动 |
| `launch --dry-run` | 正确打印命令与 Start-Process 片段，未启动进程 |

环境异常（非工具问题）记录：Stage A 首次启动在 `nvcuda64.dll`（驱动 32.0.15.9282）原生崩溃，
Windows Application Error EventID 1000，exception `0xc0000409`（BEX64）；supervisor 对该场景的
`process_exited_without_result` 检测已用归档目录验证通过。

## 6. verify-checkpoint 实测（训练结束、CPU-only 窗口）

对 5 个里程碑 checkpoint 执行 `verify-checkpoint --path <cp> --out <run_dir>`（训练已停止、无 CUDA context，
未使用 GPU）。输出 JSON 已存 `artifacts/training/seed_0/monitor/verify_<step>.json`，汇总在
`monitor/verify_summary.json`。

| checkpoint | phase1_format | next_update | environment_steps | model finite | optimizer 非空 | strict 重建加载 | config 严格匹配 | resume_compatible | 退出码 |
|---|---|---|---|---|---|---|---|---|---|
| stage_a_10k/checkpoint_000010000.pt | 1 | 5 | 10,000 | 是（0 个非有限参数） | 是 | 是（2,086,600 参数） | 否（仅 benign） | **是** | 0 |
| stage_b_100k/checkpoint_000100000.pt | 1 | 50 | 100,000 | 是 | 是 | 是 | 否（仅 benign） | **是** | 0 |
| stage_c_300k/checkpoint_000300000.pt | 1 | 150 | 300,000 | 是 | 是 | 是 | 否（仅 benign） | **是** | 0 |
| stage_d_1m/checkpoint_000500000.pt | 1 | 250 | 500,000 | 是 | 是 | 是 | 否（仅 benign） | **是** | 0 |
| stage_d_1m/checkpoint_001000000.pt | 1 | 500 | 1,000,000 | 是 | 是 | 是 | 否（仅 benign） | **是** | 0 |

- 5 个文件均 24.16 MiB；`readonly=true`（校验前后 mtime/size 不变，未修改 checkpoint）。
- 唯一 config 差异是 checkpoint 比 `config.yaml` 多出运行时注入的
  `environment/reset_params/monitor_more_stats=False`，属 benign 差异，不影响 `restore_checkpoint`
  （恢复时该键已由 env wrapper 注入到内存 config，两边的 section 实际相等）。
- Smoke checkpoint 对照测试（任务要求）：
  - `verify-checkpoint --path artifacts/smoke-gpu-resumed/checkpoint_000000064.pt --out artifacts/smoke-gpu-resumed`
    → 退出码 0；结构/模型重建全部通过；environment_steps=64、next_update=2；
    strict mismatch 仅 benign（`monitor_more_stats`），`resume_compatible=true`；
    结果存 `monitor/verify_smoke_gpu_resumed.json`。
  - 同一 checkpoint 对 `configs/train_seed0.yaml` 比较（`--config configs/train_seed0.yaml`）
    → 退出码 1，实质性不匹配（smoke 用 2×16、epochs=1、checkpoint_interval=1，且 environment
    name/init-state 为 smoke 绝对路径等），`resume_compatible=false`；如实报告不匹配。
    结果存 `monitor/verify_smoke_vs_seed0config.json`。

## 7. prune-checkpoints 实际执行结果（训练结束、评估窗口）

评估目标为 10k/100k/300k/**500k**/1M，因此 milestone 列表采用
`10000,100000,300000,500000,1000000`；`--keep-last 10`；先 `--dry-run` 再实际移动（只 `move` 到
`<out>/_archive/`，不删除）。日志：`monitor/prune_{dryrun,actual}_{stage}.log`，汇总
`monitor/prune_dryrun.json`、`monitor/prune_actual.json`。

| 目录 | 原有 checkpoint | 实际移动 | 保留 | `_archive` | 说明 |
|---|---|---|---|---|---|
| stage_a_10k | 1（10k） | 0 | 1 | 无 | 按要求不动 |
| stage_b_100k | 9（20k…100k） | 0 | 9 | 无 | keep-last 10 已覆盖全部；100k milestone 保留 |
| stage_c_300k | 20（110k…300k） | **10**（110k…200k） | 10（210k…300k） | 10 个（0.24 GiB） | 300k milestone 保留 |
| stage_d_1m | 70（310k…1M） | **59**（310k…490k、510k…900k） | 11（500k milestone + 910k…1M） | 59 个（1.39 GiB） | 500k、1M milestone 均保留 |

- 实际执行后复核：5 个评估目标路径全部存在且大小一致（10k/100k/300k/500k/1M），
  全部 milestone 文件**未进入 `_archive`**；移动过程无错误（`errors: []`）。
- 注意 `stage_d_1m` 目录内保留集合为 {500000} + 最近 10 个（910k…1M），因此 500k 与 1M 两个
  milestone 都在原目录可用。
- 评估进程只读取上述 milestone 路径（非 milestone 旧 checkpoint 才会被移动）。

## 8. 限制与注意事项

- `watch` 的退出条件针对**当前 state 指向的 stage**；阶段边界 Lead 更新 state 后的几秒内 watch 会自动读取新
  stage，不需要重启 watch。若 state 更新晚于新 stage 启动，`newer_stage_detected` 会提示。
- `status/watch` 不做 CUDA 初始化、不占用 GPU；`verify-checkpoint` 会短暂占用 CPU 内存（~300 MB RSS 峰值），
  训练运行期间不要执行。
- PID 记录可能指向 Start-Process 的 venv launcher 或实际父进程（本机实测为 2 个进程的 launcher+parent 链），
  supervisor 以 cmdline+out_dir 全进程扫描为准，state PID 仅作辅助；不会 kill 任何进程。
- 监控目录只保证在 `artifacts/training/seed_0/monitor/`（及自测文件）；不改动任何 run 目录。
- 若 metrics.jsonl 出现写了一半的行（理论上 flush 保证完整行），读取端跳过无法解析的行。
- 本工具不改变 PPO 数学、奖励、模型、观测、动作定义，也不改变训练步数预算。
