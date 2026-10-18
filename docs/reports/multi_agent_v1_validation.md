# Pokémon Red Multi-Agent v1 — 验证报告

状态：实现完成；G0–G7 全部跑通（汇总见 `artifacts/multi_agent_v1/acceptance_summary.json`：
8/8 场景 PASS；G7 的真实运行数据与多次运行差异见
`docs/reports/multi_agent_v1_results.md`）。本文件只列测试与验收方法/证据路径，不虚构数据。

所有数据均来自本机真实执行：
- Windows 10 / Python 3.10.11 / PyBoy（`local/PokemonRed.gb` 合法 ROM）；
- DeepSeek API（`deepseek-flash`，真实 key 来自本地 `.env`，未提交）；
- 验收存档由合法游戏操作生成（`scripts/multi_agent_make_states.py`；
  `work/multi_agent_v1/states/*.state`，gitignored）。

---

## 1. 测试统计（实际执行结果）

命令与结果（2026-10-18 本机执行）：

```powershell
# 原有基线（Navigation v2 + 单 Agent + Phase1）
.\.venv\Scripts\python.exe -m pytest tests\navigation -q      # 319 passed
.\.venv\Scripts\python.exe -m pytest tests\llm_agent -q       # 364 passed
.\.venv\Scripts\python.exe -m pytest tests\phase1 -q          #  11 passed

# 新增多 Agent 子系统
.\.venv\Scripts\python.exe -m pytest tests\multi_agent -q     # 341 passed
.\.venv\Scripts\python.exe -m pytest tests\story -q           #  88 passed
.\.venv\Scripts\python.exe -m pytest tests\pokemon -q         # 136 passed
```

| 套件 | 数量 | 说明 |
|---|---|---|
| tests/navigation | 319 | Navigation v2 原样保留（未修改任何 nav 源码/测试） |
| tests/llm_agent | 364 | 单 Agent 基线 357 + 新增 7 项多 Agent 边界测试 |
| tests/phase1 | 11 | 数据/环境基线 |
| tests/multi_agent | 341 | 协议/总线/仲裁/任务/调度/记忆/监控/恢复/checkpoint/director/session/model-router |
| tests/story | 88 | 对话判定/有界交互探针/逐回合工具调用/证据与事实标注/API 重试/记忆 |
| tests/pokemon | 136 | 战斗检测/菜单布局(真实帧校准)/intro A 推进/菜单驱动/决策路径/经验/记忆 |
| **合计** | **1259** | 基线 687 全部保留，其余为新增；无删除/跳过测试 |

* 原 687 基线 = 319 (navigation) + 357 (llm_agent) + 11 (phase1)。新增后 llm_agent 因
  追加 7 项边界测试变为 364，因此“原测试集合”实际 694 项，全部通过。
* `pytest tests` 全量收集在本机会因 `tests/envs` 缺 `memory_gym`/`minigrid` 依赖报 4 个
  collection error（**改动前就存在**，与本次无关；见 known_limitations 第 6.4 条）。
* 全部新测试离线、确定性、无 ROM/网络/API key，秒级完成。

### 1.1 导入边界与“盲探索”约束测试（强制项）

`tests/llm_agent/test_import_boundaries.py` 已扩展并全部通过（24 项），强制：

* 决策回路 + Navigation v2 + multi_agent + story + pokemon **不得** import
  PyBoy 或验证用 RAM 探针；
* 不得出现 RAM 地址字面量（`0x` + 4 hex）；
* Agent 提示词/代码不得出现地点/道馆/徽章/精灵名等攻略知识；
* Story/Pokémon 提示词必须声明“第一次玩/盲探索/只依据可见屏幕”；
* 多 Agent 包的所有公开出口可导入。

### 1.2 新增测试文件清单

```text
tests/multi_agent/
  test_contracts.py           # 协议消息/任务/知识/分支/幂等/边界
  test_event_bus.py           # 持久化/去重/优先级/重放/损坏容忍
  test_control_arbiter.py     # 唯一 owner/lease/token/抢占/GuardedEmulator/无死锁
  test_task_manager.py        # 生命周期/去重/依赖环/尝试上限/快照/分支失效
  test_scheduler.py           # 确定性排序/可解释 score/aging/轮换/不饿死
  test_shared_memory.py       # 证据/置信度/验证状态/历史分支/prompt 预算
  test_progress_monitor.py    # 各模块真实增益判定/阈值/训练宽限/硬信号
  test_recovery.py            # L1/L2/L3/回档预算/冷却/hook 容错/不无限循环
  test_checkpoint.py          # manifest 原子写/sha256/校验/篡改/无碰撞
  test_director.py            # 接管/中断恢复/去重/复盘/上限/snapshot 单调计数器
  test_session.py             # 子系统装配/结果落盘/checkpoint 往返/G1 探针
  test_model_router.py        # 每 Agent 模型/预算/记账/失败/冷却/不回退
tests/story/                  # prompts/tools/memory/agent（对话循环与证据）
tests/pokemon/                # detection/menu driver/agent decisions/memory/tools/prompts
```

---

## 2. 真实验收方法（G0–G7）

统一入口：

```powershell
.\.venv\Scripts\python.exe scripts\multi_agent_acceptance.py --scenario g0
# ... 或 --scenario g2 --scenario g3 ... / --all
```

每个场景把原始证据写入 `artifacts/multi_agent_v1/<G>/`：
`report.json`、`evidence/*.json`、`screenshots/`、以及该次运行的 `runs/<run_id>/`
（audit JSONL、导航地图、agent 截图、tasks.json、shared_memory、checkpoints、budget）。
`artifacts/` 与 `work/` 均已 `.gitignore`，不随仓库发布。

### G0 原有回归

* 运行 6 个 pytest 套件 + Navigation v2 三个真实游戏验收脚本
  （`nav_v2_g1_localization` / `nav_v2_g2_explore_room` / `nav_v2_g3_reentry`）。
* 结果：全部 returncode=0（详见 `artifacts/multi_agent_v1/g0/evidence/*.json`）。
* PPO/训练代码、单 Agent 运行入口未改动。

### G1 控制权互斥（真实 PyBoy）

* `ControlArbiter` + 三个 `GuardedEmulator`；真实模拟器按键与帧计数。
* 验证项：并发请求始终单一 owner（历史重放最大并发 lease=1）；确定性抢占场景下
  两个竞争 agent 被拒绝且被拒操作不推进帧；同 owner 续约可继续；抢占后旧 token
  无法操作；异常 agent 崩溃后 `release_all` 恢复、无死锁；交接按键数精确
  （3 次导航按键=72 帧、2 次 Pokémon 按键=48 帧，action_frames=24/次）。
* 证据：`artifacts/multi_agent_v1/g1/report.json`（attempts/denied/帧差/stale lease/
  crash 安全/history）。

### G2 Story 接管真实对话（真实 ROM + 真实 DeepSeek）

* 合法存档：`work/multi_agent_v1/states/g2_dialog_ready.state`（真实 OAK 对话可被打开）。
* 流程：Director 观察 → DIALOG 上下文 → 授予 STORY lease → StoryAgent 真实截图送
  DeepSeek 读取真实对话 → 逐回合真实按键推进/结束 → 记录知识（VERIFIED 带证据截图）
  → 释放控制 → Navigation 继续执行探索任务 → Story 交互探针任务再次接管对话。
* 证据：`g2/report.json`（director_steps、story tasks、knowledge、截图、API 记账、
  owner_switches、control 历史）。

### G3 Pokémon 接管真实战斗（真实 ROM + 真实 DeepSeek）

* 合法存档：`work/multi_agent_v1/states/g3_battle_ready.state`（Route 1 草丛，
  下一步即可能触发野生战斗）。
* 流程：Navigation 探索 → BATTLE 事件 → 暂停导航任务 → 授予 POKEMON lease →
  PokémonAgent 等 intro 文本（有界 A 推进）→ 真实菜单截图送 DeepSeek 决策 →
  像素校验的菜单驱动按真实键 → 观察结果 → 战斗结束 → 释放控制 →
  重新观察 + Navigation 重定位 → 恢复原导航任务。
* 证据：`g3/report.json`（intro_wait、逐回合 actions/presses/verified、lessons、
  knowledge、navigation task 恢复、API 记账）。

### G4 跨 Agent 任务协作（真实 ROM + 真实总线/任务/导航）

* Story 通过 EventBus 发 `NAVIGATION_REQUEST` → Director 检索并创建导航任务
  （父任务=Story 任务）→ Navigation 真实执行 → 任务完成 → `TASK_COMPLETED` 事件 →
  父任务自动恢复 READY → Story 基于结果继续提出后续任务。
* 同时验证：同 message_id 重放不重复建任务（持久化 processed_messages 幂等）、
  同内容不同 id 被总线去重拒绝。
* 说明：该场景 LLM Agent 使用离线 fake（只验证协作基础设施；不冒充模型能力）。
* 证据：`g4/report.json`。

### G5 Checkpoint 回档与经验保留（真实 ROM + 真实 checkpoint 代码）

* 学习知识 → 保存完整 branch manifest checkpoint → 继续行动/学习 → 触发回档 →
  恢复 PyBoy 状态 + 任务/director/agent 上下文快照 + 重新观察/重定位 →
  新 branch 广播到 tasks/shared/bus/recovery。
* 验证：manifest sha256/verify 通过；恢复后 emulator 确实 load 了 state；
  回档产生新分支（旧分支 ROLLED_BACK）；旧分支知识标 HISTORICAL（不删除）；
  回档后新学的经验仍在（历史标注）；知识版本单调；**API 花费不回退**；
  导航重定位成功。
* 说明：LLM Agent 使用离线 fake（事件/记账真实）。
* 证据：`g5/report.json`。

### G6 停滞检测与自主恢复（真实 ROM + 真实 monitor/recovery/checkpoint）

* 用极小阈值构造可重复停滞（无新格 + 大量无效动作）→
  `ProgressMonitor` 产生 LOCAL_STAGNATION/NO_PROGRESS → `RecoveryManager`
  L1（局部 hint）→ L2（全局复盘、限频）→ L3（回档，受 per-branch/total/cooldown 限制）
  → 达到硬上限后**安全暂停**（保存现场 + checkpoint + LIMIT_REACHED 事件）。
* 证据：`g6/report.json`（stagnation_events、recovery stats/attempts_by_level、
  rollback_history、safe_pause、分支数）。

### G7 真实 DeepSeek 多 Agent 集成运行

* 同一进程、同一真实游戏实例、真实 API；Director 自动调度三个 Agent，
  记录每个交接、任务、事件、截图、token/费用与最终游戏进展。
* 详见 `multi_agent_v1_results.md` 的 G7 章节（含实际费用与进展）。

---

## 3. 验收证据目录

```text
artifacts/multi_agent_v1/
  acceptance_summary.json      # 各场景 PASS/FAIL/BLOCKED 汇总
  g0/{report.json,evidence/}
  g1/{report.json,evidence/,screenshots/}
  g2/{report.json,evidence/,screenshots/,runs/...}
  g3/{report.json,evidence/,screenshots/,runs/...}
  g4/{report.json,evidence/,runs/...}
  g5/{report.json,evidence/,runs/...}
  g6/{report.json,evidence/,runs/...}
  g7/{report.json,evidence/,screenshots/,runs/...}
```

---

## 4. 复现命令

```powershell
# 1) 全量测试（不含需缺失依赖的 tests/envs）
.\.venv\Scripts\python.exe -m pytest tests\navigation tests\llm_agent tests\phase1 tests\multi_agent tests\story tests\pokemon -q

# 2) 创建（或重建）合法验收存档
.\.venv\Scripts\python.exe scripts\multi_agent_make_states.py --all

# 3) 真实验收（需要 .env 中的真实 DeepSeek key 与 local/ 下的合法 ROM）
.\.venv\Scripts\python.exe scripts\multi_agent_acceptance.py --scenario g0 --scenario g1
.\.venv\Scripts\python.exe scripts\multi_agent_acceptance.py --scenario g2 --scenario g3
.\.venv\Scripts\python.exe scripts\multi_agent_acceptance.py --scenario g4 --scenario g5 --scenario g6
.\.venv\Scripts\python.exe scripts\multi_agent_acceptance.py --scenario g7

# 4) 多 Agent 交互运行（可选 GUI）
.\.venv\Scripts\python.exe scripts\run_multi_agent.py --headless --max-decision-turns 40
```

若 API key 缺失/余额不足/网络失败：真实验收场景会诚实地记录 `BLOCKED`，
不会用 mock 结果替代（G4/G5/G6 的离线 fake 只用于基础设施场景并在报告中标注）。
