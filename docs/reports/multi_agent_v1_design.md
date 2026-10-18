# Pokémon Red Multi-Agent v1 — 设计报告

状态：已实现并通过真实游戏 + 真实 DeepSeek API 验收（验收数据见
`docs/reports/multi_agent_v1_results.md`、`docs/reports/multi_agent_v1_validation.md`）。

本文只描述**实际落地**的代码。基线为 Navigation v2（commit `1dff809`，687 项测试），
本阶段在其之上新增多智能体协作子系统，未重写、未降级 Navigation v2。

---

## 1. 最终架构（实际模块与调用关系）

### 1.1 目录

```text
llm_agent/
├── multi_agent/                     # 新增：协作子系统（Lead 实现 + 并行队友实现）
│   ├── contracts.py                 # 冻结协议：AgentMessage/EventType/Task/KnowledgeEntry/ControlLease/BranchRecord
│   ├── event_bus.py                 # 持久化优先级事件总线（JSONL + state，去重/重放/幂等）
│   ├── control_arbiter.py           # 唯一控制权 lease/token + GuardedEmulator 代理
│   ├── task_manager.py              # 任务 DAG、生命周期、去重、分支感知、原子持久化
│   ├── scheduler.py                 # 确定性任务选择（可解释 score、老化/失败惩罚/owner 轮换）
│   ├── shared_memory.py             # 共享世界知识库（证据/置信度/验证状态/branch 标记）
│   ├── progress_monitor.py          # L1/L2/L3 分层进展监测（阈值全部可配）
│   ├── recovery.py                  # L1 局部调整 / L2 全局复盘 / L3 回档 + 预算与冷却
│   ├── checkpoint.py                # 分支 checkpoint manifest（复用现有 CheckpointStore）
│   ├── agent_registry.py            # 注册表：各 agent 能力/工具/任务类型
│   ├── agent_context.py             # 每 agent 独立上下文 + 唯一动作原语 ActionPrimitives
│   ├── model_router.py              # 统一模型路由 + 共享预算 + 按 agent 记账
│   ├── director.py                  # GameDirector 确定性状态机
│   └── integration.py               # NavigationAdapter + MultiAgentSession（装配/运行/落盘）
├── story/                           # 新增：Story Agent（独立 prompt/记忆/工具）
│   ├── agent.py  prompts.py  memory.py  tools.py
├── pokemon/                         # 新增：Pokémon Agent（战斗菜单驱动/战斗记忆/培养结构）
│   ├── agent.py  prompts.py  memory.py  tools.py
└── （Navigation v2 / runtime / tools / memory / budget / api_client / emulator 均未重写）

configs/multi_agent_v1.json         # 多 Agent 运行配置（可单独调 模型/预算/阈值/回档上限）
scripts/run_multi_agent.py          # 多 Agent 运行入口（不影响 run_llm_agent.py）
scripts/multi_agent_acceptance.py   # G0–G7 真实验收
scripts/multi_agent_make_states.py  # 由合法脚本操作生成验收用 PyBoy 存档（测试脚手架）
tests/multi_agent/ tests/story/ tests/pokemon/   # 新增测试
```

### 1.2 运行时装配（`MultiAgentSession`）

```text
GameBoyEmulator（唯一实例，来自 llm_agent/emulator.py）
        │
ControlArbiter（唯一控制权）───GuardedEmulator 代理（每 agent 一个）
        │
EventBus（JSONL 持久化）  TaskManager  TaskScheduler  SharedWorldMemory
ProgressMonitor  RecoveryManager  BranchCheckpointManager(复用 CheckpointStore)
        │
ModelRouter（共享预算 BudgetTracker + 每 agent model/限额/记账）
        │
NavigationAdapter ──> 现有 NavigationAgent / NavigationController / WorldModel（未改动）
StoryAgent（llm_agent/story）        PokemonAgent（llm_agent/pokemon）
        │
GameDirector（deterministic main loop）
```

`MultiAgentSession` 是唯一持有 `GameBoyEmulator` 的对象；三个专业 Agent 只拿到经
`ControlArbiter.proxy()` 产生的 `GuardedEmulator`。Director 只观察（`observe()`）与调度，
自身不按任何按键。

### 1.3 控制权数据流

```text
Director.step()
  ├─ observe() + observation.analyze()        # 纯像素上下文分类
  ├─ drain inbox（EventBus → Director）        # 任务/事件/agent 消息
  ├─ 紧急上下文优先：BATTLE / DIALOG / MENU / TRANSITION
  │     ├─ suspend 当前任务（PAUSED，保留 task_id/上下文）
  │     ├─ arbiter.request(owner) → ControlLease
  │     ├─ agent.handle_*（真实截图、真实按键、真实 DeepSeek 决策）
  │     ├─ arbiter.release(owner, context=...)
  │     ├─ 结果 → TaskManager / SharedWorldMemory / ProgressMonitor / EventBus
  │     └─ post_interruption：重新 observe + Navigation relocalize + resume 原任务
  └─ 否则按 Scheduler 选择任务 → 派发对应 owner（bounded step）
```

---

## 2. 核心行为

### 2.1 Director 如何决定控制权

* **谁行动**：只有持有 `ControlLease` 的 agent 能通过其 `GuardedEmulator` 按键；
  `press/ wait / run_batch` 校验 lease_id + generation，未授权调用直接抛
  `ControlDeniedError` 且不触碰模拟器。读操作（`observe`）对所有 agent 开放。
* **决定规则（确定性，非 LLM）**：
  1. 硬资源上限（运行时间/模拟器操作数/预算/监控硬信号）→ 安全暂停；
  2. 屏幕出现 BATTLE → Pokémon Agent 立即接管（最高优先级）；
  3. DIALOG/MENU → Story Agent 接管；
  4. TRANSITION → Navigation 处理转场 + 重定位；
  5. 否则 Scheduler 从任务队列选任务（恢复 > 续做高优 > 剧情 > 导航 > 低价值，
     含 aging/失败惩罚/owner 轮换），派发对应 owner；
  6. 空闲且无可调度任务 → Story Agent 提出新任务（限频），否则创建/复用探索任务。
* **不每一步调用 Director LLM**：Director 的 DeepSeek 复盘只在停滞/冲突时触发，
  且有最小间隔、预算与次数上限（`director_llm_*`）。

### 2.2 Agent 如何提交任务

三个 Agent 都可创建任务/子任务、修改状态、提出跨模块请求：

* 直接在返回值中给出 `tasks_created: [{owner_agent, task_type, description, priority, payload}]`；
  Director 用 `TaskManager.create_task()` 做**去重**（owner+type+description+parent）后入队。
* 通过 EventBus 发布 `NAVIGATION_REQUEST` / `TASK_*` / `STORY_DISCOVERY` 等结构化消息；
  Director 在 `_handle_message()` 中转换为任务/知识，且**按 message_id 幂等**
  （持久化 `processed_messages.json`，崩溃重放不会重复执行）。
* Story 的对话结果会自然产生导航类任务（talk/investigate/explore），形成
  Story → Director → Navigation 的委托链。

### 2.3 消息如何传递

* `AgentMessage`：`message_id / correlation_id / parent_task_id / source_agent /
  target_agent / event_type / payload / priority / created_at / branch_id (+sequence)`。
* `EventBus`：内存优先级队列 + 每行 JSONL 日志 + `.state.json`（原子写）记录
  dedupe 窗口、每 role 消费进度、branch、计数；启动时自动加载并 replay 未消费消息。
* 类型/参数验证（`EVENT_PAYLOAD_SPEC`）、重复消息去重（message_id 幂等 + 内容
  dedupe_key 窗口）、优先级、错误返回、可追踪审计、重启恢复均已实现并测试。
* 不引入 Redis/Kafka；单机单进程内存队列 + 本地持久化。

### 2.4 共享记忆如何更新

* `SharedWorldMemory` 保存 `KnowledgeEntry`：category/content/source_agent/
  evidence_refs/confidence/verification_status/branch_id/created_at/updated_at。
* **只有真实观察带证据才能 `VERIFIED`**（无证据的 verified 会被降级为 inferred）；
  推断/假设分别标注，可被 `refute/supersede`。
* 检索按类别/关键词/subject，且有严格字符预算（不会把全部记忆塞进 prompt）。
* 回档时旧分支条目标记为 `HISTORICAL`（保留内容与证据，不删除、不当事实使用）；
  新分支的 `branch_id` 传播到任务/总线/知识库。
* 空间信息仍以 Navigation v2 的世界地图为唯一权威，共享库不复制坐标数据库。

### 2.5 Checkpoint 如何恢复

`BranchCheckpointManager`（复用现有 `CheckpointStore`）保存**完整 snapshot manifest**：

```text
branch_id / parent_branch_id / checkpoint_id / frame / state_sha256 / state_size
tasks_snapshot / director_state / agent_contexts
navigation_snapshot / knowledge_version / message_progress
budget_snapshot / memory_snapshot / reason / evidence / manifest_version
```

* 原子写 + sha256 校验；`checkpoint_{seq}` 单调计数避免帧重启后 ID 冲突；
* **回档步骤**：保存回档前 checkpoint → 记录原因/证据 → 创建新 `branch_id`
  → `release_all` + SYSTEM 权限 `load_state()` 恢复 PyBoy 状态 → 恢复任务/director/
  agent 上下文快照 → 重新 observe + Navigation relocalize → 新分支广播到
  tasks/shared/bus/recovery → 后续任务继续；
* **API 花费、尝试次数、学习经验不因回档重置**（费用账本单调；
  经验写独立 `learning.json` 并在知识库中保留为 HISTORICAL）。
* 测试覆盖：manifest 往返/校验/篡改检测、rollback 分支链、知识保留与旧分支标注、
  预算不回退。

### 2.6 任务如何继续（中断-恢复）

```text
Navigation explore/navigate（bounded）
   → 检测到 BATTLE/DIALOG/TRANSITION
   → 任务 PAUSED（保留 id/描述/payload/attempts）
   → 控制权交给对应 Agent
   → Agent 结束后 Director 重新截图观察 + Navigation 视觉重定位
   → 原任务 READY 继续（或按 Scheduler 重新排序）
```

真实验收 G2/G3 已证明该链路：Navigation 先行动，事件触发后 Story/Pokémon 接管，
结束后重新观察/重定位，任务恢复。

---

## 3. 十一位架构决策的落地对照

| # | 决策 | 实现 |
|---|---|---|
| 1 | 完全盲探索 | 所有 Agent 只经 `Frame`/`ScreenAnalysis`/`GuardedEmulator`；`tests/llm_agent/test_import_boundaries.py` 强制 multi_agent/story/pokemon 不 import PyBoy/`llm_agent.progress`、无 RAM 地址、无地点/剧情词；验证脚本的 RAM 只存在于 `scripts/multi_agent_make_states.py`（测试脚手架） |
| 2 | Pokémon Agent 完全自主 | `handle_battle` 每回合真实截图 → DeepSeek 决策（攻击/招式序号/逃跑/捕捉/换人/观察）→ 确定性菜单驱动执行；无固定战斗脚本 |
| 3 | Director 混合调度 | Python 状态机 + 确定性 Scheduler；Director LLM 仅在停滞/复盘时按间隔调用（`director_review`），不逐步调用 |
| 4 | 独立记忆 + 共享知识 | 每 agent 独立 `AgentMemory` root + `AgentContext`；`SharedWorldMemory` 按证据/置信度/验证状态分层 |
| 5 | 自主纠错/自动回档 | `ProgressMonitor` → `RecoveryManager` L1/L2/L3；预算/尝试/停滞/循环保护；硬上限安全暂停 |
| 6 | 三独立 Agent 共用 API | 独立 system prompt/上下文/工具/记忆/任务；`ModelRouter` 支持每 agent 模型覆盖 + 共享 `BudgetTracker` |
| 7 | 动态任务优先级 | `TaskScheduler` 可解释 score（优先级/类型/aging/attempts/失败/owner 轮换），Director 可复盘改优先级 |
| 8 | 回档保留学习经验 | branch_id + verification_status(HISTORICAL) + 独立 learning 存储；费用/尝试不重置 |
| 9 | 分层进展监测 | `ProgressMonitor` 按模块真实增益判定（新格/新区域/线索/战斗结果），按键数不算进度；L1 本地/L2 复盘/L3 回档 |
| 10 | Agent 自主创建任务 | 三个 agent 均可创建任务/子任务并改状态；Director 只做去重/依赖/优先级/调度 |
| 11 | 结构化消息总线 | `EventBus` + `AgentMessage`；唯一 `ControlArbiter`，禁止并发按键 |

---

## 4. 与现有系统的集成点（未破坏 Navigation v2）

1. **Navigation v2 适配**：`NavigationAdapter` 只包装现有
   `NavigationAgent(emulator=GuardedEmulator, world_root=..., on_frame/on_step/on_status)`，
   暴露 `explore / navigate_to / status / local_map / world_summary / relocalize /
   handle_transition / pause / resume / save`。定位、建图、A*、Frontier、世界图、
   持久化、可视化代码零改动；视觉位移验证、UNKNOWN 不更新坐标、exactly-once 移动
   同步、区域匹配、导航预算全部保留。
2. **控制权约束**：`GuardedEmulator` 是代理层，不是重写 NavigationController；
   所有按键（含 legacy press/walk/execute_actions 若在 multi-agent 模式下使用）
   都必须经过仲裁入口。`NavigationController` 的 `on_step` 事件仍驱动中断检测。
3. **工具层**：多 Agent 模式下不使用单 Agent 的 `ToolExecutor` 逐步按键路径；
   每个专业 Agent 有显式工具白名单（Story: observe/advance_dialog/choose_option/
   record_fact/propose_task/finish；Pokémon: battle_action/observe/record_lesson/finish）。
   单 Agent 的 `run_llm_agent.py` 及其工具完全未改动。
4. **复用而非重造**：`CheckpointStore`、`AuditLogger`、`BudgetTracker`、
   `AgentMemory`、`DeepSeekClient`、`GameBoyEmulator`、`observation` 全部复用。
5. **运行入口**：`scripts/run_multi_agent.py`（独立入口，默认读
   `configs/multi_agent_v1.json`，支持模型/预算/上限/停止文件等覆盖），
   `run_llm_agent.py` 行为不变。

---

## 5. 模型与预算

* `ModelRouter`：`director.model / story.model / pokemon.model / navigation.model(可选)`
  独立可配；空值回落到全局 `api.model`（默认 `deepseek-flash`，未更换 API 型号）。
* 共享 `BudgetTracker`（run + total 双层持久化）保证全局上限；每 agent 另有
  calls/USD 上限与记账（估算成本、真实 token、延迟、错误、时间戳），账本持久化
  `multi_agent/model_usage.json`；`restore` 只允许费用单调不回退。
* 成本控制：Agent 截图缩放（默认 3×，640×576→480×432）、短输出（400 tokens）、
  只在需要时调用 Director LLM、真实缓存命中由 DeepSeek 自动处理。

---

## 6. 测试与证据策略

* 单元/集成：`tests/multi_agent`（协议/总线/仲裁/任务/调度/记忆/监控/恢复/checkpoint/
  director/session）、`tests/story`、`tests/pokemon`；全部离线、快速、无 ROM/API。
* 扩展 `tests/llm_agent/test_import_boundaries.py`：新增模块同样受“无 RAM/无 PyBoy/
  无剧透”约束，并新增各 Agent 上下文/工具权限测试。
* 真实验收：`scripts/multi_agent_acceptance.py --scenario gN`，证据写入
  `artifacts/multi_agent_v1/<gN>/`（gitignored）：截图、事件、任务、知识、checkpoint、
  manifest、API 记账、report.json。Mock/Smoke 结果从不作为验收证据。
* 验收用合法存档由 `scripts/multi_agent_make_states.py` 通过真实游戏操作创建
 （`work/multi_agent_v1/states/*.state`，gitignored；不修改 ROM/RAM 数据）。

---

## 7. 关键设计取舍与已知边界

* **Director 不按按键**：动作只来自专业 Agent，Director 只观察/调度/恢复。
* **事件上下文用像素分类**（`observation.analyze`）而非 RAM：可能把战斗过场、
  升级文本等短暂界面判为 BATTLE/DIALOG，Agent 必须自己判定真实状态（
  Story 有 `no_dialog`，Pokémon 有 intro/menu 等待逻辑）。
* **Director LLM 罕见**：仅在 L2 复盘/复杂冲突时调用，不能指定战斗操作。
* **回档语义**：游戏状态回档 + 分支隔离；历史知识保留但标 HISTORICAL；
  费用/尝试次数全局单调。
* **消息至少一次投递**：消费端按 message_id 幂等；不可逆动作在重试前先重新观察
  （由 context 分类 + Agent 内部先 observe 保证）。
* 详细限制见 `docs/reports/multi_agent_v1_known_limitations.md`。
