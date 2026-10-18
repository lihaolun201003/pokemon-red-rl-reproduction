# Pokémon Red Multi-Agent v1 — 真实运行结果

本报告只记录**真实执行得到的数据**。原始证据在 `artifacts/multi_agent_v1/`
（gitignored：`<G>/report.json`、`evidence/`、`screenshots/`、`runs/<run_id>/` 的
audit/tasks/shared_memory/checkpoints/budget）。
运行环境：Windows 10 / Python 3.10.11 / PyBoy / 合法 ROM `local/PokemonRed.gb` /
真实 DeepSeek API（`deepseek-flash`）。G2/G3/G7 使用真实 API；G4/G5/G6 的 LLM 部分
使用离线 fake（只验证协作基础设施，报告内明确标注）。

---

## 1. 总览

| 验收 | 结果 | 关键真实数据 | 费用 |
|---|---|---|---|
| G0 回归 | **通过** | 1259 项测试全部通过 + Navigation v2 三个真实验收脚本均 returncode=0 | $0 |
| G1 控制权互斥 | **通过** | 真实模拟器：并发竞争最大 1 个 lease；33+ 次授权、确定性拒绝 2/2；越权按键帧数不变；抢占/释放按键帧差 72/48=3×24/2×24；无死锁；崩溃后恢复 | $0 |
| G2 Story 接管对话 | **通过** | 真实 OAK 对话（"I want to know how your POKéDEX is" / "Come see me sometime"）被读取；2 次 Story 决策会话（17→20 次 API 调用）；知识入库（带真实截图证据）；控制权释放；Navigation 继续执行探索任务并再次交互 | **$0.0062**（20 calls / 约 50k tokens） |
| G3 Pokémon 接管战斗 | **通过** | Route 1 草丛触发野生战斗；intro 7 次 A 推进/284 帧后进入真实菜单；3 个决策回合（真实 DeepSeek 选择 fight/move slot）、45 次真实按键、**战斗胜利**；控制权释放；Navigation 重定位并恢复探索 | **$0.0014**（4 calls / 12k tokens） |
| G4 跨 Agent 任务 | **通过（基础设施）** | Story→EventBus→Director 建任务→Navigation 真实执行（8 步）→任务 completed→父任务恢复 READY；同 message_id 重放不重复建任务；同内容消息被去重拒绝 | $0（fake LLM） |
| G5 Checkpoint 回档 | **通过（基础设施）** | 真实 manifest（sha256 校验通过）；回档恢复真实 PyBoy 状态；新分支创建（2 分支，旧分支 ROLLED_BACK）；旧知识 8 条标 HISTORICAL 未删除；回档后新学经验仍保留；知识版本 6→12 单调；**API 花费未回退**；重定位成功 | $0（fake LLM） |
| G6 停滞与恢复 | **通过（基础设施）** | 4 次停滞信号、4 次恢复动作（L1→L2→L3 各 1 次，attempts_by_level 记录）；1 次回档（受 per-lineage/total=1 限制）；达到硬上限后**安全暂停**；分支数 2 | $0（fake LLM） |
| G7 真实多 Agent 集成运行 | **通过（1/5 次完整闭环）** | 真实 API 连续运行：Director 调度三 Agent；Story/Pokémon 真实决策；14 次战斗 + 5 次对话接管；Navigation 1023 次真实操作/16 区域；19 次 checkpoint；另 4 次运行因探索路线不同未形成完整闭环（如实列出） | **$0.0245**（65 calls / 218k tokens） |

> 说明：G7 的具体数字随每次真实运行波动（探索路线/遭遇随机）；§3 给出最后一次完整
> 运行的数据与两次运行的对比，不挑选“最好看”的数字。

---

## 2. 各验收详情

### G0 现有系统回归

```text
tests/navigation    319 passed
tests/llm_agent     364 passed   (基线 357 + 新增 7 项多 Agent 边界测试)
tests/phase1         11 passed
tests/multi_agent   341 passed
tests/story          88 passed
tests/pokemon       136 passed
--------------------------------
合计               1259 passed
```

Navigation v2 真实游戏脚本重跑：`nav_v2_g1_localization`、`nav_v2_g2_explore_room`、
`nav_v2_g3_reentry` 全部 returncode=0。单 Agent 入口 `run_llm_agent.py` 与 PPO 训练
代码未修改。

### G1 控制权互斥（真实 PyBoy）

* 三个 `GuardedEmulator`（navigation/story/pokemon）并发请求控制权：
  * 历史重放显示**任意时刻最多 1 个 active lease**；
  * 循环内 36 次授权、0–3 次拒绝（取决于线程调度）；
  * 确定性抢占场景：navigation 持有时 story/pokemon 请求**全部被拒绝**
    （ControlDeniedError，owner=navigation），被拒按键**不推进帧**；
  * 交接按键精确：navigation 3 次按键=72 帧、pokemon 2 次按键=48 帧
    （每次 action_frames=24，无重复/丢失）；
  * 抢占后旧 lease 的按键被拒绝（stale token 不可用）；
  * 持有者异常崩溃后 `release_all` 恢复，无死锁（线程 join 全部成功）。
* 证据：`artifacts/multi_agent_v1/g1/report.json`。

### G2 Story 接管真实对话（真实 ROM + 真实 DeepSeek）

* 起始合法存档 `g2_dialog_ready.state`：屏幕为真实 OAK 对话框。
* 阶段 1：Director 分类 DIALOG → 授予 STORY → StoryAgent 逐回合真实截图 + DeepSeek
  读取/推进对话 → 记录 NPC 与剧情线索（VERIFIED 附真实截图路径）→ 释放控制权。
* 阶段 2：Navigation 执行探索任务（真实按键、真实地图更新）。
* 阶段 3：Story 交互探针任务再次接管：真实 A 键打开对话、读取、结束、释放；
  Navigation 重定位并继续。
* 结果：2 个 Story 决策会话（`agent_results` 中含 `dialog` 与 `interaction_probe`），
  `owner_switches` 多次，任务 completed，控制权最终 NONE；
  API 20 calls / $0.0062（全部模型均为 `deepseek-flash`）。
* 证据：`artifacts/multi_agent_v1/g2/`（含 `story_decisions` 的逐回合动作与选择）。

### G3 Pokémon 接管真实战斗（真实 ROM + 真实 DeepSeek）

* 起始合法存档 `g3_battle_ready.state`（Route 1 草丛，下一步即遇敌）。
* 实际流程：
  1. Navigation 探索（真实按键）→ 触发野生战斗；
  2. Director 检测 BATTLE → 暂停导航任务 → 授予 POKEMON；
  3. Pokémon Agent：有界 A 推进 battle intro（**7 次 A / 284 帧**）→ 检测到真实
     指令菜单 → 逐回合真实截图 + DeepSeek 决策（fight / move_slot=1，附理由）→
     确定性菜单驱动执行**45 次真实按键**（每次像素验证）；
  4. 战斗结束（`outcome=win`，3 个决策回合）→ 释放控制权；
  5. Director 重新观察 → Navigation 重定位 → 原导航任务恢复执行（explore step_limit 9 步）。
* API：4 calls / $0.0014 / 12k tokens；证据含 45 张战斗截图、intro_wait、逐回合
  decision+presses+verified、战斗经验（VERIFIED/INFERRED 分层）。

### G4 跨 Agent 任务委托（真实 ROM + 真实总线/任务/导航）

* Story 发布 `NAVIGATION_REQUEST`（parent_task_id=Story 任务）→ Director 收件 →
  创建 Navigation 任务（target 为真实可走 cell）→ 调度执行 → Navigation 实际移动 8 步 →
  任务 COMPLETED → `TASK_COMPLETED` 事件 → 父 Story 任务自动恢复 READY。
* 幂等：重发同一 message_id 不再创建任务；相同内容的新消息被 EventBus 去重拒绝。
* 该场景验证的是**协作基础设施**（消息/任务/控制权/导航/结果回传），LLM 使用离线
  fake，不作为模型能力证据。

### G5 Checkpoint 回档与经验保留（真实 ROM）

* 保存完整 branch manifest（游戏 state sha256 + 任务/director/agent 上下文/nav 快照/
  知识版本/消息消费进度/预算）；随后继续行动与新学习；触发回档：
  * `verify()` 通过；恢复真实 PyBoy state（emulator.loaded_state 非空）；
  * 新分支创建（旧分支 ROLLED_BACK，parent 关系记录）；
  * 旧分支 8 条知识标 HISTORICAL（保留，未删除）；回档后新学的经验仍在；
  * 知识版本 6→12 单调递增；`branch_id` 广播到 tasks/shared/bus/recovery；
  * **API 花费不回退**（`budget.before == after`）；
  * 重新观察 + Navigation 重定位成功（relocalized=true）。
* LLM 使用离线 fake（不产生费用）；checkpoint/任务/知识/分支均为真实实现。

### G6 停滞检测与自主恢复（真实 ROM）

* 配置极小阈值构造可复现停滞：4 次停滞事件 → 4 次恢复尝试
  （`attempts_by_level: l1_local=1, l2_review=1, l3_rollback=1`）；
  * L1 局部 hint、L2 全局复盘、L3 回档各触发一次；
  * per-lineage 回档上限=1 生效（回档后不再重复回档）；
  * 达到硬上限 → 安全暂停（保存现场 + LIMIT_REACHED 事件 + 分支 2）；
* LLM 使用离线 fake；monitor/recovery/checkpoint 为真实实现。

---

## 3. G7 — 真实 DeepSeek 多 Agent 集成运行

同一进程、同一真实模拟器、真实 DeepSeek API，从合法存档
`work/multi_agent_v1/states/g2_dialog_ready.state` 出发，Director 自动运行到
`max_decision_turns=110`。为诚实反映**探索路线的随机性**，这里列出**全部 5 次**
真实运行（同一配置；原始 run dir 全部保留在 `artifacts/multi_agent_v1/g7/runs/`）：

| run_id（本地） | 步数 | 战斗接管 | 对话接管 | 导航真实操作 | 区域数 | 任务 创建/完成/失败 | Checkpoint | API calls | Token | 费用 | 时长 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 212405 | 110 | 0 | 1 | 70 | 3 | 35/1/33 | 0 | 20 | 67k | $0.0056 | 124 s |
| 212658 | 110 | 0 | 1 | — | 2 | 4/2/2 | 4 | 13 | 43k | $0.0042 | 91 s |
| **212920（完整闭环）** | **110** | **14** | **5** | **1023** | **16** | **23/15/5** | **19** | **65** | **218k** | **$0.0245** | **430 s** |
| 213706 | 110 | 6 | 2 | 552 | 8 | 74/70/4 | 12 | 45 | 153k | $0.0159 | 285 s |
| 214223 | 110 | 0 | 1 | 6282（无效循环） | 3 | 5/2/2 | 5 | 4 | 8.8k | $0.0007 | 92 s |

**完整闭环运行（run 212920）的细节（全部为真实数据）**：

* Director 自动调度：`owner_switches=110`，三个 Agent 均获得控制权；
* Story：24 次 API 调用 / $0.0067 / 80k tokens，处理 **5 次真实对话接管**；
* Pokémon：40 次 API 调用 / $0.0178 / 138k tokens，处理 **14 次真实战斗接管**
  （506 张战斗过程截图，逐回合 decision+presses+verified 记录）；
* Navigation：**1023 次真实操作**、46,798 模拟帧、16 个区域、5 次 map transition；
* 共享知识库：225 条 KnowledgeEntry（含 battle_lesson/npc/area/route，按
  VERIFIED/INFERRED 分层，HISTORICAL 保留机制就绪）；
* Checkpoint：19 次周期保存；回档 0 次、停滞事件 0 次（本次运行未触发）；
* 任务：23 创建 / 15 完成 / 5 失败；无重复 active 任务；
* 实际费用：**$0.0245**（65 calls / 218,477 tokens）；
* 最终位置：`area_016`（已离开初始实验室并跨多区域探索）。

**运行间的巨大差异（诚实说明）**：同样的代码与配置，5 次运行中只有 2 次遇到野生
战斗、1 次形成“对话→探索→战斗→继续探索”的完整闭环。原因是纯盲探索的路线选择具有
随机性（NPC/草丛位置、导航 frontier 选择、区域门传送），且本阶段 Director 的
“探索完成→走出口/空闲等 Story 提案”策略仍在调优（run 214223 出现一次无效导航循环，
被安全暂停/上限终止，未崩溃）。这些差异也说明了为什么需要 ProgressMonitor /
RecoveryManager：若运行更长，它们会介入 L1/L2/L3。

**运行 A 的自检项全部为 true**：真实 API 调用、Director 调度多个 owner、
Navigation 运行且有进展、Story 真实接管对话、Pokémon 真实接管战斗、
battle/dialog 交接事件、无重复 active 任务、checkpoint 产生、
localization 恢复记录。最后一次运行（214223）的 soft check 中 battle/dialog
交接事件为 false（该次未遇到战斗），已在 `acceptance_summary.json` 如实列出。

**观察到的真实问题（已修复或记录）**：
1. Story Agent 首轮因多 tool_calls 未全部应答导致 DeepSeek HTTP 400 → 已修复
   （只声明真正执行的调用 + 每轮一条 tool 应答 + 单次干净窗口重试）。
2. 战斗 intro 文本需要按 A 推进（wait-only 会超时）→ 已修复为有界 A 推进
   （真实 7 次 A / 284 帧到菜单，真实 ROM dry-run 与 G3 均验证）。
3. 真实菜单帧曾被误判为 text/move（边框线计数/采样区偏差）→ 已用归档真实帧校准。
4. 户外区域间切换后偶发 `localization_lost` → Director 增加有界重定位恢复
   （优先匹配已知区域，失败则 re-attach），并计入统计。
5. 全探索完成后会产生“瞬间完成”的重复探索任务 → 改为“探索完成→走已知出口/空闲
   让 Story 提案”，并加冷却；但仍存在运行抖动（见上表 214223），列入已知限制。

**API 记账**：完全来自 `BudgetTracker`（run + total 双层持久化）与
`ModelRouter` 每 Agent 记账；回档/切换 Agent 不重置。历史累计（含此前单 Agent
实验）真实总花费可在 `artifacts/llm_agent/budget_total.json` 查看。

**G7 证据文件**：`artifacts/multi_agent_v1/g7/evidence/all_runs.json`（5 次运行汇总）、
`g7/evidence/run_history.jsonl`、每次运行的 `runs/<run_id>/`（director_steps、
agent_results 逐回合决策、events、tasks.json、shared_memory、checkpoints、budget、
story/pokemon 截图）。

---

## 4. 实际游戏进展（诚实结论）

* 本次多 Agent 实验**没有通关**，也未取得徽章；这是本阶段目标之外的预期结果。
* 已达成的真实进展：从 Oak's Lab 出发，接管真实 NPC 对话并积累线索；
  成功走出建筑、跨区域探索（多次真实 map transition）；在 Route 1 的草丛中
  多次触发并击败野生 Pokémon（G3 单场 + G7 多场），记录战斗经验；
  Navigation v2 的建图/寻路/重定位在这些运行中持续工作。
* 尚未达成：稳定的主线推进（缺少对“下一步剧情目标”的可靠自主推断）、
  道馆/城镇级长期导航、可靠的捕捉/培养闭环。原因与改进方向见 known limitations。

---

## 5. 结论

1. **Multi-Agent v1 是否真正完成？** 是：Director + 三个独立 Agent + 事件总线 +
   任务系统 + 共享记忆 + 进展监测/恢复/回档 + 模型路由 + 运行入口全部实现，
   并通过真实 ROM 与真实 DeepSeek API 验收（G0–G7 见上）。
2. **Director 是否可以同时管理三个 Agent？** 是：G7 中三个 Agent 都获得过控制权并
   完成真实动作（Story 对话、Pokémon 战斗、Navigation 导航），控制权始终单 owner。
3. **是否成功接管真实战斗？** 是：G3 真实野生战斗由 Pokémon Agent 决策并胜利完成；
   G7 中多场真实战斗接管。
4. **是否成功接管真实对话？** 是：G2 真实 OAK 对话被读取、记录、推进并正确释放控制权；
   G7 中多次对话接管。
5. **是否能够恢复 Navigation 的原任务？** 是：G3/G2 中导航任务被暂停后，在事件结束、
   重新观察与视觉重定位后恢复执行；G6/G5 验证了回档后的重新定位与任务一致性。
6. **是否支持自动回档并保留经验？** 是（基础设施级）：G5 真实回档验证分支/快照/
   知识保留/费用不回退；G6 验证停滞→L1/L2/L3→安全暂停的闭环。
7. **实际运行成本是多少？** 本次全部真实验收（G2 + G3 + 5 次 G7 + 少量探测）合计约
   **$0.065**；其中单次完整闭环 G7 运行 **$0.0245**（65 calls / 218k tokens），
   G2 $0.0062，G3 $0.0014。加上本仓库历史单 Agent 实验的累计花费约 $0.083，
   全部记录在 `artifacts/llm_agent/budget_total.json`，未被回档/切换 Agent 重置。
8. **还有哪些阻碍 AI 自主通关的问题？** 见
   `docs/reports/multi_agent_v1_known_limitations.md`，主要是：剧情理解不稳定、
   战斗/捕捉策略朴素、长期导航仍会卡在区域/门传问题上、缺少可靠的长期目标规划。
