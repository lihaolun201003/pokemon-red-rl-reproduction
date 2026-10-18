# Pokémon Red Navigation v2 — 验收报告（真实游戏 + 真实运行）

日期：2026-10-18
仓库：`C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`
设计文档：[navigation_v2_design.md](navigation_v2_design.md)
结果报告：[navigation_v2_results.md](navigation_v2_results.md)

本报告只写**真实运行得到的数据**。所有 G1–G4 验收都使用：

- 真实 ROM：`local/PokemonRed.gb`（英文 pret/pokered build，SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`）；
- 真实 PyBoy 2.4 模拟器（headless, speed=0）；
- 合法存档：`local/has_pokedex_nballs.state` 与由合法脚本游戏创建的工作存档
  （`work/nav_v2/states/*.state`，gitignored；创建方式见 `work/nav_v2/make_states.py`，未修改 ROM/RAM）；
- **RAM 真值只出现在验收脚本中**（`llm_agent.progress` / `scripts/nav_v2_g*`），用于独立校验
  视觉定位漂移；导航模块与 DeepSeek 决策链路从不读取 RAM（由 `tests/llm_agent/test_import_boundaries.py` 强制）。

复现命令（PowerShell，项目根目录）：

```powershell
# G1 纯视觉定位验收（>=100 次真实方向操作）
.\.venv\Scripts\python.exe scripts\nav_v2_g1_localization.py

# G2 陌生房间自主探索（Oak's Lab）
.\.venv\Scripts\python.exe scripts\nav_v2_g2_explore_room.py --max-total-steps 900 --round-steps 300

# G3 离开并重新进入已探索区域
.\.venv\Scripts\python.exe scripts\nav_v2_g3_reentry.py --max-explore-steps 1000

# G4 有限预算无人值守多区域长跑（纯导航，无 LLM）
.\.venv\Scripts\python.exe scripts\nav_v2_g4_long_explore.py --max-total-steps 3000 --max-seconds 1200

# G4-API 真实 DeepSeek API + Navigation v2 集成运行（真实 API，费用约 $0.003）
.\.venv\Scripts\python.exe scripts\run_llm_agent.py --config configs\navigation_v2_api.json --fresh --headless --max-turns 8 --run-dir artifacts\nav_v2\api_run

# 单元/集成测试
.\.venv\Scripts\python.exe -m pytest tests\navigation tests\llm_agent -q
```

---

## G1 — 纯视觉位移定位验收（真实 ROM/PyBoy，122 次方向操作）

脚本：`scripts/nav_v2_g1_localization.py`；结果：`artifacts/nav_v2/g1/report.json` +
`ops.jsonl`（逐操作记录：真实坐标、视觉估计、证据分数）。

场景覆盖（全部真实操作，非 Mock）：

| 场景 | 操作数 | 内容 |
|---|---|---|
| `lab_walk` | 50 | Oak's Lab 内确定性随机方向行走（家具墙、走动 NPC） |
| `lab_turn_block` | 16 | 转角处"原地转身 / 连续撞同一墙 / 正常移动"混合 |
| `multi_tile` | 8 | 按住方向 32 帧（一次动作移动 2 格） |
| `live_menu` | 2 | 打开 START 菜单后按方向键（真实菜单中断） |
| `pallet_walk` | 30 | Pallet Town 室外随机行走（NPC/动画噪声） |
| `transition_out/in` | 10 | 进出 Oak's Lab 门口传送（区域切换） |
| `real_frames` | 6 | 历史真实运行截图：战斗×2、对话×2、菜单×1、转场×1（真实画面，非合成） |

指标（真实 RAM 真值对照）：

| 指标 | 数值 |
|---|---|
| 方向操作总数 | **122** |
| 真实位移次数（非传送） | **88** |
| 正确位移（含方向与格数） | **88 / 88 = 100.00%** |
| **错误位移（没走却认为走了）** | **0** |
| 漏检位移（走了却判 blocked） | **0** |
| UNKNOWN 次数 | **0** |
| 定位丢失次数 | **0** |
| 区域切换正确检测 | **4 / 4**（0 误判新地图） |
| 菜单/对话/战斗正确中断 | **6**（真实菜单 + 历史真实战斗/对话截图） |
| 累计位置误差（视觉 pose vs RAM 真值） | **0 格**（sum=0, max=0） |

**关键验收条件满足**：

- ✅ **不能把未移动误判成已移动**：122 次操作中错误位移 = 0。
- ✅ **不能通过隐藏 RAM 生成导航结果**：所有估计只来自截图；脚本中 RAM 仅用于统计对照
  （且该脚本不属于 `llm_agent.navigation`，边界测试强制隔离）。
- ✅ 单步/转身/撞墙/多格/菜单中断/门口传送全部正确分类。
- ✅ 边界与动画噪声（室外 NPC/水面）不导致坐标漂移（室外 30 步中 13 次真实位移全部正确）。

> 修误差的方式是修改定位算法，而不是放宽测试：开发过程中发现并修复了两个真实问题：
> (1) 用 `FrameView.stable` 默认 True 提前判定转场结束；(2) 室外动画帧的 MOVED 分数低于
> 0.965 被误判 UNKNOWN —— 后者改为"score≥0.93 且相对零位移假设 margin≥0.22"的宽松通道，
> 并加入黑屏/转换帧优先判定。修复后重跑 G1 全部通过。

---

## G2 — 陌生房间自主探索（真实游戏，Oak's Lab）

脚本：`scripts/nav_v2_g2_explore_room.py`；结果：`artifacts/nav_v2/g2/report.json` +
`summary.md`；截图 27 张、地图快照、`map_gallery/`（每区域地图+世界图）、
`navigation_steps.jsonl`（逐动作审计）。

运行方式：从合法存档 `lab_start.state` 开始（世界模型目录全新，**无任何预制地图/攻略**）。
中途未知门口触发了一次真实传送：脚本等待屏幕冻结后按反方向走回，`agent.attach()`
**视觉重定位成功（score=0.9101）** 恢复原区域与坐标，然后继续探索。

| 指标 | 数值 |
|---|---|
| 探索轮次 / 方向操作 / 真实位移 | 2 / **268** / **173** |
| 新发现格子 | **263** |
| 本地网格总格数 | **353** |
| 已验证可行走格 | **79** |
| 已确认障碍格 / 不同障碍点 | **56 / 56** |
| 发现出口 EXIT | **1**（门口传送已证实；第二格门口被延迟为"门口相邻未知"） |
| 未知可达边界 remaining | **2**（门口相邻，按策略延迟） |
| 覆盖率 coverage_ratio | **97.53%** |
| 同格同方向重复撞墙（同一障碍重复尝试） | 37 次**转向**（TURNED_ONLY，非盲目重复撞同一面墙）；BLOCKED 0 次重复目标 |
| 区域切换 / 重进入重定位成功 | 1 / **1（score 0.9101）** |
| 视觉位置漂移 vs RAM 真值 | **0 格** |
| 结束原因 | `only exit frontiers remain (exits=1), area covered; use navigate_to({'exit': true})` |
| 结论 | **探索完成（房间覆盖）**，通过 |

诚实说明：探索策略在"只剩出口类 frontier"时**不会跨出门**，而是报告完成并提示用
`navigate_to({"exit": true})` 换场景；这符合"探索房间"与"离开房间"职责分离。剩余
2 个未知可达边界是门口相邻格（doorway contiguous），被有意延迟；因此报告
coverage_ratio=97.53% 而不是 100%。

---

## G3 — 重复进入已探索区域（永久地图 + 重定位）

脚本：`scripts/nav_v2_g3_reentry.py`；结果：`artifacts/nav_v2/g3/report.json` +
`summary.md`；地图 3 张 + 截图 + `map_gallery/`。

流程：A) 自主探索 Oak's Lab 至完成；B) `navigate_to({"exit": true})` 走出房间（真实门口传送）；
C) 反方向走回门口，`agent.attach()` 仅凭截图重定位；D) 用历史地图 `navigate_to({cell})`
走到房间内最远已知道路格。

| 指标 | 数值 |
|---|---|
| 首次自主探索 | 268 步（含 1 次门口传送往返） |
| 首次探索结果 | 完成，79 可行走格，353 已知格 |
| 离开房间 | **成功**（`navigate_to exit`，区域切换 area_001→area_002） |
| 重新进入识别（重定位） | **成功**：`matched=True, area_id=area_001, pose=(0,8), score=0.9213` |
| 新建重复区域数 | **0**（未因弱证据错误合并/重建） |
| 恢复历史地图 | 是（继续使用 area_001 的 353 格历史地图） |
| 到达最远已知目标 | **成功**（`reached=True`） |
| 复用地图导航步数 | **14 步**（首次探索 268 步，比例 **5.2%**） |
| 复用期间新增格 / 撞墙数 | **0 / 0** |
| 定位漂移 vs RAM 真值 | **0 格** |
| 结论 | **通过**：能识别旧区域、恢复历史地图/坐标系、直接复用已知道路，且无无效往返 |

---

## G4 — 有限预算真实无人值守长跑（多区域，纯导航）

脚本：`scripts/nav_v2_g4_long_explore.py`；结果：`artifacts/nav_v2/g4/report.json` +
`summary.md`；426 条逐动作日志（`navigation_steps.jsonl`）、27 张截图、地图快照、
`map_gallery/`（4 个区域）、世界图、截图拼图。

运行：从 `lab_start.state` 开始，纯 Python 导航循环**无人工逐步控制、无 LLM**，预算
3000 导航步 / 1200 秒。房间覆盖后主动走向已知出口进入下一区域继续探索。

| 指标 | 数值 |
|---|---|
| 总方向操作 / 真实位移 | **426 / 259**（有效位移比例 62.4%） |
| 新发现格子 | **1164**（4 个区域） |
| 已知世界总格数 | **1254**（353 + 157 + 654 + 90） |
| 区域数 / 连接数 | **4 / 2** |
| 区域切换 / 重定位成功 | 3 / **2（66.7%）**（另一次为首次进入全新区域，按设计新建区域） |
| 区域覆盖：area_001 / 002 / 003 / 004 | 97.53% / 36.36% / 40.52% / 20.00%（后三者为室外大地图/新进入区域，未跑完预算） |
| 撞墙/受阻尝试 | 84 次 blocked + 50 次 turned_only |
| UNKNOWN 步数 / 比例 | 21 / **7.9%**（全部未导致坐标错误；连续 2 次 UNKNOWN 会重定位或暂停） |
| 重复访问率（同区域同格再次到达/位移步） | 46.2% |
| 路径规划失败 | **0** 次 |
| 视觉位置漂移 vs RAM 真值（可验证区段） | **0 格**（3 个可验证区段全部 0；转场区段 RAM 坐标系不同，不计入） |
| 结束原因 | **真实战斗事件中断**：`battle_event`（"battle on screen; navigation paused"），导航诚实暂停并返回结构事件 |
| DeepSeek API 调用/费用 | **0 次 / $0**（本脚本是纯导航长跑，用于验证"导航不再每步问 LLM"） |
| 结论 | **通过**：4 区域、1164 新格、426 步真实操作、0 漂移、0 规划失败，遇战斗正确暂停 |

失败/限制（如实）：area_002/003/004 是 Pallet Town 及相邻新区域，尚未在本次预算内探索完成；
其中一次转场由未知门口触发；G4 内 UNKNOWN 7.9% 主要来自室外动画/战斗转场帧。

---

## G4-API — 真实 DeepSeek API 与 Navigation v2 集成运行

脚本/配置：`scripts/run_llm_agent.py --config configs/navigation_v2_api.json`；
结果目录：`artifacts/nav_v2/api_run/`（`report.md`、`audit/decisions.jsonl`、
`audit/actions.jsonl`、`audit/navigation.jsonl`、`navigation/world.json`、`screens/`、checkpoints）。

| 指标 | 数值 |
|---|---|
| 模式 | 真实 `deepseek-flash` API（非 scripted；`report.md` 标注 real_api=true） |
| API 调用 / 轮次 / 耗时 | **8 次 / 9 轮 / 52.6s** |
| 估算费用 | **$0.003082**（预算上限 $0.60 远未触顶；真实账单以平台为准） |
| Token | 52,461（含缓存命中；单次平均约 5.8s 延迟） |
| DeepSeek 调用的高层工具 | `explore_current_area` ×4、`get_navigation_status` ×2、`get_local_map` ×2、`observe_game` ×1、`walk` ×1 |
| Python 执行的已验证导航步 | **159 步**（方向操作） |
| 区域数 / 连接数 / 已建地图格 | **2 / 2 / 342 格（54 可行走）** |
| 每 API 调用驱动导航步数 | **≈20 步/调用**（旧版盲目探索是每个决策只走 1–6 步） |
| 真实战斗/事件处理 | 模型按工具返回结构化状态决策；导航遇事件暂停 |
| 结论 | **通过**：真实 API + 真实模拟器 + Navigation v2 工具链路端到端可用，成本极低 |

> 说明：该 run 为 8 轮集成演示，未追求地图覆盖率；其 `navigation/world.json` 与
> `audit/navigation.jsonl` 是"DeepSeek 决定探索什么、Python 决定怎么走"的真实证据。

---

## 单元 / 集成测试

| 套件 | 结果 |
|---|---|
| `pytest tests\navigation -q` | **319 passed**（2.7s，无需 ROM/网络/API key） |
| `pytest tests\llm_agent -q` | **357 passed**（4.6s；含真实 ROM 的 scripted dry-run 端到端） |
| `pytest tests\phase1 -q` | **11 passed**（8.5s，既有环境/ROM 验证未受影响） |
| 合计 | **687 passed, 0 failed** |

覆盖要点（节选）：

- localization：四方向 MOVED、转身、同向撞墙、2 格移动、NPC 动画、UNKNOWN 不更新坐标、
  连续 UNKNOWN 使 pose 失效、对话中断、黑屏/区域切换、重置/set_pose。
- grid_map：观察≠验证、撞墙不墙化已走格、TEMP_BLOCKED、EXIT 标记、frontier、覆盖率数学、
  ASCII、JSON roundtrip、损坏容忍。
- world_model：tile hash 滑窗重定位、重叠感知排序（防"薄候选假高分"）、margin 接受规则、
  歧义拒绝、连接记录、多区域持久化。
- pathfinder/explorer：A* 最优性（随机网格 vs Dijkstra）、成本模型、无路原因、frontier 选择、
  防振荡、完成判定、attempt 历史。
- controller/navigation_agent：MOVED/BLOCKED/TURNED_ONLY/UNKNOWN 的 exactly-once 地图更新、
  转场像素级冻结等待 + 重定位/新建区域、事件暂停、预算、localization lost、持久化、
  legacy raw outcome 去重同步。
- tools/runtime 集成：7 个导航工具 schema/校验/结果、navigate_to 目标形式转换、
  pause/resume、未启用时错误路径、真实运行时（真实 ROM）探索→world model 持久化→
  prompt 注入 nav 状态→report 生成。

## 隔离性与真实性声明

1. **纯视觉**：导航决策链路仅输入截图（`FrameView` 由 framebuffer 生成）；无 RAM/地图 ID/剧情 flag。
   `test_import_boundaries.py` 检查 `llm_agent/navigation/*` 不导入 PyBoy/progress、无 RAM 地址、
   无剧情知识。
2. **RAM 只在验收脚本**：G1–G4 的漂移统计来自 `llm_agent.progress.OverworldProbe`（VERIFICATION-ONLY），
   用于独立校验，不进入导航决策。
3. **无攻略/预制地图**：世界模型只由 agent 自己的截图与验证移动构建；仓库无任何地图数据文件。
4. **不修改游戏**：只加载合法 PyBoy save state；不写 ROM/游戏存档；API key 不打印、不入库；
   `.env`/`artifacts`/`work`/ROM 均 gitignored。
5. **失败如实记录**：G4 area_002/003/004 未覆盖完（预算终止）、UNKNOWN 7.9%、重定位 2/3
   （另一次为全新区域）等均在本报告写明，未把 UNKNOWN 或未完成标为成功。
