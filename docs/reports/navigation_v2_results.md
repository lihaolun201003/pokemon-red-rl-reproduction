# Pokémon Red Navigation v2 — 最终结果报告

日期：2026-10-18
设计：[navigation_v2_design.md](navigation_v2_design.md) · 验收：[navigation_v2_validation.md](navigation_v2_validation.md)
GitHub：https://github.com/lihaolun201003/pokemon-red-rl-reproduction

## 1. 一页结论（只写真实做到的）

| 问题 | 结论 | 证据 |
|---|---|---|
| 原来的"撞墙 / 地图漂移"问题是否解决 | **是**：122 次真实方向操作错误位移 0、漏检 0、漂移 0 格；G2/G3/G4 可验证区段漂移 0 格 | G1/G2/G3/G4 report.json |
| 能否纯视觉定位（无 RAM） | **是**：只输入 160×144 截图；RAM 仅用于验收脚本对照 | `tests/llm_agent/test_import_boundaries.py`、G1 |
| 能否自主探索陌生房间 | **是**：Oak's Lab 从零建图 353 格、可行走 79、障碍 56、覆盖率 **97.53%**，自行判定完成 | G2 |
| 实际发现多少格地图 | 单房间 **353 格**；G4 多区域长跑 **1164 个新格 / 4 个区域 / 1254 总已知格** | G2、G4 |
| 能否稳定绕开障碍 | **是**：已确认障碍写入地图并在 A* 中避开；G2 同一障碍同方向盲目重复 0，G4 规划失败 0 次 | G2/G4 |
| 能否返回已探索地点 | **是**：离开→再进入识别成功（score 0.9213），恢复历史地图与坐标系，**0 重复区域**，再用 14 步到达最远已知点（首次探索 268 步的 5.2%） | G3 |
| 能否连续无人值守探索多久 | G4 纯导航：**426 个真实方向操作 / 259 次位移 / 4 个区域**，约 50 秒游戏时间；遇**真实战斗事件**诚实暂停（非崩溃） | G4 |
| 导航是否还每步问 DeepSeek | **否**：G4-API 中 DeepSeek 仅 **8 次调用**驱动 **159 步**已验证导航（≈20 步/调用），费用 **$0.0031** | G4-API |
| 是否完成真实房间探索 | **是**（Oak's Lab，G2） | G2 |
| 是否通关游戏 | **否**，本阶段不做剧情/道馆/战斗策略 | 本报告 §6 |

## 2. 本次修改了哪些代码

新增 Navigation v2 子系统（`llm_agent/navigation/`，11 个模块 + 冻结契约）：

| 文件 | 作用 |
|---|---|
| `types.py` | 冻结数据契约（MotionResult/CellState/EdgeState/FrameView/ExploreReport…） |
| `vision.py` | 纯像素/tile 基元：16px 网格、tile hash、位移匹配评分、diff 证据、黑屏检测 |
| `localization.py` | **纯视觉位移定位**（MOVED/BLOCKED/TURNED_ONLY/UNKNOWN/TRANSITION/INTERRUPTED + 置信度） |
| `grid_map.py` | 单区域逻辑网格（cell/edge 证据、可通行验证、错误修正、诚实覆盖率、JSON 持久化） |
| `world_model.py` | 多区域世界图、区域连接、**视觉重定位**（重叠感知排序 + margin 接受规则） |
| `pathfinder.py` | 确定性 BFS/A*（成本模型、无路原因、扩展上限） |
| `explorer.py` | Frontier-Based 探索（评分、防振荡、出口策略、完成判定） |
| `controller.py` | 验证式导航循环（每次按方向恰好一次地图更新、重规划、转场冻结等待、事件暂停） |
| `navigation_agent.py` | 持久化 facade（attach/explore/navigate/status/local_map/world_summary/pause/save/外部动作同步） |
| `viz.py` | 地图/世界/状态/composite 渲染（numpy+PIL） |
| `NAV_CONTRACT.md` | 模块接口契约（本阶段冻结） |

修改现有模块（约 +1130 行 / −72 行）：

- `llm_agent/tools.py`：新增 7 个高层导航工具（schema+严格校验+处理），并让 legacy
  `press/execute_actions/walk` 的移动结果通过回调**恰好一次**并入地图；
- `llm_agent/runtime.py`：创建导航子系统、状态/事件/审计接线、检查点保存与恢复时重定位、
  提示词注入导航状态与 ASCII 地图、实时地图面板数据、报告/指标加入导航统计；
- `llm_agent/prompts.py`：系统提示词新增 NAVIGATION 段（DeepSeek 决定探索什么，Python 决定怎么走），
  用户消息注入导航状态块；
- `llm_agent/config.py`：`NavigationConfig`（预算/持久化/可视化/上限）+ 校验 + env/JSON 覆盖；
- `llm_agent/viewer.py`：Tk 观战窗口新增实时地图面板（线程安全、节流、不阻塞导航）；
- `llm_agent/contracts.py`：工具名与导航工具列表；
- `llm_agent/agent.py`：透传 `nav_status_text`；
- `llm_agent/progress.py`：新增 `OverworldProbe`（验证专用，含真实 map/x/y/facing）；
- `tests/llm_agent/*`：更新 3 个既有测试以匹配新工具/导航持久化/边界检查。

新增测试与脚本：

- `tests/navigation/`：**319 个测试**（定位/网格/世界/寻路/探索/控制器/可视化/集成，均无需 ROM/API）；
- `scripts/nav_v2_common.py`、`nav_v2_g1_localization.py`、`nav_v2_g2_explore_room.py`、
  `nav_v2_g3_reentry.py`、`nav_v2_g4_long_explore.py`、`nav_v2_map_gallery.py`；
- `configs/navigation_v2_api.json`（真实 API 集成的可复现配置）。

未改动：`neroRL/`、`reproduction/`、`scripts/train.py`、`configs/train_*.yaml`、
checkpoint、PPO 评估脚本等训练系统；ROM/存档/API Key/录像/权重均未提交。

## 3. 原问题的修复（对应任务书第三节）

### 3.1 错误的位移判断（原来用像素变化比例）

**已重构为有置信度的视觉定位系统**，不再以单一变化率作为最终依据：

- 成功一步 = 背景 16px 整格平移（相机跟随玩家于屏幕 tile (4,4) 固定），用 tile 对齐的
  位移匹配（含第二名候选/margin）判定；支持 0/1/2 格；转身 = 仅玩家 sprite 框内变化；
  同向撞墙 = 像素完全相同；黑屏/对话/菜单/战斗优先判为 TRANSITION/INTERRUPTED。
- 置信度不足时返回 `UNKNOWN` 且**不更新坐标**；连续 UNKNOWN 触发短观测/重定位/安全暂停。
- 保留朝向来区分 TURNED_ONLY 与 BLOCKED。
- 真实验收：122 次方向操作，错误位移 0、漏检 0、UNKNOWN 0、漂移 0（G1）。

### 3.2 心理地图更新遗漏 / 重复处理

**已消除**：`ToolExecutor` 每次执行 raw movement 工具通过 `on_action_outcome` 回调
**恰好一次**把结果交给导航系统（含 execution 中断场景）；导航自身 `move_once` 也是
"一次按压 → 一次 Localize → 一次地图更新"。运行时的旧 `executor.last_outcome` 轮询已移除。
控制器/agent 测试专门验证 exactly-once（fake grid 计数 + 外部 outcome 去重）。

### 3.3 错误的区域切换判断

**已重构**：区域切换只在以下情况成立：画面黑屏/淡入淡出（TRANSITION）、门口传送，
或"屏幕像素级冻结后重定位匹配/创建新区域"。绝不再用"画面变化 ≥55%"认定新地图。
转场处理器要求连续 8 帧块像素差 ≤2% 且背景位移为 0（2 次连续 + 8 帧确认），并显式拒绝
从淡入淡出中途帧创建区域。实测：G1 中 4/4 真实传送正确、0 误判；G2 入口传送后
视觉重定位 score 0.9101 恢复原地图。

## 4. 现在的实际能力（真实运行数据）

### 4.1 单房间自主探索（G2）

从合法存档进入 Oak's Lab（世界模型为空）：2 轮、268 个方向操作、173 次真实位移，
建出 **353 格**地图（**79 可行走、56 障碍、1 出口**），覆盖率 **97.53%**，
期间未知门口触发 1 次传送并成功走回+重定位（score 0.9101），最终自行判定
"only exit frontiers remain, area covered" 并停止。全程 0 漂移、0 坐标错误。

### 4.2 离开并返回（G3，永久记忆）

- 探索完成（268 步）→ `navigate_to({"exit": true})` 成功走出房间（真实传送）；
- 反方向走回门口 → `attach()` 仅凭截图识别旧区域：**matched=True, area=area_001,
  pose=(0,8), score=0.9213**，**0 个重复区域**；
- 用历史地图导航到房间最远已知格：14 步到达，0 新格、0 撞墙、0 规划失败，
  是首次探索步数的 **5.2%**。

### 4.3 有限预算多区域长跑（G4）

纯 Python 导航、无 LLM、无人值守：**426 个真实方向操作、259 次位移、4 个区域、
1164 个新格（总已知 1254 格）**，跨区域连接 2 条，规划失败 **0** 次，可验证区段漂移 **0**。
探索阶段结束后主动走向已知出口进入下一区域继续；最终遇到**真实战斗**时导航按设计
**暂停并返回 `battle_event`**（不是崩溃、不是乱按）。UNKNOWN 率 7.9%（室外动画/转场帧），
均未导致坐标准确性错误。

### 4.4 真实 DeepSeek API 集成（G4-API）

真实 `deepseek-flash` + 真实模拟器：8 次 API 调用（**$0.0031**，52.6s）驱动 **159 步**
已验证导航、2 个区域、342 格地图。DeepSeek 实际调用了 `explore_current_area`、
`get_navigation_status`、`get_local_map`、`observe_game`、`walk`，说明"高层目标 → Python 导航"
的职责分离在真实 API 下可用。平均 **≈20 个导航步/API 调用**（旧模式每决策只走 1–6 步且
方向由模型猜）。

## 5. 稳定性与安全性

- **不漂移**：导航只接受高置信视觉位移；UNKNOWN 不更新坐标；转场必重定位；检查点恢复后
  `attach()` 强制视觉重定位，不盲信旧坐标。
- **不死循环**：探索/导航有步数、时间、无进展、连续 UNKNOWN、连续 BLOCKED 上限；
  防振荡（失败目标惩罚/上限）；地图有最大格数/区域数上限。
- **不误穿事件**：对话框/菜单/战斗先暂停并上报结构化事件，绝不"盲目按 A 推剧情"。
- **预算**：探索/导航是本地 CPU 计算，不花 API 费用；只有高层决策花费 API；API 预算/次数
  机制沿用原有实现（G4-API 实测 $0.003 < $0.60 上限）。
- **可审计**：`audit/navigation.jsonl`（逐导航步：方向/结果/坐标/置信度/证据）、
  `navigation/world.json` + `areas/*.json`（永久地图）、`screens/`（截图）、
  `map_gallery/`（每区域地图/世界图/状态）。

## 6. 尚未完成 / 已知限制（如实）

1. **本阶段不做剧情**：没有 NPC 对话理解与剧情推进、没有道馆/任务攻略、没有战斗策略，
   因此**没有通关**；战斗/对话只做到"暂停 + 结构化上报"，恢复由上层处理。
2. **室外大地图未跑完**：G4 的 area_002/003/004（Pallet Town 及相邻新区域）覆盖率
   36%/40%/20%，是本次预算（3000 步）用完/遇战斗中断所致，不是"探索完成"。
3. **室外动画降低定位置信度**：NPC 走动/水面动画使 UNKNOWN 率 ~7.9%；没有造成坐标错误，
   但会降低有效探索速度。
4. **未知门口首访可能触发一次传送**：门口在被确认 EXIT 前是普通 frontier；G2 中发生 1 次，
   走回并重定位成功；之后门口相邻未知格会被延迟。
5. **弱证据不合并**：`attach()` 匹配不足时不猜测，宁可新建区域（可能重复建区）；这是
   安全优先的设计（宁可多一张地图，不可错误合并导致坐标全错）。
6. **Tk 实时地图面板**已实现并接线，但本机验收以 headless 运行 + `viz` 离线渲染证据
   （`artifacts/nav_v2/evidence/*.png`）为准；真实窗口交互（暂停/保存按钮）未在自动化
   测试中截图验证。
7. G4-API 只是 8 轮集成演示，不是长时真实 API 探索；长时 API 会受 5–6 秒/调用的延迟限制。

## 7. 证据文件位置（本地，gitignored）

- G1：`artifacts/nav_v2/g1/report.json`、`ops.jsonl`、`summary.md`、截图/转场 PNG
- G2：`artifacts/nav_v2/g2/{report.json,summary.md,screens/,maps/,map_gallery/}`
- G3：`artifacts/nav_v2/g3/{report.json,summary.md,screens/,maps/,map_gallery/}`
- G4：`artifacts/nav_v2/g4/{report.json,summary.md,screens/,maps/,map_gallery/,navigation_steps.jsonl}`
- G4-API：`artifacts/nav_v2/api_run/{report.md,metrics.json,audit/*.jsonl,navigation/,screens/}`
- 地图/世界/可视化证据：`artifacts/nav_v2/evidence/`（各 run 的 local map、world graph、
  composite viewer 截图）
- 单元/集成测试：`.\.venv\Scripts\python.exe -m pytest tests\navigation tests\llm_agent -q`
  → **676 passed**（本次结果：navigation 319 + llm_agent 357）

## 8. 下一步建议

1. 在导航之上实现 **Story Agent**（利用已有 `NavEvent` 事件接口 + `navigate_to(landmark/exit/area)`），
   处理对话/战斗后 `resume()` 继续探索，逐步推进真实剧情。
2. 室外探索策略增强：给"不可达/延迟"区域明确原因并尝试绕行；对动态 NPC 用 TEMP_BLOCKED
   重试策略；提高长距离室外探索的覆盖率。
3. 真实 API 的长时运行（例如 30–60 分钟、$0.5 预算）以验证真实 LLM 闭环下的探索效率。
4. 用同一套 Navigation 状态驱动 PPO/环境接口（保留训练系统不变的前提下，提供可选观测）。
