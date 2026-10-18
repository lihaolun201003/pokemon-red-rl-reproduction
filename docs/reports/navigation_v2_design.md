# Pokémon Red Navigation v2 — 纯视觉自主探索与永久地图记忆：设计文档

日期：2026-10-18
项目：`C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`
范围：Navigation / Exploration（不含剧情、道馆、战斗策略、PPO 重训）

## 1. 目标与本阶段边界

### 1.1 唯一核心目标

让 DeepSeek Agent 能够**通过真实游戏截图自主探索未知区域**，准确记录走过的道路和障碍，并在
再次进入已探索区域时**直接使用已有地图导航**。本阶段只交付导航/探索能力，不追求通关。

### 1.2 明确不做（但保留接口）

- NPC 对话理解、剧情推进（Story Agent 后续实现，接口见 §11）。
- 道馆挑战与任务攻略。
- 宝可梦战斗策略（导航遇到战斗会暂停并返回结构化事件，交给上层）。
- PPO 重新训练；`neroRL/`、`reproduction/`、`scripts/train.py`、checkpoint 等训练系统
  完全未改动，原有测试保持通过。
- 完整游戏通关逻辑。

### 1.3 硬性规则（贯穿全部实现）

1. 导航决策只允许使用 **160×144 framebuffer** 与自身记录。
2. **禁止读 RAM / 隐藏地图 ID / 坐标 / 剧情 flag** 作为导航依据。RAM 只出现在
   `llm_agent/progress.py`（验证专用）与 `tests/`、`scripts/nav_v2_*` 验收脚本中，
   且由测试 `test_import_boundaries.py` 强制隔离。
3. 不读取外部攻略/预制地图，不修改 ROM、存档、游戏内存（只使用合法 PyBoy save state）。
4. 导航过程不逐步请求 DeepSeek；Python 低层自主移动，DeepSeek 只决定"探索什么/去哪"。
5. 无法确认位移时不得更新坐标（返回 `UNKNOWN` 并触发短观测/重定位）。

## 2. 核心视觉事实（在真实 ROM 上实测，作为算法基础）

G1 验收与手工实验（`work/nav_v2/exp*`，仅测试脚本，不入库）确认了本 ROM 的渲染规律：

| 观察 | 结论 |
|---|---|
| 屏幕 160×144 = 10×9 个 16×16 tile，滚动位移恒为 16px 的整数倍 | 16px 网格相位固定，可直接做 tile 对齐的位移匹配 |
| 成功走一步时，整幅背景平移 16px；玩家 sprite 始终停留在屏幕 tile **(4,4)** | 相机跟随、玩家居中：**位移 = 背景平移的相反数**，不需要追踪角色在屏幕上的位置 |
| 被挡住且朝向不变时，两帧像素**完全相同**（实测 diff=0） | `BLOCKED` 可用零位移 + 零像素变化高置信判定 |
| 被挡住但转向时，变化像素**全部位于玩家 16×16 sprite 框内**（实测 20–25 px） | `TURNED_ONLY` 可判定 |
| 一次按住 8 帧 = 1 格；按住 32 帧 = 2 格（多格移动同样是 16px 整数倍） | 支持 0/1/2… 格位移估计 |
| 对话/菜单/战斗/淡入淡出/门口传送会产生大面积像素变化 | 必须用屏幕上下文（对话框/菜单/战斗/黑屏）判定，**禁止用单一变化率判定新地图** |
| 室外（Pallet Town）有 NPC 走动/水面动画，单帧匹配分数会从 ~0.99 降到 ~0.93–0.98 | 阈值需带"相对零位移假设的 margin"，并允许低置信 MOVED |
| 入口/出口传送（door warp）固定为 2 格宽的门口 tile | 转场需等待屏幕冻结后重定位；EXIT 需要专门标记与"探索时不乱穿门"策略 |

## 3. 总体架构

```
DeepSeek Brain (deepseek-flash, vision + tool calls)
   │  只做高层决策：探索当前区域 / 去哪个已知目标 / 处理事件
   ▼
Tool layer (llm_agent/tools.py)
   │  explore_current_area / navigate_to / get_navigation_status / get_local_map /
   │  get_world_map_summary / pause_exploration / resume_exploration
   │  另有 legacy raw tools（press/execute_actions/walk）用于菜单/对话/战斗
   ▼
NavigationAgent (navigation/navigation_agent.py)   ← 持久化 facade / 会话入口
   ▼
NavigationController (navigation/controller.py)     ← 唯一按方向键的导航循环
   ├── Localizer      (localization.py)   纯视觉位移估计 + pose/置信度
   ├── GridMap        (grid_map.py)       单区域逻辑网格（cell/edge/证据/覆盖）
   ├── WorldModel     (world_model.py)    多区域世界图 + 视觉重定位
   ├── Pathfinder     (pathfinder.py)     A* / BFS（成本模型）
   ├── Explorer       (explorer.py)       Frontier-Based 探索策略
   └── vision.py                          像素/tile 基元（shift 匹配、diff、hash）
   ▼
GameBoyEmulator (llm_agent/emulator.py)  PyBoy（唯一允许触碰模拟器的模块之一）
   +  Viewer (llm_agent/viewer.py)       Tk 游戏窗口 + 实时地图面板
   +  viz.py                             map/world/status/composite 渲染
```

职责分离（任务书要求）：

- **DeepSeek 决定探索什么**：调用 `explore_current_area()` 或 `navigate_to(...)`。
- **Navigation 决定怎么过去**：Python 内完成视觉定位、BFS/A*、逐步执行、失败重规划。
- **Story/事件**：导航遇到战斗/对话/菜单/转场时立即暂停，返回结构化
  `BATTLE_EVENT / DIALOG_EVENT / MENU_EVENT / TRANSITION_EVENT`，上层处理后再恢复。

## 4. 模块与冻结接口

接口契约见 `llm_agent/navigation/NAV_CONTRACT.md` 与 `llm_agent/navigation/types.py`
（本阶段冻结；实现全部对齐）。

### 4.1 `types.py`（共享数据契约）

枚举：`MotionResult{MOVED,BLOCKED,TURNED_ONLY,UNKNOWN,TRANSITION,INTERRUPTED}`、
`FrameContext{OVERWORLD,DIALOG,MENU,BATTLE,TRANSITION,BLACK,UNKNOWN}`、
`CellState{UNKNOWN,OBSERVED,WALKABLE,BLOCKED,TEMP_BLOCKED,EXIT,INTERACTIVE,UNREACHABLE,DEFERRED}`、
`EdgeState{UNKNOWN,OPEN,BLOCKED,TEMP_BLOCKED,EXIT}`、`ExploreStatus`、`NavEventKind`。

数据：`TileObs`、`FrameView`、`PlayerPose`、`MotionEstimate`、`PathPlan`、
`ExploreTarget`、`AreaMatch`、`NavEvent`、`CoverageStats`、`ExploreReport`、
`NavReport`、`NavStep`。坐标约定：屏幕/区域坐标 y 向下；`up` 使 y-1；
`delta = -camera_shift`；玩家所在屏幕 tile 固定为 `VIEW_PLAYER=(4,4)`。

### 4.2 `vision.py`（像素基元，无 PyBoy）

`frame_luma`、`tile_grid_offset`、`extract_tiles`（90 个 16×16 tile hash）、
`match_shift`/`best_shift`（候选位移 ±1/±2 格的匹配率与第二名分数）、`diff_mask`、
`diff_stats`（变化 bbox、玩家框内外像素、连通域数）、`changed_tiles`、
`screen_signature`、`same_screen`、`is_black`。

### 4.3 `localization.py`（纯视觉定位，本阶段核心）

- 输入：`prev Frame` / `cur Frame` + 请求方向；输出 `MotionEstimate`（含证据字典）。
- 分类决策（严格按顺序，禁止只用像素变化率）：
  1. 黑屏帧（淡入淡出/战斗转场）→ `TRANSITION`。
  2. 当前屏幕上下文为 DIALOG/MENU/BATTLE（像素启发式）→ `INTERRUPTED`。
  3. 计算 `best_shift`。若最佳位移与请求方向一致且
     `score ≥ 0.965`，或 `score ≥ 0.93` 且相对零位移假设 margin ≥ 0.22（动画噪声下的
     宽松通道），且不存在同分歧义 → `MOVED(delta = -shift, tiles=k)`。
  4. 无位移且像素完全相同 → `BLOCKED`；变化全在玩家 sprite 框内 → `TURNED_ONLY`
     （两者都给出"该方向受阻"证据）；有 NPC/动画噪声但无位移 → `BLOCKED`（低置信）。
  5. 无位移且出现大幅无法解释变化（≥阈值）→ `TRANSITION`；其余 → `UNKNOWN`（**不改坐标**，
     置信度衰减，连续 2 次 UNKNOWN 使 pose 失效，触发重定位/暂停）。
- 设计要点：一次输入只更新一次坐标；多格移动支持；朝向跟踪；证据全部可审计。

### 4.4 `grid_map.py`（单区域逻辑网格）

每个区域独立二维网格。每个 cell 记录：局部坐标、`CellState`、terrain（主 tile hash）、
`tile_hashes`（动画/多变体）、observed/visit 计数、最近观察帧、障碍/通行证据、置信度、
标签。每条相邻边记录方向、`EdgeState`、尝试/成功次数、置信度。

关键语义（防止"一次撞到 NPC 就被永久当墙"）：

- `OBSERVED`（看见）≠ `WALKABLE`（实测走过）；只有**已验证的移动**才把 cell 标为 WALKABLE。
- `confirm_blocked(temporary)`：一次撞墙 → BLOCKED；疑似 NPC/动态障碍用 TEMP_BLOCKED，
  之后可重试；**曾走过的 cell 不会被一次碰撞永久墙化**（edge 记为 TEMP_BLOCKED）。
- `confirm_turn` 与"撞墙"分离（Gen-1 首次按方向可能只是转身）。
- 支持错误修正：重复观察、证据计数与置信度更新、`UNREACHABLE/DEFERRED` 标记。
- 覆盖统计诚实：`coverage_ratio = walkable / (walkable + reachable unknown_frontier)`；
  unreachable/deferred 单独统计，绝不虚报"房间已探索完"。
- 持久化：JSON（versioned、原子写、容忍损坏），布局 `<root>/areas/<area_id>.json`。

### 4.5 `world_model.py`（世界图 + 重定位）

- `AreaRecord`（区域 id/kind/网格/访问次数/entry cells/是否已探索/备注）。
- `WorldConnection`（from_area/cell/direction → to_area/to_pose/kind=warp|walk/置信度/次数）。
- 视觉重定位 `match_view`：把当前 10×9 可见 tile hash 窗口在区域网格上滑动，
  `score = matched/comparable`（忽略玩家自身 tile 与无 hash 的未知 cell）。
  采用**重叠感知排序**：解释屏幕面积更大的候选优先，避免 30/30 均匀地板"假高分"压过
  82/89 的真实匹配；margin 只与同样"实质重叠"的候选比较。
- 接受规则：`score ≥ 0.55` 且 `margin ≥ 0.08`，或与刚记录的连接/预期区域一致；
  否则拒绝（**弱证据绝不合并区域**，宁可新建区域并保留待确认连接）。
- 持久化：`<root>/world.json` + 每区域网格 JSON；支持多区域 save/load。

### 4.6 `pathfinder.py`（BFS/A*）

成本：walkable=1、observed=3、temp_blocked=5（可选）、unknown=6（保守）、exit=2、
blocked/unreachable/deferred=∞。确定性 A*（Manhattan 启发式、插入顺序 tie-break、
60k 扩展上限）；`plan`/`plan_to_nearest`/`reachable`/`direction_between`/`plan_from_cells`；
无路返回 `PathPlan(reachable=False, reason=...)`，不抛异常。

### 4.7 `explorer.py`（Frontier-Based 探索）

- frontier = 与"已验证可走区域"相邻的 OBSERVED/UNKNOWN/TEMP_BLOCKED/EXIT cell；
  EXIT 单独作为目标类型。
- 评分 = 新格期望 ×1.6 + 出口奖励 − 路径成本 ×0.35 − 重复失败惩罚 − 未知探测风险。
- 防振荡：`max_attempts_per_cell`、重复惩罚、选择上限、`max_target_path`、no-progress 计数。
- 完成判定：不再存在"可达且值得探测"的 frontier 时返回完成，并给出统计原因
  （blocked/unreachable/deferred/path-too-long 等分别计数），如实报告不可达区域。
- 房间探索时出口策略：有非出口 frontier 时优先它们；只剩出口/门口相邻未知格时
  `COMPLETED` 并提示 "use navigate_to({'exit': true})"，不擅自穿门。
- 不允许无意义往返：已确认道路可用于最短路径，但不会把同一失败目标反复当目标。

### 4.8 `controller.py`（验证式导航循环，唯一按方向键处）

- `move_once(direction)`：一次按方向 → 新截图 → `Localizer.update` → **恰好一次**地图更新：
  MOVED→`confirm_move`；BLOCKED→`confirm_blocked`；TURNED_ONLY→`confirm_turn`+受阻证据；
  UNKNOWN→零更新并计数；INTERRUPTED/TRANSITION→零更新并返回事件。
- `follow_path`：逐格执行，撞墙 → 已知该边 → 重规划一次；状态机返回
  ok/blocked/event/transition/localization_lost/step_limit。
- 转场 `handle_transition`：等待屏幕**像素级冻结**（连续 8 帧块像素差 ≤ 0.02 且背景位移=0，
  2 次连续 + 8 帧确认；不信任 signature 相等，避免 NPC/水面动画误判），然后重定位；
  匹配 → 切换区域、恢复 pose、记录连接；不匹配 → 新建区域 pose(0,0)（弱证据不合并）；
  超时（黑屏/未冻结）→ 不建区域、pose 失效、返回结构化事件。
- `explore_current_area` / `navigate_to`：预算（步数/时间/无进展）、暂停/恢复、事件中断、
  定位丢失恢复；`last_path`/`last_target` 供可视化。
- `NavigationAgent`：持久化 facade，`attach()`（载入状态/回到已知区域时视觉重定位）、
  `observe_external_outcome`（把 LLM 的 legacy raw 移动结果以 exactly-once 方式并入地图，
  用 outcome/frame 去重）、`save()` 等。

### 4.9 `viz.py`（可视化，纯 numpy/PIL）

`render_map_image`（固定调色板：可行走绿/已观察灰蓝/障碍红/临时障碍橙/出口黄/交互青/
未知黑/玩家白/路径品红/目标青环 + 图例 + 覆盖率文字）、`render_world_image`（区域节点+
连接边）、`render_status_panel`、`composite_viewer`（480×360：游戏画面+地图+状态单图）、
`coverage_bar`、`describe_cell`。渲染在 runtime 线程完成，Tk 主线程只贴图，不阻塞导航。

## 5. DeepSeek 集成（工具与提示词）

新增工具（OpenAI function-calling schema，严格参数校验，`llm_agent/tools.py`）：

| 工具 | 作用 |
|---|---|
| `explore_current_area(max_steps?, max_seconds?)` | 自主探索当前区域；返回覆盖率、新格、障碍、出口、停止原因 |
| `navigate_to(kind=exit/cell/landmark/return_to_entry/area, …)` | 用已建地图导航到已知目标 |
| `get_navigation_status()` | 区域/pose/置信度/覆盖率/事件 |
| `get_local_map(radius?)` | ASCII 局部地图 + 覆盖统计 |
| `get_world_map_summary()` | 区域列表、覆盖率、区域连接 |
| `pause_exploration()` / `resume_exploration()` | 安全暂停/恢复导航 |

保留 legacy 工具（`press_button`/`execute_actions`/`walk`/`wait_frames`/…）用于对话、菜单、
战斗等非导航场景；其方向移动结果通过 `ToolExecutor.on_action_outcome` 回调**恰好一次**
并入导航地图（`ActionOutcome` 去重），不会重复处理上一次结果。

提示词（`prompts.py`）：
- 服务端 system prompt 中新增 "NAVIGATION SYSTEM" 段：明确"DeepSeek 决定探索什么、Python
  决定怎么走"，禁止逐格按方向键探索房间，说明事件暂停与恢复方式。
- 每轮 user prompt 注入 `NAVIGATION STATUS / LOCAL MAP` 块（区域、pose、置信度、覆盖率、
  ASCII 本地地图、世界摘要），长度受配置限制（默认 2400 字符）。
- 预算/成本策略不变：API 调用前最坏情况估算 + 运行/总预算上限。

## 6. Story Agent 预留接口

Navigation 子系统**独立存在、独立测试**，不依赖剧情理解。未来 Story Agent 可直接调用：

- `explore_current_area()` —— 探索未知区域；
- `navigate_to({"area": id, "cell": [x,y]})` —— 前往已知房间/坐标；
- `navigate_to({"exit": true})` —— 走向当前区域出口（用于换场景）；
- `navigate_to({"landmark": name})` / `{"return_to_entry": true}` —— 去已观察 NPC/地标附近、
  返回入口；
- `status()` / `local_map()` / `world_summary()` —— 查询地图状态；
- 事件回调：`NavEventKind{DIALOG_EVENT,BATTLE_EVENT,MENU_EVENT,STORY_EVENT,TRANSITION_EVENT}`，
  导航遇到非导航事件即暂停并返回结构，Story Agent 处理后可调用 `resume()` 继续。

## 7. 可视化

- 在现有 Tk 观战窗口（`llm_agent/viewer.py`）新增**实时地图面板**：游戏画面下方显示
  `viz.render_map_image`（玩家、已走格、障碍、未知、出口、当前路径/目标、覆盖率图例），
  由 runtime 以节流方式（≤4 次/秒）post 到 Tk 主线程，绝不在 GUI 线程跑导航/渲染。
- `nav_v2_map_gallery.py` CLI：读取持久化 world model，输出每区域地图、世界图、状态面板、
  index.md，作为报告证据（本报告图片位于 `artifacts/nav_v2/evidence/`）。
- `composite_viewer` 提供 480×360 单图模式（游戏画面+地图+状态+覆盖率条），供离线截图/回放。

## 8. 配置、预算与安全检查（`config.py: NavigationConfig`）

`enabled`、`world_root`、`max_explore_steps/seconds/no_progress`、`max_navigate_steps/seconds`、
`localizer_max_tiles`、`status_in_prompt`、`prompt_map_radius/max_chars`、
`persist_on_checkpoint`、`viewer_map_panel`、`max_map_cells`、`max_areas`。
支持 `.env`/JSON 配置覆盖；`AgentConfig.validate()` 校验。

安全：导航永不按 A 推剧情（遇到对话框/菜单/战斗先暂停）；地图有 `max_map_cells/max_areas`
防止错误地图无限扩张；探索/导航有硬预算与无进展上限；`save_progress`/checkpoint 与
world model 同步保存（恢复后强制视觉重定位，绝不盲信旧坐标）。

## 9. 合规性与隔离

- `llm_agent/navigation/**` 不导入 PyBoy、不导入 `llm_agent.progress`、不含 RAM 地址、
  不含剧情/地点知识；由 `tests/llm_agent/test_import_boundaries.py` 强制检查。
- RAM 真值仅存在于 `llm_agent/progress.py`（VERIFICATION-ONLY）与验收脚本
  `scripts/nav_v2_g1/g2/g3/g4*.py` / `tests/navigation`，只用于统计漂移与验收，**不进入
  决策链路**（本设计文档中所有 RAM 数字均来自这些验证脚本）。
- ROM、存档、API Key、录像、本地模型权重均不提交（`.gitignore` 已覆盖）。

## 10. 已知限制（如实）

1. 第一次探索时若未知格恰好是门口（warp tile），仍可能触发一次不期望的转场；之后该门被
   标为 EXIT、门口相邻未知格被延迟，不再反复穿门。
2. 门口为 2 格宽：第二格门口在首次探索时才被发现，可能多一次转场往返（G2 实测 1 次）。
3. 动态 NPC 站在未验证格上被当作永久障碍的概率存在（实现允许 TEMP_BLOCKED 重试，但需要
   `temporary=True` 的证据来源；当前 BLOCKED 默认 conservative）。
4. 视距只有 10×9 tile；不可达/被家具围住的区域只能标记 `UNREACHABLE/DEFERRED` 并如实
   报告（G2 最终 coverage 97.53%，而非 100%）。
5. 相似房间（相同地板纹理）可能产生重定位歧义：`locate()` 会诚实返回 None 而不猜测；
   `attach()` 弱证据时新建区域（可能造成重复区域）。
6. 战斗/对话等事件导航不处理，只暂停并上报；恢复需要上层（当前由 DeepSeek 或验收脚本）。
7. 室外动画（NPC/水面）会降低单步置信度：G4 长跑 UNKNOWN 率约 7.9%，均未导致错误坐标。

## 11. 文件清单（新增/修改）

新增：
- `llm_agent/navigation/`：`__init__.py`、`types.py`、`vision.py`、`localization.py`、
  `grid_map.py`、`world_model.py`、`pathfinder.py`、`explorer.py`、`controller.py`、
  `navigation_agent.py`、`viz.py`、`NAV_CONTRACT.md`
- `tests/navigation/`：conftest + 视野/定位/网格/世界/寻路/探索/控制器/可视化/集成测试
- `scripts/`：`nav_v2_common.py`、`nav_v2_g1_localization.py`、`nav_v2_g2_explore_room.py`、
  `nav_v2_g3_reentry.py`、`nav_v2_g4_long_explore.py`、`nav_v2_map_gallery.py`
- `configs/navigation_v2_api.json`

修改：
- `llm_agent/contracts.py`（新增工具名/导航工具列表）、`config.py`（NavigationConfig）、
  `prompts.py`（导航工具与导航状态块）、`agent.py`（nav_status_text 透传）、
  `tools.py`（7 个导航工具 + raw movement exactly-once 同步回调）、
  `runtime.py`（导航子系统接线/状态/检查点/报告/地图面板）、`viewer.py`（实时地图面板）、
  `progress.py`（验证用 OverworldProbe）、相关测试文件。
