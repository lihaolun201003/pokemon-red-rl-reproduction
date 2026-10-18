# Pokémon Red Autonomous Agent

**Vision-Based Autonomous Exploration and Multi-Agent Decision-Making**
（视觉驱动的自主探索与多智能体决策）

这是一个**视觉大模型自主游戏智能体**的研究工程项目：让视觉 LLM（DeepSeek）只通过真实 Game Boy
模拟器画面观察游戏、通过受限按键操作游戏，研究视觉环境建模、自主探索、长期决策与多智能体协作。

> ⚠️ **合规声明（请先读）**：本仓库**不包含**任何游戏 ROM、模拟器存档、模型权重、游戏录像或
> API Key。运行需要你**自行合法拥有**的英文 Pokémon Red ROM 与兼容的初始存档；代码中的
> SHA1/SHA256 校验只用于确认你提供的输入兼容，不提供下载。详见 [NOTICE.md](NOTICE.md)。

---

## 1. 项目概述

本项目以真实 Game Boy 游戏作为实验环境，研究**只依赖游戏画面**的自主智能体。视觉 LLM 观察
模拟器截图并决定动作；Python 侧负责所有"非决策"部分：模拟器控制、视觉定位、地图构建、任务
调度、记忆、预算与安全边界。

智能体被设计为**第一次玩的盲探索玩家**：

* 决策回路**不能读取游戏 RAM**；
* 没有攻略、没有预设地图、没有隐藏坐标/剧情 Flag/NPC 位置；
* 它记录的每一条事实都必须能由截图或失败的尝试解释；
* 判断"是否真的取得进展"依靠一个隔离的、仅用于验证的 RAM 探针
  （`llm_agent/progress.py`），只进入报告/测试，永不进入决策回路（有导入边界测试强制约束）。

本仓库的前身是一个 PPO 强化学习复现工程。为了聚焦当前研究，旧 PPO 训练管线已从 `main`
移除，完整保留在 Git 分支 **`archive/ppo-reproduction`**（未改写任何历史）。

## 2. 研究问题

* **视觉环境建模**：视觉模型能否只凭 160×144 截图重建可用的空间地图（格、墙、门、NPC）？
* **自主探索**：没有地图时如何选择去哪探索？如何判断一个区域"探索完了"？
* **长期决策**：每一步都是一次 LLM 调用，如何在上千帧的尺度上保持连贯目标？
* **多智能体任务协作**：导航、剧情、战斗三个独立智能体如何共享同一台模拟器，并保证控制权唯一？
* **记忆与经验复用**：持久化知识是否真的改善决策，还是只是不断累积？
* **重复与无效行为**：已经有地图的智能体，为什么还是走已经走过的路？
* **失败恢复与策略调整**：进展停滞时能否检测出来并改变策略（提示 → 复盘 → 回档）？

## 3. 系统架构

```
                    ┌──────────────────────────────────────────────┐
                    │              Game Director（游戏导演）        │
                    │  确定性 Python 调度器；仅少量调用 LLM 复盘     │
                    │  拥有任务图与控制权决策                        │
                    └───────┬──────────────┬──────────────┬────────┘
                            │              │              │
                   ┌────────▼───┐   ┌──────▼─────┐   ┌────▼───────┐
                   │ Navigation │   │   Story    │   │  Pokémon   │
                   │   Agent    │   │   Agent    │   │   Agent    │
                   │ 纯视觉导航  │   │ 视觉 LLM   │   │ 视觉 LLM   │
                   │ + Python   │   │ 接管对话    │   │ 接管战斗    │
                   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
                         │                │                │
             ┌───────────▼────────────────▼────────────────▼─────────┐
             │ Event Bus（持久化消息队列）· TaskManager / Scheduler    │
             │ SharedWorldMemory · CheckpointStore + 分支 manifest    │
             │ ProgressMonitor / RecoveryManager · BudgetTracker      │
             └───────────────────────────┬───────────────────────────┘
                                         │
                             ┌───────────▼───────────┐
                             │ ControlArbiter        │
                             │ 单 owner 控制权租约    │
                             └───────────┬───────────┘
                                         │
                             ┌───────────▼───────────┐
                             │ GuardedEmulator/PyBoy │
                             │  真实 160×144 画面     │
                             └───────────────────────┘
```

* **Game Director（游戏导演）**：确定性调度器。分类屏幕状态
  （`overworld / dialog / battle / menu / transition`），执行硬性限制，启动/暂停/重排任务，
  做进展检查与恢复决策。它自己从不按键，只偶尔调用 LLM 复盘/提案。
* **Navigation Agent（导航智能体）**：Navigation v2。纯视觉定位（16×16 tile 网格 + 相机
  跟随位移匹配）、每区域永久逻辑地图、跨区域世界图、frontier 自主探索、A* 寻路与逐步视觉验证。
  DeepSeek 只决定"探索什么/去哪里"，Python 决定"怎么走"。
* **Story Agent（剧情智能体）**：接管真实对话画面：从截图读取对话、用有界真实按键推进、把
  NPC 与线索连同证据记入记忆，结束后释放控制权并提出后续任务。
* **Pokémon Agent（宝可梦智能体）**：接管战斗：从像素判断战斗菜单、用有界 A 键跳过开场文本，
  每个回合由视觉 LLM 决定（攻击/第几招/逃跑/捕捉/换人），再由确定性的"真实菜单驱动"逐格验证执行，
  并记录战斗经验。
* **共享基础设施**：持久化 Event Bus（去重、优先级、重放）、带显式租约的 `ControlArbiter`
  与拒绝非 owner 按键的 `GuardedEmulator`、TaskManager（DAG、重试、分支感知）、可解释评分的
  确定性 TaskScheduler、SharedWorldMemory（证据、置信度、回档后标 HISTORICAL）、分层
  ProgressMonitor、RecoveryManager（L1 提示 / L2 导演复盘 / L3 checkpoint 回档，含预算）、
  分支 checkpoint manifest（state SHA256 + 任务 + 导演 + 导航 + 记忆 + 消息进度 + 预算）、
  持久化 `BudgetTracker` 与按 Agent 记账的 `ModelRouter`。

## 4. 已实现的功能

* 真实 PyBoy 模拟器接入：每次按键约 24 帧、只暴露截图、存档往返、可选 Tk 观战窗口（游戏画面 +
  实时导航地图面板 + 多智能体状态面板）。
* 纯视觉定位与永久地图：每区域逻辑网格（已观察/已走过/障碍/临时障碍/出口/可交互/置信度/证据）、
  跨区域世界图、再次进入的视觉重定位、地图画廊导出。
* 自主探索：frontier 选择 + 每个任务的步数/时间预算 + 诚实的覆盖率统计。
* 单 owner 控制权仲裁：`navigation / story / pokemon / system`，默认拒绝，持有者异常不会死锁。
* 多智能体协作：持久化消息队列（内容去重 + message-id 幂等）、任务委托（Story → Navigation）、
  任务与记忆的分支感知。
* 真实剧情/战斗接管：Story 读取并推进真实对话；Pokémon 在真实野生战斗中逐回合决策并取得胜利
  （均已在真实 ROM + 真实 DeepSeek API 上验证）。
* Checkpoint / 回档系统：完整 manifest、回档后经验保留并标记 HISTORICAL、API 花费单调不回退。
* 进展监测（L1/L2/L3）与硬上限安全暂停，每个决策都有可审计的 JSONL 记录（本地 gitignored）。
* 成本控制：按 Agent 的 token 上限、图片缩放、持久化的单次/累计预算、按 Agent 的费用台账。
* **1,248 项**单元/集成测试 + 真实游戏验收脚本（G0–G7）。

## 5. 实验结果

以下数字全部来自真实 ROM 上的真实运行（标注处的 G2/G3/G7 使用真实 DeepSeek API）；完整证据与
诚实的偏差说明见链接报告。

**Navigation v2（真实 ROM + 真实 PyBoy）**

* G1 定位：122 次方向操作，88 次真实位移全部判对，**错误位移 0、漏检 0、UNKNOWN 0、漂移 0 格**
  （包含转身、撞墙、菜单/战斗画面、门口传送）。
* G2 陌生房间：从零建成 Oak's Lab 地图——**353 个逻辑格（79 可行走 / 56 障碍 / 1 出口），
  覆盖率 97.53%**，自主判定完成。
* G3 再次进入：视觉重定位 score **0.9213**，**0 个重复区域**，用历史地图 14 步到达最远已知点
  （首次探索 268 步的 5.2%）。
* G4 无人值守长跑（无 LLM）：426 次真实方向操作、259 次真实位移、4 个区域、1,164 个新格，
  规划失败 0。
* G4-API（真实 DeepSeek）：8 次 API 调用（**$0.0031**）驱动 **159 步**已验证导航（约 20 步/调用）。

**Multi-Agent v1（真实 ROM；G2/G3/G7 使用真实 DeepSeek）**

* G0 回归：**1,248 项测试全部通过**，Navigation v2 真实验收脚本全部通过。
* G1 控制权互斥：任意时刻最多 1 个 lease；越权按键不推进帧；确定性拒绝；精确交接
  （3 次按键 = 72 帧，2 次按键 = 48 帧）；持有者异常后无死锁。
* G2 真实对话接管：真实 OAK 对话被读取并记录，2 次 Story 会话、20 次 API 调用、**$0.0062**，
  控制权释放且导航恢复。
* G3 真实战斗接管：Route 1 野生战斗，7 次 A 键 / 284 帧跳过开场，3 个真实决策回合、
  **45 次验证按键后获胜**，4 次 API 调用、**$0.0014**，随后导航任务恢复。
* G4 跨 Agent 委托：Story 提出导航任务，Director 执行，父任务恢复；重复 message-id 与重复内容
  两种去重均验证。
* G5 回档：真实状态恢复 + 新分支，8 条旧知识标记 HISTORICAL（未删除），回档后新学经验保留，
  API 花费不回退。
* G6 停滞恢复：4 次停滞信号 → L1/L2/L3 各级恢复；按 lineage 的回档上限生效；达到硬上限安全暂停。
* G7 真实集成运行：单次 110 步运行中 Director 调度了全部三个 Agent——**14 场真实战斗、
  5 次真实对话、1,023 次导航操作、16 个区域、19 次 checkpoint、65 次 API 调用、$0.0245**。
  同配置重复 5 次只有 1 次形成完整闭环，探索路线差异与仍存在的循环问题已在结果报告中如实记录。

**更早的单 Agent 真实运行**：228 次真实决策 / 250 次工具调用，约 **$0.055**；离开初始房间、
到达 Route 1、经历多场野生战斗（初始宝可梦 L6 → L7）——**未获得徽章、未到达新城镇、未通关**。

## 6. 当前局限

* **智能体已能构建并复用视觉地图，但仍难以把空间记忆转化为有效的长期探索策略。**
  （The agent can construct and reuse visual maps, but still struggles to translate spatial memory
  into effective long-horizon exploration policies.）区域探索完成后仍会重走已探索道路，
  出口有时呈来回往返模式。
* 长期规划不稳定：跨多个区域的目标选择仍偏被动（剧情提案 + frontier 探索），没有持久计划。
* 剧情理解不可靠：能读懂对话，但把对话转化为正确的中期目标仍然很弱。
* 经验学习的实际效果尚未验证：战斗经验、NPC 事实会累积，但可测量的策略提升仍是开放问题。
* 长跑会出现重复/低价值任务与偶发定位漂移；系统有恢复机制（视觉重定位、门口往返检测、
  安全暂停），但底层探索策略才是瓶颈。
* 战斗策略朴素（按设计不使用属性克制/图鉴知识）；捕捉、道具、换人流程未单独在真实 ROM 验证。
* 没有进程内全局 HTTP 看门狗；一次挂起的 API 请求可能卡住运行（验收脚本有超时/看门狗缓解）。
* 未测试多小时的连续运行；5 次 G7 中只有 1 次形成完整交互闭环。
* 本项目**不宣称已通关**，也不应被作为"打通游戏"的成果展示。

## 7. 快速开始

**环境要求**：Windows 10/11 + Python 3.10（唯一实测环境）、合法拥有的英文 Pokémon Red ROM，
可选 DeepSeek API Key。

```powershell
# 1. 安装（不需要 PyTorch / CUDA / PPO 依赖）
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt      # 运行依赖
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt  # 测试依赖

# 2. 放入你的合法游戏数据（gitignored，永不提交）
#    local\PokemonRed.gb                (SHA1 ea9bcae617fdf159b045185467ae58b2e4a48b9a)
#    local\has_pokedex_nballs.state     (报告所用的初始存档)

# 3. 可选：配置 API Key（只有真实 LLM 运行需要）
Copy-Item .env.example .env   # 然后填写 DEEPSEEK_API_KEY=...
#    没有 Key 也可以运行下面全部不花 API 费用的检查

# 4. 环境与 ROM 自检（不花 API 费用）
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe scripts\check_game.py --steps 64

# 5. 在真实游戏上运行多智能体系统
.\.venv\Scripts\python.exe scripts\run_multi_agent.py --config configs\multi_agent_v1.json --viewer
#    加 --headless 不显示窗口；--print-config 可查看最终生效配置

# 6. 单 Agent 旧入口（调试用）
.\.venv\Scripts\python.exe scripts\run_llm_agent.py --config configs\navigation_v2_api.json --headless

# 7. 测试（不需要 ROM）与真实 Navigation v2 验收
.\.venv\Scripts\python.exe -m pytest tests\llm_agent tests\navigation tests\multi_agent tests\story tests\pokemon -q
.\.venv\Scripts\python.exe scripts\nav_v2_g1_localization.py
```

预算安全：智能体会持久化记录 API 花费（`llm_agent/budget.py`），达到上限时安全暂停。
一次 G7 规模的真实运行目前约 **$0.02–0.03**。

## 8. 报告

设计、验收与真实运行证据：

* Navigation v2 —— [设计](docs/reports/navigation_v2_design.md) ·
  [验收](docs/reports/navigation_v2_validation.md) ·
  [结果](docs/reports/navigation_v2_results.md)
* Multi-Agent v1 —— [设计](docs/reports/multi_agent_v1_design.md) ·
  [验收](docs/reports/multi_agent_v1_validation.md) ·
  [真实运行结果](docs/reports/multi_agent_v1_results.md) ·
  [已知限制](docs/reports/multi_agent_v1_known_limitations.md)
* 单 Agent LLM 智能体 —— [设计](docs/reports/llm_agent_design.md) ·
  [验收](docs/reports/llm_agent_validation.md) ·
  [首次真实运行](docs/reports/llm_agent_run_report.md)
* 面向科研交流的研究简介：[docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md)
* 仓库清理报告：[docs/reports/repository_cleanup_report.md](docs/reports/repository_cleanup_report.md)
* 归档的 PPO 复现工程（旧代码、旧报告、完整历史）：Git 分支 `archive/ppo-reproduction`

---

**License**：本仓库 MIT。第三方声明见 [NOTICE.md](NOTICE.md)。
Pokémon 及相关游戏内容版权归其权利人所有。
