# DeepSeek LLM Agent — 设计说明（独立于 PPO）

日期：2026-10-18
状态：**已实现并通过首次真实 API 运行**（见 `llm_agent_run_report.md`；离线测试 353 项通过；真实 API 验收 PASS）。

## 1. 目标

在已有 Pokémon Red + PyBoy 工程上，新增一个**由 DeepSeek 官方 API 直接驱动**的游戏智能体，让 DeepSeek 像玩家一样通过真实截图观察、决策、按键、记录经验，并在无人逐步指挥的情况下长期自主探索，最终尝试通关。

本项目**不再以 PPO 训练为主要执行方式**；PPO 代码、checkpoint、训练报告全部保留不动，便于后续比较：

- PPO Agent（已有）
- DeepSeek LLM Agent（本项目，纯 LLM）
- 未来：LLM + PPO 混合

## 2. 核心架构

```
DeepSeek Brain ──tool calls──► ToolExecutor ──► PyBoy (真实模拟器)
      ▲                                              │
      └────────── screenshot + 观察文本 ◄────────────┘
```

一个*决策周期*：

1. `GameBoyEmulator.observe()` 抓取当前 160x144 RGBA 帧（PyBoy `screen.ndarray`）。
2. `observation.analyze()` 用**纯像素**启发式给出屏幕事件（对话框/菜单/战斗/换屏/黑屏）、变化率、感知签名。
3. `LoopGuard` 更新防卡死/防循环信号。
4. `DeepSeekBrain.prepare()` 组装有界上下文：静态系统提示词 + 当前目标 + 长期记忆摘要 + 最近历史 + Guard 告警 + 屏幕分析 + 1 张 4x 最近邻放大截图（PNG data URI）。
5. `BudgetTracker.check_before_call()` 以最坏情况估算费用，未超预算才允许请求。
6. `DeepSeekClient.chat()` 调用 `https://api.deepseek.com/chat/completions`（`deepseek-flash`，非流式，默认 thinking disabled）。
7. 模型返回 1-2 句决策理由 + 一个或多个工具调用；`ToolExecutor` 按契约严格校验参数后执行；遇到对话框/战斗/菜单/换屏等显著事件时中断剩余批次，交回模型重新决策。
8. 记录审计日志、更新记忆、按需 checkpoint；循环。

**不做每帧调用**：普通移动由 `walk`/`execute_actions` 在 Python 内批量执行 8+n 帧/动作，一次 API 决策可覆盖多个地块、多段对话或多个人机确认。DeepSeek 只在关键决策点被调用。

## 3. 盲探索与合规边界

| 允许 | 禁止 |
|---|---|
| 真实游戏画面截图 | 网络攻略/剧情路线 |
| 游戏内 NPC 对话文本（截图所见） | 预读完整地图 / NPC 位置 |
| 自己记录的地点、机制、失败经验 | 固定通关脚本 / 人工动作轨迹 |
| 基于自建记忆的局部路径规划 | RAM 反编译/隐藏 Flag/徽章位置 |
| 合法存档与检查点恢复 | 修改 RAM/背包/队伍/剧情 |

实现层面的强制措施：

- 决策回路模块（`agent/runtime/prompts/tools/memory/guard/emulator/observation/api_client/budget/viewer/scripted/config/contracts/dotenv`）禁止导入 `llm_agent.progress`、禁止出现 4 位十六进制 RAM 地址、禁止决策链路直接 import PyBoy（仅 `emulator.py` 允许）。
- `tests/llm_agent/test_import_boundaries.py` 用 AST 静态检查 + 系统提示词扫描（剧情地名、角色、道馆、御三家等关键词）强制执行。
- `llm_agent/progress.py` 是唯一读取 RAM 的模块，文件头明确标注 VERIFICATION-ONLY，只被 `scripts/llm_agent_progress.py` 和报告/测试使用。
- 系统提示词不含任何游戏剧情知识；要求模型把预训练印象当作"待画面验证的假设"。

## 4. 模块清单

| 模块 | 职责 |
|---|---|
| `contracts.py` | 冻结的共享数据契约：按钮/限值/错误类型/ScreenEvent/Frame/ActionOutcome/ToolCall/ToolResult/ApiUsage/ModelReply/BudgetState/RuntimeState |
| `config.py` | 分节配置（api/budget/emulator/memory/guard/runtime/viewer）、环境变量与 `.env`、校验、路径解析 |
| `dotenv.py` | 零依赖 `.env` 解析与密钥脱敏 |
| `emulator.py` | PyBoy 封装：加载 ROM/状态、按键批次、单调帧计数、save/load state、错误处理 |
| `observation.py` | 纯像素 Frame/ScreenAnalysis：哈希、变化率、黑屏、对话框/菜单/战斗/换屏启发式、4x PNG/JPEG 编码 |
| `tools.py` | 11 个受约束工具 + 严格参数校验 + 紧凑结果摘要（模型只通过工具行动） |
| `memory.py` | 持久化长期记忆（`llm_agent/memory/`）：地点/发现/对话/教训/目标/开放问题、检索、上下文压缩、快照、损坏恢复 |
| `guard.py` | 防循环：原地不动、撞墙、重复批次、同一屏停留过久、黑屏、无新发现、连续工具/API 错误 |
| `prompts.py` | 静态系统提示词 + 每轮用户消息渲染（目标/记忆/历史/告警/屏幕分析/截图） |
| `agent.py` | `DeepSeekBrain`：组装请求、调用 API、返回决策（单轮、有界上下文、缓存友好） |
| `api_client.py` | 标准库 DeepSeek 客户端：视觉/工具调用/thinking 开关、重试与退避、401/402/429/5xx 分类、密钥脱敏 |
| `budget.py` | 预算/费用：峰值-非峰值定价、最坏情况预估、单次+累计 USD、调用数/小时、持久化 |
| `runtime.py` | 编排：决策循环、工具批次执行与中断、guard 反馈、checkpoint/恢复、审计 JSONL、指标与中文报告、异常现场、单实例锁、暂停/保存/停止 |
| `viewer.py` | Tk 实时观看窗口（主线程），队列安全，显示画面/目标/动作/决策/步数/地点/API 费用/状态，按钮暂停/保存/退出 |
| `progress.py` | **仅验证**：RAM 探针读取 map/坐标/徽章/队伍等级，供 `scripts/llm_agent_progress.py` 生成独立进度报告 |
| `scripted.py` | 仅开发自测的确定性测试脑（明确标注 SCRIPTED DRY RUN，不作为验收证据） |
| `scripts/run_llm_agent.py` | 启动入口（真实 API / 观看窗口 / headless / 续跑 / 预算参数） |
| `scripts/llm_agent_api_probe.py` | 真实 API 验收探针：文本、视觉读码、工具调用、坏密钥错误处理 |
| `scripts/llm_agent_progress.py` | 独立 RAM 进度核验（报告用，不参与决策） |

## 5. DeepSeek API 接入

- Base URL `https://api.deepseek.com`（OpenAI 兼容），模型默认 `deepseek-flash`（DeepSeek-V4.1-Flash-1017，支持视觉与工具调用）。均可通过 `--base-url/--model` 或环境变量覆盖。
- 视觉：仅 user 消息内嵌 `image_url` data URI；图片为真实游戏帧的 4x 最近邻放大 PNG（160x144 → 640x576）。
- thinking 模式默认关闭以控制成本；请求体顶层 `{"thinking": {"type": "disabled"}}`。可配置开启并设 `reasoning_effort=low|high|max`。
- 计费：输入（缓存命中/未命中）+ 输出 token；官方 Flash 价格（USD/1M tokens，非峰值）：命中 $0.003、未命中 $0.15、输出 $0.60，峰值（周一至周五 UTC 01:00-04:00、06:00-10:00）2 倍。实际以 API 返回 usage 为准，价格可在配置中修改。
- 错误处理：400/422 请求格式问题（不可重试）；401/402 致命（blocked 等人工）；429/5xx/超时指数退避重试（尊重 Retry-After），连续失败超过阈值 blocked。
- 密钥：只从环境变量/`.env` 读取，绝不硬编码、不打印、不写日志/检查点/报告/ git。API 尝试日志已脱敏。

## 6. 工具集（模型唯一行动方式）

`observe_game`, `press_button`, `execute_actions`, `wait_frames`, `walk`, `inspect_memory`, `record_discovery`, `set_goal`, `save_progress`, `restore_checkpoint`, `get_recent_history`。

约束：按钮枚举 = Game Boy 8 键；单次 `execute_actions` ≤ 32 个动作；`wait_frames` 1..600；`walk` ≤ 20 步；文本长度上限；未知工具/字段/越界参数一律返回结构化错误给模型，不崩溃、不改状态。

## 7. 成本控制策略

- 决策级而非帧级调用；历史/记忆有界；系统提示词静态（利于 KV cache 命中）。
- 生成前最坏情况成本估算 + 硬预算（单次运行 / 累计 / 调用数 / 每小时）；命中任一限制 → 安全暂停 + 保存 checkpoint + 写报告，不再调用 API。
- 审计每次调用的 usage、缓存命中、费用、延迟；`report.md` 输出累计费用与 token 统计。

## 8. 长期运行与恢复

- run 目录：`artifacts/llm_agent/runs/run_<时间>_<id>/`（`run.json`、`metrics.json`、`report.md`、`audit/`、`screens/`、`checkpoints/`、`errors/`）。
- 每 N 个动作批次自动 checkpoint（PyBoy state + 记忆快照 + 预算快照 + 运行时快照）；退出/预算暂停/异常时也保存。
- 启动时若 run 目录已有最新 checkpoint 且未 `--fresh`，自动恢复；`--resume-from` 可跨 run 恢复。
- 模拟器异常：保存错误现场（截图/state/traceback/meta）→ 尝试恢复最新 checkpoint（次数上限）→ 仍失败则 blocked。
- 单实例锁 `artifacts/llm_agent/agent.lock`（PID 存活检测，陈旧锁自动清理）。
- 外部控制：`--stop-file`、`--pause-file`、观看窗口按钮、Ctrl+C。所有退出路径都会保存状态并写报告。

## 9. 验收方式（真实证据）

1. `scripts/llm_agent_api_probe.py`：真实 API 文本/视觉/工具/错误处理通过；
2. `scripts/run_llm_agent.py --headless`：真实 API 驱动，PyBoy 画面随按键真实变化，DeepSeek 依据新截图继续决策；
3. `audit/decisions.jsonl` 与 `audit/actions.jsonl` 可逐条审计：模型回复、工具调用、执行结果、tokens、费用；
4. `scripts/llm_agent_progress.py --run-dir ...` 从最终 checkpoint 独立读取真实游戏进度（map/坐标/徽章/队伍等级），写入运行报告；
5. 如实记录达到的最高进度；未通关就明确写"未通关"。
