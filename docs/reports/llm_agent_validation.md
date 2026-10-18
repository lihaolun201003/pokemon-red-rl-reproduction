# DeepSeek LLM Agent — 离线验证与真实 API 验收记录

日期：2026-10-18
对应设计：[llm_agent_design.md](llm_agent_design.md)
运行报告：[llm_agent_run_report.md](llm_agent_run_report.md)

本文件记录**真实执行**得到的验证结果。分为两部分：

1. 不需要 API Key 的自动测试/集成验证（离线、不联网）；
2. 真实 DeepSeek API 验收（真实网络请求，成本如实记录）。

---

## 1. 离线自动测试（无需 API Key，不联网）

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\llm_agent -q
```

结果（2026-10-18 11:09 左右，真实运行）：

```
329 passed, 1 warning in 2.92s
```

覆盖范围（按文件）：

| 文件 | 测试数 | 覆盖内容 |
|---|---:|---|
| `test_emulator.py` | 20+ | 真实 ROM/state 加载、observe/press/wait/run_batch、24 帧/动作时序、save_state 往返、单调帧计数、错误处理 |
| `test_observation.py` | 30+ | 真实帧哈希/黑屏/变化率/签名、对话框/菜单/战斗/换屏启发式（合成图 + 真实帧）、4x PNG/JPEG 编码、保存 |
| `test_tools.py` | 50+ | 11 个工具 schema 与严格参数校验、动作批次执行与事件中断、状态计数、审计截图、错误不崩溃 |
| `test_api_client.py` | 46 | 假 transport：请求体/视觉块/tool_call 解析/usage、429/5xx/超时退避、401/402 不重试、Retry-After、畸形响应、密钥不泄漏 |
| `test_budget.py` | 40 | 峰值/非峰值定价与缓存命中/未命中、最坏成本估算、run/global/每小时限额、持久化与损坏恢复、原子写 |
| `test_dotenv.py` | 31 | `.env` 解析、加载优先级、placeholder 检测、密钥掩码 |
| `test_memory.py` | 39 | 记忆持久化/检索/上下文渲染/压缩/快照恢复/损坏恢复/journal 裁剪 |
| `test_guard.py` | 28 | 7 条防循环规则、三级严重度、告警渲染、快照/恢复 |
| `test_prompts.py` | 5 | 系统提示词无剧透、用户消息结构/截图块、脚本脑合法工具调用 |
| `test_runtime.py` | 12 | 单实例锁、audit、checkpoint 存取/裁剪、预算暂停、save/restore 工具、恢复 checkpoint、完整 scripted 干跑、清理 |
| `test_progress.py` | 4 | RAM 探针（真实 ROM/state）、checkpoint 定位、初始 vs 最终对比 |
| `test_import_boundaries.py` | 13 | **防作弊边界**：决策回路不得 import `progress`/PyBoy、不得含 RAM 地址、提示词不得含剧情地名/角色/道馆/御三家、工具 schema 约束 |

原有 PPO 测试不受影响：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\phase1 -q
# 11 passed
```

### 1.1 真实模拟器干跑（scripted 测试脑，非 API 证据）

```powershell
.\.venv\Scripts\python.exe scripts\run_llm_agent.py --headless --scripted --max-turns 4
```

真实结果：`status=finished`，`frames=506`，`buttons=21`，`checkpoints>=2`，退出码 0；
生成 `run.json` / `metrics.json` / `report.md` / `audit/*.jsonl` / `screens/*.png` / `checkpoints/*.state`；
报告顶部明确标注 `SCRIPTED DRY RUN — NOT a real DeepSeek API run`（脚本脑不污染真实记忆：使用独立
`artifacts/llm_agent/scripted_memory` 与 `scripted_runs`）。

独立 RAM 探针验证过干跑确实驱动了真实游戏状态：

```
final  : map_id=40 coords=(6,8) badges=0 party=1 levels=[6] hp=1.00
initial: map_id=40 coords=(5,3) badges=0 party=1 levels=[6] hp=1.00
```

### 1.2 观看窗口 + 录像冒烟

真实运行（脚本脑 3 轮）：Tk 窗口正常显示画面与状态，运行结束自动关闭；`--video` 生成 MP4
（56,499 字节，可由 OpenCV 解码）；窗口按钮/键盘事件路径无异常。

---

## 2. 真实 DeepSeek API 验收费（probe）

命令：

```powershell
.\.venv\Scripts\python.exe scripts\llm_agent_api_probe.py --timeout 180 --max-retries 4
```

真实结果（2026-10-18 11:06，模型 `deepseek-flash`，thinking=disabled，真实网络请求）：

| 检查 | 结果 | 延迟 | 说明 |
|---|---|---|---|
| text_completion | **PASS** | 5.47 s | 返回 PONG |
| vision | **PASS** | 5.45 s | 模型正确读出一张随机 4 位验证码图片（8399），证明真实视觉输入 |
| tool_call | **PASS** | 5.55 s | 模型返回合法 `get_timestamp({})` 工具调用（id/参数可解析） |
| bad_key_auth | **PASS** | 4.72 s | 故意无效密钥收到 HTTP 401，`ApiAuthenticationError` 未重试 |

- 退出码：0（PASS）
- 成本：约 **$0.000104**（4 次 attempt，报告记录 token 用量）
- 报告：`artifacts/llm_agent_api_probe/report.json`（已脱敏，无密钥/Authorization）

**结论：真实 API 的文本/视觉/工具调用/错误处理全部可用。**

---

## 3. 真实 API 驱动游戏的首次短运行（冒烟）

在正式长运行之前，用真实 API 跑 3 个决策轮验证完整链路：

```powershell
.\.venv\Scripts\python.exe scripts\run_llm_agent.py --headless --max-turns 3 --max-usd 0.30
```

真实结果（run `run_20261018_110631_f26e9c`）：

- DeepSeek 真实决策 3 次（另有 1 次 warmup），模型输出中文理由后调用真实工具；
- PyBoy 帧数从 2 → 50 → 194 → 266（按键 11 次），画面真实变化；
- 模型第 2/3 轮看到了新画面并继续决策，例如理由：
  - “我先看看周围环境，向北被挡住，先向南走探索出口。”
  - “这是一个陈列着多台游戏机的房间，我打算往下走到那个灰色垫子/出口处看看。”
- 成本：**$0.000841**，10,116 tokens；真实 API 缓存命中后单次约 $0.00015–0.00053。

发现并修复的问题（真实运行暴露）：

1. 早期 scripted 干跑把测试目标写进了全局记忆，导致真实运行继承了 “Scripted dry run...” 目标 → 已归档
   scripted 记忆（`artifacts/llm_agent_recon/scripted_memory_archive_*`），并让 `--scripted` 使用独立记忆/运行目录；
2. 首轮模型可能只返回工具调用、`content` 为空 → runtime 现在用“工具调用摘要”兜底观看窗口/报告中的“最近决策”。

正式长运行见 [llm_agent_run_report.md](llm_agent_run_report.md)。

---

## 4. 关键实现保障（对应任务验收标准）

| 要求 | 实现与证据 |
|---|---|
| 真实 API 分析真实截图 | probe vision PASS；真实运行 decisions.jsonl 中每次请求都带真实 640x576 截图（image_bytes>0） |
| DeepSeek 产生可执行工具调用 | 真实运行 tool_calls 通过严格 schema 校验后由 ToolExecutor 执行，actions.jsonl 逐条可审计 |
| PyBoy 正确执行动作 | emulator 测试 + 真实运行 frames/buttons 增长；独立 RAM 探针证实坐标变化 |
| 画面随操作真实变化 | 每轮截图 sha256/变化率记录在 actions.jsonl；观察文本给出 changed 比例与事件 |
| 根据新画面继续决策 | 每轮请求都重新截图并附在消息中；真实运行连续多轮无人工干预 |
| 无人逐步指挥连续运行 | 官方运行由 `scripts/run_llm_agent.py --headless` 单命令启动，内部循环自行决策/执行/存档 |
| 操作、模型回复、费用可审计 | `audit/decisions.jsonl`（理由/工具调用/usage/费用/延迟）、`audit/actions.jsonl`、`audit/api_attempts.jsonl`、`audit/events.jsonl`、`report.md` |

## 5. 诚实性声明

- 本文所有“PASS/通过/真实结果”均来自本机真实执行；没有使用 Mock API 冒充真实成功。
- `--scripted` 结果只用于开发自测，报告中明确标注，不作为 API/游戏验收证据。
- 游戏进度一律以真实观察 + 独立 RAM 探针（仅报告用）为准；未达到的进度不会宣称。

## 6. 最终状态（2026-10-18 收尾）

- 完整离线测试：`pytest tests/llm_agent -q` → **353 passed**（含心理地图、60fps 渲染、预算/恢复等新增测试）；原 PPO `tests/phase1` 仍 11 passed。
- 真实 API 验收探针：**PASS**（文本/视觉读码/工具调用/坏密钥错误处理）。
- 首次真实游戏运行已完成并停止（用户关闭观看窗口）：见 [llm_agent_run_report.md](llm_agent_run_report.md)。
  真实统计：228 次决策、250 次工具调用（0 错误）、约 **$0.055**、平均延迟 5.81 s/决策；初始房间已离开，到过 Pallet Town 外景与 Route 1 草地，经历野生战斗，队伍 L6→L7；**0 徽章、未通关**。
- 运行期间根据用户反馈完成的三项改进：60fps 逐帧实时渲染（消除跳帧）、AI 思考期间画面继续实时运行、自绘心理地图（航位推算记录可走/走不通）。

