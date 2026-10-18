# DeepSeek LLM Agent — 首次真实游戏运行报告（诚实版）

日期：2026-10-18
设计：[llm_agent_design.md](llm_agent_design.md) · 离线/API 验收：[llm_agent_validation.md](llm_agent_validation.md)
运行目录（本地，不入库）：`artifacts/llm_agent/runs/run_20261018_111635_80f907/`

## 1. 结论摘要（只写真实做到的）

**DeepSeek（`deepseek-flash`）真实 API 已成功驱动真实 PyBoy 模拟器连续游戏**，每轮流程为：
真实截图 → DeepSeek 决策（中文理由 + 工具调用）→ PyBoy 执行按键 → 新截图 → 继续决策，全程无人工逐步指挥。

实际达到的最高进度（由独立 RAM 探针 + 审计日志双重核验）：

| 项目 | 初始 | 最终 | 结论 |
|---|---|---|---|
| 地图 map_id | 40（初始室内） | 37 | **已离开初始房间**，进入过 Pallet Town 外景并走到 Route 1 草地 |
| 坐标 | (5,3) | (3,6) | 在同/邻接地图内探索 |
| 徽章 badges | 0 | **0** | **未获得任何徽章** |
| 队伍等级 | L6 | **L7** | 真实野生战斗后升级 |
| 队伍数量 | 1 | 1 | 未捕获新宝可梦 |
| 已发现地点（Agent 自己记录） | 0 | 0（发现记录 14 条，地点类 0） | 模型更多用发现记录而非 place 分类，属实现使用习惯 |
| 自绘心理地图 | — | 12+ 格、已标记撞墙点 | 航位推算地图工作正常 |

已观察到并确认的游戏事件：

- 走出 Oak 研究所 → 进入 Pallet Town 外景（可见房屋、围栏、告示牌）；
- 沿北侧出口走到 **Route 1 草地**，进入草丛并触发多场**真实野生战斗**（Rattata、Pidgey）；
- 战斗中按照真实画面选择攻击、跳过文本，队伍升到 L7，也出现过我方 HP 1/24 的危险局面；
- 曾走回/走进民宅室内（模型理由："又回到了室内（主角自己的房间）"）；
- 未到达任何新城镇、未获得徽章、未完成任何道馆/剧情目标。

**没有通关，也没有宣称通关。** 这是当前真实进度。

## 2. 真实性与证据

- 运行模式：`deepseek-flash` 真实 API（非 mock/scripted；`--scripted` 只用于开发自测且报告有醒目标注，本次结果不含任何 scripted 数据）；
- 审计日志（本地运行目录）：
  - `audit/decisions.jsonl`：**228 条真实 DeepSeek 决策**（模型理由、工具调用、tokens、费用、延迟）；
  - `audit/actions.jsonl`：**250 次工具执行，0 次工具错误**；
  - `audit/api_attempts.jsonl`：236 条 API attempt，0 次重试失败；
  - `audit/events.jsonl`：检查点、恢复、防循环干预、模式升级等事件；
  - `audit/errors.jsonl`：0 条错误记录；
- 截图：`screens/` 每个动作批次一张真实 160×144 画面帧；
- 检查点：`checkpoints/` 合法 PyBoy 状态 + 记忆/地图/预算快照；
- 录像（可选，本地）：`artifacts/llm_agent/demo_final.mp4` 等；
- 独立验证：`scripts/llm_agent_progress.py --run-dir <run>` 从最终 checkpoint 读取真实游戏状态（仅供报告，绝不进入 Agent 决策）；
- API 视觉验收：`scripts/llm_agent_api_probe.py` 中模型真实读出随机 4 位验证码（见 validation 报告），且每次决策请求都附带真实游戏截图（`image_bytes` 约 5–11 KB，640×576 PNG）。

## 3. API 用量与费用（真实统计）

| 指标 | 数值 |
|---|---|
| 真实 DeepSeek 决策数 | 228 |
| 工具调用执行数 | 250（0 错误） |
| 输入 tokens | 962,983（其中缓存命中 679,808，约 70.6%） |
| 输出 tokens | 17,489 |
| 估算费用 | **$0.055011**（按 DeepSeek Flash 官方价格，实际以账单为准） |
| 平均单次决策延迟 | 5.81 s |
| 平均单次决策费用 | ≈ $0.00024 |
| 单次运行预算 / 上限 | `--max-usd 5.00` / `--max-calls 2000`（未触发，因为运行被安全停止） |

成本控制机制均在真实运行中生效：每次请求前最坏情况估算 + 预算上限；本次远未触顶。

## 4. 自主性与自纠错（真实事件）

- **防重复对话循环**：真实运行中曾出现“对话结束后继续按 A 反复触发同一 NPC”（约 30 轮）；提示词加入反循环规则后，运行时检测到“连续只按 A + 对话框仍在”的循环并自动走开一步打破（`repetition_loop_break` 事件 1 次），随后模型从新画面恢复探索；
- **检查点/恢复**：20 次检查点保存；开发过程中多次安全停止/恢复（7 次 `checkpoint_restored`），全部从合法存档继续，没有改 RAM/背包/剧情；
- **Guard**：0 次 critical；无工具错误；无 API 重试失败；
- **长期记忆**：14 条发现记录、1 条教训（"对话结束后不要反复按 A"）、当前目标与原因会跨会话保留；
- **自绘心理地图**（2026-10-18 新增，用户建议）：根据自己每步移动结果做航位推算，`·`=走过的路、`#`=试过走不通、`?`=未探索，每轮注入提示词；跨进程持久化，避免反复撞同一栅栏/树。

## 5. 已知限制与诚实说明

1. **API 延迟是硬瓶颈**：实测即使最小请求（只回 PONG）也约 5.5 s；平均 5.8 s/决策。已通过“一次决策批量执行更多动作 + 60fps 实时回放”缓解观感，但游戏推进速度无法与真人相比；
2. 模型会迷路/绕圈：本次真实运行中在 Pallet Town 围栏与草地一带探索较久，且曾走回室内；没有攻略提示是设计使然；
3. 未获得徽章、未到新城镇，剧情推进非常有限（约 14 分钟游戏内时间、51,318 emulated frames）；
4. 观看窗口/60fps 渲染/心理地图是在真实运行期间迭代加入的（同一 run 目录、多次安全重启），因此第 2 节统计是**多个会话合并**的真实审计数据；
5. 费用是按官方 Flash 价格表估算（缓存命中/未命中/峰谷价），最终以 DeepSeek 平台账单为准。

## 6. 如何继续 / 复现

```powershell
# 真实 API 连通性/视觉/工具调用验收（必需）
.\.venv\Scripts\python.exe scripts\llm_agent_api_probe.py

# 继续同一局（自动读取最新 checkpoint / 记忆 / 心理地图）
.\.venv\Scripts\python.exe scripts\run_llm_agent.py --run-dir artifacts\llm_agent\runs\run_20261018_111635_80f907 --headless --max-usd 2

# 观看模式（实时窗口 + 60fps 渲染 + 暂停/保存/退出）
.\.venv\Scripts\python.exe scripts\run_llm_agent.py --run-dir artifacts\llm_agent\runs\run_20261018_111635_80f907

# 独立核验真实进度（仅报告用）
.\.venv\Scripts\python.exe scripts\llm_agent_progress.py --run-dir artifacts\llm_agent\runs\run_20261018_111635_80f907

# 汇总任意 run 的可审查报告
.\.venv\Scripts\python.exe scripts\llm_agent_summarize_run.py --run-dir artifacts\llm_agent\runs\<run>
```

安全与合规：`.env` 与 `llm_agent/memory/`、`artifacts/` 均被 gitignore；报告/日志不含 API Key；未修改 ROM、未写入游戏存档、未上传任何内容。
