"""Summarize an LLM-agent run into a reviewable Markdown report + JSON metrics.

Combines the run's own artifacts (run.json / metrics.json / report.json / audit
JSONL logs) with the **verification-only** RAM progress probe on the final
checkpoint.  Never performs any API call.

Example:
    .\\.venv\\Scripts\\python.exe scripts\\llm_agent_summarize_run.py --run-dir artifacts\\llm_agent\\runs\\run_xxx
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

from llm_agent.config import AgentConfig  # noqa: E402
from llm_agent.progress import compare_progress, probe_from_state_file  # noqa: E402


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records[-limit:] if limit else records


def _clip(text: str, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def resolve_checkpoint_state(run_dir: Path) -> Path | None:
    latest = _read_json(run_dir / "checkpoints" / "latest.json")
    state_file = latest.get("state_file")
    if state_file and Path(state_file).is_file():
        return Path(state_file)
    states = sorted((run_dir / "checkpoints" / "states").glob("*.state"))
    return states[-1] if states else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="", help="markdown output path (default: <run>/review_report.md)")
    parser.add_argument("--json-out", default="", help="json summary output path")
    parser.add_argument("--rom", default="")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"[error] run dir not found: {run_dir}", file=sys.stderr)
        return 2
    cfg = AgentConfig.load(env_file="")
    rom = Path(args.rom or cfg.emulator.rom)

    run_json = _read_json(run_dir / "run.json")
    metrics = _read_json(run_dir / "metrics.json")
    own_report = _read_json(run_dir / "report.json")
    decisions = _read_jsonl(run_dir / "audit" / "decisions.jsonl")
    actions = _read_jsonl(run_dir / "audit" / "actions.jsonl")
    events = _read_jsonl(run_dir / "audit" / "events.jsonl")
    errors = _read_jsonl(run_dir / "audit" / "errors.jsonl")
    attempts = _read_jsonl(run_dir / "audit" / "api_attempts.jsonl")

    result = own_report.get("result", {}) if isinstance(own_report, dict) else {}
    budget = own_report.get("budget", {}) if isinstance(own_report, dict) else {}
    memory = own_report.get("memory", {}) if isinstance(own_report, dict) else {}
    memory_data = memory.get("data", memory) if isinstance(memory, dict) else {}

    tool_counter = Counter(record.get("tool", "?") for record in actions)
    tool_errors = [record for record in actions if not record.get("ok")]
    api_calls = [record for record in decisions if record.get("mode") == "deepseek"]
    total_prompt = sum(int((record.get("usage") or {}).get("prompt_tokens") or 0) for record in decisions)
    total_completion = sum(int((record.get("usage") or {}).get("completion_tokens") or 0) for record in decisions)
    total_cache_hit = sum(int((record.get("usage") or {}).get("cache_hit_tokens") or 0) for record in decisions)
    total_cost = sum(float(record.get("cost_usd") or 0.0) for record in decisions)
    latencies = [float(record.get("latency_s") or 0.0) for record in decisions if record.get("latency_s")]

    progress = None
    state_file = resolve_checkpoint_state(run_dir)
    if state_file is not None and rom.is_file():
        try:
            initial_path = Path(run_json.get("initial_state", {}).get("path") or cfg.emulator.state)
            final_snapshot = probe_from_state_file(rom, state_file)
            initial_snapshot = probe_from_state_file(rom, initial_path) if initial_path.is_file() else None
            progress = {
                "checkpoint_state": str(state_file),
                "final": final_snapshot.to_dict(),
                "initial": initial_snapshot.to_dict() if initial_snapshot else None,
                "comparison": compare_progress(initial_snapshot, final_snapshot) if initial_snapshot else None,
            }
        except Exception as exc:  # noqa: BLE001
            progress = {"error": f"{type(exc).__name__}: {exc}"}

    summary = {
        "run_dir": str(run_dir),
        "run": run_json,
        "metrics": metrics,
        "result": result,
        "budget": budget,
        "api": {
            "decision_records": len(decisions),
            "deepseek_calls": len(api_calls),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "cache_hit_tokens": total_cache_hit,
            "cost_usd": round(total_cost, 6),
            "attempt_records": len(attempts),
            "retry_failures": sum(1 for attempt in attempts if int(attempt.get("attempt") or 1) > 1),
            "mean_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        },
        "tools": {
            "executions": len(actions),
            "by_tool": dict(tool_counter),
            "errors": len(tool_errors),
            "error_samples": tool_errors[:5],
        },
        "loop": {
            "events": len(events),
            "repetition_loop_breaks": sum(1 for event in events if event.get("event") == "repetition_loop_break"),
            "checkpoints": sum(1 for event in events if event.get("event") == "checkpoint_saved"),
            "restores": sum(1 for event in events if event.get("event") == "checkpoint_restored"),
            "guard_critical": sum(1 for event in events if event.get("event") == "guard_critical"),
            "error_records": len(errors),
            "error_samples": errors[-5:],
        },
        "memory": memory_data,
        "progress_probe": progress,
        "decisions_sample": decisions[-12:],
    }

    out = Path(args.out) if args.out else run_dir / "review_report.md"
    json_out = Path(args.json_out) if args.json_out else run_dir / "review_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    real_api = bool(result.get("real_api"))
    lines: list[str] = []
    lines.append(f"# LLM Agent run review — `{run_dir.name}`")
    lines.append("")
    lines.append(f"- 真实 API 驱动: **{'是' if real_api else '否（SCRIPTED DRY RUN，非验收证据）'}**")
    lines.append(f"- 模型: `{result.get('model') or run_json.get('model')}`")
    lines.append(f"- 状态: **{result.get('status') or metrics.get('status')}** — {_clip(result.get('reason') or '')}")
    lines.append(f"- 时长: {result.get('duration_s') or metrics.get('duration_s')} s")
    lines.append(f"- 目标: {_clip(str(result.get('goal') or memory_data.get('goal', {}).get('text', '')))}")
    lines.append("")
    lines.append("## 运行统计")
    lines.append("")
    lines.append(f"- 决策轮数 turns: {result.get('turns', metrics.get('turns'))}")
    lines.append(f"- PyBoy 帧数 frames: {result.get('frames', metrics.get('frames'))}")
    lines.append(f"- 按键次数 buttons: {result.get('buttons', metrics.get('buttons'))}")
    lines.append(f"- 动作批次 batches: {result.get('action_batches', metrics.get('action_batches'))}")
    lines.append(f"- 检查点 checkpoints: {result.get('checkpoints', metrics.get('checkpoints'))}")
    lines.append(f"- 已发现地点 places: {result.get('places')}")
    lines.append(f"- 发现记录 discoveries: {result.get('discoveries')}")
    lines.append("")
    lines.append("## DeepSeek API 用量")
    lines.append("")
    api = summary["api"]
    lines.append(f"- 决策记录: {api['decision_records']}（deepseek 模式 {api['deepseek_calls']}）")
    lines.append(f"- tokens: prompt {api['prompt_tokens']:,} / completion {api['completion_tokens']:,} / cache-hit {api['cache_hit_tokens']:,}")
    lines.append(f"- 累计费用: **${api['cost_usd']:.6f}**")
    lines.append(f"- 平均延迟: {api['mean_latency_s']} s；API attempt 记录 {api['attempt_records']}（重试尝试 {api['retry_failures']}）")
    lines.append(f"- 预算: 上限 ${cfg.budget.max_usd_per_run:.2f}/run，调用 {cfg.budget.max_calls_per_run}")
    lines.append("")
    lines.append("## 工具调用")
    lines.append("")
    lines.append(f"- 总执行: {summary['tools']['executions']}，错误: {summary['tools']['errors']}")
    for name, count in tool_counter.most_common():
        lines.append(f"  - `{name}`: {count}")
    lines.append("")
    lines.append("## 游戏进度（独立 RAM 探针，仅报告用）")
    lines.append("")
    if progress and "final" in progress:
        final = progress["final"]
        lines.append(f"- 最终 checkpoint: `{Path(progress['checkpoint_state']).name}`")
        lines.append(f"- 最终: map_id={final['map_id']} coords={tuple(final['coords'])} badges={final['badges']} "
                     f"party_levels={final['party_levels']} hp={final['hp_fraction']}")
        if progress.get("initial"):
            initial = progress["initial"]
            lines.append(f"- 初始: map_id={initial['map_id']} coords={tuple(initial['coords'])} badges={initial['badges']} "
                         f"party_levels={initial['party_levels']}")
        if progress.get("comparison"):
            comp = progress["comparison"]
            lines.append(f"- 变化: map_changed={comp['map_changed']} badges+{comp['badges_gained']} "
                         f"levels+{comp['levels_gained']} party{comp['party_size_change']:+d} "
                         f"coord_distance={comp['coord_distance']}")
    else:
        lines.append(f"- 探针不可用: {progress}")
    lines.append("")
    lines.append("## 防循环 / 自纠错 / 异常")
    lines.append("")
    loop = summary["loop"]
    lines.append(f"- repetition loop breaks: {loop['repetition_loop_breaks']}")
    lines.append(f"- 自动/手动 checkpoint: {loop['checkpoints']} 个；restore: {loop['restores']} 次")
    lines.append(f"- guard critical: {loop['guard_critical']}")
    lines.append(f"- error 记录: {loop['error_records']}")
    for error in loop["error_samples"]:
        lines.append(f"  - `{error.get('kind')}` {_clip(error.get('message') or error.get('error') or '', 160)}")
    lines.append("")
    lines.append("## 最近决策（模型理由 + 工具调用）")
    lines.append("")
    for record in decisions[-12:]:
        tools = "; ".join(f"{call.get('name')}({json.dumps(call.get('arguments'), ensure_ascii=False)})"
                          for call in record.get("tool_calls", []))
        lines.append(f"- **turn {record.get('turn')}** [{record.get('trigger')}] ${record.get('cost_usd', 0):.6f} — "
                     f"{_clip(record.get('rationale') or '(空)')}")
        if tools:
            lines.append(f"  - tools: {_clip(tools, 300)}")
    lines.append("")
    lines.append("## 审计文件")
    lines.append("")
    for name in ("run.json", "metrics.json", "report.md", "report.json", "audit/decisions.jsonl",
                 "audit/actions.jsonl", "audit/api_attempts.jsonl", "audit/events.jsonl", "audit/errors.jsonl"):
        path = run_dir / name
        if path.is_file():
            lines.append(f"- `{name}` ({path.stat().st_size:,} bytes)")
    lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"review written: {out}")
    print(f"summary json : {json_out}")
    print(f"real_api={real_api} status={result.get('status')} cost=${api['cost_usd']:.6f} "
          f"turns={result.get('turns')} frames={result.get('frames')}")
    if progress and progress.get("final"):
        print(f"RAM probe final: {progress['final']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
