"""Launch the DeepSeek LLM agent that plays Pokémon Red autonomously.

Examples
--------
Headless long run (typical unattended usage):
    .\\.venv\\Scripts\\python.exe scripts\\run_llm_agent.py --headless

With the real-time viewer (pause/save/quit buttons):
    .\\.venv\\Scripts\\python.exe scripts\\run_llm_agent.py

Development dry run without any API calls (never acceptance evidence):
    .\\.venv\\Scripts\\python.exe scripts\\run_llm_agent.py --headless --scripted --max-turns 8

The real API key is read from the environment or a local ``.env`` file
(``DEEPSEEK_API_KEY``); it is never printed, logged or committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _configure_stdio() -> None:
    """Force UTF-8 console output so Chinese reports are readable in pipes/logs."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


_configure_stdio()

from llm_agent.agent import DeepSeekBrain
from llm_agent.api_client import DeepSeekClient
from llm_agent.config import AgentConfig
from llm_agent.runtime import AgentRuntime, RunResult
from llm_agent.scripted import ScriptedBrain
from llm_agent.viewer import AgentViewer

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_BLOCKED = 3
EXIT_NO_KEY = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek (deepseek-flash) plays Pokémon Red autonomously on a real Game Boy emulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="optional JSON config file (sections api/budget/emulator/...)")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"), help="local .env file with DEEPSEEK_API_KEY")
    parser.add_argument("--rom", default=None, help="lawful English Pokémon Red ROM path")
    parser.add_argument("--state", default=None, help="official initial emulator state path")
    parser.add_argument("--run-dir", default=None, help="reuse an explicit run directory")
    parser.add_argument("--resume-from", default=None, help="resume from the latest checkpoint of a previous run dir")
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoints, start from the initial state")
    parser.add_argument("--label", default=None, help="human-readable label for this run")
    # API / model
    parser.add_argument("--api-key-env", default=None, help="environment variable name holding the API key")
    parser.add_argument("--model", default=None, help="DeepSeek model name (default: deepseek-flash)")
    parser.add_argument("--base-url", default=None, help="DeepSeek API base URL")
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default=None,
                        help="DeepSeek thinking mode (default: disabled -> cheaper)")
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"], default=None)
    parser.add_argument("--timeout", type=float, default=None, help="single API request timeout seconds")
    parser.add_argument("--image-scale", type=int, default=None, help="screenshot nearest-neighbour upscale (1..8)")
    # budget
    parser.add_argument("--max-usd", type=float, default=None, help="hard USD budget for this run")
    parser.add_argument("--max-usd-total", type=float, default=None, help="hard USD budget across all runs")
    parser.add_argument("--max-calls", type=int, default=None, help="max API calls for this run")
    parser.add_argument("--max-calls-per-hour", type=int, default=None, help="rate cap for API calls")
    # runtime
    parser.add_argument("--max-turns", type=int, default=None, help="stop after N DeepSeek decision turns (0=unlimited)")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="checkpoint every N action batches")
    parser.add_argument("--stop-file", default=None, help="stop safely when this file exists")
    parser.add_argument("--pause-file", default=None, help="pause while this file exists")
    parser.add_argument("--headless", action="store_true", help="no GUI; print status lines instead")
    parser.add_argument("--paused", action="store_true", help="start the run paused (viewer mode)")
    parser.add_argument("--video", default=None, help="optional MP4 recording path")
    parser.add_argument("--scripted", action="store_true",
                        help="SCRIPTED DRY RUN: deterministic test brain, no API calls (not acceptance evidence)")
    parser.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    return parser


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    direct = {
        "api.api_key_env": args.api_key_env,
        "api.model": args.model,
        "api.base_url": args.base_url,
        "api.thinking": args.thinking,
        "api.reasoning_effort": args.reasoning_effort,
        "api.timeout_s": args.timeout,
        "api.image_scale": args.image_scale,
        "budget.max_usd_per_run": args.max_usd,
        "budget.max_usd_total": args.max_usd_total,
        "budget.max_calls_per_run": args.max_calls,
        "budget.max_calls_per_hour": args.max_calls_per_hour,
        "emulator.rom": args.rom,
        "emulator.state": args.state,
        "runtime.run_dir": args.run_dir,
        "runtime.resume_from": args.resume_from,
        "runtime.max_turns": args.max_turns,
        "runtime.checkpoint_every_actions": args.checkpoint_every,
        "runtime.stop_file": args.stop_file,
        "runtime.pause_file": args.pause_file,
        "viewer.record_video": args.video,
    }
    for key, value in direct.items():
        if value is not None:
            overrides[key] = value
    if args.fresh:
        overrides["runtime.resume"] = False
    if args.headless:
        overrides["viewer.enabled"] = False
    if args.scripted:
        # A dry run is deterministic test data and must never pollute the real
        # agent's persistent memory or share its single-instance lock.
        overrides["memory.root"] = str(PROJECT_ROOT / "artifacts" / "llm_agent" / "scripted_memory")
        overrides["runtime.run_root"] = str(PROJECT_ROOT / "artifacts" / "llm_agent" / "scripted_runs")
        overrides["runtime.lock_path"] = str(PROJECT_ROOT / "artifacts" / "llm_agent" / "scripted_agent.lock")
    return overrides


def make_brain(cfg: AgentConfig, args: argparse.Namespace):
    if args.scripted:
        return ScriptedBrain()
    client = DeepSeekClient(cfg.api)
    return DeepSeekBrain(cfg, client)


def _print_key_help(cfg: AgentConfig) -> None:
    env_name = cfg.api.api_key_env
    print(
        "\n[blocked] 未检测到 DeepSeek API Key / DeepSeek API key is not configured.\n"
        f"  需要环境变量 {env_name}（或项目根目录的 .env 文件）。\n"
        "  请任选其一：\n"
        '    1) 在项目根目录创建 .env（参考 .env.example）：\n'
        f"         {env_name}=sk-你的密钥\n"
        f"    2) 或在 PowerShell 中设置： $env:{env_name} = 'sk-你的密钥'\n"
        "  API Key 不会被打印、写入日志或提交到 git。\n"
        "  配置完成后重新运行本命令即可；未配置密钥时也可用 --scripted 做本地流程自测。\n",
        flush=True,
    )


def print_run_summary(result: RunResult) -> None:
    print("\n=== DeepSeek LLM Agent — run finished ===", flush=True)
    print(f"  status      : {result.status} ({result.reason or 'ok'})", flush=True)
    print(f"  mode        : {result.mode}" + ("" if result.real_api else "  [SCRIPTED DRY RUN — not real API evidence]"), flush=True)
    print(f"  model       : {result.model or '-'}", flush=True)
    print(f"  run dir     : {result.run_dir}", flush=True)
    print(f"  duration    : {result.duration_s:.1f}s", flush=True)
    print(f"  turns       : {result.turns} (API calls {result.calls})", flush=True)
    print(f"  frames      : {result.frames}   buttons: {result.buttons}   batches: {result.action_batches}", flush=True)
    print(f"  cost        : ${result.cost_usd:.6f}   tokens: {result.total_tokens:,}", flush=True)
    print(f"  places      : {result.places}   discoveries: {result.discoveries}", flush=True)
    print(f"  goal        : {result.goal or '-'}", flush=True)
    print(f"  last action : {result.last_action or '-'}", flush=True)
    if result.run_dir:
        print(f"\n  report      : {Path(result.run_dir) / 'report.md'}", flush=True)
        print(
            "  audit logs  : audit/decisions.jsonl, audit/actions.jsonl, audit/api_attempts.jsonl",
            flush=True,
        )
        print(
            "  verify RAM progress (report-only):\n"
            f"    .\\.venv\\Scripts\\python.exe scripts\\llm_agent_progress.py --run-dir \"{result.run_dir}\"",
            flush=True,
        )
    print("", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = AgentConfig.load(args.config, env_file=args.env_file, overrides=build_overrides(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"[config error] {exc}", file=sys.stderr, flush=True)
        return EXIT_CONFIG
    if args.label:
        cfg.label = args.label

    if args.print_config:
        print(json.dumps(cfg.to_safe_dict(), ensure_ascii=False, indent=2), flush=True)
        return EXIT_OK

    problems = cfg.validate()
    if problems:
        print("[config error] configuration problems:", file=sys.stderr, flush=True)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr, flush=True)
        return EXIT_CONFIG

    if not args.scripted and cfg.api_key() is None:
        _print_key_help(cfg)
        return EXIT_NO_KEY

    if args.scripted:
        print(
            "\n" + "=" * 78 + "\n"
            "SCRIPTED DRY RUN — deterministic test brain, NO DeepSeek API calls.\n"
            "This exercises the emulator/tools/memory/audit pipeline only.\n"
            "It is NOT evidence of real API-driven gameplay and must not be reported as such.\n"
            "Isolated memory/run dirs: artifacts\\llm_agent\\scripted_memory, artifacts\\llm_agent\\scripted_runs\n"
            + "=" * 78 + "\n",
            flush=True,
        )

    print(f"[run] model={cfg.api.model} mode={'scripted' if args.scripted else 'deepseek'} "
          f"budget=${cfg.budget.max_usd_per_run:.2f} max_calls={cfg.budget.max_calls_per_run} "
          f"thinking={cfg.api.thinking} image_scale={cfg.api.image_scale}", flush=True)
    print(f"[run] rom={cfg.emulator.rom}", flush=True)
    print(f"[run] key={cfg.api_key_status()}", flush=True)

    brain = make_brain(cfg, args)

    viewer: AgentViewer | None = None
    if cfg.viewer.enabled:
        viewer = AgentViewer(cfg.viewer)
        if args.paused:
            viewer.pause_event.set()
            print("[run] starting paused — press Space in the viewer (or click 继续) to start.", flush=True)
    runtime = AgentRuntime(cfg, brain=brain, viewer=viewer)

    if viewer is None:
        try:
            result = runtime.run()
        except KeyboardInterrupt:
            runtime.request_stop("Ctrl+C")
            print("[run] interrupted, saving state...", flush=True)
            result = runtime.run()
        print_run_summary(result)
        return result.exit_code

    holder: dict[str, object] = {}

    def worker() -> None:
        try:
            holder["result"] = runtime.run()
        except BaseException as exc:  # noqa: BLE001 - GUI thread must be told
            holder["error"] = exc
            holder["traceback"] = traceback.format_exc()
            print(f"[run] fatal error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=worker, name="agent-runtime", daemon=True)
    thread.start()
    try:
        viewer.start()
    except KeyboardInterrupt:
        runtime.request_stop("Ctrl+C")
        viewer.request_shutdown()
    finally:
        runtime.request_stop("viewer closed")
        thread.join(timeout=180)
        if thread.is_alive():
            print("[run] warning: runtime did not finish within 180s after window close", file=sys.stderr, flush=True)
    if "result" in holder:
        result = holder["result"]
        assert isinstance(result, RunResult)
        print_run_summary(result)
        return result.exit_code
    if "traceback" in holder:
        print(holder["traceback"], file=sys.stderr, flush=True)
    error = holder.get("error")
    print(f"[run] no result available ({type(error).__name__ if error else 'unknown error'})", file=sys.stderr, flush=True)
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
