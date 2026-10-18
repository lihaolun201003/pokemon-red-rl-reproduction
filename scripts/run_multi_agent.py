"""Launch the Pokémon Red Multi-Agent v1 system (Game Director + 3 agents).

The director is a deterministic Python scheduler; Navigation v2 keeps doing all
low-level navigation, the Story agent reads real dialogue screenshots, and the
Pokémon agent plays real battles — each one holding the single control lease in
turn.  DeepSeek is only called by the specialised agents and (rarely) by the
director for high-level coordination.

Examples
--------
Default config, headless, until the configured limits are reached:
    .\\.venv\\Scripts\\python.exe scripts\\run_multi_agent.py --headless

Short bounded acceptance-style run (real API, real ROM):
    .\\.venv\\Scripts\\python.exe scripts\\run_multi_agent.py --headless \
        --max-decision-turns 60 --max-usd 0.30 --max-runtime-minutes 20

Inspect the resolved configuration without running:
    .\\.venv\\Scripts\\python.exe scripts\\run_multi_agent.py --print-config

The real API key is read from the environment or the local ``.env`` file and is
never printed or committed.  Without a key the run refuses to start (exit 4):
the multi-agent system is validated only against the real DeepSeek API.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


_configure_stdio()

from llm_agent.config import AgentConfig  # noqa: E402
from llm_agent.multi_agent.integration import MultiAgentResult, MultiAgentSession  # noqa: E402
from llm_agent.viewer import AgentViewer  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_BLOCKED = 3
EXIT_NO_KEY = 4

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "multi_agent_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pokémon Red Multi-Agent v1 — Game Director + Navigation/Story/Pokémon agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="multi-agent JSON config")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"), help="local .env with DEEPSEEK_API_KEY")
    parser.add_argument("--rom", default=None, help="lawful English ROM path")
    parser.add_argument("--state", default=None, help="official initial emulator state path")
    parser.add_argument("--run-dir", default=None, help="explicit run directory")
    parser.add_argument("--resume-from", default=None, help="previous run dir to resume the latest checkpoint from")
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoints / stored maps")
    parser.add_argument("--label", default=None, help="human readable run label")
    parser.add_argument("--evidence-root", default=None, help="artifacts/multi_agent_v1/<gn> evidence directory")
    # models / API
    parser.add_argument("--api-key-env", default=None, help="env var holding the API key")
    parser.add_argument("--model", default=None, help="global DeepSeek model")
    parser.add_argument("--director-model", default=None, help="model for the director's rare reviews")
    parser.add_argument("--story-model", default=None, help="model for the Story agent")
    parser.add_argument("--pokemon-model", default=None, help="model for the Pokémon agent")
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default=None)
    parser.add_argument("--image-scale", type=int, default=None, help="screenshot upscale for agent vision")
    parser.add_argument("--max-tokens-agent", type=int, default=None)
    # budgets / limits
    parser.add_argument("--max-usd", type=float, default=None, help="hard USD cap for this run")
    parser.add_argument("--max-usd-total", type=float, default=None, help="hard USD cap across all runs")
    parser.add_argument("--max-calls", type=int, default=None, help="hard API call cap for this run")
    parser.add_argument("--max-decision-turns", type=int, default=None, help="stop after N director steps")
    parser.add_argument("--max-runtime-minutes", type=float, default=None)
    parser.add_argument("--max-emulator-actions", type=int, default=None)
    parser.add_argument("--max-rollbacks", type=int, default=None, help="total rollback cap")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="checkpoint every N agent actions")
    parser.add_argument("--stop-file", default=None, help="stop safely when this file appears")
    parser.add_argument("--pause-file", default=None, help="pause while this file exists")
    parser.add_argument("--no-director-llm", action="store_true", help="disable the director's DeepSeek reviews")
    parser.add_argument("--max-usd-story", type=float, default=None)
    parser.add_argument("--max-usd-pokemon", type=float, default=None)
    # runtime
    parser.add_argument("--headless", action="store_true", help="no GUI (default for unattended runs)")
    parser.add_argument("--viewer", action="store_true", help="show the existing Tk viewer")
    parser.add_argument("--video", default=None, help="optional MP4 recording path (viewer mode)")
    parser.add_argument("--paused", action="store_true", help="start paused (viewer mode)")
    parser.add_argument("--no-warmup", action="store_true", help="skip the initial API probe (debug only)")
    parser.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    return parser


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    direct = {
        "api.api_key_env": args.api_key_env,
        "api.model": args.model,
        "api.thinking": args.thinking,
        "api.image_scale": args.image_scale,
        "budget.max_usd_per_run": args.max_usd,
        "budget.max_usd_total": args.max_usd_total,
        "budget.max_calls_per_run": args.max_calls,
        "emulator.rom": args.rom,
        "emulator.state": args.state,
        "runtime.run_dir": args.run_dir,
        "runtime.resume_from": args.resume_from,
        "runtime.stop_file": args.stop_file,
        "runtime.pause_file": args.pause_file,
        "viewer.record_video": args.video,
        "multi_agent.enabled": True,
        "multi_agent.max_decision_turns": args.max_decision_turns,
        "multi_agent.max_runtime_minutes": args.max_runtime_minutes,
        "multi_agent.max_emulator_actions": args.max_emulator_actions,
        "multi_agent.max_rollbacks_total": args.max_rollbacks,
        "multi_agent.checkpoint_every_actions": args.checkpoint_every,
        "multi_agent.max_tokens_agent": args.max_tokens_agent,
        "multi_agent.image_scale_agent": args.image_scale,
        "multi_agent.evidence_root": args.evidence_root,
        "multi_agent.director_model": args.director_model,
        "multi_agent.story_model": args.story_model,
        "multi_agent.pokemon_model": args.pokemon_model,
        "multi_agent.max_usd_story": args.max_usd_story,
        "multi_agent.max_usd_pokemon": args.max_usd_pokemon,
    }
    for key, value in direct.items():
        if value is not None:
            overrides[key] = value
    if args.fresh:
        overrides["runtime.resume"] = False
    if args.no_director_llm:
        overrides["multi_agent.director_llm_enabled"] = False
    if args.headless or not args.viewer:
        overrides["viewer.enabled"] = False
    else:
        overrides["viewer.enabled"] = True
    return overrides


def print_summary(result: MultiAgentResult) -> None:
    print("\n=== Pokémon Red Multi-Agent v1 — run finished ===", flush=True)
    print(f"  status        : {result.status} ({result.reason or 'ok'})", flush=True)
    print(f"  run dir       : {result.run_dir}", flush=True)
    print(f"  duration      : {result.duration_s:.1f}s", flush=True)
    print(f"  branch        : {result.branch_id}", flush=True)
    print(f"  director steps: {result.steps}   owner switches: {result.owner_switches}", flush=True)
    print(f"  battles       : {result.battles}   dialogs: {result.dialogs}", flush=True)
    print(f"  tasks         : {result.tasks_created} created / {result.tasks_completed} completed / {result.tasks_failed} failed", flush=True)
    print(f"  checkpoints   : {result.checkpoints}   rollbacks: {result.rollbacks}   stagnation: {result.stagnation_events}", flush=True)
    print(f"  api           : {result.api_calls} calls, ${result.cost_usd:.6f}, {result.total_tokens:,} tokens "
          f"(director reviews: {result.director_llm_calls})", flush=True)
    print(f"  area          : {result.current_area or '-'}", flush=True)
    print(f"  buttons/frames: {result.buttons} / {result.frames}", flush=True)
    if result.run_dir:
        print(f"\n  result json   : {Path(result.run_dir) / 'multi_agent_result.json'}", flush=True)
        print(f"  audit         : {Path(result.run_dir) / 'audit'}", flush=True)
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
    if cfg.api_key() is None:
        print(
            f"\n[blocked] {cfg.api.api_key_env} is not configured. The multi-agent system is only "
            "validated against the real DeepSeek API; no mock run is reported as acceptance evidence.\n"
            "Create a local .env file (see .env.example) with your key and rerun.\n",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_NO_KEY

    print(f"[run] multi-agent model={cfg.api.model} "
          f"director={cfg.multi_agent.director_model or cfg.api.model} "
          f"story={cfg.multi_agent.story_model or cfg.api.model} "
          f"pokemon={cfg.multi_agent.pokemon_model or cfg.api.model}", flush=True)
    print(f"[run] budget=${cfg.budget.max_usd_per_run:.2f} max_calls={cfg.budget.max_calls_per_run} "
          f"max_steps={cfg.multi_agent.director_max_steps} "
          f"max_runtime={cfg.multi_agent.max_runtime_minutes}min", flush=True)
    print(f"[run] rom={cfg.emulator.rom}", flush=True)
    print(f"[run] key={cfg.api_key_status()}", flush=True)

    viewer: AgentViewer | None = None
    if cfg.viewer.enabled:
        viewer = AgentViewer(cfg.viewer)
        if args.paused:
            viewer.pause_event.set()
            print("[run] starting paused — press Space in the viewer to start.", flush=True)

    session = MultiAgentSession(
        cfg,
        viewer_bridge=viewer,
        warmup=not args.no_warmup,
        use_lock=True,
    )
    result: MultiAgentResult | None = None
    error_text = ""
    import threading

    holder: dict[str, object] = {}

    def worker() -> None:
        try:
            holder["result"] = session.run()
        except BaseException:  # noqa: BLE001 - the GUI thread must be told
            holder["traceback"] = traceback.format_exc()
            print(f"[run] fatal error:\n{holder['traceback']}", file=sys.stderr, flush=True)

    if viewer is None:
        try:
            result = session.run()
        except KeyboardInterrupt:
            session.request_stop("Ctrl+C")
            result = session.last_result
        except Exception:  # noqa: BLE001 - already normalised by session.run
            error_text = traceback.format_exc()
            print(error_text, file=sys.stderr, flush=True)
    else:
        thread = threading.Thread(target=worker, name="multi-agent", daemon=True)
        thread.start()
        try:
            viewer.start()
        except KeyboardInterrupt:
            session.request_stop("Ctrl+C")
            viewer.request_shutdown()
        finally:
            session.request_stop("viewer closed")
            thread.join(timeout=240)
        result = holder.get("result")  # type: ignore[assignment]

    if result is None:
        print(f"[run] no result available. {error_text}", file=sys.stderr, flush=True)
        return EXIT_ERROR
    print_summary(result)
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
