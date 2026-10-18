"""VERIFICATION-ONLY progress report for a DeepSeek LLM agent run.

The agent's decision loop never reads hidden RAM.  This separate tool loads a
saved emulator checkpoint in a throw-away PyBoy and reports the true in-game
progress (map id, coordinates, badges, party levels) so that the run report can
be independently verified.  It is NOT part of the agent loop.

Examples:
    .\\.venv\\Scripts\\python.exe scripts\\llm_agent_progress.py --run-dir artifacts\\llm_agent\\runs\\<run>
    .\\.venv\\Scripts\\python.exe scripts\\llm_agent_progress.py --state path\\to\\checkpoint_000000123.state
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

from llm_agent.config import AgentConfig, ROOT
from llm_agent.progress import ProgressSnapshot, compare_progress, probe_from_state_file


def resolve_state_path(path: Path) -> Path:
    """Accept a .state file, a checkpoint .json, or a run directory."""
    path = path.resolve()
    if path.is_file() and path.suffix == ".state":
        return path
    if path.is_file() and path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"cannot parse checkpoint json {path}: {exc}")
        meta = payload.get("meta", payload)
        state_file = meta.get("state_file")
        if not state_file:
            raise SystemExit(f"checkpoint json has no state_file: {path}")
        candidate = Path(state_file)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if not candidate.is_file():
            candidate = path.parent / "states" / Path(state_file).name
        return candidate.resolve()
    if path.is_dir():
        for candidate in (path / "checkpoints" / "latest.json", path / "latest.json"):
            if candidate.is_file():
                return resolve_state_path(candidate)
        states = sorted((path / "checkpoints" / "states").glob("*.state")) if (path / "checkpoints" / "states").is_dir() else []
        if states:
            return states[-1].resolve()
    raise SystemExit(f"no checkpoint state found at {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", help="run directory containing checkpoints/latest.json")
    parser.add_argument("--state", help="explicit .state file or checkpoint json")
    parser.add_argument("--rom", help="ROM path (default: configured local ROM)")
    parser.add_argument("--initial-state", help="initial state for comparison (default: configured initial state)")
    parser.add_argument("--json", dest="json_out", help="optional path to write the full comparison JSON")
    args = parser.parse_args(argv)

    cfg = AgentConfig.load(env_file="")
    rom = Path(args.rom or cfg.emulator.rom)
    if not rom.is_file():
        print(f"[error] ROM not found: {rom}", file=sys.stderr)
        return 2

    if args.state:
        state_path = resolve_state_path(Path(args.state))
    elif args.run_dir:
        state_path = resolve_state_path(Path(args.run_dir))
    else:
        parser.error("pass --run-dir or --state")
        return 2

    final = probe_from_state_file(rom, state_path)
    initial_path = Path(args.initial_state or cfg.emulator.state)
    initial = None
    if initial_path.is_file():
        initial = probe_from_state_file(rom, initial_path)

    print("=== VERIFICATION-ONLY RAM progress probe (never fed to the agent) ===")
    print(f"rom          : {rom}")
    print(f"final state  : {state_path}")
    print(f"final        : {final.summary()}")
    if initial is not None:
        print(f"initial      : {initial.summary()}")
    payload: dict[str, object] = {"rom": str(rom), "final_state": str(state_path), "final": final.to_dict(),
                                  "initial": initial.to_dict() if initial else None}
    if initial is not None:
        comparison = compare_progress(initial, final)
        payload["comparison"] = comparison
        print("--- comparison -------------------------------------------------")
        print(f"map changed  : {comparison['map_changed']}  (map_id {initial.map_id} -> {final.map_id})")
        print(f"badges       : {initial.badges} -> {final.badges} (gained {comparison['badges_gained']})")
        print(f"party levels : {initial.party_levels} -> {final.party_levels} (gained {comparison['levels_gained']})")
        print(f"party size   : {initial.party_count} -> {final.party_count}")
        print(f"coords       : ({initial.x},{initial.y}) -> ({final.x},{final.y})")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json written : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
