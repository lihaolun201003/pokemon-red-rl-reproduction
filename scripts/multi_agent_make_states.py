"""Create legal PyBoy save states for the Multi-Agent v1 acceptance scenarios.

TEST SETUP ONLY — these states are produced by legal in-game button presses on the
real emulator (no RAM writes, no ROM modification, no walkthrough data).  A
read-only RAM *read* is used only to label/verify the created state; it is never
imported by any agent or decision-loop module.

Scenario states
---------------
``g2_dialog_ready.state``
    Player standing in Oak's Lab, facing an interactive NPC/object so that a
    single ``A`` press opens a **real dialogue box**.  Used by G2 (real dialogue
    takeover / Story Agent).

``g3_battle_ready.state``
    Player standing in tall grass on Route 1, one step before a wild encounter.
    Used by G3 (real battle takeover / Pokémon Agent).

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\multi_agent_make_states.py --g2
    .\\.venv\\Scripts\\python.exe scripts\\multi_agent_make_states.py --g3
    .\\.venv\\Scripts\\python.exe scripts\\multi_agent_make_states.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_agent.config import AgentConfig
from llm_agent.emulator import GameBoyEmulator
from llm_agent.navigation.navigation_agent import NavigationAgent
from llm_agent.observation import analyze, save_png

DEFAULT_STATE = PROJECT_ROOT / "local" / "has_pokedex_nballs.state"
DEFAULT_OUT = PROJECT_ROOT / "work" / "multi_agent_v1" / "states"
WORK_STATES = PROJECT_ROOT / "work" / "nav_v2" / "states"

#: TEST SETUP ONLY: read directly from the emulator memory to label the state.
_RED_MAP_LABELS = {0: "town_outdoor", 12: "route_1", 40: "lab_interior"}

#: Deterministic legal walk used to find a dialogue-triggering position.
#: (Each entry is one button press; the probe presses A after each.)
G2_WALK = (
    ["up", "up", "up", "left", "left", "left", "left", "down", "down", "down",
     "right", "right", "right", "right", "up", "up", "left", "left", "down",
     "down", "right", "right", "up", "up", "up", "left", "left", "left",
     "down", "down", "down", "right", "right", "right", "right", "up", "up"]
    + ["up", "down", "left", "right", "up", "left", "down", "right"] * 8
)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


_configure_stdio()


def _ram_location(emulator: GameBoyEmulator) -> dict[str, int | str]:
    """TEST SETUP ONLY: read map/x/y from the emulator for labelling/verification."""
    try:  # pragma: no cover - only valid on a real PyBoy instance
        pyboy = getattr(emulator, "_pyboy", None)
        if pyboy is None:
            return {}
        mem = pyboy.memory
        map_id = int(mem[0xD35E])
        return {
            "map_id": map_id,
            "map_label": _RED_MAP_LABELS.get(map_id, f"map_{map_id}"),
            "x": int(mem[0xD362]),
            "y": int(mem[0xD361]),
        }
    except Exception:  # noqa: BLE001 - setup tooling must never crash on this
        return {}


def _save_state(emulator: GameBoyEmulator, path: Path, payload: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if payload is not None else emulator.save_state_bytes())
    return path


# --------------------------------------------------------------------------- #
# G2: stand facing an interactive object and open a real dialogue box
# --------------------------------------------------------------------------- #

def make_g2_state(*, out_dir: Path) -> dict:
    cfg = AgentConfig()
    start = WORK_STATES / "lab_start.state"
    emulator = GameBoyEmulator(cfg.emulator).start(state_path=start if start.is_file() else None)
    print(f"[g2] start ram={_ram_location(emulator)}", flush=True)
    log: list[dict] = []
    result: dict = {"found": False}
    try:
        for index, direction in enumerate(G2_WALK):
            try:
                emulator.press(direction)
                pre_a = emulator.save_state_bytes()  # exact state before the A press
                frame = emulator.press("a")
                analysis = analyze(frame, None)
            except Exception as exc:  # noqa: BLE001
                log.append({"step": index, "direction": direction,
                            "error": f"{type(exc).__name__}: {exc}"})
                continue
            entry = {"step": index, "direction": direction, "event": analysis.event.value}
            log.append(entry)
            if analysis.event.value == "dialog":
                state_path = _save_state(emulator, out_dir / "g2_dialog_ready.state", pre_a)
                shot = out_dir / "g2_dialog_open.png"
                save_png(frame, shot)
                result = {
                    "found": True,
                    "state": str(state_path),
                    "screenshot": str(shot),
                    "direction": direction,
                    "step": index,
                    "location": _ram_location(emulator),
                    "created_at": time.time(),
                }
                print(f"[g2] FOUND real dialogue: {state_path} ({result['location']})", flush=True)
                break
            if analysis.event.value == "menu":
                # interactive object but opens a menu; close and keep probing
                try:
                    emulator.press("b")
                except Exception:  # noqa: BLE001
                    pass
            if index % 20 == 0:
                print(f"  [g2] step {index} dir={direction} event={analysis.event.value}", flush=True)
    finally:
        emulator.close()
    result["walk"] = log[-40:]
    return result


# --------------------------------------------------------------------------- #
# G3: reach the tall grass and stop right before a wild battle
# --------------------------------------------------------------------------- #

def _explore_until_done(nav: NavigationAgent, *, rounds: int = 6, max_steps: int = 200) -> dict:
    last: dict = {}
    for _ in range(rounds):
        report = nav.explore_current_area(max_steps=max_steps, max_seconds=240)
        last = {"status": str(getattr(report, "status", "")), "steps": int(getattr(report, "steps", 0) or 0)}
        print(f"  [explore] {last}", flush=True)
        if last["status"] in ("completed", "blocked", "error", "paused"):
            break
    return last


def _encounter_hunt(emulator: GameBoyEmulator, *, max_steps: int = 160) -> dict:
    """Walk in tall grass until a wild battle starts; return the pre-battle state."""
    pattern = ["up", "up", "down", "down", "left", "left", "right", "right"]
    for index in range(max_steps):
        direction = pattern[index % len(pattern)]
        pre = emulator.save_state_bytes()
        try:
            frame = emulator.press(direction)
            analysis = analyze(frame, None)
        except Exception as exc:  # noqa: BLE001
            return {"found": False, "error": f"{type(exc).__name__}: {exc}"}
        if analysis.event.value == "battle":
            return {"found": True, "pre_state": pre, "frame": frame, "step": index, "direction": direction}
    return {"found": False, "reason": "no encounter within budget"}


def make_g3_state(*, out_dir: Path, max_rounds: int = 30, round_steps: int = 140) -> dict:
    cfg = AgentConfig()
    start_state = WORK_STATES / "pallet_west.state"
    if not start_state.is_file():
        start_state = DEFAULT_STATE
    emulator = GameBoyEmulator(cfg.emulator).start(state_path=start_state)
    world_root = PROJECT_ROOT / "work" / "multi_agent_v1" / "g3_setup_world"
    nav = NavigationAgent(emulator=emulator, world_root=world_root, config=cfg)
    nav.attach()
    print(f"[g3] start ram={_ram_location(emulator)}", flush=True)
    started = time.time()
    result: dict = {"found": False, "rounds": []}
    try:
        for round_index in range(max_rounds):
            if time.time() - started > 780:
                result["reason"] = "time limit"
                break
            location = _ram_location(emulator)
            nav_status = nav.status()
            pre_round = emulator.save_state_bytes()  # state before this explore round
            current_event = analyze(emulator.observe(), None).event.value
            print(f"[g3] round {round_index} area={nav_status.get('area_id')} "
                  f"areas={nav_status.get('areas_total')} ram={location} event={current_event}", flush=True)
            if current_event == "battle":
                result["reason"] = "started a round already inside a battle; stopping scouting"
                break
            outside = location.get("map_id") not in (None, 0, 40) or location.get("map_label") == "route_1"
            if outside or int(nav_status.get("areas_total") or 0) >= 3:
                # In the fields: hunt for a wild encounter from a known no-battle state.
                hunt = _encounter_hunt(emulator, max_steps=140)
                result["rounds"].append({"round": round_index, "phase": "encounter_hunt",
                                         "found": hunt.get("found"), "step": hunt.get("step"),
                                         "location": _ram_location(emulator)})
                print(f"  [g3] encounter hunt: found={hunt.get('found')} step={hunt.get('step')}", flush=True)
                if hunt.get("found"):
                    state_path = _save_state(emulator, out_dir / "g3_battle_ready.state", hunt["pre_state"])
                    shot = out_dir / "g3_battle_ready.png"
                    save_png(hunt["frame"], shot)
                    result.update({
                        "found": True, "state": str(state_path), "screenshot": str(shot),
                        "location": _ram_location(emulator), "round": round_index,
                        "created_at": time.time(),
                    })
                    print(f"[g3] FOUND battle-ready grass state: {state_path} "
                          f"({result['location']})", flush=True)
                    break
                # keep exploring new ground after an unsuccessful hunt
            report = nav.explore_current_area(max_steps=round_steps, max_seconds=240)
            analysis = analyze(emulator.observe(), None)
            result["rounds"].append({
                "round": round_index,
                "status": str(getattr(report, "status", "")),
                "area": nav.status().get("area_id"),
                "areas_total": nav.status().get("areas_total"),
                "event": analysis.event.value,
                "location": _ram_location(emulator),
            })
            print(f"  [g3] explore {result['rounds'][-1]}", flush=True)
            if analysis.event.value == "battle":
                # A wild battle started during this round: the state saved at the
                # start of the round is a legal, battle-free grass state.
                state_path = _save_state(emulator, out_dir / "g3_battle_ready.state", pre_round)
                shot = out_dir / "g3_battle_ready.png"
                save_png(emulator.observe(), shot)
                result.update({
                    "found": True, "state": str(state_path), "screenshot": str(shot),
                    "location": _ram_location(emulator), "round": round_index,
                    "note": "state captured at the start of the round that triggered the battle",
                    "created_at": time.time(),
                })
                print(f"[g3] battle appeared in round {round_index}; saved pre-round grass state "
                      f"{state_path} ({result['location']})", flush=True)
                break
    finally:
        try:
            nav.save()
        except Exception:  # noqa: BLE001
            pass
        emulator.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create acceptance save states (test setup only).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--g2", action="store_true")
    parser.add_argument("--g3", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=30)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"out_dir": str(out_dir), "created_at": time.time()}
    do_g2 = args.g2 or args.all
    do_g3 = args.g3 or args.all
    if not (do_g2 or do_g3):
        parser.print_help()
        return 2
    if do_g2:
        print("[g2] creating dialogue-ready state …", flush=True)
        report["g2"] = make_g2_state(out_dir=out_dir)
    if do_g3:
        print("[g3] creating battle-ready state …", flush=True)
        report["g3"] = make_g3_state(out_dir=out_dir, max_rounds=args.max_rounds)
    (out_dir / "make_states_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str), flush=True)
    ok = all(bool((report.get(key) or {}).get("found")) for key in ("g2", "g3") if key in report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
