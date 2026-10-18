"""G4 — bounded unattended long exploration on the real game (multi-area).

Runs the navigation subsystem autonomously for a bounded budget.  There is **no
LLM in the loop**: the Python explorer decides targets, plans with A*, verifies
every step visually and keeps the permanent world model.  When a room is covered
it walks to a known exit, crosses into the next area, and keeps exploring there.
The harness measures honest statistics (effective moves, new cells, revisit rate,
blocked attempts, planning failures, localization drift, relocalization success,
per-area coverage) and saves screenshots/map snapshots as evidence.

This is the *navigation-level* long-run test; the DeepSeek integration is
validated separately by the real-API run (G4-API) with the same navigation tools.

Usage:
  .venv\\Scripts\\python.exe scripts\\nav_v2_g4_long_explore.py --max-total-steps 1200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nav_v2_common as common  # noqa: E402
from llm_agent.navigation.types import ExploreStatus  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G4: bounded unattended multi-area exploration")
    parser.add_argument("--state", default="lab_start.state")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "nav_v2" / "g4"))
    parser.add_argument("--max-total-steps", type=int, default=1200)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--round-steps", type=int, default=260)
    parser.add_argument("--round-seconds", type=float, default=150.0)
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--no-fresh", action="store_true")
    return parser.parse_args(argv)


def area_cell_counts(agent) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        world = agent.world_summary()
        for area in world.get("areas") or []:
            counts[str(area.get("area_id"))] = int(area.get("cells") or 0)
    except Exception:  # noqa: BLE001
        pass
    return counts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common.utf8_stdout()
    started = time.time()
    harness = common.build_harness(args.out, state=args.state, fresh=not args.no_fresh,
                                   label="g4", screen_every=15, map_every=60)
    agent, emu, rec = harness.agent, harness.emu, harness.recorder
    initial_truth = common.ram_truth(emu)
    initial_counts = area_cell_counts(agent)
    initial_areas = set(initial_counts)
    print(f"[g4] start RAM map={initial_truth.map_id} coords={initial_truth.x},{initial_truth.y}")

    totals = {"steps": 0, "moves": 0, "blocked": 0, "turned": 0, "unknown": 0,
              "transitions": 0, "areas_created": 0, "explore_rounds": 0,
              "navigate_calls": 0, "navigate_reached": 0, "planning_failures": 0}
    rounds: list[dict] = []
    area_segments: list[dict] = []
    relocalized_areas: list[str] = []
    anchor = common.anchor_now(agent, emu)
    current_area = str(agent.status().get("area_id") or "")
    segment_steps = 0
    budget_steps = max(1, int(args.max_total_steps))
    deadline = started + max(30.0, float(args.max_seconds))
    final_reason = ""
    interrupted_event: dict | None = None

    def close_segment(reason: str) -> None:
        nonlocal anchor, current_area, segment_steps
        # Drift can only be verified while the RAM map stays the same; a segment
        # closed by an area transition changes the RAM coordinate frame entirely.
        truth_now = common.ram_truth(emu)
        same_ram_map = int(truth_now.map_id) == int((anchor.get("truth") or {}).get("map_id", -999))
        area_now = str(agent.status().get("area_id") or "")
        same_area = area_now == str(anchor.get("area_id") or "")
        verifiable = (
            reason not in ("transition", "exit_transition") and same_ram_map and same_area
        )
        drift = common.drift(agent, emu, anchor) if verifiable else -1
        area_segments.append({
            "area_id": current_area,
            "steps": segment_steps,
            "drift": drift,
            "verifiable": bool(verifiable),
            "closed_by": reason,
            "at_step": totals["steps"],
        })
        anchor = common.anchor_now(agent, emu)
        segment_steps = 0

    for round_index in range(max(1, int(args.max_rounds))):
        if totals["steps"] >= budget_steps:
            final_reason = f"step budget {budget_steps} exhausted"
            break
        if time.time() >= deadline:
            final_reason = "wall-clock budget exhausted"
            break
        remaining = budget_steps - totals["steps"]
        before_counts = area_cell_counts(agent)
        known_areas_before = set(before_counts)
        report = agent.explore_current_area(
            max_steps=min(int(args.round_steps), remaining),
            max_seconds=min(float(args.round_seconds), max(10.0, deadline - time.time())),
        )
        payload = rec.on_report(report, phase=f"g4_explore_{round_index}")
        rounds.append(payload)
        totals["explore_rounds"] += 1
        for key in ("moves", "blocked", "turned", "unknown"):
            totals[key] += common.metric_int(payload, key)
        round_steps = common.metric_int(payload, "steps")
        totals["steps"] += round_steps
        segment_steps += round_steps
        area_after = str(payload.get("area_id") or "")
        if report.area_switched:
            totals["transitions"] += 1
            close_segment("transition")
            new_area = str(payload.get("new_area_id") or "")
            if new_area and new_area not in before_counts:
                totals["areas_created"] += 1
            elif new_area:
                relocalized_areas.append(new_area)
            current_area = str(agent.status().get("area_id") or new_area or current_area)
            anchor = common.anchor_now(agent, emu)
            print(f"[g4] round {round_index}: transitioned {area_after} -> {new_area} "
                  f"(steps={report.steps})")
            continue
        print(f"[g4] round {round_index}: status={report.status.value} steps={report.steps} "
              f"moves={report.moves} new_cells={report.new_cells} area={area_after}")
        if report.status is ExploreStatus.COMPLETED:
            close_segment("completed")
            # Go and find a known exit to open a new area.
            nav = agent.navigate_to({"exit": True}, max_steps=180)
            nav_payload = rec.on_report(nav, phase=f"g4_exit_{round_index}")
            rounds.append(nav_payload)
            totals["navigate_calls"] += 1
            totals["steps"] += int(nav.steps or 0)
            segment_steps += int(nav.steps or 0)
            if nav.reached:
                totals["navigate_reached"] += 1
            if nav.area_switched:
                totals["transitions"] += 1
                close_segment("exit_transition")
                exit_area = str((nav_payload or {}).get("new_area_id") or "")
                if exit_area and exit_area in known_areas_before:
                    relocalized_areas.append(exit_area)
                current_area = str(agent.status().get("area_id") or "")
                anchor = common.anchor_now(agent, emu)
                continue
            if nav.status in (ExploreStatus.BLOCKED, ExploreStatus.NO_PROGRESS,
                              ExploreStatus.LOCALIZATION_LOST, ExploreStatus.ERROR):
                totals["planning_failures"] += 1
            final_reason = (
                "current area fully explored and no further exit could be reached: "
                + str(nav.reason or nav.status.value)
            )
            break
        if report.status is ExploreStatus.PAUSED:
            # The controller pauses itself after non-navigation events; for the
            # bounded unattended run we resume unless an event asked us to stop.
            resume = getattr(agent, "resume", None)
            if callable(resume):
                try:
                    resume()
                except Exception:  # noqa: BLE001
                    pass
            common.wait_stable(emu, max_frames=120)
            continue
        if report.status is ExploreStatus.INTERRUPTED_EVENT:
            interrupted_event = None if report.event is None else report.event.to_dict()
            final_reason = report.reason or "non-navigation event"
            close_segment("event")
            break
        if report.status is ExploreStatus.TRANSITION:
            # report.area_switched was False but a transition estimate happened;
            # settle and relocalize/attach, then continue from the new situation.
            common.wait_stable(emu, max_frames=360)
            agent.attach()
            close_segment("transition_attach")
            current_area = str(agent.status().get("area_id") or current_area)
            anchor = common.anchor_now(agent, emu)
            continue
        if report.status in (ExploreStatus.LOCALIZATION_LOST, ExploreStatus.BLOCKED,
                             ExploreStatus.ERROR):
            totals["planning_failures"] += 1
            final_reason = report.reason or report.status.value
            previous_closed_by = area_segments[-1].get("closed_by") if area_segments else ""
            close_segment("blocked")
            common.wait_stable(emu, max_frames=240)
            attach = getattr(agent, "attach", None)
            if callable(attach):
                try:
                    attach()
                except Exception:  # noqa: BLE001
                    pass
            # allow one recovery attempt; stop only on two consecutive failures
            if previous_closed_by == "blocked":
                break
            anchor = common.anchor_now(agent, emu)
            continue
        if report.status in (ExploreStatus.STEP_LIMIT, ExploreStatus.TIME_LIMIT,
                             ExploreStatus.NO_PROGRESS):
            final_reason = report.reason or report.status.value
            continue
    close_segment("final")

    # -- statistics -------------------------------------------------------- #
    final_status = agent.status()
    final_truth = common.ram_truth(emu)
    world = agent.world_summary()
    final_counts = area_cell_counts(agent)
    new_cells_total = sum(max(0, final_counts.get(aid, 0) - initial_counts.get(aid, 0))
                          for aid in final_counts)
    # revisit rate from the recorded navigation steps (area + cell).
    visits: Counter = Counter()
    moves = revisits = 0
    results = Counter()
    unknown_steps = 0
    for step in rec.steps:
        motion = step.get("motion") or {}
        result = str(motion.get("result") or "")
        results[result] += 1
        if result == "unknown":
            unknown_steps += 1
        if result != "moved":
            continue
        moves += 1
        key = (str(step.get("area_id") or ""), tuple(step.get("pose_after") or ()))
        visits[key] += 1
        if visits[key] > 1:
            revisits += 1
    coverage_rows = []
    for area in world.get("areas") or []:
        coverage_rows.append({
            "area_id": area.get("area_id"),
            "kind": area.get("kind"),
            "cells": area.get("cells"),
            "walkable": area.get("walkable"),
            "coverage_ratio": area.get("coverage_ratio"),
            "visits": area.get("visits"),
            "explored": area.get("explored"),
        })
    drift_values = [item.get("drift") for item in area_segments if (item.get("drift") or 0) >= 0]
    relocalized = len(relocalized_areas)
    transitions = int(totals["transitions"])
    montage = common.screenshot_montage(
        harness.out_dir, harness.out_dir / "screens_montage.png", cols=5, scale=2, limit=30
    )
    gallery = common.run_map_gallery(harness.out_dir / "navigation",
                                     harness.out_dir / "map_gallery")
    elapsed = time.time() - started
    effective_ratio = (moves / totals["steps"]) if totals["steps"] else 0.0
    unknown_rate = (unknown_steps / moves) if moves else 0.0
    stopped_honestly = final_reason != "" and not final_reason.lower().startswith("too many consecutive")
    passed = bool(
        moves >= 50
        and new_cells_total >= 200
        and int(world.get("areas_total") or 0) >= 2
        and max(drift_values or [0]) == 0
        and unknown_rate <= 0.15
        and int(totals["planning_failures"]) <= 2
    )
    payload = {
        "task": "navigation_v2_g4_long_exploration",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state": str(harness.state),
        "elapsed_s": round(elapsed, 2),
        "passed": passed,
        "final_reason": final_reason,
        "interrupted_event": interrupted_event,
        "totals": {
            **totals,
            "direction_ops_recorded": len(rec.steps),
            "area_segments": len(area_segments),
            "new_cells": new_cells_total,
            "free_cells_walkable": sum(int(row.get("walkable") or 0) for row in coverage_rows),
            "motion_results": dict(results),
            "revisit_rate": round(revisits / moves, 4) if moves else 0.0,
            "effective_move_ratio": round(effective_ratio, 4),
            "unknown_rate": round(unknown_rate, 4),
            "stopped_honestly": bool(stopped_honestly),
            "valid_drift_segments": len(drift_values),
        },
        "deepseek_api": {"calls": 0, "cost_usd": 0.0,
                         "note": "this run is navigation-only (no LLM in the loop); the "
                                 "real-API integration run is reported separately as G4-API"},
        "coverage": coverage_rows,
        "localization": {
            "drift_samples": drift_values,
            "drift_max": max(drift_values or [0]),
            "unknown_steps": unknown_steps,
            "relocalization_events": relocalized,
            "transitions": transitions,
            "relocalization_success_ratio": (relocalized / transitions) if transitions else None,
            "ram_truth_start": {"map_id": initial_truth.map_id, "x": initial_truth.x, "y": initial_truth.y},
            "ram_truth_end": {"map_id": final_truth.map_id, "x": final_truth.x, "y": final_truth.y},
            "ram_note": "RAM truth is VERIFICATION ONLY; navigation never reads it",
        },
        "world": {
            "areas_total": world.get("areas_total"),
            "connections": world.get("connections"),
            "current_area": world.get("current_area"),
            "areas": world.get("areas"),
        },
        "rounds": rounds,
        "artifacts": {
            "screens": len(list((harness.out_dir / "screens").glob("frame_*.png"))),
            "maps": len(list((harness.out_dir / "maps").glob("map_*.png"))),
            "montage": montage,
            "world_graph": common.render_world_png(agent, harness.out_dir / "final_world_graph.png"),
            "final_map": common.render_map_png(agent, harness.out_dir / "final_local_map.png"),
            "map_gallery": gallery,
        },
    }
    common.write_json(harness.out_dir / "report.json", payload)

    lines = [
        "# G4 — 有限预算真实无人值守探索（多区域）",
        "",
        f"- 存档: `{Path(harness.state).name}`；运行时长 {elapsed:.1f}s",
        f"- 方向操作: {len(rec.steps)} 次；真实位移: {moves} 次；"
        f"有效位移比例: {effective_ratio:.1%}",
        f"- 新发现格子: {new_cells_total}；已知世界格数: "
        f"{sum(int(row.get('cells') or 0) for row in coverage_rows)}",
        f"- 区域数: {world.get('areas_total')}；连接数: {world.get('connections')}；"
        f"区域切换: {transitions}（重定位成功 {relocalized}）",
        f"- 撞墙/受阻尝试: {results.get('blocked', 0) + results.get('turned_only', 0)} 次；"
        f"UNKNOWN: {unknown_steps} 次",
        f"- 重复访问率: {(revisits / moves) if moves else 0:.1%}；"
        f"路径规划失败: {totals['planning_failures']} 次",
        f"- 定位漂移（视觉 vs RAM，仅验证用）: max={max(drift_values or [0])} 格",
        f"- DeepSeek API 调用: 0 次（本脚本为纯导航长跑；真实 API 集成见 G4-API）",
        f"- 结束原因: {final_reason}",
        "",
        "## 每区域覆盖",
        "",
        "| 区域 | 类型 | 已知格 | 可行走 | 覆盖率 | 访问次数 | 已探索 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in coverage_rows:
        lines.append(
            f"| {row['area_id']} | {row.get('kind')} | {row.get('cells')} | "
            f"{row.get('walkable')} | {float(row.get('coverage_ratio') or 0):.1%} | "
            f"{row.get('visits')} | {'是' if row.get('explored') else '否'} |"
        )
    lines += [
        "",
        "## 证据文件",
        f"- 截图: `{harness.out_dir / 'screens'}`；地图快照: `{harness.out_dir / 'maps'}`",
        f"- 截图拼图: `{montage}`",
        f"- 世界图: `{payload['artifacts']['world_graph']}`；最终地图: "
        f"`{payload['artifacts']['final_map']}`",
        f"- 地图画廊: `{harness.out_dir / 'map_gallery'}`；"
        f"逐动作日志: `{harness.out_dir / 'navigation_steps.jsonl'}`",
        "",
        "_RAM 真值仅用于本验收脚本的独立校验，导航决策链路从不读取 RAM。_",
    ]
    common.write_text(harness.out_dir / "summary.md", "\n".join(lines) + "\n")
    harness.close()
    print(f"[g4] passed={passed} moves={moves} new_cells={new_cells_total} "
          f"areas={world.get('areas_total')} transitions={transitions} reason={final_reason}")
    print(f"[g4] report -> {harness.out_dir / 'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
