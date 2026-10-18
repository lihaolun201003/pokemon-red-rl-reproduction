"""G2 — autonomous exploration of an unfamiliar real-game room (Oak's Lab).

The navigation agent starts from the repository's legal save state with **no
prior map** (fresh world-model directory).  It must build the room's logical grid
map by itself, discover obstacles and exits, avoid repeatedly bumping the same
obstacle, and decide honestly whether exploration is complete.

When the frontier policy happens to step onto the room's exit warp (the door), the
script waits for the transition to settle, walks back through the same door and
attaches again; a successful re-entry relocalizes against the stored map and
exploration continues.  No human step-by-step control, no walkthrough, no RAM in
the navigation path.  RAM is used only by this harness to *verify* drift.

Usage:
  .venv\\Scripts\\python.exe scripts\\nav_v2_g2_explore_room.py
  ... --state lab_start.state --out artifacts/nav_v2/g2 --max-total-steps 600
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
    parser = argparse.ArgumentParser(description="G2: autonomous unfamiliar-room exploration")
    parser.add_argument("--state", default="lab_start.state", help="fixed legal save state to start from")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "nav_v2" / "g2"),
                        help="output directory for logs/screens/maps/report")
    parser.add_argument("--max-total-steps", type=int, default=700)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--round-steps", type=int, default=220)
    parser.add_argument("--round-seconds", type=float, default=120.0)
    parser.add_argument("--no-fresh", action="store_true", help="keep an existing world model dir")
    return parser.parse_args(argv)


def bump_repeats(recorder: common.Recorder) -> tuple[int, int]:
    """Count repeated blocked attempts from the same cell+same direction."""
    seen: Counter = Counter()
    repeated = 0
    for step in recorder.steps:
        motion = step.get("motion") or {}
        if motion.get("result") not in ("blocked", "turned_only"):
            continue
        key = (tuple(step.get("pose_before") or ()), step.get("direction"))
        seen[key] += 1
        if seen[key] >= 2:
            repeated += 1
    return repeated, len(seen)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common.utf8_stdout()
    started = time.time()
    harness = common.build_harness(
        args.out, state=args.state, fresh=not args.no_fresh, label="g2",
        screen_every=10, map_every=40,
    )
    agent, emu, rec = harness.agent, harness.emu, harness.recorder
    report_path = harness.out_dir / "report.json"
    summary_path = harness.out_dir / "summary.md"
    initial_status = agent.status()
    anchor = common.anchor_now(agent, emu)
    initial_truth = common.ram_truth(emu)
    print(f"[g2] start state={harness.state.name} map={initial_truth.map_id} "
          f"coords={initial_truth.x},{initial_truth.y} area={initial_status.get('area_id')}")

    rounds: list[dict] = []
    totals = {"steps": 0, "moves": 0, "blocked": 0, "turned": 0, "unknown": 0,
              "new_cells": 0, "new_edges": 0, "new_exits": 0, "transitions": 0}
    reentries: list[dict] = []
    drifts: list[int] = []
    final_status_reason = ""
    completed = False
    interrupted_event: dict | None = None
    localization_lost = False
    previous_area = initial_status.get("area_id", "")
    step_budget = max(1, int(args.max_total_steps))
    # Room-exploration policy: EXIT cells are landmarks, not wander targets.  The
    # navigation core also prefers non-exit frontiers; this harness flag keeps the
    # room-only scope explicit and testable.
    exit_bonus = common.set_exit_last_resort(agent, True)

    for round_index in range(max(1, int(args.max_rounds))):
        remaining = max(1, step_budget - totals["steps"])
        if remaining <= 1 and totals["steps"] > 0:
            final_status_reason = f"total step budget {step_budget} exhausted"
            break
        if round_index > 0 and common.room_frontiers_left(agent) == 0:
            completed = True
            final_status_reason = (
                "no reachable non-exit frontier left (only exit landmarks remain): "
                "room coverage complete, leaving the room is a separate navigation decision"
            )
            break
        before_areas, before_cells = common.area_signature(agent)
        truth_before = common.ram_truth(emu)
        report = agent.explore_current_area(
            max_steps=min(int(args.round_steps), remaining),
            max_seconds=float(args.round_seconds),
        )
        payload = rec.on_report(report, phase=f"explore_round_{round_index}")
        rounds.append(payload)
        for key in ("steps", "moves", "blocked", "turned", "unknown"):
            totals[key] += common.metric_int(payload, key)
        for key in ("new_cells", "new_edges", "new_exits"):
            totals[key] += common.metric_int(payload, key)
        status = report.status
        print(f"[g2] round {round_index}: status={status.value} steps={report.steps} "
              f"moves={report.moves} new_cells={report.new_cells} "
              f"switched={report.area_switched} reason={report.reason}")
        if not report.area_switched:
            drifts.append(common.drift(agent, emu, anchor))
        if status is ExploreStatus.COMPLETED:
            completed = True
            final_status_reason = report.reason or "exploration complete"
            break
        if status is ExploreStatus.INTERRUPTED_EVENT:
            interrupted_event = None if report.event is None else report.event.to_dict()
            final_status_reason = report.reason or "non-navigation event"
            break
        if status in (ExploreStatus.LOCALIZATION_LOST, ExploreStatus.BLOCKED,
                      ExploreStatus.ERROR):
            localization_lost = status is ExploreStatus.LOCALIZATION_LOST
            final_status_reason = report.reason or status.value
            break
        if status in (ExploreStatus.NO_PROGRESS, ExploreStatus.STEP_LIMIT,
                      ExploreStatus.TIME_LIMIT):
            # allow a fresh round (re-plan); stop if this repeats twice
            final_status_reason = report.reason or status.value
            if len(rounds) >= 3 and all(
                item.get("status") in ("step_limit", "time_limit", "no_progress")
                for item in [r for r in rounds][-2:]
            ):
                break
            continue
        if status is ExploreStatus.TRANSITION:
            totals["transitions"] += 1
            previous_area = str(payload.get("area_id") or previous_area)
            crossing_area = str(payload.get("area_id") or "")
            new_area = str(payload.get("new_area_id") or "")
            common.wait_stable(emu, max_frames=360)
            # The crossing press is the last direction step (its result is
            # 'transition', not 'moved'), so reverse that exact direction.
            direction = common.last_step_direction(rec)
            reverse = common.reverse_direction(direction)
            entry: dict = {
                "round": round_index,
                "from_area": crossing_area,
                "new_area": new_area,
                "crossing_direction": direction,
                "reverse": reverse,
                "matched": False,
            }
            ok = False
            if reverse is not None:
                emu.press(reverse)
                common.wait_stable(emu, max_frames=360)
                agent.attach()
                attach = agent.last_attach or {}
                entry.update({
                    "attach": attach,
                    "matched": bool(attach.get("matched")),
                    "attach_area": attach.get("area_id"),
                    "attach_pose": attach.get("pose"),
                    "attach_score": attach.get("score"),
                })
                ok = bool(attach.get("matched")) and str(attach.get("area_id")) == str(crossing_area)
                if ok:
                    anchor = common.anchor_now(agent, emu)
                    print(f"[g2] re-entered {crossing_area} after transition "
                          f"(match score={attach.get('score')})")
                    if common.room_frontiers_left(agent) == 0:
                        completed = True
                        final_status_reason = (
                            "room coverage complete after re-entry: only exit landmarks remain"
                        )
                        break
                    print("[g2] -> continuing exploration of the room")
            reentries.append(entry)
            if not ok:
                final_status_reason = (
                    "room exit triggered a transition and re-entry did not relocalize; "
                    "exploration stopped honestly instead of remapping a duplicate area"
                )
                break
            continue
        final_status_reason = report.reason or status.value
        break

    # -- finalise -------------------------------------------------------- #
    common.restore_exit_bonus(agent, exit_bonus)
    final_status = agent.status()
    final_truth = common.ram_truth(emu)
    drift_final = common.drift(agent, emu, anchor)
    if not drifts or drifts[-1] != drift_final:
        drifts.append(drift_final)
    coverage = final_status.get("coverage") or {}
    world = agent.world_summary()
    interior_areas = [a for a in (world.get("areas") or []) if a.get("kind") != "exterior"]
    exit_cells = 0
    try:
        grid = common._current_grid(agent)
        counts = Counter(str(getattr(getattr(cell, "state", None), "value", ""))
                         for cell in (getattr(grid, "cells", {}) or {}).values())
        exit_cells = counts.get("exit", 0)
        blocked_cells = counts.get("blocked", 0) + counts.get("temp_blocked", 0)
        walkable_cells = counts.get("walkable", 0)
        observed_cells = counts.get("observed", 0)
    except Exception:  # noqa: BLE001
        blocked_cells = walkable_cells = observed_cells = 0
    repeated_bumps, distinct_bump_sites = bump_repeats(rec)
    screens = len(list((harness.out_dir / "screens").glob("frame_*.png")))
    maps = len(list((harness.out_dir / "maps").glob("map_*.png")))
    montage = common.screenshot_montage(
        harness.out_dir, harness.out_dir / "screens_montage.png", cols=4, scale=2, limit=24
    )
    map_png = common.render_map_png(agent, harness.out_dir / "final_local_map.png")
    world_png = common.render_world_png(agent, harness.out_dir / "final_world_graph.png")
    gallery = common.run_map_gallery(
        harness.out_dir / "navigation", harness.out_dir / "map_gallery"
    )
    try:
        agent.save()
    except Exception:  # noqa: BLE001
        pass
    elapsed = time.time() - started

    passed = bool(completed and totals["moves"] >= 10 and not localization_lost)
    payload = {
        "task": "navigation_v2_g2_room_exploration",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state": str(harness.state),
        "elapsed_s": round(elapsed, 2),
        "passed": passed,
        "completed": bool(completed),
        "final_status_reason": final_status_reason,
        "localization_lost": bool(localization_lost),
        "interrupted_event": interrupted_event,
        "rounds": rounds,
        "totals": {**totals, "rounds": len(rounds), "reentries": len(reentries),
                   "reentries_matched": sum(1 for item in reentries if item.get("matched"))},
        "reentries": reentries,
        "coverage": coverage,
        "map": {
            "cells_total": final_status.get("cells_total"),
            "walkable": walkable_cells or final_status.get("walkable"),
            "blocked": blocked_cells or final_status.get("blocked"),
            "observed": observed_cells,
            "unknown_frontier": final_status.get("unknown_frontier"),
            "exits": exit_cells,
            "coverage_ratio": final_status.get("coverage_ratio"),
        },
        "obstacles": {
            "blocked_cells": blocked_cells,
            "distinct_bump_sites": distinct_bump_sites,
            "repeated_same_obstacle_attempts": repeated_bumps,
        },
        "world": {
            "areas_total": world.get("areas_total"),
            "connections": world.get("connections"),
            "areas": world.get("areas"),
            "interior_areas": len(interior_areas),
        },
        "localization": {
            "drift_samples": drifts,
            "drift_max": max([d for d in drifts if d >= 0], default=0),
            "drift_final": drift_final,
            "ram_truth_start": {"map_id": initial_truth.map_id, "x": initial_truth.x, "y": initial_truth.y},
            "ram_truth_end": {"map_id": final_truth.map_id, "x": final_truth.x, "y": final_truth.y},
            "ram_note": "RAM truth is VERIFICATION ONLY; navigation never reads it",
        },
        "pose": final_status.get("pose"),
        "artifacts": {
            "screens": screens,
            "map_snapshots": maps,
            "screens_montage": montage,
            "final_local_map": map_png,
            "final_world_graph": world_png,
            "map_gallery": gallery,
            "world_model": str(harness.out_dir / "navigation"),
        },
    }
    common.write_json(report_path, payload)

    lines = [
        "# G2 — 陌生房间自主探索（真实游戏）",
        "",
        f"- 起始合法存档: `{Path(harness.state).name}`（RAM 校验：map_id={initial_truth.map_id}）",
        f"- 总耗时: {elapsed:.1f}s；方向操作 {totals['steps']} 次（其中真实位移 {totals['moves']} 次）",
        f"- 探索轮次: {len(rounds)}；区域切换: {totals['transitions']}；"
        f"切换后重新进入并重定位成功: {sum(1 for item in reentries if item.get('matched'))}/{len(reentries)}",
        f"- 本地网格: 已知 {final_status.get('cells_total')} 格、可行走 {walkable_cells}、"
        f"障碍 {blocked_cells}、出口 {exit_cells}、未知边界 {final_status.get('unknown_frontier')}",
        f"- 覆盖率 coverage_ratio: {float(final_status.get('coverage_ratio') or 0.0):.1%}",
        f"- 世界模型: {world.get('areas_total')} 个区域 / {world.get('connections')} 条连接",
        f"- 同一障碍重复撞击（同格同方向再次尝试）: {repeated_bumps} 次",
        f"- 定位漂移检查（视觉位置 vs RAM 真值，仅验证用）: max={max([d for d in drifts if d >= 0], default=0)} 格",
        f"- 结论: {'**探索完成**' if completed else '**未完成（诚实记录原因）**'} — {final_status_reason}",
        "",
        "## 每轮报告",
        "",
        "| 轮次 | 状态 | 步数 | 位移 | 新格 | 新出口 | 原因 |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, item in enumerate(rounds):
        lines.append(
            f"| {index} | {item.get('status')} | {item.get('steps')} | {item.get('moves')} | "
            f"{item.get('new_cells')} | {item.get('new_exits')} | {str(item.get('reason') or '')[:70]} |"
        )
    lines += [
        "",
        "## 证据文件",
        f"- 截图: `{harness.out_dir / 'screens'}`（{screens} 张）",
        f"- 地图快照: `{harness.out_dir / 'maps'}`（{maps} 张）",
        f"- 截图拼图: `{montage}`",
        f"- 最终本地地图: `{map_png}`；世界图: `{world_png}`",
        f"- 地图画廊: `{harness.out_dir / 'map_gallery'}`",
        "",
        "_RAM 真值仅用于本验收脚本的独立校验，导航决策链路从不读取 RAM。_",
    ]
    common.write_text(summary_path, "\n".join(lines) + "\n")
    harness.close()
    print(f"[g2] passed={passed} completed={completed} reason={final_status_reason}")
    print(f"[g2] report -> {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
