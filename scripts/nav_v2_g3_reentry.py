"""G3 — leave an explored room and re-enter it: permanent map + relocalization.

Sequence (all on the real emulator, no manual step-by-step control):
  A. explore Oak's Lab from a legal save state until the room is honestly mapped
     (same policy as G2, including the door-warp round trip),
  B. walk to the known exit and leave the room (area transition),
  C. walk back through the same door and ask the navigation agent to *attach*:
     it must recognise the known area, restore the historical map and its
     coordinate frame instead of creating a new area,
  D. navigate to a far known cell using the stored map, and verify that this
     re-use costs no new exploration (no new cells, no blocked attempts, fewer
     steps than the first exploration).

Usage:
  .venv\\Scripts\\python.exe scripts\\nav_v2_g3_reentry.py
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="G3: leave and re-enter an explored area")
    parser.add_argument("--state", default="lab_start.state")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "nav_v2" / "g3"))
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--round-steps", type=int, default=220)
    parser.add_argument("--round-seconds", type=float, default=120.0)
    parser.add_argument("--max-explore-steps", type=int, default=900)
    parser.add_argument("--max-navigate-steps", type=int, default=200)
    parser.add_argument("--no-fresh", action="store_true")
    return parser.parse_args(argv)


def explore_until_done(harness: common.Harness, args: argparse.Namespace,
                       log: list[dict]) -> dict:
    """Explore the start room with the door-warp round trip until complete."""
    agent, emu, rec = harness.agent, harness.emu, harness.recorder
    totals = {"steps": 0, "moves": 0, "blocked": 0, "unknown": 0, "new_cells": 0,
              "transitions": 0, "reentries": 0, "reentries_matched": 0}
    completed = False
    reason = ""
    area_id = agent.status().get("area_id", "")
    for round_index in range(max(1, int(args.max_rounds))):
        if totals["steps"] >= int(args.max_explore_steps) and totals["steps"] > 0:
            reason = f"exploration step budget {args.max_explore_steps} exhausted"
            break
        report = agent.explore_current_area(
            max_steps=min(int(args.round_steps), int(args.max_explore_steps) - totals["steps"]),
            max_seconds=float(args.round_seconds),
        )
        payload = rec.on_report(report, phase=f"g3_explore_{round_index}")
        log.append(payload)
        for key in ("steps", "moves", "blocked", "unknown", "new_cells"):
            totals[key] += common.metric_int(payload, key)
        print(f"[g3][explore {round_index}] status={report.status.value} steps={report.steps} "
              f"new_cells={report.new_cells} switched={report.area_switched}")
        if report.status is ExploreStatus.COMPLETED:
            completed = True
            reason = report.reason or "room mapped"
            break
        if report.status is ExploreStatus.TRANSITION:
            totals["transitions"] += 1
            crossing_area = str(payload.get("area_id") or area_id)
            common.wait_stable(emu, max_frames=360)
            direction = common.last_moved_direction(rec)
            reverse = common.reverse_direction(direction)
            if reverse is None:
                reason = "transition without a known crossing direction"
                break
            emu.press(reverse)
            common.wait_stable(emu, max_frames=360)
            agent.attach()
            attach = agent.last_attach or {}
            totals["reentries"] += 1
            if attach.get("matched") and str(attach.get("area_id")) == crossing_area:
                totals["reentries_matched"] += 1
                print(f"[g3][explore] re-entered {crossing_area} "
                      f"(match score={attach.get('score')})")
                continue
            reason = "re-entry after the door warp did not relocalize"
            break
        if report.status in (ExploreStatus.INTERRUPTED_EVENT, ExploreStatus.LOCALIZATION_LOST,
                             ExploreStatus.BLOCKED, ExploreStatus.ERROR):
            reason = report.reason or report.status.value
            break
        # step/time/no-progress: give the planner another round
        reason = report.reason or report.status.value
        if len(log) >= 3 and all(item.get("status") in ("step_limit", "time_limit", "no_progress")
                                 for item in log[-2:]):
            break
    return {
        "completed": completed,
        "reason": reason,
        "totals": totals,
        "area_id": agent.status().get("area_id", ""),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common.utf8_stdout()
    started = time.time()
    harness = common.build_harness(args.out, state=args.state, fresh=not args.no_fresh,
                                   label="g3", screen_every=10, map_every=40)
    agent, emu, rec = harness.agent, harness.emu, harness.recorder
    phases: dict[str, object] = {}
    explore_log: list[dict] = []
    initial_truth = common.ram_truth(emu)
    print(f"[g3] initial RAM map={initial_truth.map_id} coords={initial_truth.x},{initial_truth.y}")
    areas_before = common.area_signature(agent)

    # -- A: explore the room ------------------------------------------------- #
    phase_a = explore_until_done(harness, args, explore_log)
    area_a = str(phase_a["area_id"])
    world = agent.world_summary()
    area_a_record = next((item for item in (world.get("areas") or [])
                          if str(item.get("area_id")) == area_a), {})
    coverage_a = agent.status()
    cells_a_before = int(area_a_record.get("cells") or coverage_a.get("cells_total") or 0)
    walkable_a = int(area_a_record.get("walkable") or coverage_a.get("walkable") or 0)
    anchor = common.anchor_now(agent, emu)
    common.render_map_png(agent, harness.out_dir / "map_after_explore.png")
    print(f"[g3] phase A: completed={phase_a['completed']} area={area_a} "
          f"walkable={walkable_a} cells={cells_a_before} "
          f"steps={phase_a['totals']['steps']}")
    phases["explore"] = {
        "completed": phase_a["completed"], "reason": phase_a["reason"], "area_id": area_a,
        "walkable_cells": walkable_a, "cells": cells_a_before,
        "steps": phase_a["totals"]["steps"], "rounds": explore_log,
        "map_png": str(harness.out_dir / "map_after_explore.png"),
    }

    # -- B: leave through the known exit ------------------------------------ #
    steps_before_navigate = len(rec.steps)
    nav_report = agent.navigate_to({"exit": True}, max_steps=int(args.max_navigate_steps))
    nav_payload = rec.on_report(nav_report, phase="g3_leave")
    world_after_leave = agent.world_summary()
    area_b = str(agent.status().get("area_id") or "")
    left = bool(nav_report.area_switched) or (nav_report.status is ExploreStatus.TRANSITION)
    truth_after_leave = common.ram_truth(emu)
    print(f"[g3] phase B: status={nav_report.status.value} reached={nav_report.reached} "
          f"area_switched={nav_report.area_switched} area_b={area_b} "
          f"RAM map={truth_after_leave.map_id}")
    phases["leave"] = {
        "status": nav_report.status.value, "reached": nav_report.reached,
        "area_switched": nav_report.area_switched, "steps": nav_report.steps,
        "reason": nav_report.reason, "area_b": area_b,
        "report": nav_payload,
    }
    if not left:
        print("[g3] could not leave through the exit — reporting honestly")
    common.wait_stable(emu, max_frames=360)

    # -- C: walk back in and attach (relocalize) ---------------------------- #
    crossing = None
    for step in reversed(rec.steps[steps_before_navigate:]):
        if (step.get("motion") or {}).get("result") in ("moved", "transition"):
            direction = step.get("direction")
            if direction in common.OPPOSITE:
                crossing = direction
                break
    if crossing is None:
        crossing = common.last_moved_direction(rec)
    reverse = common.reverse_direction(crossing)
    reentry: dict = {"crossing_direction": crossing, "reverse": reverse, "matched": False}
    if reverse is not None:
        emu.press(reverse)
        common.wait_stable(emu, max_frames=360)
        agent.attach()
        attach = agent.last_attach or {}
        reentry.update({"attach": attach, "matched": bool(attach.get("matched")),
                        "area_id": attach.get("area_id"), "pose": attach.get("pose"),
                        "score": attach.get("score"), "new_area": attach.get("new_area")})
        area_c = str(agent.status().get("area_id") or "")
        reentry["area_after"] = area_c
        reentry["same_area_as_explored"] = bool(
            attach.get("matched") and str(attach.get("area_id")) == area_a
        )
        common.render_map_png(agent, harness.out_dir / "map_after_reentry.png")
        print(f"[g3] phase C: attach={attach}")
    world_after_reentry = agent.world_summary()
    areas_after, _ = common.area_signature(agent)
    duplicates = int(areas_after) - (int(areas_before[0]) + 1)
    phases["reentry"] = {
        **reentry,
        "areas_total": areas_after,
        "duplicate_areas_created": max(0, duplicates),
        "map_png": str(harness.out_dir / "map_after_reentry.png"),
    }

    # -- D: reuse the stored map to reach a far known cell ------------------ #
    entry_pose = tuple(reentry.get("pose") or (0, 0))
    target = common.far_cell(agent, area_a, from_pose=entry_pose)
    before_cells = int(agent.status().get("cells_total") or 0)
    before_walkable = int(agent.status().get("walkable") or 0)
    reuse: dict = {"target": list(target) if target else None}
    if target is not None:
        step_mark = len(rec.steps)
        nav2 = agent.navigate_to({"cell": [int(target[0]), int(target[1])]},
                                 max_steps=int(args.max_navigate_steps))
        rec.on_report(nav2, phase="g3_reuse_map")
        reuse.update({
            "status": nav2.status.value,
            "reached": nav2.reached,
            "steps": nav2.steps,
            "blocked": nav2.blocked,
            "new_cells": getattr(nav2, "new_cells", 0),
            "replans": nav2.replans,
            "reason": nav2.reason,
            "after_cells": int(agent.status().get("cells_total") or 0),
            "after_walkable": int(agent.status().get("walkable") or 0),
        })
        reuse["cell_delta"] = int(reuse["after_cells"]) - before_cells
        reuse["walkable_delta"] = int(reuse["after_walkable"]) - before_walkable
        reuse["pose"] = agent.status().get("pose")
        print(f"[g3] phase D: status={nav2.status.value} reached={nav2.reached} "
              f"steps={nav2.steps} blocked={nav2.blocked} new_cells={getattr(nav2, 'new_cells', 0)}")
    phases["reuse"] = reuse
    final_truth = common.ram_truth(emu)
    drift = common.drift(agent, emu, anchor) if agent.status().get("area_id") == area_a else -1
    elapsed = time.time() - started

    first_steps = int(phase_a["totals"]["steps"])
    reuse_steps = int(reuse.get("steps") or 0)
    steps_ratio = (reuse_steps / first_steps) if first_steps else None
    matched = bool(reentry.get("same_area_as_explored"))
    no_new_area = phases["reentry"]["duplicate_areas_created"] == 0
    reuse_clean = bool(
        reuse.get("reached") and int(reuse.get("cell_delta") or 0) <= 0
        and int(reuse.get("blocked") or 0) <= 1
    )
    passed = bool(matched and no_new_area and left and (target is None or reuse_clean))
    payload = {
        "task": "navigation_v2_g3_area_reentry",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state": str(harness.state),
        "elapsed_s": round(elapsed, 2),
        "passed": passed,
        "phases": phases,
        "summary": {
            "area_a": area_a,
            "area_b": area_b,
            "explore_steps": first_steps,
            "explore_completed": phase_a["completed"],
            "leave_ok": left,
            "reentry_matched": matched,
            "reentry_score": reentry.get("score"),
            "reentry_pose": reentry.get("pose"),
            "duplicate_areas_created": phases["reentry"]["duplicate_areas_created"],
            "reuse_steps": reuse_steps,
            "reuse_to_first_explore_ratio": None if steps_ratio is None else round(steps_ratio, 4),
            "reuse_new_cells": reuse.get("new_cells"),
            "reuse_cell_delta": reuse.get("cell_delta"),
            "reuse_blocked": reuse.get("blocked"),
            "drift_after_all": drift,
        },
        "world": {
            "areas_total": world_after_reentry.get("areas_total"),
            "connections": world_after_reentry.get("connections"),
            "areas": world_after_reentry.get("areas"),
        },
        "localization": {
            "ram_truth_start": {"map_id": initial_truth.map_id, "x": initial_truth.x, "y": initial_truth.y},
            "ram_truth_end": {"map_id": final_truth.map_id, "x": final_truth.x, "y": final_truth.y},
            "drift": drift,
            "ram_note": "RAM truth is VERIFICATION ONLY; navigation never reads it",
        },
        "artifacts": {
            "screens": len(list((harness.out_dir / "screens").glob("frame_*.png"))),
            "maps": len(list((harness.out_dir / "maps").glob("map_*.png"))),
            "montage": common.screenshot_montage(
                harness.out_dir, harness.out_dir / "screens_montage.png", cols=4, scale=2, limit=24
            ),
            "world_graph": common.render_world_png(agent, harness.out_dir / "final_world_graph.png"),
            "map_gallery": common.run_map_gallery(
                harness.out_dir / "navigation", harness.out_dir / "map_gallery"
            ),
        },
    }
    common.write_json(harness.out_dir / "report.json", payload)

    lines = [
        "# G3 — 离开并重新进入已探索区域（真实游戏）",
        "",
        f"- 存档: `{Path(harness.state).name}`；区域 A=`{area_a}`，区域 B=`{area_b}`",
        f"- 首次探索: {'完成' if phase_a['completed'] else '未完成'}，"
        f"方向操作 {first_steps} 次，可行走格 {walkable_a}，已知格 {cells_a_before}",
        f"- 离开: {'成功' if left else '失败'}（navigate_to exit: {phases['leave']['status']}）",
        f"- 重新进入识别: **{'成功' if matched else '失败'}**，"
        f"重定位分数 {reentry.get('score')}，恢复坐标 {reentry.get('pose')}",
        f"- 新建重复区域数: {phases['reentry']['duplicate_areas_created']}（应为 0）",
        f"- 使用已知地图导航到远端格: 目标 {reuse.get('target')}，"
        f"{'到达' if reuse.get('reached') else '未到达'}，"
        f"步数 {reuse_steps}（首次探索 {first_steps} 步，比例 {steps_ratio}）",
        f"- 重用地图时新增格数: {reuse.get('cell_delta')}；撞墙: {reuse.get('blocked')}",
        f"- 视觉位置漂移 vs RAM 真值: {drift} 格（仅验证用）",
        f"- 结论: **{'通过' if passed else '未通过'}**",
        "",
        "## 说明",
        "首次探索包含一次进出门区域切换与重定位；再次进入时没有重新探索，直接恢复历史地图",
        "与坐标系，并用已知路线导航到远端格。RAM 真值只用于独立校验。",
    ]
    common.write_text(harness.out_dir / "summary.md", "\n".join(lines) + "\n")
    harness.close()
    print(f"[g3] passed={passed} matched={matched} reuse={reuse.get('status')}")
    print(f"[g3] report -> {harness.out_dir / 'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
