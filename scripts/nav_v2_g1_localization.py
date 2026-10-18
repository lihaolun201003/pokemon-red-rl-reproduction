"""G1 — real-game visual localization acceptance test (Navigation v2).

Runs **real PyBoy** with the real ROM / save states and verifies the pure-vision
localizer against an independent RAM probe.  The RAM probe is used ONLY in this
validation script (never by the navigation modules): see
``llm_agent/progress.py`` (verification-only module) and the import-boundary test.

Scenarios
---------
* ``lab_walk``        — deterministic random direction presses inside Oak's Lab
                        (furniture walls, walking NPCs).
* ``lab_turn_block``  — mixed turn-in-place / blocked / move sequence at the lab's
                        bottom-right corner (known wall layout).
* ``multi_tile``      — held direction presses that move 2 tiles at once.
* ``live_menu``       — opening the START menu then pressing a direction must be
                        classified as INTERRUPTED (no coordinate change).
* ``pallet_walk``     — outdoor random walk in Pallet Town (camera scroll, open map).
* ``transitions``     — walking through interior door warps both ways.
* ``real_frames``     — real screenshots of battle/dialog/menu/transition captured
                        from earlier genuine runs: must never be MOVED.

Acceptance criteria (hard):
* ``wrong_moves == 0``: never claim a movement that did not happen.
* movement accuracy over truly moved ops >= 95 % (UNKNOWN is reported honestly,
  never silently turned into success).
* no lost localizations.

Outputs: ``artifacts/nav_v2/g1/report.json`` + ``summary.md`` + a few PNGs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_agent.config import AgentConfig  # noqa: E402
from llm_agent.emulator import GameBoyEmulator  # noqa: E402
from llm_agent.navigation import vision  # noqa: E402
from llm_agent.navigation.localization import Localizer  # noqa: E402
from llm_agent.navigation.types import DIRECTIONS, MotionResult  # noqa: E402
from llm_agent.observation import analyze, frame_from_rgba  # noqa: E402
from llm_agent.progress import OverworldProbe, probe_overworld  # noqa: E402

STATES = ROOT / "work" / "nav_v2" / "states"
ARTIFACTS = ROOT / "artifacts" / "nav_v2" / "g1"


def _pyboy(emu: GameBoyEmulator):
    pyboy = getattr(emu, "_pyboy", None)
    if pyboy is None:
        raise RuntimeError("GameBoyEmulator is not started")
    return pyboy


def probe(emu: GameBoyEmulator) -> OverworldProbe:
    return probe_overworld(_pyboy(emu))


def wait_until_stable(emu: GameBoyEmulator, *, max_frames: int = 240, step: int = 8):
    """Tick until two consecutive frames are (nearly) identical; return frames."""
    seen = [emu.observe()]
    total = 0
    while total < max_frames:
        nxt = emu.wait(step)
        total += step
        diff = vision.diff_mask(vision.frame_luma(seen[-1]), vision.frame_luma(nxt))
        seen.append(nxt)
        if float(diff.mean()) <= 0.003 and not nxt.is_black:
            break
    return seen


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@dataclass
class OpRecord:
    scenario: str
    index: int
    direction: str
    truth_before: dict
    truth_after: dict
    estimate: dict
    outcome: str
    pos_error: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "index": self.index,
            "direction": self.direction,
            "truth_before": self.truth_before,
            "truth_after": self.truth_after,
            "estimate": self.estimate,
            "outcome": self.outcome,
            "pos_error": self.pos_error,
            "notes": self.notes,
        }


@dataclass
class Metrics:
    ops: int = 0
    moves_truth: int = 0
    correct_moves: int = 0
    wrong_moves: int = 0
    missed_moves: int = 0
    blocked_truth: int = 0
    correct_blocked: int = 0
    turn_truth: int = 0
    correct_turns: int = 0
    unknown: int = 0
    unknown_but_moved: int = 0
    lost: int = 0
    transitions_detected: int = 0
    transitions_missed: int = 0
    false_transitions: int = 0
    interruptions_correct: int = 0
    false_interruptions: int = 0
    pos_error_sum: int = 0
    pos_error_max: int = 0
    pos_error_final: int = 0

    def merge(self, other: "Metrics") -> None:
        for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict:
        accuracy = self.correct_moves / self.moves_truth if self.moves_truth else 0.0
        return {
            "ops": self.ops,
            "moves_truth": self.moves_truth,
            "correct_moves": self.correct_moves,
            "wrong_moves": self.wrong_moves,
            "missed_moves": self.missed_moves,
            "move_accuracy": round(accuracy, 4),
            "blocked_truth": self.blocked_truth,
            "correct_blocked": self.correct_blocked,
            "turn_truth": self.turn_truth,
            "correct_turns": self.correct_turns,
            "unknown": self.unknown,
            "unknown_but_moved": self.unknown_but_moved,
            "lost": self.lost,
            "transitions_detected": self.transitions_detected,
            "transitions_missed": self.transitions_missed,
            "false_transitions": self.false_transitions,
            "interruptions_correct": self.interruptions_correct,
            "false_interruptions": self.false_interruptions,
            "pos_error_sum": self.pos_error_sum,
            "pos_error_max": self.pos_error_max,
            "pos_error_final": self.pos_error_final,
        }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

class G1Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        cfg = AgentConfig.load()
        cfg.emulator.window = "null"
        cfg.emulator.settle_frames = 2
        self.cfg = cfg
        self.emu = GameBoyEmulator(cfg.emulator).start()
        self.localizer = Localizer()
        self.records: list[OpRecord] = []
        self.metrics = Metrics()
        self.scenarios: list[dict] = []
        self.screens: dict[str, np.ndarray] = {}
        self._origin = (0, 0)
        self._scenario = ""
        self._index = 0

    def close(self) -> None:
        try:
            self.emu.close()
        except Exception:  # noqa: BLE001
            pass

    # -- lifecycle ------------------------------------------------------ #
    def _load_state(self, name: str) -> OverworldProbe:
        path = STATES / name
        if not path.is_file():
            raise FileNotFoundError(
                f"missing navigation test state {path}; run work/nav_v2/make_states.py first"
            )
        self.emu.load_state(path)
        wait_until_stable(self.emu, max_frames=180)
        return probe(self.emu)

    def begin_scenario(self, name: str, state: str) -> OverworldProbe:
        self._scenario = name
        self._index = 0
        truth = self._load_state(state)
        self.localizer.reset(area_id=name, cell=(0, 0), facing=truth.facing_name)
        self._origin = (truth.x, truth.y)
        return truth

    def _expected_pose(self, truth: OverworldProbe) -> tuple[int, int]:
        return (truth.x - self._origin[0], truth.y - self._origin[1])

    def _position_error_after(self, truth: OverworldProbe) -> int:
        expected = self._expected_pose(truth)
        pose = self.localizer.pose
        return abs(int(pose.x) - expected[0]) + abs(int(pose.y) - expected[1])

    def _slice_metrics(self, start: int) -> Metrics:
        metrics = Metrics()
        for record in self.records[start:]:
            outcome = record.outcome
            metrics.ops += 1
            metrics.pos_error_sum += record.pos_error
            metrics.pos_error_max = max(metrics.pos_error_max, record.pos_error)
            if outcome == "correct_move":
                metrics.correct_moves += 1
                metrics.moves_truth += 1
            elif outcome in ("wrong_move", "wrong_move_delta"):
                metrics.wrong_moves += 1
                metrics.moves_truth += 1
            elif outcome == "transition_walk":
                metrics.transitions_detected += 1
            elif outcome == "missed_move":
                metrics.missed_moves += 1
                metrics.moves_truth += 1
            elif outcome == "unknown_but_moved":
                metrics.unknown += 1
                metrics.unknown_but_moved += 1
                metrics.moves_truth += 1
            elif outcome == "unknown":
                metrics.unknown += 1
            elif outcome == "lost":
                metrics.unknown += 1
                metrics.unknown_but_moved += 1
                metrics.lost += 1
            elif outcome in ("correct_blocked", "correct_turn"):
                metrics.blocked_truth += 1
                metrics.correct_blocked += 1
                if outcome == "correct_turn":
                    metrics.turn_truth += 1
                    metrics.correct_turns += 1
            elif outcome == "correct_transition":
                metrics.transitions_detected += 1
            elif outcome == "missed_transition":
                metrics.transitions_missed += 1
            elif outcome == "false_transition":
                metrics.transitions_detected += 1
                metrics.false_transitions += 1
            elif outcome == "correct_interruption":
                metrics.interruptions_correct += 1
            elif outcome == "false_interruption":
                metrics.false_interruptions += 1
        return metrics

    def _classify(self, direction, before, after, est, pos_error, cur_frame, prev_frame) -> None:
        map_changed = before.map_id != after.map_id
        truth_delta = (after.x - before.x, after.y - before.y)
        moved = truth_delta != (0, 0) and not map_changed
        turned = not moved and after.facing != before.facing
        result = est.result
        outcome = "correct"
        notes = ""

        if result is MotionResult.MOVED:
            if map_changed:
                # a walk onto a door tile that immediately warps: the step did
                # happen, but the area changed inside the same input, so count it
                # as a transition rather than a wrong move.
                outcome = "transition_walk"
            elif truth_delta == (0, 0):
                outcome = "wrong_move"
            elif tuple(est.delta) == truth_delta:
                outcome = "correct_move"
            else:
                outcome = "wrong_move_delta"
                notes = f"delta {est.delta} != truth {truth_delta}"
        elif result in (MotionResult.BLOCKED, MotionResult.TURNED_ONLY):
            if moved:
                outcome = "missed_move"
                notes = f"truth moved {truth_delta}, visual said {result.value}"
            elif map_changed:
                outcome = "missed_transition"
            else:
                outcome = "correct_turn" if (turned and result is MotionResult.TURNED_ONLY) else "correct_blocked"
        elif result is MotionResult.UNKNOWN:
            outcome = "unknown_but_moved" if moved else "unknown"
            if not self.localizer.pose.valid:
                outcome = "lost"
        elif result is MotionResult.TRANSITION:
            screen = analyze(cur_frame, prev_frame)
            if (
                map_changed
                or cur_frame.is_black
                or screen.event.value in ("transition", "map_change")
            ):
                outcome = "correct_transition"
            else:
                outcome = "false_transition"
                notes = f"truth_delta {truth_delta}"
        elif result is MotionResult.INTERRUPTED:
            screen = analyze(cur_frame, prev_frame)
            if map_changed:
                outcome = "missed_transition"
            elif screen.event.value in ("dialog", "menu", "battle"):
                outcome = "correct_interruption"
            else:
                outcome = "false_interruption"

        self.records.append(
            OpRecord(
                scenario=self._scenario,
                index=self._index,
                direction=direction,
                truth_before=before.to_dict(),
                truth_after=after.to_dict(),
                estimate=est.to_dict(),
                outcome=outcome,
                pos_error=pos_error,
                notes=notes,
            )
        )
        self._index += 1

    def step(self, direction: str):
        """Run one real directional input and classify the visual estimate."""
        emu = self.emu
        prev_frame = emu.observe()
        prev_view = self.localizer.observe(prev_frame)
        before = probe(emu)
        cur_frame = emu.press(direction)
        after = probe(emu)
        cur_view = self.localizer.observe(cur_frame)
        est = self.localizer.update(prev_view, cur_view, direction)
        pos_error = self._position_error_after(after)
        self._classify(direction, before, after, est, pos_error, cur_frame, prev_frame)
        # A warp started by this input: let it settle and start a fresh local
        # coordinate frame for the new area (the localizer cannot know the new
        # map; area identity/relocalization is a separate subsystem).
        if after.map_id != before.map_id:
            wait_until_stable(self.emu, max_frames=240)
            truth = probe(self.emu)
            self.localizer.reset(
                area_id=f"{self._scenario}:auto", cell=(0, 0), facing=truth.facing_name
            )
            self._origin = (truth.x, truth.y)
        return est

    def finish_scenario(self, metrics: Metrics, **extra) -> None:
        truth = probe(self.emu)
        error = self._position_error_after(truth)
        self.metrics.pos_error_final = error
        entry = {
            "scenario": self._scenario,
            "ops": metrics.ops,
            "moves_truth": metrics.moves_truth,
            "correct_moves": metrics.correct_moves,
            "wrong_moves": metrics.wrong_moves,
            "missed_moves": metrics.missed_moves,
            "blocked_truth": metrics.blocked_truth,
            "turn_truth": metrics.turn_truth,
            "correct_turns": metrics.correct_turns,
            "unknown": metrics.unknown,
            "lost": metrics.lost,
            "transitions_detected": metrics.transitions_detected,
            "transitions_missed": metrics.transitions_missed,
            "interruptions_correct": metrics.interruptions_correct,
            "final_pose": list(self.localizer.pose.cell),
            "final_truth": [truth.x, truth.y],
            "map_id": truth.map_id,
            "pos_error_final": error,
            **extra,
        }
        self.scenarios.append(entry)
        print(
            f"  {self._scenario:16s} ops={metrics.ops:3d} moves={metrics.moves_truth:3d} "
            f"ok={metrics.correct_moves:3d} wrong={metrics.wrong_moves} "
            f"missed={metrics.missed_moves} unk={metrics.unknown} lost={metrics.lost} "
            f"trans={metrics.transitions_detected} err={error}"
        )

    def save_screen(self, name: str, frame) -> None:
        self.screens[name] = np.array(frame.rgba, copy=True)

    def _run_script(self, name: str, state: str, script: list[str]) -> Metrics:
        self.begin_scenario(name, state)
        start = len(self.records)
        for direction in script:
            self.step(direction)
        metrics = self._slice_metrics(start)
        self.finish_scenario(metrics)
        return metrics

    # -- scenarios ------------------------------------------------------ #
    def run_lab_walk(self, ops: int) -> Metrics:
        rng = np.random.default_rng(2026)
        script = [DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))] for _ in range(ops)]
        return self._run_script("lab_walk", "lab_start.state", script)

    def run_lab_turn_block(self) -> Metrics:
        script = ["down", "down", "up", "up", "left", "left", "left", "up",
                  "right", "right", "up", "down", "left", "up", "left", "down"]
        return self._run_script("lab_turn_block", "lab_corner.state", script)

    def run_multi_tile(self) -> Metrics:
        self.begin_scenario("multi_tile", "lab_start.state")
        start = len(self.records)
        old_press = self.cfg.emulator.press_frames
        old_action = self.cfg.emulator.action_frames
        self.cfg.emulator.press_frames = 32
        self.cfg.emulator.action_frames = 48
        try:
            for direction in ["down", "down", "up", "up", "down", "down", "up", "up"]:
                self.step(direction)
        finally:
            self.cfg.emulator.press_frames = old_press
            self.cfg.emulator.action_frames = old_action
        metrics = self._slice_metrics(start)
        self.finish_scenario(metrics)
        return metrics

    def run_live_menu(self) -> Metrics:
        self.begin_scenario("live_menu", "lab_start.state")
        start = len(self.records)
        self.emu.press("start")
        wait_until_stable(self.emu, max_frames=120)
        menu_frame = self.emu.observe()
        menu_analysis = analyze(menu_frame, None)
        self.save_screen("live_menu", menu_frame)
        print(f"  live_menu: menu screen classified as {menu_analysis.event.value}")
        self.step("up")        # must be INTERRUPTED, no coordinate change
        self.emu.press("b")    # close the START menu
        wait_until_stable(self.emu, max_frames=120)
        self.step("down")      # normal gameplay again
        metrics = self._slice_metrics(start)
        self.finish_scenario(metrics, menu_event=menu_analysis.event.value)
        return metrics

    def run_pallet_walk(self, ops: int) -> Metrics:
        rng = np.random.default_rng(77)
        script = [DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))] for _ in range(ops)]
        metrics = self._run_script("pallet_walk", "pallet_door.state", script)
        self.save_screen("pallet_walk", self.emu.observe())
        return metrics

    def run_transitions(self) -> Metrics:
        results: list[Metrics] = []
        self.begin_scenario("transition_out", "lab_start.state")
        start = len(self.records)
        for direction in ["down"] * 8:
            self.step(direction)
        self.step("down")  # walk onto the door mat -> warp
        wait_until_stable(self.emu, max_frames=240)
        truth = probe(self.emu)
        self.save_screen("transition_out", self.emu.observe())
        self.localizer.reset(area_id="pallet", cell=(0, 0), facing=truth.facing_name)
        self._origin = (truth.x, truth.y)
        metrics = self._slice_metrics(start)
        self.finish_scenario(metrics, map_id=truth.map_id, coords=[truth.x, truth.y])
        results.append(metrics)

        self.begin_scenario("transition_in", "pallet_door.state")
        start2 = len(self.records)
        self.step("up")
        wait_until_stable(self.emu, max_frames=240)
        truth = probe(self.emu)
        self.save_screen("transition_in", self.emu.observe())
        self.localizer.reset(area_id="lab", cell=(0, 0), facing=truth.facing_name)
        self._origin = (truth.x, truth.y)
        metrics2 = self._slice_metrics(start2)
        self.finish_scenario(metrics2, map_id=truth.map_id, coords=[truth.x, truth.y])
        results.append(metrics2)
        combined = Metrics()
        for item in results:
            combined.merge(item)
        return combined

    def run_real_frames(self) -> Metrics:
        self._scenario = "real_frames"
        self._index = 0
        start = len(self.records)
        frames = self._find_real_frames()
        metrics = Metrics()
        if not frames:
            print("  real_frames: no screenshots found; skipped")
            self.finish_scenario(metrics, skipped=True)
            return metrics
        overworld = None
        for label, path in frames:
            frame = self._load_png(path)
            analysis = analyze(frame, None)
            if label == "overworld":
                overworld = frame
                print(f"  real_frames: overworld reference {path.name}")
                continue
            if overworld is None:
                overworld = self.emu.observe()
            loc = Localizer()
            prev_view = loc.observe(overworld)
            cur_view = loc.observe(frame)
            est = loc.update(prev_view, cur_view, "down")
            before = probe(self.emu)
            print(f"  real_frames: {label:10s} -> {est.result.value:11s} (screen={analysis.event.value})")
            self._classify("down", before, before, est, 0, frame, overworld)
        metrics = self._slice_metrics(start)
        self.finish_scenario(metrics)
        return metrics

    def _find_real_frames(self) -> list[tuple[str, Path]]:
        found: list[tuple[str, Path]] = []
        wanted = {"battle": 2, "dialog": 2, "menu": 1, "transition": 1}
        runs = ROOT / "artifacts" / "llm_agent" / "runs"
        if not runs.is_dir():
            return found
        have_overworld = False
        for path in sorted(runs.rglob("*.png")):
            try:
                frame = self._load_png(path)
            except Exception:  # noqa: BLE001
                continue
            label = analyze(frame, None).event.value
            if label == "none" and not have_overworld:
                found.append(("overworld", path))
                have_overworld = True
            elif wanted.get(label, 0) > 0:
                found.append((label, path))
                wanted[label] -= 1
            if sum(wanted.values()) == 0 and have_overworld:
                break
        return found

    @staticmethod
    def _load_png(path: Path):
        image = Image.open(path).convert("RGBA")
        rgba = np.array(image, dtype=np.uint8)
        if rgba.shape != (144, 160, 4):
            raise ValueError(f"unexpected screenshot size {rgba.shape}")
        return frame_from_rgba(0, rgba)

    # -- main ----------------------------------------------------------- #
    def run(self) -> dict:
        t0 = time.time()
        print(f"G1 visual localization acceptance — ROM={self.cfg.emulator.rom}")
        collected = [
            self.run_lab_walk(self.args.lab_ops),
            self.run_lab_turn_block(),
            self.run_multi_tile(),
            self.run_live_menu(),
            self.run_pallet_walk(self.args.pallet_ops),
            self.run_transitions(),
            self.run_real_frames(),
        ]
        totals = Metrics()
        for item in collected:
            totals.merge(item)
        totals.pos_error_final = sum(s.get("pos_error_final", 0) for s in self.scenarios)
        elapsed = time.time() - t0
        accuracy = totals.correct_moves / totals.moves_truth if totals.moves_truth else 0.0
        report = {
            "task": "navigation_v2_g1_localization",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rom": str(self.cfg.emulator.rom),
            "states_dir": str(STATES),
            "elapsed_s": round(elapsed, 2),
            "direction_ops": totals.ops,
            "totals": totals.to_dict(),
            "scenarios": self.scenarios,
            "acceptance": {
                "no_false_moves": totals.wrong_moves == 0,
                "move_accuracy_ok": accuracy >= 0.95,
                "no_lost_localization": totals.lost == 0,
                "transitions_detected": totals.transitions_detected,
                "interruptions_correct": totals.interruptions_correct,
            },
            "passed": totals.wrong_moves == 0 and totals.lost == 0 and accuracy >= 0.95,
        }
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        with (ARTIFACTS / "ops.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        for name, rgba in self.screens.items():
            Image.fromarray(rgba, "RGBA").resize((320, 288), Image.NEAREST).save(
                ARTIFACTS / f"{name}.png"
            )
        (ARTIFACTS / "summary.md").write_text(self.render_markdown(report), encoding="utf-8")
        print("\nG1 totals:", json.dumps(totals.to_dict(), indent=1, ensure_ascii=False))
        print(f"elapsed {elapsed:.1f}s, report -> {ARTIFACTS / 'report.json'}")
        return report

    def render_markdown(self, report: dict) -> str:
        totals = report["totals"]
        lines = [
            "# G1 真实视觉定位验收（自动生成）",
            "",
            f"- ROM: `{report['rom']}`",
            f"- 方向操作总数: **{report['direction_ops']}**",
            f"- 真实位移(非传送)次数: {totals['moves_truth']}",
            f"- 正确位移: {totals['correct_moves']}（准确率 {totals['move_accuracy'] * 100:.2f}%）",
            f"- 错误位移(把没走当走了): **{totals['wrong_moves']}**",
            f"- 漏检位移(走了却判 blocked): {totals['missed_moves']}",
            f"- UNKNOWN: {totals['unknown']}（其中真实位移 {totals['unknown_but_moved']}）",
            f"- 定位丢失: {totals['lost']}",
            f"- 区域切换检测: {totals['transitions_detected']}（漏检 {totals['transitions_missed']}）",
            f"- 中断(菜单/对话/战斗)正确: {totals['interruptions_correct']}（误判 {totals['false_interruptions']}）",
            f"- 位置误差: sum={totals['pos_error_sum']} max={totals['pos_error_max']}",
            "",
            "| 场景 | 操作 | 真实位移 | 正确 | 错误 | 漏检 | UNKNOWN | 定位丢失 | 切换漏检 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for item in report["scenarios"]:
            lines.append(
                f"| {item['scenario']} | {item['ops']} | {item.get('moves_truth', 0)} | "
                f"{item.get('correct_moves', 0)} | {item.get('wrong_moves', 0)} | "
                f"{item.get('missed_moves', 0)} | {item.get('unknown', 0)} | "
                f"{item.get('lost', 0)} | {item.get('transitions_missed', 0)} |"
            )
        lines.append("")
        lines.append(f"- 验收通过: **{report['passed']}**")
        return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 real-game visual localization acceptance")
    parser.add_argument("--lab-ops", type=int, default=50, help="random direction ops in the lab")
    parser.add_argument("--pallet-ops", type=int, default=30, help="random direction ops in Pallet Town")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = G1Runner(args)
    try:
        report = runner.run()
    finally:
        runner.close()
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
