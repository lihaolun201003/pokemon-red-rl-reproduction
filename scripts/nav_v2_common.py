"""Shared helpers for the Navigation v2 real-game validation scripts (G2/G3/G4).

These scripts are **validation harnesses**, not part of the agent decision loop.
They may read game RAM through :mod:`llm_agent.progress` (verification-only) to
*measure* drift / relocalization quality; the navigation modules themselves never
touch that data.  Nothing here modifies the ROM, RAM or save files: only legal
PyBoy save states created by scripted gameplay are loaded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_agent.config import AgentConfig  # noqa: E402
from llm_agent.emulator import GameBoyEmulator  # noqa: E402
from llm_agent.navigation import vision  # noqa: E402
from llm_agent.progress import OverworldProbe, probe_overworld  # noqa: E402

FIXTURE_STATES = ROOT / "work" / "nav_v2" / "states"
DEFAULT_STATE = FIXTURE_STATES / "lab_start.state"
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - older/odd consoles
        pass


# --------------------------------------------------------------------------- #
# Recorder: screenshots, step log, map snapshots, area transitions
# --------------------------------------------------------------------------- #

class Recorder:
    """Records evidence from a navigation run (all callbacks are exception-safe)."""

    def __init__(self, out_dir: str | Path, *, label: str = "", screen_every: int = 10,
                 map_every: int = 40) -> None:
        self.out_dir = Path(out_dir)
        self.label = label
        self.screen_every = max(1, int(screen_every))
        self.map_every = max(1, int(map_every))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "screens").mkdir(exist_ok=True)
        (self.out_dir / "maps").mkdir(exist_ok=True)
        self.agent: Any = None
        self.frame_count = 0
        self.step_count = 0
        self.steps: list[dict[str, Any]] = []
        self.reports: list[dict[str, Any]] = []
        self.statuses: list[dict[str, Any]] = []
        self.area_changes: list[dict[str, Any]] = []
        self.screens_saved = 0
        self.maps_saved = 0
        self._last_area = ""
        self._nav_log = (self.out_dir / "navigation_steps.jsonl").open("w", encoding="utf-8")

    # -- callbacks ------------------------------------------------------ #
    def on_frame(self, frame: Any) -> None:
        self.frame_count += 1
        try:
            if self.frame_count % self.screen_every == 1:
                self.save_screen(frame)
        except Exception:  # noqa: BLE001 - recording must never break navigation
            pass

    def on_step(self, step: Any) -> None:
        try:
            payload = step.to_dict() if hasattr(step, "to_dict") else {"step": str(step)}
        except Exception:  # noqa: BLE001
            payload = {"step": str(step)}
        self.step_count += 1
        self.steps.append(payload)
        try:
            motion = payload.get("motion") or {}
            self._nav_log.write(json.dumps({
                "k": "step",
                "t": round(time.time(), 3),
                "dir": payload.get("direction"),
                "result": motion.get("result"),
                "delta": motion.get("delta"),
                "area": payload.get("area_id"),
                "before": payload.get("pose_before"),
                "after": payload.get("pose_after"),
                "confidence": motion.get("confidence"),
                "evidence": {k: motion.get("evidence", {}).get(k) for k in ("shift_score", "diff_ratio")},
            }, ensure_ascii=False) + "\n")
            self._nav_log.flush()
        except Exception:  # noqa: BLE001
            pass
        if self.step_count % self.map_every == 0:
            self.save_map()

    def on_status(self, status: Any) -> None:
        if not isinstance(status, dict):
            return
        self.statuses.append({k: v for k, v in status.items() if k != "world"})
        area = str(status.get("area_id") or "")
        if area and area != self._last_area:
            self._last_area = area
            self.area_changes.append({
                "at": round(time.time(), 3),
                "area_id": area,
                "area_kind": status.get("area_kind"),
                "pose": status.get("pose"),
                "at_step": self.step_count,
            })
            self.save_map()

    # -- helpers -------------------------------------------------------- #
    def on_report(self, report: Any, *, phase: str) -> dict[str, Any]:
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report or {})
        payload["phase"] = phase
        payload["recorded_at"] = round(time.time(), 3)
        self.reports.append(payload)
        try:
            if self.agent is not None and self.area_changes:
                pass
        except Exception:  # noqa: BLE001
            pass
        self.save_map()
        return payload

    def save_screen(self, frame: Any) -> Path:
        path = self.out_dir / "screens" / f"frame_{int(getattr(frame, 'frame_index', self.frame_count)):09d}.png"
        try:
            Image.fromarray(np.array(frame.rgba, dtype=np.uint8), "RGBA").save(path)
            self.screens_saved += 1
        except Exception:  # noqa: BLE001
            pass
        return path

    def save_map(self) -> None:
        if self.agent is None:
            return
        try:
            from llm_agent.navigation import viz

            controller = getattr(self.agent, "controller", None)
            grid = _current_grid(self.agent)
            if grid is None:
                return
            localizer = getattr(self.agent, "localizer", None)
            pose = getattr(localizer, "pose", None)
            path = getattr(controller, "last_path", None)
            target = getattr(controller, "last_target", None)
            image = viz.render_map_image(
                grid, pose=pose, path=path, target=target, cell_px=8, max_cells=40, legend=True
            )
            Image.fromarray(np.array(image, dtype=np.uint8), "RGBA").save(
                self.out_dir / "maps" / f"map_step_{self.step_count:05d}.png"
            )
            self.maps_saved += 1
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            self._nav_log.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Runtime construction / measurement helpers
# --------------------------------------------------------------------------- #

@dataclass
class Harness:
    cfg: AgentConfig
    emu: GameBoyEmulator
    agent: Any
    recorder: Recorder
    out_dir: Path
    state: Path
    started_at: float = field(default_factory=time.time)

    def close(self) -> None:
        try:
            self.agent.save()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.recorder.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.emu.close()
        except Exception:  # noqa: BLE001
            pass


def _prepare_out(out_dir: Path, *, fresh: bool) -> None:
    if fresh and out_dir.exists():
        resolved = out_dir.resolve()
        root = ROOT.resolve()
        if root not in resolved.parents and resolved != root:
            raise SystemExit(f"refusing to delete outside the project: {resolved}")
        shutil.rmtree(resolved)
    out_dir.mkdir(parents=True, exist_ok=True)


def build_harness(
    out_dir: str | Path,
    *,
    state: str | Path = DEFAULT_STATE,
    fresh: bool = True,
    label: str = "",
    screen_every: int = 10,
    map_every: int = 40,
) -> Harness:
    """Create emulator + NavigationAgent with a fresh persistence root."""
    utf8_stdout()
    out = Path(out_dir)
    _prepare_out(out, fresh=fresh)
    state_path = Path(state)
    if not state_path.is_absolute():
        state_path = FIXTURE_STATES / state_path
    if not state_path.is_file():
        raise SystemExit(
            f"missing save state {state_path}; run work/nav_v2/make_states.py first"
        )
    cfg = AgentConfig.load()
    cfg.emulator.window = "null"
    cfg.emulator.settle_frames = 2
    cfg.navigation.world_root = str(out / "navigation")
    recorder = Recorder(out, label=label, screen_every=screen_every, map_every=map_every)
    emu = GameBoyEmulator(cfg.emulator).start(state_path)
    from llm_agent.navigation.navigation_agent import NavigationAgent

    agent = NavigationAgent(
        emulator=emu,
        world_root=str(out / "navigation"),
        config=cfg,
        on_frame=recorder.on_frame,
        on_step=recorder.on_step,
        on_status=recorder.on_status,
    )
    recorder.agent = agent
    agent.attach()
    recorder.save_map()
    return Harness(cfg=cfg, emu=emu, agent=agent, recorder=recorder, out_dir=out, state=state_path)


def ram_truth(emu: GameBoyEmulator) -> OverworldProbe:
    """Verification-only RAM ground truth (NEVER fed to navigation)."""
    pyboy = getattr(emu, "_pyboy", None)
    if pyboy is None:
        raise RuntimeError("emulator is not started")
    return probe_overworld(pyboy)


def screen_signature(emu: GameBoyEmulator) -> str:
    return vision.screen_signature(vision.frame_luma(emu.observe()))


def wait_stable(emu: GameBoyEmulator, *, max_frames: int = 360, step: int = 8,
                need_same: int = 3) -> int:
    """Tick until the screen signature is frozen for ``need_same`` chunks."""
    prev = screen_signature(emu)
    total = 0
    same = 0
    while total < max_frames:
        frame = emu.wait(step)
        total += step
        sig = vision.screen_signature(vision.frame_luma(frame))
        if sig == prev and not frame.is_black:
            same += 1
            if same >= need_same:
                return total
        else:
            same = 0
        prev = sig
    return total


def last_moved_direction(recorder: Recorder) -> str | None:
    for step in reversed(recorder.steps):
        motion = step.get("motion") or {}
        if motion.get("result") == "moved":
            direction = step.get("direction")
            if direction in OPPOSITE:
                return direction
    return None


def last_step_direction(recorder: Recorder) -> str | None:
    for step in reversed(recorder.steps):
        direction = step.get("direction")
        if direction in OPPOSITE:
            return direction
    return None


def reverse_direction(direction: str | None) -> str | None:
    return OPPOSITE.get(direction or "")


def walkable_cells(agent: Any, area_id: str | None = None) -> list[tuple[int, int]]:
    try:
        area = agent.world.area(area_id or None)
    except Exception:  # noqa: BLE001
        return []
    grid = getattr(area, "grid", None)
    if grid is None:
        return []
    cells = []
    for key, cell in (getattr(grid, "cells", {}) or {}).items():
        if str(getattr(getattr(cell, "state", None), "value", "")) == "walkable":
            cells.append((int(key[0]), int(key[1])))
    return cells


def far_cell(agent: Any, area_id: str | None = None, *, from_pose: tuple[int, int] = (0, 0)):
    cells = walkable_cells(agent, area_id)
    if not cells:
        return None
    return max(cells, key=lambda cell: abs(cell[0] - from_pose[0]) + abs(cell[1] - from_pose[1]))


def _current_grid(agent: Any) -> Any:
    """Return the current area grid (supports both method and property APIs)."""
    try:
        world = getattr(agent, "world", None)
        if world is None:
            return None
        grid = getattr(world, "current_grid", None)
        return grid() if callable(grid) else grid
    except Exception:  # noqa: BLE001
        return None


def frontier_kinds(agent: Any) -> dict[str, int]:
    """Count reachable frontier candidates by kind (``frontier`` / ``exit``...)."""
    try:
        explorer = getattr(agent.controller, "explorer", None)
        grid = _current_grid(agent)
        pose = agent.localizer.pose
        if explorer is None or grid is None:
            return {}
        candidates = explorer.frontier_candidates(grid, tuple(pose.cell)) or []
        return dict(Counter(str(getattr(item, "kind", "frontier")) for item in candidates))
    except Exception:  # noqa: BLE001
        return {}


def room_frontiers_left(agent: Any) -> int:
    """Number of reachable non-exit frontiers left in the current room."""
    counts = frontier_kinds(agent)
    return int(counts.get("frontier", 0)) + int(counts.get("unreachable_probe", 0))


def set_exit_last_resort(agent: Any, enabled: bool) -> float | None:
    """Temporarily make EXIT cells the last exploration choice (harness policy).

    Returns the previous ``exit_bonus`` so the caller can restore it; ``None``
    when the explorer does not expose the config.
    """
    try:
        explorer = getattr(agent.controller, "explorer", None)
        config = getattr(explorer, "config", None)
        if config is None or not hasattr(config, "exit_bonus"):
            return None
        previous = float(config.exit_bonus)
        config.exit_bonus = -50.0 if enabled else previous
        return previous
    except Exception:  # noqa: BLE001
        return None


def restore_exit_bonus(agent: Any, previous: float | None) -> None:
    if previous is None:
        return
    try:
        agent.controller.explorer.config.exit_bonus = float(previous)
    except Exception:  # noqa: BLE001
        pass


def drift(agent: Any, emu: GameBoyEmulator, anchor: dict[str, Any]) -> int:
    """|visual pose delta - RAM truth delta| inside one area visit (verification only)."""
    try:
        truth = ram_truth(emu)
        pose = agent.localizer.pose
        expected = (
            int(anchor["pose"][0]) + int(truth.x) - int(anchor["truth"]["x"]),
            int(anchor["pose"][1]) + int(truth.y) - int(anchor["truth"]["y"]),
        )
        return abs(int(pose.x) - expected[0]) + abs(int(pose.y) - expected[1])
    except Exception:  # noqa: BLE001
        return -1


def anchor_now(agent: Any, emu: GameBoyEmulator) -> dict[str, Any]:
    truth = ram_truth(emu)
    pose = agent.localizer.pose
    return {"truth": {"map_id": truth.map_id, "x": truth.x, "y": truth.y},
            "pose": [int(pose.x), int(pose.y)], "area_id": agent.status().get("area_id")}


def area_signature(agent: Any) -> tuple[int, dict[str, int]]:
    summary = agent.world_summary()
    counts: dict[str, int] = {}
    for area in summary.get("areas") or []:
        counts[str(area.get("area_id"))] = int(area.get("cells") or 0)
    return int(summary.get("areas_total") or 0), counts


def render_map_png(agent: Any, path: str | Path, *, cell_px: int = 10) -> str | None:
    try:
        from llm_agent.navigation import viz

        grid = _current_grid(agent)
        if grid is None:
            return None
        image = viz.render_map_image(
            grid, pose=agent.localizer.pose, cell_px=cell_px, max_cells=64, legend=True
        )
        Image.fromarray(np.array(image, dtype=np.uint8), "RGBA").save(path)
        return str(path)
    except Exception:  # noqa: BLE001
        return None


def render_world_png(agent: Any, path: str | Path) -> str | None:
    try:
        from llm_agent.navigation import viz

        image = viz.render_world_image(agent.world, width=480, height=240)
        Image.fromarray(np.array(image, dtype=np.uint8), "RGBA").save(path)
        return str(path)
    except Exception:  # noqa: BLE001
        return None


def screenshot_montage(out_dir: Path, out_path: Path, *, cols: int = 4, scale: int = 2,
                       limit: int = 24) -> str | None:
    screens = sorted((out_dir / "screens").glob("frame_*.png"))[:limit]
    if not screens:
        return None
    thumbs = []
    for path in screens:
        image = Image.open(path).convert("RGB")
        if scale != 1:
            image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
        thumbs.append(image)
    if not thumbs:
        return None
    rows = (len(thumbs) + cols - 1) // cols
    cell_w, cell_h = thumbs[0].width, thumbs[0].height
    canvas = Image.new("RGB", (cols * cell_w + 4 * (cols - 1), rows * cell_h + 4 * (rows - 1)), (32, 32, 32))
    for index, image in enumerate(thumbs):
        x = (index % cols) * (cell_w + 4)
        y = (index // cols) * (cell_h + 4)
        canvas.paste(image, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def run_map_gallery(world_dir: Path, out_dir: Path) -> dict[str, Any]:
    script = ROOT / "scripts" / "nav_v2_map_gallery.py"
    result: dict[str, Any] = {"ok": False}
    try:
        completed = subprocess.run(
            [sys.executable, str(script), "--world-dir", str(world_dir), "--out", str(out_dir)],
            capture_output=True, text=True, timeout=180,
        )
        result = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[:2000],
            "stderr": completed.stderr.strip()[:2000],
        }
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def metric_int(stats: dict[str, Any], key: str, default: int = 0) -> int:
    value = stats.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def metric_float(stats: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = stats.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def coverage_of(report: Any) -> dict[str, Any]:
    coverage = getattr(report, "coverage", None)
    if coverage is None:
        return {}
    return coverage.to_dict() if hasattr(coverage, "to_dict") else dict(coverage)


__all__ = [
    "ROOT",
    "FIXTURE_STATES",
    "DEFAULT_STATE",
    "OPPOSITE",
    "Harness",
    "Recorder",
    "anchor_now",
    "area_signature",
    "build_harness",
    "coverage_of",
    "drift",
    "far_cell",
    "last_moved_direction",
    "last_step_direction",
    "metric_float",
    "metric_int",
    "ram_truth",
    "render_map_png",
    "render_world_png",
    "reverse_direction",
    "run_map_gallery",
    "screenshot_montage",
    "utf8_stdout",
    "wait_stable",
    "walkable_cells",
    "write_json",
    "write_text",
]
