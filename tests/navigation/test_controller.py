"""Unit tests for the verified navigation controller (nav-control).

Everything here runs without PyBoy, ROM, network or API key: the emulator,
localizer, grid, world model, planner and explorer are small local fakes that
implement exactly the documented duck-typed contracts.  The last test is an
optional real-integration smoke test that skips until the nav-map modules
(``grid_map.py`` / ``world_model.py``) land.

Covered invariants:
* the controller is the only direction-button presser (fake emulator records all
  presses),
* exactly-once map updates for MOVED / BLOCKED / TURNED_ONLY,
* NO map/coordinate change for UNKNOWN / INTERRUPTED / TRANSITION,
* consecutive-UNKNOWN relocalization + honest ``localization_lost``,
* replan-once-per-blocked-edge (and honest ``blocked`` when no alternative),
* transition handling: strong match switches area/restores pose/records
  connection; weak evidence creates a new area and never merges,
* exploration budgets (steps/time/no-progress) and event interruption,
* ``navigate_to`` target forms (cell / exit / landmark / return_to_entry /
  other-area via a recorded connection).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pytest

from llm_agent.navigation.types import (
    DIRECTIONS,
    DIRECTION_DELTAS,
    AreaMatch,
    CellState,
    CoverageStats,
    EdgeState,
    ExploreStatus,
    ExploreTarget,
    FrameContext,
    FrameView,
    MotionEstimate,
    MotionResult,
    PathPlan,
    PlayerPose,
    direction_of,
    step_delta,
)
from llm_agent.navigation.controller import (
    ControllerConfig,
    NavigationController,
)

# --------------------------------------------------------------------------- #
# Motion estimate builders (scripted localizer input)
# --------------------------------------------------------------------------- #


def moved(direction: str, tiles: int = 1) -> MotionEstimate:
    return MotionEstimate(
        result=MotionResult.MOVED,
        direction=direction,
        delta=step_delta(direction, tiles),
        tiles_moved=tiles,
        camera_shift=(-step_delta(direction, tiles)[0], -step_delta(direction, tiles)[1]),
        confidence=0.98,
        evidence={"test": "moved"},
    )


def blocked(direction: str) -> MotionEstimate:
    return MotionEstimate(
        result=MotionResult.BLOCKED, direction=direction, confidence=0.99
    )


def turned_only(direction: str) -> MotionEstimate:
    return MotionEstimate(
        result=MotionResult.TURNED_ONLY, direction=direction, confidence=0.9
    )


def unknown(direction: str = "up") -> MotionEstimate:
    return MotionEstimate(result=MotionResult.UNKNOWN, direction=direction, confidence=0.2)


def interrupted(
    direction: str = "up", context: FrameContext = FrameContext.DIALOG
) -> MotionEstimate:
    return MotionEstimate(
        result=MotionResult.INTERRUPTED,
        direction=direction,
        context=context,
        evidence={"event": context.value},
    )


def transition(direction: str = "up") -> MotionEstimate:
    return MotionEstimate(
        result=MotionResult.TRANSITION,
        direction=direction,
        context=FrameContext.TRANSITION,
        confidence=0.0,
    )


# --------------------------------------------------------------------------- #
# Plan builders
# --------------------------------------------------------------------------- #


def plan_from_cells(cells: Iterable[tuple[int, int]]) -> PathPlan:
    chain = [(int(x), int(y)) for x, y in cells]
    assert chain, "a plan needs at least one cell"
    directions: list[str] = []
    for a, b in zip(chain, chain[1:]):
        direction = direction_of((b[0] - a[0], b[1] - a[1]))
        assert direction is not None, f"plan cells must be 4-connected: {a} -> {b}"
        directions.append(direction)
    return PathPlan(
        cells=chain,
        directions=directions,
        cost=float(len(directions)),
        reachable=True,
        reason="test plan",
    )


def straight_plan(start: tuple[int, int], goal: tuple[int, int]) -> PathPlan:
    """Deterministic x-then-y plan (the fakes use open grids)."""
    x, y = int(start[0]), int(start[1])
    cells: list[tuple[int, int]] = [(x, y)]
    while x != int(goal[0]):
        x += 1 if int(goal[0]) > x else -1
        cells.append((x, y))
    while y != int(goal[1]):
        y += 1 if int(goal[1]) > y else -1
        cells.append((x, y))
    return plan_from_cells(cells)


# --------------------------------------------------------------------------- #
# Fakes (duck-typed per NAV_CONTRACT.md sections 2-5)
# --------------------------------------------------------------------------- #


class FakeEmulator:
    """Minimal emulator surface: observe/press/wait/frame_count/started."""

    def __init__(self) -> None:
        self.presses: list[str] = []
        self._frame_count = 0
        self.started = True

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def observe(self) -> Any:
        return SimpleNamespace(frame_index=self._frame_count)

    def press(self, button: str, frames: int | None = None) -> Any:
        self.presses.append(str(button))
        self._frame_count += 24
        return SimpleNamespace(frame_index=self._frame_count)

    def wait(self, frames: int = 8) -> Any:
        self._frame_count += max(1, int(frames))
        return SimpleNamespace(frame_index=self._frame_count)


class FakeLocalizer:
    """Scripted motion estimator + pose tracker (same semantics as Localizer)."""

    def __init__(
        self,
        script: Iterable[MotionEstimate] | None = None,
        pose: tuple[int, int] = (0, 0),
        facing: str = "down",
    ) -> None:
        self.script: list[MotionEstimate] = list(script or [])
        self._pose = PlayerPose(
            x=int(pose[0]), y=int(pose[1]), facing=facing, confidence=1.0, valid=True
        )
        self.area_id = ""
        self.context = FrameContext.OVERWORLD
        self.signature = "fake-signature"
        self.signature_sequence: list[str] = []
        #: optional raw luma frames attached to views (transition pixel settling)
        self.luma: np.ndarray | None = None
        self.luma_sequence: list[np.ndarray] = []
        self.is_black = False
        self.observe_calls: list[Any] = []
        self.update_calls: list[str] = []
        self._last_view: FrameView | None = None
        self._unknown_streak = 0

    # -- Localizer API --------------------------------------------------- #
    @property
    def pose(self) -> PlayerPose:
        return self._pose

    @property
    def last_view(self) -> FrameView | None:
        return self._last_view

    def reset(
        self,
        *,
        area_id: str = "",
        cell: tuple[int, int] = (0, 0),
        facing: str | None = None,
    ) -> None:
        self.area_id = str(area_id)
        self._pose = PlayerPose(
            x=int(cell[0]),
            y=int(cell[1]),
            facing=facing if facing in DIRECTIONS else None,
            confidence=1.0,
            valid=True,
        )
        self._last_view = None
        self._unknown_streak = 0

    def set_pose(
        self,
        x: int,
        y: int,
        *,
        facing: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        self._pose = PlayerPose(
            x=int(x),
            y=int(y),
            facing=facing if facing in DIRECTIONS else self._pose.facing,
            confidence=float(confidence),
            valid=True,
        )
        self._unknown_streak = 0

    def invalidate(self, reason: str = "") -> None:
        self._pose.valid = False
        self._pose.confidence = 0.0

    def observe(self, frame: Any, *, frame_context: FrameContext | None = None) -> FrameView:
        self.observe_calls.append(frame)
        if self.signature_sequence:
            signature = self.signature_sequence.pop(0)
        else:
            signature = self.signature
        view = FrameView(
            frame_index=int(getattr(frame, "frame_index", 0) or 0),
            signature=signature,
            grid_offset=(0, 0),
            context=frame_context if frame_context is not None else self.context,
            screen_event="none",
            is_black=self.is_black,
        )
        luma = None
        if self.luma_sequence:
            luma = self.luma_sequence.pop(0)
        elif self.luma is not None:
            luma = self.luma
        if luma is not None:
            view.notes["luma"] = np.asarray(luma, dtype=np.uint8).copy()
        self._last_view = view
        return view

    def update(
        self,
        prev: FrameView,
        cur: FrameView,
        direction: str,
        *,
        requested_tiles: int = 1,
    ) -> MotionEstimate:
        self.update_calls.append(direction)
        if self.script:
            estimate = copy.deepcopy(self.script.pop(0))
        else:
            estimate = unknown(direction)
        estimate.direction = direction
        if estimate.context is FrameContext.OVERWORLD and self.context is not FrameContext.OVERWORLD:
            estimate.context = self.context
        if estimate.result is MotionResult.MOVED:
            self._pose.x += int(estimate.delta[0])
            self._pose.y += int(estimate.delta[1])
            self._pose.facing = direction
            self._pose.valid = True
            self._pose.confidence = 1.0
            self._unknown_streak = 0
        elif estimate.result is MotionResult.UNKNOWN:
            self._unknown_streak += 1
            if self._unknown_streak >= 2:
                self._pose.valid = False
                self._pose.confidence = 0.0
        self._last_view = cur
        return estimate


class FakeCell:
    def __init__(self, x: int, y: int) -> None:
        self.x = int(x)
        self.y = int(y)
        self.state = CellState.UNKNOWN
        self.terrain = ""
        self.tile_hashes: list[str] = []
        self.observed_count = 0
        self.visit_count = 0
        self.confidence = 0.0
        self.last_seen_frame = -1
        self.obstacle_evidence = 0
        self.free_evidence = 0
        self.labels: list[str] = []


class FakeGrid:
    """Counting fake of the documented GridMap surface."""

    def __init__(self, area_id: str = "area_0") -> None:
        self.area_id = area_id
        self.cells: dict[tuple[int, int], FakeCell] = {}
        self.edges: dict[Any, EdgeState] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.observe_calls: list[tuple[int, tuple[int, int]]] = []
        self.kind = "unknown"

    # -- lookups --------------------------------------------------------- #
    def cell(self, x: int, y: int) -> FakeCell | None:
        return self.cells.get((int(x), int(y)))

    def get_cell(self, x: int, y: int) -> FakeCell:
        key = (int(x), int(y))
        if key not in self.cells:
            self.cells[key] = FakeCell(*key)
        return self.cells[key]

    def size(self) -> int:
        return len(self.cells)

    # -- exactly-once confirmations -------------------------------------- #
    def confirm_move(
        self,
        from_cell: tuple[int, int],
        direction: str,
        *,
        tiles: int = 1,
        frame_index: int = 0,
    ) -> tuple[int, int]:
        self.calls.append(("move", tuple(from_cell), direction, int(tiles)))
        dx, dy = DIRECTION_DELTAS[direction]
        target = (int(from_cell[0]) + dx * int(tiles), int(from_cell[1]) + dy * int(tiles))
        cell = self.get_cell(*target)
        cell.state = CellState.WALKABLE
        cell.visit_count += 1
        cell.free_evidence += 1
        self.edges[(tuple(from_cell), target)] = EdgeState.OPEN
        return target

    def confirm_blocked(
        self,
        from_cell: tuple[int, int],
        direction: str,
        *,
        temporary: bool = False,
        frame_index: int = 0,
    ) -> tuple[int, int]:
        self.calls.append(("blocked", tuple(from_cell), direction, bool(temporary)))
        dx, dy = DIRECTION_DELTAS[direction]
        target = (int(from_cell[0]) + dx, int(from_cell[1]) + dy)
        cell = self.get_cell(*target)
        if cell.state is not CellState.WALKABLE:
            cell.state = CellState.TEMP_BLOCKED if temporary else CellState.BLOCKED
        cell.obstacle_evidence += 1
        self.edges[(tuple(from_cell), target)] = EdgeState.BLOCKED
        return target

    def confirm_turn(self, from_cell: tuple[int, int], direction: str) -> None:
        self.calls.append(("turn", tuple(from_cell), direction))

    def mark_exit(
        self, cell: Any, *, direction: str | None = None, confidence: float = 1.0
    ) -> None:
        self.calls.append(("mark_exit", tuple(cell)))

    # -- observation ----------------------------------------------------- #
    def observe_view(self, view: Any, pose: Any, *, frame_index: int = 0) -> dict[str, int]:
        self.observe_calls.append(
            (int(getattr(view, "frame_index", 0) or 0), tuple(pose.cell))
        )
        return {"new_cells": 0, "new_hashes": 0, "observed": 90}

    # -- reporting ------------------------------------------------------- #
    def coverage(self) -> CoverageStats:
        cells = list(self.cells.values())
        walkable = sum(1 for cell in cells if cell.state is CellState.WALKABLE)
        blocked = sum(
            1 for cell in cells if cell.state in (CellState.BLOCKED, CellState.TEMP_BLOCKED)
        )
        walkable_cells = {(cell.x, cell.y) for cell in cells if cell.state is CellState.WALKABLE}
        frontier: set[tuple[int, int]] = set()
        for x, y in walkable_cells:
            for dx, dy in DIRECTION_DELTAS.values():
                neighbour = (x + dx, y + dy)
                if neighbour in walkable_cells:
                    continue
                other = self.cells.get(neighbour)
                if other is None or other.state in (CellState.UNKNOWN, CellState.OBSERVED):
                    frontier.add(neighbour)
        denominator = walkable + len(frontier)
        return CoverageStats(
            area_id=self.area_id,
            cells_total=len(cells),
            observed=sum(1 for cell in cells if cell.state is CellState.OBSERVED),
            walkable=walkable,
            blocked=blocked,
            temp_blocked=0,
            exits=sum(1 for cell in cells if cell.state is CellState.EXIT),
            interactive=0,
            unknown_frontier=len(frontier),
            coverage_ratio=(walkable / denominator) if denominator else 0.0,
            explored_ratio=(walkable / denominator) if denominator else 0.0,
        )

    def render_ascii(self, **_: Any) -> str:
        return "fake-ascii"

    def cells_in_rect(self, *args: Any, **kwargs: Any) -> list[FakeCell]:
        return list(self.cells.values())


class FakeArea:
    def __init__(self, area_id: str, *, kind: str = "unknown") -> None:
        self.area_id = str(area_id)
        self.kind = kind
        self.grid = FakeGrid(area_id)
        self.visits = 0
        self.first_frame = 0
        self.last_frame = 0
        self.entry_cells: list[tuple[int, int]] = []
        self.explored = False
        self.notes: list[str] = []


class FakeWorld:
    """Fake of the documented WorldModel surface (no persistence on disk)."""

    def __init__(self) -> None:
        self.areas: dict[str, FakeArea] = {}
        self.connections: list[Any] = []
        self.current_area = ""
        self.current_pose = (0, 0)
        self.match_results: list[AreaMatch] = []
        self.match_calls = 0
        self.save_count = 0
        self.load_calls = 0

    # -- registry -------------------------------------------------------- #
    def create_area(
        self, *, kind: str = "unknown", frame_index: int = 0, area_id: str | None = None
    ) -> FakeArea:
        identifier = str(area_id) if area_id else f"area_{len(self.areas)}"
        record = FakeArea(identifier, kind=kind)
        record.first_frame = int(frame_index)
        self.areas[identifier] = record
        return record

    def area(self, area_id: str | None = None) -> FakeArea:
        identifier = str(area_id) if area_id else self.current_area
        if identifier not in self.areas:
            raise KeyError(f"unknown area {identifier!r}")
        return self.areas[identifier]

    def set_current(self, area_id: str, pose: tuple[int, int]) -> None:
        self.current_area = str(area_id)
        self.current_pose = (int(pose[0]), int(pose[1]))

    def record_visit(self, area_id: str, *, frame_index: int = 0) -> None:
        self.areas[str(area_id)].visits += 1

    def add_connection(self, connection: Any) -> None:
        self.connections.append(connection)

    def match_view(
        self,
        view: Any,
        *,
        candidate_area_ids: list[str] | None = None,
        min_score: float = 0.55,
        max_results: int = 4,
    ) -> list[AreaMatch]:
        self.match_calls += 1
        return list(self.match_results)

    # -- persistence ----------------------------------------------------- #
    def save(self) -> None:
        self.save_count += 1

    def load(self) -> bool:
        self.load_calls += 1
        return False

    def summary(self) -> dict[str, Any]:
        return {"areas_total": len(self.areas), "connections": len(self.connections)}


class FakePathfinder:
    """Scripted planner: pops queued plans, otherwise plans a straight route."""

    def __init__(self, plans: Iterable[Any] | None = None) -> None:
        self.plans: list[Any] = list(plans or [])
        self.calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def plan(
        self,
        grid: Any,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        allow_unknown: bool = True,
        allow_temp_blocked: bool = False,
    ) -> PathPlan:
        self.calls.append((tuple(start), tuple(goal)))
        if self.plans:
            item = self.plans.pop(0)
            if callable(item):
                return item(tuple(start), tuple(goal))
            return copy.deepcopy(item)
        return straight_plan(start, goal)


class FakeExplorer:
    """Scripted exploration policy with attempt history."""

    def __init__(
        self,
        targets: Iterable[Any] | None = None,
        finish_sequence: Iterable[tuple[bool, str, dict]] | None = None,
    ) -> None:
        self.targets: list[Any] = list(targets or [])
        self.candidates: list[Any] = []
        self.finish_sequence: list[tuple[bool, str, dict]] = list(finish_sequence or [])
        self.attempts: list[tuple[tuple[int, int], str]] = []
        self.config = SimpleNamespace(no_progress_limit=30)

    def should_finish(
        self, grid: Any, start: tuple[int, int]
    ) -> tuple[bool, str, dict]:
        if self.finish_sequence:
            return self.finish_sequence.pop(0)
        return (False, "", {})

    def choose_target(
        self, grid: Any, start: tuple[int, int], *, step_index: int = 0
    ) -> ExploreTarget | None:
        if self.targets:
            return self.targets.pop(0)
        return None

    def frontier_candidates(
        self, grid: Any, start: tuple[int, int]
    ) -> list[Any]:
        return list(self.candidates)

    def on_attempt(self, target: Any, *, step_index: int, outcome: str) -> None:
        self.attempts.append((tuple(getattr(target, "cell", (0, 0))), str(outcome)))


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


class Harness:
    """Bundle of controller + fakes for one test."""

    def __init__(
        self,
        *,
        script: Iterable[MotionEstimate] | None = None,
        pose: tuple[int, int] = (5, 5),
        facing: str = "down",
        plans: Iterable[Any] | None = None,
        targets: Iterable[Any] | None = None,
        finish_sequence: Iterable[tuple[bool, str, dict]] | None = None,
        config: ControllerConfig | None = None,
        world: FakeWorld | None = None,
        on_frame: Any = None,
        on_step: Any = None,
    ) -> None:
        self.emulator = FakeEmulator()
        self.world = world if world is not None else FakeWorld()
        if not self.world.areas:
            area = self.world.create_area()
            self.world.set_current(area.area_id, pose)
        self.localizer = FakeLocalizer(script=script, pose=pose, facing=facing)
        self.localizer.area_id = self.world.current_area
        self.pathfinder = FakePathfinder(plans)
        self.explorer = FakeExplorer(targets=targets, finish_sequence=finish_sequence)
        self.controller = NavigationController(
            emulator=self.emulator,
            world=self.world,
            localizer=self.localizer,
            pathfinder=self.pathfinder,
            explorer=self.explorer,
            config=config or ControllerConfig(),
            on_frame=on_frame,
            on_step=on_step,
        )

    @property
    def grid(self) -> FakeGrid:
        return self.world.area().grid

    def move_calls(self) -> list[tuple[Any, ...]]:
        return [call for call in self.grid.calls if call[0] == "move"]

    def blocked_calls(self) -> list[tuple[Any, ...]]:
        return [call for call in self.grid.calls if call[0] == "blocked"]

    def turn_calls(self) -> list[tuple[Any, ...]]:
        return [call for call in self.grid.calls if call[0] == "turn"]


def _match(
    area_id: str,
    pose: tuple[int, int],
    *,
    score: float = 0.9,
    margin: float = 0.25,
    second_score: float = 0.65,
) -> AreaMatch:
    return AreaMatch(
        area_id=area_id,
        pose=(int(pose[0]), int(pose[1])),
        score=float(score),
        compared=20,
        matched=18,
        second_score=float(second_score),
        margin=float(margin),
    )


# --------------------------------------------------------------------------- #
# Observation / press plumbing
# --------------------------------------------------------------------------- #


def test_observe_builds_frame_view_and_calls_on_frame():
    frames: list[Any] = []
    harness = Harness(on_frame=frames.append)
    view = harness.controller.observe()
    assert isinstance(view, FrameView)
    assert harness.controller.last_view is view
    assert len(frames) == 1
    assert frames[0].frame_index == view.frame_index


def test_move_once_presses_exactly_one_direction_button():
    harness = Harness(script=[moved("right")])
    harness.controller.move_once("right")
    assert harness.emulator.presses == ["right"]


# --------------------------------------------------------------------------- #
# Exactly-once map updates
# --------------------------------------------------------------------------- #


def test_moved_confirms_move_exactly_once():
    harness = Harness(script=[moved("right")], pose=(5, 5))
    estimate = harness.controller.move_once("right")
    assert estimate.result is MotionResult.MOVED
    assert harness.move_calls() == [("move", (5, 5), "right", 1)]
    assert harness.blocked_calls() == []
    assert harness.turn_calls() == []
    assert harness.localizer.pose.cell == (6, 5)
    assert harness.world.current_pose == (6, 5)
    assert harness.grid.cell(6, 5).state is CellState.WALKABLE
    assert harness.grid.cell(6, 5).visit_count == 1
    assert len(harness.grid.observe_calls) == 1


def test_moved_multitile_is_recorded_with_actual_tile_count():
    harness = Harness(script=[moved("right", 2)], pose=(5, 5))
    harness.controller.config.max_tiles_per_step = 2
    estimate = harness.controller.move_once("right")
    assert estimate.result is MotionResult.MOVED
    assert harness.move_calls() == [("move", (5, 5), "right", 2)]
    assert harness.localizer.pose.cell == (7, 5)


def test_blocked_confirms_blocked_once_without_moving():
    harness = Harness(script=[blocked("up")], pose=(5, 5))
    estimate = harness.controller.move_once("up")
    assert estimate.result is MotionResult.BLOCKED
    assert harness.blocked_calls() == [("blocked", (5, 5), "up", False)]
    assert harness.move_calls() == []
    assert distance(harness.localizer.pose.cell, (5, 5)) == 0
    assert len(harness.grid.observe_calls) == 1


def test_turned_only_confirms_turn_plus_blocked_evidence():
    harness = Harness(script=[turned_only("up")], pose=(5, 5))
    estimate = harness.controller.move_once("up")
    assert estimate.result is MotionResult.TURNED_ONLY
    assert harness.turn_calls() == [("turn", (5, 5), "up")]
    # a turn may be the first half of turn-then-step, so the evidence is temporary
    assert harness.blocked_calls() == [("blocked", (5, 5), "up", True)]
    assert harness.move_calls() == []
    assert harness.localizer.pose.cell == (5, 5)


def test_unknown_makes_no_map_or_coordinate_change():
    harness = Harness(script=[unknown("left"), unknown("left")], pose=(5, 5))
    first = harness.controller.move_once("left")
    assert first.result is MotionResult.UNKNOWN
    assert harness.grid.calls == []
    assert harness.grid.observe_calls == []
    assert harness.localizer.pose.cell == (5, 5)
    assert harness.controller.stats()["unknown"] == 1
    assert harness.controller.status()["consecutive_unknown"] == 1
    second = harness.controller.move_once("left")
    assert second.result is MotionResult.UNKNOWN
    assert harness.grid.calls == []
    assert harness.controller.status()["consecutive_unknown"] == 2
    assert harness.localizer.pose.valid is False  # localizer invalidates on streak


def test_inconsistent_motion_estimate_is_downgraded_to_unknown():
    bad = MotionEstimate(
        result=MotionResult.MOVED,
        direction="right",
        delta=(0, 2),  # not parallel to the pressed direction
        tiles_moved=2,
        confidence=0.99,
    )
    harness = Harness(script=[bad], pose=(5, 5))
    estimate = harness.controller.move_once("right")
    assert estimate.result is MotionResult.UNKNOWN
    assert harness.grid.calls == []
    assert harness.localizer.pose.cell == (5, 5)  # pose restored, no silent drift
    assert estimate.evidence.get("inconsistent_delta") == [0, 2]


def test_move_once_never_presses_through_a_dialog():
    harness = Harness(pose=(5, 5))
    harness.localizer.context = FrameContext.DIALOG
    harness.controller.observe()
    estimate = harness.controller.move_once("up")
    assert estimate.result is MotionResult.INTERRUPTED
    assert estimate.evidence.get("no_press") is True
    assert harness.emulator.presses == []
    assert harness.grid.calls == []
    assert harness.controller.paused is True
    assert harness.controller.last_event is not None
    assert harness.controller.last_event.kind.value == "dialog_event"


def test_move_once_leaves_no_button_held_and_one_frame_step():
    harness = Harness(script=[moved("down")])
    harness.controller.move_once("down")
    # FakeEmulator records one press; a real emulator releases the key inside press().
    assert harness.emulator.presses == ["down"]
    assert harness.emulator.frame_count == 24


# --------------------------------------------------------------------------- #
# follow_path: events, transitions, replans, localization
# --------------------------------------------------------------------------- #


def test_follow_path_executes_plan_and_counts_moves():
    harness = Harness(script=[moved("right"), moved("right")], pose=(5, 5))
    plan = straight_plan((5, 5), (7, 5))
    result = harness.controller.follow_path(plan)
    assert result.status == "ok"
    assert result.steps == 2
    assert result.moves == 2
    assert harness.localizer.pose.cell == (7, 5)
    assert len(harness.move_calls()) == 2


def test_follow_path_completes_exactly_at_step_budget():
    """Regression (nav-planner): a path that fits the budget must end ``ok``.

    The completion check has to come before the budget check, otherwise a path of
    exactly ``max_steps`` verified moves reports ``step_limit``.
    """
    harness = Harness(script=[moved("up"), moved("up")], pose=(5, 5))
    plan = straight_plan((5, 5), (5, 3))
    result = harness.controller.follow_path(plan, max_steps=2)
    assert result.status == "ok", result.reason
    assert result.steps == 2
    assert harness.localizer.pose.cell == (5, 3)
    # one step less must still report the honest step limit
    harness2 = Harness(script=[moved("up")], pose=(5, 5))
    result2 = harness2.controller.follow_path(plan, max_steps=1)
    assert result2.status == "step_limit"
    assert harness2.localizer.pose.cell == (5, 4)


def test_follow_path_retries_turn_then_step_once():
    harness = Harness(
        script=[turned_only("right"), moved("right"), moved("right")], pose=(5, 5)
    )
    result = harness.controller.follow_path(straight_plan((5, 5), (7, 5)))
    assert result.status == "ok", result.reason
    assert result.steps == 3  # turn, retried step, second step
    assert harness.localizer.pose.cell == (7, 5)
    assert harness.turn_calls() == [("turn", (5, 5), "right")]
    assert len(harness.move_calls()) == 2


def test_last_path_and_target_are_published_for_the_viewer():
    harness = Harness(script=[moved("right"), moved("right")], pose=(5, 5))
    plan = straight_plan((5, 5), (7, 5))
    harness.controller.follow_path(plan)
    assert harness.controller.last_path == [(5, 5), (6, 5), (7, 5)]
    assert harness.controller.last_target == (7, 5)
    status = harness.controller.status()
    assert status["last_path"] == [[5, 5], [6, 5], [7, 5]]
    assert status["last_target"] == [7, 5]
    # idle exploration clears the viewer route again
    idle = Harness(finish_sequence=[(True, "done", {})])
    idle.controller.explore_current_area(max_steps=5)
    assert idle.controller.last_path == []
    assert idle.controller.last_target is None


def test_follow_path_stops_on_dialog_event_and_reports_it():
    harness = Harness(
        script=[moved("right"), interrupted("right", FrameContext.DIALOG)], pose=(5, 5)
    )
    plan = straight_plan((5, 5), (7, 5))
    result = harness.controller.follow_path(plan)
    assert result.status == "event"
    assert result.event is not None
    assert result.event.kind.value == "dialog_event"
    assert result.event.message
    assert result.steps == 2
    assert len(harness.move_calls()) == 1  # the interrupted press changed no map
    assert harness.controller.paused is True  # pause_on_event default


def test_follow_path_stops_on_battle_event():
    harness = Harness(
        script=[interrupted("right", FrameContext.BATTLE)], pose=(5, 5)
    )
    result = harness.controller.follow_path(straight_plan((5, 5), (6, 5)))
    assert result.status == "event"
    assert result.event.kind.value == "battle_event"
    assert harness.grid.calls == []


def test_follow_path_returns_transition_without_touching_the_grid():
    harness = Harness(script=[transition("right")], pose=(5, 5))
    result = harness.controller.follow_path(straight_plan((5, 5), (6, 5)))
    assert result.status == "transition"
    assert result.event is not None
    assert result.event.kind.value == "transition_event"
    assert harness.grid.calls == []
    assert harness.localizer.pose.cell == (5, 5)


def test_follow_path_replans_once_after_block_and_succeeds():
    pose = (5, 5)
    first_plan = straight_plan(pose, (7, 5))
    detour = plan_from_cells(
        [(5, 5), (5, 6), (5, 7), (6, 7), (7, 7), (7, 6), (7, 5)]
    )
    harness = Harness(
        script=[
            blocked("right"),
            moved("down"),
            moved("down"),
            moved("right"),
            moved("right"),
            moved("up"),
            moved("up"),
        ],
        pose=pose,
        plans=[detour],
    )
    result = harness.controller.follow_path(first_plan)
    assert result.status == "ok", result.reason
    assert result.steps == 7
    assert result.moves == 6
    assert result.blocked == 1
    assert result.replans == 1
    assert harness.pathfinder.calls == [(pose, (7, 5))]
    assert harness.localizer.pose.cell == (7, 5)


def test_follow_path_stops_when_no_alternative_path_exists():
    pose = (5, 5)
    harness = Harness(
        script=[blocked("right")],
        pose=pose,
        plans=[PathPlan(reachable=False, reason="wall")],
    )
    result = harness.controller.follow_path(straight_plan(pose, (7, 5)))
    assert result.status == "blocked"
    assert "no alternative" in result.reason
    assert len(harness.pathfinder.calls) == 1
    # A second block on the same edge must not replan again (once per edge).
    assert harness.controller.status()["replans"] == 0


def test_follow_path_localization_lost_after_consecutive_unknown():
    harness = Harness(
        script=[unknown("right"), unknown("right")],
        pose=(5, 5),
        config=ControllerConfig(max_consecutive_unknown=2),
    )
    harness.world.match_results = []  # relocalization impossible
    result = harness.controller.follow_path(straight_plan((5, 5), (7, 5)))
    assert result.status == "localization_lost"
    assert result.steps == 2
    assert result.unknown == 2
    assert harness.grid.calls == []
    assert harness.controller.status()["localization_lost"] is True


def test_follow_path_relocalizes_and_continues_after_unknown_streak():
    harness = Harness(
        script=[
            unknown("right"),
            unknown("right"),
            moved("right"),
            moved("right"),
        ],
        pose=(5, 5),
        config=ControllerConfig(max_consecutive_unknown=2),
        plans=[straight_plan((5, 5), (7, 5))],
    )
    harness.world.match_results = [_match("area_0", (5, 5))]
    result = harness.controller.follow_path(straight_plan((5, 5), (7, 5)))
    assert result.status == "ok", result.reason
    assert result.unknown == 2
    assert result.replans == 1
    assert harness.localizer.pose.cell == (7, 5)
    assert len(harness.move_calls()) == 2


def test_follow_path_respects_step_budget():
    harness = Harness(script=[moved("right")], pose=(5, 5))
    result = harness.controller.follow_path(straight_plan((5, 5), (8, 5)), max_steps=1)
    assert result.status == "step_limit"
    assert result.steps == 1
    assert harness.localizer.pose.cell == (6, 5)


def test_follow_path_realigns_a_stale_plan():
    harness = Harness(script=[moved("right")], pose=(5, 5))
    stale = straight_plan((3, 3), (6, 5))  # starts elsewhere
    result = harness.controller.follow_path(stale)
    assert result.status == "ok", result.reason
    assert result.replans == 1
    assert harness.localizer.pose.cell == (6, 5)


# --------------------------------------------------------------------------- #
# Relocalization acceptance
# --------------------------------------------------------------------------- #


def test_accept_match_requires_score_and_margin():
    harness = Harness()
    strong = _match("area_0", (5, 5), score=0.9, margin=0.25)
    assert harness.controller.accept_match([strong]) is strong
    ambiguous = _match("area_0", (5, 5), score=0.9, margin=0.01, second_score=0.89)
    assert harness.controller.accept_match([ambiguous]) is None
    weak = _match("area_0", (5, 5), score=0.3, margin=0.25)
    assert harness.controller.accept_match([weak]) is None


def test_accept_match_expected_area_fallback_is_bounded():
    harness = Harness()
    same_area = _match("area_0", (5, 5), score=0.8, margin=0.05, second_score=0.75)
    assert (
        harness.controller.accept_match([same_area], expected_area="area_0") is same_area
    )
    other_area = _match("area_1", (5, 5), score=0.8, margin=0.05, second_score=0.75)
    assert harness.controller.accept_match([other_area], expected_area="area_0") is None
    zero_margin = _match("area_0", (5, 5), score=0.8, margin=0.0, second_score=0.8)
    assert harness.controller.accept_match([zero_margin], expected_area="area_0") is None


def test_relocate_restores_pose_without_pressing_buttons():
    harness = Harness(pose=(5, 5))
    harness.world.match_results = [_match("area_0", (3, 2))]
    match = harness.controller.relocate()
    assert match is not None
    assert harness.controller.current_pose().cell == (3, 2)
    assert harness.world.current_pose == (3, 2)
    assert harness.emulator.presses == []
    assert len(harness.grid.observe_calls) == 1


# --------------------------------------------------------------------------- #
# Transition handling
# --------------------------------------------------------------------------- #


def test_handle_transition_matched_switches_area_restores_pose_records_connection():
    harness = Harness(pose=(5, 5))
    harness.world.create_area(area_id="area_1")
    harness.world.match_results = [_match("area_1", (3, 4))]
    result = harness.controller.handle_transition(max_wait_frames=48)
    assert result["status"] == "matched"
    assert result["handled"] is True
    assert result["switched"] is True
    assert result["area_id"] == "area_1"
    assert result["pose"] == [3, 4]
    assert harness.world.current_area == "area_1"
    assert harness.localizer.pose.cell == (3, 4)
    assert harness.localizer.area_id == "area_1"
    assert len(harness.world.connections) == 1
    connection = harness.world.connections[0]
    assert connection.from_area == "area_0"
    assert connection.to_area == "area_1"
    assert connection.to_pose == (3, 4)
    # the transition source cell is marked as an exit on the *old* area grid
    assert ("mark_exit", (5, 5)) in harness.world.area("area_0").grid.calls
    # new area got an observation and an entry cell
    assert len(harness.world.area("area_1").grid.observe_calls) == 1
    assert (3, 4) in harness.world.area("area_1").entry_cells
    # bounded wait: three 8-frame stable captures + one 8-frame confirmation
    assert result["frames"] == 32
    assert result["stable"] is True


def test_handle_transition_marks_the_pressed_onto_door_cell_as_exit():
    """The transition usually fires when the pressed step lands on a warp tile."""
    harness = Harness(pose=(5, 5), script=[transition("down")])
    harness.controller.observe()
    estimate = harness.controller.move_once("down")
    assert estimate.result is MotionResult.TRANSITION
    harness.world.create_area(area_id="area_1")
    harness.world.match_results = [_match("area_1", (0, 0))]
    result = harness.controller.handle_transition(max_wait_frames=48)
    assert result["status"] == "matched"
    assert result["source_cell"] == [5, 6]
    assert ("mark_exit", (5, 6)) in harness.world.area("area_0").grid.calls
    connection = harness.world.connections[0]
    assert connection.from_cell == (5, 6)
    assert connection.to_area == "area_1"


def test_handle_transition_weak_evidence_creates_new_area_never_merges():
    harness = Harness(pose=(5, 5))
    ambiguous = _match("area_0", (0, 0), score=0.9, margin=0.01, second_score=0.89)
    harness.world.match_results = [ambiguous]
    old_cells_before = len(harness.grid.cells)
    result = harness.controller.handle_transition(max_wait_frames=48)
    assert result["status"] == "new_area"
    assert result["new_area"] is True
    assert result["matched"] is False
    assert len(harness.world.areas) == 2
    new_area_id = result["area_id"]
    assert new_area_id and new_area_id != "area_0"
    assert result["pose"] == [0, 0]
    assert harness.world.current_area == new_area_id
    assert harness.localizer.pose.cell == (0, 0)
    assert len(harness.world.area("area_0").grid.cells) == old_cells_before
    assert len(harness.world.connections) == 1
    connection = harness.world.connections[0]
    assert connection.to_area == new_area_id
    assert connection.to_pose is None  # pending connection
    assert connection.confidence < 0.5


def test_handle_transition_black_timeout_does_not_guess_an_area():
    harness = Harness(pose=(5, 5))
    harness.localizer.is_black = True
    result = harness.controller.handle_transition(max_wait_frames=16)
    assert result["status"] == "black_timeout"
    assert result["handled"] is False
    assert result["black"] is True
    assert result["pose"] is None
    assert len(harness.world.areas) == 1
    assert harness.world.connections == []
    assert harness.localizer.pose.valid is False


def test_handle_transition_never_creates_an_area_from_a_changing_fade():
    """A mid-warp fade changes the screen every capture: no area may be created."""
    harness = Harness(pose=(5, 5))
    harness.localizer.signature_sequence = [
        "fade1",
        "fade2",
        "fade3",
        "fade4",
        "fade5",
        "fade6",
    ]
    areas_before = len(harness.world.areas)
    result = harness.controller.handle_transition(max_wait_frames=48)
    assert result["status"] == "unstable_timeout"
    assert result["handled"] is False
    assert result["stable"] is False
    assert result["black"] is False
    assert len(harness.world.areas) == areas_before
    assert harness.world.connections == []
    assert harness.controller.status()["localization_lost"] is True


def _animated_luma(phase: int) -> np.ndarray:
    """Settled outdoor frame with a 1-px-per-capture walking NPC (animated tiles)."""
    luma = np.zeros((144, 160), dtype=np.uint8)
    luma[24:40, 24:136] = 170
    luma[64:80, 40:120] = 220
    x = 24 + (int(phase) % 24)  # the NPC moves 1 px per capture: <1 % changed pixels
    luma[80:96, x : x + 16] = 255
    return luma


def _fade_luma(phase: int) -> np.ndarray:
    """A warp fade: most pixels change between captures (>10 %)."""
    level = max(0, 220 - 40 * int(phase))
    luma = np.full((144, 160), level, dtype=np.uint8)
    luma[::4, ::4] = 255 - level
    return luma


def test_handle_transition_settles_with_animated_tiles_and_npcs():
    """Frozen is decided by pixel diff: animated maps still count as settled."""
    harness = Harness(pose=(5, 5))
    harness.world.create_area(area_id="area_1")
    harness.world.match_results = [_match("area_1", (3, 4))]
    harness.localizer.luma_sequence = [_animated_luma(index) for index in range(8)]
    result = harness.controller.handle_transition(max_wait_frames=48)
    assert result["status"] == "matched", result
    assert result["handled"] is True
    assert result["area_id"] == "area_1"
    assert result["frames"] == 32  # 3 captures (24) + 8-frame confirmation
    assert harness.controller.TRANSITION_MAX_DIFF == pytest.approx(0.02)


def test_handle_transition_fade_does_not_count_as_settled():
    harness = Harness(pose=(5, 5))
    harness.localizer.luma_sequence = [_fade_luma(index) for index in range(8)]
    areas_before = len(harness.world.areas)
    result = harness.controller.handle_transition(max_wait_frames=48)
    assert result["status"] == "unstable_timeout"
    assert result["handled"] is False
    assert len(harness.world.areas) == areas_before
    assert harness.world.connections == []


def test_handle_transition_confirmation_frame_must_still_match():
    """Three frozen captures are not enough: the +8 frame confirmation must match."""
    harness = Harness(pose=(5, 5))
    harness.localizer.signature_sequence = [
        "same",
        "same",
        "same",
        "different",
    ]
    result = harness.controller.handle_transition(max_wait_frames=48)
    assert result["status"] == "unstable_timeout"
    assert result["handled"] is False
    assert len(harness.world.areas) == 1
    assert harness.world.connections == []


# --------------------------------------------------------------------------- #
# explore_current_area
# --------------------------------------------------------------------------- #


def _probe_target(
    cell: tuple[int, int], from_cell: tuple[int, int], direction: str, *, path: PathPlan | None = None
) -> ExploreTarget:
    return ExploreTarget(
        cell=cell,
        from_cell=from_cell,
        direction=direction,
        path=path,
        kind="frontier",
        expected_new_cells=4,
    )


def test_explore_completes_when_policy_says_finished():
    harness = Harness(finish_sequence=[(True, "no frontier left", {})])
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.COMPLETED
    assert "no frontier" in report.reason
    assert report.steps == 0
    assert harness.emulator.presses == []


def test_explore_probes_frontier_and_reports_counters():
    target = _probe_target(
        (6, 5), (5, 5), "right", path=PathPlan(cells=[(5, 5)], directions=[], reachable=True)
    )
    harness = Harness(script=[moved("right")], pose=(5, 5), targets=[target])
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.COMPLETED  # no target left on the 2nd cycle
    assert report.steps == 1
    assert report.moves == 1
    assert report.new_cells >= 1
    assert harness.explorer.attempts == [((6, 5), "moved")]
    assert len(harness.move_calls()) == 1
    assert report.pose is not None and report.pose.cell == (6, 5)
    assert report.coverage is not None


def test_explore_reports_no_progress_and_stays_bounded():
    harness = Harness(
        script=[blocked("right"), blocked("right"), blocked("right")],
        pose=(5, 5),
        targets=[_probe_target((6, 5), (5, 5), "right") for _ in range(6)],
    )
    report = harness.controller.explore_current_area(max_steps=20, max_no_progress=2)
    assert report.status is ExploreStatus.NO_PROGRESS
    assert report.steps <= 20
    attempts = [outcome for _, outcome in harness.explorer.attempts]
    assert attempts and set(attempts) == {"blocked"}


def test_explore_step_limit():
    harness = Harness(pose=(5, 5))
    report = harness.controller.explore_current_area(max_steps=0)
    assert report.status is ExploreStatus.STEP_LIMIT


def test_explore_time_limit():
    harness = Harness(pose=(5, 5))
    report = harness.controller.explore_current_area(max_seconds=-1.0)
    assert report.status is ExploreStatus.TIME_LIMIT


def test_explore_event_interrupts_and_pauses():
    target = _probe_target(
        (6, 5),
        (5, 5),
        "right",
        path=straight_plan((5, 5), (6, 5)),
    )
    harness = Harness(
        script=[interrupted("right", FrameContext.BATTLE)],
        pose=(5, 5),
        targets=[target],
    )
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.INTERRUPTED_EVENT
    assert report.event is not None and report.event.kind.value == "battle_event"
    assert harness.controller.paused is True
    # a paused controller must not start a new run
    paused_report = harness.controller.explore_current_area(max_steps=10)
    assert paused_report.status is ExploreStatus.PAUSED
    harness.controller.resume()
    assert harness.controller.paused is False


def test_explore_handles_transition_and_reports_area_switch():
    target = _probe_target(
        (6, 5),
        (5, 5),
        "right",
        path=straight_plan((5, 5), (6, 5)),
    )
    harness = Harness(script=[transition("right")], pose=(5, 5), targets=[target])
    harness.world.create_area(area_id="area_1")
    harness.world.match_results = [_match("area_1", (0, 0))]
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.TRANSITION
    assert report.area_switched is True
    assert report.new_area_id == "area_1"


def test_explore_transition_report_describes_the_explored_area():
    """The report/coverage stay about the area that was explored before the warp."""
    target = _probe_target(
        (6, 5),
        (5, 5),
        "right",
        path=straight_plan((5, 5), (6, 5)),
    )
    harness = Harness(script=[transition("right")], pose=(5, 5), targets=[target])
    harness.world.create_area(area_id="area_1")
    harness.world.match_results = [_match("area_1", (0, 0))]
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.TRANSITION
    assert report.area_id == "area_0"
    assert report.coverage is not None
    assert report.coverage.area_id == "area_0"
    assert report.pose is not None and report.pose.cell == (5, 5)


def test_explore_does_not_leave_the_room_through_an_exit():
    """Exit frontiers are skipped while floor frontiers remain (no door ping-pong)."""
    harness = Harness(script=[moved("down")], pose=(5, 5))
    harness.grid.get_cell(6, 5).state = CellState.EXIT
    exit_target = ExploreTarget(
        cell=(6, 5), from_cell=(5, 5), direction="right", kind="exit"
    )
    floor_candidate = ExploreTarget(
        cell=(5, 6), from_cell=(5, 5), direction="down", kind="frontier"
    )
    harness.explorer.targets = [exit_target]
    harness.explorer.candidates = [floor_candidate]
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.COMPLETED
    assert harness.emulator.presses == ["down"]  # floor probe, not the door
    assert harness.localizer.pose.cell == (5, 6)
    assert report.area_switched is False


def test_explore_completes_when_only_exit_frontiers_remain():
    """With only a door left the room counts as covered; leaving is explicit."""
    harness = Harness(pose=(5, 5))
    harness.grid.get_cell(6, 5).state = CellState.EXIT
    exit_target = ExploreTarget(
        cell=(6, 5), from_cell=(5, 5), direction="right", kind="exit"
    )
    harness.explorer.targets = [exit_target]
    harness.explorer.candidates = []
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.COMPLETED
    assert "only exit frontiers remain" in report.reason
    assert "navigate_to" in report.reason
    assert harness.emulator.presses == []
    assert report.coverage is not None and report.coverage.exits == 1
    assert report.area_switched is False


def test_explore_defers_frontiers_next_to_a_known_door():
    """Unknown tiles touching a known door are not probed while floor remains."""
    harness = Harness(script=[moved("down")], pose=(5, 5))
    harness.grid.get_cell(6, 5).state = CellState.EXIT
    door_adjacent = ExploreTarget(
        cell=(6, 6), from_cell=(5, 6), direction="down", kind="frontier"
    )
    floor_candidate = ExploreTarget(
        cell=(5, 6), from_cell=(5, 5), direction="down", kind="frontier"
    )
    harness.explorer.targets = [door_adjacent]
    harness.explorer.candidates = [door_adjacent, floor_candidate]
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.COMPLETED
    assert harness.emulator.presses == ["down"]
    assert harness.localizer.pose.cell == (5, 6)
    assert report.area_switched is False


def test_navigate_to_exit_records_the_transition_attempt():
    """Explicit exit navigation records the crossed door on the explorer."""
    harness = Harness(script=[transition("right")], pose=(5, 5))
    harness.grid.get_cell(6, 5).state = CellState.EXIT
    harness.world.create_area(area_id="area_1")
    harness.world.match_results = [_match("area_1", (0, 0))]
    report = harness.controller.navigate_to({"exit": True})
    assert report.status is ExploreStatus.TRANSITION
    assert report.area_switched is True
    assert report.reached is True  # crossing the exit means the exit was reached
    assert harness.explorer.attempts == [((6, 5), "transition")]


def test_explore_localization_lost():
    harness = Harness(
        script=[unknown("up"), unknown("up")],
        pose=(5, 5),
        config=ControllerConfig(max_consecutive_unknown=2),
        targets=[
            _probe_target((5, 4), (5, 5), "up"),
            _probe_target((5, 4), (5, 5), "up"),
        ],
    )
    harness.world.match_results = []
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.LOCALIZATION_LOST


# --------------------------------------------------------------------------- #
# navigate_to
# --------------------------------------------------------------------------- #


def test_navigate_to_cell_reaches_goal():
    harness = Harness(script=[moved("right"), moved("right")], pose=(5, 5))
    report = harness.controller.navigate_to({"cell": [7, 5]})
    assert report.status is ExploreStatus.COMPLETED
    assert report.reached is True
    assert report.pose is not None and report.pose.cell == (7, 5)
    assert report.moves == 2


def test_navigate_to_cell_already_there():
    harness = Harness(pose=(5, 5))
    report = harness.controller.navigate_to({"cell": [5, 5]})
    assert report.status is ExploreStatus.COMPLETED
    assert report.reached is True
    assert harness.emulator.presses == []


def test_navigate_to_unreachable_cell_is_blocked_with_reason():
    harness = Harness(
        pose=(5, 5), plans=[PathPlan(reachable=False, reason="walled off")]
    )
    report = harness.controller.navigate_to({"cell": [9, 9]})
    assert report.status is ExploreStatus.BLOCKED
    assert "not reachable" in report.reason
    assert report.reached is False
    assert harness.emulator.presses == []


def test_navigate_to_exit_cell():
    harness = Harness(script=[moved("right")], pose=(5, 5))
    harness.grid.get_cell(6, 5).state = CellState.EXIT
    report = harness.controller.navigate_to({"exit": True})
    assert report.status is ExploreStatus.COMPLETED
    assert report.reached is True
    assert report.pose.cell == (6, 5)


def test_navigate_to_exit_without_known_exit_is_blocked():
    harness = Harness(pose=(5, 5))
    report = harness.controller.navigate_to({"exit": True})
    assert report.status is ExploreStatus.BLOCKED
    assert "exit" in report.reason


def test_navigate_to_return_to_entry():
    harness = Harness(script=[moved("left"), moved("up")], pose=(5, 5))
    harness.world.area().entry_cells = [(4, 4)]
    report = harness.controller.navigate_to({"return_to_entry": True})
    assert report.status is ExploreStatus.COMPLETED
    assert report.reached is True
    assert report.pose.cell == (4, 4)


def test_navigate_to_landmark():
    harness = Harness(script=[moved("down")], pose=(5, 5))
    harness.grid.get_cell(5, 6).labels = ["sign:route1"]
    report = harness.controller.navigate_to({"landmark": "sign:route1"})
    assert report.status is ExploreStatus.COMPLETED
    assert report.reached is True
    assert report.pose.cell == (5, 6)


def test_navigate_to_missing_landmark_is_blocked():
    harness = Harness(pose=(5, 5))
    report = harness.controller.navigate_to({"landmark": "mt-moon"})
    assert report.status is ExploreStatus.BLOCKED
    assert "not on the current map" in report.reason


def test_navigate_to_invalid_target_is_error():
    harness = Harness(pose=(5, 5))
    report = harness.controller.navigate_to({"teleport": True})
    assert report.status is ExploreStatus.ERROR
    assert "invalid navigation target" in report.reason
    assert harness.emulator.presses == []


def test_navigate_to_event_interrupts_navigation():
    harness = Harness(script=[interrupted("right", FrameContext.DIALOG)], pose=(5, 5))
    report = harness.controller.navigate_to({"cell": [7, 5]})
    assert report.status is ExploreStatus.INTERRUPTED_EVENT
    assert report.event is not None and report.event.kind.value == "dialog_event"
    assert harness.controller.paused is True


def test_navigate_to_other_area_via_recorded_connection():
    harness = Harness(pose=(5, 5))
    harness.world.create_area(area_id="area_1")
    harness.grid.get_cell(5, 6)  # the connection cell is on the mapped grid
    harness.world.connections.append(
        SimpleNamespace(
            from_area="area_0",
            from_cell=(5, 6),
            direction="down",
            to_area="area_1",
            to_pose=None,
            kind="warp",
            confidence=0.8,
        )
    )
    harness.world.match_results = [_match("area_1", (0, 0))]
    harness.pathfinder.plans.extend(
        [straight_plan((5, 5), (5, 6)), straight_plan((0, 0), (0, 1))]
    )
    harness.localizer.script.extend(
        [moved("down"), transition("down"), moved("down")]
    )
    report = harness.controller.navigate_to({"area": "area_1", "cell": [0, 1]})
    assert report.status is ExploreStatus.COMPLETED, report.reason
    assert report.reached is True
    assert report.area_switched is True
    assert report.new_area_id == "area_1"
    assert harness.world.current_area == "area_1"
    assert report.pose.cell == (0, 1)
    assert len(harness.world.connections) == 2  # pending + recorded after transition


def test_navigate_to_other_area_without_known_route_is_blocked():
    harness = Harness(pose=(5, 5))
    harness.world.create_area(area_id="area_1")
    report = harness.controller.navigate_to({"area": "area_1", "cell": [0, 0]})
    assert report.status is ExploreStatus.BLOCKED
    assert "no known route" in report.reason


# --------------------------------------------------------------------------- #
# Status / pause
# --------------------------------------------------------------------------- #


def test_status_exposes_pose_counters_and_world_state():
    harness = Harness(script=[moved("right")], pose=(5, 5))
    harness.controller.move_once("right")
    status = harness.controller.status()
    for key in (
        "area_id",
        "area_kind",
        "pose",
        "grid",
        "steps",
        "moves",
        "blocked",
        "unknown",
        "replans",
        "localization_lost",
        "paused",
        "last_event",
    ):
        assert key in status
    assert status["area_id"] == "area_0"
    assert status["pose"]["x"] == 6
    assert status["moves"] == 1
    assert status["paused"] is False


def test_pause_and_resume_gate_exploration():
    harness = Harness(pose=(5, 5))
    harness.controller.pause()
    report = harness.controller.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.PAUSED
    harness.controller.resume()
    report = harness.controller.explore_current_area(max_steps=0)
    assert report.status is ExploreStatus.STEP_LIMIT


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


# --------------------------------------------------------------------------- #
# Real-integration smoke test (skips until nav-map modules land)
# --------------------------------------------------------------------------- #


def test_integration_follow_path_completes_at_budget_with_real_planner(
    open_room, tmp_path
):
    """Nav-planner regression: a real A* path that fits the budget ends ``ok``."""
    pytest.importorskip("llm_agent.navigation.localization")
    pytest.importorskip("llm_agent.navigation.grid_map")
    pytest.importorskip("llm_agent.navigation.world_model")
    pytest.importorskip("llm_agent.navigation.pathfinder")
    from llm_agent.navigation.localization import Localizer
    from llm_agent.navigation.pathfinder import Pathfinder
    from llm_agent.navigation.world_model import WorldModel

    from tests.navigation.conftest import SyntheticEmulator

    def build():
        emulator = SyntheticEmulator(open_room, pose=(5, 4), facing="down")
        world = WorldModel(root=tmp_path / "nav_budget")
        area = world.create_area()
        world.set_current(area.area_id, (0, 0))
        localizer = Localizer()
        # area-local origin is arbitrary: define it at the emulator's current spot
        localizer.reset(area_id=area.area_id, cell=(0, 0), facing="down")
        controller = NavigationController(
            emulator=emulator,
            world=world,
            localizer=localizer,
            pathfinder=Pathfinder(),
            explorer=FakeExplorer(),
        )
        controller.observe()
        return controller, localizer, world

    controller, localizer, world = build()
    plan = controller.pathfinder.plan(world.area().grid, (0, 0), (0, -2))
    assert plan.reachable and len(plan.directions) == 2, plan.reason
    result = controller.follow_path(plan, max_steps=2)
    assert result.status == "ok", (result.status, result.reason)
    assert result.steps == 2
    assert localizer.pose.cell == (0, -2)

    controller2, localizer2, world2 = build()
    plan2 = controller2.pathfinder.plan(world2.area().grid, (0, 0), (0, -2))
    result2 = controller2.follow_path(plan2, max_steps=1)
    assert result2.status == "step_limit"
    assert localizer2.pose.cell == (0, -1)


def test_integration_real_localizer_and_grid_map(open_room, tmp_path):
    """SyntheticEmulator + real Localizer + real GridMap/WorldModel end-to-end."""
    pytest.importorskip("llm_agent.navigation.localization")
    grid_map_module = pytest.importorskip("llm_agent.navigation.grid_map")
    world_model_module = pytest.importorskip("llm_agent.navigation.world_model")
    from llm_agent.navigation.localization import Localizer

    from tests.navigation.conftest import SyntheticEmulator

    assert grid_map_module and world_model_module
    emulator = SyntheticEmulator(open_room, pose=(5, 4), facing="down")
    world = world_model_module.WorldModel(root=tmp_path / "nav")
    area = world.create_area()
    world.set_current(area.area_id, (5, 4))
    localizer = Localizer()
    localizer.reset(area_id=area.area_id, cell=(5, 4), facing="down")
    controller = NavigationController(
        emulator=emulator,
        world=world,
        localizer=localizer,
        pathfinder=FakePathfinder(),
        explorer=FakeExplorer(),
        config=ControllerConfig(),
    )
    controller.observe()

    estimate = controller.move_once("right")
    assert estimate.result is MotionResult.MOVED, estimate.evidence
    assert localizer.pose.cell == (6, 4)
    assert emulator.presses == ["right"]
    grid = world.area().grid
    walked = grid.cell(6, 4)
    assert walked is not None and walked.state is CellState.WALKABLE
    assert walked.visit_count == 1

    # Walk up to the out-of-bounds border, then bump it: the map must not gain
    # a fake walkable tile and the coordinate must not change.
    for _ in range(4):
        step = controller.move_once("up")
        assert step.result is MotionResult.MOVED, step.evidence
    assert localizer.pose.cell == (6, 0)
    bump = controller.move_once("up")
    assert bump.result is MotionResult.BLOCKED, bump.evidence
    assert localizer.pose.cell == (6, 0)

    walkable = [
        cell for cell in grid.cells.values() if cell.state is CellState.WALKABLE
    ]
    # The starting cell is promoted by confirm_move (it was verifiably occupied),
    # so the map has 6 walkable cells but only the 5 verified targets were *entered*.
    assert len(walkable) == 6
    entered = [cell for cell in walkable if cell.visit_count > 0]
    assert len(entered) == 5
    assert sorted((cell.x, cell.y) for cell in entered) == [
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (6, 4),
    ]
    assert all(cell.visit_count == 1 for cell in entered)  # exactly-once evidence
    blocked = grid.cell(6, -1)
    assert blocked is not None
    assert blocked.state is CellState.BLOCKED
    assert blocked.visit_count == 0
    # Exactly-once evidence: one confirmation per press, no duplicates.
    assert [press for press in emulator.presses] == ["right"] + ["up"] * 5
