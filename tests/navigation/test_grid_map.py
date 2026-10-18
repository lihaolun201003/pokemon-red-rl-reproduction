"""Unit tests for the per-area logical grid map (fast, no ROM / no PyBoy).

Covers the semantic rules that matter most:

* observation (OBSERVED) is separated from verification (WALKABLE);
* one bump never permanently walls off a cell that was verified walkable;
* the player's own cell is promoted to WALKABLE by a valid pose;
* frontier / reachable region / coverage math is honest;
* persistence is deterministic JSON, atomic, versioned and tolerant of damage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_agent.navigation import grid_map as grid_map_module
from llm_agent.navigation import vision
from llm_agent.navigation.grid_map import (
    GRID_MAP_FORMAT,
    GRID_MAP_FORMAT_VERSION,
    MAX_TILE_VARIANTS,
    Cell,
    Edge,
    Frontier,
    GridMap,
)
from llm_agent.navigation.types import (
    CellState,
    EdgeState,
    FrameView,
    PlayerPose,
)

from tests.navigation.conftest import SyntheticRoom, luma_from_rgba


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pose(x: int, y: int, *, valid: bool = True, confidence: float = 1.0) -> PlayerPose:
    return PlayerPose(x=int(x), y=int(y), facing="down", confidence=float(confidence), valid=valid)


def _frame_view(room: SyntheticRoom, pose: tuple[int, int], *, facing: str = "down") -> FrameView:
    """Synthetic FrameView extracted exactly like the real pipeline (0,0 lattice)."""
    luma = luma_from_rgba(room.render(pose, facing))
    tiles = vision.extract_tiles(luma, (0, 0))
    return FrameView(
        frame_index=0,
        signature="synthetic",
        grid_offset=(0, 0),
        tiles=tiles,
    )


def _open_room(seed: int = 1, width: int = 10, height: int = 9) -> SyntheticRoom:
    return SyntheticRoom(width=width, height=height, seed=seed)


# --------------------------------------------------------------------------- #
# Dataclass records
# --------------------------------------------------------------------------- #


def test_cell_dict_roundtrip_and_defaults() -> None:
    cell = Cell(
        x=1,
        y=2,
        state=CellState.WALKABLE,
        terrain="aaaa",
        tile_hashes=["aaaa", "bbbb"],
        observed_count=3,
        visit_count=2,
        confidence=0.987,
        last_seen_frame=7,
        obstacle_evidence=1,
        free_evidence=4,
        labels=["door", "wall"],
    )
    clone = Cell.from_dict(cell.to_dict())
    assert clone == cell
    assert Cell.from_dict({"x": 0, "y": 0}) == Cell(x=0, y=0)
    # Tolerant reader: unknown state string becomes UNKNOWN, extra keys ignored.
    tolerant = Cell.from_dict({"x": 4, "y": 5, "state": "definitely-not-a-state", "jsunk": 1})
    assert tolerant.state is CellState.UNKNOWN
    with pytest.raises(ValueError):
        Cell.from_dict({"x": 0})
    with pytest.raises(ValueError):
        Cell.from_dict("not-a-mapping")  # type: ignore[arg-type]


def test_cell_hash_variants_are_bounded_and_terrain_is_dominant() -> None:
    cell = Cell(x=0, y=0)
    for index in range(MAX_TILE_VARIANTS + 3):
        assert cell.add_hash(f"h{index:02d}") is True
    assert len(cell.tile_hashes) == MAX_TILE_VARIANTS
    # Oldest variants dropped first; all remaining counts are 1 -> earliest wins.
    assert cell.tile_hashes[0] == "h03"
    assert cell.terrain == "h03"
    assert cell.add_hash("h03") is False
    assert cell.add_hash("") is False
    # A repeated look becomes dominant when counts differ.
    cell.add_hash("h03")
    cell.add_hash("h03")
    assert cell.terrain == "h03"


def test_edge_dict_roundtrip_and_canonicalization() -> None:
    edge = Edge(
        a=(0, 1),
        b=(1, 1),
        direction="right",
        state=EdgeState.OPEN,
        attempts=3,
        successes=2,
        confidence=0.75,
    )
    assert Edge.from_dict(edge.to_dict()) == edge
    reversed_edge = Edge.from_dict(
        {"a": [1, 1], "b": [0, 1], "direction": "left", "state": "open"}
    )
    assert reversed_edge.a == (0, 1)
    assert reversed_edge.b == (1, 1)
    assert reversed_edge.direction == "right"
    with pytest.raises(ValueError):
        Edge.from_dict({"a": [0, 0], "b": "nope"})


def test_frontier_normalization() -> None:
    frontier = Frontier(cell=[1, 2], from_cell=(0, 2), direction="right", expected_new=7)  # type: ignore[arg-type]
    assert frontier.cell == (1, 2)
    assert frontier.expected_new == 7
    assert frontier.to_dict()["cell"] == [1, 2]


# --------------------------------------------------------------------------- #
# Basic grid access / edges / bounds
# --------------------------------------------------------------------------- #


def test_grid_basic_access_edges_and_neighbours() -> None:
    grid = GridMap("room", frame_index=3)
    assert grid.area_id == "room"
    assert grid.created_frame == 3
    assert grid.kind == "unknown"
    assert grid.cell(0, 0) is None
    assert grid.get_cell(0, 0).state is CellState.UNKNOWN
    assert grid.size() == 0  # not "known" yet

    assert grid.neighbours((5, 5)) == [(5, 4), (5, 6), (4, 5), (6, 5)]

    assert grid.edge((0, 0), (1, 0)) is None
    edge = grid.get_edge((1, 0), (0, 0))
    assert edge is grid.get_edge((0, 0), (1, 0))
    assert (edge.a, edge.b, edge.direction) == ((0, 0), (1, 0), "right")
    assert grid.edge_state((0, 0), (1, 0)) is EdgeState.UNKNOWN
    assert grid.edge_state((7, 7), (8, 7)) is EdgeState.UNKNOWN
    # Canonical order for a vertical edge above the origin.
    assert grid.get_edge((0, 0), (0, -1)).direction == "down"
    with pytest.raises(ValueError):
        grid.get_edge((0, 0), (0, 0))


def test_cells_in_rect_and_bounds() -> None:
    grid = GridMap("r")
    for x, y in [(0, 0), (1, 0), (2, 0), (0, 1), (2, 2)]:
        grid.get_cell(x, y)
    expected = [(0, 0), (1, 0), (2, 0), (0, 1)]
    assert [cell.cell for cell in grid.cells_in_rect(0, 0, 2, 1)] == expected
    # Swapped corners are tolerated and treated as an inclusive rectangle.
    assert [cell.cell for cell in grid.cells_in_rect(2, 1, 0, 0)] == expected
    assert grid.bounds() == (0, 0, 2, 2)
    empty = GridMap("empty")
    assert empty.bounds() == (0, 0, 0, 0)
    assert empty.cells_in_rect(0, 0, 3, 3) == []


# --------------------------------------------------------------------------- #
# Observation vs verification
# --------------------------------------------------------------------------- #


def test_observe_view_marks_observed_never_walkable(monkeypatch: pytest.MonkeyPatch) -> None:
    room = _open_room(seed=3)
    view = _frame_view(room, (4, 4))
    # Disable the documented pose-cell promotion to prove observation alone never
    # creates WALKABLE.
    monkeypatch.setattr(grid_map_module, "PROMOTE_POSE_CELL", False)
    grid = GridMap("a")
    counters = grid.observe_view(view, _pose(4, 4), frame_index=10)

    assert counters["new_cells"] == 90
    assert counters["observed"] == 90
    assert counters["new_hashes"] == 90
    assert grid.count_state(CellState.OBSERVED) == 90
    assert grid.count_state(CellState.WALKABLE) == 0
    assert grid.size() == 90
    # A view is stored at the world position implied by the pose:
    # world = pose + (tx - VIEW_PLAYER[0], ty - VIEW_PLAYER[1]).
    assert grid.get_cell(0, 0).tile_hashes == [view.tile(0, 0).tile_hash]  # type: ignore[union-attr]
    assert grid.get_cell(0, 0).terrain == view.tile(0, 0).tile_hash  # type: ignore[union-attr]
    assert grid.get_cell(1, 2).tile_hashes == [view.tile(1, 2).tile_hash]  # type: ignore[union-attr]
    # The stored look matches the synthetic room tile exactly (4 DMG grey levels).
    grey_levels = np.array([0, 85, 153, 255], dtype=np.uint8)
    expected_luma = grey_levels[room.tile_luma(1, 2)]
    assert view.tile(1, 2).tile_hash == vision.tile_hash(expected_luma)  # type: ignore[union-attr]


def test_observe_view_promotes_player_cell() -> None:
    room = _open_room(seed=4)
    view = _frame_view(room, (4, 4))
    grid = GridMap("a")
    counters = grid.observe_view(view, _pose(4, 4), frame_index=1)
    assert counters["new_cells"] == 90
    player = grid.get_cell(4, 4)
    assert player.state is CellState.WALKABLE
    assert player.free_evidence >= 1
    assert grid.count_state(CellState.WALKABLE) == 1
    assert grid.count_state(CellState.OBSERVED) == 89

    # Re-observing must not downgrade and must not create spurious new cells.
    again = grid.observe_view(view, _pose(4, 4), frame_index=2)
    assert again == {"new_cells": 0, "new_hashes": 0, "observed": 90}
    assert grid.get_cell(4, 4).state is CellState.WALKABLE
    assert grid.get_cell(4, 4).observed_count == 2


def test_observe_view_invalid_pose_does_not_promote() -> None:
    room = _open_room(seed=5)
    grid = GridMap("a")
    grid.observe_view(_frame_view(room, (4, 4)), _pose(4, 4, valid=False), frame_index=0)
    assert grid.get_cell(4, 4).state is CellState.OBSERVED
    assert grid.count_state(CellState.WALKABLE) == 0


def test_observe_view_empty_view_is_a_noop() -> None:
    grid = GridMap("a")
    empty = FrameView(frame_index=0, signature="s", grid_offset=(0, 0), tiles=[])
    assert grid.observe_view(empty, _pose(0, 0)) == {
        "new_cells": 0,
        "new_hashes": 0,
        "observed": 0,
    }
    assert grid.cells == {}


def test_observe_view_never_downgrades_verified_states() -> None:
    room = _open_room(seed=6)
    view = _frame_view(room, (4, 4))
    grid = GridMap("a")
    grid.observe_view(view, _pose(4, 4), frame_index=0)
    grid.confirm_blocked((4, 4), "down", frame_index=1)  # (4, 5) -> BLOCKED
    grid.mark_exit((0, 0))
    grid.mark_interactive((1, 1), "npc")
    walked = grid.confirm_move((4, 4), "up", frame_index=2)  # (4, 3) -> WALKABLE
    assert walked == (4, 3)

    grid.observe_view(view, _pose(4, 4), frame_index=3)

    assert grid.get_cell(4, 5).state is CellState.BLOCKED
    assert grid.get_cell(0, 0).state is CellState.EXIT
    assert grid.get_cell(1, 1).state is CellState.INTERACTIVE
    assert grid.get_cell(4, 3).state is CellState.WALKABLE


# --------------------------------------------------------------------------- #
# Verified movement / blocking
# --------------------------------------------------------------------------- #


def test_confirm_move_marks_walkable_and_open_edges() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    target = grid.confirm_move((0, 0), "down", frame_index=4)
    assert target == (0, 1)
    cell = grid.get_cell(0, 1)
    assert cell.state is CellState.WALKABLE
    assert cell.visit_count == 1
    assert cell.free_evidence == 1
    edge = grid.edge((0, 0), (0, 1))
    assert edge is not None
    assert edge.state is EdgeState.OPEN
    assert (edge.attempts, edge.successes) == (1, 1)
    assert edge.confidence == pytest.approx(1.0)

    # Multi-tile moves verify every intermediate cell.
    assert grid.confirm_move((0, 0), "right", tiles=3, frame_index=5) == (3, 0)
    for x in (1, 2, 3):
        assert grid.get_cell(x, 0).state is CellState.WALKABLE
        assert grid.get_cell(x, 0).visit_count == 1
    assert grid.edge_state((2, 0), (3, 0)) is EdgeState.OPEN

    with pytest.raises(ValueError):
        grid.confirm_move((0, 0), "north")


def test_confirm_move_promotes_previously_blocked_cell() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    grid.confirm_blocked((0, 0), "right")
    assert grid.get_cell(1, 0).state is CellState.BLOCKED
    # An NPC moves away and we verifiably walk there: movement evidence wins.
    assert grid.confirm_move((0, 0), "right") == (1, 0)
    assert grid.get_cell(1, 0).state is CellState.WALKABLE
    assert grid.get_cell(1, 0).visit_count == 1
    assert grid.edge_state((0, 0), (1, 0)) is EdgeState.OPEN


def test_confirm_move_without_prior_observation_promotes_source() -> None:
    grid = GridMap("a")
    assert grid.confirm_move((2, 2), "down") == (2, 3)
    assert grid.get_cell(2, 2).state is CellState.WALKABLE
    assert grid.get_cell(2, 3).state is CellState.WALKABLE


def test_confirm_blocked_permanent_and_temporary() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    assert grid.confirm_blocked((0, 0), "down", frame_index=1) == (0, 1)
    blocked = grid.get_cell(0, 1)
    assert blocked.state is CellState.BLOCKED
    assert blocked.obstacle_evidence == 1
    assert blocked.visit_count == 0
    edge = grid.edge((0, 0), (0, 1))
    assert edge is not None
    assert edge.state is EdgeState.BLOCKED
    assert (edge.attempts, edge.successes) == (1, 0)

    grid.confirm_blocked((0, 0), "right", temporary=True)
    temp = grid.get_cell(1, 0)
    assert temp.state is CellState.TEMP_BLOCKED
    assert grid.edge_state((0, 0), (1, 0)) is EdgeState.TEMP_BLOCKED

    # A second believed-persistent bump escalates TEMP_BLOCKED -> BLOCKED.
    grid.confirm_blocked((0, 0), "right")
    assert grid.get_cell(1, 0).state is CellState.BLOCKED

    with pytest.raises(ValueError):
        grid.confirm_blocked((0, 0), "north")


def test_confirm_blocked_never_walls_off_walked_cell() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    assert grid.confirm_move((0, 0), "down") == (0, 1)
    for _ in range(5):
        target = grid.confirm_blocked((0, 0), "down", frame_index=9)
        assert target == (0, 1)
        cell = grid.get_cell(0, 1)
        assert cell.state is CellState.WALKABLE  # never permanently walled off
        assert cell.obstacle_evidence >= 1
        assert grid.edge_state((0, 0), (0, 1)) is EdgeState.TEMP_BLOCKED
    # ... and a temporary bump cannot downgrade it either.
    grid.confirm_blocked((0, 0), "down", temporary=True)
    assert grid.get_cell(0, 1).state is CellState.WALKABLE
    # Walking again re-opens the edge.
    grid.confirm_move((0, 0), "down")
    assert grid.edge_state((0, 0), (0, 1)) is EdgeState.OPEN
    assert grid.get_cell(0, 1).state is CellState.WALKABLE


def test_confirm_turn_is_not_obstacle_evidence() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.OBSERVED)
    grid.confirm_turn((0, 0), "up")
    assert grid.get_cell(0, 0).state is CellState.WALKABLE
    # Gen-1: the first press of a direction only turns; no blocked evidence yet.
    assert grid.cell(0, -1) is None
    assert grid.get_cell(0, -1).obstacle_evidence == 0
    with pytest.raises(ValueError):
        grid.confirm_turn((0, 0), "north")


def test_turn_then_blocked_is_turn_artifact_not_a_wall() -> None:
    """The controller marks TURNED_ONLY as blocked evidence; a turn is not a bump."""
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    grid.mark_state((1, 0), CellState.OBSERVED)  # seen but unverified frontier cell
    grid.confirm_turn((0, 0), "right")
    assert grid.confirm_blocked((0, 0), "right", frame_index=1) == (1, 0)

    cell = grid.get_cell(1, 0)
    assert cell.state is CellState.OBSERVED  # never walled off by a turn
    assert cell.obstacle_evidence == 1  # ... but the evidence was recorded
    assert grid.edge_state((0, 0), (1, 0)) is EdgeState.UNKNOWN
    # The follow-up genuine bump (now facing right) is real evidence.
    grid.confirm_blocked((0, 0), "right", frame_index=2)
    assert grid.get_cell(1, 0).state is CellState.BLOCKED
    assert grid.edge_state((0, 0), (1, 0)) is EdgeState.BLOCKED


def test_turn_artifact_does_not_downgrade_open_edge() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    grid.confirm_move((0, 0), "right")
    assert grid.edge_state((0, 0), (1, 0)) is EdgeState.OPEN
    # Player now faces right; a turn at the new cell must not corrupt the open edge.
    grid.confirm_turn((1, 0), "right")
    grid.confirm_blocked((1, 0), "right", frame_index=3)
    assert grid.edge_state((0, 0), (1, 0)) is EdgeState.OPEN
    assert grid.get_cell(1, 0).state is CellState.WALKABLE
    # A verified move after a pending turn supersedes the artifact cleanly.
    grid.confirm_turn((1, 0), "down")
    assert grid.confirm_move((1, 0), "down", frame_index=4) == (1, 1)


def test_turn_artifact_is_scoped_to_cell_and_direction() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    grid.confirm_turn((0, 0), "right")
    # A bump in another direction is not the turn artifact.
    grid.confirm_blocked((0, 0), "left", frame_index=1)
    assert grid.get_cell(-1, 0).state is CellState.BLOCKED
    # A new turn + bump for right is an artifact again...
    grid.confirm_turn((0, 0), "right")
    grid.confirm_blocked((0, 0), "right", frame_index=2)
    assert grid.get_cell(1, 0).state is CellState.UNKNOWN
    # ... and the next bump (no turn) is genuine.
    grid.confirm_blocked((0, 0), "right", frame_index=3)
    assert grid.get_cell(1, 0).state is CellState.BLOCKED


# --------------------------------------------------------------------------- #
# Landmarks / explicit marks
# --------------------------------------------------------------------------- #


def test_mark_exit_interactive_and_state() -> None:
    grid = GridMap("room", kind="interior")
    grid.mark_exit((3, 3), direction="down", confidence=0.9)
    assert grid.get_cell(3, 3).state is CellState.EXIT
    assert "exit" in grid.get_cell(3, 3).labels
    assert grid.edge_state((3, 3), (3, 4)) is EdgeState.EXIT

    grid.mark_interactive((5, 5), "npc")
    assert grid.get_cell(5, 5).state is CellState.INTERACTIVE
    assert "npc" in grid.get_cell(5, 5).labels

    # Interactivity on verified floor keeps WALKABLE (never lose passability).
    grid.mark_state((6, 6), CellState.WALKABLE)
    grid.mark_interactive((6, 6), "sign")
    assert grid.get_cell(6, 6).state is CellState.WALKABLE
    assert "sign" in grid.get_cell(6, 6).labels

    grid.mark_state((7, 7), CellState.BLOCKED, confidence=0.9, reason="wall")
    assert grid.get_cell(7, 7).state is CellState.BLOCKED
    assert grid.get_cell(7, 7).confidence == pytest.approx(0.9)
    assert "wall" in grid.get_cell(7, 7).labels

    with pytest.raises(ValueError):
        grid.mark_state((0, 0), "not-a-state")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        grid.mark_exit((0, 0), direction="north")


def test_mark_unreachable_and_deferred_never_downgrade_walkable() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    grid.mark_state((1, 0), CellState.WALKABLE)
    grid.mark_unreachable([(5, 5), (6, 5)], reason="walled off")
    assert grid.get_cell(5, 5).state is CellState.UNREACHABLE
    assert "walled off" in grid.get_cell(5, 5).labels
    grid.mark_unreachable([(1, 0)], reason="contradiction")
    assert grid.get_cell(1, 0).state is CellState.WALKABLE
    grid.mark_deferred([(7, 7)], reason="later")
    assert grid.get_cell(7, 7).state is CellState.DEFERRED


# --------------------------------------------------------------------------- #
# Neighbour queries / reachability / frontier
# --------------------------------------------------------------------------- #


def test_traversable_neighbours_flags() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    grid.mark_state((0, -1), CellState.OBSERVED)
    grid.mark_state((0, 1), CellState.TEMP_BLOCKED)
    grid.mark_state((-1, 0), CellState.BLOCKED)
    grid.mark_state((1, 0), CellState.WALKABLE)
    grid.get_edge((0, 0), (1, 0)).state = EdgeState.BLOCKED

    assert grid.walkable_neighbours((0, 0)) == [(1, 0)]
    assert grid.traversable_neighbours((0, 0)) == [(0, -1)]
    assert grid.traversable_neighbours((0, 0), allow_observed=False) == []
    assert (0, 1) in grid.traversable_neighbours((0, 0), allow_temp_blocked=True)
    assert (1, 0) not in grid.traversable_neighbours((0, 0), allow_temp_blocked=True)


def test_reachable_region() -> None:
    grid = GridMap("a")
    for x in range(3):
        grid.mark_state((x, 0), CellState.WALKABLE)
    grid.mark_state((9, 9), CellState.WALKABLE)  # walkable but disconnected
    region = grid.reachable_region((0, 0))
    assert region == {(0, 0), (1, 0), (2, 0)}
    assert (9, 9) not in region
    # An OBSERVED start cell is still reachable (the player stands on it), but the
    # flood fill does not spread through observed cells by default.
    observed = GridMap("b")
    observed.mark_state((3, 3), CellState.OBSERVED)
    observed.mark_state((4, 3), CellState.OBSERVED)
    assert observed.reachable_region((3, 3)) == {(3, 3)}
    assert observed.reachable_region((3, 3), passable={CellState.OBSERVED}) == {(3, 3), (4, 3)}


def test_frontier_around_reachable_region() -> None:
    grid = GridMap("c")
    grid.mark_state((0, 0), CellState.WALKABLE)
    for blocked in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
        grid.mark_state(blocked, CellState.BLOCKED)
    assert grid.frontier((0, 0)) == []

    grid.mark_state((1, 0), CellState.OBSERVED)
    frontiers = grid.frontier((0, 0))
    assert len(frontiers) == 1
    frontier = frontiers[0]
    assert frontier.cell == (1, 0)
    assert frontier.from_cell == (0, 0)
    assert frontier.direction == "right"
    assert frontier.expected_new > 0

    # Missing (never seen) neighbours are unknown frontier candidates too.
    lonely = GridMap("d")
    lonely.mark_state((0, 0), CellState.WALKABLE)
    cells = {entry.cell for entry in lonely.frontier((0, 0))}
    assert cells == {(0, -1), (0, 1), (-1, 0), (1, 0)}
    # Deterministic ordering/content.
    assert lonely.frontier((0, 0)) == lonely.frontier((0, 0))
    for entry in lonely.frontier((0, 0)):
        assert entry.from_cell == (0, 0)
        assert entry.expected_new > 0


def test_unknown_adjacent_count() -> None:
    grid = GridMap("a")
    grid.mark_state((0, 0), CellState.WALKABLE)
    assert grid.unknown_adjacent_count((0, 0)) == 4
    grid.mark_state((1, 0), CellState.OBSERVED)
    assert grid.unknown_adjacent_count((0, 0)) == 3
    grid.mark_state((-1, 0), CellState.BLOCKED)
    assert grid.unknown_adjacent_count((0, 0)) == 2


# --------------------------------------------------------------------------- #
# Honest coverage
# --------------------------------------------------------------------------- #


def test_coverage_math_is_honest() -> None:
    grid = GridMap("room")
    for x in range(3):
        grid.mark_state((x, 0), CellState.WALKABLE)
        grid.mark_state((x, -1), CellState.BLOCKED)
        grid.mark_state((x, 1), CellState.BLOCKED)
    grid.mark_unreachable([(10, 10)], reason="unreachable")
    grid.mark_deferred([(10, 12)], reason="deferred")

    stats = grid.coverage(start=(0, 0))
    assert stats.area_id == "room"
    assert stats.cells_total == 3 + 6 + 1 + 1
    assert stats.walkable == 3
    assert stats.blocked == 6
    assert stats.unreachable == 1
    assert stats.deferred == 1
    # Only the two open ends of the corridor are still unknown frontier.
    assert stats.unknown_frontier == 2
    assert stats.coverage_ratio == pytest.approx(3.0 / 5.0)
    # Unreachable/deferred are reported separately and never counted as walkable.
    assert stats.walkable == 3
    assert stats.explored_ratio == pytest.approx(9.0 / 11.0)

    # Closing the remaining frontier means the reachable room is fully covered.
    grid.mark_state((-1, 0), CellState.BLOCKED)
    grid.mark_state((3, 0), CellState.BLOCKED)
    closed = grid.coverage(start=(0, 0))
    assert closed.unknown_frontier == 0
    assert closed.coverage_ratio == pytest.approx(1.0)


def test_coverage_of_unverified_area_is_zero_not_fake() -> None:
    room = _open_room(seed=8)
    grid = GridMap("fresh")
    grid.observe_view(_frame_view(room, (4, 4)), _pose(4, 4, valid=False), frame_index=0)
    stats = grid.coverage()
    assert stats.observed == 90
    assert stats.walkable == 0
    assert stats.coverage_ratio == 0.0  # nothing verified: never claim "explored"
    assert stats.unknown_frontier > 0


def test_coverage_empty_grid() -> None:
    stats = GridMap("empty").coverage()
    assert stats.cells_total == 0
    assert stats.coverage_ratio == 0.0
    assert stats.explored_ratio == 0.0
    assert stats.unknown_frontier == 0


def test_coverage_default_start_uses_first_walkable_cell() -> None:
    grid = GridMap("a")
    grid.mark_state((5, 5), CellState.WALKABLE)
    grid.mark_state((6, 5), CellState.WALKABLE)
    default_stats = grid.coverage()
    explicit_stats = grid.coverage(start=(5, 5))
    assert default_stats.to_dict() == explicit_stats.to_dict()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_render_ascii_symbols_path_target_center_and_legend() -> None:
    grid = GridMap("r", kind="interior")
    grid.mark_state((0, 0), CellState.WALKABLE)
    grid.mark_state((1, 0), CellState.BLOCKED)
    grid.mark_state((0, 1), CellState.TEMP_BLOCKED)
    grid.mark_exit((2, 0))

    text = grid.render_ascii(
        center=(0, 0), radius=2, path=[(0, -1), (0, 0)], target=(1, 0), legend=True
    )
    for symbol in (".", "#", "x", "E", "*", "T", "@"):
        assert symbol in text
    assert "legend:" in text
    # header + (2*radius+1) rows + 2 legend lines
    assert text.count("\n") == 1 + 5 + 1
    assert grid.render_ascii(center=(0, 0), radius=2, legend=False).count("\n") == 1 + 4

    without_legend = grid.render_ascii(legend=False)
    assert "legend:" not in without_legend
    assert "@" not in without_legend  # no explicit center -> no player marker
    # Deterministic: same call yields identical output.
    assert grid.render_ascii(legend=False) == grid.render_ascii(legend=False)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _populated_grid() -> GridMap:
    grid = GridMap("room_a", frame_index=7, kind="interior")
    grid.mark_state((0, 0), CellState.WALKABLE)
    grid.confirm_move((0, 0), "right")
    grid.confirm_blocked((1, 0), "right")
    grid.mark_exit((5, 5), direction="up")
    cell = grid.get_cell(4, 4)
    cell.add_hash("hash-aaaa")
    cell.observed_count = 3
    cell.labels.append("sign")
    grid.mark_interactive((4, 5), "npc")
    return grid


def test_dict_roundtrip_preserves_everything() -> None:
    grid = _populated_grid()
    payload = grid.to_dict()
    assert payload["format"] == GRID_MAP_FORMAT
    assert payload["version"] == GRID_MAP_FORMAT_VERSION
    clone = GridMap.from_dict(payload)
    assert clone.area_id == grid.area_id
    assert clone.kind == grid.kind
    assert clone.created_frame == grid.created_frame
    assert set(clone.cells) == set(grid.cells)
    for key, item in grid.cells.items():
        assert clone.cells[key].to_dict() == item.to_dict()
    assert {key: edge.to_dict() for key, edge in clone.edges.items()} == {
        key: edge.to_dict() for key, edge in grid.edges.items()
    }
    assert clone.load_warnings == []


def test_dict_serialization_is_deterministic() -> None:
    grid = GridMap("x")
    for coords in [(2, 2), (0, 0), (1, 1)]:
        grid.get_cell(*coords)
    assert [(cell["x"], cell["y"]) for cell in grid.to_dict()["cells"]] == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]


def test_save_load_atomic_and_no_absolute_paths(tmp_path: Path) -> None:
    grid = _populated_grid()
    path = tmp_path / "nested" / "grid.json"
    grid.save(path)
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp"))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["area_id"] == "room_a"
    assert str(tmp_path) not in path.read_text(encoding="utf-8")

    first_bytes = path.read_bytes()
    grid.save(path)
    assert path.read_bytes() == first_bytes  # deterministic snapshot

    loaded = GridMap.load(path)
    assert loaded.to_dict() == grid.to_dict()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GridMap.load(tmp_path / "nope.json")


def test_load_corrupt_and_future_version(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{this is not json", encoding="utf-8")
    with pytest.raises(ValueError):
        GridMap.load(corrupt)

    future = tmp_path / "future.json"
    future.write_text(
        json.dumps({"format": GRID_MAP_FORMAT, "version": 99, "area_id": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        GridMap.load(future)


def test_from_dict_tolerates_malformed_entries() -> None:
    payload = {
        "format": GRID_MAP_FORMAT,
        "version": 1,
        "area_id": "tolerant",
        "kind": "interior",
        "cells": [
            {"x": 0, "y": 0},
            {"x": "bad"},  # missing y -> skipped
            "junk",  # not a mapping -> skipped
            {"x": 1, "y": 1, "state": "walkable"},
        ],
        "edges": [
            {"a": [0, 0], "b": [1, 0]},
            {"a": [0, 0]},  # missing endpoint -> skipped
            "junk",  # not a mapping -> skipped
        ],
    }
    grid = GridMap.from_dict(payload)
    assert (0, 0) in grid.cells
    assert grid.cells[(1, 1)].state is CellState.WALKABLE
    assert grid.edge((0, 0), (1, 0)) is not None
    assert grid.area_id == "tolerant"
    assert len(grid.load_warnings) == 4

    with pytest.raises(ValueError):
        GridMap.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GridMap.from_dict({"area_id": "x", "cells": "not-a-list"})
