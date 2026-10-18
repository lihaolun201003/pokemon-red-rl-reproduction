"""Unit tests for :mod:`llm_agent.navigation.explorer` (owner: nav-planner).

Like the pathfinder tests these run against a tiny duck-typed ``FakeGrid`` (the
``GridMap`` surface documented in ``NAV_CONTRACT.md`` section 3), so they need no
PyBoy, ROM, network or API key.  A final smoke test uses the real ``GridMap``
once nav-map's module exists.
"""

from __future__ import annotations

import json
import time

import pytest

from llm_agent.navigation.explorer import Explorer, ExplorerConfig
from llm_agent.navigation.pathfinder import Pathfinder
from llm_agent.navigation.types import (
    DIRECTION_DELTAS,
    CellState,
    EdgeState,
    ExploreTarget,
    PathPlan,
)

WALKABLE = CellState.WALKABLE
OBSERVED = CellState.OBSERVED
BLOCKED = CellState.BLOCKED
TEMP_BLOCKED = CellState.TEMP_BLOCKED
EXIT = CellState.EXIT
UNREACHABLE = CellState.UNREACHABLE
DEFERRED = CellState.DEFERRED
INTERACTIVE = CellState.INTERACTIVE


# --------------------------------------------------------------------------- #
# Minimal duck-typed GridMap / Cell
# --------------------------------------------------------------------------- #


class FakeCell:
    def __init__(self, x: int, y: int, state: CellState = CellState.UNKNOWN) -> None:
        self.x = x
        self.y = y
        self.state = state


class FakeGrid:
    """Duck-typed stand-in for ``llm_agent.navigation.grid_map.GridMap``."""

    def __init__(
        self,
        cells: dict[tuple[int, int], CellState] | None = None,
        edges: dict[tuple[tuple[int, int], tuple[int, int]], EdgeState] | None = None,
        *,
        area_id: str = "fake-area",
    ) -> None:
        self.area_id = area_id
        self.cells: dict[tuple[int, int], FakeCell] = {
            (int(x), int(y)): FakeCell(int(x), int(y), state)
            for (x, y), state in (cells or {}).items()
        }
        self.edges = dict(edges or {})

    def cell(self, x: int, y: int) -> FakeCell | None:
        return self.cells.get((int(x), int(y)))

    def get_cell(self, x: int, y: int) -> FakeCell:
        cell = self.cell(x, y)
        if cell is None:
            cell = FakeCell(int(x), int(y), CellState.UNKNOWN)
            self.cells[(int(x), int(y))] = cell
        return cell

    def edge_state(self, a, b) -> EdgeState:
        a = (int(a[0]), int(a[1]))
        b = (int(b[0]), int(b[1]))
        return self.edges.get((a, b), self.edges.get((b, a), EdgeState.UNKNOWN))

    def mark_state(self, cell, state: CellState, **_: object) -> None:
        """Only the state matters to the planner/explorer (fakes the GridMap API)."""
        cell = (int(cell[0]), int(cell[1]))
        existing = self.cells.get(cell)
        if existing is None:
            self.cells[cell] = FakeCell(cell[0], cell[1], state)
        else:
            existing.state = state


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


def row_floor(length: int = 4, y: int = 0) -> dict[tuple[int, int], CellState]:
    """A straight verified corridor along ``y`` (cheap to explore along)."""
    return {(x, y): WALKABLE for x in range(length)}


def boxed_corridor(length: int = 4) -> FakeGrid:
    """Corridor fully enclosed by known BLOCKED cells (one isolated frontier)."""
    cells = row_floor(length)
    for x in range(length):
        cells[(x, -1)] = BLOCKED
        cells[(x, 1)] = BLOCKED
    cells[(-1, 0)] = BLOCKED
    cells[(length, 0)] = BLOCKED
    # single probe-worthy cell at the far end (everything else is known)
    cells[(length - 1, 1)] = OBSERVED
    return FakeGrid(cells)


def closed_room(width: int = 3, height: int = 3) -> FakeGrid:
    """Fully known room: floor ``WALKABLE`` inside a ``BLOCKED`` ring -> complete."""
    cells: dict[tuple[int, int], CellState] = {}
    for y in range(height):
        for x in range(width):
            cells[(x + 1, y + 1)] = WALKABLE
    for y in range(-1, height + 1):
        for x in range(-1, width + 1):
            if (x + 1, y + 1) not in cells:
                cells[(x + 1, y + 1)] = BLOCKED
    return FakeGrid(cells)


def assert_target_consistent(grid: FakeGrid, target: ExploreTarget) -> None:
    """Every candidate must be a legal probe off a reachable walkable cell."""
    assert target.path is not None and target.path.reachable
    assert target.path.cells[0] == target.path.cells[0]  # sanity
    assert target.from_cell in target.path.cells
    assert target.path.cells[-1] == target.from_cell
    assert target.direction in DIRECTION_DELTAS
    assert target.cell == (
        target.from_cell[0] + DIRECTION_DELTAS[target.direction][0],
        target.from_cell[1] + DIRECTION_DELTAS[target.direction][1],
    )
    assert target.cell not in set(target.path.cells)
    for cell, direction in zip(target.path.cells, target.path.directions):
        assert direction in DIRECTION_DELTAS
        assert (cell[0] + DIRECTION_DELTAS[direction][0],
                cell[1] + DIRECTION_DELTAS[direction][1]) in target.path.cells


# --------------------------------------------------------------------------- #
# Frontier candidates & scoring
# --------------------------------------------------------------------------- #


def test_frontier_candidates_are_walkable_adjacent_cells() -> None:
    cells = row_floor(4)
    for x in range(4):
        cells[(x, -1)] = OBSERVED
    grid = FakeGrid(cells)
    explorer = Explorer()

    candidates = explorer.frontier_candidates(grid, (0, 0))
    observed_candidates = [t for t in candidates if t.cell in {(x, -1) for x in range(4)}]
    assert {t.cell for t in observed_candidates} == {(x, -1) for x in range(4)}
    for target in candidates:
        assert_target_consistent(grid, target)
        assert target.from_cell in {(x, 0) for x in range(4)}
        assert target.kind in ("frontier", "exit")
        assert target.path.cells[-1] == target.from_cell
    # best-first ordering: scores are non-increasing
    scores = [t.score for t in candidates]
    assert scores == sorted(scores, reverse=True)


def test_choose_target_prefers_nearer_equal_value_frontier() -> None:
    cells = row_floor(7, y=3)
    cells[(1, 2)] = OBSERVED
    cells[(5, 2)] = OBSERVED
    grid = FakeGrid(cells)
    explorer = Explorer(pathfinder=Pathfinder())

    target = explorer.choose_target(grid, (0, 3))
    assert target is not None
    # both observed frontier cells reveal the same number of unknown cells; the
    # one whose known path is shorter (distance 1 vs distance 5) must win.
    assert target.cell == (1, 2)
    assert target.from_cell == (1, 3)
    assert target.path is not None and target.path.length == 1
    assert_target_consistent(grid, target)


def test_choosing_is_deterministic() -> None:
    cells = row_floor(7, y=3)
    cells[(1, 2)] = OBSERVED
    cells[(5, 2)] = OBSERVED
    grid = FakeGrid(cells)
    first = Explorer().choose_target(grid, (0, 3))
    second = Explorer().choose_target(grid, (0, 3))
    assert first is not None and second is not None
    assert first.to_dict()["cell"] == second.to_dict()["cell"]
    assert first.to_dict()["direction"] == second.to_dict()["direction"]
    assert first.score == pytest.approx(second.score)


def test_expected_new_cells_estimates_the_new_view_strip() -> None:
    grid = FakeGrid({(0, 0): WALKABLE, (1, 0): OBSERVED})
    explorer = Explorer()
    candidates = {t.cell: t for t in explorer.frontier_candidates(grid, (0, 0))}
    # stepping right reveals the strip at x=+6 relative to the current pose: 9 cells
    assert candidates[(1, 0)].expected_new_cells == 9
    # an UNKNOWN target below reveals a 10-cell strip plus the target itself
    assert candidates[(0, 1)].expected_new_cells == 11


def test_exit_bonus_promotes_exit_over_nearer_frontier() -> None:
    cells = row_floor(6, y=2)
    cells[(0, 1)] = OBSERVED  # path length 0 probe
    cells[(5, 1)] = EXIT      # path length 5 probe, but exit bonus applies
    grid = FakeGrid(cells)

    default_choice = Explorer(pathfinder=Pathfinder()).choose_target(grid, (0, 2))
    assert default_choice is not None
    assert default_choice.cell == (5, 1)
    assert default_choice.kind == "exit"

    no_bonus = Explorer(
        ExplorerConfig(exit_bonus=0.0), pathfinder=Pathfinder()
    ).choose_target(grid, (0, 2))
    assert no_bonus is not None
    assert no_bonus.cell == (0, 1)


def test_exit_cell_alone_is_a_candidate() -> None:
    cells: dict[tuple[int, int], CellState] = {}
    for y in range(3):
        for x in range(3):
            cells[(x + 1, y + 1)] = WALKABLE
    for y in range(-1, 4):
        for x in range(-1, 4):
            if (x + 1, y + 1) not in cells:
                cells[(x + 1, y + 1)] = BLOCKED
    cells[(2, 0)] = EXIT
    grid = FakeGrid(cells)
    target = Explorer().choose_target(grid, (1, 1))
    assert target is not None
    assert target.kind == "exit"
    assert target.cell == (2, 0)


# --------------------------------------------------------------------------- #
# max_target_path / max_attempts_per_cell / skip statistics
# --------------------------------------------------------------------------- #


def test_isolated_frontier_uses_the_verified_route() -> None:
    grid = boxed_corridor(4)
    explorer = Explorer()
    target = explorer.choose_target(grid, (0, 0))
    assert target is not None
    assert target.cell == (3, 1)
    assert target.from_cell == (3, 0)
    assert target.direction == "down"
    assert target.path is not None
    assert target.path.cells == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert target.path.directions == ["right"] * 3
    assert target.path.cost == pytest.approx(3.0)


def test_max_target_path_drops_expensive_frontiers() -> None:
    grid = boxed_corridor(4)
    strict = Explorer(ExplorerConfig(max_target_path=2))
    assert strict.choose_target(grid, (0, 0)) is None
    finished, reason, stats = strict.should_finish(grid, (0, 0))
    assert finished is True
    assert stats["skipped"].get("path_too_long") == 1
    assert "path_too_long" in reason

    relaxed = Explorer(ExplorerConfig(max_target_path=3))
    target = relaxed.choose_target(grid, (0, 0))
    assert target is not None and target.cell == (3, 1)


def test_max_attempts_per_cell_blacklists_repeated_failures() -> None:
    grid = boxed_corridor(4)
    explorer = Explorer(ExplorerConfig(max_attempts_per_cell=2))
    target = explorer.choose_target(grid, (0, 0))
    assert target is not None
    explorer.on_attempt(target, step_index=0, outcome="blocked")
    second = explorer.choose_target(grid, (0, 0))
    assert second is not None
    explorer.on_attempt(second, step_index=1, outcome="blocked")
    assert explorer.choose_target(grid, (0, 0)) is None

    finished, _reason, stats = explorer.should_finish(grid, (0, 0))
    assert finished is True
    assert stats["frontier_candidates"] == 0
    assert stats["skipped"].get("exhausted") == 1


def test_impassable_frontiers_are_counted_not_offered() -> None:
    grid = FakeGrid(
        {
            (0, 0): WALKABLE,
            (0, -1): BLOCKED,
            (0, 1): UNREACHABLE,
            (-1, 0): DEFERRED,
            (1, 0): INTERACTIVE,
        }
    )
    explorer = Explorer()
    assert explorer.frontier_candidates(grid, (0, 0)) == []
    assert explorer.choose_target(grid, (0, 0)) is None
    finished, _reason, stats = explorer.should_finish(grid, (0, 0))
    assert finished is True
    assert stats["skipped"] == {
        "blocked_frontier": 1,
        "deferred_frontier": 1,
        "interactive": 1,
        "unreachable_frontier": 1,
    }
    assert stats["unreachable"] == 1
    assert stats["deferred"] == 1
    assert stats["interactive"] == 1


def test_edge_blocked_frontier_is_skipped() -> None:
    grid = FakeGrid(
        {(0, 0): WALKABLE, (1, 0): OBSERVED,
         (0, -1): BLOCKED, (0, 1): BLOCKED, (-1, 0): BLOCKED},
        edges={((0, 0), (1, 0)): EdgeState.BLOCKED},
    )
    explorer = Explorer()
    assert explorer.frontier_candidates(grid, (0, 0)) == []
    finished, _reason, stats = explorer.should_finish(grid, (0, 0))
    assert finished is True
    assert stats["skipped"].get("edge_blocked") == 1


def test_temp_blocked_frontier_is_retried_then_capped() -> None:
    grid = FakeGrid(
        {(0, 0): WALKABLE, (1, 0): TEMP_BLOCKED,
         (0, -1): BLOCKED, (0, 1): BLOCKED, (-1, 0): BLOCKED}
    )
    explorer = Explorer(ExplorerConfig(max_attempts_per_cell=2))
    for index in range(2):
        target = explorer.choose_target(grid, (0, 0))
        assert target is not None
        assert target.cell == (1, 0)
        assert target.kind == "frontier"
        explorer.on_attempt(target, step_index=index, outcome="blocked")
    assert explorer.choose_target(grid, (0, 0)) is None
    finished, _reason, stats = explorer.should_finish(grid, (0, 0))
    assert finished is True
    assert stats["skipped"].get("exhausted") == 1


# --------------------------------------------------------------------------- #
# should_finish: honest completion reporting
# --------------------------------------------------------------------------- #


def test_should_finish_on_fully_known_map() -> None:
    grid = closed_room(3, 3)
    explorer = Explorer()
    finished, reason, stats = explorer.should_finish(grid, (2, 2))
    assert finished is True
    assert "no reachable valuable frontier" in reason
    assert stats["finish_reason"] == "no_frontier"
    assert stats["frontier_candidates"] == 0
    assert stats["walkable"] == 9
    assert stats["blocked"] == 16
    assert explorer.choose_target(grid, (2, 2)) is None


def test_should_finish_reports_unreachable_and_deferred_separately() -> None:
    grid = closed_room(3, 3)
    grid.cells[(20, 20)] = FakeCell(20, 20, UNREACHABLE)
    grid.cells[(21, 20)] = FakeCell(21, 20, DEFERRED)
    explorer = Explorer()
    finished, _reason, stats = explorer.should_finish(grid, (2, 2))
    assert finished is True
    assert stats["unreachable"] == 1
    assert stats["deferred"] == 1
    assert stats["frontier_candidates"] == 0
    # the blocked ring around the room is honestly reported as skipped
    # (12 ring cells are orthogonally adjacent to floor; the 4 corners are not)
    assert stats["skipped"].get("blocked_frontier") == 12
    assert stats["skipped"].get("unreachable_frontier") is None


def test_should_finish_false_when_frontier_exists() -> None:
    grid = boxed_corridor(4)
    explorer = Explorer()
    finished, reason, stats = explorer.should_finish(grid, (0, 0))
    assert finished is False
    assert "reachable frontier available" in reason
    assert stats["frontier_candidates"] == 1
    assert stats["finish_reason"] == "frontier_available"
    assert stats["best_cell"] == [3, 1]
    assert stats["best_score"] > 0


def test_should_finish_after_probing_every_frontier() -> None:
    cells = row_floor(4)
    for x in range(4):
        cells[(x, -1)] = OBSERVED
    grid = FakeGrid(cells)
    explorer = Explorer()
    attempts = 0
    while attempts < 40:
        target = explorer.choose_target(grid, (0, 0))
        if target is None:
            break
        # mimic the controller: a verified blocked bump becomes BLOCKED in the map
        grid.cells[target.cell] = FakeCell(target.cell[0], target.cell[1], BLOCKED)
        explorer.on_attempt(target, step_index=attempts, outcome="blocked")
        attempts += 1
    assert target is None
    finished, reason, stats = explorer.should_finish(grid, (0, 0))
    assert finished is True
    assert stats["finish_reason"] == "no_frontier"
    assert stats["skipped"].get("blocked_frontier") == attempts
    assert attempts > 0
    assert "skipped frontier" in reason


def test_should_finish_no_progress_safety_valve() -> None:
    cells = row_floor(5)
    cells[(2, -1)] = OBSERVED
    cells[(2, 1)] = OBSERVED
    grid = FakeGrid(cells)
    explorer = Explorer(ExplorerConfig(no_progress_limit=2))
    first = explorer.choose_target(grid, (0, 0))
    assert first is not None
    explorer.on_attempt(first, step_index=0, outcome="unknown")
    second = explorer.choose_target(grid, (0, 0))
    assert second is not None
    explorer.on_attempt(second, step_index=1, outcome="blocked")
    finished, reason, stats = explorer.should_finish(grid, (0, 0))
    assert finished is True
    assert "no progress" in reason
    assert stats["finish_reason"] == "no_progress_limit"
    assert stats["consecutive_no_progress"] == 2


# --------------------------------------------------------------------------- #
# No oscillation / history / stats
# --------------------------------------------------------------------------- #


def _two_target_grid() -> tuple[FakeGrid, tuple[int, int], tuple[int, int]]:
    """Two equal-value frontier cells above/below the player, all else known."""
    cells: dict[tuple[int, int], CellState] = row_floor(5)
    for x in range(-3, 8):
        for y in (-3, -2, -1, 1, 2, 3):
            cells[(x, y)] = BLOCKED
    cells[(-1, 0)] = BLOCKED
    cells[(5, 0)] = BLOCKED
    cells[(2, -1)] = OBSERVED
    cells[(2, 1)] = OBSERVED
    return FakeGrid(cells), (2, -1), (2, 1)


def test_no_oscillation_between_two_targets() -> None:
    grid, cell_a, cell_b = _two_target_grid()
    explorer = Explorer(ExplorerConfig(max_attempts_per_cell=2))
    sequence: list[tuple[int, int]] = []
    for index in range(20):
        target = explorer.choose_target(grid, (2, 0))
        if target is None:
            break
        sequence.append(target.cell)
        explorer.on_attempt(target, step_index=index, outcome="blocked")
    assert sequence, "expected at least one target"
    assert target is None, "exploration policy oscillated forever"
    assert len(sequence) <= 2 * 2 + 2
    assert sequence.count(cell_a) <= 2
    assert sequence.count(cell_b) <= 2
    stats = explorer.stats()
    assert stats["outcomes"]["blocked"] == len(sequence)
    assert len(stats["exhausted_cells"]) == 2


def test_selection_cap_terminates_even_without_attempt_reports() -> None:
    grid, _cell_a, _cell_b = _two_target_grid()
    explorer = Explorer(ExplorerConfig(max_attempts_per_cell=2))
    selections = 0
    for _ in range(30):
        target = explorer.choose_target(grid, (2, 0))
        if target is None:
            break
        selections += 1
    assert target is None, "choose_target must not cycle without attempt feedback"
    assert selections <= 2 * (2 * 2) + 2
    stats = explorer.stats()
    assert stats["probes"] == selections
    assert len(stats["exhausted_cells"]) == 2


def test_on_attempt_history_and_validation() -> None:
    grid = boxed_corridor(4)
    explorer = Explorer()
    target = explorer.choose_target(grid, (0, 0))
    assert target is not None
    explorer.on_attempt(target, step_index=7, outcome="MOVED")
    assert explorer.stats()["outcomes"]["moved"] == 1
    assert explorer.stats()["last_outcome"] == "moved"
    with pytest.raises(ValueError):
        explorer.on_attempt(target, step_index=8, outcome="flying")
    with pytest.raises(ValueError):
        explorer.on_attempt(None, step_index=8, outcome="blocked")  # type: ignore[arg-type]
    assert explorer.stats()["history_len"] == 1


def test_transition_outcome_is_accepted_and_neutral() -> None:
    """Additive vocab: the controller reports a crossed exit as ``transition``."""
    cells: dict[tuple[int, int], CellState] = {}
    for y in range(3):
        for x in range(3):
            cells[(x + 1, y + 1)] = WALKABLE
    for y in range(-1, 4):
        for x in range(-1, 4):
            if (x + 1, y + 1) not in cells:
                cells[(x + 1, y + 1)] = BLOCKED
    cells[(2, 0)] = EXIT
    grid = FakeGrid(cells)
    explorer = Explorer()
    target = explorer.choose_target(grid, (1, 1))
    assert target is not None and target.kind == "exit"

    # one failed probe first, to prove transition does not reset/increment it
    explorer.on_attempt(target, step_index=0, outcome="blocked")
    assert explorer.stats()["consecutive_no_progress"] == 1
    explorer.on_attempt(target, step_index=1, outcome="transition")  # must not raise
    stats = explorer.stats()
    assert stats["outcomes"]["transition"] == 1
    assert stats["last_outcome"] == "transition"
    assert stats["consecutive_no_progress"] == 1  # neutral, not a failed probe
    assert stats["attempts"] == 2
    assert stats["history_len"] == 2
    # the frozen contract vocabulary is still accepted/recorded separately
    assert stats["outcomes"]["blocked"] == 1


def test_reset_clears_history() -> None:
    grid = boxed_corridor(4)
    explorer = Explorer()
    target = explorer.choose_target(grid, (0, 0))
    assert target is not None
    explorer.on_attempt(target, step_index=0, outcome="blocked")
    assert explorer.stats()["attempts"] == 1
    explorer.reset()
    stats = explorer.stats()
    assert stats["attempts"] == 0
    assert stats["probes"] == 0
    assert stats["selected_cells"] == {}
    assert stats["exhausted_cells"] == []
    assert stats["last_scan"] is None


def test_stats_and_targets_are_json_serializable() -> None:
    grid = boxed_corridor(4)
    explorer = Explorer()
    target = explorer.choose_target(grid, (0, 0))
    assert target is not None
    json.dumps(target.to_dict())
    json.dumps(explorer.stats())
    _finished, _reason, stats = explorer.should_finish(grid, (0, 0))
    json.dumps(stats)


def test_large_already_walkable_region_is_explorable() -> None:
    size = 15
    cells = {(x, y): WALKABLE for x in range(size) for y in range(size)}
    grid = FakeGrid(cells)
    explorer = Explorer()
    started = time.monotonic()
    target = explorer.choose_target(grid, (7, 7))
    elapsed = time.monotonic() - started
    assert target is not None
    assert target.cell not in cells
    assert target.path is not None
    assert target.path.length == 7
    assert target.cell == (7, -1)  # deterministic: nearest boundary, lexicographic tie
    assert_target_consistent(grid, target)
    assert elapsed < 2.0
    finished, _reason, stats = explorer.should_finish(grid, (7, 7))
    assert finished is False
    assert stats["reachable_walkable"] == size * size
    assert stats["frontier_candidates"] > 0


def test_init_validation() -> None:
    with pytest.raises(ValueError):
        ExplorerConfig(max_attempts_per_cell=0)
    with pytest.raises(ValueError):
        ExplorerConfig(max_target_path=0)
    with pytest.raises(ValueError):
        ExplorerConfig(no_progress_limit=0)
    with pytest.raises(ValueError):
        ExplorerConfig(path_cost_weight=-1.0)
    with pytest.raises(ValueError):
        ExplorerConfig(exit_bonus=float("nan"))
    with pytest.raises(TypeError):
        Explorer(config={"new_cell_weight": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Explorer(pathfinder="nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# End-to-end policy simulation: can the policy map a room and terminate?
# --------------------------------------------------------------------------- #

_VIEW_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (dx, dy) for dy in range(-4, 5) for dx in range(-4, 6)
)


def _set_observed(grid: FakeGrid, pose: tuple[int, int]) -> None:
    """Mimic ``grid.observe_view``: all tiles in the 10x9 camera view become known."""
    for dx, dy in _VIEW_OFFSETS:
        cell = (pose[0] + dx, pose[1] + dy)
        if grid.cell(*cell) is None:
            grid.cells[cell] = FakeCell(cell[0], cell[1], OBSERVED)


def _room_layout() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """9x6 room with border walls, a 2-cell pillar and 26 walkable floor cells."""
    width, height = 9, 6
    pillar = {(3, 2), (3, 3)}
    truth: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    for x in range(width):
        for y in range(height):
            if x in (0, width - 1) or y in (0, height - 1) or (x, y) in pillar:
                walls.add((x, y))
            else:
                truth.add((x, y))
    return truth, walls


def _floor_neighbour_walls(
    walls: set[tuple[int, int]], truth: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Walls orthogonally touching floor: the ones a policy can actually probe."""
    result = set()
    for wall_x, wall_y in walls:
        for dx, dy in DIRECTION_DELTAS.values():
            if (wall_x + dx, wall_y + dy) in truth:
                result.add((wall_x, wall_y))
                break
    return result


def _simulate_exploration(explorer: Explorer, grid, start: tuple[int, int],
                          truth: set[tuple[int, int]], *, max_probes: int = 300):
    """Drive choose_target/on_attempt against a ground-truth room model."""
    pose = start
    probes = 0
    if grid.cell(*start) is None:
        grid.mark_state(start, WALKABLE)
    while probes < max_probes:
        finished, reason, stats = explorer.should_finish(grid, pose)
        if finished:
            return True, reason, stats, probes, pose
        target = explorer.choose_target(grid, pose)
        if target is None:
            return False, "no target", stats, probes, pose
        probes += 1
        if target.cell in truth:
            pose = target.cell
            grid.mark_state(pose, WALKABLE)
            explorer.on_attempt(target, step_index=probes, outcome="moved")
        else:
            grid.mark_state(target.cell, BLOCKED)
            explorer.on_attempt(target, step_index=probes, outcome="blocked")
    return False, "probe limit", explorer.stats(), probes, pose


def test_simulated_room_is_fully_mapped_and_policy_terminates() -> None:
    truth, walls = _room_layout()
    probeable_walls = _floor_neighbour_walls(walls, truth)
    start = (1, 1)

    grid = FakeGrid({start: WALKABLE})
    _set_observed(grid, start)
    explorer = Explorer(pathfinder=Pathfinder())

    finished, reason, stats, probes, _pose = _simulate_exploration(
        explorer, grid, start, truth
    )
    assert finished is True, f"policy did not terminate: {reason}"
    assert "no reachable valuable frontier" in reason
    # every walkable floor cell of the room is discovered and mapped
    assert stats["walkable"] == len(truth)
    # every wall touching the floor was verified blocked (blocked bumps)
    assert stats["blocked"] == len(probeable_walls)
    # and it did that with a sensible number of probe actions
    assert probes < 200
    for cell in truth:
        assert grid.cell(*cell) is not None
        assert grid.cell(*cell).state is WALKABLE
    for cell in probeable_walls:
        assert grid.cell(*cell).state is BLOCKED
    assert stats["finish_reason"] == "no_frontier"
    assert stats["attempts"] == probes
    # completion came from a real "no frontier" judgement, not the safety valve
    assert stats["consecutive_no_progress"] < explorer.config.no_progress_limit


def test_area_switch_resets_attempt_history() -> None:
    grid_a = boxed_corridor(4)
    grid_a.area_id = "area-a"
    explorer = Explorer(ExplorerConfig(max_attempts_per_cell=1))
    target = explorer.choose_target(grid_a, (0, 0))
    assert target is not None
    explorer.on_attempt(target, step_index=0, outcome="blocked")
    assert explorer.choose_target(grid_a, (0, 0)) is None  # blacklisted in area A
    assert explorer.stats()["attempts"] == 1

    grid_b = boxed_corridor(4)
    grid_b.area_id = "area-b"
    # same area-local coordinates, but a different area: history must start fresh
    fresh = explorer.choose_target(grid_b, (0, 0))
    assert fresh is not None
    assert fresh.cell == (3, 1)
    assert explorer.stats()["attempts"] == 0

    # staying inside area B keeps the history (no spurious resets)
    explorer.on_attempt(fresh, step_index=1, outcome="blocked")
    assert explorer.choose_target(grid_b, (0, 0)) is None


# --------------------------------------------------------------------------- #
# Bridge test against the real GridMap
# --------------------------------------------------------------------------- #


def test_real_grid_map_smoke() -> None:
    """Bridge test: runs against ``grid_map.GridMap`` once nav-map delivers it."""
    grid_map_module = pytest.importorskip("llm_agent.navigation.grid_map")
    grid = grid_map_module.GridMap("smoke-explorer-area")
    for x in range(4):
        grid.mark_state((x, 1), WALKABLE)
    grid.mark_state((4, 1), OBSERVED)
    grid.mark_state((0, 0), OBSERVED)

    explorer = Explorer(pathfinder=Pathfinder())
    target = explorer.choose_target(grid, (0, 1), step_index=0)
    assert target is not None
    assert target.path is not None and target.path.reachable
    assert target.from_cell is not None
    assert target.direction in DIRECTION_DELTAS

    finished, reason, stats = explorer.should_finish(grid, (0, 1))
    assert isinstance(finished, bool)
    assert isinstance(reason, str)
    assert isinstance(stats, dict)
    assert "finish_reason" in stats

    explorer.on_attempt(target, step_index=0, outcome="blocked")
    assert explorer.stats()["attempts"] == 1
    grid.mark_state(target.cell, BLOCKED)
    next_target = explorer.choose_target(grid, (0, 1))
    assert next_target is None or next_target.cell != target.cell


def test_real_grid_map_simulated_room() -> None:
    """Same end-to-end mapping simulation, but on the real ``GridMap``."""
    grid_map_module = pytest.importorskip("llm_agent.navigation.grid_map")
    truth, walls = _room_layout()
    probeable_walls = _floor_neighbour_walls(walls, truth)
    start = (1, 1)

    grid = grid_map_module.GridMap("sim-room-real")
    for dx, dy in _VIEW_OFFSETS:
        grid.mark_state((start[0] + dx, start[1] + dy), OBSERVED)
    grid.mark_state(start, WALKABLE)

    explorer = Explorer(pathfinder=Pathfinder())
    finished, reason, stats, probes, _pose = _simulate_exploration(
        explorer, grid, start, truth
    )
    assert finished is True, f"policy did not terminate on real GridMap: {reason}"
    assert stats["walkable"] == len(truth)
    assert stats["blocked"] == len(probeable_walls)
    assert probes < 200
    for cell in truth:
        assert grid.cell(*cell) is not None
        assert grid.cell(*cell).state is WALKABLE
    for cell in probeable_walls:
        assert grid.cell(*cell).state is BLOCKED
