"""Unit tests for :mod:`llm_agent.navigation.pathfinder` (owner: nav-planner).

The tests run against a tiny duck-typed ``FakeGrid`` that mirrors the
``GridMap`` surface documented in ``NAV_CONTRACT.md`` section 3.  This keeps the
planner testable without the real map implementation (owned by another module)
and without PyBoy / the ROM / the network.  A final smoke test automatically
uses the real ``GridMap`` as soon as that module exists.
"""

from __future__ import annotations

import heapq
import json
import random

import pytest

from llm_agent.navigation.pathfinder import (
    INF,
    PathCosts,
    Pathfinder,
    cell_state,
    edge_state,
)
from llm_agent.navigation.types import (
    DIRECTION_DELTAS,
    CellState,
    EdgeState,
)


# --------------------------------------------------------------------------- #
# Minimal duck-typed GridMap / Cell (same public surface as the real one)
# --------------------------------------------------------------------------- #


class FakeCell:
    """Minimal cell object: the planner/explorer only need ``state``."""

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


_CHARS = {
    ".": CellState.WALKABLE,
    "o": CellState.OBSERVED,
    "#": CellState.BLOCKED,
    "E": CellState.EXIT,
    "T": CellState.TEMP_BLOCKED,
    "U": CellState.UNREACHABLE,
    "D": CellState.DEFERRED,
    "I": CellState.INTERACTIVE,
}


def grid_from_rows(
    rows: list[str],
    edges: dict | None = None,
    *,
    area_id: str = "rows",
) -> FakeGrid:
    """Build a fake grid from ASCII rows; ``" "`` = never observed (UNKNOWN)."""
    cells: dict[tuple[int, int], CellState] = {}
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char == " ":
                continue
            cells[(x, y)] = _CHARS[char]
    return FakeGrid(cells, edges, area_id=area_id)


# --------------------------------------------------------------------------- #
# Assertion helpers
# --------------------------------------------------------------------------- #


def assert_valid_plan(plan, start, goal=None) -> None:
    """A reachable plan must be a legal, direction-consistent 4-connected route."""
    assert plan.reachable is True, f"expected a reachable plan, got {plan.reason!r}"
    assert plan.cells, "a reachable plan must contain the start cell"
    assert plan.cells[0] == start
    if goal is not None:
        assert plan.cells[-1] == goal
    assert len(plan.directions) == len(plan.cells) - 1
    for a, b, direction in zip(plan.cells, plan.cells[1:], plan.directions):
        assert DIRECTION_DELTAS[direction] == (b[0] - a[0], b[1] - a[1])
    assert plan.cost >= 0.0


def path_cost(grid: FakeGrid, cells, costs: PathCosts | None = None) -> float:
    """Reference cost of a cell chain: sum of the entry cost of each step."""
    costs = costs or PathCosts()
    total = 0.0
    for cell in cells[1:]:
        total += costs.get(cell_state(grid, cell))
    return total


# --------------------------------------------------------------------------- #
# Basic planning
# --------------------------------------------------------------------------- #


def test_simple_straight_path() -> None:
    grid = grid_from_rows(["....."])
    plan = Pathfinder().plan(grid, (0, 0), (4, 0))
    assert_valid_plan(plan, (0, 0), (4, 0))
    assert plan.directions == ["right"] * 4
    assert plan.cost == pytest.approx(4.0)
    assert plan.length == 4
    assert plan.explored >= 1


def test_goal_equals_start() -> None:
    grid = grid_from_rows(["."])
    plan = Pathfinder().plan(grid, (1, 1), (1, 1))
    assert plan.reachable is True
    assert plan.cells == [(1, 1)]
    assert plan.directions == []
    assert plan.cost == 0.0
    assert plan.length == 0


def test_detour_around_wall() -> None:
    grid = grid_from_rows(
        [
            ".....",
            "..#..",
            ".....",
        ]
    )
    plan = Pathfinder().plan(grid, (0, 1), (4, 1), allow_unknown=False)
    assert_valid_plan(plan, (0, 1), (4, 1))
    # shortest detour: 4 right + 1 up + 1 down
    assert plan.length == 6
    assert plan.cost == pytest.approx(6.0)
    assert (2, 1) not in plan.cells  # the wall is never crossed


def test_prefers_verified_route_over_unknown_shortcut() -> None:
    # Direct middle row is UNKNOWN (unknown cost 6/step); walkable rows cost 1.
    walkable = CellState.WALKABLE
    cells: dict[tuple[int, int], CellState] = {
        **{(x, 0): walkable for x in range(5)},
        **{(x, 2): walkable for x in range(5)},
        (0, 1): walkable,
        (4, 1): walkable,
    }
    grid = FakeGrid(cells)
    plan = Pathfinder().plan(grid, (0, 1), (4, 1))
    assert_valid_plan(plan, (0, 1), (4, 1))
    assert plan.cost == pytest.approx(6.0)  # down + 4 right + up
    assert all(cell_state(grid, cell) is CellState.WALKABLE for cell in plan.cells)


def test_observed_cells_are_usable_but_cost_more() -> None:
    short_unknown = grid_from_rows([".o oooooo"])  # start(0) observed(1) ...
    plan_unknown = Pathfinder().plan(short_unknown, (0, 0), (1, 0))
    assert_valid_plan(plan_unknown, (0, 0), (1, 0))
    assert plan_unknown.cost == pytest.approx(3.0)  # observed costs 3

    grid = grid_from_rows([".o."])
    plan = Pathfinder().plan(grid, (0, 0), (2, 0))
    assert_valid_plan(plan, (0, 0), (2, 0))
    assert plan.cost == pytest.approx(4.0)  # 3 (observed) + 1 (walkable)


def test_exit_cell_is_traversable_at_exit_cost() -> None:
    grid = grid_from_rows([".E."])
    plan = Pathfinder().plan(grid, (0, 0), (2, 0), allow_unknown=False)
    assert_valid_plan(plan, (0, 0), (2, 0))
    assert plan.cells == [(0, 0), (1, 0), (2, 0)]
    assert plan.cost == pytest.approx(3.0)  # exit 2 + walkable 1


# --------------------------------------------------------------------------- #
# Impassable states / failure handling (never raise for "no path")
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state_char, expected_word",
    [
        ("#", "blocked"),
        ("U", "unreachable"),
        ("D", "deferred"),
        ("I", "interactive"),
    ],
)
def test_impassable_goal_returns_reason(state_char: str, expected_word: str) -> None:
    grid = grid_from_rows(["." + state_char])
    plan = Pathfinder().plan(grid, (0, 0), (1, 0))
    assert plan.reachable is False
    assert plan.cells == []
    assert expected_word in plan.reason
    assert plan.cost == 0.0


def test_impassable_cells_are_never_crossed() -> None:
    # A blocked wall separates the two halves; with allow_unknown=False there is
    # no path at all and with allow_unknown=True any detour must avoid the wall.
    grid = grid_from_rows(
        [
            "...#...",
            "...#...",
            "...U...",
            "...D...",
        ]
    )
    plan = Pathfinder().plan(grid, (0, 0), (6, 0), allow_unknown=False)
    assert plan.reachable is False
    assert "no path" in plan.reason

    loose = Pathfinder().plan(grid, (0, 0), (6, 0), allow_unknown=True)
    if loose.reachable:
        for cell in loose.cells:
            assert cell_state(grid, cell) not in (
                CellState.BLOCKED,
                CellState.UNREACHABLE,
                CellState.DEFERRED,
            )


def test_allow_unknown_false_rejects_unknown_goal() -> None:
    grid = grid_from_rows(["."])
    plan = Pathfinder().plan(grid, (0, 0), (2, 0), allow_unknown=False)
    assert plan.reachable is False
    assert "unknown" in plan.reason
    assert plan.explored == 0

    allowed = Pathfinder().plan(grid, (0, 0), (2, 0), allow_unknown=True)
    assert_valid_plan(allowed, (0, 0), (2, 0))
    assert allowed.cost == pytest.approx(12.0)  # two unknown steps


def test_unknown_start_is_a_legal_origin() -> None:
    # start itself is never a destination; it may be UNKNOWN right after entry.
    grid = grid_from_rows([" ."])
    plan = Pathfinder().plan(grid, (0, 0), (1, 0), allow_unknown=False)
    assert_valid_plan(plan, (0, 0), (1, 0))


def test_plan_never_raises_on_empty_grid() -> None:
    grid = FakeGrid()
    plan = Pathfinder().plan(grid, (0, 0), (2, 2), allow_unknown=False)
    assert plan.reachable is False
    assert isinstance(plan.reason, str) and plan.reason
    allowed = Pathfinder().plan(grid, (0, 0), (2, 2))
    assert_valid_plan(allowed, (0, 0), (2, 2))


def test_expansion_limit_is_bounded_and_reported() -> None:
    grid = grid_from_rows(["."])
    pathfinder = Pathfinder(max_expansions=10)
    plan = pathfinder.plan(grid, (0, 0), (40, 0))
    assert plan.reachable is False
    assert "expansion limit" in plan.reason
    assert plan.explored == 10
    assert isinstance(plan.cost, float)


def test_unknown_expansion_stays_within_bounds() -> None:
    # Pure unknown plane: the bounded search must finish fast and explore few cells.
    grid = grid_from_rows(["."])
    plan = Pathfinder().plan(grid, (0, 0), (40, 0))
    assert_valid_plan(plan, (0, 0), (40, 0))
    assert plan.explored <= 500


# --------------------------------------------------------------------------- #
# Cost model details / temp blocked / edges
# --------------------------------------------------------------------------- #


def test_temp_blocked_is_opt_in() -> None:
    grid = grid_from_rows([".T."])
    pathfinder = Pathfinder()
    blocked = pathfinder.plan(grid, (0, 0), (2, 0), allow_unknown=False)
    assert blocked.reachable is False
    assert "no path" in blocked.reason

    allowed = pathfinder.plan(
        grid, (0, 0), (2, 0), allow_unknown=False, allow_temp_blocked=True
    )
    assert_valid_plan(allowed, (0, 0), (2, 0))
    assert allowed.cost == pytest.approx(1.0 + 5.0)


def test_blocked_edge_blocks_the_step() -> None:
    grid = grid_from_rows([".."], edges={((0, 0), (1, 0)): EdgeState.BLOCKED})
    pathfinder = Pathfinder()
    plan = pathfinder.plan(grid, (0, 0), (1, 0), allow_unknown=False)
    assert plan.reachable is False
    assert pathfinder.traversable(grid, (0, 0), (1, 0), allow_unknown=False) is False
    assert pathfinder.reachable(grid, (0, 0)) == {(0, 0)}


def test_temp_blocked_edge_is_opt_in() -> None:
    grid = grid_from_rows(
        [".."], edges={((0, 0), (1, 0)): EdgeState.TEMP_BLOCKED}
    )
    pathfinder = Pathfinder()
    assert pathfinder.plan(grid, (0, 0), (1, 0), allow_unknown=False).reachable is False
    plan = pathfinder.plan(
        grid, (0, 0), (1, 0), allow_unknown=False, allow_temp_blocked=True
    )
    assert_valid_plan(plan, (0, 0), (1, 0))


def test_custom_costs_change_the_price() -> None:
    grid = grid_from_rows(["...."])
    pathfinder = Pathfinder(PathCosts(walkable=2.5))
    plan = pathfinder.plan(grid, (0, 0), (3, 0))
    assert plan.cost == pytest.approx(7.5)
    # heuristic scale = cheapest step cost that can ever be paid (default exit=2)
    assert pathfinder.heuristic_scale == pytest.approx(2.0)


def test_reach_also_respects_the_heuristic_scale_min_cost() -> None:
    costs = PathCosts(walkable=3.0, observed=1.0, exit=0.5)
    pathfinder = Pathfinder(costs)
    assert pathfinder.heuristic_scale == pytest.approx(0.5)
    grid = grid_from_rows([".o."])
    plan = pathfinder.plan(grid, (0, 0), (2, 0), allow_unknown=False)
    assert plan.cost == pytest.approx(3.0 + 1.0)


def test_invalid_costs_and_limits_raise() -> None:
    with pytest.raises(ValueError):
        PathCosts(walkable=-1.0)
    with pytest.raises(ValueError):
        PathCosts(unknown=float("nan"))
    with pytest.raises(ValueError):
        Pathfinder(max_expansions=0)
    with pytest.raises(TypeError):
        Pathfinder(costs={"walkable": 1.0})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# plan_to_nearest
# --------------------------------------------------------------------------- #


def test_plan_to_nearest_picks_the_cheapest_goal() -> None:
    grid = grid_from_rows(["....."])
    plan = Pathfinder().plan_to_nearest(grid, (0, 0), [(4, 0), (1, 0)])
    assert plan is not None
    assert_valid_plan(plan, (0, 0), (1, 0))
    assert plan.cost == pytest.approx(1.0)


def test_plan_to_nearest_ignores_impassable_goal() -> None:
    grid = grid_from_rows([".#."])
    plan = Pathfinder().plan_to_nearest(grid, (0, 0), [(1, 0), (2, 0)])
    assert plan is not None
    assert_valid_plan(plan, (0, 0), (2, 0))


def test_plan_to_nearest_empty_goals_returns_none() -> None:
    grid = grid_from_rows(["."])
    assert Pathfinder().plan_to_nearest(grid, (0, 0), []) is None


def test_plan_to_nearest_all_goals_impassable_reports_reason() -> None:
    grid = grid_from_rows(["..#"])
    plan = Pathfinder().plan_to_nearest(grid, (0, 0), [(2, 0)])
    assert plan is not None
    assert plan.reachable is False
    assert "no goal is traversable" in plan.reason
    assert plan.explored == 0


def test_plan_to_nearest_start_is_goal() -> None:
    grid = grid_from_rows([".."])
    plan = Pathfinder().plan_to_nearest(grid, (1, 0), [(1, 0), (0, 0)])
    assert plan is not None
    assert plan.reachable is True
    assert plan.cells == [(1, 0)]
    assert plan.cost == 0.0


def test_plan_to_nearest_no_reachable_goal_reports_reason() -> None:
    # The only valid goal is sealed off behind BLOCKED cells.
    grid = grid_from_rows(
        [
            ".#.",
            "###",
            "...",
        ]
    )
    plan = Pathfinder().plan_to_nearest(grid, (0, 2), [(2, 0)], allow_unknown=False)
    assert plan is not None
    assert plan.reachable is False
    assert "no reachable goal" in plan.reason


# --------------------------------------------------------------------------- #
# reachable / direction_between / plan_from_cells
# --------------------------------------------------------------------------- #


def test_reachable_follows_verified_cells_only() -> None:
    grid = grid_from_rows(["....."])
    pathfinder = Pathfinder()
    assert pathfinder.reachable(grid, (0, 0)) == {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}
    assert (5, 0) not in pathfinder.reachable(grid, (0, 0))


def test_reachable_can_include_observed_cells() -> None:
    grid = grid_from_rows([".oo#."])
    pathfinder = Pathfinder()
    assert pathfinder.reachable(grid, (0, 0)) == {(0, 0)}
    assert pathfinder.reachable(grid, (0, 0), allow_observed=True) == {
        (0, 0),
        (1, 0),
        (2, 0),
    }


def test_reachable_never_includes_unknown_or_blocked() -> None:
    grid = grid_from_rows([". U D T I # E"])
    reachable = Pathfinder().reachable(grid, (0, 0))
    assert reachable == {(0, 0)}  # every neighbour is passability-unknown/blocked
    assert all(
        cell_state(grid, cell) in (CellState.WALKABLE, CellState.EXIT)
        for cell in reachable
    )


def test_reachable_includes_adjacent_exit_cell() -> None:
    grid = grid_from_rows([".E"])
    assert Pathfinder().reachable(grid, (0, 0)) == {(0, 0), (1, 0)}


def test_reachable_bounded_around_start() -> None:
    grid = grid_from_rows(["....", "....", "...."])
    reachable = Pathfinder().reachable(grid, (1, 1))
    assert len(reachable) == 12
    assert (0, 0) in reachable


def test_direction_between() -> None:
    pathfinder = Pathfinder()
    assert pathfinder.direction_between((0, 0), (0, -1)) == "up"
    assert pathfinder.direction_between((0, 0), (0, 1)) == "down"
    assert pathfinder.direction_between((0, 0), (-1, 0)) == "left"
    assert pathfinder.direction_between((0, 0), (1, 0)) == "right"
    assert pathfinder.direction_between((2, 3), (2, 3)) == ""
    with pytest.raises(ValueError):
        pathfinder.direction_between((0, 0), (2, 0))
    with pytest.raises(ValueError):
        pathfinder.direction_between((0, 0), (1, 1))


def test_plan_from_cells() -> None:
    plan = Pathfinder.plan_from_cells([(0, 0), (0, 1), (1, 1)], cost=2.5)
    assert plan.reachable is True
    assert plan.cells == [(0, 0), (0, 1), (1, 1)]
    assert plan.directions == ["down", "right"]
    assert plan.cost == pytest.approx(2.5)
    assert plan.length == 2

    single = Pathfinder.plan_from_cells([(4, 5)])
    assert single.reachable is True
    assert single.cells == [(4, 5)]
    assert single.directions == []

    with pytest.raises(ValueError):
        Pathfinder.plan_from_cells([])
    with pytest.raises(ValueError):
        Pathfinder.plan_from_cells([(0, 0), (1, 1)])


# --------------------------------------------------------------------------- #
# Determinism / optimality
# --------------------------------------------------------------------------- #


def test_planning_is_deterministic() -> None:
    grid = grid_from_rows(
        [
            ".....",
            ".....",
            ".....",
        ]
    )
    pathfinder = Pathfinder()
    plans = [pathfinder.plan(grid, (0, 0), (4, 2), allow_unknown=False) for _ in range(5)]
    first = plans[0]
    for plan in plans[1:]:
        assert plan.cells == first.cells
        assert plan.directions == first.directions
        assert plan.cost == pytest.approx(first.cost)
        assert plan.explored == first.explored


def _reference_dijkstra(
    grid: FakeGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    costs: PathCosts,
) -> float | None:
    """Independent optimal cost used to verify the A* result."""
    allowed = {CellState.WALKABLE, CellState.OBSERVED, CellState.EXIT}
    dist = {start: 0.0}
    heap = [(0.0, start)]
    done: set[tuple[int, int]] = set()
    while heap:
        cost, cell = heapq.heappop(heap)
        if cell in done:
            continue
        done.add(cell)
        if cell == goal:
            return cost
        for dx, dy in DIRECTION_DELTAS.values():
            nb = (cell[0] + dx, cell[1] + dy)
            if nb in done or cell_state(grid, nb) not in allowed:
                continue
            candidate = cost + costs.get(cell_state(grid, nb))
            if candidate < dist.get(nb, INF) - 1e-12:
                dist[nb] = candidate
                heapq.heappush(heap, (candidate, nb))
    return None


def test_astar_matches_reference_optimum_on_random_grids() -> None:
    rng = random.Random(7)
    pathfinder = Pathfinder()
    costs = PathCosts()
    checked = 0
    for _ in range(25):
        states: dict[tuple[int, int], CellState] = {}
        for y in range(6):
            for x in range(6):
                roll = rng.random()
                if roll < 0.70:
                    states[(x, y)] = CellState.WALKABLE
                elif roll < 0.85:
                    states[(x, y)] = CellState.OBSERVED
                else:
                    states[(x, y)] = CellState.BLOCKED
        states[(0, 5)] = CellState.WALKABLE
        states[(5, 0)] = CellState.WALKABLE
        grid = FakeGrid(states)
        expected = _reference_dijkstra(grid, (0, 5), (5, 0), costs)
        plan = pathfinder.plan(grid, (0, 5), (5, 0), allow_unknown=False)
        if expected is None:
            assert plan.reachable is False
        else:
            assert_valid_plan(plan, (0, 5), (5, 0))
            assert plan.cost == pytest.approx(expected)
            assert plan.cost == pytest.approx(path_cost(grid, plan.cells, costs))
            checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# JSON friendliness / dynamic bridge to the real GridMap
# --------------------------------------------------------------------------- #


def test_path_plan_to_dict_is_json_serializable() -> None:
    grid = grid_from_rows(["...."])
    plan = Pathfinder().plan(grid, (0, 0), (3, 0))
    encoded = json.dumps(plan.to_dict())
    decoded = json.loads(encoded)
    assert decoded["reachable"] is True
    assert decoded["cells"] == [[0, 0], [1, 0], [2, 0], [3, 0]]
    assert decoded["length"] == 3


def test_real_grid_map_smoke() -> None:
    """Bridge test: runs against ``grid_map.GridMap`` once nav-map delivers it."""
    grid_map_module = pytest.importorskip("llm_agent.navigation.grid_map")
    grid_map_class = grid_map_module.GridMap

    grid = grid_map_class("smoke-area")
    for x in range(4):
        grid.mark_state((x, 0), CellState.WALKABLE)
    grid.mark_state((0, 1), CellState.OBSERVED)
    grid.mark_state((4, 0), CellState.OBSERVED)

    pathfinder = Pathfinder()
    plan = pathfinder.plan(grid, (0, 0), (3, 0), allow_unknown=False)
    assert plan.reachable is True
    assert plan.cells[0] == (0, 0)
    assert plan.cells[-1] == (3, 0)
    assert plan.cost == pytest.approx(3.0)
    assert (0, 1) in pathfinder.reachable(grid, (0, 0), allow_observed=True)
    assert (0, 1) not in pathfinder.reachable(grid, (0, 0))
    assert cell_state(grid, (3, 0)) is CellState.WALKABLE
    assert edge_state(grid, (0, 0), (1, 0)) is not None
