"""Deterministic A* path planning over a per-area logical grid map.

Owner: nav-planner.  Frozen public interface: ``NAV_CONTRACT.md`` section 5.

The planner is a pure function of the *recorded map*: no RAM, no ROM data, no
external map/walkthrough, no randomness and no wall-clock dependence.  The same
grid plus the same query always yields the same :class:`PathPlan`, which keeps
the controller reproducible and testable.

Cost model (:class:`PathCosts`)
------------------------------
Every move pays the cost of its **destination** cell:

==============  =======  =======================================================
``CellState``   default  meaning
==============  =======  =======================================================
``WALKABLE``    1.0      verified floor (walked through before) - cheapest
``OBSERVED``    3.0      seen from a distance, passability not verified - usable
``TEMP_BLOCKED``5.0      dynamic obstacle (NPC), only with ``allow_temp_blocked``
``UNKNOWN``     6.0      never seen; expensive because it may be a wall
``EXIT``        2.0      suspected/confirmed area exit (worth crossing)
``BLOCKED``     inf      persistent obstacle - never crossed
``UNREACHABLE`` inf      proven unreachable - never crossed
``DEFERRED``    inf      deliberately skipped - never crossed
``INTERACTIVE`` inf      sign/NPC/item; :class:`PathCosts` has no field for it
==============  =======  =======================================================

The A* heuristic is ``manhattan(a, b) * heuristic_scale`` where
``heuristic_scale`` is the smallest non-negative per-step cost that can ever be
paid (``min(walkable, observed, exit, unknown, temp_blocked)``).  With the
default costs that is exactly ``PathCosts.walkable`` as required by the
contract, and it stays admissible/consistent for custom cost tables as well.

Unknown-cell bound
------------------
With ``allow_unknown=True`` an unbounded all-unknown plane would make A* search a
diamond around the start.  To keep the search bounded *and* deterministic the
planner only enters ``UNKNOWN`` cells inside the bounding box of the known cells
plus ``start``/``goal`` (grown by a small margin; a detour around a known
obstacle that touches the bounding box needs only 1-2 cells outside it).  Known
cells are never bounded.  This does not change the optimality of any plan whose
optimal route goes through known cells; it just stops the search from "walking
into the void".

Failure handling: every public planning method returns a
``PathPlan(reachable=False, reason=...)`` instead of raising.  Only malformed
*arguments* (e.g. non-adjacent cells passed to :meth:`Pathfinder.plan_from_cells`)
raise ``ValueError``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

from .types import (
    DIRECTIONS,
    DIRECTION_DELTAS,
    CellState,
    EdgeState,
    PathPlan,
    direction_of,
    manhattan,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (grid_map is nav-map's file)
    from .grid_map import GridMap

__all__ = [
    "PathCosts",
    "Pathfinder",
    "cell_state",
    "edge_state",
    "normalize_cell",
]

#: A numeric cost above which a cell is considered impassable.
INF: float = float("inf")

#: Extra cells searched around the bounding box of known cells for UNKNOWN cells.
_UNKNOWN_BOUND_MARGIN: int = 2

#: ``CellState`` -> ``PathCosts`` attribute name.
_COST_FIELDS: dict[CellState, str] = {
    CellState.WALKABLE: "walkable",
    CellState.OBSERVED: "observed",
    CellState.TEMP_BLOCKED: "temp_blocked",
    CellState.UNKNOWN: "unknown",
    CellState.EXIT: "exit",
    CellState.BLOCKED: "blocked",
    CellState.UNREACHABLE: "unreachable",
    CellState.DEFERRED: "deferred",
}

_CostKey = tuple[int, int]


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #


@dataclass
class PathCosts:
    """Per-state traversal costs used by :class:`Pathfinder` (contract section 5).

    ``BLOCKED`` / ``UNREACHABLE`` / ``DEFERRED`` default to ``inf``: those cells
    are impassable no matter which ``allow_*`` flags the caller passes.
    ``INTERACTIVE`` has no field and is always impassable (a sign/NPC/item is not
    a walkable destination).
    """

    walkable: float = 1.0
    observed: float = 3.0        # seen but not verified
    temp_blocked: float = 5.0    # dynamic obstacle; opt-in via allow_temp_blocked
    unknown: float = 6.0         # conservative: exploring unknown costs more
    exit: float = 2.0
    blocked: float = float("inf")
    unreachable: float = float("inf")
    deferred: float = float("inf")

    def __post_init__(self) -> None:
        for name in _COST_FIELDS.values():
            value = float(getattr(self, name))
            if math.isnan(value) or value < 0.0:
                raise ValueError(
                    f"PathCosts.{name} must be a non-negative number or +inf, got {value!r}"
                )
            setattr(self, name, value)

    def get(self, state: CellState) -> float:
        """Raw configured cost for ``state`` (ignoring ``allow_*`` flags)."""
        field = _COST_FIELDS.get(state)
        if field is None:  # INTERACTIVE (and any future state): no cost -> wall
            return INF
        return float(getattr(self, field))

    def entry_cost(
        self,
        state: CellState,
        *,
        allow_unknown: bool = True,
        allow_temp_blocked: bool = False,
    ) -> float:
        """Cost of stepping onto a cell in ``state``, or ``inf`` if not allowed."""
        if state is CellState.UNKNOWN:
            return self.unknown if allow_unknown else INF
        if state is CellState.TEMP_BLOCKED:
            return self.temp_blocked if allow_temp_blocked else INF
        return self.get(state)


def _finite(values: Iterable[float]) -> Iterator[float]:
    for value in values:
        if math.isfinite(value) and value >= 0.0:
            yield float(value)


# --------------------------------------------------------------------------- #
# Duck-typed GridMap access helpers
# --------------------------------------------------------------------------- #


def normalize_cell(cell: Any) -> _CostKey:
    """Coerce a 2-element sequence to ``(int, int)``."""
    x, y = cell[0], cell[1]
    return int(x), int(y)


def _normalize_state(value: Any) -> CellState | None:
    """Best-effort conversion of a GridMap state value to :class:`CellState`."""
    if isinstance(value, CellState):
        return value
    if value is None:
        return None
    try:
        return CellState(value)
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return None


def cell_state(grid: GridMap | Any, cell: _CostKey) -> CellState:
    """Return the recorded state of ``cell`` (``UNKNOWN`` when absent/never seen)."""
    cell = normalize_cell(cell)
    getter = getattr(grid, "cell", None)
    obj: Any = None
    if callable(getter):
        try:
            obj = getter(cell[0], cell[1])
        except (TypeError, ValueError, KeyError):  # pragma: no cover - defensive
            obj = None
    if obj is None:
        cells = getattr(grid, "cells", None)
        if cells is not None and hasattr(cells, "get"):
            obj = cells.get(cell)
    if obj is None:
        return CellState.UNKNOWN
    if isinstance(obj, CellState):
        return obj
    state = _normalize_state(getattr(obj, "state", None))
    return state if state is not None else CellState.UNKNOWN


def edge_state(grid: GridMap | Any, a: _CostKey, b: _CostKey) -> EdgeState | None:
    """Return the recorded edge state between ``a`` and ``b`` (or ``None``)."""
    getter = getattr(grid, "edge_state", None)
    if not callable(getter):
        return None
    try:
        value = getter(normalize_cell(a), normalize_cell(b))
    except (TypeError, ValueError, KeyError):  # pragma: no cover - defensive
        return None
    if isinstance(value, EdgeState):
        return value
    try:
        return EdgeState(value)
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return None


def _known_cell_keys(grid: GridMap | Any) -> list[_CostKey]:
    """All cell keys currently stored in the map (any state, including UNKNOWN)."""
    cells = getattr(grid, "cells", None)
    if cells is None:
        return []
    keys: list[_CostKey] = []
    if hasattr(cells, "keys"):
        for key in cells.keys():
            try:
                keys.append(normalize_cell(key))
            except (TypeError, IndexError):  # pragma: no cover - defensive
                continue
    else:  # pragma: no cover - iterable of Cell objects
        for cell in cells:
            try:
                keys.append((int(cell.x), int(cell.y)))
            except (AttributeError, TypeError):
                continue
    return keys


# --------------------------------------------------------------------------- #
# Pathfinder
# --------------------------------------------------------------------------- #


class Pathfinder:
    """Deterministic, bounded A* planner over a per-area ``GridMap``.

    Parameters
    ----------
    costs:
        Per-state traversal costs, see :class:`PathCosts`.  ``None`` uses the
        contract defaults.
    max_expansions:
        Hard cap on A*/Dijkstra node expansions per query.  When the cap is hit
        the query returns ``reachable=False`` with an explanatory reason instead
        of freezing the caller.
    """

    def __init__(self, costs: PathCosts | None = None, *, max_expansions: int = 60000) -> None:
        self.costs: PathCosts = costs if costs is not None else PathCosts()
        if not isinstance(self.costs, PathCosts):
            raise TypeError(f"costs must be a PathCosts instance, got {type(costs)!r}")
        max_expansions = int(max_expansions)
        if max_expansions <= 0:
            raise ValueError(f"max_expansions must be positive, got {max_expansions}")
        self.max_expansions: int = max_expansions
        finite = list(
            _finite(
                (
                    self.costs.walkable,
                    self.costs.observed,
                    self.costs.exit,
                    self.costs.unknown,
                    self.costs.temp_blocked,
                )
            )
        )
        # Admissible heuristic scale: the cheapest step any plan could ever pay.
        self._h_scale: float = min(finite) if finite else 1.0

    # -- public helpers ---------------------------------------------------- #

    @property
    def heuristic_scale(self) -> float:
        """Per-tile cost used by the Manhattan heuristic (``<=`` every step cost)."""
        return self._h_scale

    def heuristic(self, a: _CostKey, b: _CostKey) -> float:
        """Admissible Manhattan heuristic between two cells."""
        return manhattan(normalize_cell(a), normalize_cell(b)) * self._h_scale

    def direction_between(self, a: _CostKey, b: _CostKey) -> str:
        """Cardinal direction to step from ``a`` to ``b``.

        Returns ``""`` when ``a == b``.  Raises ``ValueError`` when the cells are
        not orthogonally adjacent (callers must pass a legal grid step).
        """
        a, b = normalize_cell(a), normalize_cell(b)
        delta = (b[0] - a[0], b[1] - a[1])
        direction = direction_of(delta)
        if direction is None:
            if delta == (0, 0):
                return ""
            raise ValueError(f"cells are not orthogonally adjacent: {a} -> {b}")
        return direction

    def entry_cost(
        self,
        grid: GridMap | Any,
        cell: _CostKey,
        *,
        allow_unknown: bool = True,
        allow_temp_blocked: bool = False,
    ) -> float:
        """Cost of stepping onto ``cell`` under the given flags (``inf`` = no)."""
        state = cell_state(grid, cell)
        return self.costs.entry_cost(
            state, allow_unknown=allow_unknown, allow_temp_blocked=allow_temp_blocked
        )

    def traversable(
        self,
        grid: GridMap | Any,
        a: _CostKey,
        b: _CostKey,
        *,
        allow_unknown: bool = True,
        allow_temp_blocked: bool = False,
    ) -> bool:
        """True when one step ``a -> b`` is allowed (cell state *and* edge state)."""
        if self._edge_blocks(grid, normalize_cell(a), normalize_cell(b), allow_temp_blocked):
            return False
        return math.isfinite(
            self.entry_cost(
                grid, b, allow_unknown=allow_unknown, allow_temp_blocked=allow_temp_blocked
            )
        )

    # -- core public API (contract) ---------------------------------------- #

    def plan(
        self,
        grid: GridMap | Any,
        start: _CostKey,
        goal: _CostKey,
        *,
        allow_unknown: bool = True,
        allow_temp_blocked: bool = False,
    ) -> PathPlan:
        """A* from ``start`` to ``goal``.

        Returns ``PathPlan(reachable=True, cells=[start..goal], directions=...)``
        on success and ``PathPlan(reachable=False, reason=...)`` otherwise.  It
        never raises for "no path".
        """
        start, goal = normalize_cell(start), normalize_cell(goal)
        if start == goal:
            return PathPlan(
                cells=[start],
                directions=[],
                cost=0.0,
                reachable=True,
                reason="start is goal",
                explored=1,
            )

        blocked_reason = self._goal_impassable_reason(
            grid, goal, allow_unknown=allow_unknown, allow_temp_blocked=allow_temp_blocked
        )
        if blocked_reason is not None:
            return PathPlan(reachable=False, reason=blocked_reason, explored=0)

        result = self._astar(
            grid,
            start=start,
            goals={goal},
            extra_points=[start, goal],
            allow_unknown=allow_unknown,
            allow_temp_blocked=allow_temp_blocked,
        )
        if result is None:  # pragma: no cover - _astar always returns a PathPlan
            return PathPlan(reachable=False, reason="no path", explored=0)
        if result.reachable:
            result.reason = "path found"
        return result

    def plan_to_nearest(
        self,
        grid: GridMap | Any,
        start: _CostKey,
        goals: Iterable[_CostKey],
        *,
        allow_unknown: bool = True,
    ) -> PathPlan | None:
        """Plan to the cheapest reachable cell of ``goals``.

        Returns ``None`` only when ``goals`` is empty.  When no goal is reachable
        a ``PathPlan(reachable=False, reason=...)`` is returned (never an
        exception), naming the rejected goals and their reasons.
        """
        start = normalize_cell(start)
        ordered: list[_CostKey] = []
        seen: set[_CostKey] = set()
        for raw in goals:
            cell = normalize_cell(raw)
            if cell not in seen:
                seen.add(cell)
                ordered.append(cell)
        if not ordered:
            return None
        if start in seen:
            return PathPlan(
                cells=[start],
                directions=[],
                cost=0.0,
                reachable=True,
                reason="start is a goal",
                explored=1,
            )

        valid: set[_CostKey] = set()
        rejected: list[str] = []
        for cell in ordered:
            reason = self._goal_impassable_reason(
                grid, cell, allow_unknown=allow_unknown, allow_temp_blocked=False
            )
            if reason is None:
                valid.add(cell)
            else:
                rejected.append(reason)
        if not valid:
            return PathPlan(
                reachable=False,
                reason="no goal is traversable: " + "; ".join(rejected[:4]),
                explored=0,
            )

        return self._dijkstra_to_any(
            grid,
            start=start,
            goals=valid,
            extra_points=[start, *ordered],
            allow_unknown=allow_unknown,
            allow_temp_blocked=False,
        )

    def reachable(
        self,
        grid: GridMap | Any,
        start: _CostKey,
        *,
        allow_observed: bool = False,
    ) -> set[_CostKey]:
        """Verified-reachable region around ``start``.

        BFS over cells the player may actually stand on: ``WALKABLE`` and
        ``EXIT`` always, ``OBSERVED`` only with ``allow_observed=True``.
        ``UNKNOWN``/``BLOCKED``/``TEMP_BLOCKED``/unreachable/deferred are never
        included.  ``start`` itself is always included (that is where we are).
        Blocked *edges* are respected.
        """
        start = normalize_cell(start)
        passable = {CellState.WALKABLE, CellState.EXIT}
        if allow_observed:
            passable.add(CellState.OBSERVED)
        seen: set[_CostKey] = {start}
        queue: list[_CostKey] = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for direction in DIRECTIONS:
                dx, dy = DIRECTION_DELTAS[direction]
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in seen:
                    continue
                if self._edge_blocks(grid, cur, nb, allow_temp_blocked=False):
                    continue
                if cell_state(grid, nb) in passable:
                    seen.add(nb)
                    queue.append(nb)
        return seen

    @staticmethod
    def plan_from_cells(cells: Sequence[_CostKey], cost: float = 0.0) -> PathPlan:
        """Build a :class:`PathPlan` from an already-known 4-connected cell chain.

        ``directions`` are derived between consecutive cells.  Raises
        ``ValueError`` for an empty list or a non-adjacent step (callers use this
        only for paths they already know are legal).
        """
        chain = [normalize_cell(cell) for cell in cells]
        if not chain:
            raise ValueError("cannot build a PathPlan from an empty cell list")
        directions: list[str] = []
        for a, b in zip(chain, chain[1:]):
            direction = direction_of((b[0] - a[0], b[1] - a[1]))
            if direction is None:
                raise ValueError(f"PathPlan cells must be 4-connected: {a} -> {b}")
            directions.append(direction)
        return PathPlan(
            cells=chain,
            directions=directions,
            cost=float(cost),
            reachable=True,
            reason="provided path",
            explored=len(chain),
        )

    # -- internals --------------------------------------------------------- #

    def _edge_blocks(
        self,
        grid: GridMap | Any,
        a: _CostKey,
        b: _CostKey,
        allow_temp_blocked: bool,
    ) -> bool:
        state = edge_state(grid, a, b)
        if state is EdgeState.BLOCKED:
            return True
        if state is EdgeState.TEMP_BLOCKED and not allow_temp_blocked:
            return True
        return False

    def _goal_impassable_reason(
        self,
        grid: GridMap | Any,
        goal: _CostKey,
        *,
        allow_unknown: bool,
        allow_temp_blocked: bool,
    ) -> str | None:
        state = cell_state(grid, goal)
        if state is CellState.UNKNOWN:
            return None if allow_unknown else (
                f"goal {goal} is unknown and allow_unknown=False"
            )
        if state is CellState.TEMP_BLOCKED:
            return None if allow_temp_blocked else (
                f"goal {goal} is temp_blocked and allow_temp_blocked=False"
            )
        if state in (CellState.WALKABLE, CellState.OBSERVED, CellState.EXIT):
            return None
        return f"goal {goal} is {state.value} (impassable)"

    def _search_bounds(
        self, grid: GridMap | Any, extra_points: Iterable[_CostKey]
    ) -> tuple[int, int, int, int]:
        xs: list[int] = []
        ys: list[int] = []
        for x, y in _known_cell_keys(grid):
            xs.append(x)
            ys.append(y)
        for point in extra_points:
            x, y = normalize_cell(point)
            xs.append(x)
            ys.append(y)
        if not xs:  # pragma: no cover - start/goal always present
            return (-_UNKNOWN_BOUND_MARGIN, -_UNKNOWN_BOUND_MARGIN, _UNKNOWN_BOUND_MARGIN, _UNKNOWN_BOUND_MARGIN)
        return (
            min(xs) - _UNKNOWN_BOUND_MARGIN,
            min(ys) - _UNKNOWN_BOUND_MARGIN,
            max(xs) + _UNKNOWN_BOUND_MARGIN,
            max(ys) + _UNKNOWN_BOUND_MARGIN,
        )

    @staticmethod
    def _within(bounds: tuple[int, int, int, int], cell: _CostKey) -> bool:
        minx, miny, maxx, maxy = bounds
        return minx <= cell[0] <= maxx and miny <= cell[1] <= maxy

    def _neighbours(self, cell: _CostKey) -> Iterator[tuple[str, _CostKey]]:
        """Deterministic 4-connected neighbour iteration (``DIRECTIONS`` order)."""
        for direction in DIRECTIONS:
            dx, dy = DIRECTION_DELTAS[direction]
            yield direction, (cell[0] + dx, cell[1] + dy)

    def _astar(
        self,
        grid: GridMap | Any,
        *,
        start: _CostKey,
        goals: set[_CostKey],
        extra_points: Iterable[_CostKey],
        allow_unknown: bool,
        allow_temp_blocked: bool,
    ) -> PathPlan:
        """A* to the nearest cell of ``goals``.

        ``goals`` is usually a singleton; the multi-goal form is used by
        :meth:`plan_to_nearest`-style queries and is still admissible because the
        heuristic is the minimum Manhattan distance to any goal.
        """
        bounds = (
            self._search_bounds(grid, extra_points) if allow_unknown else None
        )
        goal_list = list(goals)
        goal_set = set(goal_list)

        def h(cell: _CostKey) -> float:
            if len(goal_list) == 1:
                return manhattan(cell, goal_list[0]) * self._h_scale
            return min(manhattan(cell, goal) for goal in goal_list) * self._h_scale

        counter = 0
        parents: dict[_CostKey, _CostKey | None] = {start: None}
        g_scores: dict[_CostKey, float] = {start: 0.0}
        closed: set[_CostKey] = set()
        heap: list[tuple[float, float, int, _CostKey]] = [(h(start), h(start), counter, start)]
        expansions = 0

        while heap:
            _f, _heuristic, _tie, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            expansions += 1

            if cur in goal_set:
                return self._reconstruct(
                    start=start,
                    goal=cur,
                    parents=parents,
                    cost=g_scores[cur],
                    explored=expansions,
                    reason=f"path found to nearest goal {cur}" if len(goal_set) > 1 else "path found",
                )

            if expansions >= self.max_expansions:
                return PathPlan(
                    reachable=False,
                    reason=f"expansion limit reached ({self.max_expansions} cells) before goal",
                    explored=expansions,
                )

            for _direction, nb in self._neighbours(cur):
                if nb in closed:
                    continue
                if self._edge_blocks(grid, cur, nb, allow_temp_blocked):
                    continue
                state = cell_state(grid, nb)
                step = self.costs.entry_cost(
                    state,
                    allow_unknown=allow_unknown,
                    allow_temp_blocked=allow_temp_blocked,
                )
                if not math.isfinite(step):
                    continue
                if allow_unknown and state is CellState.UNKNOWN:
                    if bounds is not None and not self._within(bounds, nb):
                        continue
                candidate = g_scores[cur] + step
                previous = g_scores.get(nb)
                if previous is not None and candidate >= previous - 1e-12:
                    continue
                g_scores[nb] = candidate
                parents[nb] = cur
                heuristic = h(nb)
                counter += 1
                heapq.heappush(heap, (candidate + heuristic, heuristic, counter, nb))

        goal_text = ", ".join(str(goal) for goal in sorted(goal_list)[:4])
        return PathPlan(
            reachable=False,
            reason=f"no path from {start} to {goal_text} (exhausted {expansions} cells)",
            explored=expansions,
        )

    def _dijkstra_to_any(
        self,
        grid: GridMap | Any,
        *,
        start: _CostKey,
        goals: set[_CostKey],
        extra_points: Iterable[_CostKey],
        allow_unknown: bool,
        allow_temp_blocked: bool,
    ) -> PathPlan:
        """Dijkstra to the cheapest goal (used by :meth:`plan_to_nearest`).

        Optimal without a heuristic; bounded by the same search box and
        ``max_expansions`` as A*.  Deterministic tie-breaking: (cost, steps,
        insertion order).
        """
        bounds = self._search_bounds(grid, extra_points) if allow_unknown else None
        parents: dict[_CostKey, _CostKey | None] = {start: None}
        dist: dict[_CostKey, float] = {start: 0.0}
        closed: set[_CostKey] = set()
        counter = 0
        heap: list[tuple[float, int, int, _CostKey]] = [(0.0, 0, counter, start)]
        expansions = 0

        while heap:
            cost, _steps, _tie, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            expansions += 1
            if cur in goals:
                return self._reconstruct(
                    start=start,
                    goal=cur,
                    parents=parents,
                    cost=cost,
                    explored=expansions,
                    reason=f"nearest reachable goal {cur}",
                )
            if expansions >= self.max_expansions:
                return PathPlan(
                    reachable=False,
                    reason=f"expansion limit reached ({self.max_expansions} cells) before any goal",
                    explored=expansions,
                )
            for _direction, nb in self._neighbours(cur):
                if nb in closed:
                    continue
                if self._edge_blocks(grid, cur, nb, allow_temp_blocked):
                    continue
                state = cell_state(grid, nb)
                step = self.costs.entry_cost(
                    state,
                    allow_unknown=allow_unknown,
                    allow_temp_blocked=allow_temp_blocked,
                )
                if not math.isfinite(step):
                    continue
                if allow_unknown and state is CellState.UNKNOWN:
                    if bounds is not None and not self._within(bounds, nb):
                        continue
                candidate = cost + step
                previous = dist.get(nb)
                if previous is not None and candidate >= previous - 1e-12:
                    continue
                dist[nb] = candidate
                parents[nb] = cur
                counter += 1
                heapq.heappush(heap, (candidate, _steps + 1, counter, nb))

        goal_text = ", ".join(str(goal) for goal in sorted(goals)[:4])
        return PathPlan(
            reachable=False,
            reason=f"no reachable goal among [{goal_text}] (exhausted {expansions} cells)",
            explored=expansions,
        )

    @staticmethod
    def _reconstruct(
        *,
        start: _CostKey,
        goal: _CostKey,
        parents: dict[_CostKey, _CostKey | None],
        cost: float,
        explored: int,
        reason: str,
    ) -> PathPlan:
        cells: list[_CostKey] = []
        node: _CostKey | None = goal
        while node is not None:
            cells.append(node)
            node = parents.get(node)
        cells.reverse()
        if not cells or cells[0] != start:  # pragma: no cover - invariant guard
            return PathPlan(reachable=False, reason="internal reconstruction error", explored=explored)
        directions: list[str] = []
        for a, b in zip(cells, cells[1:]):
            direction = direction_of((b[0] - a[0], b[1] - a[1]))
            if direction is None:  # pragma: no cover - invariant guard
                return PathPlan(reachable=False, reason="internal non-cardinal step", explored=explored)
            directions.append(direction)
        return PathPlan(
            cells=cells,
            directions=directions,
            cost=float(cost),
            reachable=True,
            reason=reason,
            explored=int(explored),
        )
