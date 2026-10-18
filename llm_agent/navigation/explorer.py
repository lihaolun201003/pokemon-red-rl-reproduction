"""Frontier-based exploration policy for one known area.

Owner: nav-planner.  Frozen public interface: ``NAV_CONTRACT.md`` section 5.

The explorer decides *where to go next* from the map the agent has recorded so
far; it never touches the emulator, the framebuffer or game RAM.  Its inputs are
a ``GridMap``, the current cell and the planner; its outputs are
:class:`~llm_agent.navigation.types.ExploreTarget` objects plus honest
completion reports.

Frontier definition
-------------------
A *frontier candidate* is a cell that

* is **not** part of the verified walkable region reachable from ``start``
  (``WALKABLE`` cells connected through non-blocked edges; ``EXIT`` cells are
  transition targets and are deliberately not part of the region), and
* is orthogonally adjacent to a cell of that region (the ``from_cell`` we can
  stand on), and
* is worth probing: ``OBSERVED`` (seen floor/wall whose passability is not
  verified yet), ``TEMP_BLOCKED`` (dynamic obstacle, retry later) or ``EXIT``.
  ``BLOCKED`` / ``UNREACHABLE`` / ``DEFERRED`` / ``INTERACTIVE`` cells are never
  probed by the autonomous policy (they are counted in the stats, not hidden).

``UNKNOWN`` cells are theoretically possible as neighbours of the verified
region (e.g. after a relocalization); they are probed too with a risk penalty
because a blind probe can be a wasted action.

Scoring
-------
``frontier_candidates`` returns every candidate with

``score = new_cell_weight * expected_new_cells
         + (exit_bonus if the target kind is "exit" else 0)
         - path_cost_weight * path.cost
         - repeat_penalty * min(1, failed_attempts / max_attempts_per_cell)
         - (probe_unknown_cost if the target cell is still UNKNOWN else 0)``

``expected_new_cells`` estimates how many cells the probe would reveal: of all
tiles that become visible when standing on the target cell (the 10x9 view
defined by ``types.VIEW_W`` / ``VIEW_H`` / ``VIEW_PLAYER``) but were not visible
from the ``from_cell``, how many are currently ``UNKNOWN`` (plus the target itself
when it is ``UNKNOWN``).  This is a bounded, honest estimate - the true gain is
only known after the action.

Ordering is fully deterministic: ``(-score, path_length, cell)``.

Anti-oscillation and termination
--------------------------------
* every selected target is counted and every reported ``on_attempt`` outcome is
  stored in memory (not persisted);
* a cell with ``>= max_attempts_per_cell`` failed attempts is blacklisted, and
  even a caller that never reports attempts cannot select one cell forever
  (defensive selection cap ``2 * max_attempts_per_cell``);
* repeated failures shrink the score through ``repeat_penalty``;
* ``should_finish`` returns ``True`` only when there is no useful candidate left
  (all boundary cells either probed successfully/blocked, filtered by attempt
  limits or path budget) or after ``no_progress_limit`` consecutive futile probe
  attempts (a documented safety valve) - and in both cases the reason/stats name
  exactly what was skipped and why.

The policy works for a freshly created map as well as for a map that already
contains a large walked region (re-exploration after re-entering an area).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from .pathfinder import (
    Pathfinder,
    PathCosts,
    cell_state,
    edge_state,
    normalize_cell,
)
from .types import (
    DIRECTIONS,
    DIRECTION_DELTAS,
    VIEW_H,
    VIEW_PLAYER,
    VIEW_W,
    CellState,
    EdgeState,
    ExploreTarget,
    PathPlan,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .grid_map import GridMap

__all__ = ["ExplorerConfig", "Explorer"]

#: Screen offsets (dx, dy) of all tiles visible around the player, derived from
#: the frozen coordinate conventions: the player is at screen tile (4, 4) of a
#: 10x9 tile screen, so a pose p sees p + (dx, dy) for the offsets below.
_VIEW_DX: tuple[int, ...] = tuple(range(-VIEW_PLAYER[0], VIEW_W - VIEW_PLAYER[0]))
_VIEW_DY: tuple[int, ...] = tuple(range(-VIEW_PLAYER[1], VIEW_H - VIEW_PLAYER[1]))

#: Attempt outcomes accepted by :meth:`Explorer.on_attempt`.  The first four are
#: the frozen contract vocabulary (``NAV_CONTRACT.md`` section 5); ``transition``
#: is an additive, neutral outcome the controller reports when a probe crossed an
#: area exit (not a failed probe, so it never counts as "no progress").
_OUTCOMES: tuple[str, ...] = ("moved", "blocked", "unknown", "event", "transition")

#: Maximum number of recent attempt records kept in memory.
_HISTORY_MAXLEN: int = 512


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class ExplorerConfig:
    """Tunables of the exploration policy (contract section 5)."""

    probe_unknown_cost: float = 6.0   # risk penalty when the target cell is never seen
    new_cell_weight: float = 1.6      # value of one expected newly observed cell
    path_cost_weight: float = 0.35    # penalty per unit of planned path cost
    exit_bonus: float = 6.0           # extra value for probing a suspected exit
    repeat_penalty: float = 8.0       # penalty for a cell that already failed
    max_attempts_per_cell: int = 2    # hard cap before a frontier cell is dropped
    no_progress_limit: int = 30       # consecutive futile attempts -> finish
    max_target_path: int = 80         # ignore targets whose known path is longer

    def __post_init__(self) -> None:
        for name in (
            "probe_unknown_cost",
            "new_cell_weight",
            "path_cost_weight",
            "exit_bonus",
            "repeat_penalty",
        ):
            value = float(getattr(self, name))
            if math.isnan(value) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"ExplorerConfig.{name} must be a finite number >= 0, got {value!r}")
            setattr(self, name, value)
        for name, minimum in (
            ("max_attempts_per_cell", 1),
            ("no_progress_limit", 1),
            ("max_target_path", 1),
        ):
            value = int(getattr(self, name))
            if value < minimum:
                raise ValueError(f"ExplorerConfig.{name} must be >= {minimum}, got {value}")
            setattr(self, name, value)


# --------------------------------------------------------------------------- #
# Small internal value objects
# --------------------------------------------------------------------------- #


@dataclass
class _Attempt:
    """One recorded probe attempt (memory only, never persisted)."""

    cell: tuple[int, int]
    from_cell: tuple[int, int]
    direction: str
    step_index: int
    outcome: str


@dataclass
class _ScanResult:
    """One full frontier scan (candidates + map diagnostics, deterministic)."""

    candidates: list[ExploreTarget] = field(default_factory=list)
    skips: dict[str, int] = field(default_factory=dict)
    dist: dict[tuple[int, int], float] = field(default_factory=dict)
    steps: dict[tuple[int, int], int] = field(default_factory=dict)
    prev: dict[tuple[int, int], tuple[int, int] | None] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Explorer
# --------------------------------------------------------------------------- #


class Explorer:
    """Frontier-based exploration policy over one ``GridMap``.

    Parameters
    ----------
    config:
        Policy tunables, see :class:`ExplorerConfig`.
    pathfinder:
        Planner used for the verified route to each candidate's ``from_cell``.
        When omitted a :class:`~llm_agent.navigation.pathfinder.Pathfinder` is
        created whose only custom cost is
        ``unknown = config.probe_unknown_cost``.
    """

    def __init__(
        self,
        config: ExplorerConfig | None = None,
        pathfinder: Pathfinder | None = None,
    ) -> None:
        self.config: ExplorerConfig = config if config is not None else ExplorerConfig()
        if not isinstance(self.config, ExplorerConfig):
            raise TypeError(f"config must be an ExplorerConfig, got {type(config)!r}")
        if pathfinder is None:
            pathfinder = Pathfinder(
                PathCosts(unknown=self.config.probe_unknown_cost)
            )
        if not isinstance(pathfinder, Pathfinder):
            raise TypeError(f"pathfinder must be a Pathfinder, got {type(pathfinder)!r}")
        self.pathfinder: Pathfinder = pathfinder
        #: Defensive cap on how often one cell may be *selected* even when the
        #: caller forgets to report attempts.
        self._max_selections: int = max(2, self.config.max_attempts_per_cell * 2)
        self.reset()

    # -- state ------------------------------------------------------------- #

    def reset(self) -> None:
        """Forget all attempt history / counters (map data is untouched).

        Call this when the agent switches area; a :class:`Explorer` also detects
        an area switch automatically (see :meth:`_ensure_area`), so a shared
        instance never confuses cell coordinates of two different areas.
        """
        self._attempts: dict[tuple[int, int], list[_Attempt]] = {}
        self._selections: dict[tuple[int, int], int] = {}
        self._exhausted: set[tuple[int, int]] = set()
        self._history: deque[_Attempt] = deque(maxlen=_HISTORY_MAXLEN)
        self._last_target: ExploreTarget | None = None
        self._last_outcome: str = ""
        self._consecutive_no_progress: int = 0
        self._probes: int = 0
        self._outcome_counts: dict[str, int] = {outcome: 0 for outcome in _OUTCOMES}
        self._last_scan: _ScanResult | None = None
        self._area_key: tuple[Any, ...] | None = None

    def _ensure_area(self, grid: GridMap | Any) -> None:
        """Reset per-area history when a query arrives for a different area.

        Attempt history is keyed by *area-local* cell coordinates, so it must not
        leak from one area into another.  The controller may share one Explorer
        across area transitions without calling :meth:`reset`; this defensive
        check keeps that correct (the explicit method stays available).
        """
        key = (
            id(grid),
            str(getattr(grid, "area_id", "")),
            int(getattr(grid, "created_frame", 0) or 0),
        )
        if self._area_key is not None and self._area_key != key:
            self.reset()
        self._area_key = key

    # -- public API -------------------------------------------------------- #

    def frontier_candidates(
        self, grid: GridMap | Any, start: tuple[int, int]
    ) -> list[ExploreTarget]:
        """All currently useful probe targets, best first (deterministic).

        Returns an empty list when nothing is worth probing (see
        :meth:`should_finish` for the honest explanation).
        """
        scan = self._scan(grid, start)
        self._last_scan = scan
        return scan.candidates

    def choose_target(
        self,
        grid: GridMap | Any,
        start: tuple[int, int],
        *,
        step_index: int = 0,
    ) -> ExploreTarget | None:
        """Select the best frontier target, or ``None`` when none is useful."""
        scan = self._scan(grid, start)
        self._last_scan = scan
        if not scan.candidates:
            self._last_target = None
            return None
        target = scan.candidates[0]
        cell = target.cell
        selections = self._selections.get(cell, 0) + 1
        self._selections[cell] = selections
        self._last_target = target
        self._probes += 1
        if selections >= self._max_selections:
            self._exhausted.add(cell)
        target.reason = (
            f"{target.kind} probe step={int(step_index)} "
            f"path={0 if target.path is None else target.path.length} "
            f"attempts={len(self._attempts.get(cell, ()))}/"
            f"{self.config.max_attempts_per_cell} "
            f"score={target.score:.2f}"
        )
        return target

    def should_finish(
        self, grid: GridMap | Any, start: tuple[int, int]
    ) -> tuple[bool, str, dict]:
        """Decide whether autonomous exploration of this area is done.

        Returns ``(finished, reason, stats)``.  ``finished`` is ``True`` only
        when no reachable, valuable frontier is left - or when the
        ``no_progress_limit`` safety valve fired (the reason says so).  The stats
        dict separates unreachable/deferred/probed-away cells and names every
        skip category, so callers can report coverage honestly.
        """
        scan = self._scan(grid, start)
        self._last_scan = scan
        stats = self._completion_stats(grid, start, scan)

        limit = max(1, int(self.config.no_progress_limit))
        if self._consecutive_no_progress >= limit:
            stats["finish_reason"] = "no_progress_limit"
            reason = (
                f"no progress: {self._consecutive_no_progress} consecutive futile probe "
                f"attempt(s) (limit {limit}); {len(scan.candidates)} candidate(s) remain"
            )
            return True, reason, stats

        if scan.candidates:
            stats["finish_reason"] = "frontier_available"
            return (
                False,
                f"reachable frontier available: {len(scan.candidates)} candidate(s)",
                stats,
            )

        stats["finish_reason"] = "no_frontier"
        return True, self._finished_reason(stats), stats

    def on_attempt(
        self,
        target: ExploreTarget,
        *,
        step_index: int,
        outcome: str,
    ) -> None:
        """Record the verified outcome of following/probing ``target``.

        ``outcome`` is one of ``moved|blocked|unknown|event`` (contract section
        5) plus the additive ``transition`` value used when the probe crossed an
        area exit.  ``moved`` resets the no-progress counter,
        ``blocked``/``unknown`` increment it, while ``event`` and ``transition``
        are neutral (a transition ends the run/area; it is not a failed probe).
        Repeated attempts blacklist the target cell after
        ``max_attempts_per_cell`` attempts.
        """
        if target is None:  # pragma: no cover - defensive, caller bug
            raise ValueError("on_attempt() needs the ExploreTarget that was attempted")
        outcome_key = str(outcome).strip().lower()
        if outcome_key not in _OUTCOMES:
            raise ValueError(
                f"invalid attempt outcome {outcome!r}; expected one of {_OUTCOMES}"
            )
        cell = normalize_cell(target.cell)
        from_cell = normalize_cell(target.from_cell)
        record = _Attempt(
            cell=cell,
            from_cell=from_cell,
            direction=str(target.direction),
            step_index=int(step_index),
            outcome=outcome_key,
        )
        self._attempts.setdefault(cell, []).append(record)
        self._history.append(record)
        self._outcome_counts[outcome_key] = self._outcome_counts.get(outcome_key, 0) + 1
        self._last_outcome = outcome_key
        if self._last_target is not None and normalize_cell(self._last_target.cell) == cell:
            self._last_target = None
        if outcome_key == "moved":
            self._consecutive_no_progress = 0
        elif outcome_key in ("blocked", "unknown"):
            self._consecutive_no_progress += 1
        # "event" / "transition" are neutral: no progress counter change.
        if len(self._attempts[cell]) >= self.config.max_attempts_per_cell:
            self._exhausted.add(cell)

    def stats(self) -> dict:
        """JSON-friendly snapshot of the explorer's memory (not the map)."""
        config = self.config
        last = None
        if self._last_target is not None:
            last = self._last_target.to_dict()
        return {
            "probes": int(self._probes),
            "attempts": int(sum(len(items) for items in self._attempts.values())),
            "outcomes": dict(sorted(self._outcome_counts.items())),
            "consecutive_no_progress": int(self._consecutive_no_progress),
            "exhausted_cells": sorted([list(cell) for cell in self._exhausted]),
            "selected_cells": {
                f"{cell[0]},{cell[1]}": int(count)
                for cell, count in sorted(self._selections.items())
            },
            "last_target": last,
            "last_outcome": self._last_outcome,
            "history_len": int(len(self._history)),
            "last_scan": None if self._last_scan is None else {
                "candidates": len(self._last_scan.candidates),
                **dict(sorted(self._last_scan.skips.items())),
            },
            "config": {
                "probe_unknown_cost": config.probe_unknown_cost,
                "new_cell_weight": config.new_cell_weight,
                "path_cost_weight": config.path_cost_weight,
                "exit_bonus": config.exit_bonus,
                "repeat_penalty": config.repeat_penalty,
                "max_attempts_per_cell": config.max_attempts_per_cell,
                "no_progress_limit": config.no_progress_limit,
                "max_target_path": config.max_target_path,
            },
        }

    # -- internals: region / candidates ------------------------------------ #

    def _region_search(
        self, grid: GridMap | Any, start: tuple[int, int]
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
        """Dijkstra over the verified WALKABLE region around ``start``.

        Returns ``(cost, steps, prev)``: cost of entering every reachable
        ``WALKABLE`` cell, number of moves from ``start``, and the predecessor
        chain used to rebuild the verified route.  Only ``WALKABLE`` cells are
        bases for probing and only non-blocked edges are crossed; ``start`` is
        always included even if its own state is not ``WALKABLE`` yet.

        ``EXIT`` cells are deliberately *not* part of the region: they are
        transition targets, so the policy must be able to return them as
        ``kind="exit"`` probes instead of treating them as already-explored
        floor (the controller reports the area transition when such a probe
        succeeds).
        """
        start = normalize_cell(start)
        costs = self.pathfinder.costs
        dist: dict[tuple[int, int], float] = {start: 0.0}
        steps: dict[tuple[int, int], int] = {start: 0}
        prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        closed: set[tuple[int, int]] = set()
        counter = 0
        heap: list[tuple[float, int, int, tuple[int, int]]] = [(0.0, 0, counter, start)]
        while heap:
            cost, nsteps, _tie, cell = heapq.heappop(heap)
            if cell in closed:
                continue
            closed.add(cell)
            for direction in DIRECTIONS:
                dx, dy = DIRECTION_DELTAS[direction]
                nb = (cell[0] + dx, cell[1] + dy)
                if nb in closed:
                    continue
                if cell_state(grid, nb) is not CellState.WALKABLE:
                    continue
                if not self.pathfinder.traversable(
                    grid, cell, nb, allow_unknown=False, allow_temp_blocked=False
                ):
                    continue
                step = costs.walkable
                if not math.isfinite(step) or step < 0.0:
                    continue
                candidate = cost + step
                previous = dist.get(nb)
                if previous is not None and candidate >= previous - 1e-12:
                    continue
                dist[nb] = candidate
                steps[nb] = nsteps + 1
                prev[nb] = cell
                counter += 1
                heapq.heappush(heap, (candidate, nsteps + 1, counter, nb))
        return dist, steps, prev

    def _plan_to(
        self,
        prev: dict[tuple[int, int], tuple[int, int] | None],
        start: tuple[int, int],
        from_cell: tuple[int, int],
        cost: float,
    ) -> PathPlan:
        """Rebuild the verified route ``start .. from_cell`` as a ``PathPlan``."""
        chain: list[tuple[int, int]] = []
        node: tuple[int, int] | None = from_cell
        while node is not None:
            chain.append(node)
            node = prev.get(node)
        chain.reverse()
        if not chain or chain[0] != start:  # pragma: no cover - invariant guard
            chain = [start, from_cell] if start != from_cell else [start]
        return Pathfinder.plan_from_cells(chain, cost=cost)

    @staticmethod
    def _visible_cells(cell: tuple[int, int]) -> set[tuple[int, int]]:
        """All world cells visible when standing on ``cell`` (10x9 camera view)."""
        x, y = cell
        return {(x + dx, y + dy) for dy in _VIEW_DY for dx in _VIEW_DX}

    def _expected_new_cells(
        self,
        grid: GridMap | Any,
        from_cell: tuple[int, int],
        target: tuple[int, int],
        *,
        visible_before: frozenset[tuple[int, int]] | None = None,
    ) -> int:
        """Estimate newly observed cells if the probe from ``from_cell`` succeeds.

        Counts currently ``UNKNOWN`` cells in the strip that becomes visible when
        standing on ``target`` but is invisible from ``from_cell``, plus the target
        cell itself when it is still ``UNKNOWN``.
        """
        before = (
            self._visible_cells(from_cell)
            if visible_before is None
            else set(visible_before)
        )
        after = self._visible_cells(target)
        newly_visible = after - before
        expected = sum(
            1 for cell in newly_visible if cell_state(grid, cell) is CellState.UNKNOWN
        )
        if cell_state(grid, target) is CellState.UNKNOWN:
            expected += 1
        return int(expected)

    def _score(self, grid: GridMap | Any, target: ExploreTarget) -> float:
        """Policy score of one candidate (see module docstring for the formula)."""
        config = self.config
        score = config.new_cell_weight * float(target.expected_new_cells)
        if target.kind == "exit":
            score += config.exit_bonus
        if target.path is not None:
            score -= config.path_cost_weight * float(target.path.cost)
        attempts = len(self._attempts.get(target.cell, ()))
        if attempts:
            score -= config.repeat_penalty * min(
                1.0, attempts / max(1, config.max_attempts_per_cell)
            )
        if cell_state(grid, target.cell) is CellState.UNKNOWN:
            score -= config.probe_unknown_cost
        return float(score)

    def _scan(self, grid: GridMap | Any, start: tuple[int, int]) -> _ScanResult:
        """Enumerate + score all frontier candidates; collect skip diagnostics."""
        self._ensure_area(grid)
        start = normalize_cell(start)
        dist, steps, prev = self._region_search(grid, start)
        skip_sets: dict[str, set[tuple[int, int]]] = {
            "edge_blocked": set(),
            "blocked_frontier": set(),
            "unreachable_frontier": set(),
            "deferred_frontier": set(),
            "interactive": set(),
            "exhausted": set(),
            "path_too_long": set(),
        }
        visible_cache: dict[tuple[int, int], set[tuple[int, int]]] = {}
        # target cell -> (sort key, from_cell, direction, kind)
        best: dict[tuple[int, int], tuple[tuple[Any, ...], tuple[int, int], str, str]] = {}
        from_cells = sorted(
            dist, key=lambda cell: (steps[cell], dist[cell], cell[0], cell[1])
        )
        for from_cell in from_cells:
            from_state = cell_state(grid, from_cell)
            if from_cell != start and from_state is not CellState.WALKABLE:
                # EXIT cells trigger an area transition when stepped on: never
                # use them as a probe base inside the current area.
                continue
            path_len = steps[from_cell]
            base_cost = dist[from_cell]
            for index, direction in enumerate(DIRECTIONS):
                dx, dy = DIRECTION_DELTAS[direction]
                target = (from_cell[0] + dx, from_cell[1] + dy)
                if target in dist:
                    continue  # already part of the verified walkable region
                if edge_state(grid, from_cell, target) is EdgeState.BLOCKED:
                    skip_sets["edge_blocked"].add(target)
                    continue
                state = cell_state(grid, target)
                if state is CellState.OBSERVED:
                    kind = "frontier"
                elif state is CellState.UNKNOWN:
                    kind = "frontier"
                elif state is CellState.TEMP_BLOCKED:
                    kind = "frontier"  # dynamic obstacle: retried, attempt-capped
                elif state is CellState.EXIT:
                    kind = "exit"
                elif state is CellState.BLOCKED:
                    skip_sets["blocked_frontier"].add(target)
                    continue
                elif state is CellState.UNREACHABLE:
                    skip_sets["unreachable_frontier"].add(target)
                    continue
                elif state is CellState.DEFERRED:
                    skip_sets["deferred_frontier"].add(target)
                    continue
                else:  # INTERACTIVE or a future state: not autonomous exploration
                    skip_sets["interactive"].add(target)
                    continue
                attempts = len(self._attempts.get(target, ()))
                selections = self._selections.get(target, 0)
                if (
                    attempts >= self.config.max_attempts_per_cell
                    or selections >= self._max_selections
                ):
                    self._exhausted.add(target)
                    skip_sets["exhausted"].add(target)
                    continue
                if path_len > self.config.max_target_path:
                    skip_sets["path_too_long"].add(target)
                    continue
                key = (base_cost, path_len, from_cell, index)
                current = best.get(target)
                if current is None or key < current[0]:
                    best[target] = (key, from_cell, direction, kind)

        candidates: list[ExploreTarget] = []
        for target, (_key, from_cell, direction, kind) in best.items():
            if from_cell not in visible_cache:
                visible_cache[from_cell] = self._visible_cells(from_cell)
            plan = self._plan_to(prev, start, from_cell, dist[from_cell])
            expected = self._expected_new_cells(
                grid,
                from_cell,
                target,
                visible_before=frozenset(visible_cache[from_cell]),
            )
            candidate = ExploreTarget(
                cell=target,
                from_cell=from_cell,
                direction=direction,
                path=plan,
                score=0.0,
                expected_new_cells=expected,
                kind=kind,
                reason="",
            )
            candidate.score = self._score(grid, candidate)
            candidate.reason = (
                f"{kind}: expected_new={candidate.expected_new_cells} "
                f"path={plan.length} attempts={len(self._attempts.get(target, ()))}/"
                f"{self.config.max_attempts_per_cell}"
            )
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                -item.score,
                0 if item.path is None else item.path.length,
                item.cell[0],
                item.cell[1],
            )
        )
        skips = {
            name: len(cells) for name, cells in sorted(skip_sets.items()) if cells
        }
        return _ScanResult(
            candidates=candidates,
            skips=skips,
            dist=dist,
            steps=steps,
            prev=prev,
        )

    # -- internals: completion --------------------------------------------- #

    def _completion_stats(
        self,
        grid: GridMap | Any,
        start: tuple[int, int],
        scan: _ScanResult,
    ) -> dict:
        """Honest coverage/diagnostic counters for :meth:`should_finish`."""
        counts: dict[str, int] = {
            "walkable": 0,
            "observed": 0,
            "blocked": 0,
            "temp_blocked": 0,
            "exits": 0,
            "interactive": 0,
            "unknown": 0,
            "unreachable": 0,
            "deferred": 0,
        }
        cells = getattr(grid, "cells", None)
        keys: Iterable[Any] = list(cells.keys()) if hasattr(cells, "keys") else []
        state_by_cell: dict[tuple[int, int], CellState] = {}
        for key in keys:
            cell = normalize_cell(key)
            state = cell_state(grid, cell)
            state_by_cell[cell] = state
            if state is CellState.WALKABLE:
                counts["walkable"] += 1
            elif state is CellState.OBSERVED:
                counts["observed"] += 1
            elif state is CellState.BLOCKED:
                counts["blocked"] += 1
            elif state is CellState.TEMP_BLOCKED:
                counts["temp_blocked"] += 1
            elif state is CellState.EXIT:
                counts["exits"] += 1
            elif state is CellState.INTERACTIVE:
                counts["interactive"] += 1
            elif state is CellState.UNKNOWN:
                counts["unknown"] += 1
            elif state is CellState.UNREACHABLE:
                counts["unreachable"] += 1
            elif state is CellState.DEFERRED:
                counts["deferred"] += 1

        unknown_not_bordering = 0
        observed_not_bordering = 0
        for cell, state in state_by_cell.items():
            if state not in (CellState.UNKNOWN, CellState.OBSERVED):
                continue
            bordering = False
            for direction in DIRECTIONS:
                dx, dy = DIRECTION_DELTAS[direction]
                if (cell[0] + dx, cell[1] + dy) in scan.dist:
                    bordering = True
                    break
            if bordering:
                continue
            if state is CellState.UNKNOWN:
                unknown_not_bordering += 1
            else:
                observed_not_bordering += 1

        reachable_walkable = sum(
            1
            for cell in scan.dist
            if cell_state(grid, cell) is CellState.WALKABLE
        )
        total_attempts = sum(len(items) for items in self._attempts.values())
        best = scan.candidates[0] if scan.candidates else None
        stats: dict[str, Any] = {
            "area_id": str(getattr(grid, "area_id", "")),
            "start": list(normalize_cell(start)),
            "cells_total": len(state_by_cell),
            "walkable": counts["walkable"],
            "reachable_walkable": reachable_walkable,
            "observed": counts["observed"],
            "blocked": counts["blocked"],
            "temp_blocked": counts["temp_blocked"],
            "exits": counts["exits"],
            "interactive": counts["interactive"],
            "unknown": counts["unknown"],
            "unreachable": counts["unreachable"],
            "deferred": counts["deferred"],
            "unknown_not_bordering": unknown_not_bordering,
            "observed_not_bordering": observed_not_bordering,
            "frontier_candidates": len(scan.candidates),
            "best_cell": None if best is None else list(best.cell),
            "best_score": None if best is None else round(float(best.score), 3),
            "skipped": dict(scan.skips),
            "attempts": int(total_attempts),
            "consecutive_no_progress": int(self._consecutive_no_progress),
            "exhausted_cells": len(self._exhausted),
            "finish_reason": "",
        }
        return stats

    def _finished_reason(self, stats: dict) -> str:
        """Human-readable completion reason built from honest counters."""
        parts: list[str] = []
        if stats["walkable"] == 0:
            parts.append("no walkable cell recorded yet")
        skipped = stats.get("skipped") or {}
        if skipped:
            parts.append(
                "skipped frontier: "
                + ", ".join(f"{name}={count}" for name, count in sorted(skipped.items()))
            )
        if stats["unknown_not_bordering"] or stats["observed_not_bordering"]:
            parts.append(
                "cells not bordering the reachable region: "
                f"unknown={stats['unknown_not_bordering']}, "
                f"observed={stats['observed_not_bordering']}"
            )
        parts.append(
            "no reachable valuable frontier left "
            "(walkable={walkable} blocked={blocked} temp_blocked={temp_blocked} "
            "unreachable={unreachable} deferred={deferred} unknown={unknown})".format(**stats)
        )
        return "; ".join(parts)
