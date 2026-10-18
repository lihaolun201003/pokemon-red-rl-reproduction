"""Per-area logical grid map for Navigation v2 (owner: nav-map).

One :class:`GridMap` represents one area (all cells are area-local coordinates).
It is built **only** from :class:`~llm_agent.navigation.types.FrameView`
observations and verified movement outcomes — never from RAM or map data.

Design rules that this module enforces (see ``NAV_CONTRACT.md`` §3):

* **Observation is not verification.**  Seeing a tile only creates/refreshes an
  :attr:`~llm_agent.navigation.types.CellState.OBSERVED` cell.  Only a verified
  move (``confirm_move``) creates :attr:`~llm_agent.navigation.types.CellState.WALKABLE`.
* **One bump never permanently walls off a known-good cell.**  If a cell was
  verified walkable before, ``confirm_blocked`` keeps it WALKABLE and only marks
  the traversed *edge* TEMP_BLOCKED (dynamic NPCs / transient objects).  Such a
  cell is retried later by the exploration layer.
* **A turn is not a bump.**  In Gen-1 the first press of a new direction only
  turns the player.  ``confirm_turn`` followed by ``confirm_blocked`` for the same
  cell/direction is therefore treated as a turn artifact: evidence counters are
  updated but no cell/edge state is downgraded (otherwise every planned direction
  change would wall off the cell ahead, see ``SUPPRESS_TURN_ARTIFACT``).
* **The player's own cell is verified walkable by the pose.**  ``observe_view``
  promotes ``(pose.x, pose.y)`` from UNKNOWN/OBSERVED to WALKABLE when the pose is
  valid: the localizer's verified pose proves the player is standing there, so
  the very first exploration step has a walkable seed cell.  All *other* visible
  tiles stay merely OBSERVED.
* **Honest coverage.**  ``coverage_ratio = walkable / (walkable + unknown_frontier)``
  where ``unknown_frontier`` counts unknown/observed cells orthogonally adjacent
  to the reachable walkable region.  Unreachable/deferred cells are reported
  separately and never counted as explored.
* **Persistence** is versioned JSON with an atomic write; loading is tolerant of
  unknown extra keys and skips individually malformed cells/edges, but raises
  :class:`ValueError` for a structurally corrupt file (the caller handles it).
  No absolute paths are ever stored.

All public methods are deterministic and JSON-serializable input/output only.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_agent.navigation.types import (
    DIRECTION_DELTAS,
    DIRECTIONS,
    VIEW_H,
    VIEW_PLAYER,
    VIEW_W,
    CellState,
    CoverageStats,
    EdgeState,
    FrameView,
    PlayerPose,
)

__all__ = [
    "GRID_MAP_FORMAT",
    "GRID_MAP_FORMAT_VERSION",
    "MAX_LABELS",
    "MAX_TILE_VARIANTS",
    "Cell",
    "Edge",
    "Frontier",
    "GridMap",
    "save_json_atomic",
]

#: Persistence format identifier / version written into every grid JSON file.
GRID_MAP_FORMAT = "nav.grid_map"
GRID_MAP_FORMAT_VERSION = 1

#: Bounded number of per-cell tile looks kept for re-identification (animation
#: variants).  Keeps area files small and matching cheap.
MAX_TILE_VARIANTS = 8
#: Bounded number of free-form labels kept per cell.
MAX_LABELS = 12

#: Whether a valid pose may promote the player's own cell to WALKABLE
#: (see the module docstring).  Kept as a module flag so it can be disabled if a
#: integration ever needs the strict "only confirm_move creates WALKABLE" rule.
PROMOTE_POSE_CELL = True

#: Whether ``confirm_blocked`` immediately after ``confirm_turn`` for the same
#: ``(cell, direction)`` is treated as the Gen-1 "first press only turns" artifact
#: (obstacle evidence only, no state/edge downgrade).  See :meth:`GridMap.confirm_turn`.
SUPPRESS_TURN_ARTIFACT = True

#: Default confidence per state when no explicit value is supplied.
_STATE_CONFIDENCE: dict[CellState, float] = {
    CellState.UNKNOWN: 0.0,
    CellState.OBSERVED: 0.3,
    CellState.WALKABLE: 1.0,
    CellState.BLOCKED: 0.7,
    CellState.TEMP_BLOCKED: 0.5,
    CellState.EXIT: 0.8,
    CellState.INTERACTIVE: 0.7,
    CellState.UNREACHABLE: 0.6,
    CellState.DEFERRED: 0.5,
}

#: ASCII symbol per state (see :meth:`GridMap.render_ascii`).
_SYMBOLS: dict[CellState, str] = {
    CellState.UNKNOWN: "?",
    CellState.OBSERVED: ":",
    CellState.WALKABLE: ".",
    CellState.BLOCKED: "#",
    CellState.TEMP_BLOCKED: "x",
    CellState.EXIT: "E",
    CellState.INTERACTIVE: "I",
    CellState.UNREACHABLE: "X",
    CellState.DEFERRED: "D",
}

_LEGEND_LINES = (
    "legend: @=center T=target *=path .=walkable :=observed #=blocked x=temp_blocked",
    "        E=exit I=interactive X=unreachable D=deferred ?=unknown",
)


# --------------------------------------------------------------------------- #
# Small coercion helpers (tolerant JSON readers, strict internal setters)
# --------------------------------------------------------------------------- #


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion used by tolerant JSON readers."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion used by tolerant JSON readers."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if result != result:  # NaN
        return float(default)
    return result


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


def _as_cell(value: Any) -> tuple[int, int] | None:
    """Coerce ``(x, y)`` pairs (tuple/list/numpy row) to an int tuple, else ``None``."""
    if isinstance(value, (str, bytes)):
        return None
    try:
        if len(value) != 2:
            return None
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _as_cell_strict(value: Any) -> tuple[int, int]:
    cell = _as_cell(value)
    if cell is None:
        raise ValueError(f"expected an (x, y) cell coordinate, got {value!r}")
    return cell


def _coerce_cell_state(value: Any, *, strict: bool = False) -> CellState:
    if isinstance(value, CellState):
        return value
    if isinstance(value, str):
        try:
            return CellState(value.strip().lower())
        except ValueError:
            if strict:
                raise ValueError(f"invalid CellState: {value!r}") from None
            return CellState.UNKNOWN
    if strict:
        raise ValueError(f"invalid CellState: {value!r}")
    return CellState.UNKNOWN


def _coerce_edge_state(value: Any, *, strict: bool = False) -> EdgeState:
    if isinstance(value, EdgeState):
        return value
    if isinstance(value, str):
        try:
            return EdgeState(value.strip().lower())
        except ValueError:
            if strict:
                raise ValueError(f"invalid EdgeState: {value!r}") from None
            return EdgeState.UNKNOWN
    if strict:
        raise ValueError(f"invalid EdgeState: {value!r}")
    return EdgeState.UNKNOWN


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    """Direction of the unit step ``a -> b`` or ``""`` when not adjacent."""
    delta = (int(b[0]) - int(a[0]), int(b[1]) - int(a[1]))
    for direction, step in DIRECTION_DELTAS.items():
        if step == delta:
            return direction
    return ""


def _canonical_pair(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Canonical (lexicographic) endpoint order for an undirected edge key."""
    return (a, b) if a <= b else (b, a)


def _dominant_hash(tile_hashes: Sequence[str]) -> str:
    """Most frequent hash in ``tile_hashes``; ties go to the earliest variant."""
    if not tile_hashes:
        return ""
    counts = Counter(tile_hashes)
    # ``max`` keeps the first maximal element in iteration order, and Counter
    # preserves first-insertion order -> deterministic, earliest-wins tie-break.
    return max(counts, key=lambda item: counts[item])


def _append_label(labels: list[str], label: str) -> None:
    text = str(label).strip()
    if not text or text in labels or len(labels) >= MAX_LABELS:
        return
    labels.append(text)


def save_json_atomic(path: str | Path, payload: Any) -> None:
    """Write ``payload`` as compact deterministic JSON via a tmp-file rename.

    The temporary file lives in the destination directory, is flushed and
    fsync'ed, then :func:`os.replace`d over the target, so a crash never leaves a
    half-written JSON document.  A failed write cleans the temporary file up.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".tmp", dir=str(target.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# Data records
# --------------------------------------------------------------------------- #


@dataclass
class Cell:
    """Knowledge about one area-local grid cell.

    ``state`` is the *strongest current belief*; ``tile_hashes`` are the observed
    looks (surfaced via :func:`llm_agent.navigation.vision.tile_hash`), used by
    world-model relocalization, and ``terrain`` is their dominant variant.
    Evidence counters are append-only so contradiction handling stays honest.
    """

    x: int
    y: int
    state: CellState = CellState.UNKNOWN
    terrain: str = ""
    tile_hashes: list[str] = field(default_factory=list)
    observed_count: int = 0
    visit_count: int = 0
    confidence: float = 0.0
    last_seen_frame: int = -1
    obstacle_evidence: int = 0
    free_evidence: int = 0
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x = int(self.x)
        self.y = int(self.y)
        self.state = _coerce_cell_state(self.state)
        self.terrain = str(self.terrain or "")
        self.tile_hashes = [str(item) for item in (self.tile_hashes or []) if str(item)]
        self.observed_count = max(0, _as_int(self.observed_count, 0))
        self.visit_count = max(0, _as_int(self.visit_count, 0))
        self.confidence = _clamp01(self.confidence)
        self.last_seen_frame = _as_int(self.last_seen_frame, -1)
        self.obstacle_evidence = max(0, _as_int(self.obstacle_evidence, 0))
        self.free_evidence = max(0, _as_int(self.free_evidence, 0))
        self.labels = [str(item) for item in (self.labels or []) if str(item)][:MAX_LABELS]

    # -- conveniences ------------------------------------------------------ #

    @property
    def cell(self) -> tuple[int, int]:
        """``(x, y)`` coordinate of this cell."""
        return (self.x, self.y)

    @property
    def known(self) -> bool:
        """True when the cell has been observed at least once (state != UNKNOWN)."""
        return self.state is not CellState.UNKNOWN

    def add_hash(self, tile_hash: str) -> bool:
        """Record one observed tile look; returns True when it is a new variant.

        The stored list is bounded to :data:`MAX_TILE_VARIANTS` (oldest variant is
        dropped first) and ``terrain`` is recomputed from the remaining variants.
        """
        value = str(tile_hash or "")
        if not value:
            return False
        if value in self.tile_hashes:
            return False
        self.tile_hashes.append(value)
        if len(self.tile_hashes) > MAX_TILE_VARIANTS:
            del self.tile_hashes[0]
        self.terrain = _dominant_hash(self.tile_hashes)
        return True

    # -- persistence ------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the cell."""
        return {
            "x": int(self.x),
            "y": int(self.y),
            "state": self.state.value,
            "terrain": self.terrain,
            "tile_hashes": list(self.tile_hashes),
            "observed_count": int(self.observed_count),
            "visit_count": int(self.visit_count),
            "confidence": round(float(self.confidence), 4),
            "last_seen_frame": int(self.last_seen_frame),
            "obstacle_evidence": int(self.obstacle_evidence),
            "free_evidence": int(self.free_evidence),
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Cell":
        """Tolerant inverse of :meth:`to_dict` (extra keys ignored)."""
        if not isinstance(data, Mapping):
            raise ValueError(f"cell entry must be a mapping, got {type(data)!r}")
        try:
            x = int(data["x"])
            y = int(data["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cell entry needs integer x/y: {exc}") from exc
        hashes_raw = data.get("tile_hashes") or []
        if isinstance(hashes_raw, str):
            hashes = [hashes_raw]
        elif isinstance(hashes_raw, Sequence):
            hashes = [str(item) for item in hashes_raw if str(item)]
        else:
            hashes = []
        labels_raw = data.get("labels") or []
        if isinstance(labels_raw, str):
            labels = [labels_raw]
        elif isinstance(labels_raw, Sequence):
            labels = [str(item) for item in labels_raw if str(item)]
        else:
            labels = []
        return cls(
            x=x,
            y=y,
            state=_coerce_cell_state(data.get("state")),
            terrain=str(data.get("terrain") or ""),
            tile_hashes=hashes,
            observed_count=_as_int(data.get("observed_count", 0), 0),
            visit_count=_as_int(data.get("visit_count", 0), 0),
            confidence=_clamp01(data.get("confidence", 0.0)),
            last_seen_frame=_as_int(data.get("last_seen_frame", -1), -1),
            obstacle_evidence=_as_int(data.get("obstacle_evidence", 0), 0),
            free_evidence=_as_int(data.get("free_evidence", 0), 0),
            labels=labels,
        )


@dataclass
class Edge:
    """Passability relation between two orthogonally adjacent cells.

    Edges are stored in canonical endpoint order (``a <= b`` lexicographically)
    and ``direction`` is the step direction from ``a`` to ``b``.  The relation is
    undirected on purpose: a verified step ``a -> b`` is also the best evidence
    that ``b -> a`` is traversable in this top-down map.
    """

    a: tuple[int, int]
    b: tuple[int, int]
    direction: str
    state: EdgeState = EdgeState.UNKNOWN
    attempts: int = 0
    successes: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        first = _as_cell_strict(self.a)
        second = _as_cell_strict(self.b)
        self.a = first
        self.b = second
        self.direction = str(self.direction or "")
        if self.direction not in DIRECTIONS:
            self.direction = ""
        self.state = _coerce_edge_state(self.state)
        self.attempts = max(0, _as_int(self.attempts, 0))
        self.successes = max(0, _as_int(self.successes, 0))
        if self.attempts and self.successes > self.attempts:
            self.successes = self.attempts
        self.confidence = _clamp01(self.confidence)

    @property
    def key(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Canonical ``(a, b)`` dictionary key for this edge."""
        return _canonical_pair(self.a, self.b)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the edge."""
        return {
            "a": [int(self.a[0]), int(self.a[1])],
            "b": [int(self.b[0]), int(self.b[1])],
            "direction": self.direction,
            "state": self.state.value,
            "attempts": int(self.attempts),
            "successes": int(self.successes),
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edge":
        """Tolerant inverse of :meth:`to_dict`; endpoints are canonicalized."""
        if not isinstance(data, Mapping):
            raise ValueError(f"edge entry must be a mapping, got {type(data)!r}")
        a = _as_cell(data.get("a"))
        b = _as_cell(data.get("b"))
        if a is None or b is None:
            raise ValueError("edge entry needs two (x, y) endpoints")
        direction = "" if a > b else str(data.get("direction") or "")
        first, second = _canonical_pair(a, b)
        if direction not in DIRECTIONS:
            direction = _direction_between(first, second)
        return cls(
            a=first,
            b=second,
            direction=direction,
            state=_coerce_edge_state(data.get("state")),
            attempts=_as_int(data.get("attempts", 0), 0),
            successes=_as_int(data.get("successes", 0), 0),
            confidence=_clamp01(data.get("confidence", 0.0)),
        )


@dataclass
class Frontier:
    """One exploration candidate: a non-walkable cell next to the reached region."""

    cell: tuple[int, int]
    from_cell: tuple[int, int]
    direction: str
    expected_new: int = 0

    def __post_init__(self) -> None:
        self.cell = _as_cell_strict(self.cell)
        self.from_cell = _as_cell_strict(self.from_cell)
        self.direction = str(self.direction or "")
        self.expected_new = max(0, _as_int(self.expected_new, 0))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the frontier entry."""
        return {
            "cell": [int(self.cell[0]), int(self.cell[1])],
            "from_cell": [int(self.from_cell[0]), int(self.from_cell[1])],
            "direction": self.direction,
            "expected_new": int(self.expected_new),
        }


# --------------------------------------------------------------------------- #
# GridMap
# --------------------------------------------------------------------------- #


class GridMap:
    """One logical 2D map of one area, built from observations + verified moves.

    The map is unbounded (cells are created lazily); ``bounds()`` reports the
    extent of currently known cells.  See the module docstring for the semantic
    rules; the public API is frozen by ``NAV_CONTRACT.md`` §3.
    """

    FORMAT = GRID_MAP_FORMAT
    VERSION = GRID_MAP_FORMAT_VERSION

    def __init__(self, area_id: str, *, frame_index: int = 0, kind: str = "unknown") -> None:
        self.area_id: str = str(area_id or "")
        self.created_frame: int = _as_int(frame_index, 0)
        self.kind: str = kind if kind in ("interior", "exterior", "unknown") else "unknown"
        self.cells: dict[tuple[int, int], Cell] = {}
        self.edges: dict[tuple[tuple[int, int], tuple[int, int]], Edge] = {}
        #: Non-fatal problems collected by a tolerant :meth:`from_dict`/:meth:`load`.
        self.load_warnings: list[str] = []
        #: Last frame index that touched this map (bookkeeping only).
        self.updated_frame: int = _as_int(frame_index, 0)
        #: A turn-only input waiting for the follow-up map call (Gen-1 "first press
        #: turns" artifact; see :meth:`confirm_turn` / :data:`SUPPRESS_TURN_ARTIFACT`).
        self._pending_turn: tuple[tuple[int, int], str] | None = None

    # -- basic access ------------------------------------------------------ #

    @staticmethod
    def _key(x: int, y: int) -> tuple[int, int]:
        return (int(x), int(y))

    @staticmethod
    def _key_of(cell: tuple[int, int] | Sequence[int]) -> tuple[int, int]:
        return _as_cell_strict(cell)

    def cell(self, x: int, y: int) -> Cell | None:
        """Return the stored cell at ``(x, y)`` or ``None`` when never created."""
        return self.cells.get(self._key(x, y))

    def get_cell(self, x: int, y: int) -> Cell:
        """Return the cell at ``(x, y)``, creating an UNKNOWN one if needed."""
        key = self._key(x, y)
        existing = self.cells.get(key)
        if existing is not None:
            return existing
        created = Cell(x=key[0], y=key[1])
        self.cells[key] = created
        return created

    def cells_in_rect(self, x0: int, y0: int, x1: int, y1: int) -> list[Cell]:
        """All known cells inside the inclusive rectangle (corners may be swapped)."""
        lo_x, hi_x = sorted((int(x0), int(x1)))
        lo_y, hi_y = sorted((int(y0), int(y1)))
        found = [
            item
            for (cx, cy), item in self.cells.items()
            if lo_x <= cx <= hi_x and lo_y <= cy <= hi_y
        ]
        found.sort(key=lambda item: (item.y, item.x))
        return found

    # -- edges ------------------------------------------------------------- #

    def edge(self, a: tuple[int, int], b: tuple[int, int]) -> Edge | None:
        """Return the stored edge between ``a`` and ``b`` (either argument order)."""
        first = self._key_of(a)
        second = self._key_of(b)
        if first == second:
            return None
        return self.edges.get(_canonical_pair(first, second))

    def get_edge(self, a: tuple[int, int], b: tuple[int, int]) -> Edge:
        """Return the edge between ``a`` and ``b``, creating an UNKNOWN one if needed."""
        first = self._key_of(a)
        second = self._key_of(b)
        if first == second:
            raise ValueError("an edge needs two distinct cells")
        key = _canonical_pair(first, second)
        existing = self.edges.get(key)
        if existing is not None:
            return existing
        created = Edge(a=key[0], b=key[1], direction=_direction_between(key[0], key[1]))
        self.edges[key] = created
        return created

    def edge_state(self, a: tuple[int, int], b: tuple[int, int]) -> EdgeState:
        """Edge state between two cells; UNKNOWN when no edge was recorded."""
        stored = self.edge(a, b)
        return stored.state if stored is not None else EdgeState.UNKNOWN

    # -- neighbours -------------------------------------------------------- #

    def neighbours(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """The four orthogonally adjacent coordinates (``up, down, left, right``)."""
        x, y = self._key_of(cell)
        result: list[tuple[int, int]] = []
        for direction in DIRECTIONS:
            dx, dy = DIRECTION_DELTAS[direction]
            result.append((x + dx, y + dy))
        return result

    def walkable_neighbours(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Neighbours whose cell state is verified WALKABLE."""
        result: list[tuple[int, int]] = []
        for neighbour in self.neighbours(cell):
            known = self.cells.get(neighbour)
            if known is not None and known.state is CellState.WALKABLE:
                result.append(neighbour)
        return result

    def traversable_neighbours(
        self,
        cell: tuple[int, int],
        *,
        allow_observed: bool = True,
        allow_temp_blocked: bool = False,
        allow_exit: bool = True,
    ) -> list[tuple[int, int]]:
        """Neighbours a planner may consider stepping into.

        WALKABLE cells are always allowed.  OBSERVED cells are included when
        ``allow_observed`` (they are seen ground, not yet verified).  EXIT cells are
        included by default because they are the passages to other areas.  A
        neighbour whose *edge* is BLOCKED is never returned; TEMP_BLOCKED edges are
        skipped unless ``allow_temp_blocked`` (dynamic obstruction — retry later).
        """
        allowed: set[CellState] = {CellState.WALKABLE}
        if allow_observed:
            allowed.add(CellState.OBSERVED)
        if allow_temp_blocked:
            allowed.add(CellState.TEMP_BLOCKED)
        if allow_exit:
            allowed.add(CellState.EXIT)
        result: list[tuple[int, int]] = []
        for neighbour in self.neighbours(cell):
            known = self.cells.get(neighbour)
            if known is None or known.state not in allowed:
                continue
            stored_edge = self.edge(cell, neighbour)
            if stored_edge is not None:
                if stored_edge.state is EdgeState.BLOCKED:
                    continue
                if stored_edge.state is EdgeState.TEMP_BLOCKED and not allow_temp_blocked:
                    continue
            result.append(neighbour)
        return result

    # -- observation ------------------------------------------------------- #

    def observe_view(
        self,
        view: FrameView,
        pose: PlayerPose,
        *,
        frame_index: int = 0,
    ) -> dict[str, int]:
        """Merge one visible 10x9 tile view into the map.

        Every visible tile becomes a known cell (UNKNOWN -> OBSERVED, never
        downgrading any verified state), its tile look is recorded as a re-ID
        variant, and observation/frame counters are refreshed.  The player's own
        cell is additionally promoted to WALKABLE when the pose is valid (see the
        module docstring).

        Returns counters ``{"new_cells", "new_hashes", "observed"}``.
        """
        counters = {"new_cells": 0, "new_hashes": 0, "observed": 0}
        if view is None or not getattr(view, "tiles", None):
            return counters
        frame = _as_int(frame_index, 0)
        pose_x = _as_int(getattr(pose, "x", 0), 0)
        pose_y = _as_int(getattr(pose, "y", 0), 0)
        offset_x = VIEW_PLAYER[0]
        offset_y = VIEW_PLAYER[1]
        for tile in view.tiles:
            world_x = pose_x + _as_int(getattr(tile, "tx", 0), 0) - offset_x
            world_y = pose_y + _as_int(getattr(tile, "ty", 0), 0) - offset_y
            key = (world_x, world_y)
            item = self.cells.get(key)
            if item is None:
                item = Cell(x=world_x, y=world_y)
                self.cells[key] = item
                counters["new_cells"] += 1
            if item.add_hash(getattr(tile, "tile_hash", "")):
                counters["new_hashes"] += 1
            item.observed_count += 1
            if frame > item.last_seen_frame:
                item.last_seen_frame = frame
            if item.state is CellState.UNKNOWN:
                item.state = CellState.OBSERVED
                item.confidence = max(item.confidence, _STATE_CONFIDENCE[CellState.OBSERVED])
            counters["observed"] += 1
        if frame > self.updated_frame:
            self.updated_frame = frame
        if PROMOTE_POSE_CELL and bool(getattr(pose, "valid", True)):
            # The verified pose proves the player stands on this cell.  This is the
            # only observation-based path to WALKABLE and it is essential: the
            # explorer needs a reachable seed cell in a freshly entered area.
            player = self.cells.get((pose_x, pose_y))
            if player is None:
                player = self.get_cell(pose_x, pose_y)
            if player.state in (CellState.UNKNOWN, CellState.OBSERVED):
                player.state = CellState.WALKABLE
                player.free_evidence = max(1, player.free_evidence)
                player.confidence = max(player.confidence, _STATE_CONFIDENCE[CellState.WALKABLE])
                if frame > player.last_seen_frame:
                    player.last_seen_frame = frame
        return counters

    # -- verified movement ------------------------------------------------- #

    def _promote_occupied(self, key: tuple[int, int], *, frame_index: int = 0) -> Cell:
        """Mark a cell the player verifiably occupied as walkable (if unverified)."""
        item = self.get_cell(key[0], key[1])
        if item.state in (CellState.UNKNOWN, CellState.OBSERVED):
            item.state = CellState.WALKABLE
            item.free_evidence = max(1, item.free_evidence)
            item.confidence = max(item.confidence, _STATE_CONFIDENCE[CellState.WALKABLE])
        if frame_index > item.last_seen_frame:
            item.last_seen_frame = _as_int(frame_index, 0)
        return item

    @staticmethod
    def _record_edge_traversal(edge: Edge, state: EdgeState, *, success: bool) -> None:
        edge.state = state
        edge.attempts += 1
        if success:
            edge.successes += 1
        edge.confidence = _clamp01(edge.successes / edge.attempts) if edge.attempts else 0.0

    def confirm_move(
        self,
        from_cell: tuple[int, int],
        direction: str,
        *,
        tiles: int = 1,
        frame_index: int = 0,
    ) -> tuple[int, int]:
        """Record a verified move from ``from_cell`` in ``direction``.

        Every target cell becomes (or stays) WALKABLE with ``visit_count`` and
        ``free_evidence`` incremented; intermediate cells of a multi-tile move are
        verified too.  Traversed edges become OPEN.  A cell that was previously
        blocked is re-promoted to WALKABLE — direct movement evidence beats a bump
        (this is the "never permanently wall off a walked cell" rule).  Returns the
        final cell.
        """
        if direction not in DIRECTION_DELTAS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        steps = max(1, _as_int(tiles, 1))
        frame = _as_int(frame_index, 0)
        current = self._key_of(from_cell)
        delta = DIRECTION_DELTAS[direction]
        self._pending_turn = None  # a verified move supersedes any turn artifact
        self._promote_occupied(current, frame_index=frame)
        for _ in range(steps):
            target = (current[0] + delta[0], current[1] + delta[1])
            item = self.get_cell(*target)
            # A verified walk is the strongest passability evidence there is.
            item.state = CellState.WALKABLE
            item.visit_count += 1
            item.free_evidence += 1
            item.confidence = max(item.confidence, _STATE_CONFIDENCE[CellState.WALKABLE])
            if frame > item.last_seen_frame:
                item.last_seen_frame = frame
            stored_edge = self.get_edge(current, target)
            self._record_edge_traversal(stored_edge, EdgeState.OPEN, success=True)
            current = target
        if frame > self.updated_frame:
            self.updated_frame = frame
        return current

    def confirm_blocked(
        self,
        from_cell: tuple[int, int],
        direction: str,
        *,
        temporary: bool = False,
        frame_index: int = 0,
    ) -> tuple[int, int]:
        """Record a verified bump from ``from_cell`` towards ``direction``.

        ``temporary=False`` means "believed persistent" (a wall-like obstacle):
        the target cell becomes BLOCKED.  ``temporary=True`` records a dynamic
        obstruction (NPC/object): the target cell becomes TEMP_BLOCKED and the
        explorer retries it later.  **Contradiction rule:** a cell that was
        verifiably walkable before (WALKABLE / visited / EXIT) is never turned into
        a wall by a bump — it keeps its state and only the traversed edge becomes
        TEMP_BLOCKED.  Returns the bumped target cell.

        When the immediately preceding call was :meth:`confirm_turn` for the same
        ``(from_cell, direction)``, this bump is the Gen-1 "first press only turns"
        artifact: obstacle/edge counters are updated (that is the requested blocked
        evidence) but no cell or edge state is downgraded.  Without that rule every
        planned direction change would permanently wall off the cell ahead of the
        player.
        """
        if direction not in DIRECTION_DELTAS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        frame = _as_int(frame_index, 0)
        current = self._key_of(from_cell)
        pending_turn = self._pending_turn
        self._pending_turn = None
        delta = DIRECTION_DELTAS[direction]
        target = (current[0] + delta[0], current[1] + delta[1])
        self._promote_occupied(current, frame_index=frame)
        item = self.get_cell(*target)
        item.obstacle_evidence += 1
        if frame > item.last_seen_frame:
            item.last_seen_frame = frame
        stored_edge = self.get_edge(current, target)

        if SUPPRESS_TURN_ARTIFACT and pending_turn == (current, direction):
            # Evidence only: the press turned the player instead of walking.  The
            # cell/edge keep whatever passability knowledge they had.
            stored_edge.attempts += 1
            stored_edge.confidence = (
                _clamp01(stored_edge.successes / stored_edge.attempts)
                if stored_edge.attempts
                else 0.0
            )
            if frame > self.updated_frame:
                self.updated_frame = frame
            return target

        was_walked = (
            item.state in (CellState.WALKABLE, CellState.EXIT) or item.visit_count > 0
        )
        if was_walked:
            # Never permanently wall off a cell we have walked through; the bump is
            # a dynamic/transient obstruction (NPC, moving object, stale read).
            if item.state is not CellState.WALKABLE:
                item.state = CellState.WALKABLE
            item.confidence = max(item.confidence, _STATE_CONFIDENCE[CellState.WALKABLE])
            _append_label(item.labels, "temporary_obstruction")
            self._record_edge_traversal(stored_edge, EdgeState.TEMP_BLOCKED, success=False)
            if frame > self.updated_frame:
                self.updated_frame = frame
            return target

        if temporary:
            if item.state is not CellState.BLOCKED:
                item.state = CellState.TEMP_BLOCKED
                item.confidence = max(
                    item.confidence, min(0.9, 0.35 + 0.1 * item.obstacle_evidence)
                )
            self._record_edge_traversal(stored_edge, EdgeState.TEMP_BLOCKED, success=False)
        else:
            item.state = CellState.BLOCKED
            item.confidence = max(item.confidence, min(1.0, 0.5 + 0.2 * item.obstacle_evidence))
            self._record_edge_traversal(stored_edge, EdgeState.BLOCKED, success=False)
        if frame > self.updated_frame:
            self.updated_frame = frame
        return target

    def confirm_turn(self, from_cell: tuple[int, int], direction: str) -> None:
        """Record a turn-only input (no displacement, facing changed).

        In Gen-1 overworld movement the *first* press of a new direction only
        turns the player, so a turn is **not** passability evidence for the target
        cell; it does however prove the player occupied ``from_cell``.  The turn is
        remembered so that an immediately following ``confirm_blocked`` for the same
        ``(from_cell, direction)`` is treated as evidence-only (see
        :data:`SUPPRESS_TURN_ARTIFACT`) instead of permanently blocking walkable
        ground.
        """
        if direction not in DIRECTION_DELTAS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        key = self._key_of(from_cell)
        self._promote_occupied(key)
        self._pending_turn = (key, direction)

    # -- explicit landmarks ------------------------------------------------ #

    def mark_exit(
        self,
        cell: tuple[int, int],
        *,
        direction: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Flag a cell (and optionally the outgoing edge) as an area exit."""
        key = self._key_of(cell)
        item = self.get_cell(*key)
        item.state = CellState.EXIT
        item.confidence = max(item.confidence, _clamp01(confidence, 1.0))
        _append_label(item.labels, "exit")
        if direction is not None:
            if direction not in DIRECTION_DELTAS:
                raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
            delta = DIRECTION_DELTAS[direction]
            target = (key[0] + delta[0], key[1] + delta[1])
            stored_edge = self.get_edge(key, target)
            stored_edge.state = EdgeState.EXIT
            stored_edge.confidence = max(stored_edge.confidence, _clamp01(confidence, 1.0))

    def mark_interactive(self, cell: tuple[int, int], label: str = "") -> None:
        """Flag a cell as probably interactive (sign/NPC/item).

        A cell already verified WALKABLE keeps WALKABLE (interactable floor tiles
        exist) and only gets the label, so verified passability is never lost.
        """
        key = self._key_of(cell)
        item = self.get_cell(*key)
        _append_label(item.labels, label or "interactive")
        if item.state is not CellState.WALKABLE:
            item.state = CellState.INTERACTIVE
            item.confidence = max(item.confidence, _STATE_CONFIDENCE[CellState.INTERACTIVE])

    def mark_state(
        self,
        cell: tuple[int, int],
        state: CellState,
        *,
        confidence: float | None = None,
        reason: str = "",
    ) -> None:
        """Explicitly set one cell's state (authoritative mark, e.g. area entry)."""
        key = self._key_of(cell)
        coerced = _coerce_cell_state(state, strict=True)
        item = self.get_cell(*key)
        item.state = coerced
        if confidence is not None:
            item.confidence = _clamp01(confidence)
        else:
            item.confidence = max(item.confidence, _STATE_CONFIDENCE[coerced])
        _append_label(item.labels, reason)

    def mark_unreachable(self, cells: Iterable[tuple[int, int]], reason: str = "") -> None:
        """Report cells that cannot be reached from the explored region.

        Walkable cells are skipped (direct contradiction): the map never forgets
        that a cell was walked.  These cells are reported separately in coverage
        and never counted as explored.
        """
        for reference in cells:
            key = self._key_of(reference)
            item = self.get_cell(*key)
            if item.state is CellState.WALKABLE:
                continue
            item.state = CellState.UNREACHABLE
            item.confidence = max(item.confidence, _STATE_CONFIDENCE[CellState.UNREACHABLE])
            _append_label(item.labels, reason)

    def mark_deferred(self, cells: Iterable[tuple[int, int]], reason: str = "") -> None:
        """Report cells reachable in principle but deliberately skipped."""
        for reference in cells:
            key = self._key_of(reference)
            item = self.get_cell(*key)
            if item.state is CellState.WALKABLE:
                continue
            item.state = CellState.DEFERRED
            item.confidence = max(item.confidence, _STATE_CONFIDENCE[CellState.DEFERRED])
            _append_label(item.labels, reason)

    # -- reachability / exploration geometry ------------------------------- #

    def reachable_region(
        self,
        start: tuple[int, int],
        *,
        passable: set[CellState] | None = None,
    ) -> set[tuple[int, int]]:
        """Flood-fill from ``start`` through cells whose state is in ``passable``.

        ``start`` is always included (the player stands there, so it is reachable
        even if the caller marked it merely OBSERVED).  Default passable states are
        WALKABLE and EXIT.
        """
        allowed = set(passable) if passable is not None else {CellState.WALKABLE, CellState.EXIT}
        start_key = self._key_of(start)
        region: set[tuple[int, int]] = {start_key}
        stack: list[tuple[int, int]] = [start_key]
        while stack:
            current = stack.pop()
            for neighbour in self.neighbours(current):
                if neighbour in region:
                    continue
                known = self.cells.get(neighbour)
                if known is None or known.state not in allowed:
                    continue
                region.add(neighbour)
                stack.append(neighbour)
        return region

    def frontier(
        self,
        start: tuple[int, int],
        *,
        reachable: set[tuple[int, int]] | None = None,
    ) -> list[Frontier]:
        """Exploration candidates around the reachable region, deterministic order.

        A candidate is an unknown or merely-observed cell orthogonally adjacent to
        a reachable **walkable** cell (the start cell also qualifies if the caller
        has not marked it walkable yet).  ``expected_new`` counts how many still
        unknown cells the 10x9 view centred on the candidate would cover.
        """
        start_key = self._key_of(start)
        region = reachable if reachable is not None else self.reachable_region(start_key)
        candidates: dict[tuple[int, int], Frontier] = {}
        for current in sorted(region, key=lambda item: (item[1], item[0])):
            known = self.cells.get(current)
            is_seed = current == start_key
            if not is_seed and (known is None or known.state is not CellState.WALKABLE):
                continue
            for direction in DIRECTIONS:
                delta = DIRECTION_DELTAS[direction]
                candidate = (current[0] + delta[0], current[1] + delta[1])
                if candidate in region or candidate in candidates:
                    continue
                target = self.cells.get(candidate)
                if target is not None and target.state not in (
                    CellState.UNKNOWN,
                    CellState.OBSERVED,
                ):
                    continue
                candidates[candidate] = Frontier(
                    cell=candidate,
                    from_cell=current,
                    direction=direction,
                    expected_new=self._expected_new_cells(candidate),
                )
        return sorted(candidates.values(), key=lambda item: (item.cell[1], item.cell[0]))

    def _expected_new_cells(self, cell: tuple[int, int]) -> int:
        """Number of still-unknown cells a 10x9 view centred on ``cell`` would show."""
        center_x, center_y = cell
        count = 0
        for dy in range(-VIEW_PLAYER[1], VIEW_H - VIEW_PLAYER[1]):
            for dx in range(-VIEW_PLAYER[0], VIEW_W - VIEW_PLAYER[0]):
                known = self.cells.get((center_x + dx, center_y + dy))
                if known is None or known.state is CellState.UNKNOWN:
                    count += 1
        return count

    def unknown_adjacent_count(self, cell: tuple[int, int]) -> int:
        """How many of the four neighbours are still unknown (missing or UNKNOWN)."""
        count = 0
        for neighbour in self.neighbours(cell):
            known = self.cells.get(neighbour)
            if known is None or known.state is CellState.UNKNOWN:
                count += 1
        return count

    # -- statistics -------------------------------------------------------- #

    def count_state(self, state: CellState) -> int:
        """Number of mapped cells whose state is exactly ``state``."""
        coerced = _coerce_cell_state(state, strict=True)
        return sum(1 for item in self.cells.values() if item.state is coerced)

    def size(self) -> int:
        """Number of *known* cells (observed or deeper knowledge)."""
        return sum(1 for item in self.cells.values() if item.known)

    def bounds(self) -> tuple[int, int, int, int]:
        """Inclusive ``(min_x, min_y, max_x, max_y)`` of mapped cells; zeros if empty."""
        if not self.cells:
            return (0, 0, 0, 0)
        xs = [item[0] for item in self.cells]
        ys = [item[1] for item in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))

    def coverage(self, *, start: tuple[int, int] | None = None) -> CoverageStats:
        """Honest exploration counters (see module docstring / ``NAV_CONTRACT.md``).

        ``coverage_ratio = walkable / (walkable + unknown_frontier)`` uses the
        reachable walkable set.  ``explored_ratio`` is the share of
        passability-resolved cells (walkable/blocked/temp_blocked/exit/interactive)
        among resolved + unknown frontier; unreachable/deferred cells are excluded
        from both and only reported in their own counters.
        """
        stats = CoverageStats(area_id=self.area_id)
        walkable_cells: list[tuple[int, int]] = []
        for key, item in self.cells.items():
            if not item.known:
                continue
            stats.cells_total += 1
            if item.state is CellState.OBSERVED:
                stats.observed += 1
            elif item.state is CellState.WALKABLE:
                stats.walkable += 1
                walkable_cells.append(key)
            elif item.state is CellState.BLOCKED:
                stats.blocked += 1
            elif item.state is CellState.TEMP_BLOCKED:
                stats.temp_blocked += 1
            elif item.state is CellState.EXIT:
                stats.exits += 1
            elif item.state is CellState.INTERACTIVE:
                stats.interactive += 1
            elif item.state is CellState.UNREACHABLE:
                stats.unreachable += 1
            elif item.state is CellState.DEFERRED:
                stats.deferred += 1

        chosen = self._coverage_start(start, walkable_cells)
        if chosen is not None:
            region = self.reachable_region(chosen)
            reachable_walkable = 0
            frontier: set[tuple[int, int]] = set()
            for key in region:
                item = self.cells.get(key)
                is_seed = key == chosen
                if item is not None and (item.state is CellState.WALKABLE or is_seed):
                    if item.state is CellState.WALKABLE:
                        reachable_walkable += 1
                    for neighbour in self.neighbours(key):
                        if neighbour in region:
                            continue
                        adjacent = self.cells.get(neighbour)
                        if adjacent is None or adjacent.state in (
                            CellState.UNKNOWN,
                            CellState.OBSERVED,
                        ):
                            frontier.add(neighbour)
            stats.unknown_frontier = len(frontier)
            denominator = reachable_walkable + stats.unknown_frontier
            stats.coverage_ratio = (
                reachable_walkable / denominator if denominator > 0 else 0.0
            )
        resolved = (
            stats.walkable
            + stats.blocked
            + stats.temp_blocked
            + stats.exits
            + stats.interactive
        )
        explored_denominator = resolved + stats.unknown_frontier
        stats.explored_ratio = (
            resolved / explored_denominator if explored_denominator > 0 else 0.0
        )
        return stats

    def _coverage_start(
        self,
        start: tuple[int, int] | None,
        walkable_cells: Sequence[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if start is not None:
            return self._key_of(start)
        if walkable_cells:
            return min(walkable_cells, key=lambda item: (item[1], item[0]))
        known = [key for key, item in self.cells.items() if item.known]
        if known:
            return min(known, key=lambda item: (item[1], item[0]))
        return None

    # -- rendering --------------------------------------------------------- #

    def render_ascii(
        self,
        *,
        center: tuple[int, int] | None = None,
        radius: int = 10,
        path: list[tuple[int, int]] | None = None,
        target: tuple[int, int] | None = None,
        legend: bool = True,
    ) -> str:
        """Compact deterministic ASCII rendering for prompts and the viewer.

        ``center`` defaults to the middle of the known bounds; when it is supplied
        explicitly that cell is drawn with ``@`` (typically the player pose).
        ``path`` cells are drawn with ``*`` and ``target`` with ``T``.
        """
        span = max(1, _as_int(radius, 10))
        if center is not None:
            center_x, center_y = self._key_of(center)
            mark_center = True
        else:
            min_x, min_y, max_x, max_y = self.bounds()
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            mark_center = False
        path_set = {self._key_of(item) for item in (path or [])}
        target_key = self._key_of(target) if target is not None else None
        lines: list[str] = [
            f"area={self.area_id or '?'} kind={self.kind} cells={self.size()} "
            f"walkable={self.count_state(CellState.WALKABLE)} center=({center_x},{center_y})",
        ]
        for world_y in range(center_y - span, center_y + span + 1):
            row: list[str] = []
            for world_x in range(center_x - span, center_x + span + 1):
                known = self.cells.get((world_x, world_y))
                symbol = _SYMBOLS[CellState.UNKNOWN] if known is None else _SYMBOLS[known.state]
                if (world_x, world_y) in path_set:
                    symbol = "*"
                if target_key == (world_x, world_y):
                    symbol = "T"
                if mark_center and (world_x, world_y) == (center_x, center_y):
                    symbol = "@"
                row.append(symbol)
            lines.append("".join(row))
        if legend:
            lines.extend(_LEGEND_LINES)
        return "\n".join(lines)

    # -- persistence ------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot (deterministic cell/edge ordering)."""
        cells = [item.to_dict() for item in sorted(self.cells.values(), key=lambda c: (c.y, c.x))]
        edges = [
            item.to_dict()
            for item in sorted(
                self.edges.values(),
                key=lambda e: (e.a[1], e.a[0], e.b[1], e.b[0]),
            )
        ]
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "area_id": self.area_id,
            "kind": self.kind,
            "created_frame": int(self.created_frame),
            "updated_frame": int(self.updated_frame),
            "cells": cells,
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GridMap":
        """Tolerant inverse of :meth:`to_dict` (see module docstring)."""
        if not isinstance(data, Mapping):
            raise ValueError(f"grid map data must be a mapping, got {type(data)!r}")
        version = _as_int(data.get("version", GRID_MAP_FORMAT_VERSION), GRID_MAP_FORMAT_VERSION)
        if version > GRID_MAP_FORMAT_VERSION:
            raise ValueError(
                f"grid map version {version} is newer than supported version {GRID_MAP_FORMAT_VERSION}"
            )
        kind = str(data.get("kind") or "unknown")
        grid = cls(
            str(data.get("area_id") or ""),
            frame_index=_as_int(data.get("created_frame", 0), 0),
            kind=kind if kind in ("interior", "exterior", "unknown") else "unknown",
        )
        grid.updated_frame = _as_int(data.get("updated_frame", grid.created_frame), grid.created_frame)
        warnings: list[str] = []

        raw_cells = data.get("cells", [])
        if raw_cells is None:
            raw_cells = []
        if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
            raise ValueError("grid map 'cells' must be a list")
        for entry in raw_cells:
            try:
                item = Cell.from_dict(entry)
            except (TypeError, ValueError) as exc:
                warnings.append(f"skipped malformed cell: {exc}")
                continue
            grid.cells[item.cell] = item

        raw_edges = data.get("edges", [])
        if raw_edges is None:
            raw_edges = []
        if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
            raise ValueError("grid map 'edges' must be a list")
        for entry in raw_edges:
            try:
                item = Edge.from_dict(entry)
            except (TypeError, ValueError) as exc:
                warnings.append(f"skipped malformed edge: {exc}")
                continue
            grid.edges[item.key] = item

        grid.load_warnings = warnings
        return grid

    def save(self, path: str | Path) -> None:
        """Atomically persist this grid as versioned JSON at ``path``."""
        save_json_atomic(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "GridMap":
        """Load a grid JSON file; raises ``ValueError``/``OSError`` on bad data."""
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt grid map JSON in {source}: {exc}") from exc
        return cls.from_dict(data)
