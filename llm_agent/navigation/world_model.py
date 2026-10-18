"""Multi-area world graph + visual relocalization for Navigation v2 (owner: nav-map).

The :class:`WorldModel` owns the persistent knowledge of the whole explored world:

* a registry of :class:`AreaRecord` objects (one :class:`GridMap` each),
* directed :class:`WorldConnection` edges between areas (warps / walk transitions),
* the current area and pose,
* :meth:`WorldModel.match_view`, which re-identifies the current area + pose from
  a pure :class:`~llm_agent.navigation.types.FrameView` by matching visible tile
  hashes against the stored per-cell tile looks.

Relocalization acceptance rule (``NAV_CONTRACT.md`` §4): a match is accepted only
when ``score >= min_score`` **and** ``margin >= 0.08`` over the runner-up, or when
an expected area (typically derived from a just-recorded connection) agrees.
Ambiguous candidates are kept in the returned list so the caller can decide —
the world model never merges two areas on weak evidence.

Persistence is plain versioned JSON (atomic writes, tolerant/versioned loads,
no absolute paths): ``<root>/navigation/world.json`` plus
``<root>/navigation/areas/<area_id>.json``.  If ``root`` itself is a directory
named ``navigation`` it is used directly, and a previously loaded layout is
remembered for subsequent saves.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_agent.navigation.grid_map import GridMap, save_json_atomic
from llm_agent.navigation.types import (
    DIRECTIONS,
    VIEW_PLAYER,
    AreaMatch,
    CellState,
    CoverageStats,
    FrameView,
)

__all__ = [
    "DEFAULT_MATCH_MIN_MARGIN",
    "DEFAULT_MATCH_MIN_SCORE",
    "MIN_COMPARABLE_TILES",
    "WORLD_MODEL_FORMAT",
    "WORLD_MODEL_FORMAT_VERSION",
    "AreaRecord",
    "WorldConnection",
    "WorldModel",
]

#: Persistence format identifier / version written into ``world.json``.
WORLD_MODEL_FORMAT = "nav.world_model"
WORLD_MODEL_FORMAT_VERSION = 1

#: Minimum number of comparable (visible + hash-known) tiles for a pose score to
#: be meaningful.  Below this the normalization is noise, so the pose is skipped.
MIN_COMPARABLE_TILES = 6

#: Default acceptance thresholds for :meth:`WorldModel.match_view`/:meth:`locate`.
DEFAULT_MATCH_MIN_SCORE = 0.55
DEFAULT_MATCH_MIN_MARGIN = 0.08

#: Bounded number of recorded entry cells per area (keeps world.json small).
MAX_ENTRY_CELLS = 8

_VALID_AREA_KINDS = ("interior", "exterior", "unknown")
_VALID_CONNECTION_KINDS = ("warp", "walk", "unknown")
_VALID_CONNECTION_DIRECTIONS = set(DIRECTIONS) | {"warp"}


# --------------------------------------------------------------------------- #
# Coercion helpers (tolerant readers)
# --------------------------------------------------------------------------- #


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
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


def _known_kind(value: Any) -> str:
    kind = str(value or "unknown")
    return kind if kind in _VALID_AREA_KINDS else "unknown"


# --------------------------------------------------------------------------- #
# Area record
# --------------------------------------------------------------------------- #


@dataclass
class AreaRecord:
    """Registry entry for one visited area.

    The :class:`GridMap` is persisted in its own JSON file
    (``areas/<area_id>.json``), therefore :meth:`to_dict` only serializes the
    metadata; :meth:`from_dict` accepts the grid via an optional argument.
    """

    area_id: str
    kind: str = "unknown"
    grid: GridMap = field(default_factory=lambda: GridMap(""))
    visits: int = 0
    first_frame: int = 0
    last_frame: int = 0
    entry_cells: list[tuple[int, int]] = field(default_factory=list)
    explored: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.area_id = str(self.area_id or "")
        self.kind = _known_kind(self.kind)
        if not isinstance(self.grid, GridMap):
            raise TypeError(f"AreaRecord.grid must be a GridMap, got {type(self.grid)!r}")
        if not self.grid.area_id:
            self.grid.area_id = self.area_id
        self.visits = max(0, _as_int(self.visits, 0))
        self.first_frame = _as_int(self.first_frame, 0)
        self.last_frame = _as_int(self.last_frame, 0)
        normalized_entries: list[tuple[int, int]] = []
        for entry in self.entry_cells or []:
            cell = _as_cell(entry)
            if cell is None or cell in normalized_entries:
                continue
            normalized_entries.append(cell)
            if len(normalized_entries) >= MAX_ENTRY_CELLS:
                break
        self.entry_cells = normalized_entries
        self.explored = bool(self.explored)
        self.notes = [str(note) for note in (self.notes or []) if str(note)]

    # -- conveniences ------------------------------------------------------ #

    def coverage(self) -> CoverageStats:
        """Coverage statistics of this area's grid (see :meth:`GridMap.coverage`)."""
        return self.grid.coverage()

    def walkable_count(self) -> int:
        """Number of verified-walkable cells in this area."""
        return self.grid.count_state(CellState.WALKABLE)

    # -- persistence ------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable metadata snapshot (the grid is stored separately)."""
        return {
            "area_id": self.area_id,
            "kind": self.kind,
            "visits": int(self.visits),
            "first_frame": int(self.first_frame),
            "last_frame": int(self.last_frame),
            "entry_cells": [[int(cell[0]), int(cell[1])] for cell in self.entry_cells],
            "explored": bool(self.explored),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, grid: GridMap | None = None) -> "AreaRecord":
        """Tolerant inverse of :meth:`to_dict`; ``grid`` may be attached later."""
        if not isinstance(data, Mapping):
            raise ValueError(f"area record must be a mapping, got {type(data)!r}")
        area_id = str(data.get("area_id") or "")
        if not area_id:
            raise ValueError("area record needs a non-empty area_id")
        entries_raw = data.get("entry_cells") or []
        entries: list[tuple[int, int]] = []
        if isinstance(entries_raw, Sequence) and not isinstance(entries_raw, (str, bytes)):
            for entry in entries_raw:
                cell = _as_cell(entry)
                if cell is not None:
                    entries.append(cell)
        notes_raw = data.get("notes") or []
        if isinstance(notes_raw, str):
            notes = [notes_raw]
        elif isinstance(notes_raw, Sequence):
            notes = [str(note) for note in notes_raw]
        else:
            notes = []
        return cls(
            area_id=area_id,
            kind=_known_kind(data.get("kind")),
            grid=grid if grid is not None else GridMap(area_id),
            visits=_as_int(data.get("visits", 0), 0),
            first_frame=_as_int(data.get("first_frame", 0), 0),
            last_frame=_as_int(data.get("last_frame", 0), 0),
            entry_cells=entries,
            explored=bool(data.get("explored", False)),
            notes=notes,
        )


# --------------------------------------------------------------------------- #
# World connection
# --------------------------------------------------------------------------- #


@dataclass
class WorldConnection:
    """One directed transition between two areas (warp or walkable connection)."""

    from_area: str
    from_cell: tuple[int, int]
    direction: str
    to_area: str
    to_pose: tuple[int, int] | None = None
    kind: str = "warp"
    confidence: float = 0.5
    observations: int = 1

    def __post_init__(self) -> None:
        self.from_area = str(self.from_area or "")
        self.to_area = str(self.to_area or "")
        self.from_cell = _as_cell_strict(self.from_cell)
        direction = str(self.direction or "")
        self.direction = direction if direction in _VALID_CONNECTION_DIRECTIONS else ""
        self.to_pose = _as_cell(self.to_pose) if self.to_pose is not None else None
        self.kind = self.kind if self.kind in _VALID_CONNECTION_KINDS else "unknown"
        self.confidence = _clamp01(self.confidence, 0.5)
        self.observations = max(1, _as_int(self.observations, 1))

    @property
    def key(self) -> tuple[str, tuple[int, int], str, str]:
        """Identity used to merge repeated observations of the same transition."""
        return (self.from_area, self.from_cell, self.direction, self.to_area)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the connection."""
        return {
            "from_area": self.from_area,
            "from_cell": [int(self.from_cell[0]), int(self.from_cell[1])],
            "direction": self.direction,
            "to_area": self.to_area,
            "to_pose": None if self.to_pose is None else [int(self.to_pose[0]), int(self.to_pose[1])],
            "kind": self.kind,
            "confidence": round(float(self.confidence), 4),
            "observations": int(self.observations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldConnection":
        """Tolerant inverse of :meth:`to_dict`."""
        if not isinstance(data, Mapping):
            raise ValueError(f"connection entry must be a mapping, got {type(data)!r}")
        from_area = str(data.get("from_area") or "")
        to_area = str(data.get("to_area") or "")
        if not from_area and not to_area:
            raise ValueError("connection entry needs from_area and/or to_area")
        return cls(
            from_area=from_area,
            from_cell=_as_cell(data.get("from_cell")) or (0, 0),
            direction=str(data.get("direction") or ""),
            to_area=to_area,
            to_pose=_as_cell(data.get("to_pose")),
            kind=str(data.get("kind") or "unknown"),
            confidence=_clamp01(data.get("confidence", 0.5), 0.5),
            observations=_as_int(data.get("observations", 1), 1),
        )


# --------------------------------------------------------------------------- #
# World model
# --------------------------------------------------------------------------- #


class WorldModel:
    """Persistent multi-area map + visual relocalization.

    ``root`` is the run/output directory; files are written under
    ``<root>/navigation/``.  When ``root`` itself is named ``navigation`` the
    model writes directly into it.  With ``root=None`` the model is memory-only
    and :meth:`save` raises ``ValueError`` (``load`` returns ``False``).
    """

    FORMAT = WORLD_MODEL_FORMAT
    VERSION = WORLD_MODEL_FORMAT_VERSION

    def __init__(self, *, root: str | Path | None = None) -> None:
        self.root: Path | None = Path(root) if root is not None else None
        self.areas: dict[str, AreaRecord] = {}
        self.connections: list[WorldConnection] = []
        self.current_area: str = ""
        self.current_pose: tuple[int, int] = (0, 0)
        #: Non-fatal problems collected by the tolerant :meth:`load`.
        self.load_warnings: list[str] = []
        #: Directory a previous :meth:`load`/``save`` resolved to (layout memory).
        self._storage_dir: Path | None = None

    # -- convenience properties ------------------------------------------- #

    @property
    def current_record(self) -> AreaRecord | None:
        """The :class:`AreaRecord` of :attr:`current_area` (or ``None``)."""
        return self.areas.get(self.current_area)

    @property
    def current_grid(self) -> GridMap | None:
        """The :class:`GridMap` of the current area (or ``None``)."""
        record = self.current_record
        return record.grid if record is not None else None

    @property
    def world_path(self) -> Path | None:
        """Path of ``world.json`` (``None`` for a memory-only model)."""
        if self.root is None:
            return None
        return self.storage_dir() / "world.json"

    def storage_dir(self) -> Path:
        """Directory holding ``world.json`` and ``areas/`` (resolved layout)."""
        if self._storage_dir is not None:
            return self._storage_dir
        if self.root is None:
            raise ValueError("WorldModel has no root directory (root=None)")
        root = Path(self.root)
        if root.name.lower() == "navigation":
            return root
        return root / "navigation"

    # -- area registry ----------------------------------------------------- #

    def create_area(
        self,
        *,
        kind: str = "unknown",
        frame_index: int = 0,
        area_id: str | None = None,
        entry_cell: tuple[int, int] | None = None,
    ) -> AreaRecord:
        """Create (or fetch) an area record with a fresh empty grid.

        With ``area_id=None`` a deterministic unused id ``area_001``, ``area_002``,
        ... is generated.  Creating an existing id is idempotent: metadata is
        refreshed and the existing grid is kept.  The new area becomes current only
        through :meth:`set_current` (or :meth:`area` without an argument), which is
        also what records an entry cell.  ``entry_cell`` can be supplied explicitly
        when the pose is already known.
        """
        frame = _as_int(frame_index, 0)
        normalized_kind = _known_kind(kind)
        if area_id is None:
            index = 1
            while f"area_{index:03d}" in self.areas:
                index += 1
            area_id = f"area_{index:03d}"
        key = str(area_id)
        existing = self.areas.get(key)
        if existing is not None:
            if normalized_kind != "unknown":
                existing.kind = normalized_kind
                existing.grid.kind = normalized_kind
            existing.last_frame = max(existing.last_frame, frame)
            if entry_cell is not None:
                self._add_entry_cell(existing, entry_cell)
            return existing
        grid = GridMap(key, frame_index=frame, kind=normalized_kind)
        record = AreaRecord(
            area_id=key,
            kind=normalized_kind,
            grid=grid,
            visits=0,
            first_frame=frame,
            last_frame=frame,
            entry_cells=[],
        )
        if entry_cell is not None:
            self._add_entry_cell(record, entry_cell)
        self.areas[key] = record
        return record

    def area(self, area_id: str | None = None) -> AreaRecord:
        """Fetch an area record; ``None`` means the current area.

        A missing id is created on demand (the grid starts empty) so callers never
        have to special-case "unknown area yet" bookkeeping.
        """
        if area_id is None:
            current = self.areas.get(self.current_area)
            if current is not None:
                return current
            record = self.create_area()
            self.current_area = record.area_id
            self.current_pose = (0, 0)
            self._add_entry_cell(record, (0, 0))
            return record
        key = str(area_id)
        existing = self.areas.get(key)
        if existing is not None:
            return existing
        return self.create_area(area_id=key)

    def set_current(self, area_id: str, pose: tuple[int, int]) -> None:
        """Switch the current area/pose and remember the pose as an entry cell."""
        key = str(area_id)
        cell = _as_cell(pose) or (0, 0)
        record = self.areas.get(key)
        if record is None:
            record = self.create_area(area_id=key, entry_cell=cell)
        else:
            self._add_entry_cell(record, cell)
        self.current_area = record.area_id
        self.current_pose = cell

    def record_visit(self, area_id: str, *, frame_index: int) -> None:
        """Count one visit to ``area_id`` and refresh its frame bookkeeping."""
        record = self.area(area_id)
        frame = _as_int(frame_index, 0)
        if record.visits == 0:
            record.first_frame = frame
        record.visits += 1
        record.last_frame = max(record.last_frame, frame)

    @staticmethod
    def _add_entry_cell(record: AreaRecord, cell: tuple[int, int] | None) -> None:
        key = _as_cell(cell)
        if key is None or key in record.entry_cells:
            return
        if len(record.entry_cells) >= MAX_ENTRY_CELLS:
            return
        record.entry_cells.append(key)

    def set_area_kind(self, area_id: str, kind: str) -> None:
        """Set an area's ``interior``/``exterior``/``unknown`` classification."""
        record = self.area(area_id)
        normalized = _known_kind(kind)
        record.kind = normalized
        record.grid.kind = normalized

    def mark_area_explored(self, area_id: str, explored: bool = True) -> None:
        """Mark an area as explored (used by :meth:`unexplored_areas`)."""
        self.area(area_id).explored = bool(explored)

    def unexplored_areas(self) -> list[str]:
        """Sorted ids of areas not yet marked as explored."""
        return sorted(area_id for area_id, record in self.areas.items() if not record.explored)

    # -- connections ------------------------------------------------------- #

    def add_connection(self, conn: WorldConnection) -> None:
        """Record a transition; repeated observations merge into one edge."""
        if not isinstance(conn, WorldConnection):
            raise TypeError(f"expected a WorldConnection, got {type(conn)!r}")
        for existing in self.connections:
            if existing.key == conn.key:
                existing.observations += max(1, conn.observations)
                existing.confidence = max(existing.confidence, conn.confidence)
                if existing.to_pose is None and conn.to_pose is not None:
                    existing.to_pose = conn.to_pose
                if existing.kind == "unknown" and conn.kind != "unknown":
                    existing.kind = conn.kind
                return
        self.connections.append(conn)

    # -- relocalization ---------------------------------------------------- #

    @staticmethod
    def _score_pose(
        grid: GridMap,
        pose: tuple[int, int],
        tiles: list[tuple[int, int, str]],
    ) -> tuple[int, int]:
        """Score one candidate pose: (matched, comparable) over the 10x9 window."""
        pose_x, pose_y = pose
        cells = grid.cells
        matched = 0
        compared = 0
        for dx, dy, tile_hash in tiles:
            cell = cells.get((pose_x + dx, pose_y + dy))
            if cell is None or not cell.tile_hashes:
                continue
            compared += 1
            if tile_hash in cell.tile_hashes:
                matched += 1
        return matched, compared

    @staticmethod
    def _visible_tiles(view: FrameView) -> list[tuple[int, int, str]]:
        """Player-relative visible tiles, excluding the player sprite tile and blanks."""
        tiles: list[tuple[int, int, str]] = []
        for tile in getattr(view, "tiles", None) or ():
            dx = _as_int(getattr(tile, "tx", 0), 0) - VIEW_PLAYER[0]
            dy = _as_int(getattr(tile, "ty", 0), 0) - VIEW_PLAYER[1]
            if (dx, dy) == (0, 0):
                # The player's own sprite is not world content: never compare it.
                continue
            tile_hash = str(getattr(tile, "tile_hash", "") or "")
            if tile_hash:
                tiles.append((dx, dy, tile_hash))
        return tiles

    def match_view(
        self,
        view: FrameView,
        *,
        candidate_area_ids: list[str] | None = None,
        min_score: float = DEFAULT_MATCH_MIN_SCORE,
        max_results: int = 4,
    ) -> list[AreaMatch]:
        """Rank (area, pose) hypotheses for one frame, best first.

        For every candidate area the visible 10x9 tile-hash window is slid over the
        known cells and scored ``matched / comparable`` (comparable = visible tiles
        whose cell has stored tile hashes; the player's own tile is ignored).
        Candidate poses are generated from tile-hash hits — mathematically the same
        set of scores as trying every known cell, but fast on large maps.  Poses
        with fewer than :data:`MIN_COMPARABLE_TILES` comparable tiles are skipped.

        The returned :class:`AreaMatch` entries carry ``second_score``/``margin``
        against the next-best hypothesis; ambiguous candidates are all returned
        (the caller applies the acceptance rule, see :meth:`locate`).
        """
        tiles = self._visible_tiles(view)
        if not tiles:
            return []
        if candidate_area_ids is None:
            area_ids = sorted(self.areas)
        else:
            area_ids = sorted({str(item) for item in candidate_area_ids if str(item) in self.areas})
        if not area_ids:
            return []
        limit = max(1, _as_int(max_results, 4))
        threshold = _as_float(min_score, DEFAULT_MATCH_MIN_SCORE)
        if threshold < 0.0:
            threshold = 0.0

        # Candidate poses from hash hits.  A pose with zero matching tiles can
        # never reach min_score > 0, so it cannot affect the accepted result.
        hit_counts: dict[tuple[str, tuple[int, int]], int] = {}
        for area_id in area_ids:
            grid = self.areas[area_id].grid
            index: dict[str, list[tuple[int, int]]] = {}
            for cell in grid.cells.values():
                for tile_hash in cell.tile_hashes:
                    index.setdefault(tile_hash, []).append(cell.cell)
            if not index:
                continue
            for dx, dy, tile_hash in tiles:
                for cell_x, cell_y in index.get(tile_hash, ()):
                    key = (area_id, (cell_x - dx, cell_y - dy))
                    hit_counts[key] = hit_counts.get(key, 0) + 1

        # Overlap-aware ranking.  A plain matched/comparable ratio can be
        # inflated by thin overlaps (e.g. 30/30 uniform-floor tiles at a map
        # corner outranking a true 82/89 full-window match; this happened on
        # real frames and blocked re-entry relocalization).  Hypotheses that
        # explain a substantial part of the visible window therefore rank above
        # thin ones; among equally substantive hypotheses the ratio decides.
        # When the known map is still sparse, every candidate is "thin" and the
        # plain ratio ranking is kept (conservative margin rules still apply).
        substantive = max(MIN_COMPARABLE_TILES, int(round(0.6 * max(1, len(tiles)))))
        ranked: list[tuple[float, str, tuple[int, int], int, int]] = []
        for (area_id, pose), _hits in hit_counts.items():
            matched, compared = self._score_pose(self.areas[area_id].grid, pose, tiles)
            if compared < MIN_COMPARABLE_TILES:
                continue
            ranked.append((matched / compared, area_id, pose, matched, compared))
        ranked.sort(
            key=lambda item: (
                0 if int(item[4]) >= substantive else 1,
                -item[0],
                -item[3],
                item[1],
                item[2][1],
                item[2][0],
            )
        )

        results: list[AreaMatch] = []
        for index, (score, area_id, pose, matched, compared) in enumerate(ranked):
            if len(results) >= limit:
                break
            if score < threshold:
                break
            top_is_substantive = int(compared) >= substantive
            runner = None
            for candidate in ranked[index + 1:]:
                if top_is_substantive and int(candidate[4]) < substantive:
                    continue
                runner = candidate
                break
            second_score = runner[0] if runner is not None else 0.0
            evidence: dict[str, Any] = {
                "candidate_count": len(ranked),
                "rank": index,
                "visible_tiles": len(tiles),
                "compared": int(compared),
                "substantive_overlap": int(substantive),
                "substantive": bool(top_is_substantive),
            }
            if runner is not None:
                evidence["second_area"] = runner[1]
                evidence["second_pose"] = [int(runner[2][0]), int(runner[2][1])]
                evidence["second_compared"] = int(runner[4])
            results.append(
                AreaMatch(
                    area_id=area_id,
                    pose=pose,
                    score=round(float(score), 4),
                    compared=int(compared),
                    matched=int(matched),
                    second_score=round(float(second_score), 4),
                    margin=round(float(score - second_score), 4),
                    evidence=evidence,
                )
            )
        return results

    def locate(
        self,
        view: FrameView,
        *,
        expected_area: str | None = None,
        min_score: float = DEFAULT_MATCH_MIN_SCORE,
        min_margin: float = DEFAULT_MATCH_MIN_MARGIN,
    ) -> AreaMatch | None:
        """Accept one relocalization hypothesis, or ``None`` when ambiguous.

        Acceptance: the expected area agrees (a just-recorded connection), or the
        top match reaches ``min_score`` with ``margin >= min_margin`` over the
        runner-up.  The model's current area/pose is **not** modified here; the
        caller decides how to apply a positive result.
        """
        matches = self.match_view(view, min_score=min_score, max_results=8)
        if not matches:
            return None
        expected = str(expected_area) if expected_area else ""
        if expected:
            for match in matches:
                if match.area_id == expected:
                    match.evidence["accepted_by"] = "expected_area"
                    return match
        best = matches[0]
        if best.score >= min_score and best.margin >= min_margin:
            best.evidence["accepted_by"] = "margin"
            return best
        return None

    # -- persistence ------------------------------------------------------- #

    def _candidate_dirs(self) -> list[Path]:
        """Preferred storage directory first, then ``root`` itself (layout probe)."""
        if self.root is None:
            return []
        root = Path(self.root)
        preferred = root if root.name.lower() == "navigation" else root / "navigation"
        if preferred == root:
            return [root]
        return [preferred, root]

    def _find_storage_dir(self) -> Path | None:
        """Existing directory containing ``world.json``, if any."""
        for candidate in self._candidate_dirs():
            if (candidate / "world.json").is_file():
                return candidate
        return None

    @staticmethod
    def _area_filename(area_id: str) -> str:
        """Deterministic safe file name for one area grid (``<area_id>.json``)."""
        text = str(area_id)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "area"
        if safe == text:
            return f"{safe}.json"
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()
        return f"{safe}_{digest}.json"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable world snapshot (area metadata + connections)."""
        connections = sorted(
            self.connections,
            key=lambda conn: (conn.from_area, conn.from_cell, conn.direction, conn.to_area),
        )
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "current_area": self.current_area,
            "current_pose": [int(self.current_pose[0]), int(self.current_pose[1])],
            "areas": [self.areas[area_id].to_dict() for area_id in sorted(self.areas)],
            "connections": [conn.to_dict() for conn in connections],
        }

    def save(self) -> None:
        """Atomically persist world.json + per-area grid JSON files.

        Grids are written before ``world.json`` so an interrupted save never points
        at missing area files.  Requires ``root`` (raises ``ValueError`` otherwise).
        """
        if self.root is None:
            raise ValueError("WorldModel cannot save without a root directory (root=None)")
        base = self.storage_dir()
        areas_dir = base / "areas"
        areas_dir.mkdir(parents=True, exist_ok=True)
        for area_id in sorted(self.areas):
            record = self.areas[area_id]
            record.grid.area_id = area_id
            record.grid.kind = record.kind
            record.grid.save(areas_dir / self._area_filename(area_id))
        save_json_atomic(base / "world.json", self.to_dict())
        self._storage_dir = base

    def load(self) -> bool:
        """Tolerantly load a saved world; returns ``True`` when anything was read.

        Missing or corrupt ``world.json`` -> ``False`` with entries in
        :attr:`load_warnings` (a fresh world is a valid state).  A corrupt/missing
        per-area grid file keeps the area metadata and falls back to an empty grid,
        also recorded in :attr:`load_warnings`.  Unsupported future versions are
        refused with ``False`` (never guessed at).
        """
        self.load_warnings = []
        if self.root is None:
            return False
        base = self._find_storage_dir()
        if base is None:
            return False
        world_file = base / "world.json"
        try:
            text = world_file.read_text(encoding="utf-8")
        except OSError as exc:
            self.load_warnings.append(f"cannot read {world_file.name}: {exc}")
            return False
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self.load_warnings.append(f"corrupt world.json: {exc}")
            return False
        if not isinstance(data, Mapping):
            self.load_warnings.append("corrupt world.json: top level is not an object")
            return False
        version = _as_int(data.get("version", WORLD_MODEL_FORMAT_VERSION), WORLD_MODEL_FORMAT_VERSION)
        if version > WORLD_MODEL_FORMAT_VERSION:
            self.load_warnings.append(
                f"unsupported world.json version {version} "
                f"(supported: {WORLD_MODEL_FORMAT_VERSION})"
            )
            return False

        areas: dict[str, AreaRecord] = {}
        raw_areas = data.get("areas") or []
        if not isinstance(raw_areas, Sequence) or isinstance(raw_areas, (str, bytes)):
            self.load_warnings.append("corrupt world.json: 'areas' is not a list")
            raw_areas = []
        for entry in raw_areas:
            try:
                record = AreaRecord.from_dict(entry)
            except (TypeError, ValueError) as exc:
                self.load_warnings.append(f"skipped malformed area record: {exc}")
                continue
            grid_file = base / "areas" / self._area_filename(record.area_id)
            try:
                record.grid = GridMap.load(grid_file)
            except FileNotFoundError:
                self.load_warnings.append(f"missing grid file for area {record.area_id!r}")
                record.grid = GridMap(record.area_id, frame_index=record.first_frame, kind=record.kind)
            except (OSError, ValueError) as exc:
                self.load_warnings.append(f"unreadable grid file for area {record.area_id!r}: {exc}")
                record.grid = GridMap(record.area_id, frame_index=record.first_frame, kind=record.kind)
            if record.grid.area_id != record.area_id:
                record.grid.area_id = record.area_id
            if record.kind == "unknown" and record.grid.kind != "unknown":
                record.kind = record.grid.kind
            record.grid.kind = record.kind
            areas[record.area_id] = record

        connections: list[WorldConnection] = []
        raw_connections = data.get("connections") or []
        if not isinstance(raw_connections, Sequence) or isinstance(raw_connections, (str, bytes)):
            self.load_warnings.append("corrupt world.json: 'connections' is not a list")
            raw_connections = []
        for entry in raw_connections:
            try:
                connections.append(WorldConnection.from_dict(entry))
            except (TypeError, ValueError) as exc:
                self.load_warnings.append(f"skipped malformed connection record: {exc}")

        self.areas = areas
        self.connections = connections
        current_area = str(data.get("current_area") or "")
        if current_area in areas:
            self.current_area = current_area
            self.current_pose = _as_cell(data.get("current_pose")) or (0, 0)
        else:
            self.current_area = ""
            self.current_pose = (0, 0)
            if current_area:
                self.load_warnings.append(f"current area {current_area!r} not present in saved areas")
        self._storage_dir = base
        return True

    # -- summaries --------------------------------------------------------- #

    def summary(self) -> dict[str, Any]:
        """Compact machine-readable overview (current area, per-area counters)."""
        rows: list[dict[str, Any]] = []
        cells_total = 0
        walkable_total = 0
        explored_count = 0
        for area_id in sorted(self.areas):
            record = self.areas[area_id]
            coverage = record.grid.coverage()
            cells_total += coverage.cells_total
            walkable_total += coverage.walkable
            if record.explored:
                explored_count += 1
            rows.append(
                {
                    "area_id": area_id,
                    "kind": record.kind,
                    "visits": int(record.visits),
                    "cells": int(coverage.cells_total),
                    "walkable": int(coverage.walkable),
                    "coverage_ratio": round(float(coverage.coverage_ratio), 4),
                    "explored": bool(record.explored),
                    "current": area_id == self.current_area,
                }
            )
        current_coverage = None
        current = self.areas.get(self.current_area)
        if current is not None:
            current_coverage = current.grid.coverage().to_dict()
        return {
            "areas_total": len(self.areas),
            "connections": len(self.connections),
            "current_area": self.current_area,
            "current_pose": [int(self.current_pose[0]), int(self.current_pose[1])],
            "areas": rows,
            "areas_compact": [
                [row["area_id"], row["kind"], row["visits"], row["walkable"]] for row in rows
            ],
            "cells_total": int(cells_total),
            "walkable_total": int(walkable_total),
            "explored_areas": int(explored_count),
            "unexplored": self.unexplored_areas(),
            "current_coverage": current_coverage,
        }

    def render_summary(self) -> str:
        """Compact deterministic text/markdown summary for prompts and viewers."""
        data = self.summary()
        pose_x, pose_y = data["current_pose"]
        lines = [
            f"World: {data['areas_total']} areas | {data['connections']} connections | "
            f"current: {data['current_area'] or '-'} pose=({pose_x},{pose_y})"
        ]
        if data["areas"]:
            lines.append("Areas:")
            for row in data["areas"][:12]:
                marker = "*" if row["current"] else "-"
                flags = " (explored)" if row["explored"] else ""
                lines.append(
                    f"  {marker} {row['area_id']} [{row['kind']}] visits={row['visits']} "
                    f"walkable={row['walkable']} cov={float(row['coverage_ratio']):.0%}{flags}"
                )
            remaining = len(data["areas"]) - 12
            if remaining > 0:
                lines.append(f"  ... (+{remaining} more)")
        else:
            lines.append("Areas: (none)")
        if self.connections:
            lines.append("Connections:")
            ordered = sorted(
                self.connections,
                key=lambda conn: (conn.from_area, conn.from_cell, conn.direction, conn.to_area),
            )
            for conn in ordered[:12]:
                target_pose = "" if conn.to_pose is None else f" pose={list(conn.to_pose)}"
                lines.append(
                    f"  - {conn.from_area} {list(conn.from_cell)} {conn.direction} -> "
                    f"{conn.to_area}{target_pose} [{conn.kind}] "
                    f"conf={float(conn.confidence):.2f} obs={int(conn.observations)}"
                )
            remaining = len(ordered) - 12
            if remaining > 0:
                lines.append(f"  ... (+{remaining} more)")
        return "\n".join(lines)

    def graph_dict(self) -> dict[str, Any]:
        """Area nodes + connection edges for visualisation / tool status."""
        nodes: list[dict[str, Any]] = []
        for area_id in sorted(self.areas):
            record = self.areas[area_id]
            min_x, min_y, max_x, max_y = record.grid.bounds()
            coverage = record.grid.coverage()
            nodes.append(
                {
                    "id": area_id,
                    "area_id": area_id,
                    "kind": record.kind,
                    "visits": int(record.visits),
                    "cells": int(coverage.cells_total),
                    "walkable": int(coverage.walkable),
                    "coverage": round(float(coverage.coverage_ratio), 4),
                    "explored": bool(record.explored),
                    "current": area_id == self.current_area,
                    "entry_cells": [[int(cell[0]), int(cell[1])] for cell in record.entry_cells],
                    "bounds": [int(min_x), int(min_y), int(max_x), int(max_y)],
                }
            )
        connections = sorted(
            self.connections,
            key=lambda conn: (conn.from_area, conn.from_cell, conn.direction, conn.to_area),
        )
        edges = [conn.to_dict() for conn in connections]
        return {
            "current_area": self.current_area,
            "current_pose": [int(self.current_pose[0]), int(self.current_pose[1])],
            "areas": nodes,
            "nodes": nodes,
            "connections": edges,
            "edges": edges,
        }
