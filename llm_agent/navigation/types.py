"""Frozen, dependency-light data contracts for the Navigation v2 subsystem.

This module is the shared boundary between:

* ``vision.py``          — pure pixel/tile primitives (Lead)
* ``localization.py``    — visual motion estimation + pose tracking (Lead)
* ``grid_map.py``        — per-area logical grid map (nav-map owner)
* ``world_model.py``     — multi-area world graph + relocalization (nav-map owner)
* ``pathfinder.py``      — BFS/A* planning (nav-planner owner)
* ``explorer.py``        — frontier-based exploration policy (nav-planner owner)
* ``controller.py``      — verified low-level navigation loop (nav-control owner)
* ``navigation_agent.py``— high level API used by the DeepSeek tool layer (nav-control owner)
* ``viz.py``             — map/status rendering (nav-viz owner)

Hard rule: **nothing in this package may read game RAM**.  RAM is only used by the
repo's separate verification module :mod:`llm_agent.progress` (test/reporting only)
and by scripts under ``tests/navigation`` / ``work``.

Coordinate conventions (verified against real PyBoy frames of this ROM):

* Screen: 160x144 px, 10x9 tiles of 16x16 px.  ``grid_offset`` is the pixel
  offset (0..15, 0..15) of the tile lattice inside the screen; all frames of one
  area share it because the camera scrolls in whole tiles.
* The overworld camera keeps the player sprite **fixed at screen tile (4,4)**
  in every overworld map (verified indoors and outdoors).  Therefore a
  successful step shifts the *background content* by exactly one tile, while the
  player's screen position is unchanged.  ``VIEW_PLAYER`` is that screen tile.
* Area-local world coordinates are tile coordinates with **x to the right and
  y downward**, identical in sign to the screen axes.  A step in direction
  ``up`` decreases y by 1.  ``delta`` in :class:`MotionEstimate` is the world
  delta (``-camera_shift``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------- #
# Screen / grid geometry
# --------------------------------------------------------------------------- #

SCREEN_WIDTH: int = 160
SCREEN_HEIGHT: int = 144
TILE_PX: int = 16
VIEW_W: int = SCREEN_WIDTH // TILE_PX   # 10
VIEW_H: int = SCREEN_HEIGHT // TILE_PX  # 9

#: Screen tile the player sprite occupies in every overworld map (camera follow).
VIEW_PLAYER: tuple[int, int] = (4, 4)
#: Inclusive-exclusive pixel box that may change during a turn/walk animation.
PLAYER_BOX: tuple[int, int, int, int] = (
    VIEW_PLAYER[0] * TILE_PX - 4,
    VIEW_PLAYER[1] * TILE_PX - 4,
    (VIEW_PLAYER[0] + 1) * TILE_PX + 4,
    (VIEW_PLAYER[1] + 1) * TILE_PX + 4,
)

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")
#: direction -> (dx, dy) in world tile coordinates (y grows downward)
DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE: dict[str, str] = {"up": "down", "down": "up", "left": "right", "right": "left"}

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class MotionResult(str, Enum):
    """Verified result of exactly one directional input action."""

    MOVED = "moved"
    BLOCKED = "blocked"          # no displacement, obstacle evidence (pressed direction)
    TURNED_ONLY = "turned_only"  # no displacement, facing changed only
    UNKNOWN = "unknown"          # cannot decide, do NOT update coordinates
    TRANSITION = "transition"    # screen/map transition, needs relocalization
    INTERRUPTED = "interrupted"  # dialog/menu/battle/etc blocked the overworld


class FrameContext(str, Enum):
    """What kind of screen this frame shows (pixel only)."""

    OVERWORLD = "overworld"
    DIALOG = "dialog"
    MENU = "menu"
    BATTLE = "battle"
    TRANSITION = "transition"
    BLACK = "black"
    UNKNOWN = "unknown"


class CellState(str, Enum):
    """Knowledge state of one logical grid cell."""

    UNKNOWN = "unknown"            # never observed
    OBSERVED = "observed"          # seen, passability not verified
    WALKABLE = "walkable"          # actually walked through/onto
    BLOCKED = "blocked"            # obstacle evidence, probably persistent
    TEMP_BLOCKED = "temp_blocked"  # dynamic obstruction (NPC/object), may change
    EXIT = "exit"                  # suspected/confirmed area exit (warp/door)
    INTERACTIVE = "interactive"    # probably interactive object (sign/NPC/item)
    UNREACHABLE = "unreachable"    # cannot be reached from any known path
    DEFERRED = "deferred"          # reachable in principle but deliberately skipped


class EdgeState(str, Enum):
    """Passability relation between two orthogonally adjacent cells."""

    UNKNOWN = "unknown"
    OPEN = "open"                  # verified walkable in this direction
    BLOCKED = "blocked"            # verified obstacle
    TEMP_BLOCKED = "temp_blocked"  # dynamic obstacle, retry later
    EXIT = "exit"                  # crossing here triggers a transition


class ExploreStatus(str, Enum):
    """Terminal status of one ``explore_current_area`` / ``navigate_to`` run."""

    COMPLETED = "completed"              # no safe, valuable frontier left
    STEP_LIMIT = "step_limit"            # navigation action budget exhausted
    TIME_LIMIT = "time_limit"
    NO_PROGRESS = "no_progress"          # consecutive attempts produced nothing new
    INTERRUPTED_EVENT = "interrupted_event"  # dialog/battle/menu/story event
    TRANSITION = "transition"            # an area transition happened mid-run
    LOCALIZATION_LOST = "localization_lost"
    BLOCKED = "blocked"                  # cannot make progress at all
    PAUSED = "paused"
    ERROR = "error"

    @property
    def success(self) -> bool:
        return self is ExploreStatus.COMPLETED


class NavEventKind(str, Enum):
    """Non-navigation game events surfaced to the DeepSeek brain."""

    NONE = "none"
    DIALOG_EVENT = "dialog_event"
    BATTLE_EVENT = "battle_event"
    MENU_EVENT = "menu_event"
    STORY_EVENT = "story_event"
    TRANSITION_EVENT = "transition_event"
    UNKNOWN_EVENT = "unknown_event"


# --------------------------------------------------------------------------- #
# Visual observations (produced by vision.py / localization.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TileObs:
    """One visible 16x16 tile of the current frame."""

    tx: int                 # screen tile x (0..9)
    ty: int                 # screen tile y (0..8)
    tile_hash: str          # stable hash of the 16x16 luma tile
    luma: bytes = b""       # raw 256-byte luma tile (optional, for clustering)


@dataclass
class FrameView:
    """Pixel-only observation of one screenshot.

    ``tiles`` holds ``VIEW_W * VIEW_H`` entries ordered row-major.  ``rel``
    positions (relative to the player) are implied: ``dx = tx - VIEW_PLAYER[0]``.
    """

    frame_index: int
    signature: str                          # perceptual screen signature
    grid_offset: tuple[int, int]            # (ox, oy) px of the 16px lattice
    tiles: list[TileObs] = field(default_factory=list)
    context: FrameContext = FrameContext.OVERWORLD
    screen_event: str = "none"              # contracts.ScreenEvent value
    is_black: bool = False
    luma_mean: float = 0.0
    stable: bool = True                     # same screen as the previous capture
    notes: dict[str, Any] = field(default_factory=dict)

    def tile(self, tx: int, ty: int) -> TileObs | None:
        for obs in self.tiles:
            if obs.tx == tx and obs.ty == ty:
                return obs
        return None

    def rel_tiles(self) -> dict[tuple[int, int], TileObs]:
        return {(obs.tx - VIEW_PLAYER[0], obs.ty - VIEW_PLAYER[1]): obs for obs in self.tiles}


@dataclass
class PlayerPose:
    """Estimated player position inside the current area (local coordinates)."""

    x: int = 0
    y: int = 0
    facing: str | None = None
    confidence: float = 1.0
    valid: bool = True

    @property
    def cell(self) -> tuple[int, int]:
        return (int(self.x), int(self.y))

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "facing": self.facing,
            "confidence": round(float(self.confidence), 3),
            "valid": bool(self.valid),
        }


@dataclass
class MotionEstimate:
    """Result of analysing one directional action (exactly one input)."""

    result: MotionResult
    direction: str
    requested_tiles: int = 1
    delta: tuple[int, int] = (0, 0)          # world delta in tiles
    tiles_moved: int = 0
    camera_shift: tuple[int, int] = (0, 0)   # background content shift in tiles
    screen_delta: tuple[int, int] = (0, 0)   # player sprite screen tile delta
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    frame_index: int = 0
    context: FrameContext = FrameContext.OVERWORLD

    @property
    def moved(self) -> bool:
        return self.result is MotionResult.MOVED

    @property
    def blocked(self) -> bool:
        return self.result in (MotionResult.BLOCKED, MotionResult.TURNED_ONLY)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.value,
            "direction": self.direction,
            "requested_tiles": self.requested_tiles,
            "delta": list(self.delta),
            "tiles_moved": int(self.tiles_moved),
            "camera_shift": list(self.camera_shift),
            "screen_delta": list(self.screen_delta),
            "confidence": round(float(self.confidence), 3),
            "frame_index": int(self.frame_index),
            "context": self.context.value,
            "evidence": dict(self.evidence),
        }


# --------------------------------------------------------------------------- #
# Map / plan value objects
# --------------------------------------------------------------------------- #


@dataclass
class PathPlan:
    """A planned route through the grid map."""

    cells: list[tuple[int, int]] = field(default_factory=list)  # start..goal inclusive
    directions: list[str] = field(default_factory=list)        # len(cells)-1
    cost: float = 0.0
    reachable: bool = False
    reason: str = ""
    explored: int = 0

    @property
    def length(self) -> int:
        return max(0, len(self.cells) - 1)

    def next_direction(self) -> str | None:
        return self.directions[0] if self.directions else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [list(cell) for cell in self.cells],
            "directions": list(self.directions),
            "cost": round(float(self.cost), 3),
            "reachable": bool(self.reachable),
            "reason": self.reason,
            "length": self.length,
        }


@dataclass
class ExploreTarget:
    """One selected exploration goal."""

    cell: tuple[int, int] = (0, 0)          # cell to probe (usually unknown/observed)
    from_cell: tuple[int, int] = (0, 0)     # reachable walkable cell next to it
    direction: str = ""                     # step direction from from_cell to cell
    path: PathPlan | None = None
    score: float = 0.0
    expected_new_cells: int = 0
    kind: str = "frontier"                  # frontier|exit|interactive|unreachable_probe
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": list(self.cell),
            "from_cell": list(self.from_cell),
            "direction": self.direction,
            "score": round(float(self.score), 3),
            "expected_new_cells": int(self.expected_new_cells),
            "kind": self.kind,
            "reason": self.reason,
            "path_length": None if self.path is None else self.path.length,
            "path": None if self.path is None else self.path.to_dict(),
        }


@dataclass
class AreaMatch:
    """A relocalization hypothesis of one area for the current view."""

    area_id: str
    pose: tuple[int, int]
    score: float
    compared: int = 0
    matched: int = 0
    second_score: float = 0.0
    margin: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_id": self.area_id,
            "pose": list(self.pose),
            "score": round(float(self.score), 4),
            "compared": int(self.compared),
            "matched": int(self.matched),
            "second_score": round(float(self.second_score), 4),
            "margin": round(float(self.margin), 4),
            "evidence": dict(self.evidence),
        }


# --------------------------------------------------------------------------- #
# Reports returned to the DeepSeek tool layer / viewer
# --------------------------------------------------------------------------- #


@dataclass
class NavEvent:
    """A non-navigation game event detected while driving the emulator."""

    kind: NavEventKind = NavEventKind.NONE
    screen_event: str = "none"
    message: str = ""
    frame_index: int = 0
    position: tuple[int, int] | None = None
    area_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "screen_event": self.screen_event,
            "message": self.message,
            "frame_index": int(self.frame_index),
            "position": None if self.position is None else list(self.position),
            "area_id": self.area_id,
        }


@dataclass
class CoverageStats:
    """Honest per-area exploration coverage counters."""

    area_id: str = ""
    cells_total: int = 0           # known cells (observed+) in this area
    observed: int = 0
    walkable: int = 0
    blocked: int = 0
    temp_blocked: int = 0
    exits: int = 0
    interactive: int = 0
    unknown_frontier: int = 0      # unknown/observed cells bordering reachable area
    unreachable: int = 0
    deferred: int = 0
    coverage_ratio: float = 0.0    # walkable / (walkable + reachable frontier)
    explored_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_id": self.area_id,
            "cells_total": int(self.cells_total),
            "observed": int(self.observed),
            "walkable": int(self.walkable),
            "blocked": int(self.blocked),
            "temp_blocked": int(self.temp_blocked),
            "exits": int(self.exits),
            "interactive": int(self.interactive),
            "unknown_frontier": int(self.unknown_frontier),
            "unreachable": int(self.unreachable),
            "deferred": int(self.deferred),
            "coverage_ratio": round(float(self.coverage_ratio), 4),
            "explored_ratio": round(float(self.explored_ratio), 4),
        }


@dataclass
class ExploreReport:
    """Structured result of an autonomous ``explore_current_area`` run."""

    status: ExploreStatus = ExploreStatus.ERROR
    area_id: str = ""
    reason: str = ""
    steps: int = 0                   # navigation primitive calls (verified actions)
    moves: int = 0
    blocked: int = 0
    turned: int = 0
    unknown: int = 0
    new_cells: int = 0
    new_edges: int = 0
    new_exits: int = 0
    backtracked: int = 0
    coverage: CoverageStats | None = None
    pose: PlayerPose | None = None
    event: NavEvent | None = None
    path: list[tuple[int, int]] = field(default_factory=list)
    started_at: float = 0.0
    elapsed_s: float = 0.0
    frames: int = 0
    area_switched: bool = False
    new_area_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "status_success": self.status.success,
            "area_id": self.area_id,
            "reason": self.reason,
            "steps": int(self.steps),
            "moves": int(self.moves),
            "blocked": int(self.blocked),
            "turned": int(self.turned),
            "unknown": int(self.unknown),
            "new_cells": int(self.new_cells),
            "new_edges": int(self.new_edges),
            "new_exits": int(self.new_exits),
            "backtracked": int(self.backtracked),
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
            "pose": None if self.pose is None else self.pose.to_dict(),
            "event": None if self.event is None else self.event.to_dict(),
            "path": [list(cell) for cell in self.path],
            "started_at": float(self.started_at),
            "elapsed_s": round(float(self.elapsed_s), 3),
            "frames": int(self.frames),
            "area_switched": bool(self.area_switched),
            "new_area_id": self.new_area_id,
        }


@dataclass
class NavReport:
    """Structured result of a ``navigate_to`` run."""

    status: ExploreStatus = ExploreStatus.ERROR
    area_id: str = ""
    reason: str = ""
    target: dict[str, Any] = field(default_factory=dict)
    reached: bool = False
    steps: int = 0
    moves: int = 0
    blocked: int = 0
    replans: int = 0
    pose: PlayerPose | None = None
    coverage: CoverageStats | None = None
    event: NavEvent | None = None
    started_at: float = 0.0
    elapsed_s: float = 0.0
    frames: int = 0
    area_switched: bool = False
    new_area_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "status_success": self.status.success,
            "area_id": self.area_id,
            "reason": self.reason,
            "target": dict(self.target),
            "reached": bool(self.reached),
            "steps": int(self.steps),
            "moves": int(self.moves),
            "blocked": int(self.blocked),
            "replans": int(self.replans),
            "pose": None if self.pose is None else self.pose.to_dict(),
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
            "event": None if self.event is None else self.event.to_dict(),
            "started_at": float(self.started_at),
            "elapsed_s": round(float(self.elapsed_s), 3),
            "frames": int(self.frames),
            "area_switched": bool(self.area_switched),
            "new_area_id": self.new_area_id,
        }


@dataclass
class NavStep:
    """One verified navigation action (kept in the controller's step log)."""

    index: int
    direction: str
    motion: MotionEstimate
    pose_before: tuple[int, int] = (0, 0)
    pose_after: tuple[int, int] = (0, 0)
    area_id: str = ""
    intent: str = ""  # explore|navigate|probe|recover|exit

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "direction": self.direction,
            "intent": self.intent,
            "area_id": self.area_id,
            "pose_before": list(self.pose_before),
            "pose_after": list(self.pose_after),
            "motion": self.motion.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def step_delta(direction: str, tiles: int = 1) -> tuple[int, int]:
    """World delta for ``tiles`` steps in ``direction``."""
    dx, dy = DIRECTION_DELTAS.get(direction, (0, 0))
    return (dx * int(tiles), dy * int(tiles))


def direction_of(delta: tuple[int, int]) -> str | None:
    """Direction matching a unit cardinal delta (or None)."""
    for direction, (dx, dy) in DIRECTION_DELTAS.items():
        if (dx, dy) == (int(delta[0]), int(delta[1])):
            return direction
    return None


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


__all__ = [
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
    "TILE_PX",
    "VIEW_W",
    "VIEW_H",
    "VIEW_PLAYER",
    "PLAYER_BOX",
    "DIRECTIONS",
    "DIRECTION_DELTAS",
    "OPPOSITE",
    "MotionResult",
    "FrameContext",
    "CellState",
    "EdgeState",
    "ExploreStatus",
    "NavEventKind",
    "TileObs",
    "FrameView",
    "PlayerPose",
    "MotionEstimate",
    "PathPlan",
    "ExploreTarget",
    "AreaMatch",
    "NavEvent",
    "CoverageStats",
    "ExploreReport",
    "NavReport",
    "NavStep",
    "step_delta",
    "direction_of",
    "manhattan",
]
