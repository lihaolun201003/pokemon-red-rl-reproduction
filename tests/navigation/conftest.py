"""Shared fixtures for Navigation v2 tests.

The fixtures here allow the navigation modules to be tested **without PyBoy**:
a :class:`SyntheticRoom` renders real 160x144 frames (10x9 tiles of 16x16 px,
player sprite fixed at screen tile (4,4), background shifted by one tile per
successful step) using deterministic pseudo-random tile patterns.

Optional real-ROM fixtures (``rom_path``, ``state_path``, ``require_rom``) let
integration tests skip cleanly when the local ROM/state are absent (they are in
``local/`` which is gitignored).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

TILE = 16
VIEW_W, VIEW_H = 10, 9
PLAYER_SCREEN = (4, 4)

# 4 DMG grey levels -> RGBA
_LEVELS = np.array([0, 85, 153, 255], dtype=np.uint8)

#: direction -> (dx, dy)
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #

def rgba_from_luma(luma: np.ndarray) -> np.ndarray:
    """Expand an (H, W) 0..3 level array to an (H, W, 4) RGBA uint8 image."""
    levels = np.asarray(luma)
    if levels.shape[:2] != (144, 160):
        raise ValueError(f"expected a 144x160 luma map, got {levels.shape}")
    rgba = np.empty((144, 160, 4), dtype=np.uint8)
    grey = _LEVELS[np.clip(levels, 0, 3)]
    rgba[..., :3] = grey[..., None]
    rgba[..., 3] = 255
    return rgba


def luma_from_rgba(rgba: np.ndarray) -> np.ndarray:
    """Convert RGBA (or RGB) pixels back to an (H, W) uint8 luma map."""
    rgb = np.asarray(rgba)[..., :3].astype(np.uint16)
    return ((rgb[..., 0] * 77 + rgb[..., 1] * 150 + rgb[..., 2] * 29) >> 8).astype(np.uint8)


def make_frame(rgba: np.ndarray, frame_index: int = 0):
    """Build a contracts.Frame from raw RGBA pixels."""
    from llm_agent.observation import frame_from_rgba

    return frame_from_rgba(frame_index, np.ascontiguousarray(rgba, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Synthetic room world
# --------------------------------------------------------------------------- #

def _pattern(seed: int, size: int = TILE) -> np.ndarray:
    """Deterministic 4-level 16x16 tile pattern from a seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=(size, size), dtype=np.uint8)


_WALL_PATTERN = np.zeros((TILE, TILE), dtype=np.uint8)
_WALL_PATTERN[1:15, 1:15] = 2
_WALL_PATTERN[2:14, 2:14] = 3
_WALL_PATTERN[7:9, :] = 0
_WALL_PATTERN[:, 7:9] = 0

_PLAYER_PATTERNS = {
    # a small directional marker so turn-only changes the player tile pixels
    "down": np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 2, 2, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 2, 2, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
        ]
    ),
}
_PLAYER_PATTERNS["up"] = np.flipud(_PLAYER_PATTERNS["down"])
_PLAYER_PATTERNS["left"] = np.rot90(_PLAYER_PATTERNS["down"])
_PLAYER_PATTERNS["right"] = np.flipud(_PLAYER_PATTERNS["left"])


def _player_tile(facing: str) -> np.ndarray:
    marker = _PLAYER_PATTERNS[facing]
    tile = np.zeros((TILE, TILE), dtype=np.uint8)
    oy = (TILE - marker.shape[0]) // 2
    ox = (TILE - marker.shape[1]) // 2
    tile[oy : oy + marker.shape[0], ox : ox + marker.shape[1]] = marker
    return tile


class SyntheticRoom:
    """A deterministic grid world rendered as 160x144 GB frames."""

    def __init__(
        self,
        width: int = 10,
        height: int = 9,
        *,
        seed: int = 0,
        walls: Iterable[tuple[int, int]] | None = None,
        exits: Iterable[tuple[int, int]] | None = None,
        animated: Iterable[tuple[int, int]] | None = None,
        border_level: int = 0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.border_level = int(border_level) & 3
        self.walls: set[tuple[int, int]] = set(walls or ())
        self.exits: set[tuple[int, int]] = set(exits or ())
        self.animated: set[tuple[int, int]] = set(animated or ())
        self._tiles: dict[tuple[int, int], np.ndarray] = {}
        self._animation_phase = 0
        for y in range(self.height):
            for x in range(self.width):
                self._tiles[(x, y)] = _pattern(self.seed * 100_003 + x * 811 + y * 977)

    # -- world queries ---------------------------------------------------- #
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self.walls

    def is_exit(self, x: int, y: int) -> bool:
        return (x, y) in self.exits

    def tile_luma(self, x: int, y: int) -> np.ndarray:
        """16x16 luma tile at world (x, y); out-of-bounds -> border."""
        if not self.in_bounds(x, y):
            return np.full((TILE, TILE), self.border_level, dtype=np.uint8)
        if (x, y) in self.walls:
            return _WALL_PATTERN.copy()
        base = self._tiles[(x, y)]
        if (x, y) in self.animated:
            # animated tiles alternate between two visuals (water/flowers)
            shift = self._animation_phase % 2
            return np.roll(base, shift * 3, axis=1)
        return base

    def advance_animation(self) -> None:
        self._animation_phase += 1

    def render(self, pose: tuple[int, int], facing: str = "down") -> np.ndarray:
        """Render the player-centred 160x144 frame as RGBA."""
        px, py = int(pose[0]), int(pose[1])
        if facing not in _PLAYER_PATTERNS:
            facing = "down"
        luma = np.zeros((144, 160), dtype=np.uint8)
        for ty in range(VIEW_H):
            for tx in range(VIEW_W):
                wx = px + tx - PLAYER_SCREEN[0]
                wy = py + ty - PLAYER_SCREEN[1]
                luma[ty * TILE : (ty + 1) * TILE, tx * TILE : (tx + 1) * TILE] = self.tile_luma(wx, wy)
        # draw the player sprite on top of screen tile (4,4)
        sx, sy = PLAYER_SCREEN
        luma[sy * TILE : (sy + 1) * TILE, sx * TILE : (sx + 1) * TILE] = _player_tile(facing)
        return rgba_from_luma(luma)

    def render_numbered(self, pose: tuple[int, int], facing: str = "down") -> np.ndarray:
        """Same as render but player tile carries a distinctive marker (debug)."""
        return self.render(pose, facing)


class SyntheticEmulator:
    """Minimal emulator stand-in for controller tests.

    Implements the subset of :class:`llm_agent.emulator.GameBoyEmulator` that the
    navigation controller uses: ``observe()``, ``press()``, ``wait()``,
    ``frame_count``, ``started``.  It understands walkable/blocked/turn-only and
    exit transposition via :meth:`walk_through_exit_to`.
    """

    def __init__(self, room: SyntheticRoom, pose: tuple[int, int] = (1, 1),
                 facing: str = "down", *, frame_index: int = 0) -> None:
        self.room = room
        self.pose = (int(pose[0]), int(pose[1]))
        self.facing = facing if facing in DELTAS else "down"
        self._frame = 0
        self._last = room.render(self.pose, self.facing)
        self._frame_count = int(frame_index)
        self.started = True
        #: optional (exit_cell, direction) -> (new_room, new_pose) mapping
        self.warps: dict[tuple[tuple[int, int], str], tuple[SyntheticRoom, tuple[int, int]]] = {}
        self.presses: list[str] = []
        self.blocked_attempts = 0
        self.transition_pending = False

    # -- emulator surface -------------------------------------------------- #
    @property
    def frame_count(self) -> int:
        return self._frame_count

    def observe(self):
        return self._frame_obj()

    def wait(self, frames: int = 1):
        self._frame_count += max(0, int(frames))
        self.room.advance_animation()
        return self._frame_obj()

    def press(self, button: str, frames: int | None = None):
        self._frame_count += 24
        button = str(button).lower()
        self.presses.append(button)
        if button not in DELTAS:
            return self._frame_obj()
        dx, dy = DELTAS[button]
        target = (self.pose[0] + dx, self.pose[1] + dy)
        want_face = button
        if button in self.warps or (want_face, self.pose) in self.warps:
            pass
        warp = self.warps.get((self.pose, button)) or self.warps.get((target, button))
        if warp is not None:
            self.room, new_pose = warp
            self.pose = (int(new_pose[0]), int(new_pose[1]))
            self.facing = button
            self.transition_pending = True
            self._last = self.room.render(self.pose, self.facing)
            return self._frame_obj()
        if self.room.is_walkable(*target):
            self.pose = target
            self.facing = button
        else:
            self.blocked_attempts += 1
            self.facing = button
        self.room.advance_animation()
        self._last = self.room.render(self.pose, self.facing)
        return self._frame_obj()

    # -- helpers ----------------------------------------------------------- #
    def _frame_obj(self):
        from llm_agent.observation import frame_from_rgba

        return frame_from_rgba(self._frame_count, self._last)

    def set_pose(self, pose: tuple[int, int], facing: str = "down") -> None:
        self.pose = (int(pose[0]), int(pose[1]))
        self.facing = facing
        self._last = self.room.render(self.pose, self.facing)

    @property
    def last_frame(self):
        return self._frame_obj()


def frame_view_from_frame(frame, *, context=None):
    """Build a :class:`FrameView` from a Frame using the vision primitives."""
    from llm_agent.navigation import vision
    from llm_agent.navigation.types import FrameContext, FrameView

    luma = vision.frame_luma(frame)
    offset = vision.tile_grid_offset(luma)
    return FrameView(
        frame_index=frame.frame_index,
        signature=vision.screen_signature(luma),
        grid_offset=offset,
        tiles=vision.extract_tiles(luma, offset),
        context=context or FrameContext.OVERWORLD,
        is_black=vision.is_black(luma),
        luma_mean=float(luma.mean()),
    )


def frame_view_from_parts(
    frame_index: int = 0,
    *,
    signature: str = "0" * 32,
    tiles=None,
    context=None,
):
    """Build a FrameView from explicit parts (for tests that do not render)."""
    from llm_agent.navigation.types import FrameContext, FrameView

    return FrameView(
        frame_index=int(frame_index),
        signature=signature,
        grid_offset=(0, 0),
        tiles=list(tiles or []),
        context=context or FrameContext.OVERWORLD,
    )


# --------------------------------------------------------------------------- #
# pytest fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def rom_path() -> Path:
    return ROOT / "local" / "PokemonRed.gb"


@pytest.fixture(scope="session")
def state_path() -> Path:
    return ROOT / "local" / "has_pokedex_nballs.state"


@pytest.fixture(scope="session")
def require_rom(rom_path: Path, state_path: Path) -> tuple[Path, Path]:
    if not rom_path.is_file() or not state_path.is_file():
        pytest.skip("local ROM/save state not available")
    return rom_path, state_path


@pytest.fixture()
def simple_room() -> SyntheticRoom:
    """5x5 room, walls around the player's start, (1,1) is the player."""
    walls = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
        (1, 0), (2, 0), (3, 0),
        (1, 4), (2, 4), (3, 4),
    }
    return SyntheticRoom(width=5, height=5, walls=walls, exits={(2, 4)})


@pytest.fixture()
def open_room() -> SyntheticRoom:
    """Empty 10x9 room with black border and an exit below the start cell."""
    return SyntheticRoom(width=10, height=9, exits={(3, 8)})


@pytest.fixture()
def fake_emulator(open_room: SyntheticRoom) -> SyntheticEmulator:
    return SyntheticEmulator(open_room, pose=(5, 4), facing="down")
