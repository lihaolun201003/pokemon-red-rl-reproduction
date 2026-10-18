"""Pure pixel/tile primitives for Navigation v2 (no PyBoy, no RAM).

Everything here operates on the 160x144 Game Boy framebuffer (RGBA or a
pre-computed luma array).  The functions are deterministic, allocation-light and
covered by unit tests with synthetic images.

Facts verified on real frames of this repository's ROM (see
``docs/reports/navigation_v2_design.md``):

* The screen is exactly 10x9 tiles of 16x16 px; the tile lattice is pixel
  aligned (the camera scrolls by whole tiles: a step moves the background 16 px).
* A successful step shifts the background content by one tile; the player sprite
  stays at screen tile (4,4).  A blocked press that also changes facing only
  changes pixels inside the player sprite box; a blocked press facing the same
  way is pixel-identical.
* ``best_shift()`` on a real one-tile step returns shift == -direction with a
  match score of about 0.98-0.993 (NPC animation lowers it slightly); a failed
  bump scores 1.0 at shift (0,0).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from llm_agent.contracts import Frame
from llm_agent.navigation.types import (
    PLAYER_BOX,
    TILE_PX,
    VIEW_H,
    VIEW_PLAYER,
    VIEW_W,
    TileObs,
)

#: DMG renders exactly four grey levels; luma at/below this counts as "black"
BLACK_LUMA: int = 8
#: a frame is black when at least this fraction of pixels is black
BLACK_FRACTION: float = 0.98

_LUMA_R, _LUMA_G, _LUMA_B = 77, 150, 29

#: candidate shifts are cardinal only: Gen-1 overworld movement is single-axis
_CARDINAL: tuple[tuple[int, int], ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------- #
# Luma / frame helpers
# --------------------------------------------------------------------------- #

def _rgba(frame: Frame | np.ndarray) -> np.ndarray:
    rgba = frame.rgba if isinstance(frame, Frame) else frame
    if not isinstance(rgba, np.ndarray):
        raise TypeError(f"expected Frame or numpy array, got {type(rgba)!r}")
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError(f"expected (H, W, 3+) pixels, got {rgba.shape}")
    return rgba


def luma_from_rgba(rgba: np.ndarray) -> np.ndarray:
    """Return an (H, W) uint8 luma map (BT.601 integer approximation)."""
    pixels = _rgba(rgba)[..., :3].astype(np.uint16)
    return (
        (pixels[..., 0] * _LUMA_R + pixels[..., 1] * _LUMA_G + pixels[..., 2] * _LUMA_B) >> 8
    ).astype(np.uint8)


def frame_luma(frame: Frame | np.ndarray) -> np.ndarray:
    """Luma map for a :class:`~llm_agent.contracts.Frame` or RGBA array."""
    rgba = _rgba(frame)
    if rgba.shape[0] != 144 or rgba.shape[1] != 160:
        raise ValueError(f"expected a 144x160 screen buffer, got {rgba.shape}")
    return luma_from_rgba(rgba)


def is_black(luma: np.ndarray) -> bool:
    """True when >= 98 % of the pixels are black (fade / transition screen)."""
    return float((np.asarray(luma) <= BLACK_LUMA).mean()) >= BLACK_FRACTION


def tile_grid_offset(luma: np.ndarray) -> tuple[int, int]:
    """Pixel offset of the 16 px tile lattice inside the 160x144 screen.

    The overworld camera keeps the player sprite at a fixed *tile* position and
    scrolls by whole tiles, so the lattice phase is constant (0, 0) for every
    frame of every area; this helper exists so the assumption can be probed and
    adapted by tests/tooling without changing call sites.
    """
    arr = np.asarray(luma)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D luma array, got {arr.shape}")
    # Sanity check: a full 16 px shift must be a good self-match when the lattice
    # is pixel aligned; otherwise report the best phase measured by 16 px
    # self-correlation (kept cheap: only used at area-entry time).
    height, width = arr.shape
    if height < 2 * TILE_PX or width < 2 * TILE_PX:
        return (0, 0)
    best_phase = (0, 0)
    best_score = -1.0
    for oy in (0, 4, 8, 12):
        for ox in (0, 4, 8, 12):
            # Compare the frame with itself shifted by exactly one tile while
            # ignoring the phase-dependent border strip.
            a = arr[oy : height - TILE_PX, ox : width - TILE_PX]
            b = arr[oy + TILE_PX : height, ox + TILE_PX : width]
            if a.size == 0:
                continue
            score = float((a == b).mean())
            if score > best_score:
                best_score = score
                best_phase = (ox, oy)
    return best_phase


# --------------------------------------------------------------------------- #
# Tile extraction / hashing
# --------------------------------------------------------------------------- #

def tile_hash(tile: np.ndarray) -> str:
    """Stable 16-char hex digest of one 16x16 luma tile."""
    array = np.ascontiguousarray(np.asarray(tile, dtype=np.uint8))
    return hashlib.blake2b(array.tobytes(), digest_size=8).hexdigest()


def extract_tiles(
    luma: np.ndarray,
    offset: tuple[int, int] = (0, 0),
) -> list[TileObs]:
    """Extract the 10x9 visible tiles (row-major) as :class:`TileObs` records."""
    arr = np.asarray(luma)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D luma array, got {arr.shape}")
    ox, oy = int(offset[0]) % TILE_PX, int(offset[1]) % TILE_PX
    tiles: list[TileObs] = []
    for ty in range(VIEW_H):
        y0 = oy + ty * TILE_PX
        for tx in range(VIEW_W):
            x0 = ox + tx * TILE_PX
            tile = arr[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
            if tile.shape != (TILE_PX, TILE_PX):
                # Out-of-frame tiles are impossible on a 160x144 buffer, but keep
                # the function total for synthetic / odd inputs.
                padded = np.zeros((TILE_PX, TILE_PX), dtype=np.uint8)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            tiles.append(
                TileObs(
                    tx=tx,
                    ty=ty,
                    tile_hash=tile_hash(tile),
                    luma=np.ascontiguousarray(tile, dtype=np.uint8).tobytes(),
                )
            )
    return tiles


def tile_mean_luma(tile_obs: TileObs) -> float:
    """Mean luma of a :class:`TileObs` (uses stored raw bytes when present)."""
    if tile_obs.luma:
        return float(np.frombuffer(tile_obs.luma, dtype=np.uint8).mean())
    return 0.0


# --------------------------------------------------------------------------- #
# Shift estimation (background scroll <-> movement)
# --------------------------------------------------------------------------- #

@dataclass
class ShiftScore:
    """Result of :func:`best_shift`."""

    shift: tuple[int, int] = (0, 0)
    score: float = 0.0
    second_shift: tuple[int, int] | None = None
    second_score: float = 0.0
    scores: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        return float(self.score - self.second_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shift": list(self.shift),
            "score": round(float(self.score), 4),
            "second_shift": None if self.second_shift is None else list(self.second_shift),
            "second_score": round(float(self.second_score), 4),
            "margin": round(self.margin, 4),
        }


def match_shift(
    prev_luma: np.ndarray,
    cur_luma: np.ndarray,
    sx: int,
    sy: int,
) -> float:
    """Fraction of equal pixels between ``cur`` and ``prev`` moved by (sx, sy) tiles.

    The comparison is over the overlapping region only, so newly exposed border
    tiles (which the game renders deterministically) do not have to be padded.
    A shift of ``(sx, sy)`` means "the background content moved by (sx, sy) tiles
    between prev and cur", i.e. ``cur(x, y) == prev(x - sx*16, y - sy*16)``.
    """
    prev = np.asarray(prev_luma)
    cur = np.asarray(cur_luma)
    if prev.shape != cur.shape:
        raise ValueError(f"cannot compare frames of different shapes: {prev.shape} vs {cur.shape}")
    height, width = prev.shape[:2]
    dx_px, dy_px = int(sx) * TILE_PX, int(sy) * TILE_PX
    if abs(dx_px) >= width or abs(dy_px) >= height:
        return 0.0
    ax0, ax1 = max(0, -dx_px), min(width, width - dx_px)
    ay0, ay1 = max(0, -dy_px), min(height, height - dy_px)
    bx0, bx1 = max(0, dx_px), min(width, width + dx_px)
    by0, by1 = max(0, dy_px), min(height, height + dy_px)
    first = prev[ay0:ay1, ax0:ax1]
    second = cur[by0:by1, bx0:bx1]
    if first.size == 0:
        return 0.0
    return float((first == second).mean())


def candidate_shifts(max_tiles: int = 2) -> list[tuple[int, int]]:
    """Cardinal shift candidates up to ``max_tiles`` tiles (plus no movement)."""
    limit = max(0, int(max_tiles))
    shifts: list[tuple[int, int]] = list(_CARDINAL)
    if limit >= 2:
        shifts.extend([(2, 0), (-2, 0), (0, 2), (0, -2)])
    if limit >= 3:
        shifts.extend([(3, 0), (-3, 0), (0, 3), (0, -3)])
    # deterministic order, no duplicates
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for shift in shifts:
        if shift not in seen:
            seen.add(shift)
            unique.append(shift)
    return unique


def best_shift(
    prev_luma: np.ndarray,
    cur_luma: np.ndarray,
    *,
    max_tiles: int = 2,
) -> ShiftScore:
    """Best background shift between two frames (see :func:`match_shift`)."""
    scores: dict[tuple[int, int], float] = {}
    for shift in candidate_shifts(max_tiles):
        scores[shift] = match_shift(prev_luma, cur_luma, shift[0], shift[1])
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_shift_value, best_score = ranked[0]
    second_shift = ranked[1][0] if len(ranked) > 1 else None
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return ShiftScore(
        shift=best_shift_value,
        score=best_score,
        second_shift=second_shift,
        second_score=second_score,
        scores=scores,
    )


def shift_image(luma: np.ndarray, sx: int, sy: int) -> np.ndarray:
    """Return ``luma`` shifted by (sx, sy) *pixels* (zero-filled, for tests)."""
    array = np.asarray(luma)
    height, width = array.shape[:2]
    dx, dy = int(sx), int(sy)
    out = np.zeros_like(array)
    if abs(dx) >= width or abs(dy) >= height:
        return out
    ys = slice(max(0, -dy), min(height, height - dy))
    yd = slice(max(0, dy), min(height, height + dy))
    xs = slice(max(0, -dx), min(width, width - dx))
    xd = slice(max(0, dx), min(width, width + dx))
    out[yd, xd] = array[ys, xs]
    return out


# --------------------------------------------------------------------------- #
# Frame differences (turn / blocked / interruption evidence)
# --------------------------------------------------------------------------- #

def diff_mask(prev_luma: np.ndarray, cur_luma: np.ndarray) -> np.ndarray:
    """Boolean pixel-difference mask between two equal-shaped luma arrays."""
    prev = np.asarray(prev_luma)
    cur = np.asarray(cur_luma)
    if prev.shape != cur.shape:
        raise ValueError(f"cannot diff frames of different shapes: {prev.shape} vs {cur.shape}")
    return prev != cur


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))


def box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Boolean mask of ``box`` (x0, y0, x1, y1) inside ``shape`` (H, W)."""
    height, width = shape
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
    return mask


def player_box_mask(shape: tuple[int, int] = (144, 160)) -> np.ndarray:
    """Mask of the player sprite box (generous, includes animation overshoot)."""
    return box_mask(shape, PLAYER_BOX)


def diff_stats(prev_luma: np.ndarray, cur_luma: np.ndarray) -> dict[str, Any]:
    """Summarise where two frames differ (turn vs blocked vs interruption)."""
    mask = diff_mask(prev_luma, cur_luma)
    total = int(mask.sum())
    if total == 0:
        return {
            "pixels": 0,
            "ratio": 0.0,
            "bbox": None,
            "inside_player": 0,
            "outside_player": 0,
            "inside_ratio": 0.0,
            "components": 0,
        }
    box = player_box_mask(mask.shape)
    inside = int((mask & box).sum())
    outside = total - inside
    return {
        "pixels": total,
        "ratio": round(total / float(mask.size), 5),
        "bbox": _bounding_box(mask),
        "inside_player": inside,
        "outside_player": outside,
        "inside_ratio": round(inside / float(total), 4),
        "components": _component_count(mask),
    }


def _component_count(mask: np.ndarray, *, min_pixels: int = 4) -> int:
    """Count 4-connected changed components with at least ``min_pixels`` pixels."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for y in range(height):
        row = mask[y]
        for x in range(width):
            if not row[x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if size >= min_pixels:
                count += 1
    return count


def changed_tiles(
    prev_luma: np.ndarray,
    cur_luma: np.ndarray,
    *,
    min_fraction: float = 0.10,
) -> list[tuple[int, int, float]]:
    """Per-tile change fractions for tiles whose luma changed enough."""
    mask = diff_mask(prev_luma, cur_luma)
    height, width = mask.shape
    changed: list[tuple[int, int, float]] = []
    for ty in range(min(VIEW_H, height // TILE_PX)):
        for tx in range(min(VIEW_W, width // TILE_PX)):
            tile = mask[ty * TILE_PX : (ty + 1) * TILE_PX, tx * TILE_PX : (tx + 1) * TILE_PX]
            fraction = float(tile.mean())
            if fraction >= min_fraction:
                changed.append((tx, ty, round(fraction, 4)))
    return changed


# --------------------------------------------------------------------------- #
# Screen signature / stability helpers
# --------------------------------------------------------------------------- #

def screen_signature(luma: np.ndarray) -> str:
    """Fast 16x16-layout perceptual signature (stable across small sprite changes)."""
    array = np.asarray(luma)
    height, width = array.shape
    if height < 16 or width < 16:
        return hashlib.blake2b(array.tobytes(), digest_size=8).hexdigest()
    # Split into 16x16 cells and compare every cell mean with the global mean.
    rows = np.array_split(array.astype(np.float32), 16, axis=0)
    blocks = np.empty((16, 16), dtype=np.float32)
    for index, row in enumerate(rows):
        cols = np.array_split(row, 16, axis=1)
        blocks[index] = [float(col.mean()) for col in cols]
    bits = (blocks > blocks.mean()).reshape(-1)
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def same_screen(a: np.ndarray, b: np.ndarray, *, max_diff_ratio: float = 0.004) -> bool:
    """True when two luma frames are effectively identical (no visible change)."""
    return float(diff_mask(a, b).mean()) <= max_diff_ratio


def relative_tiles(tiles: Iterable[TileObs]) -> dict[tuple[int, int], TileObs]:
    """Map screen tiles to player-relative positions."""
    return {
        (obs.tx - VIEW_PLAYER[0], obs.ty - VIEW_PLAYER[1]): obs for obs in tiles
    }


__all__ = [
    "BLACK_LUMA",
    "BLACK_FRACTION",
    "ShiftScore",
    "frame_luma",
    "luma_from_rgba",
    "is_black",
    "tile_grid_offset",
    "tile_hash",
    "extract_tiles",
    "tile_mean_luma",
    "match_shift",
    "candidate_shifts",
    "best_shift",
    "shift_image",
    "diff_mask",
    "diff_stats",
    "changed_tiles",
    "box_mask",
    "player_box_mask",
    "screen_signature",
    "same_screen",
    "relative_tiles",
]
