"""Map / world / status visualisation for Navigation v2 (numpy + Pillow only).

This module is deliberately **pure rendering**: it never imports Tk, PyBoy, the
game ROM, network libraries or any other navigation module at runtime.  Its only
inputs are the data objects described by ``NAV_CONTRACT.md`` (``GridMap``,
``WorldModel``, ``PlayerPose``, ``CoverageStats`` and the ``NavigationAgent.status()``
dictionary) plus, optionally, a raw 160x144 RGBA frame.

Everything is **duck-typed and tolerant**: a real :class:`GridMap` /
:class:`WorldModel` is used when available, but any object with the documented
shape (``.cells`` mapping, ``.bounds()``, ``.coverage()``, ``.areas``, …) works
too.  Missing keys never raise; they render as ``"-"`` placeholders.  The output
is deterministic (identical bytes for identical input) and cheap enough for the
Tk viewer (~10 fps with a few hundred cells).

Public API (frozen by ``NAV_CONTRACT.md`` section 7)::

    render_map_image(grid, *, pose=None, path=None, target=None,
                     cell_px=10, max_cells=64, legend=True) -> np.ndarray
    render_world_image(world, *, width=320, height=160) -> np.ndarray
    render_status_panel(status, *, width=320, height=160) -> np.ndarray
    composite_viewer(state, *, frame_rgba=None, width=480, height=360) -> np.ndarray
    coverage_bar(coverage, *, width=120) -> np.ndarray
    describe_cell(cell) -> str

All image functions return ``(h, w, 4)`` uint8 **RGBA** arrays with an opaque
alpha channel.  The palette is fixed and small (see :data:`PALETTE`): walkable
green, observed grey-blue, blocked red, temp_blocked orange, exit yellow,
interactive cyan, unknown dark, player white, path magenta and a cyan target
ring.  Unknown/blocked cells are always drawn; the map image is a window of at
most ``max_cells`` cells per axis centred on the player (whole grid when it is
smaller), so a few hundred cells stay cheap.

Wiring ``composite_viewer`` into the Tk viewer (owner: nav-lead)::

    from llm_agent.navigation import viz

    state = {
        # all keys optional; missing data renders as placeholders
        "grid": grid,                 # GridMap-like (needs .cells)
        "pose": pose,                 # PlayerPose | {"x","y",...} | (x, y)
        "path": plan.cells,           # list[(x, y)] (magenta)
        "target": (x, y),             # target cell (cyan ring)
        "status": agent.status(),     # dict from NavigationAgent.status()
        "world": world,               # WorldModel-like (only counts used)
        "pause": False,               # or "paused": bool
        "frame_index": 1234,
        "last_action": "press_up",    # optional bottom-strip note
    }
    rgba = viz.composite_viewer(state, frame_rgba=frame.rgba)   # (360, 480, 4)
    image = Image.fromarray(rgba)          # RGBA PIL image for Tk (needs .convert("RGB")?)
    photo = ImageTk.PhotoImage(Image.fromarray(rgba[..., :3]))   # RGB for Tk labels

The game frame is drawn scaled 2x in the top-left (320x288); the map panel sits
top-right and the status panel below it; the bottom strip carries a coverage bar
and a compact text summary.  Without ``frame_rgba`` the map panel gets the whole
left half instead ("panels only" layout).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from llm_agent.navigation.types import CellState, CoverageStats, PlayerPose

if TYPE_CHECKING:  # pragma: no cover - typing only, modules owned by other agents
    from llm_agent.navigation.grid_map import GridMap
    from llm_agent.navigation.world_model import WorldModel

__all__ = [
    "PALETTE",
    "CELL_STATE_COLORS",
    "LEGEND_ITEMS",
    "MAP_BACKGROUND",
    "TARGET_RING_COLOR",
    "render_map_image",
    "render_world_image",
    "render_status_panel",
    "composite_viewer",
    "coverage_bar",
    "describe_cell",
    "cell_color",
]

# --------------------------------------------------------------------------- #
# Fixed palette
# --------------------------------------------------------------------------- #

#: Fixed RGB palette shared by every renderer (never random, never anti-aliased
#: for cell fills, so tests can assert exact pixel colours).
PALETTE: dict[str, tuple[int, int, int]] = {
    # CellState colours -----------------------------------------------------
    "walkable": (74, 196, 112),      # green: verified passable
    "observed": (105, 138, 168),     # grey-blue: seen but not walked
    "blocked": (214, 74, 74),        # red: persistent obstacle
    "temp_blocked": (232, 150, 56),  # orange: dynamic obstacle (NPC/object)
    "exit": (240, 214, 74),          # yellow: suspected/known area exit
    "interactive": (86, 214, 214),   # cyan: sign / NPC / item candidate
    "unknown": (34, 38, 45),         # dark: never observed
    "unreachable": (124, 92, 104),   # muted plum: cannot be reached
    "deferred": (96, 112, 96),       # muted moss: reachable but skipped
    # Overlays --------------------------------------------------------------
    "player": (255, 255, 255),       # white player marker
    "path": (221, 91, 208),          # magenta planned path
    "target": (128, 246, 255),       # cyan target ring
    # Chrome / text ---------------------------------------------------------
    "background": (16, 18, 22),
    "panel": (24, 27, 33),
    "panel_alt": (31, 36, 43),
    "panel_border": (66, 74, 86),
    "grid_line": (45, 50, 58),
    "text": (232, 238, 244),
    "text_dim": (154, 164, 176),
    "text_faint": (108, 118, 130),
    "coverage_fill": (74, 196, 112),
    "coverage_bg": (41, 46, 54),
    "paused": (240, 194, 64),
    "warning": (232, 150, 56),
    "current_area": (96, 220, 255),
    "connection_walk": (116, 198, 140),
    "connection_warp": (116, 176, 220),
    "connection_unknown": (132, 136, 146),
}

#: Convenience alias for the background colour of every canvas.
MAP_BACKGROUND: tuple[int, int, int] = PALETTE["background"]
#: Alias for the target-ring colour (cyan).
TARGET_RING_COLOR: tuple[int, int, int] = PALETTE["target"]

#: ``CellState`` value -> RGB colour.  ``UNKNOWN`` is also the "no cell" fill.
CELL_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    CellState.UNKNOWN.value: PALETTE["unknown"],
    CellState.OBSERVED.value: PALETTE["observed"],
    CellState.WALKABLE.value: PALETTE["walkable"],
    CellState.BLOCKED.value: PALETTE["blocked"],
    CellState.TEMP_BLOCKED.value: PALETTE["temp_blocked"],
    CellState.EXIT.value: PALETTE["exit"],
    CellState.INTERACTIVE.value: PALETTE["interactive"],
    CellState.UNREACHABLE.value: PALETTE["unreachable"],
    CellState.DEFERRED.value: PALETTE["deferred"],
}

#: ``(short label, PALETTE key)`` pairs for the map legend, in drawing order.
LEGEND_ITEMS: tuple[tuple[str, str], ...] = (
    ("walk", "walkable"),
    ("seen", "observed"),
    ("wall", "blocked"),
    ("temp", "temp_blocked"),
    ("exit", "exit"),
    ("obj", "interactive"),
    ("unk", "unknown"),
    ("unr", "unreachable"),
    ("def", "deferred"),
    ("you", "player"),
    ("path", "path"),
    ("tgt", "target"),
)

_LEGEND_MIN_WIDTH = 240
_LEGEND_FONT_SIZE = 8
_LEGEND_PAD = 3
_LEGEND_SWATCH = 7

_STATUS_KEYS: tuple[str, ...] = (
    "area_id",
    "area_kind",
    "pose",
    "cells_total",
    "walkable",
    "blocked",
    "temp_blocked",
    "unknown_frontier",
    "coverage_ratio",
    "coverage",
    "areas_total",
    "last_event",
    "paused",
    "world",
    "frame_index",
)

# Composite viewer geometry (designed at 480x360; other sizes are rescaled).
_COMPOSITE_W = 480
_COMPOSITE_H = 360
_GAME_W = 320
_GAME_H = 288
_MARGIN = 4
_FRAME_MAP_BOX = (328, 4, 148, 160)
_FRAME_STATUS_BOX = (328, 168, 148, 124)
_NOFRAME_MAP_BOX = (4, 4, 320, 288)
_NOFRAME_STATUS_BOX = (328, 4, 148, 288)
_BOTTOM_BOX = (4, 296, 472, 60)

_STATE_ALIASES: dict[str, str] = {
    "tempblocked": "temp_blocked",
    "temp-blocked": "temp_blocked",
    "passable": "walkable",
    "free": "walkable",
    "open": "walkable",
    "seen": "observed",
    "wall": "blocked",
    "obstacle": "blocked",
    "door": "exit",
    "warp": "exit",
}

# --------------------------------------------------------------------------- #
# Small typing aliases (input tolerance: real objects or plain mappings)
# --------------------------------------------------------------------------- #

_PoseLike = PlayerPose | Mapping[str, Any] | tuple[int, int] | list[int] | None
_CellCoord = tuple[int, int]

# --------------------------------------------------------------------------- #
# Fonts / text helpers
# --------------------------------------------------------------------------- #

_FONTS: dict[int, Any] = {}


def _font(size: int = 9) -> Any:
    """Cached PIL default font (no font files required, deterministic)."""
    size = int(size)
    cached = _FONTS.get(size)
    if cached is not None:
        return cached
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow (bitmap default font)
        font = ImageFont.load_default()
    _FONTS[size] = font
    return font


def _line_height(font: Any) -> int:
    """Height of one text line for ``font`` (fallback: bbox of "Ay")."""
    getmetrics = getattr(font, "getmetrics", None)
    if callable(getmetrics):
        try:
            ascent, descent = getmetrics()
            return max(1, int(ascent) + int(descent))
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        bbox = font.getbbox("Ay")
        return max(1, int(bbox[3]) - int(bbox[1]))
    except Exception:  # pragma: no cover - defensive
        return 11


def _measure_text(text: str, font: Any) -> int:
    """Pixel width of ``text`` for ``font`` (cached fonts, no draw object needed)."""
    if not text:
        return 0
    getlength = getattr(font, "getlength", None)
    if callable(getlength):
        try:
            return int(math.ceil(float(getlength(text))))
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        bbox = font.getbbox(text)
        return int(bbox[2]) - int(bbox[0])
    except Exception:  # pragma: no cover - defensive
        return 6 * len(text)


def _clip_text(text: Any, font: Any, max_width: int) -> str:
    """Clip ``text`` to ``max_width`` pixels, appending ``".."`` when truncated."""
    value = "" if text is None else str(text)
    max_width = int(max_width)
    if max_width <= 0 or not value:
        return ""
    if _measure_text(value, font) <= max_width:
        return value
    ellipsis = ".."
    budget = max_width - _measure_text(ellipsis, font)
    if budget <= 0:
        return ""
    lo, hi = 0, len(value)
    while lo < hi:  # binary search on characters to keep the string short
        mid = (lo + hi + 1) // 2
        if _measure_text(value[:mid], font) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return value[:lo] + ellipsis


# --------------------------------------------------------------------------- #
# Generic accessors (duck typing; never raise on missing data)
# --------------------------------------------------------------------------- #

def _palette(name: str) -> tuple[int, int, int]:
    return PALETTE.get(name, PALETTE["unknown"])


def _to_rgba(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(h, w, 3)`` uint8 array to a contiguous opaque RGBA array."""
    pixels = np.asarray(rgb)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError(f"expected an (h, w, 3+) RGB array, got shape {pixels.shape}")
    out = np.empty(pixels.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = pixels[..., :3]
    out[..., 3] = 255
    return out


def _as_uint8_rgb(pixels: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def _normalise_frame(frame: Any) -> np.ndarray | None:
    """Accept ``Frame``/ndarray/None and return an RGB ``(h, w, 3)`` uint8 array."""
    if frame is None:
        return None
    rgba = getattr(frame, "rgba", frame)
    if rgba is None:
        return None
    arr = np.asarray(rgba)
    if arr.ndim == 2:  # tolerate a greyscale screen dump
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected a frame with (h, w, 3+) pixels, got shape {arr.shape}")
    return _as_uint8_rgb(arr)


def _mapping_or_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or object attribute (``None``-safe)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _key_xy(key: Any) -> _CellCoord | None:
    """Normalise a ``(x, y)``-like key to an int tuple (or ``None``)."""
    if isinstance(key, (tuple, list)) and len(key) >= 2:
        try:
            return (int(key[0]), int(key[1]))
        except (TypeError, ValueError):
            return None
    return None


def _cell_xy(cell: Any) -> _CellCoord | None:
    """Normalise a cell-like object to ``(x, y)`` (``None`` when unavailable)."""
    if cell is None:
        return None
    if isinstance(cell, (tuple, list)):
        return _key_xy(cell)
    if isinstance(cell, np.ndarray):  # e.g. array([x, y])
        if cell.ndim == 1 and cell.size >= 2:
            return _key_xy(cell.tolist())
        return None
    if isinstance(cell, Mapping):
        x, y = cell.get("x"), cell.get("y")
    else:
        x, y = getattr(cell, "x", None), getattr(cell, "y", None)
        if x is None or y is None:
            maybe = getattr(cell, "cell", None)  # PlayerPose.cell property
            if callable(maybe):
                try:
                    return _key_xy(maybe())
                except Exception:  # pragma: no cover - defensive
                    return None
            if maybe is not None:
                return _key_xy(maybe)
    try:
        if x is None or y is None:
            return None
        return (int(x), int(y))
    except (TypeError, ValueError):
        return None


def _pose_xy(pose: Any) -> _CellCoord | None:
    """Extract the ``(x, y)`` cell of a pose-like value (``None`` if unknown)."""
    if pose is None:
        return None
    if isinstance(pose, (tuple, list, np.ndarray)):
        return _cell_xy(pose)
    return _cell_xy(pose)


def _pose_facing(pose: Any) -> str | None:
    value = _mapping_or_attr(pose, "facing", None)
    return None if value is None else str(value)


def _pose_confidence(pose: Any) -> float | None:
    value = _mapping_or_attr(pose, "confidence", None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pose_valid(pose: Any) -> bool | None:
    value = _mapping_or_attr(pose, "valid", None)
    return None if value is None else bool(value)


def _cells_of(grid: Any) -> dict[_CellCoord, Any]:
    """Return ``{(x, y): cell}`` for any GridMap-like object (never raises)."""
    if grid is None:
        return {}
    raw = getattr(grid, "cells", None)
    if raw is None and isinstance(grid, Mapping):
        raw = grid.get("cells")
    if raw is None:
        raw = getattr(grid, "cells_dict", None)
    result: dict[_CellCoord, Any] = {}
    if isinstance(raw, Mapping):
        for key, cell in raw.items():
            xy = _key_xy(key) or _cell_xy(cell)
            if xy is not None:
                result[xy] = cell
        return result
    if isinstance(raw, (list, tuple)):
        for cell in raw:
            xy = _cell_xy(cell)
            if xy is not None:
                result[xy] = cell
    return result


def _grid_area_id(grid: Any) -> str:
    value = _mapping_or_attr(grid, "area_id", "")
    return "" if value is None else str(value)


def _grid_bounds(grid: Any, cells: Mapping[_CellCoord, Any]) -> tuple[int, int, int, int] | None:
    """``(minx, miny, maxx, maxy)`` preferring ``grid.bounds()``, else the cells."""
    bounds_fn = getattr(grid, "bounds", None)
    if callable(bounds_fn):
        try:
            raw = bounds_fn()
            values = [int(value) for value in tuple(raw)[:4]]
            if len(values) == 4:
                return (values[0], values[1], values[2], values[3])
        except Exception:  # pragma: no cover - defensive fallback
            pass
    if isinstance(grid, Mapping):
        raw = grid.get("bounds")
        if isinstance(raw, (tuple, list)) and len(raw) >= 4:
            try:
                return tuple(int(v) for v in tuple(raw)[:4])  # type: ignore[return-value]
            except (TypeError, ValueError):
                pass
    if not cells:
        return None
    xs = [xy[0] for xy in cells]
    ys = [xy[1] for xy in cells]
    return (min(xs), min(ys), max(xs), max(ys))


def _grid_coverage(grid: Any) -> Any:
    """Call ``grid.coverage()`` defensively (``CoverageStats`` or mapping)."""
    if grid is None:
        return None
    coverage_fn = getattr(grid, "coverage", None)
    if callable(coverage_fn):
        try:
            return coverage_fn()
        except Exception:  # pragma: no cover - defensive
            return None
    if isinstance(grid, Mapping):
        return grid.get("coverage")
    return None


def _cov_get(coverage: Any, key: str, default: Any = None) -> Any:
    """Read one coverage counter from a ``CoverageStats``/mapping-like value."""
    if coverage is None:
        return default
    if isinstance(coverage, Mapping):
        return coverage.get(key, default)
    if hasattr(coverage, key):
        return getattr(coverage, key)
    to_dict = getattr(coverage, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:  # pragma: no cover - defensive
            return default
        if isinstance(data, Mapping):
            return data.get(key, default)
    return default


def _coverage_ratio(coverage: Any) -> float:
    """Honest coverage fraction in ``[0, 1]`` from counters or a float.

    Accepts a ``CoverageStats``, a mapping, a plain float/int ratio or ``None``.
    Falls back to ``walkable / (walkable + unknown_frontier)`` when the object
    only carries counters.
    """
    if coverage is None:
        return 0.0
    if isinstance(coverage, bool):
        return 0.0
    if isinstance(coverage, (int, float)):
        ratio: Any = float(coverage)
    else:
        ratio = _cov_get(coverage, "coverage_ratio", None)
        if ratio is None:
            ratio = _cov_get(coverage, "ratio", None)
        if ratio is None:
            walkable = _cov_get(coverage, "walkable", None)
            frontier = _cov_get(coverage, "unknown_frontier", None)
            try:
                walk_f = float(walkable)
                front_f = float(frontier)
            except (TypeError, ValueError):
                return 0.0
            total = walk_f + front_f
            ratio = walk_f / total if total > 0 else 0.0
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _state_name(cell: Any) -> str:
    """Normalised ``CellState`` value for a cell-like object (default unknown)."""
    state = _mapping_or_attr(cell, "state", None)
    if state is None:
        return CellState.UNKNOWN.value
    value = getattr(state, "value", state)
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if name in CELL_STATE_COLORS:
        return name
    return _STATE_ALIASES.get(name, CellState.UNKNOWN.value)


def cell_color(cell: Any) -> tuple[int, int, int]:
    """Palette RGB colour used to fill ``cell`` in the map image."""
    return CELL_STATE_COLORS[_state_name(cell)]


def _count_states(cells: Mapping[_CellCoord, Any]) -> dict[str, int]:
    counts = {name: 0 for name in CELL_STATE_COLORS}
    for cell in cells.values():
        counts[_state_name(cell)] = counts.get(_state_name(cell), 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Map window / drawing primitives
# --------------------------------------------------------------------------- #

def _map_window(
    grid: Any,
    cells: Mapping[_CellCoord, Any],
    pose: Any,
    path: Sequence[Any] | None,
    target: Any,
    max_cells: int,
) -> tuple[int, int, int, int]:
    """``(x0, y0, view_w, view_h)`` of the rendered map window.

    The window shows at most ``max_cells`` cells per axis and is centred on the
    player pose (or the grid centre when the pose is unknown); it is clamped so
    that it always overlaps the known part of the grid.  A grid smaller than
    ``max_cells`` is rendered completely.
    """
    side = max(1, int(max_cells))
    if not cells:
        view = max(1, min(side, 8))
        return (-(view // 2), -(view // 2), view, view)
    bounds = _grid_bounds(grid, cells)
    if bounds is None:  # pragma: no cover - _grid_bounds only returns None for empty cells
        view = max(1, min(side, 8))
        return (-(view // 2), -(view // 2), view, view)
    minx, miny, maxx, maxy = bounds
    span_x = maxx - minx + 1
    span_y = maxy - miny + 1
    centre = _pose_xy(pose)
    if centre is None:
        centre = _cell_xy(target)
    if centre is None and path:
        for item in path:
            centre = _cell_xy(item)
            if centre is not None:
                break
    if centre is None:
        centre = (minx + (span_x - 1) // 2, miny + (span_y - 1) // 2)
    view_w = min(side, span_x)
    view_h = min(side, span_y)
    x0 = int(centre[0]) - view_w // 2
    y0 = int(centre[1]) - view_h // 2
    x0 = max(minx, min(x0, maxx - view_w + 1))
    y0 = max(miny, min(y0, maxy - view_h + 1))
    return (x0, y0, view_w, view_h)


def _fill_rect(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    """Fill a clipped pixel rectangle in place."""
    height_px, width_px = canvas.shape[:2]
    x1 = min(width_px, x0 + width)
    y1 = min(height_px, y0 + height)
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = color


def _fill_cell_inset(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    cell_px: int,
    color: tuple[int, int, int],
    *,
    inset: int,
) -> None:
    """Fill a cell rect shrunk by ``inset`` on every side (overlay drawing)."""
    inset = max(0, min(int(inset), cell_px // 2 - (1 if cell_px > 2 else 0)))
    _fill_rect(canvas, x0 + inset, y0 + inset, cell_px - 2 * inset, cell_px - 2 * inset, color)


def _draw_ring(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    cell_px: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a rectangular outline (target ring) around one cell."""
    thickness = max(1, cell_px // 6)
    x1 = x0 + cell_px
    y1 = y0 + cell_px
    _fill_rect(canvas, x0, y0, cell_px, thickness, color)
    _fill_rect(canvas, x0, y1 - thickness, cell_px, thickness, color)
    _fill_rect(canvas, x0, y0, thickness, cell_px, color)
    _fill_rect(canvas, x1 - thickness, y0, thickness, cell_px, color)


def _draw_player(canvas: np.ndarray, x0: int, y0: int, cell_px: int) -> None:
    """Draw the white player marker inside one cell (always covering its centre)."""
    side = max(2, cell_px // 2)
    side = min(side, cell_px)
    inset = (cell_px - side) // 2
    _fill_rect(canvas, x0 + inset, y0 + inset, side, side, PALETTE["player"])


def _inside(xy: _CellCoord | None, x0: int, y0: int, view_w: int, view_h: int) -> bool:
    if xy is None:
        return False
    return x0 <= xy[0] < x0 + view_w and y0 <= xy[1] < y0 + view_h


# --------------------------------------------------------------------------- #
# Map legend helpers
# --------------------------------------------------------------------------- #

def _legend_rows(img_w: int) -> list[list[tuple[str, tuple[int, int, int], int]]]:
    """Pack legend items into rows that fit ``img_w`` (deterministic layout)."""
    font = _font(_LEGEND_FONT_SIZE)
    rows: list[list[tuple[str, tuple[int, int, int], int]]] = []
    row: list[tuple[str, tuple[int, int, int], int]] = []
    cursor = 0
    available = max(1, int(img_w) - 2 * _LEGEND_PAD)
    for label, key in LEGEND_ITEMS:
        color = _palette(key)
        item_w = _LEGEND_SWATCH + 3 + _measure_text(label, font)
        if row and cursor + item_w > available:
            rows.append(row)
            row = []
            cursor = 0
        row.append((label, color, item_w))
        cursor += item_w + 7
    if row:
        rows.append(row)
    return rows


def _legend_height(rows: Sequence[Sequence[Any]]) -> int:
    line_h = _line_height(_font(_LEGEND_FONT_SIZE))
    return 1 + 2 + len(rows) * line_h + 2 + line_h + 3


# --------------------------------------------------------------------------- #
# Coverage / status text helpers
# --------------------------------------------------------------------------- #

def _fmt_int(value: Any, *, placeholder: str = "-") -> str:
    if value is None:
        return placeholder
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return placeholder


def _fmt_text(value: Any, *, placeholder: str = "-") -> str:
    if value is None:
        return placeholder
    text = " ".join(str(value).split())
    return text or placeholder


def _fmt_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "-"
    return f"{ratio * 100:.1f}%"


def _status_get(status: Any, key: str, default: Any = None) -> Any:
    return _mapping_or_attr(status, key, default)


def _status_ratio(status: Any) -> float | None:
    raw = _status_get(status, "coverage_ratio", None)
    if raw is not None:
        return _coverage_ratio(raw)
    coverage = _status_get(status, "coverage", None)
    if coverage is not None:
        return _coverage_ratio(coverage)
    return None


def _status_links(status: Any) -> int | None:
    world = _status_get(status, "world", None)
    if isinstance(world, Mapping):
        links = world.get("connections")
        if isinstance(links, (int, float)):
            return int(links)
        if isinstance(links, (list, tuple)):
            return len(links)
    return None


def _status_areas_total(status: Any) -> int | None:
    raw = _status_get(status, "areas_total", None)
    if raw is not None:
        return int(raw)
    world = _status_get(status, "world", None)
    if isinstance(world, Mapping):
        areas = world.get("areas")
        if isinstance(areas, (list, tuple)):
            return len(areas)
        if isinstance(areas, Mapping):
            return len(areas)
    return None


def _coverage_text(
    grid: Any,
    cells: Mapping[_CellCoord, Any],
    pose: Any,
    path: Sequence[Any] | None,
    target: Any,
) -> str:
    """One-line coverage summary drawn under the map legend."""
    counts = _count_states(cells)
    area = _grid_area_id(grid) or "?"
    coverage = _grid_coverage(grid)
    total = _cov_get(coverage, "cells_total", len(cells)) if coverage is not None else len(cells)
    walkable = _cov_get(coverage, "walkable", counts["walkable"]) if coverage is not None else counts["walkable"]
    blocked = _cov_get(coverage, "blocked", counts["blocked"]) if coverage is not None else counts["blocked"]
    frontier = _cov_get(coverage, "unknown_frontier", None) if coverage is not None else None
    ratio = _coverage_ratio(coverage) if coverage is not None else None
    parts = [
        f"area={area}",
        f"cells={_fmt_int(total)}",
        f"walk={_fmt_int(walkable)}",
        f"wall={_fmt_int(blocked)}",
        f"front={_fmt_int(frontier)}",
    ]
    parts.append(f"cov={_fmt_ratio(ratio)}")
    pose_xy = _pose_xy(pose)
    if pose_xy is not None:
        parts.append(f"pose=({pose_xy[0]},{pose_xy[1]})")
    if path:
        parts.append(f"path={len(path)}")
    target_xy = _cell_xy(target)
    if target_xy is not None:
        parts.append(f"tgt=({target_xy[0]},{target_xy[1]})")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# render_map_image
# --------------------------------------------------------------------------- #

def render_map_image(
    grid: GridMap,
    *,
    pose: _PoseLike = None,
    path: Sequence[Any] | None = None,
    target: Any = None,
    cell_px: int = 10,
    max_cells: int = 64,
    legend: bool = True,
) -> np.ndarray:
    """Render one area's grid map as an RGBA image.

    Parameters
    ----------
    grid:
        GridMap-like object (``.cells`` mapping plus optional ``bounds()``,
        ``coverage()``, ``area_id``).  Missing methods/keys are tolerated.
    pose:
        Player pose (``PlayerPose``, a mapping with ``x``/``y`` or ``(x, y)``);
        the window is centred on it when known.
    path:
        Planned path ``[(x, y), ...]`` drawn magenta.
    target:
        Target cell ``(x, y)`` drawn as a cyan ring.
    cell_px:
        Pixel size of one cell (``>= 1``).
    max_cells:
        Maximum number of cells per axis in the rendered window (the map is
        centred on the player when the grid is larger).
    legend:
        Draw the palette legend and a coverage summary below the map.

    Returns
    -------
    np.ndarray
        ``(h, w, 4)`` uint8 RGBA image.  Without ``legend`` the size is exactly
        ``(view_h * cell_px, view_w * cell_px)``.
    """
    cell_px = max(1, int(cell_px))
    max_cells = max(1, int(max_cells))
    cells = _cells_of(grid)
    path_cells = _normalise_path(path)
    target_xy = _cell_xy(target)
    pose_xy = _pose_xy(pose)

    x0, y0, view_w, view_h = _map_window(
        grid, cells, pose, path_cells, target_xy, max_cells
    )
    map_w = view_w * cell_px
    map_h = view_h * cell_px

    legend_rows: list[list[tuple[str, tuple[int, int, int], int]]] = []
    if legend:
        img_w = max(map_w, _LEGEND_MIN_WIDTH)
        legend_rows = _legend_rows(img_w)
        legend_h = _legend_height(legend_rows)
    else:
        img_w = map_w
        legend_h = 0
    map_x = (img_w - map_w) // 2

    canvas = np.full((map_h + legend_h, img_w, 3), PALETTE["background"], dtype=np.uint8)
    # The window interior starts as "unknown" so holes / unseen cells are dark.
    _fill_rect(canvas, map_x, 0, map_w, map_h, PALETTE["unknown"])
    for (cx, cy), cell in cells.items():
        if _inside((cx, cy), x0, y0, view_w, view_h):
            _fill_rect(
                canvas,
                map_x + (cx - x0) * cell_px,
                (cy - y0) * cell_px,
                cell_px,
                cell_px,
                cell_color(cell),
            )

    # Overlays, bottom to top: path -> target ring -> player.
    if path_cells:
        inset = max(0, min(2, cell_px // 4))
        for item in path_cells:
            if _inside(item, x0, y0, view_w, view_h):
                _fill_cell_inset(
                    canvas,
                    map_x + (item[0] - x0) * cell_px,
                    (item[1] - y0) * cell_px,
                    cell_px,
                    PALETTE["path"],
                    inset=inset,
                )
    if _inside(target_xy, x0, y0, view_w, view_h):
        _draw_ring(
            canvas,
            map_x + (target_xy[0] - x0) * cell_px,
            (target_xy[1] - y0) * cell_px,
            cell_px,
            PALETTE["target"],
        )
    if _inside(pose_xy, x0, y0, view_w, view_h):
        _draw_player(
            canvas,
            map_x + (pose_xy[0] - x0) * cell_px,
            (pose_xy[1] - y0) * cell_px,
            cell_px,
        )

    if legend:
        font = _font(_LEGEND_FONT_SIZE)
        line_h = _line_height(font)
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.line(
            [(0, map_h), (img_w - 1, map_h)],
            fill=PALETTE["panel_border"],
            width=1,
        )
        y = map_h + 3
        for row in legend_rows:
            x = _LEGEND_PAD
            for label, color, item_w in row:
                swatch_y = y + max(0, (line_h - _LEGEND_SWATCH) // 2)
                draw.rectangle(
                    [x, swatch_y, x + _LEGEND_SWATCH - 1, swatch_y + _LEGEND_SWATCH - 1],
                    fill=color,
                    outline=PALETTE["panel_border"],
                )
                draw.text(
                    (x + _LEGEND_SWATCH + 3, y),
                    label,
                    font=font,
                    fill=PALETTE["text_dim"],
                )
                x += item_w + 7
            y += line_h
        text_y = map_h + 1 + 2 + len(legend_rows) * line_h + 2
        summary = _coverage_text(grid, cells, pose, path_cells, target_xy)
        draw.text(
            (_LEGEND_PAD, text_y),
            _clip_text(summary, font, img_w - 2 * _LEGEND_PAD),
            font=font,
            fill=PALETTE["text"],
        )
        canvas = np.asarray(image, dtype=np.uint8)

    return _to_rgba(canvas)


# --------------------------------------------------------------------------- #
# render_status_panel
# --------------------------------------------------------------------------- #

def _status_lines(status: Any) -> list[tuple[str, str]]:
    """Human-readable ``(name, value)`` rows for the status panel (tolerant)."""
    area_id = _fmt_text(_status_get(status, "area_id", ""))
    area_kind = _fmt_text(_status_get(status, "area_kind", ""))
    pose = _status_get(status, "pose", None)
    pose_xy = _pose_xy(pose)
    if pose_xy is not None:
        facing = _pose_facing(pose)
        confidence = _pose_confidence(pose)
        valid = _pose_valid(pose)
        pose_text = f"({pose_xy[0]},{pose_xy[1]}) {facing or '-'}"
        if confidence is not None:
            pose_text += f" conf {confidence:.2f}"
        if valid is False:
            pose_text += " INVALID"
    else:
        pose_text = "-"
    cells_total = _fmt_int(_status_get(status, "cells_total", None))
    walkable = _fmt_int(_status_get(status, "walkable", None))
    blocked = _fmt_int(_status_get(status, "blocked", None))
    frontier = _fmt_int(_status_get(status, "unknown_frontier", None))
    ratio = _status_ratio(status)
    areas_total = _fmt_int(_status_areas_total(status), placeholder="-")
    links = _fmt_int(_status_links(status), placeholder="-")
    last_event = _fmt_text(_status_get(status, "last_event", ""), placeholder="none")
    frame = _fmt_int(
        _status_get(status, "frame_index", _status_get(status, "frame", None)),
        placeholder="-",
    )
    lines: list[tuple[str, str]] = []
    if bool(_status_get(status, "paused", False)):
        lines.append(("state", "PAUSED"))
    lines.extend(
        [
            ("area", f"{area_id} ({area_kind})"),
            ("pose", pose_text),
            ("cells", f"{cells_total} total / {walkable} walk / {blocked} block"),
            ("front", f"{frontier} unknown frontier"),
            ("cov", _fmt_ratio(ratio)),
            ("world", f"{areas_total} areas / {links} links"),
            ("event", last_event),
            ("frame", frame),
        ]
    )
    return lines


def render_status_panel(
    status: Mapping[str, Any],
    *,
    width: int = 320,
    height: int = 160,
) -> np.ndarray:
    """Render a navigation status dictionary as an RGBA panel.

    Tolerates missing keys, ``None`` and attribute-style objects: every unknown
    field renders as ``"-"`` instead of raising.  The expected keys are the ones
    produced by ``NavigationAgent.status()`` (``area_id``, ``area_kind``,
    ``pose``, ``cells_total``, ``walkable``, ``blocked``, ``unknown_frontier``,
    ``coverage_ratio``, ``areas_total``, ``last_event``, ``paused``).
    """
    width = max(48, int(width))
    height = max(24, int(height))
    image = Image.new("RGB", (width, height), PALETTE["background"])
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [0, 0, width - 1, height - 1],
        fill=PALETTE["panel"],
        outline=PALETTE["panel_border"],
    )
    # Scale the type with the panel so large report panels do not look sparse.
    body_size = max(8, min(12, width // 40, max(8, height // 18)))
    title_size = min(14, body_size + 2)
    title_font = _font(title_size)
    body_font = _font(body_size)
    line_h = _line_height(body_font)
    header_h = max(12, _line_height(title_font) + 3)
    draw.rectangle([1, 1, width - 2, header_h], fill=PALETTE["panel_alt"])
    title = "NAV STATUS"
    draw.text((4, 2), _clip_text(title, title_font, width - 8), font=title_font, fill=PALETTE["text"])
    if bool(_status_get(status, "paused", False)):
        badge = "PAUSED"
        badge_font = _font(max(7, body_size - 1))
        badge_w = _measure_text(badge, badge_font)
        badge_h = _line_height(badge_font)
        badge_x1 = width - 4
        badge_x0 = max(4, badge_x1 - badge_w - 6)
        badge_y0 = 2
        badge_y1 = min(header_h - 1, badge_y0 + badge_h + 1)
        draw.rectangle(
            [badge_x0, badge_y0, badge_x1, badge_y1], fill=PALETTE["paused"]
        )
        draw.text(
            (badge_x0 + 3, badge_y0),
            badge,
            font=badge_font,
            fill=PALETTE["background"],
        )

    ratio = _status_ratio(status)
    bar_reserve = 0
    if height >= 56:
        bar_reserve = 12
    available_bottom = height - 1 - bar_reserve
    lines = _status_lines(status)
    y = header_h + 3
    # Distribute any spare vertical space between the rows (capped so the panel
    # stays compact for the small composite slot).
    spare = available_bottom - y - len(lines) * line_h
    row_gap = 0
    if len(lines) > 0 and spare > 0:
        row_gap = min(8, spare // len(lines))
    for name, value in lines:
        if y + line_h > available_bottom:
            break
        text = _clip_text(f"{name}  {value}", body_font, width - 8)
        draw.text((4, y), text, font=body_font, fill=PALETTE["text_dim"])
        y += line_h + row_gap

    if bar_reserve:
        bar_y0 = height - 10
        bar_y1 = height - 5
        inner_w = max(0, width - 10)
        draw.rectangle(
            [4, bar_y0, width - 5, bar_y1],
            fill=PALETTE["coverage_bg"],
            outline=PALETTE["panel_border"],
        )
        filled = int(round((ratio if ratio is not None else 0.0) * inner_w))
        filled = max(0, min(inner_w, filled))
        if filled > 0:
            draw.rectangle(
                [5, bar_y0 + 1, 5 + filled - 1, bar_y1 - 1],
                fill=PALETTE["coverage_fill"],
            )
    return _to_rgba(np.asarray(image, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# World graph rendering
# --------------------------------------------------------------------------- #

def _world_areas(world: Any) -> dict[str, Any]:
    areas = getattr(world, "areas", None)
    if areas is None and isinstance(world, Mapping):
        areas = world.get("areas")
    if isinstance(areas, Mapping):
        return {str(key): value for key, value in areas.items()}
    if isinstance(areas, (list, tuple)):
        result: dict[str, Any] = {}
        for index, area in enumerate(areas):
            area_id = _mapping_or_attr(area, "area_id", None)
            result[str(area_id) if area_id is not None else str(index)] = area
        return result
    return {}


def _world_connections(world: Any) -> list[Any]:
    connections = getattr(world, "connections", None)
    if connections is None and isinstance(world, Mapping):
        connections = world.get("connections")
    if isinstance(connections, (list, tuple)):
        return list(connections)
    return []


def _world_current_area(world: Any) -> str:
    value = getattr(world, "current_area", None)
    if value is None and isinstance(world, Mapping):
        value = world.get("current_area")
    if value is None:
        summary = getattr(world, "summary", None)
        if callable(summary):
            try:
                data = summary()
            except Exception:  # pragma: no cover - defensive
                data = None
            if isinstance(data, Mapping):
                value = data.get("current_area")
    return "" if value is None else str(value)


def _area_get(area: Any, key: str, default: Any = None) -> Any:
    return _mapping_or_attr(area, key, default)


def _area_stats(area: Any) -> dict[str, Any]:
    """``area_id``/``kind``/``visits``/``walkable``/``cells_total`` of one area."""
    area_id = _area_get(area, "area_id", "")
    kind = _area_get(area, "kind", "unknown")
    visits = _fmt_int(_area_get(area, "visits", 0), placeholder="0")
    grid = _area_get(area, "grid", None)
    walkable = None
    cells_total = None
    if grid is not None:
        coverage = _grid_coverage(grid)
        if coverage is not None:
            walkable = _cov_get(coverage, "walkable", None)
            cells_total = _cov_get(coverage, "cells_total", None)
        if walkable is None or cells_total is None:
            cells = _cells_of(grid)
            counts = _count_states(cells)
            if walkable is None:
                walkable = counts["walkable"]
            if cells_total is None:
                cells_total = len(cells)
    return {
        "area_id": "" if area_id is None else str(area_id),
        "kind": _fmt_text(kind, placeholder="unknown"),
        "visits": int(visits) if visits.lstrip("-").isdigit() else 0,
        "walkable": walkable,
        "cells_total": cells_total,
    }


def _world_node_layout(
    count: int,
    width: int,
    height: int,
    node_h: int = 34,
) -> tuple[int, int, int, int, int, list[tuple[int, int, int, int]]]:
    """Deterministic grid layout for ``count`` area nodes.

    Nodes are packed in a roughly square grid (``round(sqrt(n))`` columns,
    capped by the available width), sized/capped at a readable width and
    centred both horizontally and vertically.  Returns
    ``(margin, title_h, node_w, node_h, grid_w, rects)`` where ``rects`` are
    ``(x, y, w, h)`` boxes in drawing order.
    """
    margin = 5
    title_h = 15
    gap_x, gap_y = 6, 14
    usable_w = max(1, width - 2 * margin)
    usable_h = max(1, height - title_h - 2 * margin)
    min_node_w = 78
    max_cols = max(1, usable_w // (min_node_w + gap_x))
    ideal_cols = max(1, int(round(math.sqrt(max(1, count)))))
    cols = max(1, min(count, max_cols, ideal_cols))
    node_w = max(40, min(150, (usable_w - (cols - 1) * gap_x) // cols))
    rows = max(1, (count + cols - 1) // cols)
    node_h = max(20, int(node_h))
    total_h = rows * node_h + max(0, rows - 1) * gap_y
    if total_h > usable_h:
        gap_y = 8
        node_h = max(20, (usable_h - max(0, rows - 1) * gap_y) // rows)
        total_h = rows * node_h + max(0, rows - 1) * gap_y
    grid_w = cols * node_w + max(0, cols - 1) * gap_x
    x_start = margin + max(0, (usable_w - grid_w) // 2)
    y_start = title_h + margin + max(0, (usable_h - total_h) // 2)
    rects: list[tuple[int, int, int, int]] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        x = x_start + col * (node_w + gap_x)
        y = y_start + row * (node_h + gap_y)
        rects.append((x, y, node_w, node_h))
    return (margin, title_h, node_w, node_h, grid_w, rects)


def _connection_label_position(
    from_centre: tuple[int, int],
    to_centre: tuple[int, int],
    label_w: int,
    label_h: int,
    rects: Sequence[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> tuple[int, int]:
    """Best label spot along a connection line (prefer the midpoint, avoid boxes)."""
    best: tuple[int, int, int] | None = None  # (overlap, x, y)
    for fraction in (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7, 0.25, 0.75):
        mid_x = int(round(from_centre[0] + (to_centre[0] - from_centre[0]) * fraction))
        mid_y = int(round(from_centre[1] + (to_centre[1] - from_centre[1]) * fraction))
        text_x = max(1, min(width - label_w - 2, mid_x - label_w // 2))
        text_y = max(1, min(height - label_h - 1, mid_y - label_h))
        overlap = 0
        for box_x, box_y, box_w, box_h in rects:
            overlap_x = max(0, min(text_x + label_w, box_x + box_w) - max(text_x, box_x))
            overlap_y = max(0, min(text_y + label_h, box_y + box_h) - max(text_y, box_y))
            overlap += overlap_x * overlap_y
        if overlap == 0:
            return (text_x, text_y)
        if best is None or overlap < best[0]:
            best = (overlap, text_x, text_y)
    if best is None:  # pragma: no cover - there is always at least one candidate
        return (1, 1)
    return (best[1], best[2])


def render_world_image(
    world: WorldModel,
    *,
    width: int = 320,
    height: int = 160,
) -> np.ndarray:
    """Render the multi-area world graph as an RGBA image.

    Area nodes show ``area_id``, ``kind``, visits and walkable-cell count; the
    current area is highlighted with a cyan border.  Connections are drawn as
    labelled lines (``warp``/``walk``/``unknown``).  Works with a real
    ``WorldModel`` or any duck-typed object exposing ``.areas``,
    ``.connections`` and ``.current_area``.
    """
    width = max(80, int(width))
    height = max(48, int(height))
    areas = _world_areas(world)
    connections = _world_connections(world)
    current = _world_current_area(world)
    image = Image.new("RGB", (width, height), PALETTE["background"])
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width - 1, height - 1], outline=PALETTE["panel_border"])
    title_size = max(9, min(12, width // 45))
    title_font = _font(title_size)
    title = f"WORLD / {len(areas)} areas / {len(connections)} links"
    if current:
        title += f" / current: {current}"
    draw.text(
        (4, 2),
        _clip_text(title, title_font, width - 8),
        font=title_font,
        fill=PALETTE["text"],
    )
    if not areas:
        draw.text(
            (4, 20),
            _clip_text("no areas recorded", title_font, width - 8),
            font=title_font,
            fill=PALETTE["text_faint"],
        )
        return _to_rgba(np.asarray(image, dtype=np.uint8))

    area_ids = sorted(areas, key=str)
    id_font = _font(max(9, min(12, width // 46)))
    small_font = _font(max(7, min(10, width // 60)))
    small_h = _line_height(small_font)
    node_h = max(34, min(52, _line_height(id_font) + 2 * small_h + 6))
    _margin, _title_h, node_w, _node_h, _grid_w, rects = _world_node_layout(
        len(area_ids), width, height, node_h=node_h
    )
    rect_by_id = {area_id: rects[index] for index, area_id in enumerate(area_ids)}
    centres = {
        area_id: (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
        for area_id, rect in rect_by_id.items()
    }

    def _conn_get(connection: Any, key: str, default: Any = None) -> Any:
        return _mapping_or_attr(connection, key, default)

    normalised: list[dict[str, Any]] = []
    for connection in connections:
        from_area = _conn_get(connection, "from_area", "")
        to_area = _conn_get(connection, "to_area", "")
        kind = _fmt_text(_conn_get(connection, "kind", "warp"), placeholder="warp")
        direction = _fmt_text(_conn_get(connection, "direction", ""), placeholder="")
        if from_area is None or to_area is None:
            continue
        normalised.append(
            {
                "from": str(from_area),
                "to": str(to_area),
                "kind": kind,
                "direction": direction,
            }
        )
    normalised.sort(key=lambda item: (item["from"], item["to"], item["kind"], item["direction"]))

    def _connection_color(kind: str) -> tuple[int, int, int]:
        name = kind.strip().lower()
        if name == "walk":
            return PALETTE["connection_walk"]
        if name == "warp":
            return PALETTE["connection_warp"]
        return PALETTE["connection_unknown"]

    # Lines first, then boxes (boxes cover line ends), then labels on top.
    drawn_links: list[tuple[dict[str, Any], tuple[int, int], tuple[int, int]]] = []
    for link in normalised:
        from_centre = centres.get(link["from"])
        to_centre = centres.get(link["to"])
        if from_centre is None or to_centre is None:
            continue
        draw.line(
            [from_centre, to_centre],
            fill=_connection_color(link["kind"]),
            width=1,
        )
        drawn_links.append((link, from_centre, to_centre))

    for area_id in area_ids:
        x, y, box_w, box_h = rect_by_id[area_id]
        stats = _area_stats(areas[area_id])
        is_current = area_id == current
        fill = (36, 52, 62) if is_current else PALETTE["panel_alt"]
        outline = PALETTE["current_area"] if is_current else PALETTE["panel_border"]
        draw.rectangle(
            [x, y, x + box_w - 1, y + box_h - 1],
            fill=fill,
            outline=outline,
            width=2 if is_current else 1,
        )
        text_color = PALETTE["text"] if is_current else PALETTE["text_dim"]
        draw.text(
            (x + 4, y + 2),
            _clip_text(stats["area_id"] or "?", id_font, box_w - 8),
            font=id_font,
            fill=text_color,
        )
        draw.text(
            (x + 4, y + 2 + _line_height(id_font)),
            _clip_text(stats["kind"], small_font, box_w - 8),
            font=small_font,
            fill=PALETTE["text_faint"],
        )
        detail = f"v{stats['visits']} w{_fmt_int(stats['walkable'])}"
        draw.text(
            (x + 4, y + 2 + _line_height(id_font) + small_h),
            _clip_text(detail, small_font, box_w - 8),
            font=small_font,
            fill=PALETTE["text_faint"],
        )

    for link, from_centre, to_centre in drawn_links:
        label = _clip_text(link["kind"], small_font, max(10, node_w // 2))
        if not label:
            continue
        label_w = _measure_text(label, small_font)
        text_x, text_y = _connection_label_position(
            from_centre,
            to_centre,
            label_w + 2,
            small_h + 2,
            rects,
            width,
            height,
        )
        # Small background plate keeps the label readable where it crosses a line.
        draw.rectangle(
            [text_x, text_y, text_x + label_w + 1, text_y + small_h + 1],
            fill=PALETTE["background"],
        )
        draw.text(
            (text_x + 1, text_y + 1),
            label,
            font=small_font,
            fill=_connection_color(link["kind"]),
        )
    return _to_rgba(np.asarray(image, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Composite viewer
# --------------------------------------------------------------------------- #

def _normalise_path(path: Any) -> list[_CellCoord] | None:
    if path is None:
        return None
    result: list[_CellCoord] = []
    try:
        iterator = list(path)
    except TypeError:
        return None
    for item in iterator:
        xy = _cell_xy(item)
        if xy is not None:
            result.append(xy)
    return result or None


def _composite_status(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build the status dict passed to ``render_status_panel`` from viewer state."""
    status: dict[str, Any] = {}
    raw = state.get("status")
    if isinstance(raw, Mapping):
        status.update(raw)
    elif raw is not None:
        for key in _STATUS_KEYS:
            value = getattr(raw, key, None)
            if value is not None:
                status[key] = value
    for key in _STATUS_KEYS:
        if key not in status and key in state:
            status[key] = state[key]
    if "paused" not in status:
        if "paused" in state:
            status["paused"] = state["paused"]
        elif "pause" in state:
            status["paused"] = state["pause"]
    if "frame_index" not in status and "frame_index" in state:
        status["frame_index"] = state["frame_index"]
    if "areas_total" not in status:
        world = state.get("world")
        if world is not None:
            status["areas_total"] = len(_world_areas(world))
    return status


def _scale_nearest(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Nearest-neighbour scale an RGB frame into the ``target_w``/``target_h`` box."""
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame
    scale = min(target_w / float(width), target_h / float(height))
    if scale <= 0:
        return frame
    integer = int(scale)
    if integer >= 1 and abs(scale - integer) < 1e-9:
        return np.ascontiguousarray(np.repeat(np.repeat(frame, integer, axis=0), integer, axis=1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    image = Image.fromarray(frame).resize((new_w, new_h), Image.NEAREST)
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _paste_rgb(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    """Paste ``image`` onto ``canvas`` at ``(x, y)`` with clipping."""
    height, width = canvas.shape[:2]
    image_h, image_w = image.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x) + image_w)
    y1 = min(height, int(y) + image_h)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] = image[y0 - int(y) : y1 - int(y), x0 - int(x) : x1 - int(x)]


def _paste_rgba_centered(
    canvas: np.ndarray,
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    x, y, box_w, box_h = box
    image_h, image_w = image.shape[:2]
    offset_x = x + max(0, (box_w - image_w) // 2)
    offset_y = y + max(0, (box_h - image_h) // 2)
    _paste_rgb(canvas, image[..., :3], offset_x, offset_y)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: Any,
    color: tuple[int, int, int],
) -> None:
    x, y, box_w, box_h = box
    label = _clip_text(text, font, box_w - 8)
    text_w = _measure_text(label, font)
    text_h = _line_height(font)
    draw.text(
        (x + (box_w - text_w) // 2, y + (box_h - text_h) // 2),
        label,
        font=font,
        fill=color,
    )


def _composite_text_lines(
    state: Mapping[str, Any],
    status: Mapping[str, Any],
    path: Sequence[_CellCoord] | None,
    target: _CellCoord | None,
) -> list[str]:
    """Three compact summary lines for the bottom strip (never raises)."""
    area_id = _fmt_text(_status_get(status, "area_id", state.get("area_id", "")))
    area_kind = _fmt_text(_status_get(status, "area_kind", state.get("area_kind", "")))
    pose = _status_get(status, "pose", state.get("pose", None))
    pose_xy = _pose_xy(pose)
    facing = _pose_facing(pose) or "-"
    confidence = _pose_confidence(pose)
    pose_text = (
        f"({pose_xy[0]},{pose_xy[1]}) {facing} conf {confidence:.2f}"
        if pose_xy is not None and confidence is not None
        else (f"({pose_xy[0]},{pose_xy[1]}) {facing}" if pose_xy is not None else "-")
    )
    frame_index = state.get("frame_index")
    if frame_index is None:
        frame_index = _status_get(status, "frame_index", None)
    ratio = _status_ratio(status)
    paused = bool(_status_get(status, "paused", state.get("paused", False)))
    target_xy = _cell_xy(target)
    target_text = f"({target_xy[0]},{target_xy[1]})" if target_xy is not None else "-"
    last_event = _fmt_text(
        _status_get(status, "last_event", state.get("last_event", "")),
        placeholder="none",
    )
    last_action = _fmt_text(state.get("last_action", ""), placeholder="")
    line1 = f"area {area_id} ({area_kind}) / pose {pose_text}"
    line2 = (
        f"cells {_fmt_int(_status_get(status, 'cells_total', None))}"
        f" / walk {_fmt_int(_status_get(status, 'walkable', None))}"
        f" / block {_fmt_int(_status_get(status, 'blocked', None))}"
        f" / frontier {_fmt_int(_status_get(status, 'unknown_frontier', None))}"
        f" / cov {_fmt_ratio(ratio)}"
    )
    line3 = (
        f"path {len(path) if path else 0} -> {target_text}"
        f" / frame {_fmt_int(frame_index)}"
        f" / event {last_event}"
        f" / paused {'yes' if paused else 'no'}"
    )
    if last_action:
        line3 += f" / {last_action}"
    return [line1, line2, line3]


def _composite_map_cell_px(grid: Any, box: tuple[int, int, int, int], max_cells: int) -> int:
    """Pick a cell size that fills the map panel for the grid's known extent."""
    cells = _cells_of(grid)
    bounds = _grid_bounds(grid, cells)
    fallback = max(1, min(box[2] // max_cells, box[3] // max_cells))
    if bounds is None:
        return fallback
    span_x = min(max_cells, max(1, bounds[2] - bounds[0] + 1))
    span_y = min(max_cells, max(1, bounds[3] - bounds[1] + 1))
    cell_px = min(box[2] // span_x, box[3] // span_y)
    return max(1, min(int(cell_px), 24))


def _composite_rgb(
    frame: np.ndarray | None,
    grid: Any,
    pose: Any,
    path: Sequence[_CellCoord] | None,
    target: _CellCoord | None,
    status: Mapping[str, Any],
    state: Mapping[str, Any],
) -> np.ndarray:
    canvas = np.full((_COMPOSITE_H, _COMPOSITE_W, 3), PALETTE["background"], dtype=np.uint8)
    if frame is not None:
        scaled = _scale_nearest(frame, _GAME_W, _GAME_H)
        _paste_rgb(canvas, scaled, _MARGIN, _MARGIN)
        map_box = _FRAME_MAP_BOX
        status_box = _FRAME_STATUS_BOX
    else:
        map_box = _NOFRAME_MAP_BOX
        status_box = _NOFRAME_STATUS_BOX

    map_image: np.ndarray | None = None
    if grid is not None:
        max_cells = 24 if frame is not None else 40
        cell_px = _composite_map_cell_px(grid, map_box, max_cells)
        map_image = render_map_image(
            grid,
            pose=pose,
            path=path,
            target=target,
            cell_px=cell_px,
            max_cells=max_cells,
            legend=False,
        )
        _paste_rgba_centered(canvas, map_image, map_box)
    status_image = render_status_panel(status, width=status_box[2], height=status_box[3])
    _paste_rgb(canvas, status_image[..., :3], status_box[0], status_box[1])

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [map_box[0], map_box[1], map_box[0] + map_box[2] - 1, map_box[1] + map_box[3] - 1],
        outline=PALETTE["panel_border"],
    )
    if map_image is None:
        placeholder_box = (
            map_box[0] + 1,
            map_box[1] + 1,
            map_box[2] - 2,
            map_box[3] - 2,
        )
        draw.rectangle(
            [
                placeholder_box[0],
                placeholder_box[1],
                placeholder_box[0] + placeholder_box[2] - 1,
                placeholder_box[1] + placeholder_box[3] - 1,
            ],
            fill=PALETTE["panel"],
        )
        _draw_centered_text(draw, "no map data", placeholder_box, _font(9), PALETTE["text_faint"])

    # Bottom strip: coverage bar + three summary lines.
    bx, by, bw, bh = _BOTTOM_BOX
    draw.rectangle(
        [bx, by, bx + bw - 1, by + bh - 1],
        fill=PALETTE["panel"],
        outline=PALETTE["panel_border"],
    )
    bar_w, bar_h = 116, 12
    bar_x, bar_y = bx + 6, by + 6
    ratio_value = _status_ratio(status)
    if ratio_value is None:
        ratio_value = 0.0
    draw.rectangle(
        [bar_x, bar_y, bar_x + bar_w - 1, bar_y + bar_h - 1],
        fill=PALETTE["coverage_bg"],
        outline=PALETTE["panel_border"],
    )
    inner_w = bar_w - 2
    filled = int(round(ratio_value * inner_w))
    filled = max(0, min(inner_w, filled))
    if filled > 0:
        draw.rectangle(
            [bar_x + 1, bar_y + 1, bar_x + filled, bar_y + bar_h - 2],
            fill=PALETTE["coverage_fill"],
        )
    bar_font = _font(7)
    draw.text(
        (bar_x, bar_y + bar_h + 3),
        f"coverage {_fmt_ratio(ratio_value)}",
        font=bar_font,
        fill=PALETTE["text_faint"],
    )

    text_x = bar_x + bar_w + 16
    text_w = bx + bw - text_x - 4
    body_font = _font(8)
    line_h = _line_height(body_font)
    text_y = by + 4
    for line in _composite_text_lines(state, status, path, target):
        if text_y + line_h > by + bh - 2:
            break
        draw.text(
            (text_x, text_y),
            _clip_text(line, body_font, text_w),
            font=body_font,
            fill=PALETTE["text_dim"],
        )
        text_y += line_h
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def composite_viewer(
    state: Mapping[str, Any],
    *,
    frame_rgba: np.ndarray | None = None,
    width: int = 480,
    height: int = 360,
) -> np.ndarray:
    """Render the full viewer composite (game frame + map + status + summary).

    Parameters
    ----------
    state:
        Mapping with optional keys (all tolerated when missing)::

            grid, pose, path, target, status, world, paused/pause,
            frame_index, last_action

        ``status`` should be ``NavigationAgent.status()``; flat status keys are
        also accepted directly on ``state``.  ``pose`` may be a ``PlayerPose``,
        a mapping or ``(x, y)``.  ``frame_rgba`` may also be a ``Frame``.
    frame_rgba:
        Raw 160x144 RGBA/RGB game frame (drawn scaled 2x); ``None`` renders the
        panels-only layout.
    width, height:
        Output size; the 480x360 layout is nearest-neighbour rescaled when
        another size is requested.

    Returns
    -------
    np.ndarray
        ``(height, width, 4)`` uint8 RGBA composite.
    """
    state_map: Mapping[str, Any] = state if isinstance(state, Mapping) else {}
    grid = state_map.get("grid")
    pose = state_map.get("pose")
    if pose is None:
        pose = state_map.get("player_pose")
    if pose is None:
        pose = state_map.get("pose_dict")
    path = _normalise_path(state_map.get("path"))
    target = state_map.get("target")
    if target is None:
        target = state_map.get("goal_cell")
    target_xy = _cell_xy(target)
    status = _composite_status(state_map)
    frame = _normalise_frame(frame_rgba)
    rgb = _composite_rgb(frame, grid, pose, path, target_xy, status, state_map)
    out_w = max(16, int(width))
    out_h = max(16, int(height))
    if (out_w, out_h) != (_COMPOSITE_W, _COMPOSITE_H):
        resized = Image.fromarray(rgb).resize((out_w, out_h), Image.NEAREST)
        rgb = np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))
    return _to_rgba(rgb)


# --------------------------------------------------------------------------- #
# Coverage bar
# --------------------------------------------------------------------------- #

def coverage_bar(coverage: CoverageStats, *, width: int = 120) -> np.ndarray:
    """Render a small horizontal coverage bar as an RGBA image.

    ``coverage`` may be a :class:`CoverageStats`, a mapping with a
    ``coverage_ratio`` field (or the raw counters), or a plain float ratio.
    The returned image is 10 px high and ``width`` px wide.
    """
    width = max(6, int(width))
    height = 10
    ratio = _coverage_ratio(coverage)
    canvas = np.full((height, width, 3), PALETTE["background"], dtype=np.uint8)
    _fill_rect(canvas, 1, 1, width - 2, height - 2, PALETTE["coverage_bg"])
    inner_w = width - 2
    filled = int(round(ratio * inner_w))
    filled = max(0, min(inner_w, filled))
    if filled > 0:
        _fill_rect(canvas, 1, 1, filled, height - 2, PALETTE["coverage_fill"])
    _fill_rect(canvas, 0, 0, width, 1, PALETTE["panel_border"])
    _fill_rect(canvas, 0, height - 1, width, 1, PALETTE["panel_border"])
    _fill_rect(canvas, 0, 0, 1, height, PALETTE["panel_border"])
    _fill_rect(canvas, width - 1, 0, 1, height, PALETTE["panel_border"])
    return _to_rgba(canvas)


# --------------------------------------------------------------------------- #
# Cell description
# --------------------------------------------------------------------------- #

def describe_cell(cell: Any) -> str:
    """One-line human description of a cell (map/debug/report helper).

    Accepts a real :class:`Cell`, a mapping or ``None``; unknown fields are
    skipped and the result is clipped to ~120 characters.
    """
    if cell is None:
        return "no cell"
    xy = _cell_xy(cell)
    state = _state_name(cell)
    parts: list[str] = []
    if xy is None:
        parts.append("(?, ?)")
    else:
        parts.append(f"({xy[0]}, {xy[1]})")
    parts.append(state)
    terrain = _mapping_or_attr(cell, "terrain", "")
    if terrain:
        parts.append(f"terrain={str(terrain)[:10]}")
    visits = _mapping_or_attr(cell, "visit_count", _mapping_or_attr(cell, "visits", None))
    if visits is not None:
        parts.append(f"visits={_fmt_int(visits, placeholder='?')}")
    observed = _mapping_or_attr(cell, "observed_count", _mapping_or_attr(cell, "observations", None))
    if observed is not None:
        parts.append(f"observed={_fmt_int(observed, placeholder='?')}")
    confidence = _mapping_or_attr(cell, "confidence", None)
    if confidence is not None:
        try:
            parts.append(f"conf={float(confidence):.2f}")
        except (TypeError, ValueError):
            pass
    labels = _mapping_or_attr(cell, "labels", None)
    if labels:
        try:
            label_text = ",".join(str(label) for label in list(labels)[:3])
        except TypeError:
            label_text = str(labels)
        if label_text:
            parts.append(f"labels={label_text}")
    text = " ".join(parts)
    if len(text) > 120:
        text = text[:119] + ".."
    return text
