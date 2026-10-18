"""Tests for :mod:`llm_agent.navigation.viz` and the map-gallery CLI.

The renderer is pure numpy/Pillow, so every test here is synthetic: fake
``GridMap``/``WorldModel`` objects (duck typing, no PyBoy/ROM/network) plus the
shared ``tests/navigation/conftest.py`` synthetic room for the viewer frame.
Fast and deterministic; the map-gallery script is tested in-process with a fake
``WorldModel`` class and once end-to-end when the real (nav-map owned)
``world_model.py`` is available.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llm_agent.navigation import viz
from llm_agent.navigation.types import CellState, CoverageStats, PlayerPose

ROOT = Path(__file__).resolve().parents[2]
GALLERY_SCRIPT = ROOT / "scripts" / "nav_v2_map_gallery.py"

WALKABLE = CellState.WALKABLE
OBSERVED = CellState.OBSERVED
BLOCKED = CellState.BLOCKED
TEMP_BLOCKED = CellState.TEMP_BLOCKED
EXIT = CellState.EXIT
INTERACTIVE = CellState.INTERACTIVE
UNKNOWN = CellState.UNKNOWN
UNREACHABLE = CellState.UNREACHABLE
DEFERRED = CellState.DEFERRED


# --------------------------------------------------------------------------- #
# Fake grid / world objects (duck-typed, no dependency on nav-map's modules)
# --------------------------------------------------------------------------- #
class FakeCell:
    """Minimal stand-in for ``grid_map.Cell``."""

    def __init__(self, x: int, y: int, state: CellState = CellState.UNKNOWN, **kwargs) -> None:
        self.x = int(x)
        self.y = int(y)
        self.state = state
        self.terrain = kwargs.get("terrain", "")
        self.tile_hashes = list(kwargs.get("tile_hashes", []))
        self.observed_count = int(kwargs.get("observed_count", 0))
        self.visit_count = int(kwargs.get("visit_count", 0))
        self.confidence = float(kwargs.get("confidence", 0.0))
        self.last_seen_frame = int(kwargs.get("last_seen_frame", -1))
        self.obstacle_evidence = int(kwargs.get("obstacle_evidence", 0))
        self.free_evidence = int(kwargs.get("free_evidence", 0))
        self.labels = list(kwargs.get("labels", []))

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "state": self.state.value}


class FakeGrid:
    """Minimal stand-in for ``grid_map.GridMap`` (bounds()/coverage() included)."""

    def __init__(
        self,
        area_id: str = "test_area",
        cells: dict[tuple[int, int], FakeCell] | None = None,
        *,
        kind: str = "interior",
    ) -> None:
        self.area_id = area_id
        self.kind = kind
        self.cells: dict[tuple[int, int], FakeCell] = dict(cells or {})

    def bounds(self) -> tuple[int, int, int, int]:
        xs = [x for x, _ in self.cells]
        ys = [y for _, y in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))

    def coverage(self) -> CoverageStats:
        counts = {state.value: 0 for state in CellState}
        for cell in self.cells.values():
            counts[cell.state.value] += 1
        walkable = counts[CellState.WALKABLE.value]
        blocked = counts[CellState.BLOCKED.value]
        temp = counts[CellState.TEMP_BLOCKED.value]
        return CoverageStats(
            area_id=self.area_id,
            cells_total=len(self.cells),
            observed=counts[CellState.OBSERVED.value],
            walkable=walkable,
            blocked=blocked,
            temp_blocked=temp,
            exits=counts[CellState.EXIT.value],
            interactive=counts[CellState.INTERACTIVE.value],
            unknown_frontier=4,
            coverage_ratio=(walkable / (walkable + 4)) if walkable else 0.0,
        )


def make_grid(
    states: dict[tuple[int, int], CellState] | None = None,
    *,
    area_id: str = "test_area",
    width: int = 3,
    height: int = 3,
    default: CellState = CellState.WALKABLE,
) -> FakeGrid:
    """Build a ``FakeGrid`` from a ``{(x, y): CellState}`` map (default fill)."""
    states = dict(states or {})
    cells: dict[tuple[int, int], FakeCell] = {}
    for y in range(height):
        for x in range(width):
            cells[(x, y)] = FakeCell(x, y, states.get((x, y), default))
    for (x, y), state in states.items():
        cells[(x, y)] = FakeCell(x, y, state)
    return FakeGrid(area_id, cells)


class FakeArea:
    """Minimal stand-in for ``world_model.AreaRecord``."""

    def __init__(
        self,
        area_id: str,
        *,
        kind: str = "unknown",
        grid: FakeGrid | None = None,
        visits: int = 0,
        entry_cells: list[tuple[int, int]] | None = None,
        last_frame: int = 0,
    ) -> None:
        self.area_id = area_id
        self.kind = kind
        self.grid = grid
        self.visits = visits
        self.entry_cells = list(entry_cells or [(0, 0)])
        self.last_frame = last_frame


class FakeConnection:
    """Minimal stand-in for ``world_model.WorldConnection``."""

    def __init__(
        self,
        from_area: str,
        to_area: str,
        *,
        kind: str = "warp",
        direction: str = "down",
    ) -> None:
        self.from_area = from_area
        self.to_area = to_area
        self.kind = kind
        self.direction = direction


class FakeWorld:
    """Minimal stand-in for ``world_model.WorldModel``."""

    def __init__(
        self,
        areas: dict[str, FakeArea] | None = None,
        connections: list[FakeConnection] | None = None,
        *,
        current_area: str = "",
        current_pose: tuple[int, int] | None = (0, 0),
    ) -> None:
        self.areas = dict(areas or {})
        self.connections = list(connections or [])
        self.current_area = current_area
        self.current_pose = current_pose


def make_world(count: int = 3, *, current: str = "area_0") -> FakeWorld:
    areas: dict[str, FakeArea] = {}
    for index in range(count):
        area_id = f"area_{index}"
        grid = make_grid(area_id=area_id, width=4, height=3)
        areas[area_id] = FakeArea(
            area_id,
            kind="interior" if index % 2 == 0 else "exterior",
            grid=grid,
            visits=index,
            entry_cells=[(1, 1)],
            last_frame=100 * (index + 1),
        )
    connections = [
        FakeConnection("area_0", "area_1", kind="warp", direction="down"),
        FakeConnection("area_1", "area_2", kind="walk", direction="up"),
    ]
    return FakeWorld(areas, connections, current_area=current, current_pose=(1, 1))


# --------------------------------------------------------------------------- #
# Image inspection helpers
# --------------------------------------------------------------------------- #
def _pixel(image: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
    return tuple(int(value) for value in image[y, x, :3])


def _contains(image: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(np.any(np.all(image[..., :3] == np.asarray(color, dtype=np.uint8), axis=-1)))


def _count_color(image: np.ndarray, color: tuple[int, int, int]) -> int:
    return int(np.all(image[..., :3] == np.asarray(color, dtype=np.uint8), axis=-1).sum())


def _non_background(image: np.ndarray) -> int:
    """Pixels not one of the flat chrome colours (i.e. text / glyph pixels)."""
    rgb = image[..., :3]
    chrome = [
        viz.PALETTE["background"],
        viz.PALETTE["panel"],
        viz.PALETTE["panel_alt"],
        viz.PALETTE["panel_border"],
        viz.PALETTE["coverage_bg"],
    ]
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for color in chrome:
        mask |= np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
    return int(rgb.shape[0] * rgb.shape[1] - mask.sum())


def _assert_rgba(image: np.ndarray, height: int, width: int) -> None:
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.shape == (height, width, 4)
    assert int(image[..., 3].min()) == 255, "composite/render output must be opaque"


# =========================================================================== #
# Module hygiene / palette
# =========================================================================== #
REQUIRED_FUNCTIONS = (
    "render_map_image",
    "render_world_image",
    "render_status_panel",
    "composite_viewer",
    "coverage_bar",
    "describe_cell",
)


def test_module_exports_contract_functions() -> None:
    for name in REQUIRED_FUNCTIONS:
        assert callable(getattr(viz, name)), f"missing viz.{name}"
    assert set(viz.__all__) >= set(REQUIRED_FUNCTIONS)


def test_module_imports_no_tk_or_pyboy_and_no_runtime_other_owners() -> None:
    source = Path(viz.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for module in imported:
        lowered = module.lower()
        assert "tkinter" not in lowered, f"viz must not import Tk ({module})"
        assert "pyboy" not in lowered, f"viz must not import PyBoy ({module})"
        assert "requests" not in lowered and "httpx" not in lowered
    # grid_map / world_model are only imported under TYPE_CHECKING, so they must
    # not exist as runtime attributes (important while other owners build them).
    assert "GridMap" not in vars(viz)
    assert "WorldModel" not in vars(viz)


def test_palette_is_fixed_and_complete() -> None:
    required = (
        "walkable",
        "observed",
        "blocked",
        "temp_blocked",
        "exit",
        "interactive",
        "unknown",
        "player",
        "path",
        "target",
    )
    for name in required:
        color = viz.PALETTE[name]
        assert isinstance(color, tuple) and len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)
    colors = [viz.PALETTE[name] for name in required]
    # Palette colours must be unambiguous for pixel-level assertions.
    assert len(set(colors)) == len(colors)
    assert viz.PALETTE["player"] == (255, 255, 255)
    assert viz.PALETTE["target"] == viz.TARGET_RING_COLOR


def test_every_cell_state_has_its_own_colour() -> None:
    assert set(viz.CELL_STATE_COLORS) == {state.value for state in CellState}
    assert len(set(viz.CELL_STATE_COLORS.values())) == len(viz.CELL_STATE_COLORS)
    assert viz.cell_color(FakeCell(0, 0, CellState.WALKABLE)) == viz.PALETTE["walkable"]
    # String / alias states are normalised instead of raising.
    assert viz.cell_color(SimpleNamespace(state="temp-blocked")) == viz.PALETTE["temp_blocked"]
    assert viz.cell_color(SimpleNamespace(state="bogus")) == viz.PALETTE["unknown"]


# =========================================================================== #
# render_map_image: shape / dtype / determinism
# =========================================================================== #
def test_render_map_image_shape_dtype_and_alpha() -> None:
    grid = make_grid(width=4, height=3)
    image = viz.render_map_image(grid, legend=False, cell_px=8, max_cells=32)
    _assert_rgba(image, 3 * 8, 4 * 8)
    assert image.flags["C_CONTIGUOUS"]


def test_render_map_image_legend_grows_image() -> None:
    grid = make_grid(width=2, height=2)
    plain = viz.render_map_image(grid, legend=False, cell_px=6, max_cells=8)
    with_legend = viz.render_map_image(grid, legend=True, cell_px=6, max_cells=8)
    assert plain.shape == (2 * 6, 2 * 6, 4)
    assert with_legend.shape[0] > plain.shape[0]
    # Legend image is at least _LEGEND_MIN_WIDTH wide so text fits.
    assert with_legend.shape[1] >= 240
    # 12 swatches + coverage text -> a good number of non-chrome pixels.
    assert _non_background(with_legend) > 200


def test_render_map_image_deterministic() -> None:
    grid = make_grid(width=6, height=5)
    path = [(0, 0), (1, 1), (2, 2)]
    kwargs = dict(pose=PlayerPose(2, 2, facing="down"), path=path, target=(4, 4), cell_px=9)
    first = viz.render_map_image(grid, legend=True, **kwargs)
    second = viz.render_map_image(grid, legend=True, **kwargs)
    assert first.tobytes() == second.tobytes()
    first_plain = viz.render_map_image(grid, legend=False, **kwargs)
    second_plain = viz.render_map_image(grid, legend=False, **kwargs)
    assert first_plain.tobytes() == second_plain.tobytes()


# =========================================================================== #
# render_map_image: palette per state
# =========================================================================== #
def test_palette_colour_per_cell_state() -> None:
    states = [
        WALKABLE,
        OBSERVED,
        BLOCKED,
        TEMP_BLOCKED,
        EXIT,
        INTERACTIVE,
        UNKNOWN,
        UNREACHABLE,
        DEFERRED,
    ]
    layout = {(index % 3, index // 3): state for index, state in enumerate(states)}
    grid = make_grid(layout, width=3, height=3)
    cell_px = 10
    image = viz.render_map_image(grid, legend=False, cell_px=cell_px, max_cells=16)
    for index, state in enumerate(states):
        x, y = index % 3, index // 3
        expected = viz.CELL_STATE_COLORS[state.value]
        got = _pixel(image, x * cell_px + cell_px // 2, y * cell_px + cell_px // 2)
        assert got == expected, f"state {state.value}: {got} != {expected}"


def test_unknown_hole_inside_window_is_dark() -> None:
    # Bounds span (0,0)..(2,2) but the centre cell was never observed.
    cells = {
        (0, 0): FakeCell(0, 0, WALKABLE),
        (2, 0): FakeCell(2, 0, WALKABLE),
        (0, 2): FakeCell(0, 2, WALKABLE),
        (2, 2): FakeCell(2, 2, WALKABLE),
    }
    grid = FakeGrid("holes", cells)
    image = viz.render_map_image(grid, legend=False, cell_px=10)
    assert _pixel(image, 15, 15) == viz.PALETTE["unknown"]


# =========================================================================== #
# render_map_image: overlays (player / path / target)
# =========================================================================== #
def test_player_marker_is_white_and_drawn_on_top() -> None:
    grid = make_grid(width=3, height=3)
    image = viz.render_map_image(
        grid, pose=PlayerPose(x=1, y=1, facing="down"), legend=False, cell_px=10
    )
    assert _pixel(image, 15, 15) == viz.PALETTE["player"]
    # Neighbouring cells still show their map state.
    assert _pixel(image, 5, 15) == viz.PALETTE["walkable"]


def test_path_cells_are_magenta() -> None:
    grid = make_grid(width=4, height=4)
    path = [(0, 0), (1, 1), (2, 2)]
    image = viz.render_map_image(grid, path=path, legend=False, cell_px=10)
    for x, y in path:
        assert _pixel(image, x * 10 + 5, y * 10 + 5) == viz.PALETTE["path"]
    # Off-path walkable cell keeps the walkable colour.
    assert _pixel(image, 3 * 10 + 5, 0 * 10 + 5) == viz.PALETTE["walkable"]


def test_target_ring_is_cyan_around_target_cell() -> None:
    grid = make_grid(width=4, height=3)
    image = viz.render_map_image(grid, target=(2, 1), legend=False, cell_px=10)
    # Ring pixels on the cell border are the cyan target colour...
    assert _pixel(image, 25, 10) == viz.PALETTE["target"]  # top edge
    assert _pixel(image, 25, 19) == viz.PALETTE["target"]  # bottom edge
    assert _pixel(image, 20, 15) == viz.PALETTE["target"]  # left edge
    # ...while the cell interior keeps the cell state colour.
    assert _pixel(image, 25, 15) == viz.PALETTE["walkable"]
    assert _count_color(image, viz.PALETTE["target"]) < 4 * 10  # thin ring, not a fill


def test_target_and_player_and_path_together_are_deterministic() -> None:
    grid = make_grid(width=5, height=5)
    kwargs = dict(
        pose={"x": 2, "y": 3, "facing": "up", "confidence": 0.9, "valid": True},
        path=[(0, 0), (1, 0), (2, 0)],
        target=(4, 4),
        cell_px=7,
        max_cells=32,
    )
    first = viz.render_map_image(grid, legend=True, **kwargs)
    second = viz.render_map_image(grid, legend=True, **kwargs)
    assert first.tobytes() == second.tobytes()
    assert _contains(first, viz.PALETTE["player"])
    assert _contains(first, viz.PALETTE["path"])
    assert _contains(first, viz.PALETTE["target"])


def test_pose_variants_render_identically() -> None:
    grid = make_grid(width=4, height=4)
    pose_object = PlayerPose(x=1, y=2, facing="left", confidence=1.0)
    a = viz.render_map_image(grid, pose=pose_object, legend=False, cell_px=8)
    b = viz.render_map_image(grid, pose={"x": 1, "y": 2}, legend=False, cell_px=8)
    c = viz.render_map_image(grid, pose=(1, 2), legend=False, cell_px=8)
    assert a.tobytes() == b.tobytes() == c.tobytes()


# =========================================================================== #
# render_map_image: windowing (max_cells) / tolerance
# =========================================================================== #
def test_max_cells_windows_around_player() -> None:
    grid = make_grid(width=20, height=20)
    image = viz.render_map_image(
        grid, pose=(10, 10), legend=False, cell_px=4, max_cells=8
    )
    assert image.shape == (8 * 4, 8 * 4, 4)
    # The player marker must be visible and the window limited to max_cells.
    assert _contains(image, viz.PALETTE["player"])
    assert int(np.all(image[..., :3] == np.asarray(viz.PALETTE["player"]), axis=-1).sum()) < 8 * 8


def test_max_cells_caps_large_grid() -> None:
    grid = make_grid(width=60, height=40)
    image = viz.render_map_image(grid, pose=(30, 20), legend=False, cell_px=5, max_cells=16)
    assert image.shape == (16 * 5, 16 * 5, 4)


def test_small_grid_renders_completely() -> None:
    grid = make_grid(width=5, height=4)
    image = viz.render_map_image(grid, legend=False, cell_px=6, max_cells=64)
    assert image.shape == (4 * 6, 5 * 6, 4)


def test_empty_and_none_grids_are_tolerated() -> None:
    empty = SimpleNamespace(cells={})
    image = viz.render_map_image(empty, legend=False, cell_px=5)
    _assert_rgba(image, 8 * 5, 8 * 5)
    assert _contains(image, viz.PALETTE["unknown"])
    none_image = viz.render_map_image(None, legend=True)
    _assert_rgba(none_image, none_image.shape[0], none_image.shape[1])
    assert none_image.shape[1] >= 240


def test_grid_with_only_cells_mapping_is_tolerated() -> None:
    bare = SimpleNamespace(
        cells={(0, 0): FakeCell(0, 0, WALKABLE), (1, 1): FakeCell(1, 1, BLOCKED)}
    )
    image = viz.render_map_image(bare, legend=False, cell_px=10)
    _assert_rgba(image, 2 * 10, 2 * 10)
    assert _pixel(image, 5, 5) == viz.PALETTE["walkable"]
    assert _pixel(image, 15, 15) == viz.PALETTE["blocked"]


def test_pose_outside_grid_is_tolerated() -> None:
    grid = make_grid(width=3, height=3)
    image = viz.render_map_image(grid, pose=(100, -50), legend=False, cell_px=6)
    assert image.shape[2] == 4
    assert not _contains(image, viz.PALETTE["player"])


def test_legend_toggle_presence_of_swatches() -> None:
    grid = make_grid(width=3, height=3)
    plain = viz.render_map_image(grid, legend=False, cell_px=8)
    legend = viz.render_map_image(grid, legend=True, cell_px=8)
    # No player/path/target in the map itself: only the legend swatches carry them.
    assert not _contains(plain, viz.PALETTE["target"])
    assert _contains(legend, viz.PALETTE["target"])
    assert _contains(legend, viz.PALETTE["path"])
    assert _contains(legend, viz.PALETTE["player"])


def test_legend_coverage_text_reflects_area_id() -> None:
    first = viz.render_map_image(make_grid(area_id="area_alpha"), legend=True, cell_px=8)
    second = viz.render_map_image(make_grid(area_id="area_beta"), legend=True, cell_px=8)
    assert not np.array_equal(first, second), "legend must include the area id / coverage text"


def test_legend_coverage_uses_coverage_stats() -> None:
    coverage = CoverageStats(
        area_id="cov_area",
        cells_total=42,
        walkable=20,
        blocked=5,
        unknown_frontier=10,
        coverage_ratio=0.75,
    )
    grid = FakeGrid("cov_area", make_grid(width=2, height=2).cells)
    grid.coverage = lambda: coverage  # type: ignore[method-assign]
    image = viz.render_map_image(grid, legend=True, cell_px=8)
    assert _non_background(image) > 200


# =========================================================================== #
# render_status_panel
# =========================================================================== #
def test_status_panel_default_shape_and_text() -> None:
    status = {
        "area_id": "pallet_town",
        "area_kind": "exterior",
        "pose": {"x": 4, "y": 5, "facing": "down", "confidence": 0.97, "valid": True},
        "cells_total": 120,
        "walkable": 55,
        "blocked": 20,
        "unknown_frontier": 7,
        "coverage_ratio": 0.62,
        "areas_total": 3,
        "last_event": "none",
        "paused": False,
        "frame_index": 4321,
    }
    image = viz.render_status_panel(status)
    _assert_rgba(image, 160, 320)
    assert _non_background(image) > 300
    assert _contains(image, viz.PALETTE["coverage_fill"])


def test_status_panel_missing_keys_are_tolerated() -> None:
    for status in ({}, None, {"area_id": "only_area"}, {"pose": None}):
        image = viz.render_status_panel(status)  # type: ignore[arg-type]
        _assert_rgba(image, 160, 320)
        assert _non_background(image) > 200


def test_status_panel_tolerant_of_attribute_object() -> None:
    status = SimpleNamespace(
        area_id="lab",
        area_kind="interior",
        pose=PlayerPose(x=1, y=2, facing="up", confidence=0.5),
        cells_total=10,
        walkable=6,
        blocked=2,
        unknown_frontier=2,
        coverage_ratio=0.5,
        areas_total=1,
        last_event="dialog_event",
        paused=True,
    )
    image = viz.render_status_panel(status)  # type: ignore[arg-type]
    _assert_rgba(image, 160, 320)
    assert _contains(image, viz.PALETTE["paused"])


def test_status_panel_coverage_stats_nested() -> None:
    coverage = CoverageStats(cells_total=30, walkable=10, unknown_frontier=5, coverage_ratio=0.5)
    image = viz.render_status_panel({"coverage": coverage, "coverage_ratio": None})
    _assert_rgba(image, 160, 320)
    assert _contains(image, viz.PALETTE["coverage_fill"])


def test_status_panel_paused_changes_image() -> None:
    paused = viz.render_status_panel({"area_id": "x", "paused": True})
    running = viz.render_status_panel({"area_id": "x", "paused": False})
    assert not np.array_equal(paused, running)


def test_status_panel_custom_size_and_determinism() -> None:
    status = {"area_id": "custom", "walkable": 3, "blocked": 1}
    first = viz.render_status_panel(status, width=200, height=90)
    second = viz.render_status_panel(status, width=200, height=90)
    _assert_rgba(first, 90, 200)
    assert first.tobytes() == second.tobytes()


# =========================================================================== #
# render_world_image
# =========================================================================== #
def test_world_image_shape_and_graph_pixels() -> None:
    world = make_world(4)
    image = viz.render_world_image(world)
    _assert_rgba(image, 160, 320)
    assert _non_background(image) > 300
    assert _contains(image, viz.PALETTE["current_area"])  # highlighted current area


def test_world_image_custom_size() -> None:
    world = make_world(3)
    image = viz.render_world_image(world, width=600, height=300)
    _assert_rgba(image, 300, 600)
    assert _non_background(image) > 300


def test_world_image_deterministic_and_current_area_highlight() -> None:
    world = make_world(3, current="area_0")
    first = viz.render_world_image(world)
    second = viz.render_world_image(world)
    assert first.tobytes() == second.tobytes()
    other = make_world(3, current="area_2")
    assert not np.array_equal(first, viz.render_world_image(other))


def test_world_image_empty_world_tolerated() -> None:
    image = viz.render_world_image(FakeWorld())
    _assert_rgba(image, 160, 320)
    assert _non_background(image) > 50  # "no areas recorded" title text


def test_world_image_duck_dict_world_tolerated() -> None:
    grid = make_grid(width=3, height=3)
    world = {
        "areas": {"a": {"area_id": "a", "kind": "interior", "visits": 1, "grid": grid}},
        "connections": [{"from_area": "a", "to_area": "a", "kind": "walk", "direction": "up"}],
        "current_area": "a",
    }
    image = viz.render_world_image(world)  # type: ignore[arg-type]
    _assert_rgba(image, 160, 320)


def test_world_image_many_areas_is_clipped_not_crashing() -> None:
    world = make_world(30)
    image = viz.render_world_image(world, width=240, height=140)
    _assert_rgba(image, 140, 240)


def test_world_image_connection_labels_drawn() -> None:
    grid = make_grid(width=3, height=3)
    a = FakeArea("a", grid=grid)
    b = FakeArea("b", grid=grid)
    world = FakeWorld(
        {"a": a, "b": b},
        [FakeConnection("a", "b", kind="warp")],
        current_area="a",
    )
    image = viz.render_world_image(world, width=400, height=200)
    # The warp label uses the connection colour (a non-chrome pixel colour).
    assert _contains(image, viz.PALETTE["connection_warp"])


# =========================================================================== #
# coverage_bar
# =========================================================================== #
def test_coverage_bar_shape_dtype_and_fill_ratio() -> None:
    width = 60
    inner = width - 2
    empty = viz.coverage_bar({"coverage_ratio": 0.0}, width=width)
    _assert_rgba(empty, 10, width)
    assert _count_color(empty, viz.PALETTE["coverage_fill"]) == 0
    full = viz.coverage_bar({"coverage_ratio": 1.0}, width=width)
    assert _count_color(full, viz.PALETTE["coverage_fill"]) == inner * 8
    half = viz.coverage_bar({"coverage_ratio": 0.5}, width=width)
    expected = int(round(0.5 * inner)) * 8
    assert _count_color(half, viz.PALETTE["coverage_fill"]) == expected


def test_coverage_bar_tolerates_inputs() -> None:
    inner = 46
    for coverage, expected_ratio in (
        (None, 0.0),
        ({}, 0.0),
        ({"walkable": 10, "unknown_frontier": 10}, 0.5),
        (CoverageStats(walkable=10, unknown_frontier=10, coverage_ratio=0.5), 0.5),
        (0.25, 0.25),
        (float("nan"), 0.0),
        (2.0, 1.0),
        (-1.0, 0.0),
    ):
        image = viz.coverage_bar(coverage, width=48)  # type: ignore[arg-type]
        _assert_rgba(image, 10, 48)
        filled = _count_color(image, viz.PALETTE["coverage_fill"]) // 8
        expected = int(round(expected_ratio * inner))
        assert filled == expected, f"{coverage!r}: {filled} != {expected}"


def test_coverage_bar_deterministic() -> None:
    first = viz.coverage_bar(CoverageStats(coverage_ratio=0.42), width=100)
    second = viz.coverage_bar(CoverageStats(coverage_ratio=0.42), width=100)
    assert first.tobytes() == second.tobytes()


# =========================================================================== #
# describe_cell
# =========================================================================== #
def test_describe_cell_full_fields() -> None:
    cell = FakeCell(
        3,
        4,
        WALKABLE,
        terrain="abcd1234",
        visit_count=2,
        observed_count=7,
        confidence=0.8,
        labels=["sign"],
    )
    text = viz.describe_cell(cell)
    assert isinstance(text, str)
    assert "(3, 4)" in text
    assert "walkable" in text
    assert "visits=2" in text
    assert "observed=7" in text
    assert "conf=0.80" in text
    assert "sign" in text
    assert len(text) <= 120


def test_describe_cell_tolerates_dicts_none_and_bare_objects() -> None:
    assert viz.describe_cell(None) == "no cell"
    mapping = viz.describe_cell({"x": 1, "y": 2, "state": "blocked", "terrain": "wall"})
    assert "(1, 2)" in mapping and "blocked" in mapping and "terrain=wall" in mapping
    bare = viz.describe_cell(SimpleNamespace(x=0, y=0))
    assert "(0, 0)" in bare and "unknown" in bare
    empty = viz.describe_cell({})
    assert "unknown" in empty and "?" in empty


# =========================================================================== #
# composite_viewer
# =========================================================================== #
@pytest.fixture()
def synthetic_frame(open_room) -> np.ndarray:  # noqa: ANN001 - conftest fixture
    """A real 160x144 RGBA frame from the shared synthetic room."""
    return open_room.render((5, 4), "down")


def test_composite_shape_with_and_without_frame(synthetic_frame) -> None:
    without = viz.composite_viewer({})
    _assert_rgba(without, 360, 480)
    with_frame = viz.composite_viewer({}, frame_rgba=synthetic_frame)
    _assert_rgba(with_frame, 360, 480)
    assert not np.array_equal(without, with_frame)


def test_composite_frame_is_scaled_2x(synthetic_frame) -> None:
    frame = np.zeros((144, 160, 4), dtype=np.uint8)
    frame[..., 0] = 60
    frame[..., 1] = 120
    frame[..., 2] = 180
    frame[..., 3] = 255
    image = viz.composite_viewer({}, frame_rgba=frame)
    # Game image is pasted at (4, 4) scaled by exactly 2 -> this colour dominates.
    expected = (60, 120, 180)
    assert _pixel(image, 4 + 10, 4 + 10) == expected
    assert _count_color(image, expected) >= 320 * 288 * 0.95


def test_composite_frame_accepts_frame_like_objects(synthetic_frame) -> None:
    as_array = viz.composite_viewer({}, frame_rgba=synthetic_frame)
    as_object = viz.composite_viewer({}, frame_rgba=SimpleNamespace(rgba=synthetic_frame))
    assert as_array.tobytes() == as_object.tobytes()
    rgb = np.ascontiguousarray(synthetic_frame[..., :3])
    as_rgb = viz.composite_viewer({}, frame_rgba=rgb)
    assert as_array.tobytes() == as_rgb.tobytes()


def test_composite_renders_grid_palette_and_player() -> None:
    grid = make_grid(width=6, height=5)
    state = {
        "grid": grid,
        "pose": PlayerPose(x=2, y=2, facing="down"),
        "path": [(0, 0), (1, 0), (2, 0)],
        "target": (4, 3),
        "status": {"area_id": "lab", "area_kind": "interior", "walkable": 18},
        "frame_index": 1000,
        "paused": False,
    }
    image = viz.composite_viewer(state)
    _assert_rgba(image, 360, 480)
    assert _contains(image, viz.PALETTE["walkable"])
    assert _contains(image, viz.PALETTE["player"])
    assert _contains(image, viz.PALETTE["path"])
    assert _contains(image, viz.PALETTE["target"])
    # Status panel and bottom strip text are present.
    assert _non_background(image) > 1000


def test_composite_with_frame_and_full_state(synthetic_frame) -> None:
    world = make_world(2)
    state = {
        "grid": make_grid(width=5, height=4),
        "pose": (2, 2),
        "status": {"area_id": "pallet", "areas_total": 2},
        "world": world,
        "frame_index": 55,
        "last_action": "press_up",
    }
    image = viz.composite_viewer(state, frame_rgba=synthetic_frame)
    _assert_rgba(image, 360, 480)
    assert _contains(image, viz.PALETTE["walkable"])


def test_composite_missing_keys_tolerated() -> None:
    for state in ({}, None, {"area_id": "x"}, {"grid": None, "pose": None}):
        image = viz.composite_viewer(state)  # type: ignore[arg-type]
        _assert_rgba(image, 360, 480)
        assert _contains(image, viz.PALETTE["text_faint"]) or _non_background(image) > 20


def test_composite_deterministic(synthetic_frame) -> None:
    state = {
        "grid": make_grid(width=4, height=3),
        "pose": (1, 1),
        "path": [(0, 0), (1, 0)],
        "target": (2, 2),
        "status": {"area_id": "det", "walkable": 5},
        "frame_index": 7,
    }
    first = viz.composite_viewer(state, frame_rgba=synthetic_frame)
    second = viz.composite_viewer(state, frame_rgba=synthetic_frame)
    assert first.tobytes() == second.tobytes()


def test_composite_custom_size(synthetic_frame) -> None:
    image = viz.composite_viewer({}, frame_rgba=synthetic_frame, width=240, height=180)
    _assert_rgba(image, 180, 240)


def test_composite_flat_status_keys_on_state() -> None:
    image_a = viz.composite_viewer({"area_id": "flat", "walkable": 9, "blocked": 2})
    image_b = viz.composite_viewer(
        {"status": {"area_id": "flat", "walkable": 9, "blocked": 2}}
    )
    _assert_rgba(image_a, 360, 480)
    _assert_rgba(image_b, 360, 480)
    # Flat status keys are merged into the status panel, so both layouts match.
    assert image_a.tobytes() == image_b.tobytes()


# =========================================================================== #
# Performance smoke tests (10 fps target; generous bounds to avoid flakiness)
# =========================================================================== #
def test_render_map_is_cheap_enough() -> None:
    grid = make_grid(width=20, height=20)
    path = [(index, index) for index in range(12)]
    start = time.perf_counter()
    for _ in range(20):
        viz.render_map_image(
            grid, pose=(10, 10), path=path, target=(19, 19), cell_px=7, max_cells=32
        )
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"20 map renders took {elapsed:.3f}s (>150 ms each)"


def test_composite_is_cheap_enough(synthetic_frame) -> None:
    grid = make_grid(width=20, height=20)
    state = {
        "grid": grid,
        "pose": (10, 10),
        "path": [(index, index) for index in range(12)],
        "target": (19, 19),
        "status": {"area_id": "perf", "walkable": 40, "blocked": 5, "coverage_ratio": 0.5},
    }
    start = time.perf_counter()
    for _ in range(20):
        viz.composite_viewer(state, frame_rgba=synthetic_frame)
    elapsed = time.perf_counter() - start
    assert elapsed < 4.0, f"20 composites took {elapsed:.3f}s (>200 ms each)"


# =========================================================================== #
# Map gallery CLI (scripts/nav_v2_map_gallery.py)
# =========================================================================== #
def _load_gallery_module(name: str = "nav_v2_map_gallery_under_test"):
    spec = importlib.util.spec_from_file_location(name, GALLERY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def gallery_module():
    return _load_gallery_module()


class FakeWorldModelForGallery:
    """Fake ``WorldModel``: only the API surface used by the gallery script."""

    def __init__(self, *, root=None) -> None:
        self.root = root
        self.areas = {
            "oaks_lab": FakeArea(
                "oaks_lab",
                kind="interior",
                grid=make_grid(width=5, height=4),
                visits=2,
                entry_cells=[(1, 2)],
                last_frame=120,
            ),
            "pallet_town": FakeArea(
                "pallet_town",
                kind="exterior",
                grid=make_grid(width=7, height=5),
                visits=5,
                entry_cells=[(3, 3)],
                last_frame=240,
            ),
        }
        self.connections = [
            FakeConnection("oaks_lab", "pallet_town", kind="warp", direction="down")
        ]
        self.current_area = "pallet_town"
        self.current_pose = (3, 3)

    def load(self) -> bool:
        return True

    def summary(self) -> dict:
        return {"areas": len(self.areas), "connections": len(self.connections)}

    def render_summary(self) -> str:
        return "oaks_lab -> pallet_town (warp)"


@pytest.fixture()
def fake_world_dir(tmp_path: Path) -> Path:
    world_dir = tmp_path / "navigation"
    (world_dir / "areas").mkdir(parents=True)
    (world_dir / "world.json").write_text('{"version": 1}', encoding="utf-8")
    (world_dir / "areas" / "oaks_lab.json").write_text("{}", encoding="utf-8")
    return world_dir


def test_gallery_help_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, str(GALLERY_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--world-dir" in result.stdout
    assert "--out" in result.stdout
    assert "--cell-px" in result.stdout and "--max-cells" in result.stdout


def test_gallery_missing_world_dir_fails_gracefully_subprocess(tmp_path: Path) -> None:
    out_dir = tmp_path / "gallery"
    missing = tmp_path / "does_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            str(GALLERY_SCRIPT),
            "--world-dir",
            str(missing),
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode != 0
    assert "world model directory not found" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out_dir.exists()


def test_gallery_import_has_no_side_effects(gallery_module) -> None:
    assert callable(gallery_module.main)
    assert callable(gallery_module.build_parser)
    parser = gallery_module.build_parser()
    option_names = {option for action in parser._actions for option in action.option_strings}
    assert {"--world-dir", "--out", "--cell-px", "--max-cells"} <= option_names


def test_gallery_writes_png_gallery_with_fake_world(
    gallery_module, fake_world_dir: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        gallery_module, "_load_world_class", lambda: FakeWorldModelForGallery
    )
    out_dir = tmp_path / "gallery" / "nested"
    code = gallery_module.main(
        ["--world-dir", str(fake_world_dir), "--out", str(out_dir)]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "files=" in captured.out
    assert out_dir.is_dir()
    expected = {
        "map_oaks_lab.png",
        "map_pallet_town.png",
        "world_graph.png",
        "status.png",
        "index.md",
    }
    present = {path.name for path in out_dir.iterdir()}
    assert expected <= present
    from PIL import Image

    for name in ("map_oaks_lab.png", "map_pallet_town.png", "world_graph.png", "status.png"):
        with Image.open(out_dir / name) as image:
            assert image.format == "PNG"
            assert image.size[0] > 10 and image.size[1] > 10
    index_text = (out_dir / "index.md").read_text(encoding="utf-8")
    assert "oaks_lab" in index_text and "pallet_town" in index_text
    assert "world_graph.png" in index_text


def test_gallery_corrupt_world_model_fails(
    gallery_module, fake_world_dir: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    class RaisingWorldModel:
        def __init__(self, *, root=None) -> None:
            pass

        def load(self) -> bool:
            raise ValueError("corrupt world.json")

    monkeypatch.setattr(gallery_module, "_load_world_class", lambda: RaisingWorldModel)
    out_dir = tmp_path / "should_not_exist"
    code = gallery_module.main(
        ["--world-dir", str(fake_world_dir), "--out", str(out_dir)]
    )
    assert code != 0
    stderr = capsys.readouterr().err
    assert "failed to load world model" in stderr
    assert "ValueError" in stderr
    assert not out_dir.exists()


def test_gallery_load_returning_false_fails(
    gallery_module, fake_world_dir: Path, monkeypatch, capsys
) -> None:
    class FalseWorldModel:
        def __init__(self, *, root=None) -> None:
            pass

        def load(self) -> bool:
            return False

    monkeypatch.setattr(gallery_module, "_load_world_class", lambda: FalseWorldModel)
    code = gallery_module.main(["--world-dir", str(fake_world_dir), "--out", str(fake_world_dir / "out")])
    assert code != 0
    assert "load() reported failure" in capsys.readouterr().err


def test_gallery_empty_world_fails(gallery_module, fake_world_dir: Path, monkeypatch, capsys) -> None:
    class EmptyWorldModel:
        def __init__(self, *, root=None) -> None:
            self.areas = {}
            self.connections = []
            self.current_area = ""

        def load(self) -> bool:
            return True

    monkeypatch.setattr(gallery_module, "_load_world_class", lambda: EmptyWorldModel)
    code = gallery_module.main(
        ["--world-dir", str(fake_world_dir), "--out", str(fake_world_dir / "out")]
    )
    assert code != 0
    assert "no areas" in capsys.readouterr().err


def test_gallery_missing_world_json_fails(
    gallery_module, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(gallery_module, "_load_world_class", lambda: FakeWorldModelForGallery)
    world_dir = tmp_path / "incomplete"
    world_dir.mkdir()
    code = gallery_module.main(["--world-dir", str(world_dir), "--out", str(tmp_path / "out")])
    assert code != 0
    assert "world.json missing" in capsys.readouterr().err


def test_gallery_missing_world_model_module_fails(
    gallery_module, fake_world_dir: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    def _raising_import():
        raise ImportError("No module named 'llm_agent.navigation.world_model'")

    monkeypatch.setattr(gallery_module, "_load_world_class", _raising_import)
    code = gallery_module.main(
        ["--world-dir", str(fake_world_dir), "--out", str(tmp_path / "out")]
    )
    assert code != 0
    assert "world_model is unavailable" in capsys.readouterr().err


def test_gallery_real_world_model_if_available(tmp_path: Path) -> None:
    """End-to-end smoke test once nav-map's real WorldModel exists (else skip)."""
    world_model = pytest.importorskip(
        "llm_agent.navigation.world_model",
        reason="world_model.py is owned by nav-map and may not exist yet",
    )
    world_class = getattr(world_model, "WorldModel", None)
    if world_class is None:  # pragma: no cover - contract violation in another module
        pytest.skip("world_model.WorldModel not implemented yet")
    world_root = tmp_path / "navigation"
    try:
        world = world_class(root=world_root)
        area = world.create_area(kind="interior", frame_index=0, area_id="smoke_area")
        grid = area.grid
        for y in range(3):
            for x in range(4):
                grid.mark_state((x, y), CellState.WALKABLE, reason="test")
        grid.mark_state((0, 1), CellState.BLOCKED, reason="test")
        world.set_current("smoke_area", (1, 1))
        world.save()
    except Exception as exc:  # noqa: BLE001 - other owner's module may be incomplete
        pytest.skip(f"real WorldModel API not usable yet: {type(exc).__name__}: {exc}")
    gallery = _load_gallery_module("nav_v2_map_gallery_real_test")
    out_dir = tmp_path / "gallery"
    code = gallery.main(["--world-dir", str(world_root), "--out", str(out_dir)])
    assert code == 0, "gallery must succeed on a freshly saved real world model"
    assert (out_dir / "world_graph.png").is_file()
    assert (out_dir / "status.png").is_file()
    assert any(path.name.startswith("map_") for path in out_dir.iterdir())


def test_render_real_grid_map_if_available() -> None:
    """Palette/rendering smoke test against nav-map's real GridMap (skip if absent)."""
    grid_map = pytest.importorskip(
        "llm_agent.navigation.grid_map",
        reason="grid_map.py is owned by nav-map and may not exist yet",
    )
    grid_class = getattr(grid_map, "GridMap", None)
    if grid_class is None:  # pragma: no cover - contract violation in another module
        pytest.skip("grid_map.GridMap not implemented yet")
    grid = grid_class("real_area")
    states = [
        (0, 0, WALKABLE),
        (1, 0, OBSERVED),
        (2, 0, BLOCKED),
        (0, 1, TEMP_BLOCKED),
        (1, 1, EXIT),
        (2, 1, INTERACTIVE),
    ]
    for x, y, state in states:
        grid.mark_state((x, y), state, reason="viz-test")
    image = viz.render_map_image(
        grid,
        pose=PlayerPose(x=0, y=0, facing="down"),
        path=[(0, 0), (0, 1)],
        target=(2, 1),
        cell_px=10,
        legend=False,
    )
    assert isinstance(image, np.ndarray) and image.dtype == np.uint8
    assert image.shape[:2] == (2 * 10, 3 * 10)
    for _x, _y, state in states:
        assert _contains(image, viz.CELL_STATE_COLORS[state.value]), state.value
    description = viz.describe_cell(grid.cells[(0, 0)])
    assert isinstance(description, str) and "walkable" in description
    coverage = grid.coverage()
    bar = viz.coverage_bar(coverage, width=80)
    _assert_rgba(bar, 10, 80)
