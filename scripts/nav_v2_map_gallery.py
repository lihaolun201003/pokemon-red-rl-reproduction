"""Render a saved Navigation v2 world model into a PNG gallery (report evidence).

The script loads a persisted world model directory (``navigation/world.json`` +
``navigation/areas/<area_id>.json``, as written by
:meth:`llm_agent.navigation.world_model.WorldModel.save`) and renders:

* ``map_<area>.png``      — one grid-map image per area (palette legend + coverage)
* ``world_graph.png``     — the area graph (nodes = areas, edges = connections)
* ``status.png``          — the aggregated navigation status panel
* ``index.md``            — a small Markdown index listing the images + area table

It is the evidence generator for ``docs/reports/navigation_v2_*.md``.

Examples::

    .\\.venv\\Scripts\\python.exe scripts\\nav_v2_map_gallery.py `
        --world-dir artifacts\\nav_v2\\navigation `
        --out artifacts\\nav_v2\\map_gallery

    # default world dir: artifacts/nav_v2/navigation
    .\\.venv\\Scripts\\python.exe scripts\\nav_v2_map_gallery.py

Failure behaviour (never a traceback to the user): a missing/corrupt world model
prints one clear ``[nav_v2_map_gallery] error: ...`` line to stderr and exits
non-zero (2 = missing input, 3 = world_model module unavailable, 4 = corrupt /
empty world model, 5 = render failure, 6 = output directory failure).

This module has **no import side effects** (importing it does not touch disk, the
emulator or the API); ``llm_agent.navigation.world_model`` is imported lazily
inside :func:`main` because it is owned by another module of the round.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:  # Windows consoles: make sure Unicode ids never crash the report
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - non-reconfigurable streams (pytest)
        pass

from llm_agent.navigation import viz  # noqa: E402  (needs PROJECT_ROOT on sys.path)

#: Fallback world model directory (used when ``--world-dir`` is not given).
DEFAULT_WORLD_DIR = PROJECT_ROOT / "artifacts" / "nav_v2" / "navigation"
#: Default output directory for the generated PNG evidence.
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "nav_v2" / "map_gallery"

_EXIT_OK = 0
_EXIT_MISSING_INPUT = 2
_EXIT_NO_MODULE = 3
_EXIT_CORRUPT_WORLD = 4
_EXIT_RENDER_FAILED = 5
_EXIT_OUTPUT_FAILED = 6


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser (kept separate so tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="nav_v2_map_gallery.py",
        description=(
            "Render a saved Navigation v2 world model (navigation/world.json + "
            "areas/*.json) into a PNG map gallery for the report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts\\\\nav_v2_map_gallery.py --world-dir artifacts\\\\nav_v2\\\\navigation\n"
            "  python scripts\\\\nav_v2_map_gallery.py --world-dir <run>\\\\navigation --out work\\\\gallery\n"
        ),
    )
    parser.add_argument(
        "--world-dir",
        default="",
        metavar="DIR",
        help=(
            "saved navigation world model directory containing world.json and "
            "areas/*.json (default: artifacts/nav_v2/navigation)"
        ),
    )
    parser.add_argument(
        "--out",
        default="",
        metavar="DIR",
        help="output directory for the PNG gallery (default: artifacts/nav_v2/map_gallery)",
    )
    parser.add_argument(
        "--cell-px",
        type=int,
        default=10,
        metavar="N",
        help="pixel size of one map cell (default: 10)",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=64,
        metavar="N",
        help="maximum rendered cells per map axis, centred on the player (default: 64)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        metavar="PX",
        help="world-graph and status-panel image width (default: 480)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        metavar="PX",
        help="world-graph and status-panel image height (default: 240)",
    )
    return parser


# --------------------------------------------------------------------------- #
# Lazy dependency + small duck-typing helpers
# --------------------------------------------------------------------------- #

def _load_world_class() -> Any:
    """Lazily import :class:`WorldModel` (owned by the ``nav-map`` module).

    Kept as a separate function so tests can monkeypatch it with a fake class
    while ``world_model.py`` is still being written.
    """
    from llm_agent.navigation.world_model import WorldModel  # noqa: PLC0415

    return WorldModel


def _resolve_path(raw: str, default: Path) -> Path:
    """Resolve a CLI path argument: empty -> default, relative -> CWD."""
    if not raw:
        return default.resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _safe_name(text: str) -> str:
    """Filesystem-safe, deterministic file stem for an area id."""
    cleaned = "".join(char if (char.isalnum() or char in "-_") else "_" for char in str(text))
    cleaned = cleaned.strip("_") or "area"
    return cleaned[:48]


def _relative(path: Path) -> str:
    """Best-effort path relative to the project root (portable evidence)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:  # pragma: no cover - external path (e.g. tmp dir in tests)
        return path.resolve().as_posix()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _state_value(cell: Any) -> str:
    state = getattr(cell, "state", None)
    if state is None and isinstance(cell, dict):
        state = cell.get("state")
    value = getattr(state, "value", state)
    return str(value).strip().lower() if value is not None else "unknown"


def _grid_cells(grid: Any) -> dict[Any, Any]:
    cells = getattr(grid, "cells", None)
    if cells is None and isinstance(grid, dict):
        cells = grid.get("cells")
    if isinstance(cells, dict):
        return cells
    return {}


def _grid_coverage(grid: Any) -> Any:
    if grid is None:
        return None
    coverage_fn = getattr(grid, "coverage", None)
    if callable(coverage_fn):
        try:
            return coverage_fn()
        except Exception:  # pragma: no cover - defensive: keep the gallery going
            return None
    return None


def _coverage_value(coverage: Any, key: str, default: Any = None) -> Any:
    if coverage is None:
        return default
    if isinstance(coverage, dict):
        return coverage.get(key, default)
    return getattr(coverage, key, default)


def _area_grid(area: Any) -> Any:
    return getattr(area, "grid", None)


def _area_cell_counts(area: Any) -> dict[str, int]:
    """Count cell states of one area (coverage-independent fallback)."""
    counts: dict[str, int] = {}
    for cell in _grid_cells(_area_grid(area)).values():
        name = _state_value(cell)
        counts[name] = counts.get(name, 0) + 1
    counts["total"] = sum(value for key, value in counts.items() if key != "total")
    return counts


def _area_numbers(area: Any) -> dict[str, Any]:
    """``walkable``/``blocked``/``cells_total``/``unknown_frontier`` of one area."""
    coverage = _grid_coverage(_area_grid(area))
    counts = _area_cell_counts(area)
    walkable = _coverage_value(coverage, "walkable", counts.get("walkable", 0))
    blocked = _coverage_value(coverage, "blocked", counts.get("blocked", 0))
    temp_blocked = _coverage_value(
        coverage, "temp_blocked", counts.get("temp_blocked", 0)
    )
    cells_total = _coverage_value(coverage, "cells_total", counts.get("total", 0))
    frontier = _coverage_value(coverage, "unknown_frontier", None)
    ratio = _coverage_value(coverage, "coverage_ratio", None)
    if ratio is None and walkable is not None and frontier is not None:
        denominator = _as_int(walkable) + _as_int(frontier)
        ratio = (_as_int(walkable) / denominator) if denominator > 0 else 0.0
    return {
        "walkable": _as_int(walkable),
        "blocked": _as_int(blocked),
        "temp_blocked": _as_int(temp_blocked),
        "cells_total": _as_int(cells_total),
        "unknown_frontier": None if frontier is None else _as_int(frontier, -1),
        "coverage_ratio": None if ratio is None else float(ratio),
    }


def _area_id(area: Any, fallback: str = "") -> str:
    value = getattr(area, "area_id", None)
    if value is None:
        value = fallback
    return str(value) if value is not None else ""


def _area_kind(area: Any) -> str:
    value = getattr(area, "kind", None)
    return str(value) if value else "unknown"


def _area_visits(area: Any) -> int:
    return _as_int(getattr(area, "visits", 0))


def _area_pose(area: Any) -> tuple[int, int] | None:
    """Best-known local pose inside an area (entry cell or current pose)."""
    entry_cells = getattr(area, "entry_cells", None)
    if isinstance(entry_cells, (list, tuple)) and entry_cells:
        first = entry_cells[0]
        try:
            return (int(first[0]), int(first[1]))
        except (TypeError, ValueError, IndexError):
            return None
    return None


# --------------------------------------------------------------------------- #
# World model loading
# --------------------------------------------------------------------------- #

def _validate_world_dir(world_dir: Path) -> str | None:
    """Return an error message when ``world_dir`` is not a saved world model."""
    if not world_dir.is_dir():
        return (
            f"world model directory not found: {world_dir}\n"
            "  pass --world-dir <run>/navigation (the directory that contains "
            "world.json and areas/*.json)"
        )
    if not (world_dir / "world.json").is_file():
        return (
            f"world.json missing in world model directory: {world_dir}\n"
            f"  expected layout: {world_dir / 'world.json'} and {world_dir / 'areas' / '<area_id>.json'}"
        )
    if not (world_dir / "areas").is_dir():
        return (
            f"areas/ directory missing in world model directory: {world_dir}\n"
            "  the world model looks incomplete; run WorldModel.save() first"
        )
    return None


def _load_world(world_dir: Path) -> tuple[Any | None, int, str]:
    """Import + load a world model; returns ``(world, exit_code, error_message)``."""
    try:
        world_class = _load_world_class()
    except ImportError as exc:
        return (
            None,
            _EXIT_NO_MODULE,
            f"llm_agent.navigation.world_model is unavailable: {exc}\n"
            "  the module is owned by the nav-map workstream; retry once it lands",
        )
    try:
        world = world_class(root=world_dir)
    except Exception as exc:  # noqa: BLE001 - never traceback at the user
        return (
            None,
            _EXIT_CORRUPT_WORLD,
            f"failed to open world model {world_dir}: {type(exc).__name__}: {exc}",
        )
    try:
        loaded = world.load()
    except Exception as exc:  # noqa: BLE001 - corrupt JSON is a normal failure mode
        return (
            None,
            _EXIT_CORRUPT_WORLD,
            f"failed to load world model {world_dir}: {type(exc).__name__}: {exc}",
        )
    if loaded is False:
        return (
            None,
            _EXIT_CORRUPT_WORLD,
            f"world model load() reported failure for {world_dir} "
            "(corrupt or incompatible world.json)",
        )
    areas = getattr(world, "areas", None)
    if not areas:
        return (
            None,
            _EXIT_CORRUPT_WORLD,
            f"world model {world_dir} contains no areas; nothing to render",
        )
    return (world, _EXIT_OK, "")


def _areas_dict(world: Any) -> dict[str, Any]:
    areas = getattr(world, "areas", None)
    return dict(areas) if isinstance(areas, dict) else {}


def _connections_list(world: Any) -> list[Any]:
    connections = getattr(world, "connections", None)
    return list(connections) if isinstance(connections, (list, tuple)) else []


# --------------------------------------------------------------------------- #
# Status / summary building
# --------------------------------------------------------------------------- #

def _build_status(world: Any) -> dict[str, Any]:
    """Aggregate a tolerant ``NavigationAgent.status()``-shaped dict for a world."""
    areas = _areas_dict(world)
    connections = _connections_list(world)
    current_area = str(getattr(world, "current_area", "") or "")
    current_pose = getattr(world, "current_pose", None)
    current_record = areas.get(current_area) if current_area else None
    total_walkable = total_blocked = total_frontier = total_cells = 0
    latest_frame = 0
    for area in areas.values():
        numbers = _area_numbers(area)
        total_walkable += numbers["walkable"]
        total_blocked += numbers["blocked"]
        total_cells += numbers["cells_total"]
        if numbers["unknown_frontier"] is not None:
            total_frontier += max(0, numbers["unknown_frontier"])
        latest_frame = max(latest_frame, _as_int(getattr(area, "last_frame", 0)))
    denominator = total_walkable + total_frontier
    coverage_ratio = (total_walkable / denominator) if denominator > 0 else 0.0
    pose_dict: dict[str, Any] | None = None
    if isinstance(current_pose, (tuple, list)) and len(current_pose) >= 2:
        pose_dict = {
            "x": _as_int(current_pose[0]),
            "y": _as_int(current_pose[1]),
            "facing": None,
            "confidence": 1.0,
            "valid": True,
        }
    world_areas: list[list[Any]] = []
    for area_id, area in sorted(areas.items(), key=lambda item: str(item[0])):
        numbers = _area_numbers(area)
        world_areas.append(
            [
                _area_id(area, str(area_id)),
                _area_kind(area),
                _area_visits(area),
                numbers["walkable"],
            ]
        )
    return {
        "area_id": current_area or "-",
        "area_kind": _area_kind(current_record) if current_record is not None else "-",
        "pose": pose_dict,
        "cells_total": total_cells,
        "walkable": total_walkable,
        "blocked": total_blocked,
        "temp_blocked": sum(_area_numbers(area)["temp_blocked"] for area in areas.values()),
        "unknown_frontier": total_frontier,
        "coverage_ratio": coverage_ratio,
        "areas_total": len(areas),
        "last_event": "none",
        "paused": False,
        "frame_index": latest_frame,
        "world": {"areas": world_areas, "connections": len(connections)},
    }


def _index_markdown(
    world_dir: Path,
    status: dict[str, Any],
    world: Any,
    image_rows: list[tuple[str, str]],
) -> str:
    """Deterministic Markdown index describing the rendered gallery."""
    areas = _areas_dict(world)
    lines: list[str] = []
    lines.append("# Navigation v2 — map gallery")
    lines.append("")
    lines.append(f"- world model: `{_relative(world_dir)}`")
    lines.append(
        f"- areas: {len(areas)} / connections: {status['world']['connections']}"
    )
    lines.append(f"- current area: `{status['area_id']}`")
    lines.append(
        f"- cells: {status['cells_total']} total / {status['walkable']} walkable / "
        f"{status['blocked']} blocked / {status['unknown_frontier']} frontier"
    )
    lines.append(f"- coverage: {status['coverage_ratio'] * 100:.1f}%")
    lines.append("")
    lines.append("## Images")
    lines.append("")
    for filename, description in image_rows:
        lines.append(f"- `{filename}` — {description}")
    lines.append("")
    lines.append("## Areas")
    lines.append("")
    lines.append("| area | kind | visits | walkable | cells | coverage |")
    lines.append("|---|---|---|---|---|---|")
    for area_id, area in sorted(areas.items(), key=lambda item: str(item[0])):
        numbers = _area_numbers(area)
        ratio = numbers["coverage_ratio"]
        ratio_text = "-" if ratio is None else f"{ratio * 100:.1f}%"
        lines.append(
            f"| `{_area_id(area, str(area_id))}` | {_area_kind(area)} | "
            f"{_area_visits(area)} | {numbers['walkable']} | "
            f"{numbers['cells_total']} | {ratio_text} |"
        )
    render_summary = getattr(world, "render_summary", None)
    if callable(render_summary):
        try:
            summary_text = str(render_summary() or "").strip()
        except Exception:  # pragma: no cover - summary is a nice-to-have
            summary_text = ""
        if summary_text:
            lines.append("")
            lines.append("## World summary")
            lines.append("")
            lines.append("```")
            lines.extend(summary_text.splitlines()[:80])
            lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Rendering / writing
# --------------------------------------------------------------------------- #

def _save_png(rgba: np.ndarray, path: Path) -> None:
    """Write one RGBA array as a PNG file (Pillow only)."""
    from PIL import Image  # noqa: PLC0415 - local import keeps module import cheap

    pixels = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8))
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(f"expected an (h, w, 4) RGBA array, got {pixels.shape}")
    image = Image.fromarray(pixels)
    image.save(str(path), format="PNG")


def _render_gallery(
    world: Any,
    world_dir: Path,
    out_dir: Path,
    *,
    cell_px: int,
    max_cells: int,
    width: int,
    height: int,
) -> tuple[int, list[tuple[str, str]], int]:
    """Render every gallery image; returns ``(files_written, image_rows, exit_code)``."""
    areas = _areas_dict(world)
    current_area = str(getattr(world, "current_area", "") or "")
    current_pose = getattr(world, "current_pose", None)
    files_written = 0
    image_rows: list[tuple[str, str]] = []
    used_names: dict[str, int] = {}

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"[nav_v2_map_gallery] error: cannot create output directory {out_dir}: {exc}",
            file=sys.stderr,
        )
        return (0, image_rows, _EXIT_OUTPUT_FAILED)

    for area_id, area in sorted(areas.items(), key=lambda item: str(item[0])):
        display_id = _area_id(area, str(area_id))
        grid = _area_grid(area)
        if grid is None:
            print(
                f"[nav_v2_map_gallery] warning: area {display_id!r} has no grid; skipped",
                file=sys.stderr,
            )
            continue
        pose = None
        if str(area_id) == current_area:
            if isinstance(current_pose, (tuple, list)) and len(current_pose) >= 2:
                pose = (_as_int(current_pose[0]), _as_int(current_pose[1]))
        if pose is None:
            pose = _area_pose(area)
        try:
            image = viz.render_map_image(
                grid,
                pose=pose,
                cell_px=cell_px,
                max_cells=max_cells,
                legend=True,
            )
        except Exception as exc:  # noqa: BLE001 - report, do not traceback
            print(
                f"[nav_v2_map_gallery] error: failed to render area {display_id!r}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return (files_written, image_rows, _EXIT_RENDER_FAILED)
        stem = _safe_name(display_id)
        count = used_names.get(stem, 0) + 1
        used_names[stem] = count
        filename = f"map_{stem}.png" if count == 1 else f"map_{stem}_{count}.png"
        try:
            _save_png(image, out_dir / filename)
        except OSError as exc:
            print(
                f"[nav_v2_map_gallery] error: cannot write {out_dir / filename}: {exc}",
                file=sys.stderr,
            )
            return (files_written, image_rows, _EXIT_OUTPUT_FAILED)
        files_written += 1
        numbers = _area_numbers(area)
        image_rows.append(
            (
                filename,
                f"area `{display_id}` ({_area_kind(area)}, visits {_area_visits(area)}, "
                f"{numbers['walkable']} walkable / {numbers['cells_total']} cells)",
            )
        )

    if files_written == 0:
        print(
            "[nav_v2_map_gallery] error: no renderable areas in the world model",
            file=sys.stderr,
        )
        return (files_written, image_rows, _EXIT_CORRUPT_WORLD)

    status = _build_status(world)
    try:
        world_image = viz.render_world_image(world, width=width, height=height)
        status_image = viz.render_status_panel(status, width=width, height=height)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        print(
            f"[nav_v2_map_gallery] error: failed to render world/status panels: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return (files_written, image_rows, _EXIT_RENDER_FAILED)
    try:
        _save_png(world_image, out_dir / "world_graph.png")
        _save_png(status_image, out_dir / "status.png")
    except OSError as exc:
        print(f"[nav_v2_map_gallery] error: cannot write world/status PNGs: {exc}", file=sys.stderr)
        return (files_written, image_rows, _EXIT_OUTPUT_FAILED)
    files_written += 2
    image_rows.append(
        (
            "world_graph.png",
            f"area graph ({len(areas)} areas, {status['world']['connections']} connections)",
        )
    )
    image_rows.append(("status.png", "aggregated navigation status panel"))

    try:
        index_text = _index_markdown(world_dir, status, world, image_rows)
        (out_dir / "index.md").write_text(index_text, encoding="utf-8")
    except OSError as exc:
        print(f"[nav_v2_map_gallery] error: cannot write index.md: {exc}", file=sys.stderr)
        return (files_written, image_rows, _EXIT_OUTPUT_FAILED)
    files_written += 1
    print(
        f"[nav_v2_map_gallery] world={_relative(world_dir)} areas={len(areas)} "
        f"connections={status['world']['connections']} files={files_written} out={_relative(out_dir)}"
    )
    for row in image_rows:
        print(f"[nav_v2_map_gallery] wrote {row[0]}")
    return (files_written, image_rows, _EXIT_OK)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code (never raises for bad input)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    world_dir = _resolve_path(args.world_dir, DEFAULT_WORLD_DIR)
    out_dir = _resolve_path(args.out, DEFAULT_OUT_DIR)
    cell_px = max(1, int(args.cell_px))
    max_cells = max(2, int(args.max_cells))
    width = max(80, int(args.width))
    height = max(48, int(args.height))

    error = _validate_world_dir(world_dir)
    if error is not None:
        print(f"[nav_v2_map_gallery] error: {error}", file=sys.stderr)
        return _EXIT_MISSING_INPUT

    world, exit_code, error = _load_world(world_dir)
    if world is None:
        print(f"[nav_v2_map_gallery] error: {error}", file=sys.stderr)
        return exit_code

    files_written, _image_rows, render_code = _render_gallery(
        world,
        world_dir,
        out_dir,
        cell_px=cell_px,
        max_cells=max_cells,
        width=width,
        height=height,
    )
    if render_code != _EXIT_OK:
        return render_code
    print(f"[nav_v2_map_gallery] done: {files_written} files written")
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
