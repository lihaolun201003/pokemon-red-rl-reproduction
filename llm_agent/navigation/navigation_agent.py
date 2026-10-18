"""High-level, persistent navigation facade for Navigation v2 (nav-control).

``NavigationAgent`` is the stable API the DeepSeek tool layer / runtime uses:

* ``attach()`` — sync the pose with a freshly loaded emulator state,
* ``explore_current_area(...)`` — frontier exploration with honest budgets,
* ``navigate_to(target)`` — goal-directed navigation (cell / exit / landmark /
  other-area / return-to-entry),
* ``status()`` / ``local_map()`` / ``world_summary()`` — structured reporting
  for the tools, prompts and the Tk viewer,
* ``pause()`` / ``resume()`` — stop starting navigation actions (dialog/battle
  handling) without losing the map,
* ``save()`` — persist the world model under ``world_root``,
* ``observe_external_outcome(outcome)`` — exactly-once map sync for the legacy
  raw press/walk tools (``ActionOutcome``), so the model and the controller can
  never double-count the same button press.

Hard rules: no DeepSeek API calls, no RAM reads, no infinite loops, no new
dependencies.  This module imports only :mod:`llm_agent.navigation.types` at
import time; the real localizer/world/planner/explorer modules are imported
lazily in the factory methods (and may be replaced through dependency-injection
keyword arguments for tests).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from llm_agent.navigation.controller import (
    ControllerConfig,
    NavigationController,
)
from llm_agent.navigation.types import (
    DIRECTIONS,
    CellState,
    CoverageStats,
    ExploreReport,
    ExploreStatus,
    MotionResult,
    NavEvent,
    NavEventKind,
    NavReport,
    PlayerPose,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llm_agent.contracts import ActionOutcome, Frame
    from llm_agent.navigation.explorer import Explorer
    from llm_agent.navigation.localization import Localizer
    from llm_agent.navigation.pathfinder import Pathfinder
    from llm_agent.navigation.world_model import WorldModel

__all__ = ["NavigationAgent", "NavigationAgentConfig"]

#: contracts.ScreenEvent value -> NavEventKind (string values, no runtime import)
_SCREEN_EVENT_KIND: dict[str, NavEventKind] = {
    "dialog": NavEventKind.DIALOG_EVENT,
    "menu": NavEventKind.MENU_EVENT,
    "battle": NavEventKind.BATTLE_EVENT,
    "map_change": NavEventKind.STORY_EVENT,
    "transition": NavEventKind.TRANSITION_EVENT,
    "unknown": NavEventKind.UNKNOWN_EVENT,
}

#: event kinds that pause autonomous navigation (the brain must react first)
_PAUSING_EVENT_KINDS: frozenset[NavEventKind] = frozenset(
    {
        NavEventKind.DIALOG_EVENT,
        NavEventKind.BATTLE_EVENT,
        NavEventKind.MENU_EVENT,
        NavEventKind.STORY_EVENT,
        NavEventKind.UNKNOWN_EVENT,
    }
)

#: additional tuning/behaviour knobs of the facade (all optional)
@dataclass
class NavigationAgentConfig:
    """Configuration of the high-level facade.

    ``controller`` holds the verified-loop budgets/thresholds; ``autosave``
    persists the world model after every mutating call when ``world_root`` is
    set.  The remaining fields are reserved for the runtime integration.
    """

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    autosave: bool = True
    #: maximum number of recent external outcomes remembered for deduplication
    external_dedupe_cache: int = 128


class NavigationAgent:
    """Persistent facade over the verified navigation controller.

    Parameters
    ----------
    emulator:
        Emulator wrapper (``observe/press/wait/frame_count``).
    world_root:
        Directory for world-model persistence (``navigation/world.json`` and
        ``navigation/areas/*.json``).  ``None`` keeps everything in memory.
    config:
        :class:`NavigationAgentConfig` (or any object exposing a compatible
        ``controller`` / ``autosave``); ``None`` uses defaults.
    on_frame, on_step, on_status:
        Optional observation callbacks for the viewer/tool layer.
    world, localizer, pathfinder, explorer, controller:
        Optional dependency injection (tests, or a runtime that already owns the
        objects).  When omitted the real navigation modules are imported lazily.
    """

    def __init__(
        self,
        *,
        emulator: Any,
        world_root: str | Path | None = None,
        config: NavigationAgentConfig | Any | None = None,
        on_frame: Callable[["Frame"], None] | None = None,
        on_step: Callable[[Any], None] | None = None,
        on_status: Callable[[dict[str, Any]], None] | None = None,
        # --- additive dependency-injection hooks (tests / advanced wiring) --
        world: "WorldModel | None" = None,
        localizer: "Localizer | None" = None,
        pathfinder: "Pathfinder | None" = None,
        explorer: "Explorer | None" = None,
        controller: NavigationController | None = None,
    ) -> None:
        self._emulator = emulator
        self._config = self._normalize_config(config)
        self._world_root: Path | None = (
            Path(world_root) if world_root is not None else None
        )
        self._injected_world = world is not None
        self._on_frame = on_frame
        self._on_step = on_step
        self._on_status = on_status
        self._paused: bool = False
        self._last_event: NavEvent | None = None
        self._last_report: ExploreReport | NavReport | None = None
        self._last_attach: dict[str, Any] | None = None
        self._last_save_error: str | None = None
        self._world_load_error: str | None = None
        self._external_processed: set[str] = set()
        self._external_order: list[str] = []
        # -- injected or lazily created components ------------------------- #
        self._world = world if world is not None else self._make_world()
        self._localizer = localizer if localizer is not None else self._make_localizer()
        self._pathfinder = (
            pathfinder if pathfinder is not None else self._make_pathfinder()
        )
        self._explorer = explorer if explorer is not None else self._make_explorer()
        self._controller = (
            controller
            if controller is not None
            else NavigationController(
                emulator=emulator,
                world=self._world,
                localizer=self._localizer,
                pathfinder=self._pathfinder,
                explorer=self._explorer,
                config=self._config.controller,
                on_frame=on_frame,
                on_step=on_step,
                on_status=on_status,
            )
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_config(config: NavigationAgentConfig | Any | None) -> NavigationAgentConfig:
        """Accept ``None``, our config, or any object with compatible fields."""
        if config is None:
            return NavigationAgentConfig()
        if isinstance(config, NavigationAgentConfig):
            return config
        normalized = NavigationAgentConfig()
        controller = getattr(config, "controller", None)
        if isinstance(controller, ControllerConfig):
            normalized.controller = controller
        autosave = getattr(config, "autosave", None)
        if isinstance(autosave, bool):
            normalized.autosave = autosave
        dedupe = getattr(config, "external_dedupe_cache", None)
        if isinstance(dedupe, int) and dedupe > 0:
            normalized.external_dedupe_cache = dedupe
        return normalized

    def _make_world(self) -> "WorldModel":
        """Build the world model (lazy import; tolerant load)."""
        from llm_agent.navigation.world_model import WorldModel

        try:
            world = WorldModel(root=self._world_root)
        except Exception as exc:  # noqa: BLE001 - corrupt/unreadable persistence
            self._world_load_error = f"WorldModel init failed: {type(exc).__name__}: {exc}"
            self._world_root = None
            return WorldModel()
        if self._world_root is not None:
            load = getattr(world, "load", None)
            if callable(load):
                try:
                    load()
                except Exception as exc:  # noqa: BLE001 - tolerant; start fresh
                    self._world_load_error = (
                        f"world load failed: {type(exc).__name__}: {exc}"
                    )
        return world

    @staticmethod
    def _make_localizer() -> "Localizer":
        from llm_agent.navigation.localization import Localizer

        return Localizer()

    @staticmethod
    def _make_pathfinder() -> "Pathfinder":
        from llm_agent.navigation.pathfinder import Pathfinder

        return Pathfinder()

    def _make_explorer(self) -> "Explorer":
        from llm_agent.navigation.explorer import Explorer

        try:
            return Explorer(pathfinder=self._pathfinder)
        except TypeError:  # pragma: no cover - older explorer signature
            return Explorer()

    # ------------------------------------------------------------------ #
    # Public component access
    # ------------------------------------------------------------------ #
    @property
    def controller(self) -> NavigationController:
        return self._controller

    @property
    def world(self) -> "WorldModel":
        return self._world

    @property
    def localizer(self) -> "Localizer":
        return self._localizer

    @property
    def pathfinder(self) -> "Pathfinder":
        return self._pathfinder

    @property
    def explorer(self) -> "Explorer":
        return self._explorer

    @property
    def config(self) -> NavigationAgentConfig:
        return self._config

    @property
    def last_event(self) -> NavEvent | None:
        return self._last_event

    @property
    def last_report(self) -> ExploreReport | NavReport | None:
        return self._last_report

    @property
    def last_attach(self) -> dict[str, Any] | None:
        return self._last_attach

    @property
    def last_save_error(self) -> str | None:
        return self._last_save_error

    @property
    def world_load_error(self) -> str | None:
        return self._world_load_error

    @property
    def paused(self) -> bool:
        """True when autonomous navigation must not start a new action."""
        return bool(self._paused or self._controller.paused)

    def pause(self) -> None:
        """Pause autonomous navigation (raw emulator tools still work)."""
        self._paused = True
        self._controller.pause()

    def resume(self) -> None:
        """Resume autonomous navigation; relocalizes first if the pose was lost."""
        self._paused = False
        self._controller.resume()
        pose = self._localizer.pose
        if not pose.valid:
            try:
                self._controller.relocate()
            except Exception:  # noqa: BLE001 - resume must not raise
                pass
        self._autosave()

    # ------------------------------------------------------------------ #
    # Attach / persistence
    # ------------------------------------------------------------------ #
    def attach(self) -> None:
        """Synchronise the pose with a freshly loaded emulator state.

        Tries to relocalize the current screenshot in the known world; on a
        strong match the area/pose are restored.  Otherwise a **new** area with
        pose ``(0, 0)`` is created (areas are never merged on weak evidence; the
        origin of a new area is the first cell where it was entered).
        """
        view = self._controller.observe()
        area_id = str(getattr(self._world, "current_area", "") or "")
        matches = self._match_view(view)
        match = self._controller.accept_match(matches, expected_area=area_id or None)
        if match is not None:
            area_id = str(match.area_id)
            pose = (int(match.pose[0]), int(match.pose[1]))
            self._world.set_current(area_id, pose)
            self._reset_localizer(area_id, pose, confidence=max(0.5, float(match.score)))
            self._controller.record_observation(view)
            self._record_visit(area_id, view)
            event = NavEvent(
                kind=NavEventKind.NONE,
                screen_event=str(getattr(view, "screen_event", "none") or "none"),
                message=f"attached to known area {area_id!r} at {pose}",
                frame_index=int(getattr(view, "frame_index", 0) or 0),
                position=pose,
                area_id=area_id,
            )
            self._last_attach = {
                "matched": True,
                "area_id": area_id,
                "pose": list(pose),
                "score": float(match.score),
                "new_area": False,
            }
        else:
            try:
                area = self._world.create_area(
                    frame_index=int(getattr(view, "frame_index", 0) or 0)
                )
                area_id = str(getattr(area, "area_id", "") or "")
            except Exception as exc:  # noqa: BLE001 - keep the facade structured
                self._last_attach = {
                    "matched": False,
                    "area_id": area_id,
                    "pose": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "new_area": False,
                }
                return
            self._world.set_current(area_id, (0, 0))
            self._reset_localizer(area_id, (0, 0), confidence=1.0)
            self._controller.record_observation(view)
            self._record_visit(area_id, view)
            self._record_entry(area_id, (0, 0))
            event = NavEvent(
                kind=NavEventKind.NONE,
                screen_event=str(getattr(view, "screen_event", "none") or "none"),
                message=f"attached at a new area origin ({area_id!r})",
                frame_index=int(getattr(view, "frame_index", 0) or 0),
                position=(0, 0),
                area_id=area_id,
            )
            self._last_attach = {
                "matched": False,
                "area_id": area_id,
                "pose": [0, 0],
                "score": 0.0,
                "new_area": True,
            }
        if event.kind in _PAUSING_EVENT_KINDS:
            self._last_event = event
        self._autosave()
        self._notify_status()

    def save(self) -> None:
        """Persist the world model under ``world_root`` (raises on failure).

        With no ``world_root`` and an internally created world this is a no-op;
        an injected world is always asked to save (useful for tests/runtimes that
        own the persistence root themselves).
        """
        if self._world_root is None and not self._injected_world:
            self._last_save_error = None
            return
        self._world.save()
        self._last_save_error = None

    def _autosave(self) -> None:
        """Best-effort persistence after a mutating call (never raises)."""
        if not self._config.autosave:
            return
        if self._world_root is None and not self._injected_world:
            return
        try:
            self._world.save()
            self._last_save_error = None
        except Exception as exc:  # noqa: BLE001 - report, never break navigation
            self._last_save_error = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ #
    # Exploration / navigation
    # ------------------------------------------------------------------ #
    def explore_current_area(
        self, *, max_steps: int = 300, max_seconds: float = 600.0
    ) -> ExploreReport:
        """Explore the current area with the frontier policy (bounded)."""
        if self.paused:
            report = ExploreReport(
                status=ExploreStatus.PAUSED,
                area_id=self._current_area_id(),
                reason="navigation is paused; call resume() first",
                coverage=self._coverage_stats(),
                pose=self._localizer.pose,
                started_at=time.time(),
            )
            self._last_report = report
            return report
        report = self._controller.explore_current_area(
            max_steps=max_steps, max_seconds=max_seconds
        )
        self._last_report = report
        self._absorb_report(report)
        self._autosave()
        self._notify_status()
        return report

    def navigate_to(self, target: dict[str, Any], *, max_steps: int = 200) -> NavReport:
        """Navigate to a structured target; events/blocked paths are reported."""
        if self.paused:
            report = NavReport(
                status=ExploreStatus.PAUSED,
                area_id=self._current_area_id(),
                reason="navigation is paused; call resume() first",
                target=dict(target) if isinstance(target, dict) else {},
                coverage=self._coverage_stats(),
                pose=self._localizer.pose,
                started_at=time.time(),
            )
            self._last_report = report
            return report
        report = self._controller.navigate_to(target, max_steps=max_steps)
        self._last_report = report
        self._absorb_report(report)
        self._autosave()
        self._notify_status()
        return report

    def explore_and_report(
        self,
        *,
        max_steps: int = 300,
        max_seconds: float = 600.0,
        include_map: bool = True,
    ) -> dict[str, Any]:
        """Convenience wrapper for the tool layer: report + status + map dicts."""
        report = self.explore_current_area(max_steps=max_steps, max_seconds=max_seconds)
        data: dict[str, Any] = {
            "report": report.to_dict(),
            "status": self.status(),
            "world": self.world_summary(),
        }
        if include_map:
            data["map"] = self.local_map()
        return data

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """Structured navigation state (stable field list consumed by tools/viewer).

        Keys: ``area_id``, ``area_kind``, ``pose``, ``cells_total``, ``walkable``,
        ``blocked``, ``unknown_frontier``, ``coverage_ratio``, ``areas_total``,
        ``last_event``, ``paused``, ``world``.
        """
        localizer = self._localizer
        pose: PlayerPose = localizer.pose
        area_id = self._current_area_id()
        area = self._area(area_id)
        grid = getattr(area, "grid", None) if area is not None else None
        coverage = self._coverage_for(grid)
        if coverage is not None:
            cells_total = int(coverage.cells_total)
            walkable = int(coverage.walkable)
            blocked = int(coverage.blocked) + int(coverage.temp_blocked)
            unknown_frontier = int(coverage.unknown_frontier)
            coverage_ratio = float(coverage.coverage_ratio)
        else:
            counts = self._count_cell_states(grid)
            cells_total = counts["cells_total"]
            walkable = counts["walkable"]
            blocked = counts["blocked"]
            unknown_frontier = counts["unknown_frontier"]
            denominator = walkable + unknown_frontier
            coverage_ratio = walkable / denominator if denominator else 0.0
        areas = getattr(self._world, "areas", {}) or {}
        world_rows: list[list[Any]] = []
        for aid in sorted(areas):
            record = areas[aid]
            area_grid = getattr(record, "grid", None)
            row_coverage = self._coverage_for(area_grid)
            if row_coverage is not None:
                row_walkable = int(row_coverage.walkable)
            else:
                row_walkable = self._count_cell_states(area_grid)["walkable"]
            world_rows.append(
                [
                    str(aid),
                    str(getattr(record, "kind", "unknown") or "unknown"),
                    int(getattr(record, "visits", 0) or 0),
                    int(row_walkable),
                ]
            )
        return {
            "area_id": area_id,
            "area_kind": str(getattr(area, "kind", "unknown") or "unknown"),
            "pose": pose.to_dict(),
            "cells_total": int(cells_total),
            "walkable": int(walkable),
            "blocked": int(blocked),
            "unknown_frontier": int(unknown_frontier),
            "coverage_ratio": round(float(coverage_ratio), 4),
            "areas_total": int(len(areas)),
            "last_event": None if self._last_event is None else self._last_event.to_dict(),
            "paused": bool(self.paused),
            "world": {
                "areas": world_rows,
                "connections": len(getattr(self._world, "connections", []) or []),
            },
        }

    def local_map(self, *, radius: int = 12) -> dict[str, Any]:
        """Local map window around the player (ASCII + cell list + coverage)."""
        area_id = self._current_area_id()
        area = self._area(area_id)
        grid = getattr(area, "grid", None) if area is not None else None
        pose = self._localizer.pose
        ascii_text = ""
        if grid is not None:
            render = getattr(grid, "render_ascii", None)
            if callable(render):
                try:
                    ascii_text = str(
                        render(center=tuple(pose.cell), radius=int(radius))
                    )
                except TypeError:  # pragma: no cover - older render signature
                    ascii_text = str(render())
                except Exception:  # noqa: BLE001 - rendering must not break tools
                    ascii_text = ""
        cells: list[dict[str, Any]] = []
        if grid is not None:
            center = tuple(pose.cell)
            radius_i = max(0, int(radius))
            for key in sorted(getattr(grid, "cells", {}) or {}):
                x, y = int(key[0]), int(key[1])
                if abs(x - center[0]) > radius_i or abs(y - center[1]) > radius_i:
                    continue
                cell = (getattr(grid, "cells", {}) or {})[key]
                cells.append(
                    {
                        "x": x,
                        "y": y,
                        "state": str(getattr(getattr(cell, "state", ""), "value", getattr(cell, "state", ""))),
                        "terrain": str(getattr(cell, "terrain", "") or ""),
                        "visits": int(getattr(cell, "visit_count", 0) or 0),
                        "confidence": round(float(getattr(cell, "confidence", 0.0) or 0.0), 3),
                    }
                )
        coverage = self._coverage_for(grid)
        return {
            "area_id": area_id,
            "pose": pose.to_dict(),
            "ascii": ascii_text,
            "cells": cells,
            "coverage": None if coverage is None else coverage.to_dict(),
        }

    def world_summary(self) -> dict[str, Any]:
        """World model summary for prompts/viewer (areas, connections, current)."""
        summary: dict[str, Any] = {}
        get_summary = getattr(self._world, "summary", None)
        if callable(get_summary):
            try:
                value = get_summary()
                if isinstance(value, dict):
                    summary = dict(value)
            except Exception:  # noqa: BLE001 - fall back to our own summary
                summary = {}
        if not summary:
            areas = getattr(self._world, "areas", {}) or {}
            summary = {
                "areas": [
                    {
                        "area_id": str(aid),
                        "kind": str(getattr(record, "kind", "unknown") or "unknown"),
                        "visits": int(getattr(record, "visits", 0) or 0),
                        "walkable": self._count_cell_states(
                            getattr(record, "grid", None)
                        )["walkable"],
                    }
                    for aid, record in sorted(areas.items())
                ],
                "connections": [
                    {
                        "from_area": str(getattr(connection, "from_area", "")),
                        "from_cell": list(getattr(connection, "from_cell", (0, 0))),
                        "direction": str(getattr(connection, "direction", "")),
                        "to_area": str(getattr(connection, "to_area", "")),
                        "to_pose": (
                            None
                            if getattr(connection, "to_pose", None) is None
                            else list(getattr(connection, "to_pose"))
                        ),
                        "kind": str(getattr(connection, "kind", "")),
                    }
                    for connection in (getattr(self._world, "connections", []) or [])
                ],
            }
        summary.setdefault("current_area", self._current_area_id())
        summary.setdefault("current_pose", list(self._localizer.pose.cell))
        summary.setdefault("areas_total", len(getattr(self._world, "areas", {}) or {}))
        return summary

    # ------------------------------------------------------------------ #
    # External (legacy raw tool) action bookkeeping
    # ------------------------------------------------------------------ #
    def observe_external_outcome(self, outcome: "ActionOutcome | None") -> None:
        """Synchronise the map with a raw tool ``ActionOutcome`` (exactly once).

        Each directional :class:`~llm_agent.contracts.ActionStep` is verified
        through the same exactly-once path as :meth:`NavigationController.move_once`
        (without pressing anything again).  Outcomes are deduplicated by a stable
        id, and steps are additionally deduplicated by the monotonic frame index,
        so replaying the same outcome can never update the map twice.
        """
        if outcome is None:
            return
        outcome_id = self._outcome_id(outcome)
        if outcome_id in self._external_processed:
            return
        self._remember_external(outcome_id)
        steps = list(getattr(outcome, "steps", []) or [])
        for step in steps:
            button = str(getattr(step, "button", "") or "").strip().lower()
            frame = getattr(step, "frame", None)
            if frame is None:
                continue
            if button in DIRECTIONS:
                estimate = self._controller.register_external_step(button, frame)
                if estimate is None:
                    continue
                if estimate.result is MotionResult.TRANSITION:
                    self._controller.handle_transition()
                    self._last_event = NavEvent(
                        kind=NavEventKind.TRANSITION_EVENT,
                        screen_event="transition",
                        message="area transition caused by an external action",
                        frame_index=int(getattr(frame, "frame_index", 0) or 0),
                        position=tuple(self._localizer.pose.cell)
                        if self._localizer.pose.valid
                        else None,
                        area_id=self._current_area_id(),
                    )
                elif estimate.result is MotionResult.INTERRUPTED:
                    event = self._controller.last_event or self._controller.classify_event(
                        estimate=estimate
                    )
                    self._last_event = event
                    if self._config.controller.pause_on_event and event.kind in _PAUSING_EVENT_KINDS:
                        self.pause()
            else:
                # Non-directional raw press (A/B/start/select): keep the baseline
                # anchored; it must never update the robot coordinates.
                self._controller.sync_observation(frame)
        # Surface a game event that the raw batch reported but that the per-step
        # motion estimates did not classify (e.g. dialog advanced by A).
        kind = self._outcome_event_kind(outcome)
        if kind is not None:
            self._last_event = NavEvent(
                kind=kind,
                screen_event=self._outcome_screen_event(outcome) or "none",
                message="non-navigation event reported by an external action batch",
                frame_index=int(getattr(outcome, "end_frame", 0) or 0),
                position=tuple(self._localizer.pose.cell)
                if self._localizer.pose.valid
                else None,
                area_id=self._current_area_id(),
            )
            if (
                self._config.controller.pause_on_event
                and kind in _PAUSING_EVENT_KINDS
                and not self.paused
            ):
                self.pause()
        self._autosave()
        self._notify_status()

    @staticmethod
    def _outcome_id(outcome: "ActionOutcome") -> str:
        """Stable identity of one ``ActionOutcome`` for exactly-once processing."""
        steps = list(getattr(outcome, "steps", []) or [])
        last_frame = getattr(steps[-1], "frame", None) if steps else None
        return "|".join(
            [
                str(getattr(outcome, "start_frame", "")),
                str(getattr(outcome, "end_frame", "")),
                ",".join(str(getattr(step, "button", "")) for step in steps),
                str(len(steps)),
                str(getattr(last_frame, "frame_index", "")),
            ]
        )

    def _remember_external(self, outcome_id: str) -> None:
        self._external_processed.add(outcome_id)
        self._external_order.append(outcome_id)
        while len(self._external_order) > max(1, int(self._config.external_dedupe_cache)):
            oldest = self._external_order.pop(0)
            self._external_processed.discard(oldest)

    @staticmethod
    def _outcome_screen_event(outcome: "ActionOutcome") -> str | None:
        events = list(getattr(outcome, "events", []) or [])
        for event in reversed(events):
            value = str(getattr(event, "value", event) or "none")
            if value and value != "none":
                return value
        steps = list(getattr(outcome, "steps", []) or [])
        for step in reversed(steps):
            value = str(getattr(getattr(step, "event", None), "value", "none") or "none")
            if value and value != "none":
                return value
        return None

    @classmethod
    def _outcome_event_kind(cls, outcome: "ActionOutcome") -> NavEventKind | None:
        screen_event = cls._outcome_screen_event(outcome)
        if screen_event is None:
            return None
        kind = _SCREEN_EVENT_KIND.get(screen_event)
        if kind is None or kind is NavEventKind.NONE:
            return None
        return kind

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _current_area_id(self) -> str:
        area_id = str(getattr(self._world, "current_area", "") or "")
        if area_id:
            return area_id
        return self._controller.current_area_id()

    def _area(self, area_id: str | None = None) -> Any | None:
        try:
            return self._world.area(area_id or None)
        except Exception:  # noqa: BLE001 - empty/partial fake world
            return None

    def _match_view(self, view: Any) -> list[Any]:
        try:
            return list(self._world.match_view(view) or [])
        except Exception:  # noqa: BLE001 - fake world without relocalization
            return []

    def _reset_localizer(
        self, area_id: str, pose: tuple[int, int], *, confidence: float
    ) -> None:
        try:
            self._localizer.reset(area_id=area_id, cell=pose)
        except TypeError:  # pragma: no cover - older localizer signature
            self._localizer.reset(area_id=area_id, cell=pose, facing=None)
        self._localizer.set_pose(
            int(pose[0]), int(pose[1]), confidence=float(confidence)
        )

    def _record_visit(self, area_id: str, view: Any) -> None:
        record_visit = getattr(self._world, "record_visit", None)
        if not callable(record_visit):
            return
        try:
            record_visit(area_id, frame_index=int(getattr(view, "frame_index", 0) or 0))
        except TypeError:  # pragma: no cover
            record_visit(area_id)
        except Exception:  # noqa: BLE001 - optional bookkeeping
            pass

    def _record_entry(self, area_id: str, cell: tuple[int, int]) -> None:
        area = self._area(area_id)
        entries = getattr(area, "entry_cells", None)
        if isinstance(entries, list) and tuple(cell) not in [tuple(e) for e in entries]:
            entries.append((int(cell[0]), int(cell[1])))

    def _coverage_for(self, grid: Any | None) -> CoverageStats | None:
        if grid is None:
            return None
        coverage = getattr(grid, "coverage", None)
        if not callable(coverage):
            return None
        try:
            stats = coverage()
        except Exception:  # noqa: BLE001
            return None
        return stats if isinstance(stats, CoverageStats) else None

    def _coverage_stats(self) -> CoverageStats | None:
        return self._coverage_for(self._controller.grid())

    @staticmethod
    def _count_cell_states(grid: Any | None) -> dict[str, int]:
        counts = {
            "cells_total": 0,
            "walkable": 0,
            "blocked": 0,
            "unknown_frontier": 0,
        }
        if grid is None:
            return counts
        cells = getattr(grid, "cells", {}) or {}
        counts["cells_total"] = len(cells)
        for cell in cells.values():
            state = str(getattr(getattr(cell, "state", ""), "value", getattr(cell, "state", "")))
            if state == CellState.WALKABLE.value:
                counts["walkable"] += 1
            elif state in (CellState.BLOCKED.value, CellState.TEMP_BLOCKED.value):
                counts["blocked"] += 1
        # Unknown frontier: non-walkable cells adjacent to a walkable cell.
        walkable_cells = {
            (int(key[0]), int(key[1]))
            for key, cell in cells.items()
            if str(getattr(getattr(cell, "state", ""), "value", getattr(cell, "state", "")))
            == CellState.WALKABLE.value
        }
        frontier: set[tuple[int, int]] = set()
        for x, y in walkable_cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (x + dx, y + dy)
                if neighbour in walkable_cells or neighbour in frontier:
                    continue
                other = cells.get(neighbour)
                state = str(
                    getattr(getattr(other, "state", ""), "value", getattr(other, "state", ""))
                ) if other is not None else ""
                if state in ("", CellState.UNKNOWN.value, CellState.OBSERVED.value):
                    frontier.add(neighbour)
        counts["unknown_frontier"] = len(frontier)
        return counts

    def _absorb_report(self, report: ExploreReport | NavReport) -> None:
        """Update facade state from a finished controller report."""
        event = getattr(report, "event", None)
        if event is not None and event.kind is not NavEventKind.NONE:
            self._last_event = event
            if (
                self._config.controller.pause_on_event
                and event.kind in _PAUSING_EVENT_KINDS
            ):
                self.pause()
        if getattr(report, "status", None) is ExploreStatus.PAUSED:
            self._paused = True
        if getattr(report, "status", None) is ExploreStatus.LOCALIZATION_LOST:
            self._paused = False

    def _notify_status(self) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(self.status())
        except Exception:  # noqa: BLE001 - presentation only
            pass

    # ------------------------------------------------------------------ #
    # Introspection helpers used by tests / runtime logs
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        """Compact dict combining ``status`` and controller counters."""
        data = self.status()
        data["controller"] = self._controller.stats()
        data["last_event"] = None if self._last_event is None else self._last_event.to_dict()
        data["paused"] = bool(self.paused)
        return data
