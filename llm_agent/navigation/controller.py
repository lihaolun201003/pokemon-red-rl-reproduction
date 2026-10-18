"""Verified low-level navigation controller for Navigation v2 (owner: nav-control).

The controller is the **only** place in the navigation subsystem that presses a
direction button.  Every directional press is verified visually through the
injected :class:`~llm_agent.navigation.localization.Localizer` and produces a
:class:`~llm_agent.navigation.types.MotionEstimate`; the map/world is then
updated **exactly once** according to that estimate:

======================  =====================================================
``MotionResult``        exactly-once map update
======================  =====================================================
``MOVED``               ``grid.confirm_move`` (pose already advanced by the
                        localizer) + ``world.set_current``
``BLOCKED``             ``grid.confirm_blocked(temporary=False)``
``TURNED_ONLY``         ``grid.confirm_turn`` + ``confirm_blocked`` evidence
``UNKNOWN``             *no* map/coordinate change, counted; after
                        ``max_consecutive_unknown`` the controller relocalizes
                        (or reports ``localization_lost``)
``INTERRUPTED``         *no* map update, structured :class:`NavEvent`
                        (dialog / battle / menu / story), navigation pauses
``TRANSITION``          *no* grid update; the caller runs ``handle_transition``
                        (bounded wait for a stable screen + relocalization)
======================  =====================================================

Design rules (see ``NAV_CONTRACT.md`` §6):

* No RAM reads, no DeepSeek API calls, no external map data — screenshots and
  the agent's own recorded observations only.
* The controller never presses a button while a dialog/battle/menu is on screen:
  such frames are reported structurally and navigation stops.
* Every loop is bounded (``max_steps``, ``max_seconds``, ``max_no_progress``,
  consecutive UNKNOWN/BLOCKED limits); no infinite loops are possible.
* The emulator is never left mid-keypress: presses are one button at a time and
  the emulator wrapper released the button before returning the frame.
* Grid/world/localizer/planner/explorer modules are injected (duck-typed), so
  this module only imports :mod:`llm_agent.navigation.types` at import time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from llm_agent.navigation.types import (
    DIRECTIONS,
    AreaMatch,
    CellState,
    CoverageStats,
    ExploreReport,
    ExploreStatus,
    ExploreTarget,
    FrameContext,
    FrameView,
    MotionEstimate,
    MotionResult,
    NavEvent,
    NavEventKind,
    NavReport,
    NavStep,
    PathPlan,
    PlayerPose,
    step_delta,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from llm_agent.contracts import Frame
    from llm_agent.navigation.explorer import Explorer
    from llm_agent.navigation.grid_map import GridMap
    from llm_agent.navigation.localization import Localizer
    from llm_agent.navigation.pathfinder import Pathfinder
    from llm_agent.navigation.world_model import WorldModel

__all__ = [
    "ControllerConfig",
    "ControllerResult",
    "NavigationController",
]

#: FrameContext -> NavEventKind for non-navigation screens.
_CONTEXT_EVENT: dict[FrameContext, NavEventKind] = {
    FrameContext.DIALOG: NavEventKind.DIALOG_EVENT,
    FrameContext.MENU: NavEventKind.MENU_EVENT,
    FrameContext.BATTLE: NavEventKind.BATTLE_EVENT,
    FrameContext.TRANSITION: NavEventKind.TRANSITION_EVENT,
}

#: contracts.ScreenEvent value -> NavEventKind (string values to avoid importing
#: ``llm_agent.contracts`` at module import time).
_SCREEN_EVENT_KIND: dict[str, NavEventKind] = {
    "dialog": NavEventKind.DIALOG_EVENT,
    "menu": NavEventKind.MENU_EVENT,
    "battle": NavEventKind.BATTLE_EVENT,
    "map_change": NavEventKind.STORY_EVENT,
    "transition": NavEventKind.TRANSITION_EVENT,
    "unknown": NavEventKind.UNKNOWN_EVENT,
}

#: NavEventKind that mean "the overworld is occluded by a game event".
_BLOCKING_EVENT_KINDS: frozenset[NavEventKind] = frozenset(
    {
        NavEventKind.DIALOG_EVENT,
        NavEventKind.BATTLE_EVENT,
        NavEventKind.MENU_EVENT,
        NavEventKind.STORY_EVENT,
        NavEventKind.UNKNOWN_EVENT,
    }
)


# --------------------------------------------------------------------------- #
# Configuration / result dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class ControllerConfig:
    """Budgets and thresholds of the verified navigation loop.

    The first six fields are the frozen contract; the remaining fields are
    additive tuning knobs with defaults that keep the documented behaviour.
    """

    #: tiles requested per verified step (a real held press may produce more; the
    #: localizer reports the actual count and the grid records that count).
    max_tiles_per_step: int = 1
    #: consecutive UNKNOWN results before the controller relocalizes / gives up.
    max_consecutive_unknown: int = 2
    #: consecutive BLOCKED/TURNED_ONLY results before stopping with ``blocked``.
    max_consecutive_blocked: int = 4
    #: extra frames to wait (``emulator.wait``) after a press before observing.
    settle_frames: int = 0
    #: stop and pause navigation when a dialog/battle/menu/story event appears.
    pause_on_event: bool = True
    #: soft per-press timeout (seconds); slow presses are counted, never killed.
    step_timeout_s: float = 30.0
    # -- additive tuning knobs -------------------------------------------- #
    #: relocalization acceptance: minimum tile-match score (contract: 0.55).
    relocalize_min_score: float = 0.55
    #: relocalization acceptance: margin over the runner-up (contract: 0.08).
    relocalize_min_margin: float = 0.08
    #: expected-area agreement lowers the required margin to this value.
    relocalize_expected_margin: float = 0.03
    #: maximum number of area hops one ``navigate_to`` call may perform.
    max_area_hops: int = 3
    #: hard cap on primitive actions for one ``navigate_to`` call.
    max_navigate_steps: int = 400
    #: how many frames a transition may take before giving up on a stable screen.
    transition_settle_frames: int = 120
    #: maximum fraction of changed pixels between two 8-frame transition captures
    #: for the screen to count as frozen (animated tiles/NPCs are <2 %, fades >10 %).
    transition_max_diff: float = 0.02
    #: TURNED_ONLY is recorded as *temporary* blocked evidence (a fresh-direction
    #: press may only turn the player before stepping; permanent avoidance is
    #: reserved for verified BLOCKED presses).
    turned_only_temporary: bool = True


@dataclass
class ControllerResult:
    """Result of one bounded controller primitive (``follow_path`` / probes)."""

    status: str = "ok"  # ok|blocked|event|transition|localization_lost|step_limit
    steps: int = 0
    moves: int = 0
    blocked: int = 0
    unknown: int = 0
    replans: int = 0
    event: NavEvent | None = None
    steps_log: list[NavStep] = field(default_factory=list)
    #: additive: short human-readable reason for the terminal status
    reason: str = ""


@dataclass
class _ConnectionStub:
    """Duck-typed fallback for ``world_model.WorldConnection``.

    ``world_model.py`` is imported lazily so this module stays importable while
    the nav-map owner is still writing it; tests may inject a fake world that
    records connection objects without importing the real dataclass.
    """

    from_area: str
    from_cell: tuple[int, int]
    direction: str = "warp"
    to_area: str = ""
    to_pose: tuple[int, int] | None = None
    kind: str = "warp"
    confidence: float = 0.5
    observations: int = 1


def _make_connection(**kwargs: Any) -> Any:
    """Build a real ``WorldConnection`` when available, else the local stub."""
    try:  # lazy import: module owned by nav-map, may not exist in early tests
        from llm_agent.navigation.world_model import WorldConnection

        return WorldConnection(**kwargs)
    except Exception:  # noqa: BLE001 - fall back to the duck-typed stub
        return _ConnectionStub(**kwargs)


def _state_value(value: Any) -> str:
    """String value of an :class:`~llm_agent.navigation.types.CellState`-like."""
    raw = getattr(value, "value", value)
    return str(raw)


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #


class NavigationController:
    """Verified navigation loop: the only code that presses direction buttons.

    Parameters
    ----------
    emulator:
        The real (or synthetic) emulator wrapper; only ``observe()``,
        ``press(direction)``, ``wait(frames)`` and ``frame_count`` are used.
    world:
        Multi-area world model (``world_model.WorldModel`` API).
    localizer:
        Visual motion estimator (``localization.Localizer`` API).
    pathfinder, explorer:
        Planner / exploration policy (``pathfinder`` / ``explorer`` APIs).
    config:
        :class:`ControllerConfig`.
    on_frame, on_step, on_status:
        Optional observation callbacks (the viewer / tool layer); exceptions in
        callbacks are swallowed so presentation can never break navigation.
    sleep:
        Optional pacing hook used only when the emulator has no ``wait()``.
    """

    #: consecutive "no change" 8-frame diffs required to call a screen frozen
    TRANSITION_STABLE_DIFFS: int = 2
    #: emulated frames per stability capture in :meth:`handle_transition`
    TRANSITION_CAPTURE_FRAMES: int = 8
    #: extra frames waited after the provisional settle before relocalizing
    TRANSITION_CONFIRM_FRAMES: int = 8
    #: default maximum changed-pixel fraction between captures (config-overridable)
    TRANSITION_MAX_DIFF: float = 0.02

    def __init__(
        self,
        *,
        emulator: Any,
        world: "WorldModel",
        localizer: "Localizer",
        pathfinder: "Pathfinder",
        explorer: "Explorer",
        config: ControllerConfig | None = None,
        on_frame: Callable[["Frame"], None] | None = None,
        on_step: Callable[[NavStep], None] | None = None,
        on_status: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._emulator = emulator
        self._world = world
        self._localizer = localizer
        self._pathfinder = pathfinder
        self._explorer = explorer
        self.config: ControllerConfig = config or ControllerConfig()
        self._on_frame = on_frame
        self._on_step = on_step
        self._on_status = on_status
        self._sleep = sleep

        # -- runtime state -------------------------------------------------- #
        self._last_view: FrameView | None = None
        self._last_estimate: MotionEstimate | None = None
        self._last_event: NavEvent | None = None
        self._step_index: int = 0
        self._step_log: list[NavStep] = []
        #: last verified route (world cells, start..goal) for the viewer map panel
        self.last_path: list[tuple[int, int]] = []
        #: last navigation target cell of the active run (``None`` when idle)
        self.last_target: tuple[int, int] | None = None
        self._paused: bool = False
        self._localization_lost: bool = False
        self._consecutive_unknown: int = 0
        self._consecutive_blocked: int = 0
        self._replanned_edges: set[tuple[tuple[int, int], str]] = set()
        self._run_path: list[tuple[int, int]] = []
        # -- counters (monotonic; reports use deltas) ----------------------- #
        self._steps: int = 0
        self._moves: int = 0
        self._blocked: int = 0
        self._turned: int = 0
        self._unknown: int = 0
        self._replans: int = 0
        self._backtracked: int = 0
        self._observations: int = 0
        self._slow_steps: int = 0

    # ------------------------------------------------------------------ #
    # Public read-only state
    # ------------------------------------------------------------------ #
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
    def last_view(self) -> FrameView | None:
        return self._last_view

    @property
    def last_estimate(self) -> MotionEstimate | None:
        return self._last_estimate

    @property
    def last_event(self) -> NavEvent | None:
        return self._last_event

    @property
    def step_log(self) -> list[NavStep]:
        return self._step_log

    @property
    def paused(self) -> bool:
        return bool(self._paused)

    def pause(self) -> None:
        """Stop starting new navigation actions (``resume`` re-enables them)."""
        self._paused = True

    def resume(self) -> None:
        """Re-enable navigation actions after :meth:`pause`."""
        self._paused = False

    def current_area_id(self) -> str:
        """Id of the area the world model says we are in (best effort)."""
        area_id = str(getattr(self._world, "current_area", "") or "")
        if area_id:
            return area_id
        return str(getattr(self._localizer, "area_id", "") or "")

    def current_pose(self) -> PlayerPose:
        """Current local pose as tracked by the localizer."""
        return self._localizer.pose

    def grid(self) -> "GridMap | None":
        """Grid map of the current area (``None`` when the world is empty)."""
        return getattr(self._area(), "grid", None)

    def stats(self) -> dict[str, int]:
        """Monotonic controller action counters (for logging/tests)."""
        return {
            "steps": self._steps,
            "moves": self._moves,
            "blocked": self._blocked,
            "turned": self._turned,
            "unknown": self._unknown,
            "replans": self._replans,
            "backtracked": self._backtracked,
            "observations": self._observations,
            "slow_steps": self._slow_steps,
        }

    def status(self) -> dict[str, Any]:
        """Structured controller state (embedded in the agent status report)."""
        pose = self._localizer.pose
        coverage = self._coverage()
        area = self._area()
        return {
            "area_id": self.current_area_id(),
            "area_kind": str(getattr(area, "kind", "unknown") or "unknown"),
            "pose": pose.to_dict(),
            "grid": None if coverage is None else coverage.to_dict(),
            "steps": self._steps,
            "moves": self._moves,
            "blocked": self._blocked,
            "turned": self._turned,
            "unknown": self._unknown,
            "replans": self._replans,
            "consecutive_unknown": self._consecutive_unknown,
            "consecutive_blocked": self._consecutive_blocked,
            "localization_lost": bool(self._localization_lost),
            "paused": bool(self._paused),
            "last_event": None if self._last_event is None else self._last_event.to_dict(),
            "last_path": [list(cell) for cell in self.last_path],
            "last_target": None if self.last_target is None else list(self.last_target),
        }

    # ------------------------------------------------------------------ #
    # Route bookkeeping exposed to the viewer
    # ------------------------------------------------------------------ #
    def _set_route(
        self,
        cells: Iterable[tuple[int, int]] | None = None,
        target: tuple[int, int] | None = None,
    ) -> None:
        """Publish the route/target currently being executed (viewer hook)."""
        if cells is not None:
            self.last_path = [(int(cell[0]), int(cell[1])) for cell in cells]
        self.last_target = (
            None if target is None else (int(target[0]), int(target[1]))
        )

    def _clear_route(self) -> None:
        """Reset the viewer route/target when navigation is idle."""
        self.last_path = []
        self.last_target = None

    @staticmethod
    def _plan_cells(plan: PathPlan | Any) -> list[tuple[int, int]]:
        cells = list(getattr(plan, "cells", []) or [])
        return [(int(cell[0]), int(cell[1])) for cell in cells]

    # ------------------------------------------------------------------ #
    # Emulator / observation primitives
    # ------------------------------------------------------------------ #
    def observe(self) -> FrameView:
        """Capture the current framebuffer and build a :class:`FrameView`."""
        frame = self._capture(self._emulator.observe())
        return self._observe_frame(frame)

    def _capture(self, frame: Any) -> Any:
        """Pass a captured frame to the optional viewer callback (guarded)."""
        if frame is not None and self._on_frame is not None:
            try:
                self._on_frame(frame)
            except Exception:  # noqa: BLE001 - presentation must never break nav
                pass
        return frame

    def _observe_frame(self, frame: Any) -> FrameView:
        view = self._localizer.observe(frame)
        self._last_view = view
        return view

    def _wait(self, frames: int) -> Any:
        """Advance ``frames`` without input, returning the new frame."""
        frames = max(1, int(frames))
        wait = getattr(self._emulator, "wait", None)
        if callable(wait):
            return wait(frames)
        if self._sleep is not None:  # fallback for minimal fake emulators
            self._sleep(frames / 60.0)
            return self._emulator.observe()
        raise RuntimeError("emulator has neither wait() nor a sleep() fallback")

    def _press(self, direction: str) -> Any:
        """Press exactly one direction button and return the captured frame."""
        started = time.monotonic()
        frame = self._emulator.press(direction)
        elapsed = time.monotonic() - started
        if elapsed > float(self.config.step_timeout_s):
            self._slow_steps += 1
        frame = self._capture(frame)
        settle = int(self.config.settle_frames)
        if settle > 0:
            frame = self._capture(self._wait(settle))
        return frame

    def _frame_count(self) -> int:
        try:
            return int(getattr(self._emulator, "frame_count", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0

    # ------------------------------------------------------------------ #
    # Verified one-step movement (the exactly-once heart)
    # ------------------------------------------------------------------ #
    def move_once(self, direction: str, *, intent: str = "probe") -> MotionEstimate:
        """Press one direction, verify the motion visually, update the map once.

        This is the only method that presses a direction button.  The returned
        :class:`MotionEstimate` is exactly the result the map was updated with
        (an inconsistent estimate is downgraded to ``UNKNOWN`` and changes
        nothing).
        """
        direction = str(direction).strip().lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        previous = self._last_view
        if previous is None:
            previous = self.observe()
        pre_press_event = self.screen_blocks_navigation(previous)
        if pre_press_event is not None:
            # Never press blindly through a dialog/battle/menu: report the event
            # instead of pressing (and do not count a verified step).
            self._handle_event(pre_press_event)
            estimate = MotionEstimate(
                result=MotionResult.INTERRUPTED,
                direction=direction,
                context=getattr(previous, "context", FrameContext.UNKNOWN),
                confidence=0.0,
                evidence={
                    "event": pre_press_event.screen_event,
                    "no_press": True,
                    "reason": "screen was already occluded before the press",
                },
                frame_index=int(getattr(previous, "frame_index", 0) or 0),
            )
            self._last_estimate = estimate
            self._notify_status()
            return estimate
        frame = self._press(direction)
        current = self._observe_frame(frame)
        return self._verify(previous, current, direction, intent=intent)

    def screen_blocks_navigation(self, view: FrameView | None = None) -> NavEvent | None:
        """Return a blocking :class:`NavEvent` when the screen is not overworld."""
        view = view if view is not None else self._last_view
        if view is None:
            return None
        context = getattr(view, "context", FrameContext.OVERWORLD)
        if context in (FrameContext.DIALOG, FrameContext.MENU, FrameContext.BATTLE):
            return self.classify_event(view=view)
        screen_event = str(getattr(view, "screen_event", "") or "")
        if screen_event in ("dialog", "menu", "battle"):
            return self.classify_event(view=view)
        return None

    def register_external_step(self, direction: str, frame: Any) -> MotionEstimate | None:
        """Verify a direction press that was performed *outside* the controller.

        Used by :meth:`NavigationAgent.observe_external_outcome` for the legacy
        raw press/walk tools.  It never presses a button itself; it applies the
        same exactly-once map update as :meth:`move_once`.  Steps are deduplicated
        by the monotonic ``frame_index`` so a replayed outcome can never update
        the map twice.
        """
        direction = str(direction).strip().lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        if frame is None:
            return None
        frame_index = int(getattr(frame, "frame_index", -1))
        if self._last_estimate is not None:
            last_external = int(getattr(self._last_estimate, "frame_index", -1))
            if frame_index >= 0 and last_external >= 0 and frame_index <= last_external:
                return None  # already processed this frame (dedupe)
        previous = self._last_view
        if previous is None:
            previous = self._observe_frame(frame)
            return MotionEstimate(
                result=MotionResult.UNKNOWN,
                direction=direction,
                evidence={"reason": "no baseline view for external step"},
                frame_index=frame_index,
            )
        current = self._observe_frame(frame)
        return self._verify(previous, current, direction, intent="external")

    def sync_observation(self, frame: Any) -> FrameView | None:
        """Refresh the baseline view for an external non-directional press.

        No map update happens here; it only keeps the next motion comparison
        anchored on the most recent screenshot.
        """
        if frame is None:
            return None
        return self._observe_frame(frame)

    def _verify(
        self,
        previous: FrameView,
        current: FrameView,
        direction: str,
        *,
        intent: str,
    ) -> MotionEstimate:
        """Run the localizer and apply the exactly-once map update."""
        pose_before = tuple(self._localizer.pose.cell)
        requested = max(1, int(self.config.max_tiles_per_step))
        estimate = self._localizer.update(
            previous, current, direction, requested_tiles=requested
        )
        return self._apply_motion(estimate, direction, pose_before, current, intent=intent)

    def _accepted_tiles(self, direction: str, delta: tuple[int, int]) -> int:
        """Number of tiles a MOVED estimate verifiably represents (0 = reject)."""
        limit = max(1, int(self.config.max_tiles_per_step))
        for tiles in range(1, limit + 1):
            if (int(delta[0]), int(delta[1])) == step_delta(direction, tiles):
                return tiles
        return 0

    def _apply_motion(
        self,
        estimate: MotionEstimate,
        direction: str,
        from_cell: tuple[int, int],
        current_view: FrameView,
        *,
        intent: str,
    ) -> MotionEstimate:
        """Exactly-once grid/world bookkeeping for one verified action."""
        result = estimate.result
        grid = self.grid()
        frame_index = int(getattr(current_view, "frame_index", 0) or 0)
        area_id = self.current_area_id()
        accepted = estimate

        if result is MotionResult.MOVED:
            delta = (int(estimate.delta[0]), int(estimate.delta[1]))
            tiles = self._accepted_tiles(direction, delta)
            if tiles <= 0:
                # Never trust a movement that does not match the pressed direction.
                if delta != (0, 0):
                    try:
                        self._localizer.set_pose(
                            from_cell[0], from_cell[1], facing=direction, confidence=0.2
                        )
                    except Exception:  # noqa: BLE001 - best effort pose restore
                        pass
                self._consecutive_unknown += 1
                self._unknown += 1
                accepted = MotionEstimate(
                    result=MotionResult.UNKNOWN,
                    direction=direction,
                    requested_tiles=max(1, int(estimate.requested_tiles)),
                    confidence=min(0.3, float(estimate.confidence)),
                    evidence={
                        "raw_result": "moved",
                        "inconsistent_delta": [delta[0], delta[1]],
                        "reason": "movement delta does not match requested direction",
                    },
                    frame_index=frame_index,
                    context=estimate.context,
                )
                result = MotionResult.UNKNOWN
            else:
                target = (
                    from_cell[0] + delta[0],
                    from_cell[1] + delta[1],
                )
                # Backtrack bookkeeping must look at the map *before* confirm_move.
                previous_cell = None
                if grid is not None:
                    try:
                        previous_cell = grid.cell(*target)
                    except Exception:  # noqa: BLE001 - fake/partial grid
                        previous_cell = None
                if previous_cell is not None and _state_value(
                    getattr(previous_cell, "state", "")
                ) == CellState.WALKABLE.value:
                    self._backtracked += 1
                if grid is not None:
                    grid.confirm_move(
                        from_cell, direction, tiles=tiles, frame_index=frame_index
                    )
                pose_after = tuple(self._localizer.pose.cell)
                if area_id:
                    try:
                        self._world.set_current(area_id, pose_after)
                    except Exception:  # noqa: BLE001 - fake/partial world
                        pass
                self._moves += 1
                self._consecutive_unknown = 0
                self._consecutive_blocked = 0
                self._run_path.append(pose_after)
                self._record_view(current_view)

        elif result is MotionResult.BLOCKED:
            if grid is not None:
                grid.confirm_blocked(
                    from_cell, direction, temporary=False, frame_index=frame_index
                )
            self._blocked += 1
            self._consecutive_blocked += 1
            self._consecutive_unknown = 0
            self._record_view(current_view)

        elif result is MotionResult.TURNED_ONLY:
            if grid is not None:
                grid.confirm_turn(from_cell, direction)
                # Blocked evidence per contract; a fresh-direction turn may only
                # be the turn half of a turn-then-step, so it defaults to
                # *temporary* evidence (verify: turned_only_temporary).
                grid.confirm_blocked(
                    from_cell,
                    direction,
                    temporary=bool(self.config.turned_only_temporary),
                    frame_index=frame_index,
                )
            self._turned += 1
            self._consecutive_blocked += 1
            self._consecutive_unknown = 0
            self._record_view(current_view)

        elif result is MotionResult.UNKNOWN:
            # Contract: no map/coordinate change.  Only count it.
            self._unknown += 1
            self._consecutive_unknown += 1

        elif result is MotionResult.INTERRUPTED:
            self._consecutive_unknown = 0
            event = self.classify_event(estimate=estimate, view=current_view)
            self._handle_event(event)

        elif result is MotionResult.TRANSITION:
            self._consecutive_unknown = 0
            self._consecutive_blocked = 0
            event = NavEvent(
                kind=NavEventKind.TRANSITION_EVENT,
                screen_event=str(getattr(current_view, "screen_event", "none") or "none"),
                message="screen transition detected; relocalization required",
                frame_index=frame_index,
                position=tuple(self._localizer.pose.cell),
                area_id=area_id,
            )
            self._last_event = event

        self._last_estimate = accepted
        self._log_step(direction, accepted, from_cell, intent=intent)
        return accepted

    # ------------------------------------------------------------------ #
    # Map observation helpers
    # ------------------------------------------------------------------ #
    def record_observation(self, view: FrameView | None = None) -> dict[str, int]:
        """Record the visible tiles of ``view`` (default: last view) into the grid.

        This is a *pixel observation*, not a movement verification: it may add
        ``OBSERVED`` cells and tile hashes but never confirms a move.  The
        controller only calls it for frames whose pose is known (MOVED /
        BLOCKED / TURNED_ONLY / after relocalization), never for UNKNOWN,
        INTERRUPTED or TRANSITION frames.
        """
        view = view if view is not None else self._last_view
        grid = self.grid()
        if view is None or grid is None:
            return {}
        try:
            counters = grid.observe_view(
                view, self._localizer.pose, frame_index=int(view.frame_index)
            )
        except TypeError:  # pragma: no cover - very old/partial fakes
            counters = grid.observe_view(view, self._localizer.pose)
        self._observations += 1
        return dict(counters or {})

    def _record_view(self, view: FrameView | None) -> None:
        try:
            self.record_observation(view)
        except Exception:  # noqa: BLE001 - observation is best effort, never fatal
            return

    # ------------------------------------------------------------------ #
    # Step log / events / callbacks
    # ------------------------------------------------------------------ #
    def _log_step(
        self,
        direction: str,
        estimate: MotionEstimate,
        pose_before: tuple[int, int],
        *,
        intent: str,
    ) -> None:
        self._steps += 1
        self._step_index += 1
        step = NavStep(
            index=self._step_index,
            direction=direction,
            motion=estimate,
            pose_before=(int(pose_before[0]), int(pose_before[1])),
            pose_after=tuple(self._localizer.pose.cell),
            area_id=self.current_area_id(),
            intent=intent,
        )
        self._step_log.append(step)
        if len(self._step_log) > 4000:  # bounded memory for long runs
            del self._step_log[:2000]
        if self._on_step is not None:
            try:
                self._on_step(step)
            except Exception:  # noqa: BLE001 - callbacks are presentation only
                pass

    def classify_event(
        self,
        *,
        view: FrameView | None = None,
        estimate: MotionEstimate | None = None,
    ) -> NavEvent:
        """Classify a non-navigation screen into a structured :class:`NavEvent`."""
        view = view if view is not None else self._last_view
        context = None
        if estimate is not None:
            context = estimate.context
        if context is None and view is not None:
            context = getattr(view, "context", None)
        kind = _CONTEXT_EVENT.get(context, NavEventKind.NONE)  # type: ignore[arg-type]
        notes = dict(getattr(view, "notes", {}) or {}) if view is not None else {}
        if notes.get("story") or notes.get("story_event"):
            kind = NavEventKind.STORY_EVENT
        screen_event = ""
        if estimate is not None:
            screen_event = str((estimate.evidence or {}).get("event", "") or "")
        if view is not None and not screen_event:
            screen_event = str(getattr(view, "screen_event", "") or "")
        if kind is NavEventKind.NONE:
            kind = _SCREEN_EVENT_KIND.get(screen_event, NavEventKind.UNKNOWN_EVENT)
        message = {
            NavEventKind.DIALOG_EVENT: "dialog on screen; navigation paused",
            NavEventKind.BATTLE_EVENT: "battle on screen; navigation paused",
            NavEventKind.MENU_EVENT: "menu on screen; navigation paused",
            NavEventKind.STORY_EVENT: "story/map event on screen; navigation paused",
            NavEventKind.TRANSITION_EVENT: "screen transition detected",
            NavEventKind.UNKNOWN_EVENT: "overworld interrupted by an unknown screen",
        }.get(kind, "non-navigation event")
        frame_index = 0
        if view is not None:
            frame_index = int(getattr(view, "frame_index", 0) or 0)
        elif estimate is not None:
            frame_index = int(getattr(estimate, "frame_index", 0) or 0)
        pose = self.current_pose()
        return NavEvent(
            kind=kind,
            screen_event=screen_event or "none",
            message=message,
            frame_index=frame_index,
            position=tuple(pose.cell) if pose.valid else None,
            area_id=self.current_area_id(),
        )

    def _handle_event(self, event: NavEvent) -> None:
        self._last_event = event
        if self.config.pause_on_event and event.kind in _BLOCKING_EVENT_KINDS:
            self._paused = True

    def _notify_status(self) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(self.status())
        except Exception:  # noqa: BLE001 - presentation only
            pass

    # ------------------------------------------------------------------ #
    # Relocalization
    # ------------------------------------------------------------------ #
    def _accept_match(
        self,
        matches: Sequence[AreaMatch] | None,
        *,
        expected_area: str | None = None,
        min_score: float | None = None,
        min_margin: float | None = None,
    ) -> AreaMatch | None:
        """Apply the contract relocalization acceptance rule.

        Accept only a match with ``score >= min_score`` **and**
        ``margin >= min_margin`` over the runner-up, or a slightly weaker match
        whose area equals the just-recorded expected area (connection/pose
        evidence).  Ambiguous matches are rejected so areas are never merged on
        weak evidence.
        """
        if not matches:
            return None
        min_score = (
            float(self.config.relocalize_min_score) if min_score is None else float(min_score)
        )
        min_margin = (
            float(self.config.relocalize_min_margin) if min_margin is None else float(min_margin)
        )
        best = matches[0]
        score = float(getattr(best, "score", 0.0))
        second_score = float(getattr(best, "second_score", 0.0))
        if len(matches) > 1:
            second_score = max(second_score, float(getattr(matches[1], "score", 0.0)))
        # Effective margin over the runner-up.  A single returned candidate with no
        # runner-up information legitimately has margin == score.
        effective_margin = score - second_score
        declared_margin = float(getattr(best, "margin", 0.0) or 0.0)
        if declared_margin > 0.0:
            effective_margin = min(effective_margin, declared_margin)
        if score < min_score or not str(getattr(best, "area_id", "") or ""):
            return None
        if effective_margin >= min_margin:
            return best
        expected_area = expected_area if expected_area is not None else self.current_area_id()
        if (
            expected_area
            and str(getattr(best, "area_id", "")) == expected_area
            and score >= min_score
            and effective_margin >= float(self.config.relocalize_expected_margin)
        ):
            return best
        return None

    def accept_match(
        self,
        matches: Sequence[AreaMatch] | None,
        *,
        expected_area: str | None = None,
        min_score: float | None = None,
        min_margin: float | None = None,
    ) -> AreaMatch | None:
        """Public wrapper around the relocalization acceptance rule."""
        return self._accept_match(
            matches,
            expected_area=expected_area,
            min_score=min_score,
            min_margin=min_margin,
        )

    def relocate(self, *, expected_area: str | None = None) -> AreaMatch | None:
        """Relocalize the current view in the world model and restore the pose.

        Returns the accepted :class:`AreaMatch` (area/pose already restored) or
        ``None`` when the evidence is weak/ambiguous — in that case nothing is
        changed (areas are never merged on weak evidence).
        """
        view = self.observe()
        try:
            matches = list(self._world.match_view(view) or [])
        except Exception:  # noqa: BLE001 - empty world / fake without match data
            matches = []
        match = self._accept_match(matches, expected_area=expected_area)
        if match is None:
            self._last_view = view
            return None
        area_id = str(match.area_id)
        pose = (int(match.pose[0]), int(match.pose[1]))
        try:
            self._world.set_current(area_id, pose)
        except Exception:  # noqa: BLE001 - fake/partial world
            pass
        try:
            self._localizer.reset(area_id=area_id, cell=pose)
        except TypeError:  # pragma: no cover - older localizer signature
            self._localizer.reset(area_id=area_id, cell=pose, facing=None)
        self._localizer.set_pose(
            pose[0], pose[1], confidence=max(0.5, float(match.score))
        )
        self._localization_lost = False
        self._consecutive_unknown = 0
        self._consecutive_blocked = 0
        self._record_view(view)
        try:
            self._world.record_visit(area_id, frame_index=int(view.frame_index))
        except Exception:  # noqa: BLE001 - optional bookkeeping
            pass
        self._notify_status()
        return match

    def _maybe_relocalize(self) -> bool:
        """Relocalize after the UNKNOWN streak threshold; returns success."""
        match = self.relocate()
        if match is None:
            self._localization_lost = True
            try:
                invalidate = getattr(self._localizer, "invalidate", None)
                if callable(invalidate):
                    invalidate("relocalization failed after repeated UNKNOWN")
            except Exception:  # noqa: BLE001
                pass
            return False
        return True

    # ------------------------------------------------------------------ #
    # Transitions / area changes
    # ------------------------------------------------------------------ #
    def handle_transition(self, *, max_wait_frames: int = 120) -> dict[str, Any]:
        """Wait for a *frozen* non-black screen, then relocalize.

        A transition screen must have settled completely before we interpret it.
        "Frozen" is decided on **pixels**, not on a signature: two consecutive
        8-frame captures must differ by at most ``transition_max_diff`` (2 %)
        changed pixels *and* their best background shift must be ``(0, 0)`` so a
        slow fade/scroll never counts.  Animated outdoor tiles and walking NPCs
        change well under 2 % per 8 frames, a fade changes far more.  After two
        consecutive frozen diffs one extra ``wait(8)`` confirmation must still
        show a frozen screen.  ``FrameView.stable`` is deliberately ignored.

        If the screen never freezes within the budget, nothing is created or
        merged: the status is ``black_timeout`` / ``unstable_timeout`` and the
        pose is invalidated.

        Strong match: switch to that area, restore the pose and record the
        connection.  No match: create a **new** area with pose ``(0, 0)`` and
        record the connection as pending (``to_pose=None``).  Areas are never
        merged on weak evidence.

        Returns a dict with ``status`` (``matched`` / ``new_area`` /
        ``black_timeout`` / ``unstable_timeout``), ``handled``, ``matched``,
        ``new_area``, ``switched``, ``area_id``, ``pose``, ``frames``,
        ``stable``, ``black``, ``match`` (dict|None), ``connection``
        (dict|None), ``event`` and the previous area/pose.
        """
        previous_area = self.current_area_id()
        previous_pose = tuple(self._localizer.pose.cell)
        # The transition usually happens when the *pressed step lands on a warp
        # tile*, so the actual exit cell is one tile from the last known pose.
        source_cell = previous_pose
        last_motion = self._last_estimate
        if last_motion is not None and last_motion.result is MotionResult.TRANSITION:
            direction = str(getattr(last_motion, "direction", "") or "")
            if direction in DIRECTIONS:
                delta = step_delta(direction)
                source_cell = (
                    previous_pose[0] + delta[0],
                    previous_pose[1] + delta[1],
                )
        wait_budget = max(1, int(max_wait_frames))
        chunk = self.TRANSITION_CAPTURE_FRAMES
        chunks = max(1, wait_budget // chunk)
        max_diff = float(
            getattr(self.config, "transition_max_diff", self.TRANSITION_MAX_DIFF)
            or self.TRANSITION_MAX_DIFF
        )
        frames_waited = 0
        view: FrameView | None = None
        previous_view: FrameView | None = None
        frozen = 0
        non_black_seen = False
        for _ in range(chunks):
            frame = self._capture(self._wait(chunk))
            frames_waited += chunk
            view = self._observe_frame(frame)
            if bool(getattr(view, "is_black", False)):
                previous_view = None
                frozen = 0
                continue
            non_black_seen = True
            if previous_view is not None and self._transition_frame_unchanged(
                previous_view, view, max_diff
            ):
                frozen += 1
            else:
                frozen = 0
            previous_view = view
            if frozen >= self.TRANSITION_STABLE_DIFFS:
                break
        stable = previous_view is not None and frozen >= self.TRANSITION_STABLE_DIFFS
        if stable:
            # Confirm the frozen screen is still frozen a moment later.
            confirm_frame = self._capture(self._wait(self.TRANSITION_CONFIRM_FRAMES))
            frames_waited += self.TRANSITION_CONFIRM_FRAMES
            confirm_view = self._observe_frame(confirm_frame)
            kept_frozen = self._transition_frame_unchanged(
                previous_view, confirm_view, max_diff
            )
            view = confirm_view
            if bool(getattr(confirm_view, "is_black", False)) or not kept_frozen:
                stable = False
        if not stable:
            # NEVER create/merge an area from a mid-transition (fade) frame.
            last_black = bool(getattr(view, "is_black", False)) if view is not None else True
            status = "black_timeout" if (not non_black_seen or last_black) else "unstable_timeout"
            self._localizer.pose.valid = False
            self._localizer.pose.confidence = 0.0
            self._localization_lost = True
            message = (
                "screen stayed black after a transition; not creating an area"
                if status == "black_timeout"
                else "transition screen never froze long enough; not creating an area"
            )
            event = NavEvent(
                kind=NavEventKind.TRANSITION_EVENT,
                screen_event="transition",
                message=message,
                frame_index=int(getattr(view, "frame_index", 0) or 0) if view else 0,
                position=None,
                area_id=previous_area,
            )
            self._localizer.last_motion = MotionEstimate(
                result=MotionResult.UNKNOWN,
                direction="",
                confidence=0.0,
                evidence={"reason": message},
                frame_index=event.frame_index,
            )
            result = {
                "status": status,
                "handled": False,
                "matched": False,
                "new_area": False,
                "switched": False,
                "area_id": previous_area,
                "pose": None,
                "prev_area_id": previous_area,
                "prev_pose": list(previous_pose),
                "source_cell": list(source_cell),
                "frames": frames_waited,
                "stable": False,
                "black": status == "black_timeout",
                "match": None,
                "connection": None,
                "event": event.to_dict(),
            }
            self._notify_status()
            return result

        try:
            matches = list(self._world.match_view(view) or [])
        except Exception:  # noqa: BLE001 - fake world without match data
            matches = []
        match = self._accept_match(matches)
        connection: Any = None
        if match is not None:
            area_id = str(match.area_id)
            pose = (int(match.pose[0]), int(match.pose[1]))
            try:
                self._world.set_current(area_id, pose)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._localizer.reset(area_id=area_id, cell=pose)
            except TypeError:  # pragma: no cover
                self._localizer.reset(area_id=area_id, cell=pose, facing=None)
            self._localizer.set_pose(pose[0], pose[1], confidence=max(0.5, float(match.score)))
            self._localization_lost = False
            self._consecutive_unknown = 0
            self._consecutive_blocked = 0
            self._record_view(view)
            try:
                self._world.record_visit(area_id, frame_index=int(view.frame_index))
            except Exception:  # noqa: BLE001
                pass
            switched = bool(previous_area and previous_area != area_id)
            if switched:
                self._mark_transition_source(previous_area, source_cell)
                connection = self._record_connection(
                    previous_area, source_cell, area_id, pose, confidence=float(match.score)
                )
            self._record_entry(area_id, pose)
            result = {
                "status": "matched",
                "handled": True,
                "matched": True,
                "new_area": False,
                "switched": switched,
                "area_id": area_id,
                "pose": list(pose),
                "prev_area_id": previous_area,
                "prev_pose": list(previous_pose),
                "source_cell": list(source_cell),
                "frames": frames_waited,
                "stable": stable,
                "black": False,
                "match": match.to_dict(),
                "connection": self._connection_dict(connection),
            }
        else:
            # Weak evidence: create a fresh area (never merge into an existing one).
            try:
                area = self._world.create_area(frame_index=int(view.frame_index))
                area_id = str(getattr(area, "area_id", "") or "")
                if not area_id:  # pragma: no cover - defensive
                    raise RuntimeError("world.create_area returned no area_id")
            except Exception as exc:  # noqa: BLE001
                self._localization_lost = True
                self._notify_status()
                return {
                    "status": "error",
                    "handled": False,
                    "matched": False,
                    "new_area": False,
                    "switched": False,
                    "area_id": previous_area,
                    "pose": None,
                    "prev_area_id": previous_area,
                    "prev_pose": list(previous_pose),
                "source_cell": list(source_cell),
                    "frames": frames_waited,
                    "stable": stable,
                    "black": False,
                    "match": None,
                    "connection": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            try:
                self._world.set_current(area_id, (0, 0))
            except Exception:  # noqa: BLE001
                pass
            try:
                self._localizer.reset(area_id=area_id, cell=(0, 0))
            except TypeError:  # pragma: no cover
                self._localizer.reset(area_id=area_id, cell=(0, 0), facing=None)
            self._localizer.set_pose(0, 0, confidence=0.5)
            self._localization_lost = False
            self._consecutive_unknown = 0
            self._consecutive_blocked = 0
            self._record_view(view)
            try:
                self._world.record_visit(area_id, frame_index=int(view.frame_index))
            except Exception:  # noqa: BLE001
                pass
            self._record_entry(area_id, (0, 0))
            if previous_area:
                self._mark_transition_source(previous_area, source_cell)
                connection = self._record_connection(
                    previous_area, source_cell, area_id, None, confidence=0.3, pending=True
                )
            result = {
                "status": "new_area",
                "handled": True,
                "matched": False,
                "new_area": True,
                "switched": bool(previous_area != area_id),
                "area_id": area_id,
                "pose": [0, 0],
                "prev_area_id": previous_area,
                "prev_pose": list(previous_pose),
                "source_cell": list(source_cell),
                "frames": frames_waited,
                "stable": stable,
                "black": False,
                "match": None,
                "connection": self._connection_dict(connection),
            }
        self._notify_status()
        return result

    def _transition_frame_unchanged(
        self,
        previous: FrameView | None,
        current: FrameView | None,
        max_diff: float,
    ) -> bool:
        """True when two consecutive transition captures show a frozen screen.

        Uses raw luma pixels when available (``FrameView.notes["luma"]`` from the
        real localizer): the changed-pixel fraction must be ``<= max_diff`` and
        the best background shift must be ``(0, 0)``.  Falls back to signature
        equality for duck-typed/fake views without raw pixels.
        """
        if previous is None or current is None:
            return False
        if bool(getattr(current, "is_black", False)):
            return False
        previous_luma = (getattr(previous, "notes", None) or {}).get("luma")
        current_luma = (getattr(current, "notes", None) or {}).get("luma")
        if previous_luma is not None and current_luma is not None:
            try:  # lazy import: vision is owned by the Lead
                from llm_agent.navigation import vision

                changed_ratio = float(
                    vision.diff_mask(previous_luma, current_luma).mean()
                )
                if changed_ratio > float(max_diff):
                    return False
                shift = vision.best_shift(previous_luma, current_luma, max_tiles=2)
                if tuple(int(value) for value in shift.shift) != (0, 0):
                    return False
                return True
            except Exception:  # noqa: BLE001 - fall back to signatures
                pass
        previous_signature = str(getattr(previous, "signature", "") or "")
        current_signature = str(getattr(current, "signature", "") or "")
        return bool(previous_signature) and previous_signature == current_signature

    def _mark_transition_source(
        self, area_id: str, cell: tuple[int, int]
    ) -> None:
        """Mark the cell we transitioned from as an exit (verified evidence)."""
        try:
            grid = self._world.area(area_id).grid
        except Exception:  # noqa: BLE001
            return
        mark_exit = getattr(grid, "mark_exit", None)
        if not callable(mark_exit):
            return
        try:
            mark_exit(cell, confidence=0.8)
        except Exception:  # noqa: BLE001
            pass

    def _record_connection(
        self,
        from_area: str,
        from_cell: tuple[int, int],
        to_area: str,
        to_pose: tuple[int, int] | None,
        *,
        confidence: float,
        pending: bool = False,
    ) -> Any:
        """Record a world connection (real ``WorldConnection`` or stub)."""
        try:
            connection = _make_connection(
                from_area=from_area,
                from_cell=(int(from_cell[0]), int(from_cell[1])),
                direction="warp",
                to_area=to_area,
                to_pose=None if to_pose is None else (int(to_pose[0]), int(to_pose[1])),
                kind="warp",
                confidence=float(confidence),
                observations=1,
            )
        except Exception:  # noqa: BLE001 - stub always accepts the fields
            connection = _ConnectionStub(
                from_area=from_area,
                from_cell=(int(from_cell[0]), int(from_cell[1])),
                to_area=to_area,
                to_pose=None if to_pose is None else (int(to_pose[0]), int(to_pose[1])),
                confidence=float(confidence),
            )
        add_connection = getattr(self._world, "add_connection", None)
        if callable(add_connection):
            try:
                add_connection(connection)
            except Exception:  # noqa: BLE001 - duplicate/dedup policy is nav-map's
                pass
        return connection

    @staticmethod
    def _connection_dict(connection: Any) -> dict[str, Any] | None:
        if connection is None:
            return None
        return {
            "from_area": str(getattr(connection, "from_area", "")),
            "from_cell": list(getattr(connection, "from_cell", (0, 0))),
            "direction": str(getattr(connection, "direction", "warp")),
            "to_area": str(getattr(connection, "to_area", "")),
            "to_pose": (
                None
                if getattr(connection, "to_pose", None) is None
                else list(getattr(connection, "to_pose"))
            ),
            "kind": str(getattr(connection, "kind", "warp")),
            "confidence": float(getattr(connection, "confidence", 0.5)),
        }

    def _record_entry(self, area_id: str, cell: tuple[int, int]) -> None:
        """Remember ``cell`` as an entry cell of ``area_id`` (best effort)."""
        try:
            area = self._world.area(area_id)
        except Exception:  # noqa: BLE001
            return
        entries = getattr(area, "entry_cells", None)
        if isinstance(entries, list) and tuple(cell) not in [tuple(e) for e in entries]:
            entries.append((int(cell[0]), int(cell[1])))

    # ------------------------------------------------------------------ #
    # Bounded path following
    # ------------------------------------------------------------------ #
    def follow_path(
        self, plan: PathPlan | Any, *, max_steps: int = 100, intent: str = "navigate"
    ) -> ControllerResult:
        """Execute a planned route one verified step at a time.

        On ``BLOCKED`` the blocked edge is now known, so the route is replanned
        **once per blocked edge**; when no alternative exists the result is
        ``blocked`` (honest failure, no blind pressing).
        """
        result = ControllerResult(status="ok")
        snapshot = self._snapshot()
        goal = self._plan_goal(plan)
        directions = self._plan_directions(plan)
        if goal is None:
            result.status = "blocked"
            result.reason = "empty plan"
            self._fill_result(result, snapshot)
            return result
        # Align a stale plan with the actual pose before executing it.
        if self._plan_start(plan) != tuple(self._localizer.pose.cell):
            replanned = self._replan_to(goal)
            if replanned is None:
                result.status = "blocked"
                result.reason = "plan does not start at the current pose and no alternative path exists"
                self._fill_result(result, snapshot)
                return result
            plan = replanned
            directions = self._plan_directions(plan)
            self._replans += 1
        self._set_route(self._plan_cells(plan), goal)
        steps = 0
        index = 0
        divergence_replans = 0
        turn_retry_pending = False
        while True:
            if index >= len(directions):
                result.status = "ok"
                break
            if steps >= max_steps:
                result.status = "step_limit"
                result.reason = f"step budget {int(max_steps)} exhausted"
                break
            pose = self._localizer.pose
            if not pose.valid:
                if not self._maybe_relocalize():
                    result.status = "localization_lost"
                    result.reason = "pose invalid and relocalization failed"
                    break
                replanned = self._replan_to(goal)
                if replanned is None:
                    result.status = "blocked"
                    result.reason = "relocalized but goal is no longer reachable"
                    break
                plan = replanned
                directions = self._plan_directions(plan)
                self._set_route(self._plan_cells(plan), goal)
                index = 0
                self._replans += 1
                continue
            direction = directions[index]
            expected = (
                pose.cell[0] + step_delta(direction)[0],
                pose.cell[1] + step_delta(direction)[1],
            )
            cell_before = tuple(pose.cell)
            estimate = self.move_once(direction, intent=intent)
            steps += 1
            if estimate.result is MotionResult.INTERRUPTED:
                result.status = "event"
                result.event = self._last_event or self.classify_event(estimate=estimate)
                result.reason = "non-navigation event interrupted navigation"
                break
            if estimate.result is MotionResult.TRANSITION:
                result.status = "transition"
                result.event = self._last_event or NavEvent(
                    kind=NavEventKind.TRANSITION_EVENT,
                    message="transition detected while following a path",
                    frame_index=int(estimate.frame_index),
                    position=cell_before,
                    area_id=self.current_area_id(),
                )
                result.reason = "area transition while following a path"
                break
            if estimate.result is MotionResult.UNKNOWN:
                if self._consecutive_unknown >= max(
                    1, int(self.config.max_consecutive_unknown)
                ):
                    if not self._maybe_relocalize():
                        result.status = "localization_lost"
                        result.reason = "too many consecutive UNKNOWN motion estimates"
                        break
                    replanned = self._replan_to(goal)
                    if replanned is None:
                        result.status = "blocked"
                        result.reason = "relocalized but goal is no longer reachable"
                        break
                    plan = replanned
                    directions = self._plan_directions(plan)
                    self._set_route(self._plan_cells(plan), goal)
                    index = 0
                    self._replans += 1
                continue  # retry the same planned direction
            if estimate.result is MotionResult.TURNED_ONLY and not turn_retry_pending:
                # Gen-1 turn-then-step: the first press of a *new* direction may
                # only turn the player.  Retry the same direction once; only a
                # second non-move is treated as blocked evidence for replanning.
                turn_retry_pending = True
                continue
            turn_retry_pending = False
            if estimate.result is MotionResult.MOVED:
                actual = tuple(self._localizer.pose.cell)
                if actual == expected:
                    index += 1
                    continue
                # The verified move does not match the plan: replan from reality.
                divergence_replans += 1
                if divergence_replans > 2:
                    result.status = "blocked"
                    result.reason = "path diverged from the plan repeatedly"
                    break
                replanned = self._replan_to(goal)
                if replanned is None:
                    result.status = "blocked"
                    result.reason = "moved off-plan and no alternative path exists"
                    break
                plan = replanned
                directions = self._plan_directions(plan)
                self._set_route(self._plan_cells(plan), goal)
                index = 0
                self._replans += 1
                continue
            # BLOCKED / TURNED_ONLY: the blocked edge is now known; replan once.
            if self._consecutive_blocked >= max(
                1, int(self.config.max_consecutive_blocked)
            ):
                result.status = "blocked"
                result.reason = "too many consecutive blocked/turned-only results"
                break
            edge: tuple[tuple[int, int], str] = (cell_before, direction)
            replanned = self._replan_to(goal, edge=edge)
            if replanned is None:
                result.status = "blocked"
                result.reason = f"blocked by {direction!r} at {cell_before} and no alternative path exists"
                break
            plan = replanned
            directions = self._plan_directions(plan)
            self._set_route(self._plan_cells(plan), goal)
            index = 0
            self._replans += 1
        self._fill_result(result, snapshot)
        self._notify_status()
        return result

    # -- plan helpers --------------------------------------------------- #
    @staticmethod
    def _plan_goal(plan: PathPlan | Any) -> tuple[int, int] | None:
        cells = list(getattr(plan, "cells", []) or [])
        if not cells:
            return None
        goal = cells[-1]
        return (int(goal[0]), int(goal[1]))

    @staticmethod
    def _plan_directions(plan: PathPlan | Any) -> list[str]:
        return [str(d) for d in (getattr(plan, "directions", []) or []) if str(d) in DIRECTIONS]

    @staticmethod
    def _plan_start(plan: PathPlan | Any) -> tuple[int, int] | None:
        cells = list(getattr(plan, "cells", []) or [])
        if not cells:
            return None
        start = cells[0]
        return (int(start[0]), int(start[1]))

    def _plan_to(self, goal: tuple[int, int]) -> PathPlan | None:
        """Plan from the current pose to ``goal`` (``None`` when unreachable)."""
        grid = self.grid()
        if grid is None:
            return None
        pose = tuple(self._localizer.pose.cell)
        try:
            plan = self._pathfinder.plan(grid, pose, (int(goal[0]), int(goal[1])), allow_unknown=True)
        except TypeError:  # pragma: no cover - older planner signature
            plan = self._pathfinder.plan(grid, pose, (int(goal[0]), int(goal[1])))
        if plan is None or not bool(getattr(plan, "reachable", False)):
            return None
        return plan

    def _plan_to_nearest(self, goals: Iterable[tuple[int, int]]) -> PathPlan | None:
        """Plan to the closest of ``goals`` (``None`` when none is reachable)."""
        goal_list = [(int(cell[0]), int(cell[1])) for cell in goals]
        if not goal_list:
            return None
        grid = self.grid()
        if grid is None:
            return None
        pose = tuple(self._localizer.pose.cell)
        if pose in goal_list:
            return PathPlan(cells=[pose], directions=[], cost=0.0, reachable=True, reason="already at goal")
        plan_to_nearest = getattr(self._pathfinder, "plan_to_nearest", None)
        if callable(plan_to_nearest):
            try:
                plan = plan_to_nearest(grid, pose, goal_list, allow_unknown=True)
            except TypeError:  # pragma: no cover
                plan = plan_to_nearest(grid, pose, goal_list)
            if plan is not None and bool(getattr(plan, "reachable", False)):
                return plan
            return None
        best: PathPlan | None = None
        best_cost = float("inf")
        for goal in goal_list:
            plan = self._plan_to(goal)
            if plan is None:
                continue
            cost = float(getattr(plan, "cost", 0.0)) or float(len(getattr(plan, "cells", []) or []))
            if cost < best_cost:
                best, best_cost = plan, cost
        return best

    def _replan_to(
        self,
        goal: tuple[int, int],
        *,
        edge: tuple[tuple[int, int], str] | None = None,
    ) -> PathPlan | None:
        """Replan to ``goal``; each blocked edge may trigger only one replan."""
        pose = tuple(self._localizer.pose.cell)
        if pose == (int(goal[0]), int(goal[1])):
            return PathPlan(
                cells=[pose], directions=[], cost=0.0, reachable=True, reason="already at goal"
            )
        if edge is not None:
            edge_key = ((int(edge[0][0]), int(edge[0][1])), str(edge[1]))
            if edge_key in self._replanned_edges:
                return None
            plan = self._plan_to(goal)
            if plan is None:
                self._replanned_edges.add(edge_key)
                return None
            directions = self._plan_directions(plan)
            if directions and (pose, directions[0]) == edge_key:
                # A* still wants to use the known blocked edge: no real alternative.
                self._replanned_edges.add(edge_key)
                return None
            self._replanned_edges.add(edge_key)
            return plan
        return self._plan_to(goal)

    # ------------------------------------------------------------------ #
    # Area helpers
    # ------------------------------------------------------------------ #
    def _area(self, area_id: str | None = None) -> Any | None:
        """Fetch an area record **without side effects** (never creates one)."""
        world = self._world
        wanted = (
            str(area_id)
            if area_id is not None
            else str(getattr(world, "current_area", "") or "")
        )
        areas = getattr(world, "areas", None)
        if isinstance(areas, dict):
            record = areas.get(wanted) if wanted else None
            if record is not None:
                return record
            if wanted:
                return None
        record = getattr(world, "current_record", None)
        if record is not None and (
            not wanted or str(getattr(record, "area_id", "")) == wanted
        ):
            return record
        if not wanted:
            return None
        try:  # documented lookup; may create on demand for a real WorldModel
            return world.area(wanted)
        except Exception:  # noqa: BLE001 - partial/fake worlds
            return None

    def _grid_size(self, grid: Any | None) -> int:
        if grid is None:
            return 0
        size = getattr(grid, "size", None)
        if callable(size):
            try:
                return int(size())
            except Exception:  # noqa: BLE001
                pass
        return len(getattr(grid, "cells", {}) or {})

    @staticmethod
    def _grid_edges(grid: Any | None) -> int:
        if grid is None:
            return 0
        return len(getattr(grid, "edges", {}) or {})

    @staticmethod
    def _grid_exits(grid: Any | None) -> int:
        if grid is None:
            return 0
        count = 0
        for cell in (getattr(grid, "cells", {}) or {}).values():
            if _state_value(getattr(cell, "state", "")) == CellState.EXIT.value:
                count += 1
        return count

    def _coverage_for(self, grid: Any | None) -> CoverageStats | None:
        """Coverage stats of one grid (``None`` when unavailable)."""
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

    def _coverage(self) -> CoverageStats | None:
        return self._coverage_for(self.grid())

    def _coverage_for_area(self, area_id: str | None) -> CoverageStats | None:
        """Coverage of a specific area record (side-effect free)."""
        if not area_id:
            return None
        grid = getattr(self._area(area_id), "grid", None)
        return self._coverage_for(grid)

    def _report_context(
        self,
    ) -> tuple[str, CoverageStats | None, PlayerPose]:
        """Snapshot (area, coverage, pose) of the area being explored right now.

        Used before ``handle_transition`` switches the current area so the
        resulting report still describes the area that was actually explored.
        """
        pose = self._localizer.pose
        frozen_pose = PlayerPose(
            x=int(pose.x),
            y=int(pose.y),
            facing=pose.facing,
            confidence=float(pose.confidence),
            valid=bool(pose.valid),
        )
        return (self.current_area_id(), self._coverage(), frozen_pose)

    def _cells_with_state(self, grid: Any | None, state: CellState) -> list[tuple[int, int]]:
        if grid is None:
            return []
        wanted = state.value
        cells: list[tuple[int, int]] = []
        for key, cell in sorted((getattr(grid, "cells", {}) or {}).items()):
            if _state_value(getattr(cell, "state", "")) == wanted:
                cells.append((int(key[0]), int(key[1])))
        return cells

    def _landmark_cells(self, landmark: str) -> list[tuple[int, int]]:
        grid = self.grid()
        if grid is None:
            return []
        needle = str(landmark).strip().lower()
        if not needle:
            return []
        cells: list[tuple[int, int]] = []
        for key, cell in sorted((getattr(grid, "cells", {}) or {}).items()):
            labels = [str(label).lower() for label in (getattr(cell, "labels", []) or [])]
            if any(needle in label for label in labels):
                cells.append((int(key[0]), int(key[1])))
        return cells

    def _cell_known(self, grid: Any | None, cell: tuple[int, int]) -> bool:
        if grid is None:
            return False
        finder = getattr(grid, "cell", None)
        if callable(finder):
            try:
                return finder(int(cell[0]), int(cell[1])) is not None
            except Exception:  # noqa: BLE001
                pass
        return (int(cell[0]), int(cell[1])) in (getattr(grid, "cells", {}) or {})

    def _find_connection(self, from_area: str, to_area: str) -> Any | None:
        for connection in getattr(self._world, "connections", []) or []:
            if (
                str(getattr(connection, "from_area", "")) == str(from_area)
                and str(getattr(connection, "to_area", "")) == str(to_area)
            ):
                return connection
        return None

    def _trigger_connection(
        self, connection: Any, *, max_steps: int
    ) -> tuple[bool, int, MotionEstimate | None]:
        """Press the recorded connection direction(s) to trigger a transition."""
        direction = str(getattr(connection, "direction", "") or "")
        attempts = [direction] if direction in DIRECTIONS else list(DIRECTIONS)
        steps = 0
        last: MotionEstimate | None = None
        for candidate in attempts:
            if steps >= max(1, int(max_steps)):
                break
            last = self.move_once(candidate, intent="exit")
            steps += 1
            if last.result is MotionResult.TRANSITION:
                return True, steps, last
            if last.result is MotionResult.INTERRUPTED:
                return False, steps, last
        return False, steps, last

    # ------------------------------------------------------------------ #
    # Exploration
    # ------------------------------------------------------------------ #
    def _explorer_finish(self, grid: Any, start: tuple[int, int]) -> tuple[bool, str, dict]:
        should_finish = getattr(self._explorer, "should_finish", None)
        if not callable(should_finish):
            return (False, "", {})
        try:
            finished, reason, stats = should_finish(grid, start)
        except TypeError:  # pragma: no cover
            result = should_finish(grid, start)  # type: ignore[misc]
            finished, reason, stats = result
        return bool(finished), str(reason or ""), dict(stats or {})

    def _explorer_choose(self, grid: Any, start: tuple[int, int]) -> ExploreTarget | None:
        choose = getattr(self._explorer, "choose_target", None)
        if not callable(choose):
            return None
        try:
            return choose(grid, start, step_index=self._steps)
        except TypeError:  # pragma: no cover
            return choose(grid, start)  # type: ignore[misc]

    def _explorer_on_attempt(
        self, target: ExploreTarget | Any, outcome: str
    ) -> None:
        on_attempt = getattr(self._explorer, "on_attempt", None)
        if not callable(on_attempt):
            return
        try:
            on_attempt(target, step_index=self._steps, outcome=outcome)
        except TypeError:  # pragma: no cover
            on_attempt(target, outcome=outcome)  # type: ignore[misc]

    def _record_transition_attempt(self, target: ExploreTarget | Any | None) -> None:
        """Record that following/probing ``target`` ended in an area transition.

        The contract vocabulary of ``Explorer.on_attempt`` is
        ``moved|blocked|unknown|event``; ``transition`` is tried first (requested
        by the Lead) and a rejecting explorer falls back to the neutral
        ``event`` so the attempt still counts (and can blacklist the door).
        """
        if target is None:
            return
        on_attempt = getattr(self._explorer, "on_attempt", None)
        if not callable(on_attempt):
            return
        for outcome in ("transition", "event"):
            try:
                on_attempt(target, step_index=self._steps, outcome=outcome)
                return
            except (TypeError, ValueError):
                continue
            except Exception:  # noqa: BLE001 - attempt bookkeeping is best effort
                return

    @staticmethod
    def _target_is_exit(target: ExploreTarget | Any | None) -> bool:
        """True when the exploration target is an area exit (door/warp)."""
        if target is None:
            return False
        return str(getattr(target, "kind", "") or "").strip().lower() == "exit"

    def _candidate_is_exit(self, grid: Any | None, target: ExploreTarget | Any | None) -> bool:
        """True when a target is an exit, by kind or by the mapped cell state."""
        if self._target_is_exit(target):
            return True
        if grid is None or target is None:
            return False
        cell = tuple(int(value) for value in getattr(target, "cell", (0, 0)))
        finder = getattr(grid, "cell", None)
        if callable(finder):
            try:
                mapped = finder(cell[0], cell[1])
            except Exception:  # noqa: BLE001
                mapped = None
            if mapped is not None:
                return _state_value(getattr(mapped, "state", "")) == CellState.EXIT.value
        return False

    def _cell_is_exit_adjacent(
        self, grid: Any | None, cell: tuple[int, int] | None
    ) -> bool:
        """True when ``cell`` orthogonally touches a known EXIT cell.

        Door/warp regions are contiguous (a 2-tile-wide doorway): probing an
        unknown tile right next to a known exit risks walking through the door
        again while the goal was to cover the room, so room exploration defers
        such frontier probes unless nothing else is left.
        """
        if grid is None or cell is None:
            return False
        finder = getattr(grid, "cell", None)
        if not callable(finder):
            return False
        for delta in DIRECTIONS:
            dx, dy = step_delta(delta)
            try:
                neighbour = finder(int(cell[0]) + dx, int(cell[1]) + dy)
            except Exception:  # noqa: BLE001
                continue
            if neighbour is not None and _state_value(
                getattr(neighbour, "state", "")
            ) == CellState.EXIT.value:
                return True
        return False

    def _non_exit_candidate(
        self, grid: Any, start: tuple[int, int]
    ) -> ExploreTarget | Any | None:
        """Best reachable frontier candidate that is not an exit/door tile."""
        candidates = getattr(self._explorer, "frontier_candidates", None)
        if not callable(candidates):
            return None
        try:
            items = candidates(grid, start) or []
        except Exception:  # noqa: BLE001 - policy query is optional
            return None
        for candidate in items:
            if self._candidate_is_exit(grid, candidate):
                continue
            cell = tuple(int(value) for value in getattr(candidate, "cell", (0, 0)))
            if self._cell_is_exit_adjacent(grid, cell):
                continue
            if str(getattr(candidate, "direction", "") or "") in DIRECTIONS:
                return candidate
        return None

    def _no_progress_limit(self, override: int | None) -> int:
        if override is not None:
            return max(1, int(override))
        config = getattr(self._explorer, "config", None)
        limit = getattr(config, "no_progress_limit", None)
        if limit:
            return max(1, int(limit))
        return 30

    def _plan_for_target(self, target: ExploreTarget | Any) -> PathPlan | None:
        """Resolve the plan that reaches the walkable cell adjacent to a target."""
        pose = tuple(self._localizer.pose.cell)
        path = getattr(target, "path", None)
        if (
            path is not None
            and bool(getattr(path, "reachable", False))
            and self._plan_start(path) == pose
        ):
            return path
        from_cell = getattr(target, "from_cell", None)
        if from_cell is None:
            from_cell = pose
        from_cell = (int(from_cell[0]), int(from_cell[1]))
        if from_cell == pose:
            return PathPlan(
                cells=[pose], directions=[], cost=0.0, reachable=True, reason="already adjacent"
            )
        return self._plan_to(from_cell)

    @staticmethod
    def _motion_outcome(estimate: MotionEstimate) -> str:
        if estimate.result is MotionResult.MOVED:
            return "moved"
        if estimate.result in (MotionResult.BLOCKED, MotionResult.TURNED_ONLY):
            return "blocked"
        if estimate.result is MotionResult.UNKNOWN:
            return "unknown"
        return "event"

    def explore_current_area(
        self,
        *,
        max_steps: int = 300,
        max_seconds: float = 600.0,
        max_no_progress: int | None = None,
    ) -> ExploreReport:
        """Frontier exploration of the current area under hard budgets."""
        started_at = time.time()
        started_monotonic = time.monotonic()
        snapshot = self._snapshot()
        frames_start = self._frame_count()
        self._run_path = []
        self._clear_route()
        report = ExploreReport(
            status=ExploreStatus.ERROR,
            area_id=self.current_area_id(),
            started_at=started_at,
        )
        steps = 0
        no_progress = 0
        progress_limit = self._no_progress_limit(max_no_progress)
        new_cells = 0
        new_edges = 0
        new_exits = 0
        last_reason = ""
        #: (area, coverage, pose) of the explored area at the moment a transition
        #: was handled; keeps the report about the area that was explored.
        locked_context: tuple[str, CoverageStats | None, PlayerPose] | None = None
        try:
            while True:
                if self._paused:
                    report.status = ExploreStatus.PAUSED
                    last_reason = last_reason or "navigation is paused"
                    break
                if steps >= max(0, int(max_steps)):
                    report.status = ExploreStatus.STEP_LIMIT
                    last_reason = f"step budget {int(max_steps)} exhausted"
                    break
                if time.monotonic() - started_monotonic >= float(max_seconds):
                    report.status = ExploreStatus.TIME_LIMIT
                    last_reason = f"time budget {float(max_seconds):.1f}s exhausted"
                    break
                pose = self._localizer.pose
                if not pose.valid:
                    if not self._maybe_relocalize():
                        report.status = ExploreStatus.LOCALIZATION_LOST
                        last_reason = "pose invalid and relocalization failed"
                        break
                    continue
                grid = self.grid()
                if grid is None:
                    report.status = ExploreStatus.ERROR
                    last_reason = "no grid map for the current area"
                    break
                finished, finish_reason, _stats = self._explorer_finish(grid, pose.cell)
                if finished:
                    report.status = ExploreStatus.COMPLETED
                    last_reason = finish_reason or "no reachable frontier left"
                    break
                target = self._explorer_choose(grid, pose.cell)
                if target is None:
                    report.status = ExploreStatus.COMPLETED
                    last_reason = "exploration policy has no target left"
                    break
                # A room-exploration run must not leave through a door while there
                # is still floor to map: never probe an EXIT candidate (or an
                # unknown tile touching a known door) unless it is the only
                # valuable frontier left (then the area counts as covered; leaving
                # is requested explicitly with navigate_to({"exit": true})).
                target_cell = tuple(int(value) for value in getattr(target, "cell", (0, 0)))
                target_is_exit = self._candidate_is_exit(grid, target)
                target_near_exit = self._cell_is_exit_adjacent(grid, target_cell)
                if target_is_exit or target_near_exit:
                    alternative = self._non_exit_candidate(grid, pose.cell)
                    if alternative is not None:
                        target = alternative
                    else:
                        report.status = ExploreStatus.COMPLETED
                        known_exits = self._grid_exits(grid)
                        deferred_doors = (
                            "exit-adjacent" if target_near_exit and not target_is_exit else ""
                        )
                        last_reason = (
                            "only exit frontiers remain "
                            f"(exits={known_exits}"
                            f"{', ' + deferred_doors if deferred_doors else ''}), "
                            "area covered; use navigate_to({'exit': true}) to leave "
                            "the area"
                        )
                        break
                cycle_grid = grid
                size_before = self._grid_size(cycle_grid)
                edges_before = self._grid_edges(cycle_grid)
                exits_before = self._grid_exits(cycle_grid)
                moves_before = self._moves
                plan = self._plan_for_target(target)
                if plan is None:
                    self._explorer_on_attempt(target, "blocked")
                    no_progress += 1
                    last_reason = f"target {tuple(getattr(target, 'cell', (0, 0)))} is not reachable"
                    if no_progress >= progress_limit:
                        report.status = ExploreStatus.NO_PROGRESS
                        break
                    continue
                remaining_steps = max(1, int(max_steps) - steps)
                path_length = len(self._plan_directions(plan))
                if path_length > 0:
                    self._set_route(
                        self._plan_cells(plan), tuple(getattr(target, "cell", (0, 0)))
                    )
                    # Allow at least one replan inside the follow (a dynamic
                    # obstacle can invalidate the first route and a budget of
                    # exactly path_length would return step_limit without trying
                    # the alternative).
                    result = self.follow_path(
                        plan,
                        max_steps=min(remaining_steps, max(2, path_length + 1)),
                        intent="explore",
                    )
                    steps += int(result.steps)
                    if result.status == "event":
                        report.status = ExploreStatus.INTERRUPTED_EVENT
                        report.event = result.event
                        last_reason = result.reason or "non-navigation event interrupted exploration"
                        break
                    if result.status == "transition":
                        if self._candidate_is_exit(grid, target):
                            self._record_transition_attempt(target)
                        locked_context = self._report_context()
                        handled = self.handle_transition()
                        report.area_switched = bool(handled.get("switched"))
                        report.new_area_id = str(handled.get("area_id") or "")
                        if report.area_switched:
                            report.status = ExploreStatus.TRANSITION
                        else:
                            report.status = ExploreStatus.TRANSITION
                        last_reason = (
                            f"area transition during exploration (status={handled.get('status')})"
                        )
                        break
                    if result.status == "localization_lost":
                        report.status = ExploreStatus.LOCALIZATION_LOST
                        last_reason = result.reason or "localization lost while moving to a frontier"
                        break
                    if result.status == "step_limit":
                        report.status = ExploreStatus.STEP_LIMIT
                        last_reason = result.reason or "step budget exhausted"
                        break
                    if result.status == "blocked":
                        self._explorer_on_attempt(target, "blocked")
                        no_progress += 1
                        last_reason = result.reason or "path to frontier blocked"
                        if no_progress >= progress_limit:
                            report.status = ExploreStatus.NO_PROGRESS
                            break
                        continue
                # Probe the frontier cell itself (the only place we press into
                # a non-walkable/unknown cell).
                if self._paused:
                    report.status = ExploreStatus.PAUSED
                    last_reason = "navigation paused by a callback"
                    break
                pose = self._localizer.pose
                target_cell = tuple(int(v) for v in getattr(target, "cell", (0, 0)))
                direction = str(getattr(target, "direction", "") or "")
                if direction in DIRECTIONS and target_cell != tuple(pose.cell):
                    self._set_route([], target_cell)
                    estimate = self.move_once(direction, intent="probe")
                    steps += 1
                    if estimate.result is MotionResult.TURNED_ONLY:
                        # Gen-1 turn-then-step: retry the same direction once.
                        estimate = self.move_once(direction, intent="probe")
                        steps += 1
                    outcome = self._motion_outcome(estimate)
                    self._explorer_on_attempt(target, outcome)
                    if estimate.result is MotionResult.INTERRUPTED:
                        report.status = ExploreStatus.INTERRUPTED_EVENT
                        report.event = self._last_event or self.classify_event(estimate=estimate)
                        last_reason = "non-navigation event while probing a frontier"
                        break
                    if estimate.result is MotionResult.TRANSITION:
                        if self._candidate_is_exit(grid, target):
                            self._record_transition_attempt(target)
                        locked_context = self._report_context()
                        handled = self.handle_transition()
                        report.area_switched = bool(handled.get("switched"))
                        report.new_area_id = str(handled.get("area_id") or "")
                        report.status = ExploreStatus.TRANSITION
                        last_reason = "area transition while probing a frontier"
                        break
                    if (
                        estimate.result is MotionResult.UNKNOWN
                        and self._consecutive_unknown
                        >= max(1, int(self.config.max_consecutive_unknown))
                    ):
                        if not self._maybe_relocalize():
                            report.status = ExploreStatus.LOCALIZATION_LOST
                            last_reason = "too many consecutive UNKNOWN estimates while probing"
                            break
                # Progress accounting: moving or growing the map both count.
                grid_after = self.grid()
                cycle_new_cells = max(0, self._grid_size(grid_after) - size_before)
                cycle_new_edges = max(0, self._grid_edges(grid_after) - edges_before)
                cycle_new_exits = max(0, self._grid_exits(grid_after) - exits_before)
                new_cells += cycle_new_cells
                new_edges += cycle_new_edges
                new_exits += cycle_new_exits
                if self._moves > moves_before or cycle_new_cells > 0:
                    no_progress = 0
                else:
                    no_progress += 1
                if no_progress >= progress_limit:
                    report.status = ExploreStatus.NO_PROGRESS
                    last_reason = f"no progress for {no_progress} consecutive cycles"
                    break
        except Exception as exc:  # noqa: BLE001 - report structured failure, never crash
            report.status = ExploreStatus.ERROR
            last_reason = f"{type(exc).__name__}: {exc}"
        finally:
            report.reason = last_reason
            report.steps = self._steps - snapshot["steps"]
            report.moves = self._moves - snapshot["moves"]
            report.blocked = self._blocked - snapshot["blocked"]
            report.turned = self._turned - snapshot["turned"]
            report.unknown = self._unknown - snapshot["unknown"]
            report.backtracked = self._backtracked - snapshot["backtracked"]
            report.new_cells = new_cells
            report.new_edges = new_edges
            report.new_exits = new_exits
            if locked_context is not None:
                # Report about the area that was being explored, not the new area
                # we transitioned into.
                locked_area, locked_coverage, locked_pose = locked_context
                report.area_id = locked_area or report.area_id
                report.coverage = locked_coverage
                report.pose = locked_pose
            else:
                report.coverage = self._coverage()
                report.pose = self._localizer.pose
                report.area_id = report.area_id or self.current_area_id()
            report.elapsed_s = max(0.0, time.monotonic() - started_monotonic)
            report.frames = max(0, self._frame_count() - frames_start)
            report.path = list(self._run_path)
            self._run_path = []
            self._clear_route()
            self._notify_status()
        return report

    # ------------------------------------------------------------------ #
    # Goal-directed navigation
    # ------------------------------------------------------------------ #
    def _parse_target(
        self, target: dict[str, Any] | None
    ) -> tuple[str, str | None, tuple[int, int] | None, str | None] | None:
        """Normalise a navigation target dict (contract forms only)."""
        if not isinstance(target, dict):
            return None
        cell = target.get("cell")
        area = target.get("area")
        if cell is not None:
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                return None
            try:
                parsed = (int(cell[0]), int(cell[1]))
            except (TypeError, ValueError):
                return None
            if area is not None:
                return ("area_cell", str(area), parsed, None)
            return ("cell", None, parsed, None)
        if target.get("exit"):
            return ("exit", None, None, None)
        landmark = target.get("landmark")
        if landmark:
            return ("landmark", None, None, str(landmark))
        if target.get("return_to_entry"):
            return ("return_to_entry", None, None, None)
        return None

    def navigate_to(self, target: dict[str, Any], *, max_steps: int = 200) -> NavReport:
        """Navigate to a structured target (no LLM calls; events are reported)."""
        started_at = time.time()
        started_monotonic = time.monotonic()
        snapshot = self._snapshot()
        frames_start = self._frame_count()
        self._run_path = []
        self._clear_route()
        try:
            target_dict = dict(target) if isinstance(target, dict) else {}
        except (TypeError, ValueError):
            target_dict = {}
        report = NavReport(
            status=ExploreStatus.ERROR,
            area_id=self.current_area_id(),
            target=target_dict,
            started_at=started_at,
        )
        budget = max(0, min(int(max_steps), int(self.config.max_navigate_steps)))
        steps = 0
        area_hops = max(0, int(self.config.max_area_hops))
        last_reason = ""
        spec = self._parse_target(target_dict)
        if spec is None:
            report.status = ExploreStatus.ERROR
            report.reason = (
                "invalid navigation target; expected one of "
                "{'cell':[x,y]}, {'exit':True}, {'landmark':str}, "
                "{'area':id,'cell':[x,y]}, {'return_to_entry':True}"
            )
            self._fill_nav_report(report, snapshot, started_at, started_monotonic, frames_start)
            return report
        kind, goal_area, goal_cell, landmark = spec
        previous_pose: tuple[int, int] | None = None
        try:
            while True:
                if self._paused:
                    report.status = ExploreStatus.PAUSED
                    last_reason = last_reason or "navigation is paused"
                    break
                if steps >= budget:
                    report.status = ExploreStatus.STEP_LIMIT
                    last_reason = f"step budget {budget} exhausted"
                    break
                pose = self._localizer.pose
                if not pose.valid:
                    if not self._maybe_relocalize():
                        report.status = ExploreStatus.LOCALIZATION_LOST
                        last_reason = "pose invalid and relocalization failed"
                        break
                    previous_pose = None
                    continue
                current_area = self.current_area_id()
                grid = self.grid()
                if grid is None:
                    report.status = ExploreStatus.ERROR
                    last_reason = "no grid map for the current area"
                    break
                # ---------------------------------------------------------- #
                # Cross-area leg: follow a known connection toward goal_area.
                # ---------------------------------------------------------- #
                if kind == "area_cell" and goal_area and goal_area != current_area:
                    if area_hops <= 0:
                        report.status = ExploreStatus.BLOCKED
                        last_reason = f"area hop budget exhausted before reaching {goal_area}"
                        break
                    connection = self._find_connection(current_area, goal_area)
                    if connection is None:
                        report.status = ExploreStatus.BLOCKED
                        last_reason = (
                            f"no known route from {current_area!r} to {goal_area!r}; "
                            "explore/record a connection first"
                        )
                        break
                    from_cell = tuple(int(v) for v in getattr(connection, "from_cell", (0, 0)))
                    if not self._cell_known(grid, from_cell):
                        report.status = ExploreStatus.BLOCKED
                        last_reason = f"connection cell {from_cell} is not on the mapped grid"
                        break
                    plan = (
                        PathPlan(
                            cells=[tuple(pose.cell)],
                            directions=[],
                            cost=0.0,
                            reachable=True,
                            reason="already at connection cell",
                        )
                        if from_cell == tuple(pose.cell)
                        else self._plan_to(from_cell)
                    )
                    if plan is None:
                        report.status = ExploreStatus.BLOCKED
                        last_reason = f"cannot reach the connection cell {from_cell}"
                        break
                    if plan.directions:
                        self._set_route(self._plan_cells(plan), from_cell)
                        leg = self.follow_path(
                            plan, max_steps=max(1, budget - steps), intent="navigate"
                        )
                        steps += int(leg.steps)
                        if leg.status == "event":
                            report.status = ExploreStatus.INTERRUPTED_EVENT
                            report.event = leg.event
                            last_reason = leg.reason
                            break
                        if leg.status == "transition":
                            handled = self.handle_transition()
                            report.area_switched = report.area_switched or bool(
                                handled.get("switched")
                            )
                            report.new_area_id = str(handled.get("area_id") or "")
                            if str(handled.get("status")) in ("matched", "new_area"):
                                area_hops -= 1
                                continue
                            report.status = ExploreStatus.TRANSITION
                            last_reason = "transition handling failed on the way to a connection"
                            break
                        if leg.status == "localization_lost":
                            report.status = ExploreStatus.LOCALIZATION_LOST
                            last_reason = leg.reason
                            break
                        if leg.status == "step_limit":
                            report.status = ExploreStatus.STEP_LIMIT
                            last_reason = leg.reason
                            break
                        if leg.status == "blocked":
                            report.status = ExploreStatus.BLOCKED
                            last_reason = leg.reason or "path to connection blocked"
                            break
                    if tuple(self._localizer.pose.cell) != from_cell:
                        report.status = ExploreStatus.BLOCKED
                        last_reason = "could not reach the connection cell"
                        break
                    self._set_route([], from_cell)
                    triggered, trigger_steps, trigger_estimate = self._trigger_connection(
                        connection, max_steps=max(1, budget - steps)
                    )
                    steps += trigger_steps
                    if (
                        trigger_estimate is not None
                        and trigger_estimate.result is MotionResult.INTERRUPTED
                    ):
                        report.status = ExploreStatus.INTERRUPTED_EVENT
                        report.event = self._last_event or self.classify_event(
                            estimate=trigger_estimate
                        )
                        last_reason = "non-navigation event at the connection cell"
                        break
                    if not triggered:
                        report.status = ExploreStatus.BLOCKED
                        last_reason = (
                            f"connection via {getattr(connection, 'direction', '?')!r} "
                            "did not trigger a transition"
                        )
                        break
                    handled = self.handle_transition()
                    report.area_switched = report.area_switched or bool(handled.get("switched"))
                    report.new_area_id = str(handled.get("area_id") or "")
                    area_hops -= 1
                    if self.current_area_id() != goal_area:
                        report.status = ExploreStatus.TRANSITION
                        last_reason = (
                            f"transitioned to area {self.current_area_id()!r} "
                            f"instead of {goal_area!r}"
                        )
                        break
                    continue
                # ---------------------------------------------------------- #
                # Same-area objective.
                # ---------------------------------------------------------- #
                target_cells: list[tuple[int, int]] = []
                reason_prefix = ""
                if kind in ("cell", "area_cell"):
                    target_cells = [(int(goal_cell[0]), int(goal_cell[1]))]  # type: ignore[index]
                    reason_prefix = f"cell {target_cells[0]}"
                elif kind == "exit":
                    target_cells = self._cells_with_state(grid, CellState.EXIT)
                    reason_prefix = "area exit"
                    if not target_cells:
                        report.status = ExploreStatus.BLOCKED
                        last_reason = "no area exit is known on the current map"
                        break
                elif kind == "landmark":
                    target_cells = self._landmark_cells(landmark or "")
                    reason_prefix = f"landmark {landmark!r}"
                    if not target_cells:
                        report.status = ExploreStatus.BLOCKED
                        last_reason = f"landmark {landmark!r} is not on the current map"
                        break
                else:  # return_to_entry
                    area = self._area()
                    entries = list(getattr(area, "entry_cells", []) or []) if area is not None else []
                    target_cells = [
                        (int(cell[0]), int(cell[1])) for cell in entries if len(cell) == 2
                    ]
                    reason_prefix = "area entry"
                    if not target_cells:
                        report.status = ExploreStatus.BLOCKED
                        last_reason = "no entry cell recorded for the current area"
                        break
                pose_cell = tuple(self._localizer.pose.cell)
                if pose_cell in target_cells:
                    report.reached = True
                    report.status = ExploreStatus.COMPLETED
                    last_reason = f"reached {reason_prefix}"
                    break
                plan = (
                    self._plan_to(target_cells[0])
                    if len(target_cells) == 1
                    else self._plan_to_nearest(target_cells)
                )
                if plan is None:
                    report.status = ExploreStatus.BLOCKED
                    last_reason = f"{reason_prefix} is not reachable from {pose_cell}"
                    break
                if not plan.directions:
                    report.reached = True
                    report.status = ExploreStatus.COMPLETED
                    last_reason = f"reached {reason_prefix}"
                    break
                self._set_route(self._plan_cells(plan), self._plan_goal(plan))
                leg = self.follow_path(plan, max_steps=max(1, budget - steps), intent="navigate")
                steps += int(leg.steps)
                if leg.status == "event":
                    report.status = ExploreStatus.INTERRUPTED_EVENT
                    report.event = leg.event
                    last_reason = leg.reason
                    break
                if leg.status == "localization_lost":
                    report.status = ExploreStatus.LOCALIZATION_LOST
                    last_reason = leg.reason
                    break
                if leg.status == "step_limit":
                    report.status = ExploreStatus.STEP_LIMIT
                    last_reason = leg.reason
                    break
                if leg.status == "blocked":
                    report.status = ExploreStatus.BLOCKED
                    last_reason = leg.reason
                    break
                if leg.status == "transition":
                    if kind == "exit":
                        exit_cell = self._plan_goal(plan) or tuple(self._localizer.pose.cell)
                        from_cell = self._plan_start(plan) or exit_cell
                        directions = self._plan_directions(plan)
                        exit_target = ExploreTarget(
                            cell=exit_cell,
                            from_cell=from_cell,
                            direction=directions[-1] if directions else "",
                            kind="exit",
                            reason="explicit exit navigation crossed a transition",
                        )
                        self._record_transition_attempt(exit_target)
                    handled = self.handle_transition()
                    report.area_switched = report.area_switched or bool(handled.get("switched"))
                    report.new_area_id = str(handled.get("area_id") or "")
                    if kind == "area_cell" and self.current_area_id() == (goal_area or ""):
                        continue
                    if kind == "exit" and bool(handled.get("handled")):
                        # walking onto/past the exit crossed the transition: the
                        # exit target was reached even though the run ends here
                        report.reached = True
                    report.status = ExploreStatus.TRANSITION
                    last_reason = "area transition while navigating to the target"
                    break
                current = tuple(self._localizer.pose.cell)
                if current in target_cells:
                    report.reached = True
                    report.status = ExploreStatus.COMPLETED
                    last_reason = f"reached {reason_prefix}"
                    break
                if previous_pose is not None and current == previous_pose:
                    report.status = ExploreStatus.BLOCKED
                    last_reason = f"no progress towards {reason_prefix}"
                    break
                previous_pose = current
        except Exception as exc:  # noqa: BLE001 - structured failure, never crash
            report.status = ExploreStatus.ERROR
            last_reason = f"{type(exc).__name__}: {exc}"
        finally:
            report.reason = last_reason
            self._fill_nav_report(report, snapshot, started_at, started_monotonic, frames_start)
        return report

    def _fill_nav_report(
        self,
        report: NavReport,
        snapshot: dict[str, int],
        started_at: float,
        started_monotonic: float,
        frames_start: int,
    ) -> None:
        report.steps = self._steps - snapshot["steps"]
        report.moves = self._moves - snapshot["moves"]
        report.blocked = self._blocked - snapshot["blocked"]
        report.replans = self._replans - snapshot["replans"]
        report.pose = self._localizer.pose
        report.coverage = self._coverage()
        report.area_id = self.current_area_id() or report.area_id
        report.started_at = started_at
        report.elapsed_s = max(0.0, time.monotonic() - started_monotonic)
        report.frames = max(0, self._frame_count() - frames_start)
        self._run_path = []
        self._clear_route()
        self._notify_status()

    # ------------------------------------------------------------------ #
    # Report snapshot helpers
    # ------------------------------------------------------------------ #
    def _snapshot(self) -> dict[str, int]:
        return {
            "steps": self._steps,
            "moves": self._moves,
            "blocked": self._blocked,
            "turned": self._turned,
            "unknown": self._unknown,
            "replans": self._replans,
            "backtracked": self._backtracked,
            "log": len(self._step_log),
        }

    def _fill_result(self, result: ControllerResult, snapshot: dict[str, int]) -> None:
        result.steps = self._steps - snapshot["steps"]
        result.moves = self._moves - snapshot["moves"]
        result.blocked = self._blocked - snapshot["blocked"]
        result.unknown = self._unknown - snapshot["unknown"]
        result.replans = self._replans - snapshot["replans"]
        start = int(snapshot["log"])
        result.steps_log = list(self._step_log[start:])
