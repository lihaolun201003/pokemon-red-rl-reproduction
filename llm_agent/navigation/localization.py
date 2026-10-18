"""Pure-vision motion estimation and pose tracking for Navigation v2.

The localizer answers exactly one question per directional input: **did the player
actually move, and by how many tiles?**  It uses nothing but the 160x144
framebuffer and the agent's own action history — no RAM, no map ids.

Verified behaviour of this ROM (real PyBoy frames, see the design report):

* The overworld camera keeps the player sprite at a fixed screen tile (4,4).
  A successful step therefore shifts the **background content** by exactly one
  tile: ``best_shift(prev, cur) == -world_delta`` with a pixel-match score of
  about 0.98-0.993 (NPC animation lowers it slightly).
* A blocked press that keeps the facing is **pixel identical** (score 1.0 at
  shift (0,0)).
* A blocked press that turns the player changes only pixels inside the player
  sprite box (< 1 % of the screen, fully inside ``PLAYER_BOX``).
* A door/warp transition changes the screen completely; dialog/menu/battle
  screens are recognised and must not update the map.
* A held direction may produce several tiles in one action (score still high at
  ``-2`` etc.); this is handled as ``MOVED`` with ``tiles_moved == k``.

The governing rule from the task: **when the evidence is insufficient, return
``UNKNOWN`` and do not touch the coordinates.**  The controller then performs a
short observation / relocalization instead of accumulating drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_agent.contracts import DIRECTIONS, Frame, ScreenEvent
from llm_agent.navigation import vision
from llm_agent.navigation.types import (
    DIRECTION_DELTAS,
    FrameContext,
    FrameView,
    MotionEstimate,
    MotionResult,
    PlayerPose,
    TileObs,
    step_delta,
)

#: ScreenEvent -> what kind of frame we are looking at
_EVENT_CONTEXT: dict[ScreenEvent, FrameContext] = {
    ScreenEvent.NONE: FrameContext.OVERWORLD,
    ScreenEvent.DIALOG: FrameContext.DIALOG,
    ScreenEvent.MENU: FrameContext.MENU,
    ScreenEvent.BATTLE: FrameContext.BATTLE,
    ScreenEvent.MAP_CHANGE: FrameContext.TRANSITION,
    ScreenEvent.TRANSITION: FrameContext.TRANSITION,
    ScreenEvent.UNKNOWN: FrameContext.UNKNOWN,
}


@dataclass
class LocalizationConfig:
    """Thresholds for the visual motion estimator (calibrated on real frames)."""

    #: candidate background shifts, in tiles (one held press is usually 1 tile)
    max_tiles: int = 2
    #: minimum pixel-match score for a *confident* MOVED classification
    move_min_score: float = 0.965
    #: relaxed MOVED threshold for animation-heavy outdoor frames: additionally
    #: requires this much margin over the no-shift hypothesis (NPCs/flowers can
    #: add a few percent of changed pixels on real Pallet Town frames)
    move_relaxed_score: float = 0.93
    move_relaxed_margin: float = 0.22
    #: below this score even the best shift is not movement evidence
    move_reject_score: float = 0.90
    #: score(0,0) required to call a no-shift result BLOCKED with high confidence
    blocked_min_score: float = 0.985
    #: a turn-only change must be small and inside the player sprite box
    turn_max_diff_ratio: float = 0.02
    turn_min_inside_ratio: float = 0.90
    #: two shift candidates closer than this are ambiguous -> UNKNOWN
    ambiguous_margin: float = 0.003
    #: a no-shift frame with more changed pixels than this may be on-screen motion
    onscreen_suspect_pixels: int = 320
    #: below this best-shift score a big visual change is a transition
    transition_max_score: float = 0.90
    transition_min_diff_ratio: float = 0.35
    #: consecutive UNKNOWN results before the pose is invalidated
    max_unknown_streak: int = 2
    #: confidence lost per UNKNOWN result
    unknown_confidence_decay: float = 0.5
    #: score at/above which a no-shift frame is considered "clean"
    clean_no_shift_score: float = 0.995


@dataclass
class _MotionEvidence:
    """Human/audit friendly evidence bundle (JSON-safe)."""

    shift: tuple[int, int] = (0, 0)
    shift_score: float = 0.0
    second_shift: tuple[int, int] | None = None
    second_score: float = 0.0
    diff_ratio: float = 0.0
    diff_pixels: int = 0
    inside_player: int = 0
    outside_player: int = 0
    components: int = 0
    bbox: tuple[int, int, int, int] | None = None
    event: str = "none"
    changed_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shift": list(self.shift),
            "shift_score": round(self.shift_score, 4),
            "second_shift": None if self.second_shift is None else list(self.second_shift),
            "second_score": round(self.second_score, 4),
            "diff_ratio": round(self.diff_ratio, 5),
            "diff_pixels": int(self.diff_pixels),
            "inside_player": int(self.inside_player),
            "outside_player": int(self.outside_player),
            "components": int(self.components),
            "bbox": None if self.bbox is None else list(self.bbox),
            "event": self.event,
            "changed_ratio": round(self.changed_ratio, 4),
        }


class Localizer:
    """Visual dead-reckoning with confidence and explicit UNKNOWN handling.

    Usage::

        loc = Localizer()
        prev = loc.observe(frame0)
        est = loc.update(prev, loc.observe(frame1), "up")
        if est.result is MotionResult.MOVED:
            ...  # loc.pose has already advanced
    """

    def __init__(self, config: LocalizationConfig | None = None) -> None:
        self.config = config or LocalizationConfig()
        self._pose = PlayerPose(x=0, y=0, facing=None, confidence=1.0, valid=True)
        self._area_id = ""
        self._last_view: FrameView | None = None
        self._unknown_streak = 0
        self._moves = 0
        self._blocked = 0
        self._turned = 0
        self._unknown = 0
        self._transitions = 0
        self._interruptions = 0
        self.last_motion: MotionEstimate | None = None
        self.last_transition_frame: int = -1

    # ------------------------------------------------------------------ #
    # Pose state
    # ------------------------------------------------------------------ #
    @property
    def pose(self) -> PlayerPose:
        return self._pose

    @property
    def area_id(self) -> str:
        return self._area_id

    @property
    def last_view(self) -> FrameView | None:
        return self._last_view

    @property
    def stats(self) -> dict[str, int]:
        return {
            "moves": self._moves,
            "blocked": self._blocked,
            "turned": self._turned,
            "unknown": self._unknown,
            "transitions": self._transitions,
            "interruptions": self._interruptions,
        }

    def reset(
        self,
        *,
        area_id: str = "",
        cell: tuple[int, int] = (0, 0),
        facing: str | None = None,
    ) -> None:
        """Start a fresh area coordinate frame."""
        self._area_id = str(area_id)
        self._pose = PlayerPose(
            x=int(cell[0]),
            y=int(cell[1]),
            facing=facing if facing in DIRECTIONS else None,
            confidence=1.0,
            valid=True,
        )
        self._last_view = None
        self._unknown_streak = 0
        self._transitions += 1
        self.last_transition_frame = -1

    def set_pose(
        self,
        x: int,
        y: int,
        *,
        facing: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Overwrite the pose (used after relocalization)."""
        self._pose = PlayerPose(
            x=int(x),
            y=int(y),
            facing=facing if facing in DIRECTIONS else self._pose.facing,
            confidence=float(max(0.0, min(1.0, confidence))),
            valid=True,
        )
        self._unknown_streak = 0

    def invalidate(self, reason: str = "") -> None:
        """Mark the pose as lost (controller must relocalize)."""
        self._pose.valid = False
        self._pose.confidence = 0.0
        self.last_motion = MotionEstimate(
            result=MotionResult.UNKNOWN,
            direction="",
            confidence=0.0,
            evidence={"reason": reason or "localization lost"},
        )

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def observe(self, frame: Frame, *, frame_context: FrameContext | None = None) -> FrameView:
        """Build a :class:`FrameView` (tiles, signature, screen context)."""
        luma = vision.frame_luma(frame)
        offset = vision.tile_grid_offset(luma)
        tiles = vision.extract_tiles(luma, offset)
        event = ScreenEvent.NONE
        if frame_context is None:
            try:
                from llm_agent.observation import analyze as _analyze

                event = _analyze(frame, None).event
            except Exception:  # noqa: BLE001 - observation is a hint, never required
                event = ScreenEvent.NONE
        context = (
            frame_context
            if frame_context is not None
            else _EVENT_CONTEXT.get(event, FrameContext.UNKNOWN)
        )
        is_black = vision.is_black(luma)
        if is_black:
            context = FrameContext.BLACK
        view = FrameView(
            frame_index=frame.frame_index,
            signature=vision.screen_signature(luma),
            grid_offset=offset,
            tiles=tiles,
            context=context,
            screen_event=event.value,
            is_black=is_black,
            luma_mean=float(luma.mean()),
        )
        # `stable` means "this screenshot is frozen versus the previous capture".
        # A single observation cannot know that (and transition fades can look
        # static for a few frames), so callers must establish stability with a
        # sequence of signatures.  Never report a false positive here.
        view.stable = False
        # Raw arrays are cached on the view for cheap motion estimation; they are
        # never serialised (notes is not part of any JSON output).
        view.notes["luma"] = luma
        view.notes["frame"] = frame
        self._last_view = view
        if context in (FrameContext.DIALOG, FrameContext.MENU, FrameContext.BATTLE):
            self._pose.facing = self._pose.facing  # facing unknown while occluded
        return view

    # ------------------------------------------------------------------ #
    # Motion estimation
    # ------------------------------------------------------------------ #
    def update(
        self,
        prev: FrameView,
        cur: FrameView,
        direction: str,
        *,
        requested_tiles: int = 1,
    ) -> MotionEstimate:
        """Classify one directional input; advances the pose only on MOVED.

        ``prev`` and ``cur`` must come from :meth:`observe` (they carry the raw
        luma arrays in ``notes``).  One call == exactly one input action.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        prev_luma = prev.notes.get("luma")
        cur_luma = cur.notes.get("luma")
        if prev_luma is None or cur_luma is None:
            raise ValueError("FrameView objects passed to Localizer.update must come from Localizer.observe")
        requested_tiles = max(1, int(requested_tiles))

        event, changed_ratio = self._classify_event(prev, cur)
        evidence = _MotionEvidence(event=event.value, changed_ratio=changed_ratio)

        # 0) A black frame means a fade/warp is in progress (or a battle intro):
        #    never derive a movement from it.
        if bool(cur.is_black) or bool(prev.is_black):
            return self._finish(
                MotionResult.TRANSITION, direction, evidence, cur, requested_tiles
            )

        # 1) Unambiguous screen occlusions first: dialogs/battles/menus/black
        #    frames never move us.  (These come from pixel heuristics that do not
        #    depend on a screenshot-difference ratio, so they cannot be fooled by
        #    a large camera scroll.)
        context_result = self._context_result(cur.context, prev.context)
        if context_result is not None:
            return self._finish(context_result, direction, evidence, cur, requested_tiles)

        # 2) Background shift is the primary movement signal.  A high-quality tile
        #    shift wins over the frame-difference classifier: a legit one-tile step
        #    can change >55% of the pixels on a detailed map, which must NOT be
        #    confused with a map transition.
        shift = vision.best_shift(prev_luma, cur_luma, max_tiles=self.config.max_tiles)
        evidence.shift = shift.shift
        evidence.shift_score = shift.score
        evidence.second_shift = shift.second_shift
        evidence.second_score = shift.second_score
        diff = vision.diff_stats(prev_luma, cur_luma)
        evidence.diff_ratio = float(diff["ratio"])
        evidence.diff_pixels = int(diff["pixels"])
        evidence.inside_player = int(diff["inside_player"])
        evidence.outside_player = int(diff["outside_player"])
        evidence.components = int(diff["components"])
        evidence.bbox = diff["bbox"]

        if shift.shift != (0, 0):
            world_delta = (-int(shift.shift[0]), -int(shift.shift[1]))
            return self._classify_shift(
                direction, requested_tiles, shift, world_delta, evidence, cur
            )

        # 3) No background shift: a genuine map/warp transition is a huge change
        #    whose shift cannot be explained; otherwise blocked / turn-only.
        if (
            event in (ScreenEvent.MAP_CHANGE, ScreenEvent.TRANSITION)
            or cur.context is FrameContext.TRANSITION
            or (
                float(diff["ratio"]) >= self.config.transition_min_diff_ratio
                and shift.score < self.config.move_reject_score
            )
            or (
                float(diff["ratio"]) >= self.config.transition_min_diff_ratio
                and event is ScreenEvent.UNKNOWN
            )
        ):
            return self._finish(MotionResult.TRANSITION, direction, evidence, cur, requested_tiles)
        return self._classify_no_shift(direction, shift, diff, evidence, cur, requested_tiles)

    # -- internals ------------------------------------------------------ #
    def _classify_event(self, prev: FrameView, cur: FrameView) -> tuple[ScreenEvent, float]:
        prev_frame = prev.notes.get("frame")
        cur_frame = cur.notes.get("frame")
        if prev_frame is not None and cur_frame is not None:
            try:
                from llm_agent.observation import analyze as _analyze

                analysis = _analyze(cur_frame, prev_frame)
                # Trust the pixel-only event classifier, but never overrule an
                # explicitly passed context for non-overworld frames.
                return analysis.event, float(analysis.changed_ratio)
            except Exception:  # noqa: BLE001 - fall back to the stored context
                pass
        return ScreenEvent.NONE, 0.0

    @staticmethod
    def _context_result(
        cur_context: FrameContext,
        prev_context: FrameContext,
    ) -> MotionResult | None:
        """Unambiguous occlusion checks that never depend on frame-difference ratios."""
        if cur_context is FrameContext.BLACK:
            return MotionResult.TRANSITION
        if cur_context is FrameContext.BATTLE:
            return MotionResult.INTERRUPTED
        if cur_context is FrameContext.DIALOG:
            return MotionResult.INTERRUPTED
        if cur_context is FrameContext.MENU:
            return MotionResult.INTERRUPTED
        if prev_context is FrameContext.BLACK and cur_context is FrameContext.OVERWORLD:
            # fade-in after a transition: the map may have changed
            return MotionResult.TRANSITION
        return None

    def _classify_shift(
        self,
        direction: str,
        requested_tiles: int,
        shift: vision.ShiftScore,
        world_delta: tuple[int, int],
        evidence: _MotionEvidence,
        cur: FrameView,
    ) -> MotionEstimate:
        config = self.config
        expected = step_delta(direction, requested_tiles)
        # Multi-tile presses produce shift k*expected; accept any k in 1..max.
        axis_aligned = (
            (world_delta[0] == 0 and world_delta[1] == 0)
            or (world_delta[0] == 0) != (world_delta[1] == 0)
        )
        parallel = (
            world_delta[0] == 0 and world_delta[1] == 0
        ) or (
            (world_delta[0] == 0) == (expected[0] == 0)
            and (world_delta[1] == 0) == (expected[1] == 0)
            and (world_delta[0] * expected[0] + world_delta[1] * expected[1]) > 0
        )
        tiles_moved = abs(int(world_delta[0])) + abs(int(world_delta[1]))
        axis_consistent = parallel and axis_aligned and 1 <= tiles_moved <= max(
            requested_tiles, config.max_tiles
        )
        ambiguous = (
            shift.second_shift is not None
            and shift.second_score >= shift.score - config.ambiguous_margin
            and shift.second_shift != shift.shift
        )
        zero_score = float(shift.scores.get((0, 0), 0.0))
        strong_move = shift.score >= config.move_min_score
        relaxed_move = (
            shift.score >= config.move_relaxed_score
            and (shift.score - zero_score) >= config.move_relaxed_margin
        )
        if (strong_move or relaxed_move) and axis_consistent and not ambiguous:
            confidence = self._move_confidence(shift, tiles_moved)
            self._moves += 1
            delta = world_delta
            self._pose.x += int(delta[0])
            self._pose.y += int(delta[1])
            self._pose.facing = direction
            self._pose.confidence = max(self._pose.confidence * 0.99, 0.5)
            self._pose.valid = True
            self._unknown_streak = 0
            return self._finish(
                MotionResult.MOVED,
                direction,
                evidence,
                cur,
                requested_tiles,
                delta=delta,
                tiles_moved=tiles_moved,
                camera_shift=(int(shift.shift[0]), int(shift.shift[1])),
                confidence=confidence,
            )
        if shift.score < config.move_reject_score:
            # Big unexplained change: most likely a transition/warp.
            if evidence.diff_ratio >= config.transition_min_diff_ratio:
                return self._finish(
                    MotionResult.TRANSITION, direction, evidence, cur, requested_tiles
                )
            return self._finish(
                MotionResult.UNKNOWN, direction, evidence, cur, requested_tiles
            )
        # High score but not parallel to the requested direction / ambiguous.
        evidence.to_dict()
        return self._finish(
            MotionResult.UNKNOWN,
            direction,
            evidence,
            cur,
            requested_tiles,
            extra={"reason": "shift inconsistent with input or ambiguous"},
        )

    def _classify_no_shift(
        self,
        direction: str,
        shift: vision.ShiftScore,
        diff: dict[str, Any],
        evidence: _MotionEvidence,
        cur: FrameView,
        requested_tiles: int,
    ) -> MotionEstimate:
        config = self.config
        score = shift.score
        pixels = int(diff["pixels"])
        if pixels == 0 and score >= config.clean_no_shift_score:
            self._blocked += 1
            self._pose.facing = direction
            self._unknown_streak = 0
            return self._finish(
                MotionResult.BLOCKED, direction, evidence, cur, requested_tiles,
                confidence=min(1.0, score),
            )
        if score >= config.blocked_min_score:
            if (
                float(diff["ratio"]) <= config.turn_max_diff_ratio
                and float(diff["inside_ratio"]) >= config.turn_min_inside_ratio
                and pixels > 0
            ):
                # The only changed pixels are the player sprite: a turn in place.
                self._turned += 1
                self._blocked += 1
                self._pose.facing = direction
                self._unknown_streak = 0
                return self._finish(
                    MotionResult.TURNED_ONLY, direction, evidence, cur, requested_tiles,
                    confidence=float(min(1.0, diff["inside_ratio"])),
                )
            # Dynamic world (NPC/animation) but no camera shift: still blocked.
            self._blocked += 1
            self._pose.facing = direction
            self._unknown_streak = 0
            confidence = float(max(0.55, min(0.95, score)))
            return self._finish(
                MotionResult.BLOCKED, direction, evidence, cur, requested_tiles,
                confidence=confidence,
                extra={"reason": "no shift; changes outside the player box (dynamic objects)"},
            )
        # Low match with no shift: possible unusual state -> never guess.
        if pixels >= config.onscreen_suspect_pixels:
            return self._finish(
                MotionResult.UNKNOWN, direction, evidence, cur, requested_tiles,
                extra={"reason": "no shift but large unexplained change"},
            )
        return self._finish(MotionResult.UNKNOWN, direction, evidence, cur, requested_tiles)

    def _move_confidence(self, shift: vision.ShiftScore, tiles_moved: int) -> float:
        base = shift.score
        # A one-tile move on real frames scores ~0.98-0.993; multi-tile slightly less.
        confidence = min(1.0, max(0.0, (base - 0.90) / 0.09))
        if tiles_moved > 1:
            confidence *= 0.95
        if shift.second_score > 0.5:
            confidence *= max(0.6, 1.0 - (shift.second_score - 0.5))
        return round(confidence, 4)

    def _finish(
        self,
        result: MotionResult,
        direction: str,
        evidence: _MotionEvidence,
        cur: FrameView,
        requested_tiles: int,
        *,
        delta: tuple[int, int] = (0, 0),
        tiles_moved: int = 0,
        camera_shift: tuple[int, int] = (0, 0),
        confidence: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> MotionEstimate:
        evidence_dict = evidence.to_dict()
        if extra:
            evidence_dict.update(extra)
        if result is MotionResult.BLOCKED or result is MotionResult.TURNED_ONLY:
            evidence_dict["blocked_direction"] = direction
        if result is MotionResult.UNKNOWN:
            self._unknown += 1
            self._unknown_streak += 1
            self._pose.confidence = max(
                0.0, self._pose.confidence * self.config.unknown_confidence_decay
            )
            if self._unknown_streak >= self.config.max_unknown_streak:
                self._pose.valid = False
        elif result is MotionResult.TRANSITION:
            self._transitions += 1
            self._pose.valid = False
            self._pose.confidence = 0.0
            self.last_transition_frame = cur.frame_index
        elif result is MotionResult.INTERRUPTED:
            self._interruptions += 1
        estimate = MotionEstimate(
            result=result,
            direction=direction,
            requested_tiles=requested_tiles,
            delta=delta,
            tiles_moved=tiles_moved,
            camera_shift=camera_shift,
            screen_delta=(0, 0),
            confidence=float(confidence),
            evidence=evidence_dict,
            frame_index=cur.frame_index,
            context=cur.context,
        )
        self.last_motion = estimate
        return estimate

    # ------------------------------------------------------------------ #
    # Convenience helpers used by the controller / tests
    # ------------------------------------------------------------------ #
    def estimate_motion(self, prev_frame: Frame, cur_frame: Frame, direction: str) -> MotionEstimate:
        """One-shot convenience wrapper around :meth:`observe` + :meth:`update`."""
        prev = self.observe(prev_frame)
        cur = self.observe(cur_frame)
        return self.update(prev, cur, direction)

    def remaining_confidence(self) -> float:
        return float(self._pose.confidence * (1.0 if self._pose.valid else 0.0))

    def summary(self) -> dict[str, Any]:
        return {
            "area_id": self._area_id,
            "pose": self._pose.to_dict(),
            **self.stats,
        }


__all__ = ["LocalizationConfig", "Localizer"]
