"""Unit tests for the pure-vision localizer (no PyBoy, no RAM)."""

from __future__ import annotations

import numpy as np
import pytest

from llm_agent.contracts import Frame
from llm_agent.navigation import localization, vision
from llm_agent.navigation.types import FrameContext, MotionResult

from tests.navigation.conftest import (
    SyntheticEmulator,
    SyntheticRoom,
    luma_from_rgba,
    make_frame,
    rgba_from_luma,
)


def _step(localizer, prev_frame, cur_frame, direction):
    prev = localizer.observe(prev_frame)
    cur = localizer.observe(cur_frame)
    return localizer.update(prev, cur, direction)


def test_observe_builds_view():
    room = SyntheticRoom(width=10, height=9)
    emu = SyntheticEmulator(room, pose=(5, 4))
    loc = localization.Localizer()
    view = loc.observe(emu.observe())
    assert len(view.tiles) == 90
    assert view.context is FrameContext.OVERWORLD
    assert len(view.signature) == 64
    assert view.notes.get("luma") is not None


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_moved_detection(direction):
    room = SyntheticRoom(width=12, height=11)
    emu = SyntheticEmulator(room, pose=(5, 5))
    loc = localization.Localizer()
    before = (loc.pose.x, loc.pose.y)
    prev_frame = emu.observe()
    cur_frame = emu.press(direction)
    est = _step(loc, prev_frame, cur_frame, direction)
    assert est.result is MotionResult.MOVED, est.evidence
    assert est.tiles_moved == 1
    assert est.confidence > 0.5
    dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]
    assert (loc.pose.x - before[0], loc.pose.y - before[1]) == (dx, dy)
    assert loc.pose.facing == direction


def test_blocked_same_facing_is_pixel_identical():
    room = SyntheticRoom(width=10, height=9, walls={(4, 4)})
    emu = SyntheticEmulator(room, pose=(5, 4), facing="left")
    loc = localization.Localizer()
    prev = emu.observe()
    cur = emu.press("left")

    def identical(a, b):
        assert np.array_equal(luma_from_rgba(a.rgba), luma_from_rgba(b.rgba))

    identical(prev, cur)
    est = _step(loc, prev, cur, "left")
    assert est.result is MotionResult.BLOCKED
    assert (loc.pose.x, loc.pose.y) == (0, 0)


def test_turn_only_detection():
    room = SyntheticRoom(width=10, height=9, walls={(4, 4)})
    emu = SyntheticEmulator(room, pose=(5, 4), facing="up")
    loc = localization.Localizer()
    prev = emu.observe()
    cur = emu.press("left")  # blocked but facing changes up -> left
    est = _step(loc, prev, cur, "left")
    assert est.result is MotionResult.TURNED_ONLY, est.evidence
    assert est.evidence["inside_player"] > 0
    assert est.evidence["outside_player"] == 0
    assert loc.pose.facing == "left"
    assert (loc.pose.x, loc.pose.y) == (0, 0)


def test_multi_tile_move():
    room = SyntheticRoom(width=14, height=11)
    emu = SyntheticEmulator(room, pose=(7, 5))
    loc = localization.Localizer()
    prev = emu.observe()
    # simulate a held press: render the pose two tiles to the right
    emu.set_pose((9, 5), facing="right")
    cur = emu.observe()
    est = _step(loc, prev, cur, "right")
    assert est.result is MotionResult.MOVED
    assert est.tiles_moved == 2
    assert est.delta == (2, 0)
    assert (loc.pose.x, loc.pose.y) == (2, 0)


def test_npc_animation_does_not_fake_movement():
    room = SyntheticRoom(width=10, height=9, walls={(4, 4)})
    emu = SyntheticEmulator(room, pose=(5, 4), facing="up")
    loc = localization.Localizer()
    prev = emu.observe()
    # same pose/facing but an NPC tile is animated: no background shift
    room._tiles[(1, 1)] = np.roll(room._tiles[(1, 1)], 5, axis=0)
    cur = make_frame(room.render((5, 4), "up"))
    est = _step(loc, prev, cur, "up")
    assert est.result in (MotionResult.BLOCKED, MotionResult.UNKNOWN)
    assert (loc.pose.x, loc.pose.y) == (0, 0)


def test_unknown_evidence_never_moves_coordinates():
    rng = np.random.default_rng(7)
    room = SyntheticRoom(width=10, height=9)
    emu = SyntheticEmulator(room, pose=(5, 4))
    loc = localization.Localizer()
    prev = make_frame(room.render((5, 4), "down"))
    # unrelated random frame: evidence is unusable
    random_luma = rng.integers(0, 4, size=(144, 160), dtype=np.uint8)
    cur = make_frame(rgba_from_luma(random_luma))
    est = _step(loc, prev, cur, "down")
    assert est.result in (MotionResult.UNKNOWN, MotionResult.TRANSITION, MotionResult.INTERRUPTED)
    assert (loc.pose.x, loc.pose.y) == (0, 0)
    if est.result is MotionResult.UNKNOWN:
        assert loc.pose.confidence < 1.0


def test_two_unknowns_invalidate_pose():
    loc = localization.Localizer()
    loc.reset(area_id="a1", cell=(3, 3))
    room = SyntheticRoom(width=10, height=9)
    base = room.render((5, 4), "down")
    for _ in range(2):
        prev = make_frame(base)
        # corrupt ~10 % of the pixels in a non-tile-aligned way: no usable shift
        noisy = base.copy()
        rng = np.random.default_rng(3)
        pixels = rng.integers(0, 144 * 160, size=2200)
        ys, xs = np.unravel_index(pixels, (144, 160))
        noisy[ys, xs, :3] = 255 - noisy[ys, xs, :3]
        cur = make_frame(noisy)
        est = _step(loc, prev, cur, "down")
        assert est.result is MotionResult.UNKNOWN, est.evidence
    assert loc.pose.valid is False
    assert (loc.pose.x, loc.pose.y) == (3, 3)


def _dialog_frame(room: SyntheticRoom) -> Frame:
    """Render the room then overlay a Gen-1-style white text box."""
    rgba = room.render((5, 4), "down").copy()
    rgba[88:96, :, :3] = 0          # dark gap above the box
    rgba[96:136, :, :3] = 255       # white text box interior
    rgba[134:144, :, :3] = 0        # dark bottom border
    rgba[95:97, :, :3] = 0          # top border rows
    return make_frame(rgba)


def test_dialog_interrupts_movement():
    room = SyntheticRoom(width=10, height=9)
    loc = localization.Localizer()
    prev = loc.observe(make_frame(room.render((5, 4), "down")))
    cur = loc.observe(_dialog_frame(room))
    est = loc.update(prev, cur, "down")
    assert est.result is MotionResult.INTERRUPTED
    assert (loc.pose.x, loc.pose.y) == (0, 0)


def test_black_frame_is_transition():
    loc = localization.Localizer()
    room = SyntheticRoom(width=10, height=9)
    prev = loc.observe(make_frame(room.render((5, 4), "down")))
    black = np.zeros((144, 160, 4), dtype=np.uint8)
    black[..., 3] = 255
    cur = loc.observe(make_frame(black))
    est = loc.update(prev, cur, "down")
    assert est.result is MotionResult.TRANSITION
    assert loc.pose.valid is False


def test_area_change_is_transition():
    room_a = SyntheticRoom(width=10, height=9, seed=1)
    room_b = SyntheticRoom(width=10, height=9, seed=99)
    loc = localization.Localizer()
    prev = loc.observe(make_frame(room_a.render((5, 4), "down")))
    cur = loc.observe(make_frame(room_b.render((2, 2), "up")))
    est = loc.update(prev, cur, "down")
    assert est.result is MotionResult.TRANSITION, est.evidence
    assert loc.pose.valid is False


def test_reset_and_set_pose():
    loc = localization.Localizer()
    loc.reset(area_id="area-1", cell=(4, 4), facing="up")
    assert (loc.pose.x, loc.pose.y) == (4, 4)
    assert loc.pose.facing == "up"
    assert loc.area_id == "area-1"
    loc.set_pose(10, -3, confidence=0.8)
    assert (loc.pose.x, loc.pose.y) == (10, -3)
    assert loc.pose.confidence == pytest.approx(0.8)


def test_estimate_motion_convenience_wrapper():
    room = SyntheticRoom(width=10, height=9)
    emu = SyntheticEmulator(room, pose=(5, 4))
    loc = localization.Localizer()
    prev = emu.observe()
    cur = emu.press("up")
    est = loc.estimate_motion(prev, cur, "up")
    assert est.result is MotionResult.MOVED
    assert (loc.pose.x, loc.pose.y) == (0, -1)


def test_update_requires_observed_views():
    loc = localization.Localizer()
    from llm_agent.navigation.types import FrameView

    empty = FrameView(frame_index=0, signature="0", grid_offset=(0, 0))
    with pytest.raises(ValueError):
        loc.update(empty, empty, "up")


def test_invalid_direction_rejected():
    loc = localization.Localizer()
    room = SyntheticRoom(width=10, height=9)
    frame = make_frame(room.render((5, 4), "down"))
    prev = loc.observe(frame)
    with pytest.raises(ValueError):
        loc.update(prev, prev, "sideways")
