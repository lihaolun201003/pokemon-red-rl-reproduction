"""Unit tests for the pure pixel/tile primitives in ``navigation.vision``."""

from __future__ import annotations

import numpy as np
import pytest

from llm_agent.navigation import vision
from llm_agent.navigation.types import PLAYER_BOX

from tests.navigation.conftest import SyntheticRoom, luma_from_rgba, make_frame


def _render(room: SyntheticRoom, pose=(5, 4), facing="down"):
    return room.render(pose, facing)


def test_luma_roundtrip_and_shape():
    room = SyntheticRoom(width=10, height=9)
    rgba = _render(room)
    luma = vision.frame_luma(rgba)
    assert luma.shape == (144, 160)
    assert luma.dtype == np.uint8
    # DMG levels only
    assert set(np.unique(luma)).issubset({0, 85, 153, 255})


def test_luma_accepts_frame():
    room = SyntheticRoom(width=10, height=9)
    frame = make_frame(_render(room))
    luma = vision.frame_luma(frame)
    assert luma.shape == (144, 160)


def test_frame_luma_rejects_bad_shape():
    with pytest.raises((ValueError, TypeError)):
        vision.frame_luma(np.zeros((10, 10, 4), dtype=np.uint8))


def test_extract_tiles_hashes_are_stable_and_size_90():
    room = SyntheticRoom(width=10, height=9)
    luma = vision.frame_luma(_render(room))
    tiles = vision.extract_tiles(luma)
    assert len(tiles) == 90
    assert tiles[0].tx == 0 and tiles[0].ty == 0
    assert tiles[-1].tx == 9 and tiles[-1].ty == 8
    hashes_a = [tile.tile_hash for tile in tiles]
    tiles_b = vision.extract_tiles(luma.copy())
    assert hashes_a == [tile.tile_hash for tile in tiles_b]
    assert len(set(hashes_a)) > 20  # discriminative on a real-ish room


def test_tile_hash_changes_for_different_tiles():
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.full((16, 16), 3, dtype=np.uint8)
    assert vision.tile_hash(a) != vision.tile_hash(b)
    assert vision.tile_hash(a) == vision.tile_hash(a.copy())


def test_tile_grid_offset_returns_in_range():
    room = SyntheticRoom(width=10, height=9)
    luma = vision.frame_luma(_render(room))
    ox, oy = vision.tile_grid_offset(luma)
    assert 0 <= ox < 16
    assert 0 <= oy < 16


@pytest.mark.parametrize("direction,expected_shift", [
    ("up", (0, 1)), ("down", (0, -1)), ("left", (1, 0)), ("right", (-1, 0)),
])
def test_best_shift_matches_movement_inverse(direction, expected_shift):
    room = SyntheticRoom(width=14, height=12)
    base = (7, 6)
    before = vision.frame_luma(_render(room, base, direction))
    dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]
    after = vision.frame_luma(_render(room, (base[0] + dx, base[1] + dy), direction))
    result = vision.best_shift(before, after)
    assert result.shift == expected_shift
    assert result.score > 0.95


def test_best_shift_identity_scores_one():
    room = SyntheticRoom(width=10, height=9)
    luma = vision.frame_luma(_render(room))
    result = vision.best_shift(luma, luma.copy())
    assert result.shift == (0, 0)
    assert result.score == pytest.approx(1.0)


def test_match_shift_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        vision.match_shift(np.zeros((144, 160), np.uint8), np.zeros((100, 100), np.uint8), 0, 0)


def test_shift_image_inverts_match_shift():
    room = SyntheticRoom(width=12, height=10)
    luma = vision.frame_luma(_render(room))
    shifted = vision.shift_image(luma, 16, 0)
    # content moved right by one tile: cur(x) == prev(x - 16)
    assert vision.match_shift(luma, shifted, 1, 0) > 0.98


def test_diff_stats_empty_when_identical():
    room = SyntheticRoom(width=10, height=9)
    luma = vision.frame_luma(_render(room))
    stats = vision.diff_stats(luma, luma.copy())
    assert stats["pixels"] == 0
    assert stats["ratio"] == 0.0
    assert stats["bbox"] is None


def test_diff_stats_player_box_and_turn():
    room = SyntheticRoom(width=10, height=9)
    before = vision.frame_luma(_render(room, (5, 4), "up"))
    after = vision.frame_luma(_render(room, (5, 4), "down"))
    stats = vision.diff_stats(before, after)
    assert stats["pixels"] > 0
    assert stats["inside_ratio"] == pytest.approx(1.0)
    # changes are inside the declared player box
    x0, y0, x1, y1 = PLAYER_BOX
    bx0, by0, bx1, by1 = stats["bbox"]
    assert x0 <= bx0 and bx1 <= x1 and y0 <= by0 and by1 <= y1


def test_screen_signature_stable_and_sensitive():
    room = SyntheticRoom(width=10, height=9, seed=3)
    a = vision.frame_luma(_render(room, (5, 4)))
    b = vision.frame_luma(_render(room, (5, 4)))
    other = vision.frame_luma(_render(room, (7, 6)))
    assert vision.screen_signature(a) == vision.screen_signature(b)
    assert vision.screen_signature(a) != vision.screen_signature(other)
    assert len(vision.screen_signature(a)) == 64


def test_same_screen_tolerance():
    room = SyntheticRoom(width=10, height=9)
    a = vision.frame_luma(_render(room))
    b = a.copy()
    b[0, 0] = 255  # a few pixels of noise must not count as a screen change
    assert vision.same_screen(a, b)
    c = vision.frame_luma(_render(room, (6, 4)))
    assert not vision.same_screen(a, c)


def test_is_black():
    assert vision.is_black(np.zeros((144, 160), dtype=np.uint8))
    room = SyntheticRoom(width=10, height=9)
    assert not vision.is_black(vision.frame_luma(_render(room)))


def test_candidate_shifts_deterministic_and_cardinal():
    shifts = vision.candidate_shifts(2)
    assert (0, 0) in shifts
    assert len(shifts) == len(set(shifts))
    assert all(abs(s[0]) + abs(s[1]) <= 2 for s in shifts)
    assert shifts == vision.candidate_shifts(2)


def test_changed_tiles_reports_tiles():
    room = SyntheticRoom(width=10, height=9)
    a = vision.frame_luma(_render(room, (5, 4), "down"))
    b = vision.frame_luma(_render(room, (6, 4), "right"))
    changed = vision.changed_tiles(a, b)
    assert changed
    assert all(0 <= tx < 10 and 0 <= ty < 9 for tx, ty, _ in changed)
