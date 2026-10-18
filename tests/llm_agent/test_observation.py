"""Tests for ``llm_agent.observation`` —pure pixel heuristics, no ROM needed."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from llm_agent import observation
from llm_agent.contracts import SCREEN_HEIGHT, SCREEN_WIDTH, Frame, ScreenEvent, sha256_rgba

# --------------------------------------------------------------------------- #
# Synthetic screen builders (independent of the real ROM)
# --------------------------------------------------------------------------- #

def _base(fill: int = 153) -> np.ndarray:
    rgba = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 4), fill, dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


def overworld_rgba() -> np.ndarray:
    """Neutral textured map view: grey floor with dark furniture blocks."""
    rgba = _base(153)
    rgba[16:48, 8:40, :3] = 0
    rgba[64:96, 112:152, :3] = 85
    rgba[112:140, 24:72, :3] = 0
    return rgba


def dialog_rgba() -> np.ndarray:
    """Overworld plus a Gen-1-style bottom text box (white box + black border)."""
    rgba = overworld_rgba()
    rgba[96:142, 4:156, :3] = 0       # border / box area
    rgba[98:140, 6:154, :3] = 255     # white interior
    rgba[104:110, 16:40, :3] = 0      # sparse text marks
    rgba[122:128, 16:52, :3] = 0
    return rgba


def menu_rgba() -> np.ndarray:
    """Right-side menu window (white panel with black border and text)."""
    rgba = overworld_rgba()
    rgba[4:92, 76:156, :3] = 255
    rgba[4:6, 74:158, :3] = 0
    rgba[90:92, 74:158, :3] = 0
    rgba[4:92, 74:76, :3] = 0
    rgba[4:92, 156:158, :3] = 0
    for y in (12, 24, 36, 48, 60, 72):
        rgba[y:y + 2, 84:150, :3] = 0
    return rgba


def battle_rgba() -> np.ndarray:
    """Battle-like view: white background, sprites, full-width bottom text box."""
    rgba = _base(255)
    rgba[80:112, 16:56, :3] = 0       # player back sprite
    rgba[16:56, 112:150, :3] = 85     # enemy sprite
    rgba[110:113, 4:156, :3] = 0      # text-box top border rows
    rgba[139:142, 4:156, :3] = 0      # text-box bottom border rows
    return rgba


def black_rgba() -> np.ndarray:
    rgba = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


def dark_fade_rgba() -> np.ndarray:
    """Fade-out frame: majority of the screen turned black."""
    rgba = _base(153)
    rgba[:, :96, :3] = 0
    return rgba


def near_black_rgba() -> np.ndarray:
    """Dark but not fully black and not classified as a transition fade."""
    rgba = _base(15)
    rgba[:58, :, :3] = 0
    return rgba


def mid_fade_rgba() -> np.ndarray:
    """Mid fade-out: 80 % of the screen black, rest dim grey (not is_black yet)."""
    rgba = _base(85)
    rgba[:115, :, :3] = 0
    return rgba


def map_change_rgba() -> np.ndarray:
    """Completely different bright layout (checkerboard) for a map-change test."""
    rgba = _base(85)
    rgba[::2, ::2, :3] = 255
    rgba[1::2, 1::2, :3] = 255
    return rgba


def frame(rgba: np.ndarray, frame_index: int = 0, now: float = 1000.0) -> Frame:
    return observation.frame_from_rgba(frame_index, rgba, now=now)


# --------------------------------------------------------------------------- #
# frame_from_rgba / Frame basics
# --------------------------------------------------------------------------- #

def test_frame_from_rgba_metadata_and_copy() -> None:
    rgba = overworld_rgba()
    f = observation.frame_from_rgba(7, rgba, now=123.5)
    assert isinstance(f, Frame)
    assert f.frame_index == 7
    assert f.rgba.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 4)
    assert f.rgba.dtype == np.uint8
    assert f.rgba.flags["C_CONTIGUOUS"]
    assert f.sha256 == sha256_rgba(rgba)
    assert f.captured_at == pytest.approx(123.5)
    assert f.is_black is False
    assert 0.0 <= f.luma_mean <= 255.0
    # the frame owns its pixels: mutating the source must not change it
    rgba[..., :3] = 0
    assert f.sha256 == sha256_rgba(f.rgba)
    assert f.rgba[..., 0].max() > 0


def test_frame_from_rgba_black_detection() -> None:
    black = observation.frame_from_rgba(0, black_rgba())
    assert black.is_black is True
    assert black.luma_mean == pytest.approx(0.0)
    almost = black_rgba()
    almost[:SCREEN_HEIGHT - 2, :, :3] = 255  # 2 of 144 rows black -> ~1.4 %
    assert observation.frame_from_rgba(0, almost).is_black is False


def test_frame_from_rgba_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        observation.frame_from_rgba(0, np.zeros((144, 160, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        observation.frame_from_rgba(0, np.zeros((100, 100, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        observation.frame_from_rgba(-1, overworld_rgba())
    with pytest.raises(TypeError):
        observation.frame_from_rgba(0, "not a frame")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# diff_ratio / is_black_frame / perceptual_signature
# --------------------------------------------------------------------------- #

def test_diff_ratio_identical_and_changed() -> None:
    a = overworld_rgba()
    b = a.copy()
    assert observation.diff_ratio(frame(a), a) == pytest.approx(0.0)
    b[:72, :, :3] = 0
    ratio = observation.diff_ratio(frame(a), frame(b))
    assert 0.4 < ratio < 0.6
    # a single-pixel change is tiny but non-zero
    c = a.copy()
    c[0, 0, 0] = 0 if c[0, 0, 0] != 0 else 255
    assert 0.0 < observation.diff_ratio(a, c) < 0.01
    with pytest.raises(ValueError):
        observation.diff_ratio(a, np.zeros((10, 10, 4), dtype=np.uint8))


def test_is_black_frame_threshold() -> None:
    assert observation.is_black_frame(frame(black_rgba())) is True
    assert observation.is_black_frame(frame(overworld_rgba())) is False
    mostly = black_rgba()
    mostly[0:4, :, :3] = 255  # ~2.8 % non-black -> not black
    assert observation.is_black_frame(frame(mostly)) is False


def test_perceptual_signature_stability_and_length() -> None:
    first = observation.perceptual_signature(frame(overworld_rgba()))
    second = observation.perceptual_signature(frame(overworld_rgba()))
    assert first == second
    assert len(first) == 64
    assert all(char in "0123456789abcdef" for char in first)
    different = observation.perceptual_signature(frame(dialog_rgba()))
    assert first != different


# --------------------------------------------------------------------------- #
# upscale_rgba / encode_for_api / save_png
# --------------------------------------------------------------------------- #

def test_upscale_rgba_nearest_neighbour() -> None:
    rgba = overworld_rgba()
    upscaled = observation.upscale_rgba(rgba, 4)
    assert upscaled.shape == (SCREEN_HEIGHT * 4, SCREEN_WIDTH * 4, 4)
    assert upscaled.dtype == np.uint8
    assert np.array_equal(upscaled[0:4, 0:4], np.repeat(np.repeat(rgba[0:1, 0:1], 4, axis=0), 4, axis=1))
    identity = observation.upscale_rgba(rgba, 1)
    assert np.array_equal(identity, rgba) and identity is not rgba
    for bad in (0, 9, -1, 2.5, True):
        with pytest.raises(ValueError):
            observation.upscale_rgba(rgba, bad)  # type: ignore[arg-type]


def test_encode_for_api_png_is_upscaled_and_lossless() -> None:
    f = frame(overworld_rgba())
    payload, mime = observation.encode_for_api(f, scale=4, image_format="png")
    assert mime == "image/png" and len(payload) > 0
    image = Image.open(io.BytesIO(payload))
    assert image.size == (SCREEN_WIDTH * 4, SCREEN_HEIGHT * 4)
    decoded = np.asarray(image.convert("RGBA"))
    assert np.array_equal(decoded[0:4, 0:4], observation.upscale_rgba(f.rgba, 4)[0:4, 0:4])
    # deterministic encoding
    again, _ = observation.encode_for_api(f, scale=4, image_format="png")
    assert again == payload


def test_encode_for_api_jpeg_and_validation() -> None:
    f = frame(battle_rgba())
    payload, mime = observation.encode_for_api(
        f, scale=2, image_format="jpeg", jpeg_quality=85
    )
    assert mime == "image/jpeg" and len(payload) > 0
    image = Image.open(io.BytesIO(payload))
    assert image.size == (SCREEN_WIDTH * 2, SCREEN_HEIGHT * 2)
    assert image.mode == "RGB"
    with pytest.raises(ValueError):
        observation.encode_for_api(f, scale=4, image_format="gif")
    with pytest.raises(ValueError):
        observation.encode_for_api(f, scale=2, image_format="jpeg", jpeg_quality=0)
    with pytest.raises(ValueError):
        observation.encode_for_api(f, scale=0, image_format="png")


def test_save_png_is_atomic_and_valid(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "screen.png"
    written = observation.save_png(frame(overworld_rgba()), target)
    assert written == target
    assert target.is_file() and target.stat().st_size > 0
    image = Image.open(target)
    assert image.size == (SCREEN_WIDTH, SCREEN_HEIGHT)
    decoded = np.asarray(image.convert("RGBA"))
    assert np.array_equal(decoded, overworld_rgba())
    # no temporary files left behind
    assert not list(target.parent.glob("*.tmp"))
    # overwrite works
    observation.save_png(frame(black_rgba()), target)
    assert np.asarray(Image.open(target).convert("RGBA"))[..., 0].max() == 0


# --------------------------------------------------------------------------- #
# Screen classification heuristics (synthetic layouts)
# --------------------------------------------------------------------------- #

def test_dialog_heuristic() -> None:
    analysis = observation.analyze(frame(dialog_rgba()), frame(overworld_rgba()))
    assert analysis.event == ScreenEvent.DIALOG
    assert analysis.stats["border_rows_bottom"] >= 2
    assert analysis.stats["band_white_ratio"] >= 0.45
    assert analysis.summary and "dialog" in analysis.summary


def test_menu_heuristic() -> None:
    analysis = observation.analyze(frame(menu_rgba()), frame(overworld_rgba()))
    assert analysis.event == ScreenEvent.MENU
    assert analysis.stats["right_white_ratio"] >= 0.40
    assert analysis.stats["right_black_ratio"] >= 0.10
    assert analysis.stats["band_white_ratio"] < 0.45


def test_battle_heuristic_wins_over_dialog() -> None:
    analysis = observation.analyze(frame(battle_rgba()), frame(overworld_rgba()))
    assert analysis.event == ScreenEvent.BATTLE
    assert analysis.stats["battle_score"] == pytest.approx(1.0)
    assert analysis.stats["white_ratio"] >= 0.72
    assert analysis.stats["luma_mean"] >= 175


def test_black_frame_is_transition() -> None:
    assert observation.screen_event(frame(black_rgba()), None) == ScreenEvent.TRANSITION
    assert observation.is_black_frame(frame(black_rgba())) is True


def test_map_change_heuristic() -> None:
    analysis = observation.analyze(frame(map_change_rgba()), frame(overworld_rgba()))
    assert analysis.event == ScreenEvent.MAP_CHANGE
    assert analysis.changed is True
    assert analysis.changed_ratio >= 0.55
    assert analysis.stats["prev_black"] is False


def test_fade_out_and_fade_in_are_transitions() -> None:
    fade = observation.analyze(frame(dark_fade_rgba()), frame(_base(153)))
    assert fade.event == ScreenEvent.TRANSITION
    fade_in = observation.analyze(frame(overworld_rgba()), frame(black_rgba()))
    assert fade_in.event == ScreenEvent.TRANSITION
    # mid fade-out (mostly black, not yet >= 98 % black) is still a transition
    mid = observation.analyze(frame(mid_fade_rgba()), frame(_base(153)))
    assert mid.event == ScreenEvent.TRANSITION
    assert observation.screen_event(frame(mid_fade_rgba()), None) == ScreenEvent.TRANSITION


def test_unknown_event_for_dark_unusual_change() -> None:
    analysis = observation.analyze(frame(near_black_rgba()), frame(_base(153)))
    assert analysis.event == ScreenEvent.UNKNOWN
    assert analysis.luma_mean < 20


def test_plain_overworld_is_none_and_unchanged() -> None:
    current = frame(overworld_rgba())
    analysis = observation.analyze(current, current)
    assert analysis.event == ScreenEvent.NONE
    assert analysis.changed is False
    assert analysis.changed_ratio == pytest.approx(0.0)
    assert analysis.stats["dialog_score"] == pytest.approx(0.0)
    assert analysis.stats["battle_score"] == pytest.approx(0.0)


def test_analyze_without_previous_only_uses_layout() -> None:
    assert observation.analyze(frame(dialog_rgba()), None).event == ScreenEvent.DIALOG
    assert observation.analyze(frame(battle_rgba()), None).event == ScreenEvent.BATTLE
    assert observation.analyze(frame(menu_rgba()), None).event == ScreenEvent.MENU
    plain = observation.analyze(frame(overworld_rgba()), None)
    assert plain.event == ScreenEvent.NONE
    assert plain.changed is False and plain.changed_ratio == 0.0


def test_screen_event_accepts_explicit_changed_ratio() -> None:
    event = observation.screen_event(
        frame(overworld_rgba()), frame(overworld_rgba()), changed_ratio=0.9
    )
    assert event in (ScreenEvent.MAP_CHANGE, ScreenEvent.UNKNOWN)


# --------------------------------------------------------------------------- #
# ScreenAnalysis / text output / performance
# --------------------------------------------------------------------------- #

def test_screen_analysis_fields_and_serialization() -> None:
    analysis = observation.analyze(frame(dialog_rgba()), frame(overworld_rgba()))
    for field in ("event", "changed", "changed_ratio", "signature", "is_black",
                  "luma_mean", "summary", "stats"):
        assert hasattr(analysis, field)
    assert isinstance(analysis.event, ScreenEvent)
    assert isinstance(analysis.changed, bool)
    assert isinstance(analysis.signature, str) and len(analysis.signature) == 64
    payload = analysis.to_dict()
    json.dumps(payload)  # must be JSON serializable for audit logs
    for key in ("luma_mean", "white_ratio", "black_ratio", "dialog_score",
                "menu_score", "battle_score"):
        assert key in analysis.stats


def test_describe_screen_and_analyze_and_describe() -> None:
    current, previous = frame(dialog_rgba()), frame(overworld_rgba())
    text = observation.describe_screen(current, None)
    assert "frame" in text.lower()
    assert 1 <= len(text.splitlines()) <= 3
    analysis = observation.analyze_and_describe(current, previous)
    assert analysis.event == ScreenEvent.DIALOG
    assert analysis.summary == observation.describe_screen(current, analysis)
    assert "dialog" in analysis.summary.lower()
    assert len(analysis.summary.splitlines()) <= 3
    black_text = observation.describe_screen(frame(black_rgba()), None)
    assert "black" in black_text.lower()


def test_heuristics_are_fast() -> None:
    frames = [
        frame(overworld_rgba(), 0),
        frame(dialog_rgba(), 1),
        frame(battle_rgba(), 2),
        frame(menu_rgba(), 3),
    ]
    start = time.perf_counter()
    for index in range(40):
        current = frames[index % len(frames)]
        previous = frames[(index + 1) % len(frames)]
        observation.analyze(current, previous)
    elapsed = time.perf_counter() - start
    average_ms = elapsed / 40.0 * 1000.0
    assert average_ms < 10.0, f"analyze() too slow: {average_ms:.2f} ms/frame"


def test_make_frame_fixture_compatibility(make_frame, sample_rgba, black_rgba) -> None:
    """The shared conftest factories must interoperate with the observation API."""
    shared = make_frame(sample_rgba, frame_index=3)
    assert isinstance(shared, Frame)
    analysis = observation.analyze(
        observation.frame_from_rgba(4, observation.upscale_rgba(shared.rgba, 1))
    )
    assert isinstance(analysis, observation.ScreenAnalysis)
    assert observation.is_black_frame(make_frame(black_rgba)) is True

