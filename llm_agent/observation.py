"""Pure pixel-only screen analysis for the DeepSeek Pokémon Red agent.

Everything in this module is derived **only** from the 160x144 RGBA framebuffer.
There is no PyBoy import, no RAM read, no map id, no story flag: the agent may see
exactly what a human player sees.

The analysis is a *trigger hint* for the runtime, never ground truth: whenever a
classification matters the model still receives the real screenshot.  Heuristics
are therefore cheap (< 5 ms/frame), deterministic and deliberately conservative.

Pixel facts used below (verified on real frames of this ROM):

* The DMG palette renders exactly 4 grey levels: 0, 85, 153, 255.
* A Gen-1 text box is a white rectangle at the bottom with full-width black
  border rows (``border_rows_bottom``), so it can be told apart from a bright
  overworld path (which has white pixels but no full-width dark border rows).
* The start/menu window is a white panel on the right half with a black border
  and visible "ink" (``right_white_ratio`` high, ``right_black_ratio`` > 0.10),
  while a plain overworld has almost no dark pixels on the right.
* A battle screen is dominated by white (background), has a high mean
  brightness, and draws a full-width bordered text box at the bottom.
* Walking scrolls the map by one tile, which changes roughly 25-40 % of pixels;
  a real map change / battle transition changes >= 55 % (or fades through black).
"""

from __future__ import annotations

import io
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from llm_agent.contracts import (
    SCREEN_CHANNELS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    Frame,
    ScreenEvent,
    sha256_rgba,
)

# --------------------------------------------------------------------------- #
# Thresholds (calibrated against real frames of this repository's ROM/state)
# --------------------------------------------------------------------------- #

#: luma at/below this value counts as "black" (palette darkest level is 0)
BLACK_LUMA: int = 8
#: luma at/above this value counts as "white" (palette brightest level is 255)
WHITE_LUMA: int = 248
#: a frame is "black" when at least this fraction of pixels is black
BLACK_FRACTION: float = 0.98

#: luma weights (BT.601 integer approximation, no float ops on the hot path)
_LUMA_R, _LUMA_G, _LUMA_B = 77, 150, 29

#: bottom text-box band used by the dialog heuristic (inclusive start, exclusive end)
DIALOG_BAND_TOP: int = 88
DIALOG_BAND_BOTTOM: int = 136
#: rows below this line are considered "bottom" for the text-box checks
DIALOG_BOTTOM_ROW: int = 70
#: horizontal pixel span used for the "mostly white row" test (box interior)
DIALOG_X_START: int = 8
DIALOG_X_STOP: int = 152
#: a row is "mostly white" when >= this fraction of its middle pixels are white
WHITE_ROW_FRACTION: float = 0.9
#: minimum number of mostly-white rows in the bottom half to call it a text box
DIALOG_MIN_WHITE_ROWS: int = 12
#: minimum fraction of white pixels in the bottom band
DIALOG_MIN_BAND_WHITE: float = 0.45
#: a "border row" is a row with at least this many black pixels (box frame)
BORDER_ROW_DARK: int = 120
#: minimum number of full-width black border rows in the bottom half
DIALOG_MIN_BORDER_ROWS: int = 2

#: right-half menu-window thresholds
MENU_MIN_RIGHT_WHITE: float = 0.40
MENU_MIN_RIGHT_BLACK: float = 0.10
MENU_MAX_LEFT_WHITE: float = 0.45
MENU_MAX_BAND_WHITE: float = 0.45

#: battle-layout thresholds
BATTLE_MIN_WHITE: float = 0.72
BATTLE_MIN_LUMA: float = 175.0
BATTLE_MIN_BORDER_ROWS: int = 2

#: fraction of changed pixels that cannot be explained by a one-tile scroll
MAP_CHANGE_MIN_DIFF: float = 0.55
#: minimum mean brightness for a "normal" (not faded) frame
NORMAL_LUMA: float = 20.0
#: a fade-in from black changes fewer pixels than a normal map change
FADE_IN_MIN_DIFF: float = 0.25
#: a fade-out grows the black share a lot while changing the screen
FADE_OUT_MIN_DIFF: float = 0.35
FADE_OUT_MIN_BLACK: float = 0.50
FADE_OUT_MAX_PREV_BLACK: float = 0.20

#: event -> short human/LLM phrasing
_EVENT_TEXT: dict[ScreenEvent, str] = {
    ScreenEvent.NONE: "ordinary game view",
    ScreenEvent.DIALOG: "a text box / dialogue",
    ScreenEvent.MENU: "an open menu",
    ScreenEvent.BATTLE: "a battle scene",
    ScreenEvent.MAP_CHANGE: "a new area (screen changed completely)",
    ScreenEvent.TRANSITION: "a screen transition / fade",
    ScreenEvent.UNKNOWN: "an unusual screen",
}

_EVENT_HINT: dict[ScreenEvent, str] = {
    ScreenEvent.NONE: "Normal gameplay: decide where to go and act.",
    ScreenEvent.DIALOG: "Press A to advance the text, B to close it; read it carefully before acting.",
    ScreenEvent.MENU: "Use directions to move the cursor, A to confirm, B to cancel.",
    ScreenEvent.BATTLE: "Pick a move/item from the visible battle options; watch HP and PP.",
    ScreenEvent.MAP_CHANGE: "The view changed completely: look around before committing to a direction.",
    ScreenEvent.TRANSITION: "The screen is fading/transitioning: wait a few frames before a new decision.",
    ScreenEvent.UNKNOWN: "The screen is unusual: observe again before acting.",
}


# --------------------------------------------------------------------------- #
# Analysis result
# --------------------------------------------------------------------------- #

@dataclass
class ScreenAnalysis:
    """Cheap pixel-only description of one screen (no hidden game data)."""

    event: ScreenEvent
    changed: bool
    changed_ratio: float
    signature: str
    is_black: bool
    luma_mean: float
    summary: str
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (safe for audit logs)."""
        return {
            "event": self.event.value,
            "changed": self.changed,
            "changed_ratio": round(self.changed_ratio, 4),
            "signature": self.signature,
            "is_black": self.is_black,
            "luma_mean": round(self.luma_mean, 3),
            "summary": self.summary,
            "stats": dict(self.stats),
        }


# --------------------------------------------------------------------------- #
# Low-level pixel helpers
# --------------------------------------------------------------------------- #

def _as_rgba(image: Frame | np.ndarray) -> np.ndarray:
    """Return the (144, 160, 4) uint8 pixel buffer of a Frame or ndarray."""
    rgba = image.rgba if isinstance(image, Frame) else image
    if not isinstance(rgba, np.ndarray):
        raise TypeError(f"expected Frame or numpy array, got {type(rgba)!r}")
    if rgba.shape != (SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_CHANNELS):
        raise ValueError(
            f"expected RGBA frame of shape ({SCREEN_HEIGHT}, {SCREEN_WIDTH}, "
            f"{SCREEN_CHANNELS}), got {rgba.shape}"
        )
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba, 0, 255).astype(np.uint8)
    return rgba


def luma_array(rgba: np.ndarray) -> np.ndarray:
    """Return an (H, W) uint16 luma map (BT.601 integer approximation)."""
    rgb = _as_rgba(rgba)[..., :3].astype(np.uint16)
    return (rgb[..., 0] * _LUMA_R + rgb[..., 1] * _LUMA_G + rgb[..., 2] * _LUMA_B) >> 8


def _luma_mean(luma: np.ndarray) -> float:
    return float(luma.mean(dtype=np.float32))


def _features(rgba: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute every pixel statistic the heuristics use, in a single pass.

    Returns ``(luma, stats)`` where ``stats`` holds JSON-friendly numbers and a
    few confidence scores.  Kept separate from :class:`ScreenAnalysis` so the
    scores can be exposed to the model without hiding the raw ratios.
    """
    luma = luma_array(rgba)
    white = luma >= WHITE_LUMA
    dark = luma <= BLACK_LUMA
    height, width = luma.shape

    white_ratio = float(white.mean())
    black_ratio = float(dark.mean())

    band = luma[DIALOG_BAND_TOP:DIALOG_BAND_BOTTOM, :]
    band_white_ratio = float((band >= WHITE_LUMA).mean())

    middle_white_rows = white[:, DIALOG_X_START:DIALOG_X_STOP].mean(axis=1)
    white_rows_bottom = int((middle_white_rows[DIALOG_BOTTOM_ROW:] >= WHITE_ROW_FRACTION).sum())

    row_dark = dark.sum(axis=1)
    border_rows_bottom = int(
        ((row_dark >= BORDER_ROW_DARK) & (np.arange(height) >= 60)).sum()
    )

    half = width // 2
    right_white_ratio = float(white[:, half:].mean())
    right_black_ratio = float(dark[:, half:].mean())
    left_white_ratio = float(white[:, :half].mean())

    # -- confidence scores (0..1, condition based, deterministic) ---------- #
    dialog_score = (
        0.40 * float(band_white_ratio >= DIALOG_MIN_BAND_WHITE)
        + 0.30 * float(white_rows_bottom >= DIALOG_MIN_WHITE_ROWS)
        + 0.30 * float(border_rows_bottom >= DIALOG_MIN_BORDER_ROWS)
    )
    menu_score = (
        0.35 * float(right_white_ratio >= MENU_MIN_RIGHT_WHITE)
        + 0.25 * float(right_black_ratio >= MENU_MIN_RIGHT_BLACK)
        + 0.20 * float(left_white_ratio <= MENU_MAX_LEFT_WHITE)
        + 0.20 * float(band_white_ratio < MENU_MAX_BAND_WHITE)
    )
    luma_mean = _luma_mean(luma)
    battle_score = (
        0.40 * float(white_ratio >= BATTLE_MIN_WHITE)
        + 0.30 * float(luma_mean >= BATTLE_MIN_LUMA)
        + 0.30 * float(border_rows_bottom >= BATTLE_MIN_BORDER_ROWS)
    )

    stats: dict[str, Any] = {
        "luma_mean": round(luma_mean, 3),
        "white_ratio": round(white_ratio, 4),
        "black_ratio": round(black_ratio, 4),
        "band_white_ratio": round(band_white_ratio, 4),
        "white_rows_bottom": white_rows_bottom,
        "border_rows_bottom": border_rows_bottom,
        "right_white_ratio": round(right_white_ratio, 4),
        "right_black_ratio": round(right_black_ratio, 4),
        "left_white_ratio": round(left_white_ratio, 4),
        "dialog_score": round(dialog_score, 4),
        "menu_score": round(menu_score, 4),
        "battle_score": round(battle_score, 4),
        "prev_black": False,
    }
    return luma, stats


def _is_dialog(stats: dict[str, Any]) -> bool:
    """Bottom white text box with full-width dark border rows."""
    return (
        stats["band_white_ratio"] >= DIALOG_MIN_BAND_WHITE
        and stats["white_rows_bottom"] >= DIALOG_MIN_WHITE_ROWS
        and stats["border_rows_bottom"] >= DIALOG_MIN_BORDER_ROWS
    )


def _is_menu(stats: dict[str, Any]) -> bool:
    """White panel with dark ink/border on the right half (not a text box)."""
    return (
        stats["right_white_ratio"] >= MENU_MIN_RIGHT_WHITE
        and stats["right_black_ratio"] >= MENU_MIN_RIGHT_BLACK
        and stats["left_white_ratio"] <= MENU_MAX_LEFT_WHITE
        and stats["band_white_ratio"] < MENU_MAX_BAND_WHITE
    )


def _is_battle(stats: dict[str, Any]) -> bool:
    """Bright white-dominant screen with a full-width bordered bottom text box."""
    return (
        stats["white_ratio"] >= BATTLE_MIN_WHITE
        and stats["luma_mean"] >= BATTLE_MIN_LUMA
        and stats["border_rows_bottom"] >= BATTLE_MIN_BORDER_ROWS
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def frame_from_rgba(frame_index: int, rgba: np.ndarray, *, now: float | None = None) -> Frame:
    """Build a :class:`~llm_agent.contracts.Frame` from a PyBoy RGBA buffer.

    The pixel buffer is **copied**: PyBoy reuses its screen buffer, and captured
    frames must stay valid after further emulation.
    """
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError(f"frame_index must be a non-negative int, got {frame_index!r}")
    pixels = _as_rgba(rgba)
    copied = np.array(pixels, dtype=np.uint8, copy=True, order="C")
    luma = luma_array(copied)
    dark_fraction = float((luma <= BLACK_LUMA).mean())
    return Frame(
        frame_index=frame_index,
        rgba=copied,
        sha256=sha256_rgba(copied),
        captured_at=time.time() if now is None else float(now),
        is_black=dark_fraction >= BLACK_FRACTION,
        luma_mean=_luma_mean(luma),
    )


def diff_ratio(a: Frame | np.ndarray, b: Frame | np.ndarray) -> float:
    """Fraction of pixels (0..1) that differ between two frames (any channel)."""
    first = _as_rgba(a)
    second = _as_rgba(b)
    if first.shape != second.shape:
        raise ValueError(f"cannot diff frames of different shapes: {first.shape} vs {second.shape}")
    changed = np.any(first != second, axis=2)
    return float(changed.mean())


def perceptual_signature(frame: Frame) -> str:
    """Stable 64-char hex average-hash of the screen (16x16 luma blocks).

    Two screens with the same coarse brightness layout share a signature; a
    different map/room almost always changes it.  Used for loop detection.
    """
    rgba = _as_rgba(frame)
    luma = luma_array(rgba).astype(np.float32)
    blocks = luma.reshape(16, SCREEN_HEIGHT // 16, 16, SCREEN_WIDTH // 16).mean(axis=(1, 3))
    threshold = float(blocks.mean())
    bits = (blocks > threshold).reshape(-1)
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def is_black_frame(frame: Frame | np.ndarray) -> bool:
    """True when >= 98 % of pixels are black (full fade / transition screen)."""
    luma = luma_array(_as_rgba(frame))
    return float((luma <= BLACK_LUMA).mean()) >= BLACK_FRACTION


def upscale_rgba(rgba: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upscale of an RGBA buffer (pixel art must stay crisp)."""
    if not isinstance(scale, int) or isinstance(scale, bool) or not 1 <= scale <= 8:
        raise ValueError(f"scale must be an int in 1..8, got {scale!r}")
    pixels = _as_rgba(rgba)
    if scale == 1:
        return np.array(pixels, dtype=np.uint8, copy=True, order="C")
    upscaled = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
    return np.ascontiguousarray(upscaled, dtype=np.uint8)


def encode_for_api(
    frame: Frame,
    *,
    scale: int,
    image_format: str,
    jpeg_quality: int = 90,
) -> tuple[bytes, str]:
    """Encode a frame as a budget-friendly image for the vision API.

    Returns ``(image_bytes, mime_type)``.  The frame is upscaled with nearest
    neighbour (default scale 4 -> 640x576) so the model can read the pixel font;
    PNG is strongly preferred for the 4-colour Game Boy palette (smaller and
    lossless), JPEG is supported for comparison experiments.
    """
    fmt = str(image_format or "").strip().lower()
    if fmt not in ("png", "jpeg"):
        raise ValueError(f"image_format must be 'png' or 'jpeg', got {image_format!r}")
    quality = int(jpeg_quality)
    if not 1 <= quality <= 100:
        raise ValueError(f"jpeg_quality must be in 1..100, got {jpeg_quality!r}")

    pixels = upscale_rgba(frame.rgba, scale)
    image = Image.fromarray(pixels, "RGBA")
    buffer = io.BytesIO()
    if fmt == "png":
        image.save(buffer, format="PNG", compress_level=6)
        return buffer.getvalue(), "image/png"
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue(), "image/jpeg"


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically (unique temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover - best effort cleanup
            pass
    return path


def save_png(frame: Frame, path: str | Path) -> Path:
    """Atomically save a frame as a native-resolution (160x144) PNG."""
    rgba = _as_rgba(frame)
    image = Image.fromarray(rgba, "RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return _atomic_write_bytes(Path(path), buffer.getvalue())


def _prev_black_ratio(previous: Frame | None) -> float:
    """Black-pixel share of the previous frame (cheap: luma only)."""
    if previous is None:
        return 0.0
    luma = luma_array(_as_rgba(previous.rgba))
    return float((luma <= BLACK_LUMA).mean())


def _classify(
    current: Frame,
    stats: dict[str, Any],
    previous: Frame | None,
    ratio: float,
    previous_black_ratio: float,
) -> ScreenEvent:
    """Turn pixel statistics + frame delta into one coarse screen event."""
    if current.is_black or stats["black_ratio"] >= BLACK_FRACTION:
        return ScreenEvent.TRANSITION
    # Battle before dialog: a battle also draws a bottom text box.
    if _is_battle(stats):
        return ScreenEvent.BATTLE
    if _is_dialog(stats):
        return ScreenEvent.DIALOG
    if _is_menu(stats):
        return ScreenEvent.MENU
    if stats["black_ratio"] >= 0.75 and stats["luma_mean"] < NORMAL_LUMA:
        # Mostly-dark frame (mid fade-out): a transition, not a normal dark area.
        return ScreenEvent.TRANSITION
    if previous is None:
        return ScreenEvent.NONE
    prev_black = previous_black_ratio >= BLACK_FRACTION
    if prev_black and ratio >= FADE_IN_MIN_DIFF:
        # Fade-in from a black screen: the *change* is the transition.
        return ScreenEvent.TRANSITION
    if (
        ratio >= FADE_OUT_MIN_DIFF
        and stats["black_ratio"] >= FADE_OUT_MIN_BLACK
        and previous_black_ratio <= FADE_OUT_MAX_PREV_BLACK
    ):
        return ScreenEvent.TRANSITION
    if ratio >= MAP_CHANGE_MIN_DIFF and current.luma_mean >= NORMAL_LUMA:
        return ScreenEvent.MAP_CHANGE
    if ratio >= MAP_CHANGE_MIN_DIFF:
        return ScreenEvent.UNKNOWN
    return ScreenEvent.NONE


def screen_event(
    current: Frame,
    previous: Frame | None = None,
    changed_ratio: float | None = None,
) -> ScreenEvent:
    """Classify the current screen into a :class:`ScreenEvent` (pixel-only)."""
    if not isinstance(current, Frame):
        raise TypeError(f"current must be a Frame, got {type(current)!r}")
    ratio = (
        float(changed_ratio)
        if changed_ratio is not None
        else (diff_ratio(current, previous) if previous is not None else 0.0)
    )
    _, stats = _features(current.rgba)
    return _classify(current, stats, previous, ratio, _prev_black_ratio(previous))


def _compact_summary(
    event: ScreenEvent, ratio: float, signature: str, luma_mean: float, black: bool
) -> str:
    parts = [
        f"event={event.value}",
        f"changed={ratio:.2f}",
        f"sig={signature[:8]}",
        f"luma={luma_mean:.0f}/255",
    ]
    if black:
        parts.append("BLACK")
    return " ".join(parts)


def analyze(current: Frame, previous: Frame | None = None) -> ScreenAnalysis:
    """Analyze a frame; when ``previous`` is given also classify the change.

    Pure, deterministic and cheap.  Returns :class:`ScreenAnalysis` whose
    ``summary`` is a compact one-line text and whose ``stats`` expose the raw
    pixel ratios plus ``dialog_score`` / ``menu_score`` / ``battle_score``.
    """
    if not isinstance(current, Frame):
        raise TypeError(f"current must be a Frame, got {type(current)!r}")
    if previous is not None and not isinstance(previous, Frame):
        raise TypeError(f"previous must be a Frame or None, got {type(previous)!r}")

    rgba = _as_rgba(current.rgba)
    _, stats = _features(rgba)
    ratio = diff_ratio(current, previous) if previous is not None else 0.0
    previous_black_ratio = _prev_black_ratio(previous)
    if previous is not None:
        stats["prev_black"] = bool(previous.is_black)
    event = _classify(current, stats, previous, ratio, previous_black_ratio)
    signature = perceptual_signature(current)
    changed = previous is not None and ratio > 0.0
    summary = _compact_summary(event, ratio, signature, current.luma_mean, current.is_black)
    return ScreenAnalysis(
        event=event,
        changed=changed,
        changed_ratio=ratio,
        signature=signature,
        is_black=bool(current.is_black),
        luma_mean=float(current.luma_mean),
        summary=summary,
        stats=stats,
    )


def describe_screen(frame: Frame, analysis: ScreenAnalysis | None = None) -> str:
    """Return 1-3 lines of human/LLM-readable text about the screen."""
    analysis = analysis if analysis is not None else analyze(frame, None)
    event_text = _EVENT_TEXT.get(analysis.event, "an unusual screen")
    lines = [
        f"Screen frame {frame.frame_index}: {event_text} "
        f"(signature {analysis.signature[:8]}, brightness {analysis.luma_mean:.0f}/255"
        f"{', fully black' if analysis.is_black else ''})."
    ]
    if analysis.changed:
        lines.append(f"Changed vs the previous frame: {analysis.changed_ratio * 100:.0f}% of pixels.")
    elif analysis.changed_ratio > 0.0:
        lines.append("Only a tiny change versus the previous frame.")
    else:
        lines.append("Pixel-identical to the previous frame (no visible change yet).")
    hint = _EVENT_HINT.get(analysis.event)
    if hint:
        lines.append(hint)
    return "\n".join(lines)


def analyze_and_describe(current: Frame, previous: Frame | None = None) -> ScreenAnalysis:
    """Analyze a frame and replace ``summary`` with the full description text."""
    analysis = analyze(current, previous)
    analysis.summary = describe_screen(current, analysis)
    return analysis
