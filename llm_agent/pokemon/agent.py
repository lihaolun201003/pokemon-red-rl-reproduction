"""Autonomous battle agent: real pixels -> DeepSeek decision -> verified presses.

:class:`PokemonAgent` takes over the emulator whenever a battle starts, asks the
model **every turn** what to do, converts that decision into real button presses
with a deterministic menu driver, and writes the battle lessons to memory and to
the shared world memory.

Hard boundaries (enforced by construction, not by convention):

* It never imports PyBoy and never reads RAM.  All observation is through the
  :class:`~llm_agent.multi_agent.agent_context.ActionPrimitives` wrapper around
  the director-issued :class:`GuardedEmulator`; every press is arbiter-gated and
  therefore raises :class:`~llm_agent.multi_agent.contracts.ControlDeniedError`
  when this agent does not hold the control lease.
* The model chooses **one action per turn** (``battle_action`` tool).  The
  *decision* is always the LLM's; the Python menu driver only executes it
  deterministically (move the cursor to the named option/slot, confirm, verify
  the screen changed).  There is no hard-coded battle script.
* The model may only use the screenshot and its own battle history.  Prompts
  contain no species/move/location knowledge and all "observed facts" are
  validated against a closed set of visible fields (see
  :mod:`llm_agent.pokemon.tools`).
* Every press is verified by a pixel diff; when the expected menu is not
  visible the driver re-observes and returns control instead of pressing
  blindly.

The public surface is the frozen multi-agent interface:
``handles()``, ``handle_battle(...)``, ``on_event(...)``, ``status()``,
``pause()`` / ``resume()`` / ``paused``.

Battle turn flow (one LLM turn)::

    observe until the battle menu is visible (advancing text boxes, waiting out
    animations / fades, detecting the end of the battle)
      -> screenshot + context sent to DeepSeek (next to the static system prompt
         and the model's own battle memory)
      -> exactly one validated battle_action / observe / finish decision
      -> deterministic, per-press-verified menu execution
      -> visible observations / explicit lessons recorded (VERIFIED vs INFERRED)
      -> next turn

Anything that is not observable (hidden enemy HP, hidden moves, stats,
type-chart knowledge) is never accepted or recorded.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from llm_agent import contracts
from llm_agent.budget import usage_cost_usd
from llm_agent.contracts import (
    ActionOutcome,
    ApiUsage,
    Frame,
    ModelReply,
    ScreenEvent,
    ToolCall,
)
from llm_agent.multi_agent.agent_context import (
    ActionPrimitives,
    AgentContext,
    AgentContextConfig,
)
from llm_agent.multi_agent.contracts import (
    AgentMessage,
    AgentRole,
    BudgetLimitError,
    ControlDeniedError,
    ControlOwner,
    EventType,
    KnowledgeEntry,
    Task,
    now_ts,
)
from llm_agent.observation import ScreenAnalysis, analyze, diff_ratio
from llm_agent.pokemon import prompts, tools
from llm_agent.pokemon.memory import PokemonMemory

# --------------------------------------------------------------------------- #
# Battle loop budgets (all conservative; the director can override max_turns)
# --------------------------------------------------------------------------- #

#: Default hard cap on LLM battle turns for one ``handle_battle`` call.
DEFAULT_MAX_TURNS: int = 20
#: How many observations may be spent waiting for the battle screen to appear.
BATTLE_START_MAX_OBSERVATIONS: int = 20
#: Consecutive non-battle observations required before the battle counts as over.
BATTLE_END_CONSECUTIVE: int = 3
#: Bounded wait for the battle command menu after the (long) battle intro:
#: up to ``BATTLE_MENU_WAIT_ATTEMPTS`` waits of ``BATTLE_MENU_WAIT_FRAMES``
#: emulated frames for non-battle transition frames (= 240 frames by default).
BATTLE_MENU_WAIT_ATTEMPTS: int = 8
BATTLE_MENU_WAIT_FRAMES: int = 30
#: Battle intro text waits for the A button in Pokémon Red (verified on real ROM
#: frames), so the intro phase advances it with a bounded number of A presses and
#: a hard emulated-frame budget (12 × (24 action + 8 wait) = 384 ≤ 400 frames).
INTRO_MAX_A_PRESSES: int = 12
INTRO_MAX_FRAMES: int = 520
#: Frames waited for an unknown (animation/transition) battle frame while looking
#: for the command menu, instead of burning an A press on it.
INTRO_IDLE_WAIT_FRAMES: int = 12
#: Frames to let a text page render after an A press before observing again.
TEXT_ADVANCE_WAIT_FRAMES: int = 8
#: Maximum observe/wait/advance steps while settling onto a decision point.
#: Each step is up to ``WAIT_FRAMES`` emulated frames, so this bounds how long a
#: battle animation / message sequence may run before control is returned.
SETTLE_MAX_STEPS: int = 40
#: Frames waited per "let the animation continue" observation.
WAIT_FRAMES: int = 24
#: Maximum A/B presses spent advancing text boxes within one settle pass.
DIALOG_ADVANCE_LIMIT: int = 16
#: Maximum text-box presses after a ``finish`` decision.
POST_BATTLE_RESOLVE_STEPS: int = 8
#: Consecutive invalid model replies tolerated before the battle is failed.
MAX_CONSECUTIVE_PARSE_ERRORS: int = 3
#: Consecutive failed menu-driver executions tolerated before returning control.
MAX_CONSECUTIVE_DRIVER_FAILURES: int = 3
#: Pixel-diff threshold under which a press does not count as a visible change.
PRESS_MIN_CHANGE: float = 0.0002

# --------------------------------------------------------------------------- #
# Visible battle menu layout detection (pixels only)
# --------------------------------------------------------------------------- #
#
# Gen-1 battle screens draw their menus inside the bottom message band.  The
# heuristics below are intentionally simple and conservative (they only have to
# distinguish "main 2x2 option grid" from "vertical move/item/party list" from
# "plain message box"); the model always sees the real screenshot and makes the
# decision.  Thresholds live here so they can be recalibrated against real-ROM
# captures without touching the driver logic.

BATTLE_MENU_MAIN: str = "main"
BATTLE_MENU_MOVE: str = "move"
BATTLE_MENU_TEXT: str = "text"
BATTLE_MENU_UNKNOWN: str = "unknown"

# rows of the bottom text/menu band used for layout detection (exclusive end)
LAYOUT_BAND_TOP: int = 90
LAYOUT_BAND_BOTTOM: int = 135
#: column separating the left (move list) and right (option grid) halves
LAYOUT_SPLIT_X: int = 80
#: luma at/below this value counts as menu/text "ink"
INK_LUMA_MAX: int = 8
#: sample margins that skip the message-box border columns/letters.  The real
#: command menu draws a vertical separator at x≈66 and its cursor arrow at
#: x≈73-78, so the left sample stops at 60 and the right sample starts at 84;
#: the outer box borders (x≈2-6 / x≈154-158) are excluded on both sides.
LAYOUT_LEFT_X0: int = 8
LAYOUT_LEFT_X1: int = 60
LAYOUT_RIGHT_X0: int = 84
LAYOUT_RIGHT_X1: int = 150
#: rows scanned looking for the message-box border that frames the menu content
BORDER_SCAN_TOP: int = 60
BORDER_SCAN_BOTTOM: int = 144
#: a row is a full-width message-box border when it has this many dark pixels
FULL_BORDER_DARK: int = 140
#: a half-row is a half-width box border when it has this many dark pixels
HALF_BORDER_DARK: int = 60
#: main option grid: ink mostly on the right half, two label rows
MAIN_MENU_MIN_RIGHT_INK: float = 0.015
MAIN_MENU_MAX_LEFT_INK: float = 0.012
MAIN_MENU_MIN_RIGHT_LINES: int = 2
#: move/item/party list: several text rows on the left half
MOVE_MENU_MIN_LEFT_INK: float = 0.015
MOVE_MENU_MIN_LEFT_LINES: int = 3
#: a plain message box: a little ink on the left, too few rows for a list
TEXT_BOX_MIN_LEFT_INK: float = 0.005
#: a frame "looks like a battle screen" when it is bright, white-dominant and
#: carries a bottom message box.  This is the battle classifier's signature
#: without the full-width-border requirement: the real battle *menu* frames are
#: white-dominant with a half-width box, and the empty battle text box frames
#: have a box but no ink yet.  Calibrated against the repository's archived real
#: frames: 0 false negatives on 45 real battle frames, 0 false positives on the
#: 550 other archived frames.
BATTLE_HINT_MIN_WHITE: float = 0.72
BATTLE_HINT_MIN_LUMA: float = 175.0
BATTLE_HINT_MIN_BLACK: float = 0.05

#: Main-menu positions as (row, column) in the visible 2x2 option grid.  The
#: game resets the cursor to the top-left option at the start of every turn.
#: ``catch`` opens the same visible item list as ``item`` (the model picks the
#: row); the deterministic driver only knows the menu geometry.
MAIN_MENU_TARGETS: dict[str, tuple[int, int]] = {
    "fight": (0, 0),
    "switch": (0, 1),
    "item": (1, 0),
    "catch": (1, 0),
    "run": (1, 1),
}

#: Action -> outcome inferred when the battle ends right after that action.
_OUTCOME_HINTS: dict[str, str] = {"run": "ran", "catch": "caught"}

#: Statuses this agent may return (frozen by the director contract).
STATUSES: tuple[str, ...] = (
    "battle_ended",
    "interrupted",
    "failed",
    "budget",
    "control_denied",
)

#: Outcomes this agent may return.
OUTCOMES: tuple[str, ...] = ("win", "loss", "caught", "ran", "unknown")


# --------------------------------------------------------------------------- #
# Small config helpers (tolerant of dicts / stubs / missing sections)
# --------------------------------------------------------------------------- #

def _config_value(config: Any, section: str, attribute: str, default: Any) -> Any:
    """Read ``config.<section>.<attribute>`` tolerantly."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        block = config.get(section)
    else:
        block = getattr(config, section, None)
    if block is None:
        return default
    if isinstance(block, Mapping):
        value = block.get(attribute, default)
    else:
        value = getattr(block, attribute, default)
    return default if value is None else value


# --------------------------------------------------------------------------- #
# Pixel-only menu layout detection
# --------------------------------------------------------------------------- #

def _frame_luma(frame: Any) -> np.ndarray | None:
    """Return the (144, 160) luma map of a frame, or ``None`` when unavailable."""
    rgba = getattr(frame, "rgba", None)
    if rgba is None:
        return None
    try:
        pixels = np.asarray(rgba)
        if pixels.ndim != 3 or pixels.shape[0] < LAYOUT_BAND_BOTTOM:
            return None
        rgb = pixels[..., :3].astype(np.uint16)
    except (TypeError, ValueError):
        return None
    return ((rgb[..., 0] * 77 + rgb[..., 1] * 150 + rgb[..., 2] * 29) >> 8).astype(np.uint8)


def _ink_row_groups(row_has_ink: np.ndarray) -> int:
    """Number of consecutive "row contains ink" groups (i.e. text lines)."""
    groups = 0
    previous = False
    for value in row_has_ink.tolist():
        current = bool(value)
        if current and not previous:
            groups += 1
        previous = current
    return groups


def looks_like_battle(frame: Any, analysis: ScreenAnalysis | None = None) -> bool:
    """Pixel-only battle hint that also accepts real battle *menu* frames.

    ``llm_agent.observation.analyze`` classifies a screen as ``BATTLE`` when it
    is white-dominant, bright and shows a full-width message-box border.  In a
    real battle the main option grid / move list only draws a *half-width* box,
    and the empty message box during an animation has no text ink yet, so those
    frames would otherwise be classified as a generic screen.  This helper
    accepts the same white/bright signature as long as a bottom message box is
    actually drawn (rows of border pixels).  It is only used to keep observing /
    deciding inside a battle the director already handed to this agent.

    Calibration: 0 false negatives on the 45 real battle frames archived under
    ``artifacts/`` and 0 false positives on the other 550 archived frames.
    """
    if analysis is not None and analysis.event is ScreenEvent.BATTLE:
        return True
    luma = _frame_luma(frame)
    if luma is None:
        return False
    ink = luma <= INK_LUMA_MAX
    white_ratio = float((luma >= 248).mean())
    black_ratio = float(ink.mean())
    mean = float(analysis.luma_mean) if analysis is not None else float(luma.mean())
    if (
        white_ratio < BATTLE_HINT_MIN_WHITE
        or mean < BATTLE_HINT_MIN_LUMA
        or black_ratio < BATTLE_HINT_MIN_BLACK
    ):
        return False
    return _content_window(ink) is not None


def _content_window(ink: np.ndarray) -> tuple[int, int] | None:
    """Find the vertical interior of the bottom message/menu box from pixels.

    The box border rows are the darkest full/half rows below the upper battle
    UI.  Content (menu labels or text lines) is sampled strictly *between* the
    detected top and bottom borders so border rows never look like menu lines.
    Returns ``(top, bottom)`` row bounds or ``None`` when nothing box-like was
    found (the caller then falls back to a fixed band).
    """
    dark_per_row = ink.sum(axis=1)
    left_dark = ink[:, :LAYOUT_SPLIT_X].sum(axis=1)
    right_dark = ink[:, LAYOUT_SPLIT_X:].sum(axis=1)
    border_rows: list[int] = []
    for y in range(BORDER_SCAN_TOP, min(BORDER_SCAN_BOTTOM, ink.shape[0])):
        if (
            int(dark_per_row[y]) >= FULL_BORDER_DARK
            or int(left_dark[y]) >= HALF_BORDER_DARK
            or int(right_dark[y]) >= HALF_BORDER_DARK
        ):
            border_rows.append(y)
    if not border_rows:
        return None
    opens = [y for y in border_rows if y <= 120]
    top = max(opens) + 1 if opens else min(border_rows) + 1
    closes = [y for y in border_rows if y > top]
    bottom = min(closes) if closes else min(BORDER_SCAN_BOTTOM, ink.shape[0])
    if bottom - top < 8:
        return None
    return int(top), int(bottom)


#: A row only counts as a text/menu line when it has at least this many ink
#: pixels in the sampled half (a border artefact of 1-3 pixels must not create a
#: phantom extra line — this exact artefact made a real battle text frame look
#: like a 4-entry move list). Lines whose rows are separated by at most
#: ``LINE_MERGE_GAP_ROWS`` blank/edge rows belong to the same text line.
LINE_MIN_INK_PIXELS: int = 6
LINE_MERGE_GAP_ROWS: int = 2


def _ink_line_count(
    half_ink: np.ndarray,
    *,
    min_pixels: int = LINE_MIN_INK_PIXELS,
    merge_gap: int = LINE_MERGE_GAP_ROWS,
) -> int:
    """Number of text-line groups in one half, ignoring border-row artefacts."""
    per_row = half_ink.sum(axis=1)
    minimum = max(1, int(min_pixels))
    allowed_gap = max(0, int(merge_gap))
    lines = 0
    gap = allowed_gap + 1
    for count in per_row.tolist():
        if int(count) >= minimum:
            if gap > allowed_gap:
                lines += 1
            gap = 0
        else:
            gap += 1
    return lines


def battle_menu_layout(frame: Any) -> str:
    """Classify the visible bottom menu area of a battle screen from pixels.

    Returns one of :data:`BATTLE_MENU_MAIN`, :data:`BATTLE_MENU_MOVE`,
    :data:`BATTLE_MENU_TEXT` or :data:`BATTLE_MENU_UNKNOWN`.  Only the *interior*
    of the bottom message/menu box is sampled (border rows, border columns and
    the upper status/HUD area are excluded), which is what makes the same
    heuristic work for the real message-box frames and for the battle menus.

    This is a conservative layout hint: an ``unknown`` result makes the driver
    re-observe instead of pressing anything.
    """
    luma = _frame_luma(frame)
    if luma is None:
        return BATTLE_MENU_UNKNOWN
    ink = luma <= INK_LUMA_MAX
    window = _content_window(ink)
    if window is None:
        top, bottom = LAYOUT_BAND_TOP, LAYOUT_BAND_BOTTOM
    else:
        top, bottom = window
    content = ink[top:bottom, :]
    if not bool(content.any()):
        return BATTLE_MENU_UNKNOWN
    left = content[:, LAYOUT_LEFT_X0:LAYOUT_LEFT_X1]
    right = content[:, LAYOUT_RIGHT_X0:LAYOUT_RIGHT_X1]
    if left.size == 0 or right.size == 0:
        return BATTLE_MENU_UNKNOWN
    left_ratio = float(left.mean())
    right_ratio = float(right.mean())
    left_lines = _ink_line_count(left)
    right_lines = _ink_line_count(right)
    if (
        right_ratio >= MAIN_MENU_MIN_RIGHT_INK
        and left_ratio <= MAIN_MENU_MAX_LEFT_INK
        and left_lines <= 1
        and right_lines >= MAIN_MENU_MIN_RIGHT_LINES
    ):
        return BATTLE_MENU_MAIN
    if left_ratio >= MOVE_MENU_MIN_LEFT_INK and left_lines >= MOVE_MENU_MIN_LEFT_LINES:
        return BATTLE_MENU_MOVE
    if left_ratio >= TEXT_BOX_MIN_LEFT_INK:
        return BATTLE_MENU_TEXT
    return BATTLE_MENU_UNKNOWN


# --------------------------------------------------------------------------- #
# Deterministic menu driver
# --------------------------------------------------------------------------- #

class BattleMenuDriver:
    """Turn one validated :class:`~llm_agent.pokemon.tools.BattleDecision` into presses.

    The decision always comes from the model; this class only performs the
    deterministic UI navigation (cursor moves + confirm) and verifies the result
    of **every** press with a pixel diff.  If the expected menu is not visible it
    re-observes once and then returns ``verified=False`` with **no further
    presses** instead of pressing blindly — the caller then decides (usually by
    asking the model again with a fresh observation).

    The game resets the battle cursor to the top-left option at the start of each
    turn; the driver relies on that UI mechanic and verifies each intermediate
    screen change, so a wrong assumption is detected (``verified=False``) instead
    of silently confirming the wrong option.

    Real-ROM calibration note: the menu layout thresholds and the exact number of
    confirmation presses for the party/switch flow are calibrated against
    synthetic frames in the unit tests and must be re-checked on real captures
    during the acceptance run.
    """

    def __init__(
        self,
        *,
        wait_frames: int = WAIT_FRAMES,
        min_press_change: float = PRESS_MIN_CHANGE,
        layout_attempts: int = 3,
    ) -> None:
        self.wait_frames = max(1, int(wait_frames))
        self.min_press_change = max(0.0, float(min_press_change))
        self.layout_attempts = max(1, int(layout_attempts))
        self.log: list[dict[str, Any]] = []

    # -- public entry point ---------------------------------------------- #
    def reset(self) -> None:
        """Forget the previous execution log."""
        self.log = []

    def execute(
        self,
        decision: tools.BattleDecision,
        ap: ActionPrimitives,
        layout: str,
    ) -> dict[str, Any]:
        """Execute ``decision``; returns presses / verified flag / reason."""
        self.reset()
        try:
            return self._execute(decision, ap, layout)
        except ControlDeniedError:
            raise
        except Exception as exc:  # noqa: BLE001 - a driver bug must not crash the battle
            return self._result(False, f"driver error: {type(exc).__name__}: {exc}", layout)

    # -- internals -------------------------------------------------------- #
    def _execute(
        self,
        decision: tools.BattleDecision,
        ap: ActionPrimitives,
        layout: str,
    ) -> dict[str, Any]:
        action = decision.action
        if action == "observe":
            ap.wait(self.wait_frames)
            return self._result(True, "observed a fresh screen without pressing", layout)
        if action == "finish":
            return self._result(True, "finish requested; no presses made", layout)

        if layout == BATTLE_MENU_MOVE and action == "fight":
            ok = self._choose_move_slot(ap, decision.move_slot or 1)
            return self._result(
                ok,
                "move slot confirmed" if ok else "could not confirm the move slot",
                BATTLE_MENU_MOVE,
            )
        if layout == BATTLE_MENU_MOVE:
            if not self._press(ap, "b", "cancel the move list"):
                return self._result(False, "B did not visibly cancel the move list", layout)
            layout = self._current_layout(ap)
            if layout != BATTLE_MENU_MAIN:
                return self._result(
                    False,
                    f"expected the main battle menu after cancelling, saw {layout}",
                    layout,
                )
        # Never trust the caller-provided layout blindly: re-check what is on
        # screen before pressing anything (the last frame is cheap to classify).
        observed_layout = self._current_layout(ap)
        if observed_layout != BATTLE_MENU_MAIN:
            observed_layout = self._observe_layout(ap)
        if observed_layout != BATTLE_MENU_MAIN:
            return self._result(
                False,
                f"no visible main battle menu at the decision point "
                f"(layout={observed_layout}); re-observed without pressing",
                observed_layout,
            )
        layout = observed_layout

        target = MAIN_MENU_TARGETS.get(action)
        if target is None:
            return self._result(False, f"unsupported battle action {action!r}", layout)
        if not self._move_main_cursor(ap, target):
            return self._result(False, "could not move the option cursor", layout)
        if not self._press(ap, "a", f"confirm the visible {action} option"):
            return self._result(False, "A did not visibly change the screen", layout)
        if action == "run":
            return self._result(True, "run option selected", layout)

        layout_after = self._await_layout(
            ap, {BATTLE_MENU_MOVE, BATTLE_MENU_TEXT}, attempts=self.layout_attempts
        )
        if action == "fight":
            if layout_after == BATTLE_MENU_MOVE:
                ok = self._choose_move_slot(ap, decision.move_slot or 1)
                return self._result(
                    ok,
                    "move selected" if ok else "move slot was not confirmed",
                    BATTLE_MENU_MOVE,
                )
            if layout_after == BATTLE_MENU_TEXT:
                # A single-move shortcut or an immediate message: the action was
                # accepted by the game even without an explicit move list.
                return self._result(True, "move accepted without a move list", layout_after)
            return self._result(
                False,
                f"expected the move list or a message after FIGHT, saw {layout_after}",
                layout_after,
            )
        if action in ("catch", "item"):
            slot_name = "catch" if action == "catch" else "item"
            if layout_after not in (BATTLE_MENU_MOVE, BATTLE_MENU_TEXT):
                return self._result(
                    False, f"expected the visible item list, saw {layout_after}", layout_after
                )
            slot = decision.item_slot or 1
            if not self._move_down(ap, slot - 1, f"{slot_name} row {slot}"):
                return self._result(False, "could not move to the item row", layout_after)
            if not self._press(ap, "a", f"use {slot_name} row {slot}"):
                return self._result(False, f"A did not visibly use the {slot_name}", layout_after)
            return self._result(True, f"{slot_name} row {slot} used", layout_after)
        if action == "switch":
            if layout_after not in (BATTLE_MENU_MOVE, BATTLE_MENU_TEXT):
                return self._result(
                    False, f"expected the visible party list, saw {layout_after}", layout_after
                )
            slot = decision.party_slot or 1
            if not self._move_down(ap, slot - 1, f"party row {slot}"):
                return self._result(False, "could not move to the party row", layout_after)
            if not self._press(ap, "a", f"choose party row {slot}"):
                return self._result(False, "A did not visibly choose the party row", layout_after)
            if not self._press(ap, "a", "confirm the switch"):
                return self._result(False, "switch confirmation changed nothing", layout_after)
            return self._result(True, f"party row {slot} chosen", layout_after)
        return self._result(False, f"unsupported battle action {action!r}", layout)

    # -- press / layout primitives --------------------------------------- #
    def _press(self, ap: ActionPrimitives, button: str, note: str = "") -> bool:
        """Press one button and verify the screen changed (pixel diff)."""
        before = ap.last_frame
        if before is None:
            before = ap.observe()
        frame = ap.press(str(button), note=note)
        try:
            ratio = float(diff_ratio(before, frame))
        except (TypeError, ValueError):
            ratio = 0.0
        record = {
            "button": str(button),
            "note": str(note),
            "changed_ratio": round(ratio, 5),
            "verified": ratio >= self.min_press_change,
            "frame": int(getattr(frame, "frame_index", 0) or 0),
        }
        self.log.append(record)
        return bool(record["verified"])

    def _current_layout(self, ap: ActionPrimitives) -> str:
        frame = ap.last_frame
        if frame is None:
            frame = ap.observe()
        return battle_menu_layout(frame)

    def _observe_layout(self, ap: ActionPrimitives) -> str:
        return battle_menu_layout(ap.observe())

    def _await_layout(
        self,
        ap: ActionPrimitives,
        expected: set[str],
        *,
        attempts: int = 3,
    ) -> str:
        """Wait (bounded) for one of the expected visible layouts."""
        layout = BATTLE_MENU_UNKNOWN
        for attempt in range(max(1, int(attempts))):
            layout = self._current_layout(ap)
            if layout in expected:
                return layout
            if attempt + 1 < attempts:
                ap.wait(self.wait_frames)
        return layout

    def _move_main_cursor(self, ap: ActionPrimitives, target: tuple[int, int]) -> bool:
        """Move the 2x2 option cursor from the default (0, 0) to ``target``."""
        row, column = max(0, int(target[0])), max(0, int(target[1]))
        vertical = "down" if row > 0 else "up"
        horizontal = "right" if column > 0 else "left"
        for index in range(row):
            if not self._press(ap, vertical, f"cursor row {row} ({index + 1}/{row})"):
                return False
        for index in range(column):
            if not self._press(ap, horizontal, f"cursor column {column} ({index + 1}/{column})"):
                return False
        return True

    def _move_down(self, ap: ActionPrimitives, count: int, note: str) -> bool:
        steps = max(0, int(count))
        for index in range(steps):
            if not self._press(ap, "down", f"{note} ({index + 1}/{steps})"):
                return False
        return True

    def _choose_move_slot(self, ap: ActionPrimitives, slot: int) -> bool:
        slot_value = max(1, min(tools.MAX_MOVE_SLOTS, int(slot or 1)))
        if not self._move_down(ap, slot_value - 1, f"move slot {slot_value}"):
            return False
        return self._press(ap, "a", f"confirm move slot {slot_value}")

    def _result(self, ok: bool, reason: str, layout: str) -> dict[str, Any]:
        presses = [str(entry["button"]) for entry in self.log]
        verified_all = all(bool(entry["verified"]) for entry in self.log) if self.log else True
        return {
            "presses": presses,
            "press_log": [dict(entry) for entry in self.log],
            "verified": bool(ok) and bool(verified_all),
            "reason": str(reason)[:300],
            "layout": str(layout or BATTLE_MENU_UNKNOWN)[:20],
        }


# --------------------------------------------------------------------------- #
# PokemonAgent
# --------------------------------------------------------------------------- #

def _empty_result() -> dict[str, Any]:
    """The frozen return structure of ``handle_battle`` (empty values)."""
    return {
        "status": "failed",
        "outcome": "unknown",
        "turns": 0,
        "api_calls": 0,
        "cost_usd": 0.0,
        "actions": [],
        "lessons": [],
        "evidence": {"screens": [], "frames": []},
        "reason": "",
        # Extra accounting keys consumed by the director (never game data):
        # real button presses executed, denied presses and a failure message.
        "presses": 0,
        "buttons": 0,
        "denied": 0,
        "error": "",
        #: evidence of the battle-intro wait (observations / frames / screens)
        "intro_wait": {},
    }


def _public_lesson(lesson: Mapping[str, Any]) -> dict[str, Any]:
    """Project a stored lesson record onto the frozen ``lessons`` shape."""
    return {
        "category": str(lesson.get("category") or "battle_lesson"),
        "content": str(lesson.get("content") or ""),
        "verification_status": str(lesson.get("verification_status") or "inferred"),
        "confidence": float(lesson.get("confidence") or 0.0),
        "evidence_refs": list(lesson.get("evidence_refs") or []),
        "knowledge_id": str(lesson.get("knowledge_id") or ""),
    }


class PokemonAgent:
    """The autonomous battle agent (one instance per agent run).

    Parameters
    ----------
    router:
        Duck-typed model router: ``model_for(role)``, ``can_call(role)`` and
        ``complete(role, messages, tools=..., max_tokens=..., thinking=...)``
        returning :class:`~llm_agent.contracts.ModelReply`.
    bus:
        Shared event bus (duck-typed ``publish(message)`` / ``emit`` / ``send``);
        optional and always best-effort.
    arbiter:
        :class:`~llm_agent.multi_agent.control_arbiter.ControlArbiter`; the
        actual authorisation happens in the ``GuardedEmulator`` handed to
        :meth:`handle_battle`, this reference is only used for status/reporting.
    shared_memory:
        Shared world memory (duck-typed ``add_knowledge(entry)`` / ``add`` /
        ``record``); every battle lesson is offered to it as a
        :class:`~llm_agent.multi_agent.contracts.KnowledgeEntry`.
    memory:
        Generic :class:`~llm_agent.memory.AgentMemory` instance (own root);
        wrapped by :class:`~llm_agent.pokemon.memory.PokemonMemory`.  ``None``
        creates a private one under the configured memory root.
    config:
        :class:`~llm_agent.config.AgentConfig` (tolerates dicts / partial stubs).
    screen_dir:
        Directory for the saved battle screenshots (evidence).
    clock:
        Injectable monotonic clock (latency/statistics only).
    """

    #: Role name used by the multi-agent protocol.
    role: AgentRole = AgentRole.POKEMON

    def __init__(
        self,
        *,
        router: Any,
        bus: Any,
        arbiter: Any,
        shared_memory: Any,
        memory: Any,
        config: Any,
        screen_dir: str | Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if router is None:
            raise ValueError("PokemonAgent requires a model router")
        self.router = router
        self.bus = bus
        self.arbiter = arbiter
        self.shared_memory = shared_memory
        self.config = config
        self._clock: Callable[[], float] = clock or time.monotonic
        self._screen_dir: Path | None = Path(screen_dir) if screen_dir else None
        if self._screen_dir is not None:
            try:
                self._screen_dir.mkdir(parents=True, exist_ok=True)
            except OSError:  # pragma: no cover - evidence saving is best effort
                pass

        # -- API / image settings ---------------------------------------- #
        # Inside the multi-agent run the (cheaper) agent request shape comes from
        # ``config.multi_agent``; the legacy single-agent run keeps using
        # ``config.api`` unchanged.
        multi_enabled = bool(_config_value(config, "multi_agent", "enabled", False))
        if multi_enabled:
            self._image_scale = max(
                1, min(8, int(_config_value(config, "multi_agent", "image_scale_agent", 3)))
            )
            self._max_tokens = min(
                800, max(200, int(_config_value(config, "multi_agent", "max_tokens_agent", 400)))
            )
            thinking = str(
                _config_value(config, "multi_agent", "thinking_agent", "disabled") or "disabled"
            ).lower()
            self._save_screenshots = bool(
                _config_value(config, "multi_agent", "save_screenshots", True)
            )
        else:
            self._image_scale = max(1, min(8, int(_config_value(config, "api", "image_scale", 3))))
            self._max_tokens = min(800, max(200, int(_config_value(config, "api", "max_tokens", 400))))
            thinking = str(_config_value(config, "api", "thinking", "disabled") or "disabled").lower()
            self._save_screenshots = True
        self._image_format = str(_config_value(config, "api", "image_format", "png") or "png")
        self._jpeg_quality = int(_config_value(config, "api", "image_jpeg_quality", 90))
        self._image_detail = str(_config_value(config, "api", "image_detail", "original") or "original")
        self._thinking = thinking if thinking in ("enabled", "disabled") else "disabled"
        #: emulated frames advanced by one emulator press (for the intro budget)
        self._action_frames = min(
            600, max(2, int(_config_value(config, "emulator", "action_frames", 24)))
        )

        # -- memory / context -------------------------------------------- #
        self._memory: PokemonMemory | None
        self._memory_errors: list[str] = []
        try:
            self._memory = PokemonMemory.create(memory=memory, config=config, clock=self._clock)
        except Exception as exc:  # noqa: BLE001 - memory must not prevent a battle
            self._memory = None
            self._memory_errors.append(f"{type(exc).__name__}: {exc}")
        #: Public battle memory (the director snapshots it and renders it into
        #: its own :class:`~llm_agent.multi_agent.agent_context.AgentContext`).
        self.memory: PokemonMemory | None = self._memory
        self._ctx = AgentContext(
            agent=self.role,
            system_prompt=prompts.SYSTEM_PROMPT,
            tools=tuple(tools.TOOL_SCHEMAS),
            memory=self._memory,
            config=AgentContextConfig(max_memory_chars=2500),
        )
        self._executor = tools.BattleToolExecutor()
        self._driver = BattleMenuDriver(
            wait_frames=WAIT_FRAMES, min_press_change=PRESS_MIN_CHANGE
        )
        #: Public system prompt (the director snapshots it for the audit report).
        self.system_prompt: str = prompts.SYSTEM_PROMPT
        #: One-line L1 recovery strategy notes used by the next battle prompts.
        self._recovery_hints: list[dict[str, Any]] = []

        # -- agent lifecycle state --------------------------------------- #
        self._paused: bool = False
        self._battle_active: bool = False
        self._battle_flagged: bool = False
        self._needs_help: bool = False
        self._task: Task | None = None
        self._task_cancelled: bool = False
        self._last_event: EventType | None = None
        self._event_log: list[dict[str, Any]] = []
        self._publish_errors: list[str] = []
        self._shared_errors: list[str] = []
        self._ap: ActionPrimitives | None = None
        self._state: dict[str, Any] = {
            "last_frame": None,
            "clear_observations": 0,
            "dialog_presses": 0,
        }
        self._stats: dict[str, Any] = {
            "api_calls": 0,
            "cost_usd": 0.0,
            "battles": 0,
            "denied": 0,
            "last_battle_turns": 0,
            "last_status": "",
            "last_outcome": "",
            "last_reason": "",
            "pauses": 0,
        }
        self._reset_run_state(task=None)

    # ------------------------------------------------------------------ #
    # Frozen public surface
    # ------------------------------------------------------------------ #
    def handles(self) -> tuple[str, ...]:
        """Context kinds this agent owns (the director routes by this)."""
        return ("battle",)

    def on_event(self, event: Any) -> None:
        """Handle an inbox event (battle flags, pause/resume, cancellation)."""
        event_type, payload, target = self._coerce_event(event)
        if event_type is None:
            return
        self._last_event = event_type
        self._event_log.append({"event": event_type.value, "at": now_ts()})
        self._event_log = self._event_log[-20:]
        if event_type is EventType.BATTLE_STARTED:
            self._battle_flagged = True
        elif event_type is EventType.BATTLE_ENDED:
            self._battle_flagged = False
        elif event_type is EventType.POKEMON_NEEDS_HELP:
            self._needs_help = True
        elif event_type is EventType.AGENT_PAUSED:
            if self._event_applies_to_me(payload, target):
                self.pause()
        elif event_type is EventType.AGENT_RESUMED:
            if self._event_applies_to_me(payload, target):
                self.resume()
        elif event_type is EventType.TASK_CANCELLED:
            task_id = str(payload.get("task_id") or "")
            if self._task is not None and task_id and task_id == self._task.task_id:
                self._task_cancelled = True
                self.pause()

    def apply_recovery_hint(self, level: str, hint: str) -> None:
        """Append an L1 recovery strategy note for the next battle prompt.

        The director's :class:`~llm_agent.multi_agent.recovery.RecoveryManager`
        calls this after a local strategy correction; the hint is rendered into
        every battle turn prompt until the next battle ends (then it is
        considered used and cleared).
        """
        text = " ".join(str(hint or "").split())[:300]
        if not text:
            return
        self._recovery_hints.append(
            {"level": str(level or "L1")[:20], "hint": text, "at": now_ts()}
        )
        self._recovery_hints = self._recovery_hints[-5:]

    def pause(self) -> None:
        """Pause at the next safe point (between battle turns)."""
        if not self._paused:
            self._paused = True
            self._stats["pauses"] = int(self._stats.get("pauses", 0)) + 1

    def resume(self) -> None:
        """Resume a paused battle agent."""
        self._paused = False

    @property
    def paused(self) -> bool:
        """True while the agent refuses to start/continue a battle."""
        return self._paused

    def status(self) -> dict[str, Any]:
        """Compact, JSON-friendly status for the director/viewer."""
        try:
            memory_stats = self._memory.stats() if self._memory is not None else {}
        except Exception:  # noqa: BLE001 - status must never raise
            memory_stats = {}
        return {
            "agent": self.role.value,
            "role": self.role.value,
            "paused": bool(self._paused),
            "battle_active": bool(self._battle_active),
            "battle_flagged": bool(self._battle_flagged),
            "needs_help": bool(self._needs_help),
            "task_cancelled": bool(self._task_cancelled),
            "turns": int(self._stats.get("last_battle_turns", 0)),
            "api_calls": int(self._stats.get("api_calls", 0)),
            "cost_usd": round(float(self._stats.get("cost_usd", 0.0)), 8),
            "battles_handled": int(self._stats.get("battles", 0)),
            "denied": int(self._stats.get("denied", 0)),
            "last_status": str(self._stats.get("last_status") or ""),
            "last_outcome": str(self._stats.get("last_outcome") or ""),
            "last_reason": str(self._stats.get("last_reason") or ""),
            "last_event": self._last_event.value if self._last_event is not None else "",
            "events": list(self._event_log[-10:]),
            "model": self._model_name(),
            "tools": list(tools.TOOL_NAMES),
            "handles": list(self.handles()),
            "system_prompt_chars": len(self.system_prompt),
            "recovery_hints": [dict(item) for item in self._recovery_hints[-5:]],
            "memory": memory_stats,
            "memory_errors": list(self._memory_errors[-5:]),
            "shared_memory_errors": list(self._shared_errors[-5:]),
            "publish_errors": list(self._publish_errors[-5:]),
            "arbiter_owner": self._arbiter_owner(),
            "screen_dir": str(self._screen_dir) if self._screen_dir is not None else None,
        }

    # ------------------------------------------------------------------ #
    # Battle entry point
    # ------------------------------------------------------------------ #
    def handle_battle(
        self,
        *,
        emulator: Any,
        task: Any | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> dict[str, Any]:
        """Handle one battle autonomously; always returns the frozen dict.

        ``emulator`` is the director-issued ``GuardedEmulator``: observing is
        free, every press is lease-gated.  The method never raises; failures are
        reported through ``status`` (battle_ended / interrupted / failed /
        budget / control_denied).
        """
        self._reset_run_state(task=task)
        try:
            if emulator is None:
                raise ValueError("an emulator (GuardedEmulator) is required")
            self._ap = self._make_primitives(emulator)
            self._begin_task(task)
            turn_budget = self._coerce_max_turns(max_turns)
            start = self._detect_battle_start(self._ap)
            if start["kind"] == "interrupted":
                return self._finalize("interrupted", reason=str(start.get("reason") or "paused"))
            if start["kind"] != "battle":
                return self._finalize(
                    "failed", reason=str(start.get("reason") or "battle not detected")
                )
            frame: Frame = start["frame"]
            analysis: ScreenAnalysis = start["analysis"]
            self._battle_active = True
            self._publish(
                EventType.BATTLE_STARTED,
                {"frame": int(frame.frame_index), "context": analysis.summary},
            )
            # A battle starts with an animation / intro message box: wait for the
            # command menu to actually appear before asking the model for a
            # decision (never report an intro screen as "interrupted").
            startup = self.wait_for_battle_menu(self._ap)
            if startup["kind"] == "interrupted":
                return self._finalize("interrupted", reason=str(startup.get("reason") or "paused"))
            if startup["kind"] == "failed":
                return self._finalize(
                    "failed",
                    reason=str(startup.get("reason") or "battle intro did not reach a menu"),
                )
            if startup["kind"] == "battle_ended":
                self._battle_active = False
                self._publish(
                    EventType.BATTLE_ENDED,
                    {"outcome": "unknown", "turns": 0, "task_id": self._task_id()},
                )
                return self._finalize(
                    "battle_ended",
                    reason="the battle ended before a command menu appeared",
                )
            outcome = self._run_turns(self._ap, turn_budget)
            return self._finish_battle(self._ap, outcome)
        except ControlDeniedError as exc:
            self._run_denied += 1
            self._stats["denied"] = int(self._stats.get("denied", 0)) + 1
            status = "budget" if str(getattr(exc, "owner", "")) == "budget" else "control_denied"
            return self._finalize(status, reason=str(exc) or "control denied")
        except BudgetLimitError as exc:
            return self._finalize("budget", reason=str(exc) or "budget exhausted")
        except contracts.BudgetExceededError as exc:
            return self._finalize("budget", reason=str(exc) or "budget exhausted")
        except contracts.StopRequested as exc:
            return self._finalize("interrupted", reason=str(exc) or "stop requested")
        except Exception as exc:  # noqa: BLE001 - an agent must never crash the director
            self._record_error(exc)
            return self._finalize("failed", reason=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    # Battle turn loop
    # ------------------------------------------------------------------ #
    def _run_turns(self, ap: ActionPrimitives, max_turns: int) -> dict[str, Any]:
        """Run up to ``max_turns`` decision turns; returns a terminal marker."""
        consecutive_parse_errors = 0
        consecutive_driver_failures = 0
        for turn in range(1, max_turns + 1):
            if self._paused:
                return {"kind": "interrupted", "reason": "paused"}
            blocked = self._budget_block_reason()
            if blocked:
                return {"kind": "budget", "reason": blocked}
            allowed, reason = self._can_call_model()
            if not allowed:
                return {"kind": "budget", "reason": reason or "model calls not allowed"}

            point = self._settle_to_decision_point(ap)
            if point["kind"] == "battle_ended":
                return {"kind": "battle_ended", "reason": "screen left the battle"}
            if point["kind"] == "interrupted":
                return {"kind": "interrupted", "reason": str(point.get("reason") or "paused")}
            if point["kind"] != "menu":
                # A bounded settle timeout is a genuine failure to reach a
                # decision point, not a pause/interruption.
                return {
                    "kind": "failed",
                    "reason": str(point.get("reason") or "no visible battle decision point"),
                }

            frame: Frame = point["frame"]
            analysis: ScreenAnalysis = point["analysis"]
            layout: str = point["layout"]
            evidence = ap.screenshot(frame, prefix="battle")
            screen_path = str(evidence.get("path") or "")
            if screen_path and screen_path not in self._run_screens:
                self._run_screens.append(screen_path)

            turn_text = prompts.render_battle_turn_prompt(
                turn=turn,
                max_turns=max_turns,
                screen_summary=analysis.summary,
                menu_layout=layout,
                last_action=self._last_action_summary,
                visible_observations=self._last_visible_observations,
                extra_notes=self._runtime_notes(),
            )
            messages = self._build_messages(turn_text, evidence)
            # Keep the text of this turn in the conversation history (the image
            # itself is sent once and omitted from later history for cost).
            self._ctx.add_user(turn_text)
            reply = self._call_model(messages)
            self._ctx.add_assistant(reply)
            parsed = self._executor.parse_reply(reply)
            self._turns_completed = turn
            refs = [screen_path] if screen_path else []

            if parsed.decision is None:
                consecutive_parse_errors += 1
                self._collect_explicit_lessons(parsed.lessons, refs)
                self._report_tool_results(parsed, executed_summary="")
                self._last_action_summary = (
                    "the previous reply was rejected: " + "; ".join(parsed.errors[:2])
                )
                if consecutive_parse_errors >= MAX_CONSECUTIVE_PARSE_ERRORS:
                    return {"kind": "failed", "reason": "model produced no valid decision"}
                continue
            consecutive_parse_errors = 0
            decision = parsed.decision

            if decision.action == "finish":
                self._collect_decision_lessons(decision, refs)
                self._collect_explicit_lessons(parsed.lessons, refs)
                presses_before = len(self._press_log)
                resolution = self._settle_after_finish(ap)
                resolution_presses = [
                    str(entry.get("button") or "") for entry in self._press_log[presses_before:]
                ]
                self._run_actions.append(
                    {
                        "turn": turn,
                        "context": f"layout={layout}; {analysis.summary}",
                        "decision": decision.to_dict(),
                        "presses": resolution_presses,
                        "verified": bool(resolution.get("cleared")),
                    }
                )
                visible_text = str(decision.visible_observations.get("battle_text") or "")
                if not visible_text:
                    raw_visible = decision.raw_arguments.get("visible_text")
                    if isinstance(raw_visible, str):
                        visible_text = raw_visible
                if resolution.get("cleared"):
                    self._report_tool_results(
                        parsed,
                        executed_summary=(
                            "finish accepted; the screen is now clearly back to the "
                            "overworld (verified by pixels)"
                        ),
                    )
                    return {
                        "kind": "battle_ended",
                        "reason": "model declared the battle finished and the screen cleared",
                        "declared": {
                            "outcome": decision.outcome,
                            "visible_text": visible_text,
                            "reason": decision.reason,
                        },
                    }
                if resolution.get("battle_visible"):
                    # The model claimed the battle was over but a battle menu is
                    # still visible: reject the finish and keep fighting.
                    self._report_tool_results(
                        parsed,
                        executed_summary=(
                            "finish REJECTED: the battle menu is still visible on screen; "
                            "keep fighting (no presses were made for this turn)"
                        ),
                    )
                    self._last_action_summary = (
                        "finish was rejected because the battle menu is still visible; "
                        "choose a real battle action now"
                    )
                    self._pending_outcome_hint = None
                    if self._paused:
                        return {"kind": "interrupted", "reason": "paused"}
                    continue
                self._report_tool_results(
                    parsed,
                    executed_summary=(
                        "finish accepted, but the post-battle screen did not visibly clear"
                    ),
                )
                resolution_reason = str(
                    resolution.get("reason") or "post-battle resolution did not clear"
                )
                return {
                    "kind": "interrupted" if "paused" in resolution_reason.lower() else "failed",
                    "reason": resolution_reason,
                }

            execution = self._driver.execute(decision, ap, layout)
            self._last_action_summary = self._describe_execution(decision, execution)
            if execution.get("press_log"):
                self._press_log.extend(execution["press_log"])
            self._run_actions.append(
                {
                    "turn": turn,
                    "context": f"layout={layout}; {analysis.summary}",
                    "decision": decision.to_dict(),
                    "presses": list(execution.get("presses") or []),
                    "verified": bool(execution.get("verified")),
                }
            )
            if decision.visible_observations:
                self._last_visible_observations = dict(decision.visible_observations)
            self._collect_decision_lessons(decision, refs)
            self._collect_explicit_lessons(parsed.lessons, refs)
            self._report_tool_results(parsed, executed_summary=self._last_action_summary)
            self._pending_outcome_hint = _OUTCOME_HINTS.get(decision.action)

            if not execution.get("verified"):
                consecutive_driver_failures += 1
                if consecutive_driver_failures >= MAX_CONSECUTIVE_DRIVER_FAILURES:
                    return {
                        "kind": "interrupted",
                        "reason": str(execution.get("reason") or "menu driver failed"),
                    }
            else:
                consecutive_driver_failures = 0
            if self._paused:
                return {"kind": "interrupted", "reason": "paused"}
        return {"kind": "budget", "reason": f"battle turn cap reached ({max_turns})"}

    def _finish_battle(self, ap: ActionPrimitives, outcome: Mapping[str, Any]) -> dict[str, Any]:
        """Record the battle outcome and build the frozen return dict."""
        kind = str(outcome.get("kind") or "failed")
        if kind == "interrupted":
            self._battle_active = False
            return self._finalize("interrupted", reason=str(outcome.get("reason") or ""))
        if kind == "budget":
            self._battle_active = False
            return self._finalize("budget", reason=str(outcome.get("reason") or ""))
        if kind == "failed":
            self._battle_active = False
            return self._finalize("failed", reason=str(outcome.get("reason") or ""))
        if kind != "battle_ended":
            self._battle_active = False
            return self._finalize("failed", reason=f"unexpected loop result: {kind}")

        # The turn loop only reports battle_ended after the battle screen *and*
        # its post-battle level-up/victory text cleared (consecutive overworld
        # observations).  If a stray text box reappeared, clear it (bounded)
        # before reporting.
        if self._last_observed_battle_visible(ap):
            self._resolve_after_battle(ap)
        declared = outcome.get("declared") if isinstance(outcome.get("declared"), Mapping) else {}
        resolved_outcome, verification, visible_text = self._resolve_outcome(declared or {}, ap)
        self._record_battle_memory(
            outcome=resolved_outcome,
            verification=verification,
            visible_text=visible_text,
            reason=str(outcome.get("reason") or ""),
        )
        self._publish(
            EventType.BATTLE_ENDED,
            {
                "outcome": resolved_outcome,
                "turns": int(self._turns_completed),
                "task_id": self._task_id(),
            },
        )
        return self._finalize(
            "battle_ended",
            outcome=resolved_outcome,
            reason=str(outcome.get("reason") or "battle ended"),
        )

    # ------------------------------------------------------------------ #
    # Battle start / end detection
    # ------------------------------------------------------------------ #
    def _detect_battle_start(self, ap: ActionPrimitives) -> dict[str, Any]:
        """Observe until the battle screen is visible (bounded)."""
        presses = 0
        for _ in range(BATTLE_START_MAX_OBSERVATIONS):
            if self._paused:
                return {"kind": "interrupted", "reason": "paused"}
            frame, analysis = self._observe(ap)
            if looks_like_battle(frame, analysis):
                return {"kind": "battle", "frame": frame, "analysis": analysis}
            if analysis.event is ScreenEvent.DIALOG and presses < DIALOG_ADVANCE_LIMIT:
                self._press(ap, "a", "advance pre-battle text")
                ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                presses += 1
                continue
            if analysis.event is ScreenEvent.MENU and presses < DIALOG_ADVANCE_LIMIT:
                self._press(ap, "b", "close a pre-battle menu")
                ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                presses += 1
                continue
            ap.wait(WAIT_FRAMES)
        return {"kind": "failed", "reason": "battle screen was not detected (pixels only)"}

    def wait_for_battle_menu(
        self,
        ap: ActionPrimitives,
        *,
        max_a_presses: int = INTRO_MAX_A_PRESSES,
        frame_budget: int = INTRO_MAX_FRAMES,
        wait_frames: int = BATTLE_MENU_WAIT_FRAMES,
    ) -> dict[str, Any]:
        """Wait (bounded) for the FIRST battle command menu after the intro.

        Real Pokémon Red battle intro text **waits for A** (verified on real ROM
        frames: after 120 emulated frames without input the "Wild X appeared!"
        page is still on screen; a send-out page follows; the command menu only
        appears after every intro page has been advanced).  So this phase
        advances the text with a bounded number of A presses, each followed by a
        short ``wait`` and a fresh observation, and it stops immediately when the
        command menu becomes visible.

        Hard budget: ``max_a_presses`` A presses and ``frame_budget`` emulated
        frames (default 12 presses / 400 frames, i.e. 12 × (24 + 8) = 384 frames
        for the standard 24-frame button action).  If the menu does not appear, a
        structured ``failed`` is returned with every observed layout and every
        intro press recorded — the agent never presses blindly beyond the budget,
        never counts intro presses as decision turns, and never reports an intro
        screen as ``interrupted`` (that is reserved for pause/lease issues).
        """
        max_presses = max(0, int(max_a_presses))
        wait_frame_budget = max(1, int(frame_budget))
        frames_per_wait = max(1, int(wait_frames))
        action_frames = max(1, int(self._action_frames))
        text_wait = max(1, int(TEXT_ADVANCE_WAIT_FRAMES))
        self._state["clear_observations"] = 0
        observations = 0
        frames_used = 0
        layouts: list[str] = []
        screens: list[str] = []
        presses: list[dict[str, Any]] = []
        last_frame: Frame | None = None
        last_analysis: ScreenAnalysis | None = None
        battle_visible = False
        while frames_used < wait_frame_budget:
            if self._paused:
                return self._intro_result(
                    "interrupted",
                    reason="paused",
                    observations=observations,
                    waited_frames=frames_used,
                    layouts=layouts,
                    screens=screens,
                    presses=presses,
                )
            frame, analysis = self._observe(ap)
            last_frame, last_analysis = frame, analysis
            observations += 1
            if looks_like_battle(frame, analysis):
                battle_visible = True
                self._state["clear_observations"] = 0
                layout = battle_menu_layout(frame)
                layouts.append(layout)
                if layout in (BATTLE_MENU_MAIN, BATTLE_MENU_MOVE):
                    return self._intro_result(
                        "menu",
                        frame=frame,
                        analysis=analysis,
                        layout=layout,
                        observations=observations,
                        waited_frames=frames_used,
                        layouts=layouts,
                        screens=screens,
                        presses=presses,
                    )
                shot = self._capture_intro_shot(ap, frame)
                if shot:
                    screens.append(shot)
                # A real intro message page (layout text) waits for A; pressing A
                # advances it.  An "unknown" battle frame is an animation/transition
                # frame instead: wait for it to settle rather than burning an A
                # press (the real-ROM intro only needs ~4-6 A presses, so the
                # press budget stays available for the actual text pages).
                if layout == BATTLE_MENU_TEXT:
                    cycle_cost = action_frames + text_wait
                    if (
                        len(presses) < max_presses
                        and frames_used + cycle_cost <= wait_frame_budget
                    ):
                        record = self._press(
                            ap,
                            "a",
                            f"advance battle intro text ({len(presses) + 1}/{max_presses})",
                        )
                        presses.append(
                            {
                                "button": "a",
                                "note": "battle intro text",
                                "frame": int(record.get("frame") or 0),
                                "changed_ratio": float(record.get("changed_ratio") or 0.0),
                                "verified": bool(record.get("verified")),
                            }
                        )
                        ap.wait(text_wait)
                        frames_used += cycle_cost
                        continue
                    frames_used = wait_frame_budget
                    break
                # Unknown battle frame: let the animation finish, then re-observe.
                idle_wait = max(1, int(INTRO_IDLE_WAIT_FRAMES))
                if frames_used + idle_wait > wait_frame_budget:
                    frames_used = wait_frame_budget
                    break
                ap.wait(idle_wait)
                frames_used += idle_wait
                continue
            event = analysis.event
            battle_visible = False
            if event in (ScreenEvent.NONE, ScreenEvent.MAP_CHANGE):
                self._state["clear_observations"] = (
                    int(self._state["clear_observations"]) + 1
                )
                if (
                    self._state["clear_observations"]
                    >= BATTLE_END_CONSECUTIVE
                ):
                    return self._intro_result(
                        "battle_ended",
                        frame=frame,
                        analysis=analysis,
                        observations=observations,
                        waited_frames=frames_used,
                        layouts=layouts,
                        screens=screens,
                        presses=presses,
                    )
            else:
                self._state["clear_observations"] = 0
            if frames_used + frames_per_wait > wait_frame_budget:
                frames_used = wait_frame_budget
                break
            ap.wait(frames_per_wait)
            frames_used += frames_per_wait
        if battle_visible:
            reason = (
                "battle intro did not reach the command menu within the bounded budget "
                f"({len(presses)} A presses / {frames_used} frames; "
                f"layouts={layouts[-8:]})"
            )
        else:
            reason = (
                "battle screen went away before the command menu appeared "
                f"({len(presses)} A presses / {frames_used} frames; layouts={layouts[-8:]})"
            )
        return self._intro_result(
            "failed",
            reason=reason,
            observations=observations,
            waited_frames=frames_used,
            layouts=layouts,
            screens=screens,
            presses=presses,
            budget_exhausted=True,
        )

    def _intro_result(
        self,
        kind: str,
        *,
        reason: str = "",
        observations: int = 0,
        waited_frames: int = 0,
        layouts: list[str] | None = None,
        screens: list[str] | None = None,
        presses: list[dict[str, Any]] | None = None,
        budget_exhausted: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """Store the intro-wait evidence and return the structured result."""
        self._intro_wait = {
            "observations": int(observations),
            "waited_frames": int(waited_frames),
            "layouts": list(layouts or []),
            "screens": list(screens or []),
            "presses": [dict(entry) for entry in (presses or [])],
            "a_presses": len(presses or []),
            "budget_exhausted": bool(budget_exhausted),
            "timed_out": bool(budget_exhausted),
        }
        for screen in self._intro_wait["screens"]:
            if screen and screen not in self._run_screens:
                self._run_screens.append(str(screen))
        payload: dict[str, Any] = {"kind": kind, **extra}
        if reason:
            payload["reason"] = reason
        return payload

    def _capture_intro_shot(self, ap: ActionPrimitives, frame: Frame) -> str:
        """Save one intro observation as evidence (bounded, best effort)."""
        if len(self._intro_screens) >= 10:
            return ""
        try:
            evidence = ap.screenshot(frame, prefix="battle_intro")
        except Exception:  # noqa: BLE001 - evidence saving is best effort
            return ""
        path = str(evidence.get("path") or "")
        if path:
            self._intro_screens.append(path)
        return path

    def _settle_to_decision_point(self, ap: ActionPrimitives) -> dict[str, Any]:
        """Advance text/animations until a visible battle menu is ready.

        The battle only counts as over after :data:`BATTLE_END_CONSECUTIVE`
        consecutive *overworld-like* observations (``NONE``/``MAP_CHANGE``); a
        post-battle level-up/victory text box resets that counter and is
        advanced page by page (its screenshots are kept as outcome evidence), so
        the agent can never report "battle_ended" while the battle or its
        resolution text is still visible.
        """
        self._state["dialog_presses"] = 0
        self._state["clear_observations"] = 0
        for _ in range(SETTLE_MAX_STEPS):
            if self._paused:
                return {"kind": "interrupted", "reason": "paused"}
            frame, analysis = self._observe(ap)
            event = analysis.event
            if looks_like_battle(frame, analysis):
                self._state["clear_observations"] = 0
                layout = battle_menu_layout(frame)
                if layout in (BATTLE_MENU_MAIN, BATTLE_MENU_MOVE):
                    return {
                        "kind": "menu",
                        "frame": frame,
                        "analysis": analysis,
                        "layout": layout,
                    }
                # Battle message text waits for the A button in Pokémon Red; an
                # empty/printing message box shows no layout but is advanced the
                # same bounded way. The first command menu of the battle was
                # already reached before the turn loop, so this is a post-turn
                # settle (the caller's turn counter is not touched).
                if self._state["dialog_presses"] < DIALOG_ADVANCE_LIMIT:
                    self._capture_resolution_shot(ap, frame)
                    self._press(ap, "a", f"advance battle text ({layout})")
                    ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                    self._state["dialog_presses"] += 1
                    continue
                return {
                    "kind": "stuck",
                    "reason": "battle text did not clear within the settle budget",
                }
            if event in (ScreenEvent.NONE, ScreenEvent.MAP_CHANGE):
                self._state["clear_observations"] = int(self._state["clear_observations"]) + 1
                if self._state["clear_observations"] >= BATTLE_END_CONSECUTIVE:
                    return {"kind": "battle_ended", "frame": frame, "analysis": analysis}
                ap.wait(WAIT_FRAMES)
                continue
            if event is ScreenEvent.DIALOG:
                self._state["clear_observations"] = 0
                if self._state["dialog_presses"] >= DIALOG_ADVANCE_LIMIT:
                    return {
                        "kind": "stuck",
                        "reason": "post-battle text did not clear within the settle budget",
                    }
                self._capture_resolution_shot(ap, frame)
                self._press(ap, "a", "advance post-battle text")
                ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                self._state["dialog_presses"] += 1
                continue
            if event is ScreenEvent.MENU:
                self._state["clear_observations"] = 0
                if self._state["dialog_presses"] >= DIALOG_ADVANCE_LIMIT:
                    return {"kind": "stuck", "reason": "a menu blocks the battle exit"}
                self._capture_resolution_shot(ap, frame)
                self._press(ap, "b", "close unexpected menu")
                ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                self._state["dialog_presses"] += 1
                continue
            # TRANSITION / fully black / unknown frame: not clear yet.
            self._state["clear_observations"] = 0
            ap.wait(WAIT_FRAMES)
        return {
            "kind": "stuck",
            "reason": "no battle decision point became visible within the settle budget",
        }

    def _settle_after_finish(self, ap: ActionPrimitives) -> dict[str, Any]:
        """Wait for a model-declared ``finish`` to actually clear the screen.

        A ``finish`` decision is only trusted when the pixels agree: the method
        advances post-battle text (bounded) and requires consecutive
        overworld-like observations.  If the battle menu is still visible the
        decision is rejected (``battle_visible``) so the caller keeps fighting
        instead of returning a false "battle_ended".
        """
        self._state["clear_observations"] = 0
        self._state["dialog_presses"] = 0
        last_frame: Any = None
        last_analysis: Any = None
        for _ in range(SETTLE_MAX_STEPS):
            if self._paused:
                return {"cleared": False, "reason": "paused", "battle_visible": False}
            frame, analysis = self._observe(ap)
            last_frame, last_analysis = frame, analysis
            event = analysis.event
            if event in (ScreenEvent.NONE, ScreenEvent.MAP_CHANGE):
                self._state["clear_observations"] = int(self._state["clear_observations"]) + 1
                if self._state["clear_observations"] >= BATTLE_END_CONSECUTIVE:
                    return {
                        "cleared": True,
                        "battle_visible": False,
                        "frame": frame,
                        "analysis": analysis,
                    }
                ap.wait(WAIT_FRAMES)
                continue
            if looks_like_battle(frame, analysis):
                return {"cleared": False, "battle_visible": True, "frame": frame, "analysis": analysis}
            if event is ScreenEvent.DIALOG:
                self._state["clear_observations"] = 0
                if self._state["dialog_presses"] >= DIALOG_ADVANCE_LIMIT:
                    return {
                        "cleared": False,
                        "battle_visible": False,
                        "reason": "post-battle text did not clear",
                        "frame": frame,
                        "analysis": analysis,
                    }
                self._capture_resolution_shot(ap, frame)
                self._press(ap, "a", "advance post-battle text")
                ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                self._state["dialog_presses"] += 1
                continue
            if event is ScreenEvent.MENU:
                self._state["clear_observations"] = 0
                if self._state["dialog_presses"] >= DIALOG_ADVANCE_LIMIT:
                    return {
                        "cleared": False,
                        "battle_visible": False,
                        "reason": "a menu blocks the battle exit",
                        "frame": frame,
                        "analysis": analysis,
                    }
                self._capture_resolution_shot(ap, frame)
                self._press(ap, "b", "close post-battle menu")
                ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                self._state["dialog_presses"] += 1
                continue
            self._state["clear_observations"] = 0
            ap.wait(WAIT_FRAMES)
        return {
            "cleared": False,
            "battle_visible": False,
            "reason": "post-battle resolution did not clear",
            "frame": last_frame,
            "analysis": last_analysis,
        }

    def _resolve_after_battle(self, ap: ActionPrimitives) -> int:
        """Advance post-battle text boxes (bounded); returns presses made."""
        presses = 0
        for _ in range(POST_BATTLE_RESOLVE_STEPS):
            frame, analysis = self._observe(ap)
            event = analysis.event
            if event in (ScreenEvent.NONE, ScreenEvent.MAP_CHANGE, ScreenEvent.UNKNOWN):
                return presses
            if looks_like_battle(frame, analysis):
                # Still visibly in battle: the model declared finish; stop here.
                return presses
            if event is ScreenEvent.DIALOG:
                if presses >= POST_BATTLE_RESOLVE_STEPS - 1:
                    return presses
                self._capture_resolution_shot(ap, frame)
                self._press(ap, "a", "advance post-battle text")
                ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                presses += 1
                continue
            if event is ScreenEvent.MENU:
                if presses >= POST_BATTLE_RESOLVE_STEPS - 1:
                    return presses
                self._capture_resolution_shot(ap, frame)
                self._press(ap, "b", "close post-battle menu")
                ap.wait(TEXT_ADVANCE_WAIT_FRAMES)
                presses += 1
                continue
            ap.wait(WAIT_FRAMES)
        return presses

    # ------------------------------------------------------------------ #
    # Model calls
    # ------------------------------------------------------------------ #
    def _call_model(self, messages: Sequence[Mapping[str, Any]]) -> ModelReply:
        """One DeepSeek call through the router, with local usage accounting."""
        complete = getattr(self.router, "complete", None)
        if not callable(complete):
            raise RuntimeError("model router has no complete() method")
        reply = complete(
            self.role,
            list(messages),
            tools=list(tools.TOOL_SCHEMAS),
            max_tokens=self._max_tokens,
            thinking=self._thinking,
        )
        self._run_api_calls += 1
        self._stats["api_calls"] = int(self._stats.get("api_calls", 0)) + 1
        usage = getattr(reply, "usage", None) or ApiUsage()
        cost = self._usage_cost(usage)
        self._run_cost_usd += cost
        self._stats["cost_usd"] = float(self._stats.get("cost_usd", 0.0)) + cost
        try:
            self._ctx.record_call(reply, cost_usd=cost)
        except Exception:  # noqa: BLE001 - accounting must never break a battle
            pass
        return reply

    def _can_call_model(self) -> tuple[bool, str]:
        """Ask the router whether another paid call is allowed right now."""
        candidate = getattr(self.router, "can_call", None)
        if not callable(candidate):
            return True, ""
        try:
            verdict = candidate(self.role)
        except Exception:  # noqa: BLE001 - a router hiccup must not fake a budget stop
            return True, ""
        if isinstance(verdict, tuple) and verdict:
            return bool(verdict[0]), str(verdict[1]) if len(verdict) > 1 else ""
        if isinstance(verdict, bool):
            return verdict, ""
        return True, ""

    def _budget_block_reason(self) -> str:
        """Local per-agent call/USD caps from ``config.budget`` (0 = disabled)."""
        max_calls = int(
            _config_value(self.config, "budget", "max_calls_per_run", 0) or 0
        )
        if max_calls > 0 and self._run_api_calls >= max_calls:
            return f"agent call cap reached ({max_calls})"
        max_usd = float(_config_value(self.config, "budget", "max_usd_per_run", 0.0) or 0.0)
        if max_usd > 0 and self._run_cost_usd >= max_usd:
            return f"agent USD cap reached (${max_usd:.4f})"
        return ""

    def _usage_cost(self, usage: ApiUsage) -> float:
        """Estimated USD cost of one call from the reported token usage."""
        api_cfg = getattr(self.config, "api", None) if self.config is not None else None
        if api_cfg is None or not hasattr(api_cfg, "price_input_cache_miss_per_mtok"):
            return 0.0
        try:
            return float(usage_cost_usd(api_cfg, usage, peak_multiplier=1.0))
        except Exception:  # noqa: BLE001 - cost estimation is informational
            return 0.0

    def _build_messages(
        self,
        turn_text: str,
        evidence: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Build the OpenAI-style message list (system + history + screenshot)."""
        try:
            return self._ctx.build_messages(
                user_text=turn_text,
                image_png=evidence.get("bytes"),
                mime=str(evidence.get("mime") or "image/png"),
                detail=self._image_detail,
                include_history=True,
            )
        except Exception:  # noqa: BLE001 - fall back to a minimal correct request
            image_block = self._image_block(evidence)
            return [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": turn_text}, image_block],
                },
            ]

    @staticmethod
    def _image_block(evidence: Mapping[str, Any]) -> dict[str, Any]:
        """Build a vision content block from the captured screenshot bytes."""
        import base64

        raw = evidence.get("bytes") or b""
        mime = str(evidence.get("mime") or "image/png")
        encoded = base64.b64encode(bytes(raw)).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "original"},
        }

    # ------------------------------------------------------------------ #
    # Observation / press helpers
    # ------------------------------------------------------------------ #
    def _make_primitives(self, emulator: Any) -> ActionPrimitives:
        return ActionPrimitives(
            emulator,
            agent=self.role,
            screen_dir=self._screen_dir,
            save_screenshots=self._screen_dir is not None and self._save_screenshots,
            max_actions=0,
            image_scale=self._image_scale,
            image_format=self._image_format,
            jpeg_quality=self._jpeg_quality,
            clock=self._clock,
        )

    def _observe(self, ap: ActionPrimitives) -> tuple[Frame, ScreenAnalysis]:
        """Observe one frame and analyze the change since the previous one."""
        frame = ap.observe()
        analysis = analyze(frame, self._state.get("last_frame"))
        self._state["last_frame"] = frame
        return frame, analysis

    def _press(self, ap: ActionPrimitives, button: str, note: str = "") -> dict[str, Any]:
        """Press one button outside the menu driver (text advancing) and log it."""
        before = ap.last_frame
        if before is None:
            before = ap.observe()
        frame = ap.press(str(button), note=note)
        try:
            ratio = float(diff_ratio(before, frame))
        except (TypeError, ValueError):
            ratio = 0.0
        record = {
            "button": str(button),
            "note": str(note),
            "changed_ratio": round(ratio, 5),
            "verified": ratio >= PRESS_MIN_CHANGE,
            "frame": int(getattr(frame, "frame_index", 0) or 0),
        }
        self._press_log.append(record)
        return record

    # ------------------------------------------------------------------ #
    # Lessons / memory / shared world memory
    # ------------------------------------------------------------------ #
    def _collect_decision_lessons(
        self,
        decision: tools.BattleDecision,
        evidence_refs: Sequence[str],
    ) -> None:
        """Store the visible facts + the labelled hypothesis of one decision."""
        if self._memory is None:
            return
        refs = list(evidence_refs)
        observations = dict(decision.visible_observations or {})
        move_name = str(observations.get("move_name") or "")
        try:
            if move_name:
                lesson = self._memory.record_move_observed(
                    move_name,
                    slot=decision.move_slot,
                    visible_text=str(
                        observations.get("damage_text")
                        or observations.get("battle_text")
                        or ""
                    ),
                    evidence_refs=refs,
                )
                self._collect_lesson(lesson)
            damage_text = str(observations.get("damage_text") or observations.get("hp_text") or "")
            if damage_text:
                lesson = self._memory.record_damage_observation(
                    damage_text,
                    visible_text=str(observations.get("battle_text") or ""),
                    move_name=move_name,
                    evidence_refs=refs,
                )
                self._collect_lesson(lesson)
            for bar_key in ("enemy_hp_bar", "my_hp_bar"):
                bar_text = str(observations.get(bar_key) or "")
                if bar_text:
                    lesson = self._memory.record_damage_observation(
                        f"{bar_key} on screen: {bar_text}",
                        evidence_refs=refs,
                    )
                    self._collect_lesson(lesson)
            battle_text = str(observations.get("battle_text") or "")
            if battle_text:
                lesson = self._memory.record_battle_text(battle_text, evidence_refs=refs)
                self._collect_lesson(lesson)
            if decision.hypothesis:
                lesson = self._memory.record_inference(
                    decision.hypothesis,
                    confidence=0.4,
                    evidence_refs=refs,
                    subject=f"decision:{decision.action}",
                )
                self._collect_lesson(lesson)
        except Exception as exc:  # noqa: BLE001 - memory may never break a battle
            self._memory_errors.append(f"lesson recording failed: {type(exc).__name__}: {exc}")

    def _collect_explicit_lessons(
        self,
        lessons: Sequence[tools.LessonSpec],
        evidence_refs: Sequence[str],
    ) -> None:
        """Store the model's explicit ``record_lesson`` calls with their labels."""
        if self._memory is None:
            return
        for spec in lessons or ():
            try:
                lesson = self._memory.record_lesson_entry(
                    spec.content,
                    verification=spec.verification,
                    category=spec.category,
                    confidence=spec.confidence,
                    evidence_refs=list(spec.evidence_refs) + list(evidence_refs),
                    success=True if spec.verification == "verified" else None,
                )
                self._collect_lesson(lesson)
            except Exception as exc:  # noqa: BLE001
                self._memory_errors.append(
                    f"explicit lesson rejected: {type(exc).__name__}: {exc}"
                )

    def _collect_lesson(self, lesson: Mapping[str, Any] | None) -> None:
        """Add one stored lesson to the run result and offer it to shared memory."""
        if not lesson:
            return
        knowledge_id = str(lesson.get("knowledge_id") or "")
        if knowledge_id and knowledge_id in self._run_lesson_ids:
            return
        if knowledge_id:
            self._run_lesson_ids.add(knowledge_id)
        self._run_lessons.append(_public_lesson(lesson))
        if self._memory is not None and knowledge_id:
            entry = self._memory.knowledge_entry(knowledge_id)
            if entry is not None:
                self._store_knowledge(entry)

    def _store_knowledge(self, entry: KnowledgeEntry | None) -> None:
        """Offer one :class:`KnowledgeEntry` to the shared world memory."""
        if entry is None or self.shared_memory is None:
            return
        knowledge_id = entry.knowledge_id
        if knowledge_id in self._shared_knowledge_ids:
            return
        for method_name in ("add_knowledge", "record_knowledge", "add", "record", "store", "upsert"):
            method = getattr(self.shared_memory, method_name, None)
            if not callable(method):
                continue
            try:
                method(entry)
                self._shared_knowledge_ids.add(knowledge_id)
                return
            except TypeError:
                try:
                    method(entry.to_dict())
                    self._shared_knowledge_ids.add(knowledge_id)
                    return
                except Exception as exc:  # noqa: BLE001
                    self._shared_errors.append(
                        f"{method_name}(dict) failed: {type(exc).__name__}: {exc}"
                    )
                    continue
            except Exception as exc:  # noqa: BLE001 - shared memory must not break a battle
                self._shared_errors.append(f"{method_name} failed: {type(exc).__name__}: {exc}")
                continue
        if not self._shared_errors:
            self._shared_errors.append("no usable shared-memory write method found")

    def _record_battle_memory(
        self,
        *,
        outcome: str,
        verification: str,
        visible_text: str,
        reason: str,
    ) -> None:
        """Store the battle summary + outcome lesson in battle memory."""
        if self._memory is None:
            return
        actions: list[dict[str, Any]] = []
        for action in self._run_actions:
            decision = action.get("decision") if isinstance(action.get("decision"), Mapping) else {}
            slot = None
            for key in ("move_slot", "item_slot", "party_slot"):
                value = decision.get(key)
                if isinstance(value, int) and value > 0:
                    slot = value
                    break
            actions.append(
                {
                    "turn": int(action.get("turn") or 0),
                    "action": str(decision.get("action") or ""),
                    "slot": slot,
                    "presses": list(action.get("presses") or []),
                    "verified": bool(action.get("verified")),
                }
            )
        try:
            lesson = self._memory.record_battle(
                outcome=outcome,
                turns=int(self._turns_completed),
                actions=actions,
                cost_usd=self._run_cost_usd,
                evidence_refs=list(self._run_screens[-6:]),
                visible_text=visible_text,
                notes=f"ended via: {reason}; verification: {verification}",
                verification=verification,
            )
            self._collect_lesson(lesson)
        except Exception as exc:  # noqa: BLE001
            self._memory_errors.append(f"battle record failed: {type(exc).__name__}: {exc}")

    def _resolve_outcome(
        self, declared: Mapping[str, Any], ap: ActionPrimitives | None = None
    ) -> tuple[str, str, str]:
        """Derive the reported outcome + its honest verification label.

        Priority: (1) the model's ``finish`` decision, (2) one extra "read the
        last visible resolution screenshot" model call, (3) the deterministic
        effect hint of the final action (run/catch), (4) ``unknown``.
        """
        declared_outcome = str(declared.get("outcome") or "unknown").strip().lower()
        visible_text = str(declared.get("visible_text") or "")
        if declared_outcome in OUTCOMES and declared_outcome != "unknown":
            verification = "verified" if visible_text.strip() else "inferred"
            return declared_outcome, verification, visible_text
        read = self._read_resolution_outcome()
        if read is not None:
            return read
        hint = self._pending_outcome_hint
        if hint in OUTCOMES and hint != "unknown":
            return hint, "inferred", visible_text
        return "unknown", "inferred", visible_text

    def _read_resolution_outcome(self) -> tuple[str, str, str] | None:
        """One model call: read the outcome from the last resolution screenshot.

        The pixels around a battle end are the only evidence of win/loss/caught/
        run, so the agent hands the last visible battle-resolution screenshot
        back to the model once and asks it to call ``finish`` with the outcome it
        can actually read.  Any failure keeps the honest fallback (inferred hint
        or ``unknown``) instead of failing the whole battle.
        """
        if self._outcome_read_attempted or not self._resolution_evidence:
            return None
        self._outcome_read_attempted = True
        if self._paused or self._budget_block_reason():
            return None
        allowed, _reason = self._can_call_model()
        if not allowed:
            return None
        # The outcome text is often on the *previous* page ("the enemy fainted")
        # and the last page may be the level-up notice; send up to two pages.
        screenshots = self._resolution_evidence[-2:]
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompts.render_resolution_outcome_prompt()},
        ]
        for index, evidence in enumerate(screenshots):
            label = "Earlier resolution screen:" if index == 0 and len(screenshots) > 1 else (
                "Last resolution screen:" if len(screenshots) > 1 else "Resolution screen:"
            )
            content.append({"type": "text", "text": label})
            content.append(self._image_block(evidence))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        try:
            reply = self._call_model(messages)
        except ControlDeniedError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bonus read must not fail the battle
            self._memory_errors.append(f"outcome read failed: {type(exc).__name__}: {exc}")
            return None
        parsed = self._executor.parse_reply(reply)
        if parsed.decision is None or parsed.decision.action != "finish":
            return None
        outcome = parsed.decision.outcome
        if outcome not in OUTCOMES or outcome == "unknown":
            return None
        visible_text = str(parsed.decision.visible_observations.get("battle_text") or "")
        if not visible_text:
            # The finish tool has no visible_observations field in some models'
            # output; the raw arguments may still carry the visible text.
            raw = parsed.decision.raw_arguments.get("visible_text")
            if isinstance(raw, str):
                visible_text = raw
        verification = "verified" if visible_text.strip() else "inferred"
        return outcome, verification, visible_text

    def _capture_resolution_shot(self, ap: ActionPrimitives, frame: Frame) -> None:
        """Keep the last few visible battle-resolution screenshots as evidence."""
        try:
            evidence = ap.screenshot(frame, prefix="resolution")
        except Exception:  # noqa: BLE001 - evidence is best effort
            return
        if not evidence.get("bytes"):
            return
        self._resolution_evidence.append(
            {
                "bytes": evidence["bytes"],
                "mime": str(evidence.get("mime") or "image/png"),
                "path": str(evidence.get("path") or ""),
                "frame": int(evidence.get("frame") or 0),
            }
        )
        self._resolution_evidence = self._resolution_evidence[-4:]
        path = str(evidence.get("path") or "")
        if path and path not in self._run_screens:
            self._run_screens.append(path)

    def _last_observed_battle_visible(self, ap: ActionPrimitives) -> bool:
        """True when the most recently observed frame is still battle-ish."""
        frame = ap.last_frame
        if frame is None:
            return False
        if looks_like_battle(frame):
            return True
        analysis = analyze(frame, None)
        return analysis.event is ScreenEvent.DIALOG

    # ------------------------------------------------------------------ #
    # Tool-result replies to the model
    # ------------------------------------------------------------------ #
    def _report_tool_results(
        self,
        parsed: tools.ParsedReply,
        *,
        executed_summary: str,
    ) -> None:
        """Feed one tool-result message per model tool call (protocol hygiene)."""
        results = list(parsed.tool_results or [])
        for index, call in enumerate(parsed.tool_calls):
            result = results[index] if index < len(results) else {}
            if not result.get("ok"):
                self._ctx.add_tool_result(
                    call, str(result.get("error") or "invalid tool call"), ok=False
                )
                continue
            decision = result.get("decision")
            lesson = result.get("lesson")
            if decision is not None:
                text = executed_summary or str(result.get("summary") or "accepted")
            elif lesson is not None:
                text = (
                    f"lesson stored as {lesson.verification} "
                    f"({lesson.category}); you may continue with your action"
                )
            else:
                text = str(result.get("summary") or "accepted")
            self._ctx.add_tool_result(call, text, ok=True)

    # ------------------------------------------------------------------ #
    # Run bookkeeping / result
    # ------------------------------------------------------------------ #
    def _reset_run_state(self, task: Any | None) -> None:
        """Reset all per-battle state (call at the start of ``handle_battle``)."""
        self._run_result = _empty_result()
        self._run_actions: list[dict[str, Any]] = []
        self._run_lessons: list[dict[str, Any]] = []
        self._run_screens: list[str] = []
        self._run_lesson_ids: set[str] = set()
        self._shared_knowledge_ids: set[str] = set()
        self._press_log: list[dict[str, Any]] = []
        self._run_api_calls: int = 0
        self._run_cost_usd: float = 0.0
        self._run_denied: int = 0
        self._turns_completed: int = 0
        self._last_action_summary: str = ""
        self._last_visible_observations: dict[str, str] = {}
        self._pending_outcome_hint: str | None = None
        self._state = {"last_frame": None, "clear_observations": 0, "dialog_presses": 0}
        #: last visible battle-resolution screenshots (used for the outcome read)
        self._resolution_evidence: list[dict[str, Any]] = []
        self._outcome_read_attempted: bool = False
        #: evidence of the battle-intro wait (observations / frames / screens)
        self._intro_wait: dict[str, Any] = {}
        self._intro_screens: list[str] = []
        self._battle_active = False
        self._task_cancelled = False
        self._task = task if isinstance(task, Task) else None
        if self._driver is not None:
            self._driver.reset()

    def _begin_task(self, task: Any | None) -> None:
        """Start a fresh conversation for the battle task."""
        task_obj = task if isinstance(task, Task) else None
        try:
            self._ctx.begin_task(task_obj, keep_history=False)
        except Exception:  # noqa: BLE001 - context health must not block the battle
            pass
        if self._memory is not None and task_obj is not None:
            try:
                self._memory.set_branch(str(getattr(task_obj, "branch_id", "") or ""))
            except Exception:  # noqa: BLE001
                pass

    def _finalize(
        self,
        status: str,
        *,
        outcome: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Build the frozen return dict and persist the battle memory."""
        result = _empty_result()
        result["status"] = status if status in STATUSES else "failed"
        if outcome is not None and outcome in OUTCOMES:
            result["outcome"] = outcome
        elif status == "battle_ended":
            result["outcome"] = self._pending_outcome_hint or "unknown"
        result["turns"] = int(self._turns_completed)
        result["api_calls"] = int(self._run_api_calls)
        result["cost_usd"] = round(float(self._run_cost_usd), 8)
        result["actions"] = list(self._run_actions)
        result["lessons"] = [_public_lesson(item) for item in self._run_lessons]
        screens = list(self._run_screens)
        frames: list[int] = []
        if self._ap is not None:
            screens.extend(str(path) for path in self._ap.evidence_screens if path)
            frames = [int(index) for index in self._ap.evidence_frames]
        result["evidence"] = {
            "screens": list(dict.fromkeys(screens))[-40:],
            "frames": frames[-200:],
        }
        result["reason"] = str(reason)[:400]
        presses = int(getattr(self._ap.stats, "presses", 0)) if self._ap is not None else 0
        result["presses"] = presses
        result["buttons"] = presses
        result["denied"] = int(self._run_denied)
        result["error"] = (
            result["reason"] if result["status"] in ("failed", "control_denied") else ""
        )
        result["intro_wait"] = dict(self._intro_wait)
        self._battle_active = False
        if result["status"] == "battle_ended":
            # The recovery hints were applied to this battle's prompts: they are
            # consumed now, so they cannot leak into an unrelated later battle.
            self._recovery_hints = []
        self._stats["battles"] = int(self._stats.get("battles", 0)) + 1
        self._stats["last_battle_turns"] = int(self._turns_completed)
        self._stats["last_status"] = result["status"]
        self._stats["last_outcome"] = result["outcome"]
        self._stats["last_reason"] = result["reason"]
        try:
            if self._memory is not None:
                self._memory.save()
        except Exception:  # noqa: BLE001 - persistence must not lose the result
            pass
        try:
            self._ctx.end_task(summary=f"battle {result['status']} ({result['outcome']})")
        except Exception:  # noqa: BLE001
            pass
        return result

    # ------------------------------------------------------------------ #
    # Events / publishing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_event(event: Any) -> tuple[EventType | None, dict[str, Any], AgentRole | None]:
        """Tolerantly extract ``(event_type, payload, target)`` from an event."""
        if event is None:
            return None, {}, None
        raw_type: Any = getattr(event, "event_type", None)
        payload: Any = getattr(event, "payload", None)
        target: Any = getattr(event, "target_agent", None)
        if isinstance(event, Mapping):
            raw_type = event.get("event_type") or event.get("type")
            payload = event.get("payload")
            target = event.get("target_agent") or target
        if raw_type is None:
            raw_type = event
        try:
            event_type = EventType.coerce(raw_type)
        except Exception:  # noqa: BLE001 - unknown events are ignored, never fatal
            return None, {}, None
        payload_dict = dict(payload) if isinstance(payload, Mapping) else {}
        role: AgentRole | None = None
        if target is not None:
            try:
                role = AgentRole.coerce(target)
            except Exception:  # noqa: BLE001
                role = None
        return event_type, payload_dict, role

    def _event_applies_to_me(self, payload: Mapping[str, Any], target: AgentRole | None) -> bool:
        """True when a pause/resume event is addressed to the Pokémon agent."""
        agent = str(payload.get("agent") or "").strip().lower()
        if agent:
            return agent in ("pokemon", "pokémon", self.role.value)
        if target is not None:
            return target is self.role
        return True

    def _publish(self, event_type: EventType, payload: Mapping[str, Any]) -> AgentMessage | None:
        """Best-effort publish of an agent event (bus APIs differ while the
        protocol subsystem is still being built, so several shapes are tried)."""
        if self.bus is None:
            return None
        try:
            message = AgentMessage.create(
                event_type,
                source_agent=self.role,
                payload=dict(payload),
                parent_task_id=self._task_id(),
            )
        except Exception as exc:  # noqa: BLE001 - publishing is never fatal
            self._publish_errors.append(f"message build failed: {type(exc).__name__}: {exc}")
            return None
        for method_name in ("publish", "emit", "send", "put", "enqueue", "append"):
            method = getattr(self.bus, method_name, None)
            if not callable(method):
                continue
            try:
                method(message)
                return message
            except TypeError:
                try:
                    method(event_type, dict(payload))
                    return message
                except Exception as exc:  # noqa: BLE001
                    self._publish_errors.append(
                        f"bus.{method_name} failed: {type(exc).__name__}: {exc}"
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                self._publish_errors.append(f"bus.{method_name} failed: {type(exc).__name__}: {exc}")
                continue
        if not self._publish_errors:
            self._publish_errors.append("no usable event-bus publish method found")
        return None

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def _task_id(self) -> str:
        return self._task.task_id if self._task is not None else ""

    def _coerce_max_turns(self, max_turns: Any) -> int:
        """Clamp the turn cap; fall back to the multi-agent config when unset."""
        try:
            value = int(max_turns)
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            configured = int(
                _config_value(self.config, "multi_agent", "battle_max_turns", 0) or 0
            )
            value = configured if configured > 0 else DEFAULT_MAX_TURNS
        return max(1, min(200, value))

    def _describe_execution(
        self, decision: tools.BattleDecision, execution: Mapping[str, Any]
    ) -> str:
        parts = [f"action={decision.action}"]
        if decision.slot is not None:
            parts.append(f"slot={decision.slot}")
        parts.append(f"presses={list(execution.get('presses') or [])}")
        parts.append(
            "verified: every press changed the screen"
            if execution.get("verified")
            else f"NOT verified ({execution.get('reason')})"
        )
        return "; ".join(parts)

    def _runtime_notes(self) -> str:
        notes: list[str] = []
        for hint in self._recovery_hints[-3:]:
            level = str(hint.get("level") or "L1")
            notes.append(
                f"{level} recovery strategy note from the director: {hint.get('hint')}"
            )
        if self._memory_errors:
            notes.append("one battle lesson could not be stored; keep deciding from the screen")
        if self._shared_errors:
            notes.append("shared-memory write failed; local battle notes still work")
        return " | ".join(notes)

    def _model_name(self) -> str:
        model_for = getattr(self.router, "model_for", None)
        if not callable(model_for):
            return ""
        for role in (self.role, self.role.value):
            try:
                return str(model_for(role) or "")
            except Exception:  # noqa: BLE001 - status only
                continue
        return ""

    def _arbiter_owner(self) -> str:
        if self.arbiter is None:
            return ""
        for attribute in ("owner", "current_owner"):
            value = getattr(self.arbiter, attribute, None)
            if isinstance(value, ControlOwner):
                return value.value
            if isinstance(value, str):
                return value
        current = getattr(self.arbiter, "state", None)
        owner = getattr(current, "owner", None) if current is not None else None
        if isinstance(owner, ControlOwner):
            return owner.value
        return ""

    def _record_error(self, exc: BaseException) -> None:
        self._stats["last_error"] = f"{type(exc).__name__}: {exc}"
        try:
            self._ctx.record_error(exc)
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "PokemonAgent",
    "BattleMenuDriver",
    "battle_menu_layout",
    "looks_like_battle",
    "DEFAULT_MAX_TURNS",
    "BATTLE_END_CONSECUTIVE",
    "BATTLE_MENU_MAIN",
    "BATTLE_MENU_MOVE",
    "BATTLE_MENU_TEXT",
    "BATTLE_MENU_UNKNOWN",
    "MAIN_MENU_TARGETS",
    "STATUSES",
    "OUTCOMES",
]
