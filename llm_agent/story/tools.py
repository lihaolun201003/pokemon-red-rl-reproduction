"""Constrained tool layer for the story/dialogue agent.

The model cannot run code: everything it can do is one of the six validated
tool calls in :data:`TOOL_SCHEMAS`.  This module owns

* the OpenAI-compatible function-calling schemas (strict JSON schemas with
  enums/ranges and ``additionalProperties: false``),
* strict argument validation: an unknown tool name (e.g. a raw ``press_button``
  or a navigation/execution tool) or an invalid argument raises
  :class:`ToolError` and **nothing is pressed**, so tool-permission isolation is
  testable,
* execution against the injected :class:`ActionPrimitives` (which goes through
  the arbiter-guarded emulator, so ``ControlDeniedError`` propagates unchanged),
  the :class:`StoryMemory` and the current conversation evidence,
* :func:`acquire_dialog`: a deterministic, bounded interaction probe that lets
  the agent *start* a conversation itself (a blind walker only presses
  directions, so dialogue boxes would otherwise never open).

Only the story-relevant whitelist exists here: ``observe``, ``advance_dialog``,
``choose_option``, ``record_fact``, ``propose_task``, ``finish``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from llm_agent.contracts import AgentError, Frame, ScreenEvent, ToolCall, ToolResult
from llm_agent.multi_agent.contracts import (
    AgentRole,
    ControlDeniedError,
)
from llm_agent.observation import ScreenAnalysis, analyze
from llm_agent.story.memory import (
    STORY_KNOWLEDGE_CATEGORIES,
    STORY_PRIORITIES,
    STORY_PROPOSAL_OWNERS,
    STORY_PROPOSAL_TYPES,
    StoryMemory,
)

__all__ = [
    "STORY_TOOL_NAMES",
    "TOOL_SCHEMAS",
    "ALLOWED_PRESS_BUTTONS",
    "VERIFICATION_NAMES",
    "DIALOG_SCREEN_EVENTS",
    "PROBE_DIRECTIONS",
    "MAX_PROBE_PRESSES",
    "is_dialog_screen",
    "acquire_dialog",
    "ToolError",
    "validate_call",
    "StoryTools",
]

#: The complete tool whitelist of the story agent (order is stable for prompts).
STORY_TOOL_NAMES: tuple[str, ...] = (
    "observe",
    "advance_dialog",
    "choose_option",
    "record_fact",
    "propose_task",
    "finish",
)

#: Buttons the dialogue tools may press (directions only move the menu cursor).
ALLOWED_PRESS_BUTTONS: tuple[str, ...] = ("a", "b", "up", "down", "left", "right")

#: Honesty levels accepted by ``record_fact``.
VERIFICATION_NAMES: tuple[str, ...] = ("verified", "inferred", "hypothesis")

#: Hard cap on the number of real button presses one tool call may produce.
MAX_PRESSES_PER_CALL: int = 4

#: Screen events that count as an open dialogue box or menu window.
DIALOG_SCREEN_EVENTS: frozenset[ScreenEvent] = frozenset({ScreenEvent.DIALOG, ScreenEvent.MENU})

#: Directions tried by the bounded interaction probe (order is deterministic).
PROBE_DIRECTIONS: tuple[str, ...] = ("up", "left", "right", "down")

#: Hard cap of the interaction probe: 4 directions x 2 A presses (+ turns) <= 12.
MAX_PROBE_PRESSES: int = 12

#: Observations inspected after each probe press.
PROBE_OBSERVATIONS_PER_STEP: int = 3

#: Character caps (kept well below the global discovery limits).
MAX_CONTENT_CHARS: int = 1000
MAX_DIALOGUE_TEXT_CHARS: int = 600
MAX_DESCRIPTION_CHARS: int = 300
MAX_SUMMARY_CHARS: int = 600


# --------------------------------------------------------------------------- #
# Screen classification (shared with the agent)
# --------------------------------------------------------------------------- #

def is_dialog_screen(analysis: ScreenAnalysis | None) -> bool:
    """True when the screen looks like an open dialogue box or a menu window.

    The pixel classifier in :mod:`llm_agent.observation` is the primary signal;
    this helper only adds a conservative fallback on the raw ``dialog_score`` /
    ``menu_score`` statistics for transitional frames.  The screenshot remains
    the ground truth for the model.
    """
    if analysis is None:
        return False
    if analysis.event in DIALOG_SCREEN_EVENTS:
        return True
    if analysis.event is ScreenEvent.BATTLE:
        return False
    stats = analysis.stats or {}
    try:
        dialog_score = float(stats.get("dialog_score") or 0.0)
        menu_score = float(stats.get("menu_score") or 0.0)
        band_white = float(stats.get("band_white_ratio") or 0.0)
        right_white = float(stats.get("right_white_ratio") or 0.0)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    if dialog_score >= 0.70:
        return True
    if menu_score >= 0.85 and band_white < 0.45 and right_white >= 0.40:
        return True
    return False


def _agent_holds_control(arbiter: Any) -> bool:
    """Best-effort lease check used before any probe/tool press."""
    if arbiter is None:
        return True
    holds = getattr(arbiter, "holds", None)
    if callable(holds):
        try:
            return bool(holds(AgentRole.STORY))
        except TypeError:  # pragma: no cover - string-based arbiters
            try:
                return bool(holds("story"))
            except Exception:  # noqa: BLE001
                return False
        except Exception:  # noqa: BLE001 - never press on a broken check
            return False
    owner = getattr(arbiter, "owner", None)
    if owner is None:
        return True
    value = getattr(owner, "value", owner)
    return str(value).strip().lower() == "story"


# --------------------------------------------------------------------------- #
# Interaction probe: start a conversation on purpose
# --------------------------------------------------------------------------- #

def acquire_dialog(
    actions: Any,
    *,
    arbiter: Any = None,
    max_presses: int = MAX_PROBE_PRESSES,
    max_observations_per_step: int = PROBE_OBSERVATIONS_PER_STEP,
    prefix: str = "story_probe",
) -> dict[str, Any]:
    """Try to open a dialogue with a bounded deterministic button probe.

    The probe is an *affordance discovery* only: it presses A while facing the
    current direction and then while facing each of :data:`PROBE_DIRECTIONS`
    (up to two A presses each), observing the screen after every press.  It
    never decides what the dialogue says -- the agent still sends the real
    screenshot to the model once a text box becomes visible.

    Returns a dict::

        {"status": "dialog" | "no_dialog" | "interrupted" | "control_denied",
         "presses": [...], "attempts": [{"step", "frame", "event"}],
         "evidence": {"screens": [...], "frames": [...]},
         "reason": str,
         # only for status == "dialog":
         "frame": Frame, "analysis": ScreenAnalysis, "shot": {...},
         "evidence_ref": str}

    A menu that opens during the probe is closed with B (at most twice); if it
    cannot be closed the probe stops with ``interrupted``.
    """
    if actions is None:
        raise ValueError("acquire_dialog requires an ActionPrimitives-like object")
    cap = max(0, int(max_presses))
    observations = max(1, int(max_observations_per_step))
    presses: list[str] = []
    attempts: list[dict[str, Any]] = []
    screens: list[str] = []
    frames: list[int] = []

    def _capture(frame: Frame, analysis: ScreenAnalysis, shot: Mapping[str, Any]) -> dict[str, Any]:
        index = int(getattr(frame, "frame_index", 0) or 0)
        path = str((shot or {}).get("path") or "")
        frames.append(index)
        if path:
            screens.append(path)
        return {
            "frame": frame,
            "analysis": analysis,
            "shot": dict(shot or {}),
            "evidence_ref": path or f"frame:{index}",
        }

    def _result(status: str, *, reason: str = "", captured: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "presses": list(presses),
            "attempts": [dict(item) for item in attempts],
            "evidence": {"screens": list(screens[-30:]), "frames": list(frames[-30:])},
            "reason": str(reason),
        }
        if captured:
            payload.update(dict(captured))
        return payload

    def _press(button: str) -> bool:
        if len(presses) >= cap:
            return False
        actions.press(button, note=f"probe:{button}")
        presses.append(button)
        return True

    def _classify(frame: Frame, previous: Frame | None, step: str) -> tuple[str, dict[str, Any]]:
        analysis = analyze(frame, previous)
        shot = actions.screenshot(frame, prefix=prefix)
        captured = _capture(frame, analysis, shot)
        attempts.append(
            {
                "step": str(step),
                "frame": int(getattr(frame, "frame_index", 0) or 0),
                "event": analysis.event.value,
                "presses": len(presses),
            }
        )
        if analysis.event is ScreenEvent.BATTLE:
            return "battle", captured
        if analysis.event is ScreenEvent.MENU:
            return "menu", captured
        if is_dialog_screen(analysis):
            return "dialog", captured
        return "none", captured

    def _observe_step(step: str) -> tuple[str, dict[str, Any]]:
        last: dict[str, Any] = {}
        for index in range(observations):
            if not _agent_holds_control(arbiter):
                return "control_denied", last
            frame = actions.observe()
            status, captured = _classify(frame, previous_frame[0], f"{step}#{index + 1}")
            previous_frame[0] = frame
            last = captured
            if status != "none":
                return status, captured
        return "none", last

    def _close_menu(step: str) -> tuple[str, dict[str, Any]]:
        last: dict[str, Any] = {}
        for attempt in range(2):
            if not _press("b"):
                return "interrupted", last
            frame = actions.observe()
            status, captured = _classify(frame, previous_frame[0], f"{step}_b{attempt + 1}")
            previous_frame[0] = frame
            last = captured
            if status == "dialog":
                return "dialog", captured
            if status == "battle":
                return "battle", captured
            if status == "none":
                return "closed", captured
        return "interrupted", last

    if not _agent_holds_control(arbiter):
        return _result("control_denied", reason="the control lease is not held by the story agent")

    previous_frame: list[Frame | None] = [None]
    frame = actions.observe()
    analysis = analyze(frame, None)
    shot = actions.screenshot(frame, prefix=prefix)
    initial = _capture(frame, analysis, shot)
    previous_frame[0] = frame
    attempts.append(
        {
            "step": "initial",
            "frame": int(getattr(frame, "frame_index", 0) or 0),
            "event": analysis.event.value,
            "presses": 0,
        }
    )
    if is_dialog_screen(analysis):
        return _result("dialog", reason="a dialogue box or menu was already visible", captured=initial)
    if analysis.event is ScreenEvent.BATTLE:
        return _result("interrupted", reason="a battle is on screen; not probing for dialogue", captured=initial)

    # 1) one A press in the current facing.
    if _press("a"):
        status, captured = _observe_step("a")
        if status == "dialog":
            return _result("dialog", reason="a dialogue opened after pressing A", captured=captured)
        if status == "battle":
            return _result("interrupted", reason="a battle appeared during the probe", captured=captured)
        if status == "control_denied":
            return _result("control_denied", reason="the control lease was lost during the probe")
        if status == "menu":
            status, captured = _close_menu("a_menu")
            if status == "interrupted":
                return _result("interrupted", reason="a menu opened and could not be closed", captured=captured)
            if status == "dialog":
                return _result("dialog", reason="a dialogue appeared after closing a menu", captured=captured)
            if status == "battle":
                return _result("interrupted", reason="a battle appeared during the probe", captured=captured)

    # 2) face each direction and try A (up to twice per direction), bounded.
    for direction in PROBE_DIRECTIONS:
        if len(presses) >= cap:
            break
        _press(direction)
        for attempt_index in range(2):
            if len(presses) >= cap:
                break
            if not _press("a"):
                break
            status, captured = _observe_step(f"{direction}_a{attempt_index + 1}")
            if status == "dialog":
                return _result(
                    "dialog",
                    reason=f"a dialogue opened while facing {direction}",
                    captured=captured,
                )
            if status == "battle":
                return _result("interrupted", reason="a battle appeared during the probe", captured=captured)
            if status == "control_denied":
                return _result("control_denied", reason="the control lease was lost during the probe")
            if status == "menu":
                status, captured = _close_menu(f"{direction}_menu")
                if status == "interrupted":
                    return _result(
                        "interrupted",
                        reason="a menu opened and could not be closed",
                        captured=captured,
                    )
                if status == "dialog":
                    return _result(
                        "dialog",
                        reason="a dialogue appeared after closing a menu",
                        captured=captured,
                    )
                if status == "battle":
                    return _result("interrupted", reason="a battle appeared during the probe", captured=captured)
                break  # menu closed; continue with the next direction
            if len(presses) >= cap:
                break

    return _result(
        "no_dialog",
        reason=(
            f"no dialogue started after {len(presses)} bounded probe press(es); "
            "the current position has nothing to talk to"
        ),
    )



# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class ToolError(AgentError):
    """A rejected story tool call (unknown name, bad argument, not allowed)."""

    def __init__(self, message: str, *, tool: str = "", details: Mapping[str, Any] | None = None) -> None:
        super().__init__(str(message))
        self.tool = str(tool or "")
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"error": "ToolError", "tool": self.tool, "message": str(self)}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

def _object_schema(properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _function_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _button_enum() -> dict[str, Any]:
    return {"type": "string", "enum": list(ALLOWED_PRESS_BUTTONS)}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _function_schema(
        "observe",
        "Look at the current Game Boy screen again without pressing any button. "
        "Use it only when you need a fresh look; the next screenshot is attached anyway.",
        _object_schema({}),
    ),
    _function_schema(
        "advance_dialog",
        "Advance the dialogue text with A, or close/cancel it with B. "
        "Prefer A while a text box is open; use B to close a box or cancel a menu.",
        _object_schema(
            {
                "button": {
                    "type": "string",
                    "enum": ["a", "b"],
                    "description": "A advances the text, B closes or cancels it.",
                },
                "times": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "description": "How many short presses to perform (1-3).",
                },
            },
            required=["button"],
        ),
    ),
    _function_schema(
        "choose_option",
        "Pick a visible menu option. Move the cursor with the direction buttons, then "
        "confirm with A (or cancel with B). Only choose text you can actually read.",
        _object_schema(
            {
                "direction": _button_enum(),
                "presses": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "description": "How many times to press the direction (0 = cursor already on the option).",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "True confirms with A, False cancels with B.",
                },
            },
            required=["direction"],
        ),
    ),
    _function_schema(
        "record_fact",
        "Write down exactly one person, clue, interpretation or hypothesis you read on screen. "
        "Choose the honest level: verified = visible in this screenshot right now, "
        "inferred = a reasonable reading of what was shown, hypothesis = an untested guess.",
        _object_schema(
            {
                "category": {
                    "type": "string",
                    "enum": list(STORY_KNOWLEDGE_CATEGORIES),
                    "description": "Kind of note: a person, an interactive thing, a story clue, a quest, an area or other.",
                },
                "content": {
                    "type": "string",
                    "description": "The fact/interpretation in one short sentence, based only on the screen.",
                },
                "verification_status": {
                    "type": "string",
                    "enum": list(VERIFICATION_NAMES),
                    "description": "Honest level of this note.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence 0..1.",
                },
                "npc_name": {
                    "type": "string",
                    "description": "For a person: the name or short label visible/derivable on screen.",
                },
                "dialogue_text": {
                    "type": "string",
                    "description": "The literal dialogue text you just read (required for a verified note).",
                },
            },
            required=["category", "content", "verification_status"],
        ),
    ),
    _function_schema(
        "propose_task",
        "Ask another agent for follow-up work that follows from what you just read. "
        "Describe the target in words: no coordinates, no map knowledge.",
        _object_schema(
            {
                "owner_agent": {
                    "type": "string",
                    "enum": list(STORY_PROPOSAL_OWNERS),
                    "description": "Which agent should do the work.",
                },
                "task_type": {
                    "type": "string",
                    "enum": list(STORY_PROPOSAL_TYPES),
                    "description": "Kind of follow-up work.",
                },
                "description": {
                    "type": "string",
                    "description": "What should be done, described in plain words.",
                },
                "priority": {
                    "type": "string",
                    "enum": list(STORY_PRIORITIES),
                    "description": "How urgent the follow-up is.",
                },
                "payload": {
                    "type": "object",
                    "description": "Structured hints for the other agent (described target, reason).",
                },
            },
            required=["owner_agent", "task_type", "description"],
        ),
    ),
    _function_schema(
        "finish",
        "End this conversation now and report honestly what was learned (or that nothing was).",
        _object_schema(
            {
                "summary": {
                    "type": "string",
                    "description": "Short honest report of what was read and learned.",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["dialog_ended", "no_clue", "blocked"],
                    "description": "dialog_ended = conversation over; no_clue = nothing learned; blocked = cannot continue.",
                },
            },
            required=["summary"],
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #

def _as_str(value: Any, *, field: str, allow_empty: bool = False, limit: int = 0) -> str:
    if isinstance(value, (bytes, bytearray)):
        raise ToolError(f"{field} must be a string", tool="validation")
    if not isinstance(value, str):
        raise ToolError(f"{field} must be a string, got {type(value).__name__}", tool="validation")
    text = value.strip()
    if not text and not allow_empty:
        raise ToolError(f"{field} must not be empty", tool="validation")
    return text[:limit] if limit else text


def _as_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{field} must be an integer, got {type(value).__name__}", tool="validation")
    if not minimum <= value <= maximum:
        raise ToolError(f"{field} must be in [{minimum}, {maximum}], got {value}", tool="validation")
    return int(value)


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ToolError(f"{field} must be a boolean, got {type(value).__name__}", tool="validation")


def _as_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{field} must be a number, got {type(value).__name__}", tool="validation")
    return max(0.0, min(1.0, float(value)))


def _as_enum(value: Any, *, field: str, allowed: Sequence[str]) -> str:
    text = _as_str(value, field=field).lower()
    if text not in allowed:
        raise ToolError(
            f"{field} must be one of {list(allowed)}, got {text!r}", tool="validation"
        )
    return text


def _as_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ToolError(f"{field} must be a JSON object", tool="validation")
    out = dict(value)
    try:
        json.dumps(out, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ToolError(f"{field} is not JSON-serialisable: {exc}", tool="validation") from exc
    return out


def validate_call(name: Any, arguments: Any) -> dict[str, Any]:
    """Validate and normalise one tool call; raise :class:`ToolError` on failure.

    Returns the normalised argument dict (defaults filled in).
    """
    if not isinstance(name, str) or not name.strip():
        raise ToolError("tool name must be a non-empty string", tool="validation")
    tool = name.strip()
    if tool not in STORY_TOOL_NAMES:
        raise ToolError(
            f"tool {tool!r} is not allowed for the story agent; "
            f"allowed tools: {list(STORY_TOOL_NAMES)}",
            tool=tool,
        )
    if arguments is None:
        data: dict[str, Any] = {}
    elif isinstance(arguments, Mapping):
        data = dict(arguments)
    else:
        raise ToolError(
            f"arguments for {tool!r} must be a JSON object, got {type(arguments).__name__}",
            tool=tool,
        )

    if tool == "observe":
        _reject_unknown(data, tool, allowed=())
        return {}

    if tool == "advance_dialog":
        _reject_unknown(data, tool, allowed=("button", "times"))
        return {
            "button": _as_enum(data.get("button"), field="button", allowed=("a", "b")),
            "times": _as_int(data.get("times", 1), field="times", minimum=1, maximum=3),
        }

    if tool == "choose_option":
        _reject_unknown(data, tool, allowed=("direction", "presses", "confirm"))
        return {
            "direction": _as_enum(
                data.get("direction"), field="direction", allowed=("up", "down", "left", "right")
            ),
            "presses": _as_int(data.get("presses", 1), field="presses", minimum=0, maximum=3),
            "confirm": _as_bool(data.get("confirm", True), field="confirm"),
        }

    if tool == "record_fact":
        _reject_unknown(
            data,
            tool,
            allowed=(
                "category",
                "content",
                "verification_status",
                "confidence",
                "npc_name",
                "dialogue_text",
            ),
        )
        return {
            "category": _as_enum(
                data.get("category"), field="category", allowed=STORY_KNOWLEDGE_CATEGORIES
            ),
            "content": _as_str(data.get("content"), field="content", limit=MAX_CONTENT_CHARS),
            "verification_status": _as_enum(
                data.get("verification_status"),
                field="verification_status",
                allowed=VERIFICATION_NAMES,
            ),
            "confidence": _as_float(data.get("confidence", 0.5), field="confidence"),
            "npc_name": _as_str(
                data.get("npc_name", ""), field="npc_name", allow_empty=True, limit=120
            ),
            "dialogue_text": _as_str(
                data.get("dialogue_text", ""),
                field="dialogue_text",
                allow_empty=True,
                limit=MAX_DIALOGUE_TEXT_CHARS,
            ),
        }

    if tool == "propose_task":
        _reject_unknown(
            data,
            tool,
            allowed=("owner_agent", "task_type", "description", "priority", "payload"),
        )
        description = _as_str(
            data.get("description"), field="description", limit=MAX_DESCRIPTION_CHARS
        )
        payload = _as_mapping(data.get("payload"), field="payload")
        if "target" not in payload:
            # Always hand the other agent a described target, never a guessed cell.
            payload = {"target": {"kind": "described", "description": description[:200]}, **payload}
        return {
            "owner_agent": _as_enum(
                data.get("owner_agent"), field="owner_agent", allowed=STORY_PROPOSAL_OWNERS
            ),
            "task_type": _as_enum(
                data.get("task_type"), field="task_type", allowed=STORY_PROPOSAL_TYPES
            ),
            "description": description,
            "priority": _as_enum(
                data.get("priority", "normal"), field="priority", allowed=STORY_PRIORITIES
            ),
            "payload": payload,
        }

    # tool == "finish"
    _reject_unknown(data, tool, allowed=("summary", "outcome"))
    return {
        "summary": _as_str(data.get("summary"), field="summary", limit=MAX_SUMMARY_CHARS),
        "outcome": _as_enum(
            data.get("outcome", "dialog_ended"),
            field="outcome",
            allowed=("dialog_ended", "no_clue", "blocked"),
        ),
    }


def _reject_unknown(data: Mapping[str, Any], tool: str, *, allowed: Sequence[str]) -> None:
    unknown = sorted(str(key) for key in data if str(key) not in allowed)
    if unknown:
        raise ToolError(
            f"unknown argument(s) for {tool!r}: {unknown}; allowed: {list(allowed)}",
            tool=tool,
        )


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

class StoryTools:
    """Execute validated story tool calls against the real game surface.

    Parameters
    ----------
    memory:
        The agent's :class:`StoryMemory`.
    actions:
        An :class:`ActionPrimitives` instance (preferred, created by the agent)
        or a plain emulator exposing ``observe``/``press``/``wait``.
    arbiter:
        Optional :class:`ControlArbiter`; when it exposes ``holds(role)`` a press
        is refused locally with :class:`ControlDeniedError` if the story agent is
        not the current owner.  The guarded emulator enforces the same rule.
    """

    def __init__(
        self,
        *,
        memory: StoryMemory,
        actions: Any = None,
        emulator: Any = None,
        arbiter: Any = None,
        max_presses_per_call: int = MAX_PRESSES_PER_CALL,
        clock: Any = None,
    ) -> None:
        if memory is None:
            raise ValueError("StoryTools requires a StoryMemory instance")
        if actions is None and emulator is None:
            raise ValueError("StoryTools requires an ActionPrimitives/emulator")
        self.memory = memory
        self.actions = actions
        self.emulator = emulator
        self.arbiter = arbiter
        self.max_presses_per_call = max(1, min(int(max_presses_per_call), MAX_PRESSES_PER_CALL))
        self._clock = clock
        #: per-conversation execution state
        self.presses: list[str] = []
        self.knowledge_created: list[dict[str, Any]] = []
        self.proposals_created: list[dict[str, Any]] = []
        self.npc_updates: list[dict[str, Any]] = []
        self.finished: bool = False
        self.finish_payload: dict[str, Any] = {}
        self.tool_errors: int = 0
        self._presses_this_call: int = 0

    # -- lifecycle ------------------------------------------------------- #

    def begin(self) -> None:
        """Reset the per-conversation execution state."""
        self.presses = []
        self.knowledge_created = []
        self.proposals_created = []
        self.npc_updates = []
        self.finished = False
        self.finish_payload = {}
        self.tool_errors = 0
        self._presses_this_call = 0

    # -- low level ------------------------------------------------------- #

    def _agent_holds_control(self) -> bool:
        return _agent_holds_control(self.arbiter)

    def _press(self, button: str, *, note: str = "") -> None:
        """Press one button through the guarded surface (raises on denial)."""
        name = str(button).strip().lower()
        if name not in ALLOWED_PRESS_BUTTONS:
            raise ToolError(f"button {button!r} is not allowed", tool="press")
        if self._presses_this_call >= self.max_presses_per_call:
            raise ToolError(
                f"this tool call already used the maximum of {self.max_presses_per_call} presses",
                tool="press",
            )
        if not self._agent_holds_control():
            raise ControlDeniedError(
                "story agent tried to press without holding the control lease",
                agent=AgentRole.STORY.value,
                owner=str(getattr(self.arbiter, "owner", "") or "other"),
            )
        if self.actions is not None:
            self.actions.press(name, note=note)
        else:
            self.emulator.press(name)
        self.presses.append(name)
        self._presses_this_call += 1

    # -- execution ------------------------------------------------------- #

    def execute(
        self,
        call: ToolCall | Any,
        *,
        evidence_ref: str = "",
        frame_index: int | None = None,
    ) -> ToolResult:
        """Execute one tool call; validation failures raise :class:`ToolError`."""
        tool_call_id = str(getattr(call, "id", "") or "call")
        name = getattr(call, "name", "")
        arguments = getattr(call, "arguments", {})
        normalised = validate_call(name, arguments)
        tool = str(name).strip()
        self._presses_this_call = 0
        if tool == "observe":
            return self._execute_observe(tool_call_id)
        if tool == "advance_dialog":
            return self._execute_advance(tool_call_id, normalised, evidence_ref=evidence_ref)
        if tool == "choose_option":
            return self._execute_choose(tool_call_id, normalised)
        if tool == "record_fact":
            return self._execute_record(
                tool_call_id, normalised, evidence_ref=evidence_ref, frame_index=frame_index
            )
        if tool == "propose_task":
            return self._execute_propose(tool_call_id, normalised)
        return self._execute_finish(tool_call_id, normalised)

    def execute_safe(
        self,
        call: ToolCall | Any,
        *,
        evidence_ref: str = "",
        frame_index: int | None = None,
    ) -> ToolResult:
        """Like :meth:`execute` but converts :class:`ToolError` into a result.

        ``ControlDeniedError`` deliberately propagates so the agent can report
        ``status="control_denied"`` instead of pretending it acted.
        """
        tool_call_id = str(getattr(call, "id", "") or "call")
        name = str(getattr(call, "name", "") or "")
        try:
            return self.execute(call, evidence_ref=evidence_ref, frame_index=frame_index)
        except ToolError as exc:
            self.tool_errors += 1
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                summary=f"rejected: {exc}",
                error=str(exc),
                data={"tool_error": exc.to_dict()},
            )
        except (ControlDeniedError,):
            raise
        except Exception as exc:  # noqa: BLE001 - a broken tool must not kill the loop
            self.tool_errors += 1
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                summary=f"tool failed: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )

    # -- individual tools ------------------------------------------------ #

    def _execute_observe(self, tool_call_id: str) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            name="observe",
            ok=True,
            summary="Looked at the screen again; no button was pressed.",
            data={"observed": True},
        )

    def _execute_advance(
        self, tool_call_id: str, args: Mapping[str, Any], *, evidence_ref: str
    ) -> ToolResult:
        button = str(args["button"])
        times = int(args["times"])
        for index in range(times):
            self._press(button, note=f"advance_dialog:{index + 1}")
        return ToolResult(
            tool_call_id=tool_call_id,
            name="advance_dialog",
            ok=True,
            summary=(
                f"Pressed {button.upper()} {times} time(s). The game advanced and the "
                "screen was re-captured for the next decision."
            ),
            data={"pressed": [button] * times, "button": button, "times": times},
            screens=[evidence_ref] if evidence_ref else [],
        )

    def _execute_choose(self, tool_call_id: str, args: Mapping[str, Any]) -> ToolResult:
        direction = str(args["direction"])
        presses = int(args["presses"])
        confirm = bool(args["confirm"])
        for _ in range(presses):
            self._press(direction, note="choose_option:cursor")
        final = "a" if confirm else "b"
        self._press(final, note="choose_option:confirm" if confirm else "choose_option:cancel")
        return ToolResult(
            tool_call_id=tool_call_id,
            name="choose_option",
            ok=True,
            summary=(
                f"Moved the cursor with {direction.upper()} x{presses} and pressed "
                f"{final.upper()} to {'confirm' if confirm else 'cancel'}."
            ),
            data={
                "direction": direction,
                "presses": presses,
                "confirm": confirm,
                "pressed": [direction] * presses + [final],
            },
        )

    def _execute_record(
        self,
        tool_call_id: str,
        args: Mapping[str, Any],
        *,
        evidence_ref: str,
        frame_index: int | None,
    ) -> ToolResult:
        category = str(args["category"])
        content = str(args["content"])
        status = str(args["verification_status"])
        npc_name = str(args.get("npc_name") or "")
        dialogue_text = str(args.get("dialogue_text") or "")
        evidence_refs = [evidence_ref] if evidence_ref else []
        result = self.memory.record_knowledge(
            category=category,
            content=content,
            verification_status=status,
            confidence=float(args.get("confidence", 0.5)),
            evidence_refs=evidence_refs,
            subject=npc_name,
            npc_name=npc_name,
            dialogue_text=dialogue_text,
            frame_index=frame_index,
        )
        entry = dict(result.get("entry") or {})
        public = StoryMemory.public_knowledge(entry) if entry else {}
        created = bool(result.get("created"))
        downgraded = bool(result.get("downgraded"))
        data: dict[str, Any] = {
            "knowledge": public,
            "created": created,
            "downgraded": downgraded,
        }
        if dialogue_text:
            data["dialogue_text"] = dialogue_text
        if npc_name:
            npc_verified = public.get("verification_status") == "verified"
            npc = self.memory.record_npc(
                npc_name,
                notes=content,
                verified=npc_verified,
                evidence_ref=evidence_ref,
            )
            self.npc_updates.append(npc)
            data["npc"] = npc
        if created:
            self.knowledge_created.append(public)
        parts = [f"Recorded as {public.get('verification_status', status).upper()}: {content}"]
        if downgraded:
            parts.append(
                "NOTE: a 'verified' note needs the visible dialogue text and its evidence; "
                "this one was stored as INFERRED instead."
            )
        elif not created:
            parts.append("This note already existed; it was not duplicated.")
        if npc_name:
            parts.append(f"Person recorded: {npc_name}.")
        return ToolResult(
            tool_call_id=tool_call_id,
            name="record_fact",
            ok=True,
            summary=" ".join(parts),
            data=data,
            screens=[evidence_ref] if evidence_ref else [],
        )

    def _execute_propose(self, tool_call_id: str, args: Mapping[str, Any]) -> ToolResult:
        proposal, created = self.memory.add_proposal(
            owner_agent=str(args["owner_agent"]),
            task_type=str(args["task_type"]),
            description=str(args["description"]),
            priority=str(args["priority"]),
            payload=dict(args.get("payload") or {}),
            source="story_agent",
        )
        if created:
            public = StoryMemory.public_proposal(proposal)
            self.proposals_created.append(public)
            summary = (
                f"Created a {public['task_type']} proposal for the {public['owner_agent']} "
                f"agent: {public['description']}"
            )
            data = {"proposal": public, "created": True}
        else:
            summary = (
                "An equivalent proposal already exists, so it was not duplicated: "
                f"{args['description']}"
            )
            data = {"proposal": StoryMemory.public_proposal(proposal) if proposal else {}, "created": False}
        return ToolResult(
            tool_call_id=tool_call_id,
            name="propose_task",
            ok=True,
            summary=summary,
            data=data,
        )

    def _execute_finish(self, tool_call_id: str, args: Mapping[str, Any]) -> ToolResult:
        self.finished = True
        self.finish_payload = {
            "summary": str(args["summary"]),
            "outcome": str(args["outcome"]),
        }
        return ToolResult(
            tool_call_id=tool_call_id,
            name="finish",
            ok=True,
            summary=f"Conversation ended ({self.finish_payload['outcome']}): {args['summary']}",
            data=dict(self.finish_payload),
        )
