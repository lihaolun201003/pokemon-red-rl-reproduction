"""Constrained tool layer for the DeepSeek Pokémon Red agent.

The model cannot run code: everything it can do is one of the eleven validated
tool calls in :data:`TOOL_SCHEMAS`.  This module owns

* the OpenAI-compatible function-calling schemas (names, descriptions, strict
  JSON-schema parameters with enums/min/max),
* strict argument validation (unknown names/keys, wrong types, out-of-range
  values and oversized batches are rejected with an error *result*, never an
  exception into the runtime),
* execution against the injected :class:`~llm_agent.emulator.GameBoyEmulator`,
  the agent's own memory and the :class:`~llm_agent.contracts.RuntimeState`,
* compact, factual summaries fed back to the model plus audit screenshots.

Interfaces are injected by the runtime so the layer stays independent and
testable (``emulator``, ``memory``, ``state``, ``screen_dir``,
``save_progress``, ``restore_progress``, optional ``screen_writer``/``cfg``).

Hard rules: no RAM reads, no hidden game state, no arbitrary code.  A tool call
that fails returns ``ToolResult(ok=False, error=...)`` and is recorded in the
runtime history; it never propagates an exception.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from llm_agent.config import AgentConfig
from llm_agent.contracts import (
    BUTTONS,
    DIRECTIONS,
    DISCOVERY_KINDS,
    MAX_ACTIONS_PER_CALL,
    MAX_DISCOVERY_CHARS,
    MAX_GOAL_CHARS,
    MAX_HISTORY_ITEMS,
    MAX_MEMORY_QUERY_CHARS,
    MAX_PLACE_CHARS,
    MAX_REASON_CHARS,
    MAX_WAIT_FRAMES,
    MAX_WALK_STEPS,
    TOOL_NAMES,
    ActionOutcome,
    ActionStep,
    Frame,
    ScreenEvent,
    SIGNIFICANT_EVENTS,
    ToolCall,
    ToolExecutionError,
    ToolResult,
    ToolValidationError,
)
from llm_agent.emulator import GameBoyEmulator
from llm_agent.observation import ScreenAnalysis, analyze, save_png

#: checkpoint ids accepted by ``restore_checkpoint``: "latest" or a checkpoint id
_CHECKPOINT_RE = re.compile(r"checkpoint_[A-Za-z0-9_.-]{1,60}")
#: default number of steps for ``walk``
DEFAULT_WALK_STEPS: int = 10
#: maximum number of memory entries returned by ``inspect_memory``
MAX_MEMORY_RESULTS: int = 50
#: default number of history entries for ``get_recent_history``
DEFAULT_HISTORY_LIMIT: int = 10


# --------------------------------------------------------------------------- #
# OpenAI function-calling schemas (exactly the eleven tools in contracts)
# --------------------------------------------------------------------------- #

def _object_schema(properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    """Build a strict JSON-schema object (no additional properties allowed)."""
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


def _button_property(description: str = "Game Boy button to press.") -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list(BUTTONS),
        "description": description,
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _function_schema(
        "observe_game",
        "Look at the current Game Boy screen again. Returns a short pixel-only text analysis "
        "of the screenshot (what kind of screen it is, how much changed, a stable signature). "
        "Use it after a batch when you need a fresh, detailed look before deciding.",
        _object_schema({}),
    ),
    _function_schema(
        "press_button",
        "Press one Game Boy button once (a short real press, then the game advances). "
        "The returned analysis shows whether the screen changed and what kind of screen it is now.",
        _object_schema(
            {
                "button": _button_property("Button to press."),
                "frames": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": MAX_WAIT_FRAMES,
                    "description": "Optional total frames for this action (default 24).",
                },
            },
            required=("button",),
        ),
    ),
    _function_schema(
        "execute_actions",
        "Press a short sequence of Game Boy buttons in one call (cheaper than many single presses). "
        "Execution stops early (and tells you why) when an important screen event happens.",
        _object_schema(
            {
                "actions": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(BUTTONS)},
                    "minItems": 1,
                    "maxItems": MAX_ACTIONS_PER_CALL,
                    "description": f"Ordered buttons to press (1..{MAX_ACTIONS_PER_CALL}).",
                },
                "stop_on_event": {
                    "type": "boolean",
                    "default": True,
                    "description": "Stop the batch early when the screen changes significantly.",
                },
            },
            required=("actions",),
        ),
    ),
    _function_schema(
        "wait_frames",
        f"Advance the game without pressing any button ({1}..{MAX_WAIT_FRAMES} frames). "
        "Use it for animations, fades, scrolling text that advances by itself, or transitions.",
        _object_schema(
            {
                "frames": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WAIT_FRAMES,
                    "description": "Number of frames to advance.",
                },
            },
            required=("frames",),
        ),
    ),
    _function_schema(
        "walk",
        "Keep pressing one direction until you are blocked (two identical screens in a row), a significant "
        "screen event happens, or the step limit is reached. The efficient way to cross a corridor or road.",
        _object_schema(
            {
                "direction": {
                    "type": "string",
                    "enum": list(DIRECTIONS),
                    "description": "Direction to walk.",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WALK_STEPS,
                    "default": DEFAULT_WALK_STEPS,
                    "description": f"Maximum button presses (1..{MAX_WALK_STEPS}).",
                },
                "stop_on_event": {
                    "type": "boolean",
                    "default": True,
                    "description": "Stop early on significant screen events or when blocked.",
                },
            },
            required=("direction",),
        ),
    ),
    _function_schema(
        "inspect_memory",
        "Search your own recorded notes (places, discoveries, dialogues, lessons) and get the compact "
        "memory context. Use it when returning to a known area or when unsure what you already know.",
        _object_schema(
            {
                "query": {
                    "type": "string",
                    "maxLength": MAX_MEMORY_QUERY_CHARS,
                    "description": "Optional keywords to search for (empty = most recent entries).",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_MEMORY_RESULTS,
                    "default": DEFAULT_HISTORY_LIMIT,
                    "description": "Maximum matching entries to return.",
                },
            }
        ),
    ),
    _function_schema(
        "record_discovery",
        "Save a fact you actually observed on the screen (a place, an NPC, a dialogue, a mechanic, an item, "
        "a battle lesson, a quest hint …) to your long-term memory. Only record what you saw.",
        _object_schema(
            {
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_DISCOVERY_CHARS,
                    "description": "One or two factual sentences about what you observed.",
                },
                "kind": {
                    "type": "string",
                    "enum": list(DISCOVERY_KINDS),
                    "default": "other",
                    "description": "Category of the discovery.",
                },
                "place": {
                    "type": "string",
                    "maxLength": MAX_PLACE_CHARS,
                    "description": "Optional place label (only if you know it from what you saw).",
                },
            },
            required=("text",),
        ),
    ),
    _function_schema(
        "set_goal",
        "Set or update your current objective (and the reason for it). Keep the goal short and concrete; "
        "update it whenever it changes or is completed.",
        _object_schema(
            {
                "goal": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_GOAL_CHARS,
                    "description": "New short-term objective.",
                },
                "reason": {
                    "type": "string",
                    "maxLength": MAX_REASON_CHARS,
                    "description": "Optional reason / evidence for this goal.",
                },
            },
            required=("goal",),
        ),
    ),
    _function_schema(
        "save_progress",
        "Save a legal emulator checkpoint (plus a memory snapshot) so you can safely restore this position later. "
        "Do it before dangerous transitions or long journeys.",
        _object_schema(
            {
                "reason": {
                    "type": "string",
                    "maxLength": MAX_REASON_CHARS,
                    "description": "Optional short reason for the checkpoint.",
                },
            }
        ),
    ),
    _function_schema(
        "restore_checkpoint",
        "Reload a saved emulator checkpoint (last resort when badly stuck). Use 'latest' for the newest one. "
        "After restoring, re-verify the surroundings before trusting recent notes.",
        _object_schema(
            {
                "checkpoint": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 80,
                    "default": "latest",
                    "description": "'latest' or a checkpoint id returned by save_progress.",
                },
            }
        ),
    ),
    _function_schema(
        "get_recent_history",
        "Review the most recent actions and tool results (compact recent history). Use it when you feel "
        "stuck, repeated a mistake, or need to remember what you were doing.",
        _object_schema(
            {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_HISTORY_ITEMS,
                    "default": DEFAULT_HISTORY_LIMIT,
                    "description": "How many recent entries to show.",
                },
            }
        ),
    ),
]

#: name -> schema for tests/inspection
TOOL_SCHEMAS_BY_NAME: dict[str, dict[str, Any]] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}


# --------------------------------------------------------------------------- #
# Strict argument validation helpers
# --------------------------------------------------------------------------- #

def _check_keys(arguments: Mapping[str, Any], *, required: Sequence[str], optional: Sequence[str]) -> None:
    """Reject unknown keys and missing required keys."""
    allowed = set(required) | set(optional)
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ToolValidationError(
            f"unknown argument(s) {unknown}; allowed argument(s): {sorted(allowed)}"
        )
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolValidationError(f"missing required argument(s): {missing}")


def _string_arg(
    arguments: Mapping[str, Any],
    key: str,
    *,
    max_length: int,
    min_length: int = 1,
    strip: bool = True,
) -> str | None:
    if key not in arguments:
        return None
    value = arguments[key]
    if not isinstance(value, str):
        raise ToolValidationError(f"{key!r} must be a string, got {type(value).__name__}")
    text = value.strip() if strip else value
    if len(text) < min_length:
        raise ToolValidationError(f"{key!r} must not be empty")
    if len(text) > max_length:
        raise ToolValidationError(f"{key!r} is too long: {len(text)} > {max_length} characters")
    return text


def _int_arg(arguments: Mapping[str, Any], key: str, *, minimum: int, maximum: int) -> int | None:
    if key not in arguments:
        return None
    value = arguments[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError(f"{key!r} must be an integer, got {type(value).__name__}")
    if not minimum <= value <= maximum:
        raise ToolValidationError(f"{key!r} must be in {minimum}..{maximum}, got {value}")
    return value


def _bool_arg(arguments: Mapping[str, Any], key: str) -> bool | None:
    if key not in arguments:
        return None
    value = arguments[key]
    if not isinstance(value, bool):
        raise ToolValidationError(f"{key!r} must be a boolean, got {type(value).__name__}")
    return value


def _button_value(value: Any, *, argument: str = "button") -> str:
    """Validate and normalize a button argument (case-insensitive enum)."""
    if not isinstance(value, str):
        raise ToolValidationError(f"{argument!r} must be a string, got {type(value).__name__}")
    normalized = value.strip().lower()
    if normalized not in BUTTONS:
        raise ToolValidationError(
            f"invalid {argument} {value!r}; allowed buttons: {', '.join(BUTTONS)}"
        )
    return normalized


def _direction_value(value: Any) -> str:
    if not isinstance(value, str):
        raise ToolValidationError(f"'direction' must be a string, got {type(value).__name__}")
    normalized = value.strip().lower()
    if normalized not in DIRECTIONS:
        raise ToolValidationError(
            f"invalid direction {value!r}; allowed directions: {', '.join(DIRECTIONS)}"
        )
    return normalized


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

class ToolExecutor:
    """Execute validated model tool calls against the injected game environment.

    The runtime constructs exactly one executor and reuses it for the whole run;
    all dependencies are injected so the class is unit-testable without the real
    ROM or a real network/API.
    """

    def __init__(
        self,
        *,
        emulator: GameBoyEmulator,
        memory: Any,
        state: Any,
        screen_dir: str | Path,
        save_progress: Callable[[str], str],
        restore_progress: Callable[[str], str],
        screen_writer: Callable[[Frame], str] | None = None,
        cfg: AgentConfig | None = None,
    ) -> None:
        if emulator is None:
            raise ValueError("ToolExecutor requires an emulator instance")
        if state is None:
            raise ValueError("ToolExecutor requires a RuntimeState")
        if not callable(save_progress):
            raise ValueError("save_progress must be callable")
        if not callable(restore_progress):
            raise ValueError("restore_progress must be callable")
        if screen_writer is not None and not callable(screen_writer):
            raise ValueError("screen_writer must be callable or None")

        self.emulator = emulator
        self.memory = memory
        self.state = state
        self.screen_dir = Path(screen_dir) if screen_dir is not None else Path("screens")
        self.save_progress = save_progress
        self.restore_progress = restore_progress
        self.screen_writer = screen_writer
        self.cfg = cfg if cfg is not None else AgentConfig()
        self.max_summary_chars = int(getattr(self.cfg.api, "max_tool_result_chars", 700) or 700)

        #: last captured frame / its pixel analysis (read by runtime/guard/viewer)
        self.last_frame: Frame | None = None
        self.last_analysis: ScreenAnalysis | None = None
        #: last executed action batch (read by the loop guard)
        self.last_outcome: ActionOutcome | None = None
        #: name of the last tool call seen by :meth:`execute`
        self.last_tool: str = ""

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def last_observation(self) -> tuple[Frame, ScreenAnalysis] | None:
        """Last captured frame and analysis, or ``None`` before any observation."""
        if self.last_frame is None or self.last_analysis is None:
            return None
        return self.last_frame, self.last_analysis

    def execute(self, call: ToolCall) -> ToolResult:
        """Validate and execute one tool call.

        Never raises for a bad call: validation errors and execution failures are
        returned as ``ToolResult(ok=False, error=...)`` with ``needs_llm=True`` so
        the runtime can stop the batch and let the model correct itself.
        """
        name = str(getattr(call, "name", "") or "")
        call_id = str(getattr(call, "id", "") or "")
        self.last_tool = name
        try:
            arguments = self.validate(name, getattr(call, "arguments", None) or {})
            handler = getattr(self, f"_handle_{name}")
            return handler(call_id, arguments)
        except ToolValidationError as exc:
            return self._error_result(call_id, name, str(exc))
        except Exception as exc:  # noqa: BLE001 - the tool layer must never crash the run
            return self._error_result(call_id, name, f"{type(exc).__name__}: {exc}")

    def validate(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Strictly validate one tool call's arguments and return normalized values.

        Raises :class:`ToolValidationError` for unknown tools, unknown/missing
        keys, wrong types or out-of-range values.  Does not touch the emulator.
        """
        if not isinstance(name, str) or not name:
            raise ToolValidationError("tool name must be a non-empty string")
        if name not in TOOL_NAMES:
            raise ToolValidationError(
                f"unknown tool {name!r}; available tools: {', '.join(TOOL_NAMES)}"
            )
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise ToolValidationError(
                f"arguments for {name} must be a JSON object, got {type(arguments).__name__}"
            )
        validator = getattr(self, f"_validate_{name}")
        return validator(dict(arguments))

    # ------------------------------------------------------------------ #
    # Per-tool validation (strict, no side effects)
    # ------------------------------------------------------------------ #
    def _validate_observe_game(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=(), optional=())
        return {}

    def _validate_press_button(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=("button",), optional=("frames",))
        normalized: dict[str, Any] = {"button": _button_value(arguments["button"])}
        frames = _int_arg(arguments, "frames", minimum=2, maximum=MAX_WAIT_FRAMES)
        if frames is not None:
            normalized["frames"] = frames
        return normalized

    def _validate_execute_actions(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=("actions",), optional=("stop_on_event",))
        raw = arguments["actions"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ToolValidationError(
                f"'actions' must be a list of button names, got {type(raw).__name__}"
            )
        actions = list(raw)
        if not actions:
            raise ToolValidationError("'actions' must not be empty")
        if len(actions) > MAX_ACTIONS_PER_CALL:
            raise ToolValidationError(
                f"'actions' has {len(actions)} items; at most {MAX_ACTIONS_PER_CALL} are allowed"
            )
        normalized: dict[str, Any] = {
            "actions": [_button_value(item, argument="actions[i]") for item in actions]
        }
        stop_on_event = _bool_arg(arguments, "stop_on_event")
        if stop_on_event is not None:
            normalized["stop_on_event"] = stop_on_event
        return normalized

    def _validate_wait_frames(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=("frames",), optional=())
        frames = _int_arg(arguments, "frames", minimum=1, maximum=MAX_WAIT_FRAMES)
        if frames is None:  # pragma: no cover - required key checked above
            raise ToolValidationError("'frames' is required")
        return {"frames": frames}

    def _validate_walk(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=("direction",), optional=("max_steps", "stop_on_event"))
        normalized: dict[str, Any] = {"direction": _direction_value(arguments["direction"])}
        max_steps = _int_arg(arguments, "max_steps", minimum=1, maximum=MAX_WALK_STEPS)
        if max_steps is not None:
            normalized["max_steps"] = max_steps
        stop_on_event = _bool_arg(arguments, "stop_on_event")
        if stop_on_event is not None:
            normalized["stop_on_event"] = stop_on_event
        return normalized

    def _validate_inspect_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=(), optional=("query", "limit"))
        normalized: dict[str, Any] = {}
        query = _string_arg(
            arguments, "query", max_length=MAX_MEMORY_QUERY_CHARS, min_length=0
        )
        if query:
            normalized["query"] = query
        limit = _int_arg(arguments, "limit", minimum=1, maximum=MAX_MEMORY_RESULTS)
        if limit is not None:
            normalized["limit"] = limit
        return normalized

    def _validate_record_discovery(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=("text",), optional=("kind", "place"))
        text = _string_arg(arguments, "text", max_length=MAX_DISCOVERY_CHARS)
        if text is None:  # pragma: no cover - required key checked above
            raise ToolValidationError("'text' is required")
        normalized: dict[str, Any] = {"text": text}
        kind = _string_arg(arguments, "kind", max_length=32)
        if kind is not None:
            if kind not in DISCOVERY_KINDS:
                raise ToolValidationError(
                    f"invalid kind {kind!r}; allowed kinds: {', '.join(DISCOVERY_KINDS)}"
                )
            normalized["kind"] = kind
        place = _string_arg(arguments, "place", max_length=MAX_PLACE_CHARS, min_length=0)
        if place:
            normalized["place"] = place
        return normalized

    def _validate_set_goal(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=("goal",), optional=("reason",))
        goal = _string_arg(arguments, "goal", max_length=MAX_GOAL_CHARS)
        if goal is None:  # pragma: no cover - required key checked above
            raise ToolValidationError("'goal' is required")
        normalized: dict[str, Any] = {"goal": goal}
        reason = _string_arg(arguments, "reason", max_length=MAX_REASON_CHARS, min_length=0)
        if reason:
            normalized["reason"] = reason
        return normalized

    def _validate_save_progress(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=(), optional=("reason",))
        normalized: dict[str, Any] = {}
        reason = _string_arg(arguments, "reason", max_length=MAX_REASON_CHARS, min_length=0)
        if reason:
            normalized["reason"] = reason
        return normalized

    def _validate_restore_checkpoint(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=(), optional=("checkpoint",))
        checkpoint = _string_arg(arguments, "checkpoint", max_length=80)
        if checkpoint is None:
            return {"checkpoint": "latest"}
        if checkpoint != "latest" and not _CHECKPOINT_RE.fullmatch(checkpoint):
            raise ToolValidationError(
                f"invalid checkpoint id {checkpoint!r}; use 'latest' or an id like 'checkpoint_000000100'"
            )
        return {"checkpoint": checkpoint}

    def _validate_get_recent_history(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _check_keys(arguments, required=(), optional=("limit",))
        limit = _int_arg(arguments, "limit", minimum=1, maximum=MAX_HISTORY_ITEMS)
        return {"limit": limit if limit is not None else DEFAULT_HISTORY_LIMIT}

    # ------------------------------------------------------------------ #
    # Tool handlers: observation
    # ------------------------------------------------------------------ #
    def _handle_observe_game(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        frame = self.emulator.observe()
        analysis = analyze(frame, self.last_frame)
        self._remember_observation(frame, analysis)
        self._stamp_action("observe_game")
        self._refresh_memory_stats()
        summary = (
            f"Observed frame {frame.frame_index}: event={analysis.event.value}, "
            f"changed={analysis.changed_ratio:.2f}, signature={analysis.signature[:8]}, "
            f"luma={analysis.luma_mean:.0f}/255, black={'yes' if analysis.is_black else 'no'}."
        )
        data = self._observation_data(frame, analysis)
        self._add_history("tool", f"observe_game: {analysis.summary}")
        return self._result(
            call_id,
            "observe_game",
            summary=summary,
            data=data,
            screens=self._persist_screen(frame),
            needs_llm=False,
        )

    # ------------------------------------------------------------------ #
    # Tool handlers: actions
    # ------------------------------------------------------------------ #
    def _handle_press_button(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        button = arguments["button"]
        frames = arguments.get("frames")
        previous = self._previous_frame()
        baseline_event = analyze(previous, None).event if previous is not None else ScreenEvent.NONE
        baseline_black = bool(previous.is_black) if previous is not None else False
        frame = self.emulator.press(button, frames=frames) if frames is not None else self.emulator.press(button)
        analysis = analyze(frame, previous)
        step = ActionStep(
            button=button,
            frame=frame,
            screen_changed=analysis.changed,
            changed_ratio=analysis.changed_ratio,
            event=analysis.event,
        )
        # Only an event *transition* interrupts the remaining batch: pressing A
        # while a dialogue is already open must continue that dialogue.
        interrupted = self._event_transition(baseline_event, analysis.event) or (
            analysis.is_black and not baseline_black
        )
        stopped_reason = "event" if interrupted else "completed"
        outcome = ActionOutcome(
            buttons=[button],
            steps=[step],
            start_frame=previous.frame_index if previous is not None else frame.frame_index,
            end_frame=frame.frame_index,
            stopped_reason=stopped_reason,
            events=[analysis.event] if analysis.event != ScreenEvent.NONE else [],
            summary=(
                f"Pressed {button}: changed={analysis.changed_ratio:.2f}, "
                f"event={analysis.event.value}, start_event={baseline_event.value}, "
                f"stopped={stopped_reason}"
            ),
        )
        self._remember_outcome(outcome)
        self._remember_observation(frame, analysis)
        self._count_actions(1)
        self._stamp_action(f"press_button {button}")
        self._refresh_memory_stats()
        summary = self._action_summary(
            f"Pressed {button} ({frames or self._action_frames()} frames)",
            executed=1,
            changed=1 if analysis.changed else 0,
            analysis=analysis,
            stopped_reason=stopped_reason,
            start_event=baseline_event,
        )
        data = self._action_data(frame, analysis, stopped_reason, changed_steps=1 if analysis.changed else 0)
        data["button"] = button
        data["start_event"] = baseline_event.value
        self._add_history("tool", f"press {button}: {analysis.summary}")
        return self._result(
            call_id,
            "press_button",
            summary=summary,
            data=data,
            screens=self._persist_screen(frame),
            needs_llm=self._needs_llm(analysis, stopped_reason),
        )

    def _handle_execute_actions(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        actions = list(arguments["actions"])
        stop_on_event = bool(arguments.get("stop_on_event", True))
        previous = self._previous_frame()
        baseline_event = analyze(previous, None).event if previous is not None else ScreenEvent.NONE
        outcome = self.emulator.run_batch(actions, stop_on_event=stop_on_event)
        self.last_outcome = outcome
        if outcome.steps:
            frame = outcome.final_frame
            analysis = analyze(frame, previous)
        else:  # pragma: no cover - run_batch always executes at least one step
            frame = self.emulator.observe()
            analysis = analyze(frame, previous)
        self._remember_observation(frame, analysis)
        executed = len(outcome.steps)
        changed = outcome.changed_steps
        final_event = outcome.steps[-1].event if outcome.steps else ScreenEvent.NONE
        self._count_actions(executed)
        self._stamp_action(f"execute_actions {actions[:8]}{'…' if len(actions) > 8 else ''}")
        self._refresh_memory_stats()
        summary = self._action_summary(
            f"Executed {executed}/{len(actions)} actions",
            executed=executed,
            changed=changed,
            analysis=analysis,
            stopped_reason=outcome.stopped_reason,
            events=[event.value for event in outcome.events],
            start_event=baseline_event,
        )
        data = self._action_data(
            frame, analysis, outcome.stopped_reason, changed_steps=changed,
        )
        data.update(
            {
                "requested_actions": actions,
                "executed_actions": executed,
                "events": [event.value for event in outcome.events],
                "final_event": final_event.value,
                "start_event": baseline_event.value,
            }
        )
        self._add_history(
            "tool",
            f"execute_actions {executed}/{len(actions)}: event={final_event.value}, "
            f"start_event={baseline_event.value}, stopped={outcome.stopped_reason}, changed={changed}",
        )
        return self._result(
            call_id,
            "execute_actions",
            summary=summary,
            data=data,
            screens=self._persist_screen(frame),
            needs_llm=self._needs_llm(analysis, outcome.stopped_reason),
        )

    def _handle_wait_frames(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        frames = int(arguments["frames"])
        previous = self._previous_frame()
        baseline_event = analyze(previous, None).event if previous is not None else ScreenEvent.NONE
        baseline_black = bool(previous.is_black) if previous is not None else False
        frame = self.emulator.wait(frames)
        analysis = analyze(frame, previous)
        step = ActionStep(
            button="wait",
            frame=frame,
            screen_changed=analysis.changed,
            changed_ratio=analysis.changed_ratio,
            event=analysis.event,
        )
        interrupted = self._event_transition(baseline_event, analysis.event) or (
            analysis.is_black and not baseline_black
        )
        stopped_reason = "event" if interrupted else "completed"
        outcome = ActionOutcome(
            buttons=[],
            steps=[step],
            start_frame=previous.frame_index if previous is not None else frame.frame_index,
            end_frame=frame.frame_index,
            stopped_reason=stopped_reason,
            events=[analysis.event] if analysis.event != ScreenEvent.NONE else [],
            summary=(
                f"Waited {frames} frames: changed={analysis.changed_ratio:.2f}, "
                f"event={analysis.event.value}, start_event={baseline_event.value}"
            ),
        )
        self._remember_outcome(outcome)
        self._remember_observation(frame, analysis)
        self._count_actions(1)
        self._stamp_action(f"wait_frames {frames}")
        self._refresh_memory_stats()
        summary = self._action_summary(
            f"Waited {frames} frames",
            executed=0,
            changed=1 if analysis.changed else 0,
            analysis=analysis,
            stopped_reason=stopped_reason,
            start_event=baseline_event,
        )
        data = self._action_data(frame, analysis, stopped_reason, changed_steps=1 if analysis.changed else 0)
        data["waited_frames"] = frames
        data["start_event"] = baseline_event.value
        self._add_history("tool", f"wait {frames}: {analysis.summary}")
        return self._result(
            call_id,
            "wait_frames",
            summary=summary,
            data=data,
            screens=self._persist_screen(frame),
            needs_llm=self._needs_llm(analysis, stopped_reason),
        )

    def _handle_walk(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        direction = arguments["direction"]
        max_steps = int(arguments.get("max_steps", DEFAULT_WALK_STEPS))
        stop_on_event = bool(arguments.get("stop_on_event", True))
        previous = self._previous_frame()
        start_frame = previous.frame_index if previous is not None else self.emulator.frame_count
        baseline_event = analyze(previous, None).event if previous is not None else ScreenEvent.NONE
        baseline_black = bool(previous.is_black) if previous is not None else False
        steps: list[ActionStep] = []
        events: list[ScreenEvent] = []
        identical_streak = 0
        stopped_reason = "max_steps"
        final_analysis: ScreenAnalysis | None = None
        for _ in range(max_steps):
            before = previous
            frame = self.emulator.press(direction)
            final_analysis = analyze(frame, before)
            step = ActionStep(
                button=direction,
                frame=frame,
                screen_changed=final_analysis.changed,
                changed_ratio=final_analysis.changed_ratio,
                event=final_analysis.event,
            )
            steps.append(step)
            if final_analysis.event != ScreenEvent.NONE:
                events.append(final_analysis.event)
            pixel_identical = before is not None and frame.sha256 == before.sha256
            identical_streak = identical_streak + 1 if pixel_identical else 0
            previous = frame
            if stop_on_event:
                # An event *transition* (new dialogue/event or dialogue ending)
                # interrupts; a persisting event such as an open menu does not.
                if self._event_transition(baseline_event, final_analysis.event) or (
                    final_analysis.is_black and not baseline_black
                ):
                    stopped_reason = "event"
                    break
                if identical_streak >= 2:
                    stopped_reason = "blocked"
                    break

        final_frame = steps[-1].frame
        assert final_analysis is not None  # max_steps >= 1 is guaranteed by validation
        changed_steps = sum(1 for step in steps if step.screen_changed)
        outcome = ActionOutcome(
            buttons=[direction] * len(steps),
            steps=steps,
            start_frame=start_frame,
            end_frame=final_frame.frame_index,
            stopped_reason=stopped_reason,
            events=events,
            summary=(
                f"Walked {direction} {len(steps)} step(s): {changed_steps} changed screen, "
                f"event={steps[-1].event.value}, start_event={baseline_event.value}, "
                f"stopped={stopped_reason}"
            ),
        )
        self._remember_outcome(outcome)
        self._remember_observation(final_frame, final_analysis)
        self._count_actions(len(steps))
        self._stamp_action(f"walk {direction} x{len(steps)}")
        self._refresh_memory_stats()
        summary = self._action_summary(
            f"Walked {direction} {len(steps)}/{max_steps} step(s)",
            executed=len(steps),
            changed=changed_steps,
            analysis=final_analysis,
            stopped_reason=stopped_reason,
            events=[event.value for event in events],
            start_event=baseline_event,
        )
        data = self._action_data(
            final_frame, final_analysis, stopped_reason, changed_steps=changed_steps
        )
        data.update(
            {
                "direction": direction,
                "steps": len(steps),
                "max_steps": max_steps,
                "events": [event.value for event in events],
                "start_event": baseline_event.value,
            }
        )
        self._add_history(
            "tool",
            f"walk {direction} x{len(steps)}: event={steps[-1].event.value}, "
            f"stopped={stopped_reason}, changed={changed_steps}",
        )
        return self._result(
            call_id,
            "walk",
            summary=summary,
            data=data,
            screens=self._persist_screen(final_frame),
            needs_llm=self._needs_llm(final_analysis, stopped_reason),
        )

    # ------------------------------------------------------------------ #
    # Tool handlers: memory / control
    # ------------------------------------------------------------------ #
    def _handle_inspect_memory(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "")
        limit = int(arguments.get("limit", int(getattr(self.cfg.memory, "retrieval_limit", 10))))
        entries: list[dict[str, Any]] = []
        search = getattr(self.memory, "search", None)
        if callable(search):
            results = search(query, limit)
            entries = [dict(item) for item in (results or []) if isinstance(item, Mapping)]
        max_chars = int(getattr(self.cfg.memory, "max_context_chars", 7000) or 7000)
        context = str(self._call_memory("render_context", query or None, max_chars=max_chars))
        header = f"Memory search ({len(entries)} match(es)"
        if query:
            header += f" for {query!r}"
        header += ")"
        summary = f"{header}:\n{context}" if context else f"{header}: (memory is empty)"
        data = {"query": query, "limit": limit, "matches": len(entries), "entries": entries[:limit]}
        self._add_history("tool", f"inspect_memory {query!r}: {len(entries)} match(es)")
        return self._result(
            call_id,
            "inspect_memory",
            summary=summary,
            data=data,
            needs_llm=False,
        )

    def _handle_record_discovery(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        text = arguments["text"]
        kind = str(arguments.get("kind") or "other")
        place = str(arguments.get("place") or "")
        entry = self._call_memory("record_discovery", text, kind=kind, place=place)
        entry_dict = dict(entry) if isinstance(entry, Mapping) else {"text": text, "kind": kind}
        stats = self._refresh_memory_stats()
        self._add_history("note", f"discovery ({kind}): {_clip(text, 160)}")
        discoveries = int(stats.get("discoveries") or getattr(self.state, "discoveries", 0))
        places = int(stats.get("places") or getattr(self.state, "places_discovered", 0))
        summary = (
            f"Discovery recorded ({kind}, id={entry_dict.get('id', '-')}): {_clip(text, 200)} "
            f"Memory now: {discoveries} discoveries, {places} places."
        )
        data = {
            "entry": entry_dict,
            "kind": kind,
            "place": place,
            "discoveries": discoveries,
            "places": places,
        }
        return self._result(
            call_id,
            "record_discovery",
            summary=summary,
            data=data,
            needs_llm=False,
        )

    def _handle_set_goal(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        goal = arguments["goal"]
        reason = str(arguments.get("reason") or "")
        record = self._call_memory("set_goal", goal, reason)
        try:
            self.state.goal = goal
            self.state.goal_reason = reason
            self.state.touch()
        except Exception:  # noqa: BLE001 - state is duck-typed
            pass
        self._add_history("note", f"goal set: {_clip(goal, 200)}")
        summary = f"Goal set: {goal}"
        if reason:
            summary += f" (reason: {_clip(reason, 160)})"
        data = {"goal": goal, "reason": reason, "memory": dict(record) if isinstance(record, Mapping) else {}}
        return self._result(call_id, "set_goal", summary=summary, data=data, needs_llm=False)

    def _handle_save_progress(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        reason = str(arguments.get("reason") or "agent_request")
        confirmation = self.save_progress(reason)
        confirmation_text = str(confirmation or "saved")
        self._add_history("checkpoint", f"save_progress ({reason}): {_clip(confirmation_text, 200)}")
        summary = f"Progress saved: {confirmation_text}"
        data = {"reason": reason, "confirmation": confirmation_text}
        return self._result(call_id, "save_progress", summary=summary, data=data, needs_llm=False)

    def _handle_restore_checkpoint(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        checkpoint = str(arguments.get("checkpoint") or "latest")
        confirmation = self.restore_progress(checkpoint)
        confirmation_text = str(confirmation or "restored")
        # The game state jumped: cached observations are stale.
        self.last_frame = None
        self.last_analysis = None
        self.last_outcome = None
        self._add_history("checkpoint", f"restore {checkpoint}: {_clip(confirmation_text, 200)}")
        summary = f"Checkpoint restored ({checkpoint}): {confirmation_text}"
        data = {"checkpoint": checkpoint, "confirmation": confirmation_text}
        return self._result(call_id, "restore_checkpoint", summary=summary, data=data, needs_llm=False)

    def _handle_get_recent_history(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        limit = int(arguments.get("limit", DEFAULT_HISTORY_LIMIT))
        recent = getattr(self.state, "recent_history", None)
        entries = list(recent(limit)) if callable(recent) else []
        lines = [f"[{entry.kind}] {_clip(entry.summary, 220)}" for entry in entries]
        body = "\n".join(lines) if lines else "(no history yet)"
        summary = f"Recent history (last {len(entries)} entries):\n{body}"
        data = {
            "limit": limit,
            "entries": [entry.to_dict() for entry in entries],
        }
        return self._result(call_id, "get_recent_history", summary=summary, data=data, needs_llm=False)

    # ------------------------------------------------------------------ #
    # Result / state / audit helpers
    # ------------------------------------------------------------------ #
    def _result(
        self,
        call_id: str,
        name: str,
        *,
        summary: str,
        data: dict[str, Any] | None = None,
        screens: list[str] | None = None,
        needs_llm: bool = False,
        ok: bool = True,
        error: str | None = None,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=str(call_id),
            name=name,
            ok=ok,
            summary=_clip(summary, self.max_summary_chars),
            data=dict(data or {}),
            screens=list(screens or []),
            error=error,
            needs_llm=bool(needs_llm),
        )

    def _error_result(self, call_id: str, name: str, message: str) -> ToolResult:
        text = _clip(str(message or "tool call failed"), 400)
        self._add_history("error", f"{name or 'tool'} rejected: {text}")
        error_text = f"{name}: {text}" if name else text
        return ToolResult(
            tool_call_id=str(call_id),
            name=str(name),
            ok=False,
            summary=f"Tool call failed: {text}",
            data={},
            screens=[],
            error=text,
            needs_llm=True,
        )

    def _previous_frame(self) -> Frame | None:
        """Frame to diff the next action against (fresh observation when needed)."""
        if self.last_frame is not None:
            return self.last_frame
        return self.emulator.observe()

    def _remember_observation(self, frame: Frame, analysis: ScreenAnalysis) -> None:
        self.last_frame = frame
        self.last_analysis = analysis
        try:
            self.state.frame_index = frame.frame_index
            self.state.touch()
        except Exception:  # noqa: BLE001 - state is duck-typed
            pass

    def _remember_outcome(self, outcome: ActionOutcome) -> None:
        self.last_outcome = outcome

    def _count_actions(self, count: int) -> None:
        """Update action counters after a batch/step completed."""
        try:
            self.state.buttons_pressed = int(getattr(self.state, "buttons_pressed", 0)) + int(count)
            self.state.action_batches = int(getattr(self.state, "action_batches", 0)) + 1
            self.state.touch()
        except Exception:  # noqa: BLE001
            pass

    def _stamp_action(self, description: str) -> None:
        try:
            self.state.last_action = description
            self.state.last_action_at = time.time()
            self.state.touch()
        except Exception:  # noqa: BLE001
            pass

    def _add_history(self, kind: str, summary: str) -> None:
        add = getattr(self.state, "add_history", None)
        if not callable(add):
            return
        try:
            add(kind, summary)
        except Exception:  # noqa: BLE001 - history is best effort
            pass

    def _call_memory(self, method: str, *args: Any, **kwargs: Any) -> Any:
        function = getattr(self.memory, method, None)
        if not callable(function):
            raise ToolExecutionError(f"memory does not implement {method}()")
        return function(*args, **kwargs)

    def _memory_stats_safe(self) -> dict[str, Any]:
        stats = getattr(self.memory, "stats", None)
        if not callable(stats):
            return {}
        try:
            result = stats()
        except Exception:  # noqa: BLE001 - memory stats must not break a tool
            return {}
        return dict(result) if isinstance(result, Mapping) else {}

    def _refresh_memory_stats(self) -> dict[str, Any]:
        """Mirror memory counters onto the shared RuntimeState; returns the stats."""
        stats = self._memory_stats_safe()
        if not stats:
            return stats
        try:
            discoveries = stats.get("discoveries")
            if discoveries is not None:
                self.state.discoveries = int(discoveries)
            places = stats.get("places")
            if places is not None:
                self.state.places_discovered = int(places)
            self.state.touch()
        except (AttributeError, TypeError, ValueError):
            pass
        return stats

    def _persist_screen(self, frame: Frame) -> list[str]:
        """Save an audit PNG (via injected writer or ``screen_dir``), never raises."""
        names: list[str] = []
        try:
            if callable(self.screen_writer):
                name = self.screen_writer(frame)
                if name:
                    names.append(str(name))
                return names
            self.screen_dir.mkdir(parents=True, exist_ok=True)
            filename = f"tool_{frame.frame_index:09d}_{frame.sha256[:8]}.png"
            save_png(frame, self.screen_dir / filename)
            names.append(filename)
        except Exception:  # noqa: BLE001 - audit screenshots are best effort
            return names
        return names

    def _action_frames(self) -> int:
        try:
            return int(self.cfg.emulator.action_frames)
        except AttributeError:  # pragma: no cover - defensive
            return 24

    @staticmethod
    def _event_transition(baseline: ScreenEvent, current: ScreenEvent) -> bool:
        """True when the screen event changed into/out of a significant event.

        A persisting significant event (e.g. ``DIALOG`` -> ``DIALOG`` while
        reading several text pages) is not a transition and must not interrupt a
        batch.  ``DIALOG`` -> ``NONE`` or ``NONE`` -> ``DIALOG`` *is* a transition.
        """
        if current == baseline:
            return False
        return current in SIGNIFICANT_EVENTS or baseline in SIGNIFICANT_EVENTS

    def _needs_llm(self, analysis: ScreenAnalysis, stopped_reason: str) -> bool:
        """A significant/unexpected *transition* must interrupt the remaining batch.

        A merely persisting dialogue/menu/battle does NOT force a new decision:
        pressing A through several dialogue pages is one legitimate tool call.
        """
        if stopped_reason in ("event", "blocked"):
            return True
        if analysis.event == ScreenEvent.UNKNOWN:
            return True
        return bool(analysis.is_black)

    def _observation_data(self, frame: Frame, analysis: ScreenAnalysis) -> dict[str, Any]:
        return {
            "frame": frame.frame_index,
            "sha256": frame.sha256,
            "event": analysis.event.value,
            "changed": analysis.changed,
            "changed_ratio": round(analysis.changed_ratio, 4),
            "signature": analysis.signature,
            "is_black": analysis.is_black,
            "luma_mean": round(analysis.luma_mean, 3),
            "stats": analysis.stats,
        }

    def _action_data(
        self,
        frame: Frame,
        analysis: ScreenAnalysis,
        stopped_reason: str,
        *,
        changed_steps: int,
    ) -> dict[str, Any]:
        data = self._observation_data(frame, analysis)
        data.update(
            {
                "changed_steps": int(changed_steps),
                "stopped_reason": stopped_reason,
            }
        )
        return data

    def _action_summary(
        self,
        prefix: str,
        *,
        executed: int,
        changed: int,
        analysis: ScreenAnalysis,
        stopped_reason: str,
        events: Sequence[str] = (),
        start_event: ScreenEvent | None = None,
    ) -> str:
        parts = [
            f"{prefix}.",
            f"actions={executed}",
            f"changed_steps={changed}",
            f"event={analysis.event.value}",
            f"signature={analysis.signature[:8]}",
            f"black={'yes' if analysis.is_black else 'no'}",
            f"stopped={stopped_reason}",
        ]
        if start_event is not None:
            parts.append(f"start_event={start_event.value}")
        if events:
            unique: list[str] = []
            for name in events:
                if name not in unique:
                    unique.append(name)
            parts.append(f"events={','.join(unique)}")
        return " ".join(parts)

