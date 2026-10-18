"""Shared, dependency-free data contracts for the LLM game agent.

This module is the stable boundary between the emulator layer, the tool layer,
the memory layer, the DeepSeek API client, the loop guard and the runtime.

Hard rules encoded here:

* Buttons are limited to the 8 Game Boy buttons; nothing else can be pressed.
* Every model action is a validated tool call, never generated Python code.
* Screens are pixel observations only.  No RAM value, map id or story flag
  crosses this boundary — see :mod:`llm_agent.progress` for the separate
  verification-only probe.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque

import numpy as np

# --------------------------------------------------------------------------- #
# Constants shared by every layer
# --------------------------------------------------------------------------- #

BUTTONS: tuple[str, ...] = ("a", "b", "start", "select", "up", "down", "left", "right")
PRESSABLE_BUTTONS: tuple[str, ...] = BUTTONS
DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

SCREEN_WIDTH: int = 160
SCREEN_HEIGHT: int = 144
SCREEN_CHANNELS: int = 4  # PyBoy renders RGBA

MAX_ACTIONS_PER_CALL: int = 32
MAX_WAIT_FRAMES: int = 600
MAX_WALK_STEPS: int = 20
MAX_HISTORY_ITEMS: int = 32

MAX_GOAL_CHARS: int = 300
MAX_REASON_CHARS: int = 300
MAX_DISCOVERY_CHARS: int = 1200
MAX_PLACE_CHARS: int = 80
MAX_MEMORY_QUERY_CHARS: int = 200
MAX_DECISION_CHARS: int = 400

# Discovery kinds exposed to the model (validation enum).
DISCOVERY_KINDS: tuple[str, ...] = (
    "place",
    "npc",
    "dialogue",
    "mechanic",
    "item",
    "battle",
    "quest",
    "other",
)

# Tool names.  The tool layer owns the JSON schemas; the runtime only needs the
# names to route results and to audit calls.
TOOL_NAMES: tuple[str, ...] = (
    "observe_game",
    "press_button",
    "execute_actions",
    "wait_frames",
    "walk",
    "inspect_memory",
    "record_discovery",
    "set_goal",
    "save_progress",
    "restore_checkpoint",
    "get_recent_history",
    # Navigation v2 (high level; the navigation subsystem moves the player in
    # Python without an LLM call per step)
    "explore_current_area",
    "navigate_to",
    "get_navigation_status",
    "get_local_map",
    "get_world_map_summary",
    "pause_exploration",
    "resume_exploration",
)

#: Tools that drive the player through the navigation subsystem.
NAVIGATION_TOOLS: tuple[str, ...] = (
    "explore_current_area",
    "navigate_to",
    "get_navigation_status",
    "get_local_map",
    "get_world_map_summary",
    "pause_exploration",
    "resume_exploration",
)

#: Raw movement tools kept for compatibility (dialog/battle handling); when the
#: navigation subsystem is enabled their movement results are still fed to the
#: map exactly once through the executor callback.
RAW_MOVEMENT_TOOLS: tuple[str, ...] = (
    "press_button",
    "execute_actions",
    "walk",
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class AgentError(RuntimeError):
    """Base class for all agent errors."""


class ToolValidationError(AgentError):
    """A model tool call had an invalid name/arguments (recoverable)."""


class ToolExecutionError(AgentError):
    """A validated tool call failed while talking to the emulator/memory."""


class EmulatorError(AgentError):
    """PyBoy failed to start, tick, load or save state."""


class ApiError(AgentError):
    """Base class for DeepSeek API failures."""

    def __init__(self, message: str, *, status: int | None = None,
                 retryable: bool = False, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.body = body

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "message": str(self),
            "status": self.status,
            "retryable": self.retryable,
        }


class ApiAuthenticationError(ApiError):
    """401 — invalid API key.  Not retryable, run must pause for the user."""


class ApiBalanceError(ApiError):
    """402 — insufficient balance.  Not retryable, run must pause for the user."""


class ApiRateLimitError(ApiError):
    """429 — retryable with backoff."""


class ApiTimeoutError(ApiError):
    """Network timeout — retryable."""


class ApiResponseError(ApiError):
    """Malformed/unexpected response body."""


class BudgetExceededError(AgentError):
    """The configured cost/call budget is exhausted; pause safely."""


class CheckpointError(AgentError):
    """Checkpoint save/load/restore problem."""


class LockError(AgentError):
    """Another agent instance already controls this game state."""


class BlockedError(AgentError):
    """The run cannot continue automatically; state was saved for recovery."""

    def __init__(self, reason: str, *, recoverable_state: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.recoverable_state = recoverable_state or {}


class StopRequested(AgentError):
    """The user requested a safe stop (viewer, stop file, signal)."""


# --------------------------------------------------------------------------- #
# Observation primitives
# --------------------------------------------------------------------------- #

class ScreenEvent(str, Enum):
    """Coarse, pixel-only classification of what is on screen.

    Derived exclusively from the 160x144 framebuffer; never from RAM.  It is
    only a trigger hint for the runtime — the model always sees the real image.
    """

    NONE = "none"
    DIALOG = "dialog"
    MENU = "menu"
    BATTLE = "battle"
    MAP_CHANGE = "map_change"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


#: Events that should interrupt a movement batch and force a fresh LLM decision.
SIGNIFICANT_EVENTS: frozenset[ScreenEvent] = frozenset(
    {
        ScreenEvent.DIALOG,
        ScreenEvent.MENU,
        ScreenEvent.BATTLE,
        ScreenEvent.MAP_CHANGE,
        ScreenEvent.TRANSITION,
    }
)


@dataclass
class Frame:
    """One captured Game Boy frame plus observation-only metadata."""

    frame_index: int
    rgba: np.ndarray  # (144, 160, 4) uint8
    sha256: str
    captured_at: float
    is_black: bool
    luma_mean: float

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.rgba.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "sha256": self.sha256,
            "captured_at": self.captured_at,
            "is_black": self.is_black,
            "luma_mean": round(self.luma_mean, 3),
        }


@dataclass
class ActionStep:
    """Outcome of a single button press inside a batch."""

    button: str
    frame: Frame
    screen_changed: bool
    changed_ratio: float
    event: ScreenEvent = ScreenEvent.NONE


@dataclass
class ActionOutcome:
    """Outcome of a batch of button presses (one tool call)."""

    buttons: list[str]
    steps: list[ActionStep]
    start_frame: int
    end_frame: int
    stopped_reason: str = "completed"  # completed|event|blocked|max_steps|error
    events: list[ScreenEvent] = field(default_factory=list)
    summary: str = ""

    @property
    def changed_steps(self) -> int:
        return sum(1 for step in self.steps if step.screen_changed)

    @property
    def final_frame(self) -> Frame:
        if not self.steps:
            raise ToolExecutionError("ActionOutcome has no captured steps")
        return self.steps[-1].frame


# --------------------------------------------------------------------------- #
# Tool-call protocol (OpenAI-compatible function calling)
# --------------------------------------------------------------------------- #

@dataclass
class ToolCall:
    """A validated request from the model to execute one constrained tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}


@dataclass
class ToolResult:
    """Result of executing one tool call; fed back to the model as text."""

    tool_call_id: str
    name: str
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    screens: list[str] = field(default_factory=list)
    error: str | None = None
    #: set when the tool observed something that requires an immediate new decision
    needs_llm: bool = False

    def to_tool_message(self) -> dict[str, Any]:
        content = self.summary if self.ok else f"ERROR: {self.error or self.summary}"
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": content}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "ok": self.ok,
            "summary": self.summary,
            "data": self.data,
            "screens": self.screens,
            "error": self.error,
            "needs_llm": self.needs_llm,
        }


# --------------------------------------------------------------------------- #
# DeepSeek API accounting
# --------------------------------------------------------------------------- #

@dataclass
class ApiUsage:
    """Token usage returned by the API (source of truth for billing)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    reasoning_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> "ApiUsage":
        raw = raw or {}
        details = raw.get("completion_tokens_details") or raw.get("completion_tokens_details", {})
        prompt_details = raw.get("prompt_tokens_details") or {}
        cache_hit = raw.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens"))
        cache_miss = raw.get("prompt_cache_miss_tokens")
        if cache_miss is None and cache_hit is not None and raw.get("prompt_tokens"):
            cache_miss = max(int(raw["prompt_tokens"]) - int(cache_hit), 0)
        reasoning = None
        if isinstance(details, dict):
            reasoning = details.get("reasoning_tokens")
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
            cache_hit_tokens=None if cache_hit is None else int(cache_hit),
            cache_miss_tokens=None if cache_miss is None else int(cache_miss),
            reasoning_tokens=None if reasoning is None else int(reasoning),
            extra={k: v for k, v in raw.items() if k not in {
                "prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                "prompt_tokens_details", "completion_tokens_details",
            }},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class ModelReply:
    """One parsed DeepSeek chat-completion reply."""

    content: str
    tool_calls: list[ToolCall]
    usage: ApiUsage
    model: str
    latency_s: float = 0.0
    finish_reason: str = ""
    reasoning_content: str | None = None
    response_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def rationale(self, limit: int = MAX_DECISION_CHARS) -> str:
        text = (self.content or "").strip()
        if not text and self.reasoning_content:
            text = (self.reasoning_content or "").strip()[:limit]
        return text[:limit]


@dataclass
class BudgetState:
    """Persistent cost/usage accounting for the API budget."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cost_usd: float = 0.0
    consecutive_failures: int = 0
    started_at: float = 0.0
    last_call_at: float = 0.0
    #: timestamps (unix) of recent paid calls, for calls/hour limiting
    call_timestamps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "consecutive_failures": self.consecutive_failures,
            "started_at": self.started_at,
            "last_call_at": self.last_call_at,
            "call_timestamps": self.call_timestamps[-400:],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BudgetState":
        data = data or {}
        state = cls()
        for name in (
            "calls", "prompt_tokens", "completion_tokens", "total_tokens",
            "cache_hit_tokens", "consecutive_failures", "started_at",
            "last_call_at",
        ):
            if name in data and data[name] is not None:
                setattr(state, name, type(getattr(state, name))(data[name]))
        state.cost_usd = float(data.get("cost_usd") or 0.0)
        stamps = data.get("call_timestamps") or []
        state.call_timestamps = [float(t) for t in stamps if isinstance(t, (int, float))]
        return state


# --------------------------------------------------------------------------- #
# Runtime shared state (tools mutate counters; viewer/runtime read them)
# --------------------------------------------------------------------------- #

@dataclass
class HistoryEntry:
    kind: str  # tool|event|error|checkpoint|api|note
    summary: str
    at: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "summary": self.summary, "at": self.at, "data": self.data}


@dataclass
class RuntimeState:
    """Mutable state shared by runtime, tools and the viewer."""

    run_id: str = ""
    goal: str = ""
    goal_reason: str = ""
    status: str = "starting"  # starting|running|waiting_api|acting|paused|saving|blocked|finished|stopped
    paused: bool = False
    stop_requested: bool = False
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    frame_index: int = 0
    llm_calls: int = 0
    llm_turns: int = 0
    buttons_pressed: int = 0
    action_batches: int = 0
    checkpoints: int = 0
    places_discovered: int = 0
    discoveries: int = 0
    api_failures: int = 0
    consecutive_failures: int = 0
    last_action: str = ""
    last_action_at: float = 0.0
    last_decision: str = ""
    last_decision_at: float = 0.0
    last_trigger: str = ""
    last_error: str = ""
    blocked_reason: str = ""
    loop_warnings: list[str] = field(default_factory=list)
    history: Deque[HistoryEntry] = field(default_factory=lambda: _new_history())

    def touch(self) -> None:
        self.updated_at = time.time()

    def add_history(self, kind: str, summary: str, **data: Any) -> None:
        self.history.append(HistoryEntry(kind=kind, summary=summary, data=data))
        self.touch()

    def recent_history(self, n: int = 10) -> list[HistoryEntry]:
        items = list(self.history)
        return items[-max(1, min(n, MAX_HISTORY_ITEMS)):]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "goal_reason": self.goal_reason,
            "status": self.status,
            "paused": self.paused,
            "started_at": self.started_at,
            "frame_index": self.frame_index,
            "llm_calls": self.llm_calls,
            "llm_turns": self.llm_turns,
            "buttons_pressed": self.buttons_pressed,
            "action_batches": self.action_batches,
            "checkpoints": self.checkpoints,
            "places_discovered": self.places_discovered,
            "discoveries": self.discoveries,
            "api_failures": self.api_failures,
            "consecutive_failures": self.consecutive_failures,
            "last_action": self.last_action,
            "last_decision": self.last_decision,
            "last_trigger": self.last_trigger,
            "last_error": self.last_error,
            "blocked_reason": self.blocked_reason,
            "history": [entry.to_dict() for entry in self.history],
        }

    def stats(self) -> dict[str, Any]:
        """Compact stats for prompts/viewer (no hidden state)."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "frame": self.frame_index,
            "llm_calls": self.llm_calls,
            "turns": self.llm_turns,
            "buttons": self.buttons_pressed,
            "batches": self.action_batches,
            "checkpoints": self.checkpoints,
            "places": self.places_discovered,
            "discoveries": self.discoveries,
            "api_failures": self.api_failures,
            "goal": self.goal,
        }


def _new_history() -> Deque[HistoryEntry]:
    from collections import deque

    return deque(maxlen=64)


# --------------------------------------------------------------------------- #
# Callback protocols injected by the runtime into the tool layer
# --------------------------------------------------------------------------- #

#: save a checkpoint for a reason, return a human-readable checkpoint id
SaveProgressFn = Callable[[str], str]
#: restore a checkpoint ("latest" or id), return a human-readable confirmation
RestoreProgressFn = Callable[[str], str]
#: persist a captured frame for audit; returns the relative file name
ScreenWriter = Callable[[Frame], str]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_rgba(rgba: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(rgba).tobytes())
