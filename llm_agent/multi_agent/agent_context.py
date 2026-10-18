"""Per-agent isolated context and low-level action primitives.

Each specialised agent (Navigation / Story / Pokémon) owns exactly one
:class:`AgentContext`:

* its own system prompt, tool schemas, conversation window and recent results,
* its own professional memory object (reusing :class:`llm_agent.memory.AgentMemory`),
* its own active task and per-agent statistics.

The context never talks to PyBoy directly.  All button presses go through
:class:`ActionPrimitives`, which wraps the :class:`GuardedEmulator` issued by the
:class:`ControlArbiter` — so an agent that does not hold the lease cannot press a
button no matter how it calls the action layer.

There is deliberately **no game knowledge** in this module: no routes, no story
order, no species data.  Agents learn only from screenshots they actually saw.
"""

from __future__ import annotations

import base64
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from llm_agent.contracts import (
    ActionOutcome,
    Frame,
    ModelReply,
    ScreenEvent,
    ToolCall,
)
from llm_agent.multi_agent.contracts import (
    AgentRole,
    ControlDeniedError,
    Task,
    now_ts,
)
from llm_agent.observation import (
    ScreenAnalysis,
    analyze,
    encode_for_api,
    save_png,
)

#: Buttons an agent may ever press (validated again by the emulator).
AGENT_BUTTONS: frozenset[str] = frozenset(
    {"a", "b", "up", "down", "left", "right", "start", "select"}
)


# --------------------------------------------------------------------------- #
# Conversation context
# --------------------------------------------------------------------------- #

@dataclass
class AgentContextConfig:
    """Bounds that keep one agent's prompt cheap and self-consistent."""

    max_conversation_messages: int = 10
    max_recent_results: int = 8
    max_user_text_chars: int = 8000
    max_result_chars: int = 700
    max_memory_chars: int = 2500
    max_total_prompt_chars: int = 36000


@dataclass
class AgentContext:
    """The isolated working memory of one professional agent."""

    agent: AgentRole
    system_prompt: str
    tools: tuple[dict[str, Any], ...] = ()
    memory: Any | None = None
    config: AgentContextConfig = field(default_factory=AgentContextConfig)
    #: alternating user/assistant/tool messages (no system message inside)
    messages: list[dict[str, Any]] = field(default_factory=list)
    recent_results: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    active_task: Task | None = None
    branch_id: str = ""
    stats: dict[str, Any] = field(
        default_factory=lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "latency_s": 0.0,
            "errors": 0,
            "actions": 0,
            "denied": 0,
            "decisions": 0,
            "created_at": 0.0,
            "last_call_at": 0.0,
        }
    )

    # -- lifecycle ------------------------------------------------------- #
    def __post_init__(self) -> None:
        if not isinstance(self.agent, AgentRole):
            raise ValueError("AgentContext.agent must be an AgentRole")
        if not self.stats.get("created_at"):
            self.stats["created_at"] = now_ts()
        if self.recent_results.maxlen != self.config.max_recent_results:
            self.recent_results = deque(self.recent_results, maxlen=self.config.max_recent_results)

    def begin_task(self, task: Task | None, *, keep_history: bool = False) -> None:
        """Start a fresh task: reset per-task conversation unless asked otherwise."""
        self.active_task = task
        if not keep_history:
            self.messages.clear()
            self.recent_results.clear()

    def end_task(self, *, summary: str = "") -> None:
        if summary:
            self.note_result(summary)
        self.active_task = None

    # -- conversation ---------------------------------------------------- #
    def add_user(
        self,
        text: str,
        *,
        image_png: bytes | None = None,
        mime: str = "image/png",
        detail: str = "original",
        previous_text: str = "",
        previous_image_png: bytes | None = None,
        previous_mime: str = "image/png",
    ) -> None:
        content = self._user_content(
            text=text,
            image_png=image_png,
            mime=mime,
            detail=detail,
            previous_text=previous_text,
            previous_image_png=previous_image_png,
            previous_mime=previous_mime,
        )
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, reply: ModelReply | str) -> None:
        if isinstance(reply, ModelReply):
            message: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
            if reply.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": _json_args(call.arguments),
                        },
                    }
                    for call in reply.tool_calls
                ]
            self.messages.append(message)
        else:
            self.messages.append({"role": "assistant", "content": str(reply)})
        self.stats["decisions"] = int(self.stats.get("decisions", 0)) + 1
        self._trim()

    def add_tool_result(self, call: ToolCall | None, text: str, *, ok: bool = True) -> None:
        call_id = getattr(call, "id", "") if call is not None else ""
        content = str(text or "")
        if not ok:
            content = f"ERROR: {content}"
        self.messages.append(
            {"role": "tool", "tool_call_id": call_id or "tool", "content": content[: self.config.max_result_chars * 3]}
        )
        self._trim()

    def build_messages(
        self,
        *,
        user_text: str = "",
        image_png: bytes | None = None,
        mime: str = "image/png",
        detail: str = "original",
        include_history: bool = True,
        extra_system: str = "",
    ) -> list[dict[str, Any]]:
        """Assemble the exact OpenAI-style message list for the next call."""
        system = self.system_prompt
        memory_text = self.render_memory()
        if memory_text:
            system = f"{system}\n\n{memory_text}"
        if self.active_task is not None:
            system = f"{system}\n\n{self.render_task()}"
        if extra_system:
            system = f"{system}\n\n{extra_system}"
        if len(system) > self.config.max_total_prompt_chars:
            system = system[: self.config.max_total_prompt_chars]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if include_history:
            messages.extend(self._history_payload())
        if user_text or image_png:
            messages.append(
                {
                    "role": "user",
                    "content": self._user_content(
                        text=user_text,
                        image_png=image_png,
                        mime=mime,
                        detail=detail,
                    ),
                }
            )
        return messages

    # -- rendering helpers ----------------------------------------------- #
    def render_memory(self) -> str:
        if self.memory is None:
            return ""
        max_chars = int(self.config.max_memory_chars)
        for method in ("render_context", "render_for_prompt", "render_brief"):
            func = getattr(self.memory, method, None)
            if callable(func):
                try:
                    try:
                        text = func(max_chars=max_chars)
                    except TypeError:
                        text = func()
                except Exception:  # noqa: BLE001 - memory rendering must not break a turn
                    return ""
                text = str(text or "").strip()
                if text:
                    return "## PROFESSIONAL MEMORY\n" + text[:max_chars]
        return ""

    def render_task(self) -> str:
        task = self.active_task
        if task is None:
            return ""
        lines = [
            "## CURRENT TASK",
            f"id={task.task_id} type={task.task_type.value} owner={task.owner_agent.value}",
            f"priority={int(task.priority)} attempts={task.attempts} progress={task.progress:.2f}",
            f"description: {task.description}",
        ]
        if task.payload:
            lines.append(f"payload: {_json(task.payload)[:600]}")
        return "\n".join(lines)

    def render_recent(self, *, limit: int = 5) -> str:
        items = list(self.recent_results)[-max(1, limit):]
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    def note_result(self, text: str) -> None:
        clean = " ".join(str(text or "").split())
        if not clean:
            return
        self.recent_results.append(clean[: self.config.max_result_chars])

    def render_tool_brief(self, *, max_chars: int = 900) -> str:
        if not self.tools:
            return ""
        names = ", ".join(
            str((schema.get("function") or {}).get("name") or "?") for schema in self.tools
        )
        return f"available tools: {names}"[:max_chars]

    # -- statistics ------------------------------------------------------ #
    def record_call(self, reply: ModelReply, cost_usd: float = 0.0) -> None:
        usage = reply.usage
        stats = self.stats
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["prompt_tokens"] = int(stats.get("prompt_tokens", 0)) + int(usage.prompt_tokens)
        stats["completion_tokens"] = int(stats.get("completion_tokens", 0)) + int(usage.completion_tokens)
        stats["total_tokens"] = int(stats.get("total_tokens", 0)) + int(usage.total_tokens or 0)
        stats["cost_usd"] = round(float(stats.get("cost_usd", 0.0)) + float(cost_usd), 8)
        stats["latency_s"] = round(float(stats.get("latency_s", 0.0)) + float(reply.latency_s or 0.0), 3)
        stats["last_call_at"] = now_ts()

    def record_error(self, exc: BaseException | str) -> None:
        self.stats["errors"] = int(self.stats.get("errors", 0)) + 1
        self.note_result(f"error: {type(exc).__name__ if isinstance(exc, BaseException) else ''}{exc}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "agent": self.agent.value,
            "system_prompt_chars": len(self.system_prompt or ""),
            "conversation_tail": [_compact_message(item) for item in self.messages[-self.config.max_conversation_messages:]],
            "recent_results": list(self.recent_results),
            "active_task_id": self.active_task.task_id if self.active_task else "",
            "active_task": self.active_task.to_dict() if self.active_task else None,
            "branch_id": self.branch_id,
            "stats": dict(self.stats),
            "saved_at": now_ts(),
        }

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        """Restore the persisted part; API spend is *never* reset downward."""
        if not snapshot:
            return
        try:
            self.recent_results = deque(
                [str(item) for item in snapshot.get("recent_results") or []],
                maxlen=self.config.max_recent_results,
            )
            self.branch_id = str(snapshot.get("branch_id") or self.branch_id)
            incoming = snapshot.get("stats") or {}
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "errors", "actions", "denied", "decisions"):
                self.stats[key] = max(int(self.stats.get(key, 0)), int(incoming.get(key, 0) or 0))
            self.stats["cost_usd"] = max(
                float(self.stats.get("cost_usd", 0.0)), float(incoming.get("cost_usd", 0.0) or 0.0)
            )
            self.stats["latency_s"] = max(
                float(self.stats.get("latency_s", 0.0)), float(incoming.get("latency_s", 0.0) or 0.0)
            )
            task_payload = snapshot.get("active_task")
            if task_payload:
                try:
                    self.active_task = Task.from_dict(task_payload)
                except Exception:  # noqa: BLE001 - a stale task must not break restore
                    self.active_task = None
        except Exception:  # noqa: BLE001 - best-effort restore
            return

    # -- internals ------------------------------------------------------- #
    def _history_payload(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in self.messages[-self.config.max_conversation_messages:]:
            role = str(message.get("role") or "")
            if role == "user":
                content = message.get("content")
                out.append({"role": "user", "content": _compact_content(content, self.config.max_user_text_chars)})
            elif role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": str(message.get("content") or "")[:2000]}
                if message.get("tool_calls"):
                    item["tool_calls"] = message["tool_calls"]
                out.append(item)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(message.get("tool_call_id") or "tool"),
                        "content": str(message.get("content") or "")[: self.config.max_result_chars],
                    }
                )
        return _sanitize_tool_pairing(out)

    def _user_content(
        self,
        *,
        text: str,
        image_png: bytes | None,
        mime: str,
        detail: str,
        previous_text: str = "",
        previous_image_png: bytes | None = None,
        previous_mime: str = "image/png",
    ) -> Any:
        clean = str(text or "")[: self.config.max_user_text_chars]
        if image_png is None and previous_image_png is None:
            return clean
        blocks: list[dict[str, Any]] = [{"type": "text", "text": clean or "Observe the screen."}]
        if previous_image_png:
            blocks.append({"type": "text", "text": previous_text or "Previous screen (for comparison):"})
            blocks.append(_image_block(previous_image_png, previous_mime, detail))
        if image_png:
            blocks.append(_image_block(image_png, mime, detail))
        return blocks

    def _trim(self) -> None:
        limit = max(2, int(self.config.max_conversation_messages))
        if len(self.messages) > limit:
            del self.messages[: len(self.messages) - limit]


# --------------------------------------------------------------------------- #
# Action primitives
# --------------------------------------------------------------------------- #

@dataclass
class ActionStats:
    actions: int = 0
    presses: int = 0
    waits: int = 0
    denied: int = 0
    screens: int = 0
    errors: int = 0


class ActionPrimitives:
    """The only way a professional agent touches the game.

    ``emulator`` is expected to be the :class:`GuardedEmulator` for ``agent``;
    the arbiter check happens inside that proxy.  This class adds the
    agent-level budget and evidence collection on top.
    """

    def __init__(
        self,
        emulator: Any,
        *,
        agent: AgentRole,
        screen_dir: str | Path | None = None,
        save_screenshots: bool = True,
        max_actions: int = 0,
        image_scale: int = 3,
        image_format: str = "png",
        jpeg_quality: int = 90,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if emulator is None:
            raise ValueError("ActionPrimitives requires an emulator/guard proxy")
        self.emulator = emulator
        self.agent = AgentRole.coerce(agent)
        self.screen_dir = Path(screen_dir) if screen_dir else None
        self.save_screenshots = bool(save_screenshots)
        self.max_actions = max(0, int(max_actions))
        self.image_scale = max(1, min(8, int(image_scale)))
        self.image_format = str(image_format or "png")
        self.jpeg_quality = int(jpeg_quality)
        self._clock = clock or time.time
        self.stats = ActionStats()
        self.last_frame: Frame | None = None
        self.last_analysis: ScreenAnalysis | None = None
        self.evidence_frames: list[int] = []
        self.evidence_screens: list[str] = []
        self.press_log: list[dict[str, Any]] = []

    # -- read-only ------------------------------------------------------- #
    def observe(self) -> Frame:
        frame = self.emulator.observe()
        self.last_frame = frame
        self.last_analysis = analyze(frame, None)
        self.evidence_frames.append(int(getattr(frame, "frame_index", 0) or 0))
        return frame

    def analyze_frame(self, frame: Frame | None = None, previous: Frame | None = None) -> ScreenAnalysis:
        target = frame if frame is not None else self.observe()
        result = analyze(target, previous)
        self.last_analysis = result
        return result

    def screen_event(self, frame: Frame | None = None) -> ScreenEvent:
        analysis = self.last_analysis if frame is None else analyze(frame, None)
        return analysis.event if analysis is not None else ScreenEvent.UNKNOWN

    # -- mutating (arbiter-gated) ---------------------------------------- #
    def _check_budget(self) -> None:
        if self.max_actions and self.stats.actions >= self.max_actions:
            raise ControlDeniedError(
                f"action budget exhausted for {self.agent.value} ({self.max_actions})",
                agent=self.agent.value,
                owner="budget",
            )

    def press(self, button: str, *, frames: int | None = None, note: str = "") -> Frame:
        self._check_budget()
        name = str(button or "").strip().lower()
        if name not in AGENT_BUTTONS:
            raise ValueError(f"invalid button {button!r}; allowed: {sorted(AGENT_BUTTONS)}")
        frame = self.emulator.press(name, frames=frames)
        self.stats.presses += 1
        self.stats.actions += 1
        self.press_log.append({"button": name, "note": note, "frame": int(getattr(frame, "frame_index", 0) or 0)})
        self.last_frame = frame
        return frame

    def wait(self, frames: int) -> Frame:
        self._check_budget()
        frame = self.emulator.wait(int(frames))
        self.stats.waits += 1
        self.stats.actions += 1
        self.last_frame = frame
        return frame

    def run(self, buttons: Sequence[str], *, stop_on_event: bool = True) -> ActionOutcome:
        if not buttons:
            raise ValueError("buttons must not be empty")
        self._check_budget()
        outcome = self.emulator.run_batch(list(buttons), stop_on_event=bool(stop_on_event))
        executed = len(getattr(outcome, "steps", []) or [])
        self.stats.actions += executed
        self.stats.presses += executed
        self.press_log.append({"buttons": list(buttons), "frame": int(getattr(outcome, "end_frame", 0) or 0)})
        if getattr(outcome, "steps", None):
            self.last_frame = outcome.steps[-1].frame
        return outcome

    # -- evidence -------------------------------------------------------- #
    def screenshot(
        self,
        frame: Frame | None = None,
        *,
        prefix: str = "",
        save: bool | None = None,
        detail: str = "original",
    ) -> dict[str, Any]:
        """Encode ``frame`` for the vision API and optionally save a PNG."""
        target = frame if frame is not None else (self.last_frame or self.observe())
        return capture_evidence(
            target,
            screen_dir=self.screen_dir,
            prefix=prefix or f"{self.agent.value}",
            save=self.save_screenshots if save is None else bool(save),
            image_scale=self.image_scale,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            detail=detail,
            evidence=self,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "agent": self.agent.value,
            "actions": self.stats.actions,
            "presses": self.stats.presses,
            "waits": self.stats.waits,
            "denied": self.stats.denied,
            "screens_saved": self.stats.screens,
            "evidence_frames": self.evidence_frames[-40:],
            "evidence_screens": self.evidence_screens[-20:],
            "max_actions": self.max_actions,
        }


def capture_evidence(
    frame: Frame,
    *,
    screen_dir: str | Path | None = None,
    prefix: str = "agent",
    save: bool = True,
    image_scale: int = 3,
    image_format: str = "png",
    jpeg_quality: int = 90,
    detail: str = "original",
    evidence: ActionPrimitives | None = None,
) -> dict[str, Any]:
    """Encode one frame for the API and optionally persist it as evidence."""
    png, mime = encode_for_api(
        frame,
        scale=max(1, min(8, int(image_scale))),
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )
    path = ""
    if save and screen_dir is not None:
        try:
            frame_index = int(getattr(frame, "frame_index", 0) or 0)
            ext = "jpg" if mime == "image/jpeg" else "png"
            target = Path(screen_dir) / f"{prefix}_{frame_index:08d}.{ext}"
            if ext == "png":
                save_png(frame, target)
            else:  # pragma: no cover - jpeg evidence is a debugging aid
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(png)
            path = str(target)
            if evidence is not None:
                evidence.stats.screens += 1
                evidence.evidence_screens.append(path)
        except Exception:  # noqa: BLE001 - evidence saving is best effort
            path = ""
    return {
        "bytes": png,
        "mime": mime,
        "path": path,
        "detail": detail,
        "frame": int(getattr(frame, "frame_index", 0) or 0),
    }


def image_content_block(png_bytes: bytes, mime: str = "image/png", *, detail: str = "original") -> dict[str, Any]:
    """Build a vision content block without importing the API client."""
    normalized = str(mime or "").strip().lower()
    if normalized == "image/jpg":
        normalized = "image/jpeg"
    if normalized not in ("image/png", "image/jpeg"):
        raise ValueError(f"unsupported image MIME type: {mime!r}")
    if not png_bytes:
        raise ValueError("image bytes must not be empty")
    encoded = base64.b64encode(bytes(png_bytes)).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{normalized};base64,{encoded}", "detail": detail},
    }


# --------------------------------------------------------------------------- #
# Small JSON helpers
# --------------------------------------------------------------------------- #

def _json_args(arguments: Mapping[str, Any] | None) -> str:
    import json

    try:
        return json.dumps(dict(arguments or {}), ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001 - never fail on weird tool arguments
        return "{}"


def _json(value: Any) -> str:  # pragma: no cover - tiny helper
    return _json_args({"value": value})


def _image_block(png: bytes, mime: str, detail: str) -> dict[str, Any]:
    return image_content_block(png, mime, detail=detail)


def _sanitize_tool_pairing(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tool-call declarations that have no matching tool result message.

    The DeepSeek/OpenAI protocol rejects a request where an assistant message
    declares ``tool_calls`` that are not answered by one ``tool`` message per
    ``tool_call_id``.  An agent may legitimately execute only the first tool from
    a multi-call reply (or fail mid-turn), so the compact history is repaired
    here: unanswered tool calls are removed from the assistant message.  This is
    the only place the conversation is serialised, so every agent benefits.
    """
    sanitized: list[dict[str, Any]] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            declared = [
                call for call in message.get("tool_calls") or [] if isinstance(call, Mapping)
            ]
            declared_ids = [str((call or {}).get("id") or "") for call in declared]
            # collect the immediate run of tool messages that follow
            answered: dict[str, dict[str, Any]] = {}
            look = index + 1
            while look < total and messages[look].get("role") == "tool":
                tool_message = messages[look]
                tool_id = str(tool_message.get("tool_call_id") or "")
                if tool_id in declared_ids and tool_id not in answered:
                    answered[tool_id] = tool_message
                look += 1
            kept_calls = [call for call, call_id in zip(declared, declared_ids) if call_id in answered]
            repaired = {"role": "assistant", "content": str(message.get("content") or "")[:2000]}
            if kept_calls:
                repaired["tool_calls"] = kept_calls
            sanitized.append(repaired)
            for call_id in [str((call or {}).get("id") or "") for call in kept_calls]:
                sanitized.append(answered[call_id])
            index = look
            continue
        if message.get("role") == "tool":
            # orphan tool message (its assistant message was trimmed away): skip it
            index += 1
            continue
        sanitized.append(message)
        index += 1
    return sanitized


def _compact_content(content: Any, limit: int) -> Any:
    if isinstance(content, str):
        return content[:limit]
    if isinstance(content, list):
        out: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "text":
                out.append({"type": "text", "text": str(block.get("text") or "")[:limit]})
            else:
                out.append({"type": "text", "text": "[screenshot omitted from history]"})
        return out
    return str(content)[:limit]


def _compact_message(message: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"role": str(message.get("role") or ""), "content": _compact_content(message.get("content"), 400)}
    if message.get("tool_calls"):
        compact["tool_calls"] = [
            {
                "id": str((call or {}).get("id") or ""),
                "function": {"name": str(((call or {}).get("function") or {}).get("name") or "")},
            }
            for call in message["tool_calls"]
        ]
    return compact


def _safe_message(message: Any) -> Any:  # pragma: no cover - compatibility helper
    return _compact_message(message if isinstance(message, Mapping) else {})


__all__ = [
    "AgentContext",
    "AgentContextConfig",
    "ActionPrimitives",
    "ActionStats",
    "AGENT_BUTTONS",
    "capture_evidence",
    "image_content_block",
]
