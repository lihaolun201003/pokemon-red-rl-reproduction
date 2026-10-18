"""The DeepSeek "brain": turns an observation into a validated decision.

The brain is deliberately thin and stateless: every decision is one API request
carrying a bounded system prompt, the agent's own memory, the recent history and
exactly one screenshot.  Conversation context never grows unboundedly, which
keeps prompt-cache hits high and token cost predictable.

Only DeepSeek's own decisions are made here.  Tool execution lives in
``llm_agent.tools``; game state and screenshots come from the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llm_agent.config import AgentConfig
from llm_agent.contracts import Frame, ModelReply, ToolCall, ApiUsage
from llm_agent import prompts


@dataclass
class PreparedDecision:
    """A fully built request plus its audit metadata (before spending money)."""

    messages: list[dict[str, Any]]
    audit: dict[str, Any]
    trigger: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Brain(Protocol):
    """Minimal interface the runtime needs from a decision maker."""

    mode: str

    def prepare(self, **kwargs: Any) -> PreparedDecision:  # pragma: no cover - protocol
        ...

    def complete(self, prepared: PreparedDecision) -> ModelReply:  # pragma: no cover - protocol
        ...


class DeepSeekBrain:
    """Real brain: DeepSeek ``deepseek-flash`` with vision and tool calling."""

    mode = "deepseek"

    def __init__(self, cfg: AgentConfig, client: Any) -> None:
        from llm_agent.tools import TOOL_SCHEMAS  # deferred: tools imports the emulator stack

        self.cfg = cfg
        self.client = client
        self.tool_schemas = TOOL_SCHEMAS
        self._tool_schema_chars = len(prompts.tool_schemas_json(self.tool_schemas))

    # ------------------------------------------------------------------ #
    @property
    def model(self) -> str:
        return str(getattr(self.client.cfg, "model", self.cfg.api.model))

    def warmup(self) -> ModelReply:
        """Cheap real-API sanity check so auth/quota problems fail fast."""
        return self.client.probe_text("Reply with the single word OK.")

    # ------------------------------------------------------------------ #
    def prepare(
        self,
        *,
        frame: Frame,
        analysis: Any,
        state: Any,
        memory: Any,
        guard: Any,
        last_outcome_text: str,
        trigger: str,
        previous_frame: Frame | None = None,
        mental_map_text: str = "",
        nav_status_text: str = "",
    ) -> PreparedDecision:
        from llm_agent import observation

        api = self.cfg.api
        history_limit = max(1, api.max_history_entries_for_prompt)

        memory_context = memory.render_context(
            query=getattr(state, "goal", "") or None,
            max_chars=self.cfg.memory.max_context_chars,
        )
        try:
            guard_text = guard.render_warnings(max_chars=700) if guard is not None else ""
        except Exception:
            guard_text = ""

        user_text = prompts.build_user_text(
            trigger=trigger,
            frame=frame,
            analysis=analysis,
            state=state,
            memory_context=memory_context,
            guard_text=guard_text,
            last_outcome_text=last_outcome_text,
            history_limit=history_limit,
            mental_map_text=mental_map_text,
            nav_status_text=nav_status_text,
        )
        history_chars = len(prompts.render_history(state.recent_history(history_limit), history_limit))

        image_png, mime = observation.encode_for_api(
            frame,
            scale=api.image_scale,
            image_format=api.image_format,
            jpeg_quality=api.image_jpeg_quality,
        )
        previous_png = None
        previous_mime = "image/png"
        if api.include_previous_screenshot and previous_frame is not None:
            previous_png, previous_mime = observation.encode_for_api(
                previous_frame,
                scale=api.image_scale,
                image_format=api.image_format,
                jpeg_quality=api.image_jpeg_quality,
            )

        messages = prompts.build_messages(
            user_text=user_text,
            image_png=image_png,
            mime=mime,
            detail=api.image_detail,
            previous_text="Previous screen (for comparison):" if previous_png else "",
            previous_image_png=previous_png,
            previous_mime=previous_mime,
        )

        image_width = frame.rgba.shape[1] * api.image_scale
        image_height = frame.rgba.shape[0] * api.image_scale
        audit = {
            "mode": self.mode,
            "model": api.model,
            "trigger": trigger,
            "prompt_chars": len(prompts.SYSTEM_PROMPT) + len(user_text) + self._tool_schema_chars,
            "system_chars": len(prompts.SYSTEM_PROMPT),
            "user_chars": len(user_text),
            "memory_chars": len(memory_context),
            "history_chars": history_chars,
            "guard_chars": len(guard_text),
            "tool_schema_chars": self._tool_schema_chars,
            "image_count": 2 if previous_png else 1,
            "image_bytes": len(image_png) + (len(previous_png) if previous_png else 0),
            "image_width": image_width,
            "image_height": image_height,
            "image_format": api.image_format,
            "max_tokens": api.max_tokens,
            "thinking": api.thinking,
        }
        return PreparedDecision(messages=messages, audit=audit, trigger=trigger)

    def complete(self, prepared: PreparedDecision) -> ModelReply:
        return self.client.chat(
            prepared.messages,
            tools=self.tool_schemas,
            max_tokens=self.cfg.api.max_tokens,
        )
