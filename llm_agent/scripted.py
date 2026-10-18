"""Deterministic scripted brain used ONLY for dry-run integration testing.

This is **not** the DeepSeek agent.  It exists so the full runtime (PyBoy, tools,
memory, checkpoints, viewer, audit) can be exercised end-to-end before an API key
is configured.  Whenever it is used, the run is labelled
``SCRIPTED DRY RUN — not a real DeepSeek run`` in the audit log, metrics and
report.  It must never be presented as evidence of real API/gameplay success.
"""

from __future__ import annotations

from typing import Any

from llm_agent.agent import PreparedDecision
from llm_agent.contracts import (
    ApiUsage,
    ModelReply,
    ScreenEvent,
    ToolCall,
)

_DIRECTIONS = ("up", "down", "left", "right")


class ScriptedBrain:
    """A tiny observation-driven policy: reads dialogues, explores, avoids loops."""

    mode = "scripted"

    def __init__(self, *, max_actions: int = 24) -> None:
        self.model = "scripted"
        self.max_actions = max_actions
        self._direction_index = 0
        self._last_screen_signature = ""
        self._same_screen_count = 0
        self._turn = 0
        self._scenario = "explore"
        self._scenario_step = 0

    # ------------------------------------------------------------------ #
    def prepare(self, **kwargs: Any) -> PreparedDecision:
        self._context = kwargs
        return PreparedDecision(
            messages=[{"role": "user", "content": "SCRIPTED DRY RUN BROWSER"}],
            audit={"mode": self.mode, "prompt_chars": 0, "image_count": 0, "image_bytes": 0,
                   "max_tokens": 0, "model": self.model},
            trigger=str(kwargs.get("trigger", "")),
        )

    def complete(self, prepared: PreparedDecision) -> ModelReply:
        self._turn += 1
        analysis = self._context.get("analysis")
        frame = self._context.get("frame")
        state = self._context.get("state")
        guard = self._context.get("guard")
        signature = getattr(analysis, "signature", "") if analysis is not None else ""
        if signature == self._last_screen_signature:
            self._same_screen_count += 1
        else:
            self._last_screen_signature = signature
            self._same_screen_count = 0
        event = getattr(analysis, "event", ScreenEvent.NONE) if analysis is not None else ScreenEvent.NONE

        calls: list[ToolCall] = []

        def call(name: str, arguments: dict[str, Any] | None = None) -> None:
            calls.append(ToolCall(id=f"scripted_{self._turn}_{len(calls)}", name=name, arguments=arguments or {}))

        warnings = guard.check() if guard is not None else []
        critical = any(getattr(w, "severity", "") == "critical" for w in warnings)

        if event == ScreenEvent.DIALOG:
            call("execute_actions", {"actions": ["a", "a", "a"]})
        elif event == ScreenEvent.MENU:
            call("press_button", {"button": "b"})
        elif event == ScreenEvent.BATTLE:
            call("execute_actions", {"actions": ["a", "a"]})
        elif event == ScreenEvent.TRANSITION or (frame is not None and frame.is_black):
            call("wait_frames", {"frames": 30})
        elif critical:
            call("record_discovery", {"kind": "lesson",
                                      "text": "Scripted dry-run: critical loop warning, chose a new direction."})
            call("walk", {"direction": self._next_direction(), "max_steps": 2})
        elif self._same_screen_count >= 2:
            call("record_discovery", {"kind": "other",
                                      "text": "Scripted dry-run: screen is not changing; trying another direction."})
            call("walk", {"direction": self._next_direction(), "max_steps": 2})
        else:
            if self._turn % 5 == 1:
                call("set_goal", {"goal": "Scripted dry run: explore the current area and find exits.",
                                  "reason": "deterministic test policy"})
            call("walk", {"direction": self._next_direction(), "max_steps": self._walker_steps()})
            if self._turn % 7 == 0:
                call("record_discovery", {"kind": "other",
                                          "text": f"Scripted dry-run note at turn {self._turn}."})
        if not calls:  # pragma: no cover - defensive
            call("observe_game", {})

        rationale = f"[SCRIPTED DRY RUN] turn {self._turn}: {event.value if hasattr(event, 'value') else event}"
        return ModelReply(
            content=rationale,
            tool_calls=calls,
            usage=ApiUsage(),
            model=self.model,
            latency_s=0.0,
            finish_reason="scripted",
            raw={"scripted": True},
        )

    # ------------------------------------------------------------------ #
    def _next_direction(self) -> str:
        if self._same_screen_count >= 4:
            self._direction_index = (self._direction_index + 1) % len(_DIRECTIONS)
        else:
            self._direction_index = (self._direction_index + 1) % len(_DIRECTIONS)
        return _DIRECTIONS[self._direction_index]

    def _walker_steps(self) -> int:
        return max(1, min(self.max_actions, 6))
