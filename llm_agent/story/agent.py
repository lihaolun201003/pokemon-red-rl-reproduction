"""The story / dialogue agent of the Pokémon Red multi-agent system.

The agent is a *real* dialogue driver, not a stub:

1. it observes the arbiter-guarded emulator and classifies the screen with
   :mod:`llm_agent.observation` plus its own conservative pixel heuristics;
2. it sends the **actual screenshot** to the model (vision content block built by
   :class:`llm_agent.multi_agent.agent_context.AgentContext`) and asks for one
   tool call per turn;
3. it executes the chosen tool for real: ``advance_dialog`` / ``choose_option``
   press buttons through the guarded emulator, ``record_fact`` /
   ``propose_task`` write memory/proposals, ``finish`` ends the conversation;
4. it re-observes the resulting screen and loops, detecting the end of the
   dialogue (no dialogue/menu for N observations after having been in one);
5. it records every lesson with an honest verification status (verified /
   inferred / hypothesis), mirrors it into the shared world memory and proposes
   concrete follow-up tasks for the director;
6. it never invents story content and never presses a button without owning the
   control lease.

There is deliberately no walkthrough knowledge in this module: no story order, no
locations, no characters, no species.  The model learns everything from the
screenshots it receives and the notes it wrote itself.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from llm_agent.config import DEFAULT_MEMORY_ROOT, MemoryConfig
from llm_agent.contracts import (
    ApiAuthenticationError,
    ApiBalanceError,
    ApiError,
    BudgetExceededError,
    Frame,
    ModelReply,
    ScreenEvent,
    ToolCall,
    ToolResult,
)
from llm_agent.multi_agent.agent_context import (
    ActionPrimitives,
    AgentContext,
    AgentContextConfig,
    image_content_block,
)
from llm_agent.multi_agent.contracts import (
    AgentMessage,
    AgentRole,
    BudgetLimitError,
    ControlDeniedError,
    EventType,
    Priority,
    Task,
    now_ts,
)
from llm_agent.observation import ScreenAnalysis, analyze
from llm_agent.story.memory import SharedMemoryAdapter, StoryMemory
from llm_agent.story.prompts import (
    SYSTEM_PROMPT,
    render_dialog_summary,
    render_no_tool_nudge,
    render_observation,
)
from llm_agent.story.tools import (
    DIALOG_SCREEN_EVENTS,
    TOOL_SCHEMAS,
    StoryTools,
    ToolError,
    acquire_dialog,
    is_dialog_screen,
)

__all__ = [
    "StoryAgent",
    "StoryAgentConfig",
    "is_dialog_context",
    "DIALOG_SCREEN_EVENTS",
    "MAX_EVIDENCE_SCREENS",
    "EXPLORATION_PROPOSAL",
]

#: Evidence lists are hard-capped so a result dict can never grow without bound.
MAX_EVIDENCE_SCREENS: int = 60

#: Text cap for the tool result fed back to the model.
MAX_TOOL_RESULT_CHARS: int = 800

#: Default exploration description (generic; contains no location or plot).
EXPLORATION_PROPOSAL = (
    "Continue exploring the current area for people to talk to and clues visible on screen"
)


# --------------------------------------------------------------------------- #
# Screen classification
# --------------------------------------------------------------------------- #

def is_dialog_context(analysis: ScreenAnalysis | None) -> bool:
    """True when the screen looks like an open dialogue box or a menu window.

    Delegates to :func:`llm_agent.story.tools.is_dialog_screen` so the agent and
    the interaction probe share exactly one conservative heuristic.
    """
    return is_dialog_screen(analysis)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class StoryAgentConfig:
    """Bounds and model settings for the story agent (never game knowledge)."""

    max_turns: int = 12
    dialogue_end_observations: int = 2
    no_dialogue_observations: int = 3
    max_tokens: int = 400
    thinking: str = "disabled"
    image_scale: int = 3
    image_format: str = "png"
    jpeg_quality: int = 90
    image_detail: str = "original"
    max_presses_per_call: int = 4
    max_conversation_messages: int = 10
    max_memory_chars: int = 2500
    max_context_chars: int = 3000
    max_result_chars: int = 700
    max_evidence_screens: int = MAX_EVIDENCE_SCREENS
    max_empty_replies: int = 2
    max_tool_errors: int = 3
    #: How many times a failed API request may be retried with a reset history.
    max_api_retries: int = 1
    max_auto_proposals: int = 2
    max_proposals_returned: int = 16
    max_probe_presses: int = 12
    probe_observations_per_step: int = 3
    max_recovery_hints: int = 4
    save_screenshots: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "StoryAgentConfig":
        """Resolve the story settings from an optional :class:`AgentConfig`."""
        cfg = cls()
        if config is None:
            return cfg
        multi = getattr(config, "multi_agent", None)
        if multi is not None:
            cfg.max_turns = int(getattr(multi, "dialogue_max_turns", cfg.max_turns) or cfg.max_turns)
            cfg.max_tokens = int(getattr(multi, "max_tokens_agent", cfg.max_tokens) or cfg.max_tokens)
            cfg.image_scale = int(getattr(multi, "image_scale_agent", cfg.image_scale) or cfg.image_scale)
            thinking = getattr(multi, "thinking_agent", cfg.thinking)
            if str(thinking or "").strip().lower() in ("disabled", "enabled"):
                cfg.thinking = str(thinking).strip().lower()
            cfg.save_screenshots = bool(getattr(multi, "save_screenshots", cfg.save_screenshots))
        api = getattr(config, "api", None)
        if api is not None:
            image_format = str(getattr(api, "image_format", cfg.image_format) or cfg.image_format)
            if image_format.strip().lower() in ("png", "jpeg"):
                cfg.image_format = image_format.strip().lower()
            try:
                cfg.jpeg_quality = int(getattr(api, "image_jpeg_quality", cfg.jpeg_quality))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
            detail = str(getattr(api, "image_detail", cfg.image_detail) or cfg.image_detail)
            if detail.strip().lower() in ("original", "low", "high", "auto"):
                cfg.image_detail = detail.strip().lower()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_turns": self.max_turns,
            "dialogue_end_observations": self.dialogue_end_observations,
            "no_dialogue_observations": self.no_dialogue_observations,
            "max_tokens": self.max_tokens,
            "thinking": self.thinking,
            "image_scale": self.image_scale,
            "image_format": self.image_format,
            "image_detail": self.image_detail,
            "max_presses_per_call": self.max_presses_per_call,
            "max_auto_proposals": self.max_auto_proposals,
            "max_probe_presses": self.max_probe_presses,
            "probe_observations_per_step": self.probe_observations_per_step,
            "max_api_retries": self.max_api_retries,
            "save_screenshots": self.save_screenshots,
        }


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class StoryAgent:
    """Real dialogue takeover agent (frozen constructor interface).

    Parameters
    ----------
    router:
        Duck-typed :class:`ModelRouter`: ``model_for`` / ``can_call`` /
        ``complete``.  Tests inject a fake router.
    bus:
        Optional event bus; ``publish(AgentMessage)`` is called best-effort for
        discoveries/hypotheses and the dialogue end.
    arbiter:
        Optional :class:`ControlArbiter`; ``holds(AgentRole.STORY)`` is checked
        before every press and before the conversation starts.
    shared_memory:
        Shared world knowledge base (frozen ``remember``/``search`` interface).
    memory:
        Existing ``AgentMemory``-like object (the director passes one) or None.
    config:
        Optional ``AgentConfig``; ``None`` keeps :class:`StoryAgentConfig`
        defaults (unit tests).
    screen_dir:
        Directory for evidence PNGs (``None`` means do not persist screenshots).
    clock:
        Injectable clock (defaults to :func:`time.time`).
    """

    agent_role = AgentRole.STORY

    def __init__(
        self,
        *,
        router: Any,
        bus: Any = None,
        arbiter: Any = None,
        shared_memory: Any = None,
        memory: Any = None,
        config: Any = None,
        screen_dir: str | Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.router = router
        self.bus = bus
        self.arbiter = arbiter
        self.shared_memory = shared_memory
        self.config = config
        self._clock: Callable[[], float] = clock or time.time
        self._cfg = StoryAgentConfig.from_config(config)

        self.screen_dir = self._resolve_screen_dir(screen_dir, config)
        self.story_memory = self._build_memory(memory, config, shared_memory)
        #: Public copy of the static system prompt (the director snapshots it).
        self.system_prompt: str = SYSTEM_PROMPT
        #: Strategy notes pushed by the recovery manager (L1 suggestion, …).
        self._recovery_hints: list[dict[str, str]] = []
        self.context = AgentContext(
            agent=AgentRole.STORY,
            system_prompt=self.system_prompt,
            tools=tuple(TOOL_SCHEMAS),
            memory=self.story_memory,
            config=AgentContextConfig(
                max_conversation_messages=self._cfg.max_conversation_messages,
                max_memory_chars=self._cfg.max_memory_chars,
                max_result_chars=self._cfg.max_result_chars,
            ),
        )

        self._paused: bool = False
        self._events: deque[dict[str, Any]] = deque(maxlen=32)
        self._last_event: dict[str, Any] = {}
        self._publish_errors: list[str] = []
        self._context_errors: list[str] = []
        self._counters: dict[str, Any] = {
            "dialogs": 0,
            "turns": 0,
            "api_calls": 0,
            "cost_usd": 0.0,
            "discoveries": 0,
            "proposals": 0,
            "failures": 0,
            "denied": 0,
            "api_retries": 0,
        }
        self._last_status: str = "idle"
        self._last_summary: str = ""
        self._last_error: str = ""

    # -- frozen interface ------------------------------------------------ #

    def handles(self) -> tuple[str, ...]:
        """Event kinds this agent owns (the director assigns them here)."""
        return ("dialog", "menu", "story")

    @property
    def paused(self) -> bool:
        """True while the agent refuses to touch the emulator."""
        return self._paused

    def pause(self) -> None:
        """Pause at the next safe point (never in the middle of a press)."""
        self._paused = True

    def resume(self) -> None:
        """Resume normal operation."""
        self._paused = False

    def apply_recovery_hint(self, level: str, hint: str) -> None:
        """Store a recovery strategy note used by the next prompts (L1 recovery).

        The note is rendered as an extra system block on every following model
        call so a local strategy adjustment (e.g. "stop repeating A, try B")
        actually reaches the model.  Notes are bounded and de-duplicated.
        """
        clean_hint = " ".join(str(hint or "").split())[:300]
        if not clean_hint:
            return
        clean_level = " ".join(str(level or "").split()).lower()[:40] or "l1_local"
        entry = {"level": clean_level, "hint": clean_hint}
        for existing in self._recovery_hints:
            if existing.get("level") == clean_level and existing.get("hint") == clean_hint:
                return
        self._recovery_hints.append(entry)
        limit = max(1, int(self._cfg.max_recovery_hints))
        if len(self._recovery_hints) > limit:
            del self._recovery_hints[: len(self._recovery_hints) - limit]

    def _render_recovery_hints(self) -> str:
        if not self._recovery_hints:
            return ""
        lines = ["## RECOVERY HINT FROM THE DIRECTOR (follow it in this conversation)"]
        for item in self._recovery_hints[-self._cfg.max_recovery_hints:]:
            lines.append(f"- ({item.get('level') or 'l1_local'}) {item.get('hint')}")
        return "\n".join(lines)[:1200]

    def status(self) -> dict[str, Any]:
        """Compact, JSON-serialisable status for the director/viewer."""
        return {
            "agent": AgentRole.STORY.value,
            "handles": list(self.handles()),
            "paused": bool(self._paused),
            "screen_dir": str(self.screen_dir) if self.screen_dir is not None else "",
            "config": self._cfg.to_dict(),
            "counters": dict(self._counters),
            "last_status": self._last_status,
            "last_summary": self._last_summary,
            "last_error": self._last_error,
            "last_event": dict(self._last_event),
            "memory": self.story_memory.stats(),
            "context": dict(self.context.stats),
            "publish_errors": list(self._publish_errors[-3:]),
            "context_errors": list(self._context_errors[-3:]),
            "proposal_count": len(self.story_memory.proposals()),
            "recovery_hints": [dict(item) for item in self._recovery_hints],
        }

    def on_event(self, event: Any) -> None:
        """Optional notification hook (AgentMessage / EventType / mapping)."""
        info = self._event_info(event)
        if info is None:
            return
        self._events.append(info)
        self._last_event = info
        event_type = info.get("event_type", "")
        payload = info.get("payload") or {}
        agent = str(payload.get("agent", "")).strip().lower()
        if event_type == EventType.AGENT_PAUSED.value and agent in ("story", "story_agent"):
            self.pause()
        elif event_type == EventType.AGENT_RESUMED.value and agent in ("story", "story_agent"):
            self.resume()

    # -- main entry point ------------------------------------------------ #

    def handle_dialog(
        self,
        *,
        emulator: Any,
        task: Any | None = None,
        max_turns: int = 12,
    ) -> dict:
        """Take over the emulator and drive the current dialogue to its end.

        The director grants the control lease before calling this method.  The
        agent never requests the lease itself; if it is not the owner it returns
        ``status="control_denied"`` without pressing anything.
        """
        turns_limit = max(1, int(max_turns or self._cfg.max_turns))
        if self._paused:
            return self._finalize(
                "interrupted",
                summary="The story agent is paused; the dialogue was not touched.",
                error="paused",
            )
        if not self._holds_lease():
            return self._finalize(
                "control_denied",
                summary="The story agent does not hold the control lease; no button was pressed.",
                error="control lease is not held by the story agent",
            )
        if emulator is None:
            return self._finalize("failed", summary="No emulator was provided.", error="emulator is None")

        primitives = ActionPrimitives(
            emulator,
            agent=AgentRole.STORY,
            screen_dir=self.screen_dir,
            save_screenshots=self._cfg.save_screenshots,
            image_scale=self._cfg.image_scale,
            image_format=self._cfg.image_format,
            jpeg_quality=self._cfg.jpeg_quality,
            clock=self._clock,
        )
        tools = StoryTools(
            memory=self.story_memory,
            actions=primitives,
            arbiter=self.arbiter,
            max_presses_per_call=self._cfg.max_presses_per_call,
            clock=self._clock,
        )
        tools.begin()
        self.context.begin_task(task if isinstance(task, Task) else None)

        discoveries: list[dict[str, Any]] = []
        tasks_created: list[dict[str, Any]] = []
        dialog_texts: list[str] = []
        status = ""
        error = ""
        finish_summary = ""
        ever_in_dialog = False
        turns = 0
        api_calls = 0
        cost_usd = 0.0

        try:
            frame = primitives.observe()
            analysis = primitives.last_analysis or analyze(frame, None)
            shot = primitives.screenshot(frame, prefix="story")
            evidence_ref = self._evidence_ref(shot, frame)
            ever_in_dialog = is_dialog_context(analysis)

            if analysis.event is ScreenEvent.BATTLE:
                return self._finalize(
                    "interrupted",
                    summary=(
                        "The screen shows a battle, not a dialogue; returning control so the "
                        "battle specialist can take over."
                    ),
                    error="",
                    npcs=self.story_memory.npcs()[:20],
                    evidence=self._collect_evidence(primitives, dialog_texts),
                )

            # -- acquire a dialogue ourselves when none is visible ----------- #
            # A blind walker only presses directions, so a conversation has to be
            # started on purpose: bounded deterministic A probes while facing the
            # current direction and then each direction.  The probe only discovers
            # the affordance; the content is still read by the vision model below.
            if not is_dialog_context(analysis):
                if analysis.event is ScreenEvent.TRANSITION or bool(analysis.is_black):
                    primitives.wait(8)
                    frame = primitives.observe()
                    analysis = analyze(frame, None)
                    shot = primitives.screenshot(frame, prefix="story")
                    evidence_ref = self._evidence_ref(shot, frame)
                if analysis.event is ScreenEvent.BATTLE:
                    return self._finalize(
                        "interrupted",
                        summary="A battle started before the conversation could begin.",
                        error="battle on screen",
                        npcs=self.story_memory.npcs()[:20],
                        evidence=self._collect_evidence(primitives, dialog_texts),
                    )
                if not is_dialog_context(analysis):
                    probe = acquire_dialog(
                        primitives,
                        arbiter=self.arbiter,
                        max_presses=self._cfg.max_probe_presses,
                        max_observations_per_step=self._cfg.probe_observations_per_step,
                        prefix="story_probe",
                    )
                    probe_status = str(probe.get("status") or "no_dialog")
                    probe_reason = str(probe.get("reason") or "")
                    probe_evidence = self._collect_evidence(primitives, dialog_texts)
                    probe_npcs = self.story_memory.npcs()[:20]
                    if probe_status == "control_denied":
                        return self._finalize(
                            "control_denied",
                            summary=(
                                "The interaction probe stopped because the story agent does not "
                                "hold the control lease; no further button was pressed."
                            ),
                            error=probe_reason or "control lease not held",
                            npcs=probe_npcs,
                            evidence=probe_evidence,
                        )
                    if probe_status == "interrupted":
                        return self._finalize(
                            "interrupted",
                            summary=f"The interaction probe was interrupted: {probe_reason}",
                            error=probe_reason,
                            npcs=probe_npcs,
                            evidence=probe_evidence,
                        )
                    if probe_status != "dialog":
                        return self._finalize(
                            "no_dialog",
                            summary=(
                                "No dialogue or menu could be started from the current position "
                                f"({len(probe.get('presses') or [])} bounded probe press(es)); "
                                "nothing else was attempted."
                            ),
                            error="",
                            npcs=probe_npcs,
                            evidence=probe_evidence,
                        )
                    # A dialogue was acquired: continue with the normal vision loop.
                    acquired = probe.get("captured") or probe
                    frame = acquired.get("frame", frame)
                    analysis = acquired.get("analysis", analysis)
                    shot = dict(acquired.get("shot") or shot)
                    evidence_ref = str(acquired.get("evidence_ref") or evidence_ref)
                    ever_in_dialog = True

            non_dialog_streak = 0 if ever_in_dialog else 1
            empty_streak = 0
            tool_error_streak = 0
            api_retry_count = 0
            pending_nudge = ""
            previous: Frame | None = None
            previous_event = ""

            while turns < turns_limit:
                if self._paused:
                    status, error = "interrupted", "paused by request"
                    break
                if not self._holds_lease():
                    status = "control_denied"
                    error = "control lease was revoked while handling the dialogue"
                    break
                allowed, reason = self._can_call()
                if not allowed:
                    status, error = "budget", reason or "API budget exhausted"
                    break

                turns += 1
                user_text = pending_nudge or render_observation(
                    frame_index=int(getattr(frame, "frame_index", 0) or 0),
                    evidence_ref=evidence_ref,
                    analysis_summary=str(analysis.summary),
                    event_name=analysis.event.value,
                    dialog_visible=is_dialog_context(analysis),
                    turn=turns,
                    max_turns=turns_limit,
                )
                messages = self._build_prompt_messages(
                    user_text=user_text,
                    shot=shot,
                    extra_system=self._render_recovery_hints(),
                )
                cost_before = self._cost_before()
                try:
                    reply = self.router.complete(
                        AgentRole.STORY,
                        messages,
                        tools=list(TOOL_SCHEMAS),
                        max_tokens=self._cfg.max_tokens,
                        thinking=self._cfg.thinking,
                    )
                except BudgetLimitError as exc:
                    status, error = "budget", f"{type(exc).__name__}: {exc}"
                    break
                except BudgetExceededError as exc:
                    status, error = "budget", f"{type(exc).__name__}: {exc}"
                    break
                except ApiError as exc:
                    # A transient request failure (e.g. an HTTP 400 caused by a
                    # stale/odd conversation history) must not fail the story
                    # task: drop the conversation tail and retry once from the
                    # current observation.  Auth/balance failures stay fatal.
                    fatal = isinstance(exc, (ApiAuthenticationError, ApiBalanceError))
                    if not fatal and api_retry_count < self._cfg.max_api_retries:
                        api_retry_count += 1
                        self._counters["api_retries"] = int(
                            self._counters.get("api_retries", 0)
                        ) + 1
                        pending_nudge = ""
                        self._reset_conversation_window()
                        continue
                    status, error = "failed", f"{type(exc).__name__}: {exc}"
                    break
                except ControlDeniedError as exc:
                    status, error = "control_denied", str(exc)
                    break
                except Exception as exc:  # noqa: BLE001 - the agent must never crash the run
                    status, error = "failed", f"model call failed: {type(exc).__name__}: {exc}"
                    break

                if not isinstance(reply, ModelReply):
                    status, error = "failed", f"router returned {type(reply).__name__}, not ModelReply"
                    break

                api_calls += 1
                call_cost = self._cost_after(cost_before, reply)
                cost_usd += float(call_cost)
                try:
                    self.context.record_call(reply, call_cost)
                except Exception:  # noqa: BLE001 - accounting must not stop the dialogue
                    pass
                pending_nudge = ""

                # The API requires every declared tool_call_id to be answered by a
                # tool message.  We execute exactly one tool per turn, so the
                # assistant message we keep must declare exactly that single call.
                call = self._pick_call(reply)
                trimmed_reply = self._trim_reply_for_call(reply, call)
                self.context.add_user(user_text)
                self.context.add_assistant(trimmed_reply)

                if call is None:
                    empty_streak += 1
                    if empty_streak >= self._cfg.max_empty_replies:
                        status, error = "failed", "the model returned no tool call"
                        break
                    pending_nudge = render_no_tool_nudge()
                    continue
                empty_streak = 0

                previous_event = analysis.event.value
                try:
                    result = tools.execute_safe(
                        call,
                        evidence_ref=evidence_ref,
                        frame_index=int(getattr(frame, "frame_index", 0) or 0),
                    )
                except ControlDeniedError as exc:
                    # Keep the conversation history valid even on this abort path.
                    self.context.add_tool_result(
                        call, f"ERROR: control denied: {exc}", ok=False
                    )
                    status, error = "control_denied", str(exc)
                    break
                except ToolError as exc:  # defensive: execute already converts these
                    result = ToolResult(
                        tool_call_id=str(getattr(call, "id", "") or "call"),
                        name=str(getattr(call, "name", "") or ""),
                        ok=False,
                        summary=f"rejected: {exc}",
                        error=str(exc),
                    )
                except Exception as exc:  # noqa: BLE001 - unexpected tool crash
                    self.context.add_tool_result(
                        call, f"ERROR: tool crashed: {type(exc).__name__}: {exc}", ok=False
                    )
                    status, error = "failed", f"tool crashed: {type(exc).__name__}: {exc}"
                    break

                self.context.add_tool_result(call, self._render_result_text(result), ok=result.ok)
                self.context.note_result(result.summary)
                if result.ok:
                    self._collect_effects(result, discoveries, tasks_created, dialog_texts)
                else:
                    tool_error_streak += 1
                    if tool_error_streak >= self._cfg.max_tool_errors:
                        status, error = "failed", "too many rejected tool calls"
                        break

                if tools.finished:
                    finish_summary = str(tools.finish_payload.get("summary") or "")
                    outcome = str(tools.finish_payload.get("outcome") or "dialog_ended")
                    status = "interrupted" if outcome == "blocked" else "dialog_ended"
                    break

                previous = frame
                frame, analysis, shot, evidence_ref = self._observe_again(
                    primitives, previous, prefix="story"
                )
                in_dialog = is_dialog_context(analysis)
                if in_dialog:
                    ever_in_dialog = True
                    non_dialog_streak = 0
                else:
                    non_dialog_streak += 1
                if analysis.event is ScreenEvent.BATTLE:
                    status = "interrupted"
                    break
                if ever_in_dialog and non_dialog_streak >= self._cfg.dialogue_end_observations:
                    status = "dialog_ended"
                    break
                if (not ever_in_dialog) and non_dialog_streak >= self._cfg.no_dialogue_observations:
                    status = "dialog_ended"
                    break

            if not status:
                status = "budget"
                error = f"turn budget of {turns_limit} turn(s) exhausted"

            # -- concrete follow-up proposals (deduped through StoryMemory) - #
            for proposal in self._auto_proposals(discoveries, tools, status):
                tasks_created.append(proposal)

            npcs = self.story_memory.npcs()[:20]
            summary = self._build_summary(
                status=status,
                error=error,
                finish_summary=finish_summary,
                turns=turns,
                discoveries=discoveries,
                tasks_created=tasks_created,
                npcs=npcs,
                ever_in_dialog=ever_in_dialog,
            )
            evidence = self._collect_evidence(primitives, dialog_texts)
            return self._finalize(
                status,
                summary=summary,
                error=error,
                npcs=npcs,
                discoveries=discoveries,
                tasks_created=tasks_created,
                turns=turns,
                api_calls=api_calls,
                cost_usd=cost_usd,
                evidence=evidence,
            )
        except ControlDeniedError as exc:
            self._last_error = str(exc)
            self._publish(
                EventType.AGENT_ERROR,
                {"agent": AgentRole.STORY.value, "error": str(exc), "context": {"status": "control_denied"}},
            )
            return self._finalize(
                "control_denied",
                summary="The story agent lost the control lease; no further button was pressed.",
                error=str(exc),
                npcs=self.story_memory.npcs()[:20],
            )
        except Exception as exc:  # noqa: BLE001 - always return a structured failure
            return self._finalize(
                "failed",
                summary="The story agent hit an unexpected error and stopped safely.",
                error=f"{type(exc).__name__}: {exc}",
                npcs=self.story_memory.npcs()[:20],
            )

    # -- task proposals -------------------------------------------------- #

    def propose_tasks(self, *, context: Mapping[str, Any] | None = None) -> list[dict]:
        """Return de-duplicated follow-up proposals for the director.

        When nothing concrete has been learned yet, this honestly proposes
        continued exploration instead of inventing story progress.
        """
        context = context or {}
        known = self._known_descriptions(context)
        proposals = self.story_memory.proposals()
        if not proposals:
            proposal, _created = self.story_memory.add_proposal(
                owner_agent="navigation",
                task_type="explore",
                description=EXPLORATION_PROPOSAL,
                priority="low",
                payload={
                    "reason": "no concrete clue has been observed yet; keep mapping the visible area",
                    "source": "story_agent",
                },
            )
            if proposal:
                proposals = [StoryMemory.public_proposal(proposal)]
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for proposal in proposals:
            description = str(proposal.get("description") or "")
            key = " ".join(description.casefold().split())
            if not key or key in known or key in seen:
                continue
            seen.add(key)
            out.append(dict(proposal))
            if len(out) >= self._cfg.max_proposals_returned:
                break
        return out

    # -- memory construction --------------------------------------------- #

    @staticmethod
    def _known_descriptions(context: Mapping[str, Any]) -> set[str]:
        """Collect already-known task descriptions from the director context."""
        known: set[str] = set()
        items = context.get("existing_tasks") or context.get("known_tasks") or ()
        if isinstance(items, Mapping):
            items = list(items.values())
        if isinstance(items, (str, bytes)):
            items = [items]
        for item in items:
            if isinstance(item, Mapping):
                description = item.get("description") or item.get("text") or ""
            else:
                description = getattr(item, "description", "") or item
            key = " ".join(str(description or "").split()).casefold()
            if key:
                known.add(key)
        return known


    def _resolve_screen_dir(self, screen_dir: Any, config: Any) -> Path | None:
        if screen_dir is not None:
            return Path(screen_dir)
        multi = getattr(config, "multi_agent", None)
        if multi is None:
            return None
        if not bool(getattr(multi, "save_screenshots", True)):
            return None
        evidence_root = str(getattr(multi, "evidence_root", "") or "").strip()
        if not evidence_root:
            return None
        return Path(evidence_root) / "screens" / "story"

    def _build_memory(self, memory: Any, config: Any, shared_memory: Any) -> StoryMemory:
        if isinstance(memory, StoryMemory):
            if shared_memory is not None:
                memory.shared = SharedMemoryAdapter(shared_memory, clock=self._clock)
            return memory
        memory_config = None
        agent_memory = None
        root: Path | None = None
        if isinstance(memory, MemoryConfig):
            memory_config = memory
            root = Path(memory.root) / "story"
        elif isinstance(memory, (str, Path)):
            root = Path(memory)
        elif memory is not None:
            agent_memory = memory
            root = Path(getattr(memory, "root", "") or "")
            root = root if str(root) else None
        if root is None:
            config_memory = getattr(config, "memory", None) if config is not None else None
            if isinstance(config_memory, MemoryConfig):
                memory_config = memory_config or config_memory
                root = Path(config_memory.root) / "story"
            else:
                root = Path(DEFAULT_MEMORY_ROOT) / "story"
        return StoryMemory(
            root,
            agent_memory=agent_memory,
            shared_memory=shared_memory,
            memory_config=memory_config,
            max_context_chars=self._cfg.max_context_chars,
            clock=self._clock,
            save_on_write=True,
        )

    # -- observation helpers --------------------------------------------- #

    @staticmethod
    def _evidence_ref(shot: Mapping[str, Any], frame: Any) -> str:
        path = str(shot.get("path") or "")
        if path:
            return path
        return f"frame:{int(getattr(frame, 'frame_index', 0) or 0)}"

    def _observe_again(
        self,
        primitives: ActionPrimitives,
        previous: Frame | None,
        *,
        prefix: str,
    ) -> tuple[Frame, ScreenAnalysis, dict[str, Any], str]:
        frame = primitives.observe()
        analysis = analyze(frame, previous)
        shot = primitives.screenshot(frame, prefix=prefix)
        return frame, analysis, shot, self._evidence_ref(shot, frame)

    def _collect_evidence(
        self, primitives: ActionPrimitives, dialog_texts: Sequence[str]
    ) -> dict[str, Any]:
        cap = max(1, int(self._cfg.max_evidence_screens))
        screens = [str(path) for path in list(primitives.evidence_screens)[-cap:]]
        frames = [int(index) for index in list(primitives.evidence_frames)[-cap:]]
        texts: list[str] = []
        for text in dialog_texts:
            clean = " ".join(str(text).split())
            if clean and clean not in texts:
                texts.append(clean)
        return {"screens": screens, "frames": frames, "dialog_texts": texts[:cap]}

    # -- model helpers --------------------------------------------------- #

    def _holds_lease(self) -> bool:
        """True when the story agent may press buttons right now.

        The director grants the lease before calling :meth:`handle_dialog`; the
        agent only *checks* it and never requests it itself.
        """
        arbiter = self.arbiter
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

    def _can_call(self) -> tuple[bool, str]:
        can_call = getattr(self.router, "can_call", None)
        if not callable(can_call):
            return True, ""
        try:
            try:
                result = can_call(AgentRole.STORY)
            except TypeError:  # pragma: no cover - string-based routers
                result = can_call("story")
        except Exception as exc:  # noqa: BLE001 - a broken budget check must stop calls
            return False, f"budget check failed: {type(exc).__name__}: {exc}"
        if isinstance(result, tuple):
            allowed = bool(result[0]) if result else False
            reason = str(result[1]) if len(result) > 1 and result[1] is not None else ""
            return allowed, reason
        if isinstance(result, bool):
            return result, "" if result else "API budget exhausted"
        return True, ""

    def _pick_call(self, reply: ModelReply) -> ToolCall | None:
        calls = list(getattr(reply, "tool_calls", None) or [])
        if not calls:
            return None
        call = calls[0]
        if isinstance(call, ToolCall):
            if not str(call.name or "").strip():
                return None
            if call.id:
                return call
            return ToolCall(id="call_1", name=call.name, arguments=dict(call.arguments or {}))
        name = str(getattr(call, "name", "") or "")
        call_id = str(getattr(call, "id", "") or "call")
        arguments = getattr(call, "arguments", {})
        if not name:
            return None
        return ToolCall(id=call_id, name=name, arguments=dict(arguments or {}))

    @staticmethod
    def _trim_reply_for_call(reply: ModelReply, call: ToolCall | None) -> ModelReply:
        """Return a reply clone declaring only the single tool call we execute.

        The chat API rejects an assistant message whose ``tool_calls`` are not all
        answered by matching ``role="tool"`` messages.  The story loop executes
        exactly one tool per turn (one decision per vision call), so any extra
        calls the model emitted must never be declared in the kept assistant
        message.
        """
        declared = list(getattr(reply, "tool_calls", None) or [])
        try:
            if call is None:
                if not declared:
                    return reply
                return replace(reply, tool_calls=[])
            if len(declared) == 1 and declared[0] is call:
                return reply
            return replace(reply, tool_calls=[call])
        except Exception:  # noqa: BLE001 - a weird reply object must not break the loop
            return reply

    def _reset_conversation_window(self) -> None:
        """Drop the conversation tail after a failed request.

        Keeps the system prompt, the task, and the persistent memory (they are
        rebuilt every turn); the next request therefore contains only the system
        message plus the latest observation/screenshot.
        """
        try:
            messages = getattr(self.context, "messages", None)
            if messages is not None and hasattr(messages, "clear"):
                messages.clear()
        except Exception:  # noqa: BLE001 - resetting must never raise
            pass
        try:
            recent = getattr(self.context, "recent_results", None)
            if recent is not None and hasattr(recent, "clear"):
                recent.clear()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _render_result_text(result: ToolResult) -> str:
        text = str(result.summary or "")
        if not result.ok:
            text = f"ERROR: {result.error or text}"
        return text[:MAX_TOOL_RESULT_CHARS]

    # -- prompt assembly ------------------------------------------------- #

    def _build_prompt_messages(
        self,
        *,
        user_text: str,
        shot: Mapping[str, Any],
        extra_system: str = "",
    ) -> list[dict[str, Any]]:
        """Build the OpenAI-style messages for the next vision call.

        Delegates to the shared :class:`AgentContext`; if its task rendering
        fails (defensive: an older context build), degrade gracefully by hiding
        the task object and rendering a compact task block ourselves instead of
        aborting the conversation.
        """
        png = shot.get("bytes")
        mime = str(shot.get("mime") or "image/png")
        detail = str(shot.get("detail") or self._cfg.image_detail)
        try:
            return self._sanitise_messages(
                self.context.build_messages(
                    user_text=user_text,
                    image_png=png,
                    mime=mime,
                    detail=detail,
                    extra_system=extra_system,
                )
            )
        except Exception as exc:  # noqa: BLE001 - never lose the conversation over framing
            self._context_errors.append(f"AgentContext: {type(exc).__name__}: {exc}")
            if len(self._context_errors) > 10:
                del self._context_errors[: len(self._context_errors) - 10]
            active_task = getattr(self.context, "active_task", None)
            task_block = self._render_task_block(active_task)
            combined = "\n\n".join(part for part in (extra_system, task_block) if part)
            try:
                self.context.active_task = None
                return self._sanitise_messages(
                    self.context.build_messages(
                        user_text=user_text,
                        image_png=png,
                        mime=mime,
                        detail=detail,
                        extra_system=combined,
                    )
                )
            except Exception:  # noqa: BLE001 - hand-build a minimal valid request
                content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
                if png:
                    content.append(image_content_block(png, mime, detail=detail))
                system = self.system_prompt
                if combined:
                    system = f"{system}\n\n{combined}"
                return [
                    {"role": "system", "content": system[: self.context.config.max_total_prompt_chars]},
                    {"role": "user", "content": content},
                ]
            finally:
                self.context.active_task = active_task

    @staticmethod
    def _render_task_block(task: Any) -> str:
        if task is None:
            return ""
        task_id = str(getattr(task, "task_id", "") or "")
        task_type = getattr(getattr(task, "task_type", None), "value", "") or ""
        description = str(getattr(task, "description", "") or "")
        payload = getattr(task, "payload", None) or {}
        lines = ["## ASSIGNED TASK"]
        if task_id:
            lines.append(f"id={task_id} type={task_type}")
        if description:
            lines.append(f"description: {description}")
        if payload:
            try:
                rendered = json.dumps(payload, ensure_ascii=False, default=str)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                rendered = str(payload)
            lines.append(f"payload: {rendered[:600]}")
        return "\n".join(lines)

    @staticmethod
    def _sanitise_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Drop orphan tool messages produced by transcript window trimming."""
        out: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "tool":
                recent = out[-2:]
                if not any(
                    item.get("role") == "assistant" and item.get("tool_calls") for item in recent
                ):
                    continue
            out.append(dict(message))
        return out

    # -- costs ------------------------------------------------------------ #

    def _prices(self) -> tuple[float, float, float]:
        api = getattr(self.config, "api", None)
        hit = float(getattr(api, "price_input_cache_hit_per_mtok", 0.003) or 0.003) if api else 0.003
        miss = (
            float(getattr(api, "price_input_cache_miss_per_mtok", 0.15) or 0.15) if api else 0.15
        )
        out = float(getattr(api, "price_output_per_mtok", 0.60) or 0.60) if api else 0.60
        return hit, miss, out

    def _estimate_cost(self, reply: ModelReply) -> float:
        usage = getattr(reply, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        hit = getattr(usage, "cache_hit_tokens", None)
        miss = getattr(usage, "cache_miss_tokens", None)
        hit_tokens = int(hit) if hit is not None else 0
        miss_tokens = int(miss) if miss is not None else max(prompt - hit_tokens, 0)
        hit_price, miss_price, out_price = self._prices()
        cost = (hit_tokens * hit_price + miss_tokens * miss_price + completion * out_price) / 1_000_000.0
        return round(max(0.0, cost), 8)

    def _cost_before(self) -> float | None:
        cost_fn = getattr(self.router, "costs_by_agent", None)
        if not callable(cost_fn):
            return None
        try:
            return float((cost_fn() or {}).get("story", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 - cost read must not stop the dialogue
            return None

    def _cost_after(self, before: float | None, reply: ModelReply) -> float:
        if before is not None:
            cost_fn = getattr(self.router, "costs_by_agent", None)
            if callable(cost_fn):
                try:
                    after = float((cost_fn() or {}).get("story", 0.0) or 0.0)
                    return round(max(0.0, after - before), 8)
                except Exception:  # noqa: BLE001
                    pass
        return self._estimate_cost(reply)

    # -- effects / proposals / summary ----------------------------------- #

    def _collect_effects(
        self,
        result: ToolResult,
        discoveries: list[dict[str, Any]],
        tasks_created: list[dict[str, Any]],
        dialog_texts: list[str],
    ) -> None:
        data = dict(result.data or {})
        knowledge = data.get("knowledge")
        if isinstance(knowledge, Mapping) and data.get("created"):
            public = StoryMemory.public_knowledge(knowledge)
            if public.get("content"):
                discoveries.append(public)
        dialogue_text = str(data.get("dialogue_text") or "")
        if dialogue_text and dialogue_text not in dialog_texts:
            dialog_texts.append(dialogue_text)
        proposal = data.get("proposal")
        if isinstance(proposal, Mapping) and data.get("created"):
            public_proposal = StoryMemory.public_proposal(proposal)
            if public_proposal.get("description"):
                tasks_created.append(public_proposal)

    def _auto_proposals(
        self,
        discoveries: Sequence[Mapping[str, Any]],
        tools: StoryTools,
        status: str,
    ) -> list[dict[str, Any]]:
        """Create honest follow-up proposals from this conversation's results."""
        if status != "dialog_ended":
            return []
        created: list[dict[str, Any]] = []
        seen_people: set[str] = set()
        limit = max(0, int(self._cfg.max_auto_proposals))
        for npc in list(tools.npc_updates):
            if len(created) >= limit:
                break
            if not npc.get("verified"):
                continue
            name = str(npc.get("name") or "").strip()
            if not name or name in seen_people:
                continue
            seen_people.add(name)
            proposal, made = self.story_memory.add_proposal(
                owner_agent="navigation",
                task_type="talk",
                description=f"Talk again with {name} to follow up on the conversation just read",
                priority="low",
                payload={
                    "target": {"kind": "person", "name": name},
                    "reason": "follow-up conversation proposed by the story agent",
                },
            )
            if made:
                created.append(StoryMemory.public_proposal(proposal))
        for discovery in discoveries:
            if len(created) >= limit:
                break
            if discovery.get("verification_status") != "verified":
                continue
            if discovery.get("category") not in ("story_clue", "quest"):
                continue
            content = " ".join(str(discovery.get("content") or "").split())
            if not content:
                continue
            proposal, made = self.story_memory.add_proposal(
                owner_agent="navigation",
                task_type="investigate",
                description=f"Investigate the lead just read on screen: {content[:140]}",
                priority="normal",
                payload={
                    "target": {"kind": "described", "description": content[:200]},
                    "reason": "verified story clue from a conversation",
                },
            )
            if made:
                created.append(StoryMemory.public_proposal(proposal))
        if not created and not tools.proposals_created:
            proposal, made = self.story_memory.add_proposal(
                owner_agent="navigation",
                task_type="explore",
                description=EXPLORATION_PROPOSAL,
                priority="low",
                payload={
                    "reason": "the last conversation yielded no concrete clue; keep exploring",
                    "source": "story_agent",
                },
            )
            if made:
                created.append(StoryMemory.public_proposal(proposal))
        return created

    def _build_summary(
        self,
        *,
        status: str,
        error: str,
        finish_summary: str,
        turns: int,
        discoveries: Sequence[Mapping[str, Any]],
        tasks_created: Sequence[Mapping[str, Any]],
        npcs: Sequence[Mapping[str, Any]],
        ever_in_dialog: bool,
    ) -> str:
        if finish_summary:
            summary = finish_summary
        elif status == "dialog_ended":
            if not ever_in_dialog:
                summary = (
                    "No dialogue box or menu was visible on screen, so nothing was advanced; "
                    "returning control."
                )
            else:
                summary = render_dialog_summary(
                    turns=turns,
                    knowledge=[dict(item) for item in discoveries],
                    proposals=len(tasks_created),
                    npcs=[dict(item) for item in npcs],
                    outcome="Dialogue finished and the normal game view returned",
                )
        elif status == "budget":
            summary = (
                f"Stopped after the turn/API budget was reached ({turns} turn(s)); the "
                "conversation may not be over."
            )
        elif status == "failed":
            summary = "The dialogue handling failed safely; see the error field."
        elif status == "control_denied":
            summary = "The story agent did not hold the control lease; no button was pressed."
        elif status == "interrupted":
            summary = (
                "The dialogue was interrupted (battle or pause); control is being handed back."
            )
        else:  # pragma: no cover - defensive
            summary = f"The dialogue ended with status {status!r}."
        if error and error not in summary and status in ("failed", "budget", "control_denied"):
            summary = f"{summary} error={error}"
        return " ".join(str(summary).split())[:1200]

    # -- events / finalisation ------------------------------------------- #

    def _event_info(self, event: Any) -> dict[str, Any] | None:
        if event is None:
            return None
        if isinstance(event, AgentMessage):
            return {
                "event_type": event.event_type.value,
                "source": event.source_agent.value,
                "payload": dict(event.payload),
            }
        if isinstance(event, EventType):
            return {"event_type": event.value, "source": "", "payload": {}}
        if isinstance(event, str):
            try:
                return {"event_type": EventType.coerce(event).value, "source": "", "payload": {}}
            except Exception:  # noqa: BLE001 - unknown strings are not events
                return None
        if isinstance(event, Mapping):
            event_type = event.get("event_type") or event.get("type")
            if event_type is None:
                return None
            payload = event.get("payload") or event.get("data") or {}
            return {
                "event_type": str(getattr(event_type, "value", event_type)),
                "source": str(event.get("source") or event.get("source_agent") or ""),
                "payload": dict(payload) if isinstance(payload, Mapping) else {},
            }
        event_type = getattr(event, "event_type", None)
        if event_type is not None:
            payload = getattr(event, "payload", {}) or {}
            return {
                "event_type": str(getattr(event_type, "value", event_type)),
                "source": str(getattr(event, "source", "") or ""),
                "payload": dict(payload) if isinstance(payload, Mapping) else {},
            }
        return None

    def _publish(self, event_type: EventType, payload: Mapping[str, Any]) -> Any:
        if self.bus is None:
            return None
        publish = getattr(self.bus, "publish", None) or getattr(self.bus, "emit", None)
        if not callable(publish):
            return None
        try:
            message = AgentMessage.create(
                event_type,
                source_agent=AgentRole.STORY,
                payload=dict(payload),
                priority=Priority.NORMAL,
            )
        except Exception as exc:  # noqa: BLE001 - protocol validation must not crash us
            self._publish_errors.append(f"build {event_type.value}: {type(exc).__name__}: {exc}")
            return None
        try:
            return publish(message)
        except Exception as exc:  # noqa: BLE001 - the bus is best effort for us
            self._publish_errors.append(f"publish {event_type.value}: {type(exc).__name__}: {exc}")
            if len(self._publish_errors) > 10:
                del self._publish_errors[: len(self._publish_errors) - 10]
            return None

    def _publish_result(self, result: Mapping[str, Any]) -> None:
        for discovery in result.get("discoveries") or []:
            content = str(discovery.get("content") or "")
            if not content:
                continue
            if discovery.get("verification_status") == "verified":
                self._publish(
                    EventType.STORY_DISCOVERY,
                    {
                        "content": content,
                        "category": str(discovery.get("category") or "other"),
                        "confidence": float(discovery.get("confidence") or 0.0),
                        "evidence_refs": list(discovery.get("evidence_refs") or []),
                    },
                )
            else:
                self._publish(
                    EventType.STORY_HYPOTHESIS,
                    {
                        "content": content,
                        "confidence": float(discovery.get("confidence") or 0.0),
                        "evidence_refs": list(discovery.get("evidence_refs") or []),
                    },
                )
        self._publish(
            EventType.DIALOG_ENDED,
            {
                "summary": str(result.get("summary") or ""),
                "npcs": [dict(item) for item in result.get("npcs") or []],
            },
        )
        status = str(result.get("status") or "")
        if status in ("failed", "budget", "control_denied"):
            self._publish(
                EventType.AGENT_ERROR,
                {
                    "agent": AgentRole.STORY.value,
                    "error": str(result.get("error") or status),
                    "context": {"status": status},
                },
            )

    def _finalize(
        self,
        status: str,
        *,
        summary: str = "",
        error: str = "",
        npcs: Sequence[Mapping[str, Any]] | None = None,
        discoveries: Sequence[Mapping[str, Any]] | None = None,
        tasks_created: Sequence[Mapping[str, Any]] | None = None,
        turns: int = 0,
        api_calls: int = 0,
        cost_usd: float = 0.0,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict:
        """Persist memory, update counters, publish events and build the result."""
        npc_list = [dict(item) for item in (npcs if npcs is not None else self.story_memory.npcs()[:20])]
        discovery_list = [dict(item) for item in (discoveries or [])]
        task_list = [dict(item) for item in (tasks_created or [])]
        evidence_dict = {
            "screens": list((evidence or {}).get("screens") or []),
            "frames": list((evidence or {}).get("frames") or []),
            "dialog_texts": list((evidence or {}).get("dialog_texts") or []),
        }
        result: dict[str, Any] = {
            "status": str(status),
            "summary": str(summary),
            "npcs": npc_list,
            "discoveries": discovery_list,
            "tasks_created": task_list,
            "turns": int(turns),
            "api_calls": int(api_calls),
            "cost_usd": round(float(cost_usd), 8),
            "evidence": evidence_dict,
            "error": str(error),
        }
        try:
            self.story_memory.save()
        except Exception as exc:  # noqa: BLE001 - memory persistence is best effort
            result["error"] = (result["error"] + f" memory save failed: {exc}").strip()
        counters = self._counters
        counters["dialogs"] = int(counters.get("dialogs", 0)) + 1
        counters["turns"] = int(counters.get("turns", 0)) + int(turns)
        counters["api_calls"] = int(counters.get("api_calls", 0)) + int(api_calls)
        counters["cost_usd"] = round(float(counters.get("cost_usd", 0.0)) + float(cost_usd), 8)
        counters["discoveries"] = int(counters.get("discoveries", 0)) + len(discovery_list)
        counters["proposals"] = int(counters.get("proposals", 0)) + len(task_list)
        if status == "control_denied":
            counters["denied"] = int(counters.get("denied", 0)) + 1
        if status == "failed":
            counters["failures"] = int(counters.get("failures", 0)) + 1
        self._last_status = str(status)
        self._last_summary = str(summary)
        self._last_error = str(error)
        self._publish_result(result)
        return result
