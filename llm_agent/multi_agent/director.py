"""Game Director: the deterministic multi-agent scheduler / state machine.

The director observes the single emulator, decides **which agent owns the game
right now**, grants the control lease, runs one bounded step of that agent, feeds
the progress monitor, and handles interruptions (battle / dialogue / transition)
by suspending the interrupted task and resuming it afterwards with a fresh
visual re-localization.

Design rules implemented here:

* the director itself is plain Python; DeepSeek is only asked for a high-level
  review during stagnation/conflict (``director_review``), and never per step;
* every button press happens inside a granted :class:`ControlLease` through the
  agent's :class:`GuardedEmulator` proxy — the director never presses buttons;
* battles and dialogues are *urgent events* that preempt scheduled tasks; the
  interrupted task is paused with its id kept, then resumed after the agent
  releases control and the screen is re-observed / navigation relocalized;
* tasks are created/deduplicated/persisted by the :class:`TaskManager`,
  knowledge by the :class:`SharedWorldMemory`, progress by the
  :class:`ProgressMonitor`, self-correction by the :class:`RecoveryManager`;
* the director never invents game progress: a task is only completed with
  evidence, and a rollback rewinds the game while keeping learned experience
  (handled by the session's checkpoint callbacks).

The module contains **no game knowledge** (no routes, NPCs, story order, species
data).  Contexts are classified purely from the pixel screen event.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from llm_agent.config import AgentConfig
from llm_agent.contracts import BlockedError, Frame, ScreenEvent
from llm_agent.multi_agent.agent_context import AgentContext, AgentContextConfig
from llm_agent.multi_agent.contracts import (
    AgentMessage,
    AgentRole,
    ControlDeniedError,
    ControlOwner,
    EventType,
    KnowledgeEntry,
    MultiAgentError,
    Priority,
    ProgressSignal,
    RecoveryLevel,
    StagnationKind,
    Task,
    TaskStatus,
    TaskType,
    VerificationStatus,
    now_ts,
)
from llm_agent.multi_agent.agent_registry import AgentRegistry
from llm_agent.multi_agent.progress_monitor import ProgressMonitor
from llm_agent.multi_agent.recovery import RecoveryManager
from llm_agent.multi_agent.scheduler import TaskScheduler
from llm_agent.multi_agent.shared_memory import SharedWorldMemory
from llm_agent.multi_agent.task_manager import TaskManager
from llm_agent.observation import ScreenAnalysis, analyze, save_png

#: Context names used in the director's audit trail / viewer.
CONTEXT_OVERWORLD = "overworld"
CONTEXT_BATTLE = "battle"
CONTEXT_DIALOG = "dialog"
CONTEXT_MENU = "menu"
CONTEXT_TRANSITION = "transition"
CONTEXT_LIMIT = "limit"
CONTEXT_IDLE = "idle"


# --------------------------------------------------------------------------- #
# Configuration / results
# --------------------------------------------------------------------------- #

@dataclass
class DirectorConfig:
    """Director behaviour limits (all safety budgets, no game knowledge)."""

    max_steps: int = 500
    navigation_steps_per_task: int = 120
    dialogue_max_turns: int = 14
    battle_max_turns: int = 24
    checkpoint_every_actions: int = 40
    checkpoint_every_minutes: float = 8.0
    max_emulator_actions: int = 20000
    max_runtime_minutes: float = 120.0
    max_consecutive_failures: int = 5
    max_decision_turns: int = 0
    #: director LLM review is rare and budget-gated
    director_llm_enabled: bool = True
    director_llm_on_stagnation_only: bool = True
    director_llm_max_tokens: int = 700
    #: progress / evidence
    save_screenshots: bool = True
    screenshot_every_steps: int = 1
    status_publish_every_steps: int = 1
    #: idle behaviour: propose story tasks at most this often
    story_proposal_interval_s: float = 90.0
    #: retry/attempt policy for agent task execution
    max_task_attempts: int = 3
    #: grace before a "no ready task" step is considered a failure
    idle_grace_steps: int = 3
    stop_file: str = ""
    pause_file: str = ""

    @classmethod
    def from_agent_config(cls, cfg: AgentConfig) -> "DirectorConfig":
        multi = cfg.multi_agent
        return cls(
            max_steps=int(multi.director_max_steps),
            navigation_steps_per_task=int(multi.navigation_steps_per_task),
            dialogue_max_turns=int(multi.dialogue_max_turns),
            battle_max_turns=int(multi.battle_max_turns),
            checkpoint_every_actions=int(multi.checkpoint_every_actions),
            checkpoint_every_minutes=float(multi.checkpoint_every_minutes),
            max_emulator_actions=int(multi.max_emulator_actions),
            max_runtime_minutes=float(multi.max_runtime_minutes),
            max_consecutive_failures=int(multi.max_consecutive_failures),
            max_decision_turns=int(multi.max_decision_turns),
            director_llm_enabled=bool(multi.director_llm_enabled),
            director_llm_on_stagnation_only=bool(multi.director_llm_on_stagnation_only),
            director_llm_max_tokens=int(multi.director_llm_max_tokens),
            save_screenshots=bool(multi.save_screenshots),
            story_proposal_interval_s=float(multi.director_llm_min_interval_s),
            stop_file=cfg.runtime.stop_file,
            pause_file=cfg.runtime.pause_file,
        )


@dataclass
class DirectorStep:
    """One auditable director iteration."""

    index: int
    context: str
    action: str
    detail: str = ""
    owner: str = ""
    task_id: str = ""
    ok: bool = True
    error: str = ""
    at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "context": self.context,
            "action": self.action,
            "detail": self.detail,
            "owner": self.owner,
            "task_id": self.task_id,
            "ok": self.ok,
            "error": self.error,
            "at": self.at or now_ts(),
        }


@dataclass
class DirectorRunResult:
    status: str
    reason: str
    exit_code: int
    steps: int = 0
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "exit_code": self.exit_code,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
        }


class DirectorStop(RuntimeError):
    """Raised internally to unwind the loop on a requested stop."""


# --------------------------------------------------------------------------- #
# Director
# --------------------------------------------------------------------------- #

class GameDirector:
    """Deterministic orchestrator for the three specialised agents."""

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        config: DirectorConfig | None = None,
        emulator: Any,
        bus: Any,
        arbiter: Any,
        tasks: TaskManager,
        scheduler: TaskScheduler,
        shared: SharedWorldMemory,
        monitor: ProgressMonitor,
        recovery: RecoveryManager,
        checkpoints: Any,
        checkpoint_store: Any,
        checkpoint_saver: Callable[[str], dict[str, Any]] | None = None,
        checkpoint_restorer: Callable[[str, str], dict[str, Any]] | None = None,
        router: Any = None,
        registry: AgentRegistry | None = None,
        navigation: Any = None,
        story: Any = None,
        pokemon: Any = None,
        audit: Any = None,
        clock: Callable[[], float] | None = None,
        on_status: Callable[[Mapping[str, Any]], None] | None = None,
        on_event: Callable[[Mapping[str, Any]], None] | None = None,
        on_branch_changed: Callable[..., Any] | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.config = config if config is not None else DirectorConfig.from_agent_config(cfg)
        self.emulator = emulator
        self.bus = bus
        self.arbiter = arbiter
        self.tasks = tasks
        self.scheduler = scheduler
        self.shared = shared
        self.monitor = monitor
        self.recovery = recovery
        self.checkpoints = checkpoints
        self.checkpoint_store = checkpoint_store
        self.router = router
        self.registry = registry or AgentRegistry()
        self.navigation = navigation
        self.story = story
        self.pokemon = pokemon
        self.audit = audit
        self._clock = clock or time.time
        self._on_status = on_status
        self._on_event = on_event
        self._on_branch_changed = on_branch_changed
        self._branch_override = ""
        self.run_dir = Path(run_dir) if run_dir else None
        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_restorer = checkpoint_restorer

        #: isolated agent working memory (used for checkpoint snapshots)
        self.story_context = AgentContext(
            agent=AgentRole.STORY,
            system_prompt=str(getattr(story, "system_prompt", "") or "Story agent (blind exploration)."),
            memory=getattr(story, "memory", None),
            config=AgentContextConfig(),
        )
        self.pokemon_context = AgentContext(
            agent=AgentRole.POKEMON,
            system_prompt=str(getattr(pokemon, "system_prompt", "") or "Pokémon agent (blind exploration)."),
            memory=getattr(pokemon, "memory", None),
            config=AgentContextConfig(),
        )

        # -- runtime state (persisted in checkpoints) -------------------- #
        self.step_index = 0
        self.frame_index = 0
        self.actions = 0
        self.buttons = 0
        self.action_batches = 0
        self.current_task_id = ""
        self.suspended_task_id = ""
        self.owner = ControlOwner.NONE
        self.last_event: dict[str, Any] = {}
        self.stats: dict[str, Any] = {
            "steps": 0,
            "owner_switches": 0,
            "battles": 0,
            "dialogs": 0,
            "transitions": 0,
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "checkpoints": 0,
            "rollbacks": 0,
            "stagnation_events": 0,
            "recoveries": 0,
            "localization_recoveries": 0,
            "director_reviews": 0,
            "errors": 0,
            "idle_steps": 0,
            "started_at": now_ts(),
        }
        self._stopped = False
        self._stop_reason = ""
        self._last_frame: Frame | None = None
        self._last_analysis: ScreenAnalysis | None = None
        self._screens_saved = 0
        self._last_checkpoint_action = 0
        self._last_checkpoint_at = self._clock()
        self._last_proposal_at = 0.0
        self._last_review_at = 0.0
        self._consecutive_errors = 0
        self._nav_seen_areas: set[str] = set()
        self._nav_seen_cells = 0
        self._localization_recoveries = 0
        self._localization_stuck = False
        self._explore_done_areas: dict[str, int] = {}
        self._waiting_tasks: dict[str, str] = {}
        #: message ids already applied to the world state (crash/restart idempotency)
        self._processed_message_ids: set[str] = set()
        self._processed_order: list[str] = []
        self._load_processed_messages()
        self._started_at = self._clock()
        self._recent_steps: list[DirectorStep] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def request_stop(self, reason: str = "user request") -> None:
        self._stopped = True
        self._stop_reason = str(reason or "user request")

    def run(self) -> DirectorRunResult:
        """Main loop; returns a structured result, never raises for game errors."""
        self._publish(
            EventType.SYSTEM_STARTED,
            {"branch_id": self._branch_id(), "run_dir": str(self.run_dir or ""), "model": self._model_for("director")},
        )
        status, reason = "finished", ""
        self._log("director_start", config=self.config.__dict__ if hasattr(self.config, "__dict__") else {})
        try:
            while True:
                if self._stopped:
                    status, reason = "stopped", self._stop_reason or "stop requested"
                    break
                if self._runtime_exceeded():
                    status, reason = "paused_limits", self._runtime_limit_reason()
                    break
                if self.config.max_steps and self.step_index >= self.config.max_steps:
                    status, reason = "finished", f"director_max_steps={self.config.max_steps}"
                    break
                if self.config.max_decision_turns and self.step_index >= self.config.max_decision_turns:
                    status, reason = "finished", f"max_decision_turns={self.config.max_decision_turns}"
                    break
                try:
                    step = self.step()
                except DirectorStop as exc:
                    status, reason = "stopped", str(exc)
                    break
                except (BlockedError, MultiAgentError) as exc:
                    self._consecutive_errors += 1
                    self.stats["errors"] += 1
                    self._log("director_step_error", error=f"{type(exc).__name__}: {exc}")
                    if self._consecutive_errors >= self.config.max_consecutive_failures:
                        status, reason = "blocked", f"too many director errors: {exc}"
                        break
                    continue
                except Exception as exc:  # noqa: BLE001 - a single bad step must not kill the run
                    self._consecutive_errors += 1
                    self.stats["errors"] += 1
                    self._log("director_step_exception", error=f"{type(exc).__name__}: {exc}")
                    if self._consecutive_errors >= self.config.max_consecutive_failures:
                        status, reason = "error", f"{type(exc).__name__}: {exc}"
                        break
                    continue
                self._consecutive_errors = 0
                self._record_step(step)
                if step.context == CONTEXT_LIMIT:
                    status, reason = "paused_limits", step.detail or "hard limit reached"
                    break
        finally:
            self._publish(EventType.SYSTEM_STOPPED, {"reason": reason, "status": status})
            self._log("director_stop", status=status, reason=reason, stats=dict(self.stats))
        exit_code = 0 if status in ("finished", "stopped", "paused_limits", "paused_budget") else 3
        return DirectorRunResult(status=status, reason=reason, exit_code=exit_code,
                                 steps=int(self.stats.get("steps", 0)), stop_reason=self._stop_reason)

    # ------------------------------------------------------------------ #
    # One director step
    # ------------------------------------------------------------------ #
    def step(self) -> DirectorStep:
        self._check_controls()
        self.step_index += 1
        self.stats["steps"] = self.step_index

        # 1. observe the real screen (read-only; allowed without the lease)
        frame, analysis, context = self._observe()

        # 2. process messages produced by agents
        self._drain_inbox()

        # 3. hard resource limits win over any task
        limit = self._hard_limit_reason()
        if limit:
            self._safe_pause(limit)
            return self._step(CONTEXT_LIMIT, "safe_pause", limit)

        # 4. urgent contexts preempt scheduled tasks
        if context == CONTEXT_BATTLE:
            return self._handle_battle(frame, analysis)
        if context in (CONTEXT_DIALOG, CONTEXT_MENU):
            return self._handle_dialog(frame, analysis, context)
        if context == CONTEXT_TRANSITION:
            return self._handle_transition(frame, analysis)

        # 5. progress monitor / self-correction
        recovery_action = self._check_progress()
        if recovery_action is not None:
            if recovery_action == "safe_pause":
                reason = "recovery requested a safe pause (hard limit / circuit breaker)"
                self._safe_pause(reason)
                return self._step(CONTEXT_LIMIT, "recovery_safe_pause", reason, ok=False)
            return self._step(CONTEXT_OVERWORLD, "recovery", recovery_action)

        # 6. periodic checkpoint
        self._maybe_checkpoint()

        # 7. normal scheduling
        return self._handle_overworld(frame, analysis)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _observe(self) -> tuple[Frame, ScreenAnalysis, str]:
        frame = self._observe_frame()
        analysis = analyze(frame, self._last_frame)
        self._last_frame = frame
        self._last_analysis = analysis
        self.frame_index = int(getattr(frame, "frame_index", 0) or 0)
        context = self._context_of(analysis)
        self._maybe_save_screen(frame, analysis)
        if self.step_index % max(1, self.config.status_publish_every_steps) == 0:
            self._publish_status()
        return frame, analysis, context

    def _observe_frame(self) -> Frame:
        observer = getattr(self.emulator, "observe", None)
        if not callable(observer):
            raise BlockedError("emulator does not support observe()")
        return observer()

    def _context_of(self, analysis: ScreenAnalysis) -> str:
        event = getattr(analysis, "event", ScreenEvent.UNKNOWN)
        if event is ScreenEvent.BATTLE:
            return CONTEXT_BATTLE
        if event is ScreenEvent.DIALOG:
            return CONTEXT_DIALOG
        if event is ScreenEvent.MENU:
            return CONTEXT_MENU
        if event in (ScreenEvent.TRANSITION, ScreenEvent.MAP_CHANGE):
            return CONTEXT_TRANSITION
        if bool(getattr(analysis, "is_black", False)):
            return CONTEXT_TRANSITION
        return CONTEXT_OVERWORLD

    def _maybe_save_screen(self, frame: Frame, analysis: ScreenAnalysis) -> None:
        if not self.config.save_screenshots or self.run_dir is None:
            return
        every = max(1, int(self.config.screenshot_every_steps))
        if self.step_index % every != 0:
            return
        try:
            target = self._evidence_dir() / "screens" / f"step_{self.step_index:05d}_{analysis.event.value}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            save_png(frame, target)
            self._screens_saved += 1
        except Exception as exc:  # noqa: BLE001 - evidence saving is best effort
            self._log("screen_save_failed", error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    # Battle takeover
    # ------------------------------------------------------------------ #
    def _handle_battle(self, frame: Frame, analysis: ScreenAnalysis) -> DirectorStep:
        self.last_event = {"event": EventType.BATTLE_STARTED.value, "frame": getattr(frame, "frame_index", 0)}
        self._publish(EventType.BATTLE_STARTED, {"frame": getattr(frame, "frame_index", 0)})
        self._suspend_current("battle_detected")
        self.monitor.mark_activity("battle", "battle detected")
        task = self._ensure_battle_task()
        if self.pokemon is None:
            if task is not None:
                self._fail_task(task, "pokemon agent unavailable")
            return self._step(CONTEXT_BATTLE, "battle_unavailable", "pokemon agent not constructed", ok=False)
        result: dict[str, Any] = {}
        try:
            self._request(AgentRole.POKEMON, reason="battle takeover", task=task)
            self._note_owner(AgentRole.POKEMON)
            if task is not None:
                self._start_task(task.task_id)
                self.current_task_id = task.task_id
            self.pokemon_context.begin_task(task)
            result = self.pokemon.handle_battle(
                emulator=self.arbiter.proxy(AgentRole.POKEMON),
                task=task,
                max_turns=int(self.config.battle_max_turns),
            )
            self._apply_agent_context_stats(self.pokemon_context, result)
        except ControlDeniedError as exc:
            result = {"status": "control_denied", "error": str(exc), "outcome": "unknown"}
        except Exception as exc:  # noqa: BLE001 - report and keep the system alive
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "outcome": "unknown"}
        finally:
            self._release(AgentRole.POKEMON, reason="battle step done", context={"result": result})
            buttons = int(result.get("buttons") or result.get("presses") or 0)
            self.buttons += buttons
            self.actions += buttons
            self.action_batches += 1
            if task is not None:
                self.current_task_id = ""
            self._note_owner(None)

        outcome = str(result.get("outcome") or "unknown")
        self._audit_agent_result("battle", result)
        self.monitor.observe_pokemon(
            battles=1,
            wins=1 if outcome == "win" else 0,
            losses=1 if outcome == "loss" else 0,
            catches=1 if outcome == "caught" else 0,
        )
        self._remember_lessons(result.get("lessons") or [], default_category="battle_lesson")
        self._publish(
            EventType.BATTLE_ENDED,
            {
                "outcome": outcome,
                "turns": int(result.get("turns") or 0),
                "task_id": task.task_id if task is not None else "",
                "status": str(result.get("status") or ""),
            },
        )
        status = str(result.get("status") or "failed")
        if task is not None:
            if status == "battle_ended":
                self._complete_task(task, f"battle ended: {outcome}", {"result": _compact(result)})
            elif status in ("interrupted", "budget"):
                self._pause_task(task, f"battle pending: {status}")
            else:
                self._fail_task(task, str(result.get("error") or status))
        self._post_interruption("battle")
        return self._step(
            CONTEXT_BATTLE,
            "pokemon_agent",
            f"status={status} outcome={outcome} turns={result.get('turns')} calls={result.get('api_calls')}",
            owner=AgentRole.POKEMON.value,
            task_id=task.task_id if task is not None else "",
            ok=status in ("battle_ended", "interrupted", "budget"),
            error=str(result.get("error") or ""),
        )

    # ------------------------------------------------------------------ #
    # Dialogue takeover
    # ------------------------------------------------------------------ #
    def _handle_dialog(self, frame: Frame, analysis: ScreenAnalysis, context: str) -> DirectorStep:
        self.last_event = {"event": EventType.DIALOG_STARTED.value, "frame": getattr(frame, "frame_index", 0), "context": context}
        self._publish(
            EventType.DIALOG_STARTED,
            {"frame": getattr(frame, "frame_index", 0), "context": context},
        )
        self._suspend_current("dialog_detected")
        self.monitor.mark_activity("dialog", "dialogue detected")
        task = self._ensure_dialog_task()
        self._unblock_story_tasks()
        # a previously blocked story task may now be runnable
        self._unblock_story_tasks()
        if self.story is None:
            if task is not None:
                self._fail_task(task, "story agent unavailable")
            return self._step(context, "story_unavailable", "story agent not constructed", ok=False)
        result: dict[str, Any] = {}
        try:
            self._request(AgentRole.STORY, reason="dialog takeover", task=task)
            self._note_owner(AgentRole.STORY)
            if task is not None:
                self._start_task(task.task_id)
                self.current_task_id = task.task_id
            self.story_context.begin_task(task)
            result = self.story.handle_dialog(
                emulator=self.arbiter.proxy(AgentRole.STORY),
                task=task,
                max_turns=int(self.config.dialogue_max_turns),
            )
            self._apply_agent_context_stats(self.story_context, result)
        except ControlDeniedError as exc:
            result = {"status": "control_denied", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._release(AgentRole.STORY, reason="dialog step done", context={"result": result})
            buttons = int(result.get("buttons") or result.get("presses") or 0)
            self.buttons += buttons
            self.actions += buttons
            self.action_batches += 1
            if task is not None:
                self.current_task_id = ""
            self._note_owner(None)

        discoveries = result.get("discoveries") or []
        npcs = result.get("npcs") or []
        self._audit_agent_result("dialog", result)
        self.monitor.observe_story(
            discoveries=len(discoveries),
            new_npcs=len(npcs),
            tasks_completed=1 if str(result.get("status")) == "dialog_ended" else 0,
            detail=str(result.get("summary") or "")[:180],
        )
        self._remember_discoveries(discoveries, default_category="story_clue")
        for npc in npcs:
            if isinstance(npc, Mapping):
                self._remember_discovery(
                    {
                        "category": "npc",
                        "content": f"{str(npc.get('name') or 'npc')}: {str(npc.get('notes') or '')}",
                        "verification_status": "verified" if npc.get("verified") else "inferred",
                        "confidence": 0.8 if npc.get("verified") else 0.5,
                    },
                    default_category="npc",
                )
        created = self._create_tasks_from_proposals(
            result.get("tasks_created") or [], requested_by=AgentRole.STORY
        )
        self._publish(
            EventType.DIALOG_ENDED,
            {
                "summary": str(result.get("summary") or "")[:300],
                "npcs": len(npcs),
                "discoveries": len(discoveries),
                "tasks_created": created,
                "task_id": task.task_id if task is not None else "",
                "status": str(result.get("status") or ""),
            },
        )
        status = str(result.get("status") or "failed")
        if task is not None:
            if status == "dialog_ended":
                self._complete_task(
                    task,
                    f"dialogue handled: {str(result.get('summary') or '')[:160]}",
                    {"discoveries": _compact(discoveries), "tasks_created": created},
                )
            elif status in ("interrupted", "budget"):
                self._pause_task(task, f"dialogue pending: {status}")
            else:
                self._fail_task(task, str(result.get("error") or status))
        self._post_interruption("dialog")
        return self._step(
            context,
            "story_agent",
            f"status={status} discoveries={len(discoveries)} npcs={len(npcs)} tasks={created}",
            owner=AgentRole.STORY.value,
            task_id=task.task_id if task is not None else "",
            ok=status in ("dialog_ended", "interrupted", "budget"),
            error=str(result.get("error") or ""),
        )

    # ------------------------------------------------------------------ #
    # Map transition
    # ------------------------------------------------------------------ #
    def _handle_transition(self, frame: Frame, analysis: ScreenAnalysis) -> DirectorStep:
        self.stats["transitions"] += 1
        if self.stats["transitions"] > 60:
            reason = "too many consecutive map transitions (possible transition loop)"
            self._safe_pause(reason)
            return self._step(CONTEXT_LIMIT, "transition_loop", reason, ok=False)
        self._suspend_current("transition_detected")
        self.monitor.mark_activity("transition", "map transition")
        if self.navigation is None or not getattr(self.navigation, "available", False):
            return self._step(CONTEXT_TRANSITION, "transition_no_navigation", "navigation unavailable")
        result: dict[str, Any] = {}
        try:
            self._request(AgentRole.NAVIGATION, reason="handle map transition")
            self._note_owner(AgentRole.NAVIGATION)
            result = self.navigation.handle_transition()
        except ControlDeniedError as exc:
            result = {"ok": False, "error": str(exc)}
        finally:
            self._release(AgentRole.NAVIGATION, reason="transition handled")
            self._note_owner(None)
        status = result.get("status") or {}
        area_id = str(status.get("area_id") or "")
        if area_id and area_id not in self._nav_seen_areas:
            self._nav_seen_areas.add(area_id)
            self._publish(EventType.NEW_AREA_DISCOVERED, {"area_id": area_id, "summary": "new area after transition"})
            self.shared.remember(
                "area",
                f"discovered area {area_id}",
                source_agent=AgentRole.NAVIGATION,
                confidence=0.9,
                verification_status=VerificationStatus.VERIFIED,
                evidence_refs=[f"screen:{self.frame_index}"],
                subject=area_id,
            )
            self.monitor.observe_navigation({"areas_total": len(self._nav_seen_areas)})
        self._publish(EventType.MAP_TRANSITION, {"area_id": area_id, "frame": self.frame_index})
        self._post_interruption("transition")
        return self._step(
            CONTEXT_TRANSITION,
            "navigation_transition",
            f"ok={result.get('ok')} area={area_id}",
            owner=AgentRole.NAVIGATION.value,
            ok=bool(result.get("ok")),
            error=str(result.get("error") or ""),
        )

    # ------------------------------------------------------------------ #
    # Normal scheduling
    # ------------------------------------------------------------------ #
    def _handle_overworld(self, frame: Frame, analysis: ScreenAnalysis) -> DirectorStep:
        self._resume_interrupted()
        self.tasks.refresh_ready()
        available = self._owner_availability()
        candidates = self.tasks.active()
        task = self.scheduler.next_task(candidates, owner_available=available) if candidates else None
        if task is None:
            task = self._ensure_explore_task()
        if task is None:
            self._maybe_story_proposals(frame)
            self.stats["idle_steps"] += 1
            return self._step(CONTEXT_IDLE, "idle", "no schedulable task; awaiting event or proposal")
        self.scheduler.on_selected(task)
        owner = task.owner_agent
        if owner is AgentRole.NAVIGATION:
            return self._run_navigation_task(task, frame)
        if owner is AgentRole.STORY:
            return self._run_story_task(task, frame)
        if owner is AgentRole.POKEMON:
            return self._run_pokemon_task(task, frame)
        self._fail_task(task, f"task owner {owner.value} cannot act")
        return self._step(CONTEXT_OVERWORLD, "unsupported_owner", owner.value, task_id=task.task_id, ok=False)

    def _owner_availability(self) -> dict[AgentRole, bool]:
        availability: dict[AgentRole, bool] = {}
        for role in (AgentRole.NAVIGATION, AgentRole.STORY, AgentRole.POKEMON):
            agent = self.registry.get(role) if self.registry is not None else None
            usable = agent is not None and self.registry.is_available(role)
            if role is AgentRole.NAVIGATION:
                usable = usable and bool(getattr(self.navigation, "available", False))
            availability[role] = usable
        return availability

    # -- navigation tasks ------------------------------------------------ #
    def _run_navigation_task(self, task: Task, frame: Frame) -> DirectorStep:
        if self.navigation is None or not getattr(self.navigation, "available", False):
            self._fail_task(task, "navigation subsystem unavailable")
            return self._step(CONTEXT_OVERWORLD, "navigation_unavailable", task.description, ok=False)
        payload = dict(task.payload or {})
        target = payload.get("target") if isinstance(payload.get("target"), Mapping) else None
        if target is None and payload.get("kind"):
            target = {key: payload[key] for key in ("kind", "x", "y", "area_id", "landmark", "max_steps") if key in payload}
        before = self.navigation.status()
        report: dict[str, Any] = {}
        try:
            self._request(AgentRole.NAVIGATION, reason=f"task:{task.task_type.value}", task=task)
            self._note_owner(AgentRole.NAVIGATION)
            self._start_task(task.task_id)
            self.current_task_id = task.task_id
            if target:
                report = self.navigation.navigate_to(target, max_steps=int(self.config.navigation_steps_per_task))
            else:
                report = self.navigation.explore(max_steps=int(self.config.navigation_steps_per_task))
        except ControlDeniedError as exc:
            report = {"ok": False, "status": "control_denied", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            report = {"ok": False, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._release(AgentRole.NAVIGATION, reason="navigation step done")
            moves = int(report.get("moves") or 0) + int(report.get("blocked") or 0)
            self.buttons += moves
            self.actions += moves
            self.action_batches += 1
            self.current_task_id = ""
            self._note_owner(None)
        after = self.navigation.status()
        self._feed_navigation_progress(before, after, report)
        status = str(report.get("status") or "error")
        ok_statuses = {"completed", "step_limit", "time_limit"}
        # A map transition / interruption is a normal navigation outcome: the
        # director handles it on the next step (urgent-context branch), so the
        # task is paused (not failed, not retried blindly).
        event_statuses = {"transition", "interrupted_event", "paused", "control_denied"}
        if status == "localization_lost":
            recovery = self._try_localization_recovery()
            if recovery.get("ok"):
                self._pause_task(task, "localization_lost: recovered, retry after relocalization")
            else:
                self._retry_or_fail(task, "localization_lost and re-attach failed")
            return self._step(
                CONTEXT_OVERWORLD, "localization_recovery",
                f"recovered={recovery.get('ok')} error={str(recovery.get('error') or '')[:80]}",
                owner=AgentRole.NAVIGATION.value, task_id=task.task_id,
                ok=bool(recovery.get("ok")),
                error=str(recovery.get("error") or ""),
            )
        if status in ok_statuses:
            if status == "completed":
                self._complete_task(task, f"navigation completed: {report.get('kind')}", {
                    "steps": report.get("steps"), "moves": report.get("moves"),
                    "coverage": report.get("coverage"), "area_id": after.get("area_id"),
                })
                if task.task_type is TaskType.EXPLORE:
                    self._explore_done_areas[str(after.get("area_id") or "")] = self.step_index
            else:
                self._pause_task(task, f"navigation budget: {status}")
            if self.bus is not None:
                self._publish(EventType.NAVIGATION_COMPLETED, {
                    "task_id": task.task_id, "steps": report.get("steps"), "moves": report.get("moves"),
                    "area_id": after.get("area_id", ""),
                })
            # completed navigation may unblock a waiting parent task
            self._maybe_unblock_parent(task)
        elif status in event_statuses:
            self._pause_task(task, f"navigation event: {status}")
            self.monitor.mark_activity("event", f"navigation {status}")
        else:
            self._publish(EventType.NAVIGATION_FAILED, {
                "reason": str(report.get("error") or status), "task_id": task.task_id,
            })
            self._retry_or_fail(task, str(report.get("error") or status))
        return self._step(
            CONTEXT_OVERWORLD,
            "navigation_task",
            f"{task.task_type.value}: {status} steps={report.get('steps')} area={after.get('area_id')}",
            owner=AgentRole.NAVIGATION.value,
            task_id=task.task_id,
            ok=status in ok_statuses,
            error=str(report.get("error") or ""),
        )

    def _try_localization_recovery(self) -> dict[str, Any]:
        """Ask Navigation v2 to re-attach after ``localization_lost`` (bounded)."""
        self._localization_recoveries += 1
        self.stats["localization_recoveries"] = int(self.stats.get("localization_recoveries", 0)) + 1
        recovery: dict[str, Any] = {}
        if self.navigation is not None and hasattr(self.navigation, "recover_localization"):
            try:
                recovery = self.navigation.recover_localization()
            except Exception as exc:  # noqa: BLE001
                recovery = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._publish(EventType.AGENT_ERROR, {
            "agent": AgentRole.NAVIGATION.value,
            "error": "localization_lost",
            "context": {"attempt": self._localization_recoveries,
                        "recovered": bool(recovery.get("ok"))},
        })
        self._log("localization_recovery", attempt=self._localization_recoveries, result=_compact(recovery))
        if self._localization_recoveries >= 8:
            self._localization_stuck = True
        return recovery

    def _feed_navigation_progress(self, before: Mapping[str, Any], after: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        try:
            self.monitor.observe_navigation(
                {
                    "cells_total": after.get("cells_total", 0),
                    "walkable": after.get("walkable", 0),
                    "coverage_ratio": after.get("coverage_ratio", 0.0),
                    "areas_total": after.get("areas_total", 0),
                    "unknown_frontier": after.get("unknown_frontier", 0),
                    "area_id": after.get("area_id", ""),
                    "localization_ok": bool((after.get("pose") or {}).get("valid")),
                }
            )
        except Exception:  # noqa: BLE001 - monitor must never break navigation
            pass
        new_cells = int(after.get("cells_total", 0) or 0) - int(before.get("cells_total", 0) or 0)
        new_areas = int(after.get("areas_total", 0) or 0) - int(before.get("areas_total", 0) or 0)
        area_id = str(after.get("area_id") or "")
        if area_id and area_id not in self._nav_seen_areas:
            self._nav_seen_areas.add(area_id)
            if self._nav_seen_areas:
                self._publish(EventType.NEW_AREA_DISCOVERED, {"area_id": area_id})
                self.shared.remember(
                    "area",
                    f"observed area {area_id}",
                    source_agent=AgentRole.NAVIGATION,
                    confidence=0.9,
                    verification_status=VerificationStatus.VERIFIED,
                    evidence_refs=[f"screen:{self.frame_index}"],
                    subject=area_id,
                )
        if new_cells > 0 or new_areas > 0:
            self._localization_recoveries = 0
            self._localization_stuck = False
            self.monitor.mark_activity("navigation", f"+{new_cells} cells, +{new_areas} areas")
            self._publish(EventType.PROGRESS_DETECTED, {"kind": "navigation_delta", "delta": new_cells})
            detail = f"navigation: +{new_cells} cells, +{new_areas} areas, movement={report.get('moves')}"
            self.monitor.mark_activity("navigation", detail)
            try:
                self.shared.remember(
                    "route",
                    f"verified navigation progress in area {area_id}: {new_cells} new cells",
                    source_agent=AgentRole.NAVIGATION,
                    confidence=0.7,
                    verification_status=VerificationStatus.VERIFIED,
                    evidence_refs=[f"screen:{self.frame_index}"],
                    subject=area_id or "unknown_area",
                )
            except Exception:  # noqa: BLE001
                pass

    # -- story tasks ----------------------------------------------------- #
    def _run_story_task(self, task: Task, frame: Frame) -> DirectorStep:
        """Run one bounded Story step (interaction probe or delegation)."""
        payload = dict(task.payload or {})
        if payload.get("probe_interaction") or payload.get("interact"):
            return self._run_story_probe(task, frame)
        # Without a dialogue on screen the story agent has nothing to read; ask
        # navigation to explore the area where the clue was found instead.
        area_id = str(payload.get("area_id") or "")
        self.tasks.block(task.task_id, reason="needs_dialog: no dialogue visible on screen")
        child, created = self.tasks.create_task(
            owner_agent=AgentRole.NAVIGATION,
            requested_by=AgentRole.STORY,
            task_type=TaskType.EXPLORE,
            description=f"explore to find the interaction point for story task {task.task_id}",
            parent_task_id=task.task_id,
            priority=Priority(int(task.priority)),
            branch_id=self._branch_id(),
            payload={"area_id": area_id} if area_id else {},
        )
        if created:
            self._waiting_tasks[child.task_id] = task.task_id
            self._publish(EventType.TASK_CREATED, {"task_id": child.task_id, "description": child.description,
                                                  "task_type": child.task_type.value,
                                                  "owner_agent": child.owner_agent.value})
            self.stats["tasks_created"] += 1
            return self._step(CONTEXT_OVERWORLD, "story_delegated", f"created navigation sub-task {child.task_id}",
                              owner=AgentRole.STORY.value, task_id=task.task_id)
        return self._step(CONTEXT_OVERWORLD, "story_blocked", "no dialogue visible; retry later",
                          owner=AgentRole.STORY.value, task_id=task.task_id, ok=False)

    def _run_story_probe(self, task: Task, frame: Frame) -> DirectorStep:
        """Let the Story agent acquire a dialogue on its own (bounded probe)."""
        if self.story is None:
            self._fail_task(task, "story agent unavailable")
            return self._step(CONTEXT_OVERWORLD, "story_unavailable", task.description, ok=False)
        result: dict[str, Any] = {}
        try:
            self._request(AgentRole.STORY, reason="interaction probe", task=task)
            self._note_owner(AgentRole.STORY)
            self._start_task(task.task_id)
            self.current_task_id = task.task_id
            self.story_context.begin_task(task)
            result = self.story.handle_dialog(
                emulator=self.arbiter.proxy(AgentRole.STORY),
                task=task,
                max_turns=int(self.config.dialogue_max_turns),
            )
            self._apply_agent_context_stats(self.story_context, result)
        except ControlDeniedError as exc:
            result = {"status": "control_denied", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._release(AgentRole.STORY, reason="interaction probe done", context={"result": result})
            buttons = int(result.get("buttons") or result.get("presses") or 0)
            self.buttons += buttons
            self.actions += buttons
            self.action_batches += 1
            self.current_task_id = ""
            self._note_owner(None)
        status = str(result.get("status") or "failed")
        discoveries = result.get("discoveries") or []
        npcs = result.get("npcs") or []
        self._audit_agent_result("interaction_probe", result)
        self.monitor.observe_story(
            discoveries=len(discoveries), new_npcs=len(npcs),
            tasks_completed=1 if status == "dialog_ended" else 0,
            detail=str(result.get("summary") or "")[:180],
        )
        self._remember_discoveries(discoveries, default_category="story_clue")
        created = self._create_tasks_from_proposals(result.get("tasks_created") or [],
                                                    requested_by=AgentRole.STORY)
        self._publish(EventType.DIALOG_ENDED, {
            "summary": str(result.get("summary") or "")[:300],
            "npcs": len(npcs), "discoveries": len(discoveries),
            "tasks_created": created, "task_id": task.task_id, "status": status,
        })
        if status == "dialog_ended":
            self._complete_task(task, f"interaction: {str(result.get('summary') or '')[:160]}",
                                {"discoveries": _compact(discoveries), "tasks_created": created})
        elif status in ("interrupted", "budget"):
            self._pause_task(task, f"interaction pending: {status}")
        else:
            self._fail_task(task, str(result.get("error") or status))
        self._post_interruption("interaction_probe")
        return self._step(
            CONTEXT_OVERWORLD, "story_probe",
            f"status={status} discoveries={len(discoveries)} npcs={len(npcs)} tasks={created}",
            owner=AgentRole.STORY.value, task_id=task.task_id,
            ok=status in ("dialog_ended", "interrupted", "budget"),
            error=str(result.get("error") or ""),
        )

    # -- pokemon tasks --------------------------------------------------- #
    def _run_pokemon_task(self, task: Task, frame: Frame) -> DirectorStep:
        self.tasks.block(task.task_id, reason="needs_battle: no battle in progress")
        return self._step(CONTEXT_OVERWORLD, "pokemon_blocked", "no battle in progress",
                          owner=AgentRole.POKEMON.value, task_id=task.task_id, ok=False)

    # ------------------------------------------------------------------ #
    # Interruption handling
    # ------------------------------------------------------------------ #
    def _suspend_current(self, reason: str) -> None:
        task_id = self.current_task_id
        if not task_id:
            return
        task = self.tasks.get(task_id)
        if task is None:
            self.current_task_id = ""
            return
        try:
            if task.status is TaskStatus.RUNNING:
                self.tasks.pause(task_id, reason=reason)
            elif task.status is TaskStatus.READY:
                self.tasks.pause(task_id, reason=reason)
        except Exception:  # noqa: BLE001 - pause is best effort
            pass
        self.suspended_task_id = task_id
        task.interruptions += 1
        try:
            self.tasks.save()
        except Exception:  # noqa: BLE001
            pass
        self.current_task_id = ""
        self._log("task_suspended", task_id=task_id, reason=reason)

    def _resume_interrupted(self) -> None:
        task_id = self.suspended_task_id
        if not task_id:
            return
        task = self.tasks.get(task_id)
        self.suspended_task_id = ""
        if task is None or task.status.terminal:
            return
        if task.status in (TaskStatus.PAUSED, TaskStatus.READY, TaskStatus.RUNNING):
            try:
                self.tasks.resume(task_id)
            except Exception:  # noqa: BLE001
                self._requeue_task(task_id, "resume after interruption")
            if task.owner_agent is AgentRole.NAVIGATION and self.navigation is not None:
                try:
                    self.navigation.resume()
                except Exception:  # noqa: BLE001
                    pass
            self._log("task_resumed", task_id=task_id)

    def _post_interruption(self, kind: str) -> None:
        """Re-observe the real screen and relocalize navigation after an event."""
        try:
            frame = self._observe_frame()
            analysis = analyze(frame, self._last_frame)
            self._last_frame = frame
            self._last_analysis = analysis
        except Exception as exc:  # noqa: BLE001
            self._log("post_interruption_observe_failed", error=f"{type(exc).__name__}: {exc}")
            analysis = None
        relocalized: dict[str, Any] = {}
        if self.navigation is not None and getattr(self.navigation, "available", False):
            try:
                relocalized = self.navigation.relocalize()
            except Exception as exc:  # noqa: BLE001
                relocalized = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._publish(
            EventType.RECOVERY_FINISHED,
            {
                "level": "interruption_handoff",
                "result": {
                    "kind": kind,
                    "relocalized": bool(relocalized.get("relocalized")),
                    "screen_event": getattr(analysis, "event", ScreenEvent.UNKNOWN).value if analysis else "",
                },
            },
        )
        if self.navigation is not None:
            self.navigation.resume()
        self._resume_interrupted()
        self._log("post_interruption", kind=kind, relocalized=bool(relocalized.get("relocalized")))

    # ------------------------------------------------------------------ #
    # Task helpers
    # ------------------------------------------------------------------ #
    def _ensure_explore_task(self) -> Task | None:
        availability = self._owner_availability()
        if not availability.get(AgentRole.NAVIGATION):
            return None
        for task in self.tasks.active():
            if task.owner_agent is AgentRole.NAVIGATION and task.task_type in (TaskType.EXPLORE, TaskType.NAVIGATE):
                return task
        area_id = ""
        coverage = 0.0
        frontier = 0
        if self.navigation is not None:
            nav_status = self.navigation.status()
            area_id = str(nav_status.get("area_id") or "")
            coverage = float(nav_status.get("coverage_ratio") or 0.0)
            frontier = int(nav_status.get("unknown_frontier") or 0)
        # A fully explored area must NOT spawn an endless stream of instant
        # "explore completed" tasks: switch once to crossing the area (known
        # exit) and then stay idle long enough for the story agent to propose new
        # objectives instead of re-running the same finished exploration.
        already_done = area_id in self._explore_done_areas
        done_recently = (
            already_done and self.step_index - self._explore_done_areas[area_id] < 20
        )
        area_finished = already_done or (coverage >= 0.95 and frontier <= 6)
        if area_id and area_finished:
            if done_recently:
                return None
            self._explore_done_areas[area_id] = self.step_index
            trigger: dict[str, Any] = {"kind": "exit"}
            exit_cell: tuple[int, int] | None = None
            if self.navigation is not None and hasattr(self.navigation, "local_map"):
                try:
                    local = self.navigation.local_map(radius=40) or {}
                    exits = [cell for cell in (local.get("cells") or [])
                             if str(cell.get("state")) == "exit"
                             and cell.get("x") is not None and cell.get("y") is not None]
                    pose = (self.navigation.status().get("pose") or {})
                    px, py = int(pose.get("x") or 0), int(pose.get("y") or 0)
                    if exits:
                        exits.sort(key=lambda cell: abs(int(cell["x"]) - px) + abs(int(cell["y"]) - py))
                        exit_cell = (int(exits[0]["x"]), int(exits[0]["y"]))
                        trigger = {"kind": "cell", "x": exit_cell[0], "y": exit_cell[1]}
                except Exception:  # noqa: BLE001 - fall back to the exit landmark
                    exit_cell = None
            description = (
                f"leave the explored area {area_id} through the known exit at {exit_cell}"
                if exit_cell is not None
                else f"leave the explored area {area_id} through a known exit"
            )
            task, created = self.tasks.create_task(
                owner_agent=AgentRole.NAVIGATION,
                requested_by=AgentRole.DIRECTOR,
                task_type=TaskType.NAVIGATE,
                description=description,
                priority=Priority.HIGH,
                branch_id=self._branch_id(),
                payload=trigger,
            )
            if created:
                self.stats["tasks_created"] += 1
                self._publish(EventType.TASK_CREATED, {
                    "task_id": task.task_id, "description": task.description,
                    "task_type": task.task_type.value, "owner_agent": task.owner_agent.value,
                })
            return task
        task, created = self.tasks.create_task(
            owner_agent=AgentRole.NAVIGATION,
            requested_by=AgentRole.DIRECTOR,
            task_type=TaskType.EXPLORE,
            description="explore the current area to discover exits, objects and clues (blind exploration)",
            priority=Priority.NORMAL,
            branch_id=self._branch_id(),
            payload={"area_id": area_id} if area_id else {},
        )
        if created:
            self.stats["tasks_created"] += 1
            self._publish(EventType.TASK_CREATED, {
                "task_id": task.task_id, "description": task.description,
                "task_type": task.task_type.value, "owner_agent": task.owner_agent.value,
            })
        return task

    def _ensure_battle_task(self) -> Task | None:
        for task in self.tasks.active():
            if task.owner_agent is AgentRole.POKEMON and task.task_type in (TaskType.BATTLE, TaskType.CATCH):
                return task
        task, created = self.tasks.create_task(
            owner_agent=AgentRole.POKEMON,
            requested_by=AgentRole.DIRECTOR,
            task_type=TaskType.BATTLE,
            description="handle the battle visible on screen using the real battle menus",
            priority=Priority.CRITICAL,
            branch_id=self._branch_id(),
            payload={"frame": self.frame_index},
        )
        if created:
            self.stats["tasks_created"] += 1
            self.stats["battles"] += 1
            self._publish(EventType.TASK_CREATED, {
                "task_id": task.task_id, "description": task.description,
                "task_type": task.task_type.value, "owner_agent": task.owner_agent.value,
            })
        return task

    def _ensure_dialog_task(self) -> Task | None:
        for task in self.tasks.active():
            if task.owner_agent is AgentRole.STORY and task.task_type in (
                TaskType.TALK, TaskType.INTERACT, TaskType.INVESTIGATE, TaskType.CUSTOM
            ):
                if task.status in (TaskStatus.RUNNING, TaskStatus.PAUSED):
                    return task
        task, created = self.tasks.create_task(
            owner_agent=AgentRole.STORY,
            requested_by=AgentRole.DIRECTOR,
            task_type=TaskType.TALK,
            description="read and understand the visible dialogue, record clues and propose follow-ups",
            priority=Priority.HIGH,
            branch_id=self._branch_id(),
            payload={"frame": self.frame_index},
        )
        if created:
            self.stats["tasks_created"] += 1
            self.stats["dialogs"] += 1
            self._publish(EventType.TASK_CREATED, {
                "task_id": task.task_id, "description": task.description,
                "task_type": task.task_type.value, "owner_agent": task.owner_agent.value,
            })
        return task

    def _complete_task(self, task: Task, result: str, evidence: Mapping[str, Any] | None = None) -> None:
        self._start_task(task.task_id)
        try:
            self.tasks.complete(task.task_id, result=result, evidence=dict(evidence or {}))
            self.stats["tasks_completed"] += 1
            self._publish(EventType.TASK_COMPLETED, {"task_id": task.task_id, "result": result[:300]})
        except Exception as exc:  # noqa: BLE001
            self._log("task_complete_failed", task_id=task.task_id, error=f"{type(exc).__name__}: {exc}")

    def _start_task(self, task_id: str) -> Task | None:
        """Move a PENDING/READY task to RUNNING using the manager's lifecycle rules."""
        task = self.tasks.get(task_id)
        if task is None or task.status.terminal:
            return task
        if task.status is TaskStatus.RUNNING:
            return task
        if task.status is TaskStatus.PENDING:
            try:
                self.tasks.refresh_ready()
            except Exception:  # noqa: BLE001
                pass
            task = self.tasks.get(task_id) or task
        if task.status in (TaskStatus.PAUSED, TaskStatus.BLOCKED):
            try:
                self.tasks.resume(task_id) if task.status is TaskStatus.PAUSED else self.tasks.unblock(task_id)
            except Exception:  # noqa: BLE001
                self._requeue_task(task_id, "start after interruption")
            task = self.tasks.get(task_id) or task
        try:
            return self.tasks.start(task_id)
        except Exception:  # noqa: BLE001 - retry once through the READY path
            try:
                self._requeue_task(task_id, "force start")
                return self.tasks.start(task_id)
            except Exception:  # noqa: BLE001
                return self.tasks.get(task_id)

    def _fail_task(self, task: Task, reason: str) -> None:
        try:
            self.tasks.fail(task.task_id, reason=reason[:500])
            self.stats["tasks_failed"] += 1
            self._publish(EventType.TASK_FAILED, {"task_id": task.task_id, "reason": reason[:300]})
        except Exception as exc:  # noqa: BLE001
            self._log("task_fail_failed", task_id=task.task_id, error=f"{type(exc).__name__}: {exc}")

    def _pause_task(self, task: Task, reason: str) -> None:
        try:
            if task.status in (TaskStatus.RUNNING, TaskStatus.READY, TaskStatus.PENDING):
                self.tasks.pause(task.task_id, reason=reason)
            self.suspended_task_id = task.task_id
        except Exception as exc:  # noqa: BLE001
            self._log("task_pause_failed", task_id=task.task_id, error=f"{type(exc).__name__}: {exc}")

    def _retry_or_fail(self, task: Task, reason: str) -> None:
        try:
            attempts = self.tasks.record_attempt(task.task_id)
        except Exception:  # noqa: BLE001 - attempts exhausted -> fail
            self._fail_task(task, f"attempts exhausted: {reason}")
            return
        if attempts >= int(self.config.max_task_attempts):
            self._fail_task(task, f"max attempts reached: {reason}")
            return
        self._requeue_task(task.task_id, f"retry after: {reason[:120]}")

    def _requeue_task(self, task_id: str, reason: str) -> None:
        """Move a task back to READY using the available TaskManager surface."""
        setter = getattr(self.tasks, "set_status", None)
        if callable(setter):
            try:
                setter(task_id, TaskStatus.READY, reason=reason)
                return
            except Exception:  # noqa: BLE001
                pass
        task = self.tasks.get(task_id)
        if task is None:
            return
        try:
            if task.status is TaskStatus.RUNNING:
                self.tasks.pause(task_id, reason=reason)
            self.tasks.resume(task_id)
        except Exception:  # noqa: BLE001 - already READY or terminal: nothing to do
            pass

    def _maybe_unblock_parent(self, task: Task) -> None:
        parent_id = self._waiting_tasks.pop(task.task_id, "") or str(task.parent_task_id or "")
        if not parent_id:
            return
        parent = self.tasks.get(parent_id)
        if parent is None:
            return
        reason = str(parent.reason)
        waiting = any(marker in reason for marker in ("needs_dialog", "waiting_for", "delegated"))
        if parent.status is TaskStatus.BLOCKED and waiting:
            try:
                self.tasks.unblock(parent_id)
            except Exception:  # noqa: BLE001
                pass

    def _unblock_story_tasks(self) -> None:
        for task in self.tasks.tasks(statuses=[TaskStatus.BLOCKED]):
            if task.owner_agent is AgentRole.STORY and "needs_dialog" in str(task.reason):
                try:
                    self.tasks.unblock(task.task_id)
                except Exception:  # noqa: BLE001
                    pass

    def _create_tasks_from_proposals(self, proposals: Sequence[Any], *, requested_by: AgentRole) -> list[str]:
        created_ids: list[str] = []
        for proposal in list(proposals)[:6]:
            if not isinstance(proposal, Mapping):
                continue
            try:
                owner = AgentRole.coerce(proposal.get("owner_agent") or AgentRole.NAVIGATION)
                if not owner.is_professional:
                    continue
                task_type = TaskType.coerce(proposal.get("task_type") or "custom")
                description = str(proposal.get("description") or "").strip()
                if not description:
                    continue
                task, created = self.tasks.create_task(
                    owner_agent=owner,
                    requested_by=requested_by,
                    task_type=task_type,
                    description=description,
                    parent_task_id=str(proposal.get("parent_task_id") or ""),
                    priority=Priority.coerce(proposal.get("priority", Priority.NORMAL)),
                    branch_id=self._branch_id(),
                    payload=dict(proposal.get("payload") or {}),
                )
                if created:
                    created_ids.append(task.task_id)
                    self.stats["tasks_created"] += 1
                    self._publish(EventType.TASK_CREATED, {
                        "task_id": task.task_id, "description": task.description,
                        "task_type": task.task_type.value, "owner_agent": task.owner_agent.value,
                    })
                    self._publish(EventType.TASK_ASSIGNED, {
                        "task_id": task.task_id, "owner_agent": task.owner_agent.value,
                    })
            except Exception as exc:  # noqa: BLE001 - a bad proposal must not break the step
                self._log("proposal_create_failed", error=f"{type(exc).__name__}: {exc}")
        return created_ids

    # ------------------------------------------------------------------ #
    # Story proposals when idle
    # ------------------------------------------------------------------ #
    def _maybe_story_proposals(self, frame: Frame | None = None) -> None:
        if self.story is None:
            return
        now = self._clock()
        if now - self._last_proposal_at < float(self.config.story_proposal_interval_s):
            return
        self._last_proposal_at = now
        propose = getattr(self.story, "propose_tasks", None)
        if not callable(propose):
            return
        try:
            context = {
                "navigation": self.navigation.status() if self.navigation is not None else {},
                "shared_memory": self.shared.render_for_prompt("", max_chars=1200,
                                                              agent=AgentRole.STORY),
                "tasks": [task.to_dict() for task in self.tasks.active()[:10]],
                "screen_event": getattr(self._last_analysis, "event", ScreenEvent.UNKNOWN).value
                if self._last_analysis is not None else "",
            }
            proposals = propose(context=context)
            created = self._create_tasks_from_proposals(proposals or [], requested_by=AgentRole.STORY)
            if created:
                self.monitor.mark_activity("story", f"story proposed {len(created)} task(s)")
                self._publish(EventType.STORY_HYPOTHESIS, {"content": f"{len(created)} task proposal(s) accepted"})
        except Exception as exc:  # noqa: BLE001
            self._log("story_proposal_failed", error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    # Progress monitoring / recovery
    # ------------------------------------------------------------------ #
    def _check_progress(self) -> str | None:
        """Run the progress monitor; return a recovery description or None.

        Returns ``"safe_pause"`` when the recovery manager asked for a hard safe
        pause (resource exhaustion / circuit breaker) instead of a rollback.
        """
        try:
            signals = self.monitor.consume_signals()
        except Exception:  # noqa: BLE001
            return None
        if not signals:
            return None
        self.stats["stagnation_events"] += 1
        kinds = [signal.value if isinstance(signal, StagnationKind) else str(signal) for signal in signals]
        self._publish(EventType.STAGNATION_DETECTED, {"kind": ",".join(kinds), "detail": "progress monitor"})
        self._log("stagnation_detected", signals=kinds)
        failed_task_ids = [task.task_id for task in self.tasks.tasks(statuses=[TaskStatus.FAILED])[-8:]]
        context = {
            "signals": kinds,
            "branch_id": self._branch_id(),
            "checkpoint_id": "latest",
            "failed_task_ids": failed_task_ids,
            "current_task": self.current_task_id,
            "suspended_task": self.suspended_task_id,
            "tasks": [task.to_dict() for task in self.tasks.tasks()[:20]],
            "navigation": self.navigation.summary() if self.navigation is not None else {},
            "monitor": self.monitor.summary(),
            "shared_memory": self.shared.render_for_prompt("", max_chars=1200),
            "stats": dict(self.stats),
        }
        try:
            result = self.recovery.handle(signals, context=context)
        except Exception as exc:  # noqa: BLE001 - recovery failure must not crash the director
            self._log("recovery_failed", error=f"{type(exc).__name__}: {exc}")
            return "recovery_failed"
        self.stats["recoveries"] += 1
        level = str(result.get("level") or "none")
        actions = [str(action) for action in (result.get("actions") or [])]
        self._log("recovery_handled", level=level, result=_compact(result))
        self._publish(EventType.RECOVERY_STARTED, {"level": level, "reason": ",".join(kinds)})
        hint = str(result.get("hint") or result.get("review") or "")
        try:
            self.monitor.reset_after_recovery(level=level)
        except Exception:  # noqa: BLE001
            pass
        if result.get("rollback"):
            self._after_rollback(result)
        elif level == RecoveryLevel.L3_ROLLBACK.value:
            self._after_rollback(result)
        self._publish(EventType.RECOVERY_FINISHED, {"level": level, "result": _compact(result)})
        if level == RecoveryLevel.NONE.value and ("safe_pause" in actions or not actions):
            return "safe_pause"
        detail = f"level={level}" + (f" hint={hint[:120]}" if hint else "")
        return detail

    def _after_rollback(self, result: Mapping[str, Any]) -> None:
        self.stats["rollbacks"] += 1
        rollback = dict(result.get("rollback") or {})
        new_branch = str(rollback.get("branch_id") or "")
        previous_branch = str(rollback.get("previous_branch_id") or "")
        if new_branch:
            self._branch_override = new_branch
            # propagate the new branch to tasks / shared memory / event bus
            if self._on_branch_changed is not None:
                try:
                    self._on_branch_changed(
                        new_branch,
                        previous_branch_id=previous_branch,
                        reason=str(result.get("reason") or "rollback"),
                    )
                except Exception as exc:  # noqa: BLE001 - branch bookkeeping must not crash
                    self._log("branch_switch_failed", error=f"{type(exc).__name__}: {exc}")
        self.current_task_id = ""
        self.suspended_task_id = ""
        self._last_frame = None
        self._last_analysis = None
        self._last_checkpoint_action = self.actions
        self._last_checkpoint_at = self._clock()
        try:
            frame = self._observe_frame()
            analysis = analyze(frame, None)
            self._last_frame = frame
            self._last_analysis = analysis
            self._maybe_save_screen(frame, analysis)
        except Exception as exc:  # noqa: BLE001
            self._log("rollback_observe_failed", error=f"{type(exc).__name__}: {exc}")
        relocalized: dict[str, Any] = {}
        if self.navigation is not None and getattr(self.navigation, "available", False):
            try:
                relocalized = self.navigation.relocalize()
            except Exception as exc:  # noqa: BLE001
                relocalized = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._publish(EventType.ROLLBACK_COMPLETED, {
            "branch_id": self._branch_id(),
            "previous_branch_id": str((result.get("rollback") or {}).get("previous_branch_id") or ""),
            "checkpoint_id": str((result.get("rollback") or {}).get("checkpoint_id") or ""),
            "reason": str(result.get("reason") or "stagnation"),
            "relocalized": bool(relocalized.get("relocalized")),
        })
        self._log("rollback_completed", relocalized=relocalized)

    # ------------------------------------------------------------------ #
    # Director LLM review (rare, high-level only)
    # ------------------------------------------------------------------ #
    def director_review(self, context: Mapping[str, Any], *, reason: str = "manual") -> dict[str, Any]:
        """Ask DeepSeek for a high-level coordination review (no button decisions)."""
        if not self.config.director_llm_enabled or self.router is None:
            return {"ok": False, "reason": "director LLM disabled or router missing"}
        allowed, why = self.router.can_call(AgentRole.DIRECTOR)
        if not allowed:
            return {"ok": False, "reason": why}
        now = self._clock()
        min_interval = 30.0
        if now - self._last_review_at < min_interval:
            return {"ok": False, "reason": "review cooldown"}
        frame = self._last_frame or self._observe_frame()
        try:
            from llm_agent.observation import encode_for_api

            image, mime = encode_for_api(
                frame,
                scale=max(1, min(4, int(self.cfg.multi_agent.image_scale_agent))),
                image_format=self.cfg.api.image_format,
                jpeg_quality=self.cfg.api.image_jpeg_quality,
            )
        except Exception:  # noqa: BLE001 - text-only review is still useful
            image, mime = b"", "image/png"
        prompt = self._review_prompt(context, reason)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._review_system_prompt()},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        if image:
            from llm_agent.multi_agent.agent_context import image_content_block

            messages[-1]["content"].append(image_content_block(image, mime, detail="low"))
        try:
            reply = self.router.complete(
                AgentRole.DIRECTOR,
                messages,
                max_tokens=int(self.config.director_llm_max_tokens),
                image_count=1 if image else 0,
                image_size=(160 * max(1, int(self.cfg.multi_agent.image_scale_agent)),
                            144 * max(1, int(self.cfg.multi_agent.image_scale_agent))),
                purpose=f"director_review:{reason}",
            )
        except Exception as exc:  # noqa: BLE001
            self._log("director_review_failed", error=f"{type(exc).__name__}: {exc}")
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        self._last_review_at = now
        self.stats["director_reviews"] += 1
        decision = _parse_json_object(reply.content)
        applied = self._apply_review(decision)
        self._log("director_review", reason=reason, decision=decision, applied=applied,
                  usage=reply.usage.to_dict())
        if self.audit is not None:
            try:
                self.audit.log("decisions", {
                    "kind": "director_review", "reason": reason, "content": (reply.content or "")[:2000],
                    "decision": decision, "applied": applied, "model": reply.model,
                    "usage": reply.usage.to_dict(),
                })
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "decision": decision, "applied": applied, "reason": reason}

    def _review_prompt(self, context: Mapping[str, Any], reason: str) -> str:
        tasks = context.get("tasks") or [task.to_dict() for task in self.tasks.tasks()[:20]]
        task_lines = []
        for task in tasks[:12]:
            if not isinstance(task, Mapping):
                continue
            task_lines.append(
                f"- {task.get('task_id')} owner={task.get('owner_agent')} type={task.get('task_type')} "
                f"status={task.get('status')} prio={task.get('priority')} attempts={task.get('attempts')} "
                f"progress={task.get('progress')} desc={str(task.get('description'))[:100]} reason={str(task.get('reason'))[:80]}"
            )
        monitor = self.monitor.summary()
        navigation_snapshot = context.get("navigation") or (
            self.navigation.summary() if self.navigation is not None else {}
        )
        return (
            "You are the high-level coordinator of a blind exploratory game-playing agent team.\n"
            f"Review reason: {reason}\n"
            f"Stagnation signals: {context.get('signals')}\n"
            f"Director step: {self.step_index}; runtime stats: {json.dumps(_compact(self.stats), ensure_ascii=False)}\n"
            f"Navigation status: {json.dumps(_compact(navigation_snapshot), ensure_ascii=False)}\n"
            f"Progress monitor: {json.dumps(_compact(monitor), ensure_ascii=False)}\n"
            "Task queue (all agents):\n" + ("\n".join(task_lines) or "- (empty)") + "\n"
            f"Shared knowledge (bounded):\n{str(context.get('shared_memory') or '')[:1800]}\n"
            "You cannot press buttons and must not specify battle tactics or individual keystrokes; "
            "only schedule/reprioritise/cancel tasks and create new high-level tasks.\n"
            "Answer with ONE JSON object only:\n"
            '{"review": "<=400 chars", '
            '"task_actions": [{"task_id": "...", "action": "prioritize|deprioritize|cancel|pause|resume", "priority": "low|normal|high|critical"}], '
            '"create_tasks": [{"owner_agent": "navigation|story|pokemon", "task_type": "explore|navigate|talk|investigate|battle|custom", '
            '"description": "...", "priority": "normal", "payload": {}}], '
            '"focus_agent": "navigation|story|pokemon|", "rollback": false}'
        )

    def _review_system_prompt(self) -> str:
        return (
            "You coordinate a team of specialised game agents. The game is controlled by one agent at a time. "
            "You only see the current screen and structured engine state; you have no walkthrough knowledge and "
            "must not invent facts about the game world. Keep every decision auditable and minimal."
        )

    def _apply_review(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        applied: list[dict[str, Any]] = []
        for action in list(decision.get("task_actions") or [])[:10]:
            if not isinstance(action, Mapping):
                continue
            task_id = str(action.get("task_id") or "")
            task = self.tasks.get(task_id)
            if task is None:
                continue
            verb = str(action.get("action") or "").lower()
            try:
                if verb == "cancel":
                    self.tasks.cancel(task_id, reason="director review")
                elif verb == "pause":
                    self.tasks.pause(task_id, reason="director review")
                elif verb == "resume":
                    self.tasks.resume(task_id)
                elif verb in ("prioritize", "deprioritize"):
                    priority = Priority.coerce(action.get("priority", Priority.HIGH if verb == "prioritize" else Priority.LOW))
                    self.tasks.set_priority(task_id, priority)
                applied.append({"task_id": task_id, "action": verb})
            except Exception as exc:  # noqa: BLE001
                applied.append({"task_id": task_id, "action": verb, "error": f"{type(exc).__name__}: {exc}"})
        created = self._create_tasks_from_proposals(decision.get("create_tasks") or [],
                                                    requested_by=AgentRole.DIRECTOR)
        applied.extend({"created_task": task_id} for task_id in created)
        return {"applied": applied, "created": created,
                "focus_agent": str(decision.get("focus_agent") or ""),
                "rollback": bool(decision.get("rollback"))}

    # ------------------------------------------------------------------ #
    # Inbox processing
    # ------------------------------------------------------------------ #
    def _drain_inbox(self) -> None:
        if self.bus is None:
            return
        try:
            messages = self.bus.drain(AgentRole.DIRECTOR, max_messages=20)
        except Exception as exc:  # noqa: BLE001
            self._log("bus_drain_failed", error=f"{type(exc).__name__}: {exc}")
            return
        for message in messages:
            try:
                self._handle_message(message)
            except Exception as exc:  # noqa: BLE001
                self._log("message_handle_failed", error=f"{type(exc).__name__}: {exc}",
                          message_id=getattr(message, "message_id", ""))
            finally:
                try:
                    self.bus.ack(message)
                except Exception:  # noqa: BLE001
                    pass

    def _handle_message(self, message: AgentMessage) -> None:
        # Message-id idempotency: a replayed message (same message_id after a
        # crash/restart) must never create duplicate tasks or duplicate knowledge.
        message_id = str(getattr(message, "message_id", "") or "")
        if message_id and message_id in self._processed_message_ids:
            self.last_event = {"event": "duplicate_message_ignored", "message_id": message_id}
            return
        event = message.event_type
        payload = dict(message.payload or {})
        if event is EventType.NAVIGATION_REQUEST:
            parent_task_id = str(message.parent_task_id or payload.get("parent_task_id") or "")
            if parent_task_id and self.tasks.get(parent_task_id) is None:
                parent_task_id = ""
            task, created = self.tasks.create_task(
                owner_agent=AgentRole.NAVIGATION,
                requested_by=message.source_agent,
                task_type=TaskType.NAVIGATE,
                description=str(payload.get("reason") or "navigation requested by another agent"),
                parent_task_id=parent_task_id,
                priority=Priority.HIGH,
                branch_id=self._branch_id(),
                payload=dict(payload.get("target") or {}),
            )
            if created:
                if parent_task_id:
                    self._waiting_tasks[task.task_id] = parent_task_id
                self._publish(EventType.TASK_CREATED, {"task_id": task.task_id, "description": task.description,
                                                      "task_type": task.task_type.value,
                                                      "owner_agent": task.owner_agent.value})
        elif event in (EventType.TASK_COMPLETED, EventType.TASK_FAILED, EventType.TASK_BLOCKED):
            task_id = str(payload.get("task_id") or "")
            if task_id:
                task = self.tasks.get(task_id)
                if task is not None and event is EventType.TASK_COMPLETED and not task.status.terminal:
                    try:
                        self._complete_task(task, str(payload.get("result") or "reported by agent"),
                                            {"reported_by": message.source_agent.value})
                    except Exception:  # noqa: BLE001
                        pass
        elif event is EventType.POKEMON_NEEDS_HELP:
            for task in self.tasks.tasks(statuses=[TaskStatus.PAUSED, TaskStatus.BLOCKED]):
                if task.owner_agent is AgentRole.POKEMON:
                    try:
                        self.tasks.unblock(task.task_id)
                    except Exception:  # noqa: BLE001
                        pass
        elif event is EventType.STORY_DISCOVERY:
            self._remember_discovery({"content": payload.get("content", ""),
                                     "category": payload.get("category", "story_clue"),
                                     "confidence": payload.get("confidence", 0.6),
                                     "verification_status": payload.get("verification_status", "inferred")})
        self._mark_message_processed(message_id)
        self.last_event = {"event": event.value, "source": message.source_agent.value}

    # -- message idempotency persistence --------------------------------- #
    def _processed_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return Path(self.run_dir) / "multi_agent" / "processed_messages.json"

    def _load_processed_messages(self) -> None:
        path = self._processed_path()
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ids = data.get("message_ids") if isinstance(data, dict) else data
            for message_id in list(ids or [])[-4000:]:
                text = str(message_id)
                if text:
                    self._processed_message_ids.add(text)
                    self._processed_order.append(text)
        except Exception:  # noqa: BLE001 - corrupt bookkeeping must not block a run
            return

    def _mark_message_processed(self, message_id: str) -> None:
        if not message_id or message_id in self._processed_message_ids:
            return
        self._processed_message_ids.add(message_id)
        self._processed_order.append(message_id)
        if len(self._processed_order) > 4000:
            drop = len(self._processed_order) - 4000
            for stale in self._processed_order[:drop]:
                self._processed_message_ids.discard(stale)
            del self._processed_order[:drop]
        path = self._processed_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp")
            tmp.write_text(json.dumps({"message_ids": self._processed_order[-4000:]}, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - bookkeeping is best effort
            pass

    # ------------------------------------------------------------------ #
    # Checkpoints / limits
    # ------------------------------------------------------------------ #
    def _maybe_checkpoint(self) -> None:
        if self._checkpoint_saver is None:
            return
        actions_due = self.actions - self._last_checkpoint_action >= int(self.config.checkpoint_every_actions)
        time_due = (
            self.config.checkpoint_every_minutes > 0
            and self._clock() - self._last_checkpoint_at >= float(self.config.checkpoint_every_minutes) * 60.0
        )
        if not (actions_due or time_due):
            return
        try:
            result = self._checkpoint_saver(f"periodic:step{self.step_index}")
            self.stats["checkpoints"] += 1
            self._last_checkpoint_action = self.actions
            self._last_checkpoint_at = self._clock()
            checkpoint_id = str(result.get("checkpoint_id") or result.get("id") or "")
            self._publish(EventType.CHECKPOINT_CREATED, {
                "checkpoint_id": checkpoint_id, "branch_id": self._branch_id(),
                "reason": f"periodic step {self.step_index}", "frame": self.frame_index,
            })
        except Exception as exc:  # noqa: BLE001
            self._log("checkpoint_failed", error=f"{type(exc).__name__}: {exc}")

    def _runtime_limit_reason(self) -> str:
        if self.config.max_runtime_minutes > 0:
            elapsed = self._clock() - self._started_at
            if elapsed >= float(self.config.max_runtime_minutes) * 60.0:
                return f"max_runtime_minutes={self.config.max_runtime_minutes} reached"
        if self.config.max_emulator_actions > 0 and self.actions >= int(self.config.max_emulator_actions):
            return f"max_emulator_actions={self.config.max_emulator_actions} reached"
        return ""

    def _runtime_exceeded(self) -> bool:
        return bool(self._runtime_limit_reason())

    def _hard_limit_reason(self) -> str:
        reason = self._runtime_limit_reason()
        if reason:
            return reason
        if self._localization_stuck:
            return ("navigation localization could not be recovered after "
                    f"{self._localization_recoveries} attempts")
        if self.router is not None and self.budget_exhausted():
            return "api budget exhausted"
        return ""

    def budget_exhausted(self) -> bool:
        """True when the shared API budget or a hard monitor limit is exhausted."""
        try:
            if self.monitor is not None and self.monitor.limits_exceeded() is not None:
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _safe_pause(self, reason: str) -> None:
        self._publish(EventType.LIMIT_REACHED, {"limit": reason, "detail": "safe pause"})
        self._log("safe_pause", reason=reason, stats=dict(self.stats))
        if self._checkpoint_saver is not None:
            try:
                self._checkpoint_saver(f"safe_pause:{reason}"[:120])
            except Exception as exc:  # noqa: BLE001
                self._log("safe_pause_checkpoint_failed", error=f"{type(exc).__name__}: {exc}")

    def _check_controls(self) -> None:
        if self.config.pause_file and Path(self.config.pause_file).is_file():
            while Path(self.config.pause_file).is_file():
                if self._stopped:
                    raise DirectorStop(self._stop_reason)
                time.sleep(1.0)
        if self.config.stop_file and Path(self.config.stop_file).is_file():
            self.request_stop(f"stop file: {self.config.stop_file}")
            raise DirectorStop(self._stop_reason)

    # ------------------------------------------------------------------ #
    # Knowledge helpers
    # ------------------------------------------------------------------ #
    def _remember_discoveries(self, discoveries: Sequence[Any], *, default_category: str) -> int:
        count = 0
        for discovery in discoveries[:10]:
            if self._remember_discovery(discovery, default_category=default_category):
                count += 1
        return count

    def _remember_discovery(self, discovery: Any, *, default_category: str = "story_clue") -> bool:
        if not isinstance(discovery, Mapping):
            return False
        content = str(discovery.get("content") or "").strip()
        if not content:
            return False
        category = str(discovery.get("category") or default_category)
        try:
            status = VerificationStatus.coerce(discovery.get("verification_status", "inferred"))
        except Exception:  # noqa: BLE001
            status = VerificationStatus.INFERRED
        try:
            self.shared.remember(
                category,
                content[:600],
                source_agent=AgentRole.STORY,
                confidence=float(discovery.get("confidence", 0.5 if status is not VerificationStatus.VERIFIED else 0.9)),
                verification_status=status,
                evidence_refs=[str(ref) for ref in (discovery.get("evidence_refs") or [])],
                subject=str(discovery.get("subject") or ""),
                branch_id=self._branch_id(),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._log("knowledge_store_failed", error=f"{type(exc).__name__}: {exc}")
            return False

    def _remember_lessons(self, lessons: Sequence[Any], *, default_category: str) -> int:
        count = 0
        for lesson in lessons[:10]:
            if self._remember_discovery(lesson, default_category=default_category):
                count += 1
        return count

    # ------------------------------------------------------------------ #
    # Status / persistence
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        tasks_stats = self.tasks.stats() if self.tasks is not None else {}
        api = self.router.usage() if self.router is not None else {"total": {}, "agents": {}}
        return {
            "step": self.step_index,
            "owner": self.owner.value,
            "current_task": self._current_task_summary(),
            "suspended_task": self.suspended_task_id,
            "branch_id": self._branch_id(),
            "last_event": dict(self.last_event),
            "tasks_completed": int(self.stats.get("tasks_completed", 0)),
            "tasks_failed": int(self.stats.get("tasks_failed", 0)),
            "tasks_created": int(self.stats.get("tasks_created", 0)),
            "battles": int(self.stats.get("battles", 0)),
            "dialogs": int(self.stats.get("dialogs", 0)),
            "owner_switches": int(self.stats.get("owner_switches", 0)),
            "transitions": int(self.stats.get("transitions", 0)),
            "checkpoints": int(self.stats.get("checkpoints", 0)),
            "rollbacks": int(self.stats.get("rollbacks", 0)),
            "stagnation_events": int(self.stats.get("stagnation_events", 0)),
            "recoveries": int(self.stats.get("recoveries", 0)),
            "localization_recoveries": int(self.stats.get("localization_recoveries", 0)),
            "director_reviews": int(self.stats.get("director_reviews", 0)),
            "errors": int(self.stats.get("errors", 0)),
            "idle_steps": int(self.stats.get("idle_steps", 0)),
            "actions": self.actions,
            "buttons": self.buttons,
            "frame_index": self.frame_index,
            "elapsed_s": round(self._clock() - self._started_at, 1),
            "task_stats": tasks_stats,
            "api": api,
            "navigation": self.navigation.summary() if self.navigation is not None else {},
            "monitor": self.monitor.summary() if self.monitor is not None else {},
        }

    def task_queue_snapshot(self, *, limit: int = 12) -> list[dict[str, Any]]:
        try:
            tasks = self.tasks.active()[:limit]
        except Exception:  # noqa: BLE001
            return []
        return [
            {
                "task_id": task.task_id,
                "owner": task.owner_agent.value,
                "type": task.task_type.value,
                "status": task.status.value,
                "priority": int(task.priority),
                "attempts": task.attempts,
                "progress": round(float(task.progress), 2),
                "description": task.description[:120],
            }
            for task in tasks
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "frame_index": self.frame_index,
            "actions": self.actions,
            "buttons": self.buttons,
            "action_batches": self.action_batches,
            "current_task_id": self.current_task_id,
            "suspended_task_id": self.suspended_task_id,
            "owner": self.owner.value,
            "last_event": dict(self.last_event),
            "stats": dict(self.stats),
            "branch_id": self._branch_id(),
            "nav_seen_areas": sorted(self._nav_seen_areas),
            "saved_at": now_ts(),
        }

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        """Restore the *current-branch* director state; spend/attempts never reset."""
        if not snapshot:
            return
        self.step_index = int(snapshot.get("step_index") or self.step_index)
        self.current_task_id = str(snapshot.get("current_task_id") or "")
        self.suspended_task_id = str(snapshot.get("suspended_task_id") or "")
        self.last_event = dict(snapshot.get("last_event") or {})
        # monotonic counters: a rollback must not rewind the cost/attempt accounting
        self.actions = max(self.actions, int(snapshot.get("actions") or 0))
        self.buttons = max(self.buttons, int(snapshot.get("buttons") or 0))
        self.action_batches = max(self.action_batches, int(snapshot.get("action_batches") or 0))
        try:
            self._nav_seen_areas = set(snapshot.get("nav_seen_areas") or self._nav_seen_areas)
        except Exception:  # noqa: BLE001
            pass
        incoming_stats = snapshot.get("stats") or {}
        for key in ("checkpoints", "rollbacks", "stagnation_events", "recoveries", "director_reviews", "errors"):
            self.stats[key] = max(int(self.stats.get(key, 0)), int(incoming_stats.get(key, 0) or 0))

    def navigation_snapshot(self) -> dict[str, Any]:
        if self.navigation is None:
            return {}
        try:
            return dict(self.navigation.snapshot())
        except Exception:  # noqa: BLE001
            return {}

    def stats_dict(self) -> dict[str, Any]:
        """Return a copy of the director's monotonic counters.

        ``self.stats`` is a dict attribute (persisted in snapshots); this helper
        exposes the same data under a method name for external callers.
        """
        return dict(self.stats)

    @property
    def buttons_pressed(self) -> int:
        """Monotonic count of verified button presses executed by the agents."""
        return int(self.buttons)

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #
    def _request(self, role: AgentRole, *, reason: str, task: Task | None = None) -> Any:
        lease = self.arbiter.request(
            role,
            reason=reason,
            task_id=task.task_id if task is not None else "",
            context={"task": task.task_id if task is not None else "", "reason": reason},
        )
        self._note_owner(role)
        return lease

    def _release(self, role: AgentRole, *, reason: str, context: Mapping[str, Any] | None = None) -> None:
        try:
            self.arbiter.release(role, reason=reason, context=dict(context or {}))
        except Exception:  # noqa: BLE001 - release must never raise
            pass
        if self.owner is ControlOwner.from_role(role):
            self.owner = ControlOwner.NONE

    def _note_owner(self, role: AgentRole | None) -> None:
        new_owner = ControlOwner.NONE if role is None else ControlOwner.from_role(role)
        if new_owner is not self.owner:
            self.stats["owner_switches"] += 1
            self.owner = new_owner

    def _publish(self, event_type: EventType, payload: Mapping[str, Any]) -> None:
        if self.bus is None:
            return
        try:
            message = AgentMessage.create(
                event_type,
                source_agent=AgentRole.DIRECTOR,
                target_agent=AgentRole.DIRECTOR,
                payload=dict(payload or {}),
                priority=Priority.NORMAL,
                branch_id=self._branch_id(),
            )
            self.bus.publish(message, dedupe=False)
        except Exception:  # noqa: BLE001 - bus hiccups never break the game loop
            pass
        if self._on_event is not None:
            try:
                self._on_event({"event": event_type.value, "payload": dict(payload or {}), "at": now_ts()})
            except Exception:  # noqa: BLE001
                pass

    def _publish_status(self) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(self.status())
        except Exception:  # noqa: BLE001
            pass

    def _step(
        self,
        context: str,
        action: str,
        detail: str = "",
        *,
        owner: str = "",
        task_id: str = "",
        ok: bool = True,
        error: str = "",
    ) -> DirectorStep:
        step = DirectorStep(
            index=self.step_index,
            context=context,
            action=action,
            detail=detail,
            owner=owner or self.owner.value,
            task_id=task_id,
            ok=ok,
            error=error,
            at=now_ts(),
        )
        if self.audit is not None:
            try:
                self.audit.log("director_steps", step.to_dict())
            except Exception:  # noqa: BLE001
                pass
        return step

    def _record_step(self, step: DirectorStep) -> None:
        self._recent_steps.append(step)
        if len(self._recent_steps) > 200:
            del self._recent_steps[:100]

    def _current_task_summary(self) -> str:
        task_id = self.current_task_id or self.suspended_task_id
        if not task_id:
            return ""
        task = self.tasks.get(task_id)
        if task is None:
            return task_id
        return f"{task.task_id} [{task.task_type.value}/{task.status.value}] {task.description[:80]}"

    def _branch_id(self) -> str:
        if self._branch_override:
            return self._branch_override
        branch = getattr(self.checkpoints, "current_branch", None)
        try:
            return str(getattr(branch, "branch_id", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _model_for(self, role: str) -> str:
        if self.router is not None:
            try:
                return str(self.router.model_for(role))
            except Exception:  # noqa: BLE001
                pass
        return str(getattr(self.cfg.api, "model", ""))

    def _evidence_dir(self) -> Path:
        root = str(self.cfg.multi_agent.evidence_root or "").strip()
        if root:
            return Path(root)
        if self.run_dir is not None:
            return self.run_dir / "multi_agent" / "evidence"
        return Path("artifacts") / "multi_agent_v1" / "evidence"

    def _apply_agent_context_stats(self, context: AgentContext, result: Mapping[str, Any]) -> None:
        try:
            context.stats["actions"] = int(context.stats.get("actions", 0)) + int(result.get("presses") or 0)
            context.stats["denied"] = int(context.stats.get("denied", 0)) + int(result.get("denied") or 0)
            context.stats["calls"] = int(context.stats.get("calls", 0)) + int(result.get("api_calls") or 0)
        except Exception:  # noqa: BLE001
            pass

    def _log(self, event: str, **fields: Any) -> None:
        if self.audit is not None:
            try:
                self.audit.log("multi_agent", {"event": event, **fields})
            except Exception:  # noqa: BLE001
                pass

    def _audit_agent_result(self, kind: str, result: Mapping[str, Any]) -> None:
        """Persist the agent's per-turn decisions/intro-wait as replayable evidence."""
        if self.audit is None:
            return
        try:
            self.audit.log(
                "agent_results",
                {
                    "kind": kind,
                    "at": now_ts(),
                    "status": result.get("status"),
                    "outcome": result.get("outcome"),
                    "turns": result.get("turns"),
                    "api_calls": result.get("api_calls"),
                    "cost_usd": result.get("cost_usd"),
                    "presses": result.get("presses") or result.get("buttons"),
                    "actions": _compact(result.get("actions") or [], limit=8000),
                    "intro_wait": _compact(result.get("intro_wait") or {}, limit=3000),
                    "lessons": _compact(result.get("lessons") or [], limit=4000),
                    "dialog_texts": _compact(result.get("evidence", {}).get("dialog_texts")
                                              if isinstance(result.get("evidence"), Mapping) else [],
                                              limit=3000),
                    "summary": str(result.get("summary") or "")[:500],
                },
            )
        except Exception:  # noqa: BLE001 - audit must never break the game loop
            pass


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _compact(value: Any, *, limit: int = 900) -> Any:
    """JSON-safe, size-bounded copy of a nested structure."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:  # noqa: BLE001
        return str(value)[:limit]
    if len(text) <= limit:
        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001
            return text
    return text[:limit] + "…"


def _parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of the first JSON object in a model reply."""
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return dict(value) if isinstance(value, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start:end + 1])
            return dict(value) if isinstance(value, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


__all__ = [
    "DirectorConfig",
    "DirectorRunResult",
    "DirectorStep",
    "DirectorStop",
    "GameDirector",
    "CONTEXT_OVERWORLD",
    "CONTEXT_BATTLE",
    "CONTEXT_DIALOG",
    "CONTEXT_MENU",
    "CONTEXT_TRANSITION",
    "CONTEXT_LIMIT",
    "CONTEXT_IDLE",
]
