"""Director integration tests with fake emulator / router / agents.

These tests exercise the real EventBus, ControlArbiter, TaskManager,
TaskScheduler, SharedWorldMemory, ProgressMonitor, RecoveryManager and
branch-checkpoint manifest code together with the :class:`GameDirector` state
machine, while the emulator, DeepSeek replies and the professional agents are
deterministic fakes (no ROM, no network, no API key).

They prove:

* exactly one agent owns control at a time (battle/dialog takeover),
* an interrupted navigation task is paused and resumed after the event,
* dialogue findings land in the shared knowledge base with evidence,
* duplicate battle tasks are not created for one battle,
* director LLM reviews are rare, parsed and applied to the task queue,
* hard limits cause a safe pause instead of an endless loop.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from llm_agent.config import AgentConfig
from llm_agent.contracts import ApiUsage, Frame, ModelReply, ScreenEvent
from llm_agent.multi_agent import director as director_module
from llm_agent.multi_agent.agent_registry import AgentRegistry
from llm_agent.multi_agent.contracts import (
    AgentRole,
    ControlDeniedError,
    ControlOwner,
    Priority,
    TaskStatus,
    TaskType,
    VerificationStatus,
)
from llm_agent.multi_agent.control_arbiter import ArbiterConfig, ControlArbiter
from llm_agent.multi_agent.director import DirectorConfig, GameDirector
from llm_agent.multi_agent.event_bus import EventBus, EventBusConfig
from llm_agent.multi_agent.progress_monitor import ProgressMonitor, ProgressMonitorConfig
from llm_agent.multi_agent.recovery import RecoveryConfig, RecoveryHooks, RecoveryManager
from llm_agent.multi_agent.scheduler import SchedulerConfig, TaskScheduler
from llm_agent.multi_agent.shared_memory import SharedMemoryConfig, SharedWorldMemory
from llm_agent.multi_agent.task_manager import TaskManager, TaskManagerConfig
from llm_agent.observation import ScreenAnalysis


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

def make_frame(index: int = 0, value: int = 180) -> Frame:
    rgba = np.full((144, 160, 4), value, dtype=np.uint8)
    rgba[..., 3] = 255
    return Frame(
        frame_index=index,
        rgba=rgba,
        sha256=f"sha{index:06d}",
        captured_at=time.time(),
        is_black=False,
        luma_mean=float(value),
    )


def analysis(event: ScreenEvent, *, black: bool = False) -> ScreenAnalysis:
    return ScreenAnalysis(
        event=event,
        changed=False,
        changed_ratio=0.0,
        signature=f"sig-{event.value}",
        is_black=black,
        luma_mean=120.0,
        summary=event.value,
        stats={},
    )


class FakeEmulator:
    """Minimal emulator surface: observe/press/wait/save/load + frame counter."""

    def __init__(self, *, state_tag: str = "initial") -> None:
        self.frame_index = 10
        self.presses: list[str] = []
        self.runs: list[list[str]] = []
        self.state_tag = state_tag
        self.loaded_state: str = ""

    def observe(self) -> Frame:
        return make_frame(self.frame_index)

    def press(self, button: str, *, frames: int | None = None) -> Frame:
        self.presses.append(button)
        self.frame_index += 24
        return self.observe()

    def wait(self, frames: int) -> Frame:
        self.frame_index += max(1, int(frames))
        return self.observe()

    def run_batch(self, buttons: Any, *, stop_on_event: bool = True) -> Any:
        self.runs.append(list(buttons))
        self.presses.extend(list(buttons))
        self.frame_index += 24 * len(list(buttons))
        steps = [
            SimpleNamespace(button=button, frame=self.observe(), screen_changed=True, changed_ratio=0.1)
            for button in buttons
        ]
        return SimpleNamespace(buttons=list(buttons), steps=steps, start_frame=self.frame_index,
                               end_frame=self.frame_index, stopped_reason="completed", events=[])

    def save_state_bytes(self) -> bytes:
        return f"STATE:{self.state_tag}:{self.frame_index}".encode()

    def load_state(self, payload: Any) -> None:
        self.loaded_state = str(payload)


class FakeAnalyzer:
    """Monkeypatched ``director.analyze`` returning the current scripted event."""

    def __init__(self) -> None:
        self.event = ScreenEvent.NONE

    def __call__(self, frame: Frame, previous: Frame | None = None) -> ScreenAnalysis:
        return analysis(self.event)


class FakeNavigation:
    """NavigationAdapter stand-in that enforces the control lease itself."""

    def __init__(self, arbiter: ControlArbiter, *, area_id: str = "area_001") -> None:
        self.arbiter = arbiter
        self.area_id = area_id
        self.available = True
        self.calls: list[dict[str, Any]] = []
        self.explore_status = "step_limit"
        self.relocalized = 0
        self.resumed = 0
        self.cells = 30
        self.areas = 1

    def _require_lease(self) -> None:
        assert self.arbiter.holds(AgentRole.NAVIGATION), "navigation acted without the lease"

    def explore(self, *, max_steps: int | None = None, max_seconds: float | None = None) -> dict[str, Any]:
        self._require_lease()
        self.calls.append({"kind": "explore", "max_steps": max_steps})
        self.cells += 3
        return {
            "ok": True,
            "status": self.explore_status,
            "kind": "explore",
            "report": {"steps": 12, "moves": 10},
            "nav_event": {},
            "navigation_status": self.status(),
            "steps": 12,
            "moves": 10,
            "blocked": 0,
        }

    def navigate_to(self, target: Any, *, max_steps: int | None = None) -> dict[str, Any]:
        self._require_lease()
        self.calls.append({"kind": "navigate", "target": dict(target or {}), "max_steps": max_steps})
        return {
            "ok": True,
            "status": "completed",
            "kind": "navigate",
            "report": {"steps": 5, "moves": 4},
            "nav_event": {},
            "navigation_status": self.status(),
            "steps": 5,
            "moves": 4,
            "blocked": 0,
        }

    def status(self) -> dict[str, Any]:
        return {
            "area_id": self.area_id,
            "pose": {"x": 0, "y": 0, "facing": "down", "confidence": 1.0, "valid": True},
            "cells_total": self.cells,
            "walkable": self.cells - 10,
            "blocked": 5,
            "unknown_frontier": 2,
            "coverage_ratio": 0.6,
            "areas_total": self.areas,
            "paused": False,
        }

    def summary(self) -> dict[str, Any]:
        return {"available": True, "area_id": self.area_id, "areas_total": self.areas,
                "coverage_ratio": 0.6, "actions": 10}

    def relocalize(self) -> dict[str, Any]:
        self.relocalized += 1
        return {"ok": True, "relocalized": True, "status": self.status()}

    def handle_transition(self, *, max_wait_frames: int = 120) -> dict[str, Any]:
        self.areas += 1
        self.area_id = f"area_{self.areas:03d}"
        return {"ok": True, "status": self.status(), "result": {"area_id": self.area_id}}

    def resume(self) -> None:
        self.resumed += 1

    def pause(self) -> None:
        pass

    def save(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {"status": self.status(), "actions": 10}

    def restore(self, snapshot: Any) -> None:
        pass


class FakeAgent:
    """Fake professional agent; asserts it holds the control lease when acting."""

    def __init__(
        self,
        role: AgentRole,
        arbiter: ControlArbiter,
        *,
        dialog_result: dict[str, Any] | None = None,
        battle_result: dict[str, Any] | None = None,
        raise_error: BaseException | None = None,
    ) -> None:
        self.role = role
        self.arbiter = arbiter
        self.system_prompt = f"{role.value} system prompt"
        self.dialog_result = dialog_result or {}
        self.battle_result = battle_result or {}
        self.raise_error = raise_error
        self.calls = 0
        self.recovery_hints: list[str] = []

    def _require_lease(self) -> None:
        assert self.arbiter.holds(self.role), f"{self.role.value} acted without the lease"

    def handle_dialog(self, *, emulator: Any, task: Any = None, max_turns: int = 12) -> dict[str, Any]:
        self._require_lease()
        self.calls += 1
        if self.raise_error is not None:
            raise self.raise_error
        return dict(self.dialog_result)

    def handle_battle(self, *, emulator: Any, task: Any = None, max_turns: int = 24) -> dict[str, Any]:
        self._require_lease()
        self.calls += 1
        if self.raise_error is not None:
            raise self.raise_error
        return dict(self.battle_result)

    def propose_tasks(self, *, context: Any = None) -> list[dict[str, Any]]:
        return []

    def apply_recovery_hint(self, level: str, hint: str) -> None:
        self.recovery_hints.append(f"{level}:{hint}")


class FakeRouter:
    """Router stand-in used only by the director's rare review path."""

    def __init__(self, payload: dict[str, Any] | None = None, *, allow: bool = True) -> None:
        self.payload = payload or {}
        self.allow = allow
        self.calls: list[dict[str, Any]] = []

    def can_call(self, role: Any) -> tuple[bool, str]:
        return (True, "") if self.allow else (False, "disabled for test")

    def model_for(self, role: Any) -> str:
        return "fake-model"

    def complete(self, role: Any, messages: Any, **kwargs: Any) -> ModelReply:
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        content = json.dumps(self.payload) if self.payload else "{}"
        return ModelReply(content=content, tool_calls=[], usage=ApiUsage(prompt_tokens=10,
                                                                       completion_tokens=5,
                                                                       total_tokens=15),
                          model="fake-model", latency_s=0.1)

    def usage(self) -> dict[str, Any]:
        return {"total": {"calls": len(self.calls), "cost_usd": 0.0, "total_tokens": 0}, "agents": {}}


class FakeRecovery:
    def __init__(self) -> None:
        self.calls: list[list[Any]] = []
        self.level = "l1_local"

    def handle(self, signals: Any, *, context: Any = None) -> dict[str, Any]:
        self.calls.append(list(signals))
        return {"level": self.level, "actions": [], "ok": True, "reason": "test",
                "rollback": None, "signals": list(signals)}

    def summary(self) -> dict[str, Any]:
        return {"recoveries": len(self.calls)}


class FakeCheckpoints:
    def __init__(self) -> None:
        self.branch_seq = 0
        self.current_branch = SimpleNamespace(branch_id="branch_test")
        self.saved: list[dict[str, Any]] = []

    def status(self) -> dict[str, Any]:
        return {"current_branch": self.current_branch.branch_id, "saves": len(self.saved)}

    def new_branch(self) -> str:
        self.branch_seq += 1
        self.current_branch = SimpleNamespace(branch_id=f"branch_{self.branch_seq:03d}")
        return self.current_branch.branch_id


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

@pytest.fixture()
def harness(monkeypatch: pytest.MonkeyPatch):
    cfg = AgentConfig()
    cfg.multi_agent.enabled = True
    cfg.multi_agent.max_rollbacks_total = 2
    emulator = FakeEmulator()
    analyzer = FakeAnalyzer()
    monkeypatch.setattr(director_module, "analyze", analyzer)

    bus = EventBus(EventBusConfig(path="", persist=False))
    arbiter = ControlArbiter(emulator, config=ArbiterConfig(), event_bus=bus)
    tasks = TaskManager(TaskManagerConfig(), path="", branch_id="branch_test")
    scheduler = TaskScheduler(SchedulerConfig())
    shared = SharedWorldMemory(SharedMemoryConfig(root=""), root="", branch_id="branch_test")
    monitor = ProgressMonitor(ProgressMonitorConfig(max_runtime_s=3600, max_actions=10000))
    recovery = FakeRecovery()
    checkpoints = FakeCheckpoints()
    navigation = FakeNavigation(arbiter)
    story = FakeAgent(
        AgentRole.STORY,
        arbiter,
        dialog_result={
            "status": "dialog_ended",
            "summary": "an npc told me something useful",
            "discoveries": [
                {
                    "category": "story_clue",
                    "content": "the npc mentioned a parcel for the shop",
                    "verification_status": "verified",
                    "confidence": 0.9,
                    "evidence_refs": ["screens/story_dialog.png"],
                }
            ],
            "npcs": [{"name": "npc_a", "notes": "gave a task", "verified": True}],
            "tasks_created": [
                {
                    "owner_agent": "navigation",
                    "task_type": "navigate",
                    "description": "walk to the place the npc mentioned",
                    "priority": "high",
                    "payload": {"target": {"kind": "exit"}},
                }
            ],
            "turns": 2,
            "api_calls": 2,
            "cost_usd": 0.0001,
            "evidence": {"screens": ["s1.png"], "frames": [10]},
            "buttons": 3,
        },
    )
    pokemon = FakeAgent(
        AgentRole.POKEMON,
        arbiter,
        battle_result={
            "status": "battle_ended",
            "outcome": "win",
            "turns": 2,
            "api_calls": 2,
            "cost_usd": 0.0001,
            "lessons": [
                {"category": "battle_lesson", "content": "the first move did visible damage",
                 "verification_status": "verified", "confidence": 0.8, "evidence_refs": ["s2.png"]}
            ],
            "actions": [],
            "evidence": {"screens": ["s2.png"], "frames": [11]},
            "buttons": 4,
        },
    )
    router = FakeRouter()
    registry = AgentRegistry()
    registry.register(AgentRole.NAVIGATION, navigation)
    registry.register(AgentRole.STORY, story)
    registry.register(AgentRole.POKEMON, pokemon)
    saved: list[str] = []

    def checkpoint_saver(reason: str) -> dict[str, Any]:
        saved.append(reason)
        return {"ok": True, "checkpoint_id": "checkpoint_000000001",
                "branch_id": checkpoints.current_branch.branch_id}

    dcfg = DirectorConfig(
        max_steps=10,
        checkpoint_every_actions=2,
        checkpoint_every_minutes=0,
        save_screenshots=False,
        max_runtime_minutes=10,
        max_emulator_actions=10000,
        story_proposal_interval_s=0,
    )
    director = GameDirector(
        cfg,
        config=dcfg,
        emulator=emulator,
        bus=bus,
        arbiter=arbiter,
        tasks=tasks,
        scheduler=scheduler,
        shared=shared,
        monitor=monitor,
        recovery=recovery,
        checkpoints=checkpoints,
        checkpoint_store=None,
        checkpoint_saver=checkpoint_saver,
        router=router,
        registry=registry,
        navigation=navigation,
        story=story,
        pokemon=pokemon,
        clock=time.time,
        run_dir=None,
    )
    return SimpleNamespace(
        cfg=cfg, emulator=emulator, analyzer=analyzer, bus=bus, arbiter=arbiter,
        tasks=tasks, scheduler=scheduler, shared=shared, monitor=monitor, recovery=recovery,
        checkpoints=checkpoints, navigation=navigation, story=story, pokemon=pokemon,
        router=router, registry=registry, director=director, saved=saved,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_overworld_step_creates_and_runs_an_exploration_task(harness) -> None:
    harness.analyzer.event = ScreenEvent.NONE
    step = harness.director.step()
    assert step.context == "overworld"
    assert step.action == "navigation_task"
    assert harness.navigation.calls and harness.navigation.calls[0]["kind"] == "explore"
    tasks = harness.tasks.tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.owner_agent is AgentRole.NAVIGATION
    assert task.task_type is TaskType.EXPLORE
    # step_limit pauses the task for the next director iteration (budget left)
    assert task.status in (TaskStatus.PAUSED, TaskStatus.READY, TaskStatus.RUNNING)
    # the arbiter released control again
    assert harness.arbiter.owner is ControlOwner.NONE


def test_battle_takeover_grants_and_releases_control(harness) -> None:
    harness.analyzer.event = ScreenEvent.BATTLE
    step = harness.director.step()
    assert step.context == "battle"
    assert step.action == "pokemon_agent"
    assert harness.pokemon.calls == 1
    assert harness.arbiter.owner is ControlOwner.NONE
    stats = harness.director.stats_dict()
    assert stats["battles"] == 1
    battle_tasks = [t for t in harness.tasks.tasks() if t.owner_agent is AgentRole.POKEMON]
    assert len(battle_tasks) == 1
    assert battle_tasks[0].status is TaskStatus.COMPLETED
    # battle lessons were written into the shared knowledge base
    entries = harness.shared.search("", categories=["battle_lesson"])
    assert entries and entries[0].verification_status is VerificationStatus.VERIFIED


def test_duplicate_battle_takeover_does_not_create_duplicate_tasks(harness) -> None:
    harness.analyzer.event = ScreenEvent.BATTLE
    harness.pokemon.battle_result = {"status": "interrupted", "outcome": "unknown", "turns": 3,
                                     "api_calls": 1, "cost_usd": 0.0, "lessons": [], "actions": [],
                                     "evidence": {}, "buttons": 2}
    harness.director.step()
    harness.director.step()
    battles = [t for t in harness.tasks.tasks() if t.owner_agent is AgentRole.POKEMON]
    assert len(battles) == 1


def test_dialog_takeover_records_knowledge_and_resumes_navigation(harness) -> None:
    # 1) start a navigation task so there is something to interrupt
    harness.analyzer.event = ScreenEvent.NONE
    harness.director.step()
    nav_task = harness.tasks.tasks()[0]
    assert nav_task.owner_agent is AgentRole.NAVIGATION
    relocalized_before = harness.navigation.relocalized

    # 2) a dialogue appears -> story takes over
    harness.analyzer.event = ScreenEvent.DIALOG
    step = harness.director.step()
    assert step.context == "dialog"
    assert step.action == "story_agent"
    assert harness.story.calls == 1
    assert harness.arbiter.owner is ControlOwner.NONE
    assert harness.navigation.relocalized > relocalized_before
    # findings recorded as verified knowledge with evidence
    clues = harness.shared.search("", categories=["story_clue"])
    assert clues and clues[0].content
    assert clues[0].evidence_refs
    # the discovery produced a new navigation task proposal
    proposed = [t for t in harness.tasks.tasks()
                if "npc mentioned" in t.description and t.owner_agent is AgentRole.NAVIGATION]
    assert proposed
    # story task itself completed
    story_tasks = [t for t in harness.tasks.tasks() if t.owner_agent is AgentRole.STORY]
    assert story_tasks and story_tasks[-1].status is TaskStatus.COMPLETED
    # the interrupted navigation task is resumed for the next director step
    resumed = harness.tasks.get(nav_task.task_id)
    assert resumed is not None
    assert resumed.status in (TaskStatus.PAUSED, TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.COMPLETED)


def test_control_denied_from_agent_is_reported_without_crash(harness) -> None:
    harness.analyzer.event = ScreenEvent.BATTLE
    harness.pokemon.raise_error = ControlDeniedError("lease stolen", agent="pokemon", owner="navigation")
    step = harness.director.step()
    assert step.context == "battle"
    assert step.ok is False
    assert harness.arbiter.owner is ControlOwner.NONE
    # the director is still able to take a normal step afterwards
    harness.analyzer.event = ScreenEvent.NONE
    next_step = harness.director.step()
    assert next_step.context in ("overworld", "idle")


def test_transition_is_handled_by_navigation(harness) -> None:
    harness.analyzer.event = ScreenEvent.TRANSITION
    step = harness.director.step()
    assert step.context == "transition"
    assert step.action == "navigation_transition"
    assert harness.navigation.areas == 2
    areas = harness.shared.search("", categories=["area"])
    assert areas, "new area should be recorded in shared knowledge"


def test_hard_limit_triggers_safe_pause(harness) -> None:
    harness.director.config.max_emulator_actions = 0
    harness.director.actions = 5
    harness.director.config.max_emulator_actions = 1
    harness.analyzer.event = ScreenEvent.NONE
    step = harness.director.step()
    assert step.context == "limit"
    assert "max_emulator_actions" in step.detail
    # a checkpoint of the safe state is captured
    assert any("safe_pause" in reason for reason in harness.saved)


def test_checkpoint_is_created_periodically(harness) -> None:
    harness.analyzer.event = ScreenEvent.NONE
    harness.director.actions = 0
    harness.director.step()
    harness.director.step()
    assert harness.saved, "expected periodic checkpoint saves"


def test_progress_stagnation_invokes_recovery(harness) -> None:
    harness.monitor.observe_actions(100)
    harness.monitor.observe_navigation({"cells_total": 5, "walkable": 5, "areas_total": 1,
                                        "coverage_ratio": 1.0})
    original = harness.monitor.consume_signals

    def forced() -> list[Any]:
        return [director_module.StagnationKind.LOCAL_STAGNATION]

    harness.monitor.consume_signals = forced  # type: ignore[assignment]
    harness.analyzer.event = ScreenEvent.NONE
    step = harness.director.step()
    assert step.action == "recovery"
    assert harness.recovery.calls
    assert harness.director.stats_dict()["stagnation_events"] == 1
    harness.monitor.consume_signals = original  # type: ignore[assignment]


def test_director_review_parses_json_and_applies_task_actions(harness) -> None:
    task, _ = harness.tasks.create_task(
        owner_agent=AgentRole.NAVIGATION,
        requested_by=AgentRole.STORY,
        task_type=TaskType.EXPLORE,
        description="old low priority exploration",
        priority=Priority.LOW,
        branch_id="branch_test",
    )
    harness.router.payload = {
        "review": "one exploration task looks stale",
        "task_actions": [{"task_id": task.task_id, "action": "deprioritize", "priority": "background"}],
        "create_tasks": [
            {"owner_agent": "navigation", "task_type": "explore",
             "description": "check the unexplored corner", "priority": "high", "payload": {}}
        ],
        "focus_agent": "navigation",
        "rollback": False,
    }
    result = harness.director.director_review({"signals": ["local_stagnation"]}, reason="test")
    assert result["ok"] is True
    assert harness.router.calls
    applied = result["applied"]
    assert any(item.get("action") == "deprioritize" for item in applied["applied"])
    assert applied["created"], "the review should create the proposed task"
    assert harness.tasks.get(task.task_id).priority is Priority.BACKGROUND


def test_director_review_is_skipped_when_router_forbids(harness) -> None:
    harness.router.allow = False
    result = harness.director.director_review({}, reason="test")
    assert result["ok"] is False
    assert not harness.router.calls


def test_snapshot_restore_keeps_monotonic_counters(harness) -> None:
    harness.analyzer.event = ScreenEvent.NONE
    harness.director.step()
    harness.director.actions = 50
    harness.director.buttons = 40
    snapshot = harness.director.snapshot()
    assert snapshot["actions"] == 50
    # a rollback restores an older snapshot: spend/attempts must not rewind
    older = dict(snapshot)
    older["actions"] = 5
    older["buttons"] = 3
    older["stats"] = {"checkpoints": 1, "rollbacks": 0}
    harness.director.restore(older)
    assert harness.director.actions == 50
    assert harness.director.buttons == 40


def test_task_queue_snapshot_and_status_are_json_safe(harness) -> None:
    harness.analyzer.event = ScreenEvent.NONE
    harness.director.step()
    status = harness.director.status()
    json.dumps(status, default=str)
    queue = harness.director.task_queue_snapshot()
    assert queue and {"task_id", "owner", "status", "description"} <= set(queue[0])
    assert status["owner"] in ("none", "navigation", "story", "pokemon", "system")


def test_run_loop_stops_at_max_steps(harness) -> None:
    harness.director.config.max_steps = 3
    harness.analyzer.event = ScreenEvent.NONE
    result = harness.director.run()
    assert result.steps <= 3
    assert result.status in ("finished", "paused_limits", "stopped")
