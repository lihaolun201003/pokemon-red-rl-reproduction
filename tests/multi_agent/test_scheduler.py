"""Unit tests for the deterministic task scheduler (no ROM/network/emulator).

Covers the mandated ordering (recovery > continue working > story > navigation >
low value), priority/aging/starvation handling, owner rotation, the "just
failed" penalty (cleared when the attempt/priority signature changes), owner
availability, explainable ranking and determinism/termination.
"""

from __future__ import annotations

import pytest

from llm_agent.multi_agent.contracts import AgentRole, Priority, Task, TaskStatus, TaskType
from llm_agent.multi_agent.scheduler import (
    DEFAULT_TYPE_WEIGHTS,
    OWNER_ROTATION_PENALTY,
    SchedulerConfig,
    TaskScheduler,
)
from llm_agent.multi_agent.task_manager import TaskManager


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class FakeClock:
    """Deterministic, manually advanced wall clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def _make_task(
    clock: FakeClock | None = None,
    *,
    owner: str = "navigation",
    task_type: str = "navigate",
    description: str = "task",
    priority: str | int = "normal",
    created_at: float | None = None,
    status: TaskStatus = TaskStatus.READY,
    progress: float = 0.0,
) -> Task:
    """Build one standalone Task (contracts-level, no TaskManager involved)."""
    stamp = float(clock.now if clock is not None else 1000.0) if created_at is None else created_at
    task = Task.create(
        owner_agent=owner,
        requested_by="director",
        task_type=task_type,
        description=description,
        priority=priority,
        created_at=stamp,
    )
    task.status = status
    task.progress = float(progress)
    return task


def _ids(items: list[dict]) -> list[str]:
    return [str(item["task_id"]) for item in items]


# --------------------------------------------------------------------------- #
# Configuration / defaults
# --------------------------------------------------------------------------- #

def test_default_weights_respect_mandated_tiers() -> None:
    weights = DEFAULT_TYPE_WEIGHTS
    assert weights[TaskType.RECOVER.value] > weights[TaskType.TALK.value]
    assert weights[TaskType.TALK.value] > weights[TaskType.INTERACT.value]
    assert weights[TaskType.INTERACT.value] > weights[TaskType.INVESTIGATE.value]
    assert weights[TaskType.INVESTIGATE.value] > weights[TaskType.NAVIGATE.value]
    assert weights[TaskType.NAVIGATE.value] > weights[TaskType.EXPLORE.value]
    assert weights[TaskType.EXPLORE.value] > weights[TaskType.CUSTOM.value]

    config = SchedulerConfig()
    assert config.owner_rotation_limit == 3
    assert config.aging_boost_per_minute > 0
    assert config.attempt_penalty > 0
    assert config.starvation_after_s > 0


def test_default_scheduler_constructs_without_arguments() -> None:
    scheduler = TaskScheduler()
    assert scheduler.stats()["selected"] == 0
    assert scheduler.next_task([]) is None


# --------------------------------------------------------------------------- #
# Mandated ordering
# --------------------------------------------------------------------------- #

def test_next_task_follows_mandated_order() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    tasks = [
        _make_task(clock, owner="navigation", task_type="explore", description="explore"),
        _make_task(clock, owner="story", task_type="talk", description="talk"),
        _make_task(clock, owner="navigation", task_type="recover", description="recover"),
        _make_task(clock, owner="navigation", task_type="navigate", description="navigate"),
        _make_task(clock, owner="story", task_type="interact", description="interact"),
        _make_task(clock, owner="story", task_type="investigate", description="investigate"),
    ]
    ranked = scheduler.rank(tasks)
    order = [item["task_type"] for item in ranked]
    assert order == ["recover", "talk", "interact", "investigate", "navigate", "explore"]
    assert scheduler.next_task(tasks).task_type is TaskType.RECOVER


def test_urgent_recovery_beats_everything() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    recovery = _make_task(
        clock, task_type="recover", priority="background", description="recover quietly"
    )
    critical_story = _make_task(
        clock, task_type="talk", priority="critical", description="critical talk"
    )
    assert scheduler.next_task([critical_story, recovery]).task_id == recovery.task_id


def test_critical_priority_lifts_low_value_work() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    normal_story = _make_task(clock, task_type="talk", description="normal talk")
    critical_nav = _make_task(
        clock, task_type="navigate", priority="critical", description="critical navigate"
    )
    assert scheduler.next_task([normal_story, critical_nav]).task_id == critical_nav.task_id


def test_high_priority_can_beat_a_higher_tier() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    normal_story = _make_task(clock, task_type="talk", description="normal talk")
    critical_low_tier = _make_task(
        clock, task_type="investigate", priority="critical", description="critical investigate"
    )
    # ``investigate`` is a lower tier than ``talk``, but critical priority lifts it.
    assert scheduler.next_task([normal_story, critical_low_tier]).task_id == critical_low_tier.task_id


def test_priority_decides_within_one_type() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    low = _make_task(clock, task_type="talk", priority="low", description="low talk")
    high = _make_task(clock, task_type="talk", priority="high", description="high talk")
    assert scheduler.next_task([low, high]).task_id == high.task_id


def test_continue_running_work_gets_boost() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    running = _make_task(
        clock, task_type="navigate", description="already running", status=TaskStatus.RUNNING
    )
    fresh = _make_task(clock, task_type="navigate", description="fresh ready")
    ranked = scheduler.rank([fresh, running])
    assert _ids(ranked)[0] == running.task_id
    assert ranked[0]["parts"]["status"] > 0
    assert ranked[1]["parts"]["status"] == 0.0


def test_paused_work_still_gets_a_boost() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    paused = _make_task(
        clock, task_type="navigate", description="paused task", status=TaskStatus.PAUSED
    )
    fresh = _make_task(clock, task_type="navigate", description="fresh again")
    assert scheduler.next_task([fresh, paused]).task_id == paused.task_id


def test_verifying_tasks_are_deprioritised() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    verifying = _make_task(
        clock, task_type="talk", description="awaiting verification", status=TaskStatus.VERIFYING
    )
    ready = _make_task(clock, task_type="talk", description="ready work")
    assert scheduler.next_task([verifying, ready]).task_id == ready.task_id


# --------------------------------------------------------------------------- #
# Aging / starvation
# --------------------------------------------------------------------------- #

def test_old_task_ages_into_selection() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    old_nav = _make_task(clock, task_type="navigate", description="old navigation")
    clock.advance(3600)  # one hour of waiting
    new_story = _make_task(clock, task_type="talk", description="new story work")

    ranked = scheduler.rank([old_nav, new_story])
    assert _ids(ranked)[0] == old_nav.task_id
    assert ranked[0]["parts"]["aging"] > 0
    assert ranked[0]["parts"]["starvation"] > 0


def test_age_boost_is_capped() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    ancient = _make_task(clock, task_type="custom", description="ancient task", created_at=0.0)
    ranked = scheduler.rank([ancient])
    assert ranked[0]["parts"]["aging"] <= 80.0
    assert ranked[0]["parts"]["starvation"] > 0


def test_starvation_after_threshold() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    waiting = _make_task(clock, task_type="explore", description="waiting task")

    fresh_reference = _make_task(clock, task_type="talk", description="fresh talk")
    assert scheduler.next_task([waiting, fresh_reference]).task_id == fresh_reference.task_id

    # The waiting task crosses the starvation threshold while a *newer* task
    # keeps being created at the current time, so only age distinguishes them.
    clock.advance(scheduler.config.starvation_after_s + 1.0)
    newer_reference = _make_task(clock, task_type="talk", description="newer talk")
    assert scheduler.next_task([waiting, newer_reference]).task_id == waiting.task_id


# --------------------------------------------------------------------------- #
# Owner rotation
# --------------------------------------------------------------------------- #

def test_owner_rotation_avoids_single_owner_forever() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    tasks: list[Task] = []
    for index in range(3):
        tasks.append(
            _make_task(clock, owner="navigation", description=f"nav {index}", task_type="navigate")
        )
        tasks.append(
            _make_task(clock, owner="story", description=f"story {index}", task_type="navigate")
        )

    selected_owners: list[str] = []
    for _ in range(9):
        task = scheduler.next_task(tasks)
        assert task is not None
        selected_owners.append(task.owner_agent.value)
        scheduler.on_selected(task)

    longest = current = 1
    for previous, current_owner in zip(selected_owners, selected_owners[1:]):
        current = current + 1 if current_owner == previous else 1
        longest = max(longest, current)
    assert longest <= scheduler.config.owner_rotation_limit
    assert len(set(selected_owners)) == 2  # both owners used


def test_owner_rotation_penalty_is_visible_in_parts() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    nav = _make_task(clock, owner="navigation", description="nav work")
    story = _make_task(clock, owner="story", description="story work")
    for _ in range(3):
        scheduler.on_selected(nav)

    ranked = scheduler.rank([nav, story])
    nav_item = next(item for item in ranked if item["task_id"] == nav.task_id)
    assert nav_item["parts"]["owner_rotation"] == -OWNER_ROTATION_PENALTY
    assert _ids(ranked)[0] == story.task_id


# --------------------------------------------------------------------------- #
# Failure memory
# --------------------------------------------------------------------------- #

def test_failed_task_is_deprioritised_until_state_changes() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    manager = TaskManager(clock=clock)
    story_task, _ = manager.create_task(
        owner_agent="story",
        requested_by="director",
        task_type="talk",
        description="talk that failed",
    )
    nav_task, _ = manager.create_task(
        owner_agent="navigation",
        requested_by="director",
        task_type="navigate",
        description="other navigation",
    )
    manager.refresh_ready()
    tasks = manager.tasks()

    scheduler.on_selected(story_task)
    scheduler.on_finished(story_task, ok=False)

    # Normally the "talk" tier wins, but the failure penalty pushes it below.
    assert scheduler.next_task(tasks).task_id == nav_task.task_id
    ranked = scheduler.rank(tasks)
    story_item = next(item for item in ranked if item["task_id"] == story_task.task_id)
    assert story_item["parts"]["failure_penalty"] < 0

    # Recording a new attempt changes the failure signature: retry allowed.
    manager.record_attempt(story_task.task_id)
    retried_tasks = manager.tasks()
    assert scheduler.next_task(retried_tasks).task_id == story_task.task_id

    # Success clears the failure memory.
    scheduler.on_finished(story_task, ok=True)
    assert scheduler.stats()["tracked_failures"] == 0


# --------------------------------------------------------------------------- #
# Availability / eligibility
# --------------------------------------------------------------------------- #

def test_owner_availability_filters_candidates() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    nav = _make_task(clock, owner="navigation", description="navigation work")
    story = _make_task(clock, owner="story", description="story work")

    picked = scheduler.next_task([nav, story], owner_available={"navigation": False})
    assert picked.task_id == story.task_id

    picked = scheduler.next_task([nav, story], owner_available={AgentRole.STORY: False})
    assert picked.task_id == nav.task_id

    assert scheduler.next_task(
        [nav, story],
        owner_available={AgentRole.NAVIGATION: False, AgentRole.STORY: False},
    ) is None

    # Unknown/missing owners are treated as available.
    assert scheduler.next_task([nav, story], owner_available={}) is not None


def test_terminal_and_blocked_tasks_are_never_selected() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    completed = _make_task(clock, status=TaskStatus.COMPLETED, description="completed")
    failed = _make_task(clock, status=TaskStatus.FAILED, description="failed")
    cancelled = _make_task(clock, status=TaskStatus.CANCELLED, description="cancelled")
    blocked = _make_task(clock, status=TaskStatus.BLOCKED, description="blocked")
    ready = _make_task(clock, description="ready")

    assert scheduler.next_task([completed, failed, cancelled, blocked]) is None
    assert scheduler.next_task([completed, blocked, ready]).task_id == ready.task_id


def test_pending_tasks_are_scored_when_handed_over() -> None:
    # The scheduler is a scoring function over whatever the director passes; a
    # pending task (dependencies just satisfied) is still rankable.
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    pending = _make_task(clock, status=TaskStatus.PENDING, description="pending task")
    assert scheduler.next_task([pending]).task_id == pending.task_id


# --------------------------------------------------------------------------- #
# Ranking contract
# --------------------------------------------------------------------------- #

def test_rank_items_have_explainable_parts_and_order() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    tasks = [
        _make_task(clock, task_type="talk", description="talk", priority="high"),
        _make_task(clock, task_type="explore", description="explore", priority="low"),
        _make_task(clock, task_type="recover", description="recover", priority="normal"),
    ]
    ranked = scheduler.rank(tasks)
    assert [_ids(ranked)[0]] == [ranked[0]["task_id"]]
    for item in ranked:
        assert set(item) == {"task_id", "score", "parts", "owner", "task_type", "priority"}
        assert sum(item["parts"].values()) == pytest.approx(item["score"], abs=1e-6)
        for part in (
            "type",
            "priority",
            "status",
            "recovery",
            "aging",
            "starvation",
            "progress",
            "attempt_penalty",
            "interruption_penalty",
            "failure_penalty",
            "recent_selection",
            "owner_rotation",
        ):
            assert part in item["parts"]
    scores = [item["score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_is_deterministic_with_task_id_tie_break() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    tasks = [
        _make_task(clock, description="a"),
        _make_task(clock, description="b"),
        _make_task(clock, description="c"),
    ]
    first = _ids(scheduler.rank(tasks))
    second = _ids(scheduler.rank(list(reversed(tasks))))
    assert first == second
    assert first == sorted(first)


def test_deterministic_tie_break_prefers_older_task() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    old = _make_task(clock, description="old", created_at=500.0)
    new = _make_task(clock, description="new", created_at=900.0)
    # Aging makes the older task score higher as well.
    assert scheduler.next_task([new, old]).task_id == old.task_id


def test_attempt_and_interruption_penalties() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    tried = _make_task(clock, description="tried")
    tried.attempts = 2
    tried.interruptions = 3
    fresh = _make_task(clock, description="fresh")
    ranked = scheduler.rank([tried, fresh])
    tried_item = next(item for item in ranked if item["task_id"] == tried.task_id)
    assert tried_item["parts"]["attempt_penalty"] == pytest.approx(-2 * 8.0)
    assert tried_item["parts"]["interruption_penalty"] == pytest.approx(-3 * 1.0)
    assert _ids(ranked)[0] == fresh.task_id


def test_progress_bonus() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    nearly_done = _make_task(clock, description="nearly done", progress=0.9)
    untouched = _make_task(clock, description="untouched")
    assert scheduler.next_task([untouched, nearly_done]).task_id == nearly_done.task_id


def test_weight_override_is_respected() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(
        SchedulerConfig(type_weights={"explore": 1000.0}), clock=clock
    )
    explore = _make_task(clock, task_type="explore", description="explore")
    recover = _make_task(clock, task_type="recover", description="recover")
    assert scheduler.next_task([recover, explore]).task_id == explore.task_id


def test_config_accepts_enum_weight_keys() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(
        SchedulerConfig(type_weights={TaskType.CUSTOM: 500.0}), clock=clock
    )
    custom = _make_task(clock, task_type="custom", description="custom")
    talk = _make_task(clock, task_type="talk", description="talk")
    assert scheduler.next_task([talk, custom]).task_id == custom.task_id
    assert scheduler.stats()["type_weights"]["custom"] == 500.0


# --------------------------------------------------------------------------- #
# Bookkeeping / termination
# --------------------------------------------------------------------------- #

def test_stats_and_reset() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    task = _make_task(clock, description="tracked")
    scheduler.next_task([task])
    scheduler.on_selected(task)
    scheduler.on_selected(task)
    scheduler.on_finished(task, ok=True)

    stats = scheduler.stats()
    assert stats["rank_calls"] == 1
    assert stats["select_calls"] == 1
    assert stats["selected"] == 2
    assert stats["finished_ok"] == 1
    assert stats["current_owner_streak"] == 2
    assert stats["owner_history"]

    scheduler.reset()
    stats = scheduler.stats()
    assert stats["selected"] == 0
    assert stats["owner_history"] == []
    assert stats["current_owner_streak"] == 0
    assert scheduler.next_task([task]).task_id == task.task_id


def test_large_ranking_terminates_and_deduplicates_ids() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    owners = ("navigation", "story", "pokemon")
    tasks = [
        _make_task(
            clock,
            owner=owners[index % len(owners)],
            description=f"task {index}",
            task_type="explore",
        )
        for index in range(150)
    ]
    ranked = scheduler.rank(tasks + tasks)  # duplicated input must not duplicate items
    assert len(ranked) == 150
    assert len({item["task_id"] for item in ranked}) == 150
    assert scheduler.next_task(tasks) is not None


def test_scheduler_does_not_repeat_one_task_forever() -> None:
    # Even when every task belongs to one owner, the most recently selected task
    # is de-prioritised so the director never loops on a single task.
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    tasks = [
        _make_task(clock, owner="navigation", task_type="explore", description=f"same owner {index}")
        for index in range(4)
    ]
    selected: list[str] = []
    for _ in range(8):
        task = scheduler.next_task(tasks)
        assert task is not None
        selected.append(task.task_id)
        scheduler.on_selected(task)
    assert len(set(selected)) > 1
    assert max(selected.count(task_id) for task_id in set(selected)) < len(selected)


def test_scheduler_accepts_only_task_records() -> None:
    clock = FakeClock()
    scheduler = TaskScheduler(clock=clock)
    assert scheduler.rank([None, "not a task", 42]) == []  # type: ignore[list-item]
    assert scheduler.next_task([None]) is None  # type: ignore[list-item]
