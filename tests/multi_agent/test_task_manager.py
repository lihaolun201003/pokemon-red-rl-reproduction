"""Unit tests for the shared task graph (no ROM, no network, no emulator).

Covers the acceptance list: creation/de-duplication, parent & per-agent limits,
dependency chain + cycle detection, the frozen lifecycle with explicit
invalid-transition errors, honest completion, attempt limits, rollback branch
invalidation, snapshot/restore and atomic persistence (including corrupt files).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_agent.multi_agent.contracts import (
    AgentRole,
    CycleDetectedError,
    Priority,
    Task,
    TaskError,
    TaskStatus,
    TaskType,
)
from llm_agent.multi_agent.task_manager import (
    TASK_MANAGER_VERSION,
    TASKS_FILENAME,
    TaskManager,
    TaskManagerConfig,
)


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


def _create(
    manager: TaskManager,
    *,
    owner: str = "navigation",
    task_type: str = "navigate",
    description: str = "go somewhere",
    **kwargs,
) -> tuple[Task, bool]:
    return manager.create_task(
        owner_agent=owner,
        requested_by="director",
        task_type=task_type,
        description=description,
        **kwargs,
    )


def _ready(manager: TaskManager, **kwargs) -> Task:
    task, _created = _create(manager, **kwargs)
    manager.refresh_ready()
    return manager.require(task.task_id)


def _running(manager: TaskManager, **kwargs) -> Task:
    task = _ready(manager, **kwargs)
    return manager.start(task.task_id)


def _completed(manager: TaskManager, **kwargs) -> Task:
    task = _running(manager, **kwargs)
    return manager.complete(task.task_id, result="done", evidence={"ok": True})


# --------------------------------------------------------------------------- #
# Creation / de-duplication
# --------------------------------------------------------------------------- #

def test_create_task_defaults_and_branch() -> None:
    clock = FakeClock()
    manager = TaskManager(clock=clock, branch_id="branch_main")
    task, created = _create(
        manager,
        description="map the nearby area",
        task_type="explore",
        payload={"area": "area_1"},
        priority="high",
    )

    assert created is True
    assert task.status is TaskStatus.PENDING
    assert task.owner_agent is AgentRole.NAVIGATION
    assert task.requested_by is AgentRole.DIRECTOR
    assert task.task_type is TaskType.EXPLORE
    assert task.branch_id == "branch_main"
    assert task.created_at == 1000.0
    assert task.updated_at == 1000.0
    assert task.payload == {"area": "area_1"}
    assert task.priority is Priority.HIGH
    assert task.dedupe_key

    # get() returns defensive copies: mutating one must not corrupt the manager.
    fetched = manager.get(task.task_id)
    assert fetched is not None
    fetched.status = TaskStatus.CANCELLED
    assert manager.require(task.task_id).status is TaskStatus.PENDING
    assert manager.branch_id == "branch_main"


def test_dedupe_suppresses_duplicate_live_goal() -> None:
    manager = TaskManager()
    first, created = _create(manager, description="talk to the person")
    second, created_again = _create(manager, description="talk to the person")

    assert created is True
    assert created_again is False
    assert second.task_id == first.task_id
    assert len(manager.tasks()) == 1

    # A different owner/type is a different logical goal.
    other, created_other = _create(manager, owner="story", task_type="talk", description="talk to the person")
    assert created_other is True
    assert other.task_id != first.task_id

    # After the original goal completes, the same goal may be queued again.
    manager.refresh_ready()
    manager.start(first.task_id)
    manager.complete(first.task_id, result="spoke to them")
    again, created_again_after_done = _create(manager, description="talk to the person")
    assert created_again_after_done is True
    assert again.task_id != first.task_id


def test_dedupe_released_after_cancel() -> None:
    manager = TaskManager()
    task, _ = _create(manager, description="walk east")
    manager.cancel(task.task_id, reason="no longer needed")
    again, created = _create(manager, description="walk east")
    assert created is True
    assert again.task_id != task.task_id


def test_explicit_task_id_collision_raises() -> None:
    manager = TaskManager()
    task, _ = _create(manager, description="fixed id task", task_id="task_fixed0001")
    assert task.task_id == "task_fixed0001"
    with pytest.raises(TaskError):
        _create(manager, description="another task", task_id="task_fixed0001")


def test_duplicate_and_invalid_creation_inputs() -> None:
    manager = TaskManager()
    with pytest.raises(TaskError):
        manager.create_task(
            owner_agent="director",  # a task cannot be owned by the director
            requested_by="director",
            task_type="navigate",
            description="bad owner",
        )
    with pytest.raises(TaskError):
        manager.create_task(
            owner_agent="not-an-agent",
            requested_by="director",
            task_type="navigate",
            description="bad role",
        )
    with pytest.raises(TaskError):
        _create(manager, description="   ")
    with pytest.raises(TaskError):
        _create(manager, description="bad priority", priority="immortal")


def test_parent_and_children_limit() -> None:
    manager = TaskManager(TaskManagerConfig(max_children_per_task=2))
    parent, _ = _create(manager, description="parent task")
    first, _ = _create(manager, description="child one", parent_task_id=parent.task_id)
    second, _ = _create(manager, description="child two", parent_task_id=parent.task_id)

    children = manager.children(parent.task_id)
    assert {task.task_id for task in children} == {first.task_id, second.task_id}
    assert len(children) == 2
    with pytest.raises(TaskError):
        _create(manager, description="child three", parent_task_id=parent.task_id)
    with pytest.raises(TaskError):
        _create(manager, description="orphan", parent_task_id="task_missing")


def test_per_agent_and_queue_limits() -> None:
    clock = FakeClock()
    manager = TaskManager(
        TaskManagerConfig(max_active_tasks=3, max_tasks_per_agent=2), clock=clock
    )
    _create(manager, owner="navigation", description="nav one")
    _create(manager, owner="navigation", description="nav two")
    with pytest.raises(TaskError):
        _create(manager, owner="navigation", description="nav three")
    _create(manager, owner="story", description="story one")
    with pytest.raises(TaskError):
        _create(manager, owner="story", description="story two")

    # Completing a navigation task frees a navigation slot.
    manager.refresh_ready()
    nav_one = next(
        task for task in manager.ready() if task.owner_agent is AgentRole.NAVIGATION
    )
    manager.start(nav_one.task_id)
    manager.complete(nav_one.task_id, result="finished")
    _create(manager, owner="navigation", description="nav three")


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #

def test_dependency_chain_and_refresh_ready() -> None:
    manager = TaskManager()
    first, _ = _create(manager, description="first")
    second, _ = _create(manager, description="second", dependencies=[first.task_id])
    third, _ = _create(manager, description="third", dependencies=[second.task_id])

    assert second.status is TaskStatus.PENDING
    changed = manager.refresh_ready()
    assert [task.task_id for task in changed] == [first.task_id]
    assert manager.require(second.task_id).status is TaskStatus.PENDING

    manager.start(first.task_id)
    manager.complete(first.task_id, result="first done")
    changed = manager.refresh_ready()
    assert [task.task_id for task in changed] == [second.task_id]

    manager.start(second.task_id)
    manager.complete(second.task_id, result="second done")
    changed = manager.refresh_ready()
    assert [task.task_id for task in changed] == [third.task_id]
    assert manager.require(third.task_id).status is TaskStatus.READY


def test_unknown_dependency_rejected() -> None:
    manager = TaskManager()
    with pytest.raises(TaskError):
        _create(manager, description="bad dep", dependencies=["task_missing"])


def test_dependency_cycle_detected() -> None:
    manager = TaskManager()
    first, _ = _create(manager, description="cycle first")
    second, _ = _create(manager, description="cycle second")

    manager.add_dependency(second.task_id, first.task_id)  # second <- first: fine
    with pytest.raises(CycleDetectedError):
        manager.add_dependency(first.task_id, second.task_id)
    with pytest.raises(CycleDetectedError):
        manager.add_dependency(first.task_id, first.task_id)
    assert manager.detect_cycles() == []

    # The failed edge must not have been recorded.
    assert manager.require(first.task_id).dependencies == []


def test_remove_dependency_and_missing_edge() -> None:
    manager = TaskManager()
    first, _ = _create(manager, description="dep target")
    second, _ = _create(manager, description="dep owner", dependencies=[first.task_id])
    manager.remove_dependency(second.task_id, first.task_id)
    assert manager.require(second.task_id).dependencies == []
    with pytest.raises(TaskError):
        manager.remove_dependency(second.task_id, first.task_id)
    with pytest.raises(TaskError):
        manager.add_dependency(second.task_id, "task_missing")


def test_detect_cycles_finds_restored_cycle() -> None:
    plain = TaskManager()
    first, _ = _create(plain, description="restored a")
    second, _ = _create(plain, description="restored b")

    first_dict = dict(first.to_dict(), dependencies=[second.task_id])
    second_dict = dict(second.to_dict(), dependencies=[first.task_id])
    manager = TaskManager()
    manager.restore({"tasks": [first_dict, second_dict]})

    cycles = manager.detect_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {first.task_id, second.task_id}

    # A corrupted cycle never becomes "ready" just because the graph exists.
    assert manager.refresh_ready() == []


# --------------------------------------------------------------------------- #
# Lifecycle / invalid transitions
# --------------------------------------------------------------------------- #

def test_invalid_transitions_raise() -> None:
    manager = TaskManager()
    pending, _ = _create(manager, description="lifecycle task")

    with pytest.raises(TaskError):
        manager.start(pending.task_id)          # PENDING -> RUNNING is invalid
    with pytest.raises(TaskError):
        manager.complete(pending.task_id, result="never ran")
    with pytest.raises(TaskError):
        manager.pause(pending.task_id)
    with pytest.raises(TaskError):
        manager.resume(pending.task_id)
    with pytest.raises(TaskError):
        manager.unblock(pending.task_id)
    with pytest.raises(TaskError):
        manager.verify(pending.task_id)

    manager.refresh_ready()
    manager.start(pending.task_id)
    with pytest.raises(TaskError):
        manager.start(pending.task_id)          # RUNNING -> RUNNING is invalid
    with pytest.raises(TaskError):
        manager.unblock(pending.task_id)

    manager.pause(pending.task_id, reason="interrupted")
    with pytest.raises(TaskError):
        manager.pause(pending.task_id)

    manager.cancel(pending.task_id, reason="enough")
    with pytest.raises(TaskError):
        manager.reopen(pending.task_id)         # CANCELLED is final
    with pytest.raises(TaskError):
        manager.fail(pending.task_id, reason="late")


def test_honest_completion_requires_evidence() -> None:
    manager = TaskManager()
    task = _running(manager, description="honest task")

    with pytest.raises(TaskError):
        manager.complete(task.task_id)
    with pytest.raises(TaskError):
        manager.complete(task.task_id, result="  ", evidence={})

    done = manager.complete(
        task.task_id,
        result=" reached the marker ",
        evidence={"screenshot": "frame_001.png", "steps": 12},
    )
    assert done.status is TaskStatus.COMPLETED
    assert done.progress == 1.0
    record = done.evidence[-1]
    assert record["kind"] == "completion"
    assert record["result"] == "reached the marker"
    assert record["screenshot"] == "frame_001.png"
    assert record["steps"] == 12

    # The director may verify the claim separately; evidence is appended.
    verified = manager.verify(done.task_id, evidence={"director": "confirmed"})
    assert verified.status is TaskStatus.COMPLETED
    assert verified.evidence[-1]["kind"] == "verification"
    assert verified.evidence[-1]["verified"] is True
    assert verified.evidence[-1]["director"] == "confirmed"


def test_verify_running_requires_evidence() -> None:
    manager = TaskManager()
    task = _running(manager, description="verify me")
    with pytest.raises(TaskError):
        manager.verify(task.task_id)
    verified = manager.verify(task.task_id, evidence={"inspection": "frame"})
    assert verified.status is TaskStatus.COMPLETED


def test_verify_from_verifying_state() -> None:
    plain = TaskManager()
    task = _running(plain, description="claimed completion")
    verifying = dict(task.to_dict(), status=TaskStatus.VERIFYING.value)

    manager = TaskManager()
    manager.restore({"tasks": [verifying]})
    verified = manager.verify(task.task_id, evidence={"director": "checked"})
    assert verified.status is TaskStatus.COMPLETED


def test_pause_resume_block_unblock() -> None:
    manager = TaskManager()
    task = _running(manager, description="interruptible task")

    paused = manager.pause(task.task_id, reason="dialog started")
    assert paused.status is TaskStatus.PAUSED
    assert paused.interruptions == 1
    assert paused.evidence[-1]["kind"] == "paused"
    assert paused.evidence[-1]["from"] == TaskStatus.RUNNING.value

    resumed = manager.resume(task.task_id)
    assert resumed.status is TaskStatus.READY
    manager.start(task.task_id)

    blocked = manager.block(task.task_id, reason="path is blocked")
    assert blocked.status is TaskStatus.BLOCKED
    assert blocked.reason == "path is blocked"
    unblocked = manager.unblock(task.task_id)
    assert unblocked.status is TaskStatus.READY
    assert unblocked.reason == ""

    with pytest.raises(TaskError):
        manager.block(task.task_id, reason="   ")


def test_blocked_task_waits_for_dependencies() -> None:
    manager = TaskManager()
    dependency, _ = _create(manager, description="blocking dependency")
    dependent, _ = _create(
        manager, description="dependent task", dependencies=[dependency.task_id]
    )
    manager.block(dependent.task_id, reason="manual pause")
    released = manager.unblock(dependent.task_id)
    assert released.status is TaskStatus.PENDING


def test_attempts_limit() -> None:
    manager = TaskManager(TaskManagerConfig(max_attempts_per_task=2))
    task = _running(manager, description="attempted task")

    assert manager.record_attempt(task.task_id) == 1
    assert manager.record_attempt(task.task_id) == 2
    with pytest.raises(TaskError):
        manager.record_attempt(task.task_id)
    assert manager.require(task.task_id).attempts == 2

    manager.complete(task.task_id, result="succeeded on the second attempt")
    with pytest.raises(TaskError):
        manager.record_attempt(task.task_id)   # terminal tasks can not be attempted


# --------------------------------------------------------------------------- #
# Metadata mutations
# --------------------------------------------------------------------------- #

def test_priority_progress_and_reassign() -> None:
    manager = TaskManager()
    task = _running(manager, description="tunable task")

    assert manager.set_priority(task.task_id, "critical").priority is Priority.CRITICAL
    with pytest.raises(TaskError):
        manager.set_priority(task.task_id, "not-a-priority")

    progressed = manager.update_progress(task.task_id, 0.4, note="half way")
    assert progressed.progress == 0.4
    assert progressed.evidence[-1]["kind"] == "progress"
    assert progressed.evidence[-1]["note"] == "half way"
    # Progress is monotonic: a lower value never rewinds recorded progress.
    assert manager.update_progress(task.task_id, 0.1).progress == 0.4
    assert manager.update_progress(task.task_id, 2.0).progress == 1.0

    moved = manager.reassign(task.task_id, "story")
    assert moved.owner_agent is AgentRole.STORY
    duplicate, created = manager.create_task(
        owner_agent="story",
        requested_by="director",
        task_type="navigate",
        description="tunable task",
    )
    assert created is False
    assert duplicate.task_id == task.task_id

    manager.complete(task.task_id, result="finished")
    with pytest.raises(TaskError):
        manager.update_progress(task.task_id, 0.5)


# --------------------------------------------------------------------------- #
# Snapshot / restore / branch scoping
# --------------------------------------------------------------------------- #

def test_snapshot_restore_roundtrip() -> None:
    clock = FakeClock()
    manager = TaskManager(clock=clock, branch_id="branch_a")
    ready, _ = _create(manager, description="ready task")
    manager.refresh_ready()
    running, _ = _create(manager, description="running task")
    manager.refresh_ready()
    manager.start(running.task_id)
    done, _ = _create(manager, description="completed task")
    manager.refresh_ready()
    manager.start(done.task_id)
    manager.complete(done.task_id, result="done", evidence={"frame": "f.png"})
    library, _ = _create(manager, description="archived task")
    manager.refresh_ready()
    manager.start(library.task_id)
    manager.complete(library.task_id, result="library done")
    manager.archive_terminal()

    snapshot = manager.snapshot()
    assert snapshot["schema_version"] == TASK_MANAGER_VERSION

    restored = TaskManager(clock=clock)
    restored.restore(snapshot)

    original = {task.task_id: task for task in manager.tasks()}
    assert {task.task_id: task for task in restored.tasks()} == original
    for task_id, expected in original.items():
        actual = restored.require(task_id)
        assert actual.status is expected.status
        assert actual.evidence == expected.evidence
        assert actual.dependencies == expected.dependencies
        assert actual.branch_id == expected.branch_id
    assert restored.stats()["archive"] == manager.stats()["archive"]
    assert restored.branch_id == "branch_a"

    # Dedupe index was rebuilt: the same live goal is still suppressed.
    duplicate, created = _create(restored, description="ready task")
    assert created is False
    assert duplicate.task_id == ready.task_id

    # Dependency graph was rebuilt: a dependent task stays pending then reads.
    dependent, _ = _create(
        restored, description="depends on running", dependencies=[running.task_id]
    )
    assert restored.refresh_ready() == []
    restored.complete(running.task_id, result="now done")
    changed = restored.refresh_ready()
    assert [task.task_id for task in changed] == [dependent.task_id]


def test_restore_is_tolerant_to_bad_records() -> None:
    manager = TaskManager()
    good, _ = _create(manager, description="good record")
    snapshot = manager.snapshot()
    snapshot["tasks"].append({"totally": "invalid"})
    snapshot["tasks"].append({"task_id": "task_broken", "description": ""})

    restored = TaskManager()
    restored.restore(snapshot)
    assert restored.get(good.task_id) is not None
    assert len(restored.tasks()) == 1
    assert restored.load_errors


def test_restore_keeps_branch_scoping() -> None:
    source = TaskManager(branch_id="branch_old")
    task, _ = _create(source, description="old branch work")
    snapshot = source.snapshot()

    target = TaskManager(branch_id="branch_new")
    target.restore(snapshot)
    assert target.get(task.task_id) is not None      # auditable
    assert target.active() == []                     # but not active in the new branch
    target.set_branch("branch_old")
    assert [item.task_id for item in target.active()] == [task.task_id]


def test_invalidate_branch_cancels_live_tasks_only() -> None:
    clock = FakeClock()
    manager = TaskManager(clock=clock, branch_id="branch_old")
    pending, _ = _create(manager, description="old pending")
    running = _running(manager, description="old running")
    completed, _ = _create(manager, description="old completed")
    manager.refresh_ready()
    manager.start(completed.task_id)
    manager.complete(completed.task_id, result="done before rollback")
    other_branch, _ = _create(manager, description="new branch work", branch_id="branch_new")

    changed = manager.invalidate_branch("branch_old", reason="stuck")
    changed_ids = {task.task_id for task in changed}
    assert changed_ids == {pending.task_id, running.task_id}
    for task in changed:
        assert task.status is TaskStatus.CANCELLED
        assert task.reason.startswith("historical_branch:branch_old")

    assert manager.require(completed.task_id).status is TaskStatus.COMPLETED
    assert manager.require(other_branch.task_id).status is TaskStatus.PENDING
    assert manager.stats()["cancelled"] >= 2

    # The cancelled goals are no longer de-duplicated, so the new branch can retry.
    retried, created = _create(manager, description="old pending", branch_id="branch_new")
    assert created is True
    assert retried.task_id != pending.task_id

    with pytest.raises(TaskError):
        manager.invalidate_branch("")


def test_archive_terminal_keeps_recent() -> None:
    clock = FakeClock()
    manager = TaskManager(TaskManagerConfig(archive_limit=2), clock=clock)
    done_ids = []
    for index in range(3):
        task = _completed(manager, description=f"finished {index}")
        done_ids.append(task.task_id)
        clock.advance(10)
    live, _ = _create(manager, description="still open")

    archived = manager.archive_terminal(keep=1)
    assert archived == 2
    remaining = {task.task_id for task in manager.tasks()}
    assert remaining == {done_ids[-1], live.task_id}
    assert manager.stats()["archive"] == 2
    assert manager.archive_terminal(keep=10) == 0


# --------------------------------------------------------------------------- #
# Reads / stats
# --------------------------------------------------------------------------- #

def test_tasks_filters_and_stats() -> None:
    manager = TaskManager()
    nav, _ = _create(manager, owner="navigation", description="nav task", task_type="navigate")
    story, _ = _create(manager, owner="story", description="story task", task_type="talk")
    manager.refresh_ready()
    manager.start(nav.task_id)

    assert [task.task_id for task in manager.tasks(statuses=[TaskStatus.RUNNING])] == [
        nav.task_id
    ]
    assert [task.task_id for task in manager.tasks(owner="story")] == [story.task_id]
    assert sorted(task.task_id for task in manager.tasks(statuses=["ready", "running"])) == sorted(
        [story.task_id, nav.task_id]
    )
    assert {task.task_id for task in manager.ready()} == {story.task_id}
    assert manager.get("task_missing") is None
    with pytest.raises(TaskError):
        manager.require("task_missing")
    with pytest.raises(TaskError):
        manager.tasks(statuses=["flying"])

    stats = manager.stats()
    for key in (
        "total",
        "live",
        "ready",
        "running",
        "completed",
        "by_status",
        "by_owner",
        "by_type",
        "dedupe_keys",
        "attempts",
        "archive",
        "cycles",
    ):
        assert key in stats
    assert stats["total"] == 2
    assert stats["running"] == 1
    assert stats["by_status"]["running"] == 1
    assert stats["by_owner"]["navigation"] == 1
    assert stats["cycles"] == 0


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def test_atomic_persistence_roundtrip(tmp_path: Path) -> None:
    clock = FakeClock()
    path = tmp_path / "tasks.json"
    manager = TaskManager(path=path, clock=clock, branch_id="branch_p")
    task, _ = _create(manager, description="persisted task")
    manager.refresh_ready()
    manager.start(task.task_id)
    manager.complete(task.task_id, result="persisted result", evidence={"frame": 1})
    manager.save()

    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"][0]["task_id"] == task.task_id
    assert not path.with_name(path.name + ".tmp").exists()

    loaded = TaskManager(path=path, clock=clock)
    assert loaded.load() is True
    assert loaded.get(task.task_id).status is TaskStatus.COMPLETED
    assert loaded.get(task.task_id).evidence[-1]["result"] == "persisted result"

    # A directory path resolves to <dir>/tasks.json.
    directory_manager = TaskManager(path=tmp_path, clock=clock)
    assert directory_manager.path == tmp_path / TASKS_FILENAME


def test_no_path_save_and_load_are_noops() -> None:
    manager = TaskManager()
    manager.save()                     # no exception, no file
    assert manager.load() is False
    assert manager.path is None


def test_corrupt_file_is_backed_up(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text("{ this is not json", encoding="utf-8")
    manager = TaskManager(path=path)

    assert manager.tasks() == []
    assert manager.load_errors
    assert manager.recovered_from is not None
    assert Path(manager.recovered_from).is_file()
    assert not path.exists()
    manager.save()
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["tasks"] == []
