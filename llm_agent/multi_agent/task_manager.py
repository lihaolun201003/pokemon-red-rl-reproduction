"""Shared task graph for the Pokémon Red multi-agent director (owner: tasks-memory).

The :class:`TaskManager` is the **single source of truth** for what the agents
are supposed to do.  It owns the :class:`~llm_agent.multi_agent.contracts.Task`
records, enforces the frozen lifecycle in
:data:`~llm_agent.multi_agent.contracts.TASK_TRANSITIONS`, de-duplicates logical
goals, keeps the dependency graph acyclic and persists everything as one
atomically-replaced JSON document.

Design rules
------------
* **Honest completion.**  ``complete()`` never marks a task ``COMPLETED``
  without a result summary or an evidence mapping; the evidence is appended to
  ``task.evidence`` and left for the director to verify separately.
  ``verify()`` records the director's confirmation with its own evidence.
* **No duplicate work.**  A task's :attr:`~llm_agent.multi_agent.contracts.Task.dedupe_key`
  is derived from (owner, type, description, parent); while a non-terminal task
  with the same key exists, ``create_task`` returns it with ``created=False``.
* **No cycles.**  ``create_task`` / ``add_dependency`` validate the prospective
  edge with a depth-first search and raise
  :class:`~llm_agent.multi_agent.contracts.CycleDetectedError`.
* **Rollback safety.**  :meth:`TaskManager.invalidate_branch` cancels every
  unfinished task that was created on a rolled-back branch while keeping the
  terminal history of that branch auditable.
* **Bounded memory.**  Evidence per task, the dependency fan-out and the archive
  are all capped.  ``archive_terminal`` moves old terminal tasks out of the
  working set.
* **Deterministic & cheap.**  No emulator, no network, no API key: the module
  imports nothing beyond the frozen protocol and the standard library.

Persistence
-----------
``path`` (when non-empty) points at a ``tasks.json`` document that is written
atomically (``*.tmp`` + :func:`os.replace`).  ``load()`` is tolerant: bad records
are skipped and a corrupt file is moved aside and replaced by an empty queue, so
a damaged file can never crash a run.  The constructor auto-loads an existing
file (so a fresh manager cannot silently clobber it); call :meth:`save`
explicitly after mutations.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from llm_agent.multi_agent.contracts import (
    AgentRole,
    CycleDetectedError,
    MultiAgentError,
    Priority,
    ProtocolError,
    Task,
    TaskError,
    TaskStatus,
    TaskType,
    now_ts,
)

__all__ = [
    "TASK_MANAGER_FORMAT",
    "TASK_MANAGER_VERSION",
    "TASKS_FILENAME",
    "MAX_EVIDENCE_PER_TASK",
    "TaskManagerConfig",
    "TaskManager",
]

#: Identifier written into the persisted document.
TASK_MANAGER_FORMAT = "multi_agent.tasks"
#: Schema version of the persisted document (bumped on breaking changes).
TASK_MANAGER_VERSION = 1
#: Default file name used when ``path`` points at a directory.
TASKS_FILENAME = "tasks.json"
#: Hard cap on the evidence records kept per task (oldest are folded away).
MAX_EVIDENCE_PER_TASK = 32

#: Statuses that still represent unfinished (live) work.
LIVE_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.PENDING,
        TaskStatus.READY,
        TaskStatus.RUNNING,
        TaskStatus.PAUSED,
        TaskStatus.BLOCKED,
        TaskStatus.VERIFYING,
    }
)

#: Statuses from which the terminal-status archive may collect a task.
TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #

def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort int coercion (never raises; used for tolerant loads)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion (never raises; used for tolerant loads)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clean_text(value: Any) -> str:
    """Normalise a human-readable one-line string (collapse whitespace)."""
    return " ".join(str(value or "").split())


def _atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 ``text`` to ``path`` atomically (tmp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _backup_corrupt(path: Path, *, now: float) -> str | None:
    """Move a corrupt persistence file to ``<stem>.corrupt-<stamp><suffix>``."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    stamp += f"_{int(now * 1_000_000) % 1_000_000:06d}"
    target = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
    attempt = 1
    while target.exists():
        target = path.with_name(f"{path.stem}.corrupt-{stamp}_{attempt}{path.suffix}")
        attempt += 1
    try:
        os.replace(path, target)
        return str(target)
    except OSError:  # pragma: no cover - disk-level failure
        return None


def _cycle_canonical(cycle: list[str]) -> list[str]:
    """Rotate a cycle so it starts at its lexicographically smallest id."""
    if not cycle:
        return []
    start = min(range(len(cycle)), key=lambda index: cycle[index])
    return cycle[start:] + cycle[:start]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class TaskManagerConfig:
    """Bounds that keep the shared task graph small and auditable.

    All limits are *task counts*: they are enforced before a task is added so
    the director can never silently grow an unbounded queue.
    """

    #: Hard cap on non-terminal tasks (the live queue).
    max_active_tasks: int = 200
    #: Hard cap on non-terminal tasks owned by one agent.
    max_tasks_per_agent: int = 60
    #: Hard cap on archived terminal task records.
    archive_limit: int = 500
    #: How often a single task may be attempted (``record_attempt``).
    max_attempts_per_task: int = 3
    #: Hard cap on direct children of one parent task.
    max_children_per_task: int = 12


# --------------------------------------------------------------------------- #
# TaskManager
# --------------------------------------------------------------------------- #

class TaskManager:
    """Owns the task graph, lifecycle transitions, de-duplication and history.

    Parameters
    ----------
    config:
        :class:`TaskManagerConfig`; a default instance is created when omitted.
    path:
        Optional JSON persistence path.  A directory is interpreted as
        ``<path>/tasks.json``.  An existing document is auto-loaded on
        construction.
    clock:
        Injectable ``() -> float`` wall-clock used for every timestamp; tests
        pass a counter to get fully deterministic records.
    branch_id:
        Current checkpoint branch.  New tasks inherit it unless they override
        it, and :meth:`active` only reports current-branch work when a branch is
        selected.  Tasks from other branches stay loaded for audit.
    """

    def __init__(
        self,
        config: TaskManagerConfig | None = None,
        *,
        path: str | Path = "",
        clock: Callable[[], float] | None = None,
        branch_id: str = "",
    ) -> None:
        self._config = config or TaskManagerConfig()
        self._clock: Callable[[], float] = clock or now_ts
        self._branch_id = str(branch_id or "")
        self._lock = threading.RLock()

        self._tasks: dict[str, Task] = {}
        self._dedupe: dict[str, str] = {}
        self._archive: list[dict[str, Any]] = []
        self._counters: dict[str, int] = {
            "created": 0,
            "started": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "reopened": 0,
            "attempts": 0,
            "archived": 0,
            "invalidated": 0,
        }

        self._path: Path | None = None
        raw_path = str(path or "").strip()
        if raw_path:
            candidate = Path(raw_path)
            # An existing directory is interpreted as ``<dir>/tasks.json``.
            self._path = candidate / TASKS_FILENAME if candidate.is_dir() else candidate
        self.load_errors: list[str] = []
        self.save_errors: list[str] = []
        self.recovered_from: str | None = None
        self.last_saved_at: float = 0.0

        if self._path is not None:
            self.load()

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> TaskManagerConfig:
        """The (immutable-by-convention) bound configuration."""
        return self._config

    @property
    def branch_id(self) -> str:
        """Current checkpoint branch id (``""`` when branch-agnostic)."""
        return self._branch_id

    def set_branch(self, branch_id: str) -> None:
        """Switch the current branch (new tasks inherit it; old ones stay)."""
        with self._lock:
            self._branch_id = str(branch_id or "")

    @property
    def path(self) -> Path | None:
        """Resolved persistence path (``None`` when persistence is disabled)."""
        return self._path

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)

    def __contains__(self, task_id: object) -> bool:
        with self._lock:
            return str(task_id) in self._tasks

    # ------------------------------------------------------------------ #
    # Creation
    # ------------------------------------------------------------------ #

    def create_task(
        self,
        *,
        owner_agent: AgentRole | str,
        requested_by: AgentRole | str,
        task_type: TaskType | str,
        description: str,
        parent_task_id: str = "",
        dependencies: Iterable[str] = (),
        priority: Priority | int | str = Priority.NORMAL,
        branch_id: str = "",
        payload: Mapping[str, Any] | None = None,
        task_id: str = "",
    ) -> tuple[Task, bool]:
        """Create (or de-duplicate) one task.

        Returns ``(task, created)``.  ``created`` is ``False`` when a
        non-terminal task with the same :attr:`Task.dedupe_key` already exists:
        that existing task is returned instead of queuing the same goal twice.

        Raises
        ------
        TaskError
            Invalid owner/field, unknown parent/dependency, or a configured
            limit exceeded (queue, per-agent, children).
        CycleDetectedError
            The requested dependencies would introduce a cycle.
        """
        with self._lock:
            owner = self._coerce_role(owner_agent, "owner_agent")
            requester = self._coerce_role(requested_by, "requested_by")
            branch = str(branch_id or self._branch_id or "")
            try:
                candidate = Task.create(
                    owner_agent=owner,
                    requested_by=requester,
                    task_type=task_type,
                    description=description,
                    parent_task_id=str(parent_task_id or ""),
                    dependencies=dependencies,
                    priority=priority,
                    branch_id=branch,
                    payload=payload,
                    task_id=task_id,
                    created_at=self._now(),
                )
            except ProtocolError as exc:
                raise TaskError(str(exc), details=exc.details) from exc

            if candidate.task_id in self._tasks:
                raise TaskError(
                    f"task id already exists: {candidate.task_id!r}",
                    details={"task_id": candidate.task_id},
                )

            existing_id = self._dedupe.get(candidate.dedupe_key)
            if existing_id is not None:
                existing = self._tasks.get(existing_id)
                if existing is not None and not existing.terminal:
                    return copy.deepcopy(existing), False

            parent_id = candidate.parent_task_id
            if parent_id:
                if parent_id not in self._tasks:
                    raise TaskError(
                        f"unknown parent task: {parent_id!r}",
                        details={"parent_task_id": parent_id},
                    )
                siblings = self._children_unlocked(parent_id)
                if len(siblings) >= int(self._config.max_children_per_task):
                    raise TaskError(
                        f"parent {parent_id!r} already has {len(siblings)} children "
                        f"(max {self._config.max_children_per_task})",
                        details={"parent_task_id": parent_id},
                    )

            for dependency in candidate.dependencies:
                if dependency not in self._tasks:
                    raise TaskError(
                        f"unknown dependency: {dependency!r}",
                        details={"task_id": candidate.task_id, "dependency": dependency},
                    )

            if self._would_create_cycle(candidate.task_id, candidate.dependencies):
                raise CycleDetectedError(
                    f"dependency cycle would be created by task {candidate.task_id!r}",
                    details={"task_id": candidate.task_id},
                )

            live = self._live_unlocked()
            if len(live) >= int(self._config.max_active_tasks):
                raise TaskError(
                    f"task queue is full: {len(live)} live tasks "
                    f"(max {self._config.max_active_tasks})"
                )
            owned = sum(1 for task in live if task.owner_agent is owner)
            if owned >= int(self._config.max_tasks_per_agent):
                raise TaskError(
                    f"agent {owner.value!r} already owns {owned} live tasks "
                    f"(max {self._config.max_tasks_per_agent})",
                    details={"owner_agent": owner.value},
                )

            self._tasks[candidate.task_id] = candidate
            self._dedupe[candidate.dedupe_key] = candidate.task_id
            self._counters["created"] += 1
            return copy.deepcopy(candidate), True

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def get(self, task_id: str) -> Task | None:
        """Return a defensive copy of ``task_id`` or ``None``."""
        with self._lock:
            task = self._tasks.get(str(task_id or ""))
            return copy.deepcopy(task) if task is not None else None

    def require(self, task_id: str) -> Task:
        """Return a defensive copy of ``task_id``; raise :class:`TaskError`."""
        with self._lock:
            return copy.deepcopy(self._require_unlocked(task_id))

    def tasks(
        self,
        *,
        statuses: Iterable[TaskStatus | str] | None = None,
        owner: AgentRole | str | None = None,
    ) -> list[Task]:
        """List tasks (all branches), optionally filtered and deterministically sorted."""
        wanted: set[TaskStatus] | None = None
        if statuses is not None:
            if isinstance(statuses, (TaskStatus, str)):
                statuses = [statuses]
            wanted = set()
            for status in statuses:
                try:
                    wanted.add(TaskStatus.coerce(status))
                except (ValueError, ProtocolError) as exc:
                    raise TaskError(f"unknown task status filter: {status!r}") from exc
        owner_role: AgentRole | None = None
        if owner is not None:
            owner_role = self._coerce_role(owner, "owner")
        with self._lock:
            items = [
                copy.deepcopy(task)
                for task in sorted(self._tasks.values(), key=lambda item: (item.created_at, item.task_id))
                if (wanted is None or task.status in wanted)
                and (owner_role is None or task.owner_agent is owner_role)
            ]
            return items

    def active(self) -> list[Task]:
        """Unfinished tasks on the current branch (all live tasks when unscoped).

        Tasks created on an older checkpoint branch remain stored for audit but
        are never reported as active once a branch is selected.
        """
        with self._lock:
            return [
                copy.deepcopy(task)
                for task in sorted(self._tasks.values(), key=lambda item: (item.created_at, item.task_id))
                if task.status in LIVE_STATUSES and self._in_scope(task)
            ]

    def children(self, task_id: str) -> list[Task]:
        """Direct children of ``task_id`` in deterministic creation order."""
        with self._lock:
            return [copy.deepcopy(task) for task in self._children_unlocked(task_id)]

    def refresh_ready(self) -> list[Task]:
        """Promote every ``PENDING`` task whose dependencies are all ``COMPLETED``.

        Returns the tasks that changed to :data:`TaskStatus.READY`.
        """
        with self._lock:
            changed: list[Task] = []
            for task in sorted(self._tasks.values(), key=lambda item: (item.created_at, item.task_id)):
                if task.status is not TaskStatus.PENDING:
                    continue
                if not self._dependencies_satisfied(task):
                    continue
                self._commit_status(task, TaskStatus.READY)
                changed.append(copy.deepcopy(task))
            return changed

    def ready(self) -> list[Task]:
        """Schedulable tasks on the current branch (deterministic order)."""
        with self._lock:
            return [
                copy.deepcopy(task)
                for task in sorted(self._tasks.values(), key=lambda item: (item.created_at, item.task_id))
                if task.status is TaskStatus.READY and self._in_scope(task)
            ]

    # ------------------------------------------------------------------ #
    # Lifecycle transitions
    # ------------------------------------------------------------------ #

    def start(self, task_id: str) -> Task:
        """Move a ``READY`` (or paused-and-restartable) task to ``RUNNING``."""
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.status not in (TaskStatus.READY, TaskStatus.PAUSED):
                raise TaskError(
                    f"cannot start task {task.task_id!r} in status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            self._commit_status(task, TaskStatus.RUNNING)
            self._append_evidence(
                task,
                {"kind": "started", "at": self._now(), "attempt": int(task.attempts)},
            )
            self._counters["started"] += 1
            return copy.deepcopy(task)

    def pause(self, task_id: str, *, reason: str = "") -> Task:
        """Pause an executing/ready task while an interrupting event is handled."""
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.status not in (TaskStatus.RUNNING, TaskStatus.READY):
                raise TaskError(
                    f"cannot pause task {task.task_id!r} in status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            from_status = task.status
            task.interruptions = int(task.interruptions) + 1
            self._commit_status(task, TaskStatus.PAUSED, reason=reason)
            self._append_evidence(
                task,
                {
                    "kind": "paused",
                    "at": self._now(),
                    "from": from_status.value,
                    "reason": _clean_text(reason),
                },
            )
            return copy.deepcopy(task)

    def resume(self, task_id: str) -> Task:
        """Return a paused task to the ready queue (the scheduler re-selects it)."""
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.status is not TaskStatus.PAUSED:
                raise TaskError(
                    f"cannot resume task {task.task_id!r} in status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            self._commit_status(task, TaskStatus.READY)
            return copy.deepcopy(task)

    def block(self, task_id: str, *, reason: str) -> Task:
        """Mark a task as blocked; a non-empty auditable ``reason`` is required."""
        clean_reason = _clean_text(reason)
        if not clean_reason:
            raise TaskError("blocking a task requires a reason")
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.status not in (
                TaskStatus.PENDING,
                TaskStatus.READY,
                TaskStatus.RUNNING,
                TaskStatus.PAUSED,
            ):
                raise TaskError(
                    f"cannot block task {task.task_id!r} in status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            self._commit_status(task, TaskStatus.BLOCKED, reason=clean_reason)
            return copy.deepcopy(task)

    def unblock(self, task_id: str) -> Task:
        """Release a blocked task: ``READY`` when its dependencies are done, else ``PENDING``."""
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.status is not TaskStatus.BLOCKED:
                raise TaskError(
                    f"cannot unblock task {task.task_id!r} in status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            target = (
                TaskStatus.READY
                if self._dependencies_satisfied(task)
                else TaskStatus.PENDING
            )
            self._commit_status(task, target)
            task.reason = ""
            return copy.deepcopy(task)

    def complete(
        self,
        task_id: str,
        *,
        result: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> Task:
        """Finish a task with evidence; refuses to complete without any evidence.

        The task must currently be ``RUNNING`` or ``VERIFYING`` (the two states
        from which the frozen transition table allows ``COMPLETED``).  The
        ``result`` summary and the ``evidence`` mapping are appended to
        ``task.evidence`` so a director can verify the claim later.
        """
        summary = _clean_text(result)
        evidence_map: dict[str, Any] = dict(evidence or {})
        if not summary and not evidence_map:
            raise TaskError(
                "completing a task requires a result summary or evidence",
                details={"task_id": str(task_id or "")},
            )
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.status not in (TaskStatus.RUNNING, TaskStatus.VERIFYING):
                raise TaskError(
                    f"cannot complete task {task.task_id!r} in status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            record: dict[str, Any] = dict(evidence_map)
            record["kind"] = "completion"
            record["at"] = self._now()
            record["result"] = summary
            self._append_evidence(task, record)
            task.progress = 1.0
            task.reason = ""
            self._commit_status(task, TaskStatus.COMPLETED)
            self._counters["completed"] += 1
            return copy.deepcopy(task)

    def verify(
        self,
        task_id: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> Task:
        """Director verification: confirm a completion (``VERIFYING``/``COMPLETED``).

        Verifying a ``RUNNING`` task directly is allowed only with evidence (a
        completion claim without any evidence is never trusted).  The
        verification record is appended with ``kind="verification"``.
        """
        evidence_map: dict[str, Any] = dict(evidence or {})
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.status is TaskStatus.RUNNING and not evidence_map:
                raise TaskError(
                    "verifying a running task requires evidence",
                    details={"task_id": task.task_id},
                )
            if task.status not in (
                TaskStatus.VERIFYING,
                TaskStatus.COMPLETED,
                TaskStatus.RUNNING,
            ):
                raise TaskError(
                    f"cannot verify task {task.task_id!r} in status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            record: dict[str, Any] = dict(evidence_map)
            record["kind"] = "verification"
            record["at"] = self._now()
            record["verified"] = True
            self._append_evidence(task, record)
            if task.status is not TaskStatus.COMPLETED:
                self._commit_status(task, TaskStatus.COMPLETED)
                self._counters["completed"] += 1
            return copy.deepcopy(task)

    def fail(self, task_id: str, *, reason: str) -> Task:
        """Mark a task failed with an auditable reason (attempts are *not* reset)."""
        clean_reason = _clean_text(reason)
        if not clean_reason:
            raise TaskError("failing a task requires a reason")
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.terminal:
                raise TaskError(
                    f"cannot fail task {task.task_id!r} in terminal status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            self._append_evidence(
                task, {"kind": "failure", "at": self._now(), "reason": clean_reason}
            )
            self._commit_status(task, TaskStatus.FAILED, reason=clean_reason)
            self._counters["failed"] += 1
            return copy.deepcopy(task)

    def cancel(self, task_id: str, *, reason: str = "") -> Task:
        """Cancel live work (``VERIFYING`` must be paused/failed first)."""
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.terminal:
                raise TaskError(
                    f"cannot cancel task {task.task_id!r} in terminal status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            if task.status is TaskStatus.VERIFYING:
                raise TaskError(
                    "cannot cancel a task awaiting verification; pause or fail it first",
                    details={"task_id": task.task_id},
                )
            self._commit_status(task, TaskStatus.CANCELLED, reason=reason)
            self._counters["cancelled"] += 1
            return copy.deepcopy(task)

    def reopen(self, task_id: str, *, reason: str = "") -> Task:
        """Put a ``COMPLETED``/``FAILED`` task back into the ready queue."""
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                raise TaskError(
                    f"cannot reopen task {task.task_id!r} in status {task.status.value!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            key = task.dedupe_key
            collision = self._dedupe.get(key)
            if collision is not None and collision != task.task_id:
                raise TaskError(
                    f"reopening {task.task_id!r} would duplicate the live goal {collision!r}",
                    details={"task_id": task.task_id, "duplicate_task_id": collision},
                )
            self._commit_status(task, TaskStatus.READY, reason=reason)
            task.progress = 0.0
            self._append_evidence(
                task,
                {"kind": "reopened", "at": self._now(), "reason": _clean_text(reason)},
            )
            self._counters["reopened"] += 1
            return copy.deepcopy(task)

    # ------------------------------------------------------------------ #
    # Progress / attempts
    # ------------------------------------------------------------------ #

    def update_progress(
        self,
        task_id: str,
        progress: float,
        *,
        note: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> Task:
        """Record monotonic progress in ``[0, 1]`` plus optional evidence."""
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.terminal:
                raise TaskError(
                    f"cannot update progress of terminal task {task.task_id!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            value = max(0.0, min(1.0, _as_float(progress, task.progress)))
            task.progress = max(float(task.progress), value)
            record: dict[str, Any] = dict(evidence or {})
            if note or record:
                record.setdefault("kind", "progress")
                record["at"] = self._now()
                record["progress"] = task.progress
                if note:
                    record["note"] = _clean_text(note)
                self._append_evidence(task, record)
            task.touch(at=self._now())
            return copy.deepcopy(task)

    def record_attempt(self, task_id: str) -> int:
        """Count one execution attempt and return the new attempt number.

        Raises :class:`TaskError` once ``max_attempts_per_task`` is reached, so a
        caller can never loop forever on a hopeless task.
        """
        with self._lock:
            task = self._require_unlocked(task_id)
            if task.terminal:
                raise TaskError(
                    f"cannot record an attempt for terminal task {task.task_id!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            limit = max(0, int(self._config.max_attempts_per_task))
            if int(task.attempts) >= limit:
                raise TaskError(
                    f"task {task.task_id!r} exhausted its {limit} attempts",
                    details={"task_id": task.task_id, "attempts": int(task.attempts)},
                )
            task.attempts = int(task.attempts) + 1
            task.touch(at=self._now())
            self._append_evidence(
                task, {"kind": "attempt", "at": self._now(), "attempt": int(task.attempts)}
            )
            self._counters["attempts"] += 1
            return int(task.attempts)

    # ------------------------------------------------------------------ #
    # Dependency graph
    # ------------------------------------------------------------------ #

    def add_dependency(self, task_id: str, dependency_id: str) -> Task:
        """Add one dependency edge; raises :class:`CycleDetectedError` on a cycle."""
        with self._lock:
            task = self._require_unlocked(task_id)
            self._require_unlocked(dependency_id)
            if task.terminal:
                raise TaskError(
                    f"cannot add a dependency to terminal task {task.task_id!r}",
                    details={"task_id": task.task_id, "status": task.status.value},
                )
            if dependency_id == task.task_id:
                raise CycleDetectedError(
                    "a task cannot depend on itself",
                    details={"task_id": task.task_id},
                )
            if dependency_id in task.dependencies:
                return copy.deepcopy(task)
            if self._reachable(start=dependency_id, target=task.task_id):
                raise CycleDetectedError(
                    f"adding {task.task_id!r} -> {dependency_id!r} would create a cycle",
                    details={"task_id": task.task_id, "dependency": dependency_id},
                )
            task.dependencies = sorted({*task.dependencies, dependency_id})
            task.touch(at=self._now())
            return copy.deepcopy(task)

    def remove_dependency(self, task_id: str, dependency_id: str) -> Task:
        """Remove one dependency edge (raises when the edge does not exist)."""
        with self._lock:
            task = self._require_unlocked(task_id)
            if dependency_id not in task.dependencies:
                raise TaskError(
                    f"task {task.task_id!r} does not depend on {dependency_id!r}",
                    details={"task_id": task.task_id, "dependency": dependency_id},
                )
            task.dependencies = [dep for dep in task.dependencies if dep != dependency_id]
            task.touch(at=self._now())
            return copy.deepcopy(task)

    def detect_cycles(self) -> list[list[str]]:
        """Return every dependency cycle in the graph (deterministic order)."""
        with self._lock:
            return self._detect_cycles_unlocked()

    # ------------------------------------------------------------------ #
    # Assignment / metadata
    # ------------------------------------------------------------------ #

    def set_status(
        self,
        task_id: str,
        status: TaskStatus | str,
        *,
        reason: str = "",
    ) -> Task:
        """Apply one explicit lifecycle transition (integration compatibility helper).

        Only transitions allowed by the frozen
        :data:`~llm_agent.multi_agent.contracts.TASK_TRANSITIONS` table are
        accepted, and a task may only be forced to ``READY`` when all its
        dependencies are actually ``COMPLETED`` — this helper can re-queue a
        task but can never bypass the honesty/ordering rules.
        """
        with self._lock:
            task = self._require_unlocked(task_id)
            try:
                target = TaskStatus.coerce(status)
            except (ValueError, ProtocolError) as exc:
                raise TaskError(f"invalid task status: {status!r}") from exc
            if target is TaskStatus.READY and not self._dependencies_satisfied(task):
                raise TaskError(
                    f"task {task.task_id!r} still has incomplete dependencies",
                    details={"task_id": task.task_id},
                )
            self._commit_status(task, target, reason=reason)
            return copy.deepcopy(task)

    def set_priority(self, task_id: str, priority: Priority | int | str) -> Task:
        """Change a task's priority (scheduler input, no lifecycle transition)."""
        with self._lock:
            task = self._require_unlocked(task_id)
            try:
                task.priority = Priority.coerce(priority)
            except (ValueError, ProtocolError) as exc:
                raise TaskError(f"invalid priority {priority!r}: {exc}") from exc
            task.touch(at=self._now())
            return copy.deepcopy(task)

    def reassign(self, task_id: str, owner_agent: AgentRole | str) -> Task:
        """Move a task to another owner, keeping the de-duplication index consistent."""
        with self._lock:
            task = self._require_unlocked(task_id)
            owner = self._coerce_role(owner_agent, "owner_agent")
            if owner is task.owner_agent:
                return copy.deepcopy(task)
            old_key = task.dedupe_key
            if self._dedupe.get(old_key) == task.task_id:
                del self._dedupe[old_key]
            old_owner = task.owner_agent
            task.owner_agent = owner
            new_key = task.dedupe_key
            collision = self._dedupe.get(new_key)
            if collision is not None and collision != task.task_id:
                # Roll the change back: never leave two live tasks with one goal.
                task.owner_agent = old_owner
                if not task.terminal:
                    self._dedupe[old_key] = task.task_id
                raise TaskError(
                    f"reassigning {task.task_id!r} would duplicate {collision!r}",
                    details={"task_id": task.task_id, "owner_agent": owner.value},
                )
            if not task.terminal:
                self._dedupe[new_key] = task.task_id
            task.touch(at=self._now())
            return copy.deepcopy(task)

    # ------------------------------------------------------------------ #
    # Rollback / archive
    # ------------------------------------------------------------------ #

    def invalidate_branch(self, old_branch_id: str, *, reason: str = "") -> list[Task]:
        """Cancel every unfinished task created on a rolled-back branch.

        Terminal tasks of that branch are kept untouched for audit; completed
        work is never deleted.  The cancelled tasks get
        ``reason="historical_branch:<old_branch_id>"`` so the audit trail shows
        why they were abandoned.
        """
        branch = str(old_branch_id or "")
        if not branch:
            raise TaskError("invalidate_branch requires a non-empty branch id")
        suffix = f" ({_clean_text(reason)})" if _clean_text(reason) else ""
        with self._lock:
            changed: list[Task] = []
            for task in sorted(self._tasks.values(), key=lambda item: (item.created_at, item.task_id)):
                if task.branch_id != branch or task.terminal:
                    continue
                if task.dedupe_key and self._dedupe.get(task.dedupe_key) == task.task_id:
                    del self._dedupe[task.dedupe_key]
                task.status = TaskStatus.CANCELLED
                task.reason = f"historical_branch:{branch}{suffix}"
                task.touch(at=self._now())
                self._counters["invalidated"] += 1
                changed.append(copy.deepcopy(task))
            self._counters["cancelled"] += len(changed)
            return changed

    def archive_terminal(self, *, keep: int = 0) -> int:
        """Move old terminal tasks out of the working set (bounded archive).

        The ``keep`` newest terminal tasks stay live; the rest are appended to
        the archive (capped by ``archive_limit``).  Returns how many moved.
        """
        keep_count = max(0, _as_int(keep, 0))
        with self._lock:
            terminal = sorted(
                (task for task in self._tasks.values() if task.terminal),
                key=lambda item: (item.updated_at, item.task_id),
                reverse=True,
            )
            stale = terminal[keep_count:]
            now = self._now()
            for task in stale:
                self._archive.append({"task": task.to_dict(), "archived_at": now})
                del self._tasks[task.task_id]
            limit = max(0, int(self._config.archive_limit))
            if limit and len(self._archive) > limit:
                del self._archive[: len(self._archive) - limit]
            self._counters["archived"] += len(stale)
            return len(stale)

    # ------------------------------------------------------------------ #
    # Snapshot / persistence
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the whole manager state."""
        with self._lock:
            return {
                "format": TASK_MANAGER_FORMAT,
                "schema_version": TASK_MANAGER_VERSION,
                "saved_at": self._now(),
                "branch_id": self._branch_id,
                "tasks": [
                    self._tasks[task_id].to_dict()
                    for task_id in sorted(self._tasks)
                ],
                "archive": copy.deepcopy(self._archive),
                "counters": dict(self._counters),
            }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Replace the current state by ``snapshot`` and rebuild all indexes."""
        if not isinstance(snapshot, Mapping):
            raise TaskError("task snapshot must be a mapping")
        with self._lock:
            self._tasks = {}
            self._archive = []
            self.load_errors = []
            tasks_raw = snapshot.get("tasks") or []
            if not isinstance(tasks_raw, (list, tuple)):
                tasks_raw = []
            for item in tasks_raw:
                if not isinstance(item, Mapping):
                    continue
                try:
                    task = Task.from_dict(item)
                except MultiAgentError as exc:
                    self.load_errors.append(f"skipped invalid task record: {exc}")
                    continue
                if task.task_id in self._tasks:
                    self.load_errors.append(f"skipped duplicate task id: {task.task_id}")
                    continue
                self._tasks[task.task_id] = task
            archive_raw = snapshot.get("archive") or []
            if isinstance(archive_raw, (list, tuple)):
                for record in archive_raw:
                    if isinstance(record, Mapping):
                        self._archive.append(copy.deepcopy(dict(record)))
            counters_raw = snapshot.get("counters")
            if isinstance(counters_raw, Mapping):
                for key, value in counters_raw.items():
                    if key in self._counters:
                        self._counters[str(key)] = max(0, _as_int(value, 0))
            branch = str(snapshot.get("branch_id") or "")
            if not self._branch_id and branch:
                self._branch_id = branch
            self._rebuild_dedupe()
            self.last_saved_at = _as_float(snapshot.get("saved_at"), 0.0)

    def save(self) -> None:
        """Persist the manager state atomically (no-op when ``path`` is unset)."""
        path = self._path
        if path is None:
            return
        with self._lock:
            payload = self.snapshot()
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
            try:
                _atomic_write_text(path, text)
            except OSError as exc:  # pragma: no cover - disk-level failure
                self.save_errors.append(f"task save failed: {exc}")
                return
            self.last_saved_at = _as_float(payload.get("saved_at"), self._now())

    def load(self) -> bool:
        """Load the persisted document; tolerant to damage (returns success)."""
        path = self._path
        self.load_errors = []
        self.recovered_from = None
        if path is None or not path.is_file():
            return False
        raw: Mapping[str, Any] | None = None
        try:
            text = path.read_text(encoding="utf-8")
            parsed = json.loads(text) if text.strip() else None
            if not isinstance(parsed, Mapping):
                raise ValueError("tasks file does not contain a JSON object")
            schema = _as_int(parsed.get("schema_version"), 0)
            if schema and schema > TASK_MANAGER_VERSION:
                raise ValueError(f"unsupported tasks schema_version {schema}")
            raw = parsed
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            self.load_errors.append(
                f"tasks file corrupt ({type(exc).__name__}: {exc}); started fresh"
            )
            backup = _backup_corrupt(path, now=self._now())
            if backup:
                self.recovered_from = backup
            self._reset_unlocked()
            return False
        assert raw is not None
        self.restore(raw)
        return True

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Compact counters for the viewer/director (deterministic)."""
        with self._lock:
            by_status = {status.value: 0 for status in TaskStatus}
            by_owner = {role.value: 0 for role in AgentRole}
            by_type = {task_type.value: 0 for task_type in TaskType}
            for task in self._tasks.values():
                by_status[task.status.value] += 1
                by_owner[task.owner_agent.value] += 1
                by_type[task.task_type.value] += 1
            live = len(self._live_unlocked())
            return {
                "branch_id": self._branch_id,
                "total": len(self._tasks),
                "live": live,
                "active": live,
                "ready": by_status[TaskStatus.READY.value],
                "running": by_status[TaskStatus.RUNNING.value],
                "paused": by_status[TaskStatus.PAUSED.value],
                "blocked": by_status[TaskStatus.BLOCKED.value],
                "pending": by_status[TaskStatus.PENDING.value],
                "verifying": by_status[TaskStatus.VERIFYING.value],
                "completed": by_status[TaskStatus.COMPLETED.value],
                "failed": by_status[TaskStatus.FAILED.value],
                "cancelled": by_status[TaskStatus.CANCELLED.value],
                "by_status": by_status,
                "by_owner": by_owner,
                "by_type": by_type,
                "dedupe_keys": len(self._dedupe),
                "dependencies": sum(len(task.dependencies) for task in self._tasks.values()),
                "attempts": sum(int(task.attempts) for task in self._tasks.values()),
                "archive": len(self._archive),
                "counters": dict(self._counters),
                "cycles": len(self._detect_cycles_unlocked()),
                "path": str(self._path) if self._path is not None else "",
                "load_errors": len(self.load_errors),
            }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _now(self) -> float:
        """Clock reading used for all mutation timestamps."""
        try:
            return float(self._clock())
        except Exception:  # pragma: no cover - a broken clock must not crash
            return now_ts()

    @staticmethod
    def _coerce_role(value: Any, field_name: str) -> AgentRole:
        try:
            return AgentRole.coerce(value)
        except ProtocolError as exc:
            raise TaskError(f"invalid {field_name}: {exc}") from exc

    def _require_unlocked(self, task_id: str) -> Task:
        """Internal requirement lookup (caller must hold the lock)."""
        key = str(task_id or "")
        task = self._tasks.get(key)
        if task is None:
            raise TaskError(f"unknown task: {key!r}", details={"task_id": key})
        return task

    def _in_scope(self, task: Task) -> bool:
        """True when ``task`` belongs to the current branch (or none is selected)."""
        if not self._branch_id:
            return True
        return task.branch_id == self._branch_id

    def _live_unlocked(self) -> list[Task]:
        return [task for task in self._tasks.values() if task.status in LIVE_STATUSES]

    def _children_unlocked(self, parent_id: str) -> list[Task]:
        key = str(parent_id or "")
        return sorted(
            (task for task in self._tasks.values() if task.parent_task_id == key),
            key=lambda item: (item.created_at, item.task_id),
        )

    def _rebuild_dedupe(self) -> None:
        """Rebuild the de-duplication index from the current task set."""
        self._dedupe = {}
        for task_id in sorted(self._tasks):
            task = self._tasks[task_id]
            if task.status in LIVE_STATUSES:
                self._dedupe.setdefault(task.dedupe_key, task_id)

    def _commit_status(
        self,
        task: Task,
        target: TaskStatus,
        *,
        reason: str = "",
    ) -> None:
        """Validate + apply a transition and keep the de-duplication index in sync."""
        task.transition(target, reason=reason, at=self._now())
        if task.status in LIVE_STATUSES:
            self._dedupe[task.dedupe_key] = task.task_id
        else:
            if self._dedupe.get(task.dedupe_key) == task.task_id:
                del self._dedupe[task.dedupe_key]

    def _append_evidence(self, task: Task, record: Mapping[str, Any]) -> None:
        """Append one bounded evidence record (the newest records are kept)."""
        task.evidence.append(copy.deepcopy(dict(record)))
        overflow = len(task.evidence) - MAX_EVIDENCE_PER_TASK
        if overflow > 0:
            del task.evidence[:overflow]

    def _dependencies_satisfied(self, task: Task) -> bool:
        """True when every dependency exists and is ``COMPLETED``."""
        for dependency in task.dependencies:
            dep_task = self._tasks.get(dependency)
            if dep_task is None or dep_task.status is not TaskStatus.COMPLETED:
                return False
        return True

    def _reachable(self, *, start: str, target: str) -> bool:
        """True when ``start`` can reach ``target`` following dependency edges."""
        if start == target:
            return True
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            task = self._tasks.get(node)
            if task is None:
                continue
            for dependency in task.dependencies:
                if dependency == target:
                    return True
                if dependency not in seen:
                    stack.append(dependency)
        return False

    def _would_create_cycle(self, new_id: str, dependencies: Iterable[str]) -> bool:
        """Cycle check for a prospective task node (before it is inserted)."""
        seen: set[str] = set()
        stack = [str(dep) for dep in dependencies]
        while stack:
            node = stack.pop()
            if node == new_id:
                return True
            if node in seen:
                continue
            seen.add(node)
            task = self._tasks.get(node)
            if task is None:
                continue
            stack.extend(task.dependencies)
        return False

    def _detect_cycles_unlocked(self) -> list[list[str]]:
        """Iterative DFS over the dependency graph; returns normalised cycles."""
        cycles: list[list[str]] = []
        seen_cycles: set[tuple[str, ...]] = set()
        color: dict[str, int] = {}
        for start in sorted(self._tasks):
            if color.get(start, 0) != 0:
                continue
            path: list[str] = [start]
            color[start] = 1
            stack: list[tuple[str, list[str], int]] = [
                (start, sorted(self._tasks[start].dependencies), 0)
            ]
            while stack:
                node, deps, index = stack[-1]
                if index >= len(deps):
                    stack.pop()
                    color[node] = 2
                    path.pop()
                    continue
                stack[-1] = (node, deps, index + 1)
                nxt = deps[index]
                if nxt not in self._tasks:
                    continue
                state = color.get(nxt, 0)
                if state == 1:
                    if nxt in path:
                        cycle = _cycle_canonical(path[path.index(nxt):])
                        key = tuple(cycle)
                        if key not in seen_cycles:
                            seen_cycles.add(key)
                            cycles.append(cycle)
                    continue
                if state == 0:
                    color[nxt] = 1
                    path.append(nxt)
                    stack.append((nxt, sorted(self._tasks[nxt].dependencies), 0))
        cycles.sort()
        return cycles

    def _reset_unlocked(self) -> None:
        """Clear the manager state (used by tolerant load)."""
        self._tasks = {}
        self._dedupe = {}
        self._archive = []
        for key in self._counters:
            self._counters[key] = 0
        self.last_saved_at = 0.0
