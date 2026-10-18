"""Deterministic task selection for the Pokémon Red multi-agent director.

The scheduler is a *pure scoring function*: it never touches the emulator, the
network or a task manager.  Given the currently known
:class:`~llm_agent.multi_agent.contracts.Task` records it ranks them and returns
the next one to execute, with a fully explainable score breakdown.

Mandated ordering (default reference weights, configurable)
-----------------------------------------------------------
1. urgent recovery tasks (``RECOVER`` type or ``CRITICAL`` priority);
2. continue already-running / paused high-priority work (status boost);
3. story progression (``TALK`` / ``INTERACT`` / ``INVESTIGATE``);
4. navigation / exploration (``NAVIGATE`` / ``EXPLORE``);
5. low-value tasks.

On top of the type tier the score applies priority, aging (older work gains
value), starvation protection (``starvation_after_s``), attempt and
interruption penalties, a "just failed" penalty (unless the task's attempt or
priority state changed) and an owner-rotation penalty so a single agent can not
be selected forever while other work is ready.

Determinism
-----------
All scores are plain floats computed from the task record and the injected
clock.  Ties are broken by age (older task first) and then by ``task_id``, so
two runs over the same inputs produce exactly the same ranking.  Selection is a
single sort — it always terminates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from llm_agent.multi_agent.contracts import (
    AgentRole,
    MultiAgentError,
    Priority,
    Task,
    TaskStatus,
    TaskType,
    now_ts,
)

__all__ = [
    "SchedulerConfig",
    "TaskScheduler",
    "DEFAULT_TYPE_WEIGHTS",
    "PRIORITY_WEIGHT",
    "CONTINUE_BONUS",
    "PAUSED_BONUS",
    "VERIFYING_PENALTY",
    "RECOVERY_TYPE_BONUS",
    "CRITICAL_PRIORITY_BONUS",
    "AGING_CAP",
    "STARVATION_BONUS",
    "PROGRESS_BONUS",
    "FAILED_TASK_PENALTY",
    "RECENT_SELECTION_PENALTY",
    "OWNER_ROTATION_PENALTY",
]

#: Default tier weights per task type (higher = more important).
DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    TaskType.RECOVER.value: 100.0,
    # story progression tier
    TaskType.TALK.value: 70.0,
    TaskType.INTERACT.value: 65.0,
    TaskType.INVESTIGATE.value: 60.0,
    # battle work happens while executing story/navigation tasks
    TaskType.BATTLE.value: 62.0,
    TaskType.CATCH.value: 58.0,
    # navigation / exploration tier
    TaskType.NAVIGATE.value: 45.0,
    TaskType.EXPLORE.value: 40.0,
    TaskType.TRAIN.value: 38.0,
    TaskType.VERIFY.value: 35.0,
    TaskType.DIRECTOR_REVIEW.value: 30.0,
    TaskType.CUSTOM.value: 20.0,
}

#: Weight applied to ``int(task.priority)`` (0..100 -> 0..50 points).
PRIORITY_WEIGHT: float = 0.5
#: Bonus for continuing an already running task (mandate rule 2).
CONTINUE_BONUS: float = 30.0
#: Bonus for resuming a paused task (still "already started" work).
PAUSED_BONUS: float = 25.0
#: Penalty for a task already awaiting verification (director owns it now).
VERIFYING_PENALTY: float = 40.0
#: Extra bonus for the recovery task type (mandate rule 1).
RECOVERY_TYPE_BONUS: float = 60.0
#: Extra bonus for CRITICAL priority tasks (urgent interrupts).
CRITICAL_PRIORITY_BONUS: float = 40.0
#: Cap of the aging component (hours-old work must not dwarf everything).
AGING_CAP: float = 80.0
#: Extra boost once a task has waited longer than ``starvation_after_s``.
STARVATION_BONUS: float = 35.0
#: Weight of the recorded progress fraction (work close to done is valuable).
PROGRESS_BONUS: float = 10.0
#: Penalty for re-selecting a task that just failed unchanged.
FAILED_TASK_PENALTY: float = 120.0
#: Penalty for re-selecting the task that was chosen in the previous slot.
RECENT_SELECTION_PENALTY: float = 15.0
#: Penalty applied to the owner that was selected too many times in a row.
OWNER_ROTATION_PENALTY: float = 400.0

#: Statuses that can never be selected again.
_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)
#: Statuses excluded from selection on top of the terminal ones.
_EXCLUDED_STATUSES: frozenset[TaskStatus] = _TERMINAL_STATUSES | {TaskStatus.BLOCKED}

#: Bounded bookkeeping so long runs never grow memory without limit.
MAX_HISTORY = 256
MAX_TRACKED_FAILURES = 256


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class SchedulerConfig:
    """Weights and penalties of the deterministic scheduler.

    ``type_weights`` may be partially overridden; every type not present keeps
    its :data:`DEFAULT_TYPE_WEIGHTS` value.  Keys are accepted as enum values
    (``"recover"``) or :class:`TaskType` members.
    """

    #: Tier weight per task type (see :data:`DEFAULT_TYPE_WEIGHTS`).
    type_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TYPE_WEIGHTS)
    )
    #: Same owner may not be selected more than this many times in a row when
    #: other work is ready.
    owner_rotation_limit: int = 3
    #: Aging boost per minute of waiting (capped by :data:`AGING_CAP`).
    aging_boost_per_minute: float = 2.0
    #: Penalty per recorded execution attempt.
    attempt_penalty: float = 8.0
    #: Penalty per interruption the task suffered.
    interruption_penalty: float = 1.0
    #: A task older than this many seconds gets the starvation boost.
    starvation_after_s: float = 120.0


# --------------------------------------------------------------------------- #
# TaskScheduler
# --------------------------------------------------------------------------- #

class TaskScheduler:
    """Rank tasks and pick the next one with an explainable score.

    Parameters
    ----------
    config:
        :class:`SchedulerConfig`; defaults apply when omitted.
    clock:
        Injectable ``() -> float`` wall clock.  All age/aging math uses it so
        tests can advance time deterministically.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config or SchedulerConfig()
        self._clock: Callable[[], float] = clock or now_ts
        self._type_weights = self._normalize_weights(self._config.type_weights)

        #: owners of the most recent selections, oldest first (bounded)
        self._owner_history: list[str] = []
        #: task_id -> wall time it was last selected
        self._last_selected: dict[str, float] = {}
        #: task selected in the immediately previous selection slot
        self._last_selected_task_id = ""
        #: task_id -> signature of the failure state (attempts/priority)
        self._failures: dict[str, dict[str, Any]] = {}
        self._rank_calls = 0
        self._select_calls = 0
        self._selected_count = 0
        self._finished_ok = 0
        self._finished_fail = 0

    # ------------------------------------------------------------------ #
    # Configuration / lifecycle
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> SchedulerConfig:
        """The scheduler configuration in use."""
        return self._config

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #

    def next_task(
        self,
        tasks: Sequence[Task],
        *,
        owner_available: Mapping[AgentRole, bool] | Mapping[str, bool] | None = None,
    ) -> Task | None:
        """Return the best eligible task, or ``None`` when nothing is schedulable."""
        self._select_calls += 1
        task_list = [task for task in tasks if isinstance(task, Task)]
        ranked = self.rank(task_list, owner_available=owner_available)
        if not ranked:
            return None
        by_id = {task.task_id: task for task in task_list}
        for item in ranked:
            task = by_id.get(str(item["task_id"]))
            if task is not None:
                return task
        return None

    def rank(
        self,
        tasks: Iterable[Task],
        *,
        owner_available: Mapping[AgentRole, bool] | Mapping[str, bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Score every eligible task; returns explainable items sorted best first.

        Each item is ``{"task_id", "score", "parts", "owner", "task_type",
        "priority"}``.  ``parts`` lists every component of the score so the
        choice is auditable/debuggable.
        """
        self._rank_calls += 1
        now = self._now()
        available = self._normalize_available(owner_available)
        streak_owner = self._owner_streak_owner()
        streak = self._current_owner_streak()
        created_at: dict[str, float] = {}
        items: list[dict[str, Any]] = []
        for task in tasks:
            if not isinstance(task, Task):
                continue
            if task.task_id in created_at:
                continue
            created_at[task.task_id] = float(task.created_at or 0.0)
            if task.status in _EXCLUDED_STATUSES:
                continue
            if available is not None and not self._role_available(task.owner_agent, available):
                continue
            score, parts = self._score(task, now=now, streak_owner=streak_owner, streak=streak)
            items.append(
                {
                    "task_id": task.task_id,
                    "score": round(score, 3),
                    "parts": parts,
                    "owner": task.owner_agent.value,
                    "task_type": task.task_type.value,
                    "priority": int(task.priority),
                }
            )
        items.sort(
            key=lambda item: (
                -item["score"],
                created_at.get(item["task_id"], 0.0),
                item["task_id"],
            )
        )
        return items

    def on_selected(self, task: Task) -> None:
        """Record that ``task`` was selected (owner rotation + audit bookkeeping)."""
        if not isinstance(task, Task):
            return
        owner = task.owner_agent.value
        self._owner_history.append(owner)
        if len(self._owner_history) > MAX_HISTORY:
            del self._owner_history[: len(self._owner_history) - MAX_HISTORY]
        self._last_selected[task.task_id] = self._now()
        self._last_selected_task_id = task.task_id
        if len(self._last_selected) > MAX_TRACKED_FAILURES:
            for key in list(self._last_selected)[: len(self._last_selected) - MAX_TRACKED_FAILURES]:
                self._last_selected.pop(key, None)
        self._selected_count += 1

    def on_finished(self, task: Task, *, ok: bool) -> None:
        """Record the outcome of a selected task.

        A failed task is remembered together with its attempt/priority
        signature.  It is de-prioritised until that signature changes (a retry
        attempt was recorded or its priority was raised).
        """
        if not isinstance(task, Task):
            return
        if ok:
            self._finished_ok += 1
            self._failures.pop(task.task_id, None)
            return
        self._finished_fail += 1
        self._failures[task.task_id] = {
            "attempts": int(task.attempts),
            "priority": int(task.priority),
            "at": self._now(),
        }
        if len(self._failures) > MAX_TRACKED_FAILURES:
            oldest = sorted(self._failures, key=lambda key: self._failures[key]["at"])
            for key in oldest[: len(self._failures) - MAX_TRACKED_FAILURES]:
                self._failures.pop(key, None)

    def reset(self) -> None:
        """Forget selection history / failure memory (keeps the configuration)."""
        self._owner_history = []
        self._last_selected = {}
        self._last_selected_task_id = ""
        self._failures = {}
        self._rank_calls = 0
        self._select_calls = 0
        self._selected_count = 0
        self._finished_ok = 0
        self._finished_fail = 0

    def stats(self) -> dict[str, Any]:
        """Compact, deterministic scheduler counters for the viewer/director."""
        return {
            "rank_calls": self._rank_calls,
            "select_calls": self._select_calls,
            "selected": self._selected_count,
            "finished_ok": self._finished_ok,
            "finished_fail": self._finished_fail,
            "owner_history": list(self._owner_history[-16:]),
            "current_owner_streak": self._current_owner_streak(),
            "last_selected_task_id": self._last_selected_task_id,
            "tracked_failures": len(self._failures),
            "tracked_selected": len(self._last_selected),
            "owner_rotation_limit": int(self._config.owner_rotation_limit),
            "type_weights": dict(self._type_weights),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _now(self) -> float:
        """Clock reading used for age/aging math."""
        try:
            return float(self._clock())
        except Exception:  # pragma: no cover - a broken clock must not crash
            return now_ts()

    @staticmethod
    def _normalize_weights(raw: Mapping[Any, Any] | None) -> dict[str, float]:
        """Merge overrides onto the default tier weights, normalising the keys."""
        weights: dict[str, float] = {
            str(key): float(value) for key, value in DEFAULT_TYPE_WEIGHTS.items()
        }
        if not raw:
            return weights
        for key, value in raw.items():
            try:
                task_type = TaskType.coerce(key)
            except MultiAgentError:  # pragma: no cover - TaskType.coerce never raises
                continue
            try:
                weights[task_type.value] = float(value)
            except (TypeError, ValueError):
                continue
        return weights

    @staticmethod
    def _normalize_available(
        raw: Mapping[AgentRole, bool] | Mapping[str, bool] | None,
    ) -> dict[AgentRole, bool] | None:
        """Normalise ``owner_available`` to ``{AgentRole: bool}`` (``None`` = all)."""
        if raw is None:
            return None
        out: dict[AgentRole, bool] = {}
        for key, value in raw.items():
            try:
                role = key if isinstance(key, AgentRole) else AgentRole.coerce(key)
            except (MultiAgentError, ValueError):  # pragma: no cover - defensive
                continue
            out[role] = bool(value)
        return out

    @staticmethod
    def _role_available(role: AgentRole, available: Mapping[AgentRole, bool]) -> bool:
        """Unknown owners count as available; only an explicit ``False`` blocks."""
        return bool(available.get(role, True))

    def _type_weight(self, task_type: TaskType) -> float:
        return float(self._type_weights.get(task_type.value, DEFAULT_TYPE_WEIGHTS.get(task_type.value, 20.0)))

    def _owner_streak_owner(self) -> str:
        """Owner of the most recent selection (``""`` when there is none)."""
        return self._owner_history[-1] if self._owner_history else ""

    def _current_owner_streak(self) -> int:
        """How many times the most recently selected owner was selected in a row."""
        if not self._owner_history:
            return 0
        owner = self._owner_history[-1]
        streak = 0
        for value in reversed(self._owner_history):
            if value != owner:
                break
            streak += 1
        return streak

    def _failure_penalty(self, task: Task) -> float:
        """Penalty for re-selecting a just-failed task whose state is unchanged."""
        record = self._failures.get(task.task_id)
        if record is None:
            return 0.0
        if int(task.attempts) != int(record.get("attempts", -1)):
            return 0.0
        if int(task.priority) != int(record.get("priority", -1)):
            return 0.0
        return FAILED_TASK_PENALTY

    def _score(
        self,
        task: Task,
        *,
        now: float,
        streak_owner: str,
        streak: int,
    ) -> tuple[float, dict[str, float]]:
        """Compute ``(score, parts)`` for one task (all components rounded)."""
        parts: dict[str, float] = {}

        type_score = self._type_weight(task.task_type)
        parts["type"] = round(type_score, 3)

        priority_score = int(task.priority) * PRIORITY_WEIGHT
        parts["priority"] = round(priority_score, 3)

        status_score = 0.0
        if task.status is TaskStatus.RUNNING:
            status_score = CONTINUE_BONUS
        elif task.status is TaskStatus.PAUSED:
            status_score = PAUSED_BONUS
        elif task.status is TaskStatus.VERIFYING:
            status_score = -VERIFYING_PENALTY
        parts["status"] = round(status_score, 3)

        recovery_score = 0.0
        if task.task_type is TaskType.RECOVER:
            recovery_score += RECOVERY_TYPE_BONUS
        if task.priority is Priority.CRITICAL:
            recovery_score += CRITICAL_PRIORITY_BONUS
        parts["recovery"] = round(recovery_score, 3)

        # Waiting time is measured from the later of creation and last selection:
        # a task that was just executed starts aging from scratch, which is what
        # prevents one task from starving the rest of the ready queue.
        created_at = float(task.created_at or 0.0)
        waiting_since = created_at
        last_selected = self._last_selected.get(task.task_id)
        if last_selected is not None:
            waiting_since = max(waiting_since, float(last_selected))
        age_s = max(0.0, now - waiting_since)
        aging = min(AGING_CAP, (age_s / 60.0) * max(0.0, float(self._config.aging_boost_per_minute)))
        parts["aging"] = round(aging, 3)

        starved = age_s >= max(0.0, float(self._config.starvation_after_s))
        parts["starvation"] = round(STARVATION_BONUS if starved else 0.0, 3)

        progress_score = max(0.0, min(1.0, float(task.progress))) * PROGRESS_BONUS
        parts["progress"] = round(progress_score, 3)

        attempt_penalty = int(task.attempts) * float(self._config.attempt_penalty)
        parts["attempt_penalty"] = round(-attempt_penalty, 3)

        interruption_penalty = int(task.interruptions) * float(self._config.interruption_penalty)
        parts["interruption_penalty"] = round(-interruption_penalty, 3)

        parts["failure_penalty"] = round(-self._failure_penalty(task), 3)

        recent = RECENT_SELECTION_PENALTY if task.task_id == self._last_selected_task_id else 0.0
        parts["recent_selection"] = round(-recent, 3)

        rotation = 0.0
        limit = max(0, int(self._config.owner_rotation_limit))
        if (
            limit
            and streak_owner
            and task.owner_agent.value == streak_owner
            and streak >= limit
        ):
            rotation = OWNER_ROTATION_PENALTY
        parts["owner_rotation"] = round(-rotation, 3)

        score = (
            type_score
            + priority_score
            + status_score
            + recovery_score
            + aging
            + (STARVATION_BONUS if starved else 0.0)
            + progress_score
            - attempt_penalty
            - interruption_penalty
            - self._failure_penalty(task)
            - recent
            - rotation
        )
        return score, parts
