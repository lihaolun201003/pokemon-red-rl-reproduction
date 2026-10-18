"""Progress / stagnation monitor for the Pokémon Red multi-agent system.

The monitor is the *single* place that decides "is the system still making
measurable progress?".  It is fed only by deterministic observation hooks that
the director calls after every agent step; it never presses buttons, never reads
RAM and never talks to the network, so it is cheap and fully testable.

Measured progress (never bare button counts)
--------------------------------------------
Progress is a *measured gain*, not activity.  Only these count:

* navigation — newly discovered map cells, verified ways/edges, new areas,
  new exits and a rising coverage ratio;
* story — new story clues, new NPC facts, completed tasks;
* pokémon — battle wins, successful catches and team changes;
* explicit task-success results (:meth:`ProgressMonitor.observe_task_result`).

Button presses, action batches and API calls are tracked for rate/resource
limits only; on their own they can **never** clear a stagnation signal.

Signals (:class:`~llm_agent.multi_agent.contracts.StagnationKind`)
------------------------------------------------------------------
* ``RESOURCE_EXHAUSTED`` — a configured hard limit (runtime seconds, actions or
  API cost) was reached; always reported, even during training grace.
* ``RECOVERY_LIMIT`` — the configured recovery cap :attr:`ProgressMonitorConfig.recovery_limit`
  was reached (safe pause instead of an endless rollback loop).
* ``GLOBAL_STAGNATION`` — no measurable global progress for
  :attr:`ProgressMonitorConfig.global_stagnation_seconds`.
* ``LOCAL_STAGNATION`` — one agent's consecutive observations/attempts without a
  measurable gain reached :attr:`ProgressMonitorConfig.local_stagnation_steps`.
* ``NO_PROGRESS`` — either :attr:`ProgressMonitorConfig.no_progress_action_limit`
  actions since the last gain, or the rolling window of
  :attr:`ProgressMonitorConfig.window` observations gained less than
  :attr:`ProgressMonitorConfig.min_progress_delta`.
* ``REPEATED_FAILURE`` — the same task failed (or the consecutive failure streak
  / failed battle streak reached) :attr:`ProgressMonitorConfig.repeated_failure_limit`.
* ``REPEATED_MESSAGE`` — the same message key was observed
  :attr:`ProgressMonitorConfig.repeated_message_limit` times or more.

Training/healing grace
----------------------
:meth:`ProgressMonitor.mark_activity` marks training/healing work.  While an
activity is within :attr:`ProgressMonitorConfig.training_grace_seconds`, every
*rollback-triggering* signal (local/global stagnation, no progress, repeated
failure) is suppressed so that legitimate grinding never causes a rollback.
Hard resource limits and loop warnings (repeated messages) are still reported.

All thresholds live in :class:`ProgressMonitorConfig`; the logic contains no
magic thresholds.  ``0`` (or a negative value) disables a threshold/limit,
matching the project convention used by ``emulator.max_actions``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Deque, Mapping

from llm_agent.multi_agent.contracts import (
    AgentRole,
    ProtocolError,
    RecoveryLevel,
    StagnationKind,
)

__all__ = [
    "ProgressMonitor",
    "ProgressMonitorConfig",
    "PROGRESS_SNAPSHOT_VERSION",
    "ROLLBACK_TRIGGER_SIGNALS",
    "SIGNAL_ORDER",
]

#: Schema version of :meth:`ProgressMonitor.snapshot`.
PROGRESS_SNAPSHOT_VERSION = 1

#: Signals ordered by severity (hard stops first).  :meth:`ProgressMonitor.check`
#: returns active signals in this order.
SIGNAL_ORDER: tuple[StagnationKind, ...] = (
    StagnationKind.RESOURCE_EXHAUSTED,
    StagnationKind.RECOVERY_LIMIT,
    StagnationKind.GLOBAL_STAGNATION,
    StagnationKind.NO_PROGRESS,
    StagnationKind.REPEATED_FAILURE,
    StagnationKind.LOCAL_STAGNATION,
    StagnationKind.REPEATED_MESSAGE,
)

#: Signals that justify a rollback and are therefore suppressed by the
#: training/healing grace window.
ROLLBACK_TRIGGER_SIGNALS: frozenset[StagnationKind] = frozenset(
    {
        StagnationKind.LOCAL_STAGNATION,
        StagnationKind.GLOBAL_STAGNATION,
        StagnationKind.NO_PROGRESS,
        StagnationKind.REPEATED_FAILURE,
    }
)

#: Candidate keys used to read a navigation ``status()`` mapping tolerantly.
_NAV_CELL_KEYS: tuple[str, ...] = (
    "cells_known",
    "cells_explored",
    "known_cells",
    "cells",
    "cell_count",
    "discovered_cells",
)
_NAV_COVERAGE_KEYS: tuple[str, ...] = ("coverage_ratio", "coverage")
_NAV_AREA_KEYS: tuple[str, ...] = ("areas_total", "areas", "area_count", "discovered_areas")
_NAV_WAY_KEYS: tuple[str, ...] = (
    "ways_total",
    "ways",
    "verified_ways",
    "links",
    "edges_total",
    "connections",
)
_NAV_EXIT_KEYS: tuple[str, ...] = (
    "exits_total",
    "exits",
    "exits_found",
    "new_exits",
    "map_exits",
)
_NAV_FRONTIER_KEYS: tuple[str, ...] = ("unknown_frontier", "frontier", "frontier_cells")
_NAV_LOCALIZATION_KEYS: tuple[str, ...] = (
    "localization_failures",
    "failed_localizations",
    "loc_failures",
    "localization_failure_count",
)

#: Bounded number of distinct message keys tracked for repetition detection.
_MAX_MESSAGE_KEYS = 1024
#: How many keys to drop when the message table overflows (insertion order).
_MESSAGE_KEYS_TRIM = 256


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class ProgressMonitorConfig:
    """Thresholds/limits for :class:`ProgressMonitor`.

    All values are configurable so that tests and the director can tune the
    monitor without touching the detection logic.  ``0`` disables the matching
    threshold; ``max_runtime_s`` / ``max_actions`` / ``max_api_cost_usd`` follow
    the project convention ``0 = unlimited``.
    """

    #: Per-agent observations/attempts without a measurable gain.
    local_stagnation_steps: int = 60
    #: No measurable global progress for this many seconds.
    global_stagnation_seconds: float = 900.0
    #: Same task fails / consecutive failures before ``REPEATED_FAILURE``.
    repeated_failure_limit: int = 4
    #: Same message key observed this many times before ``REPEATED_MESSAGE``.
    repeated_message_limit: int = 3
    #: Actions since the last measurable gain before ``NO_PROGRESS``.
    no_progress_action_limit: int = 150
    #: Training/healing grace window that suppresses rollback triggers.
    training_grace_seconds: float = 300.0
    #: Minimum measurable gain expected inside the rolling window.
    min_progress_delta: float = 1.0
    #: Number of observations kept in the rolling progress window.
    window: int = 240
    #: Hard runtime limit in seconds (0 = unlimited).
    max_runtime_s: float = 0.0
    #: Hard action limit (0 = unlimited).
    max_actions: int = 0
    #: Hard API cost limit in USD (0 = unlimited).
    max_api_cost_usd: float = 0.0
    #: Hard cap on recoveries before ``RECOVERY_LIMIT`` (0 = unlimited).
    recovery_limit: int = 6


# --------------------------------------------------------------------------- #
# Small coercion helpers
# --------------------------------------------------------------------------- #

def _int(value: Any, default: int = 0) -> int:
    """Tolerant non-negative-ish int coercion (used for counters only)."""
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _non_negative_int(value: Any, default: int = 0) -> int:
    return max(0, _int(value, default))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):  # NaN/inf
        return default
    return result


def _non_negative_float(value: Any, default: float = 0.0) -> float:
    return max(0.0, _float(value, default))


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present, non-``None`` value of ``keys`` (else ``None``)."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_agent(value: Any) -> AgentRole | None:
    """Best-effort :class:`AgentRole` coercion for action labels."""
    if value is None or isinstance(value, AgentRole):
        return value
    try:
        return AgentRole.coerce(value)
    except (ProtocolError, ValueError):
        return None


def _normalize_role_key(role: AgentRole | str) -> str:
    coerced = _as_agent(role)
    if coerced is None:
        raise ProtocolError(f"unknown agent role: {role!r}")
    return coerced.value


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #

class ProgressMonitor:
    """Deterministic progress/stagnation detector (see module docstring).

    Parameters
    ----------
    config:
        Thresholds/limits; defaults to :class:`ProgressMonitorConfig`.
    clock:
        Unix-time clock injected for deterministic tests (``time.time`` by
        default).  Every hook timestamps with this clock.

    Usage
    -----
    The director feeds observations, then calls :meth:`check` (or
    :meth:`consume_signals`) and hands the returned
    :class:`~llm_agent.multi_agent.contracts.StagnationKind` list to the
    :class:`~llm_agent.multi_agent.recovery.RecoveryManager`.  After any recovery
    :meth:`reset_after_recovery` restarts the *progress* clocks; hard resource
    counters and lifetime totals are never reset.
    """

    def __init__(
        self,
        config: ProgressMonitorConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or ProgressMonitorConfig()
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        window = max(1, _int(self.config.window, 1))
        self._window_size = window
        self._steps: Deque[float] = deque(maxlen=window)

        self._started_at: float = self._now()
        self._last_progress_at: float = self._started_at
        self._progress_total: float = 0.0
        self._progress_events: int = 0

        self._actions_total: int = 0
        self._action_batches: int = 0
        self._api_calls: int = 0
        self._api_cost_usd: float = 0.0
        self._last_api_cost_usd: float = 0.0
        self._api_observations: int = 0

        self._actions_since_progress: int = 0
        self._agent_attempts: dict[str, int] = {role.value: 0 for role in AgentRole}
        self._agent_gains: dict[str, int] = {role.value: 0 for role in AgentRole}
        self._agent_last_progress_at: dict[str, float] = {
            role.value: self._started_at for role in AgentRole
        }

        self._nav_last: dict[str, float] = {}
        self._navigation_totals: dict[str, float] = {
            "new_cells": 0.0,
            "new_areas": 0.0,
            "new_ways": 0.0,
            "new_exits": 0.0,
            "coverage_gains": 0.0,
            "observations": 0.0,
        }
        self._navigation_localization_failures: int = 0
        self._navigation_last_frontier: int = 0

        self._story_totals: dict[str, int] = {
            "discoveries": 0,
            "new_npcs": 0,
            "tasks_completed": 0,
            "repeated_dialogues": 0,
        }
        self._story_last_detail: str = ""

        self._pokemon_totals: dict[str, int] = {
            "battles": 0,
            "wins": 0,
            "losses": 0,
            "catches": 0,
            "team_changes": 0,
        }
        self._pokemon_last_detail: str = ""
        self._pokemon_consecutive_losses: int = 0

        self._task_failures: dict[str, int] = {}
        self._task_successes: int = 0
        self._failure_streak: int = 0
        self._failure_total: int = 0
        self._last_failure_detail: str = ""

        self._message_counts: dict[str, int] = {}
        self._repeated_message_total: int = 0
        self._last_message_key: str = ""

        self._activities: dict[str, float] = {}
        self._activity_events: int = 0
        self._last_activity_kind: str = ""
        self._last_activity_detail: str = ""

        self._recoveries_total: int = 0
        self._recoveries_by_level: dict[str, int] = {}
        self._last_recovery_at: float = 0.0

        self._checks: int = 0
        self._consumed: int = 0
        self._pending: list[StagnationKind] = []

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _now(self) -> float:
        return float(self._clock())

    def _limit(self, name: str) -> float:
        """Effective non-negative limit/threshold value from the config."""
        return max(0.0, _float(getattr(self.config, name, 0)))

    def _cfg_int(self, name: str) -> int:
        return max(0, _int(getattr(self.config, name, 0)))

    def _in_training_grace(self, now: float | None = None) -> bool:
        """True when a training/healing activity happened inside the grace window."""
        grace = self._limit("training_grace_seconds")
        if grace <= 0 or not self._activities:
            return False
        moment = self._now() if now is None else float(now)
        return any(moment - started <= grace for started in self._activities.values())

    def _grace_remaining(self, now: float | None = None) -> float:
        grace = self._limit("training_grace_seconds")
        if grace <= 0 or not self._activities:
            return 0.0
        moment = self._now() if now is None else float(now)
        remaining = [grace - (moment - started) for started in self._activities.values()]
        return max(0.0, max(remaining)) if remaining else 0.0

    def _push_step(self, delta: float) -> None:
        self._steps.append(max(0.0, float(delta)))

    def _record_progress(
        self,
        kind: str,
        delta: float,
        *,
        agent: AgentRole | None = None,
    ) -> None:
        """Record one measured progress gain and reset the stagnation clocks."""
        if delta <= 0:
            return
        now = self._now()
        self._progress_total += float(delta)
        self._progress_events += 1
        self._last_progress_at = now
        self._actions_since_progress = 0
        self._failure_streak = 0
        if agent is not None:
            key = agent.value
            self._agent_attempts[key] = 0
            self._agent_gains[key] = self._agent_gains.get(key, 0) + 1
            self._agent_last_progress_at[key] = now

    def _record_attempt(self, agent: AgentRole | None) -> None:
        if agent is not None:
            key = agent.value
            self._agent_attempts[key] = self._agent_attempts.get(key, 0) + 1

    def _record_failure(self, count: int = 1, *, detail: str = "") -> None:
        count = max(0, int(count))
        if count <= 0:
            return
        self._failure_streak += count
        self._failure_total += count
        if detail:
            self._last_failure_detail = str(detail)[:400]

    # ------------------------------------------------------------------ #
    # Observation hooks
    # ------------------------------------------------------------------ #
    def observe_navigation(self, status: Mapping[str, Any]) -> None:
        """Feed one :meth:`NavigationAgent.status` mapping.

        New cells / ways / areas / exits and a rising coverage ratio are counted
        as measured progress; growing localization failures accumulate failure
        evidence.  A missing/unchanged counter produces no progress (it still
        counts as a local attempt).
        """
        if not isinstance(status, Mapping):
            raise TypeError("navigation status must be a mapping")
        raw = dict(status)
        cells = _non_negative_int(_first_present(raw, _NAV_CELL_KEYS))
        areas = _non_negative_int(_first_present(raw, _NAV_AREA_KEYS))
        ways = _non_negative_int(_first_present(raw, _NAV_WAY_KEYS))
        exits = _non_negative_int(_first_present(raw, _NAV_EXIT_KEYS))
        frontier = _non_negative_int(_first_present(raw, _NAV_FRONTIER_KEYS))
        failures = _non_negative_int(_first_present(raw, _NAV_LOCALIZATION_KEYS))
        coverage = _non_negative_float(_first_present(raw, _NAV_COVERAGE_KEYS))

        previous = self._nav_last
        cells_delta = areas_delta = ways_delta = exits_delta = 0
        coverage_gain = 0.0
        if previous:
            cells_delta = max(0, cells - _int(previous.get("cells")))
            areas_delta = max(0, areas - _int(previous.get("areas")))
            ways_delta = max(0, ways - _int(previous.get("ways")))
            exits_delta = max(0, exits - _int(previous.get("exits")))
            previous_coverage = _float(previous.get("coverage"))
            if coverage > previous_coverage + 1e-9:
                coverage_gain = 1.0

        failure_delta = 0
        if previous:
            failure_delta = max(0, failures - _int(previous.get("failures")))

        self._nav_last = {
            "cells": cells,
            "areas": areas,
            "ways": ways,
            "exits": exits,
            "frontier": frontier,
            "coverage": coverage,
            "failures": failures,
        }
        self._navigation_totals["observations"] += 1.0
        self._navigation_totals["new_cells"] += float(cells_delta)
        self._navigation_totals["new_areas"] += float(areas_delta)
        self._navigation_totals["new_ways"] += float(ways_delta)
        self._navigation_totals["new_exits"] += float(exits_delta)
        self._navigation_totals["coverage_gains"] += coverage_gain
        self._navigation_localization_failures = failures
        self._navigation_last_frontier = frontier

        delta = float(cells_delta + areas_delta + ways_delta + exits_delta) + coverage_gain
        self._push_step(delta)
        if delta > 0:
            self._record_progress("navigation", delta, agent=AgentRole.NAVIGATION)
        else:
            self._record_attempt(AgentRole.NAVIGATION)
            if failure_delta > 0:
                self._record_failure(
                    failure_delta,
                    detail=f"navigation localization failures (+{failure_delta})",
                )

    def observe_story(
        self,
        *,
        discoveries: int = 0,
        new_npcs: int = 0,
        tasks_completed: int = 0,
        repeated_dialogues: int = 0,
        detail: str = "",
    ) -> None:
        """Feed one story-agent step observation.

        New clues / NPC facts / completed tasks are measured progress; repeated
        dialogue is recorded as evidence of stagnation but is never progress.
        """
        discoveries = _non_negative_int(discoveries)
        new_npcs = _non_negative_int(new_npcs)
        tasks_completed = _non_negative_int(tasks_completed)
        repeated_dialogues = _non_negative_int(repeated_dialogues)
        if detail:
            self._story_last_detail = str(detail)[:400]

        self._story_totals["discoveries"] += discoveries
        self._story_totals["new_npcs"] += new_npcs
        self._story_totals["tasks_completed"] += tasks_completed
        self._story_totals["repeated_dialogues"] += repeated_dialogues

        delta = float(discoveries + new_npcs + tasks_completed)
        self._push_step(delta)
        if delta > 0:
            self._record_progress("story", delta, agent=AgentRole.STORY)
        else:
            self._record_attempt(AgentRole.STORY)

    def observe_pokemon(
        self,
        *,
        battles: int = 0,
        wins: int = 0,
        losses: int = 0,
        catches: int = 0,
        team_changes: int = 0,
        details: str = "",
    ) -> None:
        """Feed one Pokémon-agent step observation.

        Battle wins, catches and team changes are measured progress; losses count
        as failure evidence (consecutive losses feed ``REPEATED_FAILURE``).
        """
        battles = _non_negative_int(battles)
        wins = _non_negative_int(wins)
        losses = _non_negative_int(losses)
        catches = _non_negative_int(catches)
        team_changes = _non_negative_int(team_changes)
        if details:
            self._pokemon_last_detail = str(details)[:400]

        self._pokemon_totals["battles"] += battles
        self._pokemon_totals["wins"] += wins
        self._pokemon_totals["losses"] += losses
        self._pokemon_totals["catches"] += catches
        self._pokemon_totals["team_changes"] += team_changes

        if losses > 0:
            self._pokemon_consecutive_losses += losses
            self._record_failure(losses, detail=f"battle losses (+{losses})")
        if wins > 0 or catches > 0 or team_changes > 0:
            self._pokemon_consecutive_losses = 0

        delta = float(wins + catches + team_changes)
        self._push_step(delta)
        if delta > 0:
            self._record_progress("pokemon", delta, agent=AgentRole.POKEMON)
        else:
            self._record_attempt(AgentRole.POKEMON)

    def observe_actions(self, count: int = 1, *, label: str = "") -> None:
        """Feed executed actions (never a progress signal by itself).

        ``label`` may name an agent role (``"navigation"`` …) to attribute the
        attempts to that agent.  Use either this hook **or** the dedicated
        per-agent observation hooks for the same step to avoid double counting.
        """
        count = _non_negative_int(count)
        if count <= 0:
            return
        self._action_batches += 1
        self._actions_total += count
        self._actions_since_progress += count
        role = _as_agent(label) if label else None
        if role is not None:
            key = role.value
            self._agent_attempts[key] = self._agent_attempts.get(key, 0) + count

    def observe_api(self, *, cost_usd: float = 0.0, calls: int = 0) -> None:
        """Feed API spend/call counters (only used for hard resource limits)."""
        cost = _non_negative_float(cost_usd)
        calls = _non_negative_int(calls)
        self._api_observations += 1
        self._api_cost_usd += cost
        self._api_calls += calls
        self._last_api_cost_usd = cost

    def observe_message(self, message_key: str) -> None:
        """Feed one produced/consumed message key for repetition detection."""
        key = str(message_key or "").strip()
        if not key:
            return
        if key not in self._message_counts and len(self._message_counts) >= _MAX_MESSAGE_KEYS:
            for stale in list(self._message_counts)[:_MESSAGE_KEYS_TRIM]:
                self._message_counts.pop(stale, None)
        self._message_counts[key] = self._message_counts.get(key, 0) + 1
        self._last_message_key = key
        limit = self._cfg_int("repeated_message_limit")
        if limit > 0 and self._message_counts[key] >= limit:
            self._repeated_message_total += 1

    def observe_task_result(self, *, task_id: str, ok: bool, detail: str = "") -> None:
        """Feed one task attempt result.

        ``ok=True`` clears that task's failure count and the consecutive-failure
        streak (but is not itself a measured progress gain — task *completions*
        arrive through :meth:`observe_story` / navigation gains).
        """
        key = str(task_id or "").strip() or "_unknown"
        if ok:
            self._task_successes += 1
            self._task_failures[key] = 0
            self._failure_streak = 0
            return
        self._task_failures[key] = self._task_failures.get(key, 0) + 1
        self._record_failure(1, detail=f"task {key}: {str(detail or '').strip()}"[:400])

    def mark_activity(self, kind: str, detail: str = "") -> None:
        """Mark training/healing activity to open a rollback-suppressing grace window."""
        key = str(kind or "").strip().lower() or "activity"
        self._activities[key] = self._now()
        self._activity_events += 1
        self._last_activity_kind = key
        if detail:
            self._last_activity_detail = str(detail)[:400]

    # ------------------------------------------------------------------ #
    # Signal evaluation
    # ------------------------------------------------------------------ #
    def limits_exceeded(self) -> StagnationKind | None:
        """Return the first exceeded hard limit as a signal (else ``None``)."""
        now = self._now()
        max_runtime = self._limit("max_runtime_s")
        if max_runtime > 0 and now - self._started_at >= max_runtime:
            return StagnationKind.RESOURCE_EXHAUSTED
        max_actions = self._cfg_int("max_actions")
        if max_actions > 0 and self._actions_total >= max_actions:
            return StagnationKind.RESOURCE_EXHAUSTED
        max_cost = self._limit("max_api_cost_usd")
        if max_cost > 0 and self._api_cost_usd >= max_cost:
            return StagnationKind.RESOURCE_EXHAUSTED
        recovery_limit = self._cfg_int("recovery_limit")
        if recovery_limit > 0 and self._recoveries_total >= recovery_limit:
            return StagnationKind.RECOVERY_LIMIT
        return None

    def _global_stagnation(self, now: float) -> bool:
        threshold = self._limit("global_stagnation_seconds")
        if threshold <= 0:
            return False
        return (now - self._last_progress_at) >= threshold

    def _local_stagnation(self) -> bool:
        threshold = self._cfg_int("local_stagnation_steps")
        if threshold <= 0:
            return False
        return max(self._agent_attempts.values(), default=0) >= threshold

    def _no_progress(self) -> bool:
        action_limit = self._cfg_int("no_progress_action_limit")
        if action_limit > 0 and self._actions_since_progress >= action_limit:
            return True
        if self._window_size > 1 and len(self._steps) >= self._window_size:
            if sum(self._steps) < self._limit("min_progress_delta"):
                return True
        return False

    def _repeated_failure(self) -> bool:
        limit = self._cfg_int("repeated_failure_limit")
        if limit <= 0:
            return False
        worst_task = max(self._task_failures.values(), default=0)
        return max(worst_task, self._failure_streak, self._pokemon_consecutive_losses) >= limit

    def _repeated_message(self) -> bool:
        limit = self._cfg_int("repeated_message_limit")
        if limit <= 0 or not self._message_counts:
            return False
        return max(self._message_counts.values()) >= limit

    def _compute_signals(self) -> list[StagnationKind]:
        """Pure signal evaluation (no latch side effects)."""
        now = self._now()
        active: set[StagnationKind] = set()
        hard = self.limits_exceeded()
        if hard is not None:
            active.add(hard)
        in_grace = self._in_training_grace(now)
        if not in_grace:
            if self._global_stagnation(now):
                active.add(StagnationKind.GLOBAL_STAGNATION)
            if self._local_stagnation():
                active.add(StagnationKind.LOCAL_STAGNATION)
            if self._no_progress():
                active.add(StagnationKind.NO_PROGRESS)
            if self._repeated_failure():
                active.add(StagnationKind.REPEATED_FAILURE)
        if self._repeated_message():
            active.add(StagnationKind.REPEATED_MESSAGE)
        return [kind for kind in SIGNAL_ORDER if kind in active]

    def check(self) -> list[StagnationKind]:
        """Return the currently active signals, deduped and ordered by severity.

        Also refreshes the pending latch read by :meth:`consume_signals`; it does
        not change any counter, so repeated calls with unchanged state and clock
        return equal lists.
        """
        self._checks += 1
        signals = self._compute_signals()
        self._pending = list(signals)
        return list(signals)

    def consume_signals(self) -> list[StagnationKind]:
        """Return :meth:`check` output and clear the pending latch."""
        signals = self.check()
        self._consumed += 1
        self._pending = []
        return signals

    def pending_signals(self) -> list[StagnationKind]:
        """Signals latched by the last :meth:`check` and not yet consumed."""
        return list(self._pending)

    # ------------------------------------------------------------------ #
    # Recovery resets
    # ------------------------------------------------------------------ #
    def reset_agent(self, agent: AgentRole | str) -> None:
        """Clear one agent's local stagnation counters (after a local hint)."""
        key = _normalize_role_key(agent)
        self._agent_attempts[key] = 0

    def reset_after_recovery(self, *, level: RecoveryLevel | str = RecoveryLevel.L1_LOCAL) -> None:
        """Restart the *progress* clocks after a recovery intervention.

        Episodic stagnation counters are cleared and the no-progress clock is
        restarted; lifetime totals, action/API counters and hard limits are never
        reset, so a rollback can never hide spent money, actions or attempts.
        """
        resolved = RecoveryLevel.coerce(level)
        if resolved is RecoveryLevel.NONE:
            # "none" means no recovery was executed (e.g. a safe pause): it must
            # not consume the recovery cap, so this call is deliberately a no-op.
            return
        now = self._now()
        self._recoveries_total += 1
        self._recoveries_by_level[resolved.value] = (
            self._recoveries_by_level.get(resolved.value, 0) + 1
        )
        self._last_recovery_at = now
        self._agent_attempts = {role.value: 0 for role in AgentRole}
        self._failure_streak = 0
        self._pokemon_consecutive_losses = 0
        self._task_failures = {key: 0 for key in self._task_failures}
        self._message_counts = {}
        self._repeated_message_total = 0
        self._actions_since_progress = 0
        self._steps.clear()
        self._last_progress_at = now
        self._pending = []

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        """JSON-safe dashboard counters (for the viewer/report/director prompt)."""
        now = self._now()
        return {
            "checked_at": now,
            "elapsed_s": max(0.0, now - self._started_at),
            "last_progress_at": self._last_progress_at,
            "seconds_since_progress": max(0.0, now - self._last_progress_at),
            "progress_total": self._progress_total,
            "progress_events": self._progress_events,
            "actions_total": self._actions_total,
            "action_batches": self._action_batches,
            "actions_since_progress": self._actions_since_progress,
            "api_calls": self._api_calls,
            "api_cost_usd": round(self._api_cost_usd, 8),
            "last_api_cost_usd": round(self._last_api_cost_usd, 8),
            "in_training_grace": self._in_training_grace(now),
            "grace_remaining_s": round(self._grace_remaining(now), 3),
            "last_activity": {
                "kind": self._last_activity_kind,
                "detail": self._last_activity_detail,
                "events": self._activity_events,
                "kinds": {kind: at for kind, at in sorted(self._activities.items())},
            },
            "agents": {
                role.value: {
                    "attempts_without_gain": self._agent_attempts.get(role.value, 0),
                    "gains": self._agent_gains.get(role.value, 0),
                    "seconds_since_progress": max(
                        0.0, now - self._agent_last_progress_at.get(role.value, self._started_at)
                    ),
                }
                for role in AgentRole
            },
            "navigation": {
                "totals": {key: value for key, value in self._navigation_totals.items()},
                "last_status": dict(self._nav_last),
                "localization_failures": self._navigation_localization_failures,
                "unknown_frontier": self._navigation_last_frontier,
            },
            "story": {**self._story_totals, "last_detail": self._story_last_detail},
            "pokemon": {
                **self._pokemon_totals,
                "consecutive_losses": self._pokemon_consecutive_losses,
                "last_detail": self._pokemon_last_detail,
            },
            "failures": {
                "streak": self._failure_streak,
                "total": self._failure_total,
                "task_failures": dict(sorted(self._task_failures.items())),
                "task_successes": self._task_successes,
                "last_detail": self._last_failure_detail,
            },
            "messages": {
                "distinct_keys": len(self._message_counts),
                "repeated_total": self._repeated_message_total,
                "last_key": self._last_message_key,
                "worst_repeats": max(self._message_counts.values(), default=0),
            },
            "recoveries": {
                "total": self._recoveries_total,
                "by_level": dict(sorted(self._recoveries_by_level.items())),
                "last_at": self._last_recovery_at,
            },
            "window": {
                "size": self._window_size,
                "filled": len(self._steps),
                "sum": sum(self._steps),
                "min_progress_delta": self._limit("min_progress_delta"),
            },
            "limits": {
                "max_runtime_s": self._limit("max_runtime_s"),
                "max_actions": self._cfg_int("max_actions"),
                "max_api_cost_usd": self._limit("max_api_cost_usd"),
                "recovery_limit": self._cfg_int("recovery_limit"),
                "limits_exceeded": (
                    None if self.limits_exceeded() is None else self.limits_exceeded().value
                ),
            },
            "signals": [kind.value for kind in self._compute_signals()],
            "checks": self._checks,
            "consumed": self._consumed,
            "config": asdict(self.config),
        }

    # ------------------------------------------------------------------ #
    # Snapshot / restore
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable monitor state for checkpoint manifests."""
        return {
            "version": PROGRESS_SNAPSHOT_VERSION,
            "started_at": self._started_at,
            "last_progress_at": self._last_progress_at,
            "progress_total": self._progress_total,
            "progress_events": self._progress_events,
            "actions_total": self._actions_total,
            "action_batches": self._action_batches,
            "api_calls": self._api_calls,
            "api_cost_usd": self._api_cost_usd,
            "last_api_cost_usd": self._last_api_cost_usd,
            "api_observations": self._api_observations,
            "actions_since_progress": self._actions_since_progress,
            "agent_attempts": dict(self._agent_attempts),
            "agent_gains": dict(self._agent_gains),
            "agent_last_progress_at": dict(self._agent_last_progress_at),
            "nav_last": dict(self._nav_last),
            "navigation_totals": dict(self._navigation_totals),
            "navigation_localization_failures": self._navigation_localization_failures,
            "navigation_last_frontier": self._navigation_last_frontier,
            "story_totals": dict(self._story_totals),
            "story_last_detail": self._story_last_detail,
            "pokemon_totals": dict(self._pokemon_totals),
            "pokemon_last_detail": self._pokemon_last_detail,
            "pokemon_consecutive_losses": self._pokemon_consecutive_losses,
            "task_failures": dict(self._task_failures),
            "task_successes": self._task_successes,
            "failure_streak": self._failure_streak,
            "failure_total": self._failure_total,
            "last_failure_detail": self._last_failure_detail,
            "message_counts": dict(self._message_counts),
            "repeated_message_total": self._repeated_message_total,
            "last_message_key": self._last_message_key,
            "activities": dict(self._activities),
            "activity_events": self._activity_events,
            "last_activity_kind": self._last_activity_kind,
            "last_activity_detail": self._last_activity_detail,
            "recoveries_total": self._recoveries_total,
            "recoveries_by_level": dict(self._recoveries_by_level),
            "last_recovery_at": self._last_recovery_at,
            "steps": [float(value) for value in self._steps],
            "checks": self._checks,
            "consumed": self._consumed,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a :meth:`snapshot`; junk/missing fields are ignored, never raises."""
        if not isinstance(snapshot, Mapping):
            return
        data = snapshot

        def number(key: str, default: float = 0.0) -> float:
            return _float(data.get(key), default)

        def count(key: str, default: int = 0) -> int:
            return _non_negative_int(data.get(key), default)

        def mapping(key: str) -> dict[str, Any]:
            value = data.get(key)
            return dict(value) if isinstance(value, Mapping) else {}

        self._started_at = number("started_at", self._now())
        self._last_progress_at = number("last_progress_at", self._started_at)
        self._progress_total = max(0.0, number("progress_total"))
        self._progress_events = count("progress_events")
        self._actions_total = count("actions_total")
        self._action_batches = count("action_batches")
        self._api_calls = count("api_calls")
        self._api_cost_usd = max(0.0, number("api_cost_usd"))
        self._last_api_cost_usd = max(0.0, number("last_api_cost_usd"))
        self._api_observations = count("api_observations")
        self._actions_since_progress = count("actions_since_progress")

        agent_attempts = mapping("agent_attempts")
        agent_gains = mapping("agent_gains")
        for role in AgentRole:
            key = role.value
            self._agent_attempts[key] = _non_negative_int(agent_attempts.get(key))
            self._agent_gains[key] = _non_negative_int(agent_gains.get(key))
            self._agent_last_progress_at[key] = _float(
                mapping("agent_last_progress_at").get(key), self._started_at
            )

        self._nav_last = {
            str(key): _float(value) for key, value in mapping("nav_last").items()
        }
        self._navigation_totals = {
            str(key): max(0.0, _float(value))
            for key, value in mapping("navigation_totals").items()
        } or self._default_navigation_totals()
        self._navigation_localization_failures = count("navigation_localization_failures")
        self._navigation_last_frontier = count("navigation_last_frontier")

        story = mapping("story_totals")
        self._story_totals = {
            key: _non_negative_int(story.get(key)) for key in self._story_totals
        }
        self._story_last_detail = str(data.get("story_last_detail") or "")[:400]
        pokemon = mapping("pokemon_totals")
        self._pokemon_totals = {
            key: _non_negative_int(pokemon.get(key)) for key in self._pokemon_totals
        }
        self._pokemon_last_detail = str(data.get("pokemon_last_detail") or "")[:400]
        self._pokemon_consecutive_losses = count("pokemon_consecutive_losses")

        self._task_failures = {
            str(key): _non_negative_int(value)
            for key, value in mapping("task_failures").items()
        }
        self._task_successes = count("task_successes")
        self._failure_streak = count("failure_streak")
        self._failure_total = count("failure_total")
        self._last_failure_detail = str(data.get("last_failure_detail") or "")[:400]

        self._message_counts = {
            str(key): _non_negative_int(value)
            for key, value in mapping("message_counts").items()
        }
        self._repeated_message_total = count("repeated_message_total")
        self._last_message_key = str(data.get("last_message_key") or "")[:400]

        self._activities = {
            str(key): _float(value)
            for key, value in mapping("activities").items()
            if _float(value) >= 0.0
        }
        self._activity_events = count("activity_events")
        self._last_activity_kind = str(data.get("last_activity_kind") or "")
        self._last_activity_detail = str(data.get("last_activity_detail") or "")[:400]

        self._recoveries_total = count("recoveries_total")
        self._recoveries_by_level = {
            str(key): _non_negative_int(value)
            for key, value in mapping("recoveries_by_level").items()
        }
        self._last_recovery_at = number("last_recovery_at")

        raw_steps = data.get("steps")
        self._steps.clear()
        if isinstance(raw_steps, (list, tuple)):
            for item in list(raw_steps)[-self._window_size :]:
                self._steps.append(max(0.0, _float(item)))
        self._checks = count("checks")
        self._consumed = count("consumed")
        self._pending = []

    def _default_navigation_totals(self) -> dict[str, float]:
        return {
            "new_cells": 0.0,
            "new_areas": 0.0,
            "new_ways": 0.0,
            "new_exits": 0.0,
            "coverage_gains": 0.0,
            "observations": 0.0,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ProgressMonitor(progress={self._progress_total:.1f}, "
            f"actions={self._actions_total}, recoveries={self._recoveries_total}, "
            f"signals={[kind.value for kind in self._compute_signals()]})"
        )
