"""Self-correction / recovery manager for the Pokémon Red multi-agent system.

The :class:`RecoveryManager` turns the
:class:`~llm_agent.multi_agent.progress_monitor.ProgressMonitor` signals into
*bounded* corrective actions.  It owns no game state, no emulator and no network
access: everything that touches the game is injected through
:class:`RecoveryHooks` so the director can wire the real checkpoint store,
navigation re-localization and the DeepSeek director review.

Escalation ladder
-----------------
1. **L1 — local strategy change** (cheap, always tried first): a concrete hint is
   produced for the responsible agent (*navigation* → alternative route,
   *story* → different clue/NPC, *pokémon* → different battle tactic) and handed
   to the injected ``agent_hint`` hook.  No game state is touched.
2. **L2 — global review** (rare, rate limited by
   :attr:`RecoveryConfig.max_l2_reviews`): the injected ``director_review``
   callback (the DeepSeek director) re-plans at the global level.
3. **L3 — checkpoint rollback** (last resort): a checkpoint of the current state
   is saved first (``save_checkpoint("pre_rollback:<reason>")``), then the latest
   state is restored, and the game is re-localized.  Rollbacks are capped by
   :attr:`RecoveryConfig.max_rollbacks_per_branch` /
   :attr:`RecoveryConfig.max_rollbacks_total` and a
   :attr:`RecoveryConfig.rollback_cooldown_s` cooldown.  Reason + evidence are
   recorded **before** the state is restored so a crash can never lose the audit
   trail.

Hard signals
------------
``RESOURCE_EXHAUSTED`` and ``RECOVERY_LIMIT`` are *hard stops*: no rollback is
attempted (that could not create budget) — the manager reports a safe pause and
the caller stops the run.  A circuit breaker
(:attr:`RecoveryConfig.max_consecutive_failures`) also forces a safe pause
instead of an endless recovery loop.

Invariants
----------
* **No infinite loops** — each :meth:`RecoveryManager.handle` call performs at
  most one action; the per-kind ladder is bounded (L1 → L2 → L3 → safe pause).
* **Budget is never reset** — API cost/attempt/failure counters live outside the
  manager (they are owned by the budget tracker and the monitor); the manager
  never zeroes or restores them, it only records.  A rollback restores *game
  state*, never the money already spent or the failure history.
* **Hook failures never crash the manager** — every injected callback is wrapped;
  errors are reported in the structured result.

All limits live in :class:`RecoveryConfig`; ``0`` disables a limit, matching the
project convention used by ``emulator.max_actions`` / ``budget``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from llm_agent.multi_agent.contracts import (
    AgentRole,
    EventType,
    ProtocolError,
    RecoveryLevel,
    StagnationKind,
    new_branch_id,
)

__all__ = [
    "RecoveryConfig",
    "RecoveryHooks",
    "RecoveryManager",
    "HARD_SIGNALS",
    "SIGNAL_SEVERITY",
]

#: Signals that must never trigger a rollback (they need a safe pause instead).
HARD_SIGNALS: frozenset[StagnationKind] = frozenset(
    {StagnationKind.RESOURCE_EXHAUSTED, StagnationKind.RECOVERY_LIMIT}
)

#: Severity ordering used to pick the primary signal of one recovery round.
SIGNAL_SEVERITY: dict[StagnationKind, int] = {
    StagnationKind.RESOURCE_EXHAUSTED: 100,
    StagnationKind.RECOVERY_LIMIT: 90,
    StagnationKind.GLOBAL_STAGNATION: 80,
    StagnationKind.NO_PROGRESS: 70,
    StagnationKind.REPEATED_FAILURE: 60,
    StagnationKind.LOCAL_STAGNATION: 50,
    StagnationKind.REPEATED_MESSAGE: 40,
}

#: Fixed ladder length (L1 → L2 → L3); after that the manager safe-pauses.
_LADDER: tuple[RecoveryLevel, ...] = (
    RecoveryLevel.L1_LOCAL,
    RecoveryLevel.L2_REVIEW,
    RecoveryLevel.L3_ROLLBACK,
)

#: Generic L1 hints per signal (agent-specific variations live in _AGENT_L1_HINTS).
_L1_HINTS: dict[StagnationKind, str] = {
    StagnationKind.LOCAL_STAGNATION: (
        "Local strategy is stagnating: stop repeating the current plan, re-read the "
        "last few observations and choose a clearly different next action."
    ),
    StagnationKind.GLOBAL_STAGNATION: (
        "No measurable global progress: re-plan at the director level — change the "
        "active objective or area and split the goal into smaller verifiable steps."
    ),
    StagnationKind.NO_PROGRESS: (
        "No measurable gain recently: switch to a different objective/area and pick "
        "actions that produce verifiable results (new location, dialogue, battle win)."
    ),
    StagnationKind.REPEATED_FAILURE: (
        "The same step keeps failing: do not retry it the same way; record the "
        "failure evidence, try a different approach, or defer the goal to the director."
    ),
    StagnationKind.REPEATED_MESSAGE: (
        "The same message keeps being produced: break the loop — change the action "
        "sequence, close the current menu/dialogue with B and re-observe the game."
    ),
}

#: Agent-specific L1 hints (navigation alternative route, story other clue, …).
_AGENT_L1_HINTS: dict[AgentRole, dict[StagnationKind, str]] = {
    AgentRole.NAVIGATION: {
        StagnationKind.LOCAL_STAGNATION: (
            "Navigation is repeating the same route: pick a different unexplored "
            "direction, avoid replaying the last path, and re-localize on the current "
            "screen before moving again."
        ),
        StagnationKind.REPEATED_FAILURE: (
            "The navigation target keeps failing: treat the way as blocked, mark it as "
            "a dead end, re-localize and choose a different route or exit."
        ),
    },
    AgentRole.STORY: {
        StagnationKind.LOCAL_STAGNATION: (
            "The story line is stuck on repeated dialogue: look for a different clue — "
            "talk to a different NPC, inspect an unexamined object, or explore elsewhere."
        ),
        StagnationKind.REPEATED_FAILURE: (
            "This dialogue interaction keeps failing: end the dialogue, try a different "
            "NPC or object, and record what was learned before moving on."
        ),
    },
    AgentRole.POKEMON: {
        StagnationKind.LOCAL_STAGNATION: (
            "The current battle tactic is not working: change it — use a different move, "
            "switch the active team member, use an item, or retreat to heal/train first."
        ),
        StagnationKind.REPEATED_FAILURE: (
            "The same battle approach keeps failing: change tactic or disengage, heal, "
            "and record the visible lesson (damage/health/outcome) before retrying."
        ),
    },
}

#: Default responsible agent per signal when the context does not name one.
_DEFAULT_AGENT: dict[StagnationKind, AgentRole] = {
    StagnationKind.LOCAL_STAGNATION: AgentRole.NAVIGATION,
    StagnationKind.GLOBAL_STAGNATION: AgentRole.DIRECTOR,
    StagnationKind.NO_PROGRESS: AgentRole.DIRECTOR,
    StagnationKind.REPEATED_FAILURE: AgentRole.DIRECTOR,
    StagnationKind.REPEATED_MESSAGE: AgentRole.DIRECTOR,
}

#: Bounded history sizes (audit only; never used for decisions).
_MAX_ATTEMPT_HISTORY = 200
_MAX_ROLLBACK_HISTORY = 100

#: Human-readable safe-pause hints for hard stops.
_HARD_HINTS: dict[StagnationKind, str] = {
    StagnationKind.RESOURCE_EXHAUSTED: (
        "A hard resource limit was reached: pause the run safely (save a checkpoint "
        "and report); rolling back cannot restore spent runtime/actions/API budget."
    ),
    StagnationKind.RECOVERY_LIMIT: (
        "The recovery cap was reached: stop self-correcting and pause the run before "
        "anything irreversible happens; report the failure evidence for a human review."
    ),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _now_default() -> float:
    return time.time()


def _json_safe(value: Any) -> Any:
    """Best-effort JSON-safe copy (never raises for ordinary records)."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_kind(value: StagnationKind | str) -> StagnationKind:
    if isinstance(value, StagnationKind):
        return value
    try:
        return StagnationKind(str(value or "").strip().lower())
    except ValueError as exc:
        raise ProtocolError(f"unknown stagnation kind: {value!r}") from exc


def _coerce_agent(value: Any) -> AgentRole | None:
    if value is None or isinstance(value, AgentRole):
        return value
    try:
        return AgentRole.coerce(value)
    except (ProtocolError, ValueError):
        return None


def _safe_call(
    label: str,
    func: Callable[..., Any] | None,
    *args: Any,
) -> tuple[bool, Any, str]:
    """Call an injected hook, catching every ordinary failure.

    Returns ``(ok, value, error)``; ``ok`` is ``False`` for a missing hook or a
    raised exception, and ``error`` is a compact ``"<label>: <error>"`` string.
    """
    if func is None:
        return False, None, f"{label}: hook is not configured"
    try:
        return True, func(*args), ""
    except Exception as exc:  # noqa: BLE001 - injected callbacks must never kill the manager
        return False, None, f"{label}: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Config / hooks
# --------------------------------------------------------------------------- #

@dataclass
class RecoveryConfig:
    """Caps and cooldowns for :class:`RecoveryManager` (``0`` disables a cap)."""

    #: Maximum rollbacks allowed on one branch lineage.
    max_rollbacks_per_branch: int = 2
    #: Maximum rollbacks during the whole run.
    max_rollbacks_total: int = 4
    #: Maximum L2 global director reviews (rate limit).
    max_l2_reviews: int = 8
    #: Minimum seconds between two rollbacks.
    rollback_cooldown_s: float = 120.0
    #: Consecutive failed recovery attempts before the circuit breaker opens.
    max_consecutive_failures: int = 5


@dataclass
class RecoveryHooks:
    """Callbacks injected by the director/session (all optional).

    * ``save_checkpoint(reason) -> manifest`` — snapshot the current state; used
      for the ``pre_rollback:<reason>`` audit checkpoint.
    * ``restore_checkpoint(checkpoint_id, branch_id) -> manifest`` — restore the
      game state (the caller applies the state bytes to the emulator).
    * ``relocalize() -> dict`` — re-observe the screen and re-localize navigation
      after a restoration.
    * ``director_review(payload) -> dict`` — rare L2 global re-plan (DeepSeek).
    * ``notify(event_type, payload)`` — publish recovery/rollback events.
    * ``agent_hint(agent, level, hint)`` — deliver an L1 local strategy hint.
    """

    save_checkpoint: Callable[[str], dict] | None = None
    restore_checkpoint: Callable[[str, str], dict] | None = None
    relocalize: Callable[[], dict] | None = None
    director_review: Callable[[dict], dict] | None = None
    notify: Callable[[str, dict], None] | None = None
    agent_hint: Callable[[AgentRole, str, str], None] | None = None


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #

class RecoveryManager:
    """Bounded three-level self-correction (see module docstring).

    Parameters
    ----------
    config:
        Caps/cooldowns; defaults to :class:`RecoveryConfig`.
    hooks:
        Injected callbacks; defaults to an empty :class:`RecoveryHooks` (the
        manager then only produces hints/results without touching the game).
    clock:
        Unix-time clock injected for deterministic tests.
    """

    def __init__(
        self,
        config: RecoveryConfig | None = None,
        *,
        hooks: RecoveryHooks | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or RecoveryConfig()
        self.hooks = hooks or RecoveryHooks()
        self._clock: Callable[[], float] = clock if clock is not None else time.time

        self._started_at: float = self._now()
        self._current_branch_id: str = ""
        #: branch id -> lineage root id (rollbacks keep the lineage of the branch
        #: they restore, so the per-branch cap cannot be dodged by new branch ids).
        self._lineage: dict[str, str] = {}

        self._total_attempts: int = 0
        self._attempt_counts: dict[str, int] = {}
        self._consecutive_failures: int = 0
        self._attempt_history: list[dict[str, Any]] = []

        self._kind_attempts: dict[str, int] = {}

        self._l2_reviews: int = 0
        self._last_l2_at: float = 0.0

        self._rollbacks_total: int = 0
        self._rollbacks_by_lineage: dict[str, int] = {}
        self._last_rollback_at: float = 0.0
        self._rollback_history: list[dict[str, Any]] = []

        self._last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Basic helpers
    # ------------------------------------------------------------------ #
    def _now(self) -> float:
        return float(self._clock())

    def _limit(self, name: str) -> float:
        try:
            return max(0.0, float(getattr(self.config, name)))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0

    def _cfg_int(self, name: str) -> int:
        try:
            return max(0, int(getattr(self.config, name)))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0

    @property
    def current_branch_id(self) -> str:
        """Branch the manager believes it is working on (informational)."""
        return self._current_branch_id

    def set_current_branch(self, branch_id: str, *, parent_branch_id: str = "") -> None:
        """Record the working branch (called by the director after a checkpoint)."""
        branch = str(branch_id or "").strip()
        if not branch:
            return
        parent = str(parent_branch_id or "").strip()
        self._current_branch_id = branch
        if parent:
            self._lineage[branch] = self._lineage.get(parent, parent)
        elif branch not in self._lineage:
            self._lineage[branch] = branch

    def _lineage_root(self, branch_id: str) -> str:
        branch = str(branch_id or "").strip() or "_root"
        return self._lineage.get(branch, branch)

    def _emit(self, event_type: EventType | str, payload: Mapping[str, Any]) -> str:
        """Publish one event through the notify hook (never raises)."""
        event = event_type.value if isinstance(event_type, EventType) else str(event_type)
        ok, _value, error = _safe_call("notify", self.hooks.notify, event, dict(payload))
        return "" if ok else error

    def _record_attempt(
        self,
        level: RecoveryLevel,
        *,
        ok: bool,
        detail: str = "",
    ) -> dict[str, Any]:
        record = {
            "level": level.value,
            "ok": bool(ok),
            "detail": str(detail or "")[:400],
            "at": self._now(),
            "consecutive_failures": self._consecutive_failures,
        }
        self._attempt_history.append(record)
        if len(self._attempt_history) > _MAX_ATTEMPT_HISTORY:
            del self._attempt_history[: len(self._attempt_history) - _MAX_ATTEMPT_HISTORY]
        return record

    def _append_rollback(self, record: dict[str, Any]) -> None:
        self._rollback_history.append(record)
        if len(self._rollback_history) > _MAX_ROLLBACK_HISTORY:
            del self._rollback_history[: len(self._rollback_history) - _MAX_ROLLBACK_HISTORY]

    def _breaker_tripped(self) -> bool:
        limit = self._cfg_int("max_consecutive_failures")
        return limit > 0 and self._consecutive_failures >= limit

    def _l2_available(self) -> bool:
        if self.hooks.director_review is None:
            return False
        cap = self._cfg_int("max_l2_reviews")
        return cap <= 0 or self._l2_reviews < cap

    def _rollback_available(self, branch_id: str) -> tuple[bool, str]:
        """Check the per-branch/total caps and the cooldown (no side effects)."""
        now = self._now()
        lineage = self._lineage_root(branch_id)
        cap_branch = self._cfg_int("max_rollbacks_per_branch")
        used_branch = self._rollbacks_by_lineage.get(lineage, 0)
        if cap_branch > 0 and used_branch >= cap_branch:
            return False, (
                f"per-branch rollback cap reached ({used_branch}/{cap_branch} on lineage {lineage})"
            )
        cap_total = self._cfg_int("max_rollbacks_total")
        if cap_total > 0 and self._rollbacks_total >= cap_total:
            return False, f"total rollback cap reached ({self._rollbacks_total}/{cap_total})"
        cooldown = self._limit("rollback_cooldown_s")
        if cooldown > 0 and self._last_rollback_at > 0:
            remaining = cooldown - (now - self._last_rollback_at)
            if remaining > 0:
                return False, f"rollback cooldown active ({remaining:.1f}s remaining)"
        return True, ""

    # ------------------------------------------------------------------ #
    # Level selection
    # ------------------------------------------------------------------ #
    def _next_level(self, primary: StagnationKind) -> RecoveryLevel:
        """Next rung of the bounded L1 → L2 → L3 ladder for ``primary``."""
        attempts = self._kind_attempts.get(primary.value, 0)
        if attempts <= 0:
            return RecoveryLevel.L1_LOCAL
        if attempts == 1:
            if self._l2_available():
                return RecoveryLevel.L2_REVIEW
            available, _why = self._rollback_available(self._current_branch_id)
            if available:
                return RecoveryLevel.L3_ROLLBACK
            return RecoveryLevel.L1_LOCAL
        if attempts == 2:
            available, _why = self._rollback_available(self._current_branch_id)
            if available:
                return RecoveryLevel.L3_ROLLBACK
            return RecoveryLevel.NONE
        return RecoveryLevel.NONE

    def can_recover(self, kind: StagnationKind | str) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for handling one stagnation signal."""
        resolved = _coerce_kind(kind)
        if resolved in HARD_SIGNALS:
            return False, f"hard signal {resolved.value!r}: safe pause required (no rollback)"
        if self._breaker_tripped():
            limit = self._cfg_int("max_consecutive_failures")
            return False, (
                f"circuit breaker open: {self._consecutive_failures} consecutive recovery "
                f"failures (limit {limit})"
            )
        level = self._next_level(resolved)
        if level is RecoveryLevel.NONE:
            return False, (
                f"recovery ladder exhausted for {resolved.value!r}: "
                f"{_LADDER[0].value}/{_LADDER[1].value}/{_LADDER[2].value} already attempted"
            )
        return True, ""

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _result(
        level: RecoveryLevel,
        actions: Sequence[str],
        reason: str,
        *,
        ok: bool,
        hint: str = "",
        rollback: Mapping[str, Any] | None = None,
        error: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "level": level.value,
            "actions": [str(action) for action in actions],
            "reason": str(reason),
            "rollback": dict(rollback) if rollback is not None else None,
            "ok": bool(ok),
            "hint": str(hint),
            "signals": [],
            "error": str(error),
        }
        result.update(extra)
        return result

    def _safe_pause(self, kinds: Sequence[StagnationKind], reason: str, *, ok: bool) -> dict[str, Any]:
        primary = self._primary(kinds)
        hint = _HARD_HINTS.get(primary, "Pause the run safely and report the blocking condition.")
        return self._result(
            RecoveryLevel.NONE,
            ["safe_pause"],
            reason,
            ok=ok,
            hint=hint,
            primary=primary.value,
        )

    @staticmethod
    def _primary(kinds: Sequence[StagnationKind]) -> StagnationKind:
        return max(kinds, key=lambda kind: SIGNAL_SEVERITY.get(kind, 0))

    def _pick_agent(self, primary: StagnationKind, context: Mapping[str, Any]) -> AgentRole:
        explicit = _coerce_agent(context.get("agent"))
        if explicit is not None:
            return explicit
        return _DEFAULT_AGENT.get(primary, AgentRole.DIRECTOR)

    @staticmethod
    def _hint_for(primary: StagnationKind, agent: AgentRole) -> str:
        agent_hints = _AGENT_L1_HINTS.get(agent, {})
        if primary in agent_hints:
            return agent_hints[primary]
        return _L1_HINTS.get(primary, _L1_HINTS[StagnationKind.LOCAL_STAGNATION])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def handle(
        self,
        signals: Sequence[StagnationKind],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle one round of stagnation signals (one action at most).

        Chooses the next rung of the L1/L2/L3 ladder (or a safe pause for hard
        signals/circuit breaker), executes exactly one action through the hooks,
        records the attempt and returns the structured result
        ``{"level","actions","reason","rollback","ok","hint","signals","error"}``.
        """
        return self._handle(signals, context=context, forced=False)

    def escalate(self, reason: str, *, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Force the L3 rollback path (manual escalation; caps still apply)."""
        text = str(reason or "").strip() or "manual escalation"
        context: dict[str, Any] = {
            "force_level": RecoveryLevel.L3_ROLLBACK.value,
            "reason": text,
            "evidence": dict(evidence or {}),
        }
        return self._handle([StagnationKind.GLOBAL_STAGNATION], context=context, forced=True)

    def record_attempt(
        self,
        level: RecoveryLevel | str,
        *,
        ok: bool,
        detail: str = "",
    ) -> None:
        """Record the outcome of a recovery attempt.

        :meth:`handle` / :meth:`escalate` already record their own attempts and
        rollbacks; call this only for recovery work executed outside them (e.g. a
        hint the director delivered manually).  Returning ``ok=False`` feeds the
        circuit breaker; ``ok=True`` closes it again.
        """
        resolved = RecoveryLevel.coerce(level)
        self._attempt_counts[resolved.value] = self._attempt_counts.get(resolved.value, 0) + 1
        self._total_attempts += 1
        if ok:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        self._record_attempt(resolved, ok=ok, detail=detail)

    def rollback(
        self,
        reason: str,
        *,
        evidence: Mapping[str, Any] | None = None,
        checkpoint_id: str = "latest",
    ) -> dict[str, Any]:
        """Perform one capped rollback (save-pre ➜ restore ➜ re-localize).

        Returns ``{"ok","branch_id","previous_branch_id","checkpoint_id",
        "reason","relocalized","error"}`` (+ audit extras).  ``branch_id`` is the
        new branch created on top of the restored state; ``previous_branch_id``
        is the branch that was rolled back.  Reason + evidence are recorded before
        the restore so the audit trail survives a crash mid-rollback.
        """
        now = self._now()
        text = str(reason or "").strip() or "recovery rollback"
        checkpoint = str(checkpoint_id or "").strip() or "latest"
        previous_branch_id = self._current_branch_id
        evidence_payload = _json_safe(dict(evidence or {}))
        result: dict[str, Any] = {
            "ok": False,
            "branch_id": "",
            "previous_branch_id": previous_branch_id,
            "checkpoint_id": checkpoint,
            "reason": text,
            "relocalized": False,
            "error": "",
        }
        available, why = self._rollback_available(previous_branch_id)
        if not available:
            self._append_rollback(
                {
                    "reason": text,
                    "evidence": evidence_payload,
                    "checkpoint_id": checkpoint,
                    "previous_branch_id": previous_branch_id,
                    "requested_at": now,
                    "status": "blocked",
                    "error": why,
                }
            )
            result["error"] = why
            return result

        record: dict[str, Any] = {
            "reason": text,
            "evidence": evidence_payload,
            "checkpoint_id": checkpoint,
            "previous_branch_id": previous_branch_id,
            "requested_at": now,
            "status": "requested",
            "error": "",
        }
        # Record reason + evidence BEFORE restoring anything (crash-safe audit).
        self._append_rollback(record)
        self._emit(
            EventType.ROLLBACK_REQUESTED,
            {"reason": text, "checkpoint_id": checkpoint, "branch_id": previous_branch_id},
        )

        warnings: list[str] = []
        pre_manifest: dict[str, Any] = {}
        if self.hooks.save_checkpoint is not None:
            ok, value, error = _safe_call(
                "save_checkpoint", self.hooks.save_checkpoint, f"pre_rollback:{text}"
            )
            if ok:
                pre_manifest = _mapping(value)
                record["pre_rollback_checkpoint"] = str(
                    pre_manifest.get("checkpoint_id") or ""
                )
            else:
                warnings.append(error)

        if self.hooks.restore_checkpoint is None:
            message = "restore_checkpoint: hook is not configured"
            record.update({"status": "failed", "error": message})
            result["error"] = message
            return result

        ok, value, error = _safe_call(
            "restore_checkpoint",
            self.hooks.restore_checkpoint,
            checkpoint,
            previous_branch_id,
        )
        if not ok:
            record.update({"status": "failed", "error": error})
            result["error"] = error
            result["warnings"] = warnings
            return result
        manifest = _mapping(value)

        relocalized = False
        if self.hooks.relocalize is not None:
            ok_relocalize, _value, error = _safe_call("relocalize", self.hooks.relocalize)
            if ok_relocalize:
                relocalized = True
            else:
                warnings.append(error)

        # Success accounting: caps/cooldown are monotonic, never reset by a rollback.
        self._rollbacks_total += 1
        lineage = self._lineage_root(previous_branch_id)
        self._rollbacks_by_lineage[lineage] = self._rollbacks_by_lineage.get(lineage, 0) + 1
        self._last_rollback_at = now
        new_branch = new_branch_id()
        self._lineage[new_branch] = lineage
        self._current_branch_id = new_branch
        self._consecutive_failures = 0
        record.update(
            {
                "status": "completed",
                "completed_at": now,
                "new_branch_id": new_branch,
                "relocalized": relocalized,
            }
        )
        self._emit(
            EventType.ROLLBACK_COMPLETED,
            {
                "branch_id": new_branch,
                "previous_branch_id": previous_branch_id,
                "checkpoint_id": checkpoint,
                "reason": text,
            },
        )
        result.update(
            {
                "ok": True,
                "branch_id": new_branch,
                "relocalized": relocalized,
                "error": "",
                "warnings": warnings,
                "evidence": evidence_payload,
                "manifest": manifest,
                "pre_rollback_manifest": pre_manifest,
            }
        )
        return result

    # ------------------------------------------------------------------ #
    # Ladder implementation
    # ------------------------------------------------------------------ #
    def _handle(
        self,
        signals: Sequence[StagnationKind],
        *,
        context: Mapping[str, Any] | None,
        forced: bool,
    ) -> dict[str, Any]:
        ctx = dict(context or {})
        branch_hint = str(ctx.get("branch_id") or "").strip()
        if branch_hint:
            self.set_current_branch(
                branch_hint, parent_branch_id=str(ctx.get("parent_branch_id") or "")
            )

        kinds: list[StagnationKind] = []
        for raw in signals or ():
            kind = _coerce_kind(raw)
            if kind not in kinds:
                kinds.append(kind)
        if not kinds:
            return self._result(
                RecoveryLevel.NONE, [], "no stagnation signals", ok=True,
                hint="No recovery needed.",
            )

        primary = self._primary(kinds)
        signal_values = [kind.value for kind in kinds]

        if primary in HARD_SIGNALS:
            payload = {
                "kind": primary.value,
                "detail": f"hard stop: {primary.value}",
                "level": RecoveryLevel.NONE.value,
            }
            self._emit(EventType.LIMIT_REACHED, payload)
            result = self._safe_pause(
                kinds,
                f"hard signal {primary.value}: safe pause required (no rollback can "
                f"restore spent resources)",
                ok=True,
            )
            result["signals"] = signal_values
            self._last_result = result
            return result

        if not forced:
            allowed, why = self.can_recover(primary)
            if not allowed:
                result = self._safe_pause(kinds, why, ok=False)
                result["signals"] = signal_values
                self._last_result = result
                return result

        forced_level = ctx.get("force_level")
        note = ""
        if forced_level is not None:
            level = RecoveryLevel.coerce(forced_level)
        else:
            level = self._next_level(primary)
        if level is RecoveryLevel.L2_REVIEW and not self._l2_available():
            note = "L2 global review unavailable (no hook or review cap reached)"
            level = RecoveryLevel.L1_LOCAL
        if level is RecoveryLevel.NONE:
            result = self._safe_pause(
                kinds,
                f"recovery ladder exhausted for {primary.value}",
                ok=False,
            )
        else:
            ctx.setdefault("signal_kinds", [kind.value for kind in kinds])
            result = self._dispatch(level, primary, ctx, note=note)

        self._kind_attempts[primary.value] = self._kind_attempts.get(primary.value, 0) + 1
        self.record_attempt(
            RecoveryLevel.coerce(result.get("level", RecoveryLevel.NONE.value)),
            ok=bool(result.get("ok")),
            detail=str(result.get("reason") or ""),
        )
        result["signals"] = signal_values
        result["primary"] = primary.value
        self._emit(
            EventType.RECOVERY_FINISHED,
            {
                "level": result.get("level"),
                "result": result.get("ok"),
                "reason": result.get("reason"),
            },
        )
        self._last_result = result
        return result

    def _dispatch(
        self,
        level: RecoveryLevel,
        primary: StagnationKind,
        context: Mapping[str, Any],
        *,
        note: str = "",
    ) -> dict[str, Any]:
        suffix = f" ({note})" if note else ""
        if level is RecoveryLevel.L1_LOCAL:
            result = self._handle_l1(primary, context)
        elif level is RecoveryLevel.L2_REVIEW:
            result = self._handle_l2(primary, context)
        else:
            result = self._handle_l3(primary, context)
        if note:
            result["reason"] = f"{result['reason']}{suffix}"
        return result

    def _handle_l1(
        self,
        primary: StagnationKind,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        agent = self._pick_agent(primary, context)
        hint = self._hint_for(primary, agent)
        level = RecoveryLevel.L1_LOCAL
        self._emit(
            EventType.RECOVERY_STARTED,
            {"level": level.value, "reason": f"local hint for {agent.value} ({primary.value})"},
        )
        errors: list[str] = []
        ok = True
        if self.hooks.agent_hint is not None:
            delivered, _value, error = _safe_call(
                "agent_hint", self.hooks.agent_hint, agent, level.value, hint
            )
            if not delivered:
                errors.append(error)
                ok = False
        # When no hook is configured the hint is still returned in the result for
        # the caller to deliver, so the local recovery itself is not a failure.
        return self._result(
            level,
            ["l1_local_hint"],
            f"L1 local strategy hint for {agent.value} on {primary.value}",
            ok=ok,
            hint=hint,
            error="; ".join(errors),
            agent=agent.value,
        )

    def _handle_l2(
        self,
        primary: StagnationKind,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        level = RecoveryLevel.L2_REVIEW
        if self.hooks.director_review is None:
            return self._result(
                level,
                ["l2_global_review"],
                "L2 global review unavailable (no director_review hook)",
                ok=False,
                hint=self._hint_for(primary, self._pick_agent(primary, context)),
                error="director_review: hook is not configured",
            )
        self._emit(
            EventType.RECOVERY_STARTED,
            {"level": level.value, "reason": f"global review for {primary.value}"},
        )
        payload = {
            "primary": primary.value,
            "signals": [
                self._coerce_value(kind) for kind in context.get("signal_kinds", [primary.value])
            ],
            "context": _json_safe(dict(context)),
            "stats": self.stats(),
            "attempt": self._kind_attempts.get(primary.value, 0) + 1,
        }
        ok, review, error = _safe_call(
            "director_review", self.hooks.director_review, payload
        )
        self._l2_reviews += 1
        self._last_l2_at = self._now()
        if not ok:
            return self._result(
                level,
                ["l2_global_review"],
                f"L2 global review failed: {error}",
                ok=False,
                hint=self._hint_for(primary, self._pick_agent(primary, context)),
                error=error,
            )
        review_map = _mapping(review) if isinstance(review, Mapping) else {"decision": str(review)}
        decision = str(
            review_map.get("decision")
            or review_map.get("reason")
            or review_map.get("summary")
            or ""
        ).strip()
        hint = str(
            review_map.get("hint")
            or review_map.get("next_action")
            or decision
            or self._hint_for(primary, self._pick_agent(primary, context))
        ).strip()
        reason = f"L2 global review: {decision}" if decision else "L2 global review completed"
        return self._result(
            level,
            ["l2_global_review"],
            reason[:400],
            ok=True,
            hint=hint[:800],
            review=_json_safe(review_map),
            l2_reviews=self._l2_reviews,
        )

    @staticmethod
    def _coerce_value(value: Any) -> str:
        try:
            return _coerce_kind(value).value
        except ProtocolError:
            return str(value)

    def _handle_l3(
        self,
        primary: StagnationKind,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        level = RecoveryLevel.L3_ROLLBACK
        reason = str(context.get("reason") or "").strip() or (
            f"stagnation escalation: {primary.value}"
        )
        signals = list(context.get("signal_kinds") or [primary])
        evidence: dict[str, Any] = {
            "primary_signal": primary.value,
            "signals": [self._coerce_value(kind) for kind in signals],
            "context": _json_safe(dict(context)),
            "kind_attempts": dict(self._kind_attempts),
            "stats": self.stats(),
        }
        raw_evidence = context.get("evidence")
        if isinstance(raw_evidence, Mapping):
            evidence["extra"] = _json_safe(dict(raw_evidence))
        failed_targets = context.get("failed_task_ids") or context.get("task_ids")
        if failed_targets:
            evidence["failed_targets"] = _json_safe(
                list(failed_targets)
                if isinstance(failed_targets, (list, tuple, set))
                else [failed_targets]
            )
        checkpoint_id = str(context.get("checkpoint_id") or "latest")
        rollback = self.rollback(reason, evidence=evidence, checkpoint_id=checkpoint_id)
        actions = [
            f"save_checkpoint:pre_rollback:{reason}"[:200],
            f"restore_checkpoint:{checkpoint_id}",
        ]
        if rollback.get("relocalized"):
            actions.append("relocalize")
        if rollback.get("ok"):
            hint = (
                "State restored to the last good checkpoint: re-observe the game, "
                "re-localize navigation, and continue with a different plan than the "
                "one that caused the stagnation."
            )
            return self._result(
                level,
                actions,
                reason[:400],
                ok=True,
                hint=hint,
                rollback=rollback,
            )
        return self._result(
            level,
            actions,
            reason[:400],
            ok=False,
            hint=(
                "Rollback failed: pause the run safely, keep the audit evidence and "
                "report the rollback error instead of retrying in a loop."
            ),
            rollback=rollback,
            error=str(rollback.get("error") or "rollback failed"),
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        """JSON-safe recovery counters (for the viewer/report/director prompt)."""
        now = self._now()
        cooldown = self._limit("rollback_cooldown_s")
        remaining = 0.0
        if cooldown > 0 and self._last_rollback_at > 0:
            remaining = max(0.0, cooldown - (now - self._last_rollback_at))
        return {
            "elapsed_s": max(0.0, now - self._started_at),
            "attempts": self._total_attempts,
            "attempts_by_level": dict(sorted(self._attempt_counts.items())),
            "consecutive_failures": self._consecutive_failures,
            "max_consecutive_failures": self._cfg_int("max_consecutive_failures"),
            "breaker_tripped": self._breaker_tripped(),
            "kind_attempts": dict(sorted(self._kind_attempts.items())),
            "l2_reviews": self._l2_reviews,
            "max_l2_reviews": self._cfg_int("max_l2_reviews"),
            "last_l2_at": self._last_l2_at,
            "rollbacks_total": self._rollbacks_total,
            "max_rollbacks_total": self._cfg_int("max_rollbacks_total"),
            "rollbacks_by_lineage": dict(sorted(self._rollbacks_by_lineage.items())),
            "last_rollback_at": self._last_rollback_at,
            "cooldown_remaining_s": round(remaining, 3),
            "current_branch_id": self._current_branch_id,
            "rollback_history": [
                {
                    key: value
                    for key, value in record.items()
                    if key in ("reason", "status", "requested_at", "completed_at",
                               "previous_branch_id", "new_branch_id", "checkpoint_id", "error")
                }
                for record in self._rollback_history[-3:]
            ],
        }

    def summary(self) -> dict[str, Any]:
        """Human/viewer oriented summary (config, caps, hooks, last result)."""
        hooks = {
            "save_checkpoint": self.hooks.save_checkpoint is not None,
            "restore_checkpoint": self.hooks.restore_checkpoint is not None,
            "relocalize": self.hooks.relocalize is not None,
            "director_review": self.hooks.director_review is not None,
            "notify": self.hooks.notify is not None,
            "agent_hint": self.hooks.agent_hint is not None,
        }
        return {
            "config": asdict(self.config),
            "stats": self.stats(),
            "hooks": hooks,
            "last_result": (
                {key: value for key, value in self._last_result.items() if key != "rollback"}
                | {
                    "rollback_ok": (
                        None
                        if not self._last_result.get("rollback")
                        else bool(self._last_result["rollback"].get("ok"))
                    )
                }
                if self._last_result
                else None
            ),
            "recent_attempts": [
                {
                    key: value
                    for key, value in record.items()
                    if key in ("level", "ok", "detail", "at")
                }
                for record in self._attempt_history[-5:]
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RecoveryManager(attempts={self._total_attempts}, "
            f"rollbacks={self._rollbacks_total}, l2={self._l2_reviews}, "
            f"consecutive_failures={self._consecutive_failures})"
        )
