"""Unit tests for :mod:`llm_agent.multi_agent.recovery`.

Fake injected hooks record the exact call order, so the tests verify the
mandated escalation ladder and the safety invariants without touching PyBoy, the
network or the real checkpoint store:

* L1 local strategy hints (navigation alternative route / story other clue /
  pokémon other tactic) are always tried first, per responsible agent;
* hard signals (``RESOURCE_EXHAUSTED`` / ``RECOVERY_LIMIT``) force a safe pause
  and never trigger a rollback;
* L2 director review is rate limited; L3 rollback saves a pre-rollback
  checkpoint *before* restoring, then re-localizes;
* rollback caps (per branch lineage + total) and the cooldown are enforced;
* hook failures never crash the manager and are reported in the result;
* budget/attempt counters are never reset, and the ladder terminates.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_agent.multi_agent.contracts import (
    AgentRole,
    EventType,
    ProtocolError,
    RecoveryLevel,
    StagnationKind,
)
from llm_agent.multi_agent.recovery import (
    HARD_SIGNALS,
    RecoveryConfig,
    RecoveryHooks,
    RecoveryManager,
    SIGNAL_SEVERITY,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class Recorder:
    """Records every hook call; individual hooks can be made to fail."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.order: list[str] = []
        self.fail: set[str] = set()
        self.restore_manifest: dict[str, Any] = {
            "checkpoint_id": "checkpoint_000007",
            "branch_id": "branch_parent",
            "state_sha256": "a" * 64,
            "budget_snapshot": {"run": {"cost_usd": 1.5, "calls": 3}},
        }

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail:
            raise RuntimeError(f"{name} exploded")

    # -- hooks ------------------------------------------------------ #
    def save_checkpoint(self, reason: str) -> dict[str, Any]:
        self._maybe_fail("save")
        self.order.append("save")
        self.calls.append(("save", reason))
        return {"checkpoint_id": "checkpoint_000006", "branch_id": "branch_parent"}

    def restore_checkpoint(self, checkpoint_id: str, branch_id: str) -> dict[str, Any]:
        self._maybe_fail("restore")
        self.order.append("restore")
        self.calls.append(("restore", checkpoint_id, branch_id))
        return dict(self.restore_manifest, checkpoint_id=checkpoint_id)

    def relocalize(self) -> dict[str, Any]:
        self._maybe_fail("relocalize")
        self.order.append("relocalize")
        self.calls.append(("relocalize",))
        return {"ok": True, "cells": 3}

    def director_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._maybe_fail("review")
        self.order.append("review")
        self.calls.append(("review", payload.get("primary")))
        return {"decision": "replan: talk to another npc", "hint": "try a different clue"}

    def notify(self, event_type: str, payload: dict[str, Any]) -> None:
        self._maybe_fail("notify")
        self.calls.append(("notify", event_type))

    def agent_hint(self, agent: AgentRole, level: str, hint: str) -> None:
        self._maybe_fail("hint")
        self.calls.append(("hint", agent.value, level, hint))

    def hooks(self) -> RecoveryHooks:
        return RecoveryHooks(
            save_checkpoint=self.save_checkpoint,
            restore_checkpoint=self.restore_checkpoint,
            relocalize=self.relocalize,
            director_review=self.director_review,
            notify=self.notify,
            agent_hint=self.agent_hint,
        )

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]


def _config(**overrides: Any) -> RecoveryConfig:
    values: dict[str, Any] = dict(
        max_rollbacks_per_branch=5,
        max_rollbacks_total=10,
        max_l2_reviews=8,
        rollback_cooldown_s=0.0,
        max_consecutive_failures=5,
    )
    values.update(overrides)
    return RecoveryConfig(**values)


def _manager(recorder: Recorder | None = None, clock: FakeClock | None = None, **overrides: Any) -> RecoveryManager:
    return RecoveryManager(
        _config(**overrides),
        hooks=(recorder or Recorder()).hooks(),
        clock=clock or FakeClock(),
    )


def _notify_events(recorder: Recorder) -> list[str]:
    return [call[1] for call in recorder.calls if call[0] == "notify"]


# --------------------------------------------------------------------------- #
# Config / interface
# --------------------------------------------------------------------------- #

def test_config_defaults_match_frozen_interface() -> None:
    config = RecoveryConfig()
    assert config.max_rollbacks_per_branch == 2
    assert config.max_rollbacks_total == 4
    assert config.max_l2_reviews == 8
    assert config.rollback_cooldown_s == 120.0
    assert config.max_consecutive_failures == 5


def test_hooks_default_to_all_none() -> None:
    hooks = RecoveryHooks()
    for name in (
        "save_checkpoint",
        "restore_checkpoint",
        "relocalize",
        "director_review",
        "notify",
        "agent_hint",
    ):
        assert getattr(hooks, name) is None


def test_hard_signals_are_the_two_resource_kinds() -> None:
    assert HARD_SIGNALS == {StagnationKind.RESOURCE_EXHAUSTED, StagnationKind.RECOVERY_LIMIT}
    assert SIGNAL_SEVERITY[StagnationKind.RESOURCE_EXHAUSTED] > SIGNAL_SEVERITY[
        StagnationKind.LOCAL_STAGNATION
    ]


def test_result_structure_is_stable() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    result = manager.handle([StagnationKind.LOCAL_STAGNATION])
    required = {"level", "actions", "reason", "rollback", "ok", "hint", "signals"}
    assert required.issubset(result)
    assert result["level"] == RecoveryLevel.L1_LOCAL.value
    assert isinstance(result["actions"], list)
    assert result["rollback"] is None
    assert result["signals"] == [StagnationKind.LOCAL_STAGNATION.value]
    json.dumps(result)


# --------------------------------------------------------------------------- #
# L1 local recovery
# --------------------------------------------------------------------------- #

def test_l1_recovery_is_selected_first_with_agent_hint() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    result = manager.handle(
        [StagnationKind.LOCAL_STAGNATION], context={"agent": "navigation"}
    )
    assert result["level"] == RecoveryLevel.L1_LOCAL.value
    assert result["ok"] is True
    assert result["actions"] == ["l1_local_hint"]
    assert "route" in result["hint"].lower() or "direction" in result["hint"].lower()
    hint_calls = [call for call in recorder.calls if call[0] == "hint"]
    assert hint_calls == [("hint", "navigation", "l1_local", result["hint"])]
    # L1 never touches the emulator / checkpoint store.
    assert "save" not in recorder.order
    assert "restore" not in recorder.order


def test_l1_hint_is_agent_specific() -> None:
    story = _manager().handle([StagnationKind.LOCAL_STAGNATION], context={"agent": "story"})
    assert "clue" in story["hint"].lower()
    pokemon = _manager().handle(
        [StagnationKind.REPEATED_FAILURE], context={"agent": "pokemon"}
    )
    assert "tactic" in pokemon["hint"].lower()
    director = _manager().handle([StagnationKind.GLOBAL_STAGNATION])
    assert director["agent"] == AgentRole.DIRECTOR.value
    assert "progress" in director["hint"].lower() or "plan" in director["hint"].lower()


def test_l1_without_hook_still_returns_the_hint() -> None:
    manager = RecoveryManager(_config(), clock=FakeClock())
    result = manager.handle([StagnationKind.LOCAL_STAGNATION])
    assert result["level"] == RecoveryLevel.L1_LOCAL.value
    assert result["ok"] is True
    assert result["hint"]


def test_l1_hint_hook_failure_is_reported_not_raised() -> None:
    recorder = Recorder()
    recorder.fail.add("hint")
    manager = _manager(recorder)
    result = manager.handle([StagnationKind.LOCAL_STAGNATION])
    assert result["ok"] is False
    assert "agent_hint" in result["error"]
    assert result["hint"]  # the hint is still returned for the caller


# --------------------------------------------------------------------------- #
# Hard signals → safe pause
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "kind", [StagnationKind.RESOURCE_EXHAUSTED, StagnationKind.RECOVERY_LIMIT]
)
def test_hard_signal_forces_safe_pause_without_rollback(kind: StagnationKind) -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    result = manager.handle([kind], context={"agent": "navigation"})
    assert result["level"] == RecoveryLevel.NONE.value
    assert result["actions"] == ["safe_pause"]
    assert result["ok"] is True
    assert result["rollback"] is None
    assert "no rollback" in result["reason"].lower()
    assert recorder.kinds() == ["notify"]  # only the LIMIT_REACHED event
    assert EventType.LIMIT_REACHED.value in _notify_events(recorder)
    assert manager.stats()["attempts"] == 0


def test_can_recover_rejects_hard_signals() -> None:
    manager = _manager()
    allowed, why = manager.can_recover(StagnationKind.RESOURCE_EXHAUSTED)
    assert allowed is False
    assert "safe pause" in why
    allowed, why = manager.can_recover(StagnationKind.LOCAL_STAGNATION)
    assert allowed is True
    assert why == ""


def test_empty_signals_are_a_no_op() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    result = manager.handle([])
    assert result["level"] == RecoveryLevel.NONE.value
    assert result["ok"] is True
    assert recorder.calls == []


def test_unknown_signal_raises_protocol_error() -> None:
    manager = _manager()
    with pytest.raises(ProtocolError):
        manager.handle(["definitely-not-a-signal"])


# --------------------------------------------------------------------------- #
# L2 global review
# --------------------------------------------------------------------------- #

def test_l2_review_is_used_after_the_first_local_attempt() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    first = manager.handle([StagnationKind.GLOBAL_STAGNATION])
    assert first["level"] == RecoveryLevel.L1_LOCAL.value
    second = manager.handle([StagnationKind.GLOBAL_STAGNATION])
    assert second["level"] == RecoveryLevel.L2_REVIEW.value
    assert second["ok"] is True
    assert "talk to another npc" in second["reason"]
    assert second["review"]["decision"].startswith("replan")
    assert second["hint"] == "try a different clue"
    assert [call for call in recorder.calls if call[0] == "review"] == [
        ("review", StagnationKind.GLOBAL_STAGNATION.value)
    ]
    assert manager.stats()["l2_reviews"] == 1


def test_l2_review_is_rate_limited() -> None:
    recorder = Recorder()
    manager = _manager(recorder, max_l2_reviews=1)
    manager.handle([StagnationKind.NO_PROGRESS])
    manager.handle([StagnationKind.NO_PROGRESS])  # consumes the single L2 review
    manager.handle([StagnationKind.NO_PROGRESS])  # -> L3, never another L2
    review_calls = [call for call in recorder.calls if call[0] == "review"]
    assert len(review_calls) == 1
    assert manager.stats()["l2_reviews"] == 1


def test_l2_requires_the_director_hook() -> None:
    manager = RecoveryManager(_config(), clock=FakeClock())
    manager.handle([StagnationKind.GLOBAL_STAGNATION])
    second = manager.handle([StagnationKind.GLOBAL_STAGNATION])
    assert second["level"] == RecoveryLevel.L3_ROLLBACK.value
    assert second["rollback"]["ok"] is False
    assert "restore_checkpoint" in second["error"]


def test_l2_review_failure_is_reported() -> None:
    recorder = Recorder()
    recorder.fail.add("review")
    manager = _manager(recorder)
    manager.handle([StagnationKind.GLOBAL_STAGNATION])
    second = manager.handle([StagnationKind.GLOBAL_STAGNATION])
    assert second["level"] == RecoveryLevel.L2_REVIEW.value
    assert second["ok"] is False
    assert "director_review" in second["error"]


# --------------------------------------------------------------------------- #
# L3 rollback
# --------------------------------------------------------------------------- #

def test_l3_rollback_saves_then_restores_then_relocalizes() -> None:
    recorder = Recorder()
    clock = FakeClock()
    manager = _manager(recorder, clock)
    manager.handle([StagnationKind.LOCAL_STAGNATION])
    manager.handle([StagnationKind.LOCAL_STAGNATION])
    result = manager.handle([StagnationKind.LOCAL_STAGNATION])
    assert result["level"] == RecoveryLevel.L3_ROLLBACK.value
    assert result["ok"] is True
    rollback = result["rollback"]
    assert rollback["ok"] is True
    assert rollback["previous_branch_id"] == ""
    assert rollback["branch_id"].startswith("branch_")
    assert rollback["branch_id"] != rollback["previous_branch_id"]
    assert rollback["checkpoint_id"] == "latest"
    assert rollback["relocalized"] is True
    # save BEFORE restore, then relocalize (the earlier L2 review precedes them)
    assert recorder.order[-3:] == ["save", "restore", "relocalize"]
    save_call = next(call for call in recorder.calls if call[0] == "save")
    assert save_call[1].startswith("pre_rollback:")
    restore_call = next(call for call in recorder.calls if call[0] == "restore")
    assert restore_call[1] == "latest"
    assert rollback["manifest"]["state_sha256"] == "a" * 64
    events = _notify_events(recorder)
    assert EventType.ROLLBACK_REQUESTED.value in events
    assert EventType.ROLLBACK_COMPLETED.value in events
    assert RECOVERY_EVENT in events
    stats = manager.stats()
    assert stats["rollbacks_total"] == 1
    assert stats["attempts_by_level"].get(RecoveryLevel.L3_ROLLBACK.value) == 1
    assert manager.current_branch_id == rollback["branch_id"]


RECOVERY_EVENT = EventType.RECOVERY_FINISHED.value


def test_rollback_records_reason_and_evidence_before_restore() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    result = manager.rollback(
        "stuck in dialogue",
        evidence={"failed_task_ids": ["task_a"], "detail": "same npc"},
        checkpoint_id="checkpoint_000004",
    )
    assert result["ok"] is True
    assert result["reason"] == "stuck in dialogue"
    assert result["checkpoint_id"] == "checkpoint_000004"
    record = manager._rollback_history[-1]  # noqa: SLF001 - white-box audit check
    assert record["status"] == "completed"
    assert record["reason"] == "stuck in dialogue"
    assert record["evidence"]["failed_task_ids"] == ["task_a"]
    save_call = next(call for call in recorder.calls if call[0] == "save")
    assert save_call[1] == "pre_rollback:stuck in dialogue"
    assert recorder.order[0] == "save"


def test_rollback_total_cap_is_enforced() -> None:
    recorder = Recorder()
    manager = _manager(recorder, max_rollbacks_total=1)
    assert manager.rollback("first")["ok"] is True
    blocked = manager.rollback("second")
    assert blocked["ok"] is False
    assert "total rollback cap" in blocked["error"]
    assert manager.stats()["rollbacks_total"] == 1
    assert len([call for call in recorder.calls if call[0] == "restore"]) == 1
    assert manager._rollback_history[-1]["status"] == "blocked"  # noqa: SLF001


def test_rollback_per_branch_lineage_cap_is_enforced() -> None:
    recorder = Recorder()
    manager = _manager(recorder, max_rollbacks_per_branch=1, max_rollbacks_total=10)
    manager.set_current_branch("branch_lineage", parent_branch_id="")
    assert manager.rollback("first")["ok"] is True
    # The rollback created a child branch, but the lineage cap still applies.
    second = manager.rollback("second")
    assert second["ok"] is False
    assert "per-branch rollback cap" in second["error"]


def test_rollback_cooldown_is_enforced_and_expires() -> None:
    recorder = Recorder()
    clock = FakeClock()
    manager = _manager(
        recorder, clock, rollback_cooldown_s=120.0, max_rollbacks_total=10,
        max_rollbacks_per_branch=10,
    )
    assert manager.rollback("first")["ok"] is True
    blocked = manager.rollback("too soon")
    assert blocked["ok"] is False
    assert "cooldown" in blocked["error"]
    assert manager.stats()["cooldown_remaining_s"] == pytest.approx(120.0)
    clock.advance(120.0)
    assert manager.rollback("after cooldown")["ok"] is True
    assert manager.stats()["rollbacks_total"] == 2


def test_escalate_forces_the_l3_path() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    result = manager.escalate("manual review says roll back", evidence={"why": "loop"})
    assert result["level"] == RecoveryLevel.L3_ROLLBACK.value
    assert result["ok"] is True
    assert result["rollback"]["ok"] is True
    assert manager.stats()["rollbacks_total"] == 1
    save_call = next(call for call in recorder.calls if call[0] == "save")
    assert save_call[1] == "pre_rollback:manual review says roll back"


def test_escalate_still_respects_rollback_caps() -> None:
    recorder = Recorder()
    manager = _manager(recorder, max_rollbacks_total=1)
    assert manager.escalate("first")["ok"] is True
    second = manager.escalate("second")
    assert second["ok"] is False
    assert "total rollback cap" in second["error"]


# --------------------------------------------------------------------------- #
# Hook failure tolerance
# --------------------------------------------------------------------------- #

def test_save_checkpoint_failure_does_not_block_rollback() -> None:
    recorder = Recorder()
    recorder.fail.add("save")
    manager = _manager(recorder)
    result = manager.rollback("save fails")
    assert result["ok"] is True  # state restoration is more important than the audit snapshot
    assert any("save_checkpoint" in warning for warning in result["warnings"])
    assert recorder.order == ["restore", "relocalize"]


def test_restore_failure_is_reported_and_does_not_raise() -> None:
    recorder = Recorder()
    recorder.fail.add("restore")
    manager = _manager(recorder)
    result = manager.rollback("restore fails")
    assert result["ok"] is False
    assert "restore_checkpoint" in result["error"]
    assert manager.stats()["rollbacks_total"] == 0
    assert manager._rollback_history[-1]["status"] == "failed"  # noqa: SLF001


def test_relocalize_failure_leaves_rollback_ok_with_warning() -> None:
    recorder = Recorder()
    recorder.fail.add("relocalize")
    manager = _manager(recorder)
    result = manager.rollback("relocalize fails")
    assert result["ok"] is True
    assert result["relocalized"] is False
    assert any("relocalize" in warning for warning in result["warnings"])


def test_notify_failure_never_crashes_the_manager() -> None:
    recorder = Recorder()
    recorder.fail.add("notify")
    manager = _manager(recorder)
    assert manager.handle([StagnationKind.RESOURCE_EXHAUSTED])["level"] == RecoveryLevel.NONE.value
    manager.handle([StagnationKind.LOCAL_STAGNATION])
    manager.handle([StagnationKind.LOCAL_STAGNATION])
    assert manager.handle([StagnationKind.LOCAL_STAGNATION])["ok"] is True


def test_missing_restore_hook_yields_a_clear_error() -> None:
    manager = RecoveryManager(_config(), clock=FakeClock())
    result = manager.rollback("nothing configured")
    assert result["ok"] is False
    assert "not configured" in result["error"]


# --------------------------------------------------------------------------- #
# Circuit breaker / bounded ladder / counters
# --------------------------------------------------------------------------- #

def test_circuit_breaker_opens_after_consecutive_failures() -> None:
    recorder = Recorder()
    manager = _manager(recorder, max_consecutive_failures=2)
    manager.record_attempt(RecoveryLevel.L1_LOCAL, ok=False, detail="hint ignored")
    manager.record_attempt(RecoveryLevel.L1_LOCAL, ok=False, detail="hint ignored again")
    allowed, why = manager.can_recover(StagnationKind.LOCAL_STAGNATION)
    assert allowed is False
    assert "circuit breaker" in why
    result = manager.handle([StagnationKind.LOCAL_STAGNATION])
    assert result["level"] == RecoveryLevel.NONE.value
    assert result["ok"] is False
    assert "circuit breaker" in result["reason"]
    assert not [call for call in recorder.calls if call[0] == "hint"]
    # A successful attempt closes the breaker again.
    manager.record_attempt(RecoveryLevel.L1_LOCAL, ok=True)
    assert manager.can_recover(StagnationKind.LOCAL_STAGNATION)[0] is True


def test_ladder_terminates_with_a_safe_pause() -> None:
    recorder = Recorder()
    manager = _manager(recorder, max_l2_reviews=1, max_rollbacks_total=1,
                       max_rollbacks_per_branch=1)
    levels: list[str] = []
    for _ in range(10):
        result = manager.handle([StagnationKind.NO_PROGRESS])
        levels.append(result["level"])
        assert result["level"] in {
            RecoveryLevel.NONE.value,
            RecoveryLevel.L1_LOCAL.value,
            RecoveryLevel.L2_REVIEW.value,
            RecoveryLevel.L3_ROLLBACK.value,
        }
    assert levels[:3] == [
        RecoveryLevel.L1_LOCAL.value,
        RecoveryLevel.L2_REVIEW.value,
        RecoveryLevel.L3_ROLLBACK.value,
    ]
    assert RecoveryLevel.NONE.value in levels
    assert levels[-1] == RecoveryLevel.NONE.value
    assert manager.stats()["rollbacks_total"] == 1


def test_attempt_counters_are_cumulative_and_never_reset_by_rollback() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    manager.handle([StagnationKind.LOCAL_STAGNATION])
    manager.handle([StagnationKind.LOCAL_STAGNATION])
    before = manager.stats()["attempts"]
    result = manager.rollback("manual", evidence={"budget_before": {"cost_usd": 1.5}})
    assert result["ok"] is True
    after = manager.stats()
    # A rollback restores game state only: attempt history and spend are kept.
    assert after["attempts"] == before
    assert after["rollbacks_total"] == 1
    assert result["manifest"]["budget_snapshot"]["run"]["cost_usd"] == 1.5
    assert manager.stats()["consecutive_failures"] == 0


def test_record_attempt_updates_counters_and_history() -> None:
    manager = _manager()
    manager.record_attempt(RecoveryLevel.L1_LOCAL, ok=True, detail="delivered")
    manager.record_attempt(RecoveryLevel.L2_REVIEW, ok=False, detail="review failed")
    stats = manager.stats()
    assert stats["attempts"] == 2
    assert stats["attempts_by_level"] == {"l1_local": 1, "l2_review": 1}
    assert stats["consecutive_failures"] == 1
    summary = manager.summary()
    assert summary["recent_attempts"][-1]["level"] == RecoveryLevel.L2_REVIEW.value
    assert summary["recent_attempts"][-1]["ok"] is False


# --------------------------------------------------------------------------- #
# Stats / summary / branch context
# --------------------------------------------------------------------------- #

def test_stats_and_summary_are_json_safe() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    manager.handle([StagnationKind.LOCAL_STAGNATION])
    manager.rollback("x")
    stats = manager.stats()
    json.dumps(stats)
    summary = manager.summary()
    json.dumps(summary)
    assert summary["config"]["max_rollbacks_total"] == 10
    assert all(summary["hooks"].values())
    assert summary["stats"]["rollbacks_total"] == 1
    assert summary["last_result"] is not None


def test_current_branch_and_context_branch_are_tracked() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    manager.set_current_branch("branch_abc", parent_branch_id="branch_parent")
    assert manager.current_branch_id == "branch_abc"
    manager.handle(
        [StagnationKind.LOCAL_STAGNATION],
        context={"branch_id": "branch_xyz", "parent_branch_id": "branch_abc"},
    )
    assert manager.current_branch_id == "branch_xyz"
    manager.set_current_branch("")  # ignored
    assert manager.current_branch_id == "branch_xyz"


def test_rollback_result_is_json_safe() -> None:
    recorder = Recorder()
    manager = _manager(recorder)
    result = manager.rollback("json", evidence={"nested": {"a": 1}})
    json.dumps(result, default=str)
    assert result["error"] == ""
