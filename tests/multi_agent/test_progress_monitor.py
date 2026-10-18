"""Unit tests for :mod:`llm_agent.multi_agent.progress_monitor`.

Only pure Python is used: no PyBoy, no ROM, no network, no API key.  The clock is
injected so every threshold/timeout test is deterministic and fast.  Covered:

* config defaults (frozen interface) and ``0 = disabled`` convention;
* measured progress per agent (cells/ways/areas/exits/coverage, story clues,
  battle wins/catches/team changes) and that button counts never count as progress;
* local / global / no-progress / repeated-failure / repeated-message detection;
* training-healing grace suppressing rollback triggers but not hard limits;
* hard resource limits and the recovery cap;
* ordering/dedup of signals, consume-latch semantics, reset semantics and
  snapshot/restore round-trip.
"""

from __future__ import annotations

import json

import pytest

from llm_agent.multi_agent.contracts import AgentRole, RecoveryLevel, StagnationKind
from llm_agent.multi_agent.progress_monitor import (
    ProgressMonitor,
    ProgressMonitorConfig,
    SIGNAL_ORDER,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class FakeClock:
    """Deterministic injectable clock."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


def _cfg(**overrides: object) -> ProgressMonitorConfig:
    """All thresholds disabled by default; tests enable only what they exercise."""
    values: dict[str, object] = dict(
        local_stagnation_steps=0,
        global_stagnation_seconds=0.0,
        repeated_failure_limit=0,
        repeated_message_limit=0,
        no_progress_action_limit=0,
        training_grace_seconds=0.0,
        min_progress_delta=0.0,
        window=0,
        max_runtime_s=0.0,
        max_actions=0,
        max_api_cost_usd=0.0,
        recovery_limit=0,
    )
    values.update(overrides)
    return ProgressMonitorConfig(**values)  # type: ignore[arg-type]


def _monitor(clock: FakeClock, **overrides: object) -> ProgressMonitor:
    return ProgressMonitor(_cfg(**overrides), clock=clock)


# --------------------------------------------------------------------------- #
# Config / constructor
# --------------------------------------------------------------------------- #

def test_config_defaults_match_frozen_interface() -> None:
    config = ProgressMonitorConfig()
    assert config.local_stagnation_steps == 60
    assert config.global_stagnation_seconds == 900.0
    assert config.repeated_failure_limit == 4
    assert config.repeated_message_limit == 3
    assert config.no_progress_action_limit == 150
    assert config.training_grace_seconds == 300.0
    assert config.min_progress_delta == 1.0
    assert config.window == 240
    assert config.max_runtime_s == 0.0
    assert config.max_actions == 0
    assert config.max_api_cost_usd == 0.0
    assert config.recovery_limit == 6


def test_initial_state_reports_no_signals_and_clean_counters() -> None:
    clock = FakeClock()
    monitor = _monitor(clock)
    assert monitor.check() == []
    summary = monitor.summary()
    assert summary["progress_total"] == 0
    assert summary["actions_total"] == 0
    assert summary["recoveries"]["total"] == 0
    assert summary["signals"] == []
    assert set(summary["agents"]) == {role.value for role in AgentRole}


# --------------------------------------------------------------------------- #
# Navigation progress
# --------------------------------------------------------------------------- #

def test_new_cells_count_as_progress_and_clear_local_stagnation() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=3)
    monitor.observe_navigation({"cells_known": 10})  # baseline, no gain yet
    for _ in range(3):
        monitor.observe_navigation({"cells_known": 10})
    assert monitor.check() == [StagnationKind.LOCAL_STAGNATION]
    assert monitor.summary()["agents"]["navigation"]["attempts_without_gain"] == 4

    monitor.observe_navigation({"cells_known": 13})
    assert monitor.check() == []
    summary = monitor.summary()
    assert summary["navigation"]["totals"]["new_cells"] == 3.0
    assert summary["agents"]["navigation"]["attempts_without_gain"] == 0
    assert summary["progress_total"] == 3.0


def test_ways_areas_exits_and_coverage_are_progress() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=2)
    monitor.observe_navigation(
        {"cells_known": 4, "ways_total": 1, "areas_total": 1, "exits": 0, "coverage_ratio": 0.1}
    )
    statuses = (
        {"cells_known": 4, "ways_total": 2, "areas_total": 1, "exits": 0, "coverage_ratio": 0.1},
        {"cells_known": 4, "ways_total": 2, "areas_total": 2, "exits": 0, "coverage_ratio": 0.1},
        {"cells_known": 4, "ways_total": 2, "areas_total": 2, "exits": 1, "coverage_ratio": 0.1},
        {"cells_known": 4, "ways_total": 2, "areas_total": 2, "exits": 1, "coverage_ratio": 0.3},
    )
    events: list[float] = []
    for status in statuses:
        monitor.observe_navigation(status)
        assert monitor.check() == [], f"gain in {status} should keep the agent fresh"
        monitor.observe_navigation(status)  # no new information -> a local attempt
        events.append(monitor.summary()["progress_total"])
    assert events == sorted(events)
    totals = monitor.summary()["navigation"]["totals"]
    assert totals["new_ways"] == 1.0
    assert totals["new_areas"] == 1.0
    assert totals["new_exits"] == 1.0
    assert totals["coverage_gains"] == 1.0


def test_repeated_exploration_is_not_progress() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=2)
    status = {"cells_known": 7, "ways_total": 2, "areas_total": 1, "unknown_frontier": 3}
    monitor.observe_navigation(status)
    monitor.observe_navigation(status)
    monitor.observe_navigation(status)
    summary = monitor.summary()
    assert summary["progress_total"] == 0.0
    assert summary["navigation"]["unknown_frontier"] == 3
    assert StagnationKind.LOCAL_STAGNATION in monitor.check()


def test_navigation_localization_failures_feed_repeated_failure() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, repeated_failure_limit=2)
    monitor.observe_navigation({"cells_known": 1, "localization_failures": 0})
    monitor.observe_navigation({"cells_known": 1, "localization_failures": 1})
    assert monitor.check() == []
    monitor.observe_navigation({"cells_known": 1, "localization_failures": 2})
    assert monitor.check() == [StagnationKind.REPEATED_FAILURE]
    assert "navigation" in monitor.summary()["failures"]["last_detail"]


def test_observe_navigation_rejects_non_mapping() -> None:
    monitor = _monitor(FakeClock())
    with pytest.raises(TypeError):
        monitor.observe_navigation([])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Story / Pokémon progress
# --------------------------------------------------------------------------- #

def test_story_clues_npcs_and_completed_tasks_are_progress() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=3)
    for _ in range(3):
        monitor.observe_story()
    assert monitor.check() == [StagnationKind.LOCAL_STAGNATION]
    monitor.observe_story(discoveries=1, new_npcs=1, tasks_completed=1, detail="clue")
    assert monitor.check() == []
    summary = monitor.summary()
    assert summary["story"]["discoveries"] == 1
    assert summary["story"]["new_npcs"] == 1
    assert summary["story"]["tasks_completed"] == 1
    assert summary["progress_total"] == 3.0


def test_repeated_dialogues_are_not_progress() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=2)
    monitor.observe_story(repeated_dialogues=5, detail="same npc")
    monitor.observe_story(repeated_dialogues=5, detail="same npc")
    summary = monitor.summary()
    assert summary["progress_total"] == 0.0
    assert summary["story"]["repeated_dialogues"] == 10
    assert summary["story"]["last_detail"] == "same npc"
    assert StagnationKind.LOCAL_STAGNATION in monitor.check()


def test_pokemon_wins_catches_and_team_changes_are_progress() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=2)
    monitor.observe_pokemon()
    monitor.observe_pokemon()
    assert StagnationKind.LOCAL_STAGNATION in monitor.check()
    monitor.observe_pokemon(wins=1)
    assert monitor.check() == []
    monitor.observe_pokemon(battles=2, catches=1, team_changes=1)
    assert monitor.check() == []
    totals = monitor.summary()["pokemon"]
    assert totals["wins"] == 1
    assert totals["catches"] == 1
    assert totals["team_changes"] == 1


def test_pokemon_losses_feed_repeated_failure_and_wins_reset_it() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, repeated_failure_limit=2)
    monitor.observe_pokemon(losses=1, details="lost")
    assert monitor.check() == []
    monitor.observe_pokemon(losses=1, details="lost again")
    assert monitor.check() == [StagnationKind.REPEATED_FAILURE]
    monitor.observe_pokemon(wins=1)
    assert monitor.check() == []


# --------------------------------------------------------------------------- #
# Global stagnation / actions / no progress
# --------------------------------------------------------------------------- #

def test_global_stagnation_requires_elapsed_time_and_resets_on_progress() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, global_stagnation_seconds=10.0)
    assert monitor.check() == []
    clock.advance(10.0)
    assert monitor.check() == [StagnationKind.GLOBAL_STAGNATION]
    monitor.observe_story(discoveries=1)
    assert monitor.check() == []
    assert monitor.summary()["seconds_since_progress"] == 0.0


def test_actions_alone_are_never_a_progress_signal() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, no_progress_action_limit=5, global_stagnation_seconds=100.0)
    clock.advance(5.0)
    monitor.observe_actions(3)
    summary = monitor.summary()
    assert summary["actions_total"] == 3
    assert summary["progress_total"] == 0.0
    assert summary["actions_since_progress"] == 3
    monitor.observe_actions(2)
    assert StagnationKind.NO_PROGRESS in monitor.check()
    # An action is not progress: the no-progress counter keeps growing.
    monitor.observe_actions(10)
    assert monitor.summary()["actions_since_progress"] == 15


def test_observe_actions_attributes_attempts_to_agent_label() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=3)
    monitor.observe_actions(2, label="story")
    monitor.observe_actions(1, label="pokemon")
    monitor.observe_actions(1, label="not-an-agent-label")
    summary = monitor.summary()
    assert summary["agents"]["story"]["attempts_without_gain"] == 2
    assert summary["agents"]["pokemon"]["attempts_without_gain"] == 1
    assert summary["actions_total"] == 4
    monitor.reset_agent(AgentRole.STORY)
    assert monitor.summary()["agents"]["story"]["attempts_without_gain"] == 0


def test_measured_gain_resets_the_action_counter() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, no_progress_action_limit=10)
    monitor.observe_actions(6)
    monitor.observe_navigation({"cells_known": 1})
    monitor.observe_navigation({"cells_known": 2})
    assert monitor.summary()["actions_since_progress"] == 0
    assert monitor.check() == []


def test_rolling_window_uses_min_progress_delta() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, window=3, min_progress_delta=1.0)
    monitor.observe_navigation({"cells_known": 5})
    monitor.observe_navigation({"cells_known": 5})
    assert monitor.check() == []  # window not full yet
    monitor.observe_navigation({"cells_known": 5})
    assert StagnationKind.NO_PROGRESS in monitor.check()
    monitor.observe_navigation({"cells_known": 6})
    assert monitor.check() == []


# --------------------------------------------------------------------------- #
# Repeated failures / messages
# --------------------------------------------------------------------------- #

def test_repeated_task_failure_is_detected_per_task_and_reset_by_success() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, repeated_failure_limit=3)
    monitor.observe_task_result(task_id="task_a", ok=False, detail="no target")
    monitor.observe_task_result(task_id="task_b", ok=False, detail="other")
    assert monitor.check() == []
    monitor.observe_task_result(task_id="task_a", ok=False, detail="no target again")
    assert monitor.check() == [StagnationKind.REPEATED_FAILURE]
    assert monitor.summary()["failures"]["task_failures"]["task_a"] == 2
    monitor.observe_task_result(task_id="task_a", ok=True)
    assert monitor.check() == []
    assert monitor.summary()["failures"]["task_successes"] == 1


def test_repeated_messages_are_detected() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, repeated_message_limit=3)
    monitor.observe_message("navigation:arrived")
    monitor.observe_message("navigation:arrived")
    assert monitor.check() == []
    monitor.observe_message("navigation:arrived")
    assert monitor.check() == [StagnationKind.REPEATED_MESSAGE]
    assert monitor.summary()["messages"]["worst_repeats"] == 3
    monitor.observe_message("")
    assert monitor.summary()["messages"]["distinct_keys"] == 1


# --------------------------------------------------------------------------- #
# Training / healing grace
# --------------------------------------------------------------------------- #

def test_training_grace_suppresses_rollback_triggers() -> None:
    clock = FakeClock()
    monitor = _monitor(
        clock,
        local_stagnation_steps=2,
        global_stagnation_seconds=10.0,
        training_grace_seconds=100.0,
    )
    monitor.observe_navigation({"cells_known": 1})
    monitor.observe_navigation({"cells_known": 1})
    clock.advance(20.0)
    monitor.observe_navigation({"cells_known": 1})
    monitor.mark_activity("training", detail="grinding levels")
    assert monitor.summary()["in_training_grace"] is True
    assert monitor.check() == []  # local + global suppressed
    clock.advance(99.0)
    assert monitor.check() == []
    clock.advance(2.0)  # grace expired
    assert monitor.summary()["in_training_grace"] is False
    signals = monitor.check()
    assert StagnationKind.GLOBAL_STAGNATION in signals
    assert StagnationKind.LOCAL_STAGNATION in signals


def test_training_grace_does_not_mask_hard_limits_or_loops() -> None:
    clock = FakeClock()
    monitor = _monitor(
        clock,
        training_grace_seconds=100.0,
        max_actions=5,
        repeated_message_limit=2,
    )
    monitor.mark_activity("healing")
    monitor.observe_actions(5)
    monitor.observe_message("loop")
    monitor.observe_message("loop")
    signals = monitor.check()
    assert StagnationKind.RESOURCE_EXHAUSTED in signals
    assert StagnationKind.REPEATED_MESSAGE in signals


def test_grace_suppresses_repeated_failure_but_not_resources() -> None:
    clock = FakeClock()
    monitor = _monitor(
        clock,
        training_grace_seconds=50.0,
        repeated_failure_limit=2,
        max_api_cost_usd=1.0,
    )
    monitor.mark_activity("training")
    monitor.observe_task_result(task_id="task_a", ok=False)
    monitor.observe_task_result(task_id="task_a", ok=False)
    assert monitor.check() == []
    monitor.observe_api(cost_usd=1.5)
    assert monitor.check() == [StagnationKind.RESOURCE_EXHAUSTED]


# --------------------------------------------------------------------------- #
# Resource / recovery limits
# --------------------------------------------------------------------------- #

def test_runtime_limit_triggers_resource_exhausted() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, max_runtime_s=30.0)
    assert monitor.check() == []
    clock.advance(30.0)
    assert monitor.check() == [StagnationKind.RESOURCE_EXHAUSTED]
    assert monitor.limits_exceeded() == StagnationKind.RESOURCE_EXHAUSTED


def test_action_and_api_cost_limits_trigger_resource_exhausted() -> None:
    clock = FakeClock()
    actions = _monitor(clock, max_actions=4)
    actions.observe_actions(3)
    assert actions.check() == []
    actions.observe_actions(1)
    assert actions.check() == [StagnationKind.RESOURCE_EXHAUSTED]

    cost = _monitor(clock, max_api_cost_usd=0.25)
    cost.observe_api(cost_usd=0.1, calls=1)
    assert cost.check() == []
    cost.observe_api(cost_usd=0.2, calls=1)
    assert cost.check() == [StagnationKind.RESOURCE_EXHAUSTED]
    summary = cost.summary()
    assert summary["api_cost_usd"] == pytest.approx(0.3)
    assert summary["api_calls"] == 2


def test_recovery_limit_signal() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, recovery_limit=2)
    monitor.reset_after_recovery(level=RecoveryLevel.L1_LOCAL)
    monitor.reset_after_recovery(level=RecoveryLevel.L3_ROLLBACK)
    assert monitor.check() == [StagnationKind.RECOVERY_LIMIT]
    assert monitor.limits_exceeded() == StagnationKind.RECOVERY_LIMIT
    assert monitor.summary()["recoveries"]["by_level"] == {"l1_local": 1, "l3_rollback": 1}


def test_reset_after_recovery_none_is_a_no_op() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, recovery_limit=2)
    monitor.reset_after_recovery(level="none")
    monitor.reset_after_recovery(level=RecoveryLevel.NONE)
    summary = monitor.summary()
    assert summary["recoveries"]["total"] == 0
    assert monitor.check() == []


# --------------------------------------------------------------------------- #
# Reset behaviour
# --------------------------------------------------------------------------- #

def test_reset_after_recovery_keeps_lifetime_counters() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=2, max_api_cost_usd=100.0)
    monitor.observe_actions(7)
    monitor.observe_api(cost_usd=0.5, calls=2)
    monitor.observe_navigation({"cells_known": 1})
    monitor.observe_navigation({"cells_known": 1})
    assert monitor.check() == [StagnationKind.LOCAL_STAGNATION]
    monitor.reset_after_recovery(level=RecoveryLevel.L1_LOCAL)
    summary = monitor.summary()
    assert summary["actions_total"] == 7
    assert summary["api_cost_usd"] == pytest.approx(0.5)
    assert summary["api_calls"] == 2
    assert summary["agents"]["navigation"]["attempts_without_gain"] == 0
    assert summary["recoveries"]["by_level"] == {"l1_local": 1}
    clock.advance(1.0)
    assert monitor.check() == []


def test_reset_clears_repeated_message_latch() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, repeated_message_limit=2)
    monitor.observe_message("m")
    monitor.observe_message("m")
    assert monitor.check() == [StagnationKind.REPEATED_MESSAGE]
    monitor.reset_after_recovery()
    assert monitor.check() == []
    monitor.observe_message("m")
    assert monitor.check() == []


# --------------------------------------------------------------------------- #
# Signal ordering / latch / snapshot
# --------------------------------------------------------------------------- #

def test_signals_are_deduped_and_ordered_by_severity() -> None:
    clock = FakeClock()
    monitor = _monitor(
        clock,
        local_stagnation_steps=2,
        repeated_failure_limit=2,
        repeated_message_limit=1,
        max_actions=3,
        no_progress_action_limit=3,
    )
    monitor.observe_actions(3, label="navigation")
    monitor.observe_message("m")
    monitor.observe_task_result(task_id="t", ok=False)
    monitor.observe_task_result(task_id="t", ok=False)
    signals = monitor.check()
    assert signals[0] is StagnationKind.RESOURCE_EXHAUSTED
    assert len(signals) == len(set(signals))
    # deterministic: same state -> same ordered list
    assert monitor.check() == signals
    order = {kind: index for index, kind in enumerate(SIGNAL_ORDER)}
    assert [order[kind] for kind in signals] == sorted(order[kind] for kind in signals)


def test_consume_signals_returns_and_clears_the_pending_latch() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=2)
    monitor.observe_navigation({"cells_known": 1})
    monitor.observe_navigation({"cells_known": 1})
    signals = monitor.consume_signals()
    assert signals == [StagnationKind.LOCAL_STAGNATION]
    assert monitor.pending_signals() == []
    monitor.consume_signals()
    assert monitor.summary()["consumed"] == 2


def test_snapshot_restore_roundtrip_reproduces_signals() -> None:
    clock = FakeClock()
    monitor = _monitor(
        clock,
        local_stagnation_steps=2,
        repeated_failure_limit=2,
        repeated_message_limit=2,
        max_actions=100,
    )
    monitor.observe_actions(5)
    monitor.observe_navigation({"cells_known": 3})
    monitor.observe_navigation({"cells_known": 3})
    monitor.observe_message("m")
    monitor.observe_message("m")
    monitor.observe_task_result(task_id="task_a", ok=False)
    monitor.observe_task_result(task_id="task_a", ok=False)
    snapshot = monitor.snapshot()
    json.dumps(snapshot)  # must be JSON-serialisable (checkpoint manifest)
    expected = monitor.check()

    other_clock = FakeClock(clock.now)
    restored = _monitor(other_clock, local_stagnation_steps=2, repeated_failure_limit=2,
                        repeated_message_limit=2, max_actions=100)
    restored.restore(snapshot)
    assert restored.check() == expected
    assert restored.summary()["actions_total"] == 5
    assert restored.summary()["navigation"]["totals"]["new_cells"] == 0.0


def test_restore_tolerates_junk_snapshots() -> None:
    clock = FakeClock()
    monitor = _monitor(clock, local_stagnation_steps=2)
    monitor.restore(None)  # type: ignore[arg-type]
    monitor.restore({"steps": "not-a-list", "agent_attempts": 7, "started_at": "x"})
    monitor.restore({"activities": {"training": "bad"}})
    monitor.observe_navigation({"cells_known": 1})
    monitor.observe_navigation({"cells_known": 1})
    assert StagnationKind.LOCAL_STAGNATION in monitor.check()
    json.dumps(monitor.snapshot())


def test_summary_reports_per_agent_and_limits() -> None:
    clock = FakeClock()
    monitor = _monitor(
        clock,
        local_stagnation_steps=10,
        global_stagnation_seconds=50.0,
        training_grace_seconds=5.0,
        max_runtime_s=100.0,
    )
    monitor.observe_story(new_npcs=1)
    summary = monitor.summary()
    assert summary["config"]["local_stagnation_steps"] == 10
    assert summary["limits"]["max_runtime_s"] == 100.0
    assert summary["story"]["new_npcs"] == 1
    assert summary["agents"]["story"]["gains"] == 1
    assert summary["window"]["size"] == 1
    assert summary["last_progress_at"] == clock.now
