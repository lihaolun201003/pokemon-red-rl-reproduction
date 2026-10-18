"""Unit tests for :mod:`llm_agent.guard` (owner: memory-guard).

Only contract primitives are used (synthetic frames from the shared conftest
fixtures, :class:`ActionOutcome`/``ToolResult`` objects); no PyBoy, no ROM, no
network, no API key.  Every detection rule, severity escalation,
``render_warnings`` formatting, reset and snapshot/restore are covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_agent.config import GuardConfig
from llm_agent.contracts import ActionOutcome, ActionStep, ScreenEvent, ToolResult
from llm_agent.guard import RULE_ORDER, GuardWarning, LoopGuard


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cfg(**overrides: int) -> GuardConfig:
    """GuardConfig with every rule disabled except the ones under test."""
    values = dict(
        unchanged_batch_limit=0,
        repeated_batch_limit=0,
        black_frame_limit=0,
        no_discovery_action_limit=0,
        same_screen_action_limit=0,
        max_consecutive_tool_errors=0,
        max_consecutive_api_failures=0,
    )
    values.update(overrides)
    return GuardConfig(**values)


def _outcome(
    make_frame,
    buttons: list[str],
    rgba: np.ndarray,
    *,
    changed: bool = False,
    stopped_reason: str = "completed",
) -> ActionOutcome:
    """Build a contract ActionOutcome with one step per button."""
    steps = [
        ActionStep(
            button=button,
            frame=make_frame(rgba, frame_index=index),
            screen_changed=changed,
            changed_ratio=0.4 if changed else 0.0,
            event=ScreenEvent.NONE,
        )
        for index, button in enumerate(buttons)
    ]
    return ActionOutcome(
        buttons=list(buttons),
        steps=steps,
        start_frame=0,
        end_frame=24 * len(steps),
        stopped_reason=stopped_reason,
    )


def _tool_error(name: str = "press_button") -> ToolResult:
    return ToolResult(
        tool_call_id="call-1",
        name=name,
        ok=False,
        summary="invalid arguments",
        error="missing required argument 'button'",
    )


def _tool_ok(name: str = "observe_game") -> ToolResult:
    return ToolResult(tool_call_id="call-2", name=name, ok=True, summary="ok")


def _kinds(guard: LoopGuard) -> list[str]:
    return [warning.kind for warning in guard.check()]


# --------------------------------------------------------------------------- #
# Healthy baseline
# --------------------------------------------------------------------------- #

def test_healthy_guard_has_no_warnings() -> None:
    guard = LoopGuard(GuardConfig())
    assert guard.check() == []
    assert guard.warnings() == []
    assert guard.needs_attention() is False
    assert guard.critical() is False
    assert guard.render_warnings() == ""


def test_healthy_activity_does_not_warn(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(GuardConfig())
    other = sample_rgba.copy()
    other[:8, :8, :3] = 0
    batches = [["up", "right"], ["down"], ["left"], ["a"], ["right", "right"]]
    for index, buttons in enumerate(batches):
        guard.on_action_outcome(
            _outcome(make_frame, buttons, sample_rgba if index % 2 else other, changed=True)
        )
        guard.on_observation(make_frame(sample_rgba if index % 2 else other), ScreenEvent.NONE, changed=True)
        guard.on_tool_result(_tool_ok())
        guard.on_llm_call(ok=True)
    assert guard.check() == []


# --------------------------------------------------------------------------- #
# Unchanged movement batches
# --------------------------------------------------------------------------- #

def test_unchanged_movement_batch_warning_and_escalation(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(unchanged_batch_limit=2))
    outcome = _outcome(make_frame, ["up"], sample_rgba, changed=False)
    guard.on_action_outcome(outcome)
    warnings = guard.check()
    assert [warning.kind for warning in warnings] == ["unchanged_movement"]
    assert warnings[0].severity == "info"  # single blocked walk: advisory only

    guard.on_action_outcome(outcome)
    warnings = guard.check()
    assert [warning.kind for warning in warnings] == ["unchanged_movement"]
    assert warnings[0].severity == "warning"
    assert "different direction" in warnings[0].suggestion.lower()
    assert "restore_checkpoint" in warnings[0].suggestion
    assert "no visible screen change" in warnings[0].message

    guard.on_action_outcome(outcome)
    guard.on_action_outcome(outcome)
    assert guard.critical() is True
    assert guard.warnings()[0].severity == "critical"


def test_unchanged_movement_info_escalation(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(unchanged_batch_limit=3))
    outcome = _outcome(make_frame, ["left"], sample_rgba, changed=False)
    guard.on_action_outcome(outcome)
    guard.on_action_outcome(outcome)
    warnings = guard.warnings()
    assert [warning.kind for warning in warnings] == ["unchanged_movement"]
    assert warnings[0].severity == "info"
    guard.on_action_outcome(outcome)
    assert guard.warnings()[0].severity == "warning"
    for _ in range(3):
        guard.on_action_outcome(outcome)
    assert guard.warnings()[0].severity == "critical"


def test_changed_movement_resets_unchanged_counter(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(unchanged_batch_limit=3))
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=True))
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    assert guard.check() == []


def test_non_movement_batches_do_not_count_as_unchanged(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(unchanged_batch_limit=2))
    for _ in range(5):
        guard.on_action_outcome(_outcome(make_frame, ["a"], sample_rgba, changed=False))
    assert guard.check() == []


def test_zero_threshold_disables_rule(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(unchanged_batch_limit=0))
    for _ in range(10):
        guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    assert guard.check() == []


# --------------------------------------------------------------------------- #
# Repeated identical batches
# --------------------------------------------------------------------------- #

def test_repeated_batch_warning_and_reset(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(repeated_batch_limit=3))
    batch = _outcome(make_frame, ["b", "b"], sample_rgba, changed=True)
    guard.on_action_outcome(batch)
    guard.on_action_outcome(batch)
    assert guard.warnings()[0].severity == "info"  # approaching the limit
    guard.on_action_outcome(batch)
    warnings = guard.warnings()
    assert [warning.kind for warning in warnings] == ["repeated_batch"]
    assert warnings[0].severity == "warning"
    assert "3 times" in warnings[0].message

    # a different batch resets the streak
    guard.on_action_outcome(_outcome(make_frame, ["a"], sample_rgba, changed=True))
    assert guard.check() == []


def test_repeated_batch_critical_escalation(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(repeated_batch_limit=2))
    batch = _outcome(make_frame, ["start"], sample_rgba, changed=True)
    for _ in range(4):
        guard.on_action_outcome(batch)
    assert guard.warnings()[0].kind == "repeated_batch"
    assert guard.warnings()[0].severity == "critical"


# --------------------------------------------------------------------------- #
# Black frames
# --------------------------------------------------------------------------- #

def test_black_frames_warning_and_reset(make_frame, black_rgba, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(black_frame_limit=2))
    black = make_frame(black_rgba)
    guard.on_observation(black, ScreenEvent.TRANSITION, changed=False)
    warnings = guard.warnings()
    assert [warning.kind for warning in warnings] == ["black_frame"]
    assert warnings[0].severity == "info"

    guard.on_observation(black, ScreenEvent.TRANSITION, changed=False)
    assert guard.warnings()[0].severity == "warning"
    assert "wait_frames" in guard.warnings()[0].suggestion

    # a normal frame clears the counter
    guard.on_observation(make_frame(sample_rgba), ScreenEvent.NONE, changed=True)
    assert guard.check() == []


def test_black_frames_critical_escalation(make_frame, black_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(black_frame_limit=2))
    black = make_frame(black_rgba)
    for _ in range(4):
        guard.on_observation(black, ScreenEvent.TRANSITION, changed=False)
    assert guard.warnings()[0].kind == "black_frame"
    assert guard.warnings()[0].severity == "critical"
    assert guard.critical() is True


# --------------------------------------------------------------------------- #
# Same-screen action overrun
# --------------------------------------------------------------------------- #

def test_same_screen_action_overrun(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(same_screen_action_limit=3))
    for _ in range(4):
        guard.on_action_outcome(_outcome(make_frame, ["a"], sample_rgba, changed=False))
    warnings = guard.warnings()
    assert [warning.kind for warning in warnings] == ["same_screen"]
    assert warnings[0].severity == "warning"
    assert "same screen" in warnings[0].message


def test_screen_change_resets_same_screen_counter(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(same_screen_action_limit=3))
    for _ in range(2):
        guard.on_action_outcome(_outcome(make_frame, ["a"], sample_rgba, changed=False))
    other = sample_rgba.copy()
    other[:16, :16, :3] = 0
    guard.on_action_outcome(_outcome(make_frame, ["right"], other, changed=True))
    guard.on_observation(make_frame(other), ScreenEvent.NONE, changed=True)
    assert guard.check() == []


# --------------------------------------------------------------------------- #
# No discovery for N actions
# --------------------------------------------------------------------------- #

def test_no_discovery_warning_and_reset(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(no_discovery_action_limit=4))
    batch = _outcome(make_frame, ["right", "right"], sample_rgba, changed=True)
    guard.on_action_outcome(batch)
    guard.on_action_outcome(batch)
    warnings = guard.warnings()
    assert [warning.kind for warning in warnings] == ["no_discovery"]
    assert warnings[0].severity == "warning"
    assert "record_discovery" in warnings[0].suggestion

    # a failed record_discovery does not reset the clock
    guard.on_tool_result(_tool_error("record_discovery"))
    assert any(warning.kind == "no_discovery" for warning in guard.warnings())

    # a successful record_discovery resets it
    guard.on_tool_result(_tool_ok("record_discovery"))
    assert guard.check() == []


# --------------------------------------------------------------------------- #
# Tool errors
# --------------------------------------------------------------------------- #

def test_consecutive_tool_errors(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(max_consecutive_tool_errors=2))
    guard.on_tool_result(_tool_error())
    guard.on_tool_result(_tool_error())
    warnings = guard.warnings()
    assert [warning.kind for warning in warnings] == ["tool_errors"]
    assert warnings[0].severity == "warning"
    assert "press_button" in warnings[0].message

    guard.on_tool_result(_tool_ok())
    assert guard.check() == []

    for _ in range(4):
        guard.on_tool_result(_tool_error())
    assert guard.warnings()[0].severity == "critical"


# --------------------------------------------------------------------------- #
# Consecutive API failures
# --------------------------------------------------------------------------- #

def test_consecutive_api_failures() -> None:
    guard = LoopGuard(_cfg(max_consecutive_api_failures=2))
    guard.on_llm_call(ok=False)
    guard.on_llm_call(ok=False)
    warnings = guard.warnings()
    assert [warning.kind for warning in warnings] == ["api_failures"]
    assert warnings[0].severity == "warning"
    assert "API calls failed" in warnings[0].message

    guard.on_llm_call(ok=True)
    assert guard.check() == []

    for _ in range(4):
        guard.on_llm_call(ok=False)
    assert guard.warnings()[0].severity == "critical"


# --------------------------------------------------------------------------- #
# Ordering / severity
# --------------------------------------------------------------------------- #

def test_warning_order_is_deterministic(make_frame, sample_rgba, black_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(
        _cfg(
            unchanged_batch_limit=1,
            black_frame_limit=1,
            max_consecutive_tool_errors=1,
            max_consecutive_api_failures=1,
        )
    )
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    guard.on_observation(make_frame(black_rgba), ScreenEvent.NONE, changed=False)
    guard.on_tool_result(_tool_error())
    guard.on_llm_call(ok=False)

    kinds = _kinds(guard)
    assert kinds == ["unchanged_movement", "black_frame", "tool_errors", "api_failures"]
    assert kinds == _kinds(guard)  # pure + deterministic
    assert guard.warnings() == guard.check()


def test_critical_warnings_sort_first(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(unchanged_batch_limit=1, max_consecutive_api_failures=1))
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    guard.on_llm_call(ok=False)
    guard.on_llm_call(ok=False)  # 2 >= 2 * limit -> critical
    warnings = guard.check()
    assert warnings[0].kind == "api_failures"
    assert warnings[0].severity == "critical"
    assert warnings[1].kind == "unchanged_movement"
    assert warnings[1].severity == "warning"


def test_info_warning_needs_attention_but_not_critical(make_frame, black_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(black_frame_limit=2))
    guard.on_observation(make_frame(black_rgba), ScreenEvent.TRANSITION, changed=False)
    assert guard.needs_attention() is True
    assert guard.critical() is False
    assert guard.warnings()[0].severity == "info"


# --------------------------------------------------------------------------- #
# render_warnings
# --------------------------------------------------------------------------- #

def test_render_warnings_format(make_frame, sample_rgba, black_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(black_frame_limit=1, unchanged_batch_limit=1))
    guard.on_observation(make_frame(black_rgba), ScreenEvent.TRANSITION, changed=False)
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    text = guard.render_warnings()
    assert text.startswith("GUARD WARNINGS (2):")
    assert "[warning]" in text
    assert "black_frame" in text
    assert "unchanged_movement" in text
    assert "Suggestion:" in text
    assert "restore_checkpoint" in text


def test_render_warnings_respects_max_chars(make_frame, black_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(black_frame_limit=1, max_consecutive_api_failures=1))
    guard.on_observation(make_frame(black_rgba), ScreenEvent.TRANSITION, changed=False)
    guard.on_llm_call(ok=False)
    full = guard.render_warnings(4000)
    assert len(full) > 100
    for limit in (20, 60, 120, 300):
        rendered = guard.render_warnings(limit)
        assert 0 < len(rendered) <= limit
    assert guard.render_warnings(0) == ""


def test_render_warnings_empty_when_healthy() -> None:
    guard = LoopGuard(GuardConfig())
    assert guard.render_warnings() == ""
    assert guard.render_warnings(500) == ""


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #

def test_reset_after_recovery(make_frame, sample_rgba, black_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(black_frame_limit=1, max_consecutive_tool_errors=1))
    guard.on_observation(make_frame(black_rgba), ScreenEvent.NONE, changed=False)
    guard.on_tool_result(_tool_error())
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    assert guard.needs_attention() is True

    guard.reset_after_recovery()
    assert guard.check() == []
    assert guard.render_warnings() == ""
    assert guard.stats()["actions_total"] == 1  # lifetime totals are kept


def test_full_reset_zeroes_totals() -> None:
    guard = LoopGuard(_cfg(black_frame_limit=1))
    guard.on_tool_result(_tool_error())
    guard.reset()
    assert guard.check() == []
    assert guard.stats()["actions_total"] == 0
    assert guard.stats()["batches_total"] == 0


# --------------------------------------------------------------------------- #
# Snapshot / restore
# --------------------------------------------------------------------------- #

def test_snapshot_restore_roundtrip(make_frame, sample_rgba, black_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(
        _cfg(black_frame_limit=2, max_consecutive_tool_errors=2, unchanged_batch_limit=2)
    )
    black = make_frame(black_rgba)
    guard.on_observation(black, ScreenEvent.NONE, changed=False)
    guard.on_observation(black, ScreenEvent.NONE, changed=False)
    guard.on_tool_result(_tool_error())
    guard.on_tool_result(_tool_error())
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    guard.on_action_outcome(_outcome(make_frame, ["up"], sample_rgba, changed=False))
    before_text = guard.render_warnings(4000)
    before_stats = guard.stats()

    snapshot = guard.snapshot()
    assert snapshot["warnings"]
    assert all(isinstance(item, dict) for item in snapshot["warnings"])

    # mutate the live state after the snapshot
    guard.on_tool_result(_tool_error())
    guard.on_observation(black, ScreenEvent.NONE, changed=False)
    assert guard.render_warnings(4000) != before_text

    guard.restore(snapshot)
    assert guard.render_warnings(4000) == before_text
    assert guard.stats() == before_stats

    # snapshot data is a deep copy
    if snapshot["last_batch"]:
        snapshot["last_batch"].append("left")
        snapshot["warnings"].append({"kind": "bogus", "severity": "critical"})
    guard.restore(snapshot)
    assert guard.stats() == before_stats


def test_restore_tolerates_junk(make_frame, sample_rgba) -> None:  # noqa: ANN001
    guard = LoopGuard(_cfg(black_frame_limit=1))
    guard.on_observation(make_frame(sample_rgba), ScreenEvent.NONE, changed=True)
    guard.restore(
        {
            "unchanged_batches": "junk",
            "last_batch": 42,
            "screen_signature": 123,
            "black_frames": None,
            "consecutive_api_failures": -5,
        }
    )
    assert guard.check() == []
    guard.restore("not a mapping")  # type: ignore[arg-type] - defensive contract
    assert guard.check() == []


def test_guard_warning_dataclass_roundtrip() -> None:
    warning = GuardWarning("black_frame", "warning", "message", "suggestion")
    parsed = GuardWarning.from_dict(warning.to_dict())
    assert parsed == warning
    assert GuardWarning.from_dict({"severity": "weird", "kind": "k"}).severity == "warning"
    assert GuardWarning.from_dict({"kind": ""}) is None
    assert GuardWarning.from_dict("junk") is None


def test_rule_order_constant_matches_checks() -> None:
    assert RULE_ORDER == (
        "unchanged_movement",
        "repeated_batch",
        "black_frame",
        "same_screen",
        "no_discovery",
        "tool_errors",
        "api_failures",
    )
