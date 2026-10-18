"""Unit tests for :mod:`llm_agent.budget`.

Everything is deterministic and offline: a fake unix clock drives peak/off-peak
pricing and hourly windows, the sleep function only records durations, and all
state lives in ``tmp_path``.  No test ever sleeps for real.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pytest

from llm_agent import contracts
from llm_agent.budget import (
    DEFAULT_BUDGET_TOTAL_PATH,
    IMAGE_TOKEN_CAP,
    BudgetTracker,
    is_peak_hour,
    peak_multiplier_at,
    usage_cost_usd,
)
from llm_agent.config import ApiConfig, BudgetConfig

UTC = dt.timezone.utc
#: Tuesday 2026-10-20 12:00 UTC -> off-peak.
OFF_PEAK_TS = dt.datetime(2026, 10, 20, 12, 0, tzinfo=UTC).timestamp()
#: Monday 2026-10-19 02:30 UTC -> peak window [01:00, 04:00).
PEAK_TS = dt.datetime(2026, 10, 19, 2, 30, tzinfo=UTC).timestamp()
#: Saturday 2026-10-17 02:30 UTC -> weekend, always off-peak.
WEEKEND_TS = dt.datetime(2026, 10, 17, 2, 30, tzinfo=UTC).timestamp()


class FakeClock:
    """Deterministic unix clock for peak pricing / hourly windows."""

    def __init__(self, now: float = OFF_PEAK_TS) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class SleepRecorder:
    """Sleep stand-in that records durations instead of blocking."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(float(seconds))


def make_tracker(
    tmp_path: Path,
    *,
    api_cfg: ApiConfig | None = None,
    budget_cfg: BudgetConfig | None = None,
    clock: FakeClock | None = None,
    sleep: Any = None,
    with_total: bool = True,
    state_dir: Path | None = None,
) -> BudgetTracker:
    """Build a tracker rooted in ``tmp_path`` (fresh run + optional total file).

    ``state_dir`` selects the *run* directory; ``tmp_path`` is always the root
    of the shared global total file so separate runs can be simulated.
    """
    api = api_cfg or ApiConfig()
    budget = budget_cfg or BudgetConfig(
        max_usd_per_run=2.0,
        max_usd_total=10.0,
        max_calls_per_run=3000,
        max_calls_per_hour=240,
        min_remaining_usd=0.0,
    )
    state_path = (state_dir or tmp_path) / "audit" / "budget.json"
    total_path = (tmp_path / "budget_total.json") if with_total else None
    return BudgetTracker(
        budget,
        api,
        state_path,
        total_path=total_path,
        clock=clock or FakeClock(OFF_PEAK_TS),
        sleep=sleep if sleep is not None else SleepRecorder(),
    )


def usage(prompt: int = 0, completion: int = 0, *, hit: int | None = None, miss: int | None = None,
          total: int | None = None) -> contracts.ApiUsage:
    return contracts.ApiUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total if total is not None else prompt + completion,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


# --------------------------------------------------------------------------- #
# Peak / off-peak pricing helpers
# --------------------------------------------------------------------------- #

def test_peak_hour_windows() -> None:
    api = ApiConfig()
    monday = dt.date(2026, 10, 19)
    assert monday.weekday() == 0

    def ts(hour: int, minute: int = 0) -> float:
        return dt.datetime(
            monday.year, monday.month, monday.day, hour, minute, tzinfo=UTC
        ).timestamp()

    assert is_peak_hour(api, ts(0, 30)) is False
    assert is_peak_hour(api, ts(1, 0)) is True
    assert is_peak_hour(api, ts(3, 59)) is True
    assert is_peak_hour(api, ts(4, 0)) is False
    assert is_peak_hour(api, ts(6, 0)) is True
    assert is_peak_hour(api, ts(9, 59)) is True
    assert is_peak_hour(api, ts(10, 0)) is False
    assert is_peak_hour(api, ts(12, 0)) is False
    assert is_peak_hour(api, WEEKEND_TS) is False  # Saturday
    assert peak_multiplier_at(api, ts(2, 0)) == 2.0
    assert peak_multiplier_at(api, ts(12, 0)) == 1.0


def test_peak_hour_windows_are_configurable() -> None:
    api = ApiConfig(peak_hours_utc=((12, 14),))
    noon = dt.datetime(2026, 10, 19, 13, 0, tzinfo=UTC).timestamp()
    assert is_peak_hour(api, noon) is True
    # A custom window replaces the defaults: 02:30 Monday is now off-peak.
    assert is_peak_hour(api, PEAK_TS) is False


def test_usage_cost_usd_cache_split_semantics() -> None:
    api = ApiConfig()
    # No cache info at all -> every prompt token is a cache miss (expensive case).
    assert usage_cost_usd(api, usage(1_000_000)) == pytest.approx(0.15)
    # Only a hit count -> the remainder of prompt_tokens counts as miss.
    assert usage_cost_usd(api, usage(1_000_000, hit=200_000)) == pytest.approx(
        (200_000 * 0.003 + 800_000 * 0.15) / 1e6
    )
    # Explicit split.
    assert usage_cost_usd(
        api, usage(1_000_000, 1_000_000, hit=400_000, miss=600_000)
    ) == pytest.approx((400_000 * 0.003 + 600_000 * 0.15 + 1_000_000 * 0.6) / 1e6)
    # Split that does not add up -> the remainder is charged as misses.
    assert usage_cost_usd(api, usage(1_000, hit=100, miss=100)) == pytest.approx(
        (100 * 0.003 + 900 * 0.15) / 1e6
    )
    # Explicit multiplier is applied.
    assert usage_cost_usd(api, usage(1_000_000, hit=0, miss=1_000_000), peak_multiplier=2.0) == pytest.approx(0.30)


# --------------------------------------------------------------------------- #
# Cost math through record_success
# --------------------------------------------------------------------------- #

def test_off_peak_cost_math(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path, clock=FakeClock(OFF_PEAK_TS))
    assert tracker.is_peak() is False
    cost = tracker.record_success(
        usage(1_000_000, 1_000_000, hit=0, miss=1_000_000), model="deepseek-flash"
    )
    assert cost == pytest.approx(0.75)  # 0.15 input + 0.60 output
    assert tracker.state().cost_usd == pytest.approx(0.75)
    assert tracker.state().calls == 1
    assert tracker.state().prompt_tokens == 1_000_000
    assert tracker.state().completion_tokens == 1_000_000


def test_peak_cost_is_doubled(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path, clock=FakeClock(PEAK_TS))
    assert tracker.is_peak() is True
    assert tracker.peak_multiplier() == 2.0
    cost = tracker.record_success(
        usage(1_000_000, 1_000_000, hit=0, miss=1_000_000)
    )
    assert cost == pytest.approx(1.50)


def test_weekend_is_off_peak(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path, clock=FakeClock(WEEKEND_TS))
    assert tracker.is_peak() is False
    cost = tracker.record_success(usage(1_000_000, hit=0, miss=1_000_000))
    assert cost == pytest.approx(0.15)


def test_cost_for_usage_at_override(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path, clock=FakeClock(OFF_PEAK_TS))
    record = usage(1_000_000, hit=0, miss=1_000_000)
    off = tracker.cost_for_usage(record)
    peak = tracker.cost_for_usage(record, at=PEAK_TS)
    assert off == pytest.approx(0.15)
    assert peak == pytest.approx(0.30)


# --------------------------------------------------------------------------- #
# Pre-call worst-case estimate
# --------------------------------------------------------------------------- #

def test_estimate_formula(tmp_path: Path) -> None:
    api = ApiConfig()
    budget = BudgetConfig(max_usd_per_run=2.0, min_remaining_usd=0.0)
    tracker = make_tracker(tmp_path, api_cfg=api, budget_cfg=budget, clock=FakeClock(OFF_PEAK_TS))
    text_chars = 4000
    text_tokens = math.ceil(text_chars / budget.chars_per_token_estimate)
    prompt_tokens = text_tokens + 16  # no images + chat framing overhead
    raw = (prompt_tokens * api.price_input_cache_miss_per_mtok + api.max_tokens * api.price_output_per_mtok) / 1e6
    expected = raw * budget.estimate_padding
    assert tracker.estimate_call_cost(text_chars=text_chars, image_count=0) == pytest.approx(expected)


def test_estimate_includes_images_and_caps_at_1024(tmp_path: Path) -> None:
    api = ApiConfig()
    tracker = make_tracker(tmp_path, api_cfg=api, clock=FakeClock(OFF_PEAK_TS))
    no_image = tracker.estimate_call_cost(text_chars=1000, image_count=0)
    one_token_image = tracker.estimate_call_cost(
        text_chars=1000, image_count=1, image_width=28, image_height=28
    )
    delta_one = one_token_image - no_image
    expected_delta = 1 * api.price_input_cache_miss_per_mtok / 1e6 * BudgetConfig().estimate_padding
    assert delta_one == pytest.approx(expected_delta)

    two_images = tracker.estimate_call_cost(
        text_chars=1000, image_count=2, image_width=28, image_height=28
    )
    assert two_images - no_image == pytest.approx(2 * expected_delta)

    huge = tracker.estimate_call_cost(
        text_chars=1000, image_count=1, image_width=100_000, image_height=100_000
    )
    expected_huge_delta = (
        IMAGE_TOKEN_CAP * api.price_input_cache_miss_per_mtok / 1e6 * BudgetConfig().estimate_padding
    )
    assert huge - no_image == pytest.approx(expected_huge_delta)


def test_estimate_scales_with_text_and_uses_peak_pricing(tmp_path: Path) -> None:
    api = ApiConfig()
    off = make_tracker(tmp_path / "off", api_cfg=api, clock=FakeClock(OFF_PEAK_TS))
    peak = make_tracker(tmp_path / "peak", api_cfg=api, clock=FakeClock(PEAK_TS))
    small = off.estimate_call_cost(text_chars=1000, image_count=0)
    large = off.estimate_call_cost(text_chars=100_000, image_count=0)
    assert large > small
    assert peak.estimate_call_cost(text_chars=1000, image_count=0) == pytest.approx(2.0 * small)


def test_estimate_rejects_negative_inputs(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path)
    with pytest.raises(ValueError):
        tracker.estimate_call_cost(text_chars=-1)
    with pytest.raises(ValueError):
        tracker.estimate_call_cost(text_chars=0, image_count=-1)


# --------------------------------------------------------------------------- #
# Pre-call budget checks
# --------------------------------------------------------------------------- #

def test_check_before_call_allows_at_start(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path)
    allowed, reason = tracker.check_before_call(estimated_cost_usd=0.01)
    assert allowed is True
    assert reason == ""
    assert tracker.last_estimate_usd == pytest.approx(0.01)
    assert tracker.limits_exceeded() is None
    assert tracker.near_limit() is None


def test_run_budget_refusal(tmp_path: Path) -> None:
    budget = BudgetConfig(max_usd_per_run=0.01, min_remaining_usd=0.0)
    tracker = make_tracker(tmp_path, budget_cfg=budget)
    allowed, reason = tracker.check_before_call(estimated_cost_usd=0.02)
    assert allowed is False
    assert "run budget" in reason
    assert "$" in reason

    allowed, reason = tracker.check_before_call(estimated_cost_usd=0.005)
    assert allowed is True

    tracker.record_success(usage(100_000, hit=0, miss=100_000))  # $0.015
    assert tracker.state().cost_usd > 0.01
    allowed, reason = tracker.check_before_call(estimated_cost_usd=0.0)
    assert allowed is False
    assert "run budget" in reason
    assert tracker.limits_exceeded() is not None


def test_min_remaining_reserve(tmp_path: Path) -> None:
    budget = BudgetConfig(max_usd_per_run=1.0, min_remaining_usd=0.1)
    tracker = make_tracker(tmp_path, budget_cfg=budget)
    tracker.record_success(usage(5_000_000, hit=0, miss=5_000_000))  # $0.75 spent
    assert tracker.check_before_call(estimated_cost_usd=0.1)[0] is True
    allowed, reason = tracker.check_before_call(estimated_cost_usd=0.2)
    assert allowed is False
    assert "run budget" in reason


def test_total_budget_across_runs(tmp_path: Path) -> None:
    budget = BudgetConfig(max_usd_per_run=10.0, max_usd_total=0.5, min_remaining_usd=0.0)
    clock = FakeClock(OFF_PEAK_TS)
    first = make_tracker(tmp_path, budget_cfg=budget, clock=clock)
    first.record_success(usage(2_000_000, hit=0, miss=2_000_000))  # $0.30

    second = make_tracker(tmp_path, budget_cfg=budget, clock=clock)
    assert second.total_state() is not None
    assert second.total_state().cost_usd == pytest.approx(0.30)
    allowed, reason = second.check_before_call(estimated_cost_usd=0.25)
    assert allowed is False
    assert "total budget" in reason
    assert second.check_before_call(estimated_cost_usd=0.1)[0] is True


def test_global_cap_disabled_without_total_path(tmp_path: Path) -> None:
    budget = BudgetConfig(max_usd_per_run=10.0, max_usd_total=0.000001, min_remaining_usd=0.0)
    tracker = make_tracker(tmp_path, budget_cfg=budget, with_total=False)
    assert tracker.total_state() is None
    assert tracker.check_before_call(estimated_cost_usd=5.0)[0] is True


def test_calls_per_run_limit(tmp_path: Path) -> None:
    budget = BudgetConfig(max_calls_per_run=2, max_calls_per_hour=0, min_remaining_usd=0.0)
    tracker = make_tracker(tmp_path, budget_cfg=budget)
    tracker.record_success(usage(10))
    assert tracker.check_before_call(estimated_cost_usd=0.0)[0] is True
    tracker.record_success(usage(10))
    allowed, reason = tracker.check_before_call(estimated_cost_usd=0.0)
    assert allowed is False
    assert "calls per run" in reason
    assert tracker.limits_exceeded() is not None


def test_calls_per_hour_limit_and_window(tmp_path: Path) -> None:
    clock = FakeClock(OFF_PEAK_TS)
    budget = BudgetConfig(max_calls_per_run=1000, max_calls_per_hour=2, min_remaining_usd=0.0)
    tracker = make_tracker(tmp_path, budget_cfg=budget, clock=clock)
    tracker.record_success(usage(10))
    tracker.record_success(usage(10))
    allowed, reason = tracker.check_before_call(estimated_cost_usd=0.0)
    assert allowed is False
    assert "calls per hour" in reason
    assert tracker.calls_last_hour() == 2
    assert tracker.seconds_until_call_allowed() == pytest.approx(3600.0)

    clock.advance(3601)
    assert tracker.calls_last_hour() == 0
    assert tracker.check_before_call(estimated_cost_usd=0.0) == (True, "")
    assert tracker.seconds_until_call_allowed() == 0.0


def test_wait_until_call_allowed_uses_injected_sleep(tmp_path: Path) -> None:
    clock = FakeClock(OFF_PEAK_TS)
    sleeper = SleepRecorder()
    budget = BudgetConfig(max_calls_per_run=1000, max_calls_per_hour=1, min_remaining_usd=0.0)
    tracker = make_tracker(tmp_path, budget_cfg=budget, clock=clock, sleep=sleeper)
    tracker.record_success(usage(10))
    waited = tracker.wait_until_call_allowed(max_wait_s=10.0)
    assert waited == pytest.approx(10.0)
    assert sleeper.calls == [pytest.approx(10.0)]
    clock.advance(3601)
    sleeper.calls.clear()
    assert tracker.wait_until_call_allowed() == 0.0
    assert sleeper.calls == []


def test_zero_limits_are_disabled(tmp_path: Path) -> None:
    budget = BudgetConfig(
        max_usd_per_run=0,
        max_usd_total=0,
        max_calls_per_run=0,
        max_calls_per_hour=0,
        min_remaining_usd=0.0,
    )
    tracker = make_tracker(tmp_path, budget_cfg=budget)
    assert tracker.check_before_call(estimated_cost_usd=123.0)[0] is True
    assert tracker.limits_exceeded() is None
    assert tracker.near_limit() is None
    tracker.record_success(usage(10))
    assert tracker.calls_last_hour() == 0
    assert tracker.check_before_call(estimated_cost_usd=1e6)[0] is True


def test_near_limit_warning(tmp_path: Path) -> None:
    budget = BudgetConfig(
        max_usd_per_run=0.5, warn_at_fraction=0.5, min_remaining_usd=0.0
    )
    tracker = make_tracker(tmp_path, budget_cfg=budget)
    tracker.record_success(usage(2_000_000, hit=0, miss=2_000_000))  # $0.30 = 60%
    reason = tracker.near_limit()
    assert reason is not None
    assert "run budget" in reason
    assert tracker.limits_exceeded() is None
    assert tracker.check_before_call(estimated_cost_usd=0.0)[0] is True


def test_limits_exceeded_ignores_reserve_but_check_enforces_it(tmp_path: Path) -> None:
    budget = BudgetConfig(max_usd_per_run=0.01, min_remaining_usd=0.02)
    tracker = make_tracker(tmp_path, budget_cfg=budget)
    # Nothing has been spent, so no hard limit is "hit"...
    assert tracker.limits_exceeded() is None
    # ...but the reserve already blocks any new call.
    allowed, reason = tracker.check_before_call(estimated_cost_usd=0.0)
    assert allowed is False
    assert "run budget" in reason


def test_default_budget_total_path_is_global() -> None:
    assert DEFAULT_BUDGET_TOTAL_PATH.name == "budget_total.json"
    assert DEFAULT_BUDGET_TOTAL_PATH.parent.name == "llm_agent"
    assert DEFAULT_BUDGET_TOTAL_PATH.parent.parent.name == "artifacts"


# --------------------------------------------------------------------------- #
# Hourly accounting
# --------------------------------------------------------------------------- #

def test_hourly_cost_uses_recorded_costs(tmp_path: Path) -> None:
    clock = FakeClock(OFF_PEAK_TS)
    budget = BudgetConfig(max_calls_per_run=1000, max_calls_per_hour=1000, min_remaining_usd=0.0)
    tracker = make_tracker(tmp_path, budget_cfg=budget, clock=clock)
    tracker.record_success(usage(1_000_000, hit=0, miss=1_000_000))  # $0.15 at t
    clock.advance(100)
    tracker.record_success(usage(2_000_000, hit=0, miss=2_000_000))  # $0.30 at t+100
    assert tracker.hourly_cost() == pytest.approx(0.45)
    clock.advance(3501)  # now = t + 3601 -> only the t+100 call is in the window
    assert tracker.hourly_cost() == pytest.approx(0.30)
    clock.advance(3600)
    assert tracker.hourly_cost() == pytest.approx(0.0)


def test_hourly_cost_roundtrip_survives_reload(tmp_path: Path) -> None:
    clock = FakeClock(OFF_PEAK_TS)
    tracker = make_tracker(tmp_path, clock=clock)
    tracker.record_success(usage(1_000_000, hit=0, miss=1_000_000))
    clock.advance(60)
    tracker.record_success(usage(500_000, hit=0, miss=500_000))
    expected = tracker.hourly_cost()
    reloaded = make_tracker(tmp_path, clock=clock)
    assert reloaded.hourly_cost() == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Persistence, corruption recovery and roundtrip
# --------------------------------------------------------------------------- #

def test_persistence_roundtrip(tmp_path: Path) -> None:
    clock = FakeClock(OFF_PEAK_TS)
    tracker = make_tracker(tmp_path, clock=clock)
    tracker.record_success(usage(1_000_000, 500_000, hit=200_000, miss=800_000), model="deepseek-flash")
    tracker.record_failure(status=429, retryable=True)
    before = tracker.snapshot()

    assert tracker.state_path.is_file()
    assert tracker.state_path.parent.is_dir()
    assert tracker.total_path is not None and tracker.total_path.is_file()
    on_disk = json.loads(tracker.state_path.read_text(encoding="utf-8"))
    assert on_disk["version"] == 1
    assert on_disk["kind"] == "run"

    restored = make_tracker(tmp_path, clock=FakeClock(OFF_PEAK_TS))
    state = restored.state()
    assert state.calls == 1
    assert state.prompt_tokens == 1_000_000
    assert state.completion_tokens == 500_000
    assert state.cache_hit_tokens == 200_000
    assert state.cost_usd == pytest.approx(before["run"]["cost_usd"], rel=1e-6)
    assert restored.hourly_cost() == pytest.approx(tracker.hourly_cost())
    assert restored.total_state() is not None
    assert restored.total_state().calls == 1


def test_missing_files_are_created_on_load(tmp_path: Path) -> None:
    clock = FakeClock(OFF_PEAK_TS)
    tracker = make_tracker(tmp_path, clock=clock)
    tracker.load()
    assert tracker.state_path.is_file()
    assert tracker.total_path is not None and tracker.total_path.is_file()
    assert tracker.state().started_at == pytest.approx(OFF_PEAK_TS)
    payload = json.loads(tracker.state_path.read_text(encoding="utf-8"))
    assert payload["state"]["calls"] == 0


def test_corrupt_state_file_is_backed_up_and_reset(tmp_path: Path) -> None:
    state_path = tmp_path / "audit" / "budget.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{this is not json", encoding="utf-8")
    tracker = make_tracker(tmp_path)
    state = tracker.state()
    assert state.calls == 0
    assert state.cost_usd == 0.0
    assert len(tracker.recovery_events) >= 1
    event = tracker.recovery_events[0]
    assert str(state_path) == event["path"]
    assert event["backup"]
    backups = list(state_path.parent.glob("budget.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{this is not json"
    # The replacement file is valid JSON and the tracker keeps working.
    json.loads(state_path.read_text(encoding="utf-8"))
    tracker.record_success(usage(10))
    assert tracker.state().calls == 1
    assert tracker.last_save_error == ""


def test_corrupt_total_file_is_backed_up_and_reset(tmp_path: Path) -> None:
    total_path = tmp_path / "budget_total.json"
    total_path.parent.mkdir(parents=True, exist_ok=True)
    total_path.write_text("\x00\x00garbage", encoding="utf-8")
    tracker = make_tracker(tmp_path)
    assert tracker.total_state() is not None
    assert tracker.total_state().calls == 0
    events = tracker.recovery_events
    assert any(str(total_path) == event["path"] for event in events)
    assert len(list(tmp_path.glob("budget_total.corrupt-*"))) == 1
    json.loads(total_path.read_text(encoding="utf-8"))


def test_legacy_bare_state_dict_is_tolerated(tmp_path: Path) -> None:
    state_path = tmp_path / "audit" / "budget.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"calls": 7, "cost_usd": 0.5, "started_at": OFF_PEAK_TS}),
        encoding="utf-8",
    )
    tracker = make_tracker(tmp_path)
    assert tracker.state().calls == 7
    assert tracker.state().cost_usd == pytest.approx(0.5)


def test_corrupt_negative_values_are_sanitized(tmp_path: Path) -> None:
    state_path = tmp_path / "audit" / "budget.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "state": {
                    "calls": -3,
                    "prompt_tokens": -5,
                    "cost_usd": -1.0,
                    "started_at": OFF_PEAK_TS,
                    "call_timestamps": [-100, "bad", 3],
                },
            }
        ),
        encoding="utf-8",
    )
    tracker = make_tracker(tmp_path)
    state = tracker.state()
    assert state.calls == 0
    assert state.prompt_tokens == 0
    assert state.cost_usd == 0.0
    assert state.call_timestamps == [3.0]


def test_atomic_save_leaves_no_temp_files(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path)
    tracker.record_success(usage(1_000))
    tracker.save()
    assert list((tmp_path / "audit").glob("*.tmp")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    tracker = BudgetTracker(
        BudgetConfig(min_remaining_usd=0.0),
        ApiConfig(),
        blocker / "audit" / "budget.json",
        total_path=blocker / "total.json",
        clock=FakeClock(OFF_PEAK_TS),
    )
    # load() -> save() must not raise even though the parent cannot be created.
    tracker.load()
    assert tracker.last_save_error
    assert tracker.check_before_call(estimated_cost_usd=0.0)[0] is True
    tracker.record_success(usage(10))  # still no exception
    assert tracker.state().calls == 1


def test_load_is_idempotent_and_keeps_recovery_events(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path)
    tracker.load()
    assert tracker.recovery_events == []
    tracker.record_success(usage(100_000))
    tracker.load()
    assert tracker.recovery_events == []
    assert tracker.state().calls == 1


# --------------------------------------------------------------------------- #
# Failures, copies, snapshots and secrecy
# --------------------------------------------------------------------------- #

def test_record_failure_counters_and_persistence(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path)
    tracker.record_failure(status=429, retryable=True)
    assert tracker.state().consecutive_failures == 1
    tracker.record_failure(status=401, retryable=False)
    assert tracker.state().consecutive_failures == 2
    assert tracker.fatal_failures == 1
    assert tracker.last_failure is not None
    assert tracker.last_failure["status"] == 401
    assert tracker.last_failure["retryable"] is False

    restored = make_tracker(tmp_path)
    assert restored.state().consecutive_failures == 2

    tracker.record_success(usage(10))
    assert tracker.state().consecutive_failures == 0


def test_state_returns_defensive_copy(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path)
    state = tracker.state()
    state.calls = 99
    state.call_timestamps.append(123.0)
    assert tracker.state().calls == 0
    assert tracker.state().call_timestamps == []


def test_snapshot_structure_and_no_secrets(tmp_path: Path, monkeypatch) -> None:
    secret = "sk-snapshot-secret-0123456789abcdef"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    tracker = make_tracker(tmp_path)
    tracker.record_success(usage(1000, 100), model="deepseek-flash")
    snapshot = tracker.snapshot()
    for key in (
        "decided_at",
        "peak",
        "run",
        "total",
        "hourly",
        "limits",
        "remaining",
        "prices",
        "limits_exceeded",
        "near_limit",
        "last_estimate_usd",
        "last_cost_usd",
        "last_usage",
        "last_failure",
        "recovery_events",
        "paths",
    ):
        assert key in snapshot, key
    assert snapshot["run"]["calls"] == 1
    assert snapshot["total"]["calls"] == 1
    assert snapshot["peak"]["is_peak"] is False
    assert snapshot["hourly"]["calls"] == 1
    dumped = json.dumps(snapshot)
    assert secret not in dumped
    assert "Authorization" not in dumped
    assert "DEEPSEEK" not in dumped


def test_snapshot_reports_exhausted_limits(tmp_path: Path) -> None:
    budget = BudgetConfig(max_usd_per_run=0.01, max_usd_total=0.01, min_remaining_usd=0.0)
    tracker = make_tracker(tmp_path, budget_cfg=budget)
    tracker.record_success(usage(100_000, hit=0, miss=100_000))  # $0.015 > limits
    snapshot = tracker.snapshot()
    assert snapshot["limits_exceeded"] is not None
    assert snapshot["remaining"]["usd_run"] == 0.0
    assert snapshot["remaining"]["usd_total"] == 0.0


def test_snapshot_totals_are_per_run_and_global(tmp_path: Path) -> None:
    budget = BudgetConfig(max_usd_per_run=10.0, max_usd_total=10.0, min_remaining_usd=0.0)
    clock = FakeClock(OFF_PEAK_TS)
    first = make_tracker(tmp_path, budget_cfg=budget, clock=clock, state_dir=tmp_path / "run_a")
    first.record_success(usage(1_000_000, hit=0, miss=1_000_000))
    second = make_tracker(tmp_path, budget_cfg=budget, clock=clock, state_dir=tmp_path / "run_b")
    second.record_success(usage(500_000, hit=0, miss=500_000))
    snapshot = second.snapshot()
    assert snapshot["run"]["calls"] == 1
    assert snapshot["total"]["calls"] == 2
    assert snapshot["total"]["cost_usd"] == pytest.approx(0.225)


def test_clock_is_the_only_time_source(tmp_path: Path) -> None:
    clock = FakeClock(OFF_PEAK_TS)
    tracker = make_tracker(tmp_path, clock=clock)
    tracker.record_success(usage(10))
    assert tracker.state().last_call_at == pytest.approx(OFF_PEAK_TS)
    assert tracker.state().started_at == pytest.approx(OFF_PEAK_TS)
    clock.advance(42)
    tracker.record_success(usage(10))
    assert tracker.state().last_call_at == pytest.approx(OFF_PEAK_TS + 42)
    assert tracker.state().calls == 2
