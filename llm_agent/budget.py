"""Cost / usage budget accounting for the DeepSeek game agent.

The budget tracker is the single source of truth for "may we spend more money on
this run?".  It is deliberately independent of the HTTP client and of the game
loop: the runtime estimates a worst-case cost *before* a call, checks it against
hard limits, and records the real usage afterwards.

Design points (see ``llm_agent/ARCHITECTURE.md`` section 7):

* **Pre-call worst-case estimate** — text tokens from a conservative
  character/token ratio, image tokens (capped at the documented 1024 per
  image), ``api.max_tokens`` output tokens and a configurable safety padding.
  Peak-hour pricing is applied when the current UTC time is in a peak window.
* **Lazy limits** — per-run USD/call caps, a global (across runs) USD cap read
  from an optional total file, and calls-per-hour limiting.  A limit configured
  as ``0`` or negative is treated as *disabled* (matching the project
  convention used by ``emulator.max_actions`` / ``runtime.max_turns``).
* **Atomic persistence** — JSON files are written to a temporary file in the
  target directory, fsynced and ``os.replace``d.  Missing files are created;
  corrupt files are backed up to ``<name>.corrupt-<timestamp><suffix>`` and
  reset so a bad budget file can never crash a run.
* **Injectable time** — ``clock`` and ``sleep`` can be replaced by tests (and by
  callers) so budget behaviour is deterministic and no test ever sleeps for
  real.  Single-threaded use only.

Pricing model (USD per 1M tokens, DeepSeek Flash list prices, off-peak)::

    cache hit input   0.003
    cache miss input  0.150
    output            0.600
    peak multiplier   2x when the current UTC time is Mon-Fri and the hour is
                      in [01:00, 04:00) or [06:00, 10:00) (see
                      ``ApiConfig.peak_hours_utc``).
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from llm_agent import contracts
from llm_agent.config import ROOT as PROJECT_ROOT, ApiConfig, BudgetConfig

__all__ = [
    "BudgetTracker",
    "DEFAULT_BUDGET_TOTAL_PATH",
    "IMAGE_TOKEN_CAP",
    "is_peak_hour",
    "peak_multiplier_at",
    "usage_cost_usd",
]

#: Recommended global (across-runs) accounting file; pass it as
#: ``total_path=`` to enable the ``max_usd_total`` cap.  Imported lazily by the
#: runtime rather than created here, to avoid surprising file writes.
DEFAULT_BUDGET_TOTAL_PATH: Path = Path(PROJECT_ROOT) / "artifacts" / "llm_agent" / "budget_total.json"

_PERSIST_VERSION = 1
_HOUR_S = 3600.0
_MAX_RECENT_CALLS = 400
#: Documented maximum image tokens per image for DeepSeek Flash vision.
IMAGE_TOKEN_CAP = 1024
#: Rough patch size used for the dimension-based image-token estimate; the
#: result is clamped to ``IMAGE_TOKEN_CAP`` and the padding covers the residual.
_IMAGE_PATCH_PX = 28
#: Chat-template / framing overhead assumed by the worst-case estimate.
_PROMPT_OVERHEAD_TOKENS = 16


# --------------------------------------------------------------------------- #
# Pricing helpers (pure functions, also used by the probe script)
# --------------------------------------------------------------------------- #

def is_peak_hour(api_cfg: ApiConfig, at: float | None = None) -> bool:
    """Return True when ``at`` (unix time, UTC) falls in a DeepSeek peak window.

    Peak windows are Monday-Friday (UTC) ``[01:00, 04:00)`` and
    ``[06:00, 10:00)``; weekends are always off-peak.  ``at`` defaults to now.
    """
    timestamp = time.time() if at is None else float(at)
    gm = time.gmtime(timestamp)
    if gm.tm_wday >= 5:  # Saturday=5, Sunday=6
        return False
    hour = gm.tm_hour
    for start, end in api_cfg.peak_hours_utc:
        if int(start) <= hour < int(end):
            return True
    return False


def peak_multiplier_at(api_cfg: ApiConfig, at: float | None = None) -> float:
    """Price multiplier at ``at``: ``api_cfg.peak_multiplier`` in peak windows."""
    if is_peak_hour(api_cfg, at):
        return float(api_cfg.peak_multiplier)
    return 1.0


def usage_cost_usd(
    api_cfg: ApiConfig,
    usage: contracts.ApiUsage,
    *,
    peak_multiplier: float = 1.0,
) -> float:
    """Cost of one API call in USD from the reported token usage.

    ``cost = (cache_hit*hit_price + cache_miss*miss_price + completion*out_price)
    / 1e6 * multiplier``.  Unknown cache split is treated as all cache misses
    (the expensive case); if only a cache-hit count is known the remainder of
    ``prompt_tokens`` is counted as misses.  Any prompt tokens beyond the
    reported split are also counted as misses (conservative).
    """
    usage = usage or contracts.ApiUsage()
    prompt = max(0, int(usage.prompt_tokens or 0))
    completion = max(0, int(usage.completion_tokens or 0))
    hit = usage.cache_hit_tokens
    miss = usage.cache_miss_tokens
    if hit is None and miss is None:
        hit_tokens, miss_tokens = 0, prompt
    elif hit is None:
        hit_tokens, miss_tokens = 0, max(0, int(miss or 0))
    elif miss is None:
        hit_tokens = max(0, int(hit))
        miss_tokens = max(0, prompt - hit_tokens)
    else:
        hit_tokens = max(0, int(hit))
        miss_tokens = max(0, int(miss))
        remainder = prompt - hit_tokens - miss_tokens
        if remainder > 0:
            miss_tokens += remainder
    raw_cost = (
        hit_tokens * float(api_cfg.price_input_cache_hit_per_mtok)
        + miss_tokens * float(api_cfg.price_input_cache_miss_per_mtok)
        + completion * float(api_cfg.price_output_per_mtok)
    ) / 1_000_000.0
    multiplier = float(peak_multiplier) if peak_multiplier is not None else 1.0
    return max(0.0, raw_cost * max(0.0, multiplier))


# --------------------------------------------------------------------------- #
# Budget tracker
# --------------------------------------------------------------------------- #

class BudgetTracker:
    """Tracks spend / call counts and enforces the configured budget limits.

    Parameters
    ----------
    budget_cfg:
        Hard limits, estimate padding and ``min_remaining_usd`` reserve.
    api_cfg:
        Prices and peak windows (USD per 1M tokens).
    state_path:
        Per-run accounting file, normally ``<run_dir>/audit/budget.json``.
    total_path:
        Optional global accounting file (e.g.
        ``artifacts/llm_agent/budget_total.json``).  When omitted the global
        ``max_usd_total`` cap is *not* enforced (there is nothing to read the
        cross-run spend from); pass it from the runtime to enable the cap.
    clock:
        Unix-time clock (``time.time`` by default) injected for tests.
    sleep:
        Sleep function (``time.sleep`` by default) used by
        :meth:`wait_until_call_allowed`; injected by tests so they never sleep.

    The file is loaded lazily on first use (and by an explicit :meth:`load`
    call).  Every read/write method is safe to call without calling
    :meth:`load` first.
    """

    def __init__(
        self,
        budget_cfg: BudgetConfig,
        api_cfg: ApiConfig,
        state_path: str | Path,
        *,
        total_path: str | Path | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.budget_cfg = budget_cfg
        self.api_cfg = api_cfg
        self.state_path = Path(state_path)
        self.total_path: Path | None = Path(total_path) if total_path is not None else None
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._sleep: Callable[[float], None] = sleep if sleep is not None else time.sleep

        self._state = contracts.BudgetState()
        self._total: contracts.BudgetState | None = (
            contracts.BudgetState() if self.total_path is not None else None
        )
        self._recent_calls: list[dict[str, float]] = []
        self._loaded = False

        #: last estimate passed to check_before_call (USD)
        self.last_estimate_usd: float = 0.0
        #: cost of the last successful call (USD)
        self.last_cost_usd: float = 0.0
        #: usage of the last successful call
        self.last_usage: contracts.ApiUsage | None = None
        #: model / latency of the last successful call
        self.last_model: str = ""
        self.last_latency_s: float = 0.0
        #: last recorded failure (status / retryable / at)
        self.last_failure: dict[str, Any] | None = None
        #: last persistence error, if any (saving never raises)
        self.last_save_error: str = ""
        #: corrupt/missing-file recovery events
        self.recovery_events: list[dict[str, Any]] = []
        #: number of non-retryable failures recorded this run
        self.fatal_failures: int = 0

    # -- basic helpers ------------------------------------------------------ #

    def _now(self) -> float:
        return float(self._clock())

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _limit_is_active(self, value: float | int | None) -> bool:
        """``0`` / negative limits are disabled (project convention)."""
        return value is not None and float(value) > 0.0

    # -- persistence -------------------------------------------------------- #

    def load(self) -> None:
        """Load run + global state, tolerating missing or corrupt files.

        Missing files start from an empty state and are materialised by an
        atomic :meth:`save`.  Corrupt files are renamed to
        ``<stem>.corrupt-<utc timestamp><suffix>`` (recorded in
        :attr:`recovery_events`) and reset instead of crashing the run.
        """
        self._loaded = True
        now = self._now()
        state, recent, recoveries = self._load_state_file(self.state_path, now)
        if not state.started_at:
            state.started_at = now
        self._state = state
        self._recent_calls = recent
        self.recovery_events.extend(recoveries)
        if self.total_path is not None:
            total, _, total_recoveries = self._load_state_file(self.total_path, now)
            if not total.started_at:
                total.started_at = now
            self._total = total
            self.recovery_events.extend(total_recoveries)
        # Materialise/repair the files immediately (also checks the directory).
        self.save()

    def _load_state_file(
        self, path: Path, now: float
    ) -> tuple[contracts.BudgetState, list[dict[str, float]], list[dict[str, Any]]]:
        """Load one accounting file; return (state, recent_calls, recoveries)."""
        recoveries: list[dict[str, Any]] = []
        if not path.exists():
            state = contracts.BudgetState()
            state.started_at = now
            return state, [], recoveries
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("budget file is not a JSON object")
            raw_state = data.get("state")
            if raw_state is None and "calls" in data:
                raw_state = data  # tolerate a bare BudgetState dict
            if not isinstance(raw_state, dict):
                raise ValueError("budget file has no 'state' object")
            state = contracts.BudgetState.from_dict(raw_state)
            self._sanitize_state(state)
            if not state.started_at:
                state.started_at = now
            recent = self._sanitize_recent(data.get("recent_calls"))
            return state, recent, recoveries
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            backup = self._backup_corrupt_file(path)
            recoveries.append(
                {
                    "path": str(path),
                    "backup": str(backup) if backup is not None else "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": now,
                }
            )
            state = contracts.BudgetState()
            state.started_at = now
            return state, [], recoveries

    def _backup_corrupt_file(self, path: Path) -> Path | None:
        """Rename a corrupt accounting file aside; return the backup path."""
        try:
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(self._now()))
        except (OSError, OverflowError, ValueError):  # pragma: no cover - exotic clocks
            stamp = "unknown"
        candidate = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
        counter = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}.corrupt-{stamp}-{counter}{path.suffix}")
            counter += 1
        try:
            path.replace(candidate)
            return candidate
        except OSError:
            return None

    @staticmethod
    def _sanitize_state(state: contracts.BudgetState) -> None:
        for name in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "consecutive_failures",
        ):
            try:
                setattr(state, name, max(0, int(getattr(state, name) or 0)))
            except (TypeError, ValueError):
                setattr(state, name, 0)
        try:
            state.cost_usd = max(0.0, float(state.cost_usd or 0.0))
        except (TypeError, ValueError):
            state.cost_usd = 0.0
        if not math.isfinite(state.cost_usd):
            state.cost_usd = 0.0
        for name in ("started_at", "last_call_at"):
            try:
                value = float(getattr(state, name) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            setattr(state, name, value if math.isfinite(value) else 0.0)
        stamps: list[float] = []
        for value in state.call_timestamps:
            try:
                stamp = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(stamp) and stamp >= 0:
                stamps.append(stamp)
        state.call_timestamps = stamps[-_MAX_RECENT_CALLS:]

    @staticmethod
    def _sanitize_recent(raw: Any) -> list[dict[str, float]]:
        if not isinstance(raw, list):
            return []
        recent: list[dict[str, float]] = []
        for entry in raw[-_MAX_RECENT_CALLS:]:
            if not isinstance(entry, dict):
                continue
            try:
                at = float(entry.get("at"))
                cost = float(entry.get("cost"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(at) or not math.isfinite(cost) or cost < 0:
                continue
            recent.append({"at": at, "cost": cost})
        return recent

    def save(self) -> None:
        """Atomically persist run (and global) state; never raises.

        A write failure is stored in :attr:`last_save_error` instead of raising
        so a full/read-only disk cannot kill an otherwise healthy run.
        """
        self._ensure_loaded()
        now = self._now()
        errors: list[str] = []
        try:
            self._write_state_file(self.state_path, self._state, kind="run", now=now)
        except OSError as exc:
            errors.append(f"{self.state_path}: {exc}")
        if self.total_path is not None and self._total is not None:
            try:
                self._write_state_file(self.total_path, self._total, kind="total", now=now)
            except OSError as exc:
                errors.append(f"{self.total_path}: {exc}")
        self.last_save_error = "; ".join(errors)

    def _write_state_file(
        self,
        path: Path,
        state: contracts.BudgetState,
        *,
        kind: str,
        now: float,
    ) -> None:
        payload = {
            "version": _PERSIST_VERSION,
            "kind": kind,
            "saved_at": now,
            "state": state.to_dict(),
            "recent_calls": [
                {"at": round(float(item["at"]), 3), "cost": round(float(item["cost"]), 9)}
                for item in self._recent_calls[-_MAX_RECENT_CALLS:]
            ],
        }
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    # -- pricing ------------------------------------------------------------ #

    def is_peak(self, at: float | None = None) -> bool:
        """True when the given unix time (default: now) is a peak-pricing hour."""
        return is_peak_hour(self.api_cfg, self._now() if at is None else at)

    def peak_multiplier(self, at: float | None = None) -> float:
        """Price multiplier at the given unix time (default: now)."""
        return peak_multiplier_at(self.api_cfg, self._now() if at is None else at)

    def cost_for_usage(
        self, usage: contracts.ApiUsage, *, at: float | None = None
    ) -> float:
        """USD cost of one call, applying peak pricing for the given time."""
        multiplier = self.peak_multiplier(at)
        return usage_cost_usd(self.api_cfg, usage, peak_multiplier=multiplier)

    def _estimate_image_tokens(self, width: int, height: int) -> int:
        """Dimension-based image-token estimate clamped to the 1024-token cap."""
        width = max(1, int(width))
        height = max(1, int(height))
        patches = math.ceil(width / _IMAGE_PATCH_PX) * math.ceil(height / _IMAGE_PATCH_PX)
        return int(max(1, min(IMAGE_TOKEN_CAP, patches)))

    def estimate_call_cost(
        self,
        *,
        text_chars: int,
        image_count: int = 1,
        image_width: int = 640,
        image_height: int = 576,
    ) -> float:
        """Worst-case cost (USD) of one upcoming call, including peak pricing.

        ``text_chars`` covers the whole prompt: system prompt, memory context,
        history and user text.  Every input token is priced at the cache-miss
        rate (the expensive case), ``api_cfg.max_tokens`` completion tokens are
        assumed, and the result is multiplied by ``budget_cfg.estimate_padding``.
        """
        if text_chars < 0:
            raise ValueError(f"text_chars must be >= 0, got {text_chars}")
        if image_count < 0:
            raise ValueError(f"image_count must be >= 0, got {image_count}")
        chars_per_token = float(self.budget_cfg.chars_per_token_estimate)
        if not math.isfinite(chars_per_token) or chars_per_token <= 0:
            chars_per_token = 2.0
        text_tokens = int(math.ceil(float(text_chars) / chars_per_token))
        image_tokens = (
            int(image_count) * self._estimate_image_tokens(image_width, image_height)
            if image_count
            else 0
        )
        prompt_tokens = text_tokens + image_tokens + _PROMPT_OVERHEAD_TOKENS
        output_tokens = max(0, int(self.api_cfg.max_tokens))
        raw_cost = (
            prompt_tokens * float(self.api_cfg.price_input_cache_miss_per_mtok)
            + output_tokens * float(self.api_cfg.price_output_per_mtok)
        ) / 1_000_000.0
        padding = float(self.budget_cfg.estimate_padding)
        if not math.isfinite(padding) or padding <= 0:
            padding = 1.0
        return max(0.0, raw_cost * padding * self.peak_multiplier())

    # -- limit checks ------------------------------------------------------- #

    def _calls_in_last_hour(self, now: float) -> list[float]:
        window_start = now - _HOUR_S
        return sorted(
            stamp
            for stamp in self._state.call_timestamps
            if math.isfinite(stamp) and window_start < stamp <= now
        )

    def calls_last_hour(self) -> int:
        """Number of paid calls recorded in the trailing 60-minute window."""
        self._ensure_loaded()
        if not self._limit_is_active(self.budget_cfg.max_calls_per_hour):
            return 0
        return len(self._calls_in_last_hour(self._now()))

    def hourly_cost(self) -> float:
        """Actual (recorded) cost in USD of calls in the trailing 60 minutes."""
        self._ensure_loaded()
        now = self._now()
        window_start = now - _HOUR_S
        if self._recent_calls:
            return sum(
                float(item["cost"])
                for item in self._recent_calls
                if window_start < float(item["at"]) <= now
            )
        # Fallback for files written by an older version (timestamps only):
        # approximate with the average cost of a paid call.
        count = len(self._calls_in_last_hour(now))
        state = self._state
        if state.calls <= 0 or count <= 0:
            return 0.0
        return float(state.cost_usd) / float(state.calls) * count

    def check_before_call(self, *, estimated_cost_usd: float) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for an upcoming call.

        Refuses when the worst-case estimate (plus the configured
        ``min_remaining_usd`` reserve) would exceed the per-run or global USD
        cap, or when a call-count cap is already reached.  ``reason`` is an
        empty string when the call is allowed.
        """
        self._ensure_loaded()
        estimate = max(0.0, float(estimated_cost_usd or 0.0))
        if not math.isfinite(estimate):
            estimate = float("inf")
        self.last_estimate_usd = estimate
        now = self._now()
        cfg = self.budget_cfg

        # Calls per run.
        if self._limit_is_active(cfg.max_calls_per_run):
            limit = int(cfg.max_calls_per_run)
            if self._state.calls >= limit:
                return False, f"calls per run: {self._state.calls} >= limit {limit}"

        # Calls per hour.
        if self._limit_is_active(cfg.max_calls_per_hour):
            limit_hour = int(cfg.max_calls_per_hour)
            hourly = len(self._calls_in_last_hour(now))
            if hourly >= limit_hour:
                return False, (
                    f"calls per hour: {hourly} >= limit {limit_hour} in the last 60 minutes"
                )

        reserve = max(0.0, float(cfg.min_remaining_usd or 0.0))

        # Per-run USD.
        if self._limit_is_active(cfg.max_usd_per_run):
            limit_usd = float(cfg.max_usd_per_run)
            remaining = limit_usd - float(self._state.cost_usd)
            if remaining - estimate < reserve - 1e-12:
                return False, (
                    f"run budget: worst-case ${estimate:.4f} would exceed remaining "
                    f"${remaining:.4f} (limit ${limit_usd:.4f}, reserve ${reserve:.4f})"
                )

        # Global USD.
        if self._total is not None and self._limit_is_active(cfg.max_usd_total):
            limit_total = float(cfg.max_usd_total)
            remaining_total = limit_total - float(self._total.cost_usd)
            if remaining_total - estimate < reserve - 1e-12:
                return False, (
                    f"total budget: worst-case ${estimate:.4f} would exceed remaining "
                    f"${remaining_total:.4f} (limit ${limit_total:.4f}, "
                    f"reserve ${reserve:.4f})"
                )
        return True, ""

    def limits_exceeded(self) -> str | None:
        """Hard-limit reason string if the budget is exhausted, else ``None``.

        Only reports limits that are actually *hit* (spend at/over a USD cap or
        the call-count caps reached); the ``min_remaining_usd`` reserve is
        enforced by :meth:`check_before_call` but is not reported here.
        """
        self._ensure_loaded()
        cfg = self.budget_cfg
        now = self._now()
        if self._limit_is_active(cfg.max_calls_per_run):
            limit = int(cfg.max_calls_per_run)
            if self._state.calls >= limit:
                return f"calls per run limit reached ({self._state.calls}/{limit})"
        if self._limit_is_active(cfg.max_calls_per_hour):
            limit_hour = int(cfg.max_calls_per_hour)
            hourly = len(self._calls_in_last_hour(now))
            if hourly >= limit_hour:
                return f"calls per hour limit reached ({hourly}/{limit_hour} in the last 60 minutes)"
        if self._limit_is_active(cfg.max_usd_per_run):
            limit_usd = float(cfg.max_usd_per_run)
            if float(self._state.cost_usd) >= limit_usd - 1e-12:
                return (
                    f"run budget exhausted (${float(self._state.cost_usd):.4f}/"
                    f"${limit_usd:.4f})"
                )
        if self._total is not None and self._limit_is_active(cfg.max_usd_total):
            limit_total = float(cfg.max_usd_total)
            if float(self._total.cost_usd) >= limit_total - 1e-12:
                return (
                    f"total budget exhausted (${float(self._total.cost_usd):.4f}/"
                    f"${limit_total:.4f})"
                )
        return None

    def near_limit(self) -> str | None:
        """Soft-warning reason when spend/calls crossed ``warn_at_fraction``."""
        self._ensure_loaded()
        cfg = self.budget_cfg
        try:
            fraction = min(1.0, max(0.0, float(cfg.warn_at_fraction)))
        except (TypeError, ValueError):
            fraction = 0.8
        if fraction <= 0:
            return None
        if self._limit_is_active(cfg.max_usd_per_run):
            used = float(self._state.cost_usd) / float(cfg.max_usd_per_run)
            if used >= fraction:
                return (
                    f"run budget {used * 100:.0f}% used "
                    f"(${self._state.cost_usd:.4f}/${float(cfg.max_usd_per_run):.4f})"
                )
        if self._total is not None and self._limit_is_active(cfg.max_usd_total):
            used_total = float(self._total.cost_usd) / float(cfg.max_usd_total)
            if used_total >= fraction:
                return (
                    f"total budget {used_total * 100:.0f}% used "
                    f"(${self._total.cost_usd:.4f}/${float(cfg.max_usd_total):.4f})"
                )
        if self._limit_is_active(cfg.max_calls_per_run):
            used_calls = float(self._state.calls) / float(cfg.max_calls_per_run)
            if used_calls >= fraction:
                return (
                    f"calls per run {used_calls * 100:.0f}% used "
                    f"({self._state.calls}/{int(cfg.max_calls_per_run)})"
                )
        return None

    def seconds_until_call_allowed(self) -> float:
        """Seconds until the calls/hour window frees a slot (0 when allowed)."""
        self._ensure_loaded()
        if not self._limit_is_active(self.budget_cfg.max_calls_per_hour):
            return 0.0
        limit = int(self.budget_cfg.max_calls_per_hour)
        now = self._now()
        stamps = self._calls_in_last_hour(now)
        if len(stamps) < limit:
            return 0.0
        # Enough old calls must age out for a new one to fit in the window.
        excess = len(stamps) - limit + 1
        oldest_relevant = stamps[excess - 1]
        return max(0.0, (oldest_relevant + _HOUR_S) - now)

    def wait_until_call_allowed(self, *, max_wait_s: float = 60.0) -> float:
        """Sleep (injected ``sleep``) up to ``max_wait_s``; return seconds waited.

        Never loops, so a fake clock cannot hang a test.  Returns ``0.0`` when a
        call is already allowed or waiting would not help within the cap.
        """
        self._ensure_loaded()
        wait_s = self.seconds_until_call_allowed()
        if wait_s <= 0:
            return 0.0
        duration = min(wait_s, max(0.0, float(max_wait_s)))
        if duration <= 0:
            return 0.0
        self._sleep(duration)
        return duration

    # -- recording ---------------------------------------------------------- #

    def _apply_success(
        self,
        state: contracts.BudgetState,
        usage: contracts.ApiUsage,
        cost: float,
        now: float,
    ) -> None:
        prompt = max(0, int(usage.prompt_tokens or 0))
        completion = max(0, int(usage.completion_tokens or 0))
        total = int(usage.total_tokens or 0) or (prompt + completion)
        state.calls += 1
        state.prompt_tokens += prompt
        state.completion_tokens += completion
        state.total_tokens += max(0, total)
        state.cache_hit_tokens += max(0, int(usage.cache_hit_tokens or 0))
        state.cost_usd = float(state.cost_usd) + float(cost)
        state.consecutive_failures = 0
        state.last_call_at = now
        state.call_timestamps.append(now)
        if len(state.call_timestamps) > _MAX_RECENT_CALLS:
            state.call_timestamps = state.call_timestamps[-_MAX_RECENT_CALLS:]

    def record_success(
        self,
        usage: contracts.ApiUsage,
        *,
        model: str = "",
        latency_s: float = 0.0,
    ) -> float:
        """Account one successful paid call; return its actual cost in USD."""
        self._ensure_loaded()
        usage = usage or contracts.ApiUsage()
        now = self._now()
        cost = self.cost_for_usage(usage, at=now)
        if not math.isfinite(cost) or cost < 0:
            cost = 0.0
        self._apply_success(self._state, usage, cost, now)
        if self._total is not None:
            self._apply_success(self._total, usage, cost, now)
        self._recent_calls.append({"at": now, "cost": cost})
        self._recent_calls = self._recent_calls[-_MAX_RECENT_CALLS:]
        self.last_cost_usd = cost
        self.last_usage = copy.deepcopy(usage)
        self.last_model = str(model or "")
        self.last_latency_s = max(0.0, float(latency_s or 0.0))
        self.save()
        return cost

    def record_failure(self, *, status: int | None = None, retryable: bool = True) -> None:
        """Account one failed request (no cost is recorded for failures)."""
        self._ensure_loaded()
        now = self._now()
        self._state.consecutive_failures += 1
        if not retryable:
            self.fatal_failures += 1
        self.last_failure = {
            "status": None if status is None else int(status),
            "retryable": bool(retryable),
            "at": now,
        }
        self.save()

    # -- reporting ---------------------------------------------------------- #

    def state(self) -> contracts.BudgetState:
        """Copy of the current per-run budget state."""
        self._ensure_loaded()
        return copy.deepcopy(self._state)

    def total_state(self) -> contracts.BudgetState | None:
        """Copy of the global budget state, or ``None`` without a total file."""
        self._ensure_loaded()
        return None if self._total is None else copy.deepcopy(self._total)

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe, non-secret view of the budget (for logs/viewer/reports)."""
        self._ensure_loaded()
        now = self._now()
        cfg = self.budget_cfg
        run = self._state
        total = self._total
        hourly = len(self._calls_in_last_hour(now))
        remaining_run = (
            max(0.0, float(cfg.max_usd_per_run) - float(run.cost_usd))
            if self._limit_is_active(cfg.max_usd_per_run)
            else None
        )
        remaining_total = (
            max(0.0, float(cfg.max_usd_total) - float(total.cost_usd))
            if (total is not None and self._limit_is_active(cfg.max_usd_total))
            else None
        )
        return {
            "decided_at": now,
            "peak": {
                "is_peak": self.is_peak(now),
                "multiplier": self.peak_multiplier(now),
                "utc_weekday": time.gmtime(now).tm_wday,
                "utc_hour": time.gmtime(now).tm_hour,
            },
            "run": run.to_dict(),
            "total": total.to_dict() if total is not None else None,
            "hourly": {
                "calls": hourly,
                "cost_usd": round(self.hourly_cost(), 6),
                "max_calls_per_hour": int(cfg.max_calls_per_hour),
                "seconds_until_call_allowed": round(self.seconds_until_call_allowed(), 3),
            },
            "limits": {
                "max_usd_per_run": float(cfg.max_usd_per_run),
                "max_usd_total": float(cfg.max_usd_total),
                "max_calls_per_run": int(cfg.max_calls_per_run),
                "max_calls_per_hour": int(cfg.max_calls_per_hour),
                "min_remaining_usd": float(cfg.min_remaining_usd),
                "warn_at_fraction": float(cfg.warn_at_fraction),
            },
            "remaining": {
                "usd_run": None if remaining_run is None else round(remaining_run, 6),
                "usd_total": None if remaining_total is None else round(remaining_total, 6),
                "calls_run": max(0, int(cfg.max_calls_per_run) - int(run.calls))
                if self._limit_is_active(cfg.max_calls_per_run)
                else None,
            },
            "prices": self.api_cfg.price_dict(),
            "limits_exceeded": self.limits_exceeded(),
            "near_limit": self.near_limit(),
            "last_estimate_usd": round(float(self.last_estimate_usd), 6),
            "last_cost_usd": round(float(self.last_cost_usd), 6),
            "last_usage": self.last_usage.to_dict() if self.last_usage is not None else None,
            "last_model": self.last_model,
            "last_latency_s": round(float(self.last_latency_s), 4),
            "last_failure": dict(self.last_failure) if self.last_failure else None,
            "fatal_failures": int(self.fatal_failures),
            "recovery_events": list(self.recovery_events),
            "paths": {
                "state": str(self.state_path),
                "total": None if self.total_path is None else str(self.total_path),
            },
            "last_save_error": self.last_save_error,
        }
