#!/usr/bin/env python
"""Plot real-environment-step training curves for the Pokemon Red PPO run (seed_0).

Read-only analysis of a training experiment directory:

    artifacts/training/seed_0/stage_*/metrics.jsonl   (one row per PPO update)
    artifacts/training/seed_0/stage_*/episodes.jsonl  (one row per truncated episode)

Outputs (defaults, all relative to the repository root):

    docs/reports/figures/fig_reward.png
    docs/reports/figures/fig_loss.png
    docs/reports/figures/fig_entropy_kl.png
    docs/reports/figures/fig_exploration.png
    docs/reports/figures/fig_gameplay.png
    docs/reports/figures/fig_actions.png
    docs/reports/figures/fig_throughput.png
    artifacts/training/seed_0/analysis/training_summary.json
    artifacts/training/seed_0/analysis/training_summary.md
    artifacts/training/seed_0/analysis/metrics_10k.csv
    artifacts/training/seed_0/analysis/run_log.json     (one entry appended per run)
    docs/reports/training_curves.md                     (Chinese report, regenerated)

Design notes
------------
* Only numpy + matplotlib are required (the project venv does not ship pandas);
  CSV output uses the standard library ``csv`` module.
* Stages are discovered automatically from ``stage_*`` directories, sorted by the
  cumulative environment-step range they cover, and merged on
  ``progress/environment_steps`` (duplicates at stage boundaries are dropped) so a
  stage that resumes from the previous stage's checkpoint is never double-counted.
* A stage counts as *complete* when it has a ``result.json`` (written by
  ``scripts/train.py`` when the stage exits). Incomplete/running stages are excluded
  by default; pass ``--include-running`` for a temporary preview that includes
  whatever rows a still-running stage has already written.
* Episodes carry no step field. ``episode/completed_this_update`` gives the number of
  episodes logged by each update, and its sum equals the number of episode rows, so
  each episode is located inside the update window in which it was logged (uniformly
  spread over that 2,000-step window, i.e. error <= 1 update). If the counters ever
  disagree, episodes fall back to episode-index positioning and this is reported.
* Strictly read-only for training data, no training / evaluation / GPU usage.
  matplotlib runs on the Agg backend and the process is single-threaded.
* Every figure is 1440x900 px (12.0 x 7.5 in @ 120 dpi), English labels, x axis in
  real environment steps, and carries a provenance note saying the data comes from
  the actual local training run and is not a paper result.

Usage
-----
    .venv\\Scripts\\python.exe scripts\\plot_training_curves.py
    .venv\\Scripts\\python.exe scripts\\plot_training_curves.py --root artifacts/training/seed_0
    .venv\\Scripts\\python.exe scripts\\plot_training_curves.py --include-running   # preview only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths / matplotlib environment (must happen before importing matplotlib)
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def _preparse_paths() -> tuple[Path, Path]:
    """Resolve --root/--analysis-dir early so the matplotlib cache stays in-scope."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", default="artifacts/training/seed_0")
    p.add_argument("--analysis-dir", default=None)
    a, _ = p.parse_known_args()
    root = Path(a.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    analysis = Path(a.analysis_dir) if a.analysis_dir else root / "analysis"
    if not analysis.is_absolute():
        analysis = REPO_ROOT / analysis
    return root.resolve(), analysis.resolve()


_ROOT_PRE, _ANALYSIS_PRE = _preparse_paths()
os.environ.setdefault("MPLBACKEND", "Agg")
# Keep matplotlib's font cache inside our own writable analysis directory.
os.environ.setdefault("MPLCONFIGDIR", str(_ANALYSIS_PRE / ".mplconfig"))
# Single-threaded on purpose: training is still using the CPU while this runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter  # noqa: E402

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
SCHEMA = "poke-red-ppo-training-summary/1"
FIGURE_DPI = 120
FIGURE_SIZE = (12.0, 7.5)  # 1440 x 900 @ 120 dpi
ROLLING_WINDOW = 20  # PPO updates (20 x 2,000 = 40,000 env steps)
BIN_STEPS = 10_000

STAGE_RE = re.compile(r"^stage_([a-zA-Z])_(\d+)([kmKM]?)$")
ACTION_NAMES = {
    0: "DOWN",
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "A",
    5: "B",
    6: "START",
}
SHARE_COLS = [f"actions/share_{i}" for i in range(7)]

STAGE_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#937860"]
RAW_COLOR = "#8a8f98"
ROLL_COLOR = "#1f4e9c"

# name -> raw metrics.jsonl key
METRIC_SPECS = [
    ("reward_mean", "rollout/reward_mean"),
    ("reward_sum", "rollout/reward_sum"),
    ("loss", "loss/loss"),
    ("policy_loss", "loss/policy_loss"),
    ("value_loss", "loss/value_loss"),
    ("entropy", "other/entropy"),
    ("approx_kl", "other/kl_divergence"),
    ("clip_fraction", "other/clip_fraction"),
    ("learning_rate", "decay/learning_rate"),
    ("coord_count", "game/coord_count"),
    ("max_map_progress", "game/max_map_progress"),
    ("num_badges", "game/num_badges"),
    ("event_reward", "game/event_reward"),
    ("levels_sum", "game/levels_sum"),
    ("party_size", "game/party_size"),
    ("deaths", "game/deaths"),
    ("healr", "game/healr"),
    ("mt_moon", "game/mt_moon"),
    ("cerulean", "game/cerulean"),
    ("update_seconds", "time/update_seconds"),
    ("steps_per_second", "time/steps_per_second"),
]
METRIC_KEY = dict(METRIC_SPECS)

# Keys reported as "values at the end of each stage" in the summaries.
END_VALUE_KEYS = [
    "reward_mean",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "coord_count",
    "max_map_progress",
    "num_badges",
    "event_reward",
    "levels_sum",
    "party_size",
    "deaths",
    "healr",
    "steps_per_second",
    "action_entropy",
]

LIMITATIONS = [
    "Training curves are behavioural telemetry of one single-seed training run, "
    "not evidence that the agent has learned to play Pokemon Red.",
    "rollout/reward_mean is the mean per-environment-step reward inside one 2,000-step "
    "PPO update (8 workers averaged), not an episode return and not an evaluation score. "
    "The reward is a shaped combination (reward_scale 0.5, event_weight 4.0, "
    "level_weight 1.0, heal_weight 5.0, explore_weight 0.1), so a rising reward may come "
    "from reward-shaping saturation rather than from better gameplay.",
    "Update-level game/* metrics (coord_count, max_map_progress, levels_sum, party_size, "
    "deaths) average the 8 workers' latest-step info dictionaries inside one update. Each "
    "value is the worker's running progress in its current episode, so it drops when an "
    "episode is truncated/reset (sawtooth curves) and is not the maximum of finished "
    "episodes; episode-level extremes from episodes.jsonl are reported separately.",
    "Episode-level data is very sparse (max_steps is 10,240 and episodes extend up to "
    "16,384 steps), so episode statistics have large variance. Episodes are located "
    "inside the update window in which they were logged (error <= 1 update = 2,000 steps).",
    "All numbers come from the actual local training run (PyBoy + CUDA), not from the "
    "paper. The parallel scale here is 8 x 250 = 2,000 env steps/update because of local "
    "memory limits, so curves are not directly comparable with paper results.",
    "Single seed, single configuration, no confidence intervals; later curve segments "
    "have not been repeated independently.",
    "Training metrics and independent evaluation use different protocols and throughput "
    "definitions; they must not be compared directly. Evaluation is done separately by "
    "the eval engineer with scripts/eval_checkpoint.py (docs/reports/eval_protocol.md).",
    "Compute time is the sum of time/update_seconds (sampling + PPO update only); it "
    "excludes stage startup, checkpointing, evaluation and analysis. Wall-clock span "
    "reported separately includes gaps between stage processes.",
    "The 20-update rolling mean spans 40,000 environment steps; edges of the curve use "
    "partial windows (min_periods=1). Stage A covers only 5 updates, so its statistics "
    "are weak.",
    "result.json for every stage sets performance_claim='none'; nothing in this report "
    "is a performance claim.",
]

# Provenance sentence shown in every figure footer.
SOURCE_SENTENCE = (
    "Source: actual Pokemon Red PPO training run seed_0 on the local emulator "
    "(PyBoy + CUDA). Every point = one PPO update (8 workers x 250 = 2,000 real "
    "environment steps). NOT paper results."
)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def now_local() -> datetime:
    return datetime.now().astimezone()


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping blank/chunked lines (training may still append)."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a partially flushed trailing line
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_git_commit(repo: Path) -> str | None:
    git = repo / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref: "):
        return head or None
    ref = head[len("ref: "):].strip()
    ref_path = git / ref
    if ref_path.exists():
        try:
            return ref_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    try:  # packed refs fallback
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")) and line.endswith(ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def as_float(value) -> float:
    """Numeric JSON value -> finite float, anything else -> NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    v = float(value)
    return v if math.isfinite(v) else float("nan")


def jsonable(obj):
    """Recursively convert numpy types and non-finite floats for strict JSON."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(jsonable(payload), fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def fmt_num(x, nd: int = 4) -> str:
    """Readable number formatting for Markdown tables."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    ax = abs(v)
    if ax >= 1000:
        return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}"
    if ax >= 1:
        return f"{v:.3f}"
    if ax >= 0.01:
        return f"{v:.4f}"
    return f"{v:.5f}"


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_steps(x, _pos=None) -> str:
    """0 -> '0', 10000 -> '10k', 1000000 -> '1M'."""
    v = float(x)
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def fmt_steps_short(x: float) -> str:
    v = float(x)
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def safe_mean(values: np.ndarray) -> float | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else None


def safe_std(values: np.ndarray) -> float | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if v.size > 1 else (0.0 if v.size == 1 else None)


def max_or_none(values: np.ndarray) -> float | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else None


def any_positive(values: np.ndarray) -> bool:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return bool(v.size and np.any(v > 0.0))


def episode_badges(rec: dict) -> float:
    """badge_1 + badge_2 from an episode record (NaN only if both are missing)."""
    b1 = as_float(rec.get("badge_1"))
    b2 = as_float(rec.get("badge_2"))
    if not (math.isfinite(b1) or math.isfinite(b2)):
        return float("nan")
    return (b1 if math.isfinite(b1) else 0.0) + (b2 if math.isfinite(b2) else 0.0)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with min_periods=1, NaN-aware (numpy only)."""
    y = np.asarray(values, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    filled = np.where(np.isfinite(y), y, 0.0)
    counts = np.isfinite(y).astype(np.int64)
    csum = np.cumsum(filled)
    ccnt = np.cumsum(counts)
    window = max(1, int(window))
    for i in range(n):
        lo = max(0, i - window + 1)
        total = csum[i] - (csum[lo - 1] if lo > 0 else 0.0)
        count = ccnt[i] - (ccnt[lo - 1] if lo > 0 else 0)
        if count > 0:
            out[i] = total / count
    return out


def fmt_csv(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{v:.6f}" if math.isfinite(v) else ""


# --------------------------------------------------------------------------------------
# Stage discovery and merging
# --------------------------------------------------------------------------------------
def parse_target_steps(token: str, unit: str) -> int:
    value = int(token)
    unit = unit.lower()
    if unit == "k":
        return value * 1_000
    if unit == "m":
        return value * 1_000_000
    return value


def infer_batch(stage_dir: Path, steps: list[int]) -> int:
    """Env steps consumed per PPO update (n_workers * worker_steps)."""
    diffs = sorted({b - a for a, b in zip(steps, steps[1:]) if b > a})
    if diffs:
        return int(round(float(np.median(diffs))))
    # Fallback for a stage with a single logged update: parse sampler config.
    try:
        text = (stage_dir / "config.yaml").read_text(encoding="utf-8")
        n_workers = re.search(r"^\s*n_workers:\s*(\d+)", text, re.MULTILINE)
        worker_steps = re.search(r"^\s*worker_steps:\s*(\d+)", text, re.MULTILINE)
        if n_workers and worker_steps:
            return int(n_workers.group(1)) * int(worker_steps.group(1))
    except OSError:
        pass
    return 2000  # configs/train_seed0.yaml default


def discover_stages(root: Path) -> tuple[list[dict], list[str]]:
    """Return stages sorted by their cumulative step range, plus discovery warnings."""
    stages: list[dict] = []
    warnings: list[str] = []
    if not root.is_dir():
        raise SystemExit(f"[plot] ERROR: experiment root not found: {root}")

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = STAGE_RE.match(d.name)
        if not m:
            continue
        metrics_path = d / "metrics.jsonl"
        if not metrics_path.exists():
            warnings.append(f"{d.name}: no metrics.jsonl, stage skipped")
            continue
        rows = read_jsonl(metrics_path)
        if not rows:
            warnings.append(f"{d.name}: metrics.jsonl empty, stage skipped")
            continue
        episodes = read_jsonl(d / "episodes.jsonl")
        result = read_json(d / "result.json")
        try:
            steps = [int(r["progress/environment_steps"]) for r in rows]
        except (KeyError, TypeError, ValueError):
            warnings.append(f"{d.name}: metrics rows missing progress/environment_steps, skipped")
            continue
        batch = infer_batch(d, steps)
        target = parse_target_steps(m.group(2), m.group(3))
        last_steps = max(steps)
        complete = result is not None or last_steps >= target
        stage = {
            "name": d.name,
            "dir": d,
            "letter": m.group(1).upper(),
            "target_steps": target,
            "start_steps": min(steps) - batch,
            "end_steps": last_steps,
            "batch": batch,
            "complete": complete,
            "result": result,
            "metrics_rows": rows,
            "episodes": episodes,
            "metrics_path": metrics_path,
            "episodes_path": d / "episodes.jsonl",
        }
        if stage["start_steps"] < 0:
            warnings.append(f"{d.name}: derived start_steps < 0 ({stage['start_steps']}), clamped to 0")
            stage["start_steps"] = 0
        stages.append(stage)

    stages.sort(key=lambda s: (s["start_steps"], s["target_steps"], s["name"]))
    # Basic overlap sanity check (stage ranges must be contiguous and non-overlapping).
    for prev, cur in zip(stages, stages[1:]):
        if cur["start_steps"] < prev["end_steps"]:
            warnings.append(
                f"stages {prev['name']} and {cur['name']} overlap "
                f"({prev['end_steps']} > {cur['start_steps']}); merged rows are de-duplicated"
            )
    return stages, warnings


class RunData:
    """Merged per-update metrics with numpy array access (no pandas dependency)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        n = len(rows)
        self.n = n
        self.steps = np.array([int(r["progress/environment_steps"]) for r in rows], dtype=np.int64)
        self.updates = np.array([int(r["progress/update"]) for r in rows], dtype=np.int64)
        self.stage = np.array([str(r["_stage"]) for r in rows], dtype=object)
        self.stage_letter = np.array([str(r["_letter"]) for r in rows], dtype=object)
        self.completed = np.array(
            [as_float(r.get("episode/completed_this_update")) for r in rows], dtype=float
        )
        self.completed = np.nan_to_num(self.completed, nan=0.0)
        self.time_unix = np.array([as_float(r.get("time/unix")) for r in rows], dtype=float)
        self.raw: dict[str, np.ndarray] = {}
        for _name, key in METRIC_SPECS:
            self.raw[key] = np.array([as_float(r.get(key)) for r in rows], dtype=float)

        self.shares: np.ndarray | None = None
        self.shares_norm: np.ndarray | None = None
        self.action_entropy = np.full(n, np.nan)
        if n and all(c in rows[0] for c in SHARE_COLS):
            shares = np.array(
                [[as_float(r.get(c)) for c in SHARE_COLS] for r in rows], dtype=float
            )
            shares = np.where(np.isfinite(shares) & (shares > 0.0), shares, 0.0)
            totals = shares.sum(axis=1)
            norm = np.zeros_like(shares)
            good = totals > 0.0
            norm[good] = shares[good] / totals[good, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                logp = np.log(np.where(norm > 0.0, norm, 1.0))
            self.shares = shares
            self.shares_norm = norm
            self.action_entropy = -np.sum(norm * logp, axis=1)

    def get(self, key: str) -> np.ndarray:
        if key == "action_entropy":
            return self.action_entropy
        if key in self.raw:
            return self.raw[key]
        return np.full(self.n, np.nan)

    def stage_mask(self, stage_name: str) -> np.ndarray:
        return self.stage == stage_name


def build_metrics(stages: list[dict]) -> tuple[RunData, int]:
    """Merge included stages, keep one row per update, no boundary double counting."""
    rows: list[dict] = []
    for s in stages:
        for r in s["metrics_rows"]:
            row = dict(r)
            row["_stage"] = s["name"]
            row["_letter"] = s["letter"]
            rows.append(row)
    if not rows:
        raise SystemExit("[plot] ERROR: no complete stage with metrics to analyse")

    valid: list[tuple[int, int, dict]] = []
    for r in rows:
        try:
            steps = int(r["progress/environment_steps"])
            update = int(r["progress/update"])
        except (KeyError, TypeError, ValueError):
            continue
        valid.append((steps, update, r))
    valid.sort(key=lambda item: (item[0], item[1]))

    merged: list[dict] = []
    seen_steps: set[int] = set()
    seen_updates: set[int] = set()
    dropped = 0
    for steps, update, row in valid:
        if steps in seen_steps or update in seen_updates:
            dropped += 1
            continue
        seen_steps.add(steps)
        seen_updates.add(update)
        merged.append(row)
    return RunData(merged), dropped


class EpisodeData:
    """Episode records with numpy arrays; steps may be NaN when locating failed."""

    def __init__(self, recs: list[dict]):
        self.recs = recs
        n = len(recs)
        self.n = n
        self.steps = np.array([as_float(r.get("steps")) for r in recs], dtype=float)
        self.episode_index = np.arange(1, n + 1, dtype=np.int64)
        self.stage = np.array([str(r.get("stage")) for r in recs], dtype=object)
        self.stage_letter = np.array([str(r.get("stage_letter")) for r in recs], dtype=object)
        self.reward = np.array([as_float(r.get("reward")) for r in recs], dtype=float)
        self.length = np.array([as_float(r.get("length")) for r in recs], dtype=float)
        # Episode-level game progress (full-episode extremes; the update-level game/*
        # metrics in metrics.jsonl are only 8-worker latest-info averages).
        self.coord_count = np.array([as_float(r.get("coord_count")) for r in recs], dtype=float)
        self.max_map_progress = np.array(
            [as_float(r.get("max_map_progress")) for r in recs], dtype=float
        )
        self.levels_sum = np.array([as_float(r.get("levels_sum")) for r in recs], dtype=float)
        self.party_size = np.array([as_float(r.get("party_size")) for r in recs], dtype=float)
        self.deaths = np.array([as_float(r.get("deaths")) for r in recs], dtype=float)
        self.event_reward = np.array(
            [as_float(r.get("event_reward")) for r in recs], dtype=float
        )
        self.healr = np.array([as_float(r.get("healr")) for r in recs], dtype=float)
        self.mt_moon = np.array([as_float(r.get("mt_moon")) for r in recs], dtype=float)
        self.cerulean = np.array([as_float(r.get("cerulean")) for r in recs], dtype=float)
        self.badges = np.array([episode_badges(r) for r in recs], dtype=float)
        self.located = bool(n == 0 or all(r.get("located") for r in recs))


def build_episodes(stages: list[dict], run: RunData) -> tuple[EpisodeData, bool]:
    """Locate each episode inside the update window in which it was logged.

    ``episode/completed_this_update`` is the number of episodes that finished in that
    update; its per-stage sum equals the number of episodes.jsonl rows (verified on the
    seed_0 data). Episodes are spread uniformly over that update's 2,000-step window so
    points do not overlap; positional error stays <= 1 update.
    """
    recs: list[dict] = []
    mapping_ok = True
    for s in stages:
        mask = run.stage_mask(s["name"])
        idxs = np.nonzero(mask)[0]
        episodes = list(s["episodes"])
        idx = 0
        batch = max(int(s["batch"]), 1)
        for row_idx in idxs:
            k = int(round(float(run.completed[row_idx])))
            k = max(k, 0)
            for j in range(k):
                if idx >= len(episodes):
                    break
                rec = dict(episodes[idx])
                rec.update(
                    {
                        "steps": float(
                            run.steps[row_idx] - batch + (j + 0.5) * batch / max(k, 1)
                        ),
                        "logged_update_steps": int(run.steps[row_idx]),
                        "update": int(run.updates[row_idx]),
                        "stage": s["name"],
                        "stage_letter": s["letter"],
                        "episode_index": len(recs) + 1,
                        "located": True,
                    }
                )
                recs.append(rec)
                idx += 1
        for j in range(idx, len(episodes)):
            rec = dict(episodes[j])
            rec.update(
                {
                    "steps": None,
                    "logged_update_steps": None,
                    "update": None,
                    "stage": s["name"],
                    "stage_letter": s["letter"],
                    "episode_index": len(recs) + 1,
                    "located": False,
                }
            )
            recs.append(rec)
            mapping_ok = False
    return EpisodeData(recs), mapping_ok


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def stat_dict(values: np.ndarray, steps: np.ndarray) -> dict | None:
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    if not mask.any():
        return None
    valid = v[mask]
    valid_steps = steps[mask]
    i_min = int(np.argmin(valid))
    i_max = int(np.argmax(valid))
    return {
        "first": float(valid[0]),
        "last": float(valid[-1]),
        "mean": float(valid.mean()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "min_at_steps": int(valid_steps[i_min]),
        "max_at_steps": int(valid_steps[i_max]),
        "n": int(mask.sum()),
    }


def stage_summary(stage: dict, run: RunData, eps: EpisodeData) -> dict:
    mask = run.stage_mask(stage["name"])
    n = int(mask.sum())
    update_seconds = run.get("time/update_seconds")[mask]
    compute_seconds = float(np.nansum(update_seconds)) if n else 0.0
    sps = run.get("time/steps_per_second")[mask]
    env_steps = int(stage["end_steps"] - stage["start_steps"])
    unix = run.time_unix[mask]
    unix = unix[np.isfinite(unix)]
    wall = None
    if n and len(unix):
        last_update_seconds = float(update_seconds[-1]) if np.isfinite(update_seconds[-1]) else 0.0
        wall = float(unix[-1] + last_update_seconds - unix[0])

    ep_mask = eps.stage == stage["name"] if eps.n else np.zeros(0, dtype=bool)
    ep_rewards = eps.reward[ep_mask] if eps.n else np.array([], dtype=float)
    ep_rewards = ep_rewards[np.isfinite(ep_rewards)]

    end_values = {}
    if n:
        last_idx = int(np.nonzero(mask)[0][-1])
        for key in END_VALUE_KEYS:
            col = "action_entropy" if key == "action_entropy" else METRIC_KEY.get(key, key)
            val = float(run.get(col)[last_idx])
            end_values[key] = val if math.isfinite(val) else None

    return {
        "name": stage["name"],
        "letter": stage["letter"],
        "status": "complete" if stage["complete"] else "running",
        "start_steps": int(stage["start_steps"]),
        "end_steps": int(stage["end_steps"]),
        "target_steps": int(stage["target_steps"]),
        "environment_steps": env_steps,
        "updates": n,
        "batch_steps_per_update": int(stage["batch"]),
        "compute_seconds": compute_seconds,
        "mean_steps_per_second": safe_mean(sps),
        "aggregate_steps_per_second": (env_steps / compute_seconds) if compute_seconds > 0 else None,
        "wall_span_seconds": wall,
        "episodes_completed": int(len(ep_rewards)),
        "episode_reward_mean": safe_mean(ep_rewards),
        "episode_reward_min": float(ep_rewards.min()) if len(ep_rewards) else None,
        "episode_reward_max": float(ep_rewards.max()) if len(ep_rewards) else None,
        "result_stop_reason": (stage["result"] or {}).get("stop_reason"),
        "result_environment_steps": (stage["result"] or {}).get("environment_steps"),
        "end_values": end_values,
        "metrics_sha256": sha256_file(stage["metrics_path"]),
        "episodes_sha256": sha256_file(stage["episodes_path"]),
    }


def build_summary(
    root: Path,
    stages_all: list[dict],
    stages_included: list[dict],
    stages_pending: list[dict],
    run: RunData,
    eps: EpisodeData,
    ep_mapping_ok: bool,
    duplicate_rows_dropped: int,
    discovery_warnings: list[str],
) -> dict:
    generated = now_local()
    stage_rows = [stage_summary(s, run, eps) for s in stages_included]
    total_steps = int(sum(s["environment_steps"] for s in stage_rows))
    total_updates = int(run.n)
    total_compute = float(sum(s["compute_seconds"] for s in stage_rows))

    unix = run.time_unix[np.isfinite(run.time_unix)]
    wall_span = None
    update_seconds_all = run.get("time/update_seconds")
    if run.n and len(unix) and np.isfinite(update_seconds_all[-1]):
        wall_span = float(unix[-1] + update_seconds_all[-1] - unix[0])

    sps = run.get("time/steps_per_second")
    metrics = {}
    for name, key in METRIC_SPECS:
        st = stat_dict(run.get(key), run.steps)
        if st:
            metrics[name] = st
    st_entropy = stat_dict(run.action_entropy, run.steps)
    if st_entropy:
        metrics["action_entropy"] = st_entropy

    episodes = {"count": int(eps.n)}
    if eps.n and np.isfinite(eps.reward).any():
        ep_reward = eps.reward[np.isfinite(eps.reward)]
        ep_len = eps.length[np.isfinite(eps.length)]
        episodes.update(
            {
                "reward_mean": safe_mean(ep_reward),
                "reward_std": safe_std(ep_reward),
                "reward_min": float(ep_reward.min()),
                "reward_max": float(ep_reward.max()),
                "reward_median": float(np.median(ep_reward)),
                "reward_last": float(ep_reward[-1]),
                "length_mean": safe_mean(ep_len),
                "length_min": float(ep_len.min()) if len(ep_len) else None,
                "length_max": float(ep_len.max()) if len(ep_len) else None,
                "step_locating_method": (
                    "episodes placed inside the update window in which they were logged "
                    "(uniformly spread over 2,000 steps; error <= 1 update)"
                    if ep_mapping_ok
                    else "episode/completed_this_update counters did not match episodes.jsonl; "
                         "episode plots fall back to episode index"
                ),
                "step_mapping_ok": bool(ep_mapping_ok),
                "progress_extremes": {
                    "coord_count_max": max_or_none(eps.coord_count),
                    "max_map_progress_max": max_or_none(eps.max_map_progress),
                    "levels_sum_max": max_or_none(eps.levels_sum),
                    "party_size_max": max_or_none(eps.party_size),
                    "deaths_max": max_or_none(eps.deaths),
                    "num_badges_max": max_or_none(eps.badges),
                    "event_reward_max": max_or_none(eps.event_reward),
                    "healr_max": max_or_none(eps.healr),
                    "mt_moon_any": any_positive(eps.mt_moon),
                    "cerulean_any": any_positive(eps.cerulean),
                    "length_max": max_or_none(eps.length),
                    "note": "extremes over complete episodes from episodes.jsonl; the "
                            "update-level game/* metrics are 8-worker latest-info averages",
                },
                "by_stage": {
                    s["name"]: {
                        "count": int((eps.stage == s["name"]).sum()),
                        "reward_mean": safe_mean(eps.reward[eps.stage == s["name"]]),
                        "reward_max": max_or_none(eps.reward[eps.stage == s["name"]]),
                        "coord_count_max": max_or_none(eps.coord_count[eps.stage == s["name"]]),
                        "max_map_progress_max": max_or_none(
                            eps.max_map_progress[eps.stage == s["name"]]
                        ),
                        "levels_sum_max": max_or_none(eps.levels_sum[eps.stage == s["name"]]),
                        "party_size_max": max_or_none(eps.party_size[eps.stage == s["name"]]),
                        "deaths_max": max_or_none(eps.deaths[eps.stage == s["name"]]),
                        "num_badges_max": max_or_none(eps.badges[eps.stage == s["name"]]),
                        "event_reward_max": max_or_none(
                            eps.event_reward[eps.stage == s["name"]]
                        ),
                        "mt_moon_any": any_positive(eps.mt_moon[eps.stage == s["name"]]),
                        "cerulean_any": any_positive(eps.cerulean[eps.stage == s["name"]]),
                    }
                    for s in stages_included
                },
            }
        )

    # Action distribution (shares, entropy, dominant action at the end of the run).
    actions = {}
    if run.shares_norm is not None and run.n:
        norm = run.shares_norm
        last_shares = norm[-1]
        mean_shares = np.nanmean(norm, axis=0)
        max_shares = np.nanmax(norm, axis=0)
        max_share_at = {}
        for i in range(7):
            col = norm[:, i]
            idx = int(np.argmax(col)) if np.isfinite(col).any() else None
            max_share_at[i] = int(run.steps[idx]) if idx is not None else None
        ent = st_entropy
        actions = {
            "action_order": {str(k): v for k, v in ACTION_NAMES.items()},
            "last_shares": {ACTION_NAMES[i]: float(last_shares[i]) for i in range(7)},
            "mean_shares": {ACTION_NAMES[i]: float(mean_shares[i]) for i in range(7)},
            "max_shares": {ACTION_NAMES[i]: float(max_shares[i]) for i in range(7)},
            "max_share_at_steps": {ACTION_NAMES[i]: max_share_at[i] for i in range(7)},
            "dominant_action_last": ACTION_NAMES[int(np.argmax(last_shares))],
            "dominant_share_last": float(np.max(last_shares)),
            "entropy": ent,
            "max_possible_entropy_ln7": float(math.log(7.0)),
            "updates_with_single_action_share_gt_60pct": int(
                np.sum(np.nanmax(norm, axis=1) > 0.60)
            ),
        }

    # Game-progress end state / extremes.
    def last_metric(key: str) -> float | None:
        vals = run.get(METRIC_KEY.get(key, key))
        return float(vals[-1]) if len(vals) and math.isfinite(vals[-1]) else None

    def max_metric(key: str) -> tuple[float | None, int | None]:
        vals = run.get(METRIC_KEY.get(key, key))
        mask = np.isfinite(vals)
        if not mask.any():
            return None, None
        idx = int(np.argmax(vals))
        return float(vals[idx]), int(run.steps[idx])

    progress = {
        "last": {
            "coord_count": last_metric("coord_count"),
            "max_map_progress": last_metric("max_map_progress"),
            "num_badges": last_metric("num_badges"),
            "levels_sum": last_metric("levels_sum"),
            "party_size": last_metric("party_size"),
            "deaths": last_metric("deaths"),
            "event_reward": last_metric("event_reward"),
            "healr": last_metric("healr"),
            "mt_moon": last_metric("mt_moon"),
            "cerulean": last_metric("cerulean"),
        },
        "max": {},
        "milestones_reached": {},
    }
    for key in ("coord_count", "max_map_progress", "num_badges", "levels_sum",
                "party_size", "deaths", "event_reward", "healr"):
        value, at_steps = max_metric(key)
        progress["max"][key] = {"value": value, "at_steps": at_steps}
    for key, label in (("num_badges", "any_badge"), ("mt_moon", "reached_mt_moon"),
                       ("cerulean", "reached_cerulean")):
        vals = run.get(METRIC_KEY[key])
        valid = vals[np.isfinite(vals)]
        progress["milestones_reached"][label] = bool(valid.size and np.any(valid > 0))

    rewards = run.get("rollout/reward_mean")
    finite_rewards = rewards[np.isfinite(rewards)]
    trend_n = max(1, len(finite_rewards) // 10)
    reward_trend = {
        "first_10pct_mean": float(finite_rewards[:trend_n].mean()),
        "last_10pct_mean": float(finite_rewards[-trend_n:].mean()),
        "first": float(finite_rewards[0]),
        "last": float(finite_rewards[-1]),
    }

    non_finite = 0
    for _name, key in METRIC_SPECS:
        arr = run.get(key)
        non_finite += int(np.count_nonzero(~np.isfinite(arr)))

    state = read_json(root / "training_state.json") or {}
    supervisor_snapshot = None
    if state:
        supervisor_snapshot = {
            "status": state.get("status"),
            "current_stage": (state.get("stage") or {}).get("name"),
            "current_stage_status": (state.get("stage") or {}).get("status"),
            "total_training_env_steps": state.get("total_training_env_steps"),
            "cumulative_training_seconds": state.get("cumulative_training_seconds"),
            "stages_completed": [s.get("name") for s in state.get("stages_completed", [])],
        }

    sources = {
        "experiment_root": str(root),
        "config": str(REPO_ROOT / "configs" / "train_seed0.yaml"),
        "training_state": str(root / "training_state.json"),
        "stages": [
            {
                "name": s["name"],
                "metrics_jsonl": str(s["metrics_path"]),
                "metrics_rows_read": len(s["metrics_rows"]),
                "metrics_bytes": s["metrics_path"].stat().st_size if s["metrics_path"].exists() else None,
                "episodes_jsonl": str(s["episodes_path"]),
                "episodes_rows_read": len(s["episodes"]),
                "episodes_bytes": s["episodes_path"].stat().st_size if s["episodes_path"].exists() else None,
                "start_steps": int(s["start_steps"]),
                "last_steps": int(s["end_steps"]),
                "target_steps": int(s["target_steps"]),
                "complete": bool(s["complete"]),
            }
            for s in stages_all
        ],
    }

    return {
        "schema": SCHEMA,
        "generated_at": iso(generated),
        "generated_at_utc": generated.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/plot_training_curves.py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": read_git_commit(REPO_ROOT),
        "experiment_root": str(root),
        "data_status": {
            "stages_included": [s["name"] for s in stages_included],
            "stages_pending_or_running": [s["name"] for s in stages_pending],
            "all_stages_complete": len(stages_pending) == 0,
            "environment_steps_total": total_steps,
            "stages_covered": [
                {"name": s["name"], "start_steps": int(s["start_steps"]),
                 "end_steps": int(s["end_steps"])}
                for s in stages_included
            ],
            "note": (
                "Stage D not included yet (still running / no result.json). Rerun the script "
                "after Stage D completes to get the full 1,000,000-step version."
                if stages_pending
                else "All discovered stages are complete; this is the full run."
            ),
        },
        "totals": {
            "environment_steps": total_steps,
            "ppo_updates": total_updates,
            "stages_included": len(stages_included),
            "completed_episodes": int(eps.n),
            "compute_seconds": total_compute,
            "wall_span_seconds": wall_span,
            "mean_steps_per_second": safe_mean(sps),
            "aggregate_steps_per_second": (total_steps / total_compute) if total_compute > 0 else None,
            "mean_update_seconds": safe_mean(update_seconds_all),
            "median_update_seconds": (
                float(np.nanmedian(update_seconds_all))
                if np.isfinite(update_seconds_all).any() else None
            ),
            "duplicate_rows_dropped": int(duplicate_rows_dropped),
            "non_finite_metric_values": int(non_finite),
        },
        "reward_trend": reward_trend,
        "stages": stage_rows,
        "metrics": metrics,
        "episodes": episodes,
        "actions": actions,
        "game_progress": progress,
        "supervisor_snapshot": supervisor_snapshot,
        "data_sources": sources,
        "discovery_warnings": discovery_warnings,
        "limitations": LIMITATIONS,
        "figure_files": [],  # filled in by main()
    }


# --------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------
def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.6,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def stage_bounds(stages: list[dict]) -> tuple[float, float]:
    return float(stages[0]["start_steps"]), float(max(s["end_steps"] for s in stages))


def add_stage_context(ax, stages: list[dict], letters: bool = True) -> None:
    """Shade alternating stages, draw boundaries and (optionally) stage letters."""
    colors = [STAGE_COLORS[i % len(STAGE_COLORS)] for i in range(len(stages))]
    x_max = max(float(s["end_steps"]) for s in stages) if stages else 1.0
    for i, s in enumerate(stages):
        ax.axvspan(s["start_steps"], s["end_steps"], color=colors[i], alpha=0.045, lw=0, zorder=0)
        if i > 0:
            ax.axvline(s["start_steps"], color="#999999", linestyle=":", linewidth=0.8, zorder=1)
        if letters:
            mid = (s["start_steps"] + s["end_steps"]) / 2.0
            # Narrow stages get a single letter so the label fits inside the axes and
            # does not collide with the y tick labels or the next stage label.
            width = float(s["end_steps"] - s["start_steps"])
            label = f"Stage {s['letter']}" if width >= 0.08 * x_max else s["letter"]
            mid = max(mid, x_max * 0.017)
            ax.annotate(
                label,
                xy=(mid, 0.975),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=7.5 if len(label) > 1 else 8.5,
                fontweight="bold",
                color="#444444",
                alpha=0.95,
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7),
            )


def style_axis(ax, x_max: float, letters: bool = False, stages: list[dict] | None = None,
               xlabel: bool = False) -> None:
    ax.set_xlim(0, x_max * 1.015)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
    if stages:
        add_stage_context(ax, stages, letters=letters)
    if xlabel:
        ax.set_xlabel("Environment steps (real)")


def panel_series(ax, run: RunData, key: str, color: str, rolling_window: int,
                 label_raw: str = "per update", raw_marker_size: float = 2.0) -> None:
    steps = run.steps
    y = run.get(key)
    if len(y):
        ax.plot(
            steps, y, color=RAW_COLOR, linewidth=0.7, marker=".",
            markersize=raw_marker_size, alpha=0.65, label=label_raw, zorder=2,
        )
        ax.plot(
            steps, rolling_mean(y, rolling_window), color=color, linewidth=1.9,
            label=f"rolling mean (window {rolling_window} updates = "
                  f"{rolling_window * 2000:,} steps)",
            zorder=3,
        )


def finish_figure(fig, path: Path, title: str, footer_extra: str = "") -> dict:
    footer = SOURCE_SENTENCE
    if footer_extra:
        footer = footer + "\n" + footer_extra
    fig.text(0.006, 0.008, footer, fontsize=6.6, color="#666666", ha="left", va="bottom")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    width_px = int(round(fig.get_size_inches()[0] * FIGURE_DPI))
    height_px = int(round(fig.get_size_inches()[1] * FIGURE_DPI))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return {
        "file": path.name,
        "path": str(path),
        "title": title,
        "bytes": path.stat().st_size,
        "width_px": width_px,
        "height_px": height_px,
    }


def _footer_stage_note(stages: list[dict], pending: list[dict]) -> str:
    included = "; ".join(
        f"Stage {s['letter']} {fmt_steps_short(s['start_steps'])}-{fmt_steps_short(s['end_steps'])} steps"
        for s in stages
    )
    note = f"Stages included: {included}."
    if pending:
        note += " Pending: " + ", ".join(p["name"] for p in pending) + " (not included yet)."
    return note


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_reward(run, eps, ep_ok, stages, pending, figures_dir, rolling_window) -> dict:
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.14, hspace=0.30)
    x_max = stage_bounds(stages)[1]

    ax = axes[0]
    panel_series(ax, run, "rollout/reward_mean", ROLL_COLOR, rolling_window,
                 label_raw="per-update rollout mean (2,000-step update)")
    ax.set_ylabel("Rollout reward per env step")
    ax.set_title("Rollout reward mean, per PPO update (raw + 20-update rolling mean)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), framealpha=0.9)
    style_axis(ax, x_max, letters=True, stages=stages)

    ax = axes[1]
    if eps is not None and eps.n:
        x = eps.steps if ep_ok else eps.episode_index.astype(float)
        for i, s in enumerate(stages):
            sub = eps.stage == s["name"]
            if not sub.any():
                continue
            ax.scatter(
                x[sub], eps.reward[sub],
                s=30, color=STAGE_COLORS[i % len(STAGE_COLORS)], edgecolor="white",
                linewidth=0.6, alpha=0.9, zorder=3,
                label=f"Stage {s['letter']} ({int(sub.sum())} ep)",
            )
        rewards = eps.reward[np.isfinite(eps.reward)]
        if len(rewards):
            ax.axhline(float(rewards.mean()), color="#444444", linestyle="--", linewidth=0.9,
                       label=f"episode mean = {fmt_num(float(rewards.mean()))}")
        if eps.n > 1 and ep_ok:
            order = np.argsort(eps.steps)
            ax.plot(
                eps.steps[order],
                rolling_mean(eps.reward[order], 5),
                color="#c44e52", linewidth=1.3, label="5-episode rolling mean",
            )
        ax.set_ylabel("Episode reward (sum)")
        ax.set_title(f"Episode returns: {eps.n} completed episodes in the included stages")
        ax.legend(loc="upper left", framealpha=0.9, ncol=2)
        if ep_ok:
            style_axis(ax, x_max, xlabel=True)
            ax.set_xlabel(
                "Environment steps (real) - episode located within its logging "
                "update window (+/-1 update)"
            )
        else:
            ax.set_xlim(0, max(2, eps.n) + 1)
            ax.set_xlabel("Episode index (sequential) - step mapping unavailable")
    else:
        ax.text(0.5, 0.5, "No completed episodes recorded in the included stages yet",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#555555")
        style_axis(ax, x_max, xlabel=True)

    return finish_figure(
        fig, figures_dir / "fig_reward.png",
        "Reward: rollout per-step mean and episode returns (real environment steps)",
        _footer_stage_note(stages, pending),
    )


def fig_loss(run, stages, pending, figures_dir, rolling_window) -> dict:
    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.125, hspace=0.42)
    x_max = stage_bounds(stages)[1]
    specs = [
        ("loss/loss", "Total loss (loss/loss)", "#1f4e9c"),
        ("loss/policy_loss", "Policy loss (loss/policy_loss)", "#c44e52"),
        ("loss/value_loss", "Value loss (loss/value_loss)", "#55a868"),
    ]
    for i, (key, title, color) in enumerate(specs):
        ax = axes[i]
        panel_series(ax, run, key, color, rolling_window)
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), framealpha=0.9)
        style_axis(ax, x_max, letters=(i == 0), stages=stages, xlabel=(i == len(specs) - 1))
    return finish_figure(
        fig, figures_dir / "fig_loss.png",
        "PPO losses (real environment steps) - total, policy and value loss",
        _footer_stage_note(stages, pending)
        + "\nNote: the total loss can legitimately be negative; it is not a divergence indicator.",
    )


def fig_entropy_kl(run, stages, pending, figures_dir, rolling_window) -> dict:
    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.125, hspace=0.42)
    x_max = stage_bounds(stages)[1]

    ax = axes[0]
    panel_series(ax, run, "other/entropy", "#4c72b0", rolling_window)
    ax.axhline(math.log(7.0), color="#c44e52", linestyle="--", linewidth=0.9,
               label=f"uniform policy: ln(7) = {math.log(7.0):.3f}")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Policy entropy (higher = more random policy)")
    ent_vals = run.get("other/entropy")
    ent_vals = ent_vals[np.isfinite(ent_vals)]
    if len(ent_vals):
        ax.set_ylim(max(0.0, float(ent_vals.min()) - 0.15),
                    max(float(ent_vals.max()), math.log(7.0)) + 0.12)
    ax.legend(loc="lower left", framealpha=0.9)
    style_axis(ax, x_max, letters=True, stages=stages)

    ax = axes[1]
    panel_series(ax, run, "other/kl_divergence", "#dd8452", rolling_window)
    ax.set_ylabel("Approx. KL")
    ax.set_title("Approximate KL divergence (policy change per update)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), framealpha=0.9)
    style_axis(ax, x_max, stages=stages)

    ax = axes[2]
    panel_series(ax, run, "other/clip_fraction", "#55a868", rolling_window)
    ax.set_ylabel("Clip fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("PPO clip fraction (fraction of ratios outside the clip range)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), framealpha=0.9)
    style_axis(ax, x_max, stages=stages, xlabel=True)
    return finish_figure(
        fig, figures_dir / "fig_entropy_kl.png",
        "Exploration / optimisation diagnostics: entropy, approx. KL, clip fraction "
        "(real environment steps)",
        _footer_stage_note(stages, pending),
    )


def _quad_figure(run, stages, pending, figures_dir, rolling_window, fname, title,
                 specs, letters_top=True) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.13,
                        hspace=0.40, wspace=0.16)
    x_max = stage_bounds(stages)[1]
    flat = axes.ravel()
    for i, (key, ylabel, panel_title, color) in enumerate(specs):
        ax = flat[i]
        panel_series(ax, run, key, color, rolling_window)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), framealpha=0.9,
                  fontsize=7.0)
        style_axis(ax, x_max, letters=(letters_top and i < 2), stages=stages,
                   xlabel=(i >= 2))
    return finish_figure(
        fig, figures_dir / fname, title, _footer_stage_note(stages, pending)
    )


def fig_exploration(run, stages, pending, figures_dir, rolling_window) -> dict:
    specs = [
        ("game/coord_count", "Unique coordinates visited",
         "Map coverage: coord_count (unique (x, y, map) positions)", "#4c72b0"),
        ("game/max_map_progress", "Essential-map progress (0..14)",
         "Story progress: max_map_progress (official essential maps)", "#dd8452"),
        ("game/num_badges", "Badges (0..2)",
         "Badges obtained: num_badges", "#55a868"),
        ("game/event_reward", "Event reward component",
         "Event reward component (game/event_reward)", "#8172b3"),
    ]
    return _quad_figure(
        run, stages, pending, figures_dir, rolling_window,
        "fig_exploration.png",
        "Exploration and story progress (real environment steps)",
        specs,
    )


def fig_gameplay(run, stages, pending, figures_dir, rolling_window) -> dict:
    specs = [
        ("game/levels_sum", "Party level sum", "Party level sum: levels_sum", "#4c72b0"),
        ("game/party_size", "Party size", "Party size: party_size (0..6)", "#dd8452"),
        ("game/deaths", "Deaths per update", "Deaths: game/deaths (mean per update)", "#c44e52"),
        ("game/healr", "Heal reward component", "Healing reward component: game/healr", "#55a868"),
    ]
    return _quad_figure(
        run, stages, pending, figures_dir, rolling_window,
        "fig_gameplay.png",
        "Gameplay statistics: levels, party, deaths, healing (real environment steps)",
        specs,
    )


def fig_actions(run, stages, pending, figures_dir, rolling_window) -> dict:
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.13, hspace=0.32)
    x_max = stage_bounds(stages)[1]
    steps = run.steps
    norm = run.shares_norm

    ax = axes[0]
    colors = plt.get_cmap("tab10").colors
    if norm is not None:
        ax.stackplot(
            steps, *[norm[:, i] for i in range(7)],
            labels=[f"{i} {ACTION_NAMES[i]}" for i in range(7)],
            colors=[colors[i] for i in range(7)], alpha=0.95, linewidth=0,
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Action share")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Action distribution per update (7-button Game Boy action space)")
    ax.legend(loc="upper right", ncol=2, fontsize=7.5, framealpha=0.92)
    style_axis(ax, x_max, letters=True, stages=stages)

    ax = axes[1]
    panel_series(ax, run, "action_entropy", "#8172b3", rolling_window)
    ax.axhline(math.log(7.0), color="#c44e52", linestyle="--", linewidth=0.9,
               label=f"uniform policy: ln(7) = {math.log(7.0):.3f}")
    ax.set_ylabel("Action entropy (nats)")
    ax.set_title("Action-distribution entropy (higher = closer to uniform; max = ln 7)")
    ax.legend(loc="lower left", framealpha=0.9)
    style_axis(ax, x_max, stages=stages, xlabel=True)
    return finish_figure(
        fig, figures_dir / "fig_actions.png",
        "Policy action distribution and action entropy (real environment steps)",
        _footer_stage_note(stages, pending)
        + "\nAction order: 0 DOWN, 1 LEFT, 2 RIGHT, 3 UP, 4 A, 5 B, 6 START.",
    )


def fig_throughput(run, stages, pending, figures_dir, rolling_window) -> dict:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.15)
    x_max = stage_bounds(stages)[1]
    sps = run.get("time/steps_per_second")
    mean_sps = safe_mean(sps)
    ax.plot(run.steps, sps, color=RAW_COLOR, linewidth=0.8, marker=".",
            markersize=2.2, alpha=0.7, label="per-update throughput")
    ax.plot(run.steps, rolling_mean(sps, rolling_window), color=ROLL_COLOR,
            linewidth=2.0, label=f"rolling mean (window {rolling_window} updates)")
    if mean_sps is not None:
        ax.axhline(mean_sps, color="#c44e52", linestyle="--", linewidth=1.0,
                   label=f"mean = {mean_sps:,.1f} env steps/s")
    ax.set_ylabel("Environment steps per second")
    ax.set_title("Training throughput (sampling + PPO update, end-to-end per update)")
    ax.legend(loc="lower left", framealpha=0.9)
    style_axis(ax, x_max, letters=True, stages=stages, xlabel=True)
    return finish_figure(
        fig, figures_dir / "fig_throughput.png",
        "Training throughput in real environment steps per second",
        _footer_stage_note(stages, pending)
        + "\nThroughput = 2,000 / time/update_seconds; unlike evaluation throughput it "
          "includes 8 parallel emulators and the PPO update.",
    )


# --------------------------------------------------------------------------------------
# Markdown outputs
# --------------------------------------------------------------------------------------
def write_summary_md(path: Path, summary: dict) -> None:
    totals = summary["totals"]
    stages = summary["stages"]
    status = summary["data_status"]
    lines: list[str] = []
    lines.append("# Training summary - Pokemon Red PPO, seed_0 (real environment steps)")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append(f"- Script: `{summary['script']}` (read-only analysis, Agg backend, single-threaded)")
    lines.append(f"- Experiment root: `{summary['experiment_root']}`")
    lines.append(f"- Git commit: `{summary['git_commit']}`")
    lines.append(f"- Stages included: {', '.join(s['name'] for s in stages)}")
    if status["stages_pending_or_running"]:
        lines.append(
            "- **Stage D not included yet** (still running / no result.json). "
            "Rerun this script after Stage D completes for the full 1,000,000-step version."
        )
    lines.append("")
    lines.append("This file is machine-generated data telemetry from the actual local training "
                 "run; it is not a paper result and not a performance claim. "
                 "Narrative report: `docs/reports/training_curves.md`.")
    lines.append("")

    lines.append("## Totals")
    lines.append("")
    lines.append(md_table(
        ["Metric", "Value"],
        [
            ["Environment steps covered", f"{totals['environment_steps']:,}"],
            ["PPO updates", f"{totals['ppo_updates']:,}"],
            ["Completed episodes", f"{totals['completed_episodes']:,}"],
            ["Compute time (sum of time/update_seconds)", f"{totals['compute_seconds']:,.1f} s"],
            ["Wall-clock span (time/unix, includes stage gaps)",
             f"{totals['wall_span_seconds']:,.1f} s" if totals["wall_span_seconds"] else "n/a"],
            ["Mean throughput (mean of per-update steps/s)",
             f"{totals['mean_steps_per_second']:,.1f}" if totals["mean_steps_per_second"] else "n/a"],
            ["Aggregate throughput (steps / compute seconds)",
             f"{totals['aggregate_steps_per_second']:,.1f}" if totals["aggregate_steps_per_second"] else "n/a"],
            ["Mean / median update time",
             f"{fmt_num(totals['mean_update_seconds'], 2)} / "
             f"{fmt_num(totals['median_update_seconds'], 2)} s"],
            ["Duplicate rows dropped at stage boundaries", f"{totals['duplicate_rows_dropped']:,}"],
            ["Non-finite metric values", f"{totals['non_finite_metric_values']:,}"],
        ],
    ))
    lines.append("")

    lines.append("## Stages")
    lines.append("")
    rows = []
    for s in stages:
        rows.append([
            s["name"],
            s["status"],
            f"{s['start_steps']:,} - {s['end_steps']:,}",
            f"{s['updates']:,}",
            f"{s['compute_seconds']:,.1f}",
            f"{s['mean_steps_per_second']:,.1f}" if s["mean_steps_per_second"] else "n/a",
            f"{s['episodes_completed']:,}",
        ])
    lines.append(md_table(
        ["Stage", "Status", "Environment steps", "Updates", "Compute s",
         "Mean steps/s", "Episodes"], rows))
    lines.append("")
    for s in stages:
        if s["end_values"]:
            vals = ", ".join(f"{k}={fmt_num(v)}" for k, v in s["end_values"].items()
                             if v is not None)
            lines.append(f"- End of {s['name']}: {vals}")
    lines.append("")

    lines.append("## Metric statistics (per PPO update)")
    lines.append("")
    lines.append(md_table(
        ["Metric", "First", "Last", "Mean", "Min", "Max", "Max at steps", "N"],
        [
            [name, fmt_num(st["first"]), fmt_num(st["last"]), fmt_num(st["mean"]),
             fmt_num(st["min"]), fmt_num(st["max"]), f"{st['max_at_steps']:,}", st["n"]]
            for name, st in summary["metrics"].items()
        ],
    ))
    lines.append("")

    ep = summary.get("episodes", {})
    if ep.get("count"):
        lines.append("## Episodes")
        lines.append("")
        lines.append(md_table(
            ["Metric", "Value"],
            [
                ["Completed episodes", f"{ep['count']:,}"],
                ["Episode reward mean / std",
                 f"{fmt_num(ep['reward_mean'])} / {fmt_num(ep['reward_std'])}"],
                ["Episode reward min / median / max",
                 f"{fmt_num(ep['reward_min'])} / {fmt_num(ep['reward_median'])} / {fmt_num(ep['reward_max'])}"],
                ["Episode length mean / min / max (steps)",
                 f"{fmt_num(ep['length_mean'], 1)} / {fmt_int(ep['length_min'])} / {fmt_int(ep['length_max'])}"],
                ["Episode step locating", ep["step_locating_method"]],
            ],
        ))
        lines.append("")
        pe = ep.get("progress_extremes") or {}
        if pe:
            lines.append(md_table(
                ["Episode game-progress extreme", "Value"],
                [
                    ["coord_count max", fmt_num(pe.get("coord_count_max"))],
                    ["max_map_progress max (0..14)", fmt_num(pe.get("max_map_progress_max"))],
                    ["levels_sum max", fmt_num(pe.get("levels_sum_max"))],
                    ["party_size max (0..6)", fmt_num(pe.get("party_size_max"))],
                    ["deaths max", fmt_num(pe.get("deaths_max"))],
                    ["num_badges max (0..2)", fmt_num(pe.get("num_badges_max"))],
                    ["event_reward max", fmt_num(pe.get("event_reward_max"))],
                    ["healr max", fmt_num(pe.get("healr_max"))],
                    ["mt_moon reached by any episode", pe.get("mt_moon_any")],
                    ["cerulean reached by any episode", pe.get("cerulean_any")],
                ],
            ))
            lines.append("")
            lines.append("Episode extremes are full-episode maxima from `episodes.jsonl`; the "
                         "update-level `game/*` metrics in the tables above are 8-worker "
                         "latest-info averages and are not episode maxima.")
            lines.append("")
        by_stage_rows = [
            [
                name,
                v["count"],
                fmt_num(v["reward_mean"]),
                fmt_num(v.get("reward_max")),
                fmt_num(v.get("coord_count_max")),
                fmt_num(v.get("max_map_progress_max")),
                fmt_num(v.get("levels_sum_max")),
                fmt_num(v.get("party_size_max")),
            ]
            for name, v in ep.get("by_stage", {}).items()
        ]
        if by_stage_rows:
            lines.append(md_table(
                ["Stage", "Episodes", "Reward mean", "Reward max", "coord max",
                 "map_progress max", "levels max", "party max"],
                by_stage_rows,
            ))
            lines.append("")

    actions = summary.get("actions", {})
    if actions:
        lines.append("## Action shares (last update of included stages)")
        lines.append("")
        lines.append(md_table(
            ["Action", "Last share", "Mean share", "Max share", "Max share at steps"],
            [
                [f"{i} {ACTION_NAMES[i]}",
                 fmt_num(actions["last_shares"][ACTION_NAMES[i]]),
                 fmt_num(actions["mean_shares"][ACTION_NAMES[i]]),
                 fmt_num(actions["max_shares"][ACTION_NAMES[i]]),
                 f"{actions['max_share_at_steps'][ACTION_NAMES[i]]:,}"]
                for i in range(7)
            ],
        ))
        lines.append("")
        ent = actions.get("entropy") or {}
        lines.append(
            f"- Action entropy: first={fmt_num(ent.get('first'))}, last={fmt_num(ent.get('last'))}, "
            f"mean={fmt_num(ent.get('mean'))}, max possible ln(7)={math.log(7.0):.4f} nats"
        )
        lines.append(
            f"- Dominant action in the last update: {actions['dominant_action_last']} "
            f"({fmt_num(actions['dominant_share_last'])}); updates with a single action > 60%: "
            f"{actions['updates_with_single_action_share_gt_60pct']}"
        )
        lines.append("")

    lines.append("## Data sources")
    lines.append("")
    src_rows = []
    for s in summary["data_sources"]["stages"]:
        src_rows.append([
            s["name"],
            f"{s['metrics_rows_read']:,}",
            f"{s['episodes_rows_read']:,}",
            "complete" if s["complete"] else "running",
            (s["metrics_jsonl"].split("seed_0")[-1] if "seed_0" in s["metrics_jsonl"] else s["metrics_jsonl"]),
        ])
    lines.append(md_table(["Stage", "Metrics rows", "Episode rows", "Status", "Path suffix"],
                          src_rows))
    lines.append("")
    lines.append("All SHA-256 hashes and supervisor state snapshots are recorded in "
                 "`training_summary.json`.")
    lines.append("")
    lines.extend(["## Limitations", ""])
    for i, item in enumerate(summary["limitations"], 1):
        lines.append(f"{i}. {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_10k_csv(path: Path, run: RunData, bin_steps: int) -> int:
    """Aggregate per-update metrics into fixed 10k-step environment-step bins."""
    bins = (np.ceil(run.steps / float(bin_steps)) * bin_steps).astype(np.int64)
    headers = ["steps", "reward_mean", "coord_count", "max_map_progress", "levels_sum",
               "party_size", "badges", "entropy", "kl", "clip", "loss", "steps_per_second"]
    columns = {
        "reward_mean": run.get("rollout/reward_mean"),
        "coord_count": run.get("game/coord_count"),
        "max_map_progress": run.get("game/max_map_progress"),
        "levels_sum": run.get("game/levels_sum"),
        "party_size": run.get("game/party_size"),
        "badges": run.get("game/num_badges"),
        "entropy": run.get("other/entropy"),
        "kl": run.get("other/kl_divergence"),
        "clip": run.get("other/clip_fraction"),
        "loss": run.get("loss/loss"),
        "steps_per_second": run.get("time/steps_per_second"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_bins = np.unique(bins)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for b in unique_bins:
            mask = bins == b
            row = [int(b)]
            for name in headers[1:]:
                values = columns[name][mask]
                mean = safe_mean(values)
                row.append("" if mean is None else f"{mean:.6f}")
            writer.writerow(row)
    return int(len(unique_bins))


# --------------------------------------------------------------------------------------
# Chinese report (docs/reports/training_curves.md)
# --------------------------------------------------------------------------------------
def write_report_md(path: Path, summary: dict, figures: list[dict],
                    figure_errors: list[str], rolling_window: int) -> None:
    totals = summary["totals"]
    stages = summary["stages"]
    status = summary["data_status"]
    metrics = summary["metrics"]
    actions = summary.get("actions", {})
    eps = summary.get("episodes", {})
    progress = summary.get("game_progress", {})
    trend = summary["reward_trend"]
    figure_names = {f["file"] for f in figures}

    if status["stages_pending_or_running"]:
        pending_names = ", ".join(status["stages_pending_or_running"])
        steps_so_far = totals["environment_steps"]
        status_line = (
            f"已纳入 {', '.join(s['name'] for s in stages)}（累计 {steps_so_far:,} 环境步）；"
            f"**{pending_names} 尚未完成（无 result.json），未纳入本版图表。"
            f"阶段 D 完成后重跑本脚本即可得到完整 1,000,000 步版本。**"
        )
    else:
        status_line = (
            f"已纳入全部阶段（累计 {totals['environment_steps']:,} 环境步），本版为完整版。"
        )

    L: list[str] = []
    L.append("# 训练曲线分析：Pokémon Red PPO（seed_0，真实环境步）")
    L.append("")
    L.append(f"> **数据状态**：{status_line}")
    L.append(
        f"> 生成时间：{summary['generated_at']}；脚本：`{summary['script']}`；"
        f"数据：`{summary['experiment_root']}`；Git：`{summary['git_commit']}`。"
    )
    L.append("")
    L.append(
        "本报告由 `scripts/plot_training_curves.py` 从真实训练日志自动生成，"
        "只做只读分析（不训练、不评估、不用 GPU）。所有曲线都是**本地真实训练**的"
        "遥测数据，**不是论文结果**，也不构成性能声明（各阶段 result.json 的 "
        "`performance_claim` 均为 `none`）。"
    )
    L.append("")

    # ---- 1. Data and definitions
    L.append("## 1. 数据来源与口径")
    L.append("")
    L.append(
        "- 实验：`artifacts/training/seed_0/`，配置 `configs/train_seed0.yaml`（论文 PPO "
        "超参不变；本地并行规模 8 workers × 250 steps = **2,000 环境步/update**）。"
    )
    L.append(
        "- 输入文件（只读）：每个阶段的 `metrics.jsonl`（每个 PPO update 一行）与 "
        "`episodes.jsonl`（每个被截断的 episode 一行）。阶段按 `progress/environment_steps` "
        "合并并去重，阶段边界不重复计数。"
    )
    L.append(
        f"- x 轴统一为 **real environment steps（真实环境步）**；每个点 = 一个 PPO update（2,000 步）；"
        f"滑动平均窗口 {rolling_window} updates = {rolling_window * 2000:,} 环境步"
        f"（窗口首尾用 `min_periods=1`，边缘为不完整窗口）。"
    )
    L.append(
        "- 奖励口径：配置中奖励为塑形组合（`reward_scale=0.5`、`event_weight=4.0`、"
        "`level_weight=1.0`、`heal_weight=5.0`、`explore_weight=0.1`）。`rollout/reward_mean` "
        "是每个 update 的 2,000 步内、8 个并行环境平均后的**每环境步**奖励，"
        "**不是 episode return，也不是评估分数**；`episodes.jsonl` 的 `reward` 字段是 "
        "episode 累计奖励。"
    )
    L.append(
        "- 计算时间口径：`time/update_seconds` 是每个 update 的采样 + PPO 更新时间；"
        "“阶段计算时间”为这些值求和，**不含**阶段启动、checkpoint、评估与分析开销；"
        "另给出基于 `time/unix` 的墙钟跨度（含阶段间间隙），二者不同。"
    )
    L.append(
        "- `game/*` update 指标口径：训练脚本对 8 个 worker 在 update 内最新一步返回的 info "
        "取算术平均，每个数值是**该 worker 当前 episode 的累计进度**；episode 截断/重置后会回落，"
        "因此曲线呈锯齿。它与 `episodes.jsonl` 中**完整 episode 的最大值**不是同一口径；"
        "本报告在 §3.2 给 update 级统计、在 §3.3 单独给 episode 级极值，二者不可混读。"
    )
    L.append("")

    # ---- 2. Figures
    L.append("## 2. 图表说明")
    L.append("")
    L.append("图表目录：`docs/reports/figures/`（1440×900 PNG，dpi 120，英文标签）。"
             "每张图的 x 轴均为真实环境步，底部注明数据来自真实训练、非论文结果。")
    L.append("")

    def fig_block(name: str, title: str, meaning: str, reading: str) -> None:
        L.append(f"### {title}（`figures/{name}`）")
        L.append("")
        if name in figure_names:
            L.append(f"![{name}](figures/{name})")
            L.append("")
        L.append(meaning)
        L.append("")
        L.append(reading)
        L.append("")

    reward = metrics.get("reward_mean", {})
    fig_block(
        "fig_reward.png",
        "图 1：奖励曲线",
        "上半部分：`rollout/reward_mean`（每 update 每环境步平均奖励）原始点（灰色）与 "
        f"{rolling_window}-update 滑动平均线（蓝色）。下半部分：每个已完成 episode 的累计奖励"
        "散点，颜色区分阶段；虚线为 episode 均值，细红线为 5 个 episode 的滑动平均。"
        "episode 按记录它的 update 窗口定位（在 2,000 步窗口内均匀展开，误差 ≤ 1 个 update）。",
        f"读数：`rollout/reward_mean` 首值 {fmt_num(trend['first'])}、末值 {fmt_num(trend['last'])}，"
        f"前 10% updates 均值 {fmt_num(trend['first_10pct_mean'])}、后 10% updates 均值 "
        f"{fmt_num(trend['last_10pct_mean'])}；"
        + (
            f"共 {eps['count']} 个完成 episode，episode reward 均值 {fmt_num(eps['reward_mean'])}"
            f"（范围 {fmt_num(eps['reward_min'])}–{fmt_num(eps['reward_max'])}）。"
            if eps.get("count")
            else "已纳入阶段内没有 episode 截断记录。"
        ),
    )
    loss = metrics.get("loss", {})
    fig_block(
        "fig_loss.png",
        "图 2：损失曲线",
        "三行分别为总损失 `loss/loss`、策略损失 `loss/policy_loss`、价值损失 "
        "`loss/value_loss`，均为原始点 + 滑动平均。",
        f"读数：总损失范围 [{fmt_num(loss.get('min'))}, {fmt_num(loss.get('max'))}]，"
        f"末值 {fmt_num(loss.get('last'))}；策略损失末值 "
        f"{fmt_num(metrics.get('policy_loss', {}).get('last'))}；价值损失末值 "
        f"{fmt_num(metrics.get('value_loss', {}).get('last'))}。PPO 的总损失/策略损失正负号"
        "依赖优势估计，**负值本身不代表发散**；训练脚本另有 finite 检查与“参数必须变化”断言。"
    )
    ent = metrics.get("entropy", {})
    kl = metrics.get("approx_kl", {})
    clip = metrics.get("clip_fraction", {})
    fig_block(
        "fig_entropy_kl.png",
        "图 3：探索与优化诊断",
        "三行：策略熵（nats，红色虚线为均匀策略上限 ln(7)≈1.946）、近似 KL、"
        "PPO clip fraction。",
        f"读数：熵 {fmt_num(ent.get('first'))} → {fmt_num(ent.get('last'))}"
        f"（均值 {fmt_num(ent.get('mean'))}）；近似 KL 均值 {fmt_num(kl.get('mean'))}"
        f"（最大 {fmt_num(kl.get('max'))}）；clip fraction 均值 {fmt_num(clip.get('mean'))}"
        f"（最大 {fmt_num(clip.get('max'))}）。"
    )
    fig_block(
        "fig_exploration.png",
        "图 4：探索与剧情进度",
        "2×2：`game/coord_count`（走过的唯一坐标数）、`game/max_map_progress`"
        "（官方 essential-map 进度）、`game/num_badges`（徽章数）、`game/event_reward`。"
        "注意这些是 update 级 8-worker 最新 info 平均（episode 截断重置时会回落）；"
        "完整 episode 的极值见 §3.3。",
        f"读数：`coord_count` 末值 {fmt_num(progress['last'].get('coord_count'))}"
        f"（最大 {fmt_num((progress['max'].get('coord_count') or {}).get('value'))}）；"
        f"`max_map_progress` 最大 {fmt_num((progress['max'].get('max_map_progress') or {}).get('value'))}；"
        f"徽章最大 {fmt_num((progress['max'].get('num_badges') or {}).get('value'))}。"
    )
    fig_block(
        "fig_gameplay.png",
        "图 5：游戏内统计",
        "2×2：队伍等级和、队伍人数、死亡数、治疗奖励分量（`game/healr`）。"
        "同样是 update 级 8-worker 最新 info 平均，episode 重置时会回落。",
        f"读数：`levels_sum` 末值 {fmt_num(progress['last'].get('levels_sum'))}"
        f"（最大 {fmt_num((progress['max'].get('levels_sum') or {}).get('value'))}）；"
        f"`party_size` 末值 {fmt_num(progress['last'].get('party_size'))}；"
        f"`deaths` 末值 {fmt_num(progress['last'].get('deaths'))}"
        f"（最大 {fmt_num((progress['max'].get('deaths') or {}).get('value'))}）。"
    )
    if actions:
        last_shares = ", ".join(
            f"{i} {ACTION_NAMES[i]} {fmt_num(actions['last_shares'][ACTION_NAMES[i]])}"
            for i in range(7)
        )
        ent_s = actions.get("entropy") or {}
        fig_block(
            "fig_actions.png",
            "图 6：动作分布与动作熵",
            "上半部分为 7 个按键动作占比的堆叠面积图（动作顺序 0 DOWN、1 LEFT、2 RIGHT、"
            "3 UP、4 A、5 B、6 START），下半部分为动作分布熵（上限 ln(7)≈1.946）。",
            f"读数：末值占比 {last_shares}；动作熵 "
            f"{fmt_num(ent_s.get('first'))} → {fmt_num(ent_s.get('last'))}"
            f"（均值 {fmt_num(ent_s.get('mean'))}）；末次最高占比动作 "
            f"{actions['dominant_action_last']}（{fmt_num(actions['dominant_share_last'])}）。"
            "若某一动作长期占比过高，需要结合评估检查是否存在单一动作刷屏。"
        )
    sps = metrics.get("steps_per_second", {})
    fig_block(
        "fig_throughput.png",
        "图 7：训练吞吐",
        "每个 update 的端到端吞吐（2,000 步 / `time/update_seconds`，含 8 个并行模拟器采样"
        "与 PPO 更新）原始点、滑动平均与总均值线。",
        f"读数：平均 {fmt_num(totals['mean_steps_per_second'], 1)} 环境步/秒；"
        f"范围 {fmt_num(sps.get('min'), 1)}–{fmt_num(sps.get('max'), 1)}；"
        f"总计算时间 {fmt_num(totals['compute_seconds'], 1)} 秒"
        + (
            f"，墙钟跨度 {fmt_num(totals['wall_span_seconds'], 1)} 秒（含阶段间隙）。"
            if totals.get("wall_span_seconds")
            else "。"
        )
    )

    if figure_errors:
        L.append("> ⚠️ 本版有图生成失败（详见 `analysis/run_log.json`）：")
        for err in figure_errors:
            L.append(f"> - {err}")
        L.append("")

    # ---- 3. Key numbers
    L.append("## 3. 关键数字")
    L.append("")
    L.append("### 3.1 阶段汇总")
    L.append("")
    rows = []
    for s in stages:
        rows.append([
            s["name"],
            f"{s['start_steps']:,}–{s['end_steps']:,}",
            f"{s['updates']:,}",
            fmt_num(s["compute_seconds"], 1),
            fmt_num(s["mean_steps_per_second"], 1),
            f"{s['episodes_completed']:,}",
            "完成",
        ])
    # Show running stages that are not included yet, for context.
    for s in status.get("stages_pending_or_running", []):
        match = next((x for x in summary["data_sources"]["stages"] if x["name"] == s), None)
        if match:
            rows.append([
                s,
                f"{match.get('start_steps', 0):,}–{match.get('last_steps', 0):,}（生成时，未纳入）",
                "—", "—", "—", "—", "运行中/待完成",
            ])
    L.append(md_table(
        ["阶段", "环境步区间", "Updates", "计算时间 (s)", "平均吞吐 (steps/s)",
         "完成 episodes", "状态"],
        rows,
    ))
    L.append("")

    L.append("### 3.2 指标统计（按 PPO update）")
    L.append("")
    stat_rows = []
    for key in ("reward_mean", "loss", "policy_loss", "value_loss", "entropy", "approx_kl",
                "clip_fraction", "coord_count", "max_map_progress", "num_badges",
                "event_reward", "levels_sum", "party_size", "deaths", "healr",
                "steps_per_second"):
        st = metrics.get(key)
        if not st:
            continue
        stat_rows.append([key, fmt_num(st["first"]), fmt_num(st["last"]), fmt_num(st["mean"]),
                          fmt_num(st["min"]), fmt_num(st["max"]),
                          f"{st['max_at_steps']:,}"])
    L.append(md_table(
        ["指标", "首值", "末值", "均值", "最小", "最大", "最大出现步数"], stat_rows))
    L.append("")
    L.append("注：首/末值为纳入阶段中第一个/最后一个 update 的值；均值为所有 update 的算术平均；"
             "“最大出现步数”为该指标最大值的环境步位置。其中 `game/*` 是 8 个 worker 最新 info "
             "的 update 内平均（非 episode 最大值，episode 截断时会回落；完整 episode 极值见 §3.3）。")
    L.append("")

    if eps.get("count"):
        L.append("### 3.3 Episode 统计")
        L.append("")
        L.append(md_table(
            ["指标", "数值", "说明"],
            [
                ["完成 episode 数", f"{eps['count']:,}",
                 "每个 episode 在 10,240 步（出现新事件可至 16,384 步）被截断，样本很少"],
                ["episode reward 均值 / 标准差",
                 f"{fmt_num(eps['reward_mean'])} / {fmt_num(eps['reward_std'])}", "episodes.jsonl 的累计奖励"],
                ["episode reward 最小 / 中位 / 最大",
                 f"{fmt_num(eps['reward_min'])} / {fmt_num(eps['reward_median'])} / {fmt_num(eps['reward_max'])}", "方差大，不能作为稳定评估"],
                ["episode 长度均值 / 最小 / 最大（步）",
                 f"{fmt_num(eps['length_mean'], 1)} / {fmt_int(eps['length_min'])} / {fmt_int(eps['length_max'])}",
                 "长于 10,240 表示事件延长了 episode"],
                ["episode 定位方式", eps["step_locating_method"].replace("|", "/"), "用于 fig_reward 散点的 x 轴"],
            ],
        ))
        L.append("")
        by_stage = eps.get("by_stage", {})
        if by_stage:
            L.append(md_table(
                ["阶段", "episode 数", "reward 均值", "reward 最大", "coord 最大",
                 "max_map_progress 最大", "levels_sum 最大", "party_size 最大"],
                [
                    [
                        name,
                        v["count"],
                        fmt_num(v["reward_mean"]),
                        fmt_num(v.get("reward_max")),
                        fmt_num(v.get("coord_count_max")),
                        fmt_num(v.get("max_map_progress_max")),
                        fmt_num(v.get("levels_sum_max")),
                        fmt_num(v.get("party_size_max")),
                    ]
                    for name, v in by_stage.items()
                ],
            ))
            L.append("")
            pe = eps.get("progress_extremes") or {}
            if pe:
                L.append(
                    "**episode 级极值**（`episodes.jsonl`，只统计被截断的完整 episode；"
                    "与 §3.2 的 update 级 8-worker 平均不同）："
                )
                L.append("")
                L.append(md_table(
                    ["指标", "全部 episode 最大值", "说明"],
                    [
                        ["coord_count", fmt_num(pe.get("coord_count_max")), "单 episode 走过的唯一坐标数"],
                        ["max_map_progress", fmt_num(pe.get("max_map_progress_max")), "官方 essential-map 进度（0–14）"],
                        ["levels_sum", fmt_num(pe.get("levels_sum_max")), "队伍等级和"],
                        ["party_size", fmt_num(pe.get("party_size_max")), "队伍人数（0–6）"],
                        ["deaths", fmt_num(pe.get("deaths_max")), "全灭次数"],
                        ["num_badges", fmt_num(pe.get("num_badges_max")), "前两个徽章之和（0–2）"],
                        ["event_reward", fmt_num(pe.get("event_reward_max")), "事件奖励分量累计"],
                        ["mt_moon / cerulean 是否到达",
                         f"{pe.get('mt_moon_any')} / {pe.get('cerulean_any')}",
                         "episode 记录中是否出现过该地图"],
                    ],
                ))
                L.append("")

    if actions:
        L.append("### 3.4 动作分布（纳入阶段最后一个 update）")
        L.append("")
        L.append(md_table(
            ["动作", "末值占比", "历史均值占比", "历史最大占比", "最大出现步数"],
            [
                [f"{i} {ACTION_NAMES[i]}", fmt_num(actions["last_shares"][ACTION_NAMES[i]]),
                 fmt_num(actions["mean_shares"][ACTION_NAMES[i]]),
                 fmt_num(actions["max_shares"][ACTION_NAMES[i]]),
                 f"{actions['max_share_at_steps'][ACTION_NAMES[i]]:,}"]
                for i in range(7)
            ],
        ))
        L.append("")

    L.append("### 3.5 汇总")
    L.append("")
    L.append(
        f"- 累计环境步：**{totals['environment_steps']:,}**；PPO updates：**{totals['ppo_updates']:,}**；"
        f"完成 episode：**{totals['completed_episodes']:,}**。"
    )
    L.append(
        f"- 计算时间合计 **{fmt_num(totals['compute_seconds'], 1)} s**（仅 update 内时间），"
        f"平均吞吐 **{fmt_num(totals['mean_steps_per_second'], 1)} steps/s**，"
        f"聚合吞吐 **{fmt_num(totals['aggregate_steps_per_second'], 1)} steps/s**。"
    )
    L.append(
        f"- 游戏进度：徽章最大 {fmt_num((progress['max'].get('num_badges') or {}).get('value'))}、"
        f"`max_map_progress` 最大 {fmt_num((progress['max'].get('max_map_progress') or {}).get('value'))}、"
        f"`levels_sum` 最大 {fmt_num((progress['max'].get('levels_sum') or {}).get('value'))}、"
        f"`party_size` 最大 {fmt_num((progress['max'].get('party_size') or {}).get('value'))}；"
        f"`mt_moon` 到达 = {progress['milestones_reached'].get('reached_mt_moon')}，"
        f"`cerulean` 到达 = {progress['milestones_reached'].get('reached_cerulean')}。"
    )
    pe_summary = eps.get("progress_extremes") or {}
    if pe_summary:
        L.append(
            f"- episode 级（完整 episode 的最大值）：episode reward 均值 "
            f"{fmt_num(eps.get('reward_mean'))}、最大 {fmt_num(eps.get('reward_max'))}；"
            f"coord_count 最大 {fmt_num(pe_summary.get('coord_count_max'))}；"
            f"max_map_progress 最大 {fmt_num(pe_summary.get('max_map_progress_max'))}（0–14）；"
            f"levels_sum 最大 {fmt_num(pe_summary.get('levels_sum_max'))}；"
            f"party_size 最大 {fmt_num(pe_summary.get('party_size_max'))}；"
            f"num_badges 最大 {fmt_num(pe_summary.get('num_badges_max'))}；"
            f"mt_moon/cerulean 到达 = {pe_summary.get('mt_moon_any')}/{pe_summary.get('cerulean_any')}。"
        )
    L.append("")

    # ---- 4. Limitations
    L.append("## 4. 诚实限制（请勿过度解读）")
    L.append("")
    L.append(
        "1. **训练曲线 ≠ 学会游戏。** 这些曲线只是单个 seed 的训练遥测；上述徽章/剧情指标"
        "没有突破时，不能把奖励上升解释为“学会了玩 Pokémon”。"
    )
    L.append(
        "2. **`rollout/reward_mean` 是每环境步的 shaping reward 均值**（每 update 2,000 步、"
        "8 环境平均），不是 episode return，也不是评估分数。奖励由事件/等级/治疗/探索等分量"
        "加权组成，奖励上升可能来自塑形饱和（例如探索奖励随覆盖饱和），不必然代表策略变强。"
    )
    L.append(
        "3. **episode 样本极少**（阶段 A 不足一个 episode 长度，因此没有 episode 记录），"
        "episode reward 方差大；episode 散点按“记录它的 update 窗口内均匀展开”定位，"
        "x 轴误差可达 1 个 update = 2,000 环境步。"
    )
    L.append(
        "4. **全部数据来自真实本地训练**（PyBoy + CUDA），不是论文复现结果；本地并行规模"
        "（8×250=2,000 步/update）和训练步数都与论文不同，**不能与论文曲线直接对比**。"
    )
    L.append(
        "5. **单 seed、单配置、无重复实验、无置信区间**；曲线末段的统计波动不应被解读为稳定趋势。"
    )
    L.append(
        "6. **训练指标与独立评估口径不同，不可直接比较。** 独立评估（固定协议、checkpoint 推理）"
        "由 eval-engineer 用 `scripts/eval_checkpoint.py` 单独完成，协议与限制见 "
        "[eval_protocol.md](eval_protocol.md)。"
    )
    L.append(
        "7. **计算时间只统计 update 内耗时**（采样 + PPO 更新），不含阶段启动、checkpoint、"
        "评估和分析开销；墙钟跨度另列且含阶段间隙。"
    )
    L.append(
        f"8. 滑动平均窗口 {rolling_window} updates（{rolling_window * 2000:,} 环境步）属于"
        "事后可视化选择，不改变原始指标；阶段 A 只有 5 个 update，其统计意义有限。"
    )
    L.append("9. 各阶段 `result.json` 的 `performance_claim` 均为 `none`；本报告不做性能声明。")
    L.append(
        "10. **update 级 `game/*` 不是 episode 最大值。** 它们是 8 个 worker 最新 info 的 "
        "update 内平均（每个 worker 的当前 episode 累计进度），episode 截断时会回落成锯齿；"
        "完整 episode 的极值见 §3.3，两者不可混读。"
    )
    if status["stages_pending_or_running"]:
        L.append(
            f"11. **本版缺少 {'、'.join(status['stages_pending_or_running'])}**（生成时仍在运行），"
            "图表与统计只覆盖已完成阶段；阶段 D 完成后重跑脚本即可得到完整的 1,000,000 步版本。"
        )
    elif pe_summary:
        L.append(
            f"11. **截至 {totals['environment_steps']:,} 步**：episode reward 均值 "
            f"{fmt_num(eps.get('reward_mean'))}（最大 {fmt_num(eps.get('reward_max'))}），"
            f"episode 级最深 max_map_progress = {fmt_num(pe_summary.get('max_map_progress_max'))}（0–14），"
            f"coord_count 最大 {fmt_num(pe_summary.get('coord_count_max'))}，"
            f"但徽章仍为 0、未到月见山/华蓝市。奖励与探索指标在上升，"
            "**仍不能据此声称模型学会了玩 Pokémon**。"
        )
    L.append("")

    # ---- 5. Reproduction
    L.append("## 5. 复现与更新")
    L.append("")
    L.append("```powershell")
    L.append("# 默认：自动纳入所有已完成阶段（Stage D 完成后自动纳入，得到完整 1M 版本）")
    L.append(r".venv\Scripts\python.exe scripts\plot_training_curves.py")
    L.append("")
    L.append("# 预览：把仍在运行阶段的当前数据也画进来（非最终版）")
    L.append(r".venv\Scripts\python.exe scripts\plot_training_curves.py --include-running")
    L.append("```")
    L.append("")
    L.append(
        "脚本只读 `stage_*/metrics.jsonl` 与 `stage_*/episodes.jsonl`，输出："
        "7 张 PNG（`docs/reports/figures/`）、"
        "`analysis/training_summary.json` + `.md`、`analysis/metrics_10k.csv`、"
        "本文件；每次运行会向 `analysis/run_log.json` 追加一条运行记录（时间戳、纳入阶段、"
        "步数范围、图文件、错误）。"
    )
    L.append("")
    if status["stages_pending_or_running"]:
        L.append(
            f"> **阶段 D 完成后重跑**：当前 {totals['environment_steps']:,} / 1,000,000 环境步；"
            "保留同一命令重跑即可把 Stage D 纳入，并更新全部图表、JSON/MD 与本报告。"
        )
        L.append("")

    # ---- 6. Related docs
    L.append("## 6. 相关文档")
    L.append("")
    L.append("- [独立评估协议 v1（eval_protocol.md）](eval_protocol.md)：checkpoint 评估协议、"
             "指标口径与已知限制。")
    L.append("- [训练监督 Phase 1（training_supervisor.md）](training_supervisor.md)："
             "环境审查、并行规模基准与阶段计划。")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------
# run_log.json
# --------------------------------------------------------------------------------------
def append_run_log(path: Path, entry: dict) -> None:
    data: dict = {"runs": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                data = loaded
        except (OSError, ValueError):
            data = {"runs": []}
    entry = dict(entry)
    entry["run_index"] = len(data["runs"]) + 1
    data["runs"].append(entry)
    data["last_updated"] = iso(now_local())
    write_json_atomic(path, data)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Pokemon Red PPO training curves from real environment-step logs "
                    "(read-only; no training, no evaluation, no GPU)."
    )
    p.add_argument("--root", default="artifacts/training/seed_0",
                   help="experiment root containing stage_* directories "
                        "(default: artifacts/training/seed_0)")
    p.add_argument("--figures-dir", default="docs/reports/figures",
                   help="PNG output directory (default: docs/reports/figures)")
    p.add_argument("--analysis-dir", default=None,
                   help="summary/CSV/run-log output directory "
                        "(default: <root>/analysis)")
    p.add_argument("--report", default="docs/reports/training_curves.md",
                   help="Chinese report output path")
    p.add_argument("--rolling-window", type=int, default=ROLLING_WINDOW,
                   help="rolling mean window in PPO updates (default: 20)")
    p.add_argument("--include-running", action="store_true",
                   help="also include stages that are still running (no result.json). "
                        "Temporary preview only; default excludes them.")
    p.add_argument("--bin-steps", type=int, default=BIN_STEPS,
                   help="environment-step bin width for metrics_10k.csv (default: 10000)")
    return p.parse_args(argv)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv if argv is not None else sys.argv[1:])
    root = resolve(args.root)
    figures_dir = resolve(args.figures_dir)
    analysis_dir = resolve(args.analysis_dir) if args.analysis_dir else root / "analysis"
    report_path = resolve(args.report)
    rolling_window = max(2, int(args.rolling_window))

    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    print(f"[plot] experiment root: {root}")
    stages_all, discovery_warnings = discover_stages(root)
    if not stages_all:
        print("[plot] ERROR: no stage_* directory with metrics.jsonl found", file=sys.stderr)
        return 2

    included = [s for s in stages_all if s["complete"] or args.include_running]
    included_names = {s["name"] for s in included}
    pending = [s for s in stages_all if s["name"] not in included_names]
    if not included:
        print("[plot] ERROR: no completed stage with data yet; nothing to plot", file=sys.stderr)
        return 2
    for s in stages_all:
        state = "complete" if s["complete"] else (
            "running (included, partial preview)" if s["name"] in included_names
            else "running/excluded"
        )
        print(f"[plot] stage {s['name']:<18} steps {s['start_steps']:>9,}-{s['end_steps']:>9,} "
              f"updates={len(s['metrics_rows']):>4} episodes={len(s['episodes']):>4} -> {state}")
    if pending:
        print(f"[plot] WARNING: pending stages not included: "
              f"{', '.join(s['name'] for s in pending)}. "
              f"Rerun after completion for the full run. Pass --include-running for a preview.")

    run, duplicate_rows_dropped = build_metrics(included)
    if run.n == 0:
        print("[plot] ERROR: merged metrics frame is empty", file=sys.stderr)
        return 2
    eps, ep_mapping_ok = build_episodes(included, run)
    if not ep_mapping_ok:
        print("[plot] WARNING: episode step mapping incomplete; falling back to episode index")

    summary = build_summary(
        root=root,
        stages_all=stages_all,
        stages_included=included,
        stages_pending=pending,
        run=run,
        eps=eps,
        ep_mapping_ok=ep_mapping_ok,
        duplicate_rows_dropped=duplicate_rows_dropped,
        discovery_warnings=discovery_warnings,
    )

    # ---- figures -------------------------------------------------------------------
    figure_errors: list[str] = []
    figures: list[dict] = []
    builders = [
        ("fig_reward.png", lambda: fig_reward(run, eps, ep_mapping_ok, included, pending,
                                             figures_dir, rolling_window)),
        ("fig_loss.png", lambda: fig_loss(run, included, pending, figures_dir, rolling_window)),
        ("fig_entropy_kl.png", lambda: fig_entropy_kl(run, included, pending, figures_dir,
                                                     rolling_window)),
        ("fig_exploration.png", lambda: fig_exploration(run, included, pending, figures_dir,
                                                       rolling_window)),
        ("fig_gameplay.png", lambda: fig_gameplay(run, included, pending, figures_dir,
                                                 rolling_window)),
        ("fig_actions.png", lambda: fig_actions(run, included, pending, figures_dir,
                                               rolling_window)),
        ("fig_throughput.png", lambda: fig_throughput(run, included, pending, figures_dir,
                                                     rolling_window)),
    ]
    for fname, builder in builders:
        try:
            info = builder()
            figures.append(info)
            print(f"[plot] wrote {fname} ({info['bytes']:,} bytes)")
        except Exception as exc:  # keep going so a single panel bug does not lose everything
            figure_errors.append(f"{fname}: {exc}")
            print(f"[plot] ERROR building {fname}: {exc}", file=sys.stderr)
            traceback.print_exc()

    # ---- CSV / JSON / Markdown ------------------------------------------------------
    csv_rows = write_10k_csv(analysis_dir / "metrics_10k.csv", run,
                             max(1000, int(args.bin_steps)))
    summary["figure_files"] = figures
    summary["figure_errors"] = figure_errors
    write_json_atomic(analysis_dir / "training_summary.json", summary)
    write_summary_md(analysis_dir / "training_summary.md", summary)
    write_report_md(report_path, summary, figures, figure_errors, rolling_window)

    duration = time.perf_counter() - started
    run_entry = {
        "generated_at": summary["generated_at"],
        "duration_seconds": round(duration, 2),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "root": str(root),
        "include_running": bool(args.include_running),
        "stages_included": [
            {"name": s["name"], "steps": [int(s["start_steps"]), int(s["end_steps"])],
             "updates": len(s["metrics_rows"]), "complete": bool(s["complete"])}
            for s in included
        ],
        "stages_pending": [s["name"] for s in pending],
        "environment_steps_total": summary["totals"]["environment_steps"],
        "updates_total": summary["totals"]["ppo_updates"],
        "compute_seconds_total": summary["totals"]["compute_seconds"],
        "episodes_total": summary["totals"]["completed_episodes"],
        "figures": [{"file": f["file"], "bytes": f["bytes"]} for f in figures],
        "outputs": {
            "figures_dir": str(figures_dir),
            "training_summary_json": str(analysis_dir / "training_summary.json"),
            "training_summary_md": str(analysis_dir / "training_summary.md"),
            "metrics_10k_csv": str(analysis_dir / "metrics_10k.csv"),
            "report_md": str(report_path),
        },
        "figure_errors": figure_errors,
        "discovery_warnings": discovery_warnings,
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "note": (
            "Stage D not complete yet; rerun after completion for the full 1,000,000-step version."
            if pending else "All stages complete; full run."
        ),
    }
    append_run_log(analysis_dir / "run_log.json", run_entry)

    print("")
    print(f"[plot] stages included: {', '.join(s['name'] for s in included)}")
    print(f"[plot] environment steps: {summary['totals']['environment_steps']:,} | "
          f"updates: {summary['totals']['ppo_updates']:,} | "
          f"episodes: {summary['totals']['completed_episodes']:,}")
    print(f"[plot] compute: {summary['totals']['compute_seconds']:,.1f} s | "
          f"mean throughput: {fmt_num(summary['totals']['mean_steps_per_second'], 1)} steps/s")
    print(f"[plot] figures: {len(figures)} -> {figures_dir}")
    print(f"[plot] summary JSON: {analysis_dir / 'training_summary.json'}")
    print(f"[plot] summary MD:   {analysis_dir / 'training_summary.md'}")
    print(f"[plot] 10k CSV:      {analysis_dir / 'metrics_10k.csv'} ({csv_rows} rows)")
    print(f"[plot] report:       {report_path}")
    print(f"[plot] run log:      {analysis_dir / 'run_log.json'}")
    print(f"[plot] done in {duration:.1f} s (single-threaded, read-only analysis)")
    if figure_errors:
        print(f"[plot] ERROR: {len(figure_errors)} figure(s) failed to build", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
