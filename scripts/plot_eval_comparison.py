"""Read-only comparison plot for the seed_0 independent evaluation results.

Reads the per-run `results.json` files produced by `scripts/eval_checkpoint.py`
(protocol v1 and the v1-ext extension) and renders:

    docs/reports/figures/fig_eval_comparison.png

plus a small CSV next to the figure. This script performs no evaluation, no model
loading and no environment interaction; it only aggregates recorded JSON numbers.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "artifacts" / "training" / "seed_0" / "eval"
FIG_DIR = ROOT / "docs" / "reports" / "figures"

# label -> evaluation output directory name
RUNS = [
    ("untrained", "baseline_untrained", 0),
    ("10k", "eval_10k", 10000),
    ("100k", "eval_100k", 100000),
    ("300k", "eval_300k", 300000),
    ("500k", "eval_500k", 500000),
    ("1M", "eval_1m", 1000000),
    ("random actions", "baseline_random_actions", None),
]
EXT_RUNS = [
    ("untrained", "ext_untrained", 0),
    ("300k", "ext_300k", 300000),
    ("500k", "ext_500k", 500000),
    ("1M", "ext_1m", 1000000),
]

FIELDS = ("episode_reward", "coord_count", "unique_map_count", "max_map_progress", "levels_sum", "deaths")


def load_group(directory, group="stochastic"):
    path = EVAL_DIR / directory / "results.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {}).get(group)
    if not summary:
        return None
    return {
        "n": summary["n"],
        "mean": summary["mean"],
        "std": summary.get("std", {}),
        "variants": data.get("protocol", {}).get("steps_per_episode"),
        "checkpoint_steps": (data.get("checkpoint") or {}).get("environment_steps"),
        "eval_steps": data.get("protocol", {}).get("totals", {}).get("eval_steps"),
    }


def style_axes(ax, title, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(RUNS)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    fig.suptitle(
        "Independent evaluation of seed_0 checkpoints (protocol v1: 4096 steps x 3 stochastic episodes; "
        "v1-ext: 8192 steps x 5)",
        fontsize=13,
        fontweight="bold",
    )

    # --- Panel A: episode reward, v1 -------------------------------------
    ax = axes[0][0]
    labels, means, stds, variants = [], [], [], []
    for label, directory, _steps in RUNS:
        group = load_group(directory)
        if group is None:
            continue
        labels.append(label)
        means.append(group["mean"]["episode_reward"])
        stds.append(group["std"].get("episode_reward", 0.0))
        variants.append("v1")
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=4, color=colors[: len(labels)], alpha=0.85)
    ax.set_xticks(positions, labels, fontsize=10)
    style_axes(ax, "Episode reward - stochastic policy (v1, n=3)", "episode reward (shaped)")
    ax.axhline(0, color="black", linewidth=0.6)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.08, f"{m:.1f}", ha="center", fontsize=8)

    # --- Panel B: episode reward, v1-ext ---------------------------------
    ax = axes[0][1]
    labels, means, stds = [], [], []
    ext_colors = plt.cm.plasma(np.linspace(0.15, 0.8, len(EXT_RUNS)))
    for label, directory, _steps in EXT_RUNS:
        group = load_group(directory)
        if group is None:
            continue
        labels.append(label)
        means.append(group["mean"]["episode_reward"])
        stds.append(group["std"].get("episode_reward", 0.0))
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=4, color=ext_colors[: len(labels)], alpha=0.85)
    ax.set_xticks(positions, labels, fontsize=10)
    style_axes(ax, "Episode reward - stochastic policy (v1-ext, n=5, 8192 steps)", "episode reward (shaped)")
    ax.axhline(0, color="black", linewidth=0.6)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.08, f"{m:.1f}", ha="center", fontsize=8)

    # --- Panel C: coord_count, both protocols ----------------------------
    ax = axes[1][0]
    width = 0.38
    v1_labels, v1_means, v1_stds = [], [], []
    for label, directory, _steps in RUNS:
        group = load_group(directory)
        if group is None:
            continue
        v1_labels.append(label)
        v1_means.append(group["mean"]["coord_count"])
        v1_stds.append(group["std"].get("coord_count", 0.0))
    ext_lookup = {}
    for label, directory, _steps in EXT_RUNS:
        group = load_group(directory)
        if group is not None:
            ext_lookup[label] = group
    positions = np.arange(len(v1_labels))
    ax.bar(positions - width / 2, v1_means, width, yerr=v1_stds, capsize=3,
           color="#4477aa", alpha=0.85, label="v1 (4096 steps, n=3)")
    ext_means = [ext_lookup.get(label, {}).get("mean", {}).get("coord_count") for label in v1_labels]
    ext_stds = [ext_lookup.get(label, {}).get("std", {}).get("coord_count") for label in v1_labels]
    plot_positions, plot_means, plot_stds = [], [], []
    for i, (m, s) in enumerate(zip(ext_means, ext_stds)):
        if m is not None:
            plot_positions.append(positions[i] + width / 2)
            plot_means.append(m)
            plot_stds.append(s or 0.0)
    ax.bar(plot_positions, plot_means, width, yerr=plot_stds, capsize=3,
           color="#ee7733", alpha=0.85, label="v1-ext (8192 steps, n=5)")
    ax.set_xticks(positions, v1_labels, fontsize=10)
    style_axes(ax, "Unique coordinates explored", "coord_count (unique x,y,map)")
    ax.legend(fontsize=8)

    # --- Panel D: maps visited, both protocols --------------------------
    ax = axes[1][1]
    v1_means = []
    for label, directory, _steps in RUNS:
        group = load_group(directory)
        if group is None:
            continue
        v1_means.append(group["mean"]["unique_map_count"])
    ax.bar(positions - width / 2, v1_means, width, color="#4477aa", alpha=0.85, label="v1")
    ext_means = [ext_lookup.get(label, {}).get("mean", {}).get("unique_map_count") for label in v1_labels]
    plot_positions, plot_means = [], []
    for i, m in enumerate(ext_means):
        if m is not None:
            plot_positions.append(positions[i] + width / 2)
            plot_means.append(m)
    ax.bar(plot_positions, plot_means, width, color="#ee7733", alpha=0.85, label="v1-ext")
    ax.set_xticks(positions, v1_labels, fontsize=10)
    style_axes(ax, "Maps visited per episode", "unique map count")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 5.5)

    fig.text(
        0.01,
        0.005,
        "Source: real local evaluation of seed_0 checkpoints on the real Pokemon Red ROM (scripts/eval_checkpoint.py).\n"
        "Error bars are the sample std over stochastic episodes (n=3 or n=5); they are not confidence intervals.\n"
        "Deterministic argmax episodes are degenerate (single-action lock) and are not used here. NOT paper results.",
        fontsize=7,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = FIG_DIR / "fig_eval_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    # --- CSV ------------------------------------------------------------
    rows = []
    for label, directory, steps in RUNS:
        group = load_group(directory)
        if group is None:
            continue
        rows.append({
            "protocol": "v1",
            "run": directory,
            "training_steps": steps,
            "n": group["n"],
            **{f"{field}_mean": round(group["mean"].get(field, float("nan")), 4) for field in FIELDS},
            **{f"{field}_std": round(group["std"].get(field, float("nan")), 4) for field in FIELDS},
        })
    for label, directory, steps in EXT_RUNS:
        group = load_group(directory)
        if group is None:
            continue
        rows.append({
            "protocol": "v1-ext",
            "run": directory,
            "training_steps": steps,
            "n": group["n"],
            **{f"{field}_mean": round(group["mean"].get(field, float("nan")), 4) for field in FIELDS},
            **{f"{field}_std": round(group["std"].get(field, float("nan")), 4) for field in FIELDS},
        })
    csv_path = FIG_DIR / "fig_eval_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
