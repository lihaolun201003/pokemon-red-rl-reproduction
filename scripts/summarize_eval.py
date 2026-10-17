"""Summarize fixed-protocol evaluation results into one JSON summary and a Markdown report.

This is a read-only helper around scripts/eval_checkpoint.py: it reads existing
results.json files and writes two report files. It never touches the environment,
the model, the GPU or training run directories (it only reads their evaluation
results), and it never modifies the source results.

Inputs may be:
    * one or more results.json files, or
    * one or more directories: the directory itself if it contains results.json,
      else its immediate sub-directories that contain results.json, else a
      recursive search for results.json.

Outputs (defaults match the seed_0 experiment layout):
    artifacts/training/seed_0/eval/eval_summary.json
    docs/reports/eval_results.md

The report keeps the deterministic episode separate from the stochastic episodes:
a deterministic (argmax) episode has a unique trajectory (variance 0 by
construction), while the stochastic mean/std is a 3-episode sample statistic.
Evaluation steps are explicitly not part of the training budget.

Usage (PowerShell, from the repository root):
    .venv\\Scripts\\python.exe scripts/summarize_eval.py `
        artifacts/training/seed_0/eval/eval_10k `
        artifacts/training/seed_0/eval/eval_100k `
        artifacts/training/seed_0/eval/eval_300k `
        artifacts/training/seed_0/eval/eval_1m `
        artifacts/training/seed_0/eval/baseline_untrained `
        artifacts/training/seed_0/eval/baseline_random_actions

    # or pass one parent directory and let discovery find every child result
    # (--exclude keeps CUDA smoke / self-test output out of the comparison tables):
    .venv\\Scripts\\python.exe scripts/summarize_eval.py artifacts/training/seed_0/eval `
        --exclude _cuda_smoke --exclude _selftest --exclude _demos --overwrite
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PROTOCOL = "eval-protocol-v1"
DEFAULT_JSON_OUT = ROOT / "artifacts/training/seed_0/eval/eval_summary.json"
DEFAULT_MD_OUT = ROOT / "docs/reports/eval_results.md"
GROUP_ORDER = ("stochastic", "deterministic", "random_actions")
ACTION_NAMES = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]
UNIFORM_ACTION_ENTROPY = math.log2(len(ACTION_NAMES))  # ~2.807 bits

# (field name, decimal digits) used by the Markdown tables.
MEAN_STD_FIELDS = (
    ("episode_reward", 4),
    ("length", 1),
    ("coord_count", 2),
    ("unique_map_count", 2),
    ("badges", 2),
    ("levels_sum", 2),
    ("party_size", 2),
    ("deaths", 2),
    ("max_map_progress", 2),
    ("mt_moon", 3),
    ("cerulean", 3),
)
SINGLE_VALUE_FIELDS = (
    ("episode_reward", 4),
    ("length", 1),
    ("coord_count", 2),
    ("unique_map_count", 2),
    ("badges", 2),
    ("levels_sum", 2),
    ("party_size", 2),
    ("deaths", 2),
    ("max_map_progress", 2),
    ("mt_moon", 3),
    ("cerulean", 3),
)

REPORT_NOTES = [
    "确定性评估方差为 0：每个 run 只有 1 个 deterministic episode（策略 argmax）；初始存档固定 "
    "Game Boy RNG，且每个 episode 都新建 PyBoy 环境实例，因此同一 checkpoint 的确定性轨迹唯一。"
    "deterministic 表中的数值不是统计估计，方差按定义为 0，不参与显著性比较，也不应与其他 run "
    "的 deterministic 值做方差检验。",
    "stochastic 评估方差来自策略采样：v1 每个 run 3 个 stochastic episode（seed = base seed + 0..2），"
    "v1-ext 每个 run 5 个（seed = base seed + 0..4），base seed=100000；表中 mean±std 是该组 "
    "episode 的样本统计（n=3 或 n=5），样本很小，std 只用于观察波动，不应据此做显著性结论。",
    "评估步数不属于训练预算：eval_steps = episodes × steps_per_episode，只为评估推理消耗；"
    "它不推进训练计数器、不进入 PPO 更新、也不写入训练 run 目录。表中“训练步数”列来自 "
    "checkpoint.environment_steps（未训练/随机基线的训练步数为 0）。",
    "eval 秒数与吞吐包含环境构建、reset 与视频编码，且不同设备（CPU/GPU）与机器负载下不可直接"
    "横向比较；吞吐列仅作本机运行记录。",
    "动作熵：基于该组 episode 聚合动作计数计算的 Shannon 熵（bit），7 个动作均匀分布时最大为 "
    f"log2(7)≈{UNIFORM_ACTION_ENTROPY:.3f}；最大动作占比 = 出现最多的单一动作的比例。",
    "官方环境只记录前两个徽章（badges = badge_1 + badge_2）；mt_moon / cerulean 是 0/1 标记的"
    "组内均值（到达过该地点的 episode 比例）。",
    "所有数值直接来自各 results.json，按原样聚合，未做筛选、平滑或人工修饰。",
]

VIDEO_NOTES = [
    "deterministic 视频说明：eval_*/baseline_untrained 目录里的 mp4 录制的是 argmax deterministic "
    "episode；该策略在本 run 中退化为重复单一按键（见 deterministic 表，coords=1），因此这些视频"
    "只作轨迹证据，不代表模型的真实探索能力。",
    "随机策略演示视频为另外补录（第一个 stochastic episode、2048 步、不属于 v1/v1-ext 协议内评估）："
    "artifacts/training/seed_0/eval/videos/demo_1m_stochastic.mp4 与 demo_300k_stochastic.mp4；"
    "它们用于人工复查最佳模型的探索行为，不参与聚合表、不改变协议定义。",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error(message):
    raise SystemExit(f"[summarize] ERROR: {message}")


def _sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover(inputs, excludes):
    """Resolve CLI inputs to a de-duplicated, ordered list of results.json paths."""
    found = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path)
        path = path.resolve()
        if not path.exists():
            _error(f"input does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() != ".json":
                _error(f"input file is not JSON: {path}")
            found.append(path)
            continue
        direct = path / "results.json"
        if direct.is_file():
            found.append(direct)
            continue
        children = sorted(child / "results.json" for child in path.iterdir()
                          if child.is_dir() and (child / "results.json").is_file())
        if children:
            found.extend(children)
            continue
        nested = sorted(path.rglob("results.json"))
        if not nested:
            _error(f"no results.json found under directory: {path}")
        found.extend(nested)
    unique, seen = [], set()
    for path in found:
        text = str(path)
        if text in seen:
            continue
        if any(exclude and exclude in text for exclude in excludes):
            continue
        seen.add(text)
        unique.append(path)
    if not unique:
        _error("no results.json inputs left after discovery/exclusion")
    return unique


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt_int(value):
    if value is None:
        return "-"
    if not _is_number(value):
        return str(value)
    return f"{int(value):,}"


def _fmt(value, digits):
    if value is None:
        return "-"
    if not _is_number(value):
        return str(value)
    return f"{value:,.{digits}f}"


def _mean_std_cell(group, field, digits):
    fields = group.get("fields", {})
    field_stats = fields.get(field)
    if not field_stats or field_stats.get("mean") is None:
        return "-"
    mean = field_stats["mean"]
    std = field_stats.get("std") or 0.0
    return f"{mean:,.{digits}f} ± {std:,.{digits}f}"


def _single_cell(group, field, digits):
    fields = group.get("fields", {})
    field_stats = fields.get(field)
    if not field_stats or field_stats.get("mean") is None:
        return "-"
    return _fmt(field_stats["mean"], digits)


def _entropy(shares):
    total = 0.0
    for share in shares:
        if share and share > 0:
            total -= share * math.log2(share)
    return round(total, 4)


def _display_path(path):
    """Short human-readable path for Markdown tables (repository-relative if possible)."""
    if not path:
        return "-"
    try:
        return str(Path(path).resolve().relative_to(ROOT).as_posix())
    except (ValueError, OSError):
        return str(path)


def _episode_summary(episode):
    """Compact, report-relevant view of one episode record."""
    return {
        "index": episode.get("index"),
        "mode": episode.get("mode"),
        "seed": episode.get("seed"),
        "env_seed": episode.get("env_seed"),
        "length": episode.get("length"),
        "terminated_by": episode.get("terminated_by"),
        "episode_reward": episode.get("episode_reward"),
        "coord_count": episode.get("coord_count"),
        "unique_map_count": episode.get("unique_map_count"),
        "badges": episode.get("badges"),
        "levels_sum": episode.get("levels_sum"),
        "party_size": episode.get("party_size"),
        "deaths": episode.get("deaths"),
        "max_map_progress": episode.get("max_map_progress"),
        "mt_moon": episode.get("mt_moon"),
        "cerulean": episode.get("cerulean"),
        "max_foe_level": episode.get("max_foe_level"),
        "obs_events_sum": episode.get("obs_events_sum"),
        "steps_per_second": episode.get("steps_per_second"),
        "reward_consistency": episode.get("reward_consistency"),
        "action_counts": episode.get("action_counts"),
        "maps_first_step": episode.get("maps_first_step"),
        "video": (episode.get("video") or {}).get("path"),
        "screenshots": sorted((episode.get("screenshots") or {}).keys()),
    }


def _aggregate_group(group_name, group_summary, episode_records):
    """Aggregate one episode group, including action entropy and per-episode rows."""
    means = group_summary.get("mean") or {}
    stds = group_summary.get("std") or {}
    mins = group_summary.get("min") or {}
    maxs = group_summary.get("max") or {}
    fields = {}
    for field in sorted(set(means) | set(stds) | set(mins) | set(maxs)):
        fields[field] = {
            "mean": means.get(field),
            "std": stds.get(field),
            "min": mins.get(field),
            "max": maxs.get(field),
        }
    counts = list(group_summary.get("action_counts_total") or [])
    total = sum(counts)
    shares = [round(count / total, 6) for count in counts] if total else []
    episodes = [_episode_summary(episode) for episode in episode_records
                if episode.get("mode") == group_name]
    return {
        "n": group_summary.get("n", len(episodes)),
        "fields": fields,
        "episode_reward_sum": group_summary.get("episode_reward_sum"),
        "action_names": ACTION_NAMES,
        "action_counts_total": counts,
        "action_shares_total": shares,
        "action_entropy_bits": _entropy(shares),
        "action_top_share": round(max(shares), 6) if shares else None,
        "episodes": episodes,
    }


def _load_run(path, ext_prefix="ext_"):
    """Read one results.json and turn it into a report row."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _error(f"could not read results JSON {path}: {type(exc).__name__}: {exc}")
    if not isinstance(data, dict):
        _error(f"{path} is not a JSON object")
    protocol = data.get("protocol")
    episodes = data.get("episodes")
    if not isinstance(protocol, dict) or not isinstance(episodes, list):
        _error(f"{path} is not an eval-protocol-v1 results.json (missing protocol/episodes)")
    if protocol.get("name") != SUPPORTED_PROTOCOL:
        _error(f"{path} uses unsupported protocol {protocol.get('name')!r}; expected {SUPPORTED_PROTOCOL}")
    if not episodes:
        _error(f"{path} contains no episode records")

    checkpoint = data.get("checkpoint")
    model = data.get("model") or {}
    summary = data.get("summary") or {}
    totals = protocol.get("totals") or {}
    checkpoint_metadata = None
    training_steps = 0
    if isinstance(checkpoint, dict):
        training_steps = checkpoint.get("environment_steps") or 0
        checkpoint_metadata = {
            key: checkpoint.get(key) for key in
            ("path", "sha256", "size_bytes", "environment_steps", "agent_label",
             "phase1_format", "next_update", "update", "seed", "run_id", "model_parameters")
        }
    run_mode = protocol.get("mode")
    if run_mode == "checkpoint":
        training_step_label = f"{int(training_steps):,}"
    elif run_mode == "untrained":
        training_step_label = "0 (untrained)"
    elif run_mode == "random_actions":
        training_step_label = "0 (random init)"
    else:
        training_step_label = _fmt_int(training_steps)

    row = {
        "label": path.parent.name or "results",
        "protocol_variant": "v1-ext" if (path.parent.name or "").startswith(ext_prefix) else "v1",
        "source": str(path),
        "source_sha256": _sha256(path),
        "run_mode": run_mode,
        "device": protocol.get("device"),
        "training_environment_steps": int(training_steps) if training_steps else 0,
        "training_step_label": training_step_label,
        "checkpoint": checkpoint_metadata,
        "model": {
            "source": model.get("source"),
            "parameters": model.get("parameters"),
            "finite": model.get("finite"),
        },
        "protocol": {
            "name": protocol.get("name"),
            "schema": protocol.get("schema"),
            "mode": run_mode,
            "created_utc": protocol.get("created_utc"),
            "steps_per_episode": protocol.get("steps_per_episode"),
            "episodes_requested": protocol.get("episodes_requested"),
            "seed": protocol.get("seed"),
            "no_stochastic": protocol.get("no_stochastic"),
            "stochastic_evaluation": protocol.get("stochastic_evaluation"),
            "device": protocol.get("device"),
            "eval_steps": totals.get("eval_steps"),
            "eval_seconds": totals.get("protocol_wall_seconds"),
            "steps_per_second": totals.get("steps_per_second"),
            "video": protocol.get("video"),
        },
        "groups": {},
        "warnings": [],
    }

    for group_name in GROUP_ORDER:
        if group_name in summary:
            row["groups"][group_name] = _aggregate_group(
                group_name, summary[group_name], episodes)
    if not row["groups"]:
        row["warnings"].append("results.json has no aggregated episode groups")

    # Cross-check declared totals against the per-episode lengths.
    lengths = [episode.get("length") for episode in episodes if _is_number(episode.get("length"))]
    if lengths:
        actual_steps = int(sum(lengths))
        declared_steps = totals.get("eval_steps")
        if declared_steps is not None and int(declared_steps) != actual_steps:
            row["warnings"].append(
                f"declared eval_steps={declared_steps} but episode lengths sum to {actual_steps}")
    env_truncations = [episode.get("index") for episode in episodes
                       if episode.get("terminated_by") == "env_truncation"]
    if env_truncations:
        row["warnings"].append(
            f"{len(env_truncations)} episode(s) ended by the environment's own step budget "
            f"(index {env_truncations}), before --steps")
    inconsistent = [episode.get("index") for episode in episodes
                    if episode.get("reward_consistency") is False]
    if inconsistent:
        row["warnings"].append(f"reward_consistency=False for episode(s) {inconsistent}")
    fixed_steps = protocol.get("steps_per_episode")
    complete = [episode for episode in episodes if episode.get("terminated_by") == "step_cap"]
    wrong_lengths = [episode.get("index") for episode in complete
                     if episode.get("length") != fixed_steps]
    if wrong_lengths:
        row["warnings"].append(
            f"step_cap episode(s) {wrong_lengths} have length != steps_per_episode={fixed_steps}")
    return row


def _sort_rows(rows):
    def key(row):
        variant_rank = 0 if row["protocol_variant"] == "v1" else 1
        mode = row["run_mode"]
        steps = row["training_environment_steps"] or 0
        if mode == "checkpoint":
            return (variant_rank, 0, steps, row["label"])
        if mode == "untrained":
            return (variant_rank, 1, 0, row["label"])
        if mode == "random_actions":
            return (variant_rank, 2, 0, row["label"])
        return (variant_rank, 3, steps, row["label"])

    rows = sorted(rows, key=key)
    seen = {}
    for row in rows:
        base = row["label"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            row["label"] = f"{base}_{seen[base]}"
    return rows


def _protocol_warnings(rows):
    warnings = []
    variants = {}
    for row in rows:
        variants.setdefault(row["protocol_variant"], []).append(row)
    for variant in sorted(variants):
        subset = variants[variant]
        step_values = sorted({row["protocol"].get("steps_per_episode") for row in subset
                              if row["protocol"].get("steps_per_episode") is not None})
        if len(step_values) > 1:
            warnings.append(f"protocol mismatch within {variant}: steps_per_episode differs: "
                            f"{step_values}")
        seeds = sorted({row["protocol"].get("seed") for row in subset
                        if row["protocol"].get("seed") is not None})
        if len(seeds) > 1:
            warnings.append(f"base seeds differ within {variant}: {seeds}")
    devices = sorted({str(row["device"]) for row in rows})
    if len(devices) > 1:
        warnings.append(f"devices differ across inputs: {devices}; eval seconds/throughput "
                        f"are not directly comparable")
    for row in rows:
        if row["run_mode"] == "checkpoint" and row["checkpoint"] is None:
            warnings.append(f"{row['label']}: checkpoint metadata missing")
        if row["run_mode"] in ("checkpoint", "untrained"):
            if "deterministic" not in row["groups"]:
                warnings.append(f"{row['label']}: no deterministic episode group")
            if not row["protocol"].get("no_stochastic") and "stochastic" not in row["groups"]:
                warnings.append(f"{row['label']}: no stochastic episode group")
    return warnings


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for cells in rows:
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return lines


def _stochastic_row(row):
    group = row["groups"]["stochastic"]
    return [
        row["label"],
        row["training_step_label"],
        _fmt_int(group.get("n")),
        _mean_std_cell(group, "episode_reward", 4),
        _mean_std_cell(group, "length", 1),
        _mean_std_cell(group, "coord_count", 2),
        _mean_std_cell(group, "unique_map_count", 2),
        _mean_std_cell(group, "badges", 2),
        _mean_std_cell(group, "levels_sum", 2),
        _mean_std_cell(group, "party_size", 2),
        _mean_std_cell(group, "deaths", 2),
        _mean_std_cell(group, "max_map_progress", 2),
        _mean_std_cell(group, "mt_moon", 3),
        _mean_std_cell(group, "cerulean", 3),
        _fmt(group.get("action_entropy_bits"), 3),
        _fmt(group.get("action_top_share"), 3),
        _fmt_int(row["protocol"].get("eval_steps")),
        _fmt(row["protocol"].get("eval_seconds"), 1),
        _fmt(row["protocol"].get("steps_per_second"), 1),
    ]


def _deterministic_row(row):
    group = row["groups"]["deterministic"]
    episode = group["episodes"][0] if group.get("episodes") else {}
    video = _display_path(episode.get("video"))
    return [
        row["label"],
        row["training_step_label"],
        str(episode.get("terminated_by") or "-"),
        _single_cell(group, "length", 1),
        _single_cell(group, "episode_reward", 4),
        _single_cell(group, "coord_count", 2),
        _single_cell(group, "unique_map_count", 2),
        _single_cell(group, "badges", 2),
        _single_cell(group, "levels_sum", 2),
        _single_cell(group, "party_size", 2),
        _single_cell(group, "deaths", 2),
        _single_cell(group, "max_map_progress", 2),
        _single_cell(group, "mt_moon", 3),
        _single_cell(group, "cerulean", 3),
        _fmt(group.get("action_entropy_bits"), 3),
        _fmt(group.get("action_top_share"), 3),
        _fmt_int(row["protocol"].get("eval_steps")),
        _fmt(row["protocol"].get("eval_seconds"), 1),
        _fmt(row["protocol"].get("steps_per_second"), 1),
        video,
    ]


def _random_actions_row(row):
    group = row["groups"]["random_actions"]
    return [
        row["label"],
        f"{group.get('n')} episodes",
        _mean_std_cell(group, "episode_reward", 4),
        _mean_std_cell(group, "length", 1),
        _mean_std_cell(group, "coord_count", 2),
        _mean_std_cell(group, "unique_map_count", 2),
        _mean_std_cell(group, "badges", 2),
        _mean_std_cell(group, "levels_sum", 2),
        _mean_std_cell(group, "party_size", 2),
        _mean_std_cell(group, "deaths", 2),
        _mean_std_cell(group, "max_map_progress", 2),
        _mean_std_cell(group, "mt_moon", 3),
        _mean_std_cell(group, "cerulean", 3),
        _fmt(group.get("action_entropy_bits"), 3),
        _fmt(group.get("action_top_share"), 3),
        _fmt_int(row["protocol"].get("eval_steps")),
        _fmt(row["protocol"].get("eval_seconds"), 1),
        _fmt(row["protocol"].get("steps_per_second"), 1),
    ]


def _build_markdown(rows, protocol_warnings, generated_utc):
    base_rows = [row for row in rows if row["protocol_variant"] == "v1"]
    ext_rows = [row for row in rows if row["protocol_variant"] == "v1-ext"]

    def variant_params(subset):
        steps = sorted({row["protocol"].get("steps_per_episode") for row in subset
                        if row["protocol"].get("steps_per_episode") is not None})
        episodes = sorted({row["protocol"].get("episodes_requested") for row in subset
                           if row["protocol"].get("episodes_requested") is not None})
        seeds = sorted({row["protocol"].get("seed") for row in subset
                        if row["protocol"].get("seed") is not None})
        return steps, episodes, seeds

    def add_stochastic_table(subset):
        rows_here = [row for row in subset if "stochastic" in row["groups"]]
        if not rows_here:
            lines.append("（无 stochastic 评估数据）")
            return
        lines.extend(_md_table(
            ["Run", "训练步数", "n", "episode reward", "length", "coord_count", "maps visited",
             "badges", "levels_sum", "party_size", "deaths", "max_map_progress",
             "mt_moon", "cerulean", "动作熵(bit)", "最大动作占比",
             "eval steps", "eval 秒数", "eval steps/s"],
            [_stochastic_row(row) for row in rows_here]))

    def add_deterministic_table(subset):
        rows_here = [row for row in subset if "deterministic" in row["groups"]]
        if not rows_here:
            lines.append("（无 deterministic 评估数据）")
            return
        lines.extend(_md_table(
            ["Run", "训练步数", "terminated_by", "length", "episode reward",
             "coord_count", "maps visited", "badges", "levels_sum", "party_size",
             "deaths", "max_map_progress", "mt_moon", "cerulean",
             "动作熵(bit)", "最大动作占比", "eval steps", "eval 秒数",
             "eval steps/s", "视频"],
            [_deterministic_row(row) for row in rows_here]))

    def add_random_actions_table(subset):
        rows_here = [row for row in subset if "random_actions" in row["groups"]]
        if not rows_here:
            lines.append("（无随机动作基线数据）")
            return
        lines.extend(_md_table(
            ["Run", "episodes", "episode reward", "length", "coord_count",
             "maps visited", "badges", "levels_sum", "party_size", "deaths",
             "max_map_progress", "mt_moon", "cerulean", "动作熵(bit)",
             "最大动作占比", "eval steps", "eval 秒数", "eval steps/s"],
            [_random_actions_row(row) for row in rows_here]))

    devices = sorted({str(row["device"]) for row in rows})
    modes = sorted({str(row["run_mode"]) for row in rows})
    base_steps, base_episodes, _base_seeds = variant_params(base_rows)
    ext_steps, ext_episodes, _ext_seeds = variant_params(ext_rows)
    lines = [
        "# 评估结果汇总（eval-protocol-v1，含 extended check v1-ext）",
        "",
        f"- 生成时间（UTC）：{generated_utc}",
        f"- 输入 runs：{len(rows)} 个 results.json（modes: {', '.join(modes)}；"
        f"variants: {', '.join(sorted({row['protocol_variant'] for row in rows}))}）",
        f"- 设备：{', '.join(devices)}；base seed："
        f"{sorted({row['protocol'].get('seed') for row in rows if row['protocol'].get('seed') is not None})}",
        f"- v1：steps/episode={base_steps}，episodes/episode 组={base_episodes}；"
        f"v1-ext：steps/episode={ext_steps}，episodes/episode 组={ext_episodes}",
        "- 数据全部来自 `scripts/eval_checkpoint.py` 的 `results.json`，未经人工修饰。",
        "",
        "## 1. Stochastic 策略评估（v1；每个 run 3 个 episode；mean ± std 来自该组 episode）",
        "",
    ]
    add_stochastic_table(base_rows)
    lines.extend([
        "",
        "## 2. Deterministic 策略评估（v1；每 run 1 个 argmax episode；轨迹唯一，方差按定义为 0）",
        "",
    ])
    add_deterministic_table(base_rows)
    lines.extend([
        "",
        "## 3. 随机动作基线（v1；均匀随机，无模型）",
        "",
    ])
    add_random_actions_table(base_rows)
    lines.extend([
        "",
        "## 4. Extended check（v1-ext：每 run 5 stochastic + 1 deterministic × 8192 步，单列统计）",
        "",
        "- v1-ext 与 v1 的 episode 预算不同（8192 vs 4096 步、5 vs 3 个 stochastic episode），"
        "reward/coords 等绝对值**不可直接跨协议比较**；该小节用于 n=5 降低方差，"
        "并在同一协议内比较 untrained / 300k / 500k / 1M。",
        "- v1-ext 的 deterministic episode 同样只作轨迹记录（argmax 退化问题见 v1 表），"
        "不用于模型选择。",
        "",
        "**Stochastic episodes（v1-ext）**",
        "",
    ])
    add_stochastic_table(ext_rows)
    lines.extend(["", "**Deterministic episode（v1-ext）**", ""])
    add_deterministic_table(ext_rows)
    lines.extend([
        "",
        "## 5. 数据来源与运行元数据",
        "",
    ])
    source_header = ["Run", "variant", "results.json", "results sha256", "checkpoint sha256",
                     "checkpoint env steps", "device", "steps/ep", "episodes",
                     "base seed", "eval steps", "eval 秒数", "created_utc"]
    source_rows = []
    for row in rows:
        checkpoint = row.get("checkpoint") or {}
        source_rows.append([
            row["label"],
            row["protocol_variant"],
            _display_path(row["source"]),
            str(row.get("source_sha256", ""))[:16],
            str(checkpoint.get("sha256", "-"))[:16],
            _fmt_int(checkpoint.get("environment_steps")),
            str(row.get("device")),
            _fmt_int(row["protocol"].get("steps_per_episode")),
            _fmt_int(row["protocol"].get("episodes_requested")),
            _fmt_int(row["protocol"].get("seed")),
            _fmt_int(row["protocol"].get("eval_steps")),
            _fmt(row["protocol"].get("eval_seconds"), 1),
            str(row["protocol"].get("created_utc")),
        ])
    lines.extend(_md_table(source_header, source_rows))

    lines.extend([
        "",
        "## 6. 视频说明",
        "",
    ])
    for note in VIDEO_NOTES:
        lines.append(f"- {note}")

    lines.extend([
        "",
        "## 7. 说明与限制",
        "",
    ])
    for note in REPORT_NOTES:
        lines.append(f"- {note}")
    if ext_rows:
        lines.append("- v1-ext 是独立扩展检查（5 stochastic + 1 deterministic × 8192 步），"
                     "其评估步数单列统计、不计入训练预算，也不并入 v1 对比表；"
                     "v1 与 v1-ext 的 episode 长度不同，reward/coords 的绝对值不可直接相加或比较。")
    lines.append("- 每个 run 的详细逐 episode / 每步指标见各自目录下的 `results.json`；本文件只做聚合。")
    all_warnings = list(protocol_warnings)
    for row in rows:
        for warning in row.get("warnings", []):
            all_warnings.append(f"{row['label']}: {warning}")
    lines.extend(["", "## 8. 异常与警告", ""])
    if all_warnings:
        for warning in all_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize eval-protocol-v1 results.json files into eval_summary.json + eval_results.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+",
                        help="results.json files and/or directories containing result directories")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT),
                        help=f"output JSON path (default: {DEFAULT_JSON_OUT})")
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT),
                        help=f"output Markdown path (default: {DEFAULT_MD_OUT})")
    parser.add_argument("--exclude", action="append", default=[],
                        help="skip discovered results whose path contains this substring (repeatable)")
    parser.add_argument("--ext-prefix", default="ext_",
                        help="directory-name prefix that marks protocol v1-ext runs (default: ext_)")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing existing output files")
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = _parse_args(argv)
    json_out = Path(args.json_out).expanduser()
    md_out = Path(args.md_out).expanduser()
    if not json_out.is_absolute():
        json_out = ROOT / json_out
    if not md_out.is_absolute():
        md_out = ROOT / md_out
    json_out = json_out.resolve()
    md_out = md_out.resolve()
    for target in (json_out, md_out):
        if target.exists() and not args.overwrite:
            _error(f"output already exists: {target}; pass --overwrite to replace it")

    paths = _discover(args.inputs, args.exclude)
    print(f"[summarize] reading {len(paths)} results.json file(s)")
    rows = _sort_rows([_load_run(path, args.ext_prefix) for path in paths])
    protocol_warnings = _protocol_warnings(rows)
    generated_utc = datetime.now(timezone.utc).isoformat()

    groups_present = sorted({group for row in rows for group in row["groups"]})
    devices = sorted({str(row["device"]) for row in rows})
    protocol_variants = {}
    for variant in sorted({row["protocol_variant"] for row in rows}):
        subset = [row for row in rows if row["protocol_variant"] == variant]
        protocol_variants[variant] = {
            "run_count": len(subset),
            "labels": [row["label"] for row in subset],
            "steps_per_episode": sorted({row["protocol"].get("steps_per_episode") for row in subset
                                         if row["protocol"].get("steps_per_episode") is not None}),
            "episodes_requested": sorted({row["protocol"].get("episodes_requested") for row in subset
                                          if row["protocol"].get("episodes_requested") is not None}),
            "eval_steps_total": sum(int(row["protocol"].get("eval_steps") or 0) for row in subset),
        }
    summary = {
        "schema": 1,
        "generated_utc": generated_utc,
        "source_protocol": SUPPORTED_PROTOCOL,
        "inputs": [str(path) for path in paths],
        "run_count": len(rows),
        "devices": devices,
        "groups_present": groups_present,
        "protocol_variants": protocol_variants,
        "eval_steps_note": "eval_steps is evaluation-only and is not part of the training budget",
        "deterministic_note": "deterministic episodes use argmax with fixed initial save state; "
                              "one deterministic episode per run, variance is 0 by construction",
        "stochastic_note": "stochastic mean/std are sample statistics over the seeded episode group "
                           "(v1: n=3; v1-ext: n=5); small samples, not for significance claims",
        "action_entropy_note": f"Shannon entropy in bits over aggregated action counts; "
                               f"uniform maximum is log2(7)={UNIFORM_ACTION_ENTROPY:.4f}",
        "report_notes": REPORT_NOTES,
        "video_notes": VIDEO_NOTES,
        "warnings": protocol_warnings,
        "rows": rows,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = _build_markdown(rows, protocol_warnings, generated_utc)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(markdown, encoding="utf-8")

    print(f"[summarize] wrote {json_out}")
    print(f"[summarize] wrote {md_out}")
    print(f"[summarize] runs: {len(rows)}; groups: {groups_present}; devices: {devices}")
    print(f"[summarize] protocol variants: "
          f"{ {variant: data['run_count'] for variant, data in protocol_variants.items()} }")
    for row in rows:
        group_counts = {group: row["groups"][group]["n"] for group in row["groups"]}
        print(f"[summarize]   {row['label']:<28} variant={row['protocol_variant']:<7} "
              f"mode={row['run_mode']:<14} "
              f"train_steps={_fmt_int(row['training_environment_steps']):>8} "
              f"groups={group_counts} warnings={len(row['warnings'])}")
    if protocol_warnings:
        print("[summarize] warnings:")
        for warning in protocol_warnings:
            print(f"[summarize]   - {warning}")


if __name__ == "__main__":
    main()
