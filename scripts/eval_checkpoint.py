"""Standalone fixed-protocol evaluation for Pokemon Red PPO checkpoints (protocol v1).

Evaluation only: this script never updates model weights, never touches the optimizer,
PPO hyper-parameters, the reward function or the environment configuration, and never
writes inside training run directories. It replays the deployment loop already used by
scripts/watch.py:

    checkpoint -> config snapshot -> PokeRedV2 wrapper -> feed-forward actor-critic -> action

Protocol v1 (see docs/reports/eval_protocol.md):
    * --checkpoint uses the configuration stored inside the checkpoint
      (checkpoint["configs"]) so environment, rewards and model are identical to
      training. --untrained and --random-actions use --config instead.
    * every episode resets the official initial save state (no mid-episode resume).
    * default protocol: N stochastic episodes (categorical sampling, torch seeded per
      episode with base seed + i) plus 1 deterministic episode (policy probs argmax).
    * optional --video records the deterministic episode by default (or the first
      stochastic episode with --video-mode stochastic) as a 160x144 mp4 at
      ~2.5 decisions/second and stores 25% / 50% / 100% screenshots of that episode.
      The stochastic mode is intended for human-review demo videos, because the
      argmax policy can degenerate into repeating a single button.
    * all reported numbers are measured as-is; nothing is smoothed or hand-edited.

Usage examples (PowerShell, from the repository root):
    .venv\\Scripts\\python.exe scripts/eval_checkpoint.py \\
        --checkpoint artifacts/smoke-gpu-resumed/checkpoint_000000064.pt \\
        --steps 64 --episodes 1 --device cuda --out artifacts/training/eval_smoke/smoke_ckpt

    .venv\\Scripts\\python.exe scripts/eval_checkpoint.py \\
        --untrained --config configs/train_seed0.yaml \\
        --steps 128 --episodes 1 --out artifacts/training/eval_smoke/untrained_smoke

    .venv\\Scripts\\python.exe scripts/eval_checkpoint.py \\
        --random-actions --config configs/train_seed0.yaml \\
        --steps 256 --episodes 3 --out artifacts/training/eval_smoke/random_smoke

Environment variables are not used; all paths come from the command line, the config
file or the checkpoint config snapshot.
"""

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from gymnasium import spaces
from ruamel.yaml import YAML

from neroRL.environments.wrapper import wrap_environment
from neroRL.nn.actor_critic import create_actor_critic_model
from reproduction.runtime import (
    check_observation,
    load_config,
    metadata,
    read_checkpoint,
    seed_all,
    select_device,
    validate_inputs,
    write_json,
)

PROTOCOL_NAME = "eval-protocol-v1"
SCHEMA_VERSION = 1
VIDEO_FPS = 2.5
VIDEO_SIZE = (160, 144)  # (width, height) expected by cv2.VideoWriter
DEFAULT_CONFIG = str(ROOT / "configs/train_seed0.yaml")

# Official action order from neroRL/environments/poke_red/red_gym_env_v2.py.
ACTION_NAMES = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]

# Per-step info fields recorded in every episode trace. These are the supervision
# fields the environment returns each step (see red_gym_env_v2.py info dict).
INFO_TRACE_FIELDS = (
    "deaths",
    "max_foe_level",
    "max_event_rew",
    "party_size",
    "levels_sum",
    "mt_moon",
    "cerulean",
    "event_reward",
    "healr",
    "coord_count",
    "max_map_progress",
    "badge_1",
    "badge_2",
)

# Episode-level fields aggregated in summary (mean/std/min/max per group).
SUMMARY_FIELDS = (
    "episode_reward",
    "length",
    "coord_count",
    "unique_map_count",
    "badges",
    "badge_1",
    "badge_2",
    "levels_sum",
    "party_size",
    "deaths",
    "max_map_progress",
    "mt_moon",
    "cerulean",
    "max_foe_level",
    "max_event_rew",
    "event_reward",
    "healr",
    "obs_events_sum",
    "steps_per_second",
)

LIMITATIONS = [
    "PokeRedV2 truncates only on the episode step budget; there is no explicit win "
    "condition. Episodes here end either at --steps ('step_cap') or at the "
    "environment's own truncation ('env_truncation'); both are recorded separately.",
    "The official initial save state fixes the Game Boy RNG, so a deterministic "
    "(argmax) policy always replays the same trajectory within one environment "
    "instance. Protocol v1 constructs a fresh environment for every episode and "
    "verified that deterministic episodes then produce identical traces; "
    "deterministic-episode variance is therefore zero by construction and one "
    "deterministic episode is run per invocation.",
    "A fresh PyBoy environment is created for every episode: load_state restores "
    "emulator RAM, but a reused instance keeps rendered screen-buffer history, which "
    "makes later observations slightly different even with identical RAM. Fresh "
    "instances cost ~1-2 s per episode and keep episodes independent of each other.",
    "Stochastic-policy variance comes from per-episode torch RNG seeding "
    "(base seed + episode index), i.e. it measures policy stochasticity, not "
    "environment stochasticity.",
    "Badges / story metrics are read from the official env info dict; no additional "
    "battle or story-progress validation is performed.",
    "Checkpoint configs are used verbatim. This script never changes PPO, model, "
    "reward or environment settings, and it never resumes training.",
    "Evaluation environment steps are counted separately from the training budget "
    "and do not advance any training counter.",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _error(message):
    """Abort with a clear, prefixed error message (exit code 1)."""
    raise SystemExit(f"[eval] ERROR: {message}")


def _plain(value):
    """Convert numpy scalars to plain JSON values; NaN becomes None."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _safe(text):
    """ASCII-safe console text (the Windows console is not always UTF-8)."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _relative(path, base):
    """Path relative to an output directory when possible, else absolute string."""
    path = Path(path).resolve()
    base = Path(base).resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path, chunk_size=1 << 20):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_environment_metadata():
    """Host/software metadata recorded in results.json (best effort, never fatal)."""
    data = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        data.update(metadata())
    except Exception as exc:  # psutil/git missing must not block an evaluation
        data.update({
            "python": platform.python_version(),
            "torch": torch.__version__,
            "metadata_error": f"{type(exc).__name__}: {exc}",
        })
    data["cv2"] = cv2.__version__
    data["numpy"] = np.__version__
    return data


def _check_supported_config(config):
    """Fail early on anything protocol v1 does not support."""
    if not isinstance(config, dict):
        _error("configuration is not a dictionary")
    environment = config.get("environment", {})
    if environment.get("type") != "PokeRedV2":
        _error(f"protocol v1 supports only the PokeRedV2 environment, got {environment.get('type')!r}")
    model_config = config.get("model", {})
    if "recurrence" in model_config or "transformer" in model_config:
        _error("protocol v1 supports feed-forward models only (recurrence/transformer found in config)")
    if "reset_params" not in environment:
        _error("environment config is missing reset_params")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fixed-protocol evaluation of Pokemon Red PPO checkpoints (protocol v1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", help="path to a training checkpoint (.pt); its config snapshot is authoritative")
    mode.add_argument("--untrained", action="store_true", help="build a randomly initialized model from --config, no weights loaded")
    mode.add_argument("--random-actions", action="store_true", help="uniform random actions, no model needed")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"config used by --untrained / --random-actions (default: {DEFAULT_CONFIG})")
    parser.add_argument("--out", required=True, help="output directory for results.json, config_snapshot.yaml, screenshots")
    parser.add_argument("--episodes", type=int, default=3,
                        help="number of stochastic episodes (or deterministic ones with --no-stochastic; default: 3)")
    parser.add_argument("--steps", type=int, default=2048, help="maximum steps per episode (default: 2048)")
    parser.add_argument("--seed", type=int, default=100000, help="base seed; episode i uses base seed + i (default: 100000)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="inference device (default: auto)")
    parser.add_argument("--video", help="optional mp4 path for the episode selected by --video-mode, plus its screenshots")
    parser.add_argument("--video-mode", choices=["deterministic", "stochastic"], default="deterministic",
                        help="which episode --video records: deterministic argmax episode (default) or "
                             "the first stochastic episode; model modes only "
                             "(--random-actions always records its final episode)")
    parser.add_argument("--rom", help="override ROM path from the config")
    parser.add_argument("--state", help="override initial save-state path from the config")
    parser.add_argument("--no-stochastic", action="store_true",
                        help="run deterministic (argmax) episodes only; --episodes controls how many")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing results.json / video file in the output paths")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.no_stochastic and args.random_actions:
        parser.error("--no-stochastic is meaningless with --random-actions (there is no policy)")
    if args.video_mode == "stochastic" and args.no_stochastic:
        parser.error("--video-mode stochastic requires at least one stochastic episode "
                     "(remove --no-stochastic)")
    return args


# ---------------------------------------------------------------------------
# Environment / model setup
# ---------------------------------------------------------------------------
def _create_environment(config):
    try:
        env = wrap_environment(config["environment"], 0)
    except Exception as exc:
        _error(f"could not instantiate the PokeRedV2 environment: {type(exc).__name__}: {exc}")
    if not isinstance(env.action_space, spaces.Discrete):
        env.close()
        _error(f"protocol v1 expects a Discrete action space, got {type(env.action_space).__name__}")
    return env


def _create_model(config, env, device):
    try:
        model = create_actor_critic_model(config["model"], env.observation_space, None, env.action_space, device)
    except Exception as exc:
        _error(f"could not build the actor-critic model from the config: {type(exc).__name__}: {exc}")
    model.eval()
    return model


def _load_checkpoint_weights(model, checkpoint, checkpoint_path):
    """Strict state-dict load plus a finite-parameter audit."""
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict) or not state_dict:
        _error(f"checkpoint {checkpoint_path} does not contain a model state dict")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        _error(f"checkpoint {checkpoint_path} does not match the model built from its config "
               f"(strict=True): {exc}")
    non_finite = [name for name, parameter in model.named_parameters()
                  if not torch.isfinite(parameter).all()]
    if non_finite:
        _error(f"checkpoint {checkpoint_path} contains non-finite parameters: {non_finite[:5]}")


def _model_parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


# ---------------------------------------------------------------------------
# Video / screenshots
# ---------------------------------------------------------------------------
def _open_video_writer(path, overwrite):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        _error(f"video path already exists: {path}; pass --overwrite to replace it")
    try:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE)
    except Exception as exc:
        _error(f"could not open the mp4 encoder ({path}): {type(exc).__name__}: {exc}")
    if not writer.isOpened():
        _error(f"mp4 encoder failed to open for {path}")
    return writer


def _extract_screenshots(video_path, frames_written, out_dir, prefix):
    """Store 25% / 50% / 100% frames of the recorded episode as PNG files."""
    if frames_written <= 0:
        _error("cannot extract screenshots from an empty video")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        _error(f"could not reopen the recorded video: {video_path}")
    last_index = frames_written - 1
    targets = {}
    for label, fraction in (("25pct", 0.25), ("50pct", 0.5), ("100pct", 1.0)):
        index = min(last_index, max(0, int(round(fraction * last_index))))
        targets.setdefault(index, []).append((label, fraction))
    saved = {}
    wanted = set(targets)
    index = 0
    try:
        while wanted:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            if index in wanted:
                for label, fraction in targets[index]:
                    screenshot_path = Path(out_dir) / f"{prefix}_{label}.png"
                    if not cv2.imwrite(str(screenshot_path), frame):
                        _error(f"could not write screenshot: {screenshot_path}")
                    saved[label] = {
                        "path": _relative(screenshot_path, out_dir),
                        "frame_index": index,
                        "requested_fraction": fraction,
                        "actual_fraction": round(index / last_index, 6) if last_index else 0.0,
                    }
                wanted.discard(index)
            index += 1
    finally:
        capture.release()
    if wanted:
        _error(f"could not read screenshots at frames {sorted(wanted)} from {video_path}")
    return saved


# ---------------------------------------------------------------------------
# Episode roll-out
# ---------------------------------------------------------------------------
def _select_action(model, obs, action_mode, device, n_actions):
    if action_mode == "random_actions":
        return random.randrange(n_actions)
    with torch.no_grad():
        tensors = {key: torch.as_tensor(value, device=device).unsqueeze(0) for key, value in obs.items()}
        policy, _value, _memory = model(tensors)
    branch = policy[0]
    if action_mode == "deterministic":
        return int(branch.probs.argmax(dim=-1).item())
    return int(branch.sample().item())


def _run_episode(env, model, action_mode, episode_seed, max_steps, device, video_writer):
    """Run one fixed-protocol episode and return its measured record."""
    # Re-seed every episode so action sampling (torch) and the env reset's Python RNG
    # consumption are reproducible and independent of previous episodes.
    seed_all(episode_seed)
    reset_started = time.perf_counter()
    obs, _reset_info = env.reset()
    reset_seconds = time.perf_counter() - reset_started
    check_observation(obs, env.observation_space)
    try:
        env_seed = int(env.seed)
    except (AttributeError, TypeError):
        env_seed = None

    trace_info = {field: [] for field in INFO_TRACE_FIELDS}
    trace_actions, trace_rewards, trace_coords, trace_obs_events = [], [], [], []
    inference_seconds = 0.0
    stepping_seconds = 0.0
    frame_count = 0
    final_info = {}
    terminated_by = "step_cap"

    loop_started = time.perf_counter()
    for _ in range(max_steps):
        inference_started = time.perf_counter()
        action = _select_action(model, obs, action_mode, device, int(env.action_space.n))
        inference_seconds += time.perf_counter() - inference_started

        step_started = time.perf_counter()
        obs, reward, done, info = env.step(action)
        stepping_seconds += time.perf_counter() - step_started
        check_observation(obs, env.observation_space)

        x_pos, y_pos, map_n = env.unwrapped._env.get_game_coords()
        trace_actions.append(int(action))
        trace_rewards.append(float(reward))
        trace_coords.append([int(x_pos), int(y_pos), int(map_n)])
        trace_obs_events.append(int(obs["events"].sum()))
        for field in INFO_TRACE_FIELDS:
            trace_info[field].append(_plain(info.get(field)))
        final_info = info

        if video_writer is not None:
            pixels = env.unwrapped._env.pyboy.screen.ndarray[:, :, :3]  # (144, 160, 3) RGB
            frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
            frame_count += 1

        if done:
            terminated_by = "env_truncation"
            break
    loop_seconds = time.perf_counter() - loop_started

    length = len(trace_actions)
    if length == 0:
        _error("episode executed zero steps; refusing to report an empty record")
    episode_reward = float(np.sum(trace_rewards))
    reported_reward = reported_length = None
    if terminated_by == "env_truncation":
        reported_reward = _plain(final_info.get("reward"))
        reported_length = _plain(final_info.get("length"))
    reward_consistency = None
    if reported_reward is not None:
        reward_consistency = bool(abs(float(reported_reward) - episode_reward) <= 1e-6)

    maps = [coord[2] for coord in trace_coords]
    first_steps = {}
    for step_index, map_id in enumerate(maps, start=1):
        if map_id not in first_steps:
            first_steps[map_id] = step_index
    action_counts = [int(np.sum(np.asarray(trace_actions) == action)) for action in range(7)]
    action_shares = [round(count / length, 6) for count in action_counts]

    final_badge_1 = int(_plain(final_info.get("badge_1")) or 0)
    final_badge_2 = int(_plain(final_info.get("badge_2")) or 0)
    record = {
        "index": None,  # filled by the caller
        "mode": action_mode,
        "seed": int(episode_seed),
        "env_seed": env_seed,
        "steps_requested": int(max_steps),
        "length": length,
        "terminated_by": terminated_by,
        "episode_reward": episode_reward,
        "env_reported_reward": reported_reward,
        "env_reported_length": reported_length,
        "reward_consistency": reward_consistency,
        "reset_seconds": round(reset_seconds, 6),
        "stepping_seconds": round(stepping_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "loop_seconds": round(loop_seconds, 6),
        "steps_per_second": round(length / loop_seconds, 4) if loop_seconds > 0 else None,
        "coord_count": _plain(final_info.get("coord_count")),
        "unique_map_count": len(set(maps)),
        "maps_first_step": {str(map_id): step for map_id, step in sorted(first_steps.items())},
        "final_coords": trace_coords[-1],
        "badges": final_badge_1 + final_badge_2,
        "badge_1": final_badge_1,
        "badge_2": final_badge_2,
        "badge_1_steps": _plain(final_info.get("badge_1_steps")),
        "badge_2_steps": _plain(final_info.get("badge_2_steps")),
        "levels_sum": _plain(final_info.get("levels_sum")),
        "party_size": _plain(final_info.get("party_size")),
        "deaths": _plain(final_info.get("deaths")),
        "max_map_progress": _plain(final_info.get("max_map_progress")),
        "mt_moon": _plain(final_info.get("mt_moon")),
        "cerulean": _plain(final_info.get("cerulean")),
        "max_foe_level": _plain(final_info.get("max_foe_level")),
        "max_event_rew": _plain(final_info.get("max_event_rew")),
        "event_reward": _plain(final_info.get("event_reward")),
        "healr": _plain(final_info.get("healr")),
        "obs_events_sum": trace_obs_events[-1],
        "action_counts": action_counts,
        "action_shares": action_shares,
        "video": None,
        "screenshots": None,
        "video_frame_count": frame_count,
        "trace": {
            "actions": trace_actions,
            "rewards": [round(value, 6) for value in trace_rewards],
            "coords": trace_coords,
            "obs_events_sum": trace_obs_events,
            "info": trace_info,
        },
    }
    return record


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------
def _summarize(records):
    """Mean/std/min/max per episode group; groups are never mixed."""
    summary = {}
    for group in ("stochastic", "deterministic", "random_actions"):
        rows = [record for record in records if record["mode"] == group]
        if not rows:
            continue
        group_summary = {"n": len(rows), "mean": {}, "std": {}, "min": {}, "max": {}}
        for field in SUMMARY_FIELDS:
            values = [float(record[field]) for record in rows if record.get(field) is not None]
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            group_summary["mean"][field] = round(float(array.mean()), 6)
            group_summary["std"][field] = round(float(array.std()), 6)
            group_summary["min"][field] = round(float(array.min()), 6)
            group_summary["max"][field] = round(float(array.max()), 6)
        group_summary["episode_reward_sum"] = round(sum(record["episode_reward"] for record in rows), 6)
        group_summary["action_counts_total"] = [
            int(sum(record["action_counts"][action] for record in rows)) for action in range(7)
        ]
        total_actions = sum(group_summary["action_counts_total"]) or 1
        group_summary["action_shares_total"] = [
            round(count / total_actions, 6) for count in group_summary["action_counts_total"]
        ]
        summary[group] = group_summary
    return summary


def _print_episode(record, total):
    print(
        f"[eval] episode {record['index'] + 1}/{total} mode={record['mode']} seed={record['seed']} "
        f"env_seed={record['env_seed']} length={record['length']} "
        f"terminated_by={record['terminated_by']} reward={record['episode_reward']:.4f} "
        f"coords={record['coord_count']} maps={record['unique_map_count']} "
        f"badges={record['badges']} levels={record['levels_sum']} "
        f"party={record['party_size']} deaths={record['deaths']} "
        f"progress={record['max_map_progress']} steps/s={record['steps_per_second']:.2f}",
        flush=True,
    )


def _print_summary(protocol, checkpoint, summary, records, totals):
    print()
    print("=" * 100)
    print("Evaluation protocol v1 summary")
    print("=" * 100)
    print(f"mode: {protocol['mode']}  device: {protocol['device']}  "
          f"steps/episode: {protocol['steps_per_episode']}  base seed: {protocol['seed']}")
    print(f"config source: {protocol['config_source']}")
    if checkpoint is not None:
        print(f"checkpoint: {checkpoint['path']}")
        print(f"checkpoint environment_steps: {checkpoint['environment_steps']}  "
              f"next_update: {checkpoint['next_update']}  phase1_format: {checkpoint['phase1_format']}")
        print(f"agent_label: {_safe(checkpoint['agent_label'])}")
    else:
        print(f"model: {'random actions (no model)' if protocol['mode'] == 'random_actions' else 'randomly initialized (untrained)'}")
    print(f"eval environment steps: {totals['eval_steps']} (separate from training budget)  "
          f"wall: {totals['protocol_wall_seconds']:.2f}s  "
          f"throughput: {totals['steps_per_second']:.2f} steps/s")

    print()
    header = (f"{'group':<14}{'n':>3}  {'reward mean+/-std':>22}  {'length mean+/-std':>20}  "
              f"{'coords':>7}  {'maps':>5}  {'badges':>6}  {'levels':>7}  "
              f"{'deaths':>6}  {'progress':>8}  {'steps/s':>8}")
    print(header)
    print("-" * len(header))
    for group in ("stochastic", "deterministic", "random_actions"):
        if group not in summary:
            continue
        group_summary = summary[group]

        def cell(field, fmt):
            mean = group_summary["mean"].get(field)
            std = group_summary["std"].get(field)
            if mean is None:
                return "-"
            if std is None:
                return fmt.format(mean)
            return (fmt + "+/-" + fmt).format(mean, std)

        print(f"{group:<14}{group_summary['n']:>3}  {cell('episode_reward', '{:.2f}'):>22}  "
              f"{cell('length', '{:.1f}'):>20}  {cell('coord_count', '{:.1f}'):>7}  "
              f"{cell('unique_map_count', '{:.1f}'):>5}  {cell('badges', '{:.1f}'):>6}  "
              f"{cell('levels_sum', '{:.1f}'):>7}  {cell('deaths', '{:.2f}'):>6}  "
              f"{cell('max_map_progress', '{:.1f}'):>8}  {cell('steps_per_second', '{:.1f}'):>8}")
    print()
    print("episode records:")
    for record in records:
        print(f"  #{record['index']} {record['mode']:<14} seed={record['seed']} "
              f"env_seed={record['env_seed']} length={record['length']} "
              f"reward={record['episode_reward']:.4f} coords={record['coord_count']} "
              f"maps={record['unique_map_count']} badges={record['badges']} "
              f"levels={record['levels_sum']} party={record['party_size']} "
              f"deaths={record['deaths']} progress={record['max_map_progress']} "
              f"mt_moon={record['mt_moon']} cerulean={record['cerulean']}")
        if record.get("video"):
            print(f"      video: {record['video']['path']}  screenshots: "
                  f"{', '.join(item['path'] for item in record['screenshots'].values())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = _parse_args(argv)
    process_started = time.perf_counter()

    # Resolve every user path against the invocation directory before any chdir()
    # inside load_config()/the environment wrapper changes the working directory.
    invocation_cwd = Path.cwd()
    out_dir = Path(args.out).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve() if args.video else None
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    rom_path = str(Path(args.rom).expanduser().resolve()) if args.rom else None
    state_path = str(Path(args.state).expanduser().resolve()) if args.state else None

    if checkpoint_path is not None:
        mode = "checkpoint"
    elif args.untrained:
        mode = "untrained"
    else:
        mode = "random_actions"

    # -- configuration ------------------------------------------------------
    checkpoint = None
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            _error(f"checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = read_checkpoint(checkpoint_path)
        except Exception as exc:
            _error(f"could not read checkpoint {checkpoint_path}: {type(exc).__name__}: {exc}")
        if not isinstance(checkpoint, dict):
            _error(f"checkpoint {checkpoint_path} did not contain a dictionary")
        config = checkpoint.get("configs")
        if not isinstance(config, dict) or not config:
            _error(f"checkpoint {checkpoint_path} does not contain a config snapshot ('configs')")
        config_source = f"checkpoint:{checkpoint_path}"
        if any(item == "--config" or item.startswith("--config=") for item in sys.argv[1:]):
            print("[eval] note: --config is ignored with --checkpoint; the checkpoint config snapshot is authoritative")
        print(f"[eval] loaded checkpoint: {checkpoint_path}")
    else:
        if not config_path.is_file():
            _error(f"config not found: {config_path}")
        try:
            config = load_config(config_path)
        except Exception as exc:
            _error(f"could not load config {config_path}: {type(exc).__name__}: {exc}")
        config_source = str(config_path)

    _check_supported_config(config)

    try:
        inputs = validate_inputs(config, rom_path, state_path)
    except Exception as exc:
        _error(f"input validation failed (ROM / initial state): {type(exc).__name__}: {exc}")
    try:
        device = select_device(args.device)
    except Exception as exc:
        _error(str(exc))

    # -- output directory ---------------------------------------------------
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _error(f"could not create output directory {out_dir}: {exc}")
    results_path = out_dir / "results.json"
    if results_path.exists() and not args.overwrite:
        _error(f"{results_path} already exists; pass --overwrite to replace this evaluation output")
    if video_path is not None and video_path.exists() and not args.overwrite:
        _error(f"video path already exists: {video_path}; pass --overwrite to replace it")
    try:
        with (out_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as stream:
            YAML().dump(config, stream)
    except OSError as exc:
        _error(f"could not write config snapshot: {exc}")
    print(f"[eval] config source: {config_source}")
    print(f"[eval] config snapshot: {out_dir / 'config_snapshot.yaml'}")

    # Keep the wrapper's scratch './session' directory inside the evaluation output
    # directory instead of polluting the repository root.
    os.chdir(out_dir)

    # -- environment / model ------------------------------------------------
    # A dedicated probe environment supplies observation/action spaces for model
    # construction. Every evaluated episode then gets its own freshly constructed
    # environment: reusing one PyBoy instance across episodes leaks rendered frame
    # history into later observations (load_state restores emulator RAM, but the
    # screen buffer/render history is not fully restored), which would make a
    # deterministic episode depend on the episodes that ran before it and break the
    # fixed-protocol property that a deterministic policy replays one trajectory.
    model = None
    model_parameters = 0
    probe_env = None
    try:
        if mode != "random_actions":
            probe_env = _create_environment(config)
            print(f"[eval] probe environment ready; action_space={probe_env.action_space}; "
                  f"observation keys={sorted(probe_env.observation_space.spaces.keys())}")
            seed_all(args.seed)  # reproducible random init for --untrained
            model = _create_model(config, probe_env, device)
            model_parameters = _model_parameter_count(model)
            if checkpoint is not None:
                _load_checkpoint_weights(model, checkpoint, checkpoint_path)
                print(f"[eval] model weights loaded and finite-checked ({model_parameters} parameters)")
            else:
                non_finite = [name for name, parameter in model.named_parameters()
                              if not torch.isfinite(parameter).all()]
                if non_finite:
                    _error(f"randomly initialized model has non-finite parameters: {non_finite[:5]}")
                print(f"[eval] randomly initialized model ({model_parameters} parameters, no weights loaded)")
    finally:
        if probe_env is not None:
            probe_env.close()

    # -- episode plan -------------------------------------------------------
    plan = []
    if mode == "random_actions":
        plan = [{"index": i, "mode": "random_actions", "seed": args.seed + i}
                for i in range(args.episodes)]
    elif args.no_stochastic:
        plan = [{"index": i, "mode": "deterministic", "seed": args.seed + i}
                for i in range(args.episodes)]
    else:
        plan = [{"index": i, "mode": "stochastic", "seed": args.seed + i}
                for i in range(args.episodes)]
        plan.append({"index": len(plan), "mode": "deterministic", "seed": args.seed + args.episodes})
    if mode == "random_actions":
        video_index = len(plan) - 1  # random-action mode always records its final episode
    elif args.video_mode == "deterministic":
        video_index = len(plan) - 1  # deterministic episode is last by protocol construction
    else:  # --video-mode stochastic: first stochastic episode (human-review demo video)
        video_index = next(index for index, entry in enumerate(plan)
                           if entry["mode"] == "stochastic")

    records = []
    video_meta = None
    protocol_started = time.perf_counter()
    for entry in plan:
        env = None
        writer = None
        recording = bool(video_path is not None and entry["index"] == video_index)
        try:
            env = _create_environment(config)
            if recording:
                writer = _open_video_writer(video_path, args.overwrite)
                print(f"[eval] recording episode {entry['index']} ({entry['mode']}) to {video_path}")
            record = _run_episode(env, model, entry["mode"], entry["seed"], args.steps,
                                  device, writer)
        finally:
            if writer is not None:
                writer.release()
            if env is not None:
                env.close()
        record["index"] = entry["index"]
        if recording:
            if record["video_frame_count"] <= 0:
                _error("video recording produced no frames")
            prefix = record["mode"]  # deterministic / stochastic / random_actions
            screenshots = _extract_screenshots(video_path, record["video_frame_count"], out_dir, prefix)
            video_meta = {
                "path": _relative(video_path, out_dir),
                "fps": VIDEO_FPS,
                "width": VIDEO_SIZE[0],
                "height": VIDEO_SIZE[1],
                "frames": record["video_frame_count"],
                "episode_index": entry["index"],
                "episode_mode": record["mode"],
            }
            record["video"] = video_meta
            record["screenshots"] = screenshots
        else:
            record.pop("video_frame_count", None)
        records.append(record)
        _print_episode(record, len(plan))
    protocol_wall_seconds = time.perf_counter() - protocol_started

    eval_steps = int(sum(record["length"] for record in records))
    totals = {
        "episodes": len(records),
        "eval_steps": eval_steps,
        "protocol_wall_seconds": round(protocol_wall_seconds, 6),
        "steps_per_second": round(eval_steps / protocol_wall_seconds, 4) if protocol_wall_seconds > 0 else None,
        "note": "eval_steps counts only inference steps run by this script; it is not training budget",
    }

    # -- checkpoint / model metadata ---------------------------------------
    checkpoint_metadata = None
    if checkpoint is not None:
        env_steps = checkpoint.get("environment_steps")
        checkpoint_metadata = {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "size_bytes": int(checkpoint_path.stat().st_size),
            "environment_steps": int(env_steps) if isinstance(env_steps, (int, np.integer)) else env_steps,
            "agent_label": checkpoint.get("agent_label"),
            "phase1_format": checkpoint.get("phase1_format"),
            "next_update": checkpoint.get("next_update"),
            "update": checkpoint.get("update"),
            "seed": checkpoint.get("seed"),
            "run_id": checkpoint.get("run_id"),
            "model_parameters": model_parameters,
        }
    model_metadata = {
        "source": mode,
        "parameters": model_parameters,
        "finite": True if model is not None else None,
        "device": str(device),
        "loaded_from_checkpoint": checkpoint is not None,
    }

    seeds = {
        "stochastic": [entry["seed"] for entry in plan if entry["mode"] == "stochastic"],
        "deterministic": [entry["seed"] for entry in plan if entry["mode"] == "deterministic"],
        "random_actions": [entry["seed"] for entry in plan if entry["mode"] == "random_actions"],
    }
    reset_params = config["environment"]["reset_params"]
    reward_params = {key: reset_params.get(key) for key in
                     ("reward_scale", "event_weight", "level_weight", "heal_weight",
                      "op_lvl_weight", "explore_weight")}

    protocol = {
        "name": PROTOCOL_NAME,
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "config_source": config_source,
        "checkpoint_config_used_verbatim": checkpoint is not None,
        "episodes_requested": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "seed": int(args.seed),
        "device": str(device),
        "seeds": seeds,
        "stochastic_evaluation": bool(seeds["stochastic"]),
        "no_stochastic": bool(args.no_stochastic),
        "action_selection": {
            "deterministic": "argmax over policy probabilities",
            "stochastic": "sample from categorical policy distribution (torch seeded base seed + episode index)",
            "random_actions": "uniform random action from Python random (seeded base seed + episode index)",
        },
        "video_requested": str(video_path) if video_path is not None else None,
        "video_mode": args.video_mode,
        "video": video_meta,
        "episode_order": [{"index": entry["index"], "mode": entry["mode"], "seed": entry["seed"]} for entry in plan],
        "fresh_environment_per_episode": True,
        "environment_instances": len(plan) + (1 if mode != "random_actions" else 0),
        "totals": totals,
        "reward_params": reward_params,
        "rom_sha1": inputs.get("rom_sha1"),
        "state_sha256": inputs.get("state_sha256"),
        "command": sys.argv,
        "cwd": str(invocation_cwd),
        "environment": _collect_environment_metadata(),
        "process_wall_seconds": round(time.perf_counter() - process_started, 6),
        "limitations": LIMITATIONS,
    }
    results = {
        "protocol": protocol,
        "model": model_metadata,
        "checkpoint": checkpoint_metadata,
        "episodes": records,
        "summary": _summarize(records),
    }
    write_json(results_path, results)
    print(f"[eval] results written to {results_path}")
    _print_summary(protocol, checkpoint_metadata, results["summary"], records, totals)


if __name__ == "__main__":
    main()
