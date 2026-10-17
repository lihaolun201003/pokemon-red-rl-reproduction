"""Bounded standalone launcher; model, sampler, GAE, optimizer and PPO loss are upstream."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter

from reproduction.runtime import (ROOT, load_config, validate_inputs, select_device, seed_all,
    check_observation, metadata, read_checkpoint, save_checkpoint, restore_checkpoint, write_json)
from neroRL.environments.wrapper import wrap_environment
from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer


def summarize_game_info(infos):
    """Supervision-only: average the game-progress fields the env workers return each step.

    This function never feeds data back into PPO; it only turns the already transmitted
    info dictionaries into log scalars (exploration, story events, party, deaths, badges).
    """
    fields = ("coord_count", "max_map_progress", "levels_sum", "party_size", "deaths",
              "max_event_rew", "event_reward", "healr", "max_foe_level", "mt_moon", "cerulean")
    out = {}
    for key in fields:
        values = [float(info[key]) for info in infos if key in info]
        if values:
            out[f"game/{key}"] = sum(values) / len(values)
    badges = [float(info.get("badge_1", 0)) + float(info.get("badge_2", 0))
              for info in infos if "badge_1" in info]
    if badges:
        out["game/num_badges"] = sum(badges) / len(badges)
    return out


def json_default(value):
    """Make numpy scalars/arrays JSON serializable for episodes.jsonl."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-steps", type=int, required=True, help="Absolute cumulative target, including resumed steps; exact whole rollouts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--rom")
    parser.add_argument("--state")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stop-file", default=None,
                        help="Supervisor escape hatch: when this file exists, stop cleanly after saving the current update checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = validate_inputs(config, args.rom, args.state)  # fail before spawning workers
    checkpoint = read_checkpoint(args.resume) if args.resume else None
    seed = args.seed if args.seed is not None else (checkpoint["seed"] if checkpoint else 0)
    if checkpoint and seed != checkpoint["seed"]:
        raise ValueError("Resume seed differs from checkpoint")
    seed_all(seed, args.threads)
    device = select_device(args.device)
    batch = config["sampler"]["n_workers"] * config["sampler"]["worker_steps"]
    if args.total_steps <= 0 or args.total_steps % batch:
        raise ValueError(f"--total-steps must be a positive multiple of {batch}; no automatic scale increase")
    if batch % config["trainer"]["n_mini_batches"] or batch // config["trainer"]["n_mini_batches"] < 2:
        raise ValueError("Each PPO minibatch needs at least two samples")
    if config.get("reproduction", {}).get("smoke", False) and args.total_steps > 256:
        raise ValueError("Smoke configuration has a strict 256 cumulative environment-step ceiling")
    previous_steps = checkpoint["environment_steps"] if checkpoint else 0
    if args.total_steps <= previous_steps:
        raise ValueError("Target must exceed checkpoint environment_steps")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)  # refuse accidental run overwrite
    provenance = {**metadata(), **inputs, "seed": seed, "device": str(device),
                  "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
                  "target_steps": args.total_steps, "resume_resets_episodes": True}
    write_json(out / "environment.json", provenance)
    with (out / "config.yaml").open("w", encoding="utf-8") as stream:
        YAML().dump(copy.deepcopy(config), stream)

    # Validate a real reset and all seven actions before allocating the trainer.
    env = wrap_environment(config["environment"], 0)
    try:
        obs, _ = env.reset()
        check_observation(obs, env.observation_space)
        for action in range(7):
            obs, reward, done, _ = env.step(action)
            check_observation(obs, env.observation_space)
            if not np.isfinite(reward):
                raise FloatingPointError("Non-finite environment reward")
            if done:
                env.reset()
    finally:
        env.close()
    trainer = None
    episodes_log = None
    writer = SummaryWriter(str(out / "tensorboard"))
    try:
        trainer = PPOTrainer(config, torch.device("cpu"), device, 0, out.name, str(out), seed)
        update, steps = restore_checkpoint(trainer, checkpoint) if checkpoint else (0, 0)
        episodes_log = (out / "episodes.jsonl").open("w", encoding="utf-8")
        stopped_by_supervisor = False
        with (out / "metrics.jsonl").open("w", encoding="utf-8") as log:
            while steps < args.total_steps:
                before = {k: p.detach().cpu().clone() for k, p in trainer.model.named_parameters()}
                episodes, stats, formatted, duration = trainer.step(update)
                metrics = {f"{tag.value}/{k}": float(value) for k, (tag, value) in stats.items()}
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError("Non-finite loss or gradient statistics")
                if not all(torch.isfinite(p).all() for p in trainer.model.parameters()):
                    raise FloatingPointError("Non-finite model parameters")
                delta = max((p.detach().cpu() - before[k]).abs().max().item() for k, p in trainer.model.named_parameters())
                if delta <= 0:
                    raise AssertionError("PPO did not change any model parameters")
                steps += batch
                update += 1
                metrics.update({"rollout/reward_mean": float(trainer.sampler.buffer.rewards.mean()),
                                "rollout/reward_sum": float(trainer.sampler.buffer.rewards.sum()),
                                "progress/environment_steps": steps, "progress/update": update,
                                "validation/parameter_max_delta": delta, "time/update_seconds": duration,
                                "time/unix": time.time(),
                                "time/steps_per_second": batch / duration if duration > 0 else 0.0})
                # Supervision-only diagnostics: game progress and action distribution.
                metrics.update(summarize_game_info(trainer.sampler.latest_infos))
                actions = trainer.sampler.buffer.actions[:, :, 0].reshape(-1)
                action_counts = torch.bincount(actions, minlength=7).float()
                action_shares = (action_counts / action_counts.sum().clamp(min=1)).tolist()
                metrics.update({f"actions/share_{i}": share for i, share in enumerate(action_shares)})
                new_episodes = list(trainer.sampler.last_episode_infos)
                metrics["episode/completed_this_update"] = float(len(new_episodes))
                for episode_info in new_episodes:
                    episodes_log.write(json.dumps(episode_info, default=json_default) + "\n")
                for key, value in metrics.items():
                    writer.add_scalar(key, value, steps)
                if episodes:
                    writer.add_scalar("episode/reward", np.mean([e["reward"] for e in episodes]), steps)
                path = out / f"checkpoint_{steps:09d}.pt"
                stop_requested = bool(args.stop_file) and Path(args.stop_file).exists()
                if (update % config["model"]["checkpoint_interval"] == 0 or
                        steps == args.total_steps or stop_requested):
                    save_checkpoint(trainer, path, update, steps, provenance)
                    writer.add_text("checkpoint/path", str(path), steps)
                    metrics["checkpoint"] = str(path)
                    loaded = read_checkpoint(path)
                    if loaded["environment_steps"] != steps or not loaded["optimizer"]["state"]:
                        raise AssertionError("Saved checkpoint is incomplete")
                    for key, value in trainer.model.state_dict().items():
                        if not torch.equal(value.cpu(), loaded["model"][key].cpu()):
                            raise AssertionError("Checkpoint model mismatch")
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                episodes_log.flush()
                writer.flush()
                print(f"steps={steps} update={update} {formatted} parameter_delta={delta:.3g}", flush=True)
                if stop_requested:
                    stopped_by_supervisor = True
                    print(f"Supervised stop requested via {args.stop_file} at {steps} steps", flush=True)
                    break
        write_json(out / "result.json", {"real_game": True, "updates_completed": update,
            "environment_steps": steps, "checkpoint": str(path),
            "stop_reason": "stop_file" if stopped_by_supervisor else "step_target_reached",
            "performance_claim": "none"})
    finally:
        if episodes_log is not None:
            episodes_log.close()
        if trainer is not None:
            trainer.close()
        writer.close()


if __name__ == "__main__":
    main()
