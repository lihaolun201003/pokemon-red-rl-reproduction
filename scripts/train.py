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
    writer = SummaryWriter(str(out / "tensorboard"))
    try:
        trainer = PPOTrainer(config, torch.device("cpu"), device, 0, out.name, str(out), seed)
        update, steps = restore_checkpoint(trainer, checkpoint) if checkpoint else (0, 0)
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
                                "progress/environment_steps": steps, "progress/update": update,
                                "validation/parameter_max_delta": delta, "time/update_seconds": duration})
                for key, value in metrics.items():
                    writer.add_scalar(key, value, steps)
                if episodes:
                    writer.add_scalar("episode/reward", np.mean([e["reward"] for e in episodes]), steps)
                path = out / f"checkpoint_{steps:09d}.pt"
                if update % config["model"]["checkpoint_interval"] == 0 or steps == args.total_steps:
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
                writer.flush()
                print(f"steps={steps} update={update} {formatted} parameter_delta={delta:.3g}", flush=True)
        write_json(out / "result.json", {"real_game": True, "updates_completed": update,
            "environment_steps": steps, "checkpoint": str(path), "performance_claim": "none"})
    finally:
        if trainer is not None:
            trainer.close()
        writer.close()


if __name__ == "__main__":
    main()
