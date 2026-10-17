"""Real ROM reset/step/render/state/close acceptance; fails explicitly without ROM."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from reproduction.runtime import ROOT, load_config, validate_inputs, seed_all, check_observation, write_json
from neroRL.environments.wrapper import wrap_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
    p.add_argument("--rom")
    p.add_argument("--state")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--out", default="artifacts/game_check.json")
    args = p.parse_args()
    if not 7 <= args.steps <= 256:
        p.error("Real environment acceptance is bounded to 7..256 steps")
    config = load_config(args.config)
    hashes = validate_inputs(config, args.rom, args.state)
    seed_all(0)
    env = wrap_environment(config["environment"], 0)
    try:
        obs, _ = env.reset()
        check_observation(obs, env.observation_space)
        summaries = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                         "min": float(v.min()), "max": float(v.max())} for k, v in obs.items()}
        rewards = []
        for index in range(args.steps):
            obs, reward, done, _ = env.step(index % 7)
            check_observation(obs, env.observation_space)
            assert np.isfinite(reward)
            rewards.append(float(reward))
            if done:
                obs, _ = env.reset()
        raw = env.unwrapped._env
        assert raw.render().shape == (72, 80, 1)
        assert raw.render(reduce_res=False).shape == (144, 160, 1)
        # Force the remaining budget to one action to exercise truncation without a long episode.
        raw.max_steps = raw.step_count + 1
        _, _, done, info = env.step(0)
        assert done and "reward" in info and "length" in info
        obs, _ = env.reset()
        check_observation(obs, env.observation_space)
        report = {"real_game": True, **hashes, "steps": args.steps,
                  "action_names": env.action_names, "observations": summaries,
                  "reward_sum": sum(rewards), "coords": raw.get_game_coords(),
                  "budget_truncation_and_reset": "passed"}
    finally:
        env.close()
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
