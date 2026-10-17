"""Watch the official policy through a local OpenCV game window; Q/Esc exits."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from reproduction.runtime import (ROOT, load_config, validate_inputs, select_device, seed_all,
    read_checkpoint, check_observation)
from neroRL.environments.wrapper import wrap_environment
from neroRL.nn.actor_critic import create_actor_critic_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint")
    mode.add_argument("--untrained", action="store_true")
    p.add_argument("--rom")
    p.add_argument("--state")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--speed", type=float, default=1, help="1 = 60 Game Boy frames/sec (2.5 decisions/sec)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", help="Optional MP4 path; OpenCV mp4v encoder")
    args = p.parse_args()
    if args.speed <= 0 or args.steps <= 0:
        p.error("speed and steps must be positive")
    checkpoint = read_checkpoint(args.checkpoint) if args.checkpoint else None
    config = load_config(args.config) if checkpoint is None else checkpoint["configs"]
    if checkpoint is not None and ("recurrence" in config["model"] or "transformer" in config["model"]):
        p.error("Only the feed-forward baseline has been validated by this launcher")
    validate_inputs(config, args.rom, args.state)
    seed_all(args.seed)
    device = select_device(args.device)
    label = checkpoint.get("agent_label", "Policy checkpoint - performance unverified") if checkpoint else "Untrained Agent — 仅用于运行测试"
    print(label, flush=True)
    title = "Pokemon Red | Untrained Agent - RUN TEST ONLY | Q/Esc exits" if args.untrained or checkpoint.get("environment_steps", 0) <= 256 else "Pokemon Red | Policy - performance unverified | Q/Esc exits"
    env = wrap_environment(config["environment"], 0)
    writer = None
    count, total_reward, episodes = 0, 0.0, 0
    started = time.perf_counter()
    try:
        model = create_actor_critic_model(config["model"], env.observation_space, None, env.action_space, device)
        if checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        obs, _ = env.reset()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 640, 576)
        if args.video:
            target = Path(args.video).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 2.5 * args.speed, (160, 144))
            if not writer.isOpened():
                raise RuntimeError("Video encoder failed to open")
        for _ in range(args.steps):
            tick = time.perf_counter()
            check_observation(obs, env.observation_space)
            with torch.no_grad():
                tensors = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
                policy, value, _ = model(tensors)
                action = int(policy[0].sample().item())
            obs, reward, done, info = env.step(action)
            count += 1
            total_reward += reward
            # PokeRedV2Wrapper exposes the underlying official RedGymEnv.
            pixels = env.unwrapped._env.pyboy.screen.ndarray[:, :, :3]
            frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.imshow(title, frame)
            if writer:
                writer.write(frame)
            if count % 25 == 0:
                print(f"steps={count} reward={total_reward:.4f} events={obs['events'].sum():.0f} action={action} coords={env.unwrapped._env.get_game_coords()}", flush=True)
            delay = max(1, int(1000 * (24 / 60 / args.speed - (time.perf_counter() - tick))))
            if cv2.waitKey(delay) & 0xFF in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if done:
                episodes += 1
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        env.close()
        cv2.destroyAllWindows()
        print(f"Stopped: steps={count}, reward={total_reward:.4f}, episodes={episodes}, seconds={time.perf_counter()-started:.1f}")


if __name__ == "__main__":
    main()
