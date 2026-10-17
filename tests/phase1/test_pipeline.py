"""ROM-independent checks only. Fake RAM is never reported as a real game rollout."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch

from reproduction.runtime import (ROOT, load_config, static_spaces, seed_all, check_observation,
    save_checkpoint, read_checkpoint, restore_checkpoint, validate_inputs, write_json)
from neroRL.environments.poke_red import red_gym_env_v2 as red
from neroRL.environments.wrapper import wrap_environment
from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer


class FakePyBoy:
    """A deterministic RAM fixture with one level-6 Pokemon; no emulation."""
    def __init__(self, *args, **kwargs):
        from types import SimpleNamespace
        self.memory = bytearray(65536)
        self.memory[0xD163] = 1
        self.memory[0xD18C] = 6
        self.memory[0xD16D] = self.memory[0xD18E] = 20
        self.memory[0xD361] = self.memory[0xD362] = 10
        self.screen = SimpleNamespace(ndarray=np.zeros((144, 160, 4), dtype=np.uint8))
        self.closed = False
        self.frames = 0

    def set_emulation_speed(self, value):
        pass

    def send_input(self, value):
        pass

    def tick(self, count, render=True):
        self.frames += count
        self.screen.ndarray[:, :, :3] = self.frames % 255

    def stop(self, save=False):
        self.closed = True


class LocalWorker:
    """Synchronous test transport for exercising the full official sampler and PPO."""
    def __init__(self, config, rank, worker_id):
        self.env = wrap_environment(config, worker_id)
        self.child = self

    def send(self, message):
        command, data = message
        self.reply = self.env.reset(data) if command == "reset" else self.env.step(data)

    def recv(self):
        return self.reply

    def close(self):
        self.env.close()


@pytest.fixture
def fixture_config(monkeypatch):
    monkeypatch.setattr(red, "PyBoy", FakePyBoy)
    config = load_config(ROOT / "configs/smoke.yaml")
    config["environment"]["reset_params"]["init-state"] = None
    return config


def test_observation_action_reset_close_contract(fixture_config):
    env = wrap_environment(fixture_config["environment"], 0)
    emulator = env.unwrapped._env.pyboy
    obs, _ = env.reset()
    check_observation(obs, env.observation_space)
    expected, actions = static_spaces()
    assert {k: v.shape for k, v in expected.spaces.items()} == {k: v.shape for k, v in env.observation_space.spaces.items()}
    assert env.action_names == [["down", "left", "right", "up", "a", "b", "start"]]
    assert env.action_space.n == actions.n == 7
    for index in range(7):
        obs, reward, done, info = env.step(index)
        check_observation(obs, env.observation_space)
        assert np.isfinite(reward) and not done
    assert emulator.frames == 7 * 24
    assert not np.array_equal(obs["screens"][0], obs["screens"][1])
    raw = env.unwrapped._env
    raw.max_steps = raw.step_count + 1
    assert env.step(0)[2]
    env.reset()
    assert raw.step_count == 0
    env.close()
    env.close()
    assert emulator.closed


def test_official_reward_and_event_budget(fixture_config):
    env = wrap_environment(fixture_config["environment"], 0)
    env.reset()
    raw = env.unwrapped._env
    # One newly visited coordinate: 0.5 * 0.1 * 0.1 = 0.005.
    assert env.step(0)[1] == pytest.approx(0.005)
    assert env.step(0)[1] == pytest.approx(0)
    # One event contributes 2, extends time budget by 2048.
    previous_budget = raw.max_steps
    raw.pyboy.memory[0xD747] |= 1
    assert env.step(0)[1] == pytest.approx(2)
    assert raw.max_steps == previous_budget + 2048
    # Upstream level normalization subtracts 2 per Pokemon and 4 for the starter.
    raw.pyboy.memory[0xD18C] = 7
    assert env.step(0)[1] == pytest.approx(0.5)
    # Damage then healing: official aggregate party HP behavior, not paper equation.
    raw.pyboy.memory[0xD16D] = 10
    env.step(0)
    raw.pyboy.memory[0xD16D] = 20
    assert env.step(0)[1] == pytest.approx(1.25)
    env.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_official_ppo_update_checkpoint_resume(fixture_config, monkeypatch, tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import neroRL.sampler.trajectory_sampler as sampler
    monkeypatch.setattr(sampler, "Worker", LocalWorker)
    seed_all(42)
    trainer = PPOTrainer(fixture_config, torch.device("cpu"), torch.device(device), 0, "synthetic-unit-test", str(tmp_path), 42)
    try:
        obs = {k: torch.as_tensor(v) for k, v in trainer.sampler.current_obs.items()}
        policy, value, _ = trainer.model(obs)
        assert policy[0].probs.shape == (2, 7) and value.shape == (2,)
        assert torch.allclose(policy[0].probs.sum(1), torch.ones(2))
        before = [p.detach().cpu().clone() for p in trainer.model.parameters()]
        _, stats, _, _ = trainer.step(0)
        assert all(np.isfinite(float(v)) for _, v in stats.values())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.model.parameters())
        assert any(not torch.equal(a, b.cpu()) for a, b in zip(before, trainer.model.parameters()))
        assert trainer.sampler.buffer.actions.shape == (2, 16, 1)
        assert trainer.sampler.buffer.advantages.shape == (2, 16)
        # This artifact is explicitly synthetic, not an accepted real-game checkpoint.
        out = ROOT / "artifacts" / f"synthetic-{device}"
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out / "checkpoint.pt"
        save_checkpoint(trainer, checkpoint_path, 1, 32, {"synthetic": True})
        checkpoint = read_checkpoint(checkpoint_path)
        second = PPOTrainer(copy.deepcopy(fixture_config), torch.device("cpu"), torch.device(device), 0, "synthetic-unit-test", str(tmp_path), 42)
        try:
            assert restore_checkpoint(second, checkpoint) == (1, 32)
            for key, value in trainer.model.state_dict().items():
                assert torch.equal(value.cpu(), second.model.state_dict()[key].cpu())
            assert second.optimizer.state_dict()["param_groups"] == checkpoint["optimizer"]["param_groups"]
            for key, state in checkpoint["optimizer"]["state"].items():
                for name, value in state.items():
                    restored = second.optimizer.state_dict()["state"][key][name]
                    assert torch.equal(value.cpu(), restored.cpu()) if torch.is_tensor(value) else value == restored
            trainer.model.cpu()
            second.model.cpu()
            with torch.no_grad():
                assert torch.equal(trainer.model(obs)[0][0].probs, second.model(obs)[0][0].probs)
            before = [p.detach().cpu().clone() for p in second.model.parameters()]
            _, resumed_stats, _, _ = second.step(1)
            assert all(np.isfinite(float(v)) for _, v in resumed_stats.values())
            assert any(not torch.equal(a, b.cpu()) for a, b in zip(before, second.model.parameters()))
            bad = copy.deepcopy(checkpoint)
            bad["configs"]["trainer"]["gamma"] = 0.1
            with pytest.raises(ValueError, match="mismatch"):
                restore_checkpoint(second, bad)
            write_json(out / "result.json", {"synthetic": True, "real_game": False, "device": device,
                "model_parameters": sum(p.numel() for p in trainer.model.parameters()),
                "first_loss": float(stats["loss"][1]), "resumed_loss": float(resumed_stats["loss"][1]),
                "checkpoint_save_load_resume": "passed", "updates": 2, "fixture_steps": 64})
        finally:
            second.close()
    finally:
        trainer.close()


def test_missing_rom_fails_before_training(tmp_path):
    config = load_config(ROOT / "configs/smoke.yaml")
    with pytest.raises(FileNotFoundError, match="Missing lawful"):
        validate_inputs(config, rom=tmp_path / "missing.gb")


def test_pyboy_diagnostic_cartridge():
    from scripts.doctor import emulator_probe
    assert emulator_probe()["tick_save_load_close"] == "passed"


def test_tensorboard_roundtrip(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    writer = SummaryWriter(str(tmp_path))
    writer.add_scalar("loss/loss", 0.125, 32)
    writer.add_scalar("rollout/reward_mean", 0.005, 32)
    writer.close()
    accumulator = EventAccumulator(str(tmp_path)).Reload()
    assert accumulator.Scalars("loss/loss")[0].value == pytest.approx(0.125)


def test_paper_config_is_separate():
    paper = load_config(ROOT / "configs/paper_baseline.yaml")
    smoke = load_config(ROOT / "configs/smoke.yaml")
    assert paper["sampler"] == {"n_workers": 32, "worker_steps": 2048}
    assert paper["trainer"]["beta_schedule"]["initial"] == 0
    assert smoke["sampler"] == {"n_workers": 2, "worker_steps": 16}
    assert smoke["reproduction"]["smoke"]
