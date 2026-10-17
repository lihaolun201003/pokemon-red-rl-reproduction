"""Read-only compatibility probe for an explicitly supplied, potentially modified ROM.

This diagnostic can test an unknown hash, but never approves it for training or
changes game RAM, rewards, memory addresses, ROM bytes or save-state bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from reproduction.runtime import ROOT, ROM_SHA1, load_config, check_observation, write_json
from neroRL.environments.wrapper import wrap_environment


def snapshot(emulator):
    memory = emulator.memory
    word = lambda a: 256 * memory[a] + memory[a + 1]
    return {"x_D362": memory[0xD362], "y_D361": memory[0xD361], "map_D35E": memory[0xD35E],
            "party_size_D163": memory[0xD163], "party_species_D164": memory[0xD164],
            "hp_D16C": word(0xD16C), "max_hp_D18D": word(0xD18D),
            "level_D18C": memory[0xD18C], "badges_D356": memory[0xD356],
            "event_bytes_nonzero": sum(memory[a] != 0 for a in range(0xD747, 0xD887))}


def image(emulator, path):
    cv2.imwrite(str(path), cv2.cvtColor(emulator.screen.ndarray[:, :, :3], cv2.COLOR_RGB2BGR))


def validate_snapshot(value):
    """Semantic checks: numeric observation bounds alone cannot detect broken RAM."""
    failures = []
    if not 1 <= value["party_size_D163"] <= 6:
        failures.append(f"party size {value['party_size_D163']} outside 1..6")
    if not 1 <= value["level_D18C"] <= 100:
        failures.append(f"level {value['level_D18C']} outside 1..100")
    if not 0 <= value["hp_D16C"] <= value["max_hp_D18D"] <= 1000 or value["max_hp_D18D"] == 0:
        failures.append(f"invalid HP {value['hp_D16C']}/{value['max_hp_D18D']}")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rom", required=True)
    p.add_argument("--state", default=str(ROOT / "local/has_pokedex_nballs.state"))
    p.add_argument("--out", default=str(ROOT / "artifacts/rom-compatibility"))
    args = p.parse_args()
    rom, state, out = Path(args.rom).resolve(), Path(args.state).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = rom.read_bytes()
    digest = hashlib.sha1(data).hexdigest()
    report = {"rom_sha1": digest, "rom_sha256": hashlib.sha256(data).hexdigest(),
              "size": len(data), "title": data[0x134:0x144].decode("ascii", "replace").rstrip("\0"),
              "mapper": data[0x147], "rom_size_code": data[0x148], "ram_size_code": data[0x149],
              "header_checksum_valid": ((-sum(data[0x134:0x14D])-25) & 255) == data[0x14D],
              "canonical_hash": digest == ROM_SHA1,
              "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
              "training_approved": False, "state_loaded": False}
    emulator = None
    try:
        emulator = PyBoy(str(rom), window="null", sound_emulated=False)
        emulator.set_emulation_speed(0)
        emulator.tick(180, True)
        image(emulator, out / "boot.png")
        emulator.tick(1620, True)
        image(emulator, out / "title.png")
        emulator.send_input(WindowEvent.PRESS_BUTTON_START)
        emulator.tick(8, True)
        emulator.send_input(WindowEvent.RELEASE_BUTTON_START)
        emulator.tick(120, True)
        image(emulator, out / "after_start.png")
        report["boot"] = "passed"
        report["boot_ram"] = snapshot(emulator)
        try:
            with state.open("rb") as stream:
                emulator.load_state(stream)
            report["state_loaded"] = True
            report["state_immediate_ram"] = snapshot(emulator)
            emulator.tick(2, True)
            report["state_after_tick_ram"] = snapshot(emulator)
            report["state_execution_errors"] = validate_snapshot(report["state_after_tick_ram"])
            report["state_execution_compatible"] = not report["state_execution_errors"]
            image(emulator, out / "state_loaded.png")
        except Exception as error:
            report["environment_32_steps"] = "failed"
            report["state_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    except Exception as error:
        report["boot_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if emulator is not None:
            emulator.stop(save=False)

    if report["state_loaded"]:
        env = None
        try:
            config = load_config(ROOT / "configs/smoke.yaml")
            config["environment"]["name"] = str(rom)
            config["environment"]["reset_params"]["init-state"] = str(state)
            env = wrap_environment(config["environment"], 0)
            obs, _ = env.reset()
            check_observation(obs, env.observation_space)
            report["observations"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                "min": float(v.min()), "max": float(v.max())} for k, v in obs.items()}
            initial = snapshot(env.unwrapped._env.pyboy)
            report["initial_ram"] = initial
            if not (1 <= initial["party_size_D163"] <= 6 and 1 <= initial["level_D18C"] <= 100
                    and 0 < initial["hp_D16C"] <= initial["max_hp_D18D"] <= 1000):
                raise AssertionError("Loaded state has invalid party/HP/level at official RAM addresses")
            trajectory = []
            report["trajectory"] = trajectory
            for index in range(32):
                action = index % 7
                obs, reward, done, info = env.step(action)
                check_observation(obs, env.observation_space)
                if not np.isfinite(reward):
                    raise FloatingPointError("Non-finite real reward")
                current_ram = snapshot(env.unwrapped._env.pyboy)
                trajectory.append({"step": index + 1, "action": action, "reward": float(reward),
                    "ram": current_ram, "observed_events": float(obs["events"].sum())})
                failures = validate_snapshot(current_ram)
                if failures:
                    image(env.unwrapped._env.pyboy, out / "first_invalid_step.png")
                    raise AssertionError("Official RAM semantics failed after step: " + "; ".join(failures))
                if done:
                    raise AssertionError("Unexpected episode truncation during short compatibility probe")
            image(env.unwrapped._env.pyboy, out / "after_steps.png")
            report["trajectory"] = trajectory
            env.reset()
            if snapshot(env.unwrapped._env.pyboy) != initial:
                raise AssertionError("Reset did not restore the initial RAM snapshot")
            report["environment_32_steps"] = "passed"
        except Exception as error:
            report["environment_32_steps"] = "failed"
            report["environment_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        finally:
            if env is not None:
                env.close()
    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "trajectory"}, ensure_ascii=True, indent=2))
    return 0 if report.get("state_execution_compatible") and report.get("environment_32_steps") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
