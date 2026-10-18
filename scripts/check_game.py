"""Real-ROM smoke test for the emulator wrapper (no API, no RAM reads).

Boots the user's lawful English Pokémon Red ROM in PyBoy through
:class:`llm_agent.emulator.GameBoyEmulator`, presses a few bounded buttons,
verifies that frames are the expected 160x144 RGBA screenshots, exercises a
save-state round-trip and shuts the emulator down cleanly.

Fails explicitly when the ROM is missing or not the supported dump.  This is an
environment check only — the agent itself never touches hidden game state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from llm_agent.config import AgentConfig  # noqa: E402
from llm_agent.emulator import GameBoyEmulator  # noqa: E402
from llm_agent.observation import analyze, save_png  # noqa: E402
from rom_checks import (  # noqa: E402
    ROM_FILENAME,
    STATE_FILENAME,
    sha256_file,
    validate_rom,
    validate_state,
)

BUTTONS = ("up", "down", "left", "right", "a", "b")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default=None, help=f"lawful ROM path (default: local/{ROM_FILENAME})")
    parser.add_argument("--state", default=None, help=f"initial state (default: local/{STATE_FILENAME})")
    parser.add_argument("--steps", type=int, default=64, help="number of bounded button presses (1..256)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "game_check.json"))
    args = parser.parse_args()
    if not 1 <= args.steps <= 256:
        parser.error("--steps must be in 1..256")

    rom = validate_rom(args.rom)
    state = validate_state(args.state)
    cfg = AgentConfig()
    cfg.emulator.rom = str(rom)
    cfg.emulator.state = str(state)
    cfg.emulator.window = "null"
    cfg.emulator.render = True

    emulator = GameBoyEmulator(cfg.emulator)
    try:
        emulator.start()
        frames = []
        event_counts: dict[str, int] = {}
        for index in range(args.steps):
            button = BUTTONS[index % len(BUTTONS)]
            frame = emulator.press(button)
            array = np.asarray(frame.rgba)
            assert array.shape == (144, 160, 4), f"unexpected frame shape {array.shape}"
            assert array.dtype == np.uint8, f"unexpected frame dtype {array.dtype}"
            analysis = analyze(frame, frames[-1] if frames else None)
            event_counts[analysis.event.value] = event_counts.get(analysis.event.value, 0) + 1
            frames.append(frame)
        save_png(frames[-1], PROJECT_ROOT / "artifacts" / "game_check_last_frame.png")

        # save-state round trip through the wrapper the agents use
        payload = emulator.save_state_bytes()
        emulator.load_state(payload)
        restored = emulator.observe()
        report = {
            "real_game": True,
            "rom": rom.name,
            "state": state.name,
            "state_sha256": sha256_file(state),
            "frames": [frame.frame_index for frame in frames[:8]],
            "frame_count": int(emulator.frame_count),
            "screen": list(restored.rgba.shape),
            "buttons_pressed": args.steps,
            "screen_event_counts": event_counts,
            "save_state_roundtrip": "passed",
            "mode": "real ROM via llm_agent.emulator (PyBoy), no API, no RAM reads",
        }
    finally:
        emulator.close()

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
