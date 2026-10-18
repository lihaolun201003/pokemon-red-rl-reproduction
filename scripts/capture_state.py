"""Play your lawful ROM in PyBoy's SDL window; Ctrl+C in this terminal saves a state.

The project never ships or downloads game data.  This helper lets you build your
own checkpoint from your own legally obtained English Pokémon Red ROM and the
initial state, then save it under ``local/`` (gitignored) for the agent runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pyboy import PyBoy  # noqa: E402

from rom_checks import STATE_FILENAME, default_state_path, validate_rom  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default=None, help="lawful English Pokémon Red ROM (default local/)")
    parser.add_argument("--out", default=str(default_state_path()),
                        help=f"output state path (default local/{STATE_FILENAME})")
    parser.add_argument("--speed", type=float, default=1.0, help="emulation speed (1.0 = real time)")
    args = parser.parse_args()

    rom = validate_rom(args.rom)
    target = Path(args.out).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing state: {target}")

    emulator = PyBoy(str(rom), window="SDL2")
    emulator.set_emulation_speed(float(args.speed))
    print(
        "Play the game in the PyBoy window.  A useful starting point for the agent is a "
        "fresh game state with the intro finished (Pokédex obtained, 0 Poké Balls).\n"
        "Press Ctrl+C in THIS terminal to save the current state; closing the window exits "
        "without saving."
    )
    try:
        while emulator.tick(1, render=True):
            pass
    except KeyboardInterrupt:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            emulator.save_state(stream)
        print(f"Saved state: {target}")
    finally:
        emulator.stop(save=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
