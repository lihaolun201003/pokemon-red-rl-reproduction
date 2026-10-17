"""Play your lawful ROM in PyBoy's SDL window; Ctrl+C in this terminal saves the initial state."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.runtime import ROOT, validate_rom
from pyboy import PyBoy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rom", default=str(ROOT / "local/PokemonRed.gb"))
    p.add_argument("--out", default=str(ROOT / "local/has_pokedex_nballs.state"))
    args = p.parse_args()
    validate_rom(args.rom)
    path = Path(args.out).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    emulator = PyBoy(str(Path(args.rom).resolve()), window="SDL2")
    emulator.set_emulation_speed(1)
    print("Use PyBoy's keyboard controls. For the baseline: choose Squirtle, deliver Oak's Parcel, obtain Pokedex/Poke Balls. Ctrl+C in THIS terminal saves state. Window close exits without saving.")
    try:
        while emulator.tick(1, True):
            pass
    except KeyboardInterrupt:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            emulator.save_state(stream)
        print(f"Saved private initial state: {path}")
    finally:
        emulator.stop(save=False)


if __name__ == "__main__":
    main()
