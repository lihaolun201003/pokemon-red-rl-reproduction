"""Read-only helpers for the user's lawful ROM / initial save state.

The project never ships or downloads game data.  These helpers only *verify*
that the file the user already owns is the expected English Pokémon Red dump
(the same SHA1/SHA256 the earlier reproduction recorded) so that every script
fails clearly instead of silently running on an incompatible cartridge.

Only file hashing is performed here — no emulator memory is read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Canonical English Pokémon Red (pret/pokered build), 1 MiB.
ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"
ROM_FILENAME = "PokemonRed.gb"

#: Official initial state shipped with the earlier reproduction (Pokédex, 0 balls).
STATE_SHA256 = "0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e"
STATE_FILENAME = "has_pokedex_nballs.state"


def sha1_file(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_rom_path() -> Path:
    return PROJECT_ROOT / "local" / ROM_FILENAME


def default_state_path() -> Path:
    return PROJECT_ROOT / "local" / STATE_FILENAME


def validate_rom(path: str | Path | None = None) -> Path:
    """Return the ROM path after verifying it is the canonical English dump."""
    target = Path(path) if path else default_rom_path()
    if not target.is_file():
        raise FileNotFoundError(
            f"Lawful English Pokémon Red ROM not found: {target}\n"
            f"Place your own legally obtained dump at local/{ROM_FILENAME} "
            "or pass --rom explicitly. This repository never ships or downloads ROMs."
        )
    actual = sha1_file(target)
    if actual != ROM_SHA1:
        raise ValueError(
            f"ROM SHA1 mismatch for {target}\nexpected {ROM_SHA1}\nactual   {actual}\n"
            "Only the canonical English Pokémon Red dump is supported."
        )
    return target.resolve()


def validate_state(path: str | Path | None = None) -> Path:
    """Return the initial-state path after verifying its canonical SHA256."""
    target = Path(path) if path else default_state_path()
    if not target.is_file():
        raise FileNotFoundError(
            f"Initial emulator state not found: {target}\n"
            f"Expected the official initial state at local/{STATE_FILENAME} "
            "(or pass --state explicitly)."
        )
    actual = sha256_file(target)
    if actual != STATE_SHA256:
        raise ValueError(
            f"initial state SHA256 mismatch for {target}\nexpected {STATE_SHA256}\nactual   {actual}"
        )
    return target.resolve()


__all__ = [
    "PROJECT_ROOT",
    "ROM_SHA1",
    "ROM_FILENAME",
    "STATE_SHA256",
    "STATE_FILENAME",
    "sha1_file",
    "sha256_file",
    "default_rom_path",
    "default_state_path",
    "validate_rom",
    "validate_state",
]
