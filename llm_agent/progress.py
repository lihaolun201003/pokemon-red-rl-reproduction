"""VERIFICATION-ONLY game progress probe.

This module is the single, explicitly separated exception to the agent's blind
exploration rule: it reads *hidden* game RAM (map id, coordinates, badge count,
party levels) so that reports and tests can independently verify how far the LLM
agent actually got.  That information is used for **audit/reporting only**.

It must never be imported by the decision loop:

    agent.py, runtime.py, prompts.py, tools.py, memory.py, guard.py,
    emulator.py, observation.py, api_client.py, budget.py, viewer.py

A test (``tests/llm_agent/test_import_boundaries.py``) enforces that boundary.
The model itself only ever receives screenshots and its own recorded memory.

Addresses are the fixed Pokémon Red WRAM locations used by the archived PokeRedV2
baseline (``archive/ppo-reproduction`` branch); they are consumed here solely to
count objectives for the run report and never enter the agent decision loop.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# PokeRedV2 RAM addresses (validated in this repository's PPO environment).
ADDR_BADGES = 0xD356          # bit field: obtained badges
ADDR_MAP_ID = 0xD35E          # current map bank/number byte
ADDR_X = 0xD362
ADDR_Y = 0xD361
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_SPECIES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
ADDR_PARTY_LEVELS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
ADDR_PARTY_HP = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
ADDR_PARTY_MAX_HP = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]


@dataclass
class ProgressSnapshot:
    """Independent (RAM-based) verification snapshot.  NEVER fed to the agent."""

    map_id: int = -1
    x: int = -1
    y: int = -1
    badges: int = 0
    party_count: int = 0
    party_species: list[int] = field(default_factory=list)
    party_levels: list[int] = field(default_factory=list)
    party_hp: list[int] = field(default_factory=list)
    party_max_hp: list[int] = field(default_factory=list)
    frame: int = 0

    @property
    def level_sum(self) -> int:
        return int(sum(self.party_levels))

    @property
    def hp_fraction(self) -> float:
        total = sum(self.party_hp)
        maximum = sum(self.party_max_hp)
        return float(total / maximum) if maximum else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "coords": [self.x, self.y],
            "badges": self.badges,
            "party_count": self.party_count,
            "party_species": self.party_species,
            "party_levels": self.party_levels,
            "level_sum": self.level_sum,
            "hp_fraction": round(self.hp_fraction, 4),
            "frame": self.frame,
        }

    def summary(self) -> str:
        return (
            f"map_id={self.map_id} coords=({self.x},{self.y}) badges={self.badges} "
            f"party={self.party_count} levels={self.party_levels} hp={self.hp_fraction:.2f}"
        )


#: Player overworld facing byte (sprite 0 ``imageIndex``; 0=down 4=up 8=left 12=right)
ADDR_PLAYER_FACING = 0xC102

FACING_NAMES: dict[int, str] = {0: "down", 4: "up", 8: "left", 12: "right"}


@dataclass
class OverworldProbe:
    """Ground truth for navigation tests: map, coordinates and facing.

    **TEST/VERIFICATION ONLY** — must never be imported by
    ``llm_agent.navigation`` or any decision-loop module.  It exists so the
    real-game localization validation (G1) can compare the pure-vision estimate
    with the emulator's actual position without leaking RAM into navigation.
    """

    map_id: int = -1
    x: int = -1
    y: int = -1
    facing: int = -1
    frame: int = 0

    @property
    def facing_name(self) -> str | None:
        return FACING_NAMES.get(int(self.facing))

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_id": int(self.map_id),
            "x": int(self.x),
            "y": int(self.y),
            "facing": int(self.facing),
            "facing_name": self.facing_name,
            "frame": int(self.frame),
        }


def probe_overworld(pyboy: Any) -> OverworldProbe:
    """Read map/x/y/facing from a live PyBoy instance (verification only)."""
    memory = pyboy.memory
    return OverworldProbe(
        map_id=int(memory[ADDR_MAP_ID]),
        x=int(memory[ADDR_X]),
        y=int(memory[ADDR_Y]),
        facing=int(memory[ADDR_PLAYER_FACING]),
        frame=int(getattr(pyboy, "frame_count", 0)),
    )


def probe_from_pyboy(pyboy: Any) -> ProgressSnapshot:
    """Read the verification snapshot from a live PyBoy instance."""
    memory = pyboy.memory

    def read(addr: int) -> int:
        return int(memory[addr])

    badges = bin(read(ADDR_BADGES)).count("1")
    party_count = int(np.clip(read(ADDR_PARTY_COUNT), 0, 6))
    species = [read(addr) for addr in ADDR_PARTY_SPECIES]
    levels = [read(addr) for addr in ADDR_PARTY_LEVELS]
    hp = [read(ADDR_PARTY_HP[i]) * 256 + read(ADDR_PARTY_HP[i] + 1) for i in range(party_count)]
    max_hp = [read(ADDR_PARTY_MAX_HP[i]) * 256 + read(ADDR_PARTY_MAX_HP[i] + 1) for i in range(party_count)]
    return ProgressSnapshot(
        map_id=read(ADDR_MAP_ID),
        x=read(ADDR_X),
        y=read(ADDR_Y),
        badges=badges,
        party_count=party_count,
        party_species=species[:party_count],
        party_levels=levels[:party_count],
        party_hp=hp,
        party_max_hp=max_hp,
        frame=int(getattr(pyboy, "frame_count", 0)),
    )


def probe_from_state_file(
    rom_path: str | Path,
    state_path: str | Path | bytes,
    *,
    settle_frames: int = 1,
) -> ProgressSnapshot:
    """Load a ROM + saved emulator state in a throw-away PyBoy and probe it.

    Used by report/test tooling only.  The state may be a path or raw state bytes.
    """
    from pyboy import PyBoy  # local import: this module is never part of the agent loop

    rom_path = Path(rom_path)
    if not rom_path.is_file():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False, log_level="ERROR")
    try:
        if isinstance(state_path, (bytes, bytearray)):
            pyboy.load_state(io.BytesIO(bytes(state_path)))
        else:
            state_file = Path(state_path)
            if not state_file.is_file():
                raise FileNotFoundError(f"State not found: {state_file}")
            with open(state_file, "rb") as handle:
                pyboy.load_state(handle)
        if settle_frames:
            pyboy.tick(settle_frames, render=False)
        return probe_from_pyboy(pyboy)
    finally:
        pyboy.stop(save=False)


def probe_json(rom_path: str | Path, state_path: str | Path | bytes) -> str:
    """Convenience JSON string for reports."""
    return json.dumps(probe_from_state_file(rom_path, state_path).to_dict(), ensure_ascii=False)


def compare_progress(start: ProgressSnapshot, end: ProgressSnapshot) -> dict[str, Any]:
    """Honest progress comparison for the run report (verification data only)."""
    return {
        "start": start.to_dict(),
        "end": end.to_dict(),
        "map_changed": start.map_id != end.map_id,
        "coord_distance": abs(end.x - start.x) + abs(end.y - start.y) if min(start.x, start.y, end.x, end.y) >= 0 else None,
        "badges_gained": end.badges - start.badges,
        "levels_gained": end.level_sum - start.level_sum,
        "party_size_change": end.party_count - start.party_count,
    }
