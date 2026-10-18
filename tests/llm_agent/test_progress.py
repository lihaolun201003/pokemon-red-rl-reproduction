"""Tests for the verification-only RAM progress probe and report tooling.

These tests prove the probe works on the real ROM/state, but they also document
that it is *not* part of the decision loop (see test_import_boundaries.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_agent.progress import compare_progress, probe_from_state_file


@pytest.mark.real_rom
def test_probe_initial_state_is_sane(rom_path: Path, state_path: Path) -> None:
    snapshot = probe_from_state_file(rom_path, state_path)
    data = snapshot.to_dict()
    assert data["map_id"] >= 0
    assert 0 <= data["badges"] <= 8
    assert 0 <= data["party_count"] <= 6
    assert len(data["party_levels"]) == data["party_count"]
    assert all(level >= 0 for level in data["party_levels"])
    assert 0.0 <= data["hp_fraction"] <= 1.0
    assert "map_id=" in snapshot.summary()


@pytest.mark.real_rom
def test_probe_accepts_raw_state_bytes(rom_path: Path, state_path: Path) -> None:
    raw = state_path.read_bytes()
    snapshot = probe_from_state_file(rom_path, raw)
    assert snapshot.map_id >= 0


@pytest.mark.real_rom
def test_compare_progress_reports_honest_deltas(rom_path: Path, state_path: Path) -> None:
    snapshot = probe_from_state_file(rom_path, state_path)
    comparison = compare_progress(snapshot, snapshot)
    assert comparison["map_changed"] is False
    assert comparison["badges_gained"] == 0
    assert comparison["levels_gained"] == 0
    assert comparison["coord_distance"] == 0


def test_progress_script_resolves_checkpoint_json(tmp_path: Path) -> None:
    from scripts.llm_agent_progress import resolve_state_path

    state_file = tmp_path / "checkpoints" / "states" / "checkpoint_000000010.state"
    state_file.parent.mkdir(parents=True)
    state_file.write_bytes(b"fake-state")
    checkpoint = tmp_path / "checkpoints" / "checkpoint_000000010.json"
    checkpoint.write_text(
        json.dumps({"meta": {"id": "checkpoint_000000010", "state_file": str(state_file)}}),
        encoding="utf-8",
    )
    assert resolve_state_path(checkpoint) == state_file.resolve()
    latest = tmp_path / "checkpoints" / "latest.json"
    latest.write_text(json.dumps({"id": "checkpoint_000000010", "state_file": str(state_file)}), encoding="utf-8")
    assert resolve_state_path(tmp_path) == state_file.resolve()
    direct = tmp_path / "direct.state"
    direct.write_bytes(b"x")
    assert resolve_state_path(direct) == direct.resolve()
    with pytest.raises(SystemExit):
        resolve_state_path(tmp_path / "nope")
