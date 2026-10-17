"""Regression: plausible observation bounds do not prove valid game state."""
from scripts.probe_rom import validate_snapshot


def test_initial_party_semantics():
    assert validate_snapshot({"party_size_D163":1,"level_D18C":6,"hp_D16C":22,"max_hp_D18D":22}) == []


def test_corrupt_state_rejected_despite_finite_numbers():
    # Observed after loading upstream state into the user's incompatible modified ROM.
    failures = validate_snapshot({"party_size_D163":57,"level_D18C":0,"hp_D16C":57,"max_hp_D18D":14592})
    assert len(failures)==3
