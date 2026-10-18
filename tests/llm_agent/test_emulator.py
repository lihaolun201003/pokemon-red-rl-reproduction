"""Tests for ``llm_agent.emulator`` (PyBoy 2.4 wrapper).

Real-ROM tests are marked ``real_rom`` and use the shared ``rom_path`` /
``state_path`` fixtures (they skip automatically when the lawful local files are
absent).  Each test stays tiny: load the official state and press a handful of
buttons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llm_agent import observation
from llm_agent.config import EmulatorConfig
from llm_agent.contracts import (
    SIGNIFICANT_EVENTS,
    ActionOutcome,
    ActionStep,
    EmulatorError,
    Frame,
    ScreenEvent,
    ToolValidationError,
)
from llm_agent.emulator import GameBoyEmulator


def make_config(rom_path: Path, state_path: Path, **overrides) -> EmulatorConfig:
    cfg = EmulatorConfig(rom=str(rom_path), state=str(state_path))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def emulator(rom_path: Path, state_path: Path):
    emu = GameBoyEmulator(make_config(rom_path, state_path)).start()
    try:
        yield emu
    finally:
        emu.close()


# --------------------------------------------------------------------------- #
# Unstarted-instance behavior (no ROM needed)
# --------------------------------------------------------------------------- #

def test_unstarted_operations_raise_emulator_error(tmp_path: Path) -> None:
    cfg = EmulatorConfig(rom=str(tmp_path / "missing.gb"), state=str(tmp_path / "missing.state"))
    emu = GameBoyEmulator(cfg)
    with pytest.raises(EmulatorError):
        emu.start()
    with pytest.raises(EmulatorError):
        emu.observe()
    with pytest.raises(EmulatorError):
        emu.press("a")
    with pytest.raises(EmulatorError):
        emu.wait(5)
    with pytest.raises(EmulatorError):
        emu.run_batch(["a"])
    with pytest.raises(EmulatorError):
        emu.save_state_bytes()
    with pytest.raises(EmulatorError):
        emu.load_state(b"x")
    emu.close()  # idempotent
    emu.close()
    assert emu.frame_count == 0
    assert emu.loaded_state is None
    assert emu.started is False
    with pytest.raises(EmulatorError):
        GameBoyEmulator(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Boot / observation
# --------------------------------------------------------------------------- #

@pytest.mark.real_rom
def test_start_observes_real_frame(emulator: GameBoyEmulator, state_path: Path) -> None:
    assert emulator.started is True
    frame = emulator.observe()
    assert isinstance(frame, Frame)
    assert frame.rgba.shape == (144, 160, 4)
    assert frame.rgba.dtype == np.uint8
    assert frame.sha256
    assert frame.frame_index == emulator.frame_count
    assert emulator.loaded_state == str(state_path.resolve())
    # a second observation neither ticks nor changes the screen
    again = emulator.observe()
    assert again.sha256 == frame.sha256
    assert again.frame_index == frame.frame_index


@pytest.mark.real_rom
def test_context_manager_starts_and_closes(rom_path: Path, state_path: Path) -> None:
    with GameBoyEmulator(make_config(rom_path, state_path)) as emu:
        frame = emu.observe()
        assert frame.rgba.shape == (144, 160, 4)
        assert emu.started is True
    assert emu.started is False


@pytest.mark.real_rom
def test_start_with_missing_explicit_state_raises(rom_path: Path, state_path: Path, tmp_path: Path) -> None:
    emu = GameBoyEmulator(make_config(rom_path, state_path))
    with pytest.raises(EmulatorError):
        emu.start(tmp_path / "does-not-exist.state")
    assert emu.started is False


# --------------------------------------------------------------------------- #
# Input: press / wait
# --------------------------------------------------------------------------- #

@pytest.mark.real_rom
def test_press_advances_exactly_action_frames(emulator: GameBoyEmulator) -> None:
    before = emulator.frame_count
    frame = emulator.press("down")
    assert emulator.frame_count - before == emulator.cfg.action_frames
    assert frame.frame_index == emulator.frame_count
    assert isinstance(observation.analyze(frame, None).changed, bool)


@pytest.mark.real_rom
def test_press_accepts_case_insensitive_button_and_frame_override(emulator: GameBoyEmulator) -> None:
    before = emulator.frame_count
    frame = emulator.press("A", frames=10)
    assert emulator.frame_count - before == 10
    assert frame.frame_index == emulator.frame_count


@pytest.mark.real_rom
@pytest.mark.parametrize("button", ["xyz", "", "a b", 3, None])
def test_press_rejects_invalid_buttons_before_ticking(emulator: GameBoyEmulator, button) -> None:
    before = emulator.frame_count
    with pytest.raises(ToolValidationError):
        emulator.press(button)  # type: ignore[arg-type]
    assert emulator.frame_count == before


@pytest.mark.real_rom
@pytest.mark.parametrize("frames", [1, 0, -5, 601, True, "24"])
def test_press_rejects_invalid_frame_override(emulator: GameBoyEmulator, frames) -> None:
    before = emulator.frame_count
    with pytest.raises(ToolValidationError):
        emulator.press("a", frames=frames)  # type: ignore[arg-type]
    assert emulator.frame_count == before


@pytest.mark.real_rom
def test_wait_advances_frames(emulator: GameBoyEmulator) -> None:
    before = emulator.frame_count
    frame = emulator.wait(15)
    assert emulator.frame_count - before == 15
    assert frame.frame_index == emulator.frame_count
    assert frame.rgba.shape == (144, 160, 4)


@pytest.mark.real_rom
@pytest.mark.parametrize("frames", [0, -1, 601, 1.5, False])
def test_wait_rejects_invalid_frames(emulator: GameBoyEmulator, frames) -> None:
    with pytest.raises(ToolValidationError):
        emulator.wait(frames)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# run_batch
# --------------------------------------------------------------------------- #

@pytest.mark.real_rom
def test_run_batch_completes_on_plain_movement(emulator: GameBoyEmulator) -> None:
    before = emulator.frame_count
    outcome = emulator.run_batch(["down"])
    assert isinstance(outcome, ActionOutcome)
    assert outcome.stopped_reason == "completed"
    assert outcome.buttons == ["down"]
    assert len(outcome.steps) == 1
    assert outcome.start_frame == before
    assert outcome.end_frame == emulator.frame_count
    assert emulator.frame_count - before == emulator.cfg.action_frames
    assert isinstance(outcome.steps[0], ActionStep)
    assert outcome.steps[0].button == "down"
    assert outcome.summary
    assert outcome.final_frame.frame_index == outcome.end_frame


@pytest.mark.real_rom
def test_run_batch_stops_early_on_significant_event(emulator: GameBoyEmulator) -> None:
    # Pressing A twice at the official starting position opens a text box, which
    # must interrupt the batch before the remaining presses are wasted.
    outcome = emulator.run_batch(["a", "a", "a", "a"], stop_on_event=True)
    assert outcome.stopped_reason == "event"
    assert len(outcome.steps) < 4
    assert len(outcome.steps) >= 1
    assert outcome.steps[-1].event in SIGNIFICANT_EVENTS
    assert outcome.events
    assert outcome.events[-1] in SIGNIFICANT_EVENTS


@pytest.mark.real_rom
def test_run_batch_continues_through_persistent_dialog(emulator: GameBoyEmulator) -> None:
    # Opening a dialogue needs one or two A presses at the official start position.
    event = None
    for _ in range(5):
        event = observation.analyze(emulator.press("a"), None).event
        if event == ScreenEvent.DIALOG:
            break
    assert event == ScreenEvent.DIALOG, "expected the start position to open a dialogue"
    # A DIALOG -> DIALOG sequence is one dialogue: the batch must run to the end.
    outcome = emulator.run_batch(["a", "a"], stop_on_event=True)
    assert outcome.stopped_reason == "completed"
    assert len(outcome.steps) == 2
    assert all(step.event == ScreenEvent.DIALOG for step in outcome.steps)
    assert "start_event=dialog" in outcome.summary


@pytest.mark.real_rom
def test_run_batch_continues_without_stop_on_event(emulator: GameBoyEmulator) -> None:
    outcome = emulator.run_batch(["a", "a"], stop_on_event=False)
    assert outcome.stopped_reason == "completed"
    assert len(outcome.steps) == 2


@pytest.mark.real_rom
def test_run_batch_validates_whole_batch_before_emulating(emulator: GameBoyEmulator) -> None:
    before = emulator.frame_count
    invalid_batches = [
        [],
        ["down", "nope"],
        ["down", 5],
        "down",
        ["down"] * (32 + 1),
    ]
    for batch in invalid_batches:
        with pytest.raises(ToolValidationError):
            emulator.run_batch(batch)  # type: ignore[arg-type]
    assert emulator.frame_count == before


@pytest.mark.real_rom
def test_run_batch_step_hook_receives_every_step(emulator: GameBoyEmulator) -> None:
    seen: list[ActionStep] = []
    outcome = emulator.run_batch(["down", "down"], step_hook=seen.append)
    assert len(seen) == len(outcome.steps) == 2
    assert all(step.button == "down" for step in seen)
    assert all(0.0 <= step.changed_ratio <= 1.0 for step in seen)
    with pytest.raises(ToolValidationError):
        emulator.run_batch(["down"], step_hook="not-callable")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# State save / load
# --------------------------------------------------------------------------- #

@pytest.mark.real_rom
def test_save_load_bytes_roundtrip_and_monotonic_counter(rom_path: Path, state_path: Path) -> None:
    emu = GameBoyEmulator(make_config(rom_path, state_path, settle_frames=0)).start()
    try:
        first = emu.save_state_bytes()
        assert len(first) > 0
        emu.wait(30)
        count_after_ticks = emu.frame_count
        assert count_after_ticks > 0
        emu.load_state(first)
        assert emu.frame_count >= count_after_ticks  # never moves backwards
        assert emu.loaded_state == "<memory>"
        second = emu.save_state_bytes()
        assert second == first  # the state payload itself round-trips exactly
        frame = emu.observe()
        assert frame.rgba.shape == (144, 160, 4)
        # pressing still works after a state load
        before = emu.frame_count
        emu.press("down")
        assert emu.frame_count - before == emu.cfg.action_frames
    finally:
        emu.close()


@pytest.mark.real_rom
def test_load_state_accepts_path_and_rejects_garbage(emulator: GameBoyEmulator, state_path: Path) -> None:
    with pytest.raises(EmulatorError):
        emulator.load_state(b"this is not a PyBoy state")
    with pytest.raises(EmulatorError):
        emulator.load_state(state_path.parent / "missing.state")
    emulator.load_state(state_path)
    assert emulator.loaded_state == str(state_path.resolve())
    assert emulator.observe().rgba.shape == (144, 160, 4)


@pytest.mark.real_rom
def test_save_state_file_is_atomic_and_loadable(emulator: GameBoyEmulator, tmp_path: Path) -> None:
    target = tmp_path / "checkpoints" / "state.state"
    written = emulator.save_state_file(target)
    assert written == target
    assert target.is_file() and target.stat().st_size > 0
    assert not list(target.parent.glob("*.tmp"))
    emulator.load_state(target)
    assert emulator.observe().rgba.shape == (144, 160, 4)
    # overwriting an existing checkpoint works too
    emulator.save_state_file(target)
    assert target.stat().st_size > 0
