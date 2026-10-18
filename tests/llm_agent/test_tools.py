"""Tests for ``llm_agent.tools`` —schemas, strict validation, executor behavior.

Everything here is synthetic (fake emulator + fake memory), so the suite runs
without the real ROM, without PyBoy state files and without any network/API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from llm_agent import observation
from llm_agent.config import AgentConfig
from llm_agent.contracts import (
    BUTTONS,
    DIRECTIONS,
    DISCOVERY_KINDS,
    MAX_ACTIONS_PER_CALL,
    MAX_WALK_STEPS,
    TOOL_NAMES,
    ActionOutcome,
    ActionStep,
    EmulatorError,
    Frame,
    RuntimeState,
    ScreenEvent,
    SIGNIFICANT_EVENTS,
    ToolCall,
    ToolResult,
)
from llm_agent.tools import (
    DEFAULT_HISTORY_LIMIT,
    TOOL_SCHEMAS,
    TOOL_SCHEMAS_BY_NAME,
    ToolExecutor,
)

# --------------------------------------------------------------------------- #
# Synthetic frames
# --------------------------------------------------------------------------- #

def _base(fill: int = 153) -> np.ndarray:
    rgba = np.full((144, 160, 4), fill, dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


def overworld_rgba(offset: int = 0) -> np.ndarray:
    rgba = _base(153)
    rgba[16:48, 8:40, :3] = 0
    rgba[64:96, 112:152, :3] = 85
    rgba[112:140, 24 + offset:72 + offset, :3] = 0
    return rgba


def dialog_rgba() -> np.ndarray:
    rgba = overworld_rgba()
    rgba[96:142, 4:156, :3] = 0
    rgba[98:140, 6:154, :3] = 255
    rgba[104:110, 16:40, :3] = 0
    return rgba


def dialog_page_rgba(page: int = 0) -> np.ndarray:
    """Dialog screen with different text marks per page (event stays DIALOG)."""
    rgba = overworld_rgba()
    rgba[96:142, 4:156, :3] = 0
    rgba[98:140, 6:154, :3] = 255
    for row in range(2):
        start = 12 + (page * 23 + row * 31) % 110
        top = 104 + row * 16
        rgba[top:top + 6, start:start + 18, :3] = 0
    return rgba


def frames_from(*rgbas: np.ndarray) -> list[Frame]:
    return [
        observation.frame_from_rgba(index, rgba, now=1000.0 + index)
        for index, rgba in enumerate(rgbas)
    ]


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeEmulator:
    """Minimal stand-in for :class:`GameBoyEmulator` (same public surface)."""

    def __init__(self, frames: list[Frame], action_frames: int = 24) -> None:
        assert frames
        self.frames = frames
        self.action_frames = action_frames
        self._index = 0
        self.frame_count = 0
        self.presses: list[tuple[str, int | None]] = []
        self.waits: list[int] = []
        self.batches: list[list[str]] = []
        self.loaded_state: str | None = None

    @property
    def current(self) -> Frame:
        return self.frames[min(self._index, len(self.frames) - 1)]

    def observe(self) -> Frame:
        return self.current

    def press(self, button: str, *, frames: int | None = None) -> Frame:
        self.presses.append((button, frames))
        self._index += 1
        self.frame_count += frames or self.action_frames
        return self.current

    def wait(self, frames: int) -> Frame:
        self.waits.append(frames)
        self._index += 1
        self.frame_count += frames
        return self.current

    def run_batch(
        self,
        buttons,
        *,
        stop_on_event: bool = True,
        step_hook=None,
    ) -> ActionOutcome:
        """Mirror GameBoyEmulator.run_batch: stop only on an event *transition*."""
        self.batches.append(list(buttons))
        start = self.frame_count
        previous = self.observe()
        baseline_event = observation.analyze(previous, None).event
        baseline_black = bool(previous.is_black)
        steps: list[ActionStep] = []
        events: list[ScreenEvent] = []
        stopped_reason = "completed"
        for button in buttons:
            frame = self.press(button)
            analysis = observation.analyze(frame, previous)
            step = ActionStep(
                button=button,
                frame=frame,
                screen_changed=analysis.changed,
                changed_ratio=analysis.changed_ratio,
                event=analysis.event,
            )
            steps.append(step)
            if analysis.event != ScreenEvent.NONE:
                events.append(analysis.event)
            if step_hook is not None:
                step_hook(step)
            previous = frame
            if not stop_on_event:
                continue
            transitioned = (
                analysis.event != baseline_event
                and (
                    analysis.event in SIGNIFICANT_EVENTS
                    or baseline_event in SIGNIFICANT_EVENTS
                )
            )
            if transitioned or (analysis.is_black and not baseline_black):
                stopped_reason = "event"
                break
        return ActionOutcome(
            buttons=list(buttons),
            steps=steps,
            start_frame=start,
            end_frame=self.frame_count,
            stopped_reason=stopped_reason,
            events=events,
            summary=f"fake batch {len(steps)}/{len(buttons)}",
        )


class BrokenEmulator(FakeEmulator):
    """Emulator whose actions always fail (tests error propagation)."""

    def press(self, button: str, *, frames: int | None = None) -> Frame:  # type: ignore[override]
        raise EmulatorError("emulator exploded")

    def wait(self, frames: int) -> Frame:  # type: ignore[override]
        raise EmulatorError("emulator exploded")

    def run_batch(self, buttons, *, stop_on_event: bool = True, step_hook=None) -> ActionOutcome:  # type: ignore[override]
        raise EmulatorError("emulator exploded")


class FakeMemory:
    """Minimal duck-typed memory matching the documented AgentMemory interface."""

    def __init__(self) -> None:
        self.goal = {"text": "", "reason": ""}
        self.discoveries: list[dict[str, Any]] = []
        self.places: list[dict[str, Any]] = []
        self.context = "AGENT MEMORY\\nGoal: explore the area."
        self.search_calls: list[tuple[str, int]] = []
        self.rendered = 0

    def set_goal(self, goal: str, reason: str = "") -> dict:
        self.goal = {"text": goal, "reason": reason}
        return dict(self.goal)

    def get_goal(self) -> dict:
        return dict(self.goal)

    def record_discovery(self, text: str, *, kind: str = "other", place: str = "") -> dict:
        entry = {
            "id": f"d-{len(self.discoveries) + 1}",
            "text": text,
            "kind": kind,
            "place": place,
        }
        self.discoveries.append(entry)
        return dict(entry)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        self.search_calls.append((query, limit))
        return [{"text": "found note", "category": "discoveries", "score": 1.0}][:limit]

    def render_context(self, query: str | None = None, *, max_chars: int | None = None) -> str:
        self.rendered += 1
        text = self.context
        return text[:max_chars] if max_chars else text

    def stats(self) -> dict:
        return {
            "places": len(self.places),
            "discoveries": len(self.discoveries),
            "goal": self.goal["text"],
        }


class BareMemory:
    """Memory that only implements set_goal (used to test graceful failures)."""

    def set_goal(self, goal: str, reason: str = "") -> dict:
        return {"text": goal}


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #

def call(name: str, call_id: str = "call_1", arguments: dict[str, Any] | None = None, **kwargs: Any) -> ToolCall:
    payload = dict(arguments) if arguments is not None else dict(kwargs)
    return ToolCall(id=call_id, name=name, arguments=payload)


class ExecutorEnv:
    def __init__(self) -> None:
        self.executor: ToolExecutor
        self.emulator: FakeEmulator
        self.memory: FakeMemory
        self.state: RuntimeState
        self.saved: list[str]
        self.restored: list[str]


@pytest.fixture
def make_executor(tmp_path: Path):
    def factory(
        rgbas: list[np.ndarray] | None = None,
        *,
        frames: list[Frame] | None = None,
        memory: Any = None,
        state: RuntimeState | None = None,
        emulator: FakeEmulator | None = None,
        screen_writer=None,
        cfg: AgentConfig | None = None,
    ) -> ExecutorEnv:
        env = ExecutorEnv()
        env.emulator = emulator if emulator is not None else FakeEmulator(
            frames if frames is not None else frames_from(*(rgbas or [overworld_rgba()]))
        )
        env.memory = memory if memory is not None else FakeMemory()
        env.state = state if state is not None else RuntimeState(run_id="test")
        env.saved = []
        env.restored = []

        def save_progress(reason: str) -> str:
            env.saved.append(reason)
            return "checkpoint_000000024 (frame=24, buttons=1)"

        def restore_progress(checkpoint_id: str) -> str:
            env.restored.append(checkpoint_id)
            return f"restored {checkpoint_id}"

        env.executor = ToolExecutor(
            emulator=env.emulator,
            memory=env.memory,
            state=env.state,
            screen_dir=tmp_path / "screens",
            save_progress=save_progress,
            restore_progress=restore_progress,
            screen_writer=screen_writer,
            cfg=cfg if cfg is not None else AgentConfig(),
        )
        return env

    return factory


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

def test_tool_schemas_complete_and_ordered() -> None:
    assert len(TOOL_SCHEMAS) == len(TOOL_NAMES) == 18
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert names == list(TOOL_NAMES)
    assert set(TOOL_SCHEMAS_BY_NAME) == set(TOOL_NAMES)
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["description"].strip()
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert set(parameters["properties"]) == set(
            TOOL_SCHEMAS_BY_NAME[function["name"]]["function"]["parameters"]["properties"]
        )


def test_tool_schemas_are_constrained() -> None:
    press = TOOL_SCHEMAS_BY_NAME["press_button"]["function"]["parameters"]
    assert set(press["properties"]["button"]["enum"]) == set(BUTTONS)
    assert press["properties"]["frames"]["minimum"] == 2
    assert press["required"] == ["button"]

    actions = TOOL_SCHEMAS_BY_NAME["execute_actions"]["function"]["parameters"]
    assert actions["properties"]["actions"]["maxItems"] == MAX_ACTIONS_PER_CALL
    assert actions["properties"]["actions"]["minItems"] == 1
    assert set(actions["properties"]["actions"]["items"]["enum"]) == set(BUTTONS)
    assert actions["required"] == ["actions"]

    walk = TOOL_SCHEMAS_BY_NAME["walk"]["function"]["parameters"]
    assert set(walk["properties"]["direction"]["enum"]) == set(DIRECTIONS)
    assert walk["properties"]["max_steps"]["maximum"] == MAX_WALK_STEPS

    wait = TOOL_SCHEMAS_BY_NAME["wait_frames"]["function"]["parameters"]
    assert wait["properties"]["frames"]["minimum"] == 1
    assert wait["properties"]["frames"]["maximum"] == 600

    discovery = TOOL_SCHEMAS_BY_NAME["record_discovery"]["function"]["parameters"]
    assert set(discovery["properties"]["kind"]["enum"]) == set(DISCOVERY_KINDS)
    assert discovery["required"] == ["text"]


# --------------------------------------------------------------------------- #
# Strict validation
# --------------------------------------------------------------------------- #

def test_validate_rejects_unknown_tool_and_bad_arguments(make_executor) -> None:
    executor = make_executor().executor
    with pytest.raises(Exception):
        executor.validate("hack_the_game", {})
    with pytest.raises(Exception):
        executor.validate("", {})
    with pytest.raises(Exception):
        executor.validate("press_button", ["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(Exception):
        executor.validate("press_button", {"button": "a", "bogus": 1})
    with pytest.raises(Exception):
        executor.validate("press_button", {})


def test_validate_normalizes_buttons_and_does_not_mutate_input(make_executor) -> None:
    executor = make_executor().executor
    args = {"button": "A", "frames": 12}
    snapshot = dict(args)
    normalized = executor.validate("press_button", args)
    assert normalized == {"button": "a", "frames": 12}
    assert args == snapshot
    assert executor.validate("execute_actions", {"actions": ["UP", "a", "Start"]}) == {
        "actions": ["up", "a", "start"]
    }
    assert executor.validate("walk", {"direction": "LEFT"}) == {"direction": "left"}
    assert executor.validate("observe_game", {}) == {}


@pytest.mark.parametrize(
    "arguments",
    [
        {"button": 3},
        {"button": "xyz"},
        {"button": ""},
        {"button": True},
        {"button": "a", "frames": 1},
        {"button": "a", "frames": 601},
        {"button": "a", "frames": True},
        {"button": "a", "frames": "24"},
    ],
)
def test_validate_press_button_rejects_bad_values(make_executor, arguments) -> None:
    executor = make_executor().executor
    with pytest.raises(Exception):
        executor.validate("press_button", arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"actions": []},
        {"actions": "a"},
        {"actions": ["ok", "nope"]},
        {"actions": [1]},
        {"actions": ["a"] * (MAX_ACTIONS_PER_CALL + 1)},
        {"actions": ["a"], "stop_on_event": "yes"},
    ],
)
def test_validate_execute_actions_rejects_bad_values(make_executor, arguments) -> None:
    executor = make_executor().executor
    with pytest.raises(Exception):
        executor.validate("execute_actions", arguments)
    ok = executor.validate("execute_actions", {"actions": ["a"] * MAX_ACTIONS_PER_CALL})
    assert len(ok["actions"]) == MAX_ACTIONS_PER_CALL


@pytest.mark.parametrize(
    "arguments",
    [{"frames": 0}, {"frames": 601}, {"frames": "10"}, {"frames": True}, {}],
)
def test_validate_wait_frames_rejects_bad_values(make_executor, arguments) -> None:
    executor = make_executor().executor
    with pytest.raises(Exception):
        executor.validate("wait_frames", arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"direction": "north"},
        {"direction": 3},
        {"direction": "up", "max_steps": 0},
        {"direction": "up", "max_steps": MAX_WALK_STEPS + 1},
        {"direction": "up", "stop_on_event": 1},
    ],
)
def test_validate_walk_rejects_bad_values(make_executor, arguments) -> None:
    executor = make_executor().executor
    with pytest.raises(Exception):
        executor.validate("walk", arguments)


def test_validate_record_discovery_and_goal(make_executor) -> None:
    executor = make_executor().executor
    assert executor.validate("record_discovery", {"text": "  saw a sign  "}) == {
        "text": "saw a sign"
    }
    with pytest.raises(Exception):
        executor.validate("record_discovery", {"text": ""})
    with pytest.raises(Exception):
        executor.validate("record_discovery", {"text": "x" * 1201})
    with pytest.raises(Exception):
        executor.validate("record_discovery", {"text": "ok", "kind": "story"})
    assert executor.validate(
        "record_discovery", {"text": "ok", "kind": "place", "place": "somewhere"}
    ) == {"text": "ok", "kind": "place", "place": "somewhere"}
    with pytest.raises(Exception):
        executor.validate("set_goal", {"goal": ""})
    with pytest.raises(Exception):
        executor.validate("set_goal", {"goal": "x" * 301})


def test_validate_restore_checkpoint_formats(make_executor) -> None:
    executor = make_executor().executor
    assert executor.validate("restore_checkpoint", {}) == {"checkpoint": "latest"}
    assert executor.validate("restore_checkpoint", {"checkpoint": "checkpoint_000000024"}) == {
        "checkpoint": "checkpoint_000000024"
    }
    for bad in ("bogus", "", "checkpoint_", "checkpoint_" + "x" * 61):
        with pytest.raises(Exception):
            executor.validate("restore_checkpoint", {"checkpoint": bad})


def test_validate_history_and_memory_limits(make_executor) -> None:
    executor = make_executor().executor
    assert executor.validate("get_recent_history", {}) == {"limit": DEFAULT_HISTORY_LIMIT}
    assert executor.validate("get_recent_history", {"limit": 5}) == {"limit": 5}
    for bad in ({"limit": 0}, {"limit": 33}, {"limit": "5"}):
        with pytest.raises(Exception):
            executor.validate("get_recent_history", bad)
    assert executor.validate("inspect_memory", {}) == {}
    with pytest.raises(Exception):
        executor.validate("inspect_memory", {"query": "x" * 201})
    with pytest.raises(Exception):
        executor.validate("inspect_memory", {"limit": 0})


# --------------------------------------------------------------------------- #
# execute(): never raises, returns ToolResult
# --------------------------------------------------------------------------- #

def test_execute_rejects_unknown_tool_without_raising(make_executor) -> None:
    env = make_executor()
    result = env.executor.execute(call("hack_tool", arguments={"code": "print(1)"}))
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error
    assert result.needs_llm is True
    assert result.tool_call_id == "call_1"
    assert env.executor.last_tool == "hack_tool"
    assert any(entry.kind == "error" for entry in env.state.history)


def test_execute_bad_arguments_do_not_press_buttons(make_executor) -> None:
    env = make_executor(rgbas=[overworld_rgba()])
    result = env.executor.execute(call("press_button", arguments={"button": "nope"}))
    assert result.ok is False and result.needs_llm is True
    assert env.emulator.presses == []
    assert env.executor.last_outcome is None


def test_execute_emulator_failure_returns_error_result(make_executor) -> None:
    env = make_executor(emulator=BrokenEmulator(frames_from(overworld_rgba())))
    result = env.executor.execute(call("press_button", arguments={"button": "a"}))
    assert result.ok is False
    assert "EmulatorError" in (result.error or "") or "exploded" in (result.error or "")
    assert result.needs_llm is True
    assert any(entry.kind == "error" for entry in env.state.history)


def test_executor_constructor_validation(make_executor, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ToolExecutor(
            emulator=None,  # type: ignore[arg-type]
            memory=FakeMemory(),
            state=RuntimeState(),
            screen_dir=tmp_path,
            save_progress=lambda reason: "x",
            restore_progress=lambda checkpoint: "y",
        )
    with pytest.raises(ValueError):
        ToolExecutor(
            emulator=FakeEmulator(frames_from(overworld_rgba())),
            memory=FakeMemory(),
            state=RuntimeState(),
            screen_dir=tmp_path,
            save_progress="not callable",  # type: ignore[arg-type]
            restore_progress=lambda checkpoint: "y",
        )
    with pytest.raises(ValueError):
        ToolExecutor(
            emulator=FakeEmulator(frames_from(overworld_rgba())),
            memory=FakeMemory(),
            state=RuntimeState(),
            screen_dir=tmp_path,
            save_progress=lambda reason: "x",
            restore_progress=lambda checkpoint: "y",
            screen_writer=123,  # type: ignore[arg-type]
        )


def test_last_observation_is_none_before_any_observation(make_executor) -> None:
    executor = make_executor().executor
    assert executor.last_observation() is None
    assert executor.last_frame is None
    assert executor.last_analysis is None
    assert executor.last_outcome is None


# --------------------------------------------------------------------------- #
# observe_game
# --------------------------------------------------------------------------- #

def test_observe_game_result_and_cached_observation(make_executor, tmp_path: Path) -> None:
    env = make_executor(rgbas=[overworld_rgba()])
    result = env.executor.execute(call("observe_game"))
    assert result.ok is True and result.needs_llm is False
    assert "event=" in result.summary and "signature=" in result.summary
    assert result.data["event"] == "none"
    assert result.data["frame"] == 0
    observation_pair = env.executor.last_observation()
    assert observation_pair is not None
    frame, analysis = observation_pair
    assert isinstance(frame, Frame) and isinstance(analysis, observation.ScreenAnalysis)
    assert env.state.frame_index == frame.frame_index
    assert env.state.last_action == "observe_game"
    assert len(result.screens) == 1
    assert (tmp_path / "screens" / result.screens[0]).is_file()
    assert result.screens[0].startswith("tool_")


def test_observe_game_uses_injected_screen_writer(make_executor, tmp_path: Path) -> None:
    written: list[Frame] = []

    def writer(frame: Frame) -> str:
        written.append(frame)
        return f"custom_{frame.frame_index}.png"

    env = make_executor(rgbas=[overworld_rgba()], screen_writer=writer)
    result = env.executor.execute(call("observe_game"))
    assert result.screens == ["custom_0.png"]
    assert len(written) == 1
    assert not (tmp_path / "screens").exists()


# --------------------------------------------------------------------------- #
# press_button / execute_actions / wait_frames
# --------------------------------------------------------------------------- #

def test_press_button_executes_and_updates_state(make_executor) -> None:
    env = make_executor(rgbas=[overworld_rgba(), overworld_rgba(offset=8)])
    result = env.executor.execute(call("press_button", arguments={"button": "down"}))
    assert result.ok is True
    assert env.emulator.presses == [("down", None)]
    assert env.state.buttons_pressed == 1
    assert env.state.action_batches == 1
    assert env.executor.last_outcome is not None
    outcome = env.executor.last_outcome
    assert isinstance(outcome, ActionOutcome)
    assert outcome.steps and outcome.steps[0].button == "down"
    for token in ("actions=1", "event=", "signature=", "black=", "stopped="):
        assert token in result.summary
    assert result.needs_llm is False
    assert any(entry.kind == "tool" and "press down" in entry.summary for entry in env.state.history)


def test_press_button_frame_override_is_forwarded(make_executor) -> None:
    env = make_executor(rgbas=[overworld_rgba(), overworld_rgba(offset=8)])
    result = env.executor.execute(call("press_button", arguments={"button": "a", "frames": 10}))
    assert result.ok is True
    assert env.emulator.presses == [("a", 10)]
    assert "(10 frames)" in result.summary


def test_press_button_dialog_sets_needs_llm(make_executor) -> None:
    env = make_executor(rgbas=[overworld_rgba(), dialog_rgba()])
    result = env.executor.execute(call("press_button", arguments={"button": "a"}))
    assert result.ok is True
    assert result.data["event"] == "dialog"
    assert result.needs_llm is True


def test_press_button_persistent_dialog_does_not_interrupt(make_executor) -> None:
    env = make_executor(rgbas=[dialog_page_rgba(0), dialog_page_rgba(1)])
    env.executor.execute(call("observe_game"))
    result = env.executor.execute(call("press_button", arguments={"button": "a"}))
    assert result.ok is True
    assert result.data["start_event"] == "dialog"
    assert result.data["event"] == "dialog"
    assert result.data["stopped_reason"] == "completed"
    assert result.needs_llm is False
    assert "stopped=completed" in result.summary


def test_execute_actions_stops_early_on_event(make_executor) -> None:
    env = make_executor(
        rgbas=[overworld_rgba(), overworld_rgba(offset=8), dialog_rgba(), dialog_rgba()]
    )
    result = env.executor.execute(
        call("execute_actions", arguments={"actions": ["a", "a", "a", "a"]})
    )
    assert result.ok is True
    assert env.emulator.batches == [["a", "a", "a", "a"]]
    assert result.data["executed_actions"] < 4
    assert result.needs_llm is True
    assert env.state.buttons_pressed == result.data["executed_actions"]
    assert env.state.action_batches == 1
    assert env.executor.last_outcome is not None
    assert env.executor.last_outcome.stopped_reason == "event"


def test_execute_actions_completes_without_event(make_executor) -> None:
    env = make_executor(
        rgbas=[overworld_rgba(), overworld_rgba(offset=4), overworld_rgba(offset=8)]
    )
    result = env.executor.execute(
        call("execute_actions", arguments={"actions": ["down", "down"]})
    )
    assert result.ok is True
    assert result.data["executed_actions"] == 2
    assert result.data["stopped_reason"] == "completed"
    assert result.needs_llm is False
    assert env.state.buttons_pressed == 2


def test_execute_actions_persistent_dialog_continues(make_executor) -> None:
    """Reading several dialogue pages with A presses is ONE batch, not a stop."""
    env = make_executor(
        rgbas=[
            dialog_page_rgba(0),
            dialog_page_rgba(1),
            dialog_page_rgba(2),
            dialog_page_rgba(3),
        ]
    )
    env.executor.execute(call("observe_game"))  # executor now knows the screen is DIALOG
    result = env.executor.execute(
        call("execute_actions", arguments={"actions": ["a", "a", "a"]})
    )
    assert result.ok is True
    assert result.data["start_event"] == "dialog"
    assert result.data["event"] == "dialog"
    assert result.data["executed_actions"] == 3
    assert result.data["stopped_reason"] == "completed"
    assert result.needs_llm is False
    assert "Executed 3/3 actions" in result.summary
    assert env.executor.last_outcome is not None
    assert env.executor.last_outcome.stopped_reason == "completed"


def test_execute_actions_stops_when_dialog_ends(make_executor) -> None:
    """DIALOG -> NONE: the conversation ended, do not press A again (re-open)."""
    env = make_executor(rgbas=[dialog_page_rgba(0), overworld_rgba(offset=8)])
    env.executor.execute(call("observe_game"))
    result = env.executor.execute(
        call("execute_actions", arguments={"actions": ["a", "a", "a"]})
    )
    assert result.ok is True
    assert result.data["start_event"] == "dialog"
    assert result.data["executed_actions"] == 1
    assert result.data["stopped_reason"] == "event"
    assert result.needs_llm is True


def test_execute_actions_stops_when_dialog_appears(make_executor) -> None:
    """NONE -> DIALOG: a new significant event appeared, stop the batch."""
    env = make_executor(rgbas=[overworld_rgba(), dialog_page_rgba(0), dialog_page_rgba(1)])
    result = env.executor.execute(
        call("execute_actions", arguments={"actions": ["a", "a", "a"]})
    )
    assert result.ok is True
    assert result.data["start_event"] == "none"
    assert result.data["executed_actions"] == 1
    assert result.data["stopped_reason"] == "event"
    assert result.needs_llm is True


def test_wait_frames_persistent_dialog_does_not_interrupt(make_executor) -> None:
    env = make_executor(rgbas=[dialog_page_rgba(0), dialog_page_rgba(1)])
    env.executor.execute(call("observe_game"))
    result = env.executor.execute(call("wait_frames", arguments={"frames": 12}))
    assert result.ok is True
    assert result.data["start_event"] == "dialog"
    assert result.needs_llm is False


def test_wait_frames_executes_and_updates_state(make_executor) -> None:
    env = make_executor(rgbas=[overworld_rgba(), overworld_rgba(offset=8)])
    result = env.executor.execute(call("wait_frames", arguments={"frames": 30}))
    assert result.ok is True
    assert env.emulator.waits == [30]
    assert env.state.action_batches == 1
    assert result.data["waited_frames"] == 30
    assert result.needs_llm is False


def test_wait_frames_dialog_sets_needs_llm(make_executor) -> None:
    env = make_executor(rgbas=[overworld_rgba(), dialog_rgba()])
    result = env.executor.execute(call("wait_frames", arguments={"frames": 12}))
    assert result.ok is True
    assert result.needs_llm is True


# --------------------------------------------------------------------------- #
# walk
# --------------------------------------------------------------------------- #

def test_walk_stops_when_blocked_and_requests_new_decision(make_executor) -> None:
    flat = [overworld_rgba() for _ in range(6)]
    env = make_executor(rgbas=flat)
    result = env.executor.execute(
        call("walk", arguments={"direction": "up", "max_steps": 5})
    )
    assert result.ok is True
    assert result.data["stopped_reason"] == "blocked"
    assert result.data["steps"] == 2
    assert result.needs_llm is True
    assert env.state.buttons_pressed == 2
    assert env.state.action_batches == 1
    assert env.executor.last_outcome is not None
    assert env.executor.last_outcome.stopped_reason == "blocked"


def test_walk_reaches_max_steps_without_blocking(make_executor) -> None:
    rgbas = [overworld_rgba(offset=step * 4) for step in range(6)]
    env = make_executor(rgbas=rgbas)
    result = env.executor.execute(
        call("walk", arguments={"direction": "down", "max_steps": 3})
    )
    assert result.ok is True
    assert result.data["stopped_reason"] == "max_steps"
    assert result.data["steps"] == 3
    assert result.needs_llm is False


def test_walk_stops_on_significant_event(make_executor) -> None:
    env = make_executor(rgbas=[overworld_rgba(), dialog_rgba()])
    result = env.executor.execute(
        call("walk", arguments={"direction": "up", "max_steps": 10})
    )
    assert result.ok is True
    assert result.data["stopped_reason"] == "event"
    assert result.needs_llm is True


def test_walk_through_persistent_dialog_pages_does_not_stop(make_executor) -> None:
    """A persisting DIALOG event while menu pages change must not interrupt walk."""
    rgbas = [dialog_page_rgba(index) for index in range(6)]
    env = make_executor(rgbas=rgbas)
    env.executor.execute(call("observe_game"))
    result = env.executor.execute(
        call("walk", arguments={"direction": "right", "max_steps": 4})
    )
    assert result.ok is True
    assert result.data["start_event"] == "dialog"
    assert result.data["steps"] == 4
    assert result.data["stopped_reason"] == "max_steps"
    assert result.needs_llm is False


# --------------------------------------------------------------------------- #
# memory / goal / checkpoint / history tools
# --------------------------------------------------------------------------- #

def test_set_goal_updates_memory_and_state(make_executor) -> None:
    env = make_executor()
    result = env.executor.execute(
        call("set_goal", arguments={"goal": "Find a way out", "reason": "no exits seen"})
    )
    assert result.ok is True
    assert env.memory.goal == {"text": "Find a way out", "reason": "no exits seen"}
    assert env.state.goal == "Find a way out"
    assert env.state.goal_reason == "no exits seen"
    assert "Find a way out" in result.summary
    assert any(entry.kind == "note" for entry in env.state.history)


def test_record_discovery_updates_memory_and_counters(make_executor) -> None:
    env = make_executor()
    result = env.executor.execute(
        call(
            "record_discovery",
            arguments={"text": "A sign says TOWN", "kind": "place", "place": "start"},
        )
    )
    assert result.ok is True
    assert env.memory.discoveries[0]["text"] == "A sign says TOWN"
    assert env.state.discoveries == 1
    assert result.data["entry"]["kind"] == "place"
    assert "discoveries" in result.summary


def test_inspect_memory_returns_context(make_executor) -> None:
    memory = FakeMemory()
    memory.context = "AGENT MEMORY\nGoal: explore"
    env = make_executor(memory=memory)
    result = env.executor.execute(
        call("inspect_memory", arguments={"query": "goal", "limit": 3})
    )
    assert result.ok is True
    assert memory.search_calls == [("goal", 3)]
    assert memory.rendered == 1
    assert "AGENT MEMORY" in result.summary
    assert result.data["matches"] == 1


def test_inspect_memory_failure_is_an_error_result(make_executor) -> None:
    env = make_executor(memory=BareMemory())
    result = env.executor.execute(call("inspect_memory"))
    assert result.ok is False and result.needs_llm is True
    assert "render_context" in (result.error or "")


def test_save_progress_calls_injected_callback(make_executor) -> None:
    env = make_executor()
    result = env.executor.execute(call("save_progress", arguments={"reason": "before cave"}))
    assert result.ok is True
    assert env.saved == ["before cave"]
    assert "checkpoint_" in result.summary
    assert any(entry.kind == "checkpoint" for entry in env.state.history)
    # the runtime callback owns the counter; the tool must not double count
    assert env.state.checkpoints == 0


def test_restore_checkpoint_calls_callback_and_invalidates_cache(make_executor) -> None:
    env = make_executor(rgbas=[overworld_rgba(), overworld_rgba(offset=8)])
    env.executor.execute(call("observe_game"))
    assert env.executor.last_frame is not None
    result = env.executor.execute(
        call("restore_checkpoint", arguments={"checkpoint": "latest"})
    )
    assert result.ok is True
    assert env.restored == ["latest"]
    assert env.executor.last_frame is None
    assert env.executor.last_analysis is None
    assert env.executor.last_outcome is None
    assert any(entry.kind == "checkpoint" for entry in env.state.history)
    bad = env.executor.execute(call("restore_checkpoint", arguments={"checkpoint": "bogus"}))
    assert bad.ok is False
    assert env.restored == ["latest"]


def test_get_recent_history_returns_state_entries(make_executor) -> None:
    env = make_executor()
    env.state.add_history("tool", "pressed a")
    env.state.add_history("event", "dialog opened")
    result = env.executor.execute(call("get_recent_history", arguments={"limit": 2}))
    assert result.ok is True
    assert result.data["limit"] == 2
    assert len(result.data["entries"]) == 2
    assert "dialog opened" in result.summary
    assert "pressed a" in result.summary


def test_long_tool_summaries_are_clipped(make_executor) -> None:
    memory = FakeMemory()
    memory.context = "M" * 4000
    cfg = AgentConfig()
    cfg.api.max_tool_result_chars = 300
    env = make_executor(memory=memory, cfg=cfg)
    result = env.executor.execute(call("inspect_memory"))
    assert result.ok is True
    assert len(result.summary) <= 300


