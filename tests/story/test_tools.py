"""Tool-schema, validation, permission-isolation and executor tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from llm_agent.config import MemoryConfig
from llm_agent.contracts import ToolCall, ToolResult
from llm_agent.memory import AgentMemory
from llm_agent.multi_agent.agent_context import ActionPrimitives
from llm_agent.multi_agent.contracts import AgentRole, VerificationStatus
from llm_agent.story.memory import StoryMemory
from llm_agent.story.tools import (
    MAX_PROBE_PRESSES,
    PROBE_DIRECTIONS,
    STORY_TOOL_NAMES,
    TOOL_SCHEMAS,
    StoryTools,
    ToolError,
    acquire_dialog,
    is_dialog_screen,
    validate_call,
)
from llm_agent.observation import analyze

from tests.story.conftest import (
    FakeArbiter,
    FakeEmulator,
    FakeSharedMemory,
    dialog_rgba,
    make_frame,
    menu_rgba,
    overworld_rgba,
)


def _actions(emulator: FakeEmulator, *, screen_dir=None, save=False) -> ActionPrimitives:
    return ActionPrimitives(
        emulator,
        agent=AgentRole.STORY,
        screen_dir=screen_dir,
        save_screenshots=save,
        image_scale=1,
    )


def _story_memory(tmp_path) -> StoryMemory:
    root = tmp_path / "memory"
    root.mkdir(parents=True, exist_ok=True)
    memory = AgentMemory(replace(MemoryConfig(root=str(root)), max_context_chars=2000))
    return StoryMemory(root, agent_memory=memory, shared_memory=FakeSharedMemory())


def _tools(tmp_path, emulator: FakeEmulator, *, arbiter=None) -> StoryTools:
    tools = StoryTools(
        memory=_story_memory(tmp_path),
        actions=_actions(emulator),
        arbiter=arbiter if arbiter is not None else FakeArbiter(),
    )
    tools.begin()
    return tools


# --------------------------------------------------------------------------- #
# Schemas / classification
# --------------------------------------------------------------------------- #

def test_tool_list_matches_frozen_whitelist_and_schemas() -> None:
    assert STORY_TOOL_NAMES == (
        "observe",
        "advance_dialog",
        "choose_option",
        "record_fact",
        "propose_task",
        "finish",
    )
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert names == list(STORY_TOOL_NAMES)
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        assert function["description"]
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False


def test_synthetic_frames_are_classified_as_expected() -> None:
    dialog = analyze(make_frame(dialog_rgba(0), 1), None)
    overworld = analyze(make_frame(overworld_rgba(0), 2), None)
    menu = analyze(make_frame(menu_rgba(), 3), None)
    assert is_dialog_screen(dialog) is True
    assert is_dialog_screen(overworld) is False
    assert is_dialog_screen(menu) is True
    assert is_dialog_screen(None) is False


# --------------------------------------------------------------------------- #
# Permission isolation: unknown tools never press anything
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "forbidden",
    ["press_button", "execute_actions", "walk", "navigate_to", "save_progress", "run_batch"],
)
def test_non_whitelisted_tools_are_rejected_without_press(forbidden: str, tmp_path) -> None:
    emulator = FakeEmulator(initial_screen="dialog")
    tools = _tools(tmp_path, emulator)
    with pytest.raises(ToolError) as excinfo:
        tools.execute(ToolCall(id="c1", name=forbidden, arguments={}))
    assert "not allowed" in str(excinfo.value)
    assert emulator.presses == []


def test_unknown_argument_is_rejected() -> None:
    with pytest.raises(ToolError):
        validate_call("observe", {"secret": True})
    with pytest.raises(ToolError):
        validate_call("advance_dialog", {"button": "a", "times": 1, "frames": 99})


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("advance_dialog", {}),
        ("advance_dialog", {"button": "x"}),
        ("advance_dialog", {"button": "a", "times": 9}),
        ("advance_dialog", {"button": "a", "times": True}),
        ("choose_option", {"direction": "northwest"}),
        ("choose_option", {"direction": "up", "presses": -1}),
        ("record_fact", {"category": "story_clue", "content": "x", "verification_status": "fact"}),
        ("record_fact", {"category": "banana", "content": "x", "verification_status": "inferred"}),
        ("propose_task", {"owner_agent": "director", "task_type": "talk", "description": "x"}),
        ("propose_task", {"owner_agent": "navigation", "task_type": "sleep", "description": "x"}),
        ("finish", {}),
        ("finish", {"summary": "done", "outcome": "probably"}),
        ("observe", ["not", "a", "dict"]),
    ],
)
def test_invalid_arguments_are_rejected(name: str, arguments) -> None:
    with pytest.raises(ToolError):
        validate_call(name, arguments)


def test_validation_normalises_defaults() -> None:
    assert validate_call("advance_dialog", {"button": "A"}) == {"button": "a", "times": 1}
    assert validate_call("choose_option", {"direction": "DOWN"}) == {
        "direction": "down",
        "presses": 1,
        "confirm": True,
    }
    normalised = validate_call(
        "record_fact",
        {"category": "NPC", "content": " a person waved ", "verification_status": "INFERRED"},
    )
    assert normalised["category"] == "npc"
    assert normalised["content"] == "a person waved"
    assert normalised["verification_status"] == "inferred"


# --------------------------------------------------------------------------- #
# Real executions
# --------------------------------------------------------------------------- #

def test_advance_dialog_presses_real_buttons(tmp_path) -> None:
    emulator = FakeEmulator(initial_screen="dialog", dialog_lines=5)
    tools = _tools(tmp_path, emulator)
    result = tools.execute(ToolCall(id="c1", name="advance_dialog", arguments={"button": "a", "times": 2}))
    assert isinstance(result, ToolResult) and result.ok
    assert [button for button, _frames in emulator.presses] == ["a", "a"]
    assert tools.presses == ["a", "a"]


def test_choose_option_presses_cursor_and_confirm(tmp_path) -> None:
    emulator = FakeEmulator(initial_screen="dialog", dialog_lines=5)
    tools = _tools(tmp_path, emulator)
    result = tools.execute(
        ToolCall(
            id="c2",
            name="choose_option",
            arguments={"direction": "down", "presses": 2, "confirm": True},
        )
    )
    assert result.ok
    assert [button for button, _frames in emulator.presses] == ["down", "down", "a"]


def test_press_cap_is_per_tool_call_and_enforced(tmp_path) -> None:
    emulator = FakeEmulator(initial_screen="dialog", dialog_lines=9)
    memory = _story_memory(tmp_path)
    tools = StoryTools(
        memory=memory,
        actions=_actions(emulator),
        arbiter=FakeArbiter(),
        max_presses_per_call=2,
    )
    tools.begin()
    # Two presses are fine ...
    ok = tools.execute(
        ToolCall(id="c1", name="advance_dialog", arguments={"button": "a", "times": 2})
    )
    assert ok.ok
    assert len(emulator.presses) == 2
    # ... but the same call is a separate budget, so one more call of two works.
    ok = tools.execute(
        ToolCall(id="c2", name="advance_dialog", arguments={"button": "a", "times": 2})
    )
    assert ok.ok
    assert len(emulator.presses) == 4
    # A single call asking for three presses with a cap of two raises before the
    # third press and leaves the game untouched beyond the cap.
    before = len(emulator.presses)
    with pytest.raises(ToolError):
        tools.execute(ToolCall(id="c3", name="advance_dialog", arguments={"button": "a", "times": 3}))
    assert len(emulator.presses) - before <= 2


def test_record_fact_writes_memory_and_shared_memory(tmp_path) -> None:
    emulator = FakeEmulator(initial_screen="dialog")
    shared = FakeSharedMemory()
    memory = StoryMemory(
        tmp_path / "memory",
        agent_memory=AgentMemory(
            replace(MemoryConfig(root=str(tmp_path / "memory")), max_context_chars=2000)
        ),
        shared_memory=shared,
    )
    tools = StoryTools(memory=memory, actions=_actions(emulator), arbiter=FakeArbiter())
    tools.begin()
    result = tools.execute(
        ToolCall(
            id="c1",
            name="record_fact",
            arguments={
                "category": "npc",
                "content": "A resident asked whether the player was lost.",
                "verification_status": "verified",
                "confidence": 0.9,
                "npc_name": "the resident",
                "dialogue_text": "Are you lost, young one?",
            },
        ),
        evidence_ref="screens/shot_0001.png",
        frame_index=10,
    )
    assert result.ok
    knowledge = result.data["knowledge"]
    assert knowledge["category"] == "npc"
    assert knowledge["verification_status"] == "verified"
    assert knowledge["evidence_refs"] == ["screens/shot_0001.png"]
    assert memory.knowledge()[0].verification_status is VerificationStatus.VERIFIED
    assert memory.npcs()[0]["name"] == "the resident"
    assert shared.entries and shared.entries[0].verification_status is VerificationStatus.VERIFIED
    assert shared.entries[0].source_agent is AgentRole.STORY


def test_record_fact_downgrades_unfounded_verified_claim(tmp_path) -> None:
    emulator = FakeEmulator(initial_screen="dialog")
    tools = _tools(tmp_path, emulator)
    result = tools.execute(
        ToolCall(
            id="c1",
            name="record_fact",
            arguments={
                "category": "story_clue",
                "content": "Something important is hidden here.",
                "verification_status": "verified",
            },
        )
    )
    assert result.ok
    assert result.data["knowledge"]["verification_status"] == "inferred"
    assert result.data["downgraded"] is True
    assert "INFERRED" in result.summary


def test_record_fact_inference_stays_inferred(tmp_path) -> None:
    tools = _tools(tmp_path, FakeEmulator(initial_screen="dialog"))
    result = tools.execute(
        ToolCall(
            id="c1",
            name="record_fact",
            arguments={
                "category": "story_clue",
                "content": "The text probably refers to a task someone wants done.",
                "verification_status": "inferred",
            },
        )
    )
    assert result.data["knowledge"]["verification_status"] == "inferred"


def test_propose_task_dedupes(tmp_path) -> None:
    tools = _tools(tmp_path, FakeEmulator(initial_screen="dialog"))
    arguments = {
        "owner_agent": "navigation",
        "task_type": "talk",
        "description": "Talk again with the person near the table",
        "priority": "low",
        "payload": {"target": {"kind": "described", "description": "the person near the table"}},
    }
    first = tools.execute(ToolCall(id="c1", name="propose_task", arguments=arguments))
    second = tools.execute(ToolCall(id="c2", name="propose_task", arguments=arguments))
    assert first.ok and first.data["created"] is True
    assert second.ok and second.data["created"] is False
    assert tools.memory.proposals() == [
        {
            "owner_agent": "navigation",
            "task_type": "talk",
            "description": "Talk again with the person near the table",
            "priority": "low",
            "payload": {
                "target": {"kind": "described", "description": "the person near the table"}
            },
        }
    ]


def test_tools_refuse_to_press_without_lease(tmp_path) -> None:
    emulator = FakeEmulator(initial_screen="dialog")
    tools = _tools(tmp_path, emulator, arbiter=FakeArbiter(held=False))
    from llm_agent.multi_agent.contracts import ControlDeniedError

    with pytest.raises(ControlDeniedError):
        tools.execute(ToolCall(id="c1", name="advance_dialog", arguments={"button": "a"}))
    assert emulator.presses == []


# --------------------------------------------------------------------------- #
# Interaction probe
# --------------------------------------------------------------------------- #

def test_acquire_dialog_probe_finds_dialog_on_second_press() -> None:
    emulator = FakeEmulator(
        initial_screen="overworld", dialog_after_presses=2, dialog_lines=4
    )
    actions = _actions(emulator)
    result = acquire_dialog(actions, arbiter=FakeArbiter())
    assert result["status"] == "dialog"
    assert len(result["presses"]) <= MAX_PROBE_PRESSES
    assert result["frame"].frame_index > 0
    assert result["analysis"].event.value == "dialog"
    assert result["evidence_ref"]


def test_acquire_dialog_probe_gives_up_after_bounded_presses() -> None:
    emulator = FakeEmulator(initial_screen="overworld", dialog_after_presses=None)
    actions = _actions(emulator)
    probes = 0
    for direction in PROBE_DIRECTIONS:
        probes += 1
    result = acquire_dialog(actions, arbiter=FakeArbiter())
    assert result["status"] == "no_dialog"
    assert 0 < len(result["presses"]) <= MAX_PROBE_PRESSES
    assert len(emulator.presses) == len(result["presses"])
    assert "no dialogue" in result["reason"].lower()
    assert probes == 4


def test_acquire_dialog_probe_closes_menu_and_keeps_going() -> None:
    emulator = FakeEmulator(
        initial_screen="overworld", menu_after_presses=1, menu_close_presses=1
    )
    actions = _actions(emulator)
    result = acquire_dialog(actions, arbiter=FakeArbiter())
    assert result["status"] == "no_dialog"
    assert "b" in result["presses"]
    assert len(result["presses"]) <= MAX_PROBE_PRESSES


def test_acquire_dialog_probe_returns_interrupted_when_menu_sticks() -> None:
    emulator = FakeEmulator(
        initial_screen="overworld", menu_after_presses=1, menu_close_presses=99
    )
    actions = _actions(emulator)
    result = acquire_dialog(actions, arbiter=FakeArbiter())
    assert result["status"] == "interrupted"
    assert "menu" in result["reason"].lower()
    assert result["presses"][:3] == ["a", "b", "b"]


def test_acquire_dialog_refuses_to_press_without_lease() -> None:
    emulator = FakeEmulator(initial_screen="overworld")
    result = acquire_dialog(_actions(emulator), arbiter=FakeArbiter(held=False))
    assert result["status"] == "control_denied"
    assert emulator.presses == []
