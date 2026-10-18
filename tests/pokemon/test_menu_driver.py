"""Deterministic menu-driver tests: LLM decision -> real, verified button presses.

The decision always comes from the scripted model router; these tests pin down
that the *same* decision always yields the same deterministic press sequence, and
that the driver refuses to press anything when no battle menu is visible.
"""

from __future__ import annotations

import pytest

from llm_agent.multi_agent.agent_context import ActionPrimitives
from llm_agent.multi_agent.contracts import AgentRole
from llm_agent.pokemon import tools
from llm_agent.pokemon.agent import (
    BATTLE_MENU_MAIN,
    BATTLE_MENU_UNKNOWN,
    BattleMenuDriver,
    battle_menu_layout,
)
from tests.pokemon.conftest import (
    FakeBattleEmulator,
    FakeRouter,
    battle_main_frame,
    make_frame,
)


@pytest.mark.parametrize(
    ("decision", "emulator_kwargs", "expected_presses", "expected_status", "max_turns"),
    [
        (
            {"action": "fight", "move_slot": 2},
            {"end_after_fights": 1},
            ["a", "down", "a"],
            "battle_ended",
            3,
        ),
        (
            {"action": "fight", "move_slot": 4},
            {"end_after_fights": None},
            ["a", "down", "down", "down", "a"],
            "budget",
            1,
        ),
        (
            {"action": "run"},
            {"end_after_fights": None},
            ["down", "right", "a"],
            "battle_ended",
            3,
        ),
        (
            {"action": "catch", "item_slot": 2},
            {"end_after_fights": None, "item_use_catches": True},
            ["down", "a", "down", "a"],
            "battle_ended",
            3,
        ),
        (
            {"action": "item", "item_slot": 1},
            {"end_after_fights": None, "item_use_catches": False},
            ["down", "a", "a"],
            "budget",
            1,
        ),
        (
            {"action": "switch", "party_slot": 2},
            {"end_after_fights": None},
            ["right", "a", "down", "a", "a"],
            "budget",
            1,
        ),
    ],
)
def test_decision_maps_to_deterministic_presses(
    make_agent,
    decision: dict,
    emulator_kwargs: dict,
    expected_presses: list[str],
    expected_status: str,
    max_turns: int,
) -> None:
    emulator = FakeBattleEmulator(**emulator_kwargs)
    router = FakeRouter([dict(decision)])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=max_turns)

    assert result["status"] == expected_status
    assert len(result["actions"]) == 1
    action = result["actions"][0]
    assert action["decision"]["action"] == decision["action"]
    assert action["presses"] == expected_presses
    assert action["verified"] is True
    # The real emulator received exactly those presses first.
    assert emulator.presses[: len(expected_presses)] == expected_presses


def test_observe_decision_presses_nothing(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=None)
    router = FakeRouter([{"action": "observe"}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=1)

    assert result["actions"][0]["decision"]["action"] == "observe"
    assert result["actions"][0]["presses"] == []
    assert result["actions"][0]["verified"] is True
    assert emulator.presses == []
    assert emulator.waits >= 1


def test_driver_returns_control_without_pressing_when_menu_is_not_visible() -> None:
    driver = BattleMenuDriver()
    emulator = FakeBattleEmulator(start="battle_blank", end_after_fights=None)
    primitives = ActionPrimitives(emulator, agent=AgentRole.POKEMON)
    decision = tools.validate_decision({"action": "fight", "move_slot": 2})
    layout = battle_menu_layout(make_frame(battle_main_frame()))
    assert layout == BATTLE_MENU_MAIN

    # Force the emulator into a battle screen without a readable menu; the
    # driver must re-observe and refuse to press blindly.
    emulator.force_state("battle_blank")
    result = driver.execute(decision, primitives, BATTLE_MENU_MAIN)

    assert result["verified"] is False
    assert result["presses"] == []
    assert emulator.presses == []
    assert "no visible main battle menu" in result["reason"].lower()
    assert result["layout"] == BATTLE_MENU_UNKNOWN


def test_driver_verifies_every_press_with_a_pixel_diff() -> None:
    driver = BattleMenuDriver()
    emulator = FakeBattleEmulator(end_after_fights=None)
    primitives = ActionPrimitives(emulator, agent=AgentRole.POKEMON)
    decision = tools.validate_decision({"action": "run"})

    result = driver.execute(decision, primitives, BATTLE_MENU_MAIN)

    assert result["verified"] is True
    assert result["presses"] == ["down", "right", "a"]
    assert len(result["press_log"]) == len(result["presses"])
    for entry in result["press_log"]:
        assert entry["verified"] is True
        assert entry["changed_ratio"] > 0.0


def test_driver_marks_presses_as_unverified_when_screen_does_not_change() -> None:
    driver = BattleMenuDriver()
    emulator = FakeBattleEmulator(end_after_fights=None, ignore_presses=True)
    primitives = ActionPrimitives(emulator, agent=AgentRole.POKEMON)
    decision = tools.validate_decision({"action": "fight", "move_slot": 2})

    result = driver.execute(decision, primitives, BATTLE_MENU_MAIN)

    assert result["verified"] is False
    assert result["presses"] == ["a"]  # stopped after the first unverified press
    assert "did not visibly change" in result["reason"].lower()
    assert all(entry["verified"] is False for entry in result["press_log"])


def test_repeated_unverified_drives_interrupt_the_battle(make_agent) -> None:
    """Repeated pixel-verified failures return control instead of looping."""
    emulator = FakeBattleEmulator(end_after_fights=None, ignore_presses=True)
    router = FakeRouter([{"action": "fight", "move_slot": 1}] * 5)
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=6)

    assert result["status"] == "interrupted"
    assert result["turns"] == 3
    assert all(action["verified"] is False for action in result["actions"])
