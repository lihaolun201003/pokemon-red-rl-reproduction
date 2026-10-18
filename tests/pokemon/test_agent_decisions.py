"""End-to-end battle decision tests: LLM per turn, evidence, lessons, outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from llm_agent.multi_agent.contracts import (
    KNOWLEDGE_CATEGORIES,
    AgentRole,
    KnowledgeEntry,
    VerificationStatus,
)
from tests.pokemon.conftest import (
    FakeBattleEmulator,
    FakeBus,
    FakeRouter,
    FakeSharedMemory,
    RecordOnlySharedMemory,
)


def test_fight_decision_is_logged_with_raw_args_presses_and_verification(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 2, "reason": "use slot two"}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=4)

    assert result["status"] == "battle_ended"
    action = result["actions"][0]
    assert action["turn"] == 1
    assert action["decision"]["action"] == "fight"
    assert action["decision"]["move_slot"] == 2
    assert action["decision"]["source"] == "tool_call"
    assert action["decision"]["raw"] == {"action": "fight", "move_slot": 2, "reason": "use slot two"}
    assert action["presses"] == ["a", "down", "a"]
    assert action["verified"] is True
    assert "layout=main" in action["context"]


def test_every_turn_is_decided_by_the_model(make_agent, endless_emulator) -> None:
    """Two turns -> two different model decisions -> two different press paths."""
    router = FakeRouter(
        [
            {"action": "fight", "move_slot": 1},
            {"action": "fight", "move_slot": 3},
        ]
    )
    agent = make_agent(router)

    result = agent.handle_battle(emulator=endless_emulator, max_turns=2)

    assert result["status"] == "budget"
    assert result["turns"] == 2
    assert result["api_calls"] == 2
    assert len(result["actions"]) == 2
    first, second = result["actions"]
    assert first["decision"]["move_slot"] == 1
    assert second["decision"]["move_slot"] == 3
    assert first["presses"] == ["a", "a"]
    assert second["presses"] == ["a", "down", "down", "a"]
    assert router.images_sent() == 2  # a real screenshot per turn
    assert all(call["tools"] for call in router.calls)  # tool schemas were sent


def test_system_prompt_and_tools_are_sent_with_each_turn(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    agent.handle_battle(emulator=emulator, max_turns=3)

    first_call = router.calls[0]
    system_message = first_call["messages"][0]
    assert system_message["role"] == "system"
    assert "screenshot" in system_message["content"].lower()
    user_message = first_call["messages"][-1]
    assert isinstance(user_message["content"], list)
    assert any(block.get("type") == "image_url" for block in user_message["content"])
    names = {schema["function"]["name"] for schema in first_call["tools"]}
    assert names == {"battle_action", "observe", "record_lesson", "finish"}


def test_run_action_reports_inferred_outcome(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=None, run_succeeds=True)
    router = FakeRouter([{"action": "run"}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert result["outcome"] == "ran"
    assert result["actions"][0]["presses"] == ["down", "right", "a"]
    outcome_lessons = [
        lesson for lesson in result["lessons"] if "Battle finished" in lesson["content"]
    ]
    assert outcome_lessons
    assert outcome_lessons[0]["verification_status"] == "inferred"


def test_catch_action_reports_inferred_outcome(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=None, item_use_catches=True)
    router = FakeRouter([{"action": "catch", "item_slot": 2}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert result["outcome"] == "caught"
    assert result["actions"][0]["presses"] == ["down", "a", "down", "a"]


def test_outcome_can_be_read_from_the_post_battle_screenshot(make_agent) -> None:
    """The agent asks the model once to read the visible result text."""
    emulator = FakeBattleEmulator(end_after_fights=None, run_succeeds=True)
    router = FakeRouter(
        [
            {"action": "run"},
            {
                "tool": "finish",
                "outcome": "ran",
                "visible_text": "Got away safely!",
            },
        ]
    )
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert result["outcome"] == "ran"
    assert result["api_calls"] == 2  # one decision + one resolution read
    outcome_lessons = [
        lesson for lesson in result["lessons"] if "Battle finished" in lesson["content"]
    ]
    assert outcome_lessons
    assert outcome_lessons[0]["verification_status"] == "verified"
    assert "Got away safely!" in outcome_lessons[0]["content"]
    # The model saw the real resolution screenshot(s) in the extra read call.
    assert router.images_sent() >= 2
    read_messages = router.calls[-1]["messages"]
    assert any(
        block.get("type") == "image_url"
        for block in read_messages[-1]["content"]
        if isinstance(block, dict)
    )


def test_visible_observations_are_verified_and_hypotheses_are_inferred(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter(
        [
            {
                "action": "fight",
                "move_slot": 2,
                "visible_observations": {
                    "move_name": "visible move label",
                    "damage_text": "visible damage text",
                    "battle_text": "visible battle message",
                    "enemy_hp_bar": "about a third left",
                },
                "hypothesis": "this move seems effective against this opponent",
            }
        ]
    )
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=4)

    verification = {lesson["verification_status"] for lesson in result["lessons"]}
    assert "verified" in verification
    assert "inferred" in verification
    contents = "\n".join(lesson["content"] for lesson in result["lessons"])
    assert "visible move label" in contents
    assert "visible damage text" in contents
    assert "visible battle message" in contents
    assert "this move seems effective" in contents
    inferred = [l for l in result["lessons"] if l["verification_status"] == "inferred"]
    assert any("this move seems effective" in lesson["content"] for lesson in inferred)
    for lesson in result["lessons"]:
        assert lesson["confidence"] >= 0.0
        assert isinstance(lesson["evidence_refs"], list)


def test_lessons_are_forwarded_to_shared_memory(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    shared = FakeSharedMemory()
    router = FakeRouter(
        [
            {
                "action": "fight",
                "move_slot": 1,
                "visible_observations": {"battle_text": "visible battle message"},
                "hypothesis": "a guess that must stay labelled as inferred",
            }
        ]
    )
    agent = make_agent(router, shared_memory=shared)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert shared.entries, "battle lessons must reach the shared world memory"
    assert all(isinstance(entry, KnowledgeEntry) for entry in shared.entries)
    for entry in shared.entries:
        assert entry.category in KNOWLEDGE_CATEGORIES
        assert entry.source_agent is AgentRole.POKEMON
        assert entry.content
        assert 0.0 <= entry.confidence <= 1.0
    statuses = {entry.verification_status for entry in shared.entries}
    assert VerificationStatus.VERIFIED in statuses
    assert VerificationStatus.INFERRED in statuses
    assert all(lesson["knowledge_id"] for lesson in result["lessons"])


def test_shared_memory_duck_typing_uses_record_method(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    shared = RecordOnlySharedMemory()
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router, shared_memory=shared)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert shared.entries, "a shared memory exposing only record() must still work"


def test_record_lesson_tool_call_is_stored_as_verified(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    shared = FakeSharedMemory()
    router = FakeRouter(
        [
            {
                "action": "fight",
                "move_slot": 1,
                "lessons": [
                    {
                        "content": "the visible move list has four rows",
                        "verification": "verified",
                        "category": "battle_lesson",
                    }
                ],
            }
        ]
    )
    agent = make_agent(router, shared_memory=shared)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert any("four rows" in lesson["content"] for lesson in result["lessons"])
    verified = [
        lesson
        for lesson in result["lessons"]
        if "four rows" in lesson["content"] and lesson["verification_status"] == "verified"
    ]
    assert verified


def test_content_json_decision_fallback(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"tool": "content", "action": "fight", "move_slot": 2}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    action = result["actions"][0]
    assert action["decision"]["source"] == "content_json"
    assert action["presses"] == ["a", "down", "a"]


class _VanishingBattle(FakeBattleEmulator):
    """Battle screen that clears right after the model's finish decision."""

    def __init__(self) -> None:
        super().__init__(end_after_fights=None)
        self._observes = 0

    def observe(self):
        self._observes += 1
        # Observations: 1 = battle detection, 2 = intro-menu wait, 3 = the turn's
        # decision point, 4+ = the post-finish resolution (now clear).
        if self._observes >= 4 and self.state not in ("overworld",):
            self.state = "overworld"
            self._last = self._render_current()
        return super().observe()


def test_finish_decision_accepted_when_screen_clears(make_agent) -> None:
    emulator = _VanishingBattle()
    router = FakeRouter(
        [
            {
                "tool": "finish",
                "outcome": "win",
                "visible_observations": {"battle_text": "visible victory text"},
            }
        ]
    )
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=2)

    assert result["status"] == "battle_ended"
    assert result["outcome"] == "win"
    assert len(result["actions"]) == 1
    assert result["actions"][0]["decision"]["action"] == "finish"
    assert result["actions"][0]["verified"] is True
    outcome_lessons = [
        lesson for lesson in result["lessons"] if "Battle finished" in lesson["content"]
    ]
    assert outcome_lessons and outcome_lessons[0]["verification_status"] == "verified"


def test_evidence_screens_are_real_files(make_agent, screen_dir) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    screens = result["evidence"]["screens"]
    assert screens
    for path in screens:
        assert Path(path).is_file(), path
    assert any(str(Path(screen_dir)) in path for path in screens)
    assert result["evidence"]["frames"]


def test_bus_receives_battle_lifecycle_events(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    bus = FakeBus()
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router, bus=bus)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    types = [message.event_type.value for message in bus.published]
    assert types[0] == "battle_started"
    assert types[-1] == "battle_ended"
    ended = bus.published[-1]
    assert ended.payload.get("outcome") in ("win", "loss", "caught", "ran", "unknown")
    assert ended.payload.get("turns") == 1
    assert ended.target_agent is AgentRole.DIRECTOR


def test_returns_full_frozen_result_shape(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    for key in (
        "status",
        "outcome",
        "turns",
        "api_calls",
        "cost_usd",
        "actions",
        "lessons",
        "evidence",
    ):
        assert key in result, key
    assert result["status"] in (
        "battle_ended",
        "interrupted",
        "failed",
        "budget",
        "control_denied",
    )
    assert result["outcome"] in ("win", "loss", "caught", "ran", "unknown")
    assert isinstance(result["turns"], int)
    assert isinstance(result["api_calls"], int)
    assert result["cost_usd"] > 0.0  # token usage was priced
    assert isinstance(result["evidence"], dict)
    assert set(result["evidence"]) == {"screens", "frames"}
    # Director accounting extras: real presses are counted for the run totals.
    assert result["presses"] == len(emulator.presses) > 0
    assert result["buttons"] == result["presses"]
    assert result["denied"] == 0
    assert result["error"] == ""


def test_director_facing_attributes_exist(make_agent) -> None:
    """The director snapshots ``system_prompt``/``memory`` and calls status()."""
    agent = make_agent(FakeRouter())
    assert isinstance(agent.system_prompt, str) and agent.system_prompt
    assert agent.memory is not None
    assert hasattr(agent.memory, "render_context")
    assert hasattr(agent.memory, "snapshot")
    for key in ("battles_handled", "last_outcome", "paused"):
        assert key in agent.status()
