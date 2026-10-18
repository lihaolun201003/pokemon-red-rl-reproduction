"""Tool-schema and validating-executor tests for the battle tools."""

from __future__ import annotations

import pytest

from llm_agent.contracts import ModelReply, ToolCall, ToolValidationError
from llm_agent.contracts import ApiUsage
from llm_agent.pokemon import tools


def _reply(tool_calls=None, content: str = "") -> ModelReply:
    return ModelReply(
        content=content,
        tool_calls=list(tool_calls or []),
        usage=ApiUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="fake",
    )


def test_tool_schemas_are_complete_and_constrained() -> None:
    names = []
    for schema in tools.TOOL_SCHEMAS:
        assert schema.get("type") == "function"
        function = schema["function"]
        assert function["name"]
        assert function["description"]
        parameters = function.get("parameters") or {}
        assert parameters.get("type") == "object"
        names.append(function["name"])
    assert set(names) == {"battle_action", "observe", "record_lesson", "finish"}

    battle = tools.TOOL_SCHEMA_BY_NAME["battle_action"]["function"]["parameters"]
    assert set(battle["required"]) == {"action"}
    assert set(battle["properties"]["action"]["enum"]) == set(tools.BATTLE_ACTIONS)
    assert battle["properties"]["move_slot"]["maximum"] == tools.MAX_MOVE_SLOTS
    observation_props = set(battle["properties"]["visible_observations"]["properties"])
    assert observation_props == set(tools.VISIBLE_OBSERVATION_KEYS)
    assert battle["properties"]["visible_observations"]["additionalProperties"] is False
    finish = tools.TOOL_SCHEMA_BY_NAME["finish"]["function"]["parameters"]
    assert set(finish["properties"]["outcome"]["enum"]) == set(tools.FINISH_OUTCOMES)


def test_validate_fight_decision() -> None:
    decision = tools.validate_decision(
        {"action": "fight", "move_slot": 3, "reason": "third row", "hypothesis": "seems strong"}
    )
    assert decision.action == "fight"
    assert decision.move_slot == 3
    assert decision.reason == "third row"
    assert decision.hypothesis == "seems strong"
    assert decision.raw_arguments["move_slot"] == 3
    assert decision.slot == 3


def test_validate_fight_decision_defaults_to_first_slot() -> None:
    decision = tools.validate_decision({"action": "fight"})
    assert decision.move_slot == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "dance"},
        {"action": "fight", "move_slot": 0},
        {"action": "fight", "move_slot": 5},
        {"action": "fight", "move_slot": "lots"},
        {"action": "switch", "party_slot": 7},
        {"action": "catch", "item_slot": 9},
        {"tool": "battle_action"},
        {},
    ],
)
def test_invalid_decisions_are_rejected(arguments: dict) -> None:
    with pytest.raises(ToolValidationError):
        tools.validate_decision(arguments)


def test_unknown_visible_observation_fields_are_dropped() -> None:
    decision = tools.validate_decision(
        {
            "action": "fight",
            "visible_observations": {
                "move_name": "a readable label",
                "enemy_hp": "17/42",  # hidden value: must never be accepted
                "hidden_move": "something unseen",
            },
        }
    )
    assert decision.visible_observations == {"move_name": "a readable label"}


def test_finish_outcome_validation() -> None:
    decision = tools.validate_decision({"action": "finish", "outcome": "ran"})
    assert decision.outcome == "ran"
    assert tools.validate_decision({"action": "finish"}).outcome == "unknown"
    with pytest.raises(ToolValidationError):
        tools.validate_decision({"action": "finish", "outcome": "victory"})


def test_lesson_validation() -> None:
    lesson = tools.validate_lesson(
        {"content": "a visible fact", "verification": "verified", "category": "battle_lesson"}
    )
    assert lesson.verification == "verified"
    assert lesson.confidence == pytest.approx(0.8)
    inferred = tools.validate_lesson({"content": "a guess", "verification": "inferred"})
    assert inferred.confidence == pytest.approx(0.5)
    with pytest.raises(ToolValidationError):
        tools.validate_lesson({"content": "x", "verification": "certain"})
    with pytest.raises(ToolValidationError):
        tools.validate_lesson({"content": "", "verification": "verified"})
    with pytest.raises(ToolValidationError):
        tools.validate_lesson(
            {"content": "x", "verification": "verified", "category": "not_a_category"}
        )


def test_executor_executes_each_tool() -> None:
    executor = tools.BattleToolExecutor()

    ok = executor.execute(ToolCall(id="1", name="battle_action", arguments={"action": "run"}))
    assert ok["ok"] is True
    assert ok["decision"].action == "run"

    observed = executor.execute(ToolCall(id="2", name="observe", arguments={"reason": "look"}))
    assert observed["ok"] is True
    assert observed["decision"].action == "observe"

    finished = executor.execute(
        ToolCall(id="3", name="finish", arguments={"outcome": "win"})
    )
    assert finished["ok"] is True
    assert finished["decision"].action == "finish"
    assert finished["decision"].outcome == "win"

    lesson = executor.execute(
        ToolCall(
            id="4",
            name="record_lesson",
            arguments={"content": "a visible fact", "verification": "verified"},
        )
    )
    assert lesson["ok"] is True
    assert lesson["lesson"].verification == "verified"

    bad = executor.execute(ToolCall(id="5", name="battle_action", arguments={"action": "nope"}))
    assert bad["ok"] is False
    assert "unknown battle action" in bad["error"]

    unknown_tool = executor.execute(ToolCall(id="6", name="teleport", arguments={}))
    assert unknown_tool["ok"] is False
    assert "unknown tool" in unknown_tool["error"]


def test_parse_reply_extracts_one_decision_and_lessons() -> None:
    executor = tools.BattleToolExecutor()
    reply = _reply(
        [
            ToolCall(id="a", name="battle_action", arguments={"action": "fight", "move_slot": 2}),
            ToolCall(
                id="b",
                name="record_lesson",
                arguments={"content": "a visible fact", "verification": "verified"},
            ),
        ]
    )

    parsed = executor.parse_reply(reply)

    assert parsed.ok is True
    assert parsed.decision.action == "fight"
    assert parsed.decision.move_slot == 2
    assert len(parsed.lessons) == 1
    assert len(parsed.tool_results) == 2
    assert all(result["ok"] for result in parsed.tool_results)


def test_parse_reply_rejects_second_action_and_invalid_calls() -> None:
    executor = tools.BattleToolExecutor()
    reply = _reply(
        [
            ToolCall(id="a", name="battle_action", arguments={"action": "fight", "move_slot": 1}),
            ToolCall(id="b", name="battle_action", arguments={"action": "run"}),
            ToolCall(id="c", name="record_lesson", arguments={"content": "x", "verification": "who"}),
        ]
    )

    parsed = executor.parse_reply(reply)

    assert parsed.decision.action == "fight"
    assert any("one battle action" in error for error in parsed.errors)
    assert any("verification" in error for error in parsed.errors)


def test_parse_reply_content_json_fallback() -> None:
    executor = tools.BattleToolExecutor()
    reply = _reply(content='{"action": "fight", "move_slot": 4}')

    parsed = executor.parse_reply(reply)

    assert parsed.ok is True
    assert parsed.decision.action == "fight"
    assert parsed.decision.move_slot == 4
    assert parsed.decision.source == "content_json"


def test_parse_reply_content_json_wrapped_in_code_fence() -> None:
    executor = tools.BattleToolExecutor()
    reply = _reply(content='```json\n{"action": "observe"}\n```')

    parsed = executor.parse_reply(reply)

    assert parsed.ok is True
    assert parsed.decision.action == "observe"


def test_parse_reply_without_any_decision_reports_errors() -> None:
    executor = tools.BattleToolExecutor()
    parsed = executor.parse_reply(_reply(content="I am thinking about it."))
    assert parsed.ok is False
    assert parsed.errors

    empty = executor.parse_reply(None)
    assert empty.ok is False
    assert empty.errors


def test_parse_reply_limits_record_lessons_per_turn() -> None:
    executor = tools.BattleToolExecutor(max_lessons_per_turn=1)
    reply = _reply(
        [
            ToolCall(id="a", name="battle_action", arguments={"action": "observe"}),
            ToolCall(id="b", name="record_lesson", arguments={"content": "one", "verification": "verified"}),
            ToolCall(id="c", name="record_lesson", arguments={"content": "two", "verification": "verified"}),
        ]
    )

    parsed = executor.parse_reply(reply)

    assert len(parsed.lessons) == 1
    assert any("at most" in error for error in parsed.errors)


def test_lessons_categories_are_a_subset_of_the_frozen_catalogue() -> None:
    from llm_agent.multi_agent.contracts import KNOWLEDGE_CATEGORIES

    assert set(tools.LESSON_CATEGORIES).issubset(set(KNOWLEDGE_CATEGORIES))
