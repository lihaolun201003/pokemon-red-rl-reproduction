"""Frozen interface tests: handles / on_event / status / pause / recovery hint."""

from __future__ import annotations

from llm_agent.multi_agent.contracts import (
    AgentMessage,
    AgentRole,
    EventType,
    Task,
    TaskType,
)
from llm_agent.pokemon import prompts
from tests.pokemon.conftest import FakeBattleEmulator, FakeRouter


def test_handles_returns_the_battle_context() -> None:
    from llm_agent.pokemon import PokemonAgent

    assert PokemonAgent.handles(PokemonAgent.__new__(PokemonAgent)) == ("battle",)


def test_system_prompt_is_public(make_agent) -> None:
    agent = make_agent(FakeRouter())
    assert agent.system_prompt == prompts.SYSTEM_PROMPT
    assert len(agent.system_prompt) > 500


def test_status_shape_before_any_battle(make_agent) -> None:
    agent = make_agent(FakeRouter())

    status = agent.status()

    assert status["agent"] == "pokemon"
    assert status["role"] == "pokemon"
    assert status["paused"] is False
    assert status["battles_handled"] == 0
    assert status["last_outcome"] == ""
    assert status["handles"] == ["battle"]
    assert set(status["tools"]) == {"battle_action", "observe", "record_lesson", "finish"}
    assert "memory" in status


def test_status_after_a_battle(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    agent.handle_battle(emulator=emulator, max_turns=3)
    status = agent.status()

    assert status["battles_handled"] == 1
    assert status["last_outcome"] in ("win", "loss", "caught", "ran", "unknown")
    assert status["turns"] == 1
    assert status["api_calls"] >= 1
    assert status["cost_usd"] > 0.0


def test_pause_resume_and_paused_property(make_agent) -> None:
    agent = make_agent(FakeRouter())
    assert agent.paused is False
    agent.pause()
    assert agent.paused is True
    assert agent.status()["paused"] is True
    agent.resume()
    assert agent.paused is False


def test_on_event_tracks_battle_flags(make_agent) -> None:
    agent = make_agent(FakeRouter())
    started = AgentMessage.create(
        EventType.BATTLE_STARTED,
        source_agent=AgentRole.NAVIGATION,
        payload={"frame": 12, "context": "battle screen"},
    )
    ended = AgentMessage.create(
        EventType.BATTLE_ENDED,
        source_agent=AgentRole.NAVIGATION,
        payload={"outcome": "win", "turns": 2},
    )

    agent.on_event(started)
    assert agent.status()["battle_flagged"] is True
    assert agent.status()["last_event"] == "battle_started"
    agent.on_event(ended)
    assert agent.status()["battle_flagged"] is False
    assert agent.status()["last_event"] == "battle_ended"


def test_on_event_pause_and_resume_for_this_agent(make_agent) -> None:
    agent = make_agent(FakeRouter())
    paused = AgentMessage.create(
        EventType.AGENT_PAUSED,
        source_agent=AgentRole.SYSTEM,
        payload={"agent": "pokemon", "reason": "budget warning"},
    )
    resumed = AgentMessage.create(
        EventType.AGENT_RESUMED,
        source_agent=AgentRole.SYSTEM,
        payload={"agent": "pokemon"},
    )

    agent.on_event(paused)
    assert agent.paused is True
    agent.on_event(resumed)
    assert agent.paused is False


def test_on_event_ignores_other_agents_and_unknown_events(make_agent) -> None:
    agent = make_agent(FakeRouter())
    other = AgentMessage.create(
        EventType.AGENT_PAUSED,
        source_agent=AgentRole.SYSTEM,
        payload={"agent": "story", "reason": "story pause"},
    )
    agent.on_event(other)
    assert agent.paused is False
    # Unknown / malformed events are ignored, never raised.
    agent.on_event("definitely-not-an-event")
    agent.on_event(None)
    agent.on_event({"event_type": "nope", "payload": {}})


def test_on_event_cancels_the_active_task(make_agent) -> None:
    agent = make_agent(FakeRouter())
    task = Task.create(
        owner_agent=AgentRole.POKEMON,
        requested_by=AgentRole.DIRECTOR,
        task_type=TaskType.BATTLE,
        description="handle the nearby battle",
    )
    agent._task = task  # the director sets this through handle_battle(task=...)
    cancelled = AgentMessage.create(
        EventType.TASK_CANCELLED,
        source_agent=AgentRole.DIRECTOR,
        payload={"task_id": task.task_id, "reason": "battle no longer needed"},
    )

    agent.on_event(cancelled)

    assert agent.status()["task_cancelled"] is True
    assert agent.paused is True


def test_recovery_hint_reaches_the_next_battle_prompt_and_is_consumed(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    agent.apply_recovery_hint("L1", "try a different visible option before repeating")
    assert agent.status()["recovery_hints"]
    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert "try a different visible option before repeating" in router.all_text()
    # The hint was used by this battle, so it does not leak into the next one.
    assert agent.status()["recovery_hints"] == []

    agent.apply_recovery_hint("L1", "second hint")
    emulator2 = FakeBattleEmulator(end_after_fights=1)
    agent.handle_battle(emulator=emulator2, max_turns=3)
    assert "second hint" in router.all_text()


def test_private_memory_is_created_when_none_is_injected(make_agent, agent_config) -> None:
    agent = make_agent(FakeRouter())
    status = agent.status()
    assert status["memory"]["battles"] == 0
    assert agent._memory is not None
    assert agent._memory.creation_error == ""


def test_multi_agent_config_section_is_honored(make_agent, agent_config) -> None:
    agent_config.multi_agent.enabled = True
    agent_config.multi_agent.image_scale_agent = 2
    agent_config.multi_agent.max_tokens_agent = 300
    agent_config.multi_agent.thinking_agent = "disabled"
    agent_config.multi_agent.battle_max_turns = 7
    agent = make_agent(FakeRouter(), config=agent_config)

    assert agent._image_scale == 2
    assert agent._max_tokens == 300
    assert agent._thinking == "disabled"
    assert agent._coerce_max_turns(None) == 7
    assert agent._coerce_max_turns(3) == 3


def test_constructor_requires_a_router(agent_config) -> None:
    from llm_agent.pokemon import PokemonAgent

    try:
        PokemonAgent(
            router=None,
            bus=None,
            arbiter=None,
            shared_memory=None,
            memory=None,
            config=agent_config,
        )
    except ValueError as exc:
        assert "router" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("PokemonAgent must reject a missing router")
