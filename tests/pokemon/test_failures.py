"""Failure / interruption paths: they must return structured results, never crash."""

from __future__ import annotations

from llm_agent.config import AgentConfig
from llm_agent.contracts import ApiTimeoutError
from llm_agent.multi_agent.contracts import BudgetLimitError
from tests.pokemon.conftest import FakeBattleEmulator, FakeRouter


def test_api_failure_returns_failed_without_raising(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    router.fail_with = ApiTimeoutError("network went away", status=None, retryable=True)
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "failed"
    assert "ApiTimeoutError" in result["reason"]
    assert result["actions"] == []


def test_router_budget_block_returns_budget(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=None)
    router = FakeRouter([{"action": "fight", "move_slot": 1}], allowed=False, reason="budget gone")
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "budget"
    assert "budget gone" in result["reason"]
    assert result["api_calls"] == 0
    assert router.calls == []


def test_budget_exception_from_router_returns_budget(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=None)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    router.fail_with = BudgetLimitError("global limit reached")
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "budget"
    assert "global limit" in result["reason"]


def test_configured_call_cap_returns_budget(make_agent, agent_config) -> None:
    agent_config.budget.max_calls_per_run = 1
    emulator = FakeBattleEmulator(end_after_fights=None)
    router = FakeRouter(
        [{"action": "fight", "move_slot": 1}, {"action": "fight", "move_slot": 1}]
    )
    agent = make_agent(router, config=agent_config)

    result = agent.handle_battle(emulator=emulator, max_turns=5)

    assert result["status"] == "budget"
    assert "cap" in result["reason"].lower()
    assert result["turns"] == 1
    assert result["api_calls"] == 1


def test_configured_usd_cap_returns_budget(make_agent, agent_config) -> None:
    agent_config.budget.max_usd_per_run = 1e-9
    emulator = FakeBattleEmulator(end_after_fights=None)
    router = FakeRouter(
        [{"action": "fight", "move_slot": 1}, {"action": "fight", "move_slot": 1}]
    )
    agent = make_agent(router, config=agent_config)

    result = agent.handle_battle(emulator=emulator, max_turns=5)

    assert result["status"] == "budget"
    assert result["turns"] == 1
    assert result["cost_usd"] > 0.0


def test_turn_cap_returns_budget(make_agent, endless_emulator) -> None:
    router = FakeRouter([{"action": "observe"}] * 5)
    agent = make_agent(router)

    result = agent.handle_battle(emulator=endless_emulator, max_turns=2)

    assert result["status"] == "budget"
    assert result["turns"] == 2
    assert result["api_calls"] == 2
    assert "turn cap" in result["reason"].lower()


def test_control_denied_returns_control_denied(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1, denied=True)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "control_denied"
    assert "control" in result["reason"].lower()
    assert result["api_calls"] == 1  # the decision was requested, the press was denied
    assert agent.status()["denied"] == 1


def test_emulator_error_returns_failed(make_agent) -> None:
    class BrokenEmulator(FakeBattleEmulator):
        def observe(self):
            raise RuntimeError("emulator exploded")

    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=BrokenEmulator(), max_turns=3)

    assert result["status"] == "failed"
    assert "RuntimeError" in result["reason"]


def test_repeated_invalid_decisions_return_failed(make_agent, endless_emulator) -> None:
    router = FakeRouter([{"action": "dance"}] * 5)
    agent = make_agent(router)

    result = agent.handle_battle(emulator=endless_emulator, max_turns=5)

    assert result["status"] == "failed"
    assert "no valid decision" in result["reason"].lower()
    assert result["api_calls"] == 3
    assert result["actions"] == []


def test_paused_agent_returns_interrupted(make_agent, endless_emulator) -> None:
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)
    agent.pause()

    result = agent.handle_battle(emulator=endless_emulator, max_turns=3)

    assert result["status"] == "interrupted"
    assert result["turns"] == 0
    assert result["api_calls"] == 0
    assert agent.paused is True


def test_pause_during_a_battle_stops_at_the_next_turn(make_agent, endless_emulator) -> None:
    holder: dict = {}

    class PausingRouter(FakeRouter):
        def complete(self, *args, **kwargs):
            reply = super().complete(*args, **kwargs)
            holder["agent"].pause()
            return reply

    router = PausingRouter([{"action": "fight", "move_slot": 1}] * 4)
    agent = make_agent(router)
    holder["agent"] = agent

    result = agent.handle_battle(emulator=endless_emulator, max_turns=4)

    assert result["status"] == "interrupted"
    assert result["turns"] == 1
    assert len(result["actions"]) == 1
    assert "paused" in result["reason"].lower()


def test_resume_clears_the_paused_state(make_agent, endless_emulator) -> None:
    router = FakeRouter([{"action": "observe"}])
    agent = make_agent(router)
    agent.pause()
    assert agent.paused is True
    agent.resume()
    assert agent.paused is False

    result = agent.handle_battle(emulator=endless_emulator, max_turns=1)

    assert result["status"] == "budget"  # it ran (and hit the 1-turn cap), not interrupted
