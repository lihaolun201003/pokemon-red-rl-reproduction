"""Tests for the Lead-owned glue modules: ModelRouter / AgentRegistry / AgentContext.

All offline: a fake DeepSeek client produces deterministic replies and a real
:class:`llm_agent.budget.BudgetTracker` (tmp files) proves that per-agent
accounting never bypasses the global budget and never goes backwards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from llm_agent.budget import BudgetTracker
from llm_agent.config import AgentConfig
from llm_agent.contracts import ApiError, ApiUsage, Frame, ModelReply, ToolCall
from llm_agent.multi_agent.agent_context import (
    AgentContext,
    AgentContextConfig,
    ActionPrimitives,
    capture_evidence,
    image_content_block,
)
from llm_agent.multi_agent.agent_registry import AgentRegistry
from llm_agent.multi_agent.contracts import (
    AgentRole,
    BudgetLimitError,
    ControlDeniedError,
    ControlOwner,
    Priority,
    Task,
    TaskType,
)
from llm_agent.multi_agent.control_arbiter import ArbiterConfig, ControlArbiter
from llm_agent.multi_agent.model_router import ModelRouter, ModelRouterConfig


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeClient:
    def __init__(self, *, content: str = "ok", tool_calls: list[ToolCall] | None = None,
                 error: Exception | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: Any, **kwargs: Any) -> ModelReply:
        self.requests.append({"messages": messages, **kwargs})
        if self.error is not None:
            raise self.error
        return ModelReply(
            content=self.content,
            tool_calls=list(self.tool_calls),
            usage=ApiUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            model="fake-model",
            latency_s=0.01,
        )


class FakeEmulator:
    def __init__(self) -> None:
        self.frame_index = 3
        self.presses: list[str] = []

    def observe(self) -> Frame:
        rgba = np.full((144, 160, 4), 160, dtype=np.uint8)
        rgba[..., 3] = 255
        return Frame(frame_index=self.frame_index, rgba=rgba, sha256="x",
                     captured_at=time.time(), is_black=False, luma_mean=160.0)

    def press(self, button: str, *, frames: int | None = None) -> Frame:
        self.presses.append(button)
        self.frame_index += 24
        return self.observe()

    def wait(self, frames: int) -> Frame:
        self.frame_index += int(frames)
        return self.observe()

    def run_batch(self, buttons: Any, *, stop_on_event: bool = True) -> Any:
        self.presses.extend(list(buttons))
        from types import SimpleNamespace

        steps = [SimpleNamespace(button=b, frame=self.observe(), screen_changed=True,
                                 changed_ratio=0.1) for b in buttons]
        return SimpleNamespace(buttons=list(buttons), steps=steps, start_frame=0, end_frame=0,
                               stopped_reason="completed", events=[])


@pytest.fixture()
def budget(tmp_path: Path) -> BudgetTracker:
    cfg = AgentConfig()
    cfg.budget.max_usd_per_run = 1.0
    cfg.budget.max_calls_per_run = 100
    tracker = BudgetTracker(
        cfg.budget,
        cfg.api,
        tmp_path / "budget.json",
        total_path=tmp_path / "budget_total.json",
    )
    tracker.load()
    return tracker


def make_router(tmp_path: Path, budget: BudgetTracker, **config_kwargs: Any) -> ModelRouter:
    cfg = AgentConfig()
    config_kwargs.setdefault("director_min_interval_s", 0.0)
    config = ModelRouterConfig(
        models={"story": "story-model", "pokemon": "pokemon-model"},
        ledger_path=str(tmp_path / "model_usage.json"),
        **config_kwargs,
    )
    clients = {"story-model": FakeClient(content="story"), "pokemon-model": FakeClient(content="pokemon"),
               "deepseek-flash": FakeClient(content="default")}
    return ModelRouter(cfg, config=config, budget=budget, clients=clients)


# --------------------------------------------------------------------------- #
# ModelRouter
# --------------------------------------------------------------------------- #

def test_model_router_uses_per_agent_model_and_records_usage(tmp_path: Path, budget: BudgetTracker) -> None:
    router = make_router(tmp_path, budget)
    assert router.model_for(AgentRole.STORY) == "story-model"
    assert router.model_for(AgentRole.POKEMON) == "pokemon-model"
    assert router.model_for(AgentRole.DIRECTOR) == "deepseek-flash"
    reply = router.complete(AgentRole.STORY, [{"role": "user", "content": "hi"}])
    assert reply.content == "story"
    usage = router.usage()
    assert usage["agents"]["story"]["calls"] == 1
    assert usage["agents"]["story"]["cost_usd"] > 0
    assert usage["total"]["calls"] == 1
    # the shared budget tracker was charged as well
    assert budget.state().calls == 1
    # the ledger is persisted atomically and reloadable
    ledger = json.loads((tmp_path / "model_usage.json").read_text(encoding="utf-8"))
    assert ledger["usage"]["agents"]["story"]["calls"] == 1
    reloaded = make_router(tmp_path, budget)
    assert reloaded.usage()["agents"]["story"]["calls"] == 1


def test_model_router_enforces_per_agent_call_cap(tmp_path: Path, budget: BudgetTracker) -> None:
    router = make_router(tmp_path, budget, max_calls={"story": 1})
    router.complete(AgentRole.STORY, [{"role": "user", "content": "one"}])
    allowed, reason = router.can_call(AgentRole.STORY)
    assert allowed is False and "max_calls" in reason
    with pytest.raises(BudgetLimitError):
        router.complete(AgentRole.STORY, [{"role": "user", "content": "two"}])
    # other agents are unaffected
    assert router.can_call(AgentRole.POKEMON)[0] is True


def test_model_router_enforces_global_budget(tmp_path: Path, budget: BudgetTracker) -> None:
    budget.budget_cfg.max_calls_per_run = 1
    router = make_router(tmp_path, budget)
    router.complete(AgentRole.STORY, [{"role": "user", "content": "one"}])
    allowed, reason = router.can_call(AgentRole.POKEMON)
    assert allowed is False
    assert "call" in reason.lower()


def test_model_router_records_api_failures(tmp_path: Path, budget: BudgetTracker) -> None:
    cfg = AgentConfig()
    router = ModelRouter(
        cfg,
        config=ModelRouterConfig(models={"story": "broken"}, ledger_path=str(tmp_path / "usage.json")),
        budget=budget,
        clients={"broken": FakeClient(error=ApiError("boom", status=500, retryable=True))},  # type: ignore[call-arg]
    )
    with pytest.raises(ApiError):
        router.complete(AgentRole.STORY, [{"role": "user", "content": "x"}])
    usage = router.usage()["agents"]["story"]
    assert usage["failed_calls"] == 1


def test_model_router_director_cooldown(tmp_path: Path, budget: BudgetTracker) -> None:
    router = make_router(tmp_path, budget, director_min_interval_s=3600.0)
    router.complete(AgentRole.DIRECTOR, [{"role": "user", "content": "review"}])
    allowed, reason = router.can_call(AgentRole.DIRECTOR)
    assert allowed is False and "cooldown" in reason
    # story/pokemon are not rate-limited by the director cooldown
    assert router.can_call(AgentRole.STORY)[0] is True


def test_model_router_restore_never_rewinds_spend(tmp_path: Path, budget: BudgetTracker) -> None:
    router = make_router(tmp_path, budget)
    router.complete(AgentRole.STORY, [{"role": "user", "content": "one"}])
    snapshot = router.snapshot()
    router.complete(AgentRole.STORY, [{"role": "user", "content": "two"}])
    calls_before = router.usage()["agents"]["story"]["calls"]
    router.restore(snapshot)  # an old checkpoint must not reduce real spend
    assert router.usage()["agents"]["story"]["calls"] >= calls_before


# --------------------------------------------------------------------------- #
# AgentRegistry
# --------------------------------------------------------------------------- #

def test_agent_registry_tracks_availability_and_tool_permissions() -> None:
    registry = AgentRegistry()
    registry.register(AgentRole.NAVIGATION, object())
    registry.register(AgentRole.STORY, object())
    registry.register(AgentRole.POKEMON, object())
    assert set(registry.available()) == {"navigation", "story", "pokemon"}
    registry.pause(AgentRole.STORY, reason="budget")
    assert registry.is_available(AgentRole.STORY) is False
    registry.resume(AgentRole.STORY)
    assert registry.is_available(AgentRole.STORY) is True
    assert "explore_current_area" in registry.tools_for(AgentRole.NAVIGATION)
    assert "battle_action" in registry.tools_for(AgentRole.POKEMON)
    assert registry.owner_for_task_type(TaskType.BATTLE) is AgentRole.POKEMON
    assert registry.owner_for_task_type(TaskType.EXPLORE) is AgentRole.NAVIGATION
    description = registry.describe()
    assert "navigation" in description and "pokemon" in description
    with pytest.raises(KeyError):
        registry.require(AgentRole.DIRECTOR)


# --------------------------------------------------------------------------- #
# AgentContext
# --------------------------------------------------------------------------- #

def _task() -> Task:
    return Task.create(
        owner_agent=AgentRole.STORY,
        requested_by=AgentRole.DIRECTOR,
        task_type=TaskType.TALK,
        description="read the dialogue on screen",
        payload={"probe_interaction": True, "area_id": "area_001"},
    )


def test_agent_context_builds_bounded_multimodal_messages() -> None:
    context = AgentContext(
        agent=AgentRole.STORY,
        system_prompt="you are a blind player",
        config=AgentContextConfig(max_conversation_messages=4),
    )
    context.begin_task(_task())
    context.add_user("first", image_png=b"\x89PNG-fake", mime="image/png")
    context.add_assistant(ModelReply(content="looking", tool_calls=[
        ToolCall(id="call_1", name="observe", arguments={})
    ], usage=ApiUsage(), model="m"))
    context.add_tool_result(ToolCall(id="call_1", name="observe", arguments={}), "observed")
    messages = context.build_messages(user_text="next", image_png=b"\x89PNG-fake2", mime="image/png")
    assert messages[0]["role"] == "system"
    assert "blind player" in messages[0]["content"]
    assert "CURRENT TASK" in messages[0]["content"]
    assert "payload" in messages[0]["content"]  # the render_task bug regression guard
    assert messages[-1]["role"] == "user"
    assert any(isinstance(block, dict) and block.get("type") == "image_url"
               for block in messages[-1]["content"])
    # history stays bounded and protocol-valid
    assert len(messages) <= 4 + 2


def test_agent_context_sanitizes_unanswered_tool_calls() -> None:
    context = AgentContext(agent=AgentRole.STORY, system_prompt="s", config=AgentContextConfig(max_conversation_messages=8))
    context.add_assistant(ModelReply(content="two tools", tool_calls=[
        ToolCall(id="call_a", name="observe", arguments={}),
        ToolCall(id="call_b", name="finish", arguments={}),
    ], usage=ApiUsage(), model="m"))
    context.add_tool_result(ToolCall(id="call_a", name="observe", arguments={}), "ok")
    messages = context.build_messages()
    history = [message for message in messages if message["role"] != "system"]
    assistant = next(message for message in history if message["role"] == "assistant")
    assert len(assistant.get("tool_calls") or []) == 1
    assert assistant["tool_calls"][0]["id"] == "call_a"
    tool_ids = [message.get("tool_call_id") for message in history if message["role"] == "tool"]
    assert tool_ids == ["call_a"]


def test_agent_context_stats_and_snapshot_restore() -> None:
    context = AgentContext(agent=AgentRole.POKEMON, system_prompt="s")
    context.begin_task(_task())
    reply = ModelReply(content="c", tool_calls=[], usage=ApiUsage(prompt_tokens=10, completion_tokens=5,
                                                                 total_tokens=15), model="m")
    context.record_call(reply, cost_usd=0.001)
    assert context.stats["calls"] == 1
    snapshot = context.snapshot()
    other = AgentContext(agent=AgentRole.POKEMON, system_prompt="s")
    other.restore(snapshot)
    assert other.stats["calls"] == 1
    # a "rollback" to an older snapshot cannot reduce the recorded spend
    older = dict(snapshot)
    older["stats"] = {"calls": 0, "cost_usd": 0.0}
    other.restore(older)
    assert other.stats["calls"] == 1


# --------------------------------------------------------------------------- #
# ActionPrimitives / GuardedEmulator integration
# --------------------------------------------------------------------------- #

def test_action_primitives_only_press_while_holding_the_lease() -> None:
    emulator = FakeEmulator()
    arbiter = ControlArbiter(emulator, config=ArbiterConfig())
    primitives = ActionPrimitives(emulator=arbiter.proxy(AgentRole.STORY), agent=AgentRole.STORY,
                                  max_actions=5, save_screenshots=False)
    # not the owner yet
    with pytest.raises(ControlDeniedError):
        primitives.press("a")
    lease = arbiter.request(AgentRole.STORY, reason="test")
    try:
        frame = primitives.press("a")
        assert frame.frame_index > 0
        assert emulator.presses == ["a"]
        evidence = primitives.screenshot(save=False)
        assert evidence["bytes"] and evidence["mime"] == "image/png"
        assert image_content_block(evidence["bytes"], evidence["mime"])["type"] == "image_url"
    finally:
        arbiter.release(AgentRole.STORY, reason="done")
    with pytest.raises(ControlDeniedError):
        primitives.press("b")
    assert emulator.presses == ["a"]


def test_action_primitives_enforces_action_budget() -> None:
    emulator = FakeEmulator()
    arbiter = ControlArbiter(emulator, config=ArbiterConfig())
    primitives = ActionPrimitives(emulator=arbiter.proxy(AgentRole.POKEMON), agent=AgentRole.POKEMON,
                                  max_actions=2, save_screenshots=False)
    arbiter.request(AgentRole.POKEMON, reason="budget test")
    try:
        primitives.press("a")
        primitives.press("a")
        with pytest.raises(ControlDeniedError):
            primitives.press("a")
    finally:
        arbiter.release(AgentRole.POKEMON, reason="done")
    assert primitives.summary()["actions"] == 2


def test_capture_evidence_saves_png(tmp_path: Path) -> None:
    emulator = FakeEmulator()
    evidence = capture_evidence(emulator.observe(), screen_dir=tmp_path, prefix="test", save=True)
    assert Path(evidence["path"]).is_file()
    assert Path(evidence["path"]).suffix == ".png"
