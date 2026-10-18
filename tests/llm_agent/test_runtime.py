"""Runtime integration tests for the DeepSeek LLM agent (no network, no API key).

The full scripted dry run uses the real local ROM/state; it validates the
emulator -> tools -> memory -> guard -> audit -> checkpoint -> report pipeline.
A scripted run is explicitly *not* real-API evidence and never claims to be.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from llm_agent.config import AgentConfig
from llm_agent.contracts import BudgetState, RuntimeState
from llm_agent.runtime import (
    AgentRuntime,
    AuditLogger,
    CheckpointStore,
    SingleInstanceLock,
    RunResult,
    write_json_atomic,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_cfg(tmp_path: Path, rom: Path, state: Path, **overrides) -> AgentConfig:
    base = {
        "emulator.rom": str(rom),
        "emulator.state": str(state),
        "runtime.run_root": str(tmp_path / "runs"),
        "runtime.lock_path": str(tmp_path / "agent.lock"),
        "runtime.run_dir": str(tmp_path / "runs" / "run_test"),
        "runtime.max_turns": 2,
        "runtime.checkpoint_every_actions": 1,
        "runtime.save_screens": False,
        "runtime.screenshot_every_batches": 1,
        "memory.root": str(tmp_path / "memory"),
        "viewer.enabled": False,
        "budget.max_usd_per_run": 0.5,
        "budget.max_calls_per_run": 100,
        "api.max_retries": 0,
        "api.timeout_s": 30,
    }
    base.update(overrides)
    cfg = AgentConfig.load(env_file="", overrides=base)
    return cfg


# --------------------------------------------------------------------------- #
# Filesystem / lock / checkpoint primitives
# --------------------------------------------------------------------------- #

def test_audit_logger_writes_jsonl(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit")
    logger.log("events", {"event": "hello", "value": 1})
    logger.log("decisions", {"turn": 2})
    events = (tmp_path / "audit" / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 1
    payload = json.loads(events[0])
    assert payload["event"] == "hello" and payload["value"] == 1 and "ts" in payload
    assert json.loads((tmp_path / "audit" / "decisions.jsonl").read_text(encoding="utf-8"))["turn"] == 2


def test_single_instance_lock_blocks_second_agent(tmp_path: Path) -> None:
    path = tmp_path / "agent.lock"
    first = SingleInstanceLock(path)
    first.acquire(run_id="run_a")
    assert path.is_file()
    second = SingleInstanceLock(path)
    with pytest.raises(Exception):
        second.acquire(run_id="run_b")
    first.release()
    assert not path.exists()
    second.acquire(run_id="run_b")
    second.release()


def test_write_json_atomic_replaces(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.json"
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8"))["a"] == 2
    assert not target.with_name("file.json.tmp").exists()


def test_checkpoint_store_roundtrip_and_prune(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints", keep=2)
    for index in range(3):
        store.save(
            state_bytes=b"state-" + bytes([index]),
            frame=index * 10,
            buttons=index,
            batches=index,
            reason="test",
            memory_snapshot={"goal": {"text": "x"}},
            budget_snapshot={},
            runtime_snapshot={},
        )
    metas = store.list()
    assert len(metas) == 2  # pruned to keep=2
    assert metas[-1].frame == 20
    latest = store.latest()
    assert latest is not None and latest.id == "checkpoint_000000020"
    assert store.state_bytes(latest) == b"state-\x02"
    old = store.find("checkpoint_000000010")
    assert store.state_bytes(old) == b"state-\x01"
    assert store.meta_payload(old)["memory_snapshot"]["goal"]["text"] == "x"
    with pytest.raises(Exception):
        store.find("bogus-id")
    with pytest.raises(Exception):
        store.find("checkpoint_999999999")


# --------------------------------------------------------------------------- #
# Config / setup guards
# --------------------------------------------------------------------------- #

def test_config_validation_rejects_missing_rom(tmp_path: Path) -> None:
    cfg = AgentConfig.load(
        env_file="",
        overrides={
            "emulator.rom": str(tmp_path / "missing.gb"),
            "emulator.state": str(tmp_path / "missing.state"),
        },
    )
    problems = cfg.validate()
    assert any("ROM not found" in problem for problem in problems)
    assert any("initial state not found" in problem for problem in problems)


def test_missing_api_key_blocks_real_run(tmp_path: Path, rom_path: Path, state_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = make_cfg(tmp_path, rom_path, state_path, **{"runtime.max_turns": 1})

    from llm_agent.agent import DeepSeekBrain
    from llm_agent.api_client import DeepSeekClient

    brain = DeepSeekBrain(cfg, DeepSeekClient(cfg.api))
    runtime = AgentRuntime(cfg, brain=brain)
    result = runtime.run()
    assert isinstance(result, RunResult)
    assert result.status == "blocked"
    assert result.exit_code == 3
    assert "API" in result.reason or "api" in result.reason
    # no emulator was started, so no run directory report is required
    assert result.real_api is True


# --------------------------------------------------------------------------- #
# Full scripted dry run against the real emulator
# --------------------------------------------------------------------------- #

@pytest.mark.real_rom
def test_scripted_dry_run_end_to_end(tmp_path: Path, rom_path: Path, state_path: Path) -> None:
    from llm_agent.scripted import ScriptedBrain

    cfg = make_cfg(tmp_path, rom_path, state_path, **{"runtime.max_turns": 2})
    runtime = AgentRuntime(cfg, brain=ScriptedBrain())
    result = runtime.run()

    assert result.status in ("finished", "paused_budget", "stopped"), result.reason
    assert result.real_api is False
    assert result.mode == "scripted"
    assert result.turns >= 1
    assert result.buttons > 0
    assert result.frames > 0

    run_dir = Path(result.run_dir)
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "report.md").is_file()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "SCRIPTED DRY RUN" in report
    assert "真实 DeepSeek API" not in report or "不得作为" in report

    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["real_api"] is False
    assert run_json["rom"]["sha1"]
    assert run_json["initial_state"]["sha256"]

    actions = (run_dir / "audit" / "actions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    decisions = (run_dir / "audit" / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(actions) >= 1
    assert len(decisions) >= 1
    decision = json.loads(decisions[0])
    assert decision["mode"] == "scripted"
    assert decision["cost_usd"] == 0.0
    assert "rationale" in decision

    assert (run_dir / "checkpoints" / "latest.json").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] in ("finished", "paused_budget", "stopped")
    assert metrics["buttons"] > 0
    assert not (tmp_path / "agent.lock").exists()  # lock released
    # the self-recorded mental map must be persisted and summarised
    assert (run_dir / "mental_map.json").is_file()
    map_snapshot = json.loads((run_dir / "mental_map.json").read_text(encoding="utf-8"))
    assert map_snapshot["version"] == 1
    assert "mental_map" in metrics
    assert metrics["mental_map"]["explored_cells"] >= 1


@pytest.mark.real_rom
def test_resume_from_checkpoint(tmp_path: Path, rom_path: Path, state_path: Path) -> None:
    from llm_agent.scripted import ScriptedBrain

    cfg = make_cfg(tmp_path, rom_path, state_path, **{"runtime.max_turns": 1})
    first = AgentRuntime(cfg, brain=ScriptedBrain()).run()
    assert first.status in ("finished", "paused_budget", "stopped")
    run_dir = Path(first.run_dir)
    latest = json.loads((run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
    assert latest["frame"] >= 0

    second_cfg = make_cfg(tmp_path, rom_path, state_path, **{"runtime.max_turns": 1})
    second = AgentRuntime(second_cfg, brain=ScriptedBrain()).run()
    assert second.status in ("finished", "paused_budget", "stopped")
    events_path = Path(second.run_dir) / "audit" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").strip().splitlines()]
    restored = [event for event in events if event.get("event") == "checkpoint_restored"]
    assert restored, "second run should have resumed the saved checkpoint"
    assert restored[0]["frame"] == latest["frame"]


# --------------------------------------------------------------------------- #
# Small unit checks
# --------------------------------------------------------------------------- #

def test_runtime_state_history_is_bounded() -> None:
    state = RuntimeState(run_id="x")
    for index in range(100):
        state.add_history("tool", f"event {index}")
    assert len(state.recent_history(10)) == 10
    assert state.recent_history(10)[-1].summary == "event 99"
    assert len(state.history) <= 64


def test_budget_state_roundtrip() -> None:
    state = BudgetState(calls=3, prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.01)
    state.call_timestamps = [time.time()]
    restored = BudgetState.from_dict(state.to_dict())
    assert restored.calls == 3
    assert restored.cost_usd == pytest.approx(0.01)
    assert len(restored.call_timestamps) == 1


# --------------------------------------------------------------------------- #
# Budget stop / checkpoint tools (still no real API: synthetic usage only)
# --------------------------------------------------------------------------- #

class _CostlyScriptedBrain:
    """Scripted brain that reports synthetic token usage so budget logic can be tested."""

    mode = "scripted"
    model = "fake-costly"

    def __init__(self) -> None:
        from llm_agent.scripted import ScriptedBrain

        self._inner = ScriptedBrain()

    def prepare(self, **kwargs):
        return self._inner.prepare(**kwargs)

    def complete(self, prepared):
        from llm_agent.contracts import ApiUsage

        reply = self._inner.complete(prepared)
        reply.usage = ApiUsage(prompt_tokens=5000, completion_tokens=500, total_tokens=5500,
                               cache_hit_tokens=0, cache_miss_tokens=5000)
        reply.model = self.model
        return reply


@pytest.mark.real_rom
def test_budget_limit_pauses_safely(tmp_path: Path, rom_path: Path, state_path: Path) -> None:
    cfg = make_cfg(
        tmp_path, rom_path, state_path,
        **{
            "runtime.max_turns": 5,
            "budget.max_usd_per_run": 0.0009,
            "budget.min_remaining_usd": 0.0,
        },
    )
    result = AgentRuntime(cfg, brain=_CostlyScriptedBrain()).run()
    assert result.status == "paused_budget", result.reason
    assert result.exit_code == 0
    run_dir = Path(result.run_dir)
    assert (run_dir / "checkpoints" / "latest.json").is_file()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "paused_budget" in report
    assert result.cost_usd > 0


class _CheckpointThenRestoreBrain(_CostlyScriptedBrain):
    """Turn 1 saves a checkpoint; turn 2 restores it; then explores."""

    def __init__(self) -> None:
        super().__init__()
        self.turn = 0

    def complete(self, prepared):
        from llm_agent.contracts import ApiUsage, ModelReply, ToolCall

        self.turn += 1
        if self.turn == 1:
            calls = [ToolCall(id="c1", name="save_progress", arguments={})]
        elif self.turn == 2:
            calls = [ToolCall(id="c2", name="restore_checkpoint", arguments={"checkpoint": "latest"})]
        else:
            calls = [ToolCall(id=f"c{self.turn}", name="walk", arguments={"direction": "up", "max_steps": 2})]
        return ModelReply(content=f"synthetic turn {self.turn}", tool_calls=calls,
                          usage=ApiUsage(), model=self.model, latency_s=0.0, finish_reason="test")


@pytest.mark.real_rom
def test_agent_checkpoint_save_and_restore_tools(tmp_path: Path, rom_path: Path, state_path: Path) -> None:
    cfg = make_cfg(tmp_path, rom_path, state_path, **{"runtime.max_turns": 3})
    result = AgentRuntime(cfg, brain=_CheckpointThenRestoreBrain()).run()
    assert result.status in ("finished", "paused_budget", "stopped"), result.reason
    run_dir = Path(result.run_dir)
    events = [
        json.loads(line)
        for line in (run_dir / "audit" / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    checkpoint_saved = [event for event in events if event.get("event") == "checkpoint_saved"]
    restored = [event for event in events if event.get("event") == "checkpoint_restored"]
    assert checkpoint_saved, "save_progress should have saved a checkpoint"
    assert restored, "restore_checkpoint should have restored a checkpoint"
    actions = [
        json.loads(line)
        for line in (run_dir / "audit" / "actions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert any(record["tool"] == "restore_checkpoint" for record in actions)
    # The restore tool must interrupt the rest of its batch.
    assert any(event.get("event") == "batch_interrupted" for event in events)


class _DialogLoopBrain:
    """Opens a real dialogue, then stubbornly presses A forever (loop-breaker test)."""

    mode = "scripted"
    model = "fake-dialog-loop"

    def __init__(self) -> None:
        from llm_agent.scripted import ScriptedBrain

        self._inner = ScriptedBrain()
        self.turn = 0

    def prepare(self, **kwargs):
        return self._inner.prepare(**kwargs)

    def complete(self, prepared):
        from llm_agent.contracts import ApiUsage, ModelReply, ToolCall

        self.turn += 1
        if self.turn == 1:
            actions = ["up", "a", "a"]  # deterministic dialogue opener from the initial state
        else:
            actions = ["a", "a", "a"]
        return ModelReply(
            content="I will keep pressing A.",
            tool_calls=[ToolCall(id=f"a{self.turn}", name="execute_actions", arguments={"actions": actions})],
            usage=ApiUsage(),
            model=self.model,
            latency_s=0.0,
            finish_reason="test",
        )


@pytest.mark.real_rom
def test_runtime_breaks_repeated_a_dialogue_loop(tmp_path: Path, rom_path: Path, state_path: Path) -> None:
    """A pure A-button loop must be interrupted by the runtime self-correction logic."""
    cfg = make_cfg(
        tmp_path, rom_path, state_path,
        **{"runtime.max_turns": 10, "budget.max_calls_per_run": 100},
    )
    result = AgentRuntime(cfg, brain=_DialogLoopBrain()).run()
    assert result.status in ("finished", "paused_budget", "stopped"), result.reason
    run_dir = Path(result.run_dir)
    events = [
        json.loads(line)
        for line in (run_dir / "audit" / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert any(event.get("event") == "repetition_loop_break" for event in events), (
        "runtime should detect and break the repeated-A dialogue loop"
    )
    assert any(event.get("event") == "repetition_loop_step" for event in events), (
        "the runtime loop breaker should step the player away"
    )

