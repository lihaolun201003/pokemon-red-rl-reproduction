"""Integration tests for Navigation v2 inside the DeepSeek tool/runtime stack.

Two levels:
1. ``ToolExecutor`` navigation tools with a fake navigator (no emulator/ROM):
   validation, target conversion, structured tool results, pause/resume, and the
   "navigation disabled" error path.
2. A real-runtime dry run (real ROM/state, deterministic custom brain, **no API
   key/network**) that proves the wiring end-to-end: the brain asks for
   ``explore_current_area``, the Python navigation subsystem drives the emulator,
   the permanent world model is persisted, navigation audit/status reach the
   prompt, and the run finishes cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_agent.config import AgentConfig, NavigationConfig
from llm_agent.contracts import ApiUsage, ModelReply, RuntimeState, ToolCall
from llm_agent.tools import ToolExecutor


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeEmulator:
    """Minimal emulator surface for the executor (never actually pressed)."""

    def __init__(self) -> None:
        self.frame_count = 0

    def observe(self):  # pragma: no cover - nav tool tests never observe
        raise AssertionError("nav tool tests must not observe the emulator")


class FakeMemory:
    def stats(self) -> dict[str, Any]:
        return {"places": 0, "discoveries": 0}


class FakeReport:
    def __init__(self, status: str = "completed", *, area: str = "area_001") -> None:
        self.status = type("S", (), {"value": status})()
        self.area_id = area
        self.reason = "test report"
        self.steps = 12
        self.moves = 10
        self.blocked = 2
        self.unknown = 0
        self.new_cells = 42
        self.new_edges = 33
        self.new_exits = 1
        self.area_switched = False
        self.new_area_id = ""
        self.elapsed_s = 1.5
        self.frames = 240
        self.pose = None
        self.coverage = None
        self.event = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "area_id": self.area_id,
            "reason": self.reason,
            "steps": self.steps,
            "moves": self.moves,
            "blocked": self.blocked,
            "unknown": self.unknown,
            "new_cells": self.new_cells,
            "new_edges": self.new_edges,
            "new_exits": self.new_exits,
            "elapsed_s": self.elapsed_s,
            "frames": self.frames,
            "area_switched": self.area_switched,
            "new_area_id": self.new_area_id,
        }


class FakeNavigator:
    def __init__(self) -> None:
        self.paused = False
        self.targets: list[dict[str, Any]] = []
        self.explore_calls: list[tuple[int, float]] = []
        self.last_attach = {"matched": False}

    def explore_current_area(self, *, max_steps: int = 300, max_seconds: float = 600.0):
        self.explore_calls.append((max_steps, max_seconds))
        return FakeReport()

    def navigate_to(self, target: dict[str, Any], *, max_steps: int = 200):
        self.targets.append(dict(target))
        return FakeReport("completed")

    def status(self) -> dict[str, Any]:
        return {
            "area_id": "area_001",
            "area_kind": "interior",
            "pose": {"x": 1, "y": 2, "facing": "down", "confidence": 0.9, "valid": True},
            "cells_total": 90,
            "walkable": 12,
            "blocked": 5,
            "unknown_frontier": 3,
            "coverage_ratio": 0.8,
            "areas_total": 2,
            "last_event": None,
            "paused": self.paused,
            "world": {"areas": [["area_001", "interior", 2, 12]], "connections": 1},
        }

    def local_map(self, *, radius: int = 12) -> dict[str, Any]:
        return {
            "area_id": "area_001",
            "pose": [1, 2],
            "ascii": "@..#\n..E?",
            "coverage": {"cells_total": 90, "walkable": 12, "blocked": 5,
                         "unknown_frontier": 3, "coverage_ratio": 0.8},
        }

    def world_summary(self) -> dict[str, Any]:
        return {"areas_total": 2, "connections": 1,
                "areas": [{"area_id": "area_001", "kind": "interior", "cells": 90}],
                "text": "area_001 (interior) -> area_002"}

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def observe_external_outcome(self, outcome: Any) -> None:  # pragma: no cover
        pass

    def save(self) -> None:
        pass


def _executor(navigator: Any = None) -> ToolExecutor:
    return ToolExecutor(
        emulator=FakeEmulator(),
        memory=FakeMemory(),
        state=RuntimeState(),
        screen_dir=Path("."),
        save_progress=lambda reason="": "checkpoint_000000001",
        restore_progress=lambda checkpoint_id="latest": "restored",
        navigator=navigator,
    )


# --------------------------------------------------------------------------- #
# 1) Tool layer integration
# --------------------------------------------------------------------------- #

def test_explore_tool_calls_navigator_and_reports_structure() -> None:
    nav = FakeNavigator()
    executor = _executor(nav)
    result = executor.execute(
        ToolCall(id="c1", name="explore_current_area",
                 arguments={"max_steps": 123, "max_seconds": 45})
    )
    assert result.ok, result.error
    assert nav.explore_calls == [(123, 45.0)]
    assert result.needs_llm is True  # control returns to the brain after exploring
    assert result.data["status"] == "completed"
    assert "new_cells=42" in result.summary
    assert executor.last_nav_report is not None
    assert executor.state.buttons_pressed == 12  # verified nav steps counted
    assert executor.state.frame_index == 240


def test_navigate_tool_converts_llm_target_to_contract_form() -> None:
    nav = FakeNavigator()
    executor = _executor(nav)
    cases = [
        ({"kind": "cell", "x": 3, "y": -2}, {"cell": [3, -2]}),
        ({"kind": "exit"}, {"exit": True}),
        ({"kind": "landmark", "landmark": "door"}, {"landmark": "door"}),
        ({"kind": "return_to_entry"}, {"return_to_entry": True}),
        ({"kind": "area", "area_id": "area_002", "x": 1, "y": 1},
         {"area": "area_002", "cell": [1, 1]}),
    ]
    for index, (arguments, expected) in enumerate(cases):
        result = executor.execute(
            ToolCall(id=f"n{index}", name="navigate_to", arguments=arguments)
        )
        assert result.ok, result.error
        assert nav.targets[-1] == expected


def test_navigation_status_map_world_tools() -> None:
    nav = FakeNavigator()
    executor = _executor(nav)
    status = executor.execute(ToolCall(id="s", name="get_navigation_status", arguments={}))
    assert status.ok and "area_001" in status.summary
    local = executor.execute(ToolCall(id="m", name="get_local_map", arguments={"radius": 6}))
    assert local.ok and "@..#" in local.summary and local.data["coverage"]["walkable"] == 12
    world = executor.execute(ToolCall(id="w", name="get_world_map_summary", arguments={}))
    assert world.ok and "area_001" in world.summary


def test_pause_and_resume_exploration_tools() -> None:
    nav = FakeNavigator()
    executor = _executor(nav)
    paused = executor.execute(ToolCall(id="p", name="pause_exploration", arguments={}))
    assert paused.ok and nav.paused is True
    blocked = executor.execute(
        ToolCall(id="e", name="explore_current_area", arguments={})
    )
    assert blocked.ok is False and "paused" in (blocked.error or "")
    resumed = executor.execute(ToolCall(id="r", name="resume_exploration", arguments={}))
    assert resumed.ok and nav.paused is False


def test_navigation_tools_error_when_navigation_disabled() -> None:
    executor = _executor(navigator=None)
    result = executor.execute(ToolCall(id="x", name="explore_current_area", arguments={}))
    assert result.ok is False
    assert "navigation subsystem is disabled" in (result.error or "")


def test_navigation_tool_validation_rejects_bad_arguments() -> None:
    executor = _executor(FakeNavigator())
    with pytest.raises(Exception):
        executor.validate("navigate_to", {"kind": "cell"})  # missing x/y
    with pytest.raises(Exception):
        executor.validate("navigate_to", {"kind": "exit", "bogus": 1})
    with pytest.raises(Exception):
        executor.validate("get_local_map", {"radius": 999})
    assert executor.validate("explore_current_area", {"max_steps": 5}) == {"max_steps": 5}


def test_prompt_renders_navigation_status_block() -> None:
    from llm_agent import prompts

    frame = type("F", (), {"frame_index": 3, "rgba": None})()
    state = RuntimeState(goal="explore")
    user_text = prompts.build_user_text(
        trigger="after_actions",
        frame=frame,
        analysis=None,
        state=state,
        memory_context="",
        guard_text="",
        last_outcome_text="",
        nav_status_text="area=area_001 pose=(1,2) walkable=12\n@..#\n..E?",
    )
    assert "NAVIGATION STATUS" in user_text
    assert "walkable=12" in user_text


def test_navigation_config_defaults_and_resolution(tmp_path: Path) -> None:
    cfg = AgentConfig.load(env_file="", overrides={
        "runtime.run_dir": str(tmp_path / "run"),
    })
    assert isinstance(cfg.navigation, NavigationConfig)
    assert cfg.navigation.enabled is True
    cfg.resolve_paths()
    assert cfg.navigation.world_root == str(tmp_path / "run" / "navigation")
    assert Path(cfg.navigation.world_root).is_dir()


# --------------------------------------------------------------------------- #
# 2) Runtime integration (real emulator, deterministic brain, no API key)
# --------------------------------------------------------------------------- #

class ExploreThenFinishBrain:
    """Calls the navigation tool first, then sets a goal; never uses the API."""

    mode = "scripted"

    def __init__(self) -> None:
        self.model = "scripted"
        self.turn = 0
        self.prepared_kwargs: list[dict[str, Any]] = []

    def prepare(self, **kwargs: Any):
        from llm_agent.agent import PreparedDecision

        self.prepared_kwargs.append(dict(kwargs))
        return PreparedDecision(messages=[{"role": "user", "content": "nav integration"}],
                                audit={"mode": self.mode, "prompt_chars": 0,
                                       "image_count": 0, "image_bytes": 0,
                                       "max_tokens": 0, "model": self.model})

    def complete(self, prepared) -> ModelReply:
        self.turn += 1
        if self.turn == 1:
            calls = [ToolCall(id="t1", name="explore_current_area",
                              arguments={"max_steps": 40, "max_seconds": 20})]
        else:
            calls = [ToolCall(id=f"t{self.turn}", name="set_goal",
                              arguments={"goal": "navigation integration test"})]
        return ModelReply(content="nav integration", tool_calls=calls, usage=ApiUsage(),
                          model=self.model, finish_reason="scripted")


def _make_cfg(tmp_path: Path, rom: Path, state: Path) -> AgentConfig:
    base = {
        "emulator.rom": str(rom),
        "emulator.state": str(state),
        "runtime.run_root": str(tmp_path / "runs"),
        "runtime.lock_path": str(tmp_path / "agent.lock"),
        "runtime.run_dir": str(tmp_path / "runs" / "run_nav"),
        "runtime.max_turns": 3,
        "runtime.checkpoint_every_actions": 5,
        "runtime.save_screens": False,
        "memory.root": str(tmp_path / "memory"),
        "viewer.enabled": False,
        "budget.max_usd_per_run": 0.5,
        "api.max_retries": 0,
        "navigation.enabled": True,
        "navigation.max_explore_steps": 80,
        "navigation.max_explore_seconds": 30.0,
    }
    return AgentConfig.load(env_file="", overrides=base)


@pytest.mark.real_rom
def test_runtime_navigation_tool_end_to_end(tmp_path: Path, require_rom) -> None:
    from llm_agent.runtime import AgentRuntime

    rom, state = require_rom
    cfg = _make_cfg(tmp_path, rom, state)
    brain = ExploreThenFinishBrain()
    runtime = AgentRuntime(cfg, brain=brain)
    result = runtime.run()
    assert result.status in ("finished", "stopped", "paused_budget"), result.reason
    assert result.buttons > 0, "navigation produced no verified movement"

    run_dir = Path(result.run_dir)
    nav_root = run_dir / "navigation"
    assert (nav_root / "world.json").is_file(), "permanent world model not persisted"
    world = json.loads((nav_root / "world.json").read_text(encoding="utf-8"))
    assert world.get("areas")
    area_files = list((nav_root / "areas").glob("*.json"))
    assert area_files, "no per-area grid map persisted"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["navigation"]["areas_total"] >= 1
    assert metrics["navigation"]["cells_total"] >= 90

    navigation_audit = (run_dir / "audit" / "navigation.jsonl")
    assert navigation_audit.is_file() and navigation_audit.stat().st_size > 0
    steps = [json.loads(line) for line in navigation_audit.read_text(encoding="utf-8").splitlines()]
    assert len(steps) >= 5
    assert {step.get("motion", {}).get("result") for step in steps} & {"moved", "blocked",
                                                                      "turned_only"}

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Navigation v2" in report
    # the brain must have received the navigation prompt block on later turns
    nav_texts = [item.get("nav_status_text") or "" for item in brain.prepared_kwargs]
    assert any(text.strip() for text in nav_texts), "navigation status never reached the prompt"
    # and the world model must have survived the final checkpoint round-trip
    assert not (tmp_path / "agent.lock").exists()
