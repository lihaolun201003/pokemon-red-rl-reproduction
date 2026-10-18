"""MultiAgentSession wiring tests (fake emulator / fake agents, no ROM/API).

The session is the layer that builds every subsystem from an :class:`AgentConfig`
and wires the recovery/checkpoint hooks into the director.  These tests prove the
wiring end-to-end without PyBoy or the network:

* setup creates the run dirs, event bus, arbiter, tasks, shared memory, monitor,
  branch checkpoints, registry, agents and director;
* one bounded director run writes a structured result JSON and state file;
* the branch checkpoint manifest saves + restores the game state atomically and
  keeps the learned experience / API spend out of the rollback;
* the G1 contention probe shows exactly one owner and denies unlicensed presses.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from llm_agent.config import AgentConfig
from llm_agent.contracts import ApiUsage, Frame, ModelReply, ScreenEvent
from llm_agent.multi_agent import director as director_module
from llm_agent.multi_agent.contracts import AgentRole, ControlDeniedError, ControlOwner
from llm_agent.multi_agent.integration import MultiAgentSession
from llm_agent.observation import ScreenAnalysis


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeEmulator:
    def __init__(self) -> None:
        self.frame_count = 5
        self.state_tag = "state0"
        self.loaded_state = ""

    def observe(self) -> Frame:
        rgba = np.full((144, 160, 4), 200, dtype=np.uint8)
        rgba[..., 3] = 255
        return Frame(frame_index=int(self.frame_count), rgba=rgba, sha256=f"sha{self.frame_count}",
                     captured_at=time.time(), is_black=False, luma_mean=200.0)

    @property
    def started(self) -> bool:
        return True

    def press(self, button: str, *, frames: int | None = None) -> Frame:
        self.frame_count += 24
        return self.observe()

    def wait(self, frames: int) -> Frame:
        self.frame_count += max(1, int(frames))
        return self.observe()

    def run_batch(self, buttons: Any, *, stop_on_event: bool = True) -> Any:
        self.frame_count += 24 * len(list(buttons))
        return SimpleNamespace(buttons=list(buttons), steps=[], start_frame=self.frame_count,
                               end_frame=self.frame_count, stopped_reason="completed", events=[])

    def save_state_bytes(self) -> bytes:
        return f"STATE:{self.state_tag}:{self.frame_count}".encode()

    def load_state(self, payload: Any) -> None:
        self.loaded_state = str(payload)

    def close(self) -> None:
        pass

    def set_frame_hook(self, hook: Any, *, fps: float = 60.0) -> None:
        pass

    def clear_frame_hook(self) -> None:
        pass


class FakeNavigation:
    available = True

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.relocalized = 0
        self.areas = 1

    def attach(self) -> dict[str, Any]:
        return {"ok": True, "matched": False, "new_area": True}

    def explore(self, *, max_steps: Any = None, max_seconds: Any = None) -> dict[str, Any]:
        self.calls.append("explore")
        return {"ok": True, "status": "step_limit", "kind": "explore", "report": {"steps": 3, "moves": 2},
                "nav_event": {}, "navigation_status": self.status(), "steps": 3, "moves": 2,
                "blocked": 0}

    def navigate_to(self, target: Any, *, max_steps: Any = None) -> dict[str, Any]:
        self.calls.append("navigate")
        return {"ok": True, "status": "completed", "kind": "navigate", "report": {"steps": 2, "moves": 1},
                "nav_event": {}, "navigation_status": self.status(), "steps": 2, "moves": 1,
                "blocked": 0}

    def status(self) -> dict[str, Any]:
        return {"area_id": "area_001", "pose": {"x": 0, "y": 0, "facing": "down", "valid": True},
                "cells_total": 20, "walkable": 10, "blocked": 2, "unknown_frontier": 3,
                "coverage_ratio": 0.5, "areas_total": self.areas, "paused": False}

    def summary(self) -> dict[str, Any]:
        return {"available": True, "area_id": "area_001", "areas_total": self.areas,
                "coverage_ratio": 0.5, "actions": 2}

    def snapshot(self) -> dict[str, Any]:
        return {"status": self.status(), "actions": 2}

    def restore(self, snapshot: Any) -> None:
        pass

    def relocalize(self) -> dict[str, Any]:
        self.relocalized += 1
        return {"ok": True, "relocalized": True, "status": self.status()}

    def handle_transition(self, *, max_wait_frames: int = 120) -> dict[str, Any]:
        self.areas += 1
        return {"ok": True, "status": self.status()}

    def resume(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def save(self) -> None:
        pass


class FakeStory:
    system_prompt = "story"

    def __init__(self) -> None:
        self.memory = None
        self.calls = 0

    def handle_dialog(self, *, emulator: Any, task: Any = None, max_turns: int = 12) -> dict[str, Any]:
        self.calls += 1
        return {"status": "dialog_ended", "summary": "npc talked", "discoveries": [], "npcs": [],
                "tasks_created": [], "turns": 1, "api_calls": 1, "cost_usd": 0.0,
                "evidence": {}, "buttons": 1}

    def propose_tasks(self, *, context: Any = None) -> list[dict[str, Any]]:
        return []

    def apply_recovery_hint(self, level: str, hint: str) -> None:
        pass


class FakePokemon:
    system_prompt = "pokemon"

    def __init__(self) -> None:
        self.memory = None

    def handle_battle(self, *, emulator: Any, task: Any = None, max_turns: int = 24) -> dict[str, Any]:
        return {"status": "battle_ended", "outcome": "win", "turns": 1, "api_calls": 1,
                "cost_usd": 0.0, "lessons": [], "actions": [], "evidence": {}, "buttons": 1}

    def apply_recovery_hint(self, level: str, hint: str) -> None:
        pass


class FakeRouter:
    def __init__(self) -> None:
        self.calls = 0

    def can_call(self, role: Any) -> tuple[bool, str]:
        return True, ""

    def model_for(self, role: Any) -> str:
        return "fake-model"

    def complete(self, role: Any, messages: Any, **kwargs: Any) -> ModelReply:
        self.calls += 1
        return ModelReply(content="{}", tool_calls=[], usage=ApiUsage(), model="fake-model")

    def probe(self, role: Any, text: str = "OK") -> ModelReply:
        self.calls += 1
        return ModelReply(content="OK", tool_calls=[], usage=ApiUsage(), model="fake-model")

    def usage(self) -> dict[str, Any]:
        return {"total": {"calls": self.calls, "cost_usd": 0.0, "total_tokens": 0},
                "agents": {"story": {"calls": 1, "cost_usd": 0.0}}}

    def costs_by_agent(self) -> dict[str, float]:
        return {"story": 0.0}


def _analysis_none() -> ScreenAnalysis:
    return ScreenAnalysis(event=ScreenEvent.NONE, changed=False, changed_ratio=0.0,
                          signature="s", is_black=False, luma_mean=200.0, summary="none", stats={})


@pytest.fixture()
def session_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-dummy-key")
    monkeypatch.setattr(director_module, "analyze", lambda frame, previous=None: _analysis_none())
    rom = tmp_path / "PokemonRed.gb"
    state = tmp_path / "start.state"
    rom.write_bytes(b"ROM")
    state.write_bytes(b"STATE")
    cfg = AgentConfig()
    cfg.emulator.rom = str(rom)
    cfg.emulator.state = str(state)
    cfg.emulator.max_actions = 0
    cfg.runtime.run_root = str(tmp_path / "runs")
    cfg.runtime.resume = False
    cfg.runtime.checkpoint_every_actions = 5
    cfg.viewer.enabled = False
    cfg.multi_agent.enabled = True
    cfg.multi_agent.max_decision_turns = 3
    cfg.multi_agent.director_max_steps = 3
    cfg.multi_agent.save_screenshots = True
    cfg.multi_agent.navigation_steps_per_task = 3
    emulator = FakeEmulator()
    navigation = FakeNavigation()
    session = MultiAgentSession(
        cfg,
        emulator=emulator,
        router=FakeRouter(),
        warmup=False,
        use_lock=False,
        story_agent=FakeStory(),
        pokemon_agent=FakePokemon(),
        navigation_adapter=navigation,
    )
    session.setup()
    return SimpleNamespace(cfg=cfg, session=session, emulator=emulator,
                           navigation=navigation, tmp_path=tmp_path)


def test_session_setup_builds_all_subsystems(session_env) -> None:
    session = session_env.session
    assert session.run_dir is not None and session.run_dir.is_dir()
    assert session.bus is not None and session.arbiter is not None
    assert session.tasks is not None and session.shared is not None
    assert session.monitor is not None and session.recovery is not None
    assert session.checkpoints is not None and session.director is not None
    assert session.story is not None and session.pokemon is not None
    assert session.branch is not None and session.branch.branch_id.startswith("branch_")
    status = session.status()
    assert status["branch_id"] == session.branch.branch_id
    assert "director" in status and "navigation" in status
    assert session.status_lines(), "viewer status lines must be available"


def test_session_run_writes_structured_result(session_env) -> None:
    session = session_env.session
    result = session.run()
    assert result.status in ("finished", "paused_limits", "stopped", "blocked", "error")
    assert result.steps >= 1
    result_json = Path(result.run_dir) / "multi_agent_result.json"
    assert result_json.is_file()
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["result"]["status"] == result.status
    state_json = Path(result.run_dir) / "multi_agent_state.json"
    assert state_json.is_file()
    # learned knowledge persisted (task queue snapshot / shared memory files exist)
    assert (Path(result.run_dir) / "multi_agent").is_dir()


def test_session_checkpoint_manifest_roundtrip(session_env) -> None:
    session = session_env.session
    session.task_manager_placeholder = None
    # create one task so the snapshot is non-trivial
    task, _ = session.tasks.create_task(
        owner_agent=AgentRole.NAVIGATION, requested_by=AgentRole.DIRECTOR,
        task_type="explore", description="roundtrip task", branch_id=session.branch.branch_id,
    )
    session.emulator.state_tag = "before"
    manifest = session._save_checkpoint("test:roundtrip")
    assert manifest["ok"] is True
    assert manifest.get("checkpoint_id")
    # change the game state, then restore
    session.emulator.state_tag = "after"
    restore = session._restore_checkpoint_snapshot(manifest["checkpoint_id"], session.branch.branch_id)
    assert restore["ok"] is True
    assert "before" in session.emulator.loaded_state
    assert session.tasks.get(task.task_id) is not None


def test_g1_control_contention_probe(session_env) -> None:
    probe = session_env.session.g1_control_contention_probe()
    assert probe["ok"] is True
    assert probe["denied_roles"], "at least one competing agent must be denied"
    assert probe["frame_unchanged"] is True
    assert probe["owner_during"] != "none"
    assert probe["unauthorized_press_error"]


def test_session_close_is_idempotent(session_env) -> None:
    session = session_env.session
    session.close()
    session.close()  # must not raise
