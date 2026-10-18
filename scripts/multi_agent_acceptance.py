"""Pokémon Red Multi-Agent v1 — real acceptance harness (G0–G7).

Each scenario writes machine-readable evidence under
``artifacts/multi_agent_v1/<scenario>/`` (gitignored) and prints a summary:

======================  =======================================================
G1  control exclusivity  real emulator, 3 agents race for control, unauthorized
                         presses denied, no duplicate presses, no deadlock
G2  Story takeover       real ROM + real DeepSeek: real dialogue detected/read,
                         knowledge recorded, control released, Navigation resumes
G3  Pokémon takeover     real ROM + real DeepSeek: navigation triggers a wild
                         battle, Pokémon agent decides/plays it via real menus,
                         control released, Navigation re-localizes and resumes
G4  cross-agent task     Story→Director→Navigation delegation over the event bus,
                         result return, dedupe + idempotent replay, story resumes
G5  checkpoint rollback  legal save state, branch manifest, rollback restores the
                         game+task state, learned knowledge kept (HISTORICAL),
                         API spend never rewound, relocalization attempted
G6  stagnation/recovery  configurable thresholds, L1/L2/L3 ladder, bounded
                         rollbacks, safe pause at budget, evidence trail
G7  real DeepSeek run    one integrated real multi-agent run with the real API,
                         reporting scheduling, handoffs, cost and actual progress
======================  =======================================================

``G0`` re-runs the regression suites (Navigation v2 + existing agent) and records
the honest pass/fail counts.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\multi_agent_acceptance.py --scenario g1
    .\\.venv\\Scripts\\python.exe scripts\\multi_agent_acceptance.py --scenario g2 --g3
    .\\.venv\\Scripts\\python.exe scripts\\multi_agent_acceptance.py --scenario g7
    .\\.venv\\Scripts\\python.exe scripts\\multi_agent_acceptance.py --all

Every scenario refuses to fabricate data: real API scenarios fail honestly
(``BLOCKED``) when the key/balance/network is unavailable, and the smoke/fake
routers are never used as acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACTS = PROJECT_ROOT / "artifacts" / "multi_agent_v1"
CONFIG_PATH = PROJECT_ROOT / "configs" / "multi_agent_v1.json"
STATES = PROJECT_ROOT / "work" / "multi_agent_v1" / "states"
G2_STATE = STATES / "g2_dialog_ready.state"
G3_STATE = STATES / "g3_battle_ready.state"
DEFAULT_STATE = PROJECT_ROOT / "local" / "has_pokedex_nballs.state"


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


_configure_stdio()

from llm_agent.config import AgentConfig  # noqa: E402
from llm_agent.contracts import ApiUsage, ModelReply, ToolCall  # noqa: E402
from llm_agent.emulator import GameBoyEmulator  # noqa: E402
from llm_agent.multi_agent.agent_registry import AgentRegistry  # noqa: E402
from llm_agent.multi_agent.contracts import (  # noqa: E402
    AgentMessage,
    AgentRole,
    ControlDeniedError,
    ControlOwner,
    EventType,
    Priority,
    TaskStatus,
    TaskType,
)
from llm_agent.multi_agent.control_arbiter import ArbiterConfig, ControlArbiter  # noqa: E402
from llm_agent.multi_agent.event_bus import EventBus, EventBusConfig  # noqa: E402
from llm_agent.multi_agent.integration import MultiAgentResult, MultiAgentSession  # noqa: E402
from llm_agent.observation import analyze, save_png  # noqa: E402


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #

class Evidence:
    """One scenario evidence directory: JSON records + copied screenshots."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.dir = ARTIFACTS / scenario
        (self.dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (self.dir / "evidence").mkdir(parents=True, exist_ok=True)
        self.records: dict[str, Any] = {}

    def record(self, name: str, value: Any) -> None:
        self.records[name] = value
        try:
            (self.dir / "evidence" / f"{name}.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            pass

    def copy_screens(self, source: str | Path | None, *, limit: int = 40, prefix: str = "") -> list[str]:
        if source is None:
            return []
        source_path = Path(source)
        if not source_path.exists():
            return []
        copies: list[str] = []
        patterns = ("*.png", "*.jpg")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(source_path.rglob(pattern)))
        files = files[:limit]
        for index, path in enumerate(files):
            target = self.dir / "screenshots" / f"{prefix}{index:03d}_{path.name}"
            try:
                shutil.copy2(path, target)
                copies.append(str(target.relative_to(self.dir)))
            except Exception:  # noqa: BLE001
                continue
        return copies

    def finish(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        data.setdefault("scenario", self.scenario)
        data.setdefault("finished_at", time.time())
        data.setdefault("records", self.records)
        (self.dir / "report.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
        print(f"[{self.scenario}] evidence -> {self.dir / 'report.json'}", flush=True)
        return data


def _screenshot(frame: Any, ev: Evidence, name: str) -> str:
    target = ev.dir / "screenshots" / name
    try:
        save_png(frame, target)
        return str(target.relative_to(ev.dir))
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _state_path(preferred: Path, fallback: Path = DEFAULT_STATE) -> Path:
    return preferred if preferred.is_file() else fallback


# --------------------------------------------------------------------------- #
# Config / session helpers
# --------------------------------------------------------------------------- #

def load_config(scenario: str, **overrides: Any) -> AgentConfig:
    cfg = AgentConfig.load(CONFIG_PATH, env_file=str(PROJECT_ROOT / ".env"))
    cfg.viewer.enabled = False
    cfg.runtime.resume = False
    cfg.runtime.run_root = str(ARTIFACTS / scenario / "runs")
    cfg.multi_agent.enabled = True
    cfg.multi_agent.evidence_root = str(ARTIFACTS / scenario / "evidence")
    cfg.multi_agent.save_screenshots = True
    cfg.emulator.window = "null"
    # Acceptance runs must fail fast instead of hanging on a stalled HTTP read.
    cfg.api.timeout_s = 60.0
    cfg.api.max_retries = 2
    for dotted, value in overrides.items():
        if value is None:
            continue
        node: Any = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        current = getattr(node, parts[-1])
        if isinstance(current, bool):
            setattr(node, parts[-1], bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(node, parts[-1], int(value))
        elif isinstance(current, float):
            setattr(node, parts[-1], float(value))
        else:
            setattr(node, parts[-1], value)
    return cfg


def build_session(
    cfg: AgentConfig,
    *,
    scenario: str,
    state_path: str | Path | None = None,
    router: Any | None = None,
    story_agent: Any | None = None,
    pokemon_agent: Any | None = None,
    warmup: bool = False,
    use_lock: bool = False,
) -> MultiAgentSession:
    if state_path is not None:
        cfg.emulator.state = str(state_path)
    return MultiAgentSession(
        cfg,
        router=router,
        story_agent=story_agent,
        pokemon_agent=pokemon_agent,
        warmup=warmup,
        use_lock=use_lock,
    )


def session_snapshot(session: MultiAgentSession, *, label: str = "") -> dict[str, Any]:
    try:
        status = session.status()
    except Exception as exc:  # noqa: BLE001
        status = {"error": f"{type(exc).__name__}: {exc}"}
    director = status.get("director") or {}
    return {
        "label": label,
        "at": time.time(),
        "owner": (session.arbiter.owner.value if session.arbiter else "none"),
        "branch_id": session.current_branch_id,
        "director_step": director.get("step", 0),
        "current_task": director.get("current_task", ""),
        "suspended_task": director.get("suspended_task", ""),
        "owner_switches": director.get("owner_switches", 0),
        "battles": director.get("battles", 0),
        "dialogs": director.get("dialogs", 0),
        "checkpoints": director.get("checkpoints", 0),
        "rollbacks": director.get("rollbacks", 0),
        "stagnation_events": director.get("stagnation_events", 0),
        "recoveries": director.get("recoveries", 0),
        "navigation": (session.navigation.summary() if session.navigation else {}),
        "tasks": (session.tasks.stats() if session.tasks else {}),
        "shared_memory": (session.shared.stats() if session.shared else {}),
        "monitor": (session.monitor.summary() if session.monitor else {}),
        "api": ((session.router.usage() if session.router else {}) or {}).get("total", {}),
        "arbiter": (session.arbiter.status() if session.arbiter else {}),
        "status": status,
    }


def _finalize(session: MultiAgentSession, status: str, reason: str, started: float) -> MultiAgentResult:
    try:
        return session._finish(status, reason, 0, time.time() - started)
    except Exception:  # noqa: BLE001 - if already finished/closed just use last_result
        if session.last_result is not None:
            return session.last_result
        raise


def _bus_recent(session: MultiAgentSession, limit: int = 80) -> list[dict[str, Any]]:
    if session.bus is None:
        return []
    try:
        return [message.to_dict() for message in session.bus.recent(limit=limit)]
    except Exception:  # noqa: BLE001
        return []


def _task_rows(session: MultiAgentSession, limit: int = 60) -> list[dict[str, Any]]:
    if session.tasks is None:
        return []
    try:
        rows = []
        for task in session.tasks.tasks()[:limit]:
            data = task.to_dict()
            rows.append({
                "task_id": data["task_id"],
                "owner": data["owner_agent"],
                "type": data["task_type"],
                "status": data["status"],
                "priority": data["priority"],
                "attempts": data["attempts"],
                "description": data["description"][:160],
                "reason": data["reason"][:160],
                "parent": data["parent_task_id"],
                "evidence_len": len(data["evidence"]),
            })
        return rows
    except Exception:  # noqa: BLE001
        return []


def _knowledge_rows(session: MultiAgentSession, limit: int = 40) -> list[dict[str, Any]]:
    if session.shared is None:
        return []
    try:
        entries = session.shared.search("", limit=limit, include_historical=True)
        return [
            {
                "knowledge_id": entry.knowledge_id,
                "category": entry.category,
                "content": entry.content[:200],
                "source_agent": entry.source_agent.value,
                "verification_status": entry.verification_status.value,
                "confidence": entry.confidence,
                "branch_id": entry.branch_id,
                "evidence_refs": entry.evidence_refs,
            }
            for entry in entries
        ]
    except Exception:  # noqa: BLE001
        return []


def _audit_rows(run_dir: str | Path | None, stream: str, *, limit: int = 40) -> list[dict[str, Any]]:
    """Read the tail of one audit JSONL stream for the acceptance report."""
    if not run_dir:
        return []
    path = Path(run_dir) / "audit" / f"{stream}.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        return []
    return rows[-limit:]


# --------------------------------------------------------------------------- #
# G0 — regression suites
# --------------------------------------------------------------------------- #

def run_g0(ev: Evidence) -> dict[str, Any]:
    commands = [
        ("navigation_v2_tests", [sys.executable, "-m", "pytest", "tests/navigation", "-q"]),
        ("llm_agent_tests", [sys.executable, "-m", "pytest", "tests/llm_agent", "-q"]),
        ("multi_agent_tests", [sys.executable, "-m", "pytest", "tests/multi_agent", "-q"]),
        ("story_tests", [sys.executable, "-m", "pytest", "tests/story", "-q"]),
        ("pokemon_tests", [sys.executable, "-m", "pytest", "tests/pokemon", "-q"]),
    ]
    results: list[dict[str, Any]] = []
    for name, command in commands:
        started = time.time()
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        summary_line = next((line for line in reversed(tail) if "passed" in line or "failed" in line or "error" in line), "")
        results.append({
            "suite": name,
            "command": " ".join(command),
            "returncode": proc.returncode,
            "summary": summary_line,
            "duration_s": round(time.time() - started, 2),
            "tail": tail,
        })
        print(f"[g0] {name}: rc={proc.returncode} {summary_line}", flush=True)
    ev.record("regression_suites", results)
    # Navigation v2 core acceptance scripts (bounded re-runs)
    nav_scripts = [
        ("nav_v2_g1_localization", [sys.executable, "scripts/nav_v2_g1_localization.py"]),
        ("nav_v2_g2_explore_room", [sys.executable, "scripts/nav_v2_g2_explore_room.py"]),
        ("nav_v2_g3_reentry", [sys.executable, "scripts/nav_v2_g3_reentry.py"]),
    ]
    nav_results: list[dict[str, Any]] = []
    for name, command in nav_scripts:
        script = PROJECT_ROOT / command[1]
        if not script.is_file():
            nav_results.append({"script": name, "skipped": "missing"})
            continue
        started = time.time()
        try:
            proc = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=900)
            ok = proc.returncode == 0
            tail = (proc.stdout or "").strip().splitlines()[-6:]
            nav_results.append({
                "script": name, "returncode": proc.returncode, "ok": ok,
                "duration_s": round(time.time() - started, 1), "tail": tail,
            })
            print(f"[g0] {name}: rc={proc.returncode} ({round(time.time() - started, 1)}s)", flush=True)
        except subprocess.TimeoutExpired:
            nav_results.append({"script": name, "ok": False, "error": "timeout"})
    ev.record("navigation_v2_acceptance", nav_results)
    total_passed = sum(1 for item in results if item["returncode"] == 0)
    all_ok = total_passed == len(results) and all(item.get("ok", False) for item in nav_results)
    return ev.finish({
        "ok": all_ok,
        "status": "PASS" if all_ok else "FAIL",
        "suites_ok": total_passed,
        "suites_total": len(results),
        "nav_scripts": nav_results,
    })


# --------------------------------------------------------------------------- #
# G1 — control exclusivity on the real emulator
# --------------------------------------------------------------------------- #

def run_g1(ev: Evidence) -> dict[str, Any]:
    cfg = load_config("g1")
    state = _state_path(G2_STATE)
    emulator = GameBoyEmulator(cfg.emulator).start(state_path=state)
    bus = EventBus(EventBusConfig(path=str(ev.dir / "g1_bus.jsonl"), persist=True))
    arbiter = ControlArbiter(emulator, config=ArbiterConfig(), event_bus=bus)
    roles = (AgentRole.NAVIGATION, AgentRole.STORY, AgentRole.POKEMON)
    proxies = {role: arbiter.proxy(role) for role in roles}
    initial_frame = int(emulator.frame_count)
    lock = threading.Lock()
    attempts: list[dict[str, Any]] = []
    barrier = threading.Barrier(len(roles))

    def worker(role: AgentRole) -> None:
        barrier.wait()
        for index in range(12):
            try:
                arbiter.request(role, reason=f"g1 contention {index}")
                frame_before = int(emulator.frame_count)
                proxies[role].press("a")
                frame_after = int(emulator.frame_count)
                arbiter.release(role, reason="g1 press done", context={"index": index})
                with lock:
                    attempts.append({"role": role.value, "granted": True,
                                     "frame_delta": frame_after - frame_before})
            except ControlDeniedError as exc:
                with lock:
                    attempts.append({"role": role.value, "granted": False, "owner": exc.owner})
            time.sleep(0.002)

    threads = [threading.Thread(target=worker, args=(role,), name=f"g1-{role.value}") for role in roles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    deadlocked = [thread.name for thread in threads if thread.is_alive()]
    owner_after = arbiter.owner.value
    final_frame = int(emulator.frame_count)

    # -- deterministic contention: two other agents are denied while one holds it
    arbiter.release_all("g1 deterministic contention")
    deterministic_denied: list[dict[str, Any]] = []
    held = arbiter.request(AgentRole.NAVIGATION, reason="g1 deterministic holder")
    for role in (AgentRole.STORY, AgentRole.POKEMON):
        try:
            arbiter.request(role, reason="g1 deterministic contender", wait=False)
            deterministic_denied.append({"role": role.value, "denied": False})
        except ControlDeniedError as exc:
            deterministic_denied.append({"role": role.value, "denied": True, "owner": exc.owner,
                                         "error": str(exc)[:140]})
    # same owner may renew/continue, but nobody else can press
    frame_blocked_before = int(emulator.frame_count)
    for role in (AgentRole.STORY, AgentRole.POKEMON):
        try:
            proxies[role].press("a")
        except ControlDeniedError:
            pass
    frame_blocked_after = int(emulator.frame_count)
    proxies[AgentRole.NAVIGATION].press("a")
    arbiter.release(AgentRole.NAVIGATION, reason="g1 deterministic holder released",
                    context={"navigation_task": "g1"})
    contention = {
        "holder": "navigation",
        "contenders": deterministic_denied,
        "blocked_presses_frame_unchanged": frame_blocked_before == frame_blocked_after,
        "released_context_preserved": bool(
            (arbiter.history(limit=5)[-1].get("context") or {}).get("navigation_task")
        ),
    }

    # -- invariant: never more than one active lease ---------------------- #
    history = arbiter.history(limit=2000)
    active = 0
    max_active = 0
    invariant_errors: list[str] = []
    for entry in history:
        kind = str(entry.get("kind") or "")
        if kind == "granted":
            active += 1
            max_active = max(max_active, active)
        elif kind in ("released", "preempted", "expired", "released_all"):
            active = max(0, active - 1)
    if max_active > 1:
        invariant_errors.append(f"max concurrent leases = {max_active}")
    if active != 0:
        invariant_errors.append(f"{active} leases still active after run")

    # -- unauthorized press is denied and does not touch the game --------- #
    unauthorized: dict[str, Any] = {}
    arbiter.release_all("g1 unauthorized probe")
    frame_before = int(emulator.frame_count)
    for role in roles:
        try:
            proxies[role].press("a")
            unauthorized[role.value] = {"denied": False}
        except ControlDeniedError as exc:
            unauthorized[role.value] = {"denied": True, "error": str(exc)[:120]}
    frame_after = int(emulator.frame_count)

    # -- preemption: no duplicate presses across handoff ------------------ #
    arbiter.release_all("g1 preemption setup")
    navigation_lease = arbiter.request(AgentRole.NAVIGATION, reason="g1 lease A")
    frame0 = int(emulator.frame_count)
    proxies[AgentRole.NAVIGATION].press("a")
    proxies[AgentRole.NAVIGATION].press("a")
    proxies[AgentRole.NAVIGATION].press("a")
    frame1 = int(emulator.frame_count)
    old_lease, pokemon_lease = arbiter.preempt(AgentRole.POKEMON, reason="g1 battle preemption")
    stale_denied = ""
    try:
        proxies[AgentRole.NAVIGATION].press("a")  # stale token must fail
    except ControlDeniedError as exc:
        stale_denied = str(exc)[:160]
    proxies[AgentRole.POKEMON].press("a")
    proxies[AgentRole.POKEMON].press("a")
    frame2 = int(emulator.frame_count)
    arbiter.release(AgentRole.POKEMON, reason="g1 done")
    handoff = {
        "navigation_presses": 3,
        "pokemon_presses": 2,
        "navigation_frame_delta": frame1 - frame0,
        "pokemon_frame_delta": frame2 - frame1,
        "stale_lease_denied": bool(stale_denied),
        "stale_error": stale_denied,
    }

    # -- exception safety: a crashed owner must not deadlock ------------- #
    crash: dict[str, Any] = {}
    try:
        arbiter.request(AgentRole.STORY, reason="g1 crash test")
        raise RuntimeError("simulated agent crash while holding control")
    except RuntimeError as exc:
        arbiter.release_all("owner crashed")
        crash = {"released": arbiter.owner.value == "none", "error": str(exc)}
    finally:
        arbiter.release_all("g1 cleanup")

    screenshot = _screenshot(emulator.observe(), ev, "g1_final.png")
    expected_press_frames = 24
    ok = (
        not deadlocked
        and not invariant_errors
        and all(item.get("denied") for item in unauthorized.values())
        and frame_after == frame_before
        and all(item.get("denied") for item in deterministic_denied)
        and contention["blocked_presses_frame_unchanged"]
        and contention["released_context_preserved"]
        and handoff["navigation_frame_delta"] == 3 * expected_press_frames
        and handoff["pokemon_frame_delta"] == 2 * expected_press_frames
        and handoff["stale_lease_denied"]
        and crash.get("released") is True
        and owner_after == "none"
        and any(item["granted"] for item in attempts)
    )
    total_granted = sum(1 for item in attempts if item["granted"])
    payload = {
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
        "attempts": attempts[:120],
        "granted": total_granted,
        "denied": len(attempts) - total_granted,
        "max_concurrent_leases": max_active,
        "invariant_errors": invariant_errors,
        "deadlocked_threads": deadlocked,
        "owner_after": owner_after,
        "frame_delta_total": final_frame - initial_frame,
        "unauthorized_presses": unauthorized,
        "unauthorized_frame_unchanged": frame_after == frame_before,
        "deterministic_contention": contention,
        "handoff": handoff,
        "crash_safety": crash,
        "arbiter_status": arbiter.status(),
        "screenshot": screenshot,
        "mode": "real PyBoy emulator, fake game state only (no LLM)",
    }
    ev.record("contention", payload)
    emulator.close()
    return ev.finish(payload)


# --------------------------------------------------------------------------- #
# Fake model / agent components (used by G5/G6; never as API evidence)
# --------------------------------------------------------------------------- #

class FakeRouter:
    """Deterministic offline router for the non-API scenarios (G5/G6)."""

    def __init__(self) -> None:
        self.calls = 0

    def can_call(self, role: Any) -> tuple[bool, str]:
        return True, ""

    def model_for(self, role: Any) -> str:
        return "offline-fake"

    def complete(self, role: Any, messages: Any, **kwargs: Any) -> ModelReply:
        self.calls += 1
        return ModelReply(content="{}", tool_calls=[], usage=ApiUsage(), model="offline-fake")

    def probe(self, role: Any, text: str = "OK") -> ModelReply:
        self.calls += 1
        return ModelReply(content="OK", tool_calls=[], usage=ApiUsage(), model="offline-fake")

    def usage(self) -> dict[str, Any]:
        return {"total": {"calls": self.calls, "cost_usd": 0.0, "total_tokens": 0},
                "agents": {"story": {"calls": 1, "cost_usd": 0.0}}}

    def costs_by_agent(self) -> dict[str, float]:
        return {"story": 0.0}


class FakeStoryAgent:
    system_prompt = "offline story agent (fake, for infrastructure acceptance only)"

    def __init__(self) -> None:
        self.memory = None
        self.calls = 0
        self.hints: list[str] = []

    def handle_dialog(self, *, emulator: Any, task: Any = None, max_turns: int = 12) -> dict[str, Any]:
        self.calls += 1
        return {"status": "dialog_ended", "summary": "offline dialog", "discoveries": [], "npcs": [],
                "tasks_created": [], "turns": 1, "api_calls": 0, "cost_usd": 0.0,
                "evidence": {}, "buttons": 1}

    def propose_tasks(self, *, context: Any = None) -> list[dict[str, Any]]:
        return []

    def apply_recovery_hint(self, level: str, hint: str) -> None:
        self.hints.append(f"{level}:{hint}")


class FakePokemonAgent:
    system_prompt = "offline pokemon agent (fake, for infrastructure acceptance only)"

    def __init__(self) -> None:
        self.memory = None
        self.calls = 0

    def handle_battle(self, *, emulator: Any, task: Any = None, max_turns: int = 24) -> dict[str, Any]:
        self.calls += 1
        return {"status": "battle_ended", "outcome": "win", "turns": 1, "api_calls": 0,
                "cost_usd": 0.0, "lessons": [], "actions": [], "evidence": {}, "buttons": 1}

    def apply_recovery_hint(self, level: str, hint: str) -> None:
        pass


# --------------------------------------------------------------------------- #
# G2 — real dialogue takeover (real ROM + real DeepSeek)
# --------------------------------------------------------------------------- #

def run_g2(ev: Evidence, *, api_key: str | None = None) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(
        "g2",
        **{
            "budget.max_usd_per_run": 0.60,
            "budget.max_calls_per_run": 120,
            "multi_agent.max_usd_story": 0.45,
            "multi_agent.max_calls_story": 70,
            "multi_agent.max_usd_director": 0.10,
            "multi_agent.max_usd_pokemon": 0.05,
            "multi_agent.director_llm_enabled": False,
            "multi_agent.dialogue_max_turns": 12,
            "multi_agent.navigation_steps_per_task": 2,
            "multi_agent.checkpoint_every_actions": 9999,
            "multi_agent.max_decision_turns": 40,
        },
    )
    state = _state_path(G2_STATE)
    if not state.is_file():
        return ev.finish({"ok": False, "status": "BLOCKED",
                          "reason": f"missing acceptance state {state} (run scripts/multi_agent_make_states.py --g2)"})
    session = build_session(cfg, scenario="g2", state_path=state)
    steps: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    phase_results: dict[str, Any] = {}
    try:
        session.setup()
        director = session.director
        assertion_errors: list[str] = []

        # ---- phase 1: a real dialogue is already on screen -> Story takes over
        initial = analyze(session.emulator.observe(), None)
        snapshots.append(session_snapshot(session, label="start"))
        phase1_steps: list[dict[str, Any]] = []
        story_ok = False
        for _ in range(5):
            if director.budget_exhausted() or director._runtime_exceeded():
                break
            step = director.step()
            phase1_steps.append(step.to_dict())
            steps.append(step.to_dict())
            story_tasks = [task for task in session.tasks.tasks() if task.owner_agent is AgentRole.STORY]
            if any(task.status in (TaskStatus.COMPLETED, TaskStatus.VERIFYING) for task in story_tasks):
                story_ok = True
                break
            if analyze(session.emulator.observe(), None).event.value not in ("dialog", "menu"):
                break
        phase1_tasks = [task for task in session.tasks.tasks() if task.owner_agent is AgentRole.STORY]
        phase_results["phase1_dialog_takeover"] = {
            "initial_screen_event": initial.event.value,
            "steps": phase1_steps,
            "story_ok": story_ok,
            "story_tasks": [task.to_dict() for task in phase1_tasks],
            "shared_knowledge": _knowledge_rows(session),
            "owner_after": session.arbiter.owner.value,
        }
        if initial.event.value != "dialog":
            assertion_errors.append(f"expected dialog on screen at start, got {initial.event.value}")
        if not story_ok:
            assertion_errors.append("story task was not completed on the real dialogue")
        if session.arbiter.owner is not ControlOwner.NONE:
            assertion_errors.append("control was not released after the dialogue")
        snapshots.append(session_snapshot(session, label="after_dialog"))

        # ---- phase 2: navigation runs and gets paused by its own step budget
        nav_task, _ = session.tasks.create_task(
            owner_agent=AgentRole.NAVIGATION, requested_by=AgentRole.DIRECTOR,
            task_type=TaskType.EXPLORE,
            description="explore the room for exits while the director is free",
            priority=Priority.NORMAL, branch_id=session.current_branch_id,
        )
        director.config.navigation_steps_per_task = 2
        nav_ran = False
        phase2_steps: list[dict[str, Any]] = []
        for _ in range(6):
            if director.budget_exhausted() or director._runtime_exceeded():
                break
            step = director.step()
            phase2_steps.append(step.to_dict())
            steps.append(step.to_dict())
            if step.action in ("navigation_task", "navigation_transition") or step.owner == "navigation":
                nav_ran = True
                break
        nav_status_after = session.tasks.get(nav_task.task_id)
        phase_results["phase2_navigation"] = {
            "steps": phase2_steps,
            "nav_ran": nav_ran,
            "navigation_task": nav_status_after.to_dict() if nav_status_after else {},
            "navigation_summary": session.navigation.summary(),
        }
        if not nav_ran:
            assertion_errors.append("navigation never executed a step before the probe")
        snapshots.append(session_snapshot(session, label="after_navigation"))

        # ---- phase 3: story probe task re-acquires the dialogue while navigation is suspended
        probe_task, created = session.tasks.create_task(
            owner_agent=AgentRole.STORY, requested_by=AgentRole.DIRECTOR,
            task_type=TaskType.INTERACT,
            description="interact with the npc in front of the player to read its dialogue",
            priority=Priority.CRITICAL, branch_id=session.current_branch_id,
            payload={"probe_interaction": True},
        )
        phase_results["phase3_probe_task_created"] = created
        probe_steps: list[dict[str, Any]] = []
        probe_ok = False
        for _ in range(10):
            if director.budget_exhausted() or director._runtime_exceeded():
                break
            step = director.step()
            probe_steps.append(step.to_dict())
            steps.append(step.to_dict())
            current = session.tasks.get(probe_task.task_id)
            if current is not None and current.status.terminal:
                probe_ok = current.status in (TaskStatus.COMPLETED, TaskStatus.VERIFYING)
                break
        probe_final = session.tasks.get(probe_task.task_id)
        resume_suspended = session.tasks.get(nav_task.task_id)
        phase_results["phase3_probe"] = {
            "steps": probe_steps,
            "probe_ok": probe_ok,
            "probe_task": probe_final.to_dict() if probe_final else {},
            "navigation_task_after_dialog": resume_suspended.to_dict() if resume_suspended else {},
            "owner_switches": director.stats_dict().get("owner_switches", 0),
            "ownership_history": (session.arbiter.history(limit=80) if session.arbiter else []),
        }
        if not probe_ok:
            assertion_errors.append(f"story probe task did not complete: {getattr(probe_final, 'status', None)}")
        if not any("navigation" == str(step.get("owner", "")) for step in steps):
            assertion_errors.append("navigation never owned control during the scenario")
        snapshots.append(session_snapshot(session, label="after_probe"))

        result = _finalize(session, "finished", "acceptance_g2", started)
        usage = session.router.usage() if session.router else {}
        total = usage.get("total") or {}
        cost = float(total.get("cost_usd", 0.0))
        real_api = cost > 0.0 and int(total.get("calls", 0)) > 0
        screens = ev.copy_screens(Path(result.run_dir) / "multi_agent" / "screens", limit=60, prefix="g2_")
        task_rows = _task_rows(session)
        knowledge = _knowledge_rows(session)
        ok = (
            not assertion_errors
            and real_api
            and any(entry["source_agent"] == "story" for entry in knowledge)
            and int(director.stats_dict().get("dialogs", 0)) >= 1
            and int(director.stats_dict().get("owner_switches", 0)) >= 2
        )
        payload = {
            "ok": ok,
            "status": "PASS" if ok else "FAIL",
            "mode": "REAL ROM + REAL DeepSeek API",
            "state": str(state),
            "initial_screen_event": initial.event.value,
            "assertion_errors": assertion_errors,
            "director_steps": steps,
            "phases": phase_results,
            "snapshots": snapshots,
            "tasks": task_rows,
            "knowledge": knowledge,
            "run_result": result.to_dict(),
            "run_dir": result.run_dir,
            "api": total,
            "cost_usd": round(cost, 6),
            "real_api": real_api,
            "screenshots": screens,
            "story_decisions": _audit_rows(result.run_dir, "agent_results"),
            "duration_s": round(time.time() - started, 1),
        }
        ev.record("dialogue_takeover", payload)
        return ev.finish(payload)
    except Exception as exc:  # noqa: BLE001 - honest failure, never fake success
        error = f"{type(exc).__name__}: {exc}"
        print("[g2] FAILED:", error, flush=True)
        traceback.print_exc()
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
        return ev.finish({"ok": False, "status": "BLOCKED", "reason": error,
                          "director_steps": steps, "traceback": traceback.format_exc()[-4000:]})


# --------------------------------------------------------------------------- #
# G3 — real battle takeover (real ROM + real DeepSeek)
# --------------------------------------------------------------------------- #

def run_g3(ev: Evidence, *, api_key: str | None = None) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(
        "g3",
        **{
            "budget.max_usd_per_run": 0.60,
            "budget.max_calls_per_run": 120,
            "multi_agent.max_usd_pokemon": 0.50,
            "multi_agent.max_calls_pokemon": 80,
            "multi_agent.max_usd_director": 0.05,
            "multi_agent.max_usd_story": 0.05,
            "multi_agent.director_llm_enabled": False,
            "multi_agent.battle_max_turns": 14,
            "multi_agent.navigation_steps_per_task": 8,
            "multi_agent.checkpoint_every_actions": 9999,
            "multi_agent.max_decision_turns": 40,
        },
    )
    state = _state_path(G3_STATE)
    if not state.is_file():
        return ev.finish({"ok": False, "status": "BLOCKED",
                          "reason": f"missing acceptance state {state} (run scripts/multi_agent_make_states.py --g3)"})
    session = build_session(cfg, scenario="g3", state_path=state)
    steps: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    assertion_errors: list[str] = []
    try:
        session.setup()
        director = session.director
        initial = analyze(session.emulator.observe(), None)
        snapshots.append(session_snapshot(session, label="start"))
        # 1) navigation walks in the grass until the battle event interrupts it
        nav_task, _ = session.tasks.create_task(
            owner_agent=AgentRole.NAVIGATION, requested_by=AgentRole.DIRECTOR,
            task_type=TaskType.EXPLORE,
            description="walk through the grass until something interrupts the exploration",
            priority=Priority.NORMAL, branch_id=session.current_branch_id,
        )
        battle_step: dict[str, Any] = {}
        battle_phase: dict[str, Any] = {}
        # The battle may need several director steps (intro animation -> menu ->
        # several turns -> result text). Keep stepping until the Pokémon task is
        # terminal (or the scenario budget is exhausted).
        for index in range(30):
            if director.budget_exhausted() or director._runtime_exceeded():
                break
            step = director.step()
            steps.append(step.to_dict())
            if step.context == "battle" and not battle_step:
                battle_step = step.to_dict()
            pokemon_tasks_now = [task for task in session.tasks.tasks()
                                 if task.owner_agent is AgentRole.POKEMON]
            if battle_step and any(task.status.terminal for task in pokemon_tasks_now):
                break
            if session._stop_requested:
                break
        snapshots.append(session_snapshot(session, label="battle_handled"))
        pokemon_tasks = [task for task in session.tasks.tasks() if task.owner_agent is AgentRole.POKEMON]
        nav_after = session.tasks.get(nav_task.task_id)
        battle_phase = {
            "initial_screen_event": initial.event.value,
            "battle_step": battle_step,
            "pokemon_tasks": [task.to_dict() for task in pokemon_tasks],
            "navigation_task": nav_after.to_dict() if nav_after else {},
            "kind_of_handoff": {
                "navigation_owned_before": any(str(step.get("owner")) == "navigation" for step in steps),
                "pokemon_owned_battle": any(str(step.get("owner")) == "pokemon" for step in steps),
                "owner_after": session.arbiter.owner.value,
            },
            "shared_knowledge": _knowledge_rows(session),
        }
        # 2) control must be back in the overworld and navigation must resume
        resume_steps: list[dict[str, Any]] = []
        for _ in range(10):
            current = analyze(session.emulator.observe(), None)
            if current.event.value == "battle":
                break
            step = director.step()
            resume_steps.append(step.to_dict())
            steps.append(step.to_dict())
            if director.budget_exhausted() or director._runtime_exceeded():
                break
            if any(str(step.get("owner")) == "navigation" for step in resume_steps):
                break
        battle_phase["resume_steps"] = resume_steps
        if not battle_step:
            assertion_errors.append("no battle context was ever handled by the Pokémon agent")
        if not any(task.status in (TaskStatus.COMPLETED, TaskStatus.VERIFYING) for task in pokemon_tasks):
            assertion_errors.append("pokemon battle task was not completed")
        if session.arbiter.owner is not ControlOwner.NONE:
            assertion_errors.append("control was not released after the battle")
        relocalized = session.navigation.summary() if session.navigation else {}
        if not relocalized.get("available"):
            assertion_errors.append("navigation subsystem unavailable after battle")
        result = _finalize(session, "finished", "acceptance_g3", started)
        usage = session.router.usage() if session.router else {}
        total = usage.get("total") or {}
        cost = float(total.get("cost_usd", 0.0))
        real_api = cost > 0.0 and int(total.get("calls", 0)) > 0
        screens = ev.copy_screens(Path(result.run_dir) / "multi_agent" / "screens", limit=60, prefix="g3_")
        ok = (not assertion_errors and real_api
              and int(director.stats_dict().get("battles", 0)) >= 1
              and bool(battle_step))
        # battle decisions actually taken by the model agent
        pokemon_actions: list[dict[str, Any]] = []
        for task in pokemon_tasks:
            for evidence in task.evidence:
                if isinstance(evidence, Mapping) and "result" in evidence:
                    pokemon_actions.append(dict(evidence))
        payload = {
            "ok": ok,
            "status": "PASS" if ok else "FAIL",
            "mode": "REAL ROM + REAL DeepSeek API",
            "state": str(state),
            "assertion_errors": assertion_errors,
            "phases": battle_phase,
            "director_steps": steps,
            "tasks": _task_rows(session),
            "knowledge": _knowledge_rows(session),
            "pokemon_evidence": pokemon_actions[-20:],
            "pokemon_decisions": _audit_rows(result.run_dir, "agent_results"),
            "snapshots": snapshots,
            "run_result": result.to_dict(),
            "run_dir": result.run_dir,
            "api": total,
            "cost_usd": round(cost, 6),
            "real_api": real_api,
            "screenshots": screens,
            "duration_s": round(time.time() - started, 1),
        }
        ev.record("battle_takeover", payload)
        return ev.finish(payload)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print("[g3] FAILED:", error, flush=True)
        traceback.print_exc()
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
        return ev.finish({"ok": False, "status": "BLOCKED", "reason": error,
                          "director_steps": steps, "traceback": traceback.format_exc()[-4000:]})


# --------------------------------------------------------------------------- #
# G4 — cross-agent task delegation over the event bus
# --------------------------------------------------------------------------- #

def run_g4(ev: Evidence) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(
        "g4",
        **{
            "budget.max_usd_per_run": 0.05,
            "budget.max_calls_per_run": 5,
            "multi_agent.max_usd_story": 0.02,
            "multi_agent.max_usd_pokemon": 0.02,
            "multi_agent.director_llm_enabled": False,
            "multi_agent.navigation_steps_per_task": 15,
            "multi_agent.checkpoint_every_actions": 9999,
            "multi_agent.max_decision_turns": 20,
        },
    )
    state = _state_path(DEFAULT_STATE)
    session = build_session(cfg, scenario="g4", state_path=state,
                            story_agent=FakeStoryAgent(), pokemon_agent=FakePokemonAgent())
    assertion_errors: list[str] = []
    try:
        session.setup()
        director = session.director
        bus = session.bus
        assert bus is not None
        director.config.navigation_steps_per_task = 250
        # 0) let navigation discover the room and pick a real walkable target
        discovery_steps: list[dict[str, Any]] = []
        target: dict[str, Any] = {}
        for _ in range(8):
            if director._runtime_exceeded():
                break
            step = director.step()
            discovery_steps.append(step.to_dict())
            try:
                local = session.navigation.local_map(radius=30)
                pose = (session.navigation.status().get("pose") or {})
                px, py = int(pose.get("x") or 0), int(pose.get("y") or 0)
                cells = [cell for cell in (local.get("cells") or [])
                         if str(cell.get("state")) == "walkable"]
                far = [cell for cell in cells
                       if abs(int(cell.get("x")) - px) + abs(int(cell.get("y")) - py) >= 4]
                if far:
                    chosen = max(far, key=lambda cell: abs(int(cell.get("x")) - px) + abs(int(cell.get("y")) - py))
                    target = {"kind": "cell", "x": int(chosen.get("x")), "y": int(chosen.get("y"))}
                    break
            except Exception:  # noqa: BLE001
                continue
        if not target:
            target = {"kind": "return_to_entry"}
        # 1) the Story agent has a task that needs navigation help (parent context)
        story_task, _ = session.tasks.create_task(
            owner_agent=AgentRole.STORY, requested_by=AgentRole.DIRECTOR,
            task_type=TaskType.INTERACT,
            description="story wants to reach an interaction point it noticed",
            priority=Priority.HIGH, branch_id=session.current_branch_id,
        )
        session.tasks.block(story_task.task_id, reason="waiting_for_delegation: needs navigation help")
        # 2) the Story agent requests navigation through the structured event bus
        request = AgentMessage.create(
            EventType.NAVIGATION_REQUEST,
            source_agent=AgentRole.STORY,
            target_agent=AgentRole.DIRECTOR,
            payload={"reason": "walk to the cell the story agent wants to inspect",
                     "target": target, "task_id": story_task.task_id},
            priority=Priority.HIGH,
            parent_task_id=story_task.task_id,
            branch_id=session.current_branch_id,
        )
        bus.publish(request)
        before = {task.task_id for task in session.tasks.tasks()}
        # 3) director receives the message and creates/schedules the delegated task
        step1 = director.step()
        after_tasks = session.tasks.tasks()
        new_tasks = [task for task in after_tasks if task.task_id not in before]
        delegated = [task for task in new_tasks if task.owner_agent is AgentRole.NAVIGATION]
        steps = [step1.to_dict()]
        nav_task_id = delegated[0].task_id if delegated else ""
        for _ in range(8):
            if not nav_task_id:
                break
            task_now = session.tasks.get(nav_task_id)
            if task_now is not None and task_now.status.terminal:
                break
            if director._runtime_exceeded():
                break
            steps.append(director.step().to_dict())
        final_nav_task = session.tasks.get(nav_task_id) if nav_task_id else None
        # 4) the requesting agent resumes decision-making from the returned result
        parent_after = session.tasks.get(story_task.task_id)
        story_next = session.story.propose_tasks(context={
            "completed_tasks": [{"task_id": nav_task_id,
                                 "result": (final_nav_task.evidence[-1].get("result", "")
                                            if final_nav_task and final_nav_task.evidence else "")}],
            "navigation": session.navigation.summary(),
        }) if session.story is not None else []
        # 5) message idempotency: replaying the same message must not duplicate work
        delegated_description = "walk to the cell the story agent wants to inspect"
        same_id_count = len([task for task in session.tasks.tasks()
                             if task.description == delegated_description])
        processed_before = len(getattr(director, "_processed_message_ids", set()))
        bus.publish(request)
        director.step()
        replayed_count = len([task for task in session.tasks.tasks()
                              if task.description == delegated_description])
        processed_after = len(getattr(director, "_processed_message_ids", set()))
        duplicate_rejected = False
        try:
            duplicate = AgentMessage.create(
                EventType.NAVIGATION_REQUEST,
                source_agent=AgentRole.STORY,
                target_agent=AgentRole.DIRECTOR,
                payload={"reason": "walk to the cell the story agent wants to inspect",
                         "target": target, "task_id": story_task.task_id},
                priority=Priority.HIGH,
                parent_task_id=story_task.task_id,
                branch_id=session.current_branch_id,
            )
            bus.publish(duplicate)
        except Exception as exc:  # noqa: BLE001 - expected duplicate rejection
            duplicate_rejected = "duplicate" in str(exc).lower()
        consumed = bus.consumption_progress()
        bus_events = _bus_recent(session, limit=160)
        if not delegated:
            assertion_errors.append("director did not create the delegated navigation task")
        if final_nav_task is None or not final_nav_task.status.terminal:
            assertion_errors.append(f"delegated navigation task was not completed: "
                                    f"{getattr(final_nav_task, 'status', None)}")
        if replayed_count != same_id_count:
            assertion_errors.append("replayed message created a duplicate task")
        if not duplicate_rejected:
            assertion_errors.append("duplicate content was not rejected by the bus")
        if not any(str(event.get("event_type")) == "task_completed" for event in bus_events):
            assertion_errors.append("no TASK_COMPLETED event on the bus")
        if parent_after is None or parent_after.status is not TaskStatus.READY:
            assertion_errors.append(f"requesting story task did not resume after the delegation "
                                    f"({getattr(parent_after, 'status', None)})")
        result = _finalize(session, "finished", "acceptance_g4", started)
        payload = {
            "ok": not assertion_errors,
            "status": "PASS" if not assertion_errors else "FAIL",
            "mode": "REAL ROM + REAL event bus/tasks/navigation (fake LLM agents: this scenario validates infrastructure, not model quality)",
            "state": str(state),
            "assertion_errors": assertion_errors,
            "message": request.to_dict(),
            "navigation_discovery_steps": discovery_steps,
            "director_steps": steps,
            "delegated_task": final_nav_task.to_dict() if final_nav_task else {},
            "requesting_task_after": parent_after.to_dict() if parent_after else {},
            "story_resumed_proposals": story_next,
            "bus": {
                "consumed": consumed,
                "duplicate_content_rejected": duplicate_rejected,
                "replay_created_no_duplicate": replayed_count == same_id_count,
                "processed_message_ids_before": processed_before,
                "processed_message_ids_after": processed_after,
                "recent_events": bus_events[-50:],
            },
            "tasks": _task_rows(session),
            "run_result": result.to_dict(),
            "run_dir": result.run_dir,
            "duration_s": round(time.time() - started, 1),
        }
        ev.record("delegation", payload)
        return ev.finish(payload)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print("[g4] FAILED:", error, flush=True)
        traceback.print_exc()
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
        return ev.finish({"ok": False, "status": "BLOCKED", "reason": error,
                          "traceback": traceback.format_exc()[-4000:]})


# --------------------------------------------------------------------------- #
# G5 — checkpoint / rollback with learned-experience retention
# --------------------------------------------------------------------------- #

def run_g5(ev: Evidence) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(
        "g5",
        **{
            "budget.max_usd_per_run": 0.01,
            "budget.max_calls_per_run": 5,
            "multi_agent.director_llm_enabled": False,
            "multi_agent.navigation_steps_per_task": 10,
            "multi_agent.checkpoint_every_actions": 9999,
            "multi_agent.max_decision_turns": 20,
        },
    )
    state = _state_path(DEFAULT_STATE)
    session = build_session(cfg, scenario="g5", state_path=state,
                            story_agent=FakeStoryAgent(), pokemon_agent=FakePokemonAgent())
    assertion_errors: list[str] = []
    try:
        session.setup()
        director = session.director
        # 1) learn something on the current branch + run navigation a bit
        session.shared.remember(
            "story_clue", "a real clue recorded before the rollback",
            source_agent=AgentRole.STORY, confidence=0.95,
            verification_status="verified", evidence_refs=["screen:pre_rollback"],
            branch_id=session.current_branch_id,
        )
        task, _ = session.tasks.create_task(
            owner_agent=AgentRole.NAVIGATION, requested_by=AgentRole.DIRECTOR,
            task_type=TaskType.EXPLORE, description="explore before the rollback",
            branch_id=session.current_branch_id,
        )
        pre_steps = 0
        for _ in range(3):
            if director.budget_exhausted() or director._runtime_exceeded():
                break
            director.step()
            pre_steps += 1
        # 2) save a legal checkpoint (full branch manifest)
        pre_area = session.navigation.status().get("area_id") if session.navigation else ""
        checkpoint = session._save_checkpoint("g5:pre_rollback")
        if not checkpoint.get("ok"):
            assertion_errors.append(f"checkpoint save failed: {checkpoint}")
        budget_before = (session.router.usage().get("total") or {}).get("cost_usd", 0.0)
        knowledge_version_before = int(session.shared.version)
        knowledge_count_before = len(_knowledge_rows(session, limit=200))
        # 3) mutate the current state: new knowledge + more game actions
        session.shared.remember(
            "battle_lesson", "a lesson learned after the checkpoint (must not be lost)",
            source_agent=AgentRole.POKEMON, confidence=0.8,
            verification_status="inferred", evidence_refs=["screen:post_checkpoint"],
            branch_id=session.current_branch_id,
        )
        for _ in range(3):
            director.step()
        # 4) force a rollback to the saved checkpoint via the recovery manager
        rollback = session.recovery.rollback("g5 acceptance rollback", checkpoint_id=checkpoint["checkpoint_id"])
        # 5) the director applies the new branch + relocalizes
        director._after_rollback({"rollback": rollback, "reason": "g5 acceptance rollback"})
        post_area = session.navigation.status().get("area_id") if session.navigation else ""
        budget_after = (session.router.usage().get("total") or {}).get("cost_usd", 0.0)
        knowledge_after = _knowledge_rows(session, limit=200)
        historical = [entry for entry in knowledge_after if entry["verification_status"] == "historical"]
        retained = [entry for entry in knowledge_after
                    if entry["verification_status"] != "historical"]
        post_checkpoint_lesson = [entry for entry in knowledge_after
                                  if "must not be lost" in str(entry.get("content"))]
        branches = session.checkpoints.list_branches() if session.checkpoints else []
        manifest = session.checkpoints.load_manifest(checkpoint["checkpoint_id"])
        verify = session.checkpoints.verify(checkpoint["checkpoint_id"])
        learning_entries: list[Any] = []
        try:
            learning_entries = session.checkpoints.learning_entries()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            learning_entries = []
        if not rollback.get("ok"):
            assertion_errors.append(f"rollback failed: {rollback.get('error')}")
        if not verify.get("ok"):
            assertion_errors.append(f"checkpoint manifest verification failed: {verify.get('reason')}")
        if not manifest.get("state_sha256"):
            assertion_errors.append("manifest has no state hash")
        if len(branches) < 2:
            assertion_errors.append(f"rollback did not create a new branch ({len(branches)})")
        if not session.current_branch_id or session.current_branch_id == checkpoint.get("branch_id"):
            assertion_errors.append("session did not switch to the post-rollback branch")
        if not session.emulator.loaded_state:
            assertion_errors.append("the emulator state was not restored from the checkpoint")
        if budget_after < budget_before:
            assertion_errors.append("API spend was rewound by the rollback")
        if int(session.shared.version) < knowledge_version_before:
            assertion_errors.append("knowledge version went backwards")
        if len(knowledge_after) < knowledge_count_before:
            assertion_errors.append("knowledge entries were deleted by the rollback")
        if not historical:
            assertion_errors.append("old-branch knowledge was not marked HISTORICAL")
        if not post_checkpoint_lesson:
            assertion_errors.append("the lesson learned after the checkpoint was lost")
        if not rollback.get("relocalized") and not (rollback.get("manifest") or {}).get("relocalize_pending"):
            assertion_errors.append("navigation was not relocalized after the rollback")
        # 6) the restored state must be re-observed: director needs 2 steps to work again
        director.frame_index = 0
        recovery_steps = []
        for _ in range(2):
            step = director.step()
            recovery_steps.append(step.to_dict())
        payload = {
            "ok": not assertion_errors,
            "status": "PASS" if not assertion_errors else "FAIL",
            "mode": "REAL ROM + REAL checkpoint/task/shared-memory/branch code (offline fake LLM agents)",
            "state": str(state),
            "assertion_errors": assertion_errors,
            "checkpoint": checkpoint,
            "manifest": manifest,
            "manifest_verify": verify,
            "rollback": rollback,
            "branch_before": checkpoint.get("branch_id"),
            "branch_after": session.current_branch_id,
            "branches": [branch.to_dict() for branch in branches],
            "areas": {"before": pre_area, "after_rollback": post_area},
            "budget": {"before": budget_before, "after": budget_after,
                       "not_rewound": budget_after >= budget_before},
            "knowledge": {"version_before": knowledge_version_before,
                          "version_after": int(session.shared.version),
                          "historical_entries": historical[:10],
                          "retained_entries": retained[:10]},
            "pre_rollback_steps": pre_steps,
            "post_rollback_steps": recovery_steps,
            "tasks": _task_rows(session),
            "duration_s": round(time.time() - started, 1),
        }
        result = _finalize(session, "finished", "acceptance_g5", started)
        payload["run_dir"] = result.run_dir
        ev.record("rollback", payload)
        return ev.finish(payload)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print("[g5] FAILED:", error, flush=True)
        traceback.print_exc()
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
        return ev.finish({"ok": False, "status": "BLOCKED", "reason": error,
                          "traceback": traceback.format_exc()[-4000:]})


# --------------------------------------------------------------------------- #
# G6 — stagnation detection and bounded self-recovery
# --------------------------------------------------------------------------- #

def run_g6(ev: Evidence) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(
        "g6",
        **{
            "budget.max_usd_per_run": 0.01,
            "budget.max_calls_per_run": 5,
            "multi_agent.director_llm_enabled": False,
            "multi_agent.local_stagnation_steps": 2,
            "multi_agent.global_stagnation_seconds": 8.0,
            "multi_agent.no_progress_action_limit": 6,
            "multi_agent.repeated_failure_limit": 2,
            "multi_agent.repeated_message_limit": 2,
            "multi_agent.max_rollbacks_per_branch": 1,
            "multi_agent.max_rollbacks_total": 1,
            "multi_agent.rollback_cooldown_s": 0.0,
            "multi_agent.max_decision_turns": 24,
            "multi_agent.navigation_steps_per_task": 3,
            "multi_agent.checkpoint_every_actions": 1,
            "multi_agent.max_emulator_actions": 400,
            "multi_agent.max_runtime_minutes": 5.0,
        },
    )
    state = _state_path(DEFAULT_STATE)
    session = build_session(cfg, scenario="g6", state_path=state,
                            story_agent=FakeStoryAgent(), pokemon_agent=FakePokemonAgent())
    assertion_errors: list[str] = []
    try:
        session.setup()
        director = session.director
        steps: list[dict[str, Any]] = []
        signals_seen: list[str] = []
        recovery_levels: list[str] = []
        for _ in range(30):
            if director.budget_exhausted() or director._runtime_exceeded():
                break
            # Force a reproducible stagnation: every step claims a nav status
            # with no new cells and feeds many "actions" without progress.
            if session.monitor is not None:
                session.monitor.observe_navigation({
                    "cells_total": 20, "walkable": 10, "unknown_frontier": 0,
                    "areas_total": 1, "coverage_ratio": 0.5, "area_id": "area_001",
                    "localization_ok": False,
                })
                session.monitor.observe_actions(6)
            before_recoveries = director.stats_dict().get("recoveries", 0)
            step = director.step()
            steps.append(step.to_dict())
            if director.stats_dict().get("stagnation_events", 0) > len(signals_seen):
                signals_seen.append(step.detail or "stagnation")
            if director.stats_dict().get("recoveries", 0) > before_recoveries:
                summary = session.recovery.summary()
                level = str(summary.get("last_level") or summary.get("last_recovery", {}).get("level") or "")
                recovery_levels.append(level or "recovery")
            if step.context == "limit":
                break
        result = _finalize(session, "finished", "acceptance_g6", started)
        stats = director.stats_dict()
        monitor_summary = session.monitor.summary() if session.monitor else {}
        recovery_summary = session.recovery.summary() if session.recovery else {}
        branch_count = len(session.checkpoints.list_branches()) if session.checkpoints else 0
        # The scenario must demonstrate: detection, recovery attempt, boundedness.
        safe_pause = any(step.get("context") == "limit" for step in steps) or result.status == "paused_limits"
        if int(stats.get("stagnation_events", 0)) < 1:
            assertion_errors.append("no stagnation was detected")
        if int(stats.get("recoveries", 0)) < 1:
            assertion_errors.append("no recovery action was attempted")
        if int(stats.get("rollbacks", 0)) > 1:
            assertion_errors.append("rollback budget was exceeded (infinite loops)")
        if not safe_pause and int(stats.get("recoveries", 0)) >= 1:
            # a bounded run with a low emulator-action cap should end in a safe pause
            assertion_errors.append("run did not reach a safe pause after recovery")
        payload = {
            "ok": not assertion_errors,
            "status": "PASS" if not assertion_errors else "FAIL",
            "mode": "REAL ROM + REAL monitor/recovery/checkpoint code (offline fake LLM agents)",
            "state": str(state),
            "assertion_errors": assertion_errors,
            "director_steps": steps,
            "stagnation_events": int(stats.get("stagnation_events", 0)),
            "recoveries": int(stats.get("recoveries", 0)),
            "rollbacks": int(stats.get("rollbacks", 0)),
            "recovery_levels": recovery_levels,
            "monitor": monitor_summary,
            "recovery": recovery_summary,
            "branch_count": branch_count,
            "safe_pause": safe_pause,
            "run_result": result.to_dict(),
            "run_dir": result.run_dir,
            "tasks": _task_rows(session),
            "duration_s": round(time.time() - started, 1),
        }
        ev.record("recovery", payload)
        return ev.finish(payload)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print("[g6] FAILED:", error, flush=True)
        traceback.print_exc()
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
        return ev.finish({"ok": False, "status": "BLOCKED", "reason": error,
                          "traceback": traceback.format_exc()[-4000:]})


# --------------------------------------------------------------------------- #
# G7 — integrated real DeepSeek multi-agent run
# --------------------------------------------------------------------------- #

def run_g7(ev: Evidence) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(
        "g7",
        **{
            "budget.max_usd_per_run": 1.20,
            "budget.max_calls_per_run": 320,
            "multi_agent.director_max_steps": 110,
            "multi_agent.max_decision_turns": 110,
            "multi_agent.max_runtime_minutes": 25.0,
            "multi_agent.max_emulator_actions": 20000,
            "multi_agent.navigation_steps_per_task": 80,
            "multi_agent.checkpoint_every_actions": 30,
            "multi_agent.checkpoint_every_minutes": 1.0,
            "multi_agent.director_llm_enabled": True,
            "multi_agent.director_llm_min_interval_s": 60.0,
            "multi_agent.battle_max_turns": 14,
            "multi_agent.dialogue_max_turns": 12,
        },
    )
    state = _state_path(G2_STATE)
    session = build_session(cfg, scenario="g7", state_path=state, warmup=True)
    try:
        holder: dict[str, Any] = {}

        def worker() -> None:
            try:
                holder["result"] = session.run()
            except BaseException as exc:  # noqa: BLE001
                holder["error"] = f"{type(exc).__name__}: {exc}"
                holder["traceback"] = traceback.format_exc()

        thread = threading.Thread(target=worker, name="g7-run", daemon=True)
        thread.start()
        last_snapshot = 0.0
        snapshots: list[dict[str, Any]] = []
        watchdog_deadline = time.time() + float(cfg.multi_agent.max_runtime_minutes) * 60.0 + 420.0
        timed_out = False
        while thread.is_alive():
            now = time.time()
            if now > watchdog_deadline:
                timed_out = True
                try:
                    session.request_stop("watchdog: acceptance time limit exceeded")
                except Exception:  # noqa: BLE001
                    pass
                thread.join(timeout=90)
                break
            if now - last_snapshot >= 20.0:
                last_snapshot = now
                try:
                    snapshots.append(session_snapshot(session, label=f"t{int(now - started)}s"))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1.0)
        thread.join(timeout=60)
        result = holder.get("result") or session.last_result
        if result is None:
            return ev.finish({"ok": False, "status": "BLOCKED",
                              "reason": holder.get("error") or "no result", "traceback": holder.get("traceback", "")})
        stats = session.director.stats_dict() if session.director else {}
        usage = session.router.usage() if session.router else {}
        total = usage.get("total") or {}
        agents = usage.get("agents") or {}
        cost = float(total.get("cost_usd", 0.0))
        real_api = int(total.get("calls", 0)) > 0 and cost > 0.0
        snapshots.append(session_snapshot(session, label="final"))
        screens = ev.copy_screens(Path(result.run_dir) / "multi_agent" / "screens", limit=120, prefix="g7_")
        events = _bus_recent(session, limit=400)
        decisions = _audit_rows(result.run_dir, "agent_results", limit=200)
        decision_kinds = {str(row.get("kind")) for row in decisions}
        active_tasks = [task for task in session.tasks.tasks() if not task.status.terminal]
        active_descriptions = {task.description for task in active_tasks}
        localization_recoveries = int(stats.get("localization_recoveries", 0))
        checks = {
            "real_api_calls": real_api,
            "director_scheduled_multiple_owners": int(stats.get("owner_switches", 0)) >= 2,
            "navigation_ran": int((session.navigation.summary() or {}).get("actions", 0)) > 0,
            "navigation_made_progress": int((session.navigation.status() or {}).get("cells_total", 0)) > 0,
            "story_took_over_dialog": "dialog" in decision_kinds,
            "no_duplicate_active_tasks": len(active_descriptions) == len(active_tasks),
        }
        soft_checks = {
            "pokemon_fought_battle": int(stats.get("battles", 0)) >= 1,
            "battle_decisions": "battle" in decision_kinds,
            "battle_handoff_event": any(str(item.get("event_type")) == "battle_ended" for item in events),
            "dialog_handoff_event": any(str(item.get("event_type")) == "dialog_ended" for item in events),
            "checkpoints": int(stats.get("checkpoints", 0)) > 0,
            "localization_recoveries": localization_recoveries,
        }
        missing = {name: False for name, passed in {**checks, **soft_checks}.items() if not passed}
        core_ok = (real_api and checks["navigation_ran"]
                   and checks["story_took_over_dialog"]
                   and checks["no_duplicate_active_tasks"])
        passed_ok = core_ok and not timed_out
        payload = {
            "ok": passed_ok,
            "status": ("PASS" if passed_ok else ("PARTIAL" if real_api else "BLOCKED")),
            "timed_out": timed_out,
            "checks": checks,
            "soft_checks": soft_checks,
            "missing_checks": missing,
            "mode": "REAL ROM + REAL DeepSeek API + three agents + director",
            "state": str(state),
            "run_result": result.to_dict(),
            "run_dir": result.run_dir,
            "checks": checks,
            "stats": stats,
            "api": {"total": total, "agents": agents},
            "cost_usd": round(cost, 6),
            "snapshots": snapshots,
            "tasks": _task_rows(session),
            "knowledge": _knowledge_rows(session),
            "events": events[-200:],
            "agent_decisions": _audit_rows(result.run_dir, "agent_results", limit=120),
            "screenshots": screens,
            "navigation": session.navigation.summary() if session.navigation else {},
            "monitor": session.monitor.summary() if session.monitor else {},
            "recovery": session.recovery.summary() if session.recovery else {},
            "checkpoints": session.checkpoints.status() if session.checkpoints else {},
            "duration_s": round(time.time() - started, 1),
            "error": holder.get("error", ""),
        }
        ev.record("integrated_run", payload)
        # keep a per-run history so repeated G7 runs are all auditable
        try:
            history_path = ev.dir / "evidence" / "run_history.jsonl"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "run_dir": result.run_dir,
                    "when": time.time(),
                    "result": result.to_dict(),
                    "checks": checks,
                    "soft_checks": soft_checks,
                    "cost_usd": round(cost, 6),
                }, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass
        return ev.finish(payload)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print("[g7] FAILED:", error, flush=True)
        traceback.print_exc()
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
        return ev.finish({"ok": False, "status": "BLOCKED", "reason": error,
                          "traceback": traceback.format_exc()[-4000:]})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

SCENARIOS: dict[str, Callable[..., dict[str, Any]]] = {
    "g0": run_g0,
    "g1": run_g1,
    "g2": run_g2,
    "g3": run_g3,
    "g4": run_g4,
    "g5": run_g5,
    "g6": run_g6,
    "g7": run_g7,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent v1 real acceptance harness (G0-G7).")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), default=None,
                        help="scenario(s) to run (repeatable)")
    parser.add_argument("--all", action="store_true", help="run G0..G7 in order")
    parser.add_argument("--require-api", action="store_true",
                        help="fail early if the DeepSeek API key is missing")
    args = parser.parse_args(argv)
    selected = list(args.scenario or [])
    if args.all:
        selected = ["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7"]
    if not selected:
        parser.print_help()
        return 2
    real_scenarios = {"g2", "g3", "g7"}
    if args.require_api and real_scenarios & set(selected):
        cfg = AgentConfig.load(CONFIG_PATH, env_file=str(PROJECT_ROOT / ".env"))
        if cfg.api_key() is None:
            print("[blocked] DEEPSEEK_API_KEY missing; real-API scenarios cannot be faked.", file=sys.stderr)
            return 4
    from llm_agent.dotenv import load_env_file

    load_env_file(str(PROJECT_ROOT / ".env"))
    reports: dict[str, Any] = {}
    exit_code = 0
    for name in selected:
        print(f"\n===== {name.upper()} =====", flush=True)
        evidence = Evidence(name)
        try:
            report = SCENARIOS[name](evidence)
        except Exception as exc:  # noqa: BLE001 - harness must report, not crash
            report = {"ok": False, "status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-4000:]}
            evidence.finish(report)
        reports[name] = {"ok": bool(report.get("ok")), "status": report.get("status", ""),
                         "reason": report.get("reason", "")}
        if not report.get("ok"):
            exit_code = 1
        print(f"[{name}] {report.get('status')} ok={report.get('ok')} "
              f"{report.get('reason', '')}", flush=True)
    summary = {"generated_at": time.time(), "reports": reports}
    (ARTIFACTS / "acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
