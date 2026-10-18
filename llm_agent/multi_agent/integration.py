"""Integration layer: Navigation v2 adapter + the multi-agent session.

This module is the **only** place where the new multi-agent subsystem touches
Navigation v2.  It deliberately does not re-implement navigation:

* :class:`NavigationAdapter` wraps the existing
  :class:`llm_agent.navigation.navigation_agent.NavigationAgent` with a
  :class:`~llm_agent.multi_agent.control_arbiter.GuardedEmulator`, so every
  navigation press passes through the single-owner control arbiter while the
  verified localizer / grid map / world model / A* / explorer code stays
  untouched;
* :class:`MultiAgentSession` owns the one `GameBoyEmulator`, builds all
  subsystems once (event bus, arbiter, task manager, scheduler, shared memory,
  progress monitor, recovery, branch checkpoints, model router, registry,
  agents, director) and runs the director loop.

Everything is injectable so the session can be exercised in tests with a fake
emulator and fake model replies, while a real run uses PyBoy + DeepSeek.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from llm_agent.budget import BudgetTracker
from llm_agent.config import AgentConfig
from llm_agent.contracts import (
    AgentError,
    ApiAuthenticationError,
    BlockedError,
    BudgetExceededError,
    EmulatorError,
    Frame,
)
from llm_agent.memory import AgentMemory
from llm_agent.multi_agent.agent_registry import AgentRegistry
from llm_agent.multi_agent.contracts import (
    AgentRole,
    BranchRecord,
    BudgetLimitError,
    ControlDeniedError,
    ControlOwner,
    EventType,
    MultiAgentError,
    Priority,
    Task,
    TaskStatus,
    TaskType,
    now_ts,
)
from llm_agent.multi_agent.model_router import ModelRouter, ModelRouterConfig
from llm_agent.observation import analyze, save_png
from llm_agent.runtime import AuditLogger, CheckpointStore, write_json_atomic
from llm_agent.multi_agent.event_bus import EventBus, EventBusConfig
from llm_agent.multi_agent.control_arbiter import ArbiterConfig, ControlArbiter
from llm_agent.multi_agent.task_manager import TaskManager, TaskManagerConfig
from llm_agent.multi_agent.scheduler import SchedulerConfig, TaskScheduler
from llm_agent.multi_agent.shared_memory import SharedMemoryConfig, SharedWorldMemory
from llm_agent.multi_agent.progress_monitor import ProgressMonitor, ProgressMonitorConfig
from llm_agent.multi_agent.recovery import RecoveryConfig, RecoveryHooks, RecoveryManager
from llm_agent.multi_agent.checkpoint import BranchCheckpointConfig, BranchCheckpointManager


# --------------------------------------------------------------------------- #
# Navigation v2 adapter
# --------------------------------------------------------------------------- #

class NavigationAdapter:
    """Thin, arbiter-gated adapter over the existing Navigation v2 facade."""

    def __init__(
        self,
        *,
        emulator: Any,
        arbiter: ControlArbiter,
        cfg: AgentConfig,
        world_root: str | Path,
        bus: EventBus | None = None,
        clock: Callable[[], float] | None = None,
        on_frame: Callable[[Frame], None] | None = None,
        on_step: Callable[[Any], None] | None = None,
        on_status: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.guarded_emulator = emulator
        self.arbiter = arbiter
        self.cfg = cfg
        self.world_root = str(world_root)
        self.bus = bus
        self._clock = clock or time.time
        self._on_frame = on_frame
        self._on_step = on_step
        self._on_status = on_status
        self.agent = None
        self.available = False
        self.setup_error = ""
        self.last_report: dict[str, Any] = {}
        self.last_event: dict[str, Any] = {}
        self.actions = 0
        self._lease = None

    # -- lifecycle ------------------------------------------------------- #
    def attach(self) -> dict[str, Any]:
        """Build (if needed) and attach the NavigationAgent, then relocalize."""
        if self.agent is None:
            try:
                from llm_agent.navigation.navigation_agent import NavigationAgent

                self.agent = NavigationAgent(
                    emulator=self.guarded_emulator,
                    world_root=self.world_root,
                    config=self.cfg,
                    on_frame=self._handle_frame,
                    on_step=self._handle_step,
                    on_status=self._handle_status,
                )
                self.available = True
            except Exception as exc:  # noqa: BLE001 - navigation must never break setup
                self.available = False
                self.setup_error = f"{type(exc).__name__}: {exc}"
                return {"ok": False, "error": self.setup_error}
        try:
            self._request("attach")
            attach_fn = getattr(self.agent, "attach", None)
            if callable(attach_fn):
                attach_fn()
            status = self.status()
            return {"ok": True, "matched": bool(self.agent.last_attach), "attach": self.agent.last_attach, "status": status}
        except ControlDeniedError as exc:
            return {"ok": False, "error": f"control_denied: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._release("attach done")

    def relocalize(self) -> dict[str, Any]:
        """Re-observe after an interruption and restore the world pose."""
        if not self.available or self.agent is None:
            return {"ok": False, "error": "navigation unavailable"}
        try:
            self._request("relocalize")
            status = self.status()
            pose = status.get("pose") or {}
            relocalized = bool(pose.get("valid")) or bool(self.agent.last_attach)
            if not relocalized:
                # try the controller's own relocalization before giving up
                try:
                    match = self.agent.controller.relocate()
                    status = self.status()
                    pose = status.get("pose") or {}
                    relocalized = match is not None or bool(pose.get("valid"))
                except Exception:  # noqa: BLE001 - honest failure report below
                    relocalized = False
            return {
                "ok": bool(relocalized),
                "relocalized": bool(relocalized),
                "status": status,
                "attach": self.agent.last_attach,
                "pose": pose,
                "area_id": status.get("area_id", ""),
            }
        except ControlDeniedError as exc:
            return {"ok": False, "error": f"control_denied: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._release("relocalize done")

    def recover_localization(self) -> dict[str, Any]:
        """Recover from ``localization_lost`` by re-attaching to the world model.

        The controller invalidates its pose after repeated UNKNOWN motion
        estimates; ``NavigationAgent.attach()`` re-observes the real screen and
        either relocalizes in a known area or starts a new area at its origin.
        This is a Navigation-v2-provided operation — the localizer/controller
        algorithms are untouched.
        """
        if not self.available or self.agent is None:
            return {"ok": False, "error": "navigation unavailable"}
        try:
            self._request("recover_localization")
            recovered = False
            match = None
            # Prefer matching a known area over creating a new one: repeated
            # re-attach after every UNKNOWN streak would otherwise fragment the
            # world model into many spurious areas.
            try:
                controller = getattr(self.agent, "controller", None)
                if controller is not None:
                    match = controller.relocate()
                    recovered = match is not None
            except Exception:  # noqa: BLE001
                recovered = False
            if not recovered:
                self.agent.attach()
                status = self.status()
                pose = status.get("pose") or {}
                recovered = bool(pose.get("valid"))
            status = self.status()
            return {"ok": bool(recovered), "recovered": bool(recovered),
                    "matched": match is not None, "status": status,
                    "attach": self.agent.last_attach}
        except ControlDeniedError as exc:
            return {"ok": False, "error": f"control_denied: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._release("recover_localization done")

    def handle_transition(self, *, max_wait_frames: int = 120) -> dict[str, Any]:
        if not self.available or self.agent is None:
            return {"ok": False, "error": "navigation unavailable"}
        try:
            self._request("transition")
            controller = getattr(self.agent, "controller", None)
            result = controller.handle_transition(max_wait_frames=max_wait_frames) if controller else {}
            status = self.status()
            return {"ok": True, "result": result, "status": status}
        except ControlDeniedError as exc:
            return {"ok": False, "error": f"control_denied: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._release("transition handled")

    # -- high-level navigation actions ----------------------------------- #
    def explore(self, *, max_steps: int | None = None, max_seconds: float | None = None) -> dict[str, Any]:
        if not self.available or self.agent is None:
            return {"ok": False, "error": "navigation unavailable", "status": "error"}
        multi = self.cfg.multi_agent
        steps = int(max_steps if max_steps is not None else multi.navigation_steps_per_task)
        seconds = float(max_seconds if max_seconds is not None else self.cfg.navigation.max_explore_seconds)
        started = self._clock()
        try:
            self._request("explore")
            blocked = self._prepare_navigation()
            if blocked is not None:
                return self._interrupted_report(blocked, "explore", started)
            report = self.agent.explore_current_area(max_steps=steps, max_seconds=seconds)
            data = self._report_dict(report, "explore", started)
            self.actions += int(data.get("steps") or 0)
            self.last_report = data
            return data
        except ControlDeniedError as exc:
            self.actions += 0
            return {"ok": False, "error": f"control_denied: {exc}", "status": "control_denied"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "status": "error"}
        finally:
            self._release("explore done")

    def navigate_to(self, target: Mapping[str, Any], *, max_steps: int | None = None) -> dict[str, Any]:
        if not self.available or self.agent is None:
            return {"ok": False, "error": "navigation unavailable", "status": "error"}
        multi = self.cfg.multi_agent
        steps = int(max_steps if max_steps is not None else multi.navigation_steps_per_task)
        started = self._clock()
        try:
            self._request("navigate")
            blocked = self._prepare_navigation()
            if blocked is not None:
                return self._interrupted_report(blocked, "navigate", started)
            report = self.agent.navigate_to(_normalize_nav_target(target), max_steps=steps)
            data = self._report_dict(report, "navigate", started)
            self.actions += int(data.get("steps") or 0)
            self.last_report = data
            return data
        except ControlDeniedError as exc:
            return {"ok": False, "error": f"control_denied: {exc}", "status": "control_denied"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "status": "error"}
        finally:
            self._release("navigate done")

    def _prepare_navigation(self) -> Any | None:
        """Resume after an interruption and refresh the controller's baseline view.

        Navigation pauses itself when a dialogue/battle/menu interrupts a step.
        Before the next exploration we must (a) clear that pause and (b) refresh
        the visual baseline, otherwise the controller keeps reporting the stale
        blocking screen forever (a real bug found in the G2 acceptance run).
        Returns the blocking :class:`NavEvent` when the screen still blocks.
        """
        try:
            if bool(getattr(self.agent, "paused", False)):
                self.agent.resume()
            controller = getattr(self.agent, "controller", None)
            if controller is None:
                return None
            view = controller.observe()
            blocker = controller.screen_blocks_navigation(view)
            return blocker
        except Exception:  # noqa: BLE001 - navigation heals itself in its own loop
            return None

    def _interrupted_report(self, nav_event: Any, kind: str, started: float) -> dict[str, Any]:
        event_dict: dict[str, Any] = {}
        try:
            event_dict = nav_event.to_dict() if hasattr(nav_event, "to_dict") else {"raw": str(nav_event)}
        except Exception:  # noqa: BLE001
            event_dict = {}
        if event_dict:
            self.last_event = event_dict
            self._publish_event(event_dict)
        return {
            "ok": False,
            "status": "interrupted_event",
            "kind": kind,
            "report": {"steps": 0, "moves": 0, "blocked": 0},
            "nav_event": event_dict,
            "navigation_status": self.status(),
            "steps": 0,
            "moves": 0,
            "blocked": 0,
            "duration_s": round(self._clock() - started, 3),
        }

    # -- read-only surfaces ---------------------------------------------- #
    def status(self) -> dict[str, Any]:
        if self.agent is None:
            return {"available": False, "error": self.setup_error}
        try:
            return dict(self.agent.status() or {})
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def summary(self) -> dict[str, Any]:
        status = self.status()
        return {
            "available": self.available,
            "setup_error": self.setup_error,
            "area_id": status.get("area_id", ""),
            "pose": status.get("pose", {}),
            "coverage_ratio": status.get("coverage_ratio", 0.0),
            "cells_total": status.get("cells_total", 0),
            "walkable": status.get("walkable", 0),
            "areas_total": status.get("areas_total", 0),
            "actions": self.actions,
            "paused": bool(status.get("paused")),
            "last_event": self.last_event or status.get("last_event") or {},
        }

    def local_map(self, *, radius: int = 12) -> dict[str, Any]:
        """Read-only local grid map (delegates to the existing NavigationAgent)."""
        if self.agent is None:
            return {"ascii": "", "cells": [], "coverage": {}}
        try:
            return dict(self.agent.local_map(radius=int(radius)) or {})
        except Exception as exc:  # noqa: BLE001 - map rendering must never break the director
            return {"ascii": "", "cells": [], "coverage": {}, "error": f"{type(exc).__name__}: {exc}"}

    def world_summary(self) -> dict[str, Any]:
        if self.agent is None:
            return {}
        try:
            return dict(self.agent.world_summary() or {})
        except Exception:  # noqa: BLE001
            return {}

    def snapshot(self) -> dict[str, Any]:
        return {"status": self.status(), "last_report": dict(self.last_report), "actions": self.actions}

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        if not snapshot:
            return
        self.actions = max(self.actions, int(snapshot.get("actions") or 0))

    def pause(self) -> None:
        if self.agent is not None:
            try:
                self.agent.pause()
            except Exception:  # noqa: BLE001
                pass

    def resume(self) -> None:
        if self.agent is not None:
            try:
                self.agent.resume()
            except Exception:  # noqa: BLE001
                pass

    def save(self) -> None:
        if self.agent is not None:
            try:
                self.agent.save()
            except Exception:  # noqa: BLE001
                pass

    # -- internals ------------------------------------------------------- #
    def _request(self, reason: str) -> Any:
        self._lease = self.arbiter.request(
            AgentRole.NAVIGATION, reason=reason, context={"navigation": reason}
        )
        return self._lease

    def _release(self, reason: str) -> None:
        try:
            self.arbiter.release(AgentRole.NAVIGATION, reason=reason, context={"navigation": reason})
        except Exception:  # noqa: BLE001 - release must never raise in a finally
            pass
        self._lease = None

    def _report_dict(self, report: Any, kind: str, started: float) -> dict[str, Any]:
        try:
            payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        except Exception:  # noqa: BLE001
            payload = {"raw": str(report)}
        status = str(payload.get("status") or "")
        reason = str(payload.get("reason") or "")
        error = str(payload.get("error") or "")
        if not error and status in ("error", "blocked", "no_progress", "localization_lost"):
            error = reason
        nav_event: dict[str, Any] = {}
        try:
            event = self.agent.last_event if self.agent is not None else None
            if event is not None:
                nav_event = event.to_dict() if hasattr(event, "to_dict") else {"raw": str(event)}
        except Exception:  # noqa: BLE001
            nav_event = {}
        if nav_event:
            self.last_event = nav_event
            self._publish_event(nav_event)
        return {
            "ok": status in ("completed", "step_limit", "time_limit"),
            "status": status,
            "kind": kind,
            "report": payload,
            "nav_event": nav_event,
            "navigation_status": self.status(),
            "duration_s": round(self._clock() - started, 3),
            "steps": int(payload.get("steps") or 0),
            "moves": int(payload.get("moves") or 0),
            "blocked": int(payload.get("blocked") or 0),
            "coverage": payload.get("coverage"),
            "areas": payload.get("areas"),
            "reason": reason,
            "error": error,
        }

    def _publish_event(self, nav_event: Mapping[str, Any]) -> None:
        if self.bus is None:
            return
        kind = str(nav_event.get("kind") or "")
        mapping = {
            "battle_event": EventType.BATTLE_STARTED,
            "dialog_event": EventType.DIALOG_STARTED,
            "menu_event": EventType.DIALOG_STARTED,
            "transition_event": EventType.MAP_TRANSITION,
        }
        event_type = mapping.get(kind)
        if event_type is None:
            return
        try:
            self.bus.publish(
                self.bus_message(event_type, payload={"area_id": nav_event.get("area_id", ""), "source": "navigation"})
            )
        except Exception:  # noqa: BLE001 - bus problems must not break navigation
            pass

    def bus_message(self, event_type: EventType, *, payload: Mapping[str, Any] | None = None) -> Any:
        from llm_agent.multi_agent.contracts import AgentMessage

        return AgentMessage.create(
            event_type,
            source_agent=AgentRole.NAVIGATION,
            target_agent=AgentRole.DIRECTOR,
            payload=dict(payload or {}),
            priority=Priority.HIGH,
        )

    # -- callbacks ------------------------------------------------------- #
    def _handle_frame(self, frame: Frame) -> None:
        if self._on_frame is not None:
            try:
                self._on_frame(frame)
            except Exception:  # noqa: BLE001
                pass

    def _handle_step(self, step: Any) -> None:
        if self._on_step is not None:
            try:
                self._on_step(step)
            except Exception:  # noqa: BLE001
                pass

    def _handle_status(self, status: Any) -> None:
        if self._on_status is not None:
            try:
                self._on_status(dict(status or {}))
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Run result
# --------------------------------------------------------------------------- #

def _normalize_nav_target(target: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert the LLM-friendly flat target form to the frozen Navigation v2 form.

    The director's tasks store targets as ``{"kind": "cell", "x": 3, "y": 4}``
    (same shape as the DeepSeek navigation tools); Navigation v2 consumes
    ``{"cell": [x, y]}`` / ``{"exit": True}`` / … (see NAV_CONTRACT.md).
    """
    if not isinstance(target, Mapping):
        return {}
    data = dict(target)
    kind = str(data.get("kind") or "").strip().lower()
    if not kind:
        # already in contract form
        return data
    if kind == "cell":
        try:
            return {"cell": [int(data.get("x", 0)), int(data.get("y", 0))]}
        except Exception:  # noqa: BLE001
            return {}
    if kind == "exit":
        return {"exit": True}
    if kind == "landmark":
        return {"landmark": str(data.get("landmark") or "")}
    if kind == "return_to_entry":
        return {"return_to_entry": True}
    if kind == "area":
        normalized: dict[str, Any] = {"area": str(data.get("area_id") or "")}
        if data.get("x") is not None and data.get("y") is not None:
            normalized["cell"] = [int(data["x"]), int(data["y"])]
        return normalized
    return data

@dataclass
class MultiAgentResult:
    status: str
    reason: str
    exit_code: int
    run_dir: str
    duration_s: float
    steps: int = 0
    owner_switches: int = 0
    battles: int = 0
    dialogs: int = 0
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    checkpoints: int = 0
    rollbacks: int = 0
    stagnation_events: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0
    total_tokens: int = 0
    director_llm_calls: int = 0
    branch_id: str = ""
    current_area: str = ""
    frames: int = 0
    buttons: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "exit_code": self.exit_code,
            "run_dir": self.run_dir,
            "duration_s": round(self.duration_s, 3),
            "steps": self.steps,
            "owner_switches": self.owner_switches,
            "battles": self.battles,
            "dialogs": self.dialogs,
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "checkpoints": self.checkpoints,
            "rollbacks": self.rollbacks,
            "stagnation_events": self.stagnation_events,
            "api_calls": self.api_calls,
            "cost_usd": round(self.cost_usd, 8),
            "total_tokens": self.total_tokens,
            "director_llm_calls": self.director_llm_calls,
            "branch_id": self.branch_id,
            "current_area": self.current_area,
            "frames": self.frames,
            "buttons": self.buttons,
            "extra": dict(self.extra),
        }


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #

class MultiAgentSession:
    """Builds and runs one multi-agent session on a single PyBoy emulator."""

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        run_id_hint: str | None = None,
        emulator: Any | None = None,
        router: ModelRouter | None = None,
        viewer_bridge: Any | None = None,
        warmup: bool = True,
        use_lock: bool = True,
        clock: Callable[[], float] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        client_factory: Callable[[Any, AgentRole], Any] | None = None,
        story_agent: Any | None = None,
        pokemon_agent: Any | None = None,
        navigation_adapter: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_id_hint = run_id_hint
        self._clock = clock or time.time
        self.viewer_bridge = viewer_bridge
        self._on_event = on_event
        self.warmup = bool(warmup)
        self.use_lock = bool(use_lock)
        self._client_factory = client_factory
        self._injected_story = story_agent
        self._injected_pokemon = pokemon_agent
        self._injected_navigation = navigation_adapter

        self.run_dir: Path | None = None
        self.emulator = emulator
        self.audit: AuditLogger | None = None
        self.budget: BudgetTracker | None = None
        self.bus: EventBus | None = None
        self.arbiter: ControlArbiter | None = None
        self.tasks: TaskManager | None = None
        self.scheduler: TaskScheduler | None = None
        self.shared: SharedWorldMemory | None = None
        self.monitor: ProgressMonitor | None = None
        self.checkpoints: BranchCheckpointManager | None = None
        self.checkpoint_store: CheckpointStore | None = None
        self.recovery: RecoveryManager | None = None
        self.router: ModelRouter | None = router
        self.registry: AgentRegistry | None = None
        self.navigation: NavigationAdapter | None = None
        self.story: Any | None = None
        self.pokemon: Any | None = None
        self.director: Any | None = None
        self.story_memory: AgentMemory | None = None
        self.pokemon_memory: AgentMemory | None = None
        self.branch: BranchRecord | None = None
        #: branch id used for new tasks/knowledge/messages (updated on rollback)
        self.current_branch_id: str = ""
        self._lock: Any | None = None
        self._started_at = self._clock()
        self._stop_requested = False
        self._closed = False
        self.last_result: MultiAgentResult | None = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        cfg = self.cfg
        problems = cfg.validate()
        if problems:
            raise BlockedError("invalid configuration: " + "; ".join(problems))
        from llm_agent.dotenv import load_env_file

        if cfg.env_file:
            load_env_file(cfg.env_file)
        if cfg.api_key() is None:
            raise BlockedError(
                f"{cfg.api.api_key_env} is not configured; the multi-agent system needs the real API. "
                "No mock result is ever reported as acceptance evidence."
            )
        run_dir = cfg.resolve_paths(self.run_id_hint)
        self.run_dir = run_dir
        multi = cfg.multi_agent
        ma_root = Path(multi.artifact_root or (run_dir / "multi_agent"))
        ma_root.mkdir(parents=True, exist_ok=True)

        if self.use_lock:
            from llm_agent.runtime import SingleInstanceLock

            self._lock = SingleInstanceLock(cfg.runtime.lock_path)
            self._lock.acquire(run_id=run_dir.name)

        self.audit = AuditLogger(run_dir / "audit", cfg.runtime.audit_rotation_bytes)
        self.audit.log(
            "events",
            {
                "event": "multi_agent_start",
                "run_id": run_dir.name,
                "model": cfg.api.model,
                "models": {
                    "director": multi.director_model or cfg.api.model,
                    "story": multi.story_model or cfg.api.model,
                    "pokemon": multi.pokemon_model or cfg.api.model,
                },
                "pid": os.getpid(),
                "rom": cfg.emulator.rom,
                "state": cfg.emulator.state,
            },
        )

        # -- budget (shared, persistent; never reset by rollbacks) ------- #
        global_budget = Path(__file__).resolve().parents[2] / "artifacts" / "llm_agent" / "budget_total.json"
        self.budget = BudgetTracker(
            cfg.budget,
            cfg.api,
            run_dir / "audit" / "budget.json",
            total_path=global_budget,
        )
        self.budget.load()

        # -- emulator ----------------------------------------------------- #
        if self.emulator is None:
            from llm_agent.emulator import GameBoyEmulator

            self.emulator = GameBoyEmulator(cfg.emulator).start()
        self._attach_viewer_hook()

        # -- event bus / arbiter ----------------------------------------- #
        bus_path = str(multi.bus_path or (ma_root / "event_bus.jsonl"))
        self.bus = EventBus(EventBusConfig(path=bus_path, persist=bool(multi.log_messages), dedupe_window=5000))
        self.arbiter = ControlArbiter(self.emulator, config=ArbiterConfig(), event_bus=self.bus)

        # -- branch bookkeeping ------------------------------------------ #
        self.checkpoint_store = CheckpointStore(run_dir / "checkpoints", cfg.runtime.checkpoint_keep)
        self.checkpoints = BranchCheckpointManager(
            BranchCheckpointConfig(
                directory=str(run_dir / "checkpoints"),
                state_dir=str(ma_root / "branches"),
                keep=int(cfg.runtime.checkpoint_keep),
            ),
            store=self.checkpoint_store,
        )
        self.branch = self.checkpoints.current_branch
        branch_id = self.branch.branch_id
        self.current_branch_id = branch_id
        self.bus.set_branch(branch_id)

        # -- shared memory / tasks / scheduler / monitor ----------------- #
        self.shared = SharedWorldMemory(
            SharedMemoryConfig(root=str(multi.shared_memory_root or (ma_root / "shared_memory"))),
            root=str(multi.shared_memory_root or (ma_root / "shared_memory")),
            branch_id=branch_id,
        )
        self.shared.load()
        self.tasks = TaskManager(
            TaskManagerConfig(max_attempts_per_task=3, max_active_tasks=int(multi.max_task_queue)),
            path=str(ma_root / "tasks.json"),
            branch_id=branch_id,
        )
        self.tasks.load()
        self.scheduler = TaskScheduler(SchedulerConfig())
        self.monitor = ProgressMonitor(
            ProgressMonitorConfig(
                local_stagnation_steps=int(multi.local_stagnation_steps),
                global_stagnation_seconds=float(multi.global_stagnation_seconds),
                repeated_failure_limit=int(multi.repeated_failure_limit),
                repeated_message_limit=int(multi.repeated_message_limit),
                no_progress_action_limit=int(multi.no_progress_action_limit),
                training_grace_seconds=float(multi.training_grace_seconds),
                max_runtime_s=float(multi.max_runtime_minutes * 60.0),
                max_actions=int(multi.max_emulator_actions),
                recovery_limit=4,
            )
        )

        # -- model router ------------------------------------------------- #
        router_config = ModelRouterConfig.from_agent_config(cfg)
        router_config.ledger_path = str(ma_root / "model_usage.json")
        router_config.director_min_interval_s = float(multi.director_llm_min_interval_s)
        if self.router is None:
            self.router = ModelRouter(
                cfg,
                config=router_config,
                budget=self.budget,
                client_factory=self._client_factory,
                on_attempt=self._on_api_attempt,
            )

        # -- recovery (hooks are lazy closures; director assigned later) -- #
        self.recovery = RecoveryManager(
            RecoveryConfig(
                max_rollbacks_per_branch=int(multi.max_rollbacks_per_branch),
                max_rollbacks_total=int(multi.max_rollbacks_total),
                max_l2_reviews=int(multi.max_l2_reviews),
                rollback_cooldown_s=float(multi.rollback_cooldown_s),
                max_consecutive_failures=int(multi.max_consecutive_failures),
            ),
            hooks=RecoveryHooks(
                save_checkpoint=self._recovery_save_checkpoint,
                restore_checkpoint=self._recovery_restore_checkpoint,
                relocalize=self._recovery_relocalize,
                director_review=self._recovery_director_review,
                notify=self._recovery_notify,
                agent_hint=self._recovery_agent_hint,
            ),
        )

        # -- agents ------------------------------------------------------- #
        self.registry = AgentRegistry()
        # keep the recovery manager's branch bookkeeping aligned from the start
        try:
            setter = getattr(self.recovery, "set_current_branch", None)
            if callable(setter):
                setter(self.current_branch_id)
        except Exception:  # noqa: BLE001
            pass
        screens = ma_root / "screens"
        screens.mkdir(parents=True, exist_ok=True)
        if self._injected_navigation is not None:
            self.navigation = self._injected_navigation
        else:
            self.navigation = NavigationAdapter(
                emulator=self.arbiter.proxy(AgentRole.NAVIGATION),
                arbiter=self.arbiter,
                cfg=cfg,
                world_root=cfg.navigation.world_root or (run_dir / "navigation"),
                bus=self.bus,
                clock=self._clock,
                on_frame=self._on_agent_frame,
            )
        attach_fn = getattr(self.navigation, "attach", None)
        nav_attach = attach_fn() if callable(attach_fn) else {"ok": True}
        self.registry.register(AgentRole.NAVIGATION, self.navigation, note=f"attached={nav_attach.get('ok')}")

        agent_memory_root = Path(multi.agent_memory_root or (ma_root / "agents"))
        self.story_memory = self._make_agent_memory(agent_memory_root / "story")
        self.pokemon_memory = self._make_agent_memory(agent_memory_root / "pokemon")
        self.story = self._build_story_agent(ma_root, agent_memory_root)
        self.pokemon = self._build_pokemon_agent(ma_root, agent_memory_root)
        if self.story is not None:
            self.registry.register(AgentRole.STORY, self.story)
        if self.pokemon is not None:
            self.registry.register(AgentRole.POKEMON, self.pokemon)

        # -- director ----------------------------------------------------- #
        from llm_agent.multi_agent.director import DirectorConfig, GameDirector

        self.director = GameDirector(
            cfg,
            config=DirectorConfig.from_agent_config(cfg),
            emulator=self.emulator,
            bus=self.bus,
            arbiter=self.arbiter,
            tasks=self.tasks,
            scheduler=self.scheduler,
            shared=self.shared,
            monitor=self.monitor,
            recovery=self.recovery,
            checkpoints=self.checkpoints,
            checkpoint_store=self.checkpoint_store,
            checkpoint_saver=self._save_checkpoint,
            checkpoint_restorer=self._restore_checkpoint_snapshot,
            router=self.router,
            registry=self.registry,
            navigation=self.navigation,
            story=self.story,
            pokemon=self.pokemon,
            audit=self.audit,
            clock=self._clock,
            on_status=self._on_director_status,
            on_event=self._publish_user_event,
            on_branch_changed=self.switch_branch,
            run_dir=run_dir,
        )

        # -- warmup (real API round-trip; never mocked) ------------------- #
        if self.warmup:
            self._warmup_api()

        self._write_session_state("running")
        self.audit.log(
            "events",
            {
                "event": "multi_agent_ready",
                "branch_id": branch_id,
                "navigation": nav_attach,
                "agents": list(self.registry.available()),
                "api_key": cfg.api_key_status(),
            },
        )

    def _make_agent_memory(self, root: Path) -> AgentMemory:
        root.mkdir(parents=True, exist_ok=True)
        mem_cfg = replace(self.cfg.memory, root=str(root))
        memory = AgentMemory(mem_cfg, run_dir=root / "run")
        try:
            memory.load()
        except Exception:  # noqa: BLE001
            pass
        return memory

    def _build_story_agent(self, ma_root: Path, agent_memory_root: Path) -> Any:
        if self._injected_story is not None:
            return self._injected_story
        try:
            from llm_agent.story.agent import StoryAgent
        except Exception as exc:  # noqa: BLE001 - report honestly, keep director alive
            self._log("story_agent_unavailable", error=f"{type(exc).__name__}: {exc}")
            return None
        try:
            return StoryAgent(
                router=self.router,
                bus=self.bus,
                arbiter=self.arbiter,
                shared_memory=self.shared,
                memory=self.story_memory,
                config=self.cfg,
                screen_dir=ma_root / "screens" / "story",
                clock=self._clock,
            )
        except Exception as exc:  # noqa: BLE001
            self._log("story_agent_build_failed", error=f"{type(exc).__name__}: {exc}")
            return None

    def _build_pokemon_agent(self, ma_root: Path, agent_memory_root: Path) -> Any:
        if self._injected_pokemon is not None:
            return self._injected_pokemon
        try:
            from llm_agent.pokemon.agent import PokemonAgent
        except Exception as exc:  # noqa: BLE001
            self._log("pokemon_agent_unavailable", error=f"{type(exc).__name__}: {exc}")
            return None
        try:
            return PokemonAgent(
                router=self.router,
                bus=self.bus,
                arbiter=self.arbiter,
                shared_memory=self.shared,
                memory=self.pokemon_memory,
                config=self.cfg,
                screen_dir=ma_root / "screens" / "pokemon",
                clock=self._clock,
            )
        except Exception as exc:  # noqa: BLE001
            self._log("pokemon_agent_build_failed", error=f"{type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(self) -> MultiAgentResult:
        started = self._clock()
        status, reason, exit_code = "error", "", 1
        try:
            if self.director is None:
                self.setup()
            result = self.director.run() if self.director is not None else None
            if result is not None:
                status = str(getattr(result, "status", "finished"))
                reason = str(getattr(result, "reason", ""))
                exit_code = int(getattr(result, "exit_code", 0))
        except StopRequestedError as exc:
            status, reason, exit_code = "stopped", str(exc), 0
        except BudgetExceededError as exc:
            status, reason, exit_code = "paused_budget", str(exc), 0
        except BlockedError as exc:
            status, reason, exit_code = "blocked", str(exc), 3
        except KeyboardInterrupt:
            status, reason, exit_code = "stopped", "interrupted by user", 0
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            status, reason, exit_code = "error", f"{type(exc).__name__}: {exc}", 1
            if self.audit is not None:
                self.audit.log("errors", {"kind": "fatal", "error": reason, "traceback": traceback.format_exc()[-4000:]})
        finally:
            result = self._finish(status, reason, exit_code, self._clock() - started)
        return result

    def request_stop(self, reason: str = "user request") -> None:
        self._stop_requested = True
        if self.director is not None:
            try:
                self.director.request_stop(reason)
            except Exception:  # noqa: BLE001
                pass

    def _finish(self, status: str, reason: str, exit_code: int, duration: float) -> MultiAgentResult:
        cfg = self.cfg
        multi = cfg.multi_agent
        # final checkpoint (best effort) + persistence
        if self.emulator is not None:
            try:
                self._save_checkpoint(f"final:{status}")
            except Exception as exc:  # noqa: BLE001
                self._log("final_checkpoint_failed", error=f"{type(exc).__name__}: {exc}")
        for name, saver in (
            ("navigation", self.navigation.save if self.navigation else None),
            ("tasks", self.tasks.save if self.tasks else None),
            ("shared_memory", self.shared.save if self.shared else None),
            ("story_memory", self.story_memory.save if self.story_memory else None),
            ("pokemon_memory", self.pokemon_memory.save if self.pokemon_memory else None),
            ("bus", self.bus.save if self.bus else None),
        ):
            if saver is None:
                continue
            try:
                saver()
            except Exception as exc:  # noqa: BLE001
                self._log(f"{name}_save_failed", error=f"{type(exc).__name__}: {exc}")

        usage = self.router.usage() if self.router is not None else {"total": {}, "agents": {}}
        total = usage.get("total") or {}
        director_usage = (usage.get("agents") or {}).get("director") or {}
        director_stats = {}
        try:
            director_stats = self.director.stats_dict() if self.director is not None else {}
        except Exception:  # noqa: BLE001
            director_stats = {}
        tasks_stats = self.tasks.stats() if self.tasks is not None else {}
        result = MultiAgentResult(
            status=status,
            reason=reason,
            exit_code=exit_code,
            run_dir=str(self.run_dir or ""),
            duration_s=float(duration),
            steps=int(director_stats.get("steps", 0)),
            owner_switches=int(director_stats.get("owner_switches", 0)),
            battles=int(director_stats.get("battles", 0)),
            dialogs=int(director_stats.get("dialogs", 0)),
            tasks_created=int(tasks_stats.get("total", 0)),
            tasks_completed=int(tasks_stats.get("completed", 0)),
            tasks_failed=int(tasks_stats.get("failed", 0)),
            checkpoints=int(director_stats.get("checkpoints", 0)),
            rollbacks=int(director_stats.get("rollbacks", 0)),
            stagnation_events=int(director_stats.get("stagnation_events", 0)),
            api_calls=int(total.get("calls", 0)),
            cost_usd=float(total.get("cost_usd", 0.0)),
            total_tokens=int(total.get("total_tokens", 0)),
            director_llm_calls=int(director_usage.get("calls", 0)),
            branch_id=str(self.current_branch_id or (self.branch.branch_id if self.branch else "")),
            current_area=str((self.navigation.summary().get("area_id") if self.navigation else "") or ""),
            frames=int(getattr(self.emulator, "frame_count", 0) or 0),
            buttons=int(director_stats.get("buttons", 0)),
            extra={
                "monitor": self.monitor.summary() if self.monitor else {},
                "shared_memory": self.shared.stats() if self.shared else {},
                "recovery": self.recovery.summary() if self.recovery else {},
                "api_usage": usage,
                "navigation": self.navigation.summary() if self.navigation else {},
                "task_queue": self.director.task_queue_snapshot() if self.director is not None else [],
            },
        )
        if self.audit is not None:
            self.audit.log("events", {"event": "multi_agent_finished", "status": status, "reason": reason,
                                      "exit_code": exit_code, "duration_s": round(duration, 2),
                                      "result": result.to_dict()})
        self._write_result(result)
        self._write_session_state(status)
        self.last_result = result
        self.close()
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.emulator is not None:
            try:
                self.emulator.close()
            except Exception:  # noqa: BLE001
                pass
        if self.audit is not None:
            try:
                self.audit.close()
            except Exception:  # noqa: BLE001
                pass
        if self._lock is not None:
            try:
                self._lock.release()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # Status / reporting
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        director_status: dict[str, Any] = {}
        if self.director is not None:
            try:
                director_status = self.director.status()
            except Exception:  # noqa: BLE001
                director_status = {}
        usage = self.router.usage() if self.router is not None else {"total": {}, "agents": {}}
        return {
            "run_dir": str(self.run_dir or ""),
            "branch_id": str(self.current_branch_id or (self.branch.branch_id if self.branch else "")),
            "director": director_status,
            "navigation": self.navigation.summary() if self.navigation else {},
            "tasks": self.tasks.stats() if self.tasks else {},
            "task_queue": self.director.task_queue_snapshot() if self.director is not None else [],
            "shared_memory": self.shared.stats() if self.shared else {},
            "progress": self.monitor.summary() if self.monitor else {},
            "recovery": self.recovery.summary() if self.recovery else {},
            "checkpoints": self.checkpoints.status() if self.checkpoints else {},
            "api": usage,
            "agents": self.registry.status() if self.registry else {},
            "elapsed_s": round(self._clock() - self._started_at, 1),
        }

    def status_lines(self) -> list[str]:
        """Compact multi-agent status block for the existing Tk viewer."""
        try:
            status = self.status()
        except Exception:  # noqa: BLE001
            return []
        director = status.get("director") or {}
        nav = status.get("navigation") or {}
        api = (status.get("api") or {}).get("total") or {}
        queue = status.get("task_queue") or []
        lines = [
            f"Owner: {director.get('owner', '-')}   Task: {str(director.get('current_task') or '-')[:34]}",
            f"Branch: {str(status.get('branch_id') or '')[-10:]}   Area: {nav.get('area_id', '-')}",
            f"Queue: {len(queue)}   Done: {director.get('tasks_completed', 0)}  Failed: {director.get('tasks_failed', 0)}",
            f"Battles: {director.get('battles', 0)}  Dialogs: {director.get('dialogs', 0)}  Saves: {director.get('checkpoints', 0)}  Rollbacks: {director.get('rollbacks', 0)}",
            f"API: {api.get('calls', 0)} calls  ${float(api.get('cost_usd', 0.0)):.4f}  tokens {int(api.get('total_tokens', 0)):,}",
        ]
        last_event = director.get("last_event") or {}
        if last_event:
            lines.append(f"Last event: {str(last_event.get('event') or last_event)[:60]}")
        progress = (status.get("progress") or {}).get("last_progress_detail")
        if progress:
            lines.append(f"Progress: {str(progress)[:60]}")
        return lines

    # ------------------------------------------------------------------ #
    # Branch bookkeeping
    # ------------------------------------------------------------------ #
    def switch_branch(
        self,
        branch_id: str,
        *,
        previous_branch_id: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Propagate a new branch id to tasks / knowledge / bus / recovery.

        Rollbacks rewind the game but learned experience is kept: the previous
        branch's knowledge entries are marked HISTORICAL by the shared memory,
        and new tasks/messages are tagged with the new branch id.
        """
        new_branch = str(branch_id or "").strip()
        if not new_branch:
            return {"ok": False, "error": "empty branch id"}
        previous = str(previous_branch_id or "").strip() or self.current_branch_id
        result: dict[str, Any] = {"ok": True, "requested_branch_id": new_branch,
                                  "previous_branch_id": previous}
        # The checkpoint manager is the single source of truth for branch ids: let
        # it adopt/materialise the requested branch via a post-rollback checkpoint
        # and use the id it actually persisted for tasks/bus/knowledge/recovery.
        actual_branch = new_branch
        if self.checkpoints is not None:
            try:
                previous_record = None
                for branch in self.checkpoints.list_branches():
                    if str(branch.branch_id) == previous:
                        previous_record = branch
                        break
                if previous_record is not None and not previous_record.closed_at:
                    self.checkpoints.mark_rolled_back(previous, reason=reason or "rollback")
            except Exception as exc:  # noqa: BLE001
                result.setdefault("errors", []).append(f"mark_rolled_back: {exc}")
            try:
                checkpoint = self._save_checkpoint(f"post_rollback:{str(reason)[:80]}", branch_id=new_branch)
                result["checkpoint"] = checkpoint
                result["checkpoint_id"] = str(checkpoint.get("checkpoint_id") or "")
                actual_branch = str(checkpoint.get("branch_id") or new_branch)
            except Exception as exc:  # noqa: BLE001
                result.setdefault("errors", []).append(f"post_rollback_checkpoint: {exc}")
        self.current_branch_id = actual_branch
        result["branch_id"] = actual_branch
        if self.tasks is not None:
            try:
                self.tasks.set_branch(actual_branch)
            except Exception as exc:  # noqa: BLE001
                result.setdefault("errors", []).append(f"tasks.set_branch: {exc}")
        if self.bus is not None:
            try:
                self.bus.set_branch(actual_branch)
            except Exception as exc:  # noqa: BLE001
                result.setdefault("errors", []).append(f"bus.set_branch: {exc}")
        if self.shared is not None:
            try:
                marked = self.shared.set_branch(actual_branch, previous_branch_id=previous, reason=reason)
                result["historical_entries"] = int(marked or 0)
            except Exception as exc:  # noqa: BLE001
                result.setdefault("errors", []).append(f"shared.set_branch: {exc}")
        if self.recovery is not None:
            setter = getattr(self.recovery, "set_current_branch", None)
            if callable(setter):
                try:
                    setter(actual_branch, parent_branch_id=previous)
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.branch = self.checkpoints.current_branch if self.checkpoints is not None else self.branch
        except Exception:  # noqa: BLE001
            pass
        self._log("branch_switched", branch_id=actual_branch, previous_branch_id=previous,
                  reason=reason, result=result)
        return result

    # ------------------------------------------------------------------ #
    # Checkpoint / recovery callbacks
    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, reason: str, *, branch_id: str = "") -> dict[str, Any]:
        if self.emulator is None or self.checkpoints is None or self.director is None:
            return {"ok": False, "error": "session not ready"}
        state_bytes = self.emulator.save_state_bytes()
        frame = int(getattr(self.director, "frame_index", 0) or getattr(self.emulator, "frame_count", 0))
        manifest = self.checkpoints.save(
            state_bytes=state_bytes,
            frame=frame,
            buttons=int(getattr(self.director, "buttons_pressed", 0)),
            batches=int(getattr(self.director, "action_batches", 0)),
            reason=reason,
            tasks_snapshot=self.tasks.snapshot() if self.tasks else {},
            director_state=self.director.snapshot(),
            agent_contexts={
                "story": self.director.story_context.snapshot() if hasattr(self.director, "story_context") else {},
                "pokemon": self.director.pokemon_context.snapshot() if hasattr(self.director, "pokemon_context") else {},
            },
            navigation_snapshot=self.director.navigation_snapshot(),
            knowledge_version=int(self.shared.version if self.shared else 0),
            message_progress=self.bus.consumption_progress() if self.bus else {},
            budget_snapshot=self.budget.snapshot() if self.budget else {},
            memory_snapshot={
                "story": self.story_memory.snapshot() if self.story_memory else {},
                "pokemon": self.pokemon_memory.snapshot() if self.pokemon_memory else {},
            },
            evidence={"stats": self.director.snapshot().get("stats", {})},
            branch_id=str(branch_id or self.current_branch_id or ""),
        )
        self.branch = self.checkpoints.current_branch
        self.current_branch_id = str(self.branch.branch_id) if self.branch else self.current_branch_id
        return {"ok": True, **manifest}

    def _restore_checkpoint_snapshot(self, checkpoint_id: str, branch_id: str) -> dict[str, Any]:
        """Apply a checkpoint manifest: game state + task/director/memory snapshots."""
        if self.emulator is None or self.checkpoints is None or self.director is None:
            return {"ok": False, "error": "session not ready"}
        manifest = self.checkpoints.rollback_manifest(checkpoint_id)
        previous_branch = str(manifest.get("branch_id") or "")
        try:
            state_bytes = self._checkpoint_state_bytes(checkpoint_id, manifest)
            # Make sure no agent holds the lease while the state is swapped.
            try:
                self.arbiter.release_all("rollback")
            except Exception:  # noqa: BLE001 - best effort
                pass
            self.arbiter.enter_system("rollback")
            try:
                self.emulator.load_state(state_bytes)
            finally:
                self.arbiter.exit_system("rollback state loaded")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"state restore failed: {type(exc).__name__}: {exc}"}
        # task/director/memory snapshots: game state rewinds, learned experience does not.
        # The branch switch to the *new* post-rollback branch is propagated by the
        # director via ``switch_branch`` once the recovery manager has minted it.
        try:
            if self.tasks is not None and manifest.get("tasks_snapshot"):
                self.tasks.restore(manifest["tasks_snapshot"])
            if self.director is not None and manifest.get("director_state"):
                self.director.restore(manifest["director_state"])
            agent_contexts = manifest.get("agent_contexts") or {}
            if self.director is not None:
                if agent_contexts.get("story") and hasattr(self.director, "story_context"):
                    self.director.story_context.restore(agent_contexts["story"])
                if agent_contexts.get("pokemon") and hasattr(self.director, "pokemon_context"):
                    self.director.pokemon_context.restore(agent_contexts["pokemon"])
            if manifest.get("navigation_snapshot") and self.navigation is not None:
                self.navigation.restore(manifest["navigation_snapshot"])
        except Exception as exc:  # noqa: BLE001 - report partial restore honestly
            return {"ok": False, "error": f"snapshot restore failed: {type(exc).__name__}: {exc}", "manifest": manifest}
        return {"ok": True, "manifest": manifest, "relocalize_pending": True}

    def _checkpoint_state_bytes(self, checkpoint_id: str, manifest: Mapping[str, Any] | None = None) -> bytes:
        """Resolve the emulator state bytes for a branch-manifest checkpoint id."""
        loader = getattr(self.checkpoints, "load_state_bytes", None)
        if callable(loader):
            return loader(checkpoint_id)
        store_id = str((manifest or {}).get("store_checkpoint_id") or checkpoint_id)
        if self.checkpoint_store is None:
            raise EmulatorError("no checkpoint store available")
        meta = self.checkpoint_store.find(store_id)
        return self.checkpoint_store.state_bytes(meta)

    def _recovery_save_checkpoint(self, reason: str) -> dict[str, Any]:
        try:
            return self._save_checkpoint(reason)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _recovery_restore_checkpoint(self, checkpoint_id: str, branch_id: str) -> dict[str, Any]:
        return self._restore_checkpoint_snapshot(checkpoint_id, branch_id)

    def _recovery_relocalize(self) -> dict[str, Any]:
        if self.navigation is None:
            return {"ok": False, "error": "navigation unavailable"}
        result = self.navigation.relocalize()
        try:
            frame = self.emulator.observe() if self.emulator is not None else None
            if frame is not None and self.direction_observer is not None:
                self.direction_observer(frame)
        except Exception:  # noqa: BLE001
            pass
        return result

    def _recovery_director_review(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if self.director is None:
            return {}
        return self.director.director_review(dict(context), reason="recovery_l2")

    def _recovery_notify(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.bus is None:
            return
        try:
            self.bus.publish(self._system_message(event_type, payload), dedupe=False)
        except Exception:  # noqa: BLE001
            pass

    def _recovery_agent_hint(self, agent: AgentRole | str, level: str, hint: str) -> None:
        agent_enum = AgentRole.coerce(agent)
        target = self.story if agent_enum is AgentRole.STORY else self.pokemon if agent_enum is AgentRole.POKEMON else None
        if target is None:
            return
        note = getattr(target, "apply_recovery_hint", None)
        if callable(note):
            try:
                note(str(level), str(hint))
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # Wiring helpers
    # ------------------------------------------------------------------ #
    direction_observer: Callable[[Frame], None] | None = None

    def _attach_viewer_hook(self) -> None:
        if self.viewer_bridge is None or self.emulator is None:
            return
        post = getattr(self.viewer_bridge, "post_frame", None)
        if not callable(post):
            return
        try:
            self.emulator.set_frame_hook(
                lambda rgba: post(rgba), fps=float(self.cfg.emulator.presentation_fps)
            )
        except Exception as exc:  # noqa: BLE001
            self._log("viewer_hook_failed", error=f"{type(exc).__name__}: {exc}")

    def _on_agent_frame(self, frame: Frame) -> None:
        if self.viewer_bridge is None:
            return
        post = getattr(self.viewer_bridge, "post_frame", None)
        if callable(post):
            try:
                post(frame.rgba)
            except Exception:  # noqa: BLE001
                pass

    def _on_api_attempt(self, record: dict[str, Any]) -> None:
        if self.audit is not None:
            self.audit.log("api_attempts", record)

    def _on_director_status(self, status: Mapping[str, Any]) -> None:
        if self.viewer_bridge is None:
            return
        setter = getattr(self.viewer_bridge, "set_agent_status", None)
        if callable(setter):
            try:
                setter(dict(status))
            except Exception:  # noqa: BLE001
                pass

    def _publish_user_event(self, event: Mapping[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(dict(event))
        except Exception:  # noqa: BLE001
            pass

    def _system_message(self, event_type: str | EventType, payload: Mapping[str, Any]) -> Any:
        from llm_agent.multi_agent.contracts import AgentMessage, EventType as _EventType

        event = _EventType.coerce(event_type) if not isinstance(event_type, _EventType) else event_type
        return AgentMessage.create(
            event,
            source_agent=AgentRole.SYSTEM,
            target_agent=AgentRole.DIRECTOR,
            payload=dict(payload or {}),
        )

    def _warmup_api(self) -> None:
        """One real (billable) round-trip so a bad key/balance fails fast."""
        if self.router is None:
            return
        try:
            reply = self.router.probe(AgentRole.DIRECTOR, "Reply with the single word OK.")
        except ApiAuthenticationError as exc:  # type: ignore[name-defined]
            raise BlockedError(f"DeepSeek API authentication failed: {exc}") from exc
        except BudgetLimitError as exc:
            raise BudgetExceededError(str(exc)) from exc
        except AgentError as exc:
            raise BlockedError(f"DeepSeek API warmup failed: {type(exc).__name__}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise BlockedError(f"DeepSeek API warmup failed: {type(exc).__name__}: {exc}") from exc
        self._log("api_warmup_ok", model=reply.model, usage=reply.usage.to_dict())

    def _log(self, event: str, **fields: Any) -> None:
        if self.audit is not None:
            try:
                self.audit.log("events", {"event": event, **fields})
            except Exception:  # noqa: BLE001
                pass

    def _write_session_state(self, status: str) -> None:
        if self.run_dir is None:
            return
        try:
            write_json_atomic(
                self.run_dir / "multi_agent_state.json",
                {
                    "status": status,
                    "updated_at": now_ts(),
                    "session": self.status() if self.director is not None else {},
                    "result": self.last_result.to_dict() if self.last_result else {},
                },
            )
        except Exception:  # noqa: BLE001
            pass

    def _write_result(self, result: MultiAgentResult) -> None:
        if self.run_dir is None:
            return
        payload = {
            "schema": 1,
            "tool": "scripts/run_multi_agent.py",
            "started_at": self._started_at,
            "finished_at": now_ts(),
            "result": result.to_dict(),
            "status": self.status(),
        }
        try:
            write_json_atomic(self.run_dir / "multi_agent_result.json", payload)
            evidence = Path(self.cfg.multi_agent.evidence_root or "")
            if str(evidence) and evidence.parent.exists():
                write_json_atomic(evidence / f"result_{int(self._started_at)}.json", payload)
        except Exception as exc:  # noqa: BLE001
            self._log("result_write_failed", error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    # Extra public helpers used by acceptance scripts
    # ------------------------------------------------------------------ #
    def emulator_snapshot_png(self, path: str | Path) -> str:
        if self.emulator is None:
            raise EmulatorError("no emulator")
        frame = self.emulator.observe()
        save_png(frame, path)
        return str(path)

    def g1_control_contention_probe(self) -> dict[str, Any]:
        """G1 evidence: many agents race for control on the real emulator.

        No buttons are pressed: this only proves the arbitration invariant
        (exactly one owner, unauthorized agents denied) on the real object.
        """
        assert self.arbiter is not None
        results: list[dict[str, Any]] = []
        denied: list[str] = []
        self.arbiter.release_all("g1 probe start")
        for role in (AgentRole.NAVIGATION, AgentRole.STORY, AgentRole.POKEMON):
            try:
                lease = self.arbiter.request(role, reason="g1 probe")
                results.append({"role": role.value, "granted": True, "lease": lease.lease_id})
            except ControlDeniedError as exc:
                denials = {"role": role.value, "granted": False, "owner": exc.owner}
                results.append(denials)
                denied.append(role.value)
        owner_after = self.arbiter.owner.value
        # unauthorized press must be denied without touching PyBoy
        guard = self.arbiter.proxy(AgentRole.POKEMON)
        blocked_error = ""
        frame_before = int(getattr(self.emulator, "frame_count", 0) or 0)
        try:
            if not self.arbiter.holds(AgentRole.POKEMON):
                guard.press("a")
        except ControlDeniedError as exc:
            blocked_error = str(exc)
        frame_after = int(getattr(self.emulator, "frame_count", 0) or 0)
        self.arbiter.release_all("g1 probe end")
        ok = bool(denied) and blocked_error != "" and frame_before == frame_after and owner_after != "none"
        return {
            "ok": ok,
            "requests": results,
            "denied_roles": denied,
            "owner_during": owner_after,
            "unauthorized_press_error": blocked_error,
            "frame_unchanged": frame_before == frame_after,
        }


class StopRequestedError(RuntimeError):
    """Internal stop signal for the session loop."""


__all__ = [
    "NavigationAdapter",
    "MultiAgentSession",
    "MultiAgentResult",
    "StopRequestedError",
]
