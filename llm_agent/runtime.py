"""Runtime orchestrator for the DeepSeek Pokémon Red agent.

Responsibilities:

* single-instance lock, run directory, audit JSONL logs, metrics and final report;
* emulator lifecycle (start / resume / checkpoint / error recovery);
* the decision loop: capture -> prepare prompt -> budget check -> DeepSeek call ->
  execute validated tool calls -> guard -> checkpoint -> repeat;
* safe pause / save / stop handling (viewer, files, signals) and budget stops;
* periodic checkpointing and crash error snapshots;
* honest labelling of the brain mode (real DeepSeek API vs scripted dry run).

The loop is event-driven at the *decision* level, not per frame: DeepSeek decides,
Python executes deterministic button batches and only asks DeepSeek again when the
batch is finished or an important screen event interrupts it.  The decision loop
never reads hidden RAM state; that only happens in ``llm_agent/progress.py`` and
separate report tooling.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from llm_agent import __version__, contracts, observation
from llm_agent.agent import Brain, PreparedDecision
from llm_agent.api_client import DeepSeekClient
from llm_agent.budget import BudgetTracker
from llm_agent.config import ROOT, AgentConfig
from llm_agent.contracts import (
    ApiAuthenticationError,
    ApiBalanceError,
    ApiError,
    BlockedError,
    BudgetExceededError,
    CheckpointError,
    Frame,
    LockError,
    ModelReply,
    StopRequested,
    ToolResult,
)
from llm_agent.emulator import GameBoyEmulator
from llm_agent.guard import LoopGuard
from llm_agent.memory import AgentMemory
from llm_agent.tools import ToolExecutor
from llm_agent.viewer import AgentViewer, ViewerStatus

SCRIPTED_DISCLAIMER = "SCRIPTED DRY RUN — deterministic test brain, NOT a real DeepSeek API run"


# --------------------------------------------------------------------------- #
# Small filesystem helpers
# --------------------------------------------------------------------------- #

def _now() -> float:
    return time.time()


def _iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts if ts is not None else _now()))


def write_json_atomic(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_bytes_atomic(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _sha1_file(path: str | Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(pid))
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
        except Exception:
            return False


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, timeout=10).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True, timeout=10).strip())
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=str(ROOT), text=True, timeout=10).strip()
        return {"commit": commit, "dirty": dirty, "branch": branch}
    except Exception:
        return {}


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _safe_json(value: Any, limit: int = 800) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)[:limit]
    if len(text) <= limit:
        return value
    return text[: limit - 1] + "…"


def _goal_text(goal: Any) -> str:
    if isinstance(goal, dict):
        for key in ("text", "goal", "current", "value"):
            if goal.get(key):
                return str(goal[key])
        return ""
    return str(goal or "")


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #

class AuditLogger:
    """Append-only JSONL audit streams with simple size-based rotation."""

    def __init__(self, audit_dir: str | Path, rotate_bytes: int = 20_000_000) -> None:
        self.directory = Path(audit_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rotate_bytes = max(1_000_000, int(rotate_bytes))
        self._warned = False

    def log(self, stream: str, record: dict[str, Any]) -> None:
        payload = {"ts": round(_now(), 3), **record}
        line = json.dumps(payload, ensure_ascii=False, default=str)
        path = self.directory / f"{stream}.jsonl"
        try:
            if path.exists() and path.stat().st_size >= self.rotate_bytes:
                rotated = path.with_name(f"{path.stem}.{int(_now())}{path.suffix}")
                path.replace(rotated)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            if not self._warned:
                self._warned = True
                print(f"[warn] audit log write failed for {path}", file=sys.stderr, flush=True)

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Single instance lock
# --------------------------------------------------------------------------- #

class SingleInstanceLock:
    """Prevents two agents from controlling the same game state at once."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._held = False

    def acquire(self, *, run_id: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": _now(),
            "host": platform.node(),
        }
        for _ in range(2):
            try:
                handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    with os.fdopen(handle, "w", encoding="utf-8") as file_handle:
                        json.dump(payload, file_handle)
                except Exception:
                    pass
                self._held = True
                return
            except FileExistsError:
                existing = self._read()
                pid = int(existing.get("pid") or 0)
                if pid and _pid_alive(pid):
                    raise LockError(
                        f"another agent is already running (pid={pid}, run={existing.get('run_id')}). "
                        f"If it is stale, delete {self.path}"
                    )
                # stale lock: remove and retry once
                try:
                    self.path.unlink()
                except OSError:
                    pass
        raise LockError(f"could not acquire single-instance lock at {self.path}")

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def release(self) -> None:
        if not self._held:
            return
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass
        self._held = False


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #

@dataclass
class CheckpointMeta:
    id: str
    frame: int
    buttons: int
    batches: int
    reason: str
    created_at: float
    state_file: str
    meta_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "frame": self.frame,
            "buttons": self.buttons,
            "batches": self.batches,
            "reason": self.reason,
            "created_at": self.created_at,
            "state_file": self.state_file,
            "meta_file": self.meta_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMeta":
        return cls(
            id=str(data["id"]),
            frame=int(data.get("frame") or 0),
            buttons=int(data.get("buttons") or 0),
            batches=int(data.get("batches") or 0),
            reason=str(data.get("reason") or ""),
            created_at=float(data.get("created_at") or 0.0),
            state_file=str(data.get("state_file") or ""),
            meta_file=str(data.get("meta_file") or ""),
        )


class CheckpointStore:
    """Legal emulator state checkpoints (PyBoy save state + memory snapshot)."""

    def __init__(self, directory: str | Path, keep: int = 20) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "states").mkdir(parents=True, exist_ok=True)
        self.keep = max(1, int(keep))

    # -- write ----------------------------------------------------------- #
    def save(
        self,
        *,
        state_bytes: bytes,
        frame: int,
        buttons: int,
        batches: int,
        reason: str,
        memory_snapshot: dict[str, Any] | None = None,
        budget_snapshot: dict[str, Any] | None = None,
        runtime_snapshot: dict[str, Any] | None = None,
    ) -> CheckpointMeta:
        checkpoint_id = f"checkpoint_{frame:09d}"
        state_file = self.directory / "states" / f"{checkpoint_id}.state"
        meta_file = self.directory / f"{checkpoint_id}.json"
        write_bytes_atomic(state_file, state_bytes)
        meta = CheckpointMeta(
            id=checkpoint_id,
            frame=frame,
            buttons=buttons,
            batches=batches,
            reason=reason,
            created_at=_now(),
            state_file=str(state_file),
            meta_file=str(meta_file),
        )
        write_json_atomic(
            meta_file,
            {
                "meta": meta.to_dict(),
                "memory_snapshot": memory_snapshot or {},
                "budget_snapshot": budget_snapshot or {},
                "runtime_snapshot": runtime_snapshot or {},
            },
        )
        write_json_atomic(self.directory / "latest.json", meta.to_dict())
        self.prune()
        return meta

    def prune(self) -> None:
        metas = self.list()
        if len(metas) <= self.keep:
            return
        for meta in metas[: len(metas) - self.keep]:
            for path in (Path(meta.state_file), Path(meta.meta_file)):
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass

    # -- read ------------------------------------------------------------ #
    def list(self) -> list[CheckpointMeta]:
        metas: list[CheckpointMeta] = []
        for path in sorted(self.directory.glob("checkpoint_*.json")):
            if path.name.endswith(".tmp"):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                metas.append(CheckpointMeta.from_dict(raw["meta"]))
            except Exception:
                continue
        metas.sort(key=lambda meta: (meta.frame, meta.created_at))
        return metas

    def latest(self) -> CheckpointMeta | None:
        latest_file = self.directory / "latest.json"
        if latest_file.is_file():
            try:
                return CheckpointMeta.from_dict(json.loads(latest_file.read_text(encoding="utf-8")))
            except Exception:
                pass
        metas = self.list()
        return metas[-1] if metas else None

    def find(self, checkpoint_id: str) -> CheckpointMeta:
        checkpoint_id = (checkpoint_id or "").strip()
        if checkpoint_id in ("", "latest"):
            meta = self.latest()
            if meta is None:
                raise CheckpointError("no checkpoint has been saved yet")
            return meta
        if not checkpoint_id.startswith("checkpoint_"):
            raise CheckpointError(f"invalid checkpoint id: {checkpoint_id!r}")
        for meta in self.list():
            if meta.id == checkpoint_id:
                return meta
        raise CheckpointError(f"checkpoint not found: {checkpoint_id}")

    def state_bytes(self, meta: CheckpointMeta) -> bytes:
        path = Path(meta.state_file)
        if not path.is_file():
            # state_file may be relative to the run directory in old checkpoints
            alt = self.directory / meta.state_file
            if alt.is_file():
                path = alt
            else:
                raise CheckpointError(f"checkpoint state file missing: {path}")
        data = path.read_bytes()
        if not data:
            raise CheckpointError(f"checkpoint state file is empty: {path}")
        return data

    def meta_payload(self, meta: CheckpointMeta) -> dict[str, Any]:
        try:
            return json.loads(Path(meta.meta_file).read_text(encoding="utf-8"))
        except Exception:
            return {}


# --------------------------------------------------------------------------- #
# Run result
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    run_id: str
    status: str
    reason: str
    exit_code: int
    run_dir: str
    mode: str = "deepseek"
    real_api: bool = True
    model: str = ""
    turns: int = 0
    frames: int = 0
    buttons: int = 0
    action_batches: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    total_tokens: int = 0
    places: int = 0
    discoveries: int = 0
    checkpoints: int = 0
    duration_s: float = 0.0
    goal: str = ""
    last_decision: str = ""
    last_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "reason": self.reason,
            "exit_code": self.exit_code,
            "run_dir": self.run_dir,
            "mode": self.mode,
            "real_api": self.real_api,
            "model": self.model,
            "turns": self.turns,
            "frames": self.frames,
            "buttons": self.buttons,
            "action_batches": self.action_batches,
            "calls": self.calls,
            "cost_usd": round(self.cost_usd, 6),
            "total_tokens": self.total_tokens,
            "places": self.places,
            "discoveries": self.discoveries,
            "checkpoints": self.checkpoints,
            "duration_s": round(self.duration_s, 2),
            "goal": self.goal,
            "last_decision": self.last_decision,
            "last_action": self.last_action,
        }


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #

class AgentRuntime:
    """Drives one agent run: DeepSeek brain + PyBoy + tools + memory + audit."""

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        brain: Brain,
        viewer: AgentViewer | None = None,
        run_id_hint: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.brain = brain
        self.viewer = viewer
        self.run_id_hint = run_id_hint

        self.state = contracts.RuntimeState()
        self.audit: AuditLogger | None = None
        self.budget: BudgetTracker | None = None
        self.memory: AgentMemory | None = None
        self.guard: LoopGuard | None = None
        self.emulator: GameBoyEmulator | None = None
        self.executor: ToolExecutor | None = None
        self.checkpoints: CheckpointStore | None = None
        self.lock: SingleInstanceLock | None = None
        self.client: DeepSeekClient | None = None
        #: Navigation v2 subsystem (visual localization + permanent map memory).
        #: It owns low-level movement; the DeepSeek brain only picks goals.
        self.navigation: Any | None = None
        self._nav_last_status: dict[str, Any] = {}
        self._nav_frames_presented = 0
        self._nav_screens_saved = 0
        self._nav_last_report: dict[str, Any] | None = None

        self._run_dir: Path | None = None
        self._turn = 0
        self._no_tool_streak = 0
        self._consecutive_tool_errors = 0
        self._emulator_failures = 0
        self._stop_flag = False
        self._stop_reason = ""
        self._previous_frame: Frame | None = None
        self._last_frame: Frame | None = None
        self._last_outcome_text = ""
        self._screens_written = 0
        self._last_print_at = 0.0
        self._last_printed_status = ""
        self._started_at = _now()
        self._rom_sha1 = ""
        self._state_sha256 = ""
        self._git = _git_info()
        self._video_writer: Any = None
        self._video_path = ""
        self._video_error: str | None = None
        self._warmup_done = False
        self._just_resumed = False
        self._resume_applied = False
        # anti-loop bookkeeping (runtime-level self-correction)
        self._recent_batch_signatures: list[str] = []
        self._last_batch_signature = ""
        self._last_batch_a_only = False
        self._a_only_streak = 0
        self._loop_breaks = 0
        #: absolute frame = emulator process frame_count + offset (checkpoint resume)
        self._frame_offset = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def request_stop(self, reason: str = "user request") -> None:
        self._stop_flag = True
        self._stop_reason = reason

    @property
    def real_api(self) -> bool:
        return str(getattr(self.brain, "mode", "")) == "deepseek"

    def run(self) -> RunResult:
        status, reason, exit_code = "error", "", 1
        try:
            self._setup()
            status, reason = self._main_loop()
            exit_code = 0 if status in ("finished", "paused_budget", "stopped") else 3
        except StopRequested as exc:
            status, reason, exit_code = "stopped", str(exc) or "user requested stop", 0
        except ApiAuthenticationError as exc:
            status, reason, exit_code = "blocked", f"DeepSeek API authentication failed: {exc}", 4
        except BlockedError as exc:
            status, reason, exit_code = "blocked", exc.reason, 3
        except BudgetExceededError as exc:
            status, reason, exit_code = "paused_budget", str(exc), 0
        except KeyboardInterrupt:
            status, reason, exit_code = "stopped", "interrupted by user (Ctrl+C)", 0
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            status = "error"
            reason = f"{type(exc).__name__}: {exc}"
            exit_code = 1
            self._safe_record_error("fatal runtime error", exc)
        finally:
            result = self._finish(status, reason, exit_code)
        return result

    # ------------------------------------------------------------------ #
    # Setup / teardown
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        cfg = self.cfg
        problems = cfg.validate()
        if problems:
            raise BlockedError("invalid configuration: " + "; ".join(problems))

        if self.real_api and cfg.api_key() is None:
            raise BlockedError(
                f"{cfg.api.api_key_env} is not configured. Create a local .env file "
                f"(see .env.example) or set the environment variable, then rerun. "
                f"Never commit the key."
            )

        run_dir = cfg.resolve_paths(self.run_id_hint)
        self._run_dir = run_dir
        self.state = contracts.RuntimeState(run_id=run_dir.name)
        self.state.started_at = self._started_at

        self.lock = SingleInstanceLock(cfg.runtime.lock_path)
        self.lock.acquire(run_id=run_dir.name)

        self.audit = AuditLogger(run_dir / "audit", cfg.runtime.audit_rotation_bytes)
        self._rom_sha1 = _sha1_file(cfg.emulator.rom)
        self._state_sha256 = _sha256_file(cfg.emulator.state)
        self._write_run_json("starting")
        self.audit.log(
            "events",
            {
                "event": "run_start",
                "run_id": run_dir.name,
                "agent_version": __version__,
                "mode": getattr(self.brain, "mode", "unknown"),
                "real_api": self.real_api,
                "model": getattr(self.brain, "model", ""),
                "rom_sha1": self._rom_sha1,
                "state_sha256": self._state_sha256,
                "pid": os.getpid(),
            },
        )

        # memory -------------------------------------------------------- #
        self.memory = AgentMemory(cfg.memory, run_dir=run_dir)
        self.memory.load()
        self.state.goal = _goal_text(self.memory.get_goal())

        # budget -------------------------------------------------------- #
        global_budget = Path(ROOT) / "artifacts" / "llm_agent" / "budget_total.json"
        self.budget = BudgetTracker(
            cfg.budget,
            cfg.api,
            run_dir / "audit" / "budget.json",
            total_path=global_budget,
        )
        self.budget.load()

        # guard --------------------------------------------------------- #
        self.guard = LoopGuard(cfg.guard)

        # emulator ------------------------------------------------------ #
        self.emulator = GameBoyEmulator(cfg.emulator)
        self.emulator.start()
        self.checkpoints = CheckpointStore(run_dir / "checkpoints", cfg.runtime.checkpoint_keep)

        resume_store, resume_meta = self._find_resume_checkpoint()
        if resume_meta is not None:
            self._apply_checkpoint(resume_store, resume_meta, announce=True)
        else:
            self._last_frame = self._emulator_observe()
            self._previous_frame = self._last_frame
            self.audit.log("events", {"event": "fresh_start", "state": cfg.emulator.state})

        # navigation v2 -------------------------------------------------- #
        # Visual localization + permanent grid/world maps.  It is created before
        # the tool layer so raw movement tools can keep the map in sync, and it
        # survives checkpoint restores via attach()/re-localization.
        self.navigation = self._build_navigation(run_dir)

        # tools --------------------------------------------------------- #
        self.executor = ToolExecutor(
            emulator=self.emulator,
            memory=self.memory,
            state=self.state,
            screen_dir=run_dir / "screens",
            save_progress=self._tool_save_progress,
            restore_progress=self._tool_restore_progress,
            screen_writer=self._save_screen,
            cfg=cfg,
            navigator=self.navigation,
            on_action_outcome=self._on_action_outcome,
        )

        # real-time viewer presentation: stream every emulated frame at 60 fps --- #
        if self.viewer is not None:
            try:
                self.emulator.set_frame_hook(
                    self._on_emulator_frame,
                    fps=float(cfg.emulator.presentation_fps),
                )
                self._log_event("frame_streaming_enabled", fps=cfg.emulator.presentation_fps)
            except Exception as exc:  # noqa: BLE001
                self._log_event("frame_streaming_failed", error=f"{type(exc).__name__}: {exc}")

        # self-recorded mental map (dead reckoning from our own moves only) ---- #
        # The navigation subsystem loads its own permanent world model in
        # _build_navigation(); nothing else to restore here.

        # API client + warmup ------------------------------------------ #
        if self.real_api:
            existing = getattr(self.brain, "client", None)
            if isinstance(existing, DeepSeekClient):
                self.client = existing
                try:
                    self.client.on_attempt = self._on_api_attempt
                except Exception:
                    pass
            else:
                self.client = DeepSeekClient(cfg.api, on_attempt=self._on_api_attempt)
                setattr(self.brain, "client", self.client)
            self._warmup()

        self.state.status = "running"
        self.memory.save()
        self._publish(self._last_frame, force=True)
        self.audit.log("events", {"event": "setup_complete", "frame": self.state.frame_index})

    def _warmup(self) -> None:
        if self._warmup_done or not self.real_api:
            return
        assert self.budget is not None
        estimate = self.budget.estimate_call_cost(text_chars=200, image_count=0)
        allowed, why = self.budget.check_before_call(estimated_cost_usd=estimate)
        if not allowed:
            raise BudgetExceededError(why)
        self.state.status = "waiting_api"
        self._publish(self._last_frame, force=True)
        try:
            reply = self.brain.warmup()  # type: ignore[attr-defined]
        except ApiAuthenticationError:
            raise
        except ApiBalanceError as exc:
            raise BlockedError(f"DeepSeek API balance insufficient: {exc}") from exc
        except ApiError as exc:
            raise BlockedError(f"DeepSeek API warmup failed: {exc}") from exc
        except BlockedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BlockedError(f"DeepSeek API warmup failed: {type(exc).__name__}: {exc}") from exc
        cost = self.budget.record_success(reply.usage, model=reply.model, latency_s=reply.latency_s)
        self._warmup_done = True
        self.state.status = "running"
        self.audit.log(
            "events",
            {
                "event": "api_warmup_ok",
                "model": reply.model,
                "usage": reply.usage.to_dict(),
                "cost_usd": round(cost, 8),
                "content": _clip(reply.content or "", 120),
            },
        )
        print(
            f"[api] warmup OK model={reply.model} latency={reply.latency_s:.2f}s "
            f"tokens={reply.usage.total_tokens} cost=${cost:.6f}",
            flush=True,
        )

    def _finish(self, status: str, reason: str, exit_code: int) -> RunResult:
        cfg = self.cfg
        duration = _now() - self._started_at
        self.state.status = status
        self.state.blocked_reason = reason if status in ("blocked", "error") else ""

        # final checkpoint (best effort)
        if self.emulator is not None and status != "starting":
            try:
                self._save_navigation(force=True)
            except Exception as exc:  # noqa: BLE001
                self._log_event("navigation_final_save_failed", error=f"{type(exc).__name__}: {exc}")
            try:
                self._save_checkpoint(f"final:{status}")
            except Exception as exc:  # noqa: BLE001
                self._log_event("final_checkpoint_failed", error=f"{type(exc).__name__}: {exc}")

        if self.memory is not None:
            try:
                self.memory.save()
                if cfg.memory.mirror_to_run and self._run_dir is not None:
                    self.memory.save_report(self._run_dir / "memory_latest.md")
            except Exception as exc:  # noqa: BLE001
                self._log_event("memory_save_failed", error=f"{type(exc).__name__}: {exc}")

        if self.audit is not None:
            self.audit.log(
                "events",
                {"event": "run_finished", "status": status, "reason": reason, "exit_code": exit_code,
                 "duration_s": round(duration, 2)},
            )
            if status in ("blocked", "error") and reason:
                self.audit.log("errors", {"kind": "blocked", "status": status, "reason": reason})

        result = self._build_result(status, reason, exit_code, duration)
        try:
            if self._run_dir is not None:
                self._write_report(result)
        except Exception as exc:  # noqa: BLE001
            self._log_event("report_write_failed", error=f"{type(exc).__name__}: {exc}")
        try:
            if self._run_dir is not None:
                self._write_metrics(final=True)
                self._write_run_json(status, result.reason)
        except Exception:
            pass

        # video / viewer / lock / emulator
        self._close_video()
        if self.viewer is not None:
            try:
                self._publish(None, force=True)
                if cfg.viewer.auto_close_on_finish:
                    self.viewer.request_shutdown()
            except Exception:
                pass
        if self.emulator is not None:
            try:
                self.emulator.close()
            except Exception:
                pass
        if self.lock is not None:
            try:
                self.lock.release()
            except Exception:
                pass
        return result

    def _build_result(self, status: str, reason: str, exit_code: int, duration: float) -> RunResult:
        budget_state = self.budget.state() if self.budget is not None else contracts.BudgetState()
        memory_stats: dict[str, Any] = {}
        if self.memory is not None:
            try:
                memory_stats = self.memory.stats()
            except Exception:
                memory_stats = {}
        return RunResult(
            run_id=self.state.run_id,
            status=status,
            reason=reason,
            exit_code=exit_code,
            run_dir=str(self._run_dir or ""),
            mode=str(getattr(self.brain, "mode", "unknown")),
            real_api=self.real_api,
            model=str(getattr(self.brain, "model", "")),
            turns=self.state.llm_turns,
            frames=self.state.frame_index,
            buttons=self.state.buttons_pressed,
            action_batches=self.state.action_batches,
            calls=self.state.llm_calls,
            cost_usd=budget_state.cost_usd,
            total_tokens=budget_state.total_tokens,
            places=int(memory_stats.get("places") or self.state.places_discovered),
            discoveries=int(memory_stats.get("discoveries") or self.state.discoveries),
            checkpoints=self.state.checkpoints,
            duration_s=duration,
            goal=self.state.goal,
            last_decision=self.state.last_decision,
            last_action=self.state.last_action,
        )

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def _main_loop(self) -> tuple[str, str]:
        cfg = self.cfg
        while True:
            self._check_controls()
            self._turn += 1
            self.state.llm_turns = self._turn

            if cfg.runtime.max_turns and self._turn > cfg.runtime.max_turns:
                return "finished", f"max_turns={cfg.runtime.max_turns} reached"
            assert self.budget is not None
            limit_reason = self.budget.limits_exceeded()
            if limit_reason:
                return "paused_budget", limit_reason
            if cfg.emulator.max_actions and self.state.buttons_pressed >= cfg.emulator.max_actions:
                return "finished", f"emulator.max_actions={cfg.emulator.max_actions} reached"

            self.state.status = "running"
            try:
                frame = self._emulator_observe()
            except Exception as exc:  # noqa: BLE001
                self._handle_emulator_failure(exc)
                continue
            assert self.guard is not None and self.memory is not None
            analysis = observation.analyze(frame, self._previous_frame)
            self.state.frame_index = self._absolute_frame(frame.frame_index)
            self.guard.on_observation(
                frame,
                analysis.event,
                changed=(self._previous_frame is None or frame.sha256 != self._previous_frame.sha256),
            )
            self.state.loop_warnings = [
                f"{warning.severity}:{warning.message}" for warning in self.guard.check()
            ]
            trigger = self._pick_trigger(analysis)
            self.state.last_trigger = trigger
            self._publish(frame)

            try:
                prepared = self.brain.prepare(
                    frame=frame,
                    analysis=analysis,
                    state=self.state,
                    memory=self.memory,
                    guard=self.guard,
                    last_outcome_text=self._last_outcome_text,
                    trigger=trigger,
                    previous_frame=self._previous_frame,
                    mental_map_text="",
                    nav_status_text=self._nav_prompt_text(),
                )
            except Exception as exc:  # noqa: BLE001
                raise BlockedError(f"failed to build DeepSeek request: {type(exc).__name__}: {exc}") from exc

            estimate = self._estimate_cost(prepared)
            allowed, why = self.budget.check_before_call(estimated_cost_usd=estimate)
            if not allowed:
                return "paused_budget", why

            self.state.status = "waiting_api"
            self._publish(frame, force=True)
            reply = self._call_brain(prepared)
            if reply is None:
                continue

            cost = self.budget.record_success(reply.usage, model=reply.model, latency_s=reply.latency_s)
            self.state.llm_calls += 1
            self.state.consecutive_failures = 0
            rationale = reply.rationale()
            if not rationale:
                rationale = "工具调用: " + ", ".join(call.name for call in reply.tool_calls[:5])
            self.state.last_decision = rationale
            self.state.last_decision_at = _now()
            self._audit_decision(prepared, reply, cost, estimate)

            if not reply.tool_calls:
                self._no_tool_streak += 1
                self.state.add_history("event", f"model returned no tool calls (streak={self._no_tool_streak})")
                self._log_event("no_tool_calls", turn=self._turn, streak=self._no_tool_streak,
                                content=_clip(reply.content or "", 300))
                self._publish(frame, force=True)
                if self._no_tool_streak >= 3:
                    raise BlockedError("model repeatedly returned no tool calls")
                continue

            self._no_tool_streak = 0
            self._execute_batch(reply)
            self._after_batch(frame)

    def _pick_trigger(self, analysis: Any) -> str:
        if self._turn == 1:
            if self._just_resumed:
                self._just_resumed = False
                return "resume"
            return "start"
        event = getattr(analysis, "event", contracts.ScreenEvent.NONE)
        if event in contracts.SIGNIFICANT_EVENTS:
            return f"event:{getattr(event, 'value', event)}"
        if self.state.loop_warnings:
            return "anomaly"
        if self._no_tool_streak or self._consecutive_tool_errors:
            return "recovery"
        return "after_actions"

    # ------------------------------------------------------------------ #
    # API call handling (with live 60 fps presentation while DeepSeek thinks)
    # ------------------------------------------------------------------ #
    def _call_brain(self, prepared: PreparedDecision) -> ModelReply | None:
        if (
            self.viewer is not None
            and self.cfg.emulator.live_presentation_while_waiting
            and self.emulator is not None
        ):
            return self._call_brain_live(prepared)
        return self._call_brain_sync(prepared)

    def _call_brain_sync(self, prepared: PreparedDecision) -> ModelReply | None:
        try:
            reply = self.brain.complete(prepared)
            assert self.guard is not None
            self.guard.on_llm_call(ok=True)
            return reply
        except BaseException as exc:  # noqa: BLE001 - routed through the shared handler
            return self._handle_brain_exception(exc)

    def _call_brain_live(self, prepared: PreparedDecision) -> ModelReply | None:
        """Request the DeepSeek decision in a worker while the game keeps animating.

        PyBoy stays on the agent thread: we tick one frame at a time at the
        presentation fps and stream it to the viewer, so the viewer shows a live
        60 fps game while the ~5-6 s API round-trip happens instead of freezing.
        """
        holder: dict[str, Any] = {}

        def worker() -> None:
            try:
                holder["reply"] = self.brain.complete(prepared)
            except BaseException as exc:  # noqa: BLE001 - reported back on the agent thread
                holder["error"] = exc

        thread = threading.Thread(target=worker, name="deepseek-request", daemon=True)
        thread.start()
        interval = 1.0 / max(5, min(120, int(self.cfg.emulator.presentation_fps)))
        next_at = time.perf_counter()
        published_at = 0.0
        while thread.is_alive():
            try:
                frame = self.emulator.wait(1)  # type: ignore[union-attr]
                self._last_frame = frame
                self._sync_absolute_frame(frame)
            except Exception:  # noqa: BLE001 - a presentation hiccup must not kill the request
                pass
            now = time.perf_counter()
            if now - published_at >= 0.5:
                published_at = now
                self._publish(self._last_frame)
            next_at += interval
            delay = next_at - now
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.perf_counter()
            if self._stop_requested():
                break
        thread.join()
        self._publish(self._last_frame, force=True)
        if "error" in holder:
            return self._handle_brain_exception(holder["error"])
        reply = holder["reply"]
        assert self.guard is not None
        self.guard.on_llm_call(ok=True)
        return reply

    def _handle_brain_exception(self, exc: BaseException) -> ModelReply | None:
        assert self.guard is not None and self.audit is not None and self.budget is not None
        if isinstance(exc, (ApiAuthenticationError, ApiBalanceError)):
            raise exc
        if isinstance(exc, ApiError):
            self.state.api_failures += 1
            self.state.consecutive_failures += 1
            self.state.last_error = _clip(f"{type(exc).__name__}: {exc}", 500)
            self.guard.on_llm_call(ok=False)
            self.budget.record_failure(status=exc.status, retryable=exc.retryable)
            self.audit.log("errors", {"kind": "api", "turn": self._turn, **exc.to_dict()})
            self.state.add_history("error", f"API {type(exc).__name__}: {_clip(str(exc), 200)}")
            self._publish(force=True)
            if self.state.consecutive_failures >= self.cfg.guard.max_consecutive_api_failures:
                raise BlockedError(
                    f"too many consecutive API failures ({self.state.consecutive_failures}): {exc}"
                ) from exc
            backoff = min(
                self.cfg.api.retry_max_s,
                self.cfg.api.retry_base_s * (2 ** min(self.state.consecutive_failures, 6)),
            )
            print(f"[api] {type(exc).__name__} — retrying in {backoff:.1f}s", file=sys.stderr, flush=True)
            self._sleep(backoff)
            return None
        if isinstance(exc, (BlockedError, StopRequested, BudgetExceededError)):
            raise exc
        self.state.last_error = _clip(f"{type(exc).__name__}: {exc}", 500)
        self.audit.log("errors", {"kind": "brain", "turn": self._turn,
                                  "error": f"{type(exc).__name__}: {exc}"})
        raise BlockedError(f"unexpected brain/client failure: {type(exc).__name__}: {exc}") from exc

    def _on_emulator_frame(self, rgba: Any) -> None:
        """Per-frame presentation hook installed only in live-viewer mode."""
        if self.viewer is not None:
            try:
                self.viewer.post_frame(rgba)
            except Exception:
                pass

    def _estimate_cost(self, prepared: PreparedDecision) -> float:
        assert self.budget is not None
        audit = prepared.audit or {}
        return self.budget.estimate_call_cost(
            text_chars=int(audit.get("prompt_chars") or 0),
            image_count=int(audit.get("image_count") or 0),
            image_width=int(audit.get("image_width") or 160 * self.cfg.api.image_scale),
            image_height=int(audit.get("image_height") or 144 * self.cfg.api.image_scale),
        )

    def _audit_decision(self, prepared: PreparedDecision, reply: ModelReply, cost: float, estimate: float) -> None:
        assert self.audit is not None
        self.audit.log(
            "decisions",
            {
                "turn": self._turn,
                "trigger": prepared.trigger,
                "mode": str(getattr(self.brain, "mode", "unknown")),
                "model": reply.model,
                "latency_s": round(reply.latency_s, 3),
                "estimated_usd": round(estimate, 6),
                "cost_usd": round(cost, 6),
                "usage": reply.usage.to_dict(),
                "finish_reason": reply.finish_reason,
                "rationale": reply.rationale(),
                "tool_calls": [
                    {"name": call.name, "arguments": _safe_json(call.arguments, 600)}
                    for call in reply.tool_calls
                ],
                "prompt": prepared.audit,
            },
        )

    def _on_api_attempt(self, record: dict[str, Any]) -> None:
        if self.audit is not None:
            self.audit.log("api_attempts", record)

    # ------------------------------------------------------------------ #
    # Tool batch execution
    # ------------------------------------------------------------------ #
    def _execute_batch(self, reply: ModelReply) -> None:
        assert self.executor is not None and self.audit is not None and self.guard is not None
        self.state.status = "acting"
        self._publish(force=True)
        calls = reply.tool_calls
        self._last_batch_signature = "|".join(
            f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False, default=str)}"
            for call in calls
        )
        self._last_batch_a_only = bool(calls) and all(
            self._is_a_only_call(call) for call in calls
        )
        interrupted = False
        executed = 0
        for index, call in enumerate(calls):
            try:
                result = self.executor.execute(call)
            except Exception as exc:  # noqa: BLE001 - tool layer should not raise, but stay alive
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    ok=False,
                    summary="",
                    error=f"{type(exc).__name__}: {exc}",
                    needs_llm=True,
                    data={"_runtime_generated": True},
                )
            executed += 1
            self.audit.log(
                "actions",
                {
                    "turn": self._turn,
                    "index": index,
                    "tool": call.name,
                    "arguments": _safe_json(call.arguments, 800),
                    "ok": result.ok,
                    "error": result.error,
                    "summary": _clip(result.summary, 500),
                    "needs_llm": result.needs_llm,
                    "data": _safe_json(result.data, 600),
                    "screens": result.screens[:4],
                },
            )
            # The tool layer appends its own compact history entries (success and
            # validation errors); only runtime-generated failure results need one here.
            if result.data.get("_runtime_generated"):
                self.state.add_history("tool", f"{call.name} ERROR: {_clip(result.error or result.summary, 260)}")
            if result.ok:
                self._consecutive_tool_errors = 0
            else:
                self._consecutive_tool_errors += 1
            self.guard.on_tool_result(result)
            self._refresh_memory_stats()
            if self._consecutive_tool_errors >= self.cfg.guard.max_consecutive_tool_errors:
                raise BlockedError(
                    f"{self._consecutive_tool_errors} consecutive tool errors, last: "
                    f"{result.error or result.summary}"
                )
            stop_batch = call.name == "restore_checkpoint" or (
                result.needs_llm and index < len(calls) - 1
            )
            if stop_batch:
                interrupted = True
                skipped = len(calls) - index - 1
                self.state.add_history(
                    "event",
                    f"batch interrupted after {call.name} ({skipped} planned call(s) skipped)",
                )
                self.audit.log(
                    "events",
                    {"event": "batch_interrupted", "turn": self._turn, "after": call.name,
                     "skipped": skipped, "reason": _clip(result.summary, 200)},
                )
                break

        outcome = getattr(self.executor, "last_outcome", None)
        if outcome is not None:
            self.guard.on_action_outcome(outcome)
            self._last_outcome_text = outcome.summary
            steps = getattr(outcome, "steps", []) or []
            self._record_video(steps)
            # The navigation subsystem already received this exact outcome once,
            # through ToolExecutor.on_action_outcome (no double counting here).
        self._refresh_nav_status()
        self._save_navigation()
        last_frame = getattr(self.executor, "last_frame", None)
        last_analysis = getattr(self.executor, "last_analysis", None)
        if last_frame is not None and last_analysis is not None:
            self.guard.on_observation(
                last_frame,
                last_analysis.event,
                changed=bool(getattr(last_analysis, "changed", True)),
            )
            self._last_frame = last_frame
        self._sync_absolute_frame(self._last_frame)
        if self.state.last_decision:
            self.state.last_action = f"{executed}/{len(calls)} tools — {_clip(self.state.last_decision, 200)}"
        self._publish(self._last_frame, force=True)

    def _after_batch(self, request_frame: Frame) -> None:
        assert self.memory is not None and self.guard is not None and self.audit is not None
        cfg = self.cfg
        self._refresh_memory_stats()

        if (
            self.state.action_batches > 0
            and self.state.action_batches % cfg.runtime.checkpoint_every_actions == 0
        ):
            self._save_checkpoint("auto")
        else:
            self._write_metrics()

        try:
            self.memory.maybe_compress(actions_since=self.state.action_batches)
            self.memory.save()
        except Exception as exc:  # noqa: BLE001
            self._log_event("memory_housekeeping_failed", error=f"{type(exc).__name__}: {exc}")

        self._prune_screens()
        self._previous_frame = self._last_frame or request_frame
        self._maybe_break_repetition_loop()

        if self.guard.critical():
            self._log_event("guard_critical", warnings=self.guard.render_warnings(400))
            if cfg.guard.auto_restore_on_lock:
                try:
                    self._restore_latest("guard_critical")
                except Exception as exc:  # noqa: BLE001
                    self._log_event("guard_restore_failed", error=f"{type(exc).__name__}: {exc}")
        self._publish(self._last_frame or request_frame, force=True)

    # ------------------------------------------------------------------ #
    # Runtime-level self-correction: repeated-A dialogue loops
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_a_only_call(call: contracts.ToolCall) -> bool:
        """True when this tool call only presses the A button."""
        if call.name == "press_button":
            return call.arguments.get("button") == "a"
        if call.name == "execute_actions":
            actions = call.arguments.get("actions") or []
            return bool(actions) and all(action == "a" for action in actions)
        return False

    def _maybe_break_repetition_loop(self) -> None:
        """Detect 'press A into the same NPC forever' loops and step away once.

        The model normally recovers by itself, but a dialogue that keeps
        re-opening on every A press can trap an agent in an expensive loop.  The
        runtime breaks it deterministically (one step in a direction that
        changes the screen), records the intervention in the history/audit and
        lets DeepSeek continue from the new screen.  No hidden state is used.
        """
        if not self._last_batch_signature:
            return
        self._recent_batch_signatures.append(self._last_batch_signature)
        self._recent_batch_signatures = self._recent_batch_signatures[-8:]
        if self._last_batch_a_only:
            self._a_only_streak += 1
        else:
            self._a_only_streak = 0
        last_analysis = getattr(self.executor, "last_analysis", None) if self.executor else None
        dialog_open = (
            last_analysis is not None
            and getattr(last_analysis, "event", contracts.ScreenEvent.NONE) == contracts.ScreenEvent.DIALOG
        )
        repeated = (
            self._a_only_streak >= 3
            and dialog_open
        )
        if not repeated:
            return
        self._a_only_streak = 0
        self._loop_breaks += 1
        self.state.loop_warnings = [
            "critical: repeated A-only dialogue loop detected — runtime stepped away to break it"
        ]
        self._log_event(
            "repetition_loop_break",
            signature=self._last_batch_signature[:300],
            breaks=self._loop_breaks,
            frame=self.state.frame_index,
        )
        if self._loop_breaks > 6:
            raise BlockedError("repeated A-only dialogue loop persisted after 6 automatic breaks")
        if not self._step_away_from_loop():
            try:
                self.executor.execute(
                    contracts.ToolCall(id=f"runtime_break_{self._loop_breaks}", name="wait_frames",
                                       arguments={"frames": 60})
                )
            except Exception:  # noqa: BLE001
                pass
        if self.memory is not None:
            try:
                self.memory.record_lesson(
                    "Pressing A repeatedly at the same NPC re-opens the same dialogue; after a conversation "
                    "ends the agent must step away or choose another action.",
                    success=False,
                    context="runtime loop breaker",
                )
            except Exception:
                pass
        self.state.add_history(
            "event",
            "runtime: an A-only dialogue loop was detected; the runtime moved one step away so the next "
            "decision starts from a different screen. Do not press A repeatedly into the same NPC.",
        )
        self._recent_batch_signatures.clear()

    def _step_away_from_loop(self) -> bool:
        if self.emulator is None:
            return False
        before = self._last_frame
        directions = ("down", "left", "right", "up")
        offset = self._loop_breaks % len(directions)
        for direction in directions[offset:] + directions[:offset]:
            try:
                outcome = self.emulator.run_batch([direction], stop_on_event=False)
            except Exception:  # noqa: BLE001
                continue
            frame = outcome.final_frame
            if before is None or frame.sha256 != before.sha256:
                self._last_frame = frame
                self._previous_frame = frame
                self.state.frame_index = frame.frame_index
                self.state.buttons_pressed += 1
                self._log_event("repetition_loop_step", direction=direction, frame=frame.frame_index)
                return True
        return False

    def _refresh_memory_stats(self) -> None:
        if self.memory is None:
            return
        try:
            stats = self.memory.stats()
        except Exception:
            return
        self.state.places_discovered = int(stats.get("places") or self.state.places_discovered)
        self.state.discoveries = int(stats.get("discoveries") or self.state.discoveries)

    # ------------------------------------------------------------------ #
    # Emulator helpers
    # ------------------------------------------------------------------ #
    def _emulator_observe(self) -> Frame:
        assert self.emulator is not None
        return self.emulator.observe()

    # ------------------------------------------------------------------ #
    # Frame index bookkeeping
    # ------------------------------------------------------------------ #
    def _absolute_frame(self, process_frame: int) -> int:
        """Map PyBoy's per-process frame counter to the persistent run frame."""
        return int(process_frame) + int(self._frame_offset)

    def _sync_absolute_frame(self, frame: Frame | None = None) -> None:
        frame = frame or self._last_frame
        if frame is None or self.emulator is None:
            return
        self.state.frame_index = self._absolute_frame(frame.frame_index)

    def _handle_emulator_failure(self, exc: Exception) -> None:
        self._emulator_failures += 1
        self._safe_record_error("emulator failure", exc)
        self._log_event("emulator_failure", attempt=self._emulator_failures,
                        error=f"{type(exc).__name__}: {exc}")
        if self._emulator_failures > self.cfg.runtime.auto_resume_attempts:
            raise BlockedError(
                f"emulator failed {self._emulator_failures} times, last: {type(exc).__name__}: {exc}"
            ) from exc
        self._restore_latest(f"emulator_recovery_{self._emulator_failures}")

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #
    @property
    def _resumed(self) -> bool:
        return bool(getattr(self, "_resume_applied", False))

    def _find_resume_checkpoint(self) -> tuple[CheckpointStore, CheckpointMeta | None]:
        cfg = self.cfg
        if cfg.runtime.resume_from:
            source = Path(cfg.runtime.resume_from)
            if not (source / "checkpoints").is_dir():
                source = source / "checkpoints"
            store = CheckpointStore(source, cfg.runtime.checkpoint_keep)
            meta = store.latest()
            if meta is None:
                raise CheckpointError(f"no checkpoint found in resume_from directory: {cfg.runtime.resume_from}")
            self._log_event("resume_from_previous_run", source=str(source), checkpoint=meta.id)
            return store, meta
        if cfg.runtime.resume and self.checkpoints is not None:
            meta = self.checkpoints.latest()
            if meta is not None:
                return self.checkpoints, meta
        return self.checkpoints or CheckpointStore(Path(cfg.runtime.run_dir or ".") / "checkpoints"), None

    def _apply_checkpoint(
        self,
        store: CheckpointStore,
        meta: CheckpointMeta,
        *,
        announce: bool = False,
        restore_memory: bool = False,
    ) -> None:
        assert self.emulator is not None
        state_bytes = store.state_bytes(meta)
        self.emulator.load_state(state_bytes)
        # PyBoy's frame_count restarts with this process; keep a persistent offset so
        # checkpoint ids / metrics keep using the absolute run frame.
        try:
            self._frame_offset = max(0, int(meta.frame) - int(self.emulator.frame_count))
        except Exception:
            self._frame_offset = 0
        self.state.frame_index = meta.frame
        self.state.buttons_pressed = max(self.state.buttons_pressed, meta.buttons)
        self.state.action_batches = max(self.state.action_batches, meta.batches)
        self._resume_applied = True
        self._just_resumed = True
        if restore_memory and self.memory is not None:
            try:
                payload = store.meta_payload(meta)
                snapshot = payload.get("memory_snapshot") or {}
                if snapshot:
                    self.memory.restore(snapshot)
                    self.memory.save()
                    self._log_event("memory_restored_to_checkpoint", checkpoint=meta.id)
                nav_snapshot = (payload.get("runtime_snapshot") or {}).get("navigation")
                if nav_snapshot:
                    self._restore_navigation_snapshot(nav_snapshot)
            except Exception as exc:  # noqa: BLE001
                self._log_event("memory_restore_failed", error=f"{type(exc).__name__}: {exc}")
        self._last_frame = self._emulator_observe()
        self._previous_frame = self._last_frame
        if self.guard is not None:
            self.guard.reset_after_recovery()
        if self.memory is not None:
            try:
                self.memory.record_event("checkpoint", f"game state restored to {meta.id} (frame {meta.frame})")
            except Exception:
                pass
        self.state.add_history("event", f"restored checkpoint {meta.id} at frame {meta.frame} ({meta.reason})")
        self._log_event("checkpoint_restored", checkpoint=meta.id, frame=meta.frame,
                        reason=meta.reason, announce=announce)

    def _save_checkpoint(self, reason: str) -> CheckpointMeta:
        assert self.emulator is not None and self.checkpoints is not None
        self._sync_absolute_frame()
        state_bytes = self.emulator.save_state_bytes()
        memory_snapshot = {}
        budget_snapshot = {}
        if self.memory is not None:
            try:
                memory_snapshot = self.memory.snapshot()
            except Exception:
                memory_snapshot = {}
        if self.budget is not None:
            try:
                budget_snapshot = self.budget.snapshot()
            except Exception:
                budget_snapshot = {}
        runtime_snapshot = self.state.to_dict()
        try:
            if self.cfg.navigation.persist_on_checkpoint:
                self._save_navigation(force=True)
            nav_snapshot = self._navigation_snapshot()
            if nav_snapshot:
                runtime_snapshot["navigation"] = nav_snapshot
        except Exception:  # noqa: BLE001
            pass
        meta = self.checkpoints.save(
            state_bytes=state_bytes,
            frame=self.state.frame_index,
            buttons=self.state.buttons_pressed,
            batches=self.state.action_batches,
            reason=reason,
            memory_snapshot=memory_snapshot,
            budget_snapshot=budget_snapshot,
            runtime_snapshot=runtime_snapshot,
        )
        self.state.checkpoints += 1
        self._log_event("checkpoint_saved", checkpoint=meta.id, frame=meta.frame, reason=reason)
        self._write_metrics()
        self._publish(force=True)
        return meta

    def _restore_latest(self, reason: str) -> str:
        assert self.checkpoints is not None
        meta = self.checkpoints.latest()
        if meta is None:
            raise CheckpointError("no checkpoint available to restore")
        self._apply_checkpoint(self.checkpoints, meta, announce=True, restore_memory=True)
        self._log_event("auto_restore", reason=reason, checkpoint=meta.id)
        return meta.id

    # -- callbacks given to the tool layer ------------------------------ #
    def _tool_save_progress(self, reason: str = "") -> str:
        meta = self._save_checkpoint(f"tool:{reason or 'agent_request'}")
        return f"{meta.id} (frame={meta.frame}, buttons={meta.buttons})"

    def _tool_restore_progress(self, checkpoint_id: str = "latest") -> str:
        assert self.checkpoints is not None
        meta = self.checkpoints.find(checkpoint_id)
        self._apply_checkpoint(self.checkpoints, meta, announce=True, restore_memory=True)
        return (
            f"restored {meta.id} at frame {meta.frame}. Agent memory was rewound to that checkpoint "
            f"and the journal keeps the audit trail; re-verify the surroundings before acting."
        )

    # ------------------------------------------------------------------ #
    # Navigation v2 integration
    # ------------------------------------------------------------------ #
    def _build_navigation(self, run_dir: Path) -> Any | None:
        """Create the visual navigation subsystem (None when disabled/unavailable)."""
        if not self.cfg.navigation.enabled:
            self._log_event("navigation_disabled")
            return None
        if self.emulator is None:
            return None
        try:
            from llm_agent.navigation.navigation_agent import NavigationAgent
        except Exception as exc:  # noqa: BLE001 - navigate feature must not break legacy runs
            self._log_event("navigation_import_failed", error=f"{type(exc).__name__}: {exc}")
            print(
                f"[warn] navigation subsystem unavailable ({type(exc).__name__}: {exc}); "
                "falling back to raw movement tools",
                file=sys.stderr,
                flush=True,
            )
            return None
        world_root = self.cfg.navigation.world_root or str(run_dir / "navigation")
        try:
            agent = NavigationAgent(
                emulator=self.emulator,
                world_root=world_root,
                config=self.cfg,
                on_frame=self._on_nav_frame,
                on_step=self._on_nav_step,
                on_status=self._on_nav_status,
            )
            attach = getattr(agent, "attach", None)
            if callable(attach):
                attach()
            self._nav_last_status = dict(getattr(agent, "status", lambda: {})() or {})
            self._log_event(
                "navigation_ready",
                world_root=world_root,
                **{key: self._nav_last_status.get(key) for key in ("area_id", "areas_total")},
            )
            return agent
        except Exception as exc:  # noqa: BLE001 - never block the run on navigation setup
            self._log_event("navigation_setup_failed", error=f"{type(exc).__name__}: {exc}")
            self._safe_record_error("navigation setup", exc)
            return None

    def _on_nav_frame(self, frame: Frame) -> None:
        """Called by the navigation controller for every executed navigation step."""
        if not isinstance(frame, Frame):
            return
        self._last_frame = frame
        self._nav_frames_presented += 1
        try:
            self.state.frame_index = self._absolute_frame(int(frame.frame_index))
            self.state.touch()
        except Exception:  # noqa: BLE001
            pass
        if self.viewer is not None:
            try:
                self.viewer.post_frame(frame.rgba)
            except Exception:  # noqa: BLE001 - presentation only
                pass
        # Keep periodic visual evidence of autonomous navigation (not every step).
        if (
            self.cfg.runtime.save_screens
            and self._nav_frames_presented % max(1, int(self.cfg.runtime.screenshot_every_batches)) == 0
            and self._nav_frames_presented % 5 == 0
        ):
            self._save_screen(frame)
            self._nav_screens_saved += 1

    def _on_nav_step(self, step: Any) -> None:
        """Audit hook: one verified navigation step (never raises into navigation)."""
        if self.audit is None:
            return
        try:
            payload = step.to_dict() if hasattr(step, "to_dict") else {"step": str(step)}
            self.audit.log("navigation", payload)
        except Exception:  # noqa: BLE001
            pass

    def _on_nav_status(self, status: Any) -> None:
        """Status hook: navigation publishes live progress (viewer / prompt cache)."""
        if isinstance(status, Mapping):
            self._nav_last_status = dict(status)
        if self.viewer is not None:
            try:
                self._publish(None, force=True)
            except Exception:  # noqa: BLE001
                pass

    def _on_action_outcome(self, outcome: Any) -> None:
        """Exactly-once sync of raw movement tool results into the permanent map.

        Called by :class:`~llm_agent.tools.ToolExecutor` whenever the model used a
        legacy raw movement tool (press/execute_actions/walk).  Navigation tools
        update the map themselves and do NOT pass through here.
        """
        if self.navigation is None:
            return
        function = getattr(self.navigation, "observe_external_outcome", None)
        if not callable(function):
            return
        try:
            function(outcome)
            self._refresh_nav_status()
        except Exception as exc:  # noqa: BLE001
            self._log_event("navigation_sync_failed", error=f"{type(exc).__name__}: {exc}")

    def _refresh_nav_status(self) -> None:
        if self.navigation is None:
            return
        function = getattr(self.navigation, "status", None)
        if not callable(function):
            return
        try:
            status = function()
            if isinstance(status, Mapping):
                self._nav_last_status = dict(status)
        except Exception:  # noqa: BLE001
            pass

    def _navigation_summary(self) -> dict[str, Any]:
        """Compact navigation status for prompt/viewer/metrics/report."""
        summary = dict(self._nav_last_status or {})
        if not summary and self.navigation is not None:
            self._refresh_nav_status()
            summary = dict(self._nav_last_status or {})
        if not summary:
            return {}
        summary.setdefault("area_id", "")
        summary.setdefault("areas_total", 0)
        if self.executor is not None:
            report = getattr(self.executor, "last_nav_report", None)
            if report:
                summary["last_report_status"] = report.get("status")
                summary["last_report_reason"] = _clip(str(report.get("reason") or ""), 300)
        return summary

    def _nav_prompt_text(self) -> str:
        """Navigation status + ASCII map block injected into the decision prompt."""
        cfg = self.cfg
        if self.navigation is None or not cfg.navigation.status_in_prompt:
            return ""
        summary = self._navigation_summary()
        if not summary:
            return ""
        try:
            from llm_agent.tools import _nav_status_summary

            text = _nav_status_summary(summary)
        except Exception:  # noqa: BLE001
            text = ""
        try:
            payload = self.navigation.local_map(radius=int(cfg.navigation.prompt_map_radius))
            ascii_map = str((payload or {}).get("ascii") or "")
            if ascii_map:
                text = (text + "\n\n" + ascii_map).strip()
        except Exception:  # noqa: BLE001 - a map rendering bug must not break prompts
            pass
        limit = int(cfg.navigation.prompt_map_max_chars)
        if len(text) > limit:
            text = text[: max(0, limit - 1)] + "…"
        return text

    def _post_nav_map(self) -> None:
        """Render the live local map panel for the Tk viewer (throttled, best effort)."""
        if self.viewer is None or self.navigation is None:
            return
        post = getattr(self.viewer, "post_map", None)
        if not callable(post):
            return
        now = _now()
        if now - getattr(self, "_nav_map_posted_at", 0.0) < 0.25:
            return
        self._nav_map_posted_at = now
        image = self._nav_map_image()
        if image is None:
            return
        try:
            post(image)
        except Exception:  # noqa: BLE001
            pass

    def _nav_map_image(self) -> Any | None:
        """Render the live local map with navigation.viz (best effort, cheap)."""
        try:
            from llm_agent.navigation import viz

            agent = self.navigation
            if agent is None:
                return None
            grid = None
            world = getattr(agent, "world", None)
            if world is not None:
                candidate = getattr(world, "current_grid", None)
                grid = candidate() if callable(candidate) else candidate
                if grid is None:
                    record_getter = getattr(world, "current_record", None)
                    if callable(record_getter):
                        record = record_getter()
                        grid = getattr(record, "grid", None)
            if grid is None:
                return None
            localizer = getattr(agent, "localizer", None)
            pose = getattr(localizer, "pose", None)
            controller = getattr(agent, "controller", None)
            path = None
            target = None
            if controller is not None:
                path = getattr(controller, "last_path", None)
                target = getattr(controller, "last_target", None)
            return viz.render_map_image(
                grid,
                pose=pose,
                path=path,
                target=target,
                cell_px=int(self.cfg.navigation.viewer_map_cell_px),
                max_cells=26,
                legend=False,
            )
        except Exception:  # noqa: BLE001 - map panel must never break a run
            return None

    # ------------------------------------------------------------------ #
    # Screenshots / video
    # ------------------------------------------------------------------ #
    def _save_screen(self, frame: Frame) -> str:
        cfg = self.cfg
        if not cfg.runtime.save_screens or self._run_dir is None:
            return ""
        name = f"frame_{frame.frame_index:09d}_{frame.sha256[:8]}.png"
        try:
            observation.save_png(frame, self._run_dir / "screens" / name)
            self._screens_written += 1
            return name
        except Exception:
            return ""

    def _save_navigation(self, *, force: bool = False) -> None:
        """Persist the permanent navigation world model (throttled, best effort)."""
        if self.navigation is None:
            return
        now = _now()
        if not force and now - float(getattr(self, "_nav_last_save_at", 0.0) or 0.0) < 2.0:
            return
        self._nav_last_save_at = now
        try:
            self.navigation.save()
        except Exception as exc:  # noqa: BLE001
            self._log_event("navigation_save_failed", error=f"{type(exc).__name__}: {exc}")

    def _navigation_snapshot(self) -> dict[str, Any]:
        """Serializable navigation state hint for checkpoints.

        The permanent world model is persisted separately under
        ``<run_dir>/navigation``; a checkpoint only stores the *pose hint*, and a
        restore always re-localizes visually (``attach``) before trusting it.
        """
        if self.navigation is None:
            return {}
        summary = self._navigation_summary()
        if not summary:
            return {}
        return {
            "area_id": summary.get("area_id", ""),
            "pose": summary.get("pose"),
            "areas_total": summary.get("areas_total", 0),
            "form": "pose-hint",
        }

    def _restore_navigation_snapshot(self, snapshot: dict[str, Any]) -> None:
        if self.navigation is None or not isinstance(snapshot, dict):
            return
        # Relocalize against the persisted world model after the emulator state
        # was restored; never blindly trust a stored coordinate (the map may be
        # newer than the checkpoint).
        function = getattr(self.navigation, "restore_snapshot", None)
        try:
            if callable(function):
                function(snapshot)
            else:
                attach = getattr(self.navigation, "attach", None)
                if callable(attach):
                    attach()
            self._refresh_nav_status()
            self._log_event("navigation_restored_to_checkpoint", **self._navigation_summary())
        except Exception as exc:  # noqa: BLE001
            self._log_event("navigation_restore_failed", error=f"{type(exc).__name__}: {exc}")

    def _prune_screens(self) -> None:
        if self._run_dir is None:
            return
        cfg = self.cfg
        limit = max(50, int(cfg.runtime.screen_limit))
        screens = sorted((self._run_dir / "screens").glob("frame_*.png"))
        if len(screens) <= limit:
            return
        for path in screens[: len(screens) - limit]:
            try:
                path.unlink()
            except OSError:
                pass

    def _post_playback(self, steps: list[Any]) -> None:
        """Replay action frames in the viewer at game speed (fallback when not streaming)."""
        if self.viewer is None or not steps:
            return
        if self.emulator is not None and getattr(self.emulator, "streaming", False):
            return  # frames were already presented live by the frame hook
        frames = [getattr(step, "frame", None) for step in steps]
        frames = [frame.rgba for frame in frames if isinstance(frame, Frame)]
        if len(frames) < 2:
            return
        fps = 60.0 / max(1, int(self.cfg.emulator.action_frames))
        try:
            self.viewer.post_playback(frames, fps=fps)
        except Exception:
            pass

    def _record_video(self, steps: list[Any]) -> None:
        cfg = self.cfg
        if not cfg.viewer.record_video:
            return
        if self._video_error:
            return
        frames = [getattr(step, "frame", None) for step in steps]
        frames = [frame for frame in frames if isinstance(frame, Frame)]
        if not frames:
            return
        try:
            import cv2  # local import: only needed when recording

            if self._video_writer is None:
                path = Path(cfg.viewer.record_video)
                if not path.is_absolute():
                    path = Path(self.cfg.runtime.run_dir or ".") / path
                path.parent.mkdir(parents=True, exist_ok=True)
                fps = max(1.0, round(60.0 / max(1, cfg.emulator.action_frames), 2))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, fps, (contracts.SCREEN_WIDTH, contracts.SCREEN_HEIGHT))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV VideoWriter could not open the output file")
                self._video_writer = writer
                self._video_path = str(path)
                self._log_event("video_recording_started", path=self._video_path, fps=fps)
            for frame in frames:
                rgb = frame.rgba[:, :, :3]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self._video_writer.write(bgr)
        except Exception as exc:  # noqa: BLE001
            self._video_error = f"{type(exc).__name__}: {exc}"
            self._log_event("video_recording_failed", error=self._video_error)
            self._close_video()

    def _close_video(self) -> None:
        if self._video_writer is not None:
            try:
                self._video_writer.release()
                self._log_event("video_recording_finished", path=self._video_path)
            except Exception:
                pass
            self._video_writer = None

    # ------------------------------------------------------------------ #
    # Pause / stop / sleep
    # ------------------------------------------------------------------ #
    def _stop_requested(self) -> bool:
        if self._stop_flag:
            return True
        if self.viewer is not None and self.viewer.stop_event.is_set():
            self._stop_reason = self._stop_reason or "viewer quit request"
            return True
        stop_file = self.cfg.runtime.stop_file
        if stop_file and Path(stop_file).exists():
            self._stop_reason = self._stop_reason or f"stop file present: {stop_file}"
            return True
        return False

    def _is_paused(self) -> bool:
        if self.viewer is not None and self.viewer.pause_event.is_set():
            return True
        pause_file = self.cfg.runtime.pause_file
        return bool(pause_file and Path(pause_file).exists())

    def _check_controls(self) -> None:
        if self._stop_requested():
            raise StopRequested(self._stop_reason or "stop requested")
        self._handle_save_request()
        if self._is_paused():
            self.state.paused = True
            self.state.status = "paused"
            self._publish(self._last_frame, force=True)
            while self._is_paused():
                if self._stop_requested():
                    raise StopRequested(self._stop_reason or "stop requested")
                self._handle_save_request()
                self._publish(self._last_frame, force=True)
                time.sleep(0.2)
            self.state.paused = False
            if self.state.status == "paused":
                self.state.status = "running"
            self._publish(self._last_frame, force=True)

    def _handle_save_request(self) -> None:
        if self.viewer is not None and self.viewer.save_event.is_set():
            self.viewer.save_event.clear()
            try:
                meta = self._save_checkpoint("user_request")
                self._log_event("save_requested", checkpoint=meta.id)
                self.state.add_history("checkpoint", f"user requested save: {meta.id}")
            except Exception as exc:  # noqa: BLE001
                self._log_event("save_request_failed", error=f"{type(exc).__name__}: {exc}")

    def _sleep(self, seconds: float) -> None:
        deadline = _now() + max(0.0, float(seconds))
        while _now() < deadline:
            if self._stop_requested():
                raise StopRequested(self._stop_reason or "stop requested")
            time.sleep(min(0.5, max(0.05, deadline - _now())))

    # ------------------------------------------------------------------ #
    # Publishing status (viewer / headless console)
    # ------------------------------------------------------------------ #
    def _publish(self, frame: Frame | None = None, *, force: bool = False) -> None:
        if frame is not None:
            self._last_frame = frame
        cfg = self.cfg
        budget_state = self.budget.state() if self.budget is not None else contracts.BudgetState()
        budget_left = max(0.0, float(cfg.budget.max_usd_per_run) - budget_state.cost_usd)
        nav_summary = self._navigation_summary()
        nav_extra: dict[str, Any] = {}
        if nav_summary:
            pose = nav_summary.get("pose") or {}
            nav_extra["导航 Navigation"] = (
                f"区域 {nav_summary.get('area_id', '?')} · 位置 ({pose.get('x', '?')},{pose.get('y', '?')}) "
                f"· 置信度 {float(pose.get('confidence', 0.0) or 0.0):.2f}"
            )
            nav_extra["地图 Map"] = (
                f"可走 {nav_summary.get('walkable', 0)} / 障碍 {nav_summary.get('blocked', 0)} / "
                f"未知边界 {nav_summary.get('unknown_frontier', 0)} / 覆盖 {float(nav_summary.get('coverage_ratio', 0.0) or 0.0):.0%}"
            )
            nav_extra["世界 World"] = (
                f"{nav_summary.get('areas_total', 0)} 区域 / "
                f"{(nav_summary.get('world') or {}).get('connections', 0)} 连接"
            )
            last_event = str(nav_summary.get("last_event") or "")
            if last_event:
                nav_extra["导航事件 Nav event"] = _clip(last_event, 90)
        status = ViewerStatus(
            status=self.state.status,
            goal=self.state.goal,
            last_action=self.state.last_action,
            decision=self.state.last_decision,
            frame=self.state.frame_index,
            turns=self.state.llm_turns,
            buttons=self.state.buttons_pressed,
            places=self.state.places_discovered,
            discoveries=self.state.discoveries,
            map_cells=int(nav_summary.get("cells_total", 0) or 0),
            calls=self.state.llm_calls or budget_state.calls,
            cost_usd=budget_state.cost_usd,
            tokens=budget_state.total_tokens,
            budget_left_usd=budget_left,
            model=str(getattr(self.brain, "model", "")),
            extra=nav_extra,
        )
        if self.viewer is not None:
            if self._last_frame is not None:
                try:
                    self.viewer.post(self._last_frame.rgba, status)
                except Exception:
                    pass
            if cfg.navigation.enabled and cfg.navigation.viewer_map_panel:
                self._post_nav_map()
        else:
            now = _now()
            status_changed = status.status != self._last_printed_status
            if status_changed or (force and now - self._last_print_at >= 1.0) or now - self._last_print_at >= 5.0:
                self._last_print_at = now
                self._last_printed_status = status.status
                print(self._format_status_line(status), flush=True)

    def _format_status_line(self, status: ViewerStatus) -> str:
        elapsed = int(_now() - self.state.started_at)
        return (
            f"[{status.status:<13s}] turn={status.turns:<4d} frame={status.frame:<7d} "
            f"btns={status.buttons:<5d} calls={status.calls:<4d} ${status.cost_usd:.4f} "
            f"places={status.places:<2d} t={elapsed}s | goal={_clip(status.goal, 46) or '-'} "
            f"| {_clip(status.last_action, 60) or '-'}"
        )

    # ------------------------------------------------------------------ #
    # Audit / metrics / report
    # ------------------------------------------------------------------ #
    def _log_event(self, event: str, **fields: Any) -> None:
        if self.audit is not None:
            self.audit.log("events", {"event": event, **fields})

    def _write_metrics(self, *, final: bool = False) -> None:
        if self._run_dir is None:
            return
        memory_stats: dict[str, Any] = {}
        if self.memory is not None:
            try:
                memory_stats = self.memory.stats()
            except Exception:
                memory_stats = {}
        budget_snapshot = self.budget.snapshot() if self.budget is not None else {}
        metrics = {
            "run_id": self.state.run_id,
            "updated_at": _now(),
            "status": self.state.status,
            "final": bool(final),
            "mode": str(getattr(self.brain, "mode", "unknown")),
            "real_api": self.real_api,
            "model": str(getattr(self.brain, "model", "")),
            "duration_s": round(_now() - self._started_at, 2),
            "turns": self.state.llm_turns,
            "frames": self.state.frame_index,
            "buttons": self.state.buttons_pressed,
            "action_batches": self.state.action_batches,
            "llm_calls": self.state.llm_calls,
            "api_failures": self.state.api_failures,
            "checkpoints": self.state.checkpoints,
            "goal": self.state.goal,
            "last_action": self.state.last_action,
            "last_decision": self.state.last_decision,
            "loop_warnings": self.state.loop_warnings,
            "memory": memory_stats,
            "navigation": self._navigation_summary(),
            "budget": budget_snapshot,
        }
        try:
            write_json_atomic(self._run_dir / "metrics.json", metrics)
        except Exception:
            pass

    def _write_run_json(self, status: str, reason: str = "") -> None:
        if self._run_dir is None:
            return
        payload = {
            "run_id": self.state.run_id or self._run_dir.name,
            "agent_version": __version__,
            "status": status,
            "reason": reason,
            "started_at": self._started_at,
            "started_at_iso": _iso(self._started_at),
            "updated_at_iso": _iso(),
            "mode": str(getattr(self.brain, "mode", "unknown")),
            "real_api": self.real_api,
            "model": str(getattr(self.brain, "model", "")),
            "config": self.cfg.to_safe_dict(),
            "rom": {"path": self.cfg.emulator.rom, "sha1": self._rom_sha1},
            "initial_state": {"path": self.cfg.emulator.state, "sha256": self._state_sha256},
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pid": os.getpid(),
                "hostname": platform.node(),
            },
            "git": self._git,
        }
        try:
            write_json_atomic(self._run_dir / "run.json", payload)
        except Exception:
            pass

    def _safe_record_error(self, context: str, exc: Exception | None = None) -> None:
        if self._run_dir is None:
            return
        try:
            directory = self._run_dir / "errors" / f"error_{int(_now())}_{context.split()[0]}"
            directory.mkdir(parents=True, exist_ok=True)
            text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else context
            (directory / "traceback.txt").write_text(
                f"context: {context}\nwhen: {_iso()}\n\n{text}\n", encoding="utf-8"
            )
            (directory / "meta.json").write_text(
                json.dumps(
                    {
                        "context": context,
                        "turn": self._turn,
                        "status": self.state.status,
                        "frame": self.state.frame_index,
                        "error": f"{type(exc).__name__}: {exc}" if exc else context,
                        "when": _now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if self._last_frame is not None:
                try:
                    observation.save_png(self._last_frame, directory / "screen.png")
                except Exception:
                    pass
            if self.emulator is not None:
                try:
                    state_bytes = self.emulator.save_state_bytes()
                    write_bytes_atomic(directory / "emulator_state.state", state_bytes)
                except Exception:
                    pass
            self._log_event("error_snapshot", context=context, directory=str(directory))
        except Exception:
            pass

    def _write_report(self, result: RunResult) -> None:
        if self._run_dir is None:
            return
        memory_snapshot: dict[str, Any] = {}
        if self.memory is not None:
            try:
                memory_snapshot = self.memory.snapshot()
            except Exception:
                memory_snapshot = {}
        memory_data = memory_snapshot.get("data") if isinstance(memory_snapshot, dict) else None
        if not isinstance(memory_data, dict):
            memory_data = memory_snapshot if isinstance(memory_snapshot, dict) else {}
        lines: list[str] = []
        lines.append(f"# DeepSeek LLM Agent — 运行报告 / Run report")
        lines.append("")
        lines.append(f"- run_id: `{result.run_id}`")
        lines.append(f"- 状态 status: **{result.status}** ({result.reason or 'ok'})")
        lines.append(f"- 运行模式 mode: `{result.mode}`")
        if result.real_api:
            lines.append("- 证据类型: **真实 DeepSeek API 驱动** (model=`{}`)".format(result.model))
        else:
            lines.append(f"- ⚠️ **{SCRIPTED_DISCLAIMER}** — 不得作为真实 API/游戏验收证据")
        lines.append(f"- 开始 started: {_iso(self._started_at)}")
        lines.append(f"- 结束 finished: {_iso()}")
        lines.append(f"- 时长 duration: {result.duration_s:.1f}s")
        lines.append("")
        lines.append("## 运行统计 Runtime")
        lines.append("")
        lines.append(f"- DeepSeek 决策轮数 turns: {result.turns}")
        lines.append(f"- 模拟器帧数 frames: {result.frames}")
        lines.append(f"- 按键次数 buttons: {result.buttons}")
        lines.append(f"- 动作批次 action batches: {result.action_batches}")
        lines.append(f"- 检查点 checkpoints: {result.checkpoints}")
        lines.append(f"- 当前目标 goal: {result.goal or '-'}")
        lines.append(f"- 最近决策 last decision: {_clip(result.last_decision, 500) or '-'}")
        lines.append(f"- 最近动作 last action: {_clip(result.last_action, 500) or '-'}")
        lines.append("")
        lines.append("## API 用量与费用 Cost")
        lines.append("")
        budget_state = self.budget.state() if self.budget is not None else contracts.BudgetState()
        lines.append(f"- API 调用 calls: {budget_state.calls}")
        lines.append(f"- 输入 tokens: {budget_state.prompt_tokens:,}")
        lines.append(f"- 输出 tokens: {budget_state.completion_tokens:,}")
        lines.append(f"- 总 tokens: {budget_state.total_tokens:,}")
        lines.append(f"- 累计估算费用 cost: ${budget_state.cost_usd:.6f}")
        lines.append(
            f"- 单次运行预算上限: ${self.cfg.budget.max_usd_per_run:.2f} / "
            f"调用上限 {self.cfg.budget.max_calls_per_run}"
        )
        lines.append("")
        lines.append("## 游戏进度（仅来自实际观察/记忆）Game progress (observation-derived only)")
        lines.append("")
        lines.append(f"- 已发现地点 places: {result.places}")
        lines.append(f"- 发现记录 discoveries: {result.discoveries}")
        goal_data = memory_data.get("goal")
        if isinstance(goal_data, dict) and (goal_data.get("text") or goal_data.get("goal")):
            lines.append(f"- 记忆中的目标: {goal_data.get('text') or goal_data.get('goal')}")
        lines.append("")
        lines.append("### 已发现地点 Places (agent-recorded)")
        places = memory_data.get("places")
        if isinstance(places, list) and places:
            for place in places[-30:]:
                if isinstance(place, dict):
                    name = place.get("name") or place.get("place") or "?"
                    description = place.get("description") or ""
                    lines.append(f"- **{name}** — {_clip(str(description), 200)}")
                else:
                    lines.append(f"- {place}")
        else:
            lines.append("- （尚未记录）")
        lines.append("")
        lines.append("### 发现与教训 Discoveries / lessons (agent-recorded, latest 30)")
        for key in ("discoveries", "lessons", "dialogues"):
            items = memory_data.get(key)
            if isinstance(items, list) and items:
                lines.append(f"#### {key}")
                for item in items[-10:]:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("summary") or item.get("content") or ""
                        kind = item.get("kind") or item.get("speaker") or ""
                        lines.append(f"- [{kind}] {_clip(str(text), 260)}")
                    else:
                        lines.append(f"- {_clip(str(item), 260)}")
        lines.append("")
        lines.append("## Navigation v2 — 永久地图与探索（纯视觉，自建）Navigation map (self-built)")
        lines.append("")
        try:
            nav_summary = self._navigation_summary()
            if nav_summary:
                pose = nav_summary.get("pose") or {}
                lines.append(
                    f"- 当前区域 area: `{nav_summary.get('area_id', '?')}` "
                    f"({nav_summary.get('area_kind', '?')})"
                )
                lines.append(
                    f"- 估计位置 pose: ({pose.get('x', '?')},{pose.get('y', '?')}) "
                    f"facing={pose.get('facing')} confidence={pose.get('confidence', 0.0)}"
                )
                coverage = nav_summary.get("coverage") or nav_summary
                lines.append(
                    f"- 覆盖 coverage: walkable={coverage.get('walkable', 0)} "
                    f"blocked={coverage.get('blocked', 0)} "
                    f"frontier={coverage.get('unknown_frontier', 0)} "
                    f"ratio={float(coverage.get('coverage_ratio', 0.0) or 0.0):.1%}"
                )
                world = nav_summary.get("world") or {}
                lines.append(
                    f"- 世界模型 world: {world.get('areas', 0)} 个区域, "
                    f"{world.get('connections', 0)} 条连接"
                )
                area_rows = world.get("areas") or []
                if isinstance(area_rows, list):
                    for row in area_rows[-12:]:
                        if isinstance(row, (list, tuple)) and len(row) >= 4:
                            lines.append(
                                f"  - `{row[0]}` {row[1]} visits={row[2]} walkable={row[3]}"
                            )
                        elif isinstance(row, dict):
                            lines.append(
                                f"  - `{row.get('area_id')}` {row.get('kind', '?')} "
                                f"visits={row.get('visits', 0)} walkable={row.get('walkable', 0)}"
                            )
                ascii_map = ""
                try:
                    if self.navigation is not None:
                        ascii_map = str(self.navigation.local_map(radius=8).get("ascii") or "")
                except Exception:  # noqa: BLE001
                    ascii_map = ""
                if ascii_map:
                    lines.append("")
                    lines.append("```")
                    lines.append(ascii_map)
                    lines.append("```")
            else:
                lines.append("- （导航系统未启用或尚无地图）")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- （导航地图不可用: {type(exc).__name__}）")
        lines.append("")
        lines.append("## 近期操作历史 Recent history")
        lines.append("")
        for entry in self.state.recent_history(20):
            lines.append(f"- `{entry.kind}` {_clip(entry.summary, 240)}")
        lines.append("")
        lines.append("## 异常与自纠错 Anomalies / recovery")
        lines.append("")
        warnings = self.state.loop_warnings
        lines.append(f"- 最近的循环告警: {warnings if warnings else '无'}")
        lines.append(f"- 累计 API 失败: {self.state.api_failures}")
        lines.append(f"- 当前连续失败: {self.state.consecutive_failures}")
        errors_dir = self._run_dir / "errors"
        error_dirs = sorted(p.name for p in errors_dir.glob("error_*")) if errors_dir.is_dir() else []
        lines.append(f"- 错误现场快照: {', '.join(error_dirs[-5:]) if error_dirs else '无'}")
        lines.append("")
        lines.append("## 文件与审计 Files")
        lines.append("")
        lines.append(f"- 运行目录: `{self._run_dir}`")
        lines.append("- `audit/decisions.jsonl` — 每次 DeepSeek 决策（理由、工具调用、tokens、成本）")
        lines.append("- `audit/actions.jsonl` — 每个工具调用的执行结果")
        lines.append("- `audit/api_attempts.jsonl` — API 重试与错误（已脱敏）")
        lines.append("- `audit/errors.jsonl` / `audit/events.jsonl` — 错误与运行事件")
        lines.append("- `screens/` — 动作批次截图；`checkpoints/` — 合法模拟器状态与记忆快照")
        lines.append("")
        lines.append("## 独立验证 Independent verification (RAM probe, report-only)")
        lines.append("")
        lines.append("决策回路从不读取 RAM。以下命令可在离线状态下验证最终检查点的真实游戏进度：")
        lines.append("")
        lines.append("```powershell")
        lines.append(".\\\\.venv\\\\Scripts\\\\python.exe scripts\\\\llm_agent_progress.py "
                     f"--run-dir \"{self._run_dir}\"")
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append(f"*Generated by llm_agent v{__version__} at {_iso()}*")
        report_path = self._run_dir / "report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # machine-readable summary next to the report
        summary = {
            "result": result.to_dict(),
            "budget": budget_state.to_dict(),
            "memory": memory_snapshot if isinstance(memory_snapshot, dict) else {},
            "history": [entry.to_dict() for entry in self.state.recent_history(30)],
            "loop_warnings": warnings,
        }
        write_json_atomic(self._run_dir / "report.json", summary)
