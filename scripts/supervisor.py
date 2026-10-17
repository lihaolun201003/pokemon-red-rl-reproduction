#!/usr/bin/env python3
"""Lightweight, read-only supervisor for the Pokemon Red PPO training runs.

The supervisor never starts, stops, kills or writes into a training run directory.
It only reads the state file / run logs / checkpoints, samples psutil and nvidia-smi,
and appends its own snapshots to ``artifacts/training/seed_0/monitor/status.jsonl``.

Subcommands
-----------
status             one-shot snapshot (human readable or --json)
watch              blocking wait for a stage result / severe anomaly / timeout
verify-checkpoint  structural + model rebuild validation of a checkpoint (no writes)
prune-checkpoints  move old checkpoints to DIR/_archive (dry-run supported)
stop-request       touch the stop_file so scripts/train.py stops at the next update boundary
launch             print the command / Start-Process snippet only (never launches)

Exit codes
----------
status           0 snapshot printed (even if alerts exist; inspect "alerts")
watch            0 stage result reached target; 1 severe anomaly; 2 timeout;
                 4 stage ended cleanly before target (e.g. stop_file)
verify-checkpoint 0 valid; 1 validation/config mismatch; 2 usage/IO error
prune-checkpoints 0 planned/performed (dry-run always 0); 1 per-file move errors
stop-request     0 touched (or already existed); 2 usage/state error
launch           0 command printed; 2 refused / usage error
Any command      2 on unhandled usage or path-safety error
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
import re
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime

import psutil

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
# The script lives in scripts/; make the project packages importable when run as
# `python scripts/supervisor.py` from any working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_STATE = ROOT / "artifacts" / "training" / "seed_0" / "training_state.json"
DEFAULT_MONITOR_DIR = ROOT / "artifacts" / "training" / "seed_0" / "monitor"

STATUS_SCHEMA = "supervisor_status/1"
VERIFY_SCHEMA = "supervisor_verify/1"
PROJECT_STEP_TARGET = 1_000_000

CHECKPOINT_MIN_BYTES = 1_000_000
DISK_MIN_FREE_BYTES = 10 * 1024 ** 3
RAM_MIN_AVAILABLE_BYTES = int(1.5 * 1024 ** 3)
GPU_TEMP_MAX_C = 85.0
GPU_MEM_MIN_FREE_MIB = 1024.0
STALE_UPDATE_SECONDS = 600.0
NO_NEW_STEPS_SECONDS = 120.0
STAGE_START_GRACE_SECONDS = 60.0

SEV_CRITICAL = "critical"
SEV_WARNING = "warning"
SEV_INFO = "info"

# Alerts that make `watch` exit 1 on the first observation.
IMMEDIATE_FATAL_CODES = frozenset({
    "process_exited_without_result",
    "metrics_nonfinite",
    "parameter_delta_nonpositive",
    "checkpoint_too_small",
})
# Alerts that make `watch` exit 1 after they are seen on two consecutive rounds.
SUSTAINED_CRITICAL_CODES = frozenset({
    "stale_metrics",
    "checkpoint_expected_missing",
    "state_checkpoint_missing",
    "low_disk",
    "gpu_hot",
    "low_ram",
})

# Keys injected into checkpoint["configs"] at runtime after config.yaml was written,
# so a strict config.yaml comparison legitimately differs on them.
RUNTIME_INJECTED_PATHS = frozenset({
    ("environment", "reset_params", "monitor_more_stats"),
})

CHECKPOINT_NAME_RE = re.compile(r"^checkpoint_(\d+)\.pt$")
CURATED_METRIC_PREFIXES = (
    "loss/", "other/", "decay/", "rollout/", "progress/", "validation/",
    "time/", "episode/", "game/", "actions/share_",
)


class SupervisorError(RuntimeError):
    """Usage or path-safety error; reported on stderr with exit code 2."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _as_int(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_unix(value):
    """Parse an ISO-8601 timestamp (with or without offset) into a unix float."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, OSError):
        return None


def _fmt_duration(seconds) -> str:
    if seconds is None:
        return "unknown"
    seconds = float(seconds)
    if seconds < 0:
        return "0s"
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 90:
        return f"{int(minutes)}m {int(sec):02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{int(hours)}h {int(minutes):02d}m"
    days, hours = divmod(hours, 24)
    return f"{int(days)}d {int(hours):02d}h"


def _fmt_bytes(size) -> str:
    if size is None:
        return "unknown"
    size = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _pct(value, digits=1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}%"


def _safe_path(value, what: str) -> Path:
    """Resolve a user/state supplied path and require it to stay inside the project."""
    if value is None or str(value).strip() == "":
        raise SupervisorError(f"{what}: empty path")
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    try:
        resolved = path.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise SupervisorError(f"{what}: cannot resolve {path}: {exc}") from exc
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SupervisorError(f"{what}: {resolved} is outside the project directory {ROOT}") from exc
    return resolved


def _tail_lines(path: Path, max_lines: int = 10, initial_chunk: int = 131072):
    """Return up to the last ``max_lines`` complete text lines without reading huge files."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size <= 0:
        return []
    chunk = min(size, initial_chunk)
    with path.open("rb") as handle:
        while True:
            handle.seek(size - chunk)
            data = handle.read()
            lines = data.splitlines()
            complete = lines[1:] if chunk < size else lines
            if len(complete) >= max_lines or chunk >= size:
                return [line.decode("utf-8", errors="replace") for line in complete[-max_lines:]]
            chunk = min(size, chunk * 4)


def _read_json(path: Path):
    """Read a JSON document with a few retries for concurrent writers."""
    if not path.is_file():
        return None, "missing"
    last_error = None
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig")), None
        except json.JSONDecodeError as exc:
            last_error = f"json_decode_error: {exc}"
        except OSError as exc:
            last_error = f"os_error: {type(exc).__name__}: {exc}"
        if attempt < 2:
            time.sleep(0.15)
    return None, last_error


def _load_state(state_path: Path):
    raw, error = _read_json(state_path)
    info = {"path": str(state_path), "exists": state_path.is_file(), "error": error,
            "fields": []}
    if not isinstance(raw, dict):
        if info["exists"] and error is None:
            info["error"] = "not_a_json_object"
        return {}, info
    info["fields"] = sorted(raw.keys())
    return raw, info


def _plain(value):
    """Recursively convert ruamel mappings/sequences into plain dict/list values."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load_yaml_config(path: Path):
    from ruamel.yaml import YAML
    data = YAML().load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, Mapping):
        raise ValueError("config is not a YAML mapping")
    return _plain(data)


# ---------------------------------------------------------------------------
# process inspection (PID reuse protection)
# ---------------------------------------------------------------------------
def _is_train_cmdline(cmdline) -> bool:
    return "scripts/train.py" in " ".join(cmdline or []).replace("\\", "/").lower()


def _extract_out_arg(cmdline):
    """Return the value of the ``--out`` argument from a command line, if present."""
    tokens = list(cmdline or [])
    for index, token in enumerate(tokens):
        if token in ("--out", "-o") and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _cmd_belongs_to_run(cmdline, stage_out: Path) -> bool:
    """True when the ``--out`` argument of this command line points at ``stage_out``.

    Matching on the bare run-directory name inside the whole argv is not safe: a later
    stage resumes from ``.../stage_c_300k/checkpoint_...`` and that path also contains
    the earlier stage directory name. The ``--out`` argument is authoritative.
    """
    target_name = stage_out.name.lower()
    target_full = str(stage_out).replace("\\", "/").lower()
    try:
        target_relative = str(stage_out.relative_to(ROOT)).replace("\\", "/").lower()
    except ValueError:
        target_relative = None
    out_value = _extract_out_arg(cmdline)
    if out_value is not None:
        normalized = str(out_value).replace("\\", "/").lower().rstrip("/")
        if normalized == target_full or normalized == target_name:
            return True
        if target_relative is not None and normalized == target_relative:
            return True
        return False
    # No --out argument at all (unexpected): fall back to a token match, but never a
    # substring match against a longer path.
    tokens = [str(token).replace("\\", "/").lower().rstrip("/") for token in (cmdline or [])]
    return any(token == target_full or token == target_name or token == target_relative
               for token in tokens)


def _inspect_process(state_pid, stage_out: Path | None):
    info = {
        "state_pid": state_pid,
        "state_pid_alive": False,
        "state_pid_cmdline_match": None,
        "live_matching_pids": [],
        "matched_cmdlines": [],
        "running": False,
        "pid_mismatch": False,
        "scan_error": None,
    }
    out_tokens = []
    if stage_out is not None:
        out_tokens.append(str(stage_out).replace("\\", "/").lower())
        out_tokens.append(stage_out.name.lower())

    matches = []
    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                name = (proc.info.get("name") or "").lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if not _is_train_cmdline(cmdline):
                continue
            # A parent shell/agent wrapper can contain the full training command in its
            # own argv (e.g. a DSH job runner). Only python interpreters are candidates.
            if not (name.startswith("python") or name == "py.exe"
                    or name.startswith("pypy")):
                continue
            command = " ".join(cmdline)
            if stage_out is not None:
                out_hit = _cmd_belongs_to_run(cmdline, stage_out)
            else:
                normalized = command.replace("\\", "/").lower()
                out_hit = any(token and token in normalized for token in out_tokens)
            matches.append({"pid": proc.info["pid"], "out_hit": out_hit, "cmdline": command})
    except (psutil.Error, OSError) as exc:  # pragma: no cover - defensive
        info["scan_error"] = f"{type(exc).__name__}: {exc}"

    live_pids = sorted({item["pid"] for item in matches if item["out_hit"]})
    info["live_matching_pids"] = live_pids
    info["matched_cmdlines"] = [item["cmdline"][:400] for item in matches]
    if stage_out is None:
        live_pids = sorted({item["pid"] for item in matches})
        info["live_matching_pids"] = live_pids
    info["running"] = bool(live_pids)

    if state_pid is not None:
        try:
            proc = psutil.Process(state_pid)
            info["state_pid_alive"] = proc.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            info["state_pid_alive"] = False
        except ValueError:
            info["state_pid_alive"] = False
        if info["state_pid_alive"]:
            matched = next((item for item in matches if item["pid"] == state_pid), None)
            info["state_pid_cmdline_match"] = bool(matched and matched["out_hit"])
            if info["state_pid_cmdline_match"]:
                info["running"] = True
                if state_pid not in live_pids:
                    live_pids = sorted(set(live_pids) | {state_pid})
                    info["live_matching_pids"] = live_pids
        if info["running"] and state_pid not in info["live_matching_pids"]:
            info["pid_mismatch"] = True
    return info


# ---------------------------------------------------------------------------
# system resources
# ---------------------------------------------------------------------------
def _query_gpu(attempts: int = 2, timeout: float = 5.0):
    """Return (gpu dict, error string). Retried once; never raises."""
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    errors = []
    for attempt in range(attempts):
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=timeout, check=False)
            if completed.returncode != 0:
                errors.append(f"exit={completed.returncode} stderr="
                              f"{(completed.stderr or '').strip()[:200]}")
            else:
                lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
                if not lines:
                    errors.append("empty output")
                else:
                    parts = [part.strip() for part in lines[0].split(",")]
                    if len(parts) < 5:
                        errors.append(f"unexpected output {lines[0]!r}")
                    else:
                        numbers = []
                        for part in parts[:5]:
                            try:
                                numbers.append(float(part))
                            except ValueError:
                                numbers.append(None)
                        used, free = numbers[1], numbers[2]
                        gpu = {
                            "utilization_percent": numbers[0],
                            "memory_used_mib": used,
                            "memory_free_mib": free,
                            "memory_total_mib": (used + free) if None not in (used, free) else None,
                            "temperature_c": numbers[3],
                            "power_w": numbers[4],
                            "raw": lines[0],
                        }
                        return gpu, None
        except FileNotFoundError:
            errors.append("nvidia-smi not found")
            break
        except subprocess.TimeoutExpired:
            errors.append(f"timeout after {timeout}s")
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            errors.append(f"{type(exc).__name__}: {exc}")
        if attempt + 1 < attempts:
            time.sleep(0.5)
    return None, " | ".join(errors) or "unknown gpu query failure"


def _query_system(cpu_interval: float, include_gpu: bool = True):
    system = {
        "cpu_percent": None,
        "ram": None,
        "swap": None,
        "disk": None,
        "gpu": None,
        "gpu_error": None,
        "cpu_interval_seconds": cpu_interval,
    }
    try:
        if cpu_interval and cpu_interval > 0:
            system["cpu_percent"] = psutil.cpu_percent(interval=cpu_interval)
        else:
            system["cpu_percent"] = psutil.cpu_percent(interval=None)
    except (psutil.Error, OSError):
        system["cpu_percent"] = None
    try:
        memory = psutil.virtual_memory()
        system["ram"] = {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_bytes": memory.used,
            "percent": memory.percent,
        }
    except (psutil.Error, OSError):
        system["ram"] = None
    try:
        swap = psutil.swap_memory()
        system["swap"] = {
            "total_bytes": swap.total,
            "used_bytes": swap.used,
            "free_bytes": swap.free,
            "percent": swap.percent,
        }
    except (psutil.Error, OSError):
        system["swap"] = None
    try:
        usage = psutil.disk_usage(str(ROOT))
        system["disk"] = {"total_bytes": usage.total, "used_bytes": usage.used,
                          "free_bytes": usage.free, "percent": usage.percent,
                          "path": str(ROOT)}
    except (psutil.Error, OSError):
        system["disk"] = None
    if include_gpu:
        system["gpu"], system["gpu_error"] = _query_gpu()
    return system


# ---------------------------------------------------------------------------
# run-directory readers
# ---------------------------------------------------------------------------
def _metrics_info(stage_out: Path | None):
    info = {
        "path": None, "exists": False, "size_bytes": None, "mtime": None,
        "rows_seen": 0, "last": None, "last_update_unix": None,
        "last_update_age_seconds": None, "nonfinite_keys": [], "error": None,
    }
    if stage_out is None:
        return info
    path = stage_out / "metrics.jsonl"
    info["path"] = str(path)
    try:
        stat = path.stat()
    except OSError as exc:
        info["error"] = type(exc).__name__
        return info
    info["exists"] = True
    info["size_bytes"] = stat.st_size
    info["mtime"] = stat.st_mtime
    rows = []
    if stat.st_size > 0:
        try:
            lines = _tail_lines(path, 8)
        except OSError as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            lines = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    info["rows_seen"] = len(rows)
    if rows:
        info["last"] = rows[-1]
        info["nonfinite_keys"] = sorted(
            key for key, value in rows[-1].items()
            if isinstance(value, float) and not math.isfinite(value))
    unix_value = _as_float((info["last"] or {}).get("time/unix"))
    if unix_value is not None:
        info["last_update_unix"] = unix_value
        info["last_update_age_seconds"] = max(0.0, time.time() - unix_value)
    elif stat.st_mtime:
        info["last_update_age_seconds"] = max(0.0, time.time() - stat.st_mtime)
    return info


def _checkpoint_info(stage_out: Path | None):
    info = {
        "dir": str(stage_out) if stage_out is not None else None,
        "count": 0, "entries": [], "latest": None,
        "total_size_bytes": 0, "error": None,
    }
    if stage_out is None or not stage_out.is_dir():
        return info
    entries = []
    try:
        candidates = list(stage_out.glob("checkpoint_*.pt"))
    except OSError as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    for path in candidates:
        match = CHECKPOINT_NAME_RE.match(path.name)
        if match is None or not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append({"name": path.name, "steps": int(match.group(1)),
                        "size_bytes": stat.st_size, "size_mb": round(stat.st_size / 1024 ** 2, 2),
                        "mtime": stat.st_mtime, "path": str(path)})
    entries.sort(key=lambda entry: entry["steps"])
    info["entries"] = entries
    info["count"] = len(entries)
    info["latest"] = entries[-1] if entries else None
    info["total_size_bytes"] = sum(entry["size_bytes"] for entry in entries)
    return info


def _result_info(stage_out: Path | None):
    info = {"path": None, "exists": False, "error": None}
    if stage_out is None:
        return info
    path = stage_out / "result.json"
    info["path"] = str(path)
    if not path.is_file():
        return info
    info["exists"] = True
    data, error = _read_json(path)
    if error:
        info["error"] = error
        return info
    if not isinstance(data, dict):
        info["error"] = "not_a_json_object"
        return info
    info.update({
        "environment_steps": _as_int(data.get("environment_steps")),
        "updates_completed": _as_int(data.get("updates_completed")),
        "stop_reason": data.get("stop_reason"),
        "checkpoint": data.get("checkpoint"),
        "raw": {key: data.get(key) for key in
                ("real_game", "updates_completed", "environment_steps", "stop_reason",
                 "performance_claim")},
    })
    return info


def _discover_stages(experiment_dir: Path):
    """Find stage run directories (config.yaml present) sorted by most recently updated."""
    found = []
    if not experiment_dir.is_dir():
        return found
    try:
        children = sorted(experiment_dir.iterdir())
    except OSError:
        return found
    for child in children:
        if not child.is_dir() or child.name.startswith("_") or child.name == "monitor":
            continue
        if not (child / "config.yaml").is_file():
            continue
        mtimes = []
        for name in ("metrics.jsonl", "result.json", "environment.json"):
            try:
                candidate = child / name
                if candidate.is_file():
                    mtimes.append(candidate.stat().st_mtime)
            except OSError:
                continue
        latest_steps = None
        metrics_path = child / "metrics.jsonl"
        if metrics_path.is_file():
            try:
                lines = _tail_lines(metrics_path, 1)
                if lines:
                    row = json.loads(lines[-1])
                    if isinstance(row, dict):
                        latest_steps = _as_int(row.get("progress/environment_steps"))
            except (OSError, json.JSONDecodeError):
                latest_steps = None
        if not mtimes:
            try:
                mtimes.append(child.stat().st_mtime)
            except OSError:
                continue
        found.append({"out_dir": child, "latest_steps": latest_steps,
                      "has_result": (child / "result.json").is_file(),
                      "mtime": max(mtimes)})
    found.sort(key=lambda item: item["mtime"])
    return found


def _training_shape(state: Mapping, stage_out: Path | None):
    """batch per update and checkpoint interval (from state first, config.yaml fallback)."""
    batch = _as_int(dict(state).get("batch_per_update"))
    interval = _as_int(dict(state).get("checkpoint_interval_updates"))
    source = "state"
    if batch and interval:
        return batch, interval, source
    if stage_out is not None and (stage_out / "config.yaml").is_file():
        try:
            config = _load_yaml_config(stage_out / "config.yaml")
            sampler = config.get("sampler") or {}
            model = config.get("model") or {}
            if not batch:
                batch = _as_int(sampler.get("n_workers")) * _as_int(sampler.get("worker_steps")) \
                    if _as_int(sampler.get("n_workers")) and _as_int(sampler.get("worker_steps")) else None
            if not interval:
                interval = _as_int(model.get("checkpoint_interval"))
            source = "config.yaml"
        except (OSError, ValueError, TypeError):
            pass
    return batch, interval, source if (batch or interval) else None


def _curate_metrics(row: Mapping | None):
    if not isinstance(row, Mapping):
        return {}
    curated = {}
    for key, value in row.items():
        if key == "checkpoint":
            continue
        if isinstance(key, str) and key.startswith(CURATED_METRIC_PREFIXES):
            if isinstance(value, float) and not math.isfinite(value):
                curated[key] = str(value)
            else:
                curated[key] = value
    return curated


# ---------------------------------------------------------------------------
# snapshot assembly
# ---------------------------------------------------------------------------
def _normalise_checkpoint(value):
    if isinstance(value, Mapping):
        return {"path": value.get("path"),
                "steps": _as_int(value.get("steps")),
                "update": _as_int(value.get("update")),
                "bytes": _as_int(value.get("bytes"))}
    if isinstance(value, str):
        return {"path": value, "steps": None, "update": None, "bytes": None}
    return None


def _resolve_stage(state, state_path: Path, out_override, target_override, missing):
    experiment_dir = state_path.parent
    stage_raw = dict(state.get("stage") or {}) if isinstance(state, Mapping) else {}
    discovered = _discover_stages(experiment_dir)
    stage_out = None
    source = None
    auto_detected = False

    cli_path = None
    if out_override:
        cli_path = _safe_path(out_override, "--out")
    state_path_value = stage_raw.get("out_dir")
    state_stage_dir = None
    if cli_path is not None:
        stage_out, source = cli_path, "cli"
    elif state_path_value:
        try:
            state_stage_dir = _safe_path(state_path_value, "state.stage.out_dir")
        except SupervisorError as exc:
            missing.append("state.stage.out_dir_invalid")
            state_stage_dir = None
        if state_stage_dir is not None:
            stage_out, source = state_stage_dir, "state"
    if stage_out is None and discovered:
        stage_out, source, auto_detected = discovered[-1]["out_dir"], "auto_detect", True
        missing.append("state.stage.out_dir")

    target = target_override
    env_json_stage = None
    if stage_out is not None and (stage_out / "environment.json").is_file():
        data, error = _read_json(stage_out / "environment.json")
        if not error and isinstance(data, dict):
            env_json_stage = data
    if target is None:
        if source == "cli" and isinstance(env_json_stage, dict):
            target = _as_int(env_json_stage.get("target_steps"))
        if target is None:
            target = _as_int(stage_raw.get("target_steps"))
        if target is None and isinstance(env_json_stage, dict):
            target = _as_int(env_json_stage.get("target_steps"))

    # When --out points at a different directory than the state-tracked stage, the state
    # stage's pid/status/start_steps/target belong to a different run: do not mix them in.
    isolated = (source == "cli" and
                (state_stage_dir is None or stage_out != state_stage_dir))
    effective_raw = {} if isolated else stage_raw

    start_steps = _as_int(effective_raw.get("start_steps"))
    if start_steps is None:
        resume_value = effective_raw.get("resume_from")
        if isinstance(resume_value, str):
            match = CHECKPOINT_NAME_RE.match(Path(resume_value).name)
            if match is not None:
                start_steps = int(match.group(1))
    if start_steps is None and isinstance(env_json_stage, dict):
        resume_value = env_json_stage.get("resumed_from")
        if isinstance(resume_value, str):
            match = CHECKPOINT_NAME_RE.match(Path(resume_value).name)
            if match is not None:
                start_steps = int(match.group(1))
    if start_steps is None and stage_out is not None:
        metrics_path = stage_out / "metrics.jsonl"
        batch_hint = _as_int(dict(state).get("batch_per_update")) \
            if isinstance(state, Mapping) else None
        if metrics_path.is_file():
            try:
                with metrics_path.open("r", encoding="utf-8", errors="replace") as handle:
                    first_line = handle.readline()
                row = json.loads(first_line)
                if isinstance(row, dict):
                    first_step = _as_int(row.get("progress/environment_steps"))
                    if first_step is not None:
                        # a resumed run's first metrics row already contains resumed steps
                        start_steps = max(0, first_step - (batch_hint or 2000))
            except (OSError, json.JSONDecodeError):
                start_steps = None
    if start_steps is None:
        start_steps = 0

    started_at = effective_raw.get("started_at")
    started_unix = _as_unix(started_at)
    if started_unix is None and stage_out is not None and stage_out.is_dir():
        try:
            started_unix = stage_out.stat().st_mtime
        except OSError:
            started_unix = None
    pid = _as_int(effective_raw.get("pid"))
    configured_status = effective_raw.get("status") or \
        (state.get("status") if isinstance(state, Mapping) and not isolated else None)
    if isolated:
        configured_status = "cli-override"

    newer = None
    if stage_out is not None:
        current = next((item for item in discovered if item["out_dir"] == stage_out), None)
        newer_candidates = [item for item in discovered if item["out_dir"] != stage_out]
        if newer_candidates and current is not None and newer_candidates[-1]["mtime"] > current["mtime"]:
            newer = newer_candidates[-1]
        elif newer_candidates and current is None and discovered:
            newer = discovered[-1]

    stage_name = effective_raw.get("name")
    if source == "cli" and stage_out is not None:
        stage_name = stage_out.name
    resume_from = effective_raw.get("resume_from")
    if resume_from is None and isinstance(env_json_stage, dict):
        resume_from = env_json_stage.get("resumed_from")
    stage = {
        "name": stage_name or (stage_out.name if stage_out is not None else None),
        "out_dir": str(stage_out) if stage_out is not None else None,
        "out_dir_exists": bool(stage_out is not None and stage_out.is_dir()),
        "target_steps": target,
        "start_steps": start_steps,
        "configured_status": configured_status,
        "pid": pid,
        "started_at": started_at,
        "started_unix": started_unix,
        "stage_age_seconds": (max(0.0, time.time() - started_unix)
                              if started_unix is not None else None),
        "resume_from": resume_from,
        "source": source,
        "auto_detected": auto_detected,
        "newer_stage_detected": ({"out_dir": str(newer["out_dir"]),
                                  "latest_steps": newer["latest_steps"],
                                  "has_result": newer["has_result"]}
                                 if newer is not None else None),
    }
    return stage, stage_out


def _collect_alerts(snapshot, state, stage, process, metrics, checkpoints, result,
                    system, shape, prev_snapshots):
    alerts = []

    def add(code, severity, message, **extra):
        alert = {"code": code, "severity": severity, "message": message}
        alert.update(extra)
        alerts.append(alert)

    running = bool(process["running"])
    stage_out = Path(stage["out_dir"]) if stage["out_dir"] else None
    configured_running = (stage.get("configured_status") == "running")
    stage_age = stage.get("stage_age_seconds")
    target = stage.get("target_steps")
    result_steps = result.get("environment_steps")
    latest_metrics = metrics.get("last") or {}
    stage_steps = _as_int(latest_metrics.get("progress/environment_steps"))
    if stage_steps is None and checkpoints.get("latest"):
        stage_steps = checkpoints["latest"]["steps"]
    if stage_steps is None:
        stage_steps = result_steps

    # --- process / lifecycle -------------------------------------------------
    if process["pid_mismatch"]:
        add("stale_state_pid", SEV_WARNING,
            f"state PID {process['state_pid']} does not match the live training "
            f"process(es) {process['live_matching_pids']} for this run directory",
            state_pid=process["state_pid"], live_pids=process["live_matching_pids"])
    if len(process["live_matching_pids"]) > 2:
        add("multiple_train_processes", SEV_WARNING,
            f"more than two live processes match scripts/train.py for this run: "
            f"{process['live_matching_pids']} (launcher+parent pair is expected = 2)",
            live_pids=process["live_matching_pids"])

    if not running:
        if result.get("exists"):
            if configured_running:
                add("state_stale", SEV_WARNING,
                    "training process is no longer running but result.json exists; "
                    "training_state.json still says running (Lead should advance the stage)")
            if target is not None and result_steps is not None and result_steps < target:
                add("stopped_before_target", SEV_WARNING,
                    f"stage stopped at {result_steps} steps before target {target} "
                    f"(stop_reason={result.get('stop_reason')})",
                    environment_steps=result_steps, target_steps=target)
        else:
            enough_age = stage_age is None or stage_age > STAGE_START_GRACE_SECONDS
            has_progress_evidence = bool(metrics.get("exists")) or configured_running \
                or stage.get("auto_detected")
            if enough_age and has_progress_evidence:
                add("process_exited_without_result", SEV_CRITICAL,
                    "no live scripts/train.py process and no result.json: the stage process "
                    "exited/crashed before writing result.json",
                    state_pid=process["state_pid"], stage_out=stage["out_dir"])

    # --- metrics -------------------------------------------------------------
    nonfinite = metrics.get("nonfinite_keys") or []
    if nonfinite:
        add("metrics_nonfinite", SEV_CRITICAL,
            f"non-finite values in the latest metrics row: {nonfinite}",
            keys=nonfinite)
    delta = _as_float(latest_metrics.get("validation/parameter_max_delta"))
    if delta is not None and delta <= 0:
        add("parameter_delta_nonpositive", SEV_CRITICAL,
            "validation/parameter_max_delta <= 0: PPO did not change model parameters",
            parameter_max_delta=delta)

    last_age = metrics.get("last_update_age_seconds")
    if running and last_age is not None and last_age > STALE_UPDATE_SECONDS:
        add("stale_metrics", SEV_CRITICAL,
            f"no new metrics for {_fmt_duration(last_age)} (threshold "
            f"{int(STALE_UPDATE_SECONDS)}s)",
            last_update_age_seconds=last_age)
    if running and not metrics.get("exists") and stage_age is not None \
            and stage_age > NO_NEW_STEPS_SECONDS:
        add("metrics_missing", SEV_WARNING,
            f"stage has been running for {_fmt_duration(stage_age)} but metrics.jsonl "
            "has not appeared yet", stage_age_seconds=stage_age)

    # consecutive snapshots with identical step count
    previous = prev_snapshots[-1] if prev_snapshots else None
    if previous is not None and running:
        previous_steps = _as_int(((previous.get("progress") or {})
                                  .get("stage_env_steps")))
        previous_unix = _as_float(previous.get("generated_unix"))
        if previous_steps is not None and previous_steps == stage_steps \
                and previous_unix is not None \
                and time.time() - previous_unix >= NO_NEW_STEPS_SECONDS:
            add("no_new_steps", SEV_WARNING,
                f"environment steps unchanged ({stage_steps}) across supervisor snapshots "
                f"for {_fmt_duration(time.time() - previous_unix)}",
                env_steps=stage_steps)

    # --- checkpoints ---------------------------------------------------------
    latest_cp = checkpoints.get("latest")
    if latest_cp is not None and latest_cp["size_bytes"] < CHECKPOINT_MIN_BYTES:
        add("checkpoint_too_small", SEV_CRITICAL,
            f"latest checkpoint {latest_cp['name']} is only "
            f"{_fmt_bytes(latest_cp['size_bytes'])} (< 1 MiB): corrupted/truncated save",
            checkpoint=latest_cp["path"], size_bytes=latest_cp["size_bytes"])

    state_latest = _normalise_checkpoint(dict(state).get("latest_checkpoint")
                                         if isinstance(state, Mapping) else None)
    if state_latest and state_latest.get("path"):
        try:
            state_cp_path = _safe_path(state_latest["path"], "state.latest_checkpoint.path")
        except SupervisorError:
            state_cp_path = None
        if state_cp_path is not None and not state_cp_path.is_file():
            add("state_checkpoint_missing", SEV_CRITICAL,
                f"state.latest_checkpoint points to a missing file: {state_cp_path}",
                checkpoint=str(state_cp_path))
        elif state_cp_path is not None:
            try:
                size = state_cp_path.stat().st_size
            except OSError:
                size = None
            if size is not None and size < CHECKPOINT_MIN_BYTES:
                add("checkpoint_too_small", SEV_CRITICAL,
                    f"state.latest_checkpoint is only {_fmt_bytes(size)} (< 1 MiB): "
                    "corrupted/truncated save", checkpoint=str(state_cp_path),
                    size_bytes=size)

    batch, interval, _shape_source = shape
    expected_step = None
    start_steps = int(stage.get("start_steps") or 0)
    if batch and interval and stage_steps:
        unit = batch * interval
        if unit > 0:
            # Checkpoints are written at global update multiples of `interval`, i.e. at
            # environment-step multiples of `unit`. A resumed stage does not re-write the
            # checkpoint it started from, so the first expectation is the first multiple
            # strictly greater than start_steps.
            expected_step = (stage_steps // unit) * unit
            if expected_step <= start_steps:
                expected_step = start_steps + unit
            if expected_step > stage_steps:
                expected_step = None
    if expected_step and running:
        actual_latest = latest_cp["steps"] if latest_cp else 0
        if actual_latest < expected_step:
            add("checkpoint_expected_missing", SEV_CRITICAL,
                f"metrics report {stage_steps} environment steps, so a checkpoint at "
                f"{expected_step} steps is expected, but the newest checkpoint is "
                f"{actual_latest if actual_latest else 'missing'}",
                expected_steps=expected_step, latest_steps=actual_latest or None)

    # --- resources -----------------------------------------------------------
    disk = system.get("disk")
    if disk and disk.get("free_bytes") is not None \
            and disk["free_bytes"] < DISK_MIN_FREE_BYTES:
        add("low_disk", SEV_CRITICAL,
            f"project disk free space {_fmt_bytes(disk['free_bytes'])} < 10 GiB",
            free_bytes=disk["free_bytes"])
    ram = system.get("ram")
    if ram and ram.get("available_bytes") is not None \
            and ram["available_bytes"] < RAM_MIN_AVAILABLE_BYTES:
        add("low_ram", SEV_CRITICAL,
            f"available RAM {_fmt_bytes(ram['available_bytes'])} < 1.5 GiB",
            available_bytes=ram["available_bytes"])
    gpu = system.get("gpu")
    if gpu:
        temperature = gpu.get("temperature_c")
        if temperature is not None and temperature > GPU_TEMP_MAX_C:
            add("gpu_hot", SEV_CRITICAL,
                f"GPU temperature {temperature:.0f} C > {GPU_TEMP_MAX_C:.0f} C",
                temperature_c=temperature)
        free = gpu.get("memory_free_mib")
        if free is not None and free < GPU_MEM_MIN_FREE_MIB:
            add("low_gpu_memory", SEV_WARNING,
                f"GPU memory free {free:.0f} MiB < 1 GiB", memory_free_mib=free)
    if system.get("gpu_error"):
        add("gpu_query_failed", SEV_WARNING,
            f"nvidia-smi query failed: {system['gpu_error']}",
            gpu_error=system["gpu_error"])

    # --- state / bookkeeping -------------------------------------------------
    if stage.get("newer_stage_detected"):
        newer = stage["newer_stage_detected"]
        add("newer_stage_detected", SEV_INFO,
            f"a run directory newer than the tracked stage exists: {newer['out_dir']} "
            f"(latest_steps={newer['latest_steps']}, result={newer['has_result']}); "
            "training_state.json may need an update",
            newer_stage=newer)

    order = {SEV_CRITICAL: 0, SEV_WARNING: 1, SEV_INFO: 2}
    alerts.sort(key=lambda alert: (order.get(alert["severity"], 3), alert["code"]))
    return alerts


def _build_snapshot(state_path, out_override=None, monitor_dir=None,
                    cpu_interval: float = 1.0, target_steps_override=None,
                    include_gpu: bool = True):
    started = time.perf_counter()
    state_path = _safe_path(state_path, "--state")
    monitor_dir = _safe_path(monitor_dir or DEFAULT_MONITOR_DIR, "--monitor-dir")
    missing = ["training_state"] if not state_path.is_file() else []

    state, state_info = _load_state(state_path)
    if not state_info["exists"]:
        missing.append("training_state")
    elif state_info.get("error"):
        missing.append("training_state_error")
    if isinstance(state, Mapping) and "stage" not in state:
        missing.append("state.stage")

    stage, stage_out = _resolve_stage(state, state_path, out_override,
                                      target_steps_override, missing)
    experiment_dir = state_path.parent

    metrics = _metrics_info(stage_out)
    checkpoints = _checkpoint_info(stage_out)
    result = _result_info(stage_out)
    process = _inspect_process(stage["pid"], stage_out)
    system = _query_system(cpu_interval, include_gpu=include_gpu)
    shape = _training_shape(state, stage_out)

    # --- progress ------------------------------------------------------------
    latest_metrics = metrics.get("last") or {}
    latest_cp = checkpoints.get("latest")
    stage_steps = _as_int(latest_metrics.get("progress/environment_steps"))
    step_basis = "metrics.progress/environment_steps"
    if stage_steps is None and latest_cp is not None:
        stage_steps = latest_cp["steps"]
        step_basis = "latest checkpoint"
    if stage_steps is None and result.get("environment_steps") is not None:
        stage_steps = result["environment_steps"]
        step_basis = "result.json"
    update = _as_int(latest_metrics.get("progress/update"))
    if update is None and result.get("updates_completed") is not None:
        update = result["updates_completed"]

    total_candidates = [
        _as_int(dict(state).get("total_training_env_steps")) if isinstance(state, Mapping) else None,
        stage_steps,
        latest_cp["steps"] if latest_cp else None,
        result.get("environment_steps"),
    ]
    total_steps = max([value for value in total_candidates if value is not None], default=None)

    throughput = None
    throughput_basis = None
    rows = metrics.get("rows_seen") or 0
    if metrics.get("exists"):
        try:
            tail_lines = _tail_lines(Path(metrics["path"]), 5)
            values = []
            for line in tail_lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = _as_float(row.get("time/steps_per_second"))
                if value is not None and value > 0:
                    values.append(value)
            if values:
                throughput = statistics.median(values)
                throughput_basis = f"median of {len(values)} recent time/steps_per_second"
        except OSError:
            throughput = None
        if throughput is None and rows >= 2:
            try:
                tail_lines = _tail_lines(Path(metrics["path"]), 6)
                parsed = []
                for line in tail_lines:
                    try:
                        parsed.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if len(parsed) >= 2:
                    steps_delta = (_as_int(parsed[-1].get("progress/environment_steps")) or 0) - \
                                  (_as_int(parsed[0].get("progress/environment_steps")) or 0)
                    time_delta = (_as_float(parsed[-1].get("time/unix")) or 0) - \
                                 (_as_float(parsed[0].get("time/unix")) or 0)
                    if steps_delta > 0 and time_delta > 0:
                        throughput = steps_delta / time_delta
                        throughput_basis = "steps elapsed over tail metrics window"
            except OSError:
                throughput = None

    target = stage.get("target_steps")
    eta_stage = None
    eta_project = None
    if throughput and target is not None and stage_steps is not None:
        eta_stage = max(0.0, (target - stage_steps) / throughput)
    if throughput and total_steps is not None:
        remaining = PROJECT_STEP_TARGET - total_steps
        eta_project = max(0.0, remaining / throughput) if remaining > 0 else 0.0

    budget = dict(dict(state).get("budget") or {}) if isinstance(state, Mapping) else {}
    max_steps_budget = _as_int(budget.get("max_training_env_steps")) or \
        _as_int(budget.get("max_steps"))
    max_hours_budget = _as_float(budget.get("max_wall_hours")) or \
        _as_float(budget.get("max_hours"))
    deadline_at = budget.get("deadline_at")
    deadline_unix = _as_unix(deadline_at)
    eta_budget = None
    if throughput and max_steps_budget and total_steps is not None:
        remaining = max_steps_budget - total_steps
        eta_budget = max(0.0, remaining / throughput) if remaining > 0 else 0.0
    wall_remaining = max(0.0, deadline_unix - time.time()) if deadline_unix else None
    if wall_remaining is None and max_hours_budget is not None:
        cumulative = _as_float(dict(state).get("cumulative_training_seconds"))
        if cumulative is not None:
            wall_remaining = max(0.0, max_hours_budget * 3600 - cumulative)

    progress = {
        "stage_env_steps": stage_steps,
        "step_basis": step_basis if stage_steps is not None else None,
        "update": update,
        "total_training_env_steps": total_steps,
        "target_steps": target,
        "target_remaining_steps": (max(0, target - stage_steps)
                                   if (target is not None and stage_steps is not None) else None),
        "target_reached": (True if (target is not None and stage_steps is not None
                                    and stage_steps >= target) else
                           (False if target is not None and stage_steps is not None else None)),
        "last_update_unix": metrics.get("last_update_unix"),
        "last_update_age_seconds": metrics.get("last_update_age_seconds"),
        "throughput_steps_per_second": throughput,
        "throughput_basis": throughput_basis,
        "eta_stage_seconds": eta_stage,
        "eta_project_1m_seconds": eta_project,
        "eta_budget_steps_seconds": eta_budget,
        "budget_wall_remaining_seconds": wall_remaining,
        "project_step_target": PROJECT_STEP_TARGET,
        "batch_per_update": shape[0],
        "checkpoint_interval_updates": shape[1],
        "shape_source": shape[2],
    }

    state_section = {
        "path": str(state_path),
        "exists": state_info["exists"],
        "error": state_info.get("error"),
        "fields": state_info.get("fields", []),
        "experiment": dict(state).get("experiment") if isinstance(state, Mapping) else None,
        "config": dict(state).get("config") if isinstance(state, Mapping) else None,
        "seed": dict(state).get("seed") if isinstance(state, Mapping) else None,
        "device": dict(state).get("device") if isinstance(state, Mapping) else None,
        "status": dict(state).get("status") if isinstance(state, Mapping) else None,
        "stage_status": stage.get("configured_status"),
        "latest_checkpoint": _normalise_checkpoint(
            dict(state).get("latest_checkpoint") if isinstance(state, Mapping) else None),
        "latest_valid_checkpoint": _normalise_checkpoint(
            dict(state).get("latest_valid_checkpoint") if isinstance(state, Mapping) else None),
        "total_training_env_steps": _as_int(
            dict(state).get("total_training_env_steps") if isinstance(state, Mapping) else None),
        "cumulative_training_seconds": _as_float(
            dict(state).get("cumulative_training_seconds") if isinstance(state, Mapping) else None),
        "budget": budget or None,
        "stages_completed_count": len(dict(state).get("stages_completed") or [])
        if isinstance(state, Mapping) else 0,
        "stop_file": None,
        "stop_file_exists": False,
    }
    stop_file_raw = dict(state).get("stop_file") if isinstance(state, Mapping) else None
    if stop_file_raw:
        try:
            stop_path = _safe_path(stop_file_raw, "state.stop_file")
            state_section["stop_file"] = str(stop_path)
            state_section["stop_file_exists"] = stop_path.exists()
        except SupervisorError:
            state_section["stop_file"] = str(stop_file_raw)
            state_section["stop_file_invalid"] = True

    checkpoints_section = {
        "count": checkpoints["count"],
        "total_size_bytes": checkpoints["total_size_bytes"],
        "latest": ({key: latest_cp[key] for key in
                    ("name", "steps", "size_bytes", "size_mb", "mtime", "path")}
                   if latest_cp else None),
        "recent": [{key: entry[key] for key in ("name", "steps", "size_bytes", "size_mb")}
                   for entry in checkpoints["entries"][-5:]],
        "error": checkpoints.get("error"),
    }

    metrics_section = {
        "path": metrics.get("path"),
        "exists": metrics.get("exists"),
        "size_bytes": metrics.get("size_bytes"),
        "mtime": metrics.get("mtime"),
        "rows_seen": metrics.get("rows_seen"),
        "last_update_unix": metrics.get("last_update_unix"),
        "last_update_age_seconds": metrics.get("last_update_age_seconds"),
        "nonfinite_keys": metrics.get("nonfinite_keys", []),
        "error": metrics.get("error"),
        "latest": _curate_metrics(latest_metrics),
    }

    snapshot = {
        "schema": STATUS_SCHEMA,
        "kind": "status",
        "generated_at": _iso_now(),
        "generated_unix": time.time(),
        "host": {"pid": os.getpid(), "python": sys.version.split()[0]},
        "paths": {
            "root": str(ROOT),
            "state": str(state_path),
            "experiment_dir": str(experiment_dir),
            "stage_out": str(stage_out) if stage_out is not None else None,
            "monitor_dir": str(monitor_dir),
            "status_log": str(monitor_dir / "status.jsonl"),
        },
        "missing": sorted(set(missing)),
        "state": state_section,
        "stage": stage,
        "process": process,
        "progress": progress,
        "metrics": metrics_section,
        "checkpoints": checkpoints_section,
        "result": result,
        "system": system,
        "alerts": [],
    }

    monitor_dir.mkdir(parents=True, exist_ok=True)
    previous = _read_recent_snapshots(monitor_dir, stage_out, limit=3)
    snapshot["alerts"] = _collect_alerts(snapshot, state, stage, process, metrics,
                                         checkpoints, result, system, shape, previous)
    snapshot["has_critical"] = any(alert["severity"] == SEV_CRITICAL
                                   for alert in snapshot["alerts"])
    snapshot["timing"] = {"collect_seconds": round(time.perf_counter() - started, 3)}
    return snapshot


# ---------------------------------------------------------------------------
# snapshot log (append-only, best effort)
# ---------------------------------------------------------------------------
def _read_recent_snapshots(monitor_dir: Path, stage_out: Path | None, limit: int = 5):
    path = monitor_dir / "status.jsonl"
    if not path.is_file():
        return []
    try:
        lines = _tail_lines(path, limit * 4)
    except OSError:
        return []
    wanted = str(stage_out) if stage_out is not None else None
    snapshots = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        record_stage = (record.get("paths") or {}).get("stage_out")
        if wanted is not None and record_stage != wanted:
            continue
        if "generated_unix" not in record:
            continue
        snapshots.append(record)
    return snapshots[-limit:]


def _append_snapshot(monitor_dir: Path, snapshot, kind: str = "status", extra=None):
    record = dict(snapshot)
    record["kind"] = kind
    if extra:
        record.update(extra)
    path = monitor_dir / "status.jsonl"
    line = json.dumps(record, ensure_ascii=True, default=str)
    last_error = None
    for attempt in range(3):
        try:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            return path, None
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.2)
    return path, last_error


# ---------------------------------------------------------------------------
# human rendering
# ---------------------------------------------------------------------------
def _render_human(snapshot, status_log_path=None) -> str:
    stage = snapshot["stage"]
    process = snapshot["process"]
    progress = snapshot["progress"]
    metrics = snapshot["metrics"]
    checkpoints = snapshot["checkpoints"]
    result = snapshot["result"]
    system = snapshot["system"]
    paths = snapshot["paths"]
    lines = []
    lines.append("=== Pokemon Red PPO supervisor status ===")
    lines.append(f"generated : {snapshot['generated_at']}  (collect "
                 f"{snapshot.get('timing', {}).get('collect_seconds', '?')}s)")
    stage_bits = [stage.get("name") or "<unknown>"]
    if stage.get("configured_status"):
        stage_bits.append(f"state={stage['configured_status']}")
    if stage.get("target_steps") is not None:
        stage_bits.append(f"target={stage['target_steps']:,}")
    lines.append(f"stage     : {' | '.join(stage_bits)}")
    lines.append(f"out_dir   : {stage.get('out_dir') or '<none>'}"
                 + ("" if stage.get("out_dir_exists", True) else "  (not created yet)"))
    lines.append(f"process   : running={process['running']} "
                 f"actual_pid={process['live_matching_pids'] or 'none'} "
                 f"state_pid={process.get('state_pid')} "
                 f"(alive={process.get('state_pid_alive')})")
    if metrics.get("last_update_age_seconds") is not None:
        age = _fmt_duration(metrics["last_update_age_seconds"])
    else:
        age = "n/a"
    steps_text = f"{progress['stage_env_steps']:,}" if progress.get("stage_env_steps") is not None else "n/a"
    if progress.get("target_steps") is not None and progress.get("stage_env_steps") is not None:
        percent = 100.0 * progress["stage_env_steps"] / progress["target_steps"] \
            if progress["target_steps"] else 0.0
        steps_text += f" / {progress['target_steps']:,} ({percent:.1f}%)"
    throughput = progress.get("throughput_steps_per_second")
    throughput_text = f"{throughput:.1f} steps/s" if throughput else "n/a"
    lines.append(f"progress  : steps={steps_text} update={progress.get('update')} "
                 f"last_update={age} ago throughput={throughput_text}")
    eta_stage = _fmt_duration(progress.get("eta_stage_seconds"))
    eta_project = _fmt_duration(progress.get("eta_project_1m_seconds"))
    lines.append(f"totals    : cumulative={progress.get('total_training_env_steps')} "
                 f"| eta_stage={eta_stage} | eta_1M={eta_project}")
    latest = checkpoints.get("latest")
    if latest:
        lines.append(f"checkpoint: {checkpoints['count']} file(s) | latest {latest['name']} "
                     f"steps={latest['steps']:,} size={latest['size_mb']} MiB")
    else:
        lines.append("checkpoint: none yet")
    if result.get("exists"):
        lines.append(f"result    : present | environment_steps={result.get('environment_steps')} "
                     f"stop_reason={result.get('stop_reason')}")
    else:
        lines.append("result    : not yet")
    cpu = _pct(system.get("cpu_percent"))
    ram = system.get("ram") or {}
    swap = system.get("swap") or {}
    disk = system.get("disk") or {}
    ram_text = (f"{_fmt_bytes(ram.get('available_bytes'))} free / "
                f"{_fmt_bytes(ram.get('total_bytes'))} total") if ram else "n/a"
    swap_text = (f"{_fmt_bytes(swap.get('used_bytes'))}/{_fmt_bytes(swap.get('total_bytes'))} "
                 f"used ({swap.get('percent')}%)") if swap and swap.get("total_bytes") else "n/a"
    disk_text = f"{_fmt_bytes(disk.get('free_bytes'))} free" if disk else "n/a"
    gpu = system.get("gpu") or {}
    if gpu:
        gpu_text = (f"{gpu.get('utilization_percent')}% | "
                    f"{gpu.get('memory_used_mib')}/{gpu.get('memory_total_mib')} MiB | "
                    f"{gpu.get('temperature_c')} C | {gpu.get('power_w')} W")
    else:
        gpu_text = f"unavailable ({system.get('gpu_error')})"
    lines.append(f"resources : cpu={cpu} ram={ram_text} swap={swap_text} disk={disk_text}")
    lines.append(f"gpu       : {gpu_text}")
    latest_metrics = metrics.get("latest") or {}
    if latest_metrics:
        keys = ("loss/loss", "other/entropy", "other/kl_divergence",
                "other/clip_fraction", "rollout/reward_mean", "decay/learning_rate")
        summary = ", ".join(
            f"{key.split('/')[-1]}={latest_metrics.get(key):.6g}" for key in keys
            if isinstance(latest_metrics.get(key), (int, float)))
        game = {key: latest_metrics[key] for key in latest_metrics
                if isinstance(key, str) and key.startswith("game/")}
        actions = {key: latest_metrics[key] for key in latest_metrics
                   if isinstance(key, str) and key.startswith("actions/share_")}
        lines.append(f"metrics   : {summary or 'n/a'}")
        if game:
            lines.append("game      : " + ", ".join(
                f"{key.split('/', 1)[1]}={value:.4g}" for key, value in game.items()))
        if actions:
            lines.append("actions   : " + ", ".join(
                f"{key.split('_')[-1]}={value:.4f}" for key, value in actions.items()))
    alerts = snapshot.get("alerts") or []
    if alerts:
        lines.append("alerts    :")
        for alert in alerts:
            lines.append(f"  [{alert['severity']}] {alert['code']}: {alert['message']}")
    else:
        lines.append("alerts    : none")
    if snapshot.get("missing"):
        lines.append(f"missing   : {', '.join(snapshot['missing'])}")
    else:
        lines.append("missing   : none")
    if status_log_path:
        lines.append(f"status_log: {status_log_path}")
    if snapshot.get("state", {}).get("stop_file_invalid"):
        lines.append("stop_file : invalid path in state")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------
def cmd_status(args) -> int:
    snapshot = _build_snapshot(args.state, out_override=args.out,
                               monitor_dir=args.monitor_dir,
                               cpu_interval=1.0)
    log_path, log_error = _append_snapshot(
        _safe_path(args.monitor_dir or DEFAULT_MONITOR_DIR, "--monitor-dir"), snapshot, "status")
    snapshot["status_log_path"] = str(log_path)
    snapshot["status_log_error"] = log_error
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=True, indent=2, default=str))
    else:
        print(_render_human(snapshot, log_path))
        if log_error:
            print(f"warning: could not append status snapshot: {log_error}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# watch command
# ---------------------------------------------------------------------------
def _watch_line(round_number: int, elapsed: float, snapshot) -> str:
    stage = snapshot["stage"]
    progress = snapshot["progress"]
    process = snapshot["process"]
    steps = progress.get("stage_env_steps")
    target = progress.get("target_steps")
    steps_text = f"{steps:,}" if steps is not None else "n/a"
    if target:
        steps_text += f"/{target:,}"
    alerts = snapshot.get("alerts") or []
    critical = [alert["code"] for alert in alerts if alert["severity"] == SEV_CRITICAL]
    warning = [alert["code"] for alert in alerts if alert["severity"] == SEV_WARNING]
    alert_text = "none"
    if critical:
        alert_text = "critical=" + ",".join(critical)
    elif warning:
        alert_text = "warning=" + ",".join(warning)
    age = progress.get("last_update_age_seconds")
    age_text = f"{age:.1f}s" if age is not None else "n/a"
    throughput = progress.get("throughput_steps_per_second")
    throughput_text = f"{throughput:.0f}/s" if throughput else "n/a"
    return (f"[watch] round={round_number} t=+{_fmt_duration(elapsed)} "
            f"stage={stage.get('name')} running={process['running']} "
            f"steps={steps_text} last_update={age_text} "
            f"throughput={throughput_text} alerts={alert_text}")


def _watch_finish(outcome: str, exit_code: int, reason: str, rounds: int,
                  elapsed: float, snapshot, final_log_path=None):
    payload = {
        "schema": STATUS_SCHEMA,
        "kind": "watch_result",
        "generated_at": _iso_now(),
        "outcome": outcome,
        "exit_code": exit_code,
        "reason": reason,
        "rounds": rounds,
        "elapsed_seconds": round(elapsed, 3),
        "stage": {"name": snapshot["stage"].get("name"),
                  "out_dir": snapshot["stage"].get("out_dir"),
                  "target_steps": snapshot["stage"].get("target_steps")},
        "result": snapshot.get("result"),
        "progress": snapshot.get("progress"),
        "process": {"running": snapshot["process"].get("running"),
                    "live_matching_pids": snapshot["process"].get("live_matching_pids"),
                    "state_pid": snapshot["process"].get("state_pid")},
        "alerts": snapshot.get("alerts"),
        "has_critical": snapshot.get("has_critical"),
        "final_snapshot": snapshot,
    }
    if final_log_path:
        payload["status_log_path"] = str(final_log_path)
    print(json.dumps(payload, ensure_ascii=True, indent=2, default=str))
    return exit_code


def cmd_watch(args) -> int:
    state_path = _safe_path(args.state, "--state")
    monitor_dir = _safe_path(args.monitor_dir or DEFAULT_MONITOR_DIR, "--monitor-dir")
    out_override = args.out
    started = time.monotonic()
    rounds = 0
    critical_streak = {}
    snapshot = None
    snapshot_extra = None
    immediate_fatal_triggered = False
    while True:
        snapshot = _build_snapshot(state_path, out_override=out_override,
                                   monitor_dir=monitor_dir,
                                   cpu_interval=1.0,
                                   target_steps_override=args.target_steps)
        rounds += 1
        elapsed = time.monotonic() - started
        stage = snapshot["stage"]
        result = snapshot["result"]
        target = args.target_steps if args.target_steps is not None \
            else stage.get("target_steps")
        if not args.json:
            print(_watch_line(rounds, elapsed, snapshot), flush=True)

        alerts = snapshot.get("alerts") or []
        critical_codes = sorted({alert["code"] for alert in alerts
                                 if alert["severity"] == SEV_CRITICAL})
        for code in list(critical_streak):
            if code not in critical_codes:
                critical_streak.pop(code, None)
        for code in critical_codes:
            critical_streak[code] = critical_streak.get(code, 0) + 1
        immediate_fatal_triggered = any(code in IMMEDIATE_FATAL_CODES
                                        for code in critical_codes)
        sustained_fatal = any(code in SUSTAINED_CRITICAL_CODES and count >= 2
                              for code, count in critical_streak.items())

        # stall detection: no new metrics for a long time while the process is alive
        process_running = bool(snapshot["process"].get("running"))
        last_age = snapshot["progress"].get("last_update_age_seconds")
        stage_age = stage.get("stage_age_seconds")
        stall_fatal = (process_running and last_age is not None
                       and last_age > args.stall_seconds
                       and (stage_age is None or stage_age > args.stall_seconds))
        if process_running and last_age is None and stage_age is not None \
                and stage_age > args.stall_seconds:
            stall_fatal = True

        extra = {"watch": {"outcome": None, "rounds": rounds,
                           "elapsed_seconds": round(elapsed, 3)}}
        log_path, log_error = _append_snapshot(monitor_dir, snapshot, "watch", extra)

        # (a) terminal result reached target
        if args.target_steps is not None:
            target_steps = args.target_steps
        elif stage.get("target_steps") is not None:
            target_steps = stage["target_steps"]
        else:
            data, error = _read_json(Path(stage["out_dir"]) / "environment.json") \
                if stage.get("out_dir") else (None, "missing")
            target_steps = _as_int(data.get("target_steps")) if isinstance(data, dict) else None
        result_steps = result.get("environment_steps")
        if result.get("exists"):
            if target_steps is None or (result_steps is not None
                                        and result_steps >= target_steps):
                return _watch_finish("completed", 0,
                                     "result.json present and target reached", rounds,
                                     elapsed, snapshot, log_path)
            return _watch_finish("stopped_before_target", 4,
                                 f"result.json present with {result_steps} steps but "
                                 f"target is {target_steps} (stop_reason="
                                 f"{result.get('stop_reason')})", rounds, elapsed,
                                 snapshot, log_path)

        # (b) severe anomalies
        if immediate_fatal_triggered or sustained_fatal or stall_fatal:
            reasons = []
            if immediate_fatal_triggered:
                reasons.append("immediate critical: " + ",".join(
                    code for code in critical_codes if code in IMMEDIATE_FATAL_CODES))
            if sustained_fatal:
                reasons.append("sustained critical: " + ",".join(
                    code for code, count in critical_streak.items()
                    if count >= 2 and code in SUSTAINED_CRITICAL_CODES))
            if stall_fatal:
                reasons.append(f"no progress for {_fmt_duration(last_age)} "
                               f"(stall threshold {int(args.stall_seconds)}s)")
            return _watch_finish("anomaly", 1, "; ".join(reasons), rounds, elapsed,
                                 snapshot, log_path)

        # (c) timeout
        elapsed = time.monotonic() - started
        if elapsed >= args.timeout:
            return _watch_finish("timeout", 2,
                                 f"watch timeout after {_fmt_duration(elapsed)} "
                                 f"(no terminal result and no fatal anomaly)", rounds,
                                 elapsed, snapshot, log_path)

        sleep_for = min(args.interval, max(0.5, args.timeout - elapsed))
        time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# verify-checkpoint command
# ---------------------------------------------------------------------------
def _diff_dict(left, right, path=()):
    """Return structural differences between two plain dict/list values."""
    differences = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            sub_path = path + (str(key),)
            if key not in left:
                differences.append({"path": "/".join(sub_path), "only_in": "config",
                                    "config": _plain(right[key])})
            elif key not in right:
                differences.append({"path": "/".join(sub_path), "only_in": "checkpoint",
                                    "checkpoint": _plain(left[key])})
            else:
                differences.extend(_diff_dict(left[key], right[key], sub_path))
    elif left != right:
        differences.append({"path": "/".join(path) or "<root>",
                            "config": _plain(left), "checkpoint": _plain(right)})
    return differences


def _compare_configs(checkpoint_configs, reference_config):
    sections = ("environment", "sampler", "trainer", "model")
    report = {"reference": None, "sections": {}, "match_strict": True,
              "resume_compatible": True, "benign_differences": [],
              "substantive_differences": []}
    for section in sections:
        left = _plain(checkpoint_configs.get(section))
        right = _plain(reference_config.get(section))
        if left is None or right is None:
            differences = [{"path": section, "only_in": "checkpoint" if left is not None else "config",
                            "config": right if right is not None else None,
                            "checkpoint": left if left is not None else None}]
        else:
            differences = _diff_dict(left, right)
        benign = []
        substantive = []
        for difference in differences:
            parts = (section,) + tuple(difference["path"].split("/"))
            is_benign = (parts in RUNTIME_INJECTED_PATHS
                         and difference.get("only_in") == "checkpoint")
            (benign if is_benign else substantive).append(difference)
        report["sections"][section] = {
            "match": not differences,
            "match_ignoring_runtime_defaults": not substantive,
            "differences": differences,
            "benign_differences": benign,
            "substantive_differences": substantive,
        }
        report["benign_differences"].extend(benign)
        report["substantive_differences"].extend(substantive)
        if differences:
            report["match_strict"] = False
        if substantive:
            report["resume_compatible"] = False
    return report


def _rebuild_model_check(checkpoint):
    import torch
    from neroRL.nn.actor_critic import create_actor_critic_model
    detail = {"rebuilt": False, "strict_load": False, "parameter_count": None,
              "nonfinite_parameters": None, "error": None}
    try:
        model = create_actor_critic_model(checkpoint["configs"]["model"],
                                          checkpoint["observation_space"],
                                          None, checkpoint["action_space"],
                                          torch.device("cpu"))
        model.eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        detail["rebuilt"] = True
        detail["strict_load"] = True
        detail["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        nonfinite = sum(1 for parameter in model.parameters()
                        if not torch.isfinite(parameter).all())
        detail["nonfinite_parameters"] = nonfinite
    except Exception as exc:  # broad: report and continue
        detail["error"] = f"{type(exc).__name__}: {exc}"
    return detail


def cmd_verify(args) -> int:
    path = _safe_path(args.path, "--path")
    result = {
        "schema": VERIFY_SCHEMA,
        "generated_at": _iso_now(),
        "path": str(path),
        "exists": path.is_file(),
        "checks": [],
        "warnings": [],
        "errors": [],
        "config": None,
        "readonly": True,
    }

    def add_check(name, ok, detail=None):
        result["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    reference_path = None
    if args.config:
        reference_path = _safe_path(args.config, "--config")
    elif args.out:
        out_dir = _safe_path(args.out, "--out")
        reference_path = out_dir / "config.yaml"
    if reference_path is not None and not reference_path.is_file():
        result["errors"].append(f"reference config not found: {reference_path}")
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
        return 2
    if not path.is_file():
        result["errors"].append(f"checkpoint not found: {path}")
        add_check("file_exists", False)
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
        return 2

    try:
        stat_before = path.stat()
        result["size_bytes"] = stat_before.st_size
        result["mtime_before"] = stat_before.st_mtime
    except OSError as exc:
        result["errors"].append(f"cannot stat checkpoint: {exc}")
        add_check("file_exists", False)
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
        return 2
    add_check("file_exists", True, {"size_bytes": result["size_bytes"]})
    if result["size_bytes"] < CHECKPOINT_MIN_BYTES:
        result["warnings"].append(
            f"checkpoint size {_fmt_bytes(result['size_bytes'])} < 1 MiB")

    import torch
    torch.set_num_threads(1)
    from reproduction.runtime import read_checkpoint
    try:
        checkpoint = read_checkpoint(path)
        add_check("loadable", True)
    except Exception as exc:
        result["errors"].append(f"load failed: {type(exc).__name__}: {exc}")
        add_check("loadable", False)
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
        return 1

    if not isinstance(checkpoint, dict):
        result["errors"].append("checkpoint is not a dict")
        add_check("mapping", False)
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
        return 1
    add_check("mapping", True)

    phase1 = _as_int(checkpoint.get("phase1_format"))
    add_check("phase1_format", phase1 == 1, {"phase1_format": checkpoint.get("phase1_format")})
    next_update = _as_int(checkpoint.get("next_update"))
    add_check("next_update_positive", next_update is not None and next_update > 0,
              {"next_update": checkpoint.get("next_update")})
    environment_steps = _as_int(checkpoint.get("environment_steps"))
    add_check("environment_steps_positive",
              environment_steps is not None and environment_steps > 0,
              {"environment_steps": checkpoint.get("environment_steps")})
    seed = _as_int(checkpoint.get("seed"))
    result["metadata"] = {"phase1_format": phase1, "next_update": next_update,
                          "environment_steps": environment_steps, "seed": seed,
                          "agent_label": checkpoint.get("agent_label"),
                          "run_id": checkpoint.get("run_id"),
                          "has_optimizer": isinstance(checkpoint.get("optimizer"), Mapping)}

    model_state = checkpoint.get("model")
    model_ok = isinstance(model_state, Mapping) and len(model_state) > 0 and \
        all(torch.is_tensor(value) for value in model_state.values())
    add_check("model_state_nonempty", model_ok,
              {"entries": len(model_state) if isinstance(model_state, Mapping) else None})
    optimizer = checkpoint.get("optimizer")
    optimizer_ok = isinstance(optimizer, Mapping) and bool(optimizer.get("state")) \
        and bool(optimizer.get("param_groups"))
    add_check("optimizer_state_nonempty", optimizer_ok,
              {"state_entries": len(optimizer.get("state") or {}) if isinstance(optimizer, Mapping) else None,
               "param_groups": len(optimizer.get("param_groups") or []) if isinstance(optimizer, Mapping) else None})

    nonfinite_entries = []
    nonfinite_optimizer = 0
    if isinstance(model_state, Mapping):
        for key, value in model_state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                nonfinite_entries.append(key)
    if isinstance(optimizer, Mapping):
        for state in (optimizer.get("state") or {}).values():
            for value in (state or {}).values():
                if torch.is_tensor(value) and torch.is_floating_point(value) \
                        and not torch.isfinite(value).all():
                    nonfinite_optimizer += 1
    finite_ok = not nonfinite_entries and nonfinite_optimizer == 0
    add_check("parameters_finite", finite_ok,
              {"nonfinite_model_entries": nonfinite_entries[:10],
               "nonfinite_optimizer_tensors": nonfinite_optimizer})

    from gymnasium import spaces
    observation_space = checkpoint.get("observation_space")
    action_space = checkpoint.get("action_space")
    obs_ok = observation_space is not None and hasattr(observation_space, "spaces")
    action_ok = isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete, spaces.Box))
    add_check("observation_space", obs_ok,
              {"type": type(observation_space).__name__ if observation_space is not None else None,
               "keys": sorted(getattr(observation_space, "spaces", {}).keys())
               if hasattr(observation_space, "spaces") else None})
    add_check("action_space", action_ok,
              {"type": type(action_space).__name__ if action_space is not None else None,
               "repr": repr(action_space)})

    rebuild = _rebuild_model_check(checkpoint) if model_ok else \
        {"rebuilt": False, "strict_load": False, "error": "model state not usable"}
    result["model_rebuild"] = rebuild
    add_check("model_rebuild_strict_load", rebuild["rebuilt"] and rebuild["strict_load"]
              and rebuild.get("nonfinite_parameters") == 0,
              rebuild)

    if reference_path is not None:
        try:
            reference_config = _load_yaml_config(reference_path)
            checkpoint_configs = checkpoint.get("configs")
            if not isinstance(checkpoint_configs, Mapping):
                result["errors"].append("checkpoint has no configs mapping")
                report = None
            else:
                report = _compare_configs(checkpoint_configs, reference_config)
                report["reference"] = str(reference_path)
                result["config"] = report
                if report["benign_differences"]:
                    result["warnings"].append(
                        "config strict mismatch only in runtime-injected checkpoint defaults: "
                        + ", ".join(diff["path"] for diff in report["benign_differences"]))
                if report["substantive_differences"]:
                    result["errors"].append(
                        "checkpoint configs differ from reference config; resume would be rejected: "
                        + ", ".join(diff["path"] for diff in report["substantive_differences"]))
                add_check("config_resume_compatible", report["resume_compatible"],
                          {"reference": str(reference_path),
                           "match_strict": report["match_strict"],
                           "benign_differences": len(report["benign_differences"]),
                           "substantive_differences": len(report["substantive_differences"])})
        except (OSError, ValueError) as exc:
            result["errors"].append(f"cannot read reference config: {exc}")
            add_check("config_resume_compatible", False)
    try:
        stat_after = path.stat()
        result["readonly"] = (stat_after.st_mtime == result["mtime_before"]
                              and stat_after.st_size == result["size_bytes"])
    except OSError:
        result["readonly"] = False

    ok = all(check["ok"] for check in result["checks"]) and not result["errors"]
    result["ok"] = ok
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# prune-checkpoints command
# ---------------------------------------------------------------------------
def _parse_milestones(text):
    milestones = set()
    for token in (text or "").split(","):
        token = token.strip()
        if token:
            milestones.add(int(token))
    return sorted(milestones)


def cmd_prune(args) -> int:
    out_dir = _safe_path(args.out, "--out")
    if not out_dir.is_dir():
        raise SupervisorError(f"--out is not a directory: {out_dir}")
    if args.keep_last < 0:
        raise SupervisorError("--keep-last must be >= 0")
    milestones = _parse_milestones(args.milestones)
    archive_dir = out_dir / "_archive"
    try:
        archive_resolved = archive_dir.resolve()
        archive_resolved.relative_to(out_dir.resolve())
    except (OSError, ValueError) as exc:
        raise SupervisorError(f"archive directory escapes --out: {exc}") from exc

    checkpoints = _checkpoint_info(out_dir)
    entries = checkpoints["entries"]
    state_path = _safe_path(args.state, "--state")
    state, _state_info = _load_state(state_path)
    protected = set()
    for key in ("latest_checkpoint", "latest_valid_checkpoint"):
        item = _normalise_checkpoint(dict(state).get(key) if isinstance(state, Mapping) else None)
        if item and item.get("path"):
            try:
                protected.add(_safe_path(item["path"], f"state.{key}.path").resolve())
            except SupervisorError:
                continue

    keep_last_steps = {entry["steps"] for entry in entries[-args.keep_last:]} \
        if args.keep_last else set()
    milestone_matches = [entry for entry in entries if entry["steps"] in set(milestones)]
    milestone_steps = {entry["steps"] for entry in milestone_matches}
    keep_steps = keep_last_steps | milestone_steps
    now = time.time()

    kept, moved_plan, skipped = [], [], []
    for entry in entries:
        path = Path(entry["path"])
        try:
            path_resolved = path.resolve()
            path_resolved.relative_to(out_dir.resolve())
        except (OSError, ValueError):
            skipped.append({"name": entry["name"], "reason": "path_not_inside_out_dir"})
            continue
        if entry["steps"] in keep_steps:
            reason = "milestone" if entry["steps"] in milestone_steps else "keep_last"
            kept.append({"name": entry["name"], "steps": entry["steps"], "reason": reason})
            continue
        if path_resolved in protected:
            kept.append({"name": entry["name"], "steps": entry["steps"],
                         "reason": "protected_by_state"})
            continue
        if now - entry["mtime"] < 60:
            skipped.append({"name": entry["name"], "reason": "modified_recently"})
            continue
        destination = archive_resolved / entry["name"]
        if destination.exists():
            skipped.append({"name": entry["name"], "reason": "already_archived"})
            continue
        moved_plan.append({"name": entry["name"], "steps": entry["steps"],
                           "size_bytes": entry["size_bytes"],
                           "source": str(path_resolved), "destination": str(destination)})

    result = {
        "schema": "supervisor_prune/1",
        "generated_at": _iso_now(),
        "out_dir": str(out_dir),
        "archive_dir": str(archive_dir),
        "dry_run": bool(args.dry_run),
        "keep_last": args.keep_last,
        "milestones": milestones,
        "checkpoint_count": checkpoints["count"],
        "kept": kept,
        "planned_moves": moved_plan,
        "moved": [],
        "skipped": skipped,
        "milestones_matched": [entry["steps"] for entry in milestone_matches],
        "milestones_unmatched": [step for step in milestones if step not in milestone_steps],
        "errors": [],
    }

    if not moved_plan:
        print(f"nothing to move: {len(entries)} checkpoint(s) in {out_dir}; "
              f"kept={len(kept)}")
        if args.json:
            print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
        return 0

    if not args.dry_run:
        try:
            archive_resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            result["errors"].append(f"cannot create archive directory: {exc}")
            print(f"error: cannot create archive directory: {exc}", file=sys.stderr)
            print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
            return 1
        for item in moved_plan:
            source = Path(item["source"])
            destination = Path(item["destination"])
            try:
                if not source.is_file():
                    raise FileNotFoundError(f"source disappeared: {source}")
                if destination.resolve().parent != archive_resolved:
                    raise SupervisorError("destination escapes archive directory")
                shutil.move(str(source), str(destination))
                result["moved"].append(item)
            except (OSError, SupervisorError) as exc:
                result["errors"].append(f"{item['name']}: {type(exc).__name__}: {exc}")

    if args.dry_run:
        print(f"dry-run: would move {len(moved_plan)} of {len(entries)} checkpoint(s) "
              f"from {out_dir} to {archive_dir}")
        for item in moved_plan:
            print(f"  would-move {item['name']} ({_fmt_bytes(item['size_bytes'])}) "
                  f"steps={item['steps']:,}")
    else:
        print(f"moved {len(result['moved'])} of {len(entries)} checkpoint(s) "
              f"to {archive_dir}")
        for item in result["moved"]:
            print(f"  moved {item['name']} steps={item['steps']:,}")
    if kept:
        print(f"kept {len(kept)}: " + ", ".join(
            f"{item['name']}({item['reason']})" for item in kept[:12])
            + (" ..." if len(kept) > 12 else ""))
    if skipped:
        print(f"skipped {len(skipped)}: " + ", ".join(
            f"{item['name']}({item['reason']})" for item in skipped))
    if result["milestones_unmatched"]:
        print("milestone steps with no exact checkpoint: "
              + ", ".join(str(step) for step in result["milestones_unmatched"]))
    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0 if not result["errors"] else 1


# ---------------------------------------------------------------------------
# stop-request command
# ---------------------------------------------------------------------------
def cmd_stop_request(args) -> int:
    state_path = _safe_path(args.state, "--state")
    state, state_info = _load_state(state_path)
    experiment_dir = state_path.parent
    monitor_dir = _safe_path(args.monitor_dir or DEFAULT_MONITOR_DIR, "--monitor-dir")
    stop_file_raw = dict(state).get("stop_file") if isinstance(state, Mapping) else None
    if stop_file_raw:
        stop_path = _safe_path(stop_file_raw, "state.stop_file")
    else:
        stop_path = _safe_path(experiment_dir / "STOP", "default stop_file")
        print("warning: state has no stop_file field; falling back to "
              f"{stop_path}", file=sys.stderr)
    existed = stop_path.exists()
    try:
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        stop_path.touch()
    except OSError as exc:
        print(f"error: cannot touch stop file {stop_path}: {exc}", file=sys.stderr)
        return 2
    message = (f"stop file {'already existed' if existed else 'created'}: {stop_path}\n"
               "scripts/train.py checks this file after finishing the current update, saves a "
               "checkpoint (atomic write + read-back verification) and exits with "
               "result.json stop_reason=stop_file.\n"
               "The supervisor never kills the training process.")
    print(message)
    _append_snapshot(monitor_dir, {
        "schema": STATUS_SCHEMA,
        "kind": "stop_request",
        "generated_at": _iso_now(),
        "generated_unix": time.time(),
        "stop_file": str(stop_path),
        "already_existed": existed,
        "state_path": str(state_path),
    }, extra={"generated_unix": time.time()})
    return 0


# ---------------------------------------------------------------------------
# launch command (dry run only; never starts training)
# ---------------------------------------------------------------------------
def cmd_launch(args) -> int:
    python_exe = str(ROOT / ".venv" / "Scripts" / "python.exe")
    config_path = _safe_path(args.config, "--config")
    out_path = _safe_path(args.out, "--out")
    stop_file = _safe_path(args.stop_file or (DEFAULT_STATE.parent / "STOP"), "--stop-file")
    resume = _safe_path(args.resume, "--resume") if args.resume else None
    if args.total_steps is None:
        raise SupervisorError("--total-steps is required")
    log_dir = out_path.parent / "logs"
    command = [python_exe, "scripts\\train.py",
               "--config", str(config_path.relative_to(ROOT)),
               "--seed", str(args.seed),
               "--total-steps", str(args.total_steps),
               "--device", args.device,
               "--out", str(out_path.relative_to(ROOT)),
               "--stop-file", str(stop_file.relative_to(ROOT))]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    stdout_log = log_dir / f"{out_path.name}.stdout.log"
    stderr_log = log_dir / f"{out_path.name}.stderr.log"
    quoted = " ".join(f'"{part}"' if " " in part else part for part in command)
    powershell = (
        f"Start-Process -FilePath '{python_exe}' -ArgumentList @("
        + ",".join(f"'{part}'" for part in command[1:])
        + f") -WorkingDirectory '{ROOT}' -RedirectStandardOutput '{stdout_log}' "
          f"-RedirectStandardError '{stderr_log}' -PassThru"
    )
    payload = {
        "schema": "supervisor_launch/1",
        "dry_run": True,
        "command": command,
        "command_line": quoted,
        "powershell_start_process": powershell,
        "working_directory": str(ROOT),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "note": "The supervisor only prints this command. Lead launches training with "
                "Start-Process; the supervisor never spawns training processes.",
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common(parser):
    parser.add_argument("--state", default=str(DEFAULT_STATE),
                        help="training_state.json (default: %(default)s)")
    parser.add_argument("--monitor-dir", default=str(DEFAULT_MONITOR_DIR),
                        help="snapshot output directory (default: %(default)s)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="supervisor.py",
        description="Read-only supervision for the Pokemon Red PPO training runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="one-shot snapshot")
    _add_common(status)
    status.add_argument("--json", action="store_true", help="print the full snapshot as JSON")
    status.add_argument("--out", default=None, help="override the stage run directory")
    status.set_defaults(func=cmd_status)

    watch = subparsers.add_parser("watch", help="block until a terminal result/anomaly/timeout")
    _add_common(watch)
    watch.add_argument("--interval", type=float, default=60.0,
                       help="seconds between snapshots (default 60)")
    watch.add_argument("--timeout", type=float, default=7200.0,
                       help="hard timeout in seconds (default 7200)")
    watch.add_argument("--target-steps", type=int, default=None,
                       help="target environment steps (default: state stage.target_steps)")
    watch.add_argument("--stall-seconds", type=float, default=900.0,
                       help="exit 1 if a running stage makes no update for this long (default 900)")
    watch.add_argument("--json", action="store_true",
                       help="suppress periodic human lines; only the final JSON is printed")
    watch.add_argument("--out", default=None, help="override the stage run directory")
    watch.set_defaults(func=cmd_watch)

    verify = subparsers.add_parser("verify-checkpoint",
                                   help="read-only structural/model validation of a checkpoint")
    verify.add_argument("--path", required=True)
    verify.add_argument("--out", default=None,
                        help="compare checkpoint configs with OUT/config.yaml")
    verify.add_argument("--config", default=None,
                        help="compare checkpoint configs with an explicit YAML config")
    verify.set_defaults(func=cmd_verify)

    prune = subparsers.add_parser("prune-checkpoints",
                                  help="move old checkpoints into RUN/_archive (never deletes)")
    prune.add_argument("--out", required=True)
    prune.add_argument("--keep-last", type=int, default=10)
    prune.add_argument("--milestones", default="10000,100000,300000,1000000")
    prune.add_argument("--dry-run", action="store_true")
    prune.add_argument("--json", action="store_true")
    prune.add_argument("--state", default=str(DEFAULT_STATE))
    prune.set_defaults(func=cmd_prune)

    stop = subparsers.add_parser("stop-request",
                                 help="touch the state stop_file (graceful stop, no kill)")
    _add_common(stop)
    stop.set_defaults(func=cmd_stop_request)

    launch = subparsers.add_parser("launch",
                                   help="print the training command / Start-Process snippet "
                                        "(never launches)")
    launch.add_argument("--config", default=str(ROOT / "configs" / "train_seed0.yaml"))
    launch.add_argument("--total-steps", type=int, default=None)
    launch.add_argument("--out", required=True)
    launch.add_argument("--device", default="cuda")
    launch.add_argument("--seed", type=int, default=0)
    launch.add_argument("--resume", default=None)
    launch.add_argument("--stop-file", default=None)
    launch.add_argument("--dry-run", action="store_true")
    launch.set_defaults(func=cmd_launch)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SupervisorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # last-resort guard: the supervisor must not crash noisily
        print(f"error: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
