"""Branch / snapshot manifest layer for the multi-agent checkpoint system.

The :class:`BranchCheckpointManager` **extends** the existing
:class:`llm_agent.runtime.CheckpointStore` — it never replaces it.  The store
still owns the legal PyBoy save-state bytes plus the memory/budget/runtime
snapshots; this layer adds the multi-agent *branch manifest*:

* which branch/parent branch the state belongs to (rollbacks create new branches),
* a sha256 checksum of both the manifest and the emulator state bytes,
* the task graph snapshot, director state, per-agent contexts, navigation world
  model snapshot, shared-knowledge version, message-consumption progress and
  budget snapshot, so the whole system can be restored state-consistently,
* a monotonically increasing saved counter (persisted in
  ``<state_dir>/counter.json``) so checkpoint ids can never collide even when the
  emulator frame counter restarts from 0 after a state load,
* atomically written JSON (temp file + ``os.replace``, optional ``fsync``) and a
  :meth:`BranchCheckpointManager.verify` that detects tampering.

Learned experience (knowledge / lessons) deliberately lives **outside** the
manifest and the store snapshot: it is written to ``<state_dir>/learning.json``
and is never rolled back, so a rollback cannot erase what the system learned.

The module imports only the light :mod:`llm_agent.multi_agent.contracts` at
import time.  ``llm_agent.runtime`` (and therefore PyBoy) is imported lazily,
inside functions, only when no store was injected — unit tests inject a fake
store and never touch the emulator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_agent.multi_agent.contracts import (
    BranchRecord,
    BranchStatus,
    new_branch_id,
    now_ts,
)

__all__ = [
    "BranchCheckpointConfig",
    "BranchCheckpointManager",
    "CHECKPOINT_ID_PREFIX",
    "MANIFEST_REQUIRED_KEYS",
    "STATE_DIR_NAME",
]

#: Prefix of every public checkpoint id written by this layer.
CHECKPOINT_ID_PREFIX = "checkpoint_"

#: Sub-directory of the store directory that holds branch manifests/state.
STATE_DIR_NAME = "branches"

#: Default root used when no directory is configured (tests always inject one).
_DEFAULT_DIRECTORY = Path("artifacts") / "multi_agent" / "branch_checkpoints"

#: Schema version understood by :meth:`BranchCheckpointManager.verify`.
_SUPPORTED_MANIFEST_VERSIONS: frozenset[int] = frozenset({1})

#: Manifest fields that must be present for a manifest to be usable.
MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "manifest_version",
    "checkpoint_id",
    "branch_id",
    "parent_branch_id",
    "frame",
    "state_sha256",
    "components_progress",
    "tasks_snapshot",
    "director_state",
    "agent_contexts",
    "navigation_snapshot",
    "knowledge_version",
    "message_progress",
    "budget_snapshot",
    "created_at",
    "reason",
    "evidence",
)

_HEX_DIGITS = frozenset("0123456789abcdef")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _json_safe(value: Any) -> Any:
    """Deep JSON-safe copy (non-serialisable objects become their ``str()``)."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    """Canonical sha256 of a manifest payload (``manifest_sha256`` removed)."""
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Any, *, fsync: bool = True) -> None:
    """Atomically write ``payload`` as UTF-8 JSON (temp file + ``os.replace``).

    Mirrors :func:`llm_agent.runtime.write_json_atomic` and additionally
    supports the configurable ``fsync`` flag (the manager is durable by default).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_dumps(payload).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, tolerating missing/corrupt files (returns ``None``)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return dict(raw) if isinstance(raw, Mapping) else None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):  # NaN / inf
        return default
    return result


def _copy_branch(record: BranchRecord) -> BranchRecord:
    """Return an independent copy so callers cannot mutate manager state."""
    return copy.deepcopy(record)


def _branch_from_dict(raw: Mapping[str, Any]) -> BranchRecord:
    """Tolerant parser for a persisted branch record.

    ``BranchRecord.from_dict`` in the frozen contracts module calls a
    ``BranchStatus.coerce`` helper that does not exist, so this layer parses the
    (plain, stable) ``BranchRecord.to_dict`` payload itself and is therefore also
    tolerant to unknown statuses written by a newer version.
    """
    branch_id = str(raw.get("branch_id") or "").strip()
    if not branch_id.startswith("branch_"):
        raise ValueError(f"invalid persisted branch id: {raw.get('branch_id')!r}")
    status_raw = str(raw.get("status") or "active").strip().lower()
    try:
        status = BranchStatus(status_raw)
    except ValueError:
        status = BranchStatus.ACTIVE
    raw_tasks = raw.get("task_ids") or []
    task_ids = (
        [str(task) for task in raw_tasks if str(task).strip()]
        if isinstance(raw_tasks, (list, tuple))
        else []
    )
    evidence = raw.get("evidence")
    return BranchRecord(
        branch_id=branch_id,
        parent_branch_id=str(raw.get("parent_branch_id") or ""),
        checkpoint_id=str(raw.get("checkpoint_id") or ""),
        reason=str(raw.get("reason") or ""),
        status=status,
        created_at=max(0.0, _float(raw.get("created_at"))),
        closed_at=max(0.0, _float(raw.get("closed_at"))),
        task_ids=task_ids,
        replaced_branch_id=str(raw.get("replaced_branch_id") or ""),
        evidence=dict(evidence) if isinstance(evidence, Mapping) else {},
        rollback_count=max(0, _int(raw.get("rollback_count"))),
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class BranchCheckpointConfig:
    """Configuration for :class:`BranchCheckpointManager`.

    ``directory`` is normally the same ``<run_dir>/checkpoints`` directory used
    by the existing :class:`llm_agent.runtime.CheckpointStore`.  ``state_dir``
    defaults to ``<directory>/branches`` and holds manifests, branch records,
    the monotonic counter and the never-rolled-back learning journal.
    """

    #: Directory of the (re-used) CheckpointStore.
    directory: str = ""
    #: Branch manifests root (default: ``<directory>/branches``).
    state_dir: str = ""
    #: How many manifests to keep.
    keep: int = 20
    #: Manifest schema version to write.
    manifest_version: int = 1
    #: Durability: fsync manifest/state metadata files before renaming them.
    fsync: bool = True


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #

class BranchCheckpointManager:
    """Branch-aware checkpoint manifests on top of the existing checkpoint store.

    Parameters
    ----------
    config:
        Directory/keep/fsync settings; defaults to :class:`BranchCheckpointConfig`.
    store:
        A :class:`llm_agent.runtime.CheckpointStore`-like object (injectable for
        tests).  When omitted, the real store is created lazily on first use via
        a function-local import (so importing this module never imports PyBoy).
    clock:
        Unix-time clock injected for deterministic tests.
    """

    def __init__(
        self,
        config: BranchCheckpointConfig | None = None,
        *,
        store: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        self.config = config or BranchCheckpointConfig()
        self._store = store
        self._store_was_injected = store is not None
        self._clock = clock if clock is not None else now_ts

        self._state_loaded = False
        self._branches: dict[str, BranchRecord] = {}
        self._current_branch_id = ""
        self._saved = 0
        self._latest_checkpoint_id = ""

    # ------------------------------------------------------------------ #
    # Paths / store plumbing
    # ------------------------------------------------------------------ #
    @property
    def directory(self) -> Path:
        """Store directory (configured, else a project-relative default)."""
        raw = str(self.config.directory or "").strip()
        return Path(raw) if raw else _DEFAULT_DIRECTORY

    @property
    def state_dir(self) -> Path:
        """Root of the branch manifest layer."""
        raw = str(self.config.state_dir or "").strip()
        return Path(raw) if raw else self.directory / STATE_DIR_NAME

    @property
    def manifests_dir(self) -> Path:
        return self.state_dir / "manifests"

    @property
    def counter_path(self) -> Path:
        return self.state_dir / "counter.json"

    @property
    def branches_path(self) -> Path:
        return self.state_dir / "branches.json"

    @property
    def learning_path(self) -> Path:
        return self.state_dir / "learning.json"

    @property
    def keep(self) -> int:
        return max(1, _int(self.config.keep, 1))

    @property
    def manifest_version(self) -> int:
        version = _int(self.config.manifest_version, 1)
        return version if version > 0 else 1

    @property
    def store(self) -> Any:
        """The injected store, or the real runtime store created lazily."""
        if self._store is None:
            # Local import: keeps module import light (no PyBoy at import time).
            from llm_agent.runtime import CheckpointStore

            directory = self.directory
            directory.mkdir(parents=True, exist_ok=True)
            self._store = CheckpointStore(directory, keep=self.keep)
        return self._store

    @property
    def store_injected(self) -> bool:
        """True when the caller supplied the store in the constructor."""
        return self._store_was_injected

    def _now(self) -> float:
        return float(self._clock())

    def _ensure_dirs(self) -> None:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _store_meta_value(meta: Any, *names: str) -> str:
        """Read one attribute/key from a store result (CheckpointMeta or dict)."""
        for name in names:
            if isinstance(meta, Mapping):
                value = meta.get(name)
            else:
                value = getattr(meta, name, None)
            if value:
                return str(value)
        return ""

    # ------------------------------------------------------------------ #
    # Branch state persistence
    # ------------------------------------------------------------------ #
    def _manifest_paths(self) -> list[Path]:
        """Manifest files sorted oldest → newest (names are zero-padded ids)."""
        directory = self.manifests_dir
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.glob(f"{CHECKPOINT_ID_PREFIX}*.json") if path.is_file())

    def _max_manifest_seq(self) -> int:
        best = 0
        for path in self._manifest_paths():
            raw = _read_json(path)
            if raw is None:
                continue
            best = max(best, max(0, _int(raw.get("seq"))))
        return best

    def _persist_state(self) -> None:
        """Atomically persist branch records + the monotonic saved counter."""
        self._ensure_dirs()
        now = self._now()
        branches = sorted(
            self._branches.values(), key=lambda record: (record.created_at, record.branch_id)
        )
        _write_json_atomic(
            self.branches_path,
            {
                "version": 1,
                "updated_at": now,
                "current_branch_id": self._current_branch_id,
                "branches": [record.to_dict() for record in branches],
            },
            fsync=bool(self.config.fsync),
        )
        _write_json_atomic(
            self.counter_path,
            {
                "version": 1,
                "updated_at": now,
                "saved": self._saved,
                "last_checkpoint_id": self._latest_checkpoint_id,
            },
            fsync=bool(self.config.fsync),
        )

    def _ensure_state(self) -> None:
        """Load (or lazily create) branch state; tolerant to corrupt/missing files."""
        if self._state_loaded:
            return
        self._ensure_dirs()
        counter = _read_json(self.counter_path)
        if counter is not None:
            self._saved = max(0, _int(counter.get("saved")))
            self._latest_checkpoint_id = str(counter.get("last_checkpoint_id") or "")

        branches: dict[str, BranchRecord] = {}
        current = ""
        data = _read_json(self.branches_path)
        if data is not None:
            current = str(data.get("current_branch_id") or "")
            raw_branches = data.get("branches")
            if isinstance(raw_branches, Sequence) and not isinstance(raw_branches, (str, bytes)):
                for raw in raw_branches:
                    if not isinstance(raw, Mapping):
                        continue
                    try:
                        record = _branch_from_dict(raw)
                    except Exception:  # noqa: BLE001 - corrupt branch rows are skipped
                        continue
                    branches[record.branch_id] = record

        if not branches:
            initial = BranchRecord.create(reason="initial branch", created_at=self._now())
            branches[initial.branch_id] = initial
            current = initial.branch_id
        if current not in branches:
            current = sorted(
                branches.values(), key=lambda record: (record.created_at, record.branch_id)
            )[-1].branch_id

        # Reconcile the counter with the manifests so a lost counter file can
        # never produce a duplicate checkpoint id.
        self._saved = max(self._saved, self._max_manifest_seq())
        self._branches = branches
        self._current_branch_id = current
        self._state_loaded = True
        self._persist_state()

    # ------------------------------------------------------------------ #
    # Public branch API
    # ------------------------------------------------------------------ #
    @property
    def current_branch(self) -> BranchRecord:
        """The branch the system is currently working on (copy)."""
        self._ensure_state()
        return _copy_branch(self._branches[self._current_branch_id])

    def list_branches(self) -> list[BranchRecord]:
        """All known branches, oldest first (copies; safe for the caller to mutate)."""
        self._ensure_state()
        ordered = sorted(
            self._branches.values(), key=lambda record: (record.created_at, record.branch_id)
        )
        return [_copy_branch(record) for record in ordered]

    def create_child_branch(
        self,
        *,
        parent_branch_id: str = "",
        reason: str = "",
        evidence: Mapping[str, Any] | None = None,
        task_ids: Sequence[str] = (),
    ) -> BranchRecord:
        """Create a new active branch from ``parent_branch_id`` and make it current."""
        self._ensure_state()
        parent = str(parent_branch_id or "").strip() or self._current_branch_id
        record = BranchRecord.create(
            parent_branch_id=parent,
            reason=str(reason or ""),
            evidence=_json_safe(dict(evidence or {})),
            task_ids=[str(task) for task in (task_ids or ())],
            created_at=self._now(),
        )
        self._branches[record.branch_id] = record
        self._current_branch_id = record.branch_id
        self._persist_state()
        return _copy_branch(record)

    def mark_rolled_back(self, branch_id: str, *, reason: str = "") -> BranchRecord:
        """Mark ``branch_id`` as rolled back and record the reason (audit)."""
        self._ensure_state()
        key = str(branch_id or "").strip()
        record = self._branches.get(key)
        if record is None:
            raise KeyError(f"unknown branch: {branch_id!r}")
        record.status = BranchStatus.ROLLED_BACK
        record.closed_at = self._now()
        record.rollback_count += 1
        if reason:
            record.evidence["rollback_reason"] = str(reason)[:400]
        self._persist_state()
        return _copy_branch(record)

    # ------------------------------------------------------------------ #
    # Checkpoint save / load
    # ------------------------------------------------------------------ #
    def _resolve_branch(self, branch_id: str) -> BranchRecord:
        """Return the branch to snapshot on, creating it when a new id is supplied."""
        requested = str(branch_id or "").strip()
        if not requested:
            return self._branches[self._current_branch_id]
        if not requested.startswith("branch_"):
            # The frozen contract requires the protocol prefix; never persist an
            # invalid branch id that BranchRecord.from_dict would reject later.
            requested = new_branch_id()
        record = self._branches.get(requested)
        if record is None:
            record = BranchRecord.create(
                parent_branch_id=self._current_branch_id,
                reason="branch created by checkpoint save",
                created_at=self._now(),
            )
            self._branches[record.branch_id] = record
            requested = record.branch_id
        self._current_branch_id = requested
        return record

    def _next_saved_seq(self) -> int:
        """Reserve the next unique saved counter value (persisted immediately)."""
        self._saved = max(self._saved, self._max_manifest_seq()) + 1
        self._ensure_dirs()
        _write_json_atomic(
            self.counter_path,
            {
                "version": 1,
                "updated_at": self._now(),
                "saved": self._saved,
                "last_checkpoint_id": self._latest_checkpoint_id,
            },
            fsync=bool(self.config.fsync),
        )
        return self._saved

    @staticmethod
    def _components_progress(
        *,
        frame: int,
        tasks_snapshot: Mapping[str, Any],
        director_state: Mapping[str, Any],
        agent_contexts: Mapping[str, Any],
        navigation_snapshot: Mapping[str, Any],
        knowledge_version: int,
    ) -> dict[str, Any]:
        """Derived per-component progress summary stored inside the manifest."""
        tasks: list[Any] = []
        raw_tasks = tasks_snapshot.get("tasks") if isinstance(tasks_snapshot, Mapping) else None
        if isinstance(raw_tasks, (list, tuple)):
            tasks = list(raw_tasks)
        elif isinstance(raw_tasks, Mapping):
            tasks = list(raw_tasks.values())
        by_status: dict[str, int] = {}
        for task in tasks:
            if isinstance(task, Mapping):
                status = str(task.get("status") or "unknown")
                by_status[status] = by_status.get(status, 0) + 1

        navigation: dict[str, Any] = {}
        if isinstance(navigation_snapshot, Mapping):
            for source, target in (
                ("cells_known", "cells"),
                ("cells", "cells"),
                ("coverage_ratio", "coverage_ratio"),
                ("areas_total", "areas"),
                ("areas", "areas"),
                ("world_model_version", "world_model_version"),
                ("version", "world_model_version"),
            ):
                if target not in navigation and navigation_snapshot.get(source) is not None:
                    navigation[target] = navigation_snapshot.get(source)

        director: dict[str, Any] = {}
        if isinstance(director_state, Mapping):
            for key in ("status", "iteration", "current_goal", "mode", "phase"):
                if director_state.get(key) is not None:
                    director[key] = director_state.get(key)

        return {
            "frame": int(frame),
            "tasks": {"total": len(tasks), "by_status": dict(sorted(by_status.items()))},
            "navigation": _json_safe(navigation),
            "director": _json_safe(director),
            "knowledge_version": int(knowledge_version),
            "agents": sorted(str(name) for name in (agent_contexts or {})),
        }

    def save(
        self,
        *,
        state_bytes: bytes,
        frame: int,
        buttons: int,
        batches: int,
        reason: str,
        tasks_snapshot: Mapping[str, Any],
        director_state: Mapping[str, Any],
        agent_contexts: Mapping[str, Any],
        navigation_snapshot: Mapping[str, Any] | None = None,
        knowledge_version: int = 0,
        message_progress: Mapping[str, Any] | None = None,
        budget_snapshot: Mapping[str, Any] | None = None,
        memory_snapshot: Mapping[str, Any] | None = None,
        branch_id: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write one branch checkpoint: store state + atomic manifest, return it.

        The emulator state bytes and the memory/budget/runtime snapshots are
        delegated to the existing checkpoint store; this method adds the branch
        manifest (checksums, task/director/agent/navigation/knowledge/message
        snapshots) and keeps the branch records + monotonic counter up to date.
        """
        self._ensure_state()
        if not isinstance(state_bytes, (bytes, bytearray)):
            raise TypeError("state_bytes must be bytes")
        state = bytes(state_bytes)
        frame_value = max(0, _int(frame))
        reason_text = str(reason or "")
        evidence_payload = _json_safe(dict(evidence or {}))
        tasks_payload = _json_safe(dict(tasks_snapshot or {}))
        director_payload = _json_safe(dict(director_state or {}))
        agents_payload = _json_safe(dict(agent_contexts or {}))
        navigation_payload = _json_safe(dict(navigation_snapshot or {}))
        message_payload = _json_safe(dict(message_progress or {}))
        budget_payload = _json_safe(dict(budget_snapshot or {}))
        memory_payload = _json_safe(dict(memory_snapshot or {}))
        knowledge = max(0, _int(knowledge_version))

        seq = self._next_saved_seq()
        checkpoint_id = f"{CHECKPOINT_ID_PREFIX}{seq:06d}"
        manifest_path = self.manifests_dir / f"{checkpoint_id}.json"
        branch = self._resolve_branch(branch_id)
        created_at = self._now()

        runtime_meta = {
            "schema": "multi_agent.branch_checkpoint",
            "manifest_version": self.manifest_version,
            "checkpoint_id": checkpoint_id,
            "manifest_path": str(manifest_path),
            "branch_id": branch.branch_id,
            "parent_branch_id": branch.parent_branch_id,
            "seq": seq,
            "reason": reason_text,
            "frame": frame_value,
            "created_at": created_at,
        }
        try:
            store_result = self.store.save(
                state_bytes=state,
                frame=frame_value,
                buttons=max(0, _int(buttons)),
                batches=max(0, _int(batches)),
                reason=reason_text,
                memory_snapshot=memory_payload or None,
                budget_snapshot=budget_payload or None,
                runtime_snapshot={
                    "multi_agent": runtime_meta,
                    "branch": branch.to_dict(),
                },
            )
        except Exception as exc:  # noqa: BLE001 - wrap with actionable context
            raise RuntimeError(
                f"checkpoint store failed to save {checkpoint_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        store_checkpoint_id = self._store_meta_value(
            store_result, "id", "checkpoint_id"
        ) or checkpoint_id
        store_frame = _int(self._store_meta_value(store_result, "frame"))
        effective_frame = frame_value or store_frame

        manifest: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "checkpoint_id": checkpoint_id,
            "store_checkpoint_id": store_checkpoint_id,
            "seq": seq,
            "branch_id": branch.branch_id,
            "parent_branch_id": branch.parent_branch_id,
            "frame": effective_frame,
            "buttons": max(0, _int(buttons)),
            "batches": max(0, _int(batches)),
            "state_sha256": _sha256_bytes(state),
            "state_size": len(state),
            "components_progress": self._components_progress(
                frame=effective_frame,
                tasks_snapshot=tasks_payload,
                director_state=director_payload,
                agent_contexts=agents_payload,
                navigation_snapshot=navigation_payload,
                knowledge_version=knowledge,
            ),
            "tasks_snapshot": tasks_payload,
            "director_state": director_payload,
            "agent_contexts": agents_payload,
            "navigation_snapshot": navigation_payload,
            "knowledge_version": knowledge,
            "message_progress": message_payload,
            "budget_snapshot": budget_payload,
            "memory_in_store": bool(memory_payload),
            "created_at": created_at,
            "reason": reason_text,
            "evidence": evidence_payload,
            "store": {
                "type": type(self._store).__name__,
                "directory": str(self.directory),
            },
        }
        manifest["manifest_sha256"] = _manifest_digest(manifest)
        _write_json_atomic(manifest_path, manifest, fsync=bool(self.config.fsync))

        branch.checkpoint_id = checkpoint_id
        self._branches[branch.branch_id] = branch
        self._latest_checkpoint_id = checkpoint_id
        self._persist_state()
        self._prune_manifests()
        return _json_safe(manifest)

    def _find_manifest_path(self, checkpoint_id: str) -> Path:
        cid = str(checkpoint_id or "").strip()
        if cid in ("", "latest"):
            latest = self.latest()
            if latest is None:
                raise KeyError("no checkpoint manifest has been saved yet")
            cid = str(latest.get("checkpoint_id") or "")
        direct = self.manifests_dir / f"{cid}.json"
        if direct.is_file():
            return direct
        for path in self._manifest_paths():
            raw = _read_json(path)
            if raw is None:
                continue
            if str(raw.get("checkpoint_id") or "") == cid or str(
                raw.get("store_checkpoint_id") or ""
            ) == cid:
                return path
        raise KeyError(f"checkpoint manifest not found: {checkpoint_id!r}")

    def load_manifest(self, checkpoint_id: str) -> dict[str, Any]:
        """Load and return one manifest (``""``/``"latest"`` resolves to newest)."""
        path = self._find_manifest_path(checkpoint_id)
        raw = _read_json(path)
        if raw is None:
            raise KeyError(f"checkpoint manifest unreadable: {path}")
        return raw

    def latest(self) -> dict[str, Any] | None:
        """The newest manifest (by monotonic saved counter), or ``None``."""
        best: dict[str, Any] | None = None
        best_seq = -1
        for path in self._manifest_paths():
            raw = _read_json(path)
            if raw is None:
                continue
            seq = _int(raw.get("seq"), -1)
            if seq > best_seq:
                best_seq = seq
                best = raw
        return None if best is None else _json_safe(best)

    def rollback_manifest(self, checkpoint_id: str) -> dict[str, Any]:
        """Return the manifest to restore.

        No emulator action happens here: the recovery manager / director applies
        the state bytes via ``emulator.load_state`` and then re-observes and
        re-localizes navigation.  The manifest is returned verbatim so the caller
        can read the component snapshots and the ``state_sha256`` to verify the
        state bytes before applying them.
        """
        return _json_safe(self.load_manifest(checkpoint_id))

    def load_state_bytes(self, checkpoint_id: str) -> bytes:
        """Return the emulator state bytes for ``checkpoint_id`` (via the store)."""
        manifest = self.load_manifest(checkpoint_id)
        store_checkpoint_id = str(
            manifest.get("store_checkpoint_id") or manifest.get("checkpoint_id") or ""
        )
        meta = self.store.find(store_checkpoint_id)
        return self.store.state_bytes(meta)

    def verify(self, checkpoint_id: str) -> dict[str, Any]:
        """Verify a manifest and its state bytes (detects tampering).

        Returns ``{"ok", "reason", "manifest", "checkpoint_id", "checks"}``; the
        ``checks`` mapping lists every individual verification (manifest
        checksum, required fields, manifest version, state sha format, store
        availability, state checksum).
        """
        cid = str(checkpoint_id or "").strip()
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        result: dict[str, Any] = {
            "ok": False,
            "reason": "",
            "manifest": None,
            "checkpoint_id": cid,
            "checks": checks,
        }
        try:
            manifest = self.load_manifest(cid)
        except KeyError as exc:
            result["reason"] = str(exc)
            return result
        result["manifest"] = _json_safe(manifest)

        stored_digest = str(manifest.get("manifest_sha256") or "")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        digest_ok = bool(stored_digest) and _manifest_digest(payload) == stored_digest
        checks["manifest_sha256_match"] = digest_ok
        if not digest_ok:
            reasons.append("manifest checksum mismatch (manifest was modified)")

        missing = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
        checks["required_fields"] = not missing
        if missing:
            reasons.append(f"missing manifest fields: {missing}")

        version = manifest.get("manifest_version")
        supported = set(_SUPPORTED_MANIFEST_VERSIONS) | {self.manifest_version}
        checks["manifest_version_ok"] = isinstance(version, int) and version in supported
        if not checks["manifest_version_ok"]:
            reasons.append(f"unsupported manifest_version: {version!r}")

        state_sha = str(manifest.get("state_sha256") or "")
        checks["state_sha256_format"] = (
            len(state_sha) == 64 and set(state_sha.lower()) <= _HEX_DIGITS
        )
        if not checks["state_sha256_format"]:
            reasons.append("invalid state_sha256 in manifest")

        store_checkpoint_id = str(
            manifest.get("store_checkpoint_id") or manifest.get("checkpoint_id") or ""
        )
        try:
            meta = self.store.find(store_checkpoint_id)
            actual = self.store.state_bytes(meta)
            checks["store_checkpoint"] = True
            checks["state_sha256_match"] = _sha256_bytes(actual) == state_sha
            if not checks["state_sha256_match"]:
                reasons.append("state bytes checksum mismatch (state file was modified)")
        except Exception as exc:  # noqa: BLE001 - report, never raise
            checks["store_checkpoint"] = False
            checks["state_sha256_match"] = False
            reasons.append(f"state bytes unavailable: {type(exc).__name__}: {exc}")

        critical = (
            "manifest_sha256_match",
            "required_fields",
            "manifest_version_ok",
            "state_sha256_format",
            "store_checkpoint",
            "state_sha256_match",
        )
        result["ok"] = all(checks.get(key, False) for key in critical)
        result["reason"] = "; ".join(reasons)
        return result

    # ------------------------------------------------------------------ #
    # Learned experience (never rolled back)
    # ------------------------------------------------------------------ #
    def record_learning(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Append one learned-experience record to the separate learning journal.

        Learning data deliberately lives outside the checkpoint manifest/store so
        a rollback can never erase it.
        """
        self._ensure_dirs()
        payload = _read_json(self.learning_path) or {}
        raw_entries = payload.get("entries")
        entries: list[dict[str, Any]] = (
            [dict(item) for item in raw_entries if isinstance(item, Mapping)]
            if isinstance(raw_entries, list)
            else []
        )
        record = {"at": self._now(), **_json_safe(dict(entry))}
        entries.append(record)
        _write_json_atomic(
            self.learning_path,
            {"version": 1, "updated_at": self._now(), "entries": entries},
            fsync=bool(self.config.fsync),
        )
        return dict(record)

    def learning_entries(self) -> list[dict[str, Any]]:
        """All learned-experience records recorded so far."""
        payload = _read_json(self.learning_path) or {}
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            return []
        return [dict(item) for item in raw_entries if isinstance(item, Mapping)]

    # ------------------------------------------------------------------ #
    # Maintenance / reporting
    # ------------------------------------------------------------------ #
    def _prune_manifests(self) -> list[str]:
        """Delete the oldest manifests beyond ``keep``; return removed ids."""
        paths = self._manifest_paths()
        excess = len(paths) - self.keep
        if excess <= 0:
            return []
        removed: list[str] = []
        for path in paths[:excess]:
            removed.append(path.stem)
            try:
                path.unlink()
            except OSError:  # pragma: no cover - best effort pruning
                pass
        return removed

    def status(self) -> dict[str, Any]:
        """JSON-safe status snapshot for the viewer/report."""
        self._ensure_state()
        latest = self.latest()
        return {
            "directory": str(self.directory),
            "state_dir": str(self.state_dir),
            "manifests_dir": str(self.manifests_dir),
            "store": "injected" if self.store_injected else "runtime.CheckpointStore (lazy)",
            "keep": self.keep,
            "manifest_version": self.manifest_version,
            "saved": self._saved,
            "latest_checkpoint_id": self._latest_checkpoint_id,
            "manifest_count": len(self._manifest_paths()),
            "current_branch": self.current_branch.to_dict(),
            "branch_count": len(self._branches),
            "active_branches": sum(
                1 for record in self._branches.values() if record.status is BranchStatus.ACTIVE
            ),
            "learning_entries": len(self.learning_entries()),
            "latest": (
                None
                if latest is None
                else {
                    key: latest.get(key)
                    for key in ("checkpoint_id", "branch_id", "frame", "seq", "created_at", "reason")
                }
            ),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BranchCheckpointManager(dir={self.directory}, saved={self._saved}, "
            f"branches={len(self._branches)}, current={self._current_branch_id})"
        )
