"""Unit tests for :mod:`llm_agent.multi_agent.checkpoint`.

A fake checkpoint store is injected for most tests (the manager must accept any
``llm_agent.runtime.CheckpointStore``-like object), plus one integration test that
uses the *real* :class:`llm_agent.runtime.CheckpointStore` to prove the manifest
layer genuinely extends it instead of replacing it.  No ROM/PyBoy/network is used
anywhere.  Covered:

* complete manifest contents (frozen field list), atomic write + checksum,
* unique ids across an emulator frame restart, monotonic persisted counter,
* branch lifecycle (initial branch, child branches, rollback marking),
* verify detects manifest tampering, state tampering and missing state,
* keep/prune, missing/corrupt state-file tolerance and missing-store tolerance,
* learned-experience journal living outside the manifest (never rolled back).
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from llm_agent.multi_agent.checkpoint import (
    BranchCheckpointConfig,
    BranchCheckpointManager,
    CHECKPOINT_ID_PREFIX,
)
from llm_agent.multi_agent.contracts import BranchStatus


# --------------------------------------------------------------------------- #
# Fake store (CheckpointMeta-compatible)
# --------------------------------------------------------------------------- #

class FakeStore:
    """Minimal stand-in for ``llm_agent.runtime.CheckpointStore``."""

    def __init__(self) -> None:
        self.states: dict[str, bytes] = {}
        self.metas: dict[str, Any] = {}
        self.order: list[Any] = []
        self.saves: list[dict[str, Any]] = []
        self.fail_save = False
        #: when set, the fake store reports its own authoritative frame
        self.frame_override: int | None = None

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
    ) -> Any:
        if self.fail_save:
            raise OSError("disk full")
        meta_frame = self.frame_override if self.frame_override is not None else frame
        checkpoint_id = f"checkpoint_{meta_frame:09d}"
        if checkpoint_id in self.states:  # frame restarted: same store id
            checkpoint_id = f"{checkpoint_id}_{len(self.order):02d}"
        meta = types.SimpleNamespace(
            id=checkpoint_id,
            frame=meta_frame,
            buttons=buttons,
            batches=batches,
            reason=reason,
            created_at=0.0,
        )
        self.states[checkpoint_id] = bytes(state_bytes)
        self.metas[checkpoint_id] = meta
        self.order.append(meta)
        self.saves.append(
            {
                "checkpoint_id": checkpoint_id,
                "state_bytes": bytes(state_bytes),
                "memory_snapshot": memory_snapshot,
                "budget_snapshot": budget_snapshot,
                "runtime_snapshot": runtime_snapshot,
            }
        )
        return meta

    def list(self) -> list[Any]:
        return list(self.order)

    def latest(self) -> Any | None:
        return self.order[-1] if self.order else None

    def find(self, checkpoint_id: str) -> Any:
        if checkpoint_id in ("", "latest"):
            if not self.order:
                raise KeyError("no checkpoint")
            return self.order[-1]
        if checkpoint_id not in self.states:
            raise KeyError(checkpoint_id)
        return self.metas[checkpoint_id]

    def state_bytes(self, meta: Any) -> bytes:
        return self.states[meta.id]

    # -- test helpers ----------------------------------------------- #
    def tamper_state(self, checkpoint_id: str) -> None:
        self.states[checkpoint_id] = b"TAMPERED-STATE"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _manager(tmp_path: Path, *, store: Any | None = None, clock: Any | None = None, **cfg: Any) -> BranchCheckpointManager:
    values: dict[str, Any] = dict(directory=str(tmp_path / "checkpoints"), keep=20, fsync=False)
    values.update(cfg)
    return BranchCheckpointManager(
        BranchCheckpointConfig(**values), store=store or FakeStore(), clock=clock
    )


def _save(manager: BranchCheckpointManager, *, frame: int = 120, reason: str = "auto",
          state: bytes = b"EMULATOR-STATE", **extra: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        state_bytes=state,
        frame=frame,
        buttons=10,
        batches=2,
        reason=reason,
        tasks_snapshot={"tasks": [{"task_id": "task_a", "status": "running"}]},
        director_state={"status": "running", "iteration": 3},
        agent_contexts={"navigation": {"goal": "unexplored cell"}},
    )
    kwargs.update(extra)
    return manager.save(**kwargs)


def _expected_manifest_fields() -> set[str]:
    """Frozen manifest field list from the interface contract."""
    return {
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
    }


# --------------------------------------------------------------------------- #
# Config / initial state
# --------------------------------------------------------------------------- #

def test_config_defaults_match_frozen_interface(tmp_path: Path) -> None:
    config = BranchCheckpointConfig()
    assert config.directory == ""
    assert config.state_dir == ""
    assert config.keep == 20
    assert config.manifest_version == 1
    assert config.fsync is True
    manager = _manager(tmp_path)
    assert manager.state_dir == Path(str(tmp_path / "checkpoints")) / "branches"


def test_initial_branch_is_created_lazily(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    branch = manager.current_branch
    assert branch.branch_id.startswith("branch_")
    assert branch.status is BranchStatus.ACTIVE
    assert manager.list_branches() == [branch]
    assert manager.latest() is None
    assert manager.verify("latest")["ok"] is False


def test_list_branches_returns_independent_copies(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    branch = manager.list_branches()[0]
    branch.reason = "mutated by caller"
    assert manager.list_branches()[0].reason != "mutated by caller"


# --------------------------------------------------------------------------- #
# Save / load / manifest contents
# --------------------------------------------------------------------------- #

def test_save_writes_complete_manifest(tmp_path: Path) -> None:
    clock = lambda: 1_000.0  # noqa: E731 - tiny deterministic clock
    store = FakeStore()
    manager = _manager(tmp_path, store=store, clock=clock)
    manifest = _save(
        manager,
        navigation_snapshot={"cells_known": 12, "areas_total": 2, "coverage_ratio": 0.25},
        knowledge_version=7,
        message_progress={"navigation": 3},
        budget_snapshot={"run": {"cost_usd": 0.5, "calls": 4}},
        memory_snapshot={"notes": ["a"]},
        evidence={"trigger": "auto"},
    )
    expected = _expected_manifest_fields()
    assert expected.issubset(manifest)
    assert manifest["manifest_version"] == 1
    assert manifest["checkpoint_id"] == f"{CHECKPOINT_ID_PREFIX}000001"
    assert manifest["store_checkpoint_id"] == "checkpoint_000000120"
    assert manifest["branch_id"].startswith("branch_")
    assert manifest["parent_branch_id"] == ""
    assert manifest["state_sha256"] == hashlib.sha256(b"EMULATOR-STATE").hexdigest()
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["knowledge_version"] == 7
    assert manifest["message_progress"] == {"navigation": 3}
    assert manifest["budget_snapshot"]["run"]["cost_usd"] == 0.5
    assert manifest["reason"] == "auto"
    assert manifest["evidence"] == {"trigger": "auto"}
    assert manifest["created_at"] == 1_000.0
    assert manifest["components_progress"]["tasks"]["by_status"] == {"running": 1}
    assert manifest["components_progress"]["navigation"]["cells"] == 12
    assert manifest["components_progress"]["knowledge_version"] == 7

    manifest_path = manager.manifests_dir / f"{manifest['checkpoint_id']}.json"
    assert manifest_path.is_file()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["manifest_sha256"] == manifest["manifest_sha256"]

    # Delegated to the existing store, including the multi-agent runtime metadata.
    assert store.saves[0]["state_bytes"] == b"EMULATOR-STATE"
    runtime_meta = store.saves[0]["runtime_snapshot"]["multi_agent"]
    assert runtime_meta["checkpoint_id"] == manifest["checkpoint_id"]
    assert runtime_meta["manifest_path"] == str(manifest_path)
    assert runtime_meta["branch_id"] == manifest["branch_id"]
    assert store.saves[0]["memory_snapshot"] == {"notes": ["a"]}
    assert store.saves[0]["budget_snapshot"]["run"]["calls"] == 4

    # The current branch now points at the checkpoint.
    assert manager.current_branch.checkpoint_id == manifest["checkpoint_id"]
    assert manager.load_manifest(manifest["checkpoint_id"]) == on_disk
    assert manager.load_manifest("latest")["checkpoint_id"] == manifest["checkpoint_id"]
    assert manager.latest()["checkpoint_id"] == manifest["checkpoint_id"]


def test_save_requires_bytes_state(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(TypeError):
        _save(manager, state="not-bytes")  # type: ignore[arg-type]


def test_save_tolerates_non_json_snapshot_values(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    class Opaque:
        def __str__(self) -> str:  # pragma: no cover - trivial
            return "opaque-value"

    manifest = _save(
        manager,
        tasks_snapshot={"tasks": [{"status": Opaque()}]},  # type: ignore[dict-item]
        director_state={"status": Opaque()},  # type: ignore[dict-item]
    )
    assert json.loads(json.dumps(manifest)) == manifest  # fully JSON-safe


def test_save_uses_store_frame_when_caller_frame_is_unknown(tmp_path: Path) -> None:
    store = FakeStore()
    store.frame_override = 777
    manager = _manager(tmp_path, store=store)
    manifest = _save(manager, frame=0)
    # frame <= 0 means "unknown": the store's CheckpointMeta.frame is authoritative.
    assert store.saves
    assert manifest["frame"] == 777
    assert manifest["store_checkpoint_id"] == "checkpoint_000000777"


def test_atomic_writes_leave_no_temp_files(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _save(manager)
    _save(manager, frame=240)
    assert list(manager.state_dir.glob("*.tmp")) == []
    assert list(manager.manifests_dir.glob("*.tmp")) == []
    json.loads(manager.counter_path.read_text(encoding="utf-8"))
    json.loads(manager.branches_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Unique ids / monotonic counter
# --------------------------------------------------------------------------- #

def test_checkpoint_ids_stay_unique_when_the_frame_restarts(tmp_path: Path) -> None:
    store = FakeStore()
    manager = _manager(tmp_path, store=store)
    first = _save(manager, frame=5_000, state=b"STATE-A")
    second = _save(manager, frame=5_000, state=b"STATE-B")  # same store id, new manifest id
    third = _save(manager, frame=0, state=b"STATE-C")  # frame restarted from 0
    ids = [first["checkpoint_id"], second["checkpoint_id"], third["checkpoint_id"]]
    assert len(set(ids)) == 3
    assert [item["seq"] for item in (first, second, third)] == [1, 2, 3]
    store_ids = [item["store_checkpoint_id"] for item in (first, second, third)]
    assert store_ids[0] != store_ids[1]  # fake store disambiguates re-used frames
    counter = json.loads(manager.counter_path.read_text(encoding="utf-8"))
    assert counter["saved"] == 3
    assert counter["last_checkpoint_id"] == third["checkpoint_id"]


def test_state_persists_across_manager_instances(tmp_path: Path) -> None:
    store = FakeStore()
    first_manager = _manager(tmp_path, store=store)
    _save(first_manager)
    _save(first_manager, frame=240)
    branches_before = [branch.branch_id for branch in first_manager.list_branches()]

    second_manager = _manager(tmp_path, store=store)
    assert second_manager.current_branch.branch_id == first_manager.current_branch.branch_id
    assert [branch.branch_id for branch in second_manager.list_branches()] == branches_before
    third = _save(second_manager, frame=360)
    assert third["seq"] == 3  # counter continued, no id collision
    third_manager = _manager(tmp_path, store=store)
    assert third_manager.latest()["checkpoint_id"] == third["checkpoint_id"]


def test_lost_or_corrupt_counter_cannot_collide_ids(tmp_path: Path) -> None:
    store = FakeStore()
    manager = _manager(tmp_path, store=store)
    _save(manager)
    _save(manager, frame=240)
    manager.counter_path.write_text('{"saved": "junk"}', encoding="utf-8")
    fresh = _manager(tmp_path, store=store)
    third = _save(fresh, frame=360)
    assert third["seq"] == 3
    manager.counter_path.unlink()
    fresh_again = _manager(tmp_path, store=store)
    fourth = _save(fresh_again, frame=480)
    assert fourth["seq"] == 4


def test_corrupt_branch_state_is_repaired(tmp_path: Path) -> None:
    store = FakeStore()
    manager = _manager(tmp_path, store=store)
    saved = _save(manager)
    manager.branches_path.write_text("not json at all", encoding="utf-8")
    fresh = _manager(tmp_path, store=store)
    branch = fresh.current_branch
    assert branch.branch_id.startswith("branch_")
    assert fresh.verify(saved["checkpoint_id"])["ok"] is True


# --------------------------------------------------------------------------- #
# Branch lifecycle
# --------------------------------------------------------------------------- #

def test_create_child_branch_parents_and_activates(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    parent = manager.current_branch
    child = manager.create_child_branch(
        reason="rollback", evidence={"why": "stuck"}, task_ids=["task_a"]
    )
    assert child.branch_id != parent.branch_id
    assert child.parent_branch_id == parent.branch_id
    assert child.status is BranchStatus.ACTIVE
    assert child.evidence == {"why": "stuck"}
    assert child.task_ids == ["task_a"]
    assert manager.current_branch.branch_id == child.branch_id
    assert len(manager.list_branches()) == 2
    # A second manager sees the same branch tree.
    assert [branch.branch_id for branch in _manager(tmp_path).list_branches()] == [
        parent.branch_id,
        child.branch_id,
    ]


def test_mark_rolled_back_updates_status_and_reason(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    branch = manager.current_branch
    rolled = manager.mark_rolled_back(branch.branch_id, reason="stagnation")
    assert rolled.status is BranchStatus.ROLLED_BACK
    assert rolled.closed_at > 0
    assert rolled.rollback_count == 1
    assert rolled.evidence["rollback_reason"] == "stagnation"
    with pytest.raises(KeyError):
        manager.mark_rolled_back("branch_unknown")


def test_save_can_target_an_explicit_branch(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    child = manager.create_child_branch(reason="new plan")
    manifest = _save(manager, branch_id=child.branch_id)
    assert manifest["branch_id"] == child.branch_id
    assert manifest["parent_branch_id"] == child.parent_branch_id
    # Unknown ids get a fresh protocol-valid branch record instead of corrupting state.
    manifest2 = _save(manager, branch_id="not-a-protocol-id")
    assert manifest2["branch_id"].startswith("branch_")
    json.loads(manager.branches_path.read_text(encoding="utf-8"))


def test_rollback_manifest_returns_snapshot_without_touching_state(tmp_path: Path) -> None:
    store = FakeStore()
    manager = _manager(tmp_path, store=store)
    manifest = _save(manager, state=b"GOOD-STATE")
    state_before = dict(store.states)
    returned = manager.rollback_manifest(manifest["checkpoint_id"])
    assert returned["checkpoint_id"] == manifest["checkpoint_id"]
    assert returned["state_sha256"] == manifest["state_sha256"]
    assert store.states == state_before  # no emulator/store mutation here
    assert manager.load_state_bytes(manifest["checkpoint_id"]) == b"GOOD-STATE"
    with pytest.raises(KeyError):
        manager.rollback_manifest("checkpoint_999999")


# --------------------------------------------------------------------------- #
# Verify / tampering
# --------------------------------------------------------------------------- #

def test_verify_accepts_an_intact_checkpoint(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest = _save(manager)
    result = manager.verify(manifest["checkpoint_id"])
    assert result["ok"] is True
    assert result["reason"] == ""
    assert result["manifest"]["checkpoint_id"] == manifest["checkpoint_id"]
    assert all(
        result["checks"][key]
        for key in (
            "manifest_sha256_match",
            "required_fields",
            "manifest_version_ok",
            "state_sha256_format",
            "store_checkpoint",
            "state_sha256_match",
        )
    )


def test_verify_detects_manifest_tampering(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest = _save(manager)
    path = manager.manifests_dir / f"{manifest['checkpoint_id']}.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["frame"] = 999_999  # edit without recomputing the checksum
    path.write_text(json.dumps(tampered), encoding="utf-8")
    result = manager.verify(manifest["checkpoint_id"])
    assert result["ok"] is False
    assert "checksum mismatch" in result["reason"]
    assert result["checks"]["manifest_sha256_match"] is False


def test_verify_detects_state_tampering(tmp_path: Path) -> None:
    store = FakeStore()
    manager = _manager(tmp_path, store=store)
    manifest = _save(manager)
    store.tamper_state(manifest["store_checkpoint_id"])
    result = manager.verify(manifest["checkpoint_id"])
    assert result["ok"] is False
    assert "state bytes checksum mismatch" in result["reason"]
    assert result["checks"]["state_sha256_match"] is False


def test_verify_reports_missing_state_and_corrupt_manifest(tmp_path: Path) -> None:
    store = FakeStore()
    manager = _manager(tmp_path, store=store)
    manifest = _save(manager)
    del store.states[manifest["store_checkpoint_id"]]
    result = manager.verify(manifest["checkpoint_id"])
    assert result["ok"] is False
    assert result["checks"]["store_checkpoint"] is False

    path = manager.manifests_dir / f"{manifest['checkpoint_id']}.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(KeyError):
        manager.load_manifest(manifest["checkpoint_id"])


# --------------------------------------------------------------------------- #
# Keep / prune / store tolerance
# --------------------------------------------------------------------------- #

def test_keep_prunes_the_oldest_manifests(tmp_path: Path) -> None:
    manager = _manager(tmp_path, keep=2)
    saved = [_save(manager, frame=index * 100) for index in range(4)]
    assert len(list(manager.manifests_dir.glob("checkpoint_*.json"))) == 2
    assert manager.latest()["checkpoint_id"] == saved[-1]["checkpoint_id"]
    assert manager.verify(saved[0]["checkpoint_id"])["ok"] is False  # pruned
    assert manager.verify(saved[-1]["checkpoint_id"])["ok"] is True


def test_store_save_failure_is_wrapped_with_context(tmp_path: Path) -> None:
    store = FakeStore()
    store.fail_save = True
    manager = _manager(tmp_path, store=store)
    with pytest.raises(RuntimeError, match="checkpoint store failed"):
        _save(manager)
    # The reserved counter value is durable: the next successful save is unique.
    store.fail_save = False
    manifest = _save(manager)
    assert manifest["seq"] >= 2


def test_missing_store_is_created_lazily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def factory(directory: Any, keep: int = 20) -> FakeStore:
        calls.append((str(directory), keep))
        return FakeStore()

    fake_runtime = types.ModuleType("llm_agent.runtime")
    fake_runtime.CheckpointStore = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llm_agent.runtime", fake_runtime)

    manager = BranchCheckpointManager(
        BranchCheckpointConfig(directory=str(tmp_path / "checkpoints"), fsync=False)
    )
    # Read-only APIs work before the store exists (no heavy import needed).
    assert manager.status()["store"] == "runtime.CheckpointStore (lazy)"
    assert manager.latest() is None
    manifest = _save(manager)
    assert calls and calls[0][1] == 20
    assert manager.store_injected is False  # originally not injected
    assert manager.verify(manifest["checkpoint_id"])["ok"] is True


def test_real_runtime_checkpoint_store_integration(tmp_path: Path) -> None:
    """The manifest layer must work with the real (non-fake) CheckpointStore."""
    from llm_agent.runtime import CheckpointStore

    directory = tmp_path / "checkpoints"
    store = CheckpointStore(directory, keep=5)
    manager = BranchCheckpointManager(
        BranchCheckpointConfig(directory=str(directory), keep=5, fsync=False), store=store
    )
    state = b"real-store-state-bytes"
    manifest = _save(manager, frame=42, state=state)
    assert manifest["store_checkpoint_id"] == "checkpoint_000000042"
    assert manifest["frame"] == 42
    assert store.latest() is not None and store.latest().id == "checkpoint_000000042"
    assert manager.load_state_bytes(manifest["checkpoint_id"]) == state
    result = manager.verify(manifest["checkpoint_id"])
    assert result["ok"] is True, result["reason"]
    assert manager.rollback_manifest(manifest["checkpoint_id"])["state_sha256"] == (
        hashlib.sha256(state).hexdigest()
    )


# --------------------------------------------------------------------------- #
# Learned experience journal
# --------------------------------------------------------------------------- #

def test_learning_entries_survive_rollback_and_new_managers(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.record_learning({"category": "battle_lesson", "content": "move A is effective"})
    manifest = _save(manager)
    manager.rollback_manifest(manifest["checkpoint_id"])
    _save(manager, frame=480, reason="later")
    entries = manager.learning_entries()
    assert len(entries) == 1
    assert entries[0]["content"] == "move A is effective"
    assert "at" in entries[0]
    # A different manager instance reads the same never-rolled-back journal.
    assert _manager(tmp_path).learning_entries() == entries
    # Learning data is deliberately outside the manifest.
    serialized = json.dumps(manifest)
    assert "battle_lesson" not in serialized
    assert manager.learning_path.is_file()


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

def test_status_reports_paths_counts_and_latest(tmp_path: Path) -> None:
    store = FakeStore()
    manager = _manager(tmp_path, store=store)
    manifest = _save(manager)
    manager.create_child_branch(reason="rollback")
    status = manager.status()
    assert status["directory"] == str(tmp_path / "checkpoints")
    assert status["state_dir"] == str(tmp_path / "checkpoints" / "branches")
    assert status["store"] == "injected"
    assert status["keep"] == 20
    assert status["saved"] == 1
    assert status["manifest_count"] == 1
    assert status["branch_count"] == 2
    assert status["active_branches"] == 2
    assert status["latest"]["checkpoint_id"] == manifest["checkpoint_id"]
    assert status["current_branch"]["branch_id"] == manager.current_branch.branch_id
    json.dumps(status)


def test_default_directory_is_relative_only_until_used() -> None:
    # The constructor must not create the project-relative default directory.
    config = BranchCheckpointConfig(directory="")
    manager = BranchCheckpointManager(config)
    assert not manager.state_dir.is_absolute()
    assert manager.latest() is None
