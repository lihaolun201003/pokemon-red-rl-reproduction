"""Tests for the frozen multi-agent protocol (AgentMessage / Task / knowledge …).

These tests are the honesty contract of the subsystem: ids, validation, priority,
branch ids, task transitions and knowledge verification status must behave
exactly as documented, with no hidden game knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_agent.multi_agent import contracts
from llm_agent.multi_agent.contracts import (
    AgentMessage,
    AgentRole,
    BranchRecord,
    BranchStatus,
    ControlDeniedError,
    ControlLease,
    ControlOwner,
    CycleDetectedError,
    EventType,
    KnowledgeEntry,
    KnowledgeError,
    MultiAgentError,
    Priority,
    ProtocolError,
    RecoveryLevel,
    StagnationKind,
    Task,
    TaskError,
    TaskStatus,
    TaskType,
    VerificationStatus,
    dedupe_key,
    is_valid_transition,
    new_branch_id,
    new_knowledge_id,
    new_lease_id,
    new_message_id,
    new_task_id,
    stable_digest,
)

PACKAGE = Path(__file__).resolve().parents[2] / "llm_agent"


# --------------------------------------------------------------------------- #
# Enums & ids
# --------------------------------------------------------------------------- #

def test_event_catalogue_covers_the_mandated_events() -> None:
    required = {
        "TASK_CREATED", "TASK_ASSIGNED", "TASK_STARTED", "TASK_COMPLETED", "TASK_FAILED",
        "TASK_BLOCKED", "TASK_CANCELLED",
        "NAVIGATION_REQUEST", "NAVIGATION_COMPLETED", "NAVIGATION_FAILED",
        "BATTLE_STARTED", "BATTLE_ENDED", "POKEMON_NEEDS_HELP",
        "DIALOG_STARTED", "DIALOG_ENDED", "STORY_DISCOVERY",
        "MAP_TRANSITION", "NEW_AREA_DISCOVERED",
        "CONTROL_REQUESTED", "CONTROL_GRANTED", "CONTROL_RELEASED",
        "PROGRESS_DETECTED", "STAGNATION_DETECTED",
        "CHECKPOINT_CREATED", "ROLLBACK_REQUESTED", "ROLLBACK_COMPLETED",
        "AGENT_ERROR", "AGENT_PAUSED", "AGENT_RESUMED",
    }
    assert required.issubset(set(EventType.__members__))
    for member in EventType:
        assert member.value == member.value.lower()


def test_every_event_has_routing_and_payload_spec() -> None:
    for event in EventType:
        assert event in contracts.EVENT_ROUTING, event
        assert event in contracts.EVENT_PAYLOAD_SPEC, event
        required, optional = contracts.EVENT_PAYLOAD_SPEC[event]
        assert isinstance(required, tuple) and isinstance(optional, tuple)


def test_id_helpers_are_prefixed_and_unique() -> None:
    assert new_message_id().startswith("msg_")
    assert new_task_id().startswith("task_")
    assert new_knowledge_id().startswith("kn_")
    assert new_branch_id().startswith("branch_")
    assert new_lease_id().startswith("lease_")
    assert len({new_message_id() for _ in range(50)}) == 50


def test_role_and_owner_coercion() -> None:
    assert AgentRole.coerce("nav") is AgentRole.NAVIGATION
    assert AgentRole.coerce("STORY") is AgentRole.STORY
    assert AgentRole.coerce("pokemon") is AgentRole.POKEMON
    assert AgentRole.POKEMON.is_professional
    assert not AgentRole.DIRECTOR.is_professional
    assert ControlOwner.from_role(AgentRole.POKEMON) is ControlOwner.POKEMON
    assert ControlOwner.from_role("navigation") is ControlOwner.NAVIGATION
    with pytest.raises(ProtocolError):
        AgentRole.coerce("bogus")
    with pytest.raises(ProtocolError):
        ControlOwner.coerce("bogus")


def test_priority_coercion() -> None:
    assert Priority.coerce("high") is Priority.HIGH
    assert Priority.coerce(Priority.CRITICAL) is Priority.CRITICAL
    assert Priority.coerce(150) is Priority.CRITICAL
    assert Priority.coerce(-5) == Priority.BACKGROUND
    with pytest.raises(ProtocolError):
        Priority.coerce("extreme")


# --------------------------------------------------------------------------- #
# AgentMessage
# --------------------------------------------------------------------------- #

def test_message_create_validates_required_payload() -> None:
    with pytest.raises(ProtocolError):
        AgentMessage.create(EventType.TASK_CREATED, source_agent=AgentRole.STORY, payload={})
    message = AgentMessage.create(
        EventType.TASK_CREATED,
        source_agent=AgentRole.STORY,
        payload={"task_id": "task_x", "description": "look around"},
        priority=Priority.HIGH,
        correlation_id="corr-1",
        parent_task_id="task_parent",
        branch_id="branch_abc",
    )
    assert message.message_id.startswith("msg_")
    assert message.target_agent is AgentRole.DIRECTOR
    assert message.priority is Priority.HIGH
    assert message.task_id == "task_x"
    assert message.branch_id == "branch_abc"
    message.validate()


def test_message_roundtrip_and_json() -> None:
    message = AgentMessage.create(
        EventType.BATTLE_STARTED,
        source_agent=AgentRole.NAVIGATION,
        target_agent=AgentRole.DIRECTOR,
        payload={"frame": 12, "context": "battle"},
        branch_id="branch_1",
    )
    data = message.to_dict()
    assert data["protocol_version"] == contracts.PROTOCOL_VERSION
    clone = AgentMessage.from_dict(json.loads(message.to_json()))
    assert clone.message_id == message.message_id
    assert clone.event_type is EventType.BATTLE_STARTED
    assert clone.payload == message.payload
    assert clone.sequence == 0


def test_message_rejects_corrupt_records() -> None:
    bad = [
        {"message_id": "nope", "source_agent": "navigation", "target_agent": "director",
         "event_type": "battle_started", "payload": {}, "created_at": 0},
        {"message_id": "msg_1", "source_agent": "dragon", "target_agent": "director",
         "event_type": "battle_started", "payload": {}, "created_at": 0},
        {"message_id": "msg_1", "source_agent": "navigation", "target_agent": "director",
         "event_type": "unknown_event", "payload": {}, "created_at": 0},
        {"message_id": "msg_1", "source_agent": "navigation", "target_agent": "director",
         "event_type": "task_created", "payload": {}, "created_at": 0},
        {"message_id": "msg_1", "source_agent": "navigation", "target_agent": "director",
         "event_type": "battle_started", "payload": [], "created_at": 0},
    ]
    for record in bad:
        with pytest.raises(ProtocolError):
            AgentMessage.from_dict(record)


def test_message_dedupe_key_is_content_based() -> None:
    a = AgentMessage.create(
        EventType.BATTLE_STARTED, source_agent=AgentRole.NAVIGATION,
        payload={"frame": 3}, branch_id="b1",
    )
    b = AgentMessage.create(
        EventType.BATTLE_STARTED, source_agent=AgentRole.NAVIGATION,
        payload={"frame": 3}, branch_id="b1",
    )
    assert a.message_id != b.message_id
    assert a.dedupe_key() == b.dedupe_key()
    c = AgentMessage.create(
        EventType.BATTLE_STARTED, source_agent=AgentRole.NAVIGATION,
        payload={"frame": 4}, branch_id="b1",
    )
    assert c.dedupe_key() != a.dedupe_key()


def test_irreversible_and_bookkeeping_event_sets() -> None:
    assert EventType.BATTLE_STARTED in contracts.IRREVERSIBLE_EVENTS
    assert EventType.DIALOG_STARTED in contracts.IRREVERSIBLE_EVENTS
    assert EventType.SYSTEM_HEARTBEAT in contracts.BOOKKEEPING_EVENTS
    # task bookkeeping is not "irreversible"
    assert EventType.TASK_CREATED not in contracts.IRREVERSIBLE_EVENTS


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #

def test_task_create_and_roundtrip() -> None:
    task = Task.create(
        owner_agent=AgentRole.NAVIGATION,
        requested_by=AgentRole.STORY,
        task_type=TaskType.NAVIGATE,
        description="walk to the place the npc mentioned",
        priority=Priority.HIGH,
        branch_id="branch_1",
        payload={"target": {"kind": "exit"}},
    )
    assert task.status is TaskStatus.PENDING
    assert task.dedupe_key
    clone = Task.from_dict(json.loads(json.dumps(task.to_dict())))
    assert clone.task_id == task.task_id
    assert clone.payload == task.payload
    assert clone.priority is Priority.HIGH


def test_task_rejects_invalid_owner_and_self_dependency() -> None:
    with pytest.raises(ProtocolError):
        Task.create(owner_agent=AgentRole.DIRECTOR, requested_by=AgentRole.STORY,
                    task_type="custom", description="x")
    task = Task.create(owner_agent=AgentRole.STORY, requested_by=AgentRole.DIRECTOR,
                       task_type="talk", description="x")
    with pytest.raises(ProtocolError):
        Task.from_dict({**task.to_dict(), "dependencies": [task.task_id]})
    with pytest.raises(ProtocolError):
        Task.from_dict({**task.to_dict(), "parent_task_id": task.task_id})


def test_task_transitions_are_strict() -> None:
    assert is_valid_transition(TaskStatus.PENDING, TaskStatus.READY)
    assert is_valid_transition(TaskStatus.READY, TaskStatus.RUNNING)
    assert is_valid_transition(TaskStatus.RUNNING, TaskStatus.COMPLETED)
    assert is_valid_transition(TaskStatus.RUNNING, TaskStatus.BLOCKED)
    assert is_valid_transition(TaskStatus.BLOCKED, TaskStatus.READY)
    assert not is_valid_transition(TaskStatus.PENDING, TaskStatus.COMPLETED)
    assert not is_valid_transition(TaskStatus.CANCELLED, TaskStatus.RUNNING)
    task = Task.create(owner_agent=AgentRole.POKEMON, requested_by=AgentRole.DIRECTOR,
                       task_type="battle", description="fight")
    with pytest.raises(TaskError):
        task.transition(TaskStatus.COMPLETED)
    task.transition(TaskStatus.READY)
    task.transition(TaskStatus.RUNNING)
    task.transition(TaskStatus.COMPLETED)
    assert task.status is TaskStatus.COMPLETED
    assert task.revision > 1


def test_task_terminal_and_schedulable_flags() -> None:
    assert TaskStatus.COMPLETED.terminal and TaskStatus.CANCELLED.terminal
    assert TaskStatus.READY.schedulable and TaskStatus.PAUSED.schedulable
    assert not TaskStatus.RUNNING.schedulable


# --------------------------------------------------------------------------- #
# KnowledgeEntry
# --------------------------------------------------------------------------- #

def test_knowledge_entry_create_and_roundtrip() -> None:
    entry = KnowledgeEntry.create(
        category="story_clue",
        content="the npc said something about a parcel",
        source_agent=AgentRole.STORY,
        confidence=0.9,
        verification_status=VerificationStatus.VERIFIED,
        branch_id="branch_1",
        evidence_refs=["screens/step_00012_dialog.png"],
        subject="npc_a",
        tags=["parcel"],
    )
    assert entry.verification_status.trusted
    clone = KnowledgeEntry.from_dict(json.loads(json.dumps(entry.to_dict())))
    assert clone.content == entry.content
    assert clone.evidence_refs == entry.evidence_refs
    assert clone.dedupe_key == entry.dedupe_key


def test_knowledge_rejects_bad_category_and_confidence() -> None:
    with pytest.raises(KnowledgeError):
        KnowledgeEntry.create(category="walkthrough", content="x", source_agent=AgentRole.STORY)
    entry = KnowledgeEntry.create(category="area", content="x", source_agent=AgentRole.NAVIGATION)
    with pytest.raises(ProtocolError):
        KnowledgeEntry.from_dict({**entry.to_dict(), "confidence": 2.0})
    assert not VerificationStatus.HYPOTHESIS.trusted
    assert VerificationStatus.HISTORICAL is not VerificationStatus.REFUTED


# --------------------------------------------------------------------------- #
# Control lease / branch records
# --------------------------------------------------------------------------- #

def test_control_lease_create_and_roundtrip() -> None:
    lease = ControlLease.create(owner=ControlOwner.POKEMON, task_id="task_1", reason="battle")
    assert lease.owner is ControlOwner.POKEMON
    assert lease.agent is AgentRole.POKEMON
    assert lease.active
    clone = ControlLease.from_dict(json.loads(json.dumps(lease.to_dict())))
    assert clone.lease_id == lease.lease_id
    clone.released_at = 5.0
    assert not clone.active
    with pytest.raises(ProtocolError):
        ControlLease.create(owner=ControlOwner.NONE)


def test_branch_record_roundtrip() -> None:
    branch = BranchRecord.create(parent_branch_id="branch_parent", checkpoint_id="checkpoint_000000001",
                                 reason="stagnation", task_ids=["task_1"], evidence={"signal": "local"})
    branch.status = BranchStatus.ROLLED_BACK
    clone = BranchRecord.from_dict(json.loads(json.dumps(branch.to_dict())))
    assert clone.status is BranchStatus.ROLLED_BACK
    assert clone.parent_branch_id == "branch_parent"
    assert clone.rollback_count == 0


def test_stagnation_and_recovery_enums() -> None:
    for name in ("LOCAL_STAGNATION", "GLOBAL_STAGNATION", "REPEATED_FAILURE",
                 "NO_PROGRESS", "RESOURCE_EXHAUSTED", "RECOVERY_LIMIT"):
        assert name in StagnationKind.__members__
    for name in ("L1_LOCAL", "L2_REVIEW", "L3_ROLLBACK"):
        assert name in RecoveryLevel.__members__


def test_stable_digest_dedup_key_are_deterministic() -> None:
    assert stable_digest({"b": 1, "a": 2}) == stable_digest({"a": 2, "b": 1})
    assert dedupe_key("navigate", "go north") == dedupe_key("navigate", "go north")
    assert dedupe_key("navigate", "go north") != dedupe_key("navigate", "go south")


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #

def test_error_hierarchy_and_details() -> None:
    error = ControlDeniedError("denied", agent="story", owner="pokemon", details={"task_id": "t"})
    assert isinstance(error, MultiAgentError)
    payload = error.to_dict()
    assert payload["error"] == "ControlDeniedError"
    assert payload["details"]["agent"] == "story"
    assert payload["details"]["owner"] == "pokemon"
    assert issubclass(CycleDetectedError, TaskError)


# --------------------------------------------------------------------------- #
# Blind-exploration boundary for the new protocol modules
# --------------------------------------------------------------------------- #

MULTI_AGENT_FORBIDDEN_IMPORTS = ("pyboy", "llm_agent.progress")
SPOILER_WORDS = (
    "pallet town", "viridian", "pewter", "cerulean", "vermilion", "celadon", "fuchsia",
    "saffron", "cinnabar", "indigo plateau", "mt. moon", "mt moon", "rock tunnel",
    "ss anne", "s.s. anne", "lavender town", "oak's lab", "oaks lab", "professor oak",
    "brock", "misty", "lt. surge", "erika", "koga", "sabrina", "blaine", "giovanni",
    "route 1", "route 2", "route 3", "route 4", "poke mart", "gym leader", "badge",
    "pokédex", "pokedex", "bulbasaur", "charmander", "squirtle", "pikachu",
)


def _imports(path: Path) -> set[str]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            modules.add(base)
            modules.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return modules


def test_protocol_modules_respect_blind_exploration_boundaries() -> None:
    for name in ("contracts.py", "agent_context.py", "agent_registry.py", "model_router.py"):
        path = PACKAGE / "multi_agent" / name
        imports = _imports(path)
        offending = {
            module for module in imports
            if module in MULTI_AGENT_FORBIDDEN_IMPORTS
            or any(module.startswith(f"{prefix}.") for prefix in MULTI_AGENT_FORBIDDEN_IMPORTS)
        }
        assert not offending, f"{name} imports hidden state: {offending}"
        text = path.read_text(encoding="utf-8").lower()
        leaked = [word for word in SPOILER_WORDS if word in text]
        assert not leaked, f"{name} leaks game knowledge: {leaked}"


def test_no_ram_addresses_in_multi_agent_modules() -> None:
    import re

    pattern = re.compile(r"0x[0-9A-Fa-f]{4}")
    offenders = []
    for path in sorted((PACKAGE / "multi_agent").glob("*.py")):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {match.group(0)}")
    assert not offenders, f"RAM addresses leaked into multi_agent: {offenders}"
