"""StoryMemory tests: honesty gate, shared mirroring, proposals, persistence."""

from __future__ import annotations

import json
from dataclasses import replace

from llm_agent.config import MemoryConfig
from llm_agent.memory import AgentMemory
from llm_agent.multi_agent.contracts import AgentRole, VerificationStatus
from llm_agent.story.memory import (
    STORY_MEMORY_FILENAME,
    STORY_PROPOSAL_OWNERS,
    STORY_PROPOSAL_TYPES,
    SharedMemoryAdapter,
    StoryMemory,
)

from tests.story.conftest import FakeSharedMemory


def _agent_memory(root) -> AgentMemory:
    return AgentMemory(replace(MemoryConfig(root=str(root)), max_context_chars=2000))


def test_verified_claim_without_evidence_is_downgraded(tmp_path) -> None:
    memory = StoryMemory(tmp_path / "memory", agent_memory=_agent_memory(tmp_path / "memory"))
    result = memory.record_knowledge(
        category="story_clue",
        content="Someone wants help with a lost thing.",
        verification_status="verified",
        confidence=0.99,
    )
    assert result["downgraded"] is True
    assert result["entry"]["verification_status"] == "inferred"
    assert result["entry"]["confidence"] <= 0.85


def test_verified_claim_with_observed_dialogue_stays_verified(tmp_path) -> None:
    memory = StoryMemory(tmp_path / "memory", agent_memory=_agent_memory(tmp_path / "memory"))
    memory.record_dialogue_text(
        "a resident",
        "I lost something near the water.",
        evidence_ref="screens/story_00000010.png",
        frame_index=10,
    )
    result = memory.record_knowledge(
        category="story_clue",
        content="A resident lost an object near water.",
        verification_status="verified",
        confidence=0.9,
        evidence_refs=["screens/story_00000010.png"],
        npc_name="a resident",
    )
    assert result["downgraded"] is False
    assert result["entry"]["verification_status"] == "verified"
    assert result["entry"]["evidence_refs"] == ["screens/story_00000010.png"]
    assert memory.has_observed_dialogue() is True


def test_inferred_and_hypothesis_never_become_facts(tmp_path) -> None:
    memory = StoryMemory(tmp_path / "memory", agent_memory=_agent_memory(tmp_path / "memory"))
    memory.record_dialogue_text("a person", "The path is closed today.", evidence_ref="frame:1")
    memory.record_knowledge(
        category="quest",
        content="Maybe the path will open later.",
        verification_status="hypothesis",
        evidence_refs=["frame:1"],
    )
    statuses = {entry.verification_status for entry in memory.knowledge()}
    assert statuses == {VerificationStatus.HYPOTHESIS}


def test_shared_memory_mirror_uses_frozen_interface(tmp_path) -> None:
    shared = FakeSharedMemory()
    memory = StoryMemory(
        tmp_path / "memory",
        agent_memory=_agent_memory(tmp_path / "memory"),
        shared_memory=shared,
    )
    memory.record_knowledge(
        category="npc",
        content="A person standing near the entrance nodded.",
        verification_status="inferred",
        confidence=0.5,
        subject="a person",
    )
    assert len(shared.entries) == 1
    entry = shared.entries[0]
    assert entry.source_agent is AgentRole.STORY
    assert entry.verification_status is VerificationStatus.INFERRED
    assert entry.subject == "a person"
    assert memory.stats()["shared_memory"] is True
    assert memory.stats()["shared_errors"] == []


def test_shared_memory_failures_never_crash_the_agent(tmp_path) -> None:
    class BrokenShared:
        def remember(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("shared store exploded")

    memory = StoryMemory(
        tmp_path / "memory",
        agent_memory=_agent_memory(tmp_path / "memory"),
        shared_memory=BrokenShared(),
    )
    result = memory.record_knowledge(
        category="other",
        content="An observation kept locally.",
        verification_status="inferred",
    )
    assert result["created"] is True
    assert memory.stats()["shared_errors"], "the shared-memory error must be reported"


def test_proposals_are_deduplicated_and_validated(tmp_path) -> None:
    memory = StoryMemory(tmp_path / "memory", agent_memory=_agent_memory(tmp_path / "memory"))
    proposal, created = memory.add_proposal(
        owner_agent="navigation",
        task_type="talk",
        description="Talk again with the person near the door",
        priority="low",
        payload={"target": {"kind": "described", "description": "near the door"}},
    )
    assert created is True
    again, created_again = memory.add_proposal(
        owner_agent="navigation",
        task_type="talk",
        description="Talk again with the person near the door",
        priority="low",
    )
    assert created_again is False
    assert again["proposal_id"] == proposal["proposal_id"]
    assert len(memory.proposals()) == 1
    bad, bad_created = memory.add_proposal(
        owner_agent="director", task_type="talk", description="not allowed"
    )
    assert bad_created is False and bad == {}
    assert set(STORY_PROPOSAL_OWNERS) == {"navigation", "pokemon", "story"}
    assert "talk" in STORY_PROPOSAL_TYPES


def test_proposal_public_view_has_frozen_keys(tmp_path) -> None:
    memory = StoryMemory(tmp_path / "memory", agent_memory=_agent_memory(tmp_path / "memory"))
    memory.add_proposal(
        owner_agent="navigation",
        task_type="explore",
        description="Look around the current area",
        priority="normal",
        payload={"reason": "nothing learned yet"},
    )
    public = memory.proposals()[0]
    assert set(public) == {"owner_agent", "task_type", "description", "priority", "payload"}


def test_npc_records_merge_and_verify(tmp_path) -> None:
    memory = StoryMemory(tmp_path / "memory", agent_memory=_agent_memory(tmp_path / "memory"))
    memory.record_npc("the shopkeeper", notes="stands behind a counter", verified=False)
    memory.record_npc("the shopkeeper", notes="mentioned a delivery", verified=True)
    npcs = memory.npcs()
    assert len(npcs) == 1
    assert npcs[0]["verified"] is True
    assert "delivery" in npcs[0]["notes"]


def test_memory_persists_across_instances(tmp_path) -> None:
    root = tmp_path / "memory"
    first = StoryMemory(root, agent_memory=_agent_memory(root))
    first.record_dialogue_text(
        "a resident", "The gate is locked.", evidence_ref="screens/x.png", frame_index=7
    )
    first.record_knowledge(
        category="story_clue",
        content="The gate is locked.",
        verification_status="verified",
        evidence_refs=["screens/x.png"],
        npc_name="a resident",
    )
    first.record_npc("a resident", notes="mentioned a locked gate", verified=True)
    first.add_proposal(
        owner_agent="navigation",
        task_type="investigate",
        description="Investigate the locked gate",
        payload={"target": {"kind": "described", "description": "the locked gate"}},
    )
    first.save()
    assert (root / STORY_MEMORY_FILENAME).is_file()

    second = StoryMemory(root, agent_memory=_agent_memory(root))
    assert len(second.knowledge()) == 1
    assert second.knowledge()[0].verification_status is VerificationStatus.VERIFIED
    assert second.npcs()[0]["name"] == "a resident"
    assert len(second.proposals()) == 1
    assert second.has_observed_dialogue() is True
    # The private notebook (AgentMemory) was persisted as well.
    assert any("gate" in str(item.get("text", "")).lower() for item in second.agent_memory.search("gate"))


def test_corrupt_sidecar_does_not_crash(tmp_path) -> None:
    root = tmp_path / "memory"
    root.mkdir(parents=True, exist_ok=True)
    (root / STORY_MEMORY_FILENAME).write_text("{not valid json", encoding="utf-8")
    memory = StoryMemory(root, agent_memory=_agent_memory(root))
    assert memory.load_errors
    memory.record_knowledge(
        category="other", content="Still works after a corrupt file.", verification_status="hypothesis"
    )
    assert json.loads((root / STORY_MEMORY_FILENAME).read_text(encoding="utf-8"))

    bad_version = tmp_path / "memory2"
    bad_version.mkdir(parents=True, exist_ok=True)
    (bad_version / STORY_MEMORY_FILENAME).write_text(
        json.dumps({"version": 99, "knowledge": []}), encoding="utf-8"
    )
    memory2 = StoryMemory(bad_version, agent_memory=_agent_memory(bad_version))
    assert memory2.load_errors


def test_render_context_is_bounded_and_prioritises_verified(tmp_path) -> None:
    memory = StoryMemory(tmp_path / "memory", agent_memory=_agent_memory(tmp_path / "memory"))
    memory.record_dialogue_text("a person", "A visible line of text.", evidence_ref="frame:3")
    memory.record_knowledge(
        category="story_clue",
        content="Important verified clue that must be shown to the model.",
        verification_status="verified",
        evidence_refs=["frame:3"],
    )
    memory.record_knowledge(
        category="other",
        content="An unconfirmed idea about the room.",
        verification_status="hypothesis",
    )
    text = memory.render_context(max_chars=240)
    assert len(text) <= 240
    # Small budget: verified facts must win over hypotheses.
    assert "Important verified clue" in text or "CONFIRMED" in text


def test_adapter_falls_back_to_legacy_add_method() -> None:
    class LegacyShared:
        def __init__(self) -> None:
            self.entries = []

        def add(self, entry):  # noqa: ANN001
            self.entries.append(entry)
            return entry

    legacy = LegacyShared()
    adapter = SharedMemoryAdapter(legacy)
    adapter.remember(category="other", content="legacy add works", verification_status="inferred")
    assert len(legacy.entries) == 1
    assert legacy.entries[0].content == "legacy add works"


def test_snapshot_restore_roundtrip(tmp_path) -> None:
    memory = StoryMemory(tmp_path / "memory", agent_memory=_agent_memory(tmp_path / "memory"))
    memory.record_knowledge(
        category="other", content="snapshot me", verification_status="hypothesis"
    )
    snapshot = memory.snapshot()
    memory.record_knowledge(
        category="other", content="not in snapshot", verification_status="hypothesis"
    )
    assert len(memory.knowledge()) == 2
    memory.restore(snapshot)
    assert [entry.content for entry in memory.knowledge()] == ["snapshot me"]
