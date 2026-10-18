"""Unit tests for the shared world knowledge base (no ROM/network/emulator).

Covers: entry validation, exact de-duplication, confidence caps, the
"only evidenced facts may be verified" rule, verify/refute/supersede, branch
rollback -> HISTORICAL marking (learning preserved), retrieval filters, strict
prompt character budgets, bounded memory, snapshot/restore and atomic
persistence including corrupt-file tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_agent.multi_agent.contracts import (
    AgentRole,
    KnowledgeEntry,
    KnowledgeError,
    VerificationStatus,
)
from llm_agent.multi_agent.shared_memory import (
    SHARED_MEMORY_FILENAME,
    SharedMemoryConfig,
    SharedWorldMemory,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class FakeClock:
    """Deterministic, manually advanced wall clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def _memory(
    clock: FakeClock | None = None,
    *,
    branch_id: str = "",
    **config_overrides,
) -> SharedWorldMemory:
    return SharedWorldMemory(
        SharedMemoryConfig(**config_overrides),
        clock=clock or FakeClock(),
        branch_id=branch_id,
    )


def _remember(
    memory: SharedWorldMemory,
    content: str = "a visible landmark",
    *,
    category: str = "area",
    **kwargs,
) -> KnowledgeEntry:
    return memory.remember(category, content, source_agent=kwargs.pop("source_agent", "navigation"), **kwargs)


# --------------------------------------------------------------------------- #
# Remember / validation / confidence
# --------------------------------------------------------------------------- #

def test_remember_stores_validated_entry() -> None:
    clock = FakeClock()
    memory = SharedWorldMemory(clock=clock, branch_id="branch_1")
    entry = _remember(
        memory,
        "a house with a red roof",
        subject="area:start",
        tags=["building", "building", "warm"],
        evidence_refs=["frame_0001"],
    )

    assert entry.knowledge_id.startswith("kn_")
    assert entry.category == "area"
    assert entry.content == "a house with a red roof"
    assert entry.source_agent is AgentRole.NAVIGATION
    assert entry.verification_status is VerificationStatus.HYPOTHESIS
    assert entry.branch_id == "branch_1"
    assert entry.tags == ["building", "warm"]
    assert entry.evidence_refs == ["frame_0001"]
    assert entry.created_at == 1000.0
    assert entry.updated_at == 1000.0
    assert memory.get(entry.knowledge_id).content == "a house with a red roof"
    assert len(memory) == 1
    assert memory.version >= 2


def test_invalid_remember_inputs_raise() -> None:
    memory = _memory()
    with pytest.raises(KnowledgeError):
        memory.remember("not-a-category", "something", source_agent="navigation")
    with pytest.raises(KnowledgeError):
        memory.remember("area", "   ", source_agent="navigation")
    with pytest.raises(KnowledgeError):
        memory.remember("area", "something", source_agent="not-an-agent")
    with pytest.raises(KnowledgeError):
        memory.remember(
            "area", "something", source_agent="navigation", verification_status="certain"
        )


def test_content_is_clean_and_bounded() -> None:
    memory = _memory(max_content_chars=16)
    entry = _remember(memory, "   a    very   long   description   here   ")
    assert entry.content == "a very long desc"
    assert len(entry.content) <= 16


def test_confidence_caps_keep_guesses_below_facts() -> None:
    memory = _memory()
    hypothesis = _remember(memory, "maybe the door opens", confidence=0.95)
    assert hypothesis.confidence <= 0.6

    inferred = _remember(
        memory,
        "probably a shop",
        verification_status=VerificationStatus.INFERRED,
        confidence=0.95,
    )
    assert inferred.confidence <= 0.85

    verified = _remember(
        memory,
        "a sign is visible",
        verification_status=VerificationStatus.VERIFIED,
        evidence_refs=["frame_0002"],
        confidence=0.4,
    )
    assert verified.verification_status is VerificationStatus.VERIFIED
    assert verified.confidence >= 0.9


def test_verified_without_evidence_is_downgraded() -> None:
    memory = _memory()
    entry = _remember(
        memory,
        "a bridge exists",
        verification_status=VerificationStatus.VERIFIED,
        evidence_refs=[],
    )
    assert entry.verification_status is VerificationStatus.INFERRED
    assert entry.confidence < 0.9
    assert memory.stats()["downgraded"] == 1


def test_verified_with_evidence_stays_verified() -> None:
    memory = _memory()
    entry = _remember(
        memory,
        "a bridge exists",
        verification_status=VerificationStatus.VERIFIED,
        evidence_refs=["frame_10"],
    )
    assert entry.verification_status is VerificationStatus.VERIFIED


# --------------------------------------------------------------------------- #
# De-duplication / merging
# --------------------------------------------------------------------------- #

def test_exact_duplicate_updates_existing_entry() -> None:
    clock = FakeClock()
    memory = SharedWorldMemory(clock=clock)
    first = _remember(memory, "the same landmark", subject="area:1", evidence_refs=["frame_1"])
    clock.advance(30)
    second = _remember(
        memory,
        "the same landmark",
        subject="area:1",
        evidence_refs=["frame_2"],
        verification_status=VerificationStatus.INFERRED,
        confidence=0.7,
    )

    assert second.knowledge_id == first.knowledge_id
    assert len(memory) == 1
    assert second.evidence_refs == ["frame_1", "frame_2"]
    assert second.updated_at == 1030.0
    assert second.verification_status is VerificationStatus.INFERRED   # upgraded
    assert second.confidence == 0.7


def test_same_content_different_subject_or_branch_is_a_new_entry() -> None:
    memory = SharedWorldMemory(branch_id="branch_1")
    first = _remember(memory, "the same landmark", subject="area:1")
    other_subject = _remember(memory, "the same landmark", subject="area:2")
    other_branch = _remember(memory, "the same landmark", subject="area:1", branch_id="branch_2")
    assert len(memory) == 3
    assert len({first.knowledge_id, other_subject.knowledge_id, other_branch.knowledge_id}) == 3


def test_add_is_idempotent_and_merges_duplicates() -> None:
    memory = _memory()
    entry = KnowledgeEntry.create(
        category="npc",
        content="a person stands near the door",
        source_agent="story",
        evidence_refs=["frame_20"],
    )
    stored = memory.add(entry)
    assert stored.knowledge_id == entry.knowledge_id
    assert len(memory) == 1

    again = memory.add(entry)
    assert again.knowledge_id == entry.knowledge_id
    assert len(memory) == 1

    duplicate = KnowledgeEntry.create(
        category="npc",
        content="a person stands near the door",
        source_agent="navigation",
        evidence_refs=["frame_21"],
    )
    merged = memory.add(duplicate)
    assert merged.knowledge_id == entry.knowledge_id
    assert len(memory) == 1
    assert merged.evidence_refs == ["frame_20", "frame_21"]


def test_add_rejects_invalid_entry() -> None:
    memory = _memory()
    broken = KnowledgeEntry.create(category="area", content="x", source_agent="navigation")
    broken.category = "not-a-category"
    with pytest.raises(KnowledgeError):
        memory.add(broken)
    with pytest.raises(KnowledgeError):
        memory.add("not an entry")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Verify / refute / supersede
# --------------------------------------------------------------------------- #

def test_verify_requires_evidence() -> None:
    memory = _memory()
    entry = _remember(memory, "a ladder leads up", confidence=0.4)
    with pytest.raises(KnowledgeError):
        memory.verify(entry.knowledge_id)

    verified = memory.verify(entry.knowledge_id, evidence_refs=["frame_30"])
    assert verified.verification_status is VerificationStatus.VERIFIED
    assert verified.confidence >= 0.9
    assert verified.evidence_refs == ["frame_30"]
    assert memory.stats()["verified"] == 1


def test_verify_merges_evidence_and_respects_confidence() -> None:
    memory = _memory()
    entry = _remember(memory, "a locked gate", evidence_refs=["frame_1"])
    verified = memory.verify(
        entry.knowledge_id,
        evidence_refs=["frame_2"],
        confidence=0.95,
    )
    assert verified.evidence_refs == ["frame_1", "frame_2"]
    assert verified.confidence == pytest.approx(0.95)
    with pytest.raises(KnowledgeError):
        memory.verify("kn_missing")


def test_refute_lowers_trust_and_hides_entry() -> None:
    memory = _memory()
    entry = _remember(memory, "the path is clear", verification_status="inferred")
    refuted = memory.refute(entry.knowledge_id, evidence_refs=["frame_40"], reason="blocked")
    assert refuted.verification_status is VerificationStatus.REFUTED
    assert refuted.confidence <= 0.2
    assert memory.search("path is clear") == []
    assert len(memory.search("path is clear", include_historical=True)) == 1
    assert memory.stats()["refuted"] == 1

    # A refuted fact does not suppress a fresh observation of the same text.
    fresh = _remember(memory, "the path is clear")
    assert fresh.knowledge_id != entry.knowledge_id


def test_supersede_hides_old_record() -> None:
    memory = _memory()
    old = _remember(memory, "the exit is to the left")
    new = _remember(memory, "the exit is to the right")
    memory.supersede(old.knowledge_id, new.knowledge_id)

    assert memory.get(old.knowledge_id).superseded_by == new.knowledge_id
    visible = {entry.knowledge_id for entry in memory.search("exit")}
    assert new.knowledge_id in visible
    assert old.knowledge_id not in visible
    with pytest.raises(KnowledgeError):
        memory.supersede(old.knowledge_id, old.knowledge_id)
    with pytest.raises(KnowledgeError):
        memory.supersede("kn_missing", new.knowledge_id)


# --------------------------------------------------------------------------- #
# Rollback / branch invalidation
# --------------------------------------------------------------------------- #

def test_set_branch_marks_previous_branch_historical() -> None:
    clock = FakeClock()
    memory = SharedWorldMemory(clock=clock, branch_id="branch_old")
    first = _remember(memory, "old branch landmark one", verification_status="verified", evidence_refs=["f1"])
    second = _remember(memory, "old branch landmark two")
    clock.advance(10)
    version_before = memory.version

    marked = memory.set_branch("branch_new", previous_branch_id="branch_old", reason="rollback")
    assert marked == 2
    assert memory.branch_id == "branch_new"
    assert memory.version > version_before

    # Learning is preserved, not deleted, and now clearly labelled historical.
    assert memory.get(first.knowledge_id).verification_status is VerificationStatus.HISTORICAL
    assert memory.get(second.knowledge_id).verification_status is VerificationStatus.HISTORICAL
    assert memory.get(first.knowledge_id).content == "old branch landmark one"

    assert memory.search("old branch landmark") == []
    historical = memory.search("old branch landmark", include_historical=True)
    assert {entry.knowledge_id for entry in historical} == {
        first.knowledge_id,
        second.knowledge_id,
    }
    rendered = memory.render_for_prompt()
    assert "HISTORICAL" in rendered
    assert memory.stats()["historical"] == 2


def test_set_branch_only_marks_the_previous_branch() -> None:
    memory = SharedWorldMemory(branch_id="branch_old")
    old_entry = _remember(memory, "from the old branch")
    new_entry = _remember(memory, "from the new branch", branch_id="branch_new")
    marked = memory.set_branch("branch_new", previous_branch_id="branch_old")
    assert marked == 1
    assert (
        memory.get(old_entry.knowledge_id).verification_status
        is VerificationStatus.HISTORICAL
    )
    assert (
        memory.get(new_entry.knowledge_id).verification_status
        is not VerificationStatus.HISTORICAL
    )


def test_cannot_verify_or_refute_historical_entry() -> None:
    memory = SharedWorldMemory(branch_id="branch_old")
    entry = _remember(memory, "old knowledge")
    memory.set_branch("branch_new", previous_branch_id="branch_old")
    with pytest.raises(KnowledgeError):
        memory.verify(entry.knowledge_id, evidence_refs=["frame"])
    with pytest.raises(KnowledgeError):
        memory.refute(entry.knowledge_id)


# --------------------------------------------------------------------------- #
# Search / retrieval
# --------------------------------------------------------------------------- #

def test_search_keywords_categories_subject_and_limits() -> None:
    clock = FakeClock()
    memory = _memory(clock, retrieval_limit=12)
    area = _remember(memory, "a tall tower is visible", category="area", subject="area:tower")
    clock.advance(5)
    npc = _remember(memory, "a keeper walks around", category="npc", subject="npc:keeper")
    clock.advance(5)
    hazard = _remember(memory, "the stairs are broken", category="hazard")

    assert {entry.knowledge_id for entry in memory.search("tower")} == {area.knowledge_id}
    assert {entry.knowledge_id for entry in memory.search("keeper")} == {npc.knowledge_id}
    assert {entry.knowledge_id for entry in memory.search("", categories=["hazard"])} == {
        hazard.knowledge_id
    }
    assert {entry.knowledge_id for entry in memory.search("", subject="npc:keeper")} == {
        npc.knowledge_id
    }
    assert len(memory.search("", limit=2)) == 2
    assert memory.search("nothing-matches-this") == []

    # Empty query returns newest first.
    ordered = memory.search("")
    assert ordered[0].knowledge_id == hazard.knowledge_id


def test_search_verified_only() -> None:
    memory = _memory()
    fact = _remember(memory, "a fact about the room", verification_status="verified", evidence_refs=["f"])
    guess = _remember(memory, "a guess about the room")
    verified_only = memory.search("", verified_only=True)
    assert {entry.knowledge_id for entry in verified_only} == {fact.knowledge_id}
    assert len(memory.search("")) == 2
    assert guess.knowledge_id != fact.knowledge_id


def test_by_category() -> None:
    memory = _memory()
    area = _remember(memory, "an area note", category="area")
    _remember(memory, "a battle note", category="battle_lesson")
    entries = memory.by_category("area")
    assert {entry.knowledge_id for entry in entries} == {area.knowledge_id}
    with pytest.raises(KnowledgeError):
        memory.by_category("not-a-category")


# --------------------------------------------------------------------------- #
# Prompt rendering budgets
# --------------------------------------------------------------------------- #

def test_render_for_prompt_has_strict_char_budget() -> None:
    memory = _memory(max_prompt_chars=260, max_content_chars=400)
    for index in range(20):
        _remember(
            memory,
            f"landmark number {index} " + "very " * 20,
            verification_status="verified",
            evidence_refs=[f"frame_{index}"],
        )
    text = memory.render_for_prompt()
    assert isinstance(text, str)
    assert 0 < len(text) <= 260
    for budget in (0, 1, 5, 20, 60, 150):
        assert len(memory.render_for_prompt(max_chars=budget)) <= max(0, budget)
    assert memory.render_for_prompt(max_chars=0) == ""


def test_render_for_prompt_groups_by_verification_status() -> None:
    memory = _memory(max_prompt_chars=4000)
    _remember(memory, "a verified thing", verification_status="verified", evidence_refs=["f1"])
    _remember(memory, "an inferred thing", verification_status="inferred")
    _remember(memory, "a hypothesised thing")
    text = memory.render_for_prompt()
    assert "FACTS" in text
    assert "INFERENCES" in text
    assert "HYPOTHESES" in text
    assert "a verified thing" in text
    assert "an inferred thing" in text
    assert "a hypothesised thing" in text
    assert "SHARED WORLD KNOWLEDGE" in text


def test_render_for_prompt_entry_cap_and_query_relevance() -> None:
    memory = _memory(max_entries_per_prompt=3, max_prompt_chars=4000)
    for index in range(10):
        _remember(memory, f"item {index}", verification_status="verified", evidence_refs=["f"])
    text = memory.render_for_prompt()
    entry_lines = [line for line in text.splitlines() if line.startswith("- [")]
    assert len(entry_lines) <= 3

    memory2 = _memory(max_prompt_chars=4000)
    _remember(memory2, "the tower is tall")
    _remember(memory2, "the river runs fast")
    query_text = memory2.render_for_prompt("tower")
    assert "tower" in query_text
    assert "river" not in query_text


def test_render_agent_brief_is_role_focused_and_bounded() -> None:
    memory = _memory(max_prompt_chars=4000)
    area = _remember(memory, "a wide room", category="area", verification_status="verified", evidence_refs=["f"])
    _remember(memory, "a battle lesson", category="battle_lesson")
    brief = memory.render_agent_brief("navigation", max_chars=600)
    assert len(brief) <= 600
    assert "navigation" in brief
    assert "a wide room" in brief
    assert brief.index("a wide room") < brief.index("a battle lesson")

    assert memory.render_agent_brief("story", max_chars=0) == ""
    empty = _memory()
    assert empty.render_agent_brief("navigation") == ""


# --------------------------------------------------------------------------- #
# Bounded memory / version
# --------------------------------------------------------------------------- #

def test_max_entries_evicts_least_valuable_entries() -> None:
    memory = _memory(max_entries=6)
    verified_ids = [
        _remember(
            memory,
            f"verified fact {index}",
            verification_status="verified",
            evidence_refs=["frame"],
        ).knowledge_id
        for index in range(5)
    ]
    for index in range(3):
        _remember(memory, f"guess {index}")
    assert len(memory) == 6
    assert memory.stats()["verified"] == 5
    assert memory.stats()["evicted"] == 2
    for knowledge_id in verified_ids:
        assert memory.get(knowledge_id) is not None


def test_version_increases_on_every_mutation() -> None:
    memory = _memory()
    versions = [memory.version]
    entry = _remember(memory, "first")
    versions.append(memory.version)
    _remember(memory, "first")                       # duplicate merge
    versions.append(memory.version)
    memory.remember("system", "note", source_agent="system", verification_status="inferred")
    versions.append(memory.version)
    second = _remember(memory, "second")
    versions.append(memory.version)
    memory.verify(second.knowledge_id, evidence_refs=["frame"])
    versions.append(memory.version)
    memory.refute(entry.knowledge_id, reason="wrong")
    versions.append(memory.version)
    memory.set_branch("branch_next", previous_branch_id="")
    versions.append(memory.version)
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


# --------------------------------------------------------------------------- #
# Snapshot / restore / persistence
# --------------------------------------------------------------------------- #

def test_snapshot_restore_roundtrip() -> None:
    clock = FakeClock()
    memory = SharedWorldMemory(clock=clock, branch_id="branch_a")
    fact = _remember(memory, "a fact", verification_status="verified", evidence_refs=["f1"])
    guess = _remember(memory, "a guess")
    memory.refute(guess.knowledge_id, reason="no")
    old = _remember(memory, "old branch knowledge")
    memory.set_branch("branch_b", previous_branch_id="branch_a")
    fresh = _remember(memory, "fresh branch knowledge")

    snapshot = memory.snapshot()
    restored = SharedWorldMemory()
    restored.restore(snapshot)

    assert restored.version == memory.version
    assert restored.branch_id == "branch_b"
    assert len(restored) == len(memory)
    # Everything created on the rolled-back branch is now clearly historical.
    assert (
        restored.get(fact.knowledge_id).verification_status
        is VerificationStatus.HISTORICAL
    )
    assert (
        restored.get(guess.knowledge_id).verification_status
        is VerificationStatus.HISTORICAL
    )
    assert (
        restored.get(old.knowledge_id).verification_status
        is VerificationStatus.HISTORICAL
    )
    assert (
        restored.get(fresh.knowledge_id).verification_status
        is not VerificationStatus.HISTORICAL
    )

    # The duplicate index is rebuilt: repeating the fresh observation merges.
    again = _remember(restored, "fresh branch knowledge")
    assert again.knowledge_id == fresh.knowledge_id


def test_restore_is_tolerant_to_bad_records() -> None:
    memory = _memory()
    good = _remember(memory, "good record")
    snapshot = memory.snapshot()
    snapshot["entries"].append({"knowledge_id": "broken"})
    snapshot["entries"].append({"not": "a record"})

    restored = SharedWorldMemory()
    restored.restore(snapshot)
    assert restored.get(good.knowledge_id) is not None
    assert restored.load_errors


def test_restore_keeps_explicit_branch() -> None:
    source = SharedWorldMemory(branch_id="branch_old")
    entry = _remember(source, "old knowledge")
    snapshot = source.snapshot()

    target = SharedWorldMemory(branch_id="branch_new")
    target.restore(snapshot)
    assert target.branch_id == "branch_new"
    assert target.get(entry.knowledge_id).branch_id == "branch_old"


def test_atomic_persistence_roundtrip(tmp_path: Path) -> None:
    clock = FakeClock()
    memory = SharedWorldMemory(
        SharedMemoryConfig(root=str(tmp_path)), clock=clock, branch_id="branch_p"
    )
    entry = _remember(memory, "persisted knowledge", evidence_refs=["frame_1"])
    memory.save()

    path = tmp_path / SHARED_MEMORY_FILENAME
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["entries"][0]["knowledge_id"] == entry.knowledge_id

    loaded = SharedWorldMemory(SharedMemoryConfig(root=str(tmp_path)), clock=clock)
    assert loaded.load() is True
    assert loaded.get(entry.knowledge_id).content == "persisted knowledge"
    # Auto-load on construction also works and keeps the explicit branch.
    auto = SharedWorldMemory(
        SharedMemoryConfig(root=str(tmp_path)), clock=clock, branch_id="branch_q"
    )
    assert len(auto) == 1
    assert auto.branch_id == "branch_q"


def test_no_root_means_persistence_disabled() -> None:
    memory = _memory()
    memory.save()
    assert memory.load() is False
    assert memory.path is None


def test_corrupt_file_is_backed_up(tmp_path: Path) -> None:
    path = tmp_path / SHARED_MEMORY_FILENAME
    path.write_text("not json at all", encoding="utf-8")
    memory = SharedWorldMemory(SharedMemoryConfig(root=str(tmp_path)))

    assert len(memory) == 0
    assert memory.load_errors
    assert memory.recovered_from is not None
    assert Path(memory.recovered_from).is_file()
    assert not path.exists()
    memory.save()
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["entries"] == []


def test_stats_reports_epistemic_breakdown() -> None:
    memory = _memory()
    _remember(memory, "fact", verification_status="verified", evidence_refs=["f"])
    _remember(memory, "guess")
    refuted = _remember(memory, "wrong")
    memory.refute(refuted.knowledge_id)
    stats = memory.stats()
    assert stats["entries"] == 3
    assert stats["verified"] == 1
    assert stats["hypotheses"] == 1
    assert stats["refuted"] == 1
    assert stats["by_status"]["verified"] == 1
    assert stats["by_category"]["area"] == 3
    assert stats["branch_id"] == ""
    assert stats["dedupe_keys"] >= 1
    assert stats["audit_events"] >= 1
