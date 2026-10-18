"""Unit tests for :mod:`llm_agent.memory` (owner: memory-guard).

Everything runs against a temporary memory root; no ROM, network or API key is
needed.  The tests pin the public contract used by the runtime/tools:
persistence roundtrip after re-instantiation, retrieval relevance,
``render_context`` determinism + truncation, compression caps/summarizer, open
questions, snapshot/restore, corrupt-file recovery, journal trimming and stats.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_agent.config import MemoryConfig
from llm_agent.memory import AgentMemory


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _populate(memory: AgentMemory) -> None:
    """Fill a memory with a representative, observation-only document."""
    memory.set_goal("Explore Route 1 north", "Find the exit from Pallet Town")
    memory.note_place("Pallet Town", "Home town with Oak's lab")
    memory.note_place("Route 1", "Tall grass on the way north")
    memory.record_discovery(
        "Tall grass on Route 1 can hide wild Pokémon", kind="mechanic", place="Route 1"
    )
    memory.record_discovery(
        "Oak's lab has three Poké Balls on the table", kind="item", place="Pallet Town"
    )
    memory.record_dialogue("Mom", "All boys leave home someday. It said so on TV.", place="Pallet Town")
    memory.record_lesson("Pressing A repeatedly can advance text boxes", success=True)
    memory.add_open_question("Where does the north exit of Viridian City lead?")


# --------------------------------------------------------------------------- #
# Basics / persistence
# --------------------------------------------------------------------------- #

def test_fresh_memory_starts_empty(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    assert memory.get_goal() == {"text": "", "reason": "", "at": 0.0}
    assert memory.stats()["places"] == 0
    assert memory.stats()["discoveries"] == 0
    assert memory.recent_journal() == []
    assert memory.render_context()  # never empty: goal placeholder is included
    assert Path(memory_config.root).is_dir()
    assert (Path(memory_config.root) / ".gitignore").is_file()


def test_persistence_roundtrip_after_reinstantiation(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    memory.save()

    reloaded = AgentMemory(memory_config)
    reloaded.load()
    assert reloaded.get_goal() == memory.get_goal()
    assert reloaded.stats() == memory.stats()
    assert reloaded.render_context() == memory.render_context()
    assert "Tall grass" in reloaded.render_context()
    # journal survived the roundtrip
    assert len(reloaded.recent_journal(100)) == len(memory.recent_journal(100))
    assert reloaded.search("three Poké Balls")


def test_journal_and_report_files(memory_config: MemoryConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    memory = AgentMemory(memory_config, run_dir=run_dir)
    _populate(memory)
    memory.save()

    journal_lines = [
        line for line in (Path(memory_config.root) / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(journal_lines) == len(memory.recent_journal(1000))
    for line in journal_lines:
        assert isinstance(json.loads(line), dict)

    report = run_dir / "reports" / "memory_latest.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "# Agent Memory Report" in text
    assert "Explore Route 1 north" in text

    # atomic writes leave no temporary files behind
    assert list(Path(memory_config.root).rglob("*.tmp")) == []
    assert list(run_dir.rglob("*.tmp")) == []


def test_journal_trimming(memory_config: MemoryConfig, tmp_path: Path) -> None:
    cfg = MemoryConfig(root=str(tmp_path / "mem"), max_journal=5)
    memory = AgentMemory(cfg)
    for index in range(20):
        memory.record_event("note", f"event {index}")
    memory.save()

    lines = [
        line for line in (Path(cfg.root) / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 5
    texts = [json.loads(line)["text"] for line in lines]
    assert texts[-1] == "event 19"
    assert "event 0" not in texts


def test_journal_appends_across_save_load(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.record_event("note", "first")
    memory.record_event("note", "second")
    memory.save()
    first = AgentMemory(memory_config)
    assert [entry["text"] for entry in first.recent_journal(10)] == ["first", "second"]
    first.record_event("note", "third")
    first.save()
    lines = [
        line for line in (Path(memory_config.root) / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [json.loads(line)["text"] for line in lines] == ["first", "second", "third"]


def test_load_is_idempotent(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    memory.save()
    before = memory.render_context()
    memory.load()
    assert memory.render_context() == before
    assert memory.stats()["discoveries"] == 2


# --------------------------------------------------------------------------- #
# Corrupt-file recovery
# --------------------------------------------------------------------------- #

def test_corrupt_memory_json_recovery(memory_config: MemoryConfig) -> None:
    root = Path(memory_config.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.json").write_text("{ this is not json", encoding="utf-8")
    (root / "journal.jsonl").write_text(
        json.dumps({"at": 1.0, "kind": "event", "text": "kept journal line"}) + "\n",
        encoding="utf-8",
    )

    memory = AgentMemory(memory_config)

    assert memory.get_goal()["text"] == ""
    assert memory.stats()["discoveries"] == 0
    assert memory.load_errors
    assert memory.recovered_from
    backups = list(root.glob("memory.corrupt-*.json"))
    assert len(backups) == 1
    assert "not json" in backups[0].read_text(encoding="utf-8")
    # journal is preserved and a recovery event was recorded
    journal_text = "\n".join(entry["text"] for entry in memory.recent_journal(20))
    assert "kept journal line" in journal_text
    assert any(entry["kind"] == "recovery" for entry in memory.recent_journal(20))
    # a fresh valid memory.json now exists on disk; reload does not re-backup
    assert isinstance(json.loads((root / "memory.json").read_text(encoding="utf-8")), dict)
    memory.load()
    assert len(list(root.glob("memory.corrupt-*.json"))) == 1


def test_unsupported_memory_version_recovers(memory_config: MemoryConfig) -> None:
    root = Path(memory_config.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.json").write_text(
        json.dumps({"version": 999, "goal": {"text": "should not survive"}}), encoding="utf-8"
    )
    memory = AgentMemory(memory_config)
    assert memory.get_goal()["text"] == ""
    assert len(list(root.glob("memory.corrupt-*.json"))) == 1


def test_corrupt_journal_lines_are_skipped(memory_config: MemoryConfig) -> None:
    root = Path(memory_config.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "journal.jsonl").write_text(
        "this is not json\n"
        + json.dumps({"at": 2.0, "kind": "event", "text": "good line"}) + "\n"
        + "[1, 2, 3]\n",
        encoding="utf-8",
    )
    memory = AgentMemory(memory_config)
    assert memory.journal_skipped == 2
    assert [entry["text"] for entry in memory.recent_journal(10)] == ["good line"]
    memory.save()
    lines = [
        line for line in (root / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "good line"


def test_malformed_entries_are_dropped(memory_config: MemoryConfig) -> None:
    root = Path(memory_config.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.json").write_text(
        json.dumps(
            {
                "version": 1,
                "goal": {"text": "Keep me"},
                "discoveries": ["junk", {"text": ""}, {"text": "valid fact", "kind": "ok"}],
                "places": {"not": "a list"},
                "dialogues": None,
                "lessons": [{"text": "valid lesson", "success": "not-a-bool"}],
            }
        ),
        encoding="utf-8",
    )
    memory = AgentMemory(memory_config)
    assert memory.get_goal()["text"] == "Keep me"
    assert memory.stats()["discoveries"] == 1
    assert memory.stats()["places"] == 0
    assert memory.stats()["dialogues"] == 0
    snapshot = memory.snapshot()["data"]
    assert snapshot["lessons"][0]["success"] is None


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def test_search_relevance_and_determinism(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.record_discovery("Tall grass on Route 1 can hide wild Pokémon", kind="mechanic", place="Route 1")
    memory.record_discovery("Oak's lab has three Poké Balls on the table", kind="item", place="Pallet Town")
    memory.record_discovery("An old man blocks the north exit of Viridian City", kind="npc", place="Viridian City")
    memory.record_dialogue("Mom", "All boys leave home someday")

    results = memory.search("tall grass", limit=5)
    assert results
    assert "tall grass" in results[0]["text"].lower()
    assert results[0]["category"] == "discoveries"
    assert results == memory.search("tall grass", limit=5)

    assert any(item["category"] == "dialogues" for item in memory.search("boys leave home"))
    assert len(memory.search("", limit=2)) == 2
    assert memory.search("no-such-keyword-anywhere") == []


def test_render_context_contains_all_sections(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    text = memory.render_context()
    assert "Goal: Explore Route 1 north" in text
    assert "Pallet Town" in text
    assert "Tall grass" in text
    assert "Mom" in text
    assert "Pressing A" in text
    assert "Open questions:" in text
    # deterministic for identical state
    assert text == memory.render_context()


def test_render_context_is_query_biased(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.set_goal("Explore Pewter City")
    memory.record_discovery(
        "The museum entrance in Pewter City needs a ticket", kind="place", place="Pewter City"
    )
    for index in range(12):
        memory.record_discovery(
            f"filler observation number {index} about rocks and trees near the path",
            kind="other",
            place="Route 1",
        )
    without_query = memory.render_context(max_chars=1600).lower()
    with_query = memory.render_context("museum ticket", max_chars=1600).lower()
    assert "museum" not in without_query
    assert "museum" in with_query


def test_render_context_truncation_keeps_goal_and_recent(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.set_goal("Explore the tall grass north of Pallet Town carefully")
    for index in range(25):
        memory.record_discovery(
            f"discovery {index}: " + "x" * 100, kind="other", place="Route 1"
        )
    for max_chars in (200, 400, 800):
        text = memory.render_context(max_chars=max_chars)
        assert len(text) <= max_chars
        assert "Goal: Explore the tall grass" in text
    text = memory.render_context(max_chars=400)
    assert "discovery 24" in text  # most recent discovery always retained


def test_render_context_tiny_budget(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.set_goal("Explore northwards from Pallet Town")
    memory.record_discovery("A long discovery that will not fit in a tiny budget " * 3)
    text = memory.render_context(max_chars=40)
    assert 0 < len(text) <= 40
    assert "Goal" in text


def test_render_context_is_deterministic_after_reload(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    memory.save()
    reloaded = AgentMemory(memory_config)
    assert reloaded.render_context() == memory.render_context()
    assert (
        reloaded.render_context("wild Pokémon", max_chars=500)
        == memory.render_context("wild Pokémon", max_chars=500)
    )


# --------------------------------------------------------------------------- #
# Dedupe / editing semantics
# --------------------------------------------------------------------------- #

def test_duplicate_discoveries_dedupe(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.record_discovery("Same fact", kind="mechanic")
    memory.record_discovery("Same fact", kind="mechanic")
    assert memory.stats()["discoveries"] == 1
    memory.record_discovery("Same fact", kind="other")
    assert memory.stats()["discoveries"] == 2


def test_note_place_updates_instead_of_duplicating(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    first = memory.note_place("Route 1", "Tall grass")
    second = memory.note_place("Route 1", "Tall grass and ledges")
    assert memory.stats()["places"] == 1
    assert second["visits"] == 2
    assert second["first_seen"] == first["first_seen"]
    assert second["last_seen"] >= first["last_seen"]
    assert second["description"] == "Tall grass and ledges"


def test_goal_history_and_reset(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.set_goal("First goal", "reason one")
    memory.set_goal("Second goal", "reason two")
    data = memory.snapshot()["data"]
    assert data["goal"]["text"] == "Second goal"
    assert [item["text"] for item in data["goal_history"]] == ["First goal"]
    # identical goal refreshes reason without appending history
    memory.set_goal("Second goal", "updated reason")
    assert len(memory.snapshot()["data"]["goal_history"]) == 1
    assert memory.get_goal()["reason"] == "updated reason"


def test_entry_length_clipping(memory_config: MemoryConfig, tmp_path: Path) -> None:
    cfg = MemoryConfig(root=str(tmp_path / "mem"), max_entry_chars=50)
    memory = AgentMemory(cfg)
    entry = memory.record_discovery("word " * 200)
    assert 0 < len(entry["text"]) <= 50
    goal = memory.set_goal("g" * 500)
    assert len(goal["text"]) <= 300


def test_unicode_content_roundtrip(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.set_goal("探索常磐森林")
    memory.record_discovery("ニビシティの北口は閉ざされている", place="ニビシティ")
    memory.save()
    reloaded = AgentMemory(memory_config)
    assert reloaded.get_goal()["text"] == "探索常磐森林"
    assert "ニビシティ" in reloaded.render_context()


def test_open_questions_helpers(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.add_open_question("Where is the exit to Route 2?")
    assert memory.stats()["open_questions"] == 1
    memory.add_open_question("Where is the exit to Route 2?")
    assert memory.stats()["open_questions"] == 1
    assert memory.resolve_open_question("Route 2")
    assert memory.stats()["open_questions"] == 0
    assert memory.resolve_open_question("missing") is False


# --------------------------------------------------------------------------- #
# Compression
# --------------------------------------------------------------------------- #

def _compression_memory(tmp_path: Path) -> AgentMemory:
    cfg = MemoryConfig(
        root=str(tmp_path / "mem"),
        max_places=3,
        max_discoveries=4,
        max_dialogue=2,
        max_lessons=2,
        max_journal=100,
        compress_every_actions=10,
    )
    return AgentMemory(cfg)


def test_compression_caps_and_goal_preserved(tmp_path: Path) -> None:
    memory = _compression_memory(tmp_path)
    memory.set_goal("Explore everywhere")
    for index in range(7):
        memory.note_place(f"Place {index}", f"description {index}")
    for index in range(9):
        memory.record_discovery(f"Discovery {index}", kind="other", place=f"Place {index}")
    for index in range(5):
        memory.record_dialogue(f"NPC {index}", f"Dialogue text {index}")
    for index in range(5):
        memory.record_lesson(f"Lesson {index}", success=index % 2 == 0)

    assert memory.maybe_compress() is True

    stats = memory.stats()
    assert (stats["places"], stats["discoveries"], stats["dialogues"], stats["lessons"]) == (3, 4, 2, 2)
    assert memory.get_goal()["text"] == "Explore everywhere"
    data = memory.snapshot()["data"]
    assert data["archived_summary"]
    assert [item["text"] for item in data["discoveries"]][-1] == "Discovery 8"
    # place names are archived verbatim, never dropped
    for name in ("Place 0", "Place 1", "Place 2", "Place 3"):
        assert name in data["archived_summary"]
    # compression is persisted
    reloaded = AgentMemory(MemoryConfig(root=str(tmp_path / "mem")))
    assert reloaded.stats()["discoveries"] == 4


def test_compression_force_calls_summarizer(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_summarizer(items: list[str]) -> str:
        calls.append(list(items))
        return f"SUMMARY OF {len(items)} OLD NOTES"

    cfg = MemoryConfig(root=str(tmp_path / "mem"), max_discoveries=3, compress_every_actions=100)
    memory = AgentMemory(cfg, summarizer=fake_summarizer)
    memory.set_goal("Keep the goal")
    memory.note_place("Pallet Town")
    for index in range(8):
        memory.record_discovery(f"old discovery {index}")

    assert memory.maybe_compress(force=True) is True
    assert calls and all(isinstance(item, str) for item in calls[0])
    data = memory.snapshot()["data"]
    assert "SUMMARY OF 5 OLD NOTES" in data["archived_summary"]
    assert memory.get_goal()["text"] == "Keep the goal"
    assert any(place["name"] == "Pallet Town" for place in data["places"])


def test_force_compress_distills_even_under_caps(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_summarizer(items: list[str]) -> str:
        calls.append(list(items))
        return "DISTILLED NOTES"

    cfg = MemoryConfig(
        root=str(tmp_path / "mem"),
        max_lessons=100,
        max_discoveries=100,
        compress_every_actions=999,
    )
    memory = AgentMemory(cfg, summarizer=fake_summarizer)
    for index in range(8):
        memory.record_lesson(f"lesson {index}", success=True)

    assert memory.maybe_compress(force=True) is True
    assert calls
    data = memory.snapshot()["data"]
    assert "DISTILLED NOTES" in data["archived_summary"]
    assert len(data["lessons"]) == 6  # oldest quarter distilled away


def test_compress_noop_when_under_caps_without_force(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.record_discovery("one fact")
    assert memory.maybe_compress() is False
    assert memory.maybe_compress(actions_since=5) is False
    assert memory.actions_since_compress == 5
    assert memory.maybe_compress(actions_since=115) is False  # threshold reached, nothing to fold
    assert memory.actions_since_compress == 0
    assert memory.stats()["discoveries"] == 1


def test_maybe_compress_actions_threshold(tmp_path: Path) -> None:
    cfg = MemoryConfig(root=str(tmp_path / "mem"), max_discoveries=3, compress_every_actions=10)
    memory = AgentMemory(cfg)
    for index in range(5):
        memory.record_discovery(f"fact {index}")
    assert memory.maybe_compress(actions_since=3) is True  # overflow triggers it
    assert memory.stats()["discoveries"] == 3


def test_place_names_never_lost_on_compression(tmp_path: Path) -> None:
    cfg = MemoryConfig(root=str(tmp_path / "mem"), max_places=2)
    memory = AgentMemory(cfg)
    names = [f"Area {index} Town" for index in range(60)]
    for name in names:
        memory.note_place(name, "a place")
    assert memory.maybe_compress(force=True) is True
    data = memory.snapshot()["data"]
    active = [place["name"] for place in data["places"]]
    archived = data["archived_summary"]
    for name in names:
        assert name in active or name in archived


def test_open_questions_fold_into_archive(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    for index in range(55):
        memory.add_open_question(f"question number {index} about something unclear")
    memory.save()
    data = memory.snapshot()["data"]
    active = [entry["text"] for entry in data["open_questions"]]
    assert len(active) == 50
    assert "question number 0 about something unclear" in data["archived_summary"]
    assert "question number 54 about something unclear" in active
    assert memory.stats()["open_questions"] == 50


def test_compressed_memory_still_renders(tmp_path: Path) -> None:
    memory = _compression_memory(tmp_path)
    memory.set_goal("Explore everywhere")
    for index in range(9):
        memory.record_discovery(f"Discovery {index}")
    memory.maybe_compress(force=True)
    text = memory.render_context(max_chars=4000)
    assert "Goal: Explore everywhere" in text
    assert "Earlier notes:" in text


# --------------------------------------------------------------------------- #
# Snapshot / restore
# --------------------------------------------------------------------------- #

def test_snapshot_restore_roundtrip(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    snapshot = memory.snapshot()
    render_before = memory.render_context()
    stats_before = memory.stats()

    memory.set_goal("A different goal")
    memory.record_discovery("an extra fact recorded after the snapshot")
    memory.add_open_question("An extra question?")
    assert memory.render_context() != render_before

    memory.restore(snapshot)
    assert memory.get_goal()["text"] == "Explore Route 1 north"
    # the append-only journal is intentionally not rewound by a restore
    comparable = {key: value for key, value in memory.stats().items() if key != "journal_entries"}
    assert comparable == {key: value for key, value in stats_before.items() if key != "journal_entries"}
    assert memory.render_context() == render_before

    # snapshot is a deep copy: mutating it does not touch live memory
    snapshot["data"]["goal"]["text"] = "MUTATED"
    assert memory.get_goal()["text"] == "Explore Route 1 north"


def test_snapshot_restore_survives_save_reload(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    snapshot = memory.snapshot()
    render_before = memory.render_context()
    memory.set_goal("Scratch goal")
    memory.restore(snapshot)
    memory.save()
    reloaded = AgentMemory(memory_config)
    assert reloaded.render_context() == render_before
    assert reloaded.get_goal()["text"] == "Explore Route 1 north"


def test_restore_tolerates_junk(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    memory.restore({"data": {"goal": "not a mapping", "discoveries": "nope"}})
    assert memory.get_goal()["text"] == ""
    assert memory.stats()["discoveries"] == 0
    memory.restore("not a mapping")  # type: ignore[arg-type] - defensive contract
    assert memory.stats()["discoveries"] == 0


# --------------------------------------------------------------------------- #
# Reports / stats / misc
# --------------------------------------------------------------------------- #

def test_stats_contents(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    stats = memory.stats()
    for key in ("places", "discoveries", "dialogues", "lessons", "goal"):
        assert key in stats
    assert stats["places"] == 2
    assert stats["discoveries"] == 2
    assert stats["dialogues"] == 1
    assert stats["lessons"] == 1
    assert stats["open_questions"] == 1
    assert stats["goal"] == "Explore Route 1 north"


def test_save_report_to_explicit_path(memory_config: MemoryConfig, tmp_path: Path) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    target = tmp_path / "custom" / "report.md"
    memory.save_report(target)
    assert target.is_file()
    assert memory.last_report_path == target
    text = target.read_text(encoding="utf-8")
    assert "# Agent Memory Report" in text
    assert "## Goal" in text
    assert "## Places (2)" in text
    assert "## Discoveries (2)" in text


def test_mirror_to_run_disabled(tmp_path: Path) -> None:
    cfg = MemoryConfig(root=str(tmp_path / "mem"), mirror_to_run=False)
    run_dir = tmp_path / "run"
    memory = AgentMemory(cfg, run_dir=run_dir)
    memory.set_goal("goal")
    memory.save()
    assert not (run_dir / "reports" / "memory_latest.md").exists()


def test_report_mirror_in_memory_root_without_run_dir(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.set_goal("goal")
    memory.save()
    assert (Path(memory_config.root) / "reports" / "memory_latest.md").is_file()


def test_events_and_journal(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    memory.record_event("note", "the screen flickered once")
    assert any(entry["kind"] == "note" for entry in memory.recent_journal(10))
    assert memory.snapshot()["data"]["events"][-1]["text"] == "the screen flickered once"


def test_repeated_saves_are_atomic(memory_config: MemoryConfig) -> None:
    memory = AgentMemory(memory_config)
    _populate(memory)
    for _ in range(3):
        memory.save()
    assert list(Path(memory_config.root).rglob("*.tmp")) == []
    assert json.loads((Path(memory_config.root) / "memory.json").read_text(encoding="utf-8"))["version"] == 1
