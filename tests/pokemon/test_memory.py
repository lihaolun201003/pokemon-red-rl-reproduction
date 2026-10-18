"""PokemonMemory tests: battle records, lessons, briefs, snapshots, hooks."""

from __future__ import annotations

import json

import pytest

from llm_agent.memory import AgentMemory
from llm_agent.multi_agent.contracts import AgentRole, KnowledgeEntry, VerificationStatus
from llm_agent.pokemon.memory import PokemonMemory


def _memory(tmp_path, cfg) -> tuple[PokemonMemory, AgentMemory]:
    agent_memory = AgentMemory(cfg.memory)
    battle_memory = PokemonMemory.create(memory=agent_memory, config=cfg, root=tmp_path / "battle_memory")
    return battle_memory, agent_memory


def test_record_battle_stores_outcome_and_lesson(tmp_path, agent_config) -> None:
    memory, _ = _memory(tmp_path, agent_config)

    lesson = memory.record_battle(
        outcome="win",
        turns=3,
        actions=[
            {"turn": 1, "action": "fight", "slot": 2, "presses": ["a", "down", "a"], "verified": True},
            {"turn": 2, "action": "observe", "slot": None, "presses": [], "verified": True},
        ],
        cost_usd=0.01,
        evidence_refs=["screens/battle_1.png"],
        visible_text="the enemy fainted",
        notes="resolved by pixels",
        verification="verified",
    )

    assert lesson["category"] == "battle_lesson"
    assert lesson["verification_status"] == VerificationStatus.VERIFIED.value
    assert "outcome=win" in lesson["content"]
    assert "turns=3" in lesson["content"]
    assert lesson["evidence_refs"] == ["screens/battle_1.png"]
    assert memory.stats()["battles"] == 1
    assert memory.stats()["wins"] == 1
    assert memory.knowledge_entry(lesson["knowledge_id"]) is not None


def test_record_move_observed_is_verified_and_deduped(tmp_path, agent_config) -> None:
    memory, agent_memory = _memory(tmp_path, agent_config)

    first = memory.record_move_observed(
        "a visible label",
        slot=2,
        visible_text="the move hit the opponent",
        evidence_refs=["screens/f1.png"],
    )
    second = memory.record_move_observed("a visible label", slot=2, evidence_refs=["screens/f2.png"])

    assert first["verification_status"] == "verified"
    assert second["verification_status"] == "verified"
    stats = memory.stats()
    assert stats["moves"] == 1  # deduped by name
    brief = memory.render_battle_brief()
    assert "a visible label" in brief
    assert "chosen 2x" in brief
    # The generic AgentMemory also received the lessons (prompt memory mirror).
    assert agent_memory.stats()["lessons"] >= 1


def test_record_damage_and_battle_text_are_verified(tmp_path, agent_config) -> None:
    memory, _ = _memory(tmp_path, agent_config)

    damage = memory.record_damage_observation(
        "HP text went down", visible_text="HP text", move_name="a visible label"
    )
    text = memory.record_battle_text("a readable battle message")

    assert damage["verification_status"] == "verified"
    assert text["verification_status"] == "verified"
    brief = memory.render_battle_brief()
    assert "HP text went down" in brief
    assert "VISIBLE DAMAGE" in brief


def test_record_inference_is_labelled_inferred(tmp_path, agent_config) -> None:
    memory, _ = _memory(tmp_path, agent_config)

    lesson = memory.record_inference("the move at slot two seems effective", confidence=0.4)

    assert lesson["verification_status"] == VerificationStatus.INFERRED.value
    assert 0.0 <= lesson["confidence"] <= 1.0
    assert "Inference" in lesson["content"]
    assert "unverified" in memory.render_battle_brief().lower()


def test_record_team_member_and_team_summary(tmp_path, agent_config) -> None:
    memory, _ = _memory(tmp_path, agent_config)

    lesson = memory.record_team_member_observed(
        "a visible party name", slot=1, hp_text="HP text", level_text="level text"
    )

    assert lesson["category"] == "pokemon"
    summary = memory.team_summary()
    assert summary["count"] == 1
    assert summary["members"][0]["name"] == "a visible party name"
    brief = memory.render_team_brief()
    assert "a visible party name" in brief
    assert "HP text" in brief


def test_persistence_roundtrip(tmp_path, agent_config) -> None:
    memory, agent_memory = _memory(tmp_path, agent_config)
    memory.record_battle(
        outcome="ran", turns=1, evidence_refs=["screens/a.png"], verification="inferred"
    )
    memory.record_move_observed("a visible label", slot=1)
    memory.record_team_member_observed("a visible party name", slot=2)
    snapshot = memory.snapshot()
    path = memory.memory_path
    assert path.is_file()

    reopened = PokemonMemory(agent_memory, root=tmp_path / "battle_memory")
    assert reopened.stats()["battles"] == 1
    assert reopened.stats()["runs"] == 1
    assert reopened.stats()["moves"] == 1
    assert reopened.team_summary()["count"] == 1
    assert reopened.knowledge_entries()
    for entry in reopened.knowledge_entries():
        assert isinstance(entry, KnowledgeEntry)

    restored = PokemonMemory(agent_memory, root=tmp_path / "battle_memory")
    assert restored.restore(snapshot) is True
    assert restored.stats()["battles"] == 1


def test_corrupt_file_resets_without_crashing(tmp_path, agent_config) -> None:
    agent_memory = AgentMemory(agent_config.memory)
    root = tmp_path / "battle_memory"
    root.mkdir(parents=True, exist_ok=True)
    (root / "pokemon_memory.json").write_text("{not valid json", encoding="utf-8")

    memory = PokemonMemory(agent_memory, root=root)

    assert memory.load_errors
    assert memory.recovered_from is not None
    assert memory.stats()["battles"] == 0
    # The corrupt file was moved aside and a fresh document written.
    assert memory.memory_path.is_file()
    json.loads(memory.memory_path.read_text(encoding="utf-8"))


def test_snapshot_restore_is_defensive(tmp_path, agent_config) -> None:
    memory, _ = _memory(tmp_path, agent_config)
    assert memory.restore(None) is False
    assert memory.restore({}) is False
    assert memory.restore({"data": "junk"}) is False


def test_render_context_combines_generic_and_battle_memory(tmp_path, agent_config) -> None:
    memory, agent_memory = _memory(tmp_path, agent_config)
    agent_memory.set_goal("survive the battles", reason="the team is weak")
    memory.record_battle(outcome="loss", turns=2, verification="inferred")
    memory.record_move_observed("a visible label", slot=1)

    text = memory.render_context(max_chars=2500)

    assert "AGENT MEMORY" in text
    assert "BATTLE HISTORY" in text
    assert "a visible label" in text
    assert len(text) <= 2500
    assert "loss" in text
    # render_for_prompt is the duck-typed alias used by AgentContext.
    assert memory.render_for_prompt(max_chars=500)


def test_long_term_hooks_are_structured_but_unvalidated(tmp_path, agent_config) -> None:
    memory, _ = _memory(tmp_path, agent_config)

    records = [
        memory.record_training_observation("trained for a while", success=True),
        memory.record_evolution_observed("a previous form", "a new form", visible_text="visible text"),
        memory.record_move_learned("a learned label", replaced_move="an old label", visible_text="visible"),
        memory.record_team_change("swapped the party order", slot=1),
        memory.record_item_use("a visible item", target="a party member", visible_text="visible"),
    ]

    for record in records:
        assert record["status"] == "structured_only"
        assert record["validated"] is False
    counts = memory.long_term_summary()
    assert counts["training"] == 1
    assert counts["evolutions"] == 1
    assert counts["moves_learned"] == 1
    assert counts["team_changes"] == 1
    assert counts["item_uses"] == 1
    with pytest.raises(ValueError):
        memory.record_training_observation("")


def test_large_memory_is_capped(tmp_path, agent_config) -> None:
    from llm_agent.pokemon import memory as memory_module

    memory, _ = _memory(tmp_path, agent_config)
    for index in range(memory_module.MAX_BATTLES + 5):
        memory.record_battle(outcome="unknown", turns=1, verification="inferred")
    assert memory.stats()["battles"] <= memory_module.MAX_BATTLES


def test_standalone_creation_without_agent_memory(tmp_path, agent_config) -> None:
    memory = PokemonMemory.create(memory=None, config=agent_config, root=tmp_path / "solo")
    lesson = memory.record_inference("a guess")
    assert lesson["knowledge_id"]
    assert memory.agent_memory is not None
    assert memory.render_context(max_chars=400)


def test_branch_tagging_reaches_knowledge_entries(tmp_path, agent_config) -> None:
    memory, _ = _memory(tmp_path, agent_config)
    memory.set_branch("branch_test123")
    lesson = memory.record_battle(outcome="win", turns=1, verification="verified")
    entry = memory.knowledge_entry(lesson["knowledge_id"])
    assert entry is not None
    assert entry.source_agent is AgentRole.POKEMON
