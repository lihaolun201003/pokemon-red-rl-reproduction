"""Tests for the persisted multi-agent :class:`EventBus`.

The tests are deliberately hermetic: no PyBoy, no ROM, no network, no API key,
only small in-memory messages plus ``tmp_path`` JSONL files.  They cover the
contracts the integration depends on:

* monotonic sequences and priority-ordered drains,
* idempotent ``publish`` and content de-duplication,
* ack/nack bookkeeping and bounded memory,
* JSONL persistence, sidecar state, restart replay and corrupt-file tolerance,
* the "never enqueue a message that could not be persisted" rule.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from llm_agent.multi_agent.contracts import (
    AgentMessage,
    AgentRole,
    DuplicateMessageError,
    EventType,
    MultiAgentError,
    Priority,
    ProtocolError,
    new_message_id,
)
from llm_agent.multi_agent.event_bus import EventBus, EventBusConfig

# Running counter so every helper message has unique content unless asked not to.
_SEQ = {"value": 0}


def _unique_text(prefix: str = "message") -> str:
    _SEQ["value"] += 1
    return f"{prefix}-{_SEQ['value']}"


def make_message(
    *,
    text: str | None = None,
    source: AgentRole = AgentRole.NAVIGATION,
    target: AgentRole = AgentRole.DIRECTOR,
    priority: Priority = Priority.NORMAL,
    task_id: str = "",
    message_id: str = "",
    event_type: EventType = EventType.AGENT_MESSAGE,
    payload: dict | None = None,
) -> AgentMessage:
    """Build a valid, unique :class:`AgentMessage` for the bus tests."""
    if source is target and target not in (AgentRole.DIRECTOR, AgentRole.SYSTEM):
        source = AgentRole.DIRECTOR  # self-addressed traffic is only legal for director/system
    return AgentMessage.create(
        event_type,
        source_agent=source,
        target_agent=target,
        payload=payload if payload is not None else {"text": text or _unique_text()},
        priority=priority,
        parent_task_id=task_id,
        message_id=message_id,
    )


def make_bus(path: str = "", **overrides) -> EventBus:
    """Build an :class:`EventBus` with the overridden config values."""
    config = EventBusConfig(path=path, **overrides)
    return EventBus(config)


@pytest.fixture()
def bus_path(tmp_path: Path) -> str:
    return str(tmp_path / "event_bus.jsonl")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_config_defaults() -> None:
    config = EventBusConfig()
    assert config.path == ""
    assert config.persist is True
    assert config.max_in_memory == 5000
    assert config.dedupe_window == 5000
    assert config.flush_every == 1
    assert config.fsync is False


def test_memory_only_bus_by_default() -> None:
    bus = make_bus()
    try:
        assert bus.persistent is False
        assert bus.branch_id == ""
        assert bus.stats()["published"] == 0
        assert bus.pending() == 0
    finally:
        bus.close()


def test_branch_id_can_be_set_cleared_and_read() -> None:
    bus = make_bus()
    try:
        bus.set_branch("branch_abc")
        assert bus.branch_id == "branch_abc"
        bus.set_branch("")
        assert bus.branch_id == ""
    finally:
        bus.close()


# --------------------------------------------------------------------------- #
# Publishing / sequence
# --------------------------------------------------------------------------- #

def test_publish_assigns_monotonic_sequence() -> None:
    bus = make_bus()
    try:
        first = bus.publish(make_message())
        second = bus.publish(make_message())
        third = bus.publish(make_message())
        assert (first.sequence, second.sequence, third.sequence) == (1, 2, 3)
        assert bus.stats()["sequence"] == 3
        assert bus.stats()["published"] == 3
    finally:
        bus.close()


def test_publish_returns_stored_copy_and_sets_sequence_on_input() -> None:
    bus = make_bus()
    try:
        message = make_message()
        stored = bus.publish(message)
        assert message.sequence == 1
        assert stored is not message
        assert stored.sequence == 1
        assert stored.to_dict() == message.to_dict()
    finally:
        bus.close()


def test_publish_tags_current_branch_unless_explicit() -> None:
    bus = make_bus()
    try:
        bus.set_branch("branch_active")
        tagged = bus.publish(make_message())
        assert tagged.branch_id == "branch_active"

        explicit = AgentMessage.create(
            EventType.AGENT_MESSAGE,
            source_agent=AgentRole.NAVIGATION,
            payload={"text": _unique_text()},
            branch_id="branch_historical",
        )
        assert bus.publish(explicit).branch_id == "branch_historical"
    finally:
        bus.close()


def test_publish_sets_created_at_from_injected_clock_when_zero() -> None:
    bus = EventBus(EventBusConfig(), clock=lambda: 42.5)
    try:
        message = AgentMessage(
            message_id=new_message_id(),
            event_type=EventType.AGENT_MESSAGE,
            source_agent=AgentRole.NAVIGATION,
            target_agent=AgentRole.DIRECTOR,
            payload={"text": _unique_text()},
            created_at=0.0,
        )
        stored = bus.publish(message)
        assert stored.created_at == pytest.approx(42.5)
    finally:
        bus.close()


def test_publish_rejects_non_message_objects() -> None:
    bus = make_bus()
    try:
        with pytest.raises(ProtocolError):
            bus.publish({"not": "a message"})  # type: ignore[arg-type]
        with pytest.raises(ProtocolError):
            bus.publish("msg_text")  # type: ignore[arg-type]
        assert bus.stats()["published"] == 0
    finally:
        bus.close()


def test_publish_rejects_invalid_payload() -> None:
    bus = make_bus()
    try:
        invalid = AgentMessage(
            message_id=new_message_id(),
            event_type=EventType.TASK_COMPLETED,
            source_agent=AgentRole.NAVIGATION,
            target_agent=AgentRole.DIRECTOR,
            payload={},  # TASK_COMPLETED requires a task_id
        )
        with pytest.raises(ProtocolError):
            bus.publish(invalid)
        assert bus.stats()["published"] == 0
        assert bus.pending(AgentRole.DIRECTOR) == 0
    finally:
        bus.close()


# --------------------------------------------------------------------------- #
# Idempotency / deduplication
# --------------------------------------------------------------------------- #

def test_republish_same_message_id_is_idempotent() -> None:
    bus = make_bus()
    try:
        message = make_message()
        first = bus.publish(message)
        replay = bus.publish(message)
        assert replay is first
        assert replay.sequence == 1
        assert bus.stats()["published"] == 1
        assert bus.stats()["sequence"] == 1
        assert bus.stats()["idempotent_replays"] == 1
        assert bus.pending(AgentRole.DIRECTOR) == 1
    finally:
        bus.close()


def test_duplicate_content_raises_and_is_not_enqueued() -> None:
    bus = make_bus()
    try:
        original = make_message(text="same content")
        bus.publish(original)
        duplicate = make_message(text="same content")
        with pytest.raises(DuplicateMessageError) as info:
            bus.publish(duplicate)
        assert "duplicate" in str(info.value).lower()
        assert bus.stats()["duplicates_dropped"] == 1
        assert bus.stats()["published"] == 1
        assert bus.pending(AgentRole.DIRECTOR) == 1
    finally:
        bus.close()


def test_dedupe_can_be_disabled_per_publish() -> None:
    bus = make_bus()
    try:
        bus.publish(make_message(text="repeated heartbeat"))
        again = make_message(text="repeated heartbeat")
        stored = bus.publish(again, dedupe=False)
        assert stored.sequence == 2
        assert bus.pending(AgentRole.DIRECTOR) == 2
    finally:
        bus.close()


def test_dedupe_window_expires_old_content() -> None:
    bus = make_bus(dedupe_window=2)
    try:
        first = make_message(text="content-A")
        bus.publish(first)
        bus.publish(make_message(text="content-B"))
        bus.publish(make_message(text="content-C"))  # windows A,B -> B,C
        # A's content left the window: republishing it is accepted again.
        replay = bus.publish(make_message(text="content-A"))
        assert replay.sequence == 4
    finally:
        bus.close()


def test_dedupe_key_of_different_targets_does_not_collide() -> None:
    bus = make_bus()
    try:
        text = "broadcast"
        bus.publish(make_message(text=text, target=AgentRole.NAVIGATION))
        bus.publish(make_message(text=text, target=AgentRole.STORY))
        assert bus.pending(AgentRole.NAVIGATION) == 1
        assert bus.pending(AgentRole.STORY) == 1
    finally:
        bus.close()


# --------------------------------------------------------------------------- #
# Drain / priority
# --------------------------------------------------------------------------- #

def test_drain_returns_highest_priority_first_stable_by_sequence() -> None:
    bus = make_bus()
    try:
        low = bus.publish(make_message(priority=Priority.LOW))
        high = bus.publish(make_message(priority=Priority.HIGH))
        normal_a = bus.publish(make_message(priority=Priority.NORMAL))
        critical = bus.publish(make_message(priority=Priority.CRITICAL))
        normal_b = bus.publish(make_message(priority=Priority.NORMAL))
        drained = bus.drain(AgentRole.DIRECTOR, max_messages=10)
        assert [m.sequence for m in drained] == [
            critical.sequence,
            high.sequence,
            normal_a.sequence,
            normal_b.sequence,
            low.sequence,
        ]
    finally:
        bus.close()


def test_drain_respects_max_messages_and_leaves_the_rest() -> None:
    bus = make_bus()
    try:
        for _ in range(5):
            bus.publish(make_message())
        first_batch = bus.drain(AgentRole.DIRECTOR, max_messages=2)
        assert len(first_batch) == 2
        assert bus.pending(AgentRole.DIRECTOR) == 3
        second_batch = bus.drain(AgentRole.DIRECTOR, max_messages=10)
        assert len(second_batch) == 3
        assert bus.pending(AgentRole.DIRECTOR) == 0
        assert {m.sequence for m in first_batch} | {m.sequence for m in second_batch} == {
            1, 2, 3, 4, 5
        }
    finally:
        bus.close()


def test_drain_min_priority_filters_low_messages() -> None:
    bus = make_bus()
    try:
        bus.publish(make_message(priority=Priority.LOW))
        bus.publish(make_message(priority=Priority.CRITICAL))
        drained = bus.drain(AgentRole.DIRECTOR, min_priority=int(Priority.NORMAL))
        assert len(drained) == 1
        assert drained[0].priority == Priority.CRITICAL
        assert bus.pending(AgentRole.DIRECTOR) == 1
    finally:
        bus.close()


def test_drain_does_not_ack() -> None:
    bus = make_bus()
    try:
        bus.publish(make_message())
        bus.drain(AgentRole.DIRECTOR)
        assert bus.consumption_progress()["director"] == 0
        assert bus.stats()["consumed"] == 0
        assert bus.stats()["inflight"] == 1
    finally:
        bus.close()


def test_drain_returns_only_messages_for_the_requested_role() -> None:
    bus = make_bus()
    try:
        bus.publish(make_message(target=AgentRole.NAVIGATION))
        bus.publish(make_message(target=AgentRole.STORY))
        bus.publish(make_message(target=AgentRole.DIRECTOR))
        assert len(bus.drain(AgentRole.NAVIGATION)) == 1
        assert len(bus.drain(AgentRole.STORY)) == 1
        assert len(bus.drain(AgentRole.DIRECTOR)) == 1
    finally:
        bus.close()


def test_drain_zero_or_negative_limit_returns_empty() -> None:
    bus = make_bus()
    try:
        bus.publish(make_message())
        assert bus.drain(AgentRole.DIRECTOR, max_messages=0) == []
        assert bus.drain(AgentRole.DIRECTOR, max_messages=-3) == []
        assert bus.pending(AgentRole.DIRECTOR) == 1
    finally:
        bus.close()


# --------------------------------------------------------------------------- #
# Ack / nack / pending
# --------------------------------------------------------------------------- #

def test_ack_by_message_object_and_by_id_records_progress() -> None:
    bus = make_bus()
    try:
        first = bus.publish(make_message())
        second = bus.publish(make_message())
        drained = bus.drain(AgentRole.DIRECTOR, max_messages=10)
        assert [m.sequence for m in drained] == [first.sequence, second.sequence]
        bus.ack(drained[0])
        assert bus.consumption_progress()["director"] == first.sequence
        bus.ack(second.message_id)
        assert bus.consumption_progress()["director"] == second.sequence
        assert bus.stats()["consumed"] == 2
        assert bus.stats()["inflight"] == 0
    finally:
        bus.close()


def test_ack_removes_a_message_that_was_never_drained() -> None:
    bus = make_bus()
    try:
        message = bus.publish(make_message())
        bus.ack(message)
        assert bus.pending(AgentRole.DIRECTOR) == 0
        assert bus.consumption_progress()["director"] == 1
    finally:
        bus.close()


def test_ack_unknown_id_is_a_noop() -> None:
    bus = make_bus()
    try:
        bus.ack("msg_does_not_exist")
        assert bus.stats()["consumed"] == 0
        assert bus.consumption_progress()["director"] == 0
    finally:
        bus.close()


def test_nack_requeues_message() -> None:
    bus = make_bus()
    try:
        message = bus.publish(make_message())
        drained = bus.drain(AgentRole.DIRECTOR)
        assert bus.pending(AgentRole.DIRECTOR) == 0
        bus.nack(drained[0])
        assert bus.pending(AgentRole.DIRECTOR) == 1
        again = bus.drain(AgentRole.DIRECTOR)
        assert [m.message_id for m in again] == [message.message_id]
    finally:
        bus.close()


def test_nack_without_requeue_drops_message() -> None:
    bus = make_bus()
    try:
        bus.publish(make_message())
        drained = bus.drain(AgentRole.DIRECTOR)
        bus.nack(drained[0], requeue=False)
        assert bus.pending(AgentRole.DIRECTOR) == 0
        assert bus.stats()["inflight"] == 0
        assert bus.stats()["consumed"] == 0
    finally:
        bus.close()


def test_nack_after_ack_does_not_resurrect() -> None:
    bus = make_bus()
    try:
        message = bus.publish(make_message())
        bus.ack(message)
        bus.nack(message)
        assert bus.pending(AgentRole.DIRECTOR) == 0
        assert bus.consumption_progress()["director"] == 1
    finally:
        bus.close()


def test_pending_per_role_and_total() -> None:
    bus = make_bus()
    try:
        bus.publish(make_message(target=AgentRole.NAVIGATION))
        bus.publish(make_message(target=AgentRole.NAVIGATION))
        bus.publish(make_message(target=AgentRole.STORY))
        assert bus.pending(AgentRole.NAVIGATION) == 2
        assert bus.pending(AgentRole.STORY) == 1
        assert bus.pending(AgentRole.POKEMON) == 0
        assert bus.pending() == 3
    finally:
        bus.close()


def test_consumption_progress_has_every_role() -> None:
    bus = make_bus()
    try:
        progress = bus.consumption_progress()
        assert set(progress) == {role.value for role in AgentRole}
        assert all(value == 0 for value in progress.values())
    finally:
        bus.close()


# --------------------------------------------------------------------------- #
# recent() / stats()
# --------------------------------------------------------------------------- #

def test_recent_is_chronological_and_bounded() -> None:
    bus = make_bus(max_in_memory=3)
    try:
        for _ in range(5):
            bus.publish(make_message())
        recent = bus.recent(limit=10)
        assert [m.sequence for m in recent] == [3, 4, 5]
        assert len(bus.recent(limit=0)) == 0
        assert bus.stats()["in_memory"] == 3
    finally:
        bus.close()


def test_recent_default_limit() -> None:
    bus = make_bus()
    try:
        for _ in range(3):
            bus.publish(make_message())
        assert len(bus.recent()) == 3
        assert len(bus.recent(limit=2)) == 2
    finally:
        bus.close()


def test_stats_shape() -> None:
    bus = make_bus()
    try:
        bus.publish(make_message())
        stats = bus.stats()
        for key in (
            "published",
            "duplicates_dropped",
            "consumed",
            "pending",
            "sequence",
            "branch_id",
        ):
            assert key in stats
        assert set(stats["pending"]) == {role.value for role in AgentRole}
    finally:
        bus.close()


# --------------------------------------------------------------------------- #
# Thread safety
# --------------------------------------------------------------------------- #

def test_concurrent_publishers_get_unique_sequences() -> None:
    bus = make_bus()
    try:
        errors: list[BaseException] = []

        def producer(count: int) -> None:
            try:
                for _ in range(count):
                    bus.publish(make_message())
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=producer, args=(25,)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert not errors, errors
        assert bus.stats()["published"] == 100
        assert bus.stats()["sequence"] == 100
        drained = bus.drain(AgentRole.DIRECTOR, max_messages=1000)
        assert len(drained) == 100
        assert sorted(m.sequence for m in drained) == list(range(1, 101))
    finally:
        bus.close()


# --------------------------------------------------------------------------- #
# Persistence / restart
# --------------------------------------------------------------------------- #

def test_persist_and_reload_restores_messages_and_sequence(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        first = bus.publish(make_message(text="first"))
        second = bus.publish(make_message(text="second"))
        bus.drain(AgentRole.DIRECTOR)
        bus.close()
    finally:
        bus.close()

    reloaded = make_bus(bus_path)
    try:
        assert reloaded.stats()["published"] == 2
        assert reloaded.stats()["sequence"] == 2
        assert reloaded.pending(AgentRole.DIRECTOR) == 2
        assert reloaded.recent(limit=2)[0].payload["text"] == "first"
        third = reloaded.publish(make_message(text="third"))
        assert third.sequence == 3
        assert third.sequence not in (first.sequence, second.sequence)
    finally:
        reloaded.close()


def test_ack_progress_persisted_across_restart(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        messages = [bus.publish(make_message(text=f"m{index}")) for index in range(3)]
        drained = bus.drain(AgentRole.DIRECTOR, max_messages=10)
        # Consume the first two in publication order; crash before the third.
        bus.ack(messages[0])
        bus.ack(messages[1])
        assert len(drained) == 3
    finally:
        bus.close()

    reloaded = make_bus(bus_path)
    try:
        assert reloaded.consumption_progress()["director"] == 2
        replayed = reloaded.replay_unconsumed()
        assert [m.payload["text"] for m in replayed] == ["m2"]
        assert reloaded.pending(AgentRole.DIRECTOR) == 1
        # Replaying again must not duplicate the inbox entry.
        assert [m.payload["text"] for m in reloaded.replay_unconsumed()] == ["m2"]
        assert reloaded.pending(AgentRole.DIRECTOR) == 1
    finally:
        reloaded.close()


def test_out_of_order_ack_cannot_skip_older_unacked_messages(
    bus_path: str,
) -> None:
    """Acking a high-priority (higher sequence) message must not lose lower ones."""
    bus = make_bus(bus_path)
    try:
        low = bus.publish(make_message(text="low", priority=Priority.LOW))
        high = bus.publish(make_message(text="high", priority=Priority.HIGH))
        drained = bus.drain(AgentRole.DIRECTOR, max_messages=10)
        assert [m.sequence for m in drained] == [high.sequence, low.sequence]
        # Handler acks in drain order: the high-priority message first.
        bus.ack(drained[0])
        # The consumed mark is blocked by the still-unacked lower sequence.
        assert bus.consumption_progress()["director"] == 0
    finally:
        bus.close()

    reloaded = make_bus(bus_path)
    try:
        replayed = reloaded.replay_unconsumed()
        assert {m.payload["text"] for m in replayed} == {"low", "high"}
        assert reloaded.pending(AgentRole.DIRECTOR) == 2
    finally:
        reloaded.close()


def test_ack_advances_to_contiguous_frontier_after_gap_is_filled(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        first = bus.publish(make_message(text="first"))
        second = bus.publish(make_message(text="second"))
        third = bus.publish(make_message(text="third"))
        # Ack the two later messages first, then the older one.
        bus.ack(third)
        bus.ack(second)
        assert bus.consumption_progress()["director"] == 0  # blocked by seq 1
        bus.ack(first)
        assert bus.consumption_progress()["director"] == third.sequence
        assert bus.stats()["consumed"] == 3
    finally:
        bus.close()


def test_drained_but_unacked_messages_are_replayed_after_restart(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        for index in range(2):
            bus.publish(make_message(text=f"pending{index}"))
        bus.drain(AgentRole.DIRECTOR, max_messages=10)  # never acked -> simulated crash
    finally:
        bus.close()

    reloaded = make_bus(bus_path)
    try:
        assert reloaded.pending(AgentRole.DIRECTOR) == 2
        assert reloaded.stats()["replayed_on_load"] == 2
    finally:
        reloaded.close()


def test_load_missing_file_returns_false(tmp_path: Path) -> None:
    missing = make_bus(str(tmp_path / "does_not_exist.jsonl"))
    try:
        assert missing.load() is False
    finally:
        missing.close()


def test_load_memory_only_bus_returns_false() -> None:
    bus = make_bus()
    try:
        assert bus.load() is False
    finally:
        bus.close()


def test_load_is_idempotent(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        bus.publish(make_message())
    finally:
        bus.close()

    reloaded = make_bus(bus_path)
    try:
        assert reloaded.load() is True
        assert reloaded.load() is True
        assert reloaded.pending(AgentRole.DIRECTOR) == 1
        assert reloaded.stats()["published"] == 1
    finally:
        reloaded.close()


def test_corrupt_trailing_line_is_tolerated(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        bus.publish(make_message(text="ok-1"))
        bus.publish(make_message(text="ok-2"))
    finally:
        bus.close()
    with open(bus_path, "a", encoding="utf-8") as handle:
        handle.write('{"sequence": 3, "message": {"broken": \n')

    reloaded = make_bus(bus_path)
    try:
        assert reloaded.stats()["published"] == 2
        assert reloaded.pending(AgentRole.DIRECTOR) == 2
        assert reloaded.stats()["load_errors"] >= 1
        recovered = reloaded.publish(make_message(text="after-corruption"))
        assert recovered.sequence == 3
    finally:
        reloaded.close()


def test_corrupt_state_file_does_not_break_load(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        bus.publish(make_message())
        bus.close()
    finally:
        bus.close()
    Path(f"{bus_path}.state.json").write_text("{ this is not json", encoding="utf-8")

    reloaded = make_bus(bus_path)
    try:
        assert reloaded.pending(AgentRole.DIRECTOR) == 1
        assert reloaded.stats()["load_errors"] >= 1
    finally:
        reloaded.close()


def test_state_file_stores_acked_sequences_dedupe_and_branch(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        message = bus.publish(make_message())
        bus.set_branch("branch_persisted")
        bus.ack(message)
        bus.save()
    finally:
        bus.close()

    state = json.loads(Path(f"{bus_path}.state.json").read_text(encoding="utf-8"))
    assert state["branch_id"] == "branch_persisted"
    assert state["acked"]["director"] == 1
    assert state["sequence"] == 1
    assert message.message_id in state["seen_ids"]
    assert state["dedupe_keys"]
    assert isinstance(state["saved_at"], float)


def test_state_write_is_atomic_and_leaves_no_tmp_file(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        bus.publish(make_message())
        bus.save()
        directory = Path(bus_path).parent
        leftovers = [path for path in directory.iterdir() if ".tmp" in path.name]
        assert leftovers == []
        assert Path(f"{bus_path}.state.json").is_file()
    finally:
        bus.close()


def test_save_flushes_buffered_journal(bus_path: str) -> None:
    bus = make_bus(bus_path, flush_every=1000)
    try:
        bus.publish(make_message())
        assert Path(bus_path).read_text(encoding="utf-8") == ""
        bus.save()
        lines = [line for line in Path(bus_path).read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["sequence"] == 1
        assert record["message"]["message_id"]
    finally:
        bus.close()


def test_journal_record_layout(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        bus.publish(make_message(text="layout"))
    finally:
        bus.close()
    lines = [line for line in Path(bus_path).read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"sequence", "message"}
    assert record["message"]["payload"]["text"] == "layout"


def test_branch_id_is_restored_from_state(bus_path: str) -> None:
    bus = make_bus(bus_path)
    try:
        bus.set_branch("branch_restored")
        bus.publish(make_message())
    finally:
        bus.close()
    reloaded = make_bus(bus_path)
    try:
        assert reloaded.branch_id == "branch_restored"
    finally:
        reloaded.close()


def test_publish_persistence_failure_does_not_enqueue(tmp_path: Path) -> None:
    # A directory at the journal path makes the append fail deterministically.
    blocked = tmp_path / "event_bus.jsonl"
    blocked.mkdir()
    bus = make_bus(str(blocked))
    try:
        message = make_message()
        with pytest.raises(MultiAgentError):
            bus.publish(message)
        assert bus.stats()["published"] == 0
        assert bus.stats()["sequence"] == 0
        assert bus.pending(AgentRole.DIRECTOR) == 0
        assert bus.recent() == []
        assert message.sequence == 0
    finally:
        bus.close()


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

def test_close_is_idempotent_and_blocks_mutations() -> None:
    bus = make_bus()
    bus.publish(make_message())
    bus.close()
    bus.close()  # idempotent
    with pytest.raises(MultiAgentError):
        bus.publish(make_message())
    with pytest.raises(MultiAgentError):
        bus.drain(AgentRole.DIRECTOR)
    with pytest.raises(MultiAgentError):
        bus.ack("msg_x")


def test_context_manager_closes_bus(bus_path: str) -> None:
    with make_bus(bus_path) as bus:
        bus.publish(make_message())
    assert Path(bus_path).is_file()
    with pytest.raises(MultiAgentError):
        bus.publish(make_message())
