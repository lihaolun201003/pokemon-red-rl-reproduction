"""Persisted, priority-ordered event bus for the multi-agent subsystem.

The bus is the single message channel between the Game Director and the
specialised agents.  It is deliberately small and dependency free so that it can
be unit-tested without PyBoy, the network or an API key.

Semantics
---------
* **Publish** validates the :class:`~llm_agent.multi_agent.contracts.AgentMessage`,
  assigns a monotonically increasing ``sequence`` (persisted), appends one JSONL
  record and enqueues the message in the target agent's inbox.
* **Idempotent replay**: publishing a ``message_id`` that is already known
  returns the stored message unchanged and does *not* allocate a new sequence,
  so a producer retry after a crash can never duplicate a logical message.
* **Content de-duplication**: a second message whose
  :meth:`AgentMessage.dedupe_key` is still inside the ``dedupe_window`` of recent
  messages raises :class:`~llm_agent.multi_agent.contracts.DuplicateMessageError`.
* **Drain** returns up to ``max_messages`` pending messages for one role,
  highest priority first and stable in publication order.  Draining does *not*
  acknowledge anything: the caller must :meth:`EventBus.ack` (or
  :meth:`EventBus.nack`) every message it processed.
* **Acknowledgement** persists a gap-free consumed sequence per role so a
  restart can replay exactly the messages that were never consumed.  Because
  draining is priority-first, acks may arrive out of sequence order; the
  persisted mark only advances past messages that are known to be consumed, so
  an early ack can never skip an older unprocessed message.  Delivery is
  *at least once*: consumers must be idempotent per ``message_id``.
* **Persistence** is one JSON object per line::

      {"sequence": 12, "message": {...AgentMessage.to_dict()...}}

  plus a sidecar ``<path>.state.json`` written atomically that stores the dedupe
  window, per-role consumed sequences and the current ``branch_id``.
* Publishing never loses a message silently: if the JSONL write fails the
  message is **not** enqueued and a
  :class:`~llm_agent.multi_agent.contracts.MultiAgentError` is raised.
* Memory is bounded: the replay window, the de-duplication window and the recent
  history are all capped, so a long run cannot grow without limit.

The bus is intentionally single-process.  An :class:`threading.RLock` protects
every mutation so producer/consumer threads of one process can share it; there
is no cross-process file locking.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Deque, Mapping, TextIO

from llm_agent.multi_agent.contracts import (
    PROTOCOL_VERSION,
    AgentMessage,
    AgentRole,
    DuplicateMessageError,
    MultiAgentError,
    ProtocolError,
)

#: Suffix appended to the configured journal path for the atomic state file.
STATE_SUFFIX: str = ".state.json"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class EventBusConfig:
    """Tuning knobs for :class:`EventBus` (all values are hard bounds)."""

    #: JSONL persistence file.  ``""`` (or ``persist=False``) keeps the bus in memory.
    path: str = ""
    #: Master switch for journal/state persistence.
    persist: bool = True
    #: Maximum number of messages kept in the ``recent`` history roll.
    max_in_memory: int = 5000
    #: Number of recent messages used for content de-duplication (0 disables it).
    dedupe_window: int = 5000
    #: Flush the JSONL journal every N published records (>= 1).
    flush_every: int = 1
    #: ``os.fsync`` the journal after every flush (durable but slower).
    fsync: bool = False


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically (unique temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:  # pragma: no cover - fsync is best effort here
                pass
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:  # pragma: no cover - cleanup is best effort
            pass


# --------------------------------------------------------------------------- #
# EventBus
# --------------------------------------------------------------------------- #

class EventBus:
    """Thread-safe, persisted, priority ordered message bus.

    Example::

        bus = EventBus(EventBusConfig(path=str(run_dir / "event_bus.jsonl")))
        bus.publish(AgentMessage.create(
            EventType.NAVIGATION_REQUEST,
            source_agent=AgentRole.DIRECTOR,
            target_agent=AgentRole.NAVIGATION,
            payload={"target": "somewhere"},
        ))
        for message in bus.drain(AgentRole.NAVIGATION):
            handle(message)
            bus.ack(message)
    """

    def __init__(
        self,
        config: EventBusConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, EventBusConfig):
            raise ProtocolError("EventBus config must be an EventBusConfig instance")
        cfg = config if config is not None else EventBusConfig()
        self._config = cfg
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._lock = threading.RLock()

        raw_path = str(cfg.path or "").strip()
        self._path = raw_path
        self._persistent = bool(cfg.persist) and bool(raw_path)
        self._max_in_memory = max(1, int(cfg.max_in_memory))
        self._dedupe_window = max(0, int(cfg.dedupe_window))
        self._flush_every = max(1, int(cfg.flush_every))
        self._fsync = bool(cfg.fsync)
        #: idempotency window: content dedupe window, or the whole history roll
        self._idempotency_window = max(1, self._dedupe_window or self._max_in_memory)

        # -- persisted / authoritative state -------------------------------- #
        self._sequence: int = 0
        self._branch_id: str = ""
        self._acked: dict[str, int] = {role.value: 0 for role in AgentRole}
        self._published: int = 0
        self._duplicates_dropped: int = 0
        self._idempotent_replays: int = 0
        self._consumed: int = 0
        self._load_errors: int = 0
        self._replayed_on_load: int = 0
        #: ``published`` counter recovered from the sidecar state (journal wins).
        self._state_published: int = 0

        # -- in-memory queues ----------------------------------------------- #
        self._inboxes: dict[str, list[AgentMessage]] = {role.value: [] for role in AgentRole}
        self._inflight: dict[str, AgentMessage] = {}
        self._recent: Deque[AgentMessage] = deque(maxlen=self._max_in_memory)
        self._seen: "OrderedDict[str, AgentMessage]" = OrderedDict()
        self._dedupe_order: Deque[str] = deque()
        self._dedupe_counts: dict[str, int] = {}
        #: consumed-but-not-yet-contiguous sequences per role (ack out of order)
        self._consumed_seqs: dict[str, set[int]] = {role.value: set() for role in AgentRole}

        # -- journal handle -------------------------------------------------- #
        self._journal: TextIO | None = None
        self._pending_writes: int = 0
        self._closed: bool = False

        if self._persistent and Path(self._path).is_file():
            self.load()

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #
    @property
    def config(self) -> EventBusConfig:
        """The immutable configuration this bus was built with."""
        return self._config

    @property
    def branch_id(self) -> str:
        """Branch id new messages are tagged with (``""`` = unassigned)."""
        with self._lock:
            return self._branch_id

    @property
    def path(self) -> str:
        """Journal path (``""`` for a memory-only bus)."""
        return self._path

    @property
    def persistent(self) -> bool:
        """True when journal/state persistence is active."""
        return self._persistent

    def set_branch(self, branch_id: str) -> None:
        """Switch the branch new messages are tagged with (called after rollback)."""
        if isinstance(branch_id, (bytes, bytearray)):
            raise ProtocolError("branch_id must be a string, got bytes")
        with self._lock:
            self._branch_id = str(branch_id or "").strip()
            self._save_state()

    # ------------------------------------------------------------------ #
    # Publish
    # ------------------------------------------------------------------ #
    def publish(self, message: AgentMessage, *, dedupe: bool = True) -> AgentMessage:
        """Validate, persist and enqueue ``message``.

        Returns the canonical stored copy (with ``sequence``/``branch_id``/
        ``created_at`` assigned).  The caller's object also receives the
        assigned ``sequence`` for convenience.

        Raises
        ------
        ProtocolError
            The payload/argument is not a valid :class:`AgentMessage`.
        DuplicateMessageError
            The same logical content is still inside the de-duplication window.
        MultiAgentError
            Persistence failed; nothing was enqueued.
        """
        with self._lock:
            self._ensure_open("publish")
            if not isinstance(message, AgentMessage):
                raise ProtocolError(
                    f"publish expects an AgentMessage, got {type(message).__name__}"
                )
            message.validate()

            # Idempotent replay: same message id -> same stored message, no new sequence.
            stored = self._seen.get(message.message_id)
            if stored is not None:
                self._idempotent_replays += 1
                return stored

            key = message.dedupe_key()
            if dedupe and key in self._dedupe_counts:
                self._duplicates_dropped += 1
                raise DuplicateMessageError(
                    f"duplicate {message.event_type.value} content within the dedupe window",
                    details={
                        "message_id": message.message_id,
                        "dedupe_key": key,
                        "event_type": message.event_type.value,
                    },
                )

            now = float(self._clock())
            created_at = float(message.created_at) if float(message.created_at) > 0.0 else now
            branch_id = message.branch_id or self._branch_id
            sequence = self._sequence + 1
            canonical = replace(
                message,
                payload=dict(message.payload),
                created_at=created_at,
                branch_id=branch_id,
                sequence=sequence,
            )

            # Persist BEFORE mutating any in-memory state: a failed write must
            # never leave an enqueued message that is not on disk.
            if self._persistent:
                self._append_record(canonical)

            self._sequence = sequence
            self._published += 1
            message.sequence = sequence
            self._seen[message.message_id] = canonical
            self._prune_seen_locked()
            self._record_dedupe_locked(key)
            self._recent.append(canonical)
            self._inboxes[canonical.target_agent.value].append(canonical)
            return canonical

    # ------------------------------------------------------------------ #
    # Consumption
    # ------------------------------------------------------------------ #
    def drain(
        self,
        role: AgentRole | str,
        *,
        max_messages: int = 8,
        min_priority: int = 0,
    ) -> list[AgentMessage]:
        """Take up to ``max_messages`` pending messages for ``role`` (no ack).

        Messages are returned highest priority first and stable in sequence
        order.  They leave the inbox and stay in flight until acked or nacked.
        """
        with self._lock:
            self._ensure_open("drain")
            role_enum = AgentRole.coerce(role)
            limit = max(0, int(max_messages))
            if limit <= 0:
                return []
            threshold = int(min_priority)
            inbox = self._inboxes[role_enum.value]
            candidates = [item for item in inbox if int(item.priority) >= threshold]
            candidates.sort(key=lambda item: (-int(item.priority), int(item.sequence)))
            chosen = candidates[:limit]
            chosen_ids = {item.message_id for item in chosen}
            self._inboxes[role_enum.value] = [
                item for item in inbox if item.message_id not in chosen_ids
            ]
            for item in chosen:
                self._inflight[item.message_id] = item
            return chosen

    def ack(self, message: AgentMessage | str) -> None:
        """Mark one message consumed and persist the per-role consumed sequence.

        Draining is priority-first, so acknowledgements may arrive out of
        sequence order.  The persisted per-role mark is therefore advanced only
        to the highest sequence whose lower predecessors were all consumed: an
        early ack for a high-priority message can never skip an older unacked
        message during restart replay (it may replay extra messages instead,
        which is the documented at-least-once behaviour).
        """
        with self._lock:
            self._ensure_open("ack")
            message_id = self._message_id(message)
            role_key, sequence = self._locate(message_id, message)
            removed = self._remove_pending_locked(message_id)
            if role_key is None or sequence is None:
                # Unknown id: acking is a harmless no-op (defensive for replays).
                return
            sequence_int = int(sequence)
            frontier = int(self._acked.get(role_key, 0))
            consumed = self._consumed_seqs[role_key]
            counted = sequence_int in consumed or sequence_int <= frontier
            consumed.add(sequence_int)
            if removed or not counted:
                self._consumed += 1
            if self._advance_frontier_locked(role_key):
                self._save_state()

    def nack(self, message: AgentMessage | str, *, requeue: bool = True) -> None:
        """Return a drained message to the inbox (``requeue=False`` drops it)."""
        with self._lock:
            self._ensure_open("nack")
            message_id = self._message_id(message)
            role_key, sequence = self._locate(message_id, message)
            stored = self._inflight.pop(message_id, None)
            if not requeue or stored is None:
                return
            if role_key is None or sequence is None:
                return
            if int(sequence) <= int(self._acked.get(role_key, 0)):
                return  # already consumed: never resurrect
            inbox = self._inboxes[role_key]
            if any(item.message_id == message_id for item in inbox):
                return
            inbox.append(stored)

    def pending(self, role: AgentRole | str | None = None) -> int:
        """Number of un-drained messages (for one role, or in total)."""
        with self._lock:
            if role is None:
                return sum(len(items) for items in self._inboxes.values())
            return len(self._inboxes[AgentRole.coerce(role).value])

    def replay_unconsumed(self) -> list[AgentMessage]:
        """Re-deliver every in-memory message above its role's acked sequence.

        Used for restart recovery (``load`` already replays automatically) and by
        the director to recover messages that were drained but never acked.
        Messages already in the inbox are not duplicated; in-flight ones are
        moved back into the inbox.
        """
        with self._lock:
            self._ensure_open("replay_unconsumed")
            candidates: "OrderedDict[str, AgentMessage]" = OrderedDict()
            for message in self._recent:
                candidates.setdefault(message.message_id, message)
            for inbox in self._inboxes.values():
                for message in inbox:
                    candidates.setdefault(message.message_id, message)
            for message in self._inflight.values():
                candidates.setdefault(message.message_id, message)

            ordered = sorted(
                candidates.values(),
                key=lambda item: (int(item.sequence), item.message_id),
            )
            replayed: list[AgentMessage] = []
            for message in ordered:
                role_key = message.target_agent.value
                if int(message.sequence) <= int(self._acked.get(role_key, 0)):
                    continue
                replayed.append(message)
                inbox = self._inboxes[role_key]
                if any(item.message_id == message.message_id for item in inbox):
                    continue
                self._inflight.pop(message.message_id, None)
                inbox.append(message)
            return replayed

    def consumption_progress(self) -> dict[str, int]:
        """Highest consumed sequence per role (persisted; replay uses these)."""
        with self._lock:
            return {role.value: int(self._acked.get(role.value, 0)) for role in AgentRole}

    def recent(self, *, limit: int = 50) -> list[AgentMessage]:
        """The most recent ``limit`` messages in chronological (sequence) order."""
        with self._lock:
            count = max(0, int(limit))
            if count <= 0:
                return []
            items = list(self._recent)
            return items[-count:]

    def stats(self) -> dict[str, Any]:
        """Counter snapshot for logs/acceptance evidence."""
        with self._lock:
            pending = {
                role.value: len(self._inboxes[role.value]) for role in AgentRole
            }
            return {
                "published": self._published,
                "duplicates_dropped": self._duplicates_dropped,
                "consumed": self._consumed,
                "idempotent_replays": self._idempotent_replays,
                "pending": pending,
                "queued": sum(pending.values()),
                "inflight": len(self._inflight),
                "sequence": self._sequence,
                "branch_id": self._branch_id,
                "acked": {role.value: int(self._acked.get(role.value, 0)) for role in AgentRole},
                "persistent": self._persistent,
                "path": self._path,
                "in_memory": len(self._recent),
                "load_errors": self._load_errors,
                "replayed_on_load": self._replayed_on_load,
            }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def load(self) -> bool:
        """Load the journal (+ sidecar state) into memory and replay unacked messages.

        Tolerant by design: malformed/truncated lines and a corrupt state file
        are counted in ``stats()["load_errors"]`` and skipped instead of raising.

        Returns ``True`` when a journal file was found, ``False`` otherwise.
        """
        with self._lock:
            if self._closed:
                raise MultiAgentError("event bus is closed")
            if not self._persistent:
                return False
            journal = Path(self._path)
            if not journal.is_file():
                return False
            self._apply_state(self._read_state_file())

            replayed = 0
            try:
                with journal.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            record = json.loads(text)
                        except json.JSONDecodeError:
                            self._load_errors += 1
                            continue
                        message = self._parse_record(record)
                        if message is None:
                            self._load_errors += 1
                            continue
                        if message.message_id in self._seen:
                            continue
                        self._ingest_locked(message)
                        role_key = message.target_agent.value
                        if int(message.sequence) > int(self._acked.get(role_key, 0)):
                            if self._enqueue_locked(message):
                                replayed += 1
            except OSError as exc:
                raise MultiAgentError(
                    f"failed to read event bus journal {journal}: {exc}",
                    details={"path": str(journal)},
                ) from exc
            self._replayed_on_load = replayed
            self._published = max(self._published, self._state_published)
            return True

    def save(self) -> None:
        """Flush the journal and atomically rewrite the state file."""
        with self._lock:
            self._flush_locked(force=True)
            self._save_state()

    def close(self) -> None:
        """Flush and close the bus (idempotent; publishing afterwards fails)."""
        with self._lock:
            if self._closed:
                return
            try:
                self._flush_locked(force=True)
                self._save_state()
            finally:
                self._close_journal()
                self._closed = True

    def __enter__(self) -> "EventBus":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals: queue bookkeeping
    # ------------------------------------------------------------------ #
    def _ensure_open(self, operation: str) -> None:
        if self._closed:
            raise MultiAgentError(f"event bus is closed; cannot {operation}")

    @staticmethod
    def _message_id(message: AgentMessage | str) -> str:
        if isinstance(message, AgentMessage):
            return message.message_id
        if isinstance(message, str):
            text = message.strip()
            if not text:
                raise ProtocolError("message id must be a non-empty string")
            return text
        raise ProtocolError(
            f"expected AgentMessage or message id, got {type(message).__name__}"
        )

    def _locate(
        self, message_id: str, message: AgentMessage | str
    ) -> tuple[str | None, int | None]:
        """Resolve (target role value, sequence) for a message, if known."""
        if isinstance(message, AgentMessage):
            try:
                message.validate()
            except ProtocolError:
                return None, None
            return message.target_agent.value, int(message.sequence)
        for role in AgentRole:
            inbox = self._inboxes[role.value]
            for item in inbox:
                if item.message_id == message_id:
                    return role.value, int(item.sequence)
        stored = self._inflight.get(message_id) or self._seen.get(message_id)
        if stored is not None:
            return stored.target_agent.value, int(stored.sequence)
        return None, None

    def _remove_pending_locked(self, message_id: str) -> bool:
        removed = False
        for inbox in self._inboxes.values():
            for index, item in enumerate(inbox):
                if item.message_id == message_id:
                    del inbox[index]
                    removed = True
                    break
        if self._inflight.pop(message_id, None) is not None:
            removed = True
        return removed

    def _enqueue_locked(self, message: AgentMessage) -> bool:
        role_key = message.target_agent.value
        inbox = self._inboxes[role_key]
        if any(item.message_id == message.message_id for item in inbox):
            return False
        if message.message_id in self._inflight:
            return False
        inbox.append(message)
        return True

    def _pending_sequences_locked(self, role_key: str) -> list[int]:
        """Sequences of messages for ``role_key`` still waiting to be consumed."""
        sequences = [int(item.sequence) for item in self._inboxes[role_key]]
        sequences.extend(
            int(item.sequence)
            for item in self._inflight.values()
            if item.target_agent.value == role_key
        )
        return sequences

    def _advance_frontier_locked(self, role_key: str) -> bool:
        """Move the per-role consumed mark to the highest safe (gap-free) value."""
        current = int(self._acked.get(role_key, 0))
        consumed = self._consumed_seqs.get(role_key) or set()
        pending = self._pending_sequences_locked(role_key)
        if pending:
            frontier = max(current, min(pending) - 1)
        else:
            frontier = max([current, *consumed]) if consumed else current
        changed = frontier > current
        self._acked[role_key] = frontier
        if consumed:
            self._consumed_seqs[role_key] = {seq for seq in consumed if seq > frontier}
        return changed

    # ------------------------------------------------------------------ #
    # Internals: ingest (load path)
    # ------------------------------------------------------------------ #
    def _ingest_locked(self, message: AgentMessage) -> None:
        self._sequence = max(self._sequence, int(message.sequence))
        self._seen[message.message_id] = message
        self._prune_seen_locked()
        self._record_dedupe_locked(message.dedupe_key())
        self._recent.append(message)
        self._published += 1

    def _parse_record(self, record: Any) -> AgentMessage | None:
        if not isinstance(record, Mapping):
            return None
        raw = record.get("message", record)
        try:
            message = AgentMessage.from_dict(raw if isinstance(raw, Mapping) else {})
        except (MultiAgentError, ValueError, TypeError):
            return None
        try:
            sequence = int(record.get("sequence", message.sequence))
        except (TypeError, ValueError):
            sequence = int(message.sequence)
        if sequence <= 0:
            sequence = int(message.sequence)
        if sequence <= 0:
            return None
        if sequence != message.sequence:
            message = replace(message, sequence=sequence)
        return message

    def _read_state_file(self) -> dict[str, Any]:
        path = self._state_path()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._load_errors += 1
            return {}
        return dict(data) if isinstance(data, Mapping) else {}

    def _apply_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        branch = state.get("branch_id")
        if isinstance(branch, str) and branch:
            self._branch_id = branch
        try:
            self._sequence = max(self._sequence, int(state.get("sequence") or 0))
        except (TypeError, ValueError):
            self._load_errors += 1
        acked = state.get("acked")
        if isinstance(acked, Mapping):
            for role in AgentRole:
                try:
                    value = int(acked.get(role.value) or 0)
                except (TypeError, ValueError):
                    self._load_errors += 1
                    continue
                self._acked[role.value] = max(int(self._acked.get(role.value, 0)), value)
        for counter in (
            "_duplicates_dropped",
            "_idempotent_replays",
            "_consumed",
        ):
            try:
                value = int(state.get(counter.lstrip("_")) or 0)
            except (TypeError, ValueError):
                continue
            setattr(self, counter, max(int(getattr(self, counter)), value))
        try:
            self._state_published = max(
                int(self._state_published), int(state.get("published") or 0)
            )
        except (TypeError, ValueError):
            self._load_errors += 1

    def _state_path(self) -> Path:
        return Path(f"{self._path}{STATE_SUFFIX}")

    def _save_state(self) -> None:
        if not self._persistent:
            return
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "branch_id": self._branch_id,
            "sequence": int(self._sequence),
            "acked": {role.value: int(self._acked.get(role.value, 0)) for role in AgentRole},
            "dedupe_keys": list(self._dedupe_order),
            "seen_ids": list(self._seen.keys()),
            "published": int(self._published),
            "duplicates_dropped": int(self._duplicates_dropped),
            "idempotent_replays": int(self._idempotent_replays),
            "consumed": int(self._consumed),
            "saved_at": float(self._clock()),
        }
        try:
            _atomic_write_json(self._state_path(), payload)
        except OSError as exc:
            raise MultiAgentError(
                f"failed to persist event bus state file {self._state_path()}: {exc}",
                details={"path": str(self._state_path())},
            ) from exc

    # ------------------------------------------------------------------ #
    # Internals: journal
    # ------------------------------------------------------------------ #
    def _append_record(self, message: AgentMessage) -> None:
        record = {"sequence": int(message.sequence), "message": message.to_dict()}
        try:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        except (TypeError, ValueError) as exc:  # pragma: no cover - payloads are JSON-able
            raise ProtocolError(f"event payload is not JSON serialisable: {exc}") from exc
        try:
            handle = self._journal
            if handle is None:
                path = Path(self._path)
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a", encoding="utf-8")
                self._journal = handle
            handle.write(line)
            self._pending_writes += 1
            self._flush_locked()
        except OSError as exc:
            self._close_journal()
            raise MultiAgentError(
                f"failed to persist event {message.message_id} to {self._path}: {exc}",
                details={"path": self._path, "message_id": message.message_id},
            ) from exc

    def _flush_locked(self, *, force: bool = False) -> None:
        handle = self._journal
        if handle is None:
            self._pending_writes = 0
            return
        if not force and self._pending_writes < self._flush_every:
            return
        try:
            handle.flush()
            if self._fsync:
                try:
                    os.fsync(handle.fileno())
                except OSError:  # pragma: no cover - fsync is best effort
                    pass
            self._pending_writes = 0
        except OSError as exc:
            self._close_journal()
            raise MultiAgentError(
                f"failed to flush event bus journal {self._path}: {exc}",
                details={"path": self._path},
            ) from exc

    def _close_journal(self) -> None:
        handle = self._journal
        self._journal = None
        self._pending_writes = 0
        if handle is None:
            return
        try:
            handle.flush()
        except OSError:  # pragma: no cover - closing must not mask errors
            pass
        try:
            handle.close()
        except OSError:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ #
    # Internals: bounded windows
    # ------------------------------------------------------------------ #
    def _prune_seen_locked(self) -> None:
        while len(self._seen) > self._idempotency_window:
            self._seen.popitem(last=False)

    def _record_dedupe_locked(self, key: str | None) -> None:
        if not key or self._dedupe_window <= 0:
            return
        self._dedupe_order.append(key)
        self._dedupe_counts[key] = self._dedupe_counts.get(key, 0) + 1
        while len(self._dedupe_order) > self._dedupe_window:
            oldest = self._dedupe_order.popleft()
            remaining = self._dedupe_counts.get(oldest, 0) - 1
            if remaining <= 0:
                self._dedupe_counts.pop(oldest, None)
            else:
                self._dedupe_counts[oldest] = remaining


__all__ = ["EventBus", "EventBusConfig"]
