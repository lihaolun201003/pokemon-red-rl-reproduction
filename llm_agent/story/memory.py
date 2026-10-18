"""Story-agent memory: own observation notebook + shared world knowledge.

The story agent reuses the existing :class:`llm_agent.memory.AgentMemory` for its
private notebook (dialogues read on screen, notes, open questions) and adds a
small, auditable layer on top:

* :class:`StoryMemory` keeps structured :class:`KnowledgeEntry` records with an
  explicit epistemic status (verified / inferred / hypothesis), the NPCs it has
  seen, and the task proposals it already made (so proposals are never
  duplicated across conversations);
* every knowledge record is mirrored into the shared world memory through a
  duck-typed adapter (:class:`SharedMemoryAdapter`) that calls the frozen
  ``remember(...)`` / ``search(...)`` interface when available;
* state is persisted atomically to ``story_memory.json`` next to the agent's own
  memory; a corrupt sidecar never crashes a run (it is ignored and rebuilt).

Honesty rule: a claim may only be stored as ``verified`` when the agent has
recorded an actually-observed dialogue/evidence reference.  A bare "verified"
claim from the model is downgraded to ``inferred``; inference is never promoted
to fact.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from llm_agent.config import MemoryConfig
from llm_agent.memory import AgentMemory
from llm_agent.multi_agent.contracts import (
    KNOWLEDGE_CATEGORIES,
    AgentRole,
    KnowledgeEntry,
    VerificationStatus,
    dedupe_key,
    now_ts,
)

__all__ = [
    "STORY_MEMORY_FILENAME",
    "STORY_KNOWLEDGE_CATEGORIES",
    "STORY_PROPOSAL_OWNERS",
    "STORY_PROPOSAL_TYPES",
    "STORY_PRIORITIES",
    "SharedMemoryAdapter",
    "StoryMemory",
]

#: Version of the story sidecar file.
STORY_MEMORY_VERSION = 1
STORY_MEMORY_FILENAME = "story_memory.json"

#: Bounded sections (a long run must never grow one prompt/file without limit).
MAX_KNOWLEDGE_ENTRIES = 200
MAX_DIALOGUE_EVIDENCE = 200
MAX_NPC_ENTRIES = 100
MAX_PROPOSALS = 64

#: Knowledge categories the story agent may write (subset of the frozen list).
STORY_KNOWLEDGE_CATEGORIES: tuple[str, ...] = (
    "npc",
    "interactive",
    "story_clue",
    "quest",
    "area",
    "other",
)

#: Task owners / task types / priorities accepted for story proposals.
STORY_PROPOSAL_OWNERS: tuple[str, ...] = ("navigation", "pokemon", "story")
STORY_PROPOSAL_TYPES: tuple[str, ...] = ("explore", "navigate", "investigate", "talk", "battle")
STORY_PRIORITIES: tuple[str, ...] = ("background", "low", "normal", "high", "critical")

_PROPOSAL_FIELDS: tuple[str, ...] = (
    "owner_agent",
    "task_type",
    "description",
    "priority",
    "payload",
)

_ADD_METHODS: tuple[str, ...] = ("add", "add_entry", "record", "record_knowledge", "upsert", "put")
_SEARCH_METHODS: tuple[str, ...] = ("search", "query", "find", "query_entries")


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #

def _clean(value: Any, limit: int = 0) -> str:
    """Collapse whitespace and optionally truncate a value."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if limit and limit > 0:
        text = text[: int(limit)]
    return text


def _norm(value: Any) -> str:
    """Case/space-insensitive normalisation used for de-duplication."""
    return _clean(value).casefold()


def _as_float(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _as_str_list(value: Any, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        items: Iterable[Any] = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    out: list[str] = []
    for item in items:
        text = _clean(item, 300)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _json_safe(value: Any) -> Any:
    """Return a JSON-serialisable copy of ``value`` (never raises)."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as UTF-8 JSON atomically (tmp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():  # pragma: no cover - only after a crash
                tmp.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Shared world memory adapter
# --------------------------------------------------------------------------- #

class SharedMemoryAdapter:
    """Best-effort bridge to the shared world memory.

    The frozen interface (owned by the shared-memory module) is::

        remember(category, content, *, source_agent=..., confidence=...,
                 verification_status=..., evidence_refs=(), subject="",
                 tags=(), branch_id="") -> KnowledgeEntry
        search(query="", *, categories=None, subject="", limit=12,
               include_historical=False, verified_only=False) -> list[KnowledgeEntry]

    Older/simpler variants (``add``/``add_entry``/``record``/``upsert`` taking a
    :class:`KnowledgeEntry`) are accepted as fallbacks so the agent can be wired
    before the shared-memory module lands.  Failures are recorded, never raised:
    losing a mirror write must not abort a conversation.
    """

    def __init__(self, shared: Any = None, *, clock: Any = None) -> None:
        self.shared = shared
        self._clock = clock or now_ts
        self.errors: list[str] = []

    @property
    def available(self) -> bool:
        return self.shared is not None

    def _record_error(self, action: str, exc: BaseException) -> None:
        self.errors.append(f"{action}: {type(exc).__name__}: {_clean(exc, 200)}")
        if len(self.errors) > 20:
            del self.errors[: len(self.errors) - 20]

    def remember(
        self,
        *,
        category: str,
        content: str,
        verification_status: VerificationStatus | str = VerificationStatus.HYPOTHESIS,
        confidence: float = 0.5,
        evidence_refs: Sequence[str] = (),
        subject: str = "",
        tags: Sequence[str] = (),
    ) -> Any:
        """Mirror one knowledge record into the shared world memory."""
        if self.shared is None:
            return None
        status = VerificationStatus.coerce(verification_status)
        refs = tuple(_as_str_list(evidence_refs, limit=8))
        tags_t = tuple(_as_str_list(tags, limit=8))
        remember = getattr(self.shared, "remember", None)
        if callable(remember):
            try:
                return remember(
                    category,
                    content,
                    source_agent=AgentRole.STORY,
                    confidence=float(confidence),
                    verification_status=status,
                    evidence_refs=refs,
                    subject=subject,
                    tags=tags_t,
                    branch_id=str(getattr(self.shared, "branch_id", "") or ""),
                )
            except TypeError:
                try:
                    return remember(
                        category,
                        content,
                        source_agent=AgentRole.STORY,
                        confidence=float(confidence),
                        verification_status=status,
                        evidence_refs=refs,
                    )
                except Exception as exc:  # noqa: BLE001 - mirror must not crash
                    self._record_error("remember", exc)
                    return None
            except Exception as exc:  # noqa: BLE001
                self._record_error("remember", exc)
                return None
        entry = KnowledgeEntry.create(
            category=category,
            content=content,
            source_agent=AgentRole.STORY,
            confidence=float(confidence),
            verification_status=status,
            evidence_refs=refs,
            subject=subject,
            tags=tags_t,
        )
        for name in _ADD_METHODS:
            func = getattr(self.shared, name, None)
            if not callable(func):
                continue
            try:
                return func(entry)
            except TypeError:
                try:
                    return func(entry.to_dict())
                except Exception as exc:  # noqa: BLE001
                    self._record_error(name, exc)
                    return None
            except Exception as exc:  # noqa: BLE001
                self._record_error(name, exc)
                return None
        # Last resort: plain container used by tests / very small integrations.
        for attr in ("entries", "knowledge"):
            container = getattr(self.shared, attr, None)
            if isinstance(container, list):
                container.append(entry.to_dict())
                return entry
        self._record_error("remember", ValueError("shared memory exposes no write method"))
        return None

    def search(self, query: str = "", *, limit: int = 10) -> list[Any]:
        """Read back shared entries (best effort; empty list when unavailable)."""
        if self.shared is None:
            return []
        for name in _SEARCH_METHODS:
            func = getattr(self.shared, name, None)
            if not callable(func):
                continue
            try:
                try:
                    return list(func(query, limit=limit) or [])
                except TypeError:
                    return list(func(query) or [])
            except Exception as exc:  # noqa: BLE001
                self._record_error(name, exc)
                return []
        return []


# --------------------------------------------------------------------------- #
# StoryMemory
# --------------------------------------------------------------------------- #

class StoryMemory:
    """The story agent's auditable notebook around an ``AgentMemory``.

    Parameters
    ----------
    root:
        Directory for the story sidecar file.  Defaults to the agent memory root.
    agent_memory:
        Existing :class:`llm_agent.memory.AgentMemory` (the director passes one);
        a private one is created from ``memory_config``/``root`` otherwise.
    shared_memory:
        Shared world memory object (frozen ``remember``/``search`` interface).
    save_on_write:
        Persist after every mutation (default) so a crash cannot lose notes.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        agent_memory: AgentMemory | Any | None = None,
        shared_memory: Any = None,
        memory_config: MemoryConfig | None = None,
        max_context_chars: int | None = None,
        clock: Any = None,
        save_on_write: bool = True,
    ) -> None:
        self._clock = clock or time.time
        self.save_on_write = bool(save_on_write)
        self.load_errors: list[str] = []
        self.last_save_error: str = ""

        if agent_memory is None:
            cfg = memory_config if memory_config is not None else MemoryConfig()
            if root is not None:
                cfg = replace(cfg, root=str(root))
            self.agent_memory = AgentMemory(cfg)
        else:
            self.agent_memory = agent_memory

        base = getattr(self.agent_memory, "root", None) or root
        self.root: Path | None = Path(base) if base is not None else None
        memory_cfg = getattr(self.agent_memory, "cfg", None)
        default_chars = int(getattr(memory_cfg, "max_context_chars", 3000) or 3000)
        self.max_context_chars = int(max_context_chars or default_chars)

        self.shared = SharedMemoryAdapter(shared_memory, clock=self._clock)
        self._knowledge: list[KnowledgeEntry] = []
        self._dialogues: list[dict[str, Any]] = []
        self._npcs: dict[str, dict[str, Any]] = {}
        self._proposals: list[dict[str, Any]] = []
        self._knowledge_keys: set[str] = set()
        self._proposal_keys: set[str] = set()
        self._load_sidecar()

    # -- paths / persistence --------------------------------------------- #

    @property
    def sidecar_path(self) -> Path | None:
        return (self.root / STORY_MEMORY_FILENAME) if self.root is not None else None

    def save(self) -> None:
        """Persist the sidecar and the underlying AgentMemory (best effort)."""
        try:
            if self.agent_memory is not None:
                save = getattr(self.agent_memory, "save", None)
                if callable(save):
                    save()
        except Exception as exc:  # noqa: BLE001 - persistence must not abort a run
            self.last_save_error = f"agent_memory save: {type(exc).__name__}: {_clean(exc, 160)}"
        self._write_sidecar()

    def _write_sidecar(self) -> None:
        path = self.sidecar_path
        if path is None:
            return
        payload = {
            "version": STORY_MEMORY_VERSION,
            "updated_at": float(self._clock()),
            "knowledge": [entry.to_dict() for entry in self._knowledge],
            "dialogues": list(self._dialogues),
            "npcs": {name: dict(value) for name, value in self._npcs.items()},
            "proposals": list(self._proposals),
        }
        try:
            _atomic_write_json(path, payload)
        except OSError as exc:  # pragma: no cover - disk-level failure
            self.last_save_error = f"sidecar write: {type(exc).__name__}: {exc}"

    def _load_sidecar(self) -> None:
        self._knowledge = []
        self._dialogues = []
        self._npcs = {}
        self._proposals = []
        self._knowledge_keys = set()
        self._proposal_keys = set()
        path = self.sidecar_path
        if path is None or not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            self.load_errors.append(f"story_memory.json unreadable ({type(exc).__name__}); rebuilt")
            return
        if not isinstance(raw, Mapping):
            self.load_errors.append("story_memory.json is not an object; rebuilt")
            return
        version = raw.get("version", STORY_MEMORY_VERSION)
        if version != STORY_MEMORY_VERSION:
            self.load_errors.append(f"unsupported story_memory.json version {version!r}; rebuilt")
            return
        for item in raw.get("knowledge") or []:
            try:
                entry = KnowledgeEntry.from_dict(item)
            except Exception:  # noqa: BLE001 - skip a single corrupt record
                continue
            if entry.dedupe_key in self._knowledge_keys:
                continue
            self._knowledge.append(entry)
            self._knowledge_keys.add(entry.dedupe_key)
        for item in raw.get("dialogues") or []:
            if not isinstance(item, Mapping):
                continue
            text = _clean(item.get("text"), 1200)
            if not text:
                continue
            self._dialogues.append(
                {
                    "speaker": _clean(item.get("speaker"), 120) or "NPC",
                    "text": text,
                    "evidence_ref": _clean(item.get("evidence_ref"), 400),
                    "frame_index": int(item.get("frame_index") or 0),
                }
            )
        for name, item in (raw.get("npcs") or {}).items():
            if not isinstance(item, Mapping):
                continue
            key = _clean(name, 120)
            if not key:
                continue
            self._npcs[key] = {
                "name": key,
                "notes": _clean(item.get("notes"), 600),
                "verified": bool(item.get("verified")),
                "evidence_refs": _as_str_list(item.get("evidence_refs"), 6),
                "mentions": max(1, int(item.get("mentions") or 1)),
                "updated_at": float(item.get("updated_at") or 0.0),
            }
        for item in raw.get("proposals") or []:
            if not isinstance(item, Mapping):
                continue
            proposal = self._normalise_proposal_record(item)
            if proposal is None or proposal["dedupe_key"] in self._proposal_keys:
                continue
            self._proposals.append(proposal)
            self._proposal_keys.add(proposal["dedupe_key"])

    def reload(self) -> None:
        """Re-read the sidecar (e.g. after a rollback to an older branch)."""
        self._load_sidecar()

    def snapshot(self) -> dict[str, Any]:
        """Serialisable snapshot for the multi-agent checkpoint manifest."""
        return {
            "version": STORY_MEMORY_VERSION,
            "knowledge": [entry.to_dict() for entry in self._knowledge],
            "dialogues": list(self._dialogues),
            "npcs": {name: dict(value) for name, value in self._npcs.items()},
            "proposals": list(self._proposals),
            "saved_at": float(self._clock()),
        }

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        """Restore a :meth:`snapshot` (never raises on a bad snapshot)."""
        if not isinstance(snapshot, Mapping):
            return
        if snapshot.get("version") not in (None, STORY_MEMORY_VERSION):
            return
        fake = {
            "version": STORY_MEMORY_VERSION,
            "knowledge": snapshot.get("knowledge") or [],
            "dialogues": snapshot.get("dialogues") or [],
            "npcs": snapshot.get("npcs") or {},
            "proposals": snapshot.get("proposals") or [],
        }
        path_backup = self.sidecar_path
        try:
            self._knowledge, self._dialogues, self._npcs, self._proposals = [], [], {}, []
            self._knowledge_keys, self._proposal_keys = set(), set()
            self._inflate(fake)
        finally:
            _ = path_backup

    def _inflate(self, raw: Mapping[str, Any]) -> None:
        """Populate in-memory state from a parsed sidecar document."""
        for item in raw.get("knowledge") or []:
            try:
                entry = KnowledgeEntry.from_dict(item) if not isinstance(
                    item, KnowledgeEntry
                ) else item
            except Exception:  # noqa: BLE001
                continue
            if entry.dedupe_key not in self._knowledge_keys:
                self._knowledge.append(entry)
                self._knowledge_keys.add(entry.dedupe_key)
        for item in raw.get("dialogues") or []:
            if isinstance(item, Mapping) and _clean(item.get("text")):
                self._dialogues.append(dict(item))
        for name, item in (raw.get("npcs") or {}).items():
            if isinstance(item, Mapping):
                self._npcs[str(name)] = dict(item)
        for item in raw.get("proposals") or []:
            if isinstance(item, Mapping):
                proposal = self._normalise_proposal_record(item)
                if proposal is not None and proposal["dedupe_key"] not in self._proposal_keys:
                    self._proposals.append(proposal)
                    self._proposal_keys.add(proposal["dedupe_key"])

    # -- dialogue evidence ----------------------------------------------- #

    def record_dialogue_text(
        self,
        speaker: str,
        text: str,
        *,
        evidence_ref: str = "",
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        """Record one dialogue line the agent actually read on screen."""
        clean_text = _clean(text, 1200)
        clean_speaker = _clean(speaker, 120) or "NPC"
        if not clean_text:
            return {}
        ref = _clean(evidence_ref, 400)
        key = (_norm(clean_speaker), _norm(clean_text))
        for item in self._dialogues:
            if (_norm(item.get("speaker")), _norm(item.get("text"))) == key:
                if ref:
                    item["evidence_ref"] = ref
                self._record_agent_dialogue(clean_speaker, clean_text, ref)
                self._persist()
                return dict(item)
        item = {
            "speaker": clean_speaker,
            "text": clean_text,
            "evidence_ref": ref,
            "frame_index": int(frame_index or 0),
        }
        self._dialogues.append(item)
        if len(self._dialogues) > MAX_DIALOGUE_EVIDENCE:
            del self._dialogues[: len(self._dialogues) - MAX_DIALOGUE_EVIDENCE]
        self._record_agent_dialogue(clean_speaker, clean_text, ref)
        self._persist()
        return dict(item)

    def _record_agent_dialogue(self, speaker: str, text: str, evidence_ref: str) -> None:
        record = getattr(self.agent_memory, "record_dialogue", None)
        if callable(record):
            try:
                record(speaker, text)
            except Exception as exc:  # noqa: BLE001 - notebook must not crash the loop
                self.last_save_error = f"record_dialogue: {type(exc).__name__}: {_clean(exc, 120)}"

    def dialogues(self) -> list[dict[str, Any]]:
        """Observed dialogue lines (oldest first)."""
        return [dict(item) for item in self._dialogues]

    def has_observed_dialogue(self) -> bool:
        """True once the agent has transcribed at least one on-screen line."""
        return bool(self._dialogues)

    # -- knowledge ------------------------------------------------------- #

    def record_knowledge(
        self,
        *,
        category: str,
        content: str,
        verification_status: VerificationStatus | str = VerificationStatus.HYPOTHESIS,
        confidence: float = 0.5,
        evidence_refs: Sequence[str] = (),
        subject: str = "",
        tags: Sequence[str] = (),
        npc_name: str = "",
        dialogue_text: str = "",
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        """Record one clue/person/interpretation with its honest epistemic status.

        Returns ``{"entry", "created", "downgraded", "reason"}``.  A ``verified``
        claim without an observed dialogue/frame reference is downgraded to
        ``inferred`` (inference is never promoted to fact).
        """
        clean_category = _clean(category, 40).lower()
        if clean_category not in STORY_KNOWLEDGE_CATEGORIES:
            clean_category = clean_category if clean_category in KNOWLEDGE_CATEGORIES else "other"
        clean_content = _clean(content, 1200)
        if not clean_content:
            return {"entry": {}, "created": False, "downgraded": False, "reason": "empty content"}
        status = VerificationStatus.coerce(verification_status)
        refs = _as_str_list(evidence_refs, limit=6)
        clean_subject = _clean(subject or npc_name, 120)

        if dialogue_text:
            self.record_dialogue_text(
                npc_name or "NPC",
                dialogue_text,
                evidence_ref=refs[0] if refs else "",
                frame_index=frame_index,
            )
        downgraded = False
        reason = ""
        if status is VerificationStatus.VERIFIED:
            # A verified note must be grounded in a dialogue line the agent
            # actually transcribed from the screen (plus its evidence reference).
            has_observation = bool(self._dialogues)
            if not refs or not has_observation:
                status = VerificationStatus.INFERRED
                downgraded = True
                reason = "a verified note needs the visible dialogue text and its evidence reference"
        if status is VerificationStatus.VERIFIED:
            confidence = max(_as_float(confidence), 0.8)
        elif status is VerificationStatus.INFERRED:
            confidence = min(max(_as_float(confidence), 0.3), 0.85)

        entry = KnowledgeEntry.create(
            category=clean_category,
            content=clean_content,
            source_agent=AgentRole.STORY,
            confidence=confidence,
            verification_status=status,
            evidence_refs=refs,
            subject=clean_subject,
            tags=tags,
        )
        # Mirror into the shared world knowledge base first: even if this is a
        # duplicate locally, the director may still be missing it.
        self.shared.remember(
            category=entry.category,
            content=entry.content,
            verification_status=entry.verification_status,
            confidence=entry.confidence,
            evidence_refs=entry.evidence_refs,
            subject=entry.subject,
            tags=entry.tags,
        )
        existing = self._find_knowledge(entry.dedupe_key)
        if existing is not None:
            return {
                "entry": existing.to_dict(),
                "created": False,
                "downgraded": downgraded,
                "reason": reason,
            }
        self._knowledge.append(entry)
        self._knowledge_keys.add(entry.dedupe_key)
        if len(self._knowledge) > MAX_KNOWLEDGE_ENTRIES:
            removed = self._knowledge[: len(self._knowledge) - MAX_KNOWLEDGE_ENTRIES]
            del self._knowledge[: len(self._knowledge) - MAX_KNOWLEDGE_ENTRIES]
            self._knowledge_keys = {item.dedupe_key for item in self._knowledge}
            _ = removed
        self._persist()
        return {
            "entry": entry.to_dict(),
            "created": True,
            "downgraded": downgraded,
            "reason": reason,
        }

    def _find_knowledge(self, key: str) -> KnowledgeEntry | None:
        for entry in self._knowledge:
            if entry.dedupe_key == key:
                return entry
        return None

    def knowledge(self) -> list[KnowledgeEntry]:
        """All knowledge records (oldest first)."""
        return list(self._knowledge)

    @staticmethod
    def public_knowledge(entry: KnowledgeEntry | Mapping[str, Any]) -> dict[str, Any]:
        """Public result-dict view of one knowledge record."""
        if isinstance(entry, KnowledgeEntry):
            return {
                "category": entry.category,
                "content": entry.content,
                "verification_status": entry.verification_status.value,
                "confidence": float(entry.confidence),
                "evidence_refs": list(entry.evidence_refs),
            }
        data = dict(entry or {})
        return {
            "category": str(data.get("category") or "other"),
            "content": str(data.get("content") or ""),
            "verification_status": str(data.get("verification_status") or "hypothesis"),
            "confidence": float(data.get("confidence") or 0.0),
            "evidence_refs": [str(ref) for ref in data.get("evidence_refs") or []],
        }

    # -- NPCs ------------------------------------------------------------ #

    def record_npc(
        self,
        name: str,
        notes: str = "",
        *,
        verified: bool = False,
        evidence_ref: str = "",
    ) -> dict[str, Any]:
        """Create or update one person the agent has seen/read on screen."""
        clean_name = _clean(name, 120)
        if not clean_name:
            return {}
        now = float(self._clock())
        key = clean_name
        existing = None
        for candidate, value in self._npcs.items():
            if _norm(candidate) == _norm(clean_name):
                existing = value
                key = candidate
                break
        if existing is not None:
            existing["mentions"] = int(existing.get("mentions") or 1) + 1
            existing["verified"] = bool(existing.get("verified")) or bool(verified)
            if _clean(notes):
                existing["notes"] = _clean(notes, 600)
            refs = _as_str_list(existing.get("evidence_refs"), 6)
            clean_ref = _clean(evidence_ref, 400)
            if clean_ref and clean_ref not in refs:
                refs.append(clean_ref)
            existing["evidence_refs"] = refs
            existing["updated_at"] = now
            record = existing
        else:
            record = {
                "name": clean_name,
                "notes": _clean(notes, 600),
                "verified": bool(verified),
                "evidence_refs": _as_str_list([evidence_ref], 6),
                "mentions": 1,
                "updated_at": now,
            }
            self._npcs[clean_name] = record
            if len(self._npcs) > MAX_NPC_ENTRIES:
                oldest = min(self._npcs.values(), key=lambda item: float(item.get("updated_at") or 0.0))
                self._npcs.pop(str(oldest.get("name")), None)
        _ = key
        self._persist()
        return dict(record)

    def npcs(self) -> list[dict[str, Any]]:
        """Public NPC view (``name``/``notes``/``verified``) in stable order."""
        out = []
        for item in self._npcs.values():
            out.append(
                {
                    "name": str(item.get("name") or ""),
                    "notes": str(item.get("notes") or ""),
                    "verified": bool(item.get("verified")),
                }
            )
        return out

    # -- task proposals -------------------------------------------------- #

    def _normalise_proposal_record(self, item: Mapping[str, Any]) -> dict[str, Any] | None:
        owner = _clean(item.get("owner_agent"), 40).lower()
        task_type = _clean(item.get("task_type"), 40).lower()
        description = _clean(item.get("description"), 300)
        priority = _clean(item.get("priority"), 20).lower() or "normal"
        if owner not in STORY_PROPOSAL_OWNERS or task_type not in STORY_PROPOSAL_TYPES:
            return None
        if priority not in STORY_PRIORITIES:
            priority = "normal"
        if not description:
            return None
        payload = item.get("payload")
        payload = dict(payload) if isinstance(payload, Mapping) else {}
        key = str(item.get("dedupe_key") or "") or dedupe_key(
            "story_proposal", description, extra={"owner": owner, "type": task_type}
        )
        return {
            "proposal_id": str(item.get("proposal_id") or f"prop_{uuid.uuid4().hex[:12]}"),
            "owner_agent": owner,
            "task_type": task_type,
            "description": description,
            "priority": priority,
            "payload": _json_safe(payload),
            "source": _clean(item.get("source"), 40) or "story_agent",
            "created_at": float(item.get("created_at") or self._clock()),
            "dedupe_key": key,
        }

    def add_proposal(
        self,
        *,
        owner_agent: str,
        task_type: str,
        description: str,
        priority: str = "normal",
        payload: Mapping[str, Any] | None = None,
        source: str = "story_agent",
    ) -> tuple[dict[str, Any], bool]:
        """Record a task proposal; returns ``(proposal, created)`` and de-dupes."""
        record = self._normalise_proposal_record(
            {
                "owner_agent": owner_agent,
                "task_type": task_type,
                "description": description,
                "priority": priority,
                "payload": payload or {},
                "source": source,
            }
        )
        if record is None:
            return {}, False
        key = record["dedupe_key"]
        for item in self._proposals:
            if item.get("dedupe_key") == key:
                return dict(item), False
        self._proposals.append(record)
        self._proposal_keys.add(key)
        if len(self._proposals) > MAX_PROPOSALS:
            del self._proposals[: len(self._proposals) - MAX_PROPOSALS]
            self._proposal_keys = {item["dedupe_key"] for item in self._proposals}
        self._persist()
        return dict(record), True

    @staticmethod
    def public_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
        """Result-dict view of a proposal (the five frozen keys only)."""
        return {
            "owner_agent": str(proposal.get("owner_agent") or ""),
            "task_type": str(proposal.get("task_type") or ""),
            "description": str(proposal.get("description") or ""),
            "priority": str(proposal.get("priority") or "normal"),
            "payload": _json_safe(proposal.get("payload") or {}),
        }

    def proposals(self) -> list[dict[str, Any]]:
        """All recorded proposals (public five-key view, oldest first)."""
        return [self.public_proposal(item) for item in self._proposals]

    def proposal_records(self) -> list[dict[str, Any]]:
        """Internal proposal records (including ids/dedupe keys)."""
        return [dict(item) for item in self._proposals]

    def has_proposal(self, description: str) -> bool:
        """True when an equivalent proposal description already exists."""
        needle = _norm(description)
        return any(_norm(item.get("description")) == needle for item in self._proposals)

    # -- retrieval / rendering ------------------------------------------- #

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Keyword search over the story notes then the private notebook."""
        count = max(1, int(limit))
        needle = _norm(query)
        results: list[dict[str, Any]] = []
        if needle:
            tokens = [token for token in needle.split() if token]
            for entry in reversed(self._knowledge):
                haystack = _norm(f"{entry.category} {entry.content} {entry.subject}")
                score = sum(1 for token in tokens if token in haystack)
                if score:
                    item = self.public_knowledge(entry)
                    item["category_of_record"] = "knowledge"
                    item["score"] = score
                    results.append(item)
        else:
            for entry in self._knowledge[-count:]:
                item = self.public_knowledge(entry)
                item["category_of_record"] = "knowledge"
                item["score"] = 0
                results.append(item)
        notebook = getattr(self.agent_memory, "search", None)
        if callable(notebook) and len(results) < count:
            try:
                for item in notebook(query, limit=count - len(results)) or []:
                    data = dict(item)
                    data["category_of_record"] = "notebook"
                    results.append(data)
            except Exception as exc:  # noqa: BLE001 - search is best effort
                self.load_errors.append(f"notebook search: {type(exc).__name__}: {_clean(exc, 120)}")
        return results[:count]

    def stats(self) -> dict[str, Any]:
        """Compact counters for the director/status endpoint."""
        verified = sum(
            1 for entry in self._knowledge if entry.verification_status is VerificationStatus.VERIFIED
        )
        out: dict[str, Any] = {
            "knowledge": len(self._knowledge),
            "verified": verified,
            "inferred": sum(
                1
                for entry in self._knowledge
                if entry.verification_status is VerificationStatus.INFERRED
            ),
            "hypotheses": sum(
                1
                for entry in self._knowledge
                if entry.verification_status is VerificationStatus.HYPOTHESIS
            ),
            "npcs": len(self._npcs),
            "proposals": len(self._proposals),
            "dialogues": len(self._dialogues),
            "root": str(self.root) if self.root is not None else "",
            "shared_memory": self.shared.available,
            "shared_errors": list(self.shared.errors[-3:]),
            "load_errors": list(self.load_errors[-3:]),
            "last_save_error": self.last_save_error,
        }
        notebook = getattr(self.agent_memory, "stats", None)
        if callable(notebook):
            try:
                out["notebook"] = dict(notebook() or {})
            except Exception:  # noqa: BLE001
                out["notebook"] = {}
        return out

    def render_context(self, query: str | None = None, *, max_chars: int | None = None) -> str:
        """Render the bounded Markdown block injected into the prompt.

        Deterministic for a given memory document.  Priority: verified facts,
        then the people the agent met, then inferences/hypotheses, then the
        private notebook (delegated to :meth:`AgentMemory.render_context`).
        """
        budget = int(max_chars if max_chars is not None else self.max_context_chars)
        budget = max(0, budget)
        if budget <= 0:
            return ""
        lines: list[str] = []
        used = 0

        def add(text: str) -> bool:
            nonlocal used
            value = str(text).strip()
            if not value:
                return False
            overhead = len(value) + (1 if lines else 0)
            if used + overhead > budget:
                return False
            lines.append(value)
            used += overhead
            return True

        verified = [
            entry for entry in self._knowledge if entry.verification_status is VerificationStatus.VERIFIED
        ]
        uncertain = [
            entry
            for entry in self._knowledge
            if entry.verification_status is not VerificationStatus.VERIFIED
        ]
        if verified and add("## STORY NOTES - CONFIRMED ON SCREEN"):
            for entry in verified[-10:]:
                refs = f" [evidence: {', '.join(entry.evidence_refs[:2])}]" if entry.evidence_refs else ""
                if not add(f"- {entry.content}{refs}"):
                    break
        npcs = list(self._npcs.values())
        if npcs and add("## PEOPLE SEEN"):
            for item in npcs[-10:]:
                state = "spoken with" if item.get("verified") else "mentioned / not confirmed"
                notes = f" - {item.get('notes')}" if item.get("notes") else ""
                if not add(f"- {item.get('name')} ({state}){notes}"):
                    break
        if uncertain and add("## IDEAS AND GUESSES - NOT CONFIRMED"):
            for entry in uncertain[-10:]:
                label = "INFERRED" if entry.verification_status is VerificationStatus.INFERRED else "HYPOTHESIS"
                if not add(f"- [{label}] {entry.content}"):
                    break
        remaining = budget - used - (2 if lines else 0)
        if remaining > 120:
            render = getattr(self.agent_memory, "render_context", None)
            if callable(render):
                try:
                    text = render(query, max_chars=remaining)
                except TypeError:
                    try:
                        text = render(max_chars=remaining)
                    except Exception:  # noqa: BLE001
                        text = ""
                except Exception:  # noqa: BLE001
                    text = ""
                text = str(text or "").strip()
                if text:
                    add("")
                    # Rebuild with the separator included so length stays correct.
                    if lines and not lines[-1]:
                        lines.pop()
                    add("## MY EARLIER NOTES")
                    for line in text.splitlines():
                        if not add(line):
                            break
        return "\n".join(lines)[:budget]

    # -- internals ------------------------------------------------------- #

    def _persist(self) -> None:
        if self.save_on_write:
            self.save()
