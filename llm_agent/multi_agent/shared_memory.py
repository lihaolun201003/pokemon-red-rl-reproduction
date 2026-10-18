"""Shared world knowledge base for the Pokémon Red multi-agent system.

Every agent (navigation, story, pokémon and the director) reads and writes the
same :class:`~llm_agent.multi_agent.contracts.KnowledgeEntry` records through
:class:`SharedWorldMemory`.  This is the **cross-agent learning layer**: what one
agent observes on screen becomes a labelled, auditable fact for the others.

Epistemic rules (enforced here, not just by convention)
-------------------------------------------------------
* Only a directly observed fact **with at least one evidence reference** may be
  ``VERIFIED``; a verification request without evidence is rejected and a
  ``VERIFIED`` record submitted without evidence is downgraded to ``INFERRED``.
* Inferences and hypotheses are labelled and their confidence is capped below
  verified facts, so a prompt can never present a guess as a fact.
* After a rollback :meth:`SharedWorldMemory.set_branch` marks every entry of the
  previous branch ``HISTORICAL``.  Learning is *preserved* (never deleted) but
  it is no longer trusted in the new branch and is rendered separately.
* No hidden state and no external game knowledge may be stored: entries are
  derivations of what an agent actually saw (screens, dialogue, own attempts).

Scope note
----------
The navigation :class:`~llm_agent.navigation.world_model.WorldModel` remains the
authoritative *spatial* store (grids, cells, pose, area graph).  This module only
stores semantic knowledge and never duplicates a coordinate database: an entry's
``subject`` is an opaque label/area id supplied by the caller.

Persistence
-----------
One atomic JSON document (``<root>/shared_memory.json``) with tolerant loading:
corrupt files are backed up and replaced by an empty knowledge base.  All writes
go through ``*.tmp`` + :func:`os.replace`.

Memory is bounded: ``max_entries`` evicts the least valuable/stalest records
(refuted first, verified facts last) and the prompt renderers obey a **strict**
character budget.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from llm_agent.multi_agent.contracts import (
    AgentRole,
    KNOWLEDGE_CATEGORIES,
    KnowledgeEntry,
    KnowledgeError,
    MultiAgentError,
    ProtocolError,
    VerificationStatus,
    dedupe_key,
    now_ts,
)

__all__ = [
    "SHARED_MEMORY_FORMAT",
    "SHARED_MEMORY_VERSION",
    "SHARED_MEMORY_FILENAME",
    "SharedMemoryConfig",
    "SharedWorldMemory",
]

#: Identifier written into the persisted document.
SHARED_MEMORY_FORMAT = "multi_agent.shared_memory"
#: Schema version of the persisted document.
SHARED_MEMORY_VERSION = 1
#: Default file name used when ``root`` points at a directory.
SHARED_MEMORY_FILENAME = "shared_memory.json"
#: Maximum number of audit events kept in memory/snapshots.
MAX_AUDIT_EVENTS = 200
#: Hard per-line cap used when rendering knowledge into a prompt.
MAX_PROMPT_LINE_CHARS = 260
#: Subject labels are short identifiers, not prose.
MAX_SUBJECT_CHARS = 120
#: Tag length cap.
MAX_TAG_CHARS = 60

#: Confidence cap per epistemic status (verified facts are the only 1.0-class).
_CONFIDENCE_CAPS: dict[VerificationStatus, float] = {
    VerificationStatus.HYPOTHESIS: 0.60,
    VerificationStatus.INFERRED: 0.85,
}
#: Verified facts are always high-confidence (>= this value).
_VERIFIED_MIN_CONFIDENCE = 0.9
#: Status rank used when merging observations (higher = more trusted).
_STATUS_RANK: dict[VerificationStatus, int] = {
    VerificationStatus.HYPOTHESIS: 1,
    VerificationStatus.INFERRED: 2,
    VerificationStatus.VERIFIED: 3,
    VerificationStatus.HISTORICAL: 0,
    VerificationStatus.REFUTED: 0,
}
#: Order in which knowledge groups are rendered into a prompt.
_GROUP_ORDER: tuple[VerificationStatus, ...] = (
    VerificationStatus.VERIFIED,
    VerificationStatus.INFERRED,
    VerificationStatus.HYPOTHESIS,
    VerificationStatus.HISTORICAL,
    VerificationStatus.REFUTED,
)
#: Human-readable heading per knowledge group (no game knowledge here).
_GROUP_TITLES: dict[VerificationStatus, str] = {
    VerificationStatus.VERIFIED: "FACTS (directly observed, verified):",
    VerificationStatus.INFERRED: "INFERENCES (derived, unverified):",
    VerificationStatus.HYPOTHESIS: "HYPOTHESES (unconfirmed guesses):",
    VerificationStatus.HISTORICAL: "HISTORICAL (previous branch, do NOT trust):",
    VerificationStatus.REFUTED: "REFUTED (observed contradictions, never repeat):",
}
#: Maximum number of entries rendered per group.
_GROUP_CAPS: dict[VerificationStatus, int] = {
    VerificationStatus.VERIFIED: 8,
    VerificationStatus.INFERRED: 6,
    VerificationStatus.HYPOTHESIS: 5,
    VerificationStatus.HISTORICAL: 3,
    VerificationStatus.REFUTED: 3,
}
#: Categories prioritised in each agent's short brief (ordered by relevance).
_AGENT_CATEGORIES: dict[AgentRole, tuple[str, ...]] = {
    AgentRole.NAVIGATION: ("area", "connection", "route", "hazard", "interactive"),
    AgentRole.STORY: ("npc", "story_clue", "quest", "interactive", "area"),
    AgentRole.POKEMON: ("battle_lesson", "pokemon", "hazard", "system"),
    AgentRole.DIRECTOR: ("system", "quest", "story_clue", "area", "hazard"),
    AgentRole.SYSTEM: ("system", "hazard", "quest"),
}
#: Eviction order: the lowest value class is discarded first.
_EVICTION_ORDER: tuple[VerificationStatus, ...] = (
    VerificationStatus.REFUTED,
    VerificationStatus.HYPOTHESIS,
    VerificationStatus.INFERRED,
    VerificationStatus.HISTORICAL,
    VerificationStatus.VERIFIED,
)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "i",
        "in", "is", "it", "me", "my", "of", "on", "or", "that", "the", "this",
        "to", "was", "we", "were", "with", "you", "your",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def _clean(text: Any, limit: int = 0) -> str:
    """Collapse whitespace and hard-truncate ``text`` to ``limit`` characters."""
    if text is None:
        return ""
    value = _WS_RE.sub(" ", str(text)).strip()
    if limit and limit > 0:
        value = value[: int(limit)]
    return value


def _norm(text: Any) -> str:
    """Case/space-insensitive comparison key."""
    return _clean(text).casefold()


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort int coercion (never raises)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort finite float coercion (never raises)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if result != result:  # NaN
        return float(default)
    return result


def _tokenize(text: Any) -> list[str]:
    """Deterministic keyword tokens for retrieval (deduped, stopwords removed)."""
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text).casefold()):
        if raw in _STOPWORDS:
            continue
        if len(raw) < 2 and raw.isascii():
            continue
        if raw in seen:
            continue
        seen.add(raw)
        tokens.append(raw)
    return tokens


def _atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 ``text`` to ``path`` atomically (tmp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _backup_corrupt(path: Path, *, now: float) -> str | None:
    """Move a corrupt persistence file to ``<stem>.corrupt-<stamp><suffix>``."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    stamp += f"_{int(now * 1_000_000) % 1_000_000:06d}"
    target = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
    attempt = 1
    while target.exists():
        target = path.with_name(f"{path.stem}.corrupt-{stamp}_{attempt}{path.suffix}")
        attempt += 1
    try:
        os.replace(path, target)
        return str(target)
    except OSError:  # pragma: no cover - disk-level failure
        return None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class SharedMemoryConfig:
    """Bounds of the shared knowledge base and its prompt renderers."""

    #: Persistence root; empty string disables persistence.
    root: str = ""
    #: Hard cap on the number of stored entries (bounded memory).
    max_entries: int = 3000
    #: Strict character budget of :meth:`SharedWorldMemory.render_for_prompt`.
    max_prompt_chars: int = 3000
    #: Maximum number of entries accepted into one prompt.
    max_entries_per_prompt: int = 24
    #: Default retrieval limit of :meth:`SharedWorldMemory.search`.
    retrieval_limit: int = 12
    #: Longest stored knowledge content (characters).
    max_content_chars: int = 800


# --------------------------------------------------------------------------- #
# SharedWorldMemory
# --------------------------------------------------------------------------- #

class SharedWorldMemory:
    """Thread-safe, bounded, auditable shared world knowledge base.

    Parameters
    ----------
    config:
        :class:`SharedMemoryConfig`; a default instance is created when omitted.
    root:
        Optional persistence root (overrides ``config.root`` when given).  A
        directory is interpreted as ``<root>/shared_memory.json``.
    clock:
        Injectable ``() -> float`` wall clock (deterministic tests).
    branch_id:
        Current checkpoint branch.  New entries inherit it unless they override
        it; :meth:`set_branch` marks the previous branch's entries
        ``HISTORICAL``.
    """

    def __init__(
        self,
        config: SharedMemoryConfig | None = None,
        *,
        root: str | Path = "",
        clock: Callable[[], float] | None = None,
        branch_id: str = "",
    ) -> None:
        self._config = config or SharedMemoryConfig()
        self._clock: Callable[[], float] = clock or now_ts
        self._branch_id = str(branch_id or "")
        self._lock = threading.RLock()

        self._entries: dict[str, KnowledgeEntry] = {}
        #: (branch_id, dedupe_key) -> knowledge_id, only for trusted live entries
        self._dedupe: dict[tuple[str, str], str] = {}
        self._audit: list[dict[str, Any]] = []
        self._version = 1
        self._downgraded = 0
        self._evicted = 0

        root_text = str(root or self._config.root or "").strip()
        self._root: Path | None = Path(root_text) if root_text else None
        self.load_errors: list[str] = []
        self.save_errors: list[str] = []
        self.recovered_from: str | None = None
        self.last_saved_at: float = 0.0

        if self._path is not None and self._path.is_file():
            self.load()

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> SharedMemoryConfig:
        """The configuration in use."""
        return self._config

    @property
    def version(self) -> int:
        """Monotonically increasing state version (bumped on every mutation)."""
        return self._version

    @property
    def branch_id(self) -> str:
        """Current checkpoint branch id (``""`` when unscoped)."""
        return self._branch_id

    @property
    def root(self) -> Path | None:
        """Persistence root (``None`` when persistence is disabled)."""
        return self._root

    @property
    def path(self) -> Path | None:
        """Resolved persistence file path (``None`` when disabled)."""
        if self._root is None:
            return None
        return self._root / SHARED_MEMORY_FILENAME

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, knowledge_id: object) -> bool:
        with self._lock:
            return str(knowledge_id) in self._entries

    # ------------------------------------------------------------------ #
    # Branch scoping / rollback
    # ------------------------------------------------------------------ #

    def set_branch(
        self,
        new_branch_id: str,
        *,
        previous_branch_id: str = "",
        reason: str = "",
    ) -> int:
        """Switch branch after a rollback; returns how many entries it marked.

        Every entry created on ``previous_branch_id`` (or the current branch when
        not given) becomes :data:`VerificationStatus.HISTORICAL`: the learning is
        preserved verbatim but no longer trusted in the new branch.  Entries of
        the new branch are left untouched.
        """
        with self._lock:
            old = str(previous_branch_id or self._branch_id or "")
            new = str(new_branch_id or "")
            marked = 0
            now = self._now()
            if old and old != new:
                for entry in self._entries.values():
                    if entry.branch_id != old:
                        continue
                    if entry.verification_status is VerificationStatus.HISTORICAL:
                        continue
                    entry.verification_status = VerificationStatus.HISTORICAL
                    entry.updated_at = now
                    entry.revision = int(entry.revision) + 1
                    marked += 1
            self._branch_id = new
            self._rebuild_index()
            self._version += 1
            self._audit_event(
                "set_branch",
                previous_branch_id=old,
                branch_id=new,
                marked=int(marked),
                reason=_clean(reason),
            )
            return marked

    # ------------------------------------------------------------------ #
    # Writing knowledge
    # ------------------------------------------------------------------ #

    def remember(
        self,
        category: str,
        content: str,
        *,
        source_agent: AgentRole | str,
        confidence: float = 0.5,
        verification_status: VerificationStatus | str = VerificationStatus.HYPOTHESIS,
        evidence_refs: Iterable[str] | str = (),
        subject: str = "",
        tags: Iterable[str] | str = (),
        branch_id: str = "",
    ) -> KnowledgeEntry:
        """Store one observation/inference/hypothesis, de-duplicating exact repeats.

        A duplicate is the same ``(category, content, subject)`` in the same
        branch: its confidence is raised toward the new value, the evidence
        references are merged and ``updated_at`` is refreshed — the existing
        entry (same ``knowledge_id``) is returned instead of a second record.
        """
        with self._lock:
            cat = self._clean_category(category)
            text = _clean(content, self.config.max_content_chars)
            if not text:
                raise KnowledgeError("knowledge content must not be empty")
            source = self._coerce_role(source_agent)
            status = self._coerce_status(verification_status)
            refs = self._clean_refs(evidence_refs)
            status = self._sanitize_status(status, refs)
            conf = self._cap_confidence(confidence, status)
            subject_text = _clean(subject, MAX_SUBJECT_CHARS)
            tag_list = self._clean_tags(tags)
            branch = str(branch_id or self._branch_id or "")

            key = dedupe_key(cat, text, extra={"subject": subject_text})
            existing = self._find_duplicate(branch, key)
            if existing is not None:
                self._merge_into(
                    existing,
                    confidence=conf,
                    status=status,
                    evidence_refs=refs,
                    subject=subject_text,
                    tags=tag_list,
                )
                self._version += 1
                return copy.deepcopy(existing)

            try:
                entry = KnowledgeEntry.create(
                    category=cat,
                    content=text,
                    source_agent=source,
                    confidence=conf,
                    verification_status=status,
                    branch_id=branch,
                    evidence_refs=refs,
                    subject=subject_text,
                    tags=tag_list,
                    created_at=self._now(),
                )
            except (ProtocolError, KnowledgeError) as exc:
                raise KnowledgeError(str(exc)) from exc
            self._entries[entry.knowledge_id] = entry
            self._index_entry(entry)
            self._version += 1
            self._audit_event(
                "remember",
                knowledge_id=entry.knowledge_id,
                category=cat,
                status=status.value,
            )
            self._enforce_limits()
            return copy.deepcopy(entry)

    def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Validate and store a pre-built entry (merging exact duplicates).

        Invalid records are rejected with :class:`KnowledgeError`; a record with
        the same ``knowledge_id`` already present is updated in place (confidence
        raised, evidence merged), so re-importing a snapshot is idempotent.
        """
        with self._lock:
            if not isinstance(entry, KnowledgeEntry):
                raise KnowledgeError("add() requires a KnowledgeEntry instance")
            try:
                normalized = KnowledgeEntry.from_dict(entry.to_dict())
            except (MultiAgentError, TypeError, ValueError) as exc:
                raise KnowledgeError(f"invalid knowledge entry: {exc}") from exc

            cat = self._clean_category(normalized.category)
            text = _clean(normalized.content, self.config.max_content_chars)
            if not text:
                raise KnowledgeError("knowledge content must not be empty")
            refs = self._clean_refs(normalized.evidence_refs)
            status = self._sanitize_status(normalized.verification_status, refs)
            conf = self._cap_confidence(normalized.confidence, status)
            subject_text = _clean(normalized.subject, MAX_SUBJECT_CHARS)
            tag_list = self._clean_tags(normalized.tags)
            branch = str(normalized.branch_id or self._branch_id or "")
            now = self._now()

            existing = self._entries.get(normalized.knowledge_id)
            if existing is not None:
                self._unindex_entry(existing)
                existing.category = cat
                existing.content = text
                existing.evidence_refs = refs
                existing.confidence = max(float(existing.confidence), conf)
                existing.confidence = self._cap_confidence(existing.confidence, status)
                if _STATUS_RANK.get(status, 0) > _STATUS_RANK.get(existing.verification_status, 0):
                    existing.verification_status = status
                if subject_text:
                    existing.subject = subject_text
                existing.tags = sorted({*existing.tags, *tag_list})
                existing.branch_id = branch
                existing.updated_at = now
                existing.revision = int(existing.revision) + 1
                self._index_entry(existing)
                self._version += 1
                return copy.deepcopy(existing)

            key = dedupe_key(cat, text, extra={"subject": subject_text})
            duplicate = self._find_duplicate(branch, key)
            if duplicate is not None:
                self._merge_into(
                    duplicate,
                    confidence=conf,
                    status=status,
                    evidence_refs=refs,
                    subject=subject_text,
                    tags=tag_list,
                )
                self._version += 1
                return copy.deepcopy(duplicate)

            normalized.category = cat
            normalized.content = text
            normalized.evidence_refs = refs
            normalized.confidence = conf
            normalized.verification_status = status
            normalized.subject = subject_text
            normalized.tags = tag_list
            normalized.branch_id = branch
            normalized.created_at = float(normalized.created_at or now)
            normalized.updated_at = max(float(normalized.updated_at or now), normalized.created_at)
            try:
                normalized.validate()
            except (ProtocolError, KnowledgeError, ValueError) as exc:
                raise KnowledgeError(f"invalid knowledge entry: {exc}") from exc
            self._entries[normalized.knowledge_id] = normalized
            self._index_entry(normalized)
            self._version += 1
            self._audit_event(
                "add",
                knowledge_id=normalized.knowledge_id,
                category=cat,
                status=status.value,
            )
            self._enforce_limits()
            return copy.deepcopy(normalized)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def get(self, knowledge_id: str) -> KnowledgeEntry | None:
        """Return a defensive copy of one entry (``None`` when unknown)."""
        with self._lock:
            entry = self._entries.get(str(knowledge_id or ""))
            return copy.deepcopy(entry) if entry is not None else None

    def search(
        self,
        query: str = "",
        *,
        categories: Iterable[str] | None = None,
        subject: str = "",
        limit: int = 12,
        include_historical: bool = False,
        verified_only: bool = False,
    ) -> list[KnowledgeEntry]:
        """Keyword + recency retrieval with explicit epistemic filtering.

        By default ``HISTORICAL`` and ``REFUTED`` records and superseded records
        are hidden; ``include_historical=True`` shows them too (clearly labelled
        by their status).  An empty query returns the most recently updated
        entries; a non-empty query returns only keyword matches, best first.
        """
        with self._lock:
            cat_filter = self._category_filter(categories)
            subject_key = _norm(subject)
            tokens = _tokenize(query)
            count = max(1, _as_int(limit, self.config.retrieval_limit))
            scored: list[tuple[float, float, str, KnowledgeEntry]] = []
            for knowledge_id in sorted(self._entries):
                entry = self._entries[knowledge_id]
                if cat_filter is not None and entry.category not in cat_filter:
                    continue
                if subject_key and subject_key not in _norm(entry.subject):
                    continue
                if verified_only and entry.verification_status is not VerificationStatus.VERIFIED:
                    continue
                if not include_historical:
                    if entry.verification_status in (
                        VerificationStatus.HISTORICAL,
                        VerificationStatus.REFUTED,
                    ):
                        continue
                    if entry.superseded_by:
                        continue
                score = self._score_entry(entry, tokens) if tokens else 0.0
                if tokens and score <= 0:
                    continue
                scored.append((score, float(entry.updated_at), knowledge_id, entry))
            scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
            return [copy.deepcopy(item[3]) for item in scored[:count]]

    def by_category(
        self,
        category: str,
        *,
        limit: int = 20,
        include_historical: bool = False,
    ) -> list[KnowledgeEntry]:
        """Most recent entries of one category (see :meth:`search`)."""
        cat = self._clean_category(category)
        return self.search(
            "",
            categories=[cat],
            limit=limit,
            include_historical=include_historical,
        )

    # ------------------------------------------------------------------ #
    # Epistemic operations
    # ------------------------------------------------------------------ #

    def verify(
        self,
        knowledge_id: str,
        *,
        evidence_refs: Iterable[str] | str = (),
        confidence: float | None = None,
    ) -> KnowledgeEntry:
        """Promote an entry to ``VERIFIED``; evidence is mandatory.

        The evidence references are merged into the entry and the confidence is
        raised to at least ``0.9``.  Historical entries can not be verified in
        the current branch.
        """
        with self._lock:
            entry = self._require(knowledge_id)
            if entry.verification_status is VerificationStatus.HISTORICAL:
                raise KnowledgeError(
                    f"cannot verify historical entry {entry.knowledge_id!r}",
                    details={"knowledge_id": entry.knowledge_id},
                )
            refs = sorted({*entry.evidence_refs, *self._clean_refs(evidence_refs)})
            if not refs:
                raise KnowledgeError(
                    "verification requires at least one evidence reference",
                    details={"knowledge_id": entry.knowledge_id},
                )
            wanted = entry.confidence if confidence is None else _as_float(confidence, entry.confidence)
            self._unindex_entry(entry)
            entry.verification_status = VerificationStatus.VERIFIED
            entry.evidence_refs = refs
            entry.confidence = max(
                self._cap_confidence(wanted, VerificationStatus.VERIFIED),
                entry.confidence if entry.confidence < _VERIFIED_MIN_CONFIDENCE else 0.0,
            )
            entry.updated_at = self._now()
            entry.revision = int(entry.revision) + 1
            self._index_entry(entry)
            self._version += 1
            self._audit_event(
                "verify", knowledge_id=entry.knowledge_id, evidence=len(refs)
            )
            return copy.deepcopy(entry)

    def refute(
        self,
        knowledge_id: str,
        *,
        evidence_refs: Iterable[str] | str = (),
        reason: str = "",
    ) -> KnowledgeEntry:
        """Mark an entry ``REFUTED`` (contradicting observation) and lower trust."""
        with self._lock:
            entry = self._require(knowledge_id)
            if entry.verification_status is VerificationStatus.HISTORICAL:
                raise KnowledgeError(
                    f"cannot refute historical entry {entry.knowledge_id!r}",
                    details={"knowledge_id": entry.knowledge_id},
                )
            refs = sorted({*entry.evidence_refs, *self._clean_refs(evidence_refs)})
            self._unindex_entry(entry)
            entry.verification_status = VerificationStatus.REFUTED
            entry.evidence_refs = refs
            entry.confidence = min(float(entry.confidence), 0.2)
            entry.updated_at = self._now()
            entry.revision = int(entry.revision) + 1
            self._version += 1
            self._audit_event(
                "refute",
                knowledge_id=entry.knowledge_id,
                reason=_clean(reason, 200),
                evidence=len(refs),
            )
            return copy.deepcopy(entry)

    def supersede(self, old_id: str, new_id: str) -> None:
        """Mark ``old_id`` as superseded by ``new_id`` (old record kept for audit)."""
        with self._lock:
            old = self._require(old_id)
            new = self._require(new_id)
            if old.knowledge_id == new.knowledge_id:
                raise KnowledgeError("an entry cannot supersede itself")
            self._unindex_entry(old)
            old.superseded_by = new.knowledge_id
            old.updated_at = self._now()
            old.revision = int(old.revision) + 1
            self._version += 1
            self._audit_event(
                "supersede",
                knowledge_id=old.knowledge_id,
                superseded_by=new.knowledge_id,
            )

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #

    def render_for_prompt(
        self,
        query: str = "",
        *,
        max_chars: int | None = None,
        categories: Iterable[str] | None = None,
        agent: AgentRole | str | None = None,
    ) -> str:
        """Render a strictly bounded, status-grouped knowledge block.

        Facts come first, then inferences, hypotheses and finally (small,
        clearly labelled) historical/refuted records.  The returned string is
        **never longer than** ``max_chars`` (default ``config.max_prompt_chars``).
        """
        budget = self.config.max_prompt_chars if max_chars is None else _as_int(max_chars, 0)
        budget = max(0, int(budget))
        if budget <= 0:
            return ""
        with self._lock:
            cat_filter = self._category_filter(categories)
            tokens = _tokenize(query)
            agent_role = self._coerce_optional_role(agent)
            groups: dict[VerificationStatus, list[KnowledgeEntry]] = {
                status: [] for status in _GROUP_ORDER
            }
            for knowledge_id in sorted(self._entries):
                entry = self._entries[knowledge_id]
                if cat_filter is not None and entry.category not in cat_filter:
                    continue
                if entry.superseded_by:
                    continue
                groups[entry.verification_status].append(entry)

            def sort_key(entry: KnowledgeEntry) -> tuple[float, int, float, str]:
                keyword = self._score_entry(entry, tokens) if tokens else 0.0
                agent_match = 1 if agent_role is not None and entry.source_agent is agent_role else 0
                return (-keyword, -agent_match, -float(entry.updated_at), entry.knowledge_id)

            lines: list[str] = []
            used = 0

            def add_line(text: str) -> bool:
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

            header = (
                f"SHARED WORLD KNOWLEDGE (branch={self._branch_id or 'unscoped'}, "
                f"v{self._version})"
            )
            if agent_role is not None:
                header += f" for {agent_role.value}"
            if not add_line(header):
                return ""

            total_cap = max(1, _as_int(self.config.max_entries_per_prompt, 24))
            added = 0
            for status in _GROUP_ORDER:
                if added >= total_cap:
                    break
                candidates = groups.get(status) or []
                if not candidates:
                    continue
                candidates.sort(key=sort_key)
                selected: list[KnowledgeEntry] = []
                for entry in candidates:
                    if len(selected) >= _GROUP_CAPS.get(status, 4):
                        break
                    if tokens and self._score_entry(entry, tokens) <= 0:
                        continue
                    selected.append(entry)
                if not selected:
                    continue
                if not add_line(_GROUP_TITLES[status]):
                    continue
                for entry in selected:
                    if added >= total_cap:
                        break
                    if not add_line(self._format_prompt_entry(entry)):
                        break
                    added += 1
            text = "\n".join(lines)
            return text[:budget]

    def render_agent_brief(
        self,
        agent: AgentRole | str,
        *,
        max_chars: int = 1500,
    ) -> str:
        """Render a short, role-focused brief for one agent (strict budget)."""
        role = self._coerce_role(agent)
        budget = max(0, _as_int(max_chars, 1500))
        if budget <= 0:
            return ""
        preferred = _AGENT_CATEGORIES.get(role, tuple(KNOWLEDGE_CATEGORIES))
        with self._lock:
            candidates: list[tuple[int, int, float, str, KnowledgeEntry]] = []
            for knowledge_id in sorted(self._entries):
                entry = self._entries[knowledge_id]
                if entry.superseded_by:
                    continue
                if entry.verification_status in (
                    VerificationStatus.HISTORICAL,
                    VerificationStatus.REFUTED,
                ):
                    continue
                try:
                    category_rank = preferred.index(entry.category)
                except ValueError:
                    category_rank = len(preferred)
                status_rank = _STATUS_RANK.get(entry.verification_status, 0)
                candidates.append(
                    (
                        category_rank,
                        -status_rank,
                        -float(entry.updated_at),
                        entry.knowledge_id,
                        entry,
                    )
                )
            candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            limit = max(1, _as_int(self.config.max_entries_per_prompt, 24))
            lines: list[str] = []
            used = 0

            def add_line(text: str) -> bool:
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

            if not candidates:
                return ""
            header = (
                f"WORLD MEMORY BRIEF for {role.value} "
                f"(branch={self._branch_id or 'unscoped'}, v{self._version})"
            )
            if not add_line(header):
                return ""
            added = 0
            for _rank, _status, _updated, _knowledge_id, entry in candidates:
                if added >= limit:
                    break
                if not add_line(self._format_brief_entry(entry)):
                    break
                added += 1
            return "\n".join(lines)[:budget]

    # ------------------------------------------------------------------ #
    # Statistics / snapshot / persistence
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Compact counters for the viewer/director (deterministic)."""
        with self._lock:
            by_status = {status.value: 0 for status in VerificationStatus}
            by_category = {category: 0 for category in KNOWLEDGE_CATEGORIES}
            for entry in self._entries.values():
                by_status[entry.verification_status.value] += 1
                if entry.category in by_category:
                    by_category[entry.category] += 1
            return {
                "version": self._version,
                "branch_id": self._branch_id,
                "entries": len(self._entries),
                "by_status": by_status,
                "by_category": by_category,
                "verified": by_status[VerificationStatus.VERIFIED.value],
                "inferred": by_status[VerificationStatus.INFERRED.value],
                "hypotheses": by_status[VerificationStatus.HYPOTHESIS.value],
                "historical": by_status[VerificationStatus.HISTORICAL.value],
                "refuted": by_status[VerificationStatus.REFUTED.value],
                "superseded": sum(1 for entry in self._entries.values() if entry.superseded_by),
                "dedupe_keys": len(self._dedupe),
                "downgraded": self._downgraded,
                "evicted": self._evicted,
                "max_entries": int(self.config.max_entries),
                "audit_events": len(self._audit),
                "path": str(self._path) if self._path is not None else "",
                "load_errors": len(self.load_errors),
            }

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the whole knowledge base."""
        with self._lock:
            return {
                "format": SHARED_MEMORY_FORMAT,
                "schema_version": SHARED_MEMORY_VERSION,
                "version": self._version,
                "branch_id": self._branch_id,
                "saved_at": self._now(),
                "entries": [
                    self._entries[knowledge_id].to_dict()
                    for knowledge_id in sorted(self._entries)
                ],
                "audit": copy.deepcopy(self._audit),
                "counters": {
                    "downgraded": self._downgraded,
                    "evicted": self._evicted,
                },
            }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Replace the knowledge base by ``snapshot`` (tolerant to bad records)."""
        if not isinstance(snapshot, Mapping):
            raise KnowledgeError("shared memory snapshot must be a mapping")
        with self._lock:
            self._restore_payload(snapshot)

    def save(self) -> None:
        """Persist the knowledge base atomically (no-op when ``root`` is unset)."""
        path = self._path
        if path is None:
            return
        with self._lock:
            payload = self.snapshot()
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
            try:
                _atomic_write_text(path, text)
            except OSError as exc:  # pragma: no cover - disk-level failure
                self.save_errors.append(f"shared memory save failed: {exc}")
                return
            self.last_saved_at = _as_float(payload.get("saved_at"), self._now())

    def load(self) -> bool:
        """Load the persisted knowledge base; tolerant to corruption."""
        path = self._path
        self.load_errors = []
        self.recovered_from = None
        if path is None or not path.is_file():
            return False
        try:
            text = path.read_text(encoding="utf-8")
            parsed = json.loads(text) if text.strip() else None
            if not isinstance(parsed, Mapping):
                raise ValueError("shared memory file does not contain a JSON object")
            schema = _as_int(parsed.get("schema_version"), 0)
            if schema and schema > SHARED_MEMORY_VERSION:
                raise ValueError(f"unsupported shared memory schema_version {schema}")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            self.load_errors.append(
                f"shared memory corrupt ({type(exc).__name__}: {exc}); started fresh"
            )
            backup = _backup_corrupt(path, now=self._now())
            if backup:
                self.recovered_from = backup
            self._entries = {}
            self._dedupe = {}
            self._audit = []
            self._version = 1
            return False
        assert isinstance(parsed, Mapping)
        with self._lock:
            self._restore_payload(parsed)
        return True

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _now(self) -> float:
        """Clock reading used for entry timestamps."""
        try:
            return float(self._clock())
        except Exception:  # pragma: no cover - a broken clock must not crash
            return now_ts()

    @property
    def _path(self) -> Path | None:
        """Internal persistence path helper (``None`` when disabled)."""
        if self._root is None:
            return None
        return self._root / SHARED_MEMORY_FILENAME

    @staticmethod
    def _coerce_role(value: Any) -> AgentRole:
        try:
            return AgentRole.coerce(value)
        except ProtocolError as exc:
            raise KnowledgeError(f"invalid source agent: {exc}") from exc

    @staticmethod
    def _coerce_optional_role(value: Any) -> AgentRole | None:
        if value is None or value == "":
            return None
        try:
            return AgentRole.coerce(value)
        except ProtocolError:
            return None

    @staticmethod
    def _coerce_status(value: Any) -> VerificationStatus:
        try:
            return VerificationStatus.coerce(value)
        except (ValueError, ProtocolError) as exc:
            raise KnowledgeError(f"invalid verification status: {value!r}") from exc

    @staticmethod
    def _clean_category(value: Any) -> str:
        category = _clean(value).lower() or "other"
        if category not in KNOWLEDGE_CATEGORIES:
            raise KnowledgeError(f"unknown knowledge category: {value!r}")
        return category

    @staticmethod
    def _clean_refs(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            items: Iterable[Any] = [value]
        else:
            items = value
        try:
            return sorted({_clean(ref, 200) for ref in items if _clean(ref, 200)})
        except TypeError as exc:  # pragma: no cover - defensive
            raise KnowledgeError("evidence_refs must be an iterable of strings") from exc

    @staticmethod
    def _clean_tags(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            items: Iterable[Any] = [value]
        else:
            items = value
        try:
            return sorted({_clean(tag, MAX_TAG_CHARS) for tag in items if _clean(tag, MAX_TAG_CHARS)})
        except TypeError as exc:  # pragma: no cover - defensive
            raise KnowledgeError("tags must be an iterable of strings") from exc

    def _sanitize_status(
        self,
        status: VerificationStatus,
        evidence_refs: Iterable[str],
    ) -> VerificationStatus:
        """Enforce "VERIFIED needs evidence"; downgrades unproven claims."""
        if status is VerificationStatus.VERIFIED and not list(evidence_refs):
            self._downgraded += 1
            return VerificationStatus.INFERRED
        return status

    @staticmethod
    def _cap_confidence(confidence: Any, status: VerificationStatus) -> float:
        """Clamp confidence and keep non-facts below the verified band."""
        value = max(0.0, min(1.0, _as_float(confidence, 0.5)))
        cap = _CONFIDENCE_CAPS.get(status)
        if cap is not None:
            value = min(value, cap)
        if status is VerificationStatus.VERIFIED:
            value = max(value, _VERIFIED_MIN_CONFIDENCE)
        return round(value, 3)

    @staticmethod
    def _category_filter(categories: Iterable[str] | None) -> set[str] | None:
        if categories is None:
            return None
        if isinstance(categories, (str, bytes)):
            categories = [categories]  # type: ignore[assignment]
        wanted: set[str] = set()
        for category in categories:
            wanted.add(_clean(category).lower())
        return wanted

    def _index_entry(self, entry: KnowledgeEntry) -> None:
        """Add one entry to the duplicate index when it is trusted/live."""
        if not self._indexable(entry):
            self._unindex_entry(entry)
            return
        self._dedupe[(entry.branch_id, entry.dedupe_key)] = entry.knowledge_id

    def _unindex_entry(self, entry: KnowledgeEntry) -> None:
        key = (entry.branch_id, entry.dedupe_key)
        if self._dedupe.get(key) == entry.knowledge_id:
            del self._dedupe[key]

    @staticmethod
    def _indexable(entry: KnowledgeEntry) -> bool:
        """Historical/refuted/superseded records must not suppress new ones."""
        return (
            entry.verification_status
            not in (VerificationStatus.HISTORICAL, VerificationStatus.REFUTED)
            and not entry.superseded_by
        )

    def _rebuild_index(self) -> None:
        """Rebuild the duplicate index deterministically from the current entries."""
        self._dedupe = {}
        for knowledge_id in sorted(self._entries):
            entry = self._entries[knowledge_id]
            if self._indexable(entry):
                self._dedupe.setdefault((entry.branch_id, entry.dedupe_key), knowledge_id)

    def _find_duplicate(self, branch: str, key: str) -> KnowledgeEntry | None:
        knowledge_id = self._dedupe.get((branch, key))
        if not knowledge_id:
            return None
        entry = self._entries.get(knowledge_id)
        if entry is None or not self._indexable(entry):
            return None
        return entry

    def _merge_into(
        self,
        entry: KnowledgeEntry,
        *,
        confidence: float,
        status: VerificationStatus,
        evidence_refs: Iterable[str],
        subject: str,
        tags: Iterable[str],
    ) -> None:
        """Merge a repeated observation into an existing entry (same identity)."""
        self._unindex_entry(entry)
        refs = sorted({*entry.evidence_refs, *evidence_refs})
        entry.evidence_refs = refs
        incoming_status = self._sanitize_status(status, refs)
        if incoming_status is VerificationStatus.REFUTED:
            entry.verification_status = VerificationStatus.REFUTED
        elif _STATUS_RANK.get(incoming_status, 0) > _STATUS_RANK.get(
            entry.verification_status, 0
        ):
            entry.verification_status = incoming_status
        entry.confidence = self._cap_confidence(
            max(float(entry.confidence), float(confidence)),
            entry.verification_status,
        )
        if subject:
            entry.subject = subject
        entry.tags = sorted({*entry.tags, *tags})
        entry.updated_at = self._now()
        entry.revision = int(entry.revision) + 1
        self._index_entry(entry)

    def _enforce_limits(self) -> None:
        """Evict the least valuable/stalest entries beyond ``max_entries``."""
        limit = _as_int(self.config.max_entries, 0)
        if limit <= 0:
            return
        while len(self._entries) > limit:
            victim = min(
                self._entries.values(),
                key=lambda item: (
                    _EVICTION_ORDER.index(item.verification_status)
                    if item.verification_status in _EVICTION_ORDER
                    else len(_EVICTION_ORDER),
                    float(item.created_at),
                    item.knowledge_id,
                ),
            )
            self._unindex_entry(victim)
            del self._entries[victim.knowledge_id]
            self._evicted += 1

    def _require(self, knowledge_id: str) -> KnowledgeEntry:
        key = str(knowledge_id or "")
        entry = self._entries.get(key)
        if entry is None:
            raise KnowledgeError(
                f"unknown knowledge entry: {key!r}", details={"knowledge_id": key}
            )
        return entry

    def _audit_event(self, kind: str, **fields: Any) -> None:
        """Append one bounded audit event (never raises)."""
        record: dict[str, Any] = {"kind": str(kind), "at": self._now()}
        record.update(fields)
        self._audit.append(record)
        if len(self._audit) > MAX_AUDIT_EVENTS:
            del self._audit[: len(self._audit) - MAX_AUDIT_EVENTS]

    def _restore_payload(self, payload: Mapping[str, Any]) -> None:
        """Reset the state from a snapshot/persisted payload (tolerant)."""
        self._entries = {}
        self._dedupe = {}
        self._audit = []
        self.load_errors = []
        entries_raw = payload.get("entries") or []
        if isinstance(entries_raw, (str, bytes)) or not isinstance(entries_raw, (list, tuple)):
            entries_raw = []
        for item in entries_raw:
            if not isinstance(item, Mapping):
                continue
            try:
                entry = KnowledgeEntry.from_dict(item)
            except (MultiAgentError, TypeError, ValueError) as exc:
                self.load_errors.append(f"skipped invalid knowledge record: {exc}")
                continue
            if entry.knowledge_id in self._entries:
                self.load_errors.append(f"skipped duplicate knowledge id: {entry.knowledge_id}")
                continue
            self._entries[entry.knowledge_id] = entry
        audit_raw = payload.get("audit") or []
        if isinstance(audit_raw, (list, tuple)):
            for record in audit_raw:
                if isinstance(record, Mapping):
                    self._audit.append(copy.deepcopy(dict(record)))
            if len(self._audit) > MAX_AUDIT_EVENTS:
                del self._audit[: len(self._audit) - MAX_AUDIT_EVENTS]
        counters = payload.get("counters")
        if isinstance(counters, Mapping):
            self._downgraded = max(0, _as_int(counters.get("downgraded"), 0))
            self._evicted = max(0, _as_int(counters.get("evicted"), 0))
        self._version = max(1, _as_int(payload.get("version"), 1))
        branch = str(payload.get("branch_id") or "")
        if not self._branch_id and branch:
            self._branch_id = branch
        self.last_saved_at = _as_float(payload.get("saved_at"), 0.0)
        self._rebuild_index()

    # ------------------------------------------------------------------ #
    # Retrieval / rendering helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_entry(entry: KnowledgeEntry, tokens: Iterable[str]) -> float:
        """Keyword match score over content/subject/tags/category (deterministic)."""
        token_list = list(tokens)
        if not token_list:
            return 0.0
        haystack = " ".join(
            [
                str(entry.content),
                str(entry.subject),
                " ".join(str(tag) for tag in entry.tags),
                str(entry.category),
            ]
        ).casefold()
        score = 0.0
        for token in token_list:
            pattern = r"(?<![0-9a-z])" + re.escape(token) + r"(?![0-9a-z])"
            if re.search(pattern, haystack):
                score += 2.0
            elif (not token.isascii() or len(token) >= 4) and token in haystack:
                score += 1.0
        return score

    def _format_prompt_entry(self, entry: KnowledgeEntry) -> str:
        """One compact prompt line for an entry (status is implied by the group)."""
        text = _clean(entry.content, min(int(self.config.max_content_chars), MAX_PROMPT_LINE_CHARS))
        parts = [f"- [{entry.category}] {text}"]
        if entry.subject:
            parts.append(f"(subject: {_clean(entry.subject, 80)})")
        parts.append(f"(confidence {float(entry.confidence):.2f})")
        if entry.evidence_refs:
            parts.append(f"[{len(entry.evidence_refs)} evidence]")
        return " ".join(parts)

    def _format_brief_entry(self, entry: KnowledgeEntry) -> str:
        """One compact brief line (status prefix keeps uncertainty explicit)."""
        status = entry.verification_status
        flag = {
            VerificationStatus.VERIFIED: "fact",
            VerificationStatus.INFERRED: "inference",
            VerificationStatus.HYPOTHESIS: "hypothesis",
            VerificationStatus.HISTORICAL: "old branch",
            VerificationStatus.REFUTED: "refuted",
        }.get(status, status.value)
        text = _clean(entry.content, min(int(self.config.max_content_chars), 200))
        subject = f" (subject: {_clean(entry.subject, 60)})" if entry.subject else ""
        return f"- [{flag}] {text}{subject}"
