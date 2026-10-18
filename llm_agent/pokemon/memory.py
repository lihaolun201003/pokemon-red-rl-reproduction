"""Persistent battle memory for the autonomous Pokémon battle agent.

:class:`PokemonMemory` is the battle pilot's own notebook.  It is built on top of
the existing :class:`llm_agent.memory.AgentMemory` (which stores the generic
agent lessons/journal under a caller-provided root) and adds a battle-specific,
structured layer persisted next to it:

* ``battles``  — one record per handled battle (outcome, turns, decisions, cost)
* ``moves``    — move names actually **shown on the battle screen** (with slot,
  visible text and how often they were chosen)
* ``damage``   — visible damage / HP-bar / HP-text observations
* ``texts``    — visible battle message lines the model read from screenshots
* ``team``     — party members visible on screen (never hidden stats)
* ``lessons``  — validated :class:`~llm_agent.multi_agent.contracts.KnowledgeEntry`
  records that are also forwarded to the shared world memory
* ``long_term`` — structured hooks for training / evolution / move learning /
  team management / item use.

Honesty rules encoded here:

* an *observation* (something readable on a screenshot, reported by the model or
  measured by the agent) is stored as ``VERIFIED``;
* an *inference* (e.g. "this move seems effective") is stored as ``INFERRED``;
* nothing hidden (enemy HP numbers, unseen moves, stats) can enter the memory:
  the public write methods only accept visible facts, and every record keeps the
  ``evidence_refs`` (screenshot paths / frame ids) that justify it.

The battle memory only *reads* CPU-friendly JSON on its own path: all writes are
atomic (temp file + ``os.replace``) and corrupt files are reset defensively
instead of crashing a run.  This module never imports PyBoy, never reads RAM and
never talks to the network.

Long-term hooks: :meth:`record_training_observation`,
:meth:`record_evolution_observed`, :meth:`record_move_learned`,
:meth:`record_team_change` and :meth:`record_item_use` are **structured but not
yet validated on a real run** — they persist typed records with
``"status": "structured_only"`` / ``"validated": False`` so the future training /
team-management flows can pick them up, but no current caller depends on them.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from llm_agent.config import MemoryConfig
from llm_agent.memory import AgentMemory
from llm_agent.multi_agent.contracts import (
    KNOWLEDGE_CATEGORIES,
    AgentRole,
    KnowledgeEntry,
    VerificationStatus,
    new_knowledge_id,
    now_ts,
)

#: Version of the persisted battle-memory document.
POKEMON_MEMORY_VERSION: int = 1
POKEMON_MEMORY_FILENAME: str = "pokemon_memory.json"

#: Hard caps that keep the file small and the prompt cheap.
MAX_BATTLES: int = 60
MAX_MOVES: int = 120
MAX_DAMAGE: int = 120
MAX_TEXTS: int = 100
MAX_TEAM_MEMBERS: int = 12
MAX_LESSONS: int = 300
MAX_LONG_TERM: int = 100

#: Default caps used when a prompt rendering budget is not provided.
DEFAULT_RENDER_CHARS: int = 2500
DEFAULT_BATTLE_BRIEF_CHARS: int = 1200
DEFAULT_TEAM_BRIEF_CHARS: int = 600

#: Outcome vocabulary shared with the agent / tool schemas.
OUTCOMES: tuple[str, ...] = ("win", "loss", "caught", "ran", "unknown")

#: Categories allowed for lessons created by this memory.
LESSON_CATEGORY_DEFAULT: str = "battle_lesson"


# --------------------------------------------------------------------------- #
# Tiny helpers
# --------------------------------------------------------------------------- #

def _clean(value: Any, limit: int = 0) -> str:
    """Collapse whitespace and truncate a value to ``limit`` characters."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[: int(limit)] if limit and limit > 0 else text


def _clean_block(value: Any, limit: int = 0) -> str:
    """Strip a multi-line value, keeping internal newlines."""
    if value is None:
        return ""
    text = str(value).strip()
    return text[: int(limit)] if limit and limit > 0 else text


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any, *, limit: int = 8, item_limit: int = 300) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    out: list[str] = []
    for item in value:
        text = _clean(item, item_limit)
        if text:
            out.append(text)
    return out[: max(0, int(limit))]


def _make_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _norm_key(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 ``text`` to ``path`` atomically (unique temp + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover - best effort cleanup
            pass


def _coerce_verification(value: Any, *, default: VerificationStatus) -> VerificationStatus:
    try:
        return VerificationStatus.coerce(value)
    except Exception:  # noqa: BLE001 - unknown labels fall back to the safe default
        return default


def _trim(items: list[Any], cap: int) -> list[Any]:
    if cap > 0 and len(items) > cap:
        del items[: len(items) - cap]
    return items


# --------------------------------------------------------------------------- #
# Fresh document
# --------------------------------------------------------------------------- #

def _fresh_data() -> dict[str, Any]:
    """Return a valid, empty battle-memory document."""
    now = now_ts()
    return {
        "version": POKEMON_MEMORY_VERSION,
        "created_at": now,
        "updated_at": now,
        "branch_id": "",
        "battles": [],
        "moves": [],
        "damage": [],
        "texts": [],
        "team": {"members": [], "updated_at": 0.0},
        "lessons": [],
        "long_term": {
            "training": [],
            "evolutions": [],
            "moves_learned": [],
            "team_changes": [],
            "item_uses": [],
        },
        "counters": {
            "battles": 0,
            "wins": 0,
            "losses": 0,
            "catches": 0,
            "runs": 0,
            "turns": 0,
        },
    }


# --------------------------------------------------------------------------- #
# Normalisation (corrupt/older files must never crash a run)
# --------------------------------------------------------------------------- #

def _normalise_battle(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    outcome = _clean(raw.get("outcome"), 20).lower()
    if outcome not in OUTCOMES:
        outcome = "unknown"
    actions: list[dict[str, Any]] = []
    raw_actions = raw.get("actions")
    if isinstance(raw_actions, Sequence) and not isinstance(raw_actions, (str, bytes)):
        for item in list(raw_actions)[-40:]:
            if not isinstance(item, Mapping):
                continue
            actions.append(
                {
                    "turn": _as_int(item.get("turn"), 0),
                    "action": _clean(item.get("action"), 40),
                    "slot": _as_int(item.get("slot"), 0) or None,
                    "presses": [str(p)[:20] for p in (item.get("presses") or []) if str(p).strip()][:16],
                    "verified": bool(item.get("verified")),
                }
            )
    return {
        "battle_id": _clean(raw.get("battle_id"), 40) or _make_id("b_"),
        "at": _as_float(raw.get("at"), 0.0),
        "outcome": outcome,
        "turns": max(0, _as_int(raw.get("turns"), 0)),
        "actions": actions,
        "cost_usd": max(0.0, _as_float(raw.get("cost_usd"), 0.0)),
        "evidence_refs": _as_str_list(raw.get("evidence_refs")),
        "visible_text": _clean(raw.get("visible_text"), 300),
        "notes": _clean(raw.get("notes"), 500),
        "verification": _clean(raw.get("verification"), 20) or "inferred",
        "branch_id": _clean(raw.get("branch_id"), 60),
    }


def _normalise_move(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    name = _clean(raw.get("name"), 120)
    if not name:
        return None
    return {
        "name": name,
        "user": _clean(raw.get("user"), 60) or "player",
        "slots": [_as_int(slot, 0) for slot in (raw.get("slots") or []) if _as_int(slot, 0) > 0][:8],
        "times_used": max(0, _as_int(raw.get("times_used"), 0)),
        "visible_texts": _as_str_list(raw.get("visible_texts"), limit=4, item_limit=200),
        "note": _clean(raw.get("note"), 300),
        "first_seen": _as_float(raw.get("first_seen"), 0.0),
        "last_seen": _as_float(raw.get("last_seen"), 0.0),
        "verification_status": _clean(raw.get("verification_status"), 20) or "verified",
        "evidence_refs": _as_str_list(raw.get("evidence_refs")),
    }


def _normalise_damage(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    description = _clean(raw.get("description"), 400)
    if not description:
        return None
    return {
        "description": description,
        "move_name": _clean(raw.get("move_name"), 120),
        "visible_text": _clean(raw.get("visible_text"), 300),
        "at": _as_float(raw.get("at"), 0.0),
        "verification_status": _clean(raw.get("verification_status"), 20) or "verified",
        "confidence": min(1.0, max(0.0, _as_float(raw.get("confidence"), 0.7))),
        "evidence_refs": _as_str_list(raw.get("evidence_refs")),
    }


def _normalise_text(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    text = _clean(raw.get("text"), 300)
    if not text:
        return None
    return {
        "text": text,
        "at": _as_float(raw.get("at"), 0.0),
        "evidence_refs": _as_str_list(raw.get("evidence_refs")),
    }


def _normalise_member(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    name = _clean(raw.get("name"), 120)
    if not name:
        return None
    slot = _as_int(raw.get("slot"), 0)
    return {
        "name": name,
        "slot": slot if slot > 0 else None,
        "hp_text": _clean(raw.get("hp_text"), 120),
        "level_text": _clean(raw.get("level_text"), 120),
        "status_text": _clean(raw.get("status_text"), 120),
        "visible_text": _clean(raw.get("visible_text"), 300),
        "first_seen": _as_float(raw.get("first_seen"), 0.0),
        "last_seen": _as_float(raw.get("last_seen"), 0.0),
        "verification_status": _clean(raw.get("verification_status"), 20) or "verified",
        "evidence_refs": _as_str_list(raw.get("evidence_refs")),
    }


def _normalise_lesson(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    content = _clean(raw.get("content"), 500)
    if not content:
        return None
    category = _clean(raw.get("category"), 40) or LESSON_CATEGORY_DEFAULT
    if category not in KNOWLEDGE_CATEGORIES:
        category = "other"
    status = _coerce_verification(raw.get("verification_status"), default=VerificationStatus.INFERRED)
    knowledge_id = _clean(raw.get("knowledge_id"), 40)
    if not knowledge_id.startswith("kn_"):
        knowledge_id = new_knowledge_id()
    return {
        "category": category,
        "content": content,
        "verification_status": status.value,
        "confidence": min(1.0, max(0.0, _as_float(raw.get("confidence"), 0.5))),
        "evidence_refs": _as_str_list(raw.get("evidence_refs")),
        "knowledge_id": knowledge_id,
        "subject": _clean(raw.get("subject"), 200),
        "tags": _as_str_list(raw.get("tags"), limit=8, item_limit=40),
        "at": _as_float(raw.get("at"), 0.0),
    }


def _normalise_long_term(raw: Any) -> dict[str, list[dict[str, Any]]]:
    buckets = ("training", "evolutions", "moves_learned", "team_changes", "item_uses")
    out: dict[str, list[dict[str, Any]]] = {}
    raw = raw if isinstance(raw, Mapping) else {}
    for bucket in buckets:
        items: list[dict[str, Any]] = []
        for item in list(raw.get(bucket) or [])[-MAX_LONG_TERM:]:
            if not isinstance(item, Mapping):
                continue
            record = {
                "kind": _clean(item.get("kind"), 40) or bucket,
                "description": _clean(item.get("description"), 400),
                "at": _as_float(item.get("at"), 0.0),
                "status": "structured_only",
                "validated": False,
            }
            for extra in ("success", "visible_text", "slot", "replaced_move", "target", "battle_context"):
                if extra in item and item.get(extra) is not None:
                    record[extra] = copy.deepcopy(item.get(extra))
            record["evidence_refs"] = _as_str_list(item.get("evidence_refs"))
            if not record["description"]:
                continue
            items.append(record)
        out[bucket] = items
    return out


def _normalise_data(raw: Any) -> dict[str, Any]:
    """Build a valid document from ``raw`` (tolerates any junk)."""
    data = _fresh_data()
    if not isinstance(raw, Mapping):
        return data
    data["created_at"] = _as_float(raw.get("created_at"), data["created_at"])
    data["updated_at"] = _as_float(raw.get("updated_at"), data["created_at"])
    data["branch_id"] = _clean(raw.get("branch_id"), 60)

    for key, normaliser, cap in (
        ("battles", _normalise_battle, MAX_BATTLES),
        ("moves", _normalise_move, MAX_MOVES),
        ("damage", _normalise_damage, MAX_DAMAGE),
        ("texts", _normalise_text, MAX_TEXTS),
        ("lessons", _normalise_lesson, MAX_LESSONS),
    ):
        items = []
        for entry in list(raw.get(key) or [])[-cap:]:
            normalised = normaliser(entry) if isinstance(entry, Mapping) else None
            if normalised is not None:
                items.append(normalised)
        data[key] = items

    team_raw = raw.get("team") if isinstance(raw.get("team"), Mapping) else {}
    members: list[dict[str, Any]] = []
    for entry in list(team_raw.get("members") or [])[-MAX_TEAM_MEMBERS:]:
        member = _normalise_member(entry) if isinstance(entry, Mapping) else None
        if member is not None:
            members.append(member)
    data["team"] = {
        "members": members,
        "updated_at": _as_float(team_raw.get("updated_at"), 0.0),
    }

    data["long_term"] = _normalise_long_term(raw.get("long_term"))
    counters = raw.get("counters") if isinstance(raw.get("counters"), Mapping) else {}
    data["counters"] = {
        key: max(0, _as_int(counters.get(key), 0)) for key in data["counters"]
    }
    return data


# --------------------------------------------------------------------------- #
# PokemonMemory
# --------------------------------------------------------------------------- #

class PokemonMemory:
    """Battle-specific long-term memory for :class:`PokemonAgent`.

    Parameters
    ----------
    agent_memory:
        The generic :class:`~llm_agent.memory.AgentMemory` instance the agent
        was given (or any duck-typed object exposing ``record_lesson`` /
        ``render_context``).  Optional: battle memory also works standalone.
    root:
        Directory for ``pokemon_memory.json``.  Defaults to
        ``<agent_memory.root>/pokemon`` (so two agents never share a file).
    config:
        Optional :class:`~llm_agent.config.AgentConfig` (only used to locate a
        memory root when ``agent_memory``/``root`` are not supplied).
    clock:
        Injectable timestamp source (tests pass a deterministic counter).
    """

    def __init__(
        self,
        agent_memory: Any | None = None,
        *,
        root: str | Path | None = None,
        config: Any | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.agent_memory = agent_memory
        self.config = config
        self._clock: Callable[[], float] = clock or time.time
        self.root: Path = Path(root) if root is not None else self._default_root(config)
        self.creation_error: str = ""
        self.load_errors: list[str] = []
        self.recovered_from: str | None = None
        self._data: dict[str, Any] = _fresh_data()
        self._knowledge: dict[str, KnowledgeEntry] = {}
        self._branch_id: str = ""
        self.load()

    # -- construction helpers -------------------------------------------- #
    @staticmethod
    def _default_root(config: Any | None = None) -> Path:
        memory_cfg = getattr(config, "memory", None) if config is not None else None
        base = getattr(memory_cfg, "root", "") if memory_cfg is not None else ""
        if base:
            return Path(str(base)) / "pokemon"
        return Path(tempfile.gettempdir()) / "pokemon_agent_memory"

    @staticmethod
    def _memory_config(config: Any | None, *, root: Path) -> MemoryConfig:
        memory_cfg = getattr(config, "memory", None) if config is not None else None
        if isinstance(memory_cfg, MemoryConfig):
            return replace(memory_cfg, root=str(root), mirror_to_run=False)
        return MemoryConfig(root=str(root), mirror_to_run=False)

    @classmethod
    def create(
        cls,
        *,
        memory: Any | None = None,
        config: Any | None = None,
        root: str | Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> "PokemonMemory":
        """Build a battle memory from whatever the director injected.

        Accepts an existing :class:`PokemonMemory` (returned unchanged), a
        generic :class:`AgentMemory`, ``None`` (a private ``AgentMemory`` is
        created under the configured memory root) or any duck-typed object with
        ``record_lesson`` / ``render_context``.
        """
        if isinstance(memory, cls):
            return memory
        explicit_root = Path(root) if root is not None else None
        agent_memory = memory
        creation_error = ""
        if agent_memory is None:
            local_root = explicit_root if explicit_root is not None else cls._default_root(config)
            try:
                agent_memory = AgentMemory(cls._memory_config(config, root=local_root))
            except Exception as exc:  # noqa: BLE001 - a memory root must not stop a battle
                agent_memory = None
                creation_error = f"{type(exc).__name__}: {exc}"
        if explicit_root is None:
            base = getattr(agent_memory, "root", None) if agent_memory is not None else None
            explicit_root = Path(str(base)) / "pokemon" if base else cls._default_root(config)
        instance = cls(agent_memory, root=explicit_root, config=config, clock=clock)
        instance.creation_error = creation_error
        return instance

    # -- paths ------------------------------------------------------------ #
    @property
    def memory_path(self) -> Path:
        """Path of the persisted battle-memory JSON file."""
        return self.root / POKEMON_MEMORY_FILENAME

    # -- persistence ------------------------------------------------------ #
    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - disk-level failure
            self.load_errors.append(f"battle memory root unavailable: {exc}")

    def load(self) -> None:
        """Read ``pokemon_memory.json``; a corrupt file resets memory (no crash)."""
        self.load_errors = []
        self.recovered_from = None
        self._ensure_root()
        raw: Mapping[str, Any] | None = None
        path = self.memory_path
        if path.is_file():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(parsed, Mapping):
                    raise ValueError("battle memory is not a JSON object")
                version = _as_int(parsed.get("version"), POKEMON_MEMORY_VERSION)
                if version != POKEMON_MEMORY_VERSION:
                    raise ValueError(f"unsupported battle memory version {version}")
                raw = parsed
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                self.load_errors.append(
                    f"pokemon_memory.json corrupt ({type(exc).__name__}: {exc}); started fresh"
                )
                backup = self._backup_corrupt(path)
                self.recovered_from = str(backup) if backup else None
                raw = None
        self._data = _normalise_data(raw or {})
        self._rebuild_knowledge()
        if self.load_errors:
            self.save()

    def _backup_corrupt(self, path: Path) -> Path | None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._clock()))
        target = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
        try:
            os.replace(path, target)
            return target
        except OSError:  # pragma: no cover - disk-level failure
            return None

    def _rebuild_knowledge(self) -> None:
        self._knowledge = {}
        for record in self._data.get("lessons") or []:
            try:
                entry = self._knowledge_entry_from_record(record)
            except Exception:  # noqa: BLE001 - stale record must not break loading
                continue
            if entry is not None:
                self._knowledge[entry.knowledge_id] = entry

    def _knowledge_entry_from_record(self, record: Mapping[str, Any]) -> KnowledgeEntry | None:
        knowledge_id = _clean(record.get("knowledge_id"), 40)
        if not knowledge_id.startswith("kn_"):
            return None
        try:
            return KnowledgeEntry.create(
                category=_clean(record.get("category"), 40) or LESSON_CATEGORY_DEFAULT,
                content=_clean(record.get("content"), 500),
                source_agent=AgentRole.POKEMON,
                confidence=min(1.0, max(0.0, _as_float(record.get("confidence"), 0.5))),
                verification_status=_coerce_verification(
                    record.get("verification_status"), default=VerificationStatus.INFERRED
                ),
                branch_id=self._branch_id,
                evidence_refs=_as_str_list(record.get("evidence_refs")),
                subject=_clean(record.get("subject"), 200),
                tags=_as_str_list(record.get("tags"), limit=8, item_limit=40),
                knowledge_id=knowledge_id,
                created_at=_as_float(record.get("at"), 0.0) or now_ts(),
            )
        except Exception:  # noqa: BLE001 - KnowledgeEntry validation is strict
            return None

    def save(self) -> None:
        """Persist the battle memory atomically (best effort: never raises)."""
        self._data["updated_at"] = self._clock()
        try:
            payload = json.dumps(self._data, ensure_ascii=False, indent=1, default=str)
            _atomic_write_text(self.memory_path, payload + "\n")
        except (OSError, TypeError, ValueError) as exc:  # pragma: no cover - disk-level
            self.load_errors.append(f"battle memory write failed: {exc}")

    # -- branch / counters ------------------------------------------------ #
    def set_branch(self, branch_id: str) -> None:
        """Tag subsequently created knowledge entries with the active branch."""
        self._branch_id = _clean(branch_id, 60)

    @property
    def branch_id(self) -> str:
        return self._branch_id

    def stats(self) -> dict[str, Any]:
        """Compact counters (safe for status dictionaries / prompts)."""
        data = self._data
        counters = data.get("counters") or {}
        return {
            "battles": len(data.get("battles") or []),
            "wins": _as_int(counters.get("wins"), 0),
            "losses": _as_int(counters.get("losses"), 0),
            "catches": _as_int(counters.get("catches"), 0),
            "runs": _as_int(counters.get("runs"), 0),
            "turns": _as_int(counters.get("turns"), 0),
            "moves": len(data.get("moves") or []),
            "damage_observations": len(data.get("damage") or []),
            "team_members": len((data.get("team") or {}).get("members") or []),
            "lessons": len(data.get("lessons") or []),
            "branch_id": self._branch_id,
            "root": str(self.root),
        }

    # -- lesson storage --------------------------------------------------- #
    def _store_lesson(
        self,
        *,
        category: str,
        content: str,
        verification: VerificationStatus | str,
        confidence: float,
        evidence_refs: Iterable[str] | None = None,
        subject: str = "",
        tags: Iterable[str] | None = None,
        success: bool | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Create + store one :class:`KnowledgeEntry` and mirror it to AgentMemory."""
        text = _clean(content, 500)
        if not text:
            return {}
        category_name = _clean(category, 40).lower()
        if category_name not in KNOWLEDGE_CATEGORIES:
            category_name = LESSON_CATEGORY_DEFAULT
        status = _coerce_verification(verification, default=VerificationStatus.INFERRED)
        confidence_value = min(1.0, max(0.0, float(confidence)))
        refs = _as_str_list(evidence_refs)
        entry = KnowledgeEntry.create(
            category=category_name,
            content=text,
            source_agent=AgentRole.POKEMON,
            confidence=confidence_value,
            verification_status=status,
            branch_id=self._branch_id,
            evidence_refs=refs,
            subject=_clean(subject, 200),
            tags=_as_str_list(tags, limit=8, item_limit=40),
        )
        record = {
            "category": entry.category,
            "content": entry.content,
            "verification_status": entry.verification_status.value,
            "confidence": entry.confidence,
            "evidence_refs": list(entry.evidence_refs),
            "knowledge_id": entry.knowledge_id,
            "subject": entry.subject,
            "tags": list(entry.tags),
            "at": self._clock(),
        }
        self._data["lessons"].append(record)
        _trim(self._data["lessons"], MAX_LESSONS)
        self._knowledge[entry.knowledge_id] = entry
        # Mirror the durable lesson into the generic agent memory so the normal
        # prompt memory renders it too (best effort; never fatal).
        if self.agent_memory is not None:
            record_lesson = getattr(self.agent_memory, "record_lesson", None)
            if callable(record_lesson):
                try:
                    record_lesson(text, success=success, context=subject or category_name)
                except Exception:  # noqa: BLE001 - generic memory must not break a battle
                    pass
        self._data["updated_at"] = self._clock()
        if persist:
            self.save()
        return copy.deepcopy(record)

    def knowledge_entry(self, knowledge_id: str) -> KnowledgeEntry | None:
        """Return the :class:`KnowledgeEntry` for an id (copy by reference)."""
        return self._knowledge.get(str(knowledge_id or ""))

    def knowledge_entries(self) -> list[KnowledgeEntry]:
        """All battle knowledge entries currently kept in memory."""
        return list(self._knowledge.values())

    # -- battle records --------------------------------------------------- #
    def record_battle(
        self,
        *,
        outcome: str,
        turns: int,
        actions: Sequence[Mapping[str, Any]] | None = None,
        cost_usd: float = 0.0,
        evidence_refs: Iterable[str] | None = None,
        visible_text: str = "",
        notes: str = "",
        verification: str | None = None,
        branch_id: str = "",
    ) -> dict[str, Any]:
        """Store one finished battle + its outcome lesson.

        ``verification`` should be ``"verified"`` only when the outcome was
        actually readable on screen; otherwise the outcome lesson is
        ``INFERRED``.  The fact that the battle ended is always pixel-verified
        by the agent loop, but the *result* is marked honestly.
        """
        outcome_name = _clean(outcome, 20).lower()
        if outcome_name not in OUTCOMES:
            outcome_name = "unknown"
        compact_actions: list[dict[str, Any]] = []
        for item in list(actions or [])[-40:]:
            if not isinstance(item, Mapping):
                continue
            compact_actions.append(
                {
                    "turn": _as_int(item.get("turn"), 0),
                    "action": _clean(item.get("action"), 40),
                    "slot": _as_int(item.get("slot"), 0) or None,
                    "presses": [str(p)[:20] for p in (item.get("presses") or [])][:16],
                    "verified": bool(item.get("verified")),
                }
            )
        refs = _as_str_list(evidence_refs)
        battle_id = _make_id("b_")
        battle = {
            "battle_id": battle_id,
            "at": self._clock(),
            "outcome": outcome_name,
            "turns": max(0, _as_int(turns, 0)),
            "actions": compact_actions,
            "cost_usd": max(0.0, _as_float(cost_usd, 0.0)),
            "evidence_refs": refs,
            "visible_text": _clean(visible_text, 300),
            "notes": _clean(notes, 500),
            "verification": _clean(
                verification,
                20,
            )
            or ("verified" if outcome_name != "unknown" and visible_text else "inferred"),
            "branch_id": _clean(branch_id, 60) or self._branch_id,
        }
        self._data["battles"].append(battle)
        _trim(self._data["battles"], MAX_BATTLES)
        counters = self._data["counters"]
        counters["battles"] = max(0, _as_int(counters.get("battles"), 0)) + 1
        counters["turns"] = max(0, _as_int(counters.get("turns"), 0)) + max(0, _as_int(turns, 0))
        bucket = {
            "win": "wins",
            "loss": "losses",
            "caught": "catches",
            "ran": "runs",
        }.get(outcome_name)
        if bucket:
            counters[bucket] = max(0, _as_int(counters.get(bucket), 0)) + 1

        parts = [
            f"Battle finished: outcome={outcome_name}",
            f"turns={max(0, _as_int(turns, 0))}",
        ]
        if visible_text:
            parts.append(f"visible text: '{_clean(visible_text, 200)}'")
        if notes:
            parts.append(f"notes: {_clean(notes, 200)}")
        if compact_actions:
            summary = ", ".join(
                f"t{item.get('turn')}:{item.get('action')}"
                + (f"#{item.get('slot')}" if item.get("slot") else "")
                for item in compact_actions[-6:]
            )
            parts.append(f"decisions: {summary}")
        status = _coerce_verification(
            battle["verification"], default=VerificationStatus.INFERRED
        )
        success = outcome_name in ("win", "caught") if outcome_name != "unknown" else None
        lesson = self._store_lesson(
            category=LESSON_CATEGORY_DEFAULT,
            content="; ".join(parts),
            verification=status,
            confidence=0.8 if status is VerificationStatus.VERIFIED else 0.5,
            evidence_refs=refs,
            subject=f"battle:{outcome_name}",
            tags=("battle", "outcome", outcome_name),
            success=success,
            persist=False,
        )
        if lesson:
            lesson["battle_id"] = battle_id
        self.save()
        return lesson

    # -- visible observations --------------------------------------------- #
    def record_move_observed(
        self,
        name: str,
        *,
        slot: int | None = None,
        user: str = "player",
        visible_text: str = "",
        effect_note: str = "",
        evidence_refs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Store a move name that was actually shown on the battle screen."""
        move_name = _clean(name, 120)
        if not move_name:
            raise ValueError("a move name visible on screen is required")
        refs = _as_str_list(evidence_refs)
        slot_value = _as_int(slot, 0)
        now = self._clock()
        entry: dict[str, Any] | None = None
        key = _norm_key(move_name)
        for item in self._data["moves"]:
            if _norm_key(item.get("name")) == key and _norm_key(item.get("user")) == _norm_key(user):
                entry = item
                break
        if entry is None:
            entry = {
                "name": move_name,
                "user": _clean(user, 60) or "player",
                "slots": [],
                "times_used": 0,
                "visible_texts": [],
                "note": "",
                "first_seen": now,
                "last_seen": now,
                "verification_status": VerificationStatus.VERIFIED.value,
                "evidence_refs": [],
            }
            self._data["moves"].append(entry)
            _trim(self._data["moves"], MAX_MOVES)
        entry["times_used"] = max(0, _as_int(entry.get("times_used"), 0)) + 1
        entry["last_seen"] = now
        if slot_value > 0 and slot_value not in entry.get("slots", []):
            entry.setdefault("slots", []).append(slot_value)
            entry["slots"] = sorted(set(int(s) for s in entry["slots"]))[:8]
        if visible_text:
            texts = entry.setdefault("visible_texts", [])
            texts.append(_clean(visible_text, 200))
            entry["visible_texts"] = texts[-4:]
        if effect_note:
            entry["note"] = _clean(effect_note, 300)
        entry["evidence_refs"] = (list(entry.get("evidence_refs") or []) + refs)[-8:]
        parts = [f"Move observed on screen: {move_name}"]
        if slot_value > 0:
            parts.append(f"slot={slot_value}")
        parts.append(f"chosen {entry['times_used']}x")
        if visible_text:
            parts.append(f"visible text: '{_clean(visible_text, 160)}'")
        lesson = self._store_lesson(
            category=LESSON_CATEGORY_DEFAULT,
            content="; ".join(parts),
            verification=VerificationStatus.VERIFIED,
            confidence=0.75,
            evidence_refs=refs,
            subject=f"move:{move_name}",
            tags=("move", "battle"),
            persist=False,
        )
        if effect_note:
            self.record_inference(
                f"Move '{move_name}' appears to work as follows: {effect_note}",
                confidence=0.4,
                evidence_refs=refs,
                subject=f"move:{move_name}",
            )
        self.save()
        return lesson

    def record_damage_observation(
        self,
        description: str,
        *,
        visible_text: str = "",
        move_name: str = "",
        evidence_refs: Iterable[str] | None = None,
        confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Store a visible damage / HP change (text or bar shape only)."""
        text = _clean(description, 400)
        if not text:
            raise ValueError("a visible damage/HP description is required")
        refs = _as_str_list(evidence_refs)
        record = {
            "description": text,
            "move_name": _clean(move_name, 120),
            "visible_text": _clean(visible_text, 300),
            "at": self._clock(),
            "verification_status": VerificationStatus.VERIFIED.value,
            "confidence": min(1.0, max(0.0, float(confidence))),
            "evidence_refs": refs,
        }
        self._data["damage"].append(record)
        _trim(self._data["damage"], MAX_DAMAGE)
        parts = [f"Visible damage/HP observation: {text}"]
        if move_name:
            parts.append(f"move={_clean(move_name, 120)}")
        if visible_text:
            parts.append(f"visible text: '{_clean(visible_text, 160)}'")
        lesson = self._store_lesson(
            category=LESSON_CATEGORY_DEFAULT,
            content="; ".join(parts),
            verification=VerificationStatus.VERIFIED,
            confidence=min(1.0, max(0.0, float(confidence))),
            evidence_refs=refs,
            subject="damage",
            tags=("damage", "hp", "battle"),
            persist=False,
        )
        self.save()
        return lesson

    def record_battle_text(
        self,
        text: str,
        *,
        evidence_refs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Store a battle message line the model read from the screenshot."""
        clean = _clean(text, 300)
        if not clean:
            raise ValueError("a visible battle text is required")
        refs = _as_str_list(evidence_refs)
        self._data["texts"].append({"text": clean, "at": self._clock(), "evidence_refs": refs})
        _trim(self._data["texts"], MAX_TEXTS)
        lesson = self._store_lesson(
            category=LESSON_CATEGORY_DEFAULT,
            content=f"Visible battle text: '{clean}'",
            verification=VerificationStatus.VERIFIED,
            confidence=0.7,
            evidence_refs=refs,
            subject="battle_text",
            tags=("text", "battle"),
            persist=False,
        )
        self.save()
        return lesson

    def record_team_member_observed(
        self,
        name: str,
        *,
        slot: int | None = None,
        hp_text: str = "",
        level_text: str = "",
        status_text: str = "",
        visible_text: str = "",
        evidence_refs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Store a party member seen on screen (never hidden stats)."""
        member_name = _clean(name, 120)
        if not member_name:
            raise ValueError("a party member name visible on screen is required")
        refs = _as_str_list(evidence_refs)
        now = self._clock()
        slot_value = _as_int(slot, 0)
        members: list[dict[str, Any]] = self._data["team"].setdefault("members", [])
        member: dict[str, Any] | None = None
        for item in members:
            same_slot = slot_value > 0 and _as_int(item.get("slot"), 0) == slot_value
            if same_slot or _norm_key(item.get("name")) == _norm_key(member_name):
                member = item
                break
        if member is None:
            member = {
                "name": member_name,
                "slot": slot_value if slot_value > 0 else None,
                "hp_text": "",
                "level_text": "",
                "status_text": "",
                "visible_text": "",
                "first_seen": now,
                "last_seen": now,
                "verification_status": VerificationStatus.VERIFIED.value,
                "evidence_refs": [],
            }
            members.append(member)
            _trim(members, MAX_TEAM_MEMBERS)
        member["last_seen"] = now
        if slot_value > 0:
            member["slot"] = slot_value
        for field_name, value in (
            ("hp_text", hp_text),
            ("level_text", level_text),
            ("status_text", status_text),
            ("visible_text", visible_text),
        ):
            if value:
                member[field_name] = _clean(value, 300)
        member["evidence_refs"] = (list(member.get("evidence_refs") or []) + refs)[-8:]
        self._data["team"]["updated_at"] = now
        parts = [f"Party member observed: {member_name}"]
        if slot_value > 0:
            parts.append(f"slot={slot_value}")
        for field_name, value in (("hp_text", hp_text), ("level_text", level_text), ("status_text", status_text)):
            if value:
                parts.append(f"{field_name}='{_clean(value, 120)}'")
        lesson = self._store_lesson(
            category="pokemon",
            content="; ".join(parts),
            verification=VerificationStatus.VERIFIED,
            confidence=0.7,
            evidence_refs=refs,
            subject=f"party:{member_name}",
            tags=("pokemon", "team"),
            persist=False,
        )
        self.save()
        return lesson

    def record_inference(
        self,
        content: str,
        *,
        confidence: float = 0.4,
        evidence_refs: Iterable[str] | None = None,
        subject: str = "",
        tags: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Store an explicitly labelled inference (never a fact)."""
        text = _clean(content, 500)
        if not text:
            raise ValueError("an inference content is required")
        lesson = self._store_lesson(
            category=LESSON_CATEGORY_DEFAULT,
            content=f"Inference (not directly observed): {text}",
            verification=VerificationStatus.INFERRED,
            confidence=min(1.0, max(0.0, float(confidence))),
            evidence_refs=evidence_refs,
            subject=subject,
            tags=("inference", *(list(tags or ()))),
            persist=False,
        )
        self.save()
        return lesson

    def record_lesson_entry(
        self,
        content: str,
        *,
        verification: str = "inferred",
        category: str = LESSON_CATEGORY_DEFAULT,
        confidence: float | None = None,
        evidence_refs: Iterable[str] | None = None,
        success: bool | None = None,
    ) -> dict[str, Any]:
        """Store an explicit model ``record_lesson`` call with its label."""
        label = _clean(verification, 20).lower()
        if label == "verified":
            status = VerificationStatus.VERIFIED
        elif label in ("hypothesis", "hypothetical"):
            status = VerificationStatus.HYPOTHESIS
        else:
            status = VerificationStatus.INFERRED
        if confidence is None:
            confidence = 0.8 if status is VerificationStatus.VERIFIED else 0.5
        lesson = self._store_lesson(
            category=category,
            content=content,
            verification=status,
            confidence=confidence,
            evidence_refs=evidence_refs,
            subject="lesson",
            tags=("lesson", label),
            success=success,
            persist=False,
        )
        self.save()
        return lesson

    # -- summaries / prompt rendering ------------------------------------- #
    def team_summary(self) -> dict[str, Any]:
        """Copy of the visible party knowledge."""
        team = self._data.get("team") or {}
        members = [copy.deepcopy(item) for item in team.get("members") or []]
        return {
            "count": len(members),
            "members": members,
            "updated_at": _as_float(team.get("updated_at"), 0.0),
        }

    def recent_battles(self, limit: int = 5) -> list[dict[str, Any]]:
        """Most recent battles (newest last), as copies."""
        count = max(1, _as_int(limit, 5))
        return [copy.deepcopy(item) for item in (self._data.get("battles") or [])[-count:]]

    def render_team_brief(self, *, max_chars: int = DEFAULT_TEAM_BRIEF_CHARS) -> str:
        """Compact Markdown team summary for the prompt."""
        budget = max(0, _as_int(max_chars, DEFAULT_TEAM_BRIEF_CHARS))
        if budget <= 0:
            return ""
        members = (self._data.get("team") or {}).get("members") or []
        lines = ["## VISIBLE TEAM"]
        if not members:
            lines.append("- no party member has been observed yet")
        else:
            for member in members[-MAX_TEAM_MEMBERS:]:
                detail = [str(member.get("name") or "?")]
                if member.get("slot"):
                    detail.append(f"slot {member['slot']}")
                for field_name, label in (
                    ("hp_text", "HP text"),
                    ("level_text", "level text"),
                    ("status_text", "status text"),
                ):
                    value = str(member.get(field_name) or "").strip()
                    if value:
                        detail.append(f"{label}: {value}")
                lines.append("- " + "; ".join(detail))
        text = "\n".join(lines)
        return text[:budget]

    def render_battle_brief(self, *, max_chars: int = DEFAULT_BATTLE_BRIEF_CHARS) -> str:
        """Compact Markdown battle summary (battles + observed moves/damage)."""
        budget = max(0, _as_int(max_chars, DEFAULT_BATTLE_BRIEF_CHARS))
        if budget <= 0:
            return ""
        data = self._data
        counters = data.get("counters") or {}
        lines = ["## BATTLE HISTORY"]
        if counters.get("battles"):
            lines.append(
                f"- battles handled: {_as_int(counters.get('battles'), 0)} "
                f"(win {_as_int(counters.get('wins'), 0)}, "
                f"loss {_as_int(counters.get('losses'), 0)}, "
                f"caught {_as_int(counters.get('catches'), 0)}, "
                f"ran {_as_int(counters.get('runs'), 0)})"
            )
        recent = (data.get("battles") or [])[-4:]
        if not recent and not counters.get("battles"):
            lines.append("- no battle handled yet")
        for battle in recent:
            summary = f"- {battle.get('outcome', 'unknown')} after {_as_int(battle.get('turns'), 0)} turn(s)"
            text = str(battle.get("visible_text") or "").strip()
            if text:
                summary += f"; visible: '{text[:100]}'"
            lines.append(summary)
        moves = data.get("moves") or []
        if moves:
            lines.append("## MOVES OBSERVED ON SCREEN")
            for move in moves[-6:]:
                line = f"- {move.get('name')}"
                if move.get("slots"):
                    line += f" (slot {move['slots'][-1]})"
                line += f" chosen {_as_int(move.get('times_used'), 0)}x"
                note = str(move.get("note") or "").strip()
                if note:
                    line += f"; note: {note[:100]}"
                lines.append(line)
        damage = data.get("damage") or []
        if damage:
            lines.append("## VISIBLE DAMAGE / HP")
            for record in damage[-4:]:
                lines.append(f"- {str(record.get('description') or '')[:140]}")
        inferred = [
            item
            for item in (data.get("lessons") or [])
            if item.get("verification_status") == VerificationStatus.INFERRED.value
        ]
        if inferred:
            lines.append("## INFERRED NOTES (unverified)")
            for record in inferred[-3:]:
                lines.append(f"- {str(record.get('content') or '')[:140]}")
        text = "\n".join(lines)
        return text[:budget]

    def render_context(self, query: str | None = None, *, max_chars: int | None = None) -> str:
        """Render the memory block that is injected into the battle prompt.

        The generic :class:`AgentMemory` rendering comes first (goal, places,
        lessons), then the battle-specific briefs, hard-truncated to
        ``max_chars`` so one rich memory can never crowd out the goal.
        """
        budget = DEFAULT_RENDER_CHARS if max_chars is None else max(0, _as_int(max_chars, 0))
        if budget <= 0:
            return ""
        parts: list[str] = []
        generic = ""
        render = getattr(self.agent_memory, "render_context", None) if self.agent_memory is not None else None
        if callable(render):
            try:
                generic = str(render(query, max_chars=max(400, budget // 2)) or "").strip()
            except TypeError:
                try:
                    generic = str(render() or "").strip()
                except Exception:  # noqa: BLE001 - memory rendering must not break a prompt
                    generic = ""
            except Exception:  # noqa: BLE001
                generic = ""
        if generic:
            parts.append(generic[: max(1, budget // 2)])
        battle = self.render_battle_brief(max_chars=max(300, budget // 2))
        if battle:
            parts.append(battle)
        team = self.render_team_brief(max_chars=max(200, budget // 4))
        if team:
            parts.append(team)
        return "\n\n".join(parts)[:budget]

    def render_for_prompt(self, query: str | None = None, *, max_chars: int | None = None) -> str:
        """Alias kept for prompt-context duck typing."""
        return self.render_context(query, max_chars=max_chars)

    # -- long-term hooks (structured but NOT validated on a real run) ----- #
    def _append_long_term(self, bucket: str, record: Mapping[str, Any]) -> dict[str, Any]:
        bucket_name = str(bucket)
        buckets = self._data.setdefault("long_term", {})
        items = buckets.setdefault(bucket_name, [])
        entry = {
            **dict(record),
            "at": self._clock(),
            "status": "structured_only",
            "validated": False,
        }
        items.append(entry)
        _trim(items, MAX_LONG_TERM)
        self.save()
        return copy.deepcopy(entry)

    def record_training_observation(
        self,
        description: str,
        *,
        success: bool | None = None,
        notes: str = "",
        evidence_refs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Structured-only hook: a training visit/repetition observation."""
        text = _clean(description, 400)
        if not text:
            raise ValueError("a training observation description is required")
        return self._append_long_term(
            "training",
            {
                "kind": "training",
                "description": text,
                "success": success if isinstance(success, bool) else None,
                "notes": _clean(notes, 200),
                "evidence_refs": _as_str_list(evidence_refs),
            },
        )

    def record_evolution_observed(
        self,
        previous_form: str,
        new_form: str,
        *,
        visible_text: str = "",
        evidence_refs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Structured-only hook: an evolution seen on screen."""
        old = _clean(previous_form, 120)
        new = _clean(new_form, 120)
        if not old or not new:
            raise ValueError("both forms are required for an evolution record")
        return self._append_long_term(
            "evolutions",
            {
                "kind": "evolution",
                "description": f"{old} changed form to {new} on screen",
                "visible_text": _clean(visible_text, 300),
                "evidence_refs": _as_str_list(evidence_refs),
            },
        )

    def record_move_learned(
        self,
        move_name: str,
        *,
        replaced_move: str = "",
        slot: int | None = None,
        visible_text: str = "",
        evidence_refs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Structured-only hook: a move learned/replaced on the move screen."""
        name = _clean(move_name, 120)
        if not name:
            raise ValueError("a learned move name is required")
        detail = f"learned move {name}"
        replaced = _clean(replaced_move, 120)
        if replaced:
            detail += f" (replacing {replaced})"
        return self._append_long_term(
            "moves_learned",
            {
                "kind": "move_learned",
                "description": detail,
                "slot": _as_int(slot, 0) or None,
                "replaced_move": replaced,
                "visible_text": _clean(visible_text, 300),
                "evidence_refs": _as_str_list(evidence_refs),
            },
        )

    def record_team_change(
        self,
        description: str,
        *,
        slot: int | None = None,
        visible_text: str = "",
        evidence_refs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Structured-only hook: party order/team composition change."""
        text = _clean(description, 400)
        if not text:
            raise ValueError("a team change description is required")
        return self._append_long_term(
            "team_changes",
            {
                "kind": "team_change",
                "description": text,
                "slot": _as_int(slot, 0) or None,
                "visible_text": _clean(visible_text, 300),
                "evidence_refs": _as_str_list(evidence_refs),
            },
        )

    def record_item_use(
        self,
        item_name: str,
        *,
        target: str = "",
        battle_context: str = "",
        visible_text: str = "",
        evidence_refs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Structured-only hook: an item used during a battle."""
        name = _clean(item_name, 120)
        if not name:
            raise ValueError("an item name is required")
        return self._append_long_term(
            "item_uses",
            {
                "kind": "item_use",
                "description": f"used item {name}"
                + (f" on {_clean(target, 120)}" if target else ""),
                "target": _clean(target, 120),
                "battle_context": _clean(battle_context, 200),
                "visible_text": _clean(visible_text, 300),
                "evidence_refs": _as_str_list(evidence_refs),
            },
        )

    def long_term_summary(self) -> dict[str, int]:
        """Counts of the structured long-term hook records."""
        buckets = self._data.get("long_term") or {}
        return {str(kind): len(items or []) for kind, items in buckets.items()}

    # -- snapshot / restore ----------------------------------------------- #
    def snapshot(self) -> dict[str, Any]:
        """Full JSON-serialisable snapshot (copy)."""
        return {
            "version": POKEMON_MEMORY_VERSION,
            "data": copy.deepcopy(self._data),
            "branch_id": self._branch_id,
            "saved_at": self._clock(),
        }

    def restore(self, snapshot: Mapping[str, Any] | None) -> bool:
        """Restore a snapshot defensively; True when a document was loaded."""
        if not isinstance(snapshot, Mapping):
            return False
        payload = snapshot.get("data")
        if not isinstance(payload, Mapping):
            return False
        self._data = _normalise_data(payload)
        self._branch_id = _clean(snapshot.get("branch_id"), 60)
        self._rebuild_knowledge()
        self.save()
        return True


__all__ = [
    "POKEMON_MEMORY_VERSION",
    "POKEMON_MEMORY_FILENAME",
    "PokemonMemory",
    "MAX_BATTLES",
    "MAX_MOVES",
    "MAX_DAMAGE",
    "MAX_TEAM_MEMBERS",
    "MAX_LESSONS",
    "OUTCOMES",
]
