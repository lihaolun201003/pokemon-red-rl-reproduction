"""Persistent long-term memory for the DeepSeek game agent (owner: memory-guard).

``AgentMemory`` is the agent's own notebook.  Everything stored in it is derived
from *observations only*: screenshots the model looked at, in-game dialogue the
model read on screen, places the agent walked through, lessons learned from its
own attempts.  It must never contain RAM values, map ids, story flags, hidden
party data or external walkthrough knowledge — those are the responsibilities of
the verification-only :mod:`llm_agent.progress` module, which this module does
not import.

Persistence lives under :attr:`MemoryConfig.root` (``llm_agent/memory/`` by
default, runtime data excluded from git):

* ``memory.json``          — versioned structured memory, atomically replaced.
* ``journal.jsonl``        — append-only observation journal, trimmed to
  ``MemoryConfig.max_journal`` lines.
* ``reports/memory_latest.md`` — human-readable mirror (when
  ``MemoryConfig.mirror_to_run``), written inside the run directory if one was
  injected, otherwise inside the memory root.

:meth:`AgentMemory.render_context` is the **only** memory text the runtime sends
to DeepSeek.  It is deterministic, compact Markdown: current goal (+ plan),
known places, recent/most relevant discoveries, dialogues, lessons, open
questions and an archive digest, hard-truncated to ``max_chars``.  The goal and
the most recent discoveries have the highest retention priority.

Corrupt files never crash a run: a bad ``memory.json`` is backed up to
``memory.corrupt-<timestamp>.json``, memory starts fresh and a recovery event is
recorded; the journal is preserved.  Individual corrupt journal lines are
skipped.  All writes are atomic (``*.tmp`` + :func:`os.replace`).
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from llm_agent import contracts
from llm_agent.config import MemoryConfig

MEMORY_VERSION = 1
MEMORY_FILENAME = "memory.json"
JOURNAL_FILENAME = "journal.jsonl"
REPORT_FILENAME = "memory_latest.md"
MAX_EVENTS = 500
MAX_GOAL_HISTORY = 100
MAX_ARCHIVE_CHARS = 12000
MAX_OPEN_QUESTIONS = 50
MAX_REPORT_ITEMS = 250

#: category -> MemoryConfig attribute holding its cap
_SECTION_CAPS: dict[str, str] = {
    "places": "max_places",
    "discoveries": "max_discoveries",
    "dialogues": "max_dialogue",
    "lessons": "max_lessons",
}

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
# Small pure helpers
# --------------------------------------------------------------------------- #

def _now() -> float:
    """Current unix timestamp (single source of truth for entry timestamps)."""
    return time.time()


def _make_id(prefix: str) -> str:
    """Opaque unique entry id (never rendered, only for snapshot/merge safety)."""
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def _clean(text: Any, limit: int = 0) -> str:
    """Collapse whitespace and hard-truncate ``text`` to ``limit`` characters."""
    if text is None:
        return ""
    value = _WS_RE.sub(" ", str(text)).strip()
    if limit and limit > 0:
        value = value[: int(limit)]
    return value


def _clean_block(text: Any, limit: int = 0) -> str:
    """Like :func:`_clean` but keeps internal newlines (multiline summaries)."""
    if text is None:
        return ""
    value = str(text).strip()
    if limit and limit > 0:
        value = value[: int(limit)]
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _norm_key(text: Any) -> str:
    """Normalised case/space-insensitive key used for dedupe and section order."""
    if text is None:
        return ""
    return _WS_RE.sub(" ", str(text).strip().casefold())


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


def _format_ts(value: float) -> str:
    """Human-readable local timestamp for reports (empty for missing times)."""
    if not value or value <= 0:
        return "?"
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "?"


# --------------------------------------------------------------------------- #
# Fresh / normalised state
# --------------------------------------------------------------------------- #

def _fresh_data() -> dict[str, Any]:
    """Return an empty, valid memory document."""
    return {
        "version": MEMORY_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "goal": {"text": "", "reason": "", "at": 0.0},
        "goal_history": [],
        "places": [],
        "discoveries": [],
        "dialogues": [],
        "lessons": [],
        "open_questions": [],
        "events": [],
        "archived_summary": "",
        "counters": {"actions": 0, "journal_entries": 0, "compressions": 0},
    }


def _normalize_goal(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"text": "", "reason": "", "at": 0.0}
    return {
        "text": _clean(raw.get("text"), contracts.MAX_GOAL_CHARS),
        "reason": _clean(raw.get("reason"), contracts.MAX_REASON_CHARS),
        "at": _as_float(raw.get("at"), 0.0),
    }


def _normalize_goal_history(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        text = _clean(item.get("text"), contracts.MAX_GOAL_CHARS)
        if not text:
            continue
        out.append(
            {
                "text": text,
                "reason": _clean(item.get("reason"), contracts.MAX_REASON_CHARS),
                "at": _as_float(item.get("at"), 0.0),
            }
        )
    return out[-MAX_GOAL_HISTORY:]


def _normalize_places(raw: Any, cfg: MemoryConfig) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        name = _clean(item.get("name"), contracts.MAX_PLACE_CHARS)
        if not name:
            continue
        first_seen = _as_float(item.get("first_seen"), 0.0)
        out.append(
            {
                "id": _clean(item.get("id"), 40) or _make_id("p-"),
                "name": name,
                "description": _clean(item.get("description"), min(cfg.max_entry_chars, 500)),
                "signature": _clean(item.get("signature"), 80),
                "first_seen": first_seen,
                "last_seen": _as_float(item.get("last_seen"), first_seen),
                "visits": max(1, _as_int(item.get("visits"), 1)),
                "seq": max(0, _as_int(item.get("seq"), index + 1)),
            }
        )
    return out


def _normalize_discoveries(raw: Any, cfg: MemoryConfig) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    limit = min(cfg.max_entry_chars, contracts.MAX_DISCOVERY_CHARS)
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        text = _clean(item.get("text"), limit)
        if not text:
            continue
        kind = item.get("kind")
        if kind not in contracts.DISCOVERY_KINDS:
            kind = "other"
        out.append(
            {
                "id": _clean(item.get("id"), 40) or _make_id("d-"),
                "kind": str(kind),
                "text": text,
                "place": _clean(item.get("place"), contracts.MAX_PLACE_CHARS),
                "at": _as_float(item.get("at"), 0.0),
                "seq": max(0, _as_int(item.get("seq"), index + 1)),
            }
        )
    return out


def _normalize_dialogues(raw: Any, cfg: MemoryConfig) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        text = _clean(item.get("text"), cfg.max_entry_chars)
        if not text:
            continue
        out.append(
            {
                "id": _clean(item.get("id"), 40) or _make_id("g-"),
                "speaker": _clean(item.get("speaker"), 120) or "NPC",
                "text": text,
                "place": _clean(item.get("place"), contracts.MAX_PLACE_CHARS),
                "at": _as_float(item.get("at"), 0.0),
                "seq": max(0, _as_int(item.get("seq"), index + 1)),
            }
        )
    return out


def _normalize_lessons(raw: Any, cfg: MemoryConfig) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        text = _clean(item.get("text"), cfg.max_entry_chars)
        if not text:
            continue
        success = item.get("success")
        if success is not None and not isinstance(success, bool):
            success = None
        out.append(
            {
                "id": _clean(item.get("id"), 40) or _make_id("l-"),
                "text": text,
                "success": success,
                "context": _clean(item.get("context"), 300),
                "at": _as_float(item.get("at"), 0.0),
                "seq": max(0, _as_int(item.get("seq"), index + 1)),
            }
        )
    return out


def _normalize_open_questions(raw: Any, cfg: MemoryConfig) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        text = _clean(item.get("text"), min(cfg.max_entry_chars, 300))
        if not text:
            continue
        out.append(
            {
                "id": _clean(item.get("id"), 40) or _make_id("q-"),
                "text": text,
                "at": _as_float(item.get("at"), 0.0),
                "seq": max(0, _as_int(item.get("seq"), index + 1)),
            }
        )
    return out


def _normalize_events(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        out.append(
            {
                "kind": _clean(item.get("kind"), 60) or "event",
                "text": _clean(item.get("text"), 500),
                "at": _as_float(item.get("at"), 0.0),
            }
        )
    return out[-MAX_EVENTS:]


def _normalize_counters(raw: Any) -> dict[str, int]:
    counters = {"actions": 0, "journal_entries": 0, "compressions": 0}
    if isinstance(raw, Mapping):
        for key in counters:
            counters[key] = max(0, _as_int(raw.get(key), 0))
    return counters


def _normalize_journal_entry(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    return {
        "at": _as_float(raw.get("at"), 0.0),
        "kind": _clean(raw.get("kind"), 60) or "event",
        "text": _clean(raw.get("text"), 500),
    }


def _normalize_data(raw: Any, cfg: MemoryConfig) -> dict[str, Any]:
    """Return a valid memory document built from ``raw`` (tolerates junk)."""
    data = _fresh_data()
    if not isinstance(raw, Mapping):
        return data
    data["created_at"] = _as_float(raw.get("created_at"), data["created_at"])
    data["updated_at"] = _as_float(raw.get("updated_at"), data["created_at"])
    data["goal"] = _normalize_goal(raw.get("goal"))
    data["goal_history"] = _normalize_goal_history(raw.get("goal_history"))
    data["places"] = _normalize_places(raw.get("places"), cfg)
    data["discoveries"] = _normalize_discoveries(raw.get("discoveries"), cfg)
    data["dialogues"] = _normalize_dialogues(raw.get("dialogues"), cfg)
    data["lessons"] = _normalize_lessons(raw.get("lessons"), cfg)
    data["open_questions"] = _normalize_open_questions(raw.get("open_questions"), cfg)
    data["events"] = _normalize_events(raw.get("events"))
    data["archived_summary"] = _clean_block(raw.get("archived_summary"), MAX_ARCHIVE_CHARS)
    data["counters"] = _normalize_counters(raw.get("counters"))
    return data


# --------------------------------------------------------------------------- #
# Default render formatting (kept small and deterministic)
# --------------------------------------------------------------------------- #

def _format_place(entry: Mapping[str, Any]) -> str:
    name = str(entry.get("name") or "").strip()
    visits = _as_int(entry.get("visits"), 1)
    desc = str(entry.get("description") or "").strip()
    line = f"- {name}"
    if visits > 1:
        line += f" ({visits}x)"
    if desc:
        line += f" — {desc}"
    return line


def _format_discovery(entry: Mapping[str, Any]) -> str:
    kind = str(entry.get("kind") or "other")
    text = str(entry.get("text") or "").strip()
    place = str(entry.get("place") or "").strip()
    line = f"- [{kind}] {text}"
    if place:
        line += f" ({place})"
    return line


def _format_dialogue(entry: Mapping[str, Any]) -> str:
    speaker = str(entry.get("speaker") or "NPC")
    text = str(entry.get("text") or "").strip()
    place = str(entry.get("place") or "").strip()
    line = f'- {speaker}: "{text}"'
    if place:
        line += f" ({place})"
    return line


def _format_lesson(entry: Mapping[str, Any]) -> str:
    success = entry.get("success")
    flag = "[worked]" if success is True else ("[failed]" if success is False else "[note]")
    text = str(entry.get("text") or "").strip()
    context = str(entry.get("context") or "").strip()
    line = f"- {flag} {text}"
    if context:
        line += f" ({context})"
    return line


def _format_question(entry: Mapping[str, Any]) -> str:
    return f"- {str(entry.get('text') or '').strip()}"


# --------------------------------------------------------------------------- #
# AgentMemory
# --------------------------------------------------------------------------- #

class AgentMemory:
    """Persistent, observation-derived long-term memory for one agent run.

    The instance auto-loads on construction (so a fresh ``AgentMemory`` can
    never silently clobber an existing memory file) and :meth:`load` can be
    called again to re-read from disk.  Mutating methods update the in-memory
    document; call :meth:`save` to persist ``memory.json`` (plus the journal
    trim and the markdown mirror).  Journal lines are appended immediately so an
    interrupted run keeps its observation log.

    Parameters
    ----------
    cfg:
        :class:`~llm_agent.config.MemoryConfig` with the memory root and all
        caps/budgets.
    run_dir:
        Optional run directory; when given the markdown mirror and any explicit
        relative report paths live under ``run_dir/reports/``.
    summarizer:
        Optional injected budget-gated summarizer used by
        :meth:`maybe_compress` when ``force=True``.  It receives a list of
        compact strings and returns one summary string.  It is optional: when
        absent, compression falls back to a deterministic rule-based archive.
    """

    def __init__(
        self,
        cfg: MemoryConfig,
        *,
        run_dir: str | Path | None = None,
        summarizer: Callable[[list[str]], str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.run_dir: Path | None = Path(run_dir) if run_dir else None
        self.summarizer = summarizer

        self._data: dict[str, Any] = _fresh_data()
        self._seq: int = 1
        self._journal_entries: list[dict[str, Any]] = []
        self._journal_file_lines: int = 0
        self._journal_needs_rewrite: bool = False
        self.journal_skipped: int = 0
        self.journal_errors: list[str] = []
        self.load_errors: list[str] = []
        self.recovered_from: str | None = None
        self.last_report_path: Path | None = None
        self._actions_since_compress: int = 0

        self.load()

    # -- paths ------------------------------------------------------------- #

    @property
    def memory_path(self) -> Path:
        """Path of the structured ``memory.json`` file."""
        return self.root / MEMORY_FILENAME

    @property
    def journal_path(self) -> Path:
        """Path of the append-only ``journal.jsonl`` file."""
        return self.root / JOURNAL_FILENAME

    def report_path(self) -> Path:
        """Default markdown report path (run dir when injected, else root)."""
        if self.run_dir is not None:
            return self.run_dir / "reports" / REPORT_FILENAME
        return self.root / "reports" / REPORT_FILENAME

    @property
    def actions_since_compress(self) -> int:
        """Actions accumulated since the last compression pass."""
        return self._actions_since_compress

    # -- lifecycle --------------------------------------------------------- #

    def _ensure_root(self) -> None:
        """Create the memory root (and its local ``.gitignore``) if needed."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            gitignore = self.root / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "# Agent runtime memory; never commit gameplay data.\n*\n!.gitignore\n",
                    encoding="utf-8",
                )
        except OSError as exc:  # pragma: no cover - disk-level failure
            self.load_errors.append(f"memory root unavailable: {exc}")

    def load(self) -> None:
        """Read ``memory.json`` and ``journal.jsonl``; never raise on corruption.

        A malformed/unsupported ``memory.json`` is moved to
        ``memory.corrupt-<timestamp>.json``, memory starts fresh and a
        ``recovery`` event is recorded.  The journal is preserved; individual
        corrupt journal lines are skipped.
        """
        self.load_errors = []
        self.recovered_from = None
        self.journal_errors = []
        self._ensure_root()

        raw: dict[str, Any] | None = None
        recovered = False
        path = self.memory_path
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                parsed = json.loads(text) if text.strip() else None
                if not isinstance(parsed, dict):
                    raise ValueError("memory.json does not contain a JSON object")
                version = parsed.get("version", MEMORY_VERSION)
                if version != MEMORY_VERSION:
                    raise ValueError(f"unsupported memory.json version {version!r}")
                raw = parsed
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                self.load_errors.append(
                    f"memory.json corrupt ({type(exc).__name__}: {exc}); started fresh"
                )
                self._backup_corrupt(path)
                raw = None
                recovered = True

        self._data = _normalize_data(raw or {}, self.cfg)
        self._seq = self._compute_next_seq()
        self._actions_since_compress = 0
        self._load_journal()

        if recovered:
            reason = self.load_errors[-1] if self.load_errors else "memory.json corrupt"
            self.record_event("recovery", f"memory recovered from corrupt file: {reason}")
            self._write_memory_json()

    def save(self) -> None:
        """Persist ``memory.json`` atomically, trim the journal, write report."""
        self._data["updated_at"] = _now()
        self._prune()
        self._write_memory_json()
        self._sync_journal()
        if self.cfg.mirror_to_run:
            try:
                self.save_report(self.report_path())
            except OSError as exc:  # pragma: no cover - disk-level failure
                self.journal_errors.append(f"report write failed: {exc}")

    def _prune(self) -> None:
        """Keep unbounded side lists bounded (events/history/questions)."""
        events = self._data["events"]
        if len(events) > MAX_EVENTS:
            del events[: len(events) - MAX_EVENTS]
        history = self._data["goal_history"]
        if len(history) > MAX_GOAL_HISTORY:
            del history[: len(history) - MAX_GOAL_HISTORY]
        questions = self._data["open_questions"]
        if len(questions) > MAX_OPEN_QUESTIONS:
            removed = questions[: len(questions) - MAX_OPEN_QUESTIONS]
            del questions[: len(questions) - MAX_OPEN_QUESTIONS]
            folded = "; ".join(
                str(entry.get("text") or "") for entry in removed if entry.get("text")
            )[:1200]
            if folded:
                self._append_archive(f"open questions: {folded}")

    def _write_memory_json(self) -> None:
        payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        _atomic_write_text(self.memory_path, payload + "\n")

    def _backup_corrupt(self, path: Path) -> str | None:
        """Move a corrupt memory file to ``memory.corrupt-<timestamp>.json``."""
        now = _now()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        stamp += f"_{int(now * 1_000_000) % 1_000_000:06d}"
        target = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
        attempt = 1
        while target.exists():
            target = path.with_name(f"{path.stem}.corrupt-{stamp}_{attempt}{path.suffix}")
            attempt += 1
        try:
            os.replace(path, target)
            self.recovered_from = str(target)
            return str(target)
        except OSError as exc:  # pragma: no cover - disk-level failure
            self.load_errors.append(f"corrupt backup failed: {exc}")
            return None

    # -- journal ----------------------------------------------------------- #

    def _load_journal(self) -> None:
        self._journal_entries = []
        self._journal_file_lines = 0
        self._journal_needs_rewrite = False
        self.journal_skipped = 0
        path = self.journal_path
        if not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - disk-level failure
            self.load_errors.append(f"journal unreadable: {exc}")
            return
        skipped = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            entry = _normalize_journal_entry(item)
            if entry is None:
                skipped += 1
                continue
            self._journal_entries.append(entry)
        self.journal_skipped = skipped
        self._journal_file_lines = len(self._journal_entries) + skipped
        if skipped:
            self._journal_needs_rewrite = True
        max_journal = _as_int(self.cfg.max_journal, 0)
        if max_journal > 0 and len(self._journal_entries) > max_journal:
            self._journal_entries = self._journal_entries[-max_journal:]
            self._journal_needs_rewrite = True

    def _append_journal(self, kind: str, text: str) -> None:
        entry = {
            "at": _now(),
            "kind": _clean(kind, 60) or "event",
            "text": _clean(text, 500),
        }
        self._journal_entries.append(entry)
        self._data["counters"]["journal_entries"] = len(self._journal_entries)
        try:
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._journal_file_lines += 1
        except OSError as exc:  # pragma: no cover - disk-level failure
            self.journal_errors.append(f"journal append failed: {exc}")
        max_journal = _as_int(self.cfg.max_journal, 0)
        if max_journal > 0 and len(self._journal_entries) > max_journal * 2:
            self._rewrite_journal()

    def _rewrite_journal(self) -> None:
        max_journal = _as_int(self.cfg.max_journal, 0)
        entries = self._journal_entries
        if max_journal > 0:
            entries = entries[-max_journal:]
        body = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
        try:
            _atomic_write_text(self.journal_path, body)
        except OSError as exc:  # pragma: no cover - disk-level failure
            self.journal_errors.append(f"journal rewrite failed: {exc}")
            return
        self._journal_entries = entries
        self._journal_file_lines = len(entries)
        self._journal_needs_rewrite = False

    def _sync_journal(self) -> None:
        max_journal = _as_int(self.cfg.max_journal, 0)
        too_long = max_journal > 0 and len(self._journal_entries) > max_journal
        out_of_sync = self._journal_file_lines != len(self._journal_entries)
        if self._journal_needs_rewrite or too_long or out_of_sync:
            self._rewrite_journal()

    def recent_journal(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent journal entries (oldest first), as copies."""
        count = max(1, _as_int(limit, 20))
        return [copy.deepcopy(entry) for entry in self._journal_entries[-count:]]

    # -- sequence ids ------------------------------------------------------ #

    def _compute_next_seq(self) -> int:
        highest = 0
        for category in ("places", "discoveries", "dialogues", "lessons", "open_questions"):
            for entry in self._data.get(category) or []:
                highest = max(highest, _as_int(entry.get("seq"), 0))
        return highest + 1

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    # -- goal -------------------------------------------------------------- #

    def get_goal(self) -> dict[str, Any]:
        """Current goal record (``{"text", "reason", "at"}``; empty when unset)."""
        return copy.deepcopy(self._data["goal"])

    def set_goal(self, goal: str, reason: str = "") -> dict:
        """Set the current goal (previous goal is kept in ``goal_history``)."""
        text = _clean(goal, contracts.MAX_GOAL_CHARS)
        reason_text = _clean(reason, contracts.MAX_REASON_CHARS)
        if not text:
            return self.get_goal()
        now = _now()
        previous = self._data["goal"]
        previous_text = str(previous.get("text") or "")
        if previous_text and previous_text != text:
            self._data["goal_history"].append(
                {
                    "text": previous_text,
                    "reason": str(previous.get("reason") or ""),
                    "at": _as_float(previous.get("at"), now),
                }
            )
        # Re-setting the identical goal refreshes reason/timestamp only.
        self._data["goal"] = {"text": text, "reason": reason_text, "at": now}
        self._append_journal("goal", text)
        return self.get_goal()

    # -- records ----------------------------------------------------------- #

    def _find_duplicate(self, category: str, predicate: Callable[[Mapping[str, Any]], bool]) -> dict | None:
        for entry in self._data.get(category) or []:
            try:
                if predicate(entry):
                    return entry
            except (TypeError, ValueError, KeyError):
                continue
        return None

    def record_discovery(self, text: str, *, kind: str = "other", place: str = "") -> dict:
        """Record something the agent saw/learned on screen; identical text dedupes."""
        stored_text = _clean(
            text, min(_as_int(self.cfg.max_entry_chars, 1200), contracts.MAX_DISCOVERY_CHARS)
        )
        if not stored_text:
            return {}
        if kind not in contracts.DISCOVERY_KINDS:
            kind = "other"
        kind = str(kind)
        stored_place = _clean(place, contracts.MAX_PLACE_CHARS)
        key = _norm_key(stored_text)
        existing = self._find_duplicate(
            "discoveries", lambda item: _norm_key(item.get("text")) == key and item.get("kind") == kind
        )
        if existing is not None:
            existing["at"] = _now()
            existing["seq"] = self._next_seq()
            if stored_place:
                existing["place"] = stored_place
            self._append_journal("discovery", stored_text)
            return copy.deepcopy(existing)
        entry = {
            "id": _make_id("d-"),
            "kind": kind,
            "text": stored_text,
            "place": stored_place,
            "at": _now(),
            "seq": self._next_seq(),
        }
        self._data["discoveries"].append(entry)
        self._append_journal("discovery", stored_text)
        return copy.deepcopy(entry)

    def record_dialogue(self, speaker: str, text: str, *, place: str = "") -> dict:
        """Record an in-game dialogue line (only what was visible on screen)."""
        stored_text = _clean(text, self.cfg.max_entry_chars)
        if not stored_text:
            return {}
        stored_speaker = _clean(speaker, 120) or "NPC"
        stored_place = _clean(place, contracts.MAX_PLACE_CHARS)
        key = (_norm_key(stored_speaker), _norm_key(stored_text))
        existing = self._find_duplicate(
            "dialogues",
            lambda item: (_norm_key(item.get("speaker")), _norm_key(item.get("text"))) == key,
        )
        if existing is not None:
            existing["at"] = _now()
            existing["seq"] = self._next_seq()
            if stored_place:
                existing["place"] = stored_place
            self._append_journal("dialogue", f"{stored_speaker}: {stored_text}")
            return copy.deepcopy(existing)
        entry = {
            "id": _make_id("g-"),
            "speaker": stored_speaker,
            "text": stored_text,
            "place": stored_place,
            "at": _now(),
            "seq": self._next_seq(),
        }
        self._data["dialogues"].append(entry)
        self._append_journal("dialogue", f"{stored_speaker}: {stored_text}")
        return copy.deepcopy(entry)

    def record_lesson(self, text: str, *, success: bool | None = None, context: str = "") -> dict:
        """Record a lesson learned from the agent's own attempts.

        ``success`` is ``True`` (worked), ``False`` (failed) or ``None`` (note).
        """
        stored_text = _clean(text, self.cfg.max_entry_chars)
        if not stored_text:
            return {}
        success_flag = success if isinstance(success, bool) else None
        stored_context = _clean(context, 300)
        key = _norm_key(stored_text)
        existing = self._find_duplicate("lessons", lambda item: _norm_key(item.get("text")) == key)
        if existing is not None:
            existing["at"] = _now()
            existing["seq"] = self._next_seq()
            existing["success"] = success_flag
            if stored_context:
                existing["context"] = stored_context
            self._append_journal("lesson", stored_text)
            return copy.deepcopy(existing)
        entry = {
            "id": _make_id("l-"),
            "text": stored_text,
            "success": success_flag,
            "context": stored_context,
            "at": _now(),
            "seq": self._next_seq(),
        }
        self._data["lessons"].append(entry)
        self._append_journal("lesson", stored_text)
        return copy.deepcopy(entry)

    def note_place(self, name: str, description: str = "", *, signature: str = "") -> dict:
        """Create or refresh a known place (visited places keep first/last seen)."""
        stored_name = _clean(name, contracts.MAX_PLACE_CHARS)
        if not stored_name:
            return {}
        now = _now()
        key = _norm_key(stored_name)
        existing = self._find_duplicate("places", lambda item: _norm_key(item.get("name")) == key)
        if existing is not None:
            existing["last_seen"] = now
            existing["visits"] = max(1, _as_int(existing.get("visits"), 1)) + 1
            existing["seq"] = self._next_seq()
            if description:
                existing["description"] = _clean(description, min(self.cfg.max_entry_chars, 500))
            if signature:
                existing["signature"] = _clean(signature, 80)
            self._append_journal("place", stored_name)
            return copy.deepcopy(existing)
        entry = {
            "id": _make_id("p-"),
            "name": stored_name,
            "description": _clean(description, min(self.cfg.max_entry_chars, 500)),
            "signature": _clean(signature, 80),
            "first_seen": now,
            "last_seen": now,
            "visits": 1,
            "seq": self._next_seq(),
        }
        self._data["places"].append(entry)
        self._append_journal("place", stored_name)
        return copy.deepcopy(entry)

    def record_event(self, kind: str, text: str) -> None:
        """Append a chronological journal event (recovery, compress, errors, ...)."""
        entry = {
            "kind": _clean(kind, 60) or "event",
            "text": _clean(text, 500),
            "at": _now(),
        }
        self._data["events"].append(entry)
        if len(self._data["events"]) > MAX_EVENTS:
            del self._data["events"][: len(self._data["events"]) - MAX_EVENTS]
        self._append_journal(entry["kind"], entry["text"])

    # -- open questions (additive helpers used by the runtime/report) ------ #

    def add_open_question(self, text: str) -> dict:
        """Record an unresolved question (deduped by normalised text)."""
        stored = _clean(text, min(self.cfg.max_entry_chars, 300))
        if not stored:
            return {}
        key = _norm_key(stored)
        existing = self._find_duplicate("open_questions", lambda item: _norm_key(item.get("text")) == key)
        if existing is not None:
            existing["at"] = _now()
            existing["seq"] = self._next_seq()
            return copy.deepcopy(existing)
        entry = {"id": _make_id("q-"), "text": stored, "at": _now(), "seq": self._next_seq()}
        self._data["open_questions"].append(entry)
        self._append_journal("question", stored)
        return copy.deepcopy(entry)

    def resolve_open_question(self, query: str) -> bool:
        """Remove the first open question containing ``query``; True when removed."""
        needle = _norm_key(query)
        if not needle:
            return False
        for index, entry in enumerate(self._data["open_questions"]):
            if needle in _norm_key(entry.get("text")):
                removed = self._data["open_questions"].pop(index)
                self._append_journal("question_resolved", str(removed.get("text") or ""))
                return True
        return False

    # -- retrieval --------------------------------------------------------- #

    def _entry_haystack(self, category: str, entry: Mapping[str, Any]) -> str:
        fields = {
            "places": ("name", "description", "signature"),
            "discoveries": ("text", "kind", "place"),
            "dialogues": ("text", "speaker", "place"),
            "lessons": ("text", "context"),
            "open_questions": ("text",),
        }.get(category, ("text",))
        parts = [str(entry.get(field) or "") for field in fields]
        return " ".join(parts).casefold()

    def _score_entry(self, category: str, entry: Mapping[str, Any], tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        haystack = self._entry_haystack(category, entry)
        if not haystack:
            return 0.0
        score = 0.0
        for token in tokens:
            pattern = r"(?<![0-9a-z])" + re.escape(token) + r"(?![0-9a-z])"
            if re.search(pattern, haystack):
                score += 2.0
            elif (not token.isascii() or len(token) >= 4) and token in haystack:
                # Substring fallback: useful for CJK (no word boundaries) and
                # partial English matches, but never lets short tokens match
                # inside unrelated words ("no" inside "north").
                score += 1.0
        return score

    def _rank_entries(
        self, category: str, entries: Sequence[Mapping[str, Any]], tokens: Sequence[str]
    ) -> list[Mapping[str, Any]]:
        """Keyword-first ranking with recency fallback; fully deterministic."""
        items = list(entries)
        if not tokens:
            return sorted(items, key=lambda item: -_as_int(item.get("seq"), 0))
        scored = [(self._score_entry(category, item, tokens), -_as_int(item.get("seq"), 0), item) for item in items]
        matched = [triple for triple in scored if triple[0] > 0]
        unmatched = [triple for triple in scored if triple[0] <= 0]
        matched.sort(key=lambda triple: (-triple[0], triple[1]))
        unmatched.sort(key=lambda triple: triple[1])
        return [triple[2] for triple in matched + unmatched]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Keyword+recency search across every memory section.

        Returns copies of matching entries with a ``category`` and ``score``
        field, best first.  An empty query returns the most recent entries
        (across categories) without scoring.
        """
        tokens = _tokenize(query)
        count = max(1, _as_int(limit, 10))
        results: list[tuple[float, int, str, dict[str, Any]]] = []
        for category in ("places", "discoveries", "dialogues", "lessons", "open_questions"):
            for entry in self._data.get(category) or []:
                score = self._score_entry(category, entry, tokens) if tokens else 0.0
                if tokens and score <= 0:
                    continue
                item = copy.deepcopy(entry)
                item["category"] = category
                item["score"] = round(score, 3)
                results.append((score, -_as_int(entry.get("seq"), 0), category, item))
        results.sort(key=lambda triple: (-triple[0], triple[1]))
        return [triple[3] for triple in results[:count]]

    # -- prompt rendering -------------------------------------------------- #

    def render_context(self, query: str | None = None, *, max_chars: int | None = None) -> str:
        """Return the compact Markdown memory block sent to DeepSeek.

        Deterministic for a given memory document.  Priority when truncating:
        goal, then the most recent (or query-relevant) discoveries, then known
        places, then dialogues/lessons, then open questions and the archive.
        The output is always at most ``max_chars`` characters.
        """
        budget = self.cfg.max_context_chars if max_chars is None else _as_int(max_chars, 0)
        budget = max(0, budget)
        if budget <= 0:
            return ""
        tokens = _tokenize(query or "")
        lines: list[str] = []
        used = 0
        seen: set[str] = set()

        def add_line(text: str, *, force: bool = False) -> bool:
            nonlocal used
            value = str(text).strip()
            if not value:
                return False
            overhead = len(value) + (1 if lines else 0)
            if used + overhead > budget:
                if not force:
                    return False
                room = budget - used - (1 if lines else 0)
                if room <= 0:
                    return False
                value = value[:room]
                overhead = len(value) + (1 if lines else 0)
            lines.append(value)
            used += overhead
            return True

        goal = self._data["goal"]
        goal_text = str(goal.get("text") or "").strip()
        goal_line = f"Goal: {goal_text}" if goal_text else "Goal: (not set yet; use set_goal)"
        header = "AGENT MEMORY"
        if len(header) + 1 + min(len(goal_line), 40) <= budget:
            add_line(header)
        add_line(goal_line, force=True)
        reason = str(goal.get("reason") or "").strip()
        if reason:
            add_line(f"Plan: {reason}")

        # -- mandatory: most recent discoveries ---------------------------- #
        discoveries = self._data["discoveries"]
        recent = sorted(discoveries, key=lambda item: -_as_int(item.get("seq"), 0))
        mandatory_ids: set[str] = set()
        for entry in recent[:3]:
            line = _format_discovery(entry)
            key = _norm_key(line)
            if key in seen:
                continue
            seen.add(key)
            if not add_line(line, force=True):
                break
            entry_id = str(entry.get("id") or "")
            if entry_id:
                mandatory_ids.add(entry_id)

        def section_items(
            category: str,
            heading: str,
            formatter: Callable[[Mapping[str, Any]], str],
            limit: int,
            skip_ids: set[str] | None = None,
        ) -> int:
            """Add a capped, deterministically ordered section; returns items added."""
            skip = skip_ids or set()
            ranked = self._rank_entries(category, self._data[category], tokens)
            added = 0
            heading_added = False
            for entry in ranked:
                if added >= limit:
                    break
                entry_id = str(entry.get("id") or "")
                if entry_id and entry_id in skip:
                    continue
                line = formatter(entry)
                key = _norm_key(line)
                if not key or key in seen:
                    continue
                if not heading_added:
                    if not add_line(heading):
                        return added
                    heading_added = True
                seen.add(key)
                if not add_line(line):
                    break
                added += 1
            return added

        # ``retrieval_limit`` caps the total number of optional (non-mandatory)
        # entries rendered, so a large memory cannot crowd out the goal.
        retrieval_budget = max(1, _as_int(self.cfg.retrieval_limit, 12))

        def optional_section(
            category: str,
            heading: str,
            formatter: Callable[[Mapping[str, Any]], str],
            cap: int,
            skip_ids: set[str] | None = None,
        ) -> None:
            nonlocal retrieval_budget
            if retrieval_budget <= 0:
                return
            added = section_items(category, heading, formatter, min(cap, retrieval_budget), skip_ids)
            retrieval_budget -= added

        optional_section("places", "Known places:", _format_place, 8)
        optional_section(
            "discoveries", "Discoveries:", _format_discovery, 8, skip_ids=mandatory_ids
        )
        optional_section("dialogues", "Dialogues:", _format_dialogue, 4)
        optional_section("lessons", "Lessons:", _format_lesson, 4)
        optional_section("open_questions", "Open questions:", _format_question, 4)

        archive = str(self._data.get("archived_summary") or "").strip()
        if archive:
            condensed = " | ".join(
                part.strip() for part in archive.splitlines() if part.strip()
            )
            prefix = "Earlier notes: "
            remaining = budget - used - (1 if lines else 0)
            if condensed and remaining > len(prefix) + 20:
                add_line(prefix + condensed, force=True)

        return "\n".join(lines)[:budget]

    # -- stats / compression ----------------------------------------------- #

    def stats(self) -> dict:
        """Compact counters for the viewer/runtime prompt."""
        data = self._data
        return {
            "places": len(data["places"]),
            "discoveries": len(data["discoveries"]),
            "dialogues": len(data["dialogues"]),
            "lessons": len(data["lessons"]),
            "open_questions": len(data["open_questions"]),
            "goal": str(data["goal"].get("text") or ""),
            "goal_history": len(data["goal_history"]),
            "journal_entries": len(self._journal_entries),
            "journal_skipped": self.journal_skipped,
            "archived_summary_chars": len(str(data.get("archived_summary") or "")),
            "compressions": _as_int(data["counters"].get("compressions"), 0),
            "actions_since_compress": self._actions_since_compress,
        }

    def _cap(self, category: str) -> int | None:
        attr = _SECTION_CAPS.get(category)
        if attr is None:
            return None
        value = getattr(self.cfg, attr, None)
        if value is None:
            return None
        cap = _as_int(value, 0)
        # non-positive caps mean "unlimited" (never silently wipe a section)
        return cap if cap > 0 else None

    def _archive_line(self, category: str, entry: Mapping[str, Any]) -> str:
        if category == "places":
            return f"place {str(entry.get('name') or '')}".strip()
        if category == "discoveries":
            kind = str(entry.get("kind") or "other")
            return f"{kind}: {str(entry.get('text') or '')}".strip()
        if category == "dialogues":
            speaker = str(entry.get("speaker") or "NPC")
            return f"dialogue {speaker}: {str(entry.get('text') or '')}".strip()
        if category == "lessons":
            success = entry.get("success")
            flag = " worked:" if success is True else (" failed:" if success is False else " note:")
            return f"lesson{flag} {str(entry.get('text') or '')}".strip()
        return _clean(entry.get("text"), 160)

    def _append_archive(self, text: str) -> None:
        value = _clean_block(text, 1400)
        if not value:
            return
        existing = str(self._data.get("archived_summary") or "").strip()
        merged = f"{existing}\n{value}".strip() if existing else value
        if len(merged) > MAX_ARCHIVE_CHARS:
            merged = "…" + merged[-(MAX_ARCHIVE_CHARS - 1):]
        self._data["archived_summary"] = merged

    def _append_archive_lines(self, lines: Sequence[str], *, chunk_chars: int = 900) -> None:
        """Append grouped lines to the archive in bounded chunks (names not clipped)."""
        chunk: list[str] = []
        size = 0
        for line in lines:
            value = str(line).strip()
            if not value:
                continue
            if chunk and size + len(value) + 2 > chunk_chars:
                self._append_archive("; ".join(chunk))
                chunk, size = [], 0
            chunk.append(value)
            size += len(value) + 2
        if chunk:
            self._append_archive("; ".join(chunk))

    def _distill_for_summary(self) -> list[str]:
        """Remove the oldest quarter of long text sections and return their lines."""
        lines: list[str] = []
        for category in ("discoveries", "dialogues", "lessons"):
            entries = self._data[category]
            if len(entries) < 3:
                continue
            drop = max(1, len(entries) // 4)
            removed = entries[:drop]
            del entries[:drop]
            for entry in removed:
                line = self._archive_line(category, entry)
                if line:
                    lines.append(line)
        return lines

    def maybe_compress(self, *, actions_since: int = 0, force: bool = False) -> bool:
        """Fold overflow entries into ``archived_summary``; True if memory changed.

        Triggered when ``force`` is set, when any section exceeds its configured
        cap, or when enough actions have accumulated since the previous pass
        (``MemoryConfig.compress_every_actions``).  Overflow beyond a cap is
        folded oldest-first into ``archived_summary``.  Place *names* are always
        archived verbatim, even when a summarizer is used, and the goal is never
        touched.  With ``force=True`` an injected ``summarizer`` is called on
        the collected overflow (or on the oldest quarter of long text sections
        when nothing overflowed); a summarizer failure never crashes the run.
        Honors ``MemoryConfig.mirror_to_run`` by saving the compressed memory.
        """
        self._actions_since_compress += max(0, _as_int(actions_since, 0))
        threshold = max(1, _as_int(self.cfg.compress_every_actions, 1))
        overflow = any(
            self._cap(category) is not None and len(self._data[category]) > int(self._cap(category) or 0)
            for category in _SECTION_CAPS
        )
        if not (force or overflow or self._actions_since_compress >= threshold):
            return False

        changed = False
        summary_inputs: list[str] = []
        place_lines: list[str] = []
        for category in _SECTION_CAPS:
            cap = self._cap(category)
            if cap is None:
                continue
            entries = self._data[category]
            if len(entries) <= cap:
                continue
            removed = entries[: len(entries) - cap]
            del entries[: len(entries) - cap]
            changed = True
            for entry in removed:
                line = self._archive_line(category, entry)
                if not line:
                    continue
                if category == "places":
                    place_lines.append(line)
                else:
                    summary_inputs.append(line)

        if force and not changed and self.summarizer is not None:
            distilled = self._distill_for_summary()
            if distilled:
                summary_inputs.extend(distilled)
                changed = True

        if not changed:
            self._actions_since_compress = 0
            return False

        # Place names always survive verbatim (never handed to the summarizer),
        # appended in chunks so a long list cannot clip names away.
        if place_lines:
            self._append_archive_lines(place_lines)
        if summary_inputs:
            archived = ""
            if self.summarizer is not None:
                try:
                    candidate = self.summarizer(list(summary_inputs))
                except Exception:  # noqa: BLE001 - injected callback must not crash the run
                    candidate = ""
                if isinstance(candidate, str):
                    archived = _clean_block(candidate, 1200)
            if not archived:
                archived = "; ".join(summary_inputs)[:1200]
            self._append_archive(archived)

        counters = self._data["counters"]
        counters["compressions"] = _as_int(counters.get("compressions"), 0) + 1
        self._actions_since_compress = 0
        self._append_journal(
            "compress",
            f"archived {len(summary_inputs)} text entries and {len(place_lines)} place names",
        )
        self.save()
        return True

    # -- snapshot / restore ------------------------------------------------ #

    def snapshot(self) -> dict:
        """Deep-copied, JSON-serialisable memory snapshot for checkpointing."""
        return {
            "version": MEMORY_VERSION,
            "data": copy.deepcopy(self._data),
            "actions_since_compress": self._actions_since_compress,
            "seq": self._seq,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a :meth:`snapshot` (or bare memory document); never raises."""
        if not isinstance(snapshot, Mapping):
            return
        container = snapshot
        data_raw: Any = container.get("data")
        if isinstance(data_raw, Mapping):
            actions = max(0, _as_int(container.get("actions_since_compress"), 0))
            seq_hint = max(0, _as_int(container.get("seq"), 0))
        else:
            data_raw = container
            actions = 0
            seq_hint = 0
        self._data = _normalize_data(data_raw, self.cfg)
        self._seq = max(self._compute_next_seq(), seq_hint)
        self._actions_since_compress = actions

    # -- reports ----------------------------------------------------------- #

    def render_report(self) -> str:
        """Human-readable Markdown report of the whole memory document."""
        data = self._data
        stats = self.stats()
        goal = data["goal"]
        parts: list[str] = [
            "# Agent Memory Report",
            "",
            f"- Updated: {_format_ts(_as_float(data.get('updated_at'), 0.0))}",
            f"- Created: {_format_ts(_as_float(data.get('created_at'), 0.0))}",
            f"- Counts: places={stats['places']} discoveries={stats['discoveries']} "
            f"dialogues={stats['dialogues']} lessons={stats['lessons']} "
            f"open_questions={stats['open_questions']} journal={stats['journal_entries']}",
            "",
            "## Goal",
            "",
        ]
        goal_text = str(goal.get("text") or "").strip()
        parts.append(f"**{goal_text}**" if goal_text else "_(not set)_")
        reason = str(goal.get("reason") or "").strip()
        if reason:
            parts.append(f"\nPlan/reason: {reason}")
        history = data["goal_history"]
        if history:
            parts.append("\n### Previous goals\n")
            for item in history[-20:]:
                parts.append(
                    f"- {str(item.get('text') or '')} "
                    f"({_format_ts(_as_float(item.get('at'), 0.0))})"
                )

        def report_section(title: str, category: str, formatter: Callable[[Mapping[str, Any]], str]) -> None:
            entries = data[category]
            parts.extend(["", f"## {title} ({len(entries)})", ""])
            if not entries:
                parts.append("_(empty)_")
                return
            shown = entries[-MAX_REPORT_ITEMS:]
            for entry in shown:
                parts.append(formatter(entry))
            if len(entries) > len(shown):
                parts.append(f"- … {len(entries) - len(shown)} older entries archived")

        report_section("Places", "places", _format_place)
        report_section("Discoveries", "discoveries", _format_discovery)
        report_section("Dialogues", "dialogues", _format_dialogue)
        report_section("Lessons", "lessons", _format_lesson)
        report_section("Open questions", "open_questions", _format_question)

        archive = str(data.get("archived_summary") or "").strip()
        parts.extend(["", "## Archived summary", ""])
        parts.append(archive if archive else "_(empty)_")

        journal = self.recent_journal(20)
        parts.extend(["", "## Recent journal", ""])
        if not journal:
            parts.append("_(empty)_")
        for entry in journal:
            parts.append(
                f"- {_format_ts(_as_float(entry.get('at'), 0.0))} "
                f"[{entry.get('kind')}] {entry.get('text')}"
            )
        parts.append("")
        return "\n".join(parts)

    def save_report(self, path: str | Path) -> None:
        """Write the Markdown mirror atomically to ``path`` (creates parents)."""
        target = Path(path)
        _atomic_write_text(target, self.render_report())
        self.last_report_path = target

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        stats = self.stats()
        return (
            f"AgentMemory(root={str(self.root)!r}, places={stats['places']}, "
            f"discoveries={stats['discoveries']}, goal={stats['goal']!r})"
        )


__all__ = [
    "AgentMemory",
    "MEMORY_VERSION",
    "MEMORY_FILENAME",
    "JOURNAL_FILENAME",
    "REPORT_FILENAME",
]
