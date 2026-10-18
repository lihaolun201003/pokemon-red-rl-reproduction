"""Frozen cross-module protocol for the Pokémon Red multi-agent system.

This module is the single source of truth for:

* the :class:`EventType` catalogue exchanged on the :class:`EventBus`,
* the :class:`AgentMessage` envelope (message id, correlation id, task id,
  source/target agent, typed payload, priority, branch id, timestamp),
* the :class:`Task` record managed by the :class:`TaskManager`,
* the :class:`KnowledgeEntry` record stored in the shared world memory,
* the :class:`ControlLease` token issued by the :class:`ControlArbiter`,
* the :class:`BranchRecord` created by the recovery/checkpoint manager.

Design rules (see ``docs/reports/multi_agent_v1_design.md``):

1. The game is controlled by exactly one agent at a time; every message that
   implies a state change carries the ``branch_id`` it was produced on so a
   rolled-back branch can never leak irreversible actions into the new one.
2. Messages are validated structurally.  Unknown events or missing required
   payload keys are rejected with :class:`ProtocolError` instead of being
   silently swallowed.
3. The protocol is pure Python (no I/O, no emulator, no API) so that every
   module and every test can import it cheaply.
4. No game knowledge (routes, gyms, story order, species data) is allowed in
   this file — only generic multi-agent scheduling concepts.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Schema version of the persisted protocol records.
PROTOCOL_VERSION = 1

#: Tolerance for clock skew when validating ``created_at`` values.
CLOCK_SKEW_S = 300.0

#: Message ids start with this prefix so audit tooling can recognise them.
MESSAGE_ID_PREFIX = "msg_"

#: Task ids start with this prefix.
TASK_ID_PREFIX = "task_"

#: Knowledge ids start with this prefix.
KNOWLEDGE_ID_PREFIX = "kn_"

#: Branch ids start with this prefix.
BRANCH_ID_PREFIX = "branch_"

#: Lease ids start with this prefix.
LEASE_ID_PREFIX = "lease_"


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #

class AgentRole(str, Enum):
    """Every participant of the multi-agent system."""

    DIRECTOR = "director"
    NAVIGATION = "navigation"
    STORY = "story"
    POKEMON = "pokemon"
    SYSTEM = "system"

    @classmethod
    def coerce(cls, value: Any) -> "AgentRole":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        aliases = {
            "nav": cls.NAVIGATION,
            "navigation_agent": cls.NAVIGATION,
            "story_agent": cls.STORY,
            "pokemon_agent": cls.POKEMON,
            "pokémon": cls.POKEMON,
            "pokemon": cls.POKEMON,
            "director": cls.DIRECTOR,
            "game_director": cls.DIRECTOR,
            "system": cls.SYSTEM,
        }
        if text in aliases:
            return aliases[text]
        for member in cls:
            if member.value == text:
                return member
        raise ProtocolError(f"unknown agent role: {value!r}")

    @property
    def is_professional(self) -> bool:
        """True for the three specialised agents (not director/system)."""
        return self in (AgentRole.NAVIGATION, AgentRole.STORY, AgentRole.POKEMON)


class ControlOwner(str, Enum):
    """Who currently holds the single emulator control lease."""

    NONE = "none"
    NAVIGATION = "navigation"
    STORY = "story"
    POKEMON = "pokemon"
    SYSTEM = "system"

    @classmethod
    def coerce(cls, value: Any) -> "ControlOwner":
        if isinstance(value, cls):
            return value
        text = str(value if value is not None else "none").strip().lower()
        if text not in cls._value2member_map_:
            raise ProtocolError(f"unknown control owner: {value!r}")
        return cls(text)

    @classmethod
    def from_role(cls, role: AgentRole | str) -> "ControlOwner":
        role = AgentRole.coerce(role)
        return {
            AgentRole.NAVIGATION: cls.NAVIGATION,
            AgentRole.STORY: cls.STORY,
            AgentRole.POKEMON: cls.POKEMON,
            AgentRole.SYSTEM: cls.SYSTEM,
            AgentRole.DIRECTOR: cls.SYSTEM,
        }[role]


class TaskStatus(str, Enum):
    """Lifecycle of a shared task (see task_manager.TaskManager)."""

    PENDING = "pending"        # created, waiting for dependencies/assignment
    READY = "ready"            # dependencies satisfied, schedulable
    RUNNING = "running"        # currently owned/executing
    PAUSED = "paused"          # temporarily suspended (event/battle/dialog)
    BLOCKED = "blocked"        # cannot progress without an external change
    VERIFYING = "verifying"    # claimed complete, awaiting verification evidence
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def coerce(cls, value: Any) -> "TaskStatus":
        if isinstance(value, cls):
            return value
        text = str(value if value is not None else "pending").strip().lower()
        if text not in cls._value2member_map_:
            raise ProtocolError(f"unknown task status: {value!r}")
        return cls(text)

    @property
    def terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def active(self) -> bool:
        return self in (TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.VERIFYING)

    @property
    def schedulable(self) -> bool:
        return self in (TaskStatus.READY, TaskStatus.PAUSED)


class TaskType(str, Enum):
    """Coarse task categories used by the deterministic scheduler."""

    EXPLORE = "explore"
    NAVIGATE = "navigate"
    INTERACT = "interact"
    TALK = "talk"
    INVESTIGATE = "investigate"
    BATTLE = "battle"
    CATCH = "catch"
    TRAIN = "train"
    RECOVER = "recover"
    VERIFY = "verify"
    DIRECTOR_REVIEW = "director_review"
    CUSTOM = "custom"

    @classmethod
    def coerce(cls, value: Any) -> "TaskType":
        if isinstance(value, cls):
            return value
        text = str(value or "custom").strip().lower()
        for member in cls:
            if member.value == text:
                return member
        return cls.CUSTOM


class VerificationStatus(str, Enum):
    """Epistemic status of a knowledge entry."""

    VERIFIED = "verified"        # directly observed in this branch with evidence
    INFERRED = "inferred"        # reasonable inference from observed facts
    HYPOTHESIS = "hypothesis"    # unconfirmed guess / plan
    REFUTED = "refuted"          # observed contradictory evidence
    HISTORICAL = "historical"    # valid only in an older, rolled-back branch

    @classmethod
    def coerce(cls, value: Any) -> "VerificationStatus":
        if isinstance(value, cls):
            return value
        text = str(value if value is not None else "hypothesis").strip().lower()
        if text not in cls._value2member_map_:
            raise ProtocolError(f"unknown verification status: {value!r}")
        return cls(text)

    @property
    def trusted(self) -> bool:
        """Only verified facts may be treated as certain by prompts."""
        return self is VerificationStatus.VERIFIED


class Priority(int, Enum):
    """Task / message priority.  Higher value wins."""

    BACKGROUND = 10
    LOW = 25
    NORMAL = 50
    HIGH = 75
    CRITICAL = 100

    @classmethod
    def coerce(cls, value: Any) -> "Priority":
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = int(value)
            if number in cls._value2member_map_:
                return cls(number)
            # snap to the nearest defined level (never out of range)
            return min(cls, key=lambda member: abs(int(member) - number))
        text = str(value if value is not None else "normal").strip().lower()
        by_name = {
            "background": cls.BACKGROUND,
            "low": cls.LOW,
            "normal": cls.NORMAL,
            "high": cls.HIGH,
            "critical": cls.CRITICAL,
        }
        if text in by_name:
            return by_name[text]
        try:
            return cls.coerce(float(text))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"unknown priority: {value!r}") from exc


class StagnationKind(str, Enum):
    """Progress-monitor signals (thresholds live in the config, not here)."""

    LOCAL_STAGNATION = "local_stagnation"
    GLOBAL_STAGNATION = "global_stagnation"
    REPEATED_FAILURE = "repeated_failure"
    NO_PROGRESS = "no_progress"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    RECOVERY_LIMIT = "recovery_limit"
    REPEATED_MESSAGE = "repeated_message"


class RecoveryLevel(str, Enum):
    """The three mandated levels of self-correction."""

    L1_LOCAL = "l1_local"
    L2_REVIEW = "l2_review"
    L3_ROLLBACK = "l3_rollback"
    NONE = "none"

    @classmethod
    def coerce(cls, value: Any) -> "RecoveryLevel":
        if isinstance(value, cls):
            return value
        text = str(value if value is not None else "none").strip().lower()
        if text not in cls._value2member_map_:
            raise ProtocolError(f"unknown recovery level: {value!r}")
        return cls(text)


class BranchStatus(str, Enum):
    """Lifecycle of a checkpoint branch."""

    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    ABANDONED = "abandoned"
    CLOSED = "closed"

    @classmethod
    def coerce(cls, value: Any) -> "BranchStatus":
        if isinstance(value, cls):
            return value
        text = str(value if value is not None else "active").strip().lower()
        if text not in cls._value2member_map_:
            raise ProtocolError(f"unknown branch status: {value!r}")
        return cls(text)


class EventType(str, Enum):
    """The frozen event catalogue exchanged over the event bus."""

    # -- task lifecycle ---------------------------------------------------- #
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_BLOCKED = "task_blocked"
    TASK_CANCELLED = "task_cancelled"
    TASK_VERIFIED = "task_verified"
    TASK_REOPENED = "task_reopened"

    # -- navigation -------------------------------------------------------- #
    NAVIGATION_REQUEST = "navigation_request"
    NAVIGATION_COMPLETED = "navigation_completed"
    NAVIGATION_FAILED = "navigation_failed"

    # -- battle / pokemon -------------------------------------------------- #
    BATTLE_STARTED = "battle_started"
    BATTLE_ENDED = "battle_ended"
    POKEMON_NEEDS_HELP = "pokemon_needs_help"
    POKEMON_TEAM_CHANGED = "pokemon_team_changed"

    # -- dialog / story ---------------------------------------------------- #
    DIALOG_STARTED = "dialog_started"
    DIALOG_ENDED = "dialog_ended"
    STORY_DISCOVERY = "story_discovery"
    STORY_HYPOTHESIS = "story_hypothesis"

    # -- world / map ------------------------------------------------------- #
    MAP_TRANSITION = "map_transition"
    NEW_AREA_DISCOVERED = "new_area_discovered"

    # -- control arbitration ----------------------------------------------- #
    CONTROL_REQUESTED = "control_requested"
    CONTROL_GRANTED = "control_granted"
    CONTROL_RELEASED = "control_released"
    CONTROL_DENIED = "control_denied"
    CONTROL_PREEMPTED = "control_preempted"

    # -- progress / recovery ----------------------------------------------- #
    PROGRESS_DETECTED = "progress_detected"
    STAGNATION_DETECTED = "stagnation_detected"
    CHECKPOINT_CREATED = "checkpoint_created"
    ROLLBACK_REQUESTED = "rollback_requested"
    ROLLBACK_COMPLETED = "rollback_completed"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_FINISHED = "recovery_finished"

    # -- agent lifecycle --------------------------------------------------- #
    AGENT_ERROR = "agent_error"
    AGENT_PAUSED = "agent_paused"
    AGENT_RESUMED = "agent_resumed"
    AGENT_MESSAGE = "agent_message"

    # -- system ------------------------------------------------------------ #
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_HEARTBEAT = "system_heartbeat"
    BUDGET_WARNING = "budget_warning"
    LIMIT_REACHED = "limit_reached"

    @classmethod
    def coerce(cls, value: Any) -> "EventType":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        for member in cls:
            if member.value == text:
                return member
        raise ProtocolError(f"unknown event type: {value!r}")

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


#: Events that must never be re-executed blindly after a crash/restart because
#: they (may) correspond to an irreversible game action.  Their consumers must
#: first re-observe the game and confirm the outcome before acting.
IRREVERSIBLE_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.ROLLBACK_COMPLETED,
        EventType.CHECKPOINT_CREATED,
        EventType.BATTLE_STARTED,
        EventType.DIALOG_STARTED,
        EventType.MAP_TRANSITION,
    }
)

#: Events that are pure bookkeeping and safe to replay idempotently.
BOOKKEEPING_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.SYSTEM_HEARTBEAT,
        EventType.PROGRESS_DETECTED,
        EventType.STAGNATION_DETECTED,
        EventType.BUDGET_WARNING,
        EventType.AGENT_PAUSED,
        EventType.AGENT_RESUMED,
    }
)

#: Required payload keys per event type (extra keys are allowed and preserved).
EVENT_PAYLOAD_SPEC: dict[EventType, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # (required keys, optional keys)
    EventType.TASK_CREATED: (("task_id",), ("description", "task_type", "owner_agent")),
    EventType.TASK_ASSIGNED: (("task_id", "owner_agent"), ()),
    EventType.TASK_STARTED: (("task_id",), ("owner_agent",)),
    EventType.TASK_PROGRESS: (("task_id",), ("progress", "note")),
    EventType.TASK_COMPLETED: (("task_id",), ("result", "evidence")),
    EventType.TASK_FAILED: (("task_id", "reason"), ()),
    EventType.TASK_BLOCKED: (("task_id", "reason"), ()),
    EventType.TASK_CANCELLED: (("task_id",), ("reason",)),
    EventType.TASK_VERIFIED: (("task_id",), ("evidence",)),
    EventType.TASK_REOPENED: (("task_id",), ("reason",)),
    EventType.NAVIGATION_REQUEST: ((), ("target", "max_steps", "reason", "task_id")),
    EventType.NAVIGATION_COMPLETED: ((), ("report", "task_id", "steps", "moves")),
    EventType.NAVIGATION_FAILED: (("reason",), ("task_id", "report")),
    EventType.BATTLE_STARTED: ((), ("frame", "context")),
    EventType.BATTLE_ENDED: ((), ("outcome", "turns", "task_id")),
    EventType.POKEMON_NEEDS_HELP: ((), ("reason", "task_id")),
    EventType.POKEMON_TEAM_CHANGED: ((), ("summary",)),
    EventType.DIALOG_STARTED: ((), ("frame", "speaker", "context")),
    EventType.DIALOG_ENDED: ((), ("summary", "npcs", "task_id")),
    EventType.STORY_DISCOVERY: (("content",), ("category", "confidence", "evidence_refs")),
    EventType.STORY_HYPOTHESIS: (("content",), ("confidence", "evidence_refs")),
    EventType.MAP_TRANSITION: ((), ("area_id", "frame")),
    EventType.NEW_AREA_DISCOVERED: (("area_id",), ("summary", "kind")),
    EventType.CONTROL_REQUESTED: (("owner",), ("reason", "task_id", "lease_id")),
    EventType.CONTROL_GRANTED: (("owner", "lease_id"), ("reason", "task_id")),
    EventType.CONTROL_RELEASED: (("owner", "lease_id"), ("reason",)),
    EventType.CONTROL_DENIED: (("owner", "reason"), ("task_id",)),
    EventType.CONTROL_PREEMPTED: (("previous_owner", "owner", "reason"), ()),
    EventType.PROGRESS_DETECTED: (("kind",), ("detail", "delta")),
    EventType.STAGNATION_DETECTED: (("kind",), ("detail", "level")),
    EventType.CHECKPOINT_CREATED: (("checkpoint_id", "branch_id"), ("reason", "frame")),
    EventType.ROLLBACK_REQUESTED: (("reason",), ("checkpoint_id", "branch_id")),
    EventType.ROLLBACK_COMPLETED: (("branch_id", "previous_branch_id"), ("checkpoint_id", "reason")),
    EventType.RECOVERY_STARTED: (("level",), ("reason", "task_id")),
    EventType.RECOVERY_FINISHED: (("level",), ("result", "task_id")),
    EventType.AGENT_ERROR: (("agent", "error"), ("context",)),
    EventType.AGENT_PAUSED: (("agent",), ("reason",)),
    EventType.AGENT_RESUMED: (("agent",), ()),
    EventType.AGENT_MESSAGE: ((), ("text", "data")),
    EventType.SYSTEM_STARTED: ((), ("run_id", "model", "branch_id")),
    EventType.SYSTEM_STOPPED: ((), ("reason", "status")),
    EventType.SYSTEM_HEARTBEAT: ((), ("frame", "actions", "cost_usd")),
    EventType.BUDGET_WARNING: ((), ("fraction", "remaining_usd")),
    EventType.LIMIT_REACHED: (("limit",), ("detail",)),
}

#: Message routing metadata: default target agent(s) for every event type.
EVENT_ROUTING: dict[EventType, tuple[AgentRole, ...]] = {
    EventType.TASK_CREATED: (AgentRole.DIRECTOR,),
    EventType.TASK_ASSIGNED: (AgentRole.DIRECTOR,),
    EventType.TASK_STARTED: (AgentRole.DIRECTOR,),
    EventType.TASK_PROGRESS: (AgentRole.DIRECTOR,),
    EventType.TASK_COMPLETED: (AgentRole.DIRECTOR,),
    EventType.TASK_FAILED: (AgentRole.DIRECTOR,),
    EventType.TASK_BLOCKED: (AgentRole.DIRECTOR,),
    EventType.TASK_CANCELLED: (AgentRole.DIRECTOR,),
    EventType.TASK_VERIFIED: (AgentRole.DIRECTOR,),
    EventType.TASK_REOPENED: (AgentRole.DIRECTOR,),
    EventType.NAVIGATION_REQUEST: (AgentRole.NAVIGATION,),
    EventType.NAVIGATION_COMPLETED: (AgentRole.DIRECTOR,),
    EventType.NAVIGATION_FAILED: (AgentRole.DIRECTOR,),
    EventType.BATTLE_STARTED: (AgentRole.DIRECTOR,),
    EventType.BATTLE_ENDED: (AgentRole.DIRECTOR,),
    EventType.POKEMON_NEEDS_HELP: (AgentRole.DIRECTOR,),
    EventType.POKEMON_TEAM_CHANGED: (AgentRole.DIRECTOR,),
    EventType.DIALOG_STARTED: (AgentRole.DIRECTOR,),
    EventType.DIALOG_ENDED: (AgentRole.DIRECTOR,),
    EventType.STORY_DISCOVERY: (AgentRole.DIRECTOR,),
    EventType.STORY_HYPOTHESIS: (AgentRole.DIRECTOR,),
    EventType.MAP_TRANSITION: (AgentRole.DIRECTOR,),
    EventType.NEW_AREA_DISCOVERED: (AgentRole.DIRECTOR,),
    EventType.CONTROL_REQUESTED: (AgentRole.DIRECTOR,),
    EventType.CONTROL_GRANTED: (AgentRole.DIRECTOR,),
    EventType.CONTROL_RELEASED: (AgentRole.DIRECTOR,),
    EventType.CONTROL_DENIED: (AgentRole.DIRECTOR,),
    EventType.CONTROL_PREEMPTED: (AgentRole.DIRECTOR,),
    EventType.PROGRESS_DETECTED: (AgentRole.DIRECTOR,),
    EventType.STAGNATION_DETECTED: (AgentRole.DIRECTOR,),
    EventType.CHECKPOINT_CREATED: (AgentRole.DIRECTOR,),
    EventType.ROLLBACK_REQUESTED: (AgentRole.DIRECTOR,),
    EventType.ROLLBACK_COMPLETED: (AgentRole.DIRECTOR,),
    EventType.RECOVERY_STARTED: (AgentRole.DIRECTOR,),
    EventType.RECOVERY_FINISHED: (AgentRole.DIRECTOR,),
    EventType.AGENT_ERROR: (AgentRole.DIRECTOR,),
    EventType.AGENT_PAUSED: (AgentRole.DIRECTOR,),
    EventType.AGENT_RESUMED: (AgentRole.DIRECTOR,),
    EventType.AGENT_MESSAGE: (AgentRole.DIRECTOR,),
    EventType.SYSTEM_STARTED: (AgentRole.DIRECTOR,),
    EventType.SYSTEM_STOPPED: (AgentRole.DIRECTOR,),
    EventType.SYSTEM_HEARTBEAT: (AgentRole.DIRECTOR,),
    EventType.BUDGET_WARNING: (AgentRole.DIRECTOR,),
    EventType.LIMIT_REACHED: (AgentRole.DIRECTOR,),
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class MultiAgentError(RuntimeError):
    """Base class for multi-agent subsystem failures."""

    def __init__(self, message: str = "", *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = str(message)
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "message": self.message,
            "details": self.details,
        }


class ProtocolError(MultiAgentError):
    """A message/task/knowledge record violates the frozen protocol."""


class ControlDeniedError(MultiAgentError):
    """An agent tried to operate the emulator without holding the control lease."""

    def __init__(
        self,
        message: str = "control denied",
        *,
        agent: str = "",
        owner: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(details or {})
        payload.setdefault("agent", agent)
        payload.setdefault("owner", owner)
        super().__init__(message, details=payload)
        self.agent = agent
        self.owner = owner


class DuplicateMessageError(MultiAgentError):
    """A message with an already-seen id was published again."""


class TaskError(MultiAgentError):
    """Invalid task lifecycle transition or duplicate/inconsistent task."""


class CycleDetectedError(TaskError):
    """Adding this dependency would create a task dependency cycle."""


class KnowledgeError(MultiAgentError):
    """Invalid knowledge record or persistence failure."""


class RecoveryLimitError(MultiAgentError):
    """A hard recovery/rollback/runtime budget has been exhausted."""


class StagnationLimitError(MultiAgentError):
    """The configured stagnation/repetition thresholds have been exceeded."""


class BudgetLimitError(MultiAgentError):
    """Per-agent or global API budget is exhausted."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def now_ts() -> float:
    """Wall-clock timestamp helper (kept injectable in tests via ``clock=``)."""
    return time.time()


def _new_uuid() -> str:
    return uuid.uuid4().hex


def new_message_id() -> str:
    return f"{MESSAGE_ID_PREFIX}{_new_uuid()}"


def new_task_id() -> str:
    return f"{TASK_ID_PREFIX}{_new_uuid()[:16]}"


def new_knowledge_id() -> str:
    return f"{KNOWLEDGE_ID_PREFIX}{_new_uuid()[:16]}"


def new_branch_id() -> str:
    return f"{BRANCH_ID_PREFIX}{_new_uuid()[:16]}"


def new_lease_id() -> str:
    return f"{LEASE_ID_PREFIX}{_new_uuid()[:16]}"


def stable_digest(value: Any) -> str:
    """Deterministic short digest of a JSON-serialisable value."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dedupe_key(kind: str, content: str, *, extra: Mapping[str, Any] | None = None) -> str:
    """Stable de-duplication key for tasks / messages / knowledge entries."""
    payload = {"kind": str(kind), "content": str(content).strip()}
    if extra:
        payload["extra"] = {str(k): extra[k] for k in sorted(extra, key=str)}
    return stable_digest(payload)


def _as_str(value: Any, *, field_name: str, allow_empty: bool = False) -> str:
    if isinstance(value, (bytes, bytearray)):
        raise ProtocolError(f"{field_name} must be a string, got bytes")
    text = str(value if value is not None else "").strip()
    if not text and not allow_empty:
        raise ProtocolError(f"{field_name} must not be empty")
    return text


def _as_id(value: Any, *, field_name: str, prefix: str = "", allow_empty: bool = False) -> str:
    text = _as_str(value, field_name=field_name, allow_empty=allow_empty)
    if prefix and text and not text.startswith(prefix):
        # Accept foreign ids (e.g. checkpoint ids) when no prefix is mandated.
        raise ProtocolError(f"{field_name} must start with {prefix!r}: {text!r}")
    return text


def _as_ts(value: Any, *, field_name: str = "created_at") -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{field_name} must be a number")
    try:
        ts = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field_name} must be a number, got {value!r}") from exc
    if ts < 0:
        raise ProtocolError(f"{field_name} must be >= 0")
    return ts


def _as_priority(value: Any) -> Priority:
    return Priority.coerce(value)


def _as_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field_name} must be a JSON object")
    return dict(value)


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{field_name} must be a list of strings")
    items: list[str] = []
    for item in value:
        items.append(_as_str(item, field_name=field_name))
    return items


# --------------------------------------------------------------------------- #
# AgentMessage
# --------------------------------------------------------------------------- #

@dataclass
class AgentMessage:
    """One validated envelope on the event bus.

    The envelope is immutable in spirit: producers create a message once and the
    bus persists it.  Helpers return new copies instead of mutating in place.
    """

    message_id: str
    event_type: EventType
    source_agent: AgentRole
    target_agent: AgentRole
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    parent_task_id: str = ""
    priority: Priority = Priority.NORMAL
    created_at: float = 0.0
    branch_id: str = ""
    #: monotonic per-bus sequence number assigned on publish (0 = unassigned)
    sequence: int = 0

    # -- construction ---------------------------------------------------- #
    @classmethod
    def create(
        cls,
        event_type: EventType | str,
        *,
        source_agent: AgentRole | str,
        target_agent: AgentRole | str | None = None,
        payload: Mapping[str, Any] | None = None,
        priority: Priority | int | str = Priority.NORMAL,
        correlation_id: str = "",
        parent_task_id: str = "",
        branch_id: str = "",
        message_id: str = "",
        created_at: float | None = None,
    ) -> "AgentMessage":
        event = EventType.coerce(event_type)
        source = AgentRole.coerce(source_agent)
        if target_agent is None:
            targets = EVENT_ROUTING.get(event) or (AgentRole.DIRECTOR,)
            target = targets[0]
        else:
            target = AgentRole.coerce(target_agent)
        payload_dict = dict(payload or {})
        required, _optional = EVENT_PAYLOAD_SPEC.get(event, ((), ()))
        missing = [key for key in required if key not in payload_dict]
        if missing:
            raise ProtocolError(
                f"event {event.value} is missing required payload keys: {missing}"
            )
        message = cls(
            message_id=message_id or new_message_id(),
            event_type=event,
            source_agent=source,
            target_agent=target,
            payload=payload_dict,
            correlation_id=str(correlation_id or ""),
            parent_task_id=str(parent_task_id or ""),
            priority=_as_priority(priority),
            created_at=float(created_at if created_at is not None else now_ts()),
            branch_id=str(branch_id or ""),
            sequence=0,
        )
        message.validate()
        return message

    # -- validation ------------------------------------------------------ #
    def validate(self) -> None:
        if not isinstance(self.message_id, str) or not self.message_id:
            raise ProtocolError("message_id must be a non-empty string")
        if not self.message_id.startswith(MESSAGE_ID_PREFIX):
            raise ProtocolError(f"message_id must start with {MESSAGE_ID_PREFIX!r}")
        if not isinstance(self.event_type, EventType):
            raise ProtocolError("event_type must be an EventType")
        if not isinstance(self.source_agent, AgentRole):
            raise ProtocolError("source_agent must be an AgentRole")
        if not isinstance(self.target_agent, AgentRole):
            raise ProtocolError("target_agent must be an AgentRole")
        if self.source_agent is self.target_agent and self.target_agent is not AgentRole.SYSTEM:
            # Self-addressed messages are allowed for the system/director only.
            if self.target_agent is not AgentRole.DIRECTOR:
                raise ProtocolError(
                    f"source and target agent are both {self.source_agent.value}"
                )
        if not isinstance(self.payload, dict):
            raise ProtocolError("payload must be a JSON object")
        required, _optional = EVENT_PAYLOAD_SPEC.get(self.event_type, ((), ()))
        missing = [key for key in required if key not in self.payload]
        if missing:
            raise ProtocolError(
                f"event {self.event_type.value} is missing required payload keys: {missing}"
            )
        if not isinstance(self.priority, Priority):
            raise ProtocolError("priority must be a Priority")
        _as_ts(self.created_at)
        if self.correlation_id:
            _as_str(self.correlation_id, field_name="correlation_id")
        if self.parent_task_id:
            _as_str(self.parent_task_id, field_name="parent_task_id")
        if self.branch_id:
            _as_str(self.branch_id, field_name="branch_id")
        if int(self.sequence) < 0:
            raise ProtocolError("sequence must be >= 0")

    # -- serialisation --------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "message_id": self.message_id,
            "event_type": self.event_type.value,
            "source_agent": self.source_agent.value,
            "target_agent": self.target_agent.value,
            "payload": dict(self.payload),
            "correlation_id": self.correlation_id,
            "parent_task_id": self.parent_task_id,
            "priority": int(self.priority),
            "created_at": float(self.created_at),
            "branch_id": self.branch_id,
            "sequence": int(self.sequence),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentMessage":
        if not isinstance(data, Mapping):
            raise ProtocolError("message record must be a JSON object")
        message = cls(
            message_id=_as_id(data.get("message_id"), field_name="message_id", prefix=MESSAGE_ID_PREFIX),
            event_type=EventType.coerce(data.get("event_type")),
            source_agent=AgentRole.coerce(data.get("source_agent")),
            target_agent=AgentRole.coerce(data.get("target_agent")),
            payload=_as_dict(data.get("payload"), field_name="payload"),
            correlation_id=str(data.get("correlation_id") or ""),
            parent_task_id=str(data.get("parent_task_id") or ""),
            priority=_as_priority(data.get("priority", Priority.NORMAL)),
            created_at=_as_ts(data.get("created_at", 0.0)),
            branch_id=str(data.get("branch_id") or ""),
            sequence=int(data.get("sequence") or 0),
        )
        message.validate()
        return message

    # -- idempotency helpers --------------------------------------------- #
    def dedupe_key(self) -> str:
        """Content-based key used to drop duplicate logical events."""
        return dedupe_key(
            self.event_type.value,
            json.dumps(self.payload, sort_keys=True, ensure_ascii=False, default=str),
            extra={
                "source": self.source_agent.value,
                "target": self.target_agent.value,
                "task": self.parent_task_id,
                "correlation": self.correlation_id,
            },
        )

    @property
    def task_id(self) -> str:
        """Best-effort task id referenced by this message."""
        value = self.payload.get("task_id")
        if isinstance(value, str) and value:
            return value
        return self.parent_task_id

    def describe(self) -> str:
        return f"{self.event_type.value} {self.source_agent.value}->{self.target_agent.value}"


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #

@dataclass
class Task:
    """A unit of work owned by exactly one professional agent.

    Tasks are created by agents (or the director) and scheduled deterministically.
    ``dedupe_key`` is derived from (owner, task_type, description) so the same
    logical goal is never queued twice while it is active.
    """

    task_id: str
    owner_agent: AgentRole
    requested_by: AgentRole
    task_type: TaskType
    description: str
    parent_task_id: str = ""
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.NORMAL
    evidence: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 0
    progress: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    branch_id: str = ""
    #: structured inputs for the owning agent (target cell, area, npc, …)
    payload: dict[str, Any] = field(default_factory=dict)
    #: why the task is blocked/failed (human readable, auditable)
    reason: str = ""
    #: number of times the task was paused by an interrupting event
    interruptions: int = 0
    #: monotonic version used for optimistic persistence checks
    revision: int = 1

    @classmethod
    def create(
        cls,
        *,
        owner_agent: AgentRole | str,
        requested_by: AgentRole | str,
        task_type: TaskType | str,
        description: str,
        parent_task_id: str = "",
        dependencies: Iterable[str] | None = None,
        priority: Priority | int | str = Priority.NORMAL,
        branch_id: str = "",
        payload: Mapping[str, Any] | None = None,
        task_id: str = "",
        created_at: float | None = None,
    ) -> "Task":
        ts = float(created_at if created_at is not None else now_ts())
        task = cls(
            task_id=task_id or new_task_id(),
            owner_agent=AgentRole.coerce(owner_agent),
            requested_by=AgentRole.coerce(requested_by),
            task_type=TaskType.coerce(task_type),
            description=_as_str(description, field_name="description"),
            parent_task_id=str(parent_task_id or ""),
            dependencies=sorted({str(dep) for dep in (dependencies or []) if str(dep).strip()}),
            status=TaskStatus.PENDING,
            priority=_as_priority(priority),
            branch_id=str(branch_id or ""),
            payload=dict(payload or {}),
            created_at=ts,
            updated_at=ts,
        )
        task.validate()
        return task

    # -- derived --------------------------------------------------------- #
    @property
    def dedupe_key(self) -> str:
        return dedupe_key(
            self.task_type.value,
            self.description,
            extra={"owner": self.owner_agent.value, "parent": self.parent_task_id},
        )

    @property
    def terminal(self) -> bool:
        return self.status.terminal

    # -- validation ------------------------------------------------------ #
    def validate(self) -> None:
        _as_id(self.task_id, field_name="task_id", prefix=TASK_ID_PREFIX)
        if not isinstance(self.owner_agent, AgentRole):
            raise ProtocolError("owner_agent must be an AgentRole")
        if self.owner_agent in (AgentRole.DIRECTOR,):
            raise ProtocolError("a task cannot be owned by the director")
        if not isinstance(self.requested_by, AgentRole):
            raise ProtocolError("requested_by must be an AgentRole")
        if not isinstance(self.task_type, TaskType):
            raise ProtocolError("task_type must be a TaskType")
        _as_str(self.description, field_name="description")
        if not isinstance(self.status, TaskStatus):
            raise ProtocolError("status must be a TaskStatus")
        if not isinstance(self.priority, Priority):
            raise ProtocolError("priority must be a Priority")
        _as_ts(self.created_at)
        _as_ts(self.updated_at, field_name="updated_at")
        if self.updated_at < self.created_at:
            raise ProtocolError("updated_at must be >= created_at")
        if self.parent_task_id:
            _as_str(self.parent_task_id, field_name="parent_task_id")
            if self.parent_task_id == self.task_id:
                raise ProtocolError("a task cannot be its own parent")
        if self.task_id in self.dependencies:
            raise ProtocolError("a task cannot depend on itself")
        if not 0.0 <= float(self.progress) <= 1.0:
            raise ProtocolError("progress must be in [0, 1]")
        if int(self.attempts) < 0 or int(self.interruptions) < 0:
            raise ProtocolError("attempts/interruptions must be >= 0")
        if not isinstance(self.payload, dict):
            raise ProtocolError("payload must be a JSON object")

    # -- transitions ----------------------------------------------------- #
    def touch(self, *, at: float | None = None) -> None:
        self.updated_at = float(at if at is not None else now_ts())
        self.revision += 1

    def transition(
        self,
        status: TaskStatus | str,
        *,
        reason: str = "",
        at: float | None = None,
    ) -> "Task":
        """Validate and apply a status transition in place."""
        target = TaskStatus.coerce(status)
        if not is_valid_transition(self.status, target):
            raise TaskError(f"invalid task transition {self.status.value} -> {target.value}")
        self.status = target
        if reason:
            self.reason = str(reason)
        self.touch(at=at)
        return self

    # -- serialisation --------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "task_id": self.task_id,
            "owner_agent": self.owner_agent.value,
            "requested_by": self.requested_by.value,
            "task_type": self.task_type.value,
            "description": self.description,
            "parent_task_id": self.parent_task_id,
            "dependencies": list(self.dependencies),
            "status": self.status.value,
            "priority": int(self.priority),
            "evidence": [dict(item) for item in self.evidence],
            "attempts": int(self.attempts),
            "progress": float(self.progress),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "branch_id": self.branch_id,
            "payload": dict(self.payload),
            "reason": self.reason,
            "interruptions": int(self.interruptions),
            "revision": int(self.revision),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        if not isinstance(data, Mapping):
            raise ProtocolError("task record must be a JSON object")
        evidence_raw = data.get("evidence") or []
        if isinstance(evidence_raw, (str, bytes)) or not isinstance(evidence_raw, Sequence):
            raise ProtocolError("evidence must be a list")
        evidence = [_as_dict(item, field_name="evidence[i]") for item in evidence_raw]
        task = cls(
            task_id=_as_id(data.get("task_id"), field_name="task_id", prefix=TASK_ID_PREFIX),
            owner_agent=AgentRole.coerce(data.get("owner_agent")),
            requested_by=AgentRole.coerce(data.get("requested_by")),
            task_type=TaskType.coerce(data.get("task_type")),
            description=_as_str(data.get("description"), field_name="description"),
            parent_task_id=str(data.get("parent_task_id") or ""),
            dependencies=_as_str_list(data.get("dependencies"), field_name="dependencies"),
            status=TaskStatus.coerce(data.get("status", TaskStatus.PENDING)),
            priority=_as_priority(data.get("priority", Priority.NORMAL)),
            evidence=evidence,
            attempts=int(data.get("attempts") or 0),
            progress=float(data.get("progress") or 0.0),
            created_at=_as_ts(data.get("created_at", 0.0)),
            updated_at=_as_ts(data.get("updated_at", data.get("created_at", 0.0)), field_name="updated_at"),
            branch_id=str(data.get("branch_id") or ""),
            payload=_as_dict(data.get("payload"), field_name="payload"),
            reason=str(data.get("reason") or ""),
            interruptions=int(data.get("interruptions") or 0),
            revision=int(data.get("revision") or 1),
        )
        task.validate()
        return task


#: Allowed task status transitions (strict to keep the audit trail honest).
TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset(
        {TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.READY: frozenset(
        {TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.PAUSED,
            TaskStatus.BLOCKED,
            TaskStatus.VERIFYING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.READY,
        }
    ),
    TaskStatus.PAUSED: frozenset(
        {TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.BLOCKED: frozenset(
        {TaskStatus.READY, TaskStatus.PENDING, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.VERIFYING: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.RUNNING, TaskStatus.PAUSED}
    ),
    # A completed task may only be re-queued explicitly (director decision) and
    # never silently revert to RUNNING.
    TaskStatus.COMPLETED: frozenset({TaskStatus.READY}),
    TaskStatus.FAILED: frozenset({TaskStatus.PENDING, TaskStatus.READY, TaskStatus.CANCELLED}),
    TaskStatus.CANCELLED: frozenset(),
}


def is_valid_transition(current: TaskStatus | str, target: TaskStatus | str) -> bool:
    """Return True when ``current -> target`` is an allowed task transition."""
    cur = TaskStatus.coerce(current)
    dst = TaskStatus.coerce(target)
    if cur == dst:
        return False
    allowed = TASK_TRANSITIONS.get(cur, frozenset())
    return dst in allowed


# --------------------------------------------------------------------------- #
# KnowledgeEntry
# --------------------------------------------------------------------------- #

#: Knowledge categories stored in the shared world memory.
KNOWLEDGE_CATEGORIES: tuple[str, ...] = (
    "area",            # discovered area / location
    "connection",      # how two areas connect
    "npc",             # NPC (visible) with role/notes
    "interactive",     # interactive object (sign, item ball, door)
    "story_clue",      # dialogue-derived story clue
    "quest",           # open objective inferred from the game
    "battle_lesson",   # battle outcome / tactic observation
    "pokemon",         # own team / observed species facts
    "route",           # verified traversable route between cells/areas
    "hazard",          # danger/blocked/failed experience
    "system",          # operational note (control handoff, bugs)
    "other",
)


@dataclass
class KnowledgeEntry:
    """One auditable fact/inference in the shared world knowledge base."""

    knowledge_id: str
    category: str
    content: str
    source_agent: AgentRole
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5
    verification_status: VerificationStatus = VerificationStatus.HYPOTHESIS
    branch_id: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    #: set when a newer entry supersedes/refutes this one
    superseded_by: str = ""
    #: optional structured location keys (area id / cell / npc label)
    subject: str = ""
    tags: list[str] = field(default_factory=list)
    revision: int = 1

    @classmethod
    def create(
        cls,
        *,
        category: str,
        content: str,
        source_agent: AgentRole | str,
        confidence: float = 0.5,
        verification_status: VerificationStatus | str = VerificationStatus.HYPOTHESIS,
        branch_id: str = "",
        evidence_refs: Iterable[str] | None = None,
        subject: str = "",
        tags: Iterable[str] | None = None,
        knowledge_id: str = "",
        created_at: float | None = None,
    ) -> "KnowledgeEntry":
        ts = float(created_at if created_at is not None else now_ts())
        entry = cls(
            knowledge_id=knowledge_id or new_knowledge_id(),
            category=str(category or "other").strip().lower(),
            content=_as_str(content, field_name="content"),
            source_agent=AgentRole.coerce(source_agent),
            evidence_refs=[str(ref) for ref in (evidence_refs or []) if str(ref).strip()],
            confidence=float(max(0.0, min(1.0, confidence))),
            verification_status=VerificationStatus.coerce(verification_status),
            branch_id=str(branch_id or ""),
            created_at=ts,
            updated_at=ts,
            subject=str(subject or ""),
            tags=sorted({str(tag) for tag in (tags or []) if str(tag).strip()}),
        )
        entry.validate()
        return entry

    @property
    def dedupe_key(self) -> str:
        return dedupe_key(self.category, self.content, extra={"subject": self.subject})

    def validate(self) -> None:
        _as_id(self.knowledge_id, field_name="knowledge_id", prefix=KNOWLEDGE_ID_PREFIX)
        if self.category not in KNOWLEDGE_CATEGORIES:
            raise KnowledgeError(f"unknown knowledge category: {self.category!r}")
        _as_str(self.content, field_name="content")
        if not isinstance(self.source_agent, AgentRole):
            raise ProtocolError("source_agent must be an AgentRole")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ProtocolError("confidence must be in [0, 1]")
        if not isinstance(self.verification_status, VerificationStatus):
            raise ProtocolError("verification_status must be a VerificationStatus")
        _as_ts(self.created_at)
        _as_ts(self.updated_at, field_name="updated_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "knowledge_id": self.knowledge_id,
            "category": self.category,
            "content": self.content,
            "source_agent": self.source_agent.value,
            "evidence_refs": list(self.evidence_refs),
            "confidence": float(self.confidence),
            "verification_status": self.verification_status.value,
            "branch_id": self.branch_id,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "superseded_by": self.superseded_by,
            "subject": self.subject,
            "tags": list(self.tags),
            "revision": int(self.revision),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeEntry":
        if not isinstance(data, Mapping):
            raise ProtocolError("knowledge record must be a JSON object")
        entry = cls(
            knowledge_id=_as_id(
                data.get("knowledge_id"), field_name="knowledge_id", prefix=KNOWLEDGE_ID_PREFIX
            ),
            category=str(data.get("category") or "other").strip().lower(),
            content=_as_str(data.get("content"), field_name="content"),
            source_agent=AgentRole.coerce(data.get("source_agent")),
            evidence_refs=_as_str_list(data.get("evidence_refs"), field_name="evidence_refs"),
            confidence=float(data.get("confidence") or 0.0),
            verification_status=VerificationStatus.coerce(
                data.get("verification_status", VerificationStatus.HYPOTHESIS)
            ),
            branch_id=str(data.get("branch_id") or ""),
            created_at=_as_ts(data.get("created_at", 0.0)),
            updated_at=_as_ts(data.get("updated_at", data.get("created_at", 0.0)), field_name="updated_at"),
            superseded_by=str(data.get("superseded_by") or ""),
            subject=str(data.get("subject") or ""),
            tags=_as_str_list(data.get("tags"), field_name="tags"),
            revision=int(data.get("revision") or 1),
        )
        entry.validate()
        return entry


# --------------------------------------------------------------------------- #
# ControlLease
# --------------------------------------------------------------------------- #

@dataclass
class ControlLease:
    """Explicit token proving which agent may press emulator buttons.

    The lease is intentionally a plain record: the :class:`ControlArbiter` keeps
    the authoritative state and hands the token to the owning agent's emulator
    proxy.  A stale token can never re-acquire control because revocations are
    recorded with a monotonically increasing ``generation``.
    """

    lease_id: str
    owner: ControlOwner
    agent: AgentRole
    task_id: str = ""
    reason: str = ""
    granted_at: float = 0.0
    released_at: float = 0.0
    generation: int = 0
    #: context handed back when the lease is released (navigation target, …)
    context: dict[str, Any] = field(default_factory=dict)
    revoked: bool = False

    @classmethod
    def create(
        cls,
        *,
        owner: ControlOwner | str,
        agent: AgentRole | str | None = None,
        task_id: str = "",
        reason: str = "",
        generation: int = 0,
        context: Mapping[str, Any] | None = None,
        granted_at: float | None = None,
    ) -> "ControlLease":
        owner_enum = ControlOwner.coerce(owner)
        if owner_enum is ControlOwner.NONE:
            raise ProtocolError("cannot create a lease with owner NONE")
        role = AgentRole.coerce(agent) if agent is not None else (
            AgentRole.NAVIGATION
            if owner_enum is ControlOwner.NAVIGATION
            else AgentRole.STORY
            if owner_enum is ControlOwner.STORY
            else AgentRole.POKEMON
            if owner_enum is ControlOwner.POKEMON
            else AgentRole.SYSTEM
        )
        return cls(
            lease_id=new_lease_id(),
            owner=owner_enum,
            agent=role,
            task_id=str(task_id or ""),
            reason=str(reason or ""),
            granted_at=float(granted_at if granted_at is not None else now_ts()),
            generation=int(generation),
            context=dict(context or {}),
        )

    @property
    def active(self) -> bool:
        return not self.revoked and self.released_at <= 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "owner": self.owner.value,
            "agent": self.agent.value,
            "task_id": self.task_id,
            "reason": self.reason,
            "granted_at": float(self.granted_at),
            "released_at": float(self.released_at),
            "generation": int(self.generation),
            "context": dict(self.context),
            "revoked": bool(self.revoked),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControlLease":
        return cls(
            lease_id=_as_id(data.get("lease_id"), field_name="lease_id", prefix=LEASE_ID_PREFIX),
            owner=ControlOwner.coerce(data.get("owner")),
            agent=AgentRole.coerce(data.get("agent")),
            task_id=str(data.get("task_id") or ""),
            reason=str(data.get("reason") or ""),
            granted_at=_as_ts(data.get("granted_at", 0.0), field_name="granted_at"),
            released_at=_as_ts(data.get("released_at", 0.0), field_name="released_at"),
            generation=int(data.get("generation") or 0),
            context=_as_dict(data.get("context"), field_name="context"),
            revoked=bool(data.get("revoked")),
        )


# --------------------------------------------------------------------------- #
# BranchRecord
# --------------------------------------------------------------------------- #

@dataclass
class BranchRecord:
    """One checkpoint branch of the game (rollbacks create new branches)."""

    branch_id: str
    parent_branch_id: str = ""
    checkpoint_id: str = ""
    reason: str = ""
    status: BranchStatus = BranchStatus.ACTIVE
    created_at: float = 0.0
    closed_at: float = 0.0
    #: ids of tasks that were active when the branch was created
    task_ids: list[str] = field(default_factory=list)
    #: rolling-back branch this one replaced (for audit)
    replaced_branch_id: str = ""
    #: free-form evidence describing why the rollback happened
    evidence: dict[str, Any] = field(default_factory=dict)
    #: number of rollbacks that happened on this branch lineage
    rollback_count: int = 0

    @classmethod
    def create(
        cls,
        *,
        parent_branch_id: str = "",
        checkpoint_id: str = "",
        reason: str = "",
        task_ids: Iterable[str] | None = None,
        evidence: Mapping[str, Any] | None = None,
        branch_id: str = "",
        created_at: float | None = None,
    ) -> "BranchRecord":
        return cls(
            branch_id=branch_id or new_branch_id(),
            parent_branch_id=str(parent_branch_id or ""),
            checkpoint_id=str(checkpoint_id or ""),
            reason=str(reason or ""),
            status=BranchStatus.ACTIVE,
            created_at=float(created_at if created_at is not None else now_ts()),
            task_ids=[str(t) for t in (task_ids or [])],
            evidence=dict(evidence or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "checkpoint_id": self.checkpoint_id,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": float(self.created_at),
            "closed_at": float(self.closed_at),
            "task_ids": list(self.task_ids),
            "replaced_branch_id": self.replaced_branch_id,
            "evidence": dict(self.evidence),
            "rollback_count": int(self.rollback_count),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BranchRecord":
        return cls(
            branch_id=_as_id(data.get("branch_id"), field_name="branch_id", prefix=BRANCH_ID_PREFIX),
            parent_branch_id=str(data.get("parent_branch_id") or ""),
            checkpoint_id=str(data.get("checkpoint_id") or ""),
            reason=str(data.get("reason") or ""),
            status=BranchStatus.coerce(data.get("status", BranchStatus.ACTIVE)),
            created_at=_as_ts(data.get("created_at", 0.0)),
            closed_at=_as_ts(data.get("closed_at", 0.0), field_name="closed_at"),
            task_ids=_as_str_list(data.get("task_ids"), field_name="task_ids"),
            replaced_branch_id=str(data.get("replaced_branch_id") or ""),
            evidence=_as_dict(data.get("evidence"), field_name="evidence"),
            rollback_count=int(data.get("rollback_count") or 0),
        )


# --------------------------------------------------------------------------- #
# Progress / recovery records
# --------------------------------------------------------------------------- #

@dataclass
class ProgressSignal:
    """A progress (or stagnation) observation produced by the ProgressMonitor."""

    kind: str
    agent: AgentRole | None
    detail: str
    delta: float = 0.0
    created_at: float = 0.0
    branch_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "agent": self.agent.value if self.agent else "",
            "detail": self.detail,
            "delta": float(self.delta),
            "created_at": float(self.created_at),
            "branch_id": self.branch_id,
        }


#: Every event type that must be routable to a professional agent's inbox.
AGENT_EVENTS: dict[AgentRole, tuple[EventType, ...]] = {
    AgentRole.NAVIGATION: (
        EventType.NAVIGATION_REQUEST,
        EventType.TASK_ASSIGNED,
        EventType.TASK_CANCELLED,
        EventType.CONTROL_GRANTED,
        EventType.CONTROL_RELEASED,
    ),
    AgentRole.STORY: (
        EventType.DIALOG_STARTED,
        EventType.TASK_ASSIGNED,
        EventType.TASK_CANCELLED,
        EventType.CONTROL_GRANTED,
        EventType.CONTROL_RELEASED,
    ),
    AgentRole.POKEMON: (
        EventType.BATTLE_STARTED,
        EventType.POKEMON_NEEDS_HELP,
        EventType.TASK_ASSIGNED,
        EventType.TASK_CANCELLED,
        EventType.CONTROL_GRANTED,
        EventType.CONTROL_RELEASED,
    ),
    AgentRole.DIRECTOR: tuple(EventType),
    AgentRole.SYSTEM: tuple(EventType),
}


def event_types_for(role: AgentRole | str) -> tuple[EventType, ...]:
    """Event types a given agent may receive."""
    return AGENT_EVENTS.get(AgentRole.coerce(role), ())


__all__ = [
    "PROTOCOL_VERSION",
    "AgentRole",
    "ControlOwner",
    "TaskStatus",
    "TaskType",
    "VerificationStatus",
    "Priority",
    "StagnationKind",
    "RecoveryLevel",
    "BranchStatus",
    "EventType",
    "EVENT_PAYLOAD_SPEC",
    "EVENT_ROUTING",
    "IRREVERSIBLE_EVENTS",
    "BOOKKEEPING_EVENTS",
    "KNOWLEDGE_CATEGORIES",
    "TASK_TRANSITIONS",
    "is_valid_transition",
    "AgentMessage",
    "Task",
    "KnowledgeEntry",
    "ControlLease",
    "BranchRecord",
    "ProgressSignal",
    "AGENT_EVENTS",
    "event_types_for",
    "MultiAgentError",
    "ProtocolError",
    "ControlDeniedError",
    "DuplicateMessageError",
    "TaskError",
    "CycleDetectedError",
    "KnowledgeError",
    "RecoveryLimitError",
    "StagnationLimitError",
    "BudgetLimitError",
    "now_ts",
    "new_message_id",
    "new_task_id",
    "new_knowledge_id",
    "new_branch_id",
    "new_lease_id",
    "stable_digest",
    "dedupe_key",
]
