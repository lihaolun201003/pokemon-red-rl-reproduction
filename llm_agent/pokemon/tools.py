"""Tool schemas and strict validation for the autonomous battle agent.

The battle brain may only act through four explicitly declared tools:

* ``battle_action`` — name ONE battle action (fight/run/catch/item/switch/
  observe/finish) together with the visible slot to use,
* ``observe`` — ask for a fresh look without pressing anything,
* ``record_lesson`` — save a durable lesson (explicitly marked verified or
  inferred),
* ``finish`` — declare that the battle is visibly over.

This module is pure Python: it never touches the emulator, the API client or
the disk.  :class:`BattleToolExecutor` is the *validating executor*: it is the
only place that converts raw model tool calls (or a JSON fallback in the reply
text) into the :class:`BattleDecision` / :class:`LessonSpec` objects the agent
loop consumes.  Invalid calls are rejected with a structured error instead of
being guessed at, so a bad model reply can never drive arbitrary input.

Honesty rules encoded in the schemas:

* only *visible* observations are accepted (the fields are a closed set, and
  unknown/hidden values such as numeric enemy HP are dropped),
* a lesson must explicitly say whether it is ``verified`` or ``inferred``,
* a finish outcome must be one of the observable results (win/loss/caught/ran)
  or ``unknown`` — the model can never invent a hidden result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from llm_agent.contracts import MAX_DECISION_CHARS, ModelReply, ToolCall, ToolValidationError
from llm_agent.multi_agent.contracts import KNOWLEDGE_CATEGORIES

#: The four battle tools the director registry expects this agent to own.
TOOL_NAMES: tuple[str, ...] = ("battle_action", "observe", "record_lesson", "finish")

#: Actions the model may choose (one per turn).
BATTLE_ACTIONS: tuple[str, ...] = (
    "fight",
    "run",
    "catch",
    "item",
    "switch",
    "observe",
    "finish",
)

#: Outcomes a ``finish`` decision may report.  Anything else is a validation
#: error: the agent must not invent hidden results.
FINISH_OUTCOMES: tuple[str, ...] = ("win", "loss", "caught", "ran", "unknown")

#: Visible facts a decision may report (closed set; read from the screenshot).
VISIBLE_OBSERVATION_KEYS: tuple[str, ...] = (
    "move_name",
    "damage_text",
    "hp_text",
    "battle_text",
    "enemy_hp_bar",
    "my_hp_bar",
)

#: Categories allowed for ``record_lesson`` (subset of the frozen catalogue).
LESSON_CATEGORIES: tuple[str, ...] = tuple(
    category
    for category in ("battle_lesson", "pokemon", "hazard", "other")
    if category in KNOWLEDGE_CATEGORIES
)

#: Verification labels allowed for a lesson (no ``hypothesis``/``refuted`` here:
#: a lesson is either something the model saw or something it inferred).
LESSON_VERIFICATIONS: tuple[str, ...] = ("verified", "inferred")

MAX_MOVE_SLOTS: int = 4
MAX_ITEM_SLOTS: int = 8
MAX_PARTY_SLOTS: int = 6
MAX_LESSON_CHARS: int = 400
MAX_REASON_CHARS: int = 300
MAX_HYPOTHESIS_CHARS: int = 240
MAX_EVIDENCE_REFS: int = 8
MAX_LESSONS_PER_TURN: int = 2

#: Name of the JSON answer the model may use when it cannot emit tool calls.
_CONTENT_DECISION_HINT = "action"


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

BATTLE_ACTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "battle_action",
        "description": (
            "Choose exactly one battle action for this turn based only on what is visible "
            "on the screenshot. The deterministic driver converts it into verified button "
            "presses; the model never presses buttons itself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(BATTLE_ACTIONS),
                    "description": (
                        "fight: use a visible move slot; run: try to flee; catch: use a "
                        "visible item row to capture; item: use a visible item row; "
                        "switch: switch to a visible party row; observe: look again "
                        "without pressing; finish: the battle is visibly over."
                    ),
                },
                "move_slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_MOVE_SLOTS,
                    "description": "Visible move row to use (1 = top row of the move list).",
                },
                "item_slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ITEM_SLOTS,
                    "description": "Visible item row to use (1 = first row of the item list).",
                },
                "party_slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_PARTY_SLOTS,
                    "description": "Visible party row (1 = first row of the party list).",
                },
                "reason": {
                    "type": "string",
                    "maxLength": MAX_REASON_CHARS,
                    "description": "One short sentence explaining the choice.",
                },
                "visible_observations": {
                    "type": "object",
                    "description": (
                        "Only facts readable on the screenshot. Never hidden values: no "
                        "numeric enemy HP, no unseen moves, no stats."
                    ),
                    "properties": {
                        key: {"type": "string", "maxLength": 200}
                        for key in VISIBLE_OBSERVATION_KEYS
                    },
                    "additionalProperties": False,
                },
                "hypothesis": {
                    "type": "string",
                    "maxLength": MAX_HYPOTHESIS_CHARS,
                    "description": "A guess/inference (stored as INFERRED, never as fact).",
                },
                "outcome": {
                    "type": "string",
                    "enum": list(FINISH_OUTCOMES),
                    "description": "Only for finish: the visibly readable result.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}

OBSERVE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "observe",
        "description": (
            "Ask for one fresh observation without pressing any button. Costs a turn; "
            "use only when the current screen is unreadable or still animating."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "maxLength": MAX_REASON_CHARS},
            },
            "additionalProperties": False,
        },
    },
}

RECORD_LESSON_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record_lesson",
        "description": (
            "Save one durable battle lesson. `verified` = you read it on screen; "
            "`inferred` = it is a guess. Never mark a guess as verified."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "maxLength": MAX_LESSON_CHARS},
                "verification": {
                    "type": "string",
                    "enum": list(LESSON_VERIFICATIONS),
                    "description": "verified (seen on screen) or inferred (guess).",
                },
                "category": {
                    "type": "string",
                    "enum": list(LESSON_CATEGORIES),
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": MAX_EVIDENCE_REFS,
                },
            },
            "required": ["content", "verification"],
            "additionalProperties": False,
        },
    },
}

FINISH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Declare that the battle is visibly over and hand control back. Report the "
            "outcome only if it is readable on screen, else 'unknown'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": list(FINISH_OUTCOMES)},
                "visible_text": {"type": "string", "maxLength": 300},
                "reason": {"type": "string", "maxLength": MAX_REASON_CHARS},
            },
            "required": ["outcome"],
            "additionalProperties": False,
        },
    },
}

#: All tool schemas handed to the API in one list.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    BATTLE_ACTION_TOOL,
    OBSERVE_TOOL,
    RECORD_LESSON_TOOL,
    FINISH_TOOL,
]

#: Name -> schema lookup (handy for tests/tools).
TOOL_SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}


# --------------------------------------------------------------------------- #
# Decision / lesson records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BattleDecision:
    """One validated battle decision produced by the model.

    ``raw_arguments`` keeps the exact dict the model emitted (or the JSON
    fallback) so the acceptance report can prove the action came from the LLM
    and not from a hard-coded script.
    """

    action: str
    move_slot: int | None = None
    item_slot: int | None = None
    party_slot: int | None = None
    reason: str = ""
    visible_observations: dict[str, str] = field(default_factory=dict)
    hypothesis: str = ""
    outcome: str = "unknown"
    source: str = "tool_call"
    tool_call_id: str = ""
    raw_arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (used in the returned ``actions`` list)."""
        return {
            "action": self.action,
            "move_slot": self.move_slot,
            "item_slot": self.item_slot,
            "party_slot": self.party_slot,
            "reason": self.reason,
            "visible_observations": dict(self.visible_observations),
            "hypothesis": self.hypothesis,
            "outcome": self.outcome,
            "source": self.source,
            "tool_call_id": self.tool_call_id,
            "raw": dict(self.raw_arguments),
        }

    @property
    def slot(self) -> int | None:
        """The slot relevant for this action (move/item/party) or ``None``."""
        if self.action == "fight":
            return self.move_slot
        if self.action in ("catch", "item"):
            return self.item_slot
        if self.action == "switch":
            return self.party_slot
        return None


@dataclass(frozen=True)
class LessonSpec:
    """One validated ``record_lesson`` call."""

    content: str
    verification: str
    category: str
    confidence: float
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "verification": self.verification,
            "category": self.category,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class ParsedReply:
    """Result of validating one model reply."""

    decision: BattleDecision | None = None
    lessons: list[LessonSpec] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: one executor result dict per entry of :attr:`tool_calls` (same order)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    assistant_content: str = ""

    @property
    def ok(self) -> bool:
        """True when at least one usable decision was extracted."""
        return self.decision is not None


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _clean_text(value: Any, limit: int) -> str:
    """Collapse whitespace and truncate ``value`` to ``limit`` characters."""
    if value is None:
        return ""
    return " ".join(str(value).split())[: max(0, int(limit))]


def _as_bool_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ToolValidationError(f"{name} must be a number, got {type(value).__name__}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"{name} must be a number, got {value!r}") from exc
    return number


def _coerce_slot(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    """Coerce a 1-based visible menu row, tolerating a numeric string."""
    if isinstance(value, bool):
        raise ToolValidationError(f"{name} must be an integer, got a boolean")
    if value is None:
        return minimum
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return minimum
        try:
            value = int(stripped)
        except ValueError as exc:
            raise ToolValidationError(f"{name} must be an integer, got {value!r}") from exc
    if not isinstance(value, int):
        try:
            value = int(float(value))
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"{name} must be an integer, got {value!r}") from exc
    if not minimum <= int(value) <= maximum:
        raise ToolValidationError(
            f"{name} must be in {minimum}..{maximum}, got {int(value)}"
        )
    return int(value)


def coerce_visible_observations(raw: Any) -> dict[str, str]:
    """Keep only the closed set of visible observation fields.

    Unknown keys are silently dropped on purpose: a model trying to smuggle a
    hidden value (e.g. numeric enemy HP or an unseen move) into the decision
    cannot make it into memory this way.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ToolValidationError("visible_observations must be an object")
    kept: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip().lower()
        if name not in VISIBLE_OBSERVATION_KEYS:
            continue
        if value is None:
            continue
        text = _clean_text(value, 200)
        if text:
            kept[name] = text
    return kept


def _coerce_outcome(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text not in FINISH_OUTCOMES:
        raise ToolValidationError(
            f"outcome must be one of {', '.join(FINISH_OUTCOMES)}, got {value!r}"
        )
    return text


def validate_decision(
    arguments: Mapping[str, Any] | None,
    *,
    source: str = "tool_call",
    tool_call_id: str = "",
) -> BattleDecision:
    """Validate ``battle_action`` / ``finish`` arguments into a decision.

    Raises
    ------
    ToolValidationError
        When the action is unknown or a slot/outcome is invalid.  A missing
        slot defaults to the first visible row (a deliberately forgiving
        default: the driver still verifies every press).
    """
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("battle_action arguments must be a JSON object")
    raw_action = arguments.get("action")
    if not isinstance(raw_action, str) or not raw_action.strip():
        raise ToolValidationError("battle_action is missing a string 'action'")
    action = raw_action.strip().lower()
    if action not in BATTLE_ACTIONS:
        raise ToolValidationError(
            f"unknown battle action {raw_action!r}; allowed: {', '.join(BATTLE_ACTIONS)}"
        )

    move_slot: int | None = None
    item_slot: int | None = None
    party_slot: int | None = None
    outcome = "unknown"
    if action == "fight":
        move_slot = _coerce_slot(
            arguments.get("move_slot"), name="move_slot", minimum=1, maximum=MAX_MOVE_SLOTS
        )
    elif action in ("catch", "item"):
        item_slot = _coerce_slot(
            arguments.get("item_slot"), name="item_slot", minimum=1, maximum=MAX_ITEM_SLOTS
        )
    elif action == "switch":
        party_slot = _coerce_slot(
            arguments.get("party_slot"), name="party_slot", minimum=1, maximum=MAX_PARTY_SLOTS
        )
    elif action == "finish":
        outcome = _coerce_outcome(arguments.get("outcome"))
    elif action == "observe":
        # The observe tool shares this validator; it never carries a slot.
        pass

    return BattleDecision(
        action=action,
        move_slot=move_slot,
        item_slot=item_slot,
        party_slot=party_slot,
        reason=_clean_text(arguments.get("reason"), MAX_REASON_CHARS),
        visible_observations=coerce_visible_observations(arguments.get("visible_observations")),
        hypothesis=_clean_text(arguments.get("hypothesis"), MAX_HYPOTHESIS_CHARS),
        outcome=outcome,
        source=str(source or "tool_call"),
        tool_call_id=str(tool_call_id or ""),
        raw_arguments=dict(arguments),
    )


def validate_lesson(arguments: Mapping[str, Any] | None) -> LessonSpec:
    """Validate ``record_lesson`` arguments into a :class:`LessonSpec`."""
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("record_lesson arguments must be a JSON object")
    content = _clean_text(arguments.get("content"), MAX_LESSON_CHARS)
    if not content:
        raise ToolValidationError("record_lesson content must not be empty")
    verification = _clean_text(arguments.get("verification") or "", 20).lower()
    if verification not in LESSON_VERIFICATIONS:
        raise ToolValidationError(
            f"verification must be one of {', '.join(LESSON_VERIFICATIONS)}, "
            f"got {arguments.get('verification')!r}"
        )
    category = _clean_text(arguments.get("category") or "battle_lesson", 40).lower()
    if category not in LESSON_CATEGORIES:
        raise ToolValidationError(
            f"unknown lesson category {category!r}; allowed: {', '.join(LESSON_CATEGORIES)}"
        )
    confidence_raw = arguments.get("confidence")
    if confidence_raw is None:
        confidence = 0.8 if verification == "verified" else 0.5
    else:
        confidence = _as_bool_number(confidence_raw, name="confidence")
    confidence = float(max(0.0, min(1.0, confidence)))
    evidence_raw = arguments.get("evidence") or ()
    if isinstance(evidence_raw, (str, bytes)) or not isinstance(evidence_raw, Sequence):
        raise ToolValidationError("evidence must be a list of strings")
    evidence: list[str] = []
    for item in evidence_raw:
        text = _clean_text(item, 300)
        if text:
            evidence.append(text)
    return LessonSpec(
        content=content,
        verification=verification,
        category=category,
        confidence=confidence,
        evidence_refs=tuple(evidence[:MAX_EVIDENCE_REFS]),
    )


# --------------------------------------------------------------------------- #
# The validating executor
# --------------------------------------------------------------------------- #

class BattleToolExecutor:
    """Validate model tool calls; never touches the emulator or the network.

    ``execute`` is the single entry point per tool call and always returns a
    structured dict (``ok``, ``summary``, ``decision``, ``lesson``, ``error``),
    so an invalid model reply is reported instead of being partially applied.
    ``parse_reply`` additionally extracts the fallback JSON decision some models
    emit in the message content when they cannot produce tool calls.
    """

    def __init__(self, *, max_lessons_per_turn: int = MAX_LESSONS_PER_TURN) -> None:
        self.max_lessons_per_turn = max(0, int(max_lessons_per_turn))

    # -- introspection --------------------------------------------------- #
    @property
    def tools(self) -> list[dict[str, Any]]:
        """The OpenAI-style tool schemas this executor accepts."""
        return [dict(schema) for schema in TOOL_SCHEMAS]

    def names(self) -> tuple[str, ...]:
        return TOOL_NAMES

    # -- single call ------------------------------------------------------ #
    def execute(self, call: Any) -> dict[str, Any]:
        """Validate one :class:`~llm_agent.contracts.ToolCall`."""
        result: dict[str, Any] = {
            "ok": False,
            "name": "",
            "summary": "",
            "decision": None,
            "lesson": None,
            "error": None,
            "call_id": "",
        }
        if not isinstance(call, ToolCall):
            result["error"] = f"expected a ToolCall, got {type(call).__name__}"
            return result
        name = str(call.name or "").strip()
        result["name"] = name
        result["call_id"] = str(call.id or "")
        if name not in TOOL_NAMES:
            result["error"] = f"unknown tool {name!r}; allowed: {', '.join(TOOL_NAMES)}"
            return result
        try:
            if name == "battle_action":
                decision = validate_decision(
                    call.arguments, source="tool_call", tool_call_id=str(call.id or "")
                )
                result["ok"] = True
                result["decision"] = decision
                result["summary"] = (
                    f"accepted battle_action: {decision.action}"
                    + (f" slot={decision.slot}" if decision.slot is not None else "")
                )
            elif name == "observe":
                decision = validate_decision(
                    {"action": "observe", "reason": (call.arguments or {}).get("reason", "")},
                    source="tool_call",
                    tool_call_id=str(call.id or ""),
                )
                result["ok"] = True
                result["decision"] = decision
                result["summary"] = "accepted observe: fresh look without pressing"
            elif name == "record_lesson":
                lesson = validate_lesson(call.arguments)
                result["ok"] = True
                result["lesson"] = lesson
                result["summary"] = (
                    f"accepted record_lesson ({lesson.verification}, {lesson.category})"
                )
            else:  # finish
                decision = validate_decision(
                    {
                        **dict(call.arguments or {}),
                        "action": "finish",
                    },
                    source="tool_call",
                    tool_call_id=str(call.id or ""),
                )
                result["ok"] = True
                result["decision"] = decision
                result["summary"] = f"accepted finish: outcome={decision.outcome}"
        except ToolValidationError as exc:
            result["error"] = str(exc)
            result["summary"] = f"rejected {name}: {exc}"
        return result

    # -- whole reply ------------------------------------------------------ #
    def parse_reply(self, reply: ModelReply | None) -> ParsedReply:
        """Validate every tool call of one reply and extract the turn decision.

        The first valid action tool call wins; further action calls are
        rejected as errors (one decision per turn).  If the model emitted no
        tool call at all, a JSON object in ``reply.content`` is accepted as a
        fallback so a slow/quirky model cannot stall the battle forever.
        """
        parsed = ParsedReply()
        if reply is None:
            parsed.errors.append("empty model reply")
            return parsed
        parsed.assistant_content = _clean_text(getattr(reply, "content", ""), 2000)
        calls = list(getattr(reply, "tool_calls", None) or [])
        for call in calls:
            if not isinstance(call, ToolCall):
                parsed.errors.append(
                    f"tool call is not a ToolCall object: {type(call).__name__}"
                )
                continue
            parsed.tool_calls.append(call)
            result = self.execute(call)
            parsed.tool_results.append(result)
            if not result["ok"]:
                parsed.errors.append(str(result["error"] or "invalid tool call"))
                continue
            decision = result.get("decision")
            lesson = result.get("lesson")
            if decision is not None:
                if parsed.decision is None:
                    parsed.decision = decision
                else:
                    parsed.errors.append(
                        "only one battle action per turn; the extra action was ignored"
                    )
            elif lesson is not None:
                if len(parsed.lessons) < self.max_lessons_per_turn:
                    parsed.lessons.append(lesson)
                else:
                    parsed.errors.append(
                        f"at most {self.max_lessons_per_turn} record_lesson calls per turn"
                    )
        if parsed.decision is None:
            fallback = self._decision_from_content(parsed.assistant_content)
            if fallback is not None:
                parsed.decision = fallback
            elif not parsed.errors:
                parsed.errors.append("reply contains no battle_action tool call or JSON decision")
        return parsed

    def _decision_from_content(self, content: str) -> BattleDecision | None:
        """Best-effort JSON decision fallback for a content-only reply."""
        text = str(content or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        arguments: Any = payload
        if "action" not in payload:
            tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
            nested = payload.get("arguments")
            if tool_name in ("battle_action", "finish") and isinstance(nested, Mapping):
                arguments = dict(nested)
            else:
                return None
        try:
            return validate_decision(arguments, source="content_json")
        except ToolValidationError:
            return None


__all__ = [
    "TOOL_NAMES",
    "BATTLE_ACTIONS",
    "FINISH_OUTCOMES",
    "VISIBLE_OBSERVATION_KEYS",
    "LESSON_CATEGORIES",
    "LESSON_VERIFICATIONS",
    "MAX_MOVE_SLOTS",
    "MAX_ITEM_SLOTS",
    "MAX_PARTY_SLOTS",
    "BATTLE_ACTION_TOOL",
    "OBSERVE_TOOL",
    "RECORD_LESSON_TOOL",
    "FINISH_TOOL",
    "TOOL_SCHEMAS",
    "TOOL_SCHEMA_BY_NAME",
    "BattleDecision",
    "LessonSpec",
    "ParsedReply",
    "validate_decision",
    "validate_lesson",
    "coerce_visible_observations",
    "BattleToolExecutor",
]
