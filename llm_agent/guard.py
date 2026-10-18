"""Loop / stuck / anomaly detection for the DeepSeek game agent (owner: memory-guard).

``LoopGuard`` watches only the primitives the runtime already has — captured
:class:`~llm_agent.contracts.Frame` objects, :class:`~llm_agent.contracts.ActionOutcome`
batches, :class:`~llm_agent.contracts.ToolResult` objects and API call
success/failure — and turns them into short, actionable warnings that the
runtime appends to the next prompt.  It never reads RAM, map ids, story flags or
NPC positions, and it never imports PyBoy or :mod:`llm_agent.progress`.

Detected anomalies (one rule each, thresholds from :class:`~llm_agent.config.GuardConfig`):

* ``unchanged_movement`` — consecutive movement batches whose steps never
  changed the screen (walking into a wall).
* ``repeated_batch`` — the exact same action-batch sequence issued over and over.
* ``black_frame`` — consecutive fully black screenshots.
* ``same_screen`` — too many actions on the same screen signature.
* ``no_discovery`` — many actions without a successful ``record_discovery``.
* ``tool_errors`` — consecutive failed tool calls.
* ``api_failures`` — consecutive failed DeepSeek API calls.

Each warning has ``severity`` in ``{"info", "warning", "critical"}``: reaching a
configured limit yields ``warning``, twice the limit yields ``critical`` and
approaching it (>= 70%) yields an ``info`` advisory.  A limit of ``0`` disables
that rule.  :meth:`LoopGuard.check` is pure and deterministic (repeated calls
with the same state return equal lists), sorted by severity then rule order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from llm_agent.config import GuardConfig
from llm_agent.contracts import (
    DIRECTIONS,
    ActionOutcome,
    Frame,
    ScreenEvent,
    ToolResult,
    sha256_rgba,
)

GUARD_SNAPSHOT_VERSION = 1
SEVERITIES: tuple[str, ...] = ("info", "warning", "critical")

#: Fixed rule order (also the tie-break order inside one severity).
RULE_ORDER: tuple[str, ...] = (
    "unchanged_movement",
    "repeated_batch",
    "black_frame",
    "same_screen",
    "no_discovery",
    "tool_errors",
    "api_failures",
)

_SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class GuardWarning:
    """One actionable loop anomaly.

    Attributes
    ----------
    kind:
        Stable rule id (see :data:`RULE_ORDER`).
    severity:
        ``"info"`` (advisory), ``"warning"`` (act now) or ``"critical"``
        (restore/intervene).
    message:
        One factual sentence describing what was observed.
    suggestion:
        Short, concrete next action for the model.
    """

    kind: str
    severity: str
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable warning representation."""
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GuardWarning | None":
        """Tolerant parser for snapshots (returns ``None`` for junk entries)."""
        if not isinstance(raw, Mapping):
            return None
        kind = str(raw.get("kind") or "").strip()
        if not kind:
            return None
        severity = str(raw.get("severity") or "warning").strip().lower()
        if severity not in SEVERITIES:
            severity = "warning"
        return cls(
            kind=kind,
            severity=severity,
            message=str(raw.get("message") or ""),
            suggestion=str(raw.get("suggestion") or ""),
        )


def _frame_signature(frame: Frame) -> str:
    """Stable screen signature from a contract :class:`Frame` (hash only)."""
    signature = str(getattr(frame, "sha256", "") or "")
    if signature:
        return signature
    rgba = getattr(frame, "rgba", None)
    if rgba is None:
        return ""
    try:
        return sha256_rgba(rgba)
    except (TypeError, ValueError):  # pragma: no cover - malformed frame
        return ""


def _format_batch(buttons: tuple[str, ...] | None) -> str:
    if not buttons:
        return "[]"
    return "[" + ",".join(buttons) + "]"


class LoopGuard:
    """Deterministic, observation-only loop/anomaly detector.

    Feed it in the runtime loop:

    * :meth:`on_action_outcome` — once per executed tool batch.
    * :meth:`on_observation` — once per captured decision-cycle frame.
    * :meth:`on_tool_result` — once per executed tool call.
    * :meth:`on_llm_call` — once per DeepSeek call (success/failure).

    Then call :meth:`check` (or :meth:`warnings`) to get the current warnings and
    :meth:`render_warnings` to build the compact prompt block.  The guard is
    cheap (a handful of counters); it holds no frame buffers.
    """

    def __init__(self, cfg: GuardConfig) -> None:
        self.cfg = cfg
        self._warnings_cache: list[GuardWarning] | None = None
        self.reset()

    # -- lifecycle --------------------------------------------------------- #

    def reset(self) -> None:
        """Drop all state (both anomaly counters and lifetime totals)."""
        self.reset_after_recovery()
        self._actions_total = 0
        self._batches_total = 0
        self._observations = 0
        self._last_event = "none"

    def reset_after_recovery(self) -> None:
        """Clear anomaly state after a checkpoint restore / recovery.

        Lifetime totals (actions, batches, observations) are kept; every
        episodic counter that produces a warning is reset so the restored state
        starts with a clean slate.
        """
        self._unchanged_batches = 0
        self._repeat_count = 0
        self._last_batch: tuple[str, ...] | None = None
        self._black_frames = 0
        self._screen_signature: str | None = None
        self._same_screen_actions = 0
        self._actions_since_discovery = 0
        self._consecutive_tool_errors = 0
        self._consecutive_api_failures = 0
        self._last_tool_error = ""
        self._last_tool_name = ""
        self._warnings_cache = None

    # -- inputs ------------------------------------------------------------ #

    def on_action_outcome(self, outcome: ActionOutcome) -> None:
        """Feed one executed action batch (movement and repeated-batch detection)."""
        self._warnings_cache = None
        steps = list(getattr(outcome, "steps", None) or [])
        n_actions = len(steps)
        self._batches_total += 1
        self._actions_total += n_actions
        self._actions_since_discovery += n_actions

        buttons = tuple(str(button) for button in (getattr(outcome, "buttons", None) or []))

        # Screen-signature streak: count per-action frames that are pixel-identical.
        for step in steps:
            signature = _frame_signature(step.frame)
            if not signature:
                continue
            if self._screen_signature is None:
                self._screen_signature = signature
                self._same_screen_actions = 0
            elif signature == self._screen_signature:
                self._same_screen_actions += 1
            else:
                self._screen_signature = signature
                self._same_screen_actions = 0

        # Unchanged movement batches (walking into an obstacle).
        if buttons and all(button in DIRECTIONS for button in buttons) and n_actions:
            changed_steps = int(getattr(outcome, "changed_steps", 0) or 0)
            if changed_steps == 0:
                self._unchanged_batches += 1
            else:
                self._unchanged_batches = 0

        # Repeated identical batch.
        if buttons:
            if buttons == self._last_batch:
                self._repeat_count += 1
            else:
                self._last_batch = buttons
                self._repeat_count = 1

    def on_observation(self, frame: Frame, event: ScreenEvent, *, changed: bool) -> None:
        """Feed one captured frame (black-frame and screen-signature tracking)."""
        self._warnings_cache = None
        self._observations += 1
        try:
            self._last_event = event.value if isinstance(event, ScreenEvent) else str(event)
        except AttributeError:  # pragma: no cover - event is always a ScreenEvent
            self._last_event = str(event)
        if bool(getattr(frame, "is_black", False)):
            self._black_frames += 1
        else:
            self._black_frames = 0
        signature = _frame_signature(frame)
        if not signature:
            return
        if self._screen_signature is None:
            self._screen_signature = signature
        elif signature != self._screen_signature:
            self._screen_signature = signature
            self._same_screen_actions = 0

    def on_tool_result(self, result: ToolResult) -> None:
        """Feed one tool result (consecutive errors; discovery resets the clock)."""
        self._warnings_cache = None
        name = str(getattr(result, "name", "") or "")
        if bool(getattr(result, "ok", False)):
            self._consecutive_tool_errors = 0
            if name == "record_discovery":
                self._actions_since_discovery = 0
        else:
            self._consecutive_tool_errors += 1
            self._last_tool_name = name
            error = getattr(result, "error", None) or getattr(result, "summary", "")
            self._last_tool_error = f"{name}: {str(error)[:180]}".strip(": ")

    def on_llm_call(self, *, ok: bool) -> None:
        """Feed one DeepSeek API call outcome (consecutive API failures)."""
        self._warnings_cache = None
        if ok:
            self._consecutive_api_failures = 0
        else:
            self._consecutive_api_failures += 1

    # -- rule evaluation --------------------------------------------------- #

    @staticmethod
    def _escalate(count: int, limit: int) -> str | None:
        """Map a counter to ``info``/``warning``/``critical`` (or ``None``).

        ``limit <= 0`` disables the rule.  Reaching the limit is a ``warning``,
        twice the limit is ``critical``; at least 70% of the limit is ``info``.
        """
        limit = int(limit or 0)
        if limit <= 0 or count <= 0:
            return None
        if count >= 2 * limit:
            return "critical"
        if count >= limit:
            return "warning"
        info_at = max(1, (limit * 7) // 10)
        if count >= info_at:
            return "info"
        return None

    def check(self) -> list[GuardWarning]:
        """Evaluate every rule and return the deterministic warning list.

        Pure: the returned list is rebuilt from counters, sorted by severity
        (critical first) and then by :data:`RULE_ORDER`.
        """
        cfg = self.cfg
        warnings: list[GuardWarning] = []

        severity = self._escalate(self._unchanged_batches, cfg.unchanged_batch_limit)
        if severity:
            warnings.append(
                GuardWarning(
                    kind="unchanged_movement",
                    severity=severity,
                    message=(
                        f"{self._unchanged_batches} consecutive movement batches produced no "
                        f"visible screen change (last batch {_format_batch(self._last_batch)})."
                    ),
                    suggestion=(
                        "The way is blocked; try a different direction or walk around the "
                        "obstacle. If every direction fails, open the menu (start) or "
                        "restore_checkpoint('latest')."
                    ),
                )
            )

        severity = self._escalate(self._repeat_count, cfg.repeated_batch_limit)
        if severity:
            warnings.append(
                GuardWarning(
                    kind="repeated_batch",
                    severity=severity,
                    message=(
                        f"Identical action batch {_format_batch(self._last_batch)} issued "
                        f"{self._repeat_count} times in a row."
                    ),
                    suggestion=(
                        "Change the action sequence: press a different button, open the menu "
                        "with start, or talk to a nearby NPC; repeating the same batch rarely "
                        "makes progress."
                    ),
                )
            )

        severity = self._escalate(self._black_frames, cfg.black_frame_limit)
        if severity:
            warnings.append(
                GuardWarning(
                    kind="black_frame",
                    severity=severity,
                    message=(
                        f"{self._black_frames} consecutive screenshots were fully black "
                        f"(no visible game content)."
                    ),
                    suggestion=(
                        "The game may still be transitioning: use wait_frames(30-60) and "
                        "observe_game again. If black frames persist, restore_checkpoint('latest')."
                    ),
                )
            )

        severity = self._escalate(self._same_screen_actions, cfg.same_screen_action_limit)
        if severity:
            screen = (self._screen_signature or "")[:8] or "unknown"
            warnings.append(
                GuardWarning(
                    kind="same_screen",
                    severity=severity,
                    message=(
                        f"{self._same_screen_actions} actions in a row on the same screen "
                        f"(signature {screen}); the screen is not changing."
                    ),
                    suggestion=(
                        "You may be stuck in a menu or a wall: try the 'b' or 'start' button "
                        "to close/leave, or a direction you have not tried yet."
                    ),
                )
            )

        severity = self._escalate(self._actions_since_discovery, cfg.no_discovery_action_limit)
        if severity:
            warnings.append(
                GuardWarning(
                    kind="no_discovery",
                    severity=severity,
                    message=(
                        f"{self._actions_since_discovery} actions without a successful "
                        f"record_discovery."
                    ),
                    suggestion=(
                        "Record what you just saw (record_discovery) before it is lost, and "
                        "try exploring a different area or object."
                    ),
                )
            )

        severity = self._escalate(self._consecutive_tool_errors, cfg.max_consecutive_tool_errors)
        if severity:
            detail = f" Last error: {self._last_tool_error}." if self._last_tool_error else ""
            warnings.append(
                GuardWarning(
                    kind="tool_errors",
                    severity=severity,
                    message=(
                        f"{self._consecutive_tool_errors} consecutive tool calls failed."
                        f"{detail}"
                    ),
                    suggestion=(
                        "Re-read the tool manual and issue a simple valid call first "
                        "(observe_game); check required arguments and value ranges."
                    ),
                )
            )

        severity = self._escalate(
            self._consecutive_api_failures, cfg.max_consecutive_api_failures
        )
        if severity:
            warnings.append(
                GuardWarning(
                    kind="api_failures",
                    severity=severity,
                    message=(
                        f"{self._consecutive_api_failures} consecutive DeepSeek API calls "
                        f"failed (network/timeout/5xx)."
                    ),
                    suggestion=(
                        "The runtime backs off automatically; if this persists, pause the run "
                        "and check the network, API key and balance."
                    ),
                )
            )

        warnings.sort(key=lambda warning: (-_SEVERITY_RANK.get(warning.severity, 0),
                                           _rule_index(warning.kind)))
        self._warnings_cache = warnings
        return list(warnings)

    def warnings(self) -> list[GuardWarning]:
        """Cached result of :meth:`check` (recomputed after any new input)."""
        if self._warnings_cache is None:
            return self.check()
        return list(self._warnings_cache)

    def needs_attention(self) -> bool:
        """True when any warning (including ``info`` advisories) is present."""
        return bool(self.warnings())

    def critical(self) -> bool:
        """True when at least one ``critical`` warning is present."""
        return any(warning.severity == "critical" for warning in self.warnings())

    # -- prompt rendering -------------------------------------------------- #

    def render_warnings(self, max_chars: int = 700) -> str:
        """Compact warning block for the prompt (empty string when healthy)."""
        warnings = self.warnings()
        if not warnings:
            return ""
        budget = max(0, int(max_chars or 0))
        if budget <= 0:
            return ""
        header = f"GUARD WARNINGS ({len(warnings)}):"
        lines = [header]
        for warning in warnings:
            line = f"- [{warning.severity}] {warning.kind}: {warning.message}"
            if warning.suggestion:
                line += f" Suggestion: {warning.suggestion}"
            lines.append(line)

        def fit(candidate_lines: list[str]) -> str:
            out: list[str] = []
            used = 0
            for line in candidate_lines:
                extra = len(line) + (1 if out else 0)
                if used + extra <= budget:
                    out.append(line)
                    used += extra
                    continue
                room = budget - used - (1 if out else 0)
                if room > 1:
                    out.append(line[:room])
                break
            return "\n".join(out)[:budget]

        rendered = fit(lines)
        if rendered.count("\n") == 0 and header not in rendered and len(warnings) > 0:
            # Tiny budget: header did not fit with a warning line; show the first
            # warning directly instead of only the header.
            rendered = fit(lines[1:])
        if not rendered:
            rendered = lines[1][:budget] if len(lines) > 1 else ""
        return rendered[:budget]

    # -- snapshot / restore ------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable guard state (for checkpoints / audit)."""
        return {
            "version": GUARD_SNAPSHOT_VERSION,
            "unchanged_batches": self._unchanged_batches,
            "repeat_count": self._repeat_count,
            "last_batch": list(self._last_batch) if self._last_batch else None,
            "black_frames": self._black_frames,
            "screen_signature": self._screen_signature,
            "same_screen_actions": self._same_screen_actions,
            "actions_since_discovery": self._actions_since_discovery,
            "consecutive_tool_errors": self._consecutive_tool_errors,
            "consecutive_api_failures": self._consecutive_api_failures,
            "last_tool_error": self._last_tool_error,
            "last_tool_name": self._last_tool_name,
            "last_event": self._last_event,
            "actions_total": self._actions_total,
            "batches_total": self._batches_total,
            "observations": self._observations,
            "warnings": [warning.to_dict() for warning in self.warnings()],
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a :meth:`snapshot`; ignores junk and never raises."""
        self.reset_after_recovery()
        if not isinstance(snapshot, Mapping):
            return

        def as_count(key: str) -> int:
            value = snapshot.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return 0
            return max(0, int(value))

        self._unchanged_batches = as_count("unchanged_batches")
        self._repeat_count = as_count("repeat_count")
        self._black_frames = as_count("black_frames")
        self._same_screen_actions = as_count("same_screen_actions")
        self._actions_since_discovery = as_count("actions_since_discovery")
        self._consecutive_tool_errors = as_count("consecutive_tool_errors")
        self._consecutive_api_failures = as_count("consecutive_api_failures")
        self._actions_total = as_count("actions_total")
        self._batches_total = as_count("batches_total")
        self._observations = as_count("observations")

        raw_batch = snapshot.get("last_batch")
        if isinstance(raw_batch, (list, tuple)) and raw_batch:
            self._last_batch = tuple(str(button) for button in raw_batch)
        signature = snapshot.get("screen_signature")
        self._screen_signature = str(signature) if isinstance(signature, str) and signature else None
        last_error = snapshot.get("last_tool_error")
        self._last_tool_error = str(last_error)[:250] if isinstance(last_error, str) else ""
        last_name = snapshot.get("last_tool_name")
        self._last_tool_name = str(last_name) if isinstance(last_name, str) else ""
        last_event = snapshot.get("last_event")
        self._last_event = str(last_event) if isinstance(last_event, str) else "none"
        self._warnings_cache = None

    # -- diagnostics ------------------------------------------------------- #

    def stats(self) -> dict[str, Any]:
        """Counter snapshot for the viewer/report (no warnings evaluation)."""
        return {
            "actions_total": self._actions_total,
            "batches_total": self._batches_total,
            "observations": self._observations,
            "unchanged_batches": self._unchanged_batches,
            "repeat_count": self._repeat_count,
            "black_frames": self._black_frames,
            "same_screen_actions": self._same_screen_actions,
            "actions_since_discovery": self._actions_since_discovery,
            "consecutive_tool_errors": self._consecutive_tool_errors,
            "consecutive_api_failures": self._consecutive_api_failures,
            "last_event": self._last_event,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"LoopGuard(warnings={len(self.warnings())}, "
            f"critical={self.critical()}, actions={self._actions_total})"
        )


def _rule_index(kind: str) -> int:
    try:
        return RULE_ORDER.index(kind)
    except ValueError:
        return len(RULE_ORDER)


__all__ = ["GuardWarning", "LoopGuard", "RULE_ORDER", "SEVERITIES", "GUARD_SNAPSHOT_VERSION"]
