"""Single-owner control arbitration for the one shared emulator.

The whole multi-agent system drives **one** emulator, but at any instant exactly
one participant may press buttons.  :class:`ControlArbiter` implements that
mutual exclusion with an explicit, monotonically versioned
:class:`~llm_agent.multi_agent.contracts.ControlLease` token:

* :meth:`ControlArbiter.request` grants control when the emulator is free (or
  renews it for the current owner) and raises
  :class:`~llm_agent.multi_agent.contracts.ControlDeniedError` when another
  participant holds it.
* :meth:`ControlArbiter.release` records that the owner is done; the released
  lease keeps its ``context``/``task_id`` so the caller can persist the hand-off.
* :meth:`ControlArbiter.preempt` and :meth:`ControlArbiter.release_all` are the
  director's escape hatches: a crashed agent can never deadlock the run because
  the director can always force control back.
* A stale/revoked lease object can never operate: every token carries a
  ``generation`` that must match the arbiter's current lease, and
  :class:`GuardedEmulator` re-validates it on every mutating call.
* :meth:`ControlArbiter.proxy` returns a cached :class:`GuardedEmulator` per
  agent.  It is a transparent read-only view of the emulator (``observe``,
  ``frame_count``, … are always allowed) but gates all mutating methods behind
  the lease.  State/checkpoint operations additionally require
  ``OWNER_SYSTEM`` so a professional agent can never rewind or rewire the game.

Optional audit events (:data:`~llm_agent.multi_agent.contracts.EventType`
``CONTROL_*``) are published on an :class:`EventBus` when one is supplied.  Bus
failures are counted (``status()["event_errors"]``) but must never break a
control change.

The arbiter is pure Python: it only needs a duck-typed emulator exposing
``observe``/``press``/``wait``/``run_batch``/``load_state``/``save_state_*``, so
tests use a recording fake and never need PyBoy or a ROM.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Deque, Mapping

from llm_agent.multi_agent.contracts import (
    AgentMessage,
    AgentRole,
    ControlDeniedError,
    ControlLease,
    ControlOwner,
    EventType,
    ProtocolError,
)
from llm_agent.multi_agent.event_bus import EventBus

#: Maximum number of arbiter audit records kept in memory.
HISTORY_LIMIT: int = 500
#: Poll interval used by ``request(wait=True)`` (real time, always bounded).
WAIT_POLL_S: float = 0.02


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ArbiterConfig:
    """Tuning knobs for :class:`ControlArbiter`."""

    #: A lease older than this is stale (``<= 0`` disables expiry).
    lease_timeout_s: float = 300.0
    #: When True, every ownership access implicitly expires a stale lease.
    auto_expire: bool = True
    #: Safety counter: ``status()["denied_limit_reached"]`` turns True at this
    #: many denied control operations so tests/director can detect a fight.
    max_denied: int = 100


# --------------------------------------------------------------------------- #
# Guarded emulator proxy
# --------------------------------------------------------------------------- #

class GuardedEmulator:
    """Transparent, lease-checked proxy around the shared emulator for one agent.

    Read-only access (``observe``, ``last_frame``, ``frame_count``, ``started``,
    ``streaming``) is always allowed.  ``press``/``wait``/``run_batch`` require a
    currently valid lease bound to this proxy.  State/checkpoint methods require
    ``OWNER_SYSTEM``.  Denied calls increment :attr:`denied_count` and never
    touch the underlying emulator.
    """

    #: Extra read-only attributes that pass straight through to the emulator.
    _READ_ONLY_ATTRIBUTES: frozenset[str] = frozenset({"loaded_state"})

    def __init__(self, arbiter: "ControlArbiter", agent: AgentRole | str) -> None:
        if arbiter is None:
            raise ValueError("GuardedEmulator requires a ControlArbiter")
        self._arbiter = arbiter
        self._agent: AgentRole = AgentRole.coerce(agent)
        self._bound_lease: ControlLease | None = None
        self._denied_count: int = 0
        self._operations: int = 0

    # -- identity / diagnostics ----------------------------------------- #
    @property
    def agent(self) -> AgentRole:
        """The single agent this proxy was created for."""
        return self._agent

    @property
    def lease(self) -> ControlLease | None:
        """The lease bound to this proxy (``None`` before the first grant)."""
        return self._bound_lease

    @property
    def denied_count(self) -> int:
        """Number of denied (rejected) operations on this proxy."""
        return self._denied_count

    @property
    def operations(self) -> int:
        """Number of authorised mutating operations executed through this proxy."""
        return self._operations

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"GuardedEmulator(agent={self._agent.value!r}, "
            f"denied={self._denied_count}, operations={self._operations})"
        )

    # -- internal lease binding (called by ControlArbiter) --------------- #
    def bind_lease(self, lease: ControlLease | None) -> None:
        """Bind a lease to this proxy (internal; only the current arbiter lease)."""
        if lease is None:
            self._bound_lease = None
            return
        if not isinstance(lease, ControlLease):
            raise ProtocolError(
                f"bind_lease expects a ControlLease, got {type(lease).__name__}"
            )
        arbiter = self._arbiter
        with arbiter._lock:
            if arbiter._lease is not lease:
                raise ControlDeniedError(
                    "refusing to bind a lease that is not the current arbiter lease",
                    agent=self._agent.value,
                    owner=lease.owner.value,
                )
        self._bound_lease = lease

    def unbind(self) -> None:
        """Drop the bound lease (internal; called when control is released)."""
        self._bound_lease = None

    # -- read-only surface (always allowed) ------------------------------ #
    def observe(self) -> Any:
        """Capture the current frame (read-only; never requires a lease)."""
        return self._arbiter._emulator.observe()

    @property
    def frame_count(self) -> int:
        """Monotonic emulated frame counter (read-only)."""
        return self._require_raw("frame_count")

    @property
    def last_frame(self) -> Any:
        """Most recently captured frame (read-only)."""
        return self._require_raw("last_frame")

    @property
    def started(self) -> bool:
        """True while the emulator is running (read-only)."""
        return bool(self._require_raw("started"))

    @property
    def streaming(self) -> bool:
        """True when a real-time frame hook is installed (read-only)."""
        return bool(self._require_raw("streaming"))

    # -- mutating surface (lease required) ------------------------------- #
    def press(self, button: str, *, frames: int | None = None) -> Any:
        """Press one button; requires the current control lease."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_control_locked("press")
            self._operations += 1
            return arbiter._emulator.press(button, frames=frames)

    def wait(self, frames: int) -> Any:
        """Advance frames without input; requires the current control lease."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_control_locked("wait")
            self._operations += 1
            return arbiter._emulator.wait(frames)

    def run_batch(
        self,
        buttons: Any,
        *,
        stop_on_event: bool = True,
        step_hook: Callable[[Any], None] | None = None,
    ) -> Any:
        """Run a button batch; requires the current control lease."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_control_locked("run_batch")
            self._operations += 1
            return arbiter._emulator.run_batch(
                buttons, stop_on_event=stop_on_event, step_hook=step_hook
            )

    # -- state/checkpoint surface (OWNER_SYSTEM only) -------------------- #
    def load_state(self, path: Any) -> None:
        """Load a save state; only while ``OWNER_SYSTEM`` holds control."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_system_locked("load_state")
            arbiter._emulator.load_state(path)

    def save_state_bytes(self) -> bytes:
        """Serialize the emulator state; only while ``OWNER_SYSTEM`` holds control."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_system_locked("save_state_bytes")
            return arbiter._emulator.save_state_bytes()

    def save_state_file(self, path: Any) -> Any:
        """Write the emulator state to a file; only under ``OWNER_SYSTEM``."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_system_locked("save_state_file")
            return arbiter._emulator.save_state_file(path)

    def set_frame_hook(
        self,
        hook: Callable[[Any], None] | None,
        *,
        fps: float = 60.0,
    ) -> None:
        """Install the viewer frame hook; only while ``OWNER_SYSTEM`` holds control."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_system_locked("set_frame_hook")
            arbiter._emulator.set_frame_hook(hook, fps=fps)

    def clear_frame_hook(self) -> None:
        """Remove the viewer frame hook; only while ``OWNER_SYSTEM`` holds control."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_system_locked("clear_frame_hook")
            arbiter._emulator.clear_frame_hook()

    def snapshot_png(self, path: Any) -> Any:
        """Save a PNG of the current frame; only while ``OWNER_SYSTEM`` holds control."""
        arbiter = self._arbiter
        with arbiter._lock:
            arbiter._maybe_expire_locked()
            self._check_system_locked("snapshot_png")
            return arbiter._emulator.snapshot_png(path)

    # -- attribute plumbing ---------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        # Only a small, explicitly read-only surface is transparently forwarded;
        # anything else raises instead of exposing a path that skips the guard.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            arbiter = object.__getattribute__(self, "_arbiter")
        except AttributeError:  # pragma: no cover - construction ordering safety
            raise AttributeError(name) from None
        if name in self._READ_ONLY_ATTRIBUTES:
            return getattr(arbiter._emulator, name)
        raise AttributeError(
            f"{type(self).__name__} does not expose {name!r}; "
            "mutating operations must go through the lease-checked API"
        )

    # -- internals ------------------------------------------------------- #
    def _require_raw(self, attribute: str) -> Any:
        raw = self._arbiter._emulator
        try:
            return getattr(raw, attribute)
        except AttributeError:
            raise AttributeError(
                f"underlying emulator has no attribute {attribute!r}"
            ) from None

    def _check_control_locked(self, operation: str) -> ControlLease:
        """Return the current lease or deny; caller must hold ``arbiter._lock``."""
        arbiter = self._arbiter
        current = arbiter._lease
        bound = self._bound_lease
        if bound is None or current is None:
            owner_value = (
                current.owner.value if current is not None else ControlOwner.NONE.value
            )
            self._denied_count += 1
            raise arbiter._deny_locked(
                self._agent,
                owner=owner_value,
                operation=operation,
                reason="agent does not hold the control lease",
            )
        if bound is not current or bound.revoked or bound.released_at > 0.0:
            self._denied_count += 1
            raise arbiter._deny_locked(
                self._agent,
                owner=current.owner.value,
                operation=operation,
                reason="control lease is stale or revoked",
            )
        if not arbiter._role_owns_locked(self._agent, current):
            self._denied_count += 1
            raise arbiter._deny_locked(
                self._agent,
                owner=current.owner.value,
                operation=operation,
                reason=f"control is held by {current.owner.value}",
            )
        return current

    def _check_system_locked(self, operation: str) -> ControlLease:
        """Return the current lease if this proxy may run system operations."""
        arbiter = self._arbiter
        current = arbiter._lease
        bound = self._bound_lease
        allowed = (
            current is not None
            and current.active
            and bound is current
            and ControlOwner.from_role(self._agent) is ControlOwner.SYSTEM
        )
        if not allowed:
            self._denied_count += 1
            owner_value = (
                current.owner.value if current is not None else ControlOwner.NONE.value
            )
            raise arbiter._deny_locked(
                self._agent,
                owner=owner_value,
                operation=operation,
                reason="state operations require OWNER_SYSTEM",
            )
        return current


# --------------------------------------------------------------------------- #
# Control arbiter
# --------------------------------------------------------------------------- #

class ControlArbiter:
    """Mutual exclusion over the single emulator via explicit lease tokens.

    Example::

        arbiter = ControlArbiter(emulator, config=ArbiterConfig(), event_bus=bus)
        lease = arbiter.request(AgentRole.NAVIGATION, task_id=task.task_id,
                                context={"reason": "walk"})
        proxy = arbiter.proxy(AgentRole.NAVIGATION)
        proxy.press("up")
        released = arbiter.release(AgentRole.NAVIGATION, reason="done")
    """

    def __init__(
        self,
        emulator: Any,
        *,
        config: ArbiterConfig | None = None,
        event_bus: EventBus | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if emulator is None:
            raise ValueError("ControlArbiter requires an emulator instance")
        if config is not None and not isinstance(config, ArbiterConfig):
            raise ProtocolError("ControlArbiter config must be an ArbiterConfig instance")
        if event_bus is not None and not hasattr(event_bus, "publish"):
            raise ProtocolError("event_bus must expose a publish() method")
        self._emulator = emulator
        self._config = config if config is not None else ArbiterConfig()
        self._bus = event_bus
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._lock = threading.RLock()

        self._lease: ControlLease | None = None
        self._owner: ControlOwner = ControlOwner.NONE
        self._generation: int = 0
        self._proxies: dict[str, GuardedEmulator] = {}
        self._history: Deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self._last_released: ControlLease | None = None

        self._denied_count: int = 0
        self._grants: int = 0
        self._releases: int = 0
        self._preemptions: int = 0
        self._expiries: int = 0
        self._event_errors: int = 0

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #
    @property
    def emulator(self) -> Any:
        """The raw emulator wrapped by this arbiter (lifecycle owner: the session)."""
        return self._emulator

    @property
    def config(self) -> ArbiterConfig:
        """The configuration this arbiter was built with."""
        return self._config

    @property
    def owner(self) -> ControlOwner:
        """Current owner (``ControlOwner.NONE`` when the emulator is free)."""
        with self._lock:
            self._maybe_expire_locked()
            return self._owner

    @property
    def lease(self) -> ControlLease | None:
        """Currently valid control lease, or ``None``."""
        with self._lock:
            self._maybe_expire_locked()
            return self._lease

    # ------------------------------------------------------------------ #
    # Acquiring / renewing control
    # ------------------------------------------------------------------ #
    def request(
        self,
        agent: AgentRole | str,
        *,
        task_id: str = "",
        reason: str = "",
        context: Mapping[str, Any] | None = None,
        wait: bool = False,
        timeout_s: float = 0.0,
    ) -> ControlLease:
        """Request control; grant it if free/own, otherwise raise or wait briefly.

        ``wait=True`` polls until ``timeout_s`` (real seconds) elapse and then
        raises :class:`ControlDeniedError`; it can never block forever.
        Re-requesting while already owning the lease renews it and preserves the
        previous ``task_id``/``context`` unless new values are supplied.
        """
        role = AgentRole.coerce(agent)
        owner = ControlOwner.from_role(role)
        with self._lock:
            self._record("requested", agent=role, owner=owner.value, reason=reason)
            self._publish_control(
                EventType.CONTROL_REQUESTED,
                source=role,
                payload={
                    "owner": owner.value,
                    "reason": str(reason or ""),
                    "task_id": str(task_id or ""),
                },
            )
        deadline = (
            time.monotonic() + max(0.0, float(timeout_s))
            if wait
            else time.monotonic()
        )
        while True:
            busy_owner = ""
            with self._lock:
                self._maybe_expire_locked()
                current = self._lease
                if current is None or not current.active:
                    return self._grant_locked(
                        role,
                        owner,
                        task_id=task_id,
                        reason=reason,
                        context=context,
                    )
                if self._role_owns_locked(role, current):
                    return self._regrant_locked(
                        role,
                        current,
                        task_id=task_id,
                        reason=reason,
                        context=context,
                    )
                busy_owner = current.owner.value
            if not wait or time.monotonic() >= deadline:
                with self._lock:
                    raise self._deny_locked(
                        role,
                        owner=busy_owner,
                        operation="request",
                        reason=f"control is held by {busy_owner}",
                    )
            remaining = max(0.0, deadline - time.monotonic())
            time.sleep(min(WAIT_POLL_S, max(0.001, remaining)))

    def holds(self, agent: AgentRole | str) -> bool:
        """True when ``agent`` currently owns the emulator control lease."""
        role = AgentRole.coerce(agent)
        with self._lock:
            self._maybe_expire_locked()
            current = self._lease
            return (
                current is not None
                and current.active
                and self._role_owns_locked(role, current)
            )

    def require(self, agent: AgentRole | str, *, operation: str = "press") -> ControlLease:
        """Return the current lease for ``agent`` or raise ``ControlDeniedError``."""
        role = AgentRole.coerce(agent)
        with self._lock:
            self._maybe_expire_locked()
            current = self._lease
            if current is None or not current.active:
                raise self._deny_locked(
                    role,
                    owner=ControlOwner.NONE.value,
                    operation=operation,
                    reason="no agent currently holds control",
                )
            if not self._role_owns_locked(role, current):
                raise self._deny_locked(
                    role,
                    owner=current.owner.value,
                    operation=operation,
                    reason=f"control is held by {current.owner.value}",
                )
            return current

    def renew(self, lease: ControlLease | None = None) -> ControlLease:
        """Extend the current lease (optionally verifying the caller's token)."""
        with self._lock:
            self._maybe_expire_locked()
            current = self._lease
            if isinstance(lease, ControlLease):
                role = AgentRole.coerce(lease.agent)
            else:
                role = AgentRole.coerce(current.agent) if current is not None else AgentRole.SYSTEM
            if current is None or not current.active:
                raise self._deny_locked(
                    role,
                    owner=ControlOwner.NONE.value,
                    operation="renew",
                    reason="no active control lease to renew",
                )
            if not self._role_owns_locked(role, current):
                raise self._deny_locked(
                    role,
                    owner=current.owner.value,
                    operation="renew",
                    reason="lease belongs to another agent",
                )
            if isinstance(lease, ControlLease):
                stale = (
                    lease.lease_id != current.lease_id
                    or int(lease.generation) != int(current.generation)
                    or lease.revoked
                    or lease.released_at > 0.0
                )
                if stale:
                    raise self._deny_locked(
                        role,
                        owner=current.owner.value,
                        operation="renew",
                        reason="stale lease cannot be renewed",
                    )
            self._generation += 1
            renewed = replace(
                current,
                granted_at=float(self._now()),
                generation=int(self._generation),
                context=dict(current.context),
            )
            self._lease = renewed
            self._owner = renewed.owner
            self._rebind_locked(current, renewed)
            self._record("renewed", lease=renewed, reason="renew")
            return renewed

    # ------------------------------------------------------------------ #
    # Releasing / preempting control
    # ------------------------------------------------------------------ #
    def release(
        self,
        agent: AgentRole | str | None = None,
        *,
        reason: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> ControlLease | None:
        """Release control; returns the released lease (``None`` if already free).

        The returned token keeps its ``task_id`` and ``context`` (merged with the
        optional release ``context``) so the hand-off is never lost.  Passing an
        ``agent`` that does not own the lease raises ``ControlDeniedError``;
        releasing when nobody owns control is a no-op.
        """
        with self._lock:
            self._maybe_expire_locked()
            current = self._lease
            if current is None or not current.active:
                self._lease = None
                self._owner = ControlOwner.NONE
                return None
            if agent is not None:
                role = AgentRole.coerce(agent)
                if not self._role_owns_locked(role, current):
                    raise self._deny_locked(
                        role,
                        owner=current.owner.value,
                        operation="release",
                        reason="cannot release control owned by another agent",
                    )
            return self._release_current_locked(reason=reason, context=context)

    def preempt(
        self,
        new_owner: AgentRole | str,
        *,
        reason: str = "",
    ) -> tuple[ControlLease | None, ControlLease]:
        """Force-revoke the current owner and grant control to ``new_owner``.

        Returns ``(revoked_old_lease_or_None, new_lease)``.  The old lease is
        marked ``revoked`` (its context is preserved) and can never operate again.
        """
        role = AgentRole.coerce(new_owner)
        owner = ControlOwner.from_role(role)
        with self._lock:
            self._maybe_expire_locked()
            old = self._revoke_current_locked(
                reason=reason or "preempted", record_kind="preempted"
            )
            if old is not None:
                self._preemptions += 1
                self._publish_control(
                    EventType.CONTROL_PREEMPTED,
                    source=AgentRole.SYSTEM,
                    payload={
                        "previous_owner": old.owner.value,
                        "owner": owner.value,
                        "reason": str(reason or ""),
                    },
                )
            new = self._grant_locked(role, owner, reason=reason or "preempted")
            return old, new

    def release_all(self, reason: str = "") -> None:
        """Emergency release used when an agent crashes (never deadlocks)."""
        with self._lock:
            old = self._revoke_current_locked(
                reason=reason or "release_all", record_kind="released"
            )
            self._lease = None
            self._owner = ControlOwner.NONE
            for proxy in self._proxies.values():
                proxy.unbind()
            if old is not None:
                self._releases += 1
                self._publish_control(
                    EventType.CONTROL_RELEASED,
                    source=old.agent,
                    payload={
                        "owner": old.owner.value,
                        "lease_id": old.lease_id,
                        "reason": str(reason or "release_all"),
                    },
                )

    # ------------------------------------------------------------------ #
    # System ownership (checkpoints / rollback / viewer)
    # ------------------------------------------------------------------ #
    def enter_system(self, reason: str = "", *, preempt: bool = False) -> ControlLease:
        """Take ``OWNER_SYSTEM`` control (the only owner allowed to touch state).

        Denied when another owner holds the emulator unless ``preempt=True``.
        Entering while the system already owns control simply renews it.
        """
        with self._lock:
            self._maybe_expire_locked()
            current = self._lease
            if current is not None and current.active:
                if current.owner is ControlOwner.SYSTEM:
                    return self._regrant_locked(
                        AgentRole.SYSTEM,
                        current,
                        task_id="",
                        reason=reason or current.reason,
                        context=None,
                    )
                if not preempt:
                    raise self._deny_locked(
                        AgentRole.SYSTEM,
                        owner=current.owner.value,
                        operation="enter_system",
                        reason="system control requires a free emulator",
                    )
                old = self._revoke_current_locked(
                    reason=reason or "system takeover", record_kind="preempted"
                )
                if old is not None:
                    self._preemptions += 1
                    self._publish_control(
                        EventType.CONTROL_PREEMPTED,
                        source=AgentRole.SYSTEM,
                        payload={
                            "previous_owner": old.owner.value,
                            "owner": ControlOwner.SYSTEM.value,
                            "reason": str(reason or "system takeover"),
                        },
                    )
            return self._grant_locked(
                AgentRole.SYSTEM,
                ControlOwner.SYSTEM,
                reason=reason or "system",
            )

    def exit_system(self, reason: str = "") -> None:
        """Release ``OWNER_SYSTEM`` control; a no-op when the system is not owner."""
        with self._lock:
            current = self._lease
            if current is None or not current.active:
                return
            if current.owner is not ControlOwner.SYSTEM:
                return
            self._release_current_locked(reason=reason or "system exit")

    # ------------------------------------------------------------------ #
    # Lease expiry
    # ------------------------------------------------------------------ #
    def expire_if_stale(self) -> bool:
        """Explicitly expire a timed-out lease; returns True when one was expired."""
        with self._lock:
            return self._expire_locked()

    # ------------------------------------------------------------------ #
    # Proxies and diagnostics
    # ------------------------------------------------------------------ #
    def proxy(self, agent: AgentRole | str) -> GuardedEmulator:
        """Return the stable :class:`GuardedEmulator` for ``agent`` (cached)."""
        role = AgentRole.coerce(agent)
        with self._lock:
            proxy = self._proxies.get(role.value)
            if proxy is None:
                proxy = GuardedEmulator(self, role)
                self._proxies[role.value] = proxy
            self._maybe_expire_locked()
            current = self._lease
            if (
                current is not None
                and current.active
                and self._role_owns_locked(role, current)
                and proxy.lease is not current
            ):
                proxy.bind_lease(current)
            return proxy

    @property
    def denied_count(self) -> int:
        """Total number of denied control operations."""
        return self._denied_count

    def status(self) -> dict[str, Any]:
        """Machine readable snapshot (for the director loop / acceptance logs)."""
        with self._lock:
            self._maybe_expire_locked()
            lease = self._lease
            now = float(self._now())
            timeout = float(self._config.lease_timeout_s)
            age = max(0.0, now - float(lease.granted_at)) if lease is not None else 0.0
            remaining = (
                max(0.0, timeout - age) if lease is not None and timeout > 0 else 0.0
            )
            max_denied = max(0, int(self._config.max_denied))
            return {
                "owner": self._owner.value,
                "lease_id": lease.lease_id if lease is not None else "",
                "agent": lease.agent.value if lease is not None else "",
                "task_id": lease.task_id if lease is not None else "",
                "reason": lease.reason if lease is not None else "",
                "context": dict(lease.context) if lease is not None else {},
                "generation": int(self._generation),
                "granted_at": float(lease.granted_at) if lease is not None else 0.0,
                "age_s": round(age, 3),
                "remaining_s": round(remaining, 3),
                "lease_timeout_s": timeout,
                "auto_expire": bool(self._config.auto_expire),
                "last_released": (
                    self._last_released.to_dict() if self._last_released is not None else None
                ),
                "denied": int(self._denied_count),
                "max_denied": max_denied,
                "denied_limit_reached": bool(max_denied > 0 and self._denied_count >= max_denied),
                "proxy_denied": {
                    key: proxy.denied_count for key, proxy in self._proxies.items()
                },
                "grants": int(self._grants),
                "releases": int(self._releases),
                "preemptions": int(self._preemptions),
                "expiries": int(self._expiries),
                "event_errors": int(self._event_errors),
                "history_len": len(self._history),
            }

    def history(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent audit records in chronological order (newest last)."""
        with self._lock:
            count = max(0, int(limit))
            if count <= 0:
                return []
            return [dict(entry) for entry in list(self._history)[-count:]]

    # ------------------------------------------------------------------ #
    # Internals: grants / revocations
    # ------------------------------------------------------------------ #
    def _grant_locked(
        self,
        role: AgentRole,
        owner: ControlOwner,
        *,
        task_id: str = "",
        reason: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> ControlLease:
        self._generation += 1
        lease = ControlLease.create(
            owner=owner,
            agent=role,
            task_id=task_id,
            reason=reason,
            generation=self._generation,
            context=context,
            granted_at=float(self._now()),
        )
        self._lease = lease
        self._owner = owner
        self._grants += 1
        for proxy in self._proxies.values():
            if self._role_owns_locked(proxy.agent, lease):
                proxy.bind_lease(lease)
        self._record("granted", lease=lease, reason=reason)
        self._publish_control(
            EventType.CONTROL_GRANTED,
            source=role,
            payload={
                "owner": owner.value,
                "lease_id": lease.lease_id,
                "reason": str(reason or ""),
                "task_id": str(task_id or ""),
            },
        )
        return lease

    def _regrant_locked(
        self,
        role: AgentRole,
        current: ControlLease,
        *,
        task_id: str = "",
        reason: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> ControlLease:
        """Renew an existing owner's control without a hand-off."""
        merged_context = dict(current.context)
        if context:
            merged_context.update(dict(context))
        merged_task_id = str(task_id or current.task_id)
        merged_reason = str(reason or current.reason)
        current.revoked = True
        current.released_at = float(self._now())
        self._lease = None
        self._owner = ControlOwner.NONE
        self._unbind_matching_locked(current)
        self._record("renewed", lease=current, reason=merged_reason)
        return self._grant_locked(
            role,
            ControlOwner.from_role(role),
            task_id=merged_task_id,
            reason=merged_reason,
            context=merged_context,
        )

    def _release_current_locked(
        self,
        *,
        reason: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> ControlLease | None:
        current = self._lease
        if current is None:
            return None
        if context:
            current.context.update(dict(context))
        current.released_at = float(self._now())
        if reason:
            current.reason = str(reason)
        self._lease = None
        self._owner = ControlOwner.NONE
        self._last_released = current
        self._unbind_matching_locked(current)
        self._releases += 1
        self._record("released", lease=current, reason=reason)
        self._publish_control(
            EventType.CONTROL_RELEASED,
            source=current.agent,
            payload={
                "owner": current.owner.value,
                "lease_id": current.lease_id,
                "reason": str(reason or ""),
            },
        )
        return current

    def _revoke_current_locked(
        self,
        *,
        reason: str = "",
        record_kind: str = "preempted",
    ) -> ControlLease | None:
        current = self._lease
        if current is None:
            return None
        current.revoked = True
        current.released_at = float(self._now())
        self._lease = None
        self._owner = ControlOwner.NONE
        self._last_released = current
        self._unbind_matching_locked(current)
        self._record(record_kind, lease=current, reason=reason)
        return current

    def _expire_locked(self) -> bool:
        current = self._lease
        if current is None or not current.active:
            return False
        timeout = float(self._config.lease_timeout_s)
        if timeout <= 0.0:
            return False
        age = float(self._now()) - float(current.granted_at)
        if age < timeout:
            return False
        expired = self._revoke_current_locked(
            reason=f"lease expired after {age:.1f}s", record_kind="expired"
        )
        self._expiries += 1
        if expired is not None:
            self._publish_control(
                EventType.CONTROL_RELEASED,
                source=expired.agent,
                payload={
                    "owner": expired.owner.value,
                    "lease_id": expired.lease_id,
                    "reason": "lease expired",
                },
            )
        return True

    def _maybe_expire_locked(self) -> None:
        if self._config.auto_expire:
            self._expire_locked()

    # ------------------------------------------------------------------ #
    # Internals: proxies / helpers
    # ------------------------------------------------------------------ #
    def _unbind_matching_locked(self, lease: ControlLease) -> None:
        for proxy in self._proxies.values():
            if proxy.lease is lease:
                proxy.unbind()

    def _rebind_locked(self, old: ControlLease, new: ControlLease) -> None:
        for proxy in self._proxies.values():
            if proxy.lease is old:
                proxy.bind_lease(new)

    @staticmethod
    def _role_owns_locked(role: AgentRole, lease: ControlLease) -> bool:
        if lease.agent is role:
            return True
        return ControlOwner.from_role(role) is lease.owner

    def _now(self) -> float:
        return float(self._clock())

    # ------------------------------------------------------------------ #
    # Internals: audit trail and bus events
    # ------------------------------------------------------------------ #
    def _record(
        self,
        kind: str,
        *,
        lease: ControlLease | None = None,
        agent: AgentRole | None = None,
        owner: str = "",
        reason: str = "",
        operation: str = "",
    ) -> None:
        target_agent = agent if agent is not None else (lease.agent if lease is not None else None)
        entry: dict[str, Any] = {
            "kind": str(kind),
            "at": float(self._now()),
            "owner": str(owner or (lease.owner.value if lease is not None else "")),
            "agent": target_agent.value if isinstance(target_agent, AgentRole) else "",
            "lease_id": lease.lease_id if lease is not None else "",
            "task_id": lease.task_id if lease is not None else "",
            "reason": str(reason or ""),
            "operation": str(operation or ""),
            "generation": int(self._generation),
            "context": dict(lease.context) if lease is not None else {},
        }
        self._history.append(entry)

    def _deny_locked(
        self,
        role: AgentRole | str,
        *,
        owner: str,
        reason: str,
        operation: str = "",
        task_id: str = "",
    ) -> ControlDeniedError:
        """Record a denied control operation and return the error to raise."""
        role_enum = AgentRole.coerce(role)
        self._denied_count += 1
        self._record(
            "denied",
            agent=role_enum,
            owner=owner or ControlOwner.NONE.value,
            reason=reason,
            operation=operation,
        )
        payload: dict[str, Any] = {
            "owner": owner or ControlOwner.NONE.value,
            "reason": reason or "control denied",
        }
        if task_id:
            payload["task_id"] = task_id
        self._publish_control(EventType.CONTROL_DENIED, source=role_enum, payload=payload)
        return ControlDeniedError(
            f"{operation or 'control'} denied for {role_enum.value}: {reason}",
            agent=role_enum.value,
            owner=owner or ControlOwner.NONE.value,
            details={"operation": operation, "reason": reason},
        )

    def _publish_control(
        self,
        event_type: EventType,
        *,
        source: AgentRole,
        payload: Mapping[str, Any],
    ) -> None:
        """Publish one control event; a bus failure must never break control."""
        bus = self._bus
        if bus is None:
            return
        try:
            message = AgentMessage.create(
                event_type,
                source_agent=source,
                target_agent=AgentRole.DIRECTOR,
                payload=dict(payload),
            )
            bus.publish(message, dedupe=False)
        except Exception:  # noqa: BLE001 - auditing is best effort, control is not
            self._event_errors += 1


__all__ = ["ArbiterConfig", "ControlArbiter", "GuardedEmulator"]
