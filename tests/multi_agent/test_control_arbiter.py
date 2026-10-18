"""Tests for :class:`ControlArbiter` and :class:`GuardedEmulator`.

Everything runs against a small recording fake emulator, so no PyBoy, ROM, save
state or API key is required.  The suite covers:

* exactly one owner at any time (contention -> exactly one grant),
* lease generations / stale-token rejection / context preservation,
* ``preempt``/``release_all`` as crash escape hatches (no permanent deadlock),
* the guarded proxy: read-only calls always work, mutating calls need the lease,
  state/checkpoint calls need ``OWNER_SYSTEM``, denied calls never touch the
  emulator,
* optional CONTROL_* audit events on an :class:`EventBus`, and that a broken bus
  can never break a control change.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from llm_agent.multi_agent.contracts import (
    AgentRole,
    ControlDeniedError,
    ControlOwner,
    EventType,
    ProtocolError,
)
from llm_agent.multi_agent.control_arbiter import (
    ArbiterConfig,
    ControlArbiter,
    GuardedEmulator,
)
from llm_agent.multi_agent.event_bus import EventBus, EventBusConfig


# --------------------------------------------------------------------------- #
# Fake emulator
# --------------------------------------------------------------------------- #

class FakeEmulator:
    """Recording stand-in for :class:`llm_agent.emulator.GameBoyEmulator`."""

    def __init__(self, frames: int = 24) -> None:
        self.frames = frames
        self.presses: list[tuple[str, int | None]] = []
        self.waits: list[int] = []
        self.batches: list[list[str]] = []
        self.load_state_calls: list[object] = []
        self.save_state_bytes_calls = 0
        self.save_state_file_calls: list[object] = []
        self.frame_hooks: list[object] = []
        self.snapshots: list[object] = []
        self.frame_count = 0
        self.started = True
        self.last_frame = SimpleNamespace(frame_index=0)
        self.loaded_state: str | None = None
        self.fail_press = False
        self.observe_calls = 0

    # -- read-only ------------------------------------------------------- #
    @property
    def streaming(self) -> bool:
        return bool(self.frame_hooks)

    def observe(self) -> SimpleNamespace:
        self.observe_calls += 1
        return SimpleNamespace(frame_index=self.frame_count)

    # -- mutating -------------------------------------------------------- #
    def press(self, button: str, *, frames: int | None = None) -> SimpleNamespace:
        if self.fail_press:
            raise RuntimeError("emulator exploded")
        self.presses.append((button, frames))
        self.frame_count += int(frames if frames is not None else self.frames)
        return SimpleNamespace(frame_index=self.frame_count)

    def wait(self, frames: int) -> SimpleNamespace:
        self.waits.append(int(frames))
        self.frame_count += int(frames)
        return SimpleNamespace(frame_index=self.frame_count)

    def run_batch(
        self,
        buttons: list[str],
        *,
        stop_on_event: bool = True,
        step_hook=None,
    ) -> SimpleNamespace:
        self.batches.append(list(buttons))
        if step_hook is not None:
            step_hook("step")
        return SimpleNamespace(
            buttons=list(buttons),
            steps=[],
            start_frame=self.frame_count,
            end_frame=self.frame_count,
            stopped_reason="completed",
            events=[],
            summary="fake",
        )

    def load_state(self, path: object) -> None:
        self.load_state_calls.append(path)
        self.loaded_state = str(path)

    def save_state_bytes(self) -> bytes:
        self.save_state_bytes_calls += 1
        return b"state-bytes"

    def save_state_file(self, path: object) -> object:
        self.save_state_file_calls.append(path)
        return path

    def set_frame_hook(self, hook, *, fps: float = 60.0) -> None:
        self.frame_hooks.append({"hook": hook, "fps": fps})

    def clear_frame_hook(self) -> None:
        self.frame_hooks.clear()

    def snapshot_png(self, path: object) -> object:
        self.snapshots.append(path)
        return path

    # -- test helper ----------------------------------------------------- #
    @property
    def write_count(self) -> int:
        """Number of calls that could have mutated the game."""
        return (
            len(self.presses)
            + len(self.waits)
            + len(self.batches)
            + len(self.load_state_calls)
            + self.save_state_bytes_calls
            + len(self.save_state_file_calls)
            + len(self.frame_hooks)
            + len(self.snapshots)
        )


class BrokenBus:
    """Event bus double whose ``publish`` always fails."""

    def __init__(self) -> None:
        self.calls = 0

    def publish(self, message, *, dedupe: bool = True) -> None:
        self.calls += 1
        raise RuntimeError("event bus unavailable")


@pytest.fixture()
def emulator() -> FakeEmulator:
    return FakeEmulator()


@pytest.fixture()
def arbiter(emulator: FakeEmulator) -> ControlArbiter:
    return ControlArbiter(emulator, config=ArbiterConfig(lease_timeout_s=300.0))


# --------------------------------------------------------------------------- #
# Configuration / construction
# --------------------------------------------------------------------------- #

def test_config_defaults() -> None:
    config = ArbiterConfig()
    assert config.lease_timeout_s == 300.0
    assert config.auto_expire is True
    assert config.max_denied == 100


def test_constructor_requires_emulator() -> None:
    with pytest.raises(ValueError):
        ControlArbiter(None)


def test_constructor_rejects_bad_config(emulator: FakeEmulator) -> None:
    with pytest.raises(ProtocolError):
        ControlArbiter(emulator, config="not-a-config")  # type: ignore[arg-type]


def test_constructor_rejects_bad_bus(emulator: FakeEmulator) -> None:
    with pytest.raises(ProtocolError):
        ControlArbiter(emulator, event_bus=object())  # type: ignore[arg-type]


def test_initial_state_is_free(arbiter: ControlArbiter, emulator: FakeEmulator) -> None:
    assert arbiter.emulator is emulator
    assert arbiter.owner is ControlOwner.NONE
    assert arbiter.lease is None
    assert arbiter.holds(AgentRole.NAVIGATION) is False
    assert arbiter.history() == []
    status = arbiter.status()
    assert status["owner"] == ControlOwner.NONE.value
    assert status["lease_id"] == ""
    assert status["generation"] == 0


# --------------------------------------------------------------------------- #
# Single owner contention
# --------------------------------------------------------------------------- #

def test_two_requesters_yield_exactly_one_owner(arbiter: ControlArbiter) -> None:
    lean_lease = arbiter.request(
        AgentRole.NAVIGATION, task_id="task_nav", reason="walk", context={"goal": "door"}
    )
    assert lean_lease.owner is ControlOwner.NAVIGATION
    with pytest.raises(ControlDeniedError) as info:
        arbiter.request(AgentRole.STORY, task_id="task_story")
    assert info.value.agent == AgentRole.STORY.value
    assert info.value.owner == ControlOwner.NAVIGATION.value
    with pytest.raises(ControlDeniedError):
        arbiter.request(AgentRole.POKEMON)
    assert arbiter.owner is ControlOwner.NAVIGATION
    assert arbiter.lease is lean_lease
    assert arbiter.holds(AgentRole.NAVIGATION) is True
    assert arbiter.holds(AgentRole.STORY) is False


def test_generation_is_monotonic_across_handoffs(arbiter: ControlArbiter) -> None:
    first = arbiter.request(AgentRole.NAVIGATION)
    arbiter.release(AgentRole.NAVIGATION)
    second = arbiter.request(AgentRole.STORY)
    old, third = arbiter.preempt(AgentRole.POKEMON, reason="handoff")
    arbiter.release(AgentRole.POKEMON)
    fourth = arbiter.enter_system("checkpoint")
    assert first.generation < second.generation < third.generation < fourth.generation
    assert old is second


def test_request_same_agent_preserves_task_and_context(arbiter: ControlArbiter) -> None:
    first = arbiter.request(
        AgentRole.NAVIGATION,
        task_id="task_a",
        reason="initial",
        context={"target": "north", "attempt": 1},
    )
    second = arbiter.request(AgentRole.NAVIGATION)
    assert second is not first
    assert second.task_id == "task_a"
    assert second.context == {"target": "north", "attempt": 1}
    assert second.reason == "initial"
    assert second.generation > first.generation
    assert first.revoked is True
    assert first.released_at > 0
    assert arbiter.holds(AgentRole.NAVIGATION)


def test_request_same_agent_can_update_context(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION, context={"target": "north", "attempt": 1})
    renewed = arbiter.request(AgentRole.NAVIGATION, context={"attempt": 2})
    assert renewed.context == {"target": "north", "attempt": 2}


def test_director_request_maps_to_system_owner(arbiter: ControlArbiter) -> None:
    lease = arbiter.request(AgentRole.DIRECTOR, reason="coordinate")
    assert lease.owner is ControlOwner.SYSTEM
    assert lease.agent is AgentRole.DIRECTOR
    assert arbiter.owner is ControlOwner.SYSTEM
    assert arbiter.holds(AgentRole.DIRECTOR) is True
    assert arbiter.holds(AgentRole.SYSTEM) is True
    assert arbiter.holds(AgentRole.NAVIGATION) is False


# --------------------------------------------------------------------------- #
# require / release
# --------------------------------------------------------------------------- #

def test_require_denied_without_lease(arbiter: ControlArbiter) -> None:
    with pytest.raises(ControlDeniedError) as info:
        arbiter.require(AgentRole.NAVIGATION)
    assert "no agent" in str(info.value).lower()
    assert arbiter.status()["denied"] == 1


def test_require_returns_current_lease(arbiter: ControlArbiter) -> None:
    lease = arbiter.request(AgentRole.NAVIGATION)
    assert arbiter.require(AgentRole.NAVIGATION, operation="observe-move") is lease


def test_require_denied_for_other_agent(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    with pytest.raises(ControlDeniedError) as info:
        arbiter.require(AgentRole.STORY, operation="press")
    assert info.value.owner == ControlOwner.NAVIGATION.value


def test_release_preserves_context_task_and_reason(arbiter: ControlArbiter) -> None:
    arbiter.request(
        AgentRole.NAVIGATION,
        task_id="task_nav",
        reason="walk",
        context={"target": "north"},
    )
    released = arbiter.release(
        AgentRole.NAVIGATION, reason="reached", context={"outcome": "ok"}
    )
    assert released is not None
    assert released.task_id == "task_nav"
    assert released.context == {"target": "north", "outcome": "ok"}
    assert released.reason == "reached"
    assert released.released_at > 0
    assert released.revoked is False
    assert arbiter.owner is ControlOwner.NONE
    assert arbiter.lease is None


def test_release_without_owner_is_noop(arbiter: ControlArbiter) -> None:
    assert arbiter.release() is None
    assert arbiter.release(AgentRole.NAVIGATION) is None
    assert arbiter.owner is ControlOwner.NONE


def test_release_by_non_owner_is_denied(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    with pytest.raises(ControlDeniedError):
        arbiter.release(AgentRole.STORY)
    assert arbiter.owner is ControlOwner.NAVIGATION


def test_release_without_agent_argument_releases_current_owner(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.STORY)
    released = arbiter.release(reason="director clean-up")
    assert released is not None
    assert released.owner is ControlOwner.STORY
    assert arbiter.owner is ControlOwner.NONE


# --------------------------------------------------------------------------- #
# Preempt / release_all (no deadlock)
# --------------------------------------------------------------------------- #

def test_preempt_revokes_old_lease_and_grants_new_owner(arbiter: ControlArbiter) -> None:
    old_lease = arbiter.request(
        AgentRole.NAVIGATION, task_id="task_nav", context={"target": "north"}
    )
    old, new = arbiter.preempt(AgentRole.STORY, reason="battle interrupt")
    assert old is old_lease
    assert old.revoked is True
    assert old.released_at > 0
    assert old.context == {"target": "north"}  # context never lost
    assert new.owner is ControlOwner.STORY
    assert new.generation > old.generation
    assert arbiter.holds(AgentRole.STORY)
    assert not arbiter.holds(AgentRole.NAVIGATION)


def test_preempt_when_free_only_grants(arbiter: ControlArbiter) -> None:
    old, new = arbiter.preempt(AgentRole.POKEMON, reason="direct start")
    assert old is None
    assert new.owner is ControlOwner.POKEMON
    assert arbiter.holds(AgentRole.POKEMON)


def test_release_all_recovers_from_crashed_owner(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    arbiter.release_all("navigation crashed")
    assert arbiter.owner is ControlOwner.NONE
    assert arbiter.lease is None
    with pytest.raises(ControlDeniedError):
        proxy.press("up")
    # The director can grant control again immediately.
    story_lease = arbiter.request(AgentRole.STORY)
    assert story_lease.owner is ControlOwner.STORY
    arbiter.release_all()
    assert arbiter.owner is ControlOwner.NONE


def test_no_deadlock_after_emulator_exception(arbiter: ControlArbiter, emulator: FakeEmulator) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    emulator.fail_press = True
    with pytest.raises(RuntimeError):
        proxy.press("a")
    # The lock was released and the lease is still usable afterwards.
    assert arbiter.holds(AgentRole.NAVIGATION)
    emulator.fail_press = False
    released = arbiter.release(AgentRole.NAVIGATION)
    assert released is not None
    arbiter.request(AgentRole.STORY)
    arbiter.proxy(AgentRole.STORY).press("b")
    assert emulator.presses[-1] == ("b", None)


# --------------------------------------------------------------------------- #
# GuardedEmulator: read-only surface
# --------------------------------------------------------------------------- #

def test_proxy_is_cached_and_typed(arbiter: ControlArbiter) -> None:
    navigation = arbiter.proxy(AgentRole.NAVIGATION)
    assert isinstance(navigation, GuardedEmulator)
    assert arbiter.proxy(AgentRole.NAVIGATION) is navigation
    assert arbiter.proxy(AgentRole.STORY) is not navigation
    assert arbiter.proxy("nav").agent is AgentRole.NAVIGATION


def test_read_only_operations_are_always_allowed(arbiter: ControlArbiter, emulator: FakeEmulator) -> None:
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    assert proxy.observe().frame_index == 0
    assert proxy.frame_count == 0
    assert proxy.last_frame is emulator.last_frame
    assert proxy.started is True
    assert proxy.streaming is False
    assert proxy.lease is None
    assert emulator.write_count == 0


def test_proxy_does_not_expose_bypass_attributes(arbiter: ControlArbiter) -> None:
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    for attribute in ("_pyboy", "start", "close", "cfg", "emulator"):
        with pytest.raises(AttributeError):
            getattr(proxy, attribute)


# --------------------------------------------------------------------------- #
# GuardedEmulator: mutating surface
# --------------------------------------------------------------------------- #

def test_guarded_mutations_denied_without_lease(arbiter: ControlArbiter, emulator: FakeEmulator) -> None:
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    with pytest.raises(ControlDeniedError):
        proxy.press("a")
    with pytest.raises(ControlDeniedError):
        proxy.wait(10)
    with pytest.raises(ControlDeniedError):
        proxy.run_batch(["up", "a"])
    assert proxy.denied_count == 3
    assert emulator.write_count == 0
    assert emulator.frame_count == 0


def test_guarded_mutations_allowed_for_owner(arbiter: ControlArbiter, emulator: FakeEmulator) -> None:
    arbiter.request(AgentRole.NAVIGATION, task_id="task_nav")
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    proxy.press("a", frames=8)
    proxy.wait(5)
    proxy.run_batch(["up", "a"], stop_on_event=False)
    assert emulator.presses == [("a", 8)]
    assert emulator.waits == [5]
    assert emulator.batches == [["up", "a"]]
    assert proxy.operations == 3


def test_guarded_mutations_denied_for_non_owner(arbiter: ControlArbiter, emulator: FakeEmulator) -> None:
    arbiter.request(AgentRole.STORY)
    navigation = arbiter.proxy(AgentRole.NAVIGATION)
    with pytest.raises(ControlDeniedError) as info:
        navigation.press("up")
    assert info.value.owner == ControlOwner.STORY.value
    assert emulator.write_count == 0


def test_stale_lease_cannot_operate_after_preempt(arbiter: ControlArbiter, emulator: FakeEmulator) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    navigation = arbiter.proxy(AgentRole.NAVIGATION)
    navigation.press("up")
    assert len(emulator.presses) == 1

    arbiter.preempt(AgentRole.STORY, reason="handoff")
    with pytest.raises(ControlDeniedError) as info:
        navigation.press("down")
    assert info.value.agent == AgentRole.NAVIGATION.value
    assert len(emulator.presses) == 1  # denied call never reached the emulator

    story = arbiter.proxy(AgentRole.STORY)
    story.press("a")
    assert emulator.presses[-1] == ("a", None)


# --------------------------------------------------------------------------- #
# GuardedEmulator: state/checkpoint surface requires OWNER_SYSTEM
# --------------------------------------------------------------------------- #

def test_state_operations_denied_without_system_owner(
    arbiter: ControlArbiter, emulator: FakeEmulator
) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    with pytest.raises(ControlDeniedError):
        proxy.load_state("state.bin")
    with pytest.raises(ControlDeniedError):
        proxy.save_state_bytes()
    with pytest.raises(ControlDeniedError):
        proxy.save_state_file("out.bin")
    with pytest.raises(ControlDeniedError):
        proxy.set_frame_hook(lambda frame: None)
    with pytest.raises(ControlDeniedError):
        proxy.clear_frame_hook()
    with pytest.raises(ControlDeniedError):
        proxy.snapshot_png("shot.png")
    assert proxy.denied_count == 6
    assert emulator.write_count == 0


def test_state_operations_allowed_for_system_owner(
    arbiter: ControlArbiter, emulator: FakeEmulator
) -> None:
    arbiter.enter_system("checkpoint")
    system = arbiter.proxy(AgentRole.SYSTEM)
    system.load_state("state.bin")
    assert system.save_state_bytes() == b"state-bytes"
    system.save_state_file("out.bin")
    hook = lambda frame: None  # noqa: E731 - tiny test hook
    system.set_frame_hook(hook, fps=30.0)
    assert system.streaming is True
    system.clear_frame_hook()
    assert emulator.frame_hooks == []
    system.snapshot_png("shot.png")
    assert emulator.load_state_calls == ["state.bin"]
    assert emulator.save_state_bytes_calls == 1
    assert emulator.save_state_file_calls == ["out.bin"]
    assert emulator.snapshots == ["shot.png"]


def test_director_proxy_can_run_state_operations_under_system_owner(
    arbiter: ControlArbiter,
) -> None:
    arbiter.enter_system("checkpoint")
    director = arbiter.proxy(AgentRole.DIRECTOR)
    assert director.save_state_bytes() == b"state-bytes"


def test_professional_proxy_cannot_run_state_operations_under_system_owner(
    arbiter: ControlArbiter, emulator: FakeEmulator
) -> None:
    navigation = arbiter.proxy(AgentRole.NAVIGATION)
    arbiter.enter_system("checkpoint")
    with pytest.raises(ControlDeniedError):
        navigation.load_state("state.bin")
    with pytest.raises(ControlDeniedError):
        navigation.save_state_bytes()
    assert emulator.write_count == 0


def test_enter_system_denied_when_busy(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    with pytest.raises(ControlDeniedError):
        arbiter.enter_system("checkpoint")
    assert arbiter.owner is ControlOwner.NAVIGATION


def test_enter_system_with_preempt_takes_control(arbiter: ControlArbiter) -> None:
    old = arbiter.request(AgentRole.NAVIGATION)
    lease = arbiter.enter_system("rollback", preempt=True)
    assert lease.owner is ControlOwner.SYSTEM
    assert old.revoked is True
    assert arbiter.holds(AgentRole.SYSTEM)


def test_enter_system_twice_renews(arbiter: ControlArbiter) -> None:
    first = arbiter.enter_system("first")
    second = arbiter.enter_system("second")
    assert second.generation > first.generation
    assert arbiter.owner is ControlOwner.SYSTEM


def test_exit_system_releases_and_denies_state_operations(
    arbiter: ControlArbiter, emulator: FakeEmulator
) -> None:
    arbiter.enter_system("checkpoint")
    system = arbiter.proxy(AgentRole.SYSTEM)
    system.save_state_bytes()
    arbiter.exit_system("done")
    assert arbiter.owner is ControlOwner.NONE
    with pytest.raises(ControlDeniedError):
        system.save_state_bytes()
    assert emulator.save_state_bytes_calls == 1
    # Exiting when the system is not owner is a harmless no-op.
    arbiter.exit_system("already free")
    assert arbiter.owner is ControlOwner.NONE


# --------------------------------------------------------------------------- #
# Expiry / renew
# --------------------------------------------------------------------------- #

def test_lease_expiry_with_injected_clock(emulator: FakeEmulator) -> None:
    now = [100.0]
    arbiter = ControlArbiter(
        emulator,
        config=ArbiterConfig(lease_timeout_s=10.0, auto_expire=True),
        clock=lambda: now[0],
    )
    arbiter.request(AgentRole.NAVIGATION)
    now[0] = 105.0
    assert arbiter.holds(AgentRole.NAVIGATION) is True
    now[0] = 111.0
    assert arbiter.holds(AgentRole.NAVIGATION) is False
    assert arbiter.owner is ControlOwner.NONE
    assert arbiter.lease is None
    assert arbiter.expire_if_stale() is False
    assert arbiter.status()["expiries"] == 1


def test_expire_if_stale_explicit_even_when_auto_expire_off(emulator: FakeEmulator) -> None:
    now = [100.0]
    arbiter = ControlArbiter(
        emulator,
        config=ArbiterConfig(lease_timeout_s=10.0, auto_expire=False),
        clock=lambda: now[0],
    )
    arbiter.request(AgentRole.NAVIGATION)
    now[0] = 120.0
    # auto_expire=False: implicit checks keep the lease alive.
    assert arbiter.holds(AgentRole.NAVIGATION) is True
    assert arbiter.owner is ControlOwner.NAVIGATION
    assert arbiter.expire_if_stale() is True
    assert arbiter.owner is ControlOwner.NONE
    assert arbiter.expire_if_stale() is False


def test_guarded_press_denied_after_expiry(emulator: FakeEmulator) -> None:
    now = [100.0]
    arbiter = ControlArbiter(
        emulator,
        config=ArbiterConfig(lease_timeout_s=10.0, auto_expire=True),
        clock=lambda: now[0],
    )
    arbiter.request(AgentRole.NAVIGATION)
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    proxy.press("up")
    now[0] = 130.0
    with pytest.raises(ControlDeniedError):
        proxy.press("down")
    assert emulator.presses == [("up", None)]


def test_renew_extends_lease_and_rejects_stale_token(emulator: FakeEmulator) -> None:
    now = [100.0]
    arbiter = ControlArbiter(
        emulator,
        config=ArbiterConfig(lease_timeout_s=10.0),
        clock=lambda: now[0],
    )
    lease = arbiter.request(AgentRole.NAVIGATION)
    now[0] = 108.0
    renewed = arbiter.renew(lease)
    assert renewed.granted_at == pytest.approx(108.0)
    assert renewed.generation > lease.generation
    assert arbiter.holds(AgentRole.NAVIGATION) is True
    # The superseded token is stale and can never extend control again.
    with pytest.raises(ControlDeniedError):
        arbiter.renew(lease)
    # renew(None) keeps renewing the current lease.
    again = arbiter.renew()
    assert again.generation > renewed.generation


def test_renew_denied_without_lease(arbiter: ControlArbiter) -> None:
    with pytest.raises(ControlDeniedError):
        arbiter.renew()


# --------------------------------------------------------------------------- #
# Status / history / audit
# --------------------------------------------------------------------------- #

def test_status_reports_ownership_and_denials(arbiter: ControlArbiter) -> None:
    arbiter.request(
        AgentRole.NAVIGATION, task_id="task_nav", reason="walk", context={"target": "x"}
    )
    status = arbiter.status()
    assert status["owner"] == ControlOwner.NAVIGATION.value
    assert status["task_id"] == "task_nav"
    assert status["context"] == {"target": "x"}
    assert status["lease_timeout_s"] == 300.0
    assert status["denied"] == 0
    assert status["grants"] == 1
    assert status["denied_limit_reached"] is False
    with pytest.raises(ControlDeniedError):
        arbiter.proxy(AgentRole.STORY).press("a")  # denied
    assert arbiter.status()["denied"] == 1


def test_denied_limit_flag(emulator: FakeEmulator) -> None:
    arbiter = ControlArbiter(emulator, config=ArbiterConfig(max_denied=2))
    proxy = arbiter.proxy(AgentRole.NAVIGATION)
    for _ in range(2):
        with pytest.raises(ControlDeniedError):
            proxy.press("a")
    assert arbiter.status()["denied"] == 2
    assert arbiter.status()["denied_limit_reached"] is True


def test_history_records_kinds_and_limit(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    with pytest.raises(ControlDeniedError):
        arbiter.proxy(AgentRole.STORY).press("a")  # denied
    arbiter.release(AgentRole.NAVIGATION, reason="done")
    kinds = [entry["kind"] for entry in arbiter.history()]
    assert kinds == ["requested", "granted", "denied", "released"]
    assert arbiter.history(limit=1)[0]["kind"] == "released"
    assert arbiter.history(limit=0) == []
    assert arbiter.status()["history_len"] == 4


def test_control_events_published_on_optional_bus(emulator: FakeEmulator) -> None:
    bus = EventBus(EventBusConfig())
    arbiter = ControlArbiter(emulator, event_bus=bus)
    try:
        arbiter.request(AgentRole.NAVIGATION, task_id="task_nav")
        with pytest.raises(ControlDeniedError):
            arbiter.request(AgentRole.STORY)
        arbiter.preempt(AgentRole.STORY, reason="handoff")
        arbiter.release(AgentRole.STORY, reason="done")
        arbiter.enter_system("checkpoint")
        arbiter.exit_system("done")
        messages = bus.drain(AgentRole.DIRECTOR, max_messages=100)
        event_types = {message.event_type for message in messages}
        assert {
            EventType.CONTROL_REQUESTED,
            EventType.CONTROL_GRANTED,
            EventType.CONTROL_DENIED,
            EventType.CONTROL_PREEMPTED,
            EventType.CONTROL_RELEASED,
        }.issubset(event_types)
        assert arbiter.status()["event_errors"] == 0
        denied = [m for m in messages if m.event_type is EventType.CONTROL_DENIED][0]
        assert {"owner", "reason"}.issubset(denied.payload)
    finally:
        bus.close()


def test_broken_bus_never_breaks_control_changes(emulator: FakeEmulator) -> None:
    broken = BrokenBus()
    arbiter = ControlArbiter(emulator, event_bus=broken)
    lease = arbiter.request(AgentRole.NAVIGATION)
    assert lease.owner is ControlOwner.NAVIGATION
    arbiter.proxy(AgentRole.NAVIGATION).press("a")
    arbiter.preempt(AgentRole.STORY, reason="handoff")
    arbiter.release(AgentRole.STORY, reason="done")
    assert broken.calls > 0
    assert arbiter.status()["event_errors"] > 0
    assert arbiter.owner is ControlOwner.NONE
    assert emulator.presses == [("a", None)]


# --------------------------------------------------------------------------- #
# request(wait=True)
# --------------------------------------------------------------------------- #

def test_request_wait_times_out_and_raises(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    started = time.monotonic()
    with pytest.raises(ControlDeniedError):
        arbiter.request(AgentRole.STORY, wait=True, timeout_s=0.05)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert arbiter.owner is ControlOwner.NAVIGATION


def test_request_wait_succeeds_after_release(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION)

    def release_later() -> None:
        time.sleep(0.08)
        arbiter.release(AgentRole.NAVIGATION)

    thread = threading.Thread(target=release_later, daemon=True)
    thread.start()
    try:
        lease = arbiter.request(AgentRole.STORY, wait=True, timeout_s=3.0)
        assert lease.owner is ControlOwner.STORY
        assert arbiter.holds(AgentRole.STORY)
    finally:
        thread.join(timeout=5)
        arbiter.release_all("test cleanup")


def test_request_wait_with_zero_timeout_fails_fast(arbiter: ControlArbiter) -> None:
    arbiter.request(AgentRole.NAVIGATION)
    started = time.monotonic()
    with pytest.raises(ControlDeniedError):
        arbiter.request(AgentRole.STORY, wait=True, timeout_s=0.0)
    assert time.monotonic() - started < 1.0
