"""Shared fakes and synthetic frames for the story-agent tests.

Nothing here needs the real ROM, the real emulator, the network or an API key.
Frames are built from numpy RGBA arrays (same 4-level DMG palette the real
observation module expects); fakes implement the documented duck-typed
interfaces (GuardedEmulator surface, ModelRouter, ControlArbiter, EventBus,
SharedWorldMemory).
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from llm_agent.config import MemoryConfig
from llm_agent.contracts import (
    ActionOutcome,
    ActionStep,
    ApiUsage,
    Frame,
    ModelReply,
    ScreenEvent,
    ToolCall,
)
from llm_agent.memory import AgentMemory
from llm_agent.multi_agent.contracts import (
    AgentRole,
    ControlDeniedError,
    KnowledgeEntry,
    VerificationStatus,
)
from llm_agent.observation import frame_from_rgba

ROOT = Path(__file__).resolve().parents[2]
SCREEN_W, SCREEN_H = 160, 144
LEVELS = np.array([0, 85, 153, 255], dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Synthetic frames (deterministic; classified by the real observation module)
# --------------------------------------------------------------------------- #

def _base_luma(seed: int = 0) -> np.ndarray:
    """Deterministic 4-level overworld-looking background (no dialogue box)."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, 4, size=(SCREEN_H, SCREEN_W), dtype=np.uint8)


def _rgba_from_luma(luma: np.ndarray) -> np.ndarray:
    levels = np.asarray(luma, dtype=np.uint8)
    rgba = np.empty((SCREEN_H, SCREEN_W, 4), dtype=np.uint8)
    rgba[..., :3] = LEVELS[np.clip(levels, 0, 3)][..., None]
    rgba[..., 3] = 255
    return rgba


def make_frame(rgba: np.ndarray, frame_index: int = 0) -> Frame:
    """Build a contracts.Frame from raw RGBA pixels."""
    return frame_from_rgba(int(frame_index), np.ascontiguousarray(rgba, dtype=np.uint8))


def overworld_rgba(seed: int = 0) -> np.ndarray:
    """Normal gameplay frame: no text box, no menu, no battle."""
    return _rgba_from_luma(_base_luma(seed))


def dialog_rgba(text_seed: int = 0) -> np.ndarray:
    """Bottom text box with a full-width dark border and pseudo-text ink."""
    luma = _base_luma(0)
    luma[90:134, 8:152] = 3
    rng = np.random.default_rng(1000 + int(text_seed))
    for row in range(96, 114):
        for _ in range(3):
            start = int(12 + rng.integers(0, 126))
            width = int(rng.integers(3, 8))
            luma[row, start : min(start + width, 150)] = 0
    for row in (88, 89, 134, 135):
        luma[row, :] = 0
    luma[88:136, 4:8] = 0
    luma[88:136, 152:156] = 0
    return _rgba_from_luma(luma)


def menu_rgba() -> np.ndarray:
    """White menu panel on the right half with dark ink (ScreenEvent.MENU)."""
    luma = _base_luma(1)
    luma[38:122, 74:160] = 3
    for row in (38, 39, 40, 41, 118, 119, 120, 121):
        luma[row, 74:160] = 0
    for column in (74, 75, 76, 77, 78, 79, 80, 81):
        luma[38:122, column] = 0
    for row in range(48, 112, 6):
        luma[row, 90:145] = 0
    return _rgba_from_luma(luma)


def battle_rgba() -> np.ndarray:
    """Bright battle-like frame with a bordered bottom text box."""
    luma = np.full((SCREEN_H, SCREEN_W), 3, dtype=np.uint8)
    for row in (88, 89, 134, 135):
        luma[row, :] = 0
    luma[120:132, 20:80] = 0
    luma[40:70, 30:70] = 2
    return _rgba_from_luma(luma)


# --------------------------------------------------------------------------- #
# Fake emulator (GuardedEmulator surface)
# --------------------------------------------------------------------------- #

class FakeEmulator:
    """Deterministic scripted emulator used by the story-agent tests.

    ``initial_screen`` is one of ``overworld`` / ``dialog`` / ``menu`` / ``battle``.
    ``dialog_after_presses`` / ``menu_after_presses`` switch an overworld screen
    to a dialogue/menu when the cumulative press count reaches the value.
    """

    def __init__(
        self,
        *,
        initial_screen: str = "dialog",
        dialog_lines: int = 3,
        dialog_after_presses: int | None = None,
        menu_after_presses: int | None = None,
        menu_close_presses: int = 1,
        raise_control_denied: bool = False,
    ) -> None:
        self.screen = str(initial_screen)
        self.dialog_lines = max(1, int(dialog_lines))
        self.dialog_after_presses = dialog_after_presses
        self.menu_after_presses = menu_after_presses
        self.menu_close_presses = max(1, int(menu_close_presses))
        self.raise_control_denied = bool(raise_control_denied)
        self._line = 0
        self._menu_b = 0
        self._frame_count = 0
        #: quoted evidence: every (button, frames) in order
        self.presses: list[tuple[str, int | None]] = []
        self.batch_calls: list[list[str]] = []
        self._rgba = self._render()

    # -- rendering ------------------------------------------------------- #
    def _render(self) -> np.ndarray:
        if self.screen == "dialog":
            return dialog_rgba(self._line)
        if self.screen == "menu":
            return menu_rgba()
        if self.screen == "battle":
            return battle_rgba()
        return overworld_rgba(0)

    def _frame(self) -> Frame:
        return make_frame(self._rgba, self._frame_count)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_screen(self) -> str:
        return self.screen

    # -- emulator surface ------------------------------------------------ #
    def observe(self) -> Frame:
        self._frame_count += 1
        return self._frame()

    def wait(self, frames: int = 1) -> Frame:
        self._frame_count += max(0, int(frames))
        return self._frame()

    def press(self, button: str, frames: int | None = None) -> Frame:
        if self.raise_control_denied:
            raise ControlDeniedError(
                "press denied by guard proxy", agent="story", owner="navigation"
            )
        name = str(button).lower()
        self.presses.append((name, frames))
        self._frame_count += int(frames if frames is not None else 24)
        total = len(self.presses)
        if self.screen == "overworld":
            if (
                self.dialog_after_presses is not None
                and total >= self.dialog_after_presses
            ):
                self.screen = "dialog"
                self._line = 0
            elif self.menu_after_presses is not None and total >= self.menu_after_presses:
                self.screen = "menu"
                self._menu_b = 0
        elif self.screen == "dialog":
            if name == "b":
                self.screen = "overworld"
                self._line = 0
            elif name == "a":
                self._line += 1
                if self._line >= self.dialog_lines:
                    self.screen = "overworld"
                    self._line = 0
        elif self.screen == "menu":
            if name == "b":
                self._menu_b += 1
                if self._menu_b >= self.menu_close_presses:
                    self.screen = "overworld"
        self._rgba = self._render()
        return self._frame()

    def run_batch(self, buttons: Sequence[str], stop_on_event: bool = True) -> ActionOutcome:
        self.batch_calls.append([str(item) for item in buttons])
        steps = []
        for button in buttons:
            frame = self.press(button)
            steps.append(
                ActionStep(button=str(button), frame=frame, screen_changed=True, changed_ratio=1.0)
            )
        return ActionOutcome(
            buttons=[str(item) for item in buttons],
            steps=steps,
            start_frame=0,
            end_frame=self._frame_count,
        )


# --------------------------------------------------------------------------- #
# Fake arbiter / bus / shared memory / router
# --------------------------------------------------------------------------- #

class FakeArbiter:
    """Minimal ControlArbiter surface: ``holds(AgentRole.STORY)``."""

    def __init__(self, held: bool = True) -> None:
        self.held = bool(held)
        self.owner = "story" if self.held else "navigation"
        self.holds_calls = 0

    def holds(self, role: Any) -> bool:
        self.holds_calls += 1
        return self.held and AgentRole.coerce(role) is AgentRole.STORY


class FakeBus:
    """Minimal EventBus surface: ``publish(AgentMessage)``."""

    def __init__(self) -> None:
        self.messages: list[Any] = []
        self.publish_errors: list[str] = []

    def publish(self, message: Any) -> int:
        self.messages.append(message)
        return len(self.messages)

    @property
    def event_types(self) -> list[str]:
        out = []
        for message in self.messages:
            event_type = getattr(message, "event_type", None)
            out.append(str(getattr(event_type, "value", event_type)))
        return out

    def payloads(self, event_type: str) -> list[dict[str, Any]]:
        return [
            dict(message.payload)
            for message in self.messages
            if str(getattr(getattr(message, "event_type", None), "value", "")) == event_type
        ]


class FakeSharedMemory:
    """Minimal frozen SharedWorldMemory surface: ``remember`` / ``search``."""

    def __init__(self, branch_id: str = "branch_test") -> None:
        self.branch_id = branch_id
        self.entries: list[KnowledgeEntry] = []

    def remember(
        self,
        category: str,
        content: str,
        *,
        source_agent: Any = AgentRole.STORY,
        confidence: float = 0.5,
        verification_status: Any = VerificationStatus.HYPOTHESIS,
        evidence_refs: Any = (),
        subject: str = "",
        tags: Any = (),
        branch_id: str = "",
    ) -> KnowledgeEntry:
        entry = KnowledgeEntry.create(
            category=category,
            content=content,
            source_agent=source_agent,
            confidence=float(confidence),
            verification_status=verification_status,
            evidence_refs=tuple(evidence_refs or ()),
            subject=subject,
            tags=tuple(tags or ()),
            branch_id=branch_id or self.branch_id,
        )
        self.entries.append(entry)
        return entry

    def search(
        self,
        query: str = "",
        *,
        categories: Any = None,
        subject: str = "",
        limit: int = 12,
        include_historical: bool = False,
        verified_only: bool = False,
    ) -> list[KnowledgeEntry]:
        out: list[KnowledgeEntry] = []
        for entry in self.entries:
            if verified_only and entry.verification_status is not VerificationStatus.VERIFIED:
                continue
            if categories and entry.category not in set(categories):
                continue
            if subject and entry.subject != subject:
                continue
            if query and str(query).lower() not in entry.content.lower():
                continue
            out.append(entry)
        return out[: max(1, int(limit))]

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self.entries), "branch_id": self.branch_id}


def scripted_reply(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    content: str = "scripted decision",
    usage: ApiUsage | None = None,
) -> ModelReply:
    """Build one scripted ModelReply containing exactly one tool call."""
    return ModelReply(
        content=str(content),
        tool_calls=[
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=str(tool_name),
                arguments=dict(arguments or {}),
            )
        ],
        usage=usage
        or ApiUsage(
            prompt_tokens=120,
            completion_tokens=20,
            total_tokens=140,
            cache_hit_tokens=0,
            cache_miss_tokens=120,
        ),
        model="fake-story-model",
        latency_s=0.01,
        finish_reason="tool_calls",
    )


def multi_call_reply(
    *calls: tuple[str, dict[str, Any] | None] | ToolCall,
    content: str = "scripted multi-call decision",
) -> ModelReply:
    """Build one ModelReply declaring SEVERAL tool calls (a protocol stress test).

    The story loop executes only one tool per turn, so it must trim the kept
    assistant message down to the executed call; otherwise the chat API rejects
    the next request for unanswered ``tool_calls``.
    """
    tool_calls: list[ToolCall] = []
    for index, item in enumerate(calls):
        if isinstance(item, ToolCall):
            tool_calls.append(item)
            continue
        name, arguments = item
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}_{index}",
                name=str(name),
                arguments=dict(arguments or {}),
            )
        )
    return ModelReply(
        content=content,
        tool_calls=tool_calls,
        usage=ApiUsage(
            prompt_tokens=140,
            completion_tokens=30,
            total_tokens=170,
            cache_hit_tokens=0,
            cache_miss_tokens=140,
        ),
        model="fake-story-model",
        latency_s=0.01,
        finish_reason="tool_calls",
    )


class FakeRouter:
    """Scripted duck-typed ModelRouter (no network, no API key)."""

    def __init__(
        self,
        replies: Sequence[Any] | None = None,
        *,
        can_call_result: tuple[bool, str] = (True, ""),
        model: str = "fake-story-model",
    ) -> None:
        self._replies: list[Any] = list(replies or [])
        self.can_call_result = can_call_result
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self.can_call_roles: list[Any] = []

    @property
    def complete_calls(self) -> int:
        return len(self.calls)

    def model_for(self, role: Any) -> str:
        assert AgentRole.coerce(role) is AgentRole.STORY
        return self.model

    def can_call(self, role: Any) -> tuple[bool, str]:
        self.can_call_roles.append(role)
        return self.can_call_result

    def complete(
        self,
        role: Any,
        messages: Any,
        *,
        tools: Any = None,
        max_tokens: Any = None,
        thinking: Any = None,
    ) -> ModelReply:
        assert AgentRole.coerce(role) is AgentRole.STORY
        self.calls.append(
            {
                "messages": copy.deepcopy(list(messages)),
                "tools": tools,
                "max_tokens": max_tokens,
                "thinking": thinking,
            }
        )
        item: Any
        if self._replies:
            item = self._replies.pop(0)
        else:
            item = scripted_reply("observe")
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(messages)
        if not isinstance(item, ModelReply):
            raise AssertionError(f"scripted reply must be a ModelReply, got {type(item)!r}")
        return item


# --------------------------------------------------------------------------- #
# Agent factory
# --------------------------------------------------------------------------- #

class StoryStack:
    """Convenience bundle returned by the ``make_agent`` fixture."""

    def __init__(
        self,
        *,
        agent: Any,
        router: FakeRouter,
        emulator: FakeEmulator,
        arbiter: FakeArbiter,
        bus: FakeBus | None,
        shared: FakeSharedMemory | None,
        memory: AgentMemory,
        screen_dir: Path,
    ) -> None:
        self.agent = agent
        self.router = router
        self.emulator = emulator
        self.arbiter = arbiter
        self.bus = bus
        self.shared = shared
        self.memory = memory
        self.screen_dir = screen_dir


def build_agent(
    tmp_path: Path,
    *,
    replies: Sequence[Any] | None = None,
    emulator: FakeEmulator | None = None,
    arbiter: FakeArbiter | None = None,
    bus: FakeBus | None = None,
    shared: FakeSharedMemory | None = None,
    can_call_result: tuple[bool, str] = (True, ""),
    config: Any = None,
    screen_dir: Path | None = None,
    with_bus: bool = False,
    with_shared: bool = False,
) -> StoryStack:
    """Create a consistent StoryAgent + fakes for one test."""
    from llm_agent.story.agent import StoryAgent

    memory_root = Path(tmp_path) / "agent_memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    memory = AgentMemory(
        replace(MemoryConfig(root=str(memory_root)), max_context_chars=2000),
        run_dir=Path(tmp_path) / "run",
    )
    router = FakeRouter(replies, can_call_result=can_call_result)
    emulator = emulator if emulator is not None else FakeEmulator()
    arbiter = arbiter if arbiter is not None else FakeArbiter()
    if with_bus and bus is None:
        bus = FakeBus()
    if with_shared and shared is None:
        shared = FakeSharedMemory()
    screens = Path(screen_dir) if screen_dir is not None else Path(tmp_path) / "screens"
    agent = StoryAgent(
        router=router,
        bus=bus,
        arbiter=arbiter,
        shared_memory=shared,
        memory=memory,
        config=config,
        screen_dir=screens,
        clock=lambda: 1_700_000_000.0,
    )
    return StoryStack(
        agent=agent,
        router=router,
        emulator=emulator,
        arbiter=arbiter,
        bus=bus,
        shared=shared,
        memory=memory,
        screen_dir=screens,
    )


# --------------------------------------------------------------------------- #
# pytest fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def make_agent(tmp_path: Path):
    """Factory fixture: ``make_agent(replies=[...], emulator=...)``."""

    def factory(**kwargs: Any) -> StoryStack:
        return build_agent(tmp_path, **kwargs)

    return factory
