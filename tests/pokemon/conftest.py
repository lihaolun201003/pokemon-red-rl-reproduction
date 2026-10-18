"""Shared fixtures for the Pokémon (battle) agent tests.

Everything here is synthetic and fast: no PyBoy, no ROM, no network.

* :func:`battle_frame`, :func:`overworld_frame`, ... render real 160x144 RGBA
  frames that the pixel classifier really classifies as BATTLE / DIALOG /
  overworld, so the production heuristics are exercised for real.
* :class:`FakeBattleEmulator` is a duck-typed ``GuardedEmulator``: it exposes
  ``observe`` / ``press`` / ``wait`` / ``run_batch`` / ``frame_count``, plays a
  deterministic battle-menu state machine and records every press.  It can raise
  :class:`ControlDeniedError` (lease not held) and can keep a battle going
  forever so budget/turn caps can be tested.
* :class:`FakeRouter` scripts the DeepSeek decisions (as tool calls or as
  content JSON), records every request and can fail / be budget-blocked.
* :class:`FakeBus` / :class:`FakeSharedMemory` record published events and
  knowledge entries.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pytest

from llm_agent.config import AgentConfig
from llm_agent.contracts import (
    SIGNIFICANT_EVENTS,
    ActionOutcome,
    ActionStep,
    ApiUsage,
    ModelReply,
    ToolCall,
)
from llm_agent.memory import AgentMemory
from llm_agent.multi_agent.contracts import ControlDeniedError
from llm_agent.observation import analyze, frame_from_rgba

# --------------------------------------------------------------------------- #
# Synthetic Game Boy frames (160x144 RGBA, four DMG grey levels)
# --------------------------------------------------------------------------- #

SCREEN_H, SCREEN_W = 144, 160
WHITE, LIGHT, MID, DARK = 255, 153, 85, 0


def _canvas(level: int = WHITE) -> np.ndarray:
    rgba = np.empty((SCREEN_H, SCREEN_W, 4), dtype=np.uint8)
    rgba[..., :3] = np.uint8(level)
    rgba[..., 3] = 255
    return rgba


def _rect(img: np.ndarray, x: int, y: int, w: int, h: int, level: int = DARK) -> None:
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(SCREEN_W, int(x) + int(w)), min(SCREEN_H, int(y) + int(h))
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1, :3] = np.uint8(level)


def _battle_base() -> np.ndarray:
    """Bright battle-like screen with a full-width bottom message-box border.

    The geometry mirrors the real ROM captures found in the repository's run
    artifacts: a full-width box border around rows 98-101 / 142-143 with the box
    interior (text or menu) between them.
    """
    img = _canvas(WHITE)
    # HP-bar-like marks above the message box (kept small so the frame stays
    # white-dominant, exactly like the calibrated battle classifier expects).
    _rect(img, 8, 18, 48, 5, DARK)
    _rect(img, 8, 28, 48, 4, DARK)
    _rect(img, 104, 18, 48, 5, DARK)
    _rect(img, 104, 28, 48, 4, DARK)
    # full-width message-box borders (battle detection needs dark full rows)
    _rect(img, 0, 98, SCREEN_W, 4, DARK)
    _rect(img, 0, 142, SCREEN_W, 2, DARK)
    return img


def _draw_main_menu(img: np.ndarray, cursor: tuple[int, int] = (0, 0)) -> None:
    """Bottom-right 2x2 option grid (FIGHT/SWITCH top, ITEM/RUN bottom)."""
    for row, y in enumerate((106, 124)):
        for column, x in enumerate((96, 128)):
            _rect(img, x, y, 24, 3, DARK)
    cx = 88 if int(cursor[1]) == 0 else 120
    cy = 104 if int(cursor[0]) == 0 else 122
    _rect(img, cx, cy, 5, 6, DARK)


def _draw_vertical_list(img: np.ndarray, slot: int, rows: int = 4) -> None:
    """Bottom-left vertical list with a right-hand info column (moves/items)."""
    for index in range(rows):
        y = 104 + index * 9
        _rect(img, 16, y, 44, 3, DARK)
        _rect(img, 112, y, 20, 3, DARK)
    _rect(img, 10, 103 + max(0, min(rows - 1, int(slot))) * 9, 4, 5, DARK)


def _draw_party_list(img: np.ndarray, slot: int = 0) -> None:
    for index in range(3):
        y = 106 + index * 12
        _rect(img, 16, y, 40, 3, DARK)
        _rect(img, 100, y, 28, 3, DARK)
    _rect(img, 10, 105 + max(0, min(2, int(slot))) * 12, 4, 5, DARK)


def _draw_text_line(img: np.ndarray, x: int, y: int, width: int, *, segments: int = 6) -> None:
    """Draw a dashed glyph-like text line (never a solid full border)."""
    segment = max(3, int(width) // (max(1, segments) * 2))
    for index in range(max(1, segments)):
        _rect(img, int(x) + index * segment * 2, y, segment, 3, DARK)


def _draw_battle_text(img: np.ndarray, page: int = 0) -> None:
    lines = 1 + (int(page) % 2)
    for index in range(lines):
        _draw_text_line(img, 16, 112 + index * 16, 90 - index * 20)


def battle_main_frame(cursor: tuple[int, int] = (0, 0)) -> np.ndarray:
    img = _battle_base()
    _draw_main_menu(img, cursor)
    return img


def battle_move_frame(slot: int = 0) -> np.ndarray:
    img = _battle_base()
    _draw_vertical_list(img, slot, rows=4)
    return img


def battle_item_frame(slot: int = 0) -> np.ndarray:
    img = _battle_base()
    _draw_vertical_list(img, slot, rows=4)
    return img


def battle_party_frame(slot: int = 0) -> np.ndarray:
    img = _battle_base()
    _draw_party_list(img, slot)
    return img


def battle_text_frame(page: int = 0) -> np.ndarray:
    img = _battle_base()
    _draw_battle_text(img, page)
    return img


def battle_intro_text_frame(remaining: int) -> np.ndarray:
    """Battle intro message page that only advances when A is pressed.

    Real Pokémon Red battle intro text waits for the A button (verified on real
    ROM frames); every ``remaining`` value renders a slightly different page so
    the per-press pixel verification is meaningful.
    """
    img = _battle_base()
    value = max(0, int(remaining))
    lines = 1 + (value % 2)
    for index in range(lines):
        width = 60 + (value * 7) % 60
        _draw_text_line(img, 16, 112 + index * 16, width, segments=4 + value % 3)
    # a "continue" arrow marker at the bottom-right of the box (like the real one)
    _rect(img, 148, 132, 6, 6, DARK)
    return img


def battle_blank_frame() -> np.ndarray:
    """A battle screen whose bottom band shows no readable menu (unknown layout)."""
    return _battle_base()


def levelup_frame(page: int = 0) -> np.ndarray:
    """Black screen with a white post-battle text box (DIALOG classification)."""
    img = _canvas(DARK)
    _rect(img, 8, 88, 144, 48, WHITE)
    _rect(img, 16, 96 + int(page) * 10, 100, 3, DARK)
    _rect(img, 16, 108 + int(page) * 10, 80, 3, DARK)
    return img


def overworld_frame(seed: int = 0) -> np.ndarray:
    """Grey tiled overworld frame (no white panel, no black rows)."""
    rng = np.random.default_rng(int(seed))
    levels = rng.integers(1, 3, size=(18, 20), dtype=np.uint8)
    tiles = np.repeat(np.repeat(levels, 8, axis=0), 8, axis=1)
    img = np.empty((SCREEN_H, SCREEN_W, 4), dtype=np.uint8)
    grey = np.where(tiles == 1, np.uint8(LIGHT), np.uint8(MID))
    img[..., :3] = grey[..., None]
    img[..., 3] = 255
    return img


def black_frame() -> np.ndarray:
    return _canvas(DARK)


def make_frame(rgba: np.ndarray, frame_index: int = 0):
    """Build one real :class:`llm_agent.contracts.Frame` from pixels."""
    return frame_from_rgba(int(frame_index), np.ascontiguousarray(rgba, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Fake GuardedEmulator
# --------------------------------------------------------------------------- #

class FakeBattleEmulator:
    """Deterministic battle state machine implementing the emulator surface.

    States: ``battle_main`` -> ``battle_moves`` / ``battle_items`` /
    ``battle_party`` / run message -> ``battle_text`` (message pages) ->
    ``levelup`` (post-battle dialog) -> ``overworld``.

    Options
    -------
    end_after_fights:
        Number of fight actions before the enemy goes down (``None`` keeps the
        battle running forever, which is how budget/turn caps are tested).
    run_succeeds / item_use_catches:
        Whether a RUN action leaves the battle / whether using an item from the
        item list ends the battle (catch scenario).
    start / intro_waits:
        Start in a different screen; ``intro_waits`` wait() calls turn a black
        transition screen into the battle screen (battle-start detection test).
    denied:
        Every press raises :class:`ControlDeniedError` (lease not held).
    ignore_presses:
        Presses are recorded but do not change the screen (verification test).
    """

    def __init__(
        self,
        *,
        start: str = "battle_main",
        intro_waits: int = 0,
        intro_observes: int = 0,
        intro_a_presses: int = 0,
        end_after_fights: int | None = 1,
        run_succeeds: bool = True,
        item_use_catches: bool = True,
        denied: bool = False,
        ignore_presses: bool = False,
        seed: int = 0,
    ) -> None:
        self.state = str(start)
        self.intro_waits = max(0, int(intro_waits))
        #: number of initial observations that return a battle-intro frame
        #: (battle screen visible, message box empty, command menu not yet shown)
        self.intro_observes = max(0, int(intro_observes))
        #: Battle intro text pages that require an A press each before the first
        #: command menu (real Pokémon Red behaviour).
        self._intro_a_remaining = max(0, int(intro_a_presses))
        self.intro_a_presses_initial = self._intro_a_remaining
        self.end_after_fights = end_after_fights
        self.run_succeeds = bool(run_succeeds)
        self.item_use_catches = bool(item_use_catches)
        self.denied = bool(denied)
        self.ignore_presses = bool(ignore_presses)
        self.seed = int(seed)
        self.main_cursor: tuple[int, int] = (0, 0)
        self.move_slot = 0
        self.item_slot = 0
        self.party_slot = 0
        self.text_page = 0
        self.text_pages = 1
        self.text_kind = "message"
        self.pending_state = "battle_main"
        self.levelup_page = 0
        self.attacks = 0
        self.presses: list[str] = []
        self.observations = 0
        self.waits = 0
        self.screen_log: list[str] = []
        self._frame_count = 0
        self._last = self._render_current()

    # -- emulator surface ------------------------------------------------- #
    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def started(self) -> bool:
        return True

    @property
    def last_frame_index(self) -> int:
        return self._frame_count

    def observe(self):
        if self._intro_a_remaining > 0:
            # Battle intro message page: waits for A (real ROM behaviour).
            self.observations += 1
            self.screen_log.append("battle_intro_text")
            return make_frame(
                battle_intro_text_frame(self._intro_a_remaining), self._frame_count
            )
        if self.intro_observes > 0:
            # Battle intro animation: battle screen visible, empty message box,
            # command menu not rendered yet.
            self.intro_observes -= 1
            self.observations += 1
            self.screen_log.append("battle_intro")
            return make_frame(battle_blank_frame(), self._frame_count)
        self.observations += 1
        self.screen_log.append(self.state)
        return self._frame()

    def wait(self, frames: int = 1):
        budget = max(1, int(frames or 1))
        self.waits += 1
        self._frame_count += budget
        if self.state == "black" and self.intro_waits > 0:
            self.intro_waits -= 1
            if self.intro_waits <= 0:
                self.state = "battle_main"
        if not self.ignore_presses:
            self._last = self._render_current()
        return self._frame()

    def press(self, button: str, *, frames: int | None = None):
        if self.denied:
            raise ControlDeniedError(
                "control lease is held by the navigation agent",
                agent="pokemon",
                owner="navigation",
            )
        name = str(button or "").strip().lower()
        if name not in ("a", "b", "up", "down", "left", "right", "start", "select"):
            raise ValueError(f"invalid button {button!r}")
        self.presses.append(name)
        self._frame_count += int(frames) if frames else 24
        if self._intro_a_remaining > 0:
            # The intro text only advances on A (bounded by the agent's own
            # intro press budget in production).
            if name in ("a", "start"):
                self._intro_a_remaining -= 1
            if self._intro_a_remaining <= 0:
                self._intro_a_remaining = 0
                self.state = "battle_main"
            self._last = self._render_current()
            return self._frame()
        if self.intro_observes > 0:
            # Battle-intro animation: presses do not reach the game state yet.
            return self._frame()
        if not self.ignore_presses:
            self._apply(name)
            self._last = self._render_current()
        return self._frame()

    def run_batch(self, buttons: Sequence[str], *, stop_on_event: bool = True) -> ActionOutcome:
        previous = self.observe()
        baseline = analyze(previous, None).event
        steps: list[ActionStep] = []
        events = []
        start_frame = self._frame_count
        stopped = "completed"
        for button in buttons:
            frame = self.press(str(button))
            analysis = analyze(frame, previous)
            steps.append(
                ActionStep(
                    button=str(button),
                    frame=frame,
                    screen_changed=analysis.changed,
                    changed_ratio=analysis.changed_ratio,
                    event=analysis.event,
                )
            )
            if analysis.event != baseline and analysis.event in SIGNIFICANT_EVENTS:
                events.append(analysis.event)
                if stop_on_event:
                    stopped = "event"
                    break
            previous = frame
        return ActionOutcome(
            buttons=[str(b) for b in buttons],
            steps=steps,
            start_frame=start_frame,
            end_frame=self._frame_count,
            stopped_reason=stopped,
            events=events,
            summary=f"{len(steps)} presses, stopped={stopped}",
        )

    # -- state machine ---------------------------------------------------- #
    def _render_current(self) -> np.ndarray:
        if self._intro_a_remaining > 0:
            return battle_intro_text_frame(self._intro_a_remaining)
        if self.state == "black":
            return black_frame()
        if self.state == "overworld":
            return overworld_frame(seed=self.seed)
        if self.state == "levelup":
            return levelup_frame(self.levelup_page)
        if self.state == "battle_moves":
            return battle_move_frame(self.move_slot)
        if self.state == "battle_items":
            return battle_item_frame(self.item_slot)
        if self.state == "battle_party":
            return battle_party_frame(self.party_slot)
        if self.state == "battle_text":
            return battle_text_frame(self.text_page)
        if self.state == "battle_blank":
            return battle_blank_frame()
        return battle_main_frame(self.main_cursor)

    def _frame(self):
        return make_frame(self._last, self._frame_count)

    def _begin_text(self, kind: str, *, pages: int, pending: str) -> None:
        self.state = "battle_text"
        self.text_kind = str(kind)
        self.text_pages = max(1, int(pages))
        self.text_page = 0
        self.pending_state = str(pending)

    def _finish_text(self) -> None:
        if self.pending_state == "levelup":
            self.state = "levelup"
            self.levelup_page = 0
            return
        self.state = self.pending_state

    def _apply(self, button: str) -> None:
        state = self.state
        if state in ("black", "overworld"):
            return
        if state == "battle_main":
            if button in ("up", "down"):
                self.main_cursor = (1 - self.main_cursor[0], self.main_cursor[1])
            elif button in ("left", "right"):
                self.main_cursor = (self.main_cursor[0], 1 - self.main_cursor[1])
            elif button in ("a", "start"):
                index = self.main_cursor[0] * 2 + self.main_cursor[1]
                if index == 0:
                    self.state = "battle_moves"
                    self.move_slot = 0
                elif index == 1:
                    self.state = "battle_party"
                    self.party_slot = 0
                elif index == 2:
                    self.state = "battle_items"
                    self.item_slot = 0
                else:
                    self._begin_text(
                        "ran",
                        pages=1,
                        pending="overworld" if self.run_succeeds else "battle_main",
                    )
            return
        if state == "battle_moves":
            if button == "down":
                self.move_slot = min(3, self.move_slot + 1)
            elif button == "up":
                self.move_slot = max(0, self.move_slot - 1)
            elif button == "b":
                self.state = "battle_main"
            elif button in ("a", "start"):
                self.attacks += 1
                if self.end_after_fights is not None and self.attacks >= self.end_after_fights:
                    self._begin_text("faint", pages=2, pending="levelup")
                else:
                    self._begin_text("attack", pages=2, pending="battle_main")
            return
        if state == "battle_items":
            if button == "down":
                self.item_slot = min(7, self.item_slot + 1)
            elif button == "up":
                self.item_slot = max(0, self.item_slot - 1)
            elif button == "b":
                self.state = "battle_main"
            elif button in ("a", "start"):
                if self.item_use_catches:
                    self._begin_text("caught", pages=2, pending="overworld")
                else:
                    self._begin_text("item", pages=2, pending="battle_main")
            return
        if state == "battle_party":
            if button == "down":
                self.party_slot = min(2, self.party_slot + 1)
            elif button == "up":
                self.party_slot = max(0, self.party_slot - 1)
            elif button == "b":
                self.state = "battle_main"
            elif button in ("a", "start"):
                self._begin_text("switched", pages=1, pending="battle_main")
            return
        if state == "battle_text":
            if button in ("a", "start"):
                self.text_page += 1
                if self.text_page >= self.text_pages:
                    self._finish_text()
            return
        if state == "levelup":
            if button in ("a", "start"):
                self.levelup_page += 1
                if self.levelup_page >= 2:
                    self.state = "overworld"
            return

    # -- test helpers ----------------------------------------------------- #
    def force_state(self, state: str) -> None:
        """Force a screen state (used to script already-ended situations)."""
        self.state = str(state)
        self._last = self._render_current()


# --------------------------------------------------------------------------- #
# Fake router / bus / shared memory
# --------------------------------------------------------------------------- #

class FakeRouter:
    """Scripted model router: returns one battle_action tool call per turn.

    A scripted decision is a dict like ``{"action": "fight", "move_slot": 2}``
    (sent as the ``battle_action`` tool call) or
    ``{"tool": "finish", "outcome": "win", ...}`` for the finish tool.
    ``{"tool": "content", ...}`` returns the decision as JSON content instead of
    a tool call.  When the script is exhausted a safe ``finish/unknown`` reply is
    returned so the agent's extra resolution-outcome read stays deterministic.
    """

    def __init__(
        self,
        decisions: Iterable[Mapping[str, Any]] | None = None,
        *,
        model: str = "fake-battle-model",
        allowed: bool = True,
        reason: str = "",
        fail_with: BaseException | None = None,
        usage: ApiUsage | None = None,
    ) -> None:
        self.decisions: list[dict[str, Any]] = [dict(item) for item in (decisions or [])]
        self.calls: list[dict[str, Any]] = []
        self.model = str(model)
        self.allowed = bool(allowed)
        self.reason = str(reason)
        self.fail_with = fail_with
        self.usage = usage or ApiUsage(prompt_tokens=1200, completion_tokens=120, total_tokens=1320)

    # -- router surface --------------------------------------------------- #
    def model_for(self, role: Any) -> str:
        return self.model

    def can_call(self, role: Any) -> tuple[bool, str]:
        return (self.allowed, self.reason)

    def complete(
        self,
        role: Any,
        messages: Any,
        *,
        tools: Any = None,
        max_tokens: Any = None,
        thinking: Any = None,
    ) -> ModelReply:
        self.calls.append(
            {
                "role": role,
                "messages": list(messages or []),
                "tools": tools,
                "max_tokens": max_tokens,
                "thinking": thinking,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        decision = dict(self.decisions.pop(0)) if self.decisions else {
            "tool": "finish",
            "outcome": "unknown",
            "reason": "no scripted decision left",
        }
        tool_name = str(decision.pop("tool", "battle_action"))
        content = str(decision.pop("content", "") or "")
        extra_lessons = decision.pop("lessons", None) or []
        if tool_name == "content":
            import json

            return self._reply(content or json.dumps(decision, ensure_ascii=False), [])
        arguments = dict(decision)
        calls = [ToolCall(id=f"call_{len(self.calls)}", name=tool_name, arguments=arguments)]
        for index, lesson in enumerate(extra_lessons):
            calls.append(
                ToolCall(
                    id=f"lesson_{len(self.calls)}_{index}",
                    name="record_lesson",
                    arguments=dict(lesson),
                )
            )
        return self._reply(content, calls)

    def _reply(self, content: str, tool_calls: list[ToolCall]) -> ModelReply:
        return ModelReply(
            content=content,
            tool_calls=tool_calls,
            usage=self.usage,
            model=self.model,
            latency_s=0.0,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    # -- assertions helpers ---------------------------------------------- #
    def all_text(self) -> str:
        """Concatenated text of every prompt sent so far (for prompt tests)."""
        chunks: list[str] = []
        for call in self.calls:
            for message in call.get("messages") or []:
                content = message.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, Mapping) and block.get("type") == "text":
                            chunks.append(str(block.get("text") or ""))
        return "\n".join(chunks)

    def images_sent(self) -> int:
        """Number of image content blocks sent (proves screenshots are real)."""
        count = 0
        for call in self.calls:
            for message in call.get("messages") or []:
                content = message.get("content")
                if isinstance(content, list):
                    count += sum(
                        1
                        for block in content
                        if isinstance(block, Mapping) and block.get("type") == "image_url"
                    )
        return count


class FakeBus:
    """Minimal event bus: records every published AgentMessage."""

    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, message: Any) -> Any:
        self.published.append(message)
        return message

    def event_types(self) -> list[str]:
        return [str(getattr(message, "event_type", getattr(message, "value", message))) for message in self.published]


class FakeSharedMemory:
    """Shared world memory stand-in recording every KnowledgeEntry."""

    def __init__(self, *, fail: bool = False) -> None:
        self.entries: list[Any] = []
        self.fail = bool(fail)
        self.calls = 0

    def add_knowledge(self, entry: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("shared memory write failed")
        self.entries.append(entry)
        return entry


class RecordOnlySharedMemory:
    """Duck-typed shared memory that only exposes ``record()``."""

    def __init__(self) -> None:
        self.entries: list[Any] = []

    def record(self, entry: Any) -> Any:
        self.entries.append(entry)
        return entry


class FakeArbiter:
    """Stand-in arbiter exposing owner state only (presses are gated elsewhere)."""

    def __init__(self, owner: str = "pokemon") -> None:
        self.owner = owner


# --------------------------------------------------------------------------- #
# pytest fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def fake_clock() -> Callable[[], float]:
    """Deterministic monotonic clock."""
    state = {"now": 1000.0}

    def _clock() -> float:
        state["now"] += 1.0
        return state["now"]

    return _clock


@pytest.fixture()
def agent_config(tmp_path) -> AgentConfig:
    """Agent config whose memory root is a temporary directory."""
    cfg = AgentConfig()
    cfg.memory.root = str(tmp_path / "memory")
    cfg.memory.mirror_to_run = False
    return cfg


@pytest.fixture()
def memory(agent_config: AgentConfig) -> AgentMemory:
    return AgentMemory(agent_config.memory)


@pytest.fixture()
def screen_dir(tmp_path) -> str:
    return str(tmp_path / "screens")


@pytest.fixture()
def make_agent(tmp_path, agent_config, screen_dir, fake_clock):
    """Factory: build a PokemonAgent with injectable fakes."""
    from llm_agent.pokemon.agent import PokemonAgent

    def _make(
        router: Any | None = None,
        *,
        config: AgentConfig | None = None,
        bus: Any | None = None,
        shared_memory: Any | None = None,
        arbiter: Any | None = None,
        memory_obj: Any | None = None,
        screen: str | None = None,
    ) -> Any:
        cfg = config if config is not None else agent_config
        return PokemonAgent(
            router=router if router is not None else FakeRouter(),
            bus=bus,
            arbiter=arbiter,
            shared_memory=shared_memory,
            memory=memory_obj,
            config=cfg,
            screen_dir=screen if screen is not None else screen_dir,
            clock=fake_clock,
        )

    return _make


@pytest.fixture()
def fight_win_emulator() -> FakeBattleEmulator:
    """Battle that ends after the first fight action (level-up dialog follows)."""
    return FakeBattleEmulator(end_after_fights=1)


@pytest.fixture()
def endless_emulator() -> FakeBattleEmulator:
    """Battle that never ends (budget / turn-cap tests)."""
    return FakeBattleEmulator(end_after_fights=None)
