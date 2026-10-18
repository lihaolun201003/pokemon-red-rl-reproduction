"""Battle start / end detection tests (pixels only, synthetic frames).

The synthetic frames used here are classified by the *real* production
heuristics (``llm_agent.observation.analyze`` and the battle layout detector), so
these tests exercise the same pixel logic the emulator run will use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llm_agent.observation import analyze
from llm_agent.pokemon.agent import (
    BATTLE_END_CONSECUTIVE,
    BATTLE_MENU_MAIN,
    BATTLE_MENU_MOVE,
    BATTLE_MENU_TEXT,
    BATTLE_MENU_UNKNOWN,
    battle_menu_layout,
    looks_like_battle,
)
from tests.pokemon.conftest import (
    FakeBattleEmulator,
    FakeBus,
    FakeRouter,
    battle_blank_frame,
    battle_main_frame,
    battle_move_frame,
    battle_text_frame,
    black_frame,
    levelup_frame,
    make_frame,
    overworld_frame,
)

#: Real frames archived by earlier runs of this project (optional: skipped when
#: the local artifacts directory is absent, e.g. in a clean checkout).
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
REAL_RUN_SCREENS = (
    ARTIFACTS / "llm_agent" / "runs" / "run_20261018_111635_80f907" / "screens"
)
REAL_EVAL_DEMO = (
    ARTIFACTS / "training" / "seed_0" / "eval" / "_demos" / "demo_300k" / "stochastic_25pct.png"
)


def _load_frame(path: Path):
    from PIL import Image

    rgba = np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)
    return make_frame(rgba, 0)


def _real_battle_frames() -> list[Path]:
    if not REAL_RUN_SCREENS.is_dir():
        pytest.skip("archived real run screenshots not available")
    frames = []
    for path in sorted(REAL_RUN_SCREENS.glob("*.png")):
        frame = _load_frame(path)
        if analyze(frame, None).event.value == "battle":
            frames.append(path)
    if not frames:
        pytest.skip("no archived real battle frames found")
    return frames


def test_heuristics_are_calibrated_on_real_battle_frames() -> None:
    """The battle hint / layout detector accept archived *real* battle frames.

    This is the calibration guard for the acceptance run: the heuristics were
    tuned on the repository's real captures (battle text boxes are 40px tall with
    a full-width border; battle menus only draw a half-width box).
    """
    paths = _real_battle_frames()
    assert len(paths) >= 10, f"expected a set of real battle frames, got {len(paths)}"
    allowed = {BATTLE_MENU_TEXT, BATTLE_MENU_MOVE, BATTLE_MENU_MAIN, BATTLE_MENU_UNKNOWN}
    layouts_calibrated = []
    for path in paths:
        frame = _load_frame(path)
        assert looks_like_battle(frame), f"real battle frame not detected: {path}"
        layout = battle_menu_layout(frame)
        assert layout in allowed, f"unexpected layout for real battle frame: {path} -> {layout}"
        layouts_calibrated.append(layout)
    # Most archived frames of that run are battle message frames.
    assert layouts_calibrated.count(BATTLE_MENU_TEXT) >= len(paths) // 2


REAL_COMMAND_MENU = ARTIFACTS / "multi_agent_v1" / "recon4c" / "p04_00_text.png"
REAL_MOVE_MENU = (
    ARTIFACTS
    / "llm_agent"
    / "runs"
    / "run_20261018_111635_80f907"
    / "screens"
    / "frame_000002126_00a60262.png"
)


def test_real_command_menu_frame_is_detected_as_the_main_menu() -> None:
    """Calibration on the REAL FIGHT/PKMN/ITEM/RUN command-menu screen."""
    if not REAL_COMMAND_MENU.is_file():
        pytest.skip("archived real command-menu frame not available")
    frame = _load_frame(REAL_COMMAND_MENU)
    assert looks_like_battle(frame)
    assert battle_menu_layout(frame) == BATTLE_MENU_MAIN


def test_real_move_menu_frame_is_detected_as_the_move_list() -> None:
    """Calibration on a REAL move-list screen (TACKLE / TAIL WHIP …)."""
    if not REAL_MOVE_MENU.is_file():
        pytest.skip("archived real move-menu frame not available")
    frame = _load_frame(REAL_MOVE_MENU)
    assert looks_like_battle(frame)
    assert battle_menu_layout(frame) == BATTLE_MENU_MOVE


def test_real_message_frames_classify_as_text() -> None:
    paths = _real_battle_frames()
    layouts = [battle_menu_layout(_load_frame(path)) for path in paths]
    # Most archived real frames carry text in the battle message box.
    assert layouts.count(BATTLE_MENU_TEXT) >= len(paths) // 2


def test_real_non_battle_frames_are_not_mistaken_for_battles() -> None:
    if not REAL_RUN_SCREENS.is_dir():
        pytest.skip("archived real run screenshots not available")
    paths = sorted(REAL_RUN_SCREENS.glob("*.png"))[:120]
    false_positives = []
    for path in paths:
        frame = _load_frame(path)
        if analyze(frame, None).event.value == "battle":
            continue
        if looks_like_battle(frame):
            false_positives.append(path.name)
    assert not false_positives, f"overworld/menu frames mistaken for battles: {false_positives}"


def test_eval_demo_battle_frame_is_detected() -> None:
    if not REAL_EVAL_DEMO.is_file():
        pytest.skip("archived eval demo frame not available")
    frame = _load_frame(REAL_EVAL_DEMO)
    assert analyze(frame, None).event.value == "battle"
    assert looks_like_battle(frame)
    assert battle_menu_layout(frame) == BATTLE_MENU_TEXT


def test_synthetic_battle_frames_classify_as_battle() -> None:
    """Sanity: the fixture frames really are battle/dialog/overworld screens."""
    cases = {
        "battle_main": (battle_main_frame(), "battle", BATTLE_MENU_MAIN),
        "battle_moves": (battle_move_frame(), "battle", BATTLE_MENU_MOVE),
        "battle_text": (battle_text_frame(0), "battle", BATTLE_MENU_TEXT),
    }
    for name, (rgba, expected_event, expected_layout) in cases.items():
        frame = make_frame(rgba)
        assert analyze(frame, None).event.value == expected_event, name
        assert battle_menu_layout(frame) == expected_layout, name
    assert analyze(make_frame(levelup_frame(0)), None).event.value == "dialog"
    assert analyze(make_frame(overworld_frame()), None).event.value == "none"
    assert analyze(make_frame(black_frame()), None).event.value == "transition"
    assert battle_menu_layout(make_frame(battle_blank_frame())) == "unknown"


def test_battle_is_detected_after_a_black_transition(make_agent) -> None:
    """The agent waits through the battle transition and detects the battle."""
    emulator = FakeBattleEmulator(start="black", intro_waits=2, end_after_fights=1)
    bus = FakeBus()
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router, bus=bus)

    result = agent.handle_battle(emulator=emulator, max_turns=5)

    assert result["status"] == "battle_ended"
    assert emulator.waits >= 2  # it waited for the transition instead of giving up
    assert any(state.startswith("battle") for state in emulator.screen_log[2:])
    started = [message for message in bus.published if message.event_type.value == "battle_started"]
    assert len(started) == 1
    assert started[0].source_agent.value == "pokemon"


def test_no_battle_returns_failed_without_api_calls(make_agent, endless_emulator) -> None:
    """A non-battle screen is reported as a structured failure (no crash)."""
    end_never = FakeBattleEmulator(start="overworld", end_after_fights=None)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=end_never, max_turns=3)

    assert result["status"] == "failed"
    assert "not detected" in result["reason"].lower()
    assert result["turns"] == 0
    assert result["api_calls"] == 0
    assert router.calls == []


def test_battle_does_not_end_while_resolution_text_is_visible(make_agent) -> None:
    """A post-battle level-up dialogue is still battle resolution, not the end."""
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=5)

    assert result["status"] == "battle_ended"
    # The level-up dialog state was observed and advanced with A presses...
    assert "levelup" in emulator.screen_log
    assert "battle_text" in emulator.screen_log
    # ...and the returned state required BATTLE_END_CONSECUTIVE clear observations.
    assert BATTLE_END_CONSECUTIVE >= 2
    clear_tail = emulator.screen_log[-BATTLE_END_CONSECUTIVE:]
    assert clear_tail == ["overworld"] * BATTLE_END_CONSECUTIVE
    assert "battle_main" not in clear_tail
    # More A presses happened than the single fight decision needed.
    action_presses = result["actions"][0]["presses"]
    assert len(emulator.presses) > len(action_presses)
    assert all(button == "a" for button in emulator.presses[len(action_presses):])


def test_post_battle_observations_are_included_in_evidence(make_agent) -> None:
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter(
        [
            {
                "action": "fight",
                "move_slot": 1,
                "visible_observations": {"battle_text": "some visible battle message"},
            }
        ]
    )
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=5)

    screens = result["evidence"]["screens"]
    frames = result["evidence"]["frames"]
    assert screens, "at least the turn screenshot must be recorded"
    assert any("resolution" in str(path) for path in screens), (
        f"post-battle resolution screenshots missing from evidence: {screens}"
    )
    assert frames and all(isinstance(index, int) for index in frames)
    # Frames include the settle/resolution observations, not just the turn frame.
    assert len(frames) > len(result["actions"]) + 1


def test_battle_intro_text_requires_a_presses_then_real_decision(make_agent) -> None:
    """Real G3 gap: intro text waits for A; produce it, then the menu, then a turn.

    Sequence: battle-intro text pages that require an A press each ("Wild X
    appeared", "Go! Y!") N times -> command menu -> a real LLM decision.  The
    agent must advance the intro with A, make no LLM call before the menu, never
    count intro presses as decision turns, and finish the battle (not interrupted).
    """
    intro_pages = 7
    emulator = FakeBattleEmulator(intro_a_presses=intro_pages, end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 2}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert result["status"] != "interrupted"
    intro = result["intro_wait"]
    assert len(intro["presses"]) == intro_pages, intro
    assert all(entry["button"] == "a" for entry in intro["presses"])
    assert intro["a_presses"] == intro_pages
    assert intro["budget_exhausted"] is False
    # Intro presses are real presses, in order, and they are NOT decisions.
    assert emulator.presses[:intro_pages] == ["a"] * intro_pages
    assert result["turns"] == 1  # max_turns counts decisions, not intro presses
    assert len(result["actions"]) == 1
    assert result["actions"][0]["turn"] == 1
    assert result["actions"][0]["decision"]["action"] == "fight"
    assert result["actions"][0]["presses"] == ["a", "down", "a"]  # no intro A mixed in
    assert result["actions"][0]["verified"] is True
    # The intro observations/evidence are recorded for the acceptance report.
    assert intro["observations"] >= intro_pages
    assert intro["waited_frames"] > 0
    assert intro["screens"]
    assert any("battle_intro" in str(path) for path in result["evidence"]["screens"])
    assert "battle_intro_text" in emulator.screen_log
    # An LLM turn happened only after the first command menu was visible.
    assert result["api_calls"] >= 1
    assert len(router.calls) == result["api_calls"]


def test_battle_intro_a_press_budget_is_enforced(make_agent) -> None:
    """A never-ending intro is bounded: structured failed, never blind/looping."""
    from llm_agent.pokemon.agent import INTRO_MAX_A_PRESSES, INTRO_MAX_FRAMES

    emulator = FakeBattleEmulator(intro_a_presses=INTRO_MAX_A_PRESSES + 30, end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "failed"
    assert "bounded budget" in result["reason"].lower()
    assert result["status"] != "interrupted"
    assert result["turns"] == 0
    assert result["api_calls"] == 0
    assert router.calls == []
    intro = result["intro_wait"]
    assert intro["budget_exhausted"] is True
    assert len(intro["presses"]) <= INTRO_MAX_A_PRESSES
    assert intro["waited_frames"] <= INTRO_MAX_FRAMES
    # No press beyond the budget reached the emulator.
    assert len(emulator.presses) == len(intro["presses"]) <= INTRO_MAX_A_PRESSES
    assert intro["layouts"], "every observed layout must be recorded as evidence"


def test_battle_intro_is_waited_out_then_a_real_turn_is_played(make_agent) -> None:
    """Intro animation frames (empty box, no menu) are waited out, not interrupted.

    The battle screen is visible for several observations before the command menu
    renders.  The agent must wait/advance (bounded), then play a real LLM-decided
    turn and finish the battle.
    """
    emulator = FakeBattleEmulator(intro_observes=5, end_after_fights=1)
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert result["status"] != "interrupted"
    assert result["turns"] == 1  # max_turns counts decisions, not intro waits
    assert result["api_calls"] >= 1
    assert result["actions"], "the agent must play at least one real decision turn"
    assert result["actions"][0]["decision"]["action"] == "fight"
    assert result["actions"][0]["verified"] is True
    # The intro observations are recorded as evidence of the wait.
    intro = result["intro_wait"]
    assert intro["observations"] >= 4
    assert intro["waited_frames"] >= 30
    assert intro["timed_out"] is False
    assert intro["layouts"]
    assert intro["screens"], "intro observations must be saved as evidence"
    assert any("battle_intro" in str(path) for path in result["evidence"]["screens"])
    assert "battle_intro" in emulator.screen_log


class _IntroThenOverworld(FakeBattleEmulator):
    """Battle-looking intro frames that fall straight back to the overworld."""

    def __init__(self) -> None:
        super().__init__(start="battle_blank", intro_observes=2, end_after_fights=None)

    def observe(self):
        if self.intro_observes <= 0 and self.state == "battle_blank":
            self.state = "overworld"
            self._last = self._render_current()
        return super().observe()


def test_battle_that_ends_before_the_menu_returns_battle_ended(make_agent) -> None:
    emulator = _IntroThenOverworld()
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert result["turns"] == 0
    assert result["api_calls"] == 0
    assert "ended before" in result["reason"].lower()


def _real_intro_frame_path() -> Path | None:
    """First archived real battle-intro TEXT page (battle visible, no menu yet)."""
    if not REAL_RUN_SCREENS.is_dir():
        return None
    for path in sorted(REAL_RUN_SCREENS.glob("*.png")):
        frame = _load_frame(path)
        if (
            analyze(frame, None).event.value == "battle"
            and battle_menu_layout(frame) == BATTLE_MENU_TEXT
        ):
            return path
    return None


def test_real_archived_intro_screen_is_waited_out(make_agent) -> None:
    """Acceptance-style check on a REAL intro frame of this ROM.

    The archived real intro frame (battle screen + empty message box, command
    menu not rendered) must be treated as an intro: waited out, followed by a
    real model decision, and reported as battle_ended — never interrupted.
    """
    intro_path = _real_intro_frame_path()
    if intro_path is None:
        pytest.skip("archived real battle-intro frame not available")
    from PIL import Image

    intro_rgba = np.array(Image.open(intro_path).convert("RGBA"), dtype=np.uint8)

    class RealIntroEmulator(FakeBattleEmulator):
        def __init__(self) -> None:
            super().__init__(end_after_fights=1)
            self._intro_remaining = 4

        def observe(self):
            if self._intro_remaining > 0:
                self._intro_remaining -= 1
                self.observations += 1
                self.screen_log.append("battle_intro")
                return make_frame(intro_rgba, self._frame_count)
            return super().observe()

        def press(self, button, *, frames=None):
            if self._intro_remaining > 0:
                # Intro animation: presses are recorded but the game state is not
                # advanced yet, exactly like a real intro animation frame.
                self.presses.append(str(button or "").strip().lower())
                self._frame_count += int(frames) if frames else 24
                return make_frame(intro_rgba, self._frame_count)
            return super().press(button, frames=frames)

    emulator = RealIntroEmulator()
    router = FakeRouter([{"action": "fight", "move_slot": 1}])
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=3)

    assert result["status"] == "battle_ended"
    assert result["turns"] == 1
    assert result["actions"] and result["actions"][0]["verified"] is True
    assert result["intro_wait"]["observations"] >= 3
    assert result["intro_wait"]["waited_frames"] >= 30
    assert result["intro_wait"]["presses"], "intro text must be advanced with A presses"
    assert emulator.presses[: len(result["intro_wait"]["presses"])] == [
        entry["button"] for entry in result["intro_wait"]["presses"]
    ]
    assert emulator.screen_log.count("battle_intro") >= 3


def test_post_turn_battle_text_is_advanced_before_the_next_menu(
    make_agent, endless_emulator
) -> None:
    """After the first menu, "used X" text still waits for A; settle advances it."""
    router = FakeRouter(
        [
            {"action": "fight", "move_slot": 1},
            {"action": "fight", "move_slot": 1},
        ]
    )
    agent = make_agent(router)

    result = agent.handle_battle(emulator=endless_emulator, max_turns=2)

    assert result["status"] == "budget"  # turn cap reached after two decisions
    assert result["turns"] == 2
    assert len(result["actions"]) == 2
    # turn 1 action (a, a) -> two battle-text pages advanced with A -> turn 2
    assert endless_emulator.presses == ["a", "a", "a", "a", "a", "a"]
    assert result["actions"][1]["presses"] == ["a", "a"]


def test_finish_is_rejected_while_the_battle_menu_is_visible(make_agent) -> None:
    """A premature `finish` is rejected by the pixels; the agent keeps fighting."""
    emulator = FakeBattleEmulator(end_after_fights=1)
    router = FakeRouter(
        [
            {"tool": "finish", "outcome": "win"},
            {"action": "fight", "move_slot": 1},
        ]
    )
    agent = make_agent(router)

    result = agent.handle_battle(emulator=emulator, max_turns=4)

    assert result["status"] == "battle_ended"
    assert len(result["actions"]) == 2
    premature, real_action = result["actions"]
    assert premature["decision"]["action"] == "finish"
    assert premature["verified"] is False
    assert real_action["decision"]["action"] == "fight"
    assert real_action["verified"] is True
    assert result["turns"] == 2
