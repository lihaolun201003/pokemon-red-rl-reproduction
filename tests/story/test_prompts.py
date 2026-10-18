"""Spoiler-boundary and honesty checks for the story agent prompt."""

from __future__ import annotations

import re
from pathlib import Path

from llm_agent.story.prompts import (
    HONESTY_GUIDE,
    SYSTEM_PROMPT,
    render_no_tool_nudge,
    render_observation,
)

STORY_PACKAGE = Path(__file__).resolve().parents[2] / "llm_agent" / "story"

#: Any of these appearing in the prompt would mean the agent was handed story
#: knowledge instead of discovering it on screen.
_SPOILER_WORDS = [
    # places
    "pallet",
    "viridian",
    "pewter",
    "cerulean",
    "vermilion",
    "lavender",
    "celadon",
    "saffron",
    "fuchsia",
    "cinnabar",
    "indigo",
    "town",
    "city",
    "route",
    # characters
    "oak",
    "gary",
    "brock",
    "misty",
    "surge",
    "erika",
    "koga",
    "sabrina",
    "blaine",
    "giovanni",
    "rocket",
    # mechanics that imply late-game knowledge
    "gym",
    "badge",
    "pokedex",
    "pokédex",
    "hm0",
    # species
    "bulbasaur",
    "charmander",
    "squirtle",
    "pikachu",
    "rattata",
    "pidgey",
    "zubat",
    "geodude",
    "magikarp",
    "eevee",
    "mewtwo",
    "mew",
]


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


def test_system_prompt_has_no_spoilers() -> None:
    lowered = SYSTEM_PROMPT.lower()
    hits = [word for word in _SPOILER_WORDS if _contains_word(lowered, word)]
    assert hits == [], f"system prompt contains forbidden story knowledge: {hits}"


def test_prompt_has_no_prewritten_dialogue() -> None:
    # No quoted dialogue line at all: every word must come from the game screen.
    assert '"' not in SYSTEM_PROMPT.replace('"verified"', "").replace('"inferred"', "").replace(
        '"hypothesis"', ""
    )
    assert ":" not in SYSTEM_PROMPT.split("# WHO YOU ARE")[0]


def test_prompt_declares_first_time_player_and_screen_only() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "first-time player" in lowered or "first time player" in lowered
    assert "do not know" in lowered
    assert "screenshot" in lowered
    assert "memory" in lowered
    assert "ground truth" in lowered
    # It must accept hypotheses/tasks but never invent the plot.
    assert "hypothesis" in lowered
    assert "never invent" in lowered or "do not invent" in lowered


def test_prompt_has_honesty_levels_and_tools() -> None:
    lowered = SYSTEM_PROMPT.lower()
    for token in ("verified", "inferred", "hypothesis"):
        assert token in lowered
    for tool in (
        "observe",
        "advance_dialog",
        "choose_option",
        "record_fact",
        "propose_task",
        "finish",
    ):
        assert tool in SYSTEM_PROMPT
    assert "never promote" in HONESTY_GUIDE.lower() or "never" in HONESTY_GUIDE.lower()


def test_story_package_source_has_no_spoilers_or_hidden_state() -> None:
    """The package itself (not just the prompt) must stay spoiler-free."""
    files = sorted(STORY_PACKAGE.glob("*.py"))
    assert files, "story package files not found"
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        hits = [word for word in _SPOILER_WORDS if _contains_word(text, word)]
        assert hits == [], f"{path.name} contains spoiler knowledge: {hits}"
        assert "pyboy" not in text, f"{path.name} must not import pyboy"
        assert "llm_agent.progress" not in text, f"{path.name} must not read hidden progress"
        assert "llm_agent import progress" not in text


def test_observation_rendering_mentions_real_screenshot_and_one_tool() -> None:
    text = render_observation(
        frame_index=12,
        evidence_ref="screens/story_00000012.png",
        analysis_summary="event=dialog changed=0.10",
        event_name="dialog",
        dialog_visible=True,
        turn=2,
        max_turns=8,
    )
    assert "frame=12" in text
    assert "screens/story_00000012.png" in text
    assert "exactly one tool call" in text.lower()
    assert "ground truth" in text.lower()
    assert "dialog" in text.lower()


def test_no_tool_nudge_is_explicit_and_actionable() -> None:
    text = render_no_tool_nudge().lower()
    assert "no tool call" in text
    assert "exactly one tool call" in text
