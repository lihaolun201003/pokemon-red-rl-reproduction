"""Prompt-boundary tests: blind exploration only, no game knowledge anywhere.

The battle prompt must (a) tell the model it may only use the visible screenshot
and its own battle history, and (b) contain no specific creature/move/location
names or walkthrough knowledge.  These tests scan the source text, so a future
edit cannot silently bake game knowledge back in.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llm_agent.pokemon import prompts

PACKAGE = Path(__file__).resolve().parents[2] / "llm_agent" / "pokemon"
MODULES = {
    name: (PACKAGE / name).read_text(encoding="utf-8")
    for name in ("__init__.py", "agent.py", "memory.py", "prompts.py", "tools.py")
}

#: Locations / places / organisations from the game (must never be hard-coded).
FORBIDDEN_LOCATIONS = (
    "pallet town",
    "viridian",
    "pewter",
    "cerulean",
    "vermilion",
    "celadon",
    "fuchsia",
    "saffron",
    "cinnabar",
    "indigo plateau",
    "mt. moon",
    "mt moon",
    "rock tunnel",
    "ss anne",
    "s.s. anne",
    "lavender town",
    "oak's lab",
    "oaks lab",
    "professor oak",
    "poke mart",
    "poké mart",
    "pokemon center",
    "poké center",
    "gym",
    "gym leader",
    "badge",
    "pokédex",
    "pokedex",
)

#: Creatures / NPCs from the game (must never be hard-coded).
FORBIDDEN_CREATURES = (
    "bulbasaur",
    "ivysaur",
    "venusaur",
    "charmander",
    "charmeleon",
    "charizard",
    "squirtle",
    "wartortle",
    "blastoise",
    "caterpie",
    "metapod",
    "butterfree",
    "weedle",
    "kakuna",
    "beedrill",
    "pidgey",
    "pidgeotto",
    "pidgeot",
    "rattata",
    "raticate",
    "spearow",
    "fearow",
    "ekans",
    "arbok",
    "sandshrew",
    "sandslash",
    "nidoran",
    "nidorino",
    "nidoking",
    "nidorina",
    "nidoqueen",
    "vulpix",
    "ninetales",
    "jigglypuff",
    "wigglytuff",
    "zubat",
    "golbat",
    "oddish",
    "gloom",
    "vileplume",
    "paras",
    "parasect",
    "venonat",
    "venomoth",
    "diglett",
    "dugtrio",
    "meowth",
    "persian",
    "psyduck",
    "golduck",
    "mankey",
    "primeape",
    "growlithe",
    "arcanine",
    "poliwag",
    "poliwhirl",
    "poliwrath",
    "abra",
    "kadabra",
    "alakazam",
    "machop",
    "machoke",
    "machamp",
    "bellsprout",
    "weepinbell",
    "victreebel",
    "tentacool",
    "tentacruel",
    "geodude",
    "graveler",
    "golem",
    "ponyta",
    "rapidash",
    "slowpoke",
    "slowbro",
    "magnemite",
    "magneton",
    "doduo",
    "dodrio",
    "seel",
    "dewgong",
    "grimer",
    "muk",
    "shellder",
    "cloyster",
    "gastly",
    "haunter",
    "gengar",
    "onix",
    "drowzee",
    "hypno",
    "krabby",
    "kingler",
    "voltorb",
    "electrode",
    "exeggcute",
    "exeggutor",
    "cubone",
    "marowak",
    "hitmonlee",
    "hitmonchan",
    "lickitung",
    "koffing",
    "weezing",
    "rhyhorn",
    "rhydon",
    "chansey",
    "tangela",
    "kangaskhan",
    "horsea",
    "seadra",
    "goldeen",
    "seaking",
    "staryu",
    "starmie",
    "scyther",
    "jynx",
    "electabuzz",
    "magmar",
    "pinsir",
    "tauros",
    "magikarp",
    "gyarados",
    "lapras",
    "ditto",
    "eevee",
    "vaporeon",
    "jolteon",
    "flareon",
    "porygon",
    "omanyte",
    "omastar",
    "kabuto",
    "kabutops",
    "aerodactyl",
    "snorlax",
    "articuno",
    "zapdos",
    "moltres",
    "dratini",
    "dragonair",
    "dragonite",
    "mewtwo",
    "mew",
    "pikachu",
    "raichu",
    "clefairy",
    "clefable",
    "vileplume",
    "brock",
    "misty",
    "erika",
    "koga",
    "sabrina",
    "blaine",
    "giovanni",
)

#: Move names from the game (must never be hard-coded as a "best move" table).
FORBIDDEN_MOVES = (
    "tackle",
    "growl",
    "tail whip",
    "ember",
    "water gun",
    "vine whip",
    "thunder shock",
    "thundershock",
    "scratch",
    "poison sting",
    "quick attack",
    "sand attack",
    "string shot",
    "leech seed",
    "razor leaf",
    "harden",
    "fury attack",
    "horn attack",
    "wing attack",
    "confuse ray",
    "sleep powder",
    "stun spore",
    "bubble",
    "gust",
    "peck",
    "absorb",
    "sing",
    "disable",
    "confusion",
    "bite",
    "splash",
    "glare",
    "solar beam",
    "hyper beam",
    "flamethrower",
    "surf",
    "thunderbolt",
    "ice beam",
    "earthquake",
    "psychic",
)

#: Term groups that are scanned in the prompt text only (move names would be
#: noise in code that talks about "moves" generically).
_PROMPT_MOVE_TERMS = FORBIDDEN_MOVES


def _matches(text: str, term: str) -> bool:
    """Word-boundary match so 'ember' does not fire inside 'member'."""
    pattern = r"(?<![0-9a-z])" + re.escape(term) + r"(?:s|es)?(?![0-9a-z])"
    return re.search(pattern, text) is not None


@pytest.mark.parametrize("module_name", sorted(MODULES))
def test_modules_contain_no_location_or_creature_knowledge(module_name: str) -> None:
    text = MODULES[module_name].lower()
    found = [term for term in FORBIDDEN_LOCATIONS + FORBIDDEN_CREATURES if _matches(text, term)]
    assert not found, f"{module_name} hard-codes game knowledge: {found}"


@pytest.mark.parametrize("module_name", sorted(MODULES))
def test_modules_contain_no_move_knowledge(module_name: str) -> None:
    text = MODULES[module_name].lower()
    found = [term for term in _PROMPT_MOVE_TERMS if _matches(text, term)]
    assert not found, f"{module_name} hard-codes move knowledge: {found}"


def test_system_prompt_declares_the_blind_input_boundary() -> None:
    lowered = " ".join(prompts.SYSTEM_PROMPT.lower().split())
    for phrase in (
        "screenshot",
        "battle history",
        "outside knowledge",
        "species data",
        "type match-ups",
        "walkthroughs",
        "cannot",
        "hidden",
        "verified",
        "inferred",
        "battle_action",
        "record_lesson",
        "finish",
    ):
        assert phrase in lowered, f"system prompt is missing {phrase!r}"


def test_system_prompt_forbids_hidden_values_and_presets() -> None:
    lowered = " ".join(prompts.SYSTEM_PROMPT.lower().split())
    assert "never invent hidden information" in lowered
    assert "best" in lowered  # explicitly forbids "best move" tables
    assert "one" in lowered and "action" in lowered
    # the deterministic driver is described as an executor, not as the brain
    assert "driver" in lowered


def test_menu_layout_hints_cover_every_layout() -> None:
    assert set(prompts.MENU_LAYOUT_HINTS) == {"main", "move", "text", "unknown"}
    for hint in prompts.MENU_LAYOUT_HINTS.values():
        assert hint.strip()


def test_turn_prompt_renders_context_without_game_knowledge() -> None:
    text = prompts.render_battle_turn_prompt(
        turn=2,
        max_turns=10,
        screen_summary="event=menu",
        menu_layout="move",
        last_action="presses=['a']; verified",
        visible_observations={"move_name": "a readable label", "damage_text": "visible text"},
        extra_notes="L1 recovery strategy note: try a different option",
    )
    lowered = text.lower()
    assert "battle turn 2/10" in lowered
    assert "move" in lowered
    assert "last action result" in lowered
    assert "visible facts" in lowered
    assert "recovery" in lowered
    assert "battle_action" in lowered
    for term in FORBIDDEN_LOCATIONS + FORBIDDEN_CREATURES + FORBIDDEN_MOVES:
        assert not _matches(lowered, term), term


def test_resolution_outcome_prompt_is_honest() -> None:
    text = prompts.render_resolution_outcome_prompt().lower()
    assert "finish" in text
    assert "unknown" in text
    for outcome in ("win", "loss", "caught", "ran"):
        assert outcome in text
    assert "never guess" in text


def test_battle_end_prompt_is_compact_and_plain() -> None:
    text = prompts.render_battle_end_prompt(
        outcome="unknown", turns=3, visible_text="a visible line", reason="screen cleared"
    )
    assert "outcome" in text and "3 turn" in text
    assert "a visible line" in text
