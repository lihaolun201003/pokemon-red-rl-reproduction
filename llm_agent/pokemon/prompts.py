"""Blind battle prompts for the autonomous Pokémon battle agent.

The battle agent is deliberately blind: on every turn it may use **only**

1. the real screenshot of the emulated Game Boy screen it is sent,
2. its own recorded battle history / lessons,
3. the generic Game Boy control scheme (directions move a cursor, A confirms,
   B cancels).

It must not use species data, stats, type match-ups, walkthroughs, "best move"
tables or any other outside knowledge, and it cannot read hidden game memory.
Everything here is therefore game-agnostic text: no creature names, no move
names, no location names, no story knowledge.  The battle prompt is static so
the API prompt cache keeps working; the per-turn state is rendered by
:func:`render_battle_turn_prompt` and sent as a user message next to the image.

The deterministic menu driver (see :mod:`llm_agent.pokemon.agent`) is the only
component that turns a decision into button presses; the model just names the
action/slot it wants, based on what the screenshot shows.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Short role label used in prompts / logs (never a game-specific term).
POKEMON_ROLE: str = "battle pilot"

#: Human-readable explanation of the four visible per-turn menu states.
MENU_LAYOUT_HINTS: dict[str, str] = {
    "main": (
        "The main battle options are visible as a small grid. Choose fight / "
        "switch / item / run; the driver moves the cursor to that option and confirms."
    ),
    "move": (
        "A vertical list of move slots is visible. Choose fight with a move_slot "
        "(1 = top row of the visible list) and the driver moves down and confirms."
    ),
    "text": (
        "A message box / animation is on screen, not a menu yet. The runtime "
        "advances it and re-observes, so no action is needed for it."
    ),
    "unknown": (
        "The screen does not clearly show a menu. The runtime re-observes instead "
        "of pressing anything blindly."
    ),
}

#: The static, cache-friendly system prompt of the battle agent.
SYSTEM_PROMPT: str = """\
You are the battle pilot of a blind LLM player inside a Game Boy monster-battling game.
You are not a chatbot here: on every turn you look at the current screenshot of the real
emulated screen and choose exactly ONE battle action through the `battle_action` tool.
A deterministic driver then turns your decision into real button presses and verifies
that the screen changed.

# WHAT YOU MAY USE
- The screenshot you receive now. It is the ground truth.
- The pixel-only screen analysis text sent with it.
- Your own battle history and lessons (they are part of this prompt).
- The generic Game Boy interface mechanics: direction keys move a menu cursor, A confirms,
  B cancels; the runtime handles message boxes and animations for you.
That is all. You may NOT use outside knowledge: no creature encyclopedias, no
species data, no stats, no type match-ups, no walkthroughs, no lists of "best" moves. You cannot
read hidden game memory, so never ask for or claim hidden values.

# HONESTY RULES
1. Only describe what is visibly readable on the screen: move names shown in the move
   menu, damage text, HP text or HP bars, battle messages, and the visible menu options.
   If something is not readable, leave it empty.
2. Never invent hidden information: no hidden enemy HP numbers, no hidden enemy moves,
   no hidden stats. If no HP bar or HP text is visible, do not claim any value.
3. Put anything that is a guess into `hypothesis` (for example "this move seems to be
   effective against this opponent"). Hypotheses are stored as INFERRED, observations
   as VERIFIED. Never dress a guess up as an observation.
4. Never claim the battle is over unless the screen visibly shows it. Use `finish` only
   then, and report `outcome` only when you can read it from the screen, else "unknown".

# VISIBLE BATTLE INTERFACE (mechanics only; learn every detail from the screen)
- The main battle options appear as a small grid of choices. The driver moves the cursor
  to the option you name and presses A.
- Choosing `fight` may open a vertical list of move slots; the driver moves the cursor
  down to the slot you name and presses A.
- Choosing `catch` or `item` opens the visible item list; name the visible row number
  you want (`item_slot`, 1 = first visible row) and the driver selects that row.
- Choosing `switch` opens the visible party list; name the visible row (`party_slot`).
- Between turns the game may show long message boxes and animations. The runtime
  advances those for you and re-observes the screen; do not describe their every line.

# DECISION PROTOCOL
Reply with exactly ONE `battle_action` tool call per turn (a short `reason` is welcome):
- `action`: fight | run | catch | item | switch | observe | finish
- `move_slot` 1-4 for fight; `item_slot` 1-8 for catch/item; `party_slot` 1-6 for switch.
- `visible_observations`: optional object that may contain only `move_name`,
  `damage_text`, `hp_text`, `battle_text`, `enemy_hp_bar`, `my_hp_bar` — and only the
  values you can actually read from the screenshot.
- `hypothesis`: optional one-line inference; it is stored as INFERRED, never as fact.
- `outcome` (finish only): win | loss | caught | ran | unknown.
Use `observe` only when the screen is unreadable or animating and a fresh look is needed;
it costs a turn. Use `finish` only when the battle is visibly finished.
You may also call `record_lesson` (verification: verified | inferred) to save a durable
lesson, but it never replaces the action for this turn.

# COST AND LOOP DISCIPLINE
One action per turn keeps the API cost bounded. If a plan keeps failing, change the plan
instead of repeating it. Stay calm and factual: a human watching reads your short reason.
"""

#: Backwards-friendly alias (other modules/tests import either name).
BATTLE_SYSTEM_PROMPT: str = SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Per-turn rendering
# --------------------------------------------------------------------------- #

def render_visible_observations(observations: Mapping[str, Any] | None) -> str:
    """Render the model's previously reported visible facts as compact bullets."""
    if not observations:
        return ""
    lines: list[str] = []
    labels = {
        "move_name": "move name on screen",
        "damage_text": "damage text on screen",
        "hp_text": "HP text on screen",
        "battle_text": "battle message on screen",
        "enemy_hp_bar": "enemy HP bar (visible shape only)",
        "my_hp_bar": "own HP bar (visible shape only)",
    }
    for key, label in labels.items():
        value = observations.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"- {label}: {value.strip()[:200]}")
    if not lines:
        return ""
    return "\n".join(["Visible facts you reported:", *lines])


def render_battle_turn_prompt(
    *,
    turn: int,
    max_turns: int,
    screen_summary: str,
    menu_layout: str,
    last_action: str = "",
    layout_hint: str = "",
    extra_notes: str = "",
    visible_observations: Mapping[str, Any] | None = None,
) -> str:
    """Build the text part of the user message for one battle turn.

    Parameters
    ----------
    turn, max_turns:
        Position in the hard turn budget (the model is told how many are left).
    screen_summary:
        One-line pixel-only description of the current frame.
    menu_layout:
        ``main`` / ``move`` / ``text`` / ``unknown`` (visible UI state).
    last_action:
        Compact report of what the previous decision actually did.
    layout_hint:
        Optional override for the generic hint (defaults to
        :data:`MENU_LAYOUT_HINTS`).
    extra_notes:
        Optional extra runtime notes (e.g. a warning about a failed press).
    visible_observations:
        Facts the model reported on a previous turn (repeated for continuity).
    """
    safe_turn = max(1, int(turn))
    safe_max = max(safe_turn, int(max_turns))
    layout = str(menu_layout or "unknown").strip().lower()
    hint = layout_hint or MENU_LAYOUT_HINTS.get(layout, MENU_LAYOUT_HINTS["unknown"])
    lines: list[str] = [
        f"## BATTLE TURN {safe_turn}/{safe_max}",
        f"Screen analysis: {str(screen_summary or 'unavailable').strip()[:400]}",
        f"Visible menu state: {layout}",
        f"Menu note: {hint}",
    ]
    if last_action:
        lines.append(f"Last action result: {str(last_action).strip()[:400]}")
    observed = render_visible_observations(visible_observations)
    if observed:
        lines.append(observed)
    if extra_notes:
        lines.append(f"Runtime notes: {str(extra_notes).strip()[:400]}")
    lines.append(
        "Decide exactly ONE action now and return it with the `battle_action` tool call. "
        "Only name slots/values you can see on the screenshot; put guesses in hypothesis."
    )
    return "\n".join(lines)


def render_battle_end_prompt(
    *,
    outcome: str,
    turns: int,
    visible_text: str = "",
    reason: str = "",
) -> str:
    """Short text used for the audit/summary report after a battle ended."""
    parts = [f"Battle finished after {max(0, int(turns))} turn(s); outcome: {outcome}."]
    if visible_text:
        parts.append(f"Visible text: {visible_text[:300]}")
    if reason:
        parts.append(f"Notes: {reason[:300]}")
    return " ".join(parts)


def render_resolution_outcome_prompt() -> str:
    """Prompt for the one extra 'read the result' call after a battle clears.

    The screenshot is the last visible screen of the battle resolution (the
    level-up / victory / caught / run-away message).  The model may only report
    what it can actually read there; otherwise the outcome stays ``unknown``.
    """
    return (
        "The battle is over and these are the last screens that were visible while the "
        "battle was resolving (post-battle messages may be shown). Read ONLY the text you "
        "can actually see in these screenshots. Then call the `finish` tool with the "
        "`outcome` that the visible text proves: win, loss, caught or ran. Visible "
        "experience-gain / level-up messages right after a battle normally mean you won; a "
        "visible black-out message means you lost; a visible escape message means you ran; "
        "a visible capture message means you caught it. If the visible text does not "
        "clearly prove the result, call finish with outcome \"unknown\" — never guess. Put "
        "any visible battle text you used into `visible_text` (or into "
        "visible_observations.battle_text) so it is recorded as a verified observation."
    )


__all__ = [
    "POKEMON_ROLE",
    "MENU_LAYOUT_HINTS",
    "SYSTEM_PROMPT",
    "BATTLE_SYSTEM_PROMPT",
    "render_visible_observations",
    "render_battle_turn_prompt",
    "render_battle_end_prompt",
    "render_resolution_outcome_prompt",
]
