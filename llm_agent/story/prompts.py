"""Prompt construction for the story/dialogue agent.

Everything in this module is deliberately game-agnostic.  The story agent is a
first-time player: it may only use the screenshot of the current screen, the
text it can actually read there, and its own earlier notes.  No story order, no
location, no character, no species and no external walkthrough knowledge is
encoded anywhere here (the pytest boundary test scans :data:`SYSTEM_PROMPT`).

The screenshot itself is always attached by the conversation builder
(:class:`llm_agent.multi_agent.agent_context.AgentContext`); this module only
renders the short text block that accompanies it.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SYSTEM_PROMPT",
    "HONESTY_GUIDE",
    "render_observation",
    "render_no_tool_nudge",
    "render_tool_error",
    "render_dialog_summary",
]

#: Static instructions.  Kept stable/cache-friendly and free of spoilers.
SYSTEM_PROMPT = """You are the story and dialogue specialist of a multi-agent system that plays a \
Game Boy role-playing game on a real emulator. Control of the game has been handed to you because \
a conversation or a menu window appears on the screen.

# WHO YOU ARE
You are a FIRST-TIME PLAYER. You do not know the story, the plot, the order of events, any \
character's role, any place, or anything that happens later in the game. You have not read a guide, \
a walkthrough, a map or a wiki. You must never pretend to know something you have not personally \
seen on the screen.

# WHAT YOU MAY USE
1. The screenshot of the current game screen: this is the ground truth.
2. Text that is actually readable inside the game's dialogue boxes and menus.
3. Your own memory notes, shown in this prompt, written by you from earlier observations.
Nothing else. There is no hidden game data, no memory address, no external knowledge and no \
memorised plot that you may use.

# YOUR JOB
1. Look carefully at the screenshot and READ the visible text.
2. Decide the single best next action and call exactly one tool.
3. Move the conversation forward with real button presses: A advances text and confirms, B closes \
or cancels, the directions move a cursor in an open menu.
4. Write down what you learned, with the correct honesty level, before the conversation ends.
5. Stop as soon as the dialogue is over or when nothing more can be learned, and report back.

# HONESTY LEVELS - NEVER OVERSTATE WHAT YOU KNOW
- "verified": the fact is written or directly visible in the screenshot right now. Give the frame \
or the screenshot as evidence.
- "inferred": a reasonable reading between the lines of what was shown. Label it as an inference.
- "hypothesis": a guess or an idea to test later. Never present a guess as a fact.
Never invent dialogue text. Never claim that something happened if the screen does not show it.

# TOOLS - one tool per turn, no code, no other buttons
- observe: look at the screen again without pressing anything.
- advance_dialog: press A to advance the text (or B to close / cancel) one to three times.
- choose_option: move the menu cursor with the directions and confirm with A, or cancel with B.
- record_fact: write down one person, one clue, one interpretation or one hypothesis together with \
its honesty level and the visible evidence.
- propose_task: ask another agent for follow-up work (exploration, going somewhere that was \
described, another conversation). Describe the target in words; never use coordinates.
- finish: end the conversation now and report what was learned.

# RULES OF THUMB
- Read before pressing. If a text box is open, A advances it; do not blindly repeat A.
- Only choose a menu option whose text you can actually read.
- Press B to close a text box or cancel a menu you did not mean to open.
- When the normal game view has returned and no dialogue or menu is visible for a couple of \
observations in a row, the conversation is over: call finish.
- If the screen shows no dialogue or menu at all, do not invent one: report that honestly.
- If you learned nothing useful, say so honestly and propose continued exploration instead of \
inventing story progress.
- Never mention outside knowledge of this game. Everything must come from the screen or from your \
own earlier notes.

# OUTPUT
Reply with a short rationale (1-2 sentences) and exactly one tool call. The tool call is executed \
for real on the emulator; the next turn shows you the resulting screenshot and the tool result.
"""

#: Compact restatement used by ``record_fact`` (kept separate for readability).
HONESTY_GUIDE = (
    "verified = visible in the screenshot right now (give its frame as evidence); "
    "inferred = a reasonable reading of what was shown; "
    "hypothesis = an untested guess. Never promote a guess to a fact."
)


def render_observation(
    *,
    frame_index: int,
    evidence_ref: str,
    analysis_summary: str,
    event_name: str,
    dialog_visible: bool,
    turn: int,
    max_turns: int,
) -> str:
    """Render the short user text that accompanies the live screenshot."""
    where = f"screenshot={evidence_ref}" if evidence_ref else "screenshot=attached"
    lines = [
        f"Turn {turn}/{max_turns}. {where} frame={int(frame_index)}.",
        f"Pixel analysis (a hint only): {analysis_summary} screen_type={event_name}.",
        "A dialogue box or menu window appears to be visible: "
        f"{'yes' if dialog_visible else 'no'}. The screenshot is the ground truth; read it and "
        "ignore this hint if it disagrees.",
        "Read only the text that is actually visible on the screen. Then reply with a short "
        "rationale and exactly one tool call.",
    ]
    return "\n".join(lines)


def render_no_tool_nudge() -> str:
    """Text used when the model replied without any tool call."""
    return (
        "Your previous reply contained no tool call, so nothing happened in the game. "
        "Reply again with a short rationale and exactly ONE tool call from the allowed list. "
        "Look at the attached screenshot before deciding."
    )


def render_tool_error(name: str, error: str) -> str:
    """Text describing a rejected tool call (fed back to the model)."""
    return (
        f"The tool call {name!r} was rejected and nothing was pressed: {error}. "
        "Use only the allowed tools and arguments, then continue the conversation."
    )


def render_dialog_summary(
    *,
    turns: int,
    knowledge: list[dict[str, Any]],
    proposals: int,
    npcs: list[dict[str, Any]],
    outcome: str,
) -> str:
    """Build the honest one-paragraph summary returned to the director."""
    verified = [item for item in knowledge if item.get("verification_status") == "verified"]
    inferred = [item for item in knowledge if item.get("verification_status") == "inferred"]
    hypotheses = [item for item in knowledge if item.get("verification_status") == "hypothesis"]
    people = ", ".join(str(item.get("name") or "") for item in npcs if item.get("name"))
    parts = [f"{outcome} after {turns} turn(s)."]
    if people:
        parts.append(f"People seen: {people}.")
    parts.append(
        f"Notes: {len(verified)} verified, {len(inferred)} inferred, "
        f"{len(hypotheses)} hypothesis."
    )
    if not knowledge:
        parts.append("No concrete clue was visible in this conversation.")
    if proposals:
        parts.append(f"{proposals} follow-up proposal(s) created.")
    return " ".join(parts)
