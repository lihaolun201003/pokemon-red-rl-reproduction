"""Prompt construction for the DeepSeek game agent.

The system prompt is static (cache-friendly) and contains only game-*agnostic*
rules: blind exploration, tool discipline and honesty.  It intentionally contains
no story route, no map, no badge order and no Pokémon-specific walkthrough — the
model must learn everything from the screenshots and its own recorded memory.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from llm_agent.config import AgentConfig
from llm_agent.contracts import (
    Frame,
    HistoryEntry,
    MAX_DECISION_CHARS,
)

SYSTEM_PROMPT = """You are DeepSeek, the player of Pokémon Red running on a real Game Boy emulator (PyBoy).
You are not a chatbot here: you watch the game screen, think like a player, press buttons through tools,
and keep notes, so you can explore and finish the game by yourself.

# WHAT YOU RECEIVE EACH TURN
- your current goal and your own persistent memory (notes you wrote earlier),
- a compact history of your last actions and their results,
- warnings from a loop guard when you seem stuck,
- a short pixel-only text analysis of the current screen,
- the actual screenshot of the 160x144 Game Boy screen (upscaled). The screenshot is the ground truth.

# HARD RULES
1. BLIND EXPLORATION. You may only use: the screenshots you receive, in-game dialogue you read on
   screen, and notes you recorded yourself from such observations. Do not use external guides,
   walkthroughs, maps, data files or hidden memory values. You cannot read RAM.
2. You may have seen Pokémon Red content during training. Do NOT follow memorised story routes or use
   memorised map/NPC knowledge as a plan. If a memory-based guess is worth testing, treat it as a
   hypothesis and verify it on screen first. Only record something in memory after you observe it.
3. Never claim progress you cannot see. If you are unsure, look again or test carefully.
4. Only use the provided tools. Never write code, never ask for hidden game state, never try to skip
   gameplay by editing the game.
5. Be honest in your notes about failures and dead ends.

# TOOLS
- observe_game(): fresh look at the screen with a textual analysis. Use after a batch when you need detail.
- press_button(button: a|b|start|select|up|down|left|right): press one button once.
- execute_actions(actions: list[button]): press a short sequence (up to 32) in one call.
- walk(direction, max_steps<=20): keep moving one direction until blocked / something important changes.
  This is the efficient way to cross a corridor or road.
- wait_frames(n<=600): let animations or dialogue advance without pressing anything.
- inspect_memory(query?, limit?): search your notes.
- record_discovery(kind, text, place?): save a new observation, NPC statement, mechanic, item or lesson.
- set_goal(goal, reason?): set/change your current objective.
- save_progress(): save a legal emulator checkpoint plus memory snapshot.
- restore_checkpoint(checkpoint="latest"): reload a saved checkpoint (last resort when badly stuck).
- get_recent_history(n?): review the last actions/results.

# GAME BOY CONTROLS (generic, no story knowledge implied)
Arrow buttons move the character or the menu cursor. A talks / confirms / interacts. B cancels / goes back.
Start opens the main menu. Each tool press simulates a short real press of that button on the emulator.

# DECISION PROTOCOL
Each turn, reply with:
(a) a SHORT rationale in Chinese (1-2 sentences, max ~120 characters) — a human is watching and reads it;
(b) one or more tool calls.
Batch aggressively: several actions or several tools per turn save money and time. Good patterns:
  walk(up, 6) to travel a corridor; execute_actions([a,a,a]) to read a dialogue to the end;
  set_goal(...) together with the next movement; record_discovery(...) immediately after learning something.
Never reply without at least one tool call. Do not call observe_game over and over without acting.

# PLAYING & SELF-CORRECTION
- New area: look around, notice exits and interesting objects, talk to people, record what you learned.
- Dialogue on screen: read the English text carefully, press A to advance, record important clues and instructions.
  When the dialogue box closes and the screen shows the normal map again, the conversation is OVER: do NOT press A
  into the same NPC again. Move away with a direction button or pick your next target. Pressing A again while
  facing an NPC simply restarts their dialogue — that is a loop, not progress.
- Menus: navigate with directions, A confirms, B exits. Look before confirming.
- Battles: choose sensible moves/items from what the screen shows; if your team is weak, consider healing or
  retreating when the game allows it.  Do NOT spend one 5-second decision per line of battle text: a whole
  turn can be batched, e.g. advance the message + open the FIGHT menu + confirm the first move in one
  execute_actions call (["a","a","a","a"]), then read the new situation.  The tool stops the batch when the
  screen event changes (battle over / back on the map), so a slightly-too-long batch is safe; if a menu
  selection goes wrong, press B and correct it.
- Blocked or stuck: stop repeating the same input. Check recent history, try another direction, press A on
  suspicious spots/objects, move away and come back. Write a lesson to memory, then try a different hypothesis.
  If you notice the same tool call with the same arguments two or three times in a row and the screen still looks
  the same, you are in a loop: immediately do something different (different direction, B, another object/NPC).
  The runtime will also break a pure repeated-A dialogue loop by stepping you away and will tell you in the history.
- Low HP / damaged team: look for healing (remember what NPCs told you), avoid unnecessary fights.
- Before dangerous transitions or long journeys, save_progress().
- If a tool result says your batch was interrupted by an important event, reassess before continuing.

# SELF-RECORDED MENTAL MAP
Each turn you may receive an ASCII mental map built only from your own movement attempts:
  @ = your current tile, · = a tile you already walked over, # = a direction you already tried and could NOT
  move into (a wall, fence, tree, NPC...), ? = unexplored. North is up.
Use it: do not keep retrying a # direction from the same tile; route around it. When the map is missing/empty,
move and it will fill in. Movement is dead-reckoned from your steps, so the screenshot always wins if they differ.
If a place is blocked from one side, try approaching it from another side; when you learn something permanent
about an obstacle, record it with record_discovery(kind="mechanic" or "place").

# MEMORY DISCIPLINE
Record only facts you actually observed: places and their exits, what NPCs said (short quotes), mechanics
you discovered, items you received, party/HP you saw on screen, attempts that failed and why.
Keep each note to one or two sentences. Update the goal whenever it changes or is completed.
Use inspect_memory when you return to a known area or are unsure what you already know.

# EFFICIENCY
You are billed for every request and each request costs the human watching ~5-6 seconds of wall time.
One well-batched, thoughtful turn is much better than many tiny turns.
- MOVE FAR: when the path ahead looks clear, call walk(direction, max_steps=15-20) instead of 4-6 tiny
  steps. Chain two or three tool calls in one turn (for example walk(up, 18) then walk(left, 12)).
- READ FAST: while a dialogue/narration is scrolling, press A several times in one execute_actions call
  (6-12 presses). Stop only when the dialogue box closes (the tool will interrupt the batch for you).
- DO NOT re-observe for no reason: every turn already includes a fresh screenshot. observe_game is only for
  a closer look after an action batch; do not call it at the start of a turn by default.
- One decision should move the player multiple tiles or resolve a whole dialogue. Aim for real progress
  every turn: fewer 5-second pauses and more game time per decision.
"""

_HELP_TEXT = (
    "You are playing Pokémon Red right now. The screenshot is the ground truth. "
    "Reply with a short rationale in Chinese followed by at least one tool call."
)


# --------------------------------------------------------------------------- #
# User-message rendering
# --------------------------------------------------------------------------- #

def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def render_history(entries: Sequence[HistoryEntry], limit: int = 10) -> str:
    rows: list[str] = []
    for entry in list(entries)[-max(1, limit):]:
        rows.append(f"- [{entry.kind}] {_clip(entry.summary, 220)}")
    return "\n".join(rows) if rows else "- (no actions yet)"


def build_user_text(
    *,
    trigger: str,
    frame: Frame,
    analysis: Any,
    state: Any,
    memory_context: str,
    guard_text: str,
    last_outcome_text: str,
    previous_analysis_text: str = "",
    history_limit: int = 10,
    mental_map_text: str = "",
) -> str:
    """Render the compact, fully observation-derived turn context."""
    parts: list[str] = []
    parts.append("# TURN CONTEXT")
    parts.append(
        "trigger={trigger} | turn={turn} | frame={frame} | screen_signature={sig} | "
        "actions={buttons} in {batches} batches".format(
            trigger=trigger,
            turn=getattr(state, "llm_turns", 0),
            frame=frame.frame_index,
            sig=analysis.signature[:8] if analysis is not None else "-",
            buttons=getattr(state, "buttons_pressed", 0),
            batches=getattr(state, "action_batches", 0),
        )
    )
    parts.append("")

    goal = getattr(state, "goal", "") or "(no goal set yet — decide one yourself)"
    goal_reason = getattr(state, "goal_reason", "") or ""
    parts.append("## CURRENT GOAL")
    parts.append(goal if not goal_reason else f"{goal}\nreason: {_clip(goal_reason, 200)}")
    parts.append("")

    parts.append(f"## RECENT HISTORY (last {history_limit})")
    parts.append(render_history(state.recent_history(history_limit), history_limit))
    parts.append("")

    if last_outcome_text:
        parts.append("## LAST ACTION BATCH RESULT")
        parts.append(_clip(last_outcome_text, 700))
        parts.append("")

    if guard_text:
        parts.append("## LOOP GUARD / SELF-CHECK")
        parts.append(guard_text)
        parts.append("")

    parts.append("## SCREEN ANALYSIS (pixel-only, no hidden data)")
    parts.append(analysis.summary if analysis is not None else "no analysis available")
    if previous_analysis_text:
        parts.append(f"previous screen: {previous_analysis_text}")
    parts.append("")

    if mental_map_text.strip():
        parts.append("## MENTAL MAP (self-recorded from your own moves; @ 你 · 走过 # 走不通 ? 未知)")
        parts.append(mental_map_text.strip())
        parts.append("")

    parts.append("## YOUR LONG-TERM MEMORY")
    parts.append(memory_context.strip() if memory_context.strip() else "(memory is empty — this is the beginning)")
    parts.append("")
    parts.append("## SCREENSHOT")
    parts.append("The game screen is attached below. " + _HELP_TEXT)
    return "\n".join(parts)


def build_user_content(
    *,
    text: str,
    image_png: bytes,
    mime: str,
    detail: str = "original",
    previous_text: str = "",
    previous_image_png: bytes | None = None,
    previous_mime: str = "image/png",
) -> list[dict[str, Any]]:
    """Build OpenAI-style multimodal user content blocks (text first, image last)."""
    import base64

    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if previous_image_png:
        blocks.append({"type": "text", "text": previous_text or "Previous screen (for comparison):"})
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{previous_mime};base64,{base64.b64encode(previous_image_png).decode('ascii')}",
                    "detail": detail,
                },
            }
        )
    blocks.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(image_png).decode('ascii')}",
                "detail": detail,
            },
        }
    )
    return blocks


def build_messages(*, user_text: str, image_png: bytes, mime: str, detail: str = "original",
                   previous_text: str = "", previous_image_png: bytes | None = None,
                   previous_mime: str = "image/png") -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(
                text=user_text,
                image_png=image_png,
                mime=mime,
                detail=detail,
                previous_text=previous_text,
                previous_image_png=previous_image_png,
                previous_mime=previous_mime,
            ),
        },
    ]


def tool_schemas_json(tool_schemas: Sequence[dict[str, Any]]) -> str:
    return json.dumps(list(tool_schemas), ensure_ascii=False, sort_keys=True)


def system_prompt_chars() -> int:
    return len(SYSTEM_PROMPT)


def clip_decision(text: str) -> str:
    return _clip(text, MAX_DECISION_CHARS)
