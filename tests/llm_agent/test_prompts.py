"""Tests for prompt construction and the scripted dry-run brain (no network/API key)."""

from __future__ import annotations

from types import SimpleNamespace

from llm_agent import contracts, prompts
from llm_agent.config import MemoryConfig
from llm_agent.contracts import Frame, HistoryEntry, RuntimeState, ScreenEvent
from llm_agent.memory import AgentMemory


def make_analysis(signature: str = "abcdef1234567890", event: ScreenEvent = ScreenEvent.NONE) -> SimpleNamespace:
    return SimpleNamespace(
        event=event,
        changed=True,
        changed_ratio=0.42,
        signature=signature,
        is_black=False,
        luma_mean=97.3,
        summary=f"event={event.value} changed=0.42 sig={signature[:8]} luma=97/255",
        stats={},
    )


def test_system_prompt_is_static_and_safe() -> None:
    assert prompts.SYSTEM_PROMPT.strip()
    lowered = prompts.SYSTEM_PROMPT.lower()
    assert "blind" in lowered
    assert "screenshot" in lowered
    assert "record_discovery" in lowered
    assert "never" in lowered
    assert "pallet" not in lowered and "badge" not in lowered and "gym" not in lowered
    assert prompts.system_prompt_chars() == len(prompts.SYSTEM_PROMPT)


def test_build_user_text_contains_every_context_section(sample_rgba, make_frame) -> None:
    frame = make_frame(sample_rgba, frame_index=123)
    state = RuntimeState(run_id="test", goal="Find the exit", goal_reason="explore")
    state.llm_turns = 4
    state.buttons_pressed = 17
    state.action_batches = 3
    state.add_history("tool", "walk up: moved 3 tiles")
    state.add_history("tool", "press a: dialogue opened")

    memory = AgentMemory(MemoryConfig(root=str(__import__("tempfile").mkdtemp())))
    memory.load()
    memory.set_goal("Find the exit", "explore")
    memory.record_discovery("A door at the north wall.", kind="place", place="Lab")
    context = memory.render_context("exit", max_chars=2000)

    text = prompts.build_user_text(
        trigger="event:dialog",
        frame=frame,
        analysis=make_analysis(event=ScreenEvent.DIALOG),
        state=state,
        memory_context=context,
        guard_text="GUARD WARNINGS (1):\n- [warning] stuck: no movement",
        last_outcome_text="walk up: stopped because dialog",
        history_limit=5,
    )
    assert "CURRENT GOAL" in text and "Find the exit" in text
    assert "RECENT HISTORY" in text and "walk up" in text
    assert "LAST ACTION BATCH RESULT" in text and "stopped because dialog" in text
    assert "LOOP GUARD" in text and "GUARD WARNINGS" in text
    assert "SCREEN ANALYSIS" in text and "event=dialog" in text
    assert "YOUR LONG-TERM MEMORY" in text and "A door at the north wall." in text
    assert "SCREENSHOT" in text
    assert "Find the exit" in text
    # nothing secret / hidden-state related may appear
    lowered = text.lower()
    for forbidden in ("0xd3", "map_id", "badge", "hidden flag", "event flag"):
        assert forbidden not in lowered


def test_build_messages_places_image_last_and_same_system_prompt() -> None:
    png = b"\x89PNG\r\n\x1a\nfake"
    messages = prompts.build_messages(user_text="hello", image_png=png, mime="image/png", detail="original")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompts.SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    blocks = messages[1]["content"]
    assert blocks[0]["type"] == "text" and blocks[0]["text"] == "hello"
    assert blocks[-1]["type"] == "image_url"
    url = blocks[-1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert blocks[-1]["image_url"]["detail"] == "original"


def test_render_history_is_clipped() -> None:
    entries = [HistoryEntry(kind="tool", summary="x" * 500) for _ in range(3)]
    rendered = prompts.render_history(entries, limit=2)
    assert rendered.count("\n") == 1
    assert all(len(line) < 260 for line in rendered.splitlines())


def test_build_user_text_includes_mental_map(sample_rgba, make_frame) -> None:
    frame = make_frame(sample_rgba, frame_index=1)
    state = RuntimeState(run_id="map-test", goal="explore")
    base = dict(
        trigger="after_actions",
        frame=frame,
        analysis=make_analysis(),
        state=state,
        memory_context="",
        guard_text="",
        last_outcome_text="",
    )
    text = prompts.build_user_text(**base, mental_map_text="@··#\n???\nblocked: up")
    assert "MENTAL MAP" in text
    assert "@··#" in text
    # an empty map must not produce an empty section
    assert "MENTAL MAP" not in prompts.build_user_text(**base)


def test_scripted_brain_returns_valid_tool_calls(sample_rgba, make_frame) -> None:
    from llm_agent.scripted import ScriptedBrain

    brain = ScriptedBrain()
    frame = make_frame(sample_rgba)
    state = RuntimeState(run_id="scripted")
    analysis = make_analysis(event=ScreenEvent.DIALOG)
    prepared = brain.prepare(frame=frame, analysis=analysis, state=state, memory=None, guard=None,
                             last_outcome_text="", trigger="start", previous_frame=None)
    reply = brain.complete(prepared)
    assert reply.model == "scripted"
    assert reply.tool_calls
    assert all(isinstance(call, contracts.ToolCall) for call in reply.tool_calls)
    assert reply.tool_calls[0].name == "execute_actions"  # dialog -> press A x3
    assert reply.tool_calls[0].arguments["actions"] == ["a", "a", "a"]
    assert brain.mode == "scripted"
