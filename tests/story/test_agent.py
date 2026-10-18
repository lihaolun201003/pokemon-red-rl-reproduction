"""End-to-end story-agent tests with a scripted router and fake emulator.

These are fast and fully offline: a fake emulator renders synthetic 160x144
frames, a fake router replays scripted ModelReply objects, and the real
observation/analysis and AgentMemory code paths are exercised.
"""

from __future__ import annotations

import base64
import os
import re

import pytest

from llm_agent.contracts import (
    ApiAuthenticationError,
    ApiResponseError,
    ApiUsage,
    Frame,
    ModelReply,
    ToolCall,
)
from llm_agent.multi_agent.contracts import Task, TaskType
from llm_agent.story.agent import is_dialog_context
from llm_agent.story.prompts import SYSTEM_PROMPT
from llm_agent.story.tools import STORY_TOOL_NAMES, TOOL_SCHEMAS

from tests.story.conftest import (
    FakeArbiter,
    FakeBus,
    FakeEmulator,
    FakeSharedMemory,
    build_agent,
    dialog_rgba,
    make_frame,
    multi_call_reply,
    scripted_reply,
)

FROZEN_RESULT_KEYS = {
    "status",
    "summary",
    "npcs",
    "discoveries",
    "tasks_created",
    "turns",
    "api_calls",
    "cost_usd",
    "evidence",
    "error",
}

IMAGE_BLOCK_RE = re.compile(r"^data:image/(png|jpeg);base64,(?P<data>[A-Za-z0-9+/=]+)$")


def _run(stack, task=None, max_turns: int = 12):
    return stack.agent.handle_dialog(emulator=stack.emulator, task=task, max_turns=max_turns)


def _last_user_images(messages) -> list[dict]:
    blocks: list[dict] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                blocks.append(block)
    return blocks


def _tool_protocol_errors(messages) -> list[str]:
    """Return chat-protocol violations: unanswered/orphaned ``tool_calls``.

    The OpenAI chat API rejects a request when an assistant message declares a
    ``tool_call`` that is not followed by a matching tool message, so every
    request the agent builds must satisfy this invariant.
    """
    errors: list[str] = []
    pending: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            if pending:
                errors.append(f"assistant at {index} before answering {sorted(pending)}")
            calls = message.get("tool_calls") or []
            pending = {str(call.get("id")) for call in calls}
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id"))
            if tool_call_id not in pending:
                errors.append(f"orphan tool message {tool_call_id} at {index}")
            pending.discard(tool_call_id)
        elif role == "user" and pending:
            errors.append(f"user message at {index} before answering {sorted(pending)}")
    if pending:
        errors.append(f"unanswered tool_calls at end: {sorted(pending)}")
    return errors


def _assert_all_requests_valid(router) -> None:
    for number, call in enumerate(router.calls):
        errors = _tool_protocol_errors(call["messages"])
        assert errors == [], f"request {number} would be rejected by the chat API: {errors}"


# --------------------------------------------------------------------------- #
# Frozen interface / status
# --------------------------------------------------------------------------- #

def test_frozen_interface_and_public_surface(make_agent) -> None:
    stack = make_agent(replies=[scripted_reply("finish", {"summary": "done"})])
    agent = stack.agent
    assert agent.handles() == ("dialog", "menu", "story")
    assert agent.system_prompt == SYSTEM_PROMPT
    assert agent.paused is False
    agent.pause()
    assert agent.paused is True
    agent.resume()
    assert agent.paused is False
    status = agent.status()
    assert status["agent"] == "story"
    assert status["handles"] == ["dialog", "menu", "story"]
    assert status["paused"] is False
    assert "memory" in status and "counters" in status
    assert isinstance(status["recovery_hints"], list)
    assert callable(agent.propose_tasks)
    assert callable(agent.on_event)
    assert callable(agent.apply_recovery_hint)


def test_result_dict_has_frozen_shape(make_agent) -> None:
    stack = make_agent(replies=[scripted_reply("finish", {"summary": "read it"})])
    result = _run(stack)
    assert set(result) == FROZEN_RESULT_KEYS
    assert result["status"] == "dialog_ended"
    assert isinstance(result["summary"], str) and result["summary"]
    assert isinstance(result["npcs"], list)
    assert isinstance(result["discoveries"], list)
    assert isinstance(result["tasks_created"], list)
    assert isinstance(result["turns"], int)
    assert isinstance(result["api_calls"], int)
    assert isinstance(result["cost_usd"], float)
    assert set(result["evidence"]) == {"screens", "frames", "dialog_texts"}


# --------------------------------------------------------------------------- #
# Real screenshot -> model, real presses -> emulator
# --------------------------------------------------------------------------- #

def test_real_screenshot_is_sent_to_the_model(make_agent) -> None:
    stack = make_agent(replies=[scripted_reply("finish", {"summary": "read"})])
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert stack.router.complete_calls == 1
    call = stack.router.calls[0]
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert "first-time player" in messages[0]["content"].lower()
    assert call["tools"] == TOOL_SCHEMAS
    assert [tool["function"]["name"] for tool in call["tools"]] == list(STORY_TOOL_NAMES)
    images = _last_user_images(messages)
    assert images, "the model must receive the real screenshot as an image block"
    match = IMAGE_BLOCK_RE.match(images[-1]["image_url"]["url"])
    assert match is not None
    payload = base64.b64decode(match.group("data"))
    assert payload.startswith(b"\x89PNG"), "screenshot must be a real encoded PNG"
    assert len(payload) > 100
    # Evidence was persisted as files.
    assert result["evidence"]["screens"]
    assert result["evidence"]["frames"]


def test_per_turn_tool_execution_presses_real_buttons(make_agent) -> None:
    stack = make_agent(
        replies=[
            scripted_reply("advance_dialog", {"button": "a", "times": 2}),
            scripted_reply(
                "choose_option", {"direction": "down", "presses": 2, "confirm": True}
            ),
            scripted_reply("finish", {"summary": "nothing else visible"}),
        ],
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=6),
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert result["turns"] == 3
    assert result["api_calls"] == 3
    assert [button for button, _frames in stack.emulator.presses] == [
        "a",
        "a",
        "down",
        "down",
        "a",
    ]
    assert result["cost_usd"] > 0.0


def test_dialogue_end_detected_after_two_overworld_observations(make_agent) -> None:
    stack = make_agent(
        replies=[scripted_reply("advance_dialog", {"button": "a", "times": 1})],
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=1),
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    # turn 1 advances the box; turn 2 observes the overworld for the second time.
    assert result["turns"] == 2
    assert result["api_calls"] == 2
    assert stack.emulator.last_screen == "overworld"
    assert "interrupt" not in result["summary"].lower()


def test_dialogue_detection_heuristic() -> None:
    dialog = make_frame(dialog_rgba(0), 1)
    from llm_agent.observation import analyze

    assert is_dialog_context(analyze(dialog, None)) is True


# --------------------------------------------------------------------------- #
# Facts vs inferences
# --------------------------------------------------------------------------- #

def test_verified_fact_with_dialogue_text_is_recorded_as_verified(make_agent) -> None:
    stack = make_agent(
        replies=[
            scripted_reply(
                "advance_dialog", {"button": "a", "times": 1}
            ),
            scripted_reply(
                "record_fact",
                {
                    "category": "story_clue",
                    "content": "A resident says the gate ahead is locked.",
                    "verification_status": "verified",
                    "confidence": 0.95,
                    "npc_name": "a resident",
                    "dialogue_text": "The gate ahead is locked, I am afraid.",
                },
            ),
            scripted_reply("finish", {"summary": "clue recorded"}),
        ],
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=6),
        with_shared=True,
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert len(result["discoveries"]) == 1
    discovery = result["discoveries"][0]
    assert discovery["verification_status"] == "verified"
    assert discovery["category"] == "story_clue"
    assert result["evidence"]["dialog_texts"] == ["The gate ahead is locked, I am afraid."]
    assert result["npcs"][0]["name"] == "a resident"
    assert stack.shared.entries[0].verification_status.value == "verified"
    # A verified clue becomes a concrete investigation proposal for the director.
    assert any(item["task_type"] == "investigate" for item in result["tasks_created"])


def test_unfounded_verified_claim_is_downgraded(make_agent) -> None:
    stack = make_agent(
        replies=[
            scripted_reply(
                "record_fact",
                {
                    "category": "story_clue",
                    "content": "Something dramatic is going to happen soon.",
                    "verification_status": "verified",
                },
            ),
            scripted_reply("finish", {"summary": "done"}),
        ],
        with_shared=True,
    )
    result = _run(stack)
    assert result["discoveries"][0]["verification_status"] == "inferred"
    assert stack.shared.entries[0].verification_status.value == "inferred"


# --------------------------------------------------------------------------- #
# Proposals and honest fallbacks
# --------------------------------------------------------------------------- #

def test_proposals_are_deduplicated(make_agent) -> None:
    duplicate = {
        "owner_agent": "navigation",
        "task_type": "talk",
        "description": "Talk again with the person near the door",
        "priority": "low",
        "payload": {"target": {"kind": "described", "description": "near the door"}},
    }
    stack = make_agent(
        replies=[
            scripted_reply("propose_task", duplicate),
            scripted_reply("propose_task", duplicate),
            scripted_reply("finish", {"summary": "done"}),
        ]
    )
    result = _run(stack)
    assert len(result["tasks_created"]) == 1
    assert result["tasks_created"][0]["description"] == duplicate["description"]
    first = stack.agent.propose_tasks()
    second = stack.agent.propose_tasks()
    assert [item["description"] for item in first] == [duplicate["description"]]
    assert [item["description"] for item in second] == [duplicate["description"]]
    filtered = stack.agent.propose_tasks(
        context={"existing_tasks": [{"description": duplicate["description"]}]}
    )
    assert filtered == []


def test_no_clue_proposes_continued_exploration(make_agent) -> None:
    stack = make_agent(
        replies=[scripted_reply("advance_dialog", {"button": "a", "times": 1})],
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=1),
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert result["discoveries"] == []
    assert "No concrete clue" in result["summary"] or "no concrete clue" in result["summary"].lower()
    explore = [item for item in result["tasks_created"] if item["task_type"] == "explore"]
    assert explore, "with no clue the agent must propose continued exploration"
    assert explore[0]["owner_agent"] == "navigation"
    assert "coordinates" not in str(explore[0]["payload"]).lower()
    assert result["tasks_created"] == stack.agent.propose_tasks()


# --------------------------------------------------------------------------- #
# Failure / budget / control paths
# --------------------------------------------------------------------------- #

def test_api_failure_returns_failed_without_crashing(make_agent) -> None:
    stack = make_agent(replies=[RuntimeError("boom")])
    result = _run(stack)
    assert result["status"] == "failed"
    assert "boom" in result["error"]
    assert result["api_calls"] == 0
    assert stack.emulator.presses == []


def test_budget_exhausted_returns_budget_before_any_call(make_agent) -> None:
    stack = make_agent(
        can_call_result=(False, "story reached max_calls=90"),
        replies=[scripted_reply("finish", {"summary": "never reached"})],
    )
    result = _run(stack)
    assert result["status"] == "budget"
    assert "max_calls" in result["error"]
    assert result["api_calls"] == 0
    assert stack.router.complete_calls == 0


def test_control_denied_when_arbiter_does_not_hold_lease(make_agent) -> None:
    stack = make_agent(
        replies=[scripted_reply("finish", {"summary": "never reached"})],
        arbiter=FakeArbiter(held=False),
    )
    result = _run(stack)
    assert result["status"] == "control_denied"
    assert result["api_calls"] == 0
    assert stack.router.complete_calls == 0
    assert stack.emulator.presses == []


def test_guarded_press_denial_maps_to_control_denied(make_agent) -> None:
    stack = make_agent(
        replies=[scripted_reply("advance_dialog", {"button": "a", "times": 1})],
        emulator=FakeEmulator(initial_screen="dialog", raise_control_denied=True),
    )
    result = _run(stack)
    assert result["status"] == "control_denied"
    assert "denied" in result["error"].lower() or "control" in result["error"].lower()
    assert stack.emulator.presses == []


def test_turn_budget_is_hard_capped(make_agent) -> None:
    stack = make_agent(
        replies=[],  # every turn returns the default observe reply
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=100),
    )
    result = _run(stack, max_turns=4)
    assert result["status"] == "budget"
    assert result["turns"] == 4
    assert result["api_calls"] == 4
    assert "turn budget" in result["error"]
    assert stack.emulator.presses == []


def test_paused_agent_returns_interrupted_without_touching_game(make_agent) -> None:
    stack = make_agent(replies=[scripted_reply("finish", {"summary": "never reached"})])
    stack.agent.pause()
    result = _run(stack)
    assert result["status"] == "interrupted"
    assert stack.router.complete_calls == 0
    assert stack.emulator.presses == []


def test_battle_screen_is_interrupted_before_any_call(make_agent) -> None:
    stack = make_agent(
        replies=[scripted_reply("finish", {"summary": "never reached"})],
        emulator=FakeEmulator(initial_screen="battle"),
    )
    result = _run(stack)
    assert result["status"] == "interrupted"
    assert result["api_calls"] == 0
    assert stack.emulator.presses == []


# --------------------------------------------------------------------------- #
# Bus / recovery hint / event hook
# --------------------------------------------------------------------------- #

def test_discoveries_and_dialog_end_are_published(make_agent) -> None:
    stack = make_agent(
        replies=[
            scripted_reply(
                "record_fact",
                {
                    "category": "npc",
                    "content": "A person near the entrance greeted the player.",
                    "verification_status": "verified",
                    "npc_name": "the greeter",
                    "dialogue_text": "Hello there!",
                },
            ),
            scripted_reply("finish", {"summary": "done"}),
        ],
        with_bus=True,
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert isinstance(stack.bus, FakeBus)
    assert "story_discovery" in stack.bus.event_types
    assert "dialog_ended" in stack.bus.event_types
    discovery_payloads = stack.bus.payloads("story_discovery")
    assert discovery_payloads and discovery_payloads[0]["content"]
    end_payloads = stack.bus.payloads("dialog_ended")
    assert end_payloads and end_payloads[0]["npcs"]


def test_recovery_hint_reaches_the_next_prompt(make_agent) -> None:
    stack = make_agent(replies=[scripted_reply("finish", {"summary": "done"})])
    stack.agent.apply_recovery_hint("l1_local", "Stop repeating A; try pressing B once.")
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    system_messages = [
        message
        for message in stack.router.calls[0]["messages"]
        if message.get("role") == "system"
    ]
    assert system_messages
    assert "Stop repeating A" in system_messages[0]["content"]
    assert stack.agent.status()["recovery_hints"][0]["level"] == "l1_local"


def test_on_event_tracks_pause_and_resume(make_agent) -> None:
    from llm_agent.multi_agent.contracts import AgentMessage, AgentRole, EventType

    stack = make_agent()
    paused = AgentMessage.create(
        EventType.AGENT_PAUSED,
        source_agent=AgentRole.DIRECTOR,
        payload={"agent": "story", "reason": "director pause"},
    )
    stack.agent.on_event(paused)
    assert stack.agent.paused is True
    resumed = AgentMessage.create(
        EventType.AGENT_RESUMED,
        source_agent=AgentRole.DIRECTOR,
        payload={"agent": "story"},
    )
    stack.agent.on_event(resumed)
    assert stack.agent.paused is False
    stack.agent.on_event({"event_type": "dialog_started", "payload": {"frame": 3}})
    assert stack.agent.status()["last_event"]["event_type"] == "dialog_started"
    assert stack.agent.on_event("not-an-event") is None


# --------------------------------------------------------------------------- #
# Interaction probe integration (real acceptance path)
# --------------------------------------------------------------------------- #

def test_probe_acquires_dialogue_then_reads_it(make_agent) -> None:
    emulator = FakeEmulator(
        initial_screen="overworld", dialog_after_presses=2, dialog_lines=5
    )
    stack = make_agent(
        replies=[scripted_reply("finish", {"summary": "read the opened conversation"})],
        emulator=emulator,
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert result["turns"] == 1
    assert result["api_calls"] == 1
    probe_buttons = [button for button, _frames in emulator.presses]
    assert len(probe_buttons) <= 12
    assert probe_buttons[0] == "a"
    # The vision call received the dialogue screenshot (image block present).
    assert _last_user_images(stack.router.calls[0]["messages"])
    # Acceptance evidence: real screenshot paths (including the acquired dialog),
    # frame indices and real turn/API/cost counters.
    assert result["evidence"]["screens"]
    for path in result["evidence"]["screens"]:
        assert path.endswith(".png") and os.path.isfile(path), path
    assert result["evidence"]["frames"]
    assert result["turns"] == 1 and result["api_calls"] == 1
    assert result["cost_usd"] > 0.0


def test_agent_works_with_a_guarded_emulator_proxy(make_agent) -> None:
    """The only emulator surface touched is observe/press/wait/run_batch/etc."""

    class StrictGuardedProxy:
        """Exposes only the documented GuardedEmulator attributes."""

        __slots__ = ("_emu", "frame_count", "last_frame", "started")

        def __init__(self, emu: FakeEmulator) -> None:
            self._emu = emu
            self.frame_count = emu.frame_count
            self.last_frame = None
            self.started = True

        def observe(self):
            frame = self._emu.observe()
            self.frame_count = self._emu.frame_count
            self.last_frame = frame
            return frame

        def press(self, button: str, frames: int | None = None):
            frame = self._emu.press(button, frames=frames)
            self.frame_count = self._emu.frame_count
            self.last_frame = frame
            return frame

        def wait(self, frames: int = 1):
            frame = self._emu.wait(frames)
            self.frame_count = self._emu.frame_count
            self.last_frame = frame
            return frame

        def run_batch(self, buttons, stop_on_event: bool = True):
            return self._emu.run_batch(buttons, stop_on_event=stop_on_event)

    proxy = StrictGuardedProxy(FakeEmulator(initial_screen="dialog", dialog_lines=1))
    stack = make_agent(
        replies=[
            scripted_reply("advance_dialog", {"button": "a", "times": 1}),
        ],
    )
    result = stack.agent.handle_dialog(emulator=proxy, max_turns=4)
    assert result["status"] == "dialog_ended"
    assert result["evidence"]["screens"]
    assert stack.router.complete_calls == 2


def test_prompt_build_fallback_keeps_the_vision_call(make_agent, monkeypatch) -> None:
    """Defensive path: a broken AgentContext must not lose the dialogue."""
    stack = make_agent(replies=[scripted_reply("finish", {"summary": "still read it"})])

    def boom(**kwargs):  # noqa: ANN003
        raise RuntimeError("context rendering exploded")

    monkeypatch.setattr(stack.agent.context, "build_messages", boom)
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert stack.router.complete_calls == 1
    messages = stack.router.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert _last_user_images(messages)
    assert stack.agent.status()["context_errors"]


def test_probe_returns_no_dialog_when_nothing_is_there(make_agent) -> None:
    emulator = FakeEmulator(initial_screen="overworld", dialog_after_presses=None)
    stack = make_agent(replies=[], emulator=emulator)
    result = _run(stack)
    assert result["status"] == "no_dialog"
    assert result["api_calls"] == 0
    assert result["turns"] == 0
    assert stack.router.complete_calls == 0
    assert 0 < len(emulator.presses) <= 12
    assert result["evidence"]["screens"]
    for path in result["evidence"]["screens"]:
        assert path.endswith(".png")


def test_probe_runs_for_interaction_task_payload(make_agent) -> None:
    emulator = FakeEmulator(
        initial_screen="overworld", dialog_after_presses=2, dialog_lines=4
    )
    task = Task.create(
        owner_agent="navigation",
        requested_by="director",
        task_type=TaskType.INTERACT,
        description="Probe the current position for something to interact with",
        payload={"probe_interaction": True},
    )
    stack = make_agent(
        replies=[scripted_reply("finish", {"summary": "read the conversation"})],
        emulator=emulator,
    )
    result = _run(stack, task=task)
    assert result["status"] == "dialog_ended"
    assert stack.router.complete_calls == 1
    assert len(emulator.presses) <= 12
    # The fixed AgentContext renders the assigned task itself (no fallback used).
    assert stack.agent.status()["context_errors"] == []


def test_probe_returns_interrupted_when_menu_cannot_be_closed(make_agent) -> None:
    emulator = FakeEmulator(
        initial_screen="overworld", menu_after_presses=1, menu_close_presses=99
    )
    stack = make_agent(replies=[], emulator=emulator)
    result = _run(stack)
    assert result["status"] == "interrupted"
    assert result["api_calls"] == 0
    assert [button for button, _frames in emulator.presses][:3] == ["a", "b", "b"]


# --------------------------------------------------------------------------- #
# Chat-protocol invariants (real-run HTTP 400 regression) and API retry
# --------------------------------------------------------------------------- #

def test_over_declared_tool_calls_are_trimmed_to_the_executed_one(make_agent) -> None:
    """Regression for the real-run HTTP 400: unanswered tool_calls in history.

    A model reply may declare several tool calls; the story loop executes exactly
    one per turn, so the kept assistant message must declare only that call and
    the next request must be a valid tool protocol.
    """
    stack = make_agent(
        replies=[
            multi_call_reply(
                ("advance_dialog", {"button": "a", "times": 1}),
                (
                    "record_fact",
                    {
                        "category": "other",
                        "content": "the second declared call must never run",
                        "verification_status": "hypothesis",
                    },
                ),
            ),
            scripted_reply("finish", {"summary": "one tool per turn"}),
        ],
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=6),
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    # Only the first declared tool was executed.
    assert [button for button, _frames in stack.emulator.presses] == ["a"]
    assert result["discoveries"] == []
    assert stack.router.complete_calls == 2
    _assert_all_requests_valid(stack.router)
    second_request = stack.router.calls[1]["messages"]
    assistant_calls = [
        message
        for message in second_request
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert assistant_calls
    assert all(len(message["tool_calls"]) == 1 for message in assistant_calls)
    tool_messages = [
        message for message in second_request if message.get("role") == "tool"
    ]
    assert len(tool_messages) == sum(
        len(message["tool_calls"]) for message in assistant_calls
    )


def test_reply_without_a_usable_tool_call_keeps_history_valid(make_agent) -> None:
    malformed = ModelReply(
        content="I am not sure what to do.",
        tool_calls=[ToolCall(id="call_bad", name="", arguments={})],
        usage=ApiUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        model="fake-story-model",
    )
    stack = make_agent(
        replies=[malformed, scripted_reply("finish", {"summary": "recovered"})],
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=6),
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert stack.emulator.presses == []
    _assert_all_requests_valid(stack.router)
    # The unusable call must not be declared in the kept assistant message.
    for request in stack.router.calls:
        for message in request["messages"]:
            if message.get("role") == "assistant":
                assert not message.get("tool_calls")


def test_api_error_retries_once_with_a_fresh_conversation_window(make_agent) -> None:
    stack = make_agent(
        replies=[
            scripted_reply("advance_dialog", {"button": "a", "times": 1}),
            ApiResponseError(
                "DeepSeek API HTTP 400: assistant tool_calls were not answered",
                status=400,
                retryable=False,
            ),
            scripted_reply("finish", {"summary": "recovered after reset"}),
        ],
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=8),
    )
    result = _run(stack)
    assert result["status"] == "dialog_ended"
    assert result["error"] == ""
    assert result["turns"] == 3
    assert result["api_calls"] == 2
    assert result["cost_usd"] > 0.0
    assert stack.agent.status()["counters"]["api_retries"] == 1
    _assert_all_requests_valid(stack.router)
    # The retried request starts from a clean window: system + latest observation.
    retried = stack.router.calls[2]["messages"]
    assert [message["role"] for message in retried] == ["system", "user"]
    assert _last_user_images(retried)


def test_api_error_retry_is_bounded(make_agent) -> None:
    stack = make_agent(
        replies=[
            ApiResponseError("HTTP 400 first", status=400, retryable=False),
            ApiResponseError("HTTP 400 again", status=400, retryable=False),
        ]
    )
    result = _run(stack)
    assert result["status"] == "failed"
    assert "HTTP 400" in result["error"]
    assert stack.router.complete_calls == 2
    assert result["api_calls"] == 0
    assert stack.agent.status()["counters"]["api_retries"] == 1


def test_auth_error_is_not_retried(make_agent) -> None:
    stack = make_agent(
        replies=[ApiAuthenticationError("invalid API key", status=401, retryable=False)]
    )
    result = _run(stack)
    assert result["status"] == "failed"
    assert stack.router.complete_calls == 1
    assert stack.agent.status()["counters"]["api_retries"] == 0


def test_failed_task_does_not_poison_the_next_task(make_agent) -> None:
    """begin_task resets the conversation window, also after a failed dialogue."""
    stack = make_agent(
        replies=[
            scripted_reply("advance_dialog", {"button": "a", "times": 1}),
            RuntimeError("transient crash"),
            scripted_reply("finish", {"summary": "second task is clean"}),
        ],
        emulator=FakeEmulator(initial_screen="dialog", dialog_lines=8),
    )
    first = _run(stack, max_turns=4)
    assert first["status"] == "failed"
    second = stack.agent.handle_dialog(emulator=stack.emulator, max_turns=4)
    assert second["status"] == "dialog_ended"
    _assert_all_requests_valid(stack.router)
    second_request = stack.router.calls[2]["messages"]
    assert [message["role"] for message in second_request] == ["system", "user"]
    assert _last_user_images(second_request)
