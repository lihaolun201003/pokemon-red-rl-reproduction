"""Story / dialogue agent package (multi-agent v1).

Public surface
--------------
* :class:`StoryAgent` -- real dialogue takeover: screenshot -> model -> real
  button presses -> memory, with honest verified/inferred/hypothesis notes and
  follow-up task proposals for the director.
* :class:`StoryMemory` -- the agent's auditable notebook: structured knowledge
  entries mirrored into the shared world memory, NPC notes, proposals and a
  validated dialogue transcript; persisted atomically next to the AgentMemory.
* :data:`TOOL_SCHEMAS` / :data:`STORY_TOOL_NAMES` / :class:`StoryTools` -- the
  six-tool permission whitelist and its strict executor.
* :data:`SYSTEM_PROMPT` -- spoiler-free first-time-player instructions.

Importing this package is cheap: no emulator, no network, no API key.
"""

from __future__ import annotations

from llm_agent.story.agent import (
    DIALOG_SCREEN_EVENTS,
    StoryAgent,
    StoryAgentConfig,
    is_dialog_context,
)
from llm_agent.story.memory import (
    STORY_KNOWLEDGE_CATEGORIES,
    STORY_MEMORY_FILENAME,
    STORY_PRIORITIES,
    STORY_PROPOSAL_OWNERS,
    STORY_PROPOSAL_TYPES,
    SharedMemoryAdapter,
    StoryMemory,
)
from llm_agent.story.prompts import SYSTEM_PROMPT
from llm_agent.story.tools import (
    ALLOWED_PRESS_BUTTONS,
    STORY_TOOL_NAMES,
    TOOL_SCHEMAS,
    StoryTools,
    ToolError,
    validate_call,
)

__all__ = [
    "StoryAgent",
    "StoryAgentConfig",
    "is_dialog_context",
    "DIALOG_SCREEN_EVENTS",
    "StoryMemory",
    "SharedMemoryAdapter",
    "STORY_MEMORY_FILENAME",
    "STORY_KNOWLEDGE_CATEGORIES",
    "STORY_PROPOSAL_OWNERS",
    "STORY_PROPOSAL_TYPES",
    "STORY_PRIORITIES",
    "SYSTEM_PROMPT",
    "StoryTools",
    "TOOL_SCHEMAS",
    "STORY_TOOL_NAMES",
    "ALLOWED_PRESS_BUTTONS",
    "ToolError",
    "validate_call",
]
