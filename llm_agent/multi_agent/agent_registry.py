"""Agent registry: who exists, what they may do, and how to reach them.

The registry is intentionally small: the director asks it which agents are
registered, whether they are available (not paused) and which tools they own.
It is not a service locator: agents receive their collaborators explicitly.

No game knowledge is stored here — only generic roles and capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from llm_agent.multi_agent.contracts import AgentRole, TaskType


#: Default task types each professional agent may own.
DEFAULT_TASK_OWNERSHIP: dict[TaskType, AgentRole] = {
    TaskType.EXPLORE: AgentRole.NAVIGATION,
    TaskType.NAVIGATE: AgentRole.NAVIGATION,
    TaskType.INTERACT: AgentRole.STORY,
    TaskType.TALK: AgentRole.STORY,
    TaskType.INVESTIGATE: AgentRole.STORY,
    TaskType.BATTLE: AgentRole.POKEMON,
    TaskType.CATCH: AgentRole.POKEMON,
    TaskType.TRAIN: AgentRole.POKEMON,
    TaskType.RECOVER: AgentRole.DIRECTOR,
    TaskType.VERIFY: AgentRole.DIRECTOR,
    TaskType.DIRECTOR_REVIEW: AgentRole.DIRECTOR,
    TaskType.CUSTOM: AgentRole.NAVIGATION,
}


@dataclass
class AgentCapability:
    """Static description of one agent's responsibility."""

    role: AgentRole
    description: str
    tools: tuple[str, ...] = ()
    can_press_buttons: bool = False
    task_types: tuple[TaskType, ...] = ()
    context_kind: tuple[str, ...] = ()


#: Capabilities fixed by the multi-agent design decision (blind exploration).
DEFAULT_CAPABILITIES: dict[AgentRole, AgentCapability] = {
    AgentRole.NAVIGATION: AgentCapability(
        role=AgentRole.NAVIGATION,
        description="Autonomous visual mapping, A* pathfinding and frontier exploration.",
        tools=(
            "explore_current_area",
            "navigate_to",
            "get_navigation_status",
            "get_local_map",
            "get_world_map_summary",
            "pause_exploration",
            "resume_exploration",
        ),
        can_press_buttons=True,
        task_types=(TaskType.EXPLORE, TaskType.NAVIGATE, TaskType.CUSTOM, TaskType.INTERACT),
        context_kind=("overworld",),
    ),
    AgentRole.STORY: AgentCapability(
        role=AgentRole.STORY,
        description="Reads real dialogue, records story clues, proposes objectives.",
        tools=("observe", "advance_dialog", "choose_option", "record_fact", "propose_task", "finish"),
        can_press_buttons=True,
        task_types=(TaskType.TALK, TaskType.INTERACT, TaskType.INVESTIGATE, TaskType.CUSTOM),
        context_kind=("dialog", "menu"),
    ),
    AgentRole.POKEMON: AgentCapability(
        role=AgentRole.POKEMON,
        description="Handles battles/catches/team autonomously from the visible battle menus.",
        tools=("battle_action", "observe", "record_lesson", "finish"),
        can_press_buttons=True,
        task_types=(TaskType.BATTLE, TaskType.CATCH, TaskType.TRAIN, TaskType.CUSTOM),
        context_kind=("battle",),
    ),
    AgentRole.DIRECTOR: AgentCapability(
        role=AgentRole.DIRECTOR,
        description="Deterministic scheduler; owns control arbitration and recovery.",
        tools=("review", "schedule", "rollback", "checkpoint", "pause", "stop"),
        can_press_buttons=False,
        task_types=(TaskType.RECOVER, TaskType.VERIFY, TaskType.DIRECTOR_REVIEW),
        context_kind=("battle", "dialog", "menu", "overworld", "transition"),
    ),
    AgentRole.SYSTEM: AgentCapability(
        role=AgentRole.SYSTEM,
        description="System-level operations (save/load state, checkpoints).",
        tools=("save_state", "load_state", "checkpoint", "stop"),
        can_press_buttons=False,
        task_types=(),
        context_kind=(),
    ),
}


class AgentRegistry:
    """Runtime registry of constructed agents and their capabilities."""

    def __init__(
        self,
        *,
        capabilities: Mapping[AgentRole, AgentCapability] | None = None,
    ) -> None:
        self._capabilities: dict[AgentRole, AgentCapability] = dict(
            capabilities if capabilities is not None else DEFAULT_CAPABILITIES
        )
        self._agents: dict[AgentRole, Any] = {}
        self._paused: set[AgentRole] = set()
        self._notes: dict[AgentRole, str] = {}

    # -- capabilities ---------------------------------------------------- #
    def capability(self, role: AgentRole | str) -> AgentCapability:
        role = AgentRole.coerce(role)
        return self._capabilities.get(role) or AgentCapability(role=role, description="")

    def tools_for(self, role: AgentRole | str) -> tuple[str, ...]:
        return tuple(self.capability(role).tools)

    def task_types_for(self, role: AgentRole | str) -> tuple[TaskType, ...]:
        return tuple(self.capability(role).task_types)

    def owner_for_task_type(self, task_type: TaskType | str) -> AgentRole:
        return DEFAULT_TASK_OWNERSHIP.get(TaskType.coerce(task_type), AgentRole.NAVIGATION)

    # -- agent instances -------------------------------------------------- #
    def register(self, role: AgentRole | str, agent: Any, *, note: str = "") -> Any:
        role_enum = AgentRole.coerce(role)
        if agent is None:
            raise ValueError(f"cannot register a null agent for {role_enum.value}")
        self._agents[role_enum] = agent
        if note:
            self._notes[role_enum] = str(note)
        return agent

    def get(self, role: AgentRole | str) -> Any | None:
        return self._agents.get(AgentRole.coerce(role))

    def require(self, role: AgentRole | str) -> Any:
        role_enum = AgentRole.coerce(role)
        agent = self._agents.get(role_enum)
        if agent is None:
            raise KeyError(f"agent not registered: {role_enum.value}")
        return agent

    def registered(self) -> list[AgentRole]:
        return sorted(self._agents, key=lambda item: item.value)

    def professional_agents(self) -> list[AgentRole]:
        return [role for role in self.registered() if role.is_professional]

    def available(self) -> dict[str, bool]:
        return {
            role.value: (role in self._agents and role not in self._paused)
            for role in sorted(self._agents, key=lambda item: item.value)
        }

    def is_available(self, role: AgentRole | str) -> bool:
        role_enum = AgentRole.coerce(role)
        return role_enum in self._agents and role_enum not in self._paused

    def pause(self, role: AgentRole | str, *, reason: str = "") -> None:
        role_enum = AgentRole.coerce(role)
        self._paused.add(role_enum)
        if reason:
            self._notes[role_enum] = f"paused: {reason}"

    def resume(self, role: AgentRole | str) -> None:
        self._paused.discard(AgentRole.coerce(role))

    @property
    def paused_roles(self) -> set[AgentRole]:
        return set(self._paused)

    # -- introspection ---------------------------------------------------- #
    def describe(self) -> str:
        """Compact text block used in the director's (rare) LLM review prompt."""
        lines: list[str] = []
        for role in self.registered():
            cap = self.capability(role)
            state = "paused" if role in self._paused else "available"
            tasks = ", ".join(t.value for t in cap.task_types) or "-"
            lines.append(f"- {role.value} [{state}] {cap.description} (tasks: {tasks})")
        return "\n".join(lines)

    def status(self) -> dict[str, Any]:
        return {
            task: {
                "registered": True,
                "available": task not in self._paused,
                "description": self.capability(task).description,
                "tools": list(self.capability(task).tools),
                "note": self._notes.get(task, ""),
            }
            for task in sorted(self._agents, key=lambda item: item.value)
        }

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._agents)

    def __contains__(self, role: object) -> bool:  # pragma: no cover - convenience
        try:
            return AgentRole.coerce(role) in self._agents  # type: ignore[arg-type]
        except Exception:
            return False


__all__ = [
    "AgentCapability",
    "AgentRegistry",
    "DEFAULT_CAPABILITIES",
    "DEFAULT_TASK_OWNERSHIP",
]
