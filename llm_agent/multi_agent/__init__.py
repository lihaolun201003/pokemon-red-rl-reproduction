"""Pokémon Red Multi-Agent v1 subsystem.

Public surface
--------------
* :mod:`llm_agent.multi_agent.contracts` — frozen protocol records/enums.
* :mod:`llm_agent.multi_agent.event_bus` — persisted priority event bus.
* :mod:`llm_agent.multi_agent.control_arbiter` — single-owner emulator lease.
* :mod:`llm_agent.multi_agent.task_manager` — shared task graph.
* :mod:`llm_agent.multi_agent.scheduler` — deterministic task selection.
* :mod:`llm_agent.multi_agent.shared_memory` — shared world knowledge base.
* :mod:`llm_agent.multi_agent.progress_monitor` — L1/L2/L3 progress checks.
* :mod:`llm_agent.multi_agent.recovery` — self-correction + rollback manager.
* :mod:`llm_agent.multi_agent.checkpoint` — branch/snapshot manifest layer.
* :mod:`llm_agent.multi_agent.director` — the :class:`GameDirector` state machine.
* :mod:`llm_agent.multi_agent.integration` — Navigation v2 adapter + session.

Heavy submodules are imported lazily with ``__getattr__`` so that importing the
package (e.g. from a unit test) never touches PyBoy, the network or the API key.
"""

from __future__ import annotations

from typing import Any

from llm_agent.multi_agent import contracts
from llm_agent.multi_agent.contracts import (
    AGENT_EVENTS,
    EVENT_PAYLOAD_SPEC,
    EVENT_ROUTING,
    KNOWLEDGE_CATEGORIES,
    PROTOCOL_VERSION,
    AgentMessage,
    AgentRole,
    BranchRecord,
    BranchStatus,
    BudgetLimitError,
    ControlDeniedError,
    ControlLease,
    ControlOwner,
    CycleDetectedError,
    DuplicateMessageError,
    EventType,
    KnowledgeEntry,
    KnowledgeError,
    MultiAgentError,
    Priority,
    ProtocolError,
    RecoveryLevel,
    RecoveryLimitError,
    StagnationKind,
    StagnationLimitError,
    Task,
    TaskError,
    TaskStatus,
    TaskType,
    VerificationStatus,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "EventBus": ("llm_agent.multi_agent.event_bus", "EventBus"),
    "EventBusConfig": ("llm_agent.multi_agent.event_bus", "EventBusConfig"),
    "ControlArbiter": ("llm_agent.multi_agent.control_arbiter", "ControlArbiter"),
    "ArbiterConfig": ("llm_agent.multi_agent.control_arbiter", "ArbiterConfig"),
    "GuardedEmulator": ("llm_agent.multi_agent.control_arbiter", "GuardedEmulator"),
    "TaskManager": ("llm_agent.multi_agent.task_manager", "TaskManager"),
    "TaskManagerConfig": ("llm_agent.multi_agent.task_manager", "TaskManagerConfig"),
    "TaskScheduler": ("llm_agent.multi_agent.scheduler", "TaskScheduler"),
    "SchedulerConfig": ("llm_agent.multi_agent.scheduler", "SchedulerConfig"),
    "SharedWorldMemory": ("llm_agent.multi_agent.shared_memory", "SharedWorldMemory"),
    "SharedMemoryConfig": ("llm_agent.multi_agent.shared_memory", "SharedMemoryConfig"),
    "ProgressMonitor": ("llm_agent.multi_agent.progress_monitor", "ProgressMonitor"),
    "ProgressMonitorConfig": ("llm_agent.multi_agent.progress_monitor", "ProgressMonitorConfig"),
    "RecoveryManager": ("llm_agent.multi_agent.recovery", "RecoveryManager"),
    "RecoveryConfig": ("llm_agent.multi_agent.recovery", "RecoveryConfig"),
    "BranchCheckpointManager": ("llm_agent.multi_agent.checkpoint", "BranchCheckpointManager"),
    "BranchCheckpointConfig": ("llm_agent.multi_agent.checkpoint", "BranchCheckpointConfig"),
    "AgentRegistry": ("llm_agent.multi_agent.agent_registry", "AgentRegistry"),
    "AgentContext": ("llm_agent.multi_agent.agent_context", "AgentContext"),
    "ModelRouter": ("llm_agent.multi_agent.model_router", "ModelRouter"),
    "ModelRouterConfig": ("llm_agent.multi_agent.model_router", "ModelRouterConfig"),
    "GameDirector": ("llm_agent.multi_agent.director", "GameDirector"),
    "DirectorConfig": ("llm_agent.multi_agent.director", "DirectorConfig"),
    "NavigationAdapter": ("llm_agent.multi_agent.integration", "NavigationAdapter"),
    "MultiAgentSession": ("llm_agent.multi_agent.integration", "MultiAgentSession"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - import plumbing
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = [
    "contracts",
    "PROTOCOL_VERSION",
    "AgentMessage",
    "AgentRole",
    "BranchRecord",
    "BranchStatus",
    "BudgetLimitError",
    "ControlDeniedError",
    "ControlLease",
    "ControlOwner",
    "CycleDetectedError",
    "DuplicateMessageError",
    "EventType",
    "EVENT_PAYLOAD_SPEC",
    "EVENT_ROUTING",
    "AGENT_EVENTS",
    "KNOWLEDGE_CATEGORIES",
    "KnowledgeEntry",
    "KnowledgeError",
    "MultiAgentError",
    "Priority",
    "ProtocolError",
    "RecoveryLevel",
    "RecoveryLimitError",
    "StagnationKind",
    "StagnationLimitError",
    "Task",
    "TaskError",
    "TaskStatus",
    "TaskType",
    "VerificationStatus",
    *_LAZY_EXPORTS.keys(),
]
