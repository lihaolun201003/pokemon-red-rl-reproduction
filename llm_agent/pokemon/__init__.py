"""Pokémon (battle) agent package for the multi-agent system.

Public surface
--------------
* :class:`~llm_agent.pokemon.agent.PokemonAgent` — the autonomous battle agent
  (frozen director interface: ``handles`` / ``handle_battle`` / ``on_event`` /
  ``status`` / ``pause`` / ``resume`` / ``paused`` / ``apply_recovery_hint``).
* :class:`~llm_agent.pokemon.memory.PokemonMemory` — battle-specific long-term
  memory built on the generic :class:`~llm_agent.memory.AgentMemory`.
* :mod:`llm_agent.pokemon.prompts` — the blind battle prompt (no game knowledge).
* :mod:`llm_agent.pokemon.tools` — the explicit battle tool schemas and the
  validating executor.

The package only imports pixel observation, memory and protocol modules: it
never imports PyBoy, never reads RAM and never contains species/move/location
knowledge.  Everything the agent believes about a battle comes from the real
screenshot the model saw and the model's own decisions, which are all recorded
per turn (raw decision + real presses + pixel verification) for the audit trail.
"""

from __future__ import annotations

from llm_agent.pokemon import prompts, tools
from llm_agent.pokemon.agent import (
    DEFAULT_MAX_TURNS,
    OUTCOMES,
    STATUSES,
    BattleMenuDriver,
    PokemonAgent,
    battle_menu_layout,
)
from llm_agent.pokemon.memory import PokemonMemory

__all__ = [
    "PokemonAgent",
    "PokemonMemory",
    "BattleMenuDriver",
    "battle_menu_layout",
    "prompts",
    "tools",
    "DEFAULT_MAX_TURNS",
    "STATUSES",
    "OUTCOMES",
]
