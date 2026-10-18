"""Navigation v2 — visual autonomous exploration and permanent map memory.

The subsystem is split into small modules with frozen contracts
(see ``NAV_CONTRACT.md`` and ``types.py``):

* :mod:`llm_agent.navigation.vision`       — pixel/tile primitives
* :mod:`llm_agent.navigation.localization` — visual motion estimation & pose
* :mod:`llm_agent.navigation.grid_map`     — per-area logical grid map
* :mod:`llm_agent.navigation.world_model`  — multi-area world graph / relocalization
* :mod:`llm_agent.navigation.pathfinder`   — BFS/A* planning
* :mod:`llm_agent.navigation.explorer`     — frontier-based exploration policy
* :mod:`llm_agent.navigation.controller`   — verified low-level navigation loop
* :mod:`llm_agent.navigation.navigation_agent` — high-level API for the LLM tools
* :mod:`llm_agent.navigation.viz`          — map / status rendering

Hard rule: no module here reads game RAM or external map data.  The only input is
the 160x144 framebuffer plus the agent's own recorded observations.
"""

from __future__ import annotations

__all__ = [
    "types",
    "vision",
    "localization",
    "grid_map",
    "world_model",
    "pathfinder",
    "explorer",
    "controller",
    "navigation_agent",
    "viz",
]
