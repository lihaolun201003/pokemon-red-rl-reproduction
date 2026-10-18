"""DeepSeek LLM Game Agent for Pokémon Red — independent of the PPO training pipeline.

This package contains a self-contained LLM agent:

    DeepSeek Brain -> constrained tool calls -> PyBoy -> screenshots -> DeepSeek Brain

Nothing in this package may inject hidden RAM state, story flags, map data or
external walkthroughs into the decision loop.  Only what a human player could
see on the 160x144 screen (plus the agent's own recorded memory) is used for
decisions.  The one exception is :mod:`llm_agent.progress`, an explicitly
separate verification-only module used by tests/reports; it must never be
imported by the decision loop (enforced by tests).
"""

from __future__ import annotations

import warnings

# PyBoy pulls in pysdl2-dll which emits a noisy UserWarning on import; it is
# harmless and must not pollute CLI output or look like a failure in logs.
warnings.filterwarnings("ignore", message=r"Using SDL2 binaries from pysdl2-dll.*")

__version__ = "0.1.0"
