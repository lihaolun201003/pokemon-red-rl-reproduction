"""Shared fixtures for the DeepSeek LLM agent test suite.

Tests never require a network connection or a real API key.  Tests that need the
real game ROM/state skip automatically when the user's lawful files are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_agent import contracts
from llm_agent.config import MemoryConfig


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def rom_path(project_root: Path) -> Path:
    path = project_root / "local" / "PokemonRed.gb"
    if not path.is_file():
        pytest.skip("lawful English Pokémon Red ROM not present in local/")
    return path


@pytest.fixture(scope="session")
def state_path(project_root: Path) -> Path:
    path = project_root / "local" / "has_pokedex_nballs.state"
    if not path.is_file():
        pytest.skip("official initial state not present in local/")
    return path


def frame_from_rgba_local(rgba: np.ndarray, frame_index: int = 0, captured_at: float = 0.0) -> contracts.Frame:
    """Minimal independent Frame factory (does not depend on observation.py)."""
    rgba = np.ascontiguousarray(rgba.astype(np.uint8))
    luma = float(rgba[..., 0].mean())
    black = bool((rgba[..., 0] <= 8).mean() >= 0.98)
    return contracts.Frame(
        frame_index=frame_index,
        rgba=rgba,
        sha256=contracts.sha256_rgba(rgba),
        captured_at=captured_at,
        is_black=black,
        luma_mean=luma,
    )


@pytest.fixture
def make_frame():
    return frame_from_rgba_local


@pytest.fixture
def sample_rgba() -> np.ndarray:
    """Synthetic 'game screen': white background, a dark room rectangle and text rows."""
    rgba = np.full((contracts.SCREEN_HEIGHT, contracts.SCREEN_WIDTH, 4), 255, dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[8:120, 8:150, 0:3] = 153
    rgba[24:40, 24:136, 0:3] = 0
    rgba[48:64, 24:136, 0:3] = 85
    rgba[128:140, 16:144, 0:3] = 0
    return rgba


@pytest.fixture
def black_rgba() -> np.ndarray:
    rgba = np.zeros((contracts.SCREEN_HEIGHT, contracts.SCREEN_WIDTH, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


@pytest.fixture
def memory_config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(root=str(tmp_path / "memory"))
