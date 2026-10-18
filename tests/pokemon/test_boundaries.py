"""Import/observation boundaries of the battle agent package.

These are the honesty guarantees of the subsystem: the battle agent may only use
the screenshots and its own memory, so its modules must not import PyBoy, the
verification-only RAM probe or contain RAM addresses.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "llm_agent" / "pokemon"
MODULE_FILES = {
    "prompts.py": PACKAGE / "prompts.py",
    "tools.py": PACKAGE / "tools.py",
    "memory.py": PACKAGE / "memory.py",
    "agent.py": PACKAGE / "agent.py",
    "__init__.py": PACKAGE / "__init__.py",
}

#: RAM addresses (e.g. 0xD16B) are a hard no in every agent module.
RAM_ADDRESS = re.compile(r"0x[0-9A-Fa-f]{4}")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            modules.add(base)
            for alias in node.names:
                modules.add(f"{base}.{alias.name}" if base else alias.name)
    return modules


def test_package_imports_cleanly() -> None:
    import importlib

    for name in (
        "llm_agent.pokemon",
        "llm_agent.pokemon.agent",
        "llm_agent.pokemon.memory",
        "llm_agent.pokemon.prompts",
        "llm_agent.pokemon.tools",
    ):
        module = importlib.import_module(name)
        assert module is not None


@pytest.mark.parametrize("module_name", sorted(MODULE_FILES))
def test_modules_never_import_pyboy_or_the_ram_probe(module_name: str) -> None:
    imports = _imported_modules(MODULE_FILES[module_name])
    forbidden = {
        module
        for module in imports
        if module == "pyboy"
        or module.startswith("pyboy.")
        or module == "llm_agent.progress"
        or module.startswith("llm_agent.progress.")
    }
    assert not forbidden, f"{module_name} imports hidden state: {forbidden}"


@pytest.mark.parametrize("module_name", sorted(MODULE_FILES))
def test_modules_have_no_ram_addresses(module_name: str) -> None:
    text = MODULE_FILES[module_name].read_text(encoding="utf-8")
    offenders = [match.group(0) for match in RAM_ADDRESS.finditer(text)]
    assert not offenders, f"{module_name} contains RAM addresses: {offenders}"


def test_agent_never_holds_an_emulator_until_handle_battle() -> None:
    """The agent stores no emulator reference on construction (no PyBoy, no ROM)."""
    import inspect

    from llm_agent.pokemon.agent import PokemonAgent

    signature = inspect.signature(PokemonAgent.__init__)
    parameters = set(signature.parameters)
    assert "router" in parameters
    assert "bus" in parameters
    assert "arbiter" in parameters
    assert "shared_memory" in parameters
    assert "memory" in parameters
    assert "config" in parameters
    assert "screen_dir" in parameters
    assert "clock" in parameters


def test_agent_module_does_not_read_game_memory() -> None:
    text = (PACKAGE / "agent.py").read_text(encoding="utf-8").lower()
    for forbidden in ("read_memory", "get_memory", "memory_scanner", "wram", "hram", "map_id"):
        assert forbidden not in text, f"agent.py references {forbidden!r}"


def test_package_exports_the_frozen_surface() -> None:
    import llm_agent.pokemon as package

    assert package.PokemonAgent is not None
    assert package.PokemonMemory is not None
    assert package.prompts is not None
    assert package.tools is not None
    for attribute in ("DEFAULT_MAX_TURNS", "STATUSES", "OUTCOMES", "BattleMenuDriver"):
        assert hasattr(package, attribute), attribute


def test_observation_uses_only_the_public_pixel_helpers() -> None:
    """The agent must analyze pixels through ``llm_agent.observation``."""
    imports = _imported_modules(PACKAGE / "agent.py")
    assert "llm_agent.observation" in imports
