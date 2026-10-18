"""Enforce the blind-exploration / no-hidden-state boundaries of the project.

These tests are part of the honesty guarantees of the LLM agent:

1. The decision loop never imports the verification-only RAM probe.
2. No decision-loop module contains hard-coded RAM addresses.
3. Only the emulator (and the separate verification probe) may import PyBoy.
4. The system prompt contains no story route / location / gym names.
5. The tool schemas stay constrained and complete.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT_ROOT / "llm_agent"

DECISION_LOOP_MODULES = [
    "contracts.py",
    "config.py",
    "dotenv.py",
    "observation.py",
    "emulator.py",
    "tools.py",
    "memory.py",
    "mental_map.py",
    "guard.py",
    "prompts.py",
    "agent.py",
    "runtime.py",
    "viewer.py",
    "scripted.py",
    "api_client.py",
    "budget.py",
]

#: Navigation v2 modules: same blind-exploration rules as the decision loop
#: (they may only use screenshots + verified actions; no RAM, no PyBoy imports).
NAVIGATION_MODULES = [
    "types.py",
    "vision.py",
    "localization.py",
    "grid_map.py",
    "world_model.py",
    "pathfinder.py",
    "explorer.py",
    "controller.py",
    "navigation_agent.py",
    "viz.py",
]

#: The only modules allowed to import the verification-only probe / PyBoy.
PYBOY_ALLOWED = {"emulator.py", "progress.py"}

#: Story spoilers that must never be baked into prompts or code.
FORBIDDEN_PROMPT_TERMS = [
    "pallet town",
    "viridian",
    "pewter",
    "cerulean",
    "vermilion",
    "celadon",
    "fuchsia",
    "saffron",
    "cinnabar",
    "indigo plateau",
    "mt. moon",
    "mt moon",
    "rock tunnel",
    "ss anne",
    "s.s. anne",
    "lavender town",
    "oak's lab",
    "oaks lab",
    "professor oak",
    "brock",
    "misty",
    "lt. surge",
    "erika",
    "koga",
    "sabrina",
    "blaine",
    "giovanni",
    "route 1",
    "route 2",
    "route 3",
    "route 4",
    "poke mart",
    "poké mart",
    "gym leader",
    "badge",
    "pokédex",
    "bulbasaur",
    "charmander",
    "squirtle",
    "pikachu",
]

RAM_ADDRESS = re.compile(r"0x[0-9A-Fa-f]{4}")


def _module_path(name: str) -> Path:
    path = PACKAGE / name
    assert path.is_file(), f"expected module missing: {path}"
    return path


def _nav_module_path(name: str) -> Path:
    path = PACKAGE / "navigation" / name
    assert path.is_file(), f"expected navigation module missing: {path}"
    return path


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


def test_decision_loop_does_not_import_progress_probe() -> None:
    for name in DECISION_LOOP_MODULES:
        imports = _imported_modules(_module_path(name))
        offending = {module for module in imports if module == "llm_agent.progress" or module.startswith("llm_agent.progress.")}
        assert not offending, f"{name} imports the verification-only probe: {offending}"


def test_only_emulator_imports_pyboy() -> None:
    for name in DECISION_LOOP_MODULES:
        if name in PYBOY_ALLOWED:
            continue
        imports = _imported_modules(_module_path(name))
        offending = {module for module in imports if module == "pyboy" or module.startswith("pyboy.")}
        assert not offending, f"{name} must not import PyBoy directly: {offending}"


def test_no_ram_addresses_in_decision_loop() -> None:
    offenders: list[str] = []
    for name in DECISION_LOOP_MODULES:
        text = _module_path(name).read_text(encoding="utf-8")
        for match in RAM_ADDRESS.finditer(text):
            offenders.append(f"{name}: {match.group(0)}")
    assert not offenders, f"RAM addresses leaked into the decision loop: {offenders}"


def test_verification_probe_is_clearly_isolated() -> None:
    path = PACKAGE / "progress.py"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "VERIFICATION-ONLY" in text
    assert "never" in text.lower() and "decision loop" in text.lower()


# --------------------------------------------------------------------------- #
# Navigation v2 boundaries: screenshots only, no RAM / PyBoy / guide data
# --------------------------------------------------------------------------- #

def test_navigation_modules_do_not_import_progress_or_pyboy() -> None:
    for name in NAVIGATION_MODULES:
        imports = _imported_modules(_nav_module_path(name))
        forbidden = {
            module
            for module in imports
            if module == "llm_agent.progress"
            or module.startswith("llm_agent.progress.")
            or module == "pyboy"
            or module.startswith("pyboy.")
        }
        assert not forbidden, f"navigation/{name} imports hidden state: {forbidden}"


def test_navigation_modules_have_no_ram_addresses() -> None:
    offenders: list[str] = []
    for name in NAVIGATION_MODULES:
        text = _nav_module_path(name).read_text(encoding="utf-8")
        for match in RAM_ADDRESS.finditer(text):
            offenders.append(f"navigation/{name}: {match.group(0)}")
    assert not offenders, f"RAM addresses leaked into navigation: {offenders}"


#: Pure engine modules (viz.py is a UI renderer; placeholder words like
#: "badge" there are cosmetic, not game knowledge).
NAVIGATION_ENGINE_MODULES = [name for name in NAVIGATION_MODULES if name != "viz.py"]


def test_navigation_modules_have_no_story_knowledge() -> None:
    for name in NAVIGATION_ENGINE_MODULES:
        text = _nav_module_path(name).read_text(encoding="utf-8").lower()
        for term in ("badge", "gym", "map_id", "wobtainedbadges", "event_flags"):
            assert term not in text, f"navigation/{name} contains game-state knowledge: {term}"


def test_runtime_uses_navigation_high_level_tools() -> None:
    """The runtime/tool layer must expose the navigation tools to the brain."""
    from llm_agent.contracts import NAVIGATION_TOOLS, TOOL_NAMES
    from llm_agent.tools import TOOL_SCHEMAS

    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert set(NAVIGATION_TOOLS).issubset(set(TOOL_NAMES))
    assert set(NAVIGATION_TOOLS).issubset(names)


def test_system_prompt_has_no_story_spoilers() -> None:
    from llm_agent.prompts import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    found = [term for term in FORBIDDEN_PROMPT_TERMS if term in lowered]
    assert not found, f"system prompt leaks story knowledge: {found}"
    assert "blind" in lowered
    assert "screenshot" in lowered
    assert "record_discovery" in lowered
    assert "tool" in lowered


def test_prompt_forbids_walkthroughs_and_hidden_state() -> None:
    from llm_agent.prompts import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    for phrase in ("walkthrough", "hidden", "screenshot", "memory"):
        assert phrase in lowered


def test_tool_schemas_are_complete_and_constrained() -> None:
    from llm_agent.contracts import BUTTONS, MAX_ACTIONS_PER_CALL, MAX_WALK_STEPS, TOOL_NAMES
    from llm_agent.tools import TOOL_SCHEMAS

    names = []
    for schema in TOOL_SCHEMAS:
        assert schema.get("type") == "function"
        function = schema["function"]
        assert function["name"] and function["description"]
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        assert parameters.get("type") == "object"
        names.append(function["name"])
    assert set(TOOL_NAMES).issubset(set(names)), f"missing tools: {set(TOOL_NAMES) - set(names)}"

    by_name = {schema["function"]["name"]: schema["function"]["parameters"] for schema in TOOL_SCHEMAS}
    button_enum = by_name["press_button"]["properties"]["button"]
    assert set(button_enum.get("enum", [])) == set(BUTTONS)
    actions = by_name["execute_actions"]["properties"]["actions"]
    assert actions.get("maxItems") == MAX_ACTIONS_PER_CALL
    walk = by_name["walk"]["parameters"]["properties"] if "parameters" in by_name["walk"] else by_name["walk"]["properties"]
    assert walk["max_steps"].get("maximum") == MAX_WALK_STEPS


def test_contracts_have_no_game_knowledge() -> None:
    """Contracts/config/agent/memory/guard/map modules must not contain story state knowledge."""
    for name in ("contracts.py", "config.py", "agent.py", "memory.py", "guard.py", "mental_map.py"):
        text = _module_path(name).read_text(encoding="utf-8").lower()
        for term in ("badge", "gym", "map_id", "wobtainedbadges", "event_flags"):
            assert term not in text, f"{name} contains game-state knowledge: {term}"


def test_api_client_never_hardcodes_a_key() -> None:
    path = PACKAGE / "api_client.py"
    text = path.read_text(encoding="utf-8")
    assert "sk-" not in text
    assert "Authorization: Bearer" not in text
    lowered = text.lower()
    assert "api_key_env" in lowered or "os.environ" in lowered or "getenv" in lowered


@pytest.mark.parametrize("module", ["config", "contracts", "dotenv", "progress"])
def test_lead_modules_import_cleanly(module: str) -> None:
    __import__(f"llm_agent.{module}")
