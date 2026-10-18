"""Environment doctor for the Pokémon Red autonomous-agent project.

Reports (never modifies) the local setup needed to run the vision-based LLM
agents: Python/emulator/vision dependencies, the user's lawful ROM + initial
save state hashes, the optional DeepSeek API key status, and the project
layout.  Useful when handing the repository to someone else.

    .\\.venv\\Scripts\\python.exe scripts\\doctor.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rom_checks import (  # noqa: E402
    ROM_FILENAME,
    ROM_SHA1,
    STATE_FILENAME,
    STATE_SHA256,
    sha1_file,
    sha256_file,
)

#: Runtime dependencies of the LLM agents (see requirements.txt).
REQUIRED_MODULES = ("pyboy", "numpy", "PIL")
#: Optional modules used by tooling / GUI / tests.
OPTIONAL_MODULES = ("tkinter", "pytest")


def _module_version(module_name: str) -> str:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report, never crash
        return f"import failed: {type(exc).__name__}: {exc}"
    distribution = {"PIL": "Pillow", "pyboy": "pyboy"}.get(module_name, module_name)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "importable (version unknown)"


def emulator_probe(window: str = "null") -> dict:
    """Boot a tiny synthetic cartridge (no Nintendo ROM) and exercise PyBoy APIs."""
    from pyboy import PyBoy

    with tempfile.TemporaryDirectory(prefix="pokemon_agent_doctor_") as directory:
        rom = bytearray(32768)
        rom[0x100:0x104] = bytes([0x00, 0xC3, 0x50, 0x01])
        rom[0x134:0x13C] = b"LLMPROBE"
        rom[0x150:0x152] = bytes([0x18, 0xFE])
        checksum = 0
        for byte in rom[0x134:0x14D]:
            checksum = (checksum - byte - 1) & 255
        rom[0x14D] = checksum
        path = Path(directory) / "probe.gb"
        path.write_bytes(rom)
        emulator = PyBoy(str(path), window=window, sound_emulated=False)
        try:
            emulator.set_emulation_speed(0)
            emulator.tick(120, True)
            assert emulator.screen.ndarray.shape == (144, 160, 4)
            state = io.BytesIO()
            emulator.save_state(state)
            emulator.tick(2, True)
            state.seek(0)
            emulator.load_state(state)
            emulator.tick(2, True)
        finally:
            emulator.stop(save=False)
    return {"cartridge": "synthetic diagnostic (not Pokémon)", "window": window,
            "tick_save_load_close": "passed"}


def rom_report() -> dict:
    candidates = sorted({*PROJECT_ROOT.glob("local/*.gb"), *PROJECT_ROOT.glob("local/*.gbc"),
                         *PROJECT_ROOT.glob("*.gb"), *PROJECT_ROOT.glob("*.gbc")})
    rows = []
    for path in candidates:
        try:
            digest = sha1_file(path)
            rows.append({
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha1": digest,
                "canonical_english_red": digest == ROM_SHA1,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    state = PROJECT_ROOT / "local" / STATE_FILENAME
    state_info: dict = {"path": f"local/{STATE_FILENAME}", "present": state.is_file()}
    if state.is_file():
        actual = sha256_file(state)
        state_info.update({"bytes": state.stat().st_size, "sha256": actual,
                           "canonical": actual == STATE_SHA256})
    return {
        "rom_filename": ROM_FILENAME,
        "canonical_rom_sha1": ROM_SHA1,
        "research_rom_present": any(row.get("canonical_english_red") for row in rows),
        "rom_candidates": rows,
        "initial_state": state_info,
    }


def api_key_report() -> dict:
    """Report the API key presence without ever printing the key itself."""
    try:
        from llm_agent.config import AgentConfig
        from llm_agent.dotenv import load_env_file

        load_env_file(str(PROJECT_ROOT / ".env"))
        cfg = AgentConfig()
        return {"status": cfg.api_key_status(), "configured": cfg.api_key() is not None,
                "env_var": cfg.api.api_key_env, "model": cfg.api.model,
                "base_url": cfg.api.base_url}
    except Exception as exc:  # noqa: BLE001
        return {"status": f"check failed: {type(exc).__name__}: {exc}", "configured": False}


def project_layout() -> dict:
    expected = [
        "llm_agent", "llm_agent/navigation", "llm_agent/multi_agent", "llm_agent/story",
        "llm_agent/pokemon", "scripts", "configs", "tests",
        "configs/multi_agent_v1.json", "scripts/run_multi_agent.py",
        "scripts/run_llm_agent.py", "docs/reports",
    ]
    return {name: (PROJECT_ROOT / name).exists() for name in expected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "artifacts" / "doctor.json"))
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when the ROM/state/dependencies are not ready")
    args = parser.parse_args()

    report = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "project_root": str(PROJECT_ROOT),
        "modules": {name: _module_version(name) for name in REQUIRED_MODULES + OPTIONAL_MODULES},
        "project_layout": project_layout(),
        "rom": rom_report(),
        "deepseek": api_key_report(),
    }
    try:
        report["pyboy_probe"] = emulator_probe()
    except Exception as exc:  # noqa: BLE001
        report["pyboy_probe"] = {"error": f"{type(exc).__name__}: {exc}"}

    missing = [name for name in REQUIRED_MODULES if str(report["modules"][name]).startswith("import failed")]
    report["ready"] = {
        "dependencies": not missing,
        "rom": bool(report["rom"]["research_rom_present"]),
        "initial_state": bool(report["rom"]["initial_state"].get("canonical")),
        "api_key": bool(report["deepseek"].get("configured")),
        "all_critical": not missing
        and bool(report["rom"]["research_rom_present"])
        and bool(report["rom"]["initial_state"].get("canonical")),
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["ready"]["all_critical"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
