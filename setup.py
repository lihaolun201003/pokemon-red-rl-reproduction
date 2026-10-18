"""Packaging metadata for the Pokémon Red autonomous LLM agent.

The import package is :mod:`llm_agent` (Vision-based Navigation, Story and
Pokémon agents driven by a Game Director on top of PyBoy).  The earlier PPO
training pipeline is archived on the ``archive/ppo-reproduction`` branch and is
not part of this distribution.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _requirements() -> list[str]:
    lines = []
    for raw in _read("requirements.txt").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


setup(
    name="pokemon-red-autonomous-agent",
    version="1.0.0",
    description=(
        "Vision-based autonomous exploration and multi-agent decision-making on "
        "real Pokémon Red (PyBoy emulator + DeepSeek vision LLM)."
    ),
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/lihaolun201003/pokemon-red-rl-reproduction",
    project_urls={
        "Github": "https://github.com/lihaolun201003/pokemon-red-rl-reproduction",
        "Reports": "https://github.com/lihaolun201003/pokemon-red-rl-reproduction/tree/main/docs/reports",
    },
    author="lihaolun201003",
    license="MIT",
    keywords=[
        "Pokemon Red", "PyBoy", "Game Boy", "LLM agent", "vision", "autonomous exploration",
        "multi-agent", "navigation", "DeepSeek",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    packages=find_packages(include=["llm_agent", "llm_agent.*"]),
    package_data={"llm_agent": ["navigation/NAV_CONTRACT.md", "memory/README.md"]},
    include_package_data=True,
    install_requires=_requirements(),
    extras_require={"dev": ["pytest>=7.4"]},
)
