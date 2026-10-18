# Notices and legal information

This repository contains research code for a vision-based autonomous agent that
plays **Pokémon Red** through the [PyBoy](https://github.com/Baekalfen/PyBoy)
Game Boy emulator using a large language model (DeepSeek API) as the decision
maker.

## Game data and intellectual property

* This repository **does not contain and never distributes** any game ROM,
  emulator save state, game recording, model weights or API key.
* To run the agent you must **lawfully own** an English *Pokémon Red* ROM and a
  compatible initial save state. The project only verifies the SHA1/SHA256 of
  files you provide (see `scripts/rom_checks.py`); it never downloads them.
* *Pokémon* and all related game content are trademarks / copyrights of their
  respective owners (Nintendo, Creatures Inc., GAME FREAK inc.). This project
  is an independent research effort with no affiliation and no endorsement.

## Third-party software

* [PyBoy](https://github.com/Baekalfen/PyBoy) — Game Boy emulator written in
  Python, MIT License (Copyright (c) 2016–2024 Mads Ynddal and contributors).
* [NumPy](https://numpy.org/) — BSD-3-Clause License.
* [Pillow](https://python-pillow.org/) — HPND License.
* DeepSeek API — governed by the DeepSeek platform terms of service. API usage
  cost is tracked by the agent (see `llm_agent/budget.py`) and is **not** free.

## Prior work archived on a branch

An earlier stage of this repository (2026) was an independent **PPO
reinforcement-learning reproduction** built on
[neroRL's `poke_red` branch](https://github.com/MarcoMeter/neroRL/tree/poke_red)
(MIT License, Copyright (c) 2019 Marco Pleines) and inspired by
[PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)
(MIT License). That code, its training scripts, configurations and reports were
removed from `main` during the repository cleanup on 2026-10-18 in order to
present a focused LLM-agent research repository.

The complete pre-cleanup state remains available (including all upstream license
and copyright notices) in the Git branch:

```
archive/ppo-reproduction
```

No Git history was rewritten. Run `git log archive/ppo-reproduction` or
`git checkout archive/ppo-reproduction` to inspect or restore it.

The current LLM-agent code is original work of this repository; the emulator
wrapper only follows the same action-timing convention (one button per ~24
emulated frames) that the earlier reproduction validated, which is an
algorithmic convention rather than copied source code.
