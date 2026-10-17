# Attribution and scope

This is an independent Phase 1 reproduction of **Pokémon Red via Reinforcement Learning**, not an official release by its authors.

The original neroRL implementation is by Marco Pleines and contributors, pinned to the `poke_red` branch and commit recorded in `UPSTREAM.json`. The original MIT license and copyright are retained verbatim in `LICENSE`. Original README and requirements are preserved in `docs/upstream/`.

The Pokémon environment is derived from Peter Whidden's [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments). Portions of the map code also acknowledge [pokemonred_puffer](https://github.com/thatguy11325/pokemonred_puffer) in their source headers. All existing source attribution is retained. The MIT license of PokemonRedExperiments is preserved in `docs/upstream/PWhiddy-LICENSE`.

Pokémon and related game content belong to their respective owners. No game ROM, save state, extracted artwork, pretrained model, or game video is included in the publishable tree. The source includes upstream RAM symbols and map metadata; this repository makes no claim to rights in game content. Any initial states in the ignored local upstream checkout are reference material only and are not part of the reproduction distribution.

Local changes add Windows launchers, dependency pins, observation dtype consistency, proper emulator shutdown, worker argument/cleanup fixes, finite-gradient checks, and checkpoint/provenance handling. The original network, PPO objective, GAE, AdamW optimizer, and reward formulas are reused.
