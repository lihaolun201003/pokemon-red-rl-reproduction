# Pokémon Red Autonomous Agent — Research Overview

*One-page research summary for academic discussion. Written to be read in about
10 minutes; all numbers are from real runs on a real Game Boy emulator, not
benchmarks or simulations.*

## 1. Motivation

Most reinforcement-learning game agents are trained with millions of environment
steps and dense reward shaping. This project asks a different question: **how far
can a general-purpose vision language model get in a long, open-ended console RPG
if it is only allowed to see what a human player sees** — the 160×144 game screen
— and must act through the same button interface?

The environment is *Pokémon Red* running in PyBoy. The project deliberately
imposes a **blind first-player constraint**: no game RAM in the decision loop, no
walkthrough or wiki, no pre-built map, no hidden coordinates/flags. Everything
the agent knows must come from screenshots, dialogue it read, and failed attempts
it recorded. A separate, isolated RAM probe exists only to *verify* progress for
reports, and an import-boundary test enforces that it can never be reached by the
decision code.

## 2. Environment and constraints

* **Observation**: one RGB screenshot per action (~24 emulated frames), plus the
  agent's own memory and action history.
* **Actions**: Game Boy buttons through a constrained tool interface; no
  arbitrary emulator manipulation, no save-scumming inside a run.
* **Hard constraint**: the LLM never controls the emulator directly. A
  single-owner control arbiter grants leases; every button press is checked
  against the lease; only the system owner may load/save state.
* **Cost**: every LLM call is real money. Runs are budgeted, and the agent
  pauses safely when a limit is reached rather than overspending.

## 3. System architecture (in brief)

A deterministic **Game Director** inspects the screen and decides *which agent
owns control right now*: dialogue screens go to a **Story Agent**, battles to a
**Pokémon Agent**, free exploration to a pure-vision **Navigation Agent**.
Navigation is mostly Python: visual localization on a 16×16 tile grid, a
permanent logical map per area, a world graph across areas, frontier-based
exploration, A* planning and per-step verification. The LLM contributes high
level decisions (what to explore, what to say/do in a dialogue, which move to
use), not per-frame control.

Supporting infrastructure: persisted event bus, task manager and scheduler,
shared world memory with evidence/confidence, hierarchical progress monitor,
recovery manager (local hint → director review → checkpoint rollback), branch
checkpoint manifests, per-agent model routing and persistent API budgeting.

## 4. What has been demonstrated

* **Visual map building works.** Navigation v2 localizes the player from pixels
  alone and builds permanent maps: 88/88 real displacements classified
  correctly in a controlled test; an unfamiliar room mapped to 353 logical cells
  with 97.53% coverage and an autonomous "done" decision; re-entering a room
  relocalizes with score 0.92 and produces **zero duplicate areas**; an
  unattended four-area run produced 1,164 new cells with zero planning failures.
* **Real hijacking of other game modes works.** Story Agent reads and advances a
  real dialogue, records NPC facts with screenshot evidence, releases control and
  resumes navigation. Pokémon Agent takes over a real wild battle, plays it to a
  win with verified button presses, then hands control back.
* **Multi-agent coordination works mechanically.** In one real 110-step run the
  Director dispatched all three agents: 14 real battles, 5 real dialogues, 1,023
  verified navigation operations across 16 areas, 19 checkpoints, 65 API calls,
  **$0.0245** total.
* **Failure handling exists.** Stagnation is detected at three levels, rollback
  restores the game while keeping learned knowledge (marked HISTORICAL), and API
  spend is monotonic across rollbacks by design.

## 5. The problem this project keeps running into: repetition

The most interesting and reproducible failure is this:

> **The agent can build a map of an area, finish exploring it with >97% coverage,
> and then still walk the same already-explored roads again — or repeatedly cross
> the same door between two areas — instead of acting on what its map already
> tells it.**

Concretely observed on the real game:

* after a room was fully mapped, a later extended run re-walked it instead of
  leaving; door transitions between two areas could alternate in a back-and-forth
  pattern (A→B→A within a few steps);
* the Director's "area finished → take the known exit" policy sometimes issued
  exit targets computed from a slightly drifted estimated pose, which then failed
  as "unreachable" and retried;
* five repeated integrated runs with identical code produced one complete
  battle/dialogue/exploration loop and four runs dominated by exploration
  variance or idle loops;
* in longer runs, many near-duplicate low-value tasks were created (e.g. repeated
  "explore this area" tasks), and a large fraction of decisions went to
  re-validating already-known terrain.

This is not a perception failure: the map is usually correct, and the agent can
read it. It is a **policy/planning failure on top of memory**.

## 6. Why this happens (working hypotheses, not conclusions)

These are hypotheses consistent with the observed logs; none has been validated
as a general explanation yet.

1. **Memory is spatial, but the objective is not.** The map answers "what is
   where"; it does not answer "what should I do now, and why". The scheduler has
   heuristics (frontier, story proposals) but no persistent, evaluated plan, so
   after finishing a local goal it defaults to cheap local behaviours (walk to
   the nearest unknown cell / nearest exit) that can cycle.
2. **Local decisions are made without a global cost model.** Choosing "nearest
   frontier" is myopic; the same door can look attractive repeatedly because the
   cost of *not making global progress* is invisible to the scorer.
3. **The "done" signal is local.** Coverage/ frontier heuristics say an area is
   complete, but completion does not translate into "now go somewhere else for a
   reason" — the loop from map knowledge to a new global intention is missing.
4. **Verification consumes the budget.** Per-step visual verification is what
   makes navigation reliable, but it also dominates wall-clock time and reduces
   how many high-level decisions a run can afford; with fewer decisions, the
   policy almost never gets to a second-order plan.
5. **LLM memory alone does not create strategy.** Story clues and battle lessons
   are recorded with evidence, but the agent does not yet use them to *score*
   alternative long-term objectives; the learned experience is descriptive
   rather than decision-shaping.

## 7. Open questions we would like advice on

* How to represent **global progress** so that local exploration policies can be
  scored against it (a persistent plan/goal stack? expected information gain per
  traversed cell? a learned value function over the map)? 
* How to make a **memory-to-intention** step: from "I know this town" to "the
  next useful thing is to leave it and why".
* How to detect and suppress **futile repetition** robustly without hard-coding
  routes — e.g. cost penalties for recently traversed edges, novelty-aware
  frontier selection, or explicit "already tried" memory with expiry.
* What is the right division of labour between the LLM and deterministic code for
  long-horizon planning? Currently Python owns movement and the LLM owns local
  semantics; perhaps the LLM should also own a slow, persistent plan.
* How should multi-agent memory be shared so that the Story Agent's clues can
  actually change the Navigation Agent's target selection?

## 8. Reproduction and evidence

All results above are reproducible from this repository with a lawfully owned
ROM and a DeepSeek API key. Design/validation/result documents and raw local
evidence (screenshots, audit JSONL, checkpoints) are linked from the
[README](../README.md#8-reports). The agent is not claimed to complete the game;
the project is offered as an experimental platform for studying memory,
exploration and long-horizon decision making in vision-based LLM agents.
