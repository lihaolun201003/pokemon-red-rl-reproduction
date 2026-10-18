# Pokémon Red Autonomous Agent

**Vision-Based Autonomous Exploration and Multi-Agent Decision-Making**

This repository studies how far a vision-based large language model agent can
autonomously play **Pokémon Red** on a real Game Boy emulator — using only the
game screen, its own actions and its own recorded experience. It is a research
codebase, not a product: nothing here claims the game has been completed.

> ⚠️ **Legal note (read first)**: this repository does **not** contain any game
> ROM, emulator save state, model weights, gameplay video or API key. To run the
> agent you must **lawfully own** an English *Pokémon Red* ROM and a compatible
> initial save state. SHA1/SHA256 checks only verify inputs you provide. See
> [NOTICE.md](NOTICE.md).

---

## 1. Project Overview

This is a research project that uses a real Game Boy game as an experimental
environment for vision-based autonomous agents. A vision LLM (DeepSeek) observes
the emulator screen and drives the game through constrained button presses; the
Python side supplies everything that is *not* a decision: emulator control,
visual localization, mapping, task scheduling, memory, budgets and safety.

The agent is a **blind first-time player**:

* it cannot read game RAM in the decision loop;
* it has no walkthrough, no pre-set map, no hidden coordinates, flags or object
  positions;
* every fact it stores must be justified by a screenshot or a recorded failed
  attempt;
* annotations of "did I make progress" come from an isolated, verification-only
  RAM probe (`llm_agent/progress.py`) used for reports/tests, never for acting.

The prior stage of this repository was a PPO reinforcement-learning reproduction.
That code, its training pipeline and its reports were removed from `main` to keep
the presentation focused; the complete old state is preserved on the
`archive/ppo-reproduction` Git branch (history was not rewritten).

## 2. Research Questions

* **Visual environment modelling** — can a vision model reconstruct a usable
  spatial map from 160×144 screenshots alone (tiles, walls, doors, NPCs)?
* **Autonomous exploration** — how should an agent choose where to explore
  without a map, and how should it know when an area is "done"?
* **Long-horizon decision making** — how to keep a coherent objective over
  thousands of game frames when every decision is a single LLM call?
* **Multi-agent task cooperation** — can independent agents (navigation, story,
  battle) share one emulator through a single-owner control protocol?
* **Memory and experience reuse** — does persisted knowledge actually improve
  decisions, or does it merely accumulate?
* **Repetition and futile behaviour** — why does an agent that already built a
  map still walk the same roads again?
* **Recovery and strategy revision** — when progress stalls, can the system
  detect it and change strategy (hint → review → checkpoint rollback)?

## 3. Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              Game Director                   │
                    │  deterministic Python scheduler; rare LLM    │
                    │  review; owns task graph & control decisions │
                    └───────┬──────────────┬──────────────┬────────┘
                            │              │              │
                   ┌────────▼───┐   ┌──────▼─────┐   ┌────▼───────┐
                   │ Navigation │   │   Story    │   │  Pokémon   │
                   │   Agent    │   │   Agent    │   │   Agent    │
                   │ (Python +  │   │ (vision    │   │ (vision    │
                   │  vision v2)│   │  LLM)      │   │  LLM)      │
                   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
                         │                │                │
             ┌───────────▼────────────────▼────────────────▼─────────┐
             │ Event Bus (persisted queue) · TaskManager/Scheduler    │
             │ SharedWorldMemory · CheckpointStore + branch manifests │
             │ ProgressMonitor/RecoveryManager · BudgetTracker        │
             └───────────────────────────┬───────────────────────────┘
                                         │
                             ┌───────────▼───────────┐
                             │ ControlArbiter        │
                             │ (single-owner lease)  │
                             └───────────┬───────────┘
                                         │
                             ┌───────────▼───────────┐
                             │ GuardedEmulator/PyBoy │
                             │  real 160×144 screen  │
                             └───────────────────────┘
```

* **Game Director** — deterministic scheduler. Classifies the screen
  (`overworld / dialog / battle / menu / transition`), enforces hard limits,
  starts/pauses/requeues tasks, runs progress checks and recovery. It never
  presses buttons itself and calls the LLM only for rare review/proposal.
* **Navigation Agent** — Navigation v2: pure-vision localization on a 16×16 tile
  grid, camera-follow displacement matching, permanent per-area grids, world
  graph across areas, frontier-based exploration, A* movement with per-step
  verification and re-planning. DeepSeek only chooses *what/where*; Python
  decides *how*.
* **Story Agent** — takes over real dialogue screens: reads the dialogue from
  screenshots, advances it with bounded real button presses, records NPCs and
  clues with evidence, then releases control and proposes follow-up tasks.
* **Pokémon Agent** — takes over battles: detects the battle menu from pixels,
  waits out intro text with bounded A presses, and each turn asks the vision LLM
  for a decision (fight / move slot / run / catch / switch) which a deterministic,
  verified menu driver executes; battle experience is recorded.
* **Shared infrastructure** — persisted Event Bus (dedupe, priority, replay),
  `ControlArbiter` with explicit leases and a `GuardedEmulator` that rejects
  presses from non-owners, TaskManager (DAG, retries, branch awareness),
  deterministic TaskScheduler with explainable scores, SharedWorldMemory
  (evidence, confidence, HISTORICAL marking after rollback), hierarchical
  ProgressMonitor, RecoveryManager (L1 hint / L2 director review / L3 checkpoint
  rollback with budgets), branch checkpoint manifests (state SHA256 + tasks +
  director + navigation + memory + message progress + budget), persistent
  `BudgetTracker` and a per-agent `ModelRouter`.

## 4. Implemented Features

* Real PyBoy emulator integration: one button per ~24 frames, screenshots only,
  save-state round-trips, optional Tk viewer with live game screen, navigation
  map panel and multi-agent status panel.
* Pure-vision localization and permanent maps: per-area logical grid (observed /
  walked / blocked / temporary-blocked / exit / interactive / confidence /
  evidence), multi-area world graph, re-entry relocalization, map gallery export.
* Autonomous exploration with frontier selection and bounded per-task budgets;
  honest coverage statistics.
* Single-owner control arbitration: `navigation / story / pokemon / system`
  owners, deny-by-default, no deadlock on owner failure.
* Multi-agent coordination: persisted message queue with content dedupe and
  message-id idempotency, task delegation (Story → Navigation), branch-aware
  tasks and memory.
* Story takeover of real dialogue and Pokémon takeover of real battles (both
  demonstrated on the real ROM with the real DeepSeek API).
* Checkpoint/rollback system with full manifests and an explicit
  "knowledge survives rollback, marked HISTORICAL" rule; API spend is monotonic
  and never rewound by rollback.
* Progress monitoring (L1 local / L2 global / L3 rollback), safe pause at hard
  limits, and a written, machine-readable audit trail for every decision
  (`artifacts/.../audit/*.jsonl`, gitignored).
* Cost control: per-agent token limits, image downscaling, persistent run/total
  budgets, per-agent cost ledger.
* 1,248 retained unit/integration tests plus real-game acceptance scripts
  (G0–G7).

## 5. Experimental Results

All numbers below come from real runs on the real ROM (and real DeepSeek API where
stated); full evidence and honest caveats are in the linked reports.

**Navigation v2 (real ROM + real PyBoy):**

* G1 localization: 122 directional operations, 88 real displacements all
  classified correctly, **0 wrong / 0 missed / 0 UNKNOWN / 0 drift**; includes
  turns, walls, menu/battle transitions and a door teleport.
* G2 unfamiliar room: built Oak's Lab from scratch — **353 logical cells
  (79 walkable / 56 blocked / 1 exit), coverage 97.53%**, finished autonomously.
* G3 re-entry: relocalized with score **0.9213**, **0 duplicate areas**, reached
  the furthest known point in 14 steps (5.2% of the 268-step first exploration).
* G4 unattended long run (no LLM): 426 real directional operations, 259 real
  displacements, 4 areas, 1,164 new cells, 0 planning failures.
* G4-API (real DeepSeek): 8 API calls (**$0.0031**) drove **159 verified
  navigation steps** (~20 steps per call).

**Multi-Agent v1 (real ROM; G2/G3/G7 with real DeepSeek):**

* G0 regression: 1,248 tests passing; Navigation v2 validation scripts pass.
* G1 control exclusivity: max 1 concurrent lease, unauthorized presses do not
  advance frames, deterministic denial, precise handoff (3 presses = 72 frames,
  2 presses = 48 frames), no deadlock after owner failure.
* G2 real dialogue takeover: real Oak dialogue read and recorded, 2 Story
  sessions, 20 API calls, **$0.0062**, control released and navigation resumed.
* G3 real battle takeover: wild battle on Route 1, 7 A-presses / 284 frames of
  intro, then 3 real decision turns and **45 verified button presses → win**,
  4 API calls, **$0.0014**; navigation task resumed afterwards.
* G4 cross-agent delegation: Story proposed a navigation task, Director executed
  it, parent task resumed; duplicate message-id and duplicate-content handling
  verified.
* G5 rollback: real state restore + new branch, 8 old knowledge entries marked
  HISTORICAL (not deleted), post-rollback learning kept, API spend not rewound.
* G6 stagnation: 4 stagnation signals → L1/L2/L3 recoveries; per-lineage rollback
  cap honoured; safe pause at the hard limit.
* G7 integrated real run: in one 110-step run the Director dispatched all three
  agents — **14 real battles, 5 real dialogues, 1,023 navigation operations,
  16 areas, 19 checkpoints, 65 API calls, $0.0245**. Across five repeated
  runs only one formed this complete loop; exploration variance and remaining
  loops are reported honestly in the results document.

**Earlier single-agent real run:** 228 real decisions / 250 tool calls, about
**$0.055**, left the starting room, reached Route 1, fought several wild battles
(starter L6 → L7) — **no badge, no new town, no completion**.

## 6. Current Limitations

* **The agent can construct and reuse visual maps, but still struggles to
  translate spatial memory into effective long-horizon exploration policies.**
  After an area is mapped it still re-walks already-explored roads, and exits
  are sometimes used in a back-and-forth pattern.
* Long-horizon planning is unstable: objective choice across many areas is still
  mostly reactive (story proposals + frontier exploration), not a persistent plan.
* Story understanding is not reliable: dialogue reading works, but turning it
  into correct medium-term goals is still weak.
* The practical effect of experience learning (battle lessons, NPC facts) is not
  yet validated — memory accumulates, but measurable strategy improvement is
  still an open question.
* Long runs show duplicate/low-value tasks and occasional localization drift;
  recovery exists (visual relocalization, door ping-pong detection, safe pause)
  but the underlying exploration policy is the bottleneck.
* Battle strategy is naive (no type/stat knowledge by design); capture, item and
  switching flows are not separately validated on the real ROM.
* No in-process global HTTP watchdog for a hung API read; a stalled request can
  block a run (mitigated in the acceptance harness by timeouts/watchdog).
* Not tested for multi-hour continuous runs; only one G7 run in five produced a
  fully closed interaction loop.
* This project does **not** claim to complete the game, and should not be
  presented as a finished player.

## 7. Quick Start

**Requirements:** Windows 10/11 + Python 3.10 (only environment tested), a
lawfully owned English Pokémon Red ROM, optionally a DeepSeek API key.

```powershell
# 1. install (no PyTorch / CUDA / PPO dependencies)
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt      # runtime
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt  # tests

# 2. place your lawful game data (gitignored, never committed)
#    local\PokemonRed.gb                (SHA1 ea9bcae617fdf159b045185467ae58b2e4a48b9a)
#    local\has_pokedex_nballs.state     (initial state used by the reports)

# 3. optional API key (only needed for real LLM runs)
Copy-Item .env.example .env   # then fill DEEPSEEK_API_KEY=...
#    without a key you can still run all non-API checks below

# 4. environment + ROM checks (no API cost)
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe scripts\check_game.py --steps 64

# 5. run the multi-agent system on the real game
.\.venv\Scripts\python.exe scripts\run_multi_agent.py --config configs\multi_agent_v1.json --viewer
#    add --headless for no window; --print-config to inspect the effective config

# 6. single-agent legacy entry (debugging tool)
.\.venv\Scripts\python.exe scripts\run_llm_agent.py --config configs\navigation_v2_api.json --headless

# 7. tests (no ROM needed) and real Navigation v2 validation
.\.venv\Scripts\python.exe -m pytest tests\llm_agent tests\navigation tests\multi_agent tests\story tests\pokemon -q
.\.venv\Scripts\python.exe scripts\nav_v2_g1_localization.py
```

Budget safety: the agent tracks API cost persistently (`llm_agent/budget.py`);
runs pause safely when a limit is reached. A G7-scale run currently costs about
$0.02–0.03.

## 8. Reports

Design, validation and real-run evidence:

* Navigation v2 — [design](docs/reports/navigation_v2_design.md) ·
  [validation](docs/reports/navigation_v2_validation.md) ·
  [results](docs/reports/navigation_v2_results.md)
* Multi-Agent v1 — [design](docs/reports/multi_agent_v1_design.md) ·
  [validation](docs/reports/multi_agent_v1_validation.md) ·
  [real results](docs/reports/multi_agent_v1_results.md) ·
  [known limitations](docs/reports/multi_agent_v1_known_limitations.md)
* Single-agent LLM agent — [design](docs/reports/llm_agent_design.md) ·
  [validation](docs/reports/llm_agent_validation.md) ·
  [first real run](docs/reports/llm_agent_run_report.md)
* Research overview for external discussion:
  [docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md)
* Repository cleanup report:
  [docs/reports/repository_cleanup_report.md](docs/reports/repository_cleanup_report.md)
* Archived PPO reproduction (old code, old reports, full history):
  `archive/ppo-reproduction` branch.

---

**License:** MIT (this repository). Third-party notices: [NOTICE.md](NOTICE.md).
Pokémon and related content belong to their respective rights holders.
