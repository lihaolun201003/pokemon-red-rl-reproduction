# Repository Cleanup Report — from PPO reproduction to the vision LLM Agent project

Date: 2026-10-18
Repository: `https://github.com/lihaolun201003/pokemon-red-rl-reproduction`
Working branch: `cleanup/llm-agent-focus` (merged back into `main` after verification)
Pre-cleanup commit (archived): **`385d40786a5cd1a1854360534d1ee0ab623301e7`**

This report documents the cleanup that turned the repository from a mixed
PPO-reproduction + LLM-agent codebase into a focused **vision-based LLM agent
research project**, without changing agent behaviour, game strategy or the
already-validated modules.

---

## 1. Safety backup (done before any deletion)

| Item | Value |
|---|---|
| Branch at cleanup start | `main` |
| Latest commit before cleanup | `385d407` (includes the Navigation v2 / Multi-Agent v1 work and the live-map viewer improvements) |
| Archive branch created | **`archive/ppo-reproduction`** → `385d407` |
| Pushed to GitHub | yes (`refs/heads/archive/ppo-reproduction` = `385d407`) |
| Cleanup branch | `cleanup/llm-agent-focus` |
| History rewritten / force push | **no** (history fully intact) |

Recovery of the old PPO project (code, training configs, scripts, reports, plots,
upstream notices):

```powershell
git fetch origin archive/ppo-reproduction
git checkout archive/ppo-reproduction          # inspect the old state
git show archive/ppo-reproduction:neroRL/train.py
git show archive/ppo-reproduction:docs/reports/training_final_report.md
```

The archive branch was verified after pushing: `neroRL/train.py`,
`reproduction/runtime.py`, `docs/reports/training_final_report.md` and the rest of
the 334 pre-cleanup files are present.

## 2. Dependency analysis (before deleting anything)

Method: static import scan of every retained module and script, followed by a
real "clean virtualenv" install/run check.

* `llm_agent/` (core, `navigation/`, `multi_agent/`, `story/`, `pokemon/`) imports
  **only the standard library + NumPy + Pillow + PyBoy**. No `torch`, `neroRL`,
  `gymnasium`, `reproduction`, OpenCV, TensorBoard.
* The retained entry scripts (`run_llm_agent.py`, `run_multi_agent.py`,
  `multi_agent_acceptance.py`, `multi_agent_make_states.py`, `nav_v2_*.py`,
  `llm_agent_*.py`) have the same dependency closure.
* The retained tests (`tests/llm_agent`, `tests/navigation`, `tests/multi_agent`,
  `tests/story`, `tests/pokemon`) need only NumPy/Pillow/PyBoy/pytest.
* The **only** references to the old PPO world were comments in
  `llm_agent/emulator.py` and `llm_agent/progress.py` describing where the
  action-timing convention / verification-only RAM addresses came from. They
  were reworded to point at the archived branch; they were provenance notes, not
  runtime dependencies.
* Used **only** by the old pipeline: `neroRL/`, `reproduction/`, `tests/envs`,
  `tests/phase1`, the PPO training/eval/plot/supervisor scripts, the YAML
  training configs and the old training reports.

Conclusion: the LLM agent is independent of the deleted code; deleting it cannot
break the current system (and the verification below confirms that).

## 3. Removed content

### Directories (177 tracked files removed; archive branch keeps all of it)

| Path | Files | What it was |
|---|---|---|
| `neroRL/` | 74 | vendored PPO framework (upstream `poke_red` branch) |
| `docs/` PPO docs/img/upstream/old reports/figures | 43 | neroRL docs, `docs/img/`, `docs/upstream/`, phase-1/eval/training reports and training-curve figures |
| `configs/` training YAML + `configs/tune/` | 23 | PPO/Gym/Procgen training and tuning configs (kept: `multi_agent_v1.json`, `navigation_v2_api.json`) |
| `tests/envs/`, `tests/phase1/` | 15 | upstream Gym-env wrapper tests and PPO pipeline tests |
| `scripts/` PPO-only scripts | 10 | `train.py`, `eval_checkpoint.py`, `watch.py`, `supervisor.py`, `check_cuda.py`, `plot_training_curves.py`, `plot_eval_comparison.py`, `summarize_eval.py`, `probe_rom.py`, `setup.ps1` |
| `notebooks/`, `result/`, `reproduction/` | 7 | upstream notebooks/results and the PPO runtime wrapper |
| Root packaging leftovers | 5 | `UPSTREAM.json`, `gen_pypi_package.py`, `requirements-lock-cuda.txt`, `requirements-lock-cpu.txt`, `requirements-phase1.txt` |

Notes:

* `scripts/check_game.py`, `scripts/capture_state.py`, `scripts/doctor.py` were
  **not deleted**: they provided useful ROM/environment checks but depended on the
  removed code, so they were **rewritten** as PyBoy-only tools (see §5).
* A shared read-only helper `scripts/rom_checks.py` was added for lawful-ROM
  SHA1/SHA256 validation (no emulator memory, no game data).
* Two historical reports (`navigation_v2_design.md`, `navigation_v2_results.md`)
  contained a sentence saying the PPO pipeline was "untouched"; they now state
  that the pipeline is archived on `archive/ppo-reproduction`.

## 4. Kept content (the current project)

* `llm_agent/` — emulator wrapper, DeepSeek client, observation/vision,
  tools, memory, guard, budget, runtime, viewer, progress (verification-only),
  `navigation/` (Navigation v2), `multi_agent/` (director, event bus, arbiter,
  tasks/scheduler, shared memory, progress/recovery, checkpoints, model router,
  integration), `story/`, `pokemon/`.
* `scripts/` — `run_llm_agent.py`, `run_multi_agent.py`,
  `multi_agent_acceptance.py`, `multi_agent_make_states.py`, `nav_v2_*.py` (6
  validation/visualisation scripts), `llm_agent_api_probe.py`,
  `llm_agent_progress.py`, `llm_agent_summarize_run.py`, `check_game.py`,
  `capture_state.py`, `doctor.py`, `rom_checks.py`, `build_red.ps1` (lawful ROM
  build helper, no download).
* `configs/multi_agent_v1.json`, `configs/navigation_v2_api.json`.
* `tests/llm_agent` (364), `tests/navigation` (319), `tests/multi_agent` (341),
  `tests/story` (88), `tests/pokemon` (136) — **1,248 tests**.
* Reports: `llm_agent_*` (3), `navigation_v2_*` (3), `multi_agent_v1_*` (4),
  plus the new `docs/RESEARCH_OVERVIEW.md` and this cleanup report.
* Legal/attribution: `LICENSE` (MIT, project), `NOTICE.md` (rewritten with
  PyBoy/NumPy/Pillow/DeepSeek notices, game-IP disclaimer and archive-branch
  acknowledgement), `CITATION.cff` (rewritten for the current project).
* Safety rules preserved: no ROM/state/API key committed; `llm_agent/progress.py`
  remains verification-only and the import-boundary test still enforces it.

## 5. Dependency / packaging changes

| File | Change |
|---|---|
| `requirements.txt` | rewritten: **`pyboy==2.4.0`, `numpy>=1.24,<3`, `Pillow>=10`** only. No PyTorch/CUDA/Gym/TensorBoard/OpenCV. |
| `requirements-dev.txt` | new: `-r requirements.txt` + `pytest>=7.4,<9` (pytest 8 is the verified range; pytest 9 changed Windows console/signal behaviour around subprocesses) |
| `setup.py` | rewritten: packages `llm_agent*`, reads `requirements.txt`, no `neroRL` import / entry points / extras |
| `pyproject.toml` | unchanged (generic setuptools build backend) |
| `MANIFEST.in` | updated: project docs/configs only, prunes local data, keeps legal exclusions |
| `.gitignore` | cleaned of PPO-specific lines; keeps ROM/state/weights/video/`.env`/local-runtime exclusions and adds the local Chinese-named ROM file |
| `.code-workspace` | pytest target updated from `tests/phase1` to the five retained test directories |

### Environment verification (fresh virtualenv, no old packages)

Created `work/venv_clean` with a **fresh** Python 3.10 venv (not the old `.venv`)
and installed from the new requirements only:

```
pyboy 2.4.0, numpy 2.2.6, Pillow 12.3.0, pytest 8.4.2
`pip install -e .` succeeds; `import llm_agent..., scripts/run_multi_agent.py --print-config` works
```

Real results in the clean environment:

* `python -m pytest tests/llm_agent tests/navigation tests/multi_agent tests/story tests/pokemon -q`
  → **1248 passed** (same as the old environment; runtime uses NumPy 2.2.6 vs
  1.26.4 in the old venv, no behaviour differences observed).
* `scripts/doctor.py` → dependencies `true`, ROM `true`, initial state `true`,
  API key `true`, `all_critical: true`.
* `scripts/check_game.py --steps 32` → real ROM boot, 32 button presses,
  frames `(144,160,4)`, save/load-state round-trip: **passed** (no API calls).
* `scripts/multi_agent_acceptance.py --scenario g1` → **PASS** (real emulator,
  control-arbiter concurrency, no API).
* `scripts/nav_v2_g1_localization.py` → real-ROM localization:
  `pos_error_sum=0, pos_error_max=0, false_transitions=0, false_interruptions=0`.

### Bug found and fixed during clean-environment verification

The fresh venv does not contain `psutil` (it was a leftover PPO dependency). The
single-instance lock then fell back to `os.kill(pid, 0)` on Windows, where signal
0 is `CTRL_C_EVENT`: it delivered Ctrl+C to the console process group and caused
a spurious `KeyboardInterrupt` in an unrelated `subprocess` call (e.g. the
provenance `git rev-parse`). `llm_agent/runtime.py::_pid_alive` now uses the
Win32 `OpenProcess`/`GetExitCodeProcess` API on Windows and keeps `os.kill` only
for POSIX. This makes the lock correct without `psutil`; the import-boundary test
still forbids hex literals in decision-loop modules, so the Win32 flag is written
as a decimal constant with a comment.

## 6. README rewrite

`README.md` was rewritten as **"Pokémon Red Autonomous Agent — Vision-Based
Autonomous Exploration and Multi-Agent Decision-Making"** with the requested
sections:

1. Project Overview (blind first-player constraints; no RAM/walkthrough).
2. Research Questions (visual modelling, exploration, long-horizon decisions,
   multi-agent cooperation, memory reuse, repetition, recovery).
3. Architecture (Game Director + Navigation/Story/Pokémon agents + shared
   infrastructure + single-owner arbiter + guarded emulator diagram).
4. Implemented Features (only what exists).
5. Experimental Results (real numbers from Navigation v2 and Multi-Agent v1;
   no completion claims).
6. Current Limitations (including the required sentence: the agent can construct
   and reuse visual maps, but still struggles to translate spatial memory into
   effective long-horizon exploration policies).
7. Quick Start (clean install, lawful ROM/state, API key, real commands).
8. Reports (links to all retained design/validation/result documents and the
   archive branch).

New: `docs/RESEARCH_OVERVIEW.md` — a 1–2 page research summary for academic
discussion, centered on the question *why an agent that already built a map still
re-explores and makes inefficient long-horizon decisions* (observed evidence,
working hypotheses, open questions; no unverified novelty claims).

## 7. Tests and acceptance after cleanup

| Check | Result |
|---|---|
| Retained unit/integration tests (old venv) | **1248 passed** |
| Retained tests (fresh clean venv, new requirements) | **1248 passed** |
| `scripts/multi_agent_acceptance.py --scenario g0` (after harness update) | **PASS** — 319 + 364 + 341 + 88 + 136 tests + 3 Navigation v2 real scripts all rc=0 |
| Multi-agent entry config load | `run_multi_agent.py --print-config` works in clean venv |
| Real emulator, **no API cost** | G1 control acceptance PASS; `check_game.py` PASS |
| Navigation v2 unaffected | navigation tests 319 pass; real-ROM G1 script (position error 0) PASS |
| Delete-boundary check | no retained file imports `neroRL`/`reproduction`/`torch`/`gym`; no retained script/test references deleted paths (historical reports note the archive branch) |
| Old PPO restorable | `git show archive/ppo-reproduction:neroRL/train.py` etc. verified |
| Secret/ROM leak check | `git ls-files` contains no `*.gb/.gbc/.state/.sav/.pt/.env`, no key material; only `local/` (gitignored) holds the user's ROM |
| History | not rewritten; no force push |

The acceptance harness no longer runs `tests/phase1` (deleted); its G0 command
list was updated accordingly. No test was weakened or deleted to fake a pass —
the deleted tests tested code that no longer exists (archived branch keeps them).

## 8. Before / after metrics

| Metric | Before (`archive/ppo-reproduction`, `385d407`) | After (`main`) |
|---|---|---|
| Tracked files | **334** | **161** |
| Tracked content | **11.35 MiB** | **~3.24 MiB** |
| Top-level trees | `neroRL/`, `reproduction/`, `notebooks/`, `result/`, plus project dirs | `llm_agent/`, `scripts/`, `configs/`, `docs/`, `tests/` |
| Python packages installed for runtime | PyTorch/CUDA + Gym stack + PyBoy | **PyBoy + NumPy + Pillow** |
| Test suites | phase1 + envs + agent + nav + multi-agent | agent + nav + multi-agent + story + pokemon (1,248) |

Git history is unchanged (no rewrite): the old blobs remain reachable via
`archive/ppo-reproduction` and the repository's earlier commits.

## 9. Open / known items

* Historical reports still say "…passed 11 phase-1 tests" or mention
  `tests/envs` (they are dated validation records from before the cleanup; the
  code they refer to is on `archive/ppo-reproduction`). This report is the
  authoritative note for that difference.
* Multi-Agent behavioural limitations are unchanged (see
  `docs/reports/multi_agent_v1_known_limitations.md`); this task did not change
  agent strategy.
* No CI pipeline was added; verification is the documented local procedure.
* The GitHub repository name/visibility and Git history were not modified, and
  no force push was used.
