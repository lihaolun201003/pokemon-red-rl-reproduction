# DeepSeek LLM Game Agent — Internal Architecture & Module Contracts

Status: **design locked** for the first implementation round.  This document is the
coordination contract between the Lead and the module owners.  If an interface
change is genuinely required, message the Lead instead of editing another owner's
files.

Project root: `C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`
Python: `.venv\Scripts\python.exe` (3.10.11). PyBoy 2.4.0, numpy, Pillow and tkinter
are the runtime dependencies. **Do not install new packages** and do not use
`requests`/`httpx`/`openai`/`python-dotenv`/`pyyaml` — implement with the stdlib.

The earlier PPO pipeline (`neroRL/`, `reproduction/`, `scripts/train.py`,
checkpoints, reports) is archived on the `archive/ppo-reproduction` Git branch and
is no longer part of `main`. Current work lives only in `llm_agent/`,
`llm_agent/navigation/`, `llm_agent/multi_agent/`, `llm_agent/story/`,
`llm_agent/pokemon/`, the agent scripts under `scripts/`, and their tests.

---

## 1. Product goal (do not lose sight of it)

DeepSeek (`deepseek-flash`, vision + tool calls) is the **player**.  Loop:

```
DeepSeek Brain -> constrained tool calls -> PyBoy -> screenshot -> DeepSeek Brain
```

Hard rules for everything in `llm_agent/` except `progress.py`:

1. The decision loop may only see the **160x144 framebuffer**, the agent's own
   recorded memory, its action history, the budget, and the clock.
2. No RAM values, map ids, story flags, badge flags, party data or NPC positions
   may be read or inferred from the ROM / `pokered` source / walkthroughs.
   `progress.py` is the only exception: verification/reporting only, never
   imported by `runtime.py`, `agent.py`, `tools.py`, `prompts.py`, `guard.py`,
   `memory.py`, `emulator.py`, `observation.py`, `api_client.py` (a test enforces this).
3. The model cannot run code.  It can only call the validated tool set in
   `tools.py`.
4. No fixed walkthrough / scripted route.  Local path planning over self-recorded
   observations is allowed later; the first round is plainly blind exploration.
5. Every action batch must be verified against a fresh screenshot.

## 2. Environment facts (verified by the Lead on this machine)

* ROM: `local/PokemonRed.gb` (1 MiB, SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`,
  English `pret/pokered` build).  Initial save: `local/has_pokedex_nballs.state`
  (SHA256 `0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e`).
  Never modify/copy over these files.  Never commit them (already gitignored).
* PyBoy 2.4 usage (verified):
  ```python
  pyboy = PyBoy(rom, window="null", sound_emulated=False, log_level="ERROR")
  pyboy.set_emulation_speed(0)               # fastest
  with open(state, "rb") as f: pyboy.load_state(f)
  pyboy.tick(count=24, render=True)          # render=True required before capture
  rgba = pyboy.screen.ndarray                # (144, 160, 4) uint8, RGBA
  img  = pyboy.screen.image                  # PIL.Image 160x144 RGBA
  pyboy.button("a", delay=8)                 # press + release after `delay` frames
  pyboy.save_state(file_like) / pyboy.load_state(file_like)   # accepts BytesIO
  pyboy.frame_count, pyboy.stop(save=False)
  ```
  The DMG palette is exactly 4 grey levels: 0, 85, 153, 255 (RGBA).
  PyBoy is **not thread-safe**: only the agent-loop thread may touch it.  The
  Lead's viewer only receives already-copied frames.
* The archived PPO baseline used `press 8 frames + release, 24 frames per
  action`; the LLM agent reuses `press_frames=8, action_frames=24` as defaults.
* DeepSeek API (official docs, 2026-10): OpenAI-compatible
  `POST {base_url}/chat/completions`; model `deepseek-flash` (V4.1 Flash) has
  **vision** and **tool calls**; thinking mode is on by default and can be
  disabled with the request body field `{"thinking": {"type": "disabled"}}`
  (top-level in raw JSON, not nested in extra_body).  Non-thinking mode is the
  default for this agent to save cost; `reasoning_effort` only matters when
  thinking is enabled.  Images are sent as
  `{"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "original"}}`
  in **user** messages only (system/assistant images are a 400).  Image tokens
  are dimension-based, ≤1024 per image.  Usage body includes `prompt_tokens`,
  `completion_tokens`, `total_tokens`, `prompt_cache_hit_tokens`,
  `prompt_cache_miss_tokens`, `completion_tokens_details.reasoning_tokens`.
  Error codes: 400/401/402/422/429/500/503.  401 and 402 are fatal (stop for
  the user); 429 / 5xx / timeouts are retryable with exponential backoff.
  Non-streaming responses may prepend empty keep-alive lines: read the whole
  body, then `json.loads` the stripped text.

## 3. File ownership (do not edit outside your scope)

| Owner | Files |
|---|---|
| **Lead** | `llm_agent/__init__.py`, `contracts.py`, `config.py`, `dotenv.py`, `prompts.py`, `agent.py`, `runtime.py`, `viewer.py`, `progress.py`, `scripted.py`, `scripts/run_llm_agent.py`, `tests/llm_agent/conftest.py`, `tests/llm_agent/test_runtime.py`, `tests/llm_agent/test_import_boundaries.py`, `docs/reports/llm_agent_*.md`, `.env.example`, `.gitignore`, README section |
| **emulator-tools** | `llm_agent/emulator.py`, `observation.py`, `tools.py`, `tests/llm_agent/test_emulator.py`, `test_observation.py`, `test_tools.py` |
| **api-budget** | `llm_agent/api_client.py`, `budget.py`, `scripts/llm_agent_api_probe.py`, `tests/llm_agent/test_api_client.py`, `test_budget.py`, `test_dotenv.py` |
| **memory-guard** | `llm_agent/memory.py`, `guard.py`, `tests/llm_agent/test_memory.py`, `test_guard.py` |

Read `llm_agent/contracts.py` and `llm_agent/config.py` first — they are the
frozen shared boundary.  Additive helpers in your own files are fine.  Tests use
`.venv\Scripts\python.exe -m pytest tests\llm_agent\test_xxx.py -q` from the
project root.  All tests must pass **without** a DeepSeek API key and without
network access (API tests use a fake transport).  Real ROM/state tests must
`pytest.skip` when `local/PokemonRed.gb` / `local/has_pokedex_nballs.state` are
absent (they are currently present).

## 4. Shared contracts (already written)

`contracts.py` defines: `BUTTONS`, `DIRECTIONS`, limits (`MAX_ACTIONS_PER_CALL=32`,
`MAX_WAIT_FRAMES=600`, `MAX_WALK_STEPS=20`, `MAX_DISCOVERY_CHARS=1200`, …),
error classes (`ToolValidationError`, `ApiError` family, `BudgetExceededError`,
`CheckpointError`, `LockError`, `BlockedError`, `StopRequested`), `ScreenEvent`
enum and `SIGNIFICANT_EVENTS`, `Frame`, `ActionStep`, `ActionOutcome`, `ToolCall`,
`ToolResult`, `ApiUsage`, `ModelReply`, `BudgetState`, `RuntimeState`,
`HistoryEntry` and callback aliases `SaveProgressFn`, `RestoreProgressFn`,
`ScreenWriter`.

`config.py` defines `ApiConfig`, `BudgetConfig`, `EmulatorConfig`,
`MemoryConfig`, `GuardConfig`, `RuntimeConfig`, `ViewerConfig`, `AgentConfig`
(with `.load(config_path=None, env_file=..., overrides=...)`, `.validate()`,
`.api_key()`, `.resolve_paths()`) and default paths.  `dotenv.py` has
`parse_env_file/load_env_file/mask_secret/get_env/looks_like_placeholder`.
Do not hardcode absolute paths; use `config.ROOT`.

`runtime.py` (Lead) orchestrates: it builds one *decision cycle* per loop —
capture frame → build prompt (screenshot + memory + recent history + guard
warnings) → `api_client.chat()` → execute returned `ToolCall`s in order via
`ToolExecutor` → feed `ActionOutcome`s to `LoopGuard` → checkpoint/budget/log →
repeat.  Tools must be independent of the runtime: everything they need is
injected by the constructor.

---

## 5. `emulator.py` (owner: emulator-tools)

```python
class GameBoyEmulator:
    def __init__(self, cfg: EmulatorConfig): ...
    def start(self, state_path: str | Path | None = None) -> "GameBoyEmulator"   # boots PyBoy, loads state
    def load_state(self, path: str | Path | bytes) -> None                        # bytes = raw state file content
    def save_state_bytes(self) -> bytes                                           # BytesIO roundtrip
    def save_state_file(self, path: str | Path) -> Path                           # atomic; verifies loadable? at least size>0
    def press(self, button: str, *, frames: int | None = None) -> Frame           # one validated press, returns new frame
    def wait(self, frames: int) -> Frame                                          # tick without input
    def observe(self) -> Frame                                                    # capture current framebuffer (tick(1, render=True) if nothing rendered yet)
    def run_batch(self, buttons: Sequence[str], *, stop_on_event: bool = True,
                  step_hook: Callable[[ActionStep], None] | None = None) -> ActionOutcome
    def close(self) -> None
    @property frame_count(self) -> int
    @property loaded_state(self) -> str | None
```
* Validate buttons against `contracts.BUTTONS`, raise `ToolValidationError`.
* Each press: `send_input`-equivalent via `pyboy.button(button, delay=press_frames)`
  then `tick(action_frames - press_frames, render=True)` (guard the remainder ≥ 1).
  Equivalent to pressing for `press_frames` frames and releasing inside a
  `action_frames` budget.  Capture final frame via `frame_from_rgba`.
* `run_batch` presses sequentially; after each step compute `ActionStep`
  (`screen_changed`, `changed_ratio`, `event`) using `observation.analyze`.
  Stop early when `stop_on_event` and the new event ∈ `SIGNIFICANT_EVENTS`
  (reason `"event"`); stop when `frames >= action_frames * len(buttons)` budget
  exhausted (reason `"max_steps"`); a movement button that leaves the screen
  pixel-identical twice in a row may be reported as `stopped_reason="blocked"`
  only for `walk()`-style batches — keep `run_batch` simple: never silently stop
  for blocking, let the caller see unchanged steps.  Fill `summary` text.
* `load_state` must raise `EmulatorError` on failure, and set `frame_count`
  counters consistently (PyBoy's own `frame_count` may reset; keep `self.frame_count`
  monotonic by tracking a base offset).
* No logging of ROM contents; no RAM reads anywhere in this class.

`observation.py` pure functions (no PyBoy import):

```python
def frame_from_rgba(frame_index: int, rgba: np.ndarray, *, now: float | None = None) -> Frame
def analyze(current: Frame, previous: Frame | None = None) -> ScreenAnalysis   # dataclass below
def encode_for_api(frame: Frame, *, scale: int, image_format: str, jpeg_quality: int = 90) -> tuple[bytes, str]  # (bytes, mime)
def save_png(frame: Frame, path: str | Path) -> Path
def upscale_rgba(rgba: np.ndarray, scale: int) -> np.ndarray
def diff_ratio(a: Frame | np.ndarray, b: Frame | np.ndarray) -> float
def perceptual_signature(frame: Frame) -> str   # stable hex (e.g. 16x16 average hash over luma)
def is_black_frame(frame: Frame) -> bool        # >=98% of pixels at luma 0
def screen_event(current: Frame, previous: Frame | None = None, changed_ratio: float | None = None) -> ScreenEvent
def describe_screen(frame: Frame, analysis: ScreenAnalysis | None = None) -> str   # 1-3 line human/LLM text
def analyze_and_describe(current: Frame, previous: Frame | None = None) -> ScreenAnalysis
```
`ScreenAnalysis` dataclass (in `observation.py`, import into tools/runtime):
`event: ScreenEvent`, `changed: bool`, `changed_ratio: float`,
`signature: str`, `is_black: bool`, `luma_mean: float`, `summary: str`,
`stats: dict` (e.g. bottom-box white ratio, menu-candidate flags — pixel-only).
Heuristics must be cheap (<5 ms/frame), deterministic, documented, and tested
with synthetic images.  Suggested: black = ≥98% luma 0; map_change = changed
ratio ≥ 0.55 with normal luma; dialog = bottom band (y≈88..136) has a large
near-white region (≥45% of the band) with a dark border row; menu = right half
predominantly white with sparse dark pixels + no dialog; battle = distinct
layout heuristic (e.g. top-left large white HP panel + bottom-right white text
panel) with confidence in `stats["battle_score"]`; otherwise NONE.  Keep it
conservative: misclassification only affects triggers, the model always sees the
true screenshot.

## 6. `tools.py` (owner: emulator-tools)

Expose `TOOL_SCHEMAS: list[dict]` in OpenAI function-calling format and:

```python
class ToolExecutor:
    def __init__(self, *, emulator: GameBoyEmulator, memory: Any, state: RuntimeState,
                 screen_dir: str | Path, save_progress: SaveProgressFn,
                 restore_progress: RestoreProgressFn,
                 screen_writer: ScreenWriter | None = None,
                 cfg: AgentConfig | None = None): ...
    def execute(self, call: ToolCall) -> ToolResult          # never raises for invalid calls; returns ok=False + error
    def validate(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]
    def last_observation(self) -> tuple[Frame, ScreenAnalysis] | None
    attributes: last_frame: Frame | None, last_analysis: ScreenAnalysis | None,
                last_outcome: ActionOutcome | None, last_tool: str
```
* Strict validation: reject unknown tool names/keys, wrong types, out-of-range
  values, non-string buttons, batches > `MAX_ACTIONS_PER_CALL`, discovery text
  too long, etc.  Errors are returned to the model (`ToolResult(ok=False, error=...)`)
  and recorded in audit — never crash the run.
* `memory` is called through the documented `AgentMemory` interface (§8) but
  type as `Any`/protocol so tools stay unit-testable with a fake memory.
* `save_progress` calls the injected callback and appends a `RuntimeState`
  history entry.  `restore_checkpoint` validates the id format (`latest` or
  `checkpoint_*`) and calls the injected callback.
* Screenshot persistence: the runtime injects `screen_writer` (or `screen_dir`
  fallback) and the executor saves a PNG of the final frame for audit; returned
  `ToolResult.screens` are file **names** (relative).
* `ToolResult.summary` is fed back to the model: keep it ≤ ~700 chars, factual,
  no hidden state.  Include: action count, changed steps, event, final screen
  signature (8 chars), black flag, stopped_reason, goal/memory confirmation.
* Tools (names fixed): `observe_game`, `press_button`, `execute_actions`,
  `wait_frames`, `walk`, `inspect_memory`, `record_discovery`, `set_goal`,
  `save_progress`, `restore_checkpoint`, `get_recent_history`.
  - `walk(direction, max_steps=10, stop_on_event=true)`: presses one direction
    until two consecutive pixel-identical screens (blocked) or a significant
    event or `max_steps`.  Reason strings: `blocked|event|max_steps`.
  - `set_goal` also sets `state.goal/goal_reason` and calls `memory.set_goal`.
  - `record_discovery` calls `memory.record_discovery(text, kind=kind, place=place)`.
  - `inspect_memory` calls `memory.search(...)` / `memory.render_context(...)`.
  - `get_recent_history` reads `state.recent_history(n)`.
  - `execute_actions`/`walk`/`press_button`/`wait_frames` set `needs_llm=True`
    when a significant event is detected (dialogue/battle/menu/map change) or
    after an unexpected outcome, so the runtime stops the remaining batch.
* Update `state.buttons_pressed`, `state.action_batches`, `state.frame_index`,
  `state.last_action`, `state.discoveries/places_discovered` (from memory stats),
  and append `state.add_history(...)`.

## 7. `api_client.py` + `budget.py` (owner: api-budget)

`DeepSeekClient`:
```python
class DeepSeekClient:
    def __init__(self, cfg: ApiConfig, *, transport: Transport | None = None,
                 on_attempt: Callable[[dict], None] | None = None): ...
    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             max_tokens: int | None = None, thinking: str | None = None,
             reasoning_effort: str | None = None, timeout_s: float | None = None) -> ModelReply
    def probe_text(self, prompt: str) -> ModelReply          # tiny request, used by probe script
    @staticmethod
    def image_content_block(png_bytes: bytes, mime: str, *, detail: str = "original") -> dict
    def api_key_status(self) -> str                          # masked, never the key
```
* Stdlib `urllib.request` (or `http.client`); no third-party HTTP libs.
  `Transport` is a tiny protocol (`def post(url, headers, body: bytes, timeout_s) -> tuple[int, dict, bytes]`)
  with a default `UrllibTransport`; tests inject a fake transport (no network).
* Request body: `{"model", "messages", "tools"(omit if None), "max_tokens",
  "thinking": {"type": "disabled"}, "stream": False}`; when thinking enabled also
  `"reasoning_effort": "low|high|max"`; do **not** send `temperature` when
  thinking is enabled (it is ignored anyway — keep body clean).
* Retries: exponential backoff `min(retry_max_s, retry_base_s * 2**attempt)`
  with jitter, honoring `Retry-After` when present; retryable = 429/500/503,
  network timeouts/URLError, truncated bodies; non-retryable = 400/401/402/422
  (raise `ApiAuthenticationError`/`ApiBalanceError`/`ApiResponseError`).
  Max attempts = `max_retries + 1`.  `on_attempt` receives a redacted dict
  (attempt, status, error type, sleep_s, latency_s) for the audit log.
* Parse: role assistant `content` (may be null), `tool_calls[]` →
  `contracts.ToolCall(id, name, json.loads(arguments))`; invalid tool JSON →
  `ApiResponseError`; unknown finish_reason tolerated.  Return `ModelReply`
  with `ApiUsage.from_api(body["usage"])`.
* Never log/print the API key; exceptions must not embed `Authorization` headers.
* Body keep-alive: read full response then `json.loads(body.strip())`; non-JSON →
  `ApiResponseError`.
* `connectivity_probe()` helper for `scripts/llm_agent_api_probe.py`: text call,
  image call (build a small PNG with PIL showing a known code, ask the model to
  read it — validates real vision), tool-call call (dummy `get_timestamp` tool),
  and a bad-key auth-error check.  Probe script prints a concise pass/fail
  table, writes `artifacts/llm_agent_api_probe/report.json`, masks the key and
  never prints token values.  Exit codes: 0 pass, 2 missing key, 3 auth
  failure, 4 other failure.

`BudgetTracker`:
```python
class BudgetTracker:
    def __init__(self, budget_cfg: BudgetConfig, api_cfg: ApiConfig,
                 state_path: str | Path, *, total_path: str | Path | None = None,
                 clock: Callable[[], float] | None = None): ...
    def load(self) -> None                     # tolerate missing/corrupt; atomic save()
    def check_before_call(self, *, estimated_cost_usd: float) -> tuple[bool, str]
    def estimate_call_cost(self, *, text_chars: int, image_count: int = 1,
                           image_width: int = 640, image_height: int = 576) -> float  # worst case
    def record_success(self, usage: ApiUsage, *, model: str = "", latency_s: float = 0.0) -> float  # cost USD
    def record_failure(self, *, status: int | None = None, retryable: bool = True) -> None
    def state(self) -> BudgetState
    def snapshot(self) -> dict[str, Any]
    def save(self) -> None
    def hourly_cost(self) -> float
    def limits_exceeded(self) -> str | None    # reason string if a hard limit is hit
```
* Prices from `ApiConfig` (USD per 1M tokens); peak multiplier applies when the
  current UTC time is Mon-Fri and hour ∈ [01:00,04:00) or [06:00,10:00).
  Cost = (`cache_hit*hit_price + cache_miss*miss_price + completion*out_price`)/1e6
  × peak multiplier.  Unknown cache split → count all prompt tokens as misses.
* Pre-call check: estimate = price of `(text_chars/chars_per_token_estimate +
  image_tokens(image))` input + `max_tokens` output, padded; refuse if it would
  exceed `max_usd_per_run` (or `max_usd_total` via total file) or the call/hour
  limit; state the reason.  Single-threaded use only.
* Persist `audit/budget.json` in the run dir and a global
  `artifacts/llm_agent/budget_total.json`; atomic writes; corrupted files are
  backed up + reset (never crash the run).
* Tests must cover: retry/backoff (fake clock/transport, no real sleep >0.01 s
  via injectable `sleep`), auth errors not retried, 429 retried, budget refusal,
  peak/off-peak pricing math, persistence roundtrip, corrupt-state recovery,
  no key leakage in `str(exc)`/logs.

## 8. `memory.py` (owner: memory-guard)

```python
class AgentMemory:
    def __init__(self, cfg: MemoryConfig, *, run_dir: str | Path | None = None,
                 summarizer: Callable[[list[str]], str] | None = None): ...
    def load(self) -> None
    def save(self) -> None
    def set_goal(self, goal: str, reason: str = "") -> dict
    def record_discovery(self, text: str, *, kind: str = "other", place: str = "") -> dict
    def record_dialogue(self, speaker: str, text: str, *, place: str = "") -> dict
    def record_lesson(self, text: str, *, success: bool | None = None, context: str = "") -> dict
    def note_place(self, name: str, description: str = "", *, signature: str = "") -> dict
    def record_event(self, kind: str, text: str) -> None
    def get_goal(self) -> dict
    def search(self, query: str, limit: int = 10) -> list[dict]
    def render_context(self, query: str | None = None, *, max_chars: int | None = None) -> str
    def stats(self) -> dict            # {"places": n, "discoveries": n, "dialogues": n, "lessons": n, "goal": "..."}
    def maybe_compress(self, *, actions_since: int = 0, force: bool = False) -> bool
    def snapshot(self) -> dict
    def restore(self, snapshot: Mapping[str, Any]) -> None
    def save_report(self, path: str | Path) -> None
```
* Persistence in `MemoryConfig.root` (`llm_agent/memory/` by default):
  `memory.json` (atomic replace, versioned), `journal.jsonl` (append; trim to
  `max_journal`), `reports/memory_latest.md` (human-readable mirror when
  `mirror_to_run`), plus a `.gitignore` keeping runtime data out of git.
* Sections: `goal` (+ history), `places` (name, description, signature, first/last
  seen), `discoveries` (kind, text, place), `dialogues`, `lessons` (success/None),
  `open_questions`, `journal`/`events`.  Every entry gets `id`, `at` timestamp,
  `turn`/action context where supplied.
* `render_context` is the **only** thing sent to DeepSeek from memory: a compact
  Markdown block (goal + plan, known places, recent discoveries/dialogues,
  lessons, open questions) truncated to `max_chars`.  It must be deterministic
  and never include hidden state.  Retrieval = keyword scoring + recency, dedupe.
* Compression: cap each category (`max_*`), fold overflow into an
  `archived_summary`; `maybe_compress(force=True)` also calls the injected
  `summarizer` (optional, budget-gated by the Lead) and is tested with a fake
  summarizer.  Compression must never lose the goal or places.
* Corrupt/missing files: back up to `memory.corrupt-<ts>.json`, start fresh,
  keep the journal, record a recovery event.  Tests cover persistence roundtrip,
  retrieval, compression, snapshot/restore and corrupt recovery.

## 9. `guard.py` (owner: memory-guard)

```python
@dataclass
class GuardWarning:
    kind: str; severity: str; message: str; suggestion: str = ""

class LoopGuard:
    def __init__(self, cfg: GuardConfig): ...
    def on_action_outcome(self, outcome: ActionOutcome) -> None
    def on_observation(self, frame: Frame, event: ScreenEvent, *, changed: bool) -> None
    def on_tool_result(self, result: ToolResult) -> None
    def on_llm_call(self, *, ok: bool) -> None
    def check(self) -> list[GuardWarning]
    def warnings(self) -> list[GuardWarning]          # cached result of check()
    def needs_attention(self) -> bool                 # any warning or critical
    def critical(self) -> bool
    def render_warnings(self, max_chars: int = 700) -> str
    def reset_after_recovery(self) -> None
    def snapshot(self) -> dict; def restore(self, snapshot: Mapping[str, Any]) -> None
```
Detect (all from observations): unchanged movement batches (`unchanged_batch_limit`),
repeated identical batches (`repeated_batch_limit`), black frames
(`black_frame_limit`), many actions on the same screen signature
(`same_screen_action_limit`), no discovery recorded for N actions
(`no_discovery_action_limit` — needs the runtime to feed `on_tool_result`
record_discovery successes), consecutive tool errors (`max_consecutive_tool_errors`),
consecutive API failures (`max_consecutive_api_failures`).  Warnings have
severity `info|warning|critical` and actionable suggestions (“try a different
direction”, “open the menu”, “restore latest checkpoint”, …).  No RAM reads.

## 10. What the Lead wires together

* `prompts.py`: system prompt (blind-exploration rules, tool manual, cost
  discipline, memory discipline, self-correction) + user message builder
  (goal, memory context, recent history, guard warnings, screen analysis text,
  screenshot).
* `agent.py`: `DeepSeekBrain.decide(...)` — builds messages, calls
  `DeepSeekClient`, normalizes tool calls, records the decision summary, returns
  `ModelReply`.
* `runtime.py`: run loop, triggers, tool batch execution with interruption,
  guard feeding, checkpoint/resume, audit JSONL (`audit/decisions.jsonl`,
  `audit/actions.jsonl`, `audit/errors.jsonl`, `audit/api_attempts.jsonl`),
  budget enforcement, auto-recovery, safe stop/pause, single-instance lock,
  reports.
* `viewer.py`: Tk window (main thread) showing game frame + goal + last action +
  decision summary + steps + places + API calls/cost + status; Pause/Resume/Save/Quit
  controls; optional `cv2.VideoWriter` recording.
* `progress.py`: verification-only RAM probe (map id, coords, badges, party
  levels) used by tests/reports; must never be imported by the decision loop.
* `scripted.py`: deterministic fake brain for integration tests only
  (explicitly labelled; never counted as real-API evidence).
* `scripts/run_llm_agent.py`: CLI entry.
* `tests/llm_agent/conftest.py`: shared fixtures (`rom_path`, `state_path`,
  `require_rom`, `project_root`, `sample_rgba`, `fake_frame`, `tmp_memory_config`).

## 11. Definition of done per owner

1. Your modules import cleanly (`python -c "import llm_agent.<module>"`).
2. Your tests pass: `.venv\Scripts\python.exe -m pytest tests\llm_agent\test_<your>.py -q`
   and do not depend on network or an API key.
3. No edits outside your file scope; no new pip dependencies.
4. Docstrings + type hints; deterministic behavior; atomic file writes.
5. Message the Lead with: files delivered, test command/result counts, any
   contract concern, and any known limitation.  Claim your shared Team task
   before starting and complete it when done.
