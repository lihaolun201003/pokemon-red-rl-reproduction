# Navigation v2 — Internal Module Contract (frozen interface)

Status: **frozen for the second implementation round**.  If you believe an interface
change is genuinely required, message the Lead instead of editing another owner's
file.  Additive helpers in your own files are always fine.

Project root: `C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`
Python: `.venv\Scripts\python.exe` (3.10.11). Available: numpy, Pillow, OpenCV,
PyBoy, torch, tkinter.  **Do not install new packages.**  Do not use requests /
httpx / yaml / python-dotenv.  Windows/PowerShell: write files with the editor
tools, run tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\navigation\test_<module>.py -q
```

## 0. Product goal (do not lose sight of it)

The DeepSeek brain decides **what** to explore / where to go; the navigation
subsystem decides **how** to get there in Python, without an LLM call per step.
Everything is built **only from screenshots and verified actions** — no RAM,
no map ids, no external walkthrough/maps, no ROM data files.

Read `llm_agent/navigation/types.py` first: it is the frozen data contract
(enums, dataclasses, coordinate conventions).  Read `llm_agent/ARCHITECTURE.md`
for the existing agent architecture and `llm_agent/contracts.py` for shared
primitives (`Frame`, `ScreenEvent`, `ActionOutcome`, `ToolResult`, …).

Coordinate conventions (verified on real frames, see `types.py` docstring):
screen 160x144 = 10x9 tiles of 16 px; the camera keeps the player sprite fixed
at screen tile **VIEW_PLAYER=(4,4)**; one successful step shifts the background
by one tile; a step `up` decreases world y.  Area-local coordinates start at
(0,0) for the cell where the area was first entered.

## 1. File ownership (do not edit outside your scope)

| Owner | Files |
|---|---|
| **nav-lead** | `llm_agent/navigation/__init__.py`, `types.py`, `vision.py`, `localization.py`, `llm_agent/{runtime,tools,prompts,viewer,agent,config,contracts,emulator}.py` integration, `tests/navigation/conftest.py`, `tests/navigation/test_localization.py`, `tests/navigation/test_vision.py`, `tests/navigation/test_integration_nav.py`, `scripts/nav_v2_*.py`, `docs/reports/navigation_v2_*.md`, README |
| **nav-map** | `llm_agent/navigation/grid_map.py`, `llm_agent/navigation/world_model.py`, `tests/navigation/test_grid_map.py`, `tests/navigation/test_world_model.py` |
| **nav-planner** | `llm_agent/navigation/pathfinder.py`, `llm_agent/navigation/explorer.py`, `tests/navigation/test_pathfinder.py`, `tests/navigation/test_explorer.py` |
| **nav-control** | `llm_agent/navigation/controller.py`, `llm_agent/navigation/navigation_agent.py`, `tests/navigation/test_controller.py`, `tests/navigation/test_navigation_agent.py` |
| **nav-viz** | `llm_agent/navigation/viz.py`, `scripts/nav_v2_map_gallery.py`, `tests/navigation/test_viz.py` |

Never run long real-emulator experiments (the Lead runs those); your tests must be
fast, deterministic and must not need the ROM, network or an API key.  Real-ROM
tests may `pytest.skip` when `local/PokemonRed.gb` is absent, but prefer synthetic
frames.  Do not modify `neroRL/`, `reproduction/`, `configs/`, training code,
`local/`, or another owner's files.  Do not commit.

## 2. Shared pieces written by nav-lead

### `llm_agent/navigation/vision.py` (pure pixel helpers, no PyBoy)

```python
def frame_luma(frame: Frame | np.ndarray) -> np.ndarray            # (144,160) uint8
def tile_grid_offset(frame) -> tuple[int, int]                    # 16px lattice phase
def extract_tile_hashes(luma, offset) -> list[TileObs]            # 10x9 row-major
def tile_hash(tile: np.ndarray) -> str
def match_shift(prev_luma, cur_luma, sx_tiles: int, sy_tiles: int) -> float
def best_shift(prev_luma, cur_luma, *, max_tiles: int = 2) -> ShiftScore
   # ShiftScore(shift=(sx,sy) in tiles, score, second_shift, second_score, margin)
   # cur(x) == prev(x - shift*16), i.e. content moved by shift*16 pixels
def diff_mask(prev_luma, cur_luma) -> np.ndarray                  # bool HxW
def changed_tiles(prev_luma, cur_luma) -> list[tuple[int,int,float]]
def diff_stats(prev_luma, cur_luma) -> dict                       # bbox, counts, inside/outside player box
def screen_signature(luma) -> str
def is_black(luma) -> bool
```

Notes: `match_shift` compares overlapping pixels only; the score is "fraction of
equal pixels".  Real data: a successful step gives best shift `= -direction` with
score ~0.98–0.993; a blocked same-facing press is pixel identical (score 1.0);
a turn-only press changes only the player box (diff ratio < 0.01).

### `llm_agent/navigation/localization.py`

```python
@dataclass
class LocalizationConfig:
    move_min_score: float = 0.965       # confidence for MOVED
    blocked_min_score: float = 0.985    # score(0,0) needed for a confident BLOCKED
    ambiguous_margin: float = 0.004     # two shifts within margin -> UNKNOWN
    max_tiles: int = 2                  # candidate shift range
    transition_max_score: float = 0.75  # below this a huge change is a transition
    ...

class Localizer:
    def __init__(self, config: LocalizationConfig | None = None) -> None
    @property
    def pose(self) -> PlayerPose
    @property
    def last_view(self) -> FrameView | None
    def reset(self, *, area_id: str = "", cell: tuple[int, int] = (0, 0),
              facing: str | None = None) -> None
    def set_pose(self, x: int, y: int, *, facing: str | None = None,
                 confidence: float = 1.0) -> None
    def observe(self, frame: Frame, *, frame_context: ... = None) -> FrameView
    def update(self, prev: FrameView, cur: FrameView, direction: str,
               *, requested_tiles: int = 1) -> MotionEstimate
        # exactly one directional input between prev and cur; on MOVED it advances
        # self.pose by delta; on other results the pose is unchanged.  If the
        # result is UNKNOWN twice in a row the Localizer invalidates `pose.valid`
        # and the controller must call relocate().
```

## 3. `grid_map.py` (owner: nav-map)

One logical 2D grid per area.  Every cell records what the agent has actually
seen/verified; **no single failed bump may permanently mark an NPC cell as wall**.

```python
@dataclass
class Cell:
    x: int
    y: int
    state: CellState = CellState.UNKNOWN
    terrain: str = ""                     # dominant tile hash
    tile_hashes: list[str]                # observed tile looks (animation variants)
    observed_count: int = 0
    visit_count: int = 0
    confidence: float = 0.0
    last_seen_frame: int = -1
    obstacle_evidence: int = 0
    free_evidence: int = 0
    labels: list[str] = field(default_factory=list)
    def to_dict(self) -> dict; @classmethod from_dict(cls, data) -> "Cell"

@dataclass
class Edge:
    a: tuple[int, int]
    b: tuple[int, int]
    direction: str                        # direction a -> b
    state: EdgeState = EdgeState.UNKNOWN
    attempts: int = 0
    successes: int = 0
    confidence: float = 0.0
    def to_dict(...); from_dict(...)

@dataclass
class Frontier:
    cell: tuple[int, int]                 # non-walkable candidate cell to probe
    from_cell: tuple[int, int]            # reachable walkable neighbour
    direction: str
    expected_new: int

class GridMap:
    def __init__(self, area_id: str, *, frame_index: int = 0)
    area_id: str
    cells: dict[tuple[int, int], Cell]
    edges: dict[tuple[tuple[int, int], tuple[int, int]], Edge]
    created_frame: int
    kind: str                             # "interior"|"exterior"|"unknown"

    def cell(self, x, y) -> Cell | None
    def get_cell(self, x, y) -> Cell      # ensure + return (creates UNKNOWN)
    def cells_in_rect(...) -> list[Cell]
    def edge(self, a, b) -> Edge | None
    def get_edge(self, a, b) -> Edge
    def edge_state(self, a, b) -> EdgeState
    def neighbours(self, cell) -> list[tuple[int, int]]        # 4-connected

    def observe_view(self, view: FrameView, pose: PlayerPose, *,
                     frame_index: int = 0) -> dict[str, int]
      # For each visible tile (rel = tile.tx - VIEW_PLAYER[0], ...):
      #   ensure cell; record tile hash variant; observed_count += 1;
      #   last_seen_frame; state UNKNOWN -> OBSERVED (never downgrade WALKABLE,
      #   BLOCKED, TEMP_BLOCKED, EXIT, INTERACTIVE); returns counters
      #   {"new_cells": n, "new_hashes": n, "observed": n}
    def confirm_move(self, from_cell, direction: str, *, tiles: int = 1,
                     frame_index: int = 0) -> tuple[int, int]
      # verified movement: mark target cell(s) WALKABLE, visit_count += 1,
      # free_evidence += 1, state evidence upgrades, edges OPEN, returns new cell
    def confirm_blocked(self, from_cell, direction: str, *, temporary: bool = False,
                        frame_index: int = 0) -> tuple[int, int]
      # verified bump: attempts += 1, obstacle evidence; BLOCKED or TEMP_BLOCKED;
      # edge BLOCKED/TEMP_BLOCKED.  Contradiction handling: if the target cell was
      # WALKABLE before, keep it WALKABLE (temporary obstruction / stale read) and
      # mark the edge TEMP_BLOCKED instead of permanently blocking the cell.
    def confirm_turn(self, from_cell, direction: str) -> None
    def mark_exit(self, cell, *, direction: str | None = None, confidence: float = 1.0) -> None
    def mark_interactive(self, cell, label: str = "") -> None
    def mark_state(self, cell, state: CellState, *, confidence: float | None = None,
                   reason: str = "") -> None
    def mark_unreachable(self, cells: Iterable, reason: str = "") -> None
    def mark_deferred(self, cells: Iterable, reason: str = "") -> None

    def walkable_neighbours(self, cell) -> list[tuple[int, int]]
    def traversable_neighbours(self, cell, *, allow_observed: bool = True,
                               allow_temp_blocked: bool = False) -> list[...]
    def reachable_region(self, start, *, passable: set[CellState] | None = None) -> set[tuple]
    def frontier(self, start, *, reachable: set[tuple] | None = None) -> list[Frontier]
    def unknown_adjacent_count(self, cell) -> int

    def coverage(self, *, start: tuple[int, int] | None = None) -> CoverageStats
    def bounds(self) -> tuple[int, int, int, int]      # minx,miny,maxx,maxy
    def size(self) -> int                              # number of known cells
    def render_ascii(self, *, center=None, radius: int = 10,
                     path: list[tuple[int, int]] | None = None,
                     target: tuple[int, int] | None = None,
                     legend: bool = True) -> str
    def to_dict(self) -> dict; @classmethod from_dict(cls, data) -> "GridMap"
    def save(self, path) -> None; @classmethod load(cls, path) -> "GridMap"
```

Semantics that matter:

* `OBSERVED` ≠ passable.  Only `confirm_move` creates `WALKABLE`.
* `BLOCKED` needs repeated/believed-persistent evidence; a cell that was walked
  before is never turned into a permanent wall by one bump.  `TEMP_BLOCKED` for
  dynamic obstacles (NPCs) and is retried later.
* Coverage is honest: `coverage_ratio = walkable / (walkable + unknown_frontier)`
  where `unknown_frontier` counts non-walkable cells orthogonally adjacent to
  walkable cells (reachable ones).  Unreachable/deferred cells are reported
  separately, never as "explored".
* Persist as JSON: atomic write, versioned, tolerant load (corrupt file ->
  raise/ValueError; caller handles).  No absolute paths inside the data.

## 4. `world_model.py` (owner: nav-map)

```python
@dataclass
class AreaRecord:
    area_id: str
    kind: str = "unknown"                  # interior|exterior|unknown
    grid: GridMap
    visits: int = 0
    first_frame: int = 0
    last_frame: int = 0
    entry_cells: list[tuple[int, int]]
    explored: bool = False
    notes: list[str]
    def to_dict(); @classmethod from_dict(...)

@dataclass
class WorldConnection:
    from_area: str
    from_cell: tuple[int, int]
    direction: str                        # "down"|"up"|...|"warp"
    to_area: str
    to_pose: tuple[int, int] | None = None   # local pose in to_area if known
    kind: str = "warp"                    # warp|walk|unknown
    confidence: float = 0.5
    observations: int = 1
    def to_dict(); from_dict

class WorldModel:
    def __init__(self, *, root: str | Path | None = None)
    areas: dict[str, AreaRecord]
    connections: list[WorldConnection]
    current_area: str
    current_pose: tuple[int, int]

    def create_area(self, *, kind: str = "unknown", frame_index: int = 0,
                    area_id: str | None = None) -> AreaRecord
    def area(self, area_id: str | None = None) -> AreaRecord
    def set_current(self, area_id: str, pose: tuple[int, int]) -> None
    def record_visit(self, area_id: str, *, frame_index: int) -> None
    def add_connection(self, conn: WorldConnection) -> None
    def match_view(self, view: FrameView, *, candidate_area_ids: list[str] | None = None,
                   min_score: float = 0.55, max_results: int = 4) -> list[AreaMatch]
      # relocalization: for each candidate area, slide the 10x9 visible tile
      # hash window over the known cells and score matches / comparable tiles.
      # Must ignore the player's own tile (rel (0,0)) and tiles with no stored hash.
      # Return sorted candidates; ambiguous results (close scores) must keep both.
    def locate(self, view: FrameView, *, expected_area: str | None = None) -> AreaMatch | None
    def save(self) -> None                # world.json + per-area grid JSON, atomic
    def load(self) -> bool                # tolerant, versioned
    def summary(self) -> dict
    def render_summary(self) -> str       # compact markdown/text for prompts/viewer
    def unexplored_areas(self) -> list[str]
    def graph_dict(self) -> dict          # area nodes + connection edges (for viz)
```

Relocalization acceptance rule (important): accept a match only when
`score >= min_score` **and** `margin >= 0.08` over the runner-up, or when the
expected area/pose (from a just-recorded connection) agrees.  Otherwise the
controller must treat the location as unknown (do not merge areas wrongly).

Persistence layout under the run directory (nav-lead wires the root):
`navigation/world.json`, `navigation/areas/<area_id>.json`.  Keep files small;
no absolute paths.

## 5. `pathfinder.py` + `explorer.py` (owner: nav-planner)

```python
@dataclass
class PathCosts:
    walkable: float = 1.0
    observed: float = 3.0        # seen but not verified
    temp_blocked: float = 5.0
    unknown: float = 6.0         # conservative: exploring unknown costs more
    exit: float = 2.0
    blocked: float = float("inf")
    unreachable: float = float("inf")
    deferred: float = float("inf")

class Pathfinder:
    def __init__(self, costs: PathCosts | None = None, *, max_expansions: int = 60000)
    def plan(self, grid: GridMap, start: tuple[int, int], goal: tuple[int, int], *,
             allow_unknown: bool = True,
             allow_temp_blocked: bool = False) -> PathPlan
    def plan_to_nearest(self, grid, start, goals: Iterable[tuple[int, int]], *,
                        allow_unknown: bool = True) -> PathPlan | None
    def reachable(self, grid, start, *, allow_observed: bool = False) -> set[tuple[int, int]]
    def direction_between(self, a, b) -> str
    @staticmethod
    def plan_from_cells(cells: Sequence[tuple[int, int]], cost: float = 0.0) -> PathPlan
```

Requirements: deterministic, A* (Manhattan heuristic × walkable cost, admissible
for the given costs), never crosses BLOCKED/UNREACHABLE/DEFERRED cells unless the
caller allows unknown; `allow_temp_blocked` opt-in; cycles impossible; bounded
expansions; returns `reachable=False, reason=...` instead of raising.

```python
@dataclass
class ExplorerConfig:
    probe_unknown_cost: float = 6.0
    new_cell_weight: float = 1.6
    path_cost_weight: float = 0.35
    exit_bonus: float = 6.0
    repeat_penalty: float = 8.0
    max_attempts_per_cell: int = 2
    no_progress_limit: int = 30
    max_target_path: int = 80

class Explorer:
    def __init__(self, config: ExplorerConfig | None = None,
                 pathfinder: Pathfinder | None = None)
    def choose_target(self, grid: GridMap, start: tuple[int, int], *,
                      step_index: int = 0) -> ExploreTarget | None
    def should_finish(self, grid: GridMap, start: tuple[int, int]) -> tuple[bool, str, dict]
      # -> (finished, reason, stats): finished only when there is no reachable,
      # valuable frontier left.  Stats include counts of unreachable/deferred cells
      # and why they were skipped.
    def on_attempt(self, target: ExploreTarget, *, step_index: int,
                   outcome: str) -> None      # outcome: moved|blocked|unknown|event
    def frontier_candidates(self, grid, start) -> list[ExploreTarget]
    def reset(self) -> None
    def stats(self) -> dict
```

Exploration policy requirements: frontier-based; value = expected new cells,
penalised by path length, risk and repeated failed attempts; must not oscillate
between two targets forever; known paths may be reused to reach new frontiers;
record attempt history in memory only (not persisted).  It must be applicable when
the grid already has plenty of WALKABLE cells (re-exploration after re-entry).

## 6. `controller.py` + `navigation_agent.py` (owner: nav-control)

```python
@dataclass
class ControllerConfig:
    max_tiles_per_step: int = 1
    max_consecutive_unknown: int = 2
    max_consecutive_blocked: int = 4
    settle_frames: int = 0
    pause_on_event: bool = True
    step_timeout_s: float = 30.0

@dataclass
class ControllerResult:
    status: str                # ok|blocked|event|transition|localization_lost|step_limit
    steps: int = 0
    moves: int = 0
    blocked: int = 0
    unknown: int = 0
    replans: int = 0
    event: NavEvent | None = None
    steps_log: list[NavStep] = field(default_factory=list)

class NavigationController:
    def __init__(self, *, emulator, world: WorldModel, localizer: Localizer,
                 pathfinder: Pathfinder, explorer: Explorer,
                 config: ControllerConfig | None = None,
                 on_frame: Callable[[Frame], None] | None = None,
                 on_step: Callable[[NavStep], None] | None = None,
                 on_status: Callable[[dict], None] | None = None,
                 sleep: Callable[[float], None] | None = None)

    # low-level primitives (the only place that presses direction buttons)
    def observe(self) -> FrameView
    def move_once(self, direction: str, *, intent: str = "probe") -> MotionEstimate
        # press one direction via emulator, build a new FrameView, call
        # localizer.update(...), then EXACTLY ONCE update grid/world:
        #   MOVED -> grid.confirm_move + pose already advanced by localizer
        #   BLOCKED -> grid.confirm_blocked(temporary=False)
        #   TURNED_ONLY -> grid.confirm_turn + mark target as blocked evidence
        #   UNKNOWN -> no map update; count it
        #   INTERRUPTED -> return event, no map update
        #   TRANSITION -> do not touch grid; return transition estimate
    def relocate(self) -> AreaMatch | None
        # try world.match_view(current view); on unambiguous match: set pose/area;
        # else return None (caller decides).
    def handle_transition(self, *, max_wait_frames: int = 120) -> dict
        # wait until the screen stabilises (not black), then relocalize against
        # known areas; if matched: switch area, restore pose, record connection;
        # else: create a new area with pose (0,0) and record the pending connection.
    def follow_path(self, plan: PathPlan, *, max_steps: int = 100,
                    intent: str = "navigate") -> ControllerResult
        # executes directions one verified step at a time; on BLOCKED replans once
        # (the blocked edge is now known); on TRANSITION/event stops and returns.
    def explore_current_area(self, *, max_steps: int = 300, max_seconds: float = 600,
                             max_no_progress: int | None = None) -> ExploreReport
    def navigate_to(self, target: dict, *, max_steps: int = 200) -> NavReport
        # target forms: {"cell": [x,y]} | {"exit": true} | {"landmark": "..."} |
        # {"area": "<id>", "cell": [x,y]} | {"return_to_entry": true}
        # No LLM calls here; events/blocked paths are reported back structurally.
    def status(self) -> dict
```

```python
class NavigationAgent:
    """Persistent facade used by the DeepSeek tool layer and the runtime."""
    def __init__(self, *, emulator, world_root: str | Path | None = None,
                 config: AgentConfig | None = None,
                 on_frame=None, on_step=None, on_status=None)
    def attach(self) -> None                    # sync pose with a freshly loaded state
    def explore_current_area(self, *, max_steps=300, max_seconds=600) -> ExploreReport
    def navigate_to(self, target: dict, *, max_steps=200) -> NavReport
    def explore_and_report(self, ...)           # convenience used by tools
    def status(self) -> dict                    # area, pose, confidence, coverage, event
    def local_map(self, *, radius=12) -> dict   # {"ascii":..., "cells":[...], "coverage":...}
    def world_summary(self) -> dict
    def pause(self) -> None; def resume(self) -> None; @property paused: bool
    def save(self) -> None                      # world.save() to world_root
    def observe_external_outcome(self, outcome: ActionOutcome) -> None
        # called by ToolExecutor when raw press/walk tools are used directly by the
        # model: converts each ActionStep into a motion check using the SAME
        # exactly-once path (no double counting with move_once; guarded by an id).
```

Emulator surface available to the controller (already implemented, do not edit
`emulator.py`): `observe() -> Frame`, `press(button, frames=None) -> Frame`,
`wait(frames) -> Frame`, `frame_count: int`, `started: bool`.  The Lead may add
`press_snapshot(...)` later — use `getattr(emulator, "press_snapshot", None)`.

Controller/agent must NOT call the DeepSeek API and must NOT read RAM.  They must
be safe against infinite loops (budgets, no-progress limit, localization loss) and
must leave the emulator in a legal state (never mid-keypress).

`NavigationAgent.status()` field names consumed by the viewer/tools (keep stable):
`area_id`, `area_kind`, `pose:{x,y,facing,confidence,valid}`, `cells_total`,
`walkable`, `blocked`, `unknown_frontier`, `coverage_ratio`, `areas_total`,
`last_event`, `paused`, `world:{areas:[[id,kind,visits,walkable]], connections:n}`.

## 7. `viz.py` (owner: nav-viz)

Pure rendering helpers (numpy/PIL only, no Tk, no PyBoy):

```python
def render_map_image(grid: GridMap, *, pose: PlayerPose | None = None,
                     path: list[tuple[int, int]] | None = None,
                     target: tuple[int, int] | None = None,
                     cell_px: int = 10, max_cells: int = 64,
                     legend: bool = True) -> np.ndarray   # RGBA (h,w,4)
def render_world_image(world: WorldModel, *, width: int = 320, height: int = 160) -> np.ndarray
def render_status_panel(status: dict, *, width: int = 320, height: int = 160) -> np.ndarray
def composite_viewer(state: dict, *, frame_rgba: np.ndarray | None = None,
                     width: int = 480, height: int = 360) -> np.ndarray
def coverage_bar(coverage: CoverageStats, *, width: int = 120) -> np.ndarray
def describe_cell(cell) -> str
```

Colours must be a small fixed palette (walkable green, observed grey-blue,
blocked red, temp_blocked orange, exit yellow, interactive cyan, unknown dark,
player white, path magenta, target cyan ring).  Everything must be deterministic
and cheap enough for ~10 fps with a few hundred cells.  The Lead wires
`composite_viewer`/`render_map_image` into the Tk viewer and into
`scripts/nav_v2_map_gallery.py` uses it to save PNG evidence grids for the report.

## 8. Testing requirements per owner

1. Your module imports cleanly: `.venv\Scripts\python.exe -c "import llm_agent.navigation.<module>"`.
2. Your tests pass and are fast (<10 s) without ROM/network/API key.
3. Tests cover the boundary conditions listed for your module plus malformed
   persistence data where applicable (JSON roundtrip, corrupt file tolerance).
4. No edits outside your file scope.  No new pip dependencies.
5. Docstrings + type hints.  Deterministic behavior.  Atomic writes.
6. When done: claim/complete your shared Team task and message the Lead with
   files delivered, exact test command + counts, interface concerns and known
   limitations.  Do not claim success for untested behavior.

## 9. Synthetic fixtures (no emulator needed)

`tests/navigation/conftest.py` provides:

```python
def make_frame(rgba: np.ndarray, frame_index: int = 0) -> Frame
def synthetic_world(seed: int = 0) -> SyntheticRoom   # room grid + walls + tile images
def render_room(room, pose: tuple[int,int], facing="down") -> np.ndarray  # RGBA 160x144
def frame_view_from_parts(...) -> FrameView          # manual FrameView builder
```

The synthetic room renders 16x16 tiles from a deterministic pattern, so a
"move" is just re-rendering with the new camera offset.  This is enough to test
movement/relocalization end-to-end without PyBoy.
