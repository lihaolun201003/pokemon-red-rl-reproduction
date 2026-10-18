"""Configuration for the DeepSeek Pokémon Red agent.

All defaults are safe for a local supervised run.  Configuration is resolved in
this order (later wins):

1. Dataclass defaults (this file).
2. Optional JSON config file (``--config``) with sections matching
   :meth:`AgentConfig.to_dict`.
3. A local ``.env`` file (``DEEPSEEK_API_KEY`` and optional overrides).
4. Process environment variables.
5. Explicit CLI overrides as a flat ``{"section.field": value}`` mapping.

The API key itself is never stored in this configuration object and never
serialized; it is read from the environment at request time.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from llm_agent.dotenv import get_env, load_env_file, looks_like_placeholder, mask_secret

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROM = str(ROOT / "local" / "PokemonRed.gb")
DEFAULT_STATE = str(ROOT / "local" / "has_pokedex_nballs.state")
DEFAULT_RUN_ROOT = str(ROOT / "artifacts" / "llm_agent" / "runs")
DEFAULT_MEMORY_ROOT = str(ROOT / "llm_agent" / "memory")
DEFAULT_LOCK = str(ROOT / "artifacts" / "llm_agent" / "agent.lock")
DEFAULT_ENV_FILE = str(ROOT / ".env")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

@dataclass
class ApiConfig:
    """DeepSeek API connection, output shape and price model."""

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    #: request behavior
    timeout_s: float = 180.0
    connect_timeout_s: float = 15.0
    max_retries: int = 6
    retry_base_s: float = 2.0
    retry_max_s: float = 90.0
    #: generation
    temperature: float = 1.0  # ignored by the API in thinking mode
    max_tokens: int = 1200
    thinking: str = "disabled"  # "disabled" | "enabled"
    reasoning_effort: str = "low"  # only used when thinking == "enabled"
    #: image observation
    image_scale: int = 4
    image_format: str = "png"  # png | jpeg
    image_jpeg_quality: int = 90
    image_detail: str = "original"  # original | low | high | auto
    include_previous_screenshot: bool = False
    #: price model, USD per 1M tokens (DeepSeek Flash list prices, off-peak).
    #: Peak hours (01:00-04:00 and 06:00-10:00 UTC, Mon-Fri) are 2x.
    price_input_cache_hit_per_mtok: float = 0.003
    price_input_cache_miss_per_mtok: float = 0.15
    price_output_per_mtok: float = 0.60
    peak_multiplier: float = 2.0
    peak_hours_utc: tuple[tuple[int, int], ...] = ((1, 4), (6, 10))
    #: framing of the conversation (stateless, cost-bounded)
    system_prompt_chars: int = 40000
    max_history_entries_for_prompt: int = 10
    max_tool_result_chars: int = 700

    def price_dict(self) -> dict[str, float]:
        return {
            "cache_hit": self.price_input_cache_hit_per_mtok,
            "cache_miss": self.price_input_cache_miss_per_mtok,
            "output": self.price_output_per_mtok,
            "peak_multiplier": self.peak_multiplier,
        }


@dataclass
class BudgetConfig:
    """Hard budget limits enforced before every API request."""

    max_usd_per_run: float = 2.0
    max_usd_total: float = 10.0
    max_calls_per_run: int = 3000
    max_calls_per_hour: int = 240
    warn_at_fraction: float = 0.8
    #: never start a request if the worst-case estimate exceeds what is left
    min_remaining_usd: float = 0.02
    #: worst-case character->token ratio used for pre-call estimates
    chars_per_token_estimate: float = 2.0
    #: safety padding applied to the pre-call estimate (multiplier)
    estimate_padding: float = 1.15
    stop_on_auth_error: bool = True


@dataclass
class EmulatorConfig:
    """PyBoy emulator settings."""

    rom: str = DEFAULT_ROM
    state: str = DEFAULT_STATE
    window: str = "null"
    #: hold a button this many frames before releasing it (RL baseline used 8/24)
    press_frames: int = 8
    #: total frames advanced per button action (RL baseline used 24)
    action_frames: int = 24
    emulation_speed: int = 0  # 0 = as fast as possible
    render: bool = True
    max_actions: int = 0  # 0 = unlimited (rely on budget / user stop)
    #: frames to let a freshly loaded state settle before observation
    settle_frames: int = 2
    #: real-time presentation (viewer): stream every emulated frame at this fps
    presentation_fps: int = 60
    #: keep the game animating at presentation_fps while waiting for the LLM reply
    live_presentation_while_waiting: bool = True


@dataclass
class MemoryConfig:
    """Persistent long-term memory settings (``llm_agent/memory/``)."""

    root: str = DEFAULT_MEMORY_ROOT
    max_context_chars: int = 7000
    retrieval_limit: int = 12
    max_places: int = 300
    max_discoveries: int = 500
    max_dialogue: int = 400
    max_lessons: int = 400
    max_journal: int = 4000
    max_entry_chars: int = 1200
    compress_every_actions: int = 120
    #: keep a human-readable memory report in the run directory as well
    mirror_to_run: bool = True


@dataclass
class GuardConfig:
    """Loop / stuck / anomaly detection thresholds."""

    unchanged_batch_limit: int = 3       # movement batches with no screen change
    repeated_batch_limit: int = 3        # identical action batch repeated
    black_frame_limit: int = 2           # consecutive fully black screens
    no_discovery_action_limit: int = 80  # actions without recording anything new
    same_screen_action_limit: int = 24   # actions on the same screen signature
    max_consecutive_tool_errors: int = 6
    max_consecutive_api_failures: int = 6
    #: automatically restore the latest checkpoint after a hard anomaly
    auto_restore_on_lock: bool = False
    #: include a compact warning block in the next prompt whenever anomalies exist
    warn_on_anomaly: bool = True


@dataclass
class RuntimeConfig:
    """Run directory, checkpoints, audit and recovery behavior."""

    run_root: str = DEFAULT_RUN_ROOT
    run_dir: str = ""  # filled by resolve_paths()
    lock_path: str = DEFAULT_LOCK
    checkpoint_every_actions: int = 20
    checkpoint_keep: int = 20
    save_screens: bool = True
    screenshot_every_batches: int = 1
    screen_limit: int = 600
    max_turns: int = 0  # 0 = unlimited
    progress_log_every_turns: int = 10
    heartbeat_seconds: float = 180.0
    resume: bool = True
    #: optional previous run directory whose latest checkpoint should be resumed
    resume_from: str = ""
    pause_file: str = ""
    stop_file: str = ""
    auto_resume_attempts: int = 2
    audit_rotation_bytes: int = 20_000_000
    write_report: bool = True
    fail_fast_on_auth: bool = True
    #: include the previous action batch result in the next prompt
    keep_action_detail_chars: int = 500


@dataclass
class ViewerConfig:
    """Real-time Tk viewer settings."""

    enabled: bool = True
    title: str = "Pokémon Red — DeepSeek LLM Agent"
    scale: int = 4
    record_video: str = ""  # optional MP4 path
    video_fps: int = 10
    #: Tk repaint interval; 16 ms ≈ 60 fps for smooth real-time presentation
    refresh_ms: int = 16
    show_help: bool = True
    #: close the window automatically when the run ends (unattended mode)
    auto_close_on_finish: bool = False


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #

@dataclass
class AgentConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    emulator: EmulatorConfig = field(default_factory=EmulatorConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    config_path: str = ""
    env_file: str = DEFAULT_ENV_FILE
    #: optional human-readable label for this run
    label: str = ""

    # -- API key ---------------------------------------------------------- #
    def api_key(self) -> str | None:
        """Read the API key from the environment.  Never store it in config."""
        value = os.environ.get(self.api.api_key_env)
        if not value or looks_like_placeholder(value):
            return None
        return value

    def api_key_status(self) -> str:
        if self.api_key() is None:
            return f"{self.api.api_key_env} missing/placeholder"
        return f"{self.api.api_key_env} {mask_secret(self.api_key())}"

    # -- dict helpers ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_safe_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data["api_key"] = self.api_key_status()
        return data

    # -- paths ------------------------------------------------------------ #
    def run_path(self) -> Path:
        if not self.runtime.run_dir:
            raise ValueError("runtime.run_dir is not resolved; call resolve_paths()")
        return Path(self.runtime.run_dir)

    def resolve_paths(self, run_id: str | None = None) -> Path:
        """Create/fill run directory paths.  Returns the run directory."""
        if not self.runtime.run_dir:
            stamp = os.environ.get("LLM_AGENT_RUN_DIR", "").strip()
            if stamp:
                self.runtime.run_dir = str(Path(stamp).resolve())
            else:
                import datetime as _dt
                import uuid as _uuid

                now = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix = run_id or _uuid.uuid4().hex[:6]
                self.runtime.run_dir = str((Path(self.runtime.run_root) / f"run_{now}_{suffix}").resolve())
        run_dir = Path(self.runtime.run_dir)
        for sub in ("screens", "checkpoints", "audit", "errors"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        Path(self.memory.root).mkdir(parents=True, exist_ok=True)
        return run_dir

    # -- validation ------------------------------------------------------- #
    def validate(self) -> list[str]:
        problems: list[str] = []

        def check(condition: bool, message: str) -> None:
            if not condition:
                problems.append(message)

        check(self.api.base_url.startswith(("http://", "https://")), "api.base_url must be http(s)")
        check(bool(self.api.model.strip()), "api.model must not be empty")
        check(8 <= self.api.timeout_s <= 600, "api.timeout_s must be in 8..600")
        check(0 <= self.api.max_retries <= 10, "api.max_retries must be in 0..10")
        check(1.0 <= self.api.retry_base_s <= 30.0, "api.retry_base_s must be in 1..30")
        check(self.api.retry_base_s <= self.api.retry_max_s <= 600, "api.retry_max_s must be >= base and <= 600")
        check(64 <= self.api.max_tokens <= 8192, "api.max_tokens must be in 64..8192")
        check(self.api.thinking in ("disabled", "enabled"), "api.thinking must be disabled|enabled")
        check(self.api.reasoning_effort in ("low", "high", "max"), "api.reasoning_effort must be low|high|max")
        check(1 <= self.api.image_scale <= 8, "api.image_scale must be in 1..8")
        check(self.api.image_format in ("png", "jpeg"), "api.image_format must be png|jpeg")
        check(self.api.image_detail in ("original", "low", "high", "auto"), "api.image_detail invalid")
        check(self.api.price_input_cache_hit_per_mtok >= 0, "prices must be non-negative")
        check(self.api.price_input_cache_miss_per_mtok >= 0, "prices must be non-negative")
        check(self.api.price_output_per_mtok >= 0, "prices must be non-negative")

        check(self.budget.max_usd_per_run >= 0, "budget.max_usd_per_run must be >= 0")
        check(self.budget.max_usd_total >= 0, "budget.max_usd_total must be >= 0")
        check(self.budget.max_calls_per_run >= 0, "budget.max_calls_per_run must be >= 0")
        check(self.budget.max_calls_per_hour >= 0, "budget.max_calls_per_hour must be >= 0")
        check(0.0 <= self.budget.warn_at_fraction <= 1.0, "budget.warn_at_fraction must be 0..1")

        check(Path(self.emulator.rom).is_file(), f"ROM not found: {self.emulator.rom}")
        check(Path(self.emulator.state).is_file(), f"initial state not found: {self.emulator.state}")
        check(1 <= self.emulator.press_frames <= 60, "emulator.press_frames must be in 1..60")
        check(self.emulator.press_frames + 1 <= self.emulator.action_frames <= 120,
              "emulator.action_frames must be > press_frames and <= 120")
        check(self.emulator.window in ("null", "SDL2", "OpenGL"), "emulator.window invalid")
        check(self.emulator.max_actions >= 0, "emulator.max_actions must be >= 0")
        check(20 <= self.emulator.presentation_fps <= 120, "emulator.presentation_fps must be in 20..120")

        check(self.memory.max_context_chars >= 1500, "memory.max_context_chars must be >= 1500")
        check(self.memory.retrieval_limit >= 1, "memory.retrieval_limit must be >= 1")

        check(self.runtime.checkpoint_every_actions >= 1, "runtime.checkpoint_every_actions must be >= 1")
        check(self.runtime.checkpoint_keep >= 1, "runtime.checkpoint_keep must be >= 1")
        check(self.runtime.max_turns >= 0, "runtime.max_turns must be >= 0")
        check(self.runtime.heartbeat_seconds >= 10, "runtime.heartbeat_seconds must be >= 10")

        check(1 <= self.viewer.scale <= 8, "viewer.scale must be in 1..8")
        check(self.viewer.video_fps >= 1, "viewer.video_fps must be >= 1")
        check(8 <= self.viewer.refresh_ms <= 500, "viewer.refresh_ms must be in 8..500")
        return problems

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        *,
        env_file: str | Path | None = DEFAULT_ENV_FILE,
        overrides: Mapping[str, Any] | None = None,
    ) -> "AgentConfig":
        cfg = cls()
        cfg.env_file = str(env_file or "")
        if env_file:
            load_env_file(env_file)
        if config_path:
            path = Path(config_path)
            if not path.is_file():
                raise FileNotFoundError(f"Config file not found: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Config file must contain a JSON object")
            _merge_dict(cfg, data)
            cfg.config_path = str(path.resolve())
        _apply_env(cfg)
        if overrides:
            _apply_overrides(cfg, dict(overrides))
        return cfg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_ENV_MAP: dict[str, str] = {
    "DEEPSEEK_BASE_URL": "api.base_url",
    "DEEPSEEK_MODEL": "api.model",
    "DEEPSEEK_API_KEY_ENV": "api.api_key_env",
    "LLM_AGENT_LABEL": "label",
    "LLM_AGENT_ROM": "emulator.rom",
    "LLM_AGENT_STATE": "emulator.state",
    "LLM_AGENT_THINKING": "api.thinking",
    "LLM_AGENT_REASONING_EFFORT": "api.reasoning_effort",
    "LLM_AGENT_MAX_TOKENS": "api.max_tokens",
    "LLM_AGENT_TIMEOUT": "api.timeout_s",
    "LLM_AGENT_IMAGE_SCALE": "api.image_scale",
    "LLM_AGENT_IMAGE_FORMAT": "api.image_format",
    "LLM_AGENT_MAX_USD": "budget.max_usd_per_run",
    "LLM_AGENT_MAX_USD_TOTAL": "budget.max_usd_total",
    "LLM_AGENT_MAX_CALLS": "budget.max_calls_per_run",
    "LLM_AGENT_MAX_CALLS_PER_HOUR": "budget.max_calls_per_hour",
    "LLM_AGENT_MEMORY_ROOT": "memory.root",
    "LLM_AGENT_RUN_ROOT": "runtime.run_root",
    "LLM_AGENT_RUN_DIR": "runtime.run_dir",
    "LLM_AGENT_MAX_TURNS": "runtime.max_turns",
    "LLM_AGENT_CHECKPOINT_EVERY": "runtime.checkpoint_every_actions",
    "LLM_AGENT_HEARTBEAT": "runtime.heartbeat_seconds",
    "LLM_AGENT_PAUSE_FILE": "runtime.pause_file",
    "LLM_AGENT_STOP_FILE": "runtime.stop_file",
    "LLM_AGENT_VIDEO": "viewer.record_video",
    "LLM_AGENT_PRICE_CACHE_HIT": "api.price_input_cache_hit_per_mtok",
    "LLM_AGENT_PRICE_CACHE_MISS": "api.price_input_cache_miss_per_mtok",
    "LLM_AGENT_PRICE_OUTPUT": "api.price_output_per_mtok",
}

_BOOL_FIELDS = {
    "viewer.enabled",
    "runtime.resume",
    "runtime.save_screens",
    "runtime.write_report",
    "api.include_previous_screenshot",
}


def _coerce(current: Any, raw: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on", "enabled")
        return bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, str):
        return str(raw)
    if isinstance(current, (list, tuple)):
        if isinstance(raw, str):
            return tuple(part.strip() for part in raw.split(",") if part.strip())
        return type(current)(raw)
    return raw


def _resolve_attr_path(obj: Any, dotted: str) -> Any:
    node = obj
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    return node, parts[-1]


def _set_path(cfg: AgentConfig, dotted: str, raw: Any) -> None:
    parent, leaf = _resolve_attr_path(cfg, dotted)
    current = getattr(parent, leaf)
    setattr(parent, leaf, _coerce(current, raw))


def _apply_env(cfg: AgentConfig) -> None:
    for env_name, dotted in _ENV_MAP.items():
        value = get_env(env_name)
        if value is None:
            continue
        try:
            _set_path(cfg, dotted, value)
        except (AttributeError, TypeError, ValueError):
            continue


def _apply_overrides(cfg: AgentConfig, overrides: Mapping[str, Any]) -> None:
    for dotted, raw in overrides.items():
        if raw is None:
            continue
        _set_path(cfg, dotted, raw)


def _merge_dict(cfg: AgentConfig, data: Mapping[str, Any]) -> None:
    """Merge a JSON config dict section by section, validating field names."""
    for section_name, section_data in data.items():
        if section_name in ("config_path", "env_file", "api_key"):
            continue
        if not is_dataclass(getattr(cfg, section_name, None)):
            raise ValueError(f"Unknown config section: {section_name}")
        section = getattr(cfg, section_name)
        valid = {f.name for f in fields(section)}
        for key, value in dict(section_data or {}).items():
            if key not in valid:
                raise ValueError(f"Unknown config field: {section_name}.{key}")
            setattr(section, key, _coerce(getattr(section, key), value))


def default_config() -> AgentConfig:
    return AgentConfig()


def environment_summary() -> dict[str, str]:
    """Non-secret environment facts for audit logs."""
    return {
        "python": os.sys.version.split()[0],
        "platform": os.sys.platform,
        "cwd": os.getcwd(),
        "api_key": "set" if os.environ.get("DEEPSEEK_API_KEY") else "missing",
    }
