"""Unified model router: one shared DeepSeek client pool, per-agent accounting.

Every deepseek request in the multi-agent run goes through :class:`ModelRouter`.
It provides:

* independent ``director.model`` / ``story.model`` / ``pokemon.model`` (and an
  unused-by-design ``navigation.model``) configuration — empty strings fall back
  to the global ``api.model``, so the existing DeepSeek configuration keeps
  working unchanged;
* one shared :class:`~llm_agent.budget.BudgetTracker`, so the global USD/call
  budget is never bypassed and a rollback can never reset real spend;
* per-agent counters (calls, prompt/completion tokens, estimated cost, latency,
  errors) persisted atomically to ``<artifact_root>/model_usage.json``;
* rate limiting for the (rare) director LLM call;
* a test-friendly ``client_factory`` so unit tests never touch the network.

The router never decides *what* to ask; it only sends the messages an agent
built and accounts for the response.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from llm_agent.api_client import DeepSeekClient
from llm_agent.budget import BudgetTracker
from llm_agent.config import AgentConfig
from llm_agent.contracts import (
    ApiError,
    ApiUsage,
    BudgetExceededError,
    ModelReply,
)
from llm_agent.multi_agent.contracts import (
    AgentRole,
    BudgetLimitError,
    MultiAgentError,
    now_ts,
)


@dataclass
class ModelRouterConfig:
    """Per-agent models and budgets (all optional; empty = global defaults)."""

    models: dict[str, str] = field(default_factory=dict)
    max_tokens_default: int = 400
    thinking: str = "disabled"
    image_scale: int = 3
    #: per-agent soft caps in USD (0 = only the global budget applies)
    max_usd: dict[str, float] = field(default_factory=dict)
    #: per-agent call caps (0 = only the global budget applies)
    max_calls: dict[str, int] = field(default_factory=dict)
    #: directory for the persistent usage ledger ("" = memory only)
    ledger_path: str = ""
    #: minimum seconds between two paid director-LLM calls (0 = no limit)
    director_min_interval_s: float = 90.0
    #: character budget for the pre-call worst-case estimate
    max_prompt_chars_estimate: int = 40000

    @classmethod
    def from_agent_config(cls, cfg: AgentConfig) -> "ModelRouterConfig":
        multi = getattr(cfg, "multi_agent", None)
        api = cfg.api
        models: dict[str, str] = {}
        if multi is not None:
            for role, value in (
                ("director", multi.director_model),
                ("story", multi.story_model),
                ("pokemon", multi.pokemon_model),
                ("navigation", multi.navigation_model),
            ):
                if str(value or "").strip():
                    models[role] = str(value).strip()
        max_usd: dict[str, float] = {}
        max_calls: dict[str, int] = {}
        if multi is not None:
            max_usd = {
                "director": float(multi.max_usd_director),
                "story": float(multi.max_usd_story),
                "pokemon": float(multi.max_usd_pokemon),
            }
            max_calls = {
                "director": int(multi.max_calls_director),
                "story": int(multi.max_calls_story),
                "pokemon": int(multi.max_calls_pokemon),
            }
        return cls(
            models=models,
            max_tokens_default=int(getattr(multi, "max_tokens_agent", 400) if multi else api.max_tokens),
            thinking=str(getattr(multi, "thinking_agent", api.thinking) if multi else api.thinking),
            image_scale=int(getattr(multi, "image_scale_agent", api.image_scale) if multi else api.image_scale),
            max_usd=max_usd,
            max_calls=max_calls,
            ledger_path="",
            director_min_interval_s=float(
                getattr(multi, "director_llm_min_interval_s", 90.0) if multi else 90.0
            ),
        )


def _empty_agent_usage() -> dict[str, Any]:
    return {
        "calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "latency_s": 0.0,
        "last_call_at": 0.0,
        "model": "",
        "by_role": {},
    }


class ModelRouter:
    """Shared DeepSeek access with per-agent isolation and accounting."""

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        config: ModelRouterConfig | None = None,
        budget: BudgetTracker | None = None,
        client_factory: Callable[[Any, AgentRole], Any] | None = None,
        clients: Mapping[str, Any] | None = None,
        clock: Callable[[], float] | None = None,
        on_attempt: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.config = config if config is not None else ModelRouterConfig.from_agent_config(cfg)
        self.budget = budget
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._client_factory = client_factory
        self._on_attempt = on_attempt
        self._clients: dict[str, Any] = dict(clients or {})
        self._director_last_call_at: float = 0.0
        self._usage: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self.last_error: str = ""
        self._load_ledger()

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    def model_for(self, role: AgentRole | str) -> str:
        role_enum = AgentRole.coerce(role) if not isinstance(role, AgentRole) else role
        key = role_enum.value
        if key == "system":
            key = "director"
        overridden = str(self.config.models.get(key) or "").strip()
        return overridden or str(self.cfg.api.model)

    def client_for(self, role: AgentRole | str) -> Any:
        model = self.model_for(role)
        with self._lock:
            client = self._clients.get(model)
            if client is not None:
                return client
            if self._client_factory is not None:
                client = self._client_factory(self.cfg, AgentRole.coerce(role))
            else:
                api_cfg = self.cfg.api
                if model != str(api_cfg.model):
                    api_cfg = replace(api_cfg, model=model)
                client = DeepSeekClient(api_cfg, on_attempt=self._on_attempt)
            self._clients[model] = client
            return client

    def set_client(self, role: AgentRole | str, client: Any) -> None:
        """Inject a client for tests (bypasses the real network)."""
        with self._lock:
            self._clients[self.model_for(role)] = client

    # ------------------------------------------------------------------ #
    # Budget checks
    # ------------------------------------------------------------------ #
    def agent_usage(self, role: AgentRole | str) -> dict[str, Any]:
        key = self._usage_key(role)
        with self._lock:
            return copy.deepcopy(self._usage.get(key) or _empty_agent_usage())

    def can_call(self, role: AgentRole | str) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` before building a request."""
        key = self._usage_key(role)
        usage = self._usage.get(key) or _empty_agent_usage()
        max_calls = int(self.config.max_calls.get(key, 0) or 0)
        if max_calls > 0 and int(usage.get("calls", 0)) >= max_calls:
            return False, f"agent {key} reached max_calls={max_calls}"
        max_usd = float(self.config.max_usd.get(key, 0.0) or 0.0)
        if max_usd > 0 and float(usage.get("cost_usd", 0.0)) >= max_usd:
            return False, f"agent {key} reached max_usd=${max_usd:.4f}"
        if self.budget is not None:
            reason = self.budget.limits_exceeded()
            if reason:
                return False, reason
        if key == "director" and self.config.director_min_interval_s > 0:
            elapsed = float(self._clock()) - float(self._director_last_call_at)
            if self._director_last_call_at > 0 and elapsed < self.config.director_min_interval_s:
                return False, (
                    f"director LLM cooldown {elapsed:.1f}s < {self.config.director_min_interval_s:.1f}s"
                )
        return True, ""

    def estimate_cost(
        self,
        *,
        text_chars: int,
        image_count: int = 0,
        image_size: tuple[int, int] | None = None,
    ) -> float:
        if self.budget is None:
            return 0.0
        width, height = image_size or (160 * self.config.image_scale, 144 * self.config.image_scale)
        return float(
            self.budget.estimate_call_cost(
                text_chars=max(0, int(text_chars)),
                image_count=max(0, int(image_count)),
                image_width=int(width),
                image_height=int(height),
            )
        )

    def check_before_call(
        self, role: AgentRole | str, *, estimated_cost_usd: float
    ) -> tuple[bool, str]:
        allowed, reason = self.can_call(role)
        if not allowed:
            return False, reason
        if self.budget is not None:
            return self.budget.check_before_call(estimated_cost_usd=max(0.0, float(estimated_cost_usd)))
        return True, ""

    # ------------------------------------------------------------------ #
    # The one request path
    # ------------------------------------------------------------------ #
    def complete(
        self,
        role: AgentRole | str,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int | None = None,
        thinking: str | None = None,
        image_count: int = 0,
        image_size: tuple[int, int] | None = None,
        purpose: str = "",
    ) -> ModelReply:
        """Send one chat completion for ``role`` and account for the response.

        Raises :class:`BudgetLimitError` when the per-agent or global budget is
        exhausted, and re-raises underlying :class:`ApiError` instances so the
        agent can report a structured failure.
        """
        role_enum = AgentRole.coerce(role) if not isinstance(role, AgentRole) else role
        key = self._usage_key(role_enum)
        text_chars = _messages_chars(messages)
        estimate = self.estimate_cost(
            text_chars=text_chars, image_count=image_count, image_size=image_size
        )
        allowed, reason = self.check_before_call(role_enum, estimated_cost_usd=estimate)
        if not allowed:
            raise BudgetLimitError(reason or "budget exhausted", details={"agent": key})

        client = self.client_for(role_enum)
        payload_max_tokens = int(max_tokens or self.config.max_tokens_default)
        payload_thinking = str(thinking if thinking is not None else self.config.thinking)
        started = self._clock()
        try:
            reply = client.chat(
                list(messages),
                tools=list(tools) if tools else None,
                max_tokens=payload_max_tokens,
                thinking=payload_thinking,
            )
        except BaseException as exc:
            self._record_failure(role_enum, exc)
            raise
        latency = float(getattr(reply, "latency_s", 0.0) or (self._clock() - started))
        cost = 0.0
        if self.budget is not None:
            try:
                cost = float(
                    self.budget.record_success(
                        getattr(reply, "usage", None) or ApiUsage(),
                        model=str(getattr(reply, "model", "") or ""),
                        latency_s=latency,
                    )
                )
            except Exception:  # noqa: BLE001 - accounting must never lose the reply
                cost = 0.0
        else:
            cost = self.estimate_cost(text_chars=text_chars, image_count=image_count, image_size=image_size)
        self._record_success(role_enum, reply, cost=cost, latency=latency)
        return reply

    def probe(self, role: AgentRole | str, text: str = "Reply with the single word OK.") -> ModelReply:
        """Tiny connectivity/warm-up probe used by the acceptance script."""
        return self.complete(role, [{"role": "user", "content": str(text)}], max_tokens=16)

    # ------------------------------------------------------------------ #
    # Accounting
    # ------------------------------------------------------------------ #
    def _usage_key(self, role: AgentRole | str) -> str:
        role_enum = AgentRole.coerce(role) if not isinstance(role, AgentRole) else role
        return "director" if role_enum in (AgentRole.SYSTEM, AgentRole.DIRECTOR) else role_enum.value

    def _record_success(self, role: AgentRole, reply: ModelReply, *, cost: float, latency: float) -> None:
        key = self._usage_key(role)
        usage = self._usage.setdefault(key, _empty_agent_usage())
        api_usage = getattr(reply, "usage", None) or ApiUsage()
        usage["calls"] = int(usage.get("calls", 0)) + 1
        usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0)) + int(api_usage.prompt_tokens)
        usage["completion_tokens"] = int(usage.get("completion_tokens", 0)) + int(api_usage.completion_tokens)
        usage["total_tokens"] = int(usage.get("total_tokens", 0)) + int(api_usage.total_tokens or 0)
        usage["cost_usd"] = round(float(usage.get("cost_usd", 0.0)) + float(cost), 8)
        usage["latency_s"] = round(float(usage.get("latency_s", 0.0)) + latency, 3)
        usage["last_call_at"] = float(self._clock())
        usage["model"] = str(getattr(reply, "model", "") or self.model_for(role))
        if key == "director":
            self._director_last_call_at = float(self._clock())
        self._save_ledger()

    def _record_failure(self, role: AgentRole, exc: BaseException) -> None:
        key = self._usage_key(role)
        usage = self._usage.setdefault(key, _empty_agent_usage())
        usage["failed_calls"] = int(usage.get("failed_calls", 0)) + 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self.budget is not None and isinstance(exc, ApiError):
            try:
                self.budget.record_failure(
                    status=getattr(exc, "status", None),
                    retryable=bool(getattr(exc, "retryable", True)),
                )
            except Exception:  # noqa: BLE001
                pass
        self._save_ledger()

    def usage(self) -> dict[str, Any]:
        """Aggregate + per-agent usage (never decreases)."""
        with self._lock:
            agents = {key: copy.deepcopy(value) for key, value in self._usage.items()}
        total = _empty_agent_usage()
        for value in agents.values():
            for field_name in (
                "calls",
                "failed_calls",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_s",
            ):
                total[field_name] = int(total.get(field_name, 0)) + int(value.get(field_name, 0))
            total["cost_usd"] = round(float(total.get("cost_usd", 0.0)) + float(value.get("cost_usd", 0.0)), 8)
            total["last_call_at"] = max(float(total.get("last_call_at", 0.0)), float(value.get("last_call_at", 0.0)))
        return {"total": total, "agents": agents}

    def costs_by_agent(self) -> dict[str, float]:
        return {key: float(value.get("cost_usd", 0.0)) for key, value in self.usage()["agents"].items()}

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": {
                "models": dict(self.config.models),
                "max_tokens_default": self.config.max_tokens_default,
                "thinking": self.config.thinking,
                "max_usd": dict(self.config.max_usd),
                "max_calls": dict(self.config.max_calls),
            },
            "usage": self.usage(),
            "saved_at": now_ts(),
        }

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        """Restore counters defensively: real spend can only ever grow."""
        if not snapshot:
            return
        incoming = (snapshot.get("usage") or {}).get("agents") or {}
        with self._lock:
            for key, value in incoming.items():
                if not isinstance(value, Mapping):
                    continue
                current = self._usage.setdefault(str(key), _empty_agent_usage())
                for field_name in ("calls", "failed_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                    current[field_name] = max(int(current.get(field_name, 0)), int(value.get(field_name, 0) or 0))
                current["cost_usd"] = max(
                    float(current.get("cost_usd", 0.0)), float(value.get("cost_usd", 0.0) or 0.0)
                )
                current["latency_s"] = max(
                    float(current.get("latency_s", 0.0)), float(value.get("latency_s", 0.0) or 0.0)
                )
        self._save_ledger()

    # -- persistence ------------------------------------------------------ #
    def _ledger_file(self) -> Path | None:
        path = str(self.config.ledger_path or "").strip()
        return Path(path) if path else None

    def _load_ledger(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = self._ledger_file()
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt ledger must not stop the run
            return
        agents = (data.get("usage") or {}).get("agents") or {}
        for key, value in agents.items():
            if isinstance(value, Mapping):
                merged = _empty_agent_usage()
                for field_name in ("calls", "failed_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                    merged[field_name] = int(value.get(field_name, 0) or 0)
                merged["cost_usd"] = float(value.get("cost_usd", 0.0) or 0.0)
                merged["latency_s"] = float(value.get("latency_s", 0.0) or 0.0)
                merged["last_call_at"] = float(value.get("last_call_at", 0.0) or 0.0)
                merged["model"] = str(value.get("model") or "")
                self._usage[str(key)] = merged
        self._director_last_call_at = max(
            (float(value.get("last_call_at", 0.0) or 0.0) for value in self._usage.values()), default=0.0
        )

    def _save_ledger(self) -> None:
        path = self._ledger_file()
        if path is None:
            return
        data = {"version": 1, "usage": self.usage(), "saved_at": now_ts()}
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


def _messages_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for message in messages or ():
        content = message.get("content")
        total += len(str(message.get("role") or ""))
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, (list, tuple)):
            for block in content:
                if isinstance(block, Mapping):
                    if block.get("type") == "text":
                        total += len(str(block.get("text") or ""))
                    else:
                        total += 32  # image/data-url overhead counted separately
        if message.get("tool_calls"):
            total += len(json.dumps(message["tool_calls"], ensure_ascii=False, default=str))
    return total


__all__ = ["ModelRouter", "ModelRouterConfig", "MultiAgentError", "BudgetExceededError"]
