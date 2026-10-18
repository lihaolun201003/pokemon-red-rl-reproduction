"""Stdlib-only DeepSeek chat-completions client for the LLM game agent.

The client is the only place in :mod:`llm_agent` that talks to the network.  It
deliberately uses only the standard library (``urllib.request``) so the agent
runs inside the project's stock virtual environment.  The HTTP transport, the
clock and the sleep function are all injectable, so the unit tests never touch
the network and never sleep for real.

Contract (see ``llm_agent/ARCHITECTURE.md`` section 7)::

    DeepSeekClient(cfg, *, transport=None, on_attempt=None)
    client.chat(messages, *, tools=None, max_tokens=None, thinking=None,
                reasoning_effort=None, timeout_s=None) -> contracts.ModelReply
    client.probe_text(prompt) -> contracts.ModelReply
    DeepSeekClient.image_content_block(png_bytes, mime, *, detail="original")
    client.api_key_status() -> str

Safety properties enforced by ``tests/llm_agent/test_api_client.py``:

* the API key is read from the environment at request time and is never stored
  on the client, never serialized and never part of an exception message;
* ``on_attempt`` receives a *redacted* audit dict (attempt number, HTTP status,
  error class name, retryability, planned sleep, latency): no headers, no
  request/response bodies, no key;
* 401/402 and 400/422 are fatal and are not retried; 429/5xx/timeouts/truncated
  bodies are retried with exponential backoff plus jitter, honouring
  ``Retry-After`` when the server sends it;
* non-streaming response bodies may be prefixed by empty keep-alive lines: the
  whole body is decoded first, then stripped, then parsed as JSON.

Thinking mode is ON by default in the DeepSeek API.  This client always sends an
explicit top-level ``{"thinking": {"type": "disabled" | "enabled"}}`` field:
the game agent runs with thinking disabled unless the caller asks otherwise.
``temperature`` is never sent (it is ignored in thinking mode, and the request
body stays minimal/clean).
"""

from __future__ import annotations

import base64
import email.utils
import http.client
import json
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Protocol

from llm_agent import contracts
from llm_agent.config import ApiConfig
from llm_agent.dotenv import get_env, looks_like_placeholder, mask_secret

__all__ = [
    "DeepSeekClient",
    "Transport",
    "TransportTimeout",
    "UrllibTransport",
]

#: Keys allowed in the redacted attempt-audit dictionary.  Anything else (in
#: particular HTTP headers or request bodies) must never be emitted.
_ATTEMPT_KEYS: tuple[str, ...] = (
    "attempt",
    "max_attempts",
    "status",
    "error",
    "retryable",
    "sleep_s",
    "latency_s",
)

#: Maximum number of attempt records kept on the instance for debugging/audit.
_MAX_KEPT_ATTEMPTS = 512

#: Retryable HTTP statuses (429 is mapped to ApiRateLimitError separately).
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Fatal HTTP statuses, mapped to typed errors and never retried.
_FATAL_STATUS: frozenset[int] = frozenset({400, 401, 402, 403, 404, 405, 413, 422})

#: MIME types accepted by the vision content helper (DeepSeek accepts png/jpeg).
_IMAGE_MIMES: frozenset[str] = frozenset({"image/png", "image/jpeg"})

#: ``detail`` values accepted by the image_url content block.
_IMAGE_DETAILS: frozenset[str] = frozenset({"original", "low", "high", "auto"})

#: Thinking modes and reasoning efforts accepted by the API.
_THINKING_MODES: frozenset[str] = frozenset({"enabled", "disabled"})
_REASONING_EFFORTS: frozenset[str] = frozenset({"low", "high", "max"})

#: Upper bound for honouring ``Retry-After`` (protects a run from a rogue value).
_RETRY_AFTER_MAX_S = 600.0


# --------------------------------------------------------------------------- #
# Transport protocol and the default urllib implementation
# --------------------------------------------------------------------------- #

class TransportTimeout(TimeoutError):
    """A transport-level timeout / connection failure (always retryable).

    The concrete :class:`UrllibTransport` raises this for socket timeouts,
    ``urllib.error.URLError`` and truncated reads (``http.client`` exceptions),
    so the client can treat "network trouble" uniformly.
    """


class Transport(Protocol):
    """Minimal injectable HTTP transport used by :class:`DeepSeekClient`.

    Returns ``(status_code, headers, body_bytes)``.  Headers are compared
    case-insensitively by the client (the default implementation lower-cases
    them).  Raising :class:`TimeoutError` / :class:`OSError` /
    ``http.client.HTTPException`` signals a retryable network failure.
    """

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        ...


class UrllibTransport:
    """Default :class:`Transport` built on :mod:`urllib.request` (stdlib only).

    HTTP error statuses are *returned* (not raised) so the client can map them
    to typed errors itself; only network failures are raised.
    """

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = int(getattr(response, "status", None) or response.getcode() or 0)
                response_headers = {
                    str(key).lower(): str(value) for key, value in response.headers.items()
                }
                payload = response.read()
            return status, response_headers, payload
        except urllib.error.HTTPError as exc:  # a real HTTP response we can inspect
            try:
                payload = exc.read() or b""
            except Exception:  # pragma: no cover - defensive
                payload = b""
            try:
                response_headers = {str(k).lower(): str(v) for k, v in (exc.headers or {}).items()}
            except Exception:  # pragma: no cover - defensive
                response_headers = {}
            return int(exc.code), response_headers, payload
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise TransportTimeout(
                    f"request timed out after {float(timeout_s):g}s"
                ) from exc
            raise TransportTimeout(f"network error: {_short_text(reason)}") from exc
        except (TimeoutError, http.client.HTTPException, OSError) as exc:
            # socket.timeout, connection resets, IncompleteRead (truncated body)
            raise TransportTimeout(f"network error: {type(exc).__name__}: {_short_text(exc)}") from exc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _short_text(value: Any, limit: int = 240) -> str:
    """Collapse whitespace and truncate a value for safe error messages."""
    try:
        text = str(value)
    except Exception:  # pragma: no cover - defensive
        return type(value).__name__
    text = " ".join(text.split())
    return text[:limit]


def _body_snippet(raw: bytes, limit: int = 300) -> str:
    """Decode a response body for an error attribute (never contains the key)."""
    try:
        text = raw.decode("utf-8", errors="replace")
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        return ""
    return _short_text(text, limit)


def _server_error_message(raw: bytes) -> str:
    """Extract a short human-readable message from an API error body."""
    if not raw:
        return ""
    stripped = _body_snippet(raw, 4000).strip()
    if not stripped:
        return ""
    data: Any = None
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            for key in ("message", "msg", "type"):
                value = error.get(key)
                if isinstance(value, str) and value.strip():
                    return _short_text(value)
        if isinstance(error, str) and error.strip():
            return _short_text(error)
        for key in ("message", "msg", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return _short_text(value)
    return ""


def _headers_lower(headers: Mapping[str, Any] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _parse_retry_after(headers: Mapping[str, Any] | None) -> float | None:
    """Parse a ``Retry-After`` header (seconds or HTTP-date) into seconds."""
    lowered = _headers_lower(headers)
    raw = lowered.get("retry-after")
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        import datetime as _dt

        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return max(0.0, parsed.timestamp() - time.time())


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #

class DeepSeekClient:
    """OpenAI-compatible client for the DeepSeek chat-completions endpoint.

    Parameters
    ----------
    cfg:
        Frozen API configuration (base URL, model, retries, prices, image
        defaults).  The API key is *not* stored in this config; it is read from
        the environment variable named by ``cfg.api_key_env`` on every call.
    transport:
        Injectable :class:`Transport`; defaults to :class:`UrllibTransport`.
    on_attempt:
        Optional audit callback.  Receives a redacted dict for every attempt,
        including the successful one and the final failing one.  Exceptions
        raised by the callback are swallowed so auditing cannot break a run.
    sleep:
        Injectable sleep function (tests pass a recorder; default
        :func:`time.sleep`).
    clock:
        Injectable monotonic clock used for latency measurement / retry tests
        (default :func:`time.monotonic`).
    rng:
        Injectable random source for backoff jitter (default
        ``random.Random()``).
    """

    def __init__(
        self,
        cfg: ApiConfig,
        *,
        transport: Transport | None = None,
        on_attempt: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self._transport: Transport = transport if transport is not None else UrllibTransport()
        #: Redacted attempt-audit callback.  Public and mutable so the runtime
        #: can attach its audit sink after the client is constructed (the
        #: client is single-threaded; the current value is read on every attempt).
        self.on_attempt: Callable[[dict[str, Any]], None] | None = on_attempt
        self._sleep: Callable[[float], None] = sleep if sleep is not None else time.sleep
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._rng = rng if rng is not None else random.Random()
        self._endpoint = cfg.base_url.rstrip("/") + "/chat/completions"
        #: redacted attempt records (newest last, bounded) for debugging/audit
        self.attempts: list[dict[str, Any]] = []

    # -- introspection ---------------------------------------------------- #

    @property
    def endpoint(self) -> str:
        """Full chat-completions URL derived from ``cfg.base_url``."""
        return self._endpoint

    @property
    def model(self) -> str:
        """Model name used for requests."""
        return self.cfg.model

    def __repr__(self) -> str:
        # api_key_status() is always masked; the raw key never appears here.
        return (
            f"DeepSeekClient(model={self.cfg.model!r}, endpoint={self._endpoint!r}, "
            f"key={self.api_key_status()!r})"
        )

    # -- key handling ------------------------------------------------------ #

    def _resolve_api_key(self) -> str:
        """Read the API key from the environment; never store it on the client."""
        value = get_env(self.cfg.api_key_env)
        if not value or looks_like_placeholder(value):
            raise contracts.ApiAuthenticationError(
                f"{self.cfg.api_key_env} is missing or looks like a placeholder; "
                "set it in the environment or in the project .env file",
                status=None,
                retryable=False,
            )
        return value

    def api_key_status(self) -> str:
        """Masked key status (safe to print/log; never contains the key)."""
        value = get_env(self.cfg.api_key_env)
        if not value or looks_like_placeholder(value):
            return f"{self.cfg.api_key_env} missing/placeholder"
        return f"{self.cfg.api_key_env} {mask_secret(value)}"

    # -- request construction --------------------------------------------- #

    @staticmethod
    def image_content_block(png_bytes: bytes, mime: str, *, detail: str = "original") -> dict:
        """Build a vision content block from raw image bytes.

        Returns ``{"type": "image_url", "image_url": {"url": "data:...", "detail": ...}}``
        suitable for a **user** message ``content`` list (images in system or
        assistant messages are rejected by the API with HTTP 400).
        """
        if not isinstance(png_bytes, (bytes, bytearray)) or not png_bytes:
            raise ValueError("image_content_block requires non-empty image bytes")
        normalized_mime = str(mime or "").strip().lower()
        if normalized_mime == "image/jpg":
            normalized_mime = "image/jpeg"
        if normalized_mime not in _IMAGE_MIMES:
            raise ValueError(f"unsupported image MIME type: {mime!r} (expected image/png or image/jpeg)")
        raw_detail = "original" if detail is None else detail
        normalized_detail = str(raw_detail).strip().lower()
        if normalized_detail not in _IMAGE_DETAILS:
            raise ValueError(f"unsupported image detail: {detail!r} (expected one of {sorted(_IMAGE_DETAILS)})")
        encoded = base64.b64encode(bytes(png_bytes)).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{normalized_mime};base64,{encoded}",
                "detail": normalized_detail,
            },
        }

    def build_request_body(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Build the JSON request body without performing any I/O.

        This is public so tests and the probe script can assert the exact wire
        format (``thinking`` toggle, absence of ``temperature``, etc.).
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of message dicts")
        requested_max = self.cfg.max_tokens if max_tokens is None else int(max_tokens)
        if requested_max <= 0:
            raise ValueError(f"max_tokens must be positive, got {requested_max}")
        mode = str(self.cfg.thinking if thinking is None else thinking).strip().lower()
        if mode not in _THINKING_MODES:
            raise ValueError(f"thinking must be 'enabled' or 'disabled', got {mode!r}")
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": requested_max,
            "stream": False,
        }
        if tools:
            if not isinstance(tools, list):
                raise ValueError("tools must be a list of OpenAI-style tool schemas")
            body["tools"] = tools
        if mode == "enabled":
            body["thinking"] = {"type": "enabled"}
            effort = str(self.cfg.reasoning_effort if reasoning_effort is None else reasoning_effort).strip().lower()
            if effort not in _REASONING_EFFORTS:
                raise ValueError(f"reasoning_effort must be low|high|max, got {effort!r}")
            body["reasoning_effort"] = effort
        else:
            body["thinking"] = {"type": "disabled"}
        # NOTE: temperature is deliberately omitted (ignored in thinking mode
        # and the request body stays minimal/clean, per the architecture).
        return body

    def _request_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "pokemon-red-llm-agent/0.1 (+stdlib urllib)",
        }

    # -- retry / audit helpers -------------------------------------------- #

    def _emit_attempt(
        self,
        *,
        attempt: int,
        max_attempts: int,
        status: int | None,
        error: str | None,
        retryable: bool,
        sleep_s: float,
        latency_s: float,
    ) -> None:
        """Record and emit a redacted attempt dict (never contains the key)."""
        info = {
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "status": None if status is None else int(status),
            "error": error,
            "retryable": bool(retryable),
            "sleep_s": round(max(0.0, float(sleep_s)), 3),
            "latency_s": round(max(0.0, float(latency_s)), 3),
        }
        # Filter defensively: the public schema is fixed in _ATTEMPT_KEYS.
        info = {key: info[key] for key in _ATTEMPT_KEYS}
        self.attempts.append(info)
        if len(self.attempts) > _MAX_KEPT_ATTEMPTS:
            del self.attempts[: len(self.attempts) - _MAX_KEPT_ATTEMPTS]
        if self.on_attempt is None:
            return
        try:
            self.on_attempt(dict(info))
        except Exception:  # pragma: no cover - auditing must never break a run
            pass

    def _backoff_delay(self, retry_index: int, headers: Mapping[str, Any] | None = None) -> float:
        """Seconds to wait before retry number ``retry_index`` (0-based).

        ``min(retry_max_s, retry_base_s * 2**retry_index)`` scaled by an "equal
        jitter" factor in ``[0.5, 1.0]``.  A server-provided ``Retry-After`` is
        honoured exactly (clamped to ``[0, 600]`` s) and overrides the formula.
        """
        retry_after = _parse_retry_after(headers)
        if retry_after is not None:
            return max(0.0, min(float(retry_after), _RETRY_AFTER_MAX_S))
        base = max(0.0, float(self.cfg.retry_base_s))
        cap = max(base, float(self.cfg.retry_max_s))
        delay = min(cap, base * (2.0 ** max(0, int(retry_index))))
        jitter = 0.5 + 0.5 * float(self._rng.random())
        return max(0.0, delay * jitter)

    def _classify_http_error(
        self,
        status: int,
        headers: Mapping[str, Any] | None,
        raw: bytes,
    ) -> contracts.ApiError:
        """Map a non-2xx HTTP response to the typed error hierarchy."""
        detail = _server_error_message(raw)
        snippet = _body_snippet(raw)
        message = f"DeepSeek API HTTP {status}" + (f": {detail}" if detail else "")
        lowered = _headers_lower(headers)
        if status == 401:
            return contracts.ApiAuthenticationError(
                message + " (invalid API key; request not retried)",
                status=status,
                retryable=False,
                body=snippet,
            )
        if status == 402:
            return contracts.ApiBalanceError(
                message + " (insufficient balance; request not retried)",
                status=status,
                retryable=False,
                body=snippet,
            )
        if status == 429:
            retry_after = _parse_retry_after(headers)
            suffix = f" (retry after {retry_after:g}s)" if retry_after is not None else " (retryable)"
            return contracts.ApiRateLimitError(
                message + suffix,
                status=status,
                retryable=True,
                body=snippet,
            )
        if status in _FATAL_STATUS:
            return contracts.ApiResponseError(message, status=status, retryable=False, body=snippet)
        retryable = status in _RETRYABLE_STATUS or status >= 500
        return contracts.ApiResponseError(
            message + ("" if retryable else " (request not retried)"),
            status=status,
            retryable=retryable,
            body=snippet,
        )

    # -- response parsing --------------------------------------------------- #

    @staticmethod
    def _parse_tool_calls(raw_calls: Any) -> list[contracts.ToolCall]:
        """Parse ``message.tool_calls`` into validated :class:`contracts.ToolCall`."""
        if raw_calls is None:
            return []
        if not isinstance(raw_calls, list):
            raise contracts.ApiResponseError(
                "response 'tool_calls' is not a list", status=200, retryable=False
            )
        calls: list[contracts.ToolCall] = []
        for index, entry in enumerate(raw_calls):
            if not isinstance(entry, dict):
                raise contracts.ApiResponseError(
                    f"tool call #{index} is not an object", status=200, retryable=False
                )
            function = entry.get("function")
            if not isinstance(function, dict):
                raise contracts.ApiResponseError(
                    f"tool call #{index} has no 'function' object", status=200, retryable=False
                )
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise contracts.ApiResponseError(
                    f"tool call #{index} has no function name", status=200, retryable=False
                )
            raw_arguments = function.get("arguments")
            arguments: dict[str, Any]
            if raw_arguments is None:
                arguments = {}
            elif isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)
            elif isinstance(raw_arguments, str):
                if not raw_arguments.strip():
                    arguments = {}
                else:
                    try:
                        parsed = json.loads(raw_arguments)
                    except json.JSONDecodeError as exc:
                        raise contracts.ApiResponseError(
                            f"tool call #{index} has invalid JSON arguments: {exc.msg}",
                            status=200,
                            retryable=False,
                        ) from None
                    if not isinstance(parsed, dict):
                        raise contracts.ApiResponseError(
                            f"tool call #{index} arguments are not a JSON object",
                            status=200,
                            retryable=False,
                        )
                    arguments = parsed
            else:
                raise contracts.ApiResponseError(
                    f"tool call #{index} arguments have unsupported type "
                    f"{type(raw_arguments).__name__}",
                    status=200,
                    retryable=False,
                )
            call_id = entry.get("id")
            call_id = str(call_id) if call_id else f"call_{index}"
            calls.append(
                contracts.ToolCall(id=call_id, name=name.strip(), arguments=arguments)
            )
        return calls

    def _parse_response(self, raw: bytes, *, latency_s: float) -> contracts.ModelReply:
        """Parse a 2xx body into :class:`contracts.ModelReply`.

        Keep-alive blank lines before the JSON body are stripped.  Malformed or
        truncated bodies raise a *retryable* :class:`contracts.ApiResponseError`;
        semantically invalid tool-call JSON raises a non-retryable one.
        """
        text = raw.decode("utf-8", errors="replace") if raw else ""
        stripped = text.strip()
        if not stripped:
            raise contracts.ApiResponseError(
                "empty response body from DeepSeek (HTTP 200)", status=200, retryable=True
            )
        data: Any = None
        parse_error = True
        try:
            data = json.loads(stripped)
            parse_error = False
        except json.JSONDecodeError:
            # Some proxies prepend non-JSON keep-alive noise; retry from the
            # first '{' when that yields a valid object.
            start = stripped.find("{")
            if start > 0:
                try:
                    data = json.loads(stripped[start:])
                    parse_error = False
                except json.JSONDecodeError:
                    parse_error = True
        if parse_error:
            raise contracts.ApiResponseError(
                "malformed JSON response body from DeepSeek", status=200, retryable=True,
                body=stripped[:300],
            )
        if not isinstance(data, dict):
            raise contracts.ApiResponseError(
                "DeepSeek response JSON is not an object", status=200, retryable=True
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            # A 200 response without choices is a malformed exchange: retry once.
            raise contracts.ApiResponseError(
                "DeepSeek response contains no choices", status=200, retryable=True
            )
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if not isinstance(message, dict):
            message = {}
        raw_content = message.get("content")
        if raw_content is None:
            content = ""
        elif isinstance(raw_content, str):
            content = raw_content
        else:  # pragma: no cover - API always sends string/null
            content = str(raw_content)
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))
        raw_reasoning = message.get("reasoning_content")
        reasoning_content = raw_reasoning if isinstance(raw_reasoning, str) else None
        raw_usage = data.get("usage")
        usage = contracts.ApiUsage.from_api(raw_usage if isinstance(raw_usage, dict) else None)
        finish_reason = choice.get("finish_reason")
        response_id = data.get("id")
        model = data.get("model")
        return contracts.ModelReply(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=str(model) if model else self.cfg.model,
            latency_s=max(0.0, float(latency_s)),
            finish_reason=str(finish_reason) if finish_reason else "",
            reasoning_content=reasoning_content,
            response_id=str(response_id) if response_id else "",
            raw=data,
        )

    # -- public API --------------------------------------------------------- #

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
        timeout_s: float | None = None,
    ) -> contracts.ModelReply:
        """Run one chat-completion request and return the parsed reply.

        Retries transient failures up to ``cfg.max_retries`` extra attempts.
        Raises :class:`contracts.ApiAuthenticationError` (401 / missing key),
        :class:`contracts.ApiBalanceError` (402),
        :class:`contracts.ApiRateLimitError` (429 exhausted),
        :class:`contracts.ApiTimeoutError` (network/timeout exhausted) or
        :class:`contracts.ApiResponseError` (other HTTP/malformed response).
        """
        body = self.build_request_body(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
        api_key = self._resolve_api_key()  # raises before any network I/O
        headers = self._request_headers(api_key)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        timeout = float(timeout_s if timeout_s is not None else self.cfg.timeout_s)
        if timeout <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout}")
        max_attempts = max(1, int(self.cfg.max_retries) + 1)
        last_error: contracts.ApiError | None = None

        for attempt in range(1, max_attempts + 1):
            started = float(self._clock())
            try:
                status, response_headers, raw = self._transport.post(
                    self._endpoint, headers, payload, timeout
                )
            except (TimeoutError, OSError, http.client.HTTPException) as exc:
                latency_s = max(0.0, float(self._clock()) - started)
                error = contracts.ApiTimeoutError(
                    f"DeepSeek request failed: {type(exc).__name__}: {_short_text(exc)}",
                    status=None,
                    retryable=True,
                )
                last_error = error
                retry = attempt < max_attempts
                sleep_s = self._backoff_delay(attempt - 1) if retry else 0.0
                self._emit_attempt(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status=None,
                    error=type(error).__name__,
                    retryable=True,
                    sleep_s=sleep_s,
                    latency_s=latency_s,
                )
                if not retry:
                    raise error
                self._sleep(sleep_s)
                continue

            latency_s = max(0.0, float(self._clock()) - started)
            status = int(status)

            if 200 <= status < 300:
                try:
                    reply = self._parse_response(raw, latency_s=latency_s)
                except contracts.ApiResponseError as exc:
                    last_error = exc
                    retry = bool(exc.retryable) and attempt < max_attempts
                    sleep_s = self._backoff_delay(attempt - 1) if retry else 0.0
                    self._emit_attempt(
                        attempt=attempt,
                        max_attempts=max_attempts,
                        status=status,
                        error=type(exc).__name__,
                        retryable=bool(exc.retryable),
                        sleep_s=sleep_s,
                        latency_s=latency_s,
                    )
                    if not retry:
                        raise
                    self._sleep(sleep_s)
                    continue
                self._emit_attempt(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status=status,
                    error=None,
                    retryable=False,
                    sleep_s=0.0,
                    latency_s=latency_s,
                )
                return reply

            error = self._classify_http_error(status, response_headers, raw)
            last_error = error
            retry = bool(error.retryable) and attempt < max_attempts
            sleep_s = self._backoff_delay(attempt - 1, response_headers) if retry else 0.0
            self._emit_attempt(
                attempt=attempt,
                max_attempts=max_attempts,
                status=status,
                error=type(error).__name__,
                retryable=bool(error.retryable),
                sleep_s=sleep_s,
                latency_s=latency_s,
            )
            if not retry:
                raise error
            self._sleep(sleep_s)

        # Defensive: the loop above always raises/returns on the final attempt.
        if last_error is not None:  # pragma: no cover - unreachable in practice
            raise last_error
        raise contracts.ApiResponseError(  # pragma: no cover - unreachable
            "DeepSeek request failed without a recorded error", retryable=False
        )

    def probe_text(self, prompt: str) -> contracts.ModelReply:
        """Tiny one-message request used by ``scripts/llm_agent_api_probe.py``."""
        messages = [{"role": "user", "content": str(prompt)}]
        max_tokens = min(int(self.cfg.max_tokens), 64)
        return self.chat(messages, max_tokens=max_tokens)
