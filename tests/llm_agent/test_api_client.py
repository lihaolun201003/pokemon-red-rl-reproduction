"""Unit tests for :mod:`llm_agent.api_client`.

All tests are offline: a fake transport replaces the real HTTP layer, the sleep
function records durations instead of sleeping and the environment key is
monkeypatched.  No real API key and no network access are required.
"""

from __future__ import annotations

import io
import json
import random
import time
import urllib.error
import urllib.request
from email.message import Message
from typing import Any, Mapping

import pytest

from llm_agent import contracts
from llm_agent.api_client import (
    DeepSeekClient,
    TransportTimeout,
    UrllibTransport,
    _parse_retry_after,
)
from llm_agent.config import ApiConfig

TEST_KEY = "sk-test-abcdef0123456789abcdef0123456789"
CHAT_URL = "https://api.deepseek.com/chat/completions"
RETRYABLE_ATTEMPT_KEYS = {
    "attempt",
    "max_attempts",
    "status",
    "error",
    "retryable",
    "sleep_s",
    "latency_s",
}


# --------------------------------------------------------------------------- #
# Fake transport and response builders
# --------------------------------------------------------------------------- #

class FakeTransport:
    """Transport stub returning scripted responses; records every request."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": json.loads(body.decode("utf-8")),
                "raw_body": body,
                "timeout_s": timeout_s,
            }
        )
        if not self.responses:
            raise AssertionError("FakeTransport ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        status, response_headers, payload = item
        return int(status), dict(response_headers), bytes(payload)


def _reply_payload(
    *,
    content: str | None = "PONG",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    model: str = "deepseek-flash",
    response_id: str = "chatcmpl-test-1",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "id": response_id,
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage
        or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_cache_hit_tokens": 4,
            "prompt_cache_miss_tokens": 6,
        },
    }


def _http(
    status: int = 200,
    payload: Any = None,
    headers: Mapping[str, str] | None = None,
    *,
    raw: bytes | None = None,
) -> tuple[int, Mapping[str, str], bytes]:
    if raw is not None:
        body = raw
    elif isinstance(payload, bytes):
        body = payload
    else:
        body = json.dumps(payload if payload is not None else _reply_payload()).encode("utf-8")
    return status, dict(headers or {}), body


def _success() -> tuple[int, Mapping[str, str], bytes]:
    return _http(200)


class StepClock:
    """Deterministic monotonic clock: every read advances by ``step`` seconds."""

    def __init__(self, step: float = 0.01) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


@pytest.fixture
def client_factory(monkeypatch):
    """Build a client with a fake transport, a recorded sleep and test key."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", TEST_KEY)

    def _factory(
        responses: list[Any],
        *,
        cfg: ApiConfig | None = None,
        clock: Any = None,
        on_attempt: Any = None,
        sleeps: list[float] | None = None,
        rng: random.Random | None = None,
    ):
        api_cfg = cfg or ApiConfig(
            max_retries=2,
            retry_base_s=2.0,
            retry_max_s=30.0,
            timeout_s=10.0,
        )
        transport = FakeTransport(responses)
        recorded: list[float] = [] if sleeps is None else sleeps
        client = DeepSeekClient(
            api_cfg,
            transport=transport,
            on_attempt=on_attempt,
            sleep=recorded.append,
            clock=clock if clock is not None else time.monotonic,
            rng=random.Random(0) if rng is None else rng,
        )
        return client, transport, recorded

    return _factory


# --------------------------------------------------------------------------- #
# Happy path: parsing, request shape and usage
# --------------------------------------------------------------------------- #

def test_success_parsing_with_tool_calls_and_usage(client_factory) -> None:
    payload = _reply_payload(
        content="I will press A.",
        tool_calls=[
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "press_button", "arguments": '{"button": "a", "frames": 8}'},
            },
            {
                "id": "call_def",
                "type": "function",
                "function": {"name": "observe_game", "arguments": "{}"},
            },
        ],
        usage={
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
            "prompt_cache_hit_tokens": 100,
            "prompt_cache_miss_tokens": 23,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
        finish_reason="tool_calls",
        reasoning_content="thinking...",
    )
    client, transport, sleeps = client_factory([_http(200, payload)])

    reply = client.chat([{"role": "user", "content": "go"}])

    assert isinstance(reply, contracts.ModelReply)
    assert reply.content == "I will press A."
    assert reply.reasoning_content == "thinking..."
    assert reply.finish_reason == "tool_calls"
    assert reply.model == "deepseek-flash"
    assert reply.response_id == "chatcmpl-test-1"
    assert len(reply.tool_calls) == 2
    assert reply.tool_calls[0].id == "call_abc"
    assert reply.tool_calls[0].name == "press_button"
    assert reply.tool_calls[0].arguments == {"button": "a", "frames": 8}
    assert reply.tool_calls[1].name == "observe_game"
    assert reply.tool_calls[1].arguments == {}
    assert reply.usage.prompt_tokens == 123
    assert reply.usage.completion_tokens == 45
    assert reply.usage.total_tokens == 168
    assert reply.usage.cache_hit_tokens == 100
    assert reply.usage.cache_miss_tokens == 23
    assert reply.usage.reasoning_tokens == 7
    assert reply.latency_s >= 0.0
    assert sleeps == []

    assert len(transport.calls) == 1
    request = transport.calls[0]
    assert request["url"] == CHAT_URL
    assert request["headers"]["Authorization"] == f"Bearer {TEST_KEY}"
    assert request["timeout_s"] == 10.0
    body = request["body"]
    assert body["model"] == "deepseek-flash"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "go"}]
    assert body["max_tokens"] == 1200  # ApiConfig default
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    assert "temperature" not in body
    assert "tools" not in body


def test_latency_measured_with_injected_clock(client_factory) -> None:
    clock = StepClock(step=0.25)
    client, transport, _ = client_factory([_success()], clock=clock)
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.latency_s == pytest.approx(0.25)
    assert client.attempts[0]["latency_s"] == pytest.approx(0.25)


def test_keepalive_blank_lines_and_null_content_and_no_tool_calls(client_factory) -> None:
    body = b"\n\n\n" + json.dumps(_reply_payload(content=None)).encode("utf-8")
    client, transport, _ = client_factory([_http(raw=body)])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == ""
    assert reply.tool_calls == []
    assert reply.usage.prompt_tokens == 10


def test_custom_tools_thinking_and_max_tokens_in_request_body(client_factory) -> None:
    tools = [{"type": "function", "function": {"name": "get_timestamp", "parameters": {"type": "object"}}}]
    client, transport, _ = client_factory([_success()])
    client.chat(
        [{"role": "user", "content": "hi"}],
        tools=tools,
        max_tokens=64,
        thinking="enabled",
        reasoning_effort="high",
        timeout_s=3.5,
    )
    request = transport.calls[0]
    assert request["timeout_s"] == 3.5
    body = request["body"]
    assert body["tools"] == tools
    assert body["max_tokens"] == 64
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert "temperature" not in body


def test_unknown_finish_reason_is_tolerated(client_factory) -> None:
    client, _, _ = client_factory([_http(200, _reply_payload(finish_reason="something_new"))])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.finish_reason == "something_new"


def test_large_keepalive_noise_with_leading_garbage_is_parsed(client_factory) -> None:
    payload = _reply_payload(content="after noise")
    body = b"keep-alive: ok\n" + json.dumps(payload).encode("utf-8")
    client, _, _ = client_factory([_http(raw=body)])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "after noise"


# --------------------------------------------------------------------------- #
# Retries and backoff
# --------------------------------------------------------------------------- #

def test_429_then_success_honours_retry_after(client_factory) -> None:
    client, transport, sleeps = client_factory(
        [
            _http(429, {"error": {"message": "Rate limit reached"}}, {"retry-after": "3"}),
            _success(),
        ]
    )
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "PONG"
    assert len(transport.calls) == 2
    assert sleeps == [3.0]
    assert len(client.attempts) == 2
    first = client.attempts[0]
    assert set(first) == RETRYABLE_ATTEMPT_KEYS
    assert first["attempt"] == 1
    assert first["status"] == 429
    assert first["error"] == "ApiRateLimitError"
    assert first["retryable"] is True
    assert first["sleep_s"] == pytest.approx(3.0)
    assert client.attempts[1]["status"] == 200
    assert client.attempts[1]["error"] is None


def test_retry_after_http_date_is_honoured(client_factory) -> None:
    import email.utils

    retry_at = email.utils.formatdate(time.time() + 2.0, usegmt=True)
    client, transport, sleeps = client_factory(
        [_http(429, raw=b"", headers={"Retry-After": retry_at}), _success()]
    )
    client.chat([{"role": "user", "content": "hi"}])
    assert len(transport.calls) == 2
    assert len(sleeps) == 1
    assert 0.0 < sleeps[0] <= 3.0


def test_500_exhausted_raises_retryable_response_error(client_factory) -> None:
    cfg = ApiConfig(max_retries=2, retry_base_s=2.0, retry_max_s=5.0, timeout_s=10.0)
    client, transport, sleeps = client_factory(
        [_http(500, raw=b"internal error"), _http(503, raw=b"busy"), _http(500, raw=b"down")],
        cfg=cfg,
    )
    with pytest.raises(contracts.ApiResponseError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    error = excinfo.value
    assert error.retryable is True
    assert error.status == 500
    assert len(transport.calls) == 3
    assert len(sleeps) == 2
    assert all(0.0 < value <= 5.0 for value in sleeps)
    assert [item["attempt"] for item in client.attempts] == [1, 2, 3]
    assert all(item["retryable"] for item in client.attempts)
    assert client.attempts[-1]["sleep_s"] == 0.0


def test_backoff_is_exponential_and_capped(client_factory) -> None:
    cfg = ApiConfig(max_retries=3, retry_base_s=2.0, retry_max_s=3.0, timeout_s=10.0)
    responses = [_http(500, raw=b"e") for _ in range(4)]
    client, transport, sleeps = client_factory(responses, cfg=cfg)
    with pytest.raises(contracts.ApiResponseError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(transport.calls) == 4
    # delay = min(3, 2 * 2**index) with equal jitter in [0.5, 1.0] -> (1..2, 2..3 capped)
    delays = [min(3.0, 2.0 * 2.0 ** index) for index in range(3)]
    assert len(sleeps) == 3
    for sleep_s, delay in zip(sleeps, delays):
        assert 0.5 * delay - 1e-9 <= sleep_s <= delay + 1e-9


def test_timeout_retried_then_success(client_factory) -> None:
    client, transport, sleeps = client_factory([TimeoutError("boom"), _success()])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "PONG"
    assert len(transport.calls) == 2
    assert len(sleeps) == 1
    assert client.attempts[0]["status"] is None
    assert client.attempts[0]["error"] == "ApiTimeoutError"
    assert client.attempts[0]["retryable"] is True


def test_network_oserror_is_treated_as_retryable_timeout(client_factory) -> None:
    client, transport, sleeps = client_factory([ConnectionResetError("reset"), _success()])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "PONG"
    assert len(transport.calls) == 2
    assert len(sleeps) == 1


def test_timeout_exhausted_raises_api_timeout_error(client_factory) -> None:
    cfg = ApiConfig(max_retries=1, retry_base_s=1.0, retry_max_s=2.0, timeout_s=10.0)
    client, transport, sleeps = client_factory(
        [TimeoutError("one"), TimeoutError("two")], cfg=cfg
    )
    with pytest.raises(contracts.ApiTimeoutError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    assert excinfo.value.retryable is True
    assert len(transport.calls) == 2
    assert len(sleeps) == 1


# --------------------------------------------------------------------------- #
# Fatal errors are never retried
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        (401, contracts.ApiAuthenticationError),
        (402, contracts.ApiBalanceError),
        (400, contracts.ApiResponseError),
        (422, contracts.ApiResponseError),
    ],
)
def test_fatal_http_errors_are_not_retried(client_factory, status, error_class) -> None:
    cfg = ApiConfig(max_retries=5, retry_base_s=1.0, retry_max_s=2.0, timeout_s=10.0)
    payload = {"error": {"message": f"server says {status}"}}
    client, transport, sleeps = client_factory([_http(status, payload)], cfg=cfg)
    with pytest.raises(error_class) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    error = excinfo.value
    assert error.status == status
    assert error.retryable is False
    assert len(transport.calls) == 1
    assert sleeps == []
    assert len(client.attempts) == 1
    assert client.attempts[0]["sleep_s"] == 0.0
    assert f"server says {status}" in str(error)


def test_401_message_never_contains_the_api_key(client_factory, monkeypatch) -> None:
    key = "sk-unique-secret-key-0123456789"
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    client, transport, _ = client_factory([_http(401, {"error": {"message": "Authentication Fails"}})])
    attempts_seen: list[dict[str, Any]] = []
    client = DeepSeekClient(
        ApiConfig(max_retries=3),
        transport=transport,
        on_attempt=attempts_seen.append,
    )
    with pytest.raises(contracts.ApiAuthenticationError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    error = excinfo.value
    assert key not in str(error)
    assert key not in repr(error)
    assert key not in json.dumps(error.to_dict())
    assert key not in json.dumps(attempts_seen)
    assert key not in json.dumps(client.attempts)
    assert key not in repr(client)
    assert len(transport.calls) == 1
    # The Authorization header is only inside the transport call, never audited.
    for attempt in attempts_seen:
        assert "Authorization" not in attempt
        assert set(attempt) == RETRYABLE_ATTEMPT_KEYS


# --------------------------------------------------------------------------- #
# Malformed responses
# --------------------------------------------------------------------------- #

def test_malformed_json_is_retried_then_succeeds(client_factory) -> None:
    client, transport, sleeps = client_factory([_http(raw=b"not-json-at-all"), _success()])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "PONG"
    assert len(transport.calls) == 2
    assert len(sleeps) == 1
    assert client.attempts[0]["error"] == "ApiResponseError"
    assert client.attempts[0]["retryable"] is True


def test_malformed_json_exhausted(client_factory) -> None:
    cfg = ApiConfig(max_retries=1, retry_base_s=1.0, retry_max_s=2.0, timeout_s=10.0)
    client, transport, sleeps = client_factory([_http(raw=b"<html>oops"), _http(raw=b"<html>")], cfg=cfg)
    with pytest.raises(contracts.ApiResponseError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    assert excinfo.value.retryable is True
    assert len(transport.calls) == 2
    assert len(sleeps) == 1


def test_empty_response_body_is_retried(client_factory) -> None:
    client, transport, _ = client_factory([_http(raw=b""), _success()])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "PONG"
    assert len(transport.calls) == 2


def test_200_without_choices_is_retried(client_factory) -> None:
    client, transport, _ = client_factory(
        [_http(200, {"choices": []}), _success()]
    )
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "PONG"
    assert len(transport.calls) == 2


def test_json_array_body_is_rejected(client_factory) -> None:
    cfg = ApiConfig(max_retries=0, retry_base_s=1.0, retry_max_s=2.0, timeout_s=10.0)
    client, transport, _ = client_factory([_http(raw=b"[]")], cfg=cfg)
    with pytest.raises(contracts.ApiResponseError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(transport.calls) == 1


def test_invalid_tool_call_arguments_json_is_not_retried(client_factory) -> None:
    payload = _reply_payload(
        content=None,
        tool_calls=[
            {"id": "call_x", "type": "function", "function": {"name": "press_button", "arguments": "{not json"}}
        ],
        finish_reason="tool_calls",
    )
    client, transport, sleeps = client_factory([_http(200, payload)])
    with pytest.raises(contracts.ApiResponseError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    assert "invalid JSON arguments" in str(excinfo.value)
    assert excinfo.value.retryable is False
    assert len(transport.calls) == 1
    assert sleeps == []


def test_tool_call_without_function_name_is_rejected(client_factory) -> None:
    payload = _reply_payload(
        tool_calls=[{"id": "call_x", "type": "function", "function": {"arguments": "{}"}}],
        finish_reason="tool_calls",
    )
    client, _, _ = client_factory([_http(200, payload)])
    with pytest.raises(contracts.ApiResponseError):
        client.chat([{"role": "user", "content": "hi"}])


def test_tool_call_arguments_may_be_an_object(client_factory) -> None:
    payload = _reply_payload(
        tool_calls=[
            {"id": "call_obj", "type": "function", "function": {"name": "wait_frames", "arguments": {"frames": 4}}}
        ],
        finish_reason="tool_calls",
    )
    client, _, _ = client_factory([_http(200, payload)])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.tool_calls[0].arguments == {"frames": 4}


def test_tool_call_missing_id_gets_generated_id(client_factory) -> None:
    payload = _reply_payload(
        tool_calls=[{"type": "function", "function": {"name": "observe_game", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )
    client, _, _ = client_factory([_http(200, payload)])
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.tool_calls[0].id == "call_0"


# --------------------------------------------------------------------------- #
# Key handling / redaction
# --------------------------------------------------------------------------- #

def test_missing_key_raises_before_network(client_factory, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client, transport, _ = client_factory([_success()])
    with pytest.raises(contracts.ApiAuthenticationError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    assert "DEEPSEEK_API_KEY" in str(excinfo.value)
    assert transport.calls == []
    assert client.attempts == []


def test_placeholder_key_is_rejected_without_network(client_factory, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-xxxxxxxxxxxx")
    client, transport, _ = client_factory([_success()])
    with pytest.raises(contracts.ApiAuthenticationError):
        client.chat([{"role": "user", "content": "hi"}])
    assert transport.calls == []


def test_api_key_status_is_masked(client_factory) -> None:
    client, _, _ = client_factory([])
    status = client.api_key_status()
    assert "DEEPSEEK_API_KEY" in status
    assert TEST_KEY not in status
    assert "<set:" in status
    assert TEST_KEY[-2:] in status  # last two characters only
    assert TEST_KEY not in repr(client)


def test_api_key_status_without_key(client_factory, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client, _, _ = client_factory([])
    assert "missing/placeholder" in client.api_key_status()


def test_on_attempt_callback_errors_are_swallowed(client_factory) -> None:
    def broken_callback(_info: dict[str, Any]) -> None:
        raise RuntimeError("audit sink exploded")

    client, transport, _ = client_factory([_success()], on_attempt=broken_callback)
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "PONG"
    assert len(transport.calls) == 1


def test_on_attempt_receives_success_attempt(client_factory) -> None:
    events: list[dict[str, Any]] = []
    client, _, _ = client_factory([_success()], on_attempt=events.append)
    client.chat([{"role": "user", "content": "hi"}])
    assert len(events) == 1
    assert events[0]["attempt"] == 1
    assert events[0]["status"] == 200
    assert events[0]["error"] is None
    assert events[0]["retryable"] is False
    assert events[0]["sleep_s"] == 0.0
    assert set(events[0]) == RETRYABLE_ATTEMPT_KEYS


def test_public_cfg_and_mutable_on_attempt_attribute(client_factory) -> None:
    """The runtime attaches its audit sink with ``client.on_attempt = ...``."""
    events: list[dict[str, Any]] = []
    client, _, _ = client_factory([_success(), _success()])
    assert client.cfg.model == "deepseek-flash"  # public cfg (DeepSeekBrain.model)
    assert client.on_attempt is None
    client.on_attempt = events.append
    client.chat([{"role": "user", "content": "hi"}])
    assert len(events) == 1
    assert set(events[0]) == RETRYABLE_ATTEMPT_KEYS
    client.on_attempt = None  # detaching works too
    client.chat([{"role": "user", "content": "hi"}])
    assert len(events) == 1


# --------------------------------------------------------------------------- #
# Image content blocks and request validation
# --------------------------------------------------------------------------- #

def test_image_content_block_png() -> None:
    png = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    block = DeepSeekClient.image_content_block(png, "image/png")
    assert block["type"] == "image_url"
    image_url = block["image_url"]
    assert image_url["detail"] == "original"
    assert image_url["url"].startswith("data:image/png;base64,")
    import base64

    encoded = image_url["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == png


def test_image_content_block_jpeg_alias_and_detail() -> None:
    block = DeepSeekClient.image_content_block(b"jpegbytes", "image/jpg", detail="high")
    assert block["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert block["image_url"]["detail"] == "high"


@pytest.mark.parametrize(
    ("mime", "detail"),
    [
        ("image/gif", "original"),
        ("text/plain", "original"),
        ("image/png", "huge"),
        ("image/png", ""),
    ],
)
def test_image_content_block_rejects_invalid_values(mime: str, detail: str) -> None:
    with pytest.raises(ValueError):
        DeepSeekClient.image_content_block(b"bytes", mime, detail=detail)


def test_image_content_block_rejects_empty_bytes() -> None:
    with pytest.raises(ValueError):
        DeepSeekClient.image_content_block(b"", "image/png")


def test_build_request_body_validation() -> None:
    client = DeepSeekClient(ApiConfig())
    with pytest.raises(ValueError):
        client.build_request_body([])
    with pytest.raises(ValueError):
        client.build_request_body([{"role": "user", "content": "hi"}], max_tokens=0)
    with pytest.raises(ValueError):
        client.build_request_body([{"role": "user", "content": "hi"}], thinking="maybe")
    with pytest.raises(ValueError):
        client.build_request_body(
            [{"role": "user", "content": "hi"}], thinking="enabled", reasoning_effort="extreme"
        )


def test_probe_text_uses_small_request(client_factory) -> None:
    client, transport, _ = client_factory([_success()])
    reply = client.probe_text("hello")
    assert reply.content == "PONG"
    body = transport.calls[0]["body"]
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_tokens"] == 64


# --------------------------------------------------------------------------- #
# Default urllib transport mapping (still no network: urlopen is patched)
# --------------------------------------------------------------------------- #

def test_urllib_transport_maps_http_error_to_tuple(monkeypatch) -> None:
    message = Message()
    message["Content-Type"] = "application/json"

    def fake_urlopen(*args: Any, **kwargs: Any):
        raise urllib.error.HTTPError(
            CHAT_URL, 401, "Unauthorized", message, io.BytesIO(b'{"error": "nope"}')
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, headers, body = UrllibTransport().post(CHAT_URL, {"Authorization": "Bearer x"}, b"{}", 5.0)
    assert status == 401
    assert headers.get("content-type") == "application/json"
    assert body == b'{"error": "nope"}'


def test_urllib_transport_maps_url_error_to_transport_timeout(monkeypatch) -> None:
    def fake_urlopen(*args: Any, **kwargs: Any):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(TransportTimeout):
        UrllibTransport().post(CHAT_URL, {}, b"{}", 2.0)


def test_parse_retry_after_helpers() -> None:
    assert _parse_retry_after({"Retry-After": "7"}) == pytest.approx(7.0)
    assert _parse_retry_after({"retry-after": "0"}) == 0.0
    assert _parse_retry_after({}) is None
    assert _parse_retry_after({"Retry-After": "not-a-date"}) is None
