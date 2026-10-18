"""Real-API acceptance probe for the DeepSeek game agent.

This script is the *only* place in the project that intentionally talks to the
live DeepSeek API on purpose (besides the agent runtime).  It is meant to be run
by a human once a real API key is available, to prove that the transport, the
vision path, the tool-call path and the auth-error path all work with the real
service.  It never prints or stores the API key.

Checks
------
1. ``text_completion`` — a tiny chat request must answer ``PONG``.
2. ``vision`` — either a generated PNG containing a known 4-digit code (default;
   the model must read it back) or a user-supplied game screenshot (``--image``;
   the model must return a non-empty description).
3. ``tool_call`` — the model must call a dummy ``get_timestamp`` tool and the
   response must parse into a valid ``contracts.ToolCall``.
4. ``bad_key_auth`` — a deliberately invalid key must raise
   ``ApiAuthenticationError`` on the **first** attempt (401 is never retried).

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\llm_agent_api_probe.py
    .\\.venv\\Scripts\\python.exe scripts\\llm_agent_api_probe.py --image artifacts\\...\\screen.png
    .\\.venv\\Scripts\\python.exe scripts\\llm_agent_api_probe.py --config configs\\llm_agent.json --env-file .env

Exit codes: ``0`` pass, ``2`` missing/placeholder API key, ``3`` authentication
failure with the real key, ``4`` any other failure.

A masked JSON report is written to
``artifacts/llm_agent_api_probe/report.json`` (override with ``--outdir``).  The
report contains token counts and costs but never the API key or the
``Authorization`` header.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_agent import contracts  # noqa: E402
from llm_agent.api_client import DeepSeekClient  # noqa: E402
from llm_agent.budget import peak_multiplier_at, usage_cost_usd  # noqa: E402
from llm_agent.config import DEFAULT_ENV_FILE, AgentConfig  # noqa: E402

EXIT_OK = 0
EXIT_NO_KEY = 2
EXIT_AUTH_FAILURE = 3
EXIT_FAILURE = 4

DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "llm_agent_api_probe"
REPORT_NAME = "report.json"
GENERATED_IMAGE_NAME = "probe_code.png"
#: Deliberately invalid key used only to prove that 401 is not retried.
INVALID_KEY = "sk-probe-invalid-key-do-not-use-0000"
#: Reused for both the tool-call check and the request body.
DUMMY_TOOL_NAME = "get_timestamp"
DUMMY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DUMMY_TOOL_NAME,
        "description": (
            "Return the current Unix timestamp in seconds (UTC). "
            "Use this tool whenever the user asks for the current time."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _preview(text: Any, limit: int = 140) -> str:
    """Collapse whitespace and truncate text for console/report output."""
    try:
        value = " ".join(str(text).split())
    except Exception:  # pragma: no cover - defensive
        return type(text).__name__
    return value[:limit]


def _relative_or_name(path: Path) -> str:
    """Path relative to the project root when possible (keeps reports portable)."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except (ValueError, OSError):
        return path.name


def _read_image(path: Path) -> tuple[bytes, str]:
    """Read an image file and infer the MIME type from its suffix."""
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    mime_by_suffix = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = mime_by_suffix.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"unsupported image type {path.suffix!r} (use .png/.jpg/.jpeg)")
    data = path.read_bytes()
    if not data:
        raise ValueError(f"image file is empty: {path}")
    return data, mime


# --------------------------------------------------------------------------- #
# Generated vision image (PIL is available in the project venv)
# --------------------------------------------------------------------------- #

def _load_font(size: int) -> Any:
    """Load a scalable font, falling back to Pillow's bundled default font."""
    from PIL import ImageFont

    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        "arialbd.ttf",
        "arial.ttf",
        str(windows_fonts / "arialbd.ttf"),
        str(windows_fonts / "arial.ttf"),
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def make_probe_image(
    path: Path,
    code: str,
    *,
    width: int = 640,
    height: int = 576,
) -> dict[str, Any]:
    """Render a white PNG with a large black 4-digit ``code``; return metadata.

    The size defaults to a scale-4 game screen (640x576) so the probe exercises
    roughly the same image-token cost as a real observation.
    """
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    border = max(2, width // 160)
    margin = max(6, width // 24)
    draw.rectangle(
        (margin, margin, width - 1 - margin, height - 1 - margin),
        outline=(0, 0, 0),
        width=border,
    )

    # Shrink the font until the code fits comfortably inside the frame.
    font: Any = _load_font(16)
    bbox = (0, 0, 0, 0)
    size = int(height * 0.55)
    while size >= 12:
        font = _load_font(size)
        try:
            bbox = draw.textbbox((0, 0), code, font=font)
        except (ValueError, AttributeError):  # pragma: no cover - older Pillow
            bbox = (0, 0, size * len(code) // 2, size)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= int(width * 0.72) and text_height <= int(height * 0.55):
            break
        size -= 6
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    origin_x = (width - text_width) // 2 - bbox[0]
    origin_y = (height - text_height) // 2 - bbox[1]
    draw.text((origin_x, origin_y), code, fill=(0, 0, 0), font=font)
    image.save(path, format="PNG")

    import hashlib

    return {
        "path": _relative_or_name(path),
        "width": width,
        "height": height,
        "code": code,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# Check implementations (each returns a report dict; raises only on auth)
# --------------------------------------------------------------------------- #

def _usage_record(cfg: AgentConfig, usage: contracts.ApiUsage) -> dict[str, Any]:
    """Cost/usage section shared by every check."""
    multiplier = peak_multiplier_at(cfg.api)
    return {
        "usage": usage.to_dict(),
        "cost_usd": round(
            usage_cost_usd(cfg.api, usage, peak_multiplier=multiplier), 8
        ),
        "peak_multiplier": multiplier,
    }


def _check_text(cfg: AgentConfig, client: DeepSeekClient) -> dict[str, Any]:
    """Tiny text completion: the answer must contain ``PONG``."""
    started = time.monotonic()
    reply = client.probe_text("Reply with exactly the four letters PONG and nothing else.")
    latency_s = time.monotonic() - started
    content = (reply.content or "").strip()
    ok = "PONG" in content.upper()
    check: dict[str, Any] = {
        "name": "text_completion",
        "ok": ok,
        "latency_s": round(latency_s, 3),
        "detail": (
            "reply contained PONG"
            if ok
            else f"reply did not contain PONG: {_preview(content, 80)!r}"
        ),
        "reply_preview": _preview(content),
        "model": reply.model,
        "finish_reason": reply.finish_reason,
    }
    check.update(_usage_record(cfg, reply.usage))
    return check


def _check_vision(
    cfg: AgentConfig, client: DeepSeekClient, image_arg: str | None, outdir: Path
) -> dict[str, Any]:
    """Vision check: read a known generated code, or describe a supplied image."""
    started = time.monotonic()
    expected_code = ""
    image_meta: dict[str, Any]
    if image_arg:
        image_path = Path(image_arg)
        image_bytes, mime = _read_image(image_path)
        image_meta = {"mode": "provided", "path": _relative_or_name(image_path), "mime": mime}
        prompt = (
            "Describe in one short sentence what is shown in this image. "
            "Answer with text only."
        )
    else:
        expected_code = str(random.SystemRandom().randrange(1000, 10000))
        image_path = outdir / GENERATED_IMAGE_NAME
        image_meta = make_probe_image(image_path, expected_code)
        image_meta["mode"] = "generated"
        image_bytes = image_path.read_bytes()
        mime = "image/png"
        prompt = (
            "The attached image contains a 4-digit numeric code printed in large "
            "black digits on a white background. Reply with only the four digits, "
            "no other text."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                DeepSeekClient.image_content_block(
                    image_bytes, mime, detail=cfg.api.image_detail
                ),
            ],
        }
    ]
    reply = client.chat(messages, max_tokens=48)
    latency_s = time.monotonic() - started
    content = (reply.content or "").strip()
    digits = re.sub(r"\D", "", content)
    if expected_code:
        ok = bool(content) and expected_code in digits
        detail = (
            f"model read the expected 4-digit code ({expected_code})"
            if ok
            else f"expected code {expected_code}, reply digits {digits[:12]!r}"
        )
    else:
        ok = len(content) >= 3
        detail = (
            f"model described the supplied image: {_preview(content, 90)!r}"
            if ok
            else f"empty/short description: {_preview(content, 60)!r}"
        )
    check: dict[str, Any] = {
        "name": "vision",
        "ok": ok,
        "latency_s": round(latency_s, 3),
        "detail": detail,
        "image": image_meta,
        "expected_code": expected_code,
        "reply_preview": _preview(content),
        "model": reply.model,
        "finish_reason": reply.finish_reason,
    }
    check.update(_usage_record(cfg, reply.usage))
    return check


def _aggregate_usage(usages: list[contracts.ApiUsage]) -> contracts.ApiUsage:
    """Sum several usage records (the tool check may need a second attempt)."""
    total = contracts.ApiUsage()
    for usage in usages:
        total.prompt_tokens += max(0, int(usage.prompt_tokens or 0))
        total.completion_tokens += max(0, int(usage.completion_tokens or 0))
        total.total_tokens += max(0, int(usage.total_tokens or 0))
        total.cache_hit_tokens = (total.cache_hit_tokens or 0) + max(
            0, int(usage.cache_hit_tokens or 0)
        )
        total.cache_miss_tokens = (total.cache_miss_tokens or 0) + max(
            0, int(usage.cache_miss_tokens or 0)
        )
    return total


def _check_tool_call(cfg: AgentConfig, client: DeepSeekClient) -> dict[str, Any]:
    """Tool-call check: the model must call the dummy ``get_timestamp`` tool."""
    started = time.monotonic()
    usages: list[contracts.ApiUsage] = []
    last_reply: contracts.ModelReply | None = None
    tool_attempts = 0
    for _ in range(2):
        tool_attempts += 1
        messages = [
            {
                "role": "user",
                "content": (
                    "Call the get_timestamp tool exactly once to get the current "
                    "Unix timestamp. Do not answer with text; only make the tool call."
                ),
            }
        ]
        last_reply = client.chat(messages, tools=[DUMMY_TOOL_SCHEMA], max_tokens=128)
        usages.append(last_reply.usage)
        if last_reply.tool_calls:
            break
    latency_s = time.monotonic() - started
    calls = last_reply.tool_calls if last_reply is not None else []
    first = calls[0] if calls else None
    ok = (
        first is not None
        and first.name == DUMMY_TOOL_NAME
        and isinstance(first.arguments, dict)
        and bool(first.id)
    )
    if ok:
        detail = (
            f"model called {first.name}({json.dumps(first.arguments, sort_keys=True)}) "
            f"with id {first.id!r}"
        )
    elif first is not None:
        detail = f"unexpected tool call {first.name!r} / invalid arguments"
    else:
        preview = _preview(last_reply.content or "" if last_reply else "", 80)
        detail = f"no tool call returned (text reply: {preview!r})"
    check: dict[str, Any] = {
        "name": "tool_call",
        "ok": ok,
        "latency_s": round(latency_s, 3),
        "detail": detail,
        "attempts": tool_attempts,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments} for call in calls[:2]
        ],
        "model": last_reply.model if last_reply is not None else cfg.api.model,
        "finish_reason": last_reply.finish_reason if last_reply is not None else "",
    }
    check.update(_usage_record(cfg, _aggregate_usage(usages)))
    return check


def _check_bad_key(cfg: AgentConfig) -> dict[str, Any]:
    """Prove that a 401 raises ApiAuthenticationError on the first attempt."""
    env_name = cfg.api.api_key_env
    previous = os.environ.get(env_name)
    bad_cfg = dataclasses.replace(
        cfg.api,
        max_retries=min(2, max(0, int(cfg.api.max_retries))),
        timeout_s=min(30.0, float(cfg.api.timeout_s)),
    )
    attempts: list[dict[str, Any]] = []
    os.environ[env_name] = INVALID_KEY
    started = time.monotonic()
    try:
        client = DeepSeekClient(bad_cfg, on_attempt=attempts.append)
        try:
            client.chat([{"role": "user", "content": "ping"}], max_tokens=8)
        except contracts.ApiAuthenticationError:
            latency_s = time.monotonic() - started
            ok = len(attempts) == 1 and attempts[0].get("status") == 401
            detail = (
                "invalid key rejected with HTTP 401; "
                "ApiAuthenticationError raised without retrying"
                if ok
                else f"invalid key rejected but attempt behaviour was unexpected "
                f"({len(attempts)} attempt(s), statuses {[a.get('status') for a in attempts]})"
            )
            return {
                "name": "bad_key_auth",
                "ok": ok,
                "latency_s": round(latency_s, 3),
                "detail": detail,
                "attempts": attempts,
            }
        except contracts.ApiError as exc:
            latency_s = time.monotonic() - started
            return {
                "name": "bad_key_auth",
                "ok": False,
                "latency_s": round(latency_s, 3),
                "detail": (
                    f"expected ApiAuthenticationError, got {type(exc).__name__} "
                    f"(status={getattr(exc, 'status', None)})"
                ),
                "attempts": attempts,
            }
        latency_s = time.monotonic() - started
        return {
            "name": "bad_key_auth",
            "ok": False,
            "latency_s": round(latency_s, 3),
            "detail": "invalid API key unexpectedly succeeded; expected HTTP 401",
            "attempts": attempts,
        }
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


def _run_check(name: str, func: Any) -> dict[str, Any]:
    """Run one check; convert non-auth errors into a failed check record."""
    started = time.monotonic()
    try:
        return dict(func())
    except contracts.ApiAuthenticationError:
        raise  # fatal for the whole probe
    except contracts.ApiError as exc:
        return {
            "name": name,
            "ok": False,
            "latency_s": round(time.monotonic() - started, 3),
            "detail": f"{type(exc).__name__}: {_preview(exc)}",
            "error_status": getattr(exc, "status", None),
        }
    except Exception as exc:  # noqa: BLE001 - a probe must report, not crash
        return {
            "name": name,
            "ok": False,
            "latency_s": round(time.monotonic() - started, 3),
            "detail": f"{type(exc).__name__}: {_preview(exc)}",
        }


def connectivity_probe(
    cfg: AgentConfig,
    *,
    image_path: str | None = None,
    outdir: str | Path | None = None,
) -> dict[str, Any]:
    """Run all four live checks and return the raw (masked) results.

    This is the reusable helper behind the CLI ``main()``: it never prints and
    never writes the report.  Returns::

        {"checks": [...], "real_attempts": [...],
         "bad_key_attempts": [...], "auth_failure": <ApiError|None>}

    ``auth_failure`` is set (and the bad-key check skipped) when the real API
    key is rejected; individual check failures are recorded as ``ok: false``.
    """
    out = Path(outdir) if outdir is not None else DEFAULT_OUTDIR
    real_attempts: list[dict[str, Any]] = []
    client = DeepSeekClient(
        cfg.api, on_attempt=lambda info: real_attempts.append(dict(info))
    )
    checks: list[dict[str, Any]] = []
    auth_failure: contracts.ApiError | None = None
    for name, func in (
        ("text_completion", lambda: _check_text(cfg, client)),
        ("vision", lambda: _check_vision(cfg, client, image_path, out)),
        ("tool_call", lambda: _check_tool_call(cfg, client)),
    ):
        try:
            checks.append(_run_check(name, func))
        except contracts.ApiAuthenticationError as exc:
            auth_failure = exc
            break

    bad_key_attempts: list[dict[str, Any]] = []
    if auth_failure is None:
        try:
            bad_key_check = _check_bad_key(cfg)
        except Exception as exc:  # noqa: BLE001 - a probe must report, not crash
            bad_key_check = {
                "name": "bad_key_auth",
                "ok": False,
                "latency_s": 0.0,
                "detail": f"unexpected {type(exc).__name__}: {_preview(exc)}",
            }
        bad_key_attempts = list(bad_key_check.get("attempts") or [])
        bad_key_check.pop("attempts", None)
        checks.append(bad_key_check)
    return {
        "checks": checks,
        "real_attempts": real_attempts,
        "bad_key_attempts": bad_key_attempts,
        "auth_failure": auth_failure,
    }


# --------------------------------------------------------------------------- #
# Reporting / console output
# --------------------------------------------------------------------------- #

def _sanitize_attempts(attempts: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    """Attach the probe phase to already-redacted attempt dicts."""
    return [
        {
            "phase": phase,
            "attempt": item.get("attempt"),
            "status": item.get("status"),
            "error": item.get("error"),
            "retryable": item.get("retryable"),
            "sleep_s": item.get("sleep_s"),
            "latency_s": item.get("latency_s"),
        }
        for item in attempts
    ]


def _write_report(outdir: Path, payload: dict[str, Any]) -> Path:
    """Atomically write the masked JSON report; returns its path."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / REPORT_NAME
    data = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False).encode("utf-8")
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    return path


def _print_header(cfg: AgentConfig) -> None:
    print("DeepSeek API probe (real network calls, key is never printed)")
    print(f"  model     : {cfg.api.model}")
    print(f"  endpoint  : {cfg.api.base_url.rstrip('/')}/chat/completions")
    print(f"  key       : {cfg.api_key_status()}")
    print(
        f"  thinking  : {cfg.api.thinking}"
        + (f" (effort={cfg.api.reasoning_effort})" if cfg.api.thinking == "enabled" else "")
    )
    print(
        f"  images    : detail={cfg.api.image_detail}, "
        f"timeout={cfg.api.timeout_s:g}s, max_retries={cfg.api.max_retries}"
    )


def _print_checks(checks: list[dict[str, Any]]) -> None:
    print()
    print(f"  {'check':<18}{'result':<8}{'latency':>9}  detail")
    print(f"  {'-' * 18}{'-' * 8}{'-' * 9}  {'-' * 46}")
    for check in checks:
        name = str(check.get("name", "?"))
        result = "PASS" if check.get("ok") else "FAIL"
        try:
            latency = float(check.get("latency_s") or 0.0)
        except (TypeError, ValueError):
            latency = 0.0
        detail = _preview(check.get("detail", ""), 110)
        print(f"  {name:<18}{result:<8}{latency:>8.2f}s  {detail}")
    print()


def _print_missing_key_help(cfg: AgentConfig, report_path: Path) -> None:
    print("DeepSeek API key not found - no request was made.")
    print(f"  env var : {cfg.api.api_key_env}")
    print(f"  status  : {cfg.api_key_status()}")
    print(f"  .env    : {cfg.env_file or '(not set)'}")
    print("")
    print("Set the key, then re-run this probe:")
    print('  PowerShell :  $env:DEEPSEEK_API_KEY = "sk-..."')
    print("  or create .env in the project root (copy .env.example):")
    print("               DEEPSEEK_API_KEY=sk-...")
    print("Get a key at https://platform.deepseek.com/api_keys")
    print("")
    print(f"Report: {report_path}")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the real DeepSeek API for the Pokemon Red LLM agent "
            "(text, vision, tool call and invalid-key auth checks)."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="optional JSON config file passed to AgentConfig.load()",
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help=f"optional .env file (default: {DEFAULT_ENV_FILE})",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "optional PNG/JPEG image for the vision check (e.g. a real game "
            "screenshot); when omitted a PNG with a known 4-digit code is generated"
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=f"report/output directory (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="override the per-request timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="override api.max_retries for the probe run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Keep console output safe on legacy Windows code pages: model replies may
    # contain Unicode.  The JSON report is always written as UTF-8 regardless.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # pragma: no cover - unusual streams
                pass
    started_at = _iso_now()
    started_monotonic = time.monotonic()

    try:
        cfg = AgentConfig.load(args.config, env_file=args.env_file)
    except (OSError, ValueError) as exc:
        print(f"error: failed to load configuration: {_preview(exc)}")
        return EXIT_FAILURE
    overrides: dict[str, Any] = {}
    if args.timeout is not None:
        overrides["timeout_s"] = float(args.timeout)
    if args.max_retries is not None:
        overrides["max_retries"] = int(args.max_retries)
    if overrides:
        cfg.api = dataclasses.replace(cfg.api, **overrides)

    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    report_path = outdir / REPORT_NAME
    checks: list[dict[str, Any]] = []
    real_attempts: list[dict[str, Any]] = []
    bad_key_attempts: list[dict[str, Any]] = []
    auth_failure: contracts.ApiError | None = None
    unexpected_failure: Exception | None = None
    exit_code = EXIT_FAILURE

    try:
        if cfg.api_key() is None:
            exit_code = EXIT_NO_KEY
        else:
            _print_header(cfg)
            probe_result = connectivity_probe(
                cfg, image_path=args.image, outdir=outdir
            )
            checks = probe_result["checks"]
            real_attempts = probe_result["real_attempts"]
            bad_key_attempts = probe_result["bad_key_attempts"]
            auth_failure = probe_result["auth_failure"]
            if auth_failure is not None:
                exit_code = EXIT_AUTH_FAILURE
            elif checks and all(bool(check.get("ok")) for check in checks):
                exit_code = EXIT_OK
            else:
                exit_code = EXIT_FAILURE
    except contracts.ApiAuthenticationError as exc:  # defensive
        auth_failure = exc
        exit_code = EXIT_AUTH_FAILURE
    except Exception as exc:  # noqa: BLE001 - a probe must report, not crash
        unexpected_failure = exc
        exit_code = EXIT_FAILURE

    failed_checks = [check for check in checks if not check.get("ok")]
    totals = {
        "checks": len(checks),
        "checks_passed": len(checks) - len(failed_checks),
        "api_attempts": len(real_attempts) + len(bad_key_attempts),
        "retried_attempts": sum(
            1
            for item in real_attempts
            if isinstance(item.get("attempt"), int) and item["attempt"] > 1
        ),
    }
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost_total = 0.0
    for check in checks:
        usage = check.get("usage") or {}
        if isinstance(usage, dict):
            for key in token_totals:
                try:
                    token_totals[key] += int(usage.get(key) or 0)
                except (TypeError, ValueError):
                    pass
        try:
            cost_total += float(check.get("cost_usd") or 0.0)
        except (TypeError, ValueError):
            pass
    totals.update(token_totals)
    totals["estimated_cost_usd"] = round(cost_total, 8)

    failure_payload: dict[str, Any] | None = None
    if auth_failure is not None:
        failure_payload = {
            "type": type(auth_failure).__name__,
            "message": _preview(auth_failure, 300),
            "status": getattr(auth_failure, "status", None),
        }
    elif unexpected_failure is not None:
        failure_payload = {
            "type": type(unexpected_failure).__name__,
            "message": _preview(unexpected_failure, 300),
        }

    report: dict[str, Any] = {
        "schema": 1,
        "tool": "scripts/llm_agent_api_probe.py",
        "started_at": started_at,
        "finished_at": _iso_now(),
        "duration_s": round(time.monotonic() - started_monotonic, 3),
        "ok": exit_code == EXIT_OK,
        "exit_code": exit_code,
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cwd": os.getcwd(),
            "project_root": str(PROJECT_ROOT),
        },
        "config": {
            "config_path": cfg.config_path,
            "env_file": cfg.env_file,
            "label": cfg.label,
        },
        "api": {
            "base_url": cfg.api.base_url,
            "model": cfg.api.model,
            "thinking": cfg.api.thinking,
            "reasoning_effort": cfg.api.reasoning_effort,
            "image_detail": cfg.api.image_detail,
            "image_format": cfg.api.image_format,
            "max_retries": cfg.api.max_retries,
            "timeout_s": cfg.api.timeout_s,
        },
        "key": cfg.api_key_status(),  # masked; never the key itself
        "pricing": {
            "prices": cfg.api.price_dict(),
            "peak_multiplier_now": peak_multiplier_at(cfg.api),
        },
        "checks": checks,
        "attempts": _sanitize_attempts(real_attempts, "real"),
        "bad_key_attempts": _sanitize_attempts(bad_key_attempts, "bad_key"),
        "totals": totals,
        "failure": failure_payload,
        "notes": [
            "No API key or auth header is stored in this report.",
            "The bad_key check intentionally sends an invalid key once to prove "
            "that HTTP 401 is not retried.",
        ],
    }

    if cfg.api_key() is None:
        _print_missing_key_help(cfg, report_path)
    else:
        _print_checks(checks)
        if failure_payload is not None:
            print(f"  failure : {failure_payload['type']}: {failure_payload['message']}")
        if failed_checks:
            print(
                "  failed  : "
                + ", ".join(str(check.get("name")) for check in failed_checks)
            )
        print(
            f"  cost    : ~${totals['estimated_cost_usd']:.6f} "
            f"({totals['api_attempts']} API attempt(s))"
        )

    try:
        written_path = _write_report(outdir, report)
    except OSError as exc:
        print(f"warning: could not write report: {_preview(exc)}")
        written_path = report_path
    print(f"  report  : {written_path}")
    if exit_code == EXIT_OK:
        print("RESULT: PASS - real DeepSeek API works for text, vision and tool calls (exit 0).")
    elif exit_code == EXIT_NO_KEY:
        print("RESULT: NO KEY - nothing was sent to the API (exit 2).")
    elif exit_code == EXIT_AUTH_FAILURE:
        print("RESULT: AUTH FAILURE - the configured API key was rejected (exit 3).")
    else:
        print("RESULT: FAIL - the real API probe did not pass (exit 4).")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
