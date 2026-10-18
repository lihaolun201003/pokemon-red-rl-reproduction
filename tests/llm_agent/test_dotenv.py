"""Unit tests for :mod:`llm_agent.dotenv` (Lead-owned, verified by api-budget).

The .env helpers are the only supported way to load ``DEEPSEEK_API_KEY`` into the
process; these tests pin the parsing rules and the "never reveal a secret"
behaviour of :func:`mask_secret`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_agent.dotenv import (
    get_env,
    load_env_file,
    looks_like_placeholder,
    mask_secret,
    parse_env_file,
)


def write_env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# parse_env_file
# --------------------------------------------------------------------------- #

def test_parse_env_file_supports_export_quotes_and_comments(tmp_path: Path) -> None:
    path = write_env(
        tmp_path,
        "\n".join(
            [
                "# a comment line",
                "",
                "PLAIN=value",
                "export EXPORTED=exported value",
                'DOUBLE_QUOTED="quoted value"',
                "SINGLE_QUOTED='quoted value'",
                "WITH_EQUALS=a=b=c",
                "INLINE_COMMENT=value # trailing comment",
                "HASH_IN_QUOTES='value # kept'",
                "  SPACES  =  trimmed  ",
                "EMPTY=",
                "not a valid line",
                "1INVALID=skipped",
            ]
        ),
    )
    values = parse_env_file(path)
    assert values["PLAIN"] == "value"
    assert values["EXPORTED"] == "exported value"
    assert values["DOUBLE_QUOTED"] == "quoted value"
    assert values["SINGLE_QUOTED"] == "quoted value"
    assert values["WITH_EQUALS"] == "a=b=c"
    assert values["INLINE_COMMENT"] == "value"
    assert values["HASH_IN_QUOTES"] == "value # kept"
    assert values["SPACES"] == "trimmed"
    assert values["EMPTY"] == ""
    assert "not a valid line" not in values
    assert "1INVALID" not in values


def test_parse_env_file_handles_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes("DEEPSEEK_API_KEY=sk-from-bom\n".encode("utf-8-sig"))
    values = parse_env_file(path)
    assert values["DEEPSEEK_API_KEY"] == "sk-from-bom"


def test_parse_env_file_duplicate_keys_last_wins(tmp_path: Path) -> None:
    path = write_env(tmp_path, "KEY=first\nKEY=second\n")
    assert parse_env_file(path)["KEY"] == "second"


def test_parse_env_file_missing_file_returns_empty(tmp_path: Path) -> None:
    assert parse_env_file(tmp_path / "does-not-exist.env") == {}


# --------------------------------------------------------------------------- #
# load_env_file
# --------------------------------------------------------------------------- #

def test_load_env_file_sets_new_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_AGENT_TEST_NEW", raising=False)
    path = write_env(tmp_path, "LLM_AGENT_TEST_NEW=from-file\n")
    loaded = load_env_file(path)
    assert loaded == {"LLM_AGENT_TEST_NEW": "from-file"}
    assert os.environ["LLM_AGENT_TEST_NEW"] == "from-file"


def test_load_env_file_does_not_override_process_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_AGENT_TEST_EXISTING", "from-process")
    path = write_env(tmp_path, "LLM_AGENT_TEST_EXISTING=from-file\n")
    load_env_file(path)
    assert os.environ["LLM_AGENT_TEST_EXISTING"] == "from-process"


def test_load_env_file_override_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_AGENT_TEST_OVERRIDE", "from-process")
    path = write_env(tmp_path, "LLM_AGENT_TEST_OVERRIDE=from-file\n")
    load_env_file(path, override=True)
    assert os.environ["LLM_AGENT_TEST_OVERRIDE"] == "from-file"


def test_load_env_file_ignores_empty_existing_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_AGENT_TEST_EMPTY", "")
    path = write_env(tmp_path, "LLM_AGENT_TEST_EMPTY=filled\n")
    load_env_file(path)
    assert os.environ["LLM_AGENT_TEST_EMPTY"] == "filled"


# --------------------------------------------------------------------------- #
# get_env
# --------------------------------------------------------------------------- #

def test_get_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_AGENT_TEST_GET", "value")
    assert get_env("LLM_AGENT_TEST_GET") == "value"
    assert get_env("LLM_AGENT_TEST_GET", "fallback") == "value"
    monkeypatch.delenv("LLM_AGENT_TEST_GET", raising=False)
    assert get_env("LLM_AGENT_TEST_GET") is None
    assert get_env("LLM_AGENT_TEST_GET", "fallback") == "fallback"


def test_get_env_treats_empty_as_missing(monkeypatch) -> None:
    monkeypatch.setenv("LLM_AGENT_TEST_EMPTY_ENV", "")
    assert get_env("LLM_AGENT_TEST_EMPTY_ENV") is None
    assert get_env("LLM_AGENT_TEST_EMPTY_ENV", "fallback") == "fallback"


# --------------------------------------------------------------------------- #
# mask_secret / looks_like_placeholder
# --------------------------------------------------------------------------- #

def test_mask_secret_never_returns_the_secret() -> None:
    secret = "sk-1234567890abcdefghijklmnopqrstuvwxyz"
    masked = mask_secret(secret)
    assert secret not in masked
    assert masked == "<set:{} chars, ...{}>".format(len(secret), secret[-2:])
    # The full secret's prefix/body must not leak.
    assert secret[:10] not in masked


def test_mask_secret_handles_missing_and_short_values() -> None:
    assert mask_secret(None) == "<missing>"
    assert mask_secret("") == "<missing>"
    assert mask_secret("short") == "<set:5 chars>"
    assert mask_secret("12345678") == "<set:8 chars>"
    assert mask_secret("123456789") == "<set:9 chars, ...89>"


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "changeme",
        "your_api_key",
        "your-api-key",
        "none",
        "null",
        "todo",
        "<your key>",
        "xxxx",
        "XXXXXXXX",
        "sk-xxxx",
        "sk-xxxxxxxxxxxx",
        "....",
        "0",
    ],
)
def test_looks_like_placeholder_true(value: str | None) -> None:
    assert looks_like_placeholder(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "sk-1234567890abcdef",
        "sk-proj-abcdef0123456789",
        "a-real-looking-key-0123456789",
        "0a1b2c3d4e5f",
    ],
)
def test_looks_like_placeholder_false(value: str) -> None:
    assert looks_like_placeholder(value) is False
