"""Zero-dependency ``.env`` loader for the LLM agent.

Only used to read local configuration (most importantly ``DEEPSEEK_API_KEY``).
Values are never logged.  Existing process environment variables win over the
file so that a user can override a checked-out ``.env`` from the shell.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` file.  Supports ``export``, quotes and comments."""
    path = Path(path)
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not _KEY_RE.match(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # strip trailing inline comments only for unquoted values
            comment = value.find(" #")
            if comment >= 0:
                value = value[:comment].rstrip()
        result[key] = value
    return result


def load_env_file(path: str | Path, *, override: bool = False) -> dict[str, str]:
    """Load a ``.env`` file into ``os.environ``.

    Returns the parsed values.  By default existing environment variables are
    left untouched (process environment has priority).
    """
    values = parse_env_file(path)
    for key, value in values.items():
        if override or not os.environ.get(key):
            os.environ[key] = value
    return values


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def mask_secret(value: str | None) -> str:
    """Return a safe representation of a secret (never the secret itself)."""
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "<set:{} chars>".format(len(value))
    return "<set:{} chars, ...{}>".format(len(value), value[-2:])


def looks_like_placeholder(value: str | None) -> bool:
    """Detect obvious placeholder values such as ``sk-xxxx`` or ``<your key>``."""
    if not value:
        return True
    lowered = value.strip().lower()
    if lowered in {"changeme", "your_api_key", "your-api-key", "none", "null", "todo"}:
        return True
    if lowered.startswith("<") and lowered.endswith(">"):
        return True
    if set(lowered) <= {"x", "x-", "0", "*", "."}:
        return True
    if re.fullmatch(r"(sk-)?x{4,}", lowered):
        return True
    return False
