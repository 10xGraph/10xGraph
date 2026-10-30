"""Helpers that keep model-controlled text from forging log lines."""

from __future__ import annotations

import re


_CONTROL = re.compile(r"[\x00-\x1f\x7f\u2028\u2029]")


def log_safe(value: object, limit: int = 200) -> str:
    """Return ``value`` as one log-safe line.

    Control characters (CR, LF, tab, NUL and so on) are replaced by their escape
    form so a model-chosen tool name cannot start a fake log record. The text is
    cut to ``limit`` characters.
    """
    text = value if isinstance(value, str) else str(value)
    text = _CONTROL.sub(lambda m: m.group().encode("unicode_escape").decode("ascii"), text)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text
