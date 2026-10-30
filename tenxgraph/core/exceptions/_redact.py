"""Redaction of error context before it is written to a log."""

from __future__ import annotations

import re
from typing import Any


_SECRET_KEY = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|credential|cookie|private[_-]?key)",
    re.IGNORECASE,
)
_REDACTED = "***REDACTED***"
_MAX_DEPTH = 5


def _scrub(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return "..."
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _SECRET_KEY.search(str(k)) else _scrub(v, depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [_scrub(v, depth + 1) for v in value]
    return value


def redact_context(context: Any) -> str:
    """Return ``context`` as a string that is safe to log.

    Values under secret-looking keys are replaced, and the rest of the text goes through
    ``mask_secrets`` (key formats, Bearer tokens, signed-URL parameters).
    """
    from tenxgraph.utils.logging import mask_secrets

    try:
        text = repr(_scrub(context))
    except Exception:
        return "<unprintable context>"
    return mask_secrets(text)
