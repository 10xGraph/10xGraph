"""Helpers that keep model-controlled text from forging log lines."""

from __future__ import annotations

import re
import traceback
from typing import Any


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


# Bounds of the exception text an event carries; the traceback keeps its end (the frames
# nearest the error and the exception line).
MAX_ERROR_MESSAGE_CHARS = 2000
MAX_TRACEBACK_CHARS = 8000
# Keys of ``error_details`` that only observability publishers receive (see publish_event).
TRACE_ONLY_ERROR_KEYS = ("error_message", "error_traceback")


def error_details(exc: BaseException, config: Any = None) -> dict[str, str]:
    """``error_type``, ``error_message`` and ``error_traceback`` of ``exc`` for an event.

    Message and traceback go through ``mask_secrets`` (key formats, Bearer tokens, signed-URL
    parameters), the token of the run's ``UserCredential`` in ``config`` is replaced, and they
    are cut to 2000 and the last 8000 characters. They can still hold paths, DSNs or SQL, so
    ``publish_event`` sends them to trace backends only.
    """
    from tenxgraph.core.authz import CREDENTIAL_KEY, UserCredential
    from tenxgraph.utils.logging import mask_secrets

    credential = config.get(CREDENTIAL_KEY) if isinstance(config, dict) else None
    token = credential.token.get_secret_value() if isinstance(credential, UserCredential) else ""

    def clean(text: str) -> str:
        return mask_secrets(text.replace(token, "***REDACTED***") if token else text)

    message = clean(str(exc))
    if len(message) > MAX_ERROR_MESSAGE_CHARS:
        message = message[:MAX_ERROR_MESSAGE_CHARS] + "..."
    trace = clean("".join(traceback.format_exception(exc)))
    if len(trace) > MAX_TRACEBACK_CHARS:
        trace = "..." + trace[-MAX_TRACEBACK_CHARS:]
    return {
        "error_type": type(exc).__name__,
        "error_message": message,
        "error_traceback": trace,
    }
