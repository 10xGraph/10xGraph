"""Shared parsing of model-produced tool-call arguments.

Models do not always emit valid JSON for tool calls: output cut off at the token
limit, an empty string, a list or a bare string instead of an object. Every
converter and the ToolNode use :func:`parse_tool_arguments` so all providers and
both execution paths treat such calls identically.
"""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger("tenxgraph.llm")

_LOG_LIMIT = 200

#: Result text sent back to the model when a tool call's arguments are unusable.
INVALID_TOOL_ARGUMENTS_MESSAGE = (
    "Tool call arguments were not valid JSON (they may have been cut off). "
    "Call the tool again with a complete JSON object."
)


def _preview(raw: Any) -> str:
    text = raw if isinstance(raw, str) else repr(raw)
    if len(text) > _LOG_LIMIT:
        return f"{text[:_LOG_LIMIT]}... ({len(text)} chars)"
    return text


def _invalid(tool_name: str, error: str, raw: Any) -> tuple[dict[str, Any], str]:
    logger.warning("Tool '%s' arguments invalid: %s; arguments=%s", tool_name, error, _preview(raw))
    return {}, error


def parse_tool_arguments(raw: Any, *, tool_name: str = "") -> tuple[dict[str, Any], str | None]:
    """Parse tool-call arguments into ``(args, error)``.

    * a dict passes through
    * ``None``, an empty or whitespace-only string means no arguments: ``({}, None)``
    * a JSON string decoding to an object gives that object
    * invalid JSON or a non-object value gives ``({}, error_message)``

    The caller must not run the tool when ``error`` is set.
    """
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if not isinstance(raw, str):
        return _invalid(
            tool_name, f"arguments must be a JSON object, got {type(raw).__name__}", raw
        )
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        return _invalid(tool_name, f"arguments are not valid JSON: {exc}", raw)
    if not isinstance(parsed, dict):
        return _invalid(
            tool_name, f"arguments must be a JSON object, got {type(parsed).__name__}", raw
        )
    return parsed, None
