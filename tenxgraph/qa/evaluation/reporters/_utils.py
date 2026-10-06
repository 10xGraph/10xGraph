"""
Shared reporter utilities.

Centralises common formatting / IO logic used by multiple reporters
to avoid duplication and ensure consistency.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from agentflow.qa.evaluation.eval_result import EvalCaseResult


def format_timestamp(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a UNIX timestamp as a human-readable string."""
    return datetime.fromtimestamp(ts).strftime(fmt)


def case_display_name(result: EvalCaseResult) -> str:
    """Return a display name for a case result."""
    return result.name or result.eval_id


def case_status_info(result: EvalCaseResult) -> tuple[str, str]:
    """Return ``(status, icon)`` for a case result.

    Returns:
        Tuple of (status_key, icon_char) where status_key is one of
        ``'error'``, ``'pass'``, ``'fail'``.
    """
    if result.is_error:
        return "error", "!"
    if result.passed:
        return "pass", "✓"
    return "fail", "✗"


def format_tool_calls(tool_calls: list[dict[str, Any]] | list[Any]) -> list[dict[str, str]]:
    """Normalise a list of tool-call dicts/models into simple display dicts.

    Each returned dict has keys ``name``, ``args``, ``result``, ``call_id``.
    No truncation — full data is preserved for complete reporting.
    """
    out: list[dict[str, str]] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            out.append(
                {
                    "name": str(tc.get("name", "")),
                    "args": _compact_json(tc.get("args", {})),
                    "result": str(tc.get("result", "")),
                    "call_id": str(tc.get("call_id", "") or ""),
                }
            )
        elif hasattr(tc, "model_dump"):
            d = tc.model_dump()
            out.append(
                {
                    "name": str(d.get("name", "")),
                    "args": _compact_json(d.get("args", {})),
                    "result": str(d.get("result", "")),
                    "call_id": str(d.get("call_id", "") or ""),
                }
            )
    return out


def _compact_json(obj: Any) -> str:
    """One-line JSON-ish representation. No truncation — full data preserved."""
    import json

    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    return s
