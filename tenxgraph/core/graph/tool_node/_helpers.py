"""Shared helper functions and constants for tool node executors."""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import json
import logging
import pathlib
import time
import typing as t
import uuid

from tenxgraph.core.state import ErrorBlock, Message, ToolCallBlock, ToolResultBlock
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import listening, publish_event
from tenxgraph.utils.log_safety import error_details, log_safe


logger = logging.getLogger("tenxgraph.graph.tool_node")

_STATUS_OK: set[str] = {"completed", "success", "ok", "done", "true", "1"}
_STATUS_FAIL: set[str] = {"failed", "failure", "error", "false", "0"}
_ERROR_TRUE: set[str] = {"true", "1", "yes", "error", "failed", "failure"}


def _stable_members(obj: set | frozenset) -> list:
    """Order a set so the same result serializes identically on every run.

    Set iteration order is not stable across processes, which would make tool output
    flap between runs and tests flaky.
    """
    try:
        return sorted(obj)
    except TypeError:
        # Mixed or uncomparable members: order is unavoidably arbitrary here, but
        # losing the values entirely would be worse.
        return list(obj)


# Rendering rules for the scalar types a tool can return but JSON cannot hold. Ordered:
# the first matching entry wins. Decimal renders as str, not float, so a money value
# does not lose precision on the way to the model.
_JSON_ENCODERS: tuple[tuple[t.Any, t.Callable[[t.Any], t.Any]], ...] = (
    ((dt.datetime, dt.date, dt.time), lambda o: o.isoformat()),
    # A path renders with POSIX separators, not str(): str() emits a backslash on
    # Windows, which both breaks the URI the model is meant to reuse and makes the
    # same tool return different text per platform. `prebuilt/tools/files.py` already
    # returns `as_posix()`, so this matches the convention the rest of the package
    # sets. UUID stays on its own rule: it is not a PurePath and has no as_posix().
    (pathlib.PurePath, lambda o: o.as_posix()),
    (uuid.UUID, str),
    (decimal.Decimal, str),
    (enum.Enum, lambda o: o.value),
    ((set, frozenset), _stable_members),
    ((bytes, bytearray), lambda o: o.decode("utf-8", errors="replace")),
)


def _json_default(obj: t.Any) -> t.Any:
    """Render the scalar types a tool can return but JSON cannot hold.

    Mirrors the scalars ``schema.py`` accepts on the way in, so a value the model sent
    as ``"2026-01-15T09:30:00"`` comes back in that same textual form rather than as a
    Python repr.

    Raises:
        TypeError: For anything else, so ``json.dumps`` propagates and the caller falls
            through to its existing repr fallback. Unknown objects must not be silently
            stringified here, or a genuinely unserializable payload would look clean.
    """
    for types_, encode in _JSON_ENCODERS:
        if isinstance(obj, types_):
            return encode(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _normalize_resource_uri(dumped: t.Any) -> t.Any:
    """Render a resource ``uri`` with forward slashes, in place.

    A URI is defined with forward slashes, but ``str(Path)`` emits a backslash on
    Windows. Left alone, the model is handed a URI it cannot reuse and the same tool
    returns different text on each platform.
    """
    if isinstance(dumped, dict) and dumped.get("type") == "resource":
        resource = dumped.get("resource", {})
        if isinstance(resource, dict) and "uri" in resource:
            uri = resource["uri"]
            resource["uri"] = uri.as_posix() if isinstance(uri, pathlib.PurePath) else str(uri)
            dumped["resource"] = resource
    return dumped


def _safe_serialize(obj: t.Any) -> dict[str, t.Any]:
    try:
        json.dumps(obj)
        return obj if isinstance(obj, dict) else {"content": obj}
    except (TypeError, OverflowError):
        if hasattr(obj, "model_dump"):
            dumped = obj.model_dump()  # type: ignore
            return _normalize_resource_uri(dumped)

        # Retry with the scalar renderer so a container keeps its shape and only the
        # offending leaves become text. Without this a single datetime collapses the
        # whole return value into one repr string.
        try:
            normalized = json.loads(json.dumps(obj, default=_json_default))
        except (TypeError, OverflowError, ValueError):
            return {"content": str(obj), "type": "fallback"}
        return normalized if isinstance(normalized, dict) else {"content": normalized}


def _as_bool(val: t.Any, truthy_set: set[str]) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in truthy_set


def _extract_block_meta(data: dict[str, t.Any]) -> bool:
    """Read the status/error keys of a tool result and return ``is_error``.

    The keys are only read. They stay in the data, which reaches the model unchanged.
    """
    raw_status = data.get("status")
    raw_is_error = data["is_error"] if "is_error" in data else data.get("error")
    raw_success = data.get("success")

    if raw_is_error is not None:
        is_error = _as_bool(raw_is_error, _ERROR_TRUE)
    elif raw_success is not None:
        is_error = not _as_bool(raw_success, _STATUS_OK)
    else:
        is_error = False

    if raw_status is not None:
        s = str(raw_status).lower()
        if s in _STATUS_FAIL:
            is_error = True

    return is_error


def generic_tool_failure(
    tool_name: str,
    tool_call_id: str,
    exc: BaseException,
    *,
    log: logging.Logger | None = None,
) -> str:
    """Log an unexpected tool exception and return a message safe for the model.

    The raw exception text can carry paths, DSNs or SQL, so it stays in the server log.
    The returned text carries a short reference that matches the log record, so an
    operator can find the failure behind a user report.
    """
    ref = uuid.uuid4().hex[:8]
    (log or logger).error(
        "Tool '%s' (tool_call_id=%s) failed with an internal error (reference %s)",
        log_safe(tool_name),
        log_safe(tool_call_id),
        ref,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return f"Tool '{tool_name}' failed with an internal error (reference {ref})."


def tool_start_event(
    config: dict[str, t.Any],
    name: str,
    args: dict[str, t.Any],
    tool_call_id: str,
    *,
    is_mcp: bool | None = None,
    is_remote: bool = False,
) -> EventModel:
    """The TOOL_EXECUTION START event of one tool call; the caller publishes it.

    One tool call sends one START and one END (or ERROR), from the code that runs it.
    ``node_name`` (and ``metadata["graph_node"]``) is the graph node running the tool; the
    tool name, as the model called it, is ``data["function_name"]``.
    """
    graph_node = config.get("_node_name", "")
    event = EventModel.default(
        config,
        data={"args": args, "tool_call_id": tool_call_id, "function_name": name},
        content_type=[ContentType.TOOL_CALL],
        event=Event.TOOL_EXECUTION,
        event_type=EventType.START,
        node_name=graph_node,
        # Kept for tracing, which parents the tool span to this graph node.
        extra={"graph_node": graph_node},
    )
    event.content_blocks = [ToolCallBlock(id=tool_call_id, name=name, args=args)]
    event._started_at = time.perf_counter()
    if is_remote:
        event.metadata["is_remote"] = True
    if is_mcp is not None:
        event.data["is_mcp"] = is_mcp
        event.metadata["is_mcp"] = is_mcp
        event.metadata["function_type"] = "mcp" if is_mcp else "internal"
    return event


def tool_message_failed(message: Message | None) -> bool:
    """Whether a tool result message reports a failure (an error block or an error result)."""
    return message is not None and any(
        isinstance(b, ErrorBlock) or (isinstance(b, ToolResultBlock) and b.is_error)
        for b in message.content or []
    )


def publish_tool_end(
    event: EventModel,
    status: str,
    *,
    message: Message | None = None,
    error: str | None = None,
    exc: BaseException | None = None,
    config: dict[str, t.Any] | None = None,
    event_type: EventType = EventType.END,
) -> None:
    """Publish the END (or ERROR) of a tool call with its result ``message`` and/or ``error``.

    ``data["status"]`` is ``"error"`` when the call raised, was denied, ended as an ERROR
    event or returned an error result, else ``"success"``; ``data["duration_ms"]`` is the time
    since the START event was built. ``exc`` adds ``error_details`` (trace backends only).
    Nothing is built when no publisher is bound.
    """
    if not listening():
        return
    failed = bool(error) or event_type == EventType.ERROR or tool_message_failed(message)
    event.data["status"] = "error" if failed else "success"
    if event._started_at is not None:
        event.data["duration_ms"] = (time.perf_counter() - event._started_at) * 1000.0
    event.event_type = event_type
    event.metadata["status"] = status
    kind = ContentType.ERROR if error else ContentType.MESSAGE
    event.content_type = [ContentType.TOOL_RESULT, kind]
    if message is not None:
        dumped = message.model_dump()
        event.data["message"] = dumped
        event.content_blocks = [
            ToolResultBlock(
                call_id=event.data.get("tool_call_id", ""),
                name=event.data.get("function_name"),
                output=dumped,
            )
        ]
    if error:
        event.data["error"] = error
    if exc is not None:
        event.data.update(error_details(exc, config))
    publish_event(event)


def publish_tool_cancelled(event: EventModel, config: dict[str, t.Any]) -> None:
    """Publish the ERROR end of a tool call that was cancelled (``tool_timeout``, stop, shutdown).

    Called from an ``except CancelledError`` block before it re-raises, so it never raises:
    a failing publisher must not replace the cancellation. ``publish_tool_end`` is sync and
    only queues the event, so nothing here awaits.
    """
    try:
        publish_tool_end(
            event,
            "Tool execution cancelled",
            error="Tool execution cancelled",
            config=config,
            event_type=EventType.ERROR,
        )
    except Exception:
        logger.warning("Could not publish the end of a cancelled tool call", exc_info=True)
