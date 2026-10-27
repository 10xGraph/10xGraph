"""Guards applied when a ToolNode dispatches the tool calls of a model message.

Shared by the invoke and stream node handlers so both paths enforce the same rules:
which ``transfer_to_*`` calls count as handoffs, which calls the model was allowed
to make, how many may run at once, and where a ``Command`` may send the run.
"""

import logging
import os
from collections.abc import Iterable
from typing import Any

from tenxgraph.core.exceptions import GraphError
from tenxgraph.core.state import Message
from tenxgraph.core.state.message_block import ErrorBlock, ToolResultBlock
from tenxgraph.utils.constants import (
    DEFAULT_MAX_PARALLEL_TOOL_CALLS,
    END,
    MAX_PARALLEL_TOOL_CALLS_ENV,
)


logger = logging.getLogger("tenxgraph.graph")


def resolve_max_parallel_tool_calls(config: dict[str, Any] | None) -> int:
    """Return how many tool calls may run at once for this run.

    Order: ``config["max_parallel_tool_calls"]``, then the
    ``TENXGRAPH_MAX_PARALLEL_TOOL_CALLS`` env var, then the default. A value that is
    not an integer or is below 1 logs a warning and falls back to the default.
    """
    raw: Any = (config or {}).get("max_parallel_tool_calls")
    source = "config max_parallel_tool_calls"
    if raw is None:
        raw = os.environ.get(MAX_PARALLEL_TOOL_CALLS_ENV)
        source = MAX_PARALLEL_TOOL_CALLS_ENV
    if raw is None or raw == "":
        return DEFAULT_MAX_PARALLEL_TOOL_CALLS

    try:
        if isinstance(raw, bool):
            raise ValueError("bool is not a valid limit")
        value = int(raw)
    except (TypeError, ValueError):
        value = 0
    if value < 1:
        logger.warning(
            "Invalid %s=%r (need an integer >= 1); using default %d",
            source,
            raw,
            DEFAULT_MAX_PARALLEL_TOOL_CALLS,
        )
        return DEFAULT_MAX_PARALLEL_TOOL_CALLS
    return value


def allowed_tool_names(message: Message) -> set[str] | None:
    """Names the model was shown when it produced ``message``, or None if unfiltered."""
    allowed = (message.metadata or {}).get("allowed_tools")
    if allowed is None:
        return None
    return {str(n) for n in allowed}


def tool_call_name(tool_call: dict[str, Any]) -> str:
    """Return the tool name of a stored tool call.

    Provider converters store ``{"id", "type": "function", "function": {"name", ...}}``.
    The flat ``{"name": ...}`` shape is read as a fallback for hand-built calls.
    """
    function = tool_call.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    name = tool_call.get("name")
    return str(name) if name else ""


def handoff_target(tool_node: Any, tool_name: str) -> str | None:
    """Return the target node if ``tool_name`` is a registered handoff tool.

    A name that merely starts with ``transfer_to_`` is not enough: the ToolNode must
    hold a tool of that name created by ``create_handoff_tool``. Otherwise the model
    could steer the run to any node by emitting a made-up call.
    """
    funcs = getattr(tool_node, "_funcs", None)
    if not isinstance(funcs, dict):
        return None
    fn = funcs.get(tool_name)
    if fn is None or not getattr(fn, "__handoff_tool__", False):
        return None
    target = getattr(fn, "__target_agent__", None)
    return target if isinstance(target, str) and target else None


def hidden_tool_message(tool_call: dict[str, Any]) -> Message:
    """The error result for a call to a tool that was not offered to the model.

    Same shape as the ToolNode's result for an unknown tool.
    """
    name = tool_call.get("function", {}).get("name", "")
    error_msg = f"Tool '{name}' not found."
    return Message.tool_message(
        content=[
            ErrorBlock(message=error_msg),
            ToolResultBlock(
                call_id=tool_call.get("id", ""),
                output=error_msg,
                is_error=True,
                status="failed",
            ),
        ],
    )


def validate_goto(target: str | None, node_names: Iterable[str], from_node: str) -> None:
    """Raise GraphError if a Command targets a node that does not exist."""
    if target is None or target == END:
        return
    names = set(node_names)
    if target in names:
        return
    raise GraphError(
        message=(
            f"Node '{from_node}' routed to '{target}', which is not a node in this graph "
            f"(and not END)."
        ),
        error_code="GRAPH_007",
        context={"from_node": from_node, "target": target, "nodes": sorted(names)},
    )
