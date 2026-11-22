"""The tool result for a call blocked with ``ToolDenied``, shared by local and MCP tools."""

from __future__ import annotations

import logging
import typing as t

from tenxgraph.core.authz import owner_id
from tenxgraph.core.exceptions import ToolDenied
from tenxgraph.core.state import ErrorBlock, Message, ToolResultBlock
from tenxgraph.runtime.publisher.events import ContentType, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.utils.constants import TRUSTED_TOOL_RESULT_META
from tenxgraph.utils.log_safety import log_safe


logger = logging.getLogger("tenxgraph.graph.tool_node")

# ``ErrorBlock.code`` and ``Message.metadata`` flag of a denied call. The result block keeps
# the existing ``status="failed"``, so stored threads and clients need no new status value.
TOOL_DENIED_CODE = "TOOL_DENIED"
TOOL_DENIED_META = "tool_denied"


def _user_id(config: dict[str, t.Any]) -> t.Any:
    try:
        return owner_id(config)
    except ValueError:  # a malformed identity; the denial is still logged
        return None


def denied_tool_message(
    denied: ToolDenied,
    name: str,
    tool_call_id: str,
    config: dict[str, t.Any],
    event: EventModel,
    meta: dict[str, t.Any],
) -> Message:
    """Log the denial, publish the tool event as denied, and build the model's tool result.

    Argument values are never logged: they may hold personal data.
    """
    if denied.tool_name is None:
        denied.tool_name = name
    text = f"Denied by policy: {name}. {denied.reason}"
    logger.warning(
        "Tool '%s' denied by policy (user %s): %s",
        log_safe(name),
        log_safe(str(_user_id(config))),
        log_safe(denied.reason, 500),
    )

    event.event_type = EventType.END
    event.data["error"] = text
    event.metadata["status"] = "denied"
    event.metadata["denied"] = True
    event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
    publish_event(event)

    return Message.tool_message(
        content=[
            ToolResultBlock(
                call_id=tool_call_id,
                name=name,
                output=text,
                status="failed",
                is_error=True,
            ),
            ErrorBlock(message=text, code=TOOL_DENIED_CODE),
        ],
        meta={**meta, TOOL_DENIED_META: True, TRUSTED_TOOL_RESULT_META: True},
    )
