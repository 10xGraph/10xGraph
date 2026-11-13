"""Make a run's 10xGraph span the current OTEL span for a block of execution code.

Libraries that trace with the global TracerProvider (the Anthropic SDK, fastmcp, the ``mcp``
SDK) parent their spans to the current OTEL context. ``OtelPublisher`` starts the graph, node,
LLM and tool spans when their START event is published, so execution code can look one up
and make it current around the LLM request or MCP session; the library spans then nest under
it in the run's trace.

Without an ``OtelPublisher`` bound (or without the ``otel`` extra) every scope here is a
``nullcontext`` and nothing from OpenTelemetry is imported.
"""

from __future__ import annotations

import contextlib
from typing import Any

from injectq import Inject

from tenxgraph.utils.injection import fresh

from .base_publisher import BasePublisher
from .composite_publisher import CompositePublisher
from .otel_publisher import OtelPublisher, llm_span_key, tool_span_key


def _otel_publishers(publisher: BasePublisher) -> list[OtelPublisher]:
    if isinstance(publisher, CompositePublisher):
        return [p for c in publisher._publishers for p in _otel_publishers(c)]
    return [publisher] if isinstance(publisher, OtelPublisher) else []


def find_span(
    config: dict[str, Any] | None,
    *keys: str,
    publisher: BasePublisher | None = Inject[BasePublisher],
) -> Any:
    """The open span of this run under the first of ``keys`` that has one, else None."""
    publisher = fresh(publisher)
    run_id = (config or {}).get("run_id")
    if publisher is None or not run_id:
        return None
    for otel in _otel_publishers(publisher):
        for key in keys:
            span = otel.open_span(str(run_id), key)
            if span is not None:
                return span
    return None


def use_trace_span(span: Any) -> contextlib.AbstractContextManager[Any]:
    """Make ``span`` current for a ``with`` block; a no-op when ``span`` is None.

    The span is not ended and exceptions are not recorded on it here: the publisher does that
    from the END and ERROR events. The context must be attached and detached in the same
    context: keep the block in one frame, or in an ``asynccontextmanager`` (its enter and exit
    run in the caller's task), never across the ``yield`` of an async generator that someone
    else iterates.
    """
    if span is None:
        return contextlib.nullcontext()
    from opentelemetry.trace import use_span

    return use_span(span, end_on_exit=False, record_exception=False, set_status_on_exception=False)


def llm_span(config: dict[str, Any] | None, node_name: str) -> Any:
    """The LLM span of node ``node_name``, else that node's span."""
    return find_span(config, llm_span_key(node_name), node_name)


def tool_span(config: dict[str, Any] | None, tool_call_id: str | None = None) -> Any:
    """The span of tool call ``tool_call_id``, else the span of the node in ``config``."""
    node = str((config or {}).get("_node_name", ""))
    if tool_call_id:
        return find_span(config, tool_span_key(tool_call_id), node)
    return find_span(config, node)
