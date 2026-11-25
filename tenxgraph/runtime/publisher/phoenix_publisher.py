"""PhoenixPublisher — OtelPublisher pre-configured for Arize Phoenix.

A dedicated publisher that builds a Phoenix OTLP span processor on
construction and installs the ``TracerProvider`` before the first span is
created.  Span dispatch is inherited from ``OtelPublisher``; this class adds the
OpenInference attributes Phoenix reads (``openinference.span.kind``, ``input.value``,
``llm.*``, ``session.id``, ``user.id``, ``metadata``).

Usage::

    from tenxgraph.core.graph import StateGraph
    from tenxgraph.runtime.publisher import ObservabilityLevel, PhoenixPublisher

    publisher = PhoenixPublisher(project="my-agent", level=ObservabilityLevel.FULL)
    graph = StateGraph(publisher=publisher)
    compiled = graph.compile(...)

Secrets stay in the environment: set ``PHOENIX_API_KEY`` (or pass ``api_key=``).
Local Phoenix needs no key.

Requires: pip install '10xgraph[phoenix]'
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from . import _openinference as oi
from ._backends import _build_phoenix_processor, _new_tracer_provider, _phoenix_project
from .identity import UserIdMode, transform_user_id
from .otel_attributes import SESSION_ID
from .otel_publisher import (
    ObservabilityLevel,
    OtelPublisher,
    SpanPhase,
    SpanRole,
    _provider_system_name,
)


if TYPE_CHECKING:
    from .events import EventModel


_KINDS: dict[str, str] = {
    "graph": oi.KIND_CHAIN,
    "node": oi.KIND_CHAIN,
    "llm": oi.KIND_LLM,
    "tool": oi.KIND_TOOL,
}

# Attributes an LLM span can carry besides the indexed input messages (kind, session, user,
# metadata, model, provider, request params, token counts, output message with up to
# ``oi.MAX_OUTPUT_TOOL_CALLS`` tool calls). The rest of the span attribute limit goes to messages.
_LLM_RESERVED_ATTRS = 64


def _input_message_budget() -> int | None:
    """How many indexed input-message attributes fit on an LLM span, None for no limit.

    The OTEL SDK keeps at most ``OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`` attributes (default 128)
    and evicts the oldest when full, which would drop the span kind, model, session and user
    set first. A limit passed to a TracerProvider in code is not seen; the default is assumed.
    """
    try:
        from opentelemetry.sdk.trace import SpanLimits

        limit = SpanLimits().max_span_attributes
    except (ImportError, ValueError):
        limit = 128
    return None if limit is None else max(limit - _LLM_RESERVED_ATTRS, 0)


def openinference_attributes(  # noqa: PLR0912
    publisher: OtelPublisher,
    role: SpanRole,
    phase: SpanPhase,
    name: str,
    event: EventModel,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    """Add the OpenInference attributes Phoenix reads to ``attrs`` and return it.

    Every span gets ``openinference.span.kind``, ``session.id`` (the thread id), ``user.id``
    and ``metadata`` (JSON with ``thread_id`` and ``run_id``). STANDARD adds model, provider,
    token counts and node state; FULL adds graph, LLM and tool input and output.
    """
    level = publisher.level
    standard = level != ObservabilityLevel.SPANS
    full = level == ObservabilityLevel.FULL
    data = event.data
    start = phase == "start"
    if start:
        attrs[oi.SPAN_KIND] = _KINDS[role]
        # On every span, so a span filter by session or user finds node, LLM and tool spans.
        attrs[SESSION_ID] = str(event.thread_id)
        if event.user_id is not None:
            attrs[oi.USER_ID] = transform_user_id(event.user_id, publisher.user_id_mode)
        attrs[oi.METADATA] = json.dumps(
            {"thread_id": str(event.thread_id), "run_id": str(event.run_id)}
        )
    if role == "graph" and full:
        text = oi.graph_input_text(data) if start else oi.graph_output_text(data)
        if text:
            key, mime = (
                (oi.INPUT_VALUE, oi.INPUT_MIME) if start else (oi.OUTPUT_VALUE, oi.OUTPUT_MIME)
            )
            attrs.update({key: text, mime: oi.TEXT_MIME})
    elif role == "node" and standard:
        attrs.update(oi.node_input_attrs(data) if start else oi.node_output_attrs(data))
    elif role == "llm" and start:
        if standard:
            provider = _provider_system_name(data.get("provider", ""))
            attrs.update(oi.llm_standard_attrs({}, data.get("model", ""), provider))
        if full and data.get("input_messages"):
            attrs.update(
                oi.llm_input_attrs(
                    data["input_messages"], data.get("system_prompt"), _input_message_budget()
                )
            )
    elif role == "llm":
        if standard:
            attrs.update(oi.llm_standard_attrs(data, "", ""))
        if full:
            attrs.update(oi.llm_output_attrs(data))
    elif role == "tool" and full:
        if start:
            args = data.get("args")
            attrs.update(oi.tool_input_attrs(data.get("tool_input") if args is None else args))
        else:
            attrs.update(oi.tool_output_attrs(data))
    return attrs


class PhoenixPublisher(OtelPublisher):
    """OtelPublisher pre-configured to send spans to Arize Phoenix over OTLP.

    Builds an OTLP HTTP span processor pointing at Phoenix during
    initialisation and attaches it to either a supplied ``TracerProvider`` or a
    fresh one set as the global provider, so tracing is live before any span is
    created.  All event dispatch and span logic is inherited from
    ``OtelPublisher``.

    This publisher must be assigned to the graph **before** ``graph.compile()``.

    Args:
        api_key: Phoenix API key. Falls back to the ``PHOENIX_API_KEY`` env var.
            Optional: local Phoenix needs none.
        project: Phoenix project name. Falls back to ``PHOENIX_PROJECT``, then
            ``PHOENIX_PROJECT_NAME``. Set as the ``openinference.project.name``
            resource attribute, so it applies only when a fresh
            ``TracerProvider`` is created here. With a supplied
            ``tracer_provider`` set
            ``OTEL_RESOURCE_ATTRIBUTES=openinference.project.name=<name>``
            before that provider is created.
        endpoint: Phoenix base URL. Falls back to ``PHOENIX_COLLECTOR_ENDPOINT``,
            then ``http://localhost:6006``.  ``/v1/traces`` is appended unless
            present.
        level: How much data to emit.  ``FULL`` by default (message content and graph state; may
            contain PII).  Use ``STANDARD`` or ``SPANS`` to keep content out.
        tracer_provider: An existing ``TracerProvider`` to attach the processor
            to.  When ``None`` a new one is created and set as the global
            provider.
        user_id_mode: ``"raw"`` (default), ``"hash"`` or ``"mask"``: how ``user_id``
            appears on spans (see ``tenxgraph.runtime.publisher.identity``).

    Raises:
        ImportError: If ``opentelemetry-exporter-otlp-proto-http`` or
            ``opentelemetry-sdk`` are not installed.

    Example::

        publisher = PhoenixPublisher(
            project="my-agent",
            level=ObservabilityLevel.STANDARD,
        )
        graph = StateGraph(publisher=publisher)
        compiled = graph.compile(...)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project: str | None = None,
        endpoint: str | None = None,
        level: ObservabilityLevel = ObservabilityLevel.FULL,
        tracer_provider: Any = None,
        user_id_mode: UserIdMode = "raw",
    ) -> None:
        processor = _build_phoenix_processor(api_key=api_key, endpoint=endpoint)

        if tracer_provider is None:
            from opentelemetry import trace

            provider = _new_tracer_provider(_phoenix_project(project))
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            # A None tracer makes OtelPublisher pick up the global provider set above.
            tracer = None
        else:
            tracer_provider.add_span_processor(processor)
            # Bind to the supplied provider explicitly (it may not be global).
            tracer = tracer_provider.get_tracer("10xgraph")

        super().__init__(tracer=tracer, level=level, user_id_mode=user_id_mode)

    def span_attributes(
        self,
        role: SpanRole,
        phase: SpanPhase,
        name: str,
        event: EventModel,
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Add the OpenInference attributes (see ``openinference_attributes``)."""
        return openinference_attributes(self, role, phase, name, event, attrs)
