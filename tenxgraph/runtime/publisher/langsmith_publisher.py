"""LangsmithPublisher — OtelPublisher pre-configured for LangSmith.

A dedicated publisher that builds a LangSmith OTLP span processor on
construction and installs the ``TracerProvider`` before the first span is
created.  Span dispatch is inherited from ``OtelPublisher``; this class adds the
``langsmith.*`` run fields (run type, name, ``metadata.thread_id``, ``metadata.user_id``).

Usage::

    from tenxgraph.runtime.publisher import LangsmithPublisher

    publisher = LangsmithPublisher(project="my-agent", level=ObservabilityLevel.FULL)
    graph._publisher = publisher
    compiled = graph.compile(...)

Or use the convenience helper which does the same in one call::

    from tenxgraph.runtime.publisher import setup_langsmith

    setup_langsmith(graph, project="my-agent")

Secrets stay in the environment: set ``LANGSMITH_API_KEY`` (or pass ``api_key=``).

Requires: pip install '10xgraph[langsmith]'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._backends import _build_langsmith_processor
from .identity import transform_user_id
from .otel_attributes import (
    LANGSMITH_SPAN_KIND,
    LANGSMITH_THREAD_ID,
    LANGSMITH_TRACE_NAME,
    LANGSMITH_USER_ID,
)
from .otel_publisher import ObservabilityLevel, OtelPublisher, SpanPhase, SpanRole


if TYPE_CHECKING:
    from .events import EventModel


# LangSmith run types: ``chain``, ``llm``, ``tool`` (also retriever, embedding, prompt, parser).
_RUN_TYPES: dict[str, str] = {"graph": "chain", "node": "chain", "llm": "llm", "tool": "tool"}


def langsmith_attributes(
    publisher: OtelPublisher,
    role: SpanRole,
    phase: SpanPhase,
    name: str,
    event: EventModel,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    """Add the ``langsmith.*`` run fields to ``attrs`` when a span starts, and return it.

    Run type and name, plus ``metadata.thread_id`` (LangSmith groups a thread's traces by it,
    and it must be on every run) and ``metadata.user_id``. Content comes from ``gen_ai.*``.
    """
    if phase != "start":
        return attrs
    attrs[LANGSMITH_SPAN_KIND] = _RUN_TYPES[role]
    attrs[LANGSMITH_TRACE_NAME] = name
    attrs[LANGSMITH_THREAD_ID] = str(event.thread_id)
    if event.user_id is not None:
        attrs[LANGSMITH_USER_ID] = transform_user_id(event.user_id, publisher.user_id_mode)
    return attrs


class LangsmithPublisher(OtelPublisher):
    """OtelPublisher pre-configured to send spans to LangSmith over OTLP.

    Builds an OTLP HTTP span processor pointing at LangSmith during
    initialisation and attaches it to either a supplied ``TracerProvider`` or a
    fresh one set as the global provider, so tracing is live before any span is
    created.  All event dispatch and span logic is inherited from
    ``OtelPublisher``.

    This publisher must be assigned to the graph **before** ``graph.compile()``.

    Args:
        api_key: LangSmith API key. Falls back to the ``LANGSMITH_API_KEY`` env
            var.
        project: LangSmith project name.  Sent as the ``Langsmith-Project``
            request header when provided.
        endpoint: Base OTEL endpoint URL for LangSmith.  ``/v1/traces`` is
            appended automatically.  Override for regional deployments, e.g.
            ``https://eu.api.smith.langchain.com/otel``.
        level: How much data to emit.  ``FULL`` by default (message content and graph state; may
            contain PII).  Use ``STANDARD`` or ``SPANS`` to keep content out.
        tracer_provider: An existing ``TracerProvider`` to attach the processor
            to.  When ``None`` a new one is created and set as the global
            provider.

    Raises:
        ImportError: If ``opentelemetry-exporter-otlp-proto-http`` or
            ``opentelemetry-sdk`` are not installed.
        ValueError: If no API key is provided and ``LANGSMITH_API_KEY`` is not
            set.

    Example::

        publisher = LangsmithPublisher(
            project="my-agent",
            level=ObservabilityLevel.STANDARD,
        )
        graph._publisher = publisher
        compiled = graph.compile(...)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project: str | None = None,
        endpoint: str = "https://api.smith.langchain.com/otel",
        level: ObservabilityLevel = ObservabilityLevel.FULL,
        tracer_provider: Any = None,
    ) -> None:
        processor = _build_langsmith_processor(api_key=api_key, project=project, endpoint=endpoint)

        if tracer_provider is None:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            provider = TracerProvider()
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            # A None tracer makes OtelPublisher pick up the global provider set above.
            tracer = None
        else:
            tracer_provider.add_span_processor(processor)
            # Bind to the supplied provider explicitly (it may not be global).
            tracer = tracer_provider.get_tracer("10xgraph")

        super().__init__(tracer=tracer, level=level)

    def span_attributes(
        self,
        role: SpanRole,
        phase: SpanPhase,
        name: str,
        event: EventModel,
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Add the LangSmith run fields (see ``langsmith_attributes``)."""
        return langsmith_attributes(self, role, phase, name, event, attrs)
