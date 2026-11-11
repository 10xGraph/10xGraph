"""PhoenixPublisher — OtelPublisher pre-configured for Arize Phoenix.

A dedicated publisher that builds a Phoenix OTLP span processor on
construction and installs the ``TracerProvider`` before the first span is
created.  All span dispatch logic is inherited from ``OtelPublisher``.

Usage::

    from tenxgraph.runtime.publisher import PhoenixPublisher

    publisher = PhoenixPublisher(project="my-agent", level=ObservabilityLevel.FULL)
    graph._publisher = publisher
    compiled = graph.compile(...)

Or use the convenience helper which does the same in one call::

    from tenxgraph.runtime.publisher import setup_phoenix

    setup_phoenix(graph, project="my-agent")

Secrets stay in the environment: set ``PHOENIX_API_KEY`` (or pass ``api_key=``).
Local Phoenix needs no key.

Requires: pip install '10xgraph[phoenix]'
"""

from __future__ import annotations

from typing import Any

from ._backends import _build_phoenix_processor, _new_tracer_provider, _phoenix_project
from .otel_publisher import ObservabilityLevel, OtelPublisher


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

    Raises:
        ImportError: If ``opentelemetry-exporter-otlp-proto-http`` or
            ``opentelemetry-sdk`` are not installed.

    Example::

        publisher = PhoenixPublisher(
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
        endpoint: str | None = None,
        level: ObservabilityLevel = ObservabilityLevel.FULL,
        tracer_provider: Any = None,
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

        super().__init__(tracer=tracer, level=level)
