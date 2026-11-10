"""Logfire, LangSmith, Phoenix and generic OTLP observability helpers for 10xGraph.

All backends are OpenTelemetry-native, so none needs a bespoke publisher.
They attach as span processors to the TracerProvider that OtelPublisher already
feeds, then `setup_tracing` is called as normal.

Usage:

    # Logfire only
    from tenxgraph.runtime.publisher import setup_logfire
    setup_logfire(graph, service_name="my-agent")

    # LangSmith only
    from tenxgraph.runtime.publisher import setup_langsmith
    setup_langsmith(graph, project="my-project")

    # Arize Phoenix only (local Phoenix needs no API key)
    from tenxgraph.runtime.publisher import setup_phoenix
    setup_phoenix(graph, project="my-project")

    # Any OTLP HTTP endpoint (collector, Grafana Tempo, Jaeger, ...)
    # Auth headers come from OTEL_EXPORTER_OTLP_HEADERS.
    from tenxgraph.runtime.publisher import setup_otlp
    setup_otlp(graph, endpoint="http://collector:4318")

    # Any mix at once (shares one TracerProvider)
    from tenxgraph.runtime.publisher import setup_observability
    setup_observability(graph, {
        "logfire":   {"enabled": True, "service_name": "my-agent"},
        "langsmith": {"enabled": True, "project": "my-project"},
        "phoenix":   {"enabled": True, "project": "my-project"},
        "otlp":      {"enabled": True, "endpoint": "http://collector:4318"},
    })

Requires extras:
  pip install '10xgraph[logfire]'       # Logfire
  pip install '10xgraph[langsmith]'     # LangSmith
  pip install '10xgraph[phoenix]'       # Phoenix
  pip install '10xgraph[otlp]'          # generic OTLP HTTP
  pip install '10xgraph[observability]' # all of them
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ._backends import (
    _build_langsmith_processor,
    _build_otlp_processor,
    _build_phoenix_processor,
    _guard_logfire,
    _logfire_accepts_resource_attributes,
    _new_tracer_provider,
    _phoenix_project,
)
from .identity import UserIdMode, validate_user_id_mode
from .otel_publisher import ObservabilityLevel, setup_tracing


logger = logging.getLogger("tenxgraph.runtime.publisher.exporters")

if TYPE_CHECKING:
    from tenxgraph.core.graph.state_graph import StateGraph


# ── Public helpers ─────────────────────────────────────────────────────────────


def setup_logfire(
    graph: StateGraph | None = None,
    *,
    token: str | None = None,
    service_name: str | None = None,
    send_to_logfire: bool = True,
    console: Any = None,
    level: ObservabilityLevel = ObservabilityLevel.STANDARD,
    additional_span_processors: list | None = None,
    user_id_mode: UserIdMode = "raw",
    **configure_kwargs: Any,
) -> None:
    """Configure Logfire as the OTEL backend and instrument the graph.

    Calls ``logfire.configure(...)`` to install the global TracerProvider, then
    calls ``setup_tracing`` so OtelPublisher feeds into it.

    Must be called **before** ``graph.compile()``.

    Args:
        graph: The StateGraph to instrument.
        token: Logfire write token. Falls back to the ``LOGFIRE_TOKEN`` env var.
        service_name: Service name shown in the Logfire UI.
        send_to_logfire: Whether to export spans to logfire.dev.
        console: Console output options. Pass ``False`` to disable, or a
            ``logfire.ConsoleOptions`` instance. ``None`` uses env-var defaults.
        level: Observability level — SPANS, STANDARD (default), or FULL.
        additional_span_processors: Extra SpanProcessors to attach alongside
            the Logfire processor (e.g. a LangSmith OTLP processor).
        user_id_mode: ``"raw"`` (default), ``"hash"`` or ``"mask"``: how ``user_id``
            appears on spans (see ``tenxgraph.runtime.publisher.identity``).
        **configure_kwargs: Additional keyword arguments forwarded to
            ``logfire.configure()``.

    Raises:
        ImportError: If ``logfire`` is not installed.
    """
    _guard_logfire()
    import logfire

    kwargs: dict[str, Any] = {
        "send_to_logfire": send_to_logfire,
        **configure_kwargs,
    }
    if token is not None:
        kwargs["token"] = token
    if service_name is not None:
        kwargs["service_name"] = service_name
    if console is not None:
        kwargs["console"] = console
    if additional_span_processors:
        kwargs["additional_span_processors"] = additional_span_processors

    logfire.configure(**kwargs)
    # graph is None when only provider configuration is wanted (e.g. the API
    # bootstrap binds the publisher into the DI container itself).
    if graph is not None:
        setup_tracing(graph, level=level, user_id_mode=user_id_mode)


def setup_langsmith(
    graph: StateGraph | None = None,
    *,
    api_key: str | None = None,
    project: str | None = None,
    endpoint: str = "https://api.smith.langchain.com/otel",
    level: ObservabilityLevel = ObservabilityLevel.STANDARD,
    tracer_provider: Any = None,
    user_id_mode: UserIdMode = "raw",
) -> None:
    """Attach a LangSmith OTLP HTTP exporter and instrument the graph.

    Builds an ``OTLPSpanExporter`` pointing at LangSmith's OTEL endpoint,
    wraps it in a ``BatchSpanProcessor``, and attaches it to either the
    supplied ``tracer_provider`` or a fresh ``TracerProvider`` set as the
    global provider.  Then calls ``setup_tracing`` so ``OtelPublisher`` feeds into it.

    Must be called **before** ``graph.compile()``.

    Args:
        graph: The StateGraph to instrument.
        api_key: LangSmith API key. Falls back to the ``LANGSMITH_API_KEY``
            env var.
        project: LangSmith project name.  Sent as the ``Langsmith-Project``
            request header when provided.
        endpoint: Base OTEL endpoint URL for LangSmith.  The exporter appends
            ``/v1/traces`` automatically.  Override for regional deployments,
            e.g. ``https://eu.api.smith.langchain.com/otel``.
        level: Observability level — SPANS, STANDARD (default), or FULL.
        tracer_provider: An existing ``TracerProvider`` to attach the processor
            to.  When ``None`` a new one is created and set as the global
            provider.
        user_id_mode: ``"raw"`` (default), ``"hash"`` or ``"mask"``: how ``user_id``
            appears on spans (see ``tenxgraph.runtime.publisher.identity``).

    Raises:
        ImportError: If ``opentelemetry-exporter-otlp-proto-http`` or
            ``opentelemetry-sdk`` are not installed.
        ValueError: If no API key is provided and ``LANGSMITH_API_KEY`` is not
            set.
    """
    processor = _build_langsmith_processor(api_key=api_key, project=project, endpoint=endpoint)

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    if tracer_provider is None:
        provider = TracerProvider()
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    else:
        tracer_provider.add_span_processor(processor)

    # graph is None when only provider configuration is wanted (e.g. the API
    # bootstrap binds the publisher into the DI container itself).
    if graph is not None:
        setup_tracing(graph, level=level, user_id_mode=user_id_mode)


def setup_phoenix(
    graph: StateGraph | None = None,
    *,
    api_key: str | None = None,
    project: str | None = None,
    endpoint: str | None = None,
    level: ObservabilityLevel = ObservabilityLevel.STANDARD,
    tracer_provider: Any = None,
    user_id_mode: UserIdMode = "raw",
) -> None:
    """Attach an Arize Phoenix OTLP HTTP exporter and instrument the graph.

    Builds an ``OTLPSpanExporter`` pointing at Phoenix's ``/v1/traces`` endpoint,
    wraps it in a ``BatchSpanProcessor``, and attaches it to either the supplied
    ``tracer_provider`` or a fresh ``TracerProvider`` set as the global provider.
    Then calls ``setup_tracing`` so ``OtelPublisher`` feeds into it.

    Must be called **before** ``graph.compile()``.

    Args:
        graph: The StateGraph to instrument.
        api_key: Phoenix API key. Falls back to ``PHOENIX_API_KEY``. Optional:
            local Phoenix needs none.
        project: Phoenix project name. Falls back to ``PHOENIX_PROJECT``, then
            ``PHOENIX_PROJECT_NAME``. Phoenix reads it from the
            ``openinference.project.name`` resource attribute, so it is applied
            only when a fresh ``TracerProvider`` is created here. A supplied
            ``tracer_provider`` has a fixed resource: set
            ``OTEL_RESOURCE_ATTRIBUTES=openinference.project.name=<name>`` before
            it is created instead.
        endpoint: Phoenix base URL. Falls back to ``PHOENIX_COLLECTOR_ENDPOINT``,
            then ``http://localhost:6006``. ``/v1/traces`` is appended unless
            present. Phoenix Cloud looks like
            ``https://app.phoenix.arize.com/s/<space>``.
        level: Observability level — SPANS, STANDARD (default), or FULL.
        tracer_provider: An existing ``TracerProvider`` to attach the processor
            to.  When ``None`` a new one is created and set as the global
            provider.
        user_id_mode: ``"raw"`` (default), ``"hash"`` or ``"mask"``: how ``user_id``
            appears on spans (see ``tenxgraph.runtime.publisher.identity``).

    Raises:
        ImportError: If ``opentelemetry-exporter-otlp-proto-http`` or
            ``opentelemetry-sdk`` are not installed.
    """
    processor = _build_phoenix_processor(api_key=api_key, endpoint=endpoint)

    from opentelemetry import trace

    if tracer_provider is None:
        provider = _new_tracer_provider(_phoenix_project(project))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    else:
        tracer_provider.add_span_processor(processor)

    # graph is None when only provider configuration is wanted (e.g. the API
    # bootstrap binds the publisher into the DI container itself).
    if graph is not None:
        setup_tracing(graph, level=level, user_id_mode=user_id_mode)


def setup_otlp(
    graph: StateGraph | None = None,
    *,
    endpoint: str | None = None,
    level: ObservabilityLevel = ObservabilityLevel.STANDARD,
    tracer_provider: Any = None,
    user_id_mode: UserIdMode = "raw",
) -> None:
    """Attach a generic OTLP HTTP exporter (collector, Tempo, Jaeger, ...) and instrument the graph.

    Auth headers are not an argument: set ``OTEL_EXPORTER_OTLP_HEADERS`` (or
    ``OTEL_EXPORTER_OTLP_TRACES_HEADERS``). Must be called **before** ``graph.compile()``.

    Args:
        graph: The StateGraph to instrument.
        endpoint: OTLP HTTP base URL; ``/v1/traces`` is appended unless present. When
            omitted, ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` must be set
            (``OTEL_EXPORTER_OTLP_ENDPOINT`` is not used).
        level: Observability level — SPANS, STANDARD (default), or FULL.
        tracer_provider: An existing ``TracerProvider`` to attach the processor to.
            When ``None`` a new one is created and set as the global provider.
        user_id_mode: ``"raw"`` (default), ``"hash"`` or ``"mask"``.

    Raises:
        ImportError: If the OTLP HTTP exporter or ``opentelemetry-sdk`` is missing.
        ValueError: If no endpoint is passed and the traces env var is unset.
    """
    processor = _build_otlp_processor(endpoint=endpoint)

    from opentelemetry import trace

    if tracer_provider is None:
        provider = _new_tracer_provider()
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    else:
        tracer_provider.add_span_processor(processor)

    # graph is None when only provider configuration is wanted (e.g. the API
    # bootstrap binds the publisher into the DI container itself).
    if graph is not None:
        setup_tracing(graph, level=level, user_id_mode=user_id_mode)


def setup_observability(  # noqa: PLR0912
    graph: StateGraph | None, config: dict[str, Any]
) -> None:
    """Unified observability entry point for declarative ``10xgraph.json`` config.

    Reads the ``observability`` block and enables Logfire, LangSmith, Phoenix
    and/or a generic OTLP endpoint, ensuring they share one ``TracerProvider`` when more than one is
    active.

    Pass ``graph=None`` to configure the providers/exporters only, without
    binding an ``OtelPublisher`` onto a graph. The API bootstrap uses this mode
    because it binds the publisher into the DI container before the graph is
    compiled (the only reliable attach point).

    Expected ``config`` shape (mirrors ``10xgraph.json``)::

        {
            "level": "standard",
            "user_id_mode": "raw",  # or "hash" / "mask"
            "logfire": {
                "enabled": True,
                "service_name": "my-agent",
                "send_to_logfire": True,
                "console": False,
            },
            "langsmith": {"enabled": True, "project": "my-project", "endpoint": null},
            "phoenix": {"enabled": True, "project": "my-project", "endpoint": null},
            "otlp": {"enabled": True, "endpoint": "http://collector:4318"},
        }

    Other keys in a backend block are ignored. For ``otlp``, a null ``endpoint``
    needs ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``; headers come from
    ``OTEL_EXPORTER_OTLP_HEADERS``.

    ``user_id_mode`` (``"raw"`` default, ``"hash"``, ``"mask"``) controls how
    ``user_id`` appears on spans; see ``tenxgraph.runtime.publisher.identity``.
    ``"hash"`` uses HMAC-SHA256 when ``TENXGRAPH_TRACE_USER_ID_KEY`` is set.

    With Logfire plus Phoenix, the Phoenix project is forwarded to
    ``logfire.configure(resource_attributes=...)`` as
    ``openinference.project.name`` when the installed logfire supports it
    (logfire>=5). Otherwise a warning is logged, spans go to Phoenix's default
    project, and ``OTEL_RESOURCE_ATTRIBUTES=openinference.project.name=<name>``
    is the fallback.

    Secrets (``LOGFIRE_TOKEN``, ``LANGSMITH_API_KEY``, ``PHOENIX_API_KEY``) must
    be in environment variables — never in the config dict.

    Args:
        graph: The StateGraph to instrument.
        config: The observability config dict.

    Raises:
        ImportError: If a requested backend's package is not installed.
        ValueError: If LangSmith has no API key, OTLP has no endpoint, or
            ``user_id_mode`` is not one of ``raw``, ``hash``, ``mask``.
    """
    user_id_mode = validate_user_id_mode(config.get("user_id_mode") or "raw")
    raw_level = config.get("level", "standard")
    try:
        level = ObservabilityLevel(raw_level)
    except ValueError:
        level = ObservabilityLevel.STANDARD

    logfire_cfg: dict[str, Any] = config.get("logfire") or {}
    langsmith_cfg: dict[str, Any] = config.get("langsmith") or {}
    phoenix_cfg: dict[str, Any] = config.get("phoenix") or {}
    otlp_cfg: dict[str, Any] = config.get("otlp") or {}

    logfire_on = bool(logfire_cfg.get("enabled", False))
    langsmith_on = bool(langsmith_cfg.get("enabled", False))
    phoenix_on = bool(phoenix_cfg.get("enabled", False))
    otlp_on = bool(otlp_cfg.get("enabled", False))

    if not (logfire_on or langsmith_on or phoenix_on or otlp_on):
        return

    ls_endpoint = langsmith_cfg.get("endpoint") or "https://api.smith.langchain.com/otel"
    logfire_kwargs: dict[str, Any] = {
        "service_name": logfire_cfg.get("service_name"),
        "send_to_logfire": bool(logfire_cfg.get("send_to_logfire", True)),
        "console": logfire_cfg.get("console"),
    }

    if logfire_on:
        _guard_logfire()

    processors: list[Any] = []
    phoenix_project: str | None = None
    if langsmith_on:
        processors.append(
            _build_langsmith_processor(project=langsmith_cfg.get("project"), endpoint=ls_endpoint)
        )
    if phoenix_on:
        processors.append(_build_phoenix_processor(endpoint=phoenix_cfg.get("endpoint")))
        phoenix_project = _phoenix_project(phoenix_cfg.get("project"))
    if otlp_on:
        processors.append(_build_otlp_processor(endpoint=otlp_cfg.get("endpoint") or None))

    if logfire_on:
        # Logfire installs the single TracerProvider and attaches the processors.
        extra: dict[str, Any] = {}
        if processors:
            extra["additional_span_processors"] = processors
        if phoenix_project:
            if _logfire_accepts_resource_attributes():
                extra["resource_attributes"] = {"openinference.project.name": phoenix_project}
            else:
                logger.warning(
                    "Phoenix project %r is ignored: logfire.configure() has no "
                    "resource_attributes before logfire 5.0. Upgrade logfire to >=5 or set "
                    "OTEL_RESOURCE_ATTRIBUTES=openinference.project.name=<name>. Spans go to "
                    "Phoenix's default project.",
                    phoenix_project,
                )
        setup_logfire(graph, level=level, user_id_mode=user_id_mode, **logfire_kwargs, **extra)
        return

    from opentelemetry import trace

    provider = _new_tracer_provider(phoenix_project)
    for processor in processors:
        provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    if graph is not None:
        setup_tracing(graph, level=level, user_id_mode=user_id_mode)
