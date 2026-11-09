"""Private backend builders and import guards for the OTLP-based observability helpers."""

from __future__ import annotations

import inspect
import os
from typing import Any


# ── Guard helpers ─────────────────────────────────────────────────────────────


def _guard_logfire() -> None:
    try:
        import logfire  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Logfire is required for logfire tracing. Install with: pip install '10xgraph[logfire]'"
        ) from exc


def _guard_otlp_http() -> None:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,  # noqa: F401
        )
    except ImportError as exc:
        raise ImportError(
            "opentelemetry-exporter-otlp-proto-http is required for LangSmith, Phoenix and "
            "OTLP tracing. Install with: pip install '10xgraph[langsmith]', "
            "'10xgraph[phoenix]' or '10xgraph[otlp]'"
        ) from exc


def _guard_otel_sdk() -> None:
    try:
        from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "opentelemetry-sdk is required for observability. "
            "Install with: pip install '10xgraph[otel]'"
        ) from exc


def _build_langsmith_processor(
    *,
    api_key: str | None = None,
    project: str | None = None,
    endpoint: str = "https://api.smith.langchain.com/otel",
) -> Any:
    """Build a ``BatchSpanProcessor`` exporting to LangSmith's OTLP endpoint.

    Resolves the API key from ``LANGSMITH_API_KEY`` when ``api_key`` is not
    passed, sets the ``x-api-key`` and (optionally) ``Langsmith-Project``
    headers, and appends ``/v1/traces`` to ``endpoint``.

    Args:
        api_key: LangSmith API key. Falls back to ``LANGSMITH_API_KEY``.
        project: LangSmith project name, sent as the ``Langsmith-Project`` header.
        endpoint: Base OTEL endpoint URL. ``/v1/traces`` is appended automatically.

    Returns:
        A ``BatchSpanProcessor`` wrapping the LangSmith OTLP HTTP exporter.

    Raises:
        ImportError: If ``opentelemetry-sdk`` or the OTLP HTTP exporter are missing.
        ValueError: If no API key is available.
    """
    _guard_otel_sdk()
    _guard_otlp_http()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resolved_key = api_key or os.environ.get("LANGSMITH_API_KEY")
    if not resolved_key:
        raise ValueError(
            "A LangSmith API key is required. Pass api_key= or set the "
            "LANGSMITH_API_KEY environment variable."
        )

    headers: dict[str, str] = {"x-api-key": resolved_key}
    if project:
        headers["Langsmith-Project"] = project

    # The LangSmith OTLP endpoint expects /v1/traces
    traces_endpoint = endpoint.rstrip("/") + "/v1/traces"
    exporter = OTLPSpanExporter(endpoint=traces_endpoint, headers=headers)
    return BatchSpanProcessor(exporter)


def _phoenix_project(project: str | None = None) -> str | None:
    """Resolve the Phoenix project: arg, then ``PHOENIX_PROJECT``, then ``PHOENIX_PROJECT_NAME``."""
    return (
        project
        or os.environ.get("PHOENIX_PROJECT")
        or os.environ.get("PHOENIX_PROJECT_NAME")
        or None
    )


def _logfire_accepts_resource_attributes() -> bool:
    """True when the installed ``logfire.configure`` takes ``resource_attributes`` (logfire>=5)."""
    import logfire

    return "resource_attributes" in inspect.signature(logfire.configure).parameters


def _new_tracer_provider(project: str | None = None) -> Any:
    """Create a ``TracerProvider``; a Phoenix ``project`` becomes a resource attribute."""
    from opentelemetry.sdk.trace import TracerProvider

    if not project:
        return TracerProvider()

    from opentelemetry.sdk.resources import Resource

    return TracerProvider(resource=Resource.create({"openinference.project.name": project}))


def _build_phoenix_processor(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> Any:
    """Build a ``BatchSpanProcessor`` exporting to Arize Phoenix's OTLP endpoint.

    Resolves the endpoint from ``endpoint``, then ``PHOENIX_COLLECTOR_ENDPOINT``,
    then ``http://localhost:6006``, and appends ``/v1/traces`` unless present. The
    API key (``api_key`` or ``PHOENIX_API_KEY``) is optional: local Phoenix has no
    auth, so the ``authorization: Bearer`` header is only sent when a key exists.

    Args:
        api_key: Phoenix API key. Falls back to ``PHOENIX_API_KEY``.
        endpoint: Phoenix base URL. Falls back to ``PHOENIX_COLLECTOR_ENDPOINT``.

    Returns:
        A ``BatchSpanProcessor`` wrapping the Phoenix OTLP HTTP exporter.

    Raises:
        ImportError: If ``opentelemetry-sdk`` or the OTLP HTTP exporter are missing.
    """
    _guard_otel_sdk()
    _guard_otlp_http()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resolved_key = api_key or os.environ.get("PHOENIX_API_KEY")
    headers: dict[str, str] = {}
    if resolved_key:
        headers["authorization"] = f"Bearer {resolved_key}"

    base = (
        endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or "http://localhost:6006"
    ).rstrip("/")
    traces_endpoint = base if base.endswith("/v1/traces") else base + "/v1/traces"
    exporter = OTLPSpanExporter(endpoint=traces_endpoint, headers=headers)
    return BatchSpanProcessor(exporter)


def _build_otlp_processor(*, endpoint: str | None = None) -> Any:
    """Build a ``BatchSpanProcessor`` exporting to any OTLP HTTP traces endpoint.

    With ``endpoint``, trailing ``/`` is stripped and ``/v1/traces`` appended unless
    present. Without it, ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` must be set and the
    exporter reads it. ``OTEL_EXPORTER_OTLP_ENDPOINT`` is deliberately not used: the
    legacy API path points it at gRPC (4317), the wrong port for this HTTP exporter.

    Auth headers are not an argument: the exporter reads the standard
    ``OTEL_EXPORTER_OTLP_HEADERS`` / ``OTEL_EXPORTER_OTLP_TRACES_HEADERS`` variables.

    Args:
        endpoint: OTLP HTTP base URL, e.g. ``http://collector:4318``.

    Returns:
        A ``BatchSpanProcessor`` wrapping the OTLP HTTP exporter.

    Raises:
        ImportError: If ``opentelemetry-sdk`` or the OTLP HTTP exporter are missing.
        ValueError: If no endpoint is passed and the traces env var is unset.
    """
    _guard_otel_sdk()
    _guard_otlp_http()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if endpoint:
        base = endpoint.rstrip("/")
        endpoint = base if base.endswith("/v1/traces") else base + "/v1/traces"
    elif not os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        raise ValueError(
            "OTLP tracing needs an endpoint: pass endpoint= or set "
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        )
    return BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
