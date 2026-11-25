"""Report an unreachable OTLP trace backend once, not once per batch."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger("tenxgraph.trace_export")


class _DebugOnly(logging.LoggerAdapter):
    """Send the OTLP client's per-attempt retry and error lines to DEBUG."""

    def log(self, level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
        super().log(logging.DEBUG, msg, *args, **kwargs)


def log_export_health(exporter: Any, backend: str) -> Any:
    """Make ``exporter`` log one WARNING when exports start failing, one INFO on recovery.

    The stock OTLP exporter logs a warning per retry and an error per dropped batch,
    which floods the server log while the backend is down. After this call the first
    failed batch logs one WARNING, further failures stay silent, and the first success
    after a failure logs one INFO. The exporter's own lines move to DEBUG under
    ``tenxgraph.trace_export``.

    Returns:
        The same exporter, for chaining.
    """
    endpoint = getattr(exporter, "_endpoint", None)
    label = f"{endpoint} ({backend})" if endpoint else backend
    failing = False
    export = exporter.export

    def export_and_report(spans: Any) -> Any:
        nonlocal failing
        from opentelemetry.sdk.trace.export import SpanExportResult

        result = export(spans)
        if result is SpanExportResult.SUCCESS:
            if failing:
                failing = False
                logger.info("Trace export to %s recovered", label)
        elif not failing:
            failing = True
            logger.warning(
                "Trace export to %s failed. Spans are dropped until it is reachable; "
                "further failures are not logged.",
                label,
            )
        return result

    exporter.export = export_and_report
    # opentelemetry-exporter-otlp-proto-http 1.45.1 (the version checked) keeps the
    # logger on `_client`, taken at construction with no setter. Other versions and the
    # gRPC exporter may not have it: then the library keeps logging as it does and only
    # the reporting above applies. Older versions were not checked.
    client = getattr(exporter, "_client", None)
    if client is not None and hasattr(client, "_logger"):
        client._logger = _DebugOnly(logger, {})
    return exporter
