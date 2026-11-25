"""Test helper: a backend publisher without its exporter."""

from tenxgraph.runtime.publisher.otel_publisher import OtelPublisher


def without_exporter(cls, **kwargs):
    """Build ``cls`` (PhoenixPublisher, LangsmithPublisher, ...) with only the OtelPublisher
    setup: its own attribute keys, but no exporter and no TracerProvider change."""
    publisher = cls.__new__(cls)
    OtelPublisher.__init__(publisher, **kwargs)
    return publisher
