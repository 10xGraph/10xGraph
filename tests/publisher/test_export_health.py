"""One warning when a trace backend is down, one info when it recovers."""

import logging
from unittest.mock import MagicMock

import pytest


pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

from tenxgraph.runtime.publisher._export_health import log_export_health


LOGGER = "tenxgraph.trace_export"


def _exporter(results):
    exporter = OTLPSpanExporter(endpoint="http://localhost:6999/v1/traces")
    exporter.export = MagicMock(side_effect=list(results))
    return log_export_health(exporter, "phoenix")


def test_one_warning_for_many_failures(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    exporter = _exporter([SpanExportResult.FAILURE] * 5)
    for _ in range(5):
        assert exporter.export([]) is SpanExportResult.FAILURE
    records = [r for r in caplog.records if r.name == LOGGER]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "http://localhost:6999/v1/traces (phoenix)" in records[0].getMessage()


def test_recovered_logged_once_and_warns_again_after(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    exporter = _exporter(
        [
            SpanExportResult.FAILURE,
            SpanExportResult.SUCCESS,
            SpanExportResult.SUCCESS,
            SpanExportResult.FAILURE,
        ]
    )
    for _ in range(4):
        exporter.export([])
    levels = [r.levelno for r in caplog.records if r.name == LOGGER]
    assert levels == [logging.WARNING, logging.INFO, logging.WARNING]


def test_success_only_is_silent(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    exporter = _exporter([SpanExportResult.SUCCESS] * 3)
    for _ in range(3):
        exporter.export([])
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_closed_port_logs_one_warning_and_no_library_noise(caplog, monkeypatch):
    # Real exporter, nothing listening: retry lines go to DEBUG, not WARNING/ERROR.
    monkeypatch.setattr("opentelemetry.exporter.otlp.common.http._MAX_RETRIES", 2, raising=False)
    caplog.set_level(logging.INFO)
    exporter = log_export_health(
        OTLPSpanExporter(endpoint="http://127.0.0.1:1/v1/traces", timeout=2), "otlp"
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    for _ in range(3):
        with provider.get_tracer("t").start_as_current_span("s"):
            pass
    noisy = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [r.name for r in noisy] == [LOGGER]


def test_exporter_without_private_attributes(caplog):
    class Bare:
        def export(self, spans):
            return SpanExportResult.FAILURE

    caplog.set_level(logging.INFO, logger=LOGGER)
    exporter = log_export_health(Bare(), "otlp")
    exporter.export([])
    exporter.export([])
    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(messages) == 1
    assert messages[0].startswith("Trace export to otlp failed")
