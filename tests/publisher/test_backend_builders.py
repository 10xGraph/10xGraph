"""Unit tests for the private builders and import guards in
tenxgraph.runtime.publisher._backends.

All external dependencies (logfire, opentelemetry SDK / OTLP exporter) are
fully mocked so the tests run without any optional extras installed. The
LangSmith and Phoenix builders are covered through their publishers
(test_langsmith_publisher.py, test_phoenix_publisher.py).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fake_otel_modules():
    """Return a dict of fake opentelemetry modules suitable for sys.modules patching."""
    # opentelemetry (top-level)
    otel = types.ModuleType("opentelemetry")

    # opentelemetry.trace
    trace_mod = types.ModuleType("opentelemetry.trace")
    _mock_provider = MagicMock()
    trace_mod.set_tracer_provider = MagicMock()
    trace_mod.get_tracer_provider = MagicMock(return_value=_mock_provider)
    otel.trace = trace_mod

    # opentelemetry.sdk
    sdk = types.ModuleType("opentelemetry.sdk")
    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")

    FakeTracerProvider = MagicMock(name="TracerProvider")
    sdk_trace.TracerProvider = FakeTracerProvider
    sdk.trace = sdk_trace
    sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
    FakeResource = MagicMock(name="Resource")
    sdk_resources.Resource = FakeResource
    sdk.resources = sdk_resources

    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    FakeBatchSpanProcessor = MagicMock(name="BatchSpanProcessor")
    sdk_trace_export.BatchSpanProcessor = FakeBatchSpanProcessor
    sdk_trace.export = sdk_trace_export

    # opentelemetry.exporter.otlp.proto.http.trace_exporter
    exporter_root = types.ModuleType("opentelemetry.exporter")
    exporter_otlp = types.ModuleType("opentelemetry.exporter.otlp")
    exporter_proto = types.ModuleType("opentelemetry.exporter.otlp.proto")
    exporter_http = types.ModuleType("opentelemetry.exporter.otlp.proto.http")
    exporter_trace = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    FakeOTLPExporter = MagicMock(name="OTLPSpanExporter")
    exporter_trace.OTLPSpanExporter = FakeOTLPExporter

    return {
        "opentelemetry": otel,
        "opentelemetry.trace": trace_mod,
        "opentelemetry.sdk": sdk,
        "opentelemetry.sdk.trace": sdk_trace,
        "opentelemetry.sdk.resources": sdk_resources,
        "opentelemetry.sdk.trace.export": sdk_trace_export,
        "opentelemetry.exporter": exporter_root,
        "opentelemetry.exporter.otlp": exporter_otlp,
        "opentelemetry.exporter.otlp.proto": exporter_proto,
        "opentelemetry.exporter.otlp.proto.http": exporter_http,
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": exporter_trace,
        # Classes for direct import
        "_fake_TracerProvider": FakeTracerProvider,
        "_fake_BatchSpanProcessor": FakeBatchSpanProcessor,
        "_fake_OTLPSpanExporter": FakeOTLPExporter,
        "_fake_trace_mod": trace_mod,
        "_fake_Resource": FakeResource,
    }


def _fake_logfire_module():
    lf = types.ModuleType("logfire")
    lf.configure = MagicMock()
    return lf


# ── Guard tests ────────────────────────────────────────────────────────────────


class TestGuardLogfire:
    def test_raises_when_logfire_missing(self):
        # A None entry in sys.modules makes `import logfire` raise ImportError
        with patch.dict(sys.modules, {"logfire": None}):
            from tenxgraph.runtime.publisher._backends import _guard_logfire

            with pytest.raises(ImportError, match="logfire"):
                _guard_logfire()

    def test_passes_when_logfire_present(self):
        fake_lf = _fake_logfire_module()
        with patch.dict(sys.modules, {"logfire": fake_lf}):
            from tenxgraph.runtime.publisher._backends import _guard_logfire

            _guard_logfire()  # should not raise


class TestGuardOtlpHttp:
    def test_raises_when_package_missing(self):
        key = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        saved = sys.modules.pop(key, ...)
        try:
            sys.modules[key] = None  # type: ignore[assignment]
            from tenxgraph.runtime.publisher._backends import _guard_otlp_http

            with pytest.raises(ImportError, match="opentelemetry-exporter-otlp-proto-http"):
                _guard_otlp_http()
        finally:
            if saved is not ...:
                sys.modules[key] = saved
            else:
                sys.modules.pop(key, None)

    def test_passes_when_package_present(self):
        mods = _fake_otel_modules()
        with patch.dict(sys.modules, mods):
            from tenxgraph.runtime.publisher._backends import _guard_otlp_http

            _guard_otlp_http()  # should not raise


class TestGuardOtelSdk:
    def test_raises_when_sdk_missing(self):
        saved = sys.modules.pop("opentelemetry.sdk.trace", ...)
        try:
            sys.modules["opentelemetry.sdk.trace"] = None  # type: ignore[assignment]
            from tenxgraph.runtime.publisher._backends import _guard_otel_sdk

            with pytest.raises(ImportError, match="opentelemetry-sdk"):
                _guard_otel_sdk()
        finally:
            if saved is not ...:
                sys.modules["opentelemetry.sdk.trace"] = saved
            else:
                sys.modules.pop("opentelemetry.sdk.trace", None)

    def test_passes_when_sdk_present(self):
        mods = _fake_otel_modules()
        with patch.dict(sys.modules, mods):
            from tenxgraph.runtime.publisher._backends import _guard_otel_sdk

            _guard_otel_sdk()  # should not raise


class TestOtlp:
    def test_build_processor_appends_traces_path(self):
        from tenxgraph.runtime.publisher._backends import _build_otlp_processor

        fake = _fake_otel_modules()
        exp = fake["_fake_OTLPSpanExporter"]
        with patch.dict(sys.modules, fake):
            _build_otlp_processor(endpoint="http://collector:4318/")
            _build_otlp_processor(endpoint="http://collector:4318/v1/traces")
        assert [c.kwargs["endpoint"] for c in exp.call_args_list] == [
            "http://collector:4318/v1/traces",
            "http://collector:4318/v1/traces",
        ]

    def test_build_processor_requires_endpoint(self, monkeypatch):
        from tenxgraph.runtime.publisher._backends import _build_otlp_processor

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
        with (
            patch.dict(sys.modules, _fake_otel_modules()),
            pytest.raises(ValueError, match="OTLP tracing needs an endpoint"),
        ):
            _build_otlp_processor()

    def test_build_processor_reads_traces_env(self, monkeypatch):
        from tenxgraph.runtime.publisher._backends import _build_otlp_processor

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://t:4318/v1/traces")
        fake = _fake_otel_modules()
        exp = fake["_fake_OTLPSpanExporter"]
        with patch.dict(sys.modules, fake):
            _build_otlp_processor()
        assert exp.call_args.kwargs["endpoint"] is None
