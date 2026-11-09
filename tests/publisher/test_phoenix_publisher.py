"""Unit tests for tenxgraph.runtime.publisher.langsmith_publisher.PhoenixPublisher.

All external dependencies (opentelemetry) are fully mocked so the tests run
without any optional extras installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _fake_otel_modules():
    """Minimal opentelemetry stubs required by PhoenixPublisher.__init__."""
    provider_instance = MagicMock(name="provider_instance")

    otel = types.ModuleType("opentelemetry")
    trace_mod = types.ModuleType("opentelemetry.trace")
    trace_mod.set_tracer_provider = MagicMock()
    otel.trace = trace_mod

    sdk = types.ModuleType("opentelemetry.sdk")
    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace.TracerProvider = MagicMock(name="TracerProvider", return_value=provider_instance)
    sdk.trace = sdk_trace
    sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
    sdk_resources.Resource = MagicMock(name="Resource")
    sdk.resources = sdk_resources
    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_trace_export.BatchSpanProcessor = MagicMock(name="BatchSpanProcessor")
    sdk_trace.export = sdk_trace_export

    exporter_root = types.ModuleType("opentelemetry.exporter")
    exporter_otlp = types.ModuleType("opentelemetry.exporter.otlp")
    exporter_proto = types.ModuleType("opentelemetry.exporter.otlp.proto")
    exporter_http = types.ModuleType("opentelemetry.exporter.otlp.proto.http")
    exporter_trace = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    exporter_trace.OTLPSpanExporter = MagicMock(name="OTLPSpanExporter")

    mods = {
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
    }
    return mods, provider_instance, trace_mod, exporter_trace, sdk_trace, sdk_resources


def _make(**kwargs):
    """Build a PhoenixPublisher under fake otel modules; return it plus the fakes."""
    mods, prov, trace_mod, exporter_trace, sdk_trace, sdk_resources = _fake_otel_modules()
    env = kwargs.pop("env", {})
    with patch.dict(sys.modules, mods), patch.dict("os.environ", env, clear=True):
        from tenxgraph.runtime.publisher.phoenix_publisher import PhoenixPublisher

        pub = PhoenixPublisher(**kwargs)
    return pub, prov, trace_mod, exporter_trace, sdk_trace, sdk_resources


class TestPhoenixPublisher:
    """Tests for PhoenixPublisher."""

    def test_is_subclass_of_otel_publisher(self):
        from tenxgraph.runtime.publisher.otel_publisher import OtelPublisher
        from tenxgraph.runtime.publisher.phoenix_publisher import PhoenixPublisher

        assert issubclass(PhoenixPublisher, OtelPublisher)

    def test_no_key_needed_and_no_auth_header(self):
        _pub, _p, _t, exp, *_ = _make()
        assert "authorization" not in exp.OTLPSpanExporter.call_args[1]["headers"]

    def test_env_var_key_used(self):
        _pub, _p, _t, exp, *_ = _make(env={"PHOENIX_API_KEY": "env-key"})
        headers = exp.OTLPSpanExporter.call_args[1]["headers"]
        assert headers["authorization"] == "Bearer env-key"

    def test_explicit_api_key_overrides_env(self):
        _pub, _p, _t, exp, *_ = _make(api_key="explicit", env={"PHOENIX_API_KEY": "env-key"})
        headers = exp.OTLPSpanExporter.call_args[1]["headers"]
        assert headers["authorization"] == "Bearer explicit"

    def test_default_endpoint(self):
        _pub, _p, _t, exp, *_ = _make()
        assert exp.OTLPSpanExporter.call_args[1]["endpoint"] == "http://localhost:6006/v1/traces"

    def test_env_endpoint_used(self):
        env = {"PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/s/space/"}
        _pub, _p, _t, exp, *_ = _make(env=env)
        endpoint = exp.OTLPSpanExporter.call_args[1]["endpoint"]
        assert endpoint == "https://app.phoenix.arize.com/s/space/v1/traces"

    def test_explicit_endpoint_beats_env(self):
        env = {"PHOENIX_COLLECTOR_ENDPOINT": "http://env:6006"}
        _pub, _p, _t, exp, *_ = _make(endpoint="http://arg:6006", env=env)
        assert exp.OTLPSpanExporter.call_args[1]["endpoint"] == "http://arg:6006/v1/traces"

    def test_traces_suffix_not_doubled(self):
        _pub, _p, _t, exp, *_ = _make(endpoint="http://host:6006/v1/traces")
        assert exp.OTLPSpanExporter.call_args[1]["endpoint"] == "http://host:6006/v1/traces"

    def test_project_sets_resource_attribute(self):
        _pub, prov, _t, _e, sdk_trace, sdk_resources = _make(project="proj")
        sdk_resources.Resource.create.assert_called_once_with(
            {"openinference.project.name": "proj"}
        )
        resource = sdk_resources.Resource.create.return_value
        sdk_trace.TracerProvider.assert_called_once_with(resource=resource)

    def test_project_env_precedence(self):
        env = {"PHOENIX_PROJECT_NAME": "name", "PHOENIX_PROJECT": "winner"}
        *_, sdk_resources = _make(env=env)
        sdk_resources.Resource.create.assert_called_once_with(
            {"openinference.project.name": "winner"}
        )

    def test_project_name_env_used_alone(self):
        *_, sdk_resources = _make(env={"PHOENIX_PROJECT_NAME": "name"})
        sdk_resources.Resource.create.assert_called_once_with(
            {"openinference.project.name": "name"}
        )

    def test_no_project_no_resource(self):
        _pub, _p, _t, _e, sdk_trace, sdk_resources = _make()
        sdk_resources.Resource.create.assert_not_called()
        sdk_trace.TracerProvider.assert_called_once_with()

    def test_new_provider_created_and_set_global_when_none(self):
        _pub, prov, trace_mod, *_ = _make()
        prov.add_span_processor.assert_called_once()
        trace_mod.set_tracer_provider.assert_called_once_with(prov)

    def test_existing_provider_reused(self):
        existing = MagicMock(name="existing_provider")
        pub, _p, trace_mod, *_ = _make(tracer_provider=existing, project="ignored")
        existing.add_span_processor.assert_called_once()
        trace_mod.set_tracer_provider.assert_not_called()
        assert pub._tracer_arg is existing.get_tracer.return_value

    def test_level_stored_on_publisher(self):
        from tenxgraph.runtime.publisher.otel_publisher import ObservabilityLevel

        pub, *_ = _make(level=ObservabilityLevel.FULL)
        assert pub._level == ObservabilityLevel.FULL

    def test_default_level_is_standard(self):
        from tenxgraph.runtime.publisher.otel_publisher import ObservabilityLevel

        pub, *_ = _make()
        assert pub._level == ObservabilityLevel.STANDARD

    def test_raises_import_error_when_otlp_exporter_missing(self):
        mods, *_ = _fake_otel_modules()
        mods["opentelemetry.exporter.otlp.proto.http.trace_exporter"] = None
        with patch.dict(sys.modules, mods):
            from tenxgraph.runtime.publisher.phoenix_publisher import PhoenixPublisher

            with pytest.raises(ImportError, match="opentelemetry-exporter-otlp-proto-http"):
                PhoenixPublisher()

    def test_raises_import_error_when_otel_sdk_missing(self):
        mods, *_ = _fake_otel_modules()
        mods["opentelemetry.sdk.trace"] = None
        with patch.dict(sys.modules, mods):
            from tenxgraph.runtime.publisher.phoenix_publisher import PhoenixPublisher

            with pytest.raises(ImportError, match="opentelemetry-sdk"):
                PhoenixPublisher()
