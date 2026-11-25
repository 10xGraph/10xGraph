"""Each trace backend's attribute keys appear only for that backend (real OTEL SDK spans)."""

import json
import sys
from unittest.mock import patch

import pytest


pytest.importorskip("opentelemetry.sdk")

import opentelemetry
import opentelemetry.trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tenxgraph.runtime.publisher.backends_publisher import BackendsPublisher, enabled_backends
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.identity import transform_user_id
from tenxgraph.runtime.publisher.langsmith_publisher import LangsmithPublisher
from tenxgraph.runtime.publisher.logfire_publisher import LogfirePublisher
from tenxgraph.runtime.publisher.otel_publisher import ObservabilityLevel, OtelPublisher
from tenxgraph.runtime.publisher.phoenix_publisher import PhoenixPublisher


OI_KEYS = ("openinference.", "input.", "output.", "llm.", "session.id", "user.id", "metadata")
LANGSMITH_KEYS = ("langsmith.",)
LOGFIRE_KEYS = ("logfire.",)


@pytest.fixture(autouse=True)
def _real_otel(monkeypatch):
    """Other tests in this folder put a fake ``opentelemetry`` in sys.modules; use the real one."""
    monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", opentelemetry.trace)


def _evt(event, etype, node="MAIN", data=None, metadata=None):
    return EventModel(
        event=event,
        event_type=etype,
        node_name=node,
        run_id="r1",
        thread_id="t1",
        user_id="alice42",
        timestamp=10.0,
        data=data or {},
        metadata=metadata or {},
        content_type=[ContentType.MESSAGE],
    )


_STATE = {"context": [{"role": "user", "content": "hi"}]}
_TOOL = {"function_name": "search", "tool_call_id": "c1", "args": {"q": 1}}
_LLM_END = {"output_response": "yo", "input_tokens": 3, "output_tokens": 2}


def _run(pub: OtelPublisher, exporter: InMemorySpanExporter) -> dict[str, dict]:
    """One run with a graph, node, LLM and tool span; returns span attributes by name."""
    for e in (
        _evt(Event.GRAPH_EXECUTION, EventType.START, node="", data={"state": _STATE}),
        _evt(Event.NODE_EXECUTION, EventType.START, data={"state": _STATE}),
        _evt(
            Event.LLM_CALL,
            EventType.START,
            data={"model": "m1", "provider": "openai", "input_messages": _STATE["context"]},
        ),
        _evt(Event.LLM_CALL, EventType.END, data=_LLM_END),
        _evt(Event.NODE_EXECUTION, EventType.END, data={"state": _STATE, "messages": []}),
        _evt(Event.TOOL_EXECUTION, EventType.START, node="search", data=_TOOL),
        _evt(Event.TOOL_EXECUTION, EventType.END, node="search", data={**_TOOL, "message": "ok"}),
        _evt(Event.GRAPH_EXECUTION, EventType.END, node="", data={"messages": []}),
    ):
        pub._dispatch(e)
    return {s.name: dict(s.attributes) for s in exporter.get_finished_spans()}


def _provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _backends(names, level=ObservabilityLevel.FULL, user_id_mode="raw"):
    provider, exporter = _provider()
    pub = BackendsPublisher(
        names, tracer=provider.get_tracer("t"), level=level, user_id_mode=user_id_mode
    )
    return _run(pub, exporter)


def _has(spans, prefixes) -> bool:
    return any(k.startswith(prefixes) for attrs in spans.values() for k in attrs)


def _assert_basic(spans):
    assert set(spans) == {"invoke_agent 10xgraph", "MAIN", "chat m1", "execute_tool search"}
    assert spans["invoke_agent 10xgraph"]["tenxgraph.graph.thread_id"] == "t1"
    assert spans["chat m1"]["gen_ai.request.model"] == "m1"
    assert spans["chat m1"]["gen_ai.usage.input_tokens"] == 3
    assert spans["execute_tool search"]["gen_ai.tool.name"] == "search"


def test_plain_otel_publisher_writes_no_backend_keys():
    provider, exporter = _provider()
    spans = _run(OtelPublisher(tracer=provider.get_tracer("t")), exporter)
    _assert_basic(spans)
    assert not _has(spans, OI_KEYS + LANGSMITH_KEYS + LOGFIRE_KEYS)


def test_otlp_backend_is_basic_only():
    spans = _backends(["otlp"])
    _assert_basic(spans)
    assert not _has(spans, OI_KEYS + LANGSMITH_KEYS + LOGFIRE_KEYS)


def test_phoenix_publisher_writes_openinference_only():
    provider, exporter = _provider()
    with patch(
        "tenxgraph.runtime.publisher.phoenix_publisher._build_phoenix_processor",
        return_value=SimpleSpanProcessor(InMemorySpanExporter()),
    ):
        pub = PhoenixPublisher(tracer_provider=provider)
    spans = _run(pub, exporter)
    _assert_basic(spans)
    assert not _has(spans, LANGSMITH_KEYS + LOGFIRE_KEYS)
    for attrs in spans.values():
        assert attrs["session.id"] == "t1"
        assert attrs["user.id"] == "alice42"
        assert json.loads(attrs["metadata"]) == {"thread_id": "t1", "run_id": "r1"}
    graph, llm = spans["invoke_agent 10xgraph"], spans["chat m1"]
    assert graph["openinference.span.kind"] == "CHAIN"
    assert graph["input.value"] == "hi"
    assert llm["openinference.span.kind"] == "LLM"
    assert llm["llm.model_name"] == "m1" and llm["llm.provider"] == "openai"
    assert llm["llm.token_count.total"] == 5
    assert llm["llm.input_messages.0.message.content"] == "hi"
    assert llm["output.value"] == "yo"
    assert json.loads(spans["MAIN"]["input.value"]) == _STATE
    tool = spans["execute_tool search"]
    assert tool["openinference.span.kind"] == "TOOL"
    assert tool["input.value"] == '{"q":1}' and tool["output.value"] == "ok"


def test_langsmith_publisher_writes_langsmith_only():
    provider, exporter = _provider()
    with patch(
        "tenxgraph.runtime.publisher.langsmith_publisher._build_langsmith_processor",
        return_value=SimpleSpanProcessor(InMemorySpanExporter()),
    ):
        pub = LangsmithPublisher(tracer_provider=provider)
    spans = _run(pub, exporter)
    _assert_basic(spans)
    assert not _has(spans, OI_KEYS + LOGFIRE_KEYS)
    kinds = {"invoke_agent 10xgraph": "chain", "MAIN": "chain"}
    kinds |= {"chat m1": "llm", "execute_tool search": "tool"}
    for name, attrs in spans.items():
        assert attrs["langsmith.span.kind"] == kinds[name]
        assert attrs["langsmith.trace.name"] == name
        assert attrs["langsmith.metadata.thread_id"] == "t1"
        assert attrs["langsmith.metadata.user_id"] == "alice42"


def test_logfire_publisher_writes_logfire_only():
    provider, exporter = _provider()
    with patch("logfire.configure"):
        pub = LogfirePublisher()
    pub._tracer = provider.get_tracer("t")
    spans = _run(pub, exporter)
    _assert_basic(spans)
    assert not _has(spans, OI_KEYS + LANGSMITH_KEYS)
    for name, attrs in spans.items():
        assert attrs["logfire.msg"] == name
    assert [k for a in spans.values() for k in a if k.startswith("logfire.")] == ["logfire.msg"] * 4


def test_several_backends_write_each_dialect_on_one_span():
    spans = _backends(["langsmith", "phoenix"])
    assert not _has(spans, LOGFIRE_KEYS)
    llm = spans["chat m1"]
    assert llm["openinference.span.kind"] == "LLM" and llm["langsmith.span.kind"] == "llm"
    assert llm["session.id"] == llm["langsmith.metadata.thread_id"] == "t1"


@pytest.mark.parametrize("mode", ["hash", "mask"])
def test_user_id_mode_applies_to_every_backend_key(mode):
    spans = _backends(["phoenix", "langsmith"], user_id_mode=mode)
    expected = transform_user_id("alice42", mode)
    assert expected != "alice42"
    for attrs in spans.values():
        assert attrs["user.id"] == attrs["langsmith.metadata.user_id"] == expected
        assert "alice42" not in json.dumps(attrs)


def test_levels_hide_content_in_every_backend():
    spans_level = _backends(["phoenix", "langsmith", "logfire"], level=ObservabilityLevel.SPANS)
    assert not _has(spans_level, ("input.", "output.", "llm."))
    assert spans_level["chat m1"]["openinference.span.kind"] == "LLM"
    standard = _backends(["phoenix", "langsmith", "logfire"], level=ObservabilityLevel.STANDARD)
    llm = standard["chat m1"]
    assert llm["llm.token_count.total"] == 5
    assert not any(k.startswith(("input.", "output.", "llm.input_", "llm.output_")) for k in llm)
    assert "input.value" not in standard["invoke_agent 10xgraph"]
    assert "input.value" not in standard["execute_tool search"]
    assert "input.value" in standard["MAIN"]  # node state is STANDARD


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="nope"):
        BackendsPublisher(["phoenix", "nope"])


def test_enabled_backends_reads_the_config_block():
    cfg = {"phoenix": {"enabled": True}, "langsmith": {"enabled": False}, "otlp": {"enabled": 1}}
    assert enabled_backends(cfg) == ["phoenix", "otlp"]
    assert enabled_backends({}) == []
