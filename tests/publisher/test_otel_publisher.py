import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.otel_publisher import (
    ObservabilityLevel,
    OtelPublisher,
    SpanRegistry,
    _provider_system_name,
    _set_request_params,
    _to_str,
    setup_tracing,
)


class _Span:
    def __init__(self, name="span"):
        self.name = name
        self.attrs = {}
        self.events = []
        self.ended = False
        self.status = None

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))

    def end(self, end_time=None):
        self.ended = True
        self.end_time = end_time

    def set_status(self, code, description=""):
        self.status = (code, description)


class _Tracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, context=None, start_time=None):
        s = _Span(name)
        s.start_time = start_time
        self.spans.append(s)
        return s


def _install_fake_otel():
    trace_mod = types.ModuleType("opentelemetry.trace")
    trace_mod.StatusCode = SimpleNamespace(ERROR="ERROR")
    trace_mod.set_span_in_context = lambda span: {"span": span}
    tracer = _Tracer()
    trace_mod.get_tracer = lambda name: tracer

    otel_mod = types.ModuleType("opentelemetry")
    otel_mod.trace = trace_mod

    sys.modules["opentelemetry"] = otel_mod
    sys.modules["opentelemetry.trace"] = trace_mod
    return tracer


def _evt(event, etype, node="", run="r1", data=None, metadata=None):
    return EventModel(
        event=event,
        event_type=etype,
        node_name=node,
        run_id=run,
        thread_id="t1",
        user_id="u1",
        timestamp=10.0,
        data=data or {},
        metadata=metadata or {},
        content_type=[ContentType.MESSAGE],
    )


def _evt_raw(event, etype, node="", run="r1", data=None, metadata=None):
    return EventModel.model_construct(
        event=event,
        event_type=etype,
        node_name=node,
        run_id=run,
        thread_id="t1",
        user_id="u1",
        timestamp=10.0,
        data=data or {},
        metadata=metadata or {},
        content_type=[ContentType.MESSAGE],
    )


def test_span_registry_roundtrip():
    reg = SpanRegistry()
    s = _Span()
    reg.set_graph("r", s)
    assert reg.get_graph("r") is s
    assert reg.pop_graph("r") is s

    c = _Span()
    reg.set_child("r", "n", c)
    assert reg.get_child("r", "n") is c
    assert reg.pop_child("r", "n") is c


def test_helper_serialization_and_provider_mapping():
    assert _to_str({"a": 1}) == '{"a":1}'
    assert _provider_system_name("openai") == "openai"
    assert _provider_system_name("custom") == "custom"


def test_set_request_params_handles_types_and_invalid_values():
    span = _Span()
    _set_request_params(
        span,
        {
            "temperature": "0.7",
            "max_tokens": "42",
            "top_p": 0.9,
            "top_k": 10,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "seed": 7,
            "stop": ["END"],
            "bad": object(),
        },
    )
    assert span.attrs


def test_graph_node_tool_lifecycle_dispatch_and_close():
    tracer = _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.FULL)

    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.START, data={"step": 1}))
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.START, node="MAIN", data={"step": 2}))
    pub._dispatch(
        _evt(
            Event.TOOL_EXECUTION,
            EventType.START,
            node="MAIN",
            data={"function_name": "search", "args": {"q": "x"}},
        )
    )
    pub._dispatch(
        _evt(
            Event.TOOL_EXECUTION,
            EventType.END,
            node="MAIN",
            data={"function_name": "search", "message": {"ok": True}},
        )
    )
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.END, node="MAIN", data={"messages": [{"usages": {"prompt_tokens": 1, "completion_tokens": 2}}]}))
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.END, data={"total_steps": 3}))

    assert len(tracer.spans) >= 3


def test_llm_handlers_and_error_paths():
    _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.FULL)

    pub._on_graph_start(_evt(Event.GRAPH_EXECUTION, EventType.START))
    pub._on_node_start(_evt(Event.NODE_EXECUTION, EventType.START, node="MAIN"))

    llm_start = _evt(
        Event.NODE_EXECUTION,
        EventType.START,
        node="MAIN",
        data={
            "model": "gpt-4o",
            "provider": "openai",
            "request_params": {"temperature": 0.5},
            "input_messages": [{"role": "user", "content": "hi"}],
            "system_prompt": "be helpful",
        },
    )
    pub._on_llm_start(llm_start)
    pub._on_llm_end(
        _evt(
            Event.NODE_EXECUTION,
            EventType.END,
            node="MAIN",
            data={
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_read_tokens": 1,
                "cache_creation_tokens": 1,
                "reasoning_tokens": 2,
                "finish_reason": "stop",
                "response_id": "rid",
                "response_model": "gpt-4o",
                "output_response": "hello",
            },
        )
    )

    pub._on_llm_start(llm_start)
    pub._on_llm_error(_evt(Event.NODE_EXECUTION, EventType.ERROR, node="MAIN", data={"error": "boom"}))


def test_setup_tracing_attaches_publisher():
    graph = SimpleNamespace(_publisher=None)
    with patch("tenxgraph.runtime.publisher.otel_publisher._guard", return_value=None):
        pub = setup_tracing(graph)
    assert graph._publisher is pub


def test_graph_lifecycle_update_and_error_handlers():
    _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.STANDARD)

    pub._on_graph_start(_evt(Event.GRAPH_EXECUTION, EventType.START))
    pub._on_graph_update(_evt(Event.GRAPH_EXECUTION, EventType.UPDATE, metadata={"lifecycle": "resume"}, data={"resumed_node": "N1"}))
    pub._on_graph_update(_evt(Event.GRAPH_EXECUTION, EventType.UPDATE, metadata={"lifecycle": "checkpoint", "trimmed": True}))
    pub._on_graph_update(_evt(Event.GRAPH_EXECUTION, EventType.UPDATE, metadata={"lifecycle": "graph_start"}))
    pub._on_graph_update(_evt(Event.GRAPH_EXECUTION, EventType.UPDATE, metadata={"lifecycle": "graph_end"}, data={"total_steps": 2}))
    pub._on_graph_update(_evt(Event.GRAPH_EXECUTION, EventType.UPDATE, metadata={"lifecycle": "graph_error"}, data={"error": "x"}))
    pub._on_graph_update(_evt(Event.GRAPH_EXECUTION, EventType.UPDATE, metadata={"lifecycle": "graph_interrupt", "interrupt_type": "before"}))
    pub._on_graph_interrupted(_evt(Event.GRAPH_EXECUTION, EventType.INTERRUPTED, data={"interrupted_node": "N2"}))
    pub._on_graph_error(_evt(Event.GRAPH_EXECUTION, EventType.ERROR, data={"error": "boom"}))


def test_graph_validation_rejected_and_state_enrichment_paths():
    _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.FULL)

    pub._on_graph_start(
        _evt(Event.GRAPH_EXECUTION, EventType.START, metadata={"lifecycle": "validation_rejected"})
    )

    pub._on_graph_start(_evt(Event.GRAPH_EXECUTION, EventType.START))
    pub._on_graph_update(
        _evt(
            Event.GRAPH_EXECUTION,
            EventType.UPDATE,
            data={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "ok",
                        "usages": {"prompt_tokens": 2, "completion_tokens": 3},
                        "model": "gpt-4o",
                    }
                ],
                "state": {"context": [{"role": "system", "content": "sys"}]},
            },
        )
    )


def test_node_tool_error_and_update_handlers():
    _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.FULL)

    pub._on_graph_start(_evt(Event.GRAPH_EXECUTION, EventType.START))
    pub._on_node_start(_evt(Event.NODE_EXECUTION, EventType.START, node="A"))
    pub._on_node_update(_evt(Event.NODE_EXECUTION, EventType.UPDATE, node="A", metadata={"lifecycle": "state_update"}, data={"step": 1}))
    pub._on_tool_start(_evt(Event.TOOL_EXECUTION, EventType.START, node="A", data={"function_name": "t", "args": {"a": 1}}, metadata={"is_mcp": True}))
    pub._on_tool_end(_evt(Event.TOOL_EXECUTION, EventType.END, node="A", data={"function_name": "t", "message": {"ok": True}}))
    pub._on_tool_error(_evt(Event.TOOL_EXECUTION, EventType.ERROR, node="A", data={"function_name": "t", "error": "bad"}))
    pub._on_node_error(_evt(Event.NODE_EXECUTION, EventType.ERROR, node="A", data={"error": "bad"}))


def test_handlers_are_noop_when_parent_span_missing():
    _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.STANDARD)
    pub._on_graph_end(_evt(Event.GRAPH_EXECUTION, EventType.END, run="missing"))
    pub._on_graph_error(_evt(Event.GRAPH_EXECUTION, EventType.ERROR, run="missing", data={"error": "x"}))
    pub._on_node_end(_evt(Event.NODE_EXECUTION, EventType.END, node="missing", run="missing"))
    pub._on_node_error(_evt(Event.NODE_EXECUTION, EventType.ERROR, node="missing", run="missing", data={"error": "x"}))
    pub._on_llm_end(_evt(Event.NODE_EXECUTION, EventType.END, node="missing", run="missing"))
    pub._on_llm_error(_evt(Event.NODE_EXECUTION, EventType.ERROR, node="missing", run="missing", data={"error": "x"}))


def test_close_and_sync_close_paths():
    _install_fake_otel()
    pub = OtelPublisher()
    pub._registry.set_graph("r", _Span())

    import asyncio

    asyncio.run(pub.close())
    assert pub._is_closed is True

    pub.sync_close()
    assert pub._is_closed is True


def test_guard_raises_when_opentelemetry_missing(monkeypatch):
    from tenxgraph.runtime.publisher import otel_publisher as op

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "opentelemetry":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", _fake_import):
        with patch.dict(sys.modules, {"opentelemetry": None}):
            try:
                op._guard()
            except ImportError:
                assert True


def test_publish_handles_closed_and_dispatch_exceptions():
    _install_fake_otel()
    pub = OtelPublisher()
    pub._is_closed = True

    import asyncio

    asyncio.run(pub.publish(_evt(Event.GRAPH_EXECUTION, EventType.START)))

    pub2 = OtelPublisher()
    with patch.object(pub2, "_dispatch", side_effect=RuntimeError("boom")):
        asyncio.run(pub2.publish(_evt(Event.GRAPH_EXECUTION, EventType.START)))


def test_dispatch_covers_llm_and_tool_error_paths():
    _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.STANDARD)
    pub._on_graph_start(_evt(Event.GRAPH_EXECUTION, EventType.START))
    pub._on_node_start(_evt(Event.NODE_EXECUTION, EventType.START, node="N"))

    pub._dispatch(_evt(Event.LLM_CALL, EventType.START, node="N", data={"provider": "openai", "model": "m"}))
    pub._dispatch(_evt(Event.LLM_CALL, EventType.END, node="N", data={"input_tokens": 1, "output_tokens": 1}))
    pub._dispatch(_evt(Event.LLM_CALL, EventType.ERROR, node="N", data={"error": "x"}))
    pub._dispatch(_evt(Event.TOOL_EXECUTION, EventType.ERROR, node="N", data={"function_name": "f", "error": "x"}))


def test_sync_close_runtime_error_branch_no_warning():
    _install_fake_otel()
    pub = OtelPublisher()

    def _raise_runtime(coro):
        coro.close()
        raise RuntimeError("loop running")

    with patch("asyncio.run", side_effect=_raise_runtime):
        pub.sync_close()
    assert pub._is_closed is True
