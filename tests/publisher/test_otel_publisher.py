import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.otel_publisher import (
    ObservabilityLevel,
    OtelPublisher,
    SpanRegistry,
    _provider_system_name,
    _set_request_params,
    _to_str,
)
from tenxgraph.runtime.publisher.phoenix_publisher import PhoenixPublisher

from ._publishers import without_exporter


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
    trace_mod.StatusCode = SimpleNamespace(ERROR="ERROR", OK="OK")
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


def test_default_level_is_full():
    assert OtelPublisher()._level == ObservabilityLevel.FULL


def test_resume_event_then_run_interrupted_closes_the_graph_span():
    tracer = _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.STANDARD)

    pub._on_graph_start(_evt(Event.GRAPH_EXECUTION, EventType.START))
    pub._on_node_start(_evt(Event.NODE_EXECUTION, EventType.START, node="N2"))
    resume = {"metadata": {"lifecycle": "resume"}, "data": {"resumed_node": "N1"}}
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.UPDATE, **resume))
    pub._dispatch(
        _evt(
            Event.GRAPH_EXECUTION,
            EventType.INTERRUPTED,
            data={"interrupted_node": "N2", "interrupt_type": "before"},
            metadata={"event_name": "run.interrupted"},
        )
    )
    graph, node = tracer.spans
    assert ("graph.resumed", {"resumed_node": "N1"}) in graph.events
    assert graph.ended and graph.status == ("OK", "")
    assert graph.attrs["tenxgraph.graph.run_status"] == "interrupted"
    assert [n for n, _ in graph.events][-1] == "graph.interrupted"
    assert node.ended and node.attrs["tenxgraph.span.unfinished"] is True
    # A later error for the same run finds no span: nothing is ended twice.
    pub._on_graph_error(_evt(Event.GRAPH_EXECUTION, EventType.ERROR, data={"error": "boom"}))
    assert len(tracer.spans) == 2


def test_input_rejected_is_a_failed_span_of_its_own():
    tracer = _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.FULL)
    pub._dispatch(
        _evt(
            Event.GRAPH_EXECUTION,
            EventType.ERROR,
            data={"error": "bad input", "rejected_count": 1},
            metadata={"lifecycle": "validation_rejected", "event_name": "input.rejected"},
        )
    )
    (span,) = tracer.spans
    assert span.ended
    assert span.status == ("ERROR", "input_rejected")
    assert span.attrs["tenxgraph.graph.run_status"] == "input_rejected"
    assert pub._registry._graph == {}


def test_run_completed_puts_run_totals_on_the_graph_span():
    tracer = _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.FULL)
    pub._on_graph_start(_evt(Event.GRAPH_EXECUTION, EventType.START))
    pub._on_llm_start(_evt(Event.LLM_CALL, EventType.START, data={"model": "gpt-4o"}))
    llm_end = {"model": "gpt-4o", "response_model": "gpt-4o-2024"}
    pub._on_llm_end(_evt(Event.LLM_CALL, EventType.END, data=llm_end))
    pub._on_graph_end(
        _evt(
            Event.GRAPH_EXECUTION,
            EventType.END,
            data={
                "messages": [{"role": "assistant", "content": "ok"}],
                "state": {"context": [{"role": "system", "content": "sys"}]},
            },
            metadata={"total_input_tokens": 2, "total_output_tokens": 3},
        )
    )
    graph = tracer.spans[0]
    assert graph.attrs["tenxgraph.graph.model"] == "gpt-4o-2024"
    assert graph.attrs["tenxgraph.usage.input_tokens"] == 2
    assert graph.attrs["tenxgraph.usage.output_tokens"] == 3
    assert graph.attrs["tenxgraph.graph.run_status"] == "completed"
    names = [n for n, _ in graph.events]
    assert "gen_ai.content.completion" in names and "gen_ai.content.system" in names


def test_run_recovered_before_start_is_added_when_the_span_starts():
    tracer = _install_fake_otel()
    pub = OtelPublisher()
    pub._dispatch(
        _evt(
            Event.GRAPH_EXECUTION,
            EventType.INTERRUPTED,
            data={"dead_run_id": "old"},
            metadata={"event_name": "run.recovered"},
        )
    )
    assert tracer.spans == []
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.START))
    assert ("graph.recovered", {"dead_run_id": "old"}) in tracer.spans[0].events
    assert pub._registry._recovered == {}


def test_node_tool_and_error_handlers():
    _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.FULL)

    pub._on_graph_start(_evt(Event.GRAPH_EXECUTION, EventType.START))
    pub._on_node_start(_evt(Event.NODE_EXECUTION, EventType.START, node="A"))
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


def _node_spans(level, start_data, end_data):
    import json

    tracer = _install_fake_otel()
    pub = without_exporter(PhoenixPublisher, level=level)
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.START, data={"step": 1}))
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.START, node="MAIN", data=start_data))
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.END, node="MAIN", data=end_data))
    node = next(s for s in tracer.spans if s.name == "MAIN")
    return node, json


_STATE_IN = {"context": [{"role": "user", "content": "hi"}], "execution_meta": {"step": 0}}
_STATE_OUT = {"context": [{"role": "assistant", "content": "yo"}], "execution_meta": {"step": 1}}
_END = {
    "state": _STATE_OUT,
    "messages": [{"role": "assistant", "content": "yo"}],
    "next_node": "TOOL",
}


@pytest.mark.parametrize("level", [ObservabilityLevel.STANDARD, ObservabilityLevel.FULL])
def test_node_span_carries_state_in_and_out(level):
    node, json = _node_spans(level, {"step": 2, "state": _STATE_IN}, _END)
    assert json.loads(node.attrs["input.value"]) == _STATE_IN
    assert node.attrs["input.mime_type"] == "application/json"
    out = json.loads(node.attrs["output.value"])
    assert out == {
        "state": _STATE_OUT,
        "messages": _END["messages"],
        "next_node": "TOOL",
    }
    assert node.attrs["output.mime_type"] == "application/json"
    assert isinstance(node.attrs["input.value"], str)


def test_node_output_omits_absent_keys():
    node, json = _node_spans(
        ObservabilityLevel.STANDARD, {"state": _STATE_IN}, {"state": _STATE_OUT, "next_node": None}
    )
    assert json.loads(node.attrs["output.value"]) == {"state": _STATE_OUT}


def test_plain_publisher_node_span_has_no_state():
    tracer = _install_fake_otel()
    pub = OtelPublisher(level=ObservabilityLevel.FULL)
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.START, node="MAIN", data={"state": _STATE_IN}))
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.END, node="MAIN", data=_END))
    node = next(s for s in tracer.spans if s.name == "MAIN")
    assert not any(k.startswith(("input.", "output.", "openinference.")) for k in node.attrs)


def test_node_span_spans_level_has_no_state():
    node, _ = _node_spans(ObservabilityLevel.SPANS, {"state": _STATE_IN}, _END)
    assert not any(k.startswith(("input.", "output.")) for k in node.attrs)


@pytest.mark.parametrize("bad", ["text", None, 5, [1]])
def test_node_span_non_dict_state_is_skipped(bad):
    node, _ = _node_spans(ObservabilityLevel.STANDARD, {"state": bad}, {"state": bad})
    assert "input.value" not in node.attrs
    assert "output.value" not in node.attrs
