"""GenAI semconv messages, LangSmith fields, tool span keys and unfinished spans (real SDK)."""

import json
import sys
from unittest.mock import patch

import pytest


pytest.importorskip("opentelemetry.sdk")

import opentelemetry
import opentelemetry.trace as _real_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from tenxgraph.core.graph import ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.runtime.publisher import _genai
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.otel_publisher import ObservabilityLevel, OtelPublisher
from tenxgraph.utils import CallbackManager


@pytest.fixture(autouse=True)
def _real_otel(monkeypatch):
    """test_otel_publisher swaps fake opentelemetry modules into sys.modules for good."""
    monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", _real_trace)


def _setup(level=ObservabilityLevel.FULL):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OtelPublisher(tracer=provider.get_tracer("t"), level=level), exporter


def _evt(event, etype, node="MAIN", data=None, metadata=None, ts=10.0):
    return EventModel(
        event=event,
        event_type=etype,
        node_name=node,
        run_id="r1",
        thread_id="th1",
        user_id="u1",
        timestamp=ts,
        data=data or {},
        metadata=metadata or {},
        content_type=[ContentType.MESSAGE],
    )


def _by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_input_messages_parts_format():
    msgs = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": '{"q": "x"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "found"},
    ]
    out = _genai.input_messages(msgs, "Be brief")
    assert out[0] == {"role": "system", "parts": [{"type": "text", "content": "Be brief"}]}
    assert out[1]["parts"] == [{"type": "text", "content": "Hello"}]
    assert out[2]["parts"] == [
        {"type": "tool_call", "name": "search", "arguments": {"q": "x"}, "id": "c1"}
    ]
    assert out[3]["parts"] == [{"type": "tool_call_response", "id": "c1", "response": "found"}]


def test_input_messages_keeps_existing_system_message():
    out = _genai.input_messages([{"role": "system", "content": "s"}], "other")
    assert [m["role"] for m in out] == ["system"]


def test_output_messages_parts_format():
    out = _genai.output_messages(
        {
            "output_response": "Hi",
            "output_tool_calls": [{"name": "t", "arguments": "{}"}],
            "finish_reason": "stop",
        }
    )
    assert out == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "Hi"},
                {"type": "tool_call", "name": "t", "arguments": {}},
            ],
            "finish_reason": "stop",
        }
    ]
    assert _genai.output_messages({}) == []


def test_llm_span_attributes_and_graph_node_usage():
    pub, exporter = _setup()
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.START, node=""))
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.START))
    pub._dispatch(
        _evt(
            Event.LLM_CALL,
            EventType.START,
            data={
                "model": "gpt-4o-mini",
                "provider": "openai",
                "input_messages": [{"role": "user", "content": "Hello"}],
                "system_prompt": "Be brief",
            },
        )
    )
    pub._dispatch(
        _evt(
            Event.LLM_CALL,
            EventType.END,
            data={
                "output_response": "Hi there",
                "finish_reason": "stop",
                "input_tokens": 5,
                "output_tokens": 2,
            },
        )
    )
    usage = {"usages": {"prompt_tokens": 5, "completion_tokens": 2}}
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.END, data={"messages": [usage]}))
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.END, node=""))

    llm = _by_name(exporter, "tenxgraph.llm")[0].attributes
    assert json.loads(llm["gen_ai.input.messages"])[0]["parts"][0]["content"] == "Be brief"
    assert json.loads(llm["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hi there"}],
            "finish_reason": "stop",
        }
    ]
    assert llm["gen_ai.provider.name"] == llm["gen_ai.system"] == "openai"
    assert llm["gen_ai.usage.input_tokens"] == 5
    assert llm["logfire.msg"] == llm["langsmith.trace.name"] == "chat gpt-4o-mini"
    assert llm["langsmith.span.kind"] == "llm"

    node = _by_name(exporter, "tenxgraph.node")[0].attributes
    graph = _by_name(exporter, "tenxgraph.graph")[0].attributes
    for attrs, kind, name in ((node, "chain", "MAIN"), (graph, "chain", "graph run")):
        assert attrs["langsmith.span.kind"] == kind
        assert attrs["logfire.msg"] == attrs["langsmith.trace.name"] == name
        assert attrs["langsmith.metadata.thread_id"] == "th1"
        assert attrs["langsmith.metadata.user_id"] == "u1"
        assert not any(k.startswith("gen_ai.usage.") for k in attrs)
    assert node["tenxgraph.usage.input_tokens"] == 5
    assert node["tenxgraph.usage.output_tokens"] == 2


def _tool_start(call_id, **kw):
    return _evt(
        Event.TOOL_EXECUTION,
        EventType.START,
        data={"function_name": "search", "tool_call_id": call_id, "args": {"q": call_id}},
        **kw,
    )


def test_parallel_calls_of_one_tool_get_separate_spans():
    pub, exporter = _setup()
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.START, node=""))
    pub._dispatch(_tool_start("a"))
    pub._dispatch(_tool_start("b"))
    for call_id in ("b", "a"):
        pub._dispatch(
            _evt(
                Event.TOOL_EXECUTION,
                EventType.END,
                data={"function_name": "search", "tool_call_id": call_id},
            )
        )
    tools = _by_name(exporter, "tenxgraph.tool")
    assert len(tools) == 2
    assert all(s.attributes["langsmith.span.kind"] == "tool" for s in tools)
    assert tools[0].attributes["logfire.msg"] == "tool search"
    assert not pub._registry._children


def test_tool_end_with_error_sets_error_status():
    pub, exporter = _setup()
    pub._dispatch(_tool_start("a"))
    pub._dispatch(
        _evt(
            Event.TOOL_EXECUTION,
            EventType.END,
            data={"function_name": "search", "tool_call_id": "a", "error": "boom"},
        )
    )
    span = _by_name(exporter, "tenxgraph.tool")[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"
    assert [e.name for e in span.events if e.name == "tool.error"]


def test_remote_tool_span_ends_immediately():
    pub, exporter = _setup()
    pub._dispatch(_tool_start("a", metadata={"is_remote": True}))
    span = _by_name(exporter, "tenxgraph.tool")[0]
    assert span.attributes["tenxgraph.tool.remote"] is True
    assert span.status.status_code == StatusCode.UNSET
    assert span.start_time == span.end_time
    assert not pub._registry._children


@pytest.mark.parametrize("end_type", [EventType.END, EventType.ERROR])
def test_graph_end_closes_unfinished_children(end_type):
    pub, exporter = _setup()
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, EventType.START, node="", ts=1.0))
    pub._dispatch(_evt(Event.NODE_EXECUTION, EventType.START, ts=2.0))
    pub._dispatch(_tool_start("a", ts=3.0))
    data = {"error": "x"} if end_type == EventType.ERROR else {}
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, end_type, node="", data=data, ts=9.0))
    # a second graph END is a no-op
    pub._dispatch(_evt(Event.GRAPH_EXECUTION, end_type, node="", data=data, ts=20.0))

    assert not pub._registry._children
    for name in ("tenxgraph.node", "tenxgraph.tool"):
        span = _by_name(exporter, name)[0]
        assert span.attributes["tenxgraph.span.unfinished"] is True
        assert span.status.status_code == StatusCode.UNSET
        assert span.end_time == 9_000_000_000
    assert _by_name(exporter, "tenxgraph.graph")[0].end_time == 9_000_000_000


def _calc(x: int) -> int:
    """Double x."""
    return x * 2


@pytest.mark.asyncio
async def test_tool_node_publishes_one_start_per_call():
    node = ToolNode([_calc])
    events: list[EventModel] = []
    with (
        patch(
            "tenxgraph.core.graph.tool_node.base.publish_event",
            side_effect=lambda e: events.append(e.model_copy(deep=True)),
        ),
        patch("tenxgraph.core.graph.tool_node.local_exec.publish_event"),
    ):
        for call_id in ("c1", "c2"):
            await node.invoke(
                name="_calc",
                args={"x": 1},
                tool_call_id=call_id,
                config={},
                state=AgentState(),
                callback_manager=CallbackManager(),
            )
    starts = [e for e in events if e.event_type == EventType.START]
    assert [e.data["tool_call_id"] for e in starts] == ["c1", "c2"]
    assert all(e.metadata["is_mcp"] is False for e in starts)
    assert sum(e.event_type == EventType.END for e in events) == 2


def test_extract_tool_calls_keeps_provider_call_id():
    from types import SimpleNamespace as NS

    from tenxgraph.core.graph.agent_internal.execution import _extract_tool_calls

    call = NS(id="call_9", function=NS(name="search", arguments='{"q":"x"}'))
    response = NS(choices=[NS(message=NS(tool_calls=[call]))])
    calls = _extract_tool_calls(response)
    assert calls == [{"id": "call_9", "name": "search", "arguments": '{"q":"x"}'}]
    part = _genai.output_messages({"output_tool_calls": calls})[0]["parts"][0]
    assert part["id"] == "call_9"
