"""Span tree and attribute limits of OtelPublisher on real OTEL SDK spans (issue #357)."""

import sys

import pytest


pytest.importorskip("opentelemetry.sdk")

import opentelemetry
import opentelemetry.trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tenxgraph.runtime.publisher import _openinference as oi
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.otel_publisher import (
    OtelPublisher,
    _input_message_budget,
)
from tenxgraph.utils.constants import START


@pytest.fixture(autouse=True)
def _real_otel(monkeypatch):
    """Other tests in this folder put a fake ``opentelemetry`` in sys.modules; use the real one."""
    monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", opentelemetry.trace)


def _evt(event, etype, node="", data=None, metadata=None):
    return EventModel(
        event=event,
        event_type=etype,
        node_name=node,
        run_id="r1",
        thread_id="t1",
        user_id="u1",
        timestamp=10.0,
        data=data or {},
        metadata=metadata or {},
        content_type=[ContentType.MESSAGE],
    )


def _publisher():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OtelPublisher(tracer=provider.get_tracer("t")), exporter


async def _publish(pub, *events):
    for e in events:
        await pub.publish(e)


@pytest.mark.asyncio
async def test_tool_span_is_child_of_its_graph_node_span():
    pub, exporter = _publisher()
    tool_data = {"function_name": "get_weather", "tool_call_id": "c1", "args": {"city": "x"}}
    await _publish(
        pub,
        _evt(Event.GRAPH_EXECUTION, EventType.START),
        _evt(Event.NODE_EXECUTION, EventType.START, node="TOOL"),
        # The tool event's node_name is the tool name; the graph node is in metadata.
        _evt(
            Event.TOOL_EXECUTION,
            EventType.START,
            node="get_weather",
            data=tool_data,
            metadata={"graph_node": "TOOL", "is_mcp": True},
        ),
        _evt(Event.TOOL_EXECUTION, EventType.END, node="get_weather", data=tool_data),
        _evt(Event.NODE_EXECUTION, EventType.END, node="TOOL"),
        _evt(Event.GRAPH_EXECUTION, EventType.END),
    )
    spans = {s.name: s for s in exporter.get_finished_spans()}
    tool, node, graph = (
        spans["execute_tool get_weather"],
        spans["TOOL"],
        spans["invoke_agent 10xgraph"],
    )
    assert tool.parent.span_id == node.context.span_id
    assert node.parent.span_id == graph.context.span_id


@pytest.mark.asyncio
async def test_tool_span_without_graph_node_falls_back_to_graph_span():
    pub, exporter = _publisher()
    tool_data = {"function_name": "t", "tool_call_id": "c1"}
    await _publish(
        pub,
        _evt(Event.GRAPH_EXECUTION, EventType.START),
        _evt(Event.TOOL_EXECUTION, EventType.START, node="t", data=tool_data),
        _evt(Event.TOOL_EXECUTION, EventType.END, node="t", data=tool_data),
        _evt(Event.GRAPH_EXECUTION, EventType.END),
    )
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["execute_tool t"].parent.span_id == spans["invoke_agent 10xgraph"].context.span_id


@pytest.mark.asyncio
async def test_start_node_gets_no_span():
    pub, exporter = _publisher()
    await _publish(
        pub,
        _evt(Event.GRAPH_EXECUTION, EventType.START),
        _evt(Event.NODE_EXECUTION, EventType.START, node=START),
        _evt(Event.NODE_EXECUTION, EventType.END, node=START),
        _evt(Event.NODE_EXECUTION, EventType.START, node="MAIN"),
        _evt(Event.NODE_EXECUTION, EventType.END, node="MAIN"),
        _evt(Event.GRAPH_EXECUTION, EventType.END),
    )
    nodes = [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("gen_ai.operation.name") == "node"
    ]
    assert [s.attributes["tenxgraph.node.name"] for s in nodes] == ["MAIN"]


def _history(turns):
    messages = [{"role": "user", "content": "start"}]
    for i in range(turns):
        call = {"id": f"c{i}", "function": {"name": "search", "arguments": '{"q": "x"}'}}
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "content": f"result {i}", "tool_call_id": f"c{i}"})
    return messages


@pytest.mark.asyncio
async def test_llm_span_keeps_identifying_attrs_with_a_long_history():
    pub, exporter = _publisher()
    history = _history(22)
    params = {
        "temperature": 0.1,
        "max_tokens": 100,
        "top_p": 0.9,
        "top_k": 5,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "seed": 1,
        "stop": ["x"],
    }
    calls = [{"id": f"o{i}", "name": "search", "arguments": "{}"} for i in range(10)]
    await _publish(
        pub,
        _evt(Event.GRAPH_EXECUTION, EventType.START),
        _evt(Event.NODE_EXECUTION, EventType.START, node="MAIN"),
        _evt(
            Event.LLM_CALL,
            EventType.START,
            node="MAIN",
            data={
                "model": "claude-x",
                "provider": "anthropic",
                "request_params": params,
                "tool_count": 3,
                "input_messages": history,
                "system_prompt": "be brief",
            },
        ),
        _evt(
            Event.LLM_CALL,
            EventType.END,
            node="MAIN",
            data={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_tokens": 1,
                "cache_creation_tokens": 1,
                "reasoning_tokens": 1,
                "finish_reason": "tool_use",
                "response_id": "resp",
                "response_model": "claude-x",
                "output_response": "",
                "output_tool_calls": calls,
            },
        ),
        _evt(Event.NODE_EXECUTION, EventType.END, node="MAIN"),
        _evt(Event.GRAPH_EXECUTION, EventType.END),
    )
    llm = next(s for s in exporter.get_finished_spans() if s.name.startswith("chat"))
    a = llm.attributes
    assert llm.dropped_attributes == 0
    assert a[oi.SPAN_KIND] == "LLM"
    assert a[oi.LLM_MODEL_NAME] == "claude-x"
    assert a[oi.LLM_PROVIDER]
    assert a["session.id"] == "t1"
    assert a[oi.USER_ID] == "u1"
    assert a[oi.TOKENS_TOTAL] == 15
    # The newest messages are flattened, from index 0; the full history stays in input.value.
    indexed = [k for k in a if k.startswith("llm.input_messages.")]
    roles = [k for k in indexed if k.endswith(".message.role")]
    last = len(roles) - 1
    assert 0 < len(roles) < len(history) + 1
    assert a[f"llm.input_messages.{last}.message.content"] == "result 21"
    assert len(indexed) <= _input_message_budget()
    assert "result 0" in a[oi.INPUT_VALUE]


def test_input_message_budget_follows_the_span_attribute_limit(monkeypatch):
    monkeypatch.delenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", raising=False)
    monkeypatch.delenv("OTEL_ATTRIBUTE_COUNT_LIMIT", raising=False)
    assert _input_message_budget() == 128 - 64
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "1000")
    assert _input_message_budget() == 1000 - 64
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "10")
    assert _input_message_budget() == 0


def test_llm_input_attrs_without_budget_flattens_every_message():
    attrs = oi.llm_input_attrs(_history(30))
    assert attrs["llm.input_messages.0.message.content"] == "start"
    assert "llm.input_messages.60.message.content" in attrs


@pytest.mark.asyncio
async def test_span_names_and_operations_follow_genai_conventions():
    pub, exporter = _publisher()
    tool_data = {"function_name": "browser_navigate", "tool_call_id": "call_1", "args": {}}
    await _publish(
        pub,
        _evt(Event.GRAPH_EXECUTION, EventType.START),
        _evt(Event.NODE_EXECUTION, EventType.START, node="MAIN"),
        _evt(Event.LLM_CALL, EventType.START, node="MAIN", data={"model": "claude-haiku-5-5"}),
        _evt(Event.LLM_CALL, EventType.END, node="MAIN", data={"model": "claude-haiku-5-5"}),
        _evt(Event.NODE_EXECUTION, EventType.END, node="MAIN"),
        _evt(Event.NODE_EXECUTION, EventType.START, node="TOOL"),
        _evt(
            Event.TOOL_EXECUTION,
            EventType.START,
            node="browser_navigate",
            data=tool_data,
            metadata={"graph_node": "TOOL", "is_mcp": True},
        ),
        _evt(Event.TOOL_EXECUTION, EventType.END, node="browser_navigate", data=tool_data),
        _evt(Event.NODE_EXECUTION, EventType.END, node="TOOL"),
        _evt(Event.GRAPH_EXECUTION, EventType.END),
    )
    spans = {s.name: s.attributes for s in exporter.get_finished_spans()}
    assert set(spans) == {
        "invoke_agent 10xgraph",
        "MAIN",
        "chat claude-haiku-5-5",
        "TOOL",
        "execute_tool browser_navigate",
    }
    assert spans["invoke_agent 10xgraph"]["gen_ai.operation.name"] == "invoke_agent"
    assert spans["invoke_agent 10xgraph"]["logfire.msg"] == "invoke_agent 10xgraph"
    assert spans["chat claude-haiku-5-5"]["gen_ai.operation.name"] == "chat"
    for node in ("MAIN", "TOOL"):
        assert spans[node]["tenxgraph.node.name"] == node
        assert spans[node][oi.SPAN_KIND] == "CHAIN"
    tool = spans["execute_tool browser_navigate"]
    assert tool["gen_ai.operation.name"] == "execute_tool"
    assert tool["gen_ai.tool.name"] == tool["tenxgraph.tool.name"] == "browser_navigate"
    assert tool["gen_ai.tool.call.id"] == "call_1"
    assert tool["gen_ai.tool.type"] == "function"
    assert tool["tenxgraph.tool.type"] == "mcp"
    assert tool["logfire.msg"] == "execute_tool browser_navigate"
    # Called with no arguments: the input shows ``{}``, not nothing.
    assert tool[oi.INPUT_VALUE] == "{}"
    assert tool[oi.INPUT_MIME] == oi.JSON_MIME


@pytest.mark.asyncio
async def test_llm_span_without_model_is_named_chat():
    pub, exporter = _publisher()
    await _publish(
        pub,
        _evt(Event.LLM_CALL, EventType.START, node="MAIN"),
        _evt(Event.LLM_CALL, EventType.END, node="MAIN"),
    )
    assert [s.name for s in exporter.get_finished_spans()] == ["chat"]


@pytest.mark.asyncio
async def test_tool_span_without_args_info_has_no_input():
    pub, exporter = _publisher()
    data = {"function_name": "t", "tool_call_id": "c1"}
    await _publish(
        pub,
        _evt(Event.TOOL_EXECUTION, EventType.START, node="t", data=data),
        _evt(Event.TOOL_EXECUTION, EventType.END, node="t", data=data),
    )
    (tool,) = exporter.get_finished_spans()
    assert tool.name == "execute_tool t"
    assert oi.INPUT_VALUE not in tool.attributes


def test_tool_input_attrs_empty_and_missing_args():
    assert oi.tool_input_attrs(None) == {}
    assert oi.tool_input_attrs({}) == {oi.INPUT_VALUE: "{}", oi.INPUT_MIME: oi.JSON_MIME}
    assert oi.tool_input_attrs("")[oi.INPUT_VALUE] == "{}"
    assert oi.tool_input_attrs({"q": 1})[oi.INPUT_VALUE] == '{"q":1}'
