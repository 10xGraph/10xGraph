"""Who gets which event: one 3-node run (MAIN agent, TOOL, MAIN agent) per publisher kind,
and the OTEL span tree for a completed, failed, interrupted and stopped run.

Event publishers (``kind="events"``) get the ``all``-tagged events only, with no state except
on the run-end event. Observability publishers get every event. The run uses a real ``Agent``
(its LLM call mocked) so the LLM_CALL events are the ones a real run sends.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from injectq import InjectQ

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.composite_publisher import CompositePublisher
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import END


def lookup(city: str) -> str:
    return f"sunny in {city}"


def _response(content, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
        model="gpt-4o",
        id="chatcmpl-1",
        model_dump=MagicMock(return_value={"id": "chatcmpl-1"}),
    )


def _tool_call_response():
    call = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(name="lookup", arguments='{"city": "Paris"}'),
    )
    return _response(None, [call])


class Counting(BasePublisher):
    def __init__(self, kind: str):
        super().__init__({})
        self.kind = kind  # type: ignore[assignment]
        self.events = []

    async def publish(self, event):
        self.events.append(event)

    async def close(self):
        pass

    def sync_close(self):
        pass


def _route(state: AgentState) -> str:
    last = state.context[-1]
    return "TOOL" if last.role == "assistant" and last.tools_calls else END


def _app(publisher):
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            api_style="chat",
            tool_node="TOOL",
            reasoning_config=None,
            retry_config=None,
        )
    agent._call_llm = AsyncMock(side_effect=[_tool_call_response(), _response("done")])
    graph = StateGraph(container=InjectQ(), publisher=publisher)
    graph.add_node("MAIN", agent)
    graph.add_node("TOOL", ToolNode([lookup]))
    graph.add_conditional_edges("MAIN", _route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    return graph.compile()


def _names(events):
    return [
        (str(e.event), str(e.event_type), e.node_name, e.metadata.get("event_name")) for e in events
    ]


RUN_EVENTS = [
    ("graph_execution", "start", "", "run.started"),
    ("node_execution", "end", "MAIN", "node.completed"),
    ("tool_execution", "start", "TOOL", "tool.started"),
    ("tool_execution", "end", "TOOL", "tool.completed"),
    ("node_execution", "end", "TOOL", "node.completed"),
    ("node_execution", "end", "MAIN", "node.completed"),
    ("graph_execution", "end", "", "run.completed"),
]

TRACE_EVENTS = [
    ("graph_execution", "start", "", "run.started"),
    ("node_execution", "start", "MAIN", None),
    ("llm_call", "start", "MAIN", None),
    ("llm_call", "end", "MAIN", None),
    ("node_execution", "end", "MAIN", "node.completed"),
    ("node_execution", "start", "TOOL", None),
    ("tool_execution", "start", "TOOL", "tool.started"),
    ("tool_execution", "end", "TOOL", "tool.completed"),
    ("node_execution", "end", "TOOL", "node.completed"),
    ("node_execution", "start", "MAIN", None),
    ("llm_call", "start", "MAIN", None),
    ("llm_call", "end", "MAIN", None),
    ("node_execution", "end", "MAIN", "node.completed"),
    ("graph_execution", "end", "", "run.completed"),
]


@pytest.mark.asyncio
async def test_three_node_run_sends_each_kind_its_events():
    events, traces = Counting("events"), Counting("observability")
    app = _app(CompositePublisher([events, traces]))
    await app.ainvoke({"messages": [Message.text_message("weather?")]}, {"thread_id": "t1"})
    await app.aclose()

    assert _names(events.events) == RUN_EVENTS
    assert _names(traces.events) == TRACE_EVENTS
    # Event publishers get the state only on the run-end event.
    with_state = [e.metadata["event_name"] for e in events.events if "state" in e.data]
    assert with_state == ["run.completed"]
    end = events.events[-1]
    assert end.metadata["total_tokens"] == 30
    assert "time_taken" in end.metadata
    assert [m["role"] for m in end.data["messages"]] == ["user", "assistant", "tool", "assistant"]
    # node.completed carries the node's new messages and the chosen next node.
    first = events.events[1]
    assert [m["role"] for m in first.data["messages"]] == ["assistant"]
    assert "next_node" in first.data
    # No run config is published.
    assert all("config" not in e.metadata for e in events.events + traces.events)


@pytest.mark.asyncio
async def test_events_publisher_alone_builds_no_llm_payload():
    events = Counting("events")
    app = _app(events)
    dumps = {"n": 0}
    real = AgentState.model_dump

    def counted(self, *args, **kwargs):
        caller = sys._getframe(1).f_code.co_filename.replace("\\", "/")
        if caller.endswith("runtime/publisher/publish.py"):
            dumps["n"] += 1
        return real(self, *args, **kwargs)

    with patch.object(AgentState, "model_dump", counted):
        await app.ainvoke({"messages": [Message.text_message("weather?")]}, {"thread_id": "t2"})
    await app.aclose()
    assert _names(events.events) == RUN_EVENTS
    # The state is dumped once (run.completed), not on every node and LLM event.
    assert dumps["n"] == 1


# ── OTEL span tree ────────────────────────────────────────────────────────────


def _otel():
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    export = pytest.importorskip("opentelemetry.sdk.trace.export")
    memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    from tenxgraph.runtime.publisher.otel_publisher import ObservabilityLevel, OtelPublisher

    exporter = memory.InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    publisher = OtelPublisher(tracer=provider.get_tracer("test"), level=ObservabilityLevel.FULL)
    return publisher, exporter


def _tree(exporter):
    """``(name, parent name, status, run_status)`` per span, in end order."""
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    rows = []
    for s in spans:
        parent = by_id.get(s.parent.span_id).name if s.parent else None
        rows.append(
            (
                s.name,
                parent,
                s.status.status_code.name,
                s.attributes.get("tenxgraph.graph.run_status"),
            )
        )
    # Every parent was exported: no span is left open.
    assert all(s.parent is None or s.parent.span_id in by_id for s in spans)
    return rows


def _assert_nothing_open(publisher):
    assert publisher._registry._graph == {}
    assert publisher._registry._children == {}


@pytest.mark.asyncio
async def test_span_tree_of_a_completed_run():
    publisher, exporter = _otel()
    app = _app(publisher)
    await app.ainvoke({"messages": [Message.text_message("weather?")]}, {"thread_id": "s1"})
    _assert_nothing_open(publisher)
    await app.aclose()
    graph = "invoke_agent 10xgraph"
    assert _tree(exporter) == [
        ("chat gpt-4o", "MAIN", "OK", None),
        ("MAIN", graph, "OK", None),
        ("execute_tool lookup", "TOOL", "OK", None),
        ("TOOL", graph, "OK", None),
        ("chat gpt-4o", "MAIN", "OK", None),
        ("MAIN", graph, "OK", None),
        (graph, None, "OK", "completed"),
    ]
    root = exporter.get_finished_spans()[-1]
    assert root.attributes["tenxgraph.graph.model"] == "gpt-4o"
    assert root.attributes["tenxgraph.usage.input_tokens"] == 10
    assert root.attributes["tenxgraph.usage.output_tokens"] == 20
    assert "gen_ai.content.completion" in [e.name for e in root.events]


def _function_app(publisher, first, interrupt_before=None):
    def second(state: AgentState):
        return Message.text_message("second", role="assistant")

    graph = StateGraph(container=InjectQ(), publisher=publisher)
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.set_entry_point("FIRST")
    return graph.compile(checkpointer=InMemoryCheckpointer(), interrupt_before=interrupt_before)


@pytest.mark.asyncio
async def test_span_tree_of_a_failed_run():
    publisher, exporter = _otel()

    def first(state: AgentState):
        raise RuntimeError("boom")

    app = _function_app(publisher, first)
    with pytest.raises(Exception, match="boom"):
        await app.ainvoke({"messages": [Message.text_message("hi")]}, {"thread_id": "s2"})
    _assert_nothing_open(publisher)
    await app.aclose()
    graph = "invoke_agent 10xgraph"
    assert _tree(exporter) == [("FIRST", graph, "ERROR", None), (graph, None, "ERROR", "failed")]


@pytest.mark.asyncio
async def test_span_tree_of_an_interrupted_run():
    publisher, exporter = _otel()

    def first(state: AgentState):
        return Message.text_message("first", role="assistant")

    app = _function_app(publisher, first, interrupt_before=["SECOND"])
    await app.ainvoke({"messages": [Message.text_message("hi")]}, {"thread_id": "s3"})
    _assert_nothing_open(publisher)
    await app.aclose()
    graph = "invoke_agent 10xgraph"
    assert _tree(exporter) == [("FIRST", graph, "OK", None), (graph, None, "OK", "interrupted")]


@pytest.mark.asyncio
async def test_span_tree_of_a_stopped_run():
    publisher, exporter = _otel()
    holder: dict = {}

    async def first(state: AgentState, config: dict):
        await holder["app"].astop({"thread_id": config["thread_id"]})
        return Message.text_message("first", role="assistant")

    holder["app"] = app = _function_app(publisher, first)
    await app.ainvoke({"messages": [Message.text_message("hi")]}, {"thread_id": "s4"})
    _assert_nothing_open(publisher)
    await app.aclose()
    graph = "invoke_agent 10xgraph"
    assert _tree(exporter) == [("FIRST", graph, "OK", None), (graph, None, "OK", "stopped")]


@pytest.mark.asyncio
async def test_span_tree_of_a_run_stopped_while_a_node_runs():
    """A node cut by a stop ends ERROR "cancelled", like a cancelled tool span."""
    publisher, exporter = _otel()
    holder: dict = {}

    async def first(state: AgentState, config: dict):
        await holder["app"].astop({"thread_id": config["thread_id"]})
        await asyncio.sleep(30)
        return Message.text_message("first", role="assistant")

    holder["app"] = app = _function_app(publisher, first)
    await app.ainvoke({"messages": [Message.text_message("hi")]}, {"thread_id": "s5"})
    _assert_nothing_open(publisher)
    await app.aclose()
    graph = "invoke_agent 10xgraph"
    assert _tree(exporter) == [("FIRST", graph, "ERROR", None), (graph, None, "OK", "stopped")]
    node = next(s for s in exporter.get_finished_spans() if s.name == "FIRST")
    assert node.status.description == "cancelled"
