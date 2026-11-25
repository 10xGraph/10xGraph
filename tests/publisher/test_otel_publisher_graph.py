"""StateGraph(publisher=OtelPublisher(...)) on a real graph must produce spans."""

import pytest


pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher import ConsolePublisher
from tenxgraph.runtime.publisher.composite_publisher import CompositePublisher
from tenxgraph.runtime.publisher.otel_publisher import OtelPublisher
from tenxgraph.utils import END


def _node(state: AgentState):
    return state


def _build(**kwargs) -> StateGraph:
    graph = StateGraph(**kwargs)
    graph.add_node("MAIN", _node)
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    return graph


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


async def _run(graph):
    app = graph.compile()
    await app.ainvoke({"messages": [Message.text_message("hi")]}, config={"thread_id": "t1"})
    await app.aclose()


@pytest.mark.asyncio
async def test_otel_publisher_emits_spans():
    tracer, exporter = _tracer()
    graph = _build(publisher=OtelPublisher(tracer=tracer))
    await _run(graph)
    names = [s.name for s in exporter.get_finished_spans()]
    assert names, "no spans emitted"
    assert "invoke_agent 10xgraph" in names
    assert "MAIN" in names
    for span in exporter.get_finished_spans():
        assert span.attributes[
            "tenxgraph.graph.thread_id"
            if span.name.startswith("invoke")
            else "gen_ai.operation.name"
        ]
        assert not any(k.startswith("gen_ai.usage.") for k in span.attributes)
        # A plain OtelPublisher writes no backend dialect.
        assert not any(
            k.startswith(("langsmith.", "logfire.", "openinference.")) for k in span.attributes
        )


@pytest.mark.asyncio
async def test_otel_publisher_in_a_list_with_another_publisher():
    tracer, exporter = _tracer()
    console = ConsolePublisher()
    pub = OtelPublisher(tracer=tracer)
    graph = _build(publisher=[console, pub])
    assert isinstance(graph._publisher, CompositePublisher)
    assert console in graph._publisher._publishers
    assert pub in graph._publisher._publishers
    await _run(graph)
    assert exporter.get_finished_spans()
