"""setup_tracing on a real StateGraph must produce spans (container binding)."""

import pytest


pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher import ConsolePublisher
from tenxgraph.runtime.publisher.composite_publisher import CompositePublisher
from tenxgraph.runtime.publisher.otel_publisher import setup_tracing
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
async def test_setup_tracing_emits_spans():
    tracer, exporter = _tracer()
    graph = _build()
    setup_tracing(graph, tracer=tracer)
    await _run(graph)
    names = [s.name for s in exporter.get_finished_spans()]
    assert names, "no spans emitted"
    assert "tenxgraph.graph" in names
    assert "tenxgraph.node" in names


@pytest.mark.asyncio
async def test_setup_tracing_keeps_existing_publisher():
    tracer, exporter = _tracer()
    console = ConsolePublisher()
    graph = _build(publisher=console)
    pub = setup_tracing(graph, tracer=tracer)
    assert isinstance(graph._publisher, CompositePublisher)
    assert console in graph._publisher._publishers
    assert pub in graph._publisher._publishers
    await _run(graph)
    assert exporter.get_finished_spans()
