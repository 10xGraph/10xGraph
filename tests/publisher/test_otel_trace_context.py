"""Library spans nest under the run's 10xGraph spans; error events carry the exception."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


pytest.importorskip("opentelemetry.sdk")

import opentelemetry
import opentelemetry.trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.adapters.llm.base_converter import BaseConverter
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.otel_publisher import OtelPublisher, setup_tracing
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.runtime.publisher.trace_context import find_span, tool_span, use_trace_span
from tenxgraph.utils.constants import END
from tenxgraph.utils.log_safety import TRACE_ONLY_ERROR_KEYS, error_details


@pytest.fixture(autouse=True)
def _real_otel(monkeypatch):
    """Other tests in this folder put a fake ``opentelemetry`` in sys.modules; use the real one."""
    monkeypatch.setitem(sys.modules, "opentelemetry", opentelemetry)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", opentelemetry.trace)


@pytest.fixture
def otel():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # ``lib`` stands in for a library tracing with its own tracer (anthropic, fastmcp, mcp).
    return SimpleNamespace(
        exporter=exporter, tracer=provider.get_tracer("10xgraph"), lib=provider.get_tracer("lib")
    )


def _spans(otel, name):
    return [s for s in otel.exporter.get_finished_spans() if s.name == name]


def _tools(otel):
    finished = otel.exporter.get_finished_spans()
    return [s for s in finished if s.name.startswith("execute_tool ")]


def _parent_name(otel, span):
    by_id = {s.context.span_id: s for s in otel.exporter.get_finished_spans()}
    return by_id[span.parent.span_id].name if span.parent else None


def _completion(content=None, tool_calls=None):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def _call(call_id, name):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


class _MCPTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} description"
        self.inputSchema = {"type": "object", "properties": {}}
        self.meta = {}


class _TracingClient:
    """A fake MCP client that, like fastmcp, starts a span with the current context."""

    def __init__(self, lib):
        self._lib = lib

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        with self._lib.start_as_current_span("lib.list"):
            return [_MCPTool("a"), _MCPTool("b")]

    async def call_tool(self, name, args, *, meta=None):
        with self._lib.start_as_current_span(f"lib.call.{name}"):
            # Both calls are open at once: each must keep its own parent.
            await asyncio.sleep(0.01)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"{name} ok")])


def _graph(otel, *, fail_llm=False):
    replies = [
        _completion(tool_calls=[_call("call_a", "a"), _call("call_b", "b")]),
        _completion(content="done"),
    ]

    async def fake_llm(self, messages, tools=None, stream=False, **kwargs):
        with otel.lib.start_as_current_span("lib.llm"):
            if fail_llm:
                raise RuntimeError("provider said no")
            return replies.pop(0)

    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            tool_node="TOOL",
            retry_config=False,
            api_style="chat",
        )

    def route(state: AgentState):
        last = state.context[-1] if state.context else None
        return "TOOL" if last is not None and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("MAIN", agent)
    graph.add_node("TOOL", ToolNode([], client=_TracingClient(otel.lib)))
    graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    setup_tracing(graph, tracer=otel.tracer)
    return graph, patch.object(Agent, "_call_llm", fake_llm)


async def _run(graph, llm_patch, **kwargs):
    app = graph.compile()
    try:
        with llm_patch:
            await app.ainvoke(
                {"messages": [Message.text_message("hi")]},
                config={"thread_id": "t1", "run_id": "r1"},
                **kwargs,
            )
    finally:
        await app.aclose()


@pytest.mark.asyncio
async def test_library_spans_nest_under_llm_tool_and_node_spans(otel):
    graph, llm_patch = _graph(otel)
    await _run(graph, llm_patch)

    graph_span = _spans(otel, "invoke_agent 10xgraph")[0]
    trace_ids = {s.context.trace_id for s in otel.exporter.get_finished_spans()}
    assert trace_ids == {graph_span.context.trace_id}

    lib_llm = _spans(otel, "lib.llm")
    assert len(lib_llm) == 2
    assert all(_parent_name(otel, s) == "chat gpt-4o" for s in lib_llm)

    # Tool listing runs from the agent node: under that node's span.
    lists = _spans(otel, "lib.list")
    assert lists and all(_parent_name(otel, s) == "MAIN" for s in lists)
    for s in lists:
        parent = next(p for p in otel.exporter.get_finished_spans() if p.context == s.parent)
        assert parent.attributes["tenxgraph.node.name"] == "MAIN"

    # Two parallel calls: each library span under its own tool span.
    tools = {s.attributes["tenxgraph.tool.name"]: s for s in _tools(otel)}
    for name in ("a", "b"):
        (call,) = _spans(otel, f"lib.call.{name}")
        assert call.parent.span_id == tools[name].context.span_id


@pytest.mark.asyncio
async def test_failed_llm_call_records_an_exception_event(otel):
    graph, llm_patch = _graph(otel, fail_llm=True)
    with pytest.raises(Exception, match="provider said no"):
        await _run(graph, llm_patch)

    (llm,) = _spans(otel, "chat gpt-4o")
    assert llm.status.status_code.name == "ERROR"
    (exc,) = [e for e in llm.events if e.name == "exception"]
    assert exc.attributes["exception.type"] == "RuntimeError"
    assert exc.attributes["exception.message"] == "provider said no"
    assert "fake_llm" in exc.attributes["exception.stacktrace"]
    assert [e for e in llm.events if e.name == "llm.error"]
    (node,) = _spans(otel, "MAIN")
    assert [e for e in node.events if e.name == "exception"]
    # The library span still nests under the LLM span.
    (lib,) = _spans(otel, "lib.llm")
    assert _parent_name(otel, lib) == "chat gpt-4o"


def _event(event, etype, node="", data=None, metadata=None):
    return EventModel(
        event=event,
        event_type=etype,
        node_name=node,
        run_id="r1",
        thread_id="t1",
        data=data or {},
        metadata=metadata or {},
        content_type=[ContentType.MESSAGE],
    )


def _boom():
    try:
        raise ValueError("bad thing")
    except ValueError as exc:
        return exc


@pytest.mark.asyncio
async def test_tool_end_with_error_records_an_exception_event(otel):
    pub = OtelPublisher(tracer=otel.tracer)
    data = {"function_name": "t", "tool_call_id": "c1"}
    for e in (
        _event(Event.GRAPH_EXECUTION, EventType.START),
        _event(Event.TOOL_EXECUTION, EventType.START, "t", data),
        _event(
            Event.TOOL_EXECUTION,
            EventType.END,
            "t",
            {**data, "error": "Tool 't' failed (reference x)", **error_details(_boom())},
        ),
    ):
        await pub.publish(e)
    (tool,) = _spans(otel, "execute_tool t")
    assert tool.status.description == "Tool 't' failed (reference x)"
    (exc,) = [e for e in tool.events if e.name == "exception"]
    assert exc.attributes["exception.type"] == "ValueError"
    assert exc.attributes["exception.message"] == "bad thing"
    assert "ValueError: bad thing" in exc.attributes["exception.stacktrace"]


@pytest.mark.asyncio
async def test_error_without_details_has_no_exception_event(otel):
    pub = OtelPublisher(tracer=otel.tracer)
    await pub.publish(_event(Event.NODE_EXECUTION, EventType.START, "N"))
    await pub.publish(_event(Event.NODE_EXECUTION, EventType.ERROR, "N", {"error": "x"}))
    (node,) = _spans(otel, "N")
    assert [e.name for e in node.events] == ["node.error"]


def test_error_details_are_bounded_and_redacted():
    from tenxgraph.core.authz import UserCredential

    token = "tok-0123456789abcdef"  # noqa: S105 - test value
    exc = RuntimeError(f"refused {token} " + "x" * 5000)
    details = error_details(exc, {"credential": UserCredential(token)})
    assert token not in details["error_message"] and token not in details["error_traceback"]
    assert len(details["error_message"]) <= 2003
    assert len(details["error_traceback"]) <= 8003
    assert details["error_traceback"].rstrip().endswith("x")


class _Recorder(BasePublisher):
    def __init__(self, kind):
        super().__init__({})
        self.kind = kind
        self.events = []

    async def publish(self, event):
        self.events.append(event)

    async def close(self):
        pass

    def sync_close(self):
        pass


class _Inline(_Recorder):
    inline = True

    def publish_inline(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_exception_text_reaches_observability_publishers_only():
    from tenxgraph.runtime.publisher.composite_publisher import CompositePublisher
    from tenxgraph.utils.background_task_manager import BackgroundTaskManager

    stream, trace_bg = _Recorder("events"), _Recorder("observability")
    trace_inline = _Inline("observability")
    tasks = BackgroundTaskManager()
    event = _event(Event.NODE_EXECUTION, EventType.ERROR, "N", {"error": "x"})
    event.data.update(error_details(_boom()))

    publish_event(event, CompositePublisher([stream, trace_bg, trace_inline]), tasks)
    # Inline publishers have the event before publish_event returns.
    assert len(trace_inline.events) == 1
    await tasks.wait_for_all(timeout=1)

    for pub in (trace_bg, trace_inline):
        assert set(TRACE_ONLY_ERROR_KEYS) <= set(pub.events[0].data)
    assert not set(TRACE_ONLY_ERROR_KEYS) & set(stream.events[0].data)
    assert stream.events[0].data["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_inline_publisher_gets_start_before_end_and_keeps_others_in_background(otel):
    from tenxgraph.runtime.publisher.composite_publisher import CompositePublisher
    from tenxgraph.utils.background_task_manager import BackgroundTaskManager

    stream, trace = _Recorder("events"), OtelPublisher(tracer=otel.tracer)
    tasks = BackgroundTaskManager()
    composite = CompositePublisher([stream, trace])
    publish_event(_event(Event.GRAPH_EXECUTION, EventType.START), composite, tasks)
    publish_event(_event(Event.NODE_EXECUTION, EventType.START, "N"), composite, tasks)
    # The span exists as soon as START is published; the stream gets it later.
    assert trace.open_span("r1", "N") is not None
    assert stream.events == []
    publish_event(_event(Event.NODE_EXECUTION, EventType.END, "N"), composite, tasks)
    assert trace.open_span("r1", "N") is None
    assert len(_spans(otel, "N")) == 1
    await tasks.wait_for_all(timeout=1)
    assert len(stream.events) == 3


@pytest.mark.asyncio
async def test_streamed_reads_run_under_the_trace_span(otel):
    seen = []

    class _Conv(BaseConverter):
        async def convert_response(self, response):
            raise NotImplementedError

        async def convert_streaming_response(self, config, node_name, response, meta=None):
            for i in range(2):
                seen.append(opentelemetry.trace.get_current_span())
                yield Message.text_message(f"chunk {i}", role="assistant")

    span = otel.tracer.start_span("chat gpt-4o")
    converter = ModelResponseConverter(object(), converter=_Conv())
    converter.trace_span = span
    async for _ in converter.stream({"thread_id": "t"}, "MAIN"):
        # Not across the yield: the consumer sees its own context.
        assert opentelemetry.trace.get_current_span() is not span
    assert seen == [span, span]


def test_no_otel_publisher_bound_is_a_no_op():
    assert find_span({"run_id": "r1"}, "N", publisher=None) is None
    assert find_span(None, "N", publisher=_Recorder("events")) is None
    assert tool_span({"run_id": "r1", "_node_name": "N"}, "c1") is None
    with use_trace_span(None):
        pass


def test_trace_context_imports_without_opentelemetry(tmp_path):
    import subprocess

    code = (
        "import sys; sys.modules['opentelemetry'] = None\n"
        "from tenxgraph.runtime.publisher.trace_context import find_span, use_trace_span\n"
        "from tenxgraph.core.graph.tool_node import mcp_exec\n"
        "assert find_span({'run_id': 'r'}, 'N') is None\n"
        "with use_trace_span(None): pass\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path
    )
    assert out.stdout.strip() == "ok"
