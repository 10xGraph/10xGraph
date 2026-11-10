"""The run's UserCredential reaches local tools and stays out of everything that leaves the run.

Each sink test runs a full graph with a credential in the config and searches what left the
run (events, checkpoints, callback contexts, logs, spans, stream chunks, results) for the
token string.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import pytest

from tenxgraph.core.authz import Identity, UserCredential, get_credential
from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.otel_publisher import ObservabilityLevel, setup_tracing
from tenxgraph.storage.checkpointer import InMemoryCheckpointer, SqliteCheckpointer
from tenxgraph.utils import CallbackManager, InvocationType, tool
from tenxgraph.utils.callbacks import GraphLifecycleHook
from tenxgraph.utils.constants import END, ResponseGranularity


TOKEN = "tok-very-secret-0123456789"  # noqa: S105 - test value
SEEN: dict[str, Any] = {}


def _dump(obj: Any) -> str:
    """Serialize like an exporter would; a UserCredential would show as its class name."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return json.dumps(obj, default=repr)


def _assert_clean(text: str) -> None:
    assert TOKEN not in text
    assert "UserCredential(" not in text


# ----------------------------------------------------------------------------- tools


def annotated(x: int, credential: UserCredential) -> str:
    """Annotated tool."""
    SEEN["annotated"] = credential
    return f"ok {x}"


def optional_annotated(x: int, credential: Optional[UserCredential] = None) -> str:  # noqa: UP045
    """Optional annotation."""
    SEEN["optional_annotated"] = credential
    return "ok"


def unannotated(x: int, credential=None) -> str:
    """No annotation."""
    SEEN["unannotated"] = credential
    return "ok"


def dict_annotated(x: int, credential: dict | None = None) -> str:
    """Wrong annotation: still gets the object."""
    SEEN["dict_annotated"] = credential
    return "ok"


async def async_tool(x: int, credential: UserCredential | None = None) -> str:
    """Async tool."""
    SEEN["async_tool"] = credential
    return "ok"


@tool(name="decorated", description="Decorated tool.")
def decorated(x: int, credential: UserCredential | None = None) -> str:
    SEEN["decorated"] = credential
    return "ok"


def no_credential(x: int) -> str:
    """Takes no credential."""
    return "ok"


def leaky_error(x: int, credential: UserCredential) -> str:
    """Raises with the token in the message."""
    raise RuntimeError(f"upstream refused {credential.token.get_secret_value()}")


# ----------------------------------------------------------------------------- helpers


def _call(name: str, call_id: str = "call_1", args: str = '{"x": 1}') -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def _graph(tool_node: ToolNode, calls: list[dict], publisher=None) -> StateGraph:
    assistant = Message.text_message("calling", role="assistant")
    assistant.tools_calls = calls

    def model(state: AgentState, config: dict):
        # Inside the run the config keeps the credential.
        SEEN["node_config_credential"] = get_credential(config)
        if any(m.role == "tool" for m in state.context):
            return Message.text_message("done", role="assistant")
        return assistant.model_copy(deep=True, update={"message_id": "assistant_1"})

    graph = StateGraph(publisher=publisher)
    graph.add_node("MAIN", model)
    graph.add_node("TOOL", tool_node)
    graph.add_edge("MAIN", "TOOL")
    graph.add_edge("TOOL", END)
    graph.set_entry_point("MAIN")
    return graph


def _config(thread_id: str = "t-cred") -> dict:
    return {
        "thread_id": thread_id,
        "user": Identity(user_id="u1", claims={"name": "Ada"}),
        "credential": UserCredential(TOKEN),
    }


class RecordingPublisher(BasePublisher):
    def __init__(self):
        super().__init__({})
        self.events: list = []

    async def publish(self, event):
        self.events.append(event)

    async def close(self):
        pass

    def sync_close(self):
        pass


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()


# ----------------------------------------------------------------------------- injection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fn",
    [annotated, optional_annotated, unannotated, dict_annotated, async_tool, decorated],
)
async def test_tool_receives_the_credential_object(fn):
    node = ToolNode([fn])
    name = getattr(fn, "_py_tool_name", fn.__name__)
    cred = UserCredential(TOKEN)
    await node.invoke(name, {"x": 1}, "c1", {"credential": cred}, AgentState())
    key = "decorated" if name == "decorated" else fn.__name__
    assert SEEN[key] is cred


@pytest.mark.asyncio
async def test_tool_gets_none_without_a_credential():
    node = ToolNode([annotated])
    msg = await node.invoke("annotated", {"x": 1}, "c1", {}, AgentState())
    assert SEEN["annotated"] is None
    assert msg.content[0].output == "ok 1"


@pytest.mark.asyncio
async def test_model_cannot_supply_the_credential():
    node = ToolNode([annotated])
    cred = UserCredential(TOKEN)
    await node.invoke(
        "annotated", {"x": 1, "credential": "fake"}, "c1", {"credential": cred}, AgentState()
    )
    assert SEEN["annotated"] is cred


@pytest.mark.asyncio
async def test_raw_string_credential_in_config_is_refused():
    node = ToolNode([annotated])
    msg = await node.invoke("annotated", {"x": 1}, "c1", {"credential": TOKEN}, AgentState())
    assert "annotated" not in SEEN
    assert msg.content[0].is_error is True
    _assert_clean(_dump(msg))


def test_credential_is_hidden_from_the_model_schema():
    node = ToolNode([annotated, unannotated, dict_annotated, decorated, no_credential])
    for schema in node.get_local_tool():
        params = schema["function"]["parameters"]
        assert "credential" not in params.get("properties", {})
        assert "credential" not in params.get("required", [])


def test_wrong_annotation_warns_once_per_tool(caplog):
    caplog.set_level(logging.WARNING, logger="tenxgraph.graph.tool_node")
    ToolNode([annotated, optional_annotated, unannotated, async_tool, decorated, no_credential])
    assert not [r for r in caplog.records if "credential" in r.getMessage()]

    ToolNode([dict_annotated])
    warnings = [r for r in caplog.records if "'credential'" in r.getMessage()]
    assert len(warnings) == 1
    assert "dict_annotated" in warnings[0].getMessage()
    assert "UserCredential" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_tool_callbacks_never_see_the_credential():
    seen: list = []

    async def before(ctx, data):
        seen.append(("before", ctx, dict(data)))
        return data

    async def after(ctx, data, result):
        seen.append(("after", ctx, dict(data)))
        return result

    mgr = CallbackManager()
    mgr.register_before_invoke(InvocationType.TOOL, before)
    mgr.register_after_invoke(InvocationType.TOOL, after)
    node = ToolNode([annotated])
    await node.invoke("annotated", {"x": 1}, "c1", _config(), AgentState(), callback_manager=mgr)
    assert isinstance(SEEN["annotated"], UserCredential)
    assert len(seen) == 2
    for _, ctx, data in seen:
        assert "credential" not in data
        assert "credential" not in ctx.metadata["config"]


# ----------------------------------------------------------------------------- sinks


@pytest.mark.asyncio
async def test_full_run_events_checkpoint_callbacks_and_logs_are_clean(caplog):
    caplog.set_level(logging.DEBUG)
    publisher = RecordingPublisher()
    contexts: list = []

    class Hook(GraphLifecycleHook):
        async def on_graph_start(self, ctx, state):
            contexts.append(ctx.config)
            return state

        async def on_graph_end(self, ctx, final_state, messages, total_steps):
            contexts.append(ctx.config)

    async def before(ctx, data):
        contexts.append(ctx.metadata)
        return data

    mgr = CallbackManager()
    mgr.register_lifecycle_hook(Hook())
    mgr.register_before_invoke(InvocationType.AI, before)
    mgr.register_before_invoke(InvocationType.TOOL, before)

    checkpointer = InMemoryCheckpointer()
    calls = [_call("annotated", "call_1"), _call("async_tool", "call_2")]
    graph = _graph(ToolNode([annotated, async_tool]), calls, publisher=publisher)
    app = graph.compile(checkpointer=checkpointer, callback_manager=mgr)
    config = _config()
    result = await app.ainvoke(
        {"messages": [Message.text_message("hi")]},
        config=config,
        response_granularity=ResponseGranularity.FULL,
    )

    # The credential reached the node and both tools.
    assert SEEN["node_config_credential"] is config["credential"]
    assert SEEN["annotated"] is config["credential"]
    assert SEEN["async_tool"] is config["credential"]

    assert publisher.events
    for event in publisher.events:
        # What Redis/Kafka/RabbitMQ publishers send.
        json.dumps(event.model_dump())
        _assert_clean(_dump(event))

    _assert_clean(_dump(result))

    state = await checkpointer.aget_state({"thread_id": "t-cred", "user": config["user"]})
    _assert_clean(_dump(state))
    for msg in await checkpointer.alist_messages({"thread_id": "t-cred", "user": config["user"]}):
        _assert_clean(_dump(msg))

    assert len(contexts) >= 4
    for ctx in contexts:
        _assert_clean(_dump(ctx))

    _assert_clean(caplog.text)
    for record in caplog.records:
        _assert_clean(repr(record.args))
    await app.aclose()


@pytest.mark.asyncio
async def test_tool_error_text_is_clean():
    publisher = RecordingPublisher()
    graph = _graph(ToolNode([leaky_error]), [_call("leaky_error")], publisher=publisher)
    app = graph.compile()
    result = await app.ainvoke(
        {"messages": [Message.text_message("hi")]},
        config=_config("t-err"),
        response_granularity=ResponseGranularity.FULL,
    )
    tool_msg = next(m for m in result["messages"] if m.role == "tool")
    block = next(b for b in tool_msg.content if isinstance(b, ToolResultBlock))
    assert block.is_error is True
    _assert_clean(_dump(result))
    for event in publisher.events:
        _assert_clean(_dump(event))
    await app.aclose()


@pytest.mark.asyncio
async def test_stream_chunks_and_events_are_clean():
    publisher = RecordingPublisher()
    graph = _graph(ToolNode([annotated]), [_call("annotated")], publisher=publisher)
    app = graph.compile(checkpointer=InMemoryCheckpointer())
    chunks = [
        chunk
        async for chunk in app.astream(
            {"messages": [Message.text_message("hi")]},
            config=_config("t-stream"),
            response_granularity=ResponseGranularity.FULL,
        )
    ]
    assert chunks
    assert isinstance(SEEN["annotated"], UserCredential)
    for chunk in chunks:
        _assert_clean(_dump(chunk))
    init = [e for e in publisher.events if "config" in e.metadata]
    assert init, "the graph start event carries the run config"
    for event in publisher.events:
        json.dumps(event.model_dump())
        _assert_clean(_dump(event))
    await app.aclose()


@pytest.mark.asyncio
async def test_sqlite_checkpoint_is_clean(tmp_path):
    pytest.importorskip("aiosqlite")
    db = tmp_path / "cp.db"
    checkpointer = SqliteCheckpointer(db)
    graph = _graph(ToolNode([annotated]), [_call("annotated")])
    app = graph.compile(checkpointer=checkpointer)
    await app.ainvoke({"messages": [Message.text_message("hi")]}, config=_config("t-sqlite"))
    await app.aclose()
    assert TOKEN.encode() not in db.read_bytes()


@pytest.mark.asyncio
async def test_otel_full_level_spans_are_clean():
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    export = pytest.importorskip("opentelemetry.sdk.trace.export")
    memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

    exporter = memory.InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    graph = _graph(ToolNode([annotated]), [_call("annotated")])
    setup_tracing(graph, tracer=provider.get_tracer("test"), level=ObservabilityLevel.FULL)
    app = graph.compile()
    await app.ainvoke({"messages": [Message.text_message("hi")]}, config=_config("t-otel"))
    await app.aclose()

    spans = exporter.get_finished_spans()
    assert spans
    assert isinstance(SEEN["annotated"], UserCredential)
    for span in spans:
        _assert_clean(span.to_json())
