"""The one graph loop (engine merge, step 4): the same behaviour under ainvoke and astream.

T1 ``on_state_update`` on a client-tool pause, T3 the run's input messages on a client-tool
resume, T4 ``Command(update=..., goto=...)`` through the full loop, T6 one GRAPH_EXECUTION
INTERRUPTED per stop, T8 ainvoke builds no stream chunk and makes a non-streaming LLM call.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message, ToolCallBlock, ToolResultBlock
from tenxgraph.core.state.stream_chunks import StreamChunk
from tenxgraph.qa.testing.test_agent import TestAgent
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.events import Event, EventType
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import CallbackManager, Command, GraphLifecycleHook
from tenxgraph.utils.constants import END

from .tool_guard_helpers import MODES, RecordingPublisher, run


PICK_COLOR = {
    "name": "pick_color",
    "description": "Ask the user to pick a color.",
    "parameters": {"type": "object", "properties": {}},
}


class InlineRecorder(RecordingPublisher):
    inline = True

    def publish_inline(self, event):
        self.events.append(event.model_copy(deep=True))


def local_tool() -> str:
    return "server"


def _client_tool_graph(manager: CallbackManager | None = None):
    """MODEL asks for the client tool ``pick_color``; after its result MODEL answers."""

    async def model(state: AgentState):
        last = state.context[-1]
        if last.role == "tool":
            return Message.text_message(f"got {last.text()}", role="assistant")
        return Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="pick_color", args={})],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "pick_color", "arguments": "{}"},
                }
            ],
        )

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([local_tool]))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    return graph.compile(
        checkpointer=InMemoryCheckpointer(), callback_manager=manager or CallbackManager()
    )


async def _messages(app, mode, data, config) -> list[Message]:
    """The run's messages as the caller sees them: the response (invoke) or MESSAGE chunks."""
    if mode == "ainvoke":
        return (await app.ainvoke(data, config=dict(config)))["messages"]
    chunks = [c async for c in app.astream(data, config=dict(config))]
    return [c.message for c in chunks if c.message is not None and not c.message.delta]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_on_state_update_fires_on_a_client_tool_pause(mode):
    """T1: once for the tool node, before on_interrupt; its state is the one the pause saves."""
    calls: list[tuple[str, str]] = []

    class Hook(GraphLifecycleHook):
        async def on_state_update(self, context, node_name, old_state, new_state, step):
            calls.append(("state_update", node_name))
            if node_name != "TOOL":
                return None
            changed = new_state.model_copy(deep=True)
            changed.context_summary = "set by on_state_update"
            return changed

        async def on_interrupt(self, context, interrupted_node, interrupt_type, state):
            calls.append(("interrupt", interrupt_type))

    manager = CallbackManager()
    manager.register_lifecycle_hook(Hook())
    app = _client_tool_graph(manager)
    config = {"thread_id": f"t1-{mode}", "remote_tools": [PICK_COLOR]}
    await run(app, mode, dict(config))

    tool_steps = [c for c in calls if c == ("state_update", "TOOL")]
    assert tool_steps == [("state_update", "TOOL")]
    assert calls.index(("state_update", "TOOL")) < calls.index(("interrupt", "remote_tool"))
    saved = await app._checkpointer.aget_state(config)
    assert saved.is_interrupted()
    assert saved.context_summary == "set by on_state_update"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_client_tool_resume_records_the_tool_result_not_the_old_user_message(mode):
    """T3: the resume's messages and the saved history hold the client's tool result once."""
    app = _client_tool_graph()
    config = {"thread_id": f"t3-{mode}", "remote_tools": [PICK_COLOR]}
    first = [Message.text_message("pick"), Message.text_message("a calm one")]
    first_messages = await _messages(app, mode, {"messages": first}, config)
    assert [m.message_id for m in first_messages[:2]] == [m.message_id for m in first]

    result = Message(role="tool", content=[ToolResultBlock(call_id="c1", output="teal")])
    resumed = await _messages(app, mode, {"messages": [result]}, config)
    assert result.message_id in [m.message_id for m in resumed]
    assert not {m.message_id for m in first} & {m.message_id for m in resumed}
    assert [m.text() for m in resumed if m.role == "assistant"][-1] == "got teal"

    stored = await app._checkpointer.alist_messages(config)
    ids = [m.message_id for m in stored]
    assert ids.count(result.message_id) == 1
    assert all(ids.count(m.message_id) == 1 for m in first)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_command_goto_routes_through_the_full_loop(mode):
    """T4: Command(update=..., goto=...) from a function node beats the static edge."""
    ran: list[str] = []

    def first(state: AgentState):
        ran.append("FIRST")
        return Command(update=Message.text_message("first", role="assistant"), goto="THIRD")

    def second(state: AgentState):
        ran.append("SECOND")
        return Message.text_message("second", role="assistant")

    def third(state: AgentState):
        ran.append("THIRD")
        return Message.text_message("third", role="assistant")

    graph = StateGraph()
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_node("THIRD", third)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.add_edge("THIRD", END)
    graph.set_entry_point("FIRST")
    app = graph.compile(checkpointer=InMemoryCheckpointer())

    messages = await _messages(
        app, mode, {"messages": [Message.text_message("go")]}, {"thread_id": f"t4-{mode}"}
    )
    assert ran == ["FIRST", "THIRD"]
    assert [m.text() for m in messages if m.role == "assistant"] == ["first", "third"]


def _stop_graph(where: str, publisher):
    """FIRST -> SECOND; a stop arrives after FIRST, while FIRST runs, or between the nodes."""
    holder: dict = {}

    async def first(state: AgentState, config: dict):
        if where in ("after", "mid"):
            await holder["app"].astop({"thread_id": config["thread_id"]})
        if where == "mid":
            await asyncio.sleep(30)
        return Message.text_message("first", role="assistant")

    def second(state: AgentState):
        return Message.text_message("second", role="assistant")

    class StopOnFirstSave(GraphLifecycleHook):
        """Asks for a stop from the first per-step save: seen at the next loop step."""

        done = False

        async def on_checkpoint(self, context, state, messages, is_context_trimmed):
            if where == "between" and not self.done and state.execution_meta.step >= 2:
                self.done = True
                await holder["app"]._checkpointer.arequest_stop(context.config)

    manager = CallbackManager()
    manager.register_lifecycle_hook(StopOnFirstSave())
    graph = StateGraph(publisher=publisher)
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.set_entry_point("FIRST")
    holder["app"] = graph.compile(checkpointer=InMemoryCheckpointer(), callback_manager=manager)
    return holder["app"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("where", "node"), [("after", "FIRST"), ("mid", "FIRST"), ("between", "SECOND")]
)
async def test_a_stop_sends_one_graph_interrupted(where, node, mode):
    """T6: one GRAPH_EXECUTION INTERRUPTED per stop, naming the stopped node."""
    publisher = InlineRecorder()
    app = _stop_graph(where, publisher)
    config = {"thread_id": f"t6-{where}-{mode}"}
    await run(app, mode, config)

    interrupted = [
        e
        for e in publisher.events
        if e.event == Event.GRAPH_EXECUTION and e.event_type == EventType.INTERRUPTED
    ]
    assert [e.node_name for e in interrupted] == [node]
    assert interrupted[0].metadata["interrupt_type"] == "stop"
    saved = await app._checkpointer.aget_state(config)
    assert saved.is_interrupted()
    assert "second" not in [m.text() for m in saved.context]


@pytest.mark.asyncio
async def test_ainvoke_builds_no_stream_chunk_and_calls_the_model_without_streaming():
    """T8: invoke does not pay for streaming."""

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    def prepare(state: AgentState):
        return Message.text_message("ready", role="assistant")

    graph = StateGraph()
    graph.add_node("PREP", prepare)
    graph.add_node("MAIN", TestAgent(responses=["hi"]))
    graph.add_node("TOOL", ToolNode([local_tool]))
    graph.add_edge("PREP", "MAIN")
    graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("PREP")
    app = graph.compile(checkpointer=InMemoryCheckpointer())

    built = {"chunks": 0, "stream": 0}
    chunk_init = StreamChunk.__init__
    converter_stream = ModelResponseConverter.stream

    def count_chunk(self, *args, **kwargs):
        built["chunks"] += 1
        chunk_init(self, *args, **kwargs)

    def count_stream(self, *args, **kwargs):
        built["stream"] += 1
        return converter_stream(self, *args, **kwargs)

    with (
        patch.object(StreamChunk, "__init__", count_chunk),
        patch.object(ModelResponseConverter, "stream", count_stream),
    ):
        result = await app.ainvoke({"messages": [Message.text_message("go")]}, {"thread_id": "t8"})
        assert built == {"chunks": 0, "stream": 0}
        assert [m.text() for m in result["messages"] if m.role == "assistant"][-1] == "hi"

        chunks = [
            c
            async for c in app.astream(
                {"messages": [Message.text_message("go")]}, {"thread_id": "t8s"}
            )
        ]
        assert chunks
        assert built["chunks"] > 0
        assert built["stream"] == 1
