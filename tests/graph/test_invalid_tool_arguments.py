"""A tool call with unusable arguments gets a tool error and never runs (B7)."""

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from tenxgraph.core.graph.utils.stream_node_handler import StreamNodeHandler
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.core.state.stream_chunks import StreamEvent
from tenxgraph.runtime.adapters.llm.tool_arguments import INVALID_TOOL_ARGUMENTS_MESSAGE
from tenxgraph.utils.constants import END


BAD_ARGS = ['{"a": ', "[]", '"str"', "42"]


def _make_graph(calls_log: list, arguments: str, name: str = "optional_tool"):
    def optional_tool(a: int = 0) -> str:
        """All-optional parameters: running it with {} would succeed."""
        calls_log.append(a)
        return "ran"

    assistant = Message.text_message("calling", role="assistant")
    assistant.tools_calls = [
        {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}
    ]

    def model(state: AgentState):
        return assistant.model_copy(deep=True, update={"message_id": "assistant_1"})

    graph = StateGraph()
    graph.add_node("MAIN", model)
    graph.add_node("TOOL", ToolNode([optional_tool]))
    graph.add_edge("MAIN", "TOOL")
    graph.add_edge("TOOL", END)
    graph.set_entry_point("MAIN")
    return graph


def _outputs(messages) -> list[str]:
    return [
        b.output
        for m in messages
        if m.role == "tool"
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]


@pytest.mark.parametrize("arguments", BAD_ARGS)
@pytest.mark.asyncio
async def test_invoke_returns_error_and_does_not_run_tool(arguments):
    log: list = []
    compiled = _make_graph(log, arguments).compile()
    result = await compiled.ainvoke({"messages": [Message.text_message("hi")]})
    assert log == []
    assert _outputs(result["messages"]) == [INVALID_TOOL_ARGUMENTS_MESSAGE]
    tool_msg = next(m for m in result["messages"] if m.role == "tool")
    assert tool_msg.content[0].is_error is True
    assert tool_msg.content[0].call_id == "call_1"


@pytest.mark.parametrize("arguments", BAD_ARGS)
@pytest.mark.asyncio
async def test_stream_returns_error_and_does_not_run_tool(arguments):
    log: list = []

    def optional_tool(a: int = 0) -> str:
        log.append(a)
        return "ran"

    msg = Message.text_message("calling", role="assistant")
    msg.tools_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "optional_tool", "arguments": arguments},
        }
    ]
    handler = StreamNodeHandler("TOOL", ToolNode([optional_tool]))
    items = [
        i async for i in handler._call_tools(msg, AgentState(context=[msg]), {"thread_id": "t"})
    ]
    assert log == []
    messages = [i for i in items if isinstance(i, Message)]
    assert _outputs(messages) == [INVALID_TOOL_ARGUMENTS_MESSAGE]
    errors = [i for i in items if getattr(i, "event", None) == StreamEvent.ERROR]
    assert len(errors) == 1
    assert errors[0].data["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_graph_astream_does_not_run_tool_on_invalid_arguments():
    log: list = []
    compiled = _make_graph(log, '{"a": ').compile()
    async for _ in compiled.astream(
        {"messages": [Message.text_message("hi")]}, config={"thread_id": "t"}
    ):
        pass
    assert log == []


@pytest.mark.parametrize("arguments", ["", "   ", "{}", '{"a": 3}'])
@pytest.mark.asyncio
async def test_valid_and_empty_arguments_still_run(arguments):
    log: list = []
    compiled = _make_graph(log, arguments).compile()
    result = await compiled.ainvoke({"messages": [Message.text_message("hi")]})
    assert len(log) == 1
    assert "ran" in " ".join(_outputs(result["messages"]))


@pytest.mark.asyncio
async def test_invalid_arguments_skip_ledger_and_tool():
    log: list = []
    node = ToolNode([lambda: log.append(1)])
    handler = InvokeNodeHandler("TOOL", node)
    out = await handler._handle_single_tool(
        {"id": "c", "type": "function", "function": {"name": "x", "arguments": '{"a": '}},
        AgentState(),
        {"thread_id": "t"},
    )
    assert isinstance(out, Message)
    assert log == []
