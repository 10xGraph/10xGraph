"""Client tools supplied per run through ``config["remote_tools"]``."""

import json

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.graph.tool_node.run_tools import run_remote_tools
from tenxgraph.core.state import AgentState, Message, ToolCallBlock
from tenxgraph.core.state.message_block import RemoteToolCallBlock
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


PICK_COLOR = {
    "name": "pick_color",
    "description": "Ask the user to pick a color.",
    "parameters": {"type": "object", "properties": {"hint": {"type": "string"}}},
}


def local_tool() -> str:
    """A server tool."""
    return "server"


def test_flat_and_openai_shapes_are_normalized():
    tools = run_remote_tools(
        {
            "remote_tools": [
                PICK_COLOR,
                {"type": "function", "function": {"name": "ping", "description": "Ping."}},
                {"description": "no name"},
                PICK_COLOR,  # duplicate
            ]
        }
    )
    assert [t["function"]["name"] for t in tools] == ["pick_color", "ping"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"]["properties"]["hint"] == {"type": "string"}
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_no_config_means_no_run_tools():
    assert run_remote_tools(None) == []
    assert run_remote_tools({"remote_tools": "nope"}) == []


@pytest.mark.asyncio
async def test_tool_node_lists_run_tools_for_that_run_only():
    node = ToolNode([local_tool])
    with_run = await node.all_tools(config={"remote_tools": [PICK_COLOR]})
    without = await node.all_tools()
    assert "pick_color" in [t["function"]["name"] for t in with_run]
    assert "pick_color" not in [t["function"]["name"] for t in without]
    assert node.all_tools_sync(config={"remote_tools": [PICK_COLOR]})[-1]["function"]["name"] == (
        "pick_color"
    )


@pytest.mark.asyncio
async def test_run_tool_cannot_shadow_a_server_tool():
    node = ToolNode([local_tool])
    tools = await node.all_tools(config={"remote_tools": [{"name": "local_tool"}]})
    assert [t["function"]["name"] for t in tools].count("local_tool") == 1

    result = await node.invoke(
        "local_tool",
        {},
        tool_call_id="c1",
        config={"remote_tools": [{"name": "local_tool"}]},
        state=AgentState(),
    )
    # Still executed on the server.
    assert not any(isinstance(b, RemoteToolCallBlock) for b in result.content)


@pytest.mark.asyncio
async def test_call_to_a_run_tool_is_handed_to_the_client():
    node = ToolNode([local_tool])
    result = await node.invoke(
        "pick_color",
        {"hint": "calm"},
        tool_call_id="c1",
        config={"remote_tools": [PICK_COLOR]},
        state=AgentState(),
    )
    [block] = result.content
    assert isinstance(block, RemoteToolCallBlock)
    assert block.name == "pick_color"
    assert result.metadata.get("is_remote") is True


def _graph():
    async def model(state: AgentState):
        last = state.context[-1]
        if last.role == "tool":
            return Message.text_message(f"got {last.text()}", role="assistant")
        call = {"id": "c1", "name": "pick_color", "args": {"hint": "calm"}}
        return Message(
            role="assistant",
            content=[ToolCallBlock(**call)],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "pick_color", "arguments": json.dumps(call["args"])},
                }
            ],
        )

    def route(state: AgentState) -> str:
        last = state.context[-1]
        if last.role == "assistant" and last.tools_calls:
            return "TOOL"
        return END

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([local_tool]))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    return graph.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.asyncio
async def test_graph_pauses_for_a_run_tool_and_resumes_with_its_result():
    app = _graph()
    config = {"thread_id": "t1", "remote_tools": [PICK_COLOR]}
    first = await app.ainvoke(
        {"messages": [Message.text_message("pick")]},
        config=dict(config),
        response_granularity=ResponseGranularity.FULL,
    )
    assert first["state"].is_interrupted()

    from tenxgraph.core.state import ToolResultBlock

    result = Message(role="tool", content=[ToolResultBlock(call_id="c1", output="teal")])
    second = await app.ainvoke(
        {"messages": [result]},
        config=dict(config),
        response_granularity=ResponseGranularity.FULL,
    )
    texts = [m.text() for m in second["messages"] if m.role == "assistant"]
    assert texts[-1] == "got teal"


@pytest.mark.asyncio
async def test_stream_pauses_for_a_run_tool_and_resumes():
    from tenxgraph.core.state import StreamEvent, ToolResultBlock

    app = _graph()
    config = {"thread_id": "t-stream", "remote_tools": [PICK_COLOR]}
    first = [
        c
        async for c in app.astream(
            {"messages": [Message.text_message("pick")]},
            config=dict(config),
            response_granularity=ResponseGranularity.FULL,
        )
    ]
    reasons = [(c.data or {}).get("reason") for c in first if c.event == StreamEvent.UPDATES]
    assert "Remote tool call - graph interrupted" in reasons
    # The model did not run again with the placeholder as a tool result.
    assert not any(
        c.message and c.message.role == "assistant" and "got" in c.message.text()
        for c in first
        if c.event == StreamEvent.MESSAGE
    )

    result = Message(role="tool", content=[ToolResultBlock(call_id="c1", output="teal")])
    texts = [
        c.message.text()
        async for c in app.astream({"messages": [result]}, config=dict(config))
        if c.event == StreamEvent.MESSAGE and c.message and c.message.role == "assistant"
    ]
    assert texts[-1] == "got teal"


@pytest.mark.asyncio
async def test_parallel_server_tool_result_is_kept_while_waiting_for_the_client():
    from tenxgraph.core.state import ToolResultBlock

    async def model(state: AgentState):
        last = state.context[-1]
        if last.role == "tool":
            done = sorted(m.text() for m in state.context if m.role == "tool")
            return Message.text_message("|".join(done), role="assistant")
        calls = [
            {"id": "a", "name": "local_tool", "args": {}},
            {"id": "b", "name": "pick_color", "args": {}},
        ]
        return Message(
            role="assistant",
            content=[ToolCallBlock(**c) for c in calls],
            tools_calls=[
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": "{}"},
                }
                for c in calls
            ],
        )

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([local_tool]))
    graph.add_conditional_edges(
        "MODEL",
        lambda s: "TOOL"
        if s.context[-1].role == "assistant" and s.context[-1].tools_calls
        else END,
        {"TOOL": "TOOL", END: END},
    )
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    app = graph.compile(checkpointer=InMemoryCheckpointer())

    config = {"thread_id": "t-par", "remote_tools": [PICK_COLOR]}
    await app.ainvoke({"messages": [Message.text_message("go")]}, config=dict(config))
    result = Message(role="tool", content=[ToolResultBlock(call_id="b", output="teal")])
    final = await app.ainvoke({"messages": [result]}, config=dict(config))
    assert [m.text() for m in final["messages"] if m.role == "assistant"][-1] == "server|teal"
