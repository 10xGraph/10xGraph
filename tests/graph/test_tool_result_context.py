"""A tool that returns ``ToolResult`` must put its message in ``state.context`` on every path.

On invoke the message used to be returned to the caller but never added to the context, so the
model never saw the tool result, called the tool again, and the saved thread held a tool call
with no answer.
"""

import json

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message, ToolCallBlock, ToolResult
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


class FlagState(AgentState):
    flag: str = ""


def _graph(seen: list[list[str]]):
    def set_flag(tool_call_id: str | None = None) -> ToolResult:
        return ToolResult(message="flag is set", state={"flag": "on"})

    async def model(state: FlagState):
        outputs = [m.text() for m in state.context if m.role == "tool"]
        seen.append(outputs)
        if outputs:
            return Message.text_message(f"saw: {outputs[-1]}", role="assistant")
        return Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="set_flag", args={})],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "set_flag", "arguments": json.dumps({})},
                }
            ],
        )

    def route(state: FlagState) -> str:
        last = state.context[-1]
        if last.role == "assistant" and last.tools_calls:
            return "TOOL"
        return END

    graph = StateGraph(FlagState())
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([set_flag]))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    return graph.compile(checkpointer=InMemoryCheckpointer())


def _answered(state: AgentState) -> bool:
    calls = {c["id"] for m in state.context for c in (m.tools_calls or [])}
    done = {
        b.call_id
        for m in state.context
        if m.role == "tool"
        for b in m.content
        if getattr(b, "type", "") == "tool_result"
    }
    return calls <= done


@pytest.mark.asyncio
async def test_invoke_puts_tool_result_message_in_context():
    seen: list[list[str]] = []
    app = _graph(seen)
    result = await app.ainvoke(
        {"messages": [Message.text_message("go")]},
        {"thread_id": "t-invoke"},
        response_granularity=ResponseGranularity.FULL,
    )
    state = result["state"]
    assert seen == [[], ["flag is set"]]
    assert state.flag == "on"
    assert [m.text() for m in state.context if m.role == "tool"] == ["flag is set"]
    assert _answered(state)
    assert state.context[-1].text() == "saw: flag is set"


@pytest.mark.asyncio
async def test_stream_puts_tool_result_message_in_context():
    seen: list[list[str]] = []
    app = _graph(seen)
    _ = [
        c
        async for c in app.astream(
            {"messages": [Message.text_message("go")]},
            {"thread_id": "t-stream"},
            response_granularity=ResponseGranularity.FULL,
        )
    ]
    state = await app._checkpointer.aget_state({"thread_id": "t-stream"})
    assert seen == [[], ["flag is set"]]
    assert state.flag == "on"
    assert [m.text() for m in state.context if m.role == "tool"] == ["flag is set"]
    assert _answered(state)
