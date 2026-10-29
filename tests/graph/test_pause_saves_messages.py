"""A paused run saves the messages it produced (G-06).

Streamed runs that paused (interrupt_before/after, interrupt(), client tool) saved
``messages=[]``, so the messages table had permanent holes.
"""

import json

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message, ToolCallBlock
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity, interrupt
from tenxgraph.utils.constants import END


FULL = ResponseGranularity.FULL
MODES = ["ainvoke", "astream"]


async def run(app, mode, data, config):
    if mode == "ainvoke":
        return await app.ainvoke(data, config, response_granularity=FULL)
    return [c async for c in app.astream(data, config, response_granularity=FULL)]


def two_nodes(checkpointer, **compile_kwargs):
    async def first(state: AgentState):
        return Message.text_message("from first", role="assistant")

    async def second(state: AgentState):
        return Message.text_message("from second", role="assistant")

    graph = StateGraph()
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.set_entry_point("FIRST")
    return graph.compile(checkpointer=checkpointer, **compile_kwargs)


async def stored_texts(cp, config):
    return [m.text() for m in await cp.alist_messages(config)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_interrupt_before_saves_the_messages_so_far(mode):
    cp = InMemoryCheckpointer()
    app = two_nodes(cp, interrupt_before=["SECOND"])
    config = {"thread_id": f"t-before-{mode}"}

    await run(app, mode, {"messages": [Message.text_message("hi")]}, config)

    assert await stored_texts(cp, config) == ["hi", "from first"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_interrupt_before_the_first_node_saves_the_user_message(mode):
    cp = InMemoryCheckpointer()
    app = two_nodes(cp, interrupt_before=["FIRST"])
    config = {"thread_id": f"t-before-first-{mode}"}

    await run(app, mode, {"messages": [Message.text_message("hi")]}, config)

    assert await stored_texts(cp, config) == ["hi"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_interrupt_after_saves_the_messages_so_far(mode):
    cp = InMemoryCheckpointer()
    app = two_nodes(cp, interrupt_after=["FIRST"])
    config = {"thread_id": f"t-after-{mode}"}

    await run(app, mode, {"messages": [Message.text_message("hi")]}, config)

    assert await stored_texts(cp, config) == ["hi", "from first"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_interrupt_call_saves_the_user_message(mode):
    async def ask(state: AgentState):
        interrupt("approve?")
        return Message.text_message("approved", role="assistant")

    graph = StateGraph()
    graph.add_node("ASK", ask)
    graph.add_edge("ASK", END)
    graph.set_entry_point("ASK")
    cp = InMemoryCheckpointer()
    app = graph.compile(checkpointer=cp)
    config = {"thread_id": f"t-interrupt-{mode}"}

    await run(app, mode, {"messages": [Message.text_message("hi")]}, config)

    assert await stored_texts(cp, config) == ["hi"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_client_tool_pause_saves_the_tool_call_message(mode):
    async def model(state: AgentState):
        call = {"id": "c1", "name": "pick_color", "args": {}}
        return Message(
            role="assistant",
            content=[ToolCallBlock(**call)],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "pick_color", "arguments": json.dumps({})},
                }
            ],
        )

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([]))
    graph.add_edge("MODEL", "TOOL")
    graph.add_edge("TOOL", END)
    graph.set_entry_point("MODEL")
    cp = InMemoryCheckpointer()
    app = graph.compile(checkpointer=cp)
    config = {
        "thread_id": f"t-remote-{mode}",
        "remote_tools": [{"name": "pick_color", "description": "Pick."}],
    }

    await run(app, mode, {"messages": [Message.text_message("hi")]}, config)

    stored = await cp.alist_messages(config)
    assert "hi" in [m.text() for m in stored]
    assert any(m.tools_calls for m in stored)
