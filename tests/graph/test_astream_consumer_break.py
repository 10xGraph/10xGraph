"""A consumer that stops reading ``astream`` early (``break``, so the generator is closed).

This is not a cancel: the generator gets ``GeneratorExit`` at its suspended ``yield``. The
thread must not stay "running" and no tool or background task may be left behind.
"""

import asyncio
import gc

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.stream_emitter import StreamEmitter
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END

from .tool_guard_helpers import assistant_call


def _tasks_left(before: set[asyncio.Task]) -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() - before if not t.done()]


async def _two_node_graph(checkpointer):
    async def first(state: AgentState):
        return Message.text_message("first", role="assistant")

    async def second(state: AgentState):
        await asyncio.sleep(0.2)
        return Message.text_message("second", role="assistant")

    graph = StateGraph()
    graph.add_node("A", first)
    graph.add_node("B", second)
    graph.set_entry_point("A")
    graph.add_edge("A", "B")
    graph.add_edge("B", END)
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["aclose", "break_and_drop"])
async def test_break_between_nodes_leaves_a_usable_thread(how):
    cp = InMemoryCheckpointer()
    app = await _two_node_graph(cp)
    config = {"thread_id": "t-break"}
    before = set(asyncio.all_tasks())

    gen = app.astream(
        {"messages": [Message.text_message("go", role="user")]},
        config,
        response_granularity=ResponseGranularity.FULL,
    )
    async for _chunk in gen:
        break
    if how == "aclose":
        await gen.aclose()
    else:
        del gen
    await asyncio.sleep(0.5)

    # Nothing was saved yet (the first chunk comes before the first checkpoint), so the thread
    # is simply new; it must not read as running.
    state = await cp.aget_state(config)
    status = state.execution_meta.status if state else None
    assert "running" not in str(status).lower(), f"thread left as {status}"
    assert await cp.aget_active_run(config) is None
    assert _tasks_left(before) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["aclose", "drop"])
async def test_break_while_a_tool_runs_leaves_no_task(how):
    started = asyncio.Event()
    finished = []

    async def slow_tool(emit: StreamEmitter | None = None, tool_call_id: str | None = None):
        if emit is not None:
            emit.progress("working")
        started.set()
        await asyncio.sleep(0.5)
        finished.append(True)
        return "done"

    async def model(state: AgentState):
        return assistant_call([("c1", "slow_tool")])

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([slow_tool]))
    graph.set_entry_point("MODEL")
    graph.add_edge("MODEL", "TOOL")
    graph.add_edge("TOOL", END)
    cp = InMemoryCheckpointer()
    app = graph.compile(checkpointer=cp)
    config = {"thread_id": "t-break-tool"}
    before = set(asyncio.all_tasks())

    gen = app.astream(
        {"messages": [Message.text_message("go", role="user")]},
        config,
        response_granularity=ResponseGranularity.FULL,
    )
    async for chunk in gen:
        if started.is_set() or "working" in str(chunk.data):
            break
    if how == "aclose":
        await gen.aclose()
    else:
        del gen, chunk
        gc.collect()
    await asyncio.sleep(0.8)

    state = await cp.aget_state(config)
    status = state.execution_meta.status if state else None
    leaked = _tasks_left(before)
    active = await cp.aget_active_run(config)
    assert not finished, "the tool should have been cancelled"
    assert active is None, f"active run marker left: {active}"
    assert leaked == [], f"tasks left: {leaked}"
    assert "running" not in str(status).lower(), f"thread left as {status}"
    assert active is None, f"active run marker left: {active}"
    assert leaked == [], f"tasks left: {leaked}"
