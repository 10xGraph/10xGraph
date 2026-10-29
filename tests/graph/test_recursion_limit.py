"""Recursion limit (G-17): exact limit, usable next turn, same error on both paths."""

import pytest

from tenxgraph.core.exceptions import GraphRecursionError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


FULL = ResponseGranularity.FULL
MODES = ["ainvoke", "astream"]


async def run(app, mode, text, config):
    data = {"messages": [Message.text_message(text)]}
    if mode == "ainvoke":
        return await app.ainvoke(data, config, response_granularity=FULL)
    return [c async for c in app.astream(data, config, response_granularity=FULL)]


def two_node_graph(checkpointer=None):
    async def first(state: AgentState):
        return Message.text_message("one", role="assistant")

    async def second(state: AgentState):
        return Message.text_message("two", role="assistant")

    graph = StateGraph()
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.set_entry_point("FIRST")
    return graph.compile(checkpointer=checkpointer)


def loop_graph(checkpointer):
    async def work(state: AgentState):
        return Message.text_message("worked", role="assistant")

    def route(state: AgentState) -> str:
        user = [m for m in state.context if m.role == "user"][-1]
        return "WORK" if user.text() == "loop" else END

    graph = StateGraph()
    graph.add_node("WORK", work)
    graph.add_conditional_edges("WORK", route, {"WORK": "WORK", END: END})
    graph.set_entry_point("WORK")
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_graph_that_fits_the_limit_finishes(mode):
    # START, FIRST and SECOND: three steps. A limit of 3 is enough.
    app = two_node_graph(InMemoryCheckpointer())
    await run(app, mode, "hi", {"thread_id": f"t-fit-{mode}", "recursion_limit": 3})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_graph_over_the_limit_raises_graph_recursion_error(mode):
    app = two_node_graph(InMemoryCheckpointer())
    with pytest.raises(GraphRecursionError):
        await run(app, mode, "hi", {"thread_id": f"t-over-{mode}", "recursion_limit": 2})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_the_turn_after_a_recursion_error_gets_a_reply(mode):
    cp = InMemoryCheckpointer()
    app = loop_graph(cp)
    config = {"thread_id": f"t-next-{mode}", "recursion_limit": 4}

    with pytest.raises(GraphRecursionError):
        await run(app, mode, "loop", config)

    await run(app, mode, "hello", config)

    state = await cp.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert [m.text() for m in state.context][-2:] == ["hello", "worked"]
