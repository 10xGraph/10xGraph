"""A deploy that renames a node must not wedge threads stored at it (G-18).

Without an alias the run fails with a typed error that names the node and the thread. With
``node_aliases`` at compile the thread resumes at the new node.
"""

import pytest

from tenxgraph.core.exceptions import GraphError, StoredNodeMissingError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.utils.handler_utils import resolve_stored_node
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END, START


def _graph(second_name: str, holder: dict, **compile_kwargs):
    checkpointer = holder["cp"]

    async def first(state: AgentState, config: dict):
        if holder.get("stop"):
            holder["stop"] = False
            await holder["app"].astop({"thread_id": config["thread_id"]})
        return Message.text_message("first", role="assistant")

    async def second(state: AgentState):
        return Message.text_message("second", role="assistant")

    graph = StateGraph()
    graph.add_node("FIRST", first)
    graph.add_node(second_name, second)
    graph.add_edge("FIRST", second_name)
    graph.add_edge(second_name, END)
    graph.set_entry_point("FIRST")
    app = graph.compile(checkpointer=checkpointer, **compile_kwargs)
    holder["app"] = app
    return app


async def _store_thread_at_old_node(thread: str) -> dict:
    holder = {"cp": InMemoryCheckpointer(), "stop": True}
    old = _graph("OLD_SECOND", holder)
    await old.ainvoke({"messages": [Message.text_message("hi")]}, {"thread_id": thread})
    state = await holder["cp"].aget_state({"thread_id": thread})
    assert state.execution_meta.current_node == "OLD_SECOND"
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    return holder


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_renamed_node_with_alias_resumes(mode):
    thread = f"t-alias-{mode}"
    holder = await _store_thread_at_old_node(thread)
    new = _graph("NEW_SECOND", holder, node_aliases={"OLD_SECOND": "NEW_SECOND"})

    config = {"thread_id": thread}
    if mode == "ainvoke":
        await new.ainvoke({}, config)
    else:
        _ = [c async for c in new.astream({}, config)]

    state = await holder["cp"].aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert "second" in [m.text() for m in state.context]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_renamed_node_without_alias_raises_typed_error(mode):
    thread = f"t-noalias-{mode}"
    holder = await _store_thread_at_old_node(thread)
    new = _graph("NEW_SECOND", holder)

    config = {"thread_id": thread}
    with pytest.raises(StoredNodeMissingError) as exc:
        if mode == "ainvoke":
            await new.ainvoke({}, config)
        else:
            _ = [c async for c in new.astream({}, config)]

    err = exc.value
    assert isinstance(err, GraphError)
    assert err.error_code == "GRAPH_NODE_MISSING_001"
    assert err.node_name == "OLD_SECOND"
    assert err.thread_id == thread
    assert "OLD_SECOND" in str(err) and thread in str(err) and "node_aliases" in str(err)
    assert not isinstance(err, KeyError)


def test_resolve_stored_node_rules():
    nodes = {START: 1, END: 1, "NEW": 1}
    assert resolve_stored_node("NEW", nodes, None, "t") == "NEW"
    assert resolve_stored_node("OLD", nodes, {"OLD": "NEW"}, "t") == "NEW"
    # chained alias
    assert resolve_stored_node("V1", nodes, {"V1": "OLD", "OLD": "NEW"}, "t") == "NEW"
    # alias to a node that does not exist, and an alias cycle, both fail typed
    with pytest.raises(StoredNodeMissingError):
        resolve_stored_node("OLD", nodes, {"OLD": "GONE"}, "t")
    with pytest.raises(StoredNodeMissingError):
        resolve_stored_node("A", nodes, {"A": "B", "B": "A"}, "t")
