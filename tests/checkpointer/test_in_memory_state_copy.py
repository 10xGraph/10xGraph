"""The in-memory checkpointer hands out copies of the stored state (audit B9)."""

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.interrupt import interrupt


@pytest.mark.asyncio
async def test_mutating_a_loaded_state_does_not_change_the_store():
    cp = InMemoryCheckpointer()
    cfg = {"thread_id": "t"}
    await cp.aput_state(cfg, AgentState(context=[Message.text_message("one")]))

    loaded = await cp.aget_state(cfg)
    loaded.context.append(Message.text_message("two"))
    assert len((await cp.aget_state(cfg)).context) == 1

    cached = await cp.aget_state_cache(cfg)
    cached.context.append(Message.text_message("three"))
    assert len((await cp.aget_state_cache(cfg)).context) == 1


@pytest.mark.asyncio
async def test_mutating_the_saved_object_does_not_change_the_store():
    cp = InMemoryCheckpointer()
    cfg = {"thread_id": "t"}
    state = AgentState(context=[Message.text_message("one")])
    await cp.aput_state(cfg, state)
    state.context.append(Message.text_message("later"))
    assert len((await cp.aget_state(cfg)).context) == 1


@pytest.mark.asyncio
async def test_rejected_request_on_a_paused_thread_leaves_stored_state_alone():
    async def ask(state: AgentState):
        interrupt("approve?")
        return Message.text_message("done", role="assistant")

    graph = StateGraph()
    graph.add_node("ASK", ask)
    graph.add_edge("ASK", END)
    graph.set_entry_point("ASK")
    cp = InMemoryCheckpointer()
    app = graph.compile(checkpointer=cp)
    cfg = {"thread_id": "paused"}

    await app.ainvoke({"messages": [Message.text_message("go")]}, cfg)
    with pytest.raises(ValueError, match="paused"):
        await app.ainvoke({"messages": [Message.text_message("INJECTED")]}, cfg)

    for stored in (await cp.aget_state(cfg), await cp.aget_state_cache(cfg)):
        texts = [m.text() for m in stored.context]
        assert "INJECTED" not in texts
        assert "go" in texts
