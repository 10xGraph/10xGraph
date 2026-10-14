"""A state written with ``aput_state`` is the state the next run starts from.

A run loads its state from the checkpointer's cache first (``aget_state_cache``) and only falls
back to the durable store on a miss. ``aput_state`` wrote only the durable store, so an update
made between runs (an app applying an operator's decision, a thread-state API call) left the
cache on the previous state: the next run never saw it, and wrote the old state back over it.
On ``PgCheckpointer`` the stale cache also carried the old version, so the next run's first
write raised ``StaleStateError`` instead (see ``tests/integration/test_pg_checkpointer_real.py``).
"""

from __future__ import annotations

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from tenxgraph.utils import END


class StageState(AgentState):
    stage: str = "intro"


def _in_memory():
    return InMemoryCheckpointer()


def _sqlite():
    cp = SqliteCheckpointer(":memory:")
    cp.bind_state_type(StageState)
    return cp


CHECKPOINTERS = [pytest.param(_in_memory, id="in-memory"), pytest.param(_sqlite, id="sqlite")]


def _graph(checkpointer, seen: list[str]):
    """One node that records the stage it was run on."""

    def node(state: StageState) -> Message:
        seen.append(state.stage)
        return Message.text_message("ok", "assistant")

    graph = StateGraph(StageState())
    graph.add_node("main", node)
    graph.set_entry_point("main")
    graph.add_edge("main", END)
    return graph.compile(checkpointer=checkpointer)


def _turn(text: str) -> dict:
    return {"messages": [Message.text_message(text, "user")]}


@pytest.mark.asyncio
@pytest.mark.parametrize("make_checkpointer", CHECKPOINTERS)
async def test_the_next_run_sees_a_state_written_between_runs(make_checkpointer):
    checkpointer = make_checkpointer()
    seen: list[str] = []
    app = _graph(checkpointer, seen)
    config = {"thread_id": "t-write-through", "user_id": "u1"}

    await app.ainvoke(_turn("first"), config=dict(config))

    # A new object, as an API or an app writes it: an in-place edit of the object the
    # in-memory checkpointer handed out would also edit its cached copy and hide the bug.
    state = (await checkpointer.aget_state(dict(config))).model_copy(update={"stage": "coding"})
    await checkpointer.aput_state(dict(config), state)

    await app.ainvoke(_turn("second"), config=dict(config))

    assert seen == ["intro", "coding"]
    assert (await checkpointer.aget_state(dict(config))).stage == "coding"


@pytest.mark.asyncio
@pytest.mark.parametrize("make_checkpointer", CHECKPOINTERS)
async def test_aput_state_leaves_the_cache_matching_the_store(make_checkpointer):
    checkpointer = make_checkpointer()
    config = {"thread_id": "t-cache", "user_id": "u1"}

    await checkpointer.aput_state_cache(dict(config), StageState(stage="intro"))
    await checkpointer.aput_state(dict(config), StageState(stage="closing"))

    cached = await checkpointer.aget_state_cache(dict(config))
    assert cached is not None and cached.stage == "closing"
