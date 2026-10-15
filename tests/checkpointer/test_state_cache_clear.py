"""Clearing a thread's state, or deleting the thread, also drops its cached state.

A run loads its state from the checkpointer's cache first (``aget_state_cache``) and only falls
back to the durable store on a miss. ``aclear_state`` on ``InMemoryCheckpointer`` and
``SqliteCheckpointer`` deleted only the durable row, so the next run picked the cleared state
back up from the cache and wrote it back. ``InMemoryCheckpointer.aclean_thread`` deleted only
the thread record, leaving its state, cache and messages behind. ``PgCheckpointer`` already
drops the Redis entry in both paths.
"""

from __future__ import annotations

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from tenxgraph.utils import END
from tenxgraph.utils.thread_info import ThreadInfo


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


async def _store(checkpointer, config: dict, state: StageState) -> None:
    """Write durable state and cache together, as a finished run leaves them."""
    await checkpointer.aput_state(dict(config), state)
    await checkpointer.aput_state_cache(dict(config), state)


@pytest.mark.asyncio
@pytest.mark.parametrize("make_checkpointer", CHECKPOINTERS)
async def test_aclear_state_drops_the_cached_state(make_checkpointer):
    checkpointer = make_checkpointer()
    config = {"thread_id": "t-clear", "user_id": "u1"}
    await _store(checkpointer, config, StageState(stage="coding"))

    await checkpointer.aclear_state(dict(config))

    assert await checkpointer.aget_state(dict(config)) is None
    assert await checkpointer.aget_state_cache(dict(config)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("make_checkpointer", CHECKPOINTERS)
async def test_the_next_run_starts_fresh_after_aclear_state(make_checkpointer):
    checkpointer = make_checkpointer()
    seen: list[str] = []
    app = _graph(checkpointer, seen)
    config = {"thread_id": "t-clear-run", "user_id": "u1"}

    await app.ainvoke(_turn("first"), config=dict(config))
    await _store(checkpointer, config, StageState(stage="coding"))
    await checkpointer.aclear_state(dict(config))

    await app.ainvoke(_turn("second"), config=dict(config))

    assert seen == ["intro", "intro"]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_checkpointer", CHECKPOINTERS)
async def test_aclean_thread_drops_state_cache_and_messages(make_checkpointer):
    checkpointer = make_checkpointer()
    config = {"thread_id": "t-clean", "user_id": "u1"}
    await checkpointer.aput_thread(dict(config), ThreadInfo(thread_id="t-clean"))
    await _store(checkpointer, config, StageState(stage="coding"))
    await checkpointer.aput_messages(dict(config), [Message.text_message("hi", "user")])

    assert await checkpointer.aclean_thread(dict(config))

    assert await checkpointer.aget_thread(dict(config)) is None
    assert await checkpointer.aget_state(dict(config)) is None
    assert await checkpointer.aget_state_cache(dict(config)) is None
    assert await checkpointer.alist_messages(dict(config)) == []
