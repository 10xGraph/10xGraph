"""Sync wrappers fail fast inside a running loop, and async id factories work (G-20)."""

import asyncio

import pytest
from injectq import InjectQ

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.callable_utils import run_coroutine
from tenxgraph.utils.constants import END
from tenxgraph.utils.id_generator import AsyncIDGenerator


def _node(state: AgentState, config: dict) -> Message:
    return Message.text_message("ok", role="assistant")


def _compiled(id_generator=None):
    g = StateGraph(id_generator=id_generator, container=InjectQ())
    g.add_node("A", _node)
    g.add_edge("A", END)
    g.set_entry_point("A")
    return g.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.asyncio
async def test_run_coroutine_in_running_loop_raises_instead_of_hanging():
    async def aget_thing():
        return 1

    with pytest.raises(RuntimeError, match="aget_thing"):
        run_coroutine(aget_thing())


@pytest.mark.asyncio
async def test_sync_checkpointer_call_in_running_loop_points_to_async_method():
    cp = InMemoryCheckpointer()
    with pytest.raises(RuntimeError, match="aget_state"):
        cp.get_state({"thread_id": "t"})


@pytest.mark.asyncio
async def test_sync_invoke_in_running_loop_points_to_ainvoke():
    graph = _compiled()
    with pytest.raises(RuntimeError, match="ainvoke"):
        graph.invoke({"messages": [Message.text_message("hi")]}, {"thread_id": "t"})


@pytest.mark.asyncio
async def test_sync_stream_in_running_loop_points_to_astream():
    graph = _compiled()
    with pytest.raises(RuntimeError, match="astream"):
        next(graph.stream({"messages": [Message.text_message("hi")]}, {"thread_id": "t"}))


@pytest.mark.asyncio
async def test_message_with_async_id_factory_inside_running_loop():
    _compiled(AsyncIDGenerator())
    msg = Message.text_message("hi")
    assert isinstance(msg.message_id, str)
    assert len(msg.message_id) == 36


def test_message_with_async_id_factory_outside_loop():
    _compiled(AsyncIDGenerator())
    msg = Message.text_message("hi")
    assert isinstance(msg.message_id, str)
    assert len(msg.message_id) == 36


@pytest.mark.asyncio
async def test_async_id_factory_gives_string_thread_and_run_ids():
    graph = _compiled(AsyncIDGenerator())
    cfg = graph._prepare_config({})
    assert isinstance(cfg["thread_id"], str)
    assert isinstance(cfg["run_id"], str)
    assert not asyncio.iscoroutine(cfg["thread_id"])
