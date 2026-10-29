"""Cancel cleanup survives a level-triggered cancel scope (anyio, as used by Starlette).

A cancelled anyio scope re-raises CancelledError at every await of the cancelled task,
including the awaits in ``finally`` blocks. A plain ``task.cancel()`` cancels once, so only
these tests catch cleanup that awaits real I/O outside a task of its own.
"""

import asyncio
import logging

import anyio
import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.base_checkpointer import ACTIVE_RUN_NAMESPACE
from tenxgraph.utils.constants import END

from .test_cancel_safety import Harness, unanswered_calls


CFG = {"thread_id": "t1", "user_id": "u1", "run_id": "r1"}


def build_app():
    started = asyncio.Event()

    async def slow(state: AgentState, config: dict):
        started.set()
        await asyncio.sleep(5)
        return [Message.text_message("done", role="assistant")]

    graph = StateGraph()
    graph.add_node("SLOW", slow)
    graph.set_entry_point("SLOW")
    graph.add_edge("SLOW", END)
    checkpointer = InMemoryCheckpointer()
    return graph.compile(checkpointer=checkpointer), checkpointer, started


async def run_under_anyio_cancel(consume, started):
    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await asyncio.wait_for(started.wait(), 5)
        tg.cancel_scope.cancel()
    await asyncio.sleep(0.3)  # let the detached cleanup finish


async def assert_clean(checkpointer, cfg=CFG):
    marker = await checkpointer.aget_cache_value(
        ACTIVE_RUN_NAMESPACE, checkpointer._stop_request_key(cfg)
    )
    assert marker is None
    state = await checkpointer.aget_state_cache(cfg) or await checkpointer.aget_state(cfg)
    assert state is not None
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "cancelled"
    return state


def assert_no_save_error(caplog):
    assert not [r for r in caplog.records if "Could not save cancelled run" in r.getMessage()]


@pytest.mark.asyncio
async def test_astream_cancelled_by_anyio_scope_cleans_up(caplog):
    app, checkpointer, started = build_app()
    inp = {"messages": [Message.text_message("hi")]}

    async def consume():
        async for _ in app.astream(inp, CFG):
            pass

    with caplog.at_level(logging.INFO):
        await run_under_anyio_cancel(consume, started)
    await assert_clean(checkpointer)
    assert_no_save_error(caplog)
    await app.aclose()


@pytest.mark.asyncio
async def test_ainvoke_cancelled_by_anyio_scope_cleans_up(caplog):
    app, checkpointer, started = build_app()
    inp = {"messages": [Message.text_message("hi")]}

    async def consume():
        await app.ainvoke(inp, CFG)

    with caplog.at_level(logging.INFO):
        await run_under_anyio_cancel(consume, started)
    await assert_clean(checkpointer)
    assert_no_save_error(caplog)
    await app.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_anyio_cancel_with_unanswered_tool_call_closes_it(mode, caplog):
    h = Harness(mode)

    # The ledger lookup is the await that close_unanswered_tool_calls makes. Give it a real
    # suspension point so a re-raised cancel would land there.
    original = h.checkpointer.aget_tool_result

    async def yielding_lookup(*args, **kwargs):
        await asyncio.sleep(0)
        return await original(*args, **kwargs)

    h.checkpointer.aget_tool_result = yielding_lookup

    async def consume():
        await h.run("use tool", "t-anyio-tool")

    with caplog.at_level(logging.INFO):
        await run_under_anyio_cancel(consume, h.started)
    h.release.set()

    config = {"thread_id": "t-anyio-tool"}
    for state in (
        await h.checkpointer.aget_state(config),
        await h.checkpointer.aget_state_cache(config),
    ):
        assert state is not None
        assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
        assert state.execution_meta.interrupt_reason == "cancelled"
        assert unanswered_calls(state.context) == set()
    assert_no_save_error(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_outer_cancel_propagates_under_asyncio(mode):
    app, checkpointer, started = build_app()
    inp = {"messages": [Message.text_message("hi")]}

    async def consume():
        if mode == "ainvoke":
            await app.ainvoke(inp, CFG)
        else:
            async for _ in app.astream(inp, CFG):
                pass

    task = asyncio.ensure_future(consume())
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)
    await assert_clean(checkpointer)
    await app.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_outer_cancel_propagates_under_anyio(mode):
    app, checkpointer, started = build_app()
    inp = {"messages": [Message.text_message("hi")]}
    raised: list[BaseException] = []

    async def consume():
        try:
            if mode == "ainvoke":
                await app.ainvoke(inp, CFG)
            else:
                async for _ in app.astream(inp, CFG):
                    pass
        except asyncio.CancelledError as e:
            raised.append(e)
            raise

    await run_under_anyio_cancel(consume, started)
    assert len(raised) == 1
    await assert_clean(checkpointer)
    await app.aclose()
