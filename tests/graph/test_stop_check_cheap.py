"""P-3: the stop check reads a small flag, never the full state; a stop cuts a model stream.

Before each node the engine checks the run's in-process flag, then one small cache key for a
stop from another process. It does not reload and validate the whole state. A stop that
arrives while a model is streaming cancels the node at once and closes the provider stream.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.utils import handler_utils
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


class _CountingCheckpointer(InMemoryCheckpointer):
    """Counts full-state reads."""

    def __init__(self) -> None:
        super().__init__()
        self.state_reads = 0

    async def aget_state_cache(self, config):
        self.state_reads += 1
        return await super().aget_state_cache(config)

    async def aget_state(self, config):
        self.state_reads += 1
        return await super().aget_state(config)


def _three_nodes(checkpointer, first=None):
    async def noop(state: AgentState):
        return None

    graph = StateGraph()
    graph.add_node("A", first or noop)
    graph.add_node("B", noop)
    graph.add_node("C", noop)
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", END)
    graph.set_entry_point("A")
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_check_stop_requested_reads_no_state():
    checkpointer = MagicMock()
    checkpointer.ais_stop_requested = AsyncMock(return_value=False)
    checkpointer.aget_state_cache = AsyncMock(side_effect=AssertionError("full state read"))
    checkpointer.aget_state = AsyncMock(side_effect=AssertionError("full state read"))
    callback_mgr = MagicMock()

    stopped = await handler_utils.check_stop_requested(
        AgentState(),
        "A",
        [],
        {"thread_id": "t"},
        callback_mgr=callback_mgr,
        checkpointer=checkpointer,
    )

    assert stopped is False
    checkpointer.ais_stop_requested.assert_awaited_once()
    checkpointer.aget_state_cache.assert_not_awaited()
    checkpointer.aget_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_three_node_run_reads_the_state_once():
    checkpointer = _CountingCheckpointer()
    app = _three_nodes(checkpointer)
    config = {"thread_id": "t-reads"}
    await app.ainvoke({"messages": [Message.text_message("one")]}, config)

    checkpointer.state_reads = 0
    await app.ainvoke({"messages": [Message.text_message("two")]}, config)

    # The load at run start only; six stop checks (before and after each node) read none.
    assert checkpointer.state_reads == 1
    await app.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_cross_process_stop_is_seen_before_the_next_node(mode):
    checkpointer = InMemoryCheckpointer()
    ran: list[str] = []

    async def first(state: AgentState, config: dict):
        ran.append("A")
        # Another process records the stop: only the cache key, not this run's handle.
        await checkpointer.arequest_stop(config)

    app = _three_nodes(checkpointer, first=first)
    config = {"thread_id": f"t-xproc-{mode}"}
    data = {"messages": [Message.text_message("go")]}
    if mode == "ainvoke":
        await app.ainvoke(data, config)
    else:
        async for _ in app.astream(data, config):
            pass

    assert ran == ["A"]
    state = await checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "stop_requested"
    assert state.execution_meta.current_node == "B"
    assert await checkpointer.ais_stop_requested(config) is False  # consumed
    await app.aclose()


def _chunk(text: str):
    return SimpleNamespace(
        model="m",
        id="c1",
        usage=None,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=None,
                delta=SimpleNamespace(content=text, tool_calls=None, role="assistant"),
            )
        ],
    )


@pytest.mark.asyncio
async def test_stop_cuts_a_streaming_model_call():
    """A stop mid-stream ends the run at once and closes the provider stream."""
    progress = {"chunks": 0, "closed": False}

    async def slow_stream():
        # Stands in for the SDK stream: its finally closes the HTTP response.
        try:
            for i in range(200):
                progress["chunks"] += 1
                yield _chunk(f"t{i} ")
                await asyncio.sleep(0.05)
        finally:
            progress["closed"] = True

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        assert stream is True
        return slow_stream()

    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        agent = Agent(model="gpt-4o", provider="openai", reasoning_config=None)

    checkpointer = InMemoryCheckpointer()
    graph = StateGraph()
    graph.add_node("MODEL", agent)
    graph.add_edge("MODEL", END)
    graph.set_entry_point("MODEL")
    app = graph.compile(checkpointer=checkpointer)
    config = {"thread_id": "t-stream-stop"}
    before = asyncio.all_tasks()

    async def consume():
        stream = app.astream(
            {"messages": [Message.text_message("hi")]},
            config,
            response_granularity=ResponseGranularity.FULL,
        )
        return [c async for c in stream]

    with patch.object(Agent, "_call_llm", fake_call):
        run = asyncio.create_task(consume())
        while progress["chunks"] < 3:
            await asyncio.sleep(0.01)
        started = time.monotonic()
        await app.astop(config)
        await asyncio.wait_for(run, timeout=2)
        elapsed = time.monotonic() - started

    # The stream had about 10 s left; the stop ended it within a chunk or two.
    assert elapsed < 0.5
    assert progress["closed"] is True
    assert progress["chunks"] < 20
    state = await checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "stop_requested"
    # The cut-off reply is not saved: no half-written assistant message.
    assert [m.role for m in state.context] == ["user"]
    await asyncio.sleep(0)
    assert not [t for t in asyncio.all_tasks() - before if not t.done()]  # no task leaked
    await app.aclose()
