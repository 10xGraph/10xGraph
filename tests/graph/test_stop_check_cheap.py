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


class _StopReadCounter(InMemoryCheckpointer):
    """Counts remote stop-flag reads."""

    def __init__(self) -> None:
        super().__init__()
        self.stop_reads = 0

    async def ais_stop_requested(self, config):
        self.stop_reads += 1
        return await super().ais_stop_requested(config)


@pytest.mark.asyncio
async def test_check_stop_requested_local_only_skips_the_remote_read():
    checkpointer = MagicMock()
    checkpointer.ais_stop_requested = AsyncMock(return_value=True)

    stopped = await handler_utils.check_stop_requested(
        AgentState(),
        "A",
        [],
        {"thread_id": "t"},
        callback_mgr=MagicMock(),
        checkpointer=checkpointer,
        remote=False,
    )

    assert stopped is False
    checkpointer.ais_stop_requested.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_a_three_node_run_reads_the_stop_flag_once_per_step(mode):
    checkpointer = _StopReadCounter()
    app = _three_nodes(checkpointer)
    data = {"messages": [Message.text_message("go")]}
    config = {"thread_id": f"t-stop-reads-{mode}"}
    if mode == "ainvoke":
        await app.ainvoke(data, config)
    else:
        async for _ in app.astream(data, config):
            pass

    # Four steps (__start__, A, B, C): one read before each. Short nodes are never polled.
    # Before and after each step (8 reads) before this change.
    assert checkpointer.stop_reads == 4
    await app.aclose()


@pytest.mark.asyncio
async def test_a_long_node_adds_only_its_mid_node_polls():
    checkpointer = _StopReadCounter()

    async def slow(state: AgentState):
        await asyncio.sleep(1.4)

    app = _three_nodes(checkpointer, first=slow)
    await app.ainvoke({"messages": [Message.text_message("go")]}, {"thread_id": "t-long"})

    # 4 step reads plus one poll at 1 s into the 1.4 s node.
    assert checkpointer.stop_reads == 5
    await app.aclose()


@pytest.mark.asyncio
async def test_cross_process_stop_cancels_a_long_node():
    checkpointer = InMemoryCheckpointer()
    ran: list[str] = []
    seen: dict = {}

    async def slow(state: AgentState, config: dict):
        ran.append("A")
        seen["config"] = config
        await asyncio.sleep(10)
        ran.append("A done")

    app = _three_nodes(checkpointer, first=slow)
    config = {"thread_id": "t-xproc-long"}
    run = asyncio.create_task(app.ainvoke({"messages": [Message.text_message("go")]}, config))
    while not ran:
        await asyncio.sleep(0.01)
    # Another process records the stop: only the cache key, not this run's handle.
    await checkpointer.arequest_stop(seen["config"])
    started = time.monotonic()
    await asyncio.wait_for(run, timeout=3)

    assert time.monotonic() - started < 1.5  # within one poll interval
    assert ran == ["A"]
    state = await checkpointer.aget_state(config)
    assert state.execution_meta.interrupt_reason == "stop_requested"
    assert await checkpointer.ais_stop_requested(config) is False  # consumed
    await app.aclose()


@pytest.mark.asyncio
async def test_a_stop_during_the_last_node_is_cleared_when_the_run_ends():
    """Nothing follows the last node: the run completes and the request does not linger."""
    checkpointer = InMemoryCheckpointer()

    async def last(state: AgentState, config: dict):
        await checkpointer.arequest_stop(config)

    graph = StateGraph()
    graph.add_node("A", last)
    graph.add_edge("A", END)
    graph.set_entry_point("A")
    app = graph.compile(checkpointer=checkpointer)
    config = {"thread_id": "t-stop-last"}
    await app.ainvoke({"messages": [Message.text_message("go")]}, config)

    state = await checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert await checkpointer.ais_stop_requested(config) is False
    await app.aclose()


@pytest.mark.asyncio
async def test_a_stop_during_a_node_with_interrupt_after_does_not_leak_into_the_next_run():
    checkpointer = InMemoryCheckpointer()
    ran: list[str] = []
    stop_once = {"done": False}

    async def first(state: AgentState, config: dict):
        ran.append("A")
        if not stop_once["done"]:
            stop_once["done"] = True
            await checkpointer.arequest_stop(config)

    async def second(state: AgentState):
        ran.append("B")

    graph = StateGraph()
    graph.add_node("A", first)
    graph.add_node("B", second)
    graph.add_edge("A", "B")
    graph.add_edge("B", END)
    graph.set_entry_point("A")
    app = graph.compile(checkpointer=checkpointer, interrupt_after=["A"])
    config = {"thread_id": "t-stop-int-after"}
    await app.ainvoke({"messages": [Message.text_message("go")]}, config)

    assert ran == ["A"]
    state = await checkpointer.aget_state(config)
    assert state.is_interrupted()
    assert await checkpointer.ais_stop_requested(config) is False  # cleared when the run ended

    # The resume runs B: the old request does not stop it.
    await app.ainvoke({}, config)
    assert ran == ["A", "B"]
    state = await checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    await app.aclose()
