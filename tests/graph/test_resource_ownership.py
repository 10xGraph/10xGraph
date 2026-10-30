"""Shared resources are closed only by the last live graph that uses them (BE1, F10)."""

import asyncio as _asyncio

import pytest
from injectq import InjectQ

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


class CountingPublisher(BasePublisher):
    def __init__(self):
        super().__init__({})
        self.closed = 0
        self.events = []

    async def publish(self, event):
        self.events.append(event)

    async def close(self):
        self.closed += 1

    def sync_close(self):
        self.closed += 1


class CountingCheckpointer(InMemoryCheckpointer):
    def __init__(self):
        super().__init__()
        self.released = 0

    async def arelease(self):
        self.released += 1
        return await super().arelease()


def _node(state: AgentState, config: dict) -> Message:
    return Message.text_message("ok", role="assistant")


def _graph(publisher=None, container=None) -> StateGraph:
    g = StateGraph(publisher=publisher, container=container or InjectQ())
    g.add_node("A", _node)
    g.add_edge("A", END)
    g.set_entry_point("A")
    return g


@pytest.mark.asyncio
async def test_resource_closed_only_by_last_graph():
    cp = CountingCheckpointer()
    pub = CountingPublisher()
    g1 = _graph(pub).compile(checkpointer=cp)
    g2 = _graph(pub).compile(checkpointer=cp)

    await g1.aclose()
    assert cp.released == 0
    assert pub.closed == 0

    await g2.aclose()
    assert cp.released == 1
    assert pub.closed == 1


@pytest.mark.asyncio
async def test_single_graph_still_closes_its_resources():
    cp = CountingCheckpointer()
    pub = CountingPublisher()
    g = _graph(pub).compile(checkpointer=cp)
    await g.aclose()
    assert cp.released == 1
    assert pub.closed == 1


@pytest.mark.asyncio
async def test_owns_resources_false_never_closes():
    cp = CountingCheckpointer()
    pub = CountingPublisher()
    g = _graph(pub).compile(checkpointer=cp, owns_resources=False)
    stats = await g.aclose()
    assert cp.released == 0
    assert pub.closed == 0
    assert stats["checkpointer"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_non_owning_graph_does_not_block_owner():
    cp = CountingCheckpointer()
    owner = _graph().compile(checkpointer=cp)
    borrower = _graph().compile(checkpointer=cp, owns_resources=False)
    await owner.aclose()
    assert cp.released == 1
    await borrower.aclose()
    assert cp.released == 1


@pytest.mark.asyncio
async def test_background_tasks_belong_to_their_own_graph():
    """F9: with a shared container, each run queues its work on its own graph's manager."""
    container = InjectQ()
    g1 = _graph(CountingPublisher(), container).compile(checkpointer=InMemoryCheckpointer())
    g2 = _graph(CountingPublisher(), container).compile(checkpointer=InMemoryCheckpointer())

    created = {"g1": 0, "g2": 0}
    for key, graph in (("g1", g1), ("g2", g2)):
        original = graph._task_manager.create_task

        def counting(coro, *, _key=key, _orig=original, **kwargs):
            created[_key] += 1
            return _orig(coro, **kwargs)

        graph._task_manager.create_task = counting

    await g1.ainvoke({"messages": [Message.text_message("hi")]}, {"thread_id": "t1"})

    assert created["g1"] > 0
    assert created["g2"] == 0
    await g1.aclose()
    await g2.aclose()


class SlowPublisher(CountingPublisher):
    async def publish(self, event):
        await _asyncio.sleep(0.3)
        self.events.append(event)


def test_sync_invoke_waits_for_background_events():
    """G-22: sync invoke() must not drop slow publisher events, including the last one."""
    pub = SlowPublisher()
    g = _graph(pub).compile(checkpointer=InMemoryCheckpointer())
    g.invoke({"messages": [Message.text_message("hi")]}, {"thread_id": "sync1"})
    assert len(pub.events) > 0
    # Every event that was queued was delivered, none were cancelled.
    assert g._task_manager.get_task_count() == 0


@pytest.mark.asyncio
async def test_shed_events_log_a_warning(caplog):
    import logging

    from tenxgraph.utils.background_task_manager import BackgroundTaskManager

    manager = BackgroundTaskManager(max_pending_tasks=1)
    manager.create_task(_asyncio.sleep(0.2), name="first")
    with caplog.at_level(logging.WARNING, logger="tenxgraph.utils"):
        manager.create_task(_asyncio.sleep(0.2), name="second")
    assert any("dropping task" in r.message for r in caplog.records)
    await manager.shutdown(timeout=1)
