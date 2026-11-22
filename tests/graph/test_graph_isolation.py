"""J-1: two graphs in one process never share their checkpointer, store, callback manager,
publisher or background task manager, in the same container or not."""

from __future__ import annotations

import asyncio

import pytest
from injectq import Inject, InjectQ

from tenxgraph.core.graph import StateGraph, ToolGuard, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.qa.testing.in_memory_store import InMemoryStore
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.storage.checkpointer import BaseCheckpointer, InMemoryCheckpointer
from tenxgraph.storage.store import BaseStore
from tenxgraph.utils import CallbackManager
from tenxgraph.utils.background_task_manager import BackgroundTaskManager
from tenxgraph.utils.constants import END
from tenxgraph.utils.injection import fresh

from .tool_guard_helpers import RecordingPublisher, build


RAN: list[str] = []


def refund(amount: int = 0) -> str:
    RAN.append("refund")
    return "refunded"


@pytest.fixture(autouse=True)
def _clear():
    RAN.clear()


def _cfg(thread: str) -> dict:
    return {"thread_id": thread, "user_id": "u1"}


async def _threads(checkpointer: InMemoryCheckpointer) -> list[str]:
    """Threads (of a, b, b2, c) that have a saved state in ``checkpointer``."""
    return [t for t in ("a", "b", "b2", "c") if await checkpointer.aget_state(_cfg(t))]


def _seen_graph(container: InjectQ | None = None, publisher=None):
    """One node that records what it resolves from the active container during the run."""
    seen: dict = {}

    async def look(state: AgentState, cp: BaseCheckpointer = Inject[BaseCheckpointer]):
        seen["checkpointer"] = fresh(cp)
        seen["store"] = InjectQ.get_instance().try_get(BaseStore)
        seen["callbacks"] = InjectQ.get_instance().try_get(CallbackManager)
        seen["tasks"] = InjectQ.get_instance().try_get(BackgroundTaskManager)
        await asyncio.sleep(0.01)  # let a concurrent run interleave
        return state

    graph = StateGraph(container=container, publisher=publisher)
    graph.add_node("LOOK", look)
    graph.set_entry_point("LOOK")
    graph.add_edge("LOOK", END)
    return graph, seen


@pytest.mark.asyncio
async def test_each_graph_stores_threads_in_its_own_checkpointer():
    cpa, cpb = InMemoryCheckpointer(), InMemoryCheckpointer()
    app_a = _seen_graph()[0].compile(checkpointer=cpa)
    app_b = _seen_graph()[0].compile(checkpointer=cpb)

    await app_a.ainvoke({"messages": [Message.text_message("hi")]}, _cfg("a"))
    await app_b.ainvoke({"messages": [Message.text_message("hi")]}, _cfg("b"))

    assert await _threads(cpa) == ["a"]
    assert await _threads(cpb) == ["b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
async def test_concurrent_runs_resolve_their_own_components(shared):
    container = InjectQ() if shared else None
    graph_a, seen_a = _seen_graph(container)
    graph_b, seen_b = _seen_graph(container)
    cpa, cpb = InMemoryCheckpointer(), InMemoryCheckpointer()
    sta, stb = InMemoryStore(), InMemoryStore()
    mgr_a, mgr_b = CallbackManager(), CallbackManager()
    app_a = graph_a.compile(checkpointer=cpa, store=sta, callback_manager=mgr_a)
    app_b = graph_b.compile(checkpointer=cpb, store=stb, callback_manager=mgr_b)

    data = {"messages": [Message.text_message("hi")]}
    await asyncio.gather(app_a.ainvoke(data, _cfg("a")), app_b.ainvoke(data, _cfg("b")))

    assert seen_a == {
        "checkpointer": cpa,
        "store": sta,
        "callbacks": mgr_a,
        "tasks": graph_a._task_manager,
    }
    assert seen_b == {
        "checkpointer": cpb,
        "store": stb,
        "callbacks": mgr_b,
        "tasks": graph_b._task_manager,
    }
    assert await _threads(cpa) == ["a"]
    assert await _threads(cpb) == ["b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
async def test_tool_guard_applies_only_to_the_graph_compiled_with_it(mode):
    open_mgr = CallbackManager()
    guarded_mgr = CallbackManager()
    ToolGuard(allow={"search_web"}).register(guarded_mgr)
    app_a = build(ToolNode([refund]), [("c1", "refund")], manager=open_mgr)
    app_b = build(ToolNode([refund]), [("c1", "refund")], manager=guarded_mgr)

    data = {"messages": [Message.text_message("go")]}
    if mode == "sequential":
        await app_a.ainvoke(data, _cfg("a"))
        assert RAN == ["refund"]
        await app_b.ainvoke(data, _cfg("b"))
        assert RAN == ["refund"]  # B denied it
    else:
        await asyncio.gather(app_b.ainvoke(data, _cfg("b")), app_a.ainvoke(data, _cfg("a")))
        assert RAN == ["refund"]  # A ran it once, B denied it

    # B again on its own, after A: still denied.
    RAN.clear()
    await app_b.ainvoke(data, _cfg("b2"))
    assert RAN == []


@pytest.mark.asyncio
async def test_stream_driven_from_new_tasks_keeps_its_own_components():
    """The API drives each stream step from a new task; every step must see this graph."""
    guarded_mgr = CallbackManager()
    ToolGuard(allow={"search_web"}).register(guarded_mgr)
    pub_a, pub_b = RecordingPublisher(), RecordingPublisher()
    app_b = build(ToolNode([refund]), [("c1", "refund")], manager=guarded_mgr, publisher=pub_b)
    # Compiled last, so the parent container now holds this open graph's values.
    app_a = build(ToolNode([refund]), [("c1", "refund")], publisher=pub_a)

    gen = app_b.astream({"messages": [Message.text_message("go")]}, _cfg("b"))
    while True:
        try:
            await asyncio.ensure_future(gen.__anext__())
        except StopAsyncIteration:
            break
    await app_b._task_manager.wait_for_all(timeout=5)
    await app_a._task_manager.wait_for_all(timeout=5)
    assert RAN == []
    assert pub_b.events
    assert pub_a.events == []


@pytest.mark.asyncio
async def test_each_publisher_receives_only_its_own_graph_events():
    pub_a, pub_b = RecordingPublisher(), RecordingPublisher()
    app_a = build(ToolNode([refund]), [("c1", "refund")], publisher=pub_a)
    app_b = build(ToolNode([refund]), [("c1", "refund")], publisher=pub_b)
    app_c = build(ToolNode([refund]), [("c1", "refund")])  # no publisher of its own

    data = {"messages": [Message.text_message("go")]}
    await asyncio.gather(app_a.ainvoke(data, _cfg("a")), app_b.ainvoke(data, _cfg("b")))
    await app_c.ainvoke(data, _cfg("c"))
    for app in (app_a, app_b, app_c):
        await app._task_manager.wait_for_all(timeout=5)

    assert pub_a.events and pub_b.events
    assert {e.thread_id for e in pub_a.events} == {"a"}
    assert {e.thread_id for e in pub_b.events} == {"b"}


@pytest.mark.asyncio
async def test_tool_injects_a_user_binding_from_the_global_container():
    class Db:
        pass

    db = Db()
    InjectQ.get_instance().bind_instance(Db, db)
    got: list = []

    def lookup(order: str = "", database: Db = Inject[Db]) -> str:
        got.append(fresh(database))
        return "ok"

    app = build(ToolNode([lookup]), [("c1", "lookup")])
    await app.ainvoke({"messages": [Message.text_message("go")]}, _cfg("a"))
    assert got == [db]


@pytest.mark.asyncio
async def test_parent_container_still_holds_the_graph_values_outside_runs():
    """The API server reads the checkpointer and task manager from its container."""
    container = InjectQ()
    graph, _ = _seen_graph(container)
    cp = InMemoryCheckpointer()
    graph.compile(checkpointer=cp)
    assert container.get(BaseCheckpointer) is cp
    assert container.get(BackgroundTaskManager) is graph._task_manager


@pytest.mark.asyncio
async def test_publisher_bound_in_the_container_before_build_is_used():
    """A bootstrap publisher (the API's observability one) reaches a graph without its own."""
    boot = RecordingPublisher()
    InjectQ.get_instance().bind_instance(BasePublisher, boot)
    app = build(ToolNode([refund]), [("c1", "refund")])
    await app.ainvoke({"messages": [Message.text_message("go")]}, _cfg("a"))
    await app._task_manager.wait_for_all(timeout=5)
    assert boot.events


@pytest.mark.asyncio
async def test_aclose_of_one_graph_leaves_the_other_working():
    pub_a, pub_b = RecordingPublisher(), RecordingPublisher()
    closed: list[str] = []

    async def close_b():
        closed.append("pub_b")

    pub_b.close = close_b  # type: ignore[method-assign]
    cpa, cpb = InMemoryCheckpointer(), InMemoryCheckpointer()
    app_a = build(ToolNode([refund]), [("c1", "refund")], checkpointer=cpa, publisher=pub_a)
    app_b = build(ToolNode([refund]), [("c1", "refund")], checkpointer=cpb, publisher=pub_b)

    await app_a.aclose()
    assert closed == []
    await app_b.ainvoke({"messages": [Message.text_message("go")]}, _cfg("b"))
    await app_b._task_manager.wait_for_all(timeout=5)
    assert await _threads(cpb) == ["b"]
    assert {e.thread_id for e in pub_b.events} == {"b"}


def test_compile_without_callback_manager_gives_each_graph_its_own():
    graph_a, _ = _seen_graph()
    graph_b, _ = _seen_graph()
    graph_a.compile()
    graph_b.compile()
    assert graph_a._container.get(CallbackManager) is not graph_b._container.get(CallbackManager)
