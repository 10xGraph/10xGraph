"""Cancel cleanup ordering (G-171): what runs after a cancelled run sees its save done.

The store here is slow (``aput_checkpoint`` sleeps), so a release that does not wait for the
cancel save shows up as a marker cleared, or a scope exited, before the save finished.
"""

import asyncio
import json
import logging
import time

import anyio
import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.graph.utils import handler_utils
from tenxgraph.core.graph.utils import utils as graph_utils
from tenxgraph.core.state import AgentState, ExecutionStatus, Message, ToolCallBlock
from tenxgraph.runtime.publisher.events import EventType
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.base_checkpointer import ACTIVE_RUN_NAMESPACE
from tenxgraph.utils.constants import END


SAVE_DELAY = 0.15


class SlowCheckpointer(InMemoryCheckpointer):
    """Durable writes take ``delay`` seconds; records saves and run ends in order."""

    def __init__(self, delay: float = SAVE_DELAY):
        super().__init__()
        self.delay = delay
        self.log: list[tuple[str, str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.save_started: dict[str, asyncio.Event] = {}

    def started(self, status: str) -> asyncio.Event:
        return self.save_started.setdefault(status, asyncio.Event())

    async def aput_checkpoint(self, config, state, messages=None, metadata=None):
        status = state.execution_meta.status.value
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.log.append(("save start", status))
        self.started(status).set()
        try:
            await asyncio.sleep(self.delay)
            return await super().aput_checkpoint(config, state, messages, metadata)
        finally:
            self.in_flight -= 1
            self.log.append(("save done", status))

    async def aend_run(self, config):
        self.log.append(("end run", str(config.get("run_id"))))
        return await super().aend_run(config)

    def saved_statuses(self) -> list[str]:
        return [status for kind, status in self.log if kind == "save start"]


def cfg(run_id: str = "r1") -> dict:
    return {"thread_id": "t1", "user_id": "u1", "run_id": run_id}


async def marker(cp) -> str | None:
    return await cp.aget_cache_value(ACTIVE_RUN_NAMESPACE, cp._stop_request_key(cfg()))


async def consume(app, mode: str, config: dict, text: str = "hi") -> None:
    data = {"messages": [Message.text_message(text)]}
    if mode == "ainvoke":
        await app.ainvoke(data, config)
    else:
        async for _ in app.astream(data, config):
            pass


async def cancel_under_anyio(run, ready: asyncio.Event) -> float:
    """Run ``run`` in an anyio task group and cancel it once ``ready``; return the cancel time."""
    async with anyio.create_task_group() as tg:
        tg.start_soon(run)
        await asyncio.wait_for(ready.wait(), 5)
        cancelled_at = time.monotonic()
        tg.cancel_scope.cancel()
    return cancelled_at


def build_one_node(cp, hold: bool):
    started = asyncio.Event()

    async def node(state: AgentState, config: dict):
        started.set()
        if hold:
            await asyncio.sleep(5)
        return [Message.text_message("answer", role="assistant")]

    graph = StateGraph()
    graph.add_node("N", node)
    graph.set_entry_point("N")
    graph.add_edge("N", END)
    return graph.compile(checkpointer=cp), started


def build_two_nodes(cp):
    async def first(state: AgentState, config: dict):
        return [Message.text_message("first", role="assistant")]

    async def second(state: AgentState, config: dict):
        return [Message.text_message("second", role="assistant")]

    graph = StateGraph()
    graph.add_node("A", first)
    graph.add_node("B", second)
    graph.set_entry_point("A")
    graph.add_edge("A", "B")
    graph.add_edge("B", END)
    return graph.compile(checkpointer=cp)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_marker_released_only_after_the_cancel_save(mode):
    cp = SlowCheckpointer()
    app, started = build_one_node(cp, hold=True)

    await cancel_under_anyio(lambda: consume(app, mode, cfg()), started)

    # Checked the moment the scope exits: no sleep to let background work catch up.
    state = await cp.aget_state(cfg())
    assert state is not None
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert await marker(cp) is None
    assert cp.log.index(("save done", "interrupted_after")) < cp.log.index(("end run", "r1"))
    await app.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_cancel_during_final_save_keeps_the_run_completed(mode):
    cp = SlowCheckpointer()
    app, _ = build_one_node(cp, hold=False)

    await cancel_under_anyio(lambda: consume(app, mode, cfg()), cp.started("completed"))
    await asyncio.sleep(SAVE_DELAY * 2)  # a second save, if one was started, would land here

    assert cp.max_in_flight == 1
    assert "interrupted_after" not in cp.saved_statuses()
    assert cp.saved_statuses().count("completed") == 1
    state = await cp.aget_state(cfg())
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert state.execution_meta.interrupt_reason is None
    await app.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_cancel_during_a_step_save_saves_once_at_a_time(mode, caplog):
    cp = SlowCheckpointer()
    app = build_two_nodes(cp)

    with caplog.at_level(logging.WARNING):
        await cancel_under_anyio(lambda: consume(app, mode, cfg()), cp.started("running"))
    await asyncio.sleep(SAVE_DELAY * 2)

    assert cp.max_in_flight == 1
    assert cp.saved_statuses() == ["running", "interrupted_after"]
    state = await cp.aget_state(cfg())
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert not [r for r in caplog.records if "did not finish" in r.getMessage()]
    await app.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_interrupted_event_carries_the_saved_state(mode, monkeypatch):
    cp = InMemoryCheckpointer()
    started = asyncio.Event()
    events: list[tuple] = []

    original_publish = handler_utils.publish_event

    def capture(event, *args, **kwargs):
        if event.event_type == EventType.INTERRUPTED:
            state = event.data["state"]
            events.append((state["execution_meta"]["status"], len(state["context"])))
        return original_publish(event, *args, **kwargs)

    monkeypatch.setattr(handler_utils, "publish_event", capture)

    original_lookup = cp.aget_tool_result

    async def yielding_lookup(*args, **kwargs):  # a real ledger read suspends
        await asyncio.sleep(0.01)
        return await original_lookup(*args, **kwargs)

    cp.aget_tool_result = yielding_lookup

    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(5)
        return "tool done"

    async def model(state: AgentState):
        if state.context[-1].role == "tool":
            return Message.text_message("finished", role="assistant")
        return Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="slow_tool", args={})],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "slow_tool", "arguments": json.dumps({})},
                }
            ],
        )

    def route(state: AgentState) -> str:
        return "TOOL" if state.context[-1].tools_calls else END

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([slow_tool]))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    app = graph.compile(checkpointer=cp)

    await cancel_under_anyio(lambda: consume(app, mode, cfg(), "use tool"), started)
    await asyncio.sleep(0.1)

    saved = await cp.aget_state(cfg())
    assert saved.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert events == [(ExecutionStatus.INTERRUPTED_AFTER, len(saved.context))]
    await app.aclose()


@pytest.mark.asyncio
async def test_anyio_cancel_waits_for_the_save_without_spinning():
    cp = SlowCheckpointer(delay=0.3)
    app, started = build_one_node(cp, hold=True)

    cpu_before = time.process_time()
    cancelled_at = await cancel_under_anyio(lambda: consume(app, "astream", cfg()), started)
    wall = time.monotonic() - cancelled_at
    cpu = time.process_time() - cpu_before

    assert wall >= 0.25  # the scope waited for the cancel save
    assert cpu < wall / 2  # and did not spin on re-delivered cancels meanwhile
    await app.aclose()


# --- the helpers on their own ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_detached_holds_repeated_cancels_and_raises_one():
    job = asyncio.ensure_future(asyncio.sleep(0.2))
    waiter = asyncio.ensure_future(graph_utils.wait_detached(job, 5))
    await asyncio.sleep(0.02)
    waiter.cancel()
    await asyncio.sleep(0.02)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert job.done() and not job.cancelled()


@pytest.mark.asyncio
async def test_wait_detached_times_out_without_cancelling_the_task():
    job = asyncio.ensure_future(asyncio.sleep(0.3))
    assert await graph_utils.wait_detached(job, 0.05) is False
    assert not job.done()
    await job


@pytest.mark.asyncio
async def test_wait_detached_under_anyio_cancel_does_not_spin():
    job = asyncio.ensure_future(asyncio.sleep(0.3))
    result: list[bool] = []
    ready = asyncio.Event()

    async def child():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            result.append(await graph_utils.wait_detached(job, 5))
            raise

    async def go():
        ready.set()
        await child()

    cpu_before = time.process_time()
    cancelled_at = await cancel_under_anyio(go, ready)
    wall = time.monotonic() - cancelled_at
    cpu = time.process_time() - cpu_before
    assert result == [True]
    assert wall >= 0.25
    assert cpu < wall / 2


@pytest.mark.asyncio
async def test_detached_task_is_cancelled_after_its_lifetime(caplog):
    task = graph_utils.detach_task(asyncio.sleep(5), lifetime=0.05)
    assert task in graph_utils._detached_tasks
    with caplog.at_level(logging.ERROR):
        await asyncio.wait({task}, timeout=1)
    assert task.cancelled()
    assert task not in graph_utils._detached_tasks
    assert [r for r in caplog.records if "Cleanup task still running" in r.getMessage()]


@pytest.mark.asyncio
async def test_slow_cancel_save_logs_still_running_and_is_capped(monkeypatch, caplog):
    monkeypatch.setattr(graph_utils, "FINALIZE_TIMEOUT_SECONDS", 0.05)
    before = set(graph_utils._detached_tasks)
    cp = SlowCheckpointer(delay=1.0)
    app, started = build_one_node(cp, hold=True)

    task = asyncio.ensure_future(consume(app, "ainvoke", cfg()))
    await asyncio.wait_for(started.wait(), 5)
    with caplog.at_level(logging.INFO):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.3)  # past the 2x lifetime of the detached cleanup

    messages = [r.getMessage() for r in caplog.records]
    assert any("still running in the background" in m for m in messages)
    assert any("Cleanup task still running" in m for m in messages)
    assert not graph_utils._detached_tasks - before
    await app.aclose()
