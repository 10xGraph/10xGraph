"""``aclose()`` drains active runs in phases and refuses new work (BD1 to BD4, G-78).

Phases: refuse new runs, wait ``grace`` for runs to finish, set the in-process stop flag on
the rest (each saves INTERRUPTED) and wait ``stop_timeout``, hard-cancel what still runs, drain
background tasks, release resources.
"""

import asyncio
import json
import threading
import time

import pytest

from tenxgraph.core.exceptions import GraphClosedError, GraphError
from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, ExecutionStatus, Message, ToolCallBlock
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


def _input(text: str = "hi") -> dict:
    return {"messages": [Message.text_message(text)]}


class Rig:
    """A one-node graph whose node can be held open, stopped, or made to ignore a cancel."""

    def __init__(self, behavior: str, *, checkpointer=True, shutdown_timeout: float = 8.0):
        self.behavior = behavior
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.checkpointer = (
            InMemoryCheckpointer() if checkpointer is True else (checkpointer or None)
        )
        self.swallowed = False
        graph = StateGraph()
        graph.add_node("WORK", self._work)
        graph.add_edge("WORK", END)
        graph.set_entry_point("WORK")
        self.app = graph.compile(
            checkpointer=self.checkpointer, shutdown_timeout=shutdown_timeout
        )

    async def _work(self, state: AgentState):
        self.started.set()
        if self.behavior == "short":
            await asyncio.sleep(0.4)
        elif self.behavior == "hang":
            await asyncio.sleep(60)
        elif self.behavior == "stubborn":
            # Ignores the first cancel, so only a second one gets through.
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.swallowed = True
                await asyncio.sleep(60)
        return Message.text_message("done", role="assistant")

    def start(self, thread_id: str) -> asyncio.Task:
        return asyncio.create_task(self.app.ainvoke(_input(), {"thread_id": thread_id}))


async def _saved(rig: Rig, thread_id: str):
    return await rig.checkpointer.aget_state({"thread_id": thread_id})


@pytest.mark.asyncio
async def test_grace_lets_a_run_finish():
    rig = Rig("short")
    task = rig.start("t-grace")
    await asyncio.wait_for(rig.started.wait(), 5)

    stats = await rig.app.aclose(grace=3, stop_timeout=1, timeout=1)

    result = await task
    assert result["messages"][-1].text() == "done"
    assert stats["runs"] == {"active": 1, "finished": 1, "stopped": 0, "cancelled": 0}


@pytest.mark.asyncio
async def test_stop_phase_saves_interrupted():
    rig = Rig("hang")
    task = rig.start("t-stop")
    await asyncio.wait_for(rig.started.wait(), 5)

    begin = time.monotonic()
    stats = await rig.app.aclose(grace=0.2, stop_timeout=3, timeout=1)

    await asyncio.wait_for(task, 5)  # a stop ends the run normally
    assert time.monotonic() - begin < 3
    assert stats["runs"] == {"active": 1, "finished": 0, "stopped": 1, "cancelled": 0}
    state = await _saved(rig, "t-stop")
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "stop_requested"


@pytest.mark.asyncio
async def test_cancel_phase_ends_a_node_that_ignores_the_stop():
    rig = Rig("stubborn")
    task = rig.start("t-cancel")
    await asyncio.wait_for(rig.started.wait(), 5)

    stats = await rig.app.aclose(grace=0.1, stop_timeout=0.3, timeout=3)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert rig.swallowed
    assert stats["runs"] == {"active": 1, "finished": 0, "stopped": 0, "cancelled": 1}
    state = await _saved(rig, "t-cancel")
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "shutdown"


@pytest.mark.asyncio
async def test_cancel_phase_works_on_a_stuck_tool():
    release = asyncio.Event()
    started = asyncio.Event()
    swallowed = []

    async def stuck_tool() -> str:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            swallowed.append(True)
            await release.wait()
        return "never"

    async def model(state: AgentState):
        if state.context[-1].role == "tool":
            return Message.text_message("finished", role="assistant")
        return Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="stuck_tool", args={})],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "stuck_tool", "arguments": json.dumps({})},
                }
            ],
        )

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([stuck_tool]))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    checkpointer = InMemoryCheckpointer()
    app = graph.compile(checkpointer=checkpointer, shutdown_timeout=8)

    task = asyncio.create_task(app.ainvoke(_input("go"), {"thread_id": "t-tool"}))
    await asyncio.wait_for(started.wait(), 5)

    begin = time.monotonic()
    stats = await app.aclose(grace=0.1, stop_timeout=0.3, timeout=3)

    with pytest.raises((asyncio.CancelledError, Exception)):
        await asyncio.wait_for(task, 5)
    assert time.monotonic() - begin < 5
    assert stats["runs"]["finished"] == 0
    assert stats["runs"]["stopped"] + stats["runs"]["cancelled"] == 1
    state = await checkpointer.aget_state({"thread_id": "t-tool"})
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    release.set()


@pytest.mark.asyncio
async def test_new_runs_raise_graph_closed_error():
    rig = Rig("short")
    await rig.app.aclose(grace=0, stop_timeout=0, timeout=0)

    assert issubclass(GraphClosedError, GraphError)
    with pytest.raises(GraphClosedError):
        await rig.app.ainvoke(_input(), {"thread_id": "late"})
    with pytest.raises(GraphClosedError):
        async for _ in rig.app.astream(_input(), {"thread_id": "late"}):
            pass
    with pytest.raises(GraphClosedError):
        rig.app._refuse_if_closing()  # noqa: SLF001


@pytest.mark.asyncio
async def test_new_runs_are_refused_while_closing():
    rig = Rig("short")
    task = rig.start("t-open")
    await asyncio.wait_for(rig.started.wait(), 5)

    closer = asyncio.create_task(rig.app.aclose(grace=3, stop_timeout=1, timeout=1))
    await asyncio.sleep(0.05)
    with pytest.raises(GraphClosedError):
        await rig.app.ainvoke(_input(), {"thread_id": "late"})
    await closer
    await task


def test_sync_wrappers_raise_graph_closed_error():
    rig_holder = {}

    async def build():
        rig = Rig("short")
        await rig.app.aclose(grace=0, stop_timeout=0, timeout=0)
        rig_holder["rig"] = rig

    asyncio.run(build())
    app = rig_holder["rig"].app
    with pytest.raises(GraphClosedError):
        app.invoke(_input(), {"thread_id": "late"})
    with pytest.raises(GraphClosedError):
        list(app.stream(_input(), {"thread_id": "late"}))


@pytest.mark.asyncio
async def test_sync_invoke_in_another_thread_is_stopped_by_aclose():
    rig = Rig("hang")
    outcome: dict = {}

    def run_sync() -> None:
        try:
            outcome["result"] = rig.app.invoke(_input(), {"thread_id": "t-sync"})
        except BaseException as e:  # noqa: BLE001
            outcome["error"] = e

    # The node's Event belongs to the thread's loop, so use a thread-safe signal instead.
    started = threading.Event()
    original = rig._work  # noqa: SLF001

    async def work(state):
        started.set()
        return await original(state)

    rig.app._state_graph.nodes["WORK"].func = work  # noqa: SLF001

    thread = threading.Thread(target=run_sync)
    thread.start()
    assert await asyncio.to_thread(started.wait, 5)

    begin = time.monotonic()
    stats = await rig.app.aclose(grace=0.2, stop_timeout=3, timeout=2)
    await asyncio.to_thread(thread.join, 10)

    assert not thread.is_alive()
    assert time.monotonic() - begin < 5
    assert "error" not in outcome, outcome
    assert stats["runs"]["active"] == 1
    assert stats["runs"]["stopped"] == 1
    state = await rig.checkpointer.aget_state({"thread_id": "t-sync"})
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER


@pytest.mark.asyncio
async def test_aclose_with_no_runs_and_twice_is_a_no_op():
    rig = Rig("short")
    first = await rig.app.aclose()
    assert first["runs"] == {"active": 0, "finished": 0, "stopped": 0, "cancelled": 0}
    assert "background_tasks" in first
    assert await rig.app.aclose(grace=0, stop_timeout=0, timeout=0) == {
        "status": "already_closed"
    }


@pytest.mark.asyncio
async def test_defaults_split_the_compile_timeout():
    rig = Rig("short", shutdown_timeout=8)
    assert rig.app._shutdown_phases(None, None, None) == (4.0, 2.0, 2.0)  # noqa: SLF001
    assert rig.app._shutdown_phases(1, None, None) == (1, 2.0, 2.0)  # noqa: SLF001
    assert rig.app._shutdown_phases(None, None, 10) == (5.0, 2.5, 2.5)  # noqa: SLF001
    await rig.app.aclose()


@pytest.mark.asyncio
async def test_finished_runs_are_unregistered():
    rig = Rig("short")
    task = rig.start("t-done")
    await asyncio.wait_for(rig.started.wait(), 5)
    await task
    assert rig.app._runs.active() == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_astream_run_is_stopped_by_aclose():
    rig = Rig("hang")

    async def consume():
        return [c async for c in rig.app.astream(_input(), {"thread_id": "t-stream"})]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(rig.started.wait(), 5)
    stats = await rig.app.aclose(grace=0.1, stop_timeout=3, timeout=1)
    await asyncio.wait_for(task, 5)
    assert stats["runs"]["stopped"] == 1
    state = await _saved(rig, "t-stream")
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER


# --- BD3 and G-78: stop without Redis ----------------------------------------------------


@pytest.mark.asyncio
async def test_stop_works_with_no_checkpointer():
    rig = Rig("hang", checkpointer=False)
    task = rig.start("t-nocp")
    await asyncio.wait_for(rig.started.wait(), 5)

    res = await rig.app.astop({"thread_id": "t-nocp"})

    assert res["ok"] is True
    await asyncio.wait_for(task, 5)


class FailingStopCheckpointer(InMemoryCheckpointer):
    """The cache write behind ``astop`` fails, as when Redis is down."""

    async def arequest_stop(self, config):
        raise ConnectionError("redis down")

    async def ais_stop_requested(self, config):
        raise ConnectionError("redis down")


@pytest.mark.asyncio
async def test_stop_survives_a_failing_cache_write():
    rig = Rig("hang", checkpointer=FailingStopCheckpointer())
    task = rig.start("t-redis")
    await asyncio.wait_for(rig.started.wait(), 5)

    res = await rig.app.astop({"thread_id": "t-redis"})

    assert res["ok"] is True
    await asyncio.wait_for(task, 5)
    state = await _saved(rig, "t-redis")
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "stop_requested"


@pytest.mark.asyncio
async def test_stop_of_an_unknown_thread_without_a_checkpointer_is_not_running():
    rig = Rig("short", checkpointer=False)
    res = await rig.app.astop({"thread_id": "nobody"})
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_local_stop_flag_does_not_leak_into_the_next_run():
    rig = Rig("hang", checkpointer=False)
    task = rig.start("t-leak")
    await asyncio.wait_for(rig.started.wait(), 5)
    await rig.app.astop({"thread_id": "t-leak"})
    await asyncio.wait_for(task, 5)

    rig.behavior = "short"
    result = await rig.app.ainvoke(_input(), {"thread_id": "t-leak"})
    assert result["messages"][-1].text() == "done"
