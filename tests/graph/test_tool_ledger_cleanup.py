"""The tool ledger is cleared after a completed run, kept for paused ones, and can be turned off.

A completed run has no node left to replay, so ``sync_data`` clears that run's ledger rows (those
keyed on its tool-calling messages) in the background once the COMPLETED state is saved. Every
other end of a run (interrupt_before / interrupt_after, ``interrupt()`` inside a tool, a client
tool pause, stop, error, cancel) keeps them, so the resumed node does not run its finished tools
again.
"""

import asyncio
import logging

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.graph.utils.tool_runtime import ToolRuntimeMixin
from tenxgraph.core.graph.utils.utils import _tool_origin_ids
from tenxgraph.core.state import AgentState, ExecutionStatus, Message, ToolCallBlock
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity, interrupt, tool
from tenxgraph.utils.constants import END
from tenxgraph.utils.decorators import get_tool_metadata


FULL = ResponseGranularity.FULL
MODES = ["ainvoke", "astream"]


class SpyCheckpointer(InMemoryCheckpointer):
    """Counts ledger clears, reads and writes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clears = 0
        self.cleared_ids: list[list[str] | None] = []
        self.reads: list[str] = []
        self.writes: list[str] = []

    async def aclear_tool_results(self, config, origin_message_ids=None):
        self.clears += 1
        self.cleared_ids.append(origin_message_ids)
        return await super().aclear_tool_results(config, origin_message_ids)

    async def aget_tool_result(self, config, tool_call_id):
        self.reads.append(tool_call_id)
        return await super().aget_tool_result(config, tool_call_id)

    async def aput_tool_result(self, config, tool_call_id, result):
        self.writes.append(tool_call_id)
        return await super().aput_tool_result(config, tool_call_id, result)

    def ledger_size(self) -> int:
        return len(self._tool_results)


def assistant_call(calls: list[tuple[str, str]]) -> Message:
    return Message(
        role="assistant",
        message_id="asst-1",
        content=[ToolCallBlock(id=cid, name=name, args={}) for cid, name in calls],
        tools_calls=[
            {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}
            for cid, name in calls
        ],
    )


def build(tools, calls, cp, *, fail_after_tools=False, **compile_kwargs):
    async def model(st: AgentState):
        if st.context[-1].role == "tool":
            if fail_after_tools:
                raise RuntimeError("model broke")
            return Message.text_message("done", role="assistant")
        return assistant_call(calls)

    def route(st: AgentState) -> str:
        last = st.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode(tools))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    return graph.compile(checkpointer=cp, **compile_kwargs)


async def run(app, mode, config, data=None):
    data = data or {"messages": [Message.text_message("go")]}
    if mode == "ainvoke":
        return await app.ainvoke(data, config, response_granularity=FULL)
    return [c async for c in app.astream(data, config, response_granularity=FULL)]


async def drained(app) -> None:
    await app._task_manager.wait_for_all(timeout=5)


async def work() -> str:
    return "worked"


# ------------------------------------------------------------------ completed run


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_completed_run_clears_its_ledger_in_the_background(mode):
    cp = SpyCheckpointer()
    app = build([work], [("c1", "work")], cp)
    config = {"thread_id": f"t-done-{mode}", "user_id": "u1"}

    await run(app, mode, config)
    await drained(app)

    assert cp.writes == ["asst-1:c1"]
    assert cp.clears == 1
    assert cp.cleared_ids == [["asst-1"]]  # only this run's rows, never the whole thread
    assert cp.ledger_size() == 0
    state = await cp.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_rows_a_newer_run_writes_before_the_clear_runs_survive():
    gate = asyncio.Event()

    class Gated(InMemoryCheckpointer):
        async def aclear_tool_results(self, config, origin_message_ids=None):
            await gate.wait()
            return await super().aclear_tool_results(config, origin_message_ids)

    cp = Gated()
    app = build([work], [("c1", "work")], cp)
    config = {"thread_id": "t-race", "user_id": "u1"}

    await run(app, "ainvoke", config)
    # The clear is pending. A newer run on the thread records its own call meanwhile.
    await cp.aput_tool_result(config, "asst-2:c1", {"v": "newer"})
    gate.set()
    await drained(app)

    assert await cp.aget_tool_result(config, "asst-1:c1") is None
    assert await cp.aget_tool_result(config, "asst-2:c1") == {"v": "newer"}


@pytest.mark.asyncio
async def test_rows_of_a_message_no_longer_in_the_state_stay():
    # A message trimmed out of the state before an earlier save is not in the final state, so
    # its rows are not cleared; aprune_tool_results removes them.
    cp = SpyCheckpointer()
    app = build([work], [("c1", "work")], cp)
    config = {"thread_id": "t-trimmed", "user_id": "u1"}
    await cp.aput_tool_result(config, "trimmed-msg:c1", {"v": 1})

    await run(app, "ainvoke", config)
    await drained(app)

    assert cp.cleared_ids == [["asst-1"]]
    assert await cp.aget_tool_result(config, "trimmed-msg:c1") == {"v": 1}
    assert cp.ledger_size() == 1


def test_origin_ids_cover_tools_calls_and_tool_call_blocks():
    by_field = Message(role="assistant", message_id=7, content=[], tools_calls=[{"id": "a"}])
    by_block = Message(
        role="assistant", message_id="b", content=[ToolCallBlock(id="x", name="n", args={})]
    )
    plain = Message.text_message("hi", role="assistant")
    assert _tool_origin_ids([by_field, by_block, plain, by_field]) == ["7", "b"]


@pytest.mark.asyncio
async def test_failing_clear_is_logged_and_does_not_fail_the_run(caplog):
    class Broken(InMemoryCheckpointer):
        async def aclear_tool_results(self, config, origin_message_ids=None):
            raise RuntimeError("db down")

    cp = Broken()
    app = build([work], [("c1", "work")], cp)
    with caplog.at_level(logging.WARNING, logger="tenxgraph.graph"):
        await run(app, "ainvoke", {"thread_id": "t-broken", "user_id": "u1"})
        await drained(app)
    assert any("Could not clear the tool ledger" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_clear_dropped_by_backpressure_is_logged_at_debug(caplog):
    cp = SpyCheckpointer()
    app = build([work], [("c1", "work")], cp)
    app._task_manager._max_pending_tasks = 1
    app._task_manager._tasks.add(asyncio.get_running_loop().create_future())  # queue full
    with caplog.at_level(logging.DEBUG, logger="tenxgraph.graph"):
        await run(app, "ainvoke", {"thread_id": "t-drop", "user_id": "u1"})
    assert cp.clears == 0
    assert cp.ledger_size() == 1
    assert any("dropped by backpressure" in r.getMessage() for r in caplog.records)
    app._task_manager._tasks.clear()


@pytest.mark.asyncio
async def test_aclose_waits_for_a_pending_clear_before_releasing_the_checkpointer():
    order: list[str] = []

    class Slow(InMemoryCheckpointer):
        async def aclear_tool_results(self, config, origin_message_ids=None):
            await asyncio.sleep(0.2)
            order.append("cleared")
            return await super().aclear_tool_results(config, origin_message_ids)

        async def arelease(self):
            order.append("released")
            return await super().arelease()

    cp = Slow()
    app = build([work], [("c1", "work")], cp)
    await run(app, "ainvoke", {"thread_id": "t-close", "user_id": "u1"})
    assert order == []  # the clear is still pending: the run did not wait for it

    await app.aclose()

    assert order == ["cleared", "released"]


# ------------------------------------------------------------------ runs that did not complete


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_interrupt_before_keeps_the_ledger(mode):
    cp = SpyCheckpointer()
    app = build([work], [("c1", "work")], cp, interrupt_before=["TOOL"])
    await run(app, mode, {"thread_id": f"t-before-{mode}", "user_id": "u1"})
    await drained(app)
    assert cp.clears == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_interrupt_after_keeps_the_ledger(mode):
    cp = SpyCheckpointer()
    app = build([work], [("c1", "work")], cp, interrupt_after=["TOOL"])
    await run(app, mode, {"thread_id": f"t-after-{mode}", "user_id": "u1"})
    await drained(app)
    assert cp.clears == 0
    assert cp.ledger_size() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_interrupt_inside_a_tool_keeps_the_ledger_and_resume_skips_finished_tools(mode):
    runs: list[str] = []

    async def lookup() -> str:
        runs.append("lookup")
        return "found"

    async def refund() -> str:
        runs.append("refund")
        return "refunded" if interrupt("ok?") == "yes" else "declined"

    cp = SpyCheckpointer()
    app = build([lookup, refund], [("a", "lookup"), ("b", "refund")], cp)
    config = {"thread_id": f"t-pause-{mode}", "user_id": "u1"}

    await run(app, mode, config)
    await drained(app)
    assert cp.clears == 0
    assert await cp.aget_tool_result(config, "asst-1:a") is not None

    await run(app, mode, config, {"resume": "yes"})
    await drained(app)

    assert runs.count("lookup") == 1  # the finished tool was replayed from the ledger
    assert runs.count("refund") == 2  # the paused one re-ran and got the answer
    assert cp.clears == 1  # the resumed run completed
    assert cp.cleared_ids == [["asst-1"]]  # covers the rows of the paused invocation too
    assert cp.ledger_size() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_client_tool_pause_keeps_the_ledger(mode):
    cp = SpyCheckpointer()
    app = build([work], [("c1", "work"), ("c2", "pick_color")], cp)
    config = {
        "thread_id": f"t-remote-{mode}",
        "user_id": "u1",
        "remote_tools": [{"name": "pick_color", "description": "Pick."}],
    }
    await run(app, mode, config)
    await drained(app)
    state = await cp.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert cp.clears == 0
    assert await cp.aget_tool_result(config, "asst-1:c1") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_error_keeps_the_ledger(mode):
    cp = SpyCheckpointer()
    app = build([work], [("c1", "work")], cp, fail_after_tools=True)
    with pytest.raises(Exception, match="model broke"):
        await run(app, mode, {"thread_id": f"t-error-{mode}", "user_id": "u1"})
    await drained(app)
    assert cp.clears == 0
    assert cp.ledger_size() == 1


def _hanging(started: asyncio.Event):
    async def finished() -> str:
        return "ok"

    async def hang() -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    return [finished, hang]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_stop_keeps_the_ledger(mode):
    started = asyncio.Event()
    cp = SpyCheckpointer()
    app = build(_hanging(started), [("a", "finished"), ("b", "hang")], cp)
    config = {"thread_id": f"t-stop-{mode}", "user_id": "u1"}

    task = asyncio.create_task(run(app, mode, config))
    await asyncio.wait_for(started.wait(), 5)
    await app.astop({"thread_id": config["thread_id"], "user_id": "u1"})
    await asyncio.wait_for(task, 10)
    await drained(app)

    state = await cp.aget_state(config)
    assert state.execution_meta.interrupt_reason == "stop_requested"
    assert cp.clears == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_cancel_keeps_the_ledger(mode):
    started = asyncio.Event()
    cp = SpyCheckpointer()
    app = build(_hanging(started), [("a", "finished"), ("b", "hang")], cp)
    config = {"thread_id": f"t-cancel-{mode}", "user_id": "u1"}

    task = asyncio.create_task(run(app, mode, config))
    await asyncio.wait_for(started.wait(), 5)
    await asyncio.sleep(0.05)  # let the finished tool record itself
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await drained(app)

    state = await cp.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert cp.clears == 0
    assert await cp.aget_tool_result(config, "asst-1:a") is not None


# ------------------------------------------------------------------ record_tool_result flags


@tool(record_tool_result=False)
async def fresh_lookup() -> str:
    return "fresh"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tool_flag_off_skips_read_and_write_for_that_tool_only(mode):
    cp = SpyCheckpointer()
    app = build([work, fresh_lookup], [("c1", "work"), ("c2", "fresh_lookup")], cp)
    await run(app, mode, {"thread_id": f"t-toolflag-{mode}", "user_id": "u1"})

    assert cp.reads == ["asst-1:c1"]
    assert cp.writes == ["asst-1:c1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_checkpointer_flag_off_skips_read_write_and_clear(mode):
    cp = SpyCheckpointer(record_tool_result=False)
    app = build([work], [("c1", "work")], cp)
    await run(app, mode, {"thread_id": f"t-cpflag-{mode}", "user_id": "u1"})
    await drained(app)

    assert cp.reads == []
    assert cp.writes == []
    assert cp.clears == 0


def test_checkpointer_off_wins_over_the_tool_flag():
    on = InMemoryCheckpointer()
    off = InMemoryCheckpointer(record_tool_result=False)
    node = ToolNode([work, fresh_lookup])

    assert ToolRuntimeMixin._ledger_enabled(on, node, "work") is True
    assert ToolRuntimeMixin._ledger_enabled(on, node, "fresh_lookup") is False
    assert ToolRuntimeMixin._ledger_enabled(off, node, "work") is False
    # MCP, skill and client tools are not in the node's functions: recorded by default.
    assert ToolRuntimeMixin._ledger_enabled(on, node, "pick_color") is True
    assert ToolRuntimeMixin._ledger_enabled(None, node, "work") is False


def test_compile_logs_when_the_checkpointer_does_not_record(caplog):
    with caplog.at_level(logging.INFO, logger="tenxgraph.graph"):
        build([work], [("c1", "work")], InMemoryCheckpointer(record_tool_result=False))
    assert any("record_tool_result=False" in r.getMessage() for r in caplog.records)


def test_tool_decorator_stores_the_flag():
    @tool
    def plain() -> str:
        return "x"

    assert get_tool_metadata(fresh_lookup)["record_tool_result"] is False
    assert get_tool_metadata(plain)["record_tool_result"] is True
