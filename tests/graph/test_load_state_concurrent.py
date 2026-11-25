"""Run-start state load: the marker write and the state read run together.

``load_or_create_state`` reads the active-run marker first (crash recovery needs the run that
was alive before this one), then registers the new run and reads the saved state at the same
time. These tests pin the order that matters, the overlap, and the error behaviour.
"""

import asyncio

import pytest

from tenxgraph.core.graph.utils.utils import load_or_create_state
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.core.state.execution_state import ExecutionState
from tenxgraph.storage.checkpointer import InMemoryCheckpointer, base_checkpointer
from tenxgraph.utils.constants import START

from .test_crash_recovery import build, seed_crashed


_BASE = InMemoryCheckpointer


class RecordingCheckpointer(InMemoryCheckpointer):
    """Logs when each run-start call begins and ends; each call yields to the loop once."""

    def __init__(self, fail: str | None = None):
        super().__init__()
        self.log: list[str] = []
        self.fail = fail

    async def _traced(self, name: str, call):
        self.log.append(f"{name}:start")
        await asyncio.sleep(0.01)
        if self.fail == name:
            self.log.append(f"{name}:failed")
            raise RuntimeError(f"{name} failed")
        result = await call()
        self.log.append(f"{name}:end")
        return result

    async def aget_active_run(self, config):
        return await self._traced("active", lambda: _BASE.aget_active_run(self, config))

    async def abegin_run(self, config):
        return await self._traced("begin", lambda: _BASE.abegin_run(self, config))

    async def aget_state_cache(self, config):
        return await self._traced("cache", lambda: _BASE.aget_state_cache(self, config))

    async def aget_state(self, config):
        return await self._traced("state", lambda: _BASE.aget_state(self, config))


def new_input(text: str = "hi") -> dict:
    return {"messages": [Message.text_message(text)]}


def pending_tasks() -> set[asyncio.Task]:
    return {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}


@pytest.mark.asyncio
async def test_new_thread_reads_marker_first_then_begin_and_state_reads_together():
    cp = RecordingCheckpointer()
    config = {"thread_id": "t-new", "user_id": "u1", "run_id": "r1"}

    state = await load_or_create_state(new_input(), config, AgentState(), checkpointer=cp)

    log = cp.log
    # Every read still happens: the cache misses on a new thread, so the durable read follows.
    for name in ("active", "begin", "cache", "state"):
        assert f"{name}:start" in log and f"{name}:end" in log
    # The marker read is over before anything else starts.
    assert log[:2] == ["active:start", "active:end"]
    # The marker write and the state read overlap.
    assert log.index("cache:start") < log.index("begin:end")
    assert log.index("cache:end") < log.index("state:start")
    assert await cp.aget_active_run(config) == "r1"
    assert state.execution_meta.current_node == START
    assert state.execution_meta.run_id == "r1"
    assert [m.text() for m in state.context] == ["hi"]


@pytest.mark.asyncio
async def test_cache_hit_skips_the_durable_read():
    cp = RecordingCheckpointer()
    config = {"thread_id": "t-hit", "user_id": "u1", "run_id": "r2"}
    saved = AgentState(context=[Message.text_message("before")])
    saved.execution_meta = ExecutionState(current_node=START, status=ExecutionStatus.COMPLETED)
    await cp.aput_state_cache(config, saved)

    state = await load_or_create_state(new_input("next"), config, AgentState(), checkpointer=cp)

    assert "state:start" not in cp.log
    assert [m.text() for m in state.context] == ["before", "next"]


@pytest.mark.asyncio
async def test_dead_run_is_still_recovered():
    cp = RecordingCheckpointer()
    build(cp)  # binds the checkpointer in the container recovery resolves it from
    config = {"thread_id": "t-dead", "user_id": "u1", "run_id": "new-run"}
    await seed_crashed(cp, config, run_id="dead-run")

    state = await load_or_create_state(new_input("again"), config, AgentState(), checkpointer=cp)

    # Repaired (tool call closed, marked crashed), then the new message starts a new turn.
    assert any(m.role == "tool" for m in state.context)
    assert state.context[-1].text() == "again"
    assert state.execution_meta.current_node == START
    saved = await cp.aget_state(config)
    assert saved.execution_meta.interrupt_reason == "crashed"


@pytest.mark.asyncio
async def test_live_run_is_not_recovered():
    cp = RecordingCheckpointer()
    build(cp)
    config = {"thread_id": "t-live2", "user_id": "u1", "run_id": "new-run"}
    await seed_crashed(cp, config, run_id="alive-run")
    await cp.aput_cache_value(base_checkpointer.ACTIVE_RUN_NAMESPACE, "t-live2:u1", "alive-run", 60)

    state = await load_or_create_state({}, config, AgentState(), checkpointer=cp)

    assert state.execution_meta.status == ExecutionStatus.RUNNING
    assert not any(m.role == "tool" for m in state.context)
    assert await cp.aget_state(config) is None  # nothing was repaired or saved


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["begin", "cache", "state"])
async def test_a_failure_is_raised_after_both_calls_finish(failing):
    cp = RecordingCheckpointer(fail=failing)
    config = {"thread_id": f"t-fail-{failing}", "user_id": "u1", "run_id": "r3"}

    with pytest.raises(RuntimeError, match=f"{failing} failed"):
        await load_or_create_state(new_input(), config, AgentState(), checkpointer=cp)

    # Neither side is left running in the background.
    assert not pending_tasks()
    other_side = "cache" if failing == "begin" else "begin"
    assert f"{other_side}:end" in cp.log


@pytest.mark.asyncio
async def test_both_failing_raises_the_marker_write_error():
    class BothFail(RecordingCheckpointer):
        async def aget_state_cache(self, config):
            await asyncio.sleep(0)
            raise ValueError("cache failed")

    cp = BothFail(fail="begin")
    config = {"thread_id": "t-both", "user_id": "u1", "run_id": "r4"}

    with pytest.raises(RuntimeError, match="begin failed"):
        await load_or_create_state(new_input(), config, AgentState(), checkpointer=cp)
    assert not pending_tasks()


@pytest.mark.asyncio
async def test_cancel_during_the_load_cancels_both_calls():
    cp = RecordingCheckpointer()
    config = {"thread_id": "t-cancel", "user_id": "u1", "run_id": "r5"}
    task = asyncio.ensure_future(
        load_or_create_state(new_input(), config, AgentState(), checkpointer=cp)
    )
    while "begin:start" not in cp.log:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert not pending_tasks()
