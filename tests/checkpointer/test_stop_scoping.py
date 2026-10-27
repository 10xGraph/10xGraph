"""A stop request belongs to the run it targets; a new run clears only a stale one."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


def _cfg(run_id: str | None = None) -> dict:
    cfg = {"thread_id": "t1", "user_id": "u1"}
    if run_id:
        cfg["run_id"] = run_id
    return cfg


class _FakeRedis:
    """Dict-backed stand-in for the redis calls PgCheckpointer's generic cache makes."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def setex(self, key, ttl, value):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        return 1 if self.data.pop(key, None) is not None else 0


def _pg() -> PgCheckpointer:
    with patch("asyncpg.create_pool", return_value=MagicMock()):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            cache_ttl=3600,
        )
    cp.redis = _FakeRedis()
    return cp


@pytest.fixture(params=["memory", "sqlite", "pg"])
def cp(request):
    if request.param == "memory":
        return InMemoryCheckpointer()
    if request.param == "sqlite":
        return SqliteCheckpointer(":memory:")
    return _pg()


@pytest.mark.asyncio
async def test_stale_stop_with_no_active_run_is_cleared(cp):
    await cp.arequest_stop(_cfg())  # no run was active: targets nobody
    assert await cp.ais_stop_requested(_cfg()) is True

    await cp.abegin_run(_cfg("r2"))

    assert await cp.ais_stop_requested(_cfg("r2")) is False
    assert await cp.aget_active_run(_cfg()) == "r2"


@pytest.mark.asyncio
async def test_stop_for_a_run_that_ended_is_cleared(cp):
    await cp.abegin_run(_cfg("r1"))
    await cp.arequest_stop(_cfg())  # targets r1
    await cp.aend_run(_cfg("r1"))  # r1 finished before it saw the stop

    await cp.abegin_run(_cfg("r2"))

    assert await cp.ais_stop_requested(_cfg("r2")) is False


@pytest.mark.asyncio
async def test_stop_for_a_live_run_survives_a_new_run_and_only_that_run_sees_it(cp):
    await cp.abegin_run(_cfg("r1"))
    await cp.arequest_stop(_cfg())  # targets r1

    await cp.abegin_run(_cfg("r2"))  # a second run starts on the same thread

    assert await cp.ais_stop_requested(_cfg("r1")) is True
    assert await cp.ais_stop_requested(_cfg("r2")) is False
    # A caller without a run id (legacy) still sees the flag.
    assert await cp.ais_stop_requested(_cfg()) is True

    # r2 ending must not release r1's stop; r1 consuming it clears it.
    await cp.aend_run(_cfg("r2"))
    assert await cp.ais_stop_requested(_cfg("r1")) is True
    await cp.aclear_stop_request(_cfg("r1"))
    assert await cp.ais_stop_requested(_cfg("r1")) is False


@pytest.mark.asyncio
async def test_normal_stop_still_stops_the_current_run(cp):
    await cp.abegin_run(_cfg("r1"))
    assert await cp.ais_stop_requested(_cfg("r1")) is False

    await cp.arequest_stop(_cfg())

    assert await cp.ais_stop_requested(_cfg("r1")) is True
    await cp.aclear_stop_request(_cfg("r1"))
    assert await cp.ais_stop_requested(_cfg("r1")) is False


@pytest.mark.asyncio
async def test_legacy_boolean_flag_is_still_read():
    cp = InMemoryCheckpointer()
    await cp.aput_cache_value("stop_request", "t1:u1", True, 60)
    assert await cp.ais_stop_requested(_cfg("r1")) is True
    await cp.abegin_run(_cfg("r2"))
    assert await cp.ais_stop_requested(_cfg("r2")) is False


@pytest.mark.asyncio
async def test_live_run_keeps_its_stop_when_a_second_run_starts_end_to_end():
    """Run 1 is stopped, run 2 starts before run 1 notices: run 1 still stops."""
    started = asyncio.Event()
    release = asyncio.Event()
    ran: list[str] = []

    async def slow(state: AgentState):
        label = state.context[-1].text()
        ran.append(label)
        if label == "one":
            started.set()
            await release.wait()
        return Message.text_message(f"done {label}", role="assistant")

    graph = StateGraph()
    graph.add_node("SLOW", slow)
    graph.add_edge("SLOW", END)
    graph.set_entry_point("SLOW")
    app = graph.compile(checkpointer=InMemoryCheckpointer())

    full = ResponseGranularity.FULL
    cfg1 = {"thread_id": "e2e", "run_id": "run-1"}
    task1 = asyncio.create_task(
        app.ainvoke({"messages": [Message.text_message("one")]}, cfg1, response_granularity=full)
    )
    await asyncio.wait_for(started.wait(), 5)

    await app._checkpointer.arequest_stop({"thread_id": "e2e", "user_id": "anonymous"})

    # Run 2 starts on the same thread while run 1 is still active; it must not erase the stop.
    task2 = asyncio.create_task(
        app.ainvoke(
            {"messages": [Message.text_message("two")]},
            {"thread_id": "e2e", "run_id": "run-2"},
            response_granularity=full,
        )
    )
    # Run 1 never releases the node itself: only the stop can end it.
    res1, res2 = await asyncio.wait_for(asyncio.gather(task1, task2), 10)

    # Had run 2 erased the stop, run 1 would still be waiting on the event and the gather
    # above would time out. Run 1 ended because its stop was honoured, and run 2 was not stopped.
    assert not release.is_set()
    assert "done one" not in [m.text() for m in res1["messages"]]
    assert "done two" in [m.text() for m in res2["messages"]]
    assert (
        await app._checkpointer.ais_stop_requested({"thread_id": "e2e", "user_id": "anonymous"})
        is False
    )
    assert ran.count("two") == 1
