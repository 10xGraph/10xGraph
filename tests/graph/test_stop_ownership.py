"""``astop`` is owner-scoped (AUTH-S0-12) and its cross-process flag uses the thread owner's
keys (AUTH-S0-04).

A caller may stop a run when its reach covers the run's owner: ``own`` the same user and
tenant values, ``tenant`` the same tenant values, ``all`` anyone. A refused stop answers like
a thread with no run. With no identity in the config every run on the thread is stopped.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.authz import Identity, can_reach
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.thread_info import ThreadInfo


NO_RUN = {"ok": False, "running": False, "reason": "no-state"}


def _cfg(thread_id: str, user_id: str, reach: str = "own", **tenant: str) -> dict:
    user = {"user_id": user_id, "reach": reach, "tenant": tenant}
    return {"thread_id": thread_id, "user_id": user_id, "user": user}


class Rig:
    """A one-node graph whose node hangs until stopped (or released)."""

    def __init__(self, checkpointer=None):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        graph = StateGraph()
        graph.add_node("WORK", self._work)
        graph.add_edge("WORK", END)
        graph.set_entry_point("WORK")
        self.app = graph.compile(checkpointer=checkpointer, owns_resources=False)

    async def _work(self, state: AgentState):
        self.started.set()
        await self.release.wait()
        return Message.text_message("done", role="assistant")

    async def start(self, config: dict) -> asyncio.Task:
        task = asyncio.create_task(
            self.app.ainvoke({"messages": [Message.text_message("hi")]}, config)
        )
        await asyncio.wait_for(self.started.wait(), 5)
        return task


async def _still_running(task: asyncio.Task) -> bool:
    await asyncio.sleep(0.3)
    return not task.done()


async def _finish(rig: Rig, task: asyncio.Task) -> None:
    rig.release.set()
    await asyncio.wait_for(task, 5)


# --- the rule ------------------------------------------------------------------------------


def test_can_reach_rule():
    a = {"org": "o1"}
    assert can_reach({"user_id": "x"}, "A", a)  # no identity: unrestricted
    assert can_reach(_cfg("t", "A", org="o1"), "A", a)
    assert not can_reach(_cfg("t", "B", org="o1"), "A", a)
    assert not can_reach(_cfg("t", "A", org="o2"), "A", a)  # same user, other tenant
    assert not can_reach(_cfg("t", "A"), "A", a)  # tenant value missing
    assert can_reach(_cfg("t", "B", "tenant", org="o1"), "A", a)
    assert not can_reach(_cfg("t", "B", "tenant", org="o2"), "A", a)
    assert not can_reach(_cfg("t", "B", "tenant", org="o1"), "A", {})  # no tenant to match
    assert not can_reach(_cfg("t", "B", "tenant", org="o1"), "A", {"org": None})
    assert not can_reach(_cfg("t", "B", "tenant"), "A", {"org": None})
    assert not can_reach(_cfg("t", "A", org="o1"), "A", {"org": None})
    assert can_reach(_cfg("t", "A"), "A", {"org": None})  # legacy row: owner without value
    assert can_reach(_cfg("t", "B", "all"), "A", a)
    assert can_reach(_cfg("t", "A"), "A", {})


# --- in-process stop -----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointer", [False, True])
async def test_other_user_cannot_stop_and_owner_can(checkpointer):
    rig = Rig(InMemoryCheckpointer() if checkpointer else None)
    task = await rig.start(_cfg("t1", "A"))

    refused = await rig.app.astop(_cfg("t1", "B"))
    unknown = await rig.app.astop(_cfg("nope", "B"))

    assert refused == unknown  # never reveals that the run exists
    assert refused["ok"] is False
    assert await _still_running(task)

    res = await rig.app.astop(_cfg("t1", "A"))
    assert res["ok"] is True
    await asyncio.wait_for(task, 5)
    assert not rig.release.is_set()  # ended by the stop, not by the node


class OrgCheckpointer(InMemoryCheckpointer):
    """In-memory stand-in for a backend with an ``org`` tenant column."""

    tenant_column_names = ("org",)


@pytest.mark.asyncio
async def test_tenant_reach_stops_runs_of_its_tenant_only():
    rig = Rig(OrgCheckpointer())
    task = await rig.start(_cfg("t1", "A", org="o1"))

    assert (await rig.app.astop(_cfg("t1", "B", "tenant", org="o2")))["ok"] is False
    assert (await rig.app.astop(_cfg("t1", "A", org="o2")))["ok"] is False  # own, other org
    assert await _still_running(task)

    assert (await rig.app.astop(_cfg("t1", "B", "tenant", org="o1")))["ok"] is True
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_reach_all_stops_anyone():
    rig = Rig(InMemoryCheckpointer())
    task = await rig.start(_cfg("t1", "A", org="o1"))

    assert (await rig.app.astop(_cfg("t1", "admin", "all")))["ok"] is True
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_no_identity_is_unrestricted():
    rig = Rig(InMemoryCheckpointer())
    task = await rig.start(_cfg("t1", "A"))

    assert (await rig.app.astop({"thread_id": "t1", "user_id": "someone"}))["ok"] is True
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_run_without_identity_is_owned_by_its_user_id():
    rig = Rig()
    task = await rig.start({"thread_id": "t1", "user_id": "A"})

    assert (await rig.app.astop(_cfg("t1", "B")))["ok"] is False
    assert await _still_running(task)
    assert (await rig.app.astop(_cfg("t1", "A")))["ok"] is True
    await asyncio.wait_for(task, 5)


# --- cross-process stop (two graphs sharing one checkpointer) -------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def shared_cp(request):
    return InMemoryCheckpointer() if request.param == "memory" else SqliteCheckpointer(":memory:")


async def _two_workers(cp, owner_cfg: dict) -> tuple[Rig, Rig, asyncio.Task]:
    """Worker 1 runs the owner's run; worker 2 (no local run) takes the stop request."""
    await cp.aput_thread(owner_cfg, ThreadInfo(thread_id=owner_cfg["thread_id"], user_id="A"))
    worker1, worker2 = Rig(cp), Rig(cp)
    task = await worker1.start(owner_cfg)
    return worker1, worker2, task


@pytest.mark.asyncio
async def test_admin_stop_reaches_the_owners_run_on_another_worker(shared_cp):
    worker1, worker2, task = await _two_workers(shared_cp, _cfg("t1", "A"))

    res = await worker2.app.astop(_cfg("t1", "admin", "all"))

    assert res == {"ok": True, "running": True}
    await asyncio.wait_for(task, 10)
    assert not worker1.release.is_set()
    # Nothing was written under the admin's own keys.
    assert await shared_cp.ais_stop_requested({"thread_id": "t1", "user_id": "admin"}) is False


@pytest.mark.asyncio
async def test_other_user_stop_on_another_worker_is_refused(shared_cp):
    worker1, worker2, task = await _two_workers(shared_cp, _cfg("t1", "A"))

    refused = await worker2.app.astop(_cfg("t1", "B"))
    unknown = await worker2.app.astop(_cfg("nope", "B"))

    assert refused == unknown == NO_RUN
    assert await shared_cp.ais_stop_requested({"thread_id": "t1", "user_id": "A"}) is False
    assert await shared_cp.ais_stop_requested({"thread_id": "t1", "user_id": "B"}) is False
    assert await _still_running(task)
    await _finish(worker1, task)


@pytest.mark.asyncio
async def test_owner_stop_on_another_worker(shared_cp):
    worker1, worker2, task = await _two_workers(shared_cp, _cfg("t1", "A"))

    assert (await worker2.app.astop(_cfg("t1", "A")))["ok"] is True
    await asyncio.wait_for(task, 10)
    assert not worker1.release.is_set()


# --- PgCheckpointer (mocked Postgres, dict-backed Redis) ----------------------------------


class _FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def setex(self, key, ttl, value):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, *keys):
        return sum(1 for k in keys if self.data.pop(k, None) is not None)


def _pg_app(owner: str, column_type: str = "text", **tenant):
    with patch("asyncpg.create_pool", return_value=MagicMock()):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            cache_ttl=3600,
            tenant_columns=dict.fromkeys(tenant, column_type),
        )
    cp.redis = _FakeRedis()
    running = AgentState()
    running.execution_meta.status = ExecutionStatus.RUNNING
    row = {"user_id": owner, **tenant}
    cp.aget_thread_scope = AsyncMock(return_value=row)

    async def read_row(thread_id, user_id, config):  # the scoped thread-row read
        cp._remember_identity(config, thread_id, row)
        return row

    cp._read_thread_identity = read_row
    cp.aget_state_cache = AsyncMock(return_value=running)
    cp.aput_state_cache = AsyncMock()
    graph = StateGraph()
    graph.add_node("WORK", lambda state: state)
    graph.add_edge("WORK", END)
    graph.set_entry_point("WORK")
    return graph.compile(checkpointer=cp, owns_resources=False), cp


@pytest.mark.asyncio
async def test_pg_admin_stop_is_keyed_by_the_thread_owner():
    app, cp = _pg_app("A")

    res = await app.astop(_cfg("t1", "admin", "all"))

    assert res == {"ok": True, "running": True}
    assert await cp.ais_stop_requested({"thread_id": "t1", "user_id": "A"}) is True
    assert await cp.ais_stop_requested({"thread_id": "t1", "user_id": "admin"}) is False
    # The state-cache mirror is written under the owner's identity too.
    written_cfg = cp.aput_state_cache.await_args.args[0]
    assert written_cfg["user"].user_id == "A"


@pytest.mark.asyncio
async def test_pg_tenant_columns_admin_stop_reaches_the_owners_keys():
    app, cp = _pg_app("A", org="o1")

    res = await app.astop(_cfg("t1", "admin", "tenant", org="o1"))

    assert res == {"ok": True, "running": True}
    assert await cp.ais_stop_requested(_cfg("t1", "A", org="o1")) is True
    assert all("t1:admin" not in key for key in cp.redis.data)


@pytest.mark.asyncio
async def test_pg_tenant_columns_other_tenant_is_refused():
    app, cp = _pg_app("A", org="o1")
    assert await app.astop(_cfg("t1", "admin", "tenant", org="o2")) == NO_RUN
    assert cp.redis.data == {}


@pytest.mark.asyncio
async def test_pg_legacy_row_without_tenant_value_is_stoppable_by_its_owner():
    # AR-CORE-04: a NULL tenant value matches a caller without that value (as storage writes
    # compare it); the stop lands on the owner's "org!" keys.
    app, cp = _pg_app("A", org=None)  # row written before the column existed
    assert await app.astop(_cfg("t1", "admin", "tenant", org="o1")) == NO_RUN
    assert await app.astop(_cfg("t1", "A", org="o1")) == NO_RUN  # storage hides it too
    assert cp.redis.data == {}

    assert (await app.astop(_cfg("t1", "A")))["ok"] is True
    assert await cp.ais_stop_requested(_cfg("t1", "A")) is True
    assert cp._stop_request_key(_cfg("t1", "A")) == "t1:A:t:org!"
    assert len(cp.redis.data) == 1
    cp.redis.data.clear()
    assert (await app.astop(_cfg("t1", "admin", "all")))["ok"] is True
    assert await cp.ais_stop_requested(_cfg("t1", "A")) is True


@pytest.mark.asyncio
async def test_pg_uuid_tenant_claim_in_upper_case_matches_its_row():
    # AR-CORE-04: values are compared converted by the column type, not as raw text.
    org = uuid.uuid4()
    app, cp = _pg_app("A", "uuid", org=org)
    owner_cfg = _cfg("t1", "A", org=str(org).upper())

    assert (await app.astop(owner_cfg))["ok"] is True
    assert await cp.ais_stop_requested(owner_cfg) is True
    assert await app.astop(_cfg("t1", "A", org=str(uuid.uuid4()).upper())) == NO_RUN


@pytest.mark.asyncio
async def test_identity_without_user_id_stops_across_workers():
    # AR-CORE-05: a direct SDK run with only config["user"] polls the identity's keys.
    cp = InMemoryCheckpointer()
    cfg = {"thread_id": "t1", "user": Identity(user_id="alice")}
    await cp.aput_thread(cfg, ThreadInfo(thread_id="t1"))
    worker1, worker2 = Rig(cp), Rig(cp)
    task = await worker1.start(dict(cfg))

    assert (await worker2.app.astop(dict(cfg)))["ok"] is True
    await asyncio.wait_for(task, 10)
    assert not worker1.release.is_set()
    assert worker1.app._prepare_config(dict(cfg))["user_id"] == "alice"


@pytest.mark.asyncio
async def test_pg_refused_stop_writes_nothing():
    app, cp = _pg_app("A")

    assert await app.astop(_cfg("t1", "B")) == NO_RUN
    assert cp.redis.data == {}
    cp.aget_state_cache.assert_not_awaited()


# --- failure modes -------------------------------------------------------------------------


class FailingCacheCheckpointer(InMemoryCheckpointer):
    """Every read behind ``astop`` fails, as when Redis or Postgres is down."""

    async def aget_thread_scope(self, thread_id):
        raise ConnectionError("cache down")

    async def ais_stop_requested(self, config):
        raise ConnectionError("cache down")


@pytest.mark.asyncio
async def test_failing_cache_still_stops_the_owners_local_run():
    rig = Rig(FailingCacheCheckpointer())
    task = await rig.start(_cfg("t1", "A"))

    res = await rig.app.astop(_cfg("t1", "A"))

    assert res == {"ok": True, "running": True, "local": True}
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_failing_cache_never_stops_another_users_run():
    rig = Rig(FailingCacheCheckpointer())
    task = await rig.start(_cfg("t1", "A"))

    with pytest.raises(ConnectionError):
        await rig.app.astop(_cfg("t1", "B"))
    # The error is the same for a thread with no run: it reveals nothing.
    with pytest.raises(ConnectionError):
        await rig.app.astop(_cfg("nope", "B"))

    assert await _still_running(task)
    await _finish(rig, task)


@pytest.mark.asyncio
async def test_stop_during_shutdown_is_still_owner_scoped():
    rig = Rig(InMemoryCheckpointer())
    task = await rig.start(_cfg("t1", "A"))
    closing = asyncio.create_task(rig.app.aclose(grace=5, stop_timeout=2, timeout=2))
    await asyncio.sleep(0.1)  # aclose is in its grace phase

    assert (await rig.app.astop(_cfg("t1", "B")))["ok"] is False
    assert await _still_running(task)

    assert (await rig.app.astop(_cfg("t1", "A")))["ok"] is True
    await asyncio.wait_for(task, 5)
    stats = await asyncio.wait_for(closing, 10)
    assert stats["runs"]["finished"] == 1  # ended by the owner's stop within the grace
    assert not rig.release.is_set()
