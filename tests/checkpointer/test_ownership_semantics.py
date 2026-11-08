"""Ownership semantics of the checkpointers (auth plan A10, A13, P4, P8).

- A write to another user's (or tenant's) thread raises ``ThreadNotFoundError`` and writes
  nothing, whatever the caller's reach: writes always stay own.
- Deletes return the number of rows removed, 0 when the caller's scope matched nothing.
- Reach ``tenant`` / ``all`` widen reads, lists and deletes.
- ``aget_thread`` reports the row owner, not the caller.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from tenxgraph.core.exceptions import ThreadNotFoundError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import sqlite_checkpointer
from tenxgraph.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.utils.thread_info import ThreadInfo


pytest.importorskip("aiosqlite")
pytestmark = pytest.mark.asyncio


def _cfg(thread="t1", user="alice", reach="own"):
    return {"thread_id": thread, "user_id": user, "user": {"user_id": user, "reach": reach}}


def _state(text: str) -> AgentState:
    state = AgentState()
    state.context = [Message.text_message(text, role="user")]
    return state


def _texts(state) -> list[str]:
    return [m.text() for m in state.context]


def _msg(message_id: str, text: str = "hi") -> Message:
    return Message.text_message(text, role="user", message_id=message_id)


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def cp(request):
    if request.param == "memory":
        checkpointer = InMemoryCheckpointer()
    else:
        checkpointer = sqlite_checkpointer.SqliteCheckpointer(":memory:")
    yield checkpointer
    await checkpointer.arelease()


async def _seed(cp, thread="t1", user="alice"):
    """A thread with state, two messages, thread info and a tool result."""
    cfg = _cfg(thread, user)
    await cp.aput_thread(cfg, ThreadInfo(thread_id=thread, thread_name=f"{user} thread"))
    await cp.aput_state(_cfg(thread, user), _state(f"{user} state"))
    await cp.aput_messages(_cfg(thread, user), [_msg("m1"), _msg("m2")])
    await cp.aput_tool_result(_cfg(thread, user), "call-1", {"ok": True})


async def _assert_untouched(cp, thread="t1", user="alice"):
    cfg = lambda: _cfg(thread, user)  # noqa: E731
    assert (await cp.aget_thread(cfg())).thread_name == f"{user} thread"
    assert _texts(await cp.aget_state(cfg())) == [f"{user} state"]
    assert [str(m.message_id) for m in await cp.alist_messages(cfg())] == ["m1", "m2"]
    assert (await cp.aget_message(cfg(), "m1")).metadata.get("x") is None
    assert await cp.aget_tool_result(cfg(), "call-1") == {"ok": True}
    assert await cp.aget_tool_result(cfg(), "call-2") is None


# ---------------------------------------------------------------------------
# InMemory and SQLite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reach", ["own", "all"])
async def test_foreign_write_raises_not_found_and_writes_nothing(cp, reach):
    await _seed(cp)
    bob = lambda: _cfg(user="bob", reach=reach)  # noqa: E731

    with pytest.raises(ThreadNotFoundError) as err:
        await cp.aput_state(bob(), _state("hijack"))
    assert err.value.error_code == "STORAGE_NOT_FOUND_001"
    assert "owned" not in str(err.value)  # the message never reveals that the id exists
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_messages(bob(), [_msg("m3", "hijack")])
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_thread(bob(), ThreadInfo(thread_id="t1", thread_name="hijack"))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_tool_result(bob(), "call-2", {"ok": False})
    assert await cp.aupdate_message_metadata(bob(), "m1", {"x": 1}) is None
    await cp.aput_state_cache(bob(), _state("hijack"))  # dropped, never raises

    await _assert_untouched(cp)
    assert _texts(await cp.aget_state_cache(_cfg())) == ["alice state"]


async def test_foreign_deletes_return_zero_and_touch_nothing(cp):
    await _seed(cp)
    bob = lambda: _cfg(user="bob")  # noqa: E731

    assert await cp.adelete_message(bob(), "m1") == 0
    assert await cp.aclear_state(bob()) == 0
    assert await cp.aclean_thread(bob()) == 0

    await _assert_untouched(cp)


async def test_owner_deletes_return_counts(cp):
    await _seed(cp)

    assert await cp.adelete_message(_cfg(), "m1") == 1
    assert await cp.adelete_message(_cfg(), "m1") == 0  # already gone
    assert await cp.adelete_message(_cfg(), "nope") == 0
    assert await cp.aclear_state(_cfg()) == 1
    assert await cp.aclear_state(_cfg()) == 0
    assert await cp.aclean_thread(_cfg()) == 1
    assert await cp.aclean_thread(_cfg()) == 0
    assert await cp.aget_thread(_cfg()) is None


async def test_thread_without_record_counts_as_deleted(cp):
    # AR-CORE-07: a direct SDK run writes state and messages but no thread record.
    await cp.aput_state(_cfg(), _state("alice state"))
    await cp.aput_messages(_cfg(), [_msg("m1")])

    assert await cp.aclean_thread(_cfg(user="bob")) == 0
    assert await cp.aclean_thread(_cfg()) == 1
    assert await cp.aget_state(_cfg()) is None
    assert await cp.alist_messages(_cfg()) == []
    assert await cp.aclean_thread(_cfg()) == 0


async def test_thread_stays_owned_until_its_data_is_gone():
    # AR-CORE-06: while the delete waits for a data lock, another user cannot claim the
    # thread and write into data that is about to be removed.
    cp = InMemoryCheckpointer()
    await _seed(cp)
    await cp._messages_lock.acquire()
    delete = asyncio.create_task(cp.aclean_thread(_cfg()))
    await asyncio.sleep(0.05)  # the delete now waits on the messages lock
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_state(_cfg(user="bob"), _state("bob state"))
    cp._messages_lock.release()
    assert await delete == 1
    assert await cp.aget_thread_owner("t1") is None


async def test_reach_all_reads_and_deletes_but_never_writes(cp):
    await _seed(cp)
    admin = lambda: _cfg(user="root", reach="all")  # noqa: E731

    thread = await cp.aget_thread(admin())
    assert thread.user_id == "alice"  # the row owner, not the caller
    assert _texts(await cp.aget_state(admin())) == ["alice state"]
    assert len(await cp.alist_messages(admin())) == 2
    assert "t1" in {t.thread_id for t in await cp.alist_threads(admin())}

    with pytest.raises(ThreadNotFoundError):
        await cp.aput_state(admin(), _state("admin"))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_messages(admin(), [_msg("m3")])
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_thread(admin(), ThreadInfo(thread_id="t1", thread_name="admin"))
    assert await cp.aupdate_message_metadata(admin(), "m1", {"x": 1}) is None
    await _assert_untouched(cp)

    assert await cp.adelete_message(admin(), "m1") == 1
    assert await cp.aclear_state(admin()) == 1
    assert await cp.aclean_thread(admin()) == 1
    assert await cp.aget_thread(_cfg()) is None


async def test_reach_tenant_is_refused_without_tenant_columns(cp):
    await _seed(cp)
    with pytest.raises(ValueError, match="tenant columns"):
        await cp.aget_thread(_cfg(user="root", reach="tenant"))
    with pytest.raises(ValueError, match="tenant columns"):
        await cp.aclean_thread(_cfg(user="root", reach="tenant"))
    await _assert_untouched(cp)


async def test_thread_info_cannot_name_another_owner(cp):
    await cp.aput_thread(_cfg(), ThreadInfo(thread_id="t1", user_id="mallory", thread_name="x"))
    assert (await cp.aget_thread(_cfg())).user_id == "alice"
    assert await cp.aget_thread_owner("t1") == "alice"
    assert await cp.aget_thread(_cfg(user="mallory")) is None


async def test_delete_then_recreate_by_another_user(cp):
    await _seed(cp)
    assert await cp.aclean_thread(_cfg()) == 1

    await _seed(cp, user="bob")  # no tombstone: the id is free again
    await _assert_untouched(cp, user="bob")
    assert (await cp.aget_thread(_cfg(user="bob"))).user_id == "bob"
    assert await cp.aget_thread(_cfg()) is None
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_state(_cfg(), _state("alice again"))


@pytest.mark.parametrize("write", ["state", "thread", "messages"])
async def test_racing_creators_one_wins_never_a_merge(cp, write):
    async def create(thread, user):
        if write == "state":
            await cp.aput_state(_cfg(thread, user), _state(user))
        elif write == "thread":
            await cp.aput_thread(_cfg(thread, user), ThreadInfo(thread_id=thread, thread_name=user))
        else:
            await cp.aput_messages(_cfg(thread, user), [_msg(f"{user}-msg", user)])

    for i in range(10):
        thread = f"race-{i}"
        results = await asyncio.gather(
            create(thread, "alice"), create(thread, "bob"), return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert len(errors) == 1 and isinstance(errors[0], ThreadNotFoundError), results
        winner, loser = ("alice", "bob") if results[0] is None else ("bob", "alice")
        if write == "state":
            assert _texts(await cp.aget_state(_cfg(thread, winner))) == [winner]
            assert await cp.aget_state(_cfg(thread, loser)) is None
        elif write == "thread":
            got = await cp.aget_thread(_cfg(thread, winner))
            assert (got.thread_name, got.user_id) == (winner, winner)
            assert await cp.aget_thread(_cfg(thread, loser)) is None
        else:
            stored = await cp.alist_messages(_cfg(thread, winner))
            assert [str(m.message_id) for m in stored] == [f"{winner}-msg"]


async def test_sqlite_failed_write_leaves_nothing_behind(monkeypatch):
    """A message batch fails half-way: nothing is stored and the thread is not claimed."""
    cp = sqlite_checkpointer.SqliteCheckpointer(":memory:")
    real_dumps = sqlite_checkpointer._dumps
    calls = {"n": 0}

    def flaky_dumps(data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return real_dumps(data)

    monkeypatch.setattr(sqlite_checkpointer, "_dumps", flaky_dumps)
    with pytest.raises(RuntimeError, match="disk full"):
        await cp.aput_messages(_cfg(), [_msg("m1"), _msg("m2")])
    monkeypatch.setattr(sqlite_checkpointer, "_dumps", real_dumps)

    assert await cp.alist_messages(_cfg()) == []
    # The claim was rolled back with the batch, so the id is still free for anyone.
    await cp.aput_state(_cfg(user="bob"), _state("bob"))
    assert _texts(await cp.aget_state(_cfg(user="bob"))) == ["bob"]
    await cp.arelease()


# ---------------------------------------------------------------------------
# Postgres (mocked connection)
# ---------------------------------------------------------------------------

TENANTS = {"org_id": "text"}


def _pg(**kw) -> PgCheckpointer:
    cp = PgCheckpointer(postgres_dsn="postgresql://x/y", redis=MagicMock(), **kw)
    cp._invalidate_state_cache = AsyncMock()
    return cp


def _pg_cfg(thread="t1", user="alice", reach="own", org="A"):
    identity = {"user_id": user, "tenant": {"org_id": org}, "reach": reach}
    return {"thread_id": thread, "user_id": user, "user": identity}


class _Conn:
    """Records every call; answers fetchrow from ``rows`` (in order, then ``row``)."""

    def __init__(self, row=None, rows=(), status="OK"):
        self.calls: list[tuple[str, str, tuple]] = []
        self.row = row
        self.rows = list(rows)
        self.status = status

    async def execute(self, sql, *args):
        self.calls.append(("execute", " ".join(sql.split()), args))
        return self.status

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", " ".join(sql.split()), args))
        if "MAX(version)" in sql:
            return {"v": 0}
        return self.rows.pop(0) if self.rows else self.row

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", " ".join(sql.split()), args))
        return self.row.get("user_id") if isinstance(self.row, dict) else self.row

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", " ".join(sql.split()), args))
        return []

    @asynccontextmanager
    async def _tx(self):
        yield

    def transaction(self):
        return self._tx()

    def sql(self) -> list[str]:
        return [call[1] for call in self.calls]


def _wire(cp: PgCheckpointer, conn: _Conn) -> _Conn:
    pool = MagicMock()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    pool.acquire = acquire
    cp._pg_pool = pool
    cp._schema_initialized = True
    return conn


@pytest.mark.parametrize("reach", ["own", "tenant", "all"])
async def test_pg_state_write_to_foreign_thread_raises_unwrapped(reach):
    cp = _pg(tenant_columns=TENANTS)
    conn = _wire(cp, _Conn(row={"user_id": "bob", "org_id": "A"}))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_state(_pg_cfg(reach=reach), _state("hijack"))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_checkpoint(_pg_cfg(reach=reach), _state("hijack"), [_msg("m1")])
    assert not any("INSERT INTO" in s and '"states"' in s for s in conn.sql())
    assert not any('"messages"' in s for s in conn.sql())


@pytest.mark.parametrize("reach", ["own", "all"])
async def test_pg_message_and_ledger_writes_to_foreign_thread_raise(reach):
    cp = _pg()
    conn = _wire(cp, _Conn(row={"user_id": "bob"}))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_messages(_pg_cfg(reach=reach), [_msg("m1")])
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_tool_result(_pg_cfg(reach=reach), "call-1", {"ok": True})
    assert not any('"messages"' in s or '"tool_executions"' in s for s in conn.sql())


async def test_pg_other_tenant_raises_not_found():
    cp = _pg(tenant_columns=TENANTS)
    _wire(cp, _Conn(row={"user_id": "alice", "org_id": "B"}))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_state(_pg_cfg(reach="all"), _state("hijack"))


@pytest.mark.parametrize("reach", ["own", "all"])
async def test_pg_thread_update_stays_own(reach):
    cp = _pg()
    # Insert conflicts (None), the owner-scoped UPDATE matches nothing (None).
    conn = _wire(cp, _Conn(rows=[None, None]))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_thread(_pg_cfg(reach=reach), ThreadInfo(thread_id="t1", thread_name="x"))
    update = next(s for s in conn.sql() if s.startswith("UPDATE"))
    assert "AND user_id = $" in update


async def test_pg_thread_owner_is_the_caller_not_thread_info():
    cp = _pg()
    conn = _wire(cp, _Conn(rows=[{"thread_id": "t1"}]))
    created = await cp.aput_thread(
        _pg_cfg(), ThreadInfo(thread_id="t1", user_id="mallory", thread_name="x")
    )
    assert created is True
    insert = next(c for c in conn.calls if c[1].startswith("INSERT INTO"))
    assert insert[2][2] == "alice"


async def test_pg_aget_thread_reports_row_owner():
    cp = _pg(tenant_columns=TENANTS)
    row = {
        "thread_id": "t1",
        "thread_name": "T",
        "user_id": "bob",
        "created_at": None,
        "updated_at": None,
        "meta": None,
    }
    _wire(cp, _Conn(row=row))
    for reach in ("tenant", "all"):
        thread = await cp.aget_thread(_pg_cfg(reach=reach))
        assert thread.user_id == "bob"


async def test_pg_metadata_update_stays_own():
    cp = _pg()
    conn = _wire(cp, _Conn(row=None))
    assert await cp.aupdate_message_metadata(_pg_cfg(reach="all"), "m1", {"x": 1}) is None
    update = next(s for s in conn.sql() if s.startswith("UPDATE"))
    assert "WHERE user_id = $" in update


@pytest.mark.parametrize(("status", "count"), [("DELETE 0", 0), ("DELETE 3", 3)])
async def test_pg_clear_state_returns_rows(status, count):
    cp = _pg()
    _wire(cp, _Conn(status=status))
    assert await cp.aclear_state(_pg_cfg()) == count


@pytest.mark.parametrize(("status", "count"), [("DELETE 0", 0), ("DELETE 1", 1)])
async def test_pg_delete_message_returns_rows(status, count):
    cp = _pg()
    _wire(cp, _Conn(status=status))
    assert await cp.adelete_message(_pg_cfg(), "m1") == count


@pytest.mark.parametrize(("status", "count"), [("DELETE 0", 0), ("DELETE 1", 1)])
async def test_pg_clean_thread_returns_rows(status, count):
    cp = _pg()
    _wire(cp, _Conn(status=status))
    assert await cp.aclean_thread(_pg_cfg()) == count


@pytest.mark.parametrize(("row", "count"), [(None, 0), ({"user_id": "bob", "org_id": "A"}, 1)])
async def test_pg_clean_thread_with_tenant_columns_returns_rows(row, count):
    cp = _pg(tenant_columns=TENANTS)
    conn = _wire(cp, _Conn(row=row))
    assert await cp.aclean_thread(_pg_cfg(reach="tenant")) == count
    delete = next(s for s in conn.sql() if s.startswith("DELETE"))
    assert "user_id" not in delete.split("WHERE", 1)[1].split("RETURNING")[0]  # reach tenant
