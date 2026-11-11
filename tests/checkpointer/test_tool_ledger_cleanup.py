"""aclear_tool_results / aprune_tool_results and the record_tool_result flag on each backend.

Postgres runs against mocks here; the real-database variants are in
tests/integration/test_pg_tool_ledger_cleanup_real.py.
"""

import logging
import sqlite3
import time
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from tests.checkpointer.test_tool_ledger import NoLedgerCheckpointer


A = {"thread_id": "t1", "user_id": "u1"}
B = {"thread_id": "t2", "user_id": "u1"}


def _sqlite(path=":memory:", **kwargs):
    cp = SqliteCheckpointer(path, **kwargs)
    cp.bind_state_type(AgentState)
    return cp


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def cp(request):
    c = InMemoryCheckpointer() if request.param == "memory" else _sqlite()
    yield c
    await c.arelease()


@pytest.mark.asyncio
async def test_clear_removes_only_that_threads_rows(cp):
    await cp.aput_tool_result(A, "m:c1", {"v": 1})
    await cp.aput_tool_result(A, "m:c2", {"v": 2})
    await cp.aput_tool_result(B, "m:c1", {"v": 3})

    assert await cp.aclear_tool_results(A) == 2
    assert await cp.aget_tool_result(A, "m:c1") is None
    assert await cp.aget_tool_result(B, "m:c1") == {"v": 3}
    assert await cp.aclear_tool_results(A) == 0


@pytest.mark.asyncio
async def test_clear_by_origin_ids_removes_only_those_messages_rows(cp):
    await cp.aput_tool_result(A, "m1:c1", {"v": 1})
    await cp.aput_tool_result(A, "m1:c2", {"v": 2})
    await cp.aput_tool_result(A, "m2:c1", {"v": 3})  # a newer run's message
    await cp.aput_tool_result(A, "m10:c1", {"v": 4})  # "m1" is a prefix of it, not its id
    await cp.aput_tool_result(B, "m1:c1", {"v": 5})

    assert await cp.aclear_tool_results(A, ["m1"]) == 2
    assert await cp.aget_tool_result(A, "m1:c1") is None
    assert await cp.aget_tool_result(A, "m1:c2") is None
    assert await cp.aget_tool_result(A, "m2:c1") == {"v": 3}
    assert await cp.aget_tool_result(A, "m10:c1") == {"v": 4}
    assert await cp.aget_tool_result(B, "m1:c1") == {"v": 5}


@pytest.mark.asyncio
async def test_clear_with_an_empty_list_removes_nothing(cp):
    await cp.aput_tool_result(A, "m1:c1", {"v": 1})
    assert await cp.aclear_tool_results(A, []) == 0
    assert await cp.aget_tool_result(A, "m1:c1") == {"v": 1}


@pytest.mark.asyncio
async def test_clear_matches_wildcard_and_colon_ids_literally(cp):
    await cp.aput_tool_result(A, "a%:c1", {"v": 1})
    await cp.aput_tool_result(A, "ab:c1", {"v": 2})  # would match LIKE 'a_:%' or 'a%:%'
    await cp.aput_tool_result(A, "a_:c1", {"v": 3})
    await cp.aput_tool_result(A, "x:y:c1", {"v": 4})
    await cp.aput_tool_result(A, "x:c1", {"v": 5})

    assert await cp.aclear_tool_results(A, ["a%", "a_", "x:y"]) == 3
    assert await cp.aget_tool_result(A, "ab:c1") == {"v": 2}
    assert await cp.aget_tool_result(A, "x:c1") == {"v": 5}
    assert await cp.aget_tool_result(A, "a%:c1") is None
    assert await cp.aget_tool_result(A, "a_:c1") is None
    assert await cp.aget_tool_result(A, "x:y:c1") is None


@pytest.mark.asyncio
async def test_clear_of_another_users_thread_removes_nothing(cp):
    await cp.aput_tool_result(A, "m:c1", {"v": 1})
    intruder = {"thread_id": "t1", "user_id": "u2", "user": {"user_id": "u2", "reach": "own"}}
    assert await cp.aclear_tool_results(intruder) == 0
    assert await cp.aget_tool_result(A, "m:c1") == {"v": 1}


@pytest.mark.asyncio
async def test_prune_removes_old_rows_of_every_thread(cp, monkeypatch):
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() - 7200)
    await cp.aput_tool_result(A, "m:old", {"v": 1})
    await cp.aput_tool_result(B, "m:old", {"v": 2})
    monkeypatch.setattr(time, "time", real)
    await cp.aput_tool_result(A, "m:new", {"v": 3})

    assert await cp.aprune_tool_results(timedelta(hours=1)) == 2
    assert await cp.aget_tool_result(A, "m:old") is None
    assert await cp.aget_tool_result(B, "m:old") is None
    assert await cp.aget_tool_result(A, "m:new") == {"v": 3}


def test_sync_wrappers():
    cp = InMemoryCheckpointer()
    cp.put_state(A, AgentState())  # any sync call works outside a loop
    assert cp.clear_tool_results(A) == 0
    assert cp.clear_tool_results(A, ["m"]) == 0
    assert cp.prune_tool_results(timedelta(seconds=1)) == 0


@pytest.mark.asyncio
async def test_base_defaults_do_nothing_and_do_not_warn(caplog):
    cp = NoLedgerCheckpointer()
    with caplog.at_level(logging.WARNING):
        assert await cp.aclear_tool_results(A) == 0
        assert await cp.aprune_tool_results(timedelta(days=1)) == 0
    assert caplog.records == []
    assert cp.record_tool_result is True


def test_record_tool_result_flag_on_each_backend(monkeypatch):
    assert InMemoryCheckpointer().record_tool_result is True
    assert InMemoryCheckpointer(record_tool_result=False).record_tool_result is False
    assert _sqlite(record_tool_result=False).record_tool_result is False
    pg, _ = _pg(monkeypatch, record_tool_result=False)
    assert pg.record_tool_result is False
    assert _pg(monkeypatch)[0].record_tool_result is True


# ------------------------------------------------------------------ SQLite schema upgrade


@pytest.mark.asyncio
async def test_sqlite_adds_created_at_to_an_old_ledger_table(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as raw:
        raw.execute(
            "CREATE TABLE af_tool_executions (thread_id TEXT NOT NULL, tool_call_id TEXT NOT "
            "NULL, result TEXT NOT NULL, PRIMARY KEY (thread_id, tool_call_id))"
        )
        raw.execute("INSERT INTO af_tool_executions VALUES ('t1', 'm:c1', '{\"v\": 1}')")

    cp = _sqlite(path)
    await cp.asetup()
    # The old row is kept and stamped with the upgrade time: a prune does not drop it yet.
    assert await cp.aget_tool_result(A, "m:c1") == {"v": 1}
    assert await cp.aprune_tool_results(timedelta(minutes=5)) == 0
    assert await cp.aprune_tool_results(timedelta(seconds=-1)) == 1
    await cp.asetup()  # a second setup finds the column and changes nothing
    await cp.arelease()

    with sqlite3.connect(path) as raw:
        indexes = {r[1] for r in raw.execute("PRAGMA index_list(af_tool_executions)")}
    assert "idx_af_tool_executions_created" in indexes


# ------------------------------------------------------------------ Postgres (mocked)


class _Txn:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *a):
        return False


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


def _pg(monkeypatch, **kwargs):
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    cp = PgCheckpointer(postgres_dsn="postgres://x", redis=AsyncMock(), **kwargs)
    conn = AsyncMock()
    conn.transaction = MagicMock(return_value=_Txn())
    conn.execute = AsyncMock(return_value="DELETE 3")
    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire = MagicMock(return_value=_Acquire(conn))
    cp._schema_initialized = True
    return cp, conn


@pytest.mark.asyncio
async def test_pg_clear_is_scoped_to_the_callers_own_thread(monkeypatch):
    cp, conn = _pg(monkeypatch)
    assert await cp.aclear_tool_results(A) == 3
    sql, *params = conn.execute.await_args.args
    assert sql.startswith('DELETE FROM "public"."tool_executions" WHERE thread_id = $1')
    assert 'SELECT thread_id FROM "public"."threads" WHERE user_id = $2' in sql
    assert params == ["t1", "u1"]


@pytest.mark.asyncio
async def test_pg_clear_by_origin_ids_binds_them_as_a_text_array(monkeypatch):
    cp, conn = _pg(monkeypatch)
    assert await cp.aclear_tool_results(A, ["m1", "a%_"]) == 3
    sql, *params = conn.execute.await_args.args
    assert (
        "WHERE thread_id = $1 AND EXISTS (SELECT 1 FROM unnest($2::text[]) AS o(id) "
        "WHERE left(tool_call_id, length(o.id) + 1) = o.id || ':')"
    ) in sql
    assert "LIKE" not in sql
    assert "WHERE user_id = $3" in sql
    assert params == ["t1", ["m1", "a%_"], "u1"]


@pytest.mark.asyncio
async def test_pg_clear_with_an_empty_list_sends_no_query(monkeypatch):
    cp, conn = _pg(monkeypatch)
    assert await cp.aclear_tool_results(A, []) == 0
    conn.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_pg_clear_with_tenant_columns_and_rls_sets_the_tenant(monkeypatch):
    cp, conn = _pg(monkeypatch, tenant_columns=["org_id"], row_level_security=True)
    identity = {"user_id": "u1", "tenant": {"org_id": "o1"}}
    cfg = {"thread_id": "t1", "user_id": "u1", "user": identity}
    await cp.aclear_tool_results(cfg)
    calls = [c.args for c in conn.execute.await_args_list]
    delete = next(c for c in calls if c[0].startswith("DELETE"))
    assert '"org_id" = $3' in delete[0]
    assert list(delete[1:]) == ["t1", "u1", "o1"]
    assert any("set_config" in c[0] for c in calls)  # RLS values set in the transaction


@pytest.mark.asyncio
async def test_pg_prune_deletes_by_age_across_threads(monkeypatch):
    cp, conn = _pg(monkeypatch)
    assert await cp.aprune_tool_results(timedelta(days=7)) == 3
    sql, older_than = conn.execute.await_args.args
    assert "WHERE created_at < now() - $1::interval" in sql
    assert "user_id" not in sql
    assert older_than == timedelta(days=7)


@pytest.mark.asyncio
@pytest.mark.parametrize(("bypass", "warned"), [(False, True), (True, False)])
async def test_pg_prune_warns_when_rls_hides_the_rows(monkeypatch, caplog, bypass, warned):
    cp, conn = _pg(monkeypatch, tenant_columns=["org_id"], row_level_security=True)
    conn.fetchval = AsyncMock(return_value=bypass)
    with caplog.at_level(logging.WARNING, logger="tenxgraph.checkpointer.pg"):
        await cp.aprune_tool_results(timedelta(days=1))
    assert any("BYPASSRLS" in r.getMessage() for r in caplog.records) is warned
    assert "rolbypassrls" in conn.fetchval.await_args.args[0]


def test_pg_v7_index_is_managed_by_ensure_indexes(monkeypatch):
    cp, _ = _pg(monkeypatch)
    assert "tool_executions" in PgCheckpointer._index_tables()
    assert ("idx_tool_executions_created_at", "tool_executions", "(created_at)") in (
        PgCheckpointer._base_index_specs()
    )
    # tool_executions is created by migration v3, so the fresh-table script leaves it out.
    assert not any("tool_executions" in s for s in cp._build_create_tables_sql())


@pytest.mark.asyncio
async def test_pg_upgrade_builds_the_ledger_index_concurrently(monkeypatch):
    cp, conn = _pg(monkeypatch)
    conn.fetch = AsyncMock(return_value=[])  # empty index catalog
    await cp._ensure_indexes(conn, {"tool_executions"}, concurrently=True)
    sql = [c.args[0] for c in conn.execute.await_args_list]
    assert any(
        'CREATE INDEX CONCURRENTLY "idx_tool_executions_created_at"' in s
        and '"tool_executions"' in s
        for s in sql
    )
