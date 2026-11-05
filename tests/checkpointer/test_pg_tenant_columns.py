"""PgCheckpointer tenant/extra columns and RLS: SQL shape, cache keys, options (no database).

The SQL runs against a real database in tests/integration/test_pg_checkpointer_tenant_real.py.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.core.authz import build_authz
from tenxgraph.core.exceptions import StorageError
from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.columns import MissingColumnValueError


TENANTS = {"org_id": "text", "project_id": "text"}


def _cp(**kw):
    return PgCheckpointer(postgres_dsn="postgresql://x/y", redis=MagicMock(), **kw)


def _cfg(thread="t1", user="u1", org="A", project="P1", scope=None, **extra):
    identity = {"user_id": user, "org_id": org, "project_id": project, **extra}
    if scope is not None:
        identity["authz"] = build_authz(user, scope=scope)
    return {"thread_id": thread, "user_id": user, "user": identity}


class _Conn:
    def __init__(self, row=None):
        self.calls = []
        self.row = row
        self.transactions = 0

    async def execute(self, sql, *args):
        self.calls.append(("execute", " ".join(sql.split()), args))
        return "OK"

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", " ".join(sql.split()), args))
        if "MAX(version)" in sql:
            return {"v": 0}
        return self.row

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", " ".join(sql.split()), args))
        return self.row.get("user_id") if isinstance(self.row, dict) else self.row

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", " ".join(sql.split()), args))
        return []

    @asynccontextmanager
    async def _tx(self):
        self.transactions += 1
        yield

    def transaction(self):
        return self._tx()


def _wire(cp, conn):
    pool = MagicMock()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    pool.acquire = acquire
    cp._pg_pool = pool
    cp._schema_initialized = True
    return conn


class TestOptions:
    def test_columns_and_names(self):
        cp = _cp(tenant_columns=TENANTS, extra_columns={"public_id": "text"})
        assert cp.tenant_column_names == ("org_id", "project_id")
        assert cp.columns.extra_names == ("public_id",)

    @pytest.mark.parametrize("name", ["thread_id", "thread_name", "meta", "user_id", "created_at"])
    def test_builtin_names_rejected(self, name):
        with pytest.raises(ValueError, match="built-in column"):
            _cp(tenant_columns={name: "text"})

    def test_rls_validation(self):
        with pytest.raises(ValueError, match="needs at least one tenant column"):
            _cp(row_level_security=True)
        with pytest.raises(ValueError, match="True or False"):
            _cp(tenant_columns=TENANTS, row_level_security=1)

    def test_migration_v6(self):
        cp = _cp(tenant_columns={"org_id": "uuid"}, extra_columns={"plan": "text"})
        # Columns only: their indexes are built by _ensure_indexes (concurrently on a table
        # in use), never inside the migration transaction.
        assert cp._build_migrations()[6] == [
            'ALTER TABLE "public"."threads" ADD COLUMN IF NOT EXISTS "org_id" UUID',
            'ALTER TABLE "public"."threads" ADD COLUMN IF NOT EXISTS "plan" TEXT',
        ]
        assert _cp()._build_migrations()[6] == []


class TestScopeSql:
    def test_nothing_declared_is_unchanged(self):
        cp = _cp()
        cfg = {"thread_id": "t1", "user_id": "u1"}
        assert cp._thread_row_scope("t1", "u1", cfg) == (
            "thread_id = $1 AND user_id = $2",
            ["t1", "u1"],
        )
        sql, params = cp._thread_scope("t1", "u1", cfg)
        assert sql == (
            'thread_id = $1 AND thread_id IN (SELECT thread_id FROM "public"."threads" '
            "WHERE user_id = $2)"
        )
        assert params == ["t1", "u1"]
        params2 = ["m1", "t1"]
        query = cp._append_owner_scope("SELECT 1 WHERE x", params2, "u1", cfg)
        assert query.endswith(f"AND {cp._thread_owned_by_sql(3)}")

    def test_tenant_columns_join_every_scope(self):
        cp = _cp(tenant_columns=TENANTS)
        cfg = _cfg()
        assert cp._thread_row_scope("t1", "u1", cfg) == (
            'thread_id = $1 AND user_id = $2 AND "org_id" = $3 AND "project_id" = $4',
            ["t1", "u1", "A", "P1"],
        )
        sql, params = cp._thread_scope("t1", "u1", cfg)
        assert sql.endswith('WHERE user_id = $2 AND "org_id" = $3 AND "project_id" = $4)')
        assert params == ["t1", "u1", "A", "P1"]
        params = ["m1", "t1"]
        query = cp._append_owner_scope("SELECT 1 WHERE x", params, "u1", cfg)
        assert query.endswith('WHERE user_id = $3 AND "org_id" = $4 AND "project_id" = $5)')
        assert params == ["m1", "t1", "u1", "A", "P1"]

    def test_scope_without_user_id(self):
        cp = _cp(tenant_columns=TENANTS)
        sql, params = cp._thread_row_scope("t1", "u1", _cfg(scope=["org_id"]))
        assert sql == 'thread_id = $1 AND "org_id" = $2'
        assert params == ["t1", "A"]

    def test_tenant_columns_apply_even_when_user_isolation_is_off(self):
        cp = _cp(tenant_columns=TENANTS, enforce_user_isolation=False)
        sql, _ = cp._thread_row_scope("t1", "u1", _cfg())
        assert sql == 'thread_id = $1 AND "org_id" = $2 AND "project_id" = $3'

    def test_missing_picked_value_raises(self):
        cp = _cp(tenant_columns=TENANTS)
        cfg = {"thread_id": "t1", "user_id": "u1", "user": {"org_id": "A"}}
        with pytest.raises(MissingColumnValueError, match="project_id"):
            cp._thread_scope("t1", "u1", cfg)

    def test_insert_carries_values(self):
        cp = _cp(tenant_columns=TENANTS, extra_columns={"public_id": "text"})
        isolation = cp._isolation(_cfg(public_id="pub"), write=True)
        sql, params = cp._thread_insert_sql(
            "t1", "n", "u1", "{}", isolation, "ON CONFLICT DO NOTHING"
        )
        assert sql == (
            'INSERT INTO "public"."threads" (thread_id, thread_name, user_id, meta, "org_id", '
            '"project_id", "public_id") VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT DO NOTHING'
        )
        assert params == ["t1", "n", "u1", "{}", "A", "P1", "pub"]


class TestCacheKeys:
    def test_unchanged_without_tenant_columns(self):
        cp = _cp()
        assert cp._get_thread_key("t1", "u1", _cfg()) == "state_cache:t1:u1"

    def test_tenant_values_in_the_key(self):
        cp = _cp(tenant_columns=TENANTS)
        a = cp._get_thread_key("t1", "u1", _cfg(org="A"))
        b = cp._get_thread_key("t1", "u1", _cfg(org="B"))
        assert a == "state_cache:t1:u1:t:org_id=A:project_id=P1"
        assert a != b
        missing = cp._get_thread_key("t1", "u1", {"user": {"project_id": "P1"}})
        assert missing.endswith(":t:org_id!:project_id=P1")
        # A value cannot forge another column's part of the key.
        tricky = cp._get_thread_key("t1", "u1", _cfg(org="A:project_id=P1"))
        assert tricky != a

    def test_stop_key_is_tenant_scoped(self):
        cp = _cp(tenant_columns=TENANTS)
        assert cp._stop_request_key(_cfg(org="A")) != cp._stop_request_key(_cfg(org="B"))
        assert _cp()._stop_request_key(_cfg()) == "t1:u1"

    @pytest.mark.asyncio
    async def test_cache_read_uses_tenant_key(self):
        cp = _cp(tenant_columns=TENANTS)
        _wire(cp, _Conn(row=None))  # the thread is not visible yet: the caller's own key
        cp.redis.get = AsyncMock(return_value=None)
        cp.aget_state = AsyncMock(return_value=None)
        await cp.aget_state_cache(_cfg(org="B"))
        cp.redis.get.assert_awaited_once_with("state_cache:t1:u1:t:org_id=B:project_id=P1")


@pytest.mark.asyncio
class TestWritesAndRls:
    async def test_thread_of_other_tenant_is_forbidden(self):
        cp = _cp(tenant_columns=TENANTS)
        conn = _wire(cp, _Conn(row={"user_id": "u1", "org_id": "B", "project_id": "P1"}))
        with pytest.raises(StorageError) as err:
            await cp._ensure_thread_exists("t1", "u1", _cfg(org="A"))
        assert err.value.error_code == "STORAGE_FORBIDDEN_001"
        assert conn.calls[0][1].startswith('INSERT INTO "public"."threads"')
        assert conn.calls[1][1] == (
            'SELECT user_id, "org_id", "project_id" FROM "public"."threads" WHERE thread_id = $1'
        )

    async def test_lock_reads_tenant_values_for_update(self):
        cp = _cp(tenant_columns=TENANTS)
        conn = _wire(cp, _Conn(row={"user_id": "u1", "org_id": "A", "project_id": "P1"}))
        await cp._lock_thread_for_write(conn, "t1", "u1", _cfg())
        assert conn.calls[1][1].endswith("WHERE thread_id = $1 FOR UPDATE")

    async def test_old_thread_with_null_tenant_is_forbidden_on_write(self):
        cp = _cp(tenant_columns=TENANTS)
        conn = _wire(cp, _Conn(row={"user_id": "u1", "org_id": None, "project_id": None}))
        with pytest.raises(StorageError, match="different tenant"):
            await cp._lock_thread_for_write(conn, "t1", "u1", _cfg())

    async def test_missing_required_value_raises_before_storage_error_wrapping(self):
        cp = _cp(tenant_columns=TENANTS)
        _wire(cp, _Conn())
        cfg = {"thread_id": "t1", "user_id": "u1", "user": {"user_id": "u1"}}
        with pytest.raises(MissingColumnValueError):
            await cp.aput_state(cfg, AgentState())
        with pytest.raises(MissingColumnValueError):
            await cp.aput_checkpoint(cfg, AgentState())

    async def test_rls_sets_values_in_a_transaction(self):
        cp = _cp(tenant_columns=TENANTS, row_level_security=True)
        conn = _wire(cp, _Conn())
        await cp.aget_state(_cfg(scope=["user_id", "project_id"]))
        assert conn.transactions == 1
        _method, sql, args = conn.calls[0]
        assert sql.startswith("SELECT set_config($1, $2, true)")
        assert args == (
            "tenxgraph.scope",
            "+,project_id",
            "tenxgraph.org_id",
            "",
            "tenxgraph.project_id",
            "P1",
        )

    async def test_rls_global_lookups_see_every_tenant(self):
        cp = _cp(tenant_columns=TENANTS, row_level_security=True)
        conn = _wire(cp, _Conn(row={"user_id": "u1", "org_id": "A", "project_id": "P1"}))
        assert await cp.aget_thread_scope("t1") == {
            "user_id": "u1",
            "org_id": "A",
            "project_id": "P1",
        }
        assert conn.calls[0][2][:2] == ("tenxgraph.scope", "+")
        await cp.aget_thread_owner("t1")
        assert conn.calls[2][2][:2] == ("tenxgraph.scope", "+")

    async def test_rls_invisible_thread_is_forbidden(self):
        cp = _cp(tenant_columns=TENANTS, row_level_security=True)
        _wire(cp, _Conn(row=None))
        with pytest.raises(StorageError) as err:
            await cp._ensure_thread_exists("t1", "u1", _cfg())
        assert err.value.error_code == "STORAGE_FORBIDDEN_001"

    async def test_setup_manages_columns_and_policies(self):
        cp = _cp(tenant_columns={"org_id": "uuid"}, row_level_security=True)
        conn = _Conn(row=None)
        await cp._ensure_declared_columns(conn)
        executed = [c[1] for c in conn.calls if c[0] == "execute"]
        assert 'ALTER TABLE "public"."threads" ADD COLUMN IF NOT EXISTS "org_id" UUID' in executed
        policies = [s for s in executed if s.startswith("CREATE POLICY")]
        assert len(policies) == 4  # threads + states, messages, tool_executions
        assert "tenxgraph.org_id" in policies[0]
        assert 'EXISTS (SELECT 1 FROM "public"."threads" AS t' in policies[1]
        assert sum("FORCE ROW LEVEL SECURITY" in s for s in executed) == 4

    async def test_setup_with_nothing_declared_does_nothing(self):
        conn = _Conn()
        await _cp()._ensure_declared_columns(conn)
        assert conn.calls == []
