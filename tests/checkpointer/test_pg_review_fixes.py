"""Sprint 7 review fix S1 in PgCheckpointer (fake pool, no database).

With tenant columns the state cache, stop and active-run keys are the thread's identity (its
owner and tenant values, read under the caller's scope), and a message delete that rewrites a
state moves the thread to a new state version. Real Postgres + Redis behaviour is in
tests/integration/test_pg_checkpointer_review_fixes_real.py.
"""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.core.authz import build_authz
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.columns import MissingColumnValueError


TENANTS = {"org_id": "text", "project_id": "text"}
THREAD_ROW = {"user_id": "owner", "org_id": "A", "project_id": "P1"}
IDENTITY_KEY = "state_cache:t1:owner:t:org_id=A:project_id=P1"


def _cp(**kw):
    kw.setdefault("tenant_columns", TENANTS)
    return PgCheckpointer(postgres_dsn="postgresql://x/y", redis=MagicMock(), **kw)


def _cfg(user="admin", org="A", project="P2", scope=("org_id",), run_id=None):
    identity = {"user_id": user, "org_id": org, "project_id": project}
    if scope is not None:
        identity["authz"] = build_authz(user, scope=list(scope))
    cfg = {"thread_id": "t1", "user_id": user, "user": identity}
    if run_id:
        cfg["run_id"] = run_id
    return cfg


class _Conn:
    def __init__(self, row=None, fetch_rows=()):
        self.row = row
        self.fetch_rows = list(fetch_rows)
        self.calls = []

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", " ".join(sql.split()), args))
        return self.row

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", " ".join(sql.split()), args))
        return self.fetch_rows

    async def execute(self, sql, *args):
        self.calls.append(("execute", " ".join(sql.split()), args))
        return "OK"

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", " ".join(sql.split()), args))

    @asynccontextmanager
    async def _tx(self):
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


class TestIdentityKeys:
    def test_keys_use_remembered_thread_identity(self):
        cp = _cp()
        cfg = _cfg()
        before = cp._get_thread_key("t1", "admin", cfg)
        assert before == "state_cache:t1:admin:t:org_id=A:project_id=P2"
        cp._remember_identity(cfg, "t1", THREAD_ROW)
        assert cp._get_thread_key("t1", "admin", cfg) == IDENTITY_KEY
        assert cp._stop_request_key(cfg) == "t1:owner:t:org_id=A:project_id=P1"

    def test_identity_of_another_thread_is_ignored(self):
        cp = _cp()
        cfg = _cfg()
        cp._remember_identity(cfg, "other", THREAD_ROW)
        assert cp._get_thread_key("t1", "admin", cfg).startswith("state_cache:t1:admin:")

    def test_no_tenant_columns_unchanged(self):
        cp = PgCheckpointer(postgres_dsn="postgresql://x/y", redis=MagicMock())
        cfg = {"thread_id": "t1", "user_id": "u1", "_thread_identity": {"thread_id": "t1"}}
        assert cp._get_thread_key("t1", "u1", cfg) == "state_cache:t1:u1"
        assert cp._stop_request_key(cfg) == "t1:u1"


@pytest.mark.asyncio
class TestCacheRead:
    async def test_scoped_identity_read_then_shared_key(self):
        cp = _cp()
        conn = _wire(cp, _Conn(row={**THREAD_ROW, "durable_version": 3}))
        cp.redis.get = AsyncMock(
            return_value=json.dumps({"context": [], "__checkpoint_version__": 3})
        )
        cfg = _cfg()
        assert await cp.aget_state_cache(cfg) is not None
        cp.redis.get.assert_awaited_once_with(IDENTITY_KEY)
        _method, sql, args = conn.calls[0]
        # The identity read carries the caller's scope (org only), not the thread id alone.
        assert sql.startswith('SELECT user_id, "org_id", "project_id", (SELECT COALESCE(MAX(')
        assert sql.endswith('WHERE thread_id = $1 AND "org_id" = $2')
        assert args == ("t1", "A")
        assert cfg["_checkpoint_version"] == 3

    async def test_version_mismatch_drops_the_shared_entry(self):
        cp = _cp()
        _wire(cp, _Conn(row={**THREAD_ROW, "durable_version": 4}))
        cp.redis.get = AsyncMock(
            return_value=json.dumps({"context": [], "__checkpoint_version__": 3})
        )
        cp.redis.delete = AsyncMock()
        cp.aget_state = AsyncMock(return_value=None)
        await cp.aget_state_cache(_cfg())
        deleted = cp.redis.delete.await_args.args
        assert IDENTITY_KEY in deleted
        cp.aget_state.assert_awaited_once()

    async def test_invisible_thread_reads_only_the_callers_own_key(self):
        cp = _cp()
        _wire(cp, _Conn(row=None))
        cp.redis.get = AsyncMock(return_value=None)
        cp.aget_state = AsyncMock(return_value=None)
        await cp.aget_state_cache(_cfg(org="B"))
        cp.redis.get.assert_awaited_once_with("state_cache:t1:admin:t:org_id=B:project_id=P2")

    async def test_unsatisfiable_scope_raises_before_any_read(self):
        cp = _cp()
        conn = _wire(cp, _Conn())
        cp.redis.get = AsyncMock()
        cfg = _cfg()
        cfg["user"]["org_id"] = None
        with pytest.raises(MissingColumnValueError):
            await cp.aget_state_cache(cfg)
        assert conn.calls == []
        cp.redis.get.assert_not_awaited()


@pytest.mark.asyncio
class TestDeleteMessage:
    async def test_delete_bumps_version_and_drops_shared_entry(self):
        cp = _cp()
        keep, gone = {"message_id": "keep"}, {"message_id": "gone"}
        conn = _wire(
            cp,
            _Conn(
                row=THREAD_ROW,
                fetch_rows=[
                    {"state_id": 7, "version": 5, "state_data": json.dumps({"context": [gone]})},
                    {
                        "state_id": 8,
                        "version": 6,
                        "state_data": json.dumps({"context": [keep, gone]}),
                    },
                ],
            ),
        )
        cp.redis.delete = AsyncMock()
        cfg = _cfg(user="owner", project="P2", scope=("user_id", "org_id"))
        await cp.adelete_message(cfg, "gone")
        lock = conn.calls[0]
        assert lock[1].startswith('SELECT user_id, "org_id", "project_id" FROM')
        assert lock[1].endswith("FOR UPDATE") and '"org_id" = $3' in lock[1]
        bumps = [c for c in conn.calls if c[1].startswith('UPDATE "public"."states" SET version')]
        assert bumps == [
            ("execute", 'UPDATE "public"."states" SET version = $1 WHERE state_id = $2', (7, 8))
        ]
        deleted = cp.redis.delete.await_args.args
        assert IDENTITY_KEY in deleted  # the thread's shared entry (project P1)
        assert "state_cache:t1:owner:t:org_id=A:project_id=P2" in deleted  # caller's own

    async def test_no_bump_when_no_state_holds_the_message(self):
        cp = _cp()
        conn = _wire(
            cp,
            _Conn(
                row=THREAD_ROW,
                fetch_rows=[{"state_id": 1, "version": 1, "state_data": json.dumps({})}],
            ),
        )
        cp.redis.delete = AsyncMock()
        await cp.adelete_message(_cfg(user="owner", scope=("user_id", "org_id")), "gone")
        assert not [c for c in conn.calls if "SET version" in c[1]]


@pytest.mark.asyncio
class TestStopKeys:
    async def test_stop_by_org_admin_targets_the_threads_key(self):
        cp = _cp()
        conn = _wire(cp, _Conn(row={**THREAD_ROW, "durable_version": 1}))
        cp.redis.get = AsyncMock(return_value=None)
        cp.redis.set = AsyncMock(return_value=True)
        cp.redis.setex = AsyncMock(return_value=True)
        cfg = _cfg()
        await cp.arequest_stop(cfg)
        written = [c.args[0] for c in (*cp.redis.set.await_args_list, *cp.redis.setex.await_args_list)]
        assert written and all("t1%3Aowner%3At%3Aorg_id%3DA" in key for key in written), written
        # One identity read per config, then the remembered identity is reused.
        await cp.ais_stop_requested(cfg)
        assert len([c for c in conn.calls if c[0] == "fetchrow"]) == 1

    async def test_identity_read_failure_falls_back_to_callers_key(self):
        cp = _cp()
        cp._read_thread_identity = AsyncMock(side_effect=ConnectionError("down"))
        cfg = _cfg()
        await cp._resolve_thread_identity(cfg)
        assert "_thread_identity" not in cfg
        assert cp._stop_request_key(cfg) == "t1:admin:t:org_id=A:project_id=P2"
