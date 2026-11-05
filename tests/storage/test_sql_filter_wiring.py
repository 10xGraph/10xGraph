"""Unit tests for how PgVectorStore and the checkpointers use ``sql_filter`` (no database)."""

import pytest

from tenxgraph.core.authz import SCOPE_NONE, build_authz
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from tenxgraph.storage.sql_filter import MAX_CONDITIONS, FilterError, validate_filters
from tests.store.test_pgvector_store import CFG, _store


TENANT = {"org_id": "text"}
EXTRA = {"plan": "text", "seats": "integer"}
USER = {"user_id": "u1", "org_id": "A", "plan": "pro", "seats": 3}
TCFG = {"user_id": "u1", "thread_id": "t1", "user": USER}


def _tstore(**kw):
    return _store(tenant_columns=TENANT, extra_columns=EXTRA, **kw)


# ---------------------------------------------------------------------------
# PgVectorStore
# ---------------------------------------------------------------------------


class TestStoreFilters:
    @pytest.mark.asyncio
    async def test_schema_lists_builtins_and_extras_only(self):
        store, _, _ = _tstore()
        assert set(store.filter_schema.names) == {
            "memory_type",
            "category",
            "thread_id",
            "memory_key",
            "created_at",
            "updated_at",
            "plan",
            "seats",
        }
        assert store.filter_schema.is_isolation("org_id")
        assert validate_filters({"seats": {"gte": 2}}, store.filter_schema)

    @pytest.mark.parametrize("strategy", ["similarity", "text", "hybrid"])
    @pytest.mark.asyncio
    async def test_operators_after_isolation_in_every_strategy(self, strategy):
        store, conn, _ = _tstore()
        await store.asearch(
            TCFG,
            "q",
            retrieval_strategy=strategy,
            filters={"seats": {"gte": 2, "lt": 10}, "plan": {"in": ["pro"]}, "tag": "x"},
        )
        _, sql, args = conn.calls[-1]
        # Isolation first (collection, user, tenant), then the filters, then the metadata key.
        # Search SQL qualifies every column with the table alias ``m``.
        assert 'm."collection" = $1 AND m."user_id" = $2 AND m."org_id" = $3' in sql
        assert 'm."seats" >= $4 AND m."seats" < $5 AND m."plan" = ANY($6)' in sql
        assert 'm."metadata" @> $7::text::jsonb' in sql
        assert args[:7] == ("agentflow_memories", "u1", "A", 2, 10, ["pro"], '{"tag": "x"}')
        if strategy == "hybrid":
            # Both candidate lists carry the full scope including the filters.
            assert sql.count('m."seats" >= $4') == 2

    @pytest.mark.asyncio
    async def test_get_all_filters(self):
        store, conn, _ = _tstore()
        await store.aget_all(TCFG, limit=5, filters={"created_at": {"is_null": False}})
        _, sql, args = conn.calls[-1]
        assert '"org_id" = $3 AND "created_at" IS NOT NULL ORDER BY' in sql
        assert args == ("agentflow_memories", "u1", "A", 5)

    @pytest.mark.asyncio
    async def test_forget_filters(self):
        store, conn, _ = _tstore()
        await store.aforget_memory(TCFG, filters={"plan": {"ne": "free"}})
        _, sql, args = conn.calls[-1]
        assert sql.startswith("DELETE FROM")
        assert '"thread_id" = $4 AND "plan" IS DISTINCT FROM $5' in sql
        assert args == ("agentflow_memories", "u1", "A", "t1", "free")

    @pytest.mark.asyncio
    async def test_plain_values_on_builtin_columns_keep_equality(self):
        store, conn, _ = _store()
        await store.aget_all(CFG, filters={"memory_key": "k", "thread_id": 7})
        _, sql, args = conn.calls[-1]
        assert '"memory_key" = $3 AND "thread_id" = $4' in sql
        assert args[2:4] == ("k", "7")

    @pytest.mark.asyncio
    async def test_extra_column_plain_value_is_column_not_metadata(self):
        store, conn, _ = _tstore()
        await store.aget_all(TCFG, filters={"plan": "pro"})
        _, sql, _args = conn.calls[-1]
        assert '"plan" = $4' in sql and "metadata" not in sql.split("WHERE", 1)[1]

    @pytest.mark.parametrize("key", ["org_id", "user_id"])
    @pytest.mark.asyncio
    async def test_isolation_columns_rejected(self, key):
        store, conn, _ = _tstore()
        with pytest.raises(FilterError, match="isolation column"):
            await store.asearch(TCFG, "q", filters={key: "B"})
        with pytest.raises(FilterError, match="isolation column"):
            await store.aforget_memory(TCFG, filters={key: {"ne": "A"}})
        assert not [c for c in conn.calls if c[0] in ("fetch", "execute")]

    @pytest.mark.asyncio
    async def test_operator_on_metadata_key_rejected(self):
        store, _, _ = _store()
        with pytest.raises(FilterError, match="Unsupported filter"):
            await store.aget_all(CFG, filters={"tag": {"in": ["a"]}})

    @pytest.mark.asyncio
    async def test_bad_operator_on_column(self):
        store, _, _ = _tstore()
        with pytest.raises(FilterError, match="not supported"):
            await store.asearch(TCFG, "q", filters={"plan": {"gt": "a"}})

    @pytest.mark.asyncio
    async def test_limit_counts_metadata_keys(self):
        store, _, _ = _store()
        filters = {f"k{i}": i for i in range(MAX_CONDITIONS)}
        await store.aget_all(CFG, filters=filters)
        filters["memory_type"] = "semantic"
        with pytest.raises(FilterError, match="Too many"):
            await store.aget_all(CFG, filters=filters)
        with pytest.raises(FilterError, match="Too many"):
            await store.aget_all(
                CFG,
                filters={
                    "tag": "x",
                    "created_at": dict.fromkeys(
                        ("gt", "gte", "lt", "lte", "ne"), "2026-01-01T00:00:00+00:00"
                    ),
                    "updated_at": dict.fromkeys(
                        ("gt", "gte", "lt", "lte", "ne"), "2026-01-01T00:00:00+00:00"
                    ),
                },
            )

    @pytest.mark.asyncio
    async def test_filters_must_be_object(self):
        store, _, _ = _store()
        with pytest.raises(FilterError, match="must be an object"):
            await store.aget_all(CFG, filters=[("a", 1)])

    @pytest.mark.asyncio
    async def test_injection_in_values_never_reaches_sql(self):
        store, conn, _ = _tstore()
        payload = "x'; DROP TABLE t; --"
        await store.asearch(
            TCFG, "q", filters={"plan": {"like": payload}, "category": payload, payload: payload}
        )
        _, sql, args = conn.calls[-1]
        assert "DROP" not in sql
        assert payload in args


# ---------------------------------------------------------------------------
# PgCheckpointer.alist_threads
# ---------------------------------------------------------------------------


class _Conn:
    def __init__(self):
        self.calls = []

    async def fetch(self, sql, *args):
        self.calls.append((" ".join(sql.split()), args))
        return []


def _cp(monkeypatch, **kw):
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    cp = PgCheckpointer(postgres_dsn="postgres://x", redis=object(), **kw)
    conn = _Conn()

    async def run_scoped(fn, *, settings=None, in_transaction=False):
        return await fn(conn)

    monkeypatch.setattr(cp, "_run_scoped", run_scoped)
    return cp, conn


class TestPgCheckpointerFilters:
    @pytest.mark.asyncio
    async def test_filters_after_isolation_and_search(self, monkeypatch):
        cp, conn = _cp(monkeypatch, tenant_columns=TENANT, extra_columns=EXTRA)
        await cp.alist_threads(
            TCFG,
            search="hello",
            offset=2,
            limit=5,
            filters={"plan": "pro", "created_at": {"gte": "2026-01-01T00:00:00+00:00"}},
        )
        sql, args = conn.calls[-1]
        assert (
            'WHERE TRUE AND user_id = $1 AND "org_id" = $2 AND thread_name ILIKE $3 '
            'AND "plan" = $4 AND "created_at" >= $5 ORDER BY updated_at DESC LIMIT $6 OFFSET $7'
        ) in sql
        assert args[:4] == ("u1", "A", "%hello%", "pro")
        assert args[4].isoformat() == "2026-01-01T00:00:00+00:00"
        assert args[5:] == (5, 2)

    @pytest.mark.asyncio
    async def test_filters_without_isolation(self, monkeypatch):
        cp, conn = _cp(monkeypatch, enforce_user_isolation=False)
        await cp.alist_threads({}, filters={"thread_name": {"ilike": "%a%"}})
        sql, args = conn.calls[-1]
        assert 'WHERE TRUE AND "thread_name" ILIKE $1 ORDER BY' in sql
        assert args == ("%a%",)

    @pytest.mark.asyncio
    async def test_scope_none_still_only_narrows(self, monkeypatch):
        cp, conn = _cp(monkeypatch, tenant_columns=TENANT, extra_columns=EXTRA)
        cfg = {"user_id": "admin", "user": {"authz": build_authz("admin", scope=SCOPE_NONE)}}
        await cp.alist_threads(cfg, filters={"seats": {"in": [1, 2]}})
        sql, args = conn.calls[-1]
        assert 'WHERE TRUE AND "seats" = ANY($1) ORDER BY' in sql
        assert args[0] == [1, 2]

    @pytest.mark.parametrize("key", ["org_id", "user_id", "meta", "thread_id"])
    @pytest.mark.asyncio
    async def test_rejected_columns_raise_before_query(self, monkeypatch, key):
        cp, conn = _cp(monkeypatch, tenant_columns=TENANT, extra_columns=EXTRA)
        with pytest.raises(FilterError):
            await cp.alist_threads(TCFG, filters={key: "x"})
        assert conn.calls == []

    @pytest.mark.asyncio
    async def test_no_filters_is_unchanged(self, monkeypatch):
        cp, conn = _cp(monkeypatch)
        await cp.alist_threads({"user_id": "u1"})
        sql, args = conn.calls[-1]
        assert sql.endswith("WHERE TRUE AND user_id = $1 ORDER BY updated_at DESC")
        assert args[0] == "u1"

    def test_sync_wrapper_passes_filters(self, monkeypatch):
        cp, conn = _cp(monkeypatch, extra_columns=EXTRA)
        cp.list_threads({"user_id": "u1"}, filters={"plan": "pro"})
        assert '"plan" = $2' in conn.calls[-1][0]


# ---------------------------------------------------------------------------
# Backends without declared columns
# ---------------------------------------------------------------------------


class TestNoColumnBackends:
    @pytest.mark.asyncio
    async def test_in_memory_rejects_filters(self):
        cp = InMemoryCheckpointer()
        with pytest.raises(FilterError, match="InMemoryCheckpointer does not support"):
            await cp.alist_threads({"user_id": "u1"}, filters={"plan": "pro"})
        assert await cp.alist_threads({"user_id": "u1"}, filters={}) == []
        assert await cp.alist_threads({"user_id": "u1"}, filters=None) == []

    def test_in_memory_sync_rejects_filters(self):
        with pytest.raises(FilterError):
            InMemoryCheckpointer().list_threads({"user_id": "u1"}, filters={"plan": "pro"})

    @pytest.mark.asyncio
    async def test_sqlite_rejects_filters(self, tmp_path):
        cp = SqliteCheckpointer(db_path=str(tmp_path / "x.db"))
        try:
            with pytest.raises(FilterError, match="SqliteCheckpointer does not support"):
                await cp.alist_threads({"user_id": "u1"}, filters={"plan": "pro"})
            assert await cp.alist_threads({"user_id": "u1"}) == []
        finally:
            await cp.arelease()

    def test_base_sync_wrapper_keeps_old_overrides_working(self):
        class Legacy(BaseCheckpointer):
            pass

        seen = {}

        async def alist_threads(self, config, search=None, offset=None, limit=None):
            seen["args"] = (config, search, offset, limit)
            return []

        for name in list(getattr(BaseCheckpointer, "__abstractmethods__", ())):
            setattr(Legacy, name, alist_threads if name == "alist_threads" else _noop)
        Legacy.__abstractmethods__ = frozenset()
        cp = Legacy()
        assert cp.list_threads({"user_id": "u"}, "s", 1, 2) == []
        assert seen["args"] == ({"user_id": "u"}, "s", 1, 2)


async def _noop(*args, **kwargs):
    return None
