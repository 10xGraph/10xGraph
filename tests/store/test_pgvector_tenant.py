"""Unit tests for PgVectorStore tenant/extra columns and RLS (fake pool, SQL shape only).

The SQL runs against a real database in tests/integration/test_pgvector_tenant_real.py.
"""

import asyncpg
import pytest

from tenxgraph.storage.columns import ColumnValueError, MissingColumnValueError
from tenxgraph.storage.store.pgvector_store import PgVectorStore
from tests.store.test_pgvector_store import FakeConn, FakePool, _Embedding, _row


TENANTS = {"org_id": "text", "project_id": "text"}


def _store(ready=True, **kw):
    conn = FakeConn()
    store = PgVectorStore(_Embedding(), pool=FakePool(conn), **kw)
    if ready:
        store._setup_done = True
        store._index_metric = store.distance_metric
    return store, conn


def _cfg(user="u1", org="A", project="P1", reach=None, **extra):
    identity = {"user_id": user, "tenant": {"org_id": org, "project_id": project}, "claims": extra}
    if reach is not None:
        identity["reach"] = reach
    return {"user_id": user, "user": identity}


class TestConstructor:
    def test_columns_parsed(self):
        store, _ = _store(tenant_columns=TENANTS, extra_columns={"public_id": "text"})
        assert store.tenant_column_names == ("org_id", "project_id")
        assert store.columns.extra_names == ("public_id",)

    @pytest.mark.parametrize("name", ["content", "embedding", "tsv", "user_id", "metadata"])
    def test_builtin_names_rejected(self, name):
        with pytest.raises(ValueError, match="built-in column"):
            _store(tenant_columns={name: "text"})

    def test_rls_needs_tenant_columns_and_a_bool(self):
        with pytest.raises(ValueError, match="needs at least one tenant column"):
            _store(row_level_security=True)
        with pytest.raises(ValueError, match="True or False"):
            _store(tenant_columns=TENANTS, row_level_security="yes")

    def test_nothing_declared_keeps_key_index(self):
        store, _ = _store()
        assert store._key_columns() == ["collection", "user_id", "memory_type", "memory_key"]
        assert not store.columns


@pytest.mark.asyncio
class TestReads:
    async def test_get_filters_by_every_tenant_column(self):
        store, conn = _store(tenant_columns=TENANTS)
        conn.fetchrow_results = [_row()]
        await store.aget(_cfg(), "m1")
        sql = conn.sql("fetchrow")[0]
        assert (
            'WHERE "collection" = $1 AND "user_id" = $2 AND "org_id" = $3 AND "project_id" = $4 '
            'AND "id" = $5'
        ) in sql
        assert conn.calls[0][2][:4] == ("agentflow_memories", "u1", "A", "P1")

    async def test_reach_tenant_filters_by_tenant_columns_only(self):
        store, conn = _store(tenant_columns=TENANTS)
        await store.aget_all(_cfg(reach="tenant"))
        where = conn.sql("fetch")[0].split("WHERE")[1].split("ORDER")[0]
        assert '"org_id" = $2' in where and '"project_id" = $3' in where
        assert '"user_id"' not in where

    async def test_reach_all_has_no_user_or_tenant_filter(self):
        store, conn = _store(tenant_columns=TENANTS)
        await store.aget_all(_cfg(reach="all"))
        where = conn.sql("fetch")[0].split("WHERE")[1].split("ORDER")[0]
        assert '"user_id"' not in where and '"org_id"' not in where

    async def test_every_read_path_carries_the_tenant_filter(self):
        store, conn = _store(tenant_columns=TENANTS)
        cfg = _cfg()
        await store.aget_all(cfg)
        await store.asearch(cfg, "q")
        await store.asearch(cfg, "q", retrieval_strategy="text")
        await store.asearch(cfg, "q", retrieval_strategy="hybrid")
        conn.fetchrow_results = [_row()]
        await store.adelete(cfg, "m1")
        await store.aforget_memory(cfg)
        conn.fetchrow_results = [{"id": "m1"}]
        await store.aupdate(cfg, "m1", "new")
        statements = [c[1] for c in conn.calls if "SELECT set_config" not in c[1]]
        assert len(statements) == 7
        for sql in statements:
            assert '"org_id" = $' in sql and '"project_id" = $' in sql, sql

    async def test_missing_picked_value_raises_before_any_sql(self):
        store, conn = _store(tenant_columns=TENANTS)
        cfg = {"user_id": "u1", "user": {"user_id": "u1", "tenant": {"org_id": "A"}}}
        for call in (
            store.aget(cfg, "m1"),
            store.aget_all(cfg),
            store.asearch(cfg, "q"),
            store.adelete(cfg, "m1"),
            store.aforget_memory(cfg),
        ):
            with pytest.raises(ValueError, match="project_id"):
                await call
        assert conn.calls == []

    async def test_values_from_request_config_are_ignored(self):
        store, conn = _store(tenant_columns=TENANTS)
        cfg = {
            "user_id": "u1",
            "org_id": "FORGED",
            "project_id": "FORGED",
            "user": {"user_id": "u1"},
        }
        with pytest.raises(ValueError, match="no tenant value"):
            await store.aget_all(cfg)
        assert conn.calls == []

    async def test_bad_value_type_raises(self):
        store, _ = _store(tenant_columns={"org_id": "uuid"})
        with pytest.raises(ColumnValueError):
            await store.aget_all(
                {"user_id": "u1", "user": {"user_id": "u1", "tenant": {"org_id": "not-a-uuid"}}}
            )

    async def test_reach_tenant_without_tenant_columns_raises(self):
        store, _ = _store()
        with pytest.raises(ValueError, match="tenant columns"):
            await store.aforget_memory(_cfg(reach="tenant"))

    async def test_identity_user_id_wins_over_config_user_id(self):
        store, conn = _store()
        cfg = {"user_id": "forged", "user": {"user_id": "real"}}
        await store.aget_all(cfg)
        assert "real" in conn.calls[0][2] and "forged" not in conn.calls[0][2]


@pytest.mark.asyncio
class TestWrites:
    async def test_store_writes_tenant_and_extra_values(self):
        store, conn = _store(tenant_columns=TENANTS, extra_columns={"public_id": "text"})
        conn.fetchrow_results = [{"id": "m1"}]
        await store.astore(_cfg(public_id="pub"), "hello", memory_id="m1")
        sql = conn.sql("fetchrow")[0]
        assert '"org_id", "project_id", "public_id") VALUES (' in sql
        assert conn.calls[0][2][-3:] == ("A", "P1", "pub")
        assert (
            'WHERE m.user_id = EXCLUDED.user_id AND m."org_id" IS NOT DISTINCT FROM '
            'EXCLUDED."org_id" AND m."project_id" IS NOT DISTINCT FROM EXCLUDED."project_id"'
        ) in sql
        assert '"org_id" = EXCLUDED."org_id"' in sql  # the replaced record takes the writer's

    async def test_keyed_upsert_uses_tenant_key(self):
        store, conn = _store(tenant_columns=TENANTS, extra_columns={"public_id": "text"})
        conn.fetchrow_results = [{"id": "m1"}]
        await store.astore(_cfg(), "hello", memory_key="name")
        sql = conn.sql("fetchrow")[0]
        assert (
            'ON CONFLICT (collection, user_id, "org_id", "project_id", memory_type, memory_key) '
            "WHERE memory_key IS NOT NULL"
        ) in sql
        assert '"public_id" = COALESCE(EXCLUDED."public_id", m."public_id")' in sql

    async def test_required_values_enforced_on_write(self):
        store, conn = _store(
            tenant_columns=TENANTS, extra_columns={"plan": {"type": "text", "required": True}}
        )
        with pytest.raises(MissingColumnValueError, match="plan"):
            await store.astore(_cfg(), "x")
        with pytest.raises(MissingColumnValueError, match="plan"):
            await store.aupdate(_cfg(), "m1", "x")
        assert conn.calls == []

    async def test_optional_tenant_value_may_be_missing_on_store(self):
        store, conn = _store(
            tenant_columns={"org_id": "text", "team": {"type": "text", "required": False}}
        )
        conn.fetchrow_results = [{"id": "m1"}]
        await store.astore(
            {"user_id": "u1", "user": {"user_id": "u1", "tenant": {"org_id": "A"}}},
            "x",
            memory_id="m1",
        )
        assert conn.calls[0][2][-2:] == ("A", None)

    async def test_update_sets_only_present_extras_never_tenants(self):
        store, conn = _store(
            tenant_columns=TENANTS, extra_columns={"public_id": "text", "plan": "text"}
        )
        conn.fetchrow_results = [{"id": "m1"}]
        await store.aupdate(_cfg(plan="pro"), "m1", "x")
        set_part = conn.sql("fetchrow")[0].split(" WHERE ")[0]
        assert '"plan" = $' in set_part
        assert '"public_id"' not in set_part and '"org_id"' not in set_part


@pytest.mark.asyncio
class TestRlsAndIterativeScan:
    async def test_rls_sets_values_in_a_transaction(self):
        store, conn = _store(tenant_columns=TENANTS, row_level_security=True)
        await store.aget_all(_cfg())
        _method, sql, args = conn.calls[0]
        assert (
            sql == "SELECT set_config($1, $2, true), set_config($3, $4, true), "
            "set_config($5, $6, true)"
        )
        assert args == (
            "tenxgraph.scope",
            "+,org_id,project_id",
            "tenxgraph.org_id",
            "A",
            "tenxgraph.project_id",
            "P1",
        )
        assert conn.transactions == 1

    async def test_no_rls_no_transaction(self):
        store, conn = _store(tenant_columns=TENANTS)
        await store.aget_all(_cfg())
        assert conn.transactions == 0
        assert not any("set_config" in s for s in conn.sql())

    async def test_backfill_under_rls_sees_all_rows(self):
        store, conn = _store(tenant_columns=TENANTS, row_level_security=True)
        await store.abackfill_embeddings()
        assert conn.calls[0][2][:2] == ("tenxgraph.scope", "+")

    async def test_iterative_scan_only_for_vector_search_when_supported(self):
        store, conn = _store()
        await store.asearch({"user_id": "u1"}, "q")
        assert not any("set_config" in s for s in conn.sql())
        store._iterative_scan = True
        await store.asearch({"user_id": "u1"}, "q")
        assert conn.calls[-2][2] == ("hnsw.iterative_scan", "strict_order")
        conn.calls.clear()
        await store.asearch({"user_id": "u1"}, "q", retrieval_strategy="text")
        assert not any("set_config" in s for s in conn.sql())

    async def test_rls_policy_error_on_upsert_is_permission_error(self):
        store, conn = _store(tenant_columns=TENANTS, row_level_security=True)
        conn.fetchrow_results = [
            asyncpg.InsufficientPrivilegeError("new row violates row-level security policy")
        ]
        with pytest.raises(PermissionError):
            await store.astore(_cfg(), "x", memory_id="m1")


@pytest.mark.asyncio
class TestSetup:
    async def test_setup_adds_missing_columns_indexes_and_policy(self):
        store, conn = _store(
            ready=False,
            tenant_columns={"org_id": "uuid"},
            extra_columns={"plan": "text"},
            row_level_security=True,
        )

        def respond(sql, args):
            if "extversion" in sql:
                return "0.8.0"
            if "format_type" in sql:
                return "vector(4)"
            if "server_version_num" in sql:
                return 170000
            return None

        conn.fetchval_responder = respond
        await store.asetup()
        executed = conn.sql("execute")
        assert not any("UNIQUE INDEX IF NOT EXISTS" in s for s in executed[:8])
        assert any('ADD COLUMN IF NOT EXISTS "org_id" UUID' in s for s in executed)
        assert any('ADD COLUMN IF NOT EXISTS "plan" TEXT' in s for s in executed)
        assert any('"tenxgraph_memories_c_org_id" ON' in s for s in executed)
        # With tenant columns the key index name carries a hash of its definition, so it never
        # reuses the name of the index older processes still rely on.
        assert any(
            'CREATE UNIQUE INDEX "tenxgraph_memories_kuq_' in s
            and '(collection, user_id, "org_id", memory_type, memory_key) NULLS NOT DISTINCT' in s
            for s in executed
        )
        assert any(s.startswith("CREATE POLICY") and "tenxgraph.org_id" in s for s in executed)
        assert any("ENABLE ROW LEVEL SECURITY" in s for s in executed)
        assert any("FORCE ROW LEVEL SECURITY" in s for s in executed)
        assert store._iterative_scan is True

    async def test_setup_with_nothing_declared_runs_no_column_or_policy_sql(self):
        store, conn = _store(ready=False)
        conn.fetchval_responder = lambda sql, args: (
            "0.7.0" if "extversion" in sql else "vector(4)" if "format_type" in sql else None
        )
        await store.asetup()
        assert not any("pg_policies" in s or "ADD COLUMN" in s for s in conn.sql())
        assert store._iterative_scan is False  # pgvector 0.7 has no iterative scans
