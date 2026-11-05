"""Sprint 7 review fixes in PgVectorStore (fake pool, no database).

S3: every column in the search SQL is qualified, so declared columns cannot collide with names
the SQL introduces. S5: zero vectors and non-finite scores. S2: the setup session, the key
index of an earlier declaration, the lock-timeout option. Real Postgres behaviour is in
tests/integration/test_store_review_fixes_real.py.
"""

import re

import asyncpg
import pytest

from tenxgraph.storage.store.pgvector_store import PgVectorStore
from tenxgraph.storage.store.store_schema import DistanceMetric

from .test_pgvector_store import CFG, FakeConn, FakePool, _Embedding, _row, _store


SQL_NAMES = ("tsq", "q", "score", "similarity", "rnk", "rrf", "m", "f", "vec", "txt", "fused")
TCFG = {"user_id": "u1", "user": {"user_id": "u1", "org_id": "A"}}


def _unqualified(sql: str, name: str) -> bool:
    """Whether ``name`` appears as a column reference that is not ``alias.name``."""
    quoted = f'"{name}"'
    return bool(re.search(rf'(?<![.\w]){re.escape(quoted)}', sql))


@pytest.mark.asyncio
class TestQualifiedSearchSql:
    @pytest.mark.parametrize("strategy", ["similarity", "text", "hybrid"])
    async def test_declared_columns_never_unqualified(self, strategy):
        extra = {name: "text" for name in SQL_NAMES}
        store, conn, _ = _store(tenant_columns=["org_id"], extra_columns=extra)
        filters = {name: f"v_{name}" for name in ("tsq", "q", "score", "rnk")}
        await store.asearch(TCFG, "apples", retrieval_strategy=strategy, filters=filters)
        _, sql, _ = conn.calls[-1]
        for name in ("org_id", "tsq", "q", "score", "rnk"):
            assert not _unqualified(sql, name), (name, sql)
        assert " tsq " not in sql and "tsv @@ tsq " not in sql
        if strategy != "similarity":
            assert "AS tsq_input(q)" in sql and "m.tsv @@ tsq_input.q" in sql


@pytest.mark.asyncio
class TestZeroVectors:
    async def test_zero_vector_write_refused_for_cosine(self):
        store, conn, _ = _store(embedding=_Embedding(vector=[0.0, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="zero vector"):
            await store.astore(CFG, "x")
        assert conn.calls == []

    async def test_zero_vector_write_allowed_for_euclidean(self):
        store, conn, _ = _store(
            embedding=_Embedding(vector=[0.0, 0.0, 0.0, 0.0]),
            distance_metric=DistanceMetric.EUCLIDEAN,
        )
        conn.fetchrow_results = [{"id": "m1"}]
        assert await store.astore(CFG, "x") == "m1"

    async def test_zero_query_falls_back_to_text(self):
        store, conn, _ = _store(embedding=_Embedding(vector=[0.0, 0.0, 0.0, 0.0]))
        await store.asearch(CFG, "apples")
        assert "websearch_to_tsquery" in conn.calls[-1][1]
        assert "<=>" not in conn.calls[-1][1]

    async def test_zero_query_raises_under_raise(self):
        store, _, _ = _store(
            embedding=_Embedding(vector=[0.0, 0.0, 0.0, 0.0]), on_embedding_error="raise"
        )
        with pytest.raises(ValueError, match="zero vector"):
            await store.asearch(CFG, "apples")

    async def test_zero_vector_backfill_refused(self):
        store, conn, _ = _store(embedding=_Embedding(vector=[0.0, 0.0, 0.0, 0.0]))
        conn.fetch_results = [[{"collection": "c", "id": "1", "content": "x"}]]
        with pytest.raises(ValueError, match="zero vector"):
            await store.abackfill_embeddings()

    async def test_nan_score_never_passes_threshold(self):
        store, conn, _ = _store()
        conn.fetch_results = [[_row(id="nan", score=float("nan")), _row(id="ok", score=0.9)]]
        results = await store.asearch(CFG, "q", score_threshold=0.5)
        assert [r.id for r in results] == ["ok"]

    async def test_nan_hybrid_similarity_never_passes_threshold(self):
        store, conn, _ = _store(retrieval_strategy="hybrid")
        conn.fetch_results = [
            [
                _row(id="nan", score=0.03, similarity=float("nan")),
                _row(id="ok", score=0.02, similarity=0.9),
            ]
        ]
        results = await store.asearch(CFG, "q", score_threshold=0.5)
        assert [r.id for r in results] == ["ok"]


def _key_row(name, columns, valid=True, nulls=False):
    definition = "CREATE UNIQUE INDEX ... " + ("NULLS NOT DISTINCT " if nulls else "")
    return {
        "relname": name,
        "indisvalid": valid,
        "qualified": f'"{name}"',
        "definition": definition,
        "columns": columns,
    }


@pytest.mark.asyncio
class TestSetupSession:
    async def test_ddl_lock_timeout_validated(self):
        with pytest.raises(ValueError, match="ddl_lock_timeout"):
            PgVectorStore(_Embedding(), pool=FakePool(FakeConn()), ddl_lock_timeout=0)
        store = PgVectorStore(_Embedding(), pool=FakePool(FakeConn()), ddl_lock_timeout=None)
        assert store.ddl_lock_timeout is None

    async def test_existing_table_with_everything_in_place_runs_no_ddl(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = lambda sql, args: (
            "0.8.0"
            if "extversion" in sql
            else True
            if "to_regclass($1) IS NOT NULL" in sql
            else "vector(4)"
            if "format_type" in sql
            else "vector_cosine_ops"
            if "opcname" in sql
            else 2
            if "max(version)" in sql
            else None
        )
        t = "tenxgraph_memories"
        catalog = [
            {"relname": f"{t}_{s}", "indisvalid": True, "qualified": f'"{t}_{s}"'}
            for s in ("uid_idx", "tsv_idx", "emb_idx", "key_uq")
        ]
        keys = [_key_row(f"{t}_key_uq", ["collection", "user_id", "memory_type", "memory_key"])]
        conn.fetch_results = [catalog, keys]
        await store.asetup()
        ddl = [s for s in conn.sql("execute") if s.split()[0] in ("CREATE", "ALTER", "DROP")]
        assert ddl == []
        executed = conn.sql("execute")
        assert executed[0] == "SELECT pg_advisory_lock($1)"
        assert executed[1] == "SELECT set_config('lock_timeout', $1, false)"
        assert conn.calls[1][2] == ("10000ms",)
        assert executed[-2:] == ["RESET lock_timeout", "SELECT pg_advisory_unlock($1)"]

    async def test_legacy_key_index_kept_and_new_one_built_concurrently(self):
        store, conn, _ = _store(ready=False, tenant_columns=["org_id"])
        conn.fetchval_responder = lambda sql, args: (
            "0.8.0"
            if "extversion" in sql
            else True
            if "to_regclass($1) IS NOT NULL" in sql
            else "vector(4)"
            if "format_type(a.atttypid" in sql and "embedding" in sql
            else "vector_cosine_ops"
            if "opcname" in sql
            else 170000
            if "server_version_num" in sql
            else 2
            if "max(version)" in sql
            else None
        )
        t = "tenxgraph_memories"
        catalog = [
            {"relname": f"{t}_{s}", "indisvalid": True, "qualified": f'"{t}_{s}"'}
            for s in ("uid_idx", "tsv_idx", "emb_idx", "key_uq")
        ]
        legacy = _key_row(f"{t}_key_uq", ["collection", "user_id", "memory_type", "memory_key"])
        # Fetch order: pg_attribute (ensure_columns), pg_policies (ensure_rls), the index
        # catalog, then the key indexes.
        conn.fetch_results = [[], [], catalog, [legacy]]
        await store.asetup()
        executed = conn.sql("execute")
        assert not any(s.startswith("DROP INDEX") for s in executed)
        built = [s for s in executed if s.startswith("CREATE UNIQUE INDEX CONCURRENTLY")]
        assert len(built) == 1 and f'"{t}_kuq_' in built[0]
        assert '(collection, user_id, "org_id", memory_type, memory_key) NULLS NOT DISTINCT' in (
            built[0]
        )
        assert any(s.startswith(f'CREATE INDEX CONCURRENTLY "{t}_c_org_id"') for s in executed)
        assert store._legacy_key_indexes == [f"{t}_key_uq"]

    async def test_legacy_key_violation_message(self):
        store, conn, _ = _store(tenant_columns=["org_id"])
        err = asyncpg.UniqueViolationError("duplicate key")
        err.constraint_name = "tenxgraph_memories_key_uq"
        conn.fetchrow_results = [err]
        with pytest.raises(ValueError, match="adrop_legacy_key_index"):
            await store.astore(TCFG, "x", memory_key="k")

    async def test_drop_legacy_refuses_without_current_index(self):
        store, conn, _ = _store(tenant_columns=["org_id"])
        conn.fetch_results = [
            [_key_row("tenxgraph_memories_key_uq", ["collection", "user_id", "memory_type"])]
        ]
        with pytest.raises(RuntimeError, match="run asetup"):
            await store.adrop_legacy_key_index()
        assert not any(s.startswith("DROP") for s in conn.sql("execute"))
