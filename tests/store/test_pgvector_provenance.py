"""PgVectorStore origin and expiry: the SQL each method sends (fake pool, no database).

The real behaviour (migration, top-k, cleanup) is checked in
tests/integration/test_pgvector_provenance_real.py.
"""

from datetime import UTC, datetime, timedelta

import pytest

from tenxgraph.storage.store import TRUSTED_ORIGINS, MemoryOrigin
from tenxgraph.storage.store.pgvector_store import STORE_BUILTIN_COLUMNS, PgVectorStore
from tenxgraph.storage.store.store_schema import RetrievalStrategy
from tests.store.test_pgvector_store import FakeConn, FakePool, _Embedding, _row


CFG = {"user_id": "u1"}
LIVE = '(m."expires_at" IS NULL OR m."expires_at" > now())'
FUTURE = datetime.now(UTC) + timedelta(days=1)


def _store(**kw):
    conn = FakeConn()
    store = PgVectorStore(_Embedding(), pool=FakePool(conn), **kw)
    store._setup_done = True
    store._index_metric = store.distance_metric
    return store, conn


def test_flag_and_reserved_names():
    assert PgVectorStore.supports_provenance is True
    assert {"origin", "expires_at"} <= STORE_BUILTIN_COLUMNS
    with pytest.raises(ValueError):
        PgVectorStore(None, pool=FakePool(FakeConn()), extra_columns={"origin": "text"})


@pytest.mark.asyncio
class TestReads:
    @pytest.mark.parametrize(
        "strategy",
        [RetrievalStrategy.SIMILARITY, RetrievalStrategy.TEXT, RetrievalStrategy.HYBRID],
    )
    async def test_search_hides_expired_in_the_query(self, strategy):
        store, conn = _store()
        await store.asearch(CFG, "q", retrieval_strategy=strategy)
        sql = conn.sql("fetch")[0]
        # Hybrid: both candidate lists carry the clause (the scope is used twice).
        assert sql.count(LIVE) == (2 if strategy == RetrievalStrategy.HYBRID else 1)
        assert sql.index(LIVE) < sql.index("LIMIT")
        assert '"origin" = ANY' not in sql  # no origin filter by default

    async def test_search_origins_clause(self):
        store, conn = _store()
        await store.asearch(CFG, "q", origins=list(TRUSTED_ORIGINS))
        _, sql, args = conn.calls[0]
        assert 'm."origin" = ANY($3::text[])' in sql
        assert 'IS NULL OR m."origin"' not in sql
        assert args[2] == ["system", "user"]

    async def test_origins_with_none_keeps_unknown(self):
        store, conn = _store()
        await store.asearch(
            CFG, "q", retrieval_strategy=RetrievalStrategy.HYBRID, origins=["agent", None]
        )
        sql = conn.sql("fetch")[0]
        assert sql.count('(m."origin" = ANY($3::text[]) OR m."origin" IS NULL)') == 2

    async def test_get_and_list_hide_expired(self):
        store, conn = _store()
        conn.fetchrow_results = [_row(origin="user", expires_at=FUTURE)]
        got = await store.aget(CFG, "m1")
        assert (got.origin, got.expires_at) == (MemoryOrigin.USER, FUTURE)
        await store.aget_all(CFG, origins=[None])
        get_sql, list_sql = conn.sql("fetchrow")[0], conn.sql("fetch")[0]
        assert '("expires_at" IS NULL OR "expires_at" > now())' in get_sql
        assert '"origin" IS NULL' in list_sql and "expires_at" in list_sql

    async def test_unknown_stored_origin_reads_as_none(self):
        store, conn = _store()
        conn.fetchrow_results = [_row(origin="web")]
        assert (await store.aget(CFG, "m1")).origin is None

    async def test_bad_origins_refused(self):
        store, _ = _store()
        with pytest.raises(ValueError, match="unknown origin"):
            await store.asearch(CFG, "q", origins=["web"])


@pytest.mark.asyncio
class TestWrites:
    async def test_store_writes_both_columns_and_upserts_them(self):
        store, conn = _store()
        conn.fetchrow_results = [{"id": "m1"}, {"id": "m1"}]
        await store.astore(CFG, "x", origin=MemoryOrigin.AGENT, expires_at=FUTURE)
        await store.astore(CFG, "x", memory_key="k")
        (_, plain, args), (_, keyed, keyed_args) = conn.calls
        assert args[-2:] == ("agent", FUTURE)
        assert "origin, expires_at) VALUES" in plain
        for sql in (plain, keyed):
            assert "origin = EXCLUDED.origin, expires_at = EXCLUDED.expires_at" in sql
        assert keyed_args[-2:] == (None, None)

    async def test_store_refuses_naive_expires_at(self):
        store, conn = _store()
        with pytest.raises(ValueError, match="timezone-aware"):
            await store.astore(CFG, "x", expires_at=datetime(2030, 1, 1))
        assert conn.calls == []

    async def test_update_keeps_when_absent(self):
        store, conn = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        await store.aupdate(CFG, "m1", "x", origin="system")
        _, sql, args = conn.calls[0]
        assert "origin = COALESCE($" in sql and "expires_at = COALESCE($" in sql
        assert args[-2:] == ("system", None)
        # Updates are not filtered by expiry: an expired memory can be given a new one.
        assert "now()" not in sql.split("WHERE", 1)[1]

    async def test_adelete_expired_batches_until_short(self):
        store, conn = _store()
        conn.execute_results = ["DELETE 2", "DELETE 2", "DELETE 1"]
        assert await store.adelete_expired(batch_size=2) == 5
        sqls = conn.sql("execute")
        assert len(sqls) == 3
        assert (
            "WHERE expires_at <= now() ORDER BY expires_at LIMIT $1 FOR UPDATE SKIP LOCKED"
            in (sqls[0])
        )
        assert conn.calls[0][2] == (2,)

    async def test_adelete_expired_one_collection(self):
        store, conn = _store()
        conn.execute_results = ["DELETE 0"]
        assert await store.adelete_expired(collection="c1") == 0
        _, sql, args = conn.calls[0]
        assert "AND collection = $2" in sql and args == (1000, "c1")

    async def test_adelete_expired_rls_reads_all_tenants(self):
        store, conn = _store(tenant_columns=["org_id"], row_level_security=True)
        conn.execute_results = ["DELETE 0"]
        await store.adelete_expired()
        assert any("set_config" in s for s in conn.sql())

    async def test_adelete_expired_bad_batch(self):
        store, _ = _store()
        with pytest.raises(ValueError, match="batch_size"):
            await store.adelete_expired(batch_size=0)
