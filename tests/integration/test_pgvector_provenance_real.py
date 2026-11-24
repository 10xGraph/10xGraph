"""PgVectorStore origin and expiry against a REAL Postgres with pgvector.

    pytest tests/integration/test_pgvector_provenance_real.py --integration

Uses POSTGRES_DSN (see test_pgvector_store_real.py). Checks the v2 -> v3 migration on an existing
table, that expired rows never shrink top-k (the filter runs before LIMIT) in every strategy,
the origins filter, keyed upsert revival and batched cleanup (also under RLS).
"""

from datetime import UTC, datetime, timedelta

import pytest

from tenxgraph.storage.store import TRUSTED_ORIGINS, MemoryOrigin, PgVectorStore
from tenxgraph.storage.store.pgvector_store import CURRENT_SCHEMA_VERSION
from tenxgraph.storage.store.store_schema import RetrievalStrategy
from tests.integration.test_pgvector_store_real import WordEmbedding, pool, table  # noqa: F401
from tests.integration.test_pgvector_tenant_real import _cfg as _tenant_cfg
from tests.integration.test_pgvector_tenant_real import rls_role  # noqa: F401


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

CFG = {"user_id": "alice"}
PAST = datetime.now(UTC) - timedelta(hours=1)
FUTURE = datetime.now(UTC) + timedelta(days=1)


def _store(pool, table, **kw):  # noqa: F811
    return PgVectorStore(WordEmbedding(), pool=pool, table=table, **kw)


async def _downgrade_to_v2(pool, table):  # noqa: F811
    """Turn a fresh table into what a 1.0 (schema v2) setup left behind."""
    qualified = f'"public"."{table}"'
    await pool.execute(f'DROP INDEX IF EXISTS "public"."{table}_exp_idx"')
    await pool.execute(f"ALTER TABLE {qualified} DROP COLUMN origin, DROP COLUMN expires_at")
    version_table = f'"public"."{table}_schema_version"'
    await pool.execute(f"DELETE FROM {version_table}")
    await pool.execute(f"INSERT INTO {version_table} (version) VALUES (2)")


V2_INSERT = (
    "INSERT INTO {t} (id, collection, user_id, memory_type, category, content, metadata) "
    "VALUES ($1, 'agentflow_memories', 'alice', 'episodic', 'general', $2, '{{}}'::jsonb)"
)


class TestMigration:
    async def test_v2_table_upgrades_in_place(self, pool, table):  # noqa: F811
        await _store(pool, table).asetup()
        await _downgrade_to_v2(pool, table)
        qualified = f'"public"."{table}"'
        await pool.execute(V2_INSERT.format(t=qualified), "old", "old memory about tea")

        store = _store(pool, table)
        await store.asetup()
        assert await store.aget_schema_version() == CURRENT_SCHEMA_VERSION == 3
        old = await store.aget(CFG, "old")
        assert (old.origin, old.expires_at) == (None, None)
        indexdef = await pool.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = $1", f"{table}_exp_idx"
        )
        assert "WHERE (expires_at IS NOT NULL)" in indexdef
        # An old process still inserts with the v2 column list.
        await pool.execute(V2_INSERT.format(t=qualified), "old2", "written by 1.0")
        assert (await store.aget(CFG, "old2")).origin is None
        # Setup again is a no-op.
        await _store(pool, table).asetup()


class TestReads:
    @pytest.mark.parametrize(
        "strategy",
        [RetrievalStrategy.SIMILARITY, RetrievalStrategy.TEXT, RetrievalStrategy.HYBRID],
    )
    async def test_expired_nearest_rows_do_not_shrink_top_k(
        self,
        pool,  # noqa: F811
        table,  # noqa: F811
        strategy,
    ):
        store = _store(pool, table)
        # The exact query text is the best match for every strategy, but those rows expired.
        for i in range(3):
            await store.astore(CFG, "green tea", memory_id=f"x{i}", expires_at=PAST)
        live = {f"l{i}" for i in range(2)}
        for mid in live:
            await store.astore(CFG, "green tea with milk", memory_id=mid, expires_at=FUTURE)
        found = await store.asearch(CFG, "green tea", limit=2, retrieval_strategy=strategy)
        assert {r.id for r in found} == live
        assert {r.id for r in await store.aget_all(CFG)} == live
        assert await store.aget(CFG, "x0") is None

    async def test_origins_filter_and_default(self, pool, table):  # noqa: F811
        store = _store(pool, table)
        for origin in (MemoryOrigin.USER, MemoryOrigin.AGENT, MemoryOrigin.SYSTEM, None):
            await store.astore(CFG, "note", memory_id=str(origin), origin=origin)
        assert len(await store.asearch(CFG, "note")) == 4
        trusted = await store.asearch(CFG, "note", origins=list(TRUSTED_ORIGINS))
        assert {r.origin for r in trusted} == {MemoryOrigin.USER, MemoryOrigin.SYSTEM}
        mixed = await store.aget_all(CFG, origins=[MemoryOrigin.AGENT, None])
        assert {r.origin for r in mixed} == {MemoryOrigin.AGENT, None}
        assert len(await store.asearch(CFG, "note", filters={"origin": "user"})) == 1


class TestWrites:
    async def test_keyed_upsert_revives_expired_row(self, pool, table):  # noqa: F811
        store = _store(pool, table)
        mid = await store.astore(CFG, "uses vim", memory_key="editor", expires_at=PAST)
        assert await store.aget(CFG, mid) is None
        again = await store.astore(
            CFG, "uses helix", memory_key="editor", origin=MemoryOrigin.AGENT
        )
        got = await store.aget(CFG, again)
        assert again == mid
        assert (got.content, got.origin, got.expires_at) == ("uses helix", MemoryOrigin.AGENT, None)

    async def test_update_keeps_or_replaces(self, pool, table):  # noqa: F811
        store = _store(pool, table)
        mid = await store.astore(CFG, "a", origin=MemoryOrigin.TOOL, expires_at=FUTURE)
        await store.aupdate(CFG, mid, "b")
        got = await store.aget(CFG, mid)
        assert (got.origin, got.expires_at) == (MemoryOrigin.TOOL, FUTURE)
        await store.aupdate(CFG, mid, "c", origin=MemoryOrigin.SYSTEM, expires_at=PAST)
        assert await store.aget(CFG, mid) is None
        await store.aupdate(CFG, mid, "d", expires_at=FUTURE)  # an expired row can be updated
        assert (await store.aget(CFG, mid)).origin == MemoryOrigin.SYSTEM

    async def test_naive_expires_at_refused(self, pool, table):  # noqa: F811
        with pytest.raises(ValueError, match="timezone-aware"):
            await _store(pool, table).astore(CFG, "x", expires_at=datetime(2030, 1, 1))  # noqa: DTZ001


class TestCleanup:
    async def test_adelete_expired_in_batches(self, pool, table):  # noqa: F811
        store = _store(pool, table)
        await store.astore(CFG, "keep")
        await store.astore(CFG, "later", expires_at=FUTURE)
        for i in range(5):
            await store.astore({"user_id": f"u{i}"}, f"old {i}", expires_at=PAST)
        await store.astore({**CFG, "collection": "other"}, "old other", expires_at=PAST)
        assert await store.adelete_expired(batch_size=2, collection="agentflow_memories") == 5
        assert await store.adelete_expired(batch_size=2) == 1
        assert await store.adelete_expired() == 0
        assert len(await store.aget_all(CFG)) == 2

    async def test_adelete_expired_under_rls_reaches_every_tenant(
        self,
        pool,  # noqa: F811
        table,  # noqa: F811
        rls_role,  # noqa: F811
    ):
        _role, role_pool = rls_role
        store = _store(
            role_pool, table, tenant_columns={"org_id": "text", "project_id": "text"},
            row_level_security=True,
        )
        for org in ("A", "B"):
            await store.astore(_tenant_cfg(org=org), f"org {org}", expires_at=PAST)
        await store.astore(_tenant_cfg(org="A"), "live")
        assert await store.adelete_expired() == 2
        assert await pool.fetchval(f'SELECT count(*) FROM "public"."{table}"') == 1
