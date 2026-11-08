"""PgVectorStore against a REAL Postgres with the pgvector extension.

The unit tests (tests/store/test_pgvector_store.py) check the SQL text; these check that the
SQL runs and does what it claims: DDL, operators, upserts, ownership, fallback and backfill.

Run with a pgvector image:

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=test -e POSTGRES_USER=test \\
        -e POSTGRES_DB=test_agentflow pgvector/pgvector:0.8.7-pg17

    pytest tests/integration/test_pgvector_store_real.py --integration

The DSN comes from POSTGRES_DSN (same default as the checkpointer integration tests). Tests skip
when the database cannot be reached or has no pgvector extension available. Each test uses its
own table and drops it afterwards.
"""

import asyncio
import math
import os
import uuid
import zlib

import pytest
import pytest_asyncio

from tenxgraph.storage.store import PgVectorStore
from tenxgraph.storage.store.long_term_memory import _do_write
from tenxgraph.storage.store.pgvector_store import CURRENT_SCHEMA_VERSION
from tenxgraph.storage.store.store_schema import MemoryType


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow"
)

DIM = 8


class WordEmbedding:
    """Deterministic bag-of-words embedding: similar text -> similar vectors."""

    dimension = DIM

    def __init__(self, fail=False):
        self.fail = fail

    async def aembed(self, text):
        if self.fail:
            raise RuntimeError("embedding service down")
        vector = [0.0] * DIM
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector] if any(vector) else [1.0] + [0.0] * (DIM - 1)

    async def aembed_query(self, text):
        return await self.aembed(text)

    async def aembed_batch(self, texts):
        return [await self.aembed(t) for t in texts]


@pytest_asyncio.fixture
async def pool():
    asyncpg = pytest.importorskip("asyncpg")
    try:
        pg_pool = await asyncio.wait_for(
            asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=8), timeout=5
        )
    except Exception as e:  # any connection failure means: no database here
        pytest.skip(f"Postgres not reachable at POSTGRES_DSN: {e}")
    available = await pg_pool.fetchval(
        "SELECT count(*) FROM pg_available_extensions WHERE name = 'vector'"
    )
    if not available:
        await pg_pool.close()
        pytest.skip("pgvector extension not available on this server")
    yield pg_pool
    await pg_pool.close()


@pytest_asyncio.fixture
async def table(pool):
    name = f"mem_{uuid.uuid4().hex[:12]}"
    yield name
    await pool.execute(f'DROP TABLE IF EXISTS "public"."{name}"')
    await pool.execute(f'DROP TABLE IF EXISTS "public"."{name}_schema_version"')


def _store(pool, table, embedding=None, **kw):
    emb = WordEmbedding() if embedding is None else embedding
    return PgVectorStore(emb, pool=pool, table=table, **kw)


def _cfg(user="alice", **kw):
    return {"user_id": user, **kw}


class TestSetup:
    async def test_setup_is_idempotent_and_versioned(self, pool, table):
        store = _store(pool, table)
        await store.asetup()
        await _store(pool, table).asetup()
        assert await store.aget_schema_version() == CURRENT_SCHEMA_VERSION
        indexes = {
            r["indexname"]
            for r in await pool.fetch(
                "SELECT indexname FROM pg_indexes WHERE tablename = $1", table
            )
        }
        assert {f"{table}_uid_idx", f"{table}_key_uq", f"{table}_tsv_idx", f"{table}_emb_idx"} <= (
            indexes
        )

    async def test_concurrent_setup_from_two_stores(self, pool, table):
        await asyncio.gather(_store(pool, table).asetup(), _store(pool, table).asetup())

    async def test_dimension_mismatch_raises(self, pool, table):
        await _store(pool, table).asetup()

        class Wide(WordEmbedding):
            dimension = DIM * 2

        with pytest.raises(ValueError, match="dimension mismatch"):
            await _store(pool, table, embedding=Wide()).asetup()

    async def test_text_only_store(self, pool, table):
        store = PgVectorStore(None, pool=pool, table=table)
        await store.astore(_cfg(), "the quick brown fox")
        results = await store.asearch(_cfg(), "fox")
        assert [r.content for r in results] == ["the quick brown fox"]
        column = await pool.fetchval(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = $1 AND column_name = 'embedding'",
            table,
        )
        assert column == 0


class TestCrudAndOwnership:
    async def test_store_get_update_delete(self, pool, table):
        store = _store(pool, table)
        mid = await store.astore(_cfg(thread_id="t1"), "likes green tea", metadata={"s": 1})
        got = await store.aget(_cfg(), mid)
        assert got.content == "likes green tea" and got.thread_id == "t1"
        assert got.metadata["s"] == 1
        await store.aupdate(_cfg(), mid, "likes black tea", metadata={"x": 2})
        got = await store.aget(_cfg(), mid)
        assert got.content == "likes black tea" and got.metadata["s"] == 1
        assert got.metadata["x"] == 2
        await store.adelete(_cfg(), mid)
        assert await store.aget(_cfg(), mid) is None

    async def test_other_user_cannot_read_update_delete_or_overwrite(self, pool, table):
        store = _store(pool, table)
        mid = await store.astore(_cfg("alice"), "alice secret", memory_id="shared-id")
        assert await store.aget(_cfg("bob"), mid) is None
        with pytest.raises(ValueError, match="not found"):
            await store.aupdate(_cfg("bob"), mid, "hijack")
        with pytest.raises(ValueError, match="not found"):
            await store.adelete(_cfg("bob"), mid)
        with pytest.raises(PermissionError):
            await store.astore(_cfg("bob"), "hijack", memory_id="shared-id")
        assert (await store.aget(_cfg("alice"), mid)).content == "alice secret"
        assert await store.asearch(_cfg("bob"), "alice secret") == []

    async def test_owner_can_overwrite_own_id(self, pool, table):
        store = _store(pool, table)
        await store.astore(_cfg(), "v1", memory_id="mine")
        await store.astore(_cfg(), "v2", memory_id="mine")
        assert (await store.aget(_cfg(), "mine")).content == "v2"

    async def test_reach_all_reads_every_user(self, pool, table):
        store = _store(pool, table)
        await store.astore(_cfg("alice"), "alpha")
        await store.astore(_cfg("bob"), "beta")
        admin = {"user": {"user_id": "admin", "reach": "all"}}
        assert len(await store.aget_all(admin)) == 2

    async def test_forget_memory_narrowing(self, pool, table):
        store = _store(pool, table)
        await store.astore(_cfg(), "a", category="keep")
        await store.astore(_cfg(), "b", category="drop", metadata={"tag": "x"})
        await store.astore(_cfg("bob"), "c", category="drop")
        await store.aforget_memory(_cfg(), category="drop", filters={"tag": "x"})
        assert [r.content for r in await store.aget_all(_cfg())] == ["a"]
        assert len(await store.aget_all(_cfg("bob"))) == 1

    async def test_collections_are_separate(self, pool, table):
        store = _store(pool, table)
        await store.astore(_cfg(collection="c1"), "same text", memory_id="x")
        await store.astore(_cfg(collection="c2"), "same text", memory_id="x")
        assert len(await store.aget_all(_cfg(collection="c1"))) == 1


class TestMemoryKey:
    async def test_key_upsert_updates_in_place(self, pool, table):
        store = _store(pool, table)
        first = await store.astore(_cfg(), "name is Atharv", metadata={"memory_key": "name"})
        second = await store.astore(_cfg(), "name is Prashant", metadata={"memory_key": "name"})
        assert first == second
        rows = await store.aget_all(_cfg())
        assert [r.content for r in rows] == ["name is Prashant"]
        assert rows[0].metadata["memory_key"] == "name"

    async def test_concurrent_key_upserts_leave_one_row(self, pool, table):
        store = _store(pool, table)
        await store.asetup()
        ids = await asyncio.gather(
            *(store.astore(_cfg(), f"value {i}", memory_key="k") for i in range(10))
        )
        assert len(set(ids)) == 1
        assert len(await store.aget_all(_cfg())) == 1

    async def test_same_key_different_users(self, pool, table):
        store = _store(pool, table)
        a = await store.astore(_cfg("alice"), "x", memory_key="k")
        b = await store.astore(_cfg("bob"), "y", memory_key="k")
        assert a != b

    async def test_do_write_uses_upsert(self, pool, table):
        store = _store(pool, table)
        for text in ("lang is go", "lang is rust"):
            result = await _do_write(
                store,
                _cfg(thread_id="t"),
                "store",
                text,
                "",
                MemoryType.SEMANTIC,
                "general",
                {"memory_key": "lang"},
                "merge",
            )
            assert result["status"] == "upserted_by_key"
        assert [r.content for r in await store.aget_all(_cfg())] == ["lang is rust"]


class TestSearch:
    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot_product", "manhattan"])
    async def test_similarity_each_metric(self, pool, table, metric):
        store = _store(pool, table, distance_metric=metric)
        await store.astore(_cfg(), "python programming language")
        await store.astore(_cfg(), "favorite food is pizza")
        results = await store.asearch(_cfg(), "python programming language", limit=2)
        assert results[0].content == "python programming language"
        assert results[0].score >= results[-1].score

    async def test_identical_text_passes_dedup_threshold(self, pool, table):
        store = _store(pool, table)
        await store.astore(_cfg(), "my name is Atharv")
        results = await store.asearch(_cfg(), "my name is Atharv", score_threshold=0.95)
        assert len(results) == 1 and results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_text_search(self, pool, table):
        store = _store(pool, table)
        await store.astore(_cfg(), "Running in the mountains")
        await store.astore(_cfg(), "Cooking pasta")
        results = await store.asearch(_cfg(), "run mountain", retrieval_strategy="text")
        assert [r.content for r in results] == ["Running in the mountains"]
        assert 0 < results[0].score < 1

    async def test_hybrid_search(self, pool, table):
        store = _store(pool, table, retrieval_strategy="hybrid")
        await store.astore(_cfg(), "the deploy runbook for kubernetes")
        await store.astore(_cfg(), "grocery list eggs milk")
        results = await store.asearch(_cfg(), "kubernetes runbook")
        assert results[0].content == "the deploy runbook for kubernetes"
        assert 0 < results[0].score <= 1.0

    async def test_metadata_filter(self, pool, table):
        store = _store(pool, table)
        await store.astore(_cfg(), "doc one", metadata={"source": "a"})
        await store.astore(_cfg(), "doc two", metadata={"source": "b"})
        results = await store.asearch(_cfg(), "doc", filters={"source": "b"})
        assert [r.content for r in results] == ["doc two"]


class TestFallbackAndBackfill:
    async def test_fallback_write_then_backfill(self, pool, table):
        down = _store(pool, table, embedding=WordEmbedding(fail=True))
        await down.astore(_cfg(), "remember the alamo")
        nulls = await pool.fetchval(
            f'SELECT count(*) FROM "public"."{table}" WHERE embedding IS NULL'  # noqa: S608
        )
        assert nulls == 1
        # Search falls back to text while the embedding service is down.
        assert len(await down.asearch(_cfg(), "alamo")) == 1

        up = _store(pool, table)
        assert await up.asearch(_cfg(), "remember the alamo", retrieval_strategy="similarity") == []
        assert await up.abackfill_embeddings(batch_size=1) == 1
        assert await up.abackfill_embeddings() == 0
        results = await up.asearch(_cfg(), "remember the alamo", retrieval_strategy="similarity")
        assert len(results) == 1


class TestFailureModes:
    async def test_cancel_during_embedding_leaves_no_row(self, pool, table):
        class Slow(WordEmbedding):
            async def aembed(self, text):
                await asyncio.sleep(10)

        store = _store(pool, table, embedding=Slow())
        await store.asetup()
        task = asyncio.create_task(store.astore(_cfg(), "never saved"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        count_sql = f'SELECT count(*) FROM "public"."{table}"'  # noqa: S608
        assert await pool.fetchval(count_sql) == 0

    async def test_ping_and_release(self, pool, table):
        passed = _store(pool, table)
        await passed.aping()
        await passed.arelease()
        assert await pool.fetchval("SELECT 1") == 1  # caller's pool still open

        owned = PgVectorStore(WordEmbedding(), dsn=POSTGRES_DSN, table=table)
        await owned.aping()
        inner = owned._pool
        await owned.arelease()
        assert inner.is_closing()
