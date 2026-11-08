"""Sprint 7 review fixes in PgVectorStore against a REAL Postgres + pgvector.

S2: declaring tenant columns on a live table does not block reads, keeps the old key index for
processes on the previous version, honours ``ddl_lock_timeout``, and rebuilds an invalid index
left by a failed or cancelled concurrent build. S3: declared columns named like the SQL's own
aliases. S4: uppercase table names with declared columns. S5: zero vectors. S6: setup with
everything in place does not wait behind open writes.

    POSTGRES_DSN=postgresql://test:test@localhost:55432/test_agentflow \\
        pytest tests/integration/test_store_review_fixes_real.py --integration

Each test uses its own table and drops it after.
"""

# SQL here is built only from identifiers the fixtures generate (uuid hex), never input.
# ruff: noqa: S608

import asyncio
import contextlib
import math
import os
import time
import uuid
import zlib

import pytest
import pytest_asyncio

from tenxgraph.storage.columns import SetupLockTimeoutError
from tenxgraph.storage.store import PgVectorStore
from tenxgraph.storage.store.store_schema import RetrievalStrategy


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow"
)
DIM = 8


class WordEmbedding:
    dimension = DIM

    def __init__(self, zero=False):
        self.zero = zero

    async def aembed(self, text):
        if self.zero:
            return [0.0] * DIM
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
    except Exception as e:
        pytest.skip(f"Postgres not reachable at POSTGRES_DSN: {e}")
    if not await pg_pool.fetchval(
        "SELECT count(*) FROM pg_available_extensions WHERE name = 'vector'"
    ):
        await pg_pool.close()
        pytest.skip("pgvector extension not available on this server")
    await pg_pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    yield pg_pool
    await pg_pool.close()


@pytest_asyncio.fixture
async def table(pool):
    name = f"rf_{uuid.uuid4().hex[:12]}"
    yield name
    await pool.execute(f'DROP TABLE IF EXISTS "public"."{name}" CASCADE')
    await pool.execute(f'DROP TABLE IF EXISTS "public"."{name}_schema_version"')


def _cfg(user="alice", org="A"):
    return {"user_id": user, "user": {"user_id": user, "tenant": {"org_id": org}}}


async def _connect():
    asyncpg = pytest.importorskip("asyncpg")
    return await asyncpg.connect(POSTGRES_DSN)


async def _indexes(pool, table):
    rows = await pool.fetch(
        "SELECT c.relname, i.indisvalid, pg_get_indexdef(i.indexrelid) AS def "
        "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE i.indrelid = to_regclass($1)",
        f'"public"."{table}"',
    )
    return {r["relname"]: (r["indisvalid"], r["def"]) for r in rows}


async def _fill(pool, table, rows):
    await pool.execute(
        f"""INSERT INTO "public"."{table}" (id, collection, user_id, memory_type, memory_key,
        content) SELECT g::text, 'default', 'u' || (g % 1000), 'semantic', 'k' || g,
        'some content ' || g FROM generate_series(1, {int(rows)}) g"""
    )


class TestS2LiveTable:
    async def test_reads_go_on_while_tenant_columns_are_added(self, pool, table):
        await PgVectorStore(None, pool=pool, table=table).asetup()
        await _fill(pool, table, 600_000)
        new = PgVectorStore(
            None,
            pool=pool,
            table=table,
            tenant_columns={"org_id": {"type": "text", "required": False}},
        )
        reader = await _connect()
        try:
            started = time.monotonic()
            setup = asyncio.create_task(new.asetup())
            latencies = []
            during = 0
            await asyncio.sleep(0.2)
            while not setup.done():
                t0 = time.monotonic()
                await reader.fetchval(
                    f'SELECT count(*) FROM "public"."{table}" WHERE collection = $1 '
                    "AND user_id = $2",
                    "default",
                    "u1",
                )
                latencies.append(time.monotonic() - t0)
                if not setup.done():
                    during += 1
                await asyncio.sleep(0.05)
            await setup
            took = time.monotonic() - started
        finally:
            await reader.close()
        # Before the fix the whole setup held ACCESS EXCLUSIVE (ALTER TABLE + DROP INDEX +
        # CREATE UNIQUE INDEX in one transaction): the first read waited for the full rebuild
        # (about 0.9s of a 1.3s setup here). Now reads run while the indexes build.
        assert during >= 1, "setup finished before any read could run; make the table bigger"
        assert max(latencies) < max(0.3, took * 0.25), (took, latencies)
        indexes = await _indexes(pool, table)
        assert all(valid for valid, _ in indexes.values()), indexes

    async def test_old_version_keyed_writes_keep_working(self, pool, table):
        old = PgVectorStore(WordEmbedding(), pool=pool, table=table)
        await old.asetup()
        first = await old.astore({"user_id": "bob"}, "likes tea", memory_key="drink")
        new = PgVectorStore(WordEmbedding(), pool=pool, table=table, tenant_columns=["org_id"])
        await new.asetup()
        # A process still on the previous version upserts on the old key index.
        assert await old.astore({"user_id": "bob"}, "likes green tea", memory_key="drink") == (
            first
        )
        assert new._legacy_key_indexes == [f"{table}_key_uq"]
        # New-version keyed writes work for keys the legacy rows do not hold.
        a = await new.astore(_cfg("bob", "A"), "bob@A", memory_key="food")
        assert await new.astore(_cfg("bob", "A"), "bob@A v2", memory_key="food") == a
        assert await new.adrop_legacy_key_index() == [f"{table}_key_uq"]
        b = await new.astore(_cfg("bob", "B"), "bob@B", memory_key="food")
        assert b != a

    async def test_lock_timeout_raises_clear_error(self, pool, table):
        await PgVectorStore(None, pool=pool, table=table).asetup()
        holder = await _connect()
        try:
            tr = holder.transaction()
            await tr.start()
            await holder.execute(
                f'INSERT INTO "public"."{table}" (id, collection, user_id, memory_type, '
                "content) VALUES ('x', 'default', 'u', 'episodic', 'c')"
            )
            new = PgVectorStore(
                None, pool=pool, table=table, tenant_columns=["org_id"], ddl_lock_timeout=0.5
            )
            t0 = time.monotonic()
            with pytest.raises(SetupLockTimeoutError, match="ddl_lock_timeout"):
                await asyncio.wait_for(new.asetup(), 10)
            assert time.monotonic() - t0 < 5
            await tr.rollback()
        finally:
            await holder.close()
        # Nothing half-done: the column was not added, and the next setup succeeds.
        await PgVectorStore(None, pool=pool, table=table, tenant_columns=["org_id"]).asetup()

    async def test_invalid_leftover_index_is_rebuilt(self, pool, table):
        await PgVectorStore(None, pool=pool, table=table).asetup()
        await _fill(pool, table, 50)
        # A concurrent build that failed (here on duplicates) leaves an invalid index behind
        # under the name setup wants. CREATE INDEX IF NOT EXISTS would keep it forever.
        conn = await _connect()
        try:
            await conn.execute(f'ALTER TABLE "public"."{table}" ADD COLUMN org_id TEXT')
            with pytest.raises(Exception, match="could not create unique index"):
                await conn.execute(
                    f'CREATE UNIQUE INDEX CONCURRENTLY "{table}_c_org_id" '
                    f'ON "public"."{table}" (collection)'
                )
        finally:
            await conn.close()
        assert (await _indexes(pool, table))[f"{table}_c_org_id"][0] is False
        await PgVectorStore(
            None,
            pool=pool,
            table=table,
            tenant_columns={"org_id": {"type": "text", "required": False}},
        ).asetup()
        valid, definition = (await _indexes(pool, table))[f"{table}_c_org_id"]
        assert valid and "UNIQUE" not in definition and "(org_id)" in definition

    async def test_cancelled_setup_releases_lock_and_next_setup_heals(self, pool, table):
        await PgVectorStore(None, pool=pool, table=table).asetup()
        await _fill(pool, table, 400_000)
        new = PgVectorStore(
            None,
            pool=pool,
            table=table,
            tenant_columns={"org_id": {"type": "text", "required": False}},
        )
        task = asyncio.create_task(new.asetup())
        await asyncio.sleep(0.4)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # The advisory lock is free again (the session lock went with the cancelled call).
        again = PgVectorStore(
            None,
            pool=pool,
            table=table,
            tenant_columns={"org_id": {"type": "text", "required": False}},
        )
        await asyncio.wait_for(again.asetup(), 60)
        indexes = await _indexes(pool, table)
        assert all(valid for valid, _ in indexes.values()), indexes
        assert any(name.startswith(f"{table}_kuq_") for name in indexes)

    async def test_concurrent_setups_serialise(self, pool, table):
        stores = [
            PgVectorStore(WordEmbedding(), pool=pool, table=table, tenant_columns=["org_id"])
            for _ in range(4)
        ]
        await asyncio.gather(*(s.asetup() for s in stores))
        indexes = await _indexes(pool, table)
        assert all(valid for valid, _ in indexes.values())
        assert len([n for n in indexes if n.startswith(f"{table}_kuq_")]) == 1


class TestS3ColumnNames:
    @pytest.mark.parametrize(
        "strategy",
        [RetrievalStrategy.SIMILARITY, RetrievalStrategy.TEXT, RetrievalStrategy.HYBRID],
    )
    async def test_columns_named_like_sql_aliases(self, pool, table, strategy):
        names = ("tsq", "q", "score", "similarity", "rnk", "rrf", "m", "f", "vec", "txt")
        store = PgVectorStore(
            WordEmbedding(),
            pool=pool,
            table=table,
            tenant_columns=["org_id"],
            extra_columns=dict.fromkeys(names, "text"),
        )
        cfg = _cfg()
        cfg["user"].update({name: f"v_{name}" for name in names})
        await store.astore(cfg, "hello world apples")
        results = await store.asearch(
            _cfg(), "apples", retrieval_strategy=strategy, filters={"tsq": "v_tsq", "q": "v_q"}
        )
        assert [r.content for r in results] == ["hello world apples"]


class TestS4UppercaseTable:
    async def test_setup_and_use(self, pool):
        name = f"Mem_{uuid.uuid4().hex[:8]}"
        try:
            store = PgVectorStore(WordEmbedding(), pool=pool, table=name, tenant_columns=["org_id"])
            await store.asetup()
            mid = await store.astore(_cfg(), "apples", memory_key="k")
            assert [r.id for r in await store.asearch(_cfg(), "apples")] == [mid]
            indexes = await _indexes(pool, name)
            assert f"{name}_c_org_id" in indexes and all(v for v, _ in indexes.values())
        finally:
            await pool.execute(f'DROP TABLE IF EXISTS "public"."{name}" CASCADE')
            await pool.execute(f'DROP TABLE IF EXISTS "public"."{name}_schema_version"')


class TestS5ZeroVectors:
    async def test_stored_zero_vector_never_passes_threshold(self, pool, table):
        store = PgVectorStore(WordEmbedding(), pool=pool, table=table)
        await store.asetup()
        good = await store.astore({"user_id": "u"}, "apples")
        # A zero vector stored before the fix (written directly).
        await pool.execute(
            f'INSERT INTO "public"."{table}" (id, collection, user_id, memory_type, content, '
            "embedding) VALUES ('z', $1, 'u', 'episodic', 'zero', $2::text::vector)",
            store.collection,
            "[" + ",".join(["0"] * DIM) + "]",
        )
        for strategy in (RetrievalStrategy.SIMILARITY, RetrievalStrategy.HYBRID):
            results = await store.asearch(
                {"user_id": "u"}, "apples", retrieval_strategy=strategy, score_threshold=0.1
            )
            assert [r.id for r in results] == [good], strategy

    async def test_zero_vector_write_refused_and_query_falls_back(self, pool, table):
        store = PgVectorStore(WordEmbedding(zero=True), pool=pool, table=table)
        with pytest.raises(ValueError, match="zero vector"):
            await store.astore({"user_id": "u"}, "apples")
        text_store = PgVectorStore(WordEmbedding(), pool=pool, table=table)
        mid = await text_store.astore({"user_id": "u"}, "apples")
        results = await store.asearch({"user_id": "u"}, "apples", score_threshold=0.0)
        assert [r.id for r in results] == [mid]  # TEXT fallback, not a NaN vector search
        strict = PgVectorStore(
            WordEmbedding(zero=True), pool=pool, table=table, on_embedding_error="raise"
        )
        with pytest.raises(ValueError, match="zero vector"):
            await strict.asearch({"user_id": "u"}, "apples")


class TestS6SetupDoesNotWait:
    async def test_second_setup_with_open_write(self, pool, table):
        await PgVectorStore(None, pool=pool, table=table, tenant_columns=["org_id"]).asetup()
        holder = await _connect()
        try:
            tr = holder.transaction()
            await tr.start()
            await holder.execute(
                f'INSERT INTO "public"."{table}" (id, collection, user_id, memory_type, '
                "content, org_id) VALUES ('x', 'default', 'u', 'episodic', 'c', 'A')"
            )
            again = PgVectorStore(None, pool=pool, table=table, tenant_columns=["org_id"])
            await asyncio.wait_for(again.asetup(), 3)
            await tr.rollback()
        finally:
            await holder.close()
