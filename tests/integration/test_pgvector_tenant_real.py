"""PgVectorStore tenant and extra columns against a REAL Postgres + pgvector.

Covers the v1 -> v2 migration with data, every subset of scope columns, cross-tenant denial on
every method, the same user_id in two tenants, shared (system) data inside a tenant, filtered
vector search with ``hnsw.iterative_scan``, and native RLS with a non-superuser role.

    pytest tests/integration/test_pgvector_tenant_real.py --integration

DSN from POSTGRES_DSN. Each test uses its own table (and role, for RLS) and drops it after.
"""

# SQL here is built only from identifiers the fixtures generate (uuid hex), never input.
# ruff: noqa: S608

import asyncio
import itertools
import math
import os
import random
import uuid
import zlib
from urllib.parse import urlsplit, urlunsplit

import pytest
import pytest_asyncio

from tenxgraph.core.authz import SCOPE_NONE, SCOPE_OWNER, build_authz
from tenxgraph.storage.columns import MissingColumnValueError
from tenxgraph.storage.store import PgVectorStore, system_user_id
from tenxgraph.storage.store.store_schema import RetrievalStrategy


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow"
)
DIM = 8
TENANTS = {"org_id": "text", "project_id": "text"}
SCOPE_COLUMNS = ("user_id", "org_id", "project_id")


class WordEmbedding:
    dimension = DIM

    async def aembed(self, text):
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
    name = f"mem_{uuid.uuid4().hex[:12]}"
    yield name
    await pool.execute(f'DROP TABLE IF EXISTS "public"."{name}"')
    await pool.execute(f'DROP TABLE IF EXISTS "public"."{name}_schema_version"')


def _store(pool, table, **kw):
    kw.setdefault("tenant_columns", TENANTS)
    return PgVectorStore(WordEmbedding(), pool=pool, table=table, **kw)


def _cfg(user="alice", org="A", project="P1", scope=None, **user_extra):
    identity = {"user_id": user, "org_id": org, "project_id": project, **user_extra}
    if scope is not None:
        identity["authz"] = build_authz(user, scope=scope)
    return {"user_id": user, "user": identity}


async def _columns(pool, table):
    rows = await pool.fetch(
        "SELECT column_name FROM information_schema.columns WHERE table_name = $1", table
    )
    return {r["column_name"] for r in rows}


class TestSchema:
    async def test_setup_adds_columns_indexes_and_tenant_key(self, pool, table):
        store = _store(pool, table, extra_columns={"public_id": "text"})
        await store.asetup()
        assert await store.aget_schema_version() == 2
        assert {"org_id", "project_id", "public_id"} <= await _columns(pool, table)
        indexes = {
            r["indexname"]: r["indexdef"]
            for r in await pool.fetch(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = $1", table
            )
        }
        for col in ("org_id", "project_id", "public_id"):
            assert f"{table}_c_{col}" in indexes
        key = indexes[f"{table}_key_uq"]
        assert "(collection, user_id, org_id, project_id, memory_type, memory_key)" in key
        assert "NULLS NOT DISTINCT" in key
        await _store(pool, table, extra_columns={"public_id": "text"}).asetup()  # idempotent

    async def test_nothing_declared_adds_nothing(self, pool, table):
        store = PgVectorStore(WordEmbedding(), pool=pool, table=table)
        await store.asetup()
        assert not ({"org_id", "project_id"} & await _columns(pool, table))
        key = await pool.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = $1", f"{table}_key_uq"
        )
        assert "(collection, user_id, memory_type, memory_key)" in key

    async def test_type_mismatch_on_existing_column_raises(self, pool, table):
        await _store(pool, table).asetup()
        with pytest.raises(ValueError, match="already exists"):
            await _store(pool, table, tenant_columns={"org_id": "uuid"}).asetup()

    async def test_migration_v1_to_v2_with_data(self, pool, table):
        old = PgVectorStore(WordEmbedding(), pool=pool, table=table)
        await old.asetup()
        legacy = await old.astore({"user_id": "alice"}, "legacy fact", memory_key="k")
        await pool.execute(f'DELETE FROM "public"."{table}_schema_version"')
        await pool.execute(f'INSERT INTO "public"."{table}_schema_version" (version) VALUES (1)')

        store = _store(pool, table)
        await store.asetup()
        assert await store.aget_schema_version() == 2
        assert {"org_id", "project_id"} <= await _columns(pool, table)
        # Old rows hold NULL tenant values: invisible to scopes that pick tenant columns...
        assert await store.aget(_cfg(), legacy) is None
        assert await store.aget_all(_cfg()) == []
        # ...and visible to a scope that does not.
        got = await store.aget(_cfg(scope=["user_id"]), legacy)
        assert got is not None and got.content == "legacy fact"
        # The key index was rebuilt to include the tenant columns.
        key = await pool.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = $1", f"{table}_key_uq"
        )
        assert "org_id, project_id" in key
        new_id = await store.astore(_cfg(), "tenant fact", memory_key="k")
        assert new_id != legacy


class TestTenantIsolation:
    async def test_same_user_id_in_two_tenants(self, pool, table):
        store = _store(pool, table)
        a, b = _cfg(org="A"), _cfg(org="B")
        a_key = await store.astore(a, "alice likes tea", memory_key="drink")
        b_key = await store.astore(b, "alice likes coffee", memory_key="drink")
        assert a_key != b_key  # same user_id + key, two tenants: two rows
        again = await store.astore(a, "alice likes green tea", memory_key="drink")
        assert again == a_key  # keyed upsert stays inside the tenant
        assert [m.content for m in await store.aget_all(a)] == ["alice likes green tea"]
        assert [m.content for m in await store.aget_all(b)] == ["alice likes coffee"]

    async def test_cross_tenant_denied_on_every_method(self, pool, table):
        store = _store(pool, table)
        a, b = _cfg(org="A"), _cfg(org="B")
        mem = await store.astore(a, "secret plan for apples")
        # read
        assert await store.aget(b, mem) is None
        assert await store.aget_all(b) == []
        for strategy in (
            RetrievalStrategy.SIMILARITY,
            RetrievalStrategy.TEXT,
            RetrievalStrategy.HYBRID,
        ):
            assert await store.asearch(b, "apples", retrieval_strategy=strategy) == []
        # update / delete
        with pytest.raises(ValueError, match="not found"):
            await store.aupdate(b, mem, "hijacked")
        with pytest.raises(ValueError, match="not found"):
            await store.adelete(b, mem)
        # overwrite by id
        with pytest.raises(PermissionError):
            await store.astore(b, "hijacked", memory_id=mem)
        # forget
        await store.aforget_memory(b)
        got = await store.aget(a, mem)
        assert got is not None and got.content == "secret plan for apples"
        for strategy in (
            RetrievalStrategy.SIMILARITY,
            RetrievalStrategy.TEXT,
            RetrievalStrategy.HYBRID,
        ):
            found = await store.asearch(a, "apples", retrieval_strategy=strategy)
            assert [r.id for r in found] == [mem]
        await store.aforget_memory(a)
        assert await store.aget(a, mem) is None

    async def test_every_subset_of_scope_columns(self, pool, table):
        store = _store(pool, table)
        seeds = [
            ("alice", "A", "P1"),
            ("bob", "A", "P1"),
            ("alice", "A", "P2"),
            ("alice", "B", "P1"),
            ("carol", "B", "P3"),
        ]
        ids = {}
        for user, org, project in seeds:
            ids[(user, org, project)] = await store.astore(
                _cfg(user, org, project), f"note {user} {org} {project}"
            )
        caller = {"user_id": "alice", "org_id": "A", "project_id": "P1"}
        for size in range(len(SCOPE_COLUMNS) + 1):
            for scope in itertools.combinations(SCOPE_COLUMNS, size):
                cfg = _cfg("alice", "A", "P1", scope=list(scope))
                want = {
                    ids[seed]
                    for seed in seeds
                    if all(
                        dict(zip(SCOPE_COLUMNS, seed, strict=True))[c] == caller[c] for c in scope
                    )
                }
                got = {m.id for m in await store.aget_all(cfg)}
                assert got == want, scope
                found = await store.asearch(
                    cfg, "note", retrieval_strategy=RetrievalStrategy.TEXT, limit=10
                )
                assert {r.id for r in found} == want, scope

    async def test_owner_and_none_shortcuts(self, pool, table):
        store = _store(pool, table)
        await store.astore(_cfg("alice", "A", "P1"), "one")
        await store.astore(_cfg("bob", "B", "P2"), "two")
        owner = await store.aget_all(_cfg("alice", "A", "P1", scope=SCOPE_OWNER))
        assert [m.content for m in owner] == ["one"]
        everyone = await store.aget_all(_cfg("alice", "A", "P1", scope=SCOPE_NONE))
        assert {m.content for m in everyone} == {"one", "two"}

    async def test_missing_value_for_picked_column_raises(self, pool, table):
        store = _store(pool, table)
        no_org = {"user_id": "alice", "user": {"user_id": "alice", "project_id": "P1"}}
        with pytest.raises(MissingColumnValueError):
            await store.astore(no_org, "x")
        with pytest.raises(MissingColumnValueError):
            await store.aget_all(no_org)
        # A scope that does not pick org_id reads fine.
        no_org["user"]["authz"] = build_authz("alice", scope=["user_id", "project_id"])
        assert await store.aget_all(no_org) == []

    async def test_optional_tenant_column(self, pool, table):
        store = _store(
            pool,
            table,
            tenant_columns={"org_id": "text", "team": {"type": "text", "required": False}},
        )
        cfg = {"user_id": "alice", "user": {"user_id": "alice", "org_id": "A"}}
        first = await store.astore(cfg, "v1", memory_key="k")
        # NULLS NOT DISTINCT: a keyed write with the optional column NULL still dedupes.
        assert await store.astore(cfg, "v2", memory_key="k") == first
        with pytest.raises(MissingColumnValueError):
            await store.aget_all(cfg)  # the default scope picks team, which has no value
        cfg["user"]["authz"] = build_authz("alice", scope=["user_id", "org_id"])
        assert [m.content for m in await store.aget_all(cfg)] == ["v2"]

    async def test_extra_columns_stored_required_and_updated(self, pool, table):
        store = _store(
            pool,
            table,
            extra_columns={"public_id": "text", "plan": {"type": "text", "required": True}},
        )
        with pytest.raises(MissingColumnValueError, match="plan"):
            await store.astore(_cfg(), "x")
        mem = await store.astore(_cfg(plan="pro", public_id="pub-1"), "x")
        row = await pool.fetchrow(
            f'SELECT org_id, project_id, public_id, plan FROM "public"."{table}" WHERE id = $1', mem
        )
        assert dict(row) == {"org_id": "A", "project_id": "P1", "public_id": "pub-1", "plan": "pro"}
        await store.aupdate(_cfg(plan="team"), mem, "y")
        row = await pool.fetchrow(
            f'SELECT org_id, public_id, plan FROM "public"."{table}" WHERE id = $1', mem
        )
        assert dict(row) == {"org_id": "A", "public_id": "pub-1", "plan": "team"}

    async def test_typed_columns(self, pool, table):
        store = _store(pool, table, tenant_columns={"org_id": "uuid", "tier": "integer"})
        org = uuid.uuid4()
        cfg = {"user_id": "alice", "user": {"user_id": "alice", "org_id": str(org), "tier": "3"}}
        mem = await store.astore(cfg, "typed")
        assert (await store.aget(cfg, mem)).content == "typed"
        other = {
            "user_id": "alice",
            "user": {"user_id": "alice", "org_id": str(uuid.uuid4()), "tier": 3},
        }
        assert await store.aget(other, mem) is None

    async def test_system_user_data_is_shared_inside_the_tenant_only(self, pool, table):
        store = _store(pool, table)
        kb = system_user_id("knowledge_base")

        def shared(user, org):
            cfg = _cfg(user, org)
            cfg["user_id"] = kb
            return cfg

        await store.astore(shared("alice", "A"), "refund policy for org A")
        assert len(await store.aget_all(shared("bob", "A"))) == 1  # same tenant, other user
        assert await store.aget_all(shared("alice", "B")) == []  # other tenant

    async def test_unknown_scope_column_raises(self, pool, table):
        store = _store(pool, table)
        with pytest.raises(ValueError, match="does not declare"):
            await store.aget_all(_cfg(scope=["user_id", "region"]))


class _FixedEmbedding:
    dimension = DIM

    async def aembed(self, text):
        return [1.0] + [0.0] * (DIM - 1)

    async def aembed_query(self, text):
        return await self.aembed(text)


class TestIterativeScan:
    async def test_filtered_vector_search_returns_limit_rows(self, pool, table):
        """Tenant A is 1 row in 40, spread through the vector space. Without an iterative
        scan, HNSW hands back its ``ef_search`` (40) nearest candidates and the tenant filter
        keeps about one; with ``hnsw.iterative_scan`` the scan continues to ``limit`` rows.

        The B-tree indexes are dropped and seq scans disabled for this pool so the planner
        must use the HNSW index (with them it picks the B-tree for so few tenant rows, which
        is exact anyway)."""
        asyncpg = pytest.importorskip("asyncpg")
        hnsw_pool = await asyncpg.create_pool(
            POSTGRES_DSN, min_size=1, max_size=2, server_settings={"enable_seqscan": "off"}
        )
        try:
            store = PgVectorStore(
                _FixedEmbedding(), pool=hnsw_pool, table=table, tenant_columns=TENANTS
            )
            await store.asetup()
            assert store._iterative_scan is True
            for suffix in ("uid_idx", "c_org_id", "c_project_id"):
                await pool.execute(f'DROP INDEX "public"."{table}_{suffix}"')
            rng = random.Random(7)  # noqa: S311 - test data
            rows = []
            for i in range(3000):
                vector = "[" + ",".join(str(rng.gauss(0, 1)) for _ in range(DIM)) + "]"
                rows.append((f"n{i}", "A" if i % 40 == 0 else "B", vector, store.collection))
            await pool.executemany(
                f'INSERT INTO "public"."{table}" (id, collection, user_id, memory_type, '
                "content, embedding, org_id, project_id) VALUES ($1, $4, 'alice', 'episodic', "
                "'text', $3::text::vector, $2, 'P1')",
                rows,
            )
            await pool.execute(f'ANALYZE "public"."{table}"')
            found = await store.asearch(_cfg(org="A"), "q", limit=5)
            assert len(found) == 5
            assert {r.user_id for r in found} == {"alice"}
            store._iterative_scan = False  # what a pgvector < 0.8 server gets
            assert len(await store.asearch(_cfg(org="A"), "q", limit=5)) < 5
        finally:
            await hnsw_pool.close()


# ---------------------------------------------------------------------------
# Native RLS with a non-superuser role
# ---------------------------------------------------------------------------


def _dsn_for(role: str, password: str) -> str:
    parts = urlsplit(POSTGRES_DSN)
    host = parts.hostname or "localhost"
    netloc = f"{role}:{password}@{host}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


@pytest_asyncio.fixture
async def rls_role(pool):
    asyncpg = pytest.importorskip("asyncpg")
    role = f"tg_rls_{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex
    await pool.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{password}' NOSUPERUSER NOBYPASSRLS")
    await pool.execute(f"GRANT CREATE, USAGE ON SCHEMA public TO {role}")
    role_pool = await asyncpg.create_pool(_dsn_for(role, password), min_size=1, max_size=4)
    yield role, role_pool
    await role_pool.close()
    await pool.execute(f"DROP OWNED BY {role}")
    await pool.execute(f"DROP ROLE {role}")


class TestRowLevelSecurity:
    async def test_policy_denies_without_settings_and_filters_by_tenant(
        self, pool, table, rls_role
    ):
        _role, role_pool = rls_role
        store = _store(role_pool, table, row_level_security=True)
        a_mem = await store.astore(_cfg(org="A"), "org A data about apples")
        b_mem = await store.astore(_cfg(org="B"), "org B data about apples")
        assert [m.id for m in await store.aget_all(_cfg(org="A"))] == [a_mem]
        found = await store.asearch(_cfg(org="B"), "apples")
        assert [r.id for r in found] == [b_mem]

        flags = await pool.fetchrow(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = $1", table
        )
        assert flags["relrowsecurity"] and flags["relforcerowsecurity"]

        qualified = f'"public"."{table}"'
        async with role_pool.acquire() as conn:
            # A raw query that skips the store (no WHERE, no settings) sees nothing.
            assert await conn.fetchval(f"SELECT count(*) FROM {qualified}") == 0
            # The policy alone filters by tenant, even with no WHERE clause.
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('tenxgraph.scope', '+,org_id', true), "
                    "set_config('tenxgraph.org_id', 'A', true)"
                )
                ids = [r["id"] for r in await conn.fetch(f"SELECT id FROM {qualified}")]
                assert ids == [a_mem]
                # Writing a row of another tenant is refused by WITH CHECK.
                with pytest.raises(Exception, match="row-level security"):
                    await conn.execute(
                        f"UPDATE {qualified} SET org_id = 'B' WHERE id = $1",
                        a_mem,
                    )
            # The settings ended with the transaction: denied again on the same connection.
            assert await conn.fetchval(f"SELECT count(*) FROM {qualified}") == 0
        # The superuser test role bypasses RLS and sees both rows.
        assert await pool.fetchval(f"SELECT count(*) FROM {qualified}") == 2

    async def test_cross_tenant_overwrite_by_id_is_permission_error(self, pool, table, rls_role):
        _role, role_pool = rls_role
        store = _store(role_pool, table, row_level_security=True)
        mem = await store.astore(_cfg(org="A"), "mine")
        with pytest.raises(PermissionError):
            await store.astore(_cfg(org="B"), "hijack", memory_id=mem)
        assert (await store.aget(_cfg(org="A"), mem)).content == "mine"

    async def test_scope_without_tenant_columns_sees_all_and_backfill_runs(
        self, pool, table, rls_role
    ):
        _role, role_pool = rls_role
        store = _store(role_pool, table, row_level_security=True)
        await store.astore(_cfg(org="A"), "one")
        await store.astore(_cfg(org="B"), "two")
        admin = _cfg(org="A", scope=["user_id"])
        assert {m.content for m in await store.aget_all(admin)} == {"one", "two"}
        await pool.execute(f'UPDATE "public"."{table}" SET embedding = NULL')
        assert await store.abackfill_embeddings() == 2

    async def test_turning_rls_off_removes_the_policy(self, pool, table, rls_role):
        _role, role_pool = rls_role
        await _store(role_pool, table, row_level_security=True).asetup()
        await _store(role_pool, table, row_level_security=False).asetup()
        flags = await pool.fetchrow(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = $1", table
        )
        assert not flags["relrowsecurity"] and not flags["relforcerowsecurity"]
        assert not await pool.fetchval(
            "SELECT count(*) FROM pg_policies WHERE tablename = $1", table
        )
