"""Unit tests for PgVectorStore with a fake asyncpg pool (no database).

These check the SQL each method sends, the parameters, owner rules, settings precedence,
embedding fallback, text-only mode and pool ownership. The SQL itself is exercised against a
real pgvector database in tests/integration/test_pgvector_store_real.py.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.tools.memory import (
    _memory_scope_retrieval_strategy,
    _search_scope_memory,
)
from tenxgraph.storage import PgVectorStore as FromStorage
from tenxgraph.storage.store import MemoryConfig, UserMemoryConfig, system_user_id
from tenxgraph.storage.store import PgVectorStore as FromStore
from tenxgraph.storage.store.long_term_memory import _do_write
from tenxgraph.storage.store.pgvector_store import (
    CURRENT_SCHEMA_VERSION,
    PgVectorStore,
    _quote_ident,
)
from tenxgraph.storage.store.store_schema import DistanceMetric, MemoryType, RetrievalStrategy


CFG = {"user_id": "u1", "thread_id": "t1"}
ADMIN = {"user": {"user_id": "admin", "reach": "all"}}


class _Embedding:
    dimension = 4

    def __init__(self, fail=False, vector=None):
        self.fail = fail
        self.vector = vector or [0.1, 0.2, 0.3, 0.4]
        self.aembed_batch = AsyncMock(side_effect=lambda texts: [self.vector for _ in texts])

    async def aembed(self, text):
        if self.fail:
            raise RuntimeError("embedding service down")
        return self.vector

    async def aembed_query(self, text):
        return await self.aembed(text)


class FakeConn:
    """Records every call; answers from per-method queues or a ``fetchval`` responder."""

    def __init__(self):
        self.calls: list[tuple[str, str, tuple]] = []
        self.fetchrow_results: list = []
        self.fetch_results: list = []
        self.execute_results: list = []
        self.fetchval_responder = lambda sql, args: None
        self.transactions = 0

    def _record(self, method, sql, args):
        self.calls.append((method, " ".join(sql.split()), args))

    async def fetchrow(self, sql, *args):
        self._record("fetchrow", sql, args)
        result = self.fetchrow_results.pop(0) if self.fetchrow_results else None
        if isinstance(result, BaseException):
            raise result
        return result

    async def fetch(self, sql, *args):
        self._record("fetch", sql, args)
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def fetchval(self, sql, *args):
        self._record("fetchval", sql, args)
        return self.fetchval_responder(sql, args)

    async def execute(self, sql, *args):
        self._record("execute", sql, args)
        result = self.execute_results.pop(0) if self.execute_results else "OK"
        if isinstance(result, BaseException):
            raise result
        return result

    @asynccontextmanager
    async def _transaction(self):
        self.transactions += 1
        yield

    def transaction(self):
        return self._transaction()

    def sql(self, method=None):
        return [c[1] for c in self.calls if method is None or c[0] == method]


class FakePool:
    def __init__(self, conn):
        self.conn = conn
        self.close = AsyncMock()
        self.acquire_errors: list = []
        self.acquire_timeouts: list = []

    def is_closing(self):
        return False

    @asynccontextmanager
    async def _acquire(self):
        if self.acquire_errors:
            raise self.acquire_errors.pop(0)
        yield self.conn

    def acquire(self, timeout=None):
        self.acquire_timeouts.append(timeout)
        return self._acquire()


def _row(**overrides):
    row = {
        "id": "m1",
        "collection": "agentflow_memories",
        "user_id": "u1",
        "thread_id": None,
        "memory_key": None,
        "memory_type": "episodic",
        "category": "general",
        "content": "hello",
        "metadata": json.dumps({"source": "chat"}),
        "created_at": None,
        "updated_at": None,
        "score": 0.9,
    }
    row.update(overrides)
    return row


def _store(embedding="default", ready=True, **kw):
    conn = FakeConn()
    pool = FakePool(conn)
    emb = _Embedding() if embedding == "default" else embedding
    store = PgVectorStore(emb, pool=pool, **kw)
    if ready:
        store._setup_done = True
        store._index_metric = store.distance_metric
    return store, conn, pool


# ---------------------------------------------------------------------------
# Constructor and identifier validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_needs_dsn_or_pool(self):
        with pytest.raises(ValueError, match="dsn or an asyncpg pool"):
            PgVectorStore(_Embedding())

    def test_rejects_dsn_and_pool(self):
        with pytest.raises(ValueError, match="not both"):
            PgVectorStore(_Embedding(), dsn="postgresql://x/y", pool=FakePool(FakeConn()))

    @pytest.mark.parametrize(
        "table",
        ['mem"; DROP TABLE x; --', "1table", "a-b", "a.b", "", "x" * 49, "té", "a b"],
    )
    def test_rejects_bad_table_names(self, table):
        with pytest.raises(ValueError, match="Invalid table name"):
            PgVectorStore(_Embedding(), pool=FakePool(FakeConn()), table=table)

    @pytest.mark.parametrize("schema", ['public"', "pub lic", "", "x;y"])
    def test_rejects_bad_schema_names(self, schema):
        with pytest.raises(ValueError, match="Invalid schema name"):
            PgVectorStore(_Embedding(), pool=FakePool(FakeConn()), schema=schema)

    @pytest.mark.parametrize("cfg", ["english'", "English", "a;b", ""])
    def test_rejects_bad_text_search_config(self, cfg):
        with pytest.raises(ValueError, match="text_search_config"):
            PgVectorStore(_Embedding(), pool=FakePool(FakeConn()), text_search_config=cfg)

    def test_rejects_bad_on_embedding_error(self):
        with pytest.raises(ValueError, match="on_embedding_error"):
            PgVectorStore(_Embedding(), pool=FakePool(FakeConn()), on_embedding_error="ignore")

    def test_accepts_48_char_table(self):
        store, _, _ = _store(table="t" * 48)
        assert store._qualified_table == f'"public"."{"t" * 48}"'

    def test_default_strategy_with_embedding_is_similarity(self):
        store, _, _ = _store()
        assert store.retrieval_strategy == RetrievalStrategy.SIMILARITY
        assert store.distance_metric == DistanceMetric.COSINE

    def test_default_strategy_text_only_is_text(self):
        store, _, _ = _store(embedding=None)
        assert store.retrieval_strategy == RetrievalStrategy.TEXT

    def test_text_only_refuses_vector_strategies(self):
        with pytest.raises(ValueError, match="text-only"):
            _store(embedding=None, retrieval_strategy="hybrid")

    def test_unsupported_strategy(self):
        with pytest.raises(ValueError, match="not supported by PgVectorStore"):
            _store(retrieval_strategy=RetrievalStrategy.TEMPORAL)

    @pytest.mark.parametrize("weights", [(0, 0), (-1, 1), (1,), ("a", 1), (float("inf"), 1)])
    def test_rejects_bad_weights(self, weights):
        with pytest.raises(ValueError, match="hybrid_weights"):
            _store(hybrid_weights=weights)

    @pytest.mark.parametrize("k", [0, -1, 1.5, True])
    def test_rejects_bad_rrf_k(self, k):
        with pytest.raises(ValueError, match="rrf_k"):
            _store(rrf_k=k)

    def test_rejects_bad_hybrid_candidates(self):
        with pytest.raises(ValueError, match="hybrid_candidates"):
            _store(hybrid_candidates=0)

    def test_supports_memory_key_upsert(self):
        store, _, _ = _store()
        assert store.supports_memory_key_upsert is True

    def test_lock_key_is_stable_and_per_table(self):
        a, _, _ = _store()
        b, _, _ = _store()
        c, _, _ = _store(table="other")
        assert a._setup_lock_key() == b._setup_lock_key()
        assert a._setup_lock_key() != c._setup_lock_key()
        assert -(2**63) <= a._setup_lock_key() < 2**63

    def test_exported(self):
        assert FromStore is PgVectorStore
        assert FromStorage is PgVectorStore


class TestQuoteIdent:
    def test_quotes_valid(self):
        assert _quote_ident("user_id") == '"user_id"'

    @pytest.mark.parametrize("name", ['a"b', "a b", "1a", "", None, "a;b", "x" * 64])
    def test_rejects_invalid(self, name):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            _quote_ident(name)


# ---------------------------------------------------------------------------
# asetup
# ---------------------------------------------------------------------------


def _setup_responder(extversion="0.8.0", column="vector(4)", opclass=None, version=None):
    def respond(sql, args):
        if "extversion" in sql:
            return extversion
        if "format_type" in sql:
            return column
        if "opcname" in sql:
            return opclass
        if "max(version)" in sql:
            return version
        return None

    return respond


@pytest.mark.asyncio
class TestSetup:
    async def test_creates_table_indexes_and_records_version(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder()
        assert await store.asetup() is True
        executed = conn.sql("execute")
        assert conn.transactions == 1
        # Session-level lock: it also covers CREATE INDEX CONCURRENTLY outside the transaction.
        assert executed[0].startswith("SELECT pg_advisory_lock($1)")
        assert executed[-1].startswith("SELECT pg_advisory_unlock($1)")
        create = next(
            s
            for s in executed
            if s.startswith('CREATE TABLE IF NOT EXISTS "public"."tenxgraph_memories" (')
        )
        assert "embedding vector(4)" in create
        assert (
            "tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, content)) STORED"
            in create
        )
        assert "PRIMARY KEY (collection, id)" in create
        assert any("tenxgraph_memories_schema_version" in s for s in executed)
        assert any(
            '"tenxgraph_memories_uid_idx"' in s and "(collection, user_id)" in s for s in executed
        )
        assert any(
            "CREATE UNIQUE INDEX" in s
            and "(collection, user_id, memory_type, memory_key) WHERE memory_key IS NOT NULL" in s
            for s in executed
        )
        assert any("USING gin (tsv)" in s for s in executed)
        assert any("USING hnsw (embedding vector_cosine_ops)" in s for s in executed)
        assert not any("CREATE EXTENSION" in s for s in executed)
        insert = [c for c in conn.calls if c[1].startswith("INSERT INTO")]
        assert insert[0][2] == (CURRENT_SCHEMA_VERSION,)
        assert store._index_metric == DistanceMetric.COSINE

    @pytest.mark.parametrize(
        ("metric", "opclass"),
        [
            (DistanceMetric.EUCLIDEAN, "vector_l2_ops"),
            (DistanceMetric.DOT_PRODUCT, "vector_ip_ops"),
            (DistanceMetric.MANHATTAN, "vector_l1_ops"),
        ],
    )
    async def test_index_opclass_follows_metric(self, metric, opclass):
        store, conn, _ = _store(ready=False, distance_metric=metric)
        conn.fetchval_responder = _setup_responder()
        await store.asetup()
        assert any(f"USING hnsw (embedding {opclass})" in s for s in conn.sql("execute"))

    async def test_creates_extension_when_missing(self):
        store, conn, _ = _store(ready=False)
        state = {"created": False}

        def respond(sql, args):
            if "extversion" in sql:
                return "0.8.0" if state["created"] else None
            return _setup_responder()(sql, args)

        async def execute(sql, *args):
            conn._record("execute", sql, args)
            if "CREATE EXTENSION" in sql:
                state["created"] = True
            return "OK"

        conn.fetchval_responder = respond
        conn.execute = execute
        await store.asetup()
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in conn.sql("execute"))

    async def test_clear_error_when_extension_cannot_be_created(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder(extversion=None)

        async def execute(sql, *args):
            if "CREATE EXTENSION" in sql:
                raise asyncpg.InsufficientPrivilegeError("permission denied to create extension")
            return "OK"

        conn.execute = execute
        with pytest.raises(RuntimeError, match="CREATE EXTENSION vector"):
            await store.asetup()
        assert store._setup_done is False

    async def test_old_extension_version_raises(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder(extversion="0.6.2")
        with pytest.raises(RuntimeError, match=r"pgvector >= 0\.7\.0"):
            await store.asetup()

    async def test_dimension_mismatch_raises(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder(column="vector(1536)")
        with pytest.raises(ValueError, match="dimension mismatch"):
            await store.asetup()

    async def test_text_only_table_with_embedding_store_raises(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder(column=None)
        with pytest.raises(ValueError, match="text-only"):
            await store.asetup()

    async def test_text_only_store_skips_extension_and_vector(self):
        store, conn, _ = _store(embedding=None, ready=False)
        conn.fetchval_responder = _setup_responder(extversion=None, column=None)
        await store.asetup()
        executed = conn.sql("execute")
        assert not any("embedding" in s or "hnsw" in s for s in executed)
        assert not any("extversion" in s for s in conn.sql("fetchval"))

    async def test_existing_index_with_other_metric_warns(self, caplog):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder(opclass="vector_l2_ops")
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store.pgvector"):
            await store.asetup()
        assert store._index_metric == DistanceMetric.EUCLIDEAN
        assert not any("USING hnsw" in s for s in conn.sql("execute"))
        assert "vector_l2_ops" in caplog.text

    async def test_large_dimension_skips_hnsw(self, caplog):
        emb = _Embedding()
        emb.dimension = 3072
        store, conn, _ = _store(embedding=emb, ready=False)
        conn.fetchval_responder = _setup_responder(column="vector(3072)")
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store.pgvector"):
            await store.asetup()
        assert not any("USING hnsw" in s for s in conn.sql("execute"))
        assert store._index_metric is None
        assert "HNSW limit" in caplog.text

    async def test_concurrent_setup_runs_once(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder()
        await asyncio.gather(*(store.asetup() for _ in range(5)))
        assert conn.transactions == 1

    async def test_recorded_version_skips_migration_insert(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder(version=CURRENT_SCHEMA_VERSION)
        await store.asetup()
        assert not any(s.startswith("INSERT INTO") for s in conn.sql("execute"))

    async def test_first_operation_runs_setup(self):
        store, conn, _ = _store(ready=False)
        conn.fetchval_responder = _setup_responder()
        await store.aget(CFG, "m1")
        assert store._setup_done is True
        assert conn.transactions == 1


# ---------------------------------------------------------------------------
# Writes and ownership
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStore:
    async def test_store_requires_user_id(self):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="user_id is required"):
            await store.astore({}, "x")

    async def test_store_under_reach_all_writes_as_identity_user(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        await store.astore({"user_id": "forged", **ADMIN}, "x", memory_id="m1")
        args = conn.calls[0][2]
        assert "admin" in args and "forged" not in args

    async def test_store_inserts_with_ownership_guard(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        mid = await store.astore(
            CFG,
            "fact",
            memory_type=MemoryType.SEMANTIC,
            category="prefs",
            metadata={"source": "chat", "user_id": "attacker", "content": "x"},
            memory_id="m1",
        )
        assert mid == "m1"
        _, sql, args = conn.calls[-1]
        assert sql.startswith('INSERT INTO "public"."tenxgraph_memories" AS m')
        assert "ON CONFLICT (collection, id) DO UPDATE SET" in sql
        assert sql.endswith("WHERE m.user_id = EXCLUDED.user_id RETURNING id")
        assert "$10::text::vector" in sql and "$9::text::jsonb" in sql
        assert args[:8] == (
            "m1",
            "agentflow_memories",
            "u1",
            "t1",
            None,
            "semantic",
            "prefs",
            "fact",
        )
        # Caller metadata cannot replace owned columns.
        assert json.loads(args[8]) == {"source": "chat"}
        assert args[9] == "[0.1,0.2,0.3,0.4]"
        # No value is interpolated into the SQL.
        assert "fact" not in sql and "u1" not in sql

    async def test_foreign_memory_id_raises_permission_error(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [None]  # the conflict UPDATE's WHERE matched no row
        with pytest.raises(PermissionError, match="belongs to another user"):
            await store.astore(CFG, "x", memory_id="theirs")

    async def test_scope_none_overwrite_has_no_owner_guard(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        cfg = {**ADMIN, "user_id": "admin"}
        await store.astore(cfg, "x", memory_id="m1")
        assert "WHERE m.user_id" not in conn.calls[-1][1]

    async def test_generated_id_used_when_absent(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "gen"}]
        with patch.object(store, "generate_framework_id", AsyncMock(return_value="gen")):
            assert await store.astore(CFG, "x") == "gen"
        assert conn.calls[-1][2][0] == "gen"

    async def test_int_user_id_stored_as_text(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        await store.astore({"user_id": 42}, "x", memory_id="m1")
        assert conn.calls[-1][2][2] == "42"

    async def test_system_user_id_is_an_owner(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        await store.astore({"user_id": system_user_id("kb")}, "doc", memory_id="m1")
        assert conn.calls[-1][2][2] == "__system__:kb"

    async def test_memory_key_is_atomic_upsert(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "existing"}]
        mid = await store.astore(CFG, "name is Ann", metadata={"memory_key": "user_name", "a": 1})
        assert mid == "existing"
        _, sql, args = conn.calls[-1]
        assert (
            "ON CONFLICT (collection, user_id, memory_type, memory_key) "
            "WHERE memory_key IS NOT NULL DO UPDATE SET" in sql
        )
        assert "metadata = m.metadata || EXCLUDED.metadata" in sql
        assert "WHERE m.user_id" not in sql
        assert args[4] == "user_name"
        assert json.loads(args[8]) == {"a": 1}

    async def test_memory_key_kwarg(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m"}]
        await store.astore(CFG, "x", memory_key="k")
        assert conn.calls[-1][2][4] == "k"

    async def test_memory_key_id_collision_is_value_error(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [asyncpg.UniqueViolationError("duplicate key")]
        with pytest.raises(ValueError, match="already used"):
            await store.astore(CFG, "x", memory_key="k", memory_id="taken")

    async def test_bad_memory_key(self):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="memory_key"):
            await store.astore(CFG, "x", memory_key="  ")

    async def test_message_content(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m"}]
        await store.astore(CFG, Message.text_message("hi there", role="user"), memory_id="m")
        args = conn.calls[-1][2]
        assert args[7] == "hi there"
        assert json.loads(args[8])["role"] == "user"

    async def test_text_only_insert_has_no_embedding(self):
        store, conn, _ = _store(embedding=None)
        conn.fetchrow_results = [{"id": "m"}]
        await store.astore(CFG, "x", memory_id="m")
        _, sql, args = conn.calls[-1]
        assert "embedding" not in sql
        assert len(args) == 9


@pytest.mark.asyncio
class TestEmbeddingFallback:
    async def test_write_falls_back_to_null_embedding(self, caplog):
        store, conn, _ = _store(embedding=_Embedding(fail=True))
        conn.fetchrow_results = [{"id": "m"}]
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store.pgvector"):
            await store.astore(CFG, "x", memory_id="m")
        assert conn.calls[-1][2][9] is None
        assert "without an embedding" in caplog.text

    async def test_write_raise_mode(self):
        store, conn, _ = _store(embedding=_Embedding(fail=True), on_embedding_error="raise")
        with pytest.raises(RuntimeError, match="embedding service down"):
            await store.astore(CFG, "x", memory_id="m")
        assert conn.calls == []

    async def test_wrong_dimension_always_raises(self):
        store, conn, _ = _store(embedding=_Embedding(vector=[0.1, 0.2]))
        with pytest.raises(ValueError, match="invalid vector"):
            await store.astore(CFG, "x", memory_id="m")
        assert conn.calls == []

    async def test_nan_vector_raises(self):
        store, _, _ = _store(embedding=_Embedding(vector=[0.1, float("nan"), 0.3, 0.4]))
        with pytest.raises(ValueError, match="invalid vector"):
            await store.astore(CFG, "x", memory_id="m")

    async def test_search_falls_back_to_text(self, caplog):
        store, conn, _ = _store(embedding=_Embedding(fail=True), retrieval_strategy="hybrid")
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store.pgvector"):
            await store.asearch(CFG, "query")
        sql = conn.calls[-1][1]
        assert "websearch_to_tsquery" in sql and "FULL OUTER JOIN" not in sql
        assert "searching by text" in caplog.text

    async def test_search_raise_mode(self):
        store, _, _ = _store(embedding=_Embedding(fail=True), on_embedding_error="raise")
        with pytest.raises(RuntimeError):
            await store.asearch(CFG, "query")


@pytest.mark.asyncio
class TestUpdateDelete:
    async def test_update_single_statement_scoped(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        await store.aupdate(CFG, "m1", "new", metadata={"category": "c2", "x": 1})
        _, sql, args = conn.calls[-1]
        assert sql.startswith('UPDATE "public"."tenxgraph_memories" SET')
        assert '"collection" = $1 AND "user_id" = $2 AND "id" = $3' in sql
        assert "ELSE metadata || $8::text::jsonb" in sql
        assert args[:7] == ("agentflow_memories", "u1", "m1", "new", "c2", None, False)
        assert json.loads(args[7]) == {"x": 1}
        assert args[8] == "[0.1,0.2,0.3,0.4]"

    async def test_update_replace_metadata(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        await store.aupdate(CFG, "m1", "new", metadata={"x": 1}, replace_metadata=True)
        assert conn.calls[-1][2][6] is True

    async def test_update_missing_or_foreign_is_not_found(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [None]
        with pytest.raises(ValueError, match="not found"):
            await store.aupdate(CFG, "theirs", "x")

    async def test_update_scope_none_has_no_user_filter(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        await store.aupdate(dict(ADMIN), "m1", "x")
        assert '"user_id"' not in conn.calls[-1][1]

    async def test_update_requires_user_id(self):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="user_id is required"):
            await store.aupdate({}, "m1", "x")

    async def test_delete_scoped_and_not_found(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [None]
        with pytest.raises(ValueError, match="not found"):
            await store.adelete(CFG, "m1")
        _, sql, args = conn.calls[-1]
        assert '"user_id" = $2' in sql
        assert args == ("agentflow_memories", "u1", "m1")

    async def test_delete_ok(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "m1"}]
        await store.adelete(CFG, "m1")

    async def test_forget_under_reach_all_names_the_identity_user(self):
        store, conn, _ = _store()
        await store.aforget_memory({"user_id": "forged", **ADMIN})
        args = conn.calls[0][2]
        assert "admin" in args and "forged" not in args

    async def test_forget_narrowing(self):
        store, conn, _ = _store()
        await store.aforget_memory(
            CFG, memory_type="semantic", category="c", filters={"memory_key": "k", "tag": "x"}
        )
        _, sql, args = conn.calls[-1]
        assert sql.startswith('DELETE FROM "public"."tenxgraph_memories" WHERE')
        assert '"thread_id" = $3' in sql and '"memory_type" = $4' in sql
        assert '"memory_key" = $6' in sql and '"metadata" @> $7::text::jsonb' in sql
        assert args == ("agentflow_memories", "u1", "t1", "semantic", "c", "k", '{"tag": "x"}')

    async def test_forget_rejects_inexact_filter(self):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="Unsupported filter"):
            await store.aforget_memory(CFG, filters={"tags": ["a", "b"]})


# ---------------------------------------------------------------------------
# Reads and search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReads:
    async def test_get_scoped(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [_row(memory_key="k")]
        result = await store.aget(CFG, "m1")
        _, sql, args = conn.calls[-1]
        assert '"collection" = $1 AND "user_id" = $2 AND "id" = $3' in sql
        assert args == ("agentflow_memories", "u1", "m1")
        assert result.id == "m1" and result.score == 1.0
        assert result.metadata == {
            "source": "chat",
            "category": "general",
            "memory_type": "episodic",
            "memory_key": "k",
        }

    async def test_get_missing_returns_none(self):
        store, _, _ = _store()
        assert await store.aget(CFG, "m1") is None

    async def test_get_requires_user_id(self):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="user_id is required"):
            await store.aget({"user_id": "  "}, "m1")

    async def test_get_scope_none_reads_any_user(self):
        store, conn, _ = _store()
        await store.aget(dict(ADMIN), "m1")
        assert conn.calls[-1][2] == ("agentflow_memories", "m1")

    async def test_get_all(self):
        store, conn, _ = _store()
        conn.fetch_results = [[_row(), _row(id="m2", metadata={"a": 1})]]
        results = await store.aget_all(CFG, limit=5)
        assert [r.id for r in results] == ["m1", "m2"]
        assert results[1].metadata["a"] == 1
        _, sql, args = conn.calls[-1]
        assert "ORDER BY updated_at DESC, id LIMIT $3" in sql
        assert args == ("agentflow_memories", "u1", 5)

    async def test_collection_precedence(self):
        store, conn, _ = _store(collection=lambda cfg: f"org_{cfg['user']['tenant']['org']}")
        await store.aget({"user_id": "u1", "user": {"user_id": "u1", "tenant": {"org": "a"}}}, "m1")
        assert conn.calls[-1][2][0] == "org_a"
        await store.aget({"user_id": "u1", "collection": "explicit"}, "m1")
        assert conn.calls[-1][2][0] == "explicit"
        with pytest.raises(ValueError, match="collection"):
            await store.aget({"user_id": "u1", "collection": " "}, "m1")


@pytest.mark.asyncio
class TestSearch:
    @pytest.mark.parametrize(
        ("metric", "operator", "similarity"),
        [
            (DistanceMetric.COSINE, "<=>", "(1 - (m.embedding <=> $3::text::vector))"),
            (DistanceMetric.EUCLIDEAN, "<->", "(1 / (1 + (m.embedding <-> $3::text::vector)))"),
            (DistanceMetric.DOT_PRODUCT, "<#>", "((m.embedding <#> $3::text::vector) * -1)"),
            (DistanceMetric.MANHATTAN, "<+>", "(1 / (1 + (m.embedding <+> $3::text::vector)))"),
        ],
    )
    async def test_similarity_sql(self, metric, operator, similarity):
        store, conn, _ = _store(distance_metric=metric)
        await store.asearch(CFG, "what is my name", limit=3)
        _, sql, args = conn.calls[-1]
        assert f"{similarity} AS score" in sql
        assert f"ORDER BY m.embedding {operator} $3::text::vector LIMIT $4" in sql
        assert "m.embedding IS NOT NULL" in sql
        # thread_id is not a search filter (long-term memory is cross-thread).
        assert '"thread_id"' not in sql.split(" WHERE ", 1)[1]
        assert args == ("agentflow_memories", "u1", "[0.1,0.2,0.3,0.4]", 3)

    async def test_text_sql(self):
        store, conn, _ = _store(text_search_config="simple")
        await store.asearch(CFG, "it's; DROP", retrieval_strategy="text", limit=2)
        _, sql, args = conn.calls[-1]
        assert "ts_rank(m.tsv, tsq_input.q, 32) AS score" in sql
        assert "websearch_to_tsquery($3::text::regconfig, $4) AS tsq_input(q)" in sql
        assert "m.tsv @@ tsq_input.q" in sql and "DROP" not in sql
        assert args == ("agentflow_memories", "u1", "simple", "it's; DROP", 2)

    async def test_hybrid_sql(self):
        store, conn, _ = _store(retrieval_strategy="hybrid", hybrid_weights=(2.0, 1.0), rrf_k=10)
        await store.asearch(CFG, "q", limit=5)
        _, sql, args = conn.calls[-1]
        assert sql.startswith("WITH vec AS (")
        assert "ROW_NUMBER() OVER (ORDER BY m.embedding <=> $3::text::vector) AS rnk" in sql
        assert "FULL OUTER JOIN txt ON vec.id = txt.id" in sql
        assert "COALESCE($7::float8 / ($9::int + vec.rnk), 0)" in sql
        assert "m.collection = $10 AND m.id = f.id" in sql
        assert "AS similarity" in sql
        assert args == (
            "agentflow_memories",
            "u1",
            "[0.1,0.2,0.3,0.4]",
            "english",
            "q",
            20,
            2.0,
            1.0,
            10,
            "agentflow_memories",
            5,
        )

    async def test_hybrid_candidates_setting(self):
        store, conn, _ = _store(retrieval_strategy="hybrid", hybrid_candidates=7)
        await store.asearch(CFG, "q", limit=50)
        assert conn.calls[-1][2][5] == 7

    async def test_score_threshold_on_similarity(self):
        store, conn, _ = _store()
        conn.fetch_results = [[_row(id="a", score=0.97), _row(id="b", score=0.5)]]
        results = await store.asearch(CFG, "q", score_threshold=0.95)
        assert [r.id for r in results] == ["a"]
        assert results[0].score == pytest.approx(0.97)

    async def test_negative_similarity_reported_as_zero(self):
        store, conn, _ = _store()
        conn.fetch_results = [[_row(score=-0.3), _row(id="n", score=float("nan"))]]
        results = await store.asearch(CFG, "q")
        assert [r.score for r in results] == [0.0, 0.0]

    async def test_hybrid_score_normalized_and_threshold_on_similarity(self):
        store, conn, _ = _store(retrieval_strategy="hybrid")
        top = 2.0 / 61
        conn.fetch_results = [
            [
                _row(id="a", score=top, similarity=0.96),
                _row(id="b", score=1.0 / 62, similarity=None),
                _row(id="c", score=1.0 / 63, similarity=0.4),
            ]
        ]
        results = await store.asearch(CFG, "q", score_threshold=0.9)
        assert [r.id for r in results] == ["a"]
        assert results[0].score == pytest.approx(1.0)

    async def test_filters_and_memory_type(self):
        store, conn, _ = _store()
        await store.asearch(
            CFG,
            "q",
            memory_type=MemoryType.SEMANTIC,
            category="prefs",
            filters={"memory_key": "user_name", "x'; DROP TABLE t; --": True},
        )
        _, sql, args = conn.calls[-1]
        assert 'm."memory_type" = $3 AND m."category" = $4 AND m."memory_key" = $5' in sql
        assert 'm."metadata" @> $6::text::jsonb' in sql
        assert "DROP" not in sql
        assert json.loads(args[5]) == {"x'; DROP TABLE t; --": True}

    @pytest.mark.parametrize("value", [None, ["a"], {"$gt": 1}, float("inf")])
    async def test_rejects_inexact_filters(self, value):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="Unsupported filter"):
            await store.asearch(CFG, "q", filters={"k": value})

    async def test_search_requires_user_id(self):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="user_id is required"):
            await store.asearch({}, "q")

    async def test_search_scope_none_has_no_user_filter(self):
        store, conn, _ = _store()
        await store.asearch(dict(ADMIN), "q")
        assert '"user_id"' not in conn.calls[-1][1].split(" WHERE ", 1)[1]

    async def test_bad_limit(self):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="limit"):
            await store.asearch(CFG, "q", limit=0)

    async def test_text_only_search_default_and_refusal(self):
        store, conn, _ = _store(embedding=None)
        await store.asearch(CFG, "q")
        assert "websearch_to_tsquery" in conn.calls[-1][1]
        with pytest.raises(ValueError, match="text-only"):
            await store.asearch(CFG, "q", retrieval_strategy="similarity")

    async def test_max_tokens_budget(self):
        store, conn, _ = _store()
        conn.fetch_results = [[_row(id="a", content="x" * 400), _row(id="b", content="y" * 400)]]
        results = await store.asearch(CFG, "q", max_tokens=110)
        assert [r.id for r in results] == ["a"]


@pytest.mark.asyncio
class TestSettingsPrecedence:
    """Store constructor -> MemoryConfig / scope -> method call (D11)."""

    async def test_store_default_used_when_call_passes_none(self):
        store, conn, _ = _store(retrieval_strategy="hybrid")
        await store.asearch(CFG, "q", retrieval_strategy=None)
        assert "FULL OUTER JOIN" in conn.calls[-1][1]

    async def test_call_wins_over_store(self):
        store, conn, _ = _store(retrieval_strategy="hybrid")
        await store.asearch(CFG, "q", retrieval_strategy=RetrievalStrategy.TEXT)
        assert "websearch_to_tsquery" in conn.calls[-1][1]
        assert "FULL OUTER JOIN" not in conn.calls[-1][1]

    async def test_scope_config_wins_over_memory_config_and_store(self):
        store, conn, _ = _store(retrieval_strategy="similarity")
        memory = MemoryConfig(retrieval_strategy="hybrid")
        scope = UserMemoryConfig(retrieval_strategy="text")
        await _search_scope_memory(
            store=store,
            config=dict(CFG),
            query="q",
            memory_type=None,
            category=None,
            limit=3,
            score_threshold=None,
            task_manager=None,
            retrieval_strategy=_memory_scope_retrieval_strategy(memory, scope),
        )
        assert "websearch_to_tsquery" in conn.calls[-1][1]
        assert "FULL OUTER JOIN" not in conn.calls[-1][1]

    async def test_memory_config_wins_over_store(self):
        store, conn, _ = _store(retrieval_strategy="similarity")
        memory = MemoryConfig(retrieval_strategy="hybrid")
        await _search_scope_memory(
            store=store,
            config=dict(CFG),
            query="q",
            memory_type=None,
            category=None,
            limit=3,
            score_threshold=None,
            task_manager=None,
            retrieval_strategy=_memory_scope_retrieval_strategy(memory, UserMemoryConfig()),
        )
        assert "FULL OUTER JOIN" in conn.calls[-1][1]

    async def test_distance_metric_default_and_call(self, caplog):
        store, conn, _ = _store(distance_metric=DistanceMetric.EUCLIDEAN)
        await store.asearch(CFG, "q", distance_metric=None)
        assert "<->" in conn.calls[-1][1]
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store.pgvector"):
            await store.asearch(CFG, "q", distance_metric=DistanceMetric.COSINE)
            await store.asearch(CFG, "q", distance_metric=DistanceMetric.COSINE)
        assert "<=>" in conn.calls[-1][1]
        assert caplog.text.count("runs without the index") == 1


# ---------------------------------------------------------------------------
# Backfill, ping, release, retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBackfill:
    async def test_backfills_in_batches(self):
        store, conn, _ = _store()
        conn.fetch_results = [
            [
                {"collection": "c", "id": "1", "content": "a"},
                {"collection": "c", "id": "2", "content": "b"},
            ],
            [{"collection": "c", "id": "3", "content": "c"}],
        ]
        conn.execute_results = ["UPDATE 2", "UPDATE 1"]
        assert await store.abackfill_embeddings(batch_size=2) == 3
        fetches = [c for c in conn.calls if c[0] == "fetch"]
        assert fetches[0][2] == ("", "", 2)
        assert fetches[1][2] == ("c", "2", 2)
        update = next(c for c in conn.calls if c[0] == "execute")
        assert "unnest($1::text[], $2::text[], $3::text[], $4::text[])" in update[1]
        assert "m.embedding IS NULL AND m.content = d.content" in update[1]
        assert update[2][3] == ["[0.1,0.2,0.3,0.4]"] * 2
        store.embedding.aembed_batch.assert_awaited()

    async def test_nothing_to_backfill(self):
        store, _, _ = _store()
        assert await store.abackfill_embeddings() == 0

    async def test_text_only_refuses(self):
        store, _, _ = _store(embedding=None)
        with pytest.raises(ValueError, match="text-only"):
            await store.abackfill_embeddings()

    async def test_bad_batch_size(self):
        store, _, _ = _store()
        with pytest.raises(ValueError, match="batch_size"):
            await store.abackfill_embeddings(batch_size=0)


@pytest.mark.asyncio
class TestLifecycle:
    async def test_ping_is_select_one_without_setup(self):
        store, conn, _ = _store(ready=False)
        await store.aping()
        assert conn.calls == [("fetchval", "SELECT 1", ())]
        assert store._setup_done is False

    async def test_release_leaves_passed_pool_open(self):
        store, _, pool = _store()
        await store.arelease()
        pool.close.assert_not_awaited()

    async def test_release_closes_owned_pool(self):
        conn = FakeConn()
        pool = FakePool(conn)
        store = PgVectorStore(
            _Embedding(), dsn="postgresql://u:p@h/db", pool_config={"min_size": 1}
        )
        with patch("asyncpg.create_pool", AsyncMock(return_value=pool)) as create:
            await store.aping()
            await store.aping()
        create.assert_awaited_once()
        kwargs = create.await_args.kwargs
        assert kwargs["dsn"] == "postgresql://u:p@h/db"
        assert kwargs["min_size"] == 1 and kwargs["command_timeout"] == 60.0
        await store.arelease()
        pool.close.assert_awaited_once()
        assert store._pool is None

    async def test_release_owned_pool_never_built(self):
        store = PgVectorStore(_Embedding(), dsn="postgresql://h/db")
        await store.arelease()  # no pool yet: nothing to close, no error

    async def test_retries_connection_errors(self):
        store, conn, pool = _store()
        pool.acquire_errors = [asyncpg.PostgresConnectionError("gone")]
        with patch("tenxgraph.storage.store.pgvector_store.asyncio.sleep", AsyncMock()) as sleep:
            await store.aping()
        sleep.assert_awaited_once_with(1)
        assert conn.calls == [("fetchval", "SELECT 1", ())]

    async def test_gives_up_after_three_attempts(self):
        store, _, pool = _store()
        pool.acquire_errors = [asyncpg.InterfaceError("closed")] * 3
        with (
            patch("tenxgraph.storage.store.pgvector_store.asyncio.sleep", AsyncMock()),
            pytest.raises(asyncpg.InterfaceError),
        ):
            await store.aping()

    async def test_other_errors_are_not_retried(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [asyncpg.UndefinedTableError("missing")]
        with pytest.raises(asyncpg.UndefinedTableError):
            await store.aget(CFG, "m1")
        assert len(conn.calls) == 1

    async def test_acquire_timeout_passed(self):
        store, _, pool = _store(acquire_timeout=3.0)
        await store.aping()
        assert pool.acquire_timeouts == [3.0]

    async def test_schema_version(self):
        store, conn, _ = _store()
        conn.fetchval_responder = lambda sql, args: True if "to_regclass" in sql else 1
        assert await store.aget_schema_version() == 1
        conn.fetchval_responder = lambda sql, args: False
        assert await store.aget_schema_version() == 0


# ---------------------------------------------------------------------------
# Memory layer: _do_write uses the atomic key upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDoWriteCapability:
    async def test_keyed_write_uses_upsert_without_search(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "existing"}]
        with patch.object(store, "asearch", AsyncMock()) as search:
            result = await _do_write(
                store,
                CFG,
                "store",
                "name is Ann",
                "",
                MemoryType.EPISODIC,
                "general",
                {"memory_key": "user_name"},
                "merge",
            )
        search.assert_not_awaited()
        assert result == {
            "status": "upserted_by_key",
            "memory_id": "existing",
            "memory_key": "user_name",
        }
        args = conn.calls[-1][2]
        assert args[3] is None  # thread_id stripped: long-term memory is cross-thread
        assert args[4] == "user_name"

    async def test_unkeyed_write_keeps_dedup(self):
        store, conn, _ = _store()
        conn.fetchrow_results = [{"id": "new"}]
        with patch.object(store, "asearch", AsyncMock(return_value=[])) as search:
            result = await _do_write(
                store,
                CFG,
                "store",
                "fact",
                "",
                MemoryType.EPISODIC,
                "general",
                None,
                "merge",
            )
        search.assert_awaited_once()
        assert result["status"] == "stored"

    async def test_mock_store_without_flag_keeps_old_path(self):
        mock = AsyncMock()
        mock.asearch = AsyncMock(return_value=[])
        mock.astore = AsyncMock(return_value="m")
        result = await _do_write(
            mock,
            CFG,
            "store",
            "x",
            "",
            MemoryType.EPISODIC,
            "general",
            {"memory_key": "k"},
            "merge",
        )
        assert result["status"] == "stored"
        assert mock.asearch.await_count == 2
