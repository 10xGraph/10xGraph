"""Extra columns of any Postgres type, against a REAL Postgres (pgvector) + Redis.

Covers ``jsonb``, ``text[]``, ``numeric(10,2)``, ``date`` and a custom enum created in the
test, on both ``PgVectorStore`` and ``PgCheckpointer``: the server-resolved column types, a
round trip of values, codec errors that never echo the value, an unknown type failing setup,
re-setup matching by canonical name, and filters limited to ``is_null``.

    POSTGRES_DSN=postgresql://test:test@localhost:55432/test_agentflow \\
    REDIS_URL=redis://localhost:56379/1 \\
    pytest tests/integration/test_column_types_real.py --integration

Each test gets its own Postgres schema (dropped afterwards) and Redis key prefix.
"""

# SQL here is built only from identifiers the fixtures generate (uuid hex), never input.
# ruff: noqa: S608

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import uuid
from decimal import Decimal

import pytest
import pytest_asyncio

from tenxgraph.storage.checkpointer import PgCheckpointer
from tenxgraph.storage.columns import ColumnValueError
from tenxgraph.storage.sql_filter import FilterError
from tenxgraph.storage.store import PgVectorStore
from tenxgraph.utils import ThreadInfo


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
DIM = 4


class _Embedding:
    dimension = DIM

    async def aembed(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    async def aembed_query(self, text):
        return await self.aembed(text)

    async def aembed_batch(self, texts):
        return [await self.aembed(t) for t in texts]


@pytest_asyncio.fixture
async def pool():
    asyncpg = pytest.importorskip("asyncpg")
    try:
        pg_pool = await asyncio.wait_for(
            asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=6), timeout=5
        )
    except Exception as e:
        pytest.skip(f"Postgres not reachable at POSTGRES_DSN: {e}")
    if await pg_pool.fetchval("SELECT current_setting('server_version_num')::int") < 170000:
        await pg_pool.close()
        pytest.skip("type modifiers need PostgreSQL 17 (to_regtypemod)")
    await pg_pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    yield pg_pool
    await pg_pool.close()


@pytest_asyncio.fixture
async def schema(pool):
    name = f"tg_{uuid.uuid4().hex[:10]}"
    await pool.execute(f'CREATE SCHEMA "{name}"')
    await pool.execute(f"CREATE TYPE \"{name}\".mood AS ENUM ('calm', 'busy')")
    yield name
    await pool.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')


def _extra(schema):
    return {
        "settings": "jsonb",
        "tags": "text[]",
        "price": "NUMERIC(10, 2)",
        "due": "date",
        "mood": f"{schema}.mood",
        "seats": {"type": "integer", "required": True},
    }


def _user(**over):
    user = {
        "user_id": "alice",
        "org_id": "A",
        "settings": {"theme": "dark", "flags": [1, 2]},
        "tags": ["x", "y"],
        "price": Decimal("12.50"),
        "due": dt.date(2026, 11, 5),
        "mood": "calm",
        "seats": 3,
    }
    user.update(over)
    return user


EXPECTED_TYPES = {
    "org_id": "text",
    "settings": "jsonb",
    "tags": "text[]",
    "price": "numeric(10,2)",
    "due": "date",
    "seats": "integer",
}


async def _types(pool, schema, table):
    rows = await pool.fetch(
        "SELECT attname, format_type(atttypid, atttypmod) AS t FROM pg_attribute "
        "WHERE attrelid = to_regclass($1) AND attnum > 0 AND NOT attisdropped",
        f'"{schema}"."{table}"',
    )
    return {r["attname"]: r["t"] for r in rows}


def _check_types(found, schema):
    for name, type_ in EXPECTED_TYPES.items():
        assert found[name] == type_, name
    # format_type schema-qualifies a type that is not on the search_path.
    assert found["mood"] == f"{schema}.mood"


def _check_values(row):
    assert json.loads(row["settings"]) == {"theme": "dark", "flags": [1, 2]}
    assert row["tags"] == ["x", "y"]
    assert row["price"] == Decimal("12.50")
    assert row["due"] == dt.date(2026, 11, 5)
    assert row["mood"] == "calm"
    assert row["seats"] == 3


# ---------------------------------------------------------------------------
# PgVectorStore
# ---------------------------------------------------------------------------


def _store(pool, schema, **kw):
    kw.setdefault("tenant_columns", ["org_id"])
    kw.setdefault("extra_columns", _extra(schema))
    return PgVectorStore(_Embedding(), pool=pool, schema=schema, table="mem", **kw)


async def test_store_round_trip_and_types(pool, schema):
    store = _store(pool, schema)
    await store.asetup()
    _check_types(await _types(pool, schema, "mem"), schema)

    mid = await store.astore({"user_id": "alice", "user": _user()}, "hello", memory_id="m1")
    row = await pool.fetchrow(f'SELECT * FROM "{schema}"."mem" WHERE id = $1', mid)
    _check_values(row)

    # A JSON string value is stored as a JSON string, not parsed.
    cfg = {"user_id": "alice", "user": _user(settings="plain", mood="busy", tags=[])}
    await store.aupdate(cfg, mid, "hello again")
    row = await pool.fetchrow(f'SELECT * FROM "{schema}"."mem" WHERE id = $1', mid)
    assert json.loads(row["settings"]) == "plain"
    assert row["mood"] == "busy"
    assert row["tags"] == []


async def test_store_resetup_matches_canonical_names(pool, schema):
    await _store(pool, schema).asetup()
    # Declared with other spellings of the same types: no "different type" error.
    again = _store(
        pool,
        schema,
        extra_columns={
            "settings": "JSONB",
            "tags": "TEXT[]",
            "price": "numeric(10,2)",
            "due": "DATE",
            "mood": f"{schema}.MOOD",
            "seats": "int4",
        },
    )
    await again.asetup()


async def test_store_changed_type_raises(pool, schema):
    await _store(pool, schema).asetup()
    changed = _store(pool, schema, extra_columns={**_extra(schema), "price": "numeric(12,2)"})
    with pytest.raises(ValueError, match=r"already exists .* as numeric\(10,2\)"):
        await changed.asetup()


async def test_store_unknown_type_fails_setup(pool, schema):
    store = _store(pool, schema, extra_columns={"x": "no_such_type"})
    with pytest.raises(ValueError, match="does not exist in the database"):
        await store.asetup()


async def test_store_codec_error_names_column_not_value(pool, schema):
    store = _store(pool, schema)
    await store.asetup()
    hidden = "2026-11-05-hidden"
    cfg = {"user_id": "alice", "user": _user(due=hidden)}
    with pytest.raises(ColumnValueError, match="'due' is date") as info:
        await store.astore(cfg, "hello", memory_id="m1")
    assert hidden not in str(info.value)
    cfg = {"user_id": "alice", "user": _user(tags="not-a-list")}
    with pytest.raises(ColumnValueError, match="'tags' is text\\[\\]") as info:
        await store.astore(cfg, "hello", memory_id="m1")
    assert "not-a-list" not in str(info.value)
    assert await pool.fetchval(f'SELECT count(*) FROM "{schema}"."mem"') == 0


async def test_store_filters_is_null_only(pool, schema):
    store = _store(pool, schema)
    await store.asetup()
    await store.astore({"user_id": "alice", "user": _user()}, "one", memory_id="m1")
    cfg = {"user_id": "alice", "user": _user(settings=None, tags=None)}
    await store.astore(cfg, "two", memory_id="m2")
    got = await store.aget_all(cfg, filters={"settings": {"is_null": True}})
    assert [m.id for m in got] == ["m2"]
    got = await store.aget_all(cfg, filters={"tags": {"is_null": False}})
    assert [m.id for m in got] == ["m1"]
    with pytest.raises(FilterError, match="supports only is_null"):
        await store.aget_all(cfg, filters={"price": {"gt": 1}})


# ---------------------------------------------------------------------------
# PgCheckpointer
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def make_cp(schema):
    created = []
    prefix = f"tgtest:{uuid.uuid4().hex[:8]}:"

    def make(**kw):
        kw.setdefault("tenant_columns", ["org_id"])
        kw.setdefault("extra_columns", _extra(schema))
        cp = PgCheckpointer(
            postgres_dsn=POSTGRES_DSN,
            redis_url=REDIS_URL,
            schema=schema,
            redis_key_prefix=prefix,
            **kw,
        )
        created.append(cp)
        return cp

    yield make
    for cp in created:
        try:
            keys = [k async for k in cp.redis.scan_iter(match=f"{prefix}*")]
            if keys:
                await cp.redis.delete(*keys)
        except Exception:  # noqa: S110 - best-effort test cleanup
            pass
        await cp.arelease()


def _cp_cfg(thread, **user):
    return {"thread_id": thread, "user_id": "alice", "user": _user(**user)}


async def test_checkpointer_round_trip_and_types(pool, schema, make_cp):
    cp = make_cp()
    await cp.asetup()
    _check_types(await _types(pool, schema, "threads"), schema)

    cfg = _cp_cfg("t1")
    await cp.aput_thread(cfg, ThreadInfo(thread_id="t1", thread_name="one"))
    row = await pool.fetchrow(f'SELECT * FROM "{schema}"."threads" WHERE thread_id = $1', "t1")
    _check_values(row)


async def test_checkpointer_resetup_matches_canonical_names(pool, schema, make_cp):
    await make_cp().asetup()
    again = make_cp(
        extra_columns={
            "settings": "jsonb",
            "tags": "TEXT[]",
            "price": "numeric(10,2)",
            "due": "date",
            "mood": f"{schema}.mood",
            "seats": "integer",
        }
    )
    await again.asetup()


async def test_checkpointer_unknown_type_fails_setup(pool, schema, make_cp):
    cp = make_cp(extra_columns={"x": f"{schema}.no_such_type"})
    with pytest.raises(ValueError, match="does not exist in the database"):
        await cp.asetup()
    # The whole setup transaction rolled back: no threads table was left behind.
    assert await pool.fetchval("SELECT to_regclass($1)", f'"{schema}"."threads"') is None


async def test_checkpointer_codec_error_names_column_not_value(pool, schema, make_cp):
    cp = make_cp()
    await cp.asetup()
    hidden = "13.5-hidden"
    with pytest.raises(ColumnValueError, match="'price' is numeric\\(10, 2\\)") as info:
        await cp.aput_thread(_cp_cfg("t2", price=hidden), ThreadInfo(thread_id="t2"))
    assert hidden not in str(info.value)
    assert await pool.fetchval(f'SELECT count(*) FROM "{schema}"."threads"') == 0


async def test_checkpointer_filters_is_null_only(pool, schema, make_cp):
    cp = make_cp()
    await cp.asetup()
    await cp.aput_thread(_cp_cfg("t1"), ThreadInfo(thread_id="t1"))
    await cp.aput_thread(_cp_cfg("t2", settings=None), ThreadInfo(thread_id="t2"))
    cfg = _cp_cfg("t1")
    got = await cp.alist_threads(cfg, filters={"settings": {"is_null": True}})
    assert [t.thread_id for t in got] == ["t2"]
    with pytest.raises(FilterError, match="supports only is_null"):
        await cp.alist_threads(cfg, filters={"mood": "calm"})
