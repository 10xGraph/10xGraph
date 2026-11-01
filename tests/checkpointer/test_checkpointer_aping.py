"""``BaseCheckpointer.aping`` is abstract and the three built-ins implement it."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.storage.checkpointer import InMemoryCheckpointer, PgCheckpointer
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer


def test_a_subclass_without_aping_cannot_be_created():
    abstract = set(BaseCheckpointer.__abstractmethods__)
    assert "aping" in abstract

    namespace = {name: AsyncMock() for name in abstract - {"aping"}}
    incomplete = type("Incomplete", (BaseCheckpointer,), namespace)
    with pytest.raises(TypeError, match="aping"):
        incomplete()


@pytest.mark.asyncio
async def test_in_memory_aping_returns_none():
    assert await InMemoryCheckpointer().aping() is None


def _sqlite(path):
    pytest.importorskip("aiosqlite")
    from tenxgraph.storage.checkpointer import SqliteCheckpointer  # noqa: PLC0415

    return SqliteCheckpointer(str(path))


@pytest.mark.asyncio
async def test_sqlite_aping_passes_after_setup(tmp_path):
    cp = _sqlite(tmp_path / "cp.db")
    try:
        await cp.asetup()
        assert await cp.aping() is None
    finally:
        await cp.arelease()


@pytest.mark.asyncio
async def test_sqlite_aping_creates_no_tables(tmp_path):
    cp = _sqlite(tmp_path / "cp.db")
    try:
        await cp.aping()
        conn = await cp._get_conn()
        async with conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'") as cur:
            row = await cur.fetchone()
        assert row[0] == 0
    finally:
        await cp.arelease()


@pytest.mark.asyncio
async def test_sqlite_aping_raises_when_the_path_is_unusable(tmp_path):
    cp = _sqlite(tmp_path)  # a directory, not a database file
    try:
        with pytest.raises(Exception):  # noqa: B017 - the exact error is the driver's
            await cp.aping()
    finally:
        await cp.arelease()


# ---------------------------------------------------------------------------
# PgCheckpointer, mocked
# ---------------------------------------------------------------------------


def _pg(*, redis: bool = True) -> PgCheckpointer:
    with patch("asyncpg.create_pool", return_value=MagicMock()):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
        )
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=1)
    conn.execute = AsyncMock()
    cp._test_conn = conn  # type: ignore[attr-defined]

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    cp._get_pg_pool = AsyncMock(return_value=pool)  # type: ignore[method-assign]
    cp._initialize_schema = AsyncMock()  # type: ignore[method-assign]
    cp.aget_schema_version = AsyncMock(return_value=cp._get_current_schema_version())  # type: ignore[method-assign]
    if redis:
        cp.redis = MagicMock()
        cp.redis.ping = AsyncMock(return_value=True)
    else:
        cp.redis = None
    return cp


@pytest.mark.asyncio
async def test_pg_aping_passes_when_everything_is_healthy():
    cp = _pg()
    assert await cp.aping() is None
    cp._test_conn.fetchval.assert_awaited_once_with("SELECT 1")
    cp.redis.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_pg_aping_raises_when_postgres_is_unreachable():
    cp = _pg()
    cp._test_conn.fetchval = AsyncMock(side_effect=OSError("connection refused"))
    with pytest.raises(OSError, match="connection refused"):
        await cp.aping()
    cp.redis.ping.assert_not_awaited()


@pytest.mark.asyncio
async def test_pg_aping_raises_when_the_schema_is_older_than_expected():
    cp = _pg()
    expected = cp._get_current_schema_version()
    cp.aget_schema_version = AsyncMock(return_value=expected - 1)
    with pytest.raises(RuntimeError) as exc:
        await cp.aping()
    message = str(exc.value)
    assert str(expected - 1) in message
    assert str(expected) in message


@pytest.mark.asyncio
async def test_pg_aping_raises_on_a_missing_schema():
    cp = _pg()
    cp.aget_schema_version = AsyncMock(return_value=0)
    with pytest.raises(RuntimeError, match="below the expected"):
        await cp.aping()


@pytest.mark.asyncio
async def test_pg_aping_raises_when_redis_is_down_and_leaves_the_breaker_closed():
    cp = _pg()
    cp.redis.ping = AsyncMock(side_effect=ConnectionError("redis down"))
    with pytest.raises(ConnectionError, match="redis down"):
        await cp.aping()
    assert cp._redis_breaker.open_until is None


@pytest.mark.asyncio
async def test_pg_aping_checks_only_postgres_without_redis():
    cp = _pg(redis=False)
    assert await cp.aping() is None
    cp._test_conn.fetchval.assert_awaited_once_with("SELECT 1")


@pytest.mark.asyncio
async def test_pg_aping_does_no_writes():
    cp = _pg()
    await cp.aping()
    cp._test_conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_pg_aping_never_creates_the_schema():
    cp = _pg()
    await cp.aping()
    cp._initialize_schema.assert_not_awaited()


# ---------------------------------------------------------------------------
# Real Postgres and Redis
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.environ.get("DATABASE_URL") and os.environ.get("REDIS_URL")),
    reason="needs DATABASE_URL and REDIS_URL",
)
async def test_pg_aping_against_real_services():
    cp = PgCheckpointer(
        postgres_dsn=os.environ["DATABASE_URL"],
        redis_url=os.environ["REDIS_URL"],
    )
    try:
        await cp.asetup()
        assert await cp.aping() is None
    finally:
        await cp.arelease()
