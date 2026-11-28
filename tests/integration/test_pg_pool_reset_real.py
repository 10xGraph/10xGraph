"""A pool built with ``DEFAULT_POOL_CONFIG`` skips asyncpg's reset query but leaks no state.

Requires a running Postgres (``POSTGRES_DSN``). One connection per pool, so every checkout
gets the connection the previous one released.
"""

import asyncio
import os

import pytest
import pytest_asyncio

from tenxgraph.storage._pg_common import DEFAULT_POOL_CONFIG, run_on_connection
from tenxgraph.storage.columns import ddl_session


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow"
)
LOCK_KEY = 815_204_771


@pytest_asyncio.fixture
async def pool():
    asyncpg = pytest.importorskip("asyncpg")
    try:
        pg_pool = await asyncio.wait_for(
            asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=1, **DEFAULT_POOL_CONFIG),
            timeout=5,
        )
    except Exception as e:
        pytest.skip(f"Postgres not reachable at POSTGRES_DSN: {e}")
    yield pg_pool
    await pg_pool.close()


async def test_open_transaction_is_rolled_back_on_release(pool):
    async with pool.acquire() as conn:
        await conn.execute("BEGIN")
        await conn.execute("SELECT set_config('tenxgraph.tenant', 'acme', true)")
    async with pool.acquire() as conn:
        assert not conn.is_in_transaction()
        assert await conn.fetchval("SELECT current_setting('tenxgraph.tenant', true)") in ("", None)


async def test_rls_settings_do_not_reach_the_next_checkout(pool):
    seen = await run_on_connection(
        pool,
        lambda conn: conn.fetchval("SELECT current_setting('tenxgraph.tenant', true)"),
        timeout=5,
        settings=[("tenxgraph.tenant", "acme")],
    )
    assert seen == "acme"
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT current_setting('tenxgraph.tenant', true)") in ("", None)


async def test_session_setting_is_kept_so_the_reset_query_really_is_skipped(pool):
    # The contract: session-level state is not cleared, so backends must never set any.
    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('tenxgraph.probe', 'kept', false)")
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT current_setting('tenxgraph.probe', true)") == "kept"
        await conn.execute("RESET tenxgraph.probe")


async def test_ddl_session_leaves_no_lock_or_lock_timeout(pool):
    async with pool.acquire() as conn:
        default_timeout = await conn.fetchval("SHOW lock_timeout")
        async with ddl_session(conn, lock_key=LOCK_KEY, timeout=1.0, owner="test"):
            assert await conn.fetchval("SHOW lock_timeout") == "1s"
    async with pool.acquire() as conn:
        assert await conn.fetchval("SHOW lock_timeout") == default_timeout
        held = await conn.fetchval(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
        )
        assert held == 0
