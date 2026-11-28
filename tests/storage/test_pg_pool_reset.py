"""The pool ``reset`` in ``DEFAULT_POOL_CONFIG`` (no database).

The real Postgres behaviour is in tests/integration/test_pg_pool_reset_real.py.
"""

import inspect
from unittest.mock import AsyncMock

import pytest

from tenxgraph.storage._pg_common import DEFAULT_POOL_CONFIG


asyncpg = pytest.importorskip("asyncpg")


def test_default_pool_config_sets_a_reset_coroutine():
    assert inspect.iscoroutinefunction(DEFAULT_POOL_CONFIG["reset"])


def test_installed_asyncpg_accepts_reset():
    # ``reset`` was added in asyncpg 0.30.0; the extras pin that floor.
    assert "reset" in inspect.signature(asyncpg.create_pool).parameters


@pytest.mark.asyncio
async def test_reset_sends_no_query():
    conn = AsyncMock()
    await DEFAULT_POOL_CONFIG["reset"](conn)
    assert conn.mock_calls == []
