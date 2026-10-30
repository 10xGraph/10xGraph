"""The run-marker compare-and-delete and put-if-absent against real backends.

``test_run_marker_cas.py`` drives the Redis calls through a fake; here a real Redis runs the
Lua script and ``SET NX``, and a real file-backed SQLite database runs the one-statement
delete and insert. The Redis tests are ``integration`` tests (run with ``--integration``): they
use ``REDIS_URL`` or redis://localhost:6379, else start ``redis:7-alpine`` with docker on a
free port, else skip.
"""

import asyncio
import contextlib
import os
import shutil
import socket
import subprocess
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from tenxgraph.storage.checkpointer.base_checkpointer import ACTIVE_RUN_NAMESPACE


try:
    import redis.asyncio as redis_asyncio
except ImportError:  # the redis extra is not installed: the Redis tests skip
    redis_asyncio = None


def _cfg(run_id: str | None = None, thread_id: str = "t1") -> dict:
    cfg = {"thread_id": thread_id, "user_id": "u1"}
    if run_id:
        cfg["run_id"] = run_id
    return cfg


def _key(cp, cfg: dict) -> str:
    return cp._stop_request_key(cfg)


# --- Redis ----------------------------------------------------------------------------------


async def _ping(url: str) -> bool:
    client = redis_asyncio.from_url(url, socket_connect_timeout=1)
    try:
        return bool(await client.ping())
    except Exception:
        return False
    finally:
        await client.aclose()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture
async def redis_url():
    if redis_asyncio is None:
        pytest.skip("redis is not installed")
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    if await _ping(url):
        yield url
        return
    if shutil.which("docker") is None:
        pytest.skip("no Redis reachable and no docker")
    port = _free_port()
    try:
        started = subprocess.run(  # noqa: S603
            ["docker", "run", "-d", "--rm", "-p", f"127.0.0.1:{port}:6379", "redis:7-alpine"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"could not start redis:7-alpine ({e})")
    if started.returncode != 0:
        pytest.skip(f"could not start redis:7-alpine: {started.stderr.strip()}")
    container = started.stdout.strip()
    url = f"redis://127.0.0.1:{port}/0"
    try:
        deadline = time.monotonic() + 15
        while not await _ping(url):
            if time.monotonic() > deadline:
                pytest.skip("redis:7-alpine did not answer in time")
            await asyncio.sleep(0.2)
        yield url
    finally:
        with contextlib.suppress(Exception):
            subprocess.run(  # noqa: S603
                ["docker", "stop", container],  # noqa: S607
                capture_output=True,
                timeout=30,
                check=False,
            )


@pytest_asyncio.fixture
async def pg_on_real_redis(redis_url):
    """A PgCheckpointer whose generic cache (the run marker) is a real Redis; no Postgres."""
    from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer

    client = redis_asyncio.from_url(redis_url)
    with patch("asyncpg.create_pool", return_value=MagicMock()):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis=client,
            cache_ttl=3600,
        )
    try:
        yield cp
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_redis_clear_if_runs_the_lua_script(pg_on_real_redis):
    cp = pg_on_real_redis
    cfg = _cfg(thread_id=f"t-{uuid.uuid4()}")
    key = _key(cp, cfg)
    await cp.aput_cache_value(ACTIVE_RUN_NAMESPACE, key, "r2", 60)

    assert await cp.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, key, "r1") is False
    assert await cp.aget_active_run(cfg) == "r2"
    assert await cp.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, key, "r2") is True
    assert await cp.aget_active_run(cfg) is None
    assert await cp.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, key, "r2") is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_redis_put_if_absent_and_marker_lifecycle(pg_on_real_redis):
    cp = pg_on_real_redis
    cfg = _cfg(thread_id=f"t-{uuid.uuid4()}")
    key = _key(cp, cfg)

    assert await cp.aput_cache_value_if_absent(ACTIVE_RUN_NAMESPACE, key, "r1", 60) is True
    assert await cp.aput_cache_value_if_absent(ACTIVE_RUN_NAMESPACE, key, "r2", 60) is False
    assert 0 < await cp.redis.ttl(cp._get_generic_cache_key(ACTIVE_RUN_NAMESPACE, key)) <= 60

    await cp.abegin_run(_cfg("r2", cfg["thread_id"]))  # r2 takes the thread
    await cp.arefresh_run(_cfg("r1", cfg["thread_id"]))  # r1's heartbeat leaves it
    await cp.aend_run(_cfg("r1", cfg["thread_id"]))  # and so does r1's release
    assert await cp.aget_active_run(cfg) == "r2"
    await cp.aend_run(_cfg("r2", cfg["thread_id"]))
    assert await cp.aget_active_run(cfg) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_redis_put_if_absent_has_one_winner(pg_on_real_redis):
    cp = pg_on_real_redis
    key = _key(cp, _cfg(thread_id=f"t-{uuid.uuid4()}"))

    results = await asyncio.gather(
        *(cp.aput_cache_value_if_absent(ACTIVE_RUN_NAMESPACE, key, f"r{i}", 60) for i in range(20))
    )

    assert results.count(True) == 1
    assert await cp.aget_cache_value(ACTIVE_RUN_NAMESPACE, key) == f"r{results.index(True)}"
    await cp.aclear_cache_value(ACTIVE_RUN_NAMESPACE, key)


# --- SQLite (a real database file) ----------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_pair(tmp_path):
    """Two checkpointers (two connections) on one database file."""
    pytest.importorskip("aiosqlite")
    from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer

    path = tmp_path / "markers.db"
    first, second = SqliteCheckpointer(path), SqliteCheckpointer(path)
    try:
        yield first, second
    finally:
        await first.arelease()
        await second.arelease()


@pytest.mark.asyncio
async def test_sqlite_file_clear_if_is_one_statement(sqlite_pair):
    first, second = sqlite_pair
    cfg = _cfg()
    key = _key(first, cfg)
    await first.aput_cache_value(ACTIVE_RUN_NAMESPACE, key, "r2", 60)

    assert await second.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, key, "r1") is False
    assert await first.aget_active_run(cfg) == "r2"
    assert await second.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, key, "r2") is True
    assert await first.aget_active_run(cfg) is None


@pytest.mark.asyncio
async def test_sqlite_file_put_if_absent_has_one_winner(sqlite_pair):
    # Concurrent writers share one checkpointer: it keeps one connection per process, and two
    # connections writing at once in one event loop only wait on SQLite's file lock.
    first, second = sqlite_pair
    key = _key(first, _cfg())

    results = await asyncio.gather(
        *(first.aput_cache_value_if_absent(ACTIVE_RUN_NAMESPACE, key, f"r{i}", 60) for i in range(20))
    )
    assert await second.aput_cache_value_if_absent(ACTIVE_RUN_NAMESPACE, key, "late", 60) is False

    assert results.count(True) == 1
    assert await second.aget_cache_value(ACTIVE_RUN_NAMESPACE, key) == f"r{results.index(True)}"


@pytest.mark.asyncio
async def test_sqlite_file_marker_lifecycle_across_connections(sqlite_pair):
    first, second = sqlite_pair
    cfg = _cfg()

    await first.abegin_run(_cfg("r1"))
    await second.abegin_run(_cfg("r2"))  # r2 takes the thread
    await first.arefresh_run(_cfg("r1"))
    await first.aend_run(_cfg("r1"))
    assert await second.aget_active_run(cfg) == "r2"

    await second.aclear_cache_value(ACTIVE_RUN_NAMESPACE, _key(first, cfg))
    await second.arefresh_run(_cfg("r2"))  # a lost marker of the live run comes back
    assert await first.aget_active_run(cfg) == "r2"
