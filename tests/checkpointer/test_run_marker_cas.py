"""The active-run marker is compared and deleted in one step; the heartbeat restores its own.

A run that ends must not delete the marker of a newer run on the same thread, even when that
run writes its marker between the old run's read and delete.
"""

from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.base_checkpointer import (
    ACTIVE_RUN_NAMESPACE,
    STOP_REQUEST_NAMESPACE,
)
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer


def _cfg(run_id: str | None = None) -> dict:
    cfg = {"thread_id": "t1", "user_id": "u1"}
    if run_id:
        cfg["run_id"] = run_id
    return cfg


class _FakeRedis:
    """Dict-backed stand-in for the redis calls the generic cache makes, Lua included.

    ``eval`` runs the compare-and-delete script's logic; it shows the wiring, not that Redis
    runs it atomically (a real Redis does: a script is one command).
    """

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.scripts: list[str] = []

    async def setex(self, key, ttl, value):
        self.data[key] = value

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        return 1 if self.data.pop(key, None) is not None else 0

    async def eval(self, script, numkeys, key, expected):
        self.scripts.append(script)
        if self.data.get(key) == expected:
            del self.data[key]
            return 1
        return 0


def _pg() -> PgCheckpointer:
    with patch("asyncpg.create_pool", return_value=MagicMock()):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            cache_ttl=3600,
        )
    cp.redis = _FakeRedis()
    return cp


@pytest.fixture(params=["memory", "sqlite", "pg"])
def cp(request):
    if request.param == "memory":
        return InMemoryCheckpointer()
    if request.param == "sqlite":
        return SqliteCheckpointer(":memory:")
    return _pg()


def _key(cp) -> str:
    return cp._stop_request_key(_cfg())


@pytest.mark.asyncio
async def test_clear_if_deletes_only_the_expected_value(cp):
    await cp.aput_cache_value(ACTIVE_RUN_NAMESPACE, _key(cp), "r2", 60)

    assert await cp.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, _key(cp), "r1") is False
    assert await cp.aget_active_run(_cfg()) == "r2"
    assert await cp.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, _key(cp), "r2") is True
    assert await cp.aget_active_run(_cfg()) is None
    assert await cp.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, _key(cp), "r2") is False


@pytest.mark.asyncio
async def test_end_run_keeps_a_marker_written_after_its_check(cp, monkeypatch):
    await cp.abegin_run(_cfg("r1"))
    original_get = cp.aget_cache_value
    original_clear_if = getattr(cp, "aclear_cache_value_if", None)
    fired = []

    async def r2_starts(key):
        # r2 takes the thread at the first point r1's release touches the marker.
        if not fired:
            fired.append(True)
            await cp.aput_cache_value(ACTIVE_RUN_NAMESPACE, key, "r2", 60)

    async def read_then_r2_starts(namespace, key):
        value = await original_get(namespace, key)
        if namespace == ACTIVE_RUN_NAMESPACE:
            await r2_starts(key)
        return value

    async def r2_starts_then_clear_if(namespace, key, expected):
        if namespace == ACTIVE_RUN_NAMESPACE:
            await r2_starts(key)
        return await original_clear_if(namespace, key, expected)

    monkeypatch.setattr(cp, "aget_cache_value", read_then_r2_starts)
    monkeypatch.setattr(cp, "aclear_cache_value_if", r2_starts_then_clear_if, raising=False)
    await cp.aend_run(_cfg("r1"))
    monkeypatch.undo()

    assert fired
    assert await cp.aget_active_run(_cfg()) == "r2"


@pytest.mark.asyncio
async def test_end_run_keeps_a_stop_request_for_the_next_run(cp, monkeypatch):
    await cp.abegin_run(_cfg("r1"))
    await cp.arequest_stop(_cfg())  # targets r1
    original = cp.aget_cache_value

    async def read_then_stop_r2(namespace, key):
        value = await original(namespace, key)
        if namespace == STOP_REQUEST_NAMESPACE and value == {"run_id": "r1"}:
            await cp.aput_cache_value(STOP_REQUEST_NAMESPACE, key, {"run_id": "r2"}, 60)
        return value

    monkeypatch.setattr(cp, "aget_cache_value", read_then_stop_r2)
    await cp.aend_run(_cfg("r1"))
    monkeypatch.undo()

    assert await cp.aget_cache_value(STOP_REQUEST_NAMESPACE, _key(cp)) == {"run_id": "r2"}


@pytest.mark.asyncio
async def test_end_run_clears_its_own_marker_and_stop(cp):
    await cp.abegin_run(_cfg("r1"))
    await cp.arequest_stop(_cfg())

    await cp.aend_run(_cfg("r1"))

    assert await cp.aget_active_run(_cfg()) is None
    assert await cp.aget_cache_value(STOP_REQUEST_NAMESPACE, _key(cp)) is None


@pytest.mark.asyncio
async def test_heartbeat_restores_a_missing_marker_of_the_live_run(cp):
    await cp.abegin_run(_cfg("r1"))
    await cp.aclear_cache_value(ACTIVE_RUN_NAMESPACE, _key(cp))  # deleted by mistake

    await cp.arefresh_run(_cfg("r1"))

    assert await cp.aget_active_run(_cfg()) == "r1"


@pytest.mark.asyncio
async def test_heartbeat_leaves_another_runs_marker(cp):
    await cp.abegin_run(_cfg("r1"))
    await cp.abegin_run(_cfg("r2"))

    await cp.arefresh_run(_cfg("r1"))

    assert await cp.aget_active_run(_cfg()) == "r2"


@pytest.mark.asyncio
async def test_pg_clear_if_is_one_redis_script():
    cp = _pg()
    await cp.aput_cache_value(ACTIVE_RUN_NAMESPACE, _key(cp), "r1", 60)

    assert await cp.aclear_cache_value_if(ACTIVE_RUN_NAMESPACE, _key(cp), "r1") is True
    assert len(cp.redis.scripts) == 1
    assert "DEL" in cp.redis.scripts[0]


@pytest.mark.asyncio
async def test_put_if_absent_writes_only_a_missing_value(cp):
    assert await cp.aput_cache_value_if_absent(ACTIVE_RUN_NAMESPACE, _key(cp), "r1", 60) is True
    assert await cp.aput_cache_value_if_absent(ACTIVE_RUN_NAMESPACE, _key(cp), "r2", 60) is False
    assert await cp.aget_active_run(_cfg()) == "r1"


@pytest.mark.asyncio
async def test_heartbeat_keeps_a_marker_written_after_its_check(cp, monkeypatch):
    await cp.abegin_run(_cfg("r1"))
    await cp.aclear_cache_value(ACTIVE_RUN_NAMESPACE, _key(cp))  # r1's marker expired
    original_get = cp.aget_cache_value
    original_put_if_absent = cp.aput_cache_value_if_absent
    fired = []

    async def r2_starts(key):
        # r2 takes the thread at the first point r1's heartbeat touches the marker.
        if not fired:
            fired.append(True)
            await cp.aput_cache_value(ACTIVE_RUN_NAMESPACE, key, "r2", 60)

    async def read_then_r2_starts(namespace, key):
        value = await original_get(namespace, key)
        if namespace == ACTIVE_RUN_NAMESPACE:
            await r2_starts(key)
        return value

    async def r2_starts_then_put_if_absent(namespace, key, value, ttl_seconds=None):
        if namespace == ACTIVE_RUN_NAMESPACE:
            await r2_starts(key)
        return await original_put_if_absent(namespace, key, value, ttl_seconds)

    monkeypatch.setattr(cp, "aget_cache_value", read_then_r2_starts)
    monkeypatch.setattr(cp, "aput_cache_value_if_absent", r2_starts_then_put_if_absent)
    await cp.arefresh_run(_cfg("r1"))
    monkeypatch.undo()

    assert fired
    assert await cp.aget_active_run(_cfg()) == "r2"


@pytest.mark.asyncio
async def test_put_if_absent_replaces_an_expired_value(cp, monkeypatch):
    if isinstance(cp, PgCheckpointer):
        pytest.skip("Redis expires keys itself")
    await cp.aput_cache_value(ACTIVE_RUN_NAMESPACE, _key(cp), "r1", 60)
    import time

    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 120)

    assert await cp.aput_cache_value_if_absent(ACTIVE_RUN_NAMESPACE, _key(cp), "r2", 60) is True
    assert await cp.aget_active_run(_cfg()) == "r2"


@pytest.mark.asyncio
async def test_base_put_if_absent_default_uses_get_and_put():
    """A third-party checkpointer without its own put-if-absent still gets a working one."""
    from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer

    cp = InMemoryCheckpointer()
    put = BaseCheckpointer.aput_cache_value_if_absent
    assert await put(cp, ACTIVE_RUN_NAMESPACE, _key(cp), "r1", 60) is True
    assert await put(cp, ACTIVE_RUN_NAMESPACE, _key(cp), "r2", 60) is False
    assert await cp.aget_active_run(_cfg()) == "r1"
