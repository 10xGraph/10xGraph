"""PgCheckpointer skips Redis for a short cool-down after a connection failure.

Redis is a cache; Postgres is the source of truth. With Redis down each Redis call waited for
the connect timeout (seconds) before its fallback, and a run makes several per step, so an
11 s run took 84 s. After a connection failure the breaker opens: Redis calls are skipped and
take the same fallback at once until the cool-down ends, then the next call probes Redis.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer import pg_checkpointer as pg
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer


LOGGER = "tenxgraph.checkpointer.pg"
CFG = {"thread_id": "t1", "user_id": "u1"}

CONNECTION_ERRORS = [
    pytest.param(RedisConnectionError("Error -2 connecting to redis:6379"), id="redis-conn"),
    pytest.param(RedisTimeoutError("Timeout connecting to server"), id="redis-timeout"),
    pytest.param(OSError("network unreachable"), id="oserror"),
    pytest.param(TimeoutError(), id="asyncio-timeout"),  # asyncio.TimeoutError is TimeoutError
]


class _FakeRedis:
    """Dict-backed Redis with call counts; ``fail`` makes every command raise it."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.fail: BaseException | None = None
        self.calls: list[str] = []

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail is not None:
            raise self.fail

    async def get(self, key):
        self._call("get")
        return self.data.get(key)

    async def setex(self, key, ttl, value):
        self._call("setex")
        self.data[key] = value

    async def set(self, key, value, ex=None, nx=False):
        self._call("set")
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def delete(self, *keys):
        self._call("delete")
        return sum(1 for k in keys if self.data.pop(k, None) is not None)

    async def eval(self, script, numkeys, key, *args):
        self._call("eval")
        if script == pg._CACHE_CAS_LUA:
            self.data[key] = args[0]
            return 1
        if self.data.get(key) == args[0]:
            del self.data[key]
            return 1
        return 0

    async def scan(self, cursor=0, match=None, count=None):
        self._call("scan")
        return 0, [k for k in self.data if k.startswith(match.rstrip("*"))]

    async def aclose(self):
        self.calls.append("aclose")


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(pg, "_monotonic", lambda: now[0])
    return now


def _cp(**kwargs) -> tuple[PgCheckpointer, _FakeRedis]:
    with patch("asyncpg.create_pool", return_value=MagicMock()):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            **kwargs,
        )
    fake = _FakeRedis()
    cp.redis = fake
    return cp, fake


def _cfg() -> dict:
    return dict(CFG)


async def _open(cp: PgCheckpointer, fake: _FakeRedis, exc: BaseException) -> None:
    """Open the breaker with one failing call, then let Redis come back (still skipped)."""
    fake.fail = exc
    assert await cp.aget_cache_value("ns", "k") is None
    fake.fail = None
    fake.calls.clear()


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level]


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", CONNECTION_ERRORS)
async def test_connection_error_opens_the_breaker_and_later_calls_skip_redis(clock, exc):
    cp, fake = _cp()
    fake.data[cp._get_generic_cache_key("ns", "k")] = json.dumps("v")
    await _open(cp, fake, exc)

    # Redis is answering again, but the breaker is open: nothing reaches it.
    assert await cp.aget_cache_value("ns", "k") is None
    assert fake.calls == []


@pytest.mark.asyncio
async def test_every_redis_method_returns_its_usual_fallback_while_open(clock):
    cp, fake = _cp()
    durable = AgentState()
    cp.aget_state = AsyncMock(return_value=durable)
    await _open(cp, fake, RedisConnectionError("down"))

    assert await cp.aput_state_cache(_cfg(), AgentState()) is None
    assert await cp.aget_state_cache(_cfg()) is durable  # falls back to Postgres
    assert await cp.aput_cache_value("ns", "k", 1) is None
    assert await cp.aget_cache_value("ns", "k") is None
    assert await cp.aclear_cache_value("ns", "k") is None
    assert await cp.aclear_cache_value_if("ns", "k", 1) is False
    assert await cp.aput_cache_value_if_absent("ns", "k", 1) is False
    assert await cp.alist_cache_keys("ns") == []
    assert await cp.aclear_state_cache(_cfg()) is True
    await cp.arefresh_run({**_cfg(), "run_id": "r1"})
    await cp.aend_run({**_cfg(), "run_id": "r1"})
    assert await cp.ais_stop_requested({**_cfg(), "run_id": "r1"}) is False

    assert fake.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [pytest.param(ResponseError("NOSCRIPT"), id="response-error"), pytest.param(ValueError("x"))],
)
async def test_a_non_connection_error_does_not_open_the_breaker(clock, exc, caplog):
    cp, fake = _cp()
    fake.fail = exc
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert await cp.aget_cache_value("ns", "k") is None
        assert await cp.aget_cache_value("ns", "k") is None

    assert fake.calls == ["get", "get"]
    assert cp._redis_breaker.open_until is None
    assert len(_records(caplog, logging.ERROR)) == 2  # logged per call, as before
    assert _records(caplog, logging.WARNING) == []


@pytest.mark.asyncio
async def test_after_the_cooldown_a_successful_probe_closes_the_breaker(clock, caplog):
    cp, fake = _cp(redis_breaker_cooldown=5.0)
    await _open(cp, fake, RedisConnectionError("down"))

    clock[0] += 4.9
    assert await cp.aget_cache_value("ns", "k") is None
    assert fake.calls == []

    clock[0] += 0.1
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert await cp.aput_cache_value("ns", "k", "v") is True  # the probe
        assert await cp.aget_cache_value("ns", "k") == "v"

    assert fake.calls == ["setex", "get"]
    assert cp._redis_breaker.open_until is None
    assert [r.getMessage() for r in _records(caplog, logging.INFO)] == ["Redis reachable again"]


@pytest.mark.asyncio
async def test_a_failed_probe_reopens_the_breaker(clock, caplog):
    cp, fake = _cp(redis_breaker_cooldown=5.0)
    await _open(cp, fake, RedisConnectionError("down"))

    clock[0] += 5.0
    fake.fail = RedisConnectionError("still down")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert await cp.aget_cache_value("ns", "k") is None  # the probe fails
        assert await cp.aget_cache_value("ns", "k") is None  # skipped again

    assert fake.calls == ["get"]
    assert cp._redis_breaker.open_until == clock[0] + 5.0
    assert len(_records(caplog, logging.WARNING)) == 1  # a new open period, one warning
    assert _records(caplog, logging.ERROR) == []


@pytest.mark.asyncio
async def test_cooldown_zero_disables_the_breaker(clock, caplog):
    cp, fake = _cp(redis_breaker_cooldown=0)
    assert cp._redis_breaker is None
    fake.fail = RedisConnectionError("down")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        for _ in range(3):
            assert await cp.aget_cache_value("ns", "k") is None

    assert fake.calls == ["get", "get", "get"]
    assert len(_records(caplog, logging.ERROR)) == 3  # today's behaviour
    assert _records(caplog, logging.WARNING) == []


@pytest.mark.asyncio
async def test_one_warning_per_open_period_not_per_call(clock, caplog):
    cp, fake = _cp()
    cp.aget_state = AsyncMock(return_value=None)
    fake.fail = RedisConnectionError("down")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        # Calls already in flight when the breaker opens fail too; they must not re-log.
        await asyncio.gather(*(cp.aget_cache_value("ns", f"k{i}") for i in range(5)))
        for _ in range(10):
            await cp.aget_cache_value("ns", "k")
            await cp.aput_state_cache(_cfg(), AgentState())
            await cp.aget_state_cache(_cfg())

    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "Redis unavailable" in warnings[0].getMessage()
    assert "skipping cache for 5.0 s" in warnings[0].getMessage()
    assert _records(caplog, logging.ERROR) == []


@pytest.mark.asyncio
async def test_unversioned_cache_entry_is_not_served_over_a_newer_durable_state(clock):
    """A new thread's first run caches its state with no version; if the write-through of its
    durable save is lost (Redis down), that entry must not be served once Redis is back."""
    cp, fake = _cp()
    await cp.aput_state_cache(_cfg(), AgentState())  # no version: before any durable write
    key = cp._get_thread_key("t1", "u1")
    assert pg._CACHE_VERSION_KEY not in json.loads(fake.data[key])

    durable = AgentState()
    cp.aget_state = AsyncMock(return_value=durable)
    cp._run_query = AsyncMock(return_value=1)  # Postgres has moved on to version 1

    assert await cp.aget_state_cache(_cfg()) is durable
    assert "delete" in fake.calls  # the stale entry was dropped


@pytest.mark.asyncio
async def test_unversioned_cache_entry_is_served_while_postgres_has_no_state(clock):
    cp, fake = _cp()
    await cp.aput_state_cache(_cfg(), AgentState())
    cp.aget_state = AsyncMock(side_effect=AssertionError("must be a cache hit"))
    cp._run_query = AsyncMock(return_value=0)

    assert isinstance(await cp.aget_state_cache(_cfg()), AgentState)


@pytest.mark.asyncio
async def test_versioned_entry_left_behind_by_a_skipped_write_through_is_dropped(clock):
    cp, fake = _cp()
    await cp.aput_state_cache({**_cfg(), pg.STATE_VERSION_CONFIG_KEY: 1}, AgentState())
    await _open(cp, fake, RedisConnectionError("down"))
    # A durable save to version 2 happens now; its cache write-through is skipped.
    v2 = {**_cfg(), pg.STATE_VERSION_CONFIG_KEY: 2}
    assert await cp.aput_state_cache(v2, AgentState()) is None

    clock[0] += 5.0
    durable = AgentState()
    cp.aget_state = AsyncMock(return_value=durable)
    cp._run_query = AsyncMock(return_value=2)

    assert await cp.aget_state_cache(_cfg()) is durable


@pytest.mark.asyncio
async def test_skipped_invalidation_is_applied_before_anything_is_read_after_recovery(clock):
    """A thread deleted (or cleared) while Redis was down could be re-created up to the cached
    version during the outage; the version check alone would then serve the deleted state."""
    cp, fake = _cp()
    await cp.aput_state_cache({**_cfg(), pg.STATE_VERSION_CONFIG_KEY: 1}, AgentState())
    key = cp._get_thread_key("t1", "u1")
    await _open(cp, fake, RedisConnectionError("down"))

    await cp.aclear_state_cache(_cfg())  # skipped: remembered instead
    assert key in fake.data
    assert cp._skipped_invalidations == {key}

    clock[0] += 5.0
    durable = AgentState()
    cp.aget_state = AsyncMock(return_value=durable)
    cp._run_query = AsyncMock(return_value=1)  # same version as the cached entry

    assert await cp.aget_state_cache(_cfg()) is durable
    assert fake.calls[:2] == ["delete", "get"]  # deleted before the read
    assert cp._skipped_invalidations == set()


@pytest.mark.asyncio
async def test_skipped_invalidations_are_kept_when_the_probe_fails(clock):
    cp, fake = _cp()
    await _open(cp, fake, RedisConnectionError("down"))
    await cp.aclear_state_cache(_cfg())

    clock[0] += 5.0
    fake.fail = RedisConnectionError("still down")
    assert await cp.aget_cache_value("ns", "k") is None

    assert cp._skipped_invalidations == {cp._get_thread_key("t1", "u1")}


@pytest.mark.asyncio
async def test_release_closes_redis_while_the_breaker_is_open(clock):
    cp, fake = _cp()
    await _open(cp, fake, RedisConnectionError("down"))

    await cp.arelease()

    assert "aclose" in fake.calls
