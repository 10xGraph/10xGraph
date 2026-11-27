"""``redis_key_prefix`` covers every Redis key PgCheckpointer writes.

The keys are the state cache (``state_cache:*``) and the generic cache (``generic_cache:*``),
which holds stop requests and active-run markers. Default ``""`` keeps the unprefixed names.
"""

from __future__ import annotations

import fnmatch
from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer._pg_support import _CACHE_CAS_LUA, _CLEAR_IF_LUA
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer


class _SharedRedis:
    """Dict-backed stand-in for the Redis calls PgCheckpointer makes; one per 'server'."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def setex(self, key, ttl, value):
        self.data[key] = value

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, *keys):
        return sum(1 for k in keys if self.data.pop(k, None) is not None)

    async def eval(self, script, numkeys, key, *args):
        if script == _CACHE_CAS_LUA:
            self.data[key] = args[0]
            return 1
        if script == _CLEAR_IF_LUA:
            if self.data.get(key) == args[0]:
                del self.data[key]
                return 1
            return 0
        raise AssertionError("unexpected script")

    async def scan(self, cursor=0, match="*", count=100):
        # Redis glob escapes look like fnmatch's only for the characters we escape; unescape.
        pattern = match.replace("\\", "")
        return 0, [k for k in self.data if fnmatch.fnmatchcase(k, pattern)]


def _pg(redis: _SharedRedis, **kwargs) -> PgCheckpointer:
    with patch("asyncpg.create_pool", return_value=MagicMock()):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            **kwargs,
        )
    cp.redis = redis
    return cp


CFG = {"thread_id": "t1", "user_id": "u1", "run_id": "r1"}


async def _write_every_key(cp: PgCheckpointer) -> None:
    await cp.aput_state_cache(dict(CFG), AgentState())
    await cp.abegin_run(dict(CFG))  # active-run marker
    await cp.arequest_stop(dict(CFG))  # stop request


@pytest.mark.asyncio
async def test_default_keeps_the_unprefixed_key_names():
    redis = _SharedRedis()
    await _write_every_key(_pg(redis))
    assert sorted(redis.data) == [
        "generic_cache:active_run:t1%3Au1",
        "generic_cache:stop_request:t1%3Au1",
        "state_cache:t1:u1",
    ]


@pytest.mark.asyncio
async def test_a_set_prefix_applies_to_every_key():
    redis = _SharedRedis()
    await _write_every_key(_pg(redis, redis_key_prefix="prod:"))
    assert redis.data
    assert all(key.startswith("prod:") for key in redis.data), sorted(redis.data)
    assert sorted(k.removeprefix("prod:") for k in redis.data) == [
        "generic_cache:active_run:t1%3Au1",
        "generic_cache:stop_request:t1%3Au1",
        "state_cache:t1:u1",
    ]


@pytest.mark.asyncio
async def test_two_prefixes_on_one_redis_do_not_see_each_other():
    redis = _SharedRedis()
    a = _pg(redis, redis_key_prefix="a:")
    b = _pg(redis, redis_key_prefix="b:")
    await _write_every_key(a)

    # B sees no state cache entry, active run or stop request of A's.
    assert await redis.get(b._get_thread_key("t1", "u1", dict(CFG))) is None
    assert await b.aget_active_run(dict(CFG)) is None
    assert await b.ais_stop_requested(dict(CFG)) is False
    assert await b.alist_cache_keys("stop_request") == []

    # A still sees its own, and B clearing its stop request leaves A's in place.
    assert await b.aclear_stop_request(dict(CFG)) is False
    assert await a.ais_stop_requested(dict(CFG)) is True
    assert await a.aget_active_run(dict(CFG)) == "r1"
    assert await a.alist_cache_keys("stop_request") == ["t1:u1"]
