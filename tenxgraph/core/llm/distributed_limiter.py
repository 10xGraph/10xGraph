"""Opt-in, cross-process cap on in-flight LLM calls, backed by Redis.

The per-Agent ``max_concurrent_calls`` cap only counts calls inside one process. When
several workers share one provider key, this limiter shares the budget through a Redis
sorted set keyed by provider and model. Each in-flight call holds a *lease*: a member whose
score is its expiry time. A crashed or cancelled process therefore cannot leak a slot
forever, because expired members are purged by the next acquire.

It fails open: if Redis cannot be reached, a WARNING is logged and the call proceeds
without a slot, so a Redis outage never blocks runs. A running call renews its lease every
third of the lease time, so the lease only has to outlast a crashed process, not the slowest
call.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from typing import Any


logger = logging.getLogger("tenxgraph.llm.distributed_limiter")

# Purge expired leases, then take a slot if one is free. Uses the Redis server clock so
# processes with skewed clocks agree. Returns 1 when the slot was taken.
_ACQUIRE_LUA = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[1]) then
    redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[3])
    redis.call('PEXPIRE', KEYS[1], ARGV[2])
    return 1
end
return 0
"""

# Push the lease of a slot this process still holds. ZADD XX only updates a member that is
# still there, so a lease that already expired and was purged is not brought back over the cap.
_RENEW_LUA = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
if redis.call('ZADD', KEYS[1], 'XX', 'CH', now + tonumber(ARGV[1]), ARGV[2]) > 0 then
    redis.call('PEXPIRE', KEYS[1], ARGV[1])
    return 1
end
return 0
"""

_WARN_EVERY_SECONDS = 30.0


class _Slot:
    """Async context manager holding one lease (or nothing, when failing open)."""

    def __init__(self, limiter: DistributedLimiter, token: str | None) -> None:
        self._limiter = limiter
        self._token = token
        self._heartbeat: asyncio.Task[None] | None = None
        if token is not None:
            # A call can outlive the lease; renew it every lease/3 while the call runs.
            self._heartbeat = asyncio.ensure_future(limiter._renew_loop(token))

    async def __aenter__(self) -> _Slot:
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
            self._heartbeat = None
        if self._token is not None:
            await self._limiter._release(self._token)


class DistributedLimiter:
    """Redis-backed semaphore for one (provider, model) pair.

    Args:
        redis: An async Redis client, or ``None`` to build one lazily from ``redis_url``.
        redis_url: Connection URL used when ``redis`` is not given.
        key: Full Redis key for this pair (build it with :func:`limiter_key`).
        limit: Maximum concurrent calls across all processes.
        lease_seconds: How long a slot is held before it is considered abandoned.
    """

    def __init__(
        self,
        *,
        key: str,
        limit: int,
        lease_seconds: float,
        redis: Any | None = None,
        redis_url: str | None = None,
    ) -> None:
        self.key = key
        self.limit = limit
        self.lease_ms = max(1, int(lease_seconds * 1000))
        self._redis = redis
        self._redis_url = redis_url
        self._last_warn = float("-inf")

    def _client(self) -> Any:
        if self._redis is None:
            if not self._redis_url:
                raise ValueError("no redis client or redis_url configured")
            from redis.asyncio import Redis  # (optional dependency)

            self._redis = Redis.from_url(self._redis_url)
        return self._redis

    def _warn(self, what: str, exc: BaseException) -> None:
        now = time.monotonic()
        if now - self._last_warn >= _WARN_EVERY_SECONDS:
            self._last_warn = now
            logger.warning(
                "Distributed LLM limiter %s failed (%s: %s); proceeding without a shared slot",
                what,
                type(exc).__name__,
                exc,
            )

    def acquire(self) -> _SlotAcquire:
        """Return an awaitable-context: ``async with limiter.acquire(): ...``."""
        return _SlotAcquire(self)

    async def _acquire(self) -> _Slot:
        token = uuid.uuid4().hex
        delay = 0.05
        while True:
            try:
                got = await self._client().eval(
                    _ACQUIRE_LUA, 1, self.key, self.limit, self.lease_ms, token
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._warn("acquire", exc)
                return _Slot(self, None)
            if got:
                return _Slot(self, token)
            await asyncio.sleep(delay * (0.5 + random.random()))  # noqa: S311
            delay = min(delay * 2, 0.5)

    async def _renew_loop(self, token: str) -> None:
        """Keep the lease alive until cancelled. Redis errors are logged and never raised."""
        interval = self.lease_ms / 3000.0
        while True:
            await asyncio.sleep(interval)
            try:
                await self._client().eval(_RENEW_LUA, 1, self.key, self.lease_ms, token)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # fail open: the next tick tries again
                self._warn("renew", exc)

    async def _release(self, token: str) -> None:
        try:
            await self._client().zrem(self.key, token)
        except Exception as exc:  # the lease expiry reclaims the slot anyway
            self._warn("release", exc)


class _SlotAcquire:
    def __init__(self, limiter: DistributedLimiter) -> None:
        self._limiter = limiter
        self._slot: _Slot | None = None

    async def __aenter__(self) -> _Slot:
        self._slot = await self._limiter._acquire()
        return self._slot

    async def __aexit__(self, *exc: object) -> None:
        if self._slot is not None:
            await self._slot.__aexit__(*exc)


def limiter_key(prefix: str, provider: str, model: str) -> str:
    """Redis key for one (provider, model) budget; ``prefix`` follows ``redis_key_prefix``."""
    return f"{prefix}llm_slots:{provider}:{model}"
