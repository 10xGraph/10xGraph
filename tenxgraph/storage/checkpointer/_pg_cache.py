"""Redis: breaker-guarded calls, the state cache and generic cache values."""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote, unquote

from tenxgraph.utils.log_safety import describe_error

from ._pg_support import (
    _CACHE_CAS_LUA,
    _CACHE_VERSION_KEY,
    _CLEAR_IF_LUA,
    _MAX_SKIPPED_INVALIDATIONS,
    _NO_VERSION,
    _REDIS_CONNECTION_ERRORS,
    _THREAD_IDENTITY_KEY,
    STATE_VERSION_CONFIG_KEY,
    StateT,
    _RedisUnavailable,
)


logger = logging.getLogger("tenxgraph.checkpointer.pg")


class _PgCacheMixin:
    """Redis: breaker-guarded calls, the state cache and generic cache values."""

    async def _redis_call(self, call: Callable[[], Awaitable[Any]]) -> Any:
        """Run one Redis call through the circuit breaker.

        Raises :class:`_RedisUnavailable` when the breaker is open or the call fails to
        connect; callers catch it in their existing ``except`` branch and take the same
        fallback as for any Redis error, without logging it again. Other errors (a bad value,
        a script error) propagate unchanged and do not open the breaker.
        """
        breaker = self._redis_breaker
        if breaker is None:
            return await call()
        if not breaker.allow():
            raise _RedisUnavailable
        try:
            if self._skipped_invalidations:
                # Redis is reachable again: first delete the state cache entries whose
                # invalidation was skipped while it was down, before any read can serve them.
                keys = list(self._skipped_invalidations)
                await self.redis.delete(*keys)
                self._skipped_invalidations.difference_update(keys)
                logger.info(
                    "Dropped %d state cache entries invalidated during outage",
                    len(keys),
                )
            result = await call()
        except _REDIS_CONNECTION_ERRORS as e:
            breaker.record_failure(e)
            raise _RedisUnavailable from e
        except Exception:
            breaker.record_success()  # Redis answered, with an error of another kind
            raise
        breaker.record_success()
        return result

    def _get_generic_cache_key(self, namespace: str, key: str) -> str:
        """Build a Redis key for shared non-state cache entries."""
        return f"{self._generic_cache_prefix(namespace)}{quote(str(key), safe='')}"

    def _generic_cache_prefix(self, namespace: str) -> str:
        return f"{self.redis_key_prefix}generic_cache:{quote(str(namespace), safe='')}:"

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """
        Cache state in Redis with TTL.

        The durable version currently associated with the run (from
        ``config['_checkpoint_version']``) is embedded in the cached payload so a
        resume that reads from cache recovers the right version for its
        compare-and-swap. Caching remains best-effort and never raises.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            Any | None: True if cached, None if failed.
        """
        # No DB access, but keep consistent
        thread_id, user_id = self._validate_config(config)

        logger.debug("Caching state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            cache_key = self._get_thread_key(thread_id, user_id, config)
            data = self._serialize_state_payload(state)
            version = config.get(STATE_VERSION_CONFIG_KEY)
            if version is not None:
                data[_CACHE_VERSION_KEY] = version
            state_json = json.dumps(data)

            # Version-guarded write: never move the cache backwards (see
            # _CACHE_CAS_LUA). A run that is about to lose its optimistic version
            # check must not stamp its stale state over a newer committed one.
            written = await self._redis_call(
                lambda: self.redis.eval(
                    _CACHE_CAS_LUA,
                    1,
                    cache_key,
                    state_json,
                    str(version if version is not None else _NO_VERSION),
                    str(self.cache_ttl),
                    _CACHE_VERSION_KEY,
                )
            )
            if not written:
                logger.debug(
                    "Skipped stale cache write for thread_id=%s (version=%s is behind cache)",
                    thread_id,
                    version,
                )
                return None

            logger.debug("State cached with key=%s, ttl=%d", cache_key, self.cache_ttl)
            return True

        except Exception as e:
            if not isinstance(e, _RedisUnavailable):  # the breaker already logged it
                logger.error(
                    "Failed to cache state for thread_id=%s: %s", thread_id, describe_error(e)
                )
            # Don't raise - caching is optional
            return None

    async def _invalidate_state_cache(
        self,
        thread_id: str | int,
        user_id: str | int,
        retries: int = 0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Drop the cached state for a thread (best-effort).

        Called when a durable write loses its optimistic version check. The cache
        may hold state derived from the now-stale version, and because reads
        prefer the cache and seed the next write's expected version from it, a
        poisoned entry would make every later run fail its compare-and-swap.
        Dropping it forces the next read back to Postgres, so the thread heals
        itself instead of staying wedged until the TTL expires.

        ``config`` selects the key when tenant columns are declared: the thread's identity key
        once its row has been read (``_thread_identity``), plus the caller's own key, which
        may hold an entry written before the thread existed.
        """
        try:
            keys = [self._get_thread_key(thread_id, user_id, config)]
            if self._thread_identity(config) is not None:
                caller_config = {k: v for k, v in config.items() if k != _THREAD_IDENTITY_KEY}
                caller_key = self._get_thread_key(thread_id, user_id, caller_config)
                if caller_key not in keys:
                    keys.append(caller_key)
        except ValueError as e:  # a malformed tenant value: nothing of ours can be cached
            logger.warning(
                "Skipped state cache invalidation for thread_id=%s: %s",
                thread_id,
                describe_error(e),
            )
            return
        for attempt in range(retries + 1):
            try:
                await self._redis_call(lambda: self.redis.delete(*keys))
                logger.debug("Invalidated stale state cache for thread_id=%s", thread_id)
                return
            except _RedisUnavailable:
                # Redis is down: delete the entry as soon as it is reachable again (see
                # _redis_call), so it cannot be served after recovery.
                for cache_key in keys:
                    if len(self._skipped_invalidations) < _MAX_SKIPPED_INVALIDATIONS:
                        self._skipped_invalidations.add(cache_key)
                return
            except Exception as e:  # invalidation is best-effort
                if attempt >= retries:
                    logger.warning(
                        "Failed to invalidate state cache for thread_id=%s: %s",
                        thread_id,
                        e,
                    )

    async def _cache_matches_durable(
        self,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
        cached_version: int,
    ) -> bool:
        """Whether a cached state's version is the thread's current durable version.

        Every cache write follows a durable write at the same version, so a cached version that
        differs from Postgres (older after a Redis restore, or newer/absent after a delete)
        is stale. An entry cached before the first durable write counts as version 0. Costs one
        indexed ``MAX(version)`` read; if Postgres cannot be read the cache is trusted, as it
        was before this check.
        """
        # Built outside the try: a scope this caller cannot satisfy (a picked tenant column
        # without a value) must raise, never fall into "trust the cache".
        scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)
        try:
            query = (
                f"SELECT COALESCE(MAX(version), 0) FROM {self._get_table_name('states')} "  # noqa: S608
                f"WHERE {scope_sql}"
            )
            durable = await self._run_scoped(
                lambda conn: conn.fetchval(query, *scope_params),
                settings=self._rls(config),
            )
            return int(durable) == cached_version
        except Exception as e:
            logger.warning(
                "Could not verify state cache for thread_id=%s: %s", thread_id, describe_error(e)
            )
            return True

    async def _durable_version_for_cache(
        self, thread_id: str | int, user_id: str | int, config: dict[str, Any]
    ) -> int | None:
        """The thread's durable state version for a cache check (tenant columns).

        Reads the thread row under the caller's scope (``_read_thread_identity``), which also
        makes the cache key the thread's identity. A thread the caller cannot see, or that does
        not exist yet, counts as version 0 under the caller's own key, which only this same
        identity can have written. ``None`` when Postgres cannot be read: the cache is trusted,
        as ``_cache_matches_durable`` does.
        """
        self._thread_row_scope(thread_id, user_id, config)  # scope errors raise here
        try:
            row = await self._read_thread_identity(thread_id, user_id, config)
        except Exception as e:
            logger.warning(
                "Could not verify state cache for thread_id=%s: %s", thread_id, describe_error(e)
            )
            return None
        return int(row["durable_version"]) if row is not None else 0

    async def aclear_state_cache(self, config: dict[str, Any]) -> bool:
        """Drop the cached state for a thread; the Postgres state is untouched.

        The cache key includes the caller's user id (and tenant values), so one user cannot
        clear another's.
        """
        thread_id, user_id = self._validate_config(config)
        await self._invalidate_state_cache(thread_id, user_id, config=config)
        return True

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Get state from Redis cache, fallback to PostgreSQL if miss.

        On a cache hit the embedded version is restored into
        ``config['_checkpoint_version']`` so a subsequent write still performs a
        correct compare-and-swap.

        Tenant isolation holds on a hit: the key carries the caller's tenant values, and a hit
        is served only after the scoped ``MAX(version)`` read (``_cache_matches_durable``)
        agrees, so a caller outside the thread's scope always falls through to the scoped
        Postgres read.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: State object or None.
        """
        # Schema might be needed if we fall back to DB
        thread_id, user_id = self._validate_config(config)

        logger.debug("Getting cached state for thread_id=%s, user_id=%s", thread_id, user_id)

        # With tenant columns the key is the thread's identity, so the scoped durable read (the
        # thread row and its current state version) comes first. Built outside the try below:
        # a scope this caller cannot satisfy must raise, never fall back to the cache.
        durable_version: int | None = None
        if self.columns.tenant:
            durable_version = await self._durable_version_for_cache(thread_id, user_id, config)

        try:
            # Try Redis first
            cache_key = self._get_thread_key(thread_id, user_id, config)
            cached_data = await self._redis_call(lambda: self.redis.get(cache_key))

            if cached_data:
                data = json.loads(cached_data)
                version = data.get(_CACHE_VERSION_KEY)
                cached_version = int(version) if version is not None else 0
                # An entry without a version was cached before the thread's first durable
                # write, so it is current only while Postgres holds no state (version 0).
                # Served after a durable write whose cache write-through was lost (Redis
                # down), it would load an old state with no version, and the next save,
                # having nothing to compare, would overwrite the newer durable state.
                if self.columns.tenant:
                    current = durable_version is None or durable_version == cached_version
                else:
                    current = await self._cache_matches_durable(
                        thread_id, user_id, config, cached_version
                    )
                if not current:
                    # Redis came back from an older snapshot, a lagging replica, or the
                    # thread was deleted while the cache delete failed. Postgres is the
                    # source of truth: serving this would fail the run's first save with a
                    # misleading "updated by another" conflict, or bring back deleted data.
                    logger.warning(
                        "Dropping stale state cache for thread_id=%s (cached version %s does "
                        "not match Postgres)",
                        thread_id,
                        version,
                    )
                    await self._invalidate_state_cache(thread_id, user_id, config=config)
                else:
                    logger.debug("Cache hit for thread_id=%s", thread_id)
                    if version is not None:
                        config[STATE_VERSION_CONFIG_KEY] = int(version)
                    return self._deserialize_state_payload(data, thread_id)

            # Cache miss - fallback to PostgreSQL
            logger.debug("Cache miss for thread_id=%s, falling back to PostgreSQL", thread_id)
            state = await self.aget_state(config)

            # Cache the result for next time
            if state:
                await self.aput_state_cache(config, state)

            return state

        except Exception as e:
            if not isinstance(e, _RedisUnavailable):  # the breaker already logged it
                logger.error(
                    "Failed to get cached state for thread_id=%s: %s", thread_id, describe_error(e)
                )
            # Fallback to PostgreSQL on error
            return await self.aget_state(config)

    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        """Cache a generic JSON-serializable value in Redis."""
        try:
            cache_key = self._get_generic_cache_key(namespace, key)
            ttl = ttl_seconds or self.cache_ttl
            payload = json.dumps(value)
            await self._redis_call(lambda: self.redis.setex(cache_key, ttl, payload))
            return True
        except _RedisUnavailable:
            return None
        except Exception as e:
            logger.error(
                "Failed to cache value for namespace=%s key=%s: %s",
                namespace,
                key,
                describe_error(e),
            )
            return None

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        """Retrieve a generic cache value from Redis."""
        try:
            cache_key = self._get_generic_cache_key(namespace, key)
            cached_data = await self._redis_call(lambda: self.redis.get(cache_key))
            if not cached_data:
                return None
            return json.loads(cached_data)
        except _RedisUnavailable:
            return None
        except Exception as e:
            logger.error(
                "Failed to get cached value for namespace=%s key=%s: %s",
                namespace,
                key,
                e,
            )
            return None

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        """Delete a generic cache value from Redis."""
        try:
            cache_key = self._get_generic_cache_key(namespace, key)
            return await self._redis_call(lambda: self.redis.delete(cache_key))
        except _RedisUnavailable:
            return None
        except Exception as e:
            logger.error(
                "Failed to clear cached value for namespace=%s key=%s: %s",
                namespace,
                key,
                e,
            )
            return None

    async def aclear_cache_value_if(self, namespace: str, key: str, expected: Any) -> bool:
        """Delete a generic cache value only if it still equals ``expected`` (one Lua call)."""
        try:
            cache_key = self._get_generic_cache_key(namespace, key)
            payload = json.dumps(expected)
            deleted = await self._redis_call(
                lambda: self.redis.eval(_CLEAR_IF_LUA, 1, cache_key, payload)
            )
            return bool(deleted)
        except _RedisUnavailable:
            return False
        except Exception as e:
            logger.error(
                "Failed to clear cached value for namespace=%s key=%s: %s",
                namespace,
                key,
                e,
            )
            return False

    async def aput_cache_value_if_absent(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Cache a generic value only if the key is not set (one ``SET NX``); True when set."""
        try:
            cache_key = self._get_generic_cache_key(namespace, key)
            ttl = ttl_seconds or self.cache_ttl
            payload = json.dumps(value)
            return bool(
                await self._redis_call(lambda: self.redis.set(cache_key, payload, ex=ttl, nx=True))
            )
        except _RedisUnavailable:
            return False
        except Exception as e:
            logger.error(
                "Failed to cache value for namespace=%s key=%s: %s",
                namespace,
                key,
                describe_error(e),
            )
            return False

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        """List all cache keys for a namespace using Redis SCAN."""
        try:
            ns_prefix = self._generic_cache_prefix(namespace)
            # Ids are percent-encoded, which leaves no glob characters in them; the
            # namespace prefix may still carry the deployment prefix, so escape that.
            pattern = (
                re.sub(r"([\\*?\[\]])", r"\\\1", ns_prefix) + quote(prefix or "", safe="") + "*"
            )
            keys: list[str] = []
            cursor = 0
            while True:
                cursor, found = await self._redis_call(
                    lambda c=cursor: self.redis.scan(cursor=c, match=pattern, count=100)
                )
                for raw_key in found:
                    key_str = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                    # Strip the namespace prefix to return just the key part
                    if key_str.startswith(ns_prefix):
                        keys.append(unquote(key_str[len(ns_prefix) :]))
                if cursor == 0:
                    break
            return keys
        except _RedisUnavailable:
            return []
        except Exception as e:
            logger.error(
                "Failed to list cache keys for namespace=%s: %s",
                namespace,
                e,
            )
            return []
