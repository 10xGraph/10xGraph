"""Postgres + Redis checkpointer.

The class lives here; its methods are grouped in the private ``_pg_*`` mixin modules next to
this file. Module globals read at call time (``HAS_ASYNCPG``, ``HAS_REDIS``, ``Redis``,
``asyncpg``, ``_monotonic``) stay here so tests can patch them. ``CURRENT_SCHEMA_VERSION`` is
re-exported for callers that import it from this path.
"""

import asyncio
import logging
import time
import warnings
from typing import Any

from tenxgraph.storage._pg_common import (
    DEFAULT_ACQUIRE_TIMEOUT,
    DEFAULT_DDL_LOCK_TIMEOUT,
    DEFAULT_POOL_CONFIG,
)
from tenxgraph.storage.columns import (
    ddl_lock_timeout_setting,
    parse_columns,
    quote_identifier,
)
from tenxgraph.storage.sql_filter import FilterSchema
from tenxgraph.utils.log_safety import describe_error

from ._pg_access import _PgAccessMixin
from ._pg_cache import _PgCacheMixin
from ._pg_identity import _PgIdentityMixin, _plain_id_type
from ._pg_ledger import _PgLedgerMixin
from ._pg_messages import _PgMessagesMixin
from ._pg_schema import _PgSchemaMixin
from ._pg_state import _PgStateMixin
from ._pg_support import (
    CURRENT_SCHEMA_VERSION,  # noqa: F401  (re-export: callers import it from this module path)
    DEFAULT_CACHE_TTL,
    DEFAULT_REDIS_BREAKER_COOLDOWN,
    DEFAULT_REDIS_POOL_CONFIG,
    DEFAULT_STATE_HISTORY_LIMIT,
    HAS_ASYNCPG,
    HAS_REDIS,
    THREAD_BUILTIN_COLUMNS,
    BlockingConnectionPool,
    Redis,
    StateT,
    asyncpg,
)
from ._pg_threads import _PgThreadsMixin
from .base_checkpointer import BaseCheckpointer


logger = logging.getLogger("tenxgraph.checkpointer.pg")

# Clock for the breaker. A module-level name so tests can drive it without patching the
# process-wide time.monotonic, which the event loop also reads.
_monotonic = time.monotonic


class _RedisBreaker:
    """Circuit breaker for the Redis cache: closed, or open until a monotonic deadline.

    A connection failure opens it for ``cooldown`` seconds; while open, calls are skipped.
    Once the deadline passes, calls go through again as probes: one that reaches Redis closes
    it, one that fails to connect opens it for another cool-down. It is lock-free state on one
    event loop, so concurrent probes are harmless.
    """

    def __init__(self, cooldown: float):
        self.cooldown = cooldown
        self.open_until: float | None = None  # None while closed

    def allow(self) -> bool:
        return self.open_until is None or _monotonic() >= self.open_until

    def record_failure(self, exc: BaseException) -> None:
        now = _monotonic()
        if self.open_until is not None and now < self.open_until:
            return  # already open: calls that were in flight do not extend or re-log it
        self.open_until = now + self.cooldown
        logger.warning(
            "Redis unavailable (%s: %s), skipping cache for %s s",
            type(exc).__name__,
            exc,
            self.cooldown,
        )

    def record_success(self) -> None:
        if self.open_until is not None:
            self.open_until = None
            logger.info("Redis reachable again")


class PgCheckpointer(
    _PgSchemaMixin,
    _PgIdentityMixin,
    _PgAccessMixin,
    _PgStateMixin,
    _PgCacheMixin,
    _PgLedgerMixin,
    _PgMessagesMixin,
    _PgThreadsMixin,
    BaseCheckpointer[StateT],
):
    """
    Implements a checkpointer using PostgreSQL and Redis for persistent and cached state management.

    This class provides asynchronous and synchronous methods for storing, retrieving, and managing
    agent states, messages, and threads. PostgreSQL is used for durable storage, while Redis
    provides fast caching with TTL.

    Features:
        - Async-first design with sync fallbacks
        - Configurable ID types (string, int, bigint)
        - Connection pooling for both PostgreSQL and Redis
        - Proper error handling and resource management
        - Schema migration support

    Args:
        postgres_dsn (str, optional): PostgreSQL connection string.
        pg_pool (Any, optional): Existing asyncpg Pool instance.
        pool_config (dict, optional): Configuration for new pg pool creation.
        redis_url (str, optional): Redis connection URL.
        redis (Any, optional): Existing Redis instance.
        redis_pool (Any, optional): Existing Redis ConnectionPool.
        redis_pool_config (dict, optional): Configuration for new redis pool creation.
        **kwargs: Additional configuration options:
            - user_id_type: Type for user_id fields ('string', 'int', 'bigint')
            - cache_ttl: Redis cache TTL in seconds
            - release_resources: Whether to release resources on cleanup
            - state_history_limit: State snapshots retained per thread (default 20)
            - enforce_user_isolation: Treat ``user_id`` as an ownership boundary
              (default True). When enabled, threads/state/messages are scoped to
              the requesting user, so knowing a ``thread_id`` is not enough to
              read or delete another user's data. Set to False for single-tenant
              apps or when you have no real user identity (no auth configured, or
              a placeholder user_id) -- then ``user_id`` is ignored for ownership
              and queries key on ``thread_id`` alone.
            - redis_breaker_cooldown: Seconds to skip Redis after a connection
              failure (default 5.0). Calls in that window take their cache-miss
              fallback at once instead of each waiting for the connect timeout.
              0 disables the breaker.
            - tenant_columns: Columns on ``threads`` that isolate data, such as
              ``{"org_id": "uuid", "project_id": "text"}`` (a list of names means ``text``
              columns). Set once from the caller when the thread is created and never
              changed. Every query that touches a thread (threads, states, messages, tool
              executions) is filtered through the thread row by the tenant columns the
              caller's reach picks (all of them when there is no identity).
              A picked column whose value is missing raises; it is never dropped from the
              filter. Required by default (a write without the value raises);
              ``{"type": ..., "required": False}`` opts out. Tenant columns are scalar ids:
              ``text``, ``bigint``, ``integer`` or ``uuid`` (anything else raises). Threads
              created before a column existed hold NULL there and are invisible to scopes
              that pick it until you backfill them.
            - extra_columns: Data columns on ``threads``, same format, optional by default
              (``"required": True`` opts in). Set at thread creation, never changed. Any
              Postgres type: ``text``, ``bigint``, ``integer``, ``boolean``, ``uuid`` and
              ``timestamptz`` are checked in Python and filterable with the full operator
              table; any other type (``jsonb``, ``text[]``, ``numeric(10,2)``, ``date``, an
              enum) is syntax-checked when declared and resolved on the server at setup
              (unknown types fail setup, only the server's canonical name reaches DDL).
              ``json``/``jsonb`` values are serialised with ``json.dumps``; other values go to
              asyncpg as given. Filters on those columns allow only ``is_null``. Example::

                  PgCheckpointer(
                      ...,
                      tenant_columns=["org_id"],
                      extra_columns={
                          "public_id": "text",
                          "settings": "jsonb",
                          "tags": "text[]",
                          "seats": {"type": "integer", "required": True},
                      },
                  )
            - row_level_security: Also enforce the tenant columns with Postgres native
              row-level security (default False): setup runs ``ENABLE`` + ``FORCE ROW LEVEL
              SECURITY`` on ``threads`` (policy on the tenant columns against
              ``current_setting('tenxgraph.<col>', true)``) and on ``states``, ``messages``
              and ``tool_executions`` (a row is visible only when its thread is). Every call
              then runs in a transaction that sets the values with ``set_config(..., true)``.
              Costs one transaction and one statement per call, plus a thread lookup per
              child row. Superusers and ``BYPASSRLS`` roles skip RLS (a warning is logged).
              Needs tenant columns.
            - ddl_lock_timeout: Seconds a setup DDL statement waits for a table lock
              (Postgres ``lock_timeout``) before setup fails with
              ``tenxgraph.storage.columns.SetupLockTimeoutError`` (default 10; None waits
              forever). Setup reads the catalog first and runs no DDL for anything already in
              place, so a restart takes no table lock; a missing index on an existing table is
              built with ``CREATE INDEX CONCURRENTLY`` (an invalid leftover of a failed build is
              dropped and rebuilt). Setup is serialised across processes by a session-level
              advisory lock.
            - record_tool_result: Record finished tool calls in ``tool_executions`` (default
              True), so a replayed or resumed node does not run them again. False turns the
              ledger off for every tool; ``@tool(record_tool_result=False)`` does it for one.

            Redis keys with tenant columns: the state cache, stop and active-run keys belong to
            the thread, not the caller. They are built from the thread row's own ``user_id`` and
            tenant values, read under the caller's scope, so every caller allowed to see a
            thread (its owner in another project, an org admin) shares one cache entry and one
            stop key. A cache hit is still served only after the caller's scoped durable read.

            Trusted-source rule for ``tenant_columns`` and ``extra_columns``: tenant and extra
            column values come only from ``config["user"]``. ``user`` is a reserved config key:
            clients cannot set it. Your auth class fills it (``BaseAuth.authenticate()`` result
            or JWT claims). Return ``org_id``, ``project_id``, ``public_id`` there. Never read
            them from request config, metadata, state or model output. For example::

                class MyAuth(BaseAuth):
                    async def authenticate(self, request, response, credential):
                        claims = verify(credential)  # your token check
                        return {
                            "user_id": claims["sub"],
                            "org_id": claims["org_id"],
                            "project_id": claims["project_id"],
                        }

    Raises:
        ImportError: If required dependencies are missing.
        ValueError: If required connection details are missing.
    """

    def __init__(
        self,
        # postgress connection details
        postgres_dsn: str | None = None,
        pg_pool: Any | None = None,
        pool_config: dict | None = None,
        # redis connection details
        redis_url: str | None = None,
        redis: Any | None = None,
        redis_pool: Any | None = None,
        redis_pool_config: dict | None = None,
        # database schema
        schema: str = "public",
        # other configurations - combine to reduce args
        **kwargs,
    ):
        """
        Initializes PgCheckpointer with PostgreSQL and Redis connections.

        Args:
            postgres_dsn (str, optional): PostgreSQL connection string.
            pg_pool (Any, optional): Existing asyncpg Pool instance.
            pool_config (dict, optional): Configuration for new pg pool creation.
            redis_url (str, optional): Redis connection URL.
            redis (Any, optional): Existing Redis instance.
            redis_pool (Any, optional): Existing Redis ConnectionPool.
            redis_pool_config (dict, optional): Configuration for new redis pool creation.
            schema (str, optional): PostgreSQL schema name. Defaults to "public".
            **kwargs: Additional configuration options.

        Raises:
            ImportError: If required dependencies are missing.
            ValueError: If required connection details are missing.
        """
        # Check for required dependencies
        if not HAS_ASYNCPG:
            raise ImportError(
                "PgCheckpointer requires 'asyncpg' package. "
                "Install with: pip install 10xgraph[pg_checkpoint]"
            )

        if not HAS_REDIS:
            raise ImportError(
                "PgCheckpointer requires 'redis' package. "
                "Install with: pip install 10xgraph[pg_checkpoint]"
            )

        self.user_id_type = kwargs.get("user_id_type", "string")
        # The id type follows the id generator bound in InjectQ (see ``id_type``). A passed
        # ``id_type`` is still honoured but deprecated.
        self._set_declared_id_type(kwargs.get("id_type"))
        self.cache_ttl = kwargs.get("cache_ttl", DEFAULT_CACHE_TTL)
        self.release_resources = kwargs.get("release_resources", False)
        # Prefix for every Redis key this checkpointer writes, so two deployments that share
        # one Redis do not read each other's cached state (default: no prefix).
        self.redis_key_prefix = str(kwargs.get("redis_key_prefix", "") or "")
        # Seconds to wait for a free Postgres connection before failing (None waits forever).
        self.pg_acquire_timeout = kwargs.get("pg_acquire_timeout", DEFAULT_ACQUIRE_TIMEOUT)
        # Redis circuit breaker (see DEFAULT_REDIS_BREAKER_COOLDOWN); None when disabled.
        cooldown = float(kwargs.get("redis_breaker_cooldown", DEFAULT_REDIS_BREAKER_COOLDOWN) or 0)
        self._redis_breaker = _RedisBreaker(cooldown) if cooldown > 0 else None
        # State cache keys whose invalidation was skipped while Redis was down.
        self._skipped_invalidations: set[str] = set()

        # Ownership is tracked PER RESOURCE. A single shared flag was wrong: when
        # a caller passed in their own pg_pool but only a redis_url, building the
        # Redis pool flipped the flag and arelease() then closed the caller's
        # Postgres pool -- which the rest of their app was still using. We only
        # ever close what we created ourselves (or what the caller explicitly
        # told us to release via release_resources=True).
        #
        # Decided here, not in _create_pg_pool: that runs lazily on first use, so
        # a checkpointer built from a DSN but released before its first query
        # would otherwise look unowned.
        self._owns_pg_pool = pg_pool is None
        self._owns_redis = False
        # Number of historical state rows to keep per thread (append-only history
        # is pruned to this window on every durable write).
        self.state_history_limit = kwargs.get("state_history_limit", DEFAULT_STATE_HISTORY_LIMIT)
        # Whether ``user_id`` is treated as an ownership boundary.
        #
        # Secure by default: threads, state, and messages are scoped to the
        # ``user_id`` in the config, so an authenticated caller cannot read or
        # write another user's thread even if they know (or guess) its id.
        #
        # This is a framework, not a product, so the decision stays with the
        # developer. Set ``enforce_user_isolation=False`` for a single-tenant app,
        # or when there is no real user identity (no auth configured, or a
        # placeholder/None user_id). With it off, ``user_id`` is ignored for
        # ownership entirely and every query keys on ``thread_id`` alone, which
        # also skips the extra join. Only turn it off when you are not relying on
        # a thread_id being secret.
        self.enforce_user_isolation = kwargs.get("enforce_user_isolation", True)

        # Developer-declared columns on ``threads`` (see the class docstring).
        self.columns = parse_columns(
            kwargs.get("tenant_columns"),
            kwargs.get("extra_columns"),
            reserved=THREAD_BUILTIN_COLUMNS,
            table="threads",
        )
        #: Names of the declared tenant columns (read by the authz helpers and compile()).
        self.tenant_column_names: tuple[str, ...] = self.columns.tenant_names
        #: Columns ``alist_threads(filters=...)`` may name (thread built-ins plus extra
        #: columns); pass it to ``tenxgraph.storage.sql_filter.validate_filters`` to check a
        #: request early.
        self.filter_schema = FilterSchema.for_threads(self.columns)
        row_level_security = kwargs.get("row_level_security", False)
        if not isinstance(row_level_security, bool):
            raise ValueError("row_level_security must be True or False")
        if row_level_security and not self.columns.tenant:
            raise ValueError("row_level_security=True needs at least one tenant column")
        self.row_level_security = row_level_security
        # False turns the tool ledger off (see BaseCheckpointer.record_tool_result).
        self.record_tool_result = bool(kwargs.get("record_tool_result", True))
        self.ddl_lock_timeout = kwargs.get("ddl_lock_timeout", DEFAULT_DDL_LOCK_TIMEOUT)
        ddl_lock_timeout_setting(self.ddl_lock_timeout)  # validates

        quote_identifier(schema, "schema name")  # validates: prevents SQL injection
        self.schema = schema

        self._schema_initialized = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # Guards lazy pool creation so a cold-start race cannot create (and leak)
        # two pools for the same checkpointer (see _get_pg_pool).
        self._pg_pool_lock = asyncio.Lock()
        # Guards schema init within this process; a Postgres advisory lock guards
        # it across processes (see _initialize_schema).
        self._schema_init_lock = asyncio.Lock()

        # Store pool configuration for lazy initialization
        self._pg_pool_config = {
            "pg_pool": pg_pool,
            "postgres_dsn": postgres_dsn,
            "pool_config": pool_config or {},
        }

        # Initialize pool immediately if provided, otherwise defer
        if pg_pool is not None:
            self._pg_pool = pg_pool
        else:
            self._pg_pool = None

        # Now check and initialize connections
        if not pg_pool and not postgres_dsn:
            raise ValueError("Either postgres_dsn or pg_pool must be provided.")

        if not redis and not redis_url and not redis_pool:
            raise ValueError("Either redis_url, redis_pool or redis instance must be provided.")

        # Initialize Redis connection (synchronous)
        self.redis = self._create_redis_pool(redis, redis_pool, redis_url, redis_pool_config or {})

    def _create_redis_pool(
        self,
        redis: Any | None,
        redis_pool: Any | None,
        redis_url: str | None,
        redis_pool_config: dict,
    ) -> Any:
        """
        Create or use an existing Redis connection.

        Args:
            redis (Any, optional): Existing Redis instance.
            redis_pool (Any, optional): Existing Redis ConnectionPool.
            redis_url (str, optional): Redis connection URL.
            redis_pool_config (dict): Configuration for new redis pool creation.

        Returns:
            Redis: Redis connection instance.

        Raises:
            ValueError: If redis_url is not provided when creating a new connection.
        """
        # Caller-owned: they passed the client/pool in, so they close it.
        if redis:
            return redis

        if redis_pool:
            return Redis(connection_pool=redis_pool)  # type: ignore

        # We are building the pool from a URL, so we own it and must close it.
        # This marks ONLY Redis as ours -- it must not imply anything about the
        # Postgres pool, which the caller may have supplied.
        if not redis_url:
            raise ValueError("redis_url must be provided when creating new Redis connection")

        self._owns_redis = True
        return Redis(
            connection_pool=BlockingConnectionPool.from_url(  # type: ignore
                redis_url,
                **{**DEFAULT_REDIS_POOL_CONFIG, **redis_pool_config},
            )
        )

    def _set_declared_id_type(self, given: Any) -> None:
        """Keep a passed (deprecated) ``id_type``; without one the id type follows the generator."""
        self._id_type_given = given is not None
        self._pinned_id_type: str | None = None
        if given is None:
            return
        warnings.warn(
            "id_type is deprecated and removed in 2.0; the id type now follows the bound "
            "id generator",
            DeprecationWarning,
            stacklevel=3,
        )
        self._pinned_id_type = _plain_id_type(given)

    def _get_table_name(self, table: str) -> str:
        """
        Get the schema-qualified table name.

        Args:
            table (str): The base table name (e.g., 'threads', 'states', 'messages')

        Returns:
            str: The schema-qualified table name (e.g., '"public"."threads"')
        """
        # Both names are checked, so neither can break out of the quotes (SQL injection).
        schema = quote_identifier(self.schema, "schema name")
        return f"{schema}.{quote_identifier(table, 'table name')}"

    def _create_pg_pool(self, pg_pool: Any, postgres_dsn: str | None, pool_config: dict) -> Any:
        """
        Create or use an existing PostgreSQL connection pool.

        Args:
            pg_pool (Any, optional): Existing asyncpg Pool instance.
            postgres_dsn (str, optional): PostgreSQL connection string.
            pool_config (dict): Configuration for new pg pool creation.

        Returns:
            Pool: PostgreSQL connection pool.
        """
        # Caller-owned: they passed the pool in, so they close it.
        if pg_pool:
            return pg_pool
        # Built from a DSN, so we own it. Ownership was already recorded in
        # __init__ (this runs lazily on first use, which may be never).
        return asyncpg.create_pool(  # type: ignore
            dsn=postgres_dsn, **{**DEFAULT_POOL_CONFIG, **pool_config}
        )

    async def _get_pg_pool(self) -> Any:
        """
        Get PostgreSQL pool, creating it if necessary.

        Returns:
            Pool: PostgreSQL connection pool.
        """
        if self._pg_pool is None:
            async with self._pg_pool_lock:
                # Re-check under the lock: another coroutine may have created the
                # pool while we were waiting to acquire it.
                if self._pg_pool is None:
                    config = self._pg_pool_config
                    self._pg_pool = await self._create_pg_pool(
                        config["pg_pool"], config["postgres_dsn"], config["pool_config"]
                    )
        return self._pg_pool

    ###########################
    #### RESOURCE CLEANUP #####
    ###########################

    async def aping(self) -> None:
        """Check Postgres, its schema version and (when configured) Redis; raise when not usable.

        Runs ``SELECT 1`` on the pool, compares the recorded schema version with the one this
        code expects, then pings Redis. Never creates or upgrades the schema (call ``asetup``
        for that), so a missing schema raises. Redis is pinged directly, not through the
        circuit breaker, so an outage is reported and the breaker state is left unchanged.

        Raises:
            RuntimeError: When the recorded schema version is older than the expected one.
            Exception: Any error from Postgres or Redis.
        """

        async def _select_one():
            async with (await self._get_pg_pool()).acquire(timeout=self.pg_acquire_timeout) as conn:
                return await conn.fetchval("SELECT 1")

        await self._retry_on_connection_error(_select_one, max_retries=3)
        found = await self.aget_schema_version()
        expected = self._get_current_schema_version()
        if found < expected:
            raise RuntimeError(
                f"checkpointer schema version {found} is below the expected {expected}"
            )
        if self.redis is not None:
            await self.redis.ping()

    async def arelease(self) -> Any | None:
        """
        Clean up connections and resources.

        Returns:
            Any | None: None
        """
        logger.info("Releasing PgCheckpointer resources")

        # Close each resource only if we created it. ``release_resources=True``
        # remains an explicit opt-in for callers who want us to close pools they
        # handed in. Deciding per resource is what stops a caller-supplied
        # pg_pool from being closed just because we happened to build the Redis
        # pool from a URL.
        close_redis = self.release_resources or self._owns_redis
        close_pg = self.release_resources or self._owns_pg_pool

        if not close_redis and not close_pg:
            logger.info("No owned resources to release")
            return

        errors = []

        # Close Redis connection
        if close_redis:
            try:
                if hasattr(self.redis, "aclose"):
                    await self.redis.aclose()
                elif hasattr(self.redis, "close"):
                    await self.redis.close()
                logger.debug("Redis connection closed")
            except Exception as e:
                logger.error("Error closing Redis connection: %s", describe_error(e))
                errors.append(f"Redis: {describe_error(e)}")

        # Close PostgreSQL pool
        if close_pg:
            try:
                if self._pg_pool and not self._pg_pool.is_closing():
                    await self._pg_pool.close()
                logger.debug("PostgreSQL pool closed")
            except Exception as e:
                logger.error("Error closing PostgreSQL pool: %s", describe_error(e))
                errors.append(f"PostgreSQL: {describe_error(e)}")

        if errors:
            error_msg = f"Errors during resource cleanup: {'; '.join(errors)}"
            logger.warning(error_msg)
            # Don't raise - cleanup should be best effort
        else:
            logger.info("All resources released successfully")
