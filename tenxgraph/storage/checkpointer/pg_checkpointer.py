import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import timedelta
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar
from urllib.parse import quote, unquote

from injectq import InjectQ

from tenxgraph.core.authz import owner_id
from tenxgraph.core.exceptions import StateFormatVersionError
from tenxgraph.core.exceptions.storage_exceptions import (
    SchemaVersionError,
    StaleStateError,
    StorageError,
    ThreadNotFoundError,
    TransientStorageError,
)
from tenxgraph.storage.columns import (
    ColumnSpec,
    apply_settings,
    build_isolation_where,
    column_ddl,
    column_value_errors,
    ddl_lock_timeout_setting,
    ddl_session,
    ensure_column_indexes,
    ensure_columns,
    ensure_index,
    ensure_rls,
    index_catalog,
    isolation_conditions,
    parse_columns,
    rls_all_rows_settings,
    rls_policy_expression,
    rls_policy_name,
    rls_settings,
    role_bypasses_rls,
    warn_if_rls_bypassed,
    write_conditions,
)
from tenxgraph.storage.sql_filter import FilterSchema, compile_filters, validate_filters
from tenxgraph.utils import ThreadInfo, metrics

try:
    import asyncpg
    from asyncpg import Pool

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None  # type: ignore
    Pool = None  # type: ignore

try:
    from redis.asyncio import BlockingConnectionPool, Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    HAS_REDIS = True
    # Errors that mean "Redis cannot be reached", as opposed to a bad command or value. redis-py
    # wraps socket and DNS failures in its own ConnectionError / TimeoutError (neither is an
    # OSError); OSError still covers a raw socket error and the builtin TimeoutError, which is
    # also asyncio.TimeoutError.
    _REDIS_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
        RedisConnectionError,
        RedisTimeoutError,
        OSError,
    )
except ImportError:
    HAS_REDIS = False
    BlockingConnectionPool = None  # type: ignore
    Redis = None  # type: ignore
    _REDIS_CONNECTION_ERRORS = (OSError,)

from tenxgraph.core.state import AgentState, Message

from .base_checkpointer import BaseCheckpointer, validate_page

logger = logging.getLogger("tenxgraph.checkpointer.pg")


StateT = TypeVar("StateT", bound="AgentState")

# Default TTL for Redis cache (24 hours)
DEFAULT_CACHE_TTL = 86400

# Current durable schema version. Bump when schema changes and add a matching
# entry to ``PgCheckpointer._migrations``.
CURRENT_SCHEMA_VERSION = 7

# Columns of the ``threads`` table the checkpointer owns; tenant and extra columns cannot reuse
# these names.
THREAD_BUILTIN_COLUMNS = frozenset(
    {"thread_id", "thread_name", "user_id", "created_at", "updated_at", "meta"}
)

# Tables whose rows belong to a thread. Under native RLS each gets a policy that shows a row only
# when its thread row is visible (the ``threads`` policy carries the tenant columns).
_THREAD_CHILD_TABLES = ("states", "messages", "tool_executions")

# Seconds a setup DDL statement waits for a table lock before setup fails with
# ``SetupLockTimeoutError`` (``ddl_lock_timeout`` option; None waits forever).
DEFAULT_DDL_LOCK_TIMEOUT = 10.0

# Config key under which the thread's identity (owner and tenant values, read from its row after
# a scoped check) is kept for the rest of a call or run. Underscore-prefixed keys are internal:
# the API drops them from client config. See ``PgCheckpointer._thread_identity``.
_THREAD_IDENTITY_KEY = "_thread_identity"

# Advisory-lock key used to serialize schema creation/migration across processes.
# Any stable arbitrary 64-bit constant works; it only has to be the same in every
# process that initializes this schema.
SCHEMA_INIT_LOCK_ID = 0x41474E54464C5721  # "AGNTFLW!"

# How many historical state rows to retain per thread. The append-only history
# is useful for debugging/audit, but must be bounded to avoid unbounded growth.
# Older rows beyond this window are pruned on each durable write.
DEFAULT_STATE_HISTORY_LIMIT = 20

# Config key used to carry the state version read at the start of a run through
# to the durable write, so the write can perform an optimistic compare-and-swap.
STATE_VERSION_CONFIG_KEY = "_checkpoint_version"

# Marker embedded in the cached (Redis) state payload so a resume-from-cache can
# recover the durable version for the compare-and-swap. Kept out of the durable
# ``state_data`` column, which uses the authoritative ``version`` column instead.
_CACHE_VERSION_KEY = "__checkpoint_version__"

# Atomic version-guarded cache write.
#
# The realtime cache is written on every step of a run, including by runs whose
# durable write will later lose the optimistic version check. A plain SETEX would
# let such a run stamp its OLDER state over a newer one that another run already
# committed -- and because reads prefer the cache and seed the next write's
# expected version from it, the thread would then fail its compare-and-swap
# forever (wedged until the TTL expired).
#
# This script only writes when the incoming version is >= the cached one, so a
# stale run can never move the cache backwards. Equal versions are allowed: that
# is the normal within-a-run case (progress updates, stop flags) where the state
# advances but the durable version has not been bumped yet. Doing the compare and
# the set in one Lua call keeps it atomic; a GET-then-SETEX from Python would
# still race across coroutines.
_CACHE_CAS_LUA = """
local existing = redis.call('GET', KEYS[1])
if existing then
    local ok, decoded = pcall(cjson.decode, existing)
    if ok and type(decoded) == 'table' and decoded[ARGV[4]] then
        if tonumber(decoded[ARGV[4]]) > tonumber(ARGV[2]) then
            return 0
        end
    end
end
redis.call('SETEX', KEYS[1], ARGV[3], ARGV[1])
return 1
"""

# Compare-and-delete for the generic cache: delete KEYS[1] only if it still holds ARGV[1].
_CLEAR_IF_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

# Infrastructure timeouts. Without them a blackholed Postgres or Redis hangs a run instead of
# failing it. They apply to pools this class builds (a pool you pass in keeps its own settings)
# and are overridden key by key through ``pool_config`` / ``redis_pool_config``. The Redis pool
# is bounded too: a caller past ``max_connections`` waits up to ``timeout`` seconds for a free
# connection, then fails, instead of opening connections without limit.
DEFAULT_PG_ACQUIRE_TIMEOUT = 30.0  # seconds to wait for a free pooled connection
DEFAULT_PG_POOL_CONFIG: dict[str, Any] = {
    "command_timeout": 60.0,  # per query
    "timeout": 30.0,  # per new connection
}
DEFAULT_REDIS_POOL_CONFIG: dict[str, Any] = {
    "socket_connect_timeout": 5.0,
    "socket_timeout": 10.0,
    "health_check_interval": 30,
    "max_connections": 50,
    "timeout": 5.0,  # seconds to wait for a free pooled connection
}

# Sentinel version used when the caller has no durable version yet (brand-new
# thread). Lower than any real version, so it never overwrites a newer entry.
_NO_VERSION = -1

# Redis is a cache; Postgres is the source of truth. After a Redis connection failure every
# Redis call is skipped for this many seconds and takes its normal fallback at once, so an
# outage costs one connect timeout per cool-down instead of one per call (a run makes several
# Redis calls per step). Override with ``redis_breaker_cooldown``; 0 disables the breaker.
DEFAULT_REDIS_BREAKER_COOLDOWN = 5.0

# Cache keys whose invalidation was skipped while Redis was unreachable are deleted when it is
# reachable again (see PgCheckpointer._redis_call). Bounded so a long outage cannot grow it
# without limit; past the bound, the version check in aget_state_cache still guards reads.
_MAX_SKIPPED_INVALIDATIONS = 10_000

# Clock for the breaker. A module-level name so tests can drive it without patching the
# process-wide time.monotonic, which the event loop also reads.
_monotonic = time.monotonic


class _RedisUnavailable(Exception):
    """A Redis call was skipped (breaker open) or failed to connect (breaker now open)."""


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


# SQL type mapping for ID types
ID_TYPE_MAP = {
    "string": "VARCHAR(255)",
    "int": "SERIAL",
    "bigint": "BIGSERIAL",
}


@dataclass
class _Isolation:
    """Tenant and extra column values for one call, and the tenant isolation they imply.

    ``values``: every declared column's value from ``config["user"]`` (``None`` when missing).
    ``conditions``: the tenant columns the scope picks, with the caller's values.
    ``settings``: the RLS ``set_config`` pairs, or ``None`` when RLS is off.
    """

    values: dict[str, Any] = field(default_factory=dict)
    conditions: list[tuple[ColumnSpec, Any]] = field(default_factory=list)
    settings: list[tuple[str, str]] | None = None


_NO_ISOLATION = _Isolation()


def _same_value(spec: ColumnSpec, stored: Any, wanted: Any) -> bool:
    """Whether a stored tenant value equals the caller's (``IS NOT DISTINCT FROM``).

    ``wanted`` is ``None`` only on a write by a caller without a value for an optional tenant
    column; it then matches a thread that has none either. Reads never get here with ``None``
    (a picked column without a value raises first).
    """
    if stored is None or wanted is None:
        return stored is None and wanted is None
    if spec.type == "uuid" or isinstance(stored, uuid.UUID):
        return str(stored) == str(wanted)
    return stored == wanted


def _rows_affected(status: Any) -> int:
    """Rows affected from an asyncpg command tag such as ``"DELETE 3"``."""
    try:
        return int(str(status).rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        return 0


class PgCheckpointer(BaseCheckpointer[StateT]):
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
                    async def authenticate(self, request, credential):
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
        # allow explicit override via kwargs, fallback to InjectQ, then default
        self.id_type = kwargs.get(
            "id_type", InjectQ.get_instance().try_get("generated_id_type", "string")
        )
        self.cache_ttl = kwargs.get("cache_ttl", DEFAULT_CACHE_TTL)
        self.release_resources = kwargs.get("release_resources", False)
        # Prefix for every Redis key this checkpointer writes, so two deployments that share
        # one Redis do not read each other's cached state (default: no prefix).
        self.redis_key_prefix = str(kwargs.get("redis_key_prefix", "") or "")
        # Seconds to wait for a free Postgres connection before failing (None waits forever).
        self.pg_acquire_timeout = kwargs.get(
            "pg_acquire_timeout", DEFAULT_PG_ACQUIRE_TIMEOUT
        )
        # Redis circuit breaker (see DEFAULT_REDIS_BREAKER_COOLDOWN); None when disabled.
        cooldown = float(
            kwargs.get("redis_breaker_cooldown", DEFAULT_REDIS_BREAKER_COOLDOWN) or 0
        )
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
        self.state_history_limit = kwargs.get(
            "state_history_limit", DEFAULT_STATE_HISTORY_LIMIT
        )
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

        # Validate schema name to prevent SQL injection
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", schema):
            raise ValueError(
                f"Invalid schema name: {schema}. Schema must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
            )
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
            raise ValueError(
                "Either redis_url, redis_pool or redis instance must be provided."
            )

        # Initialize Redis connection (synchronous)
        self.redis = self._create_redis_pool(
            redis, redis_pool, redis_url, redis_pool_config or {}
        )

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
            raise ValueError(
                "redis_url must be provided when creating new Redis connection"
            )

        self._owns_redis = True
        return Redis(
            connection_pool=BlockingConnectionPool.from_url(  # type: ignore
                redis_url,
                **{**DEFAULT_REDIS_POOL_CONFIG, **redis_pool_config},
            )
        )

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

    def _get_table_name(self, table: str) -> str:
        """
        Get the schema-qualified table name.

        Args:
            table (str): The base table name (e.g., 'threads', 'states', 'messages')

        Returns:
            str: The schema-qualified table name (e.g., '"public"."threads"')
        """
        # Validate table name to prevent SQL injection
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
            raise ValueError(
                f"Invalid table name: {table}. Table must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
            )
        return f'"{self.schema}"."{table}"'

    def _create_pg_pool(
        self, pg_pool: Any, postgres_dsn: str | None, pool_config: dict
    ) -> Any:
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
            dsn=postgres_dsn, **{**DEFAULT_PG_POOL_CONFIG, **pool_config}
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

    def _get_sql_type(self, type_name: str) -> str:
        """
        Get SQL type for given configuration type.

        Args:
            type_name (str): Type name ('string', 'int', 'bigint').

        Returns:
            str: Corresponding SQL type.
        """
        return ID_TYPE_MAP.get(type_name, "VARCHAR(255)")

    def _get_json_serializer(self):
        """Get optimal JSON serializer based on FAST_JSON env var."""
        if os.environ.get("FAST_JSON", "0") == "1":
            try:
                import orjson

                return orjson.dumps
            except ImportError:
                try:
                    import msgspec  # type: ignore

                    return msgspec.json.encode
                except ImportError:
                    pass
        return json.dumps

    def _get_current_schema_version(self) -> int:
        """Return current expected schema version."""
        return CURRENT_SCHEMA_VERSION

    def _build_create_tables_sql(self) -> list[str]:
        """
        Build SQL statements for table creation with dynamic ID types.

        Returns:
            list[str]: List of SQL statements for table creation.
        """
        tables = self._build_table_statements()
        created = {table for table, _sql in tables}
        statements = [sql for _table, sql in tables]
        # ``tool_executions`` is made by migration v3, so its index is not part of this list.
        statements.extend(
            f"CREATE INDEX IF NOT EXISTS {name} ON {self._get_table_name(table)}{definition}"
            for name, table, definition in self._base_index_specs()
            if table in created
        )
        return statements

    @staticmethod
    def _base_index_specs() -> list[tuple[str, str, str]]:
        """``(index name, table, definition)`` of the plain indexes of a fresh schema.

        Version-dependent indexes on ``states`` are created by the schema migrations (see
        _build_migrations) so that an upgrade from a pre-version schema adds the column before
        indexing it.
        """
        return [
            ("idx_threads_user_id", "threads", "(user_id)"),
            ("idx_states_thread_id", "states", "(thread_id)"),
            ("idx_messages_thread_id", "messages", "(thread_id)"),
            # v7: aprune_tool_results deletes by age across all threads.
            ("idx_tool_executions_created_at", "tool_executions", "(created_at)"),
        ]

    def _build_table_statements(self) -> list[tuple[str | None, str]]:
        """``(table, sql)`` creating each table (``None`` for the enum type block)."""
        thread_id_type = self._get_sql_type(self.id_type)
        user_id_type = self._get_sql_type(self.user_id_type)
        message_id_type = self._get_sql_type(self.id_type)

        # For AUTO INCREMENT types, we need to handle primary key differently
        thread_pk = (
            "thread_id SERIAL PRIMARY KEY"
            if self.id_type == "int"
            else f"thread_id {thread_id_type} PRIMARY KEY"
        )
        # Message ids are unique per thread, not globally (see migration v4), so the key
        # is declared as a table constraint on (thread_id, message_id).
        message_col = (
            "message_id SERIAL"
            if self.id_type == "int"
            else f"message_id {message_id_type}"
        )

        return [
            # Schema version tracking table
            (
                "schema_version",
                f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("schema_version")} (
                version INT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            ),
            # Create message role enum (safe for older Postgres versions)
            (
                None,
                (
                    "DO $$\n"
                    "BEGIN\n"
                    "    CREATE TYPE message_role AS ENUM "
                    "('user', 'assistant', 'system', 'tool');\n"
                    "EXCEPTION\n"
                    "    WHEN duplicate_object THEN NULL;\n"
                    "END$$;"
                ),
            ),
            # Create threads table
            (
                "threads",
                f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("threads")} (
                {thread_pk},
                thread_name VARCHAR(255),
                user_id {user_id_type} NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            ),
            # Create states table.
            # ``version`` is a per-thread monotonically increasing counter used
            # for optimistic concurrency control (see aput_state/aput_checkpoint).
            # The UNIQUE (thread_id, version) constraint is what makes concurrent
            # appends collide instead of silently duplicating a version.
            (
                "states",
                f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("states")} (
                state_id SERIAL PRIMARY KEY,
                thread_id {thread_id_type} NOT NULL
                    REFERENCES {self._get_table_name("threads")}(thread_id)
                    ON DELETE CASCADE,
                version BIGINT NOT NULL DEFAULT 0,
                state_data JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            ),
            # Create messages table
            (
                "messages",
                f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("messages")} (
                {message_col} NOT NULL,
                thread_id {thread_id_type} NOT NULL
                    REFERENCES {self._get_table_name("threads")}(thread_id)
                    ON DELETE CASCADE,
                role message_role NOT NULL,
                content TEXT NOT NULL,
                tool_calls JSONB,
                tool_call_id VARCHAR(255),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                total_tokens INT DEFAULT 0,
                usages JSONB DEFAULT '{{}}'::jsonb,
                meta JSONB DEFAULT '{{}}'::jsonb,
                PRIMARY KEY (thread_id, message_id)
            )
            """,
            ),
        ]

    def _build_migrations(self) -> dict[int, list[str]]:
        """Return ordered DDL steps keyed by the schema version they produce.

        Every statement must be idempotent (``IF NOT EXISTS`` / guarded), so a
        migration can run safely both on a freshly created schema (where the
        target objects already exist and each step is a no-op) and on an older
        database being upgraded in place. Migrations are applied in ascending
        version order inside a single transaction (see
        ``_check_and_apply_schema_version``).
        """
        states = self._get_table_name("states")
        messages = self._get_table_name("messages")
        return {
            # v2: per-thread state versioning for optimistic concurrency control.
            2: [
                f"ALTER TABLE {states} ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0",
                # Backfill deterministic per-thread versions for any pre-existing
                # rows so the unique index below can be created. Ordered by
                # created_at then state_id to match the "latest wins" read order.
                f"""
                UPDATE {states} AS s
                SET version = sub.rn
                FROM (
                    SELECT state_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY thread_id
                               ORDER BY created_at, state_id
                           ) AS rn
                    FROM {states}
                ) AS sub
                WHERE s.state_id = sub.state_id
                  AND s.version = 0
                """,  # noqa: S608
                f"CREATE INDEX IF NOT EXISTS idx_states_thread_version ON "
                f"{states}(thread_id, version DESC)",
                f"CREATE UNIQUE INDEX IF NOT EXISTS uq_states_thread_version ON "
                f"{states}(thread_id, version)",
            ],
            # v3: durable tool-execution ledger, so a node replayed after a crash
            # does not re-fire tool calls that already completed (audit B2).
            # The (thread_id, tool_call_id) primary key IS the idempotency key.
            3: [
                f"""
                CREATE TABLE IF NOT EXISTS {self._get_table_name("tool_executions")} (
                    thread_id {self._get_sql_type(self.id_type)} NOT NULL
                        REFERENCES {self._get_table_name("threads")}(thread_id)
                        ON DELETE CASCADE,
                    tool_call_id VARCHAR(255) NOT NULL,
                    result JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (thread_id, tool_call_id)
                )
                """,
            ],
            # v4: message ids are unique per thread. The old key was message_id alone,
            # so one user could claim an id another thread later needed and make that
            # thread's checkpoint fail. Guarded: a schema created at v4 or later
            # already has the two-column key, so the block does nothing there.
            4: [
                f"""
                DO $$
                DECLARE pk_name text;
                BEGIN
                    SELECT conname INTO pk_name FROM pg_constraint
                    WHERE conrelid = '{messages}'::regclass
                      AND contype = 'p'
                      AND array_length(conkey, 1) = 1;
                    IF pk_name IS NOT NULL THEN
                        EXECUTE format(
                            'ALTER TABLE %s DROP CONSTRAINT %I',
                            '{messages}'::regclass, pk_name
                        );
                        EXECUTE format(
                            'ALTER TABLE %s ADD PRIMARY KEY (thread_id, message_id)',
                            '{messages}'::regclass
                        );
                    END IF;
                END$$;
                """,  # noqa: S608
            ],
            # v5: reasoning lives only in the ReasoningBlocks inside ``content``, so
            # the separate text column is gone. A fresh schema never has it.
            5: [
                f"ALTER TABLE {messages} DROP COLUMN IF EXISTS reasoning",
            ],
            # v6: the developer's tenant and extra columns on ``threads`` (ADD COLUMN IF NOT
            # EXISTS). Nothing declared: no change. A column declared later is added by
            # _ensure_declared_columns, which runs on every setup. Extra columns of other
            # types (jsonb, text[], ...) are added there too, after the server resolves the
            # declared type: their text never reaches DDL. Their B-tree indexes are built by
            # _ensure_indexes, concurrently on an existing table.
            6: column_ddl(
                self._get_table_name("threads"),
                self.columns.builtin_only(),
                "idx_threads",
                indexes=False,
            ),
            # v7: an index on tool_executions(created_at) for aprune_tool_results. No DDL
            # here: _ensure_indexes builds it after the migrations, CONCURRENTLY when the
            # table already exists, so the upgrade does not block ledger writes.
            7: [],
        }

    async def _ensure_declared_columns(self, conn) -> None:
        """Add declared columns that are missing and manage the native RLS policies.

        Runs on every schema init, under the init advisory lock, after the migrations. Reads
        the catalog first, so nothing is altered when everything is in place. With nothing
        declared it does nothing. To turn RLS off, keep the tenant columns and set
        ``row_level_security=False``: the next setup drops the policies and disables RLS.
        The columns' indexes are built by ``_ensure_indexes``.
        """
        if not self.columns:
            return
        threads = self._get_table_name("threads")
        await ensure_columns(conn, threads, self.columns, "idx_threads", indexes=False)
        tenant = self.columns.tenant
        enabled = self.row_level_security
        await ensure_rls(
            conn,
            table_sql=threads,
            schema=self.schema,
            table="threads",
            enabled=enabled,
            policy=rls_policy_name(tenant) if enabled else None,
            expression=rls_policy_expression(tenant) if enabled else None,
        )
        for child in _THREAD_CHILD_TABLES:
            table_sql = self._get_table_name(child)
            # The subquery on threads is itself filtered by the threads policy, so a child
            # row is visible exactly when its thread is.
            expression = (
                f"EXISTS (SELECT 1 FROM {threads} AS t "  # noqa: S608
                f"WHERE t.thread_id = {table_sql}.thread_id)"
            )
            await ensure_rls(
                conn,
                table_sql=table_sql,
                schema=self.schema,
                table=child,
                enabled=enabled,
                policy=rls_policy_name(tenant, base="tenxgraph_isolation_thread")
                if enabled
                else None,
                expression=expression if enabled else None,
            )
        if enabled:
            await warn_if_rls_bypassed(conn, logger, "PgCheckpointer")

    async def _get_recorded_schema_version(self, conn) -> int:
        """Return the highest schema version recorded in the tracking table.

        Returns 0 when no version has been recorded yet (first run).
        """
        row = await conn.fetchrow(
            f"SELECT version FROM {self._get_table_name('schema_version')} "  # noqa: S608
            f"ORDER BY version DESC LIMIT 1"
        )
        return int(row["version"]) if row else 0

    async def _check_and_apply_schema_version(self, conn) -> None:
        """Apply any pending migrations and record the new schema version.

        Runs every migration between the recorded version and the target version
        in ascending order, inside the caller's transaction, then records the
        target version. Idempotent DDL means this is safe on both fresh and
        upgraded databases (see ``_build_migrations``).
        """
        try:
            current_version = await self._get_recorded_schema_version(conn)
            target_version = self._get_current_schema_version()

            if current_version >= target_version:
                return

            logger.info(
                "Upgrading checkpointer schema from version %d to %d",
                current_version,
                target_version,
            )
            migrations = self._build_migrations()
            for version in range(current_version + 1, target_version + 1):
                for statement in migrations.get(version, []):
                    logger.debug(
                        "Applying migration step for v%d: %s",
                        version,
                        statement.strip(),
                    )
                    await conn.execute(statement)

            await conn.execute(
                f"INSERT INTO {self._get_table_name('schema_version')} (version) VALUES ($1) "  # noqa: S608
                f"ON CONFLICT (version) DO NOTHING",
                target_version,
            )
        except Exception as e:
            logger.error("Failed to apply schema migrations: %s", e)
            raise SchemaVersionError(
                message=f"Failed to apply schema migrations: {e}",
                error_code="STORAGE_SCHEMA_001",
                context={"error_type": type(e).__name__},
            ) from e

    async def _initialize_schema(self) -> None:
        """
        Initialize database schema if not already done.

        Returns:
            None
        """
        if self._schema_initialized:
            return

        logger.debug(
            "Initializing database schema with types: id_type=%s, user_id_type=%s",
            self.id_type,
            self.user_id_type,
        )

        async with self._schema_init_lock:
            # Re-check under the in-process lock (another coroutine may have run
            # initialization while we waited).
            if self._schema_initialized:
                return

            async with (await self._get_pg_pool()).acquire(
                timeout=self.pg_acquire_timeout
            ) as conn:
                try:
                    # Serialize schema creation ACROSS processes (multiple gunicorn workers /
                    # k8s replicas starting at once). An in-process flag is not enough:
                    # concurrent CREATE TABLE IF NOT EXISTS / CREATE TYPE can still fail with
                    # "duplicate key value violates unique constraint
                    # pg_type_typname_nsp_index", which the DO-block's duplicate_object guard
                    # does not catch. The lock is session-level so it also covers the
                    # CREATE INDEX CONCURRENTLY that runs after the transaction; every DDL
                    # statement waits at most ddl_lock_timeout for a table lock.
                    async with ddl_session(
                        conn,
                        lock_key=SCHEMA_INIT_LOCK_ID,
                        timeout=self.ddl_lock_timeout,
                        owner="PgCheckpointer",
                    ):
                        async with conn.transaction():
                            # Catalog first: a restart runs no DDL for what is in place.
                            new_tables = await self._create_missing_tables(conn)
                            # Apply pending migrations and record the schema version.
                            await self._check_and_apply_schema_version(conn)
                            # Declared tenant/extra columns and RLS policies (idempotent).
                            await self._ensure_declared_columns(conn)
                            # Tables created just now are invisible to others: plain builds.
                            await self._ensure_indexes(
                                conn, new_tables, concurrently=False
                            )
                        # Indexes missing on tables in use are built without blocking them.
                        await self._ensure_indexes(
                            conn, self._index_tables() - new_tables, concurrently=True
                        )

                    self._schema_initialized = True
                    logger.debug("Database schema initialized successfully")
                except Exception as e:
                    logger.error("Failed to initialize database schema: %s", e)
                    raise

    async def _create_missing_tables(self, conn) -> set[str]:
        """Create the tables (and the role enum) that do not exist yet; return their names.

        Reads the catalog first, so a restart runs no ``CREATE TABLE`` at all.
        """
        created: set[str] = set()
        for table, sql in self._build_table_statements():
            if table is not None:
                exists = await conn.fetchval(
                    "SELECT to_regclass($1) IS NOT NULL", self._get_table_name(table)
                )
                if exists:
                    continue
                created.add(table)
            logger.debug("Executing SQL: %s", sql.strip())
            await conn.execute(sql)
        return created

    @staticmethod
    def _index_tables() -> set[str]:
        """Tables whose indexes ``_ensure_indexes`` manages."""
        return {"threads", "states", "messages", "tool_executions"}

    async def _ensure_indexes(
        self, conn, tables: set[str], *, concurrently: bool
    ) -> None:
        """Build the missing or invalid base and declared-column indexes of ``tables``.

        Reads the catalog first and skips indexes that are in place: ``CREATE INDEX IF NOT
        EXISTS`` would still wait for a SHARE lock behind every open write. With
        ``concurrently=True`` (tables in use) the builds use ``CREATE INDEX CONCURRENTLY``,
        outside any transaction; an invalid leftover of a failed build is dropped and rebuilt.
        """
        for table in sorted(tables & self._index_tables()):
            table_sql = self._get_table_name(table)
            catalog = await index_catalog(conn, table_sql)
            for name, index_table, definition in self._base_index_specs():
                if index_table != table:
                    continue
                await ensure_index(
                    conn,
                    catalog,
                    name=name,
                    table_sql=table_sql,
                    definition=definition,
                    concurrently=concurrently,
                )
            if table == "threads" and self.columns:
                await ensure_column_indexes(
                    conn,
                    table_sql,
                    self.columns,
                    "idx_threads",
                    concurrently=concurrently,
                    catalog=catalog,
                )

    async def aget_schema_version(self) -> int:
        """The schema version recorded in the database; 0 when the schema is missing.

        One cheap indexed query that never creates anything, for readiness checks. Compare it
        with ``CURRENT_SCHEMA_VERSION`` (also ``self._get_current_schema_version()``).
        """
        async with (await self._get_pg_pool()).acquire(
            timeout=self.pg_acquire_timeout
        ) as conn:
            exists = await conn.fetchval(
                "SELECT to_regclass($1) IS NOT NULL",
                self._get_table_name("schema_version"),
            )
            if not exists:
                return 0
            return await self._get_recorded_schema_version(conn)

    ###########################
    #### SETUP METHODS ########
    ###########################

    async def asetup(self) -> Any:
        """
        Asynchronous setup method. Initializes database schema.

        Returns:
            Any: True if setup completed.
        """
        logger.info(
            "Setting up PgCheckpointer (async)",
            extra={
                "id_type": self.id_type,
                "user_id_type": self.user_id_type,
                "schema": self.schema,
            },
        )
        await self._initialize_schema()
        logger.info("PgCheckpointer setup completed")
        return True

    ###########################
    #### HELPER METHODS #######
    ###########################

    def _validate_config(self, config: dict[str, Any]) -> tuple[str | int, str | int]:
        """
        Extract and validate thread_id and user_id from config.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            tuple: (thread_id, user_id)

        Raises:
            ValueError: If required fields are missing.
        """
        thread_id = config.get("thread_id")
        user_id = owner_id(config)
        if not user_id:
            raise ValueError("user_id must be provided in config")

        if not thread_id:
            raise ValueError("Both thread_id must be provided in config")

        return self._db_thread_id(thread_id), user_id

    def _db_thread_id(self, thread_id: str | int) -> str | int:
        """``thread_id`` in the column's type.

        Callers (the API in particular) pass thread ids as strings. An ``int``/``bigint``
        column needs an int, or asyncpg rejects every query for that thread.
        """
        if self.id_type in ("int", "bigint") and not isinstance(thread_id, int):
            try:
                return int(str(thread_id).strip())
            except ValueError:
                raise ValueError(
                    f"thread_id must be an integer for id_type={self.id_type!r}: {thread_id!r}"
                ) from None
        return thread_id

    def _get_thread_key(
        self,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any] | None = None,
    ) -> str:
        """
        Get Redis cache key for thread state.

        Args:
            thread_id (str|int): Thread identifier.
            user_id (str|int): User identifier.
            config (dict, optional): The call's config. With tenant columns declared the key
                is the thread's identity: its owner and tenant values from the thread row
                (``_thread_identity``), or, before that row has been read, the caller's
                ``user_id`` and tenant values from ``config["user"]``.

        Returns:
            str: Redis cache key.
        """
        owner, suffix = self._key_identity(user_id, config)
        # Ids are percent-encoded so a ":" inside one cannot make two different
        # (thread, user) pairs produce the same key.
        return (
            f"{self.redis_key_prefix}state_cache:"
            f"{quote(str(thread_id), safe='')}:{quote(str(owner), safe='')}{suffix}"
        )

    def _tenant_key_suffix(self, config: dict[str, Any] | None) -> str:
        """``:t:org_id=<v>:project_id=<v>`` for the key identity's tenant values ('' without).

        Every declared tenant column is in the suffix (picked by the scope or not). The values
        are the thread row's (``_thread_identity``) once read, else the caller's. A missing
        value is written as ``col!``, which no percent-encoded value can produce.
        """
        return self._key_identity(None, config)[1]

    def _key_identity(
        self, user_id: Any, config: dict[str, Any] | None
    ) -> tuple[Any, str]:
        """``(owner, tenant suffix)`` the Redis keys of a thread are built from.

        With tenant columns and a ``_thread_identity`` read for this thread, both come from
        the thread row, so every caller allowed to see the thread shares its keys. Otherwise
        the caller's ``user_id`` and tenant values are used (as for a thread this caller is
        about to create, whose row will hold exactly those values).
        """
        if not self.columns.tenant:
            return user_id, ""
        identity = self._thread_identity(config)
        if identity is not None:
            owner = identity["user_id"]
            texts = identity["tenant"]
        else:
            owner = user_id
            values = self.columns.resolve(config)
            texts = {}
            for spec in self.columns.tenant:
                value = values.get(spec.name)
                texts[spec.name] = None if value is None else spec.to_setting(value)
        parts = []
        for spec in self.columns.tenant:
            text = texts.get(spec.name)
            if text is None:
                parts.append(f"{spec.name}!")
            else:
                parts.append(f"{spec.name}={quote(str(text), safe='')}")
        return owner, ":t:" + ":".join(parts)

    def _thread_identity(self, config: dict[str, Any] | None) -> dict[str, Any] | None:
        """The ``_thread_identity`` kept in ``config`` for its current thread, if any.

        Only this class writes it, after the thread row passed the caller's scoped check
        (``_remember_identity``). It is ignored when it was read for another thread.
        """
        if not self.columns.tenant or not isinstance(config, dict):
            return None
        identity = config.get(_THREAD_IDENTITY_KEY)
        if not isinstance(identity, dict):
            return None
        if str(identity.get("thread_id")) != str(config.get("thread_id")):
            return None
        return identity

    def _remember_identity(
        self, config: dict[str, Any] | None, thread_id: Any, row: Any
    ) -> None:
        """Keep the thread row's owner and tenant values in ``config`` (tenant columns only).

        ``row`` must come from a read that passed the caller's scope check.
        """
        if not self.columns.tenant or not isinstance(config, dict) or row is None:
            return
        tenant = {}
        for spec in self.columns.tenant:
            value = row[spec.name]
            tenant[spec.name] = None if value is None else spec.to_setting(value)
        config[_THREAD_IDENTITY_KEY] = {
            "thread_id": str(thread_id),
            "user_id": row["user_id"],
            "tenant": tenant,
        }

    def _identity_columns_sql(self) -> str:
        """``user_id, "org_id", ...``: the thread-row columns of its identity."""
        return ", ".join(["user_id", *(spec.quoted for spec in self.columns.tenant)])

    async def _read_thread_identity(
        self, thread_id: Any, user_id: Any, config: dict[str, Any]
    ) -> Any:
        """Read the thread row's identity and current state version under the caller's scope.

        Returns the row (``user_id``, tenant columns, ``durable_version``) and remembers the
        identity in ``config``, or ``None`` when the caller cannot see the thread or it does
        not exist. A scope the caller cannot satisfy raises before any query.
        """
        where, params = self._thread_row_scope(thread_id, user_id, config)
        query = (
            f"SELECT {self._identity_columns_sql()}, "  # noqa: S608
            f"(SELECT COALESCE(MAX(version), 0) FROM {self._get_table_name('states')} "
            "WHERE thread_id = $1) AS durable_version "
            f"FROM {self._get_table_name('threads')} WHERE {where}"
        )
        row = await self._run_scoped(
            lambda conn: conn.fetchrow(query, *params), settings=self._rls(config)
        )
        if row is None:
            if self._thread_identity(config) is not None:
                config.pop(
                    _THREAD_IDENTITY_KEY, None
                )  # no longer visible: own keys again
            return None
        self._remember_identity(config, thread_id, row)
        return row

    async def _resolve_thread_identity(self, config: dict[str, Any]) -> None:
        """Make the stop and active-run keys of this call the thread's (tenant columns only).

        Reads the thread row once per config under the caller's scope. When the caller cannot
        see the thread, or Postgres cannot be read, nothing is remembered and the caller's own
        values are used, which only ever reach keys of the caller's own identity.
        """
        if not self.columns.tenant or self._thread_identity(config) is not None:
            return
        thread_id, user_id = config.get("thread_id"), owner_id(config)
        if not thread_id or not user_id:
            return
        try:
            await self._read_thread_identity(
                self._db_thread_id(thread_id), user_id, config
            )
        except Exception as e:  # the caller's own keys are the safe fallback
            logger.debug(
                "Could not read the identity of thread_id=%s: %s", thread_id, e
            )

    def _stop_request_key(self, config: dict[str, Any]) -> str:
        """Per-thread stop/active-run key.

        Without tenant columns: ``<thread>:<caller user_id>`` as in the base class. With
        tenant columns: the thread's identity (owner and tenant values), so an owner working
        in another project, or an org admin, reaches the run's stop and active-run keys.
        """
        if not self.columns.tenant:
            return super()._stop_request_key(config)
        owner, suffix = self._key_identity(owner_id(config), config)
        return f"{config.get('thread_id')}:{owner}{suffix}"

    async def aget_active_run(self, config: dict[str, Any]) -> str | None:
        await self._resolve_thread_identity(config)
        return await super().aget_active_run(config)

    async def abegin_run(self, config: dict[str, Any]) -> None:
        await self._resolve_thread_identity(config)
        await super().abegin_run(config)

    async def arefresh_run(self, config: dict[str, Any]) -> None:
        await self._resolve_thread_identity(config)
        await super().arefresh_run(config)

    async def aend_run(self, config: dict[str, Any]) -> None:
        await self._resolve_thread_identity(config)
        await super().aend_run(config)

    async def arequest_stop(self, config: dict[str, Any]) -> Any | None:
        await self._resolve_thread_identity(config)
        return await super().arequest_stop(config)

    async def ais_stop_requested(self, config: dict[str, Any]) -> bool:
        await self._resolve_thread_identity(config)
        return await super().ais_stop_requested(config)

    async def aclear_stop_request(self, config: dict[str, Any]) -> Any | None:
        await self._resolve_thread_identity(config)
        return await super().aclear_stop_request(config)

    def _get_generic_cache_key(self, namespace: str, key: str) -> str:
        """Build a Redis key for shared non-state cache entries."""
        return f"{self._generic_cache_prefix(namespace)}{quote(str(key), safe='')}"

    def _generic_cache_prefix(self, namespace: str) -> str:
        return f"{self.redis_key_prefix}generic_cache:{quote(str(namespace), safe='')}:"

    def _serialize_state(self, state: StateT) -> str:
        """
        Serialize state to JSON string for storage.

        Args:
            state (StateT): State object.

        Returns:
            str: JSON string.
        """

        def enum_handler(obj):
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)

        return json.dumps(state.model_dump(), default=enum_handler)

    def _deserialize_state(
        self,
        data: Any,
        state_class: type[StateT],
    ) -> StateT:
        """
        Deserialize JSON/JSONB back to state object.

        Args:
            data (Any): JSON string or dict/list.
            state_class (type): State class type.

        Returns:
            StateT: Deserialized state object.

        Raises:
            Exception: If deserialization fails.
        """
        try:
            if isinstance(data, bytes | bytearray):
                data = data.decode()
            parsed = json.loads(data) if isinstance(data, str) else data
            if isinstance(parsed, dict):
                self._check_state_format(parsed)  # same refusal as _decode_state
            if isinstance(data, str):
                return state_class.model_validate(json.loads(data))
            # Assume it's already a dict/list
            return state_class.model_validate(data)
        except StateFormatVersionError:
            raise
        except Exception:
            # Last-resort: coerce to string and attempt parse, else raise
            if isinstance(data, str):
                return state_class.model_validate(json.loads(data))
            raise

    async def _retry_on_connection_error(
        self,
        operation,
        *args,
        max_retries=3,
        **kwargs,
    ):
        """
        Retry database operations on connection errors.

        Args:
            operation: Callable operation.
            *args: Arguments.
            max_retries (int): Maximum retries.
            **kwargs: Keyword arguments.

        Returns:
            Any: Result of operation or None.

        Raises:
            Exception: If all retries fail.
        """
        last_exception = None

        # Define exception types to catch (only if asyncpg is available)
        exceptions_to_catch: list[type[Exception]] = [ConnectionError]
        if HAS_ASYNCPG and asyncpg:
            exceptions_to_catch.extend(
                [asyncpg.PostgresConnectionError, asyncpg.InterfaceError]
            )

        exception_tuple = tuple(exceptions_to_catch)

        for attempt in range(max_retries):
            try:
                return await operation(*args, **kwargs)
            except exception_tuple as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # exponential backoff
                    logger.warning(
                        "Database connection error on attempt %d/%d, retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        wait_time,
                        e,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                logger.error("Failed after %d attempts: %s", max_retries, e)
                break
            except Exception as e:
                # Don't retry on non-connection errors
                logger.error("Non-retryable error: %s", e)
                raise

        if last_exception:
            raise last_exception
        return None

    async def _run_query(
        self,
        fn,
        *,
        in_transaction: bool = False,
        settings: Sequence[tuple[str, str]] | None = None,
    ):
        """Acquire a pooled connection, run ``fn(conn)``, retrying on conn errors.

        Collapses the ``acquire() as conn`` + ``_retry_on_connection_error``
        boilerplate that every query method previously repeated. When
        ``in_transaction`` is True the callback runs inside ``conn.transaction()``
        so a multi-statement write commits atomically; a connection error retries
        the whole callback on a fresh connection (``max_retries=3``, matching the
        previous per-method wrapping).

        Args:
            fn: Coroutine function taking a single ``conn`` argument.
            in_transaction: Wrap the callback in a database transaction.
            settings: Native RLS values for this call (see ``_isolation``). When given the
                callback always runs in a transaction that first sets them with
                ``set_config(name, value, true)``, so they end with the transaction.

        Returns:
            Whatever ``fn`` returns.
        """

        # First use creates or upgrades the schema (once per process, under an advisory
        # lock), so a fresh database works without a separate setup step.
        await self._initialize_schema()

        async def _operation():
            async with (await self._get_pg_pool()).acquire(
                timeout=self.pg_acquire_timeout
            ) as conn:
                if settings:
                    async with conn.transaction():
                        await apply_settings(conn, settings)
                        return await fn(conn)
                if in_transaction:
                    async with conn.transaction():
                        return await fn(conn)
                return await fn(conn)

        return await self._retry_on_connection_error(_operation, max_retries=3)

    def _run_scoped(
        self,
        fn,
        *,
        settings: Sequence[tuple[str, str]] | None,
        in_transaction: bool = False,
    ):
        """``_run_query`` with RLS settings; without them it is the plain call it always was."""
        if settings:
            return self._run_query(fn, in_transaction=in_transaction, settings=settings)
        if in_transaction:
            return self._run_query(fn, in_transaction=True)
        return self._run_query(fn)

    @staticmethod
    def _loads_jsonb(value: Any) -> Any:
        """Normalize a JSONB column to a Python object.

        asyncpg may hand back a JSONB column either already decoded (dict/list)
        or as raw ``str``/``bytes`` depending on codec configuration. This coerces
        both shapes to the decoded value, replacing the ad-hoc
        ``json.loads(x) if isinstance(x, ...) else x`` guards scattered through
        the row-mapping code.
        """
        if isinstance(value, str | bytes | bytearray):
            return json.loads(value)
        return value

    def _isolation(
        self, config: dict[str, Any] | None, *, write: bool = False
    ) -> _Isolation:
        """Resolve the declared columns' values and the tenant isolation for one call.

        Values come only from ``config["user"]``. The picked tenant columns follow the caller's
        reach (``tenxgraph.core.authz.filter_columns``), and a write always picks reach ``own``
        (every tenant column); with no identity every tenant column applies, whatever
        ``enforce_user_isolation`` says (that setting is about ``user_id`` only).
        ``write=True`` raises when a required column has no value; a read raises when a picked
        column has no value. On a write a missing optional value is compared as "no value"
        instead, so a caller without it owns only threads that have none either.
        """
        # Validates the reach even with no columns.
        picked = self._filter_columns(config, write=write)
        if not self.columns:
            return _NO_ISOLATION
        values = self.columns.resolve(config, write=write)
        specs = self.columns.scope_specs(picked)
        conditions = (
            write_conditions(specs, values)
            if write
            else isolation_conditions(specs, values)
        )
        settings = (
            rls_settings(self.columns.tenant, specs, values)
            if self.row_level_security
            else None
        )
        return _Isolation(values=values, conditions=conditions, settings=settings)

    def _rls(
        self, config: dict[str, Any] | None, *, write: bool = False
    ) -> list[tuple[str, str]] | None:
        """The RLS settings for a call (``None`` when RLS is off). Writes pass ``write=True``
        so the backstop runs at reach ``own``, like the SQL owner checks."""
        if not self.row_level_security:
            return None
        return self._isolation(config, write=write).settings

    def _thread_filter(
        self,
        user_id: str | int | None,
        config: dict[str, Any] | None,
        params: list[Any],
        isolation: _Isolation | None = None,
        *,
        write: bool = False,
    ) -> str:
        """Conditions on the ``threads`` row for this caller: owner and picked tenant columns.

        Appends the values to ``params`` and returns ``user_id = $n AND "org_id" = $m ...``,
        or an empty string when nothing applies (isolation off, no tenant columns).
        ``write=True`` scopes to the caller's own threads whatever the reach.
        """
        isolation = (
            isolation if isolation is not None else self._isolation(config, write=write)
        )
        parts: list[str] = []
        if self._isolation_active(user_id, config, write=write):
            params.append(user_id)
            parts.append(f"user_id = ${len(params)}")
        tenant_sql = build_isolation_where(isolation.conditions, params)
        if tenant_sql:
            parts.append(tenant_sql)
        return " AND ".join(parts)

    def _thread_row_scope(
        self,
        thread_id: str | int,
        user_id: str | int | None,
        config: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        """Build the WHERE fragment + params for a direct ``threads`` row lookup.

        Unlike :meth:`_thread_scope` (which scopes ``states``/``messages`` through
        a subquery), this keys straight off the ``threads`` table's own ``user_id``
        and tenant columns. Shared by the thread read/delete paths.
        """
        params: list[Any] = [thread_id]
        owner = self._thread_filter(user_id, config, params)
        if owner:
            return f"thread_id = $1 AND {owner}", params
        return "thread_id = $1", params

    def _append_owner_scope(
        self,
        query: str,
        params: list[Any],
        user_id: str | int | None,
        config: dict[str, Any] | None,
        *,
        write: bool = False,
    ) -> str:
        """Append the thread-ownership filter to a ``messages`` query when active.

        Mutates ``params`` in place (appending ``user_id`` and tenant values) and returns
        the query with the extra ``AND thread_id IN (...)`` clause. A no-op when isolation
        is off and no tenant column applies, so a single-tenant caller does not pay for the
        join. ``write=True`` scopes to the caller's own threads whatever the reach.
        """
        owner = self._thread_filter(user_id, config, params, write=write)
        if owner:
            query += (
                f" AND thread_id IN (SELECT thread_id FROM {self._get_table_name('threads')} "  # noqa: S608
                f"WHERE {owner})"
            )
        return query

    def _thread_insert_sql(
        self,
        thread_id: str | int,
        thread_name: Any,
        user_id: str | int,
        meta_json: str,
        isolation: _Isolation,
        tail: str,
    ) -> tuple[str, list[Any]]:
        """``INSERT INTO threads`` with the declared column values, plus ``tail`` (conflict)."""
        columns = "thread_id, thread_name, user_id, meta"
        values = "$1, $2, $3, $4"
        params: list[Any] = [thread_id, thread_name, user_id, meta_json]
        for spec in self.columns.all:
            params.append(isolation.values.get(spec.name))
            columns += f", {spec.quoted}"
            values += f", {spec.placeholder(f'${len(params)}')}"
        sql = (
            f"INSERT INTO {self._get_table_name('threads')} ({columns}) "  # noqa: S608
            f"VALUES ({values}) {tail}"
        )
        return sql, params

    def _insert_value_errors(self):
        """Map a codec error on a declared column of the thread insert to ``ColumnValueError``.

        The declared columns follow the four fixed parameters of :meth:`_thread_insert_sql`.
        """
        return column_value_errors(
            {5 + i: spec for i, spec in enumerate(self.columns.all)}
        )

    async def _check_thread_access(
        self,
        conn,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
        isolation: _Isolation,
        *,
        lock: bool,
    ) -> None:
        """Read the thread row back and refuse a writer who does not own it.

        Reading what is actually in the table after the insert-if-absent is race-safe: when a
        concurrent request created the thread, we see the winner's values. Writes always stay
        ``own`` (``isolation`` comes from ``_isolation(config, write=True)``).

        Raises:
            ThreadNotFoundError: The thread belongs to another user or tenant.
        """
        threads = self._get_table_name("threads")
        suffix = " FOR UPDATE" if lock else ""
        row = None
        if self.columns.tenant:
            names = ", ".join(spec.quoted for spec in self.columns.tenant)
            row = await conn.fetchrow(
                f"SELECT user_id, {names} FROM {threads} WHERE thread_id = $1{suffix}",  # noqa: S608
                thread_id,
            )
            owner = row["user_id"] if row is not None else None
        elif lock:
            owner_row = await conn.fetchrow(
                f"SELECT user_id FROM {threads} WHERE thread_id = $1 FOR UPDATE",  # noqa: S608
                thread_id,
            )
            owner = owner_row["user_id"] if owner_row is not None else None
        else:
            owner = await conn.fetchval(
                f"SELECT user_id FROM {threads} WHERE thread_id = $1",  # noqa: S608
                thread_id,
            )
        if owner is None:
            if self.row_level_security:
                # Under RLS a thread of another tenant is invisible rather than mismatched.
                raise ThreadNotFoundError(thread_id)
            raise StorageError(
                message=(
                    "Thread row disappeared during write"
                    if lock
                    else "Thread could not be created"
                ),
                error_code="STORAGE_002",
                context={"thread_id": thread_id},
            )
        if self._isolation_active(user_id, config, write=True) and str(owner) != str(
            user_id
        ):
            raise ThreadNotFoundError(thread_id)
        for spec, value in isolation.conditions:
            if not _same_value(
                spec, row[spec.name] if row is not None else None, value
            ):
                raise ThreadNotFoundError(thread_id)
        # The caller passed the check: later cache and stop keys of this call use the thread's
        # own identity (see _key_identity).
        self._remember_identity(config, thread_id, row)

    def _row_to_thread_info(self, row) -> ThreadInfo:
        """Build a :class:`ThreadInfo` from a ``threads`` row."""
        meta_dict = self._loads_jsonb(row["meta"]) if row["meta"] else {}
        return ThreadInfo(
            thread_id=row["thread_id"],
            thread_name=row["thread_name"],
            user_id=row["user_id"],
            metadata=meta_dict,
            run_id=meta_dict.get("run_id"),
            updated_at=row["updated_at"],
        )

    async def _ensure_thread_exists(
        self,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
    ) -> None:
        """
        Ensure thread exists in database, create if not.

        Args:
            thread_id (str|int): Thread identifier.
            user_id (str|int): User identifier.
            config (dict): Configuration dictionary.

        Returns:
            None

        Raises:
            Exception: If creation fails.
        """
        try:
            isolation = self._isolation(config, write=True)
            insert_sql, insert_params = self._thread_insert_sql(
                thread_id,
                config.get("thread_name", f"Thread {thread_id}"),
                user_id,
                json.dumps(config.get("thread_meta", {})),
                isolation,
                "ON CONFLICT DO NOTHING",
            )

            async def _check_and_create_thread(conn):
                # Insert-if-absent, then verify ownership from what is actually
                # in the table. Checking existence with (thread_id AND user_id)
                # first was unsafe: for a thread owned by ANOTHER user the
                # lookup missed, the INSERT silently no-opped on conflict, and
                # the caller went on to write into that user's thread.
                # Reading the owner back after the insert is also race-safe:
                # if a concurrent request created the thread, we observe the
                # winner's user_id (and tenant values) rather than assuming we created it.
                with self._insert_value_errors():
                    await conn.execute(insert_sql, *insert_params)
                await self._check_thread_access(
                    conn, thread_id, user_id, config, isolation, lock=False
                )

            await self._run_scoped(
                _check_and_create_thread, settings=isolation.settings
            )

        except Exception as e:
            logger.error("Failed to ensure thread exists: %s", e)
            raise

    def _serialize_state_payload(self, state: StateT) -> dict[str, Any]:
        """Build the JSON-safe payload persisted for a state.

        Holds the state data plus a small header (see
        :meth:`BaseCheckpointer._encode_state`). No class path is stored: rows are
        rebuilt into the state class bound by ``StateGraph.compile()``.
        """
        return self._encode_state(state)

    def _deserialize_state_payload(
        self, data: dict[str, Any], thread_id: Any = None
    ) -> StateT:
        """Reconstruct a state from a persisted payload using the bound state class.

        The Redis cache copy also carries the checkpoint version, which is not a
        state field and is dropped here.
        """
        data = dict(data)
        data.pop(_CACHE_VERSION_KEY, None)
        return self._decode_state(data, thread_id)

    def _thread_scope(
        self,
        thread_id: str | int,
        user_id: str | int | None,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, list[Any]]:
        """Build the WHERE fragment + params selecting a thread's rows.

        ``states``/``messages`` carry a ``thread_id`` but no ``user_id`` of their
        own; ownership lives on ``threads``. When isolation is active (per the caller's
        reach, or ``enforce_user_isolation`` when there is no identity) we scope through the
        parent table, so a caller can only touch threads they own. Otherwise we key on
        ``thread_id`` alone and skip the join entirely.

        Returns:
            tuple[str, list]: SQL fragment using $1..$n, and the matching params.
        """
        params: list[Any] = [thread_id]
        owner = self._thread_filter(user_id, config, params)
        if owner:
            return (
                f"thread_id = $1 AND thread_id IN "  # noqa: S608
                f"(SELECT thread_id FROM {self._get_table_name('threads')} "
                f"WHERE {owner})",
                params,
            )
        return ("thread_id = $1", params)

    def _isolation_active(
        self,
        user_id: str | int | None,
        config: dict[str, Any] | None = None,
        *,
        write: bool = False,
    ) -> bool:
        """Whether ownership scoping should be applied for this call.

        Driven by the identity in ``config["user"]`` when present (a write is always scoped
        to its own user), falling back to this checkpointer's ``enforce_user_isolation``
        setting otherwise (so ``config`` is optional and a caller that omits it keeps the
        historical behaviour). Scoping still requires an actual ``user_id`` -- scoping on a
        missing one would match nothing.
        """
        return self._isolation_enabled(config, user_id, write=write)

    def _thread_owned_by_sql(self, user_param: int) -> str:
        """Return a SQL fragment requiring the row's thread to belong to a user.

        Applied to ``messages`` queries, whose rows carry a ``thread_id`` but no
        ``user_id`` of their own. Without this, any authenticated caller who knows
        (or guesses) a ``thread_id`` could read or delete another user's messages.

        Callers must gate this on :meth:`_isolation_active` so a developer who has
        turned isolation off does not pay for the join.
        """
        # Table name is regex-validated by _get_table_name; user_param is a bind
        # placeholder index, never a value. No interpolation of caller data.
        return (
            f"thread_id IN (SELECT thread_id FROM {self._get_table_name('threads')} "  # noqa: S608
            f"WHERE user_id = ${user_param})"
        )

    async def _lock_thread_for_write(
        self,
        conn,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
    ) -> int:
        """Ensure the thread exists, take a row lock, and return its current version.

        Creating the thread (if absent) and locking its row serializes concurrent
        writers on the same thread, so their version numbers cannot collide. The
        ownership check enforces per-user isolation on writes.

        Returns:
            int: The highest existing state ``version`` for the thread (0 if none).

        Raises:
            ThreadNotFoundError: If the thread is owned by a different user or tenant.
        """
        isolation = self._isolation(config, write=True)
        insert_sql, insert_params = self._thread_insert_sql(
            thread_id,
            config.get("thread_name", f"Thread {thread_id}"),
            user_id,
            json.dumps(config.get("thread_meta", {})),
            isolation,
            "ON CONFLICT DO NOTHING",
        )
        with self._insert_value_errors():
            await conn.execute(insert_sql, *insert_params)
        await self._check_thread_access(
            conn, thread_id, user_id, config, isolation, lock=True
        )

        version_row = await conn.fetchrow(
            f"SELECT COALESCE(MAX(version), 0) AS v "  # noqa: S608
            f"FROM {self._get_table_name('states')} WHERE thread_id = $1",
            thread_id,
        )
        return int(version_row["v"]) if version_row else 0

    async def _state_row_matches(
        self, conn, thread_id: str | int, version: int, state_json: str
    ) -> bool:
        """Whether the stored state row at ``version`` holds the same state as ``state_json``."""
        row = await conn.fetchrow(
            f"SELECT state_data FROM {self._get_table_name('states')} "  # noqa: S608
            f"WHERE thread_id = $1 AND version = $2",
            thread_id,
            version,
        )
        if row is None:
            return False
        return self._loads_jsonb(row["state_data"]) == json.loads(state_json)

    async def _write_state_row(
        self,
        conn,
        thread_id: str | int,
        user_id: str | int,
        state: StateT,
        config: dict[str, Any],
        expected_version: int | None,
        *,
        retried: bool = False,
    ) -> int:
        """Append a new versioned state row inside an existing transaction.

        Performs the optimistic compare-and-swap: if ``expected_version`` is set
        and no longer matches the thread's current version, another execution
        committed in the meantime and this write is rejected as stale rather than
        silently overwriting it. Old rows beyond the history window are pruned.

        ``retried`` is set when this is a repeat of a write whose first attempt died on a
        connection error. The first attempt may have committed before the reply was lost; the
        thread is then exactly one version ahead holding this very state, and the write is
        reported as done instead of failing against its own commit.

        Returns:
            int: The new version assigned to the written row.
        """
        current_version = await self._lock_thread_for_write(
            conn, thread_id, user_id, config
        )

        state_json = json.dumps(self._serialize_state_payload(state))
        if (
            retried
            and expected_version is not None
            and current_version == int(expected_version) + 1
            and await self._state_row_matches(
                conn, thread_id, current_version, state_json
            )
        ):
            logger.warning(
                "Write for thread_id=%s had already committed before its retry; keeping it",
                thread_id,
            )
            return current_version

        if expected_version is not None and int(expected_version) != current_version:
            raise StaleStateError(
                message=(
                    "State was modified by another execution "
                    f"(expected version {expected_version}, found {current_version})"
                ),
                error_code="STORAGE_CONFLICT_001",
                context={
                    "thread_id": thread_id,
                    "expected_version": expected_version,
                    "current_version": current_version,
                },
            )

        new_version = current_version + 1
        await conn.execute(
            f"""
            INSERT INTO {self._get_table_name("states")}
                (thread_id, version, state_data, meta)
            VALUES ($1, $2, $3, $4)
            """,  # noqa: S608
            thread_id,
            new_version,
            state_json,
            json.dumps(config.get("meta", {})),
        )

        # Prune history beyond the retention window (bounded append-only log).
        if self.state_history_limit and self.state_history_limit > 0:
            await conn.execute(
                f"DELETE FROM {self._get_table_name('states')} "  # noqa: S608
                f"WHERE thread_id = $1 AND version <= $2",
                thread_id,
                new_version - self.state_history_limit,
            )

        return new_version

    ###########################
    #### STATE METHODS ########
    ###########################

    async def aput_state(
        self,
        config: dict[str, Any],
        state: StateT,
    ) -> StateT:
        """
        Store state in PostgreSQL with optimistic concurrency control.

        A new versioned row is appended under a per-thread row lock. When
        ``config['_checkpoint_version']`` is present (set by a prior read within
        the same run), the write performs a compare-and-swap and raises
        :class:`StaleStateError` if another execution advanced the thread first.
        On success the config's version marker is advanced to the new version and
        the Redis cache is refreshed with the stored state at that version, so the
        next run on the thread starts from it.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.

        Raises:
            StorageError: If storing fails.
            StaleStateError: If the optimistic version check fails.
            ThreadNotFoundError: The thread belongs to another user or tenant.
        """
        thread_id, user_id = self._validate_config(config)
        # A missing or malformed tenant value is a config error: raise it as is, before the
        # generic StorageError wrapping below.
        self._isolation(config, write=True)

        logger.debug("Storing state for thread_id=%s, user_id=%s", thread_id, user_id)
        metrics.counter("pg_checkpointer.save_state.attempts").inc()

        expected_version = config.get(STATE_VERSION_CONFIG_KEY)

        with metrics.timer("pg_checkpointer.save_state.duration"):
            try:
                attempts = 0

                async def _store_state(conn):
                    nonlocal attempts
                    attempts += 1
                    return await self._write_state_row(
                        conn,
                        thread_id,
                        user_id,
                        state,
                        config,
                        expected_version,
                        retried=attempts > 1,
                    )

                new_version = await self._run_scoped(
                    _store_state,
                    in_transaction=True,
                    settings=self._rls(config, write=True),
                )
                config[STATE_VERSION_CONFIG_KEY] = new_version
                logger.debug(
                    "State stored for thread_id=%s at version=%s",
                    thread_id,
                    new_version,
                )
                metrics.counter("pg_checkpointer.save_state.success").inc()
                # Write-through. A run loads its state from the cache first, with the
                # durable version embedded for its compare-and-swap. Without this, a
                # state written here (outside a run) left the cache on the previous
                # state AND version: the next run started from stale state, and its
                # first write raised StaleStateError, so that run and this update were
                # both lost. Best-effort and version-guarded, like every cache write.
                await self.aput_state_cache(config, state)
                return state

            except StaleStateError:
                metrics.counter("pg_checkpointer.save_state.conflict").inc()
                # The cache may hold state built on the stale version; drop it so
                # the next read falls back to Postgres instead of re-seeding the
                # same doomed expected version.
                await self._invalidate_state_cache(thread_id, user_id, config=config)
                raise
            except ThreadNotFoundError:
                raise  # another user's thread: nothing was written, the API answers 404
            except Exception as e:
                metrics.counter("pg_checkpointer.save_state.error").inc()
                logger.error("Failed to store state for thread_id=%s: %s", thread_id, e)
                if asyncpg and hasattr(asyncpg, "ConnectionDoesNotExistError"):
                    connection_errors = (
                        asyncpg.ConnectionDoesNotExistError,
                        asyncpg.InterfaceError,
                    )
                    if isinstance(e, connection_errors):
                        raise TransientStorageError(
                            message=f"Connection issue storing state: {e}",
                            error_code="STORAGE_TRANSIENT_001",
                            context={
                                "thread_id": thread_id,
                                "error_type": type(e).__name__,
                            },
                        ) from e
                raise StorageError(
                    message=f"Failed to store state: {e}",
                    error_code="STORAGE_001",
                    context={
                        "thread_id": thread_id,
                        "error_type": type(e).__name__,
                    },
                ) from e

    async def aput_checkpoint(
        self,
        config: dict[str, Any],
        state: StateT,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateT:
        """
        Atomically persist a state and its messages in a single transaction.

        This is the durable-checkpoint entry point used by the graph loop. State
        and messages are committed together, so a crash can never leave state
        advanced without the messages that justify it (audit M7). The same
        optimistic version check as :meth:`aput_state` applies.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.
            messages (list[Message], optional): Messages to store atomically.
            metadata (dict, optional): Additional message metadata.

        Returns:
            StateT: The stored state object.

        Raises:
            StorageError: If storing fails.
            StaleStateError: If the optimistic version check fails.
            ThreadNotFoundError: The thread belongs to another user or tenant.
        """
        thread_id, user_id = self._validate_config(config)
        self._isolation(
            config, write=True
        )  # config errors raise as is (see aput_state)
        messages = messages or []
        expected_version = config.get(STATE_VERSION_CONFIG_KEY)

        metrics.counter("pg_checkpointer.save_checkpoint.attempts").inc()
        with metrics.timer("pg_checkpointer.save_checkpoint.duration"):
            try:
                attempts = 0

                async def _store_checkpoint(conn):
                    nonlocal attempts
                    attempts += 1
                    version = await self._write_state_row(
                        conn,
                        thread_id,
                        user_id,
                        state,
                        config,
                        expected_version,
                        retried=attempts > 1,
                    )
                    if messages:
                        await self._insert_messages(conn, thread_id, messages, metadata)
                    return version

                new_version = await self._run_scoped(
                    _store_checkpoint,
                    in_transaction=True,
                    settings=self._rls(config, write=True),
                )
                config[STATE_VERSION_CONFIG_KEY] = new_version
                logger.debug(
                    "Checkpoint stored for thread_id=%s at version=%s (%d messages)",
                    thread_id,
                    new_version,
                    len(messages),
                )
                metrics.counter("pg_checkpointer.save_checkpoint.success").inc()
                return state

            except StaleStateError:
                metrics.counter("pg_checkpointer.save_checkpoint.conflict").inc()
                # See aput_state: drop the possibly-poisoned cache so the thread
                # is not wedged into failing every subsequent compare-and-swap.
                await self._invalidate_state_cache(thread_id, user_id, config=config)
                raise
            except ThreadNotFoundError:
                raise  # another user's thread: nothing was written, the API answers 404
            except Exception as e:
                metrics.counter("pg_checkpointer.save_checkpoint.error").inc()
                logger.error(
                    "Failed to store checkpoint for thread_id=%s: %s", thread_id, e
                )
                raise StorageError(
                    message=f"Failed to store checkpoint: {e}",
                    error_code="STORAGE_001",
                    context={"thread_id": thread_id, "error_type": type(e).__name__},
                ) from e

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve the latest state for a thread from PostgreSQL.

        Reads the highest-version row deterministically (``version DESC,
        state_id DESC``) scoped to the requesting ``user_id`` so a developer who
        enables auth cannot read another user's thread (audit B1/B3). The read
        version is recorded in ``config['_checkpoint_version']`` so a subsequent
        write in the same run can perform its compare-and-swap.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.

        Raises:
            Exception: If retrieval fails.
        """
        thread_id, user_id = self._validate_config(config)

        logger.debug(
            "Retrieving state for thread_id=%s, user_id=%s", thread_id, user_id
        )

        try:
            scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)
            query = f"""
                SELECT version, state_data FROM {self._get_table_name("states")}
                WHERE {scope_sql}
                ORDER BY version DESC, state_id DESC
                LIMIT 1
                """  # noqa: S608

            row = await self._run_scoped(
                lambda conn: conn.fetchrow(query, *scope_params),
                settings=self._rls(config),
            )

            if row:
                data = self._loads_jsonb(row["state_data"])
                logger.debug(
                    "State found for thread_id=%s at version=%s",
                    thread_id,
                    row["version"],
                )
                config[STATE_VERSION_CONFIG_KEY] = int(row["version"])
                return self._deserialize_state_payload(data, thread_id)

            logger.debug("No state found for thread_id=%s", thread_id)
            return None

        except Exception as e:
            logger.error("Failed to retrieve state for thread_id=%s: %s", thread_id, e)
            raise

    async def aclear_state(self, config: dict[str, Any]) -> int:
        """
        Clear state from PostgreSQL and Redis cache (scoped by the caller's reach).

        Args:
            config (dict): Configuration dictionary.

        Returns:
            int: The number of state rows (versions) removed; 0 when the caller's scope
            matched nothing. 0 is not "thread not found": the caller's own thread with no
            state also gives 0.

        Raises:
            Exception: If clearing fails.
        """
        thread_id, user_id = self._validate_config(config)

        logger.debug("Clearing state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)
            # With tenant columns the cache key is the thread's identity: read it (scoped)
            # before the rows go, so the shared entry is the one dropped below.
            await self._resolve_thread_identity(config)

            # Clear from PostgreSQL with retry logic, scoped to the owner.
            query = f"DELETE FROM {self._get_table_name('states')} WHERE {scope_sql}"  # noqa: S608
            status = await self._run_scoped(
                lambda conn: conn.execute(query, *scope_params),
                settings=self._rls(config),
            )

            # Clear from Redis cache (best effort: Postgres is already cleared)
            await self._invalidate_state_cache(
                thread_id, user_id, retries=1, config=config
            )
            config.pop(STATE_VERSION_CONFIG_KEY, None)

            logger.debug("State cleared for thread_id=%s", thread_id)
            return _rows_affected(status)

        except Exception as e:
            logger.error("Failed to clear state for thread_id=%s: %s", thread_id, e)
            raise

    async def aput_state_cache(
        self, config: dict[str, Any], state: StateT
    ) -> Any | None:
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
                logger.error("Failed to cache state for thread_id=%s: %s", thread_id, e)
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
                caller_config = {
                    k: v for k, v in config.items() if k != _THREAD_IDENTITY_KEY
                }
                caller_key = self._get_thread_key(thread_id, user_id, caller_config)
                if caller_key not in keys:
                    keys.append(caller_key)
        except (
            ValueError
        ) as e:  # a malformed tenant value: nothing of ours can be cached
            logger.warning(
                "Skipped state cache invalidation for thread_id=%s: %s", thread_id, e
            )
            return
        for attempt in range(retries + 1):
            try:
                await self._redis_call(lambda: self.redis.delete(*keys))
                logger.debug(
                    "Invalidated stale state cache for thread_id=%s", thread_id
                )
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
                "Could not verify state cache for thread_id=%s: %s", thread_id, e
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
                "Could not verify state cache for thread_id=%s: %s", thread_id, e
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

        logger.debug(
            "Getting cached state for thread_id=%s, user_id=%s", thread_id, user_id
        )

        # With tenant columns the key is the thread's identity, so the scoped durable read (the
        # thread row and its current state version) comes first. Built outside the try below:
        # a scope this caller cannot satisfy must raise, never fall back to the cache.
        durable_version: int | None = None
        if self.columns.tenant:
            durable_version = await self._durable_version_for_cache(
                thread_id, user_id, config
            )

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
                    current = (
                        durable_version is None or durable_version == cached_version
                    )
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
                    await self._invalidate_state_cache(
                        thread_id, user_id, config=config
                    )
                else:
                    logger.debug("Cache hit for thread_id=%s", thread_id)
                    if version is not None:
                        config[STATE_VERSION_CONFIG_KEY] = int(version)
                    return self._deserialize_state_payload(data, thread_id)

            # Cache miss - fallback to PostgreSQL
            logger.debug(
                "Cache miss for thread_id=%s, falling back to PostgreSQL", thread_id
            )
            state = await self.aget_state(config)

            # Cache the result for next time
            if state:
                await self.aput_state_cache(config, state)

            return state

        except Exception as e:
            if not isinstance(e, _RedisUnavailable):  # the breaker already logged it
                logger.error(
                    "Failed to get cached state for thread_id=%s: %s", thread_id, e
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
                "Failed to cache value for namespace=%s key=%s: %s", namespace, key, e
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

    async def aclear_cache_value_if(
        self, namespace: str, key: str, expected: Any
    ) -> bool:
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
                await self._redis_call(
                    lambda: self.redis.set(cache_key, payload, ex=ttl, nx=True)
                )
            )
        except _RedisUnavailable:
            return False
        except Exception as e:
            logger.error(
                "Failed to cache value for namespace=%s key=%s: %s", namespace, key, e
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
                re.sub(r"([\\*?\[\]])", r"\\\1", ns_prefix)
                + quote(prefix or "", safe="")
                + "*"
            )
            keys: list[str] = []
            cursor = 0
            while True:
                cursor, found = await self._redis_call(
                    lambda c=cursor: self.redis.scan(cursor=c, match=pattern, count=100)
                )
                for raw_key in found:
                    key_str = (
                        raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                    )
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

    ##################################
    #### TOOL EXECUTION LEDGER #######
    ##################################

    async def aget_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """Return the recorded result of a completed tool call, if there is one.

        Consulted before a tool runs. A hit means this exact tool call already
        completed in an earlier attempt at this node (the process died before the
        node finished, and the node is now being replayed), so the tool must NOT
        be called again -- we return what it returned last time.

        Scoped to the thread's owner when isolation is on (identity reach
        ``own``, else ``enforce_user_isolation``), like the message reads, so a caller
        naming another user's thread gets no record. With isolation off it skips the join.
        """
        thread_id, user_id = self._validate_config(config)

        # Outside the try: a scope this caller cannot satisfy raises instead of reading as
        # "no record".
        params: list[Any] = [thread_id, str(tool_call_id)]
        query = self._append_owner_scope(
            f"SELECT result FROM {self._get_table_name('tool_executions')} "  # noqa: S608
            f"WHERE thread_id = $1 AND tool_call_id = $2",
            params,
            user_id,
            config,
        )
        settings = self._rls(config)
        try:
            row = await self._run_scoped(
                lambda conn: conn.fetchrow(query, *params), settings=settings
            )
            if not row:
                return None

            return self._loads_jsonb(row["result"])

        except Exception as e:
            # A ledger read failure must not take the run down. Falling back to
            # "no record" means the tool runs again -- at-least-once, which is the
            # old behaviour, rather than a hard failure.
            logger.error(
                "Failed to read tool ledger for thread_id=%s tool_call_id=%s: %s",
                thread_id,
                tool_call_id,
                e,
            )
            return None

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Durably record that a tool call completed.

        Written as soon as the tool returns, so a crash later in the same node
        cannot cause it to be re-fired on replay. ON CONFLICT DO NOTHING keeps the
        first recorded result authoritative if this ever races with itself.

        Unlike the read side, a write failure here is raised: silently failing to
        record a completed side effect is exactly what leads to a double charge on
        the next replay, so the caller must know.
        """
        thread_id, user_id = self._validate_config(config)

        await self._ensure_thread_exists(thread_id, user_id, config)

        query = f"""
            INSERT INTO {self._get_table_name("tool_executions")}
                (thread_id, tool_call_id, result)
            VALUES ($1, $2, $3)
            ON CONFLICT (thread_id, tool_call_id) DO NOTHING
            """  # noqa: S608
        await self._run_scoped(
            lambda conn: conn.execute(
                query, thread_id, str(tool_call_id), json.dumps(result)
            ),
            settings=self._rls(config, write=True),
        )
        logger.debug(
            "Recorded tool execution thread_id=%s tool_call_id=%s",
            thread_id,
            tool_call_id,
        )
        return True

    async def aclear_tool_results(
        self,
        config: dict[str, Any],
        origin_message_ids: Sequence[str] | None = None,
    ) -> int:
        """Delete the thread's ledger rows (only those of ``origin_message_ids`` when given).

        A row belongs to an origin id when its key starts with ``"<id>:"``, compared with
        ``left()`` on bound values (no LIKE, so ``%`` and ``_`` in ids stay literal). Scoped
        like the ledger write: the caller's own thread (``user_id`` and every tenant column
        when isolation is on), with the RLS values set when RLS is on.
        """
        if origin_message_ids is not None and not origin_message_ids:
            return 0
        thread_id, user_id = self._validate_config(config)
        params: list[Any] = [thread_id]
        where = "thread_id = $1"
        if origin_message_ids is not None:
            params.append([str(i) for i in origin_message_ids])
            where += (
                " AND EXISTS (SELECT 1 FROM unnest($2::text[]) AS o(id) "
                "WHERE left(tool_call_id, length(o.id) + 1) = o.id || ':')"
            )
        query = self._append_owner_scope(
            f"DELETE FROM {self._get_table_name('tool_executions')} "  # noqa: S608
            f"WHERE {where}",
            params,
            user_id,
            config,
            write=True,
        )
        status = await self._run_scoped(
            lambda conn: conn.execute(query, *params),
            settings=self._rls(config, write=True),
        )
        return _rows_affected(status)

    async def aprune_tool_results(self, older_than: timedelta) -> int:
        """Delete ledger rows of every thread recorded more than ``older_than`` ago.

        Not scoped to a user or tenant. With ``row_level_security=True`` the rows are only
        visible to a role that bypasses RLS (superuser or ``BYPASSRLS``): any other role
        deletes nothing, and a warning says so.
        """
        query = (
            f"DELETE FROM {self._get_table_name('tool_executions')} "  # noqa: S608
            f"WHERE created_at < now() - $1::interval"
        )

        async def _prune(conn) -> Any:
            if self.row_level_security and not await role_bypasses_rls(conn):
                logger.warning(
                    "PgCheckpointer: row_level_security=True and this database role does not "
                    "bypass RLS, so aprune_tool_results sees no rows. Run it with a "
                    "maintenance role that is a superuser or has BYPASSRLS."
                )
            return await conn.execute(query, older_than)

        deleted = _rows_affected(await self._run_query(_prune))
        logger.info("Pruned %d tool ledger rows older than %s", deleted, older_than)
        return deleted

    ###########################
    #### MESSAGE METHODS ######
    ###########################

    async def _insert_messages(
        self,
        conn,
        thread_id: str | int,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert messages on an already-open connection/transaction.

        Shared by :meth:`aput_messages` (standalone) and :meth:`aput_checkpoint`
        (atomic with the state write) so both use identical insert semantics.

        The primary key is ``(thread_id, message_id)``, so ids are per thread and a
        conflict can only be a re-save of a message in this same thread.
        """
        messages_table = self._get_table_name("messages")
        for message in messages:
            await conn.execute(
                f"""
                    INSERT INTO {messages_table} (
                        message_id, thread_id, role, content, tool_calls,
                        tool_call_id, total_tokens, usages, meta
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (thread_id, message_id) DO UPDATE SET
                        role = EXCLUDED.role,
                        content = EXCLUDED.content,
                        tool_calls = EXCLUDED.tool_calls,
                        tool_call_id = EXCLUDED.tool_call_id,
                        total_tokens = EXCLUDED.total_tokens,
                        usages = EXCLUDED.usages,
                        meta = EXCLUDED.meta,
                        updated_at = NOW()
                    """,  # noqa: S608
                message.message_id,
                thread_id,
                message.role,
                json.dumps(
                    [block.model_dump(mode="json") for block in message.content]
                ),
                json.dumps(message.tools_calls) if message.tools_calls else None,
                getattr(message, "tool_call_id", None),
                message.usages.total_tokens if message.usages else 0,
                json.dumps(message.usages.model_dump()) if message.usages else None,
                json.dumps({**(metadata or {}), **(message.metadata or {})}),
            )

    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages in PostgreSQL.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: None

        Raises:
            Exception: If storing fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        if not messages:
            logger.debug("No messages to store for thread_id=%s", thread_id)
            return

        logger.debug("Storing %d messages for thread_id=%s", len(messages), thread_id)

        try:
            # Ensure thread exists
            await self._ensure_thread_exists(thread_id, user_id, config)

            # Store messages in batch with retry logic
            await self._run_scoped(
                lambda conn: self._insert_messages(conn, thread_id, messages, metadata),
                in_transaction=True,
                settings=self._rls(config, write=True),
            )
            logger.debug(
                "Stored %d messages for thread_id=%s", len(messages), thread_id
            )

        except Exception as e:
            logger.error("Failed to store messages for thread_id=%s: %s", thread_id, e)
            raise

    async def aget_message(
        self, config: dict[str, Any], message_id: str | int
    ) -> Message:
        """
        Retrieve a single message by ID.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Message: Retrieved message object.

        Raises:
            Exception: If retrieval fails.
        """
        thread_id = config.get("thread_id")
        user_id = owner_id(config)

        # Message ids are unique per thread only, so the thread is part of the key.
        if not thread_id:
            raise ValueError("thread_id must be provided in config")

        logger.debug("Retrieving message_id=%s for thread_id=%s", message_id, thread_id)

        try:
            query = f"""
                SELECT message_id, thread_id, role, content, tool_calls,
                       tool_call_id, created_at, total_tokens,
                       usages, meta
                FROM {self._get_table_name("messages")}
                WHERE message_id = $1 AND thread_id = $2
            """  # noqa: S608
            params: list[Any] = [message_id, thread_id]
            # Owner scoping: a caller may only read messages on threads they own,
            # even if they know the message/thread id.
            query = self._append_owner_scope(query, params, user_id, config)

            row = await self._run_scoped(
                lambda conn: conn.fetchrow(query, *params), settings=self._rls(config)
            )

            if not row:
                raise ValueError(f"Message not found: {message_id}")

            return self._row_to_message(row)

        except Exception as e:
            logger.error("Failed to retrieve message_id=%s: %s", message_id, e)
            raise

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages for a thread with optional search and pagination.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.

        Raises:
            Exception: If listing fails.
        """
        offset, limit = validate_page(offset, limit)
        thread_id = config.get("thread_id")
        user_id = owner_id(config)

        if not thread_id:
            raise ValueError("thread_id must be provided in config")

        logger.debug("Listing messages for thread_id=%s", thread_id)

        try:
            # Build query with optional search
            query = f"""
                SELECT message_id, thread_id, role, content, tool_calls,
                       tool_call_id, created_at, total_tokens,
                       usages, meta
                FROM {self._get_table_name("messages")}
                WHERE thread_id = $1
            """  # noqa: S608
            params: list[Any] = [thread_id]

            # Owner scoping: listing a thread you do not own returns nothing.
            query = self._append_owner_scope(query, params, user_id, config)

            if search:
                params.append(f"%{search}%")
                query += f" AND content ILIKE ${len(params)}"

            query += " ORDER BY created_at ASC"

            if limit:
                params.append(limit)
                query += f" LIMIT ${len(params)}"

            if offset:
                params.append(offset)
                query += f" OFFSET ${len(params)}"

            rows = await self._run_scoped(
                lambda conn: conn.fetch(query, *params), settings=self._rls(config)
            )
            if not rows:
                rows = []
            messages = [self._row_to_message(row) for row in rows]

            logger.debug("Found %d messages for thread_id=%s", len(messages), thread_id)
            return messages

        except Exception as e:
            logger.error("Failed to list messages for thread_id=%s: %s", thread_id, e)
            raise

    async def adelete_message(
        self,
        config: dict[str, Any],
        message_id: str | int,
    ) -> int:
        """
        Delete a message by ID.

        The message is also removed from every retained state version. When that rewrites a
        state, the thread's latest state row moves to a new version, so every cached copy of
        the old state (under any key) fails the version check in ``aget_state_cache`` and is
        dropped, and a run still holding the old state gets ``StaleStateError`` on its next
        save instead of writing the deleted message back.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            int: The number of messages removed; 0 when the message does not exist or the
            caller's scope does not reach its thread.

        Raises:
            Exception: If deletion fails.
        """
        thread_id = config.get("thread_id")
        user_id = owner_id(config)

        # Message ids are unique per thread only, so the thread is part of the key.
        if not thread_id:
            raise ValueError("thread_id must be provided in config")

        logger.debug("Deleting message_id=%s for thread_id=%s", message_id, thread_id)

        try:
            query = (
                f"DELETE FROM {self._get_table_name('messages')} "  # noqa: S608
                f"WHERE message_id = $1 AND thread_id = $2"
            )
            params: list[Any] = [message_id, thread_id]
            # Owner scoping: never let a caller delete a message on a thread they
            # do not own, even with a known message_id.
            query = self._append_owner_scope(query, params, user_id, config)
            scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)
            states = self._get_table_name("states")
            threads = self._get_table_name("threads")

            async def _delete(conn):
                # Same row lock a state write takes, so a writer cannot re-save the state
                # between our read and our rewrite (or take the version we move to below).
                if self.columns.tenant:
                    # Read the thread's identity with the lock, under the caller's scope, so
                    # the cache entry dropped below is the thread's shared one.
                    row_sql, row_params = self._thread_row_scope(
                        thread_id, user_id, config
                    )
                    owner_row = await conn.fetchrow(
                        f"SELECT {self._identity_columns_sql()} FROM {threads} "  # noqa: S608
                        f"WHERE {row_sql} FOR UPDATE",
                        *row_params,
                    )
                    self._remember_identity(config, thread_id, owner_row)
                else:
                    await conn.execute(
                        f"SELECT 1 FROM {threads} WHERE thread_id = $1 FOR UPDATE",  # noqa: S608
                        thread_id,
                    )
                deleted = _rows_affected(await conn.execute(query, *params))
                # The message also lives inside every retained state version; leaving it
                # there brings it back on the next run and in GET state.
                rows = await conn.fetch(
                    f"SELECT state_id, version, state_data FROM {states} "  # noqa: S608
                    f"WHERE {scope_sql}",
                    *scope_params,
                )
                rewritten = False
                for row in rows:
                    payload = self._loads_jsonb(row["state_data"])
                    if self._strip_message_from_payload(payload, message_id):
                        rewritten = True
                        await conn.execute(
                            f"UPDATE {states} SET state_data = $1 WHERE state_id = $2",  # noqa: S608
                            json.dumps(payload),
                            row["state_id"],
                        )
                if rewritten:
                    # An in-place rewrite keeps MAX(version), so a cached copy of the old
                    # state (under any caller's key) would still pass the version check.
                    # Moving the latest row to a new version makes every such copy stale.
                    latest = max(
                        rows, key=lambda r: (int(r["version"]), int(r["state_id"]))
                    )
                    await conn.execute(
                        f"UPDATE {states} SET version = $1 WHERE state_id = $2",  # noqa: S608
                        int(latest["version"]) + 1,
                        latest["state_id"],
                    )
                return deleted

            deleted = await self._run_scoped(
                _delete, in_transaction=True, settings=self._rls(config)
            )
            # The cached state carries the message too. Dropping it makes the next read
            # rebuild from Postgres.
            if user_id:
                await self._invalidate_state_cache(
                    thread_id, user_id, retries=1, config=config
                )
            logger.debug("Deleted message_id=%s (%d row)", message_id, deleted)
            return deleted

        except Exception as e:
            logger.error("Failed to delete message_id=%s: %s", message_id, e)
            raise

    async def aupdate_message_metadata(
        self,
        config: dict[str, Any],
        message_id: str | int,
        metadata: dict[str, Any],
    ) -> Message | None:
        """
        Shallow-merge ``metadata`` into a stored message's metadata.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.
            metadata (dict): Keys to set; a key replaces the same stored key.

        Returns:
            Message | None: The updated message, or None if it does not exist in this
            thread (or the caller does not own the thread: a write stays own whatever the
            reach).
        """
        thread_id = config.get("thread_id")
        user_id = owner_id(config)

        if not thread_id:
            raise ValueError("thread_id must be provided in config")

        logger.debug(
            "Updating metadata of message_id=%s for thread_id=%s", message_id, thread_id
        )

        try:
            query = f"""
                UPDATE {self._get_table_name("messages")}
                SET meta = COALESCE(meta, '{{}}'::jsonb) || $3::jsonb, updated_at = NOW()
                WHERE message_id = $1 AND thread_id = $2
            """  # noqa: S608
            params: list[Any] = [message_id, thread_id, json.dumps(metadata)]
            # Owner scoping: a thread you do not own is reported as "not found". A write, so
            # it stays own whatever the reach.
            query = self._append_owner_scope(query, params, user_id, config, write=True)
            query += """
                RETURNING message_id, thread_id, role, content, tool_calls,
                          tool_call_id, created_at, total_tokens,
                          usages, meta
            """

            row = await self._run_scoped(
                lambda conn: conn.fetchrow(query, *params),
                settings=self._rls(config, write=True),
            )
            return self._row_to_message(row) if row else None

        except Exception as e:
            logger.error(
                "Failed to update metadata of message_id=%s: %s", message_id, e
            )
            raise

    def _row_to_message(self, row) -> Message:  # noqa: PLR0912, PLR0915
        """
        Convert database row to Message object with robust JSON handling.

        Args:
            row: Database row.

        Returns:
            Message: Message object.
        """
        from tenxgraph.core.state.message import TokenUsages

        # Handle usages JSONB
        usages = None
        if row["usages"]:
            try:
                usages = TokenUsages(**self._loads_jsonb(row["usages"]))
            except Exception:
                usages = None

        # Handle tool_calls JSONB
        tool_calls = None
        if row["tool_calls"]:
            try:
                tool_calls = self._loads_jsonb(row["tool_calls"])
            except Exception:
                tool_calls = None

        # Handle meta JSONB
        metadata = {}
        if row["meta"]:
            try:
                metadata = self._loads_jsonb(row["meta"])
            except Exception:
                metadata = {}

        # Handle content TEXT/JSONB -> list of blocks
        content_raw = row["content"]
        content_value: list[Any] = []
        if content_raw is None:
            content_value = []
        elif isinstance(content_raw, bytes | bytearray):
            try:
                parsed = json.loads(content_raw.decode())
                if isinstance(parsed, list):
                    content_value = parsed
                elif isinstance(parsed, dict):
                    content_value = [parsed]
                else:
                    content_value = [
                        {"type": "text", "text": str(parsed), "annotations": []}
                    ]
            except Exception:
                content_value = [
                    {
                        "type": "text",
                        "text": content_raw.decode(errors="ignore"),
                        "annotations": [],
                    }
                ]
        elif isinstance(content_raw, str):
            # Try JSON parse first
            try:
                parsed = json.loads(content_raw)
                if isinstance(parsed, list):
                    content_value = parsed
                elif isinstance(parsed, dict):
                    content_value = [parsed]
                else:
                    content_value = [
                        {"type": "text", "text": content_raw, "annotations": []}
                    ]
            except Exception:
                content_value = [
                    {"type": "text", "text": content_raw, "annotations": []}
                ]
        elif isinstance(content_raw, list):
            content_value = content_raw
        elif isinstance(content_raw, dict):
            content_value = [content_raw]
        else:
            content_value = [
                {"type": "text", "text": str(content_raw), "annotations": []}
            ]

        return Message(
            message_id=row["message_id"],
            role=row["role"],
            content=content_value,
            tools_calls=tool_calls,
            timestamp=row["created_at"].timestamp() if row["created_at"] else 0,
            metadata=metadata,
            usages=usages,
        )

    ###########################
    #### THREAD METHODS #######
    ###########################

    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> Any | None:
        """
        Create or update thread information.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        The caller becomes the owner of a new thread (``thread_info.user_id`` is ignored).
        An update stays own whatever the caller's reach.

        Returns:
            bool: True if thread was created (inserted).
                  False if thread already existed and was updated.

        Raises:
            ThreadNotFoundError: The thread belongs to another user or tenant.
            Exception: If storing fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        logger.debug(
            "Storing thread info for thread_id=%s, user_id=%s", thread_id, user_id
        )

        try:
            thread_name = thread_info.thread_name
            meta = thread_info.metadata or {}
            meta.update(
                {
                    "run_id": thread_info.run_id,
                }
            )

            isolation = self._isolation(config, write=True)
            insert_sql, insert_params = self._thread_insert_sql(
                thread_id,
                thread_name,
                user_id,
                json.dumps(meta),
                isolation,
                "ON CONFLICT (thread_id) DO NOTHING RETURNING thread_id",
            )

            async def _put_thread(conn):
                # Try to insert; if inserted, RETURNING will return the row. Tenant and extra
                # values are written here only: they never change afterwards.
                with self._insert_value_errors():
                    row = await conn.fetchrow(insert_sql, *insert_params)

                if row:
                    # Insert succeeded
                    return True

                # Insert did nothing (conflict) -> update the existing row.
                # When isolation is on, the UPDATE is scoped by user_id too (whatever the
                # reach: writes stay own): without that, any caller could rename or re-tag
                # another user's thread. RETURNING lets us detect that case.
                # Only set thread_name when a non-None value is provided.
                scoped = self._isolation_active(user_id, config, write=True)

                if thread_name is None:
                    sets, params = "meta = $1, updated_at = NOW()", [json.dumps(meta)]
                else:
                    sets = "thread_name = $1, meta = $2, updated_at = NOW()"
                    params = [thread_name, json.dumps(meta)]

                params.append(thread_id)
                where = f"thread_id = ${len(params)}"
                if scoped:
                    params.append(user_id)
                    where += f" AND user_id = ${len(params)}"
                tenant_sql = build_isolation_where(isolation.conditions, params)
                if tenant_sql:
                    where += f" AND {tenant_sql}"
                    scoped = True

                updated = await conn.fetchrow(
                    f"""
                    UPDATE {self._get_table_name("threads")}
                    SET {sets}
                    WHERE {where}
                    RETURNING thread_id
                    """,  # noqa: S608
                    *params,
                )

                if updated is None and scoped:
                    # The thread exists but is owned by someone else (or was deleted since
                    # the insert): never revealed, the API answers 404.
                    raise ThreadNotFoundError(thread_id)

                return False

            created = await self._run_scoped(_put_thread, settings=isolation.settings)
            logger.debug(
                "Thread info stored for thread_id=%s (created=%s)", thread_id, created
            )
            return bool(created)

        except Exception as e:
            logger.error(
                "Failed to store thread info for thread_id=%s: %s", thread_id, e
            )
            raise e

    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        """
        Get thread information.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.

        Raises:
            Exception: If retrieval fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        logger.debug(
            "Retrieving thread info for thread_id=%s, user_id=%s", thread_id, user_id
        )

        try:
            # Owner-scoped unless the developer disabled isolation.
            where, params = self._thread_row_scope(thread_id, user_id, config)
            query = f"""
                SELECT thread_id, thread_name, user_id, created_at, updated_at, meta
                FROM {self._get_table_name("threads")}
                WHERE {where}
                """  # noqa: S608

            row = await self._run_scoped(
                lambda conn: conn.fetchrow(query, *params), settings=self._rls(config)
            )

            if row:
                # The row's owner, not the caller: they differ under reach tenant or all.
                return self._row_to_thread_info(row)

            logger.debug(
                "Thread not found for thread_id=%s, user_id=%s", thread_id, user_id
            )
            return None

        except Exception as e:
            logger.error(
                "Failed to retrieve thread info for thread_id=%s: %s", thread_id, e
            )
            raise e

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the ``user_id`` that owns ``thread_id`` (global, not owner-scoped).

        Resolves ownership across all users so an authorization layer can reject a
        different user acting on the thread. Returns None if no such thread exists.
        """

        try:
            thread_id = self._db_thread_id(thread_id)
        except ValueError:
            return None  # not a valid id for this table, so no such thread exists
        query = f"""
            SELECT user_id
            FROM {self._get_table_name("threads")}
            WHERE thread_id = $1
            """  # noqa: S608
        return await self._run_scoped(
            lambda conn: conn.fetchval(query, thread_id),
            settings=self._global_settings(),
        )

    def _global_settings(self) -> list[tuple[str, str]] | None:
        """RLS settings for a deliberate cross-tenant lookup (owner/scope resolution)."""
        if not self.row_level_security:
            return None
        return rls_all_rows_settings(self.columns.tenant)

    async def aget_thread_scope(self, thread_id: str | int) -> dict[str, Any] | None:
        """Return ``{"user_id": ..., <tenant column>: ...}`` for a thread, globally.

        Not scoped to any caller (like :meth:`aget_thread_owner`): an authorization layer
        compares these values with the caller's on every column the caller's reach picks.
        Tenant values written before a column existed are ``None``. Returns ``None`` when the
        thread does not exist.
        """
        try:
            thread_id = self._db_thread_id(thread_id)
        except ValueError:
            return None
        names = "".join(f", {spec.quoted}" for spec in self.columns.tenant)
        query = (
            f"SELECT user_id{names} FROM {self._get_table_name('threads')} "  # noqa: S608
            "WHERE thread_id = $1"
        )
        row = await self._run_scoped(
            lambda conn: conn.fetchrow(query, thread_id),
            settings=self._global_settings(),
        )
        if row is None:
            return None
        scope: dict[str, Any] = {"user_id": row["user_id"]}
        for spec in self.columns.tenant:
            scope[spec.name] = row[spec.name]
        return scope

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads for a user with optional search, filters and pagination.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.
            filters (dict, optional): Conditions on ``thread_name``, ``created_at``,
                ``updated_at`` and the declared extra columns, e.g.
                ``{"plan": "pro", "created_at": {"gte": "2026-01-01T00:00:00+00:00"}}``
                (operators and limits in ``tenxgraph.storage.sql_filter``). ANDed after the
                isolation conditions, so they can only narrow the listing. Tenant columns and
                ``user_id`` are rejected.

        Returns:
            list[ThreadInfo]: List of thread information objects.

        Raises:
            FilterError: ``filters`` is malformed or names a column it may not.
            Exception: If listing fails.
        """
        offset, limit = validate_page(offset, limit)
        conditions = validate_filters(filters, self.filter_schema)
        user_id = owner_id(config)

        # `user_id = user_id or "test-user"` used to sit here, which made the check
        # below dead code and silently listed a fake user's threads whenever the
        # caller forgot user_id. With isolation on that is a confusing empty result;
        # with it off it is a cross-tenant listing. Ask for the id instead.
        if not user_id and self.enforce_user_isolation:
            raise ValueError("user_id must be provided in config to list threads")

        logger.debug("Listing threads for user_id=%s", user_id)

        try:
            # Build query with optional search. With isolation on, a user only
            # ever sees their own threads; with it off (single-tenant) the listing
            # is not restricted by user_id.
            query = f"""
                SELECT thread_id, thread_name, user_id, created_at, updated_at, meta
                FROM {self._get_table_name("threads")}
                WHERE TRUE
            """  # noqa: S608
            params: list[Any] = []

            owner = self._thread_filter(user_id, config, params)
            if owner:
                query += f" AND {owner}"

            if search:
                params.append(f"%{search}%")
                query += f" AND thread_name ILIKE ${len(params)}"

            # Client filters come last and are only ever ANDed, so they cannot widen the
            # isolation above; their $n numbering continues from ``params``.
            filter_sql = compile_filters(conditions, params)
            if filter_sql:
                query += f" AND {filter_sql}"

            query += " ORDER BY updated_at DESC"

            if limit:
                params.append(limit)
                query += f" LIMIT ${len(params)}"

            if offset:
                params.append(offset)
                query += f" OFFSET ${len(params)}"

            rows = await self._run_scoped(
                lambda conn: conn.fetch(query, *params), settings=self._rls(config)
            )
            if not rows:
                rows = []

            threads = [self._row_to_thread_info(row) for row in rows]
            logger.debug("Found %d threads for user_id=%s", len(threads), user_id)
            return threads

        except Exception as e:
            logger.error("Failed to list threads for user_id=%s: %s", user_id, e)
            raise

    async def aclean_thread(self, config: dict[str, Any]) -> int:
        """
        Clean/delete a thread and all associated data (scoped by the caller's reach).

        Args:
            config (dict): Configuration dictionary.

        Returns:
            int: 1 when the thread was deleted, 0 when there is none the caller may see.

        Raises:
            Exception: If cleaning fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        logger.debug("Cleaning thread thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Owner-scoped unless the developer disabled isolation, so one user
            # cannot delete another's thread by id.
            where, params = self._thread_row_scope(thread_id, user_id, config)

            # Delete thread (cascade will handle messages and states) with retry logic
            query = f"DELETE FROM {self._get_table_name('threads')} WHERE {where}"  # noqa: S608
            if self.columns.tenant:
                # The deleted row's identity names the thread's shared cache entry.
                query += f" RETURNING {self._identity_columns_sql()}"
                row = await self._run_scoped(
                    lambda conn: conn.fetchrow(query, *params),
                    settings=self._rls(config),
                )
                self._remember_identity(config, thread_id, row)
                deleted = 0 if row is None else 1
            else:
                status = await self._run_scoped(
                    lambda conn: conn.execute(query, *params),
                    settings=self._rls(config),
                )
                deleted = _rows_affected(status)

            # Postgres is the source of truth and is already gone. A Redis failure here
            # must not turn a completed delete into an error; the stale cache entry is
            # ignored on read (its version no longer matches Postgres, see
            # aget_state_cache) and expires with its TTL.
            await self._invalidate_state_cache(
                thread_id, user_id, retries=1, config=config
            )

            logger.debug("Thread cleaned: thread_id=%s, user_id=%s", thread_id, user_id)
            return deleted

        except Exception as e:
            logger.error("Failed to clean thread thread_id=%s: %s", thread_id, e)
            raise

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
            async with (await self._get_pg_pool()).acquire(
                timeout=self.pg_acquire_timeout
            ) as conn:
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
                logger.error("Error closing Redis connection: %s", e)
                errors.append(f"Redis: {e}")

        # Close PostgreSQL pool
        if close_pg:
            try:
                if self._pg_pool and not self._pg_pool.is_closing():
                    await self._pg_pool.close()
                logger.debug("PostgreSQL pool closed")
            except Exception as e:
                logger.error("Error closing PostgreSQL pool: %s", e)
                errors.append(f"PostgreSQL: {e}")

        if errors:
            error_msg = f"Errors during resource cleanup: {'; '.join(errors)}"
            logger.warning(error_msg)
            # Don't raise - cleanup should be best effort
        else:
            logger.info("All resources released successfully")
