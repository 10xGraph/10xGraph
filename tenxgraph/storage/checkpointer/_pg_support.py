"""Shared imports, constants and small helpers of the Postgres + Redis checkpointer.

``pg_checkpointer.py`` re-exports everything here, so existing imports keep working.
"""

import uuid
from typing import Any, TypeVar

from tenxgraph.core.state import AgentState
from tenxgraph.storage.columns import ColumnSpec


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
# ``DEFAULT_DDL_LOCK_TIMEOUT`` (setup DDL lock wait) is shared with PgVectorStore.
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


class _RedisUnavailable(Exception):
    """A Redis call was skipped (breaker open) or failed to connect (breaker now open)."""


# SQL type mapping for ID types
ID_TYPE_MAP = {
    "string": "VARCHAR(255)",
    "int": "SERIAL",
    "bigint": "BIGSERIAL",
}


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
