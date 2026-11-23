"""Postgres plumbing shared by ``PgCheckpointer`` and ``PgVectorStore``. Private.

Retry on a lost connection, running a callback on a pooled connection (with the RLS
``set_config`` values when given), JSONB decoding, the schema-version runner and the timeout
defaults. Tenant isolation rules live in :mod:`tenxgraph.storage.columns`.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from tenxgraph.core.exceptions.storage_exceptions import SchemaVersionError
from tenxgraph.storage.columns import apply_settings
from tenxgraph.utils.log_safety import describe_error


try:
    import asyncpg
except ImportError:  # pragma: no cover - exercised only without the extra
    asyncpg = None  # type: ignore[assignment]


#: Seconds to wait for a free pooled connection.
DEFAULT_ACQUIRE_TIMEOUT = 30.0
#: Seconds a setup DDL statement waits for a table lock before setup fails with
#: ``SetupLockTimeoutError`` (``ddl_lock_timeout`` option; None waits forever).
DEFAULT_DDL_LOCK_TIMEOUT = 10.0
#: Timeouts of a pool the backend builds (a pool you pass in keeps its own settings). Without
#: them a blackholed Postgres hangs a run instead of failing it.
DEFAULT_POOL_CONFIG: dict[str, Any] = {
    "command_timeout": 60.0,  # per query
    "timeout": 30.0,  # per new connection
}
#: Attempts for an operation that loses its connection (backoff 1s, then 2s).
MAX_RETRIES = 3


def connection_errors() -> tuple[type[BaseException], ...]:
    """The errors that mean "the connection was lost", which are worth a retry."""
    errors: list[type[BaseException]] = [ConnectionError]
    if asyncpg is not None:
        errors.extend([asyncpg.PostgresConnectionError, asyncpg.InterfaceError])
    return tuple(errors)


async def retry_on_connection_error(
    operation: Callable[[], Awaitable[Any]],
    *,
    owner: str,
    logger: logging.Logger,
    max_retries: int = MAX_RETRIES,
) -> Any:
    """Run ``operation()``, retrying on a lost connection with backoff 1s, 2s, 4s...

    Every operation is one statement or one transaction, so a retry never applies a write
    twice. Any other error propagates unchanged and is not logged here: the caller logs it
    once, with its own context. Returns ``None`` only when ``max_retries`` is below 1.
    """
    errors = connection_errors()
    for attempt in range(max_retries):
        try:
            return await operation()
        except errors as e:
            if attempt == max_retries - 1:
                logger.error(
                    "%s failed after %d attempts: %s", owner, max_retries, describe_error(e)
                )
                raise
            wait = 2**attempt
            logger.warning(
                "%s connection error (attempt %d/%d), retrying in %ds: %s",
                owner,
                attempt + 1,
                max_retries,
                wait,
                describe_error(e),
            )
            await asyncio.sleep(wait)
    return None


async def run_on_connection(
    pool: Any,
    fn: Callable[[Any], Awaitable[Any]],
    *,
    timeout: float | None,
    settings: Sequence[tuple[str, str]] | None = None,
    in_transaction: bool = False,
) -> Any:
    """Acquire a connection from ``pool`` and return ``await fn(conn)``.

    With ``settings`` the callback runs in a transaction that first sets each
    ``(name, value)`` with ``set_config(name, value, true)`` (RLS values, search settings),
    so they end with the transaction. ``in_transaction=True`` runs it in a transaction too.
    """
    async with pool.acquire(timeout=timeout) as conn:
        if settings or in_transaction:
            async with conn.transaction():
                if settings:
                    await apply_settings(conn, settings)
                return await fn(conn)
        return await fn(conn)


def loads_jsonb(value: Any) -> Any:
    """Decode a JSONB column that a connection may return as text, bytes or an object.

    ``None`` (SQL ``NULL``) stays ``None``; callers that need a default apply it.
    """
    if isinstance(value, str | bytes | bytearray):
        return json.loads(value)
    return value


async def recorded_schema_version(conn: Any, version_table: str) -> int:
    """The highest version in ``version_table`` (a quoted, schema-qualified name); 0 if none."""
    recorded = await conn.fetchval(f"SELECT max(version) FROM {version_table}")  # noqa: S608
    return int(recorded or 0)


async def apply_migrations(
    conn: Any,
    *,
    version_table: str,
    target: int,
    migrations: Mapping[int, Sequence[str]],
    owner: str,
    logger: logging.Logger,
) -> None:
    """Apply every migration after the recorded version up to ``target``, then record it.

    Runs inside the caller's setup transaction, in ascending version order. Migrations are
    idempotent DDL, so this is safe on fresh and upgraded databases. Any failure raises
    ``SchemaVersionError`` (the original error is its cause, so a lock timeout is still
    reported as ``SetupLockTimeoutError`` by ``ddl_session``).
    """
    try:
        current = await recorded_schema_version(conn, version_table)
        if current >= target:
            return
        logger.info("Upgrading %s schema from version %d to %d", owner, current, target)
        for version in range(current + 1, target + 1):
            for statement in migrations.get(version, ()):
                logger.debug("Applying migration step for v%d: %s", version, statement.strip())
                await conn.execute(statement)
        await conn.execute(
            f"INSERT INTO {version_table} (version) VALUES ($1) "  # noqa: S608
            "ON CONFLICT (version) DO NOTHING",
            target,
        )
    except Exception as e:
        raise SchemaVersionError(
            message=f"{owner} failed to apply schema migrations: {describe_error(e)}",
            error_code="STORAGE_SCHEMA_001",
            context={"error_type": type(e).__name__},
        ) from e
