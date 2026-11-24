"""Fixtures and the known-gap registry for the storage contract suites.

Backends: ``memory``, ``sqlite``, ``pg`` (Postgres + Redis, marked ``integration``).
Postgres needs ``--integration`` plus ``POSTGRES_DSN`` / ``REDIS_URL`` (the variables
``tests/integration/`` uses); without a reachable database those tests skip.

``KNOWN_GAPS`` maps ``(test name, backend)`` to a reason. Every entry becomes
``xfail(strict=True)``, so a fixed gap turns the suite red until the entry is removed.
"""

# SQL here is built only from identifiers the fixtures generate (uuid hex), never input.
# ruff: noqa: S608, E501

import asyncio
import os
import uuid

import pytest
import pytest_asyncio

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer


POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
TENANTS = {"org_id": "text", "project_id": "text"}

# (test function name, backend id) -> reason. Backend ids: memory, sqlite, pg for the
# checkpointers; inmem, pgvector, qdrant for the stores. "pg" and "pgvector" entries come from
# reading the code (no database was available when the suite was written): unverified.
# No checkpointer gaps are open.
KNOWN_GAPS: dict[tuple[str, str], str] = {
    # ---- stores: verified by running inmem and qdrant (local mode); pgvector not run
}


# Entries whose test can pass by chance (xfail not strict). None are open.
NON_STRICT: set[tuple[str, str]] = set()


def pytest_collection_modifyitems(config, items):
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        backend = callspec.params.get("backend")
        name = item.originalname
        reason = KNOWN_GAPS.get((name, backend))
        if reason:
            strict = (name, backend) not in NON_STRICT
            item.add_marker(pytest.mark.xfail(strict=strict, reason=reason))


# ---------------------------------------------------------------- checkpointers

CP_BACKENDS = [
    "memory",
    "sqlite",
    pytest.param("pg", marks=pytest.mark.integration),
]


@pytest_asyncio.fixture
async def pg_env(backend):
    if backend != "pg":
        yield None, None
        return
    asyncpg = pytest.importorskip("asyncpg")
    try:
        pool = await asyncio.wait_for(
            asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=4), timeout=5
        )
    except Exception as e:
        pytest.skip(f"Postgres not reachable at POSTGRES_DSN: {e}")
    name = f"tg_{uuid.uuid4().hex[:10]}"
    await pool.execute(f'CREATE SCHEMA "{name}"')
    yield pool, name
    await pool.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
    await pool.close()


@pytest_asyncio.fixture(params=CP_BACKENDS)
async def backend(request):
    return request.param


@pytest_asyncio.fixture
async def make_cp(backend, tmp_path, pg_env):
    """Factory ``await make_cp(**pg_kwargs)``; extra kwargs are used by Postgres only."""
    created = []
    prefix = f"tgcontract:{uuid.uuid4().hex[:8]}:"
    _pool, pg_schema = pg_env

    async def make(**kw):
        if backend == "memory":
            cp = InMemoryCheckpointer()
        elif backend == "sqlite":
            cp = SqliteCheckpointer(tmp_path / f"cp_{len(created)}.db")
        else:
            from tenxgraph.storage.checkpointer import PgCheckpointer

            kw.setdefault("tenant_columns", TENANTS)
            cp = PgCheckpointer(
                postgres_dsn=POSTGRES_DSN,
                redis_url=REDIS_URL,
                schema=pg_schema,
                redis_key_prefix=prefix,
                **kw,
            )
            await cp.asetup()
        created.append(cp)
        return cp

    yield make
    for cp in created:
        if backend == "pg":
            try:
                keys = [k async for k in cp.redis.scan_iter(match=f"{prefix}*")]
                if keys:
                    await cp.redis.delete(*keys)
            except Exception:  # noqa: S110 - best-effort test cleanup
                pass
        await cp.arelease()


@pytest_asyncio.fixture
async def cp(make_cp):
    return await make_cp()


def cfg(thread="t1", user="alice", org="A", project="P1", reach="own"):
    """A config carrying an identity; the tenant is ignored by backends without tenant columns."""
    identity = {
        "user_id": user,
        "tenant": {"org_id": org, "project_id": project},
        "reach": reach,
        "claims": {},
    }
    return {"thread_id": thread, "user_id": user, "user": identity}
