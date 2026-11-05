"""Sprint 7 review fixes in PgCheckpointer against a REAL Postgres + Redis.

S1: under a scope subset a deleted message (or a cleared state) never comes back from the Redis
state cache; callers allowed to see a thread share its cache entry and its stop key; isolation
still holds on a hit. S2/S6: setup with everything in place does not wait behind open writes,
``ddl_lock_timeout`` raises a clear error, and missing indexes on a live table are built
concurrently (an invalid leftover is rebuilt).

    POSTGRES_DSN=postgresql://test:test@localhost:55432/test_agentflow \\
    REDIS_URL=redis://localhost:56379/1 \\
        pytest tests/integration/test_pg_checkpointer_review_fixes_real.py --integration

Each test gets its own Postgres schema (dropped afterwards) and its own Redis key prefix.
"""

# SQL here is built only from identifiers the fixtures generate (uuid hex), never input.
# ruff: noqa: S608

import asyncio
import os
import time
import uuid

import pytest
import pytest_asyncio

from tenxgraph.core.authz import build_authz
from tenxgraph.core.exceptions import StaleStateError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import PgCheckpointer
from tenxgraph.storage.columns import SetupLockTimeoutError


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
TENANTS = {"org_id": "text", "project_id": "text"}
USER_ORG = ["user_id", "org_id"]
ORG_ADMIN = ["org_id"]


@pytest_asyncio.fixture
async def admin_pool():
    asyncpg = pytest.importorskip("asyncpg")
    try:
        pool = await asyncio.wait_for(
            asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=4), timeout=5
        )
    except Exception as e:
        pytest.skip(f"Postgres not reachable at POSTGRES_DSN: {e}")
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def schema(admin_pool):
    name = f"rf_{uuid.uuid4().hex[:10]}"
    await admin_pool.execute(f'CREATE SCHEMA "{name}"')
    yield name
    await admin_pool.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')


@pytest_asyncio.fixture
async def make_cp(schema):
    created = []
    prefix = f"rftest:{uuid.uuid4().hex[:8]}:"

    async def make(**kw):
        kw.setdefault("tenant_columns", TENANTS)
        cp = PgCheckpointer(
            postgres_dsn=POSTGRES_DSN,
            redis_url=REDIS_URL,
            schema=schema,
            redis_key_prefix=prefix,
            **kw,
        )
        created.append(cp)
        return cp

    yield make
    for cp in created:
        try:
            keys = [k async for k in cp.redis.scan_iter(match=f"{prefix}*")]
            if keys:
                await cp.redis.delete(*keys)
        except Exception:  # noqa: S110 - best-effort test cleanup
            pass
        await cp.arelease()


def _cfg(thread, user="alice", org="A", project="P1", scope=USER_ORG, run_id=None):
    identity = {"user_id": user, "org_id": org, "project_id": project}
    identity["authz"] = build_authz(user, scope=list(scope))
    cfg = {"thread_id": thread, "user_id": user, "user": identity}
    if run_id:
        cfg["run_id"] = run_id
    return cfg


def _texts(state):
    return [m.text() for m in state.context] if state is not None else None


async def _seed(cp, thread):
    """Owner alice in A/P1 writes two messages and caches the state (repro r4)."""
    c1 = _cfg(thread)
    keep = Message.text_message("keep me", role="user")
    secret = Message.text_message("secret to delete", role="user")
    state = AgentState()
    state.context = [keep, secret]
    await cp.aput_state(c1, state)
    await cp.aput_messages(c1, [keep, secret])
    await cp.aput_state_cache(c1, state)
    return keep, secret


class TestS1DeletedMessageCache:
    async def test_scope_subset_delete_never_served_from_cache(self, make_cp):
        cp = await make_cp()
        await cp.asetup()
        thread = f"t-{uuid.uuid4()}"
        _keep, secret = await _seed(cp, thread)
        # Same user, same org, now in project P2 (the scope does not pick project_id).
        await cp.adelete_message(_cfg(thread, project="P2"), secret.message_id)
        assert _texts(await cp.aget_state(_cfg(thread, project="P2"))) == ["keep me"]
        assert _texts(await cp.aget_state_cache(_cfg(thread, project="P1"))) == ["keep me"]
        assert _texts(await cp.aget_state_cache(_cfg(thread, project="P2"))) == ["keep me"]

    async def test_org_admin_delete_never_served_from_cache(self, make_cp):
        cp = await make_cp()
        await cp.asetup()
        thread = f"t-{uuid.uuid4()}"
        _keep, secret = await _seed(cp, thread)
        admin = _cfg(thread, user="root", project="P9", scope=ORG_ADMIN)
        # The admin reads the owner's cached state through the thread's shared entry.
        assert _texts(await cp.aget_state_cache(admin)) == ["keep me", "secret to delete"]
        await cp.adelete_message(
            _cfg(thread, user="root", project="P9", scope=ORG_ADMIN), secret.message_id
        )
        assert _texts(await cp.aget_state_cache(_cfg(thread))) == ["keep me"]
        assert _texts(await cp.aget_state_cache(admin)) == ["keep me"]

    async def test_other_org_never_gets_a_hit(self, make_cp):
        cp = await make_cp()
        await cp.asetup()
        thread = f"t-{uuid.uuid4()}"
        await _seed(cp, thread)
        outsider = _cfg(thread, user="root", org="B", scope=ORG_ADMIN)
        assert await cp.aget_state_cache(outsider) is None
        # Even with a forged identity of the thread in its config, the scoped read decides.
        forged = _cfg(thread, user="root", org="B", scope=ORG_ADMIN)
        forged["_thread_identity"] = {
            "thread_id": thread,
            "user_id": "alice",
            "tenant": {"org_id": "A", "project_id": "P1"},
        }
        assert await cp.aget_state_cache(forged) is None

    async def test_clear_then_rewrite_to_same_version_not_served_stale(self, make_cp):
        cp = await make_cp()
        await cp.asetup()
        thread = f"t-{uuid.uuid4()}"
        await _seed(cp, thread)  # version 1 cached under the shared entry
        p2 = _cfg(thread, project="P2")
        await cp.aclear_state(p2)
        fresh = AgentState()
        fresh.context = [Message.text_message("after clear", role="user")]
        await cp.aput_state(_cfg(thread, project="P2"), fresh)  # version 1 again
        assert _texts(await cp.aget_state_cache(_cfg(thread))) == ["after clear"]

    async def test_run_holding_old_state_cannot_write_message_back(self, make_cp):
        cp = await make_cp()
        await cp.asetup()
        thread = f"t-{uuid.uuid4()}"
        _keep, secret = await _seed(cp, thread)
        run = _cfg(thread)
        stale_state = await cp.aget_state(run)  # the run read version 1
        await cp.adelete_message(_cfg(thread, project="P2"), secret.message_id)
        with pytest.raises(StaleStateError):
            await cp.aput_state(run, stale_state)
        assert _texts(await cp.aget_state(_cfg(thread))) == ["keep me"]


class TestS1StopKeys:
    async def test_org_admin_stop_reaches_the_run(self, make_cp):
        cp = await make_cp()
        await cp.asetup()
        thread = f"t-{uuid.uuid4()}"
        await _seed(cp, thread)
        run = _cfg(thread, run_id="run-1")
        await cp.abegin_run(run)
        admin = _cfg(thread, user="root", project="P9", scope=ORG_ADMIN)
        assert await cp.aget_active_run(admin) == "run-1"
        await cp.arequest_stop(_cfg(thread, user="root", project="P9", scope=ORG_ADMIN))
        assert await cp.ais_stop_requested(_cfg(thread, run_id="run-1"))
        # Another org cannot see the run or stop it.
        outsider = _cfg(thread, user="root", org="B", scope=ORG_ADMIN)
        assert await cp.aget_active_run(outsider) is None
        await cp.aend_run(_cfg(thread, run_id="run-1"))
        assert await cp.aget_active_run(admin) is None


class TestS2S6Setup:
    async def test_second_setup_does_not_wait_behind_open_write(self, make_cp, admin_pool, schema):
        await (await make_cp()).asetup()
        holder = await admin_pool.acquire()
        try:
            tr = holder.transaction()
            await tr.start()
            await holder.execute(
                f'INSERT INTO "{schema}"."threads" (thread_id, thread_name, user_id, meta, '
                "org_id, project_id) VALUES ('t1', 'n', 'u', '{}', 'A', 'P')"
            )
            await asyncio.wait_for((await make_cp()).asetup(), 3)
            await tr.rollback()
        finally:
            await admin_pool.release(holder)

    async def test_lock_timeout_raises_clear_error(self, make_cp, admin_pool, schema):
        await (await make_cp(tenant_columns=None)).asetup()
        holder = await admin_pool.acquire()
        try:
            tr = holder.transaction()
            await tr.start()
            await holder.execute(
                f'INSERT INTO "{schema}"."threads" (thread_id, thread_name, user_id, meta) '
                "VALUES ('t1', 'n', 'u', '{}')"
            )
            cp = await make_cp(ddl_lock_timeout=0.5)
            t0 = time.monotonic()
            with pytest.raises(SetupLockTimeoutError, match="PgCheckpointer"):
                await asyncio.wait_for(cp.asetup(), 10)
            assert time.monotonic() - t0 < 5
            await tr.rollback()
        finally:
            await admin_pool.release(holder)
        await (await make_cp()).asetup()

    async def test_column_index_on_live_table_built_concurrently(self, make_cp, admin_pool, schema):
        await (await make_cp(tenant_columns=None)).asetup()
        await admin_pool.execute(
            f'INSERT INTO "{schema}"."threads" (thread_id, thread_name, user_id, meta) '
            "SELECT 't' || g, 'n', 'u' || (g % 100), '{}' FROM generate_series(1, 2000) g"
        )
        # A failed concurrent build left an invalid index under the name setup wants.
        await admin_pool.execute(f'ALTER TABLE "{schema}"."threads" ADD COLUMN org_id TEXT')
        conn = await admin_pool.acquire()
        try:
            with pytest.raises(Exception, match="could not create unique index"):
                await conn.execute(
                    f'CREATE UNIQUE INDEX CONCURRENTLY "idx_threads_org_id" '
                    f'ON "{schema}"."threads" (thread_name)'
                )
        finally:
            await admin_pool.release(conn)
        cp = await make_cp(
            tenant_columns={"org_id": {"type": "text", "required": False}, "project_id": "text"}
        )
        await cp.asetup()
        rows = await admin_pool.fetch(
            "SELECT c.relname, i.indisvalid, pg_get_indexdef(i.indexrelid) AS def "
            "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indrelid = to_regclass($1)",
            f'"{schema}"."threads"',
        )
        indexes = {r["relname"]: (r["indisvalid"], r["def"]) for r in rows}
        assert indexes["idx_threads_org_id"][0] is True
        assert "UNIQUE" not in indexes["idx_threads_org_id"][1]
        assert indexes["idx_threads_project_id"][0] is True
