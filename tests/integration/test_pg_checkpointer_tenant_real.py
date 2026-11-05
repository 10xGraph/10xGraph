"""PgCheckpointer tenant and extra columns against a REAL Postgres + Redis.

Covers the v5 -> v6 migration with data, every subset of scope columns, cross-tenant denial on
every method (including Redis cache hits), the same user_id in two tenants, concurrent thread
creation, values fixed at creation, and native RLS with a non-superuser role.

    pytest tests/integration/test_pg_checkpointer_tenant_real.py --integration

POSTGRES_DSN / REDIS_URL from the environment. Each test gets its own Postgres schema (dropped
afterwards) and its own Redis key prefix.
"""

# SQL here is built only from identifiers the fixtures generate (uuid hex), never input.
# ruff: noqa: S608

import asyncio
import itertools
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest
import pytest_asyncio

from tenxgraph.core.authz import SCOPE_NONE, build_authz
from tenxgraph.core.exceptions import StorageError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import CURRENT_SCHEMA_VERSION
from tenxgraph.storage.columns import MissingColumnValueError
from tenxgraph.utils import ThreadInfo


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
TENANTS = {"org_id": "text", "project_id": "text"}
SCOPE_COLUMNS = ("user_id", "org_id", "project_id")


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
    name = f"tg_{uuid.uuid4().hex[:10]}"
    await admin_pool.execute(f'CREATE SCHEMA "{name}"')
    yield name
    await admin_pool.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')


@pytest_asyncio.fixture
async def make_cp(schema):
    created = []
    prefix = f"tgtest:{uuid.uuid4().hex[:8]}:"

    async def make(dsn=POSTGRES_DSN, **kw):
        kw.setdefault("tenant_columns", TENANTS)
        cp = PgCheckpointer(
            postgres_dsn=dsn, redis_url=REDIS_URL, schema=schema, redis_key_prefix=prefix, **kw
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


def _cfg(thread, user="alice", org="A", project="P1", scope=None, **extra):
    identity = {"user_id": user, "org_id": org, "project_id": project, **extra}
    if scope is not None:
        identity["authz"] = build_authz(user, scope=scope)
    return {"thread_id": thread, "user_id": user, "user": identity}


def _state(text):
    state = AgentState()
    state.context = [Message.text_message(text, role="user")]
    return state


def _tid():
    return f"t-{uuid.uuid4()}"


class TestMigration:
    async def test_v5_to_v6_with_existing_data(self, make_cp, admin_pool, schema):
        old = await make_cp(tenant_columns=None)
        thread = _tid()
        legacy = {"thread_id": thread, "user_id": "alice"}
        await old.aput_state(legacy, _state("legacy"))
        await old.aput_messages(legacy, [Message.text_message("hi", role="user")])
        # Pretend the schema was left at v5 by the previous release.
        await admin_pool.execute(f'DELETE FROM "{schema}"."schema_version"')
        await admin_pool.execute(f'INSERT INTO "{schema}"."schema_version" (version) VALUES (5)')
        cols = await admin_pool.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = 'threads'",
            schema,
        )
        assert not {"org_id", "project_id"} & {c["column_name"] for c in cols}

        cp = await make_cp(extra_columns={"public_id": "text"})
        await cp.asetup()
        assert await cp.aget_schema_version() == CURRENT_SCHEMA_VERSION == 6
        cols = await admin_pool.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = 'threads'",
            schema,
        )
        assert {"org_id", "project_id", "public_id"} <= {c["column_name"] for c in cols}
        indexes = {
            r["indexname"]
            for r in await admin_pool.fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = $1", schema
            )
        }
        assert {"idx_threads_org_id", "idx_threads_project_id", "idx_threads_public_id"} <= indexes

        # The old thread has NULL tenant values: invisible to a scope that picks them...
        assert await cp.aget_state(_cfg(thread)) is None
        assert await cp.alist_messages(_cfg(thread)) == []
        assert await cp.aget_thread_scope(thread) == {
            "user_id": "alice",
            "org_id": None,
            "project_id": None,
        }
        # ...and visible to one that does not.
        user_only = _cfg(thread, scope=["user_id"])
        assert (await cp.aget_state(user_only)).context[0].text() == "legacy"
        assert len(await cp.alist_messages(user_only)) == 1

    async def test_nothing_declared_keeps_the_schema(self, make_cp, admin_pool, schema):
        cp = await make_cp(tenant_columns=None)
        await cp.asetup()
        cols = await admin_pool.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = 'threads'",
            schema,
        )
        assert {c["column_name"] for c in cols} == {
            "thread_id",
            "thread_name",
            "user_id",
            "created_at",
            "updated_at",
            "meta",
        }


class TestTenantIsolation:
    async def test_cross_tenant_denied_on_every_method(self, make_cp):
        cp = await make_cp()
        thread = _tid()
        a = _cfg(thread, org="A")
        await cp.aput_state(a, _state("org A state"))
        msg = Message.text_message("org A message", role="user", message_id=str(uuid.uuid4()))
        await cp.aput_messages(a, [msg])
        await cp.aput_tool_result(a, "call-1", {"ok": True})
        await cp.aput_thread(a, ThreadInfo(thread_id=thread, thread_name="A thread"))
        assert (await cp.aget_state_cache(dict(a))).context[0].text() == "org A state"

        b = lambda: _cfg(thread, org="B")  # same user_id, other tenant  # noqa: E731
        assert await cp.aget_state(b()) is None
        assert await cp.aget_state_cache(b()) is None  # a cache hit must not bypass isolation
        assert cp._get_thread_key(thread, "alice", a) != cp._get_thread_key(thread, "alice", b())
        await cp.aclear_state(b())
        with pytest.raises(StorageError) as err:
            await cp.aput_state(b(), _state("hijack"))
        assert err.value.error_code == "STORAGE_001"  # wraps STORAGE_FORBIDDEN_001
        with pytest.raises(StorageError):
            await cp.aput_checkpoint(b(), _state("hijack"), [])
        with pytest.raises(StorageError, match="different tenant"):
            await cp.aput_messages(b(), [Message.text_message("x", role="user")])
        with pytest.raises(ValueError, match="not found"):
            await cp.aget_message(b(), msg.message_id)
        assert await cp.alist_messages(b()) == []
        await cp.adelete_message(b(), msg.message_id)
        assert await cp.aupdate_message_metadata(b(), msg.message_id, {"x": 1}) is None
        with pytest.raises(StorageError):
            await cp.aput_thread(b(), ThreadInfo(thread_id=thread, thread_name="hijack"))
        assert await cp.aget_thread(b()) is None
        assert thread not in {t.thread_id for t in await cp.alist_threads(b())}
        await cp.aclean_thread(b())
        assert await cp.aget_tool_result(b(), "call-1") is None
        with pytest.raises(StorageError, match="different tenant"):
            await cp.aput_tool_result(b(), "call-2", {"ok": False})

        # Tenant A still has everything, unchanged.
        fresh = _cfg(thread, org="A")
        assert (await cp.aget_state(fresh)).context[0].text() == "org A state"
        assert [m.message_id for m in await cp.alist_messages(fresh)] == [msg.message_id]
        assert (await cp.aget_message(fresh, msg.message_id)).text() == "org A message"
        assert await cp.aget_tool_result(fresh, "call-1") == {"ok": True}
        assert (await cp.aget_thread(fresh)).thread_name == "A thread"
        assert thread in {t.thread_id for t in await cp.alist_threads(fresh)}

    async def test_same_user_id_in_two_tenants(self, make_cp):
        cp = await make_cp()
        ta, tb = _tid(), _tid()
        await cp.aput_state(_cfg(ta, org="A"), _state("A"))
        await cp.aput_state(_cfg(tb, org="B"), _state("B"))
        assert {t.thread_id for t in await cp.alist_threads(_cfg(None, org="A"))} == {ta}
        assert {t.thread_id for t in await cp.alist_threads(_cfg(None, org="B"))} == {tb}
        assert (await cp.aget_state_cache(_cfg(ta, org="A"))).context[0].text() == "A"
        assert (await cp.aget_state_cache(_cfg(tb, org="B"))).context[0].text() == "B"

    async def test_every_subset_of_scope_columns(self, make_cp):
        cp = await make_cp()
        seeds = [
            ("alice", "A", "P1"),
            ("bob", "A", "P1"),
            ("alice", "A", "P2"),
            ("alice", "B", "P1"),
            ("carol", "B", "P3"),
        ]
        threads = {}
        for user, org, project in seeds:
            thread = _tid()
            threads[(user, org, project)] = thread
            await cp.aput_state(_cfg(thread, user, org, project), _state(thread))
        caller = {"user_id": "alice", "org_id": "A", "project_id": "P1"}
        for size in range(len(SCOPE_COLUMNS) + 1):
            for scope in itertools.combinations(SCOPE_COLUMNS, size):
                want = {
                    threads[seed]
                    for seed in seeds
                    if all(
                        dict(zip(SCOPE_COLUMNS, seed, strict=True))[c] == caller[c] for c in scope
                    )
                }
                listed = await cp.alist_threads(_cfg(None, scope=list(scope)), limit=100)
                assert {t.thread_id for t in listed} & set(threads.values()) == want, scope
                for seed, thread in threads.items():
                    got = await cp.aget_state(_cfg(thread, scope=list(scope)))
                    assert (got is not None) == (thread in want), (scope, seed)

    async def test_none_scope_sees_every_tenant(self, make_cp):
        cp = await make_cp()
        thread = _tid()
        await cp.aput_state(_cfg(thread, org="B", user="bob"), _state("B"))
        admin = _cfg(thread, user="admin", org="A", scope=SCOPE_NONE)
        assert (await cp.aget_state(admin)).context[0].text() == "B"

    async def test_concurrent_thread_creation(self, make_cp):
        cp = await make_cp()
        for _ in range(10):
            thread = _tid()
            results = await asyncio.gather(
                cp.aput_state(_cfg(thread, org="A"), _state("A")),
                cp.aput_state(_cfg(thread, org="B"), _state("B")),
                return_exceptions=True,
            )
            errors = [r for r in results if isinstance(r, BaseException)]
            assert len(errors) == 1 and isinstance(errors[0], StorageError)
            owner = await cp.aget_thread_scope(thread)
            winner = "A" if not isinstance(results[0], BaseException) else "B"
            assert owner["org_id"] == winner
            assert (await cp.aget_state(_cfg(thread, org=winner))).context[0].text() == winner

    async def test_values_set_at_creation_never_change(self, make_cp, admin_pool, schema):
        cp = await make_cp(extra_columns={"public_id": "text"})
        thread = _tid()
        await cp.aput_state(_cfg(thread, public_id="pub-1"), _state("x"))
        await cp.aput_thread(
            _cfg(thread, public_id="pub-2"), ThreadInfo(thread_id=thread, thread_name="renamed")
        )
        await cp.aput_state(_cfg(thread, public_id="pub-3"), _state("y"))
        row = await admin_pool.fetchrow(
            f'SELECT thread_name, org_id, public_id FROM "{schema}"."threads" WHERE thread_id = $1',
            thread,
        )
        assert dict(row) == {"thread_name": "renamed", "org_id": "A", "public_id": "pub-1"}

    async def test_missing_tenant_value_raises(self, make_cp):
        cp = await make_cp()
        cfg = {"thread_id": _tid(), "user_id": "alice", "user": {"user_id": "alice"}}
        with pytest.raises(MissingColumnValueError):
            await cp.aput_state(cfg, _state("x"))
        with pytest.raises(MissingColumnValueError):
            await cp.aget_state(cfg)
        with pytest.raises(MissingColumnValueError):
            await cp.aget_state_cache(cfg)

    async def test_unknown_scope_column_raises(self, make_cp):
        cp = await make_cp()
        with pytest.raises(ValueError, match="does not declare"):
            await cp.aget_state(_cfg(_tid(), scope=["user_id", "region"]))


# ---------------------------------------------------------------------------
# Native RLS with a non-superuser role
# ---------------------------------------------------------------------------


def _dsn_for(role: str, password: str) -> str:
    parts = urlsplit(POSTGRES_DSN)
    host = parts.hostname or "localhost"
    netloc = f"{role}:{password}@{host}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


@pytest_asyncio.fixture
async def rls_dsn(admin_pool, schema):
    role = f"tg_rls_{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex
    await admin_pool.execute(
        f"CREATE ROLE {role} LOGIN PASSWORD '{password}' NOSUPERUSER NOBYPASSRLS"
    )
    await admin_pool.execute(f'ALTER SCHEMA "{schema}" OWNER TO {role}')
    # The message_role enum lives in the first schema on the search path (public).
    await admin_pool.execute(f"GRANT CREATE, USAGE ON SCHEMA public TO {role}")
    yield _dsn_for(role, password)
    await admin_pool.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    await admin_pool.execute(f"DROP OWNED BY {role}")
    await admin_pool.execute(f"DROP ROLE {role}")


class TestRowLevelSecurity:
    async def test_policies_filter_rows_without_where(self, make_cp, admin_pool, schema, rls_dsn):
        asyncpg = pytest.importorskip("asyncpg")
        cp = await make_cp(dsn=rls_dsn, row_level_security=True)
        ta, tb = _tid(), _tid()
        await cp.aput_state(_cfg(ta, org="A"), _state("A"))
        await cp.aput_messages(_cfg(ta, org="A"), [Message.text_message("hi A", role="user")])
        await cp.aput_state(_cfg(tb, org="B"), _state("B"))
        assert (await cp.aget_state(_cfg(ta, org="A"))).context[0].text() == "A"
        assert await cp.aget_state(_cfg(ta, org="B")) is None
        assert await cp.aget_thread_scope(tb) == {
            "user_id": "alice",
            "org_id": "B",
            "project_id": "P1",
        }
        with pytest.raises(StorageError):
            await cp.aput_state(_cfg(ta, org="B"), _state("hijack"))

        for table in ("threads", "states", "messages", "tool_executions"):
            flags = await admin_pool.fetchrow(
                "SELECT c.relrowsecurity, c.relforcerowsecurity FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relname = $2",
                schema,
                table,
            )
            assert flags["relrowsecurity"] and flags["relforcerowsecurity"], table

        conn = await asyncpg.connect(rls_dsn)
        try:
            for table in ("threads", "states", "messages"):
                count = await conn.fetchval(f'SELECT count(*) FROM "{schema}"."{table}"')
                assert count == 0, table  # no settings: the policy shows nothing
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('tenxgraph.scope', '+,org_id', true), "
                    "set_config('tenxgraph.org_id', 'A', true)"
                )
                ids = {
                    r["thread_id"]
                    for r in await conn.fetch(f'SELECT thread_id FROM "{schema}"."threads"')
                }
                assert ids == {ta}
                states = {
                    r["thread_id"]
                    for r in await conn.fetch(f'SELECT thread_id FROM "{schema}"."states"')
                }
                assert states == {ta}  # child rows follow their thread's visibility
                with pytest.raises(Exception, match="row-level security"):
                    await conn.execute(
                        f'INSERT INTO "{schema}"."threads" (thread_id, user_id, org_id) '
                        "VALUES ('forged', 'alice', 'B')"
                    )
        finally:
            await conn.close()
        # The superuser test role bypasses RLS.
        assert await admin_pool.fetchval(f'SELECT count(*) FROM "{schema}"."threads"') == 2
