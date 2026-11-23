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

from tenxgraph.qa.testing.in_memory_store import InMemoryStore
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer


POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://test:test@localhost:5432/test_agentflow")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
TENANTS = {"org_id": "text", "project_id": "text"}

# (test function name, backend id) -> reason. Backend ids: memory, sqlite, pg for the
# checkpointers; inmem, pgvector, qdrant for the stores. "pg" and "pgvector" entries come from
# reading the code (no database was available when the suite was written): unverified.
KNOWN_GAPS: dict[tuple[str, str], str] = {
    # ---- verified by running memory and sqlite
    ("test_list_threads_search_ignores_other_fields", "memory"): "G10: M matches any field value of the thread, P only thread_name; maintainer decision",
    ("test_list_threads_search_ignores_other_fields", "sqlite"): "G10: S matches any field value of the thread, P only thread_name; maintainer decision",
    ("test_put_thread_rejects_mismatched_thread_id", "memory"): "G13: M stores a mismatched thread_info.thread_id under the config key, S raises ValueError; maintainer decision",
    ("test_message_search_matches_text_only", "memory"): "G8: M searches the repr of the block list (field names match); maintainer decision",
    ("test_message_search_matches_text_only", "sqlite"): "G8: S searches the repr of the block list (field names match); maintainer decision",
    ("test_put_messages_metadata_argument", "memory"): "G9: M stores aput_messages(metadata=) and never reads it back; maintainer decision",
    ("test_put_messages_metadata_argument", "sqlite"): "G9: S ignores aput_messages(metadata=); P merges it into the message; maintainer decision",
    ("test_int_message_id_found_by_str_and_int", "memory"): "G5: M compares message ids with ==, so int 5 and \"5\" differ; S compares as text",
    ("test_str_message_id_found_by_int", "memory"): "G5: M compares message ids with ==, so int 5 and \"5\" differ; S compares as text",
    ("test_int_message_id_delete_and_update", "memory"): "G5: M compares message ids with ==, so int 5 and \"5\" differ; S compares as text",
    ("test_ledger_first_result_wins", "memory"): "G1: M overwrites on a second aput_tool_result; S and P keep the first (P docstring says first wins)",
    ("test_thread_owner_with_state_but_no_record", "memory"): "G15: M returns None for a thread that has state but no thread record although it holds the owner",
    ("test_thread_owner_with_state_but_no_record", "sqlite"): "G15: S returns None for a thread that has state but no thread record although it holds the owner",
    ("test_stop_request_and_active_run_cleared_with_thread", "memory"): "G17: aclean_thread leaves the stop request (expires by TTL); maintainer decision",
    ("test_stop_request_and_active_run_cleared_with_thread", "sqlite"): "G17: aclean_thread leaves the stop request (expires by TTL); maintainer decision",
    ("test_cache_clear_returns_a_plain_value", "memory"): "G3: M returns the stored (value, expires_at) tuple from aclear_cache_value; maintainer decision",
    # ---- Postgres: from reading the code, not run (no database when written)
    ("test_get_missing_message_raises_index_error", "pg"): "G4: P raises ValueError(\"Message not found\"), M and S raise IndexError; maintainer decision (unverified, no database)",
    ("test_message_roundtrip_keeps_full_model", "pg"): "G6: P rebuilds the message from columns and drops raw and parsed_content (unverified, no database)",
    ("test_message_search_matches_text_only", "pg"): "G8: P matches content ILIKE over the JSON text, so block keys match (unverified, no database)",
    ("test_put_thread_rejects_mismatched_thread_id", "pg"): "G13: P uses the config id silently (unverified, no database)",
    ("test_put_thread_does_not_mutate_input", "pg"): "G14: P adds run_id to the caller's thread_info.metadata in place (unverified, no database)",
    ("test_anonymous_use", "pg"): "G12: P raises ValueError without a user_id on thread and ledger calls; maintainer decision (unverified, no database)",
    ("test_cache_clear_returns_a_plain_value", "pg"): "G3: P returns the Redis delete count from aclear_cache_value (unverified, no database)",
    ("test_stop_request_and_active_run_cleared_with_thread", "pg"): "G17: aclean_thread leaves the stop request (expires by TTL) (unverified, no database)",
    # ---- stores: verified by running inmem and qdrant (local mode); pgvector not run
    ("test_replace_metadata_keeps_category", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82); aupdate ignores replace_metadata and stores no category",
    ("test_replace_metadata_keeps_category", "qdrant"): "S3: Q resets the category to \"general\" on aupdate(replace_metadata=True) without a category; V keeps it",
    ("test_metadata_roundtrip_without_payload_leak", "qdrant"): "S1: Q returns the whole payload (content, user_id, thread_id, timestamp) as result metadata; V returns caller metadata plus category, memory_type, memory_key",
    ("test_get_all_newest_first_and_limit", "inmem"): "S7: InMemoryStore returns insertion order; V returns newest first; maintainer decision",
    ("test_get_all_newest_first_and_limit", "qdrant"): "S7: Q returns scroll order (random, sorted by uuid4 point id); V returns newest first; maintainer decision",
    ("test_get_all_honours_filters", "inmem"): "S7: InMemoryStore ignores the filters keyword; V honours it",
    ("test_get_all_honours_filters", "qdrant"): "S7: Q ignores the filters keyword; V honours it",
    ("test_update_foreign_memory_is_refused", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82); a foreign aupdate succeeds",
    ("test_update_foreign_memory_under_reach_all_raises_value_error", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82); a foreign aupdate succeeds",
    ("test_update_foreign_memory_under_reach_all_raises_value_error", "qdrant"): "S4: Q raises PermissionError for a foreign update under reach all (aget is unscoped there); V raises ValueError; maintainer decision",
    ("test_get_when_backend_fails_raises", "qdrant"): "S6: Q swallows every exception in aget and returns None; V raises; maintainer decision",
    ("test_write_requires_user_id", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_blank_user_id_rejected", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_own_reach_isolates_users", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_reach_all_write_stays_own", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_reach_tenant_without_tenant_columns_raises", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_delete_foreign_memory_raises_and_keeps_row", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_forget_memory_deletes_only_the_owner", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_forget_memory_under_reach_all_still_names_an_owner", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_forget_memory_needs_an_owner", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_collections_are_separate", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
    ("test_blank_collection_rejected", "inmem"): "S-I (new): InMemoryStore ignores user_id, reach and collection (documented as satisfying the BaseStore interface, server/configure.md:82)",
}


# Order is random by uuid id, so the test passes by chance about 1 run in 6: not strict.
NON_STRICT = {("test_get_all_newest_first_and_limit", "qdrant")}


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
