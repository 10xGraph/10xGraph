"""PgCheckpointer tool ledger clear, prune and the v7 index against a REAL Postgres + Redis.

    pytest tests/integration/test_pg_tool_ledger_cleanup_real.py --integration

POSTGRES_DSN / REDIS_URL from the environment; each test gets its own Postgres schema.
"""

# SQL here is built only from identifiers the fixtures generate (uuid hex), never input.
# ruff: noqa: S608, F811

import logging
from datetime import timedelta

import pytest

from tenxgraph.storage.checkpointer.pg_checkpointer import CURRENT_SCHEMA_VERSION
from tests.integration.test_pg_checkpointer_tenant_real import (  # noqa: F401 - fixtures
    _cfg,
    _tid,
    admin_pool,
    make_cp,
    rls_dsn,
    schema,
)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _index_exists(admin_pool, schema) -> bool:
    return bool(
        await admin_pool.fetchval(
            "SELECT 1 FROM pg_indexes WHERE schemaname = $1 AND tablename = 'tool_executions' "
            "AND indexname = 'idx_tool_executions_created_at'",
            schema,
        )
    )


async def test_fresh_schema_has_the_v7_index(make_cp, admin_pool, schema):
    cp = await make_cp(tenant_columns=None)
    await cp.asetup()
    assert await _index_exists(admin_pool, schema)
    assert await cp.aget_schema_version() == CURRENT_SCHEMA_VERSION == 7


async def test_v6_to_v7_builds_the_index_and_keeps_rows(make_cp, admin_pool, schema):
    old = await make_cp(tenant_columns=None)
    cfg = {"thread_id": _tid(), "user_id": "alice"}
    await old.aput_tool_result(cfg, "m:c1", {"v": 1})
    await old.arelease()
    # Pretend the previous release left the schema at v6, without the index.
    await admin_pool.execute(f'DROP INDEX "{schema}"."idx_tool_executions_created_at"')
    await admin_pool.execute(f'DELETE FROM "{schema}"."schema_version"')
    await admin_pool.execute(f'INSERT INTO "{schema}"."schema_version" (version) VALUES (6)')

    new = await make_cp(tenant_columns=None)
    await new.asetup()

    assert await _index_exists(admin_pool, schema)
    assert await new.aget_schema_version() == 7
    valid = await admin_pool.fetchval(
        "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = $1 AND c.relname = 'idx_tool_executions_created_at'",
        schema,
    )
    assert valid is True
    assert await new.aget_tool_result(cfg, "m:c1") == {"v": 1}


async def test_clear_is_scoped_to_the_callers_thread(make_cp):
    cp = await make_cp()
    a = _cfg(_tid(), org="A")
    b = _cfg(_tid(), org="A")
    await cp.aput_tool_result(a, "m:c1", {"v": 1})
    await cp.aput_tool_result(a, "m:c2", {"v": 2})
    await cp.aput_tool_result(b, "m:c1", {"v": 3})

    other_org = _cfg(a["thread_id"], org="B")
    assert await cp.aclear_tool_results(other_org) == 0
    other_user = _cfg(a["thread_id"], user="bob", org="A")
    assert await cp.aclear_tool_results(other_user) == 0

    assert await cp.aclear_tool_results(a) == 2
    assert await cp.aget_tool_result(a, "m:c1") is None
    assert await cp.aget_tool_result(b, "m:c1") == {"v": 3}


async def test_clear_by_origin_ids_matches_literally_and_keeps_other_messages(make_cp):
    cp = await make_cp()
    a = _cfg(_tid(), org="A")
    for key in ("m1:c1", "m1:c2", "m10:c1", "m2:c1", "a%:c1", "ab:c1", "a_:c1", "x:y:c1", "x:c1"):
        await cp.aput_tool_result(a, key, {"k": key})

    other_org = _cfg(a["thread_id"], org="B")
    assert await cp.aclear_tool_results(other_org, ["m1"]) == 0
    assert await cp.aclear_tool_results(a, []) == 0

    assert await cp.aclear_tool_results(a, ["m1", "a%", "a_", "x:y"]) == 5
    for gone in ("m1:c1", "m1:c2", "a%:c1", "a_:c1", "x:y:c1"):
        assert await cp.aget_tool_result(a, gone) is None
    for kept in ("m10:c1", "m2:c1", "ab:c1", "x:c1"):
        assert await cp.aget_tool_result(a, kept) == {"k": kept}


async def test_prune_deletes_old_rows_of_every_thread(make_cp, admin_pool, schema):
    cp = await make_cp(tenant_columns=None)
    old_a = {"thread_id": _tid(), "user_id": "alice"}
    old_b = {"thread_id": _tid(), "user_id": "bob"}
    await cp.aput_tool_result(old_a, "m:old", {"v": 1})
    await cp.aput_tool_result(old_b, "m:old", {"v": 2})
    await admin_pool.execute(
        f'UPDATE "{schema}"."tool_executions" SET created_at = now() - interval \'2 days\''
    )
    await cp.aput_tool_result(old_a, "m:new", {"v": 3})

    assert await cp.aprune_tool_results(timedelta(days=1)) == 2
    assert await cp.aget_tool_result(old_a, "m:old") is None
    assert await cp.aget_tool_result(old_a, "m:new") == {"v": 3}


async def test_clear_works_under_rls_and_prune_warns(make_cp, rls_dsn, caplog):
    cp = await make_cp(dsn=rls_dsn, row_level_security=True)
    a = _cfg(_tid(), org="A")
    await cp.aput_tool_result(a, "m:c1", {"v": 1})
    await cp.aput_tool_result(a, "m:c2", {"v": 2})

    with caplog.at_level(logging.WARNING, logger="tenxgraph.checkpointer.pg"):
        # The role cannot see past RLS: nothing is pruned, and the log says why.
        assert await cp.aprune_tool_results(timedelta(seconds=-1)) == 0
    assert any("BYPASSRLS" in r.getMessage() for r in caplog.records)

    assert await cp.aclear_tool_results(a, ["m"]) == 2
