"""Sprint 7 review fixes in ``storage/columns.py`` (no database).

A5: column names that collide with authorization keys of ``config["user"]`` are refused.
S2/S6: catalog-first index DDL, invalid-index rebuild, ``CONCURRENTLY``, the setup session
(session-level advisory lock + ``lock_timeout``). S4: index names follow the table-name rule.
The Postgres behaviour itself is covered in tests/integration/test_store_review_fixes_real.py.
"""

import pytest

from tenxgraph.storage.columns import (
    RESERVED_USER_KEYS,
    SetupLockTimeoutError,
    column_ddl,
    ddl_lock_timeout_setting,
    ddl_session,
    ensure_column_indexes,
    ensure_index,
    is_lock_timeout,
    parse_columns,
    quote_index_name,
)


class _Conn:
    def __init__(self, catalog_rows=(), fail_on=None):
        self.catalog_rows = list(catalog_rows)
        self.fail_on = fail_on
        self.executed = []

    async def fetch(self, sql, *args):
        return self.catalog_rows

    async def execute(self, sql, *args):
        self.executed.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise self.fail_on_error


class _LockTimeout(Exception):
    sqlstate = "55P03"


class TestReservedUserKeys:
    @pytest.mark.parametrize("name", ["authz", "scope", "scopes", "role", "roles"])
    @pytest.mark.parametrize("kind", ["tenant", "extra"])
    def test_refused_for_both_kinds(self, name, kind):
        declared = {name: "text"}
        args = (declared, None) if kind == "tenant" else (None, declared)
        with pytest.raises(ValueError, match="carries authorization data"):
            parse_columns(*args)

    def test_set_matches_the_api_keys(self):
        assert {"authz", "scope", "scopes", "role", "roles"} == RESERVED_USER_KEYS

    def test_other_names_still_allowed(self):
        cols = parse_columns(["org_id"], {"role_label": "text", "scoped": "text"})
        assert cols.tenant_names == ("org_id",)


class TestIndexNames:
    def test_uppercase_index_name_is_quoted(self):
        assert quote_index_name("Mem_ab_c_org_id") == '"Mem_ab_c_org_id"'

    @pytest.mark.parametrize("name", ['a"b', "1a", "", None, "a b", "x" * 64])
    def test_bad_index_names(self, name):
        with pytest.raises(ValueError, match="Invalid index name"):
            quote_index_name(name)

    def test_column_ddl_with_uppercase_prefix(self):
        cols = parse_columns(["org_id"])
        assert column_ddl('"public"."Mem"', cols, "Mem_c")[1] == (
            'CREATE INDEX IF NOT EXISTS "Mem_c_org_id" ON "public"."Mem" ("org_id")'
        )
        assert column_ddl('"public"."Mem"', cols, "Mem_c", indexes=False) == [
            'ALTER TABLE "public"."Mem" ADD COLUMN IF NOT EXISTS "org_id" TEXT'
        ]


@pytest.mark.asyncio
class TestEnsureIndex:
    async def test_valid_index_runs_no_ddl(self):
        conn = _Conn()
        built = await ensure_index(
            conn,
            {"t_i": (True, '"t_i"')},
            name="t_i",
            table_sql='"public"."t"',
            definition="(a)",
            concurrently=True,
        )
        assert built is False and conn.executed == []

    async def test_invalid_index_is_dropped_and_rebuilt(self):
        conn = _Conn()
        await ensure_index(
            conn,
            {"t_i": (False, 'public."t_i"')},
            name="t_i",
            table_sql='"public"."t"',
            definition="(a)",
            concurrently=True,
            unique=True,
        )
        assert conn.executed == [
            'DROP INDEX CONCURRENTLY IF EXISTS public."t_i"',
            'CREATE UNIQUE INDEX CONCURRENTLY "t_i" ON "public"."t" (a)',
        ]

    async def test_missing_index_plain_build(self):
        conn = _Conn()
        await ensure_index(
            conn, {}, name="t_i", table_sql='"public"."t"', definition="(a)", concurrently=False
        )
        assert conn.executed == ['CREATE INDEX "t_i" ON "public"."t" (a)']

    async def test_unexpected_catalog_name_refused(self):
        with pytest.raises(ValueError, match="Unexpected index name"):
            await ensure_index(
                _Conn(),
                {"t_i": (False, 't_i"; DROP TABLE x; --')},
                name="t_i",
                table_sql='"public"."t"',
                definition="(a)",
                concurrently=True,
            )

    async def test_column_indexes_read_catalog_and_skip_present(self):
        conn = _Conn(
            catalog_rows=[
                {"relname": "Mem_c_org_id", "indisvalid": True, "qualified": '"Mem_c_org_id"'}
            ]
        )
        cols = parse_columns(["org_id", "project_id"])
        await ensure_column_indexes(conn, '"public"."Mem"', cols, "Mem_c", concurrently=True)
        assert conn.executed == [
            'CREATE INDEX CONCURRENTLY "Mem_c_project_id" ON "public"."Mem" ("project_id")'
        ]


class TestLockTimeoutHelpers:
    def test_setting(self):
        assert ddl_lock_timeout_setting(None) == "0"
        assert ddl_lock_timeout_setting(2.5) == "2500ms"
        assert ddl_lock_timeout_setting(0.0001) == "1ms"
        for bad in (0, -1, True, "5"):
            with pytest.raises(ValueError, match="ddl_lock_timeout"):
                ddl_lock_timeout_setting(bad)

    def test_is_lock_timeout_follows_causes(self):
        try:
            try:
                raise _LockTimeout("canceling statement due to lock timeout")
            except _LockTimeout as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            assert is_lock_timeout(outer)
        assert not is_lock_timeout(RuntimeError("other"))
        assert not is_lock_timeout(None)


@pytest.mark.asyncio
class TestDdlSession:
    async def test_session_lock_then_timeout_then_release(self):
        conn = _Conn()
        async with ddl_session(conn, lock_key=7, timeout=1.0, owner="X"):
            conn.executed.append("work")
        assert conn.executed == [
            "SELECT pg_advisory_lock($1)",
            "SELECT set_config('lock_timeout', $1, false)",
            "work",
            "RESET lock_timeout",
            "SELECT pg_advisory_unlock($1)",
        ]

    async def test_lock_timeout_becomes_setup_error_and_lock_is_released(self):
        conn = _Conn()
        with pytest.raises(SetupLockTimeoutError, match="within 0.5s") as info:
            async with ddl_session(conn, lock_key=7, timeout=0.5, owner="PgVectorStore"):
                raise _LockTimeout("canceling statement due to lock timeout")
        assert isinstance(info.value.__cause__, _LockTimeout)
        assert conn.executed[-1] == "SELECT pg_advisory_unlock($1)"

    async def test_other_errors_propagate_and_release_failure_is_swallowed(self):
        conn = _Conn(fail_on="RESET")
        conn.fail_on_error = ConnectionError("gone")
        with pytest.raises(KeyError):
            async with ddl_session(conn, lock_key=7, timeout=None, owner="X"):
                raise KeyError("boom")
