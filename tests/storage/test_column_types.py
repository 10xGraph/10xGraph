"""Column type rules: tenant columns are scalar ids, extra columns take any Postgres type.

No database here: type resolution is exercised with a fake connection. The real server
behaviour (to_regtype / to_regtypemod / format_type, asyncpg codecs) is covered in
tests/integration/test_column_types_real.py.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest

from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.columns import (
    EXTRA_TYPE_RE,
    TENANT_COLUMN_TYPES,
    ColumnSpec,
    ColumnValueError,
    column_ddl,
    column_value_errors,
    ensure_columns,
    parse_columns,
    resolve_column_type,
)
from tenxgraph.storage.sql_filter import (
    OTHER_TYPE_OPERATORS,
    FilterError,
    FilterSchema,
    compile_filters,
    validate_filters,
)
from tenxgraph.storage.store.pgvector_store import PgVectorStore
from tests.store.test_pgvector_store import FakeConn, FakePool, _Embedding


def _extra(type_):
    return parse_columns(None, {"c": type_}).get("c")


class TestTenantTypes:
    @pytest.mark.parametrize("type_", TENANT_COLUMN_TYPES)
    def test_scalar_ids_allowed_and_required_by_default(self, type_):
        spec = parse_columns({"org_id": type_}).get("org_id")
        assert spec.type == type_
        assert spec.required is True

    def test_case_and_whitespace_normalised(self):
        assert parse_columns({"org_id": " UUID "}).get("org_id").type == "uuid"

    @pytest.mark.parametrize(
        "type_",
        ["boolean", "timestamptz", "jsonb", "text[]", "numeric", "date", "varchar", "int4"],
    )
    def test_everything_else_rejected(self, type_):
        with pytest.raises(ValueError, match="Tenant columns must be scalar ids"):
            parse_columns({"org_id": type_})
        with pytest.raises(ValueError, match="Tenant columns must be scalar ids"):
            parse_columns({"org_id": {"type": type_, "required": False}})


class TestExtraTypeSyntax:
    @pytest.mark.parametrize(
        "type_",
        [
            "text",
            "jsonb",
            "json",
            "numeric(10,2)",
            "numeric(10, 2)",
            "character varying(64)",
            "timestamp with time zone",
            "timestamp(3) with time zone",
            "double precision",
            "text[]",
            "integer[][]",
            "numeric(10,2)[]",
            "schema.my_enum",
            "date",
            "Bit Varying(8)",
            "interval day to second",
        ],
    )
    def test_accepts_type_names(self, type_):
        spec = _extra(type_)
        assert spec.kind == "extra"
        assert spec.type == " ".join(type_.split()).lower()

    @pytest.mark.parametrize(
        "type_",
        [
            "text; drop table x",
            "text--",
            "text -- x",
            "text/* x */",
            '"text"',
            "'text'",
            "text'",
            "numeric(10,2,3)",
            "numeric(a)",
            "numeric((10))",
            "text) ; select 1 --",
            "a(b)",
            "text[]x",
            "text[1;]",
            "1text",
            "",
            "   ",
            "my-type",
            "text\x00",
            "e" * 129,
        ],
    )
    def test_rejects_injection_and_junk(self, type_):
        with pytest.raises(ValueError, match="not a valid type name"):
            parse_columns(None, {"c": type_})

    def test_regex_is_anchored(self):
        assert not EXTRA_TYPE_RE.match("text\nselect")
        assert not EXTRA_TYPE_RE.match("text;")

    def test_builtin_extra_types_keep_sql_type(self):
        assert _extra("timestamptz").sql_type == "TIMESTAMPTZ"
        assert _extra("BOOLEAN").builtin

    def test_other_types_have_no_fixed_sql_type(self):
        with pytest.raises(ValueError, match="resolved on the server"):
            _ = _extra("jsonb").sql_type

    def test_column_ddl_refuses_unresolved_types(self):
        cols = parse_columns(["org_id"], {"settings": "jsonb", "n": "integer"})
        with pytest.raises(ValueError, match="resolved on the server"):
            column_ddl('"public"."t"', cols, "idx")
        ddl = column_ddl('"public"."t"', cols.builtin_only(), "idx")
        assert len(ddl) == 4  # org_id and n: one column and one index each
        assert all("settings" not in sql for sql in ddl)


class TestValues:
    @pytest.mark.parametrize("type_", ["jsonb", "json", "JSONB", "pg_catalog.jsonb"])
    @pytest.mark.parametrize(
        "value", [{"a": [1, 2]}, [1, "x"], "plain", 7, 1.5, True, False, {}, []]
    )
    def test_json_serialised(self, type_, value):
        out = _extra(type_).convert(value)
        assert isinstance(out, str)
        assert json.loads(out) == value

    def test_json_none_is_null(self):
        assert _extra("jsonb").convert(None) is None

    @pytest.mark.parametrize("bad", [b"x", {1, 2}, object()])
    def test_json_wrong_type(self, bad):
        with pytest.raises(ColumnValueError, match="is jsonb") as info:
            _extra("jsonb").convert(bad)
        assert "config" in str(info.value)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, {"a": math.nan}, {"a": object()}])
    def test_json_unserialisable(self, bad):
        with pytest.raises(ColumnValueError, match="cannot be serialised as JSON") as info:
            _extra("jsonb").convert(bad)
        assert "nan" not in str(info.value).lower()

    def test_other_types_pass_through(self):
        marker = object()
        for type_ in ("text[]", "date", "numeric(10,2)", "my_enum"):
            assert _extra(type_).convert(marker) is marker

    def test_placeholder(self):
        assert _extra("jsonb").placeholder("$5") == "$5::text::jsonb"
        assert _extra("pg_catalog.json").placeholder("$2") == "$2::text::json"
        assert _extra("text[]").placeholder("$5") == "$5"
        assert _extra("text").placeholder("$5") == "$5"
        with pytest.raises(ValueError):
            _extra("jsonb").placeholder("$5; drop")


class TestCodecErrors:
    def _spec(self):
        return ColumnSpec(name="due", type="date", required=False, kind="extra")

    def test_codec_error_becomes_column_value_error_without_value(self):
        hidden = "super-hidden-value"
        with pytest.raises(ColumnValueError) as info, column_value_errors({6: self._spec()}):
            raise ValueError(
                f"invalid input for query argument $6: '{hidden}' ('str' object has no "
                "attribute 'toordinal')"
            )
        message = str(info.value)
        assert "'due'" in message and "date" in message
        assert hidden not in message
        assert info.value.__cause__ is None
        assert info.value.__suppress_context__

    def test_other_parameter_or_error_propagates(self):
        params = {6: self._spec()}
        with pytest.raises(ValueError, match=r"argument \$2"), column_value_errors(params):
            raise ValueError("invalid input for query argument $2: 'x'")
        with pytest.raises(RuntimeError), column_value_errors({6: self._spec()}):
            raise RuntimeError("boom")


class _TypeConn:
    """Fake connection: answers the server-version and type-resolution queries."""

    def __init__(self, existing=(), types=None, version=170011, parse_error=False):
        self.existing = [{"attname": n, "type": t} for n, t in existing]
        self.types = types or {}
        self.version = version
        self.parse_error = parse_error
        self.calls = []

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        if "pg_index" in sql:  # index_catalog: no indexes yet
            return []
        return self.existing

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        if "server_version_num" in sql:
            return self.version
        if self.parse_error:
            err = Exception("syntax error at or near")
            err.sqlstate = "42601"
            raise err
        return self.types.get(args[0])

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))


@pytest.mark.asyncio
class TestResolution:
    async def test_uses_canonical_name_in_ddl(self):
        cols = parse_columns(None, {"price": "NUMERIC(10, 2)", "n": "integer"})
        conn = _TypeConn(types={"numeric(10, 2)": "numeric(10,2)"})
        await ensure_columns(conn, '"public"."t"', cols, "idx")
        resolve = [c for c in conn.calls if c[0] == "fetchval" and "to_regtype" in c[1]]
        query = "SELECT format_type(to_regtype($1), to_regtypemod($1))"
        assert resolve == [("fetchval", query, ("numeric(10, 2)",))]
        alters = [c[1] for c in conn.calls if c[1].startswith("ALTER")]
        assert alters == [
            'ALTER TABLE "public"."t" ADD COLUMN IF NOT EXISTS "price" numeric(10,2)',
            'ALTER TABLE "public"."t" ADD COLUMN IF NOT EXISTS "n" INTEGER',
        ]

    async def test_unknown_type_fails_setup(self):
        cols = parse_columns(None, {"mood": "no_such_type"})
        conn = _TypeConn()
        with pytest.raises(ValueError, match="does not exist in the database"):
            await ensure_columns(conn, '"public"."t"', cols, "idx")
        assert not [c for c in conn.calls if c[0] == "execute"]

    async def test_parse_error_fails_setup(self):
        conn = _TypeConn(parse_error=True)
        with pytest.raises(ValueError, match="cannot parse type"):
            await resolve_column_type(conn, _extra("text drop table x"))

    async def test_existing_column_compared_by_canonical_name(self):
        cols = parse_columns(None, {"code": "varchar(64)"})
        conn = _TypeConn(
            existing=[("code", "character varying(64)")],
            types={"varchar(64)": "character varying(64)"},
        )
        await ensure_columns(conn, '"public"."t"', cols, "idx")
        assert not [c for c in conn.calls if c[1].startswith("ALTER")]

    async def test_existing_column_with_other_type_raises(self):
        cols = parse_columns(None, {"code": "varchar(64)"})
        conn = _TypeConn(
            existing=[("code", "text")], types={"varchar(64)": "character varying(64)"}
        )
        with pytest.raises(ValueError, match=r"already exists .* as text"):
            await ensure_columns(conn, '"public"."t"', cols, "idx")

    async def test_old_server_refuses_modifiers(self):
        conn = _TypeConn(version=160004)
        with pytest.raises(ValueError, match="PostgreSQL 17"):
            await resolve_column_type(conn, _extra("numeric(10,2)"))

    async def test_old_server_without_modifier(self):
        conn = _TypeConn(version=160004, types={"jsonb": "jsonb"})
        assert await resolve_column_type(conn, _extra("jsonb")) == "jsonb"
        assert conn.calls[-1][1] == "SELECT format_type(to_regtype($1), NULL)"

    async def test_unexpected_canonical_name_refused(self):
        conn = _TypeConn(types={"x": "text; drop table y"})
        with pytest.raises(ValueError, match="unexpected type name"):
            await resolve_column_type(conn, _extra("x"))


class TestFilters:
    SCHEMA = FilterSchema.for_threads(
        parse_columns(
            ["org_id"],
            {"settings": "jsonb", "tags": "text[]", "price": "numeric(10,2)", "n": "integer"},
        )
    )

    def test_is_null_allowed(self):
        params: list = []
        sql = compile_filters(
            validate_filters(
                {"settings": {"is_null": True}, "tags": {"is_null": False}}, self.SCHEMA
            ),
            params,
        )
        assert sql == '"settings" IS NULL AND "tags" IS NOT NULL'
        assert params == []

    @pytest.mark.parametrize("op", ["eq", "ne", "in", "nin", "gt", "lte", "like", "ilike"])
    @pytest.mark.parametrize("column", ["settings", "tags", "price"])
    def test_other_operators_rejected(self, column, op):
        value = ["a"] if op in ("in", "nin") else "a"
        with pytest.raises(FilterError, match="supports only is_null"):
            validate_filters({column: {op: value}}, self.SCHEMA)

    def test_plain_equality_rejected(self):
        with pytest.raises(FilterError, match="supports only is_null"):
            validate_filters({"tags": ["a"]}, self.SCHEMA)

    def test_known_types_keep_the_table(self):
        assert validate_filters({"n": {"gt": 3}}, self.SCHEMA)[0].value == 3

    def test_describe(self):
        info = self.SCHEMA.describe()
        assert info["tags"] == {"type": "text[]", "operators": list(OTHER_TYPE_OPERATORS)}
        assert info["n"]["operators"][0] == "eq"


@pytest.mark.asyncio
class TestBackendSql:
    async def test_checkpointer_insert_casts_json(self):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://x/y",
            redis=MagicMock(),
            tenant_columns=["org_id"],
            extra_columns={"settings": "jsonb", "tags": "text[]"},
        )
        cfg = {
            "user_id": "u1",
            "user": {
                "user_id": "u1",
                "tenant": {"org_id": "A"},
                "claims": {"settings": {"a": 1}, "tags": ["x"]},
            },
        }
        isolation = cp._isolation(cfg, write=True)
        sql, params = cp._thread_insert_sql("t1", "n", "u1", "{}", isolation, "")
        assert "VALUES ($1, $2, $3, $4, $5, $6::text::jsonb, $7)" in sql
        assert params[4:] == ["A", '{"a": 1}', ["x"]]
        # v6 migration DDL carries only built-in types; the rest is resolved at setup.
        v6 = cp._build_migrations()[6]
        assert all("settings" not in s and "tags" not in s for s in v6)
        assert any('"org_id"' in s for s in v6)

    async def test_store_write_casts_json_and_maps_codec_errors(self):
        conn = FakeConn()
        store = PgVectorStore(
            _Embedding(),
            pool=FakePool(conn),
            extra_columns={"settings": "jsonb", "due": "date"},
        )
        store._setup_done = True
        store._index_metric = store.distance_metric
        conn.fetchrow_results = [{"id": "m1"}]
        cfg = {
            "user_id": "u1",
            "user": {"user_id": "u1", "claims": {"settings": ["a"], "due": "2026-01-01"}},
        }
        await store.astore(cfg, "hello", memory_id="m1")
        sql = conn.sql("fetchrow")[0]
        n = len(conn.calls[0][2])
        assert f"${n - 1}::text::jsonb, ${n})" in sql
        assert conn.calls[0][2][-2:] == ('["a"]', "2026-01-01")

        conn.fetchrow_results = [
            ValueError(f"invalid input for query argument ${n}: '2026-01-01' (no toordinal)")
        ]
        with pytest.raises(ColumnValueError, match="'due' is date") as info:
            await store.astore(cfg, "hello", memory_id="m2")
        assert "2026-01-01" not in str(info.value)
