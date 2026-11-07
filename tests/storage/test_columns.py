"""Unit tests for tenxgraph.storage.columns: spec parsing, values, WHERE and RLS text."""

import uuid
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tenxgraph.core.authz import Identity
from tenxgraph.storage.columns import (
    COLUMN_TYPES,
    ColumnSpec,
    ColumnValueError,
    MissingColumnValueError,
    apply_settings,
    build_isolation_where,
    column_ddl,
    index_name,
    isolation_conditions,
    parse_columns,
    quote_ident,
    rls_all_rows_settings,
    rls_policy_expression,
    rls_policy_name,
    rls_settings,
    user_value,
    write_conditions,
)


RESERVED = {"id", "content", "created_at"}


def _spec(type_="text", kind="tenant", required=True, name="org_id"):
    return ColumnSpec(name=name, type=type_, required=required, kind=kind)


class TestParse:
    def test_mapping_list_and_required_defaults(self):
        cols = parse_columns(
            {"org_id": "uuid", "team": {"type": "text", "required": False}},
            {"public_id": "text", "plan": {"type": "integer", "required": True}},
            reserved=RESERVED,
        )
        assert cols.tenant_names == ("org_id", "team")
        assert cols.extra_names == ("public_id", "plan")
        assert [s.required for s in cols.tenant] == [True, False]
        assert [s.required for s in cols.extra] == [False, True]
        assert {s.kind for s in cols.tenant} == {"tenant"}
        assert {s.kind for s in cols.extra} == {"extra"}
        assert cols.get("plan").type == "integer"
        assert cols.describe()["team"] == {"type": "text", "required": False, "kind": "tenant"}

    def test_list_of_names_means_text(self):
        cols = parse_columns(["org_id", "project_id"])
        assert [(s.name, s.type) for s in cols.tenant] == [
            ("org_id", "text"),
            ("project_id", "text"),
        ]

    def test_nothing_declared_is_falsy(self):
        assert not parse_columns(None, None)
        assert not parse_columns({}, [])

    @pytest.mark.parametrize(
        "name",
        ['org"id', "Org", "1org", "org-id", "org id", "", "o" * 64, "org;drop", "é", None, 5],
    )
    def test_rejects_bad_names(self, name):
        with pytest.raises(ValueError, match="Invalid tenant column name"):
            parse_columns({name: "text"})

    @pytest.mark.parametrize("name", ["id", "content", "created_at", "user_id"])
    def test_rejects_builtin_names(self, name):
        with pytest.raises(ValueError, match="built-in column"):
            parse_columns({name: "text"}, reserved=RESERVED)
        with pytest.raises(ValueError, match="built-in column"):
            parse_columns(None, {name: "text"}, reserved=RESERVED)

    def test_rejects_duplicates_across_kinds(self):
        with pytest.raises(ValueError, match="declared twice"):
            parse_columns({"org_id": "text"}, {"org_id": "text"})

    @pytest.mark.parametrize(
        "type_", ["varchar", "jsonb", "boolean", "timestamptz", "text[]", "text; drop"]
    )
    def test_tenant_rejects_non_id_types(self, type_):
        with pytest.raises(ValueError, match="Tenant columns must be scalar ids"):
            parse_columns({"org_id": {"type": type_}})

    @pytest.mark.parametrize("type_", [None, 1])
    def test_rejects_non_string_types(self, type_):
        with pytest.raises(ValueError, match="must be a string"):
            parse_columns({"org_id": {"type": type_}})
        with pytest.raises(ValueError, match="must be a string"):
            parse_columns(None, {"plan": {"type": type_}})

    def test_rejects_bad_spec_shapes(self):
        with pytest.raises(ValueError, match="unknown keys"):
            parse_columns({"org_id": {"type": "text", "index": False}})
        with pytest.raises(ValueError, match="'required' must be a bool"):
            parse_columns({"org_id": {"type": "text", "required": "no"}})
        with pytest.raises(ValueError, match="give a type name"):
            parse_columns({"org_id": 3})
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_columns("org_id")

    def test_all_types_supported(self):
        cols = parse_columns(None, {f"c_{t}": t for t in COLUMN_TYPES})
        assert [s.sql_type for s in cols.extra] == list(COLUMN_TYPES.values())


class TestConvert:
    def test_text(self):
        assert _spec().convert("A") == "A"
        assert _spec().convert(7) == "7"
        assert _spec().convert("  ") is None  # empty tenant value means "no value"
        assert _spec(kind="extra").convert("") == ""
        for bad in (True, 1.5, b"x", ["a"], {"a": 1}):
            with pytest.raises(ColumnValueError):
                _spec().convert(bad)

    @pytest.mark.parametrize("type_", ["integer", "bigint"])
    def test_integers(self, type_):
        spec = _spec(type_)
        assert spec.convert(5) == 5
        assert spec.convert(" -12 ") == -12
        for bad in (True, 1.0, "1.5", "abc", "1; drop"):
            with pytest.raises(ColumnValueError):
                spec.convert(bad)

    def test_integer_ranges(self):
        assert _spec("integer").convert(2**31 - 1) == 2**31 - 1
        with pytest.raises(ColumnValueError, match="range"):
            _spec("integer").convert(2**31)
        assert _spec("bigint").convert(2**63 - 1) == 2**63 - 1
        with pytest.raises(ColumnValueError, match="range"):
            _spec("bigint").convert(-(2**63) - 1)

    def test_boolean(self):
        assert _spec("boolean").convert(False) is False
        for bad in (0, 1, "true", None.__class__):
            with pytest.raises(ColumnValueError):
                _spec("boolean").convert(bad)

    def test_uuid(self):
        value = uuid.uuid4()
        assert _spec("uuid").convert(value) == value
        assert _spec("uuid").convert(str(value)) == value
        assert _spec("uuid").convert("") is None
        for bad in ("not-a-uuid", 5, "' OR 1=1 --"):
            with pytest.raises(ColumnValueError):
                _spec("uuid").convert(bad)

    def test_timestamptz(self):
        aware = datetime(2026, 1, 1, tzinfo=UTC)
        assert _spec("timestamptz").convert(aware) == aware
        parsed = _spec("timestamptz").convert("2026-01-01T00:00:00+02:00")
        assert parsed.utcoffset() == timedelta(hours=2)
        with pytest.raises(ColumnValueError, match="timezone"):
            _spec("timestamptz").convert(datetime(2026, 1, 1))
        for bad in ("yesterday", 5):
            with pytest.raises(ColumnValueError):
                _spec("timestamptz").convert(bad)

    def test_none_is_no_value(self):
        for t in COLUMN_TYPES:
            assert _spec(t).convert(None) is None

    def test_to_setting(self):
        tz = timezone(timedelta(hours=1))
        assert _spec().to_setting(None) == ""
        assert _spec("boolean").to_setting(True) == "true"
        assert _spec("integer").to_setting(3) == "3"
        assert _spec("timestamptz").to_setting(datetime(2026, 1, 1, tzinfo=tz)).endswith("+01:00")


def _u(**kw):
    return {"user_id": "u", **kw}


class TestResolve:
    def test_reads_only_config_user(self):
        cols = parse_columns({"org_id": "text"}, {"public_id": "text"})
        config = {
            "org_id": "FORGED",
            "public_id": "FORGED",
            "user": {"user_id": "u", "tenant": {"org_id": "A"}},
        }
        assert cols.resolve(config) == {"org_id": "A", "public_id": None}

    def test_identity_object_user(self):
        cols = parse_columns({"org_id": "text"}, {"plan": "text"})
        identity = Identity(user_id="u", tenant={"org_id": "A"}, claims={"plan": "pro"})
        assert cols.resolve({"user": identity}) == {"org_id": "A", "plan": "pro"}
        assert user_value(identity, "org_id") == "A"
        assert user_value(None, "org_id") is None

    def test_object_user(self):
        assert user_value(SimpleNamespace(org_id="A"), "org_id") == "A"

    def test_write_requires_required_columns(self):
        cols = parse_columns(
            {"org_id": "text", "team": {"type": "text", "required": False}},
            {"plan": {"type": "text", "required": True}, "note": "text"},
        )
        with pytest.raises(MissingColumnValueError, match="org_id"):
            cols.resolve({"user": _u(claims={"plan": "pro"})}, write=True)
        with pytest.raises(MissingColumnValueError, match="plan"):
            cols.resolve({"user": _u(tenant={"org_id": "A"})}, write=True)
        assert cols.resolve(
            {"user": _u(tenant={"org_id": "A"}, claims={"plan": "pro"})}, write=True
        ) == {
            "org_id": "A",
            "team": None,
            "plan": "pro",
            "note": None,
        }
        assert cols.resolve({})["org_id"] is None  # reads do not require

    def test_type_errors_raise(self):
        cols = parse_columns({"org_id": "uuid"})
        with pytest.raises(ColumnValueError):
            cols.resolve({"user": _u(tenant={"org_id": "nope"})})


class TestWhereAndRls:
    def test_isolation_conditions_raise_on_missing_picked_value(self):
        cols = parse_columns({"org_id": "text", "team": {"type": "text", "required": False}})
        specs = cols.scope_specs(None)
        with pytest.raises(MissingColumnValueError, match="team"):
            isolation_conditions(specs, {"org_id": "A", "team": None})
        assert isolation_conditions(cols.scope_specs(["org_id"]), {"org_id": "A"})[0][1] == "A"
        assert cols.scope_specs(["user_id"]) == ()
        assert write_conditions(specs, {"org_id": "A"})[1][1] is None

    def test_build_where_is_params_only(self):
        cols = parse_columns({"org_id": "text", "project_id": "text"})
        params = ["x"]
        sql = build_isolation_where(
            isolation_conditions(cols.tenant, {"org_id": "' OR 1=1 --", "project_id": "P"}), params
        )
        assert sql == '"org_id" = $2 AND "project_id" = $3'
        assert params == ["x", "' OR 1=1 --", "P"]
        assert build_isolation_where([], []) == ""
        assert build_isolation_where(
            isolation_conditions(cols.tenant[:1], {"org_id": "A"}), [], "t"
        ) == ('t."org_id" = $1')
        with pytest.raises(ValueError, match="alias"):
            build_isolation_where([], [], "t; drop")

    def test_rls_settings(self):
        cols = parse_columns({"org_id": "text", "project_id": "integer"})
        values = {"org_id": "A", "project_id": 7}
        assert rls_settings(cols.tenant, None, values) == [
            ("tenxgraph.scope", "+,org_id,project_id"),
            ("tenxgraph.org_id", "A"),
            ("tenxgraph.project_id", "7"),
        ]
        assert rls_settings(cols.tenant, cols.tenant[1:], values) == [
            ("tenxgraph.scope", "+,project_id"),
            ("tenxgraph.org_id", ""),
            ("tenxgraph.project_id", "7"),
        ]
        assert rls_all_rows_settings(cols.tenant)[0] == ("tenxgraph.scope", "+")
        # A picked column with no value (optional, on a write) is left out of the scope.
        assert rls_settings(cols.tenant, None, {"org_id": "A"})[0] == (
            "tenxgraph.scope",
            "+,org_id",
        )

    def test_policy_expression_and_name(self):
        cols = parse_columns({"org_id": "uuid", "project_id": "text"})
        expr = rls_policy_expression(cols.tenant)
        assert expr.startswith("(COALESCE(current_setting('tenxgraph.scope', true), '') LIKE '+%')")
        assert "\"org_id\" = NULLIF(current_setting('tenxgraph.org_id', true), '')::UUID" in expr
        assert "',project_id,'" in expr
        name = rls_policy_name(cols.tenant)
        assert name.startswith("tenxgraph_isolation_") and len(name) <= 63
        assert name != rls_policy_name(parse_columns({"org_id": "text"}).tenant)
        assert name == rls_policy_name(
            parse_columns({"org_id": "uuid", "project_id": "text"}).tenant
        )

    def test_identifier_helpers(self):
        assert quote_ident("org_id") == '"org_id"'
        with pytest.raises(ValueError):
            quote_ident('a"b')
        assert index_name("idx_threads", "org_id") == "idx_threads_org_id"
        long_a, long_b = index_name("t" * 40, "a" * 63), index_name("t" * 40, "a" * 62 + "b")
        assert len(long_a) <= 63 and long_a != long_b

    def test_column_ddl(self):
        cols = parse_columns({"org_id": "uuid"}, {"plan": "text"})
        assert column_ddl('"s"."threads"', cols, "idx_threads") == [
            'ALTER TABLE "s"."threads" ADD COLUMN IF NOT EXISTS "org_id" UUID',
            'CREATE INDEX IF NOT EXISTS "idx_threads_org_id" ON "s"."threads" ("org_id")',
            'ALTER TABLE "s"."threads" ADD COLUMN IF NOT EXISTS "plan" TEXT',
            'CREATE INDEX IF NOT EXISTS "idx_threads_plan" ON "s"."threads" ("plan")',
        ]
        assert column_ddl("t", parse_columns(), "p") == []


@pytest.mark.asyncio
async def test_apply_settings_sends_one_parameterized_statement():
    calls = []

    class Conn:
        async def execute(self, sql, *args):
            calls.append((sql, args))

    await apply_settings(Conn(), [("tenxgraph.scope", "+"), ("tenxgraph.org_id", "x'; --")])
    assert calls == [
        (
            "SELECT set_config($1, $2, true), set_config($3, $4, true)",
            ("tenxgraph.scope", "+", "tenxgraph.org_id", "x'; --"),
        )
    ]
    await apply_settings(Conn(), [])
    assert len(calls) == 1
