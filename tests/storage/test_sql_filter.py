"""Unit and property tests for ``tenxgraph.storage.sql_filter`` (no database).

The SQL is checked against a strict grammar: every clause is an allow-listed, quoted column, a
fixed operator and a ``$n`` placeholder, joined with `` AND ``. Anything else in the text fails.
"""

import math
import re
import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tenxgraph.storage.columns import ColumnSet, parse_columns
from tenxgraph.storage.sql_filter import (
    MAX_CONDITIONS,
    MAX_LIST_ITEMS,
    MAX_STRING_LENGTH,
    MEMORY_FILTER_COLUMNS,
    OPERATORS,
    OPERATORS_BY_TYPE,
    THREAD_FILTER_COLUMNS,
    FilterCondition,
    FilterError,
    FilterSchema,
    build_filter_where,
    compile_filters,
    validate_filters,
)


TYPES = ("text", "bigint", "integer", "timestamptz", "boolean", "uuid")
EXTRA = {f"c_{t}": t for t in TYPES}
COLUMNS = parse_columns({"org_id": "uuid", "project_id": "text"}, EXTRA)
SCHEMA = FilterSchema.for_threads(COLUMNS)
ALLOWED = set(THREAD_FILTER_COLUMNS) | set(EXTRA)

UID = uuid.UUID("12345678-1234-5678-1234-567812345678")
TS = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

# One valid sample value per type (and the converted value expected in params).
SAMPLES = {
    "text": ("abc", "abc"),
    "bigint": (2**40, 2**40),
    "integer": (7, 7),
    "timestamptz": (TS, TS),
    "boolean": (True, True),
    "uuid": (UID, UID),
}

_OP_SQL = (
    r"(?:= \$\d+|IS DISTINCT FROM \$\d+|> \$\d+|>= \$\d+|< \$\d+|<= \$\d+|LIKE \$\d+"
    r"|ILIKE \$\d+|= ANY\(\$\d+\)|IS NULL|IS NOT NULL)"
)


def _grammar(allowed, alias=None):
    prefix = re.escape(f"{alias}.") if alias else ""
    names = "|".join(re.escape(n) for n in sorted(allowed))
    col = rf'{prefix}"(?:{names})"'
    simple = rf"{col} {_OP_SQL}"
    nin = rf"\({col} IS NULL OR {col} <> ALL\(\$\d+\)\)"
    clause = rf"(?:{simple}|{nin})"
    return re.compile(rf"^(?:{clause}(?: AND {clause})*)?$")


GRAMMAR = _grammar(ALLOWED)


def _sql(filters, schema=SCHEMA, params=None, alias=None):
    params = [] if params is None else params
    sql = build_filter_where(filters, schema, params, alias)
    return sql, params


def _assert_safe(sql, params, grammar=GRAMMAR):
    assert grammar.match(sql), sql
    numbers = [int(n) for n in re.findall(r"\$(\d+)", sql)]
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers))) if numbers else True
    if numbers:
        assert numbers[-1] == len(params)


# ---------------------------------------------------------------------------
# Every operator for every type
# ---------------------------------------------------------------------------


OP_CASES = [(t, op) for t in TYPES for op in OPERATORS_BY_TYPE[t]]


class TestOperators:
    @pytest.mark.parametrize(("type_name", "op"), OP_CASES)
    def test_every_operator(self, type_name, op):
        col = f"c_{type_name}"
        raw, converted = SAMPLES[type_name]
        if op == "is_null":
            value, expected = True, None
        elif op in ("in", "nin"):
            value, expected = [raw], [converted]
        elif op in ("like", "ilike"):
            value, expected = "a%_b", "a%_b"
        else:
            value, expected = raw, converted
        sql, params = _sql({col: {op: value}})
        _assert_safe(sql, params)
        quoted = f'"{col}"'
        expected_sql = {
            "eq": f"{quoted} = $1",
            "ne": f"{quoted} IS DISTINCT FROM $1",
            "gt": f"{quoted} > $1",
            "gte": f"{quoted} >= $1",
            "lt": f"{quoted} < $1",
            "lte": f"{quoted} <= $1",
            "like": f"{quoted} LIKE $1",
            "ilike": f"{quoted} ILIKE $1",
            "in": f"{quoted} = ANY($1)",
            "nin": f"({quoted} IS NULL OR {quoted} <> ALL($1))",
            "is_null": f"{quoted} IS NULL",
        }[op]
        assert sql == expected_sql
        assert params == ([] if op == "is_null" else [expected])

    def test_is_null_false(self):
        assert _sql({"c_text": {"is_null": False}}) == ('"c_text" IS NOT NULL', [])

    def test_plain_value_is_eq(self):
        assert _sql({"thread_name": "x"}) == ('"thread_name" = $1', ["x"])

    def test_several_ops_on_one_column_and_order(self):
        sql, params = _sql({"c_integer": {"gte": 1, "lt": 9}, "c_text": "a"})
        assert sql == '"c_integer" >= $1 AND "c_integer" < $2 AND "c_text" = $3'
        assert params == [1, 9, "a"]

    @pytest.mark.parametrize(
        ("type_name", "op"),
        [(t, op) for t in TYPES for op in sorted(OPERATORS) if op not in OPERATORS_BY_TYPE[t]],
    )
    def test_operator_not_allowed_for_type(self, type_name, op):
        with pytest.raises(FilterError, match="not supported for"):
            validate_filters({f"c_{type_name}": {op: "x"}}, SCHEMA)

    def test_operator_table(self):
        assert OPERATORS_BY_TYPE["text"] == ("eq", "ne", "in", "nin", "like", "ilike", "is_null")
        for t in ("bigint", "integer", "timestamptz"):
            assert set(OPERATORS_BY_TYPE[t]) == {
                "eq",
                "ne",
                "gt",
                "gte",
                "lt",
                "lte",
                "in",
                "nin",
                "is_null",
            }
        for t in ("boolean", "uuid"):
            assert set(OPERATORS_BY_TYPE[t]) == {"eq", "ne", "in", "nin", "is_null"}


# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------


class TestValues:
    @pytest.mark.parametrize(
        ("col", "value", "expected"),
        [
            ("c_text", 12, "12"),
            ("c_integer", "42", 42),
            ("c_bigint", "-9000000000", -9000000000),
            ("c_uuid", str(UID), UID),
            ("c_timestamptz", "2026-01-02T03:04:05+00:00", TS),
            (
                "c_timestamptz",
                datetime(2026, 1, 2, 5, 4, 5, tzinfo=timezone(timedelta(hours=2))),
                datetime(2026, 1, 2, 5, 4, 5, tzinfo=timezone(timedelta(hours=2))),
            ),
            ("c_boolean", False, False),
        ],
    )
    def test_conversions(self, col, value, expected):
        [cond] = validate_filters({col: value}, SCHEMA)
        assert cond.value == expected
        assert type(cond.value) is type(expected)

    @pytest.mark.parametrize(
        ("col", "value"),
        [
            ("c_text", True),
            ("c_text", 1.5),
            ("c_text", b"x"),
            ("c_text", 2**64),
            ("c_text", ["a"]),
            ("c_integer", True),
            ("c_integer", 1.0),
            ("c_integer", float("nan")),
            ("c_integer", float("inf")),
            ("c_integer", 2**31),
            ("c_integer", "1.5"),
            ("c_integer", "1 OR 1=1"),
            ("c_bigint", 2**63),
            ("c_bigint", -math.inf),
            ("c_boolean", 1),
            ("c_boolean", "true"),
            ("c_uuid", "not-a-uuid"),
            ("c_uuid", ""),
            ("c_uuid", 5),
            ("c_timestamptz", "2026-01-02T03:04:05"),
            ("c_timestamptz", datetime(2026, 1, 2)),
            ("c_timestamptz", "yesterday"),
            ("c_timestamptz", 1700000000),
            ("thread_name", {"eq": None}),
            ("thread_name", None),
            ("thread_name", "a\x00b"),
        ],
    )
    def test_wrong_value_type(self, col, value):
        with pytest.raises(FilterError):
            validate_filters({col: value}, SCHEMA)

    def test_like_needs_string(self):
        with pytest.raises(FilterError, match="got int"):
            validate_filters({"c_text": {"like": 5}}, SCHEMA)

    @pytest.mark.parametrize("value", [1, "true", None, 0])
    def test_is_null_needs_bool(self, value):
        with pytest.raises(FilterError, match="true or false"):
            validate_filters({"c_text": {"is_null": value}}, SCHEMA)

    @pytest.mark.parametrize("value", ["abc", 5, {"a": 1}, None])
    def test_in_needs_list(self, value):
        with pytest.raises(FilterError, match="list"):
            validate_filters({"c_text": {"in": value}}, SCHEMA)

    def test_in_items_type_checked(self):
        with pytest.raises(FilterError):
            validate_filters({"c_integer": {"in": [1, "x"]}}, SCHEMA)
        with pytest.raises(FilterError):
            validate_filters({"c_integer": {"nin": [1, None]}}, SCHEMA)

    def test_bool_not_int(self):
        with pytest.raises(FilterError, match="got bool"):
            validate_filters({"c_bigint": {"gt": False}}, SCHEMA)


# ---------------------------------------------------------------------------
# Rejections and limits
# ---------------------------------------------------------------------------


class TestRejections:
    def test_unknown_column(self):
        with pytest.raises(FilterError, match="Unknown filter column 'nope'"):
            validate_filters({"nope": 1}, SCHEMA)

    @pytest.mark.parametrize("name", ["org_id", "project_id", "user_id"])
    def test_isolation_columns_rejected(self, name):
        with pytest.raises(FilterError, match="isolation column"):
            validate_filters({name: "x"}, SCHEMA)
        with pytest.raises(FilterError, match="isolation column"):
            validate_filters({name: {"is_null": True}}, SCHEMA)

    def test_builtin_not_listed_rejected(self):
        for name in ("thread_id", "meta", "memory_type"):
            with pytest.raises(FilterError, match="Unknown filter column"):
                validate_filters({name: "x"}, SCHEMA)

    def test_memory_schema(self):
        schema = FilterSchema.for_memories(ColumnSet())
        assert schema.names == tuple(MEMORY_FILTER_COLUMNS)
        assert _sql({"memory_type": "semantic"}, schema) == ('"memory_type" = $1', ["semantic"])
        with pytest.raises(FilterError, match="Unknown"):
            validate_filters({"thread_name": "x"}, schema)
        with pytest.raises(FilterError, match="isolation"):
            validate_filters({"user_id": "x"}, schema)

    def test_unknown_operator(self):
        with pytest.raises(FilterError, match="unknown operator 'between'"):
            validate_filters({"c_integer": {"between": [1, 2]}}, SCHEMA)
        with pytest.raises(FilterError, match="unknown operator"):
            validate_filters({"c_integer": {"$gt": 1}}, SCHEMA)
        with pytest.raises(FilterError, match="unknown operator"):
            validate_filters({"c_integer": {1: 1}}, SCHEMA)

    def test_empty_operator_object(self):
        with pytest.raises(FilterError, match="at least one operator"):
            validate_filters({"c_integer": {}}, SCHEMA)

    @pytest.mark.parametrize("filters", [[("a", 1)], "a=1", 5, ["thread_name"]])
    def test_filters_must_be_mapping(self, filters):
        with pytest.raises(FilterError, match="must be an object"):
            validate_filters(filters, SCHEMA)

    def test_none_and_empty(self):
        assert validate_filters(None, SCHEMA) == []
        assert validate_filters({}, SCHEMA) == []
        assert _sql(None) == ("", [])

    def test_condition_limit_counts_keys(self):
        schema = FilterSchema({f"k{i}": "integer" for i in range(MAX_CONDITIONS + 1)})
        ok = {f"k{i}": 1 for i in range(MAX_CONDITIONS)}
        assert len(validate_filters(ok, schema)) == MAX_CONDITIONS
        with pytest.raises(FilterError, match="Too many"):
            validate_filters({f"k{i}": 1 for i in range(MAX_CONDITIONS + 1)}, schema)

    def test_condition_limit_counts_operators(self):
        ops = {"gt": 1, "gte": 1, "lt": 9, "lte": 9, "ne": 5, "eq": 3}
        with pytest.raises(FilterError, match="Too many"):
            validate_filters({"c_integer": ops, "c_bigint": ops}, SCHEMA)

    def test_list_limit(self):
        assert validate_filters({"c_integer": {"in": list(range(MAX_LIST_ITEMS))}}, SCHEMA)
        with pytest.raises(FilterError, match=f"at most {MAX_LIST_ITEMS}"):
            validate_filters({"c_integer": {"in": list(range(MAX_LIST_ITEMS + 1))}}, SCHEMA)

    @pytest.mark.parametrize("op", ["in", "nin"])
    def test_empty_list(self, op):
        with pytest.raises(FilterError, match="empty"):
            validate_filters({"c_text": {op: []}}, SCHEMA)

    @pytest.mark.parametrize("op", ["eq", "like", "in"])
    def test_string_limit(self, op):
        ok = "x" * MAX_STRING_LENGTH
        bad = ok + "x"
        wrap = (lambda v: [v]) if op == "in" else (lambda v: v)
        validate_filters({"c_text": {op: wrap(ok)}}, SCHEMA)
        with pytest.raises(FilterError, match=f"{MAX_STRING_LENGTH} characters"):
            validate_filters({"c_text": {op: wrap(bad)}}, SCHEMA)

    def test_long_string_rejected_for_other_types(self):
        with pytest.raises(FilterError, match="characters"):
            validate_filters({"c_uuid": " " * 300 + str(UID)}, SCHEMA)

    def test_filter_error_is_value_error(self):
        assert issubclass(FilterError, ValueError)

    def test_compile_rejects_unvalidated_conditions(self):
        with pytest.raises(TypeError):
            compile_filters([("c_text", "eq", "x")], [])  # type: ignore[list-item]

    @pytest.mark.parametrize("alias", ['t"; DROP', "T", "1a", "a.b"])
    def test_compile_rejects_bad_alias(self, alias):
        with pytest.raises(ValueError, match="alias"):
            compile_filters(validate_filters({"c_text": "x"}, SCHEMA), [], alias)

    def test_schema_rejects_bad_builtins(self):
        with pytest.raises(ValueError):
            FilterSchema({'bad"name': "text"})
        with pytest.raises(ValueError):
            FilterSchema({"x": "json"})
        with pytest.raises(ValueError, match="clashes"):
            FilterSchema({"plan": "text"}, parse_columns(None, {"plan": "text"}))

    def test_describe(self):
        info = SCHEMA.describe()
        assert info["c_boolean"] == {
            "type": "boolean",
            "operators": ["eq", "ne", "in", "nin", "is_null"],
        }
        assert "org_id" not in info and "user_id" not in info


# ---------------------------------------------------------------------------
# Parameter numbering and alias
# ---------------------------------------------------------------------------


class TestParams:
    def test_numbering_continues_after_existing_params(self):
        params = ["coll", "u1", "org"]
        sql = build_filter_where(
            {"c_text": {"in": ["a", "b"]}, "c_integer": {"gt": 3}, "c_boolean": {"is_null": True}},
            SCHEMA,
            params,
        )
        assert sql == '"c_text" = ANY($4) AND "c_integer" > $5 AND "c_boolean" IS NULL'
        assert params == ["coll", "u1", "org", ["a", "b"], 3]

    def test_alias(self):
        sql, params = _sql({"c_text": {"nin": ["a"]}}, alias="m")
        assert sql == '(m."c_text" IS NULL OR m."c_text" <> ALL($1))'
        _assert_safe(sql, params, _grammar(ALLOWED, alias="m"))

    def test_conditions_are_frozen(self):
        [cond] = validate_filters({"c_text": "x"}, SCHEMA)
        assert isinstance(cond, FilterCondition)
        with pytest.raises(AttributeError):
            cond.op = "like"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Injection attempts
# ---------------------------------------------------------------------------


INJECTIONS = [
    "x'; DROP TABLE threads; --",
    '"; DROP TABLE threads; --',
    "1 OR 1=1",
    "a\" = 'a' OR \"b",
    'thread_name" OR TRUE --',
    "$1 OR TRUE",
    "') OR ('1'='1",
    "\\'; SELECT pg_sleep(10); --",
    "c_text = c_text",
    "pg_catalog.pg_user",
    "/* comment */",
    "%' OR '1'='1",
]


class TestInjection:
    @pytest.mark.parametrize("payload", INJECTIONS)
    def test_injection_in_column_name(self, payload):
        with pytest.raises(FilterError) as info:
            validate_filters({payload: "x"}, SCHEMA)
        assert payload not in str(info.value)
        assert "DROP" not in str(info.value)

    @pytest.mark.parametrize("payload", INJECTIONS)
    def test_injection_in_operator(self, payload):
        with pytest.raises(FilterError) as info:
            validate_filters({"c_text": {payload: "x"}}, SCHEMA)
        assert payload not in str(info.value)

    @pytest.mark.parametrize("payload", INJECTIONS)
    @pytest.mark.parametrize("op", ["eq", "ne", "like", "ilike", "in", "nin"])
    def test_injection_in_value_stays_a_param(self, payload, op):
        value = [payload] if op in ("in", "nin") else payload
        sql, params = _sql({"c_text": {op: value}, "thread_name": payload})
        _assert_safe(sql, params)
        assert payload not in sql
        assert params[0] == value and params[1] == payload

    @pytest.mark.parametrize("payload", INJECTIONS)
    def test_injection_in_typed_value_rejected_without_echo(self, payload):
        for col in ("c_integer", "c_uuid", "c_timestamptz", "c_boolean"):
            with pytest.raises(FilterError) as info:
                validate_filters({col: payload}, SCHEMA)
            assert payload not in str(info.value)

    def test_error_messages_never_contain_sql(self):
        cases = [
            {"nope": 1},
            {"org_id": 1},
            {"c_text": {"gt": 1}},
            {"c_text": {"zz": 1}},
            {"c_integer": "x"},
            {"c_text": {"in": []}},
        ]
        for filters in cases:
            with pytest.raises(FilterError) as info:
                validate_filters(filters, SCHEMA)
            message = str(info.value)
            for token in ("$1", "SELECT", "WHERE", "IS DISTINCT", "ANY(", '"c_'):
                assert token not in message


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**70), max_value=2**70),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=300),
    st.binary(max_size=10),
    st.datetimes(timezones=st.one_of(st.none(), st.just(UTC))),
    st.uuids(),
)
_value = st.recursive(
    _scalar,
    lambda inner: st.one_of(
        st.lists(inner, max_size=5),
        st.dictionaries(
            st.one_of(st.sampled_from(sorted(OPERATORS)), st.text()), inner, max_size=4
        ),
    ),
    max_leaves=12,
)
_key = st.one_of(
    st.sampled_from(sorted(ALLOWED | {"org_id", "project_id", "user_id", "thread_id"})),
    st.text(max_size=40),
    st.integers(),
)
_op_dict = st.dictionaries(
    st.one_of(st.sampled_from(sorted(OPERATORS)), st.text(max_size=20)), _value, max_size=4
)
_filters = st.one_of(
    st.dictionaries(_key, st.one_of(_value, _op_dict), max_size=12),
    _value,
)


class TestProperties:
    @settings(max_examples=600, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(filters=_filters, prefix=st.lists(st.integers(), max_size=3))
    def test_random_input_only_raises_filter_error_and_sql_is_safe(self, filters, prefix):
        params = list(prefix)
        try:
            sql = build_filter_where(filters, SCHEMA, params)
        except FilterError as e:
            assert not re.search(r"\$\d", str(e))
            return
        assert GRAMMAR.match(sql), sql
        numbers = [int(n) for n in re.findall(r"\$(\d+)", sql)]
        assert numbers == list(range(len(prefix) + 1, len(params) + 1))
        # Nothing from the input other than allow-listed names appears in the SQL text.
        stripped = re.sub(r'"[a-z_]+"', "", sql)
        assert re.fullmatch(r"[A-Z0-9$() <>=,_.]*", stripped), stripped

    @settings(max_examples=300, deadline=None)
    @given(name=st.text(min_size=1, max_size=60), value=st.text(max_size=50))
    def test_random_column_names_never_reach_sql(self, name, value):
        params: list = []
        try:
            sql = build_filter_where({name: value}, SCHEMA, params)
        except FilterError as e:
            # Only well-formed names are ever quoted back to the caller.
            if not re.fullmatch(r"[A-Za-z0-9_]{1,63}", name):
                assert repr(name) not in str(e)
            return
        assert name in ALLOWED
        assert sql == f'"{name}" = $1'

    @settings(max_examples=300, deadline=None)
    @given(op=st.text(min_size=1, max_size=30), value=_scalar)
    def test_random_operators(self, op, value):
        try:
            sql, _ = _sql({"c_text": {op: value}})
        except FilterError:
            return
        assert op in OPERATORS_BY_TYPE["text"]
        assert GRAMMAR.match(sql)


_VALID_SCALARS = {
    "text": st.text(max_size=MAX_STRING_LENGTH).filter(lambda s: "\x00" not in s),
    "bigint": st.integers(min_value=-(2**63), max_value=2**63 - 1),
    "integer": st.integers(min_value=-(2**31), max_value=2**31 - 1),
    "timestamptz": st.datetimes(timezones=st.just(UTC)),
    "boolean": st.booleans(),
    "uuid": st.uuids(),
}


@st.composite
def _valid_filters(draw):
    names = draw(st.lists(st.sampled_from(sorted(EXTRA)), unique=True, min_size=1, max_size=5))
    filters = {}
    budget = MAX_CONDITIONS
    for name in names:
        type_name = EXTRA[name]
        ops = draw(st.lists(st.sampled_from(OPERATORS_BY_TYPE[type_name]), unique=True, max_size=2))
        ops = ops[:budget] or ["eq"]
        budget -= len(ops)
        if budget < 0:
            break
        spec = {}
        for op in ops:
            if op == "is_null":
                spec[op] = draw(st.booleans())
            elif op in ("in", "nin"):
                spec[op] = draw(st.lists(_VALID_SCALARS[type_name], min_size=1, max_size=5))
            else:
                spec[op] = draw(_VALID_SCALARS[type_name])
        filters[name] = spec
    return filters


class TestValidProperties:
    @settings(max_examples=400, deadline=None)
    @given(filters=_valid_filters(), offset=st.integers(min_value=0, max_value=5))
    def test_valid_filters_compile_to_grammar(self, filters, offset):
        params = list(range(offset))
        sql = build_filter_where(filters, SCHEMA, params)
        assert GRAMMAR.match(sql), sql
        expected = sum(1 for spec in filters.values() for op in spec if op != "is_null")
        assert len(params) == offset + expected
        for spec in filters.values():
            for value in spec.values():
                if isinstance(value, str) and value:
                    # A string value never leaks into the SQL text unless it is SQL-ish
                    # vocabulary that the grammar already constrains.
                    assert value not in sql or re.fullmatch(r"[A-Za-z0-9 $()<>=_\"]*", value)
