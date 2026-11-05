"""Safe filters on declared columns, compiled to a SQL WHERE fragment with bind parameters.

``PgCheckpointer.alist_threads(filters=...)`` and the ``filters`` argument of ``PgVectorStore``
search, list and forget take a small filter language that a client may send as JSON::

    {"plan": "pro"}  # equality
    {"seats": {"gte": 5, "lt": 50}}  # one or more operators on a column
    {"created_at": {"gte": "2026-01-01T00:00:00+00:00"}, "region": {"in": ["eu", "us"]}}

Rules (decision D9 of the store plan):

- Columns: the backend's built-in filterable columns (see :data:`THREAD_FILTER_COLUMNS` and
  :data:`MEMORY_FILTER_COLUMNS`) and the developer's declared **extra** columns. Tenant columns
  and ``user_id`` are isolation columns: the server sets them from the caller's identity, so a
  filter naming one is rejected.
- Operators come from a fixed table per column type (:data:`OPERATORS_BY_TYPE`). An extra
  column of any other type (``jsonb``, ``text[]``, ``numeric(10,2)``, ``date``, an enum, ...)
  supports only ``is_null`` (:data:`OTHER_TYPE_OPERATORS`).
- Every value is a bind parameter (``$n``), type-checked and converted for its column type. Only
  allow-listed column names (quoted) and fixed operator SQL ever reach the query text.
- Conditions are joined with AND, and the backend always ANDs them after its isolation WHERE,
  so a filter can only narrow what the caller may already see.
- Limits: :data:`MAX_CONDITIONS` conditions, :data:`MAX_LIST_ITEMS` items per ``in`` / ``nin``
  list (which must not be empty), :data:`MAX_STRING_LENGTH` characters per string value.

Operator semantics:

- ``eq``, ``gt``, ``gte``, ``lt``, ``lte``: SQL ``=``, ``>``, ``>=``, ``<``, ``<=``. A row whose
  column is NULL never matches them.
- ``ne`` and ``nin`` treat NULL as a value of its own: ``ne`` is ``IS DISTINCT FROM`` and ``nin``
  is ``(col IS NULL OR col <> ALL(...))``, so a row with no value *does* match "not equal to x".
  Add ``{"is_null": false}`` to leave those rows out.
- ``in``: ``col = ANY($n)``.
- ``like`` / ``ilike`` (text only): the value is a parameter used as the pattern, so ``%`` and
  ``_`` are the caller's wildcards and ``\\`` escapes them. A pattern with a leading ``%`` scans
  every row the caller can see (``pg_trgm`` can index it).
- ``is_null``: takes ``true`` (``IS NULL``) or ``false`` (``IS NOT NULL``).

Values per type: ``text`` takes a string (or an integer, sent as its decimal string);
``integer`` / ``bigint`` take an integer (not a bool) or a string of digits, in range;
``boolean`` takes a bool; ``uuid`` takes a ``uuid.UUID`` or a UUID string; ``timestamptz``
takes a timezone-aware ``datetime`` or an ISO 8601 string with an offset. Floats are never
accepted, so NaN and infinity cannot reach a query. ``None`` is not a value: use ``is_null``.

Every problem raises :class:`FilterError`, a ``ValueError`` whose message is safe to return to
an API client: it names the column and operator only when they are well-formed names, gives a
value's type but never the value, and never contains SQL.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tenxgraph.storage.columns import COLUMN_TYPES, ColumnSet, ColumnSpec, quote_ident


__all__ = [
    "MAX_CONDITIONS",
    "MAX_LIST_ITEMS",
    "MAX_STRING_LENGTH",
    "MEMORY_FILTER_COLUMNS",
    "OPERATORS",
    "OPERATORS_BY_TYPE",
    "OTHER_TYPE_OPERATORS",
    "THREAD_FILTER_COLUMNS",
    "FilterColumn",
    "FilterCondition",
    "FilterError",
    "FilterSchema",
    "build_filter_where",
    "compile_filters",
    "validate_filters",
]

#: Most conditions (column + operator pairs) one ``filters`` object may hold.
MAX_CONDITIONS = 10
#: Most items in one ``in`` / ``nin`` list.
MAX_LIST_ITEMS = 100
#: Longest string value, in characters.
MAX_STRING_LENGTH = 256

_TEXT_OPS = ("eq", "ne", "in", "nin", "like", "ilike", "is_null")
_ORDERED_OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "is_null")
_EQUALITY_OPS = ("eq", "ne", "in", "nin", "is_null")

#: Operators allowed on each declared column type.
OPERATORS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "text": _TEXT_OPS,
    "bigint": _ORDERED_OPS,
    "integer": _ORDERED_OPS,
    "timestamptz": _ORDERED_OPS,
    "boolean": _EQUALITY_OPS,
    "uuid": _EQUALITY_OPS,
}

#: Operators allowed on an extra column whose type is not in :data:`OPERATORS_BY_TYPE`.
OTHER_TYPE_OPERATORS: tuple[str, ...] = ("is_null",)

#: Every operator name the filter language knows.
OPERATORS: frozenset[str] = frozenset(op for ops in OPERATORS_BY_TYPE.values() for op in ops)

# Fixed SQL for the comparison operators; nothing else is ever placed between a column and a
# parameter.
_COMPARISON_SQL = {
    "eq": "=",
    "ne": "IS DISTINCT FROM",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "like": "LIKE",
    "ilike": "ILIKE",
}

#: Built-in filterable columns of the checkpointer's ``threads`` table.
THREAD_FILTER_COLUMNS: dict[str, str] = {
    "thread_name": "text",
    "created_at": "timestamptz",
    "updated_at": "timestamptz",
}

#: Built-in filterable columns of the ``PgVectorStore`` table.
MEMORY_FILTER_COLUMNS: dict[str, str] = {
    "memory_type": "text",
    "category": "text",
    "thread_id": "text",
    "memory_key": "text",
    "created_at": "timestamptz",
    "updated_at": "timestamptz",
}

# Names shown in error messages; anything else is described, never echoed.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,63}$")
_ALIAS_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_TEXT_INT_RANGE = (-(2**63), 2**63 - 1)
_MAX_TYPE_LABEL = 128


class FilterError(ValueError):
    """A ``filters`` object is malformed or not allowed. The message is safe to show a client."""


def _label(name: Any) -> str:
    """``'name'`` for a well-formed name, a neutral description otherwise (never raw input)."""
    if isinstance(name, str) and _SAFE_NAME_RE.match(name):
        return repr(name)
    return "(invalid name)"


@dataclass(frozen=True)
class FilterColumn:
    """A column a filter may name: its (validated) name and declared type."""

    name: str
    type: str

    @property
    def quoted(self) -> str:
        return quote_ident(self.name)

    @property
    def operators(self) -> tuple[str, ...]:
        return OPERATORS_BY_TYPE.get(self.type, OTHER_TYPE_OPERATORS)


@dataclass(frozen=True)
class FilterCondition:
    """One validated condition: a column, an operator and the converted value."""

    column: FilterColumn
    op: str
    value: Any


class FilterSchema:
    """The columns one backend lets a filter name, and the isolation columns it must refuse.

    ``builtins`` maps built-in column names to types; ``columns`` is the backend's declared
    ``ColumnSet``. Extra columns become filterable; tenant columns (and ``user_id``) are
    isolation columns and are rejected by name.
    """

    def __init__(self, builtins: Mapping[str, str], columns: ColumnSet | None = None):
        columns = columns if columns is not None else ColumnSet()
        allowed: dict[str, FilterColumn] = {}
        for name, type_name in builtins.items():
            quote_ident(name)
            if type_name not in COLUMN_TYPES:
                raise ValueError(f"Unsupported built-in filter column type: {type_name!r}")
            allowed[name] = FilterColumn(name, type_name)
        for spec in columns.extra:
            if spec.name in allowed:
                raise ValueError(f"Extra column {spec.name!r} clashes with a built-in column")
            allowed[spec.name] = FilterColumn(spec.name, spec.type)
        self._columns = allowed
        self._isolation = frozenset({"user_id", *columns.tenant_names})

    @classmethod
    def for_threads(cls, columns: ColumnSet | None = None) -> FilterSchema:
        """Schema for ``PgCheckpointer.alist_threads``: thread built-ins plus extra columns."""
        return cls(THREAD_FILTER_COLUMNS, columns)

    @classmethod
    def for_memories(cls, columns: ColumnSet | None = None) -> FilterSchema:
        """Schema for ``PgVectorStore``: memory built-ins plus extra columns."""
        return cls(MEMORY_FILTER_COLUMNS, columns)

    @property
    def names(self) -> tuple[str, ...]:
        """Filterable column names, in declaration order."""
        return tuple(self._columns)

    def get(self, name: Any) -> FilterColumn | None:
        return self._columns.get(name) if isinstance(name, str) else None

    def is_isolation(self, name: Any) -> bool:
        """Whether ``name`` is ``user_id`` or a tenant column (never allowed in a filter)."""
        return isinstance(name, str) and name in self._isolation

    def describe(self) -> dict[str, dict[str, Any]]:
        """``{name: {"type", "operators"}}`` for documentation or an API schema."""
        return {
            col.name: {"type": col.type, "operators": list(col.operators)}
            for col in self._columns.values()
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_string(column: FilterColumn, value: str) -> None:
    if len(value) > MAX_STRING_LENGTH:
        raise FilterError(
            f"Filter on {_label(column.name)}: string values are limited to "
            f"{MAX_STRING_LENGTH} characters"
        )
    if "\x00" in value:
        raise FilterError(f"Filter on {_label(column.name)}: strings cannot contain NUL")


def _type_label(type_name: str) -> str:
    """A declared type for an error message (types are validated when declared)."""
    return type_name if len(type_name) <= _MAX_TYPE_LABEL else "(long type name)"


def _type_error(column: FilterColumn, op: str, value: Any) -> FilterError:
    return FilterError(
        f"Filter on {_label(column.name)} ({op}): column type is {column.type}, got "
        f"{type(value).__name__} value"
    )


def _convert_scalar(column: FilterColumn, op: str, value: Any) -> Any:
    """Check one value against the column type and return what asyncpg sends."""
    if value is None:
        raise FilterError(
            f"Filter on {_label(column.name)} ({op}): null is not a value; use is_null"
        )
    if isinstance(value, str):
        _check_string(column, value)
    if column.type == "text":
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            low, high = _TEXT_INT_RANGE
            if not low <= value <= high:
                raise FilterError(
                    f"Filter on {_label(column.name)} ({op}): integer value out of range"
                )
            return str(value)
        raise _type_error(column, op, value)
    spec = ColumnSpec(name=column.name, type=column.type, required=False, kind="extra")
    try:
        return spec.convert(value)
    except (ValueError, TypeError, OverflowError, ArithmeticError):
        # ColumnValueError's text mentions config["user"]; give a filter-specific message.
        raise _type_error(column, op, value) from None


def _convert(column: FilterColumn, op: str, value: Any) -> Any:
    if op == "is_null":
        if not isinstance(value, bool):
            raise FilterError(f"Filter on {_label(column.name)} (is_null): use true or false")
        return value
    if op in ("in", "nin"):
        if not isinstance(value, list | tuple):
            raise FilterError(f"Filter on {_label(column.name)} ({op}): give a list of values")
        if not value:
            raise FilterError(f"Filter on {_label(column.name)} ({op}): the list is empty")
        if len(value) > MAX_LIST_ITEMS:
            raise FilterError(
                f"Filter on {_label(column.name)} ({op}): at most {MAX_LIST_ITEMS} values"
            )
        return [_convert_scalar(column, op, item) for item in value]
    if op in ("like", "ilike"):
        if not isinstance(value, str):
            raise _type_error(column, op, value)
        _check_string(column, value)
        return value
    return _convert_scalar(column, op, value)


def _too_many() -> FilterError:
    return FilterError(f"Too many filter conditions: at most {MAX_CONDITIONS}")


def validate_filters(filters: Any, schema: FilterSchema) -> list[FilterCondition]:  # noqa: PLR0912
    """Check ``filters`` against ``schema`` and return the converted conditions.

    Needs no database, so an API can validate a request before running it. ``None`` or an
    empty mapping returns ``[]``. Raises :class:`FilterError` for anything not allowed.
    """
    if filters is None:
        return []
    if not isinstance(filters, Mapping):
        raise FilterError("filters must be an object mapping column names to conditions")
    if len(filters) > MAX_CONDITIONS:
        raise _too_many()
    conditions: list[FilterCondition] = []
    for name, spec in filters.items():
        if schema.is_isolation(name):
            raise FilterError(
                f"Column {_label(name)} is an isolation column set from the caller's identity; "
                "it cannot be used in filters"
            )
        column = schema.get(name)
        if column is None:
            allowed = ", ".join(schema.names) or "none"
            raise FilterError(
                f"Unknown filter column {_label(name)}; filterable columns: {allowed}"
            )
        if isinstance(spec, Mapping):
            if not spec:
                raise FilterError(f"Filter on {_label(name)}: give at least one operator")
            items = list(spec.items())
        else:
            items = [("eq", spec)]
        for op, value in items:
            if not isinstance(op, str) or op not in OPERATORS:
                raise FilterError(
                    f"Filter on {_label(name)}: unknown operator {_label(op)}; use one of "
                    f"{', '.join(column.operators)}"
                )
            if op not in column.operators and column.type not in OPERATORS_BY_TYPE:
                raise FilterError(
                    f"Filter on {_label(name)}: operator {op!r} is not supported; column type "
                    f"{_type_label(column.type)} supports only is_null"
                )
            if op not in column.operators:
                raise FilterError(
                    f"Filter on {_label(name)}: operator {op!r} is not supported for "
                    f"{column.type} columns; use one of {', '.join(column.operators)}"
                )
            if len(conditions) >= MAX_CONDITIONS:
                raise _too_many()
            conditions.append(FilterCondition(column, op, _convert(column, op, value)))
    return conditions


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


def compile_filters(
    conditions: Sequence[FilterCondition], params: list[Any], alias: str | None = None
) -> str:
    """Render validated ``conditions`` as an AND-joined WHERE fragment.

    Each value is appended to ``params`` and referenced as ``$n`` with ``n = len(params)``, so
    the numbering continues after whatever the caller (the isolation WHERE) already added.
    Returns an empty string for no conditions. ``alias`` is a table alias chosen by the
    backend, never input; it is checked anyway.
    """
    if alias is not None and not _ALIAS_RE.match(alias):
        raise ValueError(f"Invalid table alias: {alias!r}")
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    for cond in conditions:
        if not isinstance(cond, FilterCondition) or cond.op not in OPERATORS:
            raise TypeError("compile_filters takes conditions from validate_filters")
        column = f"{prefix}{cond.column.quoted}"
        if cond.op == "is_null":
            clauses.append(f"{column} IS NULL" if cond.value else f"{column} IS NOT NULL")
            continue
        params.append(cond.value)
        placeholder = f"${len(params)}"
        if cond.op == "in":
            clauses.append(f"{column} = ANY({placeholder})")
        elif cond.op == "nin":
            clauses.append(f"({column} IS NULL OR {column} <> ALL({placeholder}))")
        else:
            clauses.append(f"{column} {_COMPARISON_SQL[cond.op]} {placeholder}")
    return " AND ".join(clauses)


def build_filter_where(
    filters: Any, schema: FilterSchema, params: list[Any], alias: str | None = None
) -> str:
    """:func:`validate_filters` then :func:`compile_filters`: ``filters`` -> WHERE fragment."""
    return compile_filters(validate_filters(filters, schema), params, alias)
