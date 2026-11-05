"""Developer-declared columns on storage tables: tenant columns and extra columns.

``PgCheckpointer`` and ``PgVectorStore`` accept two opt-in column sets:

- **tenant columns** (``tenant_columns=``) isolate data. Every read, update and delete is
  filtered by the tenant columns the request's isolation scope picks, so ``org_id`` /
  ``project_id`` act like ``user_id`` does today. Required by default.
- **extra columns** (``extra_columns=``) are plain data stored with each row (and, from Sprint
  3b, filterable). Optional by default.

Trusted-source rule: tenant and extra column values come only from ``config["user"]``. ``user``
is a reserved config key: clients cannot set it. Your auth class fills it
(``BaseAuth.authenticate()`` result or JWT claims). Return ``org_id``, ``project_id``,
``public_id`` there. Never read them from request config, metadata, state or model output.

A declaration is a mapping of column name to type, or to ``{"type": ..., "required": ...}``::

    tenant_columns = ["org_id"]  # a list of names means text columns
    extra_columns = {
        "public_id": "text",
        "settings": "jsonb",
        "tags": "text[]",
        "seats": {"type": "integer", "required": True},
    }

Names match ``^[a-z_][a-z0-9_]{0,62}$`` and must not be one of the table's built-in columns.
Every identifier is validated and quoted before it reaches SQL, and every value is a bind
parameter.

Types follow two rule sets:

- **Tenant columns** are scalar ids: ``text``, ``bigint``, ``integer`` or ``uuid``. Anything
  else (``boolean``, ``timestamptz``, ``jsonb``, arrays, ...) is rejected.
- **Extra columns** take any Postgres type. The six types ``text``, ``bigint``, ``integer``,
  ``boolean``, ``uuid`` and ``timestamptz`` are checked and converted in Python (see
  :meth:`ColumnSpec.convert`) and support the filter operators of
  ``storage/sql_filter.py``. Any other type (``jsonb``, ``numeric(10,2)``, ``text[]``,
  ``date``, ``character varying(64)``, ``myschema.my_enum``, ...) gets a cheap syntax check
  when declared (:data:`EXTRA_TYPE_RE`: no quotes, semicolons, comments, or parentheses beyond
  a numeric modifier list) and is resolved on the server at setup with a parameterized
  ``format_type(to_regtype($1), to_regtypemod($1))``. Only the canonical name the server
  returns is used in DDL, so the declared text never reaches SQL; an unknown type fails setup.
  ``json`` / ``jsonb`` values are serialised with ``json.dumps`` (dicts, lists, strings,
  numbers, bools) and cast in SQL; other values go to asyncpg as given, and a value its codec
  cannot encode raises :class:`ColumnValueError` naming the column and type, never the value.
  Filters on these columns allow only ``is_null``.

This module holds the parts both backends share: spec parsing, value resolution from
``config["user"]``, the isolation WHERE builder and the native row-level security (RLS) policy
text. ``storage/sql_filter.py`` (Sprint 3b) reuses ``ColumnSpec`` for filter validation.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


__all__ = [
    "COLUMN_TYPES",
    "EXTRA_TYPE_RE",
    "FORMAT_TYPE_NAMES",
    "JSON_TYPES",
    "RLS_SCOPE_SETTING",
    "TENANT_COLUMN_TYPES",
    "ColumnSet",
    "ColumnSpec",
    "ColumnValueError",
    "MissingColumnValueError",
    "apply_settings",
    "build_isolation_where",
    "column_ddl",
    "column_value_errors",
    "ensure_columns",
    "ensure_rls",
    "index_name",
    "isolation_conditions",
    "parse_columns",
    "quote_ident",
    "resolve_column_type",
    "rls_all_rows_settings",
    "rls_policy_expression",
    "rls_policy_name",
    "rls_settings",
    "user_value",
    "warn_if_rls_bypassed",
    "write_conditions",
]

ColumnKind = Literal["tenant", "extra"]

#: The built-in types: checked and converted in Python, filterable. Declared type -> DDL type.
COLUMN_TYPES: dict[str, str] = {
    "text": "TEXT",
    "bigint": "BIGINT",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "uuid": "UUID",
    "timestamptz": "TIMESTAMPTZ",
}

#: Declared type -> ``format_type()`` output, to check a column that already exists.
FORMAT_TYPE_NAMES: dict[str, str] = {
    "text": "text",
    "bigint": "bigint",
    "integer": "integer",
    "boolean": "boolean",
    "uuid": "uuid",
    "timestamptz": "timestamp with time zone",
}

#: Types a tenant column may have: scalar ids only.
TENANT_COLUMN_TYPES: tuple[str, ...] = ("text", "bigint", "integer", "uuid")

#: Extra column types whose values are serialised with ``json.dumps`` and cast in SQL.
JSON_TYPES: dict[str, str] = {
    "json": "json",
    "jsonb": "jsonb",
    "pg_catalog.json": "json",
    "pg_catalog.jsonb": "jsonb",
}

_TYPE_WORD = r"[a-z_][a-z0-9_]{0,62}"
_TYPE_MOD = r"\( ?\d{1,10} ?(?:, ?\d{1,10} ?)?\)"
#: Syntax an extra column type must match (after lowercasing and collapsing whitespace):
#: an optionally schema-qualified name, more words (``timestamp with time zone``), at most a
#: numeric modifier list per word and array brackets. Only a cheap first check: setup resolves
#: the type on the server and uses the server's canonical name in DDL.
EXTRA_TYPE_RE = re.compile(
    rf"^{_TYPE_WORD}(?:\.{_TYPE_WORD})?(?: ?{_TYPE_MOD})?"
    rf"(?: {_TYPE_WORD}(?: ?{_TYPE_MOD})?){{0,4}}(?:\[\d{{0,10}}\]){{0,6}}$"
)
_MAX_TYPE_LENGTH = 128
# What a canonical name from format_type() may contain; checked again before it is used in DDL.
_CANONICAL_TYPE_RE = re.compile(r'^[A-Za-z0-9_ .,()\[\]"]{1,256}$')
# asyncpg's message when its codec cannot encode a bind parameter.
_ARGUMENT_ERROR_RE = re.compile(r"invalid input for query argument \$(\d+)")
# to_regtypemod() (exact typmods such as numeric(10,2)) exists from PostgreSQL 17.
_REGTYPEMOD_MIN_SERVER = 170000

#: Name of the setting that carries the isolation scope for native RLS policies.
RLS_SCOPE_SETTING = "tenxgraph.scope"
#: Prefix of the per-column settings native RLS policies compare against.
RLS_SETTING_PREFIX = "tenxgraph."

_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
# Postgres truncates identifiers longer than this many bytes.
_MAX_IDENTIFIER = 63
_INTEGER_RE = re.compile(r"^[+-]?\d{1,20}$")
_INT_RANGES = {
    "integer": (-(2**31), 2**31 - 1),
    "bigint": (-(2**63), 2**63 - 1),
}
_SPEC_KEYS = frozenset({"type", "required"})
# ``user_id`` is a scope column name in every backend, so it can never be declared.
_ALWAYS_RESERVED = frozenset({"user_id"})


class ColumnValueError(ValueError):
    """A tenant or extra column value from ``config["user"]`` has the wrong type."""


class MissingColumnValueError(ColumnValueError):
    """A required column, or a column the isolation scope picks, has no value."""


def quote_ident(name: str) -> str:
    """Return ``name`` as a double-quoted identifier after checking it against the name rule."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"Invalid column name: {name!r}")
    return f'"{name}"'


def index_name(prefix: str, column: str) -> str:
    """An index name ``<prefix>_<column>`` that fits Postgres' 63-byte identifier limit.

    Longer names are cut and given a short hash of the full name, so two long column names
    with the same start never map to the same index.
    """
    name = f"{prefix}_{column}"
    if len(name) <= _MAX_IDENTIFIER:
        return name
    digest = hashlib.blake2b(name.encode(), digest_size=4).hexdigest()
    return f"{name[:54]}_{digest}"


@dataclass(frozen=True)
class ColumnSpec:
    """One declared column: its name, type, whether a value is required, and its kind.

    ``type`` is the declared type, lowercased with whitespace collapsed. A built-in type
    (:data:`COLUMN_TYPES`) is checked and converted in Python; any other type (extra columns
    only) is resolved on the server at setup.
    """

    name: str
    type: str
    required: bool
    kind: ColumnKind

    @property
    def quoted(self) -> str:
        return quote_ident(self.name)

    @property
    def builtin(self) -> bool:
        """Whether the type is one of the six built-in types (Python-checked, filterable)."""
        return self.type in COLUMN_TYPES

    @property
    def json_type(self) -> str | None:
        """``"json"`` or ``"jsonb"`` for a JSON column, else None."""
        return JSON_TYPES.get(self.type)

    @property
    def sql_type(self) -> str:
        """The DDL type of a built-in type. Other types need :func:`resolve_column_type`."""
        if not self.builtin:
            raise ValueError(
                f"Column {self.name!r} has type {self.type!r}, which is resolved on the server "
                "at setup; it has no fixed SQL type"
            )
        return COLUMN_TYPES[self.type]

    def placeholder(self, param: str) -> str:
        """The SQL for bind parameter ``param`` (``"$5"``) holding this column's value.

        JSON columns get ``$5::text::jsonb`` (the value is sent as JSON text); every other
        column the bare parameter. ``param`` is built by the backend, never input.
        """
        if not re.fullmatch(r"\$\d{1,5}", param):
            raise ValueError(f"Invalid bind parameter: {param!r}")
        json_type = self.json_type
        return f"{param}::text::{json_type}" if json_type else param

    @property
    def setting_name(self) -> str:
        """The RLS setting this column's policy reads (``tenxgraph.<name>``)."""
        return f"{RLS_SETTING_PREFIX}{self.name}"

    def convert(self, value: Any) -> Any:  # noqa: PLR0911, PLR0912
        """Check ``value`` against the column type and return it in the type asyncpg sends.

        ``None`` (and, for tenant columns, an empty string) means "no value" and returns
        ``None``. A value of the wrong type raises ``ColumnValueError``; nothing is coerced
        silently except the documented cases below.

        - ``text``: ``str``; an ``int`` is turned into its decimal string.
        - ``integer`` / ``bigint``: ``int`` (not ``bool``) or a string of digits, in range.
        - ``boolean``: ``bool`` only.
        - ``uuid``: ``uuid.UUID`` or a UUID string.
        - ``timestamptz``: a timezone-aware ``datetime`` or ISO 8601 string with an offset.
        - ``json`` / ``jsonb`` (extra only): a ``dict``, ``list``, ``str``, ``int``, ``float``
          or ``bool``, serialised with ``json.dumps`` (NaN and infinity are refused).
        - any other type (extra only): returned as given, for asyncpg's codec to encode;
          :func:`column_value_errors` turns a codec failure into ``ColumnValueError``.
        """
        if value is None:
            return None
        kind = self.type
        if kind not in COLUMN_TYPES:
            if self.json_type is None:
                return value
            if not isinstance(value, dict | list | str | int | float | bool):
                raise self._type_error(value)
            try:
                return json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                raise ColumnValueError(
                    f'Column {self.name!r} is {self.type}: the value from config["user"] '
                    "cannot be serialised as JSON"
                ) from None
        if kind == "text":
            if isinstance(value, bool) or not isinstance(value, str | int):
                raise self._type_error(value)
            text = str(value)
            if self.kind == "tenant" and not text.strip():
                return None
            return text
        if kind in _INT_RANGES:
            if isinstance(value, bool):
                raise self._type_error(value)
            if isinstance(value, str):
                if not _INTEGER_RE.match(value.strip()):
                    raise self._type_error(value)
                value = int(value.strip())
            if not isinstance(value, int):
                raise self._type_error(value)
            low, high = _INT_RANGES[kind]
            if not low <= value <= high:
                raise ColumnValueError(f"Column {self.name!r}: {value} is out of {kind} range")
            return value
        if kind == "boolean":
            if not isinstance(value, bool):
                raise self._type_error(value)
            return value
        if kind == "uuid":
            if isinstance(value, uuid.UUID):
                return value
            if isinstance(value, str):
                if self.kind == "tenant" and not value.strip():
                    return None
                try:
                    return uuid.UUID(value.strip())
                except ValueError:
                    raise self._type_error(value) from None
            raise self._type_error(value)
        # timestamptz
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.strip())
            except ValueError:
                raise self._type_error(value) from None
        if not isinstance(value, datetime):
            raise self._type_error(value)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ColumnValueError(
                f"Column {self.name!r} is timestamptz: the datetime needs a timezone"
            )
        return value

    def to_setting(self, value: Any) -> str:
        """Text form of a converted value for ``set_config`` (empty string for no value)."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _type_error(self, value: Any) -> ColumnValueError:
        return ColumnValueError(
            f"Column {self.name!r} is {self.type}: got {type(value).__name__} value from "
            'config["user"]'
        )


@dataclass(frozen=True)
class ColumnSet:
    """The tenant and extra columns declared on one backend, in declaration order."""

    tenant: tuple[ColumnSpec, ...] = ()
    extra: tuple[ColumnSpec, ...] = ()
    _by_name: dict[str, ColumnSpec] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._by_name.update({spec.name: spec for spec in (*self.tenant, *self.extra)})

    def __bool__(self) -> bool:
        return bool(self.tenant or self.extra)

    @property
    def all(self) -> tuple[ColumnSpec, ...]:
        return (*self.tenant, *self.extra)

    @property
    def tenant_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.tenant)

    @property
    def extra_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.extra)

    def get(self, name: str) -> ColumnSpec | None:
        return self._by_name.get(name)

    def builtin_only(self) -> ColumnSet:
        """The columns whose type is built in (no server lookup needed for their DDL)."""
        return ColumnSet(
            tenant=tuple(spec for spec in self.tenant if spec.builtin),
            extra=tuple(spec for spec in self.extra if spec.builtin),
        )

    def describe(self) -> dict[str, dict[str, Any]]:
        """``{name: {"type", "required", "kind"}}``, for comparisons and error messages."""
        return {
            spec.name: {"type": spec.type, "required": spec.required, "kind": spec.kind}
            for spec in self.all
        }

    def resolve(self, config: Any, *, write: bool = False) -> dict[str, Any]:
        """Read every declared column's value from ``config["user"]`` and convert it.

        Only ``config["user"]`` is read (see the module docstring for why). With
        ``write=True`` a required column without a value raises
        ``MissingColumnValueError``; otherwise a missing value is ``None``.
        """
        user = config.get("user") if isinstance(config, Mapping) else None
        values: dict[str, Any] = {}
        for spec in self.all:
            value = spec.convert(user_value(user, spec.name))
            if write and spec.required and value is None:
                raise MissingColumnValueError(
                    f'{spec.kind.capitalize()} column {spec.name!r} is required but config["user"] '
                    f"has no {spec.name!r}. Set it in your auth class (BaseAuth.authenticate() "
                    "or JWT claims)."
                )
            values[spec.name] = value
        return values

    def scope_specs(self, picked: Sequence[str] | None) -> tuple[ColumnSpec, ...]:
        """The tenant columns an isolation scope picks, in declaration order.

        ``picked`` is the resolved scope column list (``user_id`` is ignored here; backends
        handle it). ``None`` means no authz policy: every tenant column applies.
        """
        if picked is None:
            return self.tenant
        chosen = set(picked)
        return tuple(spec for spec in self.tenant if spec.name in chosen)


def user_value(user: Any, name: str) -> Any:
    """``user[name]`` for a mapping, ``getattr(user, name)`` for an object, else ``None``."""
    if user is None:
        return None
    if isinstance(user, Mapping):
        return user.get(name)
    return getattr(user, name, None)


def _parse_one(name: Any, raw: Any, kind: ColumnKind) -> ColumnSpec:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid {kind} column name {name!r}: use lowercase letters, digits and "
            "underscores, starting with a letter or underscore, at most 63 characters."
        )
    required = kind == "tenant"
    type_name: Any
    if isinstance(raw, str):
        type_name = raw
    elif isinstance(raw, Mapping):
        unknown = set(raw) - _SPEC_KEYS
        if unknown:
            raise ValueError(
                f"{kind.capitalize()} column {name!r}: unknown keys {sorted(map(str, unknown))}; "
                "use 'type' and 'required'."
            )
        type_name = raw.get("type")
        if "required" in raw:
            if not isinstance(raw["required"], bool):
                raise ValueError(f"{kind.capitalize()} column {name!r}: 'required' must be a bool")
            required = raw["required"]
    else:
        raise ValueError(
            f"{kind.capitalize()} column {name!r}: give a type name or "
            '{"type": ..., "required": ...}'
        )
    type_name = _check_type(name, type_name, kind)
    return ColumnSpec(name=name, type=type_name, required=required, kind=kind)


def _check_type(name: str, type_name: Any, kind: ColumnKind) -> str:
    """Normalise a declared type and check it against the rules for ``kind``."""
    if not isinstance(type_name, str):
        raise ValueError(
            f"{kind.capitalize()} column {name!r}: the type must be a string, got "
            f"{type(type_name).__name__}"
        )
    normalized = " ".join(type_name.split()).lower()
    if kind == "tenant":
        if normalized not in TENANT_COLUMN_TYPES:
            raise ValueError(
                f"Tenant column {name!r}: type {type_name!r} is not supported. Tenant columns "
                f"must be scalar ids: use one of {', '.join(TENANT_COLUMN_TYPES)}."
            )
        return normalized
    if normalized in COLUMN_TYPES:
        return normalized
    if len(normalized) > _MAX_TYPE_LENGTH or not EXTRA_TYPE_RE.match(normalized):
        raise ValueError(
            f"Extra column {name!r}: {type_name!r} is not a valid type name. Give a Postgres "
            "type such as text, jsonb, numeric(10,2), character varying(64), text[] or "
            "myschema.my_enum (no quotes, semicolons or comments)."
        )
    return normalized


def _parse_kind(columns: Any, kind: ColumnKind) -> list[ColumnSpec]:
    if columns is None:
        return []
    if isinstance(columns, Mapping):
        return [_parse_one(name, raw, kind) for name, raw in columns.items()]
    if isinstance(columns, Sequence) and not isinstance(columns, str | bytes):
        return [_parse_one(name, "text", kind) for name in columns]
    raise ValueError(
        f"{kind}_columns must be a mapping of name to type (or a list of names), "
        f"got {type(columns).__name__}"
    )


def parse_columns(
    tenant_columns: Any = None,
    extra_columns: Any = None,
    *,
    reserved: Iterable[str] = (),
    table: str = "table",
) -> ColumnSet:
    """Validate the tenant and extra column declarations of one backend.

    Raises ``ValueError`` for a bad name, a built-in (``reserved``) name, an unknown type, a
    bad ``required`` flag, or a name declared twice.
    """
    reserved_names = frozenset(reserved) | _ALWAYS_RESERVED
    tenant = _parse_kind(tenant_columns, "tenant")
    extra = _parse_kind(extra_columns, "extra")
    seen: set[str] = set()
    for spec in (*tenant, *extra):
        if spec.name in reserved_names:
            raise ValueError(
                f"Column {spec.name!r} is a built-in column of the {table} table; pick "
                "another name."
            )
        if spec.name in seen:
            raise ValueError(f"Column {spec.name!r} is declared twice")
        seen.add(spec.name)
    return ColumnSet(tenant=tuple(tenant), extra=tuple(extra))


def isolation_conditions(
    specs: Sequence[ColumnSpec], values: Mapping[str, Any]
) -> list[tuple[ColumnSpec, Any]]:
    """Pair each tenant column the scope picks with the caller's value.

    Raises ``MissingColumnValueError`` when a picked column has no value: a column is never
    dropped from the filter silently, whatever its ``required`` flag says.
    """
    conditions: list[tuple[ColumnSpec, Any]] = []
    for spec in specs:
        value = values.get(spec.name)
        if value is None:
            raise MissingColumnValueError(
                f"The isolation scope picks tenant column {spec.name!r}, but config['user'] has "
                f"no {spec.name!r}. Set it in your auth class, or leave the column out of the "
                "scope."
            )
        conditions.append((spec, value))
    return conditions


def write_conditions(
    specs: Sequence[ColumnSpec], values: Mapping[str, Any]
) -> list[tuple[ColumnSpec, Any]]:
    """Like :func:`isolation_conditions` for a write: a missing value pairs with ``None``.

    Only used where the write itself stores these values (insert-if-absent ownership checks,
    the upsert guard), which compare with ``IS NOT DISTINCT FROM``: a writer without a value
    for an optional tenant column matches only rows that have none either. Required columns
    were already enforced by ``ColumnSet.resolve(write=True)``.
    """
    return [(spec, values.get(spec.name)) for spec in specs]


def build_isolation_where(
    conditions: Sequence[tuple[ColumnSpec, Any]],
    params: list[Any],
    alias: str | None = None,
) -> str:
    """Render ``conditions`` as ``"col" = $n AND ...``, appending each value to ``params``.

    Returns an empty string for no conditions. ``alias`` must be a plain identifier chosen by
    the caller (never user input); it is checked anyway.
    """
    if alias is not None and not re.match(r"^[a-z_][a-z0-9_]{0,62}$", alias):
        raise ValueError(f"Invalid table alias: {alias!r}")
    prefix = f"{alias}." if alias else ""
    clauses = []
    for spec, value in conditions:
        params.append(value)
        clauses.append(f"{prefix}{spec.quoted} = ${len(params)}")
    return " AND ".join(clauses)


def rls_settings(
    tenant: Sequence[ColumnSpec],
    picked: Sequence[ColumnSpec] | None,
    values: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """The ``(setting, value)`` pairs a call sets with ``set_config(..., true)`` under RLS.

    ``tenxgraph.scope`` lists the picked tenant columns as ``+,col1,col2`` (``+`` alone when
    the scope picks none, which the policy reads as "all rows"); each ``tenxgraph.<col>``
    carries the caller's value, or an empty string. ``picked=None`` means every tenant column.
    """
    chosen = tenant if picked is None else picked
    # A picked column without a value (optional column on a write) is left out of the
    # policy's scope; the backend's WHERE / ownership check still compares it.
    chosen = [spec for spec in chosen if values.get(spec.name) is not None]
    scope = "+" + "".join(f",{spec.name}" for spec in chosen)
    settings = [(RLS_SCOPE_SETTING, scope)]
    chosen_names = {spec.name for spec in chosen}
    for spec in tenant:
        value = values.get(spec.name) if spec.name in chosen_names else None
        settings.append((spec.setting_name, spec.to_setting(value)))
    return settings


def rls_all_rows_settings(tenant: Sequence[ColumnSpec]) -> list[tuple[str, str]]:
    """RLS settings that let one call see every tenant's rows (maintenance and global lookups)."""
    return rls_settings(tenant, (), {})


def rls_policy_expression(tenant: Sequence[ColumnSpec], alias: str | None = None) -> str:
    """The ``USING`` / ``WITH CHECK`` expression of the native RLS policy.

    Per tenant column ``c`` of type ``T``::

        (current_setting('tenxgraph.scope', true) LIKE '+%'
         AND (strpos(current_setting('tenxgraph.scope', true) || ',', ',c,') = 0
              OR c = NULLIF(current_setting('tenxgraph.c', true), '')::T))

    An unset or empty ``tenxgraph.scope`` (a connection that did not go through the backend,
    or a pooled connection after an earlier ``SET LOCAL`` ended) matches no row. A column the
    scope does not list is not restricted. Names come from validated specs and the type from a
    fixed table, so no caller data reaches the policy text.
    """
    prefix = f"{quote_ident(alias)}." if alias else ""
    scope = f"current_setting('{RLS_SCOPE_SETTING}', true)"
    parts = []
    for spec in tenant:
        quote_ident(spec.name)  # validates again; the name is used inside literals below
        parts.append(
            f"(strpos({scope} || ',', ',{spec.name},') = 0 OR {prefix}{spec.quoted} = "
            f"NULLIF(current_setting('{spec.setting_name}', true), '')::{spec.sql_type})"
        )
    guard = f"(COALESCE({scope}, '') LIKE '+%')"
    return " AND ".join([guard, *parts])


def rls_policy_name(tenant: Sequence[ColumnSpec], base: str = "tenxgraph_isolation") -> str:
    """Policy name carrying a hash of its definition, so setup can tell it is current."""
    definition = ";".join(f"{spec.name}:{spec.type}" for spec in tenant)
    digest = hashlib.blake2b(definition.encode(), digest_size=4).hexdigest()
    return f"{base}_{digest}"


# ---------------------------------------------------------------------------
# DDL shared by the backends (all identifiers validated and quoted by the caller or here)
# ---------------------------------------------------------------------------


def column_ddl(table_sql: str, columns: ColumnSet, index_prefix: str) -> list[str]:
    """Idempotent DDL adding each declared column and one B-tree index on it.

    ``table_sql`` is the caller's quoted, schema-qualified table name. Postgres combines the
    single-column indexes (bitmap AND) for any subset a scope picks, so no composite index is
    needed per scope. Only built-in types have a fixed DDL type: a column of any other type
    raises ``ValueError`` here (use :func:`ensure_columns`, which resolves it on the server,
    or pass ``columns.builtin_only()``).
    """
    statements: list[str] = []
    for spec in columns.all:
        statements.append(
            f"ALTER TABLE {table_sql} ADD COLUMN IF NOT EXISTS {spec.quoted} {spec.sql_type}"
        )
        statements.append(
            f"CREATE INDEX IF NOT EXISTS {quote_ident(index_name(index_prefix, spec.name))} "
            f"ON {table_sql} ({spec.quoted})"
        )
    return statements


async def resolve_column_type(conn: Any, spec: ColumnSpec) -> str:
    """The server's canonical name for a non-built-in column type, for DDL and comparisons.

    Runs ``SELECT format_type(to_regtype($1), to_regtypemod($1))`` with the declared type as
    a parameter: ``to_regtype`` returns NULL for an unknown type and ``to_regtypemod`` keeps a
    modifier list (``numeric(10,2)``, ``character varying(64)``), so the result is the same
    text ``format_type(atttypid, atttypmod)`` gives for an existing column. Before
    PostgreSQL 17 (no ``to_regtypemod``) a type with a modifier list is refused and others
    use ``format_type(to_regtype($1), NULL)``.

    Raises:
        ValueError: The server does not know the type (or cannot parse it).
    """
    version = await conn.fetchval("SELECT current_setting('server_version_num')::int")
    has_typmod = int(version or 0) >= _REGTYPEMOD_MIN_SERVER
    if not has_typmod and "(" in spec.type:
        raise ValueError(
            f"Extra column {spec.name!r}: type {spec.type!r} has a modifier list, which needs "
            "PostgreSQL 17 or later (to_regtypemod). Declare the type without it."
        )
    query = (
        "SELECT format_type(to_regtype($1), to_regtypemod($1))"
        if has_typmod
        else "SELECT format_type(to_regtype($1), NULL)"
    )
    try:
        canonical = await conn.fetchval(query, spec.type)
    except Exception as exc:  # a parse error from the server (asyncpg is optional here)
        if not isinstance(getattr(exc, "sqlstate", None), str):
            raise
        raise ValueError(
            f"Extra column {spec.name!r}: the database cannot parse type {spec.type!r}"
        ) from None
    if canonical is None:
        raise ValueError(
            f"Extra column {spec.name!r}: type {spec.type!r} does not exist in the database. "
            "Create it first (CREATE TYPE / CREATE EXTENSION) or fix the declaration."
        )
    canonical = str(canonical)
    if not _CANONICAL_TYPE_RE.match(canonical) or "--" in canonical or "/*" in canonical:
        raise ValueError(
            f"Extra column {spec.name!r}: the database returned an unexpected type name"
        )
    return canonical


async def ensure_columns(conn: Any, table_sql: str, columns: ColumnSet, index_prefix: str) -> None:
    """Add missing declared columns (and their indexes); raise when one exists with another type.

    Reads the catalog first, so an ``ALTER TABLE`` (and its exclusive lock) only runs for a
    column that is really missing. A non-built-in type is resolved on the server
    (:func:`resolve_column_type`) and only that canonical name reaches the DDL; an existing
    column is compared by canonical name, so ``varchar(64)`` matches
    ``character varying(64)``.
    """
    if not columns:
        return
    rows = await conn.fetch(
        "SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS type FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass($1) AND a.attnum > 0 AND NOT a.attisdropped",
        table_sql,
    )
    existing = {str(row["attname"]): str(row["type"]) for row in rows or []}
    for spec in columns.all:
        if spec.builtin:
            ddl_type, canonical = spec.sql_type, FORMAT_TYPE_NAMES[spec.type]
        else:
            ddl_type = canonical = await resolve_column_type(conn, spec)
        found = existing.get(spec.name)
        if found is None:
            await conn.execute(
                f"ALTER TABLE {table_sql} ADD COLUMN IF NOT EXISTS {spec.quoted} {ddl_type}"
            )
        elif found != canonical:
            raise ValueError(
                f"Column {spec.name!r} already exists on {table_sql} as {found}, but it is "
                f"declared as {canonical}. Change the declaration or migrate the column."
            )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {quote_ident(index_name(index_prefix, spec.name))} "
            f"ON {table_sql} ({spec.quoted})"
        )


@contextmanager
def column_value_errors(params: Mapping[int, ColumnSpec]) -> Iterator[None]:
    """Turn asyncpg's "cannot encode this parameter" error into :class:`ColumnValueError`.

    ``params`` maps 1-based bind parameter numbers to the declared column they carry. When
    asyncpg's codec rejects one of them (a str for a ``date`` column, a non-list for
    ``text[]``), the error names the column and its type but never the value (asyncpg's own
    message quotes it). Any other error propagates unchanged.
    """
    try:
        yield
    except Exception as exc:
        match = _ARGUMENT_ERROR_RE.search(str(exc))
        spec = params.get(int(match.group(1))) if match else None
        if spec is None:
            raise
        raise ColumnValueError(
            f'Column {spec.name!r} is {spec.type}: the value from config["user"] has a type '
            "the database driver cannot send for it"
        ) from None


_POLICY_PREFIX = "tenxgraph_isolation"


async def ensure_rls(
    conn: Any,
    *,
    table_sql: str,
    schema: str,
    table: str,
    enabled: bool,
    policy: str | None = None,
    expression: str | None = None,
) -> None:
    """Create, refresh or remove the native RLS policy this library manages on one table.

    Enabled: the policy ``policy`` (its name carries a hash of its definition) is created when
    missing, older ``tenxgraph_isolation_*`` policies are dropped, and the table gets
    ``ENABLE`` + ``FORCE ROW LEVEL SECURITY`` (``FORCE`` so the table owner, usually the app
    role, is filtered too). Disabled: if a policy of ours exists it is dropped and RLS is turned
    off again, so switching the option off does not leave every query denied. Policies the
    library did not create are never touched.
    """
    rows = await conn.fetch(
        "SELECT policyname FROM pg_policies WHERE schemaname = $1 AND tablename = $2 "
        "AND policyname LIKE $3",
        schema,
        table,
        _POLICY_PREFIX + r"\_%",
    )
    ours = [str(row["policyname"]) for row in rows or []]
    if not enabled:
        if not ours:
            return
        for name in ours:
            await conn.execute(f"DROP POLICY IF EXISTS {quote_ident(name)} ON {table_sql}")
        await conn.execute(f"ALTER TABLE {table_sql} NO FORCE ROW LEVEL SECURITY")
        await conn.execute(f"ALTER TABLE {table_sql} DISABLE ROW LEVEL SECURITY")
        return
    if policy is None or expression is None:
        raise ValueError("ensure_rls needs a policy name and expression when enabled")
    if policy not in ours:
        for name in ours:
            await conn.execute(f"DROP POLICY IF EXISTS {quote_ident(name)} ON {table_sql}")
        await conn.execute(
            f"CREATE POLICY {quote_ident(policy)} ON {table_sql} "
            f"USING ({expression}) WITH CHECK ({expression})"
        )
    flags = await conn.fetchrow(
        "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = to_regclass($1)",
        table_sql,
    )
    if not flags or not flags["relrowsecurity"]:
        await conn.execute(f"ALTER TABLE {table_sql} ENABLE ROW LEVEL SECURITY")
    if not flags or not flags["relforcerowsecurity"]:
        await conn.execute(f"ALTER TABLE {table_sql} FORCE ROW LEVEL SECURITY")


async def warn_if_rls_bypassed(conn: Any, logger: Any, owner: str) -> None:
    """Log a warning when the connected role skips RLS (superuser or ``BYPASSRLS``)."""
    bypass = await conn.fetchval(
        "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
    )
    if bypass:
        logger.warning(
            "%s: row_level_security=True, but the database role is a superuser or has "
            "BYPASSRLS, so Postgres skips the policies for it. The WHERE filter still applies; "
            "connect as a normal role for RLS to act as a second line of defence.",
            owner,
        )


async def apply_settings(conn: Any, settings: Sequence[tuple[str, str]]) -> None:
    """``set_config(name, value, true)`` for each pair, in one statement, all as parameters.

    ``true`` makes each setting local to the current transaction, so it ends with it and never
    leaks to the next user of a pooled connection.
    """
    if not settings:
        return
    calls = ", ".join(f"set_config(${2 * i + 1}, ${2 * i + 2}, true)" for i in range(len(settings)))
    args: list[str] = []
    for name, value in settings:
        args.extend((name, value))
    await conn.execute(f"SELECT {calls}", *args)
