"""Authorization contract carried inside ``config["authz"]``.

This is the single source of truth for authorization in Agentflow. The **contract** (what
the block looks like) and the **enforcement primitives** (``should_isolate`` / ``has_scope``)
live here in the core library, so every consumer -- checkpointer, store, graph nodes, tools --
enforces identically whether the call came from the API or from a direct SDK call.

Producers of the block are trusted and layer-specific:

- **API**: builds it from a verified identity + authorization backend and overwrites any
  client-supplied value (non-hijackable).
- **SDK** (direct ``CompiledGraph.invoke``): the developer builds it (their process is the
  trust boundary), or passes nothing and gets the permissive default below.

Shape::

    config["authz"] = {
        "user_id": "u1",
        "scope": "owner" | "none" | ["user_id", "org_id", ...],  # data isolation policy
        "scopes": ["checkpointer:read", "graph:invoke", ...],
    }

``scope`` lists the columns every storage query is filtered by: ``user_id`` and any subset of
the backend's declared tenant columns (``tenant_columns=`` on ``PgCheckpointer`` /
``PgVectorStore``). ``"owner"`` is shorthand for ``["user_id", *tenant_columns]`` and
``"none"`` for ``[]``. A list naming a column the backend does not declare raises. Leaving
``user_id`` out (for example ``["org_id"]`` for an org admin) means rows of every user in that
org match, so the picked columns are what identifies a row's owner for that caller.

When ``config["authz"]`` is absent the defaults are permissive (isolation falls back to the
storage backend's own setting; every scope is allowed) -- so existing direct-SDK usage is
unchanged. A ``scope`` that is present but not understood (an unknown string such as
``"owners"``, or a value that is neither a string nor a list) always raises ``ValueError``: a
typo must never quietly turn into "no isolation" or into the backend default.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Any


AUTHZ_KEY = "authz"

# --- Scopes (resource:action). The full, canonical set. ---------------------------------

# graph (execution) -- gated at the API layer; the checkpointer/store are the data backstop.
GRAPH_INVOKE = "graph:invoke"
GRAPH_STREAM = "graph:stream"
GRAPH_STOP = "graph:stop"
GRAPH_FIX = "graph:fix"
# Kept so existing imports work; the setup endpoint is gone and no scope check uses it.
GRAPH_SETUP = "graph:setup"
GRAPH_READ = "graph:read"

# checkpointer (thread state / messages / threads)
CHECKPOINT_READ = "checkpointer:read"
CHECKPOINT_WRITE = "checkpointer:write"
CHECKPOINT_DELETE = "checkpointer:delete"
CHECKPOINT_IMPORT = "checkpointer:import"

# memory / long-term store (the API's /v1/store routes)
STORE_READ = "store:read"
STORE_WRITE = "store:write"
STORE_DELETE = "store:delete"
# Deprecated aliases: these were once "memory:*", a name no route ever checked.
MEMORY_READ = STORE_READ
MEMORY_WRITE = STORE_WRITE
MEMORY_DELETE = STORE_DELETE

# files / media
FILES_READ = "files:read"
FILES_UPLOAD = "files:upload"

# misc
CONFIG_READ = "config:read"

ALL_SCOPES = frozenset(
    {
        GRAPH_INVOKE,
        GRAPH_STREAM,
        GRAPH_STOP,
        GRAPH_FIX,
        GRAPH_READ,
        CHECKPOINT_READ,
        CHECKPOINT_WRITE,
        CHECKPOINT_DELETE,
        CHECKPOINT_IMPORT,
        STORE_READ,
        STORE_WRITE,
        STORE_DELETE,
        FILES_READ,
        FILES_UPLOAD,
        CONFIG_READ,
    }
)


class Isolation(StrEnum):
    """The two isolation shortcuts. Members compare equal to their strings."""

    #: ``["user_id", *tenant_columns]``: only the caller's own rows.
    OWNER = "owner"
    #: ``[]``: storage is not scoped (share data on purpose).
    NONE = "none"


# Valid isolation scopes (aliases of the ``Isolation`` members, kept for existing imports).
SCOPE_OWNER = Isolation.OWNER
SCOPE_NONE = Isolation.NONE

_VALID_SHORTCUTS = ", ".join(repr(m.value) for m in Isolation)


def _unknown_scope(scope: Any) -> ValueError:
    return ValueError(
        f"Unknown isolation scope {scope!r}: use {_VALID_SHORTCUTS} (or tenxgraph.core.authz."
        "Isolation) or a list of column names"
    )


def _shortcut(scope: Any) -> Isolation | None:
    """The ``Isolation`` member for an exact shortcut string or member, else None."""
    if isinstance(scope, str):
        try:
            return Isolation(scope)
        except ValueError:
            return None
    return None


def get_authz(config: Any) -> dict[str, Any] | None:
    """Return the ``authz`` block from config, or None if absent/malformed.

    Looks in two places, in this order:

    1. ``config["user"]["authz"]`` -- where the API stamps it. The trusted ``user`` object is
       copied into ``config["user"]`` on every request, so a single stamp there reaches every
       downstream call. It always wins.
    2. a top-level ``config["authz"]`` -- for direct SDK calls with no API in front. Request
       config can carry client data, so this is only a fallback: a client-supplied top-level
       block can never override the policy the API stamped.
    """
    if not isinstance(config, dict):
        return None
    user = config.get("user")
    if isinstance(user, dict):
        block = user.get(AUTHZ_KEY)
        if isinstance(block, dict):
            return block
    block = config.get(AUTHZ_KEY)
    if isinstance(block, dict):
        return block
    return None


_SCOPE_COLUMN_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def isolation_scope(config: Any) -> str | list[str] | None:
    """Return the isolation scope: ``"owner"``, ``"none"``, a column list, or None when unset.

    None means "no policy present" (no authz block, or a block without a ``scope`` key) --
    callers apply their own default (a storage backend falls back to its
    ``enforce_user_isolation`` setting). A list is returned as given (a copy);
    :func:`isolation_columns` validates it against a backend's columns.

    Raises:
        ValueError: ``scope`` is present but is an unknown string or neither a string nor a
            list. It never falls back to the backend default.
    """
    block = get_authz(config)
    if block is None or "scope" not in block:
        return None
    scope = block["scope"]
    if isinstance(scope, list | tuple):
        return list(scope)
    member = _shortcut(scope)
    if member is None:
        raise _unknown_scope(scope)
    return member.value


def isolation_columns(config: Any, tenant_columns: Sequence[str] = ()) -> tuple[str, ...] | None:
    """Resolve the isolation scope into the columns a backend filters by.

    ``tenant_columns`` are the backend's declared tenant column names. ``"owner"`` becomes
    ``("user_id", *tenant_columns)``, ``"none"`` becomes ``()``, and a list is checked: every
    entry must be ``"user_id"`` or one of ``tenant_columns`` (duplicates are dropped, order
    kept). Returns None when no policy is present, so the backend applies its default.

    Raises:
        ValueError: The list names a column the backend does not declare, or is not a list
            of strings. A storage call never runs with a scope it cannot apply.
    """
    scope = isolation_scope(config)
    if scope is None:
        return None
    if scope == SCOPE_OWNER:
        return ("user_id", *tenant_columns)
    if scope == SCOPE_NONE:
        return ()
    allowed = ("user_id", *tenant_columns)
    picked: list[str] = []
    for column in scope:  # type: ignore[union-attr]
        if not isinstance(column, str):
            raise ValueError(f"isolation scope entries must be column names, got {column!r}")
        if column not in allowed:
            raise ValueError(
                f"isolation scope names column {column!r}, which this storage backend does not "
                f"declare (allowed: {', '.join(allowed)})"
            )
        if column not in picked:
            picked.append(column)
    return tuple(picked)


def has_scope(config: Any, scope: str) -> bool:
    """Whether the request carries ``scope``. Permissive when no authz block is present."""
    block = get_authz(config)
    if block is None:
        return True  # no policy -> allow (single-user / direct SDK)
    scopes = block.get("scopes")
    return isinstance(scopes, list | tuple | set | frozenset) and scope in scopes


def build_authz(
    user_id: str | int | None,
    *,
    scope: str | Sequence[str] = SCOPE_NONE,
    scopes: Any = (),
) -> dict[str, Any]:
    """Build a trusted authz block to place at ``config["authz"]``.

    ``scope`` is ``"owner"``, ``"none"`` (or an :class:`Isolation` member) or a list of
    column names (``user_id`` and tenant columns). The block stores the plain string.

    Raises:
        ValueError: ``scope`` is an unknown string (such as ``"owners"``), not a string or a
            list, or a list holding an invalid column name. A malformed policy never turns
            into "no isolation".
    """
    if isinstance(scope, list | tuple):
        columns: list[str] = []
        for column in scope:
            if not isinstance(column, str) or not _SCOPE_COLUMN_RE.match(column):
                raise ValueError(f"Invalid isolation scope column: {column!r}")
            if column not in columns:
                columns.append(column)
        return {"user_id": user_id, "scope": columns, "scopes": list(scopes)}
    member = _shortcut(scope)
    if member is None:
        raise _unknown_scope(scope)
    return {"user_id": user_id, "scope": member.value, "scopes": list(scopes)}
