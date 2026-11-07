"""Caller identity carried in ``config["user"]``, and the helpers storage uses to scope data.

``config["user"]`` holds an :class:`Identity` or a dict that validates into one (direct SDK
callers keep passing a dict). The API builds it from a verified token; a developer's own code
is the trust boundary for direct calls. With no identity there are no checks and each storage
backend applies its own default.

Reach says which rows a caller may see:

- ``own``: rows of this ``user_id`` (and this caller's tenant values).
- ``tenant``: rows of every user with the same tenant values.
- ``all``: every row.

A missing tenant value, a tenant reach on a backend without tenant columns, or an unknown reach
raises. Nothing ever widens silently.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Reach(StrEnum):
    """Which rows of a storage backend a caller can see."""

    OWN = "own"
    TENANT = "tenant"
    ALL = "all"


class Identity(BaseModel):
    """Who is calling and what they may do. Immutable once built.

    ``permissions`` is ``resource:action`` strings; ``*`` allows everything and ``resource:*``
    every action of that resource. ``None`` means unrestricted (direct SDK, auth off).

    Dict readers keep working: ``identity.get(key)``, ``identity[key]`` and ``key in identity``
    read the merged view ``{**claims, **tenant, "user_id": user_id}``, so ``user_id`` and
    tenant values win over raw claims.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str
    kind: Literal["user", "system"] = "user"
    tenant: dict[str, str] = {}
    roles: frozenset[str] = frozenset()
    permissions: frozenset[str] | None = None
    reach: Reach = Reach.OWN
    claims: dict[str, Any] = {}

    def can(self, permission: str) -> bool:
        """Whether ``permission`` (``resource:action``) is allowed."""
        if self.permissions is None or "*" in self.permissions:
            return True
        resource = permission.split(":", 1)[0]
        return permission in self.permissions or f"{resource}:*" in self.permissions

    def _view(self) -> dict[str, Any]:
        return {**self.claims, **self.tenant, "user_id": self.user_id}

    def get(self, key: str, default: Any = None) -> Any:
        return self._view().get(key, default)

    def keys(self) -> Iterator[str]:
        """Keys of the merged view (lets ``dict(identity)`` and ``{**identity}`` work)."""
        return iter(self._view())

    def __getitem__(self, key: str) -> Any:
        return self._view()[key]

    def __contains__(self, key: object) -> bool:
        return key in self._view()


def get_identity(config: Any) -> Identity | None:
    """The identity in ``config["user"]``, or None when absent.

    A dict is validated into an :class:`Identity`; a malformed one raises, it never becomes
    None (which would mean "no checks").
    """
    user = config.get("user") if isinstance(config, dict) else None
    if user is None:
        return None
    if isinstance(user, Identity):
        return user
    return Identity.model_validate(user)


def owner_id(config: Any) -> Any:
    """The storage owner key: ``Identity.user_id`` when there is an identity, else
    ``config["user_id"]`` (direct SDK calls). The identity wins when both are set."""
    identity = get_identity(config)
    if identity is not None:
        return identity.user_id
    return config.get("user_id") if isinstance(config, dict) else None


def can(config: Any, permission: str) -> bool:
    """Whether the caller may do ``permission``. True when there is no identity."""
    identity = get_identity(config)
    return True if identity is None else identity.can(permission)


def owner_filter(
    config: Any,
    tenant_columns: Sequence[str] = (),
    *,
    owner: str | None = None,
) -> dict[str, str] | None:
    """Column filters a storage query must apply for this caller.

    Returns None when there is no identity (the backend applies its own default). Otherwise:
    reach ``own`` gives ``{"user_id": ..., **tenant values}``, ``tenant`` gives the tenant
    values only, ``all`` gives ``{}`` (no tenant value needed).

    ``owner`` targets shared data owned by a reserved ``__system__:<name>`` id (agent memory,
    RAG knowledge bases). It pins ``user_id`` to that owner on every reach, so a wider reach
    never reads other users' rows through a shared-data query.

    Raises:
        ValueError: a tenant column has no value in ``Identity.tenant``, or reach is
            ``tenant`` and the backend has no tenant columns.
    """
    identity = get_identity(config)
    if identity is None:
        return None
    if identity.reach is Reach.ALL and owner is None:
        return {}
    tenant = _tenant_values(identity, tenant_columns)
    if owner is not None:
        return {"user_id": owner, **tenant}
    if identity.reach is Reach.TENANT:
        if not tenant_columns:
            raise ValueError(
                "reach 'tenant' needs a backend with tenant columns; this backend declares none"
            )
        return tenant
    return {"user_id": identity.user_id, **tenant}


def filter_columns(
    config: Any,
    tenant_columns: Sequence[str] = (),
    *,
    owner: str | None = None,
) -> tuple[str, ...] | None:
    """Column names of :func:`owner_filter`, or None when there is no identity."""
    filters = owner_filter(config, tenant_columns, owner=owner)
    return None if filters is None else tuple(filters)


def _tenant_values(identity: Identity, tenant_columns: Sequence[str]) -> dict[str, str]:
    missing = [c for c in tenant_columns if c not in identity.tenant]
    if missing:
        raise ValueError(f"identity has no tenant value for column(s): {', '.join(missing)}")
    return {c: identity.tenant[c] for c in tenant_columns}
