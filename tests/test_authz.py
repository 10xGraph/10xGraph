"""Tests for Identity, Reach and the owner_filter helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tenxgraph.core.authz import (
    Identity,
    Reach,
    can,
    filter_columns,
    get_identity,
    owner_filter,
)


def _cfg(**kw):
    return {"user": Identity(user_id="u1", **kw)}


def test_dict_and_model_input():
    ident = Identity(user_id="u1", reach="tenant", tenant={"org_id": "a"})
    assert get_identity({"user": ident}) is ident
    from_dict = get_identity(
        {"user": {"user_id": "u1", "reach": "tenant", "tenant": {"org_id": "a"}}}
    )
    assert from_dict == ident
    assert get_identity({}) is None
    assert get_identity({"user": None}) is None
    assert get_identity(None) is None


def test_extra_and_unknown_reach_refused():
    with pytest.raises(ValidationError):
        Identity(user_id="u1", email="x")
    with pytest.raises(ValidationError):
        Identity(user_id="u1", reach="everyone")
    with pytest.raises(ValidationError):
        get_identity({"user": {"user_id": "u1", "reach": "bogus"}})
    with pytest.raises(ValidationError):
        get_identity({"user": {"email": "x"}})


def test_frozen():
    ident = Identity(user_id="u1")
    with pytest.raises(ValidationError):
        ident.user_id = "u2"


def test_defaults():
    ident = Identity(user_id="u1")
    assert ident.reach is Reach.OWN
    assert ident.kind == "user"
    assert ident.permissions is None


def test_wildcards():
    ident = Identity(user_id="u1", permissions=frozenset({"graph:*", "threads:read"}))
    assert ident.can("graph:invoke")
    assert ident.can("graph:stop")
    assert ident.can("threads:read")
    assert not ident.can("threads:write")
    assert not ident.can("store:read")
    assert Identity(user_id="u1", permissions=frozenset({"*"})).can("admin:write")
    assert not Identity(user_id="u1", permissions=frozenset()).can("graph:read")
    assert Identity(user_id="u1").can("anything:goes")


def test_can_helper_unrestricted_without_identity():
    assert can({}, "admin:write")
    assert not can(_cfg(permissions=frozenset({"graph:*"})), "admin:write")
    assert can(_cfg(permissions=frozenset({"graph:*"})), "graph:invoke")


def test_mapping_access_merges_claims():
    ident = Identity(
        user_id="u1",
        tenant={"org_id": "acme"},
        claims={"email": "a@b.c", "user_id": "spoof", "org_id": "spoof"},
    )
    assert ident["user_id"] == "u1"
    assert ident.get("org_id") == "acme"
    assert ident.get("email") == "a@b.c"
    assert ident.get("missing") is None
    assert ident.get("missing", 5) == 5
    assert "email" in ident and "nope" not in ident
    with pytest.raises(KeyError):
        ident["nope"]
    assert dict(ident) == {"email": "a@b.c", "user_id": "u1", "org_id": "acme"}


def test_owner_filter_none_without_identity():
    assert owner_filter({}, ("org_id",)) is None
    assert filter_columns({}, ("org_id",)) is None


def test_owner_filter_own():
    cfg = _cfg(tenant={"org_id": "a", "extra": "x"})
    assert owner_filter(cfg) == {"user_id": "u1"}
    assert owner_filter(cfg, ("org_id",)) == {"user_id": "u1", "org_id": "a"}
    assert filter_columns(cfg, ("org_id",)) == ("user_id", "org_id")


def test_owner_filter_tenant():
    cfg = _cfg(reach="tenant", tenant={"org_id": "a"})
    assert owner_filter(cfg, ("org_id",)) == {"org_id": "a"}
    assert filter_columns(cfg, ("org_id",)) == ("org_id",)
    with pytest.raises(ValueError, match="tenant columns"):
        owner_filter(cfg, ())


def test_owner_filter_all():
    cfg = _cfg(reach="all")
    assert owner_filter(cfg) == {}
    assert owner_filter(cfg, ("org_id",)) == {}
    assert filter_columns(cfg) == ()


@pytest.mark.parametrize("reach", ["own", "tenant"])
def test_missing_tenant_value_raises(reach):
    cfg = _cfg(reach=reach, tenant={"other": "x"})
    with pytest.raises(ValueError, match="org_id"):
        owner_filter(cfg, ("org_id",))


@pytest.mark.parametrize("reach", ["own", "tenant", "all"])
def test_system_owner_pinned_on_every_reach(reach):
    cfg = _cfg(reach=reach, tenant={"org_id": "a"})
    assert owner_filter(cfg, ("org_id",), owner="__system__:kb") == {
        "user_id": "__system__:kb",
        "org_id": "a",
    }
    assert owner_filter(cfg, owner="__system__:kb") == {"user_id": "__system__:kb"}
    assert owner_filter({}, owner="__system__:kb") is None
