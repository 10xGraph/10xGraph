"""The list form of the isolation scope (``config["authz"]["scope"]``) and its resolver."""

import pytest

from tenxgraph.core.authz import (
    SCOPE_NONE,
    SCOPE_OWNER,
    build_authz,
    isolation_columns,
    isolation_scope,
)
from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer import InMemoryCheckpointer, SqliteCheckpointer
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.utils import ThreadInfo


def _cfg(scope):
    return {"user": {"user_id": "u1", "authz": build_authz("u1", scope=scope)}}


class TestResolver:
    def test_shortcuts(self):
        assert isolation_columns(_cfg(SCOPE_OWNER), ("org_id", "project_id")) == (
            "user_id",
            "org_id",
            "project_id",
        )
        assert isolation_columns(_cfg(SCOPE_OWNER)) == ("user_id",)
        assert isolation_columns(_cfg(SCOPE_NONE), ("org_id",)) == ()

    def test_no_policy_is_none(self):
        assert isolation_columns({}, ("org_id",)) is None
        assert isolation_columns(None) is None
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            isolation_columns({"authz": {"scope": "weird"}}, ("org_id",))

    def test_list_any_subset_keeps_order_and_drops_duplicates(self):
        tenants = ("org_id", "project_id")
        assert isolation_columns(_cfg(["project_id"]), tenants) == ("project_id",)
        assert isolation_columns(_cfg(["project_id", "user_id", "project_id"]), tenants) == (
            "project_id",
            "user_id",
        )
        assert isolation_columns(_cfg([]), tenants) == ()

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError, match="does not declare"):
            isolation_columns(_cfg(["user_id", "region"]), ("org_id",))
        with pytest.raises(ValueError, match="does not declare"):
            isolation_columns(_cfg(["org_id"]))  # backend declares no tenant columns

    def test_non_string_entries_raise(self):
        with pytest.raises(ValueError, match="column names"):
            isolation_columns({"authz": {"scope": ["user_id", 3]}}, ())

    def test_isolation_scope_returns_list_copy(self):
        cfg = _cfg(["user_id", "org_id"])
        got = isolation_scope(cfg)
        assert got == ["user_id", "org_id"]
        got.append("x")
        assert isolation_scope(cfg) == ["user_id", "org_id"]
        assert isolation_scope(_cfg(SCOPE_OWNER)) == "owner"

    def test_trusted_user_block_wins_for_lists_too(self):
        cfg = {
            "authz": build_authz("u1", scope=[]),  # forged top-level: no isolation
            "user": {"user_id": "u1", "authz": build_authz("u1", scope=["user_id", "org_id"])},
        }
        assert isolation_columns(cfg, ("org_id",)) == ("user_id", "org_id")


class TestBuildAuthz:
    def test_list_scope(self):
        block = build_authz("u1", scope=("user_id", "org_id", "org_id"))
        assert block["scope"] == ["user_id", "org_id"]

    @pytest.mark.parametrize("bad", [["org id"], ["Org"], [1], ['x"; drop']])
    def test_bad_list_raises_instead_of_falling_open(self, bad):
        with pytest.raises(ValueError, match="Invalid isolation scope column"):
            build_authz("u1", scope=bad)

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            build_authz("u1", scope="bogus")


def _min_store(tenants=()):
    class _S(BaseStore):
        tenant_column_names = tenants

        async def aping(self): ...
        async def asetup(self): ...
        async def astore(self, *a, **k): ...
        async def asearch(self, *a, **k): ...
        async def aget(self, *a, **k): ...
        async def aget_all(self, *a, **k): ...
        async def aupdate(self, *a, **k): ...
        async def adelete(self, *a, **k): ...
        async def aforget_memory(self, *a, **k): ...

    return _S()


class TestStoreHelpers:
    def test_user_filter_follows_list(self):
        store = _min_store(("org_id",))
        assert store._scope_user_id(_cfg(["user_id", "org_id"]), "A") == "A"
        assert store._scope_user_id(_cfg(["org_id"]), "A") is None
        assert store._require_user_id({**_cfg(["org_id"]), "user_id": None}) is None
        with pytest.raises(ValueError):
            store._require_user_id({**_cfg(["user_id"]), "user_id": None})

    def test_store_without_tenant_columns_rejects_tenant_scope(self):
        store = _min_store()
        with pytest.raises(ValueError, match="does not declare"):
            store._scope_user_id(_cfg(["org_id"]), "A")
        assert store._scope_user_id(_cfg(["user_id"]), "A") == "A"
        assert store._scope_user_id(_cfg([]), "A") is None


class TestCheckpointerHelpers:
    def test_isolation_enabled_follows_list(self):
        cp = InMemoryCheckpointer()
        assert cp._isolation_enabled(_cfg(["user_id"]), "u1") is True
        assert cp._isolation_enabled(_cfg([]), "u1") is False
        assert cp._isolation_enabled(_cfg(SCOPE_OWNER), "u1") is True
        assert cp._isolation_enabled(_cfg(["user_id"]), None) is False
        with pytest.raises(ValueError, match="does not declare"):
            cp._isolation_enabled(_cfg(["org_id"]), "u1")

    @pytest.mark.asyncio
    async def test_in_memory_list_scope_isolates(self):
        cp = InMemoryCheckpointer()
        owner = {"thread_id": "t1", "user_id": "alice", **_cfg(["user_id"])}
        other = {"thread_id": "t1", "user_id": "bob", **_cfg(["user_id"])}
        await cp.aput_state(owner, AgentState())
        assert await cp.aget_state(other) is None
        admin = {"thread_id": "t1", "user_id": "bob", **_cfg([])}
        assert await cp.aget_state(admin) is not None

    @pytest.mark.asyncio
    async def test_default_aget_thread_scope(self):
        cp = InMemoryCheckpointer()
        cfg = {"thread_id": "t1", "user_id": "alice"}
        await cp.aput_thread(cfg, ThreadInfo(thread_id="t1", user_id="alice"))
        assert await cp.aget_thread_scope("t1") == {"user_id": "alice"}
        assert await cp.aget_thread_scope("missing") is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tenant_columns": ["org_id"]},
            {"extra_columns": {"x": "text"}},
            {"row_level_security": True},
        ],
    )
    def test_non_postgres_checkpointers_refuse_tenant_options(self, kwargs):
        with pytest.raises(ValueError, match="need PgCheckpointer"):
            InMemoryCheckpointer(**kwargs)
        with pytest.raises(ValueError, match="need PgCheckpointer"):
            SqliteCheckpointer(":memory:", **kwargs)
