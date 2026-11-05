"""An isolation scope that is present but not understood raises everywhere; never "none".

A typo such as ``"owners"`` used to fall back to ``"none"`` (``build_authz``) or to the
storage backend's default (``isolation_scope``). Both are silent and dangerous, so every
unknown value now raises ``ValueError``. Only a missing authz block or a block without a
``scope`` key means "no policy".
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tenxgraph.core.authz import (
    SCOPE_NONE,
    SCOPE_OWNER,
    Isolation,
    build_authz,
    isolation_columns,
    isolation_scope,
)
from tenxgraph.storage.checkpointer import InMemoryCheckpointer, SqliteCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.pgvector_store import PgVectorStore
from tests.store.test_pgvector_store import FakeConn, FakePool, _Embedding


BAD_STRINGS = ["owners", "OWNER", "None", "", " owner", "all"]
BAD_VALUES = [None, 5, True, {"scope": "owner"}, b"owner", 1.5]


class TestIsolationEnum:
    def test_members_are_the_strings(self):
        assert Isolation.OWNER == "owner"
        assert Isolation.NONE == "none"
        assert Isolation("owner") is Isolation.OWNER
        assert [m.value for m in Isolation] == ["owner", "none"]

    def test_aliases_kept(self):
        assert SCOPE_OWNER is Isolation.OWNER
        assert SCOPE_NONE is Isolation.NONE
        assert SCOPE_OWNER == "owner"

    def test_build_authz_accepts_members_and_stores_plain_strings(self):
        for member in Isolation:
            block = build_authz("u1", scope=member)
            assert block["scope"] == member.value
            assert type(block["scope"]) is str
        assert json.dumps(build_authz("u1", scope=Isolation.OWNER)["scope"]) == '"owner"'

    def test_isolation_scope_reads_members(self):
        assert isolation_scope({"authz": {"scope": Isolation.NONE}}) == "none"
        assert isolation_columns({"authz": {"scope": Isolation.OWNER}}, ("org_id",)) == (
            "user_id",
            "org_id",
        )


class TestBuildAuthz:
    @pytest.mark.parametrize("bad", BAD_STRINGS)
    def test_unknown_string_raises(self, bad):
        with pytest.raises(ValueError, match="Unknown isolation scope") as info:
            build_authz("u1", scope=bad)
        assert repr(bad) in str(info.value)
        assert "'owner', 'none'" in str(info.value)

    @pytest.mark.parametrize("bad", BAD_VALUES)
    def test_non_string_non_list_raises(self, bad):
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            build_authz("u1", scope=bad)

    def test_exact_strings_and_lists_still_work(self):
        assert build_authz("u1", scope="owner")["scope"] == "owner"
        assert build_authz("u1", scope="none")["scope"] == "none"
        assert build_authz("u1", scope=["user_id", "org_id"])["scope"] == ["user_id", "org_id"]
        assert build_authz("u1")["scope"] == "none"  # the documented default


class TestIsolationScope:
    def test_no_policy_is_none(self):
        assert isolation_scope(None) is None
        assert isolation_scope({}) is None
        assert isolation_scope({"authz": {"user_id": "u1", "scopes": []}}) is None
        assert isolation_scope({"user": {"authz": {"user_id": "u1"}}}) is None

    @pytest.mark.parametrize("bad", BAD_STRINGS + BAD_VALUES)
    def test_unknown_value_raises(self, bad):
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            isolation_scope({"authz": {"scope": bad}})

    @pytest.mark.parametrize("bad", ["owners", None, 7])
    def test_unknown_value_in_trusted_user_block_raises(self, bad):
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            isolation_scope({"user": {"authz": {"scope": bad}}})

    @pytest.mark.parametrize("bad", ["owners", None])
    def test_isolation_columns_raises(self, bad):
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            isolation_columns({"authz": {"scope": bad}}, ("org_id",))


def _bad(scope="owners"):
    return {
        "thread_id": "t1",
        "user_id": "u1",
        "user": {"user_id": "u1", "org_id": "A", "authz": {"user_id": "u1", "scope": scope}},
    }


class _Store(BaseStore):
    async def aping(self): ...
    async def asetup(self): ...
    async def astore(self, *a, **k): ...
    async def asearch(self, *a, **k): ...
    async def aget(self, *a, **k): ...
    async def aget_all(self, *a, **k): ...
    async def aupdate(self, *a, **k): ...
    async def adelete(self, *a, **k): ...
    async def aforget_memory(self, *a, **k): ...


class TestStoragePaths:
    """Every storage backend refuses an unknown scope instead of applying its default."""

    def test_base_checkpointer(self):
        cp = InMemoryCheckpointer()
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            cp._isolation_enabled(_bad(), "u1")

    @pytest.mark.asyncio
    async def test_in_memory_checkpointer_read(self):
        cp = InMemoryCheckpointer()
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            await cp.alist_threads(_bad())

    def test_sqlite_checkpointer(self, tmp_path):
        cp = SqliteCheckpointer(str(tmp_path / "x.db"))
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            cp._isolation_enabled(_bad(None), "u1")

    def test_pg_checkpointer(self):
        cp = PgCheckpointer(postgres_dsn="postgresql://x/y", redis=MagicMock())
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            cp._isolation(_bad())
        tenant = PgCheckpointer(
            postgres_dsn="postgresql://x/y", redis=MagicMock(), tenant_columns=["org_id"]
        )
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            tenant._isolation(_bad(5))

    def test_base_store(self):
        store = _Store()
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            store._require_user_id(_bad())
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            store._scope_user_id(_bad(), "u1")
        # A missing user_id with a bad scope still raises on the scope, never "no filter".
        cfg = _bad("nobody")
        cfg["user_id"] = None
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            store._require_user_id(cfg)

    @pytest.mark.asyncio
    async def test_pgvector_store(self):
        conn = FakeConn()
        store = PgVectorStore(_Embedding(), pool=FakePool(conn), tenant_columns=["org_id"])
        store._setup_done = True
        store._index_metric = store.distance_metric
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            await store.aget_all(_bad())
        with pytest.raises(ValueError, match="Unknown isolation scope"):
            await store.asearch(_bad("everyone"), "q")
        assert not conn.calls  # nothing reached the database
