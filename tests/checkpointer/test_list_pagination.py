"""Paging rules shared by the checkpointer list methods.

``alist_messages`` / ``alist_threads`` reject negative ``limit`` / ``offset``, lower an explicit
``limit`` to ``MAX_LIST_LIMIT``, and return the exact ``offset`` + ``limit`` window.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from tenxgraph.core.authz import build_authz
from tenxgraph.core.state import Message
from tenxgraph.storage.checkpointer import base_checkpointer
from tenxgraph.storage.checkpointer.base_checkpointer import MAX_LIST_LIMIT, validate_page
from tenxgraph.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer
from tenxgraph.utils.thread_info import ThreadInfo


def _msg(i: int, content: str | None = None) -> Message:
    return Message.text_message(content=content or f"msg {i}", role="user", message_id=f"m{i}")


def test_validate_page():
    assert validate_page(None, None) == (0, None)
    assert validate_page(0, 0) == (0, None)
    assert validate_page(3, 5) == (3, 5)
    assert validate_page(0, MAX_LIST_LIMIT + 1) == (0, MAX_LIST_LIMIT)
    with pytest.raises(ValueError, match="limit"):
        validate_page(0, -1)
    with pytest.raises(ValueError, match="offset"):
        validate_page(-1, 5)


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def cp(request):
    c = InMemoryCheckpointer() if request.param == "memory" else SqliteCheckpointer(":memory:")
    yield c
    await c.arelease()


@pytest.mark.asyncio
async def test_negative_values_rejected(cp):
    cfg = {"thread_id": "t"}
    for kwargs in ({"limit": -1}, {"offset": -1}):
        with pytest.raises(ValueError):
            await cp.alist_messages(cfg, **kwargs)
        with pytest.raises(ValueError):
            await cp.alist_threads(cfg, **kwargs)


@pytest.mark.asyncio
async def test_message_window_keeps_last_row(cp):
    cfg = {"thread_id": "t"}
    await cp.aput_messages(cfg, [_msg(i) for i in range(10)])
    ids = lambda ms: [m.message_id for m in ms]  # noqa: E731
    assert ids(await cp.alist_messages(cfg)) == [f"m{i}" for i in range(10)]
    assert ids(await cp.alist_messages(cfg, limit=3)) == ["m0", "m1", "m2"]
    assert ids(await cp.alist_messages(cfg, offset=3, limit=4)) == ["m3", "m4", "m5", "m6"]
    assert ids(await cp.alist_messages(cfg, offset=8, limit=5)) == ["m8", "m9"]
    assert ids(await cp.alist_messages(cfg, offset=7)) == ["m7", "m8", "m9"]
    assert await cp.alist_messages(cfg, offset=10, limit=2) == []


@pytest.mark.asyncio
async def test_message_search_window(cp):
    cfg = {"thread_id": "t"}
    await cp.aput_messages(cfg, [_msg(i, "apple" if i % 2 == 0 else "pear") for i in range(10)])
    ids = lambda ms: [m.message_id for m in ms]  # noqa: E731
    assert ids(await cp.alist_messages(cfg, search="APPLE")) == ["m0", "m2", "m4", "m6", "m8"]
    assert ids(await cp.alist_messages(cfg, search="apple", offset=1, limit=2)) == ["m2", "m4"]
    assert ids(await cp.alist_messages(cfg, search="apple", offset=4, limit=3)) == ["m8"]


@pytest.mark.asyncio
async def test_limit_is_capped(cp, monkeypatch):
    monkeypatch.setattr(base_checkpointer, "MAX_LIST_LIMIT", 4)
    cfg = {"thread_id": "t"}
    await cp.aput_messages(cfg, [_msg(i) for i in range(10)])
    assert len(await cp.alist_messages(cfg, limit=100)) == 4
    # No limit given still means "everything" for in-process readers.
    assert len(await cp.alist_messages(cfg)) == 10


@pytest.mark.asyncio
async def test_thread_window_and_search(cp):
    for i in range(6):
        cfg = {"thread_id": f"t{i}", "user_id": "u1"}
        await cp.aput_thread(
            cfg, ThreadInfo(thread_id=f"t{i}", thread_name=f"name {i}", user_id="u1")
        )
    everything = await cp.alist_threads({"user_id": "u1"})
    assert len(everything) == 6
    ids = [t.thread_id for t in everything]
    page = await cp.alist_threads({"user_id": "u1"}, offset=2, limit=3)
    assert [t.thread_id for t in page] == ids[2:5]
    last = await cp.alist_threads({"user_id": "u1"}, offset=5, limit=3)
    assert [t.thread_id for t in last] == ids[5:]
    found = await cp.alist_threads({"user_id": "u1"}, search="name 3")
    assert [t.thread_id for t in found] == ["t3"]
    found = await cp.alist_threads({"user_id": "u1"}, search="NAME", offset=1, limit=2)
    assert [t.thread_id for t in found] == ids[1:3]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [InMemoryCheckpointer, lambda: SqliteCheckpointer(":memory:")])
async def test_owner_filter_with_paging(backend):
    cp = backend()
    try:
        await _check_owner_filter(cp)
    finally:
        await cp.arelease()


async def _check_owner_filter(cp):
    for owner in ("u1", "u2"):
        for i in range(4):
            tid = f"{owner}-{i}"
            cfg = {"thread_id": tid, "user_id": owner, "authz": build_authz(owner, scope="owner")}
            await cp.aput_thread(cfg, ThreadInfo(thread_id=tid, thread_name=tid, user_id=owner))
    cfg = {"user_id": "u1", "authz": build_authz("u1", scope="owner")}
    allu1 = await cp.alist_threads(cfg)
    assert sorted(t.thread_id for t in allu1) == [f"u1-{i}" for i in range(4)]
    page = await cp.alist_threads(cfg, offset=3, limit=5)
    assert [t.thread_id for t in page] == [allu1[3].thread_id]
    assert {t.user_id for t in await cp.alist_threads(cfg, search="u1", limit=2)} == {"u1"}


class _AcquireCtx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


@pytest.fixture
def pg(monkeypatch):
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    c = PgCheckpointer(postgres_dsn="postgres://x", redis=MagicMock())
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    c._pg_pool = MagicMock()
    c._pg_pool.acquire.return_value = _AcquireCtx(conn)
    c._test_conn = conn
    return c


@pytest.mark.asyncio
async def test_pg_rejects_negative_values(pg):
    for kwargs in ({"limit": -1}, {"offset": -1}):
        with pytest.raises(ValueError):
            await pg.alist_messages({"thread_id": "t", "user_id": "u"}, **kwargs)
        with pytest.raises(ValueError):
            await pg.alist_threads({"user_id": "u"}, **kwargs)
    pg._test_conn.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_pg_limit_is_capped(pg):
    await pg.alist_messages({"thread_id": "t", "user_id": "u"}, limit=10**9, offset=2)
    args = pg._test_conn.fetch.await_args.args
    assert MAX_LIST_LIMIT in args and 10**9 not in args
    assert 2 in args
    await pg.alist_threads({"user_id": "u"}, limit=10**9)
    assert MAX_LIST_LIMIT in pg._test_conn.fetch.await_args.args
