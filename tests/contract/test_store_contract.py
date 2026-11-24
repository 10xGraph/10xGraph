"""One behaviour contract, run against every memory store.

Stores: ``tenxgraph.qa.testing.in_memory_store.InMemoryStore`` (the documented test double),
``QdrantStore`` (local file mode, no server) and ``PgVectorStore`` (integration).
Known disagreements (gap ids S1 to S7 in ``cleanup-review/CORE_B.md``) live in
``conftest.KNOWN_GAPS`` as strict xfails.
"""

import asyncio
import contextlib
import hashlib
import uuid

import pytest
import pytest_asyncio

from tenxgraph.core.state import Message
from tenxgraph.storage.sql_filter import FilterError
from tenxgraph.storage.store.base_store import system_user_id
from tenxgraph.storage.store.embedding.base_embedding import BaseEmbedding
from tenxgraph.storage.store.store_schema import MemoryType

from .conftest import POSTGRES_DSN, TENANTS
from .conftest import cfg as tenant_cfg


pytestmark = pytest.mark.asyncio

DIM = 8


class FakeEmbedding(BaseEmbedding):
    """Deterministic bag-of-words hash vectors: shared words give similar vectors."""

    @property
    def dimension(self) -> int:
        return DIM

    async def aembed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in text.lower().split() or [""]:
            h = int(hashlib.md5(word.encode("utf-8", "surrogatepass")).hexdigest(), 16)  # noqa: S324
            vec[h % DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.aembed(t) for t in texts]


STORE_BACKENDS = ["inmem", "qdrant", pytest.param("pgvector", marks=pytest.mark.integration)]


@pytest_asyncio.fixture(params=STORE_BACKENDS)
async def backend(request):
    return request.param


@contextlib.asynccontextmanager
async def _open_store(kind, tmp_path, **kw):
    """A fresh store of ``kind``; ``kw`` (such as ``tenant_columns``) goes to the real stores."""
    if kind == "inmem":
        from tenxgraph.qa.testing.in_memory_store import InMemoryStore

        s = InMemoryStore()
        yield s
        await s.arelease()
        return
    if kind == "qdrant":
        from tenxgraph.storage.store.qdrant_store import QdrantStore

        s = QdrantStore(embedding=FakeEmbedding(), path=str(tmp_path / "qdrant"), **kw)
        await s.asetup()
        yield s
        await s.arelease()
        return
    asyncpg = pytest.importorskip("asyncpg")
    from tenxgraph.storage.store.pgvector_store import PgVectorStore

    try:
        pool = await asyncio.wait_for(
            asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=4), timeout=5
        )
    except Exception as e:
        pytest.skip(f"Postgres not reachable at POSTGRES_DSN: {e}")
    schema = f"tg_{uuid.uuid4().hex[:10]}"
    await pool.execute(f'CREATE SCHEMA "{schema}"')
    try:
        s = PgVectorStore(embedding=FakeEmbedding(), pool=pool, schema=schema, **kw)
        try:
            await s.asetup()
        except Exception as e:
            pytest.skip(f"pgvector not available: {e}")
        yield s
        await s.arelease()
    finally:
        await pool.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await pool.close()


@pytest_asyncio.fixture
async def store(backend, tmp_path):
    async with _open_store(backend, tmp_path) as s:
        yield s


def cfg(user="alice", reach="own", thread="t1", **extra):
    identity = {"user_id": user, "tenant": {}, "reach": reach, "claims": {}}
    return {"user_id": user, "thread_id": thread, "user": identity, **extra}


# ------------------------------------------------------------------ CRUD


async def test_store_and_get(store):
    mid = await store.astore(cfg(), "alice likes green tea", category="food")
    assert isinstance(mid, str)
    got = await store.aget(cfg(), mid)
    assert got is not None
    assert got.content == "alice likes green tea"
    assert got.id == mid


async def test_get_missing_is_none(store):
    assert await store.aget(cfg(), str(uuid.uuid4())) is None


async def test_store_accepts_a_message(store):
    mid = await store.astore(cfg(), Message.text_message("from a message", role="user"))
    assert (await store.aget(cfg(), mid)).content == "from a message"


async def test_search_finds_content(store):
    await store.astore(cfg(), "the quick brown fox")
    await store.astore(cfg(), "completely unrelated gardening notes")
    got = await store.asearch(cfg(), "quick brown fox", limit=1)
    assert [r.content for r in got] == ["the quick brown fox"]


async def test_metadata_roundtrip_without_payload_leak(store):
    """S1: the result metadata is the caller's metadata (plus category and key), not the
    whole payload (content, user_id, thread_id, timestamp)."""
    mid = await store.astore(cfg(), "meta memory", metadata={"color": "green"})
    got = await store.aget(cfg(), mid)
    assert got.metadata.get("color") == "green"
    assert not {"content", "user_id", "thread_id", "timestamp"} & set(got.metadata)


async def test_timestamp_moves_on_update(store):
    """S2: timestamp is a datetime and is not older after an update."""
    mid = await store.astore(cfg(), "stamp me")
    first = (await store.aget(cfg(), mid)).timestamp
    await asyncio.sleep(0.05)
    await store.aupdate(cfg(), mid, "stamp me again")
    second = (await store.aget(cfg(), mid)).timestamp
    assert second is not None
    assert second >= first


async def test_update_content_and_metadata(store):
    mid = await store.astore(cfg(), "old text", metadata={"a": 1})
    await store.aupdate(cfg(), mid, "new text", metadata={"b": 2})
    got = await store.aget(cfg(), mid)
    assert got.content == "new text"
    assert got.metadata.get("b") == 2


async def test_replace_metadata_keeps_category(store):
    """S3: replace_metadata=True without a category must keep the category (Q resets it)."""
    mid = await store.astore(cfg(), "cat memory", category="food", metadata={"a": 1})
    await store.aupdate(cfg(), mid, "cat memory 2", metadata={"b": 2}, replace_metadata=True)
    got = await store.aget(cfg(), mid)
    assert got.metadata.get("category") == "food"
    assert "a" not in got.metadata


async def test_update_missing_memory(store):
    """Updating an id that does not exist does not create it."""
    missing = str(uuid.uuid4())
    try:
        result = await store.aupdate(cfg(), missing, "x")
    except ValueError:
        result = None
    assert not result
    assert await store.aget(cfg(), missing) is None


async def test_delete_leaves_no_row(store):
    mid = await store.astore(cfg(), "delete me")
    await store.adelete(cfg(), mid)
    assert await store.aget(cfg(), mid) is None
    assert await store.aget_all(cfg()) == []
    assert await store.asearch(cfg(), "delete me") == []


async def test_get_all_newest_first_and_limit(store):
    """S7: newest first (V); Q scroll order."""
    for text in ("first", "second", "third"):
        await store.astore(cfg(), text)
        await asyncio.sleep(0.02)
    assert [r.content for r in await store.aget_all(cfg())] == ["third", "second", "first"]
    assert len(await store.aget_all(cfg(), limit=2)) == 2


async def test_get_all_honours_filters(store):
    """S7: Q ignores the ``filters`` keyword."""
    await store.astore(cfg(), "red one", metadata={"color": "red"})
    await store.astore(cfg(), "blue one", metadata={"color": "blue"})
    got = await store.aget_all(cfg(), filters={"color": "red"})
    assert [r.content for r in got] == ["red one"]


# ------------------------------------------------------------------ owners and reach


async def test_write_requires_user_id(store):
    """``base_store.py``: a store never falls back to "every user"; shared data uses
    system_user_id(name)."""
    anonymous = {"thread_id": "t1"}
    with pytest.raises(ValueError, match="user_id"):
        await store.astore(anonymous, "orphan")


async def test_blank_user_id_rejected(store):
    with pytest.raises(ValueError, match="user_id"):
        await store.astore({"user_id": "  ", "thread_id": "t"}, "orphan")


async def test_system_user_can_write_and_read(store):
    shared = {"user_id": system_user_id("kb"), "thread_id": "t"}
    mid = await store.astore(shared, "shared knowledge")
    assert (await store.aget(shared, mid)).content == "shared knowledge"


async def test_own_reach_isolates_users(store):
    mid = await store.astore(cfg("alice"), "alice secret")
    bob = cfg("bob")
    assert await store.aget(bob, mid) is None
    assert await store.aget_all(bob) == []
    assert await store.asearch(bob, "alice secret") == []


async def test_reach_all_reads_every_user(store):
    mid = await store.astore(cfg("alice"), "alice secret")
    await store.astore(cfg("bob"), "bob secret")
    root = cfg("root", reach="all")
    assert (await store.aget(root, mid)).content == "alice secret"
    assert {r.content for r in await store.aget_all(root)} == {"alice secret", "bob secret"}


async def test_reach_all_write_stays_own(store):
    """Writes always land under the caller's own id, whatever the reach."""
    root = cfg("root", reach="all")
    mid = await store.astore(root, "root note")
    assert await store.aget(cfg("alice"), mid) is None
    assert (await store.aget(cfg("root"), mid)).content == "root note"


async def test_reach_tenant_without_tenant_columns_raises(store):
    """A scope is never silently weakened: reach tenant on a store without tenant columns."""
    with pytest.raises(ValueError):
        await store.aget_all(cfg("alice", reach="tenant"))


async def test_update_foreign_memory_is_refused(store):
    """A foreign update is "not found" on every reach, never a change."""
    mid = await store.astore(cfg("alice"), "alice text")
    for caller in (cfg("bob"), cfg("root", reach="all")):
        with pytest.raises(ValueError, match="not found"):
            await store.aupdate(caller, mid, "hijacked")
        assert (await store.aget(cfg("alice"), mid)).content == "alice text"


async def test_update_foreign_memory_under_reach_all_raises_value_error(store):
    """S4: the exception type for a foreign update is ValueError (not found), as on V."""
    mid = await store.astore(cfg("alice"), "alice text")
    with pytest.raises(ValueError):
        await store.aupdate(cfg("root", reach="all"), mid, "hijacked")


async def test_delete_foreign_memory_raises_and_keeps_row(store):
    """S5: both real stores raise ValueError; the row stays."""
    mid = await store.astore(cfg("alice"), "alice text")
    with pytest.raises(ValueError):
        await store.adelete(cfg("bob"), mid)
    assert (await store.aget(cfg("alice"), mid)).content == "alice text"


async def test_forget_memory_deletes_only_the_owner(store):
    await store.astore(cfg("alice"), "a1")
    await store.astore(cfg("alice"), "a2")
    await store.astore(cfg("bob"), "b1")
    await store.aforget_memory(cfg("alice"))
    assert await store.aget_all(cfg("alice")) == []
    assert [r.content for r in await store.aget_all(cfg("bob"))] == ["b1"]


async def test_forget_memory_under_reach_all_still_names_an_owner(store):
    """``aforget_memory`` ignores reach all: it deletes the caller's own memories only."""
    await store.astore(cfg("alice"), "a1")
    await store.astore(cfg("root"), "r1")
    await store.aforget_memory(cfg("root", reach="all"))
    assert [r.content for r in await store.aget_all(cfg("alice"))] == ["a1"]
    assert await store.aget_all(cfg("root")) == []


async def test_forget_memory_needs_an_owner(store):
    await store.astore(cfg("alice"), "a1")
    with pytest.raises(ValueError, match="user_id"):
        await store.aforget_memory({"thread_id": "t1"})
    assert len(await store.aget_all(cfg("alice"))) == 1


# ------------------------------------------------------------------ collections, keys


async def test_collections_are_separate(store):
    mid = await store.astore(cfg(collection="notes_a"), "in a")
    assert await store.aget(cfg(collection="notes_b"), mid) is None
    assert (await store.aget(cfg(collection="notes_a"), mid)).content == "in a"
    assert await store.aget_all(cfg(collection="notes_b")) == []


async def test_blank_collection_rejected(store):
    with pytest.raises(ValueError, match="collection"):
        await store.astore(cfg(collection="   "), "x")


async def test_memory_key_upsert_matches_the_flag(store):
    """``supports_memory_key_upsert`` True: a second keyed write replaces the first for that
    owner; False: the memory layer dedups itself, and the store is not required to."""
    if not store.supports_memory_key_upsert:
        pytest.skip("store does not claim atomic memory_key upserts")
    await store.astore(cfg(), "v1", memory_key="pref:tea")
    await store.astore(cfg(), "v2", memory_key="pref:tea")
    await store.astore(cfg("bob"), "bob v1", memory_key="pref:tea")
    assert [r.content for r in await store.aget_all(cfg())] == ["v2"]
    assert [r.content for r in await store.aget_all(cfg("bob"))] == ["bob v1"]


async def test_memory_type_filter(store):
    await store.astore(cfg(), "semantic fact", memory_type=MemoryType.SEMANTIC)
    await store.astore(cfg(), "episodic fact", memory_type=MemoryType.EPISODIC)
    got = await store.asearch(cfg(), "fact", memory_type=MemoryType.SEMANTIC)
    assert [r.content for r in got] == ["semantic fact"]


# ------------------------------------------------------------------ failures, odd text


async def test_get_when_backend_fails_raises(store, backend, monkeypatch):
    """S6: V raises; Q swallows every exception and returns None."""
    if backend == "inmem":
        pytest.skip("nothing to fail in process")
    mid = await store.astore(cfg(), "x")
    if backend == "qdrant":

        async def boom(*a, **k):
            raise RuntimeError("backend down")

        monkeypatch.setattr(store.client, "retrieve", boom)
    else:
        pytest.skip("Postgres failure injection needs a closed pool; covered unit-side")
    with pytest.raises(Exception, match="backend down"):
        await store.aget(cfg(), mid)


_WRITES_WITH_BAD_TEXT = {
    "astore-content": lambda s, mid, bad: s.astore(cfg(), f"x{bad}"),
    "astore-message": lambda s, mid, bad: s.astore(cfg(), Message.text_message(f"x{bad}")),
    "astore-metadata-value": lambda s, mid, bad: s.astore(cfg(), "x", metadata={"k": bad}),
    "astore-metadata-key": lambda s, mid, bad: s.astore(cfg(), "x", metadata={bad: 1}),
    "astore-nested-metadata": lambda s, mid, bad: s.astore(cfg(), "x", metadata={"k": [bad]}),
    "astore-category": lambda s, mid, bad: s.astore(cfg(), "x", category=bad),
    "aupdate-content": lambda s, mid, bad: s.aupdate(cfg(), mid, f"x{bad}"),
    "aupdate-metadata": lambda s, mid, bad: s.aupdate(cfg(), mid, "x", metadata={"k": bad}),
}


@pytest.mark.parametrize("write", list(_WRITES_WITH_BAD_TEXT))
@pytest.mark.parametrize("bad", ["\x00", "\ud800"], ids=["nul", "lone-surrogate"])
async def test_write_with_nul_or_surrogate_is_refused(store, write, bad):
    """Every store refuses NUL and lone surrogates with ValueError before the write, and
    nothing changes."""
    mid = await store.astore(cfg(), "original", metadata={"k": "v"})
    with pytest.raises(ValueError, match="NUL character"):
        await _WRITES_WITH_BAD_TEXT[write](store, mid, bad)
    rows = await store.aget_all(cfg())
    assert [(r.id, r.content, r.metadata.get("k")) for r in rows] == [(mid, "original", "v")]


# ------------------------------------------------------------------ tenant isolation
# Stores declared with ``tenant_columns=TENANTS`` (org_id, project_id). Only the backends with
# tenant columns run here. ``tenant_cfg`` builds an identity with tenant values.

TENANT_BACKENDS = ["qdrant", pytest.param("pgvector", marks=pytest.mark.integration)]


@pytest_asyncio.fixture(params=TENANT_BACKENDS)
async def tenant_store(request, tmp_path):
    async with _open_store(request.param, tmp_path, tenant_columns=TENANTS) as s:
        yield s


async def _drop_tenant_values(store, memory_id):
    """Turn a stored memory into one written before the tenant columns existed."""
    if hasattr(store, "client"):  # Qdrant: the payload has no tenant keys
        await store.client.delete_payload(
            collection_name=store.collection, keys=list(TENANTS), points=[memory_id]
        )
    else:  # pgvector: the columns hold NULL
        await store._pool.execute(
            f"UPDATE {store._qualified_table} SET org_id = NULL, project_id = NULL "  # noqa: S608
            "WHERE id = $1",
            memory_id,
        )


async def _outcome(pending, memory_id):
    """``(exception type, message with the id masked)``, or ``("ok", result)``."""
    try:
        return "ok", await pending
    except Exception as e:
        return type(e), str(e).replace(memory_id, "<id>")


async def test_tenant_own_reach_isolates_the_same_user_across_tenants(tenant_store):
    s = tenant_store
    mine = tenant_cfg(user="alice", org="A")
    other = tenant_cfg(user="alice", org="B")
    mid = await s.astore(mine, "alice in org a")
    assert await s.aget(other, mid) is None
    assert await s.aget_all(other) == []
    assert await s.asearch(other, "alice in org a") == []
    with pytest.raises(ValueError):
        await s.aupdate(other, mid, "hijacked")
    with pytest.raises(ValueError):
        await s.adelete(other, mid)
    await s.aforget_memory(other)
    got = await s.aget(mine, mid)
    assert got.content == "alice in org a"
    assert [r.content for r in await s.aget_all(mine)] == ["alice in org a"]


async def test_tenant_project_is_part_of_the_scope(tenant_store):
    s = tenant_store
    mid = await s.astore(tenant_cfg(user="alice", org="A", project="P1"), "project one")
    assert await s.aget(tenant_cfg(user="alice", org="A", project="P2"), mid) is None


async def test_tenant_reach_tenant_shares_inside_the_tenant_only(tenant_store):
    s = tenant_store
    a_mid = await s.astore(tenant_cfg(user="alice", org="A"), "alice note")
    await s.astore(tenant_cfg(user="bob", org="A"), "bob note")
    c_mid = await s.astore(tenant_cfg(user="carol", org="B"), "carol note")
    team_a = tenant_cfg(user="bob", org="A", reach="tenant")
    assert {r.content for r in await s.aget_all(team_a)} == {"alice note", "bob note"}
    assert (await s.aget(team_a, a_mid)).content == "alice note"
    assert await s.aget(team_a, c_mid) is None
    assert "carol note" not in {r.content for r in await s.asearch(team_a, "carol note")}
    with pytest.raises(ValueError):
        await s.adelete(team_a, c_mid)
    team_b = tenant_cfg(user="dave", org="B", reach="tenant")
    assert [r.content for r in await s.aget_all(team_b)] == ["carol note"]


async def test_tenant_reach_tenant_update_stays_own(tenant_store):
    """Writes stay reach own: a teammate cannot update another user's memory."""
    s = tenant_store
    mid = await s.astore(tenant_cfg(user="alice", org="A"), "alice text")
    with pytest.raises(ValueError):
        await s.aupdate(tenant_cfg(user="bob", org="A", reach="tenant"), mid, "hijacked")
    assert (await s.aget(tenant_cfg(user="alice", org="A"), mid)).content == "alice text"


async def test_tenant_reach_tenant_forget_keeps_other_tenants(tenant_store):
    s = tenant_store
    await s.astore(tenant_cfg(user="alice", org="A"), "a in A")
    await s.astore(tenant_cfg(user="alice", org="B"), "a in B")
    await s.aforget_memory(tenant_cfg(user="alice", org="A", reach="tenant"))
    assert await s.aget_all(tenant_cfg(user="alice", org="A")) == []
    assert [r.content for r in await s.aget_all(tenant_cfg(user="alice", org="B"))] == ["a in B"]


async def test_tenant_reach_all_sees_every_tenant(tenant_store):
    s = tenant_store
    await s.astore(tenant_cfg(user="alice", org="A"), "in A")
    b_mid = await s.astore(tenant_cfg(user="carol", org="B"), "in B")
    root = tenant_cfg(user="root", org="A", reach="all")
    assert {r.content for r in await s.aget_all(root)} == {"in A", "in B"}
    assert (await s.aget(root, b_mid)).content == "in B"


async def test_tenant_foreign_id_is_indistinguishable_from_missing(tenant_store):
    s = tenant_store
    mid = await s.astore(tenant_cfg(user="alice", org="A"), "alice text")
    missing = str(uuid.uuid4())
    other = tenant_cfg(user="alice", org="B")
    for call in (
        lambda i: s.aget(other, i),
        lambda i: s.aupdate(other, i, "x"),
        lambda i: s.adelete(other, i),
    ):
        foreign = await _outcome(call(mid), mid)
        assert foreign == await _outcome(call(missing), missing)


async def test_tenant_values_come_from_the_identity_not_metadata(tenant_store):
    s = tenant_store
    mid = await s.astore(tenant_cfg(user="alice", org="A"), "spoof", metadata={"org_id": "B"})
    assert await s.aget(tenant_cfg(user="alice", org="B"), mid) is None
    assert (await s.aget(tenant_cfg(user="alice", org="A"), mid)).content == "spoof"


async def test_tenant_write_without_tenant_value_raises(tenant_store):
    no_tenant = {
        "user_id": "alice",
        "thread_id": "t1",
        "user": {"user_id": "alice", "tenant": {}, "reach": "own", "claims": {}},
    }
    with pytest.raises(ValueError):
        await tenant_store.astore(no_tenant, "orphan")


async def test_tenant_legacy_memory_without_tenant_values(tenant_store):
    """A memory written before the columns existed is invisible to scoped callers (own and
    tenant), cannot be changed or deleted by them, and stays visible to reach all."""
    s = tenant_store
    mine = tenant_cfg(user="alice", org="A")
    mid = await s.astore(mine, "legacy note")
    await _drop_tenant_values(s, mid)
    team = tenant_cfg(user="alice", org="A", reach="tenant")
    for caller in (mine, team):
        assert await s.aget(caller, mid) is None
        assert await s.aget_all(caller) == []
        assert await s.asearch(caller, "legacy note") == []
        with pytest.raises(ValueError):
            await s.adelete(caller, mid)
    with pytest.raises(ValueError):
        await s.aupdate(mine, mid, "changed")
    with pytest.raises(ValueError, match="not found"):
        await s.astore(mine, "overwrite", memory_id=mid)
    await s.aforget_memory(mine)
    root = tenant_cfg(user="root", org="A", reach="all")
    assert (await s.aget(root, mid)).content == "legacy note"


NO_IDENTITY = {"user_id": "alice", "thread_id": "t1"}
ROOT = {"user_id": "root", "user": {"user_id": "root", "reach": "all"}}


async def test_tenant_store_without_identity_cannot_read(tenant_store):
    """By design: with no identity every tenant column applies and has no value, so reads,
    deletes and forgets raise instead of reading across tenants."""
    s = tenant_store
    mid = await s.astore(tenant_cfg(user="alice", org="A"), "tenant note")
    for call in (
        lambda: s.aget(NO_IDENTITY, mid),
        lambda: s.aget_all(NO_IDENTITY),
        lambda: s.asearch(NO_IDENTITY, "tenant note"),
        lambda: s.adelete(NO_IDENTITY, mid),
        lambda: s.aforget_memory(NO_IDENTITY),
    ):
        with pytest.raises(ValueError, match="org_id"):
            await call()
    assert (await s.aget(ROOT, mid)).content == "tenant note"


@pytest.mark.parametrize("kind", TENANT_BACKENDS)
async def test_update_without_identity_compares_missing_tenant_values(kind, tmp_path):
    """An optional tenant column the caller has no value for matches only rows without one,
    on update as on store (``IS NOT DISTINCT FROM``), never ``= NULL``."""
    optional = {"org_id": {"type": "text", "required": False}}
    async with _open_store(kind, tmp_path, tenant_columns=optional) as s:
        mine = await s.astore(NO_IDENTITY, "no tenant")
        theirs = await s.astore(tenant_cfg(user="alice", org="A"), "org a")
        await s.aupdate(NO_IDENTITY, mine, "no tenant updated")
        with pytest.raises(ValueError, match="not found"):
            await s.aupdate(NO_IDENTITY, theirs, "hijacked")
        with pytest.raises(ValueError, match="not found"):
            await s.astore(NO_IDENTITY, "hijacked", memory_id=theirs)
        assert (await s.aget(ROOT, mine)).content == "no tenant updated"
        assert (await s.aget(ROOT, theirs)).content == "org a"


# ------------------------------------------------------------------ isolation matrix
# One data set, every caller, every call. ``_visible`` is the rule; the stores must match it.
# InMemoryStore has no tenant columns: tenant values are ignored and reach tenant is skipped.

# key, user, org, project, category, memory type, color
ROWS = [
    ("a1", "alice", "A", "P1", "work", MemoryType.EPISODIC, "red"),
    ("a2", "alice", "B", "P1", "home", MemoryType.SEMANTIC, "red"),
    ("ap2", "alice", "A", "P2", "work", MemoryType.SEMANTIC, "blue"),
    ("b1", "bob", "A", "P1", "home", MemoryType.EPISODIC, "blue"),
    ("c1", "carol", "B", "P1", "work", MemoryType.EPISODIC, "red"),
]
CALLERS = {
    "alice-A-own": ("alice", "A", "P1", "own"),
    "alice-B-own": ("alice", "B", "P1", "own"),
    "alice-A-P2-own": ("alice", "A", "P2", "own"),
    "bob-A-own": ("bob", "A", "P1", "own"),
    "bob-A-tenant": ("bob", "A", "P1", "tenant"),
    "dave-B-tenant": ("dave", "B", "P1", "tenant"),
    "root-A-all": ("root", "A", "P1", "all"),
}
# Filters a caller could send to reach other rows: refused on every store.
WIDENING_FILTERS = [
    {"user_id": "alice"},
    {"user_id": "carol"},
    {"org_id": "B"},
    {"project_id": "P2"},
    {"user_id": "carol", "org_id": "B"},
]


def _visible(caller, row, tenants, reach=None):
    user, org, project, own_reach = caller
    reach = reach or own_reach
    same_tenant = not tenants or (row[2], row[3]) == (org, project)
    if reach == "all":
        return True
    if reach == "tenant":
        return same_tenant
    return same_tenant and row[1] == user


@pytest_asyncio.fixture(params=STORE_BACKENDS)
async def iso(request, tmp_path):
    """``(store, has tenant columns, {row key: memory id})`` seeded with ``ROWS``."""
    kind = request.param
    kw = {} if kind == "inmem" else {"tenant_columns": TENANTS}
    async with _open_store(kind, tmp_path, **kw) as s:
        ids = {}
        for key, user, org, project, category, memory_type, color in ROWS:
            ids[key] = await s.astore(
                tenant_cfg(user=user, org=org, project=project),
                f"note {key}",
                category=category,
                memory_type=memory_type,
                metadata={"color": color},
            )
        yield s, kind != "inmem", ids


def _caller(iso, name):
    _s, tenants, _ids = iso
    user, org, project, reach = CALLERS[name]
    if reach == "tenant" and not tenants:
        pytest.skip("reach tenant needs tenant columns")
    return CALLERS[name], tenant_cfg(user=user, org=org, project=project, reach=reach)


def _notes(rows):
    return {f"note {r[0]}" for r in rows}


@pytest.mark.parametrize("name", list(CALLERS))
async def test_isolation_reads(iso, name):
    s, tenants, ids = iso
    caller, config = _caller(iso, name)
    visible = [r for r in ROWS if _visible(caller, r, tenants)]
    for row in ROWS:
        got = await s.aget(config, ids[row[0]])
        assert (got is not None) == (row in visible), row[0]
    assert await s.aget(config, str(uuid.uuid4())) is None
    assert {r.content for r in await s.aget_all(config)} == _notes(visible)
    narrowings = [
        ({}, visible),
        ({"category": "work"}, [r for r in visible if r[4] == "work"]),
        ({"memory_type": MemoryType.SEMANTIC}, [r for r in visible if r[5] is MemoryType.SEMANTIC]),
        ({"filters": {"color": "red"}}, [r for r in visible if r[6] == "red"]),
        (
            {"category": "work", "filters": {"color": "red"}},
            [r for r in visible if r[4] == "work" and r[6] == "red"],
        ),
    ]
    for kwargs, expected in narrowings:
        got = await s.asearch(config, "note", limit=50, **kwargs)
        assert {r.content for r in got} == _notes(expected), kwargs
    red = [r for r in visible if r[6] == "red"]
    got = await s.aget_all(config, filters={"color": "red"})
    assert {r.content for r in got} == _notes(red)


@pytest.mark.parametrize("name", list(CALLERS))
async def test_isolation_filters_are_refused(iso, name):
    """A filter on user_id or a tenant column is refused on every store, never ANDed, and
    forget deletes nothing when it refuses."""
    s, tenants, _ids = iso
    _caller_row, config = _caller(iso, name)
    refused = {"user_id", *(TENANTS if tenants else ())}
    before = {r.content for r in await s.aget_all(ROOT)}
    for filters in WIDENING_FILTERS:
        if not refused & filters.keys():
            continue
        for call in (
            lambda f: s.asearch(config, "note", limit=50, filters=f),
            lambda f: s.aget_all(config, filters=f),
            lambda f: s.aforget_memory(config, filters=f),
        ):
            with pytest.raises(FilterError, match="isolation column"):
                await call(filters)
        assert {r.content for r in await s.aget_all(ROOT)} == before, filters


@pytest.mark.parametrize("name", list(CALLERS))
async def test_isolation_update_and_delete(iso, name):
    """Update and store-by-id stay reach own; delete follows the read scope. A foreign id
    answers exactly like a missing one."""
    s, tenants, ids = iso
    caller, config = _caller(iso, name)
    missing = str(uuid.uuid4())
    for call in (
        lambda i: s.aupdate(config, i, "updated"),
        lambda i: s.adelete(config, i),
    ):
        assert await _outcome(call(missing), missing) == (ValueError, "Memory <id> not found")

    for row in ROWS:
        mid = ids[row[0]]
        writable = _visible(caller, row, tenants, reach="own")
        outcome = await _outcome(s.aupdate(config, mid, f"updated {row[0]}"), mid)
        assert (outcome[0] == "ok") == writable, row[0]
        if not writable:
            assert outcome == (ValueError, "Memory <id> not found")
        outcome = await _outcome(s.astore(config, f"stored {row[0]}", memory_id=mid), mid)
        assert (outcome[0] == "ok") == writable, row[0]
        if not writable:
            assert outcome == (ValueError, "Memory <id> not found")
    expected = {
        f"stored {r[0]}" if _visible(caller, r, tenants, reach="own") else f"note {r[0]}"
        for r in ROWS
    }
    assert {r.content for r in await s.aget_all(ROOT)} == expected

    for row in ROWS:
        mid = ids[row[0]]
        outcome = await _outcome(s.adelete(config, mid), mid)
        assert (outcome[0] == "ok") == _visible(caller, row, tenants), row[0]
        if outcome[0] != "ok":
            assert outcome == (ValueError, "Memory <id> not found")
    left = {r.id for r in await s.aget_all(ROOT)}
    assert left == {ids[r[0]] for r in ROWS if not _visible(caller, r, tenants)}


@pytest.mark.parametrize("name", list(CALLERS))
async def test_isolation_forget(iso, name):
    """Forget deletes the caller's own memories inside the read scope's tenant values."""
    s, tenants, _ids = iso
    caller, config = _caller(iso, name)
    mine = [r for r in ROWS if r[1] == caller[0] and _visible(caller, r, tenants)]
    await s.aforget_memory(config, category="work")
    gone = _notes(ROWS) - {r.content for r in await s.aget_all(ROOT)}
    assert gone == _notes(r for r in mine if r[4] == "work")
    await s.aforget_memory(config)
    gone = _notes(ROWS) - {r.content for r in await s.aget_all(ROOT)}
    assert gone == _notes(mine)
