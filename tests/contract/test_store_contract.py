"""One behaviour contract, run against every memory store.

Stores: ``tenxgraph.qa.testing.in_memory_store.InMemoryStore`` (the documented test double),
``QdrantStore`` (local file mode, no server) and ``PgVectorStore`` (integration).
Known disagreements (gap ids S1 to S7 in ``cleanup-review/CORE_B.md``) live in
``conftest.KNOWN_GAPS`` as strict xfails.
"""

import asyncio
import hashlib
import uuid

import pytest
import pytest_asyncio

from tenxgraph.core.state import Message
from tenxgraph.storage.store.base_store import system_user_id
from tenxgraph.storage.store.embedding.base_embedding import BaseEmbedding
from tenxgraph.storage.store.store_schema import MemoryType
from tenxgraph.utils.validators import ValidationError

from .conftest import POSTGRES_DSN


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


@pytest_asyncio.fixture
async def store(backend, tmp_path):
    if backend == "inmem":
        from tenxgraph.qa.testing.in_memory_store import InMemoryStore

        s = InMemoryStore()
        yield s
        await s.arelease()
    elif backend == "qdrant":
        from tenxgraph.storage.store.qdrant_store import QdrantStore

        s = QdrantStore(embedding=FakeEmbedding(), path=str(tmp_path / "qdrant"))
        await s.asetup()
        yield s
        await s.arelease()
    else:
        asyncpg = pytest.importorskip("asyncpg")
        from tenxgraph.storage.store.pgvector_store import PgVectorStore

        try:
            pool = await asyncio.wait_for(asyncpg.create_pool(POSTGRES_DSN, max_size=4), timeout=5)
        except Exception as e:
            pytest.skip(f"Postgres not reachable at POSTGRES_DSN: {e}")
        schema = f"tg_{uuid.uuid4().hex[:10]}"
        await pool.execute(f'CREATE SCHEMA "{schema}"')
        s = PgVectorStore(embedding=FakeEmbedding(), pool=pool, schema=schema)
        try:
            await s.asetup()
        except Exception as e:
            await pool.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await pool.close()
            pytest.skip(f"pgvector not available: {e}")
        yield s
        await s.arelease()
        await pool.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await pool.close()


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
    """S4: Q raises PermissionError under reach all, V raises ValueError; own reach: ValueError
    or a falsy result, never a change."""
    mid = await store.astore(cfg("alice"), "alice text")
    for caller in (cfg("bob"), cfg("root", reach="all")):
        try:
            await store.aupdate(caller, mid, "hijacked")
        except (ValueError, PermissionError):
            pass
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


@pytest.mark.parametrize("bad", ["a\x00b", "a\ud800b"], ids=["nul", "lone-surrogate"])
async def test_text_with_nul_or_surrogate(store, bad):
    """Round trip exactly, or refuse with ValidationError. Never corrupt, never a raw driver
    error."""
    try:
        mid = await store.astore(cfg(), bad)
    except ValidationError:
        return
    assert (await store.aget(cfg(), mid)).content == bad
