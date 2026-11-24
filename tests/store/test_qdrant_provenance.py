"""QdrantStore origin and expiry against the real client in local mode (path=":memory:")."""

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from tenxgraph.storage.store import TRUSTED_ORIGINS, MemoryOrigin
from tenxgraph.storage.store.qdrant_store import QdrantStore


pytest.importorskip("qdrant_client")

CFG = {"user_id": "alice"}
PAST = datetime.now(UTC) - timedelta(hours=1)
FUTURE = datetime.now(UTC) + timedelta(days=1)


class _Embedding:
    """Every text gets nearly the same vector; ``near`` texts sort first."""

    dimension = 4

    async def aembed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0] if "near" in text else [0.9, 0.1, 0.0, 0.0]


@pytest_asyncio.fixture
async def store():
    s = QdrantStore(embedding=_Embedding(), path=":memory:", collection="mem")
    yield s
    await s.client.close()


@pytest.mark.asyncio
class TestQdrantProvenance:
    async def test_fields_round_trip_and_metadata_cannot_set_them(self, store):
        mid = await store.astore(
            CFG,
            "fact",
            metadata={"origin": "system", "expires_at": "2000-01-01T00:00:00+00:00"},
            origin=MemoryOrigin.USER,
            expires_at=FUTURE,
        )
        got = await store.aget(CFG, mid)
        assert got.origin == MemoryOrigin.USER
        assert got.expires_at == FUTURE
        assert "origin" not in got.metadata and "expires_at" not in got.metadata

    async def test_naive_expires_at_refused(self, store):
        with pytest.raises(ValueError, match="timezone-aware"):
            await store.astore(CFG, "x", expires_at=datetime(2030, 1, 1))  # noqa: DTZ001

    async def test_expired_hidden_and_top_k_not_shrunk(self, store):
        # The nearest points are expired; limit=2 must still return the two live ones.
        expired = [await store.astore(CFG, f"near {i}", expires_at=PAST) for i in range(3)]
        live = [await store.astore(CFG, f"far {i}", expires_at=FUTURE) for i in range(2)]
        found = await store.asearch(CFG, "near", limit=2)
        assert sorted(r.id for r in found) == sorted(live)
        assert {r.id for r in await store.aget_all(CFG)} == set(live)
        assert await store.aget(CFG, expired[0]) is None
        # Delete still reaches an expired memory.
        await store.adelete(CFG, expired[0])

    async def test_origins_filter_and_default(self, store):
        ids = {
            o: await store.astore(CFG, f"note {o}", origin=o)
            for o in (MemoryOrigin.USER, MemoryOrigin.AGENT, MemoryOrigin.SYSTEM)
        }
        unknown = await store.astore(CFG, "note unknown")
        assert len(await store.asearch(CFG, "note", limit=10)) == 4
        trusted = await store.asearch(CFG, "note", limit=10, origins=list(TRUSTED_ORIGINS))
        assert {r.id for r in trusted} == {ids[MemoryOrigin.USER], ids[MemoryOrigin.SYSTEM]}
        listed = await store.aget_all(CFG, origins=[MemoryOrigin.AGENT, None])
        assert {r.id for r in listed} == {ids[MemoryOrigin.AGENT], unknown}
        only_unknown = await store.asearch(CFG, "note", limit=10, origins=[None])
        assert [r.id for r in only_unknown] == [unknown]

    async def test_update_carries_provenance_over_rebuilt_payload(self, store):
        mid = await store.astore(CFG, "a", origin=MemoryOrigin.TOOL, expires_at=FUTURE)
        await store.aupdate(CFG, mid, "b", metadata={"x": 1}, replace_metadata=True)
        got = await store.aget(CFG, mid)
        assert (got.origin, got.expires_at) == (MemoryOrigin.TOOL, FUTURE)
        await store.aupdate(CFG, mid, "c", origin=MemoryOrigin.SYSTEM, expires_at=PAST)
        assert await store.aget(CFG, mid) is None
        later = FUTURE + timedelta(days=1)
        await store.aupdate(CFG, mid, "d", expires_at=later)
        got = await store.aget(CFG, mid)
        assert (got.origin, got.expires_at, got.content) == (MemoryOrigin.SYSTEM, later, "d")

    async def test_adelete_expired(self, store):
        keep = await store.astore(CFG, "keep")
        for i in range(5):
            await store.astore({"user_id": f"u{i}"}, f"old {i}", expires_at=PAST)
        assert await store.adelete_expired(batch_size=2) == 5
        assert await store.adelete_expired() == 0
        assert [r.id for r in await store.aget_all(CFG)] == [keep]

    async def test_adelete_expired_needs_a_name_with_function_default(self):
        s = QdrantStore(embedding=_Embedding(), path=":memory:", collection=lambda c: "x")
        try:
            with pytest.raises(ValueError, match="collection name"):
                await s.adelete_expired()
            assert await s.adelete_expired(collection="x") == 0
        finally:
            await s.client.close()
