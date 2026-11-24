"""QdrantStore and InMemoryStore filter refusal and the search score clamp (local mode)."""

import pytest
import pytest_asyncio

from tenxgraph.qa.testing.in_memory_store import InMemoryStore
from tenxgraph.storage.sql_filter import FilterError
from tenxgraph.storage.store.embedding.base_embedding import BaseEmbedding
from tenxgraph.storage.store.qdrant_store import QdrantStore


pytestmark = pytest.mark.asyncio

ISOLATION = "isolation column set from the caller's identity"


class Opposite(BaseEmbedding):
    """Stored text embeds to [1, 0], the query to [-1, 0]: cosine -1."""

    dimension = 2

    async def aembed(self, text):
        return [-1.0, 0.0] if text == "query" else [1.0, 0.0]

    async def aembed_batch(self, texts):
        return [await self.aembed(t) for t in texts]


CFG = {"user_id": "alice"}


@pytest_asyncio.fixture
async def qdrant(tmp_path):
    store = QdrantStore(
        embedding=Opposite(), path=str(tmp_path / "q"), collection="c", tenant_columns=["org_id"]
    )
    await store.astore({"user": {"user_id": "alice", "tenant": {"org_id": "A"}}}, "doc")
    yield store
    await store.arelease()


ALICE = {"user": {"user_id": "alice", "tenant": {"org_id": "A"}}}


async def test_negative_cosine_score_is_clamped(qdrant):
    results = await qdrant.asearch(ALICE, "query")
    assert [r.score for r in results] == [0.0]


@pytest.mark.parametrize("key", ["user_id", "org_id"])
async def test_qdrant_refuses_isolation_filters(qdrant, key):
    with pytest.raises(FilterError, match=ISOLATION):
        await qdrant.asearch(ALICE, "doc", filters={key: "x"})
    with pytest.raises(FilterError, match=ISOLATION):
        await qdrant.aget_all(ALICE, filters={key: "x"})
    with pytest.raises(FilterError, match=ISOLATION):
        await qdrant.aforget_memory(ALICE, filters={key: "x"})
    assert len(await qdrant.aget_all(ALICE)) == 1


async def test_qdrant_refuses_unmatchable_filter_value(qdrant):
    bad = {"tags": ["a"]}
    with pytest.raises(FilterError, match="Unsupported filter"):
        await qdrant.asearch(ALICE, "doc", filters=bad)
    with pytest.raises(FilterError, match="Unsupported filter"):
        await qdrant.aget_all(ALICE, filters=bad)
    with pytest.raises(FilterError, match="Unsupported filter"):
        await qdrant.aforget_memory(ALICE, filters=bad)


async def test_in_memory_store_refuses_user_id_filter():
    store = InMemoryStore()
    await store.astore(CFG, "doc")
    with pytest.raises(FilterError, match=ISOLATION):
        await store.asearch(CFG, "doc", filters={"user_id": "bob"})
    with pytest.raises(FilterError, match=ISOLATION):
        await store.aget_all(CFG, filters={"user_id": "bob"})
    with pytest.raises(FilterError, match=ISOLATION):
        await store.aforget_memory(CFG, filters={"user_id": "bob"})
    assert len(await store.aget_all(CFG)) == 1
