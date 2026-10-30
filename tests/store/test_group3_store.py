"""Group 3 store regressions: G-30 Qdrant user 0, G-31 embedding dims, G-69 aware datetimes,
G-79 max_tokens on Qdrant."""

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.state import Message
from tenxgraph.storage.store.qdrant_store import QdrantStore
from tenxgraph.storage.store.store_schema import MemoryRecord, MemorySearchResult


class _Embedding:
    dimension = 4

    async def aembed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def client():
    mock = AsyncMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.retrieve.return_value = []
    mock.query_points.return_value = MagicMock(points=[])
    mock.scroll.return_value = ([], None)
    return mock


@pytest.fixture
def qdrant(client):
    with patch("qdrant_client.AsyncQdrantClient", return_value=client):
        return QdrantStore(embedding=_Embedding(), path="./test_data")


def test_qdrant_filter_keeps_integer_user_zero(qdrant):
    flt = qdrant._build_qdrant_filter(user_id=0)
    assert flt is not None
    assert [c.key for c in flt.must] == ["user_id"]


@pytest.mark.asyncio
async def test_qdrant_search_scopes_integer_user_zero(qdrant, client):
    await qdrant.asearch({"user_id": 0}, "q")
    flt = client.query_points.call_args.kwargs["query_filter"]
    assert flt is not None and flt.must[0].match.value == 0


@pytest.mark.parametrize(
    ("model", "dim"),
    [
        ("text-embedding-3-small", 1536),
        ("text-embedding-3-large", 3072),
        ("text-embedding-ada-002", 1536),
    ],
)
def test_openai_embedding_dimensions(model, dim):
    from tenxgraph.storage.store.embedding.openai_embedding import OpenAIEmbedding

    with patch("openai.AsyncOpenAI"):
        assert OpenAIEmbedding(model=model, api_key="k").dimension == dim


def test_openai_embedding_custom_dimensions_are_sent():
    from tenxgraph.storage.store.embedding.openai_embedding import OpenAIEmbedding

    with patch("openai.AsyncOpenAI"):
        emb = OpenAIEmbedding(model="text-embedding-3-large", api_key="k", dimensions=1024)
    assert emb.dimension == 1024


@pytest.mark.asyncio
async def test_openai_embedding_passes_dimensions_to_api():
    from tenxgraph.storage.store.embedding.openai_embedding import OpenAIEmbedding

    with patch("openai.AsyncOpenAI") as cls:
        cls.return_value.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=[0.0])])
        )
        emb = OpenAIEmbedding(model="text-embedding-3-large", api_key="k", dimensions=256)
        await emb.aembed("x")
        assert cls.return_value.embeddings.create.call_args.kwargs["dimensions"] == 256


def test_store_schema_timestamps_are_timezone_aware():
    assert MemorySearchResult().timestamp.tzinfo is not None
    assert MemoryRecord(content="x").timestamp.tzinfo is not None
    rec = MemoryRecord.from_message(Message.text_message("hi"))
    from datetime import datetime

    assert datetime.fromisoformat(rec.metadata["timestamp"]).tzinfo is not None
    assert rec.timestamp.tzinfo == UTC


@pytest.mark.asyncio
async def test_qdrant_payload_timestamps_are_timezone_aware(qdrant, client):
    from datetime import datetime

    await qdrant.astore({"user_id": "u"}, "hello")
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert datetime.fromisoformat(payload["timestamp"]).tzinfo is not None

    existing = MemorySearchResult(id="m1", content="old", user_id="u")
    with patch.object(qdrant, "aget", AsyncMock(return_value=existing)):
        await qdrant.aupdate({"user_id": "u"}, "m1", "new")
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert datetime.fromisoformat(payload["timestamp"]).tzinfo is not None


@pytest.mark.asyncio
async def test_qdrant_search_honors_max_tokens(qdrant, client):
    points = [
        MagicMock(id=str(i), score=0.9, payload={"content": "word " * 200, "user_id": "u"}, vector=None)
        for i in range(5)
    ]
    client.query_points.return_value = MagicMock(points=points)
    small = await qdrant.asearch({"user_id": "u"}, "q", max_tokens=400)
    assert 1 <= len(small) < 5
    assert len(await qdrant.asearch({"user_id": "u"}, "q", max_tokens=100_000)) == 5
