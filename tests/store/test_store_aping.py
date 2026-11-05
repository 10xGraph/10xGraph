"""aping on the stores that talk to a backend: succeeds, raises, and has no side effects."""

from unittest.mock import AsyncMock, patch

import pytest

from tenxgraph.storage.store.qdrant_store import QdrantStore


class _Embedding:
    dimension = 4

    async def aembed(self, text):
        raise AssertionError("aping must not embed")

    async def aembed_batch(self, texts):
        raise AssertionError("aping must not embed")


def _qdrant(client):
    with patch("qdrant_client.AsyncQdrantClient", return_value=client):
        return QdrantStore(embedding=_Embedding(), path="./test_data")


@pytest.mark.asyncio
async def test_qdrant_aping_succeeds_without_side_effects():
    client = AsyncMock()
    store = _qdrant(client)
    assert await store.aping() is None
    client.get_collections.assert_awaited_once()
    client.create_collection.assert_not_called()
    client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_qdrant_aping_raises_when_unreachable():
    client = AsyncMock()
    client.get_collections.side_effect = ConnectionError("refused")
    store = _qdrant(client)
    with pytest.raises(ConnectionError):
        await store.aping()
    client.create_collection.assert_not_called()


def test_base_store_subclass_without_aping_cannot_be_created():
    from tenxgraph.storage.store.base_store import BaseStore

    attrs = {n: (lambda self, *a, **k: None) for n in BaseStore.__abstractmethods__ if n != "aping"}
    broken = type("NoPing", (BaseStore,), attrs)
    with pytest.raises(TypeError, match="aping"):
        broken()


@pytest.mark.asyncio
async def test_in_memory_store_aping_returns_none():
    from tenxgraph.qa.testing.in_memory_store import InMemoryStore

    assert await InMemoryStore().aping() is None
