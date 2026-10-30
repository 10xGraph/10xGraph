"""Task type handling for GoogleEmbedding and the query/document split."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.storage.store.embedding.base_embedding import BaseEmbedding
from tenxgraph.storage.store.embedding.google_embedding import GoogleEmbedding


MOD = "tenxgraph.storage.store.embedding.google_embedding"


def _make(**kwargs):
    client = MagicMock()
    item = MagicMock()
    item.values = [0.1, 0.2]
    client.aio.models.embed_content = AsyncMock(return_value=MagicMock(embeddings=[item]))
    genai = MagicMock()
    genai.Client.return_value = client
    with patch(f"{MOD}.HAS_GOOGLE", True), patch(f"{MOD}.genai", genai):
        emb = GoogleEmbedding(api_key="k", **kwargs)
    return emb, client.aio.models.embed_content


def _task_type(mock):
    config = mock.call_args.kwargs["config"]
    return None if config is None else config.task_type


@pytest.mark.asyncio
async def test_store_uses_document_task_type_by_default():
    emb, call = _make()
    await emb.aembed("stored text")
    assert _task_type(call) == "RETRIEVAL_DOCUMENT"
    await emb.aembed_batch(["a", "b"])
    assert _task_type(call) == "RETRIEVAL_DOCUMENT"


@pytest.mark.asyncio
async def test_search_uses_query_task_type_by_default():
    emb, call = _make()
    await emb.aembed_query("what is stored?")
    assert _task_type(call) == "RETRIEVAL_QUERY"


@pytest.mark.asyncio
async def test_task_types_configurable_and_none_sends_none():
    emb, call = _make(document_task_type="SEMANTIC_SIMILARITY", query_task_type=None)
    await emb.aembed("x")
    assert _task_type(call) == "SEMANTIC_SIMILARITY"
    await emb.aembed_query("x")
    assert call.call_args.kwargs["config"] is None


@pytest.mark.asyncio
async def test_task_type_combined_with_output_dimensionality():
    emb, call = _make(output_dimensionality=256)
    await emb.aembed_query("x")
    config = call.call_args.kwargs["config"]
    assert config.task_type == "RETRIEVAL_QUERY"
    assert config.output_dimensionality == 256


@pytest.mark.asyncio
async def test_base_aembed_query_defaults_to_aembed():
    class Plain(BaseEmbedding):
        async def aembed_batch(self, texts):
            return [[1.0] for _ in texts]

        async def aembed(self, text):
            return [9.0]

        @property
        def dimension(self):
            return 1

    assert await Plain().aembed_query("q") == [9.0]
