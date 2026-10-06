"""Tests for GoogleEmbedding - using mocks to avoid real API calls."""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tenxgraph.storage.store.embedding.google_embedding import GoogleEmbedding


class TestGoogleEmbeddingAvailability:
    """Test GoogleEmbedding availability based on imports."""

    def test_raises_import_error_without_google(self):
        """Test that ImportError is raised when google-genai is not available."""
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", False):
            with pytest.raises(ImportError, match="google-genai"):
                GoogleEmbedding(api_key="test-key")


class TestGoogleEmbeddingInit:
    """Test GoogleEmbedding initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            mock_genai.Client.return_value = MagicMock()
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                emb = GoogleEmbedding(api_key="test-api-key")
                assert emb.model == "gemini-embedding-001"
                assert emb.api_key == "test-api-key"
                assert emb._output_dimensionality is None

    def test_init_with_google_api_key_env(self):
        """Test initialization with GOOGLE_API_KEY env variable."""
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            mock_genai.Client.return_value = MagicMock()
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-google-key"}, clear=False):
                    # Ensure GEMINI_API_KEY doesn't interfere
                    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
                    env["GOOGLE_API_KEY"] = "env-google-key"
                    with patch.dict(os.environ, env, clear=True):
                        emb = GoogleEmbedding()
                        assert emb.api_key == "env-google-key"

    def test_init_with_gemini_api_key_env(self):
        """Test initialization with GEMINI_API_KEY env variable."""
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            mock_genai.Client.return_value = MagicMock()
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                env = {k: v for k, v in os.environ.items() if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
                env["GEMINI_API_KEY"] = "gemini-env-key"
                with patch.dict(os.environ, env, clear=True):
                    emb = GoogleEmbedding()
                    assert emb.api_key == "gemini-env-key"

    def test_init_no_api_key_raises(self):
        """Test that ValueError is raised when no API key available."""
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                env = {k: v for k, v in os.environ.items()
                       if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
                with patch.dict(os.environ, env, clear=True):
                    with pytest.raises(ValueError, match="Google API key"):
                        GoogleEmbedding()

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            mock_genai.Client.return_value = MagicMock()
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                emb = GoogleEmbedding(
                    model="gemini-embedding-001",
                    api_key="test-key",
                )
                assert emb.model == "gemini-embedding-001"

    def test_init_with_output_dimensionality(self):
        """Test initialization with custom output dimensionality."""
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            mock_genai.Client.return_value = MagicMock()
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                emb = GoogleEmbedding(api_key="test-key", output_dimensionality=512)
                assert emb._output_dimensionality == 512


class TestGoogleEmbeddingDimension:
    """Test GoogleEmbedding.dimension property."""

    def _make_embedding(self, model="text-embedding-004", dim=None):
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            mock_genai.Client.return_value = MagicMock()
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                return GoogleEmbedding(model=model, api_key="test-key", output_dimensionality=dim)

    def test_dimension_text_embedding_004(self):
        """Test dimension for text-embedding-004."""
        emb = self._make_embedding("text-embedding-004")
        assert emb.dimension == 768

    def test_dimension_gemini_embedding_001(self):
        """Test dimension for gemini-embedding-001."""
        emb = self._make_embedding("gemini-embedding-001")
        assert emb.dimension == 3072

    def test_dimension_embedding_001(self):
        """Test dimension for embedding-001."""
        emb = self._make_embedding("embedding-001")
        assert emb.dimension == 768

    def test_dimension_custom_dimensionality(self):
        """Test dimension when custom output_dimensionality is set."""
        emb = self._make_embedding(dim=512)
        assert emb.dimension == 512

    def test_dimension_unknown_model_default(self):
        """Test dimension defaults to 768 for unknown model."""
        emb = self._make_embedding("unknown-model-xyz")
        assert emb.dimension == 768


class TestGoogleEmbeddingAembed:
    """Test GoogleEmbedding.aembed() method."""

    def _make_embedding(self, dim=None):
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                return GoogleEmbedding(api_key="test-key", output_dimensionality=dim), mock_client

    @pytest.mark.asyncio
    async def test_aembed_returns_list(self):
        """Test that aembed returns a list of floats."""
        emb, mock_client = self._make_embedding()

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)

        with patch("google.genai.types") as mock_types:
            result = await emb.aembed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_with_dimensionality(self):
        """Test aembed with custom output dimensionality."""
        emb, mock_client = self._make_embedding(dim=256)

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 256
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)

        with patch("google.genai.types") as mock_types:
            mock_types.EmbedContentConfig.return_value = MagicMock()
            result = await emb.aembed("test text")
            assert len(result) == 256

    @pytest.mark.asyncio
    async def test_aembed_raises_runtime_error_on_exception(self):
        """Test that aembed raises RuntimeError on API failure."""
        emb, mock_client = self._make_embedding()
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(side_effect=Exception("API error"))

        with patch("google.genai.types"):
            with pytest.raises(RuntimeError, match="Google API error"):
                await emb.aembed("test text")


class TestGoogleEmbeddingAembedBatch:
    """Test GoogleEmbedding.aembed_batch() method."""

    def _make_embedding(self, dim=None):
        with patch("tenxgraph.storage.store.embedding.google_embedding.HAS_GOOGLE", True):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            with patch("tenxgraph.storage.store.embedding.google_embedding.genai", mock_genai):
                return GoogleEmbedding(api_key="test-key", output_dimensionality=dim), mock_client

    @pytest.mark.asyncio
    async def test_aembed_batch_returns_list_of_lists(self):
        """Test that aembed_batch returns list of embedding lists."""
        emb, mock_client = self._make_embedding()

        mock_emb1 = MagicMock()
        mock_emb1.values = [0.1, 0.2]
        mock_emb2 = MagicMock()
        mock_emb2.values = [0.3, 0.4]
        mock_result = MagicMock()
        mock_result.embeddings = [mock_emb1, mock_emb2]

        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)

        with patch("google.genai.types"):
            result = await emb.aembed_batch(["text1", "text2"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_aembed_batch_with_dimensionality(self):
        """Test aembed_batch with custom dimensionality."""
        emb, mock_client = self._make_embedding(dim=128)

        mock_emb = MagicMock()
        mock_emb.values = [0.1] * 128
        mock_result = MagicMock()
        mock_result.embeddings = [mock_emb]
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)

        with patch("google.genai.types") as mock_types:
            mock_types.EmbedContentConfig.return_value = MagicMock()
            result = await emb.aembed_batch(["single text"])
            assert len(result) == 1
            assert len(result[0]) == 128

    @pytest.mark.asyncio
    async def test_aembed_batch_raises_runtime_error(self):
        """Test that aembed_batch raises RuntimeError on API failure."""
        emb, mock_client = self._make_embedding()
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(side_effect=Exception("batch error"))

        with patch("google.genai.types"):
            with pytest.raises(RuntimeError, match="Google API error"):
                await emb.aembed_batch(["text1", "text2"])
