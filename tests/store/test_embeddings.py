"""
Comprehensive tests for the embedding module.

This module tests BaseEmbedding abstract class and OpenAIEmbedding implementation,
including sync/async patterns, error handling, and API integration.
"""

import os
from unittest.mock import AsyncMock, Mock, patch
import pytest

from tenxgraph.storage.store.embedding.base_embedding import BaseEmbedding
from tenxgraph.storage.store.embedding.openai_embedding import OpenAIEmbedding


class MockEmbedding(BaseEmbedding):
    """Concrete implementation of BaseEmbedding for testing."""
    
    def __init__(self, test_dimension: int = 1536):
        self._test_dimension = test_dimension
    
    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Mock implementation that returns predictable embeddings."""
        return [[i * 0.1 for i in range(self._test_dimension)] for _ in texts]
    
    async def aembed(self, text: str) -> list[float]:
        """Mock implementation that returns predictable embedding."""
        return [len(text) * 0.01 for _ in range(self._test_dimension)]
    
    @property
    def dimension(self) -> int:
        """Return the test dimension."""
        return self._test_dimension


class TestBaseEmbedding:
    """Test the BaseEmbedding abstract base class."""
    
    def test_abstract_class_cannot_instantiate(self):
        """Test that BaseEmbedding cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedding()
    
    
    def test_sync_embed_calls_async_aembed(self):
        """Test that sync embed method calls async aembed."""
        embedding = MockEmbedding(test_dimension=3)
        
        result = embedding.embed("test")
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)
        # Length of "test" is 4, so each element should be 4 * 0.01 = 0.04
        assert all(abs(x - 0.04) < 0.001 for x in result)
    
    def test_sync_embed_batch_calls_async_aembed_batch(self):
        """Test that sync embed_batch method calls async aembed_batch."""
        embedding = MockEmbedding(test_dimension=2)
        texts = ["hello", "world"]
        
        result = embedding.embed_batch(texts)
        
        assert isinstance(result, list)
        assert len(result) == 2  # Two texts
        assert all(len(emb) == 2 for emb in result)  # Each embedding has 2 dimensions
        assert all(isinstance(emb, list) for emb in result)
        assert all(all(isinstance(x, float) for x in emb) for emb in result)
    
    
    
    


class TestOpenAIEmbedding:
    """Test the OpenAIEmbedding implementation."""
    
    def test_import_error_when_openai_not_available(self):
        """Test that ImportError is raised when OpenAI is not available."""
        with patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', False):
            with pytest.raises(ImportError) as exc_info:
                OpenAIEmbedding()
            
            assert "openai" in str(exc_info.value).lower()
            assert "pip install openai" in str(exc_info.value)
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    def test_initialization_with_api_key_parameter(self, mock_openai_class):
        """Test initialization with API key provided as parameter."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        embedding = OpenAIEmbedding(api_key="test-key")

        assert embedding.api_key == "test-key"
        assert embedding.model == "text-embedding-3-small"  # default
        assert embedding.client == mock_client
        mock_openai_class.assert_called_once_with(api_key="test-key", base_url=None)
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    def test_initialization_with_environment_variable(self, mock_openai_class):
        """Test initialization with API key from environment variable."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-test-key'}):
            embedding = OpenAIEmbedding()
            
            assert embedding.api_key == "env-test-key"
    
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    def test_initialization_missing_api_key(self):
        """Test that ValueError is raised when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAIEmbedding()
            
            assert "API key must be provided" in str(exc_info.value)
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_aembed_batch_success(self, mock_openai_class):
        """Test successful batch embedding."""
        # Mock the OpenAI client and response
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response structure
        mock_data_1 = Mock()
        mock_data_1.embedding = [0.1, 0.2, 0.3]
        mock_data_2 = Mock()
        mock_data_2.embedding = [0.4, 0.5, 0.6]
        
        mock_response = Mock()
        mock_response.data = [mock_data_1, mock_data_2]
        
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        embedding = OpenAIEmbedding(api_key="test-key")
        texts = ["hello", "world"]
        
        result = await embedding.aembed_batch(texts)
        
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client.embeddings.create.assert_called_once_with(
            input=texts,
            model="text-embedding-3-small"
        )
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_aembed_single_success(self, mock_openai_class):
        """Test successful single text embedding."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        mock_data = Mock()
        mock_data.embedding = [0.7, 0.8, 0.9]
        
        mock_response = Mock()
        mock_response.data = [mock_data]
        
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        embedding = OpenAIEmbedding(api_key="test-key")
        text = "hello world"
        
        result = await embedding.aembed(text)
        
        assert result == [0.7, 0.8, 0.9]
        mock_client.embeddings.create.assert_called_once_with(
            input=text,
            model="text-embedding-3-small"
        )
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_aembed_empty_response(self, mock_openai_class):
        """Test handling of empty response data."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = []
        
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        embedding = OpenAIEmbedding(api_key="test-key")
        
        result = await embedding.aembed("test")
        
        assert result == []
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_aembed_batch_openai_error(self, mock_openai_class):
        """Test handling of OpenAI API error in batch embedding."""
        from openai import OpenAIError
        
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        mock_client.embeddings.create = AsyncMock(
            side_effect=OpenAIError("Rate limit exceeded")
        )
        
        embedding = OpenAIEmbedding(api_key="test-key")
        
        with pytest.raises(RuntimeError) as exc_info:
            await embedding.aembed_batch(["test"])
        
        assert "OpenAI API error" in str(exc_info.value)
        assert "Rate limit exceeded" in str(exc_info.value)
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_aembed_openai_error(self, mock_openai_class):
        """Test handling of OpenAI API error in single embedding."""
        from openai import OpenAIError
        
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        mock_client.embeddings.create = AsyncMock(
            side_effect=OpenAIError("Invalid model")
        )
        
        embedding = OpenAIEmbedding(api_key="test-key")
        
        with pytest.raises(RuntimeError) as exc_info:
            await embedding.aembed("test")
        
        assert "OpenAI API error" in str(exc_info.value)
        assert "Invalid model" in str(exc_info.value)
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    def test_dimension_known_models(self, mock_openai_class):
        """Test dimension property for known models."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        test_cases = [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-3-xl", 1536),
            ("text-embedding-4-base", 8192),
            ("text-embedding-4-large", 8192),
        ]
        
        for model, expected_dim in test_cases:
            embedding = OpenAIEmbedding(model=model, api_key="test-key")
            assert embedding.dimension == expected_dim
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    def test_dimension_unknown_model(self, mock_openai_class):
        """Test dimension property raises error for unknown models."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        embedding = OpenAIEmbedding(model="unknown-model", api_key="test-key")
        
        with pytest.raises(ValueError) as exc_info:
            _ = embedding.dimension
        
        assert "Unknown model 'unknown-model'" in str(exc_info.value)
        assert "Cannot determine dimension" in str(exc_info.value)
    
    @patch('tenxgraph.storage.store.embedding.openai_embedding.HAS_OPENAI', True)
    @patch('openai.AsyncOpenAI')
    def test_sync_methods_work(self, mock_openai_class):
        """Test that sync wrapper methods work correctly."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response for single embedding
        mock_data = Mock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        embedding = OpenAIEmbedding(api_key="test-key")
        
        # Test sync single embedding
        result = embedding.embed("test")
        assert result == [0.1, 0.2, 0.3]
        
        # Mock response for batch embedding
        mock_data_1 = Mock()
        mock_data_1.embedding = [0.4, 0.5, 0.6]
        mock_data_2 = Mock()
        mock_data_2.embedding = [0.7, 0.8, 0.9]
        mock_batch_response = Mock()
        mock_batch_response.data = [mock_data_1, mock_data_2]
        mock_client.embeddings.create = AsyncMock(return_value=mock_batch_response)
        
        # Test sync batch embedding
        batch_result = embedding.embed_batch(["hello", "world"])
        assert batch_result == [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]


class TestEmbeddingIntegration:
    """Integration tests for embedding classes."""
    
    
    


class TestEmbeddingEdgeCases:
    """Test edge cases and error conditions."""
    
    
    
    
    


class TestOpenAIEmbeddingBaseUrl:
    """base_url reaches AsyncOpenAI, with no network."""

    def test_base_url_is_passed_to_client(self):
        emb = OpenAIEmbedding(api_key="k", base_url="http://proxy.local/v1")
        assert str(emb.client.base_url).startswith("http://proxy.local/v1")

    def test_default_uses_openai(self):
        emb = OpenAIEmbedding(api_key="k")
        assert "api.openai.com" in str(emb.client.base_url)
