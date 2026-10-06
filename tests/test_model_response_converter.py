"""Tests for ModelResponseConverter."""

import asyncio
import inspect
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.adapters.llm.base_converter import BaseConverter
from tenxgraph.core.state.message import Message


class TestModelResponseConverter:
    """Test suite for ModelResponseConverter."""

    @patch('tenxgraph.runtime.adapters.llm.openai_converter.OpenAIConverter')
    def test_initialization_with_string_converter(self, MockConverter):
        """Test initialization with string converter identifier."""
        response = "test response"
        
        mock_converter_instance = Mock(spec=BaseConverter)
        MockConverter.return_value = mock_converter_instance
        
        converter = ModelResponseConverter(response, "openai")
        
        assert converter.response == "test response"
        assert converter.converter == mock_converter_instance
        MockConverter.assert_called_once()

    def test_initialization_with_converter_instance(self):
        """Test initialization with BaseConverter instance."""
        response = "test response"
        mock_converter = Mock(spec=BaseConverter)
        
        converter = ModelResponseConverter(response, mock_converter)
        
        assert converter.response == "test response"
        assert converter.converter == mock_converter

    def test_initialization_with_unsupported_converter(self):
        """Test initialization fails with unsupported converter."""
        response = "test response"
        
        with pytest.raises(ValueError, match="Unsupported converter: invalid"):
            ModelResponseConverter(response, "invalid")

    def test_initialization_with_callable_response(self):
        """Test initialization with callable response."""
        def response_func():
            return "function response"
        
        mock_converter = Mock(spec=BaseConverter)
        
        converter = ModelResponseConverter(response_func, mock_converter)
        
        assert callable(converter.response)
        assert converter.converter == mock_converter

    @pytest.mark.asyncio
    async def test_invoke_with_static_response(self):
        """Test invoke method with static response."""
        response = "static response"
        mock_converter = AsyncMock(spec=BaseConverter)
        mock_message = Message.text_message("Converted response", "assistant")
        mock_converter.convert_response.return_value = mock_message
        
        converter = ModelResponseConverter(response, mock_converter)
        result = await converter.invoke()
        
        assert result == mock_message
        mock_converter.convert_response.assert_called_once_with("static response")

    @pytest.mark.asyncio
    async def test_invoke_logs_warning_for_none_response(self, caplog):
        """Test invoke warns when the wrapped response is empty."""
        mock_converter = AsyncMock(spec=BaseConverter)
        mock_converter.convert_response.return_value = Message.text_message("Converted response", "assistant")

        converter = ModelResponseConverter(None, mock_converter)

        with caplog.at_level(logging.WARNING, logger="tenxgraph.adapters"):
            await converter.invoke()

        assert "Received empty response while converting a model response" in caplog.text

    @pytest.mark.asyncio
    async def test_invoke_with_sync_callable(self):
        """Test invoke method with synchronous callable response."""
        def response_func():
            return "sync function response"
        
        mock_converter = AsyncMock(spec=BaseConverter)
        mock_message = Message.text_message("Converted sync response", "assistant")
        mock_converter.convert_response.return_value = mock_message
        
        converter = ModelResponseConverter(response_func, mock_converter)
        result = await converter.invoke()
        
        assert result == mock_message
        mock_converter.convert_response.assert_called_once_with("sync function response")

    @pytest.mark.asyncio
    async def test_invoke_with_async_callable(self):
        """Test invoke method with asynchronous callable response."""
        async def async_response_func():
            await asyncio.sleep(0.01)  # Simulate async work
            return "async function response"
        
        mock_converter = AsyncMock(spec=BaseConverter)
        mock_message = Message.text_message("Converted async response", "assistant")
        mock_converter.convert_response.return_value = mock_message
        
        converter = ModelResponseConverter(async_response_func, mock_converter)
        result = await converter.invoke()
        
        assert result == mock_message
        mock_converter.convert_response.assert_called_once_with("async function response")

    @pytest.mark.asyncio
    async def test_invoke_error_propagation(self):
        """Test that errors in callable responses are propagated."""
        def failing_response():
            raise RuntimeError("Response generation failed")
        
        mock_converter = AsyncMock(spec=BaseConverter)
        
        converter = ModelResponseConverter(failing_response, mock_converter)
        
        with pytest.raises(RuntimeError, match="Response generation failed"):
            await converter.invoke()

    @pytest.mark.asyncio
    async def test_invoke_converter_error_propagation(self):
        """Test that errors in converter are propagated."""
        response = "test response"
        mock_converter = AsyncMock(spec=BaseConverter)
        mock_converter.convert_response.side_effect = ValueError("Conversion failed")
        
        converter = ModelResponseConverter(response, mock_converter)
        
        with pytest.raises(ValueError, match="Conversion failed"):
            await converter.invoke()

    @pytest.mark.asyncio
    async def test_stream_with_static_response(self):
        """Test stream method with static response."""
        response = "stream response"
        config = {"thread_id": "test_thread"}
        node_name = "test_node"
        
        mock_converter = AsyncMock(spec=BaseConverter)

        # Create an async generator function that we can call multiple times
        async def mock_stream_generator():
            yield Message.text_message("Stream chunk 1", "assistant")
            yield Message.text_message("Stream chunk 2", "assistant")

        # Mock the method to return the generator when called
        mock_converter.convert_streaming_response = Mock(return_value=mock_stream_generator())
        
        converter = ModelResponseConverter(response, mock_converter)
        
        results = []
        async for message in converter.stream(config, node_name):
            results.append(message)
        
        assert len(results) == 2
        assert results[0].text() == "Stream chunk 1"
        assert results[1].text() == "Stream chunk 2"
        
        mock_converter.convert_streaming_response.assert_called_once_with(
            config, node_name=node_name, response="stream response", meta=None
        )

    @pytest.mark.asyncio
    async def test_stream_with_metadata(self):
        """Test stream method with metadata parameter."""
        response = "stream response"
        config = {"thread_id": "test_thread"}
        node_name = "test_node"
        meta = {"custom_key": "custom_value"}
        
        mock_converter = AsyncMock(spec=BaseConverter)

        async def mock_stream_generator():
            yield Message.text_message("Stream with meta", "assistant")

        mock_converter.convert_streaming_response = Mock(return_value=mock_stream_generator())
        
        converter = ModelResponseConverter(response, mock_converter)
        
        results = []
        async for message in converter.stream(config, node_name, meta):
            results.append(message)
        
        assert len(results) == 1
        mock_converter.convert_streaming_response.assert_called_once_with(
            config, node_name=node_name, response="stream response", meta=meta
        )

    @pytest.mark.asyncio
    async def test_stream_logs_warning_for_none_response(self, caplog):
        """Test stream warns when the wrapped response is empty."""
        config = {"thread_id": "test_thread"}
        node_name = "test_node"
        mock_converter = AsyncMock(spec=BaseConverter)

        async def mock_stream_generator():
            yield Message.text_message("Stream chunk", "assistant")

        mock_converter.convert_streaming_response = Mock(return_value=mock_stream_generator())
        converter = ModelResponseConverter(None, mock_converter)

        with caplog.at_level(logging.WARNING, logger="tenxgraph.adapters"):
            results = []
            async for message in converter.stream(config, node_name):
                results.append(message)

        assert len(results) == 1
        assert "Received empty response while starting streaming conversion" in caplog.text

    @pytest.mark.asyncio
    async def test_stream_with_sync_callable(self):
        """Test stream method with synchronous callable response."""
        def response_func():
            return "sync stream response"
        
        config = {"thread_id": "test_thread"}
        node_name = "test_node"
        
        mock_converter = AsyncMock(spec=BaseConverter)
        
        async def mock_stream_generator():
            yield Message.text_message("Sync stream result", "assistant")

        mock_converter.convert_streaming_response = Mock(return_value=mock_stream_generator())
        
        converter = ModelResponseConverter(response_func, mock_converter)
        
        results = []
        async for message in converter.stream(config, node_name):
            results.append(message)
        
        assert len(results) == 1
        mock_converter.convert_streaming_response.assert_called_once_with(
            config, node_name=node_name, response="sync stream response", meta=None
        )

    @pytest.mark.asyncio
    async def test_stream_with_async_callable(self):
        """Test stream method with asynchronous callable response."""
        async def async_response_func():
            await asyncio.sleep(0.01)
            return "async stream response"
        
        config = {"thread_id": "test_thread"}
        node_name = "test_node"
        
        mock_converter = AsyncMock(spec=BaseConverter)
        
        async def mock_stream_generator():
            yield Message.text_message("Async stream result", "assistant")

        mock_converter.convert_streaming_response = Mock(return_value=mock_stream_generator())
        
        converter = ModelResponseConverter(async_response_func, mock_converter)
        
        results = []
        async for message in converter.stream(config, node_name):
            results.append(message)
        
        assert len(results) == 1
        mock_converter.convert_streaming_response.assert_called_once_with(
            config, node_name=node_name, response="async stream response", meta=None
        )

    @pytest.mark.asyncio
    async def test_stream_missing_config(self):
        """Test stream method raises error when config is missing."""
        response = "test response"
        mock_converter = Mock(spec=BaseConverter)
        
        converter = ModelResponseConverter(response, mock_converter)
        
        with pytest.raises(ValueError, match="Config must be provided for streaming conversion"):
            async for message in converter.stream(None, "test_node"):  # type: ignore
                pass

    @pytest.mark.asyncio
    async def test_stream_empty_config(self):
        """Test stream method raises error when config is empty dict."""
        response = "test response"
        mock_converter = Mock(spec=BaseConverter)
        
        converter = ModelResponseConverter(response, mock_converter)
        
        with pytest.raises(ValueError, match="Config must be provided for streaming conversion"):
            async for message in converter.stream({}, "test_node"):
                pass

    @pytest.mark.asyncio
    async def test_stream_error_in_callable(self):
        """Test stream method handles errors in callable response."""
        def failing_response():
            raise RuntimeError("Stream response generation failed")
        
        config = {"thread_id": "test"}
        mock_converter = Mock(spec=BaseConverter)
        
        converter = ModelResponseConverter(failing_response, mock_converter)
        
        with pytest.raises(RuntimeError, match="Stream response generation failed"):
            async for message in converter.stream(config, "test_node"):
                pass

    @pytest.mark.asyncio
    async def test_stream_error_in_converter(self):
        """Test stream method handles errors in converter streaming."""
        response = "test response"
        config = {"thread_id": "test"}
        
        mock_converter = AsyncMock(spec=BaseConverter)
        
        async def error_generator():
            raise ValueError("Streaming failed")
            yield  # Unreachable, makes it a generator
        
        mock_converter.convert_streaming_response = Mock(return_value=error_generator())
        
        converter = ModelResponseConverter(response, mock_converter)
        
        with pytest.raises(ValueError, match="Streaming failed"):
            async for message in converter.stream(config, "test_node"):
                pass

    @pytest.mark.asyncio
    async def test_stream_empty_generator(self):
        """Test stream method handles empty generator from converter."""
        response = "test response"
        config = {"thread_id": "test"}
        
        mock_converter = AsyncMock(spec=BaseConverter)
        
        async def empty_generator():
            return
            yield  # Unreachable, makes it a generator

        mock_converter.convert_streaming_response = Mock(return_value=empty_generator())
        
        converter = ModelResponseConverter(response, mock_converter)
        
        results = []
        async for message in converter.stream(config, "test_node"):
            results.append(message)
        
        assert len(results) == 0

    def test_callable_detection_sync(self):
        """Test that synchronous functions are properly detected."""
        def sync_func():
            return "sync"
        
        mock_converter = Mock(spec=BaseConverter)
        converter = ModelResponseConverter(sync_func, mock_converter)
        
        # Should detect as callable
        assert callable(converter.response)
        assert not inspect.iscoroutinefunction(converter.response)

    def test_callable_detection_async(self):
        """Test that asynchronous functions are properly detected."""
        async def async_func():
            return "async"
        
        mock_converter = Mock(spec=BaseConverter)
        converter = ModelResponseConverter(async_func, mock_converter)
        
        # Should detect as callable and coroutine function
        assert callable(converter.response)
        assert inspect.iscoroutinefunction(converter.response)

    @pytest.mark.asyncio
    async def test_complex_response_object(self):
        """Test handling complex response objects."""
        complex_response = {
            "data": ["item1", "item2"],
            "metadata": {"source": "test"}
        }
        
        mock_converter = AsyncMock(spec=BaseConverter)
        mock_message = Message.text_message("Complex converted", "assistant")
        mock_converter.convert_response.return_value = mock_message
        
        converter = ModelResponseConverter(complex_response, mock_converter)
        result = await converter.invoke()
        
        assert result == mock_message
        mock_converter.convert_response.assert_called_once_with(complex_response)

    @pytest.mark.asyncio
    async def test_none_response(self):
        """Test handling None response."""
        mock_converter = AsyncMock(spec=BaseConverter)
        mock_message = Message.text_message("None handled", "assistant")
        mock_converter.convert_response.return_value = mock_message
        
        converter = ModelResponseConverter(None, mock_converter)
        result = await converter.invoke()
        
        assert result == mock_message
        mock_converter.convert_response.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_performance_with_large_response(self):
        """Test performance with large response data."""
        large_response = "x" * 10000  # 10KB of data
        
        mock_converter = AsyncMock(spec=BaseConverter)
        mock_message = Message.text_message("Large response processed", "assistant")
        mock_converter.convert_response.return_value = mock_message
        
        converter = ModelResponseConverter(large_response, mock_converter)
        
        start_time = asyncio.get_event_loop().time()
        result = await converter.invoke()
        end_time = asyncio.get_event_loop().time()
        
        assert result == mock_message
        # Should complete quickly even with large data
        assert end_time - start_time < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_invocations(self):
        """Test concurrent invoke operations."""
        responses = [f"response_{i}" for i in range(10)]
        mock_converter = AsyncMock(spec=BaseConverter)
        
        # Mock converter to return different messages
        def mock_convert(response):
            return Message.text_message(f"Converted: {response}", "assistant")
        
        mock_converter.convert_response.side_effect = mock_convert
        
        # Create converters for concurrent execution
        converters = [ModelResponseConverter(resp, mock_converter) for resp in responses]
        
        # Execute concurrently
        tasks = [converter.invoke() for converter in converters]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result.text() == f"Converted: response_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_streaming(self):
        """Test concurrent streaming operations."""
        config = {"thread_id": "test"}
        
        # Create multiple converters with individual mocks to avoid generator reuse issues
        converters = []
        for i in range(5):
            mock_converter = AsyncMock(spec=BaseConverter)
            
            async def mock_stream_gen(response_text=f"stream_{i}"):
                yield Message.text_message(f"Stream: {response_text}", "assistant")
            
            mock_converter.convert_streaming_response = Mock(return_value=mock_stream_gen())
            converters.append(ModelResponseConverter(f"stream_{i}", mock_converter))
        
        # Stream concurrently
        async def collect_stream(converter, node_name):
            results = []
            async for message in converter.stream(config, node_name):
                results.append(message)
            return results
        
        tasks = [
            collect_stream(converter, f"node_{i}") 
            for i, converter in enumerate(converters)
        ]
        
        all_results = await asyncio.gather(*tasks)
        
        assert len(all_results) == 5
        for i, results in enumerate(all_results):
            assert len(results) == 1
            assert results[0].text() == f"Stream: stream_{i}"

    @patch('tenxgraph.runtime.adapters.llm.google_genai_converter.GoogleGenAIConverter')
    def test_google_converter_import_and_creation(self, MockGoogle):
        """Test that GoogleGenAIConverter is properly imported and created."""
        mock_instance = Mock(spec=BaseConverter)
        MockGoogle.return_value = mock_instance
        
        converter = ModelResponseConverter("test", "google")
        
        MockGoogle.assert_called_once()
        assert converter.converter == mock_instance

    @pytest.mark.asyncio
    async def test_error_handling_chain(self):
        """Test error handling through the entire chain."""
        async def failing_async_response():
            await asyncio.sleep(0.01)
            raise ConnectionError("Network failure")
        
        mock_converter = AsyncMock(spec=BaseConverter)
        
        converter = ModelResponseConverter(failing_async_response, mock_converter)
        
        # Error should propagate from response function
        with pytest.raises(ConnectionError, match="Network failure"):
            await converter.invoke()
        
        # Converter should not be called if response fails
        mock_converter.convert_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_with_valid_config_variations(self):
        """Test stream with various valid config formats."""
        response = "test"
        mock_converter = AsyncMock(spec=BaseConverter)
        
        async def mock_gen():
            yield Message.text_message("test", "assistant")

        # Use side_effect to create a fresh generator for each call
        mock_converter.convert_streaming_response = Mock(side_effect=lambda *args, **kwargs: mock_gen())
        
        converter = ModelResponseConverter(response, mock_converter)
        
        # Test with different valid configs
        valid_configs = [
            {"thread_id": "test"},
            {"thread_id": "test", "extra": "value"},
            {"key": "value"},
            {"multiple": "values", "thread_id": "test", "other": 123}
        ]
        
        for config in valid_configs:
            results = []
            async for message in converter.stream(config, "test_node"):
                results.append(message)
            assert len(results) == 1
