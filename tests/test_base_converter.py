"""Tests for BaseConverter abstract class."""

import pytest
from collections.abc import AsyncGenerator
from unittest.mock import Mock, AsyncMock

from tenxgraph.runtime.adapters.llm.base_converter import BaseConverter, ConverterType
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher.events import EventModel


class TestConverterType:
    """Test suite for ConverterType enum."""

    def test_converter_type_values(self):
        """Test that ConverterType has expected values."""
        assert ConverterType.OPENAI.value == "openai"
        assert ConverterType.ANTHROPIC.value == "anthropic"
        assert ConverterType.GOOGLE.value == "google"
        assert ConverterType.CUSTOM.value == "custom"
        assert ConverterType.OPENAI_RESPONSES.value == "openai_responses"

    def test_converter_type_enum_membership(self):
        """Test enum membership and comparison."""
        assert ConverterType.OPENAI in ConverterType
        assert ConverterType.OPENAI_RESPONSES != ConverterType.OPENAI
        assert len(list(ConverterType)) == 5


class TestBaseConverter:
    """Test suite for BaseConverter abstract class."""

    def test_base_converter_is_abstract(self):
        """Test that BaseConverter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseConverter()

    def test_concrete_implementation_must_implement_convert_response(self):
        """Test that concrete implementations must implement convert_response."""
        
        class IncompleteConverter(BaseConverter):
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("test", "assistant")
            # Missing convert_response implementation
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteConverter()

    def test_concrete_implementation_must_implement_convert_streaming_response(self):
        """Test that concrete implementations must implement convert_streaming_response."""
        
        class IncompleteConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("test", "assistant")
            # Missing convert_streaming_response implementation
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteConverter()

    def test_concrete_implementation_works(self):
        """Test that proper concrete implementation can be instantiated."""
        
        class ConcreteConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message(f"Converted: {response}", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message(f"Streaming: {response}", "assistant")
        
        # Should not raise any exception
        converter = ConcreteConverter()
        assert isinstance(converter, BaseConverter)

    def test_initialization_without_state(self):
        """Test BaseConverter initialization without state."""
        
        class TestConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("test", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("test", "assistant")
        
        converter = TestConverter()
        assert converter.state is None

    def test_initialization_with_state(self):
        """Test BaseConverter initialization with state."""
        
        class TestConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("test", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("test", "assistant")
        
        state = AgentState(context=[])
        converter = TestConverter(state=state)
        assert converter.state is state

    @pytest.mark.asyncio
    async def test_convert_response_functionality(self):
        """Test convert_response method implementation."""
        
        class TestConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message(f"Processed: {response}", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("test", "assistant")
        
        converter = TestConverter()
        result = await converter.convert_response("test input")
        
        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.text() == "Processed: test input"

    @pytest.mark.asyncio
    async def test_convert_streaming_response_functionality(self):
        """Test convert_streaming_response method implementation."""
        
        class TestConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("test", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message(f"Stream 1: {response}", "assistant")
                yield Message.text_message(f"Stream 2: {node_name}", "assistant")
                # Can also yield EventModel
                from tenxgraph.runtime.publisher.events import Event, EventType, ContentType
                yield EventModel(
                    event=Event.STREAMING,
                    event_type=EventType.PROGRESS,
                    content=f"Event: {config.get('test_key', 'default')}",
                    content_type=[ContentType.TEXT]
                )
        
        converter = TestConverter()
        config = {"test_key": "test_value"}
        
        results = []
        async for item in converter.convert_streaming_response(config, "test_node", "input", None):
            results.append(item)
        
        assert len(results) == 3
        assert isinstance(results[0], Message)
        assert isinstance(results[1], Message)
        assert isinstance(results[2], EventModel)
        
        assert results[0].text() == "Stream 1: input"
        assert results[1].text() == "Stream 2: test_node"
        assert results[2].content == "Event: test_value"

    @pytest.mark.asyncio
    async def test_convert_streaming_response_with_metadata(self):
        """Test convert_streaming_response with metadata parameter."""
        
        class TestConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("test", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                meta_value = meta.get("custom_key", "none") if meta else "none"
                yield Message.text_message(f"Meta: {meta_value}", "assistant")
        
        converter = TestConverter()
        meta = {"custom_key": "custom_value"}
        
        results = []
        async for item in converter.convert_streaming_response({}, "node", "input", meta):
            results.append(item)
        
        assert len(results) == 1
        assert results[0].text() == "Meta: custom_value"

    @pytest.mark.asyncio
    async def test_convert_streaming_response_without_metadata(self):
        """Test convert_streaming_response without metadata (None)."""
        
        class TestConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("test", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                meta_value = meta.get("custom_key", "none") if meta else "none"
                yield Message.text_message(f"Meta: {meta_value}", "assistant")
        
        converter = TestConverter()
        
        results = []
        async for item in converter.convert_streaming_response({}, "node", "input", None):
            results.append(item)
        
        assert len(results) == 1
        assert results[0].text() == "Meta: none"

    @pytest.mark.asyncio
    async def test_error_handling_in_convert_response(self):
        """Test error handling in convert_response implementation."""
        
        class FailingConverter(BaseConverter):
            async def convert_response(self, response):
                if response == "error":
                    raise ValueError("Conversion failed")
                return Message.text_message("success", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("test", "assistant")
        
        converter = FailingConverter()
        
        # Should work normally
        result = await converter.convert_response("normal")
        assert result.text() == "success"
        
        # Should raise error
        with pytest.raises(ValueError, match="Conversion failed"):
            await converter.convert_response("error")

    @pytest.mark.asyncio
    async def test_error_handling_in_convert_streaming_response(self):
        """Test error handling in convert_streaming_response implementation."""
        
        class FailingConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("test", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("first", "assistant")
                if response == "error":
                    raise RuntimeError("Streaming failed")
                yield Message.text_message("second", "assistant")
        
        converter = FailingConverter()
        
        # Should work normally
        results = []
        async for item in converter.convert_streaming_response({}, "node", "normal", None):
            results.append(item)
        assert len(results) == 2
        
        # Should raise error after first yield
        results = []
        with pytest.raises(RuntimeError, match="Streaming failed"):
            async for item in converter.convert_streaming_response({}, "node", "error", None):
                results.append(item)
        assert len(results) == 1  # Should get first item before error

    @pytest.mark.asyncio
    async def test_state_access_in_methods(self):
        """Test that state can be accessed within converter methods."""
        
        class StateUsingConverter(BaseConverter):
            async def convert_response(self, response):
                if self.state and self.state.context:
                    context_info = f"Context size: {len(self.state.context)}"
                else:
                    context_info = "No context"
                return Message.text_message(f"{response} - {context_info}", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                if self.state:
                    yield Message.text_message(f"State available for {response}", "assistant")
                else:
                    yield Message.text_message(f"No state for {response}", "assistant")
        
        # Test without state
        converter = StateUsingConverter()
        result = await converter.convert_response("test")
        assert result.text() == "test - No context"
        
        # Test with state
        state = AgentState(context=[
            Message.text_message("msg1", "user"),
            Message.text_message("msg2", "assistant")
        ])
        converter_with_state = StateUsingConverter(state=state)
        result = await converter_with_state.convert_response("test")
        assert result.text() == "test - Context size: 2"

    @pytest.mark.asyncio
    async def test_async_generator_return_type(self):
        """Test that convert_streaming_response returns AsyncGenerator."""
        
        class TestConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("test", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("item1", "assistant")
                yield Message.text_message("item2", "assistant")
        
        converter = TestConverter()
        generator = converter.convert_streaming_response({}, "node", "input", None)
        
        # Should return an async generator
        assert hasattr(generator, '__aiter__')
        assert hasattr(generator, '__anext__')
        
        # Should be able to iterate
        items = [item async for item in generator]
        assert len(items) == 2

    def test_converter_inheritance_structure(self):
        """Test that BaseConverter follows proper inheritance structure."""
        
        class MultiLevelConverter(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("multi-level", "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("multi-level-stream", "assistant")
        
        converter = MultiLevelConverter()
        
        # Should inherit from BaseConverter
        assert isinstance(converter, BaseConverter)
        assert hasattr(converter, 'state')
        assert hasattr(converter, 'convert_response')
        assert hasattr(converter, 'convert_streaming_response')

    @pytest.mark.asyncio
    async def test_complex_response_processing(self):
        """Test converter handling complex response objects."""
        
        class ComplexConverter(BaseConverter):
            async def convert_response(self, response):
                if isinstance(response, dict):
                    content = response.get("content", "No content")
                    role = response.get("role", "assistant")
                    return Message.text_message(content, role)
                else:
                    return Message.text_message(str(response), "assistant")
            
            async def convert_streaming_response(self, config, node_name, response, meta=None):
                if isinstance(response, list):
                    for item in response:
                        yield Message.text_message(f"Item: {item}", "assistant")
                else:
                    yield Message.text_message(f"Single: {response}", "assistant")
        
        converter = ComplexConverter()
        
        # Test dict response
        dict_response = {"content": "Hello world", "role": "user"}
        result = await converter.convert_response(dict_response)
        assert result.text() == "Hello world"
        assert result.role == "user"
        
        # Test list streaming
        list_response = ["item1", "item2", "item3"]
        results = []
        async for item in converter.convert_streaming_response({}, "node", list_response, None):
            results.append(item)
        
        assert len(results) == 3
        assert results[0].text() == "Item: item1"
        assert results[1].text() == "Item: item2"
        assert results[2].text() == "Item: item3"