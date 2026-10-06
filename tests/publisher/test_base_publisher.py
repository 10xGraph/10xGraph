"""
Comprehensive tests for the BasePublisher abstract class.

This module tests the BasePublisher ABC including abstract method enforcement,
interface validation, and concrete implementation patterns.
"""

import pytest
from abc import ABC
from typing import Any
from unittest.mock import AsyncMock, Mock

from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.events import EventModel, Event, EventType, ContentType


class MockPublisher(BasePublisher):
    """Concrete implementation of BasePublisher for testing."""
    
    def __init__(self, config: dict[str, Any] | None = None, should_fail: bool = False):
        super().__init__(config or {})
        self.should_fail = should_fail
        self.published_events = []
        self.close_called = False
        self.sync_close_called = False
        
        # Call counters for verification
        self.publish_call_count = 0
        self.close_call_count = 0
        self.sync_close_call_count = 0
    
    async def publish(self, event: EventModel) -> Any:
        """Mock implementation of publish."""
        self.publish_call_count += 1
        
        if self.should_fail:
            raise RuntimeError("Mock publish failure")
        
        self.published_events.append(event)
        return f"published_{event.event}_{event.event_type}"
    
    async def close(self):
        """Mock implementation of close."""
        self.close_call_count += 1
        
        if self.should_fail:
            raise RuntimeError("Mock close failure")
        
        self.close_called = True
    
    def sync_close(self):
        """Mock implementation of sync_close."""
        self.sync_close_call_count += 1
        
        if self.should_fail:
            raise RuntimeError("Mock sync_close failure")
        
        self.sync_close_called = True


class IncompletePublisher(BasePublisher):
    """Incomplete publisher implementation for testing abstract method enforcement."""
    
    # Intentionally not implementing required abstract methods
    pass


class PartiallyCompletePublisher(BasePublisher):
    """Partially complete publisher for testing missing method enforcement."""
    
    async def publish(self, event: EventModel) -> Any:
        return "published"
    
    # Missing close() and sync_close() implementations


class TestBasePublisherAbstractClass:
    """Test that BasePublisher is properly abstract."""
    
    def test_cannot_instantiate_base_publisher(self):
        """Test that BasePublisher cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePublisher({})
    
    def test_cannot_instantiate_incomplete_implementation(self):
        """Test that incomplete implementations cannot be instantiated."""
        with pytest.raises(TypeError):
            IncompletePublisher({})
    
    def test_cannot_instantiate_partially_complete_implementation(self):
        """Test that partially complete implementations cannot be instantiated."""
        with pytest.raises(TypeError):
            PartiallyCompletePublisher({})
    
    def test_is_abstract_base_class(self):
        """Test that BasePublisher is an ABC."""
        assert issubclass(BasePublisher, ABC)
        assert BasePublisher.__abstractmethods__
    
    def test_required_abstract_methods(self):
        """Test that all required abstract methods are present."""
        required_methods = {'publish', 'close', 'sync_close'}
        assert BasePublisher.__abstractmethods__ == required_methods
    
    def test_concrete_implementation_works(self):
        """Test that complete concrete implementations work."""
        # Should not raise any errors
        publisher = MockPublisher()
        assert isinstance(publisher, BasePublisher)
        assert isinstance(publisher, MockPublisher)


class TestBasePublisherInitialization:
    """Test BasePublisher initialization."""
    
    def test_initialization_with_config(self):
        """Test initialization with configuration."""
        config = {
            "url": "redis://localhost:6379",
            "format": "json",
            "timeout": 30
        }
        
        publisher = MockPublisher(config)
        
        assert publisher.config == config
        assert publisher.config["url"] == "redis://localhost:6379"
        assert publisher.config["format"] == "json"
        assert publisher.config["timeout"] == 30
    
    def test_initialization_with_empty_config(self):
        """Test initialization with empty config."""
        publisher = MockPublisher({})
        
        assert publisher.config == {}
    
    def test_initialization_with_none_config(self):
        """Test initialization with None config."""
        publisher = MockPublisher(None)
        
        assert publisher.config == {}
    
    def test_config_is_stored_reference(self):
        """Test that config is stored as reference, not copy."""
        config = {"mutable": "value"}
        publisher = MockPublisher(config)
        
        # Modifying original config should affect publisher config
        config["new_key"] = "new_value"
        assert "new_key" in publisher.config
        assert publisher.config["new_key"] == "new_value"
    
    def test_complex_config_structure(self):
        """Test initialization with complex nested config."""
        config = {
            "connection": {
                "host": "localhost",
                "port": 6379,
                "ssl": True
            },
            "features": ["pub_sub", "streams"],
            "retry": {
                "attempts": 3,
                "backoff": 1.5
            },
            "metadata": {
                "version": "1.0",
                "debug": False
            }
        }
        
        publisher = MockPublisher(config)
        
        assert publisher.config == config
        assert publisher.config["connection"]["host"] == "localhost"
        assert publisher.config["features"] == ["pub_sub", "streams"]
        assert publisher.config["retry"]["attempts"] == 3


class TestBasePublisherMethods:
    """Test BasePublisher method interfaces."""
    
    @pytest.mark.asyncio
    async def test_publish_method_interface(self):
        """Test that publish method works with EventModel."""
        publisher = MockPublisher()
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            content="test content",
            node_name="test_node"
        )
        
        result = await publisher.publish(event)
        
        assert result == "published_graph_execution_start"
        assert publisher.publish_call_count == 1
        assert len(publisher.published_events) == 1
        assert publisher.published_events[0] == event
    
    @pytest.mark.asyncio
    async def test_publish_multiple_events(self):
        """Test publishing multiple events."""
        publisher = MockPublisher()
        
        events = [
            EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START),
            EventModel(event=Event.NODE_EXECUTION, event_type=EventType.PROGRESS),
            EventModel(event=Event.TOOL_EXECUTION, event_type=EventType.RESULT),
            EventModel(event=Event.STREAMING, event_type=EventType.END)
        ]
        
        results = []
        for event in events:
            result = await publisher.publish(event)
            results.append(result)
        
        assert len(results) == 4
        assert publisher.publish_call_count == 4
        assert len(publisher.published_events) == 4
        
        # Verify all events were stored in order
        for i, event in enumerate(events):
            assert publisher.published_events[i] == event
    
    @pytest.mark.asyncio
    async def test_publish_failure(self):
        """Test publish method failure handling."""
        publisher = MockPublisher(should_fail=True)
        
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.ERROR
        )
        
        with pytest.raises(RuntimeError, match="Mock publish failure"):
            await publisher.publish(event)
        
        assert publisher.publish_call_count == 1
        assert len(publisher.published_events) == 0  # Event not stored on failure
    
    @pytest.mark.asyncio
    async def test_close_method_interface(self):
        """Test that close method works properly."""
        publisher = MockPublisher()
        
        await publisher.close()
        
        assert publisher.close_called is True
        assert publisher.close_call_count == 1
    
    @pytest.mark.asyncio
    async def test_close_multiple_calls(self):
        """Test calling close multiple times."""
        publisher = MockPublisher()
        
        await publisher.close()
        await publisher.close()
        await publisher.close()
        
        assert publisher.close_call_count == 3
    
    @pytest.mark.asyncio
    async def test_close_failure(self):
        """Test close method failure handling."""
        publisher = MockPublisher(should_fail=True)
        
        with pytest.raises(RuntimeError, match="Mock close failure"):
            await publisher.close()
        
        assert publisher.close_call_count == 1
        assert publisher.close_called is False
    
    def test_sync_close_method_interface(self):
        """Test that sync_close method works properly."""
        publisher = MockPublisher()
        
        publisher.sync_close()
        
        assert publisher.sync_close_called is True
        assert publisher.sync_close_call_count == 1
    
    def test_sync_close_multiple_calls(self):
        """Test calling sync_close multiple times."""
        publisher = MockPublisher()
        
        publisher.sync_close()
        publisher.sync_close()
        publisher.sync_close()
        
        assert publisher.sync_close_call_count == 3
    
    def test_sync_close_failure(self):
        """Test sync_close method failure handling."""
        publisher = MockPublisher(should_fail=True)
        
        with pytest.raises(RuntimeError, match="Mock sync_close failure"):
            publisher.sync_close()
        
        assert publisher.sync_close_call_count == 1
        assert publisher.sync_close_called is False


class TestBasePublisherMethodSignatures:
    """Test method signatures and annotations."""
    
    def test_publish_method_signature(self):
        """Test publish method signature."""
        import inspect
        
        # Check abstract method signature
        sig = inspect.signature(BasePublisher.publish)
        
        params = list(sig.parameters.keys())
        assert params == ['self', 'event']
        
        # Check parameter annotations
        assert sig.parameters['event'].annotation == EventModel
        
        # Check return annotation
        assert sig.return_annotation == Any
    
    def test_close_method_signature(self):
        """Test close method signature."""
        import inspect
        
        sig = inspect.signature(BasePublisher.close)
        
        params = list(sig.parameters.keys())
        assert params == ['self']
        
        # Should be async (no return annotation specified)
        assert inspect.iscoroutinefunction(BasePublisher.close)
    
    def test_sync_close_method_signature(self):
        """Test sync_close method signature."""
        import inspect
        
        sig = inspect.signature(BasePublisher.sync_close)
        
        params = list(sig.parameters.keys())
        assert params == ['self']
        
        # Should not be async
        assert not inspect.iscoroutinefunction(BasePublisher.sync_close)
    
    def test_init_method_signature(self):
        """Test __init__ method signature."""
        import inspect
        
        sig = inspect.signature(BasePublisher.__init__)
        
        params = list(sig.parameters.keys())
        assert params == ['self', 'config']
        
        # Check parameter annotation
        config_param = sig.parameters['config']
        assert 'dict' in str(config_param.annotation)


class TestBasePublisherIntegration:
    """Integration tests for BasePublisher functionality."""
    
    @pytest.mark.asyncio
    async def test_full_publisher_lifecycle(self):
        """Test complete publisher lifecycle."""
        config = {"test": "lifecycle"}
        publisher = MockPublisher(config)
        
        # Verify initialization
        assert publisher.config == config
        assert publisher.publish_call_count == 0
        
        # Publish some events
        events = [
            EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START),
            EventModel(event=Event.NODE_EXECUTION, event_type=EventType.PROGRESS),
            EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.END)
        ]
        
        for event in events:
            await publisher.publish(event)
        
        # Verify publish phase
        assert publisher.publish_call_count == 3
        assert len(publisher.published_events) == 3
        
        # Close publisher
        await publisher.close()
        
        # Verify close phase
        assert publisher.close_called is True
        assert publisher.close_call_count == 1
    
    @pytest.mark.asyncio
    async def test_publisher_with_different_event_types(self):
        """Test publisher with various event types."""
        publisher = MockPublisher()
        
        # Test all event source types
        for event_source in Event:
            event = EventModel(
                event=event_source,
                event_type=EventType.START
            )
            result = await publisher.publish(event)
            assert f"published_{event_source.value}_start" == result
        
        # Test all event type phases
        for event_type in EventType:
            event = EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=event_type
            )
            result = await publisher.publish(event)
            assert f"published_graph_execution_{event_type.value}" == result
        
        total_events = len(Event) + len(EventType)
        assert publisher.publish_call_count == total_events
    
    @pytest.mark.asyncio
    async def test_publisher_error_resilience(self):
        """Test publisher behavior with mixed success/failure."""
        publisher = MockPublisher()
        
        # Publish successful event
        event1 = EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START)
        result1 = await publisher.publish(event1)
        assert result1 is not None
        
        # Switch to failure mode
        publisher.should_fail = True
        
        # Try to publish failing event
        event2 = EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.ERROR)
        with pytest.raises(RuntimeError):
            await publisher.publish(event2)
        
        # Switch back to success mode
        publisher.should_fail = False
        
        # Publish another successful event
        event3 = EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.END)
        result3 = await publisher.publish(event3)
        assert result3 is not None
        
        # Verify state
        assert publisher.publish_call_count == 3
        assert len(publisher.published_events) == 2  # Only successful ones stored
        assert publisher.published_events[0] == event1
        assert publisher.published_events[1] == event3
    
    def test_sync_and_async_close_independence(self):
        """Test that sync_close and close are independent."""
        publisher = MockPublisher()
        
        # Call sync_close
        publisher.sync_close()
        assert publisher.sync_close_called is True
        assert publisher.close_called is False
        
        # Reset and call async close
        publisher2 = MockPublisher()
        import asyncio
        asyncio.run(publisher2.close())
        assert publisher2.close_called is True
        assert publisher2.sync_close_called is False


class TestBasePublisherEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_publish_with_minimal_event(self):
        """Test publishing event with minimal required fields."""
        publisher = MockPublisher()
        
        # Create event with only required fields
        event = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START
        )
        
        result = await publisher.publish(event)
        
        assert result is not None
        assert publisher.publish_call_count == 1
        assert len(publisher.published_events) == 1
    
    @pytest.mark.asyncio
    async def test_publish_with_maximal_event(self):
        """Test publishing event with all fields populated."""
        publisher = MockPublisher()
        
        # Create event with all fields
        event = EventModel(
            event=Event.STREAMING,
            event_type=EventType.PROGRESS,
            content="maximal content",
            content_blocks=None,  # Could add ContentBlocks here
            delta=True,
            delta_type="json",
            block_index=5,
            chunk_index=10,
            byte_offset=1024,
            data={"key": "value", "number": 42},
            content_type=[ContentType.TEXT, ContentType.DATA],
            sequence_id=999,
            node_name="maximal_node",
            run_id="custom-run-id",
            thread_id="thread-999",
            timestamp=1234567890.123,
            is_error=False,
            metadata={"custom": "metadata"}
        )
        
        result = await publisher.publish(event)
        
        assert result is not None
        assert publisher.publish_call_count == 1
        assert publisher.published_events[0] == event
    
    def test_config_mutation_after_initialization(self):
        """Test behavior when config is mutated after initialization."""
        config = {"initial": "value"}
        publisher = MockPublisher(config)
        
        # Mutate config after initialization
        config["new_key"] = "new_value"
        config["initial"] = "modified_value"
        
        # Publisher should see the changes (reference stored)
        assert publisher.config["new_key"] == "new_value"
        assert publisher.config["initial"] == "modified_value"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent publish operations."""
        import asyncio
        
        publisher = MockPublisher()
        
        # Create multiple events
        events = [
            EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START),
            EventModel(event=Event.NODE_EXECUTION, event_type=EventType.PROGRESS),
            EventModel(event=Event.TOOL_EXECUTION, event_type=EventType.RESULT),
            EventModel(event=Event.STREAMING, event_type=EventType.END)
        ]
        
        # Publish all events concurrently
        tasks = [publisher.publish(event) for event in events]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 4
        assert all(result is not None for result in results)
        assert publisher.publish_call_count == 4
        assert len(publisher.published_events) == 4
    
    @pytest.mark.asyncio
    async def test_large_event_handling(self):
        """Test handling of events with large data payloads."""
        publisher = MockPublisher()
        
        # Create event with large data
        large_data = {
            "large_text": "x" * 100000,  # 100k characters
            "large_list": list(range(10000)),  # 10k integers
            "nested": {
                f"key_{i}": f"value_{i}" * 100 for i in range(1000)
            }
        }
        
        event = EventModel(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.RESULT,
            data=large_data,
            content="Large event content " * 1000
        )
        
        result = await publisher.publish(event)
        
        assert result is not None
        assert publisher.published_events[0].data == large_data
        assert len(publisher.published_events[0].content) == len("Large event content " * 1000)