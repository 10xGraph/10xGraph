"""Test the complete event publishing system."""

import asyncio
from typing import Any

from tenxgraph.core.graph import StateGraph
from tenxgraph.runtime.publisher import BasePublisher, ConsolePublisher, Event, EventType, SourceType
from tenxgraph.core.state import AgentState
from tenxgraph.utils import END, Message


class TestPublisher(BasePublisher):
    """Test publisher that captures events for verification."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.events: list[dict[str, Any]] = []

    async def publish(self, event: Event) -> Any:
        """Store event for verification."""
        self.events.append(event.model_dump())
        return True

    def close(self):
        """Clear events."""
        self.events.clear()

    def sync_close(self):
        """Clear events."""
        self.events.clear()

    def get_events_by_type(self, event_type: EventType) -> list[dict[str, Any]]:
        """Get events filtered by type."""
        return [event for event in self.events if event.get("event_type") == event_type]

    def get_events_by_source(self, source: SourceType) -> list[dict[str, Any]]:
        """Get events filtered by source."""
        return [event for event in self.events if event.get("source") == source]


async def simple_node(state: AgentState, **kwargs) -> str:
    """Simple test node that returns a message."""
    return "Hello from simple node!"


async def error_node(state: AgentState, **kwargs) -> str:
    """Node that raises an error for testing."""
    raise ValueError("Test error from error node")


async def test_event_publishing():
    """Test that all expected events are published during graph execution."""
    print("🧪 Testing Event Publishing System")
    print("=" * 50)

    # Create test publisher
    test_publisher = TestPublisher()

    # Create a simple graph
    graph = StateGraph(publisher=test_publisher)
    graph.add_node("start_node", simple_node)
    graph.add_node("end_node", simple_node)
    graph.add_edge("start_node", "end_node")
    graph.add_edge("end_node", END)
    graph.set_entry_point("start_node")

    # Compile graph with publisher
    compiled_graph = graph.compile()

    # Test 1: Normal execution
    print("\\n📋 Test 1: Normal Graph Execution")
    print("-" * 30)

    input_data = {"messages": [Message.from_text("Test input", role="user").model_dump()]}

    try:
        result = await compiled_graph.ainvoke(input_data)
        print("✅ Graph execution completed successfully")
        print(f"📊 Total events captured: {len(test_publisher.events)}")

        # Verify graph events
        graph_events = test_publisher.get_events_by_source(SourceType.GRAPH)
        print(f"🔧 Graph events: {len(graph_events)}")

        init_events = test_publisher.get_events_by_type(EventType.INITIALIZE)
        completed_events = test_publisher.get_events_by_type(EventType.COMPLETED)
        print(f"   - Initialize events: {len(init_events)}")
        print(f"   - Completed events: {len(completed_events)}")

        # Verify node events
        node_events = test_publisher.get_events_by_source(SourceType.NODE)
        print(f"⚙️  Node events: {len(node_events)}")

        invoked_events = [e for e in node_events if e.get("event_type") == EventType.INVOKED]
        completed_node_events = [
            e for e in node_events if e.get("event_type") == EventType.COMPLETED
        ]
        print(f"   - Node invoked events: {len(invoked_events)}")
        print(f"   - Node completed events: {len(completed_node_events)}")

        # Verify state events
        state_events = test_publisher.get_events_by_source(SourceType.STATE)
        print(f"🔄 State events: {len(state_events)}")

        # Print sample events
        print("\\n📄 Sample Events:")
        for i, event in enumerate(test_publisher.events[:3]):
            print(
                f"   {i + 1}. {event.get('source')}.{event.get('event_type')} - {event.get('payload', {}).get('node_name', 'N/A')}"
            )

    except Exception as e:
        print(f"❌ Graph execution failed: {e}")

    # Test 2: Console Publisher
    print("\\n📋 Test 2: Console Publisher")
    print("-" * 30)

    console_publisher = ConsolePublisher({"format": "simple", "include_timestamp": False})
    graph = StateGraph(publisher=console_publisher)
    graph.add_node("start_node", simple_node)
    graph.add_node("end_node", simple_node)
    graph.add_edge("start_node", "end_node")
    graph.add_edge("end_node", END)
    graph.set_entry_point("start_node")
    compiled_graph_console = graph.compile()

    try:
        result = await compiled_graph_console.ainvoke(input_data)
        print("✅ Console publisher test completed")
    except Exception as e:
        print(f"❌ Console publisher test failed: {e}")

    # Test 3: Pydantic Event Model
    print("\\n📋 Test 3: Pydantic Event Model")
    print("-" * 30)

    # Test Event creation and serialization
    test_event = Event(
        source=SourceType.GRAPH,
        event_type=EventType.INITIALIZE,
        payload={"test": "data"},
        meta={"step": 1},
    )

    print(f"✅ Event created: {test_event.id}")
    print(f"📝 Event dict: {test_event.model_dump()}")

    # Test Event from dict
    event_dict = test_event.model_dump()
    restored_event = Event.from_dict(event_dict)
    print(f"✅ Event restored from dict: {restored_event.id == test_event.id}")

    print("\\n🎉 All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_event_publishing())
