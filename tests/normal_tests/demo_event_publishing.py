"""
Demonstration of the TAF Event Publishing System

This script shows how to use the publisher system to capture and publish
all events that occur during graph execution.
"""

import asyncio
import logging
from typing import Any

from tenxgraph.core.graph import StateGraph
from tenxgraph.runtime.publisher import BasePublisher, Event, EventType, SourceType
from tenxgraph.core.state import AgentState
from tenxgraph.utils import END, Message


# Configure logging to see the events
logging.basicConfig(level=logging.INFO)


class CustomPublisher(BasePublisher):
    """Custom publisher that demonstrates event handling."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.event_count = 0

    async def publish(self, event: Event) -> Any:
        """Handle published events."""
        self.event_count += 1
        info = event.model_dump()
        event_type = info.get("event_type", "unknown")
        source = info.get("source", "unknown")

        print(f"📡 [{self.event_count}] {source.upper()}.{event_type.upper()}")

        # Handle specific event types
        if event_type == EventType.INITIALIZE:
            print("   🚀 Graph execution started")
        elif event_type == EventType.INVOKED and source == SourceType.NODE:
            node_name = info.get("payload", {}).get("node_name", "unknown")
            print(f"   ⚙️  Executing node: {node_name}")
        elif event_type == EventType.COMPLETED and source == SourceType.NODE:
            node_name = info.get("payload", {}).get("node_name", "unknown")
            print(f"   ✅ Node completed: {node_name}")
        elif event_type == EventType.CHANGED and source == SourceType.STATE:
            payload = info.get("payload", {})
            messages_added = payload.get("messages_added", 0)
            print(f"   🔄 State updated: +{messages_added} messages")
        elif event_type == EventType.COMPLETED and source == SourceType.GRAPH:
            payload = info.get("payload", {})
            final_step = payload.get("final_step", 0)
            print(f"   🏁 Graph completed in {final_step} steps")
        elif event_type == EventType.ERROR:
            error_msg = info.get("payload", {}).get("error_message", "unknown")
            print(f"   ❌ Error occurred: {error_msg}")

        return True

    async def close(self):
        print(f"📊 Publisher closed. Total events processed: {self.event_count}")

    def sync_close(self):
        pass


async def weather_agent(state: AgentState, **kwargs) -> str:
    """Example weather agent node."""
    await asyncio.sleep(0.1)  # Simulate processing
    return "The weather is sunny today! 🌞"


async def summary_agent(state: AgentState, **kwargs) -> str:
    """Example summary agent node."""
    await asyncio.sleep(0.1)  # Simulate processing
    last_message = state.context[-1].content if state.context else "No previous message"
    return f"Summary: {last_message} Temperature is 25°C."


async def demo_event_publishing():
    """Demonstrate the complete event publishing system."""
    print("🎭 TAF Event Publishing Demo")
    print("=" * 50)

    # Create custom publisher
    custom_publisher = CustomPublisher()

    # Create a multi-agent graph
    graph = StateGraph(publisher=custom_publisher)
    graph.add_node("weather", weather_agent)
    graph.add_node("summary", summary_agent)
    graph.add_edge("weather", "summary")
    graph.add_edge("summary", END)
    graph.set_entry_point("weather")

    # Compile with publisher
    compiled_graph = graph.compile()

    print("\\n🔧 Running graph with custom publisher:")
    print("-" * 40)

    # Execute the graph
    input_data = {
        "messages": [Message.from_text("What's the weather like?", role="user").model_dump()]
    }

    try:
        result = await compiled_graph.ainvoke(input_data)
        print("\\n✨ Graph execution completed successfully!")

        # Show final result
        if result.get("context"):
            final_message = result["context"][-1].get("content", "No content")
            print(f"🎯 Final result: {final_message}")

    except Exception as e:
        print(f"💥 Error during execution: {e}")
    finally:
        await custom_publisher.close()

    print("\\n" + "=" * 50)
    print("✅ Event publishing demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_event_publishing())
