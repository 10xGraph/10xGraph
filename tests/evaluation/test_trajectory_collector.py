"""
Tests for the TrajectoryCollector.
"""

import pytest

from tenxgraph.qa.evaluation import StepType, ToolCall, TrajectoryStep
from tenxgraph.qa.evaluation.collectors.event_collector import EventCollector
from tenxgraph.qa.evaluation.collectors.trajectory_collector import TrajectoryCollector
from tenxgraph.runtime.publisher.events import Event, EventModel, EventType


def create_event(
    event: Event,
    event_type: EventType,
    data: dict | None = None,
    node_name: str = "",
) -> EventModel:
    """Helper to create EventModel instances for testing."""
    return EventModel(
        event=event,
        event_id="test_id",
        event_type=event_type,
        graph_id="test_graph",
        thread_id="test_thread",
        data=data or {},
        node_name=node_name,
    )


class TestTrajectoryCollector:
    """Tests for TrajectoryCollector."""

    def test_collector_creation(self):
        """Test creating a collector."""
        collector = TrajectoryCollector()
        assert len(collector.trajectory) == 0
        assert len(collector.tool_calls) == 0
        assert len(collector.node_visits) == 0

    @pytest.mark.asyncio
    async def test_collect_node_start_event(self):
        """Test collecting a node start event."""
        collector = TrajectoryCollector()
        
        event = create_event(
            event=Event.NODE_EXECUTION,
            event_type=EventType.START,
            node_name="agent_node",
        )
        
        await collector.on_event(event)
        
        assert len(collector.trajectory) == 1
        assert collector.trajectory[0].step_type == StepType.NODE
        assert collector.trajectory[0].name == "agent_node"
        assert len(collector.node_visits) == 1

    @pytest.mark.asyncio
    async def test_collect_tool_execution_event(self):
        """Test collecting a tool execution event (START + END)."""
        collector = TrajectoryCollector()
        
        # START event parks the tool call
        start_event = create_event(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.START,
            data={
                "function_name": "get_weather",
                "tool_call_id": "call_123",
                "args": {"location": "NYC"},
            },
        )
        await collector.on_event(start_event)
        
        # END event commits the tool call
        end_event = create_event(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.END,
            data={
                "function_name": "get_weather",
                "tool_call_id": "call_123",
                "result": "Sunny",
            },
        )
        await collector.on_event(end_event)
        
        assert len(collector.trajectory) == 1
        step = collector.trajectory[0]
        assert step.step_type == StepType.TOOL
        assert step.name == "get_weather"
        assert step.args == {"location": "NYC"}
        
        assert len(collector.tool_calls) == 1
        assert collector.tool_calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_collect_tool_result_event(self):
        """Test collecting a tool result event updates the call."""
        collector = TrajectoryCollector()
        
        # Tool start
        start_event = create_event(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.START,
            data={
                "function_name": "get_weather",
                "tool_call_id": "call_123",
                "args": {"location": "NYC"},
            },
        )
        await collector.on_event(start_event)
        
        # Tool end with result
        end_event = create_event(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.END,
            data={
                "function_name": "get_weather",
                "tool_call_id": "call_123",
                "result": {"weather": "sunny"},
            },
        )
        await collector.on_event(end_event)
        
        # Should have 1 tool call with result attached
        assert len(collector.tool_calls) == 1
        assert collector.tool_calls[0].result == {"weather": "sunny"}

    @pytest.mark.asyncio
    async def test_collect_multiple_events(self):
        """Test collecting multiple events in sequence."""
        collector = TrajectoryCollector()
        
        events = [
            create_event(Event.NODE_EXECUTION, EventType.START, node_name="agent"),
            create_event(Event.TOOL_EXECUTION, EventType.START, data={"function_name": "tool1", "args": {}}),
            create_event(Event.TOOL_EXECUTION, EventType.START, data={"function_name": "tool2", "args": {}}),
            create_event(Event.NODE_EXECUTION, EventType.START, node_name="synthesizer"),
        ]
        
        for event in events:
            await collector.on_event(event)
        
        assert len(collector.trajectory) == 4
        assert collector.node_visits == ["agent", "synthesizer"]
        assert len(collector.tool_calls) == 2

    def test_get_tool_names(self):
        """Test extracting tool names from trajectory."""
        collector = TrajectoryCollector()
        collector.tool_calls = [
            ToolCall(name="get_weather", args={}),
            ToolCall(name="get_time", args={}),
            ToolCall(name="get_weather", args={}),
        ]
        
        tool_names = collector.get_tool_names()
        assert tool_names == ["get_weather", "get_time", "get_weather"]

    def test_reset_collector(self):
        """Test resetting the collector."""
        collector = TrajectoryCollector()
        collector.trajectory = [TrajectoryStep.node("agent")]
        collector.tool_calls = [ToolCall(name="test", args={})]
        collector.node_visits = ["agent"]
        
        collector.reset()
        
        assert len(collector.trajectory) == 0
        assert len(collector.tool_calls) == 0
        assert len(collector.node_visits) == 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        collector = TrajectoryCollector()
        collector.trajectory = [TrajectoryStep.node("agent")]
        collector.tool_calls = [ToolCall(name="test", args={"x": 1})]
        collector.node_visits = ["agent"]
        
        data = collector.to_dict()
        
        assert "trajectory" in data
        assert "tool_calls" in data
        assert "node_visits" in data
        assert len(data["trajectory"]) == 1

    @pytest.mark.asyncio
    async def test_filter_non_relevant_events(self):
        """Test that only relevant events are collected."""
        collector = TrajectoryCollector()
        
        # Streaming events should be ignored
        stream_event = create_event(
            event=Event.STREAMING,
            event_type=EventType.PROGRESS,
            data={"chunk": "hello"},
        )
        await collector.on_event(stream_event)
        
        # Node events should be collected
        node_event = create_event(
            event=Event.NODE_EXECUTION,
            event_type=EventType.START,
            node_name="agent",
        )
        await collector.on_event(node_event)
        
        assert len(collector.trajectory) == 1
        assert collector.trajectory[0].name == "agent"


class TestEventCollector:
    """Tests for EventCollector (raw event collection)."""

    def test_event_collector_creation(self):
        """Test creating an event collector."""
        collector = EventCollector()
        assert len(collector.events) == 0

    @pytest.mark.asyncio
    async def test_event_collector_adds_events(self):
        """Test that event collector adds all events."""
        collector = EventCollector()
        
        event = create_event(
            event=Event.NODE_EXECUTION,
            event_type=EventType.START,
            node_name="test",
        )
        
        await collector.on_event(event)
        
        assert len(collector.events) == 1
        assert collector.events[0] == event

    def test_event_collector_filter_by_event(self):
        """Test filtering events by event type."""
        collector = EventCollector()
        collector.events = [
            create_event(Event.NODE_EXECUTION, EventType.START, node_name="a"),
            create_event(Event.TOOL_EXECUTION, EventType.START, data={"function_name": "b"}),
            create_event(Event.NODE_EXECUTION, EventType.END, node_name="a"),
        ]
        
        node_events = collector.filter_by_event(Event.NODE_EXECUTION)
        assert len(node_events) == 2
        
        tool_events = collector.filter_by_event(Event.TOOL_EXECUTION)
        assert len(tool_events) == 1

    def test_event_collector_reset(self):
        """Test resetting event collector."""
        collector = EventCollector()
        collector.events = [
            create_event(Event.NODE_EXECUTION, EventType.START, node_name="a"),
        ]
        
        collector.reset()
        assert len(collector.events) == 0
