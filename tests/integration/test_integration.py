"""Integration tests for TAF framework."""

from tenxgraph.core.graph import CompiledGraph, StateGraph, ToolNode
from tenxgraph.runtime.publisher import ConsolePublisher
from tenxgraph.runtime.publisher.events import Event, EventModel, EventType
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END
from tenxgraph.core.state.message import ToolResultBlock


def dummy_ai_agent(state: AgentState) -> dict:
    """Dummy AI agent that returns a simple response."""
    return {"messages": [Message.text_message("AI response", role="assistant")]}


def dummy_tool_function(input_text: str) -> str:
    """Dummy tool function."""
    return f"Tool processed: {input_text}"


class TestBasicIntegration:
    """Test basic integration scenarios."""

    def test_simple_graph_creation_and_execution(self):
        """Test creating a simple graph and basic operations."""
        # Create a state graph
        graph = StateGraph(AgentState())

        # Add nodes
        graph.add_node("ai_agent", dummy_ai_agent)

        # Set entry and finish points
        graph.set_entry_point("ai_agent")
        graph.add_edge("ai_agent", END)

        # Compile the graph
        compiled = graph.compile()

        # Test that it compiled successfully
        assert isinstance(compiled, CompiledGraph)  # noqa: S101

    def test_tool_node_integration(self):
        """Test tool node integration."""
        # Create tool node
        tool_node = ToolNode([dummy_tool_function])

        # Test tool node creation
        assert tool_node is not None  # noqa: S101

    def test_state_with_messages(self):
        """Test state handling with messages."""
        # Create state with messages
        messages = [
            Message.text_message("Hello", role="user"),
            Message.text_message("Hi there!", role="assistant"),
        ]

        state = AgentState(context=messages)

        # Test state creation and message handling
        assert len(state.context) == 2  # noqa: S101  # noqa: PLR2004
        assert state.context[0].content[0].text == "Hello"  # noqa: S101
        assert state.context[1].role == "assistant"  # noqa: S101

    def test_publisher_integration(self):
        """Test publisher integration."""
        # Create console publisher
        publisher = ConsolePublisher(config={"verbose": True})

        # Create graph with publisher
        graph = StateGraph(AgentState(), publisher=publisher)

        # Test graph creation with publisher
        assert graph is not None  # noqa: S101

    def test_complex_graph_with_multiple_nodes(self):
        """Test creating a more complex graph."""
        graph = StateGraph(AgentState())

        # Add multiple nodes
        graph.add_node("start_node", dummy_ai_agent)
        graph.add_node("tool_node", ToolNode([dummy_tool_function]))
        graph.add_node("end_node", dummy_ai_agent)

        # Add edges
        graph.set_entry_point("start_node")
        graph.add_edge("start_node", "tool_node")
        graph.add_edge("tool_node", "end_node")
        graph.add_edge("end_node", END)

        # Compile
        compiled = graph.compile()

        # Test compilation
        assert isinstance(compiled, CompiledGraph)  # noqa: S101

    def test_conditional_edges(self):
        """Test conditional edges."""

        def routing_function(state: AgentState) -> str:
            """Simple routing function."""
            if len(state.context) > 5:  # noqa: PLR2004
                return "long_path"
            return "short_path"

        graph = StateGraph(AgentState())

        # Add nodes
        graph.add_node("router", dummy_ai_agent)
        graph.add_node("short_path", dummy_ai_agent)
        graph.add_node("long_path", dummy_ai_agent)

        # Add conditional edges
        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            routing_function,
            {
                "short_path": "short_path",
                "long_path": "long_path",
            },
        )
        graph.add_edge("short_path", END)
        graph.add_edge("long_path", END)

        # Compile
        compiled = graph.compile()
        assert isinstance(compiled, CompiledGraph)  # noqa: S101

    def test_agent_state_execution_methods(self):
        """Test AgentState execution-related methods."""
        state = AgentState()

        # Test execution state methods
        assert state.is_running()  # noqa: S101
        assert not state.is_interrupted()  # noqa: S101

        # Test setting current node
        state.set_current_node("test_node")
        assert state.execution_meta.current_node == "test_node"  # noqa: S101

    def test_message_operations(self):
        """Test various message operations."""
        # Test different message creation methods
        user_msg = Message.text_message("User message", role="user")
        assert user_msg.role == "user"  # noqa: S101

        # Test tool message
        tool_msg = Message.tool_message(
            content=[ToolResultBlock(output="Tool result", call_id="1234")]
        )
        assert tool_msg.content[0].output == "Tool result"  # noqa: S101

        # Test message copy
        copied_msg = user_msg.copy()
        assert copied_msg.content == user_msg.content  # noqa: S101
        assert copied_msg.role == user_msg.role  # noqa: S101

    def test_callback_system_usage(self):
        """Test callback system usage."""
        from tenxgraph.utils import CallbackContext, CallbackManager, InvocationType

        manager = CallbackManager()

        # Test callback context creation
        context = CallbackContext(
            invocation_type=InvocationType.AI, node_name="test_node", function_name="test_function"
        )

        assert context.node_name == "test_node"  # noqa: S101
        assert context.invocation_type == InvocationType.AI  # noqa: S101

    def test_error_handling_integration(self):
        """Test error handling with custom exceptions."""
        from tenxgraph.core.exceptions import GraphError, NodeError

        # Test exception creation and inheritance
        graph_error = GraphError("Graph failed")
        node_error = NodeError("Node failed")

        assert isinstance(graph_error, Exception)  # noqa: S101
        assert isinstance(node_error, GraphError)  # noqa: S101

        # Test exception chaining
        try:
            raise NodeError("Node error") from ValueError("Original error")
        except NodeError as e:
            assert isinstance(e.__cause__, ValueError)  # noqa: S101


def test_framework_import_coverage():
    """Test importing various framework components for coverage."""
    # Import and instantiate various components
    from tenxgraph.core.graph import Edge
    from tenxgraph.core.state import ExecutionState, MessageContextManager
    from tenxgraph.utils import Command, add_messages

    # Test Edge creation
    edge = Edge(from_node="a", to_node="b")
    assert edge.from_node == "a"  # noqa: S101

    # Test Event creation
    event = EventModel(
        event=Event.GRAPH_EXECUTION,
        event_type=EventType.START,
    )
    assert event.event_type == EventType.START  # noqa: S101

    # Test ExecutionState creation
    exec_state = ExecutionState(current_node="test")
    assert exec_state.current_node == "test"  # noqa: S101

    # Test MessageContextManager
    msg_manager = MessageContextManager()
    assert msg_manager is not None  # noqa: S101

    # Test Command creation
    cmd = Command(goto="next_node")
    assert cmd.goto == "next_node"  # noqa: S101

    # Test add_messages function
    msg1 = [Message.text_message("Hello")]
    msg2 = [Message.text_message("World")]
    combined = add_messages(msg1, msg2)
    assert len(combined) == 2  # noqa: S101  # noqa: PLR2004
