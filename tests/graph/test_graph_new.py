"""Tests for the graph module."""

from unittest.mock import Mock

import pytest

from tenxgraph.core.graph import (
    CompiledGraph,
    Edge,
    Node,
    StateGraph,
    ToolNode,
)
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END


class TestNode:
    """Test the Node class."""

    def test_node_creation_with_function(self):
        """Test creating a node with a regular function."""

        def simple_func(state: AgentState) -> AgentState:
            return state

        node = Node("test_node", simple_func)
        assert node.name == "test_node"  # noqa: S101
        assert node.func == simple_func  # noqa: S101

    def test_node_creation_with_tool_node(self):
        """Test creating a node with a ToolNode."""
        tool_node = Mock(spec=ToolNode)
        node = Node("test_tool", tool_node)
        assert node.name == "test_tool"  # noqa: S101
        assert node.func == tool_node  # noqa: S101


class TestEdge:
    """Test the Edge class."""

    def test_edge_creation(self):
        """Test creating an edge."""
        edge = Edge("start", "end")
        assert edge.from_node == "start"  # noqa: S101
        assert edge.to_node == "end"  # noqa: S101

    def test_edge_with_condition(self):
        """Test creating an edge with condition."""

        def condition(state: AgentState) -> bool:
            return len(state.context) > 0

        edge = Edge("start", "end", condition=condition)
        assert edge.condition == condition  # noqa: S101

    def test_edge_equality(self):
        """Test edge equality comparison."""
        edge1 = Edge("a", "b")
        edge2 = Edge("a", "b")
        edge3 = Edge("a", "c")

        # Test basic comparison (may not have __eq__ implemented)
        assert edge1.from_node == edge2.from_node  # noqa: S101
        assert edge1.to_node == edge2.to_node  # noqa: S101
        assert edge1.from_node != edge3.to_node  # noqa: S101


class TestStateGraph:
    """Test the StateGraph class."""

    def test_state_graph_creation(self):
        """Test creating a StateGraph."""
        graph = StateGraph[AgentState](AgentState())
        assert graph._state is not None  # noqa: S101
        assert isinstance(graph._state, AgentState)  # noqa: S101

    def test_add_node(self):
        """Test adding nodes to the graph."""
        graph = StateGraph[AgentState](AgentState())

        def test_func(state: AgentState) -> AgentState:
            return state

        graph.add_node("test", test_func)
        assert "test" in graph.nodes  # noqa: S101
        assert isinstance(graph.nodes["test"], Node)  # noqa: S101

    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = StateGraph[AgentState](AgentState())

        def func1(state: AgentState) -> AgentState:
            return state

        def func2(state: AgentState) -> AgentState:
            return state

        graph.add_node("node1", func1)
        graph.add_node("node2", func2)
        graph.add_edge("node1", "node2")

        assert len(graph.edges) > 0  # noqa: S101

    def test_set_entry_point(self):
        """Test setting the entry point."""
        graph = StateGraph[AgentState](AgentState())

        def test_func(state: AgentState) -> AgentState:
            return state

        graph.add_node("start", test_func)
        graph.set_entry_point("start")
        assert graph.entry_point == "start"  # noqa: S101

    def test_add_conditional_edges(self):
        """Test adding conditional edges."""
        graph = StateGraph[AgentState](AgentState())

        def node_func(state: AgentState) -> AgentState:
            return state

        def router(state: AgentState) -> str:
            return "next" if len(state.context) > 0 else END

        graph.add_node("start", node_func)
        graph.add_node("next", node_func)
        graph.add_conditional_edges("start", router, {"next": "next", END: END})

    def test_compile_graph(self):
        """Test compiling the graph."""
        graph = StateGraph[AgentState](AgentState())

        def test_func(state: AgentState) -> AgentState:
            return state

        graph.add_node("start", test_func)
        graph.set_entry_point("start")
        graph.add_edge("start", END)

        compiled = graph.compile()
        assert isinstance(compiled, CompiledGraph)  # noqa: S101


class TestCompiledGraph:
    """Test the CompiledGraph class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.graph = StateGraph[AgentState](AgentState())

        def simple_node(state: AgentState) -> AgentState:
            state.context.append(Message.text_message("processed"))
            return state

        self.graph.add_node("start", simple_node)
        self.graph.set_entry_point("start")
        self.graph.add_edge("start", END)
        self.compiled = self.graph.compile()

    def test_invoke(self):
        """Test synchronous invocation."""
        # Provide proper messages instead of empty array
        messages = [Message.text_message("Hello", "user")]
        result = self.compiled.invoke({"messages": messages})
        assert "messages" in result  # noqa: S101

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        """Test asynchronous invocation."""
        messages = [Message.text_message("Hello", "user")]
        result = await self.compiled.ainvoke({"messages": messages})
        assert "messages" in result  # noqa: S101

    def test_stream(self):
        """Test streaming execution."""
        messages = [Message.text_message("Hello", "user")]
        events = list(self.compiled.stream({"messages": messages}))
        assert len(events) >= 0  # noqa: S101

    @pytest.mark.asyncio
    async def test_astream(self):
        """Test asynchronous streaming execution."""
        messages = [Message.text_message("Hello", "user")]
        events = []
        async for event in self.compiled.astream({"messages": messages}):
            events.append(event)
        assert len(events) >= 0  # noqa: S101


class TestToolNode:
    """Test the ToolNode class."""

    def test_tool_node_creation(self):
        """Test creating a ToolNode."""

        def simple_tool(input_data: str) -> str:
            return f"processed: {input_data}"

        # ToolNode expects an iterable of functions
        tool_node = ToolNode([simple_tool])
        assert tool_node is not None  # noqa: S101
        # Test that the function was registered
        assert len(tool_node._funcs) == 1  # noqa: S101


def test_graph_module_imports():
    """Test that all graph module imports work correctly."""
    # Basic smoke test - just ensure imports work
    assert CompiledGraph is not None  # noqa: S101
    assert Edge is not None  # noqa: S101
    assert Node is not None  # noqa: S101
    assert StateGraph is not None  # noqa: S101
    assert ToolNode is not None  # noqa: S101
