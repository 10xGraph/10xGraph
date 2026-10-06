"""Tests for handoff tool functionality."""

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.prebuilt.tools import create_handoff_tool, is_handoff_tool
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import TextBlock
from tenxgraph.utils.constants import END


class TestHandoffToolCreation:
    """Test handoff tool creation and metadata."""

    def test_create_handoff_tool_basic(self):
        """Test creating a basic handoff tool."""
        tool = create_handoff_tool("researcher", "Transfer to researcher")

        assert tool.__name__ == "transfer_to_researcher"
        assert tool.__doc__ == "Transfer to researcher"
        assert hasattr(tool, "__handoff_tool__")
        assert tool.__handoff_tool__ is True  # type: ignore
        assert hasattr(tool, "__target_agent__")
        assert tool.__target_agent__ == "researcher"  # type: ignore

    def test_create_handoff_tool_default_description(self):
        """Test creating handoff tool with default description."""
        tool = create_handoff_tool("writer")

        assert tool.__name__ == "transfer_to_writer"
        assert tool.__doc__ is not None and "writer" in tool.__doc__.lower()
        assert tool.__doc__ is not None and "transfer" in tool.__doc__.lower()

    def test_create_handoff_tool_invalid_empty_name(self):
        """Test that empty agent_name raises ValueError."""
        with pytest.raises(ValueError, match="agent_name cannot be empty"):
            create_handoff_tool("")

    def test_create_handoff_tool_invalid_type(self):
        """Test that non-string agent_name raises TypeError."""
        with pytest.raises(TypeError, match="agent_name must be str"):
            create_handoff_tool(123)  # type: ignore

    def test_handoff_tool_execution_fallback(self):
        """Test handoff tool execution returns proper message (fallback case)."""
        tool = create_handoff_tool("researcher")

        result = tool()

        assert isinstance(result, Message)
        assert result.role == "tool"
        assert len(result.content) > 0
        assert isinstance(result.content[0], TextBlock)
        assert "researcher" in result.content[0].text.lower()


class TestHandoffToolDetection:
    """Test handoff tool name pattern detection."""

    def test_is_handoff_tool_positive(self):
        """Test detecting valid handoff tool name."""
        is_handoff, target = is_handoff_tool("transfer_to_researcher")

        assert is_handoff is True
        assert target == "researcher"

    def test_is_handoff_tool_with_underscore_in_target(self):
        """Test detecting handoff tool with underscore in target name."""
        is_handoff, target = is_handoff_tool("transfer_to_data_processor")

        assert is_handoff is True
        assert target == "data_processor"

    def test_is_handoff_tool_negative(self):
        """Test non-handoff tool name returns False."""
        is_handoff, target = is_handoff_tool("calculate")

        assert is_handoff is False
        assert target is None

    def test_is_handoff_tool_similar_prefix(self):
        """Test similar but incorrect prefix returns False."""
        is_handoff, target = is_handoff_tool("transfer_from_researcher")

        assert is_handoff is False
        assert target is None

    def test_is_handoff_tool_empty_target(self):
        """Test handoff tool with empty target returns False."""
        is_handoff, target = is_handoff_tool("transfer_to_")

        assert is_handoff is False
        assert target is None

    def test_is_handoff_tool_case_sensitive(self):
        """Test that detection is case-sensitive."""
        is_handoff, target = is_handoff_tool("Transfer_To_Researcher")

        assert is_handoff is False
        assert target is None


class TestHandoffToolSchema:
    """Test handoff tool schema generation."""

    def test_handoff_tool_schema_no_parameters(self):
        """Test that handoff tool schema has no parameters."""
        tool = create_handoff_tool("researcher", "Transfer to researcher")
        tool_node = ToolNode([tool])

        schemas = tool_node.get_local_tool()

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["function"]["name"] == "transfer_to_researcher"
        assert schema["function"]["description"] == "Transfer to researcher"
        assert schema["function"]["parameters"]["properties"] == {}
        # Required should be empty or not present
        assert schema["function"]["parameters"].get("required", []) == []

    def test_handoff_tool_mixed_with_regular_tools(self):
        """Test handoff tools alongside regular tools in schema."""

        def regular_tool(x: int) -> int:
            """Regular tool for testing."""
            return x * 2

        transfer = create_handoff_tool("researcher", "Transfer to researcher")

        tool_node = ToolNode([transfer, regular_tool])
        schemas = tool_node.get_local_tool()

        assert len(schemas) == 2

        # Find each schema
        handoff_schema = next(s for s in schemas if s["function"]["name"] == "transfer_to_researcher")
        regular_schema = next(s for s in schemas if s["function"]["name"] == "regular_tool")

        # Handoff has no parameters
        assert handoff_schema["function"]["parameters"]["properties"] == {}

        # Regular tool has its parameter
        assert "x" in regular_schema["function"]["parameters"]["properties"]
        assert "x" in regular_schema["function"]["parameters"]["required"]


@pytest.mark.asyncio
class TestHandoffIntegration:
    """Test handoff tool integration with graph execution."""

    async def test_handoff_simple_two_agents(self):
        """Test basic handoff from one agent to another."""

        # Define agents
        def agent_a(state: AgentState, config: dict):
            return {"messages": [Message.text_message("Agent A processing", role="assistant")]}

        def agent_b(state: AgentState, config: dict):
            return {"messages": [Message.text_message("Agent B processing", role="assistant")]}

        # Create handoff tool
        transfer_to_b = create_handoff_tool("agent_b", "Transfer to agent B")

        # Build graph
        graph = StateGraph()
        graph.add_node("agent_a", agent_a)
        graph.add_node("agent_a_tools", ToolNode([transfer_to_b]))
        graph.add_node("agent_b", agent_b)

        graph.set_entry_point("agent_a")
        graph.add_edge("agent_a", "agent_a_tools")
        graph.add_edge("agent_b", END)

        app = graph.compile()

        # Note: For this test to work fully, we'd need to simulate an LLM calling the handoff tool
        # This is a basic structure test
        assert app is not None

    async def test_handoff_chain_three_agents(self):
        """Test handoff chain through three agents."""

        def coordinator(state: AgentState, config: dict):
            return {"messages": [Message.text_message("Coordinator", role="assistant")]}

        def researcher(state: AgentState, config: dict):
            return {"messages": [Message.text_message("Researcher", role="assistant")]}

        def writer(state: AgentState, config: dict):
            return {"messages": [Message.text_message("Writer", role="assistant")]}

        # Create handoff tools
        transfer_to_researcher = create_handoff_tool("researcher")
        transfer_to_writer = create_handoff_tool("writer")
        transfer_to_coordinator = create_handoff_tool("coordinator")

        # Build graph
        graph = StateGraph()
        graph.add_node("coordinator", coordinator)
        graph.add_node("coordinator_tools", ToolNode([transfer_to_researcher, transfer_to_writer]))
        graph.add_node("researcher", researcher)
        graph.add_node("researcher_tools", ToolNode([transfer_to_writer, transfer_to_coordinator]))
        graph.add_node("writer", writer)
        graph.add_node("writer_tools", ToolNode([transfer_to_coordinator]))

        graph.set_entry_point("coordinator")

        # Add edges (simplified for test)
        graph.add_edge("coordinator", "coordinator_tools")
        graph.add_edge("researcher", "researcher_tools")
        graph.add_edge("writer", "writer_tools")

        app = graph.compile()

        # Structure test
        assert app is not None

    async def test_handoff_with_regular_tools(self):
        """Test handoff tools work alongside regular tools."""

        def agent_a(state: AgentState, config: dict):
            return {"messages": [Message.text_message("Agent A", role="assistant")]}

        def calculate(x: int, y: int) -> int:
            """Calculate sum."""
            return x + y

        transfer_to_b = create_handoff_tool("agent_b")

        # Tools include both handoff and regular
        tools = ToolNode([calculate, transfer_to_b])

        # Both should be in schema
        schemas = tools.get_local_tool()
        assert len(schemas) == 2

        tool_names = [s["function"]["name"] for s in schemas]
        assert "calculate" in tool_names
        assert "transfer_to_agent_b" in tool_names  # Factory prepends 'transfer_to_'


class TestHandoffToolMetadata:
    """Test handoff tool metadata and introspection."""

    def test_handoff_tool_has_markers(self):
        """Test that handoff tools have identifying markers."""
        tool = create_handoff_tool("researcher")

        # Should have special attributes
        assert hasattr(tool, "__handoff_tool__")
        assert hasattr(tool, "__target_agent__")
        assert hasattr(tool, "__name__")
        assert hasattr(tool, "__doc__")

    def test_filter_handoff_tools(self):
        """Test filtering handoff tools from a list."""

        def regular_tool(x: int) -> int:
            return x

        transfer_a = create_handoff_tool("agent_a")
        transfer_b = create_handoff_tool("agent_b")

        tools = [regular_tool, transfer_a, transfer_b]

        # Filter handoff tools
        handoff_tools = [t for t in tools if getattr(t, "__handoff_tool__", False)]

        assert len(handoff_tools) == 2
        assert transfer_a in handoff_tools
        assert transfer_b in handoff_tools
        assert regular_tool not in handoff_tools

    def test_extract_target_from_handoff_tool(self):
        """Test extracting target agent from handoff tool."""
        tool = create_handoff_tool("researcher", "Transfer to researcher")

        target = getattr(tool, "__target_agent__", None)

        assert target == "researcher"


class TestHandoffEdgeCases:
    """Test edge cases and error handling."""

    def test_handoff_tool_with_complex_agent_name(self):
        """Test handoff with complex agent names."""
        tool = create_handoff_tool("AgentWithCapsAndNumbers123")

        assert tool.__name__ == "transfer_to_AgentWithCapsAndNumbers123"
        assert tool.__target_agent__ == "AgentWithCapsAndNumbers123"  # type: ignore

        is_handoff, target = is_handoff_tool(tool.__name__)
        assert is_handoff is True
        assert target == "AgentWithCapsAndNumbers123"

    def test_multiple_handoff_tools_same_graph(self):
        """Test multiple handoff tools in the same graph."""
        transfer_a = create_handoff_tool("agent_a")
        transfer_b = create_handoff_tool("agent_b")
        transfer_c = create_handoff_tool("agent_c")

        tools = ToolNode([transfer_a, transfer_b, transfer_c])
        schemas = tools.get_local_tool()

        assert len(schemas) == 3
        names = [s["function"]["name"] for s in schemas]
        assert "transfer_to_agent_a" in names
        assert "transfer_to_agent_b" in names
        assert "transfer_to_agent_c" in names

    def test_handoff_tool_description_variations(self):
        """Test various description formats."""
        tool1 = create_handoff_tool("agent1", "Short desc")
        tool2 = create_handoff_tool("agent2", "A much longer description with details")
        tool3 = create_handoff_tool("agent3")  # Default description

        assert tool1.__doc__ is not None and len(tool1.__doc__) > 0
        assert tool2.__doc__ is not None and tool1.__doc__ is not None and len(tool2.__doc__) > len(tool1.__doc__)
        assert tool3.__doc__ is not None and "agent3" in tool3.__doc__.lower()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
