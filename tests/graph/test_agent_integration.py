"""Tests for Agent integration with StateGraph."""

import pytest

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END


pytestmark = pytest.mark.asyncio


class TestAgentIntegration:
    """Test Agent class integration with StateGraph."""

    def test_agent_can_be_added_to_graph(self):
        """Test that Agent instance can be added as a node."""
        agent = Agent(
            model="gpt-4o-mini",
            system_prompt=[{"role": "system", "content": "You are a test assistant."}],
        )

        graph = StateGraph()
        graph.add_node("agent", agent)

        assert "agent" in graph.nodes
        assert isinstance(graph.nodes["agent"].func, Agent)

    def test_agent_with_tools_can_be_added(self):
        """Test that Agent with ToolNode can be added to graph."""

        def test_tool(query: str) -> str:
            return f"Result for {query}"

        tool_node = ToolNode([test_tool])
        agent = Agent(
            model="gpt-4o-mini",
            system_prompt=[{"role": "system", "content": "You are a test assistant."}],
            tool_node=tool_node,
        )

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode([test_tool]))

        assert "agent" in graph.nodes
        assert "tools" in graph.nodes

    def test_graph_with_agent_can_compile(self):
        """Test that graph with Agent can compile successfully."""
        agent = Agent(
            model="gpt-4o-mini",
            system_prompt=[{"role": "system", "content": "You are a test assistant."}],
        )

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        compiled = graph.compile()
        assert compiled is not None

    async def test_agent_node_type_detection(self):
        """Test that Agent instances are properly detected during execution."""
        agent = Agent(
            model="gpt-4o-mini",
            system_prompt=[{"role": "system", "content": "You are a test assistant."}],
        )

        # Check that the agent's execute method exists
        assert hasattr(agent, "execute")
        assert callable(agent.execute)

    def test_agent_in_complex_graph(self):
        """Test Agent in a more complex graph structure."""

        def route(state: AgentState) -> str:
            """Simple routing function."""
            return END

        agent = Agent(
            model="gpt-4o-mini",
            system_prompt=[{"role": "system", "content": "You are a test assistant."}],
        )

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", route, {END: END})

        compiled = graph.compile()
        assert compiled is not None

    def test_multiple_agents_in_graph(self):
        """Test that multiple Agent instances can coexist in a graph."""
        agent1 = Agent(
            model="gpt-4o-mini",
            system_prompt=[{"role": "system", "content": "You are agent 1."}],
        )

        agent2 = Agent(
            model="gpt-4o-mini",
            system_prompt=[{"role": "system", "content": "You are agent 2."}],
        )

        graph = StateGraph()
        graph.add_node("agent1", agent1)
        graph.add_node("agent2", agent2)

        assert "agent1" in graph.nodes
        assert "agent2" in graph.nodes
        assert isinstance(graph.nodes["agent1"].func, Agent)
        assert isinstance(graph.nodes["agent2"].func, Agent)

    def test_agent_with_regular_function_nodes(self):
        """Test that Agent can coexist with regular function nodes."""

        def regular_function(state: AgentState, config: dict) -> dict:
            return {"state": state, "messages": [], "next_node": None}

        agent = Agent(
            model="gpt-4o-mini",
            system_prompt=[{"role": "system", "content": "You are a test assistant."}],
        )

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.add_node("function", regular_function)

        assert "agent" in graph.nodes
        assert "function" in graph.nodes
        assert isinstance(graph.nodes["agent"].func, Agent)
        assert callable(graph.nodes["function"].func)

    def test_agent_export_from_graph_module(self):
        """Test that Agent is properly exported from graph module."""
        from tenxgraph.core.graph import Agent as ExportedAgent

        assert ExportedAgent is Agent
