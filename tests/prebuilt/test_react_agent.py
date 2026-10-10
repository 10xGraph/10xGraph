"""Comprehensive tests for the React prebuilt agent."""

import pytest
from unittest.mock import Mock, patch

from tenxgraph.core.graph import ToolNode, CompiledGraph
from tenxgraph.prebuilt.agent.react import ReactAgent, _make_should_use_tools, _should_use_tools
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.core.graph.base_agent import BaseAgent


class FakeManagedAgent(BaseAgent):
    """Agent stub used to verify the model-based ReactAgent constructor path."""

    def __init__(self, model: str, tool_node: ToolNode | None = None, **kwargs):
        super().__init__(model=model, tool_node=tool_node, **kwargs)
        self._tool_node = tool_node
        self.tool_node_name = None

    def get_tool_node(self) -> ToolNode | None:
        return self._tool_node

    async def execute(self, state: AgentState, config: dict) -> AgentState:
        return state

    async def _call_llm(self, messages: list[dict], tools: list | None = None, **kwargs):
        raise NotImplementedError


class FakeToolNode:
    """ToolNode stub for constructor-only unit tests."""

    def __init__(
        self,
        tools,
        client=None,
        pass_user_info_to_mcp: bool = False,
    ):
        self.tools = list(tools)
        self.client = client
        self.pass_user_info_to_mcp = pass_user_info_to_mcp


class TestReactAgent:
    """Test the ReactAgent class."""
    
    def test_init_with_state(self):
        """Test ReactAgent initialization with custom state."""
        state = AgentState()
        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            agent = ReactAgent[AgentState](model="fake-model", provider="openai", state=state)
        assert agent is not None
        assert agent._graph is not None

    def test_init_requires_model(self):
        """ReactAgent should require a model because it owns Agent creation."""
        with pytest.raises(TypeError):
            ReactAgent[AgentState]()

    def test_constructor_builds_owned_tool_node(self):
        """ReactAgent should build ToolNode internally from tool constructor inputs."""

        def lookup_weather(location: str) -> str:
            return f"Weather for {location}"

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](
                model="fake-model",
                provider="openai",
                tools=[lookup_weather],
                system_prompt=[{"role": "system", "content": "Be helpful."}],
            )

        assert isinstance(react_agent._agent, FakeManagedAgent)
        assert react_agent._tool_node is not None
        assert "lookup_weather" in react_agent._tool_node._funcs

    def test_compile_with_internal_agent_and_tools(self):
        """ReactAgent.compile should build the ReAct graph from constructor config."""

        def lookup_weather(location: str) -> str:
            return f"Weather for {location}"

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](
                model="fake-model",
                provider="openai",
                tools=[lookup_weather],
            )

        compiled = react_agent.compile()

        assert isinstance(compiled, CompiledGraph)
        assert react_agent._main_node_name in react_agent._graph.nodes
        assert react_agent._tool_node_name in react_agent._graph.nodes

    def test_compile_with_checkpointer(self):
        """Test compiling ReactAgent with checkpointer."""
        checkpointer = Mock()

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](model="fake-model", provider="openai")

        compiled = react_agent.compile(checkpointer=checkpointer)

        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_callback_manager(self):
        """Test compiling ReactAgent with callback manager."""
        callback_manager = CallbackManager()

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](model="fake-model", provider="openai")

        compiled = react_agent.compile(callback_manager=callback_manager)

        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_interrupts(self):
        """Test compiling ReactAgent with interrupt configurations."""

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](model="fake-model", provider="openai")

        compiled = react_agent.compile(
            interrupt_before=["MAIN"],
            interrupt_after=["MAIN"],
        )

        assert isinstance(compiled, CompiledGraph)

    def test_compile_without_tools_skips_tool_node(self):
        """ReactAgent should compile a single-node graph when no tools are configured."""

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](model="fake-model", provider="openai")

        compiled = react_agent.compile()

        assert isinstance(compiled, CompiledGraph)
        assert react_agent._main_node_name in react_agent._graph.nodes
        assert react_agent._tool_node_name not in react_agent._graph.nodes

    def test_compile_with_custom_node_names(self):
        """ReactAgent should honor custom graph node names."""

        def lookup_weather(location: str) -> str:
            return f"Weather for {location}"

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](
                model="fake-model",
                provider="openai",
                tools=[lookup_weather],
                main_node_name="CUSTOM_MAIN",
                tool_node_name="CUSTOM_TOOL",
            )

        compiled = react_agent.compile()

        assert isinstance(compiled, CompiledGraph)
        assert "CUSTOM_MAIN" in react_agent._graph.nodes
        assert "CUSTOM_TOOL" in react_agent._graph.nodes

    def test_compile_forwards_media_store_and_shutdown_timeout(self):
        """ReactAgent.compile should forward graph-level options to StateGraph.compile."""

        media_store = Mock()
        compiled_graph = Mock(spec=CompiledGraph)

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](model="fake-model", provider="openai")

        with patch(
            "tenxgraph.prebuilt.agent.react.StateGraph.compile",
            autospec=True,
            return_value=compiled_graph,
        ) as compile_mock:
            result = react_agent.compile(
                media_store=media_store,
                shutdown_timeout=12.5,
            )

        assert result is compiled_graph
        assert compile_mock.call_args.kwargs["media_store"] is media_store
        assert compile_mock.call_args.kwargs["shutdown_timeout"] == 12.5

    def test_constructor_builds_tool_node_from_client_only(self):
        """Providing an MCP client alone should still build the internal ToolNode."""

        client = object()

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent), patch(
            "tenxgraph.prebuilt.agent.react.ToolNode",
            FakeToolNode,
        ):
            react_agent = ReactAgent[AgentState](
                model="fake-model",
                provider="openai",
                client=client,
                pass_user_info_to_mcp=True,
            )

        assert isinstance(react_agent._tool_node, FakeToolNode)
        assert react_agent._tool_node.tools == []
        assert react_agent._tool_node.client is client
        assert react_agent._tool_node.pass_user_info_to_mcp is True


class TestShouldUseToolsFunction:
    """Test the _should_use_tools conditional function."""
    
    def test_empty_context(self):
        """Test with empty context."""
        state = AgentState()
        result = _should_use_tools(state)
        assert result == END
        
    def test_no_context(self):
        """Test with None context."""
        state = AgentState()
        state.context = []  # Empty list instead of None
        result = _should_use_tools(state)
        assert result == END
        
    def test_assistant_with_tool_calls(self):
        """Test with assistant message that has tool calls."""
        state = AgentState()
        
        # Create a mock message with tool calls
        message = Message.text_message("I need to call a tool", role="assistant")
        message.tools_calls = [{"id": "call_123", "type": "function", "function": {"name": "test_tool"}}]
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == "TOOL"
        
    def test_assistant_without_tool_calls(self):
        """Test with assistant message without tool calls."""
        state = AgentState()
        
        message = Message.text_message("Just a regular response", role="assistant")
        message.tools_calls = []
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == END
        
    def test_tool_result_message(self):
        """Test with tool result message - routes to END (caller re-invokes MAIN via graph edges)."""
        state = AgentState()

        message = Message.text_message("Tool executed successfully", role="tool")

        state.context = [message]
        result = _should_use_tools(state)
        assert result == END
    def test_user_message(self):
        """Test with user message."""
        state = AgentState()
        
        message = Message.text_message("Hello, how are you?", role="user")
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == END
        
    def test_system_message(self):
        """Test with system message."""
        state = AgentState()
        
        message = Message.text_message("You are a helpful assistant", role="system")
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == END
        
    def test_mixed_messages_last_is_tool(self):
        """Test with multiple messages where last is tool result - routes to END."""
        state = AgentState()

        user_msg = Message.text_message("Call a tool", role="user")

        assistant_msg = Message.text_message("I'll call the tool", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_123", "type": "function"}]

        tool_msg = Message.text_message("Tool result", role="tool")

        state.context = [user_msg, assistant_msg, tool_msg]
        result = _should_use_tools(state)
        assert result == END
    def test_mixed_messages_last_is_assistant_with_tools(self):
        """Test with multiple messages where last is assistant with tool calls."""
        state = AgentState()
        
        user_msg = Message.text_message("Call a tool", role="user")
        
        assistant_msg = Message.text_message("I'll call the tool", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_123", "type": "function"}]
        
        state.context = [user_msg, assistant_msg]
        result = _should_use_tools(state)
        assert result == "TOOL"

    def test_custom_router_maps_tool_branch(self):
        """Custom tool node names should receive the tool branch from the router helper."""

        state = AgentState()
        assistant_msg = Message.text_message("Use a tool", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_123", "type": "function"}]
        state.context = [assistant_msg]

        router = _make_should_use_tools("CUSTOM_TOOL")

        assert router(state) == "CUSTOM_TOOL"

    def test_custom_router_preserves_non_tool_routes(self):
        """Custom router returns END for non-tool-call messages."""

        tool_state = AgentState()
        tool_state.context = [Message.text_message("done", role="tool")]

        end_state = AgentState()
        end_state.context = [Message.text_message("hello", role="user")]

        router = _make_should_use_tools("CUSTOM_TOOL")

        assert router(tool_state) == END
        assert router(end_state) == END


class TestReactAgentIntegration:
    """Integration tests for the ReactAgent."""

    def test_tool_node_not_created_without_tools_or_client(self):
        """ReactAgent should skip ToolNode creation when no tools or MCP client are supplied."""

        with patch("tenxgraph.prebuilt.agent.react.Agent", FakeManagedAgent):
            react_agent = ReactAgent[AgentState](
                model="fake-model",
                provider="openai",
            )

        assert react_agent._tool_node is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])