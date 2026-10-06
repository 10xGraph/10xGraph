"""Tests for the testing utilities module."""

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode, BaseAgent
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.store import  MemorySearchResult, MemoryType
from tenxgraph.qa.testing import MockToolRegistry, TestAgent, TestContext, InMemoryStore
from tenxgraph.utils import END


class TestBaseAgent:
    """Test the BaseAgent abstract class."""

    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent(model="test")  # type: ignore

    def test_base_agent_interface(self):
        """Test that BaseAgent defines required interface."""
        assert hasattr(BaseAgent, "execute")
        assert hasattr(BaseAgent, "_call_llm")


class TestTestAgent:
    """Test the TestAgent class."""

    def test_test_agent_creation(self):
        """Test creating a TestAgent."""
        agent = TestAgent(model="test-model", responses=["Hello!"])
        assert agent.model == "test-model"
        assert agent.responses == ["Hello!"]
        assert agent.call_count == 0

    def test_test_agent_default_values(self):
        """Test TestAgent default values."""
        agent = TestAgent()
        assert agent.model == "test-model"
        assert agent.responses == ["Test response"]
        assert agent.system_prompt == []

    @pytest.mark.asyncio
    async def test_test_agent_execute(self):
        """Test TestAgent execution."""
        agent = TestAgent(responses=["Test output"])
        state = AgentState()
        state.context = [Message.text_message("Hello", role="user")]

        result = await agent.execute(state, {})

        # Should return ModelResponseConverter
        assert result is not None
        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_test_agent_cycles_responses(self):
        """Test that TestAgent cycles through responses."""
        agent = TestAgent(responses=["First", "Second", "Third"])
        state = AgentState()
        state.context = [Message.text_message("Hello", role="user")]

        # Call multiple times
        for _ in range(3):
            await agent.execute(state, {})

        assert agent.call_count == 3

        # Check call history
        assert len(agent.call_history) == 3

    def test_test_agent_assert_called(self):
        """Test assert_called method."""
        agent = TestAgent()

        # Should fail when not called
        with pytest.raises(AssertionError):
            agent.assert_called()

    @pytest.mark.asyncio
    async def test_test_agent_assert_called_after_execute(self):
        """Test assert_called after execution."""
        agent = TestAgent(responses=["Hi"])
        state = AgentState()
        state.context = [Message.text_message("Hello", role="user")]

        await agent.execute(state, {})

        # Should pass after calling
        agent.assert_called()
        agent.assert_called_times(1)

    def test_test_agent_assert_not_called(self):
        """Test assert_not_called method."""
        agent = TestAgent()
        agent.assert_not_called()  # Should pass

    def test_test_agent_get_last_messages(self):
        """Test get_last_messages method."""
        agent = TestAgent()

        # Empty when never called
        assert agent.get_last_messages() == []

    @pytest.mark.asyncio
    async def test_test_agent_reset(self):
        """Test reset method."""
        agent = TestAgent(responses=["Test"])
        state = AgentState()
        state.context = [Message.text_message("Hello", role="user")]

        await agent.execute(state, {})
        assert agent.call_count == 1

        agent.reset()
        assert agent.call_count == 0
        assert agent.call_history == []


class TestMockToolRegistry:
    """Test the MockToolRegistry class."""

    def test_registry_creation(self):
        """Test creating a MockToolRegistry."""
        registry = MockToolRegistry()
        assert registry.functions == {}
        assert registry.calls == {}

    def test_register_tool(self):
        """Test registering a mock tool."""
        registry = MockToolRegistry()
        registry.register("get_weather", lambda city: f"Sunny in {city}")

        assert "get_weather" in registry.functions
        assert registry.functions["get_weather"].__name__ == "get_weather"

    def test_tool_call_tracking(self):
        """Test that tool calls are tracked."""
        registry = MockToolRegistry()
        registry.register("greet", lambda name: f"Hello {name}")

        # Call the tool
        result = registry.functions["greet"](name="World")

        assert result == "Hello World"
        assert registry.was_called("greet")
        assert registry.call_count("greet") == 1

    def test_get_calls(self):
        """Test getting call history."""
        registry = MockToolRegistry()
        registry.register("add", lambda a, b: a + b)

        registry.functions["add"](1, 2)
        registry.functions["add"](3, 4)

        calls = registry.get_calls("add")
        assert len(calls) == 2
        assert calls[0]["args"] == (1, 2)
        assert calls[1]["args"] == (3, 4)

    def test_assert_called(self):
        """Test assert_called method."""
        registry = MockToolRegistry()
        registry.register("test", lambda: "test")

        with pytest.raises(AssertionError):
            registry.assert_called("test")

        registry.functions["test"]()
        registry.assert_called("test")  # Should pass

    def test_assert_called_with(self):
        """Test assert_called_with method."""
        registry = MockToolRegistry()
        registry.register("search", lambda query, limit=10: f"Results for {query}")

        registry.functions["search"](query="python", limit=5)

        registry.assert_called_with("search", query="python", limit=5)

    def test_assert_call_count(self):
        """Test assert_call_count method."""
        registry = MockToolRegistry()
        registry.register("ping", lambda: "pong")

        registry.functions["ping"]()
        registry.functions["ping"]()

        registry.assert_call_count("ping", 2)

        with pytest.raises(AssertionError):
            registry.assert_call_count("ping", 3)

    def test_reset(self):
        """Test reset method."""
        registry = MockToolRegistry()
        registry.register("test", lambda: "test")
        registry.functions["test"]()

        assert registry.call_count("test") == 1

        registry.reset()
        assert registry.call_count("test") == 0
        # Functions should still exist
        assert "test" in registry.functions

    def test_clear(self):
        """Test clear method."""
        registry = MockToolRegistry()
        registry.register("test", lambda: "test")
        registry.functions["test"]()

        registry.clear()
        assert registry.functions == {}
        assert registry.calls == {}

    def test_get_tool_list(self):
        """Test get_tool_list method."""
        registry = MockToolRegistry()
        registry.register("tool1", lambda: "1")
        registry.register("tool2", lambda: "2")

        tool_list = registry.get_tool_list()
        assert len(tool_list) == 2

    def test_method_chaining(self):
        """Test that register returns self for chaining."""
        registry = MockToolRegistry()
        result = (
            registry
            .register("a", lambda: "a")
            .register("b", lambda: "b")
        )

        assert result is registry
        assert len(registry.functions) == 2


class TestInMemoryStore:
    """Test the InMemoryStore class."""

    @pytest.mark.asyncio
    async def test_store_creation(self):
        """Test creating an InMemoryStore."""
        store = InMemoryStore()
        assert store.memories == {}

    @pytest.mark.asyncio
    async def test_store_and_get(self):
        """Test storing and retrieving a memory."""
        store = InMemoryStore()
        config = {"user_id": "test-user"}

        mem_id = await store.astore(config, "Test memory content")

        result = await store.aget(config, mem_id)
        assert result is not None
        assert result.content == "Test memory content"

    @pytest.mark.asyncio
    async def test_store_with_message(self):
        """Test storing a Message."""
        store = InMemoryStore()
        config = {"user_id": "test-user"}
        message = Message.text_message("Hello from message", role="user")

        mem_id = await store.astore(config, message)

        result = await store.aget(config, mem_id)
        assert result is not None
        assert "Hello from message" in result.content

    @pytest.mark.asyncio
    async def test_search_with_text_match(self):
        """Test searching memories by text."""
        store = InMemoryStore()
        config = {"user_id": "test-user"}

        await store.astore(config, "Python programming language")
        await store.astore(config, "JavaScript for web development")
        await store.astore(config, "Python is great for data science")

        results = await store.asearch(config, "Python")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_with_preconfigured_results(self):
        """Test searching with pre-configured results."""
        store = InMemoryStore()
        config = {"user_id": "test-user"}

        # Pre-configure results
        store.set_search_results([
            MemorySearchResult(id="1", content="Pre-configured result", score=0.9)
        ])

        results = await store.asearch(config, "anything")
        assert len(results) == 1
        assert results[0].content == "Pre-configured result"

    @pytest.mark.asyncio
    async def test_update_memory(self):
        """Test updating a memory."""
        store = InMemoryStore()
        config = {"user_id": "test-user"}

        mem_id = await store.astore(config, "Original content")
        updated = await store.aupdate(config, mem_id, "Updated content")

        assert updated is True
        result = await store.aget(config, mem_id)
        assert result.content == "Updated content"

    @pytest.mark.asyncio
    async def test_delete_memory(self):
        """Test deleting a memory."""
        store = InMemoryStore()
        config = {"user_id": "test-user"}

        mem_id = await store.astore(config, "To be deleted")
        deleted = await store.adelete(config, mem_id)

        assert deleted is True
        result = await store.aget(config, mem_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all(self):
        """Test getting all memories."""
        store = InMemoryStore()
        config = {"user_id": "test-user"}

        await store.astore(config, "Memory 1")
        await store.astore(config, "Memory 2")

        results = await store.aget_all(config)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing the store."""
        store = InMemoryStore()
        config = {"user_id": "test-user"}

        await store.astore(config, "Test")
        store.set_search_results([MemorySearchResult(id="1", content="Test", score=0.9)])

        store.clear()

        assert store.memories == {}
        assert store._search_results == []


class TestOverrideNode:
    """Test the override_node functionality."""

    def test_override_node_before_compile(self):
        """Test overriding a node before compilation."""
        graph = StateGraph[AgentState](AgentState())

        def original_func(state: AgentState, config: dict):
            return [Message.text_message("Original", role="assistant")]

        def replacement_func(state: AgentState, config: dict):
            return [Message.text_message("Replaced", role="assistant")]

        graph.add_node("TEST", original_func)
        graph.set_entry_point("TEST")
        graph.add_edge("TEST", END)

        # Override before compile
        graph.override_node("TEST", replacement_func)

        assert graph.nodes["TEST"].func == replacement_func

    def test_override_nonexistent_node_raises(self):
        """Test that overriding nonexistent node raises KeyError."""
        graph = StateGraph[AgentState](AgentState())

        def some_func(state: AgentState, config: dict):
            return state

        with pytest.raises(KeyError):
            graph.override_node("NONEXISTENT", some_func)

    @pytest.mark.asyncio
    async def test_override_node_after_compile(self):
        """Test overriding a node after compilation."""
        graph = StateGraph[AgentState](AgentState())

        def original_func(state: AgentState, config: dict):
            return [Message.text_message("Original", role="assistant")]

        def replacement_func(state: AgentState, config: dict):
            return [Message.text_message("Replaced", role="assistant")]

        graph.add_node("TEST", original_func)
        graph.set_entry_point("TEST")
        graph.add_edge("TEST", END)

        compiled = graph.compile()

        # Override after compile
        compiled.override_node("TEST", replacement_func)

        # The node should be updated
        assert graph.nodes["TEST"].func == replacement_func

    @pytest.mark.asyncio
    async def test_override_with_test_agent(self):
        """Test overriding a node with TestAgent."""
        graph = StateGraph[AgentState](AgentState())

        def original_func(state: AgentState, config: dict):
            return [Message.text_message("Original", role="assistant")]

        graph.add_node("MAIN", original_func)
        graph.set_entry_point("MAIN")
        graph.add_edge("MAIN", END)

        # Create test agent
        test_agent = TestAgent(responses=["Test response!"])

        # Override with test agent
        graph.override_node("MAIN", test_agent)

        assert graph.nodes["MAIN"].func == test_agent


class TestTestContext:
    """Test the TestContext class."""

    def test_context_creation(self):
        """Test creating a TestContext."""
        ctx = TestContext()
        assert ctx.container is not None
        assert ctx.store is not None

    def test_context_manager(self):
        """Test using TestContext as context manager."""
        with TestContext() as ctx:
            assert ctx.container is not None

    def test_create_graph(self):
        """Test creating a graph through TestContext."""
        with TestContext() as ctx:
            graph = ctx.create_graph()
            assert graph is not None

    def test_create_test_agent(self):
        """Test creating a TestAgent through TestContext."""
        ctx = TestContext()
        agent = ctx.create_test_agent(responses=["Hello!"])

        assert isinstance(agent, TestAgent)
        assert agent.responses == ["Hello!"]

    def test_get_store(self):
        """Test getting the store from TestContext."""
        ctx = TestContext()
        store = ctx.get_store()

        assert isinstance(store, InMemoryStore)
        assert store is ctx.store

    def test_get_mock_tools(self):
        """Test getting mock tools registry from TestContext."""
        ctx = TestContext()
        tools = ctx.get_mock_tools()

        assert isinstance(tools, MockToolRegistry)

    def test_register_mock_tool(self):
        """Test registering mock tools through TestContext."""
        ctx = TestContext()
        ctx.register_mock_tool("test_tool", lambda x: x)

        assert ctx.get_mock_tools().was_called("test_tool") is False
        ctx.get_mock_tools().functions["test_tool"]("arg")
        assert ctx.get_mock_tools().was_called("test_tool") is True

    def test_reset(self):
        """Test resetting TestContext."""
        ctx = TestContext()
        ctx.register_mock_tool("test", lambda: "test")
        ctx.get_mock_tools().functions["test"]()

        ctx.reset()

        assert ctx.get_mock_tools().functions == {}
        assert ctx.store.memories == {}


class TestIntegration:
    """Integration tests for testing utilities."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_test_agent(self):
        """Test a complete workflow using TestAgent."""
        # Create graph with TestAgent
        test_agent = TestAgent(responses=["Hello from test agent!"])

        graph = StateGraph[AgentState](AgentState())
        graph.add_node("MAIN", test_agent)
        graph.set_entry_point("MAIN")
        graph.add_edge("MAIN", END)

        compiled = graph.compile()

        # Execute
        result = await compiled.ainvoke({
            "messages": [Message.text_message("Hi!", role="user")]
        })

        # Verify
        test_agent.assert_called()
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_mock_tools_in_workflow(self):
        """Test using MockToolRegistry with ToolNode."""
        # Create mock tools - need proper function signature for introspection
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny"

        registry = MockToolRegistry()
        registry.register("get_weather", get_weather)

        # Create ToolNode with mock tools
        tool_node = ToolNode(registry.get_tool_list())

        # Invoke a tool
        result = await tool_node.invoke(
            name="get_weather",
            args={"city": "NYC"},
            tool_call_id="test_123",
            config={},
            state=AgentState(),
        )

        # Verify tool was called
        registry.assert_called("get_weather")
        
        # Verify the result contains expected content
        assert "Sunny" in result.text()
