"""Tests for MockMCPClient."""

import pytest
import pytest_asyncio

from tenxgraph.qa.testing.mock_mcp import MockMCPClient


class TestMockMCPClientInit:
    """Test MockMCPClient initialization."""

    def test_init_empty(self):
        """Test that MockMCPClient initializes with empty tools and calls."""
        client = MockMCPClient()
        assert client.tools == {}
        assert client.calls == {}

    def test_add_tool_basic(self):
        """Test adding a basic tool."""
        client = MockMCPClient()
        result = client.add_tool(name="search", description="Search tool")
        assert "search" in client.tools
        assert result is client  # method chaining

    def test_add_tool_with_parameters(self):
        """Test adding a tool with parameters."""
        client = MockMCPClient()
        client.add_tool(
            name="weather",
            description="Get weather",
            parameters={"city": {"type": "string"}},
        )
        tool = client.tools["weather"]
        assert tool["name"] == "weather"
        assert tool["description"] == "Get weather"
        assert "city" in tool["inputSchema"]["properties"]
        assert "city" in tool["inputSchema"]["required"]

    def test_add_tool_with_handler(self):
        """Test adding a tool with a custom handler."""
        client = MockMCPClient()
        handler = lambda city: f"Weather in {city}: Sunny"
        client.add_tool(name="weather", handler=handler)
        assert callable(client.tools["weather"]["handler"])

    def test_add_tool_default_handler(self):
        """Test that default handler is assigned when no handler provided."""
        client = MockMCPClient()
        client.add_tool(name="test_tool")
        handler = client.tools["test_tool"]["handler"]
        assert callable(handler)

    def test_add_tool_chaining(self):
        """Test method chaining for add_tool."""
        client = MockMCPClient()
        result = client.add_tool("tool1").add_tool("tool2").add_tool("tool3")
        assert result is client
        assert len(client.tools) == 3


class TestMockMCPClientListTools:
    """Test MockMCPClient.list_tools()."""

    @pytest.mark.asyncio
    async def test_list_tools_empty(self):
        """Test listing tools when none are registered."""
        client = MockMCPClient()
        tools = await client.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_list_tools_single(self):
        """Test listing a single tool."""
        client = MockMCPClient()
        client.add_tool(name="search", description="Search the web")
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search the web"
        assert "inputSchema" in tools[0]

    @pytest.mark.asyncio
    async def test_list_tools_multiple(self):
        """Test listing multiple tools."""
        client = MockMCPClient()
        client.add_tool(name="tool1", description="Tool 1")
        client.add_tool(name="tool2", description="Tool 2")
        tools = await client.list_tools()
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        assert "tool1" in names
        assert "tool2" in names


class TestMockMCPClientCallTool:
    """Test MockMCPClient.call_tool()."""

    @pytest.mark.asyncio
    async def test_call_tool_basic(self):
        """Test calling a basic tool."""
        client = MockMCPClient()
        client.add_tool(name="echo", handler=lambda msg: f"Echo: {msg}")
        result = await client.call_tool("echo", {"msg": "hello"})
        assert result == "Echo: hello"

    @pytest.mark.asyncio
    async def test_call_tool_tracks_calls(self):
        """Test that call_tool tracks calls."""
        client = MockMCPClient()
        client.add_tool(name="search", handler=lambda query: f"Results: {query}")
        await client.call_tool("search", {"query": "test"})
        assert "search" in client.calls
        assert len(client.calls["search"]) == 1
        assert client.calls["search"][0]["arguments"]["query"] == "test"

    @pytest.mark.asyncio
    async def test_call_tool_multiple_times(self):
        """Test calling a tool multiple times."""
        client = MockMCPClient()
        client.add_tool(name="counter", handler=lambda: "ok")
        await client.call_tool("counter", {})
        await client.call_tool("counter", {})
        await client.call_tool("counter", {})
        assert len(client.calls["counter"]) == 3

    @pytest.mark.asyncio
    async def test_call_tool_unknown_raises(self):
        """Test that calling unknown tool raises ValueError."""
        client = MockMCPClient()
        with pytest.raises(ValueError, match="Unknown MCP tool: nonexistent"):
            await client.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_async_handler(self):
        """Test calling a tool with an async handler."""
        client = MockMCPClient()

        async def async_handler(query: str) -> str:
            return f"Async result: {query}"

        client.add_tool(name="async_search", handler=async_handler)
        result = await client.call_tool("async_search", {"query": "hello"})
        assert result == "Async result: hello"

    @pytest.mark.asyncio
    async def test_call_tool_default_handler(self):
        """Test calling a tool with the default handler."""
        client = MockMCPClient()
        client.add_tool(name="my_tool")
        result = await client.call_tool("my_tool", {})
        assert result is not None


class TestMockMCPClientTracking:
    """Test MockMCPClient call tracking methods."""

    @pytest.mark.asyncio
    async def test_was_called_true(self):
        """Test was_called returns True after calling."""
        client = MockMCPClient()
        client.add_tool(name="tool", handler=lambda: "ok")
        await client.call_tool("tool", {})
        assert client.was_called("tool") is True

    def test_was_called_false(self):
        """Test was_called returns False when not called."""
        client = MockMCPClient()
        client.add_tool(name="tool")
        assert client.was_called("tool") is False

    def test_was_called_unregistered(self):
        """Test was_called returns False for unregistered tool."""
        client = MockMCPClient()
        assert client.was_called("nonexistent") is False

    @pytest.mark.asyncio
    async def test_call_count(self):
        """Test call_count method."""
        client = MockMCPClient()
        client.add_tool(name="tool", handler=lambda: "ok")
        assert client.call_count("tool") == 0
        await client.call_tool("tool", {})
        assert client.call_count("tool") == 1
        await client.call_tool("tool", {})
        assert client.call_count("tool") == 2

    def test_call_count_zero(self):
        """Test call_count for tool that was never called."""
        client = MockMCPClient()
        assert client.call_count("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_get_calls(self):
        """Test get_calls method."""
        client = MockMCPClient()
        client.add_tool(name="tool", handler=lambda x: x)
        await client.call_tool("tool", {"x": "first"})
        await client.call_tool("tool", {"x": "second"})
        calls = client.get_calls("tool")
        assert len(calls) == 2
        assert calls[0]["arguments"]["x"] == "first"
        assert calls[1]["arguments"]["x"] == "second"

    def test_get_calls_empty(self):
        """Test get_calls returns empty list for uncalled tool."""
        client = MockMCPClient()
        assert client.get_calls("nonexistent") == []

    @pytest.mark.asyncio
    async def test_get_last_call(self):
        """Test get_last_call method."""
        client = MockMCPClient()
        client.add_tool(name="tool", handler=lambda x: x)
        await client.call_tool("tool", {"x": "first"})
        await client.call_tool("tool", {"x": "last"})
        last = client.get_last_call("tool")
        assert last is not None
        assert last["arguments"]["x"] == "last"

    def test_get_last_call_none(self):
        """Test get_last_call returns None for uncalled tool."""
        client = MockMCPClient()
        assert client.get_last_call("nonexistent") is None


class TestMockMCPClientAssertions:
    """Test MockMCPClient assertion methods."""

    @pytest.mark.asyncio
    async def test_assert_called_passes(self):
        """Test assert_called passes after calling."""
        client = MockMCPClient()
        client.add_tool(name="tool", handler=lambda: "ok")
        await client.call_tool("tool", {})
        client.assert_called("tool")  # Should not raise

    def test_assert_called_fails(self):
        """Test assert_called fails when not called."""
        client = MockMCPClient()
        client.add_tool(name="tool")
        with pytest.raises(AssertionError):
            client.assert_called("tool")

    @pytest.mark.asyncio
    async def test_assert_called_with_passes(self):
        """Test assert_called_with passes with correct args."""
        client = MockMCPClient()
        client.add_tool(name="search", handler=lambda query: f"Results: {query}")
        await client.call_tool("search", {"query": "python"})
        client.assert_called_with("search", query="python")  # Should not raise

    @pytest.mark.asyncio
    async def test_assert_called_with_fails_wrong_args(self):
        """Test assert_called_with fails with wrong args."""
        client = MockMCPClient()
        client.add_tool(name="search", handler=lambda query: f"Results: {query}")
        await client.call_tool("search", {"query": "python"})
        with pytest.raises(AssertionError):
            client.assert_called_with("search", query="java")

    def test_assert_called_with_fails_not_called(self):
        """Test assert_called_with fails when tool not called."""
        client = MockMCPClient()
        client.add_tool(name="tool")
        with pytest.raises(AssertionError):
            client.assert_called_with("tool", arg="value")


class TestMockMCPClientReset:
    """Test MockMCPClient reset/clear methods."""

    @pytest.mark.asyncio
    async def test_reset_clears_calls(self):
        """Test reset clears call history but keeps tools."""
        client = MockMCPClient()
        client.add_tool(name="tool", handler=lambda: "ok")
        await client.call_tool("tool", {})
        assert client.call_count("tool") == 1
        client.reset()
        assert client.call_count("tool") == 0
        assert "tool" in client.tools  # Tool still registered

    @pytest.mark.asyncio
    async def test_clear_removes_everything(self):
        """Test clear removes tools and calls."""
        client = MockMCPClient()
        client.add_tool(name="tool", handler=lambda: "ok")
        await client.call_tool("tool", {})
        client.clear()
        assert len(client.tools) == 0
        assert len(client.calls) == 0


