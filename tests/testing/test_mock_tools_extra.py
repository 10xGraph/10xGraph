"""Additional tests for MockToolRegistry to cover register_async."""

import pytest

from tenxgraph.qa.testing.mock_tools import MockToolRegistry


class TestMockToolRegistryRegisterAsync:
    """Test MockToolRegistry.register_async() method."""

    def test_register_async_basic(self):
        """Test registering an async tool."""
        registry = MockToolRegistry()

        async def async_tool(query: str) -> str:
            return f"async result: {query}"

        result = registry.register_async("async_search", async_tool)
        assert result is registry  # method chaining
        assert "async_search" in registry.functions

    def test_register_async_with_description(self):
        """Test registering an async tool with description."""
        registry = MockToolRegistry()

        async def my_async_tool() -> str:
            return "result"

        registry.register_async("tool_with_desc", my_async_tool, description="My description")
        assert "tool_with_desc" in registry.functions
        assert registry.functions["tool_with_desc"].__doc__ == "My description"

    def test_register_async_name_set_correctly(self):
        """Test that registered async tool has correct __name__."""
        registry = MockToolRegistry()

        async def original_name() -> None:
            pass

        registry.register_async("new_name", original_name)
        assert registry.functions["new_name"].__name__ == "new_name"

    @pytest.mark.asyncio
    async def test_register_async_call_tracked(self):
        """Test that calling async tool is tracked."""
        registry = MockToolRegistry()
        call_count = []

        async def trackable_tool(x: int) -> int:
            call_count.append(1)
            return x * 2

        registry.register_async("double", trackable_tool)
        func = registry.functions["double"]
        result = await func(x=5)
        assert result == 10
        assert len(call_count) == 1
        assert registry.was_called("double") is True
        assert registry.call_count("double") == 1

    @pytest.mark.asyncio
    async def test_register_async_multiple_calls_tracked(self):
        """Test multiple async calls are all tracked."""
        registry = MockToolRegistry()

        async def simple_tool(msg: str = "") -> str:
            return f"ok: {msg}"

        registry.register_async("simple", simple_tool)
        func = registry.functions["simple"]

        await func(msg="first")
        await func(msg="second")
        await func(msg="third")

        assert registry.call_count("simple") == 3

    @pytest.mark.asyncio
    async def test_register_async_args_tracked(self):
        """Test that async call args are tracked correctly."""
        registry = MockToolRegistry()

        async def search_tool(query: str, limit: int = 10) -> list:
            return []

        registry.register_async("search", search_tool)
        func = registry.functions["search"]

        await func(query="python", limit=5)

        calls = registry.get_calls("search")
        assert len(calls) == 1
        assert calls[0]["kwargs"]["query"] == "python"
        assert calls[0]["kwargs"]["limit"] == 5

    @pytest.mark.asyncio
    async def test_register_async_get_last_call(self):
        """Test get_last_call with async tool."""
        registry = MockToolRegistry()

        async def tool(x: str = "") -> str:
            return x

        registry.register_async("tool", tool)
        func = registry.functions["tool"]

        await func(x="first")
        await func(x="last")

        last = registry.get_last_call("tool")
        assert last is not None
        assert last["kwargs"]["x"] == "last"

    def test_register_async_chaining(self):
        """Test method chaining for register_async."""
        registry = MockToolRegistry()

        async def tool1() -> None:
            pass

        async def tool2() -> None:
            pass

        result = registry.register_async("tool1", tool1).register_async("tool2", tool2)
        assert result is registry
        assert len(registry.functions) == 2

    @pytest.mark.asyncio
    async def test_register_async_assert_called(self):
        """Test assert_called with async tool."""
        registry = MockToolRegistry()

        async def tool(q: str = "") -> str:
            return q

        registry.register_async("tool", tool)
        func = registry.functions["tool"]

        with pytest.raises(AssertionError):
            registry.assert_called("tool")

        await func(q="test")
        registry.assert_called("tool")  # Should not raise

    @pytest.mark.asyncio
    async def test_register_async_reset(self):
        """Test reset clears async tool call history."""
        registry = MockToolRegistry()

        async def tool() -> None:
            pass

        registry.register_async("tool", tool)
        func = registry.functions["tool"]
        await func()
        assert registry.call_count("tool") == 1

        registry.reset()
        assert registry.call_count("tool") == 0
        assert "tool" in registry.functions

    def test_get_tool_list_with_async(self):
        """Test get_tool_list includes async tools."""
        registry = MockToolRegistry()

        def sync_tool() -> None:
            pass

        async def async_tool() -> None:
            pass

        registry.register("sync", sync_tool)
        registry.register_async("async", async_tool)

        tool_list = registry.get_tool_list()
        assert len(tool_list) == 2
