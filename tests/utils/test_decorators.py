"""Unit tests for the @tool decorator and related utilities."""

import asyncio

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.utils.decorators import get_tool_metadata, has_tool_decorator, tool


class TestToolDecorator:
    """Test suite for the @tool decorator."""

    def test_tool_decorator_with_name_only(self):
        """Test decorator with just a name parameter."""

        @tool(name="test_tool")
        def sample_func(x: int) -> int:
            """Sample function."""
            return x * 2

        assert has_tool_decorator(sample_func)
        metadata = get_tool_metadata(sample_func)
        assert metadata["name"] == "test_tool"
        assert metadata["description"] == "Sample function."
        assert metadata["tags"] == set()

    def test_tool_decorator_no_args(self):
        """Test decorator without arguments (uses function name and docstring)."""

        @tool
        def my_function(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert has_tool_decorator(my_function)
        metadata = get_tool_metadata(my_function)
        assert metadata["name"] == "my_function"
        assert metadata["description"] == "Add two numbers."
        assert metadata["tags"] == set()

    def test_tool_decorator_full_metadata(self):
        """Test decorator with all metadata parameters."""

        @tool(
            name="advanced_tool",
            description="A complex tool",
            tags=["tag1", "tag2"],
            provider="test_provider",
            capabilities=["cap1", "cap2"],
            metadata={"key1": "value1", "key2": 42},
        )
        def complex_func(x: str) -> str:
            return x.upper()

        assert has_tool_decorator(complex_func)
        meta = get_tool_metadata(complex_func)
        assert meta["name"] == "advanced_tool"
        assert meta["description"] == "A complex tool"
        assert meta["tags"] == {"tag1", "tag2"}
        assert meta["provider"] == "test_provider"
        assert meta["capabilities"] == ["cap1", "cap2"]
        assert meta["metadata"] == {"key1": "value1", "key2": 42}

    def test_tool_decorator_with_list_tags(self):
        """Test that list tags are converted to sets."""

        @tool(name="test", tags=["a", "b", "c"])
        def func():
            pass

        metadata = get_tool_metadata(func)
        assert isinstance(metadata["tags"], set)
        assert metadata["tags"] == {"a", "b", "c"}

    def test_tool_decorator_with_set_tags(self):
        """Test that set tags remain as sets."""

        @tool(name="test", tags={"x", "y", "z"})
        def func():
            pass

        metadata = get_tool_metadata(func)
        assert isinstance(metadata["tags"], set)
        assert metadata["tags"] == {"x", "y", "z"}

    def test_tool_decorator_preserves_function_behavior(self):
        """Test that decorated function still works normally."""

        @tool(name="calculator")
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = add(5, 3)
        assert result == 8

    def test_tool_decorator_with_async_function(self):
        """Test decorator works with async functions."""

        @tool(name="async_tool", tags=["async"])
        async def async_func(x: int) -> int:
            """Async function."""
            await asyncio.sleep(0.01)
            return x * 2

        assert has_tool_decorator(async_func)
        metadata = get_tool_metadata(async_func)
        assert metadata["name"] == "async_tool"
        assert "async" in metadata["tags"]

        # Test it still works
        result = asyncio.run(async_func(5))
        assert result == 10

    def test_tool_decorator_without_docstring(self):
        """Test decorator with function that has no docstring."""

        @tool(name="no_doc")
        def func_without_doc(x: int) -> int:
            return x

        metadata = get_tool_metadata(func_without_doc)
        assert metadata["description"] == "No description provided."

    def test_tool_decorator_override_description(self):
        """Test that explicit description overrides docstring."""

        @tool(name="test", description="Custom description")
        def func(x: int) -> int:
            """Original docstring."""
            return x

        metadata = get_tool_metadata(func)
        assert metadata["description"] == "Custom description"

    def test_tool_decorator_invalid_input(self):
        """Test that decorator raises error for non-callable."""
        with pytest.raises(ValueError, match="can only be applied to callable"):

            @tool("invalid") # type: ignore
            class NotAFunction:  # type: ignore
                pass

    def test_has_tool_decorator_false(self):
        """Test has_tool_decorator returns False for undecorated functions."""

        def plain_function():
            pass

        assert not has_tool_decorator(plain_function)

    def test_get_tool_metadata_undecorated(self):
        """Test get_tool_metadata returns None/empty for undecorated functions."""

        def plain_function():
            pass

        metadata = get_tool_metadata(plain_function)
        assert metadata["name"] is None
        assert metadata["description"] is None
        assert metadata["tags"] == set()
        assert metadata["provider"] is None
        assert metadata["capabilities"] is None
        assert metadata["metadata"] is None


class TestToolDecoratorIntegration:
    """Test integration of decorated tools with ToolNode."""

    def test_tool_node_with_decorated_functions(self):
        """Test ToolNode recognizes and uses decorated tool metadata."""

        @tool(name="custom_add", description="Custom addition tool")
        def add(a: int, b: int) -> int:
            return a + b

        @tool(name="custom_multiply")
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        tool_node = ToolNode([add, multiply])
        tools = tool_node.get_local_tool()

        assert len(tools) == 2

        # Find the tools by name
        add_tool = next(t for t in tools if t["function"]["name"] == "custom_add")
        mult_tool = next(
            t for t in tools if t["function"]["name"] == "custom_multiply"
        )

        assert add_tool["function"]["description"] == "Custom addition tool"
        assert mult_tool["function"]["description"] == "Multiply two numbers."

    def test_tool_node_tag_filtering(self):
        """Test that ToolNode filters by tags correctly."""

        @tool(name="read_tool", tags=["database", "read"])
        def read_data(table: str) -> dict:
            return {"table": table}

        @tool(name="write_tool", tags=["database", "write"])
        def write_data(table: str, data: dict) -> bool:
            return True

        @tool(name="search_tool", tags=["web", "search"])
        def search(query: str) -> list:
            return []

        tool_node = ToolNode([read_data, write_data, search])

        # Get all tools
        all_tools = tool_node.get_local_tool()
        assert len(all_tools) == 3

        # Filter by database tag
        db_tools = tool_node.get_local_tool(tags={"database"})
        assert len(db_tools) == 2
        names = {t["function"]["name"] for t in db_tools}
        assert names == {"read_tool", "write_tool"}

        # Filter by search tag
        search_tools = tool_node.get_local_tool(tags={"search"})
        assert len(search_tools) == 1
        assert search_tools[0]["function"]["name"] == "search_tool"

        # Filter by read tag
        read_tools = tool_node.get_local_tool(tags={"read"})
        assert len(read_tools) == 1
        assert read_tools[0]["function"]["name"] == "read_tool"

    def test_tool_node_mixed_decorated_and_plain(self):
        """Test ToolNode with mix of decorated and plain functions."""

        @tool(name="decorated_func")
        def decorated(x: int) -> int:
            """Decorated function."""
            return x * 2

        def plain_func(y: int) -> int:
            """Plain function."""
            return y * 3

        tool_node = ToolNode([decorated, plain_func])
        tools = tool_node.get_local_tool()

        assert len(tools) == 2

        # Decorated function should use custom name
        decorated_tool = next(
            t for t in tools if t["function"]["name"] == "decorated_func"
        )
        assert decorated_tool is not None

        # Plain function should use function name
        plain_tool = next(t for t in tools if t["function"]["name"] == "plain_func")
        assert plain_tool is not None

    def test_tool_node_injectable_params_excluded(self):
        """Test that injectable parameters are excluded from schema."""

        @tool(name="stateful_tool")
        def stateful(
            user_input: str,
            state: AgentState | None = None,
            config: dict | None = None,
            tool_call_id: str | None = None,
        ) -> str:
            """Tool with injectable params."""
            return user_input.upper()

        tool_node = ToolNode([stateful])
        tools = tool_node.get_local_tool()

        params = tools[0]["function"]["parameters"]["properties"]

        # Only user_input should be in schema
        assert "user_input" in params
        assert "state" not in params
        assert "config" not in params
        assert "tool_call_id" not in params

    def test_tool_decorator_preserves_tool_node_functionality(self):
        """Test that decorated tools work in ToolNode execution."""

        @tool(name="test_add")
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        tool_node = ToolNode([add])

        # Tool should be callable through ToolNode
        # This test ensures the decorator doesn't break ToolNode's internal mechanisms
        tools = tool_node.get_local_tool()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "test_add"


class TestToolDecoratorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_tool_decorator_empty_tags(self):
        """Test decorator with empty tags list."""

        @tool(name="test", tags=[])
        def func():
            pass

        metadata = get_tool_metadata(func)
        assert metadata["tags"] == set()

    def test_tool_decorator_none_values(self):
        """Test decorator handles None values correctly."""

        @tool(
            name="test",
            description=None,
            provider=None,
            capabilities=None,
            metadata=None,
        )
        def func():
            """Fallback docstring."""
            pass

        metadata = get_tool_metadata(func)
        assert metadata["name"] == "test"
        assert metadata["description"] == "Fallback docstring."
        assert metadata["provider"] is None
        assert metadata["capabilities"] is None
        assert metadata["metadata"] is None

    def test_tool_decorator_complex_types(self):
        """Test decorator with complex parameter types."""

        @tool(name="complex_tool")
        def complex_func(
            items: list[str],
            mapping: dict[str, int],
            optional: str | None = None,
        ) -> dict:
            """Complex types."""
            return {}

        # Should not raise any errors
        assert has_tool_decorator(complex_func)
        metadata = get_tool_metadata(complex_func)
        assert metadata["name"] == "complex_tool"

    def test_tool_decorator_with_varargs(self):
        """Test decorator with *args and **kwargs."""

        @tool(name="varargs_tool")
        def varargs_func(required: str, *args, **kwargs) -> str:
            """Function with varargs."""
            return required

        tool_node = ToolNode([varargs_func])
        tools = tool_node.get_local_tool()

        # *args and **kwargs should be excluded from schema
        params = tools[0]["function"]["parameters"]["properties"]
        assert "required" in params
        assert "args" not in params
        assert "kwargs" not in params

    def test_multiple_decorators(self):
        """Test that tool decorator can coexist with other decorators."""

        def other_decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        @tool(name="multi_decorated")
        @other_decorator
        def multi_func(x: int) -> int:
            """Multi-decorated function."""
            return x * 2

        # Tool decorator should still work
        assert has_tool_decorator(multi_func)
        result = multi_func(5)
        assert result == 10

    def test_tool_decorator_name_fallback(self):
        """Test that name defaults to function name when not provided."""

        @tool
        def my_special_function():
            """Special function."""
            pass

        metadata = get_tool_metadata(my_special_function)
        assert metadata["name"] == "my_special_function"


class TestToolDecoratorSchemaGeneration:
    """Test schema generation for decorated tools."""

    def test_schema_uses_decorator_name(self):
        """Test that generated schema uses decorator name over function name."""

        @tool(name="custom_name")
        def original_name(x: int) -> int:
            """Test function."""
            return x

        tool_node = ToolNode([original_name])
        tools = tool_node.get_local_tool()

        assert tools[0]["function"]["name"] == "custom_name"

    def test_schema_uses_decorator_description(self):
        """Test that schema uses decorator description over docstring."""

        @tool(name="test", description="Custom description")
        def func(x: int) -> int:
            """Original docstring that should be overridden."""
            return x

        tool_node = ToolNode([func])
        tools = tool_node.get_local_tool()

        assert tools[0]["function"]["description"] == "Custom description"

    def test_schema_format_matches_openai(self):
        """Test that schema format matches OpenAI function calling format."""

        @tool(
            name="test_tool",
            description="Test description",
            tags=["test"],
        )
        def test_func(param1: str, param2: int = 42) -> str:
            return f"{param1}:{param2}"

        tool_node = ToolNode([test_func])
        tools = tool_node.get_local_tool()

        schema = tools[0]
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "param1" in params["properties"]
        assert "param2" in params["properties"]
