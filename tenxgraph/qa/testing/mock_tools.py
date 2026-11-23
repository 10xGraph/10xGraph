"""Mock tool registry for testing.

This module provides a MockToolRegistry class for tracking tool calls
during testing, making it easy to verify that tools are called with
expected arguments.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

from ._call_log import _CallLog


logger = logging.getLogger("tenxgraph.testing")


class MockToolRegistry(_CallLog):
    """Registry for managing mock tools in tests.

    Simplifies creating and tracking mock tool calls. Registered tools
    are automatically wrapped to track all calls made to them.

    Attributes:
        functions: Dictionary of registered tool functions
        calls: Dictionary tracking calls made to each tool

    Example:
        ```python
        tools = MockToolRegistry()
        tools.register("get_weather", lambda city: f"Sunny in {city}")
        tools.register("send_email", lambda **kw: "Email sent")

        tool_node = ToolNode(list(tools.functions.values()))

        # After test execution
        assert tools.was_called("get_weather")
        assert tools.call_count("send_email") == 2
        assert tools.get_calls("get_weather")[0]["kwargs"]["city"] == "NYC"
        ```
    """

    _label = "Tool"
    _args_key = "kwargs"

    def __init__(self):
        """Initialize an empty mock tool registry."""
        self.functions: dict[str, Callable] = {}
        self.calls: dict[str, list[dict[str, Any]]] = {}
        logger.debug("MockToolRegistry initialized")

    def register(
        self,
        name: str,
        mock_func: Callable,
        description: str | None = None,
    ) -> "MockToolRegistry":
        """Register a mock tool function.

        The function will be wrapped to track all calls made to it.

        Args:
            name: Name of the tool (used for tracking and as __name__)
            mock_func: The mock function to call
            description: Optional description (added to __doc__)

        Returns:
            Self for method chaining
        """

        # Wrap to track calls, preserving the original signature
        @functools.wraps(mock_func)
        def tracked_func(*args: Any, **kwargs: Any) -> Any:
            self._record_call("Tool", name, args, kwargs)
            return mock_func(*args, **kwargs)

        return self._add_tool("mock tool", name, tracked_func, description)

    def register_async(
        self,
        name: str,
        mock_func: Callable,
        description: str | None = None,
    ) -> "MockToolRegistry":
        """Register an async mock tool function.

        The function will be wrapped to track all calls made to it.

        Args:
            name: Name of the tool (used for tracking and as __name__)
            mock_func: The async mock function to call
            description: Optional description (added to __doc__)

        Returns:
            Self for method chaining
        """

        # Wrap to track calls, preserving the original signature
        @functools.wraps(mock_func)
        async def tracked_func(*args: Any, **kwargs: Any) -> Any:
            self._record_call("Async tool", name, args, kwargs)
            return await mock_func(*args, **kwargs)

        return self._add_tool("async mock tool", name, tracked_func, description)

    def _record_call(self, label: str, name: str, args: tuple, kwargs: dict[str, Any]) -> None:
        self.calls.setdefault(name, []).append({"args": args, "kwargs": kwargs})
        logger.debug("%s %s called with args=%s, kwargs=%s", label, name, args, kwargs)

    def _add_tool(
        self, kind: str, name: str, tracked_func: Callable, description: str | None
    ) -> "MockToolRegistry":
        # Override name to use the registered name
        tracked_func.__name__ = name
        if description:
            tracked_func.__doc__ = description

        self.functions[name] = tracked_func
        logger.debug("Registered %s: %s", kind, name)
        return self

    def assert_call_count(self, name: str, expected: int) -> None:
        """Assert that a tool was called exactly n times.

        Args:
            name: Name of the tool
            expected: Expected number of calls

        Raises:
            AssertionError: If call count doesn't match
        """
        actual = self.call_count(name)
        assert actual == expected, f"Tool '{name}' called {actual} times, expected {expected}"  # noqa: S101

    def reset(self) -> None:
        """Clear all call history.

        Does not remove registered functions, only clears call tracking.
        """
        self.calls.clear()
        logger.debug("MockToolRegistry call history cleared")

    def clear(self) -> None:
        """Clear both functions and call history.

        Use this for complete reset between tests.
        """
        self.functions.clear()
        self.calls.clear()
        logger.debug("MockToolRegistry completely cleared")

    def get_tool_list(self) -> list[Callable]:
        """Get list of registered tool functions.

        Use this to pass tools to ToolNode.

        Returns:
            List of tracked tool functions
        """
        return list(self.functions.values())
