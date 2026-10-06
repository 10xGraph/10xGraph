"""Mock tool registry for testing.

This module provides a MockToolRegistry class for tracking tool calls
during testing, making it easy to verify that tools are called with
expected arguments.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any


logger = logging.getLogger("agentflow.testing")


class MockToolRegistry:
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
            if name not in self.calls:
                self.calls[name] = []
            self.calls[name].append({"args": args, "kwargs": kwargs})
            logger.debug("Tool %s called with args=%s, kwargs=%s", name, args, kwargs)
            return mock_func(*args, **kwargs)

        # Override name to use the registered name
        tracked_func.__name__ = name
        if description:
            tracked_func.__doc__ = description

        self.functions[name] = tracked_func
        logger.debug("Registered mock tool: %s", name)
        return self

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
            if name not in self.calls:
                self.calls[name] = []
            self.calls[name].append({"args": args, "kwargs": kwargs})
            logger.debug("Async tool %s called with args=%s, kwargs=%s", name, args, kwargs)
            return await mock_func(*args, **kwargs)

        # Override name to use the registered name
        tracked_func.__name__ = name
        if description:
            tracked_func.__doc__ = description

        self.functions[name] = tracked_func
        logger.debug("Registered async mock tool: %s", name)
        return self

    def was_called(self, name: str) -> bool:
        """Check if a tool was called at least once.

        Args:
            name: Name of the tool to check

        Returns:
            True if tool was called, False otherwise
        """
        return name in self.calls and len(self.calls[name]) > 0

    def call_count(self, name: str) -> int:
        """Get number of times a tool was called.

        Args:
            name: Name of the tool

        Returns:
            Number of calls made to the tool
        """
        return len(self.calls.get(name, []))

    def get_calls(self, name: str) -> list[dict[str, Any]]:
        """Get all calls made to a tool.

        Args:
            name: Name of the tool

        Returns:
            List of call records, each containing 'args' and 'kwargs'
        """
        return self.calls.get(name, [])

    def get_last_call(self, name: str) -> dict[str, Any] | None:
        """Get the most recent call to a tool.

        Args:
            name: Name of the tool

        Returns:
            Call record dict with 'args' and 'kwargs', or None if never called
        """
        calls = self.get_calls(name)
        return calls[-1] if calls else None

    def assert_called(self, name: str) -> None:
        """Assert that a tool was called at least once.

        Args:
            name: Name of the tool

        Raises:
            AssertionError: If the tool was never called
        """
        assert self.was_called(name), f"Tool '{name}' was never called"  # noqa: S101

    def assert_called_with(self, name: str, **expected_kwargs: Any) -> None:
        """Assert that a tool was called with specific keyword arguments.

        Checks the last call to the tool.

        Args:
            name: Name of the tool
            **expected_kwargs: Expected keyword arguments

        Raises:
            AssertionError: If tool not called or kwargs don't match
        """
        last_call = self.get_last_call(name)
        assert last_call is not None, f"Tool '{name}' was never called"  # noqa: S101

        for key, expected in expected_kwargs.items():
            actual = last_call["kwargs"].get(key)
            assert (  # noqa: S101
                actual == expected
            ), f"Tool '{name}' called with {key}={actual}, expected {key}={expected}"

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
