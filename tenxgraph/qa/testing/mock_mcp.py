"""Mock MCP client for testing MCP tool integrations without actual MCP servers."""

import logging
from collections.abc import Callable
from typing import Any


logger = logging.getLogger("agentflow.testing")


class MockMCPClient:
    """Mock MCP client for testing ToolNode with MCP tools.

    Simulates an MCP client without requiring actual MCP servers.
    Allows registering mock tools with custom handlers for testing.

    Attributes:
        tools: Dictionary of registered mock tools
        calls: Dictionary tracking calls made to each tool

    Example:
        ```python
        mock_client = MockMCPClient()
        mock_client.add_tool(
            name="search",
            description="Search the web",
            parameters={"query": {"type": "string"}},
            handler=lambda query: f"Results for: {query}",
        )

        # Pass to ToolNode
        tools = ToolNode([], client=mock_client)
        ```
    """

    def __init__(self):
        """Initialize an empty mock MCP client."""
        self.tools: dict[str, dict[str, Any]] = {}
        self.calls: dict[str, list[dict[str, Any]]] = {}
        logger.debug("MockMCPClient initialized")

    def add_tool(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        handler: Callable | None = None,
    ) -> "MockMCPClient":
        """Register a mock MCP tool.

        Args:
            name: Name of the tool
            description: Tool description for LLM
            parameters: JSON schema for tool parameters
            handler: Function to call when tool is invoked

        Returns:
            Self for method chaining
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": parameters or {},
                "required": list((parameters or {}).keys()),
            },
            "handler": handler or (lambda **kwargs: f"Mock result for {name}"),
        }
        logger.debug("Registered mock MCP tool: %s", name)
        return self

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools (MCP protocol method).

        Returns:
            List of tool definitions in MCP format
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            }
            for tool in self.tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name (MCP protocol method).

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool doesn't exist
        """
        if name not in self.tools:
            raise ValueError(f"Unknown MCP tool: {name}")

        # Track the call
        if name not in self.calls:
            self.calls[name] = []
        self.calls[name].append({"arguments": arguments})

        # Execute handler
        handler = self.tools[name]["handler"]
        logger.debug("Calling mock MCP tool '%s' with args: %s", name, arguments)

        if callable(handler):
            # Support both sync and async handlers
            import inspect

            if inspect.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)
        else:
            result = handler

        return result

    def was_called(self, name: str) -> bool:
        """Check if a tool was called at least once.

        Args:
            name: Name of the tool to check

        Returns:
            True if tool was called
        """
        return name in self.calls and len(self.calls[name]) > 0

    def call_count(self, name: str) -> int:
        """Get number of times a tool was called.

        Args:
            name: Name of the tool

        Returns:
            Number of calls
        """
        return len(self.calls.get(name, []))

    def get_calls(self, name: str) -> list[dict[str, Any]]:
        """Get all calls made to a tool.

        Args:
            name: Name of the tool

        Returns:
            List of call records with 'arguments' key
        """
        return self.calls.get(name, [])

    def get_last_call(self, name: str) -> dict[str, Any] | None:
        """Get the most recent call to a tool.

        Args:
            name: Name of the tool

        Returns:
            Call record or None if never called
        """
        calls = self.get_calls(name)
        return calls[-1] if calls else None

    def assert_called(self, name: str) -> None:
        """Assert that a tool was called at least once.

        Args:
            name: Name of the tool

        Raises:
            AssertionError: If tool was never called
        """
        assert self.was_called(name), f"MCP tool '{name}' was never called"  # noqa: S101

    def assert_called_with(self, name: str, **expected_args: Any) -> None:
        """Assert that a tool was called with specific arguments.

        Checks the last call to the tool.

        Args:
            name: Name of the tool
            **expected_args: Expected arguments

        Raises:
            AssertionError: If tool not called or args don't match
        """
        last_call = self.get_last_call(name)
        assert last_call is not None, f"MCP tool '{name}' was never called"  # noqa: S101

        for key, expected in expected_args.items():
            actual = last_call["arguments"].get(key)
            assert actual == expected, (  # noqa: S101
                f"MCP tool '{name}' called with {key}={actual}, expected {key}={expected}"
            )

    def reset(self) -> None:
        """Clear call history (keeps tool registrations)."""
        self.calls.clear()
        logger.debug("MockMCPClient call history cleared")

    def clear(self) -> None:
        """Clear both tools and call history."""
        self.tools.clear()
        self.calls.clear()
        logger.debug("MockMCPClient completely cleared")
