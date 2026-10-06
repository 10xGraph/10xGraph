"""Test result with built-in assertions for QuickTest."""

from typing import Any


class TestResult:
    """Result from a QuickTest execution with built-in assertions.

    Attributes:
        final_response: The final response from the agent
        messages: All messages exchanged
        tool_calls: List of tool calls made
        state: Final state dict
        passed: Whether test passed
    """

    def __init__(
        self,
        final_response: str,
        messages: list[Any],
        tool_calls: list[dict[str, Any]],
        state: dict[str, Any],
    ):
        self.final_response = final_response
        self.messages = messages
        self.tool_calls = tool_calls
        self.state = state
        self.passed = True

    def assert_contains(self, text: str) -> "TestResult":
        """Assert final response contains text."""
        assert text in self.final_response, (  # noqa: S101
            f"Expected response to contain '{text}', but got: {self.final_response}"
        )
        return self

    def assert_not_contains(self, text: str) -> "TestResult":
        """Assert final response does not contain text."""
        assert text not in self.final_response, (  # noqa: S101
            f"Expected response to NOT contain '{text}', but got: {self.final_response}"
        )
        return self

    def assert_equals(self, expected: str) -> "TestResult":
        """Assert final response equals expected."""
        assert self.final_response == expected, f"Expected: {expected}\nGot: {self.final_response}"  # noqa: S101
        return self

    def assert_tool_called(self, tool_name: str, **expected_args: Any) -> "TestResult":
        """Assert a tool was called with specific arguments."""
        for call in self.tool_calls:
            if call.get("name") == tool_name:
                if expected_args:
                    call_args = call.get("args", {})
                    for key, value in expected_args.items():
                        assert call_args.get(key) == value, (  # noqa: S101
                            f"Tool {tool_name} called with {key}={call_args.get(key)}, "
                            f"expected {key}={value}"
                        )
                return self

        raise AssertionError(
            f"Tool '{tool_name}' was not called. "
            f"Called tools: {[c.get('name') for c in self.tool_calls]}"
        )

    def assert_tool_not_called(self, tool_name: str) -> "TestResult":
        """Assert a tool was NOT called."""
        for call in self.tool_calls:
            if call.get("name") == tool_name:
                raise AssertionError(f"Tool '{tool_name}' was unexpectedly called")
        return self

    def assert_message_count(self, count: int) -> "TestResult":
        """Assert number of messages."""
        actual = len(self.messages)
        assert actual == count, f"Expected {count} messages, got {actual}"  # noqa: S101
        return self

    def assert_no_errors(self) -> "TestResult":
        """Assert no error messages in conversation."""
        for msg in self.messages:
            if hasattr(msg, "role") and msg.role == "error":
                raise AssertionError(f"Found error message: {msg}")
        return self

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.passed
