"""Call log shared by the mock tool registry and the mock MCP client."""

from __future__ import annotations

from typing import Any


class _CallLog:
    """Mixin: query and assert on ``self.calls``, a dict of tool name to call records.

    ``_label`` names the thing in assertion messages and ``_args_key`` is the record key that
    holds the arguments checked by :meth:`assert_called_with`.
    """

    _label = "Tool"
    _args_key = "kwargs"

    calls: dict[str, list[dict[str, Any]]]

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
            List of call records
        """
        return self.calls.get(name, [])

    def get_last_call(self, name: str) -> dict[str, Any] | None:
        """Get the most recent call to a tool.

        Args:
            name: Name of the tool

        Returns:
            Call record dict, or None if never called
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
        assert self.was_called(name), f"{self._label} '{name}' was never called"  # noqa: S101

    def assert_called_with(self, name: str, **expected: Any) -> None:
        """Assert that a tool was called with specific arguments.

        Checks the last call to the tool.

        Args:
            name: Name of the tool
            **expected: Expected arguments

        Raises:
            AssertionError: If tool not called or arguments don't match
        """
        last_call = self.get_last_call(name)
        assert last_call is not None, f"{self._label} '{name}' was never called"  # noqa: S101

        for key, value in expected.items():
            actual = last_call[self._args_key].get(key)
            assert actual == value, (  # noqa: S101
                f"{self._label} '{name}' called with {key}={actual}, expected {key}={value}"
            )
