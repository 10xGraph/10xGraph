"""Fluent builder for creating EvalSets easily."""

from __future__ import annotations

import uuid
from typing import Any

from agentflow.qa.evaluation.dataset.eval_set import EvalCase, EvalSet, ToolCall


class EvalSetBuilder:
    """Fluent builder for creating evaluation sets.

    Example:
        ```python
        eval_set = (
            EvalSetBuilder("my_tests")
            .add_case(query="Hello", expected="Hi there")
            .add_case(query="Weather?", expected="Sunny", expected_tools=["get_weather"])
            .build()
        )
        ```
    """

    def __init__(self, name: str = "eval_set"):
        self.name = name
        self.eval_set_id = str(uuid.uuid4())
        self.cases: list[EvalCase] = []

    def add_case(
        self,
        query: str,
        expected: str,
        case_id: str | None = None,
        expected_tools: list[str | ToolCall] | None = None,
        expected_node_order: list[str] | None = None,
        name: str = "",
        description: str = "",
    ) -> EvalSetBuilder:
        """Add a single-turn test case.

        Args:
            query: User query/input.
            expected: Expected agent response.
            case_id: Optional case ID (auto-generated if not provided).
            expected_tools: Expected tool calls as names or ToolCall objects.
            expected_node_order: Expected node execution order.
            name: Human-readable name for the case.
            description: Description of the test case.

        Returns:
            Self for method chaining.
        """
        eval_id = case_id or f"case_{len(self.cases) + 1}"
        tool_calls = self._normalise_tools(expected_tools)

        case = EvalCase.single_turn(
            eval_id=eval_id,
            user_query=query,
            expected_response=expected,
            expected_tools=tool_calls or None,
            expected_node_order=expected_node_order,
            name=name,
            description=description,
        )
        self.cases.append(case)
        return self

    def add_multi_turn(
        self,
        conversation: list[tuple[str, str]],
        case_id: str | None = None,
        expected_tools: list[str | ToolCall] | None = None,
        name: str = "",
        description: str = "",
    ) -> EvalSetBuilder:
        """Add a multi-turn conversation case.

        Args:
            conversation: List of (user_query, expected_response) tuples.
            case_id: Optional case ID.
            expected_tools: Expected tool calls across the conversation.
            name: Human-readable name for the case.
            description: Description of the test case.

        Returns:
            Self for method chaining.
        """
        eval_id = case_id or f"case_{len(self.cases) + 1}"
        tool_calls = self._normalise_tools(expected_tools)

        case = EvalCase.multi_turn(
            eval_id=eval_id,
            conversation=conversation,
            expected_tools=tool_calls or None,
            name=name,
            description=description,
        )
        self.cases.append(case)
        return self

    def add_tool_test(
        self,
        query: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        expected_response: str | None = None,
        case_id: str | None = None,
    ) -> EvalSetBuilder:
        """Add a test case focused on a single tool call.

        Args:
            query: User query.
            tool_name: Expected tool to be called.
            tool_args: Expected tool arguments.
            expected_response: Expected final response (optional).
            case_id: Optional case ID.

        Returns:
            Self for method chaining.
        """
        eval_id = case_id or f"case_{len(self.cases) + 1}"
        tool_call = ToolCall(name=tool_name, args=tool_args or {})

        case = EvalCase.single_turn(
            eval_id=eval_id,
            user_query=query,
            expected_response=expected_response or f"Result from {tool_name}",
            expected_tools=[tool_call],
        )
        self.cases.append(case)
        return self

    def build(self) -> EvalSet:
        """Build and return the final EvalSet."""
        return EvalSet(
            eval_set_id=self.eval_set_id,
            name=self.name,
            eval_cases=self.cases,
        )

    def save(self, path: str) -> EvalSet:
        """Build and save the EvalSet to a JSON file.

        Args:
            path: File path to save JSON.

        Returns:
            The built EvalSet.
        """
        eval_set = self.build()
        eval_set.to_file(path)
        return eval_set

    @classmethod
    def from_conversations(
        cls,
        conversations: list[dict[str, str]],
        name: str = "conversation_tests",
    ) -> EvalSet:
        """Create eval set from conversation logs.

        Args:
            conversations: List of dicts with 'user' and 'assistant' keys.
            name: Name for the eval set.

        Returns:
            Built EvalSet.
        """
        builder = cls(name)
        for i, conv in enumerate(conversations):
            builder.add_case(
                query=conv.get("user", ""),
                expected=conv.get("assistant", ""),
                case_id=f"conv_{i + 1}",
            )
        return builder.build()

    @classmethod
    def from_file(cls, path: str) -> EvalSetBuilder:
        """Load existing eval set and convert to builder for modification.

        Args:
            path: Path to existing eval set JSON.

        Returns:
            Builder initialised with cases from file.
        """
        eval_set = EvalSet.from_file(path)
        builder = cls(eval_set.name or "eval_set")
        builder.eval_set_id = eval_set.eval_set_id
        builder.cases = list(eval_set.eval_cases)
        return builder

    @classmethod
    def quick(cls, *test_pairs: tuple[str, str]) -> EvalSet:
        """Quick builder from (query, expected) pairs.

        Example:
            ```python
            eval_set = EvalSetBuilder.quick(
                ("Hello", "Hi!"),
                ("How are you?", "Great!"),
            )
            ```
        """
        builder = cls("quick_tests")
        for i, (query, expected) in enumerate(test_pairs):
            builder.add_case(query=query, expected=expected, case_id=f"test_{i + 1}")
        return builder.build()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_tools(
        tools: list[str | ToolCall] | None,
    ) -> list[ToolCall]:
        """Convert list of tool names or ToolCall objects to ToolCall list."""
        if not tools:
            return []
        result = []
        for tool in tools:
            if isinstance(tool, str):
                result.append(ToolCall(name=tool, args={}))
            else:
                result.append(tool)
        return result
