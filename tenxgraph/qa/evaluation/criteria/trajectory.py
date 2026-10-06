"""
Trajectory matching criteria.

Compares actual tool call trajectories against expected trajectories
using EXACT, IN_ORDER, or ANY_ORDER matching strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentflow.qa.evaluation.config.eval_config import MatchType
from agentflow.qa.evaluation.criteria.base import SyncCriterion
from agentflow.qa.evaluation.dataset.eval_set import ToolCall


if TYPE_CHECKING:
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.eval_result import CriterionResult
    from agentflow.qa.evaluation.execution.result import ExecutionResult


class TrajectoryMatchCriterion(SyncCriterion):
    """Compare actual vs expected tool call trajectories.

    Supports three matching modes:
        - EXACT: Same tools, same order, no extras
        - IN_ORDER: Expected tools in order, extras allowed
        - ANY_ORDER: Expected tools in any order, extras allowed
    """

    name = "tool_trajectory_avg_score"
    description = "Evaluates tool call trajectory matching"

    def evaluate_sync(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        expected_tools: list[ToolCall] = []
        for invocation in expected.conversation:
            expected_tools.extend(invocation.expected_tool_trajectory)

        actual_tools = actual.tool_calls
        match_type = self.config.match_type
        check_args = self.config.check_args

        if match_type == MatchType.EXACT:
            score = self._exact_match(actual_tools, expected_tools, check_args)
        elif match_type == MatchType.IN_ORDER:
            score = self._in_order_match(actual_tools, expected_tools, check_args)
        else:
            score = self._any_order_match(actual_tools, expected_tools, check_args)

        actual_names = [t.name for t in actual_tools]
        expected_names = [t.name for t in expected_tools]

        return self._result(
            score,
            {
                "reason": (
                    f"Matched {score:.0%} of expected tools. "
                    f"Expected {expected_names}, got {actual_names}."
                ),
                "actual_trajectory": [t.model_dump() for t in actual_tools],
                "expected_trajectory": [t.model_dump() for t in expected_tools],
                "match_type": match_type.value,
                "check_args": check_args,
            },
        )

    def _tools_match(self, actual: ToolCall, expected: ToolCall, check_args: bool) -> bool:
        return actual.matches(expected, check_args=check_args, check_call_id=False)

    def _exact_match(
        self,
        actual: list[ToolCall],
        expected: list[ToolCall],
        check_args: bool,
    ) -> float:
        if not expected:
            return 1.0 if not actual else 0.0
        if len(actual) != len(expected):
            min_len = min(len(actual), len(expected))
            matches = sum(
                1
                for a, e in zip(actual[:min_len], expected[:min_len], strict=False)
                if self._tools_match(a, e, check_args)
            )
            return matches / len(expected)
        matches = sum(
            1 for a, e in zip(actual, expected, strict=False) if self._tools_match(a, e, check_args)
        )
        return matches / len(expected)

    def _in_order_match(
        self,
        actual: list[ToolCall],
        expected: list[ToolCall],
        check_args: bool,
    ) -> float:
        if not expected:
            return 1.0
        expected_idx = 0
        for actual_tool in actual:
            if expected_idx >= len(expected):
                break
            if self._tools_match(actual_tool, expected[expected_idx], check_args):
                expected_idx += 1
        return expected_idx / len(expected)

    def _any_order_match(
        self,
        actual: list[ToolCall],
        expected: list[ToolCall],
        check_args: bool,
    ) -> float:
        if not expected:
            return 1.0
        matched = 0
        remaining_actual = list(actual)
        for exp_tool in expected:
            for idx, act_tool in enumerate(remaining_actual):
                if self._tools_match(act_tool, exp_tool, check_args):
                    matched += 1
                    remaining_actual.pop(idx)
                    break
        return matched / len(expected)


class NodeOrderMatchCriterion(SyncCriterion):
    """Check that the graph visited nodes in the expected order.

    Supports three matching modes via config.match_type:
        - EXACT: Same nodes, same order, same count
        - IN_ORDER: Expected nodes appear in order, extras allowed
        - ANY_ORDER: Expected nodes all present, regardless of order
    """

    name = "node_order_score"
    description = "Evaluates node visit order matching"

    def evaluate_sync(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        expected_nodes: list[str] = []
        for invocation in expected.conversation:
            expected_nodes.extend(invocation.expected_node_order)

        actual_nodes = actual.node_visits
        match_type = self.config.match_type

        if not expected_nodes:
            score = 1.0
        elif match_type == MatchType.EXACT:
            score = self._exact_match(actual_nodes, expected_nodes)
        elif match_type == MatchType.IN_ORDER:
            score = self._in_order_match(actual_nodes, expected_nodes)
        else:
            score = self._any_order_match(actual_nodes, expected_nodes)

        return self._result(
            score,
            {
                "reason": (
                    f"Matched {score:.0%} of expected node order. "
                    f"Expected {expected_nodes}, got {actual_nodes}."
                ),
                "actual_node_order": actual_nodes,
                "expected_node_order": expected_nodes,
                "match_type": match_type.value,
            },
        )

    @staticmethod
    def _exact_match(actual: list[str], expected: list[str]) -> float:
        if len(actual) != len(expected):
            min_len = min(len(actual), len(expected))
            matches = sum(
                1 for a, e in zip(actual[:min_len], expected[:min_len], strict=False) if a == e
            )
            return matches / len(expected)
        matches = sum(1 for a, e in zip(actual, expected, strict=False) if a == e)
        return matches / len(expected)

    @staticmethod
    def _in_order_match(actual: list[str], expected: list[str]) -> float:
        if not expected:
            return 1.0
        expected_idx = 0
        for node in actual:
            if expected_idx >= len(expected):
                break
            if node == expected[expected_idx]:
                expected_idx += 1
        return expected_idx / len(expected)

    @staticmethod
    def _any_order_match(actual: list[str], expected: list[str]) -> float:
        if not expected:
            return 1.0
        matched = 0
        remaining = list(actual)
        for exp_node in expected:
            if exp_node in remaining:
                matched += 1
                remaining.remove(exp_node)
        return matched / len(expected)


class ToolNameMatchCriterion(SyncCriterion):
    """Simpler criterion that only checks tool names, not arguments."""

    name = "tool_name_match_score"
    description = "Evaluates tool name matching (ignores arguments)"

    def evaluate_sync(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        expected_names: list[str] = []
        for invocation in expected.conversation:
            expected_names.extend(t.name for t in invocation.expected_tool_trajectory)

        actual_names = actual.get_tool_names()

        if not expected_names:
            score = 1.0 if not actual_names else 0.5
        else:
            matched = 0
            remaining = list(actual_names)
            for exp_name in expected_names:
                if exp_name in remaining:
                    matched += 1
                    remaining.remove(exp_name)
            score = matched / len(expected_names)

        return self._result(
            score,
            {
                "expected_names": expected_names,
                "actual_names": actual_names,
            },
        )
