"""Tests for trajectory, node order, and tool name matching criteria.

These are SyncCriterion subclasses — no LLM calls, pure logic.
"""

from __future__ import annotations

import pytest

from tenxgraph.qa.evaluation.config.eval_config import CriterionConfig, MatchType
from tenxgraph.qa.evaluation.criteria.trajectory import (
    NodeOrderMatchCriterion,
    ToolNameMatchCriterion,
    TrajectoryMatchCriterion,
)
from tenxgraph.qa.evaluation.dataset.eval_set import EvalCase, ToolCall
from tenxgraph.qa.evaluation.execution.result import ExecutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tool_names: list[str] | None = None,
    node_visits: list[str] | None = None,
) -> ExecutionResult:
    tool_calls = [ToolCall(name=n, args={}) for n in (tool_names or [])]
    return ExecutionResult(
        tool_calls=tool_calls,
        node_visits=node_visits or [],
        actual_response="ok",
    )


def _make_case(
    expected_tools: list[str] | None = None,
    expected_nodes: list[str] | None = None,
) -> EvalCase:
    tool_calls = [ToolCall(name=n, args={}) for n in (expected_tools or [])]
    return EvalCase.single_turn(
        eval_id="test_case",
        user_query="query",
        expected_response="response",
        expected_tools=tool_calls if tool_calls else None,
        expected_node_order=expected_nodes,
    )


# ---------------------------------------------------------------------------
# TrajectoryMatchCriterion — EXACT
# ---------------------------------------------------------------------------

class TestTrajectoryExactMatch:
    def test_perfect_match(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(threshold=0.8, match_type=MatchType.EXACT)
        )
        actual = _make_result(["get_weather", "send_email"])
        expected = _make_case(["get_weather", "send_email"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_no_tools_expected_none_actual(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.EXACT)
        )
        actual = _make_result([])
        expected = _make_case([])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_extra_actual_tools_penalised(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.EXACT)
        )
        actual = _make_result(["tool_a", "tool_b", "tool_c"])
        expected = _make_case(["tool_a", "tool_b"])
        result = criterion.evaluate_sync(actual, expected)
        # len different: min_len=2, both match → 2/2 = 1? Actually: len(actual)=3 != len(expected)=2
        # min_len=2, matches = 2, score = 2/2 = 1.0 (it only counts the first 2)
        assert result.score == pytest.approx(1.0)

    def test_wrong_order(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.EXACT)
        )
        actual = _make_result(["tool_b", "tool_a"])
        expected = _make_case(["tool_a", "tool_b"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(0.0)

    def test_partial_match_different_length(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.EXACT)
        )
        actual = _make_result(["tool_a", "tool_x"])
        expected = _make_case(["tool_a", "tool_b", "tool_c"])
        result = criterion.evaluate_sync(actual, expected)
        # min_len=2, matches=1 (only tool_a matches), score=1/3
        assert result.score == pytest.approx(1 / 3)

    def test_no_tools_expected_but_actual_has_tools(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.EXACT)
        )
        actual = _make_result(["tool_a"])
        expected = _make_case([])
        result = criterion.evaluate_sync(actual, expected)
        # no expected → score=0.0 (because not actual is False)
        assert result.score == pytest.approx(0.0)

    def test_result_has_details(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.EXACT)
        )
        actual = _make_result(["tool_a"])
        expected = _make_case(["tool_a"])
        result = criterion.evaluate_sync(actual, expected)
        assert "actual_trajectory" in result.details
        assert "expected_trajectory" in result.details
        assert "match_type" in result.details


# ---------------------------------------------------------------------------
# TrajectoryMatchCriterion — IN_ORDER
# ---------------------------------------------------------------------------

class TestTrajectoryInOrderMatch:
    def test_all_expected_in_order(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.IN_ORDER)
        )
        actual = _make_result(["a", "b", "c", "d"])
        expected = _make_case(["a", "c"])  # "a" then "c" in order
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_not_in_order(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.IN_ORDER)
        )
        actual = _make_result(["c", "a"])
        expected = _make_case(["a", "c"])  # need a first, then c
        result = criterion.evaluate_sync(actual, expected)
        # actual has 'c' first so 'a' is not found in order after it
        # scan: 'c' != 'a', 'a' == 'a' -> expected_idx=1; then expected_idx >= len -> end
        # Actually: loop actual=['c','a']; expected_idx=0 (want 'a')
        # 'c' != 'a', skip; 'a' == 'a', expected_idx=1; loop ends -> 1/2 = 0.5
        assert result.score == pytest.approx(0.5)

    def test_no_expected(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.IN_ORDER)
        )
        actual = _make_result(["a", "b"])
        expected = _make_case([])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_partial_match(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.IN_ORDER)
        )
        actual = _make_result(["a", "b"])
        expected = _make_case(["a", "b", "c"])
        result = criterion.evaluate_sync(actual, expected)
        # finds a then b but not c -> 2/3
        assert result.score == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# TrajectoryMatchCriterion — ANY_ORDER
# ---------------------------------------------------------------------------

class TestTrajectoryAnyOrderMatch:
    def test_all_match_different_order(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.ANY_ORDER)
        )
        actual = _make_result(["c", "a", "b"])
        expected = _make_case(["a", "b", "c"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_partial_match(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.ANY_ORDER)
        )
        actual = _make_result(["a", "x"])
        expected = _make_case(["a", "b", "c"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1 / 3)

    def test_no_expected(self):
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig.trajectory(match_type=MatchType.ANY_ORDER)
        )
        actual = _make_result(["a"])
        expected = _make_case([])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NodeOrderMatchCriterion
# ---------------------------------------------------------------------------

class TestNodeOrderMatchCriterion:
    def test_exact_match(self):
        criterion = NodeOrderMatchCriterion(
            config=CriterionConfig.node_order(match_type=MatchType.EXACT)
        )
        actual = _make_result(node_visits=["MAIN", "TOOL", "MAIN"])
        expected = _make_case(expected_nodes=["MAIN", "TOOL", "MAIN"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_exact_mismatch(self):
        criterion = NodeOrderMatchCriterion(
            config=CriterionConfig.node_order(match_type=MatchType.EXACT)
        )
        actual = _make_result(node_visits=["MAIN", "MAIN"])
        expected = _make_case(expected_nodes=["MAIN", "TOOL", "MAIN"])
        result = criterion.evaluate_sync(actual, expected)
        # len mismatch: min_len=2, actual[:2]=["MAIN","MAIN"], expected[:2]=["MAIN","TOOL"]
        # matches=1, score=1/3
        assert result.score == pytest.approx(1 / 3)

    def test_no_expected_nodes(self):
        criterion = NodeOrderMatchCriterion(
            config=CriterionConfig.node_order(match_type=MatchType.EXACT)
        )
        actual = _make_result(node_visits=["MAIN"])
        expected = _make_case(expected_nodes=[])
        result = criterion.evaluate_sync(actual, expected)
        # no expected → score = 1.0 (no expectation → always pass)
        assert result.score == pytest.approx(1.0)

    def test_in_order(self):
        criterion = NodeOrderMatchCriterion(
            config=CriterionConfig.node_order(match_type=MatchType.IN_ORDER)
        )
        actual = _make_result(node_visits=["MAIN", "TOOL", "EXTRA", "MAIN"])
        expected = _make_case(expected_nodes=["MAIN", "MAIN"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_any_order(self):
        criterion = NodeOrderMatchCriterion(
            config=CriterionConfig.node_order(match_type=MatchType.ANY_ORDER)
        )
        actual = _make_result(node_visits=["TOOL", "MAIN"])
        expected = _make_case(expected_nodes=["MAIN", "TOOL"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_any_order_partial(self):
        criterion = NodeOrderMatchCriterion(
            config=CriterionConfig.node_order(match_type=MatchType.ANY_ORDER)
        )
        actual = _make_result(node_visits=["MAIN"])
        expected = _make_case(expected_nodes=["MAIN", "TOOL"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(0.5)

    def test_result_has_details(self):
        criterion = NodeOrderMatchCriterion()
        actual = _make_result(node_visits=["MAIN"])
        expected = _make_case(expected_nodes=["MAIN"])
        result = criterion.evaluate_sync(actual, expected)
        assert "actual_node_order" in result.details
        assert "expected_node_order" in result.details


# ---------------------------------------------------------------------------
# ToolNameMatchCriterion
# ---------------------------------------------------------------------------

class TestToolNameMatchCriterion:
    def test_perfect_match(self):
        criterion = ToolNameMatchCriterion()
        actual = _make_result(["tool_a", "tool_b"])
        expected = _make_case(["tool_a", "tool_b"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_partial_match(self):
        criterion = ToolNameMatchCriterion()
        actual = _make_result(["tool_a", "tool_x"])
        expected = _make_case(["tool_a", "tool_b"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(0.5)

    def test_no_expected_no_actual(self):
        criterion = ToolNameMatchCriterion()
        actual = _make_result([])
        expected = _make_case([])
        result = criterion.evaluate_sync(actual, expected)
        # no expected → score = 1.0
        assert result.score == pytest.approx(1.0)

    def test_no_expected_but_actual_has_tools(self):
        criterion = ToolNameMatchCriterion()
        actual = _make_result(["tool_a"])
        expected = _make_case([])
        result = criterion.evaluate_sync(actual, expected)
        # no expected, actual has tools → score = 0.5
        assert result.score == pytest.approx(0.5)

    def test_duplicate_tools_count_once(self):
        criterion = ToolNameMatchCriterion()
        actual = _make_result(["tool_a", "tool_a", "tool_b"])
        expected = _make_case(["tool_a", "tool_b"])
        result = criterion.evaluate_sync(actual, expected)
        assert result.score == pytest.approx(1.0)

    def test_details_included(self):
        criterion = ToolNameMatchCriterion()
        actual = _make_result(["tool_a"])
        expected = _make_case(["tool_a"])
        result = criterion.evaluate_sync(actual, expected)
        assert "expected_names" in result.details
        assert "actual_names" in result.details
