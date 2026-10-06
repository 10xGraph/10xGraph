"""
Tests for the Phase 2 criteria implementations.
"""

import pytest

from tenxgraph.qa.evaluation import (
    CriterionConfig,
    EvalCase,
    Invocation,
    MatchType,
    MessageContent,
    ToolCall,
    TrajectoryStep,
    StepType,
)
from tenxgraph.qa.evaluation.collectors.trajectory_collector import TrajectoryCollector
from tenxgraph.qa.evaluation.execution.result import ExecutionResult
from tenxgraph.qa.evaluation.criteria.trajectory import (
    TrajectoryMatchCriterion,
    ToolNameMatchCriterion,
)
from tenxgraph.qa.evaluation.criteria.response import (
    RougeMatchCriterion,
    ExactMatchCriterion,
    ContainsKeywordsCriterion,
)


class TestTrajectoryMatchCriterion:
    """Tests for TrajectoryMatchCriterion."""

    def test_exact_match_perfect(self):
        """Test exact match with identical trajectories."""
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig(
                threshold=1.0,
                match_type=MatchType.EXACT,
                check_args=True,
            )
        )
        
        # Create collector with tool calls
        collector = TrajectoryCollector()
        collector.tool_calls = [
            ToolCall(name="get_weather", args={"location": "NYC"}),
            ToolCall(name="get_time", args={"timezone": "UTC"}),
        ]
        
        # Create eval case with matching expected trajectory
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("Hello"),
                    expected_tool_trajectory=[
                        ToolCall(name="get_weather", args={"location": "NYC"}),
                        ToolCall(name="get_time", args={"timezone": "UTC"}),
                    ],
                )
            ],
        )
        
        result = criterion.evaluate_sync(collector, case)
        
        assert result.score == 1.0
        assert result.passed is True

    def test_exact_match_different_order(self):
        """Test exact match fails with different order."""
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig(
                threshold=1.0,
                match_type=MatchType.EXACT,
            )
        )
        
        collector = TrajectoryCollector()
        collector.tool_calls = [
            ToolCall(name="get_time", args={}),
            ToolCall(name="get_weather", args={}),
        ]
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("Hello"),
                    expected_tool_trajectory=[
                        ToolCall(name="get_weather", args={}),
                        ToolCall(name="get_time", args={}),
                    ],
                )
            ],
        )
        
        result = criterion.evaluate_sync(collector, case)
        
        assert result.score == 0.0
        assert result.passed is False

    def test_in_order_match_with_extras(self):
        """Test in-order match allows extra tools."""
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig(
                threshold=1.0,
                match_type=MatchType.IN_ORDER,
                check_args=False,
            )
        )
        
        collector = TrajectoryCollector()
        collector.tool_calls = [
            ToolCall(name="search", args={}),
            ToolCall(name="get_weather", args={}),
            ToolCall(name="extra_tool", args={}),
            ToolCall(name="get_time", args={}),
        ]
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("Hello"),
                    expected_tool_trajectory=[
                        ToolCall(name="get_weather", args={}),
                        ToolCall(name="get_time", args={}),
                    ],
                )
            ],
        )
        
        result = criterion.evaluate_sync(collector, case)
        
        assert result.score == 1.0
        assert result.passed is True

    def test_in_order_partial_match(self):
        """Test in-order match with partial match."""
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig(
                threshold=0.5,
                match_type=MatchType.IN_ORDER,
            )
        )
        
        collector = TrajectoryCollector()
        collector.tool_calls = [
            ToolCall(name="get_weather", args={}),
            # Missing get_time
        ]
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("Hello"),
                    expected_tool_trajectory=[
                        ToolCall(name="get_weather", args={}),
                        ToolCall(name="get_time", args={}),
                    ],
                )
            ],
        )
        
        result = criterion.evaluate_sync(collector, case)
        
        assert result.score == 0.5  # 1 of 2 found in order
        assert result.passed is True  # 0.5 >= 0.5

    def test_any_order_match(self):
        """Test any-order match with reversed order."""
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig(
                threshold=1.0,
                match_type=MatchType.ANY_ORDER,
                check_args=False,
            )
        )
        
        collector = TrajectoryCollector()
        collector.tool_calls = [
            ToolCall(name="get_time", args={}),
            ToolCall(name="get_weather", args={}),
        ]
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("Hello"),
                    expected_tool_trajectory=[
                        ToolCall(name="get_weather", args={}),
                        ToolCall(name="get_time", args={}),
                    ],
                )
            ],
        )
        
        result = criterion.evaluate_sync(collector, case)
        
        assert result.score == 1.0
        assert result.passed is True

    def test_empty_expected_trajectory(self):
        """Test with no expected tools."""
        criterion = TrajectoryMatchCriterion()
        
        collector = TrajectoryCollector()
        # No tool calls
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("Hello"),
                    # No expected tools
                )
            ],
        )
        
        result = criterion.evaluate_sync(collector, case)
        
        assert result.score == 1.0


class TestToolNameMatchCriterion:
    """Tests for ToolNameMatchCriterion."""

    def test_tool_name_match(self):
        """Test tool name matching."""
        criterion = ToolNameMatchCriterion()
        
        collector = TrajectoryCollector()
        collector.tool_calls = [
            ToolCall(name="get_weather", args={"location": "NYC"}),
            ToolCall(name="get_time", args={"tz": "EST"}),
        ]
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("Hello"),
                    expected_tool_trajectory=[
                        ToolCall(name="get_weather", args={"location": "LA"}),  # Different args
                        ToolCall(name="get_time", args={}),
                    ],
                )
            ],
        )
        
        result = criterion.evaluate_sync(collector, case)
        
        # Should pass because names match (args ignored)
        assert result.score == 1.0


class TestResponseMatchCriterion:
    """Tests for RougeMatchCriterion (ROUGE-1 based)."""

    def test_rouge1_perfect_match(self):
        """Test ROUGE-1 with identical text."""
        criterion = RougeMatchCriterion()
        
        execution = ExecutionResult(
            actual_response="The weather in NYC is sunny.",
        )
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("What's the weather?"),
                    expected_final_response=MessageContent.assistant(
                        "The weather in NYC is sunny."
                    ),
                )
            ],
        )
        
        result = criterion.evaluate_sync(execution, case)
        
        assert result.score == 1.0
        assert result.passed is True

    def test_rouge1_partial_match(self):
        """Test ROUGE-1 with partial overlap."""
        criterion = RougeMatchCriterion(
            config=CriterionConfig(threshold=0.5)
        )
        
        execution = ExecutionResult(
            actual_response="The weather is sunny today.",
        )
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("What's the weather?"),
                    expected_final_response=MessageContent.assistant(
                        "Today the weather in NYC is cloudy."
                    ),
                )
            ],
        )
        
        result = criterion.evaluate_sync(execution, case)
        
        # Should have partial overlap (weather, today, the, is)
        assert result.score > 0.3
        assert result.score < 1.0
        assert "precision" in result.details
        assert "recall" in result.details

    def test_rouge1_no_match(self):
        """Test ROUGE-1 with no overlap."""
        criterion = RougeMatchCriterion()
        
        execution = ExecutionResult(
            actual_response="Hello there!",
        )
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("What's the weather?"),
                    expected_final_response=MessageContent.assistant(
                        "Sunny skies forecast."
                    ),
                )
            ],
        )
        
        result = criterion.evaluate_sync(execution, case)
        
        assert result.score == 0.0


class TestExactMatchCriterion:
    """Tests for ExactMatchCriterion."""

    def test_exact_match_success(self):
        """Test exact match with identical text."""
        criterion = ExactMatchCriterion()
        
        execution = ExecutionResult(
            actual_response="The answer is 42.",
        )
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("What is the answer?"),
                    expected_final_response=MessageContent.assistant(
                        "The answer is 42."
                    ),
                )
            ],
        )
        
        result = criterion.evaluate_sync(execution, case)
        
        assert result.score == 1.0
        assert result.passed is True

    def test_exact_match_failure(self):
        """Test exact match fails with different text."""
        criterion = ExactMatchCriterion()
        
        execution = ExecutionResult(
            actual_response="The answer is 41.",
        )
        
        case = EvalCase(
            eval_id="test",
            conversation=[
                Invocation(
                    user_content=MessageContent.user("What is the answer?"),
                    expected_final_response=MessageContent.assistant(
                        "The answer is 42."
                    ),
                )
            ],
        )
        
        result = criterion.evaluate_sync(execution, case)
        
        assert result.score == 0.0
        assert result.passed is False


class TestContainsKeywordsCriterion:
    """Tests for ContainsKeywordsCriterion."""

    def test_keywords_all_present(self):
        """Test all keywords present."""
        criterion = ContainsKeywordsCriterion(
            keywords=["weather", "sunny", "NYC"]
        )
        
        execution = ExecutionResult(
            actual_response="The weather in NYC is sunny today.",
        )
        
        case = EvalCase(eval_id="test")
        
        result = criterion.evaluate_sync(execution, case)
        
        assert result.score == 1.0
        assert result.passed is True

    def test_keywords_partial_present(self):
        """Test only some keywords present."""
        criterion = ContainsKeywordsCriterion(
            keywords=["weather", "rain", "umbrella"],
            config=CriterionConfig(threshold=0.5),
        )
        
        execution = ExecutionResult(
            actual_response="The weather is sunny.",
        )
        
        case = EvalCase(eval_id="test")
        
        result = criterion.evaluate_sync(execution, case)
        
        # Only "weather" found out of 3
        assert result.score == pytest.approx(0.333, rel=0.01)
        assert result.passed is False
        assert "missing" in result.details
        assert "rain" in result.details["missing"]
