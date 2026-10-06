"""
Tests for the evaluation criteria and result models.
"""

import pytest
from typing import Any

from tenxgraph.qa.evaluation import (
    CriterionConfig,
    EvalCase,
    EvalConfig,
    Invocation,
    MatchType,
    MessageContent,
    ToolCall,
    TrajectoryStep,
)
from tenxgraph.qa.evaluation.eval_result import (
    CriterionResult,
    EvalCaseResult,
    EvalReport,
    EvalSummary,
)
from tenxgraph.qa.evaluation.criteria.base import (
    BaseCriterion,
    CompositeCriterion,
    SyncCriterion,
    WeightedCriterion,
)
from tenxgraph.qa.evaluation.collectors.trajectory_collector import TrajectoryCollector


class MockCriterion(BaseCriterion):
    """Mock criterion for testing."""

    name = "mock_criterion"
    description = "Mock criterion for testing"

    def __init__(
        self,
        score: float = 1.0,
        config: CriterionConfig | None = None,
    ):
        super().__init__(config)
        self._score = score

    async def evaluate(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        return CriterionResult.success(
            criterion=self.name,
            score=self._score,
            threshold=self.threshold,
        )


class MockSyncCriterion(SyncCriterion):
    """Mock sync criterion for testing."""

    name = "mock_sync"
    description = "Mock sync criterion"

    def __init__(self, score: float = 1.0):
        super().__init__()
        self._score = score

    def evaluate_sync(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        return CriterionResult.success(
            criterion=self.name,
            score=self._score,
            threshold=self.threshold,
        )


class TestCriterionResult:
    """Tests for CriterionResult model."""

    def test_criterion_result_success(self):
        """Test creating a successful criterion result."""
        result = CriterionResult.success(
            criterion="test_criterion",
            score=0.85,
            threshold=0.8,
        )
        assert result.criterion == "test_criterion"
        assert result.score == 0.85
        assert result.passed is True

    def test_criterion_result_with_details(self):
        """Test criterion result with additional details."""
        result = CriterionResult.success(
            criterion="trajectory_match",
            score=0.67,
            threshold=0.8,
            details={"missing": ["tool2"]},
        )
        assert result.details == {"missing": ["tool2"]}
        assert result.passed is False  # 0.67 < 0.8

    def test_criterion_result_failure(self):
        """Test creating a failed criterion result."""
        result = CriterionResult.failure(
            criterion="test",
            error="Something went wrong",
        )
        assert result.passed is False
        assert result.score == 0.0
        assert result.error == "Something went wrong"
        assert result.is_error is True


class TestEvalCaseResult:
    """Tests for EvalCaseResult model."""

    def test_eval_case_result_creation(self):
        """Test creating an eval case result."""
        result = EvalCaseResult.success(
            eval_id="test_001",
            criterion_results=[
                CriterionResult.success(criterion="trajectory", score=1.0, threshold=0.8),
                CriterionResult.success(criterion="response", score=0.85, threshold=0.8),
            ],
        )
        assert result.eval_id == "test_001"
        assert result.passed is True
        assert len(result.criterion_results) == 2

    def test_eval_case_result_with_failure(self):
        """Test eval case result with one failing criterion."""
        result = EvalCaseResult.success(
            eval_id="test_001",
            criterion_results=[
                CriterionResult.success(criterion="a", score=1.0, threshold=0.8),
                CriterionResult.success(criterion="b", score=0.5, threshold=0.8),
            ],
        )
        assert result.passed is False

    def test_eval_case_result_get_criterion(self):
        """Test getting a specific criterion result by name."""
        result = EvalCaseResult.success(
            eval_id="test_001",
            criterion_results=[
                CriterionResult.success(criterion="trajectory", score=1.0, threshold=0.8),
                CriterionResult.success(criterion="response", score=0.85, threshold=0.8),
            ],
        )
        
        trajectory_result = result.get_criterion_result("trajectory")
        assert trajectory_result is not None
        assert trajectory_result.score == 1.0

    def test_eval_case_result_failure(self):
        """Test creating a failed case result due to error."""
        result = EvalCaseResult.failure(
            eval_id="test_001",
            error="Agent crashed",
        )
        assert result.passed is False
        assert result.is_error is True


class TestEvalSummary:
    """Tests for EvalSummary model."""

    def test_eval_summary_from_results(self):
        """Test creating summary from results."""
        results = [
            EvalCaseResult.success(
                eval_id="1",
                criterion_results=[CriterionResult.success("c", 1.0, 0.8)],
            ),
            EvalCaseResult.success(
                eval_id="2",
                criterion_results=[CriterionResult.success("c", 0.9, 0.8)],
            ),
            EvalCaseResult.success(
                eval_id="3",
                criterion_results=[CriterionResult.success("c", 0.5, 0.8)],
            ),
            EvalCaseResult.success(
                eval_id="4",
                criterion_results=[CriterionResult.success("c", 0.85, 0.8)],
            ),
        ]
        
        summary = EvalSummary.from_results(results)
        
        assert summary.total_cases == 4
        assert summary.passed_cases == 3
        assert summary.failed_cases == 1
        assert summary.pass_rate == 0.75

    def test_eval_summary_empty_results(self):
        """Test summary with no results."""
        summary = EvalSummary.from_results([])
        
        assert summary.total_cases == 0
        assert summary.passed_cases == 0
        assert summary.pass_rate == 0.0


class TestEvalReport:
    """Tests for EvalReport model."""

    def test_eval_report_creation(self):
        """Test creating an eval report."""
        results = [
            EvalCaseResult.success(
                eval_id="1",
                criterion_results=[CriterionResult.success("c", 1.0, 0.8)],
            ),
            EvalCaseResult.success(
                eval_id="2",
                criterion_results=[CriterionResult.success("c", 0.5, 0.8)],
            ),
        ]
        
        report = EvalReport.create(
            eval_set_id="test_set",
            results=results,
        )
        
        assert report.eval_set_id == "test_set"
        assert len(report.results) == 2
        assert report.summary.total_cases == 2

    def test_eval_report_format_summary(self):
        """Test formatting summary for display."""
        results = [
            EvalCaseResult.success(
                eval_id="1",
                name="Test 1",
                criterion_results=[CriterionResult.success("trajectory", 1.0, 0.8)],
            ),
        ]
        report = EvalReport.create(
            eval_set_id="test_set",
            eval_set_name="Test Set",
            results=results,
        )
        
        formatted = report.format_summary()
        assert "Test Set" in formatted


class TestBaseCriterion:
    """Tests for BaseCriterion abstract class."""

    @pytest.mark.asyncio
    async def test_mock_criterion_evaluate(self):
        """Test evaluating with mock criterion."""
        criterion = MockCriterion(score=0.9)
        collector = TrajectoryCollector()
        case = EvalCase.single_turn("test", "Hello")
        
        result = await criterion.evaluate(collector, case)
        
        assert result.score == 0.9
        assert result.passed is True
        assert result.criterion == "mock_criterion"

    @pytest.mark.asyncio
    async def test_criterion_with_config(self):
        """Test criterion with custom config."""
        config = CriterionConfig(threshold=0.9, match_type=MatchType.IN_ORDER)
        criterion = MockCriterion(config=config)
        
        assert criterion.config.threshold == 0.9
        assert criterion.config.match_type == MatchType.IN_ORDER


class TestSyncCriterion:
    """Tests for SyncCriterion."""

    @pytest.mark.asyncio
    async def test_sync_criterion_async_wrapper(self):
        """Test that sync criterion works with async interface."""
        criterion = MockSyncCriterion(score=0.8)
        collector = TrajectoryCollector()
        case = EvalCase.single_turn("test", "Hello")
        
        result = await criterion.evaluate(collector, case)
        
        assert result.score == 0.8
        assert result.passed is True


class TestCompositeCriterion:
    """Tests for CompositeCriterion."""

    @pytest.mark.asyncio
    async def test_composite_all_pass(self):
        """Test composite criterion when all pass."""
        criteria = [
            MockCriterion(score=1.0),
            MockCriterion(score=0.9),
            MockCriterion(score=0.85),
        ]
        
        composite = CompositeCriterion(criteria=criteria, require_all=True)
        
        collector = TrajectoryCollector()
        case = EvalCase.single_turn("test", "Hello")
        
        result = await composite.evaluate(collector, case)
        
        assert result.score == 0.85  # Min score with require_all


class TestWeightedCriterion:
    """Tests for WeightedCriterion."""

    @pytest.mark.asyncio
    async def test_weighted_criterion(self):
        """Test weighted criterion scoring."""
        weighted_criteria = [
            (MockCriterion(score=1.0), 2.0),
            (MockCriterion(score=0.5), 1.0),
        ]
        
        weighted = WeightedCriterion(criteria_weights=weighted_criteria)
        
        collector = TrajectoryCollector()
        case = EvalCase.single_turn("test", "Hello")
        
        result = await weighted.evaluate(collector, case)
        
        # (1.0 * 2 + 0.5 * 1) / 3 = 0.833...
        assert result.score == pytest.approx(0.833, rel=0.01)
