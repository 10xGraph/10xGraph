"""
Base criterion interface for evaluation.

This module defines the abstract base class for all evaluation criteria,
providing a consistent interface for different types of evaluations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.eval_result import CriterionResult


if TYPE_CHECKING:
    from agentflow.qa.evaluation.config.eval_config import CriterionConfig
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.execution.result import ExecutionResult


class BaseCriterion(ABC):
    """Abstract base class for evaluation criteria.

    All evaluation criteria must inherit from this class and implement
    the evaluate method. Criteria can evaluate different aspects of
    agent behavior such as:
        - Tool trajectory matching
        - Response similarity
        - LLM-judged quality
        - Custom rubric-based evaluation

    Attributes:
        name: Unique identifier for this criterion.
        description: Human-readable description of what this criterion evaluates.
        config: Configuration for this criterion.

    Example:
        ```python
        class MyCustomCriterion(BaseCriterion):
            name = "my_custom_criterion"
            description = "Evaluates custom behavior"

            async def evaluate(
                self,
                actual: ExecutionResult,
                expected: EvalCase,
            ) -> CriterionResult:
                # Custom evaluation logic
                score = self._compute_score(actual, expected)
                return CriterionResult.success(
                    criterion=self.name,
                    score=score,
                    threshold=self.config.threshold,
                )
        ```
    """

    # Class attributes - override in subclasses
    name: str = "base_criterion"
    description: str = "Base evaluation criterion"

    def __init__(self, config: CriterionConfig | None = None):
        """Initialize the criterion with optional configuration.

        Args:
            config: Configuration for this criterion. If not provided,
                uses default configuration.
        """
        from agentflow.qa.evaluation.config.eval_config import CriterionConfig as _CriterionConfig

        self.config = config or _CriterionConfig()

    @abstractmethod
    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate the actual execution against expected outcomes.

        This is the main method that subclasses must implement to perform
        their specific evaluation logic.

        Args:
            actual: Extracted execution result from the agent run.
            expected: The evaluation case with expected outcomes.

        Returns:
            CriterionResult containing score, pass/fail status, and details.

        Raises:
            EvaluationError: If evaluation fails due to an error.
        """

    def validate_config(self) -> list[str]:
        """Validate the criterion configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        if self.config.threshold < 0.0 or self.config.threshold > 1.0:
            errors.append(f"Threshold must be between 0.0 and 1.0, got {self.config.threshold}")
        return errors

    @property
    def threshold(self) -> float:
        """Get the threshold from configuration."""
        return self.config.threshold

    @property
    def is_enabled(self) -> bool:
        """Check if this criterion is enabled."""
        return self.config.enabled

    # ------------------------------------------------------------------
    # Shared helpers for criteria that inspect EvalCase / ExecutionResult
    # ------------------------------------------------------------------

    def _extract_question(self, expected: EvalCase) -> str:
        """Return the first user message text from the eval case."""
        if expected.conversation:
            return expected.conversation[0].user_content.get_text()
        return ""

    def _extract_last_expected_response(self, expected: EvalCase) -> str:
        """Return the last non-empty expected final response text."""
        result = ""
        for invocation in expected.conversation:
            if invocation.expected_final_response:
                result = invocation.expected_final_response.get_text()
        return result

    def _result(
        self,
        score: float,
        details: dict | None = None,
        token_usage: Any | None = None,
    ) -> CriterionResult:
        """Build a passing CriterionResult for this criterion."""
        kwargs: dict = {
            "criterion": self.name,
            "score": score,
            "threshold": self.threshold,
            "details": details or {},
        }
        if token_usage is not None:
            kwargs["token_usage"] = token_usage
        return CriterionResult.success(**kwargs)

    def _failure(self, error: str) -> CriterionResult:
        """Build a failure CriterionResult for this criterion."""
        return CriterionResult.failure(criterion=self.name, error=error)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, threshold={self.threshold})"


class SyncCriterion(BaseCriterion):
    """Base class for synchronous criteria.

    Some criteria don't require async operations (e.g., pure computation).
    This class provides a sync interface that wraps the async one.
    """

    @abstractmethod
    def evaluate_sync(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        """Synchronous evaluation method.

        Args:
            actual: Extracted execution result from the agent run.
            expected: The evaluation case with expected outcomes.

        Returns:
            CriterionResult containing score, pass/fail status, and details.
        """

    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        """Async wrapper around sync evaluation."""
        return self.evaluate_sync(actual, expected)


class CompositeCriterion(BaseCriterion):
    """Criterion that combines multiple sub-criteria.

    This allows creating complex evaluation logic by combining
    simpler criteria with AND/OR semantics.

    Attributes:
        criteria: List of sub-criteria to evaluate.
        require_all: If True, all criteria must pass (AND logic).
            If False, any criterion passing is sufficient (OR logic).
    """

    name = "composite_criterion"
    description = "Combines multiple criteria"

    def __init__(
        self,
        criteria: list[BaseCriterion],
        require_all: bool = True,
        config: CriterionConfig | None = None,
    ):
        """Initialize composite criterion.

        Args:
            criteria: List of criteria to combine.
            require_all: Whether all criteria must pass.
            config: Optional configuration override.
        """
        super().__init__(config)
        self.criteria = criteria
        self.require_all = require_all

    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate all sub-criteria and combine results."""
        results = []
        scores = []

        for criterion in self.criteria:
            result = await criterion.evaluate(actual, expected)
            results.append(result)
            scores.append(result.score)

        # Compute combined score
        if self.require_all:
            # AND: use minimum score, all must pass
            combined_score = min(scores) if scores else 0.0
        else:
            # OR: use maximum score, any pass is success
            combined_score = max(scores) if scores else 0.0

        return self._result(
            combined_score,
            {
                "sub_results": [r.model_dump() for r in results],
                "require_all": self.require_all,
            },
        )


class WeightedCriterion(BaseCriterion):
    """Criterion that computes weighted average of sub-criteria.

    Useful for combining multiple criteria with different importance levels.

    Attributes:
        criteria_weights: List of (criterion, weight) tuples.
    """

    name = "weighted_criterion"
    description = "Weighted combination of criteria"

    def __init__(
        self,
        criteria_weights: list[tuple[BaseCriterion, float]],
        config: CriterionConfig | None = None,
    ):
        """Initialize weighted criterion.

        Args:
            criteria_weights: List of (criterion, weight) tuples.
            config: Optional configuration override.
        """
        super().__init__(config)
        self.criteria_weights = criteria_weights

    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate all criteria and compute weighted average."""
        results = []
        weighted_sum = 0.0
        total_weight = 0.0

        for criterion, weight in self.criteria_weights:
            result = await criterion.evaluate(actual, expected)
            results.append((result, weight))
            weighted_sum += result.score * weight
            total_weight += weight

        combined_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return self._result(
            combined_score,
            {
                "sub_results": [
                    {"criterion": r.criterion, "score": r.score, "weight": w} for r, w in results
                ],
            },
        )
