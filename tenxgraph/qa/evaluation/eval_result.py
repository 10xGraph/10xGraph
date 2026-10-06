"""
Evaluation result models.

This module defines data structures for evaluation outcomes:
    - CriterionResult: Result from a single criterion evaluation
    - EvalCaseResult: Result from evaluating one test case
    - EvalSummary: Aggregate statistics across all cases
    - EvalReport: Complete evaluation report
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from agentflow.qa.evaluation.dataset.eval_set import ToolCall, TrajectoryStep
from agentflow.qa.evaluation.token_usage import TokenUsage


class NodeDetail(BaseModel):
    """Per-node LLM I/O snapshot stored inside EvalCaseResult.

    Captures exactly what was sent to the model and what it returned for
    every node invocation in a case, enabling transparent debugging and
    per-node token accounting in reports.

    Attributes:
        node_name:      Name of the graph node that ran.
        input_messages: Conversation history sent to the LLM.
        response_text:  LLM text output; empty on tool-call turns.
        token_usage:    Tokens consumed by this LLM call.
        timestamp:      Wall-clock time when this invocation completed.
    """

    node_name: str = ""
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    response_text: str = ""
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    timestamp: float = 0.0

    @field_serializer("token_usage")
    def _ser_token_usage(self, v: TokenUsage) -> dict[str, int]:
        return v.to_dict()


class CriterionResult(BaseModel):
    """Result from evaluating a single criterion.

    Attributes:
        criterion: Name of the criterion that was evaluated.
        score: Score from 0.0 to 1.0.
        passed: Whether the score met the threshold.
        threshold: The threshold used for pass/fail.
        details: Additional criterion-specific details.
        error: Error message if evaluation failed.
        token_usage: Tokens consumed by LLM judge calls for this criterion.
    """

    criterion: str
    score: float
    passed: bool
    threshold: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    @field_serializer("token_usage")
    def _ser_token_usage(self, v: TokenUsage) -> dict[str, int]:
        return v.to_dict()

    @property
    def reason(self) -> str | None:
        if self.details and isinstance(self.details, dict):
            return self.details.get("reason")
        return None

    @classmethod
    def success(
        cls,
        criterion: str,
        score: float,
        threshold: float,
        details: dict[str, Any] | None = None,
        token_usage: TokenUsage | None = None,
    ) -> CriterionResult:
        """Create a successful criterion result."""
        return cls(
            criterion=criterion,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details=details or {},
            token_usage=token_usage or TokenUsage(),
        )

    @classmethod
    def failure(
        cls,
        criterion: str,
        error: str,
        threshold: float = 0.0,
    ) -> CriterionResult:
        """Create a failed criterion result due to error."""
        return cls(
            criterion=criterion,
            score=0.0,
            passed=False,
            threshold=threshold,
            error=error,
        )

    @property
    def is_error(self) -> bool:
        """Check if this result represents an error."""
        return self.error is not None


class EvalCaseResult(BaseModel):
    """Result from evaluating a single test case.

    Attributes:
        eval_id: ID of the evaluated case.
        name: Name of the test case.
        passed: Whether the case passed all criteria.
        criterion_results: Results from each criterion.
        actual_trajectory: The actual execution trajectory.
        actual_tool_calls: Actual tool calls made.
        actual_response: The actual final response.
        duration_seconds: Time taken to run this case.
        error: Error message if case evaluation failed.
        metadata: Additional result metadata.
    """

    eval_id: str
    name: str = ""
    passed: bool
    criterion_results: list[CriterionResult] = Field(default_factory=list)
    actual_trajectory: list[TrajectoryStep] = Field(default_factory=list)
    actual_tool_calls: list[ToolCall] = Field(default_factory=list)
    actual_response: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    node_responses: list[dict[str, Any]] = Field(default_factory=list)
    node_visits: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    turn_results: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    agent_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    node_details: list[NodeDetail] = Field(default_factory=list)

    @field_serializer("token_usage")
    def _ser_token_usage(self, v: TokenUsage) -> dict[str, int]:
        return v.to_dict()

    @field_serializer("agent_token_usage")
    def _ser_agent_token_usage(self, v: TokenUsage) -> dict[str, int]:
        return v.to_dict()

    @classmethod
    def success(  # noqa: PLR0913
        cls,
        eval_id: str,
        criterion_results: list[CriterionResult],
        actual_trajectory: list[TrajectoryStep] | None = None,
        actual_tool_calls: list[ToolCall] | None = None,
        actual_response: str = "",
        messages: list[dict[str, Any]] | None = None,
        node_responses: list[dict[str, Any]] | None = None,
        node_visits: list[str] | None = None,
        duration_seconds: float = 0.0,
        name: str = "",
        metadata: dict[str, Any] | None = None,
        turn_results: list[dict[str, Any]] | None = None,
        token_usage: TokenUsage | None = None,
        agent_token_usage: TokenUsage | None = None,
        node_details: list[NodeDetail] | None = None,
    ) -> EvalCaseResult:
        """Create a successful case result."""
        all_passed = all(r.passed for r in criterion_results)
        return cls(
            eval_id=eval_id,
            name=name,
            passed=all_passed,
            criterion_results=criterion_results,
            actual_trajectory=actual_trajectory or [],
            actual_tool_calls=actual_tool_calls or [],
            actual_response=actual_response,
            messages=messages or [],
            node_responses=node_responses or [],
            node_visits=node_visits or [],
            duration_seconds=duration_seconds,
            metadata=metadata or {},
            turn_results=turn_results or [],
            token_usage=token_usage or TokenUsage(),
            agent_token_usage=agent_token_usage or TokenUsage(),
            node_details=node_details or [],
        )

    @classmethod
    def failure(
        cls,
        eval_id: str,
        error: str,
        name: str = "",
        duration_seconds: float = 0.0,
    ) -> EvalCaseResult:
        """Create a failed case result due to error."""
        return cls(
            eval_id=eval_id,
            name=name,
            passed=False,
            error=error,
            duration_seconds=duration_seconds,
        )

    @property
    def is_error(self) -> bool:
        """Check if this result represents an error."""
        return self.error is not None

    def get_criterion_result(self, name: str) -> CriterionResult | None:
        """Get result for a specific criterion."""
        for result in self.criterion_results:
            if result.criterion == name:
                return result
        return None

    @property
    def failed_criteria(self) -> list[CriterionResult]:
        """Get list of criteria that failed."""
        return [r for r in self.criterion_results if not r.passed]

    @property
    def passed_criteria(self) -> list[CriterionResult]:
        """Get list of criteria that passed."""
        return [r for r in self.criterion_results if r.passed]


class EvalSummary(BaseModel):
    """Aggregate statistics across all evaluation cases.

    Counting convention (mirrors DeepEval):
        - passed_cases:  cases where passed=True
        - error_cases:   cases where is_error=True (these also have passed=False)
        - failed_cases:  cases where passed=False AND is_error=False
        - total_cases == passed_cases + failed_cases + error_cases (invariant)

    Attributes:
        total_cases: Total number of cases evaluated.
        passed_cases: Number of cases that passed all criteria.
        failed_cases: Number of cases that failed criteria (not errored).
        error_cases: Number of cases that raised an exception during evaluation.
        pass_rate: Fraction of cases that passed (0.0-1.0).
        avg_duration_seconds: Average time per case.
        total_duration_seconds: Total evaluation time.
        criterion_stats: Per-criterion aggregate statistics.
    """

    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    error_cases: int = 0
    pass_rate: float = 0.0
    avg_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    criterion_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    per_case_token_usage: dict[str, TokenUsage] = Field(default_factory=dict)
    avg_tokens_per_case: float = 0.0

    @field_serializer("total_token_usage")
    def _ser_total(self, v: TokenUsage) -> dict[str, int]:
        return v.to_dict()

    @field_serializer("per_case_token_usage")
    def _ser_per_case(self, v: dict[str, TokenUsage]) -> dict[str, dict[str, int]]:
        return {k: tu.to_dict() for k, tu in v.items()}

    @classmethod
    def from_results(cls, results: list[EvalCaseResult]) -> EvalSummary:
        """Compute summary statistics from a list of case results.

        Args:
            results: List of EvalCaseResult instances.

        Returns:
            EvalSummary with aggregate stats.
        """
        if not results:
            return cls()

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        errored = sum(1 for r in results if r.is_error)
        # failed = cases that are not passed AND not errored
        # (errored cases are counted separately, not double-counted in failed)
        failed = total - passed - errored
        total_duration = sum(r.duration_seconds for r in results)

        # Per-criterion stats
        criterion_stats: dict[str, dict[str, Any]] = {}
        for result in results:
            for cr in result.criterion_results:
                if cr.criterion not in criterion_stats:
                    criterion_stats[cr.criterion] = {
                        "total": 0,
                        "passed": 0,
                        "failed": 0,
                        "avg_score": 0.0,
                        "scores": [],
                    }
                stats = criterion_stats[cr.criterion]
                stats["total"] += 1
                stats["scores"].append(cr.score)
                if cr.passed:
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1

        # Finalise per-criterion averages
        for stats in criterion_stats.values():
            scores = stats.pop("scores")
            stats["avg_score"] = sum(scores) / len(scores) if scores else 0.0
            stats["pass_rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0

        # Token usage aggregation
        per_case_token_usage: dict[str, TokenUsage] = {}
        total_token_usage = TokenUsage()
        for result in results:
            per_case_token_usage[result.eval_id] = result.token_usage
            total_token_usage = total_token_usage + result.token_usage
        avg_tokens_per_case = total_token_usage.total_tokens / total if total > 0 else 0.0

        return cls(
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            error_cases=errored,
            pass_rate=passed / total if total > 0 else 0.0,
            avg_duration_seconds=total_duration / total if total > 0 else 0.0,
            total_duration_seconds=total_duration,
            criterion_stats=criterion_stats,
            total_token_usage=total_token_usage,
            per_case_token_usage=per_case_token_usage,
            avg_tokens_per_case=avg_tokens_per_case,
        )


class EvalReport(BaseModel):
    """Complete evaluation report.

    Attributes:
        eval_set_id: ID of the evaluated set.
        eval_set_name: Name of the evaluated set.
        results: Individual case results.
        summary: Aggregate statistics.
        config_used: Configuration used for evaluation.
        timestamp: When the evaluation was run.
        metadata: Additional report metadata.
    """

    eval_set_id: str
    eval_set_name: str = ""
    results: list[EvalCaseResult] = Field(default_factory=list)
    summary: EvalSummary = Field(default_factory=EvalSummary)
    config_used: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        eval_set_id: str,
        results: list[EvalCaseResult],
        eval_set_name: str = "",
        config_used: dict[str, Any] | None = None,
    ) -> EvalReport:
        """Create a report from a list of case results.

        Args:
            eval_set_id: ID of the eval set.
            results: List of case results.
            eval_set_name: Human-readable name.
            config_used: Config snapshot for provenance.

        Returns:
            EvalReport with computed summary.
        """
        return cls(
            eval_set_id=eval_set_id,
            eval_set_name=eval_set_name,
            results=results,
            summary=EvalSummary.from_results(results),
            config_used=config_used or {},
        )

    @property
    def passed(self) -> bool:
        """Check if all cases passed."""
        return self.summary.pass_rate == 1.0

    @property
    def failed_cases(self) -> list[EvalCaseResult]:
        """Get list of failed cases (includes errored cases)."""
        return [r for r in self.results if not r.passed]

    @property
    def passed_cases(self) -> list[EvalCaseResult]:
        """Get list of passed cases."""
        return [r for r in self.results if r.passed]

    def get_case_result(self, eval_id: str) -> EvalCaseResult | None:
        """Get result for a specific case by ID."""
        for result in self.results:
            if result.eval_id == eval_id:
                return result
        return None

    def to_file(self, path: str) -> None:
        """Save report to a JSON file.

        Args:
            path: Output file path.
        """
        import json
        from pathlib import Path

        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def from_file(cls, path: str) -> EvalReport:
        """Load report from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            EvalReport instance.
        """
        import json
        from pathlib import Path

        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def format_summary(self) -> str:
        """Format summary as a human-readable string."""
        lines = [
            f"Evaluation Report: {self.eval_set_name or self.eval_set_id}",
            "=" * 60,
            f"Total Cases: {self.summary.total_cases}",
            f"Passed:      {self.summary.passed_cases} ({self.summary.pass_rate:.1%})",
            f"Failed:      {self.summary.failed_cases}",
            f"Errors:      {self.summary.error_cases}",
            f"Duration:    {self.summary.total_duration_seconds:.2f}s",
            "",
            "Per-Criterion Results:",
            "-" * 40,
        ]

        for criterion, stats in self.summary.criterion_stats.items():
            lines.append(
                f"  {criterion}: "
                f"{stats['passed']}/{stats['total']} passed, "
                f"avg score: {stats['avg_score']:.2f}"
            )

        if self.failed_cases:
            lines.extend(["", "Failed Cases:", "-" * 40])
            for case in self.failed_cases:
                lines.append(f"  - {case.eval_id}: {case.name}")
                for cr in case.failed_criteria:
                    lines.append(f"      {cr.criterion}: {cr.score:.2f} < {cr.threshold}")

        return "\n".join(lines)
