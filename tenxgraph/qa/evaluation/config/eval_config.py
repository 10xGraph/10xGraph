"""
Evaluation configuration models.

This module defines configuration structures for agent evaluation:
    - CriteriaConfig: Type-safe container for which criteria to enable.
    - EvalConfig: Main configuration container.

Supporting models are defined in sibling modules:
    - types.py: DEFAULT_JUDGE_MODEL, MatchType
    - criterion_config.py: Rubric, CriterionConfig
    - reporter_config.py: UserSimulatorConfig, ReporterConfig
"""

from __future__ import annotations

import copy
import warnings
from typing import Self

from pydantic import BaseModel, Field, model_validator

# Re-export everything from sibling modules so that existing imports of the form
#   from tenxgraph.qa.evaluation.config.eval_config import XYZ
# continue to work without modification.
from tenxgraph.qa.evaluation._json_file import _JsonFileModel
from tenxgraph.qa.evaluation.config.criterion_config import CriterionConfig, Rubric
from tenxgraph.qa.evaluation.config.reporter_config import ReporterConfig, UserSimulatorConfig
from tenxgraph.qa.evaluation.config.types import DEFAULT_JUDGE_MODEL, MatchType


__all__ = [
    "CriteriaConfig",
    "CriterionConfig",
    "DEFAULT_JUDGE_MODEL",
    "EvalConfig",
    "MatchType",
    "ReporterConfig",
    "Rubric",
    "UserSimulatorConfig",
]


class CriteriaConfig(BaseModel):
    """Type-safe configuration for evaluation criteria.

    All fields are optional — only include the criteria you need.
    Field names map directly to registry keys used by AgentEvaluator.
    """

    model_config = {"extra": "forbid"}

    tool_name_match: CriterionConfig | None = None
    trajectory: CriterionConfig | None = None
    node_order: CriterionConfig | None = None
    response_match: CriterionConfig | None = None
    rouge_match: CriterionConfig | None = None
    contains_keywords: CriterionConfig | None = None
    llm_judge: CriterionConfig | None = None
    rubric_based: CriterionConfig | None = None
    factual_accuracy: CriterionConfig | None = None
    hallucination: CriterionConfig | None = None
    safety: CriterionConfig | None = None
    simulation_goals: CriterionConfig | None = None


class EvalConfig(_JsonFileModel):
    """Main evaluation configuration.

    Contains all settings for running an evaluation, including
    which criteria to use and their thresholds.

    Attributes:
        criteria: Type-safe criteria configuration (CriteriaConfig instance).
        user_simulator_config: Configuration for user simulation.
        parallel: Whether to run evaluations in parallel.
        max_concurrency: Maximum concurrent evaluations if parallel.
        timeout: Timeout for each evaluation case (seconds).
        verbose: Whether to output verbose logging.
        mock_mode: Deprecated, no effect, removed in 2.0.
        reporter: Configuration for automatic report generation.
    """

    criteria: CriteriaConfig = Field(default_factory=CriteriaConfig)
    user_simulator_config: UserSimulatorConfig | None = None
    parallel: bool = False
    max_concurrency: int = 4
    timeout: float = 300.0
    verbose: bool = False
    mock_mode: bool = False
    reporter: ReporterConfig = Field(default_factory=ReporterConfig)

    @model_validator(mode="after")
    def _warn_unused_fields(self) -> Self:
        if self.mock_mode:
            warnings.warn(
                "EvalConfig.mock_mode has no effect and will be removed in 2.0.",
                DeprecationWarning,
                stacklevel=3,
            )
        return self

    @classmethod
    def default(cls) -> EvalConfig:
        """Create default evaluation configuration.

        Default includes:
            - tool_trajectory_avg_score: EXACT match, threshold 1.0
            - response_match_score: ROUGE-1, threshold 0.8
        """
        return cls(
            criteria=CriteriaConfig(
                trajectory=CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.EXACT,
                ),
                response_match=CriterionConfig.response_match(
                    threshold=0.8,
                ),
            )
        )

    @classmethod
    def strict(cls) -> EvalConfig:
        """Create strict evaluation configuration.

        All criteria set to maximum strictness.
        """
        return cls(
            criteria=CriteriaConfig(
                trajectory=CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.EXACT,
                    check_args=True,
                ),
                response_match=CriterionConfig.response_match(
                    threshold=0.9,
                ),
                llm_judge=CriterionConfig.llm_judge(
                    threshold=0.9,
                    num_samples=5,
                ),
            )
        )

    @classmethod
    def relaxed(cls) -> EvalConfig:
        """Create relaxed evaluation configuration.

        Uses IN_ORDER matching and lower thresholds.
        """
        return cls(
            criteria=CriteriaConfig(
                trajectory=CriterionConfig.trajectory(
                    threshold=0.8,
                    match_type=MatchType.IN_ORDER,
                    check_args=False,
                ),
                response_match=CriterionConfig.response_match(
                    threshold=0.6,
                ),
            )
        )

    def get_criterion_config(self, name: str) -> CriterionConfig | None:
        """Get configuration for a specific criterion by field name."""
        return getattr(self.criteria, name, None)

    def enable_criterion(
        self,
        name: str,
        config: CriterionConfig | None = None,
    ) -> None:
        """Enable a named criterion by field name."""
        if name not in CriteriaConfig.model_fields:
            raise ValueError(f"Unknown criterion field: {name!r}")
        setattr(self.criteria, name, config or CriterionConfig())

    def disable_criterion(self, name: str) -> None:
        """Disable a criterion by field name."""
        criterion = getattr(self.criteria, name, None)
        if criterion is not None:
            criterion.enabled = False

    def with_rubrics(self, rubrics: list[Rubric]) -> EvalConfig:
        """Return a copy with rubric-based criteria added."""
        new_config = copy.deepcopy(self)
        new_config.criteria.rubric_based = CriterionConfig.rubric_based(
            rubrics=rubrics,
        )
        return new_config
