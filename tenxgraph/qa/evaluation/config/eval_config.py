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
import json
from pathlib import Path

from pydantic import BaseModel, Field

# Re-export everything from sibling modules so that existing imports of the form
#   from agentflow.qa.evaluation.config.eval_config import XYZ
# continue to work without modification.
from agentflow.qa.evaluation.config.criterion_config import CriterionConfig, Rubric
from agentflow.qa.evaluation.config.reporter_config import ReporterConfig, UserSimulatorConfig
from agentflow.qa.evaluation.config.types import DEFAULT_JUDGE_MODEL, MatchType


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


class EvalConfig(BaseModel):
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
        mock_mode: Whether to run in mock mode (no actual execution).
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

    @classmethod
    def from_file(cls, path: str) -> EvalConfig:
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Loaded EvalConfig instance.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_file(self, path: str) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

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
