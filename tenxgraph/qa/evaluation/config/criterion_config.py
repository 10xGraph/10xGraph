"""
Criterion configuration models for agent evaluation.

This module defines:
    - Rubric: Custom rubric definition for LLM-as-judge evaluation.
    - CriterionConfig: Configuration for individual evaluation criteria.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentflow.qa.evaluation.config.types import DEFAULT_JUDGE_MODEL, MatchType


class Rubric(BaseModel):
    """A custom evaluation rubric.

    Rubrics define specific criteria for evaluating agent behavior
    using LLM-as-judge evaluation.

    Attributes:
        rubric_id: Unique identifier for this rubric.
        content: The rubric description/criteria text.
        weight: Weight of this rubric in overall scoring (default 1.0).
    """

    rubric_id: str
    content: str
    weight: float = 1.0

    @classmethod
    def create(cls, rubric_id: str, content: str, weight: float = 1.0) -> Rubric:
        """Create a rubric with the given parameters."""
        return cls(rubric_id=rubric_id, content=content, weight=weight)


class CriterionConfig(BaseModel):
    """Configuration for a single evaluation criterion.

    Attributes:
        threshold: Minimum score to pass (0.0 to 1.0).
        match_type: Match type for trajectory criteria.
        judge_model: Model to use for LLM-as-judge criteria.
        num_samples: Number of samples for LLM judge (majority vote).
        rubrics: List of custom rubrics for rubric-based criteria.
        keywords: Required keywords for ContainsKeywordsCriterion.
        check_args: Whether to check tool arguments in trajectory matching.
        enabled: Whether this criterion is enabled.
        api_style: OpenAI API style — ``"responses"`` (default, Responses API)
            or ``"chat"`` (legacy Chat Completions, required for some older
            or third-party-hosted models such as Chinese LLMs).
    """

    threshold: float = 0.8
    match_type: MatchType = MatchType.EXACT
    judge_model: str = DEFAULT_JUDGE_MODEL
    num_samples: int = 3
    rubrics: list[Rubric] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    check_args: bool = False
    enabled: bool = True
    api_style: str = "responses"

    @classmethod
    def tool_name_match(cls, threshold: float = 1.0) -> CriterionConfig:
        """Create configuration for tool name matching (no LLM).

        Checks that the names of tools called match the expected list.
        """
        return cls(threshold=threshold)

    @classmethod
    def trajectory(
        cls,
        threshold: float = 1.0,
        match_type: MatchType = MatchType.EXACT,
        check_args: bool = False,
    ) -> CriterionConfig:
        """Create configuration for tool sequence matching (no LLM).

        Use match_type to control strictness:
        - EXACT: same tools, same order, no extras
        - IN_ORDER: expected tools appear in order, extras allowed
        - ANY_ORDER: expected tools appear in any order, extras allowed
        """
        return cls(
            threshold=threshold,
            match_type=match_type,
            check_args=check_args,
        )

    @classmethod
    def node_order(
        cls,
        threshold: float = 1.0,
        match_type: MatchType = MatchType.EXACT,
    ) -> CriterionConfig:
        """Create configuration for node visit order matching (no LLM).

        Checks that the graph visited nodes in the expected order.
        Use match_type to control strictness:
        - EXACT: same nodes, same order, same count
        - IN_ORDER: expected nodes appear in order, extras allowed
        - ANY_ORDER: expected nodes all present, any order
        """
        return cls(threshold=threshold, match_type=match_type)

    @classmethod
    def response_match(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for LLM-based semantic response matching.

        Uses an LLM to judge whether the actual response is semantically
        equivalent to the expected response.  Handles paraphrasing and
        differently-worded but correct answers.

        For a fast, free (no LLM) alternative use :meth:`rouge_match`.
        """
        return cls(threshold=threshold, judge_model=judge_model, num_samples=num_samples)

    @classmethod
    def rouge_match(cls, threshold: float = 0.5) -> CriterionConfig:
        """Create configuration for ROUGE-1 F1 response similarity (no LLM).

        Uses token-overlap (ROUGE-1 F1) to measure similarity between the
        actual and expected response.  No API calls — fast and free.

        Use with criterion key ``"rouge_match"`` in :class:`EvalConfig`.

        For semantic/paraphrase-aware matching use :meth:`response_match`
        (LLM-based) instead.
        """
        return cls(threshold=threshold)

    @classmethod
    def contains_keywords(
        cls,
        keywords: list[str],
        threshold: float = 1.0,
    ) -> CriterionConfig:
        """Create configuration for keyword presence check (no LLM).

        Args:
            keywords: List of keywords that must appear in the actual response.
            threshold: Fraction of keywords that must be present (0.0 to 1.0).
        """
        return cls(threshold=threshold, keywords=keywords)

    @classmethod
    def llm_judge(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for LLM-as-judge overall quality scoring."""
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            num_samples=num_samples,
        )

    @classmethod
    def rubric_based(
        cls,
        rubrics: list[Rubric],
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> CriterionConfig:
        """Create configuration for custom rubric-based scoring."""
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            rubrics=rubrics,
        )

    @classmethod
    def factual_accuracy(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for factual accuracy evaluation.

        Checks whether all stated facts in the response are correct —
        numbers, dates, names, and verifiable claims.
        """
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            num_samples=num_samples,
        )

    @classmethod
    def hallucination(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for hallucination/groundedness detection.

        Checks whether the response is grounded in the context the agent
        actually had (tool results, provided facts). Score 1.0 = fully
        grounded, 0.0 = mostly hallucinated.
        """
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            num_samples=num_samples,
        )

    @classmethod
    def safety(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for safety evaluation.

        Scores the response across five dimensions:
        harmful_content, hate_speech, privacy, misinformation, manipulation.
        The overall score is the minimum across all categories — one unsafe
        category fails the criterion.
        """
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            num_samples=num_samples,
        )
