"""Preset evaluation configurations for common scenarios."""

from agentflow.qa.evaluation.config.eval_config import (
    DEFAULT_JUDGE_MODEL,
    CriteriaConfig,
    CriterionConfig,
    EvalConfig,
    MatchType,
)


class EvalPresets:
    """Preset evaluation configurations for common use cases.

    Provides ready-to-use configurations for:
    - Response quality checking
    - Tool usage validation
    - Conversation flow testing
    - Combined scenarios

    Every preset that creates LLM-based criteria accepts an optional
    ``judge_model`` parameter (defaults to ``"gemini-2.5-flash"``).

    Example:
        ```python
        # Use a preset with default Gemini judge
        config = EvalPresets.response_quality()

        # Override the judge model
        config = EvalPresets.comprehensive(judge_model="gpt-4o")

        # Combine presets
        config = EvalPresets.combine(
            EvalPresets.response_quality(),
            EvalPresets.tool_usage(),
        )
        ```
    """

    @classmethod
    def response_quality(
        cls,
        threshold: float = 0.7,
        use_llm_judge: bool = True,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> EvalConfig:
        """Preset for checking response quality.

        Focuses on whether the agent's responses are relevant and accurate.

        Args:
            threshold: Minimum score to pass (0.0-1.0)
            use_llm_judge: Whether to use LLM-as-judge for evaluation
            judge_model: Model to use for LLM-based criteria

        Returns:
            EvalConfig with response quality criteria
        """
        return EvalConfig(
            criteria=CriteriaConfig(
                response_match=CriterionConfig.response_match(
                    threshold=threshold,
                    judge_model=judge_model,
                ),
                llm_judge=CriterionConfig.llm_judge(
                    threshold=threshold,
                    num_samples=1,
                    judge_model=judge_model,
                )
                if use_llm_judge
                else None,
            )
        )

    @classmethod
    def tool_usage(
        cls,
        threshold: float = 1.0,
        strict: bool = True,
        check_args: bool = True,
    ) -> EvalConfig:
        """Preset for validating tool usage.

        Checks whether the agent calls the right tools with correct arguments.

        Args:
            threshold: Minimum score to pass
            strict: Whether to require exact tool matches (EXACT vs IN_ORDER)
            check_args: Whether to validate tool arguments

        Returns:
            EvalConfig with tool usage criteria
        """
        match_type = MatchType.EXACT if strict else MatchType.IN_ORDER

        return EvalConfig(
            criteria=CriteriaConfig(
                tool_name_match=CriterionConfig.tool_name_match(threshold=threshold),
                trajectory=CriterionConfig.trajectory(
                    threshold=threshold,
                    match_type=match_type,
                    check_args=check_args,
                ),
            )
        )

    @classmethod
    def conversation_flow(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> EvalConfig:
        """Preset for testing conversation flow.

        Validates both responses and tool usage in multi-turn conversations.

        Args:
            threshold: Minimum score to pass
            judge_model: Model to use for LLM-based criteria

        Returns:
            EvalConfig for conversation testing
        """
        return EvalConfig(
            criteria=CriteriaConfig(
                response_match=CriterionConfig.response_match(
                    threshold=threshold,
                    judge_model=judge_model,
                ),
                trajectory=CriterionConfig.trajectory(
                    threshold=threshold,
                    match_type=MatchType.IN_ORDER,
                ),
            )
        )

    @classmethod
    def quick_check(cls) -> EvalConfig:
        """Minimal preset for quick sanity checks (no LLM, instant feedback).

        Uses ROUGE-1 token overlap for a fast free check during development.
        Switch to ``response_quality()`` for semantic accuracy.

        Returns:
            EvalConfig with no-LLM criteria
        """
        return EvalConfig(
            criteria=CriteriaConfig(
                rouge_match=CriterionConfig.rouge_match(threshold=0.5),
            )
        )

    @classmethod
    def comprehensive(
        cls,
        threshold: float = 0.8,
        use_llm_judge: bool = True,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> EvalConfig:
        """Comprehensive evaluation with all available criteria.

        Includes no-LLM criteria (tool matching, keyword, response) and all
        LLM-based criteria (judge, factual accuracy, hallucination, safety).

        Args:
            threshold: Minimum score to pass
            use_llm_judge: Whether to include LLM-based criteria
            judge_model: Model to use for LLM-based criteria

        Returns:
            EvalConfig with all criteria enabled
        """
        return EvalConfig(
            criteria=CriteriaConfig(
                tool_name_match=CriterionConfig.tool_name_match(threshold=1.0),
                trajectory=CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.IN_ORDER,
                    check_args=True,
                ),
                rouge_match=CriterionConfig.rouge_match(threshold=threshold),
                llm_judge=CriterionConfig.llm_judge(
                    threshold=threshold,
                    judge_model=judge_model,
                )
                if use_llm_judge
                else None,
                factual_accuracy=CriterionConfig.factual_accuracy(
                    threshold=threshold,
                    judge_model=judge_model,
                )
                if use_llm_judge
                else None,
                hallucination=CriterionConfig.hallucination(
                    threshold=threshold,
                    judge_model=judge_model,
                )
                if use_llm_judge
                else None,
                safety=CriterionConfig.safety(
                    threshold=threshold,
                    judge_model=judge_model,
                )
                if use_llm_judge
                else None,
            )
        )

    @classmethod
    def safety_check(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> EvalConfig:
        """Preset focused on safety and hallucination detection.

        Args:
            threshold: Minimum score to pass
            judge_model: Model to use for LLM-based criteria

        Returns:
            EvalConfig with safety and hallucination criteria
        """
        return EvalConfig(
            criteria=CriteriaConfig(
                hallucination=CriterionConfig.hallucination(
                    threshold=threshold,
                    judge_model=judge_model,
                ),
                safety=CriterionConfig.safety(
                    threshold=threshold,
                    judge_model=judge_model,
                ),
            )
        )

    @classmethod
    def combine(cls, *configs: EvalConfig) -> EvalConfig:
        """Combine multiple preset configurations.

        Args:
            *configs: Multiple EvalConfig instances to merge

        Returns:
            Combined EvalConfig with all criteria
        """
        combined: dict = {}
        for config in configs:
            for field in CriteriaConfig.model_fields:
                value = getattr(config.criteria, field)
                if value is not None:
                    combined[field] = value

        return EvalConfig(criteria=CriteriaConfig(**combined))

    @classmethod
    def custom(
        cls,
        response_threshold: float | None = None,
        tool_threshold: float | None = None,
        llm_judge_threshold: float | None = None,
        tool_match_type: MatchType = MatchType.IN_ORDER,
        check_tool_args: bool = True,
        hallucination_threshold: float | None = None,
        safety_threshold: float | None = None,
        factual_accuracy_threshold: float | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> EvalConfig:
        """Create custom configuration from individual parameters.

        Args:
            response_threshold: Enable response matching with this threshold
            tool_threshold: Enable tool name + trajectory matching with this threshold
            llm_judge_threshold: Enable LLM judge with this threshold
            tool_match_type: How to match tool trajectories
            check_tool_args: Whether to check tool arguments
            hallucination_threshold: Enable hallucination detection with this threshold
            safety_threshold: Enable safety checking with this threshold
            factual_accuracy_threshold: Enable factual accuracy with this threshold
            judge_model: Model to use for LLM-based criteria

        Returns:
            Custom EvalConfig
        """
        return EvalConfig(
            criteria=CriteriaConfig(
                response_match=CriterionConfig.response_match(
                    threshold=response_threshold,
                    judge_model=judge_model,
                )
                if response_threshold is not None
                else None,
                tool_name_match=CriterionConfig.tool_name_match(
                    threshold=tool_threshold,
                )
                if tool_threshold is not None
                else None,
                trajectory=CriterionConfig.trajectory(
                    threshold=tool_threshold,
                    match_type=tool_match_type,
                    check_args=check_tool_args,
                )
                if tool_threshold is not None
                else None,
                llm_judge=CriterionConfig.llm_judge(
                    threshold=llm_judge_threshold,
                    judge_model=judge_model,
                )
                if llm_judge_threshold is not None
                else None,
                hallucination=CriterionConfig.hallucination(
                    threshold=hallucination_threshold,
                    judge_model=judge_model,
                )
                if hallucination_threshold is not None
                else None,
                safety=CriterionConfig.safety(
                    threshold=safety_threshold,
                    judge_model=judge_model,
                )
                if safety_threshold is not None
                else None,
                factual_accuracy=CriterionConfig.factual_accuracy(
                    threshold=factual_accuracy_threshold,
                    judge_model=judge_model,
                )
                if factual_accuracy_threshold is not None
                else None,
            )
        )
