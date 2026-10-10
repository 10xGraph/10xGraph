"""Tests for EvalPresets configuration factory."""

import pytest

from tenxgraph.qa.evaluation.config.presets import EvalPresets
from tenxgraph.qa.evaluation.config.eval_config import EvalConfig, MatchType


class TestEvalPresetsResponseQuality:
    """Test EvalPresets.response_quality()."""

    def test_response_quality_default(self):
        """Test response_quality with default parameters."""
        config = EvalPresets.response_quality()
        assert isinstance(config, EvalConfig)
        assert "response_match" in config.criteria.model_dump(exclude_none=True)
        assert "llm_judge" in config.criteria.model_dump(exclude_none=True)

    def test_response_quality_without_llm_judge(self):
        """Test response_quality without LLM judge."""
        config = EvalPresets.response_quality(use_llm_judge=False)
        assert "response_match" in config.criteria.model_dump(exclude_none=True)
        assert "llm_judge" not in config.criteria.model_dump(exclude_none=True)

    def test_response_quality_custom_threshold(self):
        """Test response_quality with custom threshold."""
        config = EvalPresets.response_quality(threshold=0.9)
        assert isinstance(config, EvalConfig)

    def test_response_quality_custom_model(self):
        """Test response_quality with custom judge model."""
        config = EvalPresets.response_quality(judge_model="gpt-4o")
        assert isinstance(config, EvalConfig)


class TestEvalPresetsToolUsage:
    """Test EvalPresets.tool_usage()."""

    def test_tool_usage_default(self):
        """Test tool_usage with default parameters."""
        config = EvalPresets.tool_usage()
        assert isinstance(config, EvalConfig)
        assert "tool_name_match" in config.criteria.model_dump(exclude_none=True)
        assert "trajectory" in config.criteria.model_dump(exclude_none=True)

    def test_tool_usage_strict(self):
        """Test tool_usage with strict=True."""
        config = EvalPresets.tool_usage(strict=True)
        assert isinstance(config, EvalConfig)

    def test_tool_usage_not_strict(self):
        """Test tool_usage with strict=False."""
        config = EvalPresets.tool_usage(strict=False)
        assert isinstance(config, EvalConfig)

    def test_tool_usage_no_args_check(self):
        """Test tool_usage without arg checking."""
        config = EvalPresets.tool_usage(check_args=False)
        assert isinstance(config, EvalConfig)

    def test_tool_usage_custom_threshold(self):
        """Test tool_usage with custom threshold."""
        config = EvalPresets.tool_usage(threshold=0.8)
        assert isinstance(config, EvalConfig)


class TestEvalPresetsConversationFlow:
    """Test EvalPresets.conversation_flow()."""

    def test_conversation_flow_default(self):
        """Test conversation_flow with default parameters."""
        config = EvalPresets.conversation_flow()
        assert isinstance(config, EvalConfig)
        assert "response_match" in config.criteria.model_dump(exclude_none=True)
        assert "trajectory" in config.criteria.model_dump(exclude_none=True)

    def test_conversation_flow_custom_threshold(self):
        """Test conversation_flow with custom threshold."""
        config = EvalPresets.conversation_flow(threshold=0.9)
        assert isinstance(config, EvalConfig)

    def test_conversation_flow_custom_model(self):
        """Test conversation_flow with custom judge model."""
        config = EvalPresets.conversation_flow(judge_model="gpt-4o")
        assert isinstance(config, EvalConfig)


class TestEvalPresetsQuickCheck:
    """Test EvalPresets.quick_check()."""

    def test_quick_check_default(self):
        """Test quick_check returns config with rouge match."""
        config = EvalPresets.quick_check()
        assert isinstance(config, EvalConfig)
        assert "rouge_match" in config.criteria.model_dump(exclude_none=True)

    def test_quick_check_no_llm_criteria(self):
        """Test quick_check has no LLM criteria."""
        config = EvalPresets.quick_check()
        assert "llm_judge" not in config.criteria.model_dump(exclude_none=True)


class TestEvalPresetsComprehensive:
    """Test EvalPresets.comprehensive()."""

    def test_comprehensive_default(self):
        """Test comprehensive preset with default parameters."""
        config = EvalPresets.comprehensive()
        assert isinstance(config, EvalConfig)
        assert "tool_name_match" in config.criteria.model_dump(exclude_none=True)
        assert "rouge_match" in config.criteria.model_dump(exclude_none=True)
        assert "llm_judge" in config.criteria.model_dump(exclude_none=True)
        assert "factual_accuracy" in config.criteria.model_dump(exclude_none=True)
        assert "hallucination" in config.criteria.model_dump(exclude_none=True)
        assert "safety" in config.criteria.model_dump(exclude_none=True)

    def test_comprehensive_without_llm(self):
        """Test comprehensive without LLM criteria."""
        config = EvalPresets.comprehensive(use_llm_judge=False)
        assert "rouge_match" in config.criteria.model_dump(exclude_none=True)
        assert "llm_judge" not in config.criteria.model_dump(exclude_none=True)

    def test_comprehensive_custom_threshold(self):
        """Test comprehensive with custom threshold."""
        config = EvalPresets.comprehensive(threshold=0.9)
        assert isinstance(config, EvalConfig)

    def test_comprehensive_custom_model(self):
        """Test comprehensive with custom judge model."""
        config = EvalPresets.comprehensive(judge_model="gpt-4o")
        assert isinstance(config, EvalConfig)


class TestEvalPresetsSafetyCheck:
    """Test EvalPresets.safety_check()."""

    def test_safety_check_default(self):
        """Test safety_check with default parameters."""
        config = EvalPresets.safety_check()
        assert isinstance(config, EvalConfig)
        assert "hallucination" in config.criteria.model_dump(exclude_none=True)
        assert "safety" in config.criteria.model_dump(exclude_none=True)

    def test_safety_check_custom_threshold(self):
        """Test safety_check with custom threshold."""
        config = EvalPresets.safety_check(threshold=0.9)
        assert isinstance(config, EvalConfig)

    def test_safety_check_custom_model(self):
        """Test safety_check with custom model."""
        config = EvalPresets.safety_check(judge_model="gpt-4o")
        assert isinstance(config, EvalConfig)


class TestEvalPresetsCombine:
    """Test EvalPresets.combine()."""

    def test_combine_two_presets(self):
        """Test combining two presets."""
        config1 = EvalPresets.quick_check()
        config2 = EvalPresets.tool_usage()
        combined = EvalPresets.combine(config1, config2)
        assert "rouge_match" in combined.criteria.model_dump(exclude_none=True)
        assert "tool_name_match" in combined.criteria.model_dump(exclude_none=True)

    def test_combine_three_presets(self):
        """Test combining three presets."""
        c1 = EvalPresets.quick_check()
        c2 = EvalPresets.tool_usage()
        c3 = EvalPresets.safety_check()
        combined = EvalPresets.combine(c1, c2, c3)
        assert "rouge_match" in combined.criteria.model_dump(exclude_none=True)
        assert "tool_name_match" in combined.criteria.model_dump(exclude_none=True)
        assert "safety" in combined.criteria.model_dump(exclude_none=True)

    def test_combine_single_preset(self):
        """Test combining a single preset returns same criteria."""
        original = EvalPresets.quick_check()
        combined = EvalPresets.combine(original)
        assert combined.criteria == original.criteria


class TestEvalPresetsCustom:
    """Test EvalPresets.custom()."""

    def test_custom_empty(self):
        """Test custom with no thresholds returns empty config."""
        config = EvalPresets.custom()
        assert isinstance(config, EvalConfig)
        assert len(config.criteria.model_dump(exclude_none=True)) == 0

    def test_custom_with_response_threshold(self):
        """Test custom with response threshold."""
        config = EvalPresets.custom(response_threshold=0.8)
        assert "response_match" in config.criteria.model_dump(exclude_none=True)

    def test_custom_with_tool_threshold(self):
        """Test custom with tool threshold."""
        config = EvalPresets.custom(tool_threshold=0.9)
        assert "tool_name_match" in config.criteria.model_dump(exclude_none=True)
        assert "trajectory" in config.criteria.model_dump(exclude_none=True)

    def test_custom_with_llm_judge_threshold(self):
        """Test custom with LLM judge threshold."""
        config = EvalPresets.custom(llm_judge_threshold=0.7)
        assert "llm_judge" in config.criteria.model_dump(exclude_none=True)

    def test_custom_with_hallucination_threshold(self):
        """Test custom with hallucination threshold."""
        config = EvalPresets.custom(hallucination_threshold=0.8)
        assert "hallucination" in config.criteria.model_dump(exclude_none=True)

    def test_custom_with_safety_threshold(self):
        """Test custom with safety threshold."""
        config = EvalPresets.custom(safety_threshold=0.9)
        assert "safety" in config.criteria.model_dump(exclude_none=True)

    def test_custom_with_factual_accuracy_threshold(self):
        """Test custom with factual accuracy threshold."""
        config = EvalPresets.custom(factual_accuracy_threshold=0.85)
        assert "factual_accuracy" in config.criteria.model_dump(exclude_none=True)

    def test_custom_all_thresholds(self):
        """Test custom with all thresholds set."""
        config = EvalPresets.custom(
            response_threshold=0.8,
            tool_threshold=0.9,
            llm_judge_threshold=0.7,
            hallucination_threshold=0.8,
            safety_threshold=0.9,
            factual_accuracy_threshold=0.85,
        )
        assert "response_match" in config.criteria.model_dump(exclude_none=True)
        assert "tool_name_match" in config.criteria.model_dump(exclude_none=True)
        assert "llm_judge" in config.criteria.model_dump(exclude_none=True)
        assert "hallucination" in config.criteria.model_dump(exclude_none=True)
        assert "safety" in config.criteria.model_dump(exclude_none=True)
        assert "factual_accuracy" in config.criteria.model_dump(exclude_none=True)

    def test_custom_tool_match_type(self):
        """Test custom with different tool match type."""
        config = EvalPresets.custom(
            tool_threshold=0.8,
            tool_match_type=MatchType.EXACT,
        )
        assert "trajectory" in config.criteria.model_dump(exclude_none=True)

    def test_custom_judge_model(self):
        """Test custom with specific judge model."""
        config = EvalPresets.custom(
            llm_judge_threshold=0.7,
            judge_model="claude-3-5-sonnet",
        )
        assert "llm_judge" in config.criteria.model_dump(exclude_none=True)
