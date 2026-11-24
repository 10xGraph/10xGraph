"""Tests for EvalPresets configuration factory."""


from tenxgraph.qa.evaluation.config.eval_config import EvalConfig
from tenxgraph.qa.evaluation.config.presets import EvalPresets


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


class TestEvalPresetsResponseQuality:
    """Test EvalPresets.response_quality()."""

    def test_response_quality_default(self):
        """Test response_quality with default parameters."""
        config = EvalPresets.response_quality()
        assert isinstance(config, EvalConfig)
        assert "response_match" in config.criteria.model_dump(exclude_none=True)
        assert "llm_judge" in config.criteria.model_dump(exclude_none=True)
