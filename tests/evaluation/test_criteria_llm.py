"""Tests for LLM-based evaluation criteria.

Tests LLMJudgeCriterion, RubricBasedCriterion, SimulationGoalsCriterion
with mocked _call_llm_json / _call_llm_score to avoid actual LLM calls.
Also tests _parse_model_provider and LLMCallerMixin helper methods.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tenxgraph.qa.evaluation.config.eval_config import CriterionConfig, Rubric
from tenxgraph.qa.evaluation.criteria.llm_judge import LLMJudgeCriterion
from tenxgraph.qa.evaluation.criteria.llm_utils import LLMCallerMixin, _parse_model_provider
from tenxgraph.qa.evaluation.token_usage import TokenUsage
from tenxgraph.qa.evaluation.criteria.rubric import RubricBasedCriterion
from tenxgraph.qa.evaluation.criteria.simulation_goals import SimulationGoalsCriterion
from tenxgraph.qa.evaluation.dataset.eval_set import EvalCase, ToolCall
from tenxgraph.qa.evaluation.execution.result import ExecutionResult


# ---------------------------------------------------------------------------
# _parse_model_provider
# ---------------------------------------------------------------------------

class TestParseModelProvider:
    def test_gemini_plain(self):
        provider, model = _parse_model_provider("gemini-2.5-flash")
        assert provider == "google"
        assert model == "gemini-2.5-flash"

    def test_gemini_slash_prefix(self):
        provider, model = _parse_model_provider("gemini/gemini-2.5-flash")
        assert provider == "google"
        assert model == "gemini-2.5-flash"

    def test_google_slash_prefix(self):
        provider, model = _parse_model_provider("google/gemini-pro")
        assert provider == "google"
        assert model == "gemini-pro"

    def test_openai_plain(self):
        provider, model = _parse_model_provider("gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_openai_slash_prefix(self):
        provider, model = _parse_model_provider("openai/gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"

    def test_o1_model(self):
        provider, model = _parse_model_provider("o1-preview")
        assert provider == "openai"
        assert model == "o1-preview"

    def test_o3_model(self):
        provider, model = _parse_model_provider("o3-mini")
        assert provider == "openai"
        assert model == "o3-mini"

    def test_unknown_prefix_defaults_to_openai(self):
        provider, model = _parse_model_provider("unknown/some-model")
        # unknown prefix → uses the name part, defaults to openai
        assert provider == "openai"
        assert model == "some-model"

    def test_gpt_prefix_slash(self):
        provider, model = _parse_model_provider("gpt/gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"


# ---------------------------------------------------------------------------
# LLMCallerMixin — _call_llm_score
# ---------------------------------------------------------------------------

class TestLLMCallerMixinCallLLMScore:
    @pytest.mark.asyncio
    async def test_call_llm_score_delegates(self):
        """_call_llm_score wraps _call_llm_json and extracts score+reasoning."""

        class TestCriterion(LLMCallerMixin):
            config = CriterionConfig()

        criterion = TestCriterion()

        with patch.object(
            TestCriterion,
            "_call_llm_json",
            new=AsyncMock(return_value=({"score": 0.85, "reasoning": "looks good"}, TokenUsage())),
        ):
            score, reasoning, usage = await criterion._call_llm_score("test prompt")

        assert score == pytest.approx(0.85)
        assert reasoning == "looks good"

    @pytest.mark.asyncio
    async def test_call_llm_score_missing_fields(self):
        """_call_llm_score handles missing dict fields gracefully."""

        class TestCriterion(LLMCallerMixin):
            config = CriterionConfig()

        criterion = TestCriterion()

        with patch.object(
            TestCriterion,
            "_call_llm_json",
            new=AsyncMock(return_value=({}, TokenUsage())),
        ):
            score, reasoning, usage = await criterion._call_llm_score("test prompt")

        assert score == pytest.approx(0.0)
        assert reasoning == ""

    @pytest.mark.asyncio
    async def test_call_llm_json_no_provider_available(self):
        """Returns default dict when no LLM library is available."""

        class TestCriterion(LLMCallerMixin):
            config = CriterionConfig(judge_model="gemini-2.5-flash")

        criterion = TestCriterion()

        with (
            patch.object(TestCriterion, "_call_google_json", new=AsyncMock(return_value=None)),
            patch.object(TestCriterion, "_call_openai_json", new=AsyncMock(return_value=None)),
        ):
            result, usage = await criterion._call_llm_json("prompt")

        assert result["score"] == pytest.approx(0.5)
        assert "reasoning" in result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(response: str = "The capital of France is Paris.") -> ExecutionResult:
    return ExecutionResult(actual_response=response)


def _make_case(expected_response: str = "Paris") -> EvalCase:
    return EvalCase.single_turn(
        eval_id="test",
        user_query="What is the capital of France?",
        expected_response=expected_response,
    )


# ---------------------------------------------------------------------------
# LLMJudgeCriterion
# ---------------------------------------------------------------------------

class TestLLMJudgeCriterion:
    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        criterion = LLMJudgeCriterion(config=CriterionConfig(threshold=0.7, num_samples=1))
        actual = _make_result("The capital is Paris.")
        expected = _make_case("Paris")

        with patch.object(
            LLMJudgeCriterion,
            "_call_llm_json",
            new=AsyncMock(return_value=({"score": 0.9, "reasoning": "good match"}, TokenUsage())),
        ):
            result = await criterion.evaluate(actual, expected)

        assert result.score == pytest.approx(0.9)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_no_expected_response(self):
        criterion = LLMJudgeCriterion(config=CriterionConfig(threshold=0.5))
        actual = _make_result("some response")
        expected = EvalCase.single_turn(
            eval_id="test",
            user_query="question",
            expected_response=None,
        )
        result = await criterion.evaluate(actual, expected)
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_evaluate_multiple_samples_averaged(self):
        criterion = LLMJudgeCriterion(config=CriterionConfig(threshold=0.5, num_samples=3))
        actual = _make_result("Paris is the capital.")
        expected = _make_case("Paris")

        call_count = 0

        async def mock_llm_json(self_arg, prompt):
            nonlocal call_count
            call_count += 1
            score = call_count * 0.1 + 0.5
            return ({"score": score, "reasoning": f"sample {call_count}"}, TokenUsage())

        with patch.object(LLMJudgeCriterion, "_call_llm_json", new=mock_llm_json):
            result = await criterion.evaluate(actual, expected)

        # Scores: 0.6, 0.7, 0.8 → average = 0.7
        assert result.score == pytest.approx(0.7)
        assert result.details["samples"] == 3

    @pytest.mark.asyncio
    async def test_evaluate_exception_returns_failure(self):
        """When _run_samples itself raises, evaluate() returns failure result."""
        criterion = LLMJudgeCriterion()
        actual = _make_result("response")
        expected = _make_case("expected")

        with patch.object(
            LLMJudgeCriterion,
            "_run_samples",
            side_effect=Exception("LLM error"),
        ):
            result = await criterion.evaluate(actual, expected)

        assert result.passed is False
        assert "LLM error" in result.error

    @pytest.mark.asyncio
    async def test_evaluate_all_samples_fail(self):
        """If all LLM samples raise, score should be 0.0."""
        criterion = LLMJudgeCriterion(config=CriterionConfig(threshold=0.5, num_samples=2))
        actual = _make_result("response")
        expected = _make_case("expected")

        # _run_samples catches individual sample errors, not the outer evaluate
        # So we need to mock at the _run_samples level
        with patch.object(
            LLMJudgeCriterion,
            "_run_samples",
            new=AsyncMock(return_value=([], [], [], TokenUsage())),
        ):
            result = await criterion.evaluate(actual, expected)

        # no scores → final_score = 0.0
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_extract_question_empty_conversation(self):
        criterion = LLMJudgeCriterion()
        expected = EvalCase(eval_id="test")
        question = criterion._extract_question(expected)
        assert question == ""

    @pytest.mark.asyncio
    async def test_extract_expected_response_from_last_invocation(self):
        criterion = LLMJudgeCriterion()
        expected = _make_case("final answer")
        resp = criterion._extract_last_expected_response(expected)
        assert resp == "final answer"


# ---------------------------------------------------------------------------
# RubricBasedCriterion
# ---------------------------------------------------------------------------

class TestRubricBasedCriterion:
    def _make_config_with_rubrics(self, rubrics=None, num_samples=1):
        rubrics = rubrics or [
            Rubric(rubric_id="accuracy", content="Is the answer accurate?"),
            Rubric(rubric_id="clarity", content="Is the answer clear?"),
        ]
        return CriterionConfig(rubrics=rubrics, num_samples=num_samples, threshold=0.6)

    @pytest.mark.asyncio
    async def test_evaluate_with_rubrics(self):
        criterion = RubricBasedCriterion(config=self._make_config_with_rubrics())
        actual = _make_result("Paris")
        expected = _make_case("Paris")

        with patch.object(
            RubricBasedCriterion,
            "_call_llm_json",
            new=AsyncMock(return_value=({"score": 0.9, "reasoning": "accurate and clear"}, TokenUsage())),
        ):
            result = await criterion.evaluate(actual, expected)

        assert result.score == pytest.approx(0.9)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_no_rubrics(self):
        criterion = RubricBasedCriterion(config=CriterionConfig(rubrics=[]))
        actual = _make_result("ok")
        expected = _make_case("ok")
        result = await criterion.evaluate(actual, expected)
        assert result.score == pytest.approx(1.0)
        assert result.details["note"] == "No rubrics configured"

    @pytest.mark.asyncio
    async def test_evaluate_multiple_samples(self):
        rubrics = [Rubric(rubric_id="r1", content="Check accuracy")]
        criterion = RubricBasedCriterion(
            config=CriterionConfig(rubrics=rubrics, num_samples=2, threshold=0.5)
        )
        actual = _make_result("answer")
        expected = _make_case("expected")

        sample_idx = 0

        async def mock_score(self_arg, prompt):
            nonlocal sample_idx
            sample_idx += 1
            score = 0.8 if sample_idx % 2 == 0 else 0.6
            return ({"score": score, "reasoning": f"sample {sample_idx}"}, TokenUsage())

        with patch.object(RubricBasedCriterion, "_call_llm_json", new=mock_score):
            result = await criterion.evaluate(actual, expected)

        # Scores: 0.6, 0.8 → average = 0.7
        assert result.score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_evaluate_sample_exception_skipped(self):
        """Exceptions in individual samples are logged and skipped."""
        rubrics = [Rubric(rubric_id="r1", content="accuracy")]
        criterion = RubricBasedCriterion(
            config=CriterionConfig(rubrics=rubrics, num_samples=2, threshold=0.5)
        )
        actual = _make_result("answer")
        expected = _make_case("expected")

        call_count = 0

        async def mock_score_raises(self_arg, prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("LLM error")
            return ({"score": 0.8, "reasoning": "good"}, TokenUsage())

        with patch.object(RubricBasedCriterion, "_call_llm_json", new=mock_score_raises):
            result = await criterion.evaluate(actual, expected)

        # Only one sample succeeded
        assert result.score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_evaluate_all_samples_fail_score_zero(self):
        rubrics = [Rubric(rubric_id="r1", content="accuracy")]
        criterion = RubricBasedCriterion(
            config=CriterionConfig(rubrics=rubrics, num_samples=1, threshold=0.5)
        )
        actual = _make_result("answer")
        expected = _make_case("expected")

        with patch.object(
            RubricBasedCriterion,
            "_call_llm_json",
            side_effect=Exception("always fails"),
        ):
            result = await criterion.evaluate(actual, expected)

        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_evaluate_outer_exception_returns_failure(self):
        rubrics = [Rubric(rubric_id="r1", content="accuracy")]
        criterion = RubricBasedCriterion(config=CriterionConfig(rubrics=rubrics))

        # Force an exception from _extract_question
        with patch.object(
            RubricBasedCriterion,
            "_extract_question",
            side_effect=RuntimeError("unexpected"),
        ):
            result = await criterion.evaluate(_make_result(), _make_case())

        assert result.passed is False
        assert "unexpected" in result.error

    def test_extract_question_from_conversation(self):
        criterion = RubricBasedCriterion()
        expected = _make_case("answer")
        question = criterion._extract_question(expected)
        assert "capital" in question.lower() or question != ""

    def test_extract_question_empty_conversation(self):
        criterion = RubricBasedCriterion()
        expected = EvalCase(eval_id="empty")
        question = criterion._extract_question(expected)
        assert question == ""


# ---------------------------------------------------------------------------
# SimulationGoalsCriterion
# ---------------------------------------------------------------------------

class TestSimulationGoalsCriterion:
    def _make_goal_case(self, goals: str) -> EvalCase:
        """Create an EvalCase where expected response contains goals text."""
        return EvalCase.single_turn(
            eval_id="sim_test",
            user_query="conversation transcript here",
            expected_response=goals,
        )

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        criterion = SimulationGoalsCriterion(config=CriterionConfig(threshold=0.7))
        actual = ExecutionResult(
            actual_response="User asked about billing; agent solved it completely."
        )
        expected = self._make_goal_case("Resolve billing issue; Confirm resolution")

        with patch.object(
            SimulationGoalsCriterion,
            "_call_llm_json",
            new=AsyncMock(
                return_value=(
                    {
                        "score": 0.85,
                        "achieved_goals": ["Resolve billing issue"],
                        "unachieved_goals": [],
                        "reasoning": "Both goals addressed",
                    },
                    TokenUsage(),
                )
            ),
        ):
            result = await criterion.evaluate(actual, expected)

        assert result.score == pytest.approx(0.85)
        assert result.passed is True
        assert "achieved_goals" in result.details

    @pytest.mark.asyncio
    async def test_evaluate_no_goals_returns_pass(self):
        """If no goals defined (empty expected response), returns 1.0."""
        criterion = SimulationGoalsCriterion()
        actual = ExecutionResult(actual_response="transcript")
        # Create a case with no expected_response (empty string)
        expected = EvalCase(eval_id="test")  # no conversation, no goals
        result = await criterion.evaluate(actual, expected)
        assert result.score == pytest.approx(1.0)
        assert result.details["note"] == "No goals defined — skipping evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_exception_returns_failure(self):
        criterion = SimulationGoalsCriterion()
        actual = ExecutionResult(actual_response="transcript")
        expected = self._make_goal_case("some goals")

        with patch.object(
            SimulationGoalsCriterion,
            "_call_llm_json",
            side_effect=Exception("API error"),
        ):
            result = await criterion.evaluate(actual, expected)

        assert result.passed is False
        assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_evaluate_partial_goals(self):
        criterion = SimulationGoalsCriterion(config=CriterionConfig(threshold=0.5))
        actual = ExecutionResult(actual_response="transcript")
        expected = self._make_goal_case("Goal1; Goal2; Goal3")

        with patch.object(
            SimulationGoalsCriterion,
            "_call_llm_json",
            new=AsyncMock(
                return_value=(
                    {
                        "score": 0.33,
                        "achieved_goals": ["Goal1"],
                        "unachieved_goals": ["Goal2", "Goal3"],
                        "reasoning": "Partial",
                    },
                    TokenUsage(),
                )
            ),
        ):
            result = await criterion.evaluate(actual, expected)

        assert result.score == pytest.approx(0.33)
        assert result.passed is False  # 0.33 < threshold=0.5

    def test_extract_goals_from_invocation(self):
        criterion = SimulationGoalsCriterion()
        expected = self._make_goal_case("Goal1; Goal2")
        goals = criterion._extract_last_expected_response(expected)
        assert goals == "Goal1; Goal2"

    def test_extract_goals_empty_conversation(self):
        criterion = SimulationGoalsCriterion()
        expected = EvalCase(eval_id="empty")
        goals = criterion._extract_last_expected_response(expected)
        assert goals == ""

    @pytest.mark.asyncio
    async def test_evaluate_details_populated(self):
        criterion = SimulationGoalsCriterion()
        actual = ExecutionResult(actual_response="transcript")
        expected = self._make_goal_case("Solve problem")

        with patch.object(
            SimulationGoalsCriterion,
            "_call_llm_json",
            new=AsyncMock(
                return_value=(
                    {
                        "score": 1.0,
                        "achieved_goals": ["Solve problem"],
                        "unachieved_goals": [],
                        "reasoning": "Done",
                    },
                    TokenUsage(),
                )
            ),
        ):
            result = await criterion.evaluate(actual, expected)

        assert result.details["judge_model"] == criterion.config.judge_model
        assert "reasoning" in result.details
        assert "reason" in result.details
