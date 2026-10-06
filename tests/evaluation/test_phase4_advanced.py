"""
Tests for Phase 4: Advanced criteria and user simulation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.qa.evaluation import (
    CriterionConfig,
    CriterionResult,
    EvalCase,
    EvalConfig,
    Invocation,
    MatchType,
    MessageContent,
    TrajectoryCollector,
)
from tenxgraph.qa.evaluation.execution.result import ExecutionResult
from tenxgraph.qa.evaluation.criteria.hallucination import HallucinationCriterion
from tenxgraph.qa.evaluation.criteria.safety import SafetyCriterion
from tenxgraph.qa.evaluation.criteria.factual_accuracy import FactualAccuracyCriterion
from tenxgraph.qa.evaluation.simulators.user_simulator import (
    UserSimulator,
    BatchSimulator,
    ConversationScenario,
    SimulationResult,
)


# ============================================================================
# HallucinationCriterion Tests
# ============================================================================


class TestHallucinationCriterion:
    """Tests for the HallucinationCriterion class."""

    def test_init(self):
        """Test criterion initializes correctly."""
        criterion = HallucinationCriterion()
        assert criterion.name == "hallucinations_v1"
        assert criterion.description == "LLM-based hallucination/groundedness detection"

    def test_init_with_config(self):
        """Test criterion initializes with custom config."""
        config = CriterionConfig(threshold=0.9, judge_model="gpt-4")
        criterion = HallucinationCriterion(config=config)
        assert criterion.threshold == 0.9
        assert criterion.config.judge_model == "gpt-4"

    @pytest.mark.asyncio
    async def test_evaluate_no_response(self):
        """Test evaluation with no response returns success."""
        criterion = HallucinationCriterion()

        execution = ExecutionResult()  # actual_response defaults to ""

        case = EvalCase.single_turn(
            eval_id="test",
            user_query="What is the weather?",
        )

        result = await criterion.evaluate(execution, case)
        assert result.passed is True
        assert result.score == 1.0
        assert "No response" in result.details.get("note", "")

    @pytest.mark.asyncio
    async def test_extract_question(self):
        """Test extracting question from eval case."""
        criterion = HallucinationCriterion()

        case = EvalCase.single_turn(
            eval_id="test",
            user_query="What is the capital of France?",
        )

        question = criterion._extract_question(case)
        assert question == "What is the capital of France?"

    @pytest.mark.asyncio
    async def test_extract_actual_response(self):
        """Test that actual_response is read from ExecutionResult."""
        criterion = HallucinationCriterion()

        execution = ExecutionResult(
            actual_response="The capital is Paris.",
        )

        # The evaluate method reads actual.actual_response internally.
        # Verify the field is accessible as expected by criteria.
        assert execution.actual_response == "The capital is Paris."

    @pytest.mark.asyncio
    async def test_extract_context_from_tools(self):
        """Test extracting context from tool results."""
        criterion = HallucinationCriterion()

        from tenxgraph.qa.evaluation import ToolCall

        collector = TrajectoryCollector()
        tc = ToolCall(name="get_info", args={}, result="Paris is the capital")
        collector.tool_calls.append(tc)

        case = EvalCase.single_turn(eval_id="test", user_query="Question?")

        context = criterion._extract_context(collector, case)
        assert "Paris is the capital" in context


class TestSafetyCriterion:
    """Tests for the SafetyCriterion class."""

    def test_init(self):
        """Test criterion initializes correctly."""
        criterion = SafetyCriterion()
        assert criterion.name == "safety_v1"
        assert criterion.description == "LLM-based safety and harmlessness evaluation"

    def test_init_with_config(self):
        """Test criterion initializes with custom config."""
        config = CriterionConfig(threshold=0.95)
        criterion = SafetyCriterion(config=config)
        assert criterion.threshold == 0.95

    @pytest.mark.asyncio
    async def test_evaluate_no_response(self):
        """Test evaluation with no response returns success."""
        criterion = SafetyCriterion()

        execution = ExecutionResult()  # actual_response defaults to ""
        case = EvalCase.single_turn(
            eval_id="test",
            user_query="Hello",
        )

        result = await criterion.evaluate(execution, case)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_extract_question(self):
        """Test extracting question from eval case."""
        criterion = SafetyCriterion()

        case = EvalCase.single_turn(
            eval_id="test",
            user_query="How do I make a cake?",
        )

        question = criterion._extract_question(case)
        assert question == "How do I make a cake?"


class TestFactualAccuracyCriterion:
    """Tests for the FactualAccuracyCriterion class."""

    def test_init(self):
        """Test criterion initializes correctly."""
        criterion = FactualAccuracyCriterion()
        assert criterion.name == "factual_accuracy_v1"

    @pytest.mark.asyncio
    async def test_evaluate_no_response(self):
        """Test evaluation with no response returns success."""
        criterion = FactualAccuracyCriterion()

        execution = ExecutionResult()  # actual_response defaults to ""
        case = EvalCase.single_turn(
            eval_id="test",
            user_query="What is 2+2?",
        )

        result = await criterion.evaluate(execution, case)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_extract_reference(self):
        """Test extracting reference from eval case."""
        criterion = FactualAccuracyCriterion()

        case = EvalCase.single_turn(
            eval_id="test",
            user_query="What is the answer?",
            expected_response="The answer is 42",
        )

        reference = criterion._extract_reference(case)
        assert reference == "The answer is 42"


# ============================================================================
# UserSimulator Tests
# ============================================================================


class TestConversationScenario:
    """Tests for the ConversationScenario model."""

    def test_create_scenario(self):
        """Test creating a conversation scenario."""
        scenario = ConversationScenario(
            scenario_id="test_scenario",
            description="A test conversation",
            starting_prompt="Hello there",
            conversation_plan="1. Greet\n2. Ask question",
            goals=["Get greeting", "Get answer"],
            max_turns=5,
        )

        assert scenario.scenario_id == "test_scenario"
        assert scenario.starting_prompt == "Hello there"
        assert len(scenario.goals) == 2
        assert scenario.max_turns == 5

    def test_default_values(self):
        """Test default values are set."""
        scenario = ConversationScenario()

        assert scenario.scenario_id == ""
        assert scenario.max_turns == 10
        assert scenario.goals == []


class TestSimulationResult:
    """Tests for the SimulationResult model."""

    def test_create_result(self):
        """Test creating a simulation result."""
        result = SimulationResult(
            scenario_id="test",
            turns=3,
            conversation=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            goals_achieved=["Get greeting"],
            completed=True,
        )

        assert result.scenario_id == "test"
        assert result.turns == 3
        assert len(result.conversation) == 2
        assert result.completed is True

    def test_error_result(self):
        """Test creating an error result."""
        result = SimulationResult(
            scenario_id="test",
            error="Something went wrong",
            completed=False,
        )

        assert result.error == "Something went wrong"
        assert result.completed is False


class TestUserSimulator:
    """Tests for the UserSimulator class."""

    def test_init_default(self):
        """Test simulator initializes with defaults."""
        simulator = UserSimulator()

        assert simulator.model == "gemini/gemini-2.5-flash"
        assert simulator.temperature == 0.7
        assert simulator.max_turns == 10

    def test_init_custom(self):
        """Test simulator initializes with custom values."""
        simulator = UserSimulator(
            model="gpt-4",
            temperature=0.5,
            max_turns=15,
        )

        assert simulator.model == "gpt-4"
        assert simulator.temperature == 0.5
        assert simulator.max_turns == 15

    def test_init_with_config(self):
        """Test simulator initializes from config."""
        from tenxgraph.qa.evaluation import UserSimulatorConfig

        config = UserSimulatorConfig(
            model="gpt-3.5-turbo",
            temperature=0.8,
            max_invocations=20,
        )
        simulator = UserSimulator(config=config)

        assert simulator.model == "gpt-3.5-turbo"
        assert simulator.temperature == 0.8
        assert simulator.max_turns == 20

    def test_extract_response(self):
        """Test extracting response from graph result."""
        simulator = UserSimulator()

        result = {
            "messages": [
                {"role": "assistant", "content": "Hello!"},
            ]
        }

        response = simulator._extract_response(result)
        assert response == "Hello!"

    def test_extract_response_empty(self):
        """Test extracting response from empty result."""
        simulator = UserSimulator()

        response = simulator._extract_response({})
        assert response == ""

        response = simulator._extract_response(None)
        assert response == ""

    @pytest.mark.asyncio
    async def test_check_goals_simple(self):
        """Test simple goal checking."""
        simulator = UserSimulator()

        conversation = [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "It's sunny and warm today with a temperature of 75 degrees."},
        ]

        achieved, _ = await simulator._check_goals(
            all_goals=["sunny weather", "temperature"],
            achieved=[],
            conversation=conversation,
        )

        # The goal matching is simple keyword-based, so "sunny" and "weather" should match "sunny weather"
        # and "temperature" should match "temperature"
        assert "temperature" in achieved


class TestBatchSimulator:
    """Tests for the BatchSimulator class."""

    def test_init_default(self):
        """Test batch simulator initializes with defaults."""
        batch = BatchSimulator()
        assert batch.simulator is not None

    def test_init_with_simulator(self):
        """Test batch simulator with custom simulator."""
        simulator = UserSimulator(model="gpt-4")
        batch = BatchSimulator(simulator=simulator)

        assert batch.simulator.model == "gpt-4"

    def test_summary_empty(self):
        """Test summary with no results."""
        batch = BatchSimulator()
        summary = batch.summary([])

        assert summary["total_scenarios"] == 0

    def test_summary_with_results(self):
        """Test summary with results."""
        batch = BatchSimulator()

        results = [
            SimulationResult(
                scenario_id="s1",
                turns=3,
                goals_achieved=["goal1"],
                completed=True,
            ),
            SimulationResult(
                scenario_id="s2",
                turns=5,
                goals_achieved=["goal1", "goal2"],
                completed=False,
            ),
        ]

        summary = batch.summary(results)

        assert summary["total_scenarios"] == 2
        assert summary["completed"] == 1
        assert summary["completion_rate"] == 0.5
        assert summary["total_goals_achieved"] == 3
        assert summary["average_turns"] == 4.0


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestAdvancedCriteriaIntegration:
    """Integration tests for advanced criteria with mocked LLM."""

    @pytest.mark.asyncio
    async def test_hallucination_with_mocked_llm(self):
        """Test hallucination criterion with mocked LLM response."""
        criterion = HallucinationCriterion()

        # Mock the _call_llm_json method used by _run_samples
        async def mock_call_llm_json(prompt):
            from tenxgraph.qa.evaluation.token_usage import TokenUsage
            return ({"score": 0.9, "is_grounded": True, "hallucinations": [], "reasoning": "Response is well grounded"}, TokenUsage())

        criterion._call_llm_json = mock_call_llm_json

        execution = ExecutionResult(
            actual_response="The weather is sunny.",
        )

        case = EvalCase.single_turn(
            eval_id="test",
            user_query="What's the weather?",
        )

        result = await criterion.evaluate(execution, case)
        assert result.passed is True
        assert result.score == 0.9
        assert result.details.get("is_grounded") is True

    @pytest.mark.asyncio
    async def test_safety_with_mocked_llm(self):
        """Test safety criterion with mocked LLM response."""
        criterion = SafetyCriterion()

        async def mock_call_llm_json(prompt):
            from tenxgraph.qa.evaluation.token_usage import TokenUsage
            return ({"score": 1.0, "is_safe": True, "issues": [], "categories": {"harmful_content": 1.0, "hate_speech": 1.0, "privacy": 1.0, "misinformation": 1.0, "manipulation": 1.0}, "reasoning": "Response is safe"}, TokenUsage())

        criterion._call_llm_json = mock_call_llm_json

        execution = ExecutionResult(
            actual_response="Here's how to make a cake...",
        )

        case = EvalCase.single_turn(
            eval_id="test",
            user_query="How do I make a cake?",
        )

        result = await criterion.evaluate(execution, case)
        assert result.passed is True
        assert result.score == 1.0
        assert result.details.get("is_safe") is True

    @pytest.mark.asyncio
    async def test_factual_accuracy_with_mocked_llm(self):
        """Test factual accuracy criterion with mocked LLM response."""
        criterion = FactualAccuracyCriterion()

        async def mock_call_llm_json(prompt):
            from tenxgraph.qa.evaluation.token_usage import TokenUsage
            return ({"score": 0.85, "is_accurate": True, "errors": [], "reasoning": "Most facts are correct"}, TokenUsage())

        criterion._call_llm_json = mock_call_llm_json

        execution = ExecutionResult(
            actual_response="Paris is the capital of France.",
        )

        case = EvalCase.single_turn(
            eval_id="test",
            user_query="What is the capital of France?",
            expected_response="The capital of France is Paris.",
        )

        result = await criterion.evaluate(execution, case)
        assert result.passed is True
        assert result.score == 0.85


class TestTestingUtilities:
    """Tests for the testing utilities module."""

    def test_create_simple_eval_set(self):
        """Test creating a simple eval set."""
        from tenxgraph.qa.evaluation.testing import create_simple_eval_set

        eval_set = create_simple_eval_set(
            "my_tests",
            [
                ("Hello", "Hi!", "greeting"),
                ("What is 2+2?", "4", "math"),
            ],
        )

        assert eval_set.eval_set_id == "my_tests"
        assert len(eval_set.eval_cases) == 2
        assert eval_set.eval_cases[0].name == "greeting"

    def test_assert_eval_passed_success(self):
        """Test assert_eval_passed with passing report."""
        from tenxgraph.qa.evaluation.testing import assert_eval_passed
        from tenxgraph.qa.evaluation import EvalReport, EvalCaseResult, CriterionResult

        report = EvalReport.create(
            eval_set_id="test",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[
                        CriterionResult.success("test", 1.0, 0.8)
                    ],
                )
            ],
        )

        # Should not raise
        assert_eval_passed(report)

    def test_assert_eval_passed_failure(self):
        """Test assert_eval_passed with failing report."""
        from tenxgraph.qa.evaluation.testing import assert_eval_passed
        from tenxgraph.qa.evaluation import EvalReport, EvalCaseResult, CriterionResult

        report = EvalReport.create(
            eval_set_id="test",
            results=[
                EvalCaseResult(
                    eval_id="case1",
                    name="Failed Case",
                    passed=False,
                    criterion_results=[
                        CriterionResult(
                            criterion="test",
                            score=0.5,
                            passed=False,
                            threshold=0.8,
                        )
                    ],
                )
            ],
        )

        with pytest.raises(AssertionError):
            assert_eval_passed(report)

    def test_assert_criterion_passed_success(self):
        """Test assert_criterion_passed with passing criterion."""
        from tenxgraph.qa.evaluation.testing import assert_criterion_passed
        from tenxgraph.qa.evaluation import EvalReport, EvalCaseResult, CriterionResult

        report = EvalReport.create(
            eval_set_id="test",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[
                        CriterionResult.success("my_criterion", 0.9, 0.8)
                    ],
                )
            ],
        )

        # Should not raise
        assert_criterion_passed(report, "my_criterion", min_score=0.8)

    def test_assert_criterion_passed_not_found(self):
        """Test assert_criterion_passed with missing criterion."""
        from tenxgraph.qa.evaluation.testing import assert_criterion_passed
        from tenxgraph.qa.evaluation import EvalReport

        report = EvalReport.create(
            eval_set_id="test",
            results=[],
        )

        with pytest.raises(AssertionError, match="not found"):
            assert_criterion_passed(report, "missing_criterion")
