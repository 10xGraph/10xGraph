"""
test3 — Simulator-based evaluation tests (compact).

Verifies UserSimulator + BatchSimulator + SimulationGoalsCriterion in 3
focused tests covering: conversation mechanics, goal scoring, and batch runs.

Run:
    pytest examples/evaluation/test3/ -v -s
"""

import pytest

pytest.skip(
    "Disabled in examples/evaluation so root pytest runs are unaffected.",
    allow_module_level=True,
)

from tenxgraph.qa.evaluation import (
    BatchSimulator,
    ConversationScenario,
    CriterionConfig,
    SimulationGoalsCriterion,
    UserSimulator,
)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

WEATHER_SINGLE_CITY = ConversationScenario(
    scenario_id="weather_single_city",
    description="User wants to know the current weather in Tokyo for trip planning",
    starting_prompt="I'm thinking of visiting Tokyo soon. Can you help me?",
    conversation_plan=(
        "1. User hints at travel interest\n"
        "2. User explicitly asks for Tokyo weather\n"
        "3. User confirms they got the information they needed"
    ),
    goals=["Get weather information for Tokyo"],
    max_turns=4,
)

GENERAL_KNOWLEDGE = ConversationScenario(
    scenario_id="general_knowledge",
    description="User asks a general knowledge question — no weather tool needed",
    starting_prompt="What is the capital of France?",
    conversation_plan="1. User asks the question\n2. Agent answers directly",
    goals=["Paris is the capital of France"],
    max_turns=2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_judge(threshold: float = 0.5) -> SimulationGoalsCriterion:
    return SimulationGoalsCriterion(config=CriterionConfig(enabled=True, threshold=threshold))


def _failure_msg(result) -> str:
    lines = [f"Scenario: {result.scenario_id}", f"Turns: {result.turns}"]
    for name, score in result.criterion_scores.items():
        reasoning = result.criterion_details.get(name, {}).get("reason", "")
        lines.append(f"  [{name}] score={score:.2f}  {reasoning}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Simulator basics — conversation mechanics + empty scores
# ---------------------------------------------------------------------------


class TestSimulatorBasic:
    @pytest.mark.asyncio
    async def test_scenario_completes_with_both_roles(self, weather_app):
        """Simulator completes without errors; conversation has user + assistant."""
        simulator = UserSimulator(model="gemini/gemini-2.5-flash", max_turns=4)
        result = await simulator.run(weather_app, WEATHER_SINGLE_CITY)

        assert result.error is None
        assert result.turns >= 1
        assert len(result.conversation) >= 2
        roles = {msg["role"] for msg in result.conversation}
        assert "user" in roles and "assistant" in roles
        # No criteria → empty scores
        assert result.criterion_scores == {}


# ---------------------------------------------------------------------------
# 2. Simulator + SimulationGoalsCriterion — goal scoring
# ---------------------------------------------------------------------------


class TestSimulatorWithGoals:
    @pytest.mark.asyncio
    async def test_goals_scored_and_pass(self, weather_app):
        """Weather scenario produces a goals score ≥ 0.5, details populated."""
        judge = _make_judge(threshold=0.5)
        simulator = UserSimulator(model="gemini/gemini-2.5-flash", criteria=[judge])
        result = await simulator.run(weather_app, WEATHER_SINGLE_CITY)

        assert "simulation_goals" in result.criterion_scores, _failure_msg(result)
        score = result.criterion_scores["simulation_goals"]
        assert score >= 0.5, _failure_msg(result)
        details = result.criterion_details.get("simulation_goals", {})
        assert details, "criterion_details should not be empty"


# ---------------------------------------------------------------------------
# 3. BatchSimulator — multi-scenario batch run
# ---------------------------------------------------------------------------


class TestBatchSimulator:
    @pytest.mark.asyncio
    async def test_batch_runs_and_scores(self, weather_app):
        """Batch returns one result per scenario; all carry goals scores."""
        judge = _make_judge(threshold=0.4)
        simulator = UserSimulator(model="gemini/gemini-2.5-flash", criteria=[judge])
        batch = BatchSimulator(simulator=simulator)

        results = await batch.run_batch(weather_app, [WEATHER_SINGLE_CITY, GENERAL_KNOWLEDGE])

        assert len(results) == 2
        for result in results:
            assert (
                "simulation_goals" in result.criterion_scores
            ), f"Missing score for {result.scenario_id}"

        summary = batch.summary(results)
        assert summary["total_scenarios"] == 2
        assert summary["average_turns"] > 0
