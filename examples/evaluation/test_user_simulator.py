"""UserSimulator: let a second model play the user against the weather agent.

The agent under test is OpenAI gpt-5-mini. The simulated user (and the goals judge)
is Google gemini-2.5-flash, so two companies talk to each other here.
Run: pytest examples/evaluation/test_user_simulator.py -v
(needs OPENAI_API_KEY and GEMINI_API_KEY; the summary test runs without keys)
"""

import pytest

from tenxgraph.qa.evaluation import (
    BatchSimulator,
    ConversationScenario,
    CriterionConfig,
    SimulationGoalsCriterion,
    SimulationResult,
    UserSimulator,
)


TOKYO_TRIP = ConversationScenario(
    scenario_id="tokyo_trip",
    description="A traveller wants the current weather in Tokyo",
    starting_prompt="I'm thinking of visiting Tokyo soon. Can you help?",
    conversation_plan=(
        "1. Hint at travel interest\n"
        "2. Ask for the current Tokyo weather\n"
        "3. Confirm the answer is enough"
    ),
    goals=["Get the current weather for Tokyo"],
    max_turns=4,
)

CAPITAL_QUIZ = ConversationScenario(
    scenario_id="capital_quiz",
    description="A user asks a general knowledge question that needs no tool",
    starting_prompt="What is the capital of France?",
    conversation_plan="1. Ask the question\n2. Thank the agent",
    goals=["Learn that Paris is the capital of France"],
    max_turns=2,
)


def goals_judge(threshold: float = 0.5) -> SimulationGoalsCriterion:
    # SimulationGoalsCriterion reads the whole transcript, so it only works inside
    # UserSimulator. Its judge model is gemini-2.5-flash unless judge_model is set.
    return SimulationGoalsCriterion(config=CriterionConfig(threshold=threshold))


def test_batch_summary_counts_results():
    results = [
        SimulationResult(scenario_id="a", turns=2, completed=True, goals_achieved=["g1"]),
        SimulationResult(scenario_id="b", turns=4, completed=False, error="agent error"),
    ]

    summary = BatchSimulator().summary(results)

    assert summary["total_scenarios"] == 2
    assert summary["completed"] == 1
    assert summary["errors"] == 1
    assert summary["average_turns"] == 3


@pytest.mark.needs_openai
@pytest.mark.needs_gemini
async def test_conversation_has_both_roles(app):
    simulator = UserSimulator(model="gemini-2.5-flash", max_turns=4)

    result = await simulator.run(app, TOKYO_TRIP)

    assert result.error is None
    assert {m["role"] for m in result.conversation} == {"user", "assistant"}
    assert result.criterion_scores == {}  # no criteria were attached


@pytest.mark.needs_openai
@pytest.mark.needs_gemini
async def test_goals_are_scored(app):
    simulator = UserSimulator(model="gemini-2.5-flash", criteria=[goals_judge()])

    result = await simulator.run(app, TOKYO_TRIP)

    score = result.criterion_scores["simulation_goals"]
    assert score >= 0.5, result.criterion_details
    assert result.goals_achieved


@pytest.mark.needs_openai
@pytest.mark.needs_gemini
async def test_batch_runs_every_scenario(app):
    simulator = UserSimulator(model="gemini-2.5-flash", criteria=[goals_judge()])
    batch = BatchSimulator(simulator=simulator, max_concurrency=2)

    results = await batch.run_batch(app, [TOKYO_TRIP, CAPITAL_QUIZ])

    assert [r.scenario_id for r in results] == ["tokyo_trip", "capital_quiz"]
    assert batch.summary(results)["errors"] == 0
