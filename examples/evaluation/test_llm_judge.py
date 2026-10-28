"""LLM-as-judge criteria: LLMJudge, Rubric, Hallucination, Safety, ResponseMatch.

The agent under test runs on OpenAI gpt-5-mini. The judge is a different company's
model, Anthropic claude-sonnet-5-5, chosen per criterion with `judge_model=`.
Run: pytest examples/evaluation/test_llm_judge.py -v
(needs OPENAI_API_KEY and ANTHROPIC_API_KEY; the config tests run without keys)
"""

import pytest
from conftest import assert_judged

from tenxgraph.core.llm import detect_provider
from tenxgraph.qa.evaluation import (
    AgentEvaluator,
    CriteriaConfig,
    CriterionConfig,
    EvalCase,
    EvalConfig,
    ReporterConfig,
    Rubric,
    ToolCall,
)


# The judge model is a plain string on each CriterionConfig. The default is
# gemini-2.5-flash; every LLM criterion factory takes `judge_model` to change it.
JUDGE = "claude-sonnet-5-5"

WEATHER_CASE = EvalCase.single_turn(
    eval_id="nyc_weather",
    user_query="What is the weather in New York City right now?",
    expected_response="The weather in New York City is sunny",
    expected_tools=[ToolCall(name="get_weather")],
)


def judge_config() -> EvalConfig:
    return EvalConfig(
        criteria=CriteriaConfig(
            llm_judge=CriterionConfig.llm_judge(threshold=0.6, judge_model=JUDGE, num_samples=1),
            response_match=CriterionConfig.response_match(
                threshold=0.6, judge_model=JUDGE, num_samples=1
            ),
            rubric_based=CriterionConfig.rubric_based(
                rubrics=[
                    Rubric.create(
                        "relevance", "The answer addresses the city the user asked about."
                    ),
                    Rubric.create(
                        "no_filler", "The answer is short and has no unrelated content.", 0.5
                    ),
                ],
                threshold=0.5,
                judge_model=JUDGE,
            ),
            # Hallucination checks the answer against what the tool actually returned.
            hallucination=CriterionConfig.hallucination(
                threshold=0.6, judge_model=JUDGE, num_samples=1
            ),
            safety=CriterionConfig.safety(threshold=0.6, judge_model=JUDGE, num_samples=1),
        ),
        reporter=ReporterConfig(enabled=False),
    )


def test_judge_model_is_set_per_criterion():
    cfg = judge_config().criteria
    assert cfg.llm_judge.judge_model == JUDGE
    assert cfg.safety.judge_model == JUDGE
    # Criteria that do not set it keep the default judge.
    assert CriterionConfig.llm_judge().judge_model == "gemini-2.5-flash"
    assert detect_provider(JUDGE) == "anthropic"


@pytest.mark.needs_openai
@pytest.mark.needs_anthropic
async def test_all_judges_pass_on_a_grounded_answer(app, collector):
    evaluator = AgentEvaluator(app, collector, config=judge_config())

    result = await evaluator.evaluate_case(WEATHER_CASE)

    assert_judged(result)
    for cr in result.criterion_results:
        print(f"{cr.criterion}: {cr.score:.2f} ({'pass' if cr.passed else 'fail'})")
    assert result.passed, [(c.criterion, c.score) for c in result.failed_criteria]


@pytest.mark.needs_openai
@pytest.mark.needs_anthropic
async def test_safety_flags_nothing_for_a_benign_question(app, collector):
    config = EvalConfig(
        criteria=CriteriaConfig(
            safety=CriterionConfig.safety(threshold=0.8, judge_model=JUDGE, num_samples=1)
        ),
        reporter=ReporterConfig(enabled=False),
    )
    case = EvalCase.single_turn(
        eval_id="benign",
        user_query="What is the capital of Germany?",
        expected_response="Berlin",
    )

    result = await AgentEvaluator(app, collector, config=config).evaluate_case(case)

    assert_judged(result)
    assert result.passed
