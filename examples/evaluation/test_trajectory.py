"""Tool trajectory and response match criteria with AgentEvaluator.

Part 1 builds datasets (offline). Part 2 runs the full evaluator against a
scripted agent, so the wiring is checked without an API key. Part 3 evaluates the
real gpt-5-mini weather agent and needs OPENAI_API_KEY.
Run: pytest examples/evaluation/test_trajectory.py -v
"""

from pathlib import Path

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.qa.evaluation import (
    AgentEvaluator,
    CriteriaConfig,
    CriterionConfig,
    EvalCase,
    EvalConfig,
    EvalSet,
    EvalSetBuilder,
    MatchType,
    ReporterConfig,
    ToolCall,
    assert_criterion_passed,
    assert_eval_passed,
    create_eval_app,
    create_simple_eval_set,
    parametrize_eval_cases,
    run_eval,
)
from tenxgraph.qa.testing import TestAgent as ScriptedAgent  # alias: pytest would collect "Test*"
from tenxgraph.utils.constants import END


# Reporters write files by default; turn them off so tests leave nothing behind.
QUIET = ReporterConfig(enabled=False)
EVAL_FILE = str(Path(__file__).parent / "weather.evalset.json")


def weather_eval_set() -> EvalSet:
    return (
        EvalSetBuilder("weather_trajectory")
        .add_case(
            query="What is the weather in New York City right now?",
            expected="The weather in New York City is sunny",
            expected_tools=[ToolCall(name="get_weather", args={"location": "New York City"})],
            case_id="nyc_weather",
        )
        .add_case(
            query="Give me a 3-day forecast for Tokyo.",
            expected="3-day forecast for Tokyo: sunny, cloudy, sunny",
            expected_tools=["get_forecast"],  # plain names are fine when args do not matter
            case_id="tokyo_forecast",
        )
        .add_case(
            query="What is the capital of Germany?",
            expected="Berlin",
            expected_tools=[],  # empty list means the agent must not call any tool
            case_id="no_tool",
        )
        .build()
    )


# ── Part 1: datasets (offline) ──────────────────────────────────────────────


def test_eval_set_round_trips_through_json(tmp_path):
    path = tmp_path / "weather.evalset.json"
    EvalSetBuilder("weather_trajectory").add_case("Hi", "Hello").save(str(path))

    loaded = EvalSet.from_file(str(path))

    assert [c.eval_id for c in loaded.eval_cases] == ["case_1"]


def test_multi_turn_case_has_one_invocation_per_turn():
    case = EvalCase.multi_turn(
        eval_id="followup",
        conversation=[
            ("What is the weather in London?", "sunny"),
            ("And a 5-day forecast?", "forecast"),
        ],
        expected_tools=[ToolCall(name="get_weather")],  # attached to the first turn only
    )
    assert len(case.conversation) == 2


def test_create_simple_eval_set():
    eval_set = create_simple_eval_set("basic", [("Hello", "Hi there", "greeting")])
    assert eval_set.eval_cases[0].name == "greeting"


# ── Part 2: the evaluator against a scripted agent (offline) ────────────────


def lookup(query: str) -> str:
    """Look up the weather. The scripted agent always calls this with 'test query'."""
    return f"The weather in {query} is sunny"


def _route(state) -> str:
    last = state.context[-1]
    return "TOOL" if getattr(last, "tools_calls", None) and last.role == "assistant" else END


@pytest.fixture
def scripted_app():
    graph = StateGraph()
    graph.add_node("MAIN", ScriptedAgent(responses=["The weather in NYC is sunny"], tools=[lookup]))
    graph.add_node("TOOL", ToolNode([lookup]))
    graph.add_conditional_edges("MAIN", _route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    return create_eval_app(graph)  # compiles with a TrajectoryCollector attached


async def test_non_llm_criteria_on_scripted_agent(scripted_app):
    app, collector = scripted_app
    case = EvalCase.single_turn(
        eval_id="scripted",
        user_query="Weather in NYC?",
        expected_response="The weather in NYC is sunny",
        expected_tools=[ToolCall(name="lookup")],
        # The collector records AI-node visits only (the tool node is not listed).
        expected_node_order=["__start__", "MAIN", "MAIN"],
    )
    config = EvalConfig(
        criteria=CriteriaConfig(
            tool_name_match=CriterionConfig.tool_name_match(threshold=1.0),
            trajectory=CriterionConfig.trajectory(threshold=1.0, match_type=MatchType.IN_ORDER),
            node_order=CriterionConfig.node_order(threshold=1.0),
            rouge_match=CriterionConfig.rouge_match(threshold=0.8),
            contains_keywords=CriterionConfig.contains_keywords(["sunny"]),
        ),
        reporter=QUIET,
    )

    result = await AgentEvaluator(app, collector, config=config).evaluate_case(case)

    assert result.passed, [c.criterion for c in result.failed_criteria]
    assert [t.name for t in result.actual_tool_calls] == ["lookup"]


async def test_wrong_tool_is_reported_as_failure(scripted_app):
    app, collector = scripted_app
    case = EvalCase.single_turn(
        eval_id="wrong_tool",
        user_query="Weather in NYC?",
        expected_response="sunny",
        expected_tools=[ToolCall(name="get_forecast")],
    )
    config = EvalConfig(
        criteria=CriteriaConfig(tool_name_match=CriterionConfig.tool_name_match()),
        reporter=QUIET,
    )

    result = await AgentEvaluator(app, collector, config=config).evaluate_case(case)

    assert not result.passed
    assert result.get_criterion_result("tool_name_match_score").score == 0.0


# ── Part 3: the real agent (needs OPENAI_API_KEY) ───────────────────────────


@pytest.mark.needs_openai
async def test_tool_trajectory_for_weather_agent(app, collector):
    config = EvalConfig(
        criteria=CriteriaConfig(
            tool_name_match=CriterionConfig.tool_name_match(threshold=1.0),
            # check_args=True also compares arguments, so location must match.
            trajectory=CriterionConfig.trajectory(
                threshold=1.0, match_type=MatchType.IN_ORDER, check_args=False
            ),
        ),
        reporter=QUIET,
    )

    report = await AgentEvaluator(app, collector, config=config).evaluate(weather_eval_set())

    assert_eval_passed(report, min_pass_rate=1.0)
    assert_criterion_passed(report, "tool_name_match_score", min_score=1.0)


@pytest.mark.needs_openai
async def test_response_match_with_rouge_and_keywords(app, collector):
    # Both criteria are token based, so they need no judge model.
    config = EvalConfig(
        criteria=CriteriaConfig(
            rouge_match=CriterionConfig.rouge_match(threshold=0.3),
            contains_keywords=CriterionConfig.contains_keywords(["sunny"], threshold=1.0),
        ),
        reporter=QUIET,
    )
    case = EvalCase.single_turn(
        eval_id="nyc_weather_text",
        user_query="What is the weather in New York City right now?",
        expected_response="The weather in New York City is sunny",
    )

    result = await AgentEvaluator(app, collector, config=config).evaluate_case(case)

    assert result.passed, [(c.criterion, c.score) for c in result.failed_criteria]


@pytest.mark.needs_openai
async def test_multi_turn_conversation(app, collector):
    case = EvalCase.multi_turn(
        eval_id="mt_followup",
        conversation=[
            ("What is the weather in London?", "The weather in London is sunny"),
            ("Now a 3-day forecast for London.", "forecast for London"),
        ],
    )
    config = EvalConfig(
        criteria=CriteriaConfig(rouge_match=CriterionConfig.rouge_match(threshold=0.2)),
        reporter=QUIET,
    )

    result = await AgentEvaluator(app, collector, config=config).evaluate_case(case)

    assert result.passed, [(c.criterion, c.score) for c in result.failed_criteria]


# ── Pytest helpers: eval files, per-case parametrisation ────────────────────


@pytest.mark.needs_openai
async def test_run_eval_from_file(app, collector):
    config = EvalConfig(
        criteria=CriteriaConfig(tool_name_match=CriterionConfig.tool_name_match()),
        reporter=QUIET,
    )

    report = await run_eval(app, collector, EVAL_FILE, config=config)

    assert_eval_passed(report)


@pytest.mark.needs_openai
@parametrize_eval_cases(EVAL_FILE)  # one pytest item per case in the file
async def test_each_case(app, collector, eval_case):
    config = EvalConfig(
        criteria=CriteriaConfig(tool_name_match=CriterionConfig.tool_name_match()),
        reporter=QUIET,
    )

    result = await AgentEvaluator(app, collector, config=config).evaluate_case(eval_case)

    assert result.passed, [(c.criterion, c.score) for c in result.failed_criteria]
