"""Case results carry what the case expected, and simulations carry a duration."""

from tenxgraph.qa.evaluation import EvalCase, ToolCall
from tenxgraph.qa.evaluation.dataset.eval_set import Invocation
from tenxgraph.qa.evaluation.eval_result import EvalCaseResult, EvalReport
from tenxgraph.qa.evaluation.reporters.json import JSONReporter
from tenxgraph.qa.evaluation.simulators.user_simulator import SimulationResult


def test_single_turn_expected_summary():
    case = EvalCase.single_turn(
        eval_id="c1",
        user_query="Weather in London?",
        expected_response="The weather in London is sunny.",
        expected_tools=[ToolCall(name="get_weather", args={"location": "London"})],
        expected_node_order=["MAIN"],
    )
    assert case.expected_summary() == {
        "response": "The weather in London is sunny.",
        "tool_calls": [{"name": "get_weather", "args": {"location": "London"}}],
        "node_order": ["MAIN"],
        "input": "Weather in London?",
    }


def test_multi_turn_expected_summary_uses_last_response_and_all_tools():
    case = EvalCase(
        eval_id="m",
        conversation=[
            Invocation.simple("first", "r1", [ToolCall(name="a")], ["N1"]),
            Invocation.simple("second", "r2", [ToolCall(name="b", args={"x": 1})], ["N2"]),
        ],
    )
    summary = case.expected_summary()
    assert summary["input"] == "first"
    assert summary["response"] == "r2"
    assert [t["name"] for t in summary["tool_calls"]] == ["a", "b"]
    assert summary["node_order"] == ["N1", "N2"]


def test_missing_expectations_are_none():
    case = EvalCase.single_turn(eval_id="c", user_query="hi")
    summary = case.expected_summary()
    assert summary["response"] is None
    assert summary["tool_calls"] is None
    assert summary["node_order"] is None


def test_expected_is_serialized_in_json_report_and_optional_for_old_reports():
    expected = {"response": "r", "tool_calls": None, "node_order": None, "input": "q"}
    ok = EvalCaseResult.success(eval_id="a", criterion_results=[], expected=expected)
    bad = EvalCaseResult.failure(eval_id="b", error="boom", expected=expected)
    report = EvalReport.create(eval_set_id="s", results=[ok, bad])
    data = JSONReporter().to_dict(report)
    assert [r["expected"] for r in data["results"]] == [expected, expected]

    old = EvalCaseResult.model_validate({"eval_id": "old", "passed": True})
    assert old.expected is None


def test_summary_durations_reflect_case_durations():
    results = [
        EvalCaseResult.failure(eval_id="a", error="x", duration_seconds=2.0),
        EvalCaseResult.failure(eval_id="b", error="x", duration_seconds=4.0),
    ]
    summary = EvalReport.create(eval_set_id="s", results=results).summary
    assert summary.total_duration_seconds == 6.0
    assert summary.avg_duration_seconds == 3.0


def test_simulation_result_has_duration_field():
    assert SimulationResult().duration_seconds == 0.0
