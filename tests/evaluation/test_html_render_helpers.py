from tenxgraph.qa.evaluation.dataset.eval_set import TrajectoryStep
from tenxgraph.qa.evaluation.eval_result import CriterionResult, EvalCaseResult
from tenxgraph.qa.evaluation.reporters import _html_render as hr
from tenxgraph.qa.evaluation.token_usage import TokenUsage


def _make_result() -> EvalCaseResult:
    return EvalCaseResult.success(
        eval_id="c1",
        name="Case One",
        criterion_results=[
            CriterionResult.success(
                criterion="response_match_score",
                score=0.75,
                threshold=0.7,
                details={"reason": "close enough", "extra": 1},
            )
        ],
        actual_response="final answer",
        actual_tool_calls=[{"name": "search", "args": {"q": "x"}, "result": "ok", "call_id": "t1"}],
        actual_trajectory=[
            {"step_type": "node", "name": "router", "timestamp": 1.0, "metadata": {"a": 1}},
            TrajectoryStep.tool("search", args={"q": "x"}, timestamp=2.0),
        ],
        messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        node_responses=[
            {
                "node_name": "MAIN",
                "response_text": "thinking",
                "has_tool_calls": True,
                "tool_call_names": ["search"],
                "is_final": False,
                "timestamp": 123,
                "input_messages": [{"role": "user", "content": "hello"}],
            }
        ],
        node_visits=["router", "MAIN"],
        turn_results=[
            {
                "turn_index": 0,
                "user_input": "hello",
                "agent_response": "hi",
                "tool_calls": [{"name": "search"}],
                "node_visits": ["router", "MAIN"],
            }
        ],
        metadata={"suite": "smoke"},
    )


def test_render_criterion_includes_reason_and_details():
    cr = CriterionResult.success(
        criterion="test",
        score=0.5,
        threshold=0.7,
        details={"reason": "failed", "debug": "x"},
    )
    html = hr.render_criterion(cr)
    assert "criterion-reason" in html
    assert "debug" in html


def test_render_case_renders_all_sections():
    result = _make_result()
    html = hr.render_case(result)
    assert "Agent Response" in html
    assert "Tool Calls" in html
    assert "Execution Trajectory" in html
    assert "Node Responses" in html
    assert "Node Visits" in html
    assert "Turn-by-Turn Results" in html
    assert "Criteria" in html


def test_render_trajectory_and_node_response_helpers_handle_mixed_inputs():
    steps = hr._render_trajectory_steps([
        {"step_type": "tool", "name": "search", "args": {"q": "x"}},
        TrajectoryStep.node("END", timestamp=3.0),
        "raw_step",
    ])
    assert len(steps) == 3

    boxes = hr._render_node_responses([
        {"node_name": "MAIN", "response_text": "ok", "has_tool_calls": False, "tool_call_names": []},
        "raw",
    ])
    assert len(boxes) == 2


def test_render_messages_and_turn_results_helpers():
    msgs = hr._render_messages([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])
    assert len(msgs) == 2

    turns = hr._render_turn_results([
        {"turn_index": 1, "user_input": "u", "agent_response": "a", "tool_calls": [], "node_visits": []}
    ])
    assert len(turns) == 1


def test_score_color_clamps_outside_range():
    assert hr._score_color(-1.0).startswith("hsl(0")
    assert hr._score_color(2.0).startswith("hsl(120")


def test_token_panels_and_node_details_rendering():
    tok = TokenUsage(input_tokens=10, output_tokens=5, cache_read_tokens=2)
    panel = hr._render_token_panel(tok)
    assert "Tokens" in panel

    node_detail = type(
        "ND",
        (),
        {
            "node_name": "MAIN",
            "response_text": "hello",
            "token_usage": tok,
            "input_messages": [{"role": "user", "content": "hi"}],
            "tool_call_inputs": [{"name": "search", "args": {"q": "x"}}],
            "tool_call_outputs": [{"ok": True}],
        },
    )()
    html = hr._render_node_details([node_detail])
    assert "Node Details" not in html  # helper returns inner <details> items
    assert "search" in html


def test_render_case_with_error_and_token_breakdown():
    result = _make_result()
    result.error = "boom"
    result.token_usage = TokenUsage(input_tokens=2, output_tokens=1)
    result.agent_token_usage = TokenUsage(input_tokens=1, output_tokens=1)
    result.criterion_results[0].token_usage = TokenUsage(input_tokens=1, output_tokens=0)

    html = hr.render_case(result)
    assert "error-message" in html
    assert "Total" in html
