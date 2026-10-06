"""
EvalCase definitions for test5 — AgentEvaluator API & Config tests.

Covers both single-turn cases (for evaluate_case / evaluate) and
multi-turn conversation flows.
"""

from tenxgraph.qa.evaluation.dataset import EvalCase, ToolCall
from tenxgraph.qa.evaluation.dataset.eval_set import EvalSet


# ── Single-turn cases (used by evaluate_case / evaluate) ────────────
WEATHER_CASE = EvalCase.single_turn(
    eval_id="deep_weather",
    name="Weather evaluation case",
    description="Simple weather query — agent must call get_weather",
    user_query="What is the weather like in New York City?",
    expected_response="The weather in New York City is sunny",
    expected_tools=[ToolCall(name="get_weather")],
)

FORECAST_CASE = EvalCase.single_turn(
    eval_id="deep_forecast",
    name="Forecast evaluation case",
    description="Forecast query — agent must call get_forecast",
    user_query="Give me a 5-day forecast for London.",
    expected_response="forecast for London",
    expected_tools=[ToolCall(name="get_forecast")],
)

NO_TOOL_CASE = EvalCase.single_turn(
    eval_id="deep_no_tool",
    name="No-tool evaluation case",
    description="General knowledge — no tool expected",
    user_query="What is the capital of Germany?",
    expected_response="Berlin",
    expected_tools=[],
)

SMALL_EVAL_SET = EvalSet(
    eval_set_id="deep_eval_set",
    name="Deep Evaluation Set",
    eval_cases=[WEATHER_CASE, NO_TOOL_CASE, FORECAST_CASE],
)

# ── Multi-turn conversation flows ──────────────────────────────────
WEATHER_FOLLOWUP = EvalCase.multi_turn(
    eval_id="mt_weather_followup",
    name="Weather follow-up conversation",
    description="User asks current weather, then asks for a forecast",
    conversation=[
        ("What is the weather in London?", "The weather in London is sunny"),
        ("Can you give me a 3-day forecast for London?", "forecast for London"),
    ],
    expected_tools=[ToolCall(name="get_weather")],
)

CITY_COMPARISON = EvalCase.multi_turn(
    eval_id="mt_city_comparison",
    name="City weather comparison",
    description="User compares weather between two cities",
    conversation=[
        ("What's the weather like in Tokyo?", "The weather in Tokyo is sunny"),
        ("And what about Paris?", "The weather in Paris is sunny"),
        ("Which city has better weather?", "sunny"),
    ],
    expected_tools=[ToolCall(name="get_weather")],
)

GENERAL_TO_SPECIFIC = EvalCase.multi_turn(
    eval_id="mt_general_to_specific",
    name="General knowledge then weather",
    description="User starts with general question, then asks weather",
    conversation=[
        ("What is the capital of Italy?", "Rome"),
        ("What's the weather like in Rome?", "sunny"),
    ],
    expected_tools=[],
)

ALL_MULTI_TURN = [WEATHER_FOLLOWUP, CITY_COMPARISON, GENERAL_TO_SPECIFIC]
