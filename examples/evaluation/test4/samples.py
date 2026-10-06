"""
EvalCase definitions for test4 — single-turn evaluation.

Five cases covering:
  1. Happy-path weather query (expects get_weather tool)
  2. Forecast query (expects get_forecast tool)
  3. Multi-city query (expects get_weather for multiple cities)
  4. General knowledge — no tool expected
  5. Ambiguous query — agent may or may not call a tool
"""

from tenxgraph.qa.evaluation.dataset import EvalCase, ToolCall


# ── 1. Happy-path: current weather ──────────────────────────────────
WEATHER_NYC = EvalCase.single_turn(
    eval_id="st_weather_nyc",
    name="NYC current weather",
    description="Simple weather query — agent must call get_weather",
    user_query="What is the weather like in New York City right now?",
    expected_response="The weather in New York City is sunny",
    expected_tools=[ToolCall(name="get_weather")],
)

# ── 2. Forecast query ───────────────────────────────────────────────
FORECAST_LONDON = EvalCase.single_turn(
    eval_id="st_forecast_london",
    name="London 5-day forecast",
    description="Forecast query — agent must call get_forecast",
    user_query="Give me a 5-day forecast for London please.",
    expected_response="5-day forecast for London: sunny, cloudy, sunny",
    expected_tools=[ToolCall(name="get_forecast")],
)

# ── 3. Multi-city weather ───────────────────────────────────────────
MULTI_CITY = EvalCase.single_turn(
    eval_id="st_multi_city",
    name="Multi-city weather comparison",
    description="User asks about two cities — agent should call get_weather at least once",
    user_query="Compare the current weather in Tokyo and Paris.",
    expected_response="The weather in Tokyo and Paris is sunny",
    expected_tools=[ToolCall(name="get_weather")],
)

# ── 4. General knowledge — no tool ──────────────────────────────────
CAPITAL_QUESTION = EvalCase.single_turn(
    eval_id="st_capital",
    name="General knowledge (no tool)",
    description="Non-weather question — agent must NOT use any tool",
    user_query="What is the capital of Germany?",
    expected_response="The capital of Germany is Berlin",
    expected_tools=[],
)

# ── 5. Ambiguous — tool may be called ───────────────────────────────
AMBIGUOUS = EvalCase.single_turn(
    eval_id="st_ambiguous",
    name="Ambiguous travel question",
    description="Agent might call get_weather or answer directly — test response quality only",
    user_query="I'm thinking about visiting Sydney next month. Any tips?",
    expected_response="Sydney is a great city to visit",
    expected_tools=[],
)

# ── Grouped ─────────────────────────────────────────────────────────
TOOL_CASES = [WEATHER_NYC, FORECAST_LONDON, MULTI_CITY]
NO_TOOL_CASES = [CAPITAL_QUESTION, AMBIGUOUS]
ALL_CASES = TOOL_CASES + NO_TOOL_CASES

# ── Batch eval set (used by reporters & builder tests) ──────────────
from tenxgraph.qa.evaluation.dataset.eval_set import EvalSet


BATCH_EVAL_SET = EvalSet(
    eval_set_id="batch_eval_set",
    name="QuickEval Batch",
    eval_cases=[WEATHER_NYC, CAPITAL_QUESTION],
)
