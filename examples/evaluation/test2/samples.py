"""
EvalCase definitions for test2 — single-turn all-criteria evaluation.
"""

from tenxgraph.qa.evaluation.dataset import EvalCase, ToolCall


# ── Weather query (expects get_weather) ──────────────────────────────
WEATHER_NYC = EvalCase.single_turn(
    eval_id="st_weather_nyc",
    name="NYC current weather",
    user_query="What is the weather like in New York City right now?",
    expected_response="The weather in New York City is sunny",
    expected_tools=[ToolCall(name="get_weather")],
)

# ── General knowledge — no tool ──────────────────────────────────────
CAPITAL_QUESTION = EvalCase.single_turn(
    eval_id="st_capital",
    name="General knowledge (no tool)",
    user_query="What is the capital of Germany?",
    expected_response="Berlin",
    expected_tools=[],
)

# ── Node order case ──────────────────────────────────────────────────
WEATHER_NYC_NODE_ORDER = EvalCase.single_turn(
    eval_id="st_node_order_weather",
    name="NYC weather — node order check",
    user_query="What is the weather like in New York City right now?",
    expected_response="The weather in New York City is sunny",
    expected_tools=[ToolCall(name="get_weather")],
    expected_node_order=["__start__", "MAIN", "MAIN"],
)
