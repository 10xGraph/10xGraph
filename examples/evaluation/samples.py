"""
Shared EvalCase definitions used by both test1 and test2.

Import from here in any evaluation test suite so cases are defined once.

Sections:
  1. Happy-path cases          – basic tool-presence checks
  2. Multi-city / no-tool      – edge cases for tool presence
  3. Expected trajectory cases – multi-tool order, args, duplicates, empty
"""

from tenxgraph.qa.evaluation.dataset import EvalCase, ToolCall
from tenxgraph.qa.evaluation.dataset.eval_set import EvalSet, Invocation


# ═══════════════════════════════════════════════════════════════════════
# 1. Happy-path: agent should call get_weather and return a sunny response
# ═══════════════════════════════════════════════════════════════════════

NYC = EvalCase.single_turn(
    eval_id="nyc_happy",
    name="NYC — happy path",
    user_query="Please call the get_weather function for New York City",
    expected_response="The weather in New York City is sunny",
    expected_tools=[ToolCall(name="get_weather")],
)

LONDON = EvalCase.single_turn(
    eval_id="london_happy",
    name="London — happy path",
    user_query="What is the weather like in London?",
    expected_response="The weather in London is sunny",
    expected_tools=[ToolCall(name="get_weather")],
)

TOKYO = EvalCase.single_turn(
    eval_id="tokyo_happy",
    name="Tokyo — happy path",
    user_query="Tell me the current weather in Tokyo",
    expected_response="The weather in Tokyo is sunny",
    expected_tools=[ToolCall(name="get_weather")],
)

# ═══════════════════════════════════════════════════════════════════════
# 2. Multi-city / no-tool
# ═══════════════════════════════════════════════════════════════════════

MULTI_CITY = EvalCase.single_turn(
    eval_id="multi_city",
    name="Multi-city — tool called",
    user_query="Get weather for both Paris and Berlin",
    expected_response="",  # no strict output — just check tool was called
    expected_tools=[ToolCall(name="get_weather")],
)

CAPITAL_QUESTION = EvalCase.single_turn(
    eval_id="capital_no_tool",
    name="General knowledge — no tool",
    user_query="What is the capital of France?",
    expected_response="Paris",
    expected_tools=[],  # empty → assert zero tool calls
)

# ---------------------------------------------------------------------------
# Grouped for parametrize and bulk evaluation
# ---------------------------------------------------------------------------

HAPPY_PATH_CASES = [NYC, LONDON, TOKYO]
MULTI_CITY_CASES = [MULTI_CITY]
NO_TOOL_CASES = [CAPITAL_QUESTION]
ALL_CASES = HAPPY_PATH_CASES + MULTI_CITY_CASES + NO_TOOL_CASES

# ---------------------------------------------------------------------------
# EvalSet — ready for AgentEvaluator.evaluate(EVAL_SET)
# ---------------------------------------------------------------------------

EVAL_SET = EvalSet(
    eval_set_id="weather_agent_eval",
    name="Weather Agent Evaluation Suite",
    eval_cases=ALL_CASES,
)


# ═══════════════════════════════════════════════════════════════════════
# 3. Expected Trajectory Samples
#
#    These cases are designed for testing trajectory matching criteria
#    (TrajectoryMatchCriterion / ToolNameMatchCriterion) against expected
#    tool call sequences.  They exercise:
#      • EXACT match  – identical tools in identical order
#      • IN_ORDER     – expected tools appear in order, extras allowed
#      • ANY_ORDER    – expected tools present regardless of order
#      • check_args   – argument-level matching
#      • duplicates   – same tool called multiple times
#      • empty        – no tools expected at all
# ═══════════════════════════════════════════════════════════════════════

# ── 3a. Multi-tool exact sequence (EXACT match) ────────────────────
#    Tests: agent must call get_weather THEN get_forecast in that order.

WEATHER_THEN_FORECAST = EvalCase.single_turn(
    eval_id="traj_weather_then_forecast",
    name="Trajectory — weather then forecast (exact)",
    description="Agent must call get_weather followed by get_forecast, in that order.",
    user_query=(
        "First tell me the current weather in New York City, "
        "then give me the 5-day forecast for the same city."
    ),
    expected_response="sunny",
    expected_tools=[
        ToolCall(name="get_weather"),
        ToolCall(name="get_forecast"),
    ],
)

# ── 3b. Reversed order (EXACT match should fail, ANY_ORDER pass) ───
#    Tests: same two tools but reversed — validates order sensitivity.

FORECAST_THEN_WEATHER = EvalCase.single_turn(
    eval_id="traj_forecast_then_weather",
    name="Trajectory — forecast then weather (reversed)",
    description="Agent must call get_forecast first, then get_weather.",
    user_query=(
        "Give me the forecast for London first, " "and then tell me the current weather there."
    ),
    expected_response="sunny",
    expected_tools=[
        ToolCall(name="get_forecast"),
        ToolCall(name="get_weather"),
    ],
)

# ── 3c. Multi-tool with specific args (check_args=True) ────────────
#    Tests: tool name AND argument values must match exactly.

WEATHER_NYC_WITH_ARGS = EvalCase.single_turn(
    eval_id="traj_weather_nyc_args",
    name="Trajectory — NYC weather with args",
    description="Agent must call get_weather with location='New York City'.",
    user_query="What is the current weather in New York City?",
    expected_response="sunny",
    expected_tools=[
        ToolCall(name="get_weather", args={"location": "New York City"}),
    ],
)

WEATHER_LONDON_WITH_ARGS = EvalCase.single_turn(
    eval_id="traj_weather_london_args",
    name="Trajectory — London weather with args",
    description="Agent must call get_weather with location='London'.",
    user_query="What is the weather like in London right now?",
    expected_response="sunny",
    expected_tools=[
        ToolCall(name="get_weather", args={"location": "London"}),
    ],
)

FORECAST_TOKYO_WITH_ARGS = EvalCase.single_turn(
    eval_id="traj_forecast_tokyo_args",
    name="Trajectory — Tokyo forecast with args",
    description="Agent must call get_forecast with location='Tokyo' and days=3.",
    user_query="Give me a 3-day forecast for Tokyo.",
    expected_response="forecast",
    expected_tools=[
        ToolCall(name="get_forecast", args={"location": "Tokyo", "days": 3}),
    ],
)

# ── 3d. Duplicate tool calls (same tool called multiple times) ─────
#    Tests: agent calls get_weather twice for two different cities.

TWO_CITIES_TWO_CALLS = EvalCase.single_turn(
    eval_id="traj_two_cities_two_calls",
    name="Trajectory — two get_weather calls",
    description="Agent must call get_weather twice — once per city.",
    user_query="Tell me the weather in both Paris and Berlin.",
    expected_response="sunny",
    expected_tools=[
        ToolCall(name="get_weather"),
        ToolCall(name="get_weather"),
    ],
)

THREE_CITIES_THREE_CALLS = EvalCase.single_turn(
    eval_id="traj_three_cities_three_calls",
    name="Trajectory — three get_weather calls",
    description="Agent must call get_weather three times — once per city.",
    user_query="Compare the weather in Tokyo, London, and New York City.",
    expected_response="sunny",
    expected_tools=[
        ToolCall(name="get_weather"),
        ToolCall(name="get_weather"),
        ToolCall(name="get_weather"),
    ],
)

# ── 3e. Single-tool strict (only one tool, nothing else) ──────────
#    Tests: EXACT match verifies exactly one tool call, no extras.

WEATHER_ONLY_ONE_TOOL = EvalCase.single_turn(
    eval_id="traj_single_tool_strict",
    name="Trajectory — strictly one get_weather call",
    description="Agent must call get_weather exactly once and no other tool.",
    user_query="What is the weather in Sydney right now?",
    expected_response="sunny",
    expected_tools=[ToolCall(name="get_weather")],
)

# ── 3f. No tools expected (empty trajectory) ──────────────────────
#    Tests: agent should NOT call any tool — trajectory must be empty.

NO_TOOLS_EXPECTED = EvalCase.single_turn(
    eval_id="traj_no_tools_expected",
    name="Trajectory — no tools expected",
    description="General knowledge question — agent must not invoke any tool.",
    user_query="What is the largest planet in our solar system?",
    expected_response="Jupiter",
    expected_tools=[],
)

# ── 3g. Multi-tool mixed sequence (IN_ORDER) ─────────────────────
#    Tests: weather, weather, forecast — checks in-order with repeats.

MIXED_TOOL_SEQUENCE = EvalCase.single_turn(
    eval_id="traj_mixed_sequence",
    name="Trajectory — weather+weather+forecast sequence",
    description=(
        "Agent should call get_weather for two cities then get_forecast "
        "for one of them. IN_ORDER match validates sequence."
    ),
    user_query=(
        "Tell me the current weather in Paris and Rome, " "then give me a 3-day forecast for Paris."
    ),
    expected_response="sunny",
    expected_tools=[
        ToolCall(name="get_weather"),
        ToolCall(name="get_weather"),
        ToolCall(name="get_forecast"),
    ],
)

# ── 3h. Multi-turn trajectory ────────────────────────────────────
#    Tests: expected tools spread across multiple conversation turns.

MULTI_TURN_TRAJECTORY = EvalCase(
    eval_id="traj_multi_turn",
    name="Trajectory — multi-turn weather then forecast",
    description=(
        "Turn 1: user asks current weather (expect get_weather). "
        "Turn 2: user asks for forecast (expect get_forecast). "
        "Trajectory criteria should collect tools from ALL turns."
    ),
    conversation=[
        Invocation.simple(
            user_query="What is the weather in London?",
            expected_response="sunny",
            expected_tools=[ToolCall(name="get_weather")],
        ),
        Invocation.simple(
            user_query="Now give me the 5-day forecast for London.",
            expected_response="forecast",
            expected_tools=[ToolCall(name="get_forecast")],
        ),
    ],
)

MULTI_TURN_NO_TOOLS = EvalCase(
    eval_id="traj_multi_turn_no_tools",
    name="Trajectory — multi-turn no tools expected",
    description=(
        "Two general knowledge turns — neither should trigger any tool. "
        "Expected trajectory is empty across all turns."
    ),
    conversation=[
        Invocation.simple(
            user_query="What is the capital of Japan?",
            expected_response="Tokyo",
            expected_tools=[],
        ),
        Invocation.simple(
            user_query="And what about Germany?",
            expected_response="Berlin",
            expected_tools=[],
        ),
    ],
)

MULTI_TURN_PARTIAL_TOOLS = EvalCase(
    eval_id="traj_multi_turn_partial",
    name="Trajectory — multi-turn partial tool use",
    description=(
        "Turn 1: general knowledge (no tool). "
        "Turn 2: weather query (expect get_weather). "
        "Tests that only specific turns contribute to expected trajectory."
    ),
    conversation=[
        Invocation.simple(
            user_query="What is the capital of France?",
            expected_response="Paris",
            expected_tools=[],
        ),
        Invocation.simple(
            user_query="What is the weather in Paris right now?",
            expected_response="sunny",
            expected_tools=[ToolCall(name="get_weather")],
        ),
    ],
)

# ═══════════════════════════════════════════════════════════════════════
# Trajectory sample groups
# ═══════════════════════════════════════════════════════════════════════

EXACT_TRAJECTORY_CASES = [WEATHER_THEN_FORECAST, FORECAST_THEN_WEATHER]
ARGS_TRAJECTORY_CASES = [
    WEATHER_NYC_WITH_ARGS,
    WEATHER_LONDON_WITH_ARGS,
    FORECAST_TOKYO_WITH_ARGS,
]
MULTI_CALL_CASES = [TWO_CITIES_TWO_CALLS, THREE_CITIES_THREE_CALLS]
MULTI_TURN_TRAJECTORY_CASES = [
    MULTI_TURN_TRAJECTORY,
    MULTI_TURN_NO_TOOLS,
    MULTI_TURN_PARTIAL_TOOLS,
]

ALL_TRAJECTORY_CASES = (
    EXACT_TRAJECTORY_CASES
    + ARGS_TRAJECTORY_CASES
    + MULTI_CALL_CASES
    + [WEATHER_ONLY_ONE_TOOL, NO_TOOLS_EXPECTED, MIXED_TOOL_SEQUENCE]
    + MULTI_TURN_TRAJECTORY_CASES
)

# ---------------------------------------------------------------------------
# Trajectory EvalSet — for bulk trajectory evaluation
# ---------------------------------------------------------------------------

TRAJECTORY_EVAL_SET = EvalSet(
    eval_set_id="trajectory_eval",
    name="Expected Trajectory Evaluation Suite",
    eval_cases=ALL_TRAJECTORY_CASES,
)
