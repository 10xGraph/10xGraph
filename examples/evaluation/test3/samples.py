"""
EvalCase and ConversationScenario definitions for test3.

Part A: Multi-turn EvalCases (from old test5)
Part B: Simulator ConversationScenarios (from old test3 + test7)
"""

from tenxgraph.qa.evaluation import ConversationScenario
from tenxgraph.qa.evaluation.dataset import EvalCase, ToolCall


# ═══════════════════════════════════════════════════════════════════════
# Part A: Multi-turn EvalCases
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
# Part B: Simulator ConversationScenarios
# ═══════════════════════════════════════════════════════════════════════

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

GENERAL_KNOWLEDGE_SIM = ConversationScenario(
    scenario_id="general_knowledge",
    description="User asks a general knowledge question — no weather tool needed",
    starting_prompt="What is the capital of France?",
    conversation_plan="1. User asks the question\n2. Agent answers directly",
    goals=["Paris is the capital of France"],
    max_turns=2,
)

TRAVEL_PLANNER = ConversationScenario(
    scenario_id="travel_planner",
    description="User is planning a trip and wants weather info for their destination",
    starting_prompt="I'm planning a vacation to Tokyo next month. Can you help?",
    conversation_plan=(
        "1. User mentions travel interest\n"
        "2. User asks about Tokyo weather\n"
        "3. User asks about what to pack\n"
        "4. User confirms satisfaction"
    ),
    goals=["Get weather information for Tokyo", "Get packing or preparation advice"],
    max_turns=5,
)

QUICK_INFO = ConversationScenario(
    scenario_id="quick_info",
    description="User needs a quick weather check — single question",
    starting_prompt="What's the weather in Berlin right now?",
    conversation_plan="1. User asks directly\n2. Agent answers",
    goals=["Get Berlin weather information"],
    max_turns=2,
)
