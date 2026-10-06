"""Skills Example — simple graph with Gemini 2.5 Flash + dynamic skills.

This example shows how to use the Agentflow Skills system so a single
agent can load specialised skill modes (code-review, data-analysis,
writing-assistant) on-demand at runtime.

How it works
------------
1. Three ``SKILL.md`` files live in ``./skills/``.
2. ``Agent(skills=SkillConfig(skills_dir=...))`` auto-discovers them,
   appends an ``<available_skills>`` catalog (name + description) to the
   system prompt, and registers an ``activate_skill`` tool the LLM can call.
   Skills that bundle files (``references/``, ``scripts/``, ...) also get a
   ``read_skill_resource`` tool.
3. When the user's message matches a skill's description, the LLM calls
   ``activate_skill("<name>")``.  The tool returns the SKILL.md instructions
   wrapped in ``<skill_content>`` tags, which the LLM uses to respond.
4. Each turn, the LLM can decide which skill (if any) to load based on
   the user's request.

Run
---
    python graph.py

    # or try a specific query:
    python graph.py "Review this Python code: def add(a,b): return a+b"
    python graph.py "Help me write a professional apology email to a client"
    python graph.py "Analyse this data: sales=[120,95,140,88,160] by month"
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.skills import SkillConfig
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_context_manager import MessageContextManager
from tenxgraph.utils.constants import END


load_dotenv()


# ---------------------------------------------------------------------------
# Custom tool — get_weather (in addition to skills)
# ---------------------------------------------------------------------------
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location to get weather for

    Returns:
        A string describing the current weather
    """
    # Mock weather data
    weather_data = {
        "london": "Cloudy, 15°C",
        "new york": "Sunny, 22°C",
        "tokyo": "Rainy, 18°C",
        "paris": "Partly cloudy, 17°C",
    }

    location_lower = location.lower()
    if location_lower in weather_data:
        return f"The weather in {location} is: {weather_data[location_lower]}"
    else:
        return f"Weather data not available for {location}. Try London, New York, Tokyo, or Paris."


# ---------------------------------------------------------------------------
# Skills directory — three SKILL.md files sit alongside this script
# ---------------------------------------------------------------------------
SKILLS_DIR = str(Path(__file__).parent / "skills")

# ---------------------------------------------------------------------------
# Tool node — registered in the graph as "TOOL".
# At compile time the graph wires this into the agent automatically.
# ---------------------------------------------------------------------------
tool_node = ToolNode([get_weather])

# ---------------------------------------------------------------------------
# Agent — Gemini 2.5 Flash with skills + context trimming enabled + custom tools
# ---------------------------------------------------------------------------
agent = Agent(
    model="google/gemini-2.5-flash",
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You are a smart, multi-skilled assistant.\n"
                "You have access to specialised skill modes that give you "
                "deeper expertise in specific domains.\n"
                "You also have access to a get_weather tool for weather queries."
            ),
        }
    ],
    tool_node="TOOL",  # resolved to the ToolNode above at graph.compile() time
    skills=SkillConfig(
        skills_dir=SKILLS_DIR,
        inject_catalog=True,  # appends the <available_skills> catalog to the system prompt
        hot_reload=True,  # re-reads a SKILL.md when it changes on disk (great for dev)
    ),
    trim_context=True,
)

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_use_tools(state: AgentState) -> str:
    """Route MAIN → TOOL if there are tool calls, else → END."""
    if not state.context:
        return END

    last = state.context[-1]

    if last.role == "assistant" and hasattr(last, "tools_calls") and last.tools_calls:
        return "TOOL"

    if last.role == "tool":
        return "MAIN"  # got tool results → back to LLM for final answer

    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph = StateGraph(
    context_manager=MessageContextManager(max_messages=20),
)
graph.add_node("MAIN", agent)
graph.add_node("TOOL", tool_node)

graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile()

# Print available tools after compile — get_weather + activate_skill are registered
print("\n📋 Available tools in ToolNode (resolved at compile time):")
for tool_name in tool_node._funcs:
    print(f"  - {tool_name}")
print()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Accept an optional query from the command line
    user_input = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else (
            "Can you review this Python function for me?\n\n"
            "```python\n"
            "def calculate_average(numbers):\n"
            "    total = 0\n"
            "    for n in numbers:\n"
            "        total = total + n\n"
            "    return total / len(numbers)\n"
            "```"
        )
    )

    print("\n" + "=" * 60)
    print("USER:", user_input)
    print("=" * 60 + "\n")

    inp = {"messages": [Message.text_message(user_input)]}
    config = {"thread_id": "skills-demo-1", "recursion_limit": 15}

    res = app.invoke(inp, config=config)

    for msg in res["messages"]:
        if msg.role == "assistant":
            print("-" * 60)
            print(f"ASSISTANT ({msg.role}):")
            print(msg.text() or "(no text)")
            print("-" * 60)
