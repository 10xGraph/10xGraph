"""Plan, act, reflect: a planner and a reviewer on different companies' models.

PlanActReflectAgent loops PLAN -> ACT -> REFLECT until the reflector says [DONE] or
`max_iterations` is reached. PLAN is an LLM that decides which tools to call. ACT is the
ToolNode that runs them (no model). REFLECT is an LLM that reviews the work. `plan_model`
and `reflect_model` set a model per phase: Anthropic plans and Gemini reflects.

Run: python examples/multi_agent/plan_act_reflect.py   (needs ANTHROPIC_API_KEY, GEMINI_API_KEY)
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import PlanActReflectAgent


load_dotenv()


def search_flights(origin: str, destination: str) -> str:
    """Search flights between two cities."""
    return f"{origin} -> {destination}: 08:10 at 240 EUR, 13:45 at 190 EUR, 19:30 at 165 EUR."


def search_hotels(city: str, nights: int) -> str:
    """Search hotels in a city."""
    return f"{city}, {nights} nights: Hotel Aurora 110 EUR/night, Casa Verde 85 EUR/night."


agent = PlanActReflectAgent(
    model="claude-sonnet-5-5",  # Fallback for any phase without its own model.
    plan_model="claude-sonnet-5-5",  # Anthropic plans and chooses tools.
    reflect_model="gemini-2.5-flash",  # Google reviews the result.
    tools=[search_flights, search_hotels],
    max_iterations=3,
)


async def main() -> None:
    app = agent.compile()
    result = await app.ainvoke(
        {
            "messages": [
                Message.text_message("Plan a 3-night trip from Berlin to Lisbon under 600 EUR.")
            ]
        },
        config={"thread_id": "par-1", "recursion_limit": 40},
    )
    for msg in result["messages"]:
        if msg.role == "assistant" and msg.text():
            model = (msg.metadata or {}).get("model") or "unknown model"
            print(f"[{model}] {msg.text()[:300]}\n")
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
