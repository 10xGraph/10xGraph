"""The agent under test: a small weather assistant on OpenAI gpt-5-mini.

Every evaluation file in this folder imports `build_eval_app()` from here, so the
agent is defined once. The graph is compiled with a TrajectoryCollector so the
evaluator can see tool calls and node visits.
Run: python examples/evaluation/weather_agent.py   (needs OPENAI_API_KEY)
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.qa.evaluation import TrajectoryCollector, make_trajectory_callback
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny"


def get_forecast(location: str, days: int = 3) -> str:
    """Get a multi-day weather forecast for a location."""
    return f"{days}-day forecast for {location}: sunny, cloudy, sunny"


SYSTEM_PROMPT = (
    "You are a weather assistant. Use get_weather for current conditions and "
    "get_forecast for multi-day forecasts. Answer anything else (geography, "
    "general knowledge) directly, without calling a tool."
)


def build_agent() -> ReactAgent:
    # Constructing the agent creates the OpenAI client, so OPENAI_API_KEY must be set.
    return ReactAgent(
        model="gpt-5-mini",
        system_prompt=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=[get_weather, get_forecast],
    )


def build_eval_app():
    """Compile the agent with a collector and return `(compiled_graph, collector)`."""
    collector = TrajectoryCollector(capture_all_events=True)
    _, callback_manager = make_trajectory_callback(collector)
    app = build_agent().compile(
        checkpointer=InMemoryCheckpointer(),
        callback_manager=callback_manager,
    )
    return app, collector


async def main() -> None:
    load_dotenv()
    app, _ = build_eval_app()
    result = await app.ainvoke(
        {"messages": [Message.text_message("What is the weather in Tokyo?")]},
        config={"thread_id": "demo"},
    )
    print("assistant:", result["messages"][-1].text())


if __name__ == "__main__":
    asyncio.run(main())
