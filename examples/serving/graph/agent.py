"""The graph the 10xGraph API server serves.

`10xgraph.json` points at `app` below (`"agent": "graph.agent:app"`). The server
imports this module once at startup and exposes the compiled graph over REST and SSE.
Check it locally without a server: python graph/agent.py   (from examples/serving)
"""

import asyncio
from datetime import date

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny, 24 C"


def get_time(city: str) -> str:
    """Get the local time for a city."""
    return f"It is 14:30 in {city}"


agent = ReactAgent(
    model="claude-sonnet-5-5",
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You are a concise travel assistant. Use tools for weather and local time. "
                f"Today is {date.today().isoformat()}."
            ),
        }
    ],
    tools=[get_weather, get_time],
)

# The server needs a checkpointer to keep threads between requests. The in-memory one
# loses them on restart; use PgCheckpointer (pg_checkpoint extra) for real deployments.
app = agent.compile(checkpointer=InMemoryCheckpointer())


async def main() -> None:
    result = await app.ainvoke(
        {"messages": [Message.text_message("Weather in Lisbon?")]},
        config={"thread_id": "local-check"},
    )
    print("assistant:", result["messages"][-1].text())


if __name__ == "__main__":
    asyncio.run(main())
