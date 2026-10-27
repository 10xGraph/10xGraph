"""ConsolePublisher: watch the events a 10xGraph run emits, in your terminal.

Every graph, node, LLM and tool step publishes an event. ConsolePublisher prints
them, which is the quickest way to see what the engine is doing while you develop.

Run: python examples/observability/console_publisher.py   (needs OPENAI_API_KEY)
"""

import asyncio
import logging

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.runtime.publisher import ConsolePublisher


load_dotenv()


def get_time(city: str) -> str:
    """Return the local time in a city."""
    return f"It is 14:05 in {city}."


async def main() -> None:
    # use_logger=True sends events to the "tenxgraph.publisher" logger instead of stdout,
    # so you can route, filter or silence them with normal logging config.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("tenxgraph.publisher").setLevel(logging.INFO)

    # ReactAgent forwards `publisher` to the StateGraph it builds. With a hand-built graph
    # you would write StateGraph(publisher=ConsolePublisher({"use_logger": True})) instead.
    agent = ReactAgent(
        model="gpt-5-mini",
        tools=[get_time],
        publisher=ConsolePublisher({"use_logger": True}),
    )
    app = agent.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("What time is it in Tokyo?")]},
        config={"thread_id": "console-1"},
    )
    print("assistant:", result["messages"][-1].text())

    # Events are published from background tasks. aclose() waits for them and closes the publisher.
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
