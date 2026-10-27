"""Several publishers at once: print events and also count them with a custom publisher.

Pass a list to `publisher=` and 10xGraph wraps it in a CompositePublisher. Each
publisher gets every event, and one failing publisher does not block the others.
BasePublisher is the extension point for your own sink.

Run: python examples/observability/composite_publisher.py   (needs OPENAI_API_KEY)
"""

import asyncio
import logging
from collections import Counter

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.runtime.publisher import (
    BasePublisher,
    CompositePublisher,
    ConsolePublisher,
    EventModel,
)


load_dotenv()


def get_stock_price(symbol: str) -> str:
    """Return a fake stock price."""
    return f"{symbol} is trading at 187.20 USD."


class EventCounter(BasePublisher):
    """Custom publisher that tallies events by source and phase."""

    def __init__(self) -> None:
        super().__init__({})
        self.counts: Counter[str] = Counter()

    async def publish(self, event: EventModel) -> None:
        self.counts[f"{event.event}/{event.event_type}"] += 1

    async def close(self) -> None:
        self._is_closed = True

    def sync_close(self) -> None:
        self._is_closed = True


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("tenxgraph.publisher").setLevel(logging.INFO)

    counter = EventCounter()
    publishers: list[BasePublisher] = [ConsolePublisher({"use_logger": True}), counter]

    # A plain list works. CompositePublisher(publishers) is the explicit equivalent, and it
    # supports add_publisher() / remove_publisher() if you need to change sinks at runtime.
    agent = ReactAgent(model="gpt-5-mini", tools=[get_stock_price], publisher=publishers)
    app = agent.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("What is ACME trading at?")]},
        config={"thread_id": "composite-1"},
    )
    print("assistant:", result["messages"][-1].text())
    await app.aclose()

    print("\nevents seen by EventCounter:")
    for key, count in sorted(counter.counts.items()):
        print(f"  {key:<32} {count}")

    # Same thing, built explicitly:
    _ = CompositePublisher([ConsolePublisher(), EventCounter()])


if __name__ == "__main__":
    asyncio.run(main())
