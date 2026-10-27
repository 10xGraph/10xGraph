"""Stream run events to Redis, Kafka or RabbitMQ so other services can consume them.

Pick a broker with the BROKER environment variable (redis, kafka or rabbitmq). Each
publisher takes a plain config dict and connects on first use. A consumer elsewhere
(a dashboard, an audit service, a billing job) reads the JSON events from the broker.

Run: BROKER=redis python examples/observability/brokers.py   (needs OPENAI_API_KEY and a broker)
"""

import asyncio
import os

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.runtime.publisher import (
    BasePublisher,
    KafkaPublisher,
    RabbitMQPublisher,
    RedisPublisher,
)


load_dotenv()


def get_order_status(order_id: str) -> str:
    """Return the status of an order."""
    return f"Order {order_id} is out for delivery."


def build_publisher(broker: str) -> BasePublisher:
    if broker == "redis":
        # pip install "10xgraph[redis]". mode="stream" uses a Redis Stream instead of pub/sub.
        return RedisPublisher(
            {"url": "redis://localhost:6379/0", "mode": "pubsub", "channel": "tenxgraph.events"}
        )
    if broker == "kafka":
        # pip install "10xgraph[kafka]". TLS and SASL options are opt-in config keys.
        return KafkaPublisher({"bootstrap_servers": "localhost:9092", "topic": "tenxgraph.events"})
    if broker == "rabbitmq":
        # pip install "10xgraph[rabbitmq]". Use an amqps:// URL for TLS.
        return RabbitMQPublisher(
            {
                "url": "amqp://guest:guest@localhost/",
                "exchange": "tenxgraph.events",
                "routing_key": "tenxgraph.events",
            }
        )
    raise SystemExit(f"Unknown BROKER {broker!r}. Use redis, kafka or rabbitmq.")


async def main() -> None:
    broker = os.getenv("BROKER", "redis").lower()
    agent = ReactAgent(
        model="gpt-5-mini",
        tools=[get_order_status],
        publisher=build_publisher(broker),
    )
    app = agent.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("Where is order 8841?")]},
        config={"thread_id": f"{broker}-1", "user_id": "user-123"},
    )
    print("assistant:", result["messages"][-1].text())
    print(f"events published to {broker}")

    # Waits for queued publishes and closes the broker connection.
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
