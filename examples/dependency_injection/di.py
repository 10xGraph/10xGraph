"""Register your own services and receive them in tools and nodes with Inject[T].

A fake repository and a settings object go into an InjectQ container. The container is passed
to ReactAgent, and a tool receives the services as Inject[...] defaults.
Run: OPENAI_API_KEY=... python examples/dependency_injection/di.py
"""

import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from injectq import Inject, InjectQ

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.utils.injection import fresh


load_dotenv()


@dataclass
class Settings:
    currency: str = "USD"
    max_results: int = 3


class OrderRepository:
    """Stand-in for a database client."""

    def __init__(self) -> None:
        self._orders = {"alice": [("A-100", 42.0), ("A-101", 17.5)], "bob": [("B-200", 99.0)]}

    def for_customer(self, customer: str) -> list[tuple[str, float]]:
        return self._orders.get(customer.lower(), [])


def build_container(repo: OrderRepository | None = None) -> InjectQ:
    container = InjectQ()
    container.bind_instance(Settings, Settings())
    container.bind_instance(OrderRepository, repo or OrderRepository())
    return container


# Parameters with an Inject[...] default are filled from the container and left out of the tool
# schema, so the model only sees `customer`. fresh() resolves them from the active container on
# every call; without it the first resolution is cached for the whole process, which breaks
# rebinding in tests.
def list_orders(
    customer: str,
    repo: OrderRepository = Inject[OrderRepository],
    settings: Settings = Inject[Settings],
) -> str:
    """List the recent orders of a customer."""
    repo, settings = fresh(repo), fresh(settings)
    orders = repo.for_customer(customer)[: settings.max_results]
    if not orders:
        return f"No orders for {customer}."
    return "; ".join(f"{oid}: {amount:.2f} {settings.currency}" for oid, amount in orders)


def build_agent(container: InjectQ) -> ReactAgent:
    return ReactAgent(
        model="gpt-5-mini",
        system_prompt=[{"role": "system", "content": "Answer questions about customer orders."}],
        tools=[list_orders],
        container=container,  # the graph activates this container when it is built
    )


async def main() -> None:
    app = build_agent(build_container()).compile()
    result = await app.ainvoke(
        {"messages": [Message.text_message("What did alice order?")]},
        config={"thread_id": "di-demo"},
    )
    print("assistant:", result["messages"][-1].text())
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
