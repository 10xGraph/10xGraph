"""Send traces to Logfire or LangSmith by swapping in a pre-configured publisher.

LogfirePublisher and LangsmithPublisher are OtelPublisher subclasses. Creating one
configures the OTEL exporter, so there is no separate tracer setup. Pick a backend
with TRACE_BACKEND.

Run: TRACE_BACKEND=logfire python examples/observability/logfire_langsmith.py
     (needs OPENAI_API_KEY plus LOGFIRE_TOKEN or LANGSMITH_API_KEY)
"""

import asyncio
import os

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.runtime.publisher import (
    LangsmithPublisher,
    LogfirePublisher,
    ObservabilityLevel,
    OtelPublisher,
)


load_dotenv()


def get_ticket(ticket_id: str) -> str:
    """Return the status of a support ticket."""
    return f"Ticket {ticket_id} is open and assigned to the billing team."


def build_publisher(backend: str) -> OtelPublisher:
    # Secrets come from LOGFIRE_TOKEN / LANGSMITH_API_KEY, never from code.
    if backend == "logfire":
        # pip install "10xgraph[logfire]"
        return LogfirePublisher(service_name="support-agent", level=ObservabilityLevel.STANDARD)
    if backend == "langsmith":
        # pip install "10xgraph[langsmith]"
        return LangsmithPublisher(project="support-agent", level=ObservabilityLevel.STANDARD)
    raise SystemExit("TRACE_BACKEND must be logfire or langsmith")


async def main() -> None:
    backend = os.getenv("TRACE_BACKEND", "logfire").lower()

    agent = ReactAgent(
        model="gpt-5-mini",
        tools=[get_ticket],
        publisher=build_publisher(backend),
    )
    app = agent.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("What is the status of ticket T-311?")]},
        config={"thread_id": f"{backend}-1"},
    )
    print("assistant:", result["messages"][-1].text())
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
