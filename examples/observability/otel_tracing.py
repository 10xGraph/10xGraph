"""OpenTelemetry tracing: turn a 10xGraph run into spans for any OTEL backend.

OtelPublisher rebuilds graph, node, LLM and tool spans from the event stream, with
token counts and model names as attributes. ObservabilityLevel controls how much data
goes on each span. This script prints spans to the console with the OTEL SDK.

Run: python examples/observability/otel_tracing.py   (needs OPENAI_API_KEY)
"""

import asyncio

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.runtime.publisher import ObservabilityLevel, OtelPublisher


load_dotenv()


def get_inventory(sku: str) -> str:
    """Return stock level for a SKU."""
    return f"SKU {sku}: 42 units in stock."


def configure_tracer_provider() -> None:
    # Swap ConsoleSpanExporter for an OTLP exporter to ship spans to Jaeger, Tempo, Datadog, etc.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


async def main() -> None:
    configure_tracer_provider()

    # SPANS    -> timing and structure only
    # STANDARD -> adds token counts, model name and request parameters (default)
    # FULL     -> adds prompts, responses and tool I/O. May contain personal data.
    publisher = OtelPublisher(level=ObservabilityLevel.STANDARD)

    # Pass the publisher when the graph is created: StateGraph(publisher=publisher), or
    # ReactAgent(publisher=publisher) as here. The publisher is bound into the dependency
    # injection container at that moment. setup_tracing(graph) sets the publisher on an
    # existing StateGraph afterwards, which does not reach that container in the current
    # release, so this example avoids it.
    agent = ReactAgent(model="gpt-5-mini", tools=[get_inventory], publisher=publisher)
    app = agent.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("How many units of SKU-77 do we have?")]},
        config={"thread_id": "otel-1", "user_id": "user-123"},
    )
    print("assistant:", result["messages"][-1].text())

    # Flushes background publish tasks so the final spans are closed before exit.
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
