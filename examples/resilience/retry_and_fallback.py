"""Keep an agent answering when its LLM provider fails.

Primary model is Anthropic. RetryConfig retries transient errors with exponential
backoff and adds a circuit breaker. fallback_models switches to OpenAI, then Google.
Run: python examples/resilience/retry_and_fallback.py
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.exceptions import GraphError
from tenxgraph.core.graph import RetryConfig
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()


def get_ticket_status(ticket_id: str) -> str:
    """Return the status of a support ticket."""
    return f"Ticket {ticket_id} is open and assigned to the billing team."


retry_config = RetryConfig(
    max_retries=2,  # up to 3 calls per model: 1 try + 2 retries
    initial_delay=1.0,  # wait 1s, then 2s (delay * backoff_factor ** n)
    backoff_factor=2.0,
    max_delay=20.0,  # cap on any single wait
    retryable_status_codes=frozenset({429, 500, 502, 503, 529}),  # the default set
    # Circuit breaker, tracked per (provider, model) on this agent. After 3 failed
    # attempt cycles in a row the model is skipped for 60s, so a dead provider is not
    # retried on every request. After 60s one trial call is allowed.
    circuit_breaker_enabled=True,
    circuit_breaker_threshold=3,
    circuit_breaker_reset_timeout=60.0,
)


async def main() -> None:
    agent = ReactAgent(
        model="claude-sonnet-5-5",
        system_prompt=[{"role": "system", "content": "You are a support assistant."}],
        tools=[get_ticket_status],
        retry_config=retry_config,
        # A bare string inherits the PRIMARY model's provider, so "gpt-5-mini" alone would be
        # sent to Anthropic. To cross companies, give a (model, provider) tuple.
        fallback_models=[
            ("gpt-5-mini", "openai"),
            ("gemini-2.5-flash", "google"),
        ],
    )

    graph = agent.compile(checkpointer=InMemoryCheckpointer())
    config = {"thread_id": "ticket-7", "user_id": "demo-user"}
    try:
        result = await graph.ainvoke(
            {"messages": [Message.text_message("What is the status of ticket T-884?")]},
            config,
        )
        print("assistant:", result["messages"][-1].text())
    except GraphError as exc:
        # Library errors carry a code and context you can log or return from an API.
        print("graph error:", exc.to_dict())
    except Exception as exc:
        # When the primary and every fallback fail, the LAST provider's exception is
        # re-raised unchanged (an SDK error such as openai.APIStatusError or a
        # google.genai error). Handle it here, or let the caller retry later.
        print("all models failed:", type(exc).__name__, exc)
    finally:
        await graph.aclose()


if __name__ == "__main__":
    asyncio.run(main())
