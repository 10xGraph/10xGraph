"""Invocation callbacks: log, time and cost-track every model and tool call.

A CallbackManager runs before_invoke, after_invoke and on_error hooks around each
AI (model) and TOOL invocation. Here they log calls, measure latency and add up
token usage and an estimated cost.

Run: python examples/callbacks/invoke_hooks.py   (needs GEMINI_API_KEY)
"""

import asyncio
import logging
import time
from typing import Any

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.utils.callbacks import CallbackContext, CallbackManager, InvocationType


load_dotenv()
logging.basicConfig(level=logging.WARNING)

# Illustrative prices in USD per 1M tokens (input, output). Check your provider's price list.
PRICE_PER_M_TOKENS = {"gemini-2.5-flash": (0.30, 2.50)}


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    if city.lower() == "atlantis":
        raise ValueError("No weather station for Atlantis")
    return f"It is 21C and sunny in {city}."


class RunStats:
    """Accumulates call counts and latency across one run."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {"ai": 0, "tool": 0}
        self.seconds: dict[str, float] = {"ai": 0.0, "tool": 0.0}
        self.errors = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add_usage(self, messages: list[Message]) -> None:
        # Token counts live on the assistant messages, not on the callback payload.
        for msg in messages:
            if msg.usages:
                self.input_tokens += msg.usages.prompt_tokens
                self.output_tokens += msg.usages.completion_tokens

    def cost(self, model: str) -> float:
        price_in, price_out = PRICE_PER_M_TOKENS[model]
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


stats = RunStats()


# The same CallbackContext object reaches before_invoke and after_invoke for one call,
# so metadata is a convenient place to carry a start time between the two.
async def start_timer(context: CallbackContext, input_data: Any) -> Any:
    context.metadata = context.metadata or {}
    context.metadata["t0"] = time.perf_counter()
    print(f"[before] {context.invocation_type} {context.function_name or context.node_name}")
    return input_data  # Return the input unchanged, or return a modified copy.


async def stop_timer(context: CallbackContext, input_data: Any, output: Any) -> Any:
    kind = context.invocation_type.value
    elapsed = time.perf_counter() - (context.metadata or {}).get("t0", time.perf_counter())
    stats.calls[kind] += 1
    stats.seconds[kind] += elapsed
    print(f"[after]  {kind} {context.function_name or context.node_name} took {elapsed:.2f}s")
    return output


async def log_error(context: CallbackContext, input_data: Any, error: Exception) -> None:
    stats.errors += 1
    print(f"[error]  {context.invocation_type} {context.function_name}: {error}")
    return None  # None means "no recovery value"; the error is handled as usual.


def build_callback_manager() -> CallbackManager:
    manager = CallbackManager()
    for invocation_type in (InvocationType.AI, InvocationType.TOOL):
        manager.register_before_invoke(invocation_type, start_timer)
        manager.register_after_invoke(invocation_type, stop_timer)
        manager.register_on_error(invocation_type, log_error)
    return manager


async def main() -> None:
    agent = ReactAgent(
        model="gemini-2.5-flash",
        tools=[get_weather],
        system_prompt=[{"role": "system", "content": "Use get_weather for weather questions."}],
    )
    app = agent.compile(callback_manager=build_callback_manager())

    result = await app.ainvoke(
        {"messages": [Message.text_message("Weather in Paris and in Atlantis?")]},
        config={"thread_id": "hooks-1"},
    )
    messages = result["messages"]
    print("assistant:", messages[-1].text())

    stats.add_usage(messages)
    print(f"model calls: {stats.calls['ai']} ({stats.seconds['ai']:.2f}s)")
    print(
        f"tool calls:  {stats.calls['tool']} ({stats.seconds['tool']:.2f}s), errors: {stats.errors}"
    )
    print(f"tokens: in={stats.input_tokens} out={stats.output_tokens}")
    print(f"estimated cost: ${stats.cost('gemini-2.5-flash'):.6f}")

    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
