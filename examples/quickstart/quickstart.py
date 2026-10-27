"""Smallest useful 10xGraph agent: a ReactAgent with one tool, called with ainvoke.

Run: python examples/quickstart/quickstart.py   (needs OPENAI_API_KEY)
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()


def get_weather(location: str) -> str:
    """Get the current weather for a city."""
    # The docstring and type hints become the tool schema the model sees.
    return f"The weather in {location} is sunny, 24 C."


# ReactAgent wires the model node, the tool node and the "call tools until done"
# routing for you. The model is just a string; the provider is detected from it.
agent = ReactAgent(
    model="gpt-5-mini",
    system_prompt=[{"role": "system", "content": "You are a concise weather assistant."}],
    tools=[get_weather],
)

# compile() returns a CompiledGraph. With no checkpointer it uses an in-memory one.
app = agent.compile()


async def main() -> None:
    # `async with` calls app.aclose() on exit, which flushes background checkpoint
    # writes and releases the checkpointer, store and publisher. You can also call
    # `await app.aclose()` yourself in a finally block.
    async with app:
        result = await app.ainvoke(
            {"messages": [Message.text_message("What is the weather in Tokyo?")]},
            # thread_id names the conversation; reuse it to continue the same chat.
            config={"thread_id": "quickstart-1"},
        )

        # The default response granularity (LOW) returns only messages.
        for message in result["messages"]:
            print(f"{message.role}:", message.text() or message.tools_calls)

        print("\nassistant:", result["messages"][-1].text())


if __name__ == "__main__":
    asyncio.run(main())
