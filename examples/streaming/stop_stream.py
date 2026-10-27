"""Stop a running stream from outside with astop().

A second task requests the stop for the same thread_id. The graph checks the flag
between nodes, so the run ends after the current node finishes.

Run: python examples/streaming/stop_stream.py   (needs ANTHROPIC_API_KEY)
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message, StreamEvent
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()


async def slow_lookup(topic: str) -> str:
    """Look up one fact about a topic. Slow."""
    await asyncio.sleep(2)
    return f"Fact about {topic}: it is interesting."


app = ReactAgent(
    model="claude-sonnet-5-5",
    system_prompt=[
        {
            "role": "system",
            "content": "Call slow_lookup once for each of ten different topics, one at a time.",
        }
    ],
    tools=[slow_lookup],
).compile(
    # astop() works through the checkpointer: it records the stop request on the
    # thread. Without one it returns {"ok": False, "reason": "no-checkpointer"}.
    checkpointer=InMemoryCheckpointer(),
)

CONFIG = {"thread_id": "stop-demo"}  # astop() finds the run by this thread_id


async def consume() -> None:
    inp = {"messages": [Message.text_message("Tell me facts about ten topics.")]}
    async for chunk in app.astream(inp, config=CONFIG):
        if chunk.event == StreamEvent.MESSAGE and chunk.message and not chunk.message.delta:
            print(f"{chunk.message.role}: {chunk.message.text()[:80]}")
    print("stream ended")


async def main() -> None:
    async with app:
        reader = asyncio.create_task(consume())
        await asyncio.sleep(5)  # let a few tool calls happen

        status = await app.astop(CONFIG)
        # {"ok": True, "running": True} when a run was active;
        # {"ok": True, "running": False, "reason": "not-running"} if it already ended.
        print("stop requested:", status)

        await reader


if __name__ == "__main__":
    asyncio.run(main())
