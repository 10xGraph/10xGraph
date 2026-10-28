"""Stream a ReactAgent run with astream() and read the StreamChunk events.

Text arrives as delta messages; a tool reports progress through an injected StreamEmitter.

Run: python examples/streaming/stream_agent.py   (needs ANTHROPIC_API_KEY)
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message, StreamChunk, StreamEmitter, StreamEvent
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()


async def search_flights(
    origin: str,
    destination: str,
    # Injected by name: `emit` is hidden from the model's tool schema. It is a
    # StreamEmitter under astream() and None under ainvoke(), so always guard it.
    emit: StreamEmitter | None = None,
) -> str:
    """Search for flights between two cities."""
    for step in ("Contacting airlines", "Comparing prices", "Ranking results"):
        if emit:
            emit.progress(step, data={"route": f"{origin}-{destination}"})
        await asyncio.sleep(0.5)
    return f"Cheapest {origin} to {destination}: $212 nonstop, departs 08:15."


app = ReactAgent(
    model="claude-sonnet-5-5",
    system_prompt=[{"role": "system", "content": "You are a travel assistant. Be brief."}],
    tools=[search_flights],
).compile()


def show(chunk: StreamChunk) -> None:
    # chunk.event is one of StreamEvent.MESSAGE / STATE / UPDATES / ERROR.
    if chunk.event == StreamEvent.MESSAGE:
        msg = chunk.message
        if msg is None:
            # No Message attached: this is a tool progress chunk built by StreamEmitter.
            if chunk.data and chunk.data.get("status") == "tool_progress":
                print(f"\n[progress] {chunk.data['tool_name']}: {chunk.data['message']}")
        elif msg.delta:
            # Partial text from the model. Print it without a newline as it arrives.
            print(msg.text(), end="", flush=True)
        elif msg.role == "assistant" and msg.tools_calls:
            names = [call["function"]["name"] for call in msg.tools_calls]
            print(f"\n[model asked for tools: {names}]")
        elif msg.role == "tool":
            print(f"[tool result] {msg.text()}")
        # A final non-delta assistant message repeats the text already printed from
        # the deltas, so it is skipped here.
    elif chunk.event == StreamEvent.ERROR:
        print(f"\n[error] {chunk.data}")


async def main() -> None:
    async with app:
        inp = {"messages": [Message.text_message("Find me a flight from Oslo to Rome.")]}
        async for chunk in app.astream(inp, config={"thread_id": "stream-1"}):
            show(chunk)
        print()


if __name__ == "__main__":
    asyncio.run(main())
