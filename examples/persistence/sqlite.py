"""Persist conversation state in one local SQLite file with SqliteCheckpointer.

Run it twice with the same thread_id: the second process continues the first. Then it
lists threads and reads state and messages back.
Run: python examples/persistence/sqlite.py
"""

import asyncio
import os

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import SqliteCheckpointer
from tenxgraph.utils import ThreadInfo


load_dotenv()


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 21 C and sunny in {city}."


async def chat(graph, text: str, config: dict) -> None:
    result = await graph.ainvoke({"messages": [Message.text_message(text)]}, config)
    print("user:     ", text)
    print("assistant:", result["messages"][-1].text())


async def main() -> None:
    # One file holds state, messages and threads. Needs: pip install "10xgraph[sqlite_checkpoint]"
    # The default path (db_path=None) is ~/.10xgraph/checkpointer.db.
    checkpointer = SqliteCheckpointer("conversations.db")

    agent = ReactAgent(
        model="deepseek-chat",  # DeepSeek through its OpenAI-compatible endpoint
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        system_prompt=[{"role": "system", "content": "You are a friendly assistant."}],
        tools=[get_weather],
    )
    graph = agent.compile(checkpointer=checkpointer)
    # thread_id picks the conversation. user_id scopes ownership of the thread.
    config = {"thread_id": "customer-42", "user_id": "demo-user"}

    # Run this script twice. The first run teaches the agent a fact and the second run
    # finds it again, because the thread was saved outside this process.
    saved = await checkpointer.alist_messages(config)
    print(f"Restored {len(saved)} message(s) for this thread.\n")
    if not saved:
        await chat(graph, "Hi, my name is Sam and I live in Lisbon.", config)
    else:
        await chat(graph, "What is my name, and what is the weather where I live?", config)

    # The graph saves state and messages by itself. Thread metadata, such as a display
    # name for a thread list, is registered explicitly. It is optional.
    if await checkpointer.aget_thread(config) is None:
        info = ThreadInfo(
            thread_id=config["thread_id"], thread_name="Customer 42 chat", user_id=config["user_id"]
        )
        await checkpointer.aput_thread(config, info)

    # Read what was stored. All of these take the same config dict.
    state = await checkpointer.aget_state(config)
    print(f"\nstate: {len(state.context)} messages, step {state.execution_meta.step}")

    messages = await checkpointer.alist_messages(config)
    for m in messages:
        print(f"  [{m.role}] {m.text()[:70]}")

    threads = await checkpointer.alist_threads(config)
    print("threads:", [(t.thread_id, t.thread_name) for t in threads])
    info = await checkpointer.aget_thread(config)
    print("thread info:", info.thread_id if info else None)

    # Flushes background checkpoint writes and releases the connections.
    await graph.aclose()


if __name__ == "__main__":
    asyncio.run(main())
