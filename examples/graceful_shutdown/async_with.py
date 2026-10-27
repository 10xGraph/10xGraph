"""The short form: `async with graph:` calls aclose() for you.

Shows that the context manager releases the checkpointer, publisher and store even
when the body raises. Run: python examples/graceful_shutdown/async_with.py
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()


async def main() -> None:
    agent = ReactAgent(
        model="gemini-2.5-flash",
        system_prompt=[{"role": "system", "content": "Answer in one sentence."}],
    )
    graph = agent.compile(checkpointer=InMemoryCheckpointer(), shutdown_timeout=10.0)

    # __aexit__ runs aclose() whether the body finishes, raises or is cancelled.
    async with graph:
        result = await graph.ainvoke(
            {"messages": [Message.text_message("Why close a graph before exiting?")]},
            {"thread_id": "demo", "user_id": "demo-user"},
        )
        print("assistant:", result["messages"][-1].text())

    # aclose() is idempotent: a second call returns immediately.
    print(await graph.aclose())  # {'status': 'already_closed'}


if __name__ == "__main__":
    asyncio.run(main())
