"""Long-term memory for a Claude agent on a local Qdrant store.

Session 1 tells the agent some facts about a user. Session 2 is a brand new thread for
the same user_id and recalls them. A second user sees none of it. The example also shows
the three retrieval modes and shared agent memory (see the README).

Run:  python examples/long_term_memory/qdrant_memory.py
"""

import asyncio
import shutil
import tempfile

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.store import (
    AgentMemoryConfig,
    MemoryConfig,
    OpenAIEmbedding,
    ReadMode,
    UserMemoryConfig,
    create_local_qdrant_store,
    system_user_id,
)


load_dotenv()

# A throwaway directory keeps the demo repeatable. Point `path` at a stable folder
# (or use create_remote_qdrant_store) to keep memories between runs.
QDRANT_PATH = tempfile.mkdtemp(prefix="10xgraph-memory-")

store = create_local_qdrant_store(
    path=QDRANT_PATH,
    embedding=OpenAIEmbedding(model="text-embedding-3-small"),
    collection="assistant_memory",
)

memory = MemoryConfig(
    store=store,
    # POSTLOAD (default): the model gets user_memory_tool and decides when to search or
    #   remember. PRELOAD: relevant memories are searched on every turn and attached to the
    #   user message, with no write tool. NO_RETRIEVAL: no memory tools or injection.
    retrieval_mode=ReadMode.POSTLOAD,
    limit=5,
    score_threshold=0.0,
    # User memory is on by default and keyed by the user_id in the run config.
    user_memory=UserMemoryConfig(),
    # Agent memory is shared by every user and read-only for the model. It is off by
    # default and needs an agent_id (or app_id) to name its owner.
    agent_memory=AgentMemoryConfig(enabled=True, agent_id="support-bot"),
)

agent = ReactAgent(
    model="claude-sonnet-5-5",
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You are a personal assistant. Save durable facts and preferences the "
                "user tells you, and search memory before answering questions about them."
            ),
        }
    ],
    memory=memory,  # forwarded to the Agent, which registers the memory tools
)


async def seed_agent_memory() -> None:
    """Write one shared fact. Agent memory is owned by system_user_id(agent_id)."""
    await store.asetup()
    await store.astore(
        {
            "user_id": system_user_id("support-bot"),
            "agent_id": "support-bot",
            "thread_id": "support-bot",
            "collection": "assistant_memory",
        },
        "Company policy: refunds are available within 30 days of purchase.",
    )


async def say(app, text: str, user_id: str, thread_id: str) -> None:
    config = {"user_id": user_id, "thread_id": thread_id}
    result = await app.ainvoke({"messages": [Message.text_message(text)]}, config)
    print(f"[{user_id}/{thread_id}] user: {text}")
    print("assistant:", result["messages"][-1].text(), "\n")


async def main() -> None:
    app = agent.compile(checkpointer=InMemoryCheckpointer())
    await seed_agent_memory()

    # Session 1: the agent saves facts about alice.
    await say(app, "I'm vegetarian and I live in Lisbon. Please remember that.", "alice", "s1")
    await app.aclose()  # memory writes run in the background; aclose() flushes them

    # Session 2: new thread, same user_id. No chat history is shared, only the store.
    app = agent.compile(checkpointer=InMemoryCheckpointer())
    await say(app, "Suggest a dinner idea for tonight, close to home.", "alice", "s2")
    await say(app, "What is the refund policy?", "alice", "s2")  # agent memory

    # A different user_id gets a different slice of the store.
    await say(app, "What do you know about my diet?", "bob", "s3")

    await app.aclose()
    shutil.rmtree(QDRANT_PATH, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
