"""Long-term memory for a Claude agent on Postgres with pgvector.

One Postgres holds both the conversation checkpoints (PgCheckpointer) and the memories
(PgVectorStore). Session 1 tells the agent some facts. Session 2 is a new thread for the
same user_id and recalls them. The store searches with HYBRID retrieval (vector similarity
and full-text, fused with RRF), and one memory scope overrides that default.

Run:  docker compose -f examples/persistence/docker-compose.yml up -d
      export POSTGRES_DSN=postgresql://tenx:tenx@localhost:5432/tenx
      export REDIS_URL=redis://localhost:6379/0
      python examples/long_term_memory/pgvector_memory.py
"""

import asyncio
import os

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import PgCheckpointer
from tenxgraph.storage.store import (
    AgentMemoryConfig,
    MemoryConfig,
    OpenAIEmbedding,
    PgVectorStore,
    ReadMode,
    UserMemoryConfig,
    system_user_id,
)
from tenxgraph.storage.store.store_schema import RetrievalStrategy


load_dotenv()

POSTGRES_DSN = os.environ["POSTGRES_DSN"]
REDIS_URL = os.environ["REDIS_URL"]

# Requires the pgvector extension in the database (the pgvector/pgvector image has it).
store = PgVectorStore(
    embedding=OpenAIEmbedding(model="text-embedding-3-small"),
    dsn=POSTGRES_DSN,
    collection="assistant_memory",
    # Store-level default for every search that does not pick a strategy.
    # SIMILARITY: vectors only. TEXT: full-text only. HYBRID: both, fused with RRF.
    retrieval_strategy=RetrievalStrategy.HYBRID,
    # If the embedding provider fails, writes are saved as text and searches fall back
    # to full-text. Use "raise" to fail instead. abackfill_embeddings() fills gaps later.
    on_embedding_error="fallback",
)

memory = MemoryConfig(
    store=store,
    retrieval_mode=ReadMode.POSTLOAD,
    limit=5,
    # Settings resolve in three levels: store constructor -> MemoryConfig or scope ->
    # method call. This scope overrides the store's HYBRID default for agent memory.
    user_memory=UserMemoryConfig(),
    agent_memory=AgentMemoryConfig(
        enabled=True,
        agent_id="support-bot",
        retrieval_strategy=RetrievalStrategy.SIMILARITY,
    ),
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
    memory=memory,
)


async def say(app, text: str, user_id: str, thread_id: str) -> None:
    config = {"user_id": user_id, "thread_id": thread_id}
    result = await app.ainvoke({"messages": [Message.text_message(text)]}, config)
    print(f"[{user_id}/{thread_id}] user: {text}")
    print("assistant:", result["messages"][-1].text(), "\n")


async def main() -> None:
    checkpointer = PgCheckpointer(postgres_dsn=POSTGRES_DSN, redis_url=REDIS_URL)
    await checkpointer.asetup()
    await store.asetup()  # creates the extension, table and indexes; safe to repeat

    # Shared agent memory is owned by system_user_id(agent_id).
    await store.astore(
        {
            "user_id": system_user_id("support-bot"),
            "agent_id": "support-bot",
            "thread_id": "support-bot",
            "collection": "assistant_memory",
        },
        "Company policy: refunds are available within 30 days of purchase.",
    )

    app = agent.compile(checkpointer=checkpointer)
    await say(app, "I'm vegetarian and I live in Lisbon. Please remember that.", "alice", "s1")
    await app.aclose()  # memory writes run in the background; aclose() flushes them

    # Session 2: new thread, same user_id. Only the store carries the facts over.
    app = agent.compile(checkpointer=checkpointer)
    await say(app, "Suggest a dinner idea for tonight, close to home.", "alice", "s2")
    await say(app, "What is the refund policy?", "alice", "s2")

    # A different user_id sees none of alice's memories.
    await say(app, "What do you know about my diet?", "bob", "s3")

    # A search call can override the strategy again (third level).
    hits = await store.asearch(
        {"user_id": "alice", "collection": "assistant_memory"},
        "vegetarian",
        retrieval_strategy=RetrievalStrategy.TEXT,
    )
    print("alice TEXT search:", [h.content for h in hits])

    await app.aclose()
    await store.arelease()
    await checkpointer.arelease()


if __name__ == "__main__":
    asyncio.run(main())
