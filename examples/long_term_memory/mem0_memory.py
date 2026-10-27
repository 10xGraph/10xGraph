"""Long-term memory for a Claude agent backed by Mem0.

Same agent-level `MemoryConfig` as qdrant_memory.py, but the store is `Mem0Store`.
Mem0 extracts and deduplicates facts itself, using its own LLM and embedder (set in the
config dict below, OpenAI here). Mem0 can keep its vectors in Qdrant.

Run:  python examples/long_term_memory/mem0_memory.py
"""

import asyncio
import shutil
import tempfile

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.store import Mem0Store, MemoryConfig, ReadMode


load_dotenv()

QDRANT_PATH = tempfile.mkdtemp(prefix="10xgraph-mem0-")

# Mem0's own configuration: where it keeps vectors, which LLM extracts facts, which
# model embeds them. This is independent of the chat model the agent uses.
MEM0_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "assistant_mem0",
            "path": QDRANT_PATH,
            "embedding_model_dims": 1536,  # text-embedding-3-small
        },
    },
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    "llm": {"provider": "openai", "config": {"model": "gpt-5-mini"}},
}

# app_id scopes memories in Mem0 and defaults to a generic value, so set your own.
store = Mem0Store(config=MEM0_CONFIG, app_id="assistant-demo")

agent = ReactAgent(
    model="claude-sonnet-5-5",
    system_prompt=[
        {
            "role": "system",
            "content": "You are a personal assistant. Remember what users tell you.",
        }
    ],
    memory=MemoryConfig(
        store=store,
        retrieval_mode=ReadMode.POSTLOAD,  # the model calls user_memory_tool
        limit=5,
    ),
)


async def say(app, text: str, user_id: str, thread_id: str) -> None:
    config = {"user_id": user_id, "thread_id": thread_id}
    result = await app.ainvoke({"messages": [Message.text_message(text)]}, config)
    print(f"[{user_id}/{thread_id}] user: {text}")
    print("assistant:", result["messages"][-1].text(), "\n")


async def main() -> None:
    app = agent.compile(checkpointer=InMemoryCheckpointer())
    await say(app, "My dog is called Biscuit and I work night shifts.", "alice", "s1")
    await app.aclose()  # flush background memory writes

    app = agent.compile(checkpointer=InMemoryCheckpointer())
    await say(app, "What is my dog's name, and when do I work?", "alice", "s2")
    await app.aclose()
    shutil.rmtree(QDRANT_PATH, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
