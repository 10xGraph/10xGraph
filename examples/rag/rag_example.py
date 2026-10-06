"""RAGAgent example — knowledge-base Q&A with optional reranking.

Three progressively richer scenarios:

1. **Simple** — vector search → LLM answer (no reranker)
2. **With Cohere Rerank** — retrieve 20 candidates, rerank to top 5
3. **Fully local** — local CrossEncoder reranker, no external API

Run (requires OPENAI_API_KEY for the LLM)::

    OPENAI_API_KEY=sk-... python examples/rag/rag_example.py

To use Cohere reranking (scenario 2) also set COHERE_API_KEY.
To use local reranking (scenario 3) install sentence-transformers:

    pip install sentence-transformers
"""

from __future__ import annotations

import asyncio
import os

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.message import Message
from tenxgraph.prebuilt.agent.rag import (
    RAGAgent,
    CohereReranker,
    CrossEncoderReranker,
)
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.store_schema import MemorySearchResult


# ---------------------------------------------------------------------------
# Stub store — replace with QdrantStore in production
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    "Our refund policy allows returns within 30 days of purchase with a receipt.",
    "Products must be in original packaging to qualify for a full refund.",
    "Digital downloads are non-refundable once accessed.",
    "Shipping costs are non-refundable unless the return is due to our error.",
    "To initiate a return, contact support@example.com with your order number.",
    "Exchanges can be done in-store or by mail within 60 days.",
    "Sale items are final sale and cannot be returned or exchanged.",
    "Gift cards and store credits are non-refundable.",
    "Damaged or defective items are eligible for free replacement at any time.",
    "International orders may have additional return shipping fees.",
]


class StubStore(BaseStore):
    """Returns the top-k chunks by simple keyword overlap (demo only).

    In production use QdrantStore::

        from tenxgraph.storage import create_local_qdrant_store
        from tenxgraph.storage.store.embedding import OpenAIEmbedding

        store = create_local_qdrant_store(
            path="./knowledge_base",
            embedding=OpenAIEmbedding(model="text-embedding-3-small"),
        )
    """

    async def asetup(self): ...

    async def asearch(self, config, query: str, limit: int = 5, **kwargs):
        query_words = set(query.lower().split())
        scored = []
        for chunk in KNOWLEDGE_BASE:
            overlap = len(query_words & set(chunk.lower().split()))
            scored.append((overlap, chunk))
        scored.sort(reverse=True)
        return [
            MemorySearchResult(content=chunk, score=float(score))
            for score, chunk in scored[:limit]
            if score > 0 or not query_words
        ]

    async def astore(self, config, content, **kwargs):
        return "id"

    async def aget(self, config, record_id, **kwargs):
        return None

    async def aget_all(self, config, **kwargs):
        return []

    async def aupdate(self, config, record_id, **kwargs): ...
    async def adelete(self, config, record_id, **kwargs): ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a helpful customer-support agent. "
            "Answer questions using ONLY the provided context. "
            "If the context does not contain the answer, say so."
        ),
    }
]


def _make_state(question: str) -> AgentState:
    state = AgentState()
    state.context = [Message.text_message(question, role="user")]  # type: ignore[arg-type]
    return state


# ---------------------------------------------------------------------------
# Scenario 1 — Simple: vector search → LLM (no reranker)
# ---------------------------------------------------------------------------


async def scenario_simple():
    print("\n" + "=" * 60)
    print("Scenario 1: Simple RAG (no reranker)")
    print("=" * 60)

    rag = RAGAgent(
        store=StubStore(),
        agent=Agent(
            model="gpt-4o-mini",
            system_prompt=SYSTEM_PROMPT,
        ),
        top_k=4,
    )
    app = rag.compile()

    state = _make_state("What is your refund policy for digital products?")
    result = await app.ainvoke(state, config={})

    last = next(m for m in reversed(result.context) if m.role == "assistant")
    print(f"\nAnswer: {last.text()}")


# ---------------------------------------------------------------------------
# Scenario 2 — With Cohere Rerank
# ---------------------------------------------------------------------------


async def scenario_cohere_rerank():
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("\nSkipping Scenario 2: set COHERE_API_KEY to enable Cohere reranking.")
        return

    print("\n" + "=" * 60)
    print("Scenario 2: RAG with Cohere Rerank (rerank-v4.0-pro)")
    print("=" * 60)

    rag = RAGAgent(
        store=StubStore(),
        agent=Agent(
            model="gpt-4o",
            system_prompt=SYSTEM_PROMPT,
        ),
        reranker=CohereReranker(api_key=cohere_key, model="rerank-v4.0-pro"),
        top_k=8,  # retrieve 8 candidates …
        top_n=3,  # … rerank to best 3 for the LLM
    )
    app = rag.compile()

    state = _make_state("Can I return a damaged item?")
    result = await app.ainvoke(state, config={})

    last = next(m for m in reversed(result.context) if m.role == "assistant")
    print(f"\nAnswer: {last.text()}")


# ---------------------------------------------------------------------------
# Scenario 3 — Fully local CrossEncoder reranker
# ---------------------------------------------------------------------------


async def scenario_local_reranker():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("\nSkipping Scenario 3: pip install sentence-transformers to enable.")
        return

    print("\n" + "=" * 60)
    print("Scenario 3: RAG with local CrossEncoder (no external API)")
    print("=" * 60)

    rag = RAGAgent(
        store=StubStore(),
        agent=Agent(
            model="gpt-4o-mini",
            system_prompt=SYSTEM_PROMPT,
        ),
        reranker=CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_k=6,
        top_n=3,
    )
    app = rag.compile()

    state = _make_state("Are sale items refundable?")
    result = await app.ainvoke(state, config={})

    last = next(m for m in reversed(result.context) if m.role == "assistant")
    print(f"\nAnswer: {last.text()}")


# ---------------------------------------------------------------------------
# How to use QdrantStore (production setup comment)
# ---------------------------------------------------------------------------

"""
PRODUCTION SETUP — swap StubStore for QdrantStore:

    from tenxgraph.storage import create_local_qdrant_store
    from tenxgraph.storage.store.embedding import OpenAIEmbedding

    # 1. Create the store
    store = create_local_qdrant_store(
        path="./knowledge_base",
        embedding=OpenAIEmbedding(model="text-embedding-3-small"),
    )

    # 2. Index your documents once
    await store.asetup()
    for chunk in your_document_chunks:
        await store.astore(config={}, content=chunk, category="docs")

    # 3. Build the RAG agent
    rag = RAGAgent(
        store=store,
        agent=Agent(model="gpt-4o-mini", system_prompt=SYSTEM_PROMPT),
        reranker=CohereReranker(api_key=os.getenv("COHERE_API_KEY")),
        top_k=20,
        top_n=5,
    )
    app = rag.compile(checkpointer=InMemoryCheckpointer())
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main():
    await scenario_simple()
    await scenario_cohere_rerank()
    await scenario_local_reranker()


if __name__ == "__main__":
    asyncio.run(main())
