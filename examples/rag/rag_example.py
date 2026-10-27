"""RAGAgent over a local Qdrant knowledge base, with optional rerankers.

Ingests a few support-policy chunks, then answers questions in three setups:
plain vector search, Cohere rerank (needs COHERE_API_KEY) and a local cross-encoder
(needs sentence-transformers). Gemini writes the answers and embeds the documents.

Run:  python examples/rag/rag_example.py
"""

import asyncio
import os
import shutil
import tempfile

from dotenv import load_dotenv

from tenxgraph.core.graph import Agent
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent.rag import CohereReranker, CrossEncoderReranker, RAGAgent
from tenxgraph.storage.store import GoogleEmbedding, create_local_qdrant_store, system_user_id


load_dotenv()

DOCUMENTS = [
    "Returns are accepted within 30 days of purchase with a receipt.",
    "Items must be in original packaging to qualify for a full refund.",
    "Digital downloads are non-refundable once accessed.",
    "Shipping costs are refunded only when the return is due to our error.",
    "To start a return, email support@example.com with your order number.",
    "Exchanges can be done in store or by mail within 60 days.",
    "Sale items are final sale and cannot be returned or exchanged.",
    "Gift cards and store credit are non-refundable.",
    "Damaged or defective items are replaced free of charge at any time.",
    "International orders may carry extra return shipping fees.",
]

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a customer-support agent. Answer only from the retrieved "
            "documents. If they do not contain the answer, say so."
        ),
    }
]

QDRANT_PATH = tempfile.mkdtemp(prefix="10xgraph-rag-")
store = create_local_qdrant_store(
    path=QDRANT_PATH,
    embedding=GoogleEmbedding(model="gemini-embedding-001"),
    collection="support_kb",
)


async def ingest() -> None:
    """Index the chunks once. The shared knowledge base is owned by a reserved id."""
    await store.asetup()
    owner = {"user_id": system_user_id("knowledge_base")}
    for chunk in DOCUMENTS:
        await store.astore(owner, chunk, category="support-policy")


def build_agent() -> Agent:
    # RAGAgent calls agent.execute(...) itself and injects the retrieved documents,
    # so this is a plain Agent with no tools.
    return Agent(model="gemini-2.5-flash", system_prompt=SYSTEM_PROMPT)


async def ask(title: str, rag: RAGAgent, question: str) -> None:
    app = rag.compile()
    result = await app.ainvoke({"messages": [Message.text_message(question)]}, {"thread_id": title})
    print(f"--- {title}\nQ: {question}\nA: {result['messages'][-1].text()}\n")
    await app.aclose()


async def main() -> None:
    await ingest()

    # 1. Plain RAG. With no reranker, all top_k hits go to the model.
    await ask(
        "plain",
        RAGAgent(store=store, agent=build_agent(), top_k=4),
        "Can I return a digital download?",
    )

    # 2. Retrieve many candidates, let Cohere pick the best few.
    if os.getenv("COHERE_API_KEY"):
        reranker = CohereReranker(api_key=os.environ["COHERE_API_KEY"])
        await ask(
            "cohere",
            RAGAgent(store=store, agent=build_agent(), reranker=reranker, top_k=8, top_n=3),
            "Can I return a damaged item?",
        )
    else:
        print("Skipping Cohere: set COHERE_API_KEY.\n")

    # 3. Same idea with a local model, no API call.
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("Skipping cross-encoder: pip install sentence-transformers.\n")
    else:
        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        await ask(
            "cross-encoder",
            RAGAgent(store=store, agent=build_agent(), reranker=reranker, top_k=8, top_n=3),
            "Are sale items refundable?",
        )

    shutil.rmtree(QDRANT_PATH, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
