"""The docs assistant graph, loaded by the server through 10xgraph.json ("agent").

Three companies, one graph:
  chat model       Anthropic  claude-sonnet-5-5   (the ReAct agent below)
  query rewriter   Google     gemini-2.5-flash    (graph/tools.py)
  embeddings       OpenAI     text-embedding-3-small (graph/store.py)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from graph.store import get_store
from graph.tools import query_rewriter, search_docs
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import PgCheckpointer


load_dotenv()

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a documentation assistant for the Lumen product. "
            "For every question about the product: 1) call query_rewriter with the question, "
            "2) call search_docs with the rewritten query, 3) answer only from the returned "
            "passages and name the source file for each fact, for example (source: pricing.md). "
            "If the passages do not contain the answer, say you could not find it."
        ),
    }
]

# Postgres holds durable thread state; Redis caches it for fast reads while streaming.
checkpointer = PgCheckpointer(
    postgres_dsn=os.environ["POSTGRES_DSN"],
    redis_url=os.environ["REDIS_URL"],
)

react_agent = ReactAgent(
    model="claude-sonnet-5-5",
    system_prompt=SYSTEM_PROMPT,
    tools=[query_rewriter, search_docs],
    trim_context=True,
)

# `store` is registered so the server's graceful shutdown (compiled.aclose()) also
# releases the Qdrant client together with the checkpointer.
app = react_agent.compile(checkpointer=checkpointer, store=get_store())
