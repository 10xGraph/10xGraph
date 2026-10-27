"""Tools the docs assistant can call: a Gemini query rewriter and a Qdrant search."""

from __future__ import annotations

import json
from typing import Any

from graph.store import KB_CONFIG, get_store
from tenxgraph.core.llm import call_llm
from tenxgraph.utils import tool


REWRITER_PROMPT = (
    "Rewrite the user's question as a short keyword-style search query for a product "
    "documentation index. Resolve pronouns, drop filler words, keep product names and "
    "numbers. Reply with the query only."
)


@tool(
    name="query_rewriter",
    description=(
        "Rewrite a user question into a concise search query. Call this first, then pass "
        "its result to search_docs."
    ),
    tags=["rag"],
)
async def query_rewriter(question: str) -> str:
    # A cheap, fast model from a different company than the chat model. call_llm picks
    # the Google client from the model name.
    text, *_ = await call_llm(
        "gemini-2.5-flash",
        question,
        system_prompt=REWRITER_PROMPT,
        max_tokens=64,
        temperature=0.0,
    )
    return text.strip() or question


def format_hits(results: list[Any]) -> list[dict[str, Any]]:
    """Reduce store results to what the model and the UI need. The UI reads `source`."""
    return [
        {
            "source": r.metadata.get("source", "unknown"),
            "title": r.metadata.get("title", ""),
            "score": round(float(r.score), 3),
            "content": r.content,
        }
        for r in results
    ]


@tool(
    name="search_docs",
    description=(
        "Search the product documentation. Returns a JSON list of passages with `source` "
        "(file name), `title`, `score` and `content`. Cite the `source` in your answer."
    ),
    tags=["rag"],
)
async def search_docs(query: str, limit: int = 4) -> str:
    results = await get_store().asearch(KB_CONFIG, query, limit=limit)
    return json.dumps(format_hits(results))
