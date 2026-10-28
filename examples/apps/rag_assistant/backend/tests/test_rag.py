# ruff: noqa: INP001
"""Backend tests that need no API key, no database and no network."""

from __future__ import annotations

import asyncio
import json

from graph import tools
from graph.store import KB_OWNER
from ingest import chunk_text

from tenxgraph.qa.testing import QuickTest
from tenxgraph.storage.store.store_schema import MemorySearchResult


class FakeStore:
    """Stands in for QdrantStore and records how it was called."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, str, int]] = []

    async def asearch(self, config, query, limit=10, **kwargs):
        self.calls.append((config, query, limit))
        return [
            MemorySearchResult(
                content="Team costs $29 per seat per month.",
                score=0.91234,
                metadata={"source": "pricing.md", "title": "Plans"},
            )
        ]


def test_search_docs_returns_sources_and_uses_kb_owner(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(tools, "get_store", lambda: store)

    raw = asyncio.run(tools.search_docs("team plan price", limit=2))

    hits = json.loads(raw)
    assert hits == [
        {
            "source": "pricing.md",
            "title": "Plans",
            "score": 0.912,
            "content": "Team costs $29 per seat per month.",
        }
    ]
    config, query, limit = store.calls[0]
    assert config["user_id"] == KB_OWNER and KB_OWNER.startswith("__system__:")
    assert (query, limit) == ("team plan price", 2)


def test_query_rewriter_uses_gemini_and_falls_back_on_empty(monkeypatch):
    seen = {}

    async def fake_call_llm(model, prompt, **kwargs):
        seen["model"] = model
        return "  ", 0, 0, 0

    monkeypatch.setattr(tools, "call_llm", fake_call_llm)

    assert asyncio.run(tools.query_rewriter("how much is it?")) == "how much is it?"
    assert seen["model"] == "gemini-2.5-flash"


def test_chunk_text_splits_on_headings():
    text = "# Title\nintro\n\n## One\nfirst\n\n## Two\nsecond\n"
    chunks = chunk_text(text)
    assert [h for h, _ in chunks] == ["Title", "One", "Two"]
    assert "second" in chunks[-1][1]


def test_agent_answer_with_scripted_model():
    # TestAgent replaces the LLM, so this checks graph wiring without any provider.
    result = asyncio.run(
        QuickTest.single_turn(agent_response="Team is $29 per seat.", user_message="Price?")
    )
    result.assert_contains("$29")
    result.assert_no_errors()
