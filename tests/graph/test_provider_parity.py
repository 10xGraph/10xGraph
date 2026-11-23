"""Tests for cross-provider parity: count_tokens.

The point of these is that the *surface* is the same across providers even
where the underlying APIs differ, so switching providers does not mean
relearning the interface.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin


class _GoogleStub(AgentGoogleMixin):
    def __init__(self):
        self.model = "gemini-2.5-flash"
        self.llm_kwargs = {}
        self.client = MagicMock()
        self.client.aio.models.count_tokens = AsyncMock(
            return_value=SimpleNamespace(total_tokens=321)
        )


class TestGoogleCountTokens:
    @pytest.mark.asyncio
    async def test_returns_total_tokens(self):
        agent = _GoogleStub()
        assert await agent._count_tokens_google([{"role": "user", "content": "hi"}]) == 321

    @pytest.mark.asyncio
    async def test_counts_against_converted_contents(self):
        """Uses the same conversion the real request does, not the raw dicts."""
        agent = _GoogleStub()
        await agent._count_tokens_google([{"role": "user", "content": "hi"}])
        kwargs = agent.client.aio.models.count_tokens.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["contents"]  # converted Content objects, not plain dicts

    @pytest.mark.asyncio
    async def test_missing_total_returns_zero(self):
        agent = _GoogleStub()
        agent.client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace())
        assert await agent._count_tokens_google([]) == 0
