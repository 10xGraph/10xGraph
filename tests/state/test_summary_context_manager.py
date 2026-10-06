"""Tests for SummaryContextManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import TextBlock, ToolCallBlock, ToolResultBlock
from tenxgraph.core.state.summary_context_manager import (
    SummaryContextManager,
    _estimate_tokens,
    _messages_to_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msgs(*roles_and_texts: tuple[str, str]) -> list[Message]:
    return [Message.text_message(text, role=role) for role, text in roles_and_texts]


def _conv(n_user: int, include_system: bool = True) -> list[Message]:
    msgs: list[Message] = []
    if include_system:
        msgs.append(Message.text_message("System prompt", role="system"))
    for i in range(n_user):
        msgs.append(Message.text_message(f"User message {i + 1}", role="user"))
        msgs.append(Message.text_message(f"Assistant reply {i + 1}", role="assistant"))
    return msgs


# ---------------------------------------------------------------------------
# Unit tests — no I/O
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens([]) == 1

    def test_basic(self):
        msgs = _make_msgs(("user", "hello world"))  # 11 chars → 2 tokens
        assert _estimate_tokens(msgs) >= 1

    def test_longer_text(self):
        msgs = _make_msgs(("user", "a" * 400))  # 400 chars → 100 tokens
        assert _estimate_tokens(msgs) == 100

    def test_tool_calls_counted(self):
        msg = Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="search", args={"q": "x"})],
            tools_calls=[{"id": "c1", "name": "search", "args": {"q": "x"}}],
        )
        assert _estimate_tokens([msg]) >= 1


class TestMessagesToText:
    def test_basic_roles(self):
        msgs = _make_msgs(("user", "hi"), ("assistant", "hello"))
        text = _messages_to_text(msgs)
        assert "USER: hi" in text
        assert "ASSISTANT: hello" in text

    def test_empty_messages_skipped(self):
        msgs = [Message(role="user", content=[])]
        assert _messages_to_text(msgs) == ""

    def test_tool_calls_included(self):
        msg = Message(
            role="assistant",
            content=[],
            tools_calls=[{"name": "search"}],
        )
        text = _messages_to_text([msg])
        assert "Tool calls" in text


class TestShouldSummarize:
    def test_message_count_trigger(self):
        mgr = SummaryContextManager("gpt-4o-mini", max_messages=5, token_budget=None)
        msgs = _conv(3)  # 1 system + 6 msgs = 7 total
        assert mgr._should_summarize(msgs) is True

    def test_message_count_no_trigger(self):
        mgr = SummaryContextManager("gpt-4o-mini", max_messages=20, token_budget=None)
        msgs = _conv(3)
        assert mgr._should_summarize(msgs) is False

    def test_token_budget_trigger(self):
        mgr = SummaryContextManager("gpt-4o-mini", max_messages=None, token_budget=1)
        msgs = _make_msgs(("user", "hello world"))
        assert mgr._should_summarize(msgs) is True

    def test_token_budget_no_trigger(self):
        mgr = SummaryContextManager("gpt-4o-mini", max_messages=None, token_budget=99999)
        msgs = _conv(2)
        assert mgr._should_summarize(msgs) is False

    def test_either_trigger_fires(self):
        mgr = SummaryContextManager("gpt-4o-mini", max_messages=100, token_budget=1)
        msgs = _make_msgs(("user", "a" * 100))  # ~25 tokens > budget of 1
        assert mgr._should_summarize(msgs) is True


class TestSplitContext:
    def test_keeps_system_messages(self):
        msgs = _conv(5)  # system + 10 msgs
        mgr = SummaryContextManager("gpt-4o-mini", keep_recent=4)
        to_summarize, remaining = mgr._split_context(msgs)
        assert all(m.role == "system" for m in remaining if m.role == "system")
        assert not any(m.role == "system" for m in to_summarize)

    def test_keep_recent_respected(self):
        msgs = _conv(6, include_system=False)  # 12 messages
        mgr = SummaryContextManager("gpt-4o-mini", keep_recent=4)
        to_summarize, remaining = mgr._split_context(msgs)
        assert len(remaining) == 4
        assert len(to_summarize) == 8

    def test_no_split_when_small(self):
        msgs = _conv(2)  # 5 total (system + 4)
        mgr = SummaryContextManager("gpt-4o-mini", keep_recent=8)
        to_summarize, remaining = mgr._split_context(msgs)
        assert to_summarize == []
        assert len(remaining) == len(msgs)


# ---------------------------------------------------------------------------
# Async tests — mocked LLM
# ---------------------------------------------------------------------------

_CALL_LLM = "tenxgraph.core.state.summary_context_manager.call_llm"


@pytest.mark.anyio(loop_scope="function")
async def test_atrim_context_openai():
    mgr = SummaryContextManager("gpt-4o-mini", max_messages=5, keep_recent=2)
    state = AgentState(context=_conv(4))  # 9 messages → triggers max_messages=5

    with patch(_CALL_LLM, new=AsyncMock(return_value=("Summary text.", 10, 5, 0))) as mock_call:
        result = await mgr.atrim_context(state)

    assert result.context_summary == "Summary text."
    assert len(result.context) <= len(_conv(4))
    mock_call.assert_called_once()


@pytest.mark.anyio
async def test_atrim_context_google():
    mgr = SummaryContextManager("gemini-2.0-flash", max_messages=5, keep_recent=2)
    state = AgentState(context=_conv(4))

    with patch(_CALL_LLM, new=AsyncMock(return_value=("Google summary.", 8, 4, 0))) as mock_call:
        result = await mgr.atrim_context(state)

    assert result.context_summary == "Google summary."
    mock_call.assert_called_once()


@pytest.mark.anyio
async def test_rolling_summary_appended():
    """Subsequent summarisations append to existing context_summary."""
    mgr = SummaryContextManager("gpt-4o-mini", max_messages=5, keep_recent=2)
    state = AgentState(context=_conv(4), context_summary="Previous summary.")

    with patch(_CALL_LLM, new=AsyncMock(return_value=("New chunk.", 10, 5, 0))):
        result = await mgr.atrim_context(state)

    assert result.context_summary == "Previous summary.\n\nNew chunk."


@pytest.mark.anyio
async def test_no_summarize_when_below_threshold():
    mgr = SummaryContextManager("gpt-4o-mini", max_messages=100, token_budget=None)
    state = AgentState(context=_conv(2))
    original_context = list(state.context)

    with patch(_CALL_LLM, new=AsyncMock()) as mock_call:
        result = await mgr.atrim_context(state)

    mock_call.assert_not_called()
    assert result.context == original_context
    assert result.context_summary is None


@pytest.mark.anyio
async def test_llm_failure_leaves_context_unchanged():
    """If call_llm raises, context must not be modified."""
    mgr = SummaryContextManager("gpt-4o-mini", max_messages=5, keep_recent=2)
    state = AgentState(context=_conv(4))
    original_len = len(state.context)

    with patch(_CALL_LLM, new=AsyncMock(side_effect=RuntimeError("API down"))):
        result = await mgr.atrim_context(state)

    assert len(result.context) == original_len
    assert result.context_summary is None


@pytest.mark.anyio
async def test_token_budget_triggers_with_long_messages():
    long_text = "x" * 4000  # ~1000 tokens
    msgs = [Message.text_message(long_text, role="user")]
    mgr = SummaryContextManager("gpt-4o-mini", max_messages=None, token_budget=500, keep_recent=0)
    state = AgentState(context=msgs)

    with patch(_CALL_LLM, new=AsyncMock(return_value=("Compressed.", 20, 10, 0))):
        result = await mgr.atrim_context(state)

    assert result.context_summary == "Compressed."


# ---------------------------------------------------------------------------
# remove_tool_msgs
# ---------------------------------------------------------------------------

def _complete_tool_sequence(idx: int = 1) -> list[Message]:
    """Build a complete tool sequence: user → ai-with-tool-call → tool-result → ai-final.

    remove_tool_messages only strips COMPLETE sequences, so all four messages are needed.
    """
    tool_call = Message(
        role="assistant",
        content=[ToolCallBlock(id=f"c{idx}", name="search", args={})],
        tools_calls=[{"id": f"c{idx}", "name": "search", "args": {}}],
    )
    tool_result = Message.tool_message([ToolResultBlock(call_id=f"c{idx}", output="result")])
    ai_final = Message.text_message(f"Done {idx}", role="assistant")
    return [tool_call, tool_result, ai_final]


@pytest.mark.anyio
async def test_remove_tool_msgs_strips_before_threshold():
    """Complete tool sequences are stripped; remaining messages stay below threshold."""
    # 1 complete tool sequence (3 msgs) + 2 plain messages = 5 total
    # After stripping: 2 messages — below max_messages=3, so no summarisation
    user1 = Message.text_message("first", role="user")
    user2 = Message.text_message("second", role="user")
    msgs = _complete_tool_sequence(1) + [user1, user2]

    mgr = SummaryContextManager("gpt-4o-mini", max_messages=3, remove_tool_msgs=True)
    state = AgentState(context=msgs)
    result = await mgr.atrim_context(state)

    # Tool messages stripped from retained context
    assert all(m.role != "tool" for m in result.context)
    assert not any(
        isinstance(b, ToolCallBlock)
        for m in result.context
        for b in m.content
    )
    # Threshold not exceeded after stripping — no summary
    assert result.context_summary is None


@pytest.mark.anyio
async def test_remove_tool_msgs_then_summarizes():
    """Complete tool sequences stripped, then summarisation fires if threshold still exceeded."""
    # _conv(4) = system + 8 msgs; add 2 complete tool sequences (6 msgs) = 15 total
    # After stripping 2 sequences (6 msgs): 9 messages > max_messages=5 → summarise
    msgs = _conv(4) + _complete_tool_sequence(1) + _complete_tool_sequence(2)

    mgr = SummaryContextManager(
        "gpt-4o-mini", max_messages=5, keep_recent=2, remove_tool_msgs=True
    )
    state = AgentState(context=msgs)

    with patch(_CALL_LLM, new=AsyncMock(return_value=("Summary.", 10, 5, 0))):
        result = await mgr.atrim_context(state)

    assert result.context_summary == "Summary."
    assert all(m.role != "tool" for m in result.context)


# ---------------------------------------------------------------------------
# Provider auto-detection
# ---------------------------------------------------------------------------

def test_provider_detected_google():
    mgr = SummaryContextManager("gemini-2.0-flash")
    assert mgr._provider == "google"


def test_provider_detected_openai():
    mgr = SummaryContextManager("gpt-4o-mini")
    assert mgr._provider == "openai"


def test_custom_summary_prompt():
    mgr = SummaryContextManager("gpt-4o-mini", summary_system_prompt="Be brief.")
    assert mgr.summary_system_prompt == "Be brief."


def test_default_summary_prompt_set():
    mgr = SummaryContextManager("gpt-4o-mini")
    assert len(mgr.summary_system_prompt) > 0
