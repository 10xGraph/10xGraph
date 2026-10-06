"""SummaryContextManager — LLM-backed context summarization with token-budget support.

When the conversation context grows beyond ``max_messages`` or exceeds the
``token_budget`` estimate, the oldest messages are summarised by an LLM call
and replaced with a concise text stored in ``state.context_summary``.  The
most recent ``keep_recent`` messages are kept verbatim so the agent always has
immediate context.

``convert_messages`` already injects ``state.context_summary`` as an assistant
message before the retained context, so no changes to the execution path are
required.

Provider auto-detection follows the same rules as the rest of the framework:
``gemini-*`` / ``imagen-*`` → Google GenAI; ``gpt-*`` / ``o1-*`` etc. → OpenAI.
Any model string supported by ``detect_provider`` works here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, TypeVar

from agentflow.core.llm.caller import call_llm
from agentflow.core.llm.client_factory import detect_provider
from agentflow.core.state.agent_state import AgentState
from agentflow.core.state.base_context import BaseContextManager
from agentflow.core.state.message import Message
from agentflow.core.state.reducers import remove_tool_messages


S = TypeVar("S", bound=AgentState)

logger = logging.getLogger("agentflow.state.summary")

_DEFAULT_SUMMARY_PROMPT = (
    "You are a conversation summarizer. "
    "Summarize the following conversation history concisely, preserving all important facts, "
    "decisions, tool results, and context needed to continue the conversation. "
    "Write in third-person past tense. Be factual and specific. "
    "Do not add commentary or explanations about the summary itself."
)


def _estimate_tokens(messages: list[Message]) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    total_chars = 0
    for msg in messages:
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                total_chars += len(text)
        if msg.tools_calls:
            total_chars += len(str(msg.tools_calls))
    return max(1, total_chars // 4)


def _messages_to_text(messages: list[Message]) -> str:
    """Render messages as a readable block for the summarizer."""
    parts: list[str] = []
    for msg in messages:
        role = msg.role.upper()
        text_parts: list[str] = []
        for block in msg.content:
            t = getattr(block, "text", None)
            if t:
                text_parts.append(t.strip())
        if msg.tools_calls:
            text_parts.append(f"[Tool calls: {msg.tools_calls}]")
        if text_parts:
            parts.append(f"{role}: {' '.join(text_parts)}")
    return "\n".join(parts)


class SummaryContextManager(BaseContextManager[S]):
    """Context manager that compresses old messages into an LLM-generated summary.

    Summarisation is triggered when *either* threshold is exceeded:

    * ``max_messages``: total message count in ``state.context``
    * ``token_budget``: estimated token count of the context

    After summarisation the oldest messages are removed and the generated text
    is stored in ``state.context_summary``.  Subsequent summarisations append
    to the existing summary so no historical information is permanently lost.

    Args:
        model: Model identifier for summarisation (e.g. ``"gemini-2.0-flash"``,
            ``"gpt-4o-mini"``).  Provider is auto-detected from the name.
        max_messages: Trigger summarisation when message count exceeds this.
            ``None`` disables the count-based trigger.
        token_budget: Trigger summarisation when estimated token count exceeds
            this.  ``None`` disables the token-budget trigger.
        keep_recent: Number of most-recent non-system messages to retain
            verbatim after summarisation.
        remove_tool_msgs: When ``True``, strip tool-call and tool-result messages
            from the context before checking thresholds and before summarising.
            Mirrors the behaviour of ``MessageContextManager``.
        summary_system_prompt: Override the default summarisation instruction
            sent to the LLM.
        max_summary_tokens: Upper bound on the summary output length (tokens).
        api_style: OpenAI only. ``"responses"`` (default) uses the Responses API.
            Use ``"chat"`` for models that only support the legacy Chat Completions
            endpoint (e.g. older or third-party-hosted Chinese models).

    Example::

        from agentflow.core.state import SummaryContextManager

        manager = SummaryContextManager(
            model="gemini-2.0-flash",
            token_budget=6000,
            keep_recent=6,
        )
        app = agent.compile(context_manager=manager)

        # Or trigger on message count instead:
        manager = SummaryContextManager(
            model="gpt-4o-mini",
            max_messages=30,
            token_budget=8000,  # either threshold fires summarisation
            keep_recent=8,
        )
    """

    def __init__(
        self,
        model: str,
        *,
        max_messages: int | None = 30,
        token_budget: int | None = None,
        keep_recent: int = 8,
        remove_tool_msgs: bool = False,
        summary_system_prompt: str | None = None,
        max_summary_tokens: int = 600,
        api_style: Literal["responses", "chat"] = "responses",
    ) -> None:
        self.model = model
        self.max_messages = max_messages
        self.token_budget = token_budget
        self.keep_recent = keep_recent
        self.remove_tool_msgs = remove_tool_msgs
        self.summary_system_prompt = summary_system_prompt or _DEFAULT_SUMMARY_PROMPT
        self.max_summary_tokens = max_summary_tokens
        self.api_style = api_style

        self._provider: str = detect_provider(model)

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def _should_summarize(self, messages: list[Message]) -> bool:
        """Return True if any configured threshold is exceeded."""
        if self.max_messages is not None and len(messages) > self.max_messages:
            logger.debug(
                "Summarisation triggered by message count: %d > %d",
                len(messages),
                self.max_messages,
            )
            return True
        if self.token_budget is not None:
            estimated = _estimate_tokens(messages)
            if estimated > self.token_budget:
                logger.debug(
                    "Summarisation triggered by token budget: ~%d tokens > %d budget",
                    estimated,
                    self.token_budget,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Context split
    # ------------------------------------------------------------------

    def _split_context(self, messages: list[Message]) -> tuple[list[Message], list[Message]]:
        """Split into (messages_to_summarise, messages_to_keep).

        System messages are always kept and never summarised.
        The most recent ``keep_recent`` non-system messages are kept verbatim.
        """
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if len(non_system) <= self.keep_recent:
            return [], messages

        split_at = len(non_system) - self.keep_recent
        to_summarize = non_system[:split_at]
        to_keep = non_system[split_at:]
        return to_summarize, system_msgs + to_keep

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _summarize(self, messages: list[Message]) -> str:
        text = _messages_to_text(messages)
        if not text.strip():
            return ""
        summary, *_ = await call_llm(
            self.model,
            f"Conversation to summarize:\n\n{text}",
            system_prompt=self.summary_system_prompt,
            max_tokens=self.max_summary_tokens,
            api_style=self.api_style,
        )
        return summary

    # ------------------------------------------------------------------
    # BaseContextManager interface
    # ------------------------------------------------------------------

    async def atrim_context(self, state: S) -> S:
        messages = state.context
        if not messages:
            return state

        if self.remove_tool_msgs:
            messages = remove_tool_messages(messages)
            logger.debug("Removed tool messages; %d messages remaining", len(messages))

        if not self._should_summarize(messages):
            # If tool messages were stripped, still commit the cleaned list
            if self.remove_tool_msgs:
                state.context = messages
            return state

        to_summarize, remaining = self._split_context(messages)
        if not to_summarize:
            return state

        logger.debug(
            "Summarising %d messages; keeping %d recent (provider=%s, model=%s)",
            len(to_summarize),
            len(remaining),
            self._provider,
            self.model,
        )

        try:
            new_summary = await self._summarize(to_summarize)
        except Exception:
            logger.exception("Summarisation LLM call failed; leaving context unchanged")
            return state

        if not new_summary:
            return state

        # Rolling summary: append to any previously stored summary
        if state.context_summary:
            state.context_summary = state.context_summary + "\n\n" + new_summary
        else:
            state.context_summary = new_summary

        state.context = remaining
        logger.debug(
            "Context reduced to %d messages; cumulative summary length=%d chars",
            len(remaining),
            len(state.context_summary),
        )
        return state

    def trim_context(self, state: S) -> S:
        return asyncio.run(self.atrim_context(state))
