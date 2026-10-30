"""Shape rules that every request to a provider, and every saved history, must follow.

Providers reject (OpenAI, Anthropic) or answer with nothing (Gemini) when a request holds
an assistant tool call with no tool result, or when it ends on an assistant turn. Tests
capture the ``messages`` argument the Agent hands to ``_call_google`` / ``_call_openai`` /
``_call_anthropic`` and check it with :func:`assert_valid_provider_request`. Saved state and
the messages table are checked with :func:`assert_valid_saved_history`.
"""

from __future__ import annotations

from typing import Any

from tenxgraph.core.state import Message
from tenxgraph.core.state.message_block import ToolResultBlock


def _call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or "")
    return str(getattr(call, "id", "") or "")


def provider_request_problems(messages: list[dict[str, Any]]) -> list[str]:
    """Return every rule the request breaks (empty when it is valid).

    Rules, over the non-system messages in order:
    - every assistant tool call is answered by a ``tool`` message with the same id before
      the next assistant or user turn and before the end of the request;
    - a ``tool`` message answers a call of the assistant turn before it (no orphans);
    - the request does not end on an assistant turn.
    """
    problems: list[str] = []
    turns = [m for m in messages if m.get("role") not in ("system", "developer")]
    pending: dict[str, int] = {}

    def close_pending(at: str) -> None:
        for call_id, index in pending.items():
            problems.append(f"tool call {call_id!r} of message {index} unanswered before {at}")
        pending.clear()

    for index, message in enumerate(turns):
        role = message.get("role")
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id in pending:
                del pending[call_id]
            else:
                problems.append(f"tool result {call_id!r} at message {index} answers no open call")
            continue
        close_pending(f"message {index} ({role})")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                pending[_call_id(call)] = index
    close_pending("the end of the request")

    if turns and turns[-1].get("role") == "assistant":
        problems.append("the request ends on an assistant turn")
    if not turns:
        problems.append("the request has no conversation turns")
    return problems


def assert_valid_provider_request(messages: list[dict[str, Any]]) -> None:
    """Assert that a captured provider request follows the rules above."""
    problems = provider_request_problems(messages)
    assert not problems, "invalid provider request:\n- " + "\n- ".join(problems) + (
        "\nroles: " + str([m.get("role") for m in messages])
    )


def saved_history_problems(context: list[Message]) -> list[str]:
    """Return every rule a saved conversation breaks (empty when it is valid).

    Every assistant tool call is answered exactly once by a following tool message before
    the next assistant or user turn; no tool result answers an unknown call; no assistant
    message is empty (no text, no tool calls).
    """
    problems: list[str] = []
    pending: dict[str, int] = {}
    answered: set[str] = set()

    def close_pending(at: str) -> None:
        for call_id, index in pending.items():
            problems.append(f"tool call {call_id!r} of message {index} unanswered before {at}")
        pending.clear()

    for index, message in enumerate(context):
        if message.role == "tool":
            ids = [b.call_id for b in message.content if isinstance(b, ToolResultBlock)]
            for call_id in ids:
                if call_id in answered:
                    problems.append(f"tool call {call_id!r} answered twice (message {index})")
                elif call_id in pending:
                    del pending[call_id]
                    answered.add(call_id)
                else:
                    problems.append(f"tool result {call_id!r} at {index} answers no open call")
            continue
        if message.role == "system":
            continue
        close_pending(f"message {index} ({message.role})")
        if message.role == "assistant":
            if not message.tools_calls and not message.text().strip():
                problems.append(f"empty assistant message at {index}")
            for call in message.tools_calls or []:
                pending[_call_id(call)] = index
    close_pending("the end of the history")
    return problems


def assert_valid_saved_history(context: list[Message]) -> None:
    """Assert that a saved conversation follows the rules above."""
    problems = saved_history_problems(context)
    assert not problems, "invalid saved history:\n- " + "\n- ".join(problems) + (
        "\nroles: " + str([m.role for m in context])
    )
