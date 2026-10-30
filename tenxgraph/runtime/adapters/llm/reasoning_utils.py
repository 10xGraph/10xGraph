"""Shared utilities for reasoning extraction across LLM converters.

Provides helpers for extracting reasoning content from model responses,
including XML-tag-based reasoning (``<think>…</think>``, ``<reasoning>…</reasoning>``)
used by DeepSeek-R1, Qwen-thinking, and similar models.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------


def _extract_tag(text: str, open_tag: str, close_tag: str) -> tuple[str, str]:
    """Strip every ``open_tag ... close_tag`` pair from *text* and collect the contents.

    Each opening tag pairs with the nearest closing tag after it, like a lazy
    ``open(.*?)close`` regex with DOTALL. A ``str.find`` scan runs in linear time, where
    that regex is quadratic on model output full of unclosed tags.
    """
    contents: list[str] = []
    kept: list[str] = []
    pos = 0
    while (start := text.find(open_tag, pos)) != -1:
        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        kept.append(text[pos:start])
        contents.append(text[start + len(open_tag) : end])
        pos = end + len(close_tag)
    if not contents:
        return text, ""
    kept.append(text[pos:])
    reasoning = "\n".join(m.strip() for m in contents)
    return "".join(kept).strip(), reasoning


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_reasoning_tags(text: str) -> tuple[str, str]:
    """Extract ``<reasoning>`` blocks from *text*.

    Returns ``(clean_text, reasoning)`` where *clean_text* has the tags
    stripped out and *reasoning* is the concatenation of all tag contents.
    """
    return _extract_tag(text, "<reasoning>", "</reasoning>")


def parse_think_tags(text: str) -> tuple[str, str]:
    """Extract ``<think>`` blocks from *text*.

    Models like DeepSeek-R1, Qwen-thinking, and similar models embed
    their chain-of-thought reasoning inside ``<think>…</think>`` tags
    within the regular content field.

    Returns ``(clean_text, reasoning)`` where *clean_text* has the tags
    stripped out and *reasoning* is the concatenation of all tag contents.
    """
    return _extract_tag(text, "<think>", "</think>")


def parse_thought_tags(text: str) -> tuple[str, str]:
    """Extract ``<thought>`` blocks from *text*.

    Google Gemini models (via OpenAI-compatible endpoint with
    ``include_thoughts=True``) embed their chain-of-thought reasoning
    inside ``<thought>…</thought>`` tags within the regular content field.

    Returns ``(clean_text, reasoning)`` where *clean_text* has the tags
    stripped out and *reasoning* is the concatenation of all tag contents.
    """
    return _extract_tag(text, "<thought>", "</thought>")


# ---------------------------------------------------------------------------
# Block ordering
# ---------------------------------------------------------------------------

_BLOCK_RANK = {"reasoning": 0, "text": 1, "tool_call": 3, "remote_tool_call": 3}


def order_blocks(blocks: list[Any]) -> list[Any]:
    """Return *blocks* ordered reasoning, text, media, then tool calls.

    Every converter runs its output through this so a message has one block order
    whatever the provider sent. The sort is stable, so blocks of the same kind keep
    their relative order.
    """
    return sorted(blocks, key=lambda b: _BLOCK_RANK.get(getattr(b, "type", ""), 2))
