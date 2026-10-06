"""Shared utilities for reasoning extraction across LLM converters.

Provides helpers for extracting reasoning content from model responses,
including XML-tag-based reasoning (``<think>…</think>``, ``<reasoning>…</reasoning>``)
used by DeepSeek-R1, Qwen-thinking, and similar models.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Regex patterns for reasoning tags
# ---------------------------------------------------------------------------

_REASONING_TAG_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    re.DOTALL,
)

_THINK_TAG_RE = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL,
)

_THOUGHT_TAG_RE = re.compile(
    r"<thought>(.*?)</thought>",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_reasoning_tags(text: str) -> tuple[str, str]:
    """Extract ``<reasoning>`` blocks from *text*.

    Returns ``(clean_text, reasoning)`` where *clean_text* has the tags
    stripped out and *reasoning* is the concatenation of all tag contents.
    """
    matches = _REASONING_TAG_RE.findall(text)
    if not matches:
        return text, ""
    reasoning = "\n".join(m.strip() for m in matches)
    clean = _REASONING_TAG_RE.sub("", text).strip()
    return clean, reasoning


def parse_think_tags(text: str) -> tuple[str, str]:
    """Extract ``<think>`` blocks from *text*.

    Models like DeepSeek-R1, Qwen-thinking, and similar models embed
    their chain-of-thought reasoning inside ``<think>…</think>`` tags
    within the regular content field.

    Returns ``(clean_text, reasoning)`` where *clean_text* has the tags
    stripped out and *reasoning* is the concatenation of all tag contents.
    """
    matches = _THINK_TAG_RE.findall(text)
    if not matches:
        return text, ""
    reasoning = "\n".join(m.strip() for m in matches)
    clean = _THINK_TAG_RE.sub("", text).strip()
    return clean, reasoning


def parse_thought_tags(text: str) -> tuple[str, str]:
    """Extract ``<thought>`` blocks from *text*.

    Google Gemini models (via OpenAI-compatible endpoint with
    ``include_thoughts=True``) embed their chain-of-thought reasoning
    inside ``<thought>…</thought>`` tags within the regular content field.

    Returns ``(clean_text, reasoning)`` where *clean_text* has the tags
    stripped out and *reasoning* is the concatenation of all tag contents.
    """
    matches = _THOUGHT_TAG_RE.findall(text)
    if not matches:
        return text, ""
    reasoning = "\n".join(m.strip() for m in matches)
    clean = _THOUGHT_TAG_RE.sub("", text).strip()
    return clean, reasoning
