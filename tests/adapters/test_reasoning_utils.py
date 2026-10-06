"""Tests for shared reasoning tag extraction utilities."""

from tenxgraph.runtime.adapters.llm.reasoning_utils import (
    parse_reasoning_tags,
    parse_think_tags,
    parse_thought_tags,
)


class TestReasoningTagParsers:
    """Test parse_think_tags, parse_reasoning_tags, parse_thought_tags."""

    # -- <think> tags (DeepSeek-R1, Qwen-thinking) --

    def test_parse_think_tags_basic(self):
        clean, reasoning = parse_think_tags(
            "<think>Step 1: analyse</think>The answer is 42."
        )
        assert clean == "The answer is 42."
        assert reasoning == "Step 1: analyse"

    def test_parse_think_tags_multiple(self):
        clean, reasoning = parse_think_tags(
            "<think>First</think>Middle<think>Second</think>End"
        )
        assert "First" in reasoning
        assert "Second" in reasoning
        assert "<think>" not in clean

    def test_parse_think_tags_multiline(self):
        clean, reasoning = parse_think_tags(
            "<think>\nLine 1\nLine 2\n</think>Answer"
        )
        assert "Line 1" in reasoning
        assert "Line 2" in reasoning
        assert clean == "Answer"

    def test_parse_think_tags_no_match(self):
        clean, reasoning = parse_think_tags("No tags here")
        assert clean == "No tags here"
        assert reasoning == ""

    # -- <reasoning> tags --

    def test_parse_reasoning_tags_basic(self):
        clean, reasoning = parse_reasoning_tags(
            "<reasoning>Deep thought</reasoning> Result."
        )
        assert "Deep thought" in reasoning
        assert "<reasoning>" not in clean
        assert "Result." in clean

    def test_parse_reasoning_tags_empty_tag(self):
        clean, reasoning = parse_reasoning_tags(
            "<reasoning></reasoning> Hello"
        )
        assert reasoning == ""
        assert "Hello" in clean

    # -- <thought> tags (Gemini via OpenAI-compat) --

    def test_parse_thought_tags_basic(self):
        clean, reasoning = parse_thought_tags(
            "<thought>Hmm interesting</thought>Final answer."
        )
        assert reasoning == "Hmm interesting"
        assert clean == "Final answer."

    def test_parse_thought_tags_no_match(self):
        clean, reasoning = parse_thought_tags("Plain text")
        assert clean == "Plain text"
        assert reasoning == ""
