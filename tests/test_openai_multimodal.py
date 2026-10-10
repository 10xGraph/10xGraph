"""Tests for OpenAI multimodal support across agent_internal and converters."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# _to_responses_content (agent_internal/openai.py)
# ---------------------------------------------------------------------------
from tenxgraph.core.graph.agent_internal.openai import _to_responses_content


class TestToResponsesContent:
    """Test conversion from Chat Completions content format to Responses API format."""

    def test_string_passthrough(self):
        assert _to_responses_content("hello world") == "hello world"

    def test_empty_string(self):
        assert _to_responses_content("") == ""

    def test_none_passthrough(self):
        assert _to_responses_content(None) is None

    def test_text_part(self):
        parts = [{"type": "text", "text": "What's in this image?"}]
        result = _to_responses_content(parts)
        assert result == [{"type": "input_text", "text": "What's in this image?"}]

    def test_image_url_part_nested(self):
        """image_url with nested dict format (Chat Completions style)."""
        parts = [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
        ]
        result = _to_responses_content(parts)
        assert result == [
            {"type": "input_image", "image_url": "https://example.com/img.jpg"}
        ]

    def test_image_url_part_base64(self):
        """image_url with base64 data URL."""
        b64_url = "data:image/png;base64,iVBORw0KGgo="
        parts = [{"type": "image_url", "image_url": {"url": b64_url}}]
        result = _to_responses_content(parts)
        assert result == [{"type": "input_image", "image_url": b64_url}]

    def test_audio_part(self):
        parts = [
            {
                "type": "input_audio",
                "input_audio": {"data": "base64audiodata", "format": "mp3"},
            }
        ]
        result = _to_responses_content(parts)
        assert result == [
            {"type": "input_audio", "data": "base64audiodata", "format": "mp3"}
        ]

    def test_audio_part_default_format(self):
        parts = [
            {"type": "input_audio", "input_audio": {"data": "audiodata"}}
        ]
        result = _to_responses_content(parts)
        assert result[0]["format"] == "wav"

    def test_mixed_multimodal(self):
        parts = [
            {"type": "text", "text": "Describe this:"},
            {"type": "image_url", "image_url": {"url": "https://img.com/cat.jpg"}},
            {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
        ]
        result = _to_responses_content(parts)
        assert len(result) == 3
        assert result[0] == {"type": "input_text", "text": "Describe this:"}
        assert result[1] == {"type": "input_image", "image_url": "https://img.com/cat.jpg"}
        assert result[2] == {"type": "input_audio", "data": "abc", "format": "wav"}

    def test_unknown_type_passthrough(self):
        """Unknown content part types should pass through unchanged."""
        parts = [{"type": "custom_block", "data": "something"}]
        result = _to_responses_content(parts)
        assert result == [{"type": "custom_block", "data": "something"}]

    def test_empty_list(self):
        assert _to_responses_content([]) == []


# ---------------------------------------------------------------------------
# OpenAI Responses Converter — multimodal output handling
# ---------------------------------------------------------------------------
from tenxgraph.runtime.adapters.llm.openai_responses_converter import OpenAIResponsesConverter
from tenxgraph.core.state.message_block import AudioBlock, ImageBlock, MediaRef


class TestResponsesConverterMultimodal:
    """Test multimodal output handling in OpenAIResponsesConverter."""

    @pytest.fixture()
    def converter(self):
        return OpenAIResponsesConverter()

    # ---- _extract_media_from_message_item --------------------------------

    def test_extract_output_image_from_data(self, converter):
        item = SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(type="output_image", image_url=None, image_data="iVBORw0="),
            ],
        )
        blocks = converter._extract_media_from_message_item(item)
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].media.kind == "data"
        assert blocks[0].media.data_base64 == "iVBORw0="

    def test_extract_output_image_from_url(self, converter):
        item = SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_image",
                    image_url="https://cdn.example.com/img.png",
                    image_data=None,
                ),
            ],
        )
        blocks = converter._extract_media_from_message_item(item)
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].media.kind == "url"
        assert blocks[0].media.url == "https://cdn.example.com/img.png"

    def test_extract_output_audio(self, converter):
        item = SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_audio",
                    data="base64audio",
                    transcript="Hello there",
                ),
            ],
        )
        blocks = converter._extract_media_from_message_item(item)
        assert len(blocks) == 1
        assert isinstance(blocks[0], AudioBlock)
        assert blocks[0].media.data_base64 == "base64audio"
        assert blocks[0].transcript == "Hello there"

    def test_extract_mixed_media(self, converter):
        item = SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(type="output_text", text="Here's the image:"),
                SimpleNamespace(type="output_image", image_url=None, image_data="imgdata"),
                SimpleNamespace(type="output_audio", data="audiodata", transcript=None),
            ],
        )
        blocks = converter._extract_media_from_message_item(item)
        # output_text is NOT extracted by this method (handled separately)
        assert len(blocks) == 2
        assert isinstance(blocks[0], ImageBlock)
        assert isinstance(blocks[1], AudioBlock)

    def test_extract_media_dict_format(self, converter):
        """Test with dict-based content (not SimpleNamespace)."""
        item = SimpleNamespace(
            type="message",
            content=[
                {"type": "output_image", "image_data": "b64img", "image_url": None},
            ],
        )
        blocks = converter._extract_media_from_message_item(item)
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].media.data_base64 == "b64img"

    def test_extract_no_media(self, converter):
        item = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="Just text")],
        )
        blocks = converter._extract_media_from_message_item(item)
        assert blocks == []

    def test_extract_media_empty_content(self, converter):
        item = SimpleNamespace(type="message", content=[])
        assert converter._extract_media_from_message_item(item) == []

    def test_extract_media_none_content(self, converter):
        item = SimpleNamespace(type="message", content=None)
        assert converter._extract_media_from_message_item(item) == []

    # ---- _extract_image_generation ---------------------------------------

    def test_extract_image_generation_string_result(self, converter):
        item = SimpleNamespace(type="image_generation_call", result="base64imagedata")
        block = converter._extract_image_generation(item)
        assert block is not None
        assert isinstance(block, ImageBlock)
        assert block.media.data_base64 == "base64imagedata"

    def test_extract_image_generation_object_result_b64(self, converter):
        item = SimpleNamespace(
            type="image_generation_call",
            result=SimpleNamespace(b64_json="b64imgdata", data=None),
        )
        block = converter._extract_image_generation(item)
        assert block is not None
        assert block.media.data_base64 == "b64imgdata"

    def test_extract_image_generation_object_result_data(self, converter):
        item = SimpleNamespace(
            type="image_generation_call",
            result=SimpleNamespace(b64_json=None, data="rawimgdata"),
        )
        block = converter._extract_image_generation(item)
        assert block is not None
        assert block.media.data_base64 == "rawimgdata"

    def test_extract_image_generation_dict_result(self, converter):
        item = SimpleNamespace(
            type="image_generation_call",
            result={"b64_json": "dictb64data"},
        )
        block = converter._extract_image_generation(item)
        assert block is not None
        assert block.media.data_base64 == "dictb64data"

    def test_extract_image_generation_empty_result(self, converter):
        item = SimpleNamespace(type="image_generation_call", result=None)
        block = converter._extract_image_generation(item)
        assert block is None

    def test_extract_image_generation_empty_string(self, converter):
        item = SimpleNamespace(type="image_generation_call", result="")
        block = converter._extract_image_generation(item)
        assert block is None

    # ---- convert_response with multimodal output -------------------------

    @pytest.mark.asyncio
    async def test_convert_response_with_image_output(self, converter):
        """Full convert_response with image in message content."""
        response = SimpleNamespace(
            id="resp-123",
            model="gpt-4o",
            status="completed",
            created_at=1234567890,
            usage=None,
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="Here is the cat:"),
                        SimpleNamespace(
                            type="output_image",
                            image_url="https://cdn.example.com/cat.png",
                            image_data=None,
                        ),
                    ],
                ),
            ],
        )
        msg = await converter.convert_response(response)
        assert len(msg.content) == 2
        assert msg.text() == "Here is the cat:"
        img_blocks = [b for b in msg.content if isinstance(b, ImageBlock)]
        assert len(img_blocks) == 1
        assert img_blocks[0].media.url == "https://cdn.example.com/cat.png"

    @pytest.mark.asyncio
    async def test_convert_response_with_image_generation(self, converter):
        """Full convert_response with image_generation_call item."""
        response = SimpleNamespace(
            id="resp-456",
            model="gpt-4o",
            status="completed",
            created_at=1234567890,
            usage=None,
            output=[
                SimpleNamespace(type="image_generation_call", result="base64catimage"),
            ],
        )
        msg = await converter.convert_response(response)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ImageBlock)
        assert msg.content[0].media.data_base64 == "base64catimage"

    @pytest.mark.asyncio
    async def test_convert_response_mixed_text_and_media(self, converter):
        """Response with reasoning + text + image + audio."""
        response = SimpleNamespace(
            id="resp-789",
            model="gpt-4o",
            status="completed",
            created_at=1234567890,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                input_tokens_details=None,
                output_tokens_details=None,
            ),
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="Thinking about the image...")],
                ),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="I see a landscape."),
                        SimpleNamespace(
                            type="output_image",
                            image_url=None,
                            image_data="landscapeimg",
                        ),
                        SimpleNamespace(
                            type="output_audio",
                            data="audionarration",
                            transcript="This is a beautiful landscape.",
                        ),
                    ],
                ),
            ],
        )
        msg = await converter.convert_response(response)
        # Should have: ReasoningBlock + TextBlock + ImageBlock + AudioBlock
        from tenxgraph.core.state.message_block import ReasoningBlock, TextBlock

        block_types = [type(b).__name__ for b in msg.content]
        assert "ReasoningBlock" in block_types
        assert "TextBlock" in block_types
        assert "ImageBlock" in block_types
        assert "AudioBlock" in block_types
        assert msg.usages.prompt_tokens == 100
        assert msg.usages.completion_tokens == 50


# ---------------------------------------------------------------------------
# OpenAI converter — ensure response image/audio extraction works
# ---------------------------------------------------------------------------
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter


class TestOpenAIConverterMultimodal:
    """Verify OpenAI Chat Completions converter handles media responses."""

    @pytest.fixture()
    def converter(self):
        return OpenAIConverter()

    def test_extract_image_blocks_from_list(self, converter):
        images = [
            SimpleNamespace(url="https://cdn.example.com/img1.png"),
            SimpleNamespace(url="https://cdn.example.com/img2.png"),
        ]
        blocks = converter._extract_image_blocks(images)
        assert len(blocks) == 2
        assert all(isinstance(b, ImageBlock) for b in blocks)
        assert blocks[0].media.url == "https://cdn.example.com/img1.png"
        assert blocks[1].media.url == "https://cdn.example.com/img2.png"

    def test_extract_image_blocks_from_dict(self, converter):
        images = [{"url": "https://cdn.example.com/img.png"}]
        blocks = converter._extract_image_blocks(images)
        assert len(blocks) == 1
        assert blocks[0].media.url == "https://cdn.example.com/img.png"

    def test_extract_image_blocks_from_string(self, converter):
        images = ["https://cdn.example.com/img.png"]
        blocks = converter._extract_image_blocks(images)
        assert len(blocks) == 1
        assert blocks[0].media.url == "https://cdn.example.com/img.png"

    def test_extract_audio_block(self, converter):
        audio = SimpleNamespace(data="base64audiodata", transcript="Hello")
        block = converter._extract_audio_block(audio)
        assert block is not None
        assert isinstance(block, AudioBlock)
        assert block.media.data_base64 == "base64audiodata"
        assert block.transcript == "Hello"

    def test_extract_audio_block_dict(self, converter):
        audio = {"data": "audiodata", "transcript": "Hi"}
        block = converter._extract_audio_block(audio)
        assert block is not None
        assert block.media.data_base64 == "audiodata"
        assert block.transcript == "Hi"


# ---------------------------------------------------------------------------
# Google converter — ensure inline_data / file_data extraction works
# ---------------------------------------------------------------------------
from tenxgraph.runtime.adapters.llm.google_genai_converter import GoogleGenAIConverter


class TestGoogleConverterMultimodal:
    """Verify Google GenAI converter handles media in response parts."""

    @pytest.fixture()
    def converter(self):
        return GoogleGenAIConverter()

    def test_process_inline_image(self, converter):
        part = SimpleNamespace(
            text=None,
            thought=False,
            function_call=None,
            inline_data=SimpleNamespace(data="imgbytes", mime_type="image/png"),
            file_data=None,
        )
        blocks: list = []
        converter._process_inline_media_part(part, blocks)
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].media.data_base64 == "imgbytes"
        assert blocks[0].media.mime_type == "image/png"

    def test_process_inline_audio(self, converter):
        part = SimpleNamespace(
            text=None,
            thought=False,
            function_call=None,
            inline_data=SimpleNamespace(data="audiodata", mime_type="audio/wav"),
            file_data=None,
        )
        blocks: list = []
        converter._process_inline_media_part(part, blocks)
        assert len(blocks) == 1
        assert isinstance(blocks[0], AudioBlock)
        assert blocks[0].media.data_base64 == "audiodata"

    def test_process_file_data_image(self, converter):
        part = SimpleNamespace(
            text=None,
            thought=False,
            function_call=None,
            inline_data=None,
            file_data=SimpleNamespace(
                file_uri="gs://bucket/image.png", mime_type="image/png"
            ),
        )
        blocks: list = []
        converter._process_file_media_part(part, blocks)
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].media.url == "gs://bucket/image.png"

    def test_process_no_media(self, converter):
        part = SimpleNamespace(
            text="just text",
            thought=False,
            function_call=None,
            inline_data=None,
            file_data=None,
        )
        blocks: list = []
        converter._process_inline_media_part(part, blocks)
        converter._process_file_media_part(part, blocks)
        assert blocks == []


# ---------------------------------------------------------------------------
# converter.py — multimodal message conversion
# ---------------------------------------------------------------------------
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import DocumentBlock, TextBlock
from tenxgraph.utils.converter import (
    _build_content,
    _convert_dict,
    _has_multimodal_blocks,
)


class TestConverterMultimodal:
    """Verify converter.py handles multimodal content correctly."""

    def test_text_only_returns_string(self):
        blocks = [TextBlock(text="hello")]
        assert _build_content(blocks) == "hello"

    def test_image_block_returns_list(self):
        blocks = [
            TextBlock(text="What is this?"),
            ImageBlock(media=MediaRef(kind="url", url="https://img.com/cat.jpg")),
        ]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "What is this?"}
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "https://img.com/cat.jpg"

    def test_convert_dict_multimodal(self):
        msg = Message(
            role="user",
            content=[
                TextBlock(text="Describe:"),
                ImageBlock(
                    media=MediaRef(
                        kind="data",
                        data_base64="iVBORw0=",
                        mime_type="image/png",
                    )
                ),
            ],
        )
        d = _convert_dict(msg)
        assert d is not None
        assert d["role"] == "user"
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 2
        assert d["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_convert_dict_text_only_backward_compat(self):
        msg = Message(role="user", content=[TextBlock(text="just text")])
        d = _convert_dict(msg)
        assert d is not None
        assert isinstance(d["content"], str)
        assert d["content"] == "just text"

    def test_has_multimodal_blocks_true(self):
        blocks = [TextBlock(text="hi"), ImageBlock(media=MediaRef(kind="url", url="x"))]
        assert _has_multimodal_blocks(blocks) is True

    def test_has_multimodal_blocks_false(self):
        blocks = [TextBlock(text="hi")]
        assert _has_multimodal_blocks(blocks) is False

    def test_document_block_with_excerpt(self):
        blocks = [
            DocumentBlock(
                media=MediaRef(kind="data", data_base64="pdf", mime_type="application/pdf"),
                excerpt="Extracted PDF text",
            ),
        ]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "Extracted PDF text"}


# ---------------------------------------------------------------------------
# Google mixin — multimodal message handling
# ---------------------------------------------------------------------------
from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin


class TestGoogleMixinMultimodal:
    """Verify _handle_regular_message converts multimodal content."""

    @pytest.fixture()
    def mixin(self):
        return AgentGoogleMixin()

    def test_string_content(self, mixin):
        result = mixin._handle_regular_message("Hello", "user")
        assert result.role == "user"
        assert result.parts[0].text == "Hello"

    def test_list_content_text_part(self, mixin):
        content = [{"type": "text", "text": "Describe this image"}]
        result = mixin._handle_regular_message(content, "user")
        assert len(result.parts) == 1
        assert result.parts[0].text == "Describe this image"

    def test_list_content_image_url(self, mixin):
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        result = mixin._handle_regular_message(content, "user")
        assert len(result.parts) == 2

    def test_list_content_base64_image(self, mixin):
        import base64

        b64 = base64.b64encode(b"\x89PNG").decode()
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        ]
        result = mixin._handle_regular_message(content, "user")
        assert len(result.parts) == 1

    def test_empty_list_fallback(self, mixin):
        result = mixin._handle_regular_message([], "user")
        assert len(result.parts) == 1
        assert result.parts[0].text == ""
