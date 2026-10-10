"""Comprehensive end-to-end multimodal tests.

Tests the full multimodal pipeline across ALL three providers:
  1. OpenAI Chat Completions
  2. OpenAI Responses API
  3. Google GenAI

Covers:
  - Input: _build_content, _to_responses_content, _content_parts_to_google
  - Output: converter response extraction (non-streaming + streaming)
  - Multi-agent: strip_media_blocks for text-only agents
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.state.message import Message, TokenUsages, generate_id
from tenxgraph.core.state.message_block import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    MediaRef,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    VideoBlock,
)
from tenxgraph.utils.converter import (
    _audio_block_to_openai,
    _build_content,
    _convert_dict,
    _document_block_to_openai,
    _has_multimodal_blocks,
    _image_block_to_openai,
    _video_block_to_openai,
    convert_messages,
    strip_media_blocks,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

_TINY_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10).decode()
_TINY_AUDIO_B64 = base64.b64encode(b"RIFF" + b"\x00" * 10).decode()
_TINY_PDF_B64 = base64.b64encode(b"%PDF-1.4 tiny").decode()
_TINY_VIDEO_B64 = base64.b64encode(b"\x00\x00\x00\x1cftypisom").decode()


def _make_image_block(kind="data", url=None) -> ImageBlock:
    if kind == "data":
        return ImageBlock(media=MediaRef(kind="data", data_base64=_TINY_PNG_B64, mime_type="image/png"))
    return ImageBlock(media=MediaRef(kind="url", url=url or "https://example.com/img.png"))


def _make_audio_block() -> AudioBlock:
    return AudioBlock(
        media=MediaRef(kind="data", data_base64=_TINY_AUDIO_B64, mime_type="audio/wav"),
        transcript="hello",
    )


def _make_doc_block(excerpt=None) -> DocumentBlock:
    return DocumentBlock(
        media=MediaRef(kind="data", data_base64=_TINY_PDF_B64, mime_type="application/pdf"),
        excerpt=excerpt,
    )


def _make_video_block() -> VideoBlock:
    return VideoBlock(
        media=MediaRef(kind="data", data_base64=_TINY_VIDEO_B64, mime_type="video/mp4"),
    )


# =========================================================================
# PART 1: _build_content — Input conversion to OpenAI-style parts
# =========================================================================


class TestBuildContentMultimodal:
    """Test _build_content produces correct content parts for all media types."""

    def test_text_only_returns_string(self):
        blocks = [TextBlock(text="hello"), TextBlock(text=" world")]
        result = _build_content(blocks)
        assert result == "hello world"
        assert isinstance(result, str)

    def test_image_block_base64(self):
        blocks = [TextBlock(text="Look at this:"), _make_image_block()]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Look at this:"}
        assert result[1]["type"] == "image_url"
        assert _TINY_PNG_B64 in result[1]["image_url"]["url"]

    def test_image_block_url(self):
        blocks = [TextBlock(text="See:"), _make_image_block(kind="url")]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "https://example.com/img.png"

    def test_audio_block_base64(self):
        blocks = [TextBlock(text="Listen:"), _make_audio_block()]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert result[1]["type"] == "input_audio"
        assert result[1]["input_audio"]["data"] == _TINY_AUDIO_B64
        assert result[1]["input_audio"]["format"] == "wav"

    def test_document_block_with_excerpt(self):
        blocks = [TextBlock(text="Summary:"), _make_doc_block(excerpt="This is the doc content.")]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert result[1] == {"type": "text", "text": "This is the doc content."}

    def test_document_block_raw_base64(self):
        blocks = [TextBlock(text="Raw doc:"), _make_doc_block()]
        result = _build_content(blocks)
        assert isinstance(result, list)
        doc_part = result[1]
        assert doc_part["type"] == "document"
        assert doc_part["document"]["data"] == _TINY_PDF_B64
        assert doc_part["document"]["mime_type"] == "application/pdf"

    def test_document_block_url(self):
        doc = DocumentBlock(
            media=MediaRef(kind="url", url="https://example.com/doc.pdf", mime_type="application/pdf"),
        )
        blocks = [TextBlock(text="Link:"), doc]
        result = _build_content(blocks)
        assert isinstance(result, list)
        doc_part = result[1]
        assert doc_part["type"] == "document"
        assert doc_part["document"]["url"] == "https://example.com/doc.pdf"

    def test_video_block_base64(self):
        blocks = [TextBlock(text="Watch:"), _make_video_block()]
        result = _build_content(blocks)
        assert isinstance(result, list)
        vid_part = result[1]
        assert vid_part["type"] == "video"
        assert vid_part["video"]["data"] == _TINY_VIDEO_B64
        assert vid_part["video"]["mime_type"] == "video/mp4"

    def test_video_block_url(self):
        vid = VideoBlock(
            media=MediaRef(kind="url", url="https://example.com/vid.mp4", mime_type="video/mp4"),
        )
        blocks = [TextBlock(text="Watch:"), vid]
        result = _build_content(blocks)
        assert isinstance(result, list)
        vid_part = result[1]
        assert vid_part["type"] == "video"
        assert vid_part["video"]["url"] == "https://example.com/vid.mp4"

    def test_mixed_all_types(self):
        """Message with text + image + audio + document + video."""
        blocks = [
            TextBlock(text="Multi:"),
            _make_image_block(),
            _make_audio_block(),
            _make_doc_block(),
            _make_video_block(),
        ]
        result = _build_content(blocks)
        assert isinstance(result, list)
        types = [p["type"] for p in result]
        assert types == ["text", "image_url", "input_audio", "document", "video"]

    def test_has_multimodal_blocks_detection(self):
        assert _has_multimodal_blocks([TextBlock(text="hi")]) is False
        assert _has_multimodal_blocks([_make_image_block()]) is True
        assert _has_multimodal_blocks([_make_audio_block()]) is True
        assert _has_multimodal_blocks([_make_doc_block()]) is True
        assert _has_multimodal_blocks([_make_video_block()]) is True

    def test_convert_dict_multimodal_user_message(self):
        """_convert_dict on a user message with images returns list content."""
        msg = Message(role="user", content=[TextBlock(text="see"), _make_image_block()])
        result = _convert_dict(msg)
        assert result is not None
        assert isinstance(result["content"], list)
        assert result["role"] == "user"

    def test_convert_dict_text_only_returns_string(self):
        msg = Message(role="user", content=[TextBlock(text="hello")])
        result = _convert_dict(msg)
        assert result is not None
        assert result["content"] == "hello"
        assert isinstance(result["content"], str)


# =========================================================================
# PART 2: strip_media_blocks — Multi-agent text-only safety
# =========================================================================


class TestStripMediaBlocks:
    """Test strip_media_blocks for multi-agent workflows."""

    def test_text_only_messages_unchanged(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = strip_media_blocks(msgs)
        assert result == msgs

    def test_strips_images_from_multimodal(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]
        result = strip_media_blocks(msgs)
        assert len(result) == 1
        # Collapsed to string since only one text part remains
        assert result[0]["content"] == "What's in this image?"

    def test_strips_audio_video_document(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "analyze"},
                    {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                    {"type": "document", "document": {"data": "pdf_b64"}},
                    {"type": "video", "video": {"url": "https://example.com/v.mp4"}},
                ],
            },
        ]
        result = strip_media_blocks(msgs)
        assert result[0]["content"] == "analyze"

    def test_strips_all_media_leaves_empty_string(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]
        result = strip_media_blocks(msgs)
        assert result[0]["content"] == ""

    def test_multiple_text_parts_preserved_as_list(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "image_url", "image_url": {"url": "https://img.com/1.png"}},
                    {"type": "text", "text": "Part 2"},
                ],
            },
        ]
        result = strip_media_blocks(msgs)
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["text"] == "Part 1"
        assert result[0]["content"][1]["text"] == "Part 2"

    def test_mixed_messages(self):
        """System message (string) + user message (multimodal) + assistant (string)."""
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "https://img.com/x.png"}},
                ],
            },
            {"role": "assistant", "content": "It's a cat."},
        ]
        result = strip_media_blocks(msgs)
        assert result[0]["content"] == "Be helpful"  # unchanged
        assert result[1]["content"] == "Describe"  # image stripped, collapsed
        assert result[2]["content"] == "It's a cat."  # unchanged


# =========================================================================
# PART 3: _to_responses_content — OpenAI Responses API input
# =========================================================================


class TestToResponsesContent:
    """Test OpenAI Responses API content conversion."""

    def setup_method(self):
        from tenxgraph.core.graph.agent_internal.openai import _to_responses_content

        self.convert = _to_responses_content

    def test_string_passthrough(self):
        assert self.convert("hello") == "hello"

    def test_text_to_input_text(self):
        result = self.convert([{"type": "text", "text": "hi"}])
        assert result == [{"type": "input_text", "text": "hi"}]

    def test_image_url_to_input_image(self):
        result = self.convert([
            {"type": "image_url", "image_url": {"url": "https://img.com/x.png"}},
        ])
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "https://img.com/x.png"

    def test_input_audio_passthrough(self):
        result = self.convert([
            {"type": "input_audio", "input_audio": {"data": "abc", "format": "mp3"}},
        ])
        assert result[0]["type"] == "input_audio"
        assert result[0]["data"] == "abc"
        assert result[0]["format"] == "mp3"

    def test_document_with_text(self):
        result = self.convert([
            {"type": "document", "document": {"text": "Extracted text from doc"}},
        ])
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "Extracted text from doc"

    def test_document_with_url(self):
        result = self.convert([
            {"type": "document", "document": {"url": "https://example.com/doc.pdf"}},
        ])
        assert result[0]["type"] == "input_file"
        assert result[0]["file_url"] == "https://example.com/doc.pdf"

    def test_video_as_text_reference(self):
        result = self.convert([
            {"type": "video", "video": {"url": "https://example.com/vid.mp4"}},
        ])
        assert result[0]["type"] == "input_text"
        assert "Video:" in result[0]["text"]

    def test_mixed_content(self):
        result = self.convert([
            {"type": "text", "text": "Describe:"},
            {"type": "image_url", "image_url": {"url": "https://img.com/x.png"}},
            {"type": "document", "document": {"text": "Doc content"}},
        ])
        types = [p["type"] for p in result]
        assert types == ["input_text", "input_image", "input_text"]

    def test_unknown_type_passthrough(self):
        result = self.convert([{"type": "custom_thing", "data": "foo"}])
        assert result == [{"type": "custom_thing", "data": "foo"}]


# =========================================================================
# PART 4: _content_parts_to_google — Google GenAI input
# =========================================================================


class TestContentPartsToGoogle:
    """Test Google GenAI content part conversion."""

    def setup_method(self):
        self.mixin = self._create_mixin()

    def _create_mixin(self):
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        mixin = AgentGoogleMixin()
        return mixin

    def test_text_part(self):
        from google.genai import types

        parts = self.mixin._content_parts_to_google([{"type": "text", "text": "hello"}])
        assert len(parts) == 1
        assert parts[0].text == "hello"

    def test_image_base64(self):
        parts = self.mixin._content_parts_to_google([
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
            },
        ])
        assert len(parts) == 1
        # from_bytes returns a Part with inline_data
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.mime_type == "image/png"

    def test_image_url(self):
        """External https:// URLs are fetched and converted to inline bytes for Google."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"fake-image-data"
            mock_response.headers.get.return_value = "image/jpeg"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            parts = self.mixin._content_parts_to_google([
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.jpg"},
                },
            ])
            assert len(parts) == 1
            # External URLs are now fetched and converted to inline bytes
            assert parts[0].inline_data is not None
            assert parts[0].inline_data.mime_type == "image/jpeg"

    def test_audio_part(self):
        parts = self.mixin._content_parts_to_google([
            {
                "type": "input_audio",
                "input_audio": {"data": _TINY_AUDIO_B64, "format": "wav"},
            },
        ])
        assert len(parts) == 1
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.mime_type == "audio/wav"

    def test_document_with_text(self):
        parts = self.mixin._content_parts_to_google([
            {
                "type": "document",
                "document": {"text": "Extracted text", "data": None, "url": None},
            },
        ])
        assert len(parts) == 1
        assert parts[0].text == "Extracted text"

    def test_document_with_base64(self):
        parts = self.mixin._content_parts_to_google([
            {
                "type": "document",
                "document": {
                    "data": _TINY_PDF_B64,
                    "mime_type": "application/pdf",
                    "text": None,
                    "url": None,
                },
            },
        ])
        assert len(parts) == 1
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.mime_type == "application/pdf"

    def test_document_with_url(self):
        """External https:// URLs are fetched and converted to inline bytes for Google."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"fake-pdf-data"
            mock_response.headers.get.return_value = "application/pdf"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            parts = self.mixin._content_parts_to_google([
                {
                    "type": "document",
                    "document": {
                        "url": "https://example.com/doc.pdf",
                        "mime_type": "application/pdf",
                        "data": None,
                        "text": None,
                    },
                },
            ])
            assert len(parts) == 1
            # External URLs are now fetched and converted to inline bytes
            assert parts[0].inline_data is not None
            assert parts[0].inline_data.mime_type == "application/pdf"

    def test_video_with_base64(self):
        parts = self.mixin._content_parts_to_google([
            {
                "type": "video",
                "video": {"data": _TINY_VIDEO_B64, "mime_type": "video/mp4", "url": None},
            },
        ])
        assert len(parts) == 1
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.mime_type == "video/mp4"

    def test_video_with_url(self):
        """External https:// URLs are fetched and converted to inline bytes for Google."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"fake-video-data"
            mock_response.headers.get.return_value = "video/mp4"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            parts = self.mixin._content_parts_to_google([
                {
                    "type": "video",
                    "video": {"url": "https://example.com/vid.mp4", "mime_type": "video/mp4"},
                },
            ])
            assert len(parts) == 1
            # External URLs are now fetched and converted to inline bytes
            assert parts[0].inline_data is not None
            assert parts[0].inline_data.mime_type == "video/mp4"

    def test_mixed_all_types(self):
        parts = self.mixin._content_parts_to_google([
            {"type": "text", "text": "Analyze this:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
            {"type": "input_audio", "input_audio": {"data": _TINY_AUDIO_B64, "format": "wav"}},
            {"type": "document", "document": {"text": "Doc content", "data": None, "url": None}},
            {"type": "video", "video": {"url": "https://example.com/vid.mp4", "mime_type": "video/mp4"}},
        ])
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"fake-video-data"
            mock_response.headers.get.return_value = "video/mp4"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            parts = self.mixin._content_parts_to_google([
                {"type": "text", "text": "Analyze this:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
                {"type": "input_audio", "input_audio": {"data": _TINY_AUDIO_B64, "format": "wav"}},
                {"type": "document", "document": {"text": "Doc content", "data": None, "url": None}},
                {"type": "video", "video": {"url": "https://example.com/vid.mp4", "mime_type": "video/mp4"}},
            ])
            assert len(parts) == 5
            assert parts[0].text == "Analyze this:"
            assert parts[1].inline_data is not None  # image
            assert parts[2].inline_data is not None  # audio
            assert parts[3].text == "Doc content"  # extracted document text
            # Video URL is now fetched and converted to inline bytes
            assert parts[4].inline_data is not None


# =========================================================================
# PART 5: OpenAI Chat Converter — Output (Response → Message)
# =========================================================================


class TestOpenAIChatConverterOutput:
    """Test OpenAI Chat Completions converter handles multimodal output."""

    @pytest.fixture
    def converter(self):
        from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter

        return OpenAIConverter()

    @pytest.mark.asyncio
    async def test_text_response(self, converter):
        response = self._make_response(content="Hello!", model="gpt-4o")
        msg = await converter.convert_response(response)
        assert msg.text() == "Hello!"
        assert msg.role == "assistant"

    @pytest.mark.asyncio
    async def test_response_with_audio(self, converter):
        audio_data = SimpleNamespace(
            data=_TINY_AUDIO_B64,
            transcript="Hello world",
            id="audio-1",
            expires_at=999,
        )
        response = self._make_response(content="", audio=audio_data)
        msg = await converter.convert_response(response)
        audio_blocks = [b for b in msg.content if isinstance(b, AudioBlock)]
        assert len(audio_blocks) == 1
        assert audio_blocks[0].media.data_base64 == _TINY_AUDIO_B64
        assert audio_blocks[0].transcript == "Hello world"

    @pytest.mark.asyncio
    async def test_response_with_images(self, converter):
        images = [SimpleNamespace(url="https://example.com/gen.png")]
        response = self._make_response(content="Here's the image", images=images)
        msg = await converter.convert_response(response)
        image_blocks = [b for b in msg.content if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].media.url == "https://example.com/gen.png"

    @pytest.mark.asyncio
    async def test_response_with_reasoning(self, converter):
        response = self._make_response(
            content="42",
            reasoning_content="Let me think... 6 * 7 = 42",
        )
        msg = await converter.convert_response(response)
        reasoning_blocks = [b for b in msg.content if isinstance(b, ReasoningBlock)]
        assert len(reasoning_blocks) == 1
        assert "6 * 7" in reasoning_blocks[0].summary

    @pytest.mark.asyncio
    async def test_response_with_tool_calls(self, converter):
        tool_call = SimpleNamespace(
            id="call_123",
            type="function",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"city": "NYC"}',
            ),
        )
        tool_call.model_dump = lambda: {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        }
        response = self._make_response(content="", tool_calls=[tool_call])
        msg = await converter.convert_response(response)
        tc_blocks = [b for b in msg.content if isinstance(b, ToolCallBlock)]
        assert len(tc_blocks) == 1
        assert tc_blocks[0].name == "get_weather"

    @staticmethod
    def _make_response(
        content="",
        model="gpt-4o",
        audio=None,
        images=None,
        reasoning_content=None,
        tool_calls=None,
    ):
        message = SimpleNamespace(
            content=content,
            role="assistant",
            audio=audio,
            images=images,
            reasoning_content=reasoning_content or "",
            reasoning=None,
            tool_calls=tool_calls,
        )
        choice = SimpleNamespace(
            message=message,
            finish_reason="stop",
        )
        usage = SimpleNamespace(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        )
        return SimpleNamespace(
            id="chatcmpl-123",
            choices=[choice],
            model=model,
            usage=usage,
            created=1234567890,
            system_fingerprint="fp_abc",
            service_tier=None,
        )


# =========================================================================
# PART 6: OpenAI Responses Converter — Output (Response → Message)
# =========================================================================


class TestOpenAIResponsesConverterOutput:
    """Test OpenAI Responses API converter handles multimodal output."""

    @pytest.fixture
    def converter(self):
        from tenxgraph.runtime.adapters.llm.openai_responses_converter import OpenAIResponsesConverter

        return OpenAIResponsesConverter()

    @pytest.mark.asyncio
    async def test_text_response(self, converter):
        response = self._make_response(text="Hello!")
        msg = await converter.convert_response(response)
        assert msg.text() == "Hello!"

    @pytest.mark.asyncio
    async def test_response_with_output_image(self, converter):
        """Model returns an image in message content (output_image)."""
        content = [
            SimpleNamespace(type="output_text", text="Here's the image:"),
            SimpleNamespace(
                type="output_image",
                image_url=None,
                image_data=_TINY_PNG_B64,
            ),
        ]
        response = self._make_response(message_content=content)
        msg = await converter.convert_response(response)
        image_blocks = [b for b in msg.content if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].media.data_base64 == _TINY_PNG_B64

    @pytest.mark.asyncio
    async def test_response_with_output_audio(self, converter):
        content = [
            SimpleNamespace(type="output_text", text="Listen:"),
            SimpleNamespace(
                type="output_audio",
                data=_TINY_AUDIO_B64,
                transcript="hello",
            ),
        ]
        response = self._make_response(message_content=content)
        msg = await converter.convert_response(response)
        audio_blocks = [b for b in msg.content if isinstance(b, AudioBlock)]
        assert len(audio_blocks) == 1
        assert audio_blocks[0].transcript == "hello"

    @pytest.mark.asyncio
    async def test_response_with_image_generation(self, converter):
        """Image generation call (DALL-E via Responses API)."""
        response = self._make_response(image_gen_b64=_TINY_PNG_B64)
        msg = await converter.convert_response(response)
        image_blocks = [b for b in msg.content if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].media.data_base64 == _TINY_PNG_B64

    @pytest.mark.asyncio
    async def test_response_with_reasoning(self, converter):
        response = self._make_response(reasoning_text="Thinking hard...")
        msg = await converter.convert_response(response)
        reasoning_blocks = [b for b in msg.content if isinstance(b, ReasoningBlock)]
        assert len(reasoning_blocks) == 1

    @pytest.mark.asyncio
    async def test_response_with_function_call(self, converter):
        response = self._make_response(function_call=("get_weather", '{"city": "NYC"}', "call_1"))
        msg = await converter.convert_response(response)
        tc_blocks = [b for b in msg.content if isinstance(b, ToolCallBlock)]
        assert len(tc_blocks) == 1
        assert tc_blocks[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_streaming_with_image_output_item(self, converter):
        """Streaming: output_item.done with message containing output_image."""
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="Here"),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="Here"),
                        SimpleNamespace(
                            type="output_image",
                            image_url=None,
                            image_data=_TINY_PNG_B64,
                        ),
                    ],
                ),
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        input_tokens_details=None,
                        output_tokens_details=None,
                    ),
                ),
            ),
        ]

        async def _async_iter():
            for e in events:
                yield e

        messages = []
        async for msg in converter.convert_streaming_response(
            config={"thread_id": "t1"},
            node_name="test",
            response=_async_iter(),
        ):
            messages.append(msg)

        # Should have: text delta, media delta, final message
        delta_msgs = [m for m in messages if m.delta]
        assert any(
            any(isinstance(b, ImageBlock) for b in m.content) for m in delta_msgs
        ), "Should have an image block in streaming deltas"

    @pytest.mark.asyncio
    async def test_streaming_with_image_generation_item(self, converter):
        """Streaming: output_item.done with image_generation_call."""
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="image_generation_call",
                    result=_TINY_PNG_B64,
                ),
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        input_tokens_details=None,
                        output_tokens_details=None,
                    ),
                ),
            ),
        ]

        async def _async_iter():
            for e in events:
                yield e

        messages = []
        async for msg in converter.convert_streaming_response(
            config={"thread_id": "t1"},
            node_name="test",
            response=_async_iter(),
        ):
            messages.append(msg)

        delta_msgs = [m for m in messages if m.delta]
        assert any(
            any(isinstance(b, ImageBlock) for b in m.content) for m in delta_msgs
        ), "Should have image generation block in streaming deltas"

    @staticmethod
    def _make_response(
        text=None,
        message_content=None,
        image_gen_b64=None,
        reasoning_text=None,
        function_call=None,
    ):
        output = []

        if reasoning_text:
            output.append(SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text=reasoning_text)],
            ))

        if text:
            output.append(SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            ))
        elif message_content:
            output.append(SimpleNamespace(
                type="message",
                content=message_content,
            ))

        if image_gen_b64:
            output.append(SimpleNamespace(
                type="image_generation_call",
                result=image_gen_b64,
            ))

        if function_call:
            name, args, call_id = function_call
            output.append(SimpleNamespace(
                type="function_call",
                name=name,
                arguments=args,
                call_id=call_id,
            ))

        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=None,
            output_tokens_details=None,
        )
        return SimpleNamespace(
            id="resp_123",
            output=output,
            model="gpt-4o",
            status="completed",
            created_at=1234567890,
            usage=usage,
        )


# =========================================================================
# PART 7: Google GenAI Converter — Output (Response → Message)
# =========================================================================


class TestGoogleGenAIConverterOutput:
    """Test Google GenAI converter handles multimodal output."""

    @pytest.fixture
    def converter(self):
        from tenxgraph.runtime.adapters.llm.google_genai_converter import GoogleGenAIConverter

        return GoogleGenAIConverter()

    @pytest.mark.asyncio
    async def test_text_response(self, converter):
        response = self._make_response(text="Hello!")
        msg = await converter.convert_response(response)
        assert msg.text() == "Hello!"

    @pytest.mark.asyncio
    async def test_response_with_inline_image(self, converter):
        """Model returns an inline image (e.g., Gemini image generation)."""
        response = self._make_response(
            text="Here's the image",
            inline_image={"data": _TINY_PNG_B64, "mime_type": "image/png"},
        )
        msg = await converter.convert_response(response)
        image_blocks = [b for b in msg.content if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].media.data_base64 == _TINY_PNG_B64

    @pytest.mark.asyncio
    async def test_response_with_inline_audio(self, converter):
        response = self._make_response(
            text="Listen",
            inline_audio={"data": _TINY_AUDIO_B64, "mime_type": "audio/wav"},
        )
        msg = await converter.convert_response(response)
        audio_blocks = [b for b in msg.content if isinstance(b, AudioBlock)]
        assert len(audio_blocks) == 1

    @pytest.mark.asyncio
    async def test_response_with_inline_video(self, converter):
        response = self._make_response(
            text="Watch",
            inline_video={"data": _TINY_VIDEO_B64, "mime_type": "video/mp4"},
        )
        msg = await converter.convert_response(response)
        video_blocks = [b for b in msg.content if isinstance(b, VideoBlock)]
        assert len(video_blocks) == 1

    @pytest.mark.asyncio
    async def test_response_with_file_data(self, converter):
        """Model returns a file reference (e.g., Gemini Files API)."""
        response = self._make_response(
            text="See file",
            file_data={"file_uri": "https://genai.google.com/files/123", "mime_type": "image/jpeg"},
        )
        msg = await converter.convert_response(response)
        image_blocks = [b for b in msg.content if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].media.url == "https://genai.google.com/files/123"

    @pytest.mark.asyncio
    async def test_response_with_thought(self, converter):
        response = self._make_response(thought="Let me think...")
        msg = await converter.convert_response(response)
        reasoning_blocks = [b for b in msg.content if isinstance(b, ReasoningBlock)]
        assert len(reasoning_blocks) == 1

    @pytest.mark.asyncio
    async def test_response_with_function_call(self, converter):
        response = self._make_response(
            function_call={"name": "get_weather", "args": {"city": "NYC"}},
        )
        msg = await converter.convert_response(response)
        tc_blocks = [b for b in msg.content if isinstance(b, ToolCallBlock)]
        assert len(tc_blocks) == 1
        assert tc_blocks[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_streaming_with_inline_image(self, converter):
        """Streaming chunks with inline_data should produce ImageBlock."""
        chunks = [
            self._make_chunk(text="Here"),
            self._make_chunk(inline_image={"data": _TINY_PNG_B64, "mime_type": "image/png"}),
        ]

        messages = []
        async for msg in converter.convert_streaming_response(
            config={"thread_id": "t1"},
            node_name="test",
            response=self._async_iter(chunks),
        ):
            messages.append(msg)

        delta_msgs = [m for m in messages if m.delta]
        assert any(
            any(isinstance(b, ImageBlock) for b in m.content) for m in delta_msgs
        ), "Should have image block in streaming deltas"

    @pytest.mark.asyncio
    async def test_streaming_with_inline_audio(self, converter):
        chunks = [
            self._make_chunk(text="Listen"),
            self._make_chunk(inline_audio={"data": _TINY_AUDIO_B64, "mime_type": "audio/wav"}),
        ]

        messages = []
        async for msg in converter.convert_streaming_response(
            config={"thread_id": "t1"},
            node_name="test",
            response=self._async_iter(chunks),
        ):
            messages.append(msg)

        delta_msgs = [m for m in messages if m.delta]
        assert any(
            any(isinstance(b, AudioBlock) for b in m.content) for m in delta_msgs
        ), "Should have audio block in streaming deltas"

    @staticmethod
    async def _async_iter(chunks):
        for chunk in chunks:
            yield chunk

    @staticmethod
    def _make_part(
        text=None,
        thought=False,
        inline_data=None,
        file_data=None,
        function_call=None,
    ):
        part = SimpleNamespace(
            text=text,
            thought=thought,
            inline_data=None,
            file_data=None,
            function_call=None,
        )
        if inline_data:
            part.inline_data = SimpleNamespace(
                data=inline_data["data"],
                mime_type=inline_data["mime_type"],
            )
        if file_data:
            part.file_data = SimpleNamespace(
                file_uri=file_data["file_uri"],
                mime_type=file_data["mime_type"],
            )
        if function_call:
            part.function_call = SimpleNamespace(
                name=function_call["name"],
                args=function_call["args"],
            )
        return part

    @classmethod
    def _make_response(
        cls,
        text=None,
        thought=None,
        inline_image=None,
        inline_audio=None,
        inline_video=None,
        file_data=None,
        function_call=None,
    ):
        parts = []
        if thought:
            parts.append(cls._make_part(text=thought, thought=True))
        if text:
            parts.append(cls._make_part(text=text))
        if inline_image:
            parts.append(cls._make_part(inline_data=inline_image))
        if inline_audio:
            parts.append(cls._make_part(inline_data=inline_audio))
        if inline_video:
            parts.append(cls._make_part(inline_data=inline_video))
        if file_data:
            parts.append(cls._make_part(file_data=file_data))
        if function_call:
            parts.append(cls._make_part(function_call=function_call))

        content = SimpleNamespace(parts=parts)
        candidate = SimpleNamespace(
            content=content,
            finish_reason="STOP",
        )
        usage = SimpleNamespace(
            candidates_token_count=10,
            prompt_token_count=20,
            total_token_count=30,
            cached_content_token_count=0,
            thoughts_token_count=0,
        )
        return SimpleNamespace(
            candidates=[candidate],
            model_version="gemini-2.0-flash",
            usage_metadata=usage,
            response_id="resp_123",
            create_time=None,
        )

    @classmethod
    def _make_chunk(cls, text=None, inline_image=None, inline_audio=None):
        parts = []
        if text:
            parts.append(cls._make_part(text=text))
        if inline_image:
            parts.append(cls._make_part(inline_data=inline_image))
        if inline_audio:
            parts.append(cls._make_part(inline_data=inline_audio))

        content = SimpleNamespace(parts=parts)
        candidate = SimpleNamespace(content=content, finish_reason=None)
        return SimpleNamespace(candidates=[candidate])


# =========================================================================
# PART 8: Multi-agent workflow integration
# =========================================================================


class TestMultiAgentImageStripping:
    """Test that text-only agents strip media from context messages."""

    def test_convert_messages_with_multimodal_context(self):
        """When context has image messages, convert_messages includes them."""
        from tenxgraph.core.state import AgentState

        state = AgentState(
            context=[
                Message(role="user", content=[
                    TextBlock(text="What's in this image?"),
                    _make_image_block(),
                ]),
                Message(role="assistant", content=[TextBlock(text="It's a cat.")]),
            ],
        )
        messages = convert_messages(
            system_prompts=[{"role": "system", "content": "You are helpful"}],
            state=state,
        )
        # The user message should have list content
        user_msg = next(m for m in messages if m["role"] == "user")
        assert isinstance(user_msg["content"], list)

    def test_strip_after_convert(self):
        """strip_media_blocks removes images from converted messages."""
        from tenxgraph.core.state import AgentState

        state = AgentState(
            context=[
                Message(role="user", content=[
                    TextBlock(text="What's in this image?"),
                    _make_image_block(),
                ]),
                Message(role="assistant", content=[TextBlock(text="It's a cat.")]),
            ],
        )
        messages = convert_messages(
            system_prompts=[{"role": "system", "content": "You are helpful"}],
            state=state,
        )
        stripped = strip_media_blocks(messages)
        # After stripping, user message should be text-only
        user_msg = next(m for m in stripped if m["role"] == "user")
        assert isinstance(user_msg["content"], str)
        assert user_msg["content"] == "What's in this image?"

    def test_strip_preserves_system_and_assistant(self):
        """System and assistant text messages are unchanged after stripping."""
        from tenxgraph.core.state import AgentState

        state = AgentState(
            context=[
                Message(role="user", content=[
                    TextBlock(text="Analyze"),
                    _make_image_block(),
                    _make_audio_block(),
                    _make_doc_block(),
                    _make_video_block(),
                ]),
                Message(role="assistant", content=[TextBlock(text="Done.")]),
            ],
        )
        messages = convert_messages(
            system_prompts=[{"role": "system", "content": "Analyze images"}],
            state=state,
        )
        stripped = strip_media_blocks(messages)

        system_msg = next(m for m in stripped if m["role"] == "system")
        assert system_msg["content"] == "Analyze images"

        assistant_msg = next(m for m in stripped if m["role"] == "assistant")
        assert assistant_msg["content"] == "Done."

        user_msg = next(m for m in stripped if m["role"] == "user")
        assert user_msg["content"] == "Analyze"


# =========================================================================
# PART 9: Google _handle_regular_message multimodal routing
# =========================================================================


class TestGoogleHandleRegularMessage:
    """Test Google mixin _handle_regular_message handles multimodal content."""

    def setup_method(self):
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        self.mixin = AgentGoogleMixin()

    def test_string_content(self):
        from google.genai import types

        result = self.mixin._handle_regular_message("Hello!", "user")
        assert result.role == "user"
        assert len(result.parts) == 1
        assert result.parts[0].text == "Hello!"

    def test_list_content_with_image(self):
        """List content (from _build_content) is converted to Google parts."""
        content = [
            {"type": "text", "text": "Describe:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
        ]
        result = self.mixin._handle_regular_message(content, "user")
        assert result.role == "user"
        assert len(result.parts) == 2
        assert result.parts[0].text == "Describe:"
        assert result.parts[1].inline_data is not None

    def test_empty_list_gets_fallback(self):
        """Empty list content should produce a fallback empty text part."""
        result = self.mixin._handle_regular_message([], "user")
        assert len(result.parts) == 1
        assert result.parts[0].text == ""


# =========================================================================
# PART 10: _convert_dict edge cases
# =========================================================================


class TestConvertDictEdgeCases:
    """Edge cases for _convert_dict."""

    def test_assistant_with_tool_calls_text_only(self):
        """Assistant message with tool_calls returns text content as string."""
        msg = Message(
            role="assistant",
            content=[TextBlock(text="Calling tool")],
            tools_calls=[{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}],
        )
        result = _convert_dict(msg)
        assert result is not None
        assert result["content"] == "Calling tool"
        assert result["tool_calls"] is not None

    def test_tool_message(self):
        from tenxgraph.core.state.message_block import ToolResultBlock

        msg = Message(
            role="tool",
            content=[ToolResultBlock(call_id="call_1", output="result")],
        )
        result = _convert_dict(msg)
        assert result is not None
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"


# =========================================================================
# PART 11: Full pipeline integration — Message → _convert_dict → _build_content → provider
# =========================================================================


class TestFullPipelineIntegration:
    """Test the complete pipeline from Message objects through to provider format."""

    def test_message_to_openai_chat_with_image(self):
        """Message with image → _convert_dict → content list with image_url part."""
        msg = Message(role="user", content=[
            TextBlock(text="What is this?"),
            _make_image_block(),
        ])
        result = _convert_dict(msg)
        assert result is not None
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image_url"

    def test_message_to_openai_responses_with_image(self):
        """Message with image → _convert_dict → _to_responses_content."""
        from tenxgraph.core.graph.agent_internal.openai import _to_responses_content

        msg = Message(role="user", content=[
            TextBlock(text="Describe"),
            _make_image_block(kind="url"),
        ])
        result = _convert_dict(msg)
        assert result is not None
        responses_content = _to_responses_content(result["content"])
        assert isinstance(responses_content, list)
        assert responses_content[0]["type"] == "input_text"
        assert responses_content[1]["type"] == "input_image"

    def test_message_to_google_with_all_media(self):
        """Message with all media types → _convert_dict → Google parts."""
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        mixin = AgentGoogleMixin()
        msg = Message(role="user", content=[
            TextBlock(text="Analyze everything:"),
            _make_image_block(),
            _make_audio_block(),
            _make_doc_block(excerpt="Document text"),
            _make_video_block(),
        ])
        result = _convert_dict(msg)
        assert result is not None
        assert isinstance(result["content"], list)

        # Convert to Google format
        google_content = mixin._handle_regular_message(result["content"], "user")
        assert len(google_content.parts) == 5

    def test_text_only_message_backward_compatible(self):
        """Plain text message still produces string content (no regression)."""
        msg = Message(role="user", content=[TextBlock(text="Hello")])
        result = _convert_dict(msg)
        assert result is not None
        assert result["content"] == "Hello"
        assert isinstance(result["content"], str)

    def test_multimodal_then_strip_then_send(self):
        """Full pipeline: multimodal message → convert → strip → text only."""
        msg = Message(role="user", content=[
            TextBlock(text="Describe this:"),
            _make_image_block(),
            _make_audio_block(),
        ])
        converted = _convert_dict(msg)
        assert converted is not None
        assert isinstance(converted["content"], list)

        # Strip for text-only agent
        stripped = strip_media_blocks([converted])
        assert stripped[0]["content"] == "Describe this:"
        assert isinstance(stripped[0]["content"], str)
