"""Tests for Sprint 1: Core Multimodal Pipeline.

Covers:
- MultimodalConfig model
- _convert_dict with multimodal blocks (ImageBlock, AudioBlock, DocumentBlock)
- Google format conversion with multimodal content
- OpenAI format conversion (content as list)
- Backward compatibility: text-only messages unchanged
- Message convenience constructors (image_message, multimodal_message, from_file)
"""

import base64
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.storage.media.config import DocumentHandling, ImageHandling, MultimodalConfig
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    MediaRef,
    TextBlock,
    ToolResultBlock,
)
from tenxgraph.utils.converter import (
    _build_content,
    _convert_dict,
    _has_multimodal_blocks,
    _image_block_to_openai,
    _audio_block_to_openai,
    _document_block_to_openai,
    convert_messages,
)


# ---------------------------------------------------------------------------
# MultimodalConfig tests
# ---------------------------------------------------------------------------


class TestMultimodalConfig:
    """Test the MultimodalConfig pydantic model."""

    def test_defaults(self):
        cfg = MultimodalConfig()
        assert cfg.image_handling == ImageHandling.BASE64
        assert cfg.document_handling == DocumentHandling.EXTRACT_TEXT
        assert cfg.max_image_size_mb == 10.0
        assert cfg.max_image_dimension == 2048
        assert "image/jpeg" in cfg.supported_image_types
        assert "image/png" in cfg.supported_image_types
        assert "image/webp" in cfg.supported_image_types
        assert "image/gif" in cfg.supported_image_types
        assert "application/pdf" in cfg.supported_doc_types

    def test_custom_values(self):
        cfg = MultimodalConfig(
            image_handling=ImageHandling.URL,
            document_handling=DocumentHandling.SKIP,
            max_image_size_mb=5.0,
            max_image_dimension=1024,
        )
        assert cfg.image_handling == ImageHandling.URL
        assert cfg.document_handling == DocumentHandling.SKIP
        assert cfg.max_image_size_mb == 5.0
        assert cfg.max_image_dimension == 1024

    def test_enum_values(self):
        assert ImageHandling.BASE64 == "base64"
        assert ImageHandling.URL == "url"
        assert ImageHandling.FILE_ID == "file_id"
        assert DocumentHandling.EXTRACT_TEXT == "extract_text"
        assert DocumentHandling.FORWARD_RAW == "pass_raw"
        assert DocumentHandling.SKIP == "skip"

    def test_serialization_roundtrip(self):
        cfg = MultimodalConfig(image_handling=ImageHandling.URL)
        data = cfg.model_dump()
        restored = MultimodalConfig.model_validate(data)
        assert restored.image_handling == ImageHandling.URL


# ---------------------------------------------------------------------------
# _has_multimodal_blocks tests
# ---------------------------------------------------------------------------


class TestHasMultimodalBlocks:
    def test_text_only(self):
        blocks = [TextBlock(text="Hello")]
        assert _has_multimodal_blocks(blocks) is False

    def test_with_image_block(self):
        blocks = [
            TextBlock(text="Describe this image"),
            ImageBlock(media=MediaRef(kind="url", url="https://example.com/img.jpg")),
        ]
        assert _has_multimodal_blocks(blocks) is True

    def test_with_audio_block(self):
        blocks = [AudioBlock(media=MediaRef(kind="data", data_base64="AAAA", mime_type="audio/wav"))]
        assert _has_multimodal_blocks(blocks) is True

    def test_with_document_block(self):
        blocks = [DocumentBlock(media=MediaRef(kind="url", url="https://example.com/doc.pdf"))]
        assert _has_multimodal_blocks(blocks) is True

    def test_with_dict_image(self):
        blocks = [{"type": "image", "media": {}}]
        assert _has_multimodal_blocks(blocks) is True

    def test_empty_list(self):
        assert _has_multimodal_blocks([]) is False


# ---------------------------------------------------------------------------
# _image_block_to_openai tests
# ---------------------------------------------------------------------------


class TestImageBlockToOpenAI:
    def test_base64_image(self):
        block = ImageBlock(
            media=MediaRef(kind="data", data_base64="abc123", mime_type="image/jpeg")
        )
        result = _image_block_to_openai(block)
        assert result == {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,abc123"},
        }

    def test_base64_default_mime(self):
        block = ImageBlock(
            media=MediaRef(kind="data", data_base64="abc123")
        )
        result = _image_block_to_openai(block)
        assert "data:image/png;base64,abc123" in result["image_url"]["url"]

    def test_url_image(self):
        block = ImageBlock(
            media=MediaRef(kind="url", url="https://example.com/img.jpg")
        )
        result = _image_block_to_openai(block)
        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.jpg"},
        }

    def test_file_id_image(self):
        block = ImageBlock(
            media=MediaRef(kind="file_id", file_id="file-abc123")
        )
        result = _image_block_to_openai(block)
        assert result == {
            "type": "image_url",
            "image_url": {"url": "file-abc123"},
        }


# ---------------------------------------------------------------------------
# _audio_block_to_openai tests
# ---------------------------------------------------------------------------


class TestAudioBlockToOpenAI:
    def test_base64_audio_wav(self):
        block = AudioBlock(
            media=MediaRef(kind="data", data_base64="audiodata", mime_type="audio/wav")
        )
        result = _audio_block_to_openai(block)
        assert result == {
            "type": "input_audio",
            "input_audio": {"data": "audiodata", "format": "wav"},
        }

    def test_base64_audio_mp3(self):
        block = AudioBlock(
            media=MediaRef(kind="data", data_base64="audiodata", mime_type="audio/mp3")
        )
        result = _audio_block_to_openai(block)
        assert result["input_audio"]["format"] == "mp3"

    def test_url_audio_fallback(self):
        block = AudioBlock(
            media=MediaRef(kind="url", url="https://example.com/audio.wav")
        )
        result = _audio_block_to_openai(block)
        assert result["type"] == "text"
        assert "https://example.com/audio.wav" in result["text"]


# ---------------------------------------------------------------------------
# _document_block_to_openai tests
# ---------------------------------------------------------------------------


class TestDocumentBlockToOpenAI:
    def test_document_with_excerpt(self):
        block = DocumentBlock(
            media=MediaRef(kind="url", url="https://example.com/doc.pdf"),
            excerpt="This is the extracted text from the document.",
        )
        result = _document_block_to_openai(block)
        assert len(result) == 1
        assert result[0] == {
            "type": "text",
            "text": "This is the extracted text from the document.",
        }

    def test_document_base64_no_excerpt(self):
        block = DocumentBlock(
            media=MediaRef(kind="data", data_base64="pdfdata", mime_type="application/pdf")
        )
        result = _document_block_to_openai(block)
        assert len(result) == 1
        assert result[0]["type"] == "document"
        assert result[0]["document"]["data"] == "pdfdata"
        assert result[0]["document"]["mime_type"] == "application/pdf"

    def test_document_url_no_excerpt(self):
        block = DocumentBlock(
            media=MediaRef(kind="url", url="https://example.com/doc.pdf")
        )
        result = _document_block_to_openai(block)
        assert len(result) == 1
        assert result[0]["type"] == "document"
        assert result[0]["document"]["url"] == "https://example.com/doc.pdf"

    def test_document_file_id_no_excerpt(self):
        block = DocumentBlock(
            media=MediaRef(kind="file_id", file_id="file-xyz")
        )
        result = _document_block_to_openai(block)
        assert len(result) == 1
        assert "file-xyz" in result[0]["text"]


# ---------------------------------------------------------------------------
# _build_content tests
# ---------------------------------------------------------------------------


class TestBuildContent:
    def test_text_only_returns_string(self):
        blocks = [TextBlock(text="Hello world")]
        result = _build_content(blocks)
        assert isinstance(result, str)
        assert result == "Hello world"

    def test_multiple_text_blocks_joined(self):
        blocks = [TextBlock(text="Hello"), TextBlock(text=" world")]
        result = _build_content(blocks)
        assert result == "Hello world"

    def test_image_returns_list(self):
        blocks = [
            TextBlock(text="Describe this"),
            ImageBlock(media=MediaRef(kind="url", url="https://img.com/a.jpg")),
        ]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Describe this"}
        assert result[1]["type"] == "image_url"

    def test_empty_text_block_skipped_in_multimodal(self):
        blocks = [
            TextBlock(text=""),
            ImageBlock(media=MediaRef(kind="url", url="https://img.com/a.jpg")),
        ]
        result = _build_content(blocks)
        assert isinstance(result, list)
        # Empty text block is skipped
        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_empty_blocks_returns_empty_string(self):
        result = _build_content([])
        assert result == ""

    def test_mixed_multimodal(self):
        blocks = [
            TextBlock(text="Analyze:"),
            ImageBlock(media=MediaRef(kind="data", data_base64="img", mime_type="image/png")),
            AudioBlock(media=MediaRef(kind="data", data_base64="aud", mime_type="audio/wav")),
        ]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[2]["type"] == "input_audio"


# ---------------------------------------------------------------------------
# _convert_dict multimodal tests
# ---------------------------------------------------------------------------


class TestConvertDictMultimodal:
    """Test _convert_dict with multimodal messages."""

    def test_text_only_backward_compat(self):
        """Text-only messages should produce string content (backward compatible)."""
        msg = Message.text_message("Hello world", "user")
        result = _convert_dict(msg)
        assert result == {"role": "user", "content": "Hello world"}

    def test_user_message_with_image_url(self):
        """User message with image should produce list content."""
        msg = Message(
            role="user",
            content=[
                TextBlock(text="What is in this image?"),
                ImageBlock(media=MediaRef(kind="url", url="https://example.com/img.jpg")),
            ],
        )
        result = _convert_dict(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "What is in this image?"}
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][1]["image_url"]["url"] == "https://example.com/img.jpg"

    def test_user_message_with_base64_image(self):
        msg = Message(
            role="user",
            content=[
                TextBlock(text="Describe"),
                ImageBlock(
                    media=MediaRef(
                        kind="data", data_base64="abc123", mime_type="image/jpeg"
                    )
                ),
            ],
        )
        result = _convert_dict(msg)
        assert isinstance(result["content"], list)
        img_part = result["content"][1]
        assert img_part["type"] == "image_url"
        assert img_part["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    def test_user_message_with_audio(self):
        msg = Message(
            role="user",
            content=[
                TextBlock(text="Transcribe this"),
                AudioBlock(
                    media=MediaRef(
                        kind="data", data_base64="audiodata", mime_type="audio/wav"
                    )
                ),
            ],
        )
        result = _convert_dict(msg)
        assert isinstance(result["content"], list)
        audio_part = result["content"][1]
        assert audio_part["type"] == "input_audio"
        assert audio_part["input_audio"]["data"] == "audiodata"
        assert audio_part["input_audio"]["format"] == "wav"

    def test_user_message_with_document_excerpt(self):
        msg = Message(
            role="user",
            content=[
                TextBlock(text="Summarize this PDF"),
                DocumentBlock(
                    media=MediaRef(kind="url", url="https://example.com/doc.pdf"),
                    excerpt="Extracted text content here.",
                ),
            ],
        )
        result = _convert_dict(msg)
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][1] == {
            "type": "text",
            "text": "Extracted text content here.",
        }

    def test_tool_message_unchanged(self):
        """Tool messages should not be affected by multimodal changes."""
        tool_result = ToolResultBlock(call_id="call_123", output="Tool output")
        msg = Message(role="tool", content=[tool_result])
        result = _convert_dict(msg)
        assert result == {
            "role": "tool",
            "content": "Tool output",
            "tool_call_id": "call_123",
        }

    def test_assistant_with_tools_unchanged(self):
        """Assistant messages with tool_calls should not be affected."""
        msg = Message.text_message("Let me check", "assistant")
        msg.tools_calls = [
            {"id": "call_1", "function": {"name": "test", "arguments": "{}"}}
        ]
        result = _convert_dict(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Let me check"
        assert "tool_calls" in result


# ---------------------------------------------------------------------------
# convert_messages integration test with multimodal
# ---------------------------------------------------------------------------


class TestConvertMessagesMultimodal:
    """Test convert_messages with multimodal content in state context."""

    def test_multimodal_message_in_context(self):
        """Multimodal messages in state.context should be converted properly."""
        from tenxgraph.core.state import AgentState

        state = AgentState()
        state.context = [
            Message.text_message("Hello", "user"),
            Message(
                role="user",
                content=[
                    TextBlock(text="What is this?"),
                    ImageBlock(
                        media=MediaRef(kind="url", url="https://example.com/img.jpg")
                    ),
                ],
            ),
        ]

        system = [{"role": "system", "content": "You are a helpful assistant"}]
        result = convert_messages(system, state)

        # System + 2 context messages = 3
        assert len(result) == 3
        # First context message: text-only
        assert result[1]["content"] == "Hello"
        # Second context message: multimodal list
        assert isinstance(result[2]["content"], list)
        assert len(result[2]["content"]) == 2


# ---------------------------------------------------------------------------
# Google format conversion tests
# ---------------------------------------------------------------------------


class TestGoogleMultimodalConversion:
    """Test _handle_regular_message and _content_parts_to_google."""

    def _make_mixin(self):
        """Create a minimal AgentGoogleMixin instance."""
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        mixin = AgentGoogleMixin()
        return mixin

    def test_text_only(self):
        mixin = self._make_mixin()
        result = mixin._handle_regular_message("Hello", "user")
        assert result.role == "user"
        assert len(result.parts) == 1
        assert result.parts[0].text == "Hello"

    def test_empty_content(self):
        mixin = self._make_mixin()
        result = mixin._handle_regular_message("", "user")
        assert result.parts[0].text == ""

    def test_none_content(self):
        mixin = self._make_mixin()
        result = mixin._handle_regular_message(None, "user")
        assert result.parts[0].text == ""

    def test_list_content_with_text(self):
        mixin = self._make_mixin()
        content = [{"type": "text", "text": "Describe this image"}]
        result = mixin._handle_regular_message(content, "user")
        assert result.role == "user"
        assert len(result.parts) == 1
        assert result.parts[0].text == "Describe this image"

    def test_list_content_with_base64_image(self):
        # Create a small valid base64 payload (1x1 red pixel PNG is complex, use simple bytes)
        raw_bytes = b"\x89PNG\r\n\x1a\n"  # PNG header bytes
        b64 = base64.b64encode(raw_bytes).decode()
        mixin = self._make_mixin()
        content = [
            {"type": "text", "text": "What is this?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
        result = mixin._handle_regular_message(content, "user")
        assert result.role == "user"
        assert len(result.parts) == 2
        assert result.parts[0].text == "What is this?"
        # Second part should be from_bytes (has inline_data)
        assert result.parts[1].inline_data is not None
        assert result.parts[1].inline_data.mime_type == "image/png"

    def test_list_content_with_url_image(self):
        """External https:// URLs are fetched and converted to inline bytes."""
        mixin = self._make_mixin()
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"fake-image-data"
            mock_response.headers.get.return_value = "image/jpeg"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = mixin._handle_regular_message(content, "user")
            assert result.role == "user"
            assert len(result.parts) == 1
            # External URLs are now fetched and converted to inline bytes
            assert result.parts[0].inline_data is not None
            assert result.parts[0].inline_data.mime_type == "image/jpeg"

    def test_list_content_with_audio(self):
        raw_bytes = b"\x00\x01\x02\x03"
        b64 = base64.b64encode(raw_bytes).decode()
        mixin = self._make_mixin()
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": b64, "format": "wav"},
            },
        ]
        result = mixin._handle_regular_message(content, "user")
        assert result.role == "user"
        assert len(result.parts) == 1
        assert result.parts[1] if len(result.parts) > 1 else result.parts[0]
        # Should be from_bytes with audio mime
        part = result.parts[0]
        assert part.inline_data is not None
        assert part.inline_data.mime_type == "audio/wav"

    def test_empty_list_content(self):
        mixin = self._make_mixin()
        content = []
        result = mixin._handle_regular_message(content, "user")
        # Empty list should produce a fallback empty text part
        assert len(result.parts) == 1
        assert result.parts[0].text == ""

    def test_convert_to_google_format_with_multimodal(self):
        """End-to-end: messages with list content go through _convert_to_google_format."""
        mixin = self._make_mixin()
        messages = [
            {"role": "system", "content": "You are a visual assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/photo.jpg"},
                    },
                ],
            },
        ]
        sys_instruction, google_contents = mixin._convert_to_google_format(messages)
        assert sys_instruction == "You are a visual assistant."
        assert len(google_contents) == 1
        user_content = google_contents[0]
        assert user_content.role == "user"
        assert len(user_content.parts) == 2


# ---------------------------------------------------------------------------
# Message convenience constructor tests
# ---------------------------------------------------------------------------


class TestMessageConvenienceConstructors:
    """Test Message.image_message, .multimodal_message, .from_file."""

    def test_image_message_with_url(self):
        msg = Message.image_message(
            image_url="https://example.com/img.jpg",
            text="Describe this",
        )
        assert msg.role == "user"
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextBlock)
        assert msg.content[0].text == "Describe this"
        assert isinstance(msg.content[1], ImageBlock)
        assert msg.content[1].media.kind == "url"
        assert msg.content[1].media.url == "https://example.com/img.jpg"

    def test_image_message_with_base64(self):
        msg = Message.image_message(
            image_base64="abc123",
            mime_type="image/jpeg",
        )
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ImageBlock)
        assert msg.content[0].media.kind == "data"
        assert msg.content[0].media.data_base64 == "abc123"

    def test_image_message_no_source_raises(self):
        with pytest.raises(ValueError, match="image_url or image_base64"):
            Message.image_message()

    def test_image_message_custom_role(self):
        msg = Message.image_message(
            image_url="https://example.com/img.jpg",
            role="assistant",
        )
        assert msg.role == "assistant"

    def test_multimodal_message(self):
        blocks = [
            TextBlock(text="Hello"),
            ImageBlock(media=MediaRef(kind="url", url="https://example.com/img.jpg")),
        ]
        msg = Message.multimodal_message(blocks)
        assert msg.role == "user"
        assert len(msg.content) == 2

    def test_multimodal_message_custom_role(self):
        blocks = [TextBlock(text="Response")]
        msg = Message.multimodal_message(blocks, role="assistant")
        assert msg.role == "assistant"

    def test_from_file_image(self):
        """Test from_file with a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\nfakedata")
            f.flush()
            path = f.name

        try:
            msg = Message.from_file(path, text="Analyze this image")
            assert msg.role == "user"
            assert len(msg.content) == 2
            assert isinstance(msg.content[0], TextBlock)
            assert isinstance(msg.content[1], ImageBlock)
            assert msg.content[1].media.kind == "data"
            assert msg.content[1].media.mime_type == "image/png"
            assert msg.content[1].media.filename == Path(path).name
            assert msg.content[1].media.data_base64 is not None
            # Verify roundtrip
            decoded = base64.b64decode(msg.content[1].media.data_base64)
            assert decoded == b"\x89PNG\r\n\x1a\nfakedata"
        finally:
            Path(path).unlink()

    def test_from_file_document(self):
        """Test from_file with a PDF-like file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 fake pdf content")
            f.flush()
            path = f.name

        try:
            msg = Message.from_file(path)
            assert len(msg.content) == 1
            assert isinstance(msg.content[0], DocumentBlock)
            assert msg.content[0].media.mime_type == "application/pdf"
        finally:
            Path(path).unlink()

    def test_from_file_audio(self):
        """Test from_file with a WAV-like file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            f.flush()
            path = f.name

        try:
            msg = Message.from_file(path)
            assert len(msg.content) == 1
            assert isinstance(msg.content[0], AudioBlock)
        finally:
            Path(path).unlink()

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Message.from_file("/nonexistent/file.png")

    def test_from_file_explicit_mime(self):
        """Test from_file with explicit mime_type override."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\nfakedata")
            f.flush()
            path = f.name

        try:
            msg = Message.from_file(path, mime_type="image/png")
            assert isinstance(msg.content[0], ImageBlock)
            assert msg.content[0].media.mime_type == "image/png"
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# End-to-end: Message → _convert_dict → provider format
# ---------------------------------------------------------------------------


class TestEndToEndMultimodal:
    """Full pipeline: create multimodal Message → convert to provider format."""

    def test_image_message_to_openai_format(self):
        """Message.image_message → _convert_dict → OpenAI format."""
        msg = Message.image_message(
            image_url="https://example.com/photo.jpg",
            text="What is in this photo?",
        )
        result = _convert_dict(msg)
        assert result["role"] == "user"
        content = result["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What is in this photo?"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"},
        }

    def test_image_message_to_google_format(self):
        """Message.image_message → _convert_dict → Google format."""
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        msg = Message.image_message(
            image_url="https://example.com/photo.jpg",
            text="What is in this photo?",
        )
        converted = _convert_dict(msg)
        mixin = AgentGoogleMixin()
        messages = [converted]
        _, google_contents = mixin._convert_to_google_format(messages)
        assert len(google_contents) == 1
        user_content = google_contents[0]
        assert user_content.role == "user"
        assert len(user_content.parts) == 2
        assert user_content.parts[0].text == "What is in this photo?"

    def test_base64_image_roundtrip(self):
        """Base64 image → Message → convert → verify data URL."""
        raw = b"\x89PNG\r\n\x1a\n"
        b64 = base64.b64encode(raw).decode()
        msg = Message.image_message(
            image_base64=b64,
            mime_type="image/png",
            text="Describe",
        )
        result = _convert_dict(msg)
        content = result["content"]
        assert isinstance(content, list)
        img_part = content[1]
        expected_url = f"data:image/png;base64,{b64}"
        assert img_part["image_url"]["url"] == expected_url

    def test_attach_media_then_convert(self):
        """Use attach_media then _convert_dict."""
        msg = Message.text_message("Look at this", "user")
        media = MediaRef(kind="url", url="https://example.com/img.jpg", mime_type="image/jpeg")
        msg.attach_media(media, as_type="image")
        result = _convert_dict(msg)
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Look at this"}
        assert content[1]["type"] == "image_url"

    def test_text_only_still_string(self):
        """Ensure backward compat: text-only messages have string content."""
        msg = Message.text_message("Just a text message", "user")
        result = _convert_dict(msg)
        assert result["content"] == "Just a text message"
        assert isinstance(result["content"], str)
