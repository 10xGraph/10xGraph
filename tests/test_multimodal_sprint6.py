"""Sprint 6 tests — image processing utilities, security hardening, and provider optimizations."""
from __future__ import annotations

import base64
import io

import pytest

from tenxgraph.storage.media.config import MultimodalConfig
from tenxgraph.storage.media.processor import MediaProcessor
from tenxgraph.core.state.message_block import ImageBlock, MediaRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def _make_image_bytes(width: int = 100, height: int = 100, fmt: str = "PNG", mode: str = "RGB") -> bytes:
    """Create a small test image with PIL."""
    from PIL import Image

    img = Image.new(mode, (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_image_block(width: int = 100, height: int = 100, fmt: str = "PNG", mime: str = "image/png") -> ImageBlock:
    raw = _make_image_bytes(width, height, fmt)
    b64 = base64.b64encode(raw).decode()
    return ImageBlock(
        media=MediaRef(
            kind="data",
            data_base64=b64,
            mime_type=mime,
            width=width,
            height=height,
            size_bytes=len(raw),
        )
    )


def _make_exif_rotated_image() -> ImageBlock:
    """Create a JPEG with EXIF Orientation tag set to 6 (90° CW rotation)."""
    from PIL import Image
    import piexif

    # Create a non-square image so rotation is detectable
    img = Image.new("RGB", (200, 100), color=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # Add EXIF orientation tag
    exif_dict = {"0th": {piexif.ImageIFD.Orientation: 6}}
    exif_bytes = piexif.dump(exif_dict)

    buf2 = io.BytesIO()
    img.save(buf2, format="JPEG", exif=exif_bytes)
    raw = buf2.getvalue()
    b64 = base64.b64encode(raw).decode()

    return ImageBlock(
        media=MediaRef(
            kind="data",
            data_base64=b64,
            mime_type="image/jpeg",
            width=200,
            height=100,
            size_bytes=len(raw),
        )
    )


# ===========================================================================
# 6.1 — Image Processing Utilities
# ===========================================================================


class TestMediaProcessorResize:
    """Test resize_image() from Sprint 1, verified here for completeness."""

    def test_resize_large_image(self):
        block = _make_image_block(4000, 3000)
        proc = MediaProcessor(MultimodalConfig(max_image_dimension=2048))
        result = proc.resize_image(block)

        assert result.media.width <= 2048
        assert result.media.height <= 2048
        assert result is not block

    def test_small_image_unchanged(self):
        block = _make_image_block(100, 100)
        proc = MediaProcessor(MultimodalConfig(max_image_dimension=2048))
        result = proc.resize_image(block)
        assert result is block  # Identity — no resize needed

    def test_url_block_skipped(self):
        block = ImageBlock(
            media=MediaRef(kind="url", url="https://example.com/img.jpg", mime_type="image/jpeg")
        )
        proc = MediaProcessor()
        result = proc.resize_image(block)
        assert result is block


class TestThumbnailGeneration:
    """Test generate_thumbnail()."""

    def test_creates_thumbnail(self):
        block = _make_image_block(800, 600)
        proc = MediaProcessor()
        thumb = proc.generate_thumbnail(block, max_dim=128)

        assert thumb is not block
        assert thumb.media.width <= 128
        assert thumb.media.height <= 128
        assert thumb.media.mime_type == "image/jpeg"  # Thumbnails are always JPEG

    def test_small_image_unchanged(self):
        block = _make_image_block(64, 64)
        proc = MediaProcessor()
        thumb = proc.generate_thumbnail(block, max_dim=128)
        assert thumb is block

    def test_thumbnail_rgba_to_rgb(self):
        """RGBA images should be converted to RGB for JPEG thumbnail."""
        raw = _make_image_bytes(400, 400, fmt="PNG", mode="RGBA")
        b64 = base64.b64encode(raw).decode()
        block = ImageBlock(
            media=MediaRef(kind="data", data_base64=b64, mime_type="image/png", size_bytes=len(raw))
        )
        proc = MediaProcessor()
        thumb = proc.generate_thumbnail(block, max_dim=128)
        assert thumb.media.mime_type == "image/jpeg"
        assert thumb.media.width <= 128

    def test_thumbnail_url_block_skipped(self):
        block = ImageBlock(
            media=MediaRef(kind="url", url="https://example.com/img.jpg", mime_type="image/jpeg")
        )
        proc = MediaProcessor()
        thumb = proc.generate_thumbnail(block, max_dim=128)
        assert thumb is block


class TestOptimizeImage:
    """Test optimize_image()."""

    def test_optimize_png_to_jpeg(self):
        block = _make_image_block(200, 200, fmt="PNG", mime="image/png")
        proc = MediaProcessor()
        result = proc.optimize_image(block, target_format="JPEG", quality=75)

        assert result.media.mime_type == "image/jpeg"
        # JPEG should be smaller than PNG for a solid color image
        assert result.media.size_bytes is not None

    def test_optimize_to_webp(self):
        block = _make_image_block(200, 200)
        proc = MediaProcessor()
        result = proc.optimize_image(block, target_format="WEBP", quality=80)
        assert result.media.mime_type == "image/webp"

    def test_optimize_rgba_to_jpeg_converts_mode(self):
        raw = _make_image_bytes(200, 200, fmt="PNG", mode="RGBA")
        b64 = base64.b64encode(raw).decode()
        block = ImageBlock(
            media=MediaRef(kind="data", data_base64=b64, mime_type="image/png", size_bytes=len(raw))
        )
        proc = MediaProcessor()
        result = proc.optimize_image(block, target_format="JPEG")
        assert result.media.mime_type == "image/jpeg"

        # Verify the output is decodable
        out_bytes = base64.b64decode(result.media.data_base64)
        from PIL import Image
        img = Image.open(io.BytesIO(out_bytes))
        assert img.mode == "RGB"

    def test_optimize_url_block_skipped(self):
        block = ImageBlock(
            media=MediaRef(kind="url", url="https://example.com/img.jpg", mime_type="image/jpeg")
        )
        proc = MediaProcessor()
        result = proc.optimize_image(block, target_format="JPEG")
        assert result is block


class TestFixOrientation:
    """Test fix_orientation() with EXIF data."""

    def test_no_exif_unchanged(self):
        block = _make_image_block(200, 100)
        proc = MediaProcessor()
        result = proc.fix_orientation(block)
        # PNG has no EXIF; should be returned as-is (same object)
        assert result is block

    def test_jpeg_no_orientation_unchanged(self):
        """JPEG without orientation tag returns same block."""
        raw = _make_image_bytes(200, 100, fmt="JPEG")
        b64 = base64.b64encode(raw).decode()
        block = ImageBlock(
            media=MediaRef(kind="data", data_base64=b64, mime_type="image/jpeg", size_bytes=len(raw))
        )
        proc = MediaProcessor()
        result = proc.fix_orientation(block)
        assert result is block

    def test_url_block_skipped(self):
        block = ImageBlock(
            media=MediaRef(kind="url", url="https://example.com/img.jpg", mime_type="image/jpeg")
        )
        proc = MediaProcessor()
        result = proc.fix_orientation(block)
        assert result is block

    @pytest.mark.skipif(
        not _can_import("piexif"),
        reason="piexif not installed",
    )
    def test_exif_rotation_applied(self):
        block = _make_exif_rotated_image()
        proc = MediaProcessor()
        result = proc.fix_orientation(block)

        # After applying orientation 6 (90° CW), 200x100 becomes 100x200
        assert result.media.width == 100
        assert result.media.height == 200


class TestFullProcess:
    """Test the full_process() pipeline."""

    def test_full_process_validates_and_resizes(self):
        block = _make_image_block(4000, 3000)
        proc = MediaProcessor(MultimodalConfig(max_image_dimension=1024))
        result = proc.full_process(block)
        assert result.media.width <= 1024
        assert result.media.height <= 1024

    def test_full_process_rejects_unsupported_type(self):
        block = ImageBlock(
            media=MediaRef(
                kind="data",
                data_base64=base64.b64encode(b"fake").decode(),
                mime_type="image/bmp",
            )
        )
        proc = MediaProcessor()
        with pytest.raises(ValueError, match="Unsupported image type"):
            proc.full_process(block)


# ===========================================================================
# 6.4 — Security Hardening
# ===========================================================================

class TestMagicBytesValidation:
    """Test file type validation via magic bytes (not just extension)."""

    def test_valid_png_magic_bytes(self):
        from tenxgraph.storage.media.security import validate_magic_bytes

        raw = _make_image_bytes(10, 10, fmt="PNG")
        assert validate_magic_bytes(raw, "image/png") is True

    def test_valid_jpeg_magic_bytes(self):
        from tenxgraph.storage.media.security import validate_magic_bytes

        raw = _make_image_bytes(10, 10, fmt="JPEG")
        assert validate_magic_bytes(raw, "image/jpeg") is True

    def test_mismatched_magic_bytes(self):
        from tenxgraph.storage.media.security import validate_magic_bytes

        raw = _make_image_bytes(10, 10, fmt="PNG")
        # Claim it's JPEG but it's actually PNG
        assert validate_magic_bytes(raw, "image/jpeg") is False

    def test_unknown_mime_type_passes(self):
        from tenxgraph.storage.media.security import validate_magic_bytes

        assert validate_magic_bytes(b"some data", "application/octet-stream") is True

    def test_empty_data_fails(self):
        from tenxgraph.storage.media.security import validate_magic_bytes

        assert validate_magic_bytes(b"", "image/png") is False


class TestFilenameSanitization:
    """Test filename sanitization."""

    def test_sanitize_normal_filename(self):
        from tenxgraph.storage.media.security import sanitize_filename

        assert sanitize_filename("photo.jpg") == "photo.jpg"

    def test_sanitize_path_traversal(self):
        from tenxgraph.storage.media.security import sanitize_filename

        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_sanitize_null_bytes(self):
        from tenxgraph.storage.media.security import sanitize_filename

        result = sanitize_filename("image\x00.jpg")
        assert "\x00" not in result

    def test_sanitize_empty_string(self):
        from tenxgraph.storage.media.security import sanitize_filename

        result = sanitize_filename("")
        assert result == "unnamed"

    def test_sanitize_long_filename(self):
        from tenxgraph.storage.media.security import sanitize_filename

        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_sanitize_special_characters(self):
        from tenxgraph.storage.media.security import sanitize_filename

        result = sanitize_filename("my file (1) [copy].jpg")
        assert result  # Should not be empty


class TestFileSizeEnforcement:
    """Test max file size enforcement."""

    def test_enforce_size_passes(self):
        from tenxgraph.storage.media.security import enforce_file_size

        enforce_file_size(b"x" * 1000, max_mb=1.0)  # 1000 bytes < 1MB

    def test_enforce_size_rejects_oversized(self):
        from tenxgraph.storage.media.security import enforce_file_size

        with pytest.raises(ValueError, match="exceeds maximum"):
            enforce_file_size(b"x" * (2 * 1024 * 1024), max_mb=1.0)


# ===========================================================================
# 6.2 — Provider-specific Optimizations
# ===========================================================================


class TestProviderMediaCache:
    """Test the content-addressed provider media cache."""

    def test_content_key_deterministic(self):
        from tenxgraph.storage.media.provider_media import ProviderMediaCache

        cache = ProviderMediaCache()
        data = b"hello world"
        key1 = cache.content_key(data)
        key2 = cache.content_key(data)
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex digest

    def test_different_data_different_keys(self):
        from tenxgraph.storage.media.provider_media import ProviderMediaCache

        cache = ProviderMediaCache()
        assert cache.content_key(b"aaa") != cache.content_key(b"bbb")

    def test_put_and_get(self):
        from tenxgraph.storage.media.provider_media import ProviderMediaCache

        cache = ProviderMediaCache()
        cache.put("google", "abc123", {"uri": "gs://bucket/abc123"})
        assert cache.get("google", "abc123") == {"uri": "gs://bucket/abc123"}

    def test_get_miss_returns_none(self):
        from tenxgraph.storage.media.provider_media import ProviderMediaCache

        cache = ProviderMediaCache()
        assert cache.get("google", "nonexistent") is None

    def test_separate_providers(self):
        from tenxgraph.storage.media.provider_media import ProviderMediaCache

        cache = ProviderMediaCache()
        cache.put("google", "key1", "google_ref")
        cache.put("openai", "key1", "openai_ref")
        assert cache.get("google", "key1") == "google_ref"
        assert cache.get("openai", "key1") == "openai_ref"

    def test_eviction_at_capacity(self):
        from tenxgraph.storage.media.provider_media import ProviderMediaCache

        cache = ProviderMediaCache(max_entries=3)
        cache.put("google", "a", 1)
        cache.put("google", "b", 2)
        cache.put("google", "c", 3)
        cache.put("google", "d", 4)  # Should evict "a"

        assert cache.get("google", "a") is None
        assert cache.get("google", "d") == 4

    def test_clear_provider(self):
        from tenxgraph.storage.media.provider_media import ProviderMediaCache

        cache = ProviderMediaCache()
        cache.put("google", "k1", "v1")
        cache.put("openai", "k2", "v2")
        cache.clear("google")
        assert cache.get("google", "k1") is None
        assert cache.get("openai", "k2") == "v2"

    def test_clear_all(self):
        from tenxgraph.storage.media.provider_media import ProviderMediaCache

        cache = ProviderMediaCache()
        cache.put("google", "k1", "v1")
        cache.put("openai", "k2", "v2")
        cache.clear()
        assert cache.get("google", "k1") is None
        assert cache.get("openai", "k2") is None


class TestGoogleThresholdHelper:
    """Test Google File API threshold helpers."""

    def test_small_file_below_threshold(self):
        from tenxgraph.storage.media.provider_media import should_use_google_file_api

        assert should_use_google_file_api(1024) is False

    def test_large_file_above_threshold(self):
        from tenxgraph.storage.media.provider_media import should_use_google_file_api

        assert should_use_google_file_api(25 * 1024 * 1024) is True

    def test_exactly_at_threshold(self):
        from tenxgraph.storage.media.provider_media import (
            GOOGLE_INLINE_THRESHOLD,
            should_use_google_file_api,
        )

        assert should_use_google_file_api(GOOGLE_INLINE_THRESHOLD) is False
        assert should_use_google_file_api(GOOGLE_INLINE_THRESHOLD + 1) is True


class TestOpenAIFileHelpers:
    """Test OpenAI file attachment helper functions."""

    def test_create_file_search_tool(self):
        from tenxgraph.storage.media.provider_media import create_openai_file_search_tool

        tool = create_openai_file_search_tool(["file-abc123"])
        assert tool["type"] == "file_search"
        assert "file_search" in tool

    def test_create_file_attachment(self):
        from tenxgraph.storage.media.provider_media import create_openai_file_attachment

        att = create_openai_file_attachment("file-abc123")
        assert att["file_id"] == "file-abc123"
        assert att["tools"][0]["type"] == "file_search"

    def test_create_file_attachment_custom_tools(self):
        from tenxgraph.storage.media.provider_media import create_openai_file_attachment

        att = create_openai_file_attachment("file-abc123", tools=["code_interpreter"])
        assert att["tools"][0]["type"] == "code_interpreter"


# ===========================================================================
# 6.3 — Streaming multimodal verification
# ===========================================================================


class TestStreamingMultimodalSupport:
    """Verify that streaming converters handle multimodal content blocks."""

    def test_stream_chunk_can_hold_multimodal_message(self):
        """StreamChunk.message can carry messages with image blocks."""
        from tenxgraph.core.state.message import Message
        from tenxgraph.core.state.message_block import ImageBlock, MediaRef, TextBlock
        from tenxgraph.core.state.stream_chunks import StreamChunk

        msg = Message(
            role="assistant",
            content=[
                TextBlock(text="Here is the image:"),
                ImageBlock(media=MediaRef(kind="url", url="https://example.com/img.png", mime_type="image/png")),
            ],
        )
        chunk = StreamChunk(event="message", message=msg)
        assert chunk.message is not None
        assert len(chunk.message.content) == 2
        assert chunk.message.content[1].type == "image"

    def test_stream_chunk_serialization_with_image(self):
        """StreamChunk with image block serializes and deserializes correctly."""
        from tenxgraph.core.state.message import Message
        from tenxgraph.core.state.message_block import ImageBlock, MediaRef, TextBlock
        from tenxgraph.core.state.stream_chunks import StreamChunk

        msg = Message(
            role="assistant",
            content=[
                TextBlock(text="Generated:"),
                ImageBlock(
                    media=MediaRef(
                        kind="data",
                        data_base64=base64.b64encode(b"fake-image-data").decode(),
                        mime_type="image/png",
                    )
                ),
            ],
        )
        chunk = StreamChunk(event="message", message=msg)

        # Serialize
        data = chunk.model_dump(mode="json")
        assert data["message"]["content"][1]["type"] == "image"
        assert data["message"]["content"][1]["media"]["data_base64"] is not None

        # Deserialize
        restored = StreamChunk.model_validate(data)
        assert restored.message.content[1].type == "image"

    def test_converter_content_blocks_include_images(self):
        """Verify the converter's _build_content produces image_url parts."""
        from tenxgraph.core.state.message_block import ImageBlock, MediaRef, TextBlock
        from tenxgraph.utils.converter import _build_content

        blocks = [
            TextBlock(text="Look at this:"),
            ImageBlock(
                media=MediaRef(
                    kind="data",
                    data_base64=base64.b64encode(b"fake").decode(),
                    mime_type="image/jpeg",
                )
            ),
        ]
        result = _build_content(blocks)
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert "data:image/jpeg;base64," in result[1]["image_url"]["url"]
