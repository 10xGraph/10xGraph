"""Tests for Sprint 2 – MediaProcessor (image validation & resize).

These tests live in PyAgenity because MediaProcessor is a core library component.
Document extraction tests live in pyagenity-api.
"""

import base64
import io

import pytest

from tenxgraph.storage.media.config import MultimodalConfig
from tenxgraph.storage.media.processor import MediaProcessor, _pil_format
from tenxgraph.core.state.message_block import ImageBlock, MediaRef


# ---------------------------------------------------------------------------
# _pil_format helper
# ---------------------------------------------------------------------------


class TestPilFormat:
    def test_jpeg(self):
        assert _pil_format("image/jpeg") == "JPEG"

    def test_png(self):
        assert _pil_format("image/png") == "PNG"

    def test_webp(self):
        assert _pil_format("image/webp") == "WEBP"

    def test_gif(self):
        assert _pil_format("image/gif") == "GIF"

    def test_unknown_defaults_png(self):
        assert _pil_format("image/bmp") == "PNG"

    def test_none_defaults_png(self):
        assert _pil_format(None) == "PNG"


# ---------------------------------------------------------------------------
# MediaProcessor.validate_image
# ---------------------------------------------------------------------------


class TestValidateImage:
    def test_valid_jpeg(self):
        block = ImageBlock(media=MediaRef(kind="url", url="https://x.com/a.jpg", mime_type="image/jpeg"))
        proc = MediaProcessor()
        proc.validate_image(block)  # should not raise

    def test_valid_png(self):
        block = ImageBlock(media=MediaRef(kind="data", data_base64="abc", mime_type="image/png"))
        proc = MediaProcessor()
        proc.validate_image(block)  # should not raise

    def test_unsupported_mime_type(self):
        block = ImageBlock(media=MediaRef(kind="url", url="https://x.com/a.bmp", mime_type="image/bmp"))
        proc = MediaProcessor()
        with pytest.raises(ValueError, match="Unsupported image type"):
            proc.validate_image(block)

    def test_custom_supported_types(self):
        cfg = MultimodalConfig(supported_image_types={"image/bmp", "image/png"})
        block = ImageBlock(media=MediaRef(kind="url", url="https://x.com/a.bmp", mime_type="image/bmp"))
        proc = MediaProcessor(config=cfg)
        proc.validate_image(block)  # should not raise

    def test_size_bytes_too_large(self):
        block = ImageBlock(
            media=MediaRef(kind="url", url="https://x.com/a.jpg", mime_type="image/jpeg", size_bytes=20_000_000)
        )
        proc = MediaProcessor(config=MultimodalConfig(max_image_size_mb=10.0))
        with pytest.raises(ValueError, match="Image too large"):
            proc.validate_image(block)

    def test_size_bytes_within_limit(self):
        block = ImageBlock(
            media=MediaRef(kind="url", url="https://x.com/a.jpg", mime_type="image/jpeg", size_bytes=5_000_000)
        )
        proc = MediaProcessor(config=MultimodalConfig(max_image_size_mb=10.0))
        proc.validate_image(block)  # should not raise

    def test_base64_data_too_large(self):
        large_b64 = "A" * 20_000_000  # ~15MB decoded
        block = ImageBlock(
            media=MediaRef(kind="data", data_base64=large_b64, mime_type="image/png")
        )
        proc = MediaProcessor(config=MultimodalConfig(max_image_size_mb=10.0))
        with pytest.raises(ValueError, match="Image data too large"):
            proc.validate_image(block)

    def test_default_mime_type_is_png(self):
        block = ImageBlock(media=MediaRef(kind="data", data_base64="abc"))
        proc = MediaProcessor()
        proc.validate_image(block)  # should not raise


# ---------------------------------------------------------------------------
# MediaProcessor.resize_image
# ---------------------------------------------------------------------------


class TestResizeImage:
    def test_url_block_not_resized(self):
        block = ImageBlock(media=MediaRef(kind="url", url="https://x.com/a.jpg", mime_type="image/jpeg"))
        proc = MediaProcessor()
        result = proc.resize_image(block)
        assert result is block

    def test_file_id_block_not_resized(self):
        block = ImageBlock(media=MediaRef(kind="file_id", file_id="f-123", mime_type="image/png"))
        proc = MediaProcessor()
        result = proc.resize_image(block)
        assert result is block

    def test_small_image_not_resized(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        block = ImageBlock(media=MediaRef(kind="data", data_base64=b64, mime_type="image/png"))
        proc = MediaProcessor(config=MultimodalConfig(max_image_dimension=2048))
        result = proc.resize_image(block)
        assert result is block

    def test_large_image_is_resized(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (4000, 3000), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        block = ImageBlock(media=MediaRef(kind="data", data_base64=b64, mime_type="image/png"))
        proc = MediaProcessor(config=MultimodalConfig(max_image_dimension=1024))
        result = proc.resize_image(block)

        assert result is not block
        assert result.media.width is not None
        assert result.media.height is not None
        assert result.media.width <= 1024
        assert result.media.height <= 1024
        assert abs(result.media.width / result.media.height - 4000 / 3000) < 0.01

    def test_resize_preserves_alt_text(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (4000, 4000), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        block = ImageBlock(
            media=MediaRef(kind="data", data_base64=b64, mime_type="image/jpeg"),
            alt_text="test alt",
        )
        proc = MediaProcessor(config=MultimodalConfig(max_image_dimension=512))
        result = proc.resize_image(block)
        assert result.alt_text == "test alt"

    def test_no_pillow_skips_resize(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "PIL" or name == "PIL.Image":
                raise ImportError("No module named 'PIL'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        block = ImageBlock(
            media=MediaRef(kind="data", data_base64="abc123", mime_type="image/png")
        )
        proc = MediaProcessor()
        result = proc.resize_image(block)
        assert result is block


# ---------------------------------------------------------------------------
# MediaProcessor.process (validate + resize)
# ---------------------------------------------------------------------------


class TestMediaProcessorProcess:
    def test_process_validates_and_returns(self):
        block = ImageBlock(media=MediaRef(kind="url", url="https://x.com/a.jpg", mime_type="image/jpeg"))
        proc = MediaProcessor()
        result = proc.process(block)
        assert result is block

    def test_process_rejects_unsupported(self):
        block = ImageBlock(media=MediaRef(kind="url", url="https://x.com/a.tiff", mime_type="image/tiff"))
        proc = MediaProcessor()
        with pytest.raises(ValueError, match="Unsupported image type"):
            proc.process(block)

    def test_process_validates_then_resizes(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (5000, 5000), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        block = ImageBlock(media=MediaRef(kind="data", data_base64=b64, mime_type="image/png"))
        proc = MediaProcessor(config=MultimodalConfig(max_image_dimension=256))
        result = proc.process(block)
        assert result is not block
        assert result.media.width <= 256
        assert result.media.height <= 256


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


class TestMediaProcessorDefaults:
    def test_default_config(self):
        proc = MediaProcessor()
        assert proc.config.max_image_size_mb == 10.0
        assert proc.config.max_image_dimension == 2048
        assert "image/jpeg" in proc.config.supported_image_types

    def test_custom_config(self):
        cfg = MultimodalConfig(max_image_size_mb=5.0, max_image_dimension=1024)
        proc = MediaProcessor(config=cfg)
        assert proc.config.max_image_size_mb == 5.0
        assert proc.config.max_image_dimension == 1024
