"""Tests for Sprint 5: OpenAI image delivery and message conversion."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.state.message_block import ImageBlock, MediaRef


class TestMediaRefResolverOpenAI:
    """Test MediaRefResolver.resolve_for_openai with capability-aware routing."""

    @pytest.fixture
    def resolver(self):
        from tenxgraph.storage.media.resolver import MediaRefResolver
        return MediaRefResolver()

    @pytest.mark.asyncio
    async def test_external_url_passes_through(self, resolver):
        """OpenAI accepts external URLs directly."""
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        result = await resolver.resolve_for_openai(ref, model="gpt-4o")

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"] == "https://example.com/image.png"  # noqa: S101

    @pytest.mark.asyncio
    async def test_inline_data_base64(self, resolver):
        """OpenAI handles inline base64 data."""
        ref = MediaRef(
            kind="data",
            data_base64="iVBORw0KGgo=",
            mime_type="image/png",
        )

        result = await resolver.resolve_for_openai(ref, model="gpt-4o")

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"].startswith("data:image/png;base64,")  # noqa: S101

    @pytest.mark.asyncio
    async def test_text_only_model_raises(self, resolver):
        """OpenAI text-only model raises UnsupportedMediaInputError."""
        from tenxgraph.core.exceptions.media_exceptions import UnsupportedMediaInputError

        ref = MediaRef(kind="url", url="https://example.com/image.png")

        with pytest.raises(UnsupportedMediaInputError) as exc_info:
            await resolver.resolve_for_openai(ref, model="gpt-4")

        assert exc_info.value.provider == "openai"  # noqa: S101
        assert exc_info.value.model == "gpt-4"  # noqa: S101

    @pytest.mark.asyncio
    async def test_legacy_path_without_model(self, resolver):
        """OpenAI without model param uses legacy path (backward compat)."""
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        result = await resolver.resolve_for_openai(ref)

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"] == "https://example.com/image.png"  # noqa: S101

    @pytest.mark.asyncio
    async def test_file_id_passes_through(self, resolver):
        """OpenAI handles file_id references."""
        ref = MediaRef(kind="file_id", file_id="file-abc123")

        result = await resolver.resolve_for_openai(ref, model="gpt-4o")

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"] == "file-abc123"  # noqa: S101


class TestImageBlockToOpenAI:
    """Test _image_block_to_openai in converter.py."""

    def test_data_block(self):
        """ImageBlock with base64 data converts to data URI."""
        from tenxgraph.utils.converter import _image_block_to_openai

        b64 = base64.b64encode(b"fake-png").decode()
        block = ImageBlock(
            media=MediaRef(
                kind="data",
                data_base64=b64,
                mime_type="image/png",
            )
        )

        result = _image_block_to_openai(block)

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"].startswith("data:image/png;base64,")  # noqa: S101

    def test_url_block(self):
        """ImageBlock with URL passes through."""
        from tenxgraph.utils.converter import _image_block_to_openai

        block = ImageBlock(
            media=MediaRef(
                kind="url",
                url="https://example.com/image.png",
                mime_type="image/png",
            )
        )

        result = _image_block_to_openai(block)

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"] == "https://example.com/image.png"  # noqa: S101

    def test_file_id_block(self):
        """ImageBlock with file_id passes through."""
        from tenxgraph.utils.converter import _image_block_to_openai

        block = ImageBlock(
            media=MediaRef(
                kind="file_id",
                file_id="file-abc123",
            )
        )

        result = _image_block_to_openai(block)

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"] == "file-abc123"  # noqa: S101


class TestMediaResolverOpenAI:
    """Test MediaResolver for OpenAI models — additional coverage beyond test_media_resolver.py."""

    @pytest.mark.asyncio
    async def test_gpt4o_mini_supports_images(self):
        """gpt-4o-mini supports images."""
        from tenxgraph.storage.media.media_resolver import MediaResolver

        resolver = MediaResolver()
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        result = await resolver.resolve(ref, provider="openai", model="gpt-4o-mini")

        assert result["type"] == "image_url"  # noqa: S101

    @pytest.mark.asyncio
    async def test_o1_supports_images(self):
        """o1 supports images."""
        from tenxgraph.storage.media.media_resolver import MediaResolver

        resolver = MediaResolver()
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        result = await resolver.resolve(ref, provider="openai", model="o1")

        assert result["type"] == "image_url"  # noqa: S101
