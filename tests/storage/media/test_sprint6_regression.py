"""Tests for Sprint 6: API integration, cleanup, and regression."""

from unittest.mock import MagicMock, patch

import pytest


class TestTempCacheCleanupOnStartup:
    """Test that temp cache cleanup runs on API startup."""

    @pytest.mark.asyncio
    async def test_cleanup_returns_zero_when_no_entries(self):
        """Cleanup should return 0 when no expired entries exist."""
        from tenxgraph.storage.media.temp_cache import TemporaryMediaCache

        class FakeCheckpointer:
            async def alist_cache_keys(self, namespace, prefix=None):
                return []

        cache = TemporaryMediaCache()
        checkpointer = FakeCheckpointer()
        count = await cache.cleanup(checkpointer, None)
        assert count == 0  # noqa: S101

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_media_store(self):
        """Cleanup should be safe when no media store is available."""
        import time
        from tenxgraph.storage.media.temp_cache import (
            TEMP_CACHE_NAMESPACE,
            TempCacheEntry,
            TemporaryMediaCache,
        )

        class FakeCheckpointer:
            def __init__(self):
                self._cache = {}
            async def alist_cache_keys(self, namespace, prefix=None):
                ns_prefix = f"{namespace}:"
                return [k[len(ns_prefix):] for k in self._cache if k.startswith(ns_prefix)]
            async def aget_cache_value(self, namespace, key):
                return self._cache.get(f"{namespace}:{key}")
            async def aclear_cache_value(self, namespace, key):
                return self._cache.pop(f"{namespace}:{key}", None)
            async def aput_cache_value(self, namespace, key, value, ttl_seconds=None):
                self._cache[f"{namespace}:{key}"] = value

        cache = TemporaryMediaCache(ttl_seconds=3600)
        checkpointer = FakeCheckpointer()

        # Add an expired entry
        now = time.time()
        expired_entry = TempCacheEntry(
            content_hash="old",
            source_url="https://example.com/old.png",
            mime_type="image/png",
            storage_key="temp/old.png",
            created_at=now - 7200,
            expires_at=now - 3600,
        )
        cache._index["temp/old.png"] = expired_entry
        cache._loaded = True
        await checkpointer.aput_cache_value(
            TEMP_CACHE_NAMESPACE, "temp/old.png", expired_entry.to_dict()
        )

        # Cleanup with no media store should still succeed (best-effort)
        count = await cache.cleanup(checkpointer, None)
        assert count == 1  # noqa: S101


class TestUnsupportedMediaInputErrorHandling:
    """Test that UnsupportedMediaInputError is surfaced correctly."""

    def test_error_to_dict(self):
        """Error should be serializable for API responses."""
        from tenxgraph.core.exceptions.media_exceptions import UnsupportedMediaInputError

        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="url",
        )
        data = error.to_dict()
        assert data["error_type"] == "UnsupportedMediaInputError"  # noqa: S101
        assert data["provider"] == "openai"  # noqa: S101
        assert data["model"] == "gpt-4"  # noqa: S101
        assert "vision-capable" in data["message"]  # noqa: S101

    def test_error_message_is_actionable(self):
        """Error message should guide the user to a fix."""
        from tenxgraph.core.exceptions.media_exceptions import UnsupportedMediaInputError

        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="url",
        )
        assert "gpt-4" in error.message  # noqa: S101
        assert "vision-capable" in error.message  # noqa: S101
        assert "gpt-4o" in error.message  # noqa: S101


class TestRegressionFileIdWorkflow:
    """Test that existing file_id workflow still works."""

    @pytest.mark.asyncio
    async def test_openai_file_id_passes_through(self):
        """file_id references should work for OpenAI vision models."""
        from tenxgraph.core.state.message_block import MediaRef
        from tenxgraph.storage.media.resolver import MediaRefResolver

        resolver = MediaRefResolver()
        ref = MediaRef(kind="file_id", file_id="file-abc123")

        result = await resolver.resolve_for_openai(ref, model="gpt-4o")

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"] == "file-abc123"  # noqa: S101


class TestRegressionSignedUrlPath:
    """Test that OpenAI signed URL path still works."""

    @pytest.mark.asyncio
    async def test_internal_ref_with_signed_url(self):
        """Internal refs should resolve to signed URLs for OpenAI."""
        from tenxgraph.core.state.message_block import MediaRef
        from tenxgraph.storage.media.resolver import MediaRefResolver

        class FakeStore:
            async def get_direct_url(self, key, **kwargs):
                return f"https://signed.example.com/{key}"

            async def retrieve(self, key):
                return b"fake-data", "image/png"

        resolver = MediaRefResolver(media_store=FakeStore())
        ref = MediaRef(
            kind="url",
            url="agentflow://media/test-key",
            mime_type="image/png",
        )

        result = await resolver.resolve_for_openai(ref, model="gpt-4o")

        assert result["type"] == "image_url"  # noqa: S101
        assert "signed" in result["image_url"]["url"]  # noqa: S101


class TestRegressionMultimodalConversion:
    """Test that message conversion still works for all cases."""

    def test_image_block_to_openai(self):
        from tenxgraph.core.state.message_block import ImageBlock, MediaRef
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

    def test_strip_media_blocks(self):
        from tenxgraph.utils.converter import strip_media_blocks

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
        ]

        result = strip_media_blocks(messages)

        assert len(result) == 1  # noqa: S101
        assert result[0]["content"] == "Hello"  # noqa: S101
