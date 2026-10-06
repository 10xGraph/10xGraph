"""Tests for the unified MediaResolver with capability-based fallback."""

from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.exceptions.media_exceptions import UnsupportedMediaInputError
from tenxgraph.core.state.message_block import MediaRef
from tenxgraph.storage.media.capabilities import MediaTransportMode
from tenxgraph.storage.media.media_resolver import MediaResolver, _source_kind


class FakeMediaStore:
    """Minimal fake media store for testing."""

    def __init__(self, data: dict[str, tuple[bytes, str]] | None = None):
        self._data = data or {}
        self._direct_urls: dict[str, str] = {}

    def add(self, key: str, data: bytes, mime: str, direct_url: str | None = None):
        self._data[key] = (data, mime)
        if direct_url:
            self._direct_urls[key] = direct_url

    async def retrieve(self, key: str) -> tuple[bytes, str]:
        if key not in self._data:
            raise KeyError(f"Media key not found: {key}")
        return self._data[key]

    async def get_direct_url(self, key: str, **kwargs) -> str | None:
        return self._direct_urls.get(key)


class FakeCacheBackend:
    """Minimal fake cache backend for testing."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    async def aget_cache_value(self, namespace: str, key: str):
        return self._store.get(f"{namespace}:{key}")

    async def aput_cache_value(self, namespace: str, key: str, value: dict, ttl_seconds: int | None = None):
        self._store[f"{namespace}:{key}"] = value


class TestSourceKind:
    """Test the _source_kind helper."""

    def test_external_url(self):
        ref = MediaRef(kind="url", url="https://example.com/image.png")
        assert _source_kind(ref) == "url"  # noqa: S101

    def test_internal_ref(self):
        ref = MediaRef(kind="url", url="agentflow://media/abc123")
        assert _source_kind(ref) == "internal_ref"  # noqa: S101

    def test_data(self):
        ref = MediaRef(kind="data", data_base64="abc")
        assert _source_kind(ref) == "data"  # noqa: S101

    def test_file_id(self):
        ref = MediaRef(kind="file_id", file_id="file-123")
        assert _source_kind(ref) == "file_id"  # noqa: S101


class TestMediaResolverUnsupported:
    """Test that unsupported models raise UnsupportedMediaInputError."""

    @pytest.mark.asyncio
    async def test_text_only_model_raises(self):
        resolver = MediaResolver()
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        with pytest.raises(UnsupportedMediaInputError) as exc_info:
            await resolver.resolve(ref, provider="openai", model="gpt-4")

        assert exc_info.value.provider == "openai"  # noqa: S101
        assert exc_info.value.model == "gpt-4"  # noqa: S101
        assert exc_info.value.media_type == "image"  # noqa: S101

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        resolver = MediaResolver()
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        with pytest.raises(UnsupportedMediaInputError):
            await resolver.resolve(ref, provider="unknown", model="some-model")


class TestMediaResolverOpenAI:
    """Test MediaResolver for OpenAI models."""

    @pytest.fixture
    def resolver(self):
        store = FakeMediaStore()
        store.add(
            "test-key",
            b"\x89PNG\r\n\x1a\n",
            "image/png",
            direct_url="https://signed.example.com/image.png",
        )
        return MediaResolver(media_store=store)

    @pytest.mark.asyncio
    async def test_external_url_passes_through(self):
        """OpenAI accepts external URLs directly."""
        resolver = MediaResolver()
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        result = await resolver.resolve(ref, provider="openai", model="gpt-4o")

        assert result == {  # noqa: S101
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }

    @pytest.mark.asyncio
    async def test_internal_ref_uses_signed_url(self, resolver):
        """OpenAI uses signed URL for internal refs."""
        ref = MediaRef(
            kind="url",
            url="agentflow://media/test-key",
            mime_type="image/png",
        )

        result = await resolver.resolve(ref, provider="openai", model="gpt-4o")

        assert result["type"] == "image_url"  # noqa: S101
        assert "signed" in result["image_url"]["url"] or "example.com" in result["image_url"]["url"]  # noqa: S101

    @pytest.mark.asyncio
    async def test_inline_data_base64(self):
        """OpenAI handles inline base64 data."""
        resolver = MediaResolver()
        ref = MediaRef(
            kind="data",
            data_base64="iVBORw0KGgo=",
            mime_type="image/png",
        )

        result = await resolver.resolve(ref, provider="openai", model="gpt-4o")

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"].startswith("data:image/png;base64,")  # noqa: S101


class TestMediaResolverGoogle:
    """Test MediaResolver for Google Gemini models."""

    @pytest.mark.asyncio
    async def test_external_url_does_not_pass_through(self):
        """Google does NOT accept arbitrary external URLs — should fall back to inline bytes."""
        resolver = MediaResolver()
        ref = MediaRef(
            kind="data",
            data_base64="iVBORw0KGgo=",
            mime_type="image/png",
        )

        with patch("google.genai.types.Part") as MockPart:
            mock_part = MagicMock()
            MockPart.from_bytes.return_value = mock_part

            result = await resolver.resolve(ref, provider="google", model="gemini-1.5-pro")

            MockPart.from_bytes.assert_called_once()  # noqa: S101


class TestMediaRefResolverCapabilityAware:
    """Test MediaRefResolver with capability-aware resolution."""

    @pytest.fixture
    def resolver(self):
        store = FakeMediaStore()
        store.add(
            "test-key",
            b"\x89PNG\r\n\x1a\n",
            "image/png",
            direct_url="https://signed.example.com/image.png",
        )
        from tenxgraph.storage.media.resolver import MediaRefResolver
        return MediaRefResolver(media_store=store)

    @pytest.mark.asyncio
    async def test_openai_with_model_uses_capabilities(self, resolver):
        """OpenAI with model param uses capability-based resolution."""
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        result = await resolver.resolve_for_openai(ref, model="gpt-4o")

        assert result["type"] == "image_url"  # noqa: S101
        assert result["image_url"]["url"] == "https://example.com/image.png"  # noqa: S101

    @pytest.mark.asyncio
    async def test_openai_text_only_raises(self, resolver):
        """OpenAI text-only model raises UnsupportedMediaInputError."""
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        with pytest.raises(UnsupportedMediaInputError):
            await resolver.resolve_for_openai(ref, model="gpt-4")

    @pytest.mark.asyncio
    async def test_openai_without_model_uses_legacy(self, resolver):
        """OpenAI without model param uses legacy path (backward compat)."""
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        result = await resolver.resolve_for_openai(ref)

        assert result["type"] == "image_url"  # noqa: S101

    @pytest.mark.asyncio
    async def test_google_with_model_uses_capabilities(self, resolver):
        """Google with model param uses capability-based resolution."""
        ref = MediaRef(
            kind="data",
            data_base64="iVBORw0KGgo=",
            mime_type="image/png",
        )

        with patch("google.genai.types.Part") as MockPart:
            mock_part = MagicMock()
            MockPart.from_bytes.return_value = mock_part

            result = await resolver.resolve_for_google(ref, model="gemini-1.5-pro")

            MockPart.from_bytes.assert_called_once()  # noqa: S101

    @pytest.mark.asyncio
    async def test_google_text_only_raises(self, resolver):
        """Google text-only model raises UnsupportedMediaInputError."""
        ref = MediaRef(kind="url", url="https://example.com/image.png")

        with pytest.raises(UnsupportedMediaInputError):
            await resolver.resolve_for_google(ref, model="gpt-4")

    @pytest.mark.asyncio
    async def test_google_without_model_uses_legacy(self, resolver):
        """Google without model param uses legacy path (backward compat)."""
        ref = MediaRef(
            kind="data",
            data_base64="iVBORw0KGgo=",
            mime_type="image/png",
        )

        with patch("google.genai.types.Part") as MockPart:
            mock_part = MagicMock()
            MockPart.from_bytes.return_value = mock_part

            result = await resolver.resolve_for_google(ref)

            MockPart.from_bytes.assert_called_once()  # noqa: S101
