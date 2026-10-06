"""Tests for the media capability matrix."""

import pytest

from tenxgraph.storage.media.capabilities import (
    MediaTransportMode,
    ModelMediaCapabilities,
    get_capabilities,
)


class TestMediaTransportMode:
    """Test the MediaTransportMode enum."""

    def test_enum_values(self):
        """Test that all expected transport modes exist."""
        assert MediaTransportMode.remote_url.value == "remote_url"  # noqa: S101
        assert MediaTransportMode.provider_file.value == "provider_file"  # noqa: S101
        assert MediaTransportMode.inline_bytes.value == "inline_bytes"  # noqa: S101
        assert MediaTransportMode.unsupported.value == "unsupported"  # noqa: S101

    def test_enum_is_string(self):
        """Test that enum values are strings."""
        assert isinstance(MediaTransportMode.remote_url, str)  # noqa: S101


class TestModelMediaCapabilities:
    """Test the ModelMediaCapabilities dataclass."""

    def test_basic_creation(self):
        """Test creating a capabilities entry."""
        cap = ModelMediaCapabilities(
            provider="openai",
            model_pattern="gpt-4o*",
            transport_order={
                "image": [MediaTransportMode.remote_url, MediaTransportMode.inline_bytes],
            },
            accepts_external_urls=True,
        )
        assert cap.provider == "openai"  # noqa: S101
        assert cap.model_pattern == "gpt-4o*"  # noqa: S101
        assert cap.accepts_external_urls is True  # noqa: S101

    def test_supports_media_type_true(self):
        """Test supports_media_type returns True for supported media."""
        cap = ModelMediaCapabilities(
            provider="openai",
            model_pattern="gpt-4o*",
            transport_order={
                "image": [MediaTransportMode.remote_url, MediaTransportMode.inline_bytes],
            },
        )
        assert cap.supports_media_type("image") is True  # noqa: S101

    def test_supports_media_type_false(self):
        """Test supports_media_type returns False for unsupported media."""
        cap = ModelMediaCapabilities(
            provider="openai",
            model_pattern="gpt-4",
            transport_order={
                "image": [MediaTransportMode.unsupported],
            },
        )
        assert cap.supports_media_type("image") is False  # noqa: S101

    def test_supports_media_type_missing_type(self):
        """Test supports_media_type returns False for unknown media types."""
        cap = ModelMediaCapabilities(
            provider="openai",
            model_pattern="gpt-4o*",
            transport_order={
                "image": [MediaTransportMode.remote_url],
            },
        )
        assert cap.supports_media_type("video") is False  # noqa: S101

    def test_get_transport_order(self):
        """Test get_transport_order returns correct order."""
        cap = ModelMediaCapabilities(
            provider="google",
            model_pattern="gemini-*",
            transport_order={
                "image": [MediaTransportMode.provider_file, MediaTransportMode.inline_bytes],
            },
        )
        order = cap.get_transport_order("image")
        assert order == [MediaTransportMode.provider_file, MediaTransportMode.inline_bytes]  # noqa: S101

    def test_get_transport_order_default(self):
        """Test get_transport_order returns unsupported for missing types."""
        cap = ModelMediaCapabilities(
            provider="openai",
            model_pattern="gpt-4o*",
            transport_order={},
        )
        order = cap.get_transport_order("audio")
        assert order == [MediaTransportMode.unsupported]  # noqa: S101

    def test_frozen_dataclass(self):
        """Test that the dataclass is immutable."""
        cap = ModelMediaCapabilities(
            provider="openai",
            model_pattern="gpt-4o*",
        )
        with pytest.raises(AttributeError):
            cap.provider = "google"  # type: ignore[misc]


class TestGetCapabilities:
    """Test the get_capabilities lookup function."""

    def test_openai_vision_model(self):
        """Test capabilities for OpenAI vision models."""
        cap = get_capabilities("openai", "gpt-4o")
        assert cap.provider == "openai"  # noqa: S101
        assert cap.accepts_external_urls is True  # noqa: S101
        assert cap.supports_media_type("image") is True  # noqa: S101
        order = cap.get_transport_order("image")
        assert MediaTransportMode.remote_url in order  # noqa: S101
        assert MediaTransportMode.inline_bytes in order  # noqa: S101

    def test_openai_vision_model_mini(self):
        """Test capabilities for gpt-4o-mini."""
        cap = get_capabilities("openai", "gpt-4o-mini")
        assert cap.supports_media_type("image") is True  # noqa: S101
        assert cap.accepts_external_urls is True  # noqa: S101

    def test_openai_vision_model_o1(self):
        """Test capabilities for o1 model."""
        cap = get_capabilities("openai", "o1")
        assert cap.supports_media_type("image") is True  # noqa: S101

    def test_openai_vision_model_o3(self):
        """Test capabilities for o3 model."""
        cap = get_capabilities("openai", "o3")
        assert cap.supports_media_type("image") is True  # noqa: S101

    def test_openai_vision_preview(self):
        """Test capabilities for gpt-4-vision-preview."""
        cap = get_capabilities("openai", "gpt-4-vision-preview")
        assert cap.supports_media_type("image") is True  # noqa: S101
        assert cap.accepts_external_urls is True  # noqa: S101

    def test_google_vision_model(self):
        """Test capabilities for Google Gemini models."""
        cap = get_capabilities("google", "gemini-1.5-pro")
        assert cap.provider == "google"  # noqa: S101
        assert cap.accepts_external_urls is False  # noqa: S101
        assert cap.supports_provider_file is True  # noqa: S101
        assert cap.supports_media_type("image") is True  # noqa: S101
        order = cap.get_transport_order("image")
        assert MediaTransportMode.provider_file == order[0]  # noqa: S101

    def test_google_vision_model_flash(self):
        """Test capabilities for gemini-1.5-flash."""
        cap = get_capabilities("google", "gemini-1.5-flash")
        assert cap.supports_media_type("image") is True  # noqa: S101
        assert cap.accepts_external_urls is False  # noqa: S101

    def test_text_only_model_gpt4(self):
        """Test capabilities for text-only gpt-4."""
        cap = get_capabilities("openai", "gpt-4")
        assert cap.supports_media_type("image") is False  # noqa: S101
        order = cap.get_transport_order("image")
        assert order == [MediaTransportMode.unsupported]  # noqa: S101

    def test_text_only_model_gpt35(self):
        """Test capabilities for gpt-3.5-turbo."""
        cap = get_capabilities("openai", "gpt-3.5-turbo")
        assert cap.supports_media_type("image") is False  # noqa: S101

    def test_text_only_model_gpt35_16k(self):
        """Test capabilities for gpt-3.5-turbo-16k."""
        cap = get_capabilities("openai", "gpt-3.5-turbo-16k")
        assert cap.supports_media_type("image") is False  # noqa: S101

    def test_unknown_model_returns_unsupported(self, caplog):
        """Test that unknown models get unsupported capabilities."""
        cap = get_capabilities("anthropic", "claude-3-opus")
        assert cap.supports_media_type("image") is False  # noqa: S101
        assert cap.provider == "anthropic"  # noqa: S101
        assert "No media capabilities found" in caplog.text  # noqa: S101

    def test_exact_match_model_pattern(self):
        """Test exact model pattern matching (gpt-4 should not match gpt-4o)."""
        cap = get_capabilities("openai", "gpt-4")
        assert cap.supports_media_type("image") is False  # noqa: S101

        cap_vision = get_capabilities("openai", "gpt-4o")
        assert cap_vision.supports_media_type("image") is True  # noqa: S101
