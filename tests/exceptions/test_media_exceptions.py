"""Tests for the UnsupportedMediaInputError exception."""

import pytest

from tenxgraph.core.exceptions.media_exceptions import UnsupportedMediaInputError
from tenxgraph.storage.media.capabilities import MediaTransportMode


class TestUnsupportedMediaInputError:
    """Test the UnsupportedMediaInputError exception."""

    def test_basic_creation(self):
        """Test creating the error with minimal arguments."""
        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="url",
        )
        assert error.provider == "openai"  # noqa: S101
        assert error.model == "gpt-4"  # noqa: S101
        assert error.media_type == "image"  # noqa: S101
        assert error.source_kind == "url"  # noqa: S101
        assert error.transports_attempted == []  # noqa: S101

    def test_with_transports_attempted(self):
        """Test creating the error with transport attempts."""
        error = UnsupportedMediaInputError(
            provider="google",
            model="gemini-1.5-pro",
            media_type="image",
            source_kind="url",
            transports_attempted=[
                MediaTransportMode.provider_file,
                MediaTransportMode.inline_bytes,
            ],
        )
        assert len(error.transports_attempted) == 2  # noqa: S101
        assert MediaTransportMode.provider_file in error.transports_attempted  # noqa: S101

    def test_default_message_for_image(self):
        """Test the default message for unsupported image input."""
        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="url",
        )
        assert "gpt-4" in error.message  # noqa: S101
        assert "openai" in error.message  # noqa: S101
        assert "image" in error.message  # noqa: S101
        assert "vision-capable" in error.message  # noqa: S101

    def test_default_message_for_url_suggestion(self):
        """Test the URL suggestion in the error message."""
        error = UnsupportedMediaInputError(
            provider="google",
            model="gemini-1.5-pro",
            media_type="image",
            source_kind="url",
        )
        assert "vision-capable" in error.message.lower()  # noqa: S101

    def test_custom_message(self):
        """Test creating the error with a custom message."""
        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="data",
            message="Custom error message",
        )
        assert error.message == "Custom error message"  # noqa: S101

    def test_str_representation(self):
        """Test string representation."""
        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="url",
        )
        assert str(error) == error.message  # noqa: S101

    def test_repr_representation(self):
        """Test repr representation."""
        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="url",
        )
        repr_str = repr(error)
        assert "UnsupportedMediaInputError" in repr_str  # noqa: S101
        assert "openai" in repr_str  # noqa: S101
        assert "gpt-4" in repr_str  # noqa: S101

    def test_to_dict(self):
        """Test converting the error to a dictionary."""
        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="url",
            transports_attempted=[MediaTransportMode.remote_url],
        )
        result = error.to_dict()
        assert result == {  # noqa: S101
            "error_type": "UnsupportedMediaInputError",
            "provider": "openai",
            "model": "gpt-4",
            "media_type": "image",
            "source_kind": "url",
            "transports_attempted": ["remote_url"],
            "message": error.message,
        }

    def test_inherits_from_exception(self):
        """Test that the error inherits from Exception."""
        error = UnsupportedMediaInputError(
            provider="openai",
            model="gpt-4",
            media_type="image",
            source_kind="url",
        )
        assert isinstance(error, Exception)  # noqa: S101

    def test_can_be_raised(self):
        """Test that the error can be raised and caught."""
        with pytest.raises(UnsupportedMediaInputError) as exc_info:
            raise UnsupportedMediaInputError(
                provider="openai",
                model="gpt-4",
                media_type="image",
                source_kind="url",
            )

        assert exc_info.value.provider == "openai"  # noqa: S101
        assert exc_info.value.model == "gpt-4"  # noqa: S101

    def test_different_source_kinds(self):
        """Test with different source kinds."""
        for kind in ["url", "file_id", "data", "internal_ref"]:
            error = UnsupportedMediaInputError(
                provider="openai",
                model="gpt-4",
                media_type="image",
                source_kind=kind,
            )
            assert error.source_kind == kind  # noqa: S101

    def test_transports_in_message(self):
        """Test that attempted transports appear in the message."""
        error = UnsupportedMediaInputError(
            provider="google",
            model="gemini-1.5-pro",
            media_type="image",
            source_kind="url",
            transports_attempted=[
                MediaTransportMode.provider_file,
                MediaTransportMode.inline_bytes,
            ],
        )
        assert "provider_file" in error.message  # noqa: S101
        assert "inline_bytes" in error.message  # noqa: S101

    def test_error_chain(self):
        """Test exception chaining."""
        original = ValueError("Original error")
        try:
            raise UnsupportedMediaInputError(
                provider="openai",
                model="gpt-4",
                media_type="image",
                source_kind="url",
            ) from original
        except UnsupportedMediaInputError as media_error:
            assert media_error.__cause__ == original  # noqa: S101
