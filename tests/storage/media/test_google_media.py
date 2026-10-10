"""Tests for Sprint 4: Google-specific image delivery fixes."""

import base64
from unittest.mock import MagicMock, patch

import pytest


class TestImagePartToGoogle:
    """Test _image_part_to_google does not blindly use Part.from_uri."""

    @pytest.fixture
    def converter(self):
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        class TestConverter(AgentGoogleMixin):
            pass

        return TestConverter()

    def test_data_url_uses_from_bytes(self, converter):
        """Base64 data URLs should use Part.from_bytes."""
        from google.genai import types

        b64 = base64.b64encode(b"fake-png-data").decode()
        part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

        with patch.object(types.Part, "from_bytes") as mock_from_bytes:
            mock_part = MagicMock()
            mock_from_bytes.return_value = mock_part

            result = converter._image_part_to_google(part, types)

            assert len(result) == 1  # noqa: S101
            mock_from_bytes.assert_called_once()  # noqa: S101
            call_kwargs = mock_from_bytes.call_args
            assert call_kwargs[1]["mime_type"] == "image/png"  # noqa: S101

    def test_gs_url_uses_from_uri(self, converter):
        """gs:// URIs should use Part.from_uri."""
        from google.genai import types

        part = {"type": "image_url", "image_url": {"url": "gs://bucket/image.png"}}

        with patch.object(types.Part, "from_uri") as mock_from_uri:
            mock_part = MagicMock()
            mock_from_uri.return_value = mock_part

            result = converter._image_part_to_google(part, types)

            assert len(result) == 1  # noqa: S101
            mock_from_uri.assert_called_once()  # noqa: S101
            call_kwargs = mock_from_uri.call_args
            assert call_kwargs[1]["file_uri"] == "gs://bucket/image.png"  # noqa: S101

    def test_external_https_url_does_not_use_from_uri(self, converter):
        """External https:// URLs should NOT go to Part.from_uri."""
        from google.genai import types

        part = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}

        with patch.object(types.Part, "from_uri") as mock_from_uri:
            with patch.object(types.Part, "from_bytes") as mock_from_bytes:
                mock_part = MagicMock()
                mock_from_bytes.return_value = mock_part

                mock_response = MagicMock()
                mock_response.read.return_value = b"fake-image"
                mock_response.headers.get.return_value = "image/png"
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)

                with patch("urllib.request.urlopen", return_value=mock_response):
                    result = converter._image_part_to_google(part, types)

                mock_from_uri.assert_not_called()  # noqa: S101
                mock_from_bytes.assert_called_once()  # noqa: S101
                assert len(result) == 1  # noqa: S101

    def test_external_https_url_fetch_failure(self, converter):
        """If fetching external URL fails, return a text placeholder."""
        from google.genai import types

        part = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}

        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = converter._image_part_to_google(part, types)

        assert len(result) == 1  # noqa: S101
        assert result[0].text == "[Failed to load image]"  # noqa: S101

    def test_empty_url_returns_empty(self, converter):
        """Empty URL should return empty list."""
        from google.genai import types

        part = {"type": "image_url", "image_url": {"url": ""}}
        result = converter._image_part_to_google(part, types)
        assert result == []  # noqa: S101

    def test_invalid_image_info_returns_empty(self, converter):
        """Non-dict image_url should return empty list."""
        from google.genai import types

        part = {"type": "image_url", "image_url": "not-a-dict"}
        result = converter._image_part_to_google(part, types)
        assert result == []  # noqa: S101


class TestBinaryOrUriPartsToGoogle:
    """Test _binary_or_uri_parts_to_google for documents/video."""

    @pytest.fixture
    def converter(self):
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        class TestConverter(AgentGoogleMixin):
            pass

        return TestConverter()

    def test_data_uses_from_bytes(self, converter):
        """Base64 data should use Part.from_bytes."""
        from google.genai import types

        b64 = base64.b64encode(b"fake-pdf").decode()
        media_info = {"data": b64, "mime_type": "application/pdf"}

        with patch.object(types.Part, "from_bytes") as mock_from_bytes:
            mock_part = MagicMock()
            mock_from_bytes.return_value = mock_part

            result = converter._binary_or_uri_parts_to_google(media_info, types, default_mime="application/pdf")

            assert len(result) == 1  # noqa: S101
            mock_from_bytes.assert_called_once()  # noqa: S101

    def test_gs_url_uses_from_uri(self, converter):
        """gs:// URIs should use Part.from_uri."""
        from google.genai import types

        media_info = {"url": "gs://bucket/doc.pdf", "mime_type": "application/pdf"}

        with patch.object(types.Part, "from_uri") as mock_from_uri:
            mock_part = MagicMock()
            mock_from_uri.return_value = mock_part

            result = converter._binary_or_uri_parts_to_google(media_info, types, default_mime="application/pdf")

            assert len(result) == 1  # noqa: S101
            mock_from_uri.assert_called_once()  # noqa: S101

    def test_external_https_does_not_use_from_uri(self, converter):
        """External https:// URLs should NOT go to Part.from_uri."""
        from google.genai import types

        media_info = {"url": "https://example.com/doc.pdf", "mime_type": "application/pdf"}

        with patch.object(types.Part, "from_uri") as mock_from_uri:
            with patch.object(types.Part, "from_bytes") as mock_from_bytes:
                mock_part = MagicMock()
                mock_from_bytes.return_value = mock_part

                mock_response = MagicMock()
                mock_response.read.return_value = b"fake-pdf"
                mock_response.headers.get.return_value = "application/pdf"
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)

                with patch("urllib.request.urlopen", return_value=mock_response):
                    result = converter._binary_or_uri_parts_to_google(media_info, types, default_mime="application/pdf")

                mock_from_uri.assert_not_called()  # noqa: S101
                mock_from_bytes.assert_called_once()  # noqa: S101

    def test_no_data_no_url_returns_empty(self, converter):
        """No data or URL should return empty list."""
        from google.genai import types

        media_info = {"mime_type": "application/pdf"}
        result = converter._binary_or_uri_parts_to_google(media_info, types, default_mime="application/pdf")
        assert result == []  # noqa: S101
