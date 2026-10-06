"""Tests for CloudMediaStore (S3/GCS via cloud-storage-manager)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.storage.media.storage.cloud_store import CloudMediaStore, _mime_to_ext


# ---------------------------------------------------------------------------
# _mime_to_ext helper
# ---------------------------------------------------------------------------


class TestMimeToExt:
    def test_known_mime_types(self):
        assert _mime_to_ext("image/jpeg") in (".jpg", ".jpeg")
        assert _mime_to_ext("image/png") == ".png"
        assert _mime_to_ext("application/pdf") == ".pdf"
        assert _mime_to_ext("audio/wav") == ".wav"

    def test_fallback_types(self):
        assert _mime_to_ext("audio/mp3") == ".mp3"
        assert _mime_to_ext("image/webp") == ".webp"

    def test_unknown_type(self):
        assert _mime_to_ext("application/x-unknown-thing") == ".bin"


# ---------------------------------------------------------------------------
# CloudMediaStore
# ---------------------------------------------------------------------------


class TestCloudMediaStore:
    """Tests for CloudMediaStore using a mocked cloud storage backend."""

    @pytest.fixture()
    def mock_storage(self):
        """Create a mock BaseCloudStorage."""
        storage = AsyncMock()
        storage.upload = AsyncMock(side_effect=lambda fp, cp: cp)
        storage.delete = AsyncMock(return_value=True)
        storage.get_public_url = AsyncMock(return_value="https://signed-url.example.com/blob")
        return storage

    @pytest.fixture()
    def store(self, mock_storage):
        return CloudMediaStore(mock_storage, prefix="test-media")

    # ---- store -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_store_returns_hex_key(self, store):
        key = await store.store(b"hello", "text/plain")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    @pytest.mark.asyncio
    async def test_store_uploads_blob_and_meta(self, store, mock_storage):
        key = await store.store(b"\x89PNG", "image/png", {"tag": "avatar"})
        # Should have called upload twice: blob + meta sidecar
        assert mock_storage.upload.call_count == 2
        blob_call = mock_storage.upload.call_args_list[0]
        meta_call = mock_storage.upload.call_args_list[1]

        # Blob path includes key and extension
        blob_cloud_path = blob_call.args[1]
        assert key in blob_cloud_path
        assert blob_cloud_path.startswith("test-media/")
        assert blob_cloud_path.endswith(".png")

        # Meta path ends with .meta.json
        meta_cloud_path = meta_call.args[1]
        assert meta_cloud_path.endswith(".meta.json")
        assert key in meta_cloud_path

    @pytest.mark.asyncio
    async def test_store_temp_files_cleaned_up(self, store, mock_storage):
        """Verify temp files are deleted after upload."""
        uploaded_paths = []

        async def capture_upload(fp, cp):
            uploaded_paths.append(fp)
            return cp

        mock_storage.upload.side_effect = capture_upload
        await store.store(b"data", "image/jpeg")

        # Both temp files should have been cleaned up
        for fp in uploaded_paths:
            assert not Path(fp).exists()

    @pytest.mark.asyncio
    async def test_store_cleans_up_on_upload_error(self, store, mock_storage):
        """Verify temp file is cleaned up even if upload fails."""
        mock_storage.upload.side_effect = Exception("upload failed")
        with pytest.raises(Exception, match="upload failed"):
            await store.store(b"data", "image/jpeg")

    # ---- retrieve --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_retrieve_downloads_from_signed_url(self, store, mock_storage):
        # First store something
        key = await store.store(b"test-content", "text/plain")
        mock_storage.upload.reset_mock()

        # Mock the download flow
        meta = {"mime_type": "text/plain", "size_bytes": 12, "ext": ".txt"}
        meta_url = "https://signed-url.example.com/meta"
        blob_url = "https://signed-url.example.com/blob"

        call_count = [0]

        async def mock_get_url(cloud_path, expiration=3600):
            call_count[0] += 1
            if cloud_path.endswith(".meta.json"):
                return meta_url
            return blob_url

        mock_storage.get_public_url.side_effect = mock_get_url

        with patch.object(
            CloudMediaStore,
            "_download_from_url",
            new_callable=AsyncMock,
            side_effect=lambda url: json.dumps(meta).encode()
            if "meta" in url
            else b"test-content",
        ):
            data, mime = await store.retrieve(key)
            assert data == b"test-content"
            assert mime == "text/plain"

    @pytest.mark.asyncio
    async def test_retrieve_not_found(self, store, mock_storage):
        """Retrieve for unknown key should raise KeyError."""
        mock_storage.get_public_url.side_effect = Exception("not found")
        with pytest.raises(KeyError, match="Media not found"):
            await store.retrieve("nonexistent_key_00000000000000")

    # ---- delete ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_removes_blob_and_meta(self, store, mock_storage):
        key = await store.store(b"data", "image/png")
        mock_storage.upload.reset_mock()

        # Mock _download_meta to return valid meta
        meta = {"mime_type": "image/png", "size_bytes": 4, "ext": ".png"}
        with patch.object(store, "_download_meta", new_callable=AsyncMock, return_value=meta):
            result = await store.delete(key)
            assert result is True

        # Should have called delete twice (blob + meta)
        assert mock_storage.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_not_found_returns_false(self, store, mock_storage):
        mock_storage.get_public_url.side_effect = Exception("not found")
        result = await store.delete("nonexistent000000000000000000000")
        assert result is False

    # ---- exists ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_exists_true(self, store, mock_storage):
        meta = {"mime_type": "image/png", "size_bytes": 4, "ext": ".png"}
        with patch.object(store, "_download_meta", new_callable=AsyncMock, return_value=meta):
            assert await store.exists("somekey00000000000000000000000") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, store, mock_storage):
        with patch.object(store, "_download_meta", new_callable=AsyncMock, return_value=None):
            assert await store.exists("nonexistent000000000000000000000") is False

    @pytest.mark.asyncio
    async def test_get_metadata(self, store):
        meta = {"mime_type": "image/png", "size_bytes": 4, "ext": ".png"}
        with patch.object(store, "_download_meta", new_callable=AsyncMock, return_value=meta):
            assert await store.get_metadata("somekey00000000000000000000000") == meta

    # ---- get_public_url --------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_public_url(self, store, mock_storage):
        meta = {"mime_type": "image/png", "size_bytes": 4, "ext": ".png"}
        with patch.object(store, "_download_meta", new_callable=AsyncMock, return_value=meta):
            url = await store.get_public_url("somekey00000000000000000000000", expiration=7200)
            assert url == "https://signed-url.example.com/blob"
            # Verify expiration was passed
            mock_storage.get_public_url.assert_called_once()
            _, kwargs = mock_storage.get_public_url.call_args
            assert kwargs.get("expiration") == 7200

    @pytest.mark.asyncio
    async def test_get_public_url_not_found(self, store, mock_storage):
        with patch.object(store, "_download_meta", new_callable=AsyncMock, return_value=None):
            with pytest.raises(KeyError, match="Media not found"):
                await store.get_public_url("nonexistent000000000000000000000")

    @pytest.mark.asyncio
    async def test_get_direct_url_uses_mime_without_meta_lookup(self, store, mock_storage):
        url = await store.get_direct_url(
            "abcdef0123456789abcdef0123456789",
            mime_type="image/png",
            expiration=123,
        )

        assert url == "https://signed-url.example.com/blob"
        mock_storage.get_public_url.assert_awaited_once_with(
            "test-media/ab/cd/abcdef0123456789abcdef0123456789.png",
            expiration=123,
        )

    # ---- to_media_ref ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_to_media_ref(self, store):
        ref = store.to_media_ref("abc123", "image/jpeg")
        assert ref.kind == "url"
        assert ref.url == "agentflow://media/abc123"
        assert ref.mime_type == "image/jpeg"

    # ---- cloud path layout -----------------------------------------------

    def test_cloud_path_layout(self, store):
        key = "abcdef0123456789abcdef0123456789"
        ext = ".png"
        path = store._cloud_path(key, ext)
        assert path == f"test-media/ab/cd/{key}.png"

    def test_meta_cloud_path_layout(self, store):
        key = "abcdef0123456789abcdef0123456789"
        path = store._meta_cloud_path(key)
        assert path == f"test-media/ab/cd/{key}.meta.json"


# ---------------------------------------------------------------------------
# BaseMediaStore interface compliance
# ---------------------------------------------------------------------------


class TestCloudMediaStoreInterface:
    """Verify CloudMediaStore implements all BaseMediaStore methods."""

    def test_is_subclass_of_base(self):
        from tenxgraph.storage.media.storage.base import BaseMediaStore

        assert issubclass(CloudMediaStore, BaseMediaStore)

    def test_has_required_methods(self):
        methods = ["store", "retrieve", "delete", "exists", "to_media_ref"]
        for m in methods:
            assert hasattr(CloudMediaStore, m), f"Missing method: {m}"
