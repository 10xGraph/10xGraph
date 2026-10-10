"""Tests for the temporary external URL adaptation cache."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.storage.media.temp_cache import (
    DEFAULT_TTL_SECONDS,
    TEMP_CACHE_NAMESPACE,
    TempCacheEntry,
    TemporaryMediaCache,
    _content_hash,
    _storage_key,
    fetch_and_cache,
)


class FakeMediaStore:
    """Minimal fake media store for testing."""

    def __init__(self):
        self._data: dict[str, tuple[bytes, str]] = {}

    async def store(self, data: bytes, mime_type: str, key: str) -> None:
        self._data[key] = (data, mime_type)

    async def retrieve(self, key: str) -> tuple[bytes, str]:
        if key not in self._data:
            raise KeyError(f"Key not found: {key}")
        return self._data[key]

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


class FakeCheckpointer:
    """Minimal fake checkpointer for testing."""

    def __init__(self):
        self._cache: dict[str, dict] = {}

    async def aput_cache_value(self, namespace: str, key: str, value: dict, ttl_seconds: int | None = None):
        self._cache[f"{namespace}:{key}"] = value

    async def aget_cache_value(self, namespace: str, key: str):
        return self._cache.get(f"{namespace}:{key}")

    async def aclear_cache_value(self, namespace: str, key: str):
        return self._cache.pop(f"{namespace}:{key}", None)

    async def alist_cache_keys(self, namespace: str, prefix: str | None = None):
        ns_prefix = f"{namespace}:"
        keys = []
        for full_key in self._cache:
            if full_key.startswith(ns_prefix):
                key_part = full_key[len(ns_prefix):]
                if prefix is None or key_part.startswith(prefix):
                    keys.append(key_part)
        return keys


class TestContentHash:
    """Test content hashing."""

    def test_same_data_same_hash(self):
        data = b"hello world"
        assert _content_hash(data) == _content_hash(data)  # noqa: S101

    def test_different_data_different_hash(self):
        assert _content_hash(b"hello") != _content_hash(b"world")  # noqa: S101


class TestStorageKey:
    """Test storage key generation."""

    def test_image_png_key(self):
        key = _storage_key("abc123def456", "image/png")
        assert key.startswith("temp/abc123def45")  # noqa: S101
        assert key.endswith(".png")  # noqa: S101

    def test_unknown_mime_key(self):
        key = _storage_key("abc123", "application/octet-stream")
        assert key.startswith("temp/abc123")  # noqa: S101
        assert key.endswith(".octet-stream")  # noqa: S101


class TestTempCacheEntry:
    """Test TempCacheEntry dataclass."""

    def test_creation(self):
        now = time.time()
        entry = TempCacheEntry(
            content_hash="abc123",
            source_url="https://example.com/img.png",
            mime_type="image/png",
            storage_key="temp/abc123.png",
            created_at=now,
            expires_at=now + 3600,
        )
        assert entry.content_hash == "abc123"  # noqa: S101
        assert entry.is_expired is False  # noqa: S101

    def test_expired_entry(self):
        now = time.time()
        entry = TempCacheEntry(
            content_hash="abc123",
            source_url="https://example.com/img.png",
            mime_type="image/png",
            storage_key="temp/abc123.png",
            created_at=now - 7200,
            expires_at=now - 3600,
        )
        assert entry.is_expired is True  # noqa: S101

    def test_to_dict_and_from_dict(self):
        now = time.time()
        entry = TempCacheEntry(
            content_hash="abc123",
            source_url="https://example.com/img.png",
            mime_type="image/png",
            storage_key="temp/abc123.png",
            created_at=now,
            expires_at=now + 3600,
        )
        data = entry.to_dict()
        restored = TempCacheEntry.from_dict(data)
        assert restored.content_hash == entry.content_hash  # noqa: S101
        assert restored.source_url == entry.source_url  # noqa: S101
        assert restored.mime_type == entry.mime_type  # noqa: S101
        assert restored.storage_key == entry.storage_key  # noqa: S101
        assert restored.is_temporary is True  # noqa: S101


class TestTemporaryMediaCache:
    """Test TemporaryMediaCache class."""

    @pytest.mark.asyncio
    async def test_store_and_get(self):
        checkpointer = FakeCheckpointer()
        cache = TemporaryMediaCache(ttl_seconds=3600)

        entry = await cache.store(
            checkpointer=checkpointer,
            source_url="https://example.com/img.png",
            mime_type="image/png",
            storage_key="temp/abc123.png",
            content_hash="abc123",
        )

        assert entry.storage_key == "temp/abc123.png"  # noqa: S101

        retrieved = await cache.get(checkpointer, "temp/abc123.png")
        assert retrieved is not None  # noqa: S101
        assert retrieved.content_hash == "abc123"  # noqa: S101

    @pytest.mark.asyncio
    async def test_get_expired_returns_none(self):
        checkpointer = FakeCheckpointer()
        cache = TemporaryMediaCache(ttl_seconds=1)

        await cache.store(
            checkpointer=checkpointer,
            source_url="https://example.com/img.png",
            mime_type="image/png",
            storage_key="temp/expired.png",
            content_hash="xyz",
        )
        cache._loaded = True

        # Force expiry by manipulating the index
        cache._index["temp/expired.png"].expires_at = time.time() - 100

        result = await cache.get(checkpointer, "temp/expired.png")
        assert result is None  # noqa: S101

    @pytest.mark.asyncio
    async def test_list_expired(self):
        checkpointer = FakeCheckpointer()
        cache = TemporaryMediaCache(ttl_seconds=3600)

        # Store a valid entry
        await cache.store(
            checkpointer=checkpointer,
            source_url="https://example.com/valid.png",
            mime_type="image/png",
            storage_key="temp/valid.png",
        )

        # Manually add an expired entry
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

        expired = await cache.list_expired(checkpointer)
        assert len(expired) == 1  # noqa: S101
        assert expired[0].storage_key == "temp/old.png"  # noqa: S101

    @pytest.mark.asyncio
    async def test_cleanup_deletes_expired(self):
        checkpointer = FakeCheckpointer()
        media_store = FakeMediaStore()
        cache = TemporaryMediaCache(ttl_seconds=3600)

        # Store media and cache entry
        await media_store.store(b"fake-image-data", "image/png", "temp/old.png")
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

        count = await cache.cleanup(checkpointer, media_store)
        assert count == 1  # noqa: S101
        assert "temp/old.png" not in cache._index  # noqa: S101

    @pytest.mark.asyncio
    async def test_cleanup_safe_when_storage_gone(self):
        checkpointer = FakeCheckpointer()
        media_store = FakeMediaStore()
        cache = TemporaryMediaCache(ttl_seconds=3600)

        # Add expired entry without corresponding media
        now = time.time()
        expired_entry = TempCacheEntry(
            content_hash="ghost",
            source_url="https://example.com/ghost.png",
            mime_type="image/png",
            storage_key="temp/ghost.png",
            created_at=now - 7200,
            expires_at=now - 3600,
        )
        cache._index["temp/ghost.png"] = expired_entry
        cache._loaded = True
        await checkpointer.aput_cache_value(
            TEMP_CACHE_NAMESPACE, "temp/ghost.png", expired_entry.to_dict()
        )

        count = await cache.cleanup(checkpointer, media_store)
        assert count == 1  # noqa: S101

    @pytest.mark.asyncio
    async def test_default_ttl_is_24_hours(self):
        cache = TemporaryMediaCache()
        assert cache.ttl_seconds == DEFAULT_TTL_SECONDS  # noqa: S101
        assert cache.ttl_seconds == 24 * 60 * 60  # noqa: S101


class TestFetchAndCache:
    """Test the fetch_and_cache function."""

    @pytest.mark.asyncio
    async def test_fetches_and_stores(self):
        media_store = FakeMediaStore()
        checkpointer = FakeCheckpointer()
        cache = TemporaryMediaCache()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.read = AsyncMock(return_value=b"fake-image-bytes")
        mock_response.headers = {"Content-Type": "image/png"}

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            data, mime, key = await fetch_and_cache(
                url="https://example.com/test.png",
                media_store=media_store,
                temp_cache=cache,
                checkpointer=checkpointer,
            )

        assert data == b"fake-image-bytes"  # noqa: S101
        assert mime == "image/png"  # noqa: S101
        assert key.startswith("temp/")  # noqa: S101
        assert key.endswith(".png")  # noqa: S101

        # Verify stored in media store
        stored_data, stored_mime = await media_store.retrieve(key)
        assert stored_data == b"fake-image-bytes"  # noqa: S101

        # Verify registered in temp cache (keyed by content hash, not storage_key)
        keys = await checkpointer.alist_cache_keys("media:temp")
        assert len(keys) == 1  # noqa: S101
        entry = await cache.get(checkpointer, keys[0])
        assert entry is not None  # noqa: S101
        assert entry.source_url == "https://example.com/test.png"  # noqa: S101


class TestInMemoryCheckpointerListCacheKeys:
    """Test InMemoryCheckpointer.alist_cache_keys."""

    @pytest.mark.asyncio
    async def test_lists_keys_for_namespace(self):
        from tenxgraph.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer

        cp = InMemoryCheckpointer()

        await cp.aput_cache_value("media:temp", "key1", {"data": "a"})
        await cp.aput_cache_value("media:temp", "key2", {"data": "b"})
        await cp.aput_cache_value("other:ns", "key3", {"data": "c"})

        keys = await cp.alist_cache_keys("media:temp")
        assert len(keys) == 2  # noqa: S101
        assert "key1" in keys  # noqa: S101
        assert "key2" in keys  # noqa: S101

    @pytest.mark.asyncio
    async def test_lists_keys_with_prefix(self):
        from tenxgraph.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer

        cp = InMemoryCheckpointer()

        await cp.aput_cache_value("media:temp", "abc123", {"data": "a"})
        await cp.aput_cache_value("media:temp", "def456", {"data": "b"})
        await cp.aput_cache_value("media:temp", "abc789", {"data": "c"})

        keys = await cp.alist_cache_keys("media:temp", prefix="abc")
        assert len(keys) == 2  # noqa: S101
        assert "abc123" in keys  # noqa: S101
        assert "abc789" in keys  # noqa: S101
        assert "def456" not in keys  # noqa: S101

    @pytest.mark.asyncio
    async def test_empty_namespace_returns_empty(self):
        from tenxgraph.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer

        cp = InMemoryCheckpointer()
        keys = await cp.alist_cache_keys("nonexistent")
        assert keys == []  # noqa: S101


class TestBaseCheckpointerListCacheKeysDefault:
    """Test that base checkpointer returns empty list by default."""

    @pytest.mark.asyncio
    async def test_base_returns_empty_list(self):
        from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer

        # Create a minimal concrete subclass
        class DummyCheckpointer(BaseCheckpointer):
            async def asetup(self):
                pass
            async def aput_state(self, config, state):
                return state
            async def aget_state(self, config):
                return None
            async def aclear_state(self, config):
                pass
            async def aput_state_cache(self, config, state):
                pass
            async def aget_state_cache(self, config):
                return None
            async def aput_messages(self, config, messages, metadata=None):
                pass
            async def aget_message(self, config, message_id):
                pass
            async def alist_messages(self, config, search=None, offset=None, limit=None):
                return []
            async def adelete_message(self, config, message_id):
                pass
            async def aput_thread(self, config, thread_info):
                pass
            async def aget_thread(self, config):
                return None
            async def alist_threads(self, config, search=None, offset=None, limit=None):
                return []
            async def aclean_thread(self, config):
                pass
            async def arelease(self):
                pass

        cp = DummyCheckpointer()
        keys = await cp.alist_cache_keys("media:temp")
        assert keys == []  # noqa: S101
