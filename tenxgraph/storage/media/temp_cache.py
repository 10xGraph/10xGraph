"""Temporary external URL adaptation cache.

Stores media fetched from external URLs as temporary cached entries with
TTL-based cleanup. These entries are **not** durable user assets — they
exist only to adapt external URLs for providers that cannot consume them
directly.

Default retention: 24 hours (hardcoded for v1).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger("agentflow.media.temp_cache")

# Default TTL for temporary cached media (24 hours)
DEFAULT_TTL_SECONDS = 24 * 60 * 60

# Namespace prefix for temporary cache entries in the checkpointer
TEMP_CACHE_NAMESPACE = "media:temp"


@dataclass
class TempCacheEntry:
    """Metadata for a single temporary cache entry."""

    content_hash: str
    source_url: str
    mime_type: str
    storage_key: str
    created_at: float
    expires_at: float
    is_temporary: bool = True

    @property
    def is_expired(self) -> bool:
        """Return True if this entry has passed its expiry time."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "content_hash": self.content_hash,
            "source_url": self.source_url,
            "mime_type": self.mime_type,
            "storage_key": self.storage_key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_temporary": self.is_temporary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TempCacheEntry:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            content_hash=data["content_hash"],
            source_url=data["source_url"],
            mime_type=data["mime_type"],
            storage_key=data["storage_key"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            is_temporary=data.get("is_temporary", True),
        )


def _content_hash(data: bytes) -> str:
    """Compute a SHA-256 hex digest for content-addressed caching."""
    return hashlib.sha256(data).hexdigest()


def _storage_key(content_hash: str, mime_type: str) -> str:
    """Build a storage key for a temporary media entry."""
    ext = mime_type.split("/")[-1] if "/" in mime_type else "bin"
    return f"temp/{content_hash[:12]}.{ext}"


class TemporaryMediaCache:
    """Manages temporary media cache entries with TTL cleanup.

    This cache stores metadata about temporarily fetched external media.
    The actual media bytes live in the ``BaseMediaStore``; this class
    only tracks the index/metadata and coordinates cleanup.

    Args:
        ttl_seconds: Time-to-live for cached entries (default: 24 hours).
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        # In-memory index for fast lookups (populated from checkpointer on first use)
        self._index: dict[str, TempCacheEntry] = {}
        self._loaded = False

    async def _ensure_loaded(self, checkpointer: Any) -> None:
        """Lazy-load the index from the checkpointer."""
        if self._loaded:
            return
        self._loaded = True
        try:
            keys = await checkpointer.alist_cache_keys(TEMP_CACHE_NAMESPACE)
            for key in keys:
                data = await checkpointer.aget_cache_value(TEMP_CACHE_NAMESPACE, key)
                if data is not None:
                    entry = TempCacheEntry.from_dict(data)
                    if not entry.is_expired:
                        self._index[key] = entry
        except Exception:
            logger.debug("Failed to load temp cache index from checkpointer")

    async def store(
        self,
        checkpointer: Any,
        source_url: str,
        mime_type: str,
        storage_key: str,
        content_hash: str | None = None,
        cache_key: str | None = None,
    ) -> TempCacheEntry:
        """Store a new temporary cache entry.

        Args:
            checkpointer: The checkpointer backend for persistence.
            source_url: The original external URL.
            mime_type: MIME type of the media.
            storage_key: Key in the media store where bytes are stored.
            content_hash: Optional content hash (computed if not provided).

        Returns:
            The stored ``TempCacheEntry``.
        """
        now = time.time()
        entry = TempCacheEntry(
            content_hash=content_hash or "",
            source_url=source_url,
            mime_type=mime_type,
            storage_key=storage_key,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )

        # Use cache_key if provided (for content-hash dedup), otherwise storage_key
        index_key = cache_key or storage_key

        await checkpointer.aput_cache_value(
            TEMP_CACHE_NAMESPACE,
            index_key,
            entry.to_dict(),
            ttl_seconds=self.ttl_seconds,
        )
        self._index[index_key] = entry
        logger.debug(
            "Stored temp cache entry: url=%s, key=%s, ttl=%ds",
            source_url,
            index_key,
            self.ttl_seconds,
        )
        return entry

    async def get(
        self,
        checkpointer: Any,
        storage_key: str,
    ) -> TempCacheEntry | None:
        """Retrieve a cache entry by storage key.

        Returns None if the entry does not exist or has expired.
        """
        await self._ensure_loaded(checkpointer)

        # Check in-memory index first
        entry = self._index.get(storage_key)
        if entry is not None:
            if entry.is_expired:
                await self._remove(checkpointer, storage_key)
                return None
            return entry

        # Fallback to checkpointer
        data = await checkpointer.aget_cache_value(TEMP_CACHE_NAMESPACE, storage_key)
        if data is None:
            return None

        entry = TempCacheEntry.from_dict(data)
        if entry.is_expired:
            await self._remove(checkpointer, storage_key)
            return None

        self._index[storage_key] = entry
        return entry

    async def list_expired(self, checkpointer: Any) -> list[TempCacheEntry]:
        """Return all expired cache entries."""
        await self._ensure_loaded(checkpointer)
        return [e for e in self._index.values() if e.is_expired]

    async def cleanup(
        self,
        checkpointer: Any,
        media_store: Any,
    ) -> int:
        """Remove expired entries from the cache and delete stored media.

        Args:
            checkpointer: The checkpointer backend.
            media_store: The media store to delete bytes from.

        Returns:
            Number of entries cleaned up.
        """
        await self.list_expired(checkpointer)
        count = 0

        for key, entry in list(self._index.items()):
            if not entry.is_expired:
                continue

            # Best-effort: delete media from store
            try:
                if media_store is not None:
                    await media_store.delete(entry.storage_key)
            except Exception:
                logger.debug(
                    "Failed to delete temp media key=%s (may already be gone)",
                    entry.storage_key,
                )

            # Remove from index (keyed by cache_key, not storage_key)
            self._index.pop(key, None)

            # Remove from checkpointer
            try:
                await checkpointer.aclear_cache_value(TEMP_CACHE_NAMESPACE, key)
                count += 1
            except Exception:
                logger.debug(
                    "Failed to clear temp cache entry key=%s",
                    key,
                )

        if count:
            logger.info("Cleaned up %d expired temporary cache entries", count)
        return count

    async def _remove(self, checkpointer: Any, storage_key: str) -> None:
        """Remove a single entry from index and checkpointer."""
        self._index.pop(storage_key, None)
        try:
            await checkpointer.aclear_cache_value(TEMP_CACHE_NAMESPACE, storage_key)
        except Exception:
            logger.debug("Failed to remove temp cache entry key=%s", storage_key)


async def fetch_and_cache(
    url: str,
    media_store: Any,
    temp_cache: TemporaryMediaCache,
    checkpointer: Any,
) -> tuple[bytes, str, str]:
    """Fetch an external URL and store it as temporary cached media.

    Args:
        url: The external URL to fetch.
        media_store: The media store to save bytes to.
        temp_cache: The temporary cache manager.
        checkpointer: The checkpointer for cache metadata persistence.

    Returns:
        Tuple of (bytes, mime_type, storage_key).
    """
    import aiohttp

    async with aiohttp.ClientSession() as session, session.get(url) as resp:
        resp.raise_for_status()
        data = await resp.read()
        mime_type = resp.headers.get("Content-Type", "application/octet-stream")

    # Compute content hash
    content_hash = _content_hash(data)

    # Check if already cached (keyed by content hash)
    cache_key = f"hash:{content_hash}"
    existing = await temp_cache.get(checkpointer, cache_key)
    if existing is not None:
        # Re-fetch bytes from store
        try:
            stored_data, stored_mime = await media_store.retrieve(existing.storage_key)
            return stored_data, stored_mime, existing.storage_key
        except Exception:
            logger.debug("Failed to retrieve existing cached media, will re-store")

    # Store in media store
    storage_key = _storage_key(content_hash, mime_type)
    await media_store.store(data, mime_type, key=storage_key)

    # Register in temp cache (keyed by content hash for dedup)
    await temp_cache.store(
        checkpointer=checkpointer,
        source_url=url,
        mime_type=mime_type,
        storage_key=storage_key,
        content_hash=content_hash,
        cache_key=cache_key,
    )

    logger.debug("Fetched and cached external URL: %s -> %s", url, storage_key)
    return data, mime_type, storage_key
