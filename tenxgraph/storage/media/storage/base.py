"""Abstract base class for media storage backends."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from tenxgraph.core.state.message_block import MediaRef


_VALID_KEY_RE = re.compile(r"[a-f0-9]{32}")


def validate_storage_key(key: str) -> None:
    """Accept only keys made by ``store()`` (32 lowercase hex characters).

    Raises:
        ValueError: The key has any other shape, so it can never become part of a path.
    """
    if not isinstance(key, str) or not _VALID_KEY_RE.fullmatch(key):
        raise ValueError(f"Invalid storage key format: {key!r}")


class BaseMediaStore(ABC):
    """Abstract interface for storing binary media outside the message system.

    Implementations store actual bytes externally (filesystem, S3, PG bytea,
    etc.) and return opaque storage keys.  Messages only hold lightweight
    ``MediaRef`` references.
    """

    @abstractmethod
    async def store(
        self,
        data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store binary data and return an opaque storage key."""

    @abstractmethod
    async def retrieve(self, storage_key: str) -> tuple[bytes, str]:
        """Retrieve binary data and MIME type by storage key.

        Raises:
            KeyError: If the key does not exist.
        """

    @abstractmethod
    async def delete(self, storage_key: str) -> bool:
        """Delete stored media.  Returns ``True`` if deleted."""

    @abstractmethod
    async def exists(self, storage_key: str) -> bool:
        """Check whether a storage key exists."""

    async def get_metadata(self, storage_key: str) -> dict[str, Any] | None:
        """Return lightweight metadata without reading the full blob when possible.

        The default implementation falls back to ``retrieve()``, which is correct
        but not optimal for remote stores. Production backends should override
        this to avoid downloading object bytes for metadata-only access.
        """
        try:
            data, mime_type = await self.retrieve(storage_key)
        except KeyError:
            return None

        return {
            "mime_type": mime_type,
            "size_bytes": len(data),
        }

    async def update_metadata(self, storage_key: str, updates: dict[str, Any]) -> bool:
        """Shallow-merge ``updates`` into the stored metadata.

        A key whose value is ``None`` is removed. Returns ``False`` when the object does
        not exist. The blob itself is not touched.

        Raises:
            NotImplementedError: The backend does not support it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support update_metadata()")

    async def get_direct_url(
        self,
        storage_key: str,
        mime_type: str | None = None,
        expiration: int = 3600,
    ) -> str | None:
        """Return a provider-consumable URL when the store supports it.

        Stores backed by cloud object storage can override this to return
        a short-lived signed URL so downstream model providers can fetch the
        media directly instead of routing bytes back through the app server.
        """
        return None

    def to_media_ref(
        self,
        storage_key: str,
        mime_type: str,
        **kwargs: Any,
    ) -> MediaRef:
        """Convert a storage key into a ``MediaRef`` for embedding in messages.

        The URL uses the ``graph://media/{key}`` scheme so that resolvers
        can identify internally-stored media.
        """
        return MediaRef(
            kind="url",
            url=f"graph://media/{storage_key}",
            mime_type=mime_type,
            **kwargs,
        )
