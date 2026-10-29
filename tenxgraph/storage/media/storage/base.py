"""Abstract base class for media storage backends."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
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


class MediaTooLargeError(ValueError):
    """The object is larger than the store's ``max_size_bytes``."""


def check_size_limit(size: int, max_size_bytes: int | None) -> None:
    """Raise :class:`MediaTooLargeError` when ``size`` exceeds ``max_size_bytes`` (None: no cap)."""
    if max_size_bytes is not None and size > max_size_bytes:
        raise MediaTooLargeError(
            f"Media is {size} bytes; this store accepts at most {max_size_bytes} bytes"
        )


def _metadata_matches(metadata: dict[str, Any], match: dict[str, Any]) -> bool:
    """True when every ``match`` item equals the metadata value, or is in a list value."""
    for name, wanted in match.items():
        have = metadata.get(name)
        if isinstance(have, list):
            if wanted not in have:
                return False
        elif have != wanted:
            return False
    return True


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

    async def iter_bytes(
        self, storage_key: str, chunk_size: int = 1024 * 1024
    ) -> AsyncIterator[bytes]:
        """Yield the object in chunks of at most ``chunk_size`` bytes.

        The default reads the whole object with ``retrieve()`` and slices it. Backends that
        can read incrementally (the local file store) override this so a large object is not
        held in memory.

        Raises:
            KeyError: If the key does not exist.
        """
        data, _ = await self.retrieve(storage_key)
        for start in range(0, len(data), chunk_size):
            yield data[start : start + chunk_size]

    async def delete_matching(self, match: dict[str, Any]) -> int:
        """Delete every object whose metadata matches ``match`` and return how many went.

        Each ``match`` item must equal the metadata value, or be an element of a list value
        (for example ``{"owner_id": "u1", "threads": "t1"}``). Use it to erase one user's or
        one thread's files. Needs ``list_keys()``.

        Raises:
            ValueError: ``match`` is empty (that would delete everything).
            NotImplementedError: The backend cannot enumerate its keys.
        """
        if not match:
            raise ValueError("delete_matching needs at least one criterion")
        removed = 0
        for key in await self.list_keys():
            metadata = await self.get_metadata(key)
            if metadata and _metadata_matches(metadata, match) and await self.delete(key):
                removed += 1
        return removed

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

    async def list_keys(self) -> list[str]:
        """Return the storage keys of every stored object.

        Lets a caller enumerate files (for example to list one owner's uploads by reading
        each key's metadata). Backends that cannot enumerate keep this default.

        Raises:
            NotImplementedError: The backend does not support it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support list_keys()")

    async def update_metadata(self, storage_key: str, updates: dict[str, Any]) -> bool:
        """Shallow-merge ``updates`` into the stored metadata.

        A key whose value is ``None`` is removed. Returns ``False`` when the object does
        not exist. The blob itself is not touched.

        Raises:
            NotImplementedError: The backend does not support it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support update_metadata()")

    async def modify_metadata(
        self,
        storage_key: str,
        fn: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> bool:
        """Read the metadata, let ``fn`` decide the change, and write it as one step.

        ``fn`` receives a copy of the current metadata and returns the ``update_metadata``
        updates, or ``None`` / an empty dict for no change. It must be a plain function (no
        awaiting): a store may call it more than once. Returns ``False`` when the object does
        not exist. Use it instead of ``get_metadata`` plus ``update_metadata`` whenever the
        new value depends on the old one (a list append), so concurrent updates are not lost.

        The default is those two calls and is NOT atomic. The bundled stores override it.

        Raises:
            NotImplementedError: The backend does not support ``update_metadata``.
        """
        metadata = await self.get_metadata(storage_key)
        if metadata is None:
            return False
        updates = fn(dict(metadata))
        if updates:
            return await self.update_metadata(storage_key, updates)
        return True

    async def list_keys_for_owner(self, owner_id: str) -> list[str]:
        """Return the keys of the objects whose metadata records ``owner_id`` as owner.

        The default lists every key and reads each one's metadata. A backend that cannot
        enumerate all keys but keeps a per-owner index (the cloud store) overrides it.

        Raises:
            NotImplementedError: The backend cannot enumerate its keys.
        """
        owned = []
        for key in await self.list_keys():
            metadata = await self.get_metadata(key)
            if metadata and str(metadata.get("owner_id")) == str(owner_id):
                owned.append(key)
        return owned

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
