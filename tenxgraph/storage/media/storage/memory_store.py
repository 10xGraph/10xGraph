"""In-memory media store — for testing and ephemeral scripts."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from .base import BaseMediaStore


class InMemoryMediaStore(BaseMediaStore):
    """Dict-backed media store.  Data lives in process memory only."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, str, dict[str, Any]]] = {}

    async def store(
        self,
        data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        key = uuid4().hex
        self._data[key] = (data, mime_type, metadata or {})
        return key

    async def retrieve(self, storage_key: str) -> tuple[bytes, str]:
        try:
            data, mime, _ = self._data[storage_key]
        except KeyError:
            raise KeyError(f"Media not found: {storage_key}") from None
        return data, mime

    async def delete(self, storage_key: str) -> bool:
        return self._data.pop(storage_key, None) is not None

    async def exists(self, storage_key: str) -> bool:
        return storage_key in self._data

    async def get_metadata(self, storage_key: str) -> dict[str, Any] | None:
        try:
            data, mime_type, metadata = self._data[storage_key]
        except KeyError:
            return None

        return {
            "mime_type": mime_type,
            "size_bytes": len(data),
            **metadata,
        }

    # -- Helpers for testing --------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()
