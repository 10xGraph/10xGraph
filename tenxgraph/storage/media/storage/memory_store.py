"""In-memory media store — for testing and ephemeral scripts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from .base import BaseMediaStore, check_size_limit


class InMemoryMediaStore(BaseMediaStore):
    """Dict-backed media store.  Data lives in process memory only."""

    def __init__(self, max_size_bytes: int | None = None) -> None:
        self._max_size_bytes = max_size_bytes
        self._data: dict[str, tuple[bytes, str, dict[str, Any]]] = {}
        # Makes a read-modify-write (``modify_metadata``) one step
        self._meta_lock = asyncio.Lock()

    async def store(
        self,
        data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        check_size_limit(len(data), self._max_size_bytes)
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

    async def list_keys(self) -> list[str]:
        return list(self._data)

    async def modify_metadata(
        self,
        storage_key: str,
        fn: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> bool:
        async with self._meta_lock:
            current = await self.get_metadata(storage_key)
            if current is None:
                return False
            updates = fn(current)
            if updates:
                await self.update_metadata(storage_key, updates)
            return True

    async def update_metadata(self, storage_key: str, updates: dict[str, Any]) -> bool:
        entry = self._data.get(storage_key)
        if entry is None:
            return False
        data, mime_type, metadata = entry
        merged = {**metadata, **updates}
        self._data[storage_key] = (
            data,
            mime_type,
            {k: v for k, v in merged.items() if v is not None},
        )
        return True

    # -- Helpers for testing --------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()
