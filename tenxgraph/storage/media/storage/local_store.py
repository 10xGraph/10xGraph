"""Filesystem-backed media store — for dev / single-server deployments."""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from .base import BaseMediaStore


logger = logging.getLogger("agentflow.media.storage.local")

_VALID_KEY_RE = re.compile(r"^[a-f0-9]{32}$")

# Extension map for cases where mimetypes doesn't know the type
_FALLBACK_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "audio/wav": ".wav",
    "audio/mp3": ".mp3",
    "image/webp": ".webp",
}


class LocalFileMediaStore(BaseMediaStore):
    """Store binary media on the local filesystem.

    Layout::

        {base_dir}/{key[:2]}/{key[2:4]}/{key}.{ext}
        {base_dir}/{key[:2]}/{key[2:4]}/{key}.meta.json

    The two-level sharding prevents too many files in a single directory.
    """

    def __init__(self, base_dir: str = "./agentflow_media") -> None:
        self._base = Path(base_dir).resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    async def store(
        self,
        data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        key = uuid4().hex
        file_path = self._key_to_path(key, mime_type)
        meta_path = self._meta_path(key)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)

        meta = {
            "mime_type": mime_type,
            "size_bytes": len(data),
            **(metadata or {}),
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        logger.debug("Stored %d bytes as %s at %s", len(data), key, file_path)
        return key

    async def retrieve(self, storage_key: str) -> tuple[bytes, str]:
        self._validate_key(storage_key)
        meta_path = self._meta_path(storage_key)
        if not meta_path.exists():
            raise KeyError(f"Media not found: {storage_key}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mime = meta["mime_type"]
        file_path = self._key_to_path(storage_key, mime)

        if not file_path.exists():
            raise KeyError(f"Media file missing for key: {storage_key}")

        return file_path.read_bytes(), mime

    async def delete(self, storage_key: str) -> bool:
        self._validate_key(storage_key)
        meta_path = self._meta_path(storage_key)
        if not meta_path.exists():
            return False

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mime = meta["mime_type"]
        file_path = self._key_to_path(storage_key, mime)

        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        # Clean up empty shard directories
        for parent in [file_path.parent, file_path.parent.parent]:
            try:
                parent.rmdir()  # only succeeds if empty
            except OSError:
                break

        return deleted

    async def exists(self, storage_key: str) -> bool:
        self._validate_key(storage_key)
        return self._meta_path(storage_key).exists()

    async def get_metadata(self, storage_key: str) -> dict[str, Any] | None:
        self._validate_key(storage_key)
        meta_path = self._meta_path(storage_key)
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_key(self, key: str) -> None:
        """Prevent path traversal by validating key format."""
        if not _VALID_KEY_RE.match(key):
            raise ValueError(f"Invalid storage key format: {key!r}")

    def _key_to_path(self, key: str, mime_type: str) -> Path:
        ext = _mime_to_ext(mime_type)
        return self._base / key[:2] / key[2:4] / f"{key}{ext}"

    def _meta_path(self, key: str) -> Path:
        return self._base / key[:2] / key[2:4] / f"{key}.meta.json"


def _mime_to_ext(mime_type: str) -> str:
    """Convert MIME type to file extension."""
    ext = mimetypes.guess_extension(mime_type)
    if ext:
        return ext
    return _FALLBACK_EXT.get(mime_type, ".bin")
