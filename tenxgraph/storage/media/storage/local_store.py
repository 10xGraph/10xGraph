"""Filesystem-backed media store — for dev / single-server deployments."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import mimetypes
import os
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4


try:  # POSIX only; elsewhere the in-process lock is all there is
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from tenxgraph.utils.private_fs import (
    mkdir_private,
    write_private_atomic,
)

from .base import BaseMediaStore, check_size_limit, validate_storage_key


logger = logging.getLogger("tenxgraph.media.storage.local")

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

    def __init__(
        self, base_dir: str = "./agentflow_media", max_size_bytes: int | None = None
    ) -> None:
        self._max_size_bytes = max_size_bytes
        self._base = Path(base_dir).resolve()
        # The folder is created on first write (or first metadata lock), not here, so merely
        # building the store leaves nothing on disk
        # One lock for sidecar read-modify-writes in this process, so they do not each park a
        # worker thread on the file lock below
        self._meta_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    async def store(
        self,
        data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        check_size_limit(len(data), self._max_size_bytes)
        key = uuid4().hex
        file_path = self._key_to_path(key, mime_type)
        meta_path = self._meta_path(key)

        # Owner-only: directories 0o700, files 0o600 (the sidecar holds the owner id)
        mkdir_private(file_path.parent)
        meta = {
            "mime_type": mime_type,
            "size_bytes": len(data),
            **(metadata or {}),
        }
        # Both files are written through a temp file and renamed, so a crash never leaves a
        # half-written blob or sidecar. The sidecar goes last: it is what makes the object
        # visible, so a crash between the two leaves at most an unreferenced blob.
        write_private_atomic(file_path, data)
        try:
            write_private_atomic(meta_path, json.dumps(meta).encode("utf-8"))
        except BaseException:
            with contextlib.suppress(OSError):
                file_path.unlink()
            raise

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

    async def iter_bytes(
        self, storage_key: str, chunk_size: int = 1024 * 1024
    ) -> AsyncIterator[bytes]:
        self._validate_key(storage_key)
        meta_path = self._meta_path(storage_key)
        if not meta_path.exists():
            raise KeyError(f"Media not found: {storage_key}")
        mime = json.loads(meta_path.read_text(encoding="utf-8"))["mime_type"]
        file_path = self._key_to_path(storage_key, mime)
        if not file_path.exists():
            raise KeyError(f"Media file missing for key: {storage_key}")
        with file_path.open("rb") as fh:
            while chunk := fh.read(chunk_size):
                yield chunk

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

    async def list_keys(self) -> list[str]:
        keys = []
        for meta_path in self._base.glob("*/*/*.meta.json"):
            key = meta_path.name[: -len(".meta.json")]
            try:
                validate_storage_key(key)
            except ValueError:
                continue
            keys.append(key)
        return keys

    async def update_metadata(self, storage_key: str, updates: dict[str, Any]) -> bool:
        return await self.modify_metadata(storage_key, lambda _meta: updates)

    async def modify_metadata(
        self,
        storage_key: str,
        fn: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> bool:
        """Atomic read-modify-write of one sidecar.

        Serialized by a lock in this process and an exclusive ``flock`` on
        ``{base_dir}/.meta.lock`` across processes (several workers sharing the folder), and the
        sidecar is replaced with an atomic rename. The ``flock`` is POSIX only.
        """
        self._validate_key(storage_key)
        async with self._meta_lock:
            return await asyncio.to_thread(self._modify_sync, storage_key, fn)

    def _modify_sync(
        self, storage_key: str, fn: Callable[[dict[str, Any]], dict[str, Any] | None]
    ) -> bool:
        meta_path = self._meta_path(storage_key)
        if not self._base.is_dir():
            return False  # nothing was ever stored; do not create the folder just to say so
        with self._file_lock():
            if not meta_path.exists():
                return False
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            updates = fn(dict(meta))
            if updates:
                meta.update(updates)
                meta = {k: v for k, v in meta.items() if v is not None}
                write_private_atomic(meta_path, json.dumps(meta).encode("utf-8"))
        return True

    @contextlib.contextmanager
    def _file_lock(self):
        if fcntl is None:  # pragma: no cover
            yield
            return
        mkdir_private(self._base)
        lock_path = self._base / ".meta.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing the descriptor releases the lock

    async def sweep_orphans(self, min_age_seconds: float = 3600.0) -> int:
        """Delete files a crash left behind and return how many were removed.

        Removes a blob with no sidecar, a sidecar with no blob, and stray temp files, each only
        when it is older than ``min_age_seconds`` so an upload in progress is never touched.
        Complete objects are never removed. Runs on a worker thread.
        """
        return await asyncio.to_thread(self._sweep_orphans_sync, min_age_seconds)

    def _sweep_orphans_sync(self, min_age_seconds: float) -> int:
        cutoff = time.time() - min_age_seconds
        removed = 0
        for path in self._base.glob("*/*/*"):
            try:
                if not path.is_file() or path.stat().st_mtime > cutoff:
                    continue
                if not self._is_orphan(path):
                    continue
                path.unlink()
                removed += 1
            except OSError:
                continue
        if removed:
            logger.info("Swept %d orphan media files", removed)
        return removed

    def _is_orphan(self, path: Path) -> bool:
        name = path.name
        if name.startswith(".") and name.endswith(".tmp"):
            return True
        key = name[:32]
        try:
            validate_storage_key(key)
        except ValueError:
            return False
        meta_path = self._meta_path(key)
        if name.endswith(".meta.json"):
            return not any(p.exists() and p != meta_path for p in path.parent.glob(f"{key}.*"))
        return not meta_path.exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_key(self, key: str) -> None:
        """Prevent path traversal by validating key format."""
        validate_storage_key(key)

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
