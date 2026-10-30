"""Cloud-backed media store — S3 and GCS via ``cloud-storage-manager``.

Uses the `cloud-storage-manager <https://github.com/10xHub/cloud-storage-manager>`_
package to provide unified S3 / GCS blob storage behind the standard
:class:`BaseMediaStore` interface.

Install the optional dependency::

    pip install 10xgraph[cloud-storage]

Then create a store::

    from cloud_storage_manager import (
        CloudStorageFactory,
        StorageProvider,
        StorageConfig,
        AwsConfig,
    )
    from tenxgraph.media.storage import CloudMediaStore

    config = StorageConfig(
        aws=AwsConfig(
            bucket_name="my-bucket",
            access_key_id="...",
            secret_access_key="...",
        )
    )
    storage = CloudStorageFactory.get_storage(StorageProvider.AWS, config)
    store = CloudMediaStore(storage)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .base import BaseMediaStore, MediaTooLargeError, check_size_limit, validate_storage_key


logger = logging.getLogger("tenxgraph.storage.media.cloud")

# Extension map for cases where mimetypes doesn't know the type
_FALLBACK_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "audio/wav": ".wav",
    "audio/mp3": ".mp3",
    "image/webp": ".webp",
}
_ALLOWED_DOWNLOAD_SCHEMES = {"http", "https"}
# Signed-URL downloads. This is a storage client, not an LLM client, so the 600 s LLM timeout
# does not fit: a connect that hangs fails in 10 s, and a transfer that stalls for 60 s with no
# bytes fails (httpx applies read/write per operation, so a slow large file is not cut off).
_DOWNLOAD_CONNECT_TIMEOUT = 10.0
_DOWNLOAD_IDLE_TIMEOUT = 60.0
# Read-compare-write attempts before a metadata or index update gives up
_UPDATE_ATTEMPTS = 5


def _mime_to_ext(mime_type: str) -> str:
    """Convert MIME type to file extension."""
    ext = mimetypes.guess_extension(mime_type)
    if ext:
        return ext
    return _FALLBACK_EXT.get(mime_type, ".bin")


DEFAULT_PREFIX = "10xgraph-media"
LEGACY_PREFIX = "agentflow-media"


class CloudMediaStore(BaseMediaStore):
    """S3 / GCS media store backed by ``cloud-storage-manager``.

    Uses :class:`~cloud_storage_manager.base_storage.BaseCloudStorage` for
    the actual cloud operations (upload, delete, get_public_url).

    Layout in the bucket::

        {prefix}/{key[:2]}/{key[2:4]}/{key}{ext}       ← binary blob
        {prefix}/{key[:2]}/{key[2:4]}/{key}.meta.json   ← metadata sidecar
        {prefix}/owners/{sha256(owner_id)}.index.json   ← keys one owner stored

    The sidecar JSON contains ``mime_type``, ``size_bytes``, and any extra
    metadata passed to :meth:`store`.

    A file stored with an ``owner_id`` in its metadata is also added to that owner's index,
    which :meth:`list_keys_for_owner` reads. The bucket cannot be listed through
    ``cloud-storage-manager``, so files stored before the index existed (and files without an
    owner) are not in it; :meth:`list_keys` is not supported.

    The library has no conditional write, so a sidecar or index update re-reads the object
    right before writing and retries when it changed. Another worker that writes in the
    few milliseconds between that re-read and the upload can still be overwritten.

    Args:
        storage: A ``BaseCloudStorage`` instance (from
            :func:`cloud_storage_manager.CloudStorageFactory.get_storage`).
        prefix: Top-level "directory" in the bucket.  Defaults to
            ``"10xgraph-media"``. When the default is used, objects
            stored under the legacy ``"agentflow-media"`` prefix are still found
            on read (new writes always go to the new prefix).
        max_size_bytes: Optional cap. ``store()`` refuses larger data and ``retrieve()`` stops
            a download that grows past it. ``None`` (the default) means no cap.
    """

    def __init__(
        self,
        storage: Any,
        prefix: str = DEFAULT_PREFIX,
        max_size_bytes: int | None = None,
    ) -> None:
        self._max_size_bytes = max_size_bytes
        self._storage = storage
        self._prefix = prefix
        # Only the default prefix falls back to the pre-rename location.
        self._legacy_prefix = LEGACY_PREFIX if prefix == DEFAULT_PREFIX else None
        # Serializes read-modify-write of sidecars within this process
        self._meta_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # BaseMediaStore interface
    # ------------------------------------------------------------------

    async def store(
        self,
        data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        check_size_limit(len(data), self._max_size_bytes)
        key = uuid4().hex
        ext = _mime_to_ext(mime_type)
        blob_path = self._cloud_path(key, ext)
        meta_path = self._meta_cloud_path(key)

        # Upload binary blob via temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            await self._storage.upload(tmp_path, blob_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Upload sidecar metadata
        meta = {
            "mime_type": mime_type,
            "size_bytes": len(data),
            "ext": ext,
            **(metadata or {}),
        }
        try:
            await self._upload_meta(meta, meta_path)
        except BaseException:
            # Without its sidecar the blob can never be found again (nothing lists it, so
            # nothing would ever delete it): take it back out before reporting the failure.
            try:
                await self._storage.delete(blob_path)
            except Exception:
                logger.warning("Orphaned blob %s after a failed metadata upload", blob_path)
            raise

        owner = meta.get("owner_id")
        if owner:
            try:
                await self._modify_owner_index(owner, lambda keys: [*keys, key])
            except BaseException:
                # Not in the index means not listed and not counted against the quota
                for path in (blob_path, meta_path):
                    try:
                        await self._storage.delete(path)
                    except Exception:
                        logger.warning("Orphaned object %s after a failed index update", path)
                raise

        logger.debug("Stored %d bytes as %s at %s", len(data), key, blob_path)
        return key

    async def retrieve(self, storage_key: str) -> tuple[bytes, str]:
        meta, prefix = await self._locate(storage_key)
        if meta is None:
            raise KeyError(f"Media not found: {storage_key}")

        mime_type = meta["mime_type"]
        ext = meta.get("ext", _mime_to_ext(mime_type))
        blob_path = self._cloud_path(storage_key, ext, prefix)

        url = await self._storage.get_public_url(blob_path, expiration=300)
        if self._max_size_bytes is None:
            data = await self._download_from_url(url)
        else:
            data = await self._download_from_url(url, max_bytes=self._max_size_bytes)

        return data, mime_type

    async def delete(self, storage_key: str) -> bool:
        meta, prefix = await self._locate(storage_key)
        if meta is None:
            return False

        mime_type = meta["mime_type"]
        ext = meta.get("ext", _mime_to_ext(mime_type))
        blob_path = self._cloud_path(storage_key, ext, prefix)
        meta_path = self._meta_cloud_path(storage_key, prefix)

        # Blob first. If it cannot be removed the sidecar stays, so the object is still
        # findable and the delete can be retried; removing the sidecar alone would leave the
        # blob behind with nothing pointing at it.
        try:
            blob_deleted = bool(await self._storage.delete(blob_path))
        except Exception:
            logger.warning("Failed to delete blob %s; keeping its metadata", blob_path)
            return False

        try:
            meta_deleted = bool(await self._storage.delete(meta_path))
        except Exception:
            logger.warning("Failed to delete meta %s", meta_path)
            meta_deleted = False

        owner = meta.get("owner_id")
        if owner and (blob_deleted or meta_deleted):
            try:
                await self._modify_owner_index(
                    owner, lambda keys: [k for k in keys if k != storage_key]
                )
            except Exception as exc:
                # A stale entry is skipped when listing (its sidecar is gone), so this is safe
                logger.warning("Could not drop %s from its owner index: %s", storage_key, exc)

        return blob_deleted or meta_deleted

    async def exists(self, storage_key: str) -> bool:
        meta, _ = await self._locate(storage_key)
        return meta is not None

    async def get_metadata(self, storage_key: str) -> dict[str, Any] | None:
        meta, _ = await self._locate(storage_key)
        return meta

    async def update_metadata(self, storage_key: str, updates: dict[str, Any]) -> bool:
        return await self.modify_metadata(storage_key, lambda _meta: updates)

    async def modify_metadata(
        self,
        storage_key: str,
        fn: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> bool:
        """Read-modify-write of the sidecar with a re-read check (no conditional write exists).

        Serialized in this process by a lock. The sidecar is read again just before the upload
        and the update is retried when it changed meanwhile (another worker), so only a write
        landing between that re-read and the upload can still be lost.

        Raises:
            RuntimeError: The sidecar kept changing for ``_UPDATE_ATTEMPTS`` attempts.
        """
        async with self._meta_lock:
            for _ in range(_UPDATE_ATTEMPTS):
                meta, prefix = await self._locate(storage_key)
                if meta is None:
                    return False
                updates = fn(dict(meta))
                if not updates:
                    return True
                merged = {k: v for k, v in {**meta, **updates}.items() if v is not None}
                if await self._download_meta(storage_key, prefix) != meta:
                    continue  # changed (or unreadable) since the first read: start over
                await self._upload_meta(merged, self._meta_cloud_path(storage_key, prefix))
                return True
        raise RuntimeError(f"Metadata of {storage_key} kept changing; update not applied")

    async def list_keys_for_owner(self, owner_id: str) -> list[str]:
        """Keys of the files stored with ``owner_id``, from the owner's index object.

        Raises:
            Exception: The index could not be read (an empty list would hide the files).
        """
        return await self._read_owner_index(self._owner_index_path(owner_id)) or []

    # ------------------------------------------------------------------
    # Bonus: direct URL access
    # ------------------------------------------------------------------

    async def get_public_url(
        self,
        storage_key: str,
        expiration: int = 3600,
    ) -> str:
        """Generate a signed URL for direct browser/client access.

        Args:
            storage_key: The key returned by :meth:`store`.
            expiration: Signed URL validity in seconds (default 1 hour).

        Returns:
            A pre-signed URL (S3) or signed URL (GCS).

        Raises:
            KeyError: If the storage key does not exist.
        """
        meta, prefix = await self._locate(storage_key)
        if meta is None:
            raise KeyError(f"Media not found: {storage_key}")

        ext = meta.get("ext", _mime_to_ext(meta["mime_type"]))
        blob_path = self._cloud_path(storage_key, ext, prefix)
        return await self._storage.get_public_url(blob_path, expiration=expiration)

    async def get_direct_url(
        self,
        storage_key: str,
        mime_type: str | None = None,
        expiration: int = 3600,
    ) -> str | None:
        """Return a signed blob URL without downloading the object bytes.

        The object path (its extension, and the prefix for legacy objects) comes from the
        stored metadata, one small sidecar read; ``mime_type`` is accepted for interface
        compatibility and ignored, so a wrong caller mime cannot sign a missing path.

        Raises:
            ValueError: ``storage_key`` is not a key made by :meth:`store`.
        """
        validate_storage_key(storage_key)
        # The blob's extension is in the stored metadata, not in the key and not (safely) in
        # the caller's mime, so the sidecar is read.
        return await self.get_public_url(storage_key, expiration=expiration)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _upload_meta(self, meta: dict[str, Any], meta_path: str) -> None:
        """Upload the sidecar JSON (replacing an existing one)."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as tmp_meta:
            json.dump(meta, tmp_meta)
            tmp_meta_path = tmp_meta.name

        try:
            await self._storage.upload(tmp_meta_path, meta_path)
        finally:
            Path(tmp_meta_path).unlink(missing_ok=True)

    def _owner_index_path(self, owner_id: str) -> str:
        digest = hashlib.sha256(str(owner_id).encode("utf-8")).hexdigest()
        return f"{self._prefix}/owners/{digest}.index.json"

    async def _read_owner_index(self, path: str) -> list[str] | None:
        """Keys in the index object, ``None`` when it does not exist.

        Unlike the sidecar read, an error other than "not found" is raised: treating a timeout
        as an empty index and writing it back would erase the owner's list.
        """
        try:
            url = await self._storage.get_public_url(path, expiration=60)
            raw = await self._download_from_url(url)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        keys = json.loads(raw).get("keys")
        return [k for k in keys if isinstance(k, str)] if isinstance(keys, list) else []

    async def _modify_owner_index(
        self, owner_id: str, mutate: Callable[[list[str]], list[str]]
    ) -> None:
        """Rewrite an owner's index with ``mutate(keys)``; same re-read check as the sidecar."""
        path = self._owner_index_path(owner_id)
        async with self._meta_lock:
            for _ in range(_UPDATE_ATTEMPTS):
                before = await self._read_owner_index(path)
                current = before or []
                after = list(dict.fromkeys(mutate(list(current))))
                if after == current and (before is not None or not after):
                    return
                if await self._read_owner_index(path) != before:
                    continue
                await self._upload_meta({"keys": after}, path)
                return
        raise RuntimeError("Owner index kept changing; update not applied")

    def _cloud_path(self, key: str, ext: str, prefix: str | None = None) -> str:
        validate_storage_key(key)
        return f"{prefix or self._prefix}/{key[:2]}/{key[2:4]}/{key}{ext}"

    def _meta_cloud_path(self, key: str, prefix: str | None = None) -> str:
        validate_storage_key(key)
        return f"{prefix or self._prefix}/{key[:2]}/{key[2:4]}/{key}.meta.json"

    async def _locate(self, storage_key: str) -> tuple[dict[str, Any] | None, str]:
        """Find the sidecar metadata and the prefix it lives under.

        Tries the configured prefix first, then the legacy prefix (default
        prefix only). Returns ``(None, configured_prefix)`` when not found.

        Raises:
            ValueError: ``storage_key`` is not 32 lowercase hex characters.
        """
        validate_storage_key(storage_key)
        meta = await self._download_meta(storage_key)
        if meta is not None or self._legacy_prefix is None:
            return meta, self._prefix
        legacy_meta = await self._download_meta(storage_key, self._legacy_prefix)
        if legacy_meta is not None:
            return legacy_meta, self._legacy_prefix
        return None, self._prefix

    async def _download_meta(
        self, storage_key: str, prefix: str | None = None
    ) -> dict[str, Any] | None:
        """Download and parse the sidecar metadata JSON."""
        meta_path = self._meta_cloud_path(storage_key, prefix)
        try:
            url = await self._storage.get_public_url(meta_path, expiration=60)
            raw = await self._download_from_url(url)
            return json.loads(raw)
        except Exception as exc:
            # A missing sidecar is routine (it is how the legacy prefix is probed). Anything
            # else, such as a timeout or an auth error, would otherwise read as "no such
            # file", so it is logged where an operator will see it.
            level = logging.DEBUG if _is_not_found(exc) else logging.WARNING
            logger.log(
                level,
                "Could not download/parse sidecar metadata at %s (%s: %s)",
                meta_path,
                type(exc).__name__,
                exc,
            )
            return None

    @staticmethod
    async def _download_from_url(url: str, max_bytes: int | None = None) -> bytes:
        """Download bytes from a signed URL.

        Prefers ``httpx`` (async). The stdlib fallback is synchronous, so it is
        run in a worker thread: calling ``urllib.request.urlopen`` directly inside
        an ``async def`` blocked the whole event loop for the duration of the
        download -- stalling every other in-flight run in the process, not just
        this one.
        """
        _validate_download_url(url)
        try:
            import httpx
        except ImportError:
            return await asyncio.to_thread(
                CloudMediaStore._download_from_url_blocking, url, max_bytes
            )

        async with (
            httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(_DOWNLOAD_IDLE_TIMEOUT, connect=_DOWNLOAD_CONNECT_TIMEOUT),
            ) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf.extend(chunk)
                if max_bytes is not None and len(buf) > max_bytes:
                    raise MediaTooLargeError(f"Download exceeds {max_bytes} bytes")
            return bytes(buf)

    @staticmethod
    def _download_from_url_blocking(url: str, max_bytes: int | None = None) -> bytes:
        """Synchronous stdlib download. Must only be called off the event loop."""
        import urllib.request

        _validate_download_url(url)
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_IDLE_TIMEOUT) as resp:  # noqa: S310  # nosec B310
            if max_bytes is None:
                return resp.read()
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise MediaTooLargeError(f"Download exceeds {max_bytes} bytes")
            return data


def _is_not_found(exc: Exception) -> bool:
    """Whether ``exc`` means the object is absent (HTTP 403/404/410, a missing file or key)."""
    if isinstance(exc, FileNotFoundError | KeyError | json.JSONDecodeError):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (403, 404, 410)


def _validate_download_url(url: str) -> None:
    """Allow downloads only from explicit HTTP(S) signed URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_DOWNLOAD_SCHEMES or not parsed.netloc:
        raise ValueError(f"Unsupported download URL scheme: {parsed.scheme or 'missing'}")
