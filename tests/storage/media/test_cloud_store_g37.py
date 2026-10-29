"""Cloud media store fixes for 1.0 (G-37): no orphaned blob, errors not read as absence."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.storage.media.storage.cloud_store import CloudMediaStore


def _storage() -> MagicMock:
    storage = MagicMock()
    storage.upload = AsyncMock(return_value=None)
    storage.delete = AsyncMock(return_value=True)
    storage.get_public_url = AsyncMock(return_value="https://signed.example.com/x")
    return storage


@pytest.mark.asyncio
async def test_failed_sidecar_upload_removes_the_blob():
    storage = _storage()

    async def upload(local, remote):
        if remote.endswith(".meta.json"):
            raise RuntimeError("bucket write failed")

    storage.upload.side_effect = upload
    store = CloudMediaStore(storage, prefix="test-media")
    with pytest.raises(RuntimeError, match="bucket write failed"):
        await store.store(b"data", "image/png")
    deleted = [c.args[0] for c in storage.delete.call_args_list]
    assert len(deleted) == 1 and deleted[0].endswith(".png")


@pytest.mark.asyncio
async def test_sidecar_error_other_than_not_found_is_logged_as_warning(caplog):
    storage = _storage()
    store = CloudMediaStore(storage, prefix="test-media")
    store._download_from_url = AsyncMock(side_effect=TimeoutError("read timed out"))
    with caplog.at_level(logging.DEBUG, logger="tenxgraph.storage.media.cloud"):
        assert await store._download_meta("0" * 32) is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.asyncio
async def test_missing_sidecar_stays_quiet(caplog):
    storage = _storage()
    store = CloudMediaStore(storage, prefix="test-media")
    store._download_from_url = AsyncMock(side_effect=FileNotFoundError("nope"))
    with caplog.at_level(logging.DEBUG, logger="tenxgraph.storage.media.cloud"):
        assert await store._download_meta("0" * 32) is None
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
