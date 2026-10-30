"""Group 3 media regressions: G-34 delete by metadata, G-35 cloud delete, G-38 caps,
G-89 atomic local writes."""

import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.core.state.message_block import ImageBlock, MediaRef
from tenxgraph.storage.media import MediaProcessor, MultimodalConfig
from tenxgraph.storage.media.security import check_image_pixels
from tenxgraph.storage.media.storage import local_store as local_module
from tenxgraph.storage.media.storage.base import MediaTooLargeError
from tenxgraph.storage.media.storage.cloud_store import CloudMediaStore
from tenxgraph.storage.media.storage.local_store import LocalFileMediaStore
from tenxgraph.storage.media.storage.memory_store import InMemoryMediaStore


# ---- G-89: atomic local writes ------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_store_leaves_no_orphan_when_sidecar_write_fails(tmp_path, monkeypatch):
    store = LocalFileMediaStore(str(tmp_path))
    real = local_module.write_private_atomic
    calls = {"n": 0}

    def flaky(path, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(local_module, "write_private_atomic", flaky)
    with pytest.raises(OSError):
        await store.store(b"payload", "image/png", {"owner_id": "u"})
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


@pytest.mark.asyncio
async def test_local_store_data_write_is_atomic(tmp_path, monkeypatch):
    store = LocalFileMediaStore(str(tmp_path))

    def boom(path, data):  # simulates a crash while writing the blob
        raise OSError("crash")

    monkeypatch.setattr(local_module, "write_private_atomic", boom)
    with pytest.raises(OSError):
        await store.store(b"payload", "image/png")
    assert await store.list_keys() == []
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


# ---- G-38: size cap, streaming, pixels ----------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "local"])
async def test_store_size_cap(tmp_path, kind):
    store = (
        InMemoryMediaStore(max_size_bytes=10)
        if kind == "memory"
        else LocalFileMediaStore(str(tmp_path), max_size_bytes=10)
    )
    key = await store.store(b"x" * 10, "text/plain")
    assert await store.exists(key)
    with pytest.raises(MediaTooLargeError):
        await store.store(b"x" * 11, "text/plain")
    assert len(await store.list_keys()) == 1


@pytest.mark.asyncio
async def test_cloud_store_size_cap():
    storage = AsyncMock()
    store = CloudMediaStore(storage, prefix="p", max_size_bytes=4)
    with pytest.raises(MediaTooLargeError):
        await store.store(b"12345", "text/plain")
    storage.upload.assert_not_called()


@pytest.mark.asyncio
async def test_local_iter_bytes_streams_chunks(tmp_path):
    store = LocalFileMediaStore(str(tmp_path))
    key = await store.store(b"abcdefghij", "text/plain")
    chunks = [c async for c in store.iter_bytes(key, chunk_size=4)]
    assert chunks == [b"abcd", b"efgh", b"ij"]
    mem = InMemoryMediaStore()
    mkey = await mem.store(b"abcdefghij", "text/plain")
    assert b"".join([c async for c in mem.iter_bytes(mkey, chunk_size=4)]) == b"abcdefghij"
    with pytest.raises(KeyError):
        [c async for c in store.iter_bytes("0" * 32)]


def _png(w, h):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


def test_check_image_pixels():
    check_image_pixels(_png(10, 10), 100)
    with pytest.raises(ValueError, match="pixels"):
        check_image_pixels(_png(11, 10), 100)
    check_image_pixels(b"not an image", 100)  # undecodable data is left to other checks
    check_image_pixels(_png(500, 500), None)


def test_processor_enforces_pixel_cap_before_decoding():
    block = ImageBlock(
        media=MediaRef(
            kind="data",
            mime_type="image/png",
            data_base64=base64.b64encode(_png(200, 200)).decode(),
        )
    )
    proc = MediaProcessor(MultimodalConfig(max_image_pixels=1000))
    with pytest.raises(ValueError, match="pixels"):
        proc.validate_image(block)
    with pytest.raises(ValueError, match="pixels"):
        proc.resize_image(block)
    assert MediaProcessor(MultimodalConfig()).process(block) is not None


# ---- G-35: cloud delete -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloud_delete_keeps_sidecar_when_blob_delete_fails():
    storage = AsyncMock()
    storage.upload = AsyncMock(side_effect=lambda fp, cp: cp)
    storage.get_public_url = AsyncMock(return_value="https://example.com/x")
    store = CloudMediaStore(storage, prefix="p")
    store._locate = AsyncMock(return_value=({"mime_type": "text/plain", "ext": ".txt"}, "p"))

    deleted_paths = []

    async def delete(path):
        if not path.endswith(".meta.json"):
            raise RuntimeError("provider down")
        deleted_paths.append(path)
        return True

    storage.delete = delete
    assert await store.delete("a" * 32) is False
    assert deleted_paths == []  # sidecar kept so the delete can be retried


@pytest.mark.asyncio
async def test_cloud_delete_true_when_both_removed():
    storage = AsyncMock()
    storage.delete = AsyncMock(return_value=True)
    store = CloudMediaStore(storage, prefix="p")
    store._locate = AsyncMock(return_value=({"mime_type": "text/plain", "ext": ".txt"}, "p"))
    assert await store.delete("a" * 32) is True
    assert storage.delete.await_count == 2


# ---- G-34: delete by metadata ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_matching_removes_only_matching_owner_files():
    store = InMemoryMediaStore()
    a = await store.store(b"1", "text/plain", {"owner_id": "u1", "threads": ["t1"]})
    b = await store.store(b"2", "text/plain", {"owner_id": "u2", "threads": ["t1"]})
    c = await store.store(b"3", "text/plain", {"owner_id": "u1", "threads": ["t2"]})
    removed = await store.delete_matching({"owner_id": "u1"})
    assert removed == 2
    assert await store.list_keys() == [b]
    assert not await store.exists(a) and not await store.exists(c)


@pytest.mark.asyncio
async def test_delete_matching_requires_a_criterion():
    store = InMemoryMediaStore()
    await store.store(b"1", "text/plain")
    with pytest.raises(ValueError):
        await store.delete_matching({})
    assert len(await store.list_keys()) == 1


# ---- G-34: Google File API remote ids are kept and can be deleted -----------------------------


@pytest.mark.asyncio
async def test_google_upload_remembers_remote_name_and_can_delete_it():
    from tenxgraph.storage.media.provider_media import (
        ProviderMediaCache,
        delete_google_uploads,
        upload_to_google_file_api,
    )

    client = MagicMock()
    uploaded = MagicMock(uri="https://g/files/abc", mime_type="text/plain")
    uploaded.name = "files/abc"
    client.files.upload.return_value = uploaded
    cache = ProviderMediaCache()
    await upload_to_google_file_api(b"data", "text/plain", cache=cache, client=client)
    assert cache.remote_ids("google") == ["files/abc"]

    deleted = await delete_google_uploads(cache, client=client)
    assert deleted == 1
    client.files.delete.assert_called_once_with(name="files/abc")
    assert cache.remote_ids("google") == []
    assert cache.get("google", cache.content_key(b"data")) is None


# ---- G-130: orphan sweeper and temp-cache cleanup --------------------------------------------


@pytest.mark.asyncio
async def test_local_store_sweeps_orphan_files(tmp_path):
    import os
    import time

    store = LocalFileMediaStore(str(tmp_path))
    keep = await store.store(b"keep", "text/plain", {"owner_id": "u"})
    # a crash left a blob with no sidecar, a sidecar with no blob, and a temp file
    orphan_blob = tmp_path / "ab" / "cd" / ("ab" + "cd" + "0" * 28 + ".txt")
    orphan_blob.parent.mkdir(parents=True)
    orphan_blob.write_bytes(b"x")
    orphan_meta = tmp_path / "ab" / "cd" / ("ab" + "cd" + "1" * 28 + ".meta.json")
    orphan_meta.write_text('{"mime_type": "text/plain"}')
    tmp_file = tmp_path / "ab" / "cd" / ".something.tmp"
    tmp_file.write_bytes(b"x")
    fresh_blob = tmp_path / "ab" / "cd" / ("ab" + "cd" + "2" * 28 + ".txt")
    fresh_blob.write_bytes(b"x")  # in flight: younger than the cutoff
    old = time.time() - 7200
    for path in (orphan_blob, orphan_meta, tmp_file):
        os.utime(path, (old, old))

    assert await store.sweep_orphans(min_age_seconds=3600) == 3
    assert not orphan_blob.exists() and not orphan_meta.exists() and not tmp_file.exists()
    assert fresh_blob.exists()
    assert await store.exists(keep)
    assert (await store.retrieve(keep))[0] == b"keep"


@pytest.mark.asyncio
async def test_temp_cache_cleanup_removes_entries_expired_before_a_restart():
    import time

    from tenxgraph.storage.media.temp_cache import (
        TEMP_CACHE_NAMESPACE,
        TemporaryMediaCache,
        TempCacheEntry,
    )

    class Checkpointer:
        def __init__(self):
            self.data = {}
            self.ttls = {}

        async def aput_cache_value(self, ns, key, value, ttl_seconds=None):
            self.data[(ns, key)] = value
            self.ttls[(ns, key)] = ttl_seconds

        async def aget_cache_value(self, ns, key):
            return self.data.get((ns, key))

        async def aclear_cache_value(self, ns, key):
            self.data.pop((ns, key), None)

        async def alist_cache_keys(self, ns, prefix=None):
            return [k for (n, k) in self.data if n == ns]

    cp = Checkpointer()
    media = InMemoryMediaStore()
    key = await media.store(b"bytes", "image/png")

    first = TemporaryMediaCache(ttl_seconds=10)
    await first.store(cp, "https://x/y.png", "image/png", key)
    # the record outlives the entry's own expiry, so a later process can still clean the blob
    assert cp.ttls[(TEMP_CACHE_NAMESPACE, key)] > 10
    cp.data[(TEMP_CACHE_NAMESPACE, key)]["expires_at"] = time.time() - 5

    restarted = TemporaryMediaCache(ttl_seconds=10)
    assert await restarted.cleanup(cp, media) == 1
    assert not await media.exists(key)
    assert (TEMP_CACHE_NAMESPACE, key) not in cp.data
    assert TempCacheEntry  # imported for clarity of the data shape
