"""G-123 atomic metadata updates and G-125 cloud owner index."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.storage.media.storage.cloud_store import CloudMediaStore
from tenxgraph.storage.media.storage.local_store import LocalFileMediaStore
from tenxgraph.storage.media.storage.memory_store import InMemoryMediaStore


def _appender(thread: str):
    def fn(meta):
        threads = meta.get("threads", [])
        return {"threads": [*threads, thread]}

    return fn


class _SlowMemory(InMemoryMediaStore):
    """Yields inside get_metadata so an unlocked read-modify-write would interleave."""

    async def get_metadata(self, storage_key):
        meta = await super().get_metadata(storage_key)
        await asyncio.sleep(0)  # the value read goes stale while other tasks run
        return meta


@pytest.mark.asyncio
async def test_memory_modify_metadata_loses_no_update():
    store = _SlowMemory()
    key = await store.store(b"x", "text/plain", {"owner_id": "u", "threads": []})
    await asyncio.gather(*(store.modify_metadata(key, _appender(f"t{i}")) for i in range(20)))
    assert sorted((await store.get_metadata(key))["threads"]) == sorted(f"t{i}" for i in range(20))


@pytest.mark.asyncio
async def test_base_default_read_then_update_loses_updates():
    """Documents why the stores override modify_metadata (the default is not atomic)."""
    from tenxgraph.storage.media.storage.base import BaseMediaStore

    store = _SlowMemory()
    key = await store.store(b"x", "text/plain", {"threads": []})
    await asyncio.gather(
        *(BaseMediaStore.modify_metadata(store, key, _appender(f"t{i}")) for i in range(5))
    )
    assert len((await store.get_metadata(key))["threads"]) < 5


@pytest.mark.asyncio
async def test_local_modify_metadata_loses_no_update(tmp_path):
    store = LocalFileMediaStore(str(tmp_path))
    key = await store.store(b"x", "text/plain", {"threads": []})
    await asyncio.gather(*(store.modify_metadata(key, _appender(f"t{i}")) for i in range(20)))
    assert len((await store.get_metadata(key))["threads"]) == 20


@pytest.mark.asyncio
async def test_local_modify_metadata_loses_no_update_across_instances(tmp_path):
    """Two stores on one folder stand in for two workers: only the file lock serializes them."""
    a = LocalFileMediaStore(str(tmp_path))
    b = LocalFileMediaStore(str(tmp_path))
    key = await a.store(b"x", "text/plain", {"threads": []})
    await asyncio.gather(
        *(
            s.modify_metadata(key, _appender(f"{n}{i}"))
            for i in range(15)
            for n, s in (("a", a), ("b", b))
        )
    )
    assert len((await a.get_metadata(key))["threads"]) == 30


@pytest.mark.asyncio
async def test_local_modify_metadata_missing_key_and_lock_file_not_swept(tmp_path):
    store = LocalFileMediaStore(str(tmp_path))
    assert await store.modify_metadata("f" * 32, _appender("t")) is False
    key = await store.store(b"x", "text/plain", {})
    await store.modify_metadata(key, _appender("t"))
    assert (tmp_path / ".meta.lock").exists()
    assert await store.sweep_orphans(min_age_seconds=0) == 0
    assert await store.exists(key)


class _FakeBucket:
    """A dict standing in for the bucket behind cloud-storage-manager."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.on_url = None  # hook run when a URL for a path is requested
        self.storage = MagicMock()
        self.storage.upload = AsyncMock(side_effect=self._upload)
        self.storage.delete = AsyncMock(side_effect=self._delete)
        self.storage.get_public_url = AsyncMock(side_effect=self._url)

    async def _upload(self, local, remote):
        self.objects[remote] = Path(local).read_bytes()
        return remote

    async def _delete(self, remote):
        self.objects.pop(remote, None)
        return True

    async def _url(self, path, expiration=3600):
        return f"https://bucket.example.com/{path}"

    async def download(self, url):
        path = url.removeprefix("https://bucket.example.com/")
        if self.on_url:
            self.on_url(path)
        if path not in self.objects:
            raise FileNotFoundError(path)
        return self.objects[path]


def _cloud(bucket: _FakeBucket, monkeypatch) -> CloudMediaStore:
    monkeypatch.setattr(CloudMediaStore, "_download_from_url", staticmethod(bucket.download))
    return CloudMediaStore(bucket.storage, prefix="p")


@pytest.mark.asyncio
async def test_cloud_modify_metadata_retries_when_sidecar_changed_meanwhile(monkeypatch):
    bucket = _FakeBucket()
    store = _cloud(bucket, monkeypatch)
    key = await store.store(b"x", "text/plain", {"threads": []})
    meta_path = f"p/{key[:2]}/{key[2:4]}/{key}.meta.json"
    fired = []

    def other_worker(path):
        # Another worker links a thread right after our first read of the sidecar
        if path == meta_path and not fired and len(reads) >= 1:
            fired.append(1)
            meta = json.loads(bucket.objects[meta_path])
            meta["threads"] = ["other"]
            bucket.objects[meta_path] = json.dumps(meta).encode()
        if path == meta_path:
            reads.append(1)

    reads: list[int] = []
    bucket.on_url = other_worker
    assert await store.modify_metadata(key, _appender("mine")) is True
    assert json.loads(bucket.objects[meta_path])["threads"] == ["other", "mine"]


@pytest.mark.asyncio
async def test_cloud_modify_metadata_gives_up_when_never_stable(monkeypatch):
    bucket = _FakeBucket()
    store = _cloud(bucket, monkeypatch)
    key = await store.store(b"x", "text/plain", {"threads": []})
    meta_path = f"p/{key[:2]}/{key[2:4]}/{key}.meta.json"
    counter = iter(range(1000))

    def churn(path):
        if path == meta_path:
            bucket.objects[meta_path] = json.dumps(
                {"mime_type": "text/plain", "ext": ".txt", "n": next(counter)}
            ).encode()

    bucket.on_url = churn
    with pytest.raises(RuntimeError):
        await store.modify_metadata(key, _appender("mine"))


@pytest.mark.asyncio
async def test_cloud_owner_index_tracks_store_and_delete(monkeypatch):
    bucket = _FakeBucket()
    store = _cloud(bucket, monkeypatch)
    a = await store.store(b"1", "text/plain", {"owner_id": "u1"})
    b = await store.store(b"2", "text/plain", {"owner_id": "u1"})
    c = await store.store(b"3", "text/plain", {"owner_id": "u2"})
    await store.store(b"4", "text/plain", {})  # no owner: not indexed
    assert sorted(await store.list_keys_for_owner("u1")) == sorted([a, b])
    assert await store.list_keys_for_owner("u2") == [c]
    assert await store.list_keys_for_owner("nobody") == []
    assert await store.delete(a) is True
    assert await store.list_keys_for_owner("u1") == [b]
    with pytest.raises(NotImplementedError):
        await store.list_keys()


@pytest.mark.asyncio
async def test_cloud_owner_index_not_wiped_by_a_read_error(monkeypatch):
    bucket = _FakeBucket()
    store = _cloud(bucket, monkeypatch)
    a = await store.store(b"1", "text/plain", {"owner_id": "u1"})
    index_path = store._owner_index_path("u1")

    def timeout(path):
        if path == index_path:
            raise TimeoutError("read timed out")

    bucket.on_url = timeout
    with pytest.raises(TimeoutError):
        await store.store(b"2", "text/plain", {"owner_id": "u1"})
    bucket.on_url = None
    assert await store.list_keys_for_owner("u1") == [a]
    # The failed store took its objects back out
    assert len([p for p in bucket.objects if ".meta.json" in p]) == 1


@pytest.mark.asyncio
async def test_cloud_concurrent_stores_all_land_in_the_index(monkeypatch):
    bucket = _FakeBucket()
    store = _cloud(bucket, monkeypatch)
    keys = await asyncio.gather(
        *(store.store(b"x", "text/plain", {"owner_id": "u"}) for _ in range(10))
    )
    assert sorted(await store.list_keys_for_owner("u")) == sorted(keys)


@pytest.mark.asyncio
async def test_default_list_keys_for_owner_filters_by_metadata():
    store = InMemoryMediaStore()
    a = await store.store(b"1", "text/plain", {"owner_id": "u1"})
    await store.store(b"2", "text/plain", {"owner_id": "u2"})
    assert await store.list_keys_for_owner("u1") == [a]


@pytest.mark.asyncio
async def test_cloud_direct_url_uses_stored_extension_not_caller_mime(monkeypatch):
    bucket = _FakeBucket()
    store = _cloud(bucket, monkeypatch)
    key = await store.store(b"x", "image/png", {})
    url = await store.get_direct_url(key, mime_type="application/pdf")
    assert url.endswith(f"{key}.png")


@pytest.mark.asyncio
async def test_cloud_download_clients_have_timeouts(monkeypatch):
    """G-59 leftover: neither download path may wait forever."""
    import httpx

    seen = {}

    class _Client:
        def __init__(self, **kw):
            seen["timeout"] = kw.get("timeout")
            raise RuntimeError("stop")

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    with pytest.raises(RuntimeError, match="stop"):
        await CloudMediaStore._download_from_url("https://x.example.com/a")
    assert isinstance(seen["timeout"], httpx.Timeout)
    assert seen["timeout"].connect == 10.0
    assert seen["timeout"].read == 60.0

    import urllib.request

    got = {}

    def fake_urlopen(url, timeout=None):
        got["timeout"] = timeout
        raise RuntimeError("stop")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="stop"):
        CloudMediaStore._download_from_url_blocking("https://x.example.com/a")
    assert got["timeout"] == 60.0
