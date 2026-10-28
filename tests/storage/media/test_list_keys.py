"""BaseMediaStore.list_keys: memory and local stores enumerate; the base class does not."""

import pytest

from tenxgraph.storage.media.storage.base import BaseMediaStore
from tenxgraph.storage.media.storage.local_store import LocalFileMediaStore
from tenxgraph.storage.media.storage.memory_store import InMemoryMediaStore


@pytest.mark.asyncio
async def test_memory_store_lists_keys():
    store = InMemoryMediaStore()
    a = await store.store(b"a", "text/plain")
    b = await store.store(b"b", "text/plain")
    assert sorted(await store.list_keys()) == sorted([a, b])
    await store.delete(a)
    assert await store.list_keys() == [b]


@pytest.mark.asyncio
async def test_local_store_lists_keys(tmp_path):
    store = LocalFileMediaStore(str(tmp_path))
    assert await store.list_keys() == []
    keys = {await store.store(b"x", "text/plain"), await store.store(b"y", "image/png")}
    assert set(await store.list_keys()) == keys


@pytest.mark.asyncio
async def test_base_default_raises():
    class Minimal(InMemoryMediaStore):
        list_keys = BaseMediaStore.list_keys

    with pytest.raises(NotImplementedError):
        await Minimal().list_keys()
