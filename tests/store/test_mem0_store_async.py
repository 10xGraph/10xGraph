import asyncio
from datetime import datetime
from uuid import uuid4
from unittest.mock import patch

import pytest

from tenxgraph.storage.store.mem0_store import Mem0Store
from tenxgraph.storage.store.store_schema import MemoryType
from tenxgraph.core.state import Message


class MockAsyncMem0:
    def __init__(self):
        self.items = []  # list[dict]
        self._id = 1
        self.add_calls = []

    async def add(self, messages, user_id, agent_id=None, metadata=None, **kwargs):
        """Mock AsyncMemory.add method"""
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "agent_id": agent_id,
                "metadata": metadata,
                "kwargs": kwargs,
            }
        )
        text = messages[0]["content"] if messages else ""
        metadata = metadata or {}
        mem0_id = f"m{self._id}"
        self._id += 1
        
        # Store with the format expected by Mem0Store
        rec = {
            "id": mem0_id, 
            "memory": text, 
            "metadata": {
                **metadata,
                "memory_id": mem0_id,  # Add memory_id to metadata for _create_result
            }, 
            "user_id": user_id, 
            "score": 0.9,
            "agent_id": agent_id
        }
        self.items.append(rec)
        return {"results": [{"id": mem0_id}]}

    async def search(self, query, user_id, agent_id=None, limit=10, threshold=None, **kwargs):
        """Mock AsyncMemory.search method"""
        res = []
        for it in self.items:
            if it["user_id"] != user_id:
                continue
            if query.lower() in it["memory"].lower():
                # Apply score threshold if provided
                if threshold is None or it["score"] >= threshold:
                    res.append(it)
        return {"results": res[:limit]}

    async def get_all(self, user_id, agent_id=None, limit=100, **kwargs):
        """Mock AsyncMemory.get_all method"""
        results = [it for it in self.items if it["user_id"] == user_id]
        return {"results": results[:limit]}

    async def get(self, memory_id, **kwargs):
        """Mock AsyncMemory.get method"""
        for it in self.items:
            if it["id"] == memory_id or it.get("metadata", {}).get("memory_id") == memory_id:
                return it
        return None

    async def update(self, memory_id, data=None, metadata=None, **kwargs):
        """Mock AsyncMemory.update method"""
        for it in self.items:
            if it["id"] == memory_id or it.get("metadata", {}).get("memory_id") == memory_id:
                if data:
                    it["memory"] = data
                if metadata:
                    it["metadata"].update(metadata)
                return {"updated": True}
        return {"updated": False}

    async def delete(self, memory_id, **kwargs):
        """Mock AsyncMemory.delete method"""
        original_length = len(self.items)
        self.items = [i for i in self.items if i["id"] != memory_id and i.get("metadata", {}).get("memory_id") != memory_id]
        return {"deleted": len(self.items) < original_length}

    async def delete_all(self, user_id, **kwargs):
        """Mock AsyncMemory.delete_all method"""
        original_length = len(self.items)
        self.items = [i for i in self.items if i["user_id"] != user_id]
        deleted_count = original_length - len(self.items)
        return {"deleted_count": deleted_count}



@pytest.fixture()
def store():
    mock_instance = MockAsyncMem0()
    with patch("tenxgraph.storage.store.mem0_store.AsyncMemory") as mock_class:
        # Mock the from_config class method to return our mock instance
        async def mock_from_config(config):
            return mock_instance
        mock_class.from_config = mock_from_config
        mock_class.return_value = mock_instance
        yield Mem0Store(config={}, app_id="test_app")


@pytest.mark.asyncio
async def test_store_and_search(store):
    config = {"user_id": "u1"}
    mem_id = await store.astore(config, "Alice likes tea", memory_type=MemoryType.SEMANTIC)
    assert mem_id
    results = await store.asearch(config, "likes")
    assert results and results[0].content.startswith("Alice")
    assert "run_id" not in store._client.add_calls[-1]["kwargs"]


@pytest.mark.asyncio
async def test_store_ignores_thread_id_in_config(store):
    config = {"user_id": "u1", "thread_id": "t1"}
    mem_id = await store.astore(config, "Alice likes coffee", memory_type=MemoryType.SEMANTIC)
    assert mem_id
    results = await store.asearch(config, "coffee")
    assert results and results[0].content.startswith("Alice")
    assert "run_id" not in store._client.add_calls[-1]["kwargs"]


@pytest.mark.asyncio
async def test_get_update_delete(store):
    config = {"user_id": "u1"}
    mem_id = await store.astore(config, "Berlin is in Germany")
    got = await store.aget(config, mem_id["results"][0]["id"])  # Use actual returned ID
    assert got and got.content.startswith("Berlin")

    await store.aupdate(config, mem_id["results"][0]["id"], "Berlin is the capital of Germany")
    updated = await store.aget(config, mem_id["results"][0]["id"])
    assert updated and "capital" in updated.content

    await store.adelete(config, mem_id["results"][0]["id"])
    deleted = await store.aget(config, mem_id["results"][0]["id"])
    assert deleted is None


@pytest.mark.asyncio
async def test_batch_and_forget(store):
    config = {"user_id": "u1"}
    # Note: batch_store is not implemented in the current Mem0Store, so let's test individual stores
    await store.astore(config, "A")
    await store.astore(config, "B")
    await store.astore(config, "C")
    
    res = await store.asearch(config, "A")
    assert res
    await store.aforget_memory(config)
    res2 = await store.asearch(config, "A")
    assert not res2
