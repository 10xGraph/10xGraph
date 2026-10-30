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
        self.search_calls = []
        self.get_all_calls = []

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

    async def search(self, query, *, top_k=20, filters=None, threshold=0.1, rerank=False, **kwargs):
        """Mock mem0ai 2.x AsyncMemory.search: entity ids only inside ``filters``."""
        if set(kwargs) & {"user_id", "agent_id", "run_id"}:
            raise ValueError("Top-level entity parameters are not supported in search()")
        self.search_calls.append({"filters": filters, "top_k": top_k, "threshold": threshold})
        user_id = (filters or {}).get("user_id")
        res = []
        for it in self.items:
            if it["user_id"] != user_id:
                continue
            if query.lower() in it["memory"].lower():
                if threshold is None or it["score"] >= threshold:
                    res.append(it)
        return {"results": res[:top_k]}

    async def get_all(self, *, filters=None, top_k=20, **kwargs):
        """Mock mem0ai 2.x AsyncMemory.get_all: entity ids only inside ``filters``."""
        if set(kwargs) & {"user_id", "agent_id", "run_id"}:
            raise ValueError("Top-level entity parameters are not supported in get_all()")
        self.get_all_calls.append({"filters": filters, "top_k": top_k})
        user_id = (filters or {}).get("user_id")
        results = [it for it in self.items if it["user_id"] == user_id]
        return {"results": results[:top_k]}

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


@pytest.mark.asyncio
async def test_forget_with_narrowing_raises_and_deletes_nothing(store):
    config = {"user_id": "u1"}
    await store.astore(config, "A", category="travel")

    for narrowing in (
        {"category": "travel"},
        {"memory_type": MemoryType.SEMANTIC},
        {"filters": {"source": "chat"}},
    ):
        with pytest.raises(ValueError, match="cannot narrow"):
            await store.aforget_memory(config, **narrowing)

    assert await store.asearch(config, "A")


@pytest.mark.asyncio
async def test_forget_with_none_narrowing_deletes_all_of_user(store):
    config = {"user_id": "u1"}
    await store.astore(config, "A")

    await store.aforget_memory(config, memory_type=None, category=None, filters=None)

    assert not await store.asearch(config, "A")


@pytest.mark.asyncio
async def test_search_and_get_all_pass_scope_inside_filters(store):
    """G-29: mem0ai 2.x rejects top-level user_id/agent_id on search and get_all."""
    config = {"user_id": "u1"}
    await store.astore(config, "Alice likes tea")
    await store.asearch(config, "tea", limit=5, score_threshold=0.2)
    call = store._client.search_calls[-1]
    assert call["filters"] == {"user_id": "u1", "agent_id": "test_app"}
    assert call["top_k"] == 5 and call["threshold"] == 0.2
    assert len(await store.aget_all(config, limit=7)) == 1
    call = store._client.get_all_calls[-1]
    assert call["filters"] == {"user_id": "u1", "agent_id": "test_app"} and call["top_k"] == 7


@pytest.mark.asyncio
async def test_client_filters_cannot_replace_owner_scope(store):
    """J-4: reserved keys in caller filters never override the owner scope."""
    await store.astore({"user_id": "victim"}, "secret plan")
    mine = {"user_id": "attacker"}
    hostile = {
        "user_id": "victim",
        "agent_id": "x",
        "app_id": "y",
        "run_id": "z",
        "AND": [{"user_id": "victim"}, {"source": "chat"}],
        "OR": [{"user_id": "victim"}],
        "source": "chat",
    }
    results = await store.asearch(mine, "secret", filters=hostile)
    assert results == []
    sent = store._client.search_calls[-1]["filters"]
    assert sent["user_id"] == "attacker" and sent["agent_id"] == "test_app"
    assert "app_id" not in sent and "run_id" not in sent
    assert sent["source"] == "chat"
    assert sent["AND"] == [{"source": "chat"}]
    assert "OR" not in sent


@pytest.mark.asyncio
async def test_search_honors_max_tokens(store):
    """G-79: max_tokens bounds the total size of returned memories."""
    config = {"user_id": "u1"}
    for i in range(5):
        await store.astore(config, f"fact {i} " + "word " * 200)
    out = await store.asearch(config, "fact", max_tokens=400)
    assert 1 <= len(out) < 5
    assert len(await store.asearch(config, "fact", max_tokens=100_000)) == 5
