"""Additional tests for InMemoryStore to cover uncovered lines."""

import pytest

from tenxgraph.qa.testing.in_memory_store import InMemoryStore
from tenxgraph.core.state import Message
from tenxgraph.storage.store.store_schema import MemorySearchResult, MemoryType


class TestInMemoryStoreAsetup:
    """Test InMemoryStore.asetup()."""

    @pytest.mark.asyncio
    async def test_asetup_returns_none(self):
        """Test that asetup returns None."""
        store = InMemoryStore()
        result = await store.asetup()
        assert result is None

    @pytest.mark.asyncio
    async def test_asetup_no_side_effects(self):
        """Test that asetup doesn't affect state."""
        store = InMemoryStore()
        await store.asetup()
        assert store.memories == {}
        assert store._search_results == []


class TestInMemoryStoreLongContent:
    """Test InMemoryStore with long content strings."""

    @pytest.mark.asyncio
    async def test_store_long_content(self):
        """Test storing content longer than 50 chars (hits truncation branch)."""
        store = InMemoryStore()
        config = {"user_id": "user1"}
        long_content = "A" * 100  # Longer than 50 char cut_ratio
        mem_id = await store.astore(config, long_content)
        assert mem_id is not None
        result = await store.aget(config, mem_id)
        assert result is not None
        assert result.content == long_content

    @pytest.mark.asyncio
    async def test_store_exact_50_chars(self):
        """Test storing content of exactly 50 chars."""
        store = InMemoryStore()
        exact_content = "X" * 50
        mem_id = await store.astore({}, exact_content)
        result = await store.aget({}, mem_id)
        assert result.content == exact_content

    @pytest.mark.asyncio
    async def test_store_with_metadata(self):
        """Test storing with metadata."""
        store = InMemoryStore()
        config = {"user_id": "user1", "thread_id": "thread1"}
        mem_id = await store.astore(config, "content", metadata={"key": "value"})
        result = await store.aget(config, mem_id)
        assert result.metadata.get("key") == "value"


class TestInMemoryStoreSearchFiltering:
    """Test InMemoryStore search with filtering."""

    @pytest.mark.asyncio
    async def test_search_with_memory_type_filter(self):
        """Test search with memory_type filter."""
        store = InMemoryStore()
        config = {"user_id": "user1"}

        await store.astore(config, "episodic memory", memory_type=MemoryType.EPISODIC)
        await store.astore(config, "semantic memory", memory_type=MemoryType.SEMANTIC)

        results = await store.asearch(config, "memory", memory_type=MemoryType.EPISODIC)
        assert len(results) == 1
        assert "episodic" in results[0].content

    @pytest.mark.asyncio
    async def test_search_preconfigured_with_limit(self):
        """Test pre-configured search with a limit."""
        store = InMemoryStore()
        config = {}
        search_results = [
            MemorySearchResult(id=f"{i}", content=f"Result {i}", score=0.9 - i * 0.1)
            for i in range(5)
        ]
        store.set_search_results(search_results)
        results = await store.asearch(config, "query", limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_no_matches(self):
        """Test search with no matches."""
        store = InMemoryStore()
        config = {}
        await store.astore(config, "Python programming")
        results = await store.asearch(config, "JavaScript")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_with_limit(self):
        """Test search with limit applied."""
        store = InMemoryStore()
        config = {}
        for i in range(10):
            await store.astore(config, f"Python code example {i}")
        results = await store.asearch(config, "Python", limit=3)
        assert len(results) == 3


class TestInMemoryStoreGetAll:
    """Test InMemoryStore.aget_all()."""

    @pytest.mark.asyncio
    async def test_aget_all_empty(self):
        """Test aget_all with no memories."""
        store = InMemoryStore()
        results = await store.aget_all({})
        assert results == []

    @pytest.mark.asyncio
    async def test_aget_all_with_limit(self):
        """Test aget_all with limit."""
        store = InMemoryStore()
        config = {}
        for i in range(5):
            await store.astore(config, f"Memory {i}")
        results = await store.aget_all(config, limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_aget_not_found(self):
        """Test aget for non-existent memory."""
        store = InMemoryStore()
        result = await store.aget({}, "nonexistent-id")
        assert result is None


class TestInMemoryStoreUpdate:
    """Test InMemoryStore.aupdate()."""

    @pytest.mark.asyncio
    async def test_aupdate_not_found(self):
        """Test aupdate returns False when memory not found."""
        store = InMemoryStore()
        result = await store.aupdate({}, "nonexistent", "new content")
        assert result is False

    @pytest.mark.asyncio
    async def test_aupdate_with_message_content(self):
        """Test aupdate with Message object as content."""
        store = InMemoryStore()
        config = {"user_id": "user1"}
        mem_id = await store.astore(config, "Original")
        new_message = Message.text_message("Updated via message")
        result = await store.aupdate(config, mem_id, new_message)
        assert result is True
        updated = await store.aget(config, mem_id)
        assert "Updated via message" in updated.content

    @pytest.mark.asyncio
    async def test_aupdate_with_metadata(self):
        """Test aupdate merges metadata."""
        store = InMemoryStore()
        config = {}
        mem_id = await store.astore(config, "Content", metadata={"orig": "value"})
        await store.aupdate(config, mem_id, "New content", metadata={"new_key": "new_val"})
        updated = await store.aget(config, mem_id)
        assert updated.metadata.get("orig") == "value"
        assert updated.metadata.get("new_key") == "new_val"


class TestInMemoryStoreDelete:
    """Test InMemoryStore.adelete()."""

    @pytest.mark.asyncio
    async def test_adelete_not_found(self):
        """Test adelete returns False when memory not found."""
        store = InMemoryStore()
        result = await store.adelete({}, "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_adelete_found(self):
        """Test adelete returns True and removes memory."""
        store = InMemoryStore()
        mem_id = await store.astore({}, "To delete")
        result = await store.adelete({}, mem_id)
        assert result is True
        assert await store.aget({}, mem_id) is None


class TestInMemoryStoreForget:
    """Test InMemoryStore.aforget_memory()."""

    @pytest.mark.asyncio
    async def test_aforget_by_user_id(self):
        """Test deleting memories for a specific user."""
        store = InMemoryStore()
        await store.astore({"user_id": "user1"}, "User 1 memory")
        await store.astore({"user_id": "user2"}, "User 2 memory")
        await store.astore({"user_id": "user1"}, "Another user 1 memory")

        count = await store.aforget_memory({"user_id": "user1"})
        assert count == 2
        assert len(store.memories) == 1

    @pytest.mark.asyncio
    async def test_aforget_by_thread_id(self):
        """Test deleting memories for a specific thread."""
        store = InMemoryStore()
        await store.astore({"thread_id": "thread1"}, "Thread 1 memory")
        await store.astore({"thread_id": "thread2"}, "Thread 2 memory")

        count = await store.aforget_memory({"thread_id": "thread1"})
        assert count == 1

    @pytest.mark.asyncio
    async def test_aforget_no_user_id(self):
        """Test aforget_memory with no user_id or thread_id deletes all."""
        store = InMemoryStore()
        await store.astore({"user_id": "user1"}, "Memory 1")
        await store.astore({"user_id": "user2"}, "Memory 2")

        # No filter = delete all
        count = await store.aforget_memory({})
        assert count == 2


class TestInMemoryStoreRelease:
    """Test InMemoryStore.arelease()."""

    @pytest.mark.asyncio
    async def test_arelease_clears_all(self):
        """Test arelease clears all memories."""
        store = InMemoryStore()
        await store.astore({}, "Memory 1")
        await store.astore({}, "Memory 2")
        store.set_search_results([MemorySearchResult(id="1", content="test", score=0.9)])

        await store.arelease()
        assert len(store.memories) == 0
        assert len(store._search_results) == 0

    @pytest.mark.asyncio
    async def test_arelease_on_empty_store(self):
        """Test arelease on empty store."""
        store = InMemoryStore()
        await store.arelease()  # Should not raise
        assert len(store.memories) == 0
