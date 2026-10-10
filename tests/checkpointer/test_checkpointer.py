"""Comprehensive tests for the checkpointer module."""

import pytest

from tenxgraph.storage.checkpointer import BaseCheckpointer, InMemoryCheckpointer
from tenxgraph.core.state import Message
from tenxgraph.utils.thread_info import ThreadInfo


class TestInMemoryCheckpointer:
    """Test the InMemoryCheckpointer class."""

    def test_in_memory_checkpointer_creation(self):
        """Test creating an InMemoryCheckpointer."""
        checkpointer = InMemoryCheckpointer()
        assert checkpointer is not None  # noqa: S101

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        checkpointer = InMemoryCheckpointer()

        state_data = {"messages": ["Hello"], "step": 1}
        checkpoint_config = {"thread_id": "test_checkpoint_1"}

        # Should not raise an exception
        checkpointer.put_state(checkpoint_config, state_data)

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        checkpointer = InMemoryCheckpointer()

        state_data = {"messages": ["Hello"], "step": 1}
        checkpoint_config = {"thread_id": "test_checkpoint_1"}

        # Save first
        checkpointer.put_state(checkpoint_config, state_data)

        # Then load
        loaded_data = checkpointer.get_state(checkpoint_config)
        assert loaded_data == state_data  # noqa: S101

    def test_load_nonexistent_checkpoint(self):
        """Test loading a checkpoint that doesn't exist."""
        checkpointer = InMemoryCheckpointer()

        # Should return None for non-existent checkpoint
        loaded_data = checkpointer.get_state({"thread_id": "non_existent"})
        assert loaded_data is None  # noqa: S101

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        checkpointer = InMemoryCheckpointer()

        # Save multiple checkpoints with thread info
        checkpoints = {
            "checkpoint_1": {"step": 1},
            "checkpoint_2": {"step": 2},
            "checkpoint_3": {"step": 3},
        }

        for checkpoint_id, data in checkpoints.items():
            config = {"thread_id": checkpoint_id}
            # Save state
            checkpointer.put_state(config, data)
            # Save thread info so list_threads can find it
            info = ThreadInfo(thread_id=checkpoint_id, thread_name=checkpoint_id)
            checkpointer.put_thread(config, info)

        # List checkpoints - this tests threads functionality
        config = {"thread_id": "test_thread"}
        thread_list = checkpointer.list_threads(config)

        # Should contain all saved checkpoints as thread info
        expected_count = 3
        assert len(thread_list) >= expected_count  # noqa: S101

    def test_put_get_messages(self):
        """Test putting and getting messages."""
        checkpointer = InMemoryCheckpointer()

        thread_config = {"thread_id": "test_thread"}
        message_id = "message_id"
        messages = [Message.text_message("Hello", "user", message_id=message_id)]

        # Put messages
        checkpointer.put_messages(thread_config, messages)

        # Get messages
        retrieved_messages = checkpointer.get_message(thread_config, message_id)
        assert retrieved_messages is not None  # noqa: S101

    def test_clear_state(self):
        """Test clearing state."""
        checkpointer = InMemoryCheckpointer()

        config = {"thread_id": "test_thread"}

        # Save some state
        checkpointer.put_state(config, {"data": "test"})

        # Clear state for this config
        checkpointer.clear_state(config)

        # Verify it's gone
        loaded_data = checkpointer.get_state(config)
        assert loaded_data is None  # noqa: S101

    @pytest.mark.asyncio
    async def test_aput_state(self):
        """Test async put state."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        state = {"data": "test"}
        result = await checkpointer.aput_state(config, state)
        assert result == state  # noqa: S101

    @pytest.mark.asyncio
    async def test_aget_state(self):
        """Test async get state."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        state = {"data": "test"}
        await checkpointer.aput_state(config, state)
        result = await checkpointer.aget_state(config)
        assert result == state  # noqa: S101

    @pytest.mark.asyncio
    async def test_aclear_state(self):
        """Test async clear state."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        state = {"data": "test"}
        await checkpointer.aput_state(config, state)
        result = await checkpointer.aclear_state(config)
        assert result is True  # noqa: S101
        loaded = await checkpointer.aget_state(config)
        assert loaded is None  # noqa: S101

    @pytest.mark.asyncio
    async def test_aput_messages(self):
        """Test async put messages."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        messages = [Message.text_message("Hello", "user")]
        result = await checkpointer.aput_messages(config, messages)
        assert result is True  # noqa: S101

    @pytest.mark.asyncio
    async def test_alist_messages(self):
        """Test async list messages."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        messages = [Message.text_message("Hello", "user")]
        await checkpointer.aput_messages(config, messages)
        result = await checkpointer.alist_messages(config)
        assert len(result) == 1  # noqa: S101

    @pytest.mark.asyncio
    async def test_aput_thread(self):
        """Test async put thread."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        thread_info = ThreadInfo(thread_id="test_thread", thread_name="test")
        result = await checkpointer.aput_thread(config, thread_info)
        assert result is True  # noqa: S101

    @pytest.mark.asyncio
    async def test_aget_thread(self):
        """Test async get thread."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        thread_info = ThreadInfo(thread_id="test_thread", thread_name="test")
        await checkpointer.aput_thread(config, thread_info)
        result = await checkpointer.aget_thread(config)
        assert result == thread_info  # noqa: S101

    @pytest.mark.asyncio
    async def test_arelease(self):
        """Test async release."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        state = {"data": "test"}
        await checkpointer.aput_state(config, state)
        result = await checkpointer.arelease()
        assert result is True  # noqa: S101
        loaded = await checkpointer.aget_state(config)
        assert loaded is None  # noqa: S101

    def test_put_state_cache(self):
        """Test put state cache."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        state = {"data": "test"}
        result = checkpointer.put_state_cache(config, state)
        assert result == state  # noqa: S101

    def test_get_state_cache(self):
        """Test get state cache."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        state = {"data": "test"}
        checkpointer.put_state_cache(config, state)
        result = checkpointer.get_state_cache(config)
        assert result == state  # noqa: S101

    @pytest.mark.asyncio
    async def test_aput_state_cache(self):
        """Test async put state cache."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        state = {"data": "test"}
        result = await checkpointer.aput_state_cache(config, state)
        assert result == state  # noqa: S101

    @pytest.mark.asyncio
    async def test_aget_state_cache(self):
        """Test async get state cache."""
        checkpointer = InMemoryCheckpointer()
        config = {"thread_id": "test_thread"}
        state = {"data": "test"}
        await checkpointer.aput_state_cache(config, state)
        result = await checkpointer.aget_state_cache(config)
        assert result == state  # noqa: S101


class TestBaseCheckpointer:
    """Test the BaseCheckpointer abstract class."""

    def test_base_checkpointer_is_abstract(self):
        """Test that BaseCheckpointer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCheckpointer()  # type: ignore # Should raise TypeError for abstract class


def test_checkpointer_module_imports():
    """Test that checkpointer module imports work correctly."""
    assert BaseCheckpointer is not None  # noqa: S101
    assert InMemoryCheckpointer is not None  # noqa: S101
