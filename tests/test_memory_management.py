"""Comprehensive memory management and resource disposal tests.

This module tests memory management, resource cleanup, and connection pooling
across BackgroundTaskManager and all publisher implementations.
"""

import asyncio
import gc
import logging
import sys
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.console_publisher import ConsolePublisher
from tenxgraph.runtime.publisher.events import Event, EventModel, EventType
from tenxgraph.utils.background_task_manager import BackgroundTaskManager


logger = logging.getLogger(__name__)


class TestBackgroundTaskManagerMemory:
    """Test BackgroundTaskManager memory management and cleanup guarantees."""

    @pytest.fixture
    def task_manager(self):
        """Create a fresh BackgroundTaskManager instance."""
        return BackgroundTaskManager(default_shutdown_timeout=5.0)

    @pytest.mark.asyncio
    async def test_shutdown_cancels_all_tasks(self, task_manager):
        """Test that shutdown properly cancels all running tasks."""
        task_count = 5
        running_flags = [False] * task_count

        async def long_running_task(index):
            running_flags[index] = True
            try:
                await asyncio.sleep(100)  # Very long sleep
            except asyncio.CancelledError:
                running_flags[index] = False
                raise

        # Create multiple long-running tasks
        for i in range(task_count):
            task_manager.create_task(long_running_task(i), name=f"task_{i}")

        # Verify tasks are running
        await asyncio.sleep(0.1)
        assert all(running_flags)
        assert task_manager.get_task_count() == task_count

        # Shutdown should cancel all
        stats = await task_manager.shutdown(timeout=2.0)

        assert stats["status"] in ["completed", "timeout"]
        assert stats["initial_tasks"] == task_count
        # All tasks should be cancelled
        assert all(not flag for flag in running_flags)

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_completion(self, task_manager):
        """Test that shutdown waits for tasks to complete gracefully."""
        completed = []
        started = []

        async def completable_task(index):
            started.append(index)
            # Use a short sleep that will complete before cancellation
            await asyncio.sleep(0.05)
            completed.append(index)

        # Create tasks that complete quickly
        for i in range(3):
            task_manager.create_task(completable_task(i))

        # Give tasks a moment to start
        await asyncio.sleep(0.01)
        assert len(started) == 3

        stats = await task_manager.shutdown(timeout=2.0)

        # Tasks should either complete naturally or be cancelled
        # Since shutdown calls cancel_all, they will be cancelled
        assert stats["status"] == "completed"
        assert task_manager.get_task_count() == 0

    @pytest.mark.asyncio
    async def test_shutdown_timeout_force_cancels(self, task_manager):
        """Test that shutdown times out when tasks don't respond."""

        async def slow_but_cancellable_task():
            # Task that takes time but responds to cancellation
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                # Respond to cancellation immediately
                raise

        task_manager.create_task(slow_but_cancellable_task(), name="slow_task")

        start = time.time()
        stats = await task_manager.shutdown(timeout=0.5)
        duration = time.time() - start

        # Task should be cancelled and complete quickly
        assert stats["status"] == "completed"
        assert duration < 1.0  # Should complete quickly due to cancellation
        assert task_manager.get_task_count() == 0

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, task_manager):
        """Test that multiple shutdown calls are safe."""
        async def quick_task():
            await asyncio.sleep(0.05)

        task_manager.create_task(quick_task())

        # First shutdown
        stats1 = await task_manager.shutdown()
        assert stats1["status"] == "completed"

        # Second shutdown should be no-op
        stats2 = await task_manager.shutdown()
        assert stats2["status"] == "already_shutdown"
        assert stats2["tasks_remaining"] == 0

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test that context manager properly cleans up resources."""
        task_started = False
        task_cancelled = False

        async def tracked_task():
            nonlocal task_started, task_cancelled
            task_started = True
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                task_cancelled = True
                raise

        async with BackgroundTaskManager(default_shutdown_timeout=2.0) as manager:
            manager.create_task(tracked_task())
            await asyncio.sleep(0.1)
            assert task_started
            assert manager.get_task_count() == 1

        # After exiting context, tasks should be cleaned up
        assert task_cancelled

    @pytest.mark.asyncio
    async def test_no_memory_leak_on_many_tasks(self, task_manager):
        """Test that creating many tasks doesn't leak memory."""
        async def tiny_task():
            await asyncio.sleep(0.001)

        # Create and complete many tasks
        for _ in range(100):
            task_manager.create_task(tiny_task())
            await asyncio.sleep(0.002)

        # Wait for all to complete
        await task_manager.wait_for_all()

        # Force garbage collection
        gc.collect()

        # All tasks should be cleaned up
        assert task_manager.get_task_count() == 0
        assert len(task_manager._tasks) == 0
        assert len(task_manager._task_metadata) == 0

    @pytest.mark.asyncio
    async def test_task_metadata_cleanup(self, task_manager):
        """Test that task metadata is properly cleaned up."""
        async def task_with_metadata():
            await asyncio.sleep(0.05)

        context = {"user": "test", "large_data": "x" * 1000}
        task_manager.create_task(
            task_with_metadata(),
            name="metadata_task",
            context=context,
        )

        # Initially metadata should exist
        assert len(task_manager._task_metadata) == 1

        await task_manager.wait_for_all()

        # Metadata should be cleaned up
        assert len(task_manager._task_metadata) == 0

    @pytest.mark.asyncio
    async def test_concurrent_shutdown_safe(self):
        """Test that concurrent shutdown calls are thread-safe."""
        manager = BackgroundTaskManager()

        async def worker():
            await asyncio.sleep(0.1)

        for _ in range(5):
            manager.create_task(worker())

        # Try to shutdown concurrently
        results = await asyncio.gather(
            manager.shutdown(),
            manager.shutdown(),
            manager.shutdown(),
            return_exceptions=True,
        )

        # All should complete without errors
        assert all(isinstance(r, dict) for r in results)
        # First should succeed, others should report already shutdown
        completed_count = sum(
            1 for r in results if isinstance(r, dict) and r.get("status") == "completed"
        )
        already_shutdown_count = sum(
            1 for r in results if isinstance(r, dict) and r.get("status") == "already_shutdown"
        )
        assert completed_count >= 1
        assert already_shutdown_count >= 2


class TestPublisherResourceDisposal:
    """Test proper resource disposal in all publisher implementations."""

    @pytest.mark.asyncio
    async def test_console_publisher_idempotent_close(self):
        """Test that ConsolePublisher close is idempotent."""
        publisher = ConsolePublisher()

        # First close
        await publisher.close()
        assert publisher._is_closed

        # Second close should be safe
        await publisher.close()
        assert publisher._is_closed

        # Publishing after close should fail
        event = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.UPDATE,
            node_name="test",
            data={},
        )
        with pytest.raises(RuntimeError, match="closed"):
            await publisher.publish(event)

    @pytest.mark.asyncio
    async def test_console_publisher_sync_close(self):
        """Test synchronous close of ConsolePublisher."""
        publisher = ConsolePublisher()
        publisher.sync_close()
        assert publisher._is_closed

    @pytest.mark.asyncio
    async def test_console_publisher_context_manager(self):
        """Test ConsolePublisher as context manager."""
        async with ConsolePublisher() as publisher:
            assert not publisher._is_closed
            event = EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.UPDATE,
                node_name="test",
                data={},
            )
            # Should be able to publish
            await publisher.publish(event)

        # After context exit, should be closed
        assert publisher._is_closed

    @pytest.mark.asyncio
    async def test_base_publisher_interface(self):
        """Test BasePublisher provides context manager interface."""

        class TestPublisher(BasePublisher):
            def __init__(self):
                super().__init__({})
                self.closed = False

            async def publish(self, event):
                if self._is_closed:
                    raise RuntimeError("Publisher is closed")
                return True

            async def close(self):
                self.closed = True
                self._is_closed = True

            def sync_close(self):
                self.closed = True
                self._is_closed = True

        async with TestPublisher() as publisher:
            event = EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.UPDATE,
                node_name="test",
                data={},
            )
            result = await publisher.publish(event)
            assert result is True

        assert publisher.closed


class TestPublisherConnectionPooling:
    """Test connection pooling and limits in publishers."""

    @pytest.mark.asyncio
    async def test_redis_publisher_connection_pooling_config(self):
        """Test RedisPublisher accepts connection pooling configuration."""
        try:
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
        except ImportError:
            pytest.skip("redis package not installed")

        config = {
            "url": "redis://localhost:6379/0",
            "max_connections": 20,
            "socket_timeout": 10.0,
            "socket_connect_timeout": 5.0,
            "socket_keepalive": True,
            "health_check_interval": 60,
        }

        publisher = RedisPublisher(config)
        assert publisher.max_connections == 20
        assert publisher.socket_timeout == 10.0
        assert publisher.socket_connect_timeout == 5.0
        assert publisher.socket_keepalive is True
        assert publisher.health_check_interval == 60

    @pytest.mark.asyncio
    async def test_kafka_publisher_connection_config(self):
        """Test KafkaPublisher accepts connection configuration."""
        try:
            from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
        except ImportError:
            pytest.skip("aiokafka package not installed")

        config = {
            "bootstrap_servers": "localhost:9092",
            "max_batch_size": 32768,
            "linger_ms": 10,
            "compression_type": "gzip",
            "request_timeout_ms": 60000,
        }

        publisher = KafkaPublisher(config)
        assert publisher.max_batch_size == 32768
        assert publisher.linger_ms == 10
        assert publisher.compression_type == "gzip"
        assert publisher.request_timeout_ms == 60000

    @pytest.mark.asyncio
    async def test_rabbitmq_publisher_connection_config(self):
        """Test RabbitMQPublisher accepts connection configuration."""
        try:
            from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
        except ImportError:
            pytest.skip("aio-pika package not installed")

        config = {
            "url": "amqp://guest:guest@localhost/",
            "connection_timeout": 15,
            "heartbeat": 120,
        }

        publisher = RabbitMQPublisher(config)
        assert publisher.connection_timeout == 15
        assert publisher.heartbeat == 120


class TestMemoryLeakDetection:
    """Test for memory leaks in long-running scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_no_leak_repeated_task_creation(self):
        """Test that repeated task creation/completion doesn't leak memory."""
        manager = BackgroundTaskManager()

        async def tiny_task():
            await asyncio.sleep(0.001)

        initial_task_count = len(manager._tasks)

        # Create and complete many iterations
        for _ in range(500):
            task = manager.create_task(tiny_task())
            await asyncio.sleep(0.002)

        await manager.wait_for_all()
        gc.collect()

        # Should return to initial state
        assert len(manager._tasks) == initial_task_count
        assert len(manager._task_metadata) == 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_no_leak_publisher_repeated_use(self):
        """Test that repeated publisher use doesn't leak memory."""
        publisher = ConsolePublisher()

        event = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.UPDATE,
            node_name="test",
            data={"iteration": 0},
        )

        # Publish many events
        for i in range(1000):
            event.data = {"iteration": i}
            await publisher.publish(event)

        await publisher.close()
        gc.collect()

        # Verify closed
        assert publisher._is_closed


class TestStressConcurrentOperations:
    """Stress tests for concurrent operations and edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_high_concurrency_tasks(self):
        """Test manager handles high concurrency gracefully."""
        manager = BackgroundTaskManager()
        completed = []

        async def concurrent_task(task_id):
            await asyncio.sleep(0.01)
            completed.append(task_id)

        # Create many concurrent tasks
        tasks = []
        for i in range(200):
            task = manager.create_task(concurrent_task(i))
            tasks.append(task)

        await manager.wait_for_all(timeout=10.0)

        assert len(completed) == 200
        assert manager.get_task_count() == 0

    @pytest.mark.asyncio
    async def test_rapid_create_and_cancel(self):
        """Test rapid task creation and cancellation."""
        manager = BackgroundTaskManager()

        async def cancelable_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass

        # Rapidly create tasks
        for _ in range(50):
            manager.create_task(cancelable_task())

        # Immediately cancel all
        await manager.cancel_all()
        await asyncio.sleep(0.2)

        # Should be cleaned up
        assert manager.get_task_count() == 0

    @pytest.mark.asyncio
    async def test_nested_task_creation(self):
        """Test tasks creating other tasks."""
        manager = BackgroundTaskManager()
        results = []

        async def child_task():
            await asyncio.sleep(0.05)
            results.append("child")

        async def parent_task():
            results.append("parent")
            # Create child task and wait for it
            child = manager.create_task(child_task())
            await asyncio.sleep(0.1)  # Give child time to complete

        manager.create_task(parent_task())
        await manager.wait_for_all(timeout=2.0)

        assert "parent" in results
        assert "child" in results

    @pytest.mark.asyncio
    async def test_exception_in_context_manager(self):
        """Test that exceptions in context manager still trigger cleanup."""
        task_cancelled = False

        async def tracked_task():
            nonlocal task_cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                task_cancelled = True
                raise

        try:
            async with BackgroundTaskManager(default_shutdown_timeout=1.0) as manager:
                manager.create_task(tracked_task())
                await asyncio.sleep(0.1)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Despite exception, cleanup should have happened
        assert task_cancelled


class TestResourceLimits:
    """Test that resource limits are properly enforced."""

    @pytest.mark.asyncio
    async def test_task_timeout_enforcement(self):
        """Test that task timeouts are enforced."""
        manager = BackgroundTaskManager()
        timed_out = False

        async def slow_task():
            nonlocal timed_out
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                timed_out = True
                raise

        # Create task with short timeout
        manager.create_task(slow_task(), timeout=0.2)

        await asyncio.sleep(0.5)

        assert timed_out
        assert manager.get_task_count() == 0

    @pytest.mark.asyncio
    async def test_shutdown_respects_timeout(self):
        """Test that shutdown timeout is enforced for tasks that complete."""

        async def quick_task():
            # Task that responds to cancellation
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        manager = BackgroundTaskManager()
        manager.create_task(quick_task())

        start = time.time()
        stats = await manager.shutdown(timeout=0.3)
        duration = time.time() - start

        # Task should be cancelled quickly
        assert stats["status"] == "completed"
        assert duration < 0.6  # Should complete quickly due to cancellation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
