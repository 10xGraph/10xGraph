"""Tests for BackgroundTaskManager with performance and load testing."""

import asyncio
import pytest
import time
from unittest.mock import patch

from tenxgraph.utils.background_task_manager import BackgroundTaskManager


class TestBackgroundTaskManager:
    """Test suite for BackgroundTaskManager including performance and load tests."""

    @pytest.fixture
    def task_manager(self):
        """Create a fresh BackgroundTaskManager instance for each test."""
        return BackgroundTaskManager()

    @pytest.mark.asyncio
    async def test_create_task_basic(self, task_manager):
        """Test basic task creation and completion."""
        result = []
        
        async def simple_task():
            await asyncio.sleep(0.1)
            result.append("completed")
        
        task_manager.create_task(simple_task())
        
        # Wait for task to complete
        await task_manager.wait_for_all()
        
        assert result == ["completed"]
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_create_multiple_tasks(self, task_manager):
        """Test creating multiple tasks concurrently."""
        results = []
        
        async def numbered_task(number):
            await asyncio.sleep(0.1)
            results.append(number)
        
        # Create 5 concurrent tasks
        for i in range(5):
            task_manager.create_task(numbered_task(i))
        
        await task_manager.wait_for_all()
        
        assert len(results) == 5
        assert sorted(results) == list(range(5))
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_task_error_handling(self, task_manager):
        """Test that task errors are logged but don't crash the manager."""
        with patch('tenxgraph.utils.background_task_manager.logger') as mock_logger:

            async def failing_task():
                await asyncio.sleep(0.05)
                raise ValueError("Test error")

            task_manager.create_task(failing_task())
            
            # wait_for_all() should raise the exception
            with pytest.raises(ValueError, match="Test error"):
                await task_manager.wait_for_all()

            # Verify error was logged during task completion
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Background task raised an exception" in error_call
            assert "Test error" in str(error_call)

    @pytest.mark.asyncio
    async def test_task_cleanup_on_completion(self, task_manager):
        """Test that completed tasks are properly removed from tracking."""
        async def quick_task():
            await asyncio.sleep(0.05)
        
        # Start task
        task_manager.create_task(quick_task())
        
        # Initially should have one task
        assert len(task_manager._tasks) == 1
        
        # Wait for completion
        await asyncio.sleep(0.1)
        
        # Task should be cleaned up
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_task_cleanup_on_error(self, task_manager):
        """Test that failed tasks are properly removed from tracking."""
        with patch('tenxgraph.utils.background_task_manager.logger'):
            async def failing_task():
                await asyncio.sleep(0.05)
                raise RuntimeError("Test failure")
            
            task_manager.create_task(failing_task())
            
            # Initially should have one task
            assert len(task_manager._tasks) == 1
            
            # Wait for completion/failure
            await asyncio.sleep(0.1)
            
            # Task should be cleaned up even after failure
            assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_wait_for_all_empty(self, task_manager):
        """Test wait_for_all when no tasks are running."""
        # Should complete immediately without error
        await task_manager.wait_for_all()
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_wait_for_all_with_mixed_success_failure(self, task_manager):
        """Test wait_for_all with both successful and failing tasks."""
        results = []

        async def success_task(value):
            await asyncio.sleep(0.1)
            results.append(value)

        async def fail_task():
            await asyncio.sleep(0.05)
            raise RuntimeError("Intentional failure")

        with patch('tenxgraph.utils.background_task_manager.logger'):
            # Mix of success and failure tasks
            task_manager.create_task(success_task("success1"))
            task_manager.create_task(fail_task())
            task_manager.create_task(success_task("success2"))

            # wait_for_all should raise the first exception encountered
            with pytest.raises(RuntimeError, match="Intentional failure"):
                await task_manager.wait_for_all()

            # The successful tasks may not complete due to cancellation
            # This is expected behavior with asyncio.gather when one task fails

    @pytest.mark.asyncio
    async def test_wait_for_all_leaves_no_residue(self, task_manager):
        """wait_for_all() must return with the bookkeeping already cleaned up.

        ``asyncio.gather`` resolves the moment the last task finishes, but each task's
        ``add_done_callback`` cleanup is queued with ``call_soon`` and has not run yet.
        Returning at that point leaves every finished task in ``_tasks`` and
        ``_task_metadata``, so a caller reading ``pending_count`` immediately after
        sees a stale non-zero value. The burst-then-pause shape is what exposes it:
        earlier bursts get their callbacks drained by the pause, the final burst does
        not, so exactly one burst's worth is left behind.
        """
        for _ in range(5):
            for _ in range(20):

                async def job():
                    await asyncio.sleep(0.001)

                task_manager.create_task(job())
            await asyncio.sleep(0.01)

        await task_manager.wait_for_all()

        assert task_manager.pending_count == 0
        assert task_manager.get_task_count() == 0
        assert task_manager._task_metadata == {}

    @pytest.mark.asyncio
    async def test_wait_for_all_timeout_leaves_no_residue(self, task_manager):
        """The timeout path waits for real work, so it must not report residue either."""
        for _ in range(5):
            for _ in range(20):

                async def job():
                    await asyncio.sleep(0.001)

                task_manager.create_task(job())
            await asyncio.sleep(0.01)

        await task_manager.wait_for_all(timeout=15)

        assert task_manager.pending_count == 0
        assert task_manager._task_metadata == {}

    @pytest.mark.asyncio
    async def test_shutdown_reports_accurate_completion_counts(self, task_manager):
        """shutdown() derives completed_tasks from len(_tasks) after gather.

        ``cancel_all`` already awaits, which incidentally drains the done-callbacks,
        so this passes either way today. It is here to pin the reported statistics so
        a future change to that scheduling cannot silently make them lie.
        """

        async def job():
            await asyncio.sleep(0.001)

        for _ in range(10):
            task_manager.create_task(job())

        stats = await task_manager.shutdown()

        assert stats["status"] == "completed"
        assert stats["initial_tasks"] == 10
        assert stats["completed_tasks"] == 10
        assert stats["remaining_tasks"] == 0

    @pytest.mark.asyncio
    async def test_high_load_concurrent_tasks(self, task_manager):
        """Test BackgroundTaskManager under high load with many concurrent tasks."""
        num_tasks = 100
        results = []
        
        async def load_task(task_id):
            # Simulate some work
            await asyncio.sleep(0.01)
            results.append(task_id)
        
        start_time = time.time()
        
        # Create many tasks quickly
        for i in range(num_tasks):
            task_manager.create_task(load_task(i))
        
        # Wait for all to complete
        await task_manager.wait_for_all()
        
        end_time = time.time()
        
        # Verify all tasks completed
        assert len(results) == num_tasks
        assert sorted(results) == list(range(num_tasks))
        assert len(task_manager._tasks) == 0
        
        # Should complete in reasonable time (concurrent execution)
        # With 100 tasks of 0.01s each, sequential would take 1s+
        # Concurrent should be much faster
        assert end_time - start_time < 0.5


    @pytest.mark.asyncio
    async def test_rapid_task_creation_and_completion(self, task_manager):
        """Test rapid creation and completion of tasks."""
        completion_count = 0
        
        async def rapid_task():
            nonlocal completion_count
            completion_count += 1
        
        # Rapidly create and let tasks complete
        for _ in range(50):
            task_manager.create_task(rapid_task())
            await asyncio.sleep(0.001)  # Small delay between creations
        
        await task_manager.wait_for_all()
        
        assert completion_count == 50
        assert len(task_manager._tasks) == 0



    @pytest.mark.asyncio
    async def test_task_done_callback_thread_safety(self, task_manager):
        """Test that the done callback is thread-safe with concurrent task completions."""
        completion_order = []
        
        async def callback_test_task(task_id):
            await asyncio.sleep(0.01 * (task_id % 3))  # Variable delay
            completion_order.append(task_id)
        
        # Create tasks with different completion times
        for i in range(20):
            task_manager.create_task(callback_test_task(i))
        
        await task_manager.wait_for_all()
        
        # All tasks should complete
        assert len(completion_order) == 20
        assert len(task_manager._tasks) == 0
        # Order might vary due to different delays, but all should be present
        assert sorted(completion_order) == list(range(20))


    @pytest.mark.asyncio
    async def test_exception_types_handling(self, task_manager):
        """Test handling of different exception types."""
        with patch('tenxgraph.utils.background_task_manager.logger') as mock_logger:
            
            async def value_error_task():
                raise ValueError("Value error")
            
            async def runtime_error_task():
                raise RuntimeError("Runtime error")
            
            async def custom_error_task():
                class CustomError(Exception):
                    pass
                raise CustomError("Custom error")
            
            task_manager.create_task(value_error_task())
            task_manager.create_task(runtime_error_task())
            task_manager.create_task(custom_error_task())
            
            # wait_for_all should raise the first exception (ValueError)
            with pytest.raises(ValueError, match="Value error"):
                await task_manager.wait_for_all()
            
            # At least one error should be logged
            assert mock_logger.error.call_count >= 1

    @pytest.mark.asyncio
    async def test_stress_test_rapid_create_destroy(self, task_manager):
        """Stress test with rapid task creation and destruction."""
        total_created = 0
        total_completed = 0
        
        async def stress_task():
            nonlocal total_completed
            await asyncio.sleep(0.001)
            total_completed += 1
        
        # Rapidly create tasks in bursts
        for burst in range(5):
            for _ in range(20):
                task_manager.create_task(stress_task())
                total_created += 1
            
            # Let some tasks complete
            await asyncio.sleep(0.01)
        
        # Wait for all remaining tasks
        await task_manager.wait_for_all()
        
        assert total_created == 100
        assert total_completed == 100
        assert len(task_manager._tasks) == 0

    def test_initialization(self):
        """Test BackgroundTaskManager initialization."""
        manager = BackgroundTaskManager()
        assert isinstance(manager._tasks, set)
        assert len(manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_logging_configuration(self, task_manager):
        """Test that logging works correctly under load."""
        with patch('tenxgraph.utils.background_task_manager.logger') as mock_logger:
            
            async def logging_task():
                await asyncio.sleep(0.01)
                raise Exception("Test logging")
            
            # Create multiple failing tasks
            for _ in range(5):
                task_manager.create_task(logging_task())
            
            # wait_for_all should raise the first exception
            with pytest.raises(Exception, match="Test logging"):
                await task_manager.wait_for_all()
            
            # At least one error should be logged
            assert mock_logger.error.call_count >= 1
            
            # Check log message format for the first logged error
            first_call = mock_logger.error.call_args_list[0]
            args = first_call[0]
            assert "Background task raised an exception" in args[0]

@pytest.mark.asyncio
async def test_shutdown_lets_tasks_finish_within_timeout():
    """BD5: shutdown waits up to the timeout before cancelling, so short tasks complete."""
    done = []

    async def work(i):
        await asyncio.sleep(0.2)
        done.append(i)

    manager = BackgroundTaskManager()
    for i in range(3):
        manager.create_task(work(i), name=f"work-{i}")

    result = await manager.shutdown(timeout=2.0)

    assert sorted(done) == [0, 1, 2]
    assert result["status"] == "completed"
    assert result["remaining_tasks"] == 0


@pytest.mark.asyncio
async def test_shutdown_cancels_tasks_left_after_timeout():
    manager = BackgroundTaskManager()
    task = manager.create_task(asyncio.sleep(30), name="slow")

    result = await manager.shutdown(timeout=0.1)

    assert result["status"] == "timeout"
    await asyncio.sleep(0.05)
    assert task.cancelled()
