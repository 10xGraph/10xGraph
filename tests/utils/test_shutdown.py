"""Tests for graceful shutdown utilities."""

import asyncio
import signal

import pytest

from tenxgraph.utils.shutdown import (
    SIGNAL_NAMES,
    DelayedKeyboardInterrupt,
    GracefulShutdownManager,
    delayed_keyboard_interrupt,
    setup_exception_handler,
    shutdown_with_timeout,
)


class TestDelayedKeyboardInterrupt:
    """Test DelayedKeyboardInterrupt context manager."""

    def test_enter_exit_no_signal(self):
        """Test entering and exiting without receiving a signal."""
        dki = DelayedKeyboardInterrupt()
        with dki:
            pass  # No signal received
        # Should complete without error

    def test_enter_stores_old_handlers(self):
        """Test that entering the context manager stores old handlers."""
        dki = DelayedKeyboardInterrupt()
        with dki:
            assert len(dki._old_signal_handlers) > 0
            assert signal.SIGINT in dki._old_signal_handlers
            assert signal.SIGTERM in dki._old_signal_handlers

    def test_handler_stores_signal(self):
        """Test that _handler stores the signal and frame."""
        dki = DelayedKeyboardInterrupt()
        dki._handler(signal.SIGINT, None)
        assert dki._sig == signal.SIGINT
        assert dki._frame is None

    def test_handler_stores_sigterm(self):
        """Test that _handler stores SIGTERM."""
        dki = DelayedKeyboardInterrupt()
        dki._handler(signal.SIGTERM, None)
        assert dki._sig == signal.SIGTERM

    def test_exit_restores_handlers(self):
        """Test that exit restores original signal handlers."""
        original_sigint = signal.getsignal(signal.SIGINT)
        dki = DelayedKeyboardInterrupt()
        with dki:
            pass
        # Verify handler was restored
        restored = signal.getsignal(signal.SIGINT)
        assert restored == original_sigint

    def test_context_manager_returns_self(self):
        """Test that __enter__ returns self."""
        dki = DelayedKeyboardInterrupt()
        result = dki.__enter__()
        dki.__exit__(None, None, None)
        assert result is dki

    def test_init_defaults(self):
        """Test initial state of DelayedKeyboardInterrupt."""
        dki = DelayedKeyboardInterrupt()
        assert dki._sig is None
        assert dki._frame is None
        assert dki._old_signal_handlers == {}


class TestDelayedKeyboardInterruptFunction:
    """Test delayed_keyboard_interrupt context manager function."""

    def test_context_manager(self):
        """Test the functional context manager."""
        with delayed_keyboard_interrupt() as dki:
            assert isinstance(dki, DelayedKeyboardInterrupt)

    def test_no_exception_raised(self):
        """Test using delayed_keyboard_interrupt without signals."""
        called = False
        with delayed_keyboard_interrupt():
            called = True
        assert called


class TestGracefulShutdownManager:
    """Test GracefulShutdownManager class."""

    def test_init_defaults(self):
        """Test default initialization."""
        manager = GracefulShutdownManager()
        assert manager.shutdown_requested is False
        assert manager.shutdown_timeout == 30.0
        assert manager._original_handlers == {}
        assert manager._shutdown_callbacks == []

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        manager = GracefulShutdownManager(shutdown_timeout=60.0)
        assert manager.shutdown_timeout == 60.0

    def test_add_shutdown_callback(self):
        """Test adding shutdown callbacks."""
        manager = GracefulShutdownManager()
        callback = lambda: None
        manager.add_shutdown_callback(callback)
        assert callback in manager._shutdown_callbacks

    def test_add_multiple_callbacks(self):
        """Test adding multiple shutdown callbacks."""
        manager = GracefulShutdownManager()
        cb1 = lambda: None
        cb2 = lambda: None
        manager.add_shutdown_callback(cb1)
        manager.add_shutdown_callback(cb2)
        assert len(manager._shutdown_callbacks) == 2

    def test_signal_handler_sets_flag(self):
        """Test that _signal_handler sets shutdown_requested."""
        manager = GracefulShutdownManager()
        manager._signal_handler(signal.SIGINT)
        assert manager.shutdown_requested is True

    def test_signal_handler_calls_callbacks(self):
        """Test that _signal_handler calls registered callbacks."""
        manager = GracefulShutdownManager()
        called = []
        manager.add_shutdown_callback(lambda: called.append(True))
        manager._signal_handler(signal.SIGTERM)
        assert len(called) == 1

    def test_signal_handler_handles_callback_error(self):
        """Test that _signal_handler handles callback exceptions gracefully."""
        manager = GracefulShutdownManager()

        def failing_callback():
            raise RuntimeError("callback error")

        manager.add_shutdown_callback(failing_callback)
        # Should not raise
        manager._signal_handler(signal.SIGINT)
        assert manager.shutdown_requested is True

    def test_protect_section(self):
        """Test protect_section returns DelayedKeyboardInterrupt."""
        manager = GracefulShutdownManager()
        ctx = manager.protect_section()
        assert isinstance(ctx, DelayedKeyboardInterrupt)

    def test_protect_section_usable(self):
        """Test protect_section is usable as context manager."""
        manager = GracefulShutdownManager()
        with manager.protect_section():
            pass  # Should work without error

    @pytest.mark.asyncio
    async def test_wait_for_shutdown(self):
        """Test wait_for_shutdown waits until shutdown is requested."""
        manager = GracefulShutdownManager()

        async def trigger_shutdown():
            await asyncio.sleep(0.05)
            manager.shutdown_requested = True

        asyncio.create_task(trigger_shutdown())
        await manager.wait_for_shutdown(check_interval=0.01)
        assert manager.shutdown_requested is True

    @pytest.mark.asyncio
    async def test_register_signal_handlers(self):
        """Test register_signal_handlers works with a running loop."""
        manager = GracefulShutdownManager()
        loop = asyncio.get_event_loop()
        manager.register_signal_handlers(loop=loop)
        # Clean up
        manager.unregister_signal_handlers()

    @pytest.mark.asyncio
    async def test_register_signal_handlers_no_loop(self):
        """Test register_signal_handlers without explicitly providing loop."""
        manager = GracefulShutdownManager()
        manager.register_signal_handlers()  # Uses running loop
        manager.unregister_signal_handlers()

    def test_unregister_signal_handlers_empty(self):
        """Test unregister_signal_handlers with no handlers registered."""
        manager = GracefulShutdownManager()
        manager.unregister_signal_handlers()  # Should not raise

    @pytest.mark.asyncio
    async def test_unregister_signal_handlers(self):
        """Test that unregister_signal_handlers restores original handlers."""
        manager = GracefulShutdownManager()
        original_sigint = signal.getsignal(signal.SIGINT)
        loop = asyncio.get_event_loop()
        manager.register_signal_handlers(loop=loop)
        manager.unregister_signal_handlers()
        restored = signal.getsignal(signal.SIGINT)
        assert restored == original_sigint


class TestSetupExceptionHandler:
    """Test setup_exception_handler function."""

    @pytest.mark.asyncio
    async def test_setup_exception_handler(self):
        """Test that exception handler is set on the loop."""
        loop = asyncio.get_event_loop()
        setup_exception_handler(loop)
        # Verify handler was set (just checks it doesn't raise)
        assert loop.get_exception_handler() is not None

    @pytest.mark.asyncio
    async def test_exception_handler_suppresses_connection_reset(self):
        """Test exception handler suppresses ConnectionResetError."""
        loop = asyncio.get_event_loop()
        setup_exception_handler(loop)

        handler = loop.get_exception_handler()
        assert handler is not None

        # Test with ConnectionResetError (should just log and return)
        context = {
            "exception": ConnectionResetError("connection reset"),
            "message": "socket closed",
        }
        handler(loop, context)  # Should not raise

    @pytest.mark.asyncio
    async def test_exception_handler_logs_other_errors(self):
        """Test exception handler logs non-suppressed errors."""
        loop = asyncio.get_event_loop()
        setup_exception_handler(loop)

        handler = loop.get_exception_handler()

        context = {
            "exception": ValueError("some error"),
            "message": "unhandled exception",
        }
        handler(loop, context)  # Should not raise, just logs


class TestShutdownWithTimeout:
    """Test shutdown_with_timeout function."""

    @pytest.mark.asyncio
    async def test_shutdown_coroutine_success(self):
        """Test shutdown of coroutine that completes successfully."""

        async def quick_task():
            await asyncio.sleep(0.01)

        result = await shutdown_with_timeout(quick_task(), timeout=5.0, task_name="quick")
        assert result["status"] == "completed"
        assert "duration" in result

    @pytest.mark.asyncio
    async def test_shutdown_task_success(self):
        """Test shutdown of asyncio Task that completes."""

        async def quick_task():
            await asyncio.sleep(0.01)

        task = asyncio.create_task(quick_task())
        result = await shutdown_with_timeout(task, timeout=5.0)
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_shutdown_timeout(self):
        """Test shutdown that times out."""

        async def slow_task():
            await asyncio.sleep(100)

        result = await shutdown_with_timeout(slow_task(), timeout=0.05)
        assert result["status"] == "timeout"
        assert "duration" in result

    @pytest.mark.asyncio
    async def test_shutdown_task_timeout_cancels(self):
        """Test that timed-out task gets cancelled."""

        async def slow_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(slow_task())
        result = await shutdown_with_timeout(task, timeout=0.05)
        assert result["status"] == "timeout"
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_error(self):
        """Test shutdown that raises an exception."""

        async def failing_task():
            raise ValueError("task failed")

        result = await shutdown_with_timeout(failing_task(), timeout=5.0)
        assert result["status"] == "error"
        assert "task failed" in result["error"]

    @pytest.mark.asyncio
    async def test_signal_names_mapping(self):
        """Test SIGNAL_NAMES mapping is set up correctly."""
        assert signal.SIGINT in SIGNAL_NAMES
        assert signal.SIGTERM in SIGNAL_NAMES
        assert SIGNAL_NAMES[signal.SIGINT] == "SIGINT"
        assert SIGNAL_NAMES[signal.SIGTERM] == "SIGTERM"
