"""Additional tests for callbacks.py to cover uncovered lines."""

import pytest

from tenxgraph.utils.callbacks import (
    AfterInvokeCallback,
    BaseValidator,
    BeforeInvokeCallback,
    CallbackContext,
    CallbackManager,
    InvocationType,
    OnErrorCallback,
)
from tenxgraph.core.state.message import Message


# Concrete implementations of abstract classes for testing
class ConcreteBeforeCallback(BeforeInvokeCallback):
    """Concrete BeforeInvokeCallback for testing."""

    def __init__(self, transform=None):
        self._transform = transform or (lambda data: data)

    async def __call__(self, context, input_data):
        return self._transform(input_data)


class ConcreteAfterCallback(AfterInvokeCallback):
    """Concrete AfterInvokeCallback for testing."""

    def __init__(self, transform=None):
        self._transform = transform or (lambda inp, out: out)

    async def __call__(self, context, input_data, output_data):
        return self._transform(input_data, output_data)


class ConcreteErrorCallback(OnErrorCallback):
    """Concrete OnErrorCallback for testing."""

    def __init__(self, recovery=None):
        self._recovery = recovery

    async def __call__(self, context, input_data, error):
        return self._recovery


class ConcreteValidator(BaseValidator):
    """Concrete BaseValidator for testing."""

    def __init__(self, should_pass=True):
        self._should_pass = should_pass

    async def validate(self, messages):
        if not self._should_pass:
            raise ValueError("Validation failed")
        return True


class TestBeforeInvokeCallbackAbstract:
    """Test BeforeInvokeCallback abstract class."""

    def test_abstract_cannot_instantiate(self):
        """Test that BeforeInvokeCallback is abstract."""
        with pytest.raises(TypeError):
            BeforeInvokeCallback()

    def test_concrete_implementation_works(self):
        """Test concrete implementation can be instantiated."""
        cb = ConcreteBeforeCallback()
        assert callable(cb)

    @pytest.mark.asyncio
    async def test_concrete_implementation_called(self):
        """Test concrete implementation executes."""
        cb = ConcreteBeforeCallback(transform=lambda data: f"modified_{data}")
        context = CallbackContext(InvocationType.AI, "node")
        result = await cb(context, "input")
        assert result == "modified_input"


class TestAfterInvokeCallbackAbstract:
    """Test AfterInvokeCallback abstract class."""

    def test_abstract_cannot_instantiate(self):
        """Test that AfterInvokeCallback is abstract."""
        with pytest.raises(TypeError):
            AfterInvokeCallback()

    def test_concrete_implementation_works(self):
        """Test concrete implementation can be instantiated."""
        cb = ConcreteAfterCallback()
        assert callable(cb)

    @pytest.mark.asyncio
    async def test_concrete_implementation_called(self):
        """Test concrete implementation executes."""
        cb = ConcreteAfterCallback(transform=lambda inp, out: f"{inp}+{out}")
        context = CallbackContext(InvocationType.TOOL, "node")
        result = await cb(context, "input", "output")
        assert result == "input+output"


class TestOnErrorCallbackAbstract:
    """Test OnErrorCallback abstract class."""

    def test_abstract_cannot_instantiate(self):
        """Test that OnErrorCallback is abstract."""
        with pytest.raises(TypeError):
            OnErrorCallback()

    @pytest.mark.asyncio
    async def test_concrete_implementation_returns_recovery(self):
        """Test concrete error callback returns recovery value."""
        recovery_msg = Message.text_message("Recovery message", role="assistant")
        cb = ConcreteErrorCallback(recovery=recovery_msg)
        context = CallbackContext(InvocationType.AI, "node")
        result = await cb(context, "input", Exception("error"))
        assert result is recovery_msg


class TestBaseValidatorAbstract:
    """Test BaseValidator abstract class."""

    def test_abstract_cannot_instantiate(self):
        """Test that BaseValidator is abstract."""
        with pytest.raises(TypeError):
            BaseValidator()

    @pytest.mark.asyncio
    async def test_concrete_validator_passes(self):
        """Test concrete validator passes message validation."""
        validator = ConcreteValidator(should_pass=True)
        result = await validator.validate([Message.text_message("Test")])
        assert result is True

    @pytest.mark.asyncio
    async def test_concrete_validator_fails(self):
        """Test concrete validator can fail validation."""
        validator = ConcreteValidator(should_pass=False)
        with pytest.raises(ValueError, match="Validation failed"):
            await validator.validate([Message.text_message("Bad content")])


class TestCallbackManagerBeforeInvokeClass:
    """Test CallbackManager.execute_before_invoke with class-based callbacks."""

    @pytest.mark.asyncio
    async def test_execute_before_invoke_with_class(self):
        """Test execute_before_invoke with class-based callback."""
        manager = CallbackManager()
        cb = ConcreteBeforeCallback(transform=lambda data: f"modified_{data}")
        context = CallbackContext(InvocationType.AI, "node")
        manager.register_before_invoke(InvocationType.AI, cb)
        result = await manager.execute_before_invoke(context, "data")
        assert result == "modified_data"

    @pytest.mark.asyncio
    async def test_execute_before_invoke_with_async_callable(self):
        """Test execute_before_invoke with async callable (not class)."""
        manager = CallbackManager()

        async def async_cb(ctx, data):
            return f"async_{data}"

        context = CallbackContext(InvocationType.AI, "node")
        manager.register_before_invoke(InvocationType.AI, async_cb)
        result = await manager.execute_before_invoke(context, "input")
        assert result == "async_input"

    @pytest.mark.asyncio
    async def test_execute_before_invoke_exception_triggers_error_handler(self):
        """Test that exception in before_invoke triggers error handler."""
        manager = CallbackManager()

        context = CallbackContext(InvocationType.AI, "node")

        def failing_cb(ctx, data):
            raise ValueError("before invoke failed")

        manager.register_before_invoke(InvocationType.AI, failing_cb)

        with pytest.raises(ValueError, match="before invoke failed"):
            await manager.execute_before_invoke(context, "data")


class TestCallbackManagerAfterInvokeClass:
    """Test CallbackManager.execute_after_invoke with class-based callbacks."""

    @pytest.mark.asyncio
    async def test_execute_after_invoke_with_class(self):
        """Test execute_after_invoke with class-based callback."""
        manager = CallbackManager()
        cb = ConcreteAfterCallback(transform=lambda inp, out: f"{out}_transformed")
        context = CallbackContext(InvocationType.TOOL, "node")
        manager.register_after_invoke(InvocationType.TOOL, cb)
        result = await manager.execute_after_invoke(context, "input", "output")
        assert result == "output_transformed"

    @pytest.mark.asyncio
    async def test_execute_after_invoke_with_sync_callable(self):
        """Test execute_after_invoke with sync callable."""
        manager = CallbackManager()

        def sync_cb(ctx, inp, out):
            return f"sync_{out}"

        context = CallbackContext(InvocationType.MCP, "node")
        manager.register_after_invoke(InvocationType.MCP, sync_cb)
        result = await manager.execute_after_invoke(context, "input", "output")
        assert result == "sync_output"

    @pytest.mark.asyncio
    async def test_execute_after_invoke_with_async_callable(self):
        """Test execute_after_invoke with async callable."""
        manager = CallbackManager()

        async def async_cb(ctx, inp, out):
            return f"async_{out}"

        context = CallbackContext(InvocationType.AI, "node")
        manager.register_after_invoke(InvocationType.AI, async_cb)
        result = await manager.execute_after_invoke(context, "in", "out")
        assert result == "async_out"

    @pytest.mark.asyncio
    async def test_execute_after_invoke_exception_triggers_error(self):
        """Test exception in after_invoke triggers error handler and re-raises."""
        manager = CallbackManager()
        context = CallbackContext(InvocationType.AI, "node")

        def failing_cb(ctx, inp, out):
            raise RuntimeError("after invoke error")

        manager.register_after_invoke(InvocationType.AI, failing_cb)
        with pytest.raises(RuntimeError, match="after invoke error"):
            await manager.execute_after_invoke(context, "in", "out")


class TestCallbackManagerOnError:
    """Test CallbackManager.execute_on_error with callable paths."""

    @pytest.mark.asyncio
    async def test_execute_on_error_with_class(self):
        """Test execute_on_error with class-based callback."""
        manager = CallbackManager()
        recovery = Message.text_message("Recovery", role="assistant")
        cb = ConcreteErrorCallback(recovery=recovery)
        context = CallbackContext(InvocationType.AI, "node")
        manager.register_on_error(InvocationType.AI, cb)
        result = await manager.execute_on_error(context, "input", Exception("err"))
        assert result is recovery

    @pytest.mark.asyncio
    async def test_execute_on_error_with_sync_callable(self):
        """Test execute_on_error with sync callable."""
        manager = CallbackManager()
        context = CallbackContext(InvocationType.TOOL, "node")

        def error_cb(ctx, inp, err):
            return None

        manager.register_on_error(InvocationType.TOOL, error_cb)
        result = await manager.execute_on_error(context, "input", ValueError("err"))
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_on_error_with_async_callable(self):
        """Test execute_on_error with async callable."""
        manager = CallbackManager()
        context = CallbackContext(InvocationType.MCP, "node")

        async def async_error_cb(ctx, inp, err):
            return None

        manager.register_on_error(InvocationType.MCP, async_error_cb)
        result = await manager.execute_on_error(context, "input", RuntimeError("err"))
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_on_error_callback_failure_logged(self):
        """Test that failing error callbacks are logged but not re-raised."""
        manager = CallbackManager()
        context = CallbackContext(InvocationType.AI, "node")

        def bad_error_cb(ctx, inp, err):
            raise RuntimeError("error callback also failed")

        manager.register_on_error(InvocationType.AI, bad_error_cb)
        # Should not raise, just log the failure
        result = await manager.execute_on_error(context, "input", ValueError("original"))
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_on_error_returns_message(self):
        """Test execute_on_error returns Message value."""
        manager = CallbackManager()
        context = CallbackContext(InvocationType.AI, "node")
        recovery_msg = Message.text_message("Recovered", role="assistant")

        async def error_cb(ctx, inp, err):
            return recovery_msg

        manager.register_on_error(InvocationType.AI, error_cb)
        result = await manager.execute_on_error(context, "input", Exception("err"))
        assert isinstance(result, Message)


class TestCallbackManagerValidators:
    """Test CallbackManager.register_input_validator and execute_validators."""

    @pytest.mark.asyncio
    async def test_execute_validators_no_validators(self):
        """Test execute_validators with no validators passes."""
        manager = CallbackManager()
        result = await manager.execute_validators([Message.text_message("Test")])
        assert result is True

    @pytest.mark.asyncio
    async def test_register_and_execute_validator(self):
        """Test registering and executing a validator."""
        manager = CallbackManager()
        validator = ConcreteValidator(should_pass=True)
        manager.register_input_validator(validator)
        result = await manager.execute_validators([Message.text_message("Valid message")])
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_validators_fails(self):
        """Test execute_validators propagates validation error."""
        manager = CallbackManager()
        validator = ConcreteValidator(should_pass=False)
        manager.register_input_validator(validator)
        with pytest.raises(ValueError, match="Validation failed"):
            await manager.execute_validators([Message.text_message("Bad message")])

    @pytest.mark.asyncio
    async def test_execute_validators_multiple_validators(self):
        """Test execute_validators with multiple validators."""
        manager = CallbackManager()
        manager.register_input_validator(ConcreteValidator(should_pass=True))
        manager.register_input_validator(ConcreteValidator(should_pass=True))
        result = await manager.execute_validators([Message.text_message("Test")])
        assert result is True

    def test_register_multiple_validators(self):
        """Test registering multiple validators."""
        manager = CallbackManager()
        v1 = ConcreteValidator()
        v2 = ConcreteValidator()
        manager.register_input_validator(v1)
        manager.register_input_validator(v2)
        assert len(manager._validators) == 2
