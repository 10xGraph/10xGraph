"""Tests for Graph Lifecycle Hooks feature.

Covers:
- GraphLifecycleContext dataclass
- GraphLifecycleHook abstract base class (all 7 hooks)
- CallbackManager.register_lifecycle_hook and fire_* methods
- Integration tests: hooks fire in the right order during invoke/stream
- State mutation propagation through hooks
- Error handling: exceptions in hooks don't crash the graph
- Backward compatibility: no hooks → existing behavior unchanged
"""

from __future__ import annotations

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END
from tenxgraph.utils.callbacks import (
    CallbackManager,
    GraphLifecycleContext,
    GraphLifecycleHook,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_simple_graph(node_fn=None):
    """Build a minimal compiled graph with a single node."""

    def default_node(state: AgentState) -> list[Message]:
        return [Message.text_message("processed", "assistant")]

    fn = node_fn or default_node
    graph = StateGraph[AgentState](AgentState())
    graph.add_node("agent", fn)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


def get_di_callback_manager():
    """Get the CallbackManager from the DI container (must be called after graph.compile())."""
    from injectq import InjectQ

    return InjectQ.get_instance().get(CallbackManager)


def make_input(text: str = "Hello"):
    return {"messages": [Message.text_message(text, "user")]}


# ─── Unit Tests: GraphLifecycleContext ────────────────────────────────────────


class TestGraphLifecycleContext:
    def test_thread_id_from_config(self):
        ctx = GraphLifecycleContext(config={"thread_id": "t1", "run_id": "r1"})
        assert ctx.thread_id == "t1"

    def test_run_id_from_config(self):
        ctx = GraphLifecycleContext(config={"thread_id": "t1", "run_id": "r1"})
        assert ctx.run_id == "r1"

    def test_empty_config_defaults(self):
        ctx = GraphLifecycleContext(config={})
        assert ctx.thread_id == ""
        assert ctx.run_id == ""

    def test_config_accessible(self):
        ctx = GraphLifecycleContext(config={"custom_key": "value"})
        assert ctx.config["custom_key"] == "value"


# ─── Unit Tests: GraphLifecycleHook defaults ──────────────────────────────────


class TestGraphLifecycleHookDefaults:
    """GraphLifecycleHook base class returns None for all methods by default."""

    @pytest.mark.asyncio
    async def test_on_graph_start_default_none(self):
        hook = GraphLifecycleHook()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await hook.on_graph_start(ctx, state)
        assert result is None

    @pytest.mark.asyncio
    async def test_on_graph_end_default_none(self):
        hook = GraphLifecycleHook()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await hook.on_graph_end(ctx, state, [], 0)
        assert result is None

    @pytest.mark.asyncio
    async def test_on_graph_error_default_none(self):
        hook = GraphLifecycleHook()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await hook.on_graph_error(ctx, Exception("e"), state, [], 0, "node")
        assert result is None

    @pytest.mark.asyncio
    async def test_on_interrupt_default_none(self):
        hook = GraphLifecycleHook()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await hook.on_interrupt(ctx, "node", "before", state)
        assert result is None

    @pytest.mark.asyncio
    async def test_on_resume_default_none(self):
        hook = GraphLifecycleHook()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await hook.on_resume(ctx, "node", state, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_on_checkpoint_default_none(self):
        hook = GraphLifecycleHook()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await hook.on_checkpoint(ctx, state, [], False)
        assert result is None

    @pytest.mark.asyncio
    async def test_on_state_update_default_none(self):
        hook = GraphLifecycleHook()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await hook.on_state_update(ctx, "node", state, state, 0)
        assert result is None


# ─── Unit Tests: CallbackManager fire methods ─────────────────────────────────


class TestCallbackManagerFireMethods:
    """Unit tests for the fire_* methods on CallbackManager."""

    @pytest.mark.asyncio
    async def test_fire_on_graph_start_no_hooks_returns_original_state(self):
        mgr = CallbackManager()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_graph_start(ctx, state)
        assert result is state

    @pytest.mark.asyncio
    async def test_fire_on_graph_start_hook_modifies_state(self):
        class ModifyHook(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                state.context_summary = "injected"
                return state

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(ModifyHook())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_graph_start(ctx, state)
        assert result.context_summary == "injected"

    @pytest.mark.asyncio
    async def test_fire_on_graph_start_hook_returns_none_keeps_original(self):
        class NoOpHook(GraphLifecycleHook):
            pass  # all no-op defaults

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(NoOpHook())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_graph_start(ctx, state)
        assert result is state

    @pytest.mark.asyncio
    async def test_fire_on_graph_end_no_hooks_returns_original_state(self):
        mgr = CallbackManager()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_graph_end(ctx, state, [], 1)
        assert result is state

    @pytest.mark.asyncio
    async def test_fire_on_graph_end_modifies_state(self):
        class EndHook(GraphLifecycleHook):
            async def on_graph_end(self, ctx, final_state, messages, total_steps):
                final_state.context_summary = f"done in {total_steps} steps"
                return final_state

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(EndHook())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_graph_end(ctx, state, [], 3)
        assert result.context_summary == "done in 3 steps"

    @pytest.mark.asyncio
    async def test_fire_on_graph_error_no_hooks_returns_original(self):
        mgr = CallbackManager()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        error = ValueError("test")
        result_state, result_msg = await mgr.fire_on_graph_error(
            ctx, error, state, [], 0, "node"
        )
        assert result_state is state
        assert result_msg == str(error)

    @pytest.mark.asyncio
    async def test_fire_on_graph_error_hook_modifies_state(self):
        class ErrorHook(GraphLifecycleHook):
            async def on_graph_error(self, ctx, error, partial_state, messages, step, node_name):
                partial_state.context_summary = f"error at {node_name}"
                return partial_state, "custom error"

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(ErrorHook())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result_state, result_msg = await mgr.fire_on_graph_error(
            ctx, ValueError("e"), state, [], 1, "my_node"
        )
        assert result_state.context_summary == "error at my_node"
        assert result_msg == "custom error"

    @pytest.mark.asyncio
    async def test_fire_on_interrupt_no_hooks_returns_state(self):
        mgr = CallbackManager()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_interrupt(ctx, "node1", "before", state)
        assert result is state

    @pytest.mark.asyncio
    async def test_fire_on_interrupt_hook_can_modify_state(self):
        class InterruptHook(GraphLifecycleHook):
            async def on_interrupt(self, ctx, interrupted_node, interrupt_type, state):
                state.context_summary = f"interrupted at {interrupted_node}"
                return state

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(InterruptHook())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_interrupt(ctx, "agent", "before", state)
        assert result.context_summary == "interrupted at agent"

    @pytest.mark.asyncio
    async def test_fire_on_resume_no_hooks_returns_state(self):
        mgr = CallbackManager()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_resume(ctx, "agent", state, {})
        assert result is state

    @pytest.mark.asyncio
    async def test_fire_on_checkpoint_no_hooks_returns_state_messages(self):
        mgr = CallbackManager()
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        messages = [Message.text_message("hello")]
        result_state, result_msgs = await mgr.fire_on_checkpoint(ctx, state, messages, False)
        assert result_state is state
        assert result_msgs is messages

    @pytest.mark.asyncio
    async def test_fire_on_checkpoint_hook_returns_state_only(self):
        """Hook returning just AgentState keeps original messages."""

        class CheckpointHook(GraphLifecycleHook):
            async def on_checkpoint(self, ctx, state, messages, is_context_trimmed):
                state.context_summary = "checkpointed"
                return state  # return only state, not tuple

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(CheckpointHook())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        messages = [Message.text_message("hello")]
        result_state, result_msgs = await mgr.fire_on_checkpoint(ctx, state, messages, False)
        assert result_state.context_summary == "checkpointed"
        assert result_msgs is messages

    @pytest.mark.asyncio
    async def test_fire_on_checkpoint_hook_returns_tuple(self):
        """Hook returning (state, messages) replaces both."""

        class CheckpointHook(GraphLifecycleHook):
            async def on_checkpoint(self, ctx, state, messages, is_context_trimmed):
                return state, []

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(CheckpointHook())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        messages = [Message.text_message("hello")]
        result_state, result_msgs = await mgr.fire_on_checkpoint(ctx, state, messages, True)
        assert result_state is state
        assert result_msgs == []

    @pytest.mark.asyncio
    async def test_fire_on_state_update_no_hooks_returns_new_state(self):
        mgr = CallbackManager()
        ctx = GraphLifecycleContext(config={})
        old_state = AgentState()
        new_state = AgentState()
        result = await mgr.fire_on_state_update(ctx, "agent", old_state, new_state, 1)
        assert result is new_state

    @pytest.mark.asyncio
    async def test_fire_on_state_update_hook_modifies_state(self):
        class StateUpdateHook(GraphLifecycleHook):
            async def on_state_update(self, ctx, node_name, old_state, new_state, step):
                new_state.context_summary = f"after {node_name} step {step}"
                return new_state

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(StateUpdateHook())
        ctx = GraphLifecycleContext(config={})
        old_state = AgentState()
        new_state = AgentState()
        result = await mgr.fire_on_state_update(ctx, "agent", old_state, new_state, 2)
        assert result.context_summary == "after agent step 2"

    @pytest.mark.asyncio
    async def test_multiple_hooks_chained(self):
        """Multiple hooks are called in order, each receiving previous result."""

        class Hook1(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                state.context_summary = "hook1"
                return state

        class Hook2(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                state.context_summary += "+hook2"
                return state

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(Hook1())
        mgr.register_lifecycle_hook(Hook2())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        result = await mgr.fire_on_graph_start(ctx, state)
        assert result.context_summary == "hook1+hook2"

    @pytest.mark.asyncio
    async def test_hook_exception_is_caught_and_logged(self, caplog):
        """A failing hook logs the error but doesn't crash the fire method."""

        class BrokenHook(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                raise RuntimeError("Hook exploded!")

        mgr = CallbackManager()
        mgr.register_lifecycle_hook(BrokenHook())
        ctx = GraphLifecycleContext(config={})
        state = AgentState()
        import logging

        with caplog.at_level(logging.ERROR):
            result = await mgr.fire_on_graph_start(ctx, state)

        # Fire method should return original state despite hook failure
        assert result is state

    def test_register_lifecycle_hook(self):
        mgr = CallbackManager()
        hook = GraphLifecycleHook()
        assert len(mgr._lifecycle_hooks) == 0
        mgr.register_lifecycle_hook(hook)
        assert len(mgr._lifecycle_hooks) == 1
        assert mgr._lifecycle_hooks[0] is hook

    def test_register_multiple_hooks(self):
        mgr = CallbackManager()
        hook1 = GraphLifecycleHook()
        hook2 = GraphLifecycleHook()
        mgr.register_lifecycle_hook(hook1)
        mgr.register_lifecycle_hook(hook2)
        assert len(mgr._lifecycle_hooks) == 2


# ─── Integration Tests: hooks fire during graph execution ─────────────────────


class TestLifecycleHooksIntegration:
    """Integration tests verifying hooks fire during actual graph execution.

    IMPORTANT: Each test must:
    1. Call make_simple_graph() first (binds a fresh CallbackManager to DI container)
    2. Get the DI manager AFTER compiling (get_di_callback_manager())
    3. Register hooks on that manager
    4. Clean up with mgr._lifecycle_hooks.clear() in finally block
    """

    @pytest.mark.asyncio
    async def test_on_graph_start_fires_before_execution(self):
        call_log = []

        class TrackingHook(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                call_log.append(("on_graph_start", len(state.context)))
                return state

        compiled = make_simple_graph()
        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(TrackingHook())
        try:
            await compiled.ainvoke(make_input("Hello"))
            assert any(event == "on_graph_start" for event, _ in call_log)
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_on_graph_end_fires_after_execution(self):
        call_log = []

        class TrackingHook(GraphLifecycleHook):
            async def on_graph_end(self, ctx, final_state, messages, total_steps):
                call_log.append(("on_graph_end", total_steps))
                return final_state

        compiled = make_simple_graph()
        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(TrackingHook())
        try:
            await compiled.ainvoke(make_input("Hello"))
            assert any(event == "on_graph_end" for event, _ in call_log)
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_on_state_update_fires_after_each_node(self):
        call_log = []

        class TrackingHook(GraphLifecycleHook):
            async def on_state_update(self, ctx, node_name, old_state, new_state, step):
                call_log.append(("on_state_update", node_name, step))
                return new_state

        compiled = make_simple_graph()
        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(TrackingHook())
        try:
            await compiled.ainvoke(make_input("Hello"))
            # Should fire once for the "agent" node
            assert any(
                event == "on_state_update" and node == "agent"
                for event, node, _ in call_log
            )
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_on_graph_error_fires_on_exception(self):
        call_log = []

        class ErrorTrackingHook(GraphLifecycleHook):
            async def on_graph_error(self, ctx, error, partial_state, messages, step, node_name):
                call_log.append(("on_graph_error", str(error), node_name))
                return partial_state, str(error)

        def failing_node(state: AgentState) -> list[Message]:
            raise RuntimeError("intentional failure")

        compiled = make_simple_graph(node_fn=failing_node)
        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(ErrorTrackingHook())
        try:
            with pytest.raises(Exception):
                await compiled.ainvoke(make_input("Hello"))
            assert any(event == "on_graph_error" for event, _, _ in call_log)
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_on_checkpoint_fires_during_execution(self):
        call_log = []

        class CheckpointHook(GraphLifecycleHook):
            async def on_checkpoint(self, ctx, state, messages, is_context_trimmed):
                call_log.append("on_checkpoint")
                return state, messages

        compiled = make_simple_graph()
        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(CheckpointHook())
        try:
            await compiled.ainvoke(make_input("Hello"))
            # Checkpoint fires at graph end (sync_data call)
            assert "on_checkpoint" in call_log
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_on_graph_start_state_modification_propagates(self):
        """State modified by on_graph_start is used during graph execution."""

        class InjectSummaryHook(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                state.context_summary = "pre-injected"
                return state

        captured_states = []

        def capturing_node(state: AgentState) -> list[Message]:
            captured_states.append(state.context_summary)
            return [Message.text_message("done", "assistant")]

        graph = StateGraph[AgentState](AgentState())
        graph.add_node("agent", capturing_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        compiled = graph.compile()

        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(InjectSummaryHook())
        try:
            await compiled.ainvoke(make_input("Hello"))
            # The node should see the state modified by on_graph_start
            assert "pre-injected" in captured_states
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_hook_exception_does_not_crash_graph(self):
        """A hook that raises an exception doesn't crash the graph execution."""

        class BrokenHook(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                raise RuntimeError("Hook failed!")

            async def on_graph_end(self, ctx, final_state, messages, total_steps):
                raise RuntimeError("End hook failed!")

            async def on_state_update(self, ctx, node_name, old_state, new_state, step):
                raise RuntimeError("State update hook failed!")

        compiled = make_simple_graph()
        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(BrokenHook())
        try:
            # Should complete without raising despite hook failures
            result = await compiled.ainvoke(make_input("Hello"))
            assert result is not None
            assert "messages" in result
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_no_hooks_registered_graph_works_normally(self):
        """Without any lifecycle hooks, graph behavior is unchanged."""
        compiled = make_simple_graph()
        result = await compiled.ainvoke(make_input("Hello"))
        assert "messages" in result
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_streaming_lifecycle_hooks_fire(self):
        """Lifecycle hooks also fire during streaming execution."""
        from tenxgraph.utils import ResponseGranularity

        call_log = []

        class TrackingHook(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                call_log.append("on_graph_start")
                return state

            async def on_graph_end(self, ctx, final_state, messages, total_steps):
                call_log.append("on_graph_end")
                return final_state

            async def on_state_update(self, ctx, node_name, old_state, new_state, step):
                call_log.append(f"on_state_update:{node_name}")
                return new_state

        compiled = make_simple_graph()
        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(TrackingHook())
        try:
            chunks = []
            async for chunk in compiled.astream(
                make_input("Hello"), response_granularity=ResponseGranularity.FULL
            ):
                chunks.append(chunk)

            assert "on_graph_start" in call_log
            assert "on_graph_end" in call_log
            assert any("on_state_update:agent" in e for e in call_log)
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_execution_order_start_before_state_update_before_end(self):
        """on_graph_start fires before on_state_update which fires before on_graph_end."""
        call_order = []

        class OrderTracker(GraphLifecycleHook):
            async def on_graph_start(self, ctx, state):
                call_order.append("start")
                return state

            async def on_state_update(self, ctx, node_name, old_state, new_state, step):
                call_order.append("state_update")
                return new_state

            async def on_graph_end(self, ctx, final_state, messages, total_steps):
                call_order.append("end")
                return final_state

        compiled = make_simple_graph()
        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(OrderTracker())
        try:
            await compiled.ainvoke(make_input("Hello"))

            assert call_order.index("start") < call_order.index("state_update")
            assert call_order.index("state_update") < call_order.index("end")
        finally:
            mgr._lifecycle_hooks.clear()

    @pytest.mark.asyncio
    async def test_interrupt_lifecycle_fires_on_interrupt_before(self):
        """on_interrupt fires when a node is interrupted before execution."""
        call_log = []

        class InterruptHook(GraphLifecycleHook):
            async def on_interrupt(self, ctx, interrupted_node, interrupt_type, state):
                call_log.append(("interrupt", interrupted_node, interrupt_type))
                return state

        # Compile with interrupt_before
        graph = StateGraph[AgentState](AgentState())

        def node_fn(state: AgentState) -> list[Message]:
            return [Message.text_message("done", "assistant")]

        graph.add_node("agent", node_fn)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        compiled = graph.compile(interrupt_before=["agent"])

        mgr = get_di_callback_manager()
        mgr.register_lifecycle_hook(InterruptHook())
        try:
            thread_cfg = {"thread_id": "test-interrupt-lifecycle"}
            await compiled.ainvoke(make_input("Hello"), config=thread_cfg)
            # After interrupt fires, on_interrupt should have been called
            assert any(event == "interrupt" for event, _, _ in call_log)
            assert any(itype == "before" for _, _, itype in call_log)
        finally:
            mgr._lifecycle_hooks.clear()


# ─── Unit tests: backward compatibility ───────────────────────────────────────


class TestBackwardCompatibility:
    """Verify existing callback functionality still works with lifecycle hooks present."""

    @pytest.mark.asyncio
    async def test_existing_register_before_invoke_still_works(self):
        from tenxgraph.utils.callbacks import CallbackContext, InvocationType

        fired = []

        async def before_cb(ctx: CallbackContext, data):
            fired.append("before_invoke")
            return data

        mgr = CallbackManager()
        mgr.register_before_invoke(InvocationType.AI, before_cb)
        # Just verify the registration works
        assert len(mgr._before_callbacks[InvocationType.AI]) == 1

    def test_callback_manager_init_has_lifecycle_hooks_list(self):
        mgr = CallbackManager()
        assert hasattr(mgr, "_lifecycle_hooks")
        assert isinstance(mgr._lifecycle_hooks, list)
        assert len(mgr._lifecycle_hooks) == 0

    def test_get_callback_counts_still_works(self):
        mgr = CallbackManager()
        counts = mgr.get_callback_counts()
        assert isinstance(counts, dict)
        assert "ai" in counts

    @pytest.mark.asyncio
    async def test_graph_without_hooks_returns_messages(self):
        """Graph works normally and returns expected messages."""
        compiled = make_simple_graph()
        result = await compiled.ainvoke(make_input("Test"))
        assert "messages" in result
        # Should have both user and assistant messages
        assert len(result["messages"]) >= 1
        # The assistant message from the node should be present
        roles = {m.role for m in result["messages"]}
        assert "assistant" in roles


# ─── Import tests ─────────────────────────────────────────────────────────────


def test_imports_from_agentflow_utils():
    """GraphLifecycleContext and GraphLifecycleHook are importable from tenxgraph.utils."""
    from tenxgraph.utils import GraphLifecycleContext, GraphLifecycleHook  # noqa: F401

    assert GraphLifecycleContext is not None
    assert GraphLifecycleHook is not None


def test_imports_from_callbacks():
    """GraphLifecycleContext and GraphLifecycleHook are importable from callbacks module."""
    from tenxgraph.utils.callbacks import GraphLifecycleContext, GraphLifecycleHook  # noqa: F401

    assert GraphLifecycleContext is not None
    assert GraphLifecycleHook is not None
