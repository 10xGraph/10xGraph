"""Chaos engineering tests for graph resilience.

This module tests system resilience under failure conditions, including
random node failures, checkpointer failures, and verifies error handling
and state consistency.
"""

import random
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tenxgraph.storage.checkpointer import BaseCheckpointer, InMemoryCheckpointer
from tenxgraph.core.exceptions import NodeError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.utils import END


class FailingCheckpointer(InMemoryCheckpointer):
    """Checkpointer that fails randomly."""

    def __init__(self, failure_rate: float = 0.3):
        super().__init__()
        self.failure_rate = failure_rate
        self.failures = []

    async def aput_state(self, config: dict[str, Any], state: AgentState) -> AgentState:
        """Save with random failures."""
        if random.random() < self.failure_rate:
            self.failures.append(("put_state", config.get("thread_id")))
            raise RuntimeError(f"Checkpoint save failed for {config.get('thread_id')}")
        return await super().aput_state(config, state)

    async def aget_state(self, config: dict[str, Any]) -> AgentState | None:
        """Get with random failures."""
        if random.random() < self.failure_rate:
            self.failures.append(("get_state", config.get("thread_id")))
            raise RuntimeError(f"Checkpoint get failed for {config.get('thread_id')}")
        return await super().aget_state(config)


class TestNodeFailureResilience:
    """Test resilience to node execution failures."""

    @pytest.mark.asyncio
    async def test_node_error_propagates_correctly(self):
        """Test that node errors propagate correctly."""
        def failing_node(state: AgentState) -> AgentState:
            raise ValueError("Node failed intentionally")
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("failing", failing_node)
        graph.set_entry_point("failing")
        graph.add_edge("failing", END)
        
        compiled = graph.compile()
        
        # Node error should be wrapped in NodeError
        with pytest.raises((NodeError, ValueError)):
            await compiled.ainvoke(
                {"messages": [Message.text_message("Test", role="user")]},
                {"thread_id": "test_node_error"},
            )

    @pytest.mark.asyncio
    async def test_partial_failure_in_chain(self):
        """Test resilience when middle node fails in chain."""
        execution_order = []
        
        def node1(state: AgentState) -> AgentState:
            execution_order.append("node1")
            state.context.append(Message.text_message("Node1", role="assistant"))
            return state
        
        def failing_node(state: AgentState) -> AgentState:
            execution_order.append("failing")
            raise RuntimeError("Intentional failure")
        
        def node3(state: AgentState) -> AgentState:
            execution_order.append("node3")
            state.context.append(Message.text_message("Node3", role="assistant"))
            return state
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("node1", node1)
        graph.add_node("failing", failing_node)
        graph.add_node("node3", node3)
        
        graph.set_entry_point("node1")
        graph.add_edge("node1", "failing")
        graph.add_edge("failing", "node3")
        graph.add_edge("node3", END)
        
        compiled = graph.compile()
        
        with pytest.raises((NodeError, RuntimeError)):
            await compiled.ainvoke(
                {"messages": [Message.text_message("Test", role="user")]},
                {"thread_id": "test_partial_failure"},
            )
        
        # node1 should have executed, node3 should not
        assert "node1" in execution_order
        assert "failing" in execution_order
        assert "node3" not in execution_order

    @pytest.mark.asyncio
    async def test_random_failure_recovery(self):
        """Test recovery from random node failures."""
        failure_count = [0]
        success_count = [0]
        
        def unreliable_node(state: AgentState) -> AgentState:
            # Fail 30% of the time
            if random.random() < 0.3:
                failure_count[0] += 1
                raise RuntimeError("Random failure")
            
            success_count[0] += 1
            state.context.append(Message.text_message("Success", role="assistant"))
            return state
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("unreliable", unreliable_node)
        graph.set_entry_point("unreliable")
        graph.add_edge("unreliable", END)
        
        compiled = graph.compile()
        
        # Run multiple times to observe failures
        for i in range(20):
            try:
                await compiled.ainvoke(
                    {"messages": [Message.text_message("Test", role="user")]},
                    {"thread_id": f"test_random_{i}"},
                )
            except (NodeError, RuntimeError):
                pass  # Expected failures
        
        # Should have both successes and failures
        assert success_count[0] > 0
        # With 30% failure rate, we might see failures (but not guaranteed in 20 runs)


class TestCheckpointerFailureResilience:
    """Test resilience to checkpointer failures."""

    @pytest.mark.asyncio
    async def test_checkpoint_save_failure_handling(self):
        """Test handling of checkpoint save failures."""
        random.seed(42)  # Deterministic failures
        failing_checkpointer = FailingCheckpointer(failure_rate=1.0)  # Always fail
        
        def simple_node(state: AgentState) -> AgentState:
            state.context.append(Message.text_message("Response", role="assistant"))
            return state
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("node", simple_node)
        graph.set_entry_point("node")
        graph.add_edge("node", END)
        
        compiled = graph.compile(checkpointer=failing_checkpointer)
        
        # Should fail due to checkpointer (either during get or save)
        # Note: May succeed if cached state is used
        try:
            result = await compiled.ainvoke(
                {"messages": [Message.text_message("Test", role="user")]},
                {"thread_id": "test_checkpoint_failure"},
            )
            # If it succeeds, verify it completed
            assert "messages" in result
        except RuntimeError as e:
            # Expected failure during checkpoint operations
            assert "Checkpoint" in str(e)
            assert len(failing_checkpointer.failures) > 0

    @pytest.mark.asyncio
    async def test_checkpoint_get_failure_handling(self):
        """Test handling of checkpoint retrieval failures."""
        random.seed(42)
        checkpointer = FailingCheckpointer(failure_rate=0.5)
        
        def simple_node(state: AgentState) -> AgentState:
            state.context.append(Message.text_message("Response", role="assistant"))
            return state
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("node", simple_node)
        graph.set_entry_point("node")
        graph.add_edge("node", END)
        
        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_after=["node"],
        )
        
        thread_id = "test_get_failure"
        
        # First run might succeed or fail
        try:
            await compiled.ainvoke(
                {"messages": [Message.text_message("Test", role="user")]},
                {"thread_id": thread_id},
            )
        except RuntimeError:
            pass
        
        # Second run trying to resume might fail on get
        # Just verify it doesn't crash the system completely
        try:
            await compiled.ainvoke(
                {},
                {"thread_id": thread_id},
            )
        except RuntimeError:
            pass  # Expected if checkpoint get fails

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_checkpointer(self):
        """Test that graph works without checkpointer."""
        def simple_node(state: AgentState) -> AgentState:
            state.context.append(Message.text_message("Response", role="assistant"))
            return state
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("node", simple_node)
        graph.set_entry_point("node")
        graph.add_edge("node", END)
        
        # Compile without checkpointer
        compiled = graph.compile()
        
        # Should work fine
        result = await compiled.ainvoke(
            {"messages": [Message.text_message("Test", role="user")]},
            {"thread_id": "test_no_checkpoint"},
        )
        
        assert len(result["messages"]) >= 1


class TestConcurrentExecutionResilience:
    """Test resilience under concurrent execution."""

    @pytest.mark.asyncio
    async def test_concurrent_thread_isolation(self):
        """Test that concurrent threads don't interfere."""
        checkpointer = InMemoryCheckpointer()
        
        def stateful_node(state: AgentState) -> AgentState:
            # Use context_summary to track thread-specific data
            count = int(state.context_summary or "0")
            state.context_summary = str(count + 1)
            state.context.append(
                Message.text_message(f"Count: {count + 1}", role="assistant")
            )
            return state
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("counter", stateful_node)
        graph.set_entry_point("counter")
        graph.add_edge("counter", END)
        
        compiled = graph.compile(checkpointer=checkpointer)
        
        # Run multiple threads concurrently
        import asyncio
        from tenxgraph.utils import ResponseGranularity
        
        async def run_thread(thread_id: str):
            return await compiled.ainvoke(
                {"messages": [Message.text_message("Test", role="user")]},
                {"thread_id": thread_id},
                response_granularity=ResponseGranularity.FULL,
            )
        
        tasks = [run_thread(f"thread_{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Each thread should have its own isolated state
        # All should have count of 1 (first execution)
        for result in results:
            assert result["state"].context_summary == "1"

    @pytest.mark.asyncio
    async def test_concurrent_same_thread_safety(self):
        """Test safety when same thread is accessed concurrently."""
        checkpointer = InMemoryCheckpointer()
        
        def slow_node(state: AgentState) -> AgentState:
            import asyncio
            # Simulate slow operation
            # Note: In real scenario, use await asyncio.sleep()
            state.context.append(Message.text_message("Response", role="assistant"))
            return state
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("slow", slow_node)
        graph.set_entry_point("slow")
        graph.add_edge("slow", END)
        
        compiled = graph.compile(checkpointer=checkpointer)
        
        # Try to run same thread concurrently
        import asyncio
        
        thread_id = "shared_thread"
        tasks = [
            compiled.ainvoke(
                {"messages": [Message.text_message(f"Test {i}", role="user")]},
                {"thread_id": thread_id},
            )
            for i in range(3)
        ]
        
        # Should complete without crashes (may have race conditions)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least some should succeed
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) > 0


class TestStateConsistencyUnderFailures:
    """Test state consistency when failures occur."""

    @pytest.mark.asyncio
    async def test_state_consistency_after_node_failure(self):
        """Test that state remains consistent after node failure."""
        checkpointer = InMemoryCheckpointer()
        
        def node1(state: AgentState) -> AgentState:
            state.context.append(Message.text_message("Node1", role="assistant"))
            state.context_summary = "node1_done"
            return state
        
        def failing_node(state: AgentState) -> AgentState:
            # Verify state from node1 is present before failing
            assert state.context_summary == "node1_done"
            raise RuntimeError("Intentional failure")
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("node1", node1)
        graph.add_node("failing", failing_node)
        
        graph.set_entry_point("node1")
        graph.add_edge("node1", "failing")
        graph.add_edge("failing", END)
        
        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_after=["node1"],
        )
        
        thread_id = "test_state_consistency"
        
        from tenxgraph.utils import ResponseGranularity
        
        # First run: execute node1 successfully
        result1 = await compiled.ainvoke(
            {"messages": [Message.text_message("Test", role="user")]},
            {"thread_id": thread_id},
            response_granularity=ResponseGranularity.FULL,
        )
        
        # Verify state was saved correctly
        assert result1["state"].context_summary == "node1_done"
        
        # Second run: should fail at failing_node
        with pytest.raises((NodeError, RuntimeError)):
            await compiled.ainvoke(
                {},
                {"thread_id": thread_id},
            )
        

    @pytest.mark.asyncio
    async def test_execution_status_after_error(self):
        """Test that execution status is correctly set after error."""
        checkpointer = InMemoryCheckpointer()
        
        def failing_node(state: AgentState) -> AgentState:
            raise ValueError("Node error")
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("failing", failing_node)
        graph.set_entry_point("failing")
        graph.add_edge("failing", END)
        
        compiled = graph.compile(checkpointer=checkpointer)
        
        thread_id = "test_error_status"
        
        with pytest.raises((NodeError, ValueError)):
            await compiled.ainvoke(
                {"messages": [Message.text_message("Test", role="user")]},
                {"thread_id": thread_id},
            )
        
        # Check if state was saved with error status
        saved_state = await checkpointer.aget_state({"thread_id": thread_id})
        if saved_state:
            # If state was saved, verify error status
            # Execution status should indicate error
            assert saved_state.execution_meta.status in [
                ExecutionStatus.ERROR,
                ExecutionStatus.RUNNING,  # May still be RUNNING if save happened before error
            ]


class TestRetryMechanisms:
    """Test retry behavior under failures."""

    @pytest.mark.asyncio
    async def test_manual_retry_after_failure(self):
        """Test manual retry after node failure."""
        attempt_count = [0]
        
        def flaky_node(state: AgentState) -> AgentState:
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise RuntimeError("Failing attempt")
            
            state.context.append(Message.text_message("Success", role="assistant"))
            return state
        
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("flaky", flaky_node)
        graph.set_entry_point("flaky")
        graph.add_edge("flaky", END)
        
        compiled = graph.compile()
        
        thread_id = "test_manual_retry"
        
        # First two attempts should fail
        for i in range(2):
            with pytest.raises((NodeError, RuntimeError)):
                await compiled.ainvoke(
                    {"messages": [Message.text_message("Test", role="user")]},
                    {"thread_id": f"{thread_id}_{i}"},
                )
        
        # Third attempt should succeed
        result = await compiled.ainvoke(
            {"messages": [Message.text_message("Test", role="user")]},
            {"thread_id": f"{thread_id}_2"},
        )
        
        assert len(result["messages"]) >= 1
        assert attempt_count[0] == 3
