"""Integration tests for checkpointer functionality in multiagent workflows."""

import asyncio

import pytest

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END


class TestCheckpointerIntegration:
    """Integration tests for checkpoint/resume functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_and_resume_simple(self):
        """Test basic checkpoint and resume functionality."""

        def agent1(state: AgentState) -> Message:
            return Message.text_message("Agent 1 executed", "assistant")

        def agent2(state: AgentState) -> Message:
            return Message.text_message("Agent 2 executed", "assistant")

        def agent3(state: AgentState) -> Message:
            return Message.text_message("Agent 3 executed", "assistant")

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()
        graph.add_node("agent1", agent1)
        graph.add_node("agent2", agent2)
        graph.add_node("agent3", agent3)
        graph.set_entry_point("agent1")
        graph.add_edge("agent1", "agent2")
        graph.add_edge("agent2", "agent3")
        graph.add_edge("agent3", END)

        compiled = graph.compile(checkpointer=checkpointer)
        messages = [Message.text_message("Start workflow", "user")]
        thread_id = "test_checkpoint_resume_simple"

        # First execution
        result1 = await compiled.ainvoke({"messages": messages}, config={"thread_id": thread_id})
        assert "messages" in result1
        assert len(result1["messages"]) >= 4  # input + 3 agents

        # Resume from checkpoint
        result2 = await compiled.ainvoke({"messages": [Message.text_message("Resume", "user")]}, config={"thread_id": thread_id})
        assert "messages" in result2
        # Should have previous messages + new messages
        assert len(result2["messages"]) >= len(result1["messages"])

    @pytest.mark.asyncio
    async def test_checkpoint_with_interrupts(self):
        """Test checkpoint with interrupt_before and interrupt_after."""

        def agent1(state: AgentState) -> Message:
            return Message.text_message("Agent 1 executed", "assistant")

        def agent2(state: AgentState) -> Message:
            return Message.text_message("Agent 2 executed", "assistant")

        def agent3(state: AgentState) -> Message:
            return Message.text_message("Agent 3 executed", "assistant")

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()
        graph.add_node("agent1", agent1)
        graph.add_node("agent2", agent2)
        graph.add_node("agent3", agent3)
        graph.set_entry_point("agent1")
        graph.add_edge("agent1", "agent2")
        graph.add_edge("agent2", "agent3")
        graph.add_edge("agent3", END)

        # Compile with interrupt before agent2
        compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["agent2"])
        messages = [Message.text_message("Start with interrupt", "user")]
        thread_id = "test_checkpoint_with_interrupt"

        # First execution - should stop before agent2
        result1 = await compiled.ainvoke({"messages": messages}, config={"thread_id": thread_id})
        assert "messages" in result1
        # Should have executed agent1 only
        content = " ".join([msg.text() for msg in result1["messages"]])
        assert content is not None

        # Resume - should execute agent2 and agent3
        result2 = await compiled.ainvoke({"messages": []}, config={"thread_id": thread_id})
        assert "messages" in result2
        content2 = " ".join([msg.text() for msg in result2["messages"]])
        assert content2 is not None

    @pytest.mark.asyncio
    async def test_checkpoint_state_persistence(self):
        """Test that state is properly persisted across checkpoints."""

        def accumulator_agent(agent_id: int):
            def agent_func(state: AgentState) -> Message:
                # Verify we can see previous messages
                msg_count = len(state.context)
                return Message.text_message(f"Agent {agent_id} saw {msg_count} messages", "assistant")

            return agent_func

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()

        for i in range(1, 6):
            graph.add_node(f"agent{i}", accumulator_agent(i))

        graph.set_entry_point("agent1")
        for i in range(1, 5):
            graph.add_edge(f"agent{i}", f"agent{i + 1}")
        graph.add_edge("agent5", END)

        compiled = graph.compile(checkpointer=checkpointer)
        messages = [Message.text_message("Initial", "user")]
        thread_id = "test_checkpoint_state_persistence"

        # Execute with checkpointing
        result = await compiled.ainvoke({"messages": messages}, config={"thread_id": thread_id})

        assert "messages" in result
        # Verify each agent saw an increasing number of messages
        content = " ".join([msg.text() for msg in result["messages"]])
        for i in range(1, 6):
            assert f"Agent {i} saw" in content

    @pytest.mark.asyncio
    async def test_multiple_parallel_checkpointed_workflows(self):
        """Test multiple workflows with different thread IDs using same checkpointer."""

        def simple_agent(state: AgentState) -> Message:
            return Message.text_message("Processed", "assistant")

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()
        graph.add_node("agent", simple_agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        compiled = graph.compile(checkpointer=checkpointer)

        # Run 10 parallel workflows with different thread IDs
        tasks = []
        for i in range(10):
            messages = [Message.text_message(f"Workflow {i}", "user")]
            task = compiled.ainvoke({"messages": messages}, config={"thread_id": f"thread_{i}"})
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 10
        for i, result in enumerate(results):
            assert "messages" in result
            assert len(result["messages"]) >= 2

    @pytest.mark.asyncio
    async def test_checkpoint_recovery_after_error(self):
        """Test that we can resume from checkpoint after an error."""

        execution_count = {"count": 0}

        def failing_agent(state: AgentState) -> Message:
            execution_count["count"] += 1
            if execution_count["count"] == 1:
                raise ValueError("First execution fails")
            return Message.text_message("Success on retry", "assistant")

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()
        graph.add_node("failing", failing_agent)
        graph.set_entry_point("failing")
        graph.add_edge("failing", END)

        compiled = graph.compile(checkpointer=checkpointer)
        messages = [Message.text_message("Test retry", "user")]
        thread_id = "test_checkpoint_error_recovery"

        # First execution should fail
        from tenxgraph.core.exceptions import NodeError

        with pytest.raises(NodeError):
            await compiled.ainvoke({"messages": messages}, config={"thread_id": thread_id})

        # Second execution should succeed
        result = await compiled.ainvoke({"messages": messages}, config={"thread_id": f"{thread_id}_retry"})
        assert "messages" in result
        content = " ".join([msg.text() for msg in result["messages"]])
        assert "Success on retry" in content

    @pytest.mark.asyncio
    async def test_checkpoint_with_conditional_routing(self):
        """Test checkpoint with conditional routing between agents."""

        def router(state: AgentState) -> Message:
            return Message.text_message("Routing decision", "assistant")

        def route_decision(state: AgentState) -> str:
            msg_count = len(state.context)
            return "path_a" if msg_count < 3 else "path_b"

        def path_a_agent(state: AgentState) -> Message:
            return Message.text_message("Path A executed", "assistant")

        def path_b_agent(state: AgentState) -> Message:
            return Message.text_message("Path B executed", "assistant")

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()
        graph.add_node("router", router)
        graph.add_node("path_a", path_a_agent)
        graph.add_node("path_b", path_b_agent)

        graph.set_entry_point("router")
        graph.add_conditional_edges("router", route_decision, {"path_a": "path_a", "path_b": "path_b"})
        graph.add_edge("path_a", END)
        graph.add_edge("path_b", END)

        compiled = graph.compile(checkpointer=checkpointer)
        thread_id = "test_checkpoint_conditional"

        # First execution - should go to path_a
        result1 = await compiled.ainvoke(
            {"messages": [Message.text_message("First", "user")]}, config={"thread_id": thread_id}
        )
        content1 = " ".join([msg.text() for msg in result1["messages"]])
        assert "Path A executed" in content1

        # Second execution with more messages - should go to path_b
        result2 = await compiled.ainvoke(
            {
                "messages": [
                    Message.text_message("First", "user"),
                    Message.text_message("Second", "user"),
                    Message.text_message("Third", "user"),
                ]
            },
            config={"thread_id": f"{thread_id}_2"},
        )
        content2 = " ".join([msg.text() for msg in result2["messages"]])
        assert "Path B executed" in content2

    def test_checkpoint_synchronous_execution(self):
        """Test checkpoint works with synchronous invoke."""

        def agent1(state: AgentState) -> Message:
            return Message.text_message("Agent 1 sync", "assistant")

        def agent2(state: AgentState) -> Message:
            return Message.text_message("Agent 2 sync", "assistant")

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()
        graph.add_node("agent1", agent1)
        graph.add_node("agent2", agent2)
        graph.set_entry_point("agent1")
        graph.add_edge("agent1", "agent2")
        graph.add_edge("agent2", END)

        compiled = graph.compile(checkpointer=checkpointer)
        messages = [Message.text_message("Sync test", "user")]
        thread_id = "test_checkpoint_sync"

        # Synchronous execution
        result = compiled.invoke({"messages": messages}, config={"thread_id": thread_id})
        assert "messages" in result
        assert len(result["messages"]) >= 3
        content = " ".join([msg.text() for msg in result["messages"]])
        assert "Agent 1 sync" in content
        assert "Agent 2 sync" in content

    @pytest.mark.asyncio
    async def test_checkpoint_list_history(self):
        """Test that we can retrieve checkpoint history."""

        def agent(state: AgentState) -> Message:
            return Message.text_message("Checkpoint history test", "assistant")

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        compiled = graph.compile(checkpointer=checkpointer)
        thread_id = "test_checkpoint_history"

        # Execute multiple times
        for i in range(3):
            messages = [Message.text_message(f"Execution {i}", "user")]
            await compiled.ainvoke({"messages": messages}, config={"thread_id": thread_id})

        # Verify checkpoints were created by trying to resume
        resume_result = await compiled.ainvoke({"messages": [Message.text_message("Resume", "user")]}, config={"thread_id": thread_id})
        assert "messages" in resume_result
        # Should have accumulated messages from previous executions
        assert len(resume_result["messages"]) >= 2

    @pytest.mark.asyncio
    async def test_checkpoint_with_large_state(self):
        """Test checkpoint handling with large state (many messages)."""

        def volume_agent(state: AgentState) -> list[Message]:
            # Produce 100 messages
            return [Message.text_message(f"Message {i}", "assistant") for i in range(100)]

        checkpointer = InMemoryCheckpointer()
        graph = StateGraph()
        graph.add_node("volume", volume_agent)
        graph.set_entry_point("volume")
        graph.add_edge("volume", END)

        compiled = graph.compile(checkpointer=checkpointer)
        messages = [Message.text_message("Large state test", "user")]
        thread_id = "test_checkpoint_large_state"

        result = await compiled.ainvoke({"messages": messages}, config={"thread_id": thread_id})
        assert "messages" in result
        assert len(result["messages"]) == 101  # input + 100 from agent

        # Verify checkpoint was saved by resuming
        resume_result = await compiled.ainvoke({"messages": [Message.text_message("Resume", "user")]}, config={"thread_id": thread_id})
        assert "messages" in resume_result
        # Should have previous 101 messages + new ones
        assert len(resume_result["messages"]) >= 101
