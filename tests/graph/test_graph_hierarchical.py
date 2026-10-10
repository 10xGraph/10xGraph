"""Hierarchical multi-agent stress tests for the graph runtime.

These tests emulate a supervisor-led team where synchronous nodes call into
async helpers and vice versa. The goal is to pressure the execution pipeline
with nested graph invocations, mixed sync/async behavior, and recursion
pressure while verifying that TAF handles the orchestration cleanly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import Field

from tenxgraph.core.graph import CompiledGraph, StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import Command, END, ResponseGranularity, add_messages


class HierarchyState(AgentState):
    """Augmented state for hierarchical agent coordination."""

    plan: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    supervisor_runs: int = 0
    trace: list[str] = Field(default_factory=list)


def _build_hierarchical_graph() -> CompiledGraph[HierarchyState]:
    """Create a compiled graph representing a supervisor with two workers."""

    state = HierarchyState()
    graph = StateGraph[HierarchyState](state)

    def supervisor_node(state: HierarchyState, config: dict[str, Any]) -> Command[HierarchyState]:
        """Synchronous node that orchestrates planning via asyncio."""

        async def build_plan() -> list[str]:
            await asyncio.sleep(0)
            return ["research_agent", "writing_agent"]

        plan = asyncio.run(build_plan())
        state.plan = plan
        state.supervisor_runs += 1
        state.trace.append("supervisor")
        return Command(update=state, goto="router")

    async def router_node(state: HierarchyState, config: dict[str, Any]) -> Command[HierarchyState]:
        """Async router that decides which worker should act next."""

        state.trace.append(f"router:{len(state.plan)}")
        if not state.plan:
            return Command(update=state, goto=END)
        return Command(update=state, goto=state.plan[0])

    async def research_agent(state: HierarchyState, config: dict[str, Any]) -> Command[HierarchyState]:
        """Async research agent that uses sync helpers in a thread."""

        state.trace.append("research")
        prompt = state.context[-1].text() if state.context else "unknown request"

        def run_sync_research() -> Message:
            return Message.text_message(f"[research] {prompt.upper()}", role="assistant")

        research_message = await asyncio.to_thread(run_sync_research)
        state.completed.append("research_agent")
        if state.plan:
            state.plan.pop(0)
        state.context = add_messages(state.context, [research_message])
        return Command(update=state, goto="router")

    async def writing_agent(state: HierarchyState, config: dict[str, Any]) -> Command[HierarchyState]:
        """Async writing agent that drives a nested sync graph."""

        state.trace.append("writing")
        outline = state.context[-1].text() if state.context else "no context"

        def run_inner_sync_graph() -> Message:
            inner_state = AgentState()
            inner_graph = StateGraph[AgentState](inner_state)

            def build_draft(state: AgentState, config: dict[str, Any]) -> Command[AgentState]:
                draft = Message.text_message(f"[draft] {outline}", role="assistant")
                state.context = add_messages(state.context, [draft])
                return Command(update=state, goto=END)

            inner_graph.add_node("draft", build_draft)
            inner_graph.set_entry_point("draft")
            inner_graph.add_edge("draft", END)
            inner_compiled = inner_graph.compile()
            
            # Create isolated config with unique thread_id to avoid state pollution
            import uuid
            inner_config = {
                "thread_id": f"inner_graph_{uuid.uuid4()}",
                "user_id": config.get("user_id", "test-user"),
                "run_id": f"inner_run_{uuid.uuid4()}",
            }
            
            result = inner_compiled.invoke(
                {"messages": [Message.text_message("kickoff", role="user")]},
                config=inner_config,
                response_granularity=ResponseGranularity.FULL,
            )
            # Get the last message from the inner state context, not messages array
            inner_state_result = result["state"]
            return inner_state_result.context[-1]  # The draft message should be the last in context

        writing_message = await asyncio.to_thread(run_inner_sync_graph)
        state.completed.append("writing_agent")
        if state.plan:
            state.plan.pop(0)
        state.artifacts.append(writing_message.text())
        state.context = add_messages(state.context, [writing_message])
        return Command(update=state, goto="router")

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("router", router_node)
    graph.add_node("research_agent", research_agent)
    graph.add_node("writing_agent", writing_agent)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "router")
    graph.add_edge("router", "research_agent")
    graph.add_edge("router", "writing_agent")
    graph.add_edge("research_agent", "router")
    graph.add_edge("writing_agent", "router")
    graph.add_edge("router", END)

    return graph.compile()


def test_hierarchical_graph_sync_invoke_handles_mixed_execution() -> None:
    """Ensure synchronous invoke manages async planning and nested sync work."""

    compiled = _build_hierarchical_graph()
    user_message = Message.text_message("Create a launch memo", role="user")
    
    # Use a unique thread_id to ensure isolated execution
    import uuid
    thread_id = f"test_sync_{uuid.uuid4()}"
    
    result = compiled.invoke(
        {"messages": [user_message]},
        config={"thread_id": thread_id},
        response_granularity=ResponseGranularity.FULL,
    )

    messages = result["messages"]
    state: HierarchyState = result["state"]
    


    assert state.supervisor_runs == 1  # noqa: S101
    assert state.plan == []  # noqa: S101
    assert state.completed == ["research_agent", "writing_agent"]  # noqa: S101
    assert state.trace[:2] == ["supervisor", "router:2"]  # noqa: S101
    # Check messages in state context instead of response messages
    assert any(msg.text().startswith("[research]") for msg in state.context)  # noqa: S101
    assert any(msg.text().startswith("[draft]") for msg in state.context)  # noqa: S101


@pytest.mark.asyncio
async def test_hierarchical_graph_async_invoke_clears_plan_and_tracks_artifacts() -> None:
    """Ensure async invoke correctly handles hierarchical execution with artifact tracking."""

    compiled = _build_hierarchical_graph()
    user_message = Message.text_message("Create a product specification", role="user")
    
    # Use a unique thread_id to ensure isolated execution 
    import uuid
    thread_id = f"test_async_{uuid.uuid4()}"
    
    result = await compiled.ainvoke(
        {"messages": [user_message]},
        config={"thread_id": thread_id},
        response_granularity=ResponseGranularity.FULL,
    )

    state: HierarchyState = result["state"]
    messages = result["messages"]

    assert not state.plan  # noqa: S101
    assert state.completed == ["research_agent", "writing_agent"]  # noqa: S101
    assert state.artifacts and "[draft]" in state.artifacts[-1]  # noqa: S101
    # Check that we have assistant messages in the state context
    assert any(msg.role == "assistant" for msg in state.context)  # noqa: S101
    assert [label for label in state.trace if label.startswith("router")]  # noqa: S101
    # Context should include the original user message plus two agent outputs
    assert len(state.context) >= 3  # noqa: S101


