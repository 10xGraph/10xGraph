"""Comprehensive tests for the Router prebuilt agent."""

import pytest
from unittest.mock import Mock

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import ToolNode, CompiledGraph
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.message import Message
from tenxgraph.utils import  END


@pytest.mark.skip(reason="RouterAgent not yet implemented in this feature branch")
class TestRouterAgent:
    """Test the RouterAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state = AgentState()
        self.router_agent = RouterAgent[AgentState](state=self.state)
        
    def test_init_default(self):
        """Test RouterAgent initialization with defaults."""
        agent = RouterAgent[AgentState]()
        assert agent is not None
        assert agent._graph is not None
        
    def test_init_with_state(self):
        """Test RouterAgent initialization with custom state."""
        state = AgentState()
        agent = RouterAgent[AgentState](state=state)
        assert agent is not None
        assert agent._graph is not None
        
    def test_compile_single_route(self):
        """Test router compilation with single route (auto-condition)."""
        def mock_router(state: AgentState) -> AgentState:
            # Router node just passes through
            return state
            
        def mock_search(state: AgentState) -> AgentState:
            # Search route
            result_msg = Message.text_message("Search completed", role="assistant")
            state.context.append(result_msg)
            return state
        
        compiled = self.router_agent.compile(
            router_node=mock_router,
            routes={"search": mock_search},
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_multiple_routes_with_condition(self):
        """Test router compilation with multiple routes and condition."""
        def mock_router(state: AgentState) -> AgentState:
            # Router analyzes the query and sets a routing key
            if state.context and "search" in state.context[0].text().lower():
                object.__setattr__(state, 'routing_key', "search")
            elif state.context and "summarize" in state.context[0].text().lower():
                object.__setattr__(state, 'routing_key', "summarize")
            else:
                object.__setattr__(state, 'routing_key', END)
            return state
            
        def mock_search(state: AgentState) -> AgentState:
            result = Message.text_message("Search results found", role="assistant")
            state.context.append(result)
            return state
            
        def mock_summarize(state: AgentState) -> AgentState:
            result = Message.text_message("Content summarized", role="assistant")
            state.context.append(result)
            return state
            
        def route_condition(state: AgentState) -> str:
            return getattr(state, 'routing_key', END)
        
        compiled = self.router_agent.compile(
            router_node=mock_router,
            routes={
                "search": mock_search,
                "summarize": mock_summarize,
            },
            condition=route_condition,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_tool_node_routes(self):
        """Test router compilation with ToolNode routes."""
        def mock_router(state: AgentState) -> AgentState:
            object.__setattr__(state, 'routing_key', "tool_search")
            return state
            
        def search_tool(query: str) -> str:
            return f"Tool search results for: {query}"
            
        search_node = ToolNode([search_tool])
            
        def route_condition(state: AgentState) -> str:
            return getattr(state, 'routing_key', END)
        
        compiled = self.router_agent.compile(
            router_node=mock_router,
            routes={"tool_search": search_node},
            condition=route_condition,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_tuple_nodes(self):
        """Test router compilation with tuple node names."""
        def mock_router(state: AgentState) -> AgentState:
            object.__setattr__(state, 'routing_key', "CUSTOM_SEARCH")
            return state
            
        def mock_search(state: AgentState) -> AgentState:
            return state
            
        def route_condition(state: AgentState) -> str:
            return getattr(state, 'routing_key', END)
        
        compiled = self.router_agent.compile(
            router_node=(mock_router, "CUSTOM_ROUTER"),
            routes={"search_route": (mock_search, "CUSTOM_SEARCH")},
            condition=route_condition,
            path_map={"CUSTOM_SEARCH": "CUSTOM_SEARCH", END: END},
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_path_map(self):
        """Test router compilation with custom path mapping."""
        def mock_router(state: AgentState) -> AgentState:
            object.__setattr__(state, 'routing_key', "DO_SEARCH")  # Different from node name
            return state
            
        def mock_search(state: AgentState) -> AgentState:
            return state
            
        def route_condition(state: AgentState) -> str:
            return getattr(state, 'routing_key', END)
        
        compiled = self.router_agent.compile(
            router_node=mock_router,
            routes={"search": mock_search},
            condition=route_condition,
            path_map={"DO_SEARCH": "search", END: END},  # Maps condition output to node name
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_checkpointer(self):
        """Test router compilation with checkpointer."""
        def mock_router(state: AgentState) -> AgentState:
            return state
            
        def mock_search(state: AgentState) -> AgentState:
            return state
            
        checkpointer = InMemoryCheckpointer[AgentState]()
        
        compiled = self.router_agent.compile(
            router_node=mock_router,
            routes={"search": mock_search},
            checkpointer=checkpointer,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_interrupts(self):
        """Test router compilation with interrupt configurations."""
        def mock_router(state: AgentState) -> AgentState:
            return state
            
        def mock_search(state: AgentState) -> AgentState:
            return state
        
        compiled = self.router_agent.compile(
            router_node=mock_router,
            routes={"search": mock_search},
            interrupt_before=["ROUTER"],
            interrupt_after=["search"],
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_empty_routes_error(self):
        """Test error handling for empty routes dict."""
        def mock_router(state: AgentState) -> AgentState:
            return state
        
        with pytest.raises(ValueError, match="routes must be a non-empty dict"):
            self.router_agent.compile(
                router_node=mock_router,
                routes={},
            )
            
    def test_compile_invalid_router_tuple(self):
        """Test error handling for invalid router in tuple format."""
        def mock_search(state: AgentState) -> AgentState:
            return state
        
        with pytest.raises(ValueError, match="router_node\\[0\\] must be callable"):
            # Type ignore for intentional error testing
            self.router_agent.compile(
                router_node=("not_callable", "ROUTER"),  # type: ignore
                routes={"search": mock_search},
            )
            
    def test_compile_invalid_router_direct(self):
        """Test error handling for invalid router as direct value."""
        def mock_search(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="router_node must be callable"):
            # Type ignore for intentional error testing
            self.router_agent.compile(
                router_node="not_callable",  # type: ignore
                routes={"search": mock_search},
            )
            
    def test_compile_invalid_route_tuple(self):
        """Test error handling for invalid route in tuple format."""
        def mock_router(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="Route 'search'\\[0\\] must be callable or ToolNode"):
            # Type ignore for intentional error testing
            self.router_agent.compile(
                router_node=mock_router,
                routes={"search": ("not_callable", "SEARCH_NODE")},  # type: ignore
            )
            
    def test_compile_invalid_route_direct(self):
        """Test error handling for invalid route as direct value."""
        def mock_router(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="Route 'search' must be callable or ToolNode"):
            # Type ignore for intentional error testing
            self.router_agent.compile(
                router_node=mock_router,
                routes={"search": "not_callable"},  # type: ignore
            )
            
    def test_compile_multiple_routes_no_condition_error(self):
        """Test error for multiple routes without condition."""
        def mock_router(state: AgentState) -> AgentState:
            return state
            
        def mock_search(state: AgentState) -> AgentState:
            return state
            
        def mock_summarize(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="condition must be provided when multiple routes are defined"):
            self.router_agent.compile(
                router_node=mock_router,
                routes={
                    "search": mock_search,
                    "summarize": mock_summarize,
                },
                # No condition provided for multiple routes
            )


@pytest.mark.skip(reason="RouterAgent not yet implemented in this feature branch")
class TestRouterAgentIntegration:
    """Integration tests for RouterAgent."""
    
    def test_single_route_execution_flow(self):
        """Test basic router execution with single route."""
        def mock_router(state: AgentState) -> AgentState:
            # Router just passes control to the single route
            counter = getattr(state, 'router_calls', 0) + 1
            object.__setattr__(state, 'router_calls', counter)
            return state
            
        def mock_search(state: AgentState) -> AgentState:
            # Search route adds a result and signals completion
            if getattr(state, 'router_calls', 0) >= 2:  # Stop after 2 router calls
                object.__setattr__(state, 'should_end', True)
            result = Message.text_message(f"Search completed (call {getattr(state, 'router_calls', 0)})", role="assistant")
            state.context.append(result)
            return state
            
        def single_route_condition(state: AgentState) -> str:
            # Stop if we've made enough calls
            if getattr(state, 'should_end', False):
                return END
            return "search"
        
        router_agent = RouterAgent[AgentState]()
        compiled = router_agent.compile(
            router_node=mock_router,
            routes={"search": mock_search},
            condition=single_route_condition,
        )
        
        # Execute the router agent
        initial_state = {"messages": [Message.text_message("Please search for information", role="user")]}
        result = compiled.invoke(initial_state, config={"thread_id": "router_test"})
        
        assert isinstance(result, dict)
        assert "messages" in result
        
    def test_multi_route_execution_flow(self):
        """Test router execution with multiple routes."""
        def mock_router(state: AgentState) -> AgentState:
            # Router analyzes the latest user message to determine routing
            if hasattr(state, 'task_complete') and getattr(state, 'task_complete', False):
                object.__setattr__(state, 'routing_key', END)
                return state
                
            if state.context:
                latest_msg = state.context[-1].text().lower()
                if "search" in latest_msg:
                    object.__setattr__(state, 'routing_key', "search")
                elif "summarize" in latest_msg:
                    object.__setattr__(state, 'routing_key', "summarize")
                else:
                    object.__setattr__(state, 'routing_key', END)
            return state

        def mock_search(state: AgentState) -> AgentState:
            result = Message.text_message("Search results found and processed", role="assistant")
            state.context.append(result)
            # Mark task as complete to end routing loop
            object.__setattr__(state, 'task_complete', True)
            return state

        def mock_summarize(state: AgentState) -> AgentState:
            result = Message.text_message("Content has been summarized", role="assistant")
            state.context.append(result)
            # Mark task as complete to end routing loop
            object.__setattr__(state, 'task_complete', True)
            return state

        def multi_route_condition(state: AgentState) -> str:
            return getattr(state, 'routing_key', END)

        router_agent = RouterAgent[AgentState]()
        compiled = router_agent.compile(
            router_node=mock_router,
            routes={
                "search": mock_search,
                "summarize": mock_summarize,
            },
            condition=multi_route_condition,
        )

        # Test search routing
        search_state = {"messages": [Message.text_message("Please search for recent papers", role="user")]}
        search_result = compiled.invoke(search_state, config={"thread_id": "search_test"})
        assert isinstance(search_result, dict)
        assert "messages" in search_result
        
        # Test summarize routing
        summarize_state = {"messages": [Message.text_message("Please summarize this document", role="user")]}
        summarize_result = compiled.invoke(summarize_state, config={"thread_id": "summarize_test"})
        
        assert isinstance(summarize_result, dict)
        assert "messages" in summarize_result
        
    @pytest.mark.asyncio
    async def test_router_async_execution(self):
        """Test router agent with async execution."""
        async def mock_async_router(state: AgentState) -> AgentState:
            # Check if task is complete, if so end the routing
            if hasattr(state, 'task_complete') and getattr(state, 'task_complete', False):
                object.__setattr__(state, 'routing_key', END)
            else:
                object.__setattr__(state, 'routing_key', "async_search")
            return state

        async def mock_async_search(state: AgentState) -> AgentState:
            result = Message.text_message("Async search completed", role="assistant")
            state.context.append(result)
            # Mark task as complete to end routing loop
            object.__setattr__(state, 'task_complete', True)
            return state

        def async_condition(state: AgentState) -> str:
            return getattr(state, 'routing_key', END)

        router_agent = RouterAgent[AgentState]()
        compiled = router_agent.compile(
            router_node=mock_async_router,
            routes={"async_search": mock_async_search},
            condition=async_condition,
        )

        # Execute the router agent asynchronously
        initial_state = {"messages": [Message.text_message("Async router test", role="user")]}
        result = await compiled.ainvoke(initial_state, config={"thread_id": "async_router_test"})
        assert isinstance(result, dict)
        assert "messages" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])