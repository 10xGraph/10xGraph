import asyncio
import logging
from typing import Any

from dotenv import load_dotenv

from tenxgraph.core import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


load_dotenv()

checkpointer = InMemoryCheckpointer()


def get_weather(
    location: str,
    tool_call_id: str,
    state: AgentState,
) -> Message:
    """
    Get the current weather for a specific location.
    Demonstrates injectable parameters: tool_call_id and state are automatically injected.
    """
    if tool_call_id:
        logging.debug("[tool] Tool call ID: %s", tool_call_id)
    if state and hasattr(state, "context"):
        logging.debug("[tool] Context messages: %s", len(state.context))  # type: ignore

    res = f"The weather in {location} is sunny."
    return Message.tool_message(
        content=res,
        tool_call_id=tool_call_id,  # type: ignore
    )


tool_node = ToolNode([get_weather])

# Debug: Print registered tools
print("Registered tools:", list(tool_node._funcs.keys()))


# Create the main agent using Agent class
main_agent = Agent(
    model="gemini-2.5-flash",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": """
                You are a helpful assistant.
                Answer conversationally. Use tools when needed.
            """,
        },
    ],
    tools=tool_node,
    trim_context=True,
)


def should_use_tools(state: AgentState) -> str:
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    if last_message.role == "tool":
        return END

    return END


graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)

graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)

graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")


app = graph.compile(
    checkpointer=checkpointer,
)


async def run_stream_test() -> None:
    inp = {"messages": [Message.text_message("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "stream-1", "recursion_limit": 10}

    logging.info("--- streaming start ---")
    stream_gen = app.astream(
        inp,
        config=config,
        response_granularity=ResponseGranularity.LOW,
    )
    async for chunk in stream_gen:
        print(chunk.model_dump(), end="\n", flush=True)


async def run_sync_test() -> None:
    """Test sync main_agent implementation"""
    # Create a graph with sync main_agent using Agent class
    sync_agent = Agent(
        model="gemini-2.5-flash",
        provider="google",
        system_prompt=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant.
                    Answer conversationally. Use tools when needed.
                """,
            },
        ],
        tools=tool_node,
        trim_context=True,
    )

    sync_graph = StateGraph()
    sync_graph.add_node("MAIN", sync_agent)
    sync_graph.add_node("TOOL", tool_node)

    sync_graph.add_conditional_edges(
        "MAIN",
        should_use_tools,
        {"TOOL": "TOOL", END: END},
    )

    sync_graph.add_edge("TOOL", "MAIN")
    sync_graph.set_entry_point("MAIN")

    sync_app = sync_graph.compile(
        checkpointer=checkpointer,
    )

    inp = {"messages": [Message.text_message("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "sync-1", "recursion_limit": 10}

    logging.info("--- sync test start ---")
    stream_gen = sync_app.astream(
        inp,
        config=config,
        response_granularity=ResponseGranularity.LOW,
    )
    async for chunk in stream_gen:
        print(chunk.model_dump(), end="\n", flush=True)


async def run_sync_stream_test() -> None:
    """Test sync stream main_agent implementation"""
    # Create a graph with sync stream main_agent using Agent class
    sync_stream_agent = Agent(
        model="gemini-2.5-flash",
        provider="google",
        system_prompt=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant.
                    Answer conversationally. Use tools when needed.
                """,
            },
        ],
        tools=tool_node,
        trim_context=True,
    )

    sync_stream_graph = StateGraph()
    sync_stream_graph.add_node("MAIN", sync_stream_agent)
    sync_stream_graph.add_node("TOOL", tool_node)

    sync_stream_graph.add_conditional_edges(
        "MAIN",
        should_use_tools,
        {"TOOL": "TOOL", END: END},
    )

    sync_stream_graph.add_edge("TOOL", "MAIN")
    sync_stream_graph.set_entry_point("MAIN")

    sync_stream_app = sync_stream_graph.compile(
        checkpointer=checkpointer,
    )

    inp = {"messages": [Message.text_message("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "sync-stream-1", "recursion_limit": 10}

    logging.info("--- sync stream test start ---")
    stream_gen = sync_stream_app.astream(
        inp,
        config=config,
        response_granularity=ResponseGranularity.LOW,
    )
    async for chunk in stream_gen:
        print(chunk.model_dump(), end="\n", flush=True)


async def run_non_stream_test() -> None:
    """Test non-streaming main_agent implementation"""
    # Create a graph with non-streaming main_agent using Agent class
    non_stream_agent = Agent(
        model="gemini-2.5-flash",
        provider="google",
        system_prompt=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant.
                    Answer conversationally. Use tools when needed.
                """,
            },
        ],
        tools=tool_node,
        trim_context=True,
    )

    non_stream_graph = StateGraph()
    non_stream_graph.add_node("MAIN", non_stream_agent)
    non_stream_graph.add_node("TOOL", tool_node)

    non_stream_graph.add_conditional_edges(
        "MAIN",
        should_use_tools,
        {"TOOL": "TOOL", END: END},
    )

    non_stream_graph.add_edge("TOOL", "MAIN")
    non_stream_graph.set_entry_point("MAIN")

    non_stream_app = non_stream_graph.compile(
        checkpointer=checkpointer,
    )

    inp = {"messages": [Message.text_message("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "non-stream-1", "recursion_limit": 10}

    logging.info("--- non-stream test start ---")
    stream_gen = non_stream_app.astream(
        inp,
        config=config,
        response_granularity=ResponseGranularity.LOW,
    )
    async for chunk in stream_gen:
        print(chunk.model_dump(), end="\n", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "sync":
            asyncio.run(run_sync_test())
        elif test_type == "non-stream":
            asyncio.run(run_non_stream_test())
        elif test_type == "sync-stream":
            asyncio.run(run_sync_stream_test())
        else:
            logging.info("Usage: python stream_react_agent.py [sync|non-stream|sync-stream]")
            logging.info("Running default streaming test...")
            asyncio.run(run_stream_test())
    else:
        asyncio.run(run_stream_test())
