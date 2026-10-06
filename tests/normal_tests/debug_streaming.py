"""Simple streaming test to debug the API."""

import asyncio

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState
from tenxgraph.utils import END, Message


def simple_node(state: AgentState) -> str:
    """Simple node that returns a string."""
    return "Hello from simple node"


def test_streaming_debug():
    """Debug streaming API."""
    graph = StateGraph(AgentState)
    graph.add_node("simple", simple_node)
    graph.set_entry_point("simple")
    graph.add_edge("simple", END)

    checkpointer = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=checkpointer)

    input_data = {"messages": [Message.from_text("Test", role="user")]}
    config = {"thread_id": "debug_test"}

    print("Testing sync streaming...")
    try:
        chunks = list(compiled.stream(input_data, config))
        print(f"Sync streaming returned {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {type(chunk)} - {chunk}")
    except Exception as e:
        print(f"Sync streaming error: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()

    print("\nTesting async streaming...")
    try:

        async def test_async():
            chunks = []
            stream_gen = await compiled.astream(input_data, config)
            async for chunk in stream_gen:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(test_async())
        print(f"Async streaming returned {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {type(chunk)} - {chunk}")
    except Exception as e:
        print(f"Async streaming error: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_streaming_debug()
