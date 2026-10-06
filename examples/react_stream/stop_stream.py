import asyncio
import logging
import threading
import time

from dotenv import load_dotenv

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


# Example: Stop a running streaming graph from the frontend (or caller).
# This demonstrates how to request a stop using CompiledGraph.stop while a
# graph is streaming responses.


logging.basicConfig(level=logging.INFO)
load_dotenv()

checkpointer = InMemoryCheckpointer()


def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: AgentState | None = None,
) -> str:
    # trivial tool
    return f"The weather in {location} is sunny"


async def main_agent(state: AgentState, config: dict | None = None):
    # Produce a few messages with small delays to emulate streaming work
    # We'll return a simple list of Message objects for clarity
    for idx in range(50):
        await asyncio.sleep(1)
        yield Message.text_message(f"Chunk {idx + 1} from MAIN")


def should_use_tools(state: AgentState) -> str:
    # Simple router; never actually uses tool here, just loops to END
    return END


def build_app():
    graph = StateGraph()
    graph.add_node("MAIN", main_agent)
    graph.add_conditional_edges("MAIN", should_use_tools, {END: END})
    graph.set_entry_point("MAIN")
    return graph.compile(checkpointer=checkpointer)


def run_and_stop():
    app = build_app()
    inp = {"messages": [Message.text_message("Start streaming and then stop")]}
    config = {"thread_id": "stop-demo-thread", "recursion_limit": 10, "is_stream": True}

    def reader():
        for chunk in app.stream(inp, config=config):
            logging.info("STREAM: %s", getattr(chunk, "content", chunk))

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # Let it run briefly, then request stop
    time.sleep(1.0)
    status = app.stop(config)
    logging.info("Requested stop: %s", status)

    # Give it a moment to finish
    t.join(timeout=5)


if __name__ == "__main__":
    run_and_stop()
