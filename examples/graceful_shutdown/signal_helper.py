"""Let 10xGraph close the graph on SIGTERM and SIGINT (opt-in).

install_shutdown_signals() connects both signals to graph.aclose() on the running loop.
Core never installs it for you. Run: python examples/graceful_shutdown/signal_helper.py
then press Ctrl+C while the run is going.
"""

import asyncio
import json

from tenxgraph.core.exceptions import GraphClosedError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.shutdown import install_shutdown_signals


async def slow_work(state: AgentState):
    await asyncio.sleep(30)  # stands in for a long model call or tool
    return Message.text_message("done", role="assistant")


async def main() -> None:
    graph = StateGraph()
    graph.add_node("WORK", slow_work)
    graph.add_edge("WORK", END)
    graph.set_entry_point("WORK")
    app = graph.compile(checkpointer=InMemoryCheckpointer(), shutdown_timeout=12.0)

    # Main thread, inside the running loop. Defaults: 6s grace, 3s stop, 3s cancel and drain.
    shutdown = install_shutdown_signals(app, grace=2.0, stop_timeout=3.0, timeout=3.0)
    print("Running. Press Ctrl+C: the run is stopped and saved INTERRUPTED.")
    try:
        result = await app.ainvoke(
            {"messages": [Message.text_message("hi")]}, {"thread_id": "long-job"}
        )
        print("finished:", result["messages"][-1].text())
    except GraphClosedError:
        print("refused: the graph is closing")
    finally:
        stats = await shutdown.wait()  # stats of the close a signal started, else None
        shutdown.uninstall()
        print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
