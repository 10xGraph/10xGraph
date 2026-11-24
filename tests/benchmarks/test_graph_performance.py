"""Time per graph run on a long thread (PERFORMANCE_PLAN.md P-1).

The earlier version of this file handed an ``async def`` to ``benchmark.pedantic``, which
timed creating the coroutine, not running it (NEW-PERF-01). These tests await every run
and time it with ``time.perf_counter``, and check that the nodes really ran.

Budget: 3 function nodes, ``InMemoryCheckpointer``, 1,000 messages of history. The run
measured about 17 ms (median of 15) on a developer machine, 27 ms under coverage tracing;
the plan's target is 20 ms. ``BUDGET_MS`` is about 3x the traced time, so a busy CI runner
does not fail it, while a return of the per-step deep copies (about 370 ms) does.
"""

import statistics
import time

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import END


BUDGET_MS = 80.0
RUNS = 15


async def _median_run_ms(history: int) -> tuple[float, int]:
    """Median time of one 3-node run on a thread holding ``history`` messages."""
    calls = 0

    async def node(state: AgentState):
        nonlocal calls
        calls += 1

    graph = StateGraph()
    for name in ("A", "B", "C"):
        graph.add_node(name, node)
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", END)
    graph.set_entry_point("A")
    app = graph.compile(checkpointer=InMemoryCheckpointer())
    config = {"thread_id": f"bench-{history}", "user_id": "bench"}

    seed = [Message.text_message(f"m{i} " + "x" * 200) for i in range(history)]
    await app.ainvoke({"messages": seed}, config)

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        await app.ainvoke({"messages": [Message.text_message("next")]}, config)
        times.append((time.perf_counter() - start) * 1000)
    await app.aclose()
    return statistics.median(times), calls


@pytest.mark.slow
@pytest.mark.asyncio
async def test_runs_are_timed_not_coroutine_creation():
    median_ms, calls = await _median_run_ms(1)
    assert calls == 3 * (RUNS + 1)
    assert median_ms > 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_three_node_run_on_1000_messages_stays_in_budget():
    median_ms, _ = await _median_run_ms(1000)
    assert median_ms < BUDGET_MS, f"3-node run on 1,000 messages took {median_ms:.1f} ms"
