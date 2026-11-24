"""What ``astream`` sends to a client: the granularity filter and the final summary chunk.

``is_pause_chunk`` stays importable from here (the API imports it).
"""

from __future__ import annotations

from typing import Any

from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.utils import ResponseGranularity

from .engine import RunResult


PAUSE_KINDS = ("interrupt", "before", "after", "remote_tool")


def is_pause_chunk(chunk: StreamChunk) -> bool:
    """True for any pause update: ``interrupt()``, interrupt_before/after, remote-tool wait.

    It is a control signal (the run stopped and waits for a resume), so it reaches clients at
    every response granularity, unlike the other ``UPDATES`` progress chunks.
    """
    data = chunk.data
    return (
        chunk.event == StreamEvent.UPDATES
        and isinstance(data, dict)
        and (
            data.get("status") == "interrupted"
            or "interrupt" in data
            or data.get("pause") in PAUSE_KINDS
        )
    )


def filter_chunk(chunk: StreamChunk, granularity: ResponseGranularity) -> bool:
    """Whether ``chunk`` reaches a client that asked for ``granularity``."""
    if granularity == ResponseGranularity.FULL:
        return True
    if granularity == ResponseGranularity.PARTIAL:
        return chunk.event != StreamEvent.UPDATES or is_pause_chunk(chunk)
    return chunk.event in (StreamEvent.MESSAGE, StreamEvent.ERROR) or is_pause_chunk(chunk)


def final_chunk(result: RunResult, config: dict[str, Any]) -> StreamChunk:
    """The summary chunk a FULL-granularity stream ends with."""
    state = result.state
    return StreamChunk(
        event=StreamEvent.UPDATES,
        state=state,
        data={
            "status": "graph_invoked",
            "reason": "Graph execution finished",
            "time_taken": result.time_taken,
            "is_interrupted": state.is_interrupted(),
            "total_messages": len(state.context) if state.context else 0,
            **result.token_usage,
        },
        thread_id=config.get("thread_id"),
        run_id=config.get("run_id"),
    )
