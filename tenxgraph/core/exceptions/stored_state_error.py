"""Errors raised when a stored thread cannot be resumed by the deployed graph."""

from typing import Any

from .graph_error import GraphError


class StoredNodeMissingError(GraphError):
    """A stored thread points at a node the deployed graph no longer has.

    This happens after a deploy renamed or removed a node while threads were stored at it.
    Recover by compiling with ``node_aliases={"old_name": "new_name"}``, by restoring the
    node, or by resetting the thread.
    """

    def __init__(
        self,
        node_name: str,
        thread_id: Any = None,
        available_nodes: list[str] | None = None,
        error_code: str = "GRAPH_NODE_MISSING_001",
    ):
        message = (
            f"Thread '{thread_id}' is stored at node '{node_name}', which this graph does not "
            f"have. A deploy probably renamed or removed it. To resume the thread, pass "
            f"node_aliases={{'{node_name}': '<new node name>'}} to compile(), or restore the "
            f"node, or start the thread over with a new thread_id."
        )
        super().__init__(
            message=message,
            error_code=error_code,
            context={
                "node_name": node_name,
                "thread_id": thread_id,
                "available_nodes": sorted(available_nodes or []),
            },
        )
        self.node_name = node_name
        self.thread_id = thread_id


class StateFormatVersionError(GraphError):
    """A stored state was written in a newer format than this version can read."""

    def __init__(
        self,
        found: Any,
        supported: int,
        thread_id: Any = None,
        error_code: str = "STATE_FORMAT_001",
    ):
        who = f"Thread '{thread_id}'" if thread_id is not None else "A stored thread"
        message = (
            f"{who} was stored with state format {found!r}, but this version "
            f"of 10xGraph reads formats up to {supported}. Upgrade 10xgraph to a version that "
            f"supports it; the state is left untouched."
        )
        super().__init__(
            message=message,
            error_code=error_code,
            context={"found": found, "supported": supported, "thread_id": thread_id},
        )
        self.found = found
        self.supported = supported
        self.thread_id = thread_id
