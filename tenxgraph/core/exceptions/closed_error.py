from typing import Any

from .graph_error import GraphError


class GraphClosedError(GraphError):
    """Raised when a run is started on a graph that is closing or closed.

    ``CompiledGraph.aclose()`` refuses new runs from its first step. Catch this to answer the
    caller with "try again elsewhere" (the API maps it to HTTP 503).
    """

    def __init__(
        self,
        message: str = "The graph is closing or closed and accepts no new runs",
        error_code: str = "GRAPH_CLOSED_001",
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message=message, error_code=error_code, context=context)
