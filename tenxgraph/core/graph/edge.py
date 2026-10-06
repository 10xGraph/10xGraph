"""Graph edge representation and routing logic for TAF workflows.

This module defines the Edge class, which represents connections between nodes
in a TAF graph workflow. Edges can be either static (always followed) or
conditional (followed only when certain conditions are met), enabling complex
routing logic and decision-making within graph execution.

Edges are fundamental building blocks that define the flow of execution through
a graph, determining which node should execute next based on the current state
and any conditional logic.
"""

import logging
from collections.abc import Callable


logger = logging.getLogger("agentflow.graph")


class Edge:
    """Represents a connection between two nodes in a graph workflow.

    An Edge defines the relationship and routing logic between nodes, specifying
    how execution should flow from one node to another. Edges can be either
    static (unconditional) or conditional based on runtime state evaluation.

    Edges support complex routing scenarios including:
    - Simple static connections between nodes
    - Conditional routing based on state evaluation
    - Dynamic routing with multiple possible destinations
    - Decision trees and branching logic

    Attributes:
        from_node: Name of the source node where execution originates.
        to_node: Name of the destination node where execution continues.
        condition: Optional callable that determines if this edge should be
            followed. If None, the edge is always followed (static edge).
        condition_result: Optional value to match against condition result
            for mapped conditional edges.

    Example:
        ```python
        # Static edge - always followed
        static_edge = Edge("start", "process")


        # Conditional edge - followed only if condition returns True
        def needs_approval(state):
            return state.data.get("requires_approval", False)


        conditional_edge = Edge("process", "approval", condition=needs_approval)


        # Mapped conditional edge - follows based on specific condition result
        def get_priority(state):
            return state.data.get("priority", "normal")


        high_priority_edge = Edge("triage", "urgent", condition=get_priority)
        high_priority_edge.condition_result = "high"
        ```
    """

    def __init__(
        self,
        from_node: str,
        to_node: str,
        condition: Callable | None = None,
    ):
        """Initialize a new Edge with source, destination, and optional condition.

        Args:
            from_node: Name of the source node. Must match a node name in the graph.
            to_node: Name of the destination node. Must match a node name in the graph
                or be a special constant like END.
            condition: Optional callable that takes an AgentState as argument and
                returns a value to determine if this edge should be followed.
                If None, this is a static edge that's always followed.

        Note:
            The condition function should be deterministic and side-effect free
            for predictable execution behavior. It receives the current AgentState
            and should return a boolean (for simple conditions) or a string/value
            (for mapped conditional routing).
        """
        logger.debug(
            "Creating edge from '%s' to '%s' with condition=%s",
            from_node,
            to_node,
            "yes" if condition else "no",
        )
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition
        self.condition_result: str | None = None
