#!/usr/bin/env python3
"""
Simple test script to verify logging functionality in TAF.
"""

import logging
import sys


sys.path.insert(0, "/home/shudipto/projects/agentflow")

# Import logging setup first
from tenxgraph.utils.logging import configure_logging


# Configure logging to DEBUG level to see all logs
configure_logging(level=logging.DEBUG)

# Test imports and basic functionality
from tenxgraph.core.graph import StateGraph
from tenxgraph.utils import Message


def simple_node(state, config):
    """A simple test node."""
    return {"message": "Hello from simple node"}


def main():
    """Test basic graph creation and logging."""
    logger = logging.getLogger(__name__)
    logger.info("Testing TAF logging...")

    # Create a simple graph
    graph = StateGraph()
    graph.add_node("simple", simple_node)
    graph.add_edge("__start__", "simple")
    graph.add_edge("simple", "__end__")

    # Compile the graph
    compiled = graph.compile(debug=True)

    # Test basic functionality
    messages = [Message.from_text("Hello", role="user")]
    result = compiled.invoke({"messages": messages})

    logger.info("Test completed successfully!")
    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
