#!/usr/bin/env python3
"""
Simple test to verify logging is configured correctly.
"""

import logging
import sys


sys.path.insert(0, "/home/shudipto/projects/agentflow")

# Import and configure logging directly
from tenxgraph.utils.logging import configure_logging


# Test logging configuration
configure_logging(level=logging.DEBUG)


def test_logging():
    """Test that logging is working correctly."""
    logger = logging.getLogger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test logging from different modules
    graph_logger = logging.getLogger("tenxgraph.graph.compiled_graph")
    graph_logger.info("Testing graph module logging")

    state_logger = logging.getLogger("tenxgraph.state.agent_state")
    state_logger.info("Testing state module logging")

    utils_logger = logging.getLogger("tenxgraph.utils.dependency_injection")
    utils_logger.info("Testing utils module logging")

    print("Logging test completed - check output above for log messages")


if __name__ == "__main__":
    test_logging()
