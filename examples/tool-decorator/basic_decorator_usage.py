"""Basic examples of using the @tool decorator.

This module demonstrates various ways to use the @tool decorator for
registering tools with metadata in the agentflow framework.
"""

import asyncio

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.utils import END, START, get_tool_metadata, has_tool_decorator, tool


# Example 1: Basic tool with just a name
@tool(name="add_numbers")
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# Example 2: Tool without explicit decorator arguments (uses function name and docstring)
@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers together."""
    return x * y


# Example 3: Tool with full metadata
@tool(
    name="web_search",
    description="Search the web for information using a search engine",
    tags=["search", "web", "external"],
    provider="custom",
    capabilities=["network_access"],
    metadata={"rate_limit": 100, "timeout": 30},
)
def search_web(query: str, max_results: int = 10) -> list[str]:
    """Simulate a web search."""
    # In a real implementation, this would call a search API
    return [f"Result {i + 1} for '{query}'" for i in range(max_results)]


# Example 4: Async tool with tags
@tool(
    name="fetch_data",
    description="Asynchronously fetch data from a remote API",
    tags=["async", "api", "fetch"],
)
async def fetch_data_async(endpoint: str, timeout: int = 5) -> dict:
    """Fetch data asynchronously from an API endpoint."""
    # Simulate async API call
    await asyncio.sleep(0.1)
    return {"endpoint": endpoint, "data": "sample data", "timeout": timeout}


# Example 5: Tool with tags for filtering
@tool(name="database_read", tags=["database", "read"])
def read_from_db(table: str, record_id: int) -> dict:
    """Read a record from the database."""
    return {"table": table, "id": record_id, "data": "sample record"}


@tool(name="database_write", tags=["database", "write"])
def write_to_db(table: str, data: dict) -> bool:
    """Write a record to the database."""
    return True


# Example 6: Tool with injectable parameters
@tool(
    name="stateful_calculator",
    description="Calculator that can access agent state",
    tags=["calculator", "stateful"],
)
def stateful_add(a: int, b: int, state: AgentState | None = None) -> int:
    """Add two numbers and optionally log to state.

    The 'state' parameter is injectable and won't appear in the tool schema.
    """
    result = a + b
    if state:
        # Could log the calculation to state if needed
        pass
    return result


def demonstrate_basic_usage():
    """Demonstrate basic usage of decorated tools."""
    print("=" * 60)
    print("Basic Tool Decorator Examples")
    print("=" * 60)

    # Create a ToolNode with decorated functions
    tool_node = ToolNode([add, multiply, search_web, read_from_db, write_to_db])

    # Get all tool schemas
    tools = tool_node.get_local_tool()

    print(f"\nTotal tools registered: {len(tools)}")
    print("\nTool schemas:")
    for tool_schema in tools:
        func = tool_schema["function"]
        print(f"\n  Name: {func['name']}")
        print(f"  Description: {func['description']}")
        print(f"  Parameters: {list(func['parameters']['properties'].keys())}")


def demonstrate_tag_filtering():
    """Demonstrate filtering tools by tags."""
    print("\n" + "=" * 60)
    print("Tag Filtering Examples")
    print("=" * 60)

    # Create a ToolNode with all tools
    tool_node = ToolNode([add, multiply, search_web, read_from_db, write_to_db, stateful_add])

    # Get only database-related tools
    db_tools = tool_node.get_local_tool(tags={"database"})
    print(f"\nDatabase tools (filtered by tag 'database'): {len(db_tools)}")
    for tool_schema in db_tools:
        print(f"  - {tool_schema['function']['name']}")

    # Get only search tools
    search_tools = tool_node.get_local_tool(tags={"search"})
    print(f"\nSearch tools (filtered by tag 'search'): {len(search_tools)}")
    for tool_schema in search_tools:
        print(f"  - {tool_schema['function']['name']}")

    # Get only read-related tools
    read_tools = tool_node.get_local_tool(tags={"read"})
    print(f"\nRead tools (filtered by tag 'read'): {len(read_tools)}")
    for tool_schema in read_tools:
        print(f"  - {tool_schema['function']['name']}")


def demonstrate_metadata_inspection():
    """Demonstrate inspecting tool metadata."""
    print("\n" + "=" * 60)
    print("Metadata Inspection Examples")
    print("=" * 60)

    # Check if functions are decorated
    print(f"\nadd is decorated: {has_tool_decorator(add)}")
    print(f"multiply is decorated: {has_tool_decorator(multiply)}")

    # Plain function without decorator
    def plain_function():
        pass

    print(f"plain_function is decorated: {has_tool_decorator(plain_function)}")

    # Get metadata from decorated functions
    print("\nMetadata for 'search_web':")
    metadata = get_tool_metadata(search_web)
    for key, value in metadata.items():
        if value:  # Only print non-None/non-empty values
            print(f"  {key}: {value}")

    print("\nMetadata for 'add':")
    metadata = get_tool_metadata(add)
    for key, value in metadata.items():
        if value or key == "tags":  # Print tags even if empty
            print(f"  {key}: {value}")


async def demonstrate_async_tools():
    """Demonstrate async tool usage."""
    print("\n" + "=" * 60)
    print("Async Tool Examples")
    print("=" * 60)

    # Create state graph with async tool
    tool_node = ToolNode([fetch_data_async])

    # Get tool schema
    tools = tool_node.get_local_tool()
    print(f"\nAsync tool registered: {tools[0]['function']['name']}")
    print(f"Description: {tools[0]['function']['description']}")
    print(f"Tags: {get_tool_metadata(fetch_data_async)['tags']}")

    # The ToolNode handles async execution automatically
    print("\nAsync tools can be used seamlessly in both sync and async contexts")


def main():
    """Run all demonstrations."""
    demonstrate_basic_usage()
    demonstrate_tag_filtering()
    demonstrate_metadata_inspection()

    # Run async demonstration
    asyncio.run(demonstrate_async_tools())

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
