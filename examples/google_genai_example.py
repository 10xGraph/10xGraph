"""
Example: Using Google Generative AI (google-genai) with Agentflow

This example demonstrates how to use the Google GenAI adapter with Agentflow.
It shows both standard and streaming response handling.

Prerequisites:
- Install: pip install google-genai
- Set environment variable: GEMINI_API_KEY or GOOGLE_API_KEY

Usage:
    python examples/google_genai_example.py
"""

import asyncio
import os

from tenxgraph.runtime.adapters.llm import GoogleGenAIConverter
from tenxgraph.core.state import AgentState


async def example_standard_response():
    """Example of converting a standard Google GenAI response."""
    print("\n=== Example 1: Standard Response ===\n")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai package is not installed.")
        print("Install it with: pip install google-genai")
        return

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return

    # Create Google GenAI client
    client = genai.Client(api_key=api_key)

    try:
        # Generate content
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Write a haiku about Python programming",
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=100,
            ),
        )

        # Convert the response using GoogleGenAIConverter
        converter = GoogleGenAIConverter()
        message = await converter.convert_response(response)

        print(f"Message ID: {message.message_id}")
        print(f"Role: {message.role}")
        print(f"Content blocks: {len(message.content)}")
        print(f"\nText content:")
        for block in message.content:
            if hasattr(block, "text"):
                print(f"  {block.text}")

        print(f"\nMetadata: {message.metadata}")
        print(f"Token usage: {message.usages}")

    finally:
        client.close()


async def example_streaming_response():
    """Example of converting a streaming Google GenAI response."""
    print("\n=== Example 2: Streaming Response ===\n")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai package is not installed.")
        print("Install it with: pip install google-genai")
        return

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return

    # Create Google GenAI client
    client = genai.Client(api_key=api_key)

    try:
        # Generate streaming content
        stream = client.models.generate_content_stream(
            model="gemini-2.0-flash-exp",
            contents="Count from 1 to 5, one number at a time",
            config=types.GenerateContentConfig(
                temperature=0.7,
            ),
        )

        # Convert the streaming response
        converter = GoogleGenAIConverter()
        config = {"thread_id": "example-thread"}

        print("Streaming chunks:")
        async for message in converter.convert_streaming_response(
            config=config,
            node_name="google_genai_node",
            response=stream,
        ):
            if message.delta:
                # This is a streaming chunk
                for block in message.content:
                    if hasattr(block, "text"):
                        print(block.text, end="", flush=True)
            else:
                # This is the final message
                print(f"\n\nFinal message ID: {message.message_id}")
                print(f"Total content blocks: {len(message.content)}")

    finally:
        client.close()


async def example_function_calling():
    """Example of function calling with Google GenAI."""
    print("\n=== Example 3: Function Calling ===\n")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai package is not installed.")
        print("Install it with: pip install google-genai")
        return

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return

    # Create Google GenAI client
    client = genai.Client(api_key=api_key)

    try:
        # Define a simple function
        def get_weather(location: str) -> str:
            """Get the weather for a location.

            Args:
                location: The city and state, e.g. San Francisco, CA
            """
            return "sunny"

        # Generate content with function calling
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="What's the weather like in Boston?",
            config=types.GenerateContentConfig(
                tools=[get_weather],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True  # Disable auto-calling to see the function call
                ),
            ),
        )

        # Convert the response
        converter = GoogleGenAIConverter()
        message = await converter.convert_response(response)

        print(f"Message has {len(message.tools_calls or [])} tool calls")
        if message.tools_calls:
            for tool_call in message.tools_calls:
                print(f"\nTool call:")
                print(f"  Function: {tool_call.get('function', {}).get('name')}")
                print(f"  Arguments: {tool_call.get('function', {}).get('arguments')}")

        # Show the tool call blocks
        for block in message.content:
            if hasattr(block, "name"):  # ToolCallBlock
                print(f"\nToolCallBlock:")
                print(f"  Name: {block.name}")
                print(f"  Args: {block.args}")

    finally:
        client.close()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Google Generative AI Adapter Examples")
    print("=" * 60)

    await example_standard_response()
    await example_streaming_response()
    await example_function_calling()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
