"""Call a provider SDK yourself inside a node and return a Message through a converter.

Use this when you need an SDK feature the Agent class does not expose. The converter turns
the raw SDK response (text, tool calls, usage) into the Message the graph expects.
Run: python examples/providers/custom_converter.py [gemini|openai]
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.adapters.llm import GoogleGenAIConverter, OpenAIConverter
from tenxgraph.utils.constants import END, START


load_dotenv()


async def gemini_node(state: AgentState, config: dict) -> Message:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    prompt = state.context[-1].text()
    response = await client.aio.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return await GoogleGenAIConverter().convert_response(response)


async def openai_node(state: AgentState, config: dict) -> Message:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": state.context[-1].text()}],
    )
    return await OpenAIConverter().convert_response(response)


NODES = {"gemini": gemini_node, "openai": openai_node}


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "gemini"

    graph = StateGraph()
    graph.add_node("MAIN", NODES[name])
    graph.add_edge(START, "MAIN")
    graph.add_edge("MAIN", END)
    app = graph.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("Write a haiku about Python.")]},
        config={"thread_id": "custom-converter"},
    )
    reply = result["messages"][-1]
    print("assistant:", reply.text())
    print("usage:", reply.usages)
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
