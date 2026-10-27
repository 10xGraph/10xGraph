"""The same quickstart agent, called with the blocking invoke().

invoke() wraps ainvoke() with asyncio.run(), so use it in scripts and notebooks
without a running event loop. Inside async code (FastAPI, other agents) use ainvoke().

Run: python examples/quickstart/quickstart_sync.py   (needs OPENAI_API_KEY)
"""

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()


def get_weather(location: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {location} is sunny, 24 C."


app = ReactAgent(
    model="gpt-5-mini",
    system_prompt=[{"role": "system", "content": "You are a concise weather assistant."}],
    tools=[get_weather],
).compile()


if __name__ == "__main__":
    config = {"thread_id": "quickstart-sync"}

    first = app.invoke({"messages": [Message.text_message("Weather in Paris?")]}, config)
    print("assistant:", first["messages"][-1].text())

    # Same thread_id: the checkpointer restores the earlier turn, so "there" resolves.
    second = app.invoke(
        {"messages": [Message.text_message("And is that good for a picnic?")]}, config
    )
    print("assistant:", second["messages"][-1].text())
