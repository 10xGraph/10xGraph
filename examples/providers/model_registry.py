"""Chat with one ReactAgent that has several models in its registry, switching per message.

model_registry maps a name to the model settings. The config key "model" picks the entry for
each run, so one thread can move between providers.
Keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (only for the entries you use)
Run: python examples/providers/model_registry.py
Type a message to chat. "/model gpt" switches model, "/quit" exits.
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()

# Any key Agent takes for a model works in an entry, plus provider settings.
MODELS = {
    "claude": {"model": "claude-haiku-5-5"},
    "gpt": {"model": "gpt-5-mini"},
    "gemini": {"model": "gemini-2.5-flash"},
    # A self-hosted OpenAI-compatible server (vLLM, Ollama) is just another entry:
    # "local": {
    #     "model": "llama3.1",
    #     "provider": "openai",
    #     "base_url": "http://localhost:11434/v1",
    #     "api_style": "chat",
    # },
}


def get_time(city: str) -> str:
    """Return the current time in a city."""
    return f"It is 14:05 in {city}."


app = ReactAgent(
    model="claude",
    model_registry=MODELS,
    system_prompt=[{"role": "system", "content": "You are a helpful assistant. Be brief."}],
    tools=[get_time],
).compile(checkpointer=InMemoryCheckpointer())


async def main() -> None:
    current = "claude"
    print(f"models: {', '.join(MODELS)}  (now: {current})")
    while True:
        text = input(f"[{current}] you> ").strip()
        if text in ("/quit", "/exit"):
            break
        if text.startswith("/model"):
            current = text.split(maxsplit=1)[-1]
            print(f"switched to {current}")
            continue
        if not text:
            continue
        try:
            result = await app.ainvoke(
                {"messages": [Message.text_message(text)]},
                config={"thread_id": "registry", "model": current},
            )
        except Exception as exc:  # show the error and keep chatting
            print(f"error: {type(exc).__name__}: {exc}")
            continue
        reply = result["messages"][-1]
        print(f"[{current}] ai> {reply.text()}")
        if reply.usages:
            u = reply.usages
            print(f"    tokens: in={u.prompt_tokens} out={u.completion_tokens}")
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
