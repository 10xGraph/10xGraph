"""Run the same ReactAgent on every supported model provider.

CONFIGS maps a name to the ReactAgent arguments that select a model and backend. Nothing is
created until you pick one, so you only need the SDK and key for the provider you run.

Run: python examples/providers/providers.py [name]   (default: openai; see CONFIGS)
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny, 22C."


# The provider is detected from the model string, so no provider= argument is needed.
# Anything else is an ordinary keyword argument for ReactAgent / Agent.
CONFIGS: dict[str, dict] = {
    # --- OpenAI: pip install "10xgraph[openai]", OPENAI_API_KEY ---
    "openai": {"model": "gpt-5-mini"},  # Chat Completions
    "openai-responses": {"model": "gpt-5-mini", "api_style": "responses"},  # Responses API
    # --- Google: pip install "10xgraph[google-genai]", GEMINI_API_KEY ---
    "gemini": {"model": "gemini-2.5-flash"},
    # Same model on Vertex AI. Needs GOOGLE_CLOUD_PROJECT (+ optional GOOGLE_CLOUD_LOCATION)
    # and application default credentials instead of an API key.
    "gemini-vertex": {"model": "gemini-2.5-flash", "use_vertex_ai": True},
    # --- Anthropic: pip install "10xgraph[anthropic]", ANTHROPIC_API_KEY ---
    "claude": {"model": "claude-sonnet-5-5"},
    # Claude through Google Vertex AI: "10xgraph[anthropic-vertex]". The SDK reads the
    # project and region from its own env vars (ANTHROPIC_VERTEX_PROJECT_ID, CLOUD_ML_REGION).
    "claude-vertex": {"model": "claude-sonnet-5-5", "anthropic_backend": "vertex"},
    # Claude through Amazon Bedrock: "10xgraph[anthropic-bedrock]", AWS credentials/region
    # from the environment. The Bedrock model id keeps its "anthropic." prefix.
    "claude-bedrock": {"model": "anthropic.claude-sonnet-5-5", "anthropic_backend": "bedrock"},
    # --- OpenAI-compatible servers: unknown model names use the OpenAI client + base_url ---
    # reasoning_config=None because these models do not take OpenAI reasoning options.
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "reasoning_config": None,
    },
    # Local model served by Ollama (`ollama pull llama3.1`); the key is ignored but required.
    "ollama": {
        "model": "llama3.1",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "reasoning_config": None,
    },
}


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "openai"
    if name not in CONFIGS:
        raise SystemExit(f"Unknown provider '{name}'. Choose from: {', '.join(CONFIGS)}")

    agent = ReactAgent(
        **CONFIGS[name],
        system_prompt=[{"role": "system", "content": "You are a helpful assistant."}],
        tools=[get_weather],
    )
    app = agent.compile()

    print(f"provider config: {name} -> model {CONFIGS[name]['model']}")
    result = await app.ainvoke(
        {"messages": [Message.text_message("What is the weather in Paris?")]},
        config={"thread_id": f"providers-{name}"},
    )
    print("assistant:", result["messages"][-1].text())
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
