"""Realtime AudioAgent exposed through the 10xGraph API server.

`10xgraph.json` points the server at `app` below. The server then offers a WebSocket at
/v1/graph/live that bridges client audio to this agent: binary PCM16 frames upstream,
model audio back as binary frames, transcripts and tool calls as JSON.

Run:  cd examples/realtime_audio && 10xgraph dev --no-open
"""

from dotenv import load_dotenv

from tenxgraph.core.realtime.base import RealtimeConfig
from tenxgraph.prebuilt.agent import AudioAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()

# Gemini Live is the only realtime provider today. Model names change often, so check
# Google's Live API docs for what your key can use.
MODEL = "gemini-2.5-flash-live"


def get_weather(location: str) -> str:
    """Return the current weather for a city. Called by the model during the conversation."""
    return f"It is 22 degrees Celsius and sunny in {location}."


checkpointer = InMemoryCheckpointer()

app = AudioAgent(
    MODEL,
    realtime_config=RealtimeConfig(
        model=MODEL,
        voice="Puck",
        system_instruction="You are a concise, friendly voice assistant.",
    ),
    tools=[get_weather],
).compile(checkpointer=checkpointer)
