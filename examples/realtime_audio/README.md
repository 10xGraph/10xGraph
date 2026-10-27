---
title: Realtime Voice Agent with Gemini Live in 10xGraph (Python)
description: Build a realtime audio-to-audio voice agent with 10xGraph AudioAgent and Gemini Live. Stream a WAV file or microphone, call tools by voice, or serve it over a WebSocket.
section: Examples
group: "Multimodal and realtime"
order: 210
label: Realtime audio
updated: "2026-10-28"
faq:
  - question: Which providers support realtime audio in 10xGraph?
    answer: Only Gemini Live today. The realtime layer is provider-neutral so more can be added, but gemini is the only client implemented.
  - question: What audio format does it use?
    answer: Input is PCM16, mono, 16 kHz. Output is PCM16, mono, 24 kHz. Transcripts are saved as messages, and raw audio is never stored.
  - question: Why does arealtime exist instead of using invoke?
    answer: The provider owns the turn loop in a live session, so a realtime graph is driven by CompiledGraph.arealtime(input_queue, config) rather than the turn-based invoke and stream methods.
---

## What you will build

A voice assistant that listens and talks back in real time, and calls a `get_weather` tool when asked about the weather. There are three ways to run it: a headless WAV file in and out, a live microphone with speaker playback, and over a WebSocket through the 10xGraph API server.

Realtime currently supports Gemini Live only, so these examples have no model choice to make.

## Key concepts

- `AudioAgent`: the prebuilt realtime agent. Tools passed to it are advertised to the model.
- `RealtimeConfig`: model, voice, system instruction, transcription and VAD settings.
- `LiveInputQueue`: non-blocking queue you push PCM16 audio into with `send_audio`.
- `CompiledGraph.arealtime(queue, config)`: async iterator of events (`audio_delta`, `input_transcript`, `output_transcript`, `tool_call`, `turn_complete`, `interrupted`, `error`).
- `/v1/graph/live`: the WebSocket endpoint the API server exposes for the same agent.

## Prerequisites

```bash
pip install "10xgraph[realtime]"          # pulls in google-genai
pip install sounddevice                   # microphone example only
pip install 10xgraph-api                  # API server example only
export GEMINI_API_KEY=...
```

To use Vertex AI instead, set `GOOGLE_GENAI_USE_VERTEXAI=true` (the microphone script reads it). The model string is `gemini-2.5-flash-live`, the name used in the library's tests. Google changes Live model names often, so check the Live API docs for what your key can use and edit `MODEL` in the file.

## Run it

```bash
# 1. Headless: 16 kHz mono PCM16 WAV in, out.wav out
python examples/realtime_audio/audio_agent_file.py path/to/input.wav

# 2. Microphone and speakers
python examples/realtime_audio/audio_agent_mic.py

# 3. API server WebSocket
cd examples/realtime_audio && 10xgraph dev --no-open
```

## How it works

The agent:

```python
app = AudioAgent(
    MODEL,
    realtime_config=RealtimeConfig(model=MODEL, voice="Puck", system_instruction="..."),
    tools=[get_weather],
).compile()
```

Feed audio and read events:

```python
queue = LiveInputQueue()
queue.send_audio(pcm16_bytes)
async for event in app.arealtime(queue, {"thread_id": "t1"}):
    if event.type == "audio_delta":
        out.writeframes(event.data)
    elif event.type == "turn_complete":
        queue.close()
```

`audio_agent_file.py` feeds the file in 100 ms chunks and closes the queue after the first turn. `audio_agent_mic.py` mutes the microphone while the agent speaks, so speaker output is not heard as user input. With headphones, set `MIC_FULL_DUPLEX=1` to allow barge-in: an `interrupted` event flushes queued playback.

For the server, `10xgraph.json` points at `graph.py`:

```json
{ "agent": "graph:app", "checkpointer": "graph:checkpointer", "env": ".env" }
```

The WebSocket protocol:

- First frame: a JSON control frame such as `{"model": "...", "thread_id": "abc", "voice": "Puck"}`. Fields present override the build-time config for that session.
- Upstream: a binary frame is PCM16 input audio. A JSON frame is `{"type": "text" | "activity_start" | "activity_end" | "close", ...}`.
- Downstream: a binary frame is PCM16 model audio. A JSON text frame is any other event.

## Expected output

Illustrative output from the microphone script.

```text
Listening [echo-safe (mic muted while agent talks)]. Try: 'What's the weather in Tokyo?'
  [tool-call requested] get_weather({'location': 'Tokyo'})
  [tool] get_weather(location='Tokyo') -> 18 degrees Celsius, light rain
you:   What's the weather in Tokyo?
agent: It's 18 degrees and raining lightly in Tokyo.
```

## Using a different model

Edit `MODEL` and the `model` in `RealtimeConfig` to another Gemini Live model name. Other providers are not available for realtime yet. For a turn-based voice flow with another provider, transcribe, run a `ReactAgent`, and synthesise speech separately.

## Next steps

- [Multimodal input](../multimodal/)
- [Streaming](../streaming/)
- [Prebuilt ReAct agent](../quickstart/)
