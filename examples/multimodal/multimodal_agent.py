"""
Multimodal Agent Example
========================

Shows how to send images, audio, video, and documents to an agent.

Three ways to provide media:
1. External URL — the agent fetches and adapts it per provider
2. Inline base64 — embedded directly in the message
3. Uploaded file_id — upload first, then reference (recommended for production)

Run:
    source /Users/shudipto/Projects/agentflow/.venv/bin/activate
    cd /Users/shudipto/Projects/agentflow/agentflow
    python examples/multimodal/multimodal_agent.py
"""

import base64

from dotenv import load_dotenv

from tenxgraph import (
    END,
    Agent,
    AudioBlock,
    DocumentBlock,
    DocumentHandling,
    ImageBlock,
    ImageHandling,
    InMemoryCheckpointer,
    InMemoryMediaStore,
    MediaRef,
    Message,
    MultimodalConfig,
    StateGraph,
    TextBlock,
    VideoBlock,
)


load_dotenv()

# ---------------------------------------------------------------------------
# 1. Set up storage
# ---------------------------------------------------------------------------

checkpointer = InMemoryCheckpointer()

# MediaStore is needed for file_id uploads and internal media refs.
# For URL/base64-only workflows you can skip this.
media_store = InMemoryMediaStore()

# ---------------------------------------------------------------------------
# 2. Create the agent
# ---------------------------------------------------------------------------

agent = Agent(
    model="gemini-2.5-flash",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You are a helpful multimodal assistant. "
                "Describe what you see in any images, "
                "transcribe any audio, and summarize any documents."
            ),
        },
    ],
    multimodal_config=MultimodalConfig(
        image_handling=ImageHandling.BASE64,
        document_handling=DocumentHandling.EXTRACT_TEXT,
    ),
)

# ---------------------------------------------------------------------------
# 3. Build the graph
# ---------------------------------------------------------------------------

graph = StateGraph()
graph.add_node("agent", agent)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 4. Build multimodal messages
# ---------------------------------------------------------------------------


def build_message_with_external_url() -> list:
    """Example 1: Image from an external URL."""
    return [
        Message(
            role="user",
            content=[
                TextBlock(text="What is in this image?"),
                ImageBlock(
                    media=MediaRef(
                        kind="url",
                        url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
                        mime_type="image/png",
                    )
                ),
            ],
        ),
    ]


def build_message_with_base64() -> list:
    """Example 2: Image from inline base64 data."""
    # A valid 10x10 red pixel PNG generated with PIL
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAE0lEQVR4nGP8z4APMOGVZRip0gBBLAETee26JgAAAABJRU5ErkJggg=="

    return [
        Message(
            role="user",
            content=[
                TextBlock(text="Here is a tiny red pixel image encoded as base64."),
                ImageBlock(
                    media=MediaRef(
                        kind="data",
                        data_base64=png_b64,
                        mime_type="image/png",
                    )
                ),
            ],
        ),
    ]


def build_message_with_file_id() -> list:
    """Example 3: Upload a file first, then reference by file_id.

    This is the recommended production pattern — upload once, reference many times.
    """
    # Simulate uploading a file to the media store
    sample_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # fake PNG bytes

    # store() returns the generated key — use that as our file_id
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    file_id = loop.run_until_complete(media_store.store(data=sample_image, mime_type="image/png"))

    return [
        Message(
            role="user",
            content=[
                TextBlock(text="Analyze this uploaded image."),
                ImageBlock(
                    media=MediaRef(
                        kind="file_id",
                        file_id=file_id,
                        mime_type="image/png",
                    )
                ),
            ],
        ),
    ]


def build_message_with_audio() -> list:
    """Example 4: Audio input."""
    # Fake WAV bytes
    audio_bytes = b"RIFF" + b"\x00" * 50
    b64 = base64.b64encode(audio_bytes).decode()

    return [
        Message(
            role="user",
            content=[
                TextBlock(text="Transcribe this audio."),
                AudioBlock(
                    media=MediaRef(
                        kind="data",
                        data_base64=b64,
                        mime_type="audio/wav",
                    )
                ),
            ],
        ),
    ]


def build_message_with_document() -> list:
    """Example 5: Document input with extracted text."""
    return [
        Message(
            role="user",
            content=[
                TextBlock(text="Summarize this document."),
                DocumentBlock(
                    text="AgentFlow is a multi-agent framework inspired by LangGraph. "
                    "It provides checkpointer, storage, and media handling out of the box.",
                    media=MediaRef(
                        kind="file_id",
                        file_id="doc-001",
                        mime_type="text/plain",
                    ),
                ),
            ],
        ),
    ]


def build_message_with_video() -> list:
    """Example 6: Video input."""
    # Fake MP4 bytes
    video_bytes = b"\x00\x00\x00\x1cftypmp42" + b"\x00" * 50
    b64 = base64.b64encode(video_bytes).decode()

    return [
        Message(
            role="user",
            content=[
                TextBlock(text="Describe this video."),
                VideoBlock(
                    media=MediaRef(
                        kind="data",
                        data_base64=b64,
                        mime_type="video/mp4",
                    )
                ),
            ],
        ),
    ]


def build_mixed_message() -> list:
    """Example 7: Multiple media types in one message."""
    return [
        Message(
            role="user",
            content=[
                TextBlock(text="Here are multiple inputs. Please process all of them."),
                ImageBlock(
                    media=MediaRef(
                        kind="url",
                        url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
                        mime_type="image/png",
                    )
                ),
                DocumentBlock(
                    text="This is a short document about agent frameworks.",
                    media=MediaRef(
                        kind="file_id",
                        file_id="doc-002",
                        mime_type="text/plain",
                    ),
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# 5. Run examples
# ---------------------------------------------------------------------------

EXAMPLES = {
    "url": ("External URL image", build_message_with_external_url),
    "base64": ("Inline base64 image", build_message_with_base64),
    "file_id": ("Uploaded file_id image", build_message_with_file_id),
    "audio": ("Audio input", build_message_with_audio),
    "document": ("Document input", build_message_with_document),
    "video": ("Video input", build_message_with_video),
    "mixed": ("Mixed media types", build_mixed_message),
}


def run_example(name: str, thread_id: str = "multimodal-demo") -> None:
    """Run a single multimodal example."""
    label, builder = EXAMPLES[name]
    print(f"\n{'=' * 60}")
    print(f"  Example: {label}")
    print(f"{'=' * 60}")

    messages = builder()
    config = {"thread_id": thread_id, "recursion_limit": 10}

    result = app.invoke({"messages": messages}, config=config)

    for msg in result.get("messages", []):
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")

        print(f"\n  [{role}]")
        if isinstance(content, str):
            print(f"    {content[:300]}")
        elif isinstance(content, list):
            for block in content:
                block_type = getattr(block, "type", type(block).__name__)
                if block_type == "text":
                    text = getattr(block, "text", "")
                    print(f"    [text] {text[:200]}")
                elif block_type == "image":
                    media = getattr(block, "media", None)
                    if media:
                        print(f"    [image] kind={media.kind}")
                else:
                    print(f"    [{block_type}]")
        print()


if __name__ == "__main__":
    import sys

    # Run specific examples or all if no args
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        # Default: run just the first 3 quick examples
        names = ["url", "base64", "file_id"]

    for name in names:
        if name not in EXAMPLES:
            print(f"Unknown example: {name}. Available: {list(EXAMPLES.keys())}")
            continue
        try:
            run_example(name, thread_id=f"demo-{name}")
        except Exception as e:
            print(f"  ⚠ Example '{name}' failed: {e}\n")

    print("\n" + "=" * 60)
    print("  All multimodal examples completed!")
    print("=" * 60)
