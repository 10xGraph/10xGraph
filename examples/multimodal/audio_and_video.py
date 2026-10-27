"""Audio and video input to Gemini gemini-2.5-flash.

Audio goes in as inline base64 (a generated tone, or your own WAV/MP3). Video is read from
a local file you pass. Gemini does not fetch URLs itself here: the server downloads them
first, so a hosted video needs a MediaFetchPolicy that allows video types.

Run:  python examples/multimodal/audio_and_video.py [speech.wav] [clip.mp4]
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from sample_media import b64, make_wav

from tenxgraph.core.state import AudioBlock, MediaRef, Message, TextBlock, VideoBlock
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.media import MediaFetchPolicy


load_dotenv()

agent = ReactAgent(
    model="gemini-2.5-flash",
    system_prompt=[{"role": "system", "content": "Transcribe audio and describe video concisely."}],
)

# The default policy only allows images and PDFs. Be explicit about what the server may
# download, and from where.
policy = MediaFetchPolicy(
    allowed_hosts=("storage.googleapis.com",),
    allowed_content_types=("video/*",),
    max_bytes=50 * 1024 * 1024,
    timeout=30.0,
)


def audio_message(path: str | None) -> Message:
    data = Path(path).read_bytes() if path else make_wav()
    mime = "audio/mp3" if path and path.endswith(".mp3") else "audio/wav"
    ref = MediaRef(kind="data", data_base64=b64(data), mime_type=mime)
    return Message(
        role="user", content=[TextBlock(text="Transcribe this audio."), AudioBlock(media=ref)]
    )


def video_message(path: str) -> Message:
    # Inline base64 is fine for short clips. For a hosted file use
    # MediaRef(kind="url", url="https://storage.googleapis.com/...") and the policy above.
    ref = MediaRef(kind="data", data_base64=b64(Path(path).read_bytes()), mime_type="video/mp4")
    return Message(
        role="user", content=[TextBlock(text="Describe this video."), VideoBlock(media=ref)]
    )


async def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    video_path = sys.argv[2] if len(sys.argv) > 2 else None

    messages = {"audio": audio_message(audio_path)}
    if video_path:
        messages["video"] = video_message(video_path)
    else:
        print("No video path given, skipping the video example.\n")

    app = agent.compile(media_policy=policy)
    for label, message in messages.items():
        result = await app.ainvoke({"messages": [message]}, {"thread_id": label})
        print(f"[{label}] {result['messages'][-1].text()}\n")
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
