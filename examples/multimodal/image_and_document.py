"""Image and document input to OpenAI gpt-5-mini.

Shows the same image sent as an external URL and as inline base64, then a PDF sent by
URL. OpenAI fetches URLs itself, so the server downloads nothing. PDFs need the Responses
API. It is the OpenAI default; the second agent sets api_style="responses" so it stays
there even if a base_url is added.

Run:  python examples/multimodal/image_and_document.py
"""

import asyncio

from dotenv import load_dotenv
from sample_media import b64, make_png

from tenxgraph.core.state import DocumentBlock, ImageBlock, MediaRef, Message, TextBlock
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.media import ImageHandling, MultimodalConfig


load_dotenv()

IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
)
PDF_URL = "https://arxiv.org/pdf/1706.03762"

# Applies to images before they reach the provider: size cap, resize, accepted types.
multimodal = MultimodalConfig(
    image_handling=ImageHandling.BASE64,
    max_image_size_mb=5,
    max_image_dimension=1024,
)

vision_agent = ReactAgent(
    model="gpt-5-mini",
    system_prompt=[{"role": "system", "content": "Describe what you see in two sentences."}],
    multimodal_config=multimodal,
)

# Chat Completions has no PDF input. The Responses API (the default) takes a PDF as
# input_file; the explicit api_style keeps it there even behind a base_url.
document_agent = ReactAgent(
    model="gpt-5-mini",
    api_style="responses",
    system_prompt=[{"role": "system", "content": "Summarise documents in three bullet points."}],
)


def image_by_url() -> Message:
    ref = MediaRef(kind="url", url=IMAGE_URL, mime_type="image/png")
    return Message(
        role="user", content=[TextBlock(text="What is in this image?"), ImageBlock(media=ref)]
    )


def image_by_base64() -> Message:
    ref = MediaRef(kind="data", data_base64=b64(make_png()), mime_type="image/png")
    return Message(
        role="user", content=[TextBlock(text="Which colours do you see?"), ImageBlock(media=ref)]
    )


def document_by_url() -> Message:
    ref = MediaRef(kind="url", url=PDF_URL, mime_type="application/pdf")
    return Message(
        role="user", content=[TextBlock(text="Summarise this paper."), DocumentBlock(media=ref)]
    )


async def run(label: str, agent: ReactAgent, message: Message) -> None:
    app = agent.compile()
    result = await app.ainvoke({"messages": [message]}, {"thread_id": label})
    print(f"[{label}] {result['messages'][-1].text()}\n")
    await app.aclose()


async def main() -> None:
    await run("image-url", vision_agent, image_by_url())
    await run("image-base64", vision_agent, image_by_base64())
    await run("pdf-url", document_agent, document_by_url())


if __name__ == "__main__":
    asyncio.run(main())
