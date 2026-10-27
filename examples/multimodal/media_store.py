"""Reference media by key with a media store (Gemini gemini-2.5-flash).

Upload bytes once, keep only a small `graph://media/<key>` reference in messages and
checkpoints, and let the library turn it back into bytes at call time. Swap
InMemoryMediaStore for LocalFileMediaStore to keep files on disk between runs.

Run:  python examples/multimodal/media_store.py
"""

import asyncio
import tempfile

from dotenv import load_dotenv
from sample_media import make_png

from tenxgraph.core.state import ImageBlock, MediaRef, Message, TextBlock
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.media import (
    InMemoryMediaStore,
    LocalFileMediaStore,
    MediaFetchPolicy,
    MultimodalConfig,
)
from tenxgraph.utils.media_scheme import MEDIA_SCHEME


load_dotenv()

agent = ReactAgent(
    model="gemini-2.5-flash",
    system_prompt=[{"role": "system", "content": "Describe images briefly."}],
    multimodal_config=MultimodalConfig(max_image_size_mb=5),
)


async def main() -> None:
    # In-memory: lost on exit, good for tests. Local files: a directory you control.
    stores = {
        "memory": InMemoryMediaStore(),
        "local": LocalFileMediaStore(base_dir=tempfile.mkdtemp(prefix="10xgraph-media-")),
    }

    for name, media_store in stores.items():
        # store() returns the key. The reference below is what lives in message state.
        key = await media_store.store(data=make_png(), mime_type="image/png")
        ref = MediaRef(kind="url", url=f"{MEDIA_SCHEME}{key}", mime_type="image/png")
        message = Message(
            role="user", content=[TextBlock(text="What is in this picture?"), ImageBlock(media=ref)]
        )

        app = agent.compile(
            checkpointer=InMemoryCheckpointer(),
            media_store=media_store,
            # Governs external URLs only. Store references never go through it.
            media_policy=MediaFetchPolicy(allowed_hosts=("upload.wikimedia.org",)),
        )
        result = await app.ainvoke({"messages": [message]}, {"thread_id": f"store-{name}"})
        print(f"[{name}] {result['messages'][-1].text()}\n")
        await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
