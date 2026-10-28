# ruff: noqa: INP001, T201
"""Load the markdown/txt files in data/ into Qdrant as the shared knowledge base.

Splits each file into sections at markdown headings, embeds them with OpenAI and
stores them under the knowledge-base owner. Safe to re-run: chunk ids are stable.

Run: python ingest.py            (needs OPENAI_API_KEY and a reachable Qdrant)
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from graph.store import KB_CONFIG, get_store


DATA_DIR = Path(__file__).parent / "data"
NAMESPACE = uuid.UUID("6f1d2c1e-0b8e-4b53-9a52-3c1a7a0e5d11")


def chunk_text(text: str, max_chars: int = 1200) -> list[tuple[str, str]]:
    """Split on markdown headings. Returns (heading, body) pairs, long bodies cut by paragraph."""
    sections = re.split(r"(?m)^(?=#{1,3} )", text)
    chunks: list[tuple[str, str]] = []
    for section in filter(str.strip, sections):
        first, _, _ = section.partition("\n")
        heading = first.lstrip("# ").strip() if first.startswith("#") else ""
        body = section.strip()
        if len(body) <= max_chars:
            chunks.append((heading, body))
            continue
        current = ""
        for para in re.split(r"\n\s*\n", body):
            if current and len(current) + len(para) > max_chars:
                chunks.append((heading, current.strip()))
                current = ""
            current += para + "\n\n"
        if current.strip():
            chunks.append((heading, current.strip()))
    return chunks


async def main() -> None:
    load_dotenv()
    store = get_store()
    await store.asetup()

    total = 0
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix not in {".md", ".txt"}:
            continue
        for i, (heading, body) in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            await store.astore(
                KB_CONFIG,
                body,
                category="docs",
                metadata={"source": path.name, "title": heading, "chunk": i},
                # A fixed id makes re-ingesting update the chunk instead of duplicating it.
                memory_id=str(uuid.uuid5(NAMESPACE, f"{path.name}:{i}")),
            )
            total += 1
        print(f"ingested {path.name}")
    print(f"done: {total} chunks")
    await store.arelease()


if __name__ == "__main__":
    asyncio.run(main())
