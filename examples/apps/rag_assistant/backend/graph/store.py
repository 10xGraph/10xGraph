"""Shared Qdrant knowledge-base handle used by the agent tools and by ingest.py.

The store is built lazily so importing this module (for example in unit tests)
needs no API key and opens no connection.
"""

from __future__ import annotations

import os
from functools import lru_cache

from tenxgraph.storage import system_user_id
from tenxgraph.storage.store import QdrantStore
from tenxgraph.storage.store.embedding import OpenAIEmbedding


COLLECTION = "docs_kb"

# Shared documents are not owned by a person. They belong to a reserved system owner,
# so a user-scoped memory search can never return them (and vice versa).
KB_OWNER = system_user_id("docs_kb")

# Search and ingest must agree on both values, so both read them from here.
KB_CONFIG = {"user_id": KB_OWNER, "collection": COLLECTION}


@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    """Qdrant over the network, with OpenAI embeddings (a third company from the chat model)."""
    return QdrantStore(
        embedding=OpenAIEmbedding(model="text-embedding-3-small"),
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        collection=COLLECTION,
    )
