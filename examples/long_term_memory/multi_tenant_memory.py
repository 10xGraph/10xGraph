"""Multi-tenant memory and threads on one Postgres, with no LLM.

Both PgCheckpointer and PgVectorStore get the same developer-owned columns:

- tenant column ``org_id``: isolates data. Two orgs can reuse the same user_id and still
  never see each other's rows.
- extra columns ``public_id`` and ``plan``: plain data stored with each row. ``plan`` is
  filterable in ``alist_threads``.

Where the values come from (trusted-source rule): tenant and extra column values come only
from ``config["user"]``. ``user`` is a reserved config key that clients cannot set. In an
API deployment your auth class fills it (BaseAuth.authenticate() result or JWT claims). In
core-only use, as here, YOU build ``config["user"]`` from your own trusted auth (a verified
session or token). Never read these values from request config, metadata, state or model
output.

The store is text-only (``embedding=None``), so no embedding key is needed.

Run:  export POSTGRES_DSN=postgresql://tenx:tenx@localhost:5432/tenx
      export REDIS_URL=redis://localhost:6379/0
      python examples/long_term_memory/multi_tenant_memory.py
"""

import asyncio
import os

from tenxgraph.storage.checkpointer import PgCheckpointer
from tenxgraph.storage.store import PgVectorStore
from tenxgraph.storage.store.store_schema import RetrievalStrategy
from tenxgraph.utils import ThreadInfo


POSTGRES_DSN = os.environ["POSTGRES_DSN"]
REDIS_URL = os.environ["REDIS_URL"]

TENANT_COLUMNS = ["org_id"]
EXTRA_COLUMNS = {"public_id": "text", "plan": "text"}


def run_config(user_id: str, thread_id: str, org_id: str, public_id: str, plan: str) -> dict:
    """Build a run config. The `user` dict must come from your trusted auth, not a client."""
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "collection": "tenant_demo",
        "user": {"org_id": org_id, "public_id": public_id, "plan": plan},
    }


async def main() -> None:
    checkpointer = PgCheckpointer(
        postgres_dsn=POSTGRES_DSN,
        redis_url=REDIS_URL,
        tenant_columns=TENANT_COLUMNS,
        extra_columns=EXTRA_COLUMNS,
    )
    store = PgVectorStore(
        embedding=None,  # text-only store: TEXT search, no pgvector extension needed
        dsn=POSTGRES_DSN,
        retrieval_strategy=RetrievalStrategy.TEXT,
        tenant_columns=TENANT_COLUMNS,
        extra_columns=EXTRA_COLUMNS,
    )
    await checkpointer.asetup()
    await store.asetup()

    # Two orgs, the same user_id "sam".
    acme = run_config("sam", "t-acme", "acme", "pub-acme-1", "pro")
    globex = run_config("sam", "t-globex", "globex", "pub-globex-1", "free")

    for cfg, org in ((acme, "acme"), (globex, "globex")):
        await checkpointer.aput_thread(cfg, ThreadInfo(thread_id=cfg["thread_id"], user_id="sam"))
        await store.astore(cfg, f"Sam works at {org} and prefers email over phone.")

    # Same user_id, different org: each sees only its own memory.
    for cfg in (acme, globex):
        hits = await store.asearch(cfg, "email")
        print(cfg["user"]["org_id"], "->", [h.content for h in hits])

    # Filter threads by an extra column. Filters can only narrow the isolated listing.
    pro_threads = await checkpointer.alist_threads(acme, filters={"plan": {"in": ["pro"]}})
    print("acme pro threads:", [t.thread_id for t in pro_threads])
    none_for_globex = await checkpointer.alist_threads(globex, filters={"plan": {"in": ["pro"]}})
    print("globex pro threads:", [t.thread_id for t in none_for_globex])

    # Clean up the demo rows.
    for cfg in (acme, globex):
        await checkpointer.aclean_thread(cfg)
    await store.arelease()
    await checkpointer.arelease()


if __name__ == "__main__":
    asyncio.run(main())
