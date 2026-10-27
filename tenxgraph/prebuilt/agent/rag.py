"""RAGAgent — Retrieve → (Rerank) → Synthesize.

The agent owns the full retrieval pipeline so users only need to bring:

* a **store** (``QdrantStore`` or any :class:`~tenxgraph.storage.store.BaseStore`)
  that backs the knowledge base;
* a **pre-built agent** (``Agent`` or any :class:`~tenxgraph.core.graph.base_agent.BaseAgent`)
  that generates the final answer;
* an optional **reranker** to improve precision before synthesis.

Graph topology::

    START → RETRIEVE → [RERANK] → SYNTHESIZE → END

``RERANK`` is skipped entirely when no reranker is provided.

Retrieved documents are kept in a bounded per-run cache on the ``RAGAgent``
(keyed by ``run_id``), never in saved state.  ``SYNTHESIZE`` fences them in a
``<retrieved_documents_xxxxxxxx>`` block (random suffix, escaped body, a "not
instructions" line) and passes it to ``agent.execute(..., request_blocks=[...])``,
which adds it to the latest user message of the outgoing request only, before the
user's text.  ``state.context`` is never modified, so the saved thread keeps the
user's original message (attachments, id, metadata) and does not grow.  If the
cache has no entry (for example after a process restart between nodes),
``SYNTHESIZE`` retrieves and reranks again instead of answering without documents.
The agent's own ``system_prompt`` is untouched.  The ``agent`` must accept the
``request_blocks`` keyword (``Agent`` and ``BaseAgent`` subclasses do).

Example — no reranker::

    from tenxgraph.core.graph import Agent
    from tenxgraph.prebuilt.agent import RAGAgent
    from tenxgraph.storage import create_local_qdrant_store
    from tenxgraph.storage.store.embedding import OpenAIEmbedding

    store = create_local_qdrant_store(
        path="./knowledge_base",
        embedding=OpenAIEmbedding(model="text-embedding-3-small"),
    )
    rag = RAGAgent(
        store=store,
        agent=Agent(model="gpt-4o-mini"),
        top_k=5,
    )
    app = rag.compile()

Example — with Cohere Rerank::

    from tenxgraph.prebuilt.agent.rag import CohereReranker

    rag = RAGAgent(
        store=store,
        agent=Agent(model="gpt-4o"),
        reranker=CohereReranker(api_key="...", model="rerank-v4.0-pro"),
        top_k=20,  # retrieve 20 candidates …
        top_n=5,  # … then keep the best 5 for the LLM
    )
    app = rag.compile()

Example — fully local with CrossEncoder::

    from tenxgraph.prebuilt.agent.rag import CrossEncoderReranker

    rag = RAGAgent(
        store=store,
        agent=Agent(model="gpt-4o-mini"),
        reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_k=15,
        top_n=4,
    )
    app = rag.compile()
"""

from __future__ import annotations

import inspect
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from injectq import InjectQ

from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.graph.compiled_graph import CompiledGraph
from tenxgraph.core.graph.state_graph import StateGraph
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.media.storage.base import BaseMediaStore
from tenxgraph.storage.media.url_policy import MediaFetchPolicy
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.store_schema import RetrievalStrategy
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.utils.constants import END
from tenxgraph.utils.converter import wrap_documents_block
from tenxgraph.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


logger = logging.getLogger("tenxgraph.prebuilt.rag")

StateT = TypeVar("StateT", bound=AgentState)

# Retrieved documents are handed between nodes through a per-run cache on the RAGAgent, not
# through saved state. A run is one invoke/stream call, so a few hundred entries cover any
# realistic number of runs in flight; the LRU bounds memory in long-lived servers.
_DOCS_CACHE_SIZE = 256

# Node names — module-level constants so tests and interrupt_before/after can
# reference them without magic strings.
_RETRIEVE_NODE = "RETRIEVE"
_RERANK_NODE = "RERANK"
_SYNTHESIZE_NODE = "SYNTHESIZE"


# ---------------------------------------------------------------------------
# Reranker protocol + built-in implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseReranker(Protocol):
    """Protocol for document rerankers.

    A reranker receives the original query and a list of candidate documents
    and returns a shorter, re-ordered list of the most relevant ones.

    Implement ``arerank`` — all built-in rerankers expose this async method.
    """

    async def arerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        """Re-rank *documents* for *query* and return the top *top_n* texts."""
        ...


class CohereReranker:
    """Reranker backed by the Cohere Rerank API.

    Requires the ``cohere`` package::

        pip install cohere

    Args:
        api_key: Cohere API key.
        model: Rerank model name (default ``"rerank-v4.0-pro"``).

    Example::

        reranker = CohereReranker(api_key="...", model="rerank-v4.0-pro")
        ranked = await reranker.arerank(query, docs, top_n=5)
    """

    def __init__(self, api_key: str, model: str = "rerank-v4.0-pro") -> None:
        try:
            import cohere  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "cohere package is required for CohereReranker. Install with: pip install cohere"
            ) from e
        self._api_key = api_key
        self._model = model
        self._client: Any = None  # lazily initialised

    def _get_client(self) -> Any:
        if self._client is None:
            import cohere

            self._client = cohere.AsyncClientV2(self._api_key)
        return self._client

    async def arerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        if not documents:
            return []
        client = self._get_client()
        response = await client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
        return [documents[r.index] for r in response.results]


class CrossEncoderReranker:
    """Reranker backed by a local CrossEncoder model (sentence-transformers).

    No API key required — runs entirely on local hardware.  Suitable for
    private data or offline environments.

    Requires the ``sentence-transformers`` package::

        pip install sentence-transformers

    Args:
        model: CrossEncoder model from HuggingFace
            (default ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``).

    Example::

        reranker = CrossEncoderReranker()
        ranked = await reranker.arerank(query, docs, top_n=5)
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model_name = model
        self._encoder: Any = None  # lazily loaded

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self._model_name)
        return self._encoder

    async def arerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        """Score with CrossEncoder (CPU-bound sync) and return top-n texts."""
        import asyncio

        if not documents:
            return []

        encoder = self._get_encoder()
        pairs = [[query, doc] for doc in documents]

        # CrossEncoder.predict is CPU-bound — run in executor to not block event loop
        loop = asyncio.get_event_loop()
        scores: list[float] = await loop.run_in_executor(None, encoder.predict, pairs)

        ranked = sorted(zip(scores, documents, strict=False), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_documents_block(docs: list[str]) -> str:
    """Number the retrieved chunks and fence them as data for the outgoing request."""
    numbered = "\n".join(f"[{i + 1}] {doc.strip()}" for i, doc in enumerate(docs))
    return wrap_documents_block(numbered)


def _latest_user_query(state: AgentState) -> str:
    """Text of the latest user message in the context, or an empty string."""
    for msg in reversed(state.context):
        if msg.role == "user":
            return msg.text()
    return ""


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------


class RAGAgent[StateT: AgentState]:
    """Retrieve → (Rerank) → Synthesize agent.

    The agent owns the full retrieval pipeline.  Users provide:

    * ``store`` — the knowledge base (:class:`~tenxgraph.storage.store.BaseStore`)
    * ``agent`` — the LLM that generates the answer
        (:class:`~tenxgraph.core.graph.base_agent.BaseAgent`)
    * ``reranker`` — optional :class:`BaseReranker` for improved precision

    Args:
        store: Knowledge-base store (e.g. ``QdrantStore``).  The RETRIEVE node
            calls ``store.asearch(query, limit=top_k)`` and keeps the result
            texts in the per-run cache.
        agent: Pre-built agent that synthesises the final answer.  The
            SYNTHESIZE node calls
            ``agent.execute(state, config, request_blocks=[...])`` with the fenced
            documents, which are added to the outgoing request only.
        reranker: Optional reranker.  When provided, a RERANK node is inserted
            between RETRIEVE and SYNTHESIZE.  Use :class:`CohereReranker` for
            API-based reranking or :class:`CrossEncoderReranker` for fully
            local reranking.
        top_k: Candidates retrieved from the store (default ``5``).  With a
            reranker, increase this (e.g. ``20``) to give the reranker more
            candidates.
        top_n: Documents forwarded to the LLM after reranking (default ``3``).
            Ignored when no reranker is provided — all ``top_k`` docs are used.
        retrieval_strategy: Vector search strategy passed to ``store.asearch``
            (default :attr:`~tenxgraph.storage.store.store_schema.RetrievalStrategy.SIMILARITY`).
        score_threshold: Minimum similarity score; ``None`` means no cutoff.
        store_config: Extra key/value pairs passed as the ``config`` argument
            to every ``store.asearch`` call (e.g. ``{"user_id": "u42"}``).  The
            same dict is used for every user, so the knowledge base is shared.
        store_config_builder: Optional ``(run_config) -> dict`` that builds the
            ``config`` for ``store.asearch`` from the run config of each request.
            When set it replaces ``store_config``, so the search can be scoped per
            user or tenant.  ``QdrantStore`` filters on ``config["user_id"]`` and
            picks the collection from ``config["collection"]``::

                rag = RAGAgent(
                    store=store,
                    agent=agent,
                    store_config_builder=lambda cfg: {"user_id": cfg["user_id"]},
                )

            Take identity from the authenticated run config, never from state or
            client-chosen fields.

    Graph infra params (forwarded to :class:`StateGraph`):
        state, context_manager, publisher, id_generator, container
    """

    def __init__(  # noqa: PLR0913
        self,
        store: BaseStore,
        agent: BaseAgent,
        reranker: BaseReranker | None = None,
        top_k: int = 5,
        top_n: int = 3,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        score_threshold: float | None = None,
        store_config: dict[str, Any] | None = None,
        store_config_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        # Graph infra
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | list[BasePublisher] | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")

        self._store = store
        self._agent = agent
        self._reranker = reranker
        self._top_k = top_k
        self._top_n = top_n
        self._retrieval_strategy = retrieval_strategy
        self._score_threshold = score_threshold
        self._store_config: dict[str, Any] = store_config or {}
        self._store_config_builder = store_config_builder
        # (run_id, query) -> documents; written by RETRIEVE, updated by RERANK, read by
        # SYNTHESIZE. Never saved to state or the checkpointer.
        self._docs_cache: OrderedDict[tuple[str, str], list[str]] = OrderedDict()

        # Graph infra — stored for _new_graph() reuse across compile() calls.
        self._state = state
        self._context_manager = context_manager
        self._publisher = publisher
        self._id_generator = id_generator
        self._container = container

        self._graph = self._new_graph()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_graph(self) -> StateGraph[StateT]:
        return StateGraph[StateT](
            state=self._state,
            context_manager=self._context_manager,
            publisher=self._publisher,
            id_generator=self._id_generator,
            container=self._container,
        )

    # ------------------------------------------------------------------
    # Retrieval + per-run cache
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(config: dict[str, Any], query: str) -> tuple[str, str] | None:
        run_id = (config or {}).get("run_id")
        return (str(run_id), query) if run_id else None

    def _cache_put(self, key: tuple[str, str] | None, docs: list[str]) -> None:
        if key is None:
            return
        self._docs_cache[key] = docs
        self._docs_cache.move_to_end(key)
        while len(self._docs_cache) > _DOCS_CACHE_SIZE:
            self._docs_cache.popitem(last=False)

    async def _fetch_docs(self, query: str, config: dict[str, Any]) -> list[str]:
        """Search the knowledge base; the search config is scoped per request if configured."""
        if not query:
            logger.warning("RAGAgent RETRIEVE: no user message found in context.")
            return []
        if self._store_config_builder is not None:
            store_config = self._store_config_builder(config)
        else:
            store_config = self._store_config
        logger.debug("RAGAgent RETRIEVE: query=%r top_k=%d", query, self._top_k)
        results = await self._store.asearch(
            config=store_config,
            query=query,
            limit=self._top_k,
            score_threshold=self._score_threshold,
            retrieval_strategy=self._retrieval_strategy,
        )
        docs = [r.content for r in results if r.content]
        logger.debug("RAGAgent RETRIEVE: got %d docs", len(docs))
        return docs

    async def _rerank_docs(self, query: str, docs: list[str]) -> list[str]:
        if self._reranker is None or not docs:
            return docs
        logger.debug("RAGAgent RERANK: %d docs -> top_%d", len(docs), self._top_n)
        return await self._reranker.arerank(query, docs, top_n=self._top_n)

    # ------------------------------------------------------------------
    # Node factories
    # ------------------------------------------------------------------

    def _make_retrieve_node(self):
        async def _retrieve(state: AgentState, config: dict) -> AgentState:
            query = _latest_user_query(state)
            key = self._cache_key(config, query)
            # Without a run_id there is nowhere to hand the documents over, so SYNTHESIZE
            # does the search itself rather than searching twice.
            if key is not None:
                self._cache_put(key, await self._fetch_docs(query, config))
            return state

        return _retrieve

    def _make_rerank_node(self):
        async def _rerank(state: AgentState, config: dict) -> AgentState:
            query = _latest_user_query(state)
            key = self._cache_key(config, query)
            if key is None or key not in self._docs_cache:
                return state  # SYNTHESIZE retrieves and reranks on a miss
            docs = self._docs_cache[key]
            if docs:
                self._cache_put(key, await self._rerank_docs(query, docs))
            return state

        return _rerank

    def _make_synthesize_node(self):
        agent = self._agent

        async def _synthesize(state: AgentState, config: dict) -> AgentState:
            query = _latest_user_query(state)
            key = self._cache_key(config, query)

            if key is not None and key in self._docs_cache:
                self._docs_cache.move_to_end(key)
                docs = self._docs_cache[key]
            else:
                # No entry: no run_id, evicted, or the process restarted between nodes.
                # Search again rather than answer without documents.
                docs = await self._rerank_docs(query, await self._fetch_docs(query, config))
                self._cache_put(key, docs)

            if not docs:
                return await agent.execute(state, config)

            if "request_blocks" not in inspect.signature(agent.execute).parameters:
                raise TypeError(
                    f"{type(agent).__name__}.execute must accept a 'request_blocks' keyword "
                    "so RAGAgent can add the retrieved documents to the request without "
                    "modifying the saved thread."
                )
            logger.debug("RAGAgent SYNTHESIZE: passing %d docs.", len(docs))
            return await agent.execute(state, config, request_blocks=[_build_documents_block(docs)])

        return _synthesize

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def _configure_graph(self) -> None:
        """Wire RETRIEVE → [RERANK →] SYNTHESIZE → END."""
        self._graph = self._new_graph()
        self._graph.add_node(_RETRIEVE_NODE, self._make_retrieve_node())
        self._graph.add_node(_SYNTHESIZE_NODE, self._make_synthesize_node())

        if self._reranker is not None:
            self._graph.add_node(_RERANK_NODE, self._make_rerank_node())
            self._graph.set_entry_point(_RETRIEVE_NODE)
            self._graph.add_edge(_RETRIEVE_NODE, _RERANK_NODE)
            self._graph.add_edge(_RERANK_NODE, _SYNTHESIZE_NODE)
        else:
            self._graph.set_entry_point(_RETRIEVE_NODE)
            self._graph.add_edge(_RETRIEVE_NODE, _SYNTHESIZE_NODE)

        self._graph.add_edge(_SYNTHESIZE_NODE, END)

    def compile(
        self,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
        media_store: BaseMediaStore | None = None,
        shutdown_timeout: float = 30.0,
        media_policy: MediaFetchPolicy | None = None,
    ) -> CompiledGraph:
        """Compile the RAG graph.

        Args:
            checkpointer: Conversation-state persistence backend.
            store: Long-term agent-memory store (separate from the
                knowledge-base store used for retrieval).
            interrupt_before: Node names to pause *before*.
            interrupt_after: Node names to pause *after*.
            callback_manager: Lifecycle callbacks.
            media_store: Optional media/file store.
            shutdown_timeout: Graceful shutdown timeout (seconds).

        Returns:
            A compiled, invocable :class:`~tenxgraph.core.graph.CompiledGraph`.
        """
        self._configure_graph()
        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
            media_store=media_store,
            shutdown_timeout=shutdown_timeout,
            media_policy=media_policy,
        )


__all__ = [
    "BaseReranker",
    "CohereReranker",
    "CrossEncoderReranker",
    "RAGAgent",
]
