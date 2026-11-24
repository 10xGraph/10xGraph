"""Typing-only description of ``PgVectorStore`` for its mixins.

The mixins use attributes and methods that live on ``PgVectorStore`` or on a sibling mixin.
``_PgvHost`` declares them for mypy. It exists only while type checking: at runtime
``PgvHostBase`` is ``object``, so no stub can shadow a real method in the MRO.
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable, Sequence
    from typing import Any, Literal

    from tenxgraph.core.state import Message
    from tenxgraph.storage.columns import ColumnSet, Isolation
    from tenxgraph.storage.sql_filter import FilterSchema

    from ._pgv_support import _Where
    from .base_store import BaseStore
    from .embedding.base_embedding import BaseEmbedding
    from .store_schema import (
        DistanceMetric,
        MemorySearchResult,
        MemoryType,
        RetrievalStrategy,
    )

    class _PgvHost(BaseStore):
        # Set in PgVectorStore.__init__ (mixins set the setup-state ones).
        columns: ColumnSet
        filter_schema: FilterSchema
        row_level_security: bool
        ddl_lock_timeout: float | None
        embedding: BaseEmbedding | None
        schema: str
        table: str
        text_search_config: str
        on_embedding_error: Literal["fallback", "raise"]
        distance_metric: DistanceMetric
        retrieval_strategy: RetrievalStrategy
        hybrid_weights: tuple[float, float]
        rrf_k: int
        hybrid_candidates: int | None
        _setup_lock: asyncio.Lock
        _setup_done: bool
        _index_metric: DistanceMetric | None
        _warned_metrics: set[DistanceMetric]
        _iterative_scan: bool
        _server_version: int
        _legacy_key_indexes: list[str]

        # PgVectorStore
        def _check_strategy(self, strategy: RetrievalStrategy) -> RetrievalStrategy: ...
        @property
        def _qualified_table(self) -> str: ...
        @property
        def _version_table(self) -> str: ...
        def _key_columns(self) -> list[str]: ...
        def _dimension(self) -> int: ...
        def _setup_lock_key(self) -> int: ...
        async def _get_pool(self) -> Any: ...
        async def _retry_on_connection_error(
            self, operation: Callable[[], Awaitable[Any]]
        ) -> Any: ...
        async def _run(
            self,
            fn: Callable[[Any], Awaitable[Any]],
            *,
            setup: bool = True,
            settings: Sequence[tuple[str, str]] | None = None,
        ) -> Any: ...

        # _PgvSchemaMixin
        def _key_nulls_clause(self) -> str: ...
        def _is_key_index_name(self, name: str) -> bool: ...
        async def _ensure_setup(self) -> None: ...

        # _PgvQueryMixin
        def _tenant_context(self, config: dict[str, Any], *, write: bool = False) -> Isolation: ...
        def _search_settings(
            self, ctx: Isolation, vector_search: bool
        ) -> list[tuple[str, str]]: ...
        def _scope_where(
            self,
            params: list[Any],
            collection: str,
            user_id: Any,
            alias: str | None = None,
            tenant: Sequence[tuple[Any, Any]] = (),
        ) -> _Where: ...
        def _add_filters(
            self,
            where: _Where,
            memory_type: MemoryType | str | None = None,
            category: str | None = None,
            filters: dict[str, Any] | None = None,
        ) -> None: ...
        @staticmethod
        def _add_provenance(where: _Where, origins: Any = None) -> None: ...
        def _vector_literal(self, vector: Any, *, cosine: bool = False) -> str: ...
        async def _embed(
            self, text: str, *, query: bool, metric: DistanceMetric | None = None
        ) -> str | None: ...
        @staticmethod
        def _prepare(
            content: str | Message, memory_type: MemoryType | str, metadata: dict[str, Any] | None
        ) -> tuple[str, MemoryType, dict[str, Any]]: ...
        @staticmethod
        def _memory_key(explicit: Any, metadata: dict[str, Any] | None) -> str | None: ...
        @staticmethod
        def _row_to_result(row: Any, score: float) -> MemorySearchResult: ...
        def _columns(self, alias: str | None = None) -> str: ...
        def _warn_metric(self, metric: DistanceMetric) -> None: ...

    PgvHostBase = _PgvHost
else:
    PgvHostBase = object
