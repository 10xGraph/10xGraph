"""Typing-only description of ``PgCheckpointer`` for its mixins.

The mixins use attributes and methods that live on ``PgCheckpointer`` or on a sibling mixin.
``_PgHost`` declares them for mypy. It exists only while type checking: at runtime
``PgHostBase`` is ``object``, so no stub can shadow a real method in the MRO.
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Sequence
    from typing import Any

    from tenxgraph.storage.columns import ColumnSet, Isolation
    from tenxgraph.storage.sql_filter import FilterSchema
    from tenxgraph.utils import ThreadInfo
    from tenxgraph.utils.id_generator import IDType

    from .base_checkpointer import BaseCheckpointer
    from .pg_checkpointer import _RedisBreaker

    class _PgHost(BaseCheckpointer[Any]):
        # Set in PgCheckpointer.__init__ (mixins set the setup-state ones).
        user_id_type: IDType
        _pinned_id_type: IDType | None
        _id_type_given: bool
        cache_ttl: int
        redis_key_prefix: str
        pg_acquire_timeout: float | None
        state_history_limit: int
        enforce_user_isolation: bool
        columns: ColumnSet
        filter_schema: FilterSchema
        row_level_security: bool
        ddl_lock_timeout: float | None
        schema: str
        redis: Any
        _redis_breaker: _RedisBreaker | None
        _skipped_invalidations: set[str]
        _schema_initialized: bool
        _schema_init_lock: asyncio.Lock

        # _PgIdentityMixin
        @property
        def id_type(self) -> IDType: ...
        @staticmethod
        def _generator_id_type() -> IDType: ...

        # PgCheckpointer
        def _get_table_name(self, table: str) -> str: ...
        async def _get_pg_pool(self) -> Any: ...

        # _PgAccessMixin
        async def _retry_on_connection_error(
            self, operation: Any, *args: Any, max_retries: int = 3, **kwargs: Any
        ) -> Any: ...
        async def _run_query(
            self,
            fn: Any,
            *,
            in_transaction: bool = False,
            settings: Sequence[tuple[str, str]] | None = None,
        ) -> Any: ...
        def _isolation(
            self, config: dict[str, Any] | None, *, write: bool = False
        ) -> Isolation: ...
        def _rls(
            self, config: dict[str, Any] | None, *, write: bool = False
        ) -> list[tuple[str, str]] | None: ...
        def _thread_filter(
            self,
            user_id: str | int | None,
            config: dict[str, Any] | None,
            params: list[Any],
            isolation: Isolation | None = None,
            *,
            write: bool = False,
        ) -> str: ...
        def _thread_row_scope(
            self, thread_id: str | int, user_id: str | int | None, config: dict[str, Any] | None
        ) -> tuple[str, list[Any]]: ...
        def _append_owner_scope(
            self,
            query: str,
            params: list[Any],
            user_id: str | int | None,
            config: dict[str, Any] | None,
            *,
            write: bool = False,
        ) -> str: ...
        def _thread_insert_sql(
            self,
            thread_id: str | int,
            thread_name: Any,
            user_id: str | int,
            meta_json: str,
            isolation: Isolation,
            tail: str,
        ) -> tuple[str, list[Any]]: ...
        def _row_to_thread_info(self, row: Any) -> ThreadInfo: ...
        async def _ensure_thread_exists(
            self, thread_id: str | int, user_id: str | int, config: dict[str, Any]
        ) -> None: ...
        def _deserialize_state_payload(
            self, data: dict[str, Any], thread_id: Any = None
        ) -> Any: ...
        def _thread_scope(
            self,
            thread_id: str | int,
            user_id: str | int | None,
            config: dict[str, Any] | None = None,
        ) -> tuple[str, list[Any]]: ...
        async def _lock_thread_for_write(
            self, conn: Any, thread_id: str | int, user_id: str | int, config: dict[str, Any]
        ) -> int: ...

        # _PgCacheMixin
        async def _invalidate_state_cache(
            self,
            thread_id: str | int,
            user_id: str | int,
            retries: int = 0,
            config: dict[str, Any] | None = None,
        ) -> None: ...

        # _PgIdentityMixin
        def _validate_config(self, config: dict[str, Any]) -> tuple[str | int, str | int]: ...
        def _db_thread_id(self, thread_id: str | int) -> str | int: ...
        def _db_message_id(self, message_id: str | int) -> str | int: ...
        def _check_message_ids(self, messages: list[Any] | None) -> None: ...
        def _message_thread_id(self, config: dict[str, Any]) -> str | int: ...
        def _get_thread_key(
            self, thread_id: str | int, user_id: str | int, config: dict[str, Any] | None = None
        ) -> str: ...
        def _key_identity(self, user_id: Any, config: dict[str, Any] | None) -> tuple[Any, str]: ...
        def _thread_identity(self, config: dict[str, Any] | None) -> dict[str, Any] | None: ...
        def _remember_identity(
            self, config: dict[str, Any] | None, thread_id: Any, row: Any
        ) -> None: ...
        def _identity_columns_sql(self) -> str: ...
        async def _read_thread_identity(
            self, thread_id: Any, user_id: Any, config: dict[str, Any]
        ) -> Any: ...
        async def _resolve_thread_identity(self, config: dict[str, Any]) -> None: ...

        # _PgSchemaMixin
        async def _initialize_schema(self) -> None: ...

    PgHostBase = _PgHost
else:
    PgHostBase = object
