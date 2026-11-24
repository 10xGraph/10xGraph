"""Thread identity, Redis key building and the stop and active-run markers."""

import logging
from typing import Any
from urllib.parse import quote

from tenxgraph.core.authz import owner_id
from tenxgraph.utils.id_generator import IDType
from tenxgraph.utils.log_safety import describe_error

from ._pg_host import PgHostBase
from ._pg_support import (
    _THREAD_IDENTITY_KEY,
)


logger = logging.getLogger("tenxgraph.checkpointer.pg")


class _PgIdentityMixin(PgHostBase):
    """Thread identity, Redis key building and the stop and active-run markers."""

    ###########################
    #### HELPER METHODS #######
    ###########################

    def _validate_config(self, config: dict[str, Any]) -> tuple[str | int, str | int]:
        """
        Extract and validate thread_id and user_id from config.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            tuple: (thread_id, user_id)

        Raises:
            ValueError: If required fields are missing.
        """
        thread_id = config.get("thread_id")
        user_id = owner_id(config)
        if not user_id:
            raise ValueError("user_id must be provided in config")

        if not thread_id:
            raise ValueError("Both thread_id must be provided in config")

        return self._db_thread_id(thread_id), user_id

    @property
    def declared_id_type(self) -> IDType:
        """The id type the setup's ``id_type`` holds (``int``, ``bigint``, else text)."""
        return {"int": IDType.INTEGER, "bigint": IDType.BIGINT}.get(self.id_type, IDType.STRING)

    def _check_id(self, value: Any, what: str) -> Any:
        """``value`` unchanged when it has the type the setup's ``id_type`` expects.

        ``id_type`` text takes strings, ``int`` and ``bigint`` take integers. Anything else
        raises ``ValueError`` before any SQL, so no driver error reaches the caller.
        """
        wants_int = self.id_type in ("int", "bigint")
        ok = (
            (isinstance(value, int) and not isinstance(value, bool))
            if wants_int
            else isinstance(value, str)
        )
        if not ok:
            expected = "an integer" if wants_int else "a string"
            raise ValueError(
                f"{what} must be {expected} for id_type={self.id_type!r}, "
                f"got {type(value).__name__}: {value!r}"
            )
        return value

    def _db_thread_id(self, thread_id: str | int) -> str | int:
        """``thread_id`` checked against the setup's ``id_type`` (see ``_check_id``)."""
        return self._check_id(thread_id, "thread_id")

    def _db_message_id(self, message_id: str | int) -> str | int:
        """``message_id`` checked against the setup's ``id_type`` (see ``_check_id``)."""
        return self._check_id(message_id, "message_id")

    def _check_message_ids(self, messages: list[Any] | None) -> None:
        """Every message id must have the type ``id_type`` expects, before any SQL."""
        for message in messages or []:
            self._db_message_id(message.message_id)

    def _message_thread_id(self, config: dict[str, Any]) -> str | int:
        """``thread_id`` of the config in the column's type; message methods need no user."""
        thread_id = config.get("thread_id")
        if not thread_id:
            raise ValueError("thread_id must be provided in config")
        return self._db_thread_id(thread_id)

    def _get_thread_key(
        self,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any] | None = None,
    ) -> str:
        """
        Get Redis cache key for thread state.

        Args:
            thread_id (str|int): Thread identifier.
            user_id (str|int): User identifier.
            config (dict, optional): The call's config. With tenant columns declared the key
                is the thread's identity: its owner and tenant values from the thread row
                (``_thread_identity``), or, before that row has been read, the caller's
                ``user_id`` and tenant values from ``config["user"]``.

        Returns:
            str: Redis cache key.
        """
        owner, suffix = self._key_identity(user_id, config)
        # Ids are percent-encoded so a ":" inside one cannot make two different
        # (thread, user) pairs produce the same key.
        return (
            f"{self.redis_key_prefix}state_cache:"
            f"{quote(str(thread_id), safe='')}:{quote(str(owner), safe='')}{suffix}"
        )

    def _key_identity(self, user_id: Any, config: dict[str, Any] | None) -> tuple[Any, str]:
        """``(owner, tenant suffix)`` the Redis keys of a thread are built from.

        With tenant columns and a ``_thread_identity`` read for this thread, both come from
        the thread row, so every caller allowed to see the thread shares its keys. Otherwise
        the caller's ``user_id`` and tenant values are used (as for a thread this caller is
        about to create, whose row will hold exactly those values).
        """
        if not self.columns.tenant:
            return user_id, ""
        identity = self._thread_identity(config)
        if identity is not None:
            owner = identity["user_id"]
            texts = identity["tenant"]
        else:
            owner = user_id
            values = self.columns.resolve(config)
            texts = {}
            for spec in self.columns.tenant:
                value = values.get(spec.name)
                texts[spec.name] = None if value is None else spec.to_setting(value)
        parts = []
        for spec in self.columns.tenant:
            text = texts.get(spec.name)
            if text is None:
                parts.append(f"{spec.name}!")
            else:
                parts.append(f"{spec.name}={quote(str(text), safe='')}")
        return owner, ":t:" + ":".join(parts)

    def _thread_identity(self, config: dict[str, Any] | None) -> dict[str, Any] | None:
        """The ``_thread_identity`` kept in ``config`` for its current thread, if any.

        Only this class writes it, after the thread row passed the caller's scoped check
        (``_remember_identity``). It is ignored when it was read for another thread.
        """
        if not self.columns.tenant or not isinstance(config, dict):
            return None
        identity = config.get(_THREAD_IDENTITY_KEY)
        if not isinstance(identity, dict):
            return None
        if str(identity.get("thread_id")) != str(config.get("thread_id")):
            return None
        return identity

    def _remember_identity(self, config: dict[str, Any] | None, thread_id: Any, row: Any) -> None:
        """Keep the thread row's owner and tenant values in ``config`` (tenant columns only).

        ``row`` must come from a read that passed the caller's scope check.
        """
        if not self.columns.tenant or not isinstance(config, dict) or row is None:
            return
        tenant = {}
        for spec in self.columns.tenant:
            value = row[spec.name]
            tenant[spec.name] = None if value is None else spec.to_setting(value)
        config[_THREAD_IDENTITY_KEY] = {
            "thread_id": str(thread_id),
            "user_id": row["user_id"],
            "tenant": tenant,
        }

    def _identity_columns_sql(self) -> str:
        """``user_id, "org_id", ...``: the thread-row columns of its identity."""
        return ", ".join(["user_id", *(spec.quoted for spec in self.columns.tenant)])

    async def _read_thread_identity(
        self, thread_id: Any, user_id: Any, config: dict[str, Any]
    ) -> Any:
        """Read the thread row's identity and current state version under the caller's scope.

        Returns the row (``user_id``, tenant columns, ``durable_version``) and remembers the
        identity in ``config``, or ``None`` when the caller cannot see the thread or it does
        not exist. A scope the caller cannot satisfy raises before any query.
        """
        where, params = self._thread_row_scope(thread_id, user_id, config)
        query = (
            f"SELECT {self._identity_columns_sql()}, "  # noqa: S608
            f"(SELECT COALESCE(MAX(version), 0) FROM {self._get_table_name('states')} "
            "WHERE thread_id = $1) AS durable_version "
            f"FROM {self._get_table_name('threads')} WHERE {where}"
        )
        row = await self._run_query(
            lambda conn: conn.fetchrow(query, *params), settings=self._rls(config)
        )
        if row is None:
            if self._thread_identity(config) is not None:
                config.pop(_THREAD_IDENTITY_KEY, None)  # no longer visible: own keys again
            return None
        self._remember_identity(config, thread_id, row)
        return row

    async def _resolve_thread_identity(self, config: dict[str, Any]) -> None:
        """Make the stop and active-run keys of this call the thread's (tenant columns only).

        Reads the thread row once per config under the caller's scope. When the caller cannot
        see the thread, or Postgres cannot be read, nothing is remembered and the caller's own
        values are used, which only ever reach keys of the caller's own identity.
        """
        if not self.columns.tenant or self._thread_identity(config) is not None:
            return
        thread_id, user_id = config.get("thread_id"), owner_id(config)
        if not thread_id or not user_id:
            return
        try:
            await self._read_thread_identity(self._db_thread_id(thread_id), user_id, config)
        except Exception as e:  # the caller's own keys are the safe fallback
            logger.debug(
                "Could not read the identity of thread_id=%s: %s", thread_id, describe_error(e)
            )

    def _stop_request_key(self, config: dict[str, Any]) -> str:
        """Per-thread stop/active-run key.

        Without tenant columns: ``<thread>:<caller user_id>`` as in the base class. With
        tenant columns: the thread's identity (owner and tenant values), so an owner working
        in another project, or an org admin, reaches the run's stop and active-run keys.
        """
        if not self.columns.tenant:
            return super()._stop_request_key(config)
        owner, suffix = self._key_identity(owner_id(config), config)
        return f"{config.get('thread_id')}:{owner}{suffix}"

    async def aget_active_run(self, config: dict[str, Any]) -> str | None:
        await self._resolve_thread_identity(config)
        return await super().aget_active_run(config)

    async def abegin_run(self, config: dict[str, Any]) -> None:
        await self._resolve_thread_identity(config)
        await super().abegin_run(config)

    async def arefresh_run(self, config: dict[str, Any]) -> None:
        await self._resolve_thread_identity(config)
        await super().arefresh_run(config)

    async def aend_run(self, config: dict[str, Any]) -> None:
        await self._resolve_thread_identity(config)
        await super().aend_run(config)

    async def arequest_stop(self, config: dict[str, Any]) -> Any | None:
        await self._resolve_thread_identity(config)
        return await super().arequest_stop(config)

    async def ais_stop_requested(self, config: dict[str, Any]) -> bool:
        await self._resolve_thread_identity(config)
        return await super().ais_stop_requested(config)

    async def aclear_stop_request(self, config: dict[str, Any]) -> Any | None:
        await self._resolve_thread_identity(config)
        return await super().aclear_stop_request(config)
