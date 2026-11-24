import copy
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from itertools import islice
from typing import TYPE_CHECKING, Any, TypeVar

from tenxgraph.core.authz import Reach, owner_filter, owner_id
from tenxgraph.core.exceptions import MessageNotFoundError, ThreadNotFoundError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils.thread_info import ThreadInfo

from .base_checkpointer import (
    BaseCheckpointer,
    _reject_filters,
    _reject_tenant_options,
    reject_unstorable,
    validate_page,
)


if TYPE_CHECKING:
    from tenxgraph.core.state import AgentState, Message

logger = logging.getLogger("tenxgraph.checkpointer")

StateT = TypeVar("StateT", bound="AgentState")


class InMemoryCheckpointer[StateT: AgentState](BaseCheckpointer[StateT]):
    """
    In-memory implementation of BaseCheckpointer.

    Stores all agent state, messages, and thread info in memory using Python dictionaries.
    Data is lost when the process ends. Designed for testing and ephemeral use cases.
    Async-first design.

    Args:
        None

    Attributes:
        _states (dict): Stores agent states by thread key.
        _state_cache (dict): Stores cached agent states by thread key.
        _messages (dict): Stores messages by thread key.
        _threads (dict): Stores thread info by thread key.
    """

    def __init__(
        self,
        *,
        tenant_columns: Any = None,
        extra_columns: Any = None,
        row_level_security: bool = False,
        record_tool_result: bool = True,
    ):
        """
        Initialize all in-memory storage.

        ``tenant_columns``, ``extra_columns`` and ``row_level_security`` exist only to fail
        loudly: they need ``PgCheckpointer``, and passing any of them raises ``ValueError``.
        ``record_tool_result=False`` turns the tool ledger off for every tool (see
        :attr:`BaseCheckpointer.record_tool_result`).
        """
        _reject_tenant_options(
            "InMemoryCheckpointer",
            tenant_columns=tenant_columns,
            extra_columns=extra_columns,
            row_level_security=row_level_security,
        )
        # State storage
        self._states: dict[str, StateT] = {}
        self._state_cache: dict[str, StateT] = {}
        self._generic_cache: dict[str, tuple[Any, float | None]] = {}

        # Message storage - organized by config key
        self._messages: dict[str, list[Message]] = defaultdict(list)

        # Thread storage
        self._threads: dict[str, dict[str, Any]] = {}

        # Thread ownership: config-key (thread_id) -> owner user_id. The first writer of a
        # thread owns it. Used to enforce owner-only access when the caller identity
        # (config["user"]) asks for it -- see _owner_denied / _claim.
        self._thread_owner: dict[str, str] = {}
        # The owner's tenant column values, recorded with the owner on backends that declare
        # ``tenant_column_names``; see aget_thread_scope.
        self._thread_tenant: dict[str, dict[str, str]] = {}

        self.record_tool_result = record_tool_result
        # Tool execution ledger (idempotency): (thread, tool_call_id) -> (result, recorded at)
        self._tool_results: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}

    def setup(self) -> Any:
        """
        Synchronous setup method. No setup required for in-memory checkpointer.
        """
        logger.debug("InMemoryCheckpointer setup not required")

    async def asetup(self) -> Any:
        """
        Asynchronous setup method. No setup required for in-memory checkpointer.
        """
        logger.debug("InMemoryCheckpointer async setup not required")

    def _claim(self, config: dict[str, Any]) -> str | None:
        """Make the caller the owner of a new thread, or refuse a write to someone else's.

        Call it right before the write, with no ``await`` in between: checking and recording
        are then one step on the event loop, so of two writers racing to create one thread
        the loser is refused instead of writing into the winner's. Returns the owner.

        Raises:
            ThreadNotFoundError: Another user owns the thread (writes always stay own).
        """
        user_id = owner_id(config)
        key = self._get_config_key(config)
        owner: str | None
        if user_id:
            if key not in self._thread_owner:
                tenant_columns = getattr(self, "tenant_column_names", ())
                if tenant_columns:
                    values = owner_filter(config, tenant_columns, reach=Reach.OWN) or {}
                    self._thread_tenant[key] = {c: values[c] for c in tenant_columns if c in values}
            owner = self._thread_owner.setdefault(key, str(user_id))
        else:
            owner = self._thread_owner.get(key)
        if self._foreign_owner(config, owner, write=True):
            raise ThreadNotFoundError(config.get("thread_id"))
        return owner

    def _ensure_thread(self, config: dict[str, Any], owner: str | None) -> None:
        """Create the thread record on the first write of a thread; never overwrite one.

        Same fields as ``PgCheckpointer`` writes: id, owner, name and metadata from the config.
        """
        key = self._get_config_key(config)
        if key not in self._threads:
            info = ThreadInfo(
                thread_id=key,
                thread_name=config.get("thread_name", f"Thread {key}"),
                user_id=owner,
                metadata=config.get("thread_meta", {}),
            )
            self._threads[key] = info.model_dump()

    def _owner_denied(self, config: dict[str, Any], *, write: bool = False) -> bool:
        """True when isolation is active and the caller may not see this thread.

        Reads and deletes follow the caller's reach; ``write=True`` checks the caller owns
        it. A thread with no recorded owner (brand new) is never denied: the caller becomes
        its owner on write, and reads simply find nothing.
        """
        owner = self._thread_owner.get(self._get_config_key(config))
        return self._foreign_owner(config, owner, write=write)

    # -------------------------
    # State methods Async
    # -------------------------
    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store state asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.
        """
        reject_unstorable(state)
        key = self._get_config_key(config)
        # Store copies: loaders merge new input into the state they get back, so sharing one
        # object would let a request that is later rejected (a paused thread) change what is
        # stored. Copied before the claim, so a state that cannot be copied claims nothing.
        stored, cached = copy.deepcopy(state), copy.deepcopy(state)
        self._ensure_thread(config, self._claim(config))
        self._states[key] = stored
        # Write-through: a run reads the cache first, so leaving it behind would
        # start the next run from the state this write replaced.
        self._state_cache[key] = cached
        logger.debug(f"Stored state for key: {key}")
        return state

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.
        """
        if self._owner_denied(config):
            return None  # not the owner -> as if it does not exist
        key = self._get_config_key(config)
        state = self._states.get(key)
        logger.debug(f"Retrieved state for key: {key}, found: {state is not None}")
        return copy.deepcopy(state)

    async def aclear_state(self, config: dict[str, Any]) -> int:
        """
        Clear state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            int: 1 when a state was removed, 0 when there was none the caller may see.
            0 is not "thread not found": the caller's own thread with no state also gives 0.
        """
        key = self._get_config_key(config)
        if self._owner_denied(config):
            return 0  # outside the caller's reach: no-op, do not reveal existence
        removed = self._states.pop(key, None) is not None
        # A run reads the cache first, so a cached copy would bring the state back.
        self._state_cache.pop(key, None)
        logger.debug(f"Cleared state for key: {key}")
        return int(removed)

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store state cache asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            StateT: The cached state object.
        """
        # Same rule as aput_state, but a non-owner's write is dropped instead of raising: the
        # first writer of a thread becomes its owner, so only its owner can read it back.
        key = self._get_config_key(config)
        cached = copy.deepcopy(state)
        try:
            self._claim(config)
        except ThreadNotFoundError:
            return state
        self._state_cache[key] = cached
        logger.debug(f"Stored state cache for key: {key}")
        return state

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve state cache asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Cached state or None.
        """
        if self._owner_denied(config):
            return None
        key = self._get_config_key(config)
        cache = self._state_cache.get(key)
        logger.debug(f"Retrieved state cache for key: {key}, found: {cache is not None}")
        return copy.deepcopy(cache)

    async def aclear_state_cache(self, config: dict[str, Any]) -> bool:
        """Drop the cached state for a thread; the durable state is untouched."""
        if self._owner_denied(config):
            return True  # not the owner -> no-op, do not reveal existence
        key = self._get_config_key(config)
        self._state_cache.pop(key, None)
        return True

    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        reject_unstorable(value)
        cache_key = f"{namespace}:{key}"
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        self._generic_cache[cache_key] = (value, expires_at)
        return value

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        cache_key = f"{namespace}:{key}"
        cached = self._generic_cache.get(cache_key)
        if cached is None:
            return None

        value, expires_at = cached
        if expires_at is not None and expires_at <= time.time():
            self._generic_cache.pop(cache_key, None)
            return None
        return value

    # -------------------------
    # Tool execution ledger
    # -------------------------
    def _tool_key(self, config: dict[str, Any], tool_call_id: str) -> tuple[str, str]:
        # A tuple, not "thread:call": a thread id containing a colon must not alias another's.
        return (self._get_config_key(config), str(tool_call_id))

    async def aget_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """Return a recorded tool result, so a replayed tool call is not re-fired."""
        if self._owner_denied(config):
            return None  # not the owner -> no record, do not reveal existence
        entry = self._tool_results.get(self._tool_key(config, tool_call_id))
        return entry[0] if entry is not None else None

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Record that a tool call completed, with its result."""
        reject_unstorable(result)
        self._ensure_thread(config, self._claim(config))
        self._tool_results.setdefault(self._tool_key(config, tool_call_id), (result, time.time()))
        return True

    async def aclear_tool_results(
        self,
        config: dict[str, Any],
        origin_message_ids: Sequence[str] | None = None,
    ) -> int:
        """Delete the thread's ledger records (only those of ``origin_message_ids`` when given)."""
        if origin_message_ids is not None and not origin_message_ids:
            return 0
        if self._owner_denied(config, write=True):
            return 0
        key = self._get_config_key(config)
        prefixes = (
            None if origin_message_ids is None else tuple(f"{i}:" for i in origin_message_ids)
        )
        stale = [
            k
            for k in self._tool_results
            if k[0] == key and (prefixes is None or k[1].startswith(prefixes))
        ]
        for ledger_key in stale:
            del self._tool_results[ledger_key]
        return len(stale)

    async def aprune_tool_results(self, older_than: timedelta) -> int:
        """Delete ledger records of every thread older than ``older_than``."""
        cutoff = time.time() - older_than.total_seconds()
        stale = [k for k, (_, at) in self._tool_results.items() if at < cutoff]
        for ledger_key in stale:
            del self._tool_results[ledger_key]
        return len(stale)

    async def aclear_cache_value(self, namespace: str, key: str) -> bool:
        cache_key = f"{namespace}:{key}"
        return self._generic_cache.pop(cache_key, None) is not None

    async def aclear_cache_value_if(self, namespace: str, key: str, expected: Any) -> bool:
        """Compare and delete under the cache lock, in one step."""
        cache_key = f"{namespace}:{key}"
        cached = self._generic_cache.get(cache_key)
        if cached is None or cached[0] != expected:
            return False
        del self._generic_cache[cache_key]
        return True

    async def aput_cache_value_if_absent(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Check and store under the cache lock, in one step; an expired value counts as absent."""
        reject_unstorable(value)
        cache_key = f"{namespace}:{key}"
        now = time.time()
        cached = self._generic_cache.get(cache_key)
        if cached is not None and (cached[1] is None or cached[1] > now):
            return False
        self._generic_cache[cache_key] = (value, now + ttl_seconds if ttl_seconds else None)
        return True

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        """List all cache keys for a namespace."""
        ns_prefix = f"{namespace}:"
        keys = []
        for full_key in self._generic_cache:
            if full_key.startswith(ns_prefix):
                key_part = full_key[len(ns_prefix) :]
                if prefix is None or key_part.startswith(prefix):
                    keys.append(key_part)
        return keys

    # -------------------------
    # Message methods async
    # -------------------------
    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store messages asynchronously.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Merged into each message's metadata (the message's own
                keys win).

        Returns:
            bool: True if stored.
        """
        reject_unstorable(messages, metadata)
        key = self._get_config_key(config)
        self._ensure_thread(config, self._claim(config))
        # Upsert by message id, as the SQL backends do: a run that re-sends messages it
        # already stored (a resume, the final save after a step checkpoint) must not
        # duplicate them.
        stored = self._messages[key]
        index = {str(m.message_id): i for i, m in enumerate(stored)}
        for given in messages:
            msg = given
            if metadata:
                # Merged into a copy (the caller's message is not changed); the message's own
                # keys win.
                msg = given.model_copy(update={"metadata": {**metadata, **(given.metadata or {})}})
            at = index.get(str(msg.message_id))
            if at is None:
                index[str(msg.message_id)] = len(stored)
                stored.append(msg)
            else:
                stored[at] = msg
        logger.debug(f"Stored {len(messages)} messages for key: {key}")
        return True

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Message: Retrieved message object.

        Raises:
            MessageNotFoundError: If the message is not found.
        """
        key = self._get_config_key(config)
        if self._owner_denied(config):
            raise MessageNotFoundError
        messages = self._messages.get(key, [])
        for msg in messages:
            if str(msg.message_id) == str(message_id):
                return msg
        raise MessageNotFoundError

    async def aupdate_message_metadata(
        self,
        config: dict[str, Any],
        message_id: str | int,
        metadata: dict[str, Any],
    ) -> Message | None:
        """
        Shallow-merge ``metadata`` into a stored message's metadata.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.
            metadata (dict): Keys to set; a key replaces the same stored key.

        Returns:
            Message | None: The updated message, or None if not found or not the caller's
            (a write: it stays own whatever the reach).
        """
        reject_unstorable(metadata)
        if self._owner_denied(config, write=True):
            return None
        key = self._get_config_key(config)
        for msg in self._messages.get(key, []):
            if str(msg.message_id) == str(message_id):
                msg.metadata = {**(msg.metadata or {}), **metadata}
                return msg
        return None

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages asynchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.
        """
        start, limit = validate_page(offset, limit)
        if self._owner_denied(config):
            return []
        key = self._get_config_key(config)
        end = (start + limit) if limit else None
        messages = self._messages.get(key, [])

        # Apply search filter if provided
        if search:
            needle = search.lower()
            messages = [msg for msg in messages if needle in msg.text().lower()]

        # Apply offset and limit
        return messages[start:end]

    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> int:
        """
        Delete a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            int: 1 when deleted, 0 when there is no such message the caller may see.
        """
        key = self._get_config_key(config)
        if self._owner_denied(config):
            return 0
        messages = self._messages.get(key, [])
        for msg in messages:
            if str(msg.message_id) == str(message_id):
                messages.remove(msg)
                logger.debug(f"Deleted message with ID {message_id} for key: {key}")
                break
        else:
            return 0
        # The state (and its cache copy) carries the message too; a deleted message must not
        # come back on the next run or in GET state.
        for store in (self._states, self._state_cache):
            if key in store:
                self._strip_message_from_state(store[key], message_id)
        return 1

    # -------------------------
    # Thread methods async
    # -------------------------
    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> bool:
        """
        Store thread info asynchronously.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            bool: True if the thread was created, False if it already existed and was updated.

        Raises:
            ValueError: ``thread_info.thread_id`` differs from ``config["thread_id"]``.
            ThreadNotFoundError: Another user owns the thread.
        """
        reject_unstorable(thread_info)
        key = self._get_config_key(config) or str(thread_info.thread_id)
        if str(thread_info.thread_id) != key:
            raise ValueError(
                "ThreadInfo.thread_id must match config['thread_id']: "
                f"{thread_info.thread_id!r} != {key!r}"
            )
        record = thread_info.model_dump()
        owner = self._claim(config)
        if owner is not None and str(record.get("user_id")) != owner:
            record["user_id"] = owner  # the stored owner, never one named by thread_info
        created = key not in self._threads
        self._threads[key] = record
        logger.debug(f"Stored thread info for key: {key}")
        return created

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the owner ``user_id`` of ``thread_id``, or None if unknown.

        Threads are keyed by ``thread_id`` here, so the stored record's ``user_id``
        is the owner (``aput_thread`` stores the owner there). Note the in-memory registry
        is only populated by explicit ``aput_thread`` calls. For a thread that only ever had
        state written, there is no record, so the owner recorded by the first write is returned.
        """
        record = self._threads.get(str(thread_id))
        if not record:
            return self._thread_owner.get(str(thread_id))
        return record.get("user_id")

    async def aget_thread_scope(self, thread_id: str | int) -> dict[str, Any] | None:
        """Return ``{"user_id": ..., <tenant column>: ...}`` for a thread, globally.

        Same shape as ``PgCheckpointer``: every tenant column the backend declares, ``None``
        for a value the first writer did not carry. Returns ``None`` when the thread has no
        owner and no record.
        """
        owner = await self.aget_thread_owner(thread_id)
        if owner is None:
            return None
        tenant = self._thread_tenant.get(str(thread_id), {})
        scope: dict[str, Any] = {"user_id": owner}
        for column in getattr(self, "tenant_column_names", ()):
            scope[column] = tenant.get(column)
        return scope

    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        """
        Retrieve thread info asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.
        """
        if self._owner_denied(config):
            return None
        key = self._get_config_key(config)
        thread = self._threads.get(key)
        logger.debug(f"Retrieved thread for key: {key}, found: {thread is not None}")
        return ThreadInfo.model_validate(thread) if thread else None

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ThreadInfo]:
        """
        List all threads asynchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.
            filters (dict, optional): Not supported here (no declared columns); a non-empty
                value raises ``FilterError``.

        Returns:
            list[ThreadInfo]: List of thread information objects.
        """
        _reject_filters("InMemoryCheckpointer", filters)
        start, limit = validate_page(offset, limit)
        end = (start + limit) if limit else None
        user_id = owner_id(config)
        isolate = self._isolation_enabled(config, user_id)
        threads: list[dict[str, Any]] = list(self._threads.values())

        # Owner-scope the listing when the identity asks for it.
        if isolate:
            threads = [t for t in threads if str(t.get("user_id")) == str(user_id)]

        # Apply search filter if provided
        if search:
            needle = search.lower()
            threads = [t for t in threads if needle in str(t.get("thread_name") or "").lower()]

        # Apply offset and limit
        return [ThreadInfo.model_validate(t) for t in islice(threads, start, end)]

    async def aclean_thread(self, config: dict[str, Any]) -> int:
        """
        Delete a thread and all of its data asynchronously.

        The data goes first and the owner record last, so until the delete ends no other
        user can claim the thread and write into data that is about to be removed.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            int: 1 when anything of the thread was deleted (its record or any of its data,
            also for a thread that never got a record), 0 when there is nothing the caller
            may see.
        """
        key = self._get_config_key(config)
        if self._owner_denied(config):
            return 0  # outside the caller's reach: no-op
        # The thread's data goes with it, as on the SQLite and Postgres checkpointers;
        # a cached state left behind would also be the next run's starting point.
        found = self._states.pop(key, None) is not None
        found = self._state_cache.pop(key, None) is not None or found
        for ledger_key in [k for k in self._tool_results if k[0] == key]:
            del self._tool_results[ledger_key]
            found = True
        found = self._messages.pop(key, None) is not None or found
        found = self._threads.pop(key, None) is not None or found
        found = self._thread_owner.pop(key, None) is not None or found
        self._thread_tenant.pop(key, None)
        await self._clear_run_keys(config)
        logger.debug(f"Cleaned thread for key: {key}")
        return 1 if found else 0

    # -------------------------
    # Clean Resources
    # -------------------------
    async def aping(self) -> None:
        """Nothing to reach: the data lives in this process."""
        return

    async def arelease(self) -> bool:
        """
        Release resources asynchronously.

        Returns:
            bool: True if released.
        """
        self._states.clear()
        self._state_cache.clear()
        self._generic_cache.clear()
        self._messages.clear()
        self._threads.clear()
        self._thread_owner.clear()
        self._thread_tenant.clear()
        self._tool_results.clear()
        logger.info("Released all in-memory resources")
        return True
