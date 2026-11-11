import asyncio
import copy
import logging
import time
from collections import defaultdict
from datetime import timedelta
from itertools import islice
from typing import TYPE_CHECKING, Any, TypeVar

from tenxgraph.core.authz import owner_id
from tenxgraph.core.exceptions import ThreadNotFoundError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils.callable_utils import run_coroutine
from tenxgraph.utils.thread_info import ThreadInfo

from .base_checkpointer import (
    BaseCheckpointer,
    _reject_filters,
    _reject_tenant_options,
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
    Async-first design using asyncio locks for concurrent access.

    Args:
        None

    Attributes:
        _states (dict): Stores agent states by thread key.
        _state_cache (dict): Stores cached agent states by thread key.
        _messages (dict): Stores messages by thread key.
        _message_metadata (dict): Stores message metadata by thread key.
        _threads (dict): Stores thread info by thread key.
        _state_lock (asyncio.Lock): Lock for state operations.
        _messages_lock (asyncio.Lock): Lock for message operations.
        _threads_lock (asyncio.Lock): Lock for thread operations.
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
        Initialize all in-memory storage and locks.

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
        self._message_metadata: dict[str, dict[str, Any]] = {}

        # Thread storage
        self._threads: dict[str, dict[str, Any]] = {}

        # Thread ownership: config-key (thread_id) -> owner user_id. The first writer of a
        # thread owns it. Used to enforce owner-only access when the caller identity
        # (config["user"]) asks for it -- see _owner_denied / _claim.
        self._thread_owner: dict[str, str] = {}

        self.record_tool_result = record_tool_result
        # Tool execution ledger (idempotency): (thread, tool_call_id) -> (result, recorded at)
        self._tool_results: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}

        # Async locks for concurrent access
        self._state_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        self._messages_lock = asyncio.Lock()
        self._threads_lock = asyncio.Lock()
        self._tool_lock = asyncio.Lock()

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

    def _get_config_key(self, config: dict[str, Any]) -> str:
        """
        Generate a string key from config dict for storage indexing.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            str: Key for indexing storage.
        """
        # Sort keys for consistent hashing
        thread_id = config.get("thread_id", "")
        return str(thread_id)

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
            owner = self._thread_owner.setdefault(key, str(user_id))
        else:
            owner = self._thread_owner.get(key)
        if (
            self._isolation_enabled(config, user_id, write=True)
            and owner is not None
            and owner != str(user_id)
        ):
            raise ThreadNotFoundError(config.get("thread_id"))
        return owner

    def _owner_denied(self, config: dict[str, Any], *, write: bool = False) -> bool:
        """True when isolation is active and the caller may not see this thread.

        Reads and deletes follow the caller's reach; ``write=True`` checks the caller owns
        it. A thread with no recorded owner (brand new) is never denied: the caller becomes
        its owner on write, and reads simply find nothing.
        """
        user_id = owner_id(config)
        if not self._isolation_enabled(config, user_id, write=write):
            return False
        owner = self._thread_owner.get(self._get_config_key(config))
        return owner is not None and str(owner) != str(user_id)

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
        key = self._get_config_key(config)
        # Store copies: loaders merge new input into the state they get back, so sharing one
        # object would let a request that is later rejected (a paused thread) change what is
        # stored. Copied before the claim, so a state that cannot be copied claims nothing.
        stored, cached = copy.deepcopy(state), copy.deepcopy(state)
        async with self._state_lock:
            self._claim(config)
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
        async with self._state_lock:
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
        async with self._state_lock:
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
        async with self._state_lock:
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
        async with self._state_lock:
            cache = self._state_cache.get(key)
            logger.debug(f"Retrieved state cache for key: {key}, found: {cache is not None}")
            return copy.deepcopy(cache)

    async def aclear_state_cache(self, config: dict[str, Any]) -> bool:
        """Drop the cached state for a thread; the durable state is untouched."""
        if self._owner_denied(config):
            return True  # not the owner -> no-op, do not reveal existence
        key = self._get_config_key(config)
        async with self._state_lock:
            self._state_cache.pop(key, None)
        return True

    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        cache_key = f"{namespace}:{key}"
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        async with self._cache_lock:
            self._generic_cache[cache_key] = (value, expires_at)
        return value

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        cache_key = f"{namespace}:{key}"
        async with self._cache_lock:
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
        async with self._tool_lock:
            entry = self._tool_results.get(self._tool_key(config, tool_call_id))
            return entry[0] if entry is not None else None

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Record that a tool call completed, with its result."""
        async with self._tool_lock:
            self._claim(config)
            self._tool_results[self._tool_key(config, tool_call_id)] = (result, time.time())
            return True

    async def aclear_tool_results(self, config: dict[str, Any]) -> int:
        """Delete the thread's ledger records; returns how many went."""
        if self._owner_denied(config, write=True):
            return 0
        key = self._get_config_key(config)
        async with self._tool_lock:
            stale = [k for k in self._tool_results if k[0] == key]
            for ledger_key in stale:
                del self._tool_results[ledger_key]
        return len(stale)

    async def aprune_tool_results(self, older_than: timedelta) -> int:
        """Delete ledger records of every thread older than ``older_than``."""
        cutoff = time.time() - older_than.total_seconds()
        async with self._tool_lock:
            stale = [k for k, (_, at) in self._tool_results.items() if at < cutoff]
            for ledger_key in stale:
                del self._tool_results[ledger_key]
        return len(stale)

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        cache_key = f"{namespace}:{key}"
        async with self._cache_lock:
            return self._generic_cache.pop(cache_key, None)

    async def aclear_cache_value_if(self, namespace: str, key: str, expected: Any) -> bool:
        """Compare and delete under the cache lock, in one step."""
        cache_key = f"{namespace}:{key}"
        async with self._cache_lock:
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
        cache_key = f"{namespace}:{key}"
        now = time.time()
        async with self._cache_lock:
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
        async with self._cache_lock:
            keys = []
            for full_key in self._generic_cache:
                if full_key.startswith(ns_prefix):
                    key_part = full_key[len(ns_prefix) :]
                    if prefix is None or key_part.startswith(prefix):
                        keys.append(key_part)
            return keys

    # -------------------------
    # State methods Sync
    # -------------------------
    def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """Store state synchronously."""
        return run_coroutine(self.aput_state(config, state))

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state synchronously."""
        return run_coroutine(self.aget_state(config))

    def clear_state(self, config: dict[str, Any]) -> int:
        """Clear state synchronously."""
        return run_coroutine(self.aclear_state(config))

    def put_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        """Store state cache synchronously."""
        return run_coroutine(self.aput_state_cache(config, state))

    def get_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state cache synchronously."""
        return run_coroutine(self.aget_state_cache(config))

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
            metadata (dict, optional): Additional metadata.

        Returns:
            bool: True if stored.
        """
        key = self._get_config_key(config)
        async with self._messages_lock:
            self._claim(config)
            # Upsert by message id, as the SQL backends do: a run that re-sends messages it
            # already stored (a resume, the final save after a step checkpoint) must not
            # duplicate them.
            stored = self._messages[key]
            index = {str(m.message_id): i for i, m in enumerate(stored)}
            for msg in messages:
                at = index.get(str(msg.message_id))
                if at is None:
                    index[str(msg.message_id)] = len(stored)
                    stored.append(msg)
                else:
                    stored[at] = msg
            if metadata:
                self._message_metadata[key] = metadata
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
            IndexError: If message not found.
        """
        key = self._get_config_key(config)
        if self._owner_denied(config):
            raise IndexError(f"Message with ID {message_id} not found for config key: {key}")
        async with self._messages_lock:
            messages = self._messages.get(key, [])
            for msg in messages:
                if msg.message_id == message_id:
                    return msg
            raise IndexError(f"Message with ID {message_id} not found for config key: {key}")

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
        if self._owner_denied(config, write=True):
            return None
        key = self._get_config_key(config)
        async with self._messages_lock:
            for msg in self._messages.get(key, []):
                if msg.message_id == message_id:
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
        async with self._messages_lock:
            messages = self._messages.get(key, [])

            # Apply search filter if provided
            if search:
                # Simple string search in message content
                needle = search.lower()
                messages = [
                    msg
                    for msg in messages
                    if hasattr(msg, "content") and needle in str(msg.content).lower()
                ]

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
        async with self._messages_lock:
            if self._owner_denied(config):
                return 0
            messages = self._messages.get(key, [])
            for msg in messages:
                if msg.message_id == message_id:
                    messages.remove(msg)
                    logger.debug(f"Deleted message with ID {message_id} for key: {key}")
                    break
            else:
                return 0
        # The state (and its cache copy) carries the message too; a deleted message must not
        # come back on the next run or in GET state.
        async with self._state_lock:
            for store in (self._states, self._state_cache):
                if key in store:
                    self._strip_message_from_state(store[key], message_id)
        return 1

    # -------------------------
    # Message methods sync
    # -------------------------
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store messages synchronously."""
        return run_coroutine(self.aput_messages(config, messages, metadata))

    def get_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """Retrieve a specific message synchronously."""
        return run_coroutine(self.aget_message(config, message_id))

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """List messages synchronously with optional filtering."""
        return run_coroutine(self.alist_messages(config, search, offset, limit))

    def delete_message(self, config: dict[str, Any], message_id: str | int) -> int:
        """Delete a specific message synchronously."""
        return run_coroutine(self.adelete_message(config, message_id))

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
            bool: True if stored.

        Raises:
            ThreadNotFoundError: Another user owns the thread.
        """
        key = self._get_config_key(config)
        record = thread_info.model_dump()
        async with self._threads_lock:
            owner = self._claim(config)
            if owner is not None and str(record.get("user_id")) != owner:
                record["user_id"] = owner  # the stored owner, never one named by thread_info
            self._threads[key] = record
            logger.debug(f"Stored thread info for key: {key}")
            return True

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the owner ``user_id`` of ``thread_id``, or None if unknown.

        Threads are keyed by ``thread_id`` here, so the stored record's ``user_id``
        is the owner (``aput_thread`` stores the owner there). Note the in-memory registry
        is only populated by explicit ``aput_thread`` calls; a thread that only ever had
        state written is not recorded, so this returns None for it.
        """
        async with self._threads_lock:
            record = self._threads.get(str(thread_id))
        if not record:
            return None
        return record.get("user_id")

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
        async with self._threads_lock:
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
        async with self._threads_lock:
            threads: list[dict[str, Any]] = list(self._threads.values())

            # Owner-scope the listing when the identity asks for it.
            if isolate:
                threads = [t for t in threads if str(t.get("user_id")) == str(user_id)]

            # Apply search filter if provided
            if search:
                needle = search.lower()
                threads = [
                    thread
                    for thread in threads
                    if any(needle in str(value).lower() for value in thread.values())
                ]

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
        async with self._state_lock:
            found = self._states.pop(key, None) is not None
            found = self._state_cache.pop(key, None) is not None or found
        async with self._tool_lock:
            for ledger_key in [k for k in self._tool_results if k[0] == key]:
                del self._tool_results[ledger_key]
                found = True
        async with self._messages_lock:
            found = self._messages.pop(key, None) is not None or found
            found = self._message_metadata.pop(key, None) is not None or found
        async with self._threads_lock:
            found = self._threads.pop(key, None) is not None or found
            found = self._thread_owner.pop(key, None) is not None or found
        logger.debug(f"Cleaned thread for key: {key}")
        return 1 if found else 0

    # -------------------------
    # Thread methods sync
    # -------------------------
    def put_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> bool:
        """Store thread info synchronously."""
        return run_coroutine(self.aput_thread(config, thread_info))

    def get_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        """Retrieve thread info synchronously."""
        return run_coroutine(self.aget_thread(config))

    def list_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ThreadInfo]:
        """List all threads synchronously with optional filtering."""
        return run_coroutine(self.alist_threads(config, search, offset, limit, filters))

    def clean_thread(self, config: dict[str, Any]) -> int:
        """Clean/delete thread synchronously."""
        return run_coroutine(self.aclean_thread(config))

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
        async with (
            self._state_lock,
            self._cache_lock,
            self._messages_lock,
            self._threads_lock,
            self._tool_lock,
        ):
            self._states.clear()
            self._state_cache.clear()
            self._generic_cache.clear()
            self._messages.clear()
            self._message_metadata.clear()
            self._threads.clear()
            self._thread_owner.clear()
            self._tool_results.clear()
            logger.info("Released all in-memory resources")
            return True

    def release(self) -> bool:
        """Release resources synchronously."""
        return run_coroutine(self.arelease())
