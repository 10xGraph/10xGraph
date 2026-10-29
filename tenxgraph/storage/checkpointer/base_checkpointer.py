import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar, get_args

from tenxgraph.core.exceptions import StateFormatVersionError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import run_coroutine
from tenxgraph.utils.thread_info import ThreadInfo


logger = logging.getLogger("tenxgraph.checkpointer")

# Checkpointer classes already warned about a missing tool ledger (warn once per class).
_LEDGER_WARNED: set[str] = set()

if TYPE_CHECKING:
    from tenxgraph.core.state import AgentState, Message


StateT = TypeVar("StateT", bound="AgentState")

# Namespace for out-of-band stop requests in the generic cache.
#
# The stop flag deliberately lives in its OWN cache entry rather than inside the
# cached state blob. A running graph rewrites the state cache after every node
# (``call_realtime_sync``), and those in-run writes carry the same durable
# version as the entry a concurrent ``stop()`` just wrote -- so a flag stored in
# the state blob is legally overwritten by the loop's own flag-less copy and the
# stop is silently lost. Nothing in the run loop ever writes this key, so a stop
# recorded here cannot be clobbered by the run it is trying to stop.
STOP_REQUEST_NAMESPACE = "stop_request"

# Largest page the list methods (``alist_messages`` / ``alist_threads``) return. An explicit
# ``limit`` above it is lowered to it, and a negative ``limit`` or ``offset`` raises
# ``ValueError``. ``limit=None`` or ``0`` keeps meaning "no limit given" for in-process callers
# that read a whole thread; the API always sends a bounded page.
MAX_LIST_LIMIT = 1000


def validate_page(offset: int | None, limit: int | None) -> tuple[int, int | None]:
    """Check ``offset`` / ``limit`` and return ``(offset, limit)``; ``limit`` is None for "all"."""
    if offset is not None and offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    return offset or 0, (min(limit, MAX_LIST_LIMIT) if limit else None)


# How long an unconsumed stop request stays live. A stop is meant to be observed
# within a node or two; this bound only keeps an abandoned flag from lingering
# forever if the run dies before consuming it.
STOP_REQUEST_TTL_SECONDS = 3600

# Per-thread marker holding the id of the run that is currently active. A stop request records
# the run it targets so a new run on the same thread cannot erase a stop meant for a run that
# is still going. Like the stop flag it lives in the generic cache, so it works on every
# backend (memory, SQLite, Redis) without a schema change, and it expires with the same bound.
ACTIVE_RUN_NAMESPACE = "active_run"

# The marker is a heartbeat: it lives this long and a live run refreshes it every
# ``ACTIVE_RUN_HEARTBEAT_SECONDS``. A run that dies without cleanup (SIGKILL, OOM) therefore
# leaves a marker that expires within a minute, not after the hour a stop flag may linger.
ACTIVE_RUN_TTL_SECONDS = 60
ACTIVE_RUN_HEARTBEAT_SECONDS = 20

# Key holding the storage header inside a persisted state row. ``format`` versions
# the row layout itself; ``class`` is the state class name, kept only for
# diagnostics. Neither is ever used to import code.
STATE_META_KEY = "__state_meta__"
STATE_FORMAT_VERSION = 1

# Rows written before the state class was bound by the graph carried a module path
# under this key. It is dropped on read and never imported.
_LEGACY_CLASS_PATH_KEY = "__class_path__"


class BaseCheckpointer[StateT: AgentState](ABC):
    """
    Abstract base class for checkpointing agent state, messages, and threads.

    This class defines the contract for all checkpointer implementations, supporting both
    async and sync methods.
    Subclasses should implement async methods for optimal performance.
    Sync methods are provided for compatibility.

    Usage:
        - Async-first design: subclasses should implement `async def` methods.
        - If a subclass provides only a sync `def`, it will be executed in a worker thread
            automatically using `asyncio.run`.
        - Callers always use the async APIs (`await cp.put_state(...)`, etc.).

    Type Args:
        StateT: Type of agent state (must inherit from AgentState).
    """

    # The class persisted rows are rebuilt into. ``StateGraph.compile()`` binds the
    # graph's own state class; a checkpointer used on its own restores plain
    # ``AgentState``. A class attribute, so subclasses that never call
    # ``super().__init__()`` still get the default.
    _state_type: type[AgentState] = AgentState
    _state_type_bound: bool = False

    @property
    def state_type(self) -> type[AgentState]:
        """The state class persisted rows are restored into."""
        return self._state_type

    def bind_state_type(self, state_type: type[AgentState]) -> None:
        """Set the state class this checkpointer restores rows into.

        Called by ``StateGraph.compile()``. Binding the same class again is a no-op.

        Raises:
            TypeError: If ``state_type`` is not an ``AgentState`` subclass.
            ValueError: If a different class is already bound, or if the instance
                was created with an explicit type (``PgCheckpointer[MyState](...)``)
                that ``state_type`` does not subclass. One checkpointer instance
                serves one state class; restoring another graph's rows into the
                wrong class would silently drop its custom fields.
        """
        if not (isinstance(state_type, type) and issubclass(state_type, AgentState)):
            raise TypeError(f"State type must be an AgentState subclass, got {state_type!r}")
        # ``__orig_class__`` exists only on explicitly parametrized instances, and
        # a TypeVar argument (not a class) carries no constraint worth checking.
        declared = get_args(getattr(self, "__orig_class__", None))
        if declared and isinstance(declared[0], type) and not issubclass(state_type, declared[0]):
            raise ValueError(
                f"{type(self).__name__} was declared for state class "
                f"'{declared[0].__name__}', but the graph uses '{state_type.__name__}'."
            )
        if self._state_type_bound and self._state_type is not state_type:
            raise ValueError(
                f"{type(self).__name__} is already bound to state class "
                f"'{self._state_type.__name__}', cannot rebind it to "
                f"'{state_type.__name__}'. Use a separate checkpointer instance per "
                "state class."
            )
        self._state_type = state_type
        self._state_type_bound = True

    def _encode_state(self, state: AgentState) -> dict[str, Any]:
        """Build the JSON-safe row persisted for a state: its data plus a header."""
        data = state.model_dump(mode="json")
        data[STATE_META_KEY] = {
            "format": STATE_FORMAT_VERSION,
            "class": type(state).__name__,
        }
        return data

    @staticmethod
    def _check_state_format(data: dict[str, Any], thread_id: Any = None) -> None:
        """Refuse a row written in a newer format than this version reads."""
        found = (data.get(STATE_META_KEY) or {}).get("format")
        if isinstance(found, int) and found > STATE_FORMAT_VERSION:
            raise StateFormatVersionError(found, STATE_FORMAT_VERSION, thread_id)

    def _decode_state(self, data: dict[str, Any], thread_id: Any = None) -> StateT:
        """Rebuild a state from a persisted row using the bound state class."""
        data = dict(data)
        self._check_state_format(data, thread_id)
        meta = data.pop(STATE_META_KEY, None) or {}
        legacy_path = data.pop(_LEGACY_CLASS_PATH_KEY, None)
        stored_name = meta.get("class") or (
            legacy_path.rsplit(".", 1)[-1] if isinstance(legacy_path, str) else None
        )
        expected = self._state_type.__name__
        if stored_name and stored_name != expected:
            logger.warning(
                "Stored state was written as '%s' but is being restored as '%s'. "
                "Fields not defined on '%s' are dropped.",
                stored_name,
                expected,
                expected,
            )
        return self._state_type.model_validate(data)  # type: ignore[return-value]

    @staticmethod
    def _strip_message_from_state(state: AgentState, message_id: str | int) -> bool:
        """Remove a message from ``state.context`` in place. True when something was removed.

        Deleting a message must also remove it from the state the next run starts from,
        otherwise the "deleted" message is still sent to the model and returned by GET state.
        """
        context = getattr(state, "context", None)
        if not context:
            return False
        kept = [m for m in context if str(getattr(m, "message_id", None)) != str(message_id)]
        if len(kept) == len(context):
            return False
        state.context = kept
        return True

    @staticmethod
    def _strip_message_from_payload(payload: dict[str, Any], message_id: str | int) -> bool:
        """Same as :meth:`_strip_message_from_state` for a persisted state payload (a dict)."""
        context = payload.get("context")
        if not isinstance(context, list) or not context:
            return False
        kept = [
            m
            for m in context
            if not (isinstance(m, dict) and str(m.get("message_id")) == str(message_id))
        ]
        if len(kept) == len(context):
            return False
        payload["context"] = kept
        return True

    ###########################
    #### SETUP ################
    ###########################
    def setup(self) -> Any:
        """
        Synchronous setup method for checkpointer.

        Returns:
            Any: Implementation-defined setup result.
        """
        return run_coroutine(self.asetup())

    @abstractmethod
    async def asetup(self) -> Any:
        """
        Asynchronous setup method for checkpointer.

        Returns:
            Any: Implementation-defined setup result.
        """
        raise NotImplementedError

    # -------------------------
    # State methods Async
    # -------------------------
    @abstractmethod
    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.
        """
        raise NotImplementedError

    @abstractmethod
    async def aclear_state(self, config: dict[str, Any]) -> Any:
        """
        Clear agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any: Implementation-defined result.
        """
        raise NotImplementedError

    async def aput_checkpoint(
        self,
        config: dict[str, Any],
        state: StateT,
        messages: "list[Message] | None" = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateT:
        """
        Persist a state and its messages as one durable checkpoint.

        This default is only a fallback for backends without a shared
        transaction (e.g. in-memory): it writes state then messages sequentially,
        exactly as callers did before. Backends that can commit both together
        (e.g. Postgres) override this so the two writes are atomic and a crash
        cannot leave state advanced without the messages that justify it.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.
            messages (list[Message], optional): Messages to persist with the state.
            metadata (dict, optional): Additional message metadata.

        Returns:
            StateT: The stored state object.
        """
        await self.aput_state(config, state)
        if messages:
            await self.aput_messages(config, messages, metadata)
        return state

    @abstractmethod
    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """
        Store agent state in cache asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state from cache asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Cached state or None.
        """
        raise NotImplementedError

    async def aclear_state_cache(self, config: dict[str, Any]) -> bool:
        """
        Drop the cached state for a thread, leaving the durable state untouched.

        The next load then reads the durable store. Use it to repair a thread whose cache
        cannot be used. Backends with a state cache override this.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            bool: True once the cache entry is gone (also when there was none).
        """
        raise NotImplementedError

    # -------------------------
    # State methods Sync
    # -------------------------
    def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store agent state synchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.
        """
        return run_coroutine(self.aput_state(config, state))

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.
        """
        return run_coroutine(self.aget_state(config))

    def clear_state(self, config: dict[str, Any]) -> Any:
        """
        Clear agent state synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any: Implementation-defined result.
        """
        return run_coroutine(self.aclear_state(config))

    def put_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """
        Store agent state in cache synchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aput_state_cache(config, state))

    def get_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state from cache synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Cached state or None.
        """
        return run_coroutine(self.aget_state_cache(config))

    def clear_state_cache(self, config: dict[str, Any]) -> bool:
        """
        Drop the cached state for a thread synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            bool: True once the cache entry is gone.
        """
        return run_coroutine(self.aclear_state_cache(config))

    # -------------------------
    # Generic cache methods
    # -------------------------
    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        """Store a small JSON-serializable cache value.

        This is intentionally optional so existing checkpointers keep working
        even if they do not offer a shared cache backend.
        """
        return None

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        """Retrieve a cached value previously stored via ``aput_cache_value``."""
        return None

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        """Delete a previously cached value."""
        return None

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        """List all cache keys for a namespace.

        This is intentionally optional — default returns an empty list.
        Subclasses should override if they support key enumeration.

        Args:
            namespace: Cache namespace (e.g. "media:signed-url").
            prefix: Optional prefix to filter keys.

        Returns:
            List of cache key strings.
        """
        return []

    def put_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        """Synchronously store a small cache value."""
        return run_coroutine(self.aput_cache_value(namespace, key, value, ttl_seconds))

    def get_cache_value(self, namespace: str, key: str) -> Any | None:
        """Synchronously retrieve a small cache value."""
        return run_coroutine(self.aget_cache_value(namespace, key))

    def clear_cache_value(self, namespace: str, key: str) -> Any | None:
        """Synchronously delete a small cache value."""
        return run_coroutine(self.aclear_cache_value(namespace, key))

    # -------------------------
    # Stop requests
    # -------------------------
    def _stop_request_key(self, config: dict[str, Any]) -> str:
        """Build the per-thread stop-request key.

        Scoped by user as well as thread so one user cannot stop another user's
        run by guessing a thread_id, matching how state and messages are scoped.
        """
        return f"{config.get('thread_id')}:{config.get('user_id')}"

    async def aget_active_run(self, config: dict[str, Any]) -> str | None:
        """Return the id of the run currently marked active on this thread, if any."""
        value = await self.aget_cache_value(ACTIVE_RUN_NAMESPACE, self._stop_request_key(config))
        return value if isinstance(value, str) and value else None

    async def abegin_run(self, config: dict[str, Any]) -> None:
        """Register ``config["run_id"]`` as the active run and drop any stale stop request.

        A stop request belongs to the run it was made for. At run start only a stale flag is
        cleared: one that targeted no run, or a run that is no longer the active one. A flag
        that targets a run still marked active is kept, so a second run on the same thread
        cannot erase a stop meant for the first. Runs that finish call :meth:`aend_run`.
        """
        key = self._stop_request_key(config)
        flag = await self.aget_cache_value(STOP_REQUEST_NAMESPACE, key)
        if flag:
            target = self._stop_target(flag)
            if target is None or target != await self.aget_active_run(config):
                await self.aclear_cache_value(STOP_REQUEST_NAMESPACE, key)
        run_id = config.get("run_id")
        if run_id:
            await self.aput_cache_value(
                ACTIVE_RUN_NAMESPACE, key, str(run_id), ACTIVE_RUN_TTL_SECONDS
            )

    async def arefresh_run(self, config: dict[str, Any]) -> None:
        """Extend the active-run marker's life if it is still ``config["run_id"]``'s.

        Called every ``ACTIVE_RUN_HEARTBEAT_SECONDS`` while a run is alive. A marker that now
        belongs to another run, or is gone, is left alone.
        """
        run_id = config.get("run_id")
        if not run_id:
            return
        if await self.aget_active_run(config) == str(run_id):
            await self.aput_cache_value(
                ACTIVE_RUN_NAMESPACE,
                self._stop_request_key(config),
                str(run_id),
                ACTIVE_RUN_TTL_SECONDS,
            )

    async def aend_run(self, config: dict[str, Any]) -> None:
        """Mark ``config["run_id"]`` as finished.

        Clears the active-run marker if it is still this run's (a newer run may have
        replaced it) and drops a stop request that targeted this run.
        """
        run_id = config.get("run_id")
        if not run_id:
            return
        key = self._stop_request_key(config)
        if await self.aget_active_run(config) == str(run_id):
            await self.aclear_cache_value(ACTIVE_RUN_NAMESPACE, key)
        flag = await self.aget_cache_value(STOP_REQUEST_NAMESPACE, key)
        if flag and self._stop_target(flag) == str(run_id):
            await self.aclear_cache_value(STOP_REQUEST_NAMESPACE, key)

    @staticmethod
    def _stop_target(flag: Any) -> str | None:
        """Run id a stored stop flag targets (``None`` for a legacy or untargeted flag)."""
        if isinstance(flag, dict):
            target = flag.get("run_id")
            return str(target) if target else None
        return None

    async def arequest_stop(self, config: dict[str, Any]) -> Any | None:
        """Record an out-of-band stop request for a thread.

        Stored under :data:`STOP_REQUEST_NAMESPACE` in the generic cache rather
        than in the state blob, so the running graph cannot overwrite it with its
        own state writes. The flag records the run that is active on the thread
        (``None`` when none is known), so only that run honours it. Backends
        without a shared cache inherit the no-op generic-cache defaults, in which
        case this degrades to doing nothing rather than misbehaving.
        """
        return await self.aput_cache_value(
            STOP_REQUEST_NAMESPACE,
            self._stop_request_key(config),
            {"run_id": await self.aget_active_run(config)},
            STOP_REQUEST_TTL_SECONDS,
        )

    async def ais_stop_requested(self, config: dict[str, Any]) -> bool:
        """Return True when an unconsumed stop request exists for this run.

        A request that targets a different run than ``config["run_id"]`` is not reported.
        A config without a ``run_id``, or an untargeted request, matches any run.
        """
        value = await self.aget_cache_value(
            STOP_REQUEST_NAMESPACE,
            self._stop_request_key(config),
        )
        if not value:
            return False
        target = self._stop_target(value)
        run_id = config.get("run_id")
        return target is None or not run_id or target == str(run_id)

    async def aclear_stop_request(self, config: dict[str, Any]) -> Any | None:
        """Drop any pending stop request for a thread.

        Called once a stop has been acted on. A new run does not call this; it uses
        :meth:`abegin_run`, which keeps a request aimed at a run that is still active.
        """
        return await self.aclear_cache_value(
            STOP_REQUEST_NAMESPACE,
            self._stop_request_key(config),
        )

    def request_stop(self, config: dict[str, Any]) -> Any | None:
        """Synchronously record a stop request."""
        return run_coroutine(self.arequest_stop(config))

    def is_stop_requested(self, config: dict[str, Any]) -> bool:
        """Synchronously check for a pending stop request."""
        return run_coroutine(self.ais_stop_requested(config))

    def clear_stop_request(self, config: dict[str, Any]) -> Any | None:
        """Synchronously drop a pending stop request."""
        return run_coroutine(self.aclear_stop_request(config))

    # -------------------------
    # Tool execution ledger (idempotency)
    # -------------------------
    def _warn_ledger_unsupported(self) -> None:
        """Log once per checkpointer class that tool calls are not de-duplicated."""
        name = type(self).__name__
        if name in _LEDGER_WARNED:
            return
        _LEDGER_WARNED.add(name)
        logger.warning(
            "%s does not implement the tool execution ledger. A run killed in the middle "
            "of a node will fire that node's tools again on resume. Use a checkpointer "
            "with ledger support (InMemoryCheckpointer, SqliteCheckpointer, "
            "PgCheckpointer) or implement aget_tool_result/aput_tool_result.",
            name,
        )

    async def aget_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """Return a previously recorded result for this tool call, if any.

        The execution loop persists ``current_node`` *before* running a node and
        only advances it after the node completes. So a process killed mid-node
        (OOM, SIGKILL on deploy) resumes by re-running that node from scratch --
        and re-fires every tool in it. For a non-idempotent tool that means the
        card is charged twice.

        This ledger closes that: a tool call that already completed is recorded
        durably against its ``tool_call_id``, and on replay its recorded result is
        returned instead of the tool being called again.

        Returning None means "no record" -- the tool should run. Backends that do
        not implement the ledger inherit this default, which simply disables
        idempotency rather than breaking (behaviour matches the old at-least-once
        semantics), and a warning is logged once per checkpointer class.
        """
        self._warn_ledger_unsupported()
        return None

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Durably record that a tool call completed, with its result.

        Written immediately after the tool returns, so a crash later in the same
        node cannot cause the tool to be re-fired on resume.
        """
        self._warn_ledger_unsupported()
        return None

    # -------------------------
    # Message methods async
    # -------------------------
    @abstractmethod
    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages asynchronously.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Message: Retrieved message object.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        """
        Delete a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    async def aupdate_message_metadata(
        self,
        config: dict[str, Any],
        message_id: str | int,
        metadata: dict[str, Any],
    ) -> Message | None:
        """
        Shallow-merge ``metadata`` into a stored message's metadata.

        A key in ``metadata`` replaces the same key already stored. Applies the same
        owner isolation as :meth:`aget_message`.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.
            metadata (dict): Keys to set on the message metadata.

        Returns:
            Message | None: The updated message, or None if it does not exist in the
            thread (or the caller may not access it).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support updating message metadata"
        )

    # -------------------------
    # Message methods sync
    # -------------------------
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages synchronously.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: Implementation-defined result.
        """
        return run_coroutine(self.aput_messages(config, messages, metadata))

    def get_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a specific message synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Message: Retrieved message object.
        """
        return run_coroutine(self.aget_message(config, message_id))

    def update_message_metadata(
        self,
        config: dict[str, Any],
        message_id: str | int,
        metadata: dict[str, Any],
    ) -> Message | None:
        """
        Shallow-merge ``metadata`` into a stored message's metadata synchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.
            metadata (dict): Keys to set on the message metadata.

        Returns:
            Message | None: The updated message, or None if it does not exist.
        """
        return run_coroutine(self.aupdate_message_metadata(config, message_id, metadata))

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages synchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.
        """
        return run_coroutine(self.alist_messages(config, search, offset, limit))

    def delete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        """
        Delete a specific message synchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.adelete_message(config, message_id))

    # -------------------------
    # Thread methods async
    # -------------------------
    @abstractmethod
    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> Any | None:
        """
        Store thread info asynchronously.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads asynchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[ThreadInfo]: List of thread information objects.
        """
        raise NotImplementedError

    @abstractmethod
    async def aclean_thread(self, config: dict[str, Any]) -> Any | None:
        """
        Clean/delete thread asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the ``user_id`` that owns ``thread_id``, regardless of caller.

        Unlike :meth:`aget_thread` (which is owner-scoped), this resolves ownership
        globally so an authorization layer can decide whether a *different* user may
        act on a thread. It exists to answer three states:

        - returns the owner's ``user_id`` -> the thread exists and is owned by them;
        - returns ``None`` -> the thread does not exist yet (a brand-new session);
        - raises :class:`NotImplementedError` -> this backend cannot resolve ownership.

        The default raises so a backend cannot silently report "no owner" for every
        thread (which would defeat owner-based authorization). Concrete backends that
        persist a thread registry override this.
        """
        raise NotImplementedError

    def _isolation_enabled(self, config: dict[str, Any] | None, user_id: Any) -> bool:
        """Whether this operation must be scoped to ``user_id``.

        Driven by the trusted ``config["authz"]`` policy (see :mod:`tenxgraph.core.authz`):

        - ``scope == "owner"`` -> isolate (when a ``user_id`` is present to scope by);
        - ``scope == "none"``  -> do not isolate (allow_all / single-user);
        - no policy (e.g. a direct SDK call with no API in front) -> fall back to this
          backend's own ``enforce_user_isolation`` setting (default off for backends that
          do not define one).
        """
        from tenxgraph.core.authz import SCOPE_NONE, SCOPE_OWNER, isolation_scope

        scope = isolation_scope(config)
        if scope == SCOPE_OWNER:
            return bool(user_id)
        if scope == SCOPE_NONE:
            return False
        return bool(getattr(self, "enforce_user_isolation", False) and user_id)

    # -------------------------
    # Thread methods sync
    # -------------------------
    def put_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> Any | None:
        """
        Store thread info synchronously.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aput_thread(config, thread_info))

    def get_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        """
        Retrieve thread info synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.
        """
        return run_coroutine(self.aget_thread(config))

    def list_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads synchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[ThreadInfo]: List of thread information objects.
        """
        return run_coroutine(self.alist_threads(config, search, offset, limit))

    def clean_thread(self, config: dict[str, Any]) -> Any | None:
        """
        Clean/delete thread synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aclean_thread(config))

    # -------------------------
    # Clean Resources
    # -------------------------
    def release(self) -> Any | None:
        """
        Release resources synchronously.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.arelease())

    @abstractmethod
    async def arelease(self) -> Any | None:
        """
        Release resources asynchronously.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError
