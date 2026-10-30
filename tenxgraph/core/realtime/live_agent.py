"""LiveAgent -- the realtime (audio-to-audio) node and root session controller.

``LiveAgent`` subclasses :class:`BaseAgent` and reuses the skills/memory builder mixins,
but it deliberately **excludes** the text turn loop (``AgentExecutionMixin``): realtime
inverts control (the provider owns the turn loop), so it writes its own duplex loop.

It is entered by ``CompiledGraph.arealtime`` (Phase 3) via :meth:`arun`, which:

1. opens one provider socket (the spine, held for the whole session),
2. runs a pump task (queue -> provider) concurrently with a receive loop,
3. dispatches tool calls through the existing :class:`ToolNode` (callbacks + publisher
   events fire inside ToolNode, so transparency is identical to text mode),
4. persists finished transcripts as ``Message``s (no audio at rest), and
5. transparently reconnects on ``go_away``/drop using the cached resumption handle.

Calling the agent through the turn-based engine (``execute``) raises -- the forcing rule.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Literal

from tenxgraph.core.graph.agent_internal.memory import AgentMemoryMixin
from tenxgraph.core.graph.agent_internal.skills import AgentSkillsMixin
from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.graph.utils.tool_runtime import ToolRuntimeMixin
from tenxgraph.core.llm import detect_provider
from tenxgraph.core.realtime.base import (
    ErrorEvent,
    InputTranscriptEvent,
    OutputTranscriptEvent,
    RealtimeClient,
    RealtimeConfig,
    ToolCallEvent,
    ToolResultEvent,
)
from tenxgraph.core.realtime.providers.gemini_live import GeminiLiveClient
from tenxgraph.core.state import AgentState, Message, TextBlock, add_messages
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.utils import CallbackManager
from tenxgraph.utils.callbacks import GraphLifecycleContext


# Event kinds that constitute model/user turn content. A turn starts on the first of these
# after a turn boundary and ends on turn_complete/interrupted; control frames (session_update,
# go_away, error) never open a turn.
_TURN_CONTENT_TYPES = frozenset(
    {"audio_delta", "input_transcript", "output_transcript", "tool_call", "tool_result"}
)


if TYPE_CHECKING:
    from tenxgraph.core.realtime.base import RealtimeEvent
    from tenxgraph.core.realtime.queue import LiveInputQueue
    from tenxgraph.core.state import BaseContextManager
    from tenxgraph.storage.checkpointer import BaseCheckpointer

logger = logging.getLogger(__name__)


def _narrowed_tags(base: list[str] | None, requested: Any) -> list[str] | None:
    """The requested tags that ``base`` allows, or ``None`` when there are none to apply."""
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, list | tuple | set):
        return None
    tags = [str(tag) for tag in requested if base is None or tag in base]
    return tags or None


def _advertised_tool_names(tools: list[Any] | None) -> frozenset[str]:
    """Tool names in a session's ``tools``: OpenAI-style dicts, or provider-native objects."""
    names: set[str] = set()

    def field(item: Any, key: str) -> Any:
        return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

    for tool in tools or []:
        function = field(tool, "function")
        for entry in [function, tool, *(field(tool, "function_declarations") or [])]:
            name = field(entry, "name") if entry is not None else None
            if isinstance(name, str) and name:
                names.add(name)
    return frozenset(names)


class _LedgerAccess(ToolRuntimeMixin):
    """Borrows the text path's ledger read/write helpers (they need only ``name``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class LiveAgent(AgentSkillsMixin, AgentMemoryMixin, BaseAgent):
    """Realtime audio agent node. Run via :meth:`arun` (or ``CompiledGraph.arealtime``)."""

    def __init__(
        self,
        model: str,
        *,
        realtime_config: RealtimeConfig | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        tool_node: str | ToolNode | None = None,
        skills: Any | None = None,
        memory: Any | None = None,
        realtime_client_factory: Callable[[], RealtimeClient] | None = None,
        **kwargs: Any,
    ) -> None:
        api_key: str | None = kwargs.pop("api_key", None)
        use_vertex_ai: bool = kwargs.pop("use_vertex_ai", False)
        project: str | None = kwargs.pop("project", None)
        location: str | None = kwargs.pop("location", None)

        provider = detect_provider(model, use_vertex_ai)
        if provider != "google":
            raise ValueError(
                "LiveAgent v1 supports only Gemini Live (google provider); "
                f"resolved provider '{provider}' for model '{model}'."
            )

        super().__init__(
            model=model,
            system_prompt=system_prompt or [],
            tool_node=tool_node,
            **kwargs,
        )

        self.provider = "google"
        self.use_vertex_ai = use_vertex_ai
        self.realtime_config = realtime_config or RealtimeConfig(model=model)

        # Tool wiring (we don't use AgentExecutionMixin._setup_tools).
        self._tool_node: ToolNode | None = tool_node if isinstance(tool_node, ToolNode) else None
        self.tool_node_name: str | None = tool_node if isinstance(tool_node, str) else None

        # One client per *connection*; the factory lets reconnects get a fresh socket
        # and lets tests inject a fake provider.
        self._client_factory: Callable[[], RealtimeClient] = realtime_client_factory or (
            lambda: GeminiLiveClient(
                api_key=api_key,
                use_vertex_ai=use_vertex_ai,
                project=project,
                location=location,
            )
        )
        self._active_client: RealtimeClient | None = None
        self._resume_handle: str | None = None
        # Serializes upstream sends against reconnect (close+connect) so the pump never
        # sends on a socket being torn down, and always picks up the reconnected client.
        self._send_lock = asyncio.Lock()
        # One arun() at a time per instance: the fields above are per-session state.
        # Concurrent sessions each get their own copy from new_session().
        self._session_active = False
        self._session_id = uuid.uuid4().hex
        self._session_checkpointer: BaseCheckpointer | None = None

        # Per-session transcript accumulators (provider streams partial chunks; we flush
        # the concatenation on the finished marker). Reset at the start of each arun().
        self._input_transcript_buf = ""
        self._output_transcript_buf = ""

        # Names of the tools advertised to the model this session; a tool call for any
        # other name is refused. Set in arun() once the session's tools are resolved.
        self._session_tool_names: frozenset[str] = frozenset()

        # Error-driven reconnect backoff (go_away reconnects are immediate; only transient
        # drops back off). Seeded from RealtimeConfig.reconnect; kept as instance attributes
        # so tests can shrink them without rebuilding the config.
        rc = self.realtime_config.reconnect
        self._reconnect_base_delay = rc.base_delay
        self._reconnect_max_delay = rc.max_delay
        self._reconnect_max_attempts = rc.max_attempts

        # Builder mixins (no-op when their config is None).
        self._setup_memory(memory)
        self._setup_skills(skills)

    # ------------------------------------------------------------------ #
    # Forcing rule: a live agent is not a turn-based node.
    # ------------------------------------------------------------------ #
    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
        *,
        request_blocks: list[str] | None = None,
    ) -> Any:
        raise RuntimeError(
            "LiveAgent runs via CompiledGraph.arealtime(); it is not a turn-based node. "
            "Use .arealtime() (or the AudioAgent prebuilt), not invoke/stream."
        )

    async def _call_llm(
        self, messages: list[dict[str, Any]], tools: list | None = None, **kwargs: Any
    ) -> Any:
        # Realtime never makes a discrete turn-based LLM call; the provider owns the loop.
        raise RuntimeError(
            "LiveAgent has no discrete LLM call; audio turns are driven by the realtime socket."
        )

    def _resolve_tool_node(self) -> ToolNode | None:
        return self._tool_node

    def new_session(self) -> LiveAgent:
        """Return a copy of this agent with fresh per-session state.

        Configuration (model, prompt, tools, client factory) is shared; the client, lock,
        resume handle, transcript buffers and busy flag are new. ``CompiledGraph.arealtime``
        runs every connection on its own copy so concurrent sessions cannot cross-wire.
        """
        session = copy.copy(self)
        session._active_client = None
        session._resume_handle = None
        session._send_lock = asyncio.Lock()
        session._session_active = False
        session._session_id = uuid.uuid4().hex
        session._input_transcript_buf = ""
        session._output_transcript_buf = ""
        session._session_tool_names = frozenset()
        session._session_checkpointer = None
        return session

    # ------------------------------------------------------------------ #
    # The duplex realtime loop.
    # ------------------------------------------------------------------ #
    async def arun(  # noqa: PLR0912, PLR0915
        self,
        input_queue: LiveInputQueue,
        config: dict[str, Any],
        state: AgentState | None = None,
        *,
        checkpointer: BaseCheckpointer | None = None,
        callback_manager: CallbackManager | None = None,
        context_manager: BaseContextManager | None = None,
    ) -> AsyncIterator[RealtimeEvent]:
        """Open the session and yield normalized events until the queue/session closes."""
        if self._session_active:
            raise RuntimeError(
                "This LiveAgent already runs a session; use new_session() (arealtime does) "
                "for concurrent sessions."
            )
        self._session_active = True
        pump_task: asyncio.Task[None] | None = None
        connected = False  # nothing is saved for a session that never reached the provider
        state = state if state is not None else AgentState()
        try:
            if callback_manager is None:
                callback_manager = CallbackManager()
            self._input_transcript_buf = ""
            self._output_transcript_buf = ""
            rt = self._session_realtime_config(config)
            rt = await self._resolve_session_tools(rt)
            self._session_tool_names = _advertised_tool_names(rt.tools)
            rt = await self._resolve_session_system_instruction(rt, state, config)

            handle = await self._load_resume_handle(config, checkpointer)
            client = self._client_factory()
            # Set before connect so the cleanup below closes it even if connect fails midway.
            self._active_client = client
            self._session_checkpointer = checkpointer
            await client.connect(rt, resume_handle=handle)
            connected = True
            # Only reseed when the provider did NOT restore context from a handle; otherwise
            # the model would receive the whole conversation twice (handle restore + reseed).
            await self._maybe_reseed(
                config, checkpointer, context_manager, resumed=handle is not None
            )

            # Session start mirrors a graph run: the LIVE node *is* the graph, so
            # on_graph_start fires once here (before any turn) and on_graph_end once at end.
            state = await self._fire_graph_start(callback_manager, config, state)

            # Closing the input queue ends the session: the pump sets this when it drains the
            # close sentinel, and the receive loop stops once the provider goes idle.
            stop_event = asyncio.Event()
            pump_task = asyncio.create_task(self._pump(input_queue, stop_event))
            attempts = 0  # consecutive error-driven reconnect attempts (reset on healthy receive)
            turn_index = 0  # 1-based count of turns started; doubles as on_graph_end total_steps
            turn_active = False  # a turn is open (content seen, no turn_complete/interrupt yet)
            while True:
                reconnect = False
                forced = False  # go_away: reconnect even after input closed, to finish the turn
                received_any = False
                try:
                    async for event in self._receive_until_stop(self._active_client, stop_event):
                        received_any = True
                        if not turn_active and event.type in _TURN_CONTENT_TYPES:
                            turn_index += 1
                            turn_active = True
                            state = await self._fire_turn_start(
                                callback_manager, config, state, turn_index
                            )
                        for out in await self._handle_event(
                            event, config, state, checkpointer, callback_manager
                        ):
                            yield out
                        if turn_active and event.type in ("turn_complete", "interrupted"):
                            state = await self._fire_turn_end(
                                callback_manager, config, state, turn_index
                            )
                            turn_active = False
                        if event.type == "go_away":
                            reconnect = True
                            forced = True
                            break
                        if event.type == "error" and getattr(event, "fatal", False):
                            # break (not return) so on_graph_end still fires for the session.
                            reconnect = False
                            break
                except Exception:
                    # Transient drop: only resume if input is still open (avoid an
                    # infinite reconnect storm once the session is shutting down).
                    logger.warning("realtime receive loop error; attempting resume", exc_info=True)
                    reconnect = True

                # A turn that produced events means the connection is healthy again.
                if received_any:
                    attempts = 0

                resumable = reconnect and rt.session_resumption
                if not (resumable and (forced or not stop_event.is_set())):
                    break

                attempts, fatal = await self._attempt_reconnect(rt, forced, attempts)
                if fatal is not None:
                    yield fatal
                    break

            # Balance a turn cut off by session end (no turn_complete arrived), then close out.
            if turn_active:
                state = await self._fire_turn_end(callback_manager, config, state, turn_index)
            await self._fire_graph_end(callback_manager, config, state, turn_index)
        finally:
            try:
                # Text still buffered (a turn cut off by a disconnect) and the final state.
                if connected:
                    await self._final_save(config, state, checkpointer)
            finally:
                try:
                    if pump_task is not None:
                        pump_task.cancel()
                        # return_exceptions: the pump's own cancel is absorbed, but a cancel
                        # aimed at this task still raises out of the await.
                        await asyncio.gather(pump_task, return_exceptions=True)
                finally:
                    try:
                        await self._close_active_client()
                    finally:
                        self._session_active = False

    async def _close_active_client(self) -> None:
        client = self._active_client
        if client is None:
            return
        try:
            await client.close()
        except Exception:
            logger.warning("realtime provider close failed", exc_info=True)

    def _session_realtime_config(self, config: dict[str, Any]) -> RealtimeConfig:
        """Merge per-session overrides (``config["realtime"]``) over the agent's base config.

        Lets a caller (e.g. the API init frame) pick model/voice/modalities/vad per session
        without rebuilding the agent. Unknown keys are ignored; the result is re-validated.

        ``tools_tags`` can only narrow the agent's own tag filter: the session may drop tags
        but never add one, and an override that leaves nothing is ignored, since an empty
        filter would advertise every tool.
        """
        overrides = (config or {}).get("realtime") or {}
        if not overrides:
            return self.realtime_config
        base = self.realtime_config.model_dump()
        for key, requested in overrides.items():
            if requested is None or key not in base:
                continue
            value = requested
            if key == "tools_tags":
                value = _narrowed_tags(base["tools_tags"], requested)
                if value is None:
                    continue
            base[key] = value
        return RealtimeConfig.model_validate(base)

    async def _resolve_session_tools(self, rt: RealtimeConfig) -> RealtimeConfig:
        """Advertise the agent's ToolNode tools to the provider so the model can call them.

        No-op when tools were set explicitly on the config (the caller wins) or when there
        is no ToolNode. Tools are emitted as provider-neutral OpenAI-style dicts (the same
        shape the turn-based path uses); the provider client converts them. ``rt.tools_tags``
        filters which tools are advertised.
        """
        if rt.tools is not None:
            return rt
        tool_node = self._resolve_tool_node()
        if tool_node is None:
            return rt
        tags = set(rt.tools_tags) if rt.tools_tags else None
        schemas = await tool_node.all_tools(tags=tags)
        if not schemas:
            return rt
        return rt.model_copy(update={"tools": schemas})

    async def _resolve_session_system_instruction(
        self, rt: RealtimeConfig, state: AgentState, config: dict[str, Any]
    ) -> RealtimeConfig:
        """Flatten the agent's system prompt (+ skills, fenced memory) into ``system_instruction``.

        Gemini Live takes a single ``system_instruction`` string fixed at connect time, so
        the per-turn prompt list other agents send must be collapsed once, here. This is what
        makes ``system_prompt``, the skills catalog / session-mode content, and the
        memory system prompt actually reach the model in realtime (the matching tools are
        advertised separately by :meth:`_resolve_session_tools`).

        State-dependent pieces (session-mode skill from a state field, memory preload from the
        latest user query) are therefore a connect-time snapshot, not re-evaluated per turn;
        dynamic behaviour mid-session goes through ``activate_skill`` / memory tools instead.

        ``{field}`` placeholders in the prompt content are interpolated from ``state`` exactly
        like the turn-based path (via :func:`convert_messages`), so a system prompt that reads
        from state behaves identically here.
        """
        from tenxgraph.utils.converter import _interpolate_system_prompts

        base = list(self.system_prompt)
        if not base and rt.system_instruction:
            base = [{"role": "system", "content": rt.system_instruction}]

        # Only the developer's prompt is interpolated; skill and memory text is appended after.
        prompts = _interpolate_system_prompts(base, state)
        prompts = prompts + self._build_skill_prompts(state, [])

        instruction = "\n\n".join(str(p["content"]) for p in prompts if p.get("content")).strip()
        # Memories stay in the session instruction here because the Gemini Live client only
        # sends text as realtime input, which starts a reply; they get the same fence as the
        # turn-based path and are not system rank text.
        memory_block = await self._build_memory_block(state, config)
        if memory_block:
            instruction = f"{instruction}\n\n{memory_block}".strip()
        if not instruction:
            return rt
        return rt.model_copy(update={"system_instruction": instruction})

    async def _receive_until_stop(
        self, client: RealtimeClient, stop_event: asyncio.Event
    ) -> AsyncIterator[RealtimeEvent]:
        """Yield provider events, but return when ``stop_event`` fires *and* the provider
        is idle. Already-available events are always drained first (a closed input queue
        must not truncate the model's in-flight response)."""
        receiver = client.receive().__aiter__()
        stop_task = asyncio.ensure_future(stop_event.wait())
        next_task: asyncio.Future[Any] | None = None
        try:
            while True:
                next_task = asyncio.ensure_future(receiver.__anext__())
                done, _ = await asyncio.wait(
                    {next_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if next_task in done:
                    try:
                        yield next_task.result()
                        continue
                    except StopAsyncIteration:
                        return
                # Provider idle and stop requested: abandon the pending receive and end.
                return
        finally:
            # Also reached when this generator is cancelled or closed mid-wait: never leave
            # the pending receive running. gather() absorbs the tasks' own cancellation but
            # still raises if the cancel was aimed at this task.
            pending: list[asyncio.Future[Any]] = [stop_task]
            if next_task is not None and not next_task.done():
                pending.append(next_task)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if next_task is not None and next_task.done() and not next_task.cancelled():
                next_task.exception()  # retrieve so asyncio does not warn

    # ------------------------------------------------------------------ #
    # Pump: upstream queue -> provider socket.
    # ------------------------------------------------------------------ #
    async def _pump(
        self, input_queue: LiveInputQueue, stop_event: asyncio.Event | None = None
    ) -> None:
        try:
            async for item in input_queue:
                # Hold the lock across the send so a concurrent reconnect can't swap the
                # socket mid-send; re-read the client *inside* the lock to use the live one.
                async with self._send_lock:
                    client = self._active_client
                    if client is None:
                        continue
                    try:
                        if item.kind == "audio" and item.data is not None:
                            await client.send_audio(item.data, item.sample_rate)
                        elif item.kind == "text" and item.text is not None:
                            await client.send_text(item.text)
                        elif item.kind == "image" and item.data is not None:
                            await client.send_image(item.data, item.mime_type or "image/jpeg")
                        elif item.kind == "activity_start":
                            await client.send_activity_start()
                        elif item.kind == "activity_end":
                            await client.send_activity_end()
                    except Exception:
                        logger.warning(
                            "realtime pump send failed for %s frame", item.kind, exc_info=True
                        )
        finally:
            # Input is exhausted (queue closed): let the receive loop end once idle.
            if stop_event is not None:
                stop_event.set()

    # ------------------------------------------------------------------ #
    # Per-event handling. Returns the caller-facing events to yield.
    # ------------------------------------------------------------------ #
    async def _handle_event(
        self,
        event: RealtimeEvent,
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
        callback_manager: CallbackManager,
    ) -> list[RealtimeEvent]:
        if event.type == "tool_call":
            return await self._run_tool(event, config, state, callback_manager)

        if event.type == "input_transcript":
            return await self._accumulate_transcript(event, "user", config, state, checkpointer)
        if event.type == "output_transcript":
            return await self._accumulate_transcript(
                event, "assistant", config, state, checkpointer
            )

        if event.type == "turn_complete":
            # The end of a turn saves what the turn said: a provider may end it with no
            # transcription "finished" marker (Gemini Live does), which left the text unsaved.
            flushed = await self._flush_transcripts(config, state, checkpointer)
            await self._save_state(config, state, checkpointer)
            return [*flushed, event]

        if event.type == "interrupted":
            # Barge-in discards the model's in-flight reply (only part of it was spoken), so
            # its partial transcript is dropped and the restarted turn isn't concatenated onto
            # it. What the user said is kept.
            flushed = await self._flush_transcripts(config, state, checkpointer, keep_output=False)
            await self._save_state(config, state, checkpointer)
            self._publish_realtime(EventType.INTERRUPTED, config, ContentType.AUDIO, "barge_in")
            return [*flushed, event]
        if event.type == "go_away":
            self._publish_realtime(EventType.UPDATE, config, ContentType.UPDATE, "go_away")
        elif event.type == "session_update":
            self._resume_handle = event.resumption_handle
            await self._persist_handle(config, checkpointer)
            self._publish_realtime(EventType.UPDATE, config, ContentType.UPDATE, "session_resumed")

        return [event]

    async def _accumulate_transcript(
        self,
        event: InputTranscriptEvent | OutputTranscriptEvent,
        role: Literal["user", "assistant"],
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
    ) -> list[RealtimeEvent]:
        """Accumulate streamed transcript chunks and consolidate on the finished marker.

        Providers stream transcripts as partial chunks (``finished=False``) and end with a
        finished marker that usually carries no text. Partials pass through unchanged for live
        display; on ``finished`` we persist the full turn and emit a single consolidated
        event carrying the complete text (so consumers get the whole transcript without
        having to accumulate themselves). Turns with no transcribed text emit nothing.
        """
        is_user = role == "user"
        if is_user:
            self._input_transcript_buf += event.text
            buffered = self._input_transcript_buf
        else:
            self._output_transcript_buf += event.text
            buffered = self._output_transcript_buf

        if not event.finished:
            return [event]  # stream the partial for live UIs

        full = buffered.strip()
        if is_user:
            self._input_transcript_buf = ""
        else:
            self._output_transcript_buf = ""
        if not full:
            return []  # nothing transcribed this turn; drop the empty finish marker

        await self._persist_transcript(full, role, config, state, checkpointer)
        lifecycle = "input_transcript" if is_user else "output_transcript"
        self._publish_realtime(EventType.RESULT, config, ContentType.TRANSCRIPT, lifecycle)
        event_cls = InputTranscriptEvent if is_user else OutputTranscriptEvent
        return [event_cls(text=full, finished=True)]

    async def _run_tool(
        self,
        event: ToolCallEvent,
        config: dict[str, Any],
        state: AgentState,
        callback_manager: CallbackManager,
    ) -> list[RealtimeEvent]:
        tool_node = self._resolve_tool_node()
        if tool_node is None:
            result: Any = {"error": f"no tools registered for '{event.name}'"}
        elif event.name not in self._session_tool_names:
            # Only tools advertised to the model this session may run.
            logger.warning("realtime tool call for unadvertised tool '%s' refused", event.name)
            result = {"error": f"tool '{event.name}' is not available in this session"}
        elif not isinstance(event.args, dict):
            logger.warning("realtime tool call '%s' has non-object arguments", event.name)
            result = {"error": f"invalid arguments for tool '{event.name}': expected an object"}
        else:
            result = await self._invoke_with_ledger(
                tool_node, event, config, state, callback_manager
            )

        # Socket stays open; feed the result back to the model. Hold _send_lock (and re-read
        # the live client inside it) so this send is serialized against the pump and any
        # concurrent reconnect, exactly like the pump's own sends.
        async with self._send_lock:
            client = self._active_client
            if client is not None:
                await client.send_tool_response(event.id, event.name, result)
        return [event, ToolResultEvent(id=event.id, result=result)]

    async def _invoke_with_ledger(
        self,
        tool_node: ToolNode,
        event: ToolCallEvent,
        config: dict[str, Any],
        state: AgentState,
        callback_manager: CallbackManager,
    ) -> dict[str, Any]:
        """Run a tool call once per thread, using the same tool ledger as the text path.

        The key is the thread id, a hash of the tool name and arguments, and the provider
        call id. The provider only documents the call id as "unique", with no scope, and
        it may be absent (then a random id is made here and nothing can replay). Pairing
        it with the call content stops an id reused by a later session from skipping a
        different call, while a call re-sent after a reconnect or a crash restart of the
        same thread replays the recorded result instead of running the tool again. Without
        a thread id the key falls back to the session id (replay within the session only).
        """
        checkpointer = self._session_checkpointer
        thread_id = str((config or {}).get("thread_id") or "")
        scope = thread_id or self._session_id
        try:
            payload = json.dumps([event.name, event.args], sort_keys=True, default=str)
        except Exception:
            payload = repr((event.name, event.args))
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        key = ToolRuntimeMixin._ledger_key(f"realtime-{scope}-{digest}", event.id)
        ledger = _LedgerAccess(self._node_name_or_default())
        if checkpointer is not None and key:
            recorded = await ledger._get_recorded_tool_result(checkpointer, config, key)
            if recorded is not None:
                logger.info(
                    "realtime tool '%s' (call %s) replayed from ledger", event.name, event.id
                )
                return self._extract_tool_result(recorded)
        invoked = await tool_node.invoke(
            event.name, event.args, event.id, config, state, callback_manager=callback_manager
        )
        if checkpointer is not None and key:
            await ledger._record_tool_result(checkpointer, config, key, invoked)
        return self._extract_tool_result(invoked)

    def _node_name_or_default(self) -> str:
        return getattr(self, "_node_name", "LIVE")

    @staticmethod
    def _extract_tool_result(invoked: Any) -> dict[str, Any]:
        if isinstance(invoked, Message):
            for block in invoked.content:
                if getattr(block, "type", None) == "tool_result":
                    return {"result": getattr(block, "output", None)}
            return {"result": None}
        if isinstance(invoked, dict):
            return invoked
        return {"result": invoked}

    # ------------------------------------------------------------------ #
    # Transcript persistence (Message only; audio is never stored at rest).
    # ------------------------------------------------------------------ #
    async def _flush_transcripts(
        self,
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
        *,
        keep_output: bool = True,
    ) -> list[RealtimeEvent]:
        """Save transcript text still buffered because no "finished" marker closed it.

        The user's words first, then the model's, each as one message, and one consolidated
        transcript event each for the client. ``keep_output=False`` drops the model's buffer
        instead (barge-in). A buffer a "finished" marker already saved is empty, so nothing
        is saved twice.
        """
        flushed: list[RealtimeEvent] = []
        user_text = self._input_transcript_buf.strip()
        model_text = self._output_transcript_buf.strip() if keep_output else ""
        self._input_transcript_buf = ""
        self._output_transcript_buf = ""
        if user_text:
            await self._persist_transcript(user_text, "user", config, state, checkpointer)
            self._publish_realtime(
                EventType.RESULT, config, ContentType.TRANSCRIPT, "input_transcript"
            )
            flushed.append(InputTranscriptEvent(text=user_text, finished=True))
        if model_text:
            await self._persist_transcript(model_text, "assistant", config, state, checkpointer)
            self._publish_realtime(
                EventType.RESULT, config, ContentType.TRANSCRIPT, "output_transcript"
            )
            flushed.append(OutputTranscriptEvent(text=model_text, finished=True))
        return flushed

    async def _save_state(
        self,
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
    ) -> None:
        """Save the session's state (its context holds the transcripts) as a checkpoint.

        The messages are already saved one by one, so none are passed again. A storage
        failure is logged, not raised, for the same reason as in :meth:`_persist_transcript`.
        """
        if checkpointer is None:
            return
        try:
            await checkpointer.aput_checkpoint(config, state, [])
            await checkpointer.aput_state_cache(config, state)
        except Exception:
            logger.exception("realtime state could not be saved")

    async def _final_save(
        self,
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
    ) -> None:
        """Save what the session still holds when it ends, however it ends.

        A client that disconnects mid-turn closes this session's generator, and a cancelled
        session (anyio) is cancelled again at every await, so the save runs in a task of its
        own and is waited for a bounded time; a cancel is raised after the wait.
        """
        if checkpointer is None:
            return
        from tenxgraph.core.graph.utils.utils import detach_task, wait_detached

        async def _save() -> None:
            await self._flush_transcripts(config, state, checkpointer)
            await self._save_state(config, state, checkpointer)

        await wait_detached(detach_task(_save()))

    async def _persist_transcript(
        self,
        text: str,
        role: Literal["user", "assistant"],
        config: dict[str, Any],
        state: AgentState,
        checkpointer: BaseCheckpointer | None,
    ) -> None:
        msg = Message(role=role, content=[TextBlock(text=text)], metadata={"modality": "audio"})
        state.context = add_messages(state.context, [msg])
        if checkpointer is not None:
            # A storage failure must not look like a provider drop (which reconnects and
            # reopens the billed session), so it is logged here, not raised.
            try:
                await checkpointer.aput_messages(config, [msg])
            except Exception:
                logger.exception("realtime transcript could not be saved")

    # ------------------------------------------------------------------ #
    # Resumption: within-session reconnect + cross-session reseed.
    # ------------------------------------------------------------------ #
    async def _load_resume_handle(
        self, config: dict[str, Any], checkpointer: BaseCheckpointer | None
    ) -> str | None:
        if checkpointer is None or not self.realtime_config.session_resumption:
            return None
        try:
            thread = await checkpointer.aget_thread(config)
        except Exception:
            return None
        if thread and thread.metadata:
            handle = thread.metadata.get("resumption_handle")
            self._resume_handle = handle
            return handle
        return None

    async def _persist_handle(
        self, config: dict[str, Any], checkpointer: BaseCheckpointer | None
    ) -> None:
        if checkpointer is None:
            return
        from tenxgraph.utils.thread_info import ThreadInfo

        try:
            thread = await checkpointer.aget_thread(config)
        except Exception:
            # Writing a thin ThreadInfo now would overwrite the stored name and metadata.
            logger.warning("realtime resume handle not saved: thread read failed", exc_info=True)
            return
        metadata = dict(thread.metadata or {}) if thread else {}
        metadata["resumption_handle"] = self._resume_handle
        if thread is not None:
            info = thread.model_copy(update={"metadata": metadata})
        else:
            info = ThreadInfo(
                thread_id=config.get("thread_id", ""),
                user_id=config.get("user_id"),
                metadata=metadata,
            )
        try:
            await checkpointer.aput_thread(config, info)
        except Exception:
            logger.exception("realtime resume handle could not be saved")

    async def _maybe_reseed(
        self,
        config: dict[str, Any],
        checkpointer: BaseCheckpointer | None,
        context_manager: BaseContextManager | None,
        *,
        resumed: bool = False,
    ) -> None:
        # When the provider restored context from a resumption handle, reseeding would
        # replay the whole conversation a second time.
        if checkpointer is None or resumed:
            return
        try:
            history = await checkpointer.alist_messages(config)
        except Exception:
            history = None
        if not history:
            return
        if context_manager is not None:
            try:
                trimmed = await context_manager.atrim_context(AgentState(context=list(history)))
                history = trimmed.context
            except Exception:
                logger.warning("context compression failed during reseed; using raw history")
        if history and self._active_client is not None:
            await self._active_client.reseed_history(list(history))

    async def _attempt_reconnect(
        self, rt: RealtimeConfig, forced: bool, attempts: int
    ) -> tuple[int, ErrorEvent | None]:
        """Reconnect after a drop. Returns ``(attempts, fatal_error_or_None)``.

        ``forced`` (go_away) is an expected provider rotation: reconnect promptly, no backoff.
        Error-driven drops back off exponentially with a hard attempt cap so a flapping or
        down provider can't spin a tight reconnect storm; once the cap is hit a fatal
        :class:`ErrorEvent` is returned for the caller to surface before ending the session.
        """
        if forced:
            with contextlib.suppress(Exception):
                await self._reconnect(rt)
            return 0, None

        attempts += 1
        if attempts > self._reconnect_max_attempts:
            logger.error(
                "realtime reconnect attempts exhausted (%d); giving up",
                self._reconnect_max_attempts,
            )
            return attempts, ErrorEvent(
                code="reconnect_failed",
                message=(
                    "realtime session lost and could not be resumed after "
                    f"{self._reconnect_max_attempts} attempts"
                ),
                fatal=True,
            )

        delay = min(self._reconnect_base_delay * (2 ** (attempts - 1)), self._reconnect_max_delay)
        await asyncio.sleep(delay)
        with contextlib.suppress(Exception):
            await self._reconnect(rt)
        return attempts, None

    async def _reconnect(self, rt: RealtimeConfig) -> None:
        async with self._send_lock:
            old = self._active_client
            with contextlib.suppress(Exception):
                if old is not None:
                    await old.close()
            client = self._client_factory()
            try:
                await client.connect(rt, resume_handle=self._resume_handle)
            except BaseException:
                with contextlib.suppress(Exception):
                    await client.close()
                raise
            self._active_client = client

    # ------------------------------------------------------------------ #
    # Lifecycle hooks (session == graph run; turn == one model generation).
    # ------------------------------------------------------------------ #
    async def _fire_graph_start(
        self, cb: CallbackManager, config: dict[str, Any], state: AgentState
    ) -> AgentState:
        if not cb._lifecycle_hooks:
            return state
        return await cb.fire_on_graph_start(GraphLifecycleContext(config=config), state)

    async def _fire_graph_end(
        self, cb: CallbackManager, config: dict[str, Any], state: AgentState, turns: int
    ) -> None:
        if not cb._lifecycle_hooks:
            return
        messages = list(getattr(state, "context", []) or [])
        await cb.fire_on_graph_end(GraphLifecycleContext(config=config), state, messages, turns)

    async def _fire_turn_start(
        self, cb: CallbackManager, config: dict[str, Any], state: AgentState, turn_index: int
    ) -> AgentState:
        if not cb._lifecycle_hooks:
            return state
        return await cb.fire_on_turn_start(GraphLifecycleContext(config=config), state, turn_index)

    async def _fire_turn_end(
        self, cb: CallbackManager, config: dict[str, Any], state: AgentState, turn_index: int
    ) -> AgentState:
        if not cb._lifecycle_hooks:
            return state
        return await cb.fire_on_turn_end(GraphLifecycleContext(config=config), state, turn_index)

    # ------------------------------------------------------------------ #
    # Observability for events ToolNode doesn't already publish.
    # ------------------------------------------------------------------ #
    def _publish_realtime(
        self,
        event_type: EventType,
        config: dict[str, Any],
        content_type: ContentType,
        lifecycle: str,
    ) -> None:
        publish_event(
            EventModel.default(
                config,
                data={},
                content_type=[content_type],
                event=Event.REALTIME,
                event_type=event_type,
                node_name=getattr(self, "_node_name", "LIVE"),
                extra={"lifecycle": lifecycle, "modality": "audio"},
            )
        )
