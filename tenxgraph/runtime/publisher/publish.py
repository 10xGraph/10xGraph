import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from injectq import Inject
from pydantic import BaseModel

from tenxgraph.runtime.publisher._buffered import BufferedPublisher
from tenxgraph.runtime.publisher.base_publisher import BasePublisher, PublisherKind
from tenxgraph.runtime.publisher.composite_publisher import CompositePublisher
from tenxgraph.runtime.publisher.console_publisher import ConsolePublisher
from tenxgraph.runtime.publisher.events import (
    RUN_END_EVENTS,
    ContentType,
    Event,
    EventModel,
    EventTag,
    EventType,
)
from tenxgraph.utils import metrics
from tenxgraph.utils.background_task_manager import BackgroundTaskManager
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.log_safety import TRACE_ONLY_ERROR_KEYS
from tenxgraph.utils.temp_perf import done as _perf_done  # TEMP-PERF
from tenxgraph.utils.temp_perf import now as _perf_now  # TEMP-PERF


logger = logging.getLogger("tenxgraph.publisher")


async def _publish_event_task(
    event: EventModel,
    publisher: BasePublisher | None,
) -> None:
    """Publish an event asynchronously if publisher is configured.

    Args:
        event: The event to publish.
        publisher: The publisher instance, or None.
    """
    if publisher:
        # Optional bound on one send, set by the developer as ``publish_timeout`` (seconds)
        # in the publisher config. Unset means wait for the broker.
        config = getattr(publisher, "config", None)
        timeout = config.get("publish_timeout") if isinstance(config, dict) else None
        if not isinstance(timeout, int | float) or isinstance(timeout, bool):
            timeout = None
        attrs = {"publisher": type(publisher).__name__}
        try:
            with metrics.timer("tenxgraph.publish.duration", attributes=attrs):
                if timeout:
                    await asyncio.wait_for(publisher.publish(event), timeout=timeout)
                else:
                    await publisher.publish(event)
            logger.debug("Published event: %s", event)
        except TimeoutError:
            logger.error("Publishing event timed out after %ss; event dropped", timeout)
        except Exception as e:
            logger.error("Failed to publish event: %s", e)


# Default size above which an event's ``data["state"]`` is replaced by a bounded summary.
DEFAULT_STATE_SNAPSHOT_LIMIT = 65536
_SUMMARY_MAX_CONTEXT = 50
_SUMMARY_PART_LIMIT = 8192


def _json_len(value: Any) -> int:
    try:
        return len(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def _state_limit(publisher: BasePublisher) -> int | None:
    """Size limit from the publisher's ``state_snapshot_limit`` config; None means no limit."""
    config = getattr(publisher, "config", None)
    if not isinstance(config, dict) or "state_snapshot_limit" not in config:
        return DEFAULT_STATE_SNAPSHOT_LIMIT
    limit = config["state_snapshot_limit"]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return None
    return limit


def bound_state_snapshot(state: Any, limit: int | None) -> Any:
    """Return ``state`` as is when small, else a bounded summary of it.

    The summary keeps: ``truncated`` (True), ``bytes`` (original JSON size), ``fields``
    (top-level field names), ``context_count``, ``context`` (``message_id`` and ``role`` of the
    last 50 messages, plus full system messages up to 8 KiB each), ``context_summary`` (cut
    to 1000 characters) and ``execution_meta`` when it is under 8 KiB.
    """
    if limit is None or not isinstance(state, dict):
        return state
    size = _json_len(state)
    if size <= limit:
        return state
    context = state.get("context")
    entries: list[dict[str, Any]] = []
    if isinstance(context, list):
        for m in context[-_SUMMARY_MAX_CONTEXT:]:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system" and _json_len(m) <= _SUMMARY_PART_LIMIT:
                entries.append(m)
            else:
                entries.append({"message_id": m.get("message_id"), "role": m.get("role")})
    summary: dict[str, Any] = {
        "truncated": True,
        "bytes": size,
        "fields": sorted(state),
        "context_count": len(context) if isinstance(context, list) else 0,
        "context": entries,
    }
    ctx_summary = state.get("context_summary")
    if isinstance(ctx_summary, str):
        summary["context_summary"] = ctx_summary[:1000]
    meta = state.get("execution_meta")
    if meta is not None and _json_len(meta) <= _SUMMARY_PART_LIMIT:
        summary["execution_meta"] = meta
    return summary


def _select(publisher: BasePublisher, kind: PublisherKind) -> BasePublisher | None:
    """The part of ``publisher`` of the given kind, or None. Never changes the bound one."""
    if isinstance(publisher, CompositePublisher):
        picked = [p for c in publisher._publishers if (p := _select(c, kind)) is not None]
        return CompositePublisher(picked) if picked else None
    return publisher if publisher.kind == kind else None


def _leaves(publisher: BasePublisher) -> list[BasePublisher]:
    """The publishers inside ``publisher`` (composites walked), in order."""
    if isinstance(publisher, CompositePublisher):
        return [leaf for child in publisher._publishers for leaf in _leaves(child)]
    return [publisher]


def _receives(publisher: BasePublisher, tag: EventTag) -> PublisherKind | None:
    """Which version of an event tagged ``tag`` the publisher gets, or None for none.

    Observability publishers get every event in the observability version (state, error
    text). Event publishers get ``all`` events in the events version; ``ConsolePublisher``,
    a debug tool, gets every event in that version.
    """
    if getattr(publisher, "kind", "events") == "observability":
        return "observability"
    if tag == "all" or isinstance(publisher, ConsolePublisher):
        return "events"
    return None


def listening(
    tag: EventTag = "all", publisher: BasePublisher | None = Inject[BasePublisher]
) -> bool:
    """True when a bound publisher takes events tagged ``tag``, so one built now reaches
    someone. Call sites check it before building a costly payload."""
    publisher = fresh(publisher)
    return publisher is not None and any(
        _receives(leaf, tag) is not None for leaf in _leaves(publisher)
    )


async def flush_buffered(publisher: BasePublisher | None, *, stop: bool = False) -> None:
    """Send what the broker publishers in ``publisher`` have queued; never raises.

    ``stop=True`` also stops their workers (before an event loop closes: sync ``invoke``).
    """
    if publisher is None:
        return
    for leaf in _leaves(publisher):
        if isinstance(leaf, BufferedPublisher):
            try:
                await (leaf.stop_buffer() if stop else leaf.flush())
            except Exception as e:
                logger.error("Flushing %s failed: %s", type(leaf).__name__, e)


def _plain(value: Any) -> Any:
    """A pydantic model, or a list holding them, as plain dicts. Callers put live objects
    (state, messages) in an event, and they are dumped here, only when someone listens."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list) and any(isinstance(v, BaseModel) for v in value):
        return [v.model_dump() if isinstance(v, BaseModel) else v for v in value]
    return value


def publish_lifecycle(
    config: dict[str, Any],
    lifecycle: str,
    data: dict[str, Any],
    extra: dict[str, Any] | None = None,
    *,
    event: Event = Event.GRAPH_EXECUTION,
    event_type: EventType = EventType.UPDATE,
    content_type: ContentType = ContentType.STATE,
    node_name: str = "",
    tag: EventTag = "all",
) -> None:
    """Publish one lifecycle event (``metadata["lifecycle"]``), whether or not hooks exist."""
    if not listening(tag):
        return
    publish_event(
        EventModel.default(
            config,
            data=data,
            event=event,
            event_type=event_type,
            node_name=node_name,
            content_type=[content_type],
            extra={"lifecycle": lifecycle, **(extra or {})},
            tag=tag,
        )
    )


def _deliver(
    event: EventModel,
    publishers: list[BasePublisher],
    task_manager: BackgroundTaskManager,
) -> None:
    """Hand ``event`` to each publisher: inline ones now, broker ones to their queue, the
    rest in one background task."""
    background: list[BasePublisher] = []
    for publisher in publishers:
        try:
            # ``is True``: a mock or duck-typed publisher without the flag stays in the
            # background.
            if getattr(publisher, "inline", False) is True:
                publisher.publish_inline(event)
            elif isinstance(publisher, BufferedPublisher):
                publisher.enqueue(event)
            else:
                background.append(publisher)
        except Exception as e:
            logger.error("Failed to publish event: %s", e)
    if background:
        target = background[0] if len(background) == 1 else CompositePublisher(background)
        task_manager.create_task(_publish_event_task(event, target))


def publish_event(
    event: EventModel,
    publisher: BasePublisher | None = Inject[BasePublisher],
    task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
    only: PublisherKind | None = None,
) -> None:
    """Publish an event to the bound publishers that take its tag, without waiting.

    ``event.tag`` and each publisher's ``kind`` decide who gets it: observability
    publishers get every event, event publishers get ``"all"`` events,
    ``ConsolePublisher`` gets every event (see ``EventTag``). When nobody takes the tag
    nothing is built. Observability publishers get the full event; event publishers get
    ``data["state"]`` only on run-end events (``RUN_END_EVENTS``), and never
    ``error_message`` or ``error_traceback`` (they can hold paths, DSNs or SQL).

    Delivery is best-effort. Publishers with ``inline = True`` (OTEL tracing) get the event
    right here, in order, through ``publish_inline``. Broker publishers (RabbitMQ, Kafka,
    Redis) queue it and send batches from one worker (see ``_buffered``). Any other
    publisher gets it in a background task; when more than the task manager's cap (1000 by
    default) are pending, new events are shed and a warning is logged.
    ``CompiledGraph.aclose()`` and sync ``invoke()`` wait for pending events.
    Pydantic models (or lists of them) in ``data`` and ``metadata`` are dumped here, so a
    caller passes live objects and pays for the dump only when a publisher takes them.
    ``data["state"]`` larger than the publisher's ``state_snapshot_limit`` config (JSON
    characters, default 65536; 0 or None disables) is replaced by a bounded summary.

    Args:
        event: The event to publish.
        publisher: The publisher instance (injected).
        task_manager: The background task manager (injected).
        only: Send to publishers of this ``kind`` only (``"events"`` or ``"observability"``).
            None (default) sends to every publisher that takes the tag.
    """
    publisher = fresh(publisher)
    task_manager = fresh(task_manager)
    if publisher is not None and only is not None:
        publisher = _select(publisher, only)
    if publisher is None:
        return
    groups: dict[str, list[BasePublisher]] = {"observability": [], "events": []}
    for leaf in _leaves(publisher):
        version = _receives(leaf, event.tag)
        if version is not None:
            groups[version].append(leaf)
    # Nobody takes this tag: build nothing (an LLM payload with only a broker bound).
    if not groups["observability"] and not groups["events"]:
        return

    _tp0 = _perf_now()  # TEMP-PERF
    # Snapshot the event first: the run loop reuses one event object and edits its data,
    # metadata and phase between publishes, so a queued copy would otherwise send the last
    # values for every event. ``timestamp`` is the moment of this publish for the same reason.
    data = {k: _plain(v) for k, v in event.data.items() if k != "state"}
    metadata = {k: _plain(v) for k, v in event.metadata.items()}
    update = {"metadata": metadata, "timestamp": datetime.now().timestamp()}
    # The state is dumped only when someone gets it: observability publishers, or event
    # publishers on a run-end event.
    has_state = "state" in event.data
    obs_state = has_state and bool(groups["observability"])
    events_state = (
        has_state and bool(groups["events"]) and event.metadata.get("event_name") in RUN_END_EVENTS
    )
    state = _plain_state(event.data["state"]) if obs_state or events_state else None
    if groups["observability"]:
        full = dict(data)
        if obs_state:
            full["state"] = bound_state_snapshot(state, _group_limit(groups["observability"]))
        snapshot = event.model_copy(update={**update, "data": full})
        _deliver(snapshot, groups["observability"], task_manager)
    if groups["events"]:
        kept = {k: v for k, v in data.items() if k not in TRACE_ONLY_ERROR_KEYS}
        if events_state:
            kept["state"] = bound_state_snapshot(state, _group_limit(groups["events"]))
        _deliver(event.model_copy(update={**update, "data": kept}), groups["events"], task_manager)
    _perf_done("core_publish_event_build", _tp0)  # TEMP-PERF


def _plain_state(state: Any) -> Any:
    """A live state object keeps changing after the publish and is not a dict, so publishers
    and the size cap would not see a stable snapshot: dump it to a plain dict here, so
    streamed and invoked runs hand publishers the same shape."""
    if not isinstance(state, dict) and callable(getattr(state, "model_dump", None)):
        return state.model_dump()
    return state


def _group_limit(publishers: list[BasePublisher]) -> int | None:
    """The ``state_snapshot_limit`` of a single publisher, else the default."""
    if len(publishers) == 1:
        return _state_limit(publishers[0])
    return DEFAULT_STATE_SNAPSHOT_LIMIT
