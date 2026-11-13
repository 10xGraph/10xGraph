import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from injectq import Inject

from tenxgraph.runtime.publisher.base_publisher import BasePublisher, PublisherKind
from tenxgraph.runtime.publisher.composite_publisher import CompositePublisher
from tenxgraph.runtime.publisher.events import EventModel
from tenxgraph.utils.background_task_manager import BackgroundTaskManager, resolve_task_manager
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.log_safety import TRACE_ONLY_ERROR_KEYS


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
        try:
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


def _split_inline(publisher: BasePublisher) -> tuple[list[BasePublisher], BasePublisher | None]:
    """The ``inline`` publishers in ``publisher``, and the rest (None when nothing is left)."""
    if isinstance(publisher, CompositePublisher):
        inline: list[BasePublisher] = []
        rest: list[BasePublisher] = []
        for child in publisher._publishers:
            child_inline, child_rest = _split_inline(child)
            inline += child_inline
            if child_rest is not None:
                rest.append(child_rest)
        if not inline:
            return [], publisher
        return inline, CompositePublisher(rest) if rest else None
    # ``is True``: a mock or duck-typed publisher without the flag stays in the background.
    return ([publisher], None) if getattr(publisher, "inline", False) is True else ([], publisher)


def publish_event(
    event: EventModel,
    publisher: BasePublisher | None = Inject[BasePublisher],
    task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
    only: PublisherKind | None = None,
) -> None:
    """Publish an event asynchronously using the background task manager.

    Delivery is best-effort: events run as background tasks, and when more than the
    manager's cap (1000 by default) are pending, new events are shed and a warning is
    logged. ``CompiledGraph.aclose()`` and sync ``invoke()`` wait for pending events.
    Publishers with ``inline = True`` (OTEL tracing) get the event right here instead, in
    order, through ``publish_inline``. ``error_message`` and ``error_traceback`` reach
    ``"observability"`` publishers only.
    ``data["state"]`` larger than the publisher's ``state_snapshot_limit`` config (JSON
    characters, default 65536; 0 or None disables) is replaced by a bounded summary.

    Args:
        event: The event to publish.
        publisher: The publisher instance (injected).
        task_manager: The background task manager (injected).
        only: Send to publishers of this ``kind`` only (``"events"`` or ``"observability"``).
            None (default) sends to every publisher.
    """
    publisher = fresh(publisher)
    task_manager = resolve_task_manager(fresh(task_manager))
    # No sink bound -> nothing to publish. Spawning a task per event just to
    # discover there is nowhere to send it cost a task and kept the event alive
    # for no reason, on the hot path of every node in every run.
    if publisher is not None and only is not None:
        publisher = _select(publisher, only)
    if publisher is None:
        return

    # Store the task to prevent it from being garbage collected. May return None
    # when the manager is shedding load (see BackgroundTaskManager.create_task).
    # Snapshot the event first: the run loop reuses one event object and edits its
    # data, metadata and phase between publishes, so the queued task would otherwise
    # send the last values for every event.
    data = dict(event.data)
    # A queued event must not pin a whole run state while the sink is slow, so a large
    # ``state`` is cut to a bounded summary (see ``bound_state_snapshot``).
    if "state" in data:
        state = data["state"]
        # A live state object keeps changing after this call and is not a dict, so publishers
        # and the size cap would not see a stable snapshot. Dump it to a plain dict here, for
        # every call site, so streamed and invoked runs hand publishers the same shape.
        if not isinstance(state, dict) and callable(getattr(state, "model_dump", None)):
            state = state.model_dump()
        data["state"] = bound_state_snapshot(state, _state_limit(publisher))
    # The same reuse means ``timestamp`` is stale on the second publish of an event, so the
    # snapshot carries the moment of this publish (publish happens when the thing happens).
    snapshot = event.model_copy(
        update={
            "data": data,
            "metadata": dict(event.metadata),
            "timestamp": datetime.now().timestamp(),
        }
    )
    # Exception text and tracebacks (``error_details``) can hold paths, DSNs or SQL, so they
    # go to observability publishers (trace backends) only; event streams get the type.
    plain = snapshot
    if any(key in data for key in TRACE_ONLY_ERROR_KEYS):
        kept = {k: v for k, v in data.items() if k not in TRACE_ONLY_ERROR_KEYS}
        plain = snapshot.model_copy(update={"data": kept})

    inline, background = _split_inline(publisher)
    for sink in inline:
        try:
            sink.publish_inline(snapshot if sink.kind == "observability" else plain)
        except Exception as e:
            logger.error("Failed to publish event: %s", e)
    if background is None:
        return
    if plain is snapshot:
        task_manager.create_task(_publish_event_task(snapshot, background))
        return
    for kind, version in (("observability", snapshot), ("events", plain)):
        part = _select(background, kind)  # type: ignore[arg-type]
        if part is not None:
            task_manager.create_task(_publish_event_task(version, part))
