"""BackendsPublisher: one OtelPublisher that writes the attribute keys of several trace backends.

``setup_observability`` can send the same spans to Phoenix, LangSmith, Logfire and a plain
OTLP endpoint at once over one TracerProvider. Each backend reads its own keys, so the
publisher writes the keys of every enabled backend: OpenInference for ``phoenix``,
``langsmith.*`` for ``langsmith``, ``logfire.msg`` for ``logfire``. ``otlp`` adds none; the
``gen_ai.*`` and ``tenxgraph.*`` attributes every span carries are enough for it.

It configures no exporter: set up the TracerProvider first (``setup_observability``), or
pass ``tracer``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from .identity import UserIdMode
from .langsmith_publisher import langsmith_attributes
from .logfire_publisher import logfire_attributes
from .otel_publisher import ObservabilityLevel, OtelPublisher, SpanPhase, SpanRole
from .phoenix_publisher import openinference_attributes


if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from .events import EventModel


_Dialect = Callable[
    [OtelPublisher, SpanRole, SpanPhase, str, "EventModel", dict[str, Any]], dict[str, Any]
]

# Backend name in the ``observability`` config -> the attributes it adds (None: basic only).
BACKEND_ATTRIBUTES: dict[str, _Dialect | None] = {
    "phoenix": openinference_attributes,
    "langsmith": langsmith_attributes,
    "logfire": logfire_attributes,
    "otlp": None,
}


class BackendsPublisher(OtelPublisher):
    """OtelPublisher whose ``span_attributes`` adds the keys of each backend in ``backends``.

    Args:
        backends: Names from ``BACKEND_ATTRIBUTES`` (``phoenix``, ``langsmith``,
            ``logfire``, ``otlp``). Empty or only ``otlp`` writes the basic attributes.
        tracer: Optional explicit Tracer. Uses the global TracerProvider if None.
        level: How much data to emit, for every backend's keys.
        user_id_mode: How ``user_id`` appears, for every backend's keys.

    Raises:
        ValueError: If a backend name is unknown, or ``user_id_mode`` is invalid.
    """

    def __init__(
        self,
        backends: Iterable[str],
        *,
        tracer: Tracer | None = None,
        level: ObservabilityLevel = ObservabilityLevel.FULL,
        user_id_mode: UserIdMode = "raw",
    ) -> None:
        names = list(backends)
        unknown = sorted(set(names) - set(BACKEND_ATTRIBUTES))
        if unknown:
            raise ValueError(
                f"Unknown observability backend(s) {unknown}; use {sorted(BACKEND_ATTRIBUTES)}"
            )
        super().__init__(tracer=tracer, level=level, user_id_mode=user_id_mode)
        self.backends = [name for name in BACKEND_ATTRIBUTES if name in names]
        self._dialects = [d for name in self.backends if (d := BACKEND_ATTRIBUTES[name])]

    def span_attributes(
        self,
        role: SpanRole,
        phase: SpanPhase,
        name: str,
        event: EventModel,
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Add the keys of each enabled backend in turn."""
        for dialect in self._dialects:
            attrs = dialect(self, role, phase, name, event, attrs)
        return attrs


def enabled_backends(config: dict[str, Any]) -> list[str]:
    """Names of the backends with ``"enabled": true`` in an ``observability`` config block."""
    return [name for name in BACKEND_ATTRIBUTES if (config.get(name) or {}).get("enabled")]
