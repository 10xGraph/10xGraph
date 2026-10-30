from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .base_converter import BaseConverter


if TYPE_CHECKING:
    from tenxgraph.core.state.message import Message


logger = logging.getLogger("tenxgraph.adapters")


def is_empty_reply(message: Any) -> bool:
    """True for an assistant reply with no text, no tool calls and no media.

    A reply that holds only reasoning counts as empty: the user would see nothing.
    """
    from tenxgraph.core.state.message_block import ReasoningBlock, TextBlock

    if getattr(message, "role", None) != "assistant" or getattr(message, "tools_calls", None):
        return False
    if (message.text() or "").strip():
        return False
    blocks = message.content if isinstance(message.content, list) else []
    return all(isinstance(b, TextBlock | ReasoningBlock) for b in blocks)


class ModelResponseConverter:
    """Wrap an LLM SDK call and normalize its output via a converter.

    Supports sync/async invocation and streaming. Use `invoke()` for
    non-streaming calls and `stream()` for streaming calls.
    """

    def __init__(
        self,
        response: Any | Callable[..., Any],
        converter: BaseConverter | str,
    ) -> None:
        """
        Initialize ModelResponseConverter.

        Args:
            response (Any | Callable[..., Any]): The LLM response or a callable returning
                a response.
            converter (BaseConverter | str): Converter instance or string identifier
                (e.g., "openai").

        Raises:
            ValueError: If the converter is not supported.
        """
        self.response = response
        # Extra metadata merged into every Message this converter produces, e.g.
        # {"allowed_tools": [...]} so dispatch can tell which tools the model was shown.
        self.message_metadata: dict[str, Any] = {}
        # Called once when ``stream()`` ends (finished, failed or closed early) with the
        # last message that carried token usage (or None) and the error (or None). Lets the
        # caller attach usage that is only known once the stream has been read.
        self.on_stream_end: Callable[[Any, BaseException | None], None] | None = None
        # Callbacks run on every complete (non-delta) Message this converter produces, after
        # ``message_metadata`` is merged. They may mutate the message in place (the saved
        # message is that same object). A failing hook is logged and skipped.
        self.message_hooks: list[Callable[[Any], None]] = []
        # Empty-reply policy (set by the Agent). When ``empty_reply_retry`` is set and the
        # provider answers with an empty reply (see ``is_empty_reply``), ``on_empty_reply``
        # is called with ``(message, will_retry)``, the call is repeated once through
        # ``empty_reply_retry`` (it returns the new raw response), and a second empty reply
        # raises ``empty_reply_error(message)``. An empty reply is never returned or yielded.
        self.empty_reply_retry: Callable[[], Awaitable[Any]] | None = None
        self.on_empty_reply: Callable[[Any, bool], None] | None = None
        self.empty_reply_error: Callable[[Any], BaseException] | None = None

        if isinstance(converter, str):
            if converter == "openai":
                from .openai_converter import OpenAIConverter

                self.converter = OpenAIConverter()
                logger.debug("Using OpenAIConverter for response conversion")

            elif converter == "openai_responses":
                from .openai_responses_converter import OpenAIResponsesConverter

                self.converter = OpenAIResponsesConverter()
                logger.debug("Using OpenAIResponsesConverter for response conversion")

            elif converter == "google":
                from .google_genai_converter import GoogleGenAIConverter

                self.converter = GoogleGenAIConverter()
                logger.debug("Using GoogleGenAIConverter for response conversion")

            elif converter == "anthropic":
                from .anthropic_converter import AnthropicConverter

                self.converter = AnthropicConverter()
                logger.debug("Using AnthropicConverter for response conversion")

            else:
                logger.error(f"Unsupported converter: {converter}")
                raise ValueError(
                    f"Unsupported converter: {converter}. "
                    "Supported: 'openai', 'openai_responses', 'google', 'anthropic'"
                )

        elif isinstance(converter, BaseConverter):
            self.converter = converter
            logger.debug(f"Using custom converter: {type(converter).__name__}")
        else:
            logger.error(f"Unsupported converter: {converter}")
            raise ValueError(f"Unsupported converter: {converter}")

    @staticmethod
    def _warn_if_empty_response(response: Any, *, context: str) -> None:
        """Emit a warning when a provider returned no payload to convert."""
        if response is None:
            logger.warning("Received empty response while %s", context)

    async def invoke(self) -> Message:
        """
        Call the underlying function and convert a non-streaming response to Message.

        Returns:
            Message: The normalized message from the LLM response.

        Raises:
            Exception: If the underlying function or converter fails.
        """
        logger.debug("Invoking model response converter")
        if callable(self.response):
            if inspect.iscoroutinefunction(self.response):
                logger.debug("Detected async callable response")
                response = await self.response()
            else:
                response = self.response()
        else:
            response = self.response

        self._warn_if_empty_response(response, context="converting a model response")
        logger.debug("Converting response to Message")
        message = await self.converter.convert_response(response)  # type: ignore
        if self.empty_reply_retry is not None and is_empty_reply(message):
            self._note_empty(message, will_retry=True)
            response = await self.empty_reply_retry()
            message = await self.converter.convert_response(response)  # type: ignore
            if is_empty_reply(message):
                self._note_empty(message, will_retry=False)
                raise self._empty_error(message)
        self._apply_metadata(message)
        return message

    def _note_empty(self, message: Any, *, will_retry: bool) -> None:
        if self.on_empty_reply is not None:
            self.on_empty_reply(message, will_retry)

    def _empty_error(self, message: Any) -> BaseException:
        if self.empty_reply_error is not None:
            return self.empty_reply_error(message)
        return RuntimeError("The model returned an empty reply")

    def _apply_metadata(self, message: Any) -> None:
        """Merge ``message_metadata`` into a produced Message and run ``message_hooks``."""
        from tenxgraph.core.state.message import Message

        if not isinstance(message, Message):
            return
        if self.message_metadata:
            message.metadata = {**(message.metadata or {}), **self.message_metadata}
        if message.delta:
            return
        for hook in self.message_hooks:
            try:
                hook(message)
            except Exception:
                logger.debug("Message hook failed", exc_info=True)

    def _notify_stream_end(self, last_message: Any, error: BaseException | None) -> None:
        hook, self.on_stream_end = self.on_stream_end, None
        if hook is None:
            return
        try:
            hook(last_message, error)
        except Exception:
            logger.debug("Stream end hook failed", exc_info=True)

    async def stream(  # noqa: PLR0912
        self,
        config: dict,
        node_name: str,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Call the underlying function and yield normalized streaming events and final Message.

        Args:
            config (dict): Node configuration parameters for streaming.
            node_name (str): Name of the node processing the response.
            meta (dict | None): Optional metadata for conversion.

        Yields:
            Message: Normalized streaming message events from the LLM response.

        Raises:
            ValueError: If config is not provided.
            Exception: If the underlying function or converter fails.
        """
        if not config:
            logger.error("Config must be provided for streaming conversion")
            err = ValueError("Config must be provided for streaming conversion")
            self._notify_stream_end(None, err)
            raise err

        if callable(self.response):
            if inspect.iscoroutinefunction(self.response):
                logger.debug("Detected async callable response for streaming")
                response = await self.response()
            else:
                response = self.response()
        else:
            response = self.response

        self._warn_if_empty_response(response, context="starting streaming conversion")
        last_message: Any = None
        error: BaseException | None = None
        retried = False
        try:
            while True:
                held: Any = None
                async for item in self.converter.convert_streaming_response(  # type: ignore
                    config,
                    node_name=node_name,
                    response=response,
                    meta=meta,
                ):
                    if getattr(item, "usages", None) is not None:
                        last_message = item
                    if (
                        self.empty_reply_retry is not None
                        and not getattr(item, "delta", True)
                        and is_empty_reply(item)
                    ):
                        # Hold back the empty final message: it is retried, never emitted.
                        held = item
                        continue
                    self._apply_metadata(item)
                    yield item
                if held is None:
                    break
                if retried:
                    self._note_empty(held, will_retry=False)
                    raise self._empty_error(held)
                self._note_empty(held, will_retry=True)
                retried = True
                response = await self.empty_reply_retry()  # type: ignore[misc]
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._notify_stream_end(last_message, error)

        logger.debug("Completed streaming conversion")
