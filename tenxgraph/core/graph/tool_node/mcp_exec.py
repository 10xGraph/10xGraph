"""MCPMixin — executes tools via a Model Context Protocol (MCP) client."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import typing as t

from tenxgraph.core.state import (
    ContentBlock,
    ErrorBlock,
    Message,
    ToolResultBlock,
)
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.utils import CallbackContext, CallbackManager, InvocationType

from ._helpers import _normalize_resource_uri, generic_tool_failure


logger = logging.getLogger("tenxgraph.graph.tool_node")


DEFAULT_MCP_CONNECT_TIMEOUT = 30.0


class MCPMixin:
    _client: t.Any | None
    # The concrete ToolNode defines these
    mcp_tools: list[str]  # type: ignore[assignment]
    _pass_user_info_to_mcp: bool  # type: ignore[assignment]

    @contextlib.asynccontextmanager
    async def _mcp_session(self) -> t.AsyncIterator[None]:
        """Open the MCP client session, bounding only the connect by ``mcp_connect_timeout``."""
        timeout = getattr(self, "_mcp_connect_timeout", None)
        async with contextlib.AsyncExitStack() as stack:
            if timeout is None:
                await stack.enter_async_context(self._client)
            else:
                try:
                    async with asyncio.timeout(timeout):
                        await stack.enter_async_context(self._client)
                except TimeoutError as exc:
                    raise TimeoutError(f"MCP connect timed out after {timeout}s") from exc
            yield

    def _cap_mcp_result(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """Cut an MCP result that is longer than ``mcp_max_result_chars`` (opt-in)."""
        cap = getattr(self, "_mcp_max_result_chars", None)
        if cap is None:
            return blocks
        for block in blocks:
            output = getattr(block, "output", None)
            if output is None:
                continue
            text = output if isinstance(output, str) else json.dumps(output, default=str)
            if len(text) > cap:
                block.output = text[:cap] + f"\n[output cut: {len(text) - cap} more characters]"
        return blocks

    def _serialize_result(
        self,
        tool_call_id: str,
        res: t.Any,
    ) -> list[ContentBlock]:
        return self._cap_mcp_result(self._serialize_result_uncapped(tool_call_id, res))

    def _serialize_result_uncapped(
        self,
        tool_call_id: str,
        res: t.Any,
    ) -> list[ContentBlock]:
        def safe_serialize(obj: t.Any) -> dict[str, t.Any]:
            try:
                json.dumps(obj)
                return obj if isinstance(obj, dict) else {"content": obj}
            except (TypeError, OverflowError):
                if hasattr(obj, "model_dump"):
                    dumped = obj.model_dump()  # type: ignore
                    return _normalize_resource_uri(dumped)
                return {"content": str(obj), "type": "fallback"}

        for source in [
            getattr(res, "content", None),
            getattr(res, "structured_content", None),
            getattr(res, "data", None),
        ]:
            if source is None:
                continue
            try:
                if isinstance(source, list):
                    result = [safe_serialize(item) for item in source]
                else:
                    result = [safe_serialize(source)]

                return [
                    ToolResultBlock(
                        call_id=tool_call_id,
                        output=result,
                        is_error=False,
                        status="completed",
                    )
                ]
            except Exception as e:
                logger.exception("Serialization failure: %s", e)
                continue

        return [
            ToolResultBlock(
                call_id=tool_call_id,
                output=[
                    {
                        "content": str(res),
                        "type": "fallback",
                    }
                ],
                is_error=False,
                status="completed",
            )
        ]

    async def _list_server_tools(self, refresh: bool = False) -> list[t.Any] | None:
        """List the server's tools, from the per-node cache when ``mcp_cache_tools`` is on.

        Returns None when the server cannot be listed; a failed listing is never cached.
        """
        cached = getattr(self, "_mcp_tools_cache", None)
        if cached is not None and not refresh:
            return cached
        try:
            async with self._mcp_session():
                if not await self._client.ping():
                    return None
                server_tools = list(await self._client.list_tools())
        except Exception as e:
            logger.exception("Failed to fetch MCP tools: %s", e)
            return None
        if getattr(self, "_mcp_cache_tools", False):
            self._mcp_tools_cache = server_tools
        return server_tools

    async def refresh_mcp_tools(self) -> list[dict]:
        """Drop the cached MCP tool list and fetch it again from the server."""
        self._mcp_tools_cache = None
        if not self._client:
            return []
        await self._list_server_tools(refresh=True)
        return await self._get_mcp_tool()

    def _mcp_exposed_name(self, name: str) -> str:
        return f"{getattr(self, '_mcp_prefix', None) or ''}{name}"

    def _mcp_server_name(self, name: str) -> str:
        """The server's own name for a tool the model sees as ``name``."""
        prefix = getattr(self, "_mcp_prefix", None)
        if prefix and name.startswith(prefix):
            return name[len(prefix) :]
        return name

    async def _get_mcp_tool(self, tags: set[str] | None = None) -> list[dict]:
        """Fetch tools from the MCP client, optionally filtering by tags.

        ``mcp_tools`` is rebuilt from the server's full tool list on every call, so it stays
        bounded and dispatch works for tools hidden by the ``tags`` filter.

        Raises:
            ValueError: If an MCP tool has the same name as a local or remote tool.
        """
        tools: list[dict] = []
        if not self._client:
            return tools

        server_tools = await self._list_server_tools()
        if server_tools is None:
            return tools

        # Keep only the tools the developer allowed, so a hidden tool is also not callable.
        include = getattr(self, "_mcp_include", None)
        exclude = getattr(self, "_mcp_exclude", frozenset())
        server_tools = [
            i
            for i in server_tools
            if (include is None or i.name in include) and i.name not in exclude
        ]

        # Raised outside the try above so a collision is never swallowed.
        local_names = set(getattr(self, "_funcs", {}))
        local_names |= set(getattr(self, "remote_tool_names", []))
        names: list[str] = []
        for i in server_tools:
            exposed = self._mcp_exposed_name(i.name)
            if exposed in local_names:
                raise ValueError(
                    f"MCP tool '{exposed}' has the same name as a local or remote tool on "
                    "this ToolNode. Rename one of them."
                )
            if exposed not in names:
                names.append(exposed)

        self.mcp_tools = names

        max_desc = getattr(self, "_mcp_max_description_chars", None)
        advertised: set[str] = set()
        for i in server_tools:
            exposed = self._mcp_exposed_name(i.name)
            if exposed in advertised:
                continue
            if tags:
                meta = i.meta or {}
                tool_tags = set(meta.get("_fastmcp", {}).get("tags", []))
                if not tool_tags.intersection(tags):
                    continue
            advertised.add(exposed)
            description = i.description
            if (
                max_desc is not None
                and isinstance(description, str)
                and len(description) > max_desc
            ):
                description = description[:max_desc] + "..."
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": exposed,
                        "description": description,
                        "parameters": i.inputSchema,
                    },
                }
            )

        return tools

    async def _mcp_execute(  # noqa: PLR0915
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        callback_mgr: CallbackManager,
    ) -> Message:
        context = CallbackContext(
            invocation_type=InvocationType.MCP,
            node_name="ToolNode",
            function_name=name,
            metadata={
                "tool_call_id": tool_call_id,
                "args": args,
                "config": config,
                "mcp_client": bool(self._client),
            },
        )

        meta = {
            "function_name": name,
            "function_argument": args,
            "tool_call_id": tool_call_id,
        }

        event = EventModel.default(
            base_config=config,
            data={
                "tool_call_id": tool_call_id,
                "args": args,
                "function_name": name,
                "is_mcp": True,
            },
            content_type=[ContentType.TOOL_CALL],
            event=Event.TOOL_EXECUTION,
        )
        event.event_type = EventType.PROGRESS
        event.node_name = "ToolNode"
        publish_event(event)

        input_data = {**args}

        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            event.event_type = EventType.UPDATE
            event.metadata["status"] = "before_invoke_complete Invoke MCP"
            publish_event(event)

            if not self._client:
                error_result = Message.tool_message(
                    content=[
                        ErrorBlock(
                            message="No MCP client configured",
                        ),
                        ToolResultBlock(
                            call_id=tool_call_id,
                            output="No MCP client configured",
                            is_error=True,
                            status="failed",
                        ),
                    ],
                    meta=meta,
                )
                res = await callback_mgr.execute_after_invoke(context, input_data, error_result)
                event.event_type = EventType.ERROR
                event.metadata["error"] = "No MCP client configured"
                publish_event(event)
                return res

            async with self._mcp_session():
                if not await self._client.ping():
                    error_result = Message.tool_message(
                        content=[
                            ErrorBlock(message="MCP Server not available. Ping failed."),
                            ToolResultBlock(
                                call_id=tool_call_id,
                                output="MCP Server not available. Ping failed.",
                                is_error=True,
                                status="failed",
                            ),
                        ],
                        meta=meta,
                    )
                    event.event_type = EventType.ERROR
                    event.metadata["error"] = "MCP server not available, ping failed"
                    publish_event(event)
                    return await callback_mgr.execute_after_invoke(
                        context, input_data, error_result
                    )

                call_kwargs: dict[str, t.Any] = {}
                if self._pass_user_info_to_mcp:
                    mcp_user_info = config.get("user")
                    if not (mcp_user_info and isinstance(mcp_user_info, dict)):
                        user_id = config.get("user_id")
                        mcp_user_info = {"user_id": user_id} if user_id else None
                    if mcp_user_info:
                        # Request metadata, not a tool argument: a strict input schema
                        # would reject an extra "user" argument, and a lax one would
                        # hand every claim to the tool body.
                        params = inspect.signature(self._client.call_tool).parameters
                        if "meta" in params or any(
                            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                        ):
                            call_kwargs["meta"] = {"user": mcp_user_info}
                        else:
                            logger.warning(
                                "pass_user_info_to_mcp needs a fastmcp client whose "
                                "call_tool accepts meta; user info was not sent."
                            )

                res: t.Any = await self._client.call_tool(
                    self._mcp_server_name(name), input_data, **call_kwargs
                )

                final_res = self._serialize_result(tool_call_id, res)

                result = Message.tool_message(
                    content=final_res,
                    meta=meta,
                )

                res = await callback_mgr.execute_after_invoke(context, input_data, result)
                event.event_type = EventType.END
                # Publish what after_invoke returned, not the raw result it replaced.
                published = res if hasattr(res, "model_dump") else result
                event.data["message"] = published.model_dump()
                event.metadata["status"] = "MCP tool execution complete"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return res

        except Exception as e:
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                event.event_type = EventType.END
                event.data["message"] = recovery_result.model_dump()
                event.metadata["status"] = "MCP tool execution complete, with recovery"
                event.content_type = [ContentType.TOOL_RESULT, ContentType.MESSAGE]
                publish_event(event)
                return recovery_result

            # The raw text can hold server internals; the model and the event get a reference.
            error_text = generic_tool_failure(name, tool_call_id, e, log=logger)

            event.event_type = EventType.END
            event.data["error"] = error_text
            event.metadata["status"] = "MCP tool execution complete, with recovery"
            event.content_type = [ContentType.TOOL_RESULT, ContentType.ERROR]
            publish_event(event)

            return Message.tool_message(
                content=[
                    ToolResultBlock(
                        call_id=tool_call_id,
                        output=error_text,
                        is_error=True,
                        status="failed",
                    ),
                    ErrorBlock(message=error_text),
                ],
                meta=meta,
            )
