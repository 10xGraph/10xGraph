DEFAULT_DENIED_REASON = "This tool call is not allowed."


class ToolDenied(Exception):  # noqa: N818 - a policy decision, named for what it says
    """Raised to block a tool call by policy.

    Raise it from a ``before_invoke`` callback for ``InvocationType.TOOL`` or
    ``InvocationType.MCP`` (or from the tool itself). The ToolNode then does not run the tool,
    skips ``on_error`` recovery (a recovery result can never turn a block into a success),
    logs a warning with the tool name, user id and reason, and answers the model with a failed
    tool result: ``"Denied by policy: <tool>. <reason>"``. The tool event is published with
    status ``denied`` and the trace span is marked ``tenxgraph.tool.denied``.

    ``reason`` goes to the model, the logs and the events, so keep it short and free of
    secrets and argument values.

    Example:
        ```python
        from tenxgraph.core.exceptions import ToolDenied
        from tenxgraph.utils import CallbackManager, InvocationType


        async def no_deletes(context, input_data):
            if context.function_name.startswith("delete_"):
                raise ToolDenied("Deleting is turned off for this assistant.")
            return input_data


        manager = CallbackManager()
        manager.register_before_invoke(InvocationType.TOOL, no_deletes)
        app = graph.compile(callback_manager=manager)
        ```

    Attributes:
        reason: Why the call was denied, safe to show to the model.
        tool_name: The tool that was denied. The ToolNode fills it in when it is not given.
    """

    def __init__(self, reason: str = DEFAULT_DENIED_REASON, *, tool_name: str | None = None):
        self.reason = reason or DEFAULT_DENIED_REASON
        self.tool_name = tool_name
        super().__init__(self.reason)
