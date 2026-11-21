class ModelNotFoundError(ValueError):
    """Raised when the run config names a model that is not in the Agent's ``model_registry``.

    The message lists the registered names and carries no keys or URLs, so it is safe to
    return to the caller (the API answers 422).
    """

    def __init__(self, name: object, node_name: str | None, registered: list[str]):
        self.name = name
        self.node_name = node_name
        self.registered = registered
        node = node_name or "unknown"
        super().__init__(
            f"model {name!r} is not registered on agent node '{node}'; "
            f"registered: {', '.join(registered)}"
        )
