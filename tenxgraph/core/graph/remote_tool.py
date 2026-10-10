"""Validated configuration for tools executed by a remote client."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _empty_parameters() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "required": []}


class RemoteToolConfig(BaseModel):
    """Trusted, model-facing schema for a client-executed tool.

    ``node`` is the concise configuration spelling. ``node_name`` remains an
    accepted input for callers already using the API wire-model spelling.
    Unknown fields are rejected so configuration typos fail at startup.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    node_name: str = Field(alias="node", min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=_empty_parameters)

    @field_validator("node_name", "name", "description")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        parameters = dict(value)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.setdefault("required", [])

        if parameters["type"] != "object":
            raise ValueError("remote tool parameters.type must be 'object'")
        if not isinstance(parameters["properties"], dict):
            raise ValueError("remote tool parameters.properties must be an object")
        if not isinstance(parameters["required"], list) or not all(
            isinstance(item, str) for item in parameters["required"]
        ):
            raise ValueError("remote tool parameters.required must be a list of strings")
        return parameters

    def to_tool_schema(self) -> dict[str, Any]:
        """Return the OpenAI-compatible schema consumed by ``ToolNode``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
