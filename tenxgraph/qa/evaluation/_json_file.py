"""JSON file load and save shared by the eval models (EvalConfig, EvalSet, EvalReport)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import BaseModel


class _JsonFileModel(BaseModel):
    """A pydantic model with ``from_file`` and ``to_file`` for UTF-8 JSON files.

    Set ``_json_default`` to a ``json.dump`` ``default`` function for values JSON cannot encode.
    """

    _json_default: ClassVar[Any] = None

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Load the model from a JSON file.

        Args:
            path: Path to the JSON file.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_file(self, path: str) -> None:
        """Save the model to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, default=self._json_default)
