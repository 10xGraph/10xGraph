"""Google embedding service for vector stores."""

import os

from .base_embedding import BaseEmbedding


HAS_GOOGLE = False


try:
    from google import genai

    HAS_GOOGLE = True
except ImportError:
    pass


class GoogleEmbedding(BaseEmbedding):
    """Google embedding service using Gemini embedding models.

    ``document_task_type`` is sent when embedding text to store (``aembed``,
    ``aembed_batch``) and ``query_task_type`` when embedding a search query
    (``aembed_query``). Pass ``None`` to send no task type. Values are the Gemini
    task type strings, e.g. ``RETRIEVAL_DOCUMENT``, ``RETRIEVAL_QUERY``,
    ``SEMANTIC_SIMILARITY``.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
        output_dimensionality: int | None = None,
        document_task_type: str | None = "RETRIEVAL_DOCUMENT",
        query_task_type: str | None = "RETRIEVAL_QUERY",
    ) -> None:
        if not HAS_GOOGLE:
            raise ImportError(
                "The 'google-genai' package is required for GoogleEmbedding. "
                "Please install it via 'pip install google-genai'."
            )
        self.model = model
        self._output_dimensionality = output_dimensionality
        self._document_task_type = document_task_type
        self._query_task_type = query_task_type

        if api_key:
            self.api_key = api_key
        elif "GOOGLE_API_KEY" in os.environ:
            self.api_key = os.environ["GOOGLE_API_KEY"]
        elif "GEMINI_API_KEY" in os.environ:
            self.api_key = os.environ["GEMINI_API_KEY"]
        else:
            raise ValueError(
                "Google API key must be provided via parameter, GOOGLE_API_KEY or GEMINI_API_KEY"
            )

        # Create client
        self.client = genai.Client(api_key=self.api_key)

    def _build_config(self, task_type: str | None):
        from google.genai import types

        kwargs: dict = {}
        if self._output_dimensionality:
            kwargs["output_dimensionality"] = self._output_dimensionality
        if task_type:
            kwargs["task_type"] = task_type
        return types.EmbedContentConfig(**kwargs) if kwargs else None

    async def _embed(self, texts: list[str], task_type: str | None) -> list[list[float]]:
        try:
            result = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts,
                config=self._build_config(task_type),
            )

            if not result.embeddings or len(result.embeddings) == 0:
                raise ValueError("No embeddings returned from Google API")

            return [embedding.values for embedding in result.embeddings]  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}") from e

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate document embeddings for a list of texts."""
        return await self._embed(texts, self._document_task_type)

    async def aembed(self, text: str) -> list[float]:
        """Generate a document embedding for a single text (used when storing)."""
        return (await self._embed([text], self._document_task_type))[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Generate a query embedding for a single text (used when searching)."""
        return (await self._embed([text], self._query_task_type))[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension based on model."""
        # If custom dimensionality is set, return that
        if self._output_dimensionality:
            return self._output_dimensionality

        # Default dimensions for Google models
        model_dimensions = {
            "gemini-embedding-001": 3072,
        }
        if self.model in model_dimensions:
            return model_dimensions[self.model]

        # Default to 768 for Google models
        return 768
