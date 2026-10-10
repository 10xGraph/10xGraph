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
    """Google embedding service using Gemini embedding models."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
        output_dimensionality: int | None = None,
    ) -> None:
        if not HAS_GOOGLE:
            raise ImportError(
                "The 'google-genai' package is required for GoogleEmbedding. "
                "Please install it via 'pip install google-genai'."
            )
        self.model = model
        self._output_dimensionality = output_dimensionality

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

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        from google.genai import types

        try:
            config = None
            if self._output_dimensionality:
                config = types.EmbedContentConfig(output_dimensionality=self._output_dimensionality)

            result = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts,
                config=config,
            )

            if not result.embeddings or len(result.embeddings) == 0:
                raise ValueError("No embeddings returned from Google API")

            return [embedding.values for embedding in result.embeddings]  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}") from e

    async def aembed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        from google.genai import types

        try:
            config = None
            if self._output_dimensionality:
                config = types.EmbedContentConfig(output_dimensionality=self._output_dimensionality)

            result = await self.client.aio.models.embed_content(
                model=self.model,
                contents=[text],
                config=config,
            )

            if not result.embeddings or len(result.embeddings) == 0:
                raise ValueError("No embeddings returned from Google API")

            return result.embeddings[0].values
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}") from e

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
