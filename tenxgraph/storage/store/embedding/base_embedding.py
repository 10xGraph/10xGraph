from abc import ABC, abstractmethod

from agentflow.utils.callable_utils import run_coroutine


class BaseEmbedding(ABC):
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Synchronous wrapper for `aembed_batch` that runs the async implementation."""
        return run_coroutine(self.aembed_batch(texts))

    @abstractmethod
    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        # pragma: no cover

    def embed(self, text: str) -> list[float]:
        """Synchronous wrapper for `aembed` that runs the async implementation."""
        return run_coroutine(self.aembed(text))

    @abstractmethod
    async def aembed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Synchronous wrapper for that runs the async implementation."""
        raise NotImplementedError
