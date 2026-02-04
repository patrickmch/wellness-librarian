"""
Abstract base class for embedding providers.

Defines the interface that all embedding providers must implement,
enabling pluggable embedding backends for A/B testing and flexibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    token_count: int  # Estimated token count for cost tracking


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implementations must provide methods for:
    - Single text embedding
    - Batch text embedding
    - Token counting
    - Cost estimation
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier used by this provider."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions for this model."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 100
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def embed_text_async(self, text: str) -> list[float]:
        """
        Generate embedding for a single text asynchronously.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    async def embed_texts_async(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
        max_concurrent: int = 5,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts asynchronously.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count or estimate tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count (exact or estimated)
        """
        pass

    @abstractmethod
    def estimate_cost(self, texts: Sequence[str]) -> float:
        """
        Estimate cost to embed texts.

        Args:
            texts: List of texts to embed

        Returns:
            Estimated cost in USD
        """
        pass
