"""
Voyage AI embedding provider implementation.

Provides high-quality embeddings using Voyage AI's models,
particularly optimized for retrieval tasks.
"""

import asyncio
import logging
import time
from typing import Sequence

import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.rag.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class VoyageEmbeddingProvider(EmbeddingProvider):
    """
    Voyage AI embedding provider.

    Uses Voyage's embedding models which are specifically optimized
    for retrieval tasks and often outperform OpenAI embeddings.

    Default model: voyage-3 (1024 dimensions)
    """

    # Pricing: $0.06 per 1M tokens for voyage-3
    COST_PER_MILLION_TOKENS = 0.06

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize Voyage embedding provider.

        Args:
            api_key: Voyage API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        settings = get_settings()
        self._api_key = api_key or settings.voyage_api_key
        self._model = model or settings.voyage_embedding_model
        self._dimensions = settings.voyage_embedding_dimensions

        if not self._api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize clients
        self._sync_client = voyageai.Client(api_key=self._api_key)
        self._async_client = voyageai.AsyncClient(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        result = self._sync_client.embed(
            texts=[text],
            model=self._model,
            input_type="document",  # Use "query" for query embedding
        )
        return result.embeddings[0]

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a query.

        Voyage recommends using input_type="query" for queries
        and input_type="document" for documents being indexed.
        """
        result = self._sync_client.embed(
            texts=[text],
            model=self._model,
            input_type="query",
        )
        return result.embeddings[0]

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _embed_single_batch(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[list[float]]:
        """Embed a single batch with retry logic."""
        result = self._sync_client.embed(
            texts=texts,
            model=self._model,
            input_type=input_type,
        )
        return result.embeddings

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 50,
        input_type: str = "document",
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Voyage supports up to 128 texts per request, but we use smaller
        batches to avoid rate limits.

        Args:
            texts: List of texts to embed
            batch_size: Texts per API call (default 50 for rate limiting)
            input_type: "document" for indexing, "query" for searching

        Returns:
            List of embedding vectors
        """
        # Use smaller batches to avoid rate limits
        batch_size = min(batch_size, 50)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])

            embeddings = self._embed_single_batch(batch, input_type)
            all_embeddings.extend(embeddings)

            # Longer pause between batches for rate limiting
            if i + batch_size < len(texts):
                time.sleep(1.0)

        return all_embeddings

    async def embed_text_async(self, text: str) -> list[float]:
        """Generate embedding for a single text asynchronously."""
        result = await self._async_client.embed(
            texts=[text],
            model=self._model,
            input_type="document",
        )
        return result.embeddings[0]

    async def embed_query_async(self, text: str) -> list[float]:
        """Generate query embedding asynchronously."""
        result = await self._async_client.embed(
            texts=[text],
            model=self._model,
            input_type="query",
        )
        return result.embeddings[0]

    async def embed_texts_async(
        self,
        texts: Sequence[str],
        batch_size: int = 128,
        max_concurrent: int = 5,
        input_type: str = "document",
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        batch_size = min(batch_size, 128)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_batch(
            batch: list[str],
            start_idx: int
        ) -> tuple[int, list[list[float]]]:
            async with semaphore:
                result = await self._async_client.embed(
                    texts=batch,
                    model=self._model,
                    input_type=input_type,
                )
                return start_idx, result.embeddings

        # Create tasks for all batches
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            tasks.append(embed_batch(batch, i))

        # Execute all batches
        results = await asyncio.gather(*tasks)

        # Reconstruct in order
        all_embeddings = [None] * len(texts)
        for start_idx, embeddings in results:
            for j, emb in enumerate(embeddings):
                all_embeddings[start_idx + j] = emb

        return all_embeddings

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Voyage uses a similar tokenizer to OpenAI, so ~4 chars per token
        is a reasonable approximation.
        """
        # For more accuracy, could use tiktoken
        return len(text) // 4

    def estimate_cost(self, texts: Sequence[str]) -> float:
        """
        Estimate cost to embed texts.

        Pricing: $0.06 per 1M tokens for voyage-3
        """
        total_tokens = sum(self.count_tokens(t) for t in texts)
        return (total_tokens / 1_000_000) * self.COST_PER_MILLION_TOKENS
