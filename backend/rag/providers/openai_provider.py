"""
OpenAI embedding provider implementation.

Wraps the existing OpenAI embedding functionality to conform to the
EmbeddingProvider interface, enabling pluggable embedding backends.
"""

import asyncio
import logging
import time
from typing import Sequence

from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.rag.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding-3-small by default.

    Wraps the existing embedding functionality while conforming to the
    EmbeddingProvider interface for pluggable backends.
    """

    # Pricing: $0.02 per 1M tokens for text-embedding-3-small
    COST_PER_MILLION_TOKENS = 0.02

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
            dimensions: Embedding dimensions (defaults to settings)
        """
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dimensions

        # Lazy-initialized clients
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_sync_client(self) -> OpenAI:
        """Get or create synchronous client."""
        if self._sync_client is None:
            self._sync_client = OpenAI(api_key=self._api_key)
        return self._sync_client

    def _get_async_client(self) -> AsyncOpenAI:
        """Get or create asynchronous client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        return self._async_client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        client = self._get_sync_client()

        response = client.embeddings.create(
            model=self._model,
            input=text,
        )

        return response.data[0].embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 100
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts with batching."""
        client = self._get_sync_client()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = client.embeddings.create(
                model=self._model,
                input=list(batch),
            )

            # Sort by index to ensure order matches input
            batch_embeddings = [None] * len(batch)
            for item in response.data:
                batch_embeddings[item.index] = item.embedding

            all_embeddings.extend(batch_embeddings)

            # Brief pause between batches for rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings

    async def embed_text_async(self, text: str) -> list[float]:
        """Generate embedding for a single text asynchronously."""
        client = self._get_async_client()

        response = await client.embeddings.create(
            model=self._model,
            input=text,
        )

        return response.data[0].embedding

    async def embed_texts_async(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
        max_concurrent: int = 5,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        client = self._get_async_client()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_batch(
            batch: list[str],
            start_idx: int
        ) -> tuple[int, list[list[float]]]:
            async with semaphore:
                response = await client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
                # Sort by index within batch
                embeddings = [None] * len(batch)
                for item in response.data:
                    embeddings[item.index] = item.embedding
                return start_idx, embeddings

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
        Estimate token count for text.

        Uses rough approximation of ~4 characters per token.
        For accurate counts, we could use tiktoken.
        """
        return len(text) // 4

    def estimate_cost(self, texts: Sequence[str]) -> float:
        """
        Estimate cost to embed texts.

        Pricing: $0.02 per 1M tokens for text-embedding-3-small
        """
        total_tokens = sum(self.count_tokens(t) for t in texts)
        return (total_tokens / 1_000_000) * self.COST_PER_MILLION_TOKENS
