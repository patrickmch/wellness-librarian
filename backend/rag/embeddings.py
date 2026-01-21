"""
OpenAI embeddings integration.
Handles embedding generation with rate limiting and batching.
"""

import asyncio
import logging
from typing import Sequence

from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings

logger = logging.getLogger(__name__)


def get_openai_client() -> OpenAI:
    """Get synchronous OpenAI client."""
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def get_async_openai_client() -> AsyncOpenAI:
    """Get asynchronous OpenAI client."""
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_text(text: str) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats
    """
    settings = get_settings()
    client = get_openai_client()

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )

    return response.data[0].embedding


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_texts(texts: Sequence[str], batch_size: int = 100) -> list[list[float]]:
    """
    Generate embeddings for multiple texts with batching.

    OpenAI supports up to 2048 texts per request, but we batch smaller
    for better error handling and memory management.

    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call

    Returns:
        List of embedding vectors
    """
    settings = get_settings()
    client = get_openai_client()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model=settings.embedding_model,
            input=list(batch),
        )

        # Sort by index to ensure order matches input
        batch_embeddings = [None] * len(batch)
        for item in response.data:
            batch_embeddings[item.index] = item.embedding

        all_embeddings.extend(batch_embeddings)

        # Brief pause between batches to respect rate limits
        if i + batch_size < len(texts):
            import time
            time.sleep(0.1)

    return all_embeddings


async def embed_text_async(text: str) -> list[float]:
    """
    Generate embedding for a single text asynchronously.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats
    """
    settings = get_settings()
    client = get_async_openai_client()

    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )

    return response.data[0].embedding


async def embed_texts_async(
    texts: Sequence[str],
    batch_size: int = 100,
    max_concurrent: int = 5,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts asynchronously with concurrency control.

    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of embedding vectors
    """
    settings = get_settings()
    client = get_async_openai_client()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_batch(batch: list[str], start_idx: int) -> tuple[int, list[list[float]]]:
        async with semaphore:
            response = await client.embeddings.create(
                model=settings.embedding_model,
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


def count_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Rough approximation: ~4 characters per token for English text.
    For accurate counts, use tiktoken library.

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def estimate_embedding_cost(texts: Sequence[str]) -> float:
    """
    Estimate cost to embed texts using text-embedding-3-small.

    Pricing: $0.02 per 1M tokens

    Args:
        texts: List of texts to embed

    Returns:
        Estimated cost in USD
    """
    total_tokens = sum(count_tokens(t) for t in texts)
    cost_per_million = 0.02
    return (total_tokens / 1_000_000) * cost_per_million
