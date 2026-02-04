"""
Voyage AI reranking implementation.

Uses Voyage's rerank-2 model to re-score initial retrieval results
for improved relevance ordering.
"""

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A single reranked result."""

    index: int  # Original index in input list
    score: float  # Relevance score from reranker (0-1)
    document: Any  # The original document object


class VoyageReranker:
    """
    Voyage AI reranker using rerank-2 model.

    Reranking is a second-stage retrieval step that takes initial
    results from vector search and re-scores them using a more
    sophisticated cross-encoder model for better relevance ordering.

    Benefits:
    - Better relevance scoring than embedding similarity
    - Can consider query-document interactions
    - Particularly good at filtering out false positives
    """

    # Pricing: ~$0.05 per 1M tokens for rerank-2
    COST_PER_MILLION_TOKENS = 0.05

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-2",
    ):
        """
        Initialize Voyage reranker.

        Args:
            api_key: Voyage API key (defaults to settings)
            model: Rerank model name (default: rerank-2)
        """
        settings = get_settings()
        self._api_key = api_key or settings.voyage_api_key
        self._model = model

        if not self._api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = voyageai.Client(api_key=self._api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_n: Return only top N results (default: all)

        Returns:
            List of RerankResult sorted by relevance score descending
        """
        if not documents:
            return []

        settings = get_settings()
        top_n = top_n or settings.rerank_top_n or len(documents)

        result = self._client.rerank(
            query=query,
            documents=list(documents),
            model=self._model,
            top_k=min(top_n, len(documents)),
        )

        return [
            RerankResult(
                index=r.index,
                score=r.relevance_score,
                document=documents[r.index],
            )
            for r in result.results
        ]

    def rerank_with_metadata(
        self,
        query: str,
        documents: Sequence[tuple[str, Any]],
        top_n: int | None = None,
    ) -> list[tuple[RerankResult, Any]]:
        """
        Rerank documents while preserving associated metadata.

        Args:
            query: The search query
            documents: List of (text, metadata) tuples
            top_n: Return only top N results

        Returns:
            List of (RerankResult, metadata) tuples sorted by score
        """
        if not documents:
            return []

        texts = [doc[0] for doc in documents]
        metadata = [doc[1] for doc in documents]

        results = self.rerank(query, texts, top_n)

        return [
            (result, metadata[result.index])
            for result in results
        ]

    def rerank_objects(
        self,
        query: str,
        objects: Sequence[Any],
        text_key: str = "text",
        top_n: int | None = None,
    ) -> list[tuple[float, Any]]:
        """
        Rerank objects using a text field for comparison.

        Convenience method for reranking dicts or objects with a text attribute.

        Args:
            query: The search query
            objects: List of objects to rerank
            text_key: Key/attribute name for text content
            top_n: Return only top N results

        Returns:
            List of (score, object) tuples sorted by score descending
        """
        if not objects:
            return []

        # Extract text from objects
        texts = []
        for obj in objects:
            if isinstance(obj, dict):
                texts.append(obj.get(text_key, ""))
            elif hasattr(obj, text_key):
                texts.append(getattr(obj, text_key))
            else:
                texts.append(str(obj))

        results = self.rerank(query, texts, top_n)

        return [
            (result.score, objects[result.index])
            for result in results
        ]

    def estimate_cost(self, query: str, documents: Sequence[str]) -> float:
        """
        Estimate cost for a reranking operation.

        Args:
            query: The search query
            documents: Documents to rerank

        Returns:
            Estimated cost in USD
        """
        # Rough token estimate: query + all docs
        total_chars = len(query) + sum(len(d) for d in documents)
        total_tokens = total_chars // 4
        return (total_tokens / 1_000_000) * self.COST_PER_MILLION_TOKENS


# Convenience function
def get_reranker() -> VoyageReranker:
    """Get a VoyageReranker instance."""
    return VoyageReranker()
