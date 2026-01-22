"""
Retrieval module for semantic search over transcripts.
Provides high-level search functions with result formatting.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from backend.config import get_settings
from backend.rag.vectorstore import search, get_collection_stats
from backend.ingestion.metadata import format_source_citation

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with content and metadata."""
    chunk_id: str
    text: str
    score: float  # Similarity score (1 - distance for cosine)
    metadata: dict

    @property
    def title(self) -> str:
        return self.metadata.get("title", "Unknown")

    @property
    def category(self) -> str:
        return self.metadata.get("category", "Unknown")

    @property
    def video_url(self) -> str:
        return self.metadata.get("video_url", "")

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def citation(self) -> str:
        return format_source_citation(self.metadata)


@dataclass
class RetrievalResponse:
    """Collection of retrieval results with context."""
    query: str
    results: list[RetrievalResult]
    total_results: int

    def get_context_text(self, max_chunks: int = 8) -> str:
        """
        Format results as context for LLM.

        Args:
            max_chunks: Maximum chunks to include

        Returns:
            Formatted context string
        """
        if not self.results:
            return "No relevant content found."

        context_parts = []
        for i, result in enumerate(self.results[:max_chunks]):
            context_parts.append(
                f"[Source {i+1}: {result.title} ({result.category})]\n{result.text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_sources(self) -> list[dict]:
        """
        Get unique sources with metadata.

        Returns:
            List of source dicts with title, category, url, source
        """
        seen = set()
        sources = []

        for result in self.results:
            video_id = result.metadata.get("video_id")
            if video_id and video_id not in seen:
                seen.add(video_id)
                sources.append({
                    "title": result.title,
                    "category": result.category,
                    "video_url": result.video_url,
                    "duration": result.metadata.get("duration", ""),
                    "video_id": video_id,
                    "source": result.source,
                })

        return sources


def retrieve(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
    min_score: float | None = None,
) -> RetrievalResponse:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: Search query
        top_k: Number of results (default from settings)
        category: Filter by category
        min_score: Minimum similarity score (0-1)

    Returns:
        RetrievalResponse with results
    """
    settings = get_settings()
    top_k = top_k or settings.default_top_k
    min_score = min_score or settings.similarity_threshold

    # Build where filter
    where = None
    if category:
        where = {"category": category}

    # Execute search
    raw_results = search(query=query, n_results=top_k, where=where)

    # Convert to RetrievalResult objects
    results = []
    for i, (chunk_id, doc, meta, dist) in enumerate(zip(
        raw_results["ids"],
        raw_results["documents"],
        raw_results["metadatas"],
        raw_results["distances"],
    )):
        # Convert cosine distance to similarity score
        score = 1.0 - dist

        # Filter by minimum score
        if score >= min_score:
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=doc,
                score=score,
                metadata=meta,
            ))

    return RetrievalResponse(
        query=query,
        results=results,
        total_results=len(results),
    )


def retrieve_by_category(
    query: str,
    categories: list[str],
    top_k_per_category: int = 3,
) -> RetrievalResponse:
    """
    Retrieve from multiple specific categories.

    Args:
        query: Search query
        categories: List of categories to search
        top_k_per_category: Results per category

    Returns:
        Combined RetrievalResponse
    """
    all_results = []

    for category in categories:
        response = retrieve(query=query, top_k=top_k_per_category, category=category)
        all_results.extend(response.results)

    # Sort by score and dedupe
    all_results.sort(key=lambda r: r.score, reverse=True)

    seen_ids = set()
    unique_results = []
    for result in all_results:
        if result.chunk_id not in seen_ids:
            seen_ids.add(result.chunk_id)
            unique_results.append(result)

    return RetrievalResponse(
        query=query,
        results=unique_results,
        total_results=len(unique_results),
    )


def get_available_categories() -> list[str]:
    """Get list of categories in the collection."""
    stats = get_collection_stats()
    return sorted(stats.get("categories", {}).keys())


def get_stats() -> dict:
    """Get retrieval system statistics."""
    return get_collection_stats()
