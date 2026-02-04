"""
Retrieval package for advanced retrieval strategies.

Provides:
- Diversity filtering (per-video dedup, MMR)
- Parent-child retrieval
"""

from backend.rag.retrieval.diversity import (
    dedupe_by_video,
    maximal_marginal_relevance,
    apply_diversity_filters,
)
from backend.rag.retrieval.parent_child import ParentChildRetriever

__all__ = [
    # Diversity
    "dedupe_by_video",
    "maximal_marginal_relevance",
    "apply_diversity_filters",
    # Retrieval
    "ParentChildRetriever",
]
