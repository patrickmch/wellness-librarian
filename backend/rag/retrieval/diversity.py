"""
Diversity filtering for retrieval results.

Implements:
- Per-video deduplication (max N chunks per video)
- Maximal Marginal Relevance (MMR) for semantic diversity
"""

import logging
from typing import Any, Callable, Sequence

import numpy as np

from backend.config import get_settings

logger = logging.getLogger(__name__)


def dedupe_by_video(
    results: Sequence[Any],
    video_id_getter: Callable[[Any], str],
    max_per_video: int | None = None,
) -> list[Any]:
    """
    Deduplicate results to limit chunks per video.

    This prevents a single long video from dominating results.
    Results are expected to be pre-sorted by relevance.

    Args:
        results: List of result objects (ordered by relevance)
        video_id_getter: Function to extract video_id from result
        max_per_video: Maximum chunks per video (default from settings)

    Returns:
        Filtered list maintaining original order
    """
    settings = get_settings()
    max_per_video = max_per_video or settings.max_chunks_per_video

    video_counts: dict[str, int] = {}
    filtered = []

    for result in results:
        video_id = video_id_getter(result)
        current_count = video_counts.get(video_id, 0)

        if current_count < max_per_video:
            filtered.append(result)
            video_counts[video_id] = current_count + 1

    logger.debug(
        f"Per-video dedup: {len(results)} -> {len(filtered)} "
        f"(max {max_per_video}/video, {len(video_counts)} videos)"
    )

    return filtered


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def maximal_marginal_relevance(
    query_embedding: list[float] | np.ndarray,
    results: Sequence[Any],
    embedding_getter: Callable[[Any], list[float] | np.ndarray],
    lambda_param: float | None = None,
    k: int | None = None,
) -> list[Any]:
    """
    Apply Maximal Marginal Relevance (MMR) for diverse result selection.

    MMR balances relevance to query with diversity among selected results.
    Formula: MMR = λ * sim(d, q) - (1-λ) * max(sim(d, selected))

    Args:
        query_embedding: The query embedding vector
        results: List of result objects with embeddings
        embedding_getter: Function to extract embedding from result
        lambda_param: Balance factor (1.0 = pure relevance, 0.0 = pure diversity)
        k: Number of results to return

    Returns:
        List of k most diverse results
    """
    if not results:
        return []

    settings = get_settings()
    lambda_param = lambda_param if lambda_param is not None else settings.mmr_lambda
    k = k or len(results)

    # Convert to numpy arrays
    query_vec = np.array(query_embedding)
    result_vecs = [np.array(embedding_getter(r)) for r in results]

    # Pre-compute query similarities
    query_sims = [cosine_similarity(query_vec, vec) for vec in result_vecs]

    # Greedy MMR selection
    selected_indices: list[int] = []
    remaining_indices = list(range(len(results)))

    while len(selected_indices) < k and remaining_indices:
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            # Relevance term
            relevance = lambda_param * query_sims[idx]

            # Diversity term (max similarity to already selected)
            if selected_indices:
                max_sim_to_selected = max(
                    cosine_similarity(result_vecs[idx], result_vecs[sel_idx])
                    for sel_idx in selected_indices
                )
                diversity = (1 - lambda_param) * max_sim_to_selected
            else:
                diversity = 0.0

            mmr_score = relevance - diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    logger.debug(
        f"MMR selection: {len(results)} -> {len(selected_indices)} "
        f"(λ={lambda_param})"
    )

    return [results[i] for i in selected_indices]


def apply_diversity_filters(
    results: Sequence[Any],
    query_embedding: list[float] | np.ndarray | None = None,
    video_id_getter: Callable[[Any], str] | None = None,
    embedding_getter: Callable[[Any], list[float] | np.ndarray] | None = None,
    max_per_video: int | None = None,
    use_mmr: bool | None = None,
    mmr_lambda: float | None = None,
    final_k: int | None = None,
) -> list[Any]:
    """
    Apply diversity filters to retrieval results.

    Combines per-video deduplication and MMR for comprehensive diversity.

    Args:
        results: Initial retrieval results (ordered by relevance)
        query_embedding: Query embedding for MMR (required if use_mmr=True)
        video_id_getter: Function to extract video_id (required for per-video dedup)
        embedding_getter: Function to extract embedding (required if use_mmr=True)
        max_per_video: Maximum chunks per video
        use_mmr: Whether to apply MMR (default from settings)
        mmr_lambda: MMR lambda parameter
        final_k: Final number of results to return

    Returns:
        Filtered and potentially reordered results
    """
    if not results:
        return []

    settings = get_settings()
    use_mmr = use_mmr if use_mmr is not None else settings.enable_mmr

    filtered = list(results)

    # Step 1: Per-video deduplication
    if video_id_getter is not None:
        filtered = dedupe_by_video(
            filtered,
            video_id_getter,
            max_per_video,
        )

    # Step 2: MMR for semantic diversity
    if use_mmr and query_embedding is not None and embedding_getter is not None:
        filtered = maximal_marginal_relevance(
            query_embedding,
            filtered,
            embedding_getter,
            lambda_param=mmr_lambda,
            k=final_k or len(filtered),
        )
    elif final_k is not None:
        filtered = filtered[:final_k]

    return filtered
