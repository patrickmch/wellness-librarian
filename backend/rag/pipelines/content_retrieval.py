"""
Content Retrieval pipeline for AI agent post creation.

Pure retrieval — no LLM generation. Returns transcript chunks + video
metadata from free YouTube videos so an external AI agent can craft
social media posts independently.

Pipeline steps:
1. Call ParentChildRetriever.retrieve() with source="youtube"
2. Map results to ContentItem objects
3. Count total YouTube videos available (for context)
4. Return ContentResponse
"""

import logging

from backend.config import get_settings
from backend.rag.retrieval.parent_child import ParentChildRetriever

logger = logging.getLogger(__name__)


def retrieve_content(
    query: str,
    top_k: int = 5,
    category: str | None = None,
) -> dict:
    """
    Retrieve YouTube transcript content for a given topic.

    Args:
        query: Topic to search for
        top_k: Number of content pieces to return
        category: Optional category filter

    Returns:
        Dict with query, results (list of content items),
        total_results, and youtube_videos_available count.
    """
    settings = get_settings()
    retriever = ParentChildRetriever()

    # Retrieve with source filter locked to YouTube
    response = retriever.retrieve(
        query=query,
        final_top_k=top_k,
        category=category,
        source="youtube",
    )

    # Map results to content items
    results = []
    for r in response.results:
        excerpt = r.text[:500] + "..." if len(r.text) > 500 else r.text
        results.append({
            "video_title": r.title,
            "video_url": r.video_url,
            "video_id": r.video_id,
            "category": r.category,
            "duration": r.parent.duration,
            "content": r.text,
            "relevance_score": round(r.score, 4),
            "excerpt": excerpt,
        })

    # Count YouTube videos available
    youtube_count = _count_youtube_videos()

    return {
        "query": query,
        "results": results,
        "total_results": len(results),
        "youtube_videos_available": youtube_count,
    }


def _count_youtube_videos() -> int:
    """Count distinct YouTube videos in the store."""
    settings = get_settings()

    if settings.store_backend == "supabase":
        from backend.rag.stores.supabase_store import get_supabase_store
        store = get_supabase_store()
        try:
            with store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(DISTINCT video_id) FROM parent_chunks "
                        "WHERE source = 'youtube'"
                    )
                    return cur.fetchone()[0]
        except Exception:
            logger.warning("Failed to count YouTube videos")
            return 0
    else:
        # SQLite backend — no source filtering available
        return 0
