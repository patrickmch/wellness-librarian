"""
Transcript → Video Recommendation pipeline.

Analyzes a call transcript, extracts wellness themes, runs parallel
RAG queries, deduplicates results, and returns top-N curated
video recommendations with personalized relevance explanations.

Pipeline steps:
1. Theme extraction (Haiku) — identify key wellness topics from transcript
2. Parallel retrieval — run existing ParentChildRetriever for each theme
3. Deduplication + cross-query scoring — boost videos appearing in multiple queries
4. Relevance generation (Haiku) — write personalized explanations
5. Response assembly — return ranked recommendations with deep-links
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.config import get_settings
from backend.rag.generator import get_async_anthropic_client
from backend.rag.retrieval.parent_child import (
    ParentChildRetriever,
    ParentRetrievalResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models (internal to this pipeline)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedTheme:
    """A wellness theme extracted from the transcript."""
    theme: str
    query: str


@dataclass
class CandidateVideo:
    """A deduplicated video candidate with cross-query scoring."""
    video_id: str
    title: str
    category: str
    video_url: str
    start_time_seconds: int
    source: str  # "youtube" or "vimeo"
    duration: str
    excerpt: str
    best_rerank_score: float
    cross_query_count: int
    themes_matched: list[str] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Score combining cross-query reinforcement and reranker relevance."""
        return self.cross_query_count * 2.0 + self.best_rerank_score


@dataclass
class Recommendation:
    """A final video recommendation with personalized relevance."""
    rank: int
    title: str
    category: str
    video_url: str
    start_time_seconds: int
    source: str
    relevance: str
    themes_matched: list[str]
    excerpt: Optional[str] = None


@dataclass
class RecommendationResult:
    """Full result from the recommendation pipeline."""
    recommendations: list[Recommendation]
    themes: list[ExtractedTheme]
    total_videos_searched: int
    queries_run: int


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

THEME_EXTRACTION_SYSTEM = """You extract wellness themes from call transcripts.
Output ONLY valid JSON — no markdown fencing, no commentary."""

THEME_EXTRACTION_USER = """Analyze this call transcript and extract the {num_themes} most \
important wellness themes, health concerns, or topics discussed.

For each theme, write a focused question (under 200 words) that could be asked to a \
wellness video library to find relevant educational content. The question should be \
specific enough to get good search results but broad enough to match multiple videos.

Return ONLY a JSON array — no other text:
[
  {{"theme": "Short Label", "query": "Focused question for the video library..."}},
  ...
]

TRANSCRIPT:
{transcript}"""

RELEVANCE_SYSTEM = """You write brief, personalized explanations of why a video is \
relevant to a specific person based on their call transcript. Output ONLY valid JSON — \
no markdown fencing, no commentary."""

RELEVANCE_USER = """Given this call summary and candidate videos, write a 2-3 sentence \
explanation of why each video is relevant to THIS specific person's situation.

CALL SUMMARY (first 2000 chars):
{summary}

CANDIDATE VIDEOS:
{candidates}

Return ONLY a JSON array with one entry per video, in the same order:
[
  {{"video_id": "...", "relevance": "2-3 sentence explanation..."}},
  ...
]"""


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------

async def generate_recommendations(
    transcript: str,
    num_recommendations: int = 3,
    num_themes: int = 4,
) -> RecommendationResult:
    """
    Generate video recommendations from a call transcript.

    Args:
        transcript: Full call transcript or summary text
        num_recommendations: Number of videos to recommend (1-5)
        num_themes: Number of themes to extract and query (2-6)

    Returns:
        RecommendationResult with ranked recommendations and metadata
    """
    settings = get_settings()

    # Step 1: Extract themes
    themes = await _extract_themes(transcript, num_themes)
    logger.info(f"[recommend] Extracted {len(themes)} themes: {[t.theme for t in themes]}")

    if not themes:
        return RecommendationResult(
            recommendations=[],
            themes=[],
            total_videos_searched=0,
            queries_run=0,
        )

    # Step 2: Parallel RAG retrieval
    retriever = ParentChildRetriever()
    retrieval_results = await _parallel_retrieve(retriever, themes)
    logger.info(f"[recommend] Ran {len(retrieval_results)} retrieval queries")

    # Step 3: Deduplicate and score
    candidates = _deduplicate_and_score(retrieval_results, themes)
    logger.info(f"[recommend] {len(candidates)} unique videos after deduplication")

    if not candidates:
        return RecommendationResult(
            recommendations=[],
            themes=themes,
            total_videos_searched=0,
            queries_run=len(themes),
        )

    # Take more candidates than needed for relevance generation
    top_candidates = sorted(
        candidates.values(),
        key=lambda c: c.composite_score,
        reverse=True,
    )[:num_recommendations * 2]  # 2x for selection headroom

    # Step 4: Generate personalized relevance explanations
    relevance_map = await _generate_relevance(
        transcript, top_candidates, num_recommendations
    )

    # Step 5: Assemble final recommendations
    # Re-sort by composite score and take top N
    final_candidates = sorted(
        top_candidates[:num_recommendations],
        key=lambda c: c.composite_score,
        reverse=True,
    )

    recommendations = []
    for rank, candidate in enumerate(final_candidates, 1):
        # Build deep-link URL
        video_url = candidate.video_url
        if candidate.start_time_seconds > 0:
            if candidate.source == "youtube":
                sep = "&" if "?" in video_url else "?"
                video_url = f"{video_url}{sep}t={candidate.start_time_seconds}"
            elif candidate.source == "vimeo":
                video_url = f"{video_url}#t={candidate.start_time_seconds}s"

        recommendations.append(Recommendation(
            rank=rank,
            title=candidate.title,
            category=candidate.category,
            video_url=video_url,
            start_time_seconds=candidate.start_time_seconds,
            source=candidate.source,
            relevance=relevance_map.get(candidate.video_id, "Relevant to topics discussed in the call."),
            themes_matched=candidate.themes_matched,
            excerpt=candidate.excerpt,
        ))

    return RecommendationResult(
        recommendations=recommendations,
        themes=themes,
        total_videos_searched=len(candidates),
        queries_run=len(themes),
    )


# ---------------------------------------------------------------------------
# Step 1: Theme extraction
# ---------------------------------------------------------------------------

async def _extract_themes(transcript: str, num_themes: int) -> list[ExtractedTheme]:
    """Extract wellness themes from transcript using Haiku."""
    settings = get_settings()
    client = get_async_anthropic_client()

    # Truncate transcript if very long (Haiku context is sufficient for 50k chars
    # but we trim to keep costs low)
    truncated = transcript[:40_000]

    try:
        message = await client.messages.create(
            model=settings.critic_model,
            max_tokens=1500,
            system=THEME_EXTRACTION_SYSTEM,
            messages=[{
                "role": "user",
                "content": THEME_EXTRACTION_USER.format(
                    num_themes=num_themes,
                    transcript=truncated,
                ),
            }],
        )

        raw = message.content[0].text.strip()
        # Strip markdown fencing if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        themes_data = json.loads(raw)

        return [
            ExtractedTheme(theme=t["theme"], query=t["query"])
            for t in themes_data
            if "theme" in t and "query" in t
        ]

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"[recommend] Failed to parse theme extraction: {e}")
        return []
    except Exception as e:
        logger.error(f"[recommend] Theme extraction failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 2: Parallel retrieval
# ---------------------------------------------------------------------------

async def _parallel_retrieve(
    retriever: ParentChildRetriever,
    themes: list[ExtractedTheme],
) -> list[tuple[ExtractedTheme, ParentRetrievalResponse]]:
    """Run retrieval for each theme in parallel using asyncio.to_thread."""

    async def _retrieve_one(theme: ExtractedTheme) -> tuple[ExtractedTheme, ParentRetrievalResponse]:
        # Wrap sync retriever in thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            retriever.retrieve,
            query=theme.query,
        )
        return (theme, result)

    results = await asyncio.gather(
        *[_retrieve_one(theme) for theme in themes],
        return_exceptions=True,
    )

    # Filter out exceptions
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"[recommend] Retrieval failed: {r}")
        else:
            valid_results.append(r)

    return valid_results


# ---------------------------------------------------------------------------
# Step 3: Deduplication + cross-query scoring
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Normalize a video title for cross-platform deduplication.

    Some videos exist on both YouTube and Vimeo with slightly different
    titles (e.g. extra suffixes, date prefixes). Stripping to lowercase
    alphanumeric gives a stable key for matching.
    """
    import re
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _deduplicate_and_score(
    retrieval_results: list[tuple[ExtractedTheme, ParentRetrievalResponse]],
    themes: list[ExtractedTheme],
) -> dict[str, CandidateVideo]:
    """Deduplicate videos across queries and compute cross-query scores.

    Deduplicates by both video_id AND normalized title so that the same
    content uploaded to YouTube and Vimeo only counts once.
    """
    candidates: dict[str, CandidateVideo] = {}
    # Maps normalized title → canonical video_id (first seen)
    title_to_vid: dict[str, str] = {}

    for theme, response in retrieval_results:
        sources = response.get_sources()
        scores = [r.score for r in response.results]

        for i, source in enumerate(sources):
            vid = source["video_id"]
            score = scores[i] if i < len(scores) else 0.0
            norm_title = _normalize_title(source["title"])

            # Resolve cross-platform duplicates by title
            if norm_title in title_to_vid:
                vid = title_to_vid[norm_title]
            else:
                title_to_vid[norm_title] = vid

            if vid in candidates:
                # Update existing candidate
                candidates[vid].cross_query_count += 1
                if score > candidates[vid].best_rerank_score:
                    candidates[vid].best_rerank_score = score
                    candidates[vid].excerpt = source.get("excerpt", "")
                if theme.theme not in candidates[vid].themes_matched:
                    candidates[vid].themes_matched.append(theme.theme)
            else:
                candidates[vid] = CandidateVideo(
                    video_id=vid,
                    title=source["title"],
                    category=source["category"],
                    video_url=source["video_url"],
                    start_time_seconds=source.get("start_time_seconds", 0),
                    source=source.get("source", ""),
                    duration=source.get("duration", ""),
                    excerpt=source.get("excerpt", ""),
                    best_rerank_score=score,
                    cross_query_count=1,
                    themes_matched=[theme.theme],
                )

    return candidates


# ---------------------------------------------------------------------------
# Step 4: Relevance generation
# ---------------------------------------------------------------------------

async def _generate_relevance(
    transcript: str,
    candidates: list[CandidateVideo],
    num_recommendations: int,
) -> dict[str, str]:
    """Generate personalized relevance explanations for top candidates."""
    client = get_async_anthropic_client()

    # Build candidate descriptions for the prompt
    candidate_descriptions = []
    for i, c in enumerate(candidates, 1):
        excerpt_preview = (c.excerpt[:300] + "...") if c.excerpt and len(c.excerpt) > 300 else (c.excerpt or "")
        candidate_descriptions.append(
            f"{i}. [{c.video_id}] \"{c.title}\" ({c.category})\n"
            f"   Themes matched: {', '.join(c.themes_matched)}\n"
            f"   Excerpt: {excerpt_preview}"
        )

    candidates_text = "\n\n".join(candidate_descriptions)
    summary = transcript[:2000]

    settings = get_settings()

    try:
        message = await client.messages.create(
            model=settings.critic_model,
            max_tokens=2000,
            system=RELEVANCE_SYSTEM,
            messages=[{
                "role": "user",
                "content": RELEVANCE_USER.format(
                    summary=summary,
                    candidates=candidates_text,
                ),
            }],
        )

        raw = message.content[0].text.strip()
        # Strip markdown fencing if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        relevance_data = json.loads(raw)

        return {
            r["video_id"]: r["relevance"]
            for r in relevance_data
            if "video_id" in r and "relevance" in r
        }

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"[recommend] Failed to parse relevance: {e}")
        return {}
    except Exception as e:
        logger.warning(f"[recommend] Relevance generation failed: {e}")
        return {}
