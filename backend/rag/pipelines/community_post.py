"""
Community Post Generator pipeline.

Picks a random video from the library, fetches a content preview,
generates a WhatsApp-friendly community post using Haiku, and builds
a deep-linked video URL.

Pipeline steps:
1. Pick random video (with optional category filter + exclusions)
2. Fetch first 3 parent chunks as content preview
3. Generate post via Haiku (format-specific prompt → JSON output)
4. Build deep-link URL (YouTube ?t= / Vimeo #t=s)
5. Return assembled response
"""

import json
import logging
from dataclasses import dataclass

from backend.config import get_settings
from backend.rag.generator import get_async_anthropic_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CommunityPostResult:
    """Result from the community post pipeline."""
    post: str
    video_id: str
    video_title: str
    video_url: str  # deep-linked
    start_time_seconds: int
    category: str
    post_format: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You write short, engaging WhatsApp community posts for a wellness \
education group. The posts feature videos from Christine's wellness video library. \
Write in a warm, personal tone as if you're a community facilitator sharing a gem \
with friends. Output ONLY valid JSON — no markdown fencing, no commentary."""

FORMAT_PROMPTS = {
    "spotlight": """Write a "spotlight" community post that introduces this video as a \
discovery — "Have you seen this gem?" style. Include a warm intro and a compelling \
reason to watch. The post should feel like a personal recommendation from a friend.

Pick one specific, interesting moment or insight from the transcript to highlight.

VIDEO TITLE: {title}
CATEGORY: {category}

TRANSCRIPT PREVIEW:
{preview}

Return ONLY this JSON (no other text):
{{"post": "Your WhatsApp post text here (under 500 characters, include 1-2 emojis)", "highlight_timestamp": <seconds into the video where the highlighted insight appears, integer>}}""",

    "tip": """Write a "tip" community post that shares one specific, actionable wellness \
insight from this video. Lead with the practical takeaway, then mention the video as \
the source. The tip should be immediately useful.

VIDEO TITLE: {title}
CATEGORY: {category}

TRANSCRIPT PREVIEW:
{preview}

Return ONLY this JSON (no other text):
{{"post": "Your WhatsApp post text here (under 500 characters, include 1-2 emojis)", "highlight_timestamp": <seconds into the video where the tip is discussed, integer>}}""",

    "discussion": """Write a "discussion" community post that poses a thought-provoking \
question inspired by this video. Start with the reflection prompt, then mention the \
video as a resource for exploring it further. The goal is to spark conversation.

VIDEO TITLE: {title}
CATEGORY: {category}

TRANSCRIPT PREVIEW:
{preview}

Return ONLY this JSON (no other text):
{{"post": "Your WhatsApp post text here (under 500 characters, include 1-2 emojis)", "highlight_timestamp": <seconds into the video where the discussion topic appears, integer>}}""",
}


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------

async def generate_community_post(
    post_format: str = "spotlight",
    category: str | None = None,
    exclude_video_ids: list[str] | None = None,
) -> CommunityPostResult:
    """
    Generate a community post featuring a random video.

    Args:
        post_format: One of "spotlight", "tip", "discussion"
        category: Optional category filter
        exclude_video_ids: Video IDs to skip (recently used)

    Returns:
        CommunityPostResult with post text, video info, and deep-link URL

    Raises:
        ValueError: If no matching video is found
    """
    settings = get_settings()

    # Step 1: Pick a random video
    video_info = _pick_random_video(category, exclude_video_ids)
    if not video_info:
        raise ValueError(
            "No matching video found. Try a different category or fewer exclusions."
        )

    video_id = video_info["video_id"]
    logger.info(f"[community-post] Selected video: {video_info['title']} ({video_id})")

    # Step 2: Fetch content preview (first 3 parent chunks)
    parents = _get_content_preview(video_id)
    preview_text = "\n\n".join(p.text for p in parents[:3])

    if not preview_text:
        raise ValueError(f"No content found for video {video_id}")

    # Step 3: Generate post via Haiku
    post_text, highlight_timestamp = await _generate_post(
        post_format=post_format,
        title=video_info["title"],
        category=video_info["category"],
        preview=preview_text,
    )

    # Step 4: Build deep-link URL
    video_url = _build_deep_link(
        base_url=video_info["video_url"],
        source=video_info["source"],
        timestamp=highlight_timestamp,
    )

    return CommunityPostResult(
        post=post_text,
        video_id=video_id,
        video_title=video_info["title"],
        video_url=video_url,
        start_time_seconds=highlight_timestamp,
        category=video_info["category"],
        post_format=post_format,
    )


# ---------------------------------------------------------------------------
# Step 1: Random video selection
# ---------------------------------------------------------------------------

def _pick_random_video(
    category: str | None,
    exclude_video_ids: list[str] | None,
) -> dict | None:
    """Pick a random video from the appropriate store backend."""
    settings = get_settings()

    if settings.store_backend == "supabase":
        from backend.rag.stores.supabase_store import get_supabase_store
        store = get_supabase_store()
        return store.get_random_video_id(category, exclude_video_ids)
    else:
        from backend.rag.docstore.sqlite_store import get_docstore
        store = get_docstore()
        return store.get_random_video_id(category, exclude_video_ids)


# ---------------------------------------------------------------------------
# Step 2: Content preview
# ---------------------------------------------------------------------------

def _get_content_preview(video_id: str):
    """Fetch parent chunks for a video from the appropriate store backend."""
    settings = get_settings()

    if settings.store_backend == "supabase":
        from backend.rag.stores.supabase_store import get_supabase_store
        store = get_supabase_store()
        return store.get_parents_by_video(video_id)
    else:
        from backend.rag.docstore.sqlite_store import get_docstore
        store = get_docstore()
        return store.get_by_video_id(video_id)


# ---------------------------------------------------------------------------
# Step 3: Post generation
# ---------------------------------------------------------------------------

async def _generate_post(
    post_format: str,
    title: str,
    category: str,
    preview: str,
) -> tuple[str, int]:
    """Generate post text and highlight timestamp via Haiku."""
    settings = get_settings()
    client = get_async_anthropic_client()

    prompt_template = FORMAT_PROMPTS[post_format]
    # Truncate preview to keep costs low
    user_prompt = prompt_template.format(
        title=title,
        category=category,
        preview=preview[:4000],
    )

    message = await client.messages.create(
        model=settings.critic_model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fencing if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    data = json.loads(raw)

    post_text = data["post"]
    highlight_timestamp = int(data.get("highlight_timestamp", 0))

    return post_text, highlight_timestamp


# ---------------------------------------------------------------------------
# Step 4: Deep-link URL builder
# ---------------------------------------------------------------------------

def _build_deep_link(base_url: str, source: str, timestamp: int) -> str:
    """Build a deep-linked video URL with timestamp."""
    if not base_url or timestamp <= 0:
        return base_url or ""

    if source == "youtube":
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}t={timestamp}"
    elif source == "vimeo":
        return f"{base_url}#t={timestamp}s"

    return base_url
