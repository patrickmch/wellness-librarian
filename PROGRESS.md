# Wellness Librarian - Progress Tracking

## Current Status

**Phase:** Deployed to Production 🚀
**Last Updated:** 2026-02-24

### Session Context (for resuming)

**What's done:**
- Citation deep-linking with video timestamps
- User feedback (thumbs up/down) stored in SQLite
- Glass box compliance (disclaimer + excerpts toggle)
- Self-correction critic layer (Haiku verification)
- Retrieval diversity tuning: `max_chunks_per_video` reduced from 2 → 1
- Diagnostic script added: `scripts/diagnose_retrieval.py`
- System prompt updated to handle content/intent mismatch (e.g., specific thyroid content for general weight loss queries)
- Multi-turn conversation history: Client-side history sent with each request for context continuity
- Supabase pgvector backend - Unified Postgres storage replaces ChromaDB + SQLite
- Incremental Video Sync Pipeline - Automated discovery and ingestion from Vimeo/YouTube with tracking
- **DEPLOYED: Railway production with Supabase backend** - https://wellness-librarian-production.up.railway.app
- Transcript → Video Recommendation endpoint (`POST /api/recommend`)
- Updated Haiku model ID from deprecated `claude-3-5-haiku-20241022` → `claude-haiku-4-5-20251001`
- Community Post Generator endpoint (`POST /api/community-post`)

### Community Post Generator (2026-02-24) ✅

**Endpoint:** `POST /api/community-post` (admin-protected)
**Purpose:** Generate engaging WhatsApp posts featuring videos from Christine's library for Patrick's wellness community.

**Pipeline:** Random video → fetch content preview → Haiku generation → deep-link URL
**Cost:** ~$0.002/post (~$0.06/month at 1/day)

**Three post formats:**
| Format | Tone | Use |
|--------|------|-----|
| `spotlight` | Discovery — "Have you seen this gem?" | Warm intro + why it's worth watching |
| `tip` | Practical — actionable wellness insight | One specific takeaway + source video |
| `discussion` | Engagement — thought-provoking question | Reflection prompt + mention of video |

**Repeat avoidance:** Caller passes `exclude_video_ids` (no server-side history — keeps API stateless).

**Files created/modified:**
```
NEW: backend/rag/pipelines/community_post.py  # Pipeline implementation
MOD: backend/rag/stores/supabase_store.py      # get_random_video_id()
MOD: backend/rag/docstore/sqlite_store.py      # get_random_video_id()
MOD: backend/api/models.py                     # CommunityPostRequest/Response
MOD: backend/api/routes.py                     # POST /api/community-post route
```

**Design decisions:**
- **43. Single Haiku call:** Unlike /recommend (multi-query + dedup), community posts only need one video and one generation call — keep it cheap and fast.
- **44. No server-side history:** Repeat avoidance is the caller's responsibility via `exclude_video_ids`. Avoids a migration and keeps the API stateless.
- **45. Random video via GROUP BY:** Uses `GROUP BY ... ORDER BY RANDOM() LIMIT 1` in both Postgres and SQLite for unbiased video selection.
- **46. Format-specific prompts:** Each format (spotlight/tip/discussion) has its own prompt template to ensure tonal variety.

### Production Deployment (2026-02-10) ✅

**URL:** https://wellness-librarian-production.up.railway.app
**Backend:** Supabase pgvector (transaction pooler)
**Pipeline:** Enhanced (parent-child retrieval + Voyage reranking + Haiku critic)
**Data:** 195 videos, 2,424 parent chunks, 17 categories

**Fixes applied for deployment:**
- Dockerfile: `COPY data/` → `RUN mkdir -p ./data/chroma_db` (data/ is gitignored)
- main.py: Backend-aware startup (Supabase vs ChromaDB)
- routes.py: Health check reports correct chunk count per backend
- .env: Supabase DB URL updated to transaction pooler (`aws-1-us-east-1.pooler.supabase.com:6543`)

**Key lesson:** Supabase free-tier direct DB connections (`db.*.supabase.co`) don't resolve via IPv4. Must use the transaction pooler URL instead.

### What's Next
- **Run first video sync** against live Vimeo/YouTube channels: `python scripts/sync_videos.py --dry-run`
- **Rotate API keys** — Anthropic, OpenAI, Voyage keys were exposed in a conversation session
- **Optional enhancements:** Category-based filtering, query intent classification, feedback-driven learning
- **Admin API key** — `ADMIN_API_KEY` is still `dev-admin-key` in Railway; generate a real one for production
- **Transcript Recommendation Feature** — See detailed design below

---

### Feature Proposal: Transcript → Video Recommendations (2026-02-20)

**Status:** Built and tested locally (2026-02-24)
**Validated:** Workflow tested manually via Claude as intermediary (2026-02-19)

#### Problem

After 1-on-1 calls with community members, Patrick wants to send 3 curated video
recommendations from Christine's library based on what was discussed. Currently this
requires manually recalling which videos match — the RAG system can do it better.

#### How It Was Validated

Claude acted as the intermediary for a real call with V.A. Lytle:
1. Analyzed the Gemini transcript summary
2. Extracted 4 themes: reframing aging, meditation/mindfulness, healthy aging, science behind "woo woo"
3. Ran 4 parallel queries against `POST /api/chat` (production)
4. Collected 32 source videos, deduplicated, ranked by cross-query reinforcement + relevance
5. Selected top 3 with rationale, timestamps, and deep-links

**Key finding:** Multi-query with deduplication works significantly better than a single
query. Videos that appear across multiple theme queries are reliably the most relevant.

#### Proposed Endpoint

```
POST /api/recommend
```

**Auth:** Admin-protected (same `X-Admin-Key` header as `/api/ingest`)

#### Request Model

```python
class RecommendRequest(BaseModel):
    """Request body for transcript recommendation endpoint."""
    transcript: str = Field(
        ...,
        min_length=50,
        max_length=50_000,
        description="Call transcript or summary text"
    )
    num_recommendations: int = Field(
        3, ge=1, le=5,
        description="Number of video recommendations to return"
    )
    num_themes: int = Field(
        4, ge=2, le=6,
        description="Number of themes to extract and query"
    )
```

No `max_length=2000` limit like `/api/chat` — the transcript is processed server-side
by Claude, not sent directly to the embedding model.

#### Response Model

```python
class VideoRecommendation(BaseModel):
    """A single curated video recommendation."""
    rank: int                          # 1, 2, or 3
    title: str                         # Video title
    category: str                      # e.g. "Wellness Evolution Community"
    video_url: str                     # Full URL with timestamp
    start_time_seconds: int            # Deep-link timestamp
    relevance: str                     # 2-3 sentence explanation of why this matches
    themes_matched: list[str]          # Which extracted themes this video matched
    source: str                        # "youtube" or "vimeo"
    excerpt: Optional[str] = None      # Key transcript excerpt

class ThemeExtracted(BaseModel):
    """A theme extracted from the transcript."""
    theme: str                         # Short label, e.g. "Reframing Aging"
    query: str                         # The RAG query generated for this theme
    videos_found: int                  # How many unique videos matched

class RecommendResponse(BaseModel):
    """Response body for recommend endpoint."""
    recommendations: list[VideoRecommendation]
    themes: list[ThemeExtracted]
    total_videos_searched: int         # Total unique videos across all queries
    pipeline_used: str = "enhanced"
```

#### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  POST /api/recommend                                         │
│  { transcript: "..." }                                       │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 1: THEME EXTRACTION (Claude Haiku)                     │
│                                                              │
│  System: "Extract 3-5 wellness themes from this transcript.  │
│           Return as JSON array of {theme, query} objects."   │
│                                                              │
│  Input:  Full transcript text (up to 50k chars)              │
│  Output: [                                                   │
│    {"theme": "Reframing Aging",                              │
│     "query": "How can I reframe my attitude about aging..."},│
│    {"theme": "Meditation Basics", "query": "..."},           │
│    ...                                                       │
│  ]                                                           │
│                                                              │
│  Model: claude-3-5-haiku (fast, cheap — theme extraction     │
│         doesn't need creative reasoning)                     │
│  Cost:  ~$0.003/call                                         │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 2: PARALLEL RAG RETRIEVAL (existing pipeline)          │
│                                                              │
│  For each theme query, run:                                  │
│    ParentChildRetriever.retrieve(query=theme.query)          │
│                                                              │
│  This reuses the full existing pipeline:                     │
│    Voyage embed → pgvector search → diversity filter →       │
│    parent expansion → Voyage rerank                          │
│                                                              │
│  Queries run via asyncio.gather() for parallelism.           │
│  Each returns ~8 source videos.                              │
│                                                              │
│  Cost: 4 queries × ~$0.01 Voyage = ~$0.04                   │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 3: DEDUPLICATION + CROSS-QUERY SCORING                 │
│                                                              │
│  Collect all sources across queries into a single pool.      │
│  For each unique video_id:                                   │
│    - cross_query_count: How many theme queries returned it   │
│    - best_rerank_score: Highest reranker score across queries│
│    - themes_matched: Which themes it appeared under          │
│                                                              │
│  Score = cross_query_count * 2.0 + best_rerank_score         │
│  (Videos appearing in multiple queries get a strong boost)   │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 4: TOP-N SELECTION + RELEVANCE GENERATION (Claude)     │
│                                                              │
│  Take the top N candidates by score.                         │
│  Send to Claude with the original transcript summary:        │
│                                                              │
│  "Given this call transcript and these candidate videos,     │
│   write a 2-3 sentence explanation of why each video is      │
│   relevant to THIS specific person's situation."             │
│                                                              │
│  Model: claude-3-5-haiku (structured output)                 │
│  Cost:  ~$0.005/call                                         │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 5: RESPONSE ASSEMBLY                                   │
│                                                              │
│  Return RecommendResponse with:                              │
│    - 3 ranked recommendations with personalized relevance    │
│    - Extracted themes for transparency                       │
│    - Deep-link URLs with timestamps                          │
└─────────────────────────────────────────────────────────────┘
```

#### Files to Create/Modify

```
NEW FILES:
backend/rag/pipelines/recommend.py     # Core recommendation pipeline
                                       # - extract_themes() → Claude Haiku
                                       # - parallel_retrieve() → asyncio.gather
                                       # - deduplicate_and_rank()
                                       # - generate_relevance() → Claude Haiku

MODIFY:
backend/api/models.py                  # Add RecommendRequest, RecommendResponse,
                                       #     VideoRecommendation, ThemeExtracted
backend/api/routes.py                  # Add POST /api/recommend route
backend/config.py                      # Add recommend_* settings (optional)
```

**No changes needed to:** retrieval, embeddings, reranking, vector store, or critic.
The recommendation pipeline *composes* existing components rather than modifying them.

#### Theme Extraction Prompt

```python
THEME_EXTRACTION_PROMPT = """Analyze this call transcript and extract the {num_themes}
most important wellness themes, health concerns, or topics discussed.

For each theme, write a focused question (under 200 words) that could be asked to a
wellness video library to find relevant educational content. The question should be
specific enough to get good search results but broad enough to match multiple videos.

Return as a JSON array:
[
  {{"theme": "Short Label", "query": "Focused question for the video library..."}},
  ...
]

TRANSCRIPT:
{transcript}"""
```

#### Design Decisions

**36. Haiku for Theme Extraction:** Theme extraction is structured analysis, not
creative — Haiku is fast (~0.5s) and cheap (~$0.003). No need for Sonnet/Opus here.

**37. Parallel Retrieval via asyncio.gather:** The retriever is currently sync
(psycopg2 is blocking), but we can run multiple retrievals concurrently using
`asyncio.to_thread()` wrapping `retriever.retrieve()`. This keeps total retrieval
time ~equal to a single query rather than N × single query.

**38. Cross-Query Reinforcement Scoring:** A video that appears across multiple
independent theme queries is almost certainly relevant. This signal is stronger than
any single reranker score. Weighting: `cross_query_count * 2.0 + best_rerank_score`.

**39. No Critic on Recommendations:** The critic verifies health *claims* in
generated prose. Recommendations are curated selections with relevance explanations —
no health claims to hallucinate. Skip the critic to save latency and cost.

**40. Admin-Protected Endpoint:** Only Patrick (or an automated system) calls this
endpoint, not end users. Protects against abuse of the multi-query pipeline which
costs more per request (~$0.05 vs ~$0.01 for a single chat).

**41. Transcript Size Limit (50k chars):** A 1-hour call transcript is typically
~15-20k chars. 50k provides headroom for verbose transcripts or multiple calls
without hitting Claude's context limits for theme extraction.

**42. Cross-Platform Title Dedup:** Some videos exist on both YouTube and Vimeo
with different video_ids. Deduplication normalizes titles (lowercase alphanumeric)
to catch these duplicates and merge their cross-query scores.

#### Cost Per Recommendation Request

| Step | Model/Service | Est. Cost |
|------|---------------|-----------|
| Theme extraction | Haiku (~2k input, ~500 output tokens) | ~$0.003 |
| Parallel retrieval (4×) | Voyage embed + rerank | ~$0.04 |
| Relevance generation | Haiku (~3k input, ~300 output tokens) | ~$0.004 |
| **Total per request** | | **~$0.05** |

At ~10 calls/week, that's **~$2/month** additional cost.

#### Example Usage

```bash
# With a transcript file
curl -X POST https://wellness-librarian-production.up.railway.app/api/recommend \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -d '{
    "transcript": "Patrick and V.A. discussed reframing aging...",
    "num_recommendations": 3,
    "num_themes": 4
  }'
```

```json
{
  "recommendations": [
    {
      "rank": 1,
      "title": "Navigating Perimenopause: Hormones, Stress, and Thriving in Midlife",
      "category": "Wellness Evolution Community",
      "video_url": "https://www.youtube.com/watch?v=zjypOhuB5FY&t=3176",
      "start_time_seconds": 3176,
      "relevance": "V.A. is working on reframing aging from discomfort to appreciation. This video directly addresses that shift, with Dr. Vandenberg describing aging as 'an honor' and a transition into wisdom — exactly the perspective shift V.A. is seeking.",
      "themes_matched": ["Reframing Aging", "Healthy Aging"],
      "source": "youtube"
    },
    ...
  ],
  "themes": [
    {"theme": "Reframing Aging", "query": "How can I reframe...", "videos_found": 8},
    {"theme": "Meditation Basics", "query": "What are the benefits...", "videos_found": 8},
    ...
  ],
  "total_videos_searched": 24,
  "pipeline_used": "enhanced"
}
```

#### Future Extensions (not building now)

- **Web UI:** Textarea on the frontend for pasting transcripts, renders recommendations as cards
- **Email integration:** Auto-send recommendation email to community member after call
- **Batch mode:** Process multiple transcripts (e.g., group session notes)
- **Feedback loop:** Track which recommended videos members actually watch, use to improve ranking

**Current tuning parameters:**
```python
# backend/config.py
max_chunks_per_video: int = 1   # Changed from 2 → ensures 8 unique videos in results
child_top_k: int = 30           # Initial search pool
mmr_lambda: float = 0.5         # Balanced relevance/diversity
enable_reranking: bool = True   # Voyage rerank-2 cross-encoder
```

**To restart server:**
```bash
cd ~/code/wellness-librarian
source venv/bin/activate
uvicorn backend.main:app --reload
```

**To diagnose retrieval for a query:**
```bash
python scripts/diagnose_retrieval.py "your query here"
python scripts/diagnose_retrieval.py "how to lose weight" --max-per-video 1 --verbose
```

---

### Supabase Migration (2026-02-04) ✅

Implemented dual-backend architecture supporting both local SQLite/ChromaDB and production Supabase with pgvector.

**Why Supabase?**
- **Portability:** Single managed database, no volume mounting or file syncing
- **Maintainability:** Standard SQL, familiar to future maintainers
- **Scalability:** Can self-host Postgres later if needed
- **Simplicity:** Eliminates Railway volume seeding complexity

**Architecture:**

| Backend | Child Storage | Parent Storage | Use Case |
|---------|--------------|----------------|----------|
| `sqlite` | ChromaDB (local file) | SQLite (local file) | Local development |
| `supabase` | Supabase pgvector | Supabase Postgres | Production |

**New Files:**
```
backend/rag/stores/
├── __init__.py
└── supabase_store.py    # Unified Supabase operations

scripts/
└── migrate_to_supabase.py   # Migration from ChromaDB+SQLite
```

**Modified Files:**
```
backend/config.py              # Added supabase_url, supabase_key, supabase_db_url, store_backend
backend/rag/retrieval/parent_child.py  # Dual-backend retrieval
backend/api/routes.py          # Dual-backend for /sources and /feedback
scripts/ingest_v2.py           # Added --target supabase flag
requirements.txt               # Added psycopg2-binary
Dockerfile                     # Simplified (no init script needed)
.env.example                   # Added Supabase vars
```

**Environment Variables:**
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJhbG...
SUPABASE_DB_URL=postgresql://postgres:xxx@db.xxx.supabase.co:5432/postgres
STORE_BACKEND=supabase  # or "sqlite" for local
```

**Migration Commands:**
```bash
# Migrate existing data from ChromaDB+SQLite to Supabase
python scripts/migrate_to_supabase.py --create-index

# Or re-ingest directly to Supabase
python scripts/ingest_v2.py --target supabase --reset

# Verify migration
python scripts/migrate_to_supabase.py --verify
```

**Supabase Schema:**
```sql
-- Parent chunks (replaces SQLite parent_chunks)
CREATE TABLE parent_chunks (
    parent_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    ...
);

-- Child chunks with vectors (replaces ChromaDB)
CREATE TABLE child_chunks (
    child_id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES parent_chunks(parent_id),
    text TEXT NOT NULL,
    embedding vector(1024),  -- Voyage voyage-3 dimension
    ...
);

-- IVFFlat index for fast similarity search
CREATE INDEX idx_child_embedding ON child_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

**Cost:** Supabase free tier ($0/month for this use case: ~100MB data, low traffic)

---

### Incremental Video Sync Pipeline (2026-02-04) ✅

Implemented automated video discovery and ingestion from Vimeo and YouTube with incremental sync (only process new videos).

**Architecture:**
```
SyncOrchestrator
  1. Discover videos from Vimeo API + YouTube via yt-dlp
  2. Compare with video_sync table in Supabase
  3. Download transcripts for new videos only
  4. Generate metadata.json for ingestion
  5. Run ingest_v2 pipeline (Voyage embeddings → Supabase/ChromaDB)
  6. Mark as synced in video_sync table
```

**New Files:**
```
backend/sync/
├── __init__.py
├── video_sync_store.py    # Supabase CRUD for video_sync table
├── vimeo_downloader.py    # Vimeo API discovery + download
├── youtube_downloader.py  # yt-dlp based discovery + download
└── orchestrator.py        # Main sync coordinator

scripts/
└── sync_videos.py         # CLI entry point

supabase/migrations/
└── 001_video_sync.sql     # video_sync table schema
```

**Environment Variables:**
```bash
VIMEO_ACCESS_TOKEN=xxx           # Vimeo API token
YOUTUBE_CHANNEL_URL=https://...  # YouTube channel URL
YOUTUBE_PLAYLIST_IDS=PLxx,PLyy   # Comma-separated playlist IDs
SYNC_ENABLED=true
SYNC_TRANSCRIPT_DIR=./data/transcripts
```

**CLI Usage:**
```bash
# Full sync (discover + download + ingest new videos)
python scripts/sync_videos.py

# Preview what would sync
python scripts/sync_videos.py --dry-run

# Discovery only (no downloads)
python scripts/sync_videos.py --discover-only

# Single platform
python scripts/sync_videos.py --platform vimeo

# Force re-sync (ignore tracking)
python scripts/sync_videos.py --force

# Show sync status
python scripts/sync_videos.py --status

# Just ingest pending videos
python scripts/sync_videos.py --ingest-only
```

**Supabase video_sync Table:**
```sql
CREATE TABLE video_sync (
    video_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,       -- 'vimeo' or 'youtube'
    title TEXT,
    transcript_file TEXT,
    folder_name TEXT,
    status TEXT DEFAULT 'pending', -- pending | ingested | failed | no_transcript
    ingested_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ,
    metadata JSONB
);
```

**Design Decisions:**
- **Separation of concerns:** Discovery returns video objects, orchestrator decides what to download
- **Idempotent upserts:** Running sync twice processes 0 new videos (incremental)
- **Platform-aware paths:** Transcripts stored as `data/transcripts/vimeo/...` and `data/transcripts/youtube/...`
- **Lazy initialization:** Downloaders only initialize when their platform is requested

---

### Conversation History (2026-02-04) ✅

Implemented multi-turn conversation support using client-side history.

**Architecture:**
- Frontend tracks all messages in `messages` array
- Each request sends prior turns as `history` array
- Backend builds Claude messages list with history + current prompt (with RAG context)
- No server-side session storage needed

**Files Modified:**
```
backend/api/models.py          # Added ChatMessage model + history field
backend/rag/pipelines/base.py  # Added history param to abstract methods
backend/rag/pipelines/legacy.py    # Added _build_messages helper
backend/rag/pipelines/enhanced.py  # Added build_messages_with_history helper
backend/rag/pipelines/router.py    # Pass history through all methods
backend/api/routes.py          # Convert ChatMessage to dict + pass to pipeline
frontend/app.js                # Build and send history with each request
```

**Request Format:**
```json
{
  "message": "Can you explain more about magnesium?",
  "history": [
    {"role": "user", "content": "What supplements help with sleep?"},
    {"role": "assistant", "content": "Based on the library...magnesium..."}
  ],
  "session_id": "user-12345"
}
```

---

### Retrieval Diversity Tuning (2026-02-03) ✅

Investigated why condition-specific videos (e.g., "Hashimotos & Thyroid Health") dominated general queries (e.g., "how to lose weight").

**Problem:**
- Hashimotos video appeared 2x in top 8 results for weight loss queries
- Content genuinely mentions weight (thyroid affects metabolism) so embeddings score it highly
- Reranker also scores it highly (0.6562) because it IS semantically related

**Diagnostic Findings:**

| Stage | What Happens | Hashimotos Behavior |
|-------|--------------|---------------------|
| 1. Embedding search | Top 30 children by similarity | Ranks #4 (score 0.5548) |
| 2. Per-video dedup | Max N per video | Was getting 2 slots |
| 3. MMR diversity | Reorder for diversity | Doesn't help (reranker overrides) |
| 4. Parent expansion | Children → parents | Still present |
| 5. Reranking | Cross-encoder scoring | Stays at #4-5 (score 0.6562) |

**Key Insight:** MMR lambda doesn't affect final results because the reranker is the final arbiter. The reranker uses its own cross-encoder scores regardless of input order.

**Solution Applied:**
```python
# backend/config.py line 64
max_chunks_per_video: int = 1  # Was 2
```

**Results:**

| Setting | Videos in Final 8 | Max from any video |
|---------|-------------------|-------------------|
| `max_per_video=2` | 7 videos | 2 (25%) |
| `max_per_video=1` | **8 videos** | **1 (12.5%)** |

Hashimotos now gets 1/8 slots at position #4 instead of 2/8 slots. Perfect video diversity achieved.

**Files Added:**
```
scripts/diagnose_retrieval.py  # Pipeline diagnostic tool
```

**Files Modified:**
```
backend/config.py              # max_chunks_per_video: 2 → 1
backend/rag/generator.py       # Added "When Content Doesn't Match Intent" prompt section
```

**Future Options (if needed):**
- Category-based filtering: Penalize condition-specific videos for general queries
- Query intent classification: Detect "general" vs "specific" queries
- Feedback learning: Use 👎 data to learn bad query-video pairings

---

### Citation Deep-Linking, Feedback & Glass Box (2026-01-26) ✅

Implemented three UX/compliance features to improve citation accuracy and user trust:

**Feature 1: Inline Citations with Deep-Linking**

| Component | Implementation | Benefit |
|-----------|---------------|---------|
| Timestamp Storage | `start_time_seconds` in ParentChunk | Maps chunk position to video timestamp |
| Numbered Citations | LLM outputs `[1]`, `[2]` references | Clear attribution of claims to sources |
| Deep-Links | YouTube `?t=123` / Vimeo `#t=123s` | Click citation → jump to exact video segment |

**Feature 2: User Feedback (Thumbs Up/Down)**

| Component | Implementation | Benefit |
|-----------|---------------|---------|
| SQLite `feedback` table | Stores message_id, feedback_type, session_id, query | Analytics for retrieval improvement |
| `/api/feedback` endpoint | POST with rating submission | Simple API for frontend |
| UI Buttons | 👍 👎 below assistant messages | Low-friction user feedback |

**Feature 3: Glass Box Compliance**

| Component | Implementation | Benefit |
|-----------|---------------|---------|
| Medical Disclaimer | Persistent banner above input | Legal compliance for health content |
| Raw Excerpts | "View excerpts" toggle | Transparency: see exact source text |
| Numbered Sources | `[1] Title` format in sources list | Matches citation numbers in response |

**Files Modified:**
```
backend/rag/chunking/models.py          # Added start_time_seconds field
backend/rag/chunking/parent_chunker.py  # Timestamp calculation from VTT
backend/rag/docstore/sqlite_store.py    # Schema + feedback methods
backend/rag/retrieval/parent_child.py   # Include timestamp/excerpt in sources
backend/rag/pipelines/enhanced.py       # Numbered citation prompt
backend/api/models.py                   # SourceInfo + FeedbackRequest models
backend/api/routes.py                   # /api/feedback endpoint
frontend/app.js                         # Citation rendering + feedback
frontend/index.html                     # Feedback buttons + disclaimer + excerpts
frontend/styles.css                     # New component styles
```

**Re-ingestion Required:** Run `python scripts/ingest_v2.py --reset` to populate timestamps.

---

### Self-Correction Critic Layer (2026-01-26) ✅

Implemented post-generation verification to prevent hallucinated health claims:

**Feature 4: Critic Verification**

| Component | Implementation | Benefit |
|-----------|---------------|---------|
| Critic Module | `backend/rag/pipelines/critic.py` | Dedicated fact-checking layer |
| Fast Model | Claude 3.5 Haiku | Low latency (~1-2s), low cost (~$0.002/query) |
| Source Verification | Checks dosages/supplements/timing against excerpts | Prevents hallucinated health recommendations |
| Pipeline Integration | `verify_response_async()` in enhanced pipeline | Transparent correction with metadata tracking |

**How It Works:**

1. LLM generates initial response with citations
2. Critic receives response + source excerpts (600 chars each)
3. Critic outputs either:
   - `VERIFIED` → Original response passes through
   - Corrected text → Replaces original response
4. `critic_corrected` metadata tracks corrections for analytics

**Prompt Strategy:**
```
You are a fact-checker for wellness content. Verify that health claims are supported by the sources.

CRITICAL: Your output must be EXACTLY one of these two formats:
1. If accurate: Output the single word VERIFIED (nothing else)
2. If corrections needed: Output ONLY the corrected response text

NEVER output analysis, explanations, or commentary. Just VERIFIED or the corrected text.
```

**Files Added/Modified:**
```
backend/rag/pipelines/critic.py   # NEW: Critic module with async/sync functions
backend/rag/pipelines/enhanced.py # Integrated critic verification
backend/config.py                 # Added enable_critic, critic_model settings
backend/api/models.py             # Added critic_enabled, critic_corrected to PipelineMetadata
backend/api/routes.py             # Pass critic metadata to response
```

**Configuration:**
```python
# In config.py
enable_critic: bool = True  # Toggle critic verification
critic_model: str = "claude-3-5-haiku-20241022"  # Fast model for low latency
```

**Note:** Critic is skipped for streaming responses to avoid buffering the entire response before sending.

---

### RAG Enhancement: Parent-Child Retrieval + Reranking (2026-01-26) ✅

Implemented comprehensive RAG enhancements to address retrieval quality issues:

**Problem Solved:**
- Over-reliance on large videos (thyroid interview dominated results)
- Missing specific segments from smaller targeted videos
- Lack of retrieval diversity (top-8 often from 1-2 videos)

**Solution Components:**

| Component | Implementation | Rationale |
|-----------|---------------|-----------|
| **Parent-Child Retrieval** | Custom implementation | Search small chunks (precision), return larger context (comprehension) |
| **Reranking** | Voyage `rerank-2` | Second-stage relevance scoring improves specificity |
| **Diversity Filtering** | Per-video dedup + MMR | Prevents single video domination |
| **A/B Testing** | Session-based routing | Compare pipelines before full migration |

**New Directory Structure:**
```
backend/rag/
├── chunking/           # Token-based parent/child chunking
│   ├── models.py       # ParentChunk, ChildChunk dataclasses
│   ├── parent_chunker.py   # 500-2000 token parents
│   └── child_chunker.py    # 250 token children with overlap
├── providers/          # Embedding providers
│   ├── base.py         # EmbeddingProvider ABC
│   ├── openai_provider.py  # Legacy embeddings
│   └── voyage_provider.py  # Enhanced embeddings
├── reranking/          # Second-stage ranking
│   └── voyage_reranker.py  # Voyage rerank-2
├── retrieval/          # Retrieval strategies
│   ├── diversity.py    # Per-video dedup + MMR
│   └── parent_child.py # Core parent-child retriever
├── docstore/           # Parent storage
│   └── sqlite_store.py # SQLite for parent chunks
└── pipelines/          # Pipeline implementations
    ├── base.py         # RAGPipeline ABC
    ├── legacy.py       # Original implementation
    ├── enhanced.py     # Parent-child + reranking
    └── router.py       # A/B test routing
```

**Usage:**
```bash
# Index for enhanced pipeline
python scripts/ingest_v2.py --reset

# Enable enhanced pipeline
export RAG_PIPELINE=enhanced  # or "ab_test" for 50/50

# Test A/B routing
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "magnesium dosage", "session_id": "user123"}'
```

**New Dependencies:** `voyageai>=0.3.0`, `tiktoken>=0.5.0`, `numpy>=1.24.0`

**Cost Impact:** ~$1-3/month additional (Voyage embeddings + reranking)

---

### All Phases Complete ✅

### Multi-Source Support (2026-01-22) ✅

Refactored ingestion pipeline to support multiple video sources (YouTube + Vimeo):

- **Source-agnostic fields:** `video_id`, `video_url` replace Vimeo-specific names
- **Auto-detection:** Loader detects source from metadata JSON fields (`youtube_id` vs `vimeo_id`)
- **New metadata fields:** `source` ("youtube"/"vimeo"), `access_level` ("public"/"members_only")
- **119 YouTube videos ingested:** 21,078 chunks from public YouTube content
- **Filtering support:** Search by `source` or `access_level` in API

### Frontend Polish (2026-01-22) ✅

Implemented WEC brand polish to better match the Wellness Evolution Community aesthetic:

- **Welcome Screen:** Enhanced with decorative leaf elements, lotus-inspired icon with glow animation, larger typography, refined suggestion pill buttons
- **Header:** Gradient background with frosted glass effect, refined logo with subtle glow, styled stats badge with video icon
- **Messages:** Smooth entrance animations, gradient user bubbles, improved source citation cards with hover effects
- **Input Area:** Rounded pill-style input with shadow, gradient send button with hover lift effect
- **Loading:** Organic wave animation for loading dots
- **Overall:** Warm gradient body background, refined scrollbar, smooth transitions throughout

- ✅ **Phase 1: Foundation**
  - Project structure created
  - Config management with Pydantic Settings
  - VTT transcript loader with metadata.json integration
  - Rich metadata extraction (Vimeo IDs, URLs, durations, categories)
  - Text chunking with semantic boundary detection

- ✅ **Phase 2: Vector Store & Embeddings**
  - ChromaDB setup with persistence
  - OpenAI embeddings integration
  - Batch ingestion pipeline with progress tracking
  - CLI scripts for ingestion
  - Cost estimation (3,622 chunks, ~$0.02)

- ✅ **Phase 3: RAG Generation**
  - "Wellness Librarian" persona system prompt
  - Context assembly from retrieved chunks
  - Claude API integration (claude-sonnet-4-20250514)
  - Streaming response support
  - Source citation formatting

- ✅ **Phase 4: API Layer**
  - FastAPI application with CORS
  - POST /api/chat - Chat with librarian
  - POST /api/search - Semantic search
  - GET /api/sources - List categories
  - POST /api/ingest - Add transcript (admin protected)
  - GET /api/health - Health check

- ✅ **Phase 5: Frontend**
  - Chat interface with Tailwind CSS
  - Wellness community design aesthetic
  - Alpine.js for reactivity
  - Markdown rendering
  - Source citations display
  - Mobile responsive

- ✅ **Phase 6: Deployment**
  - Dockerfile for Railway
  - railway.toml configuration
  - docker-compose for local dev
  - Health check endpoints

## Quick Start

### Local Development

1. **Set up environment:**
   ```bash
   cd ~/code/wellness-librarian
   cp .env.example .env
   # Edit .env with your API keys:
   # - OPENAI_API_KEY
   # - ANTHROPIC_API_KEY
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Run ingestion (first time only):**
   ```bash
   python scripts/ingest.py
   ```

4. **Start the server:**
   ```bash
   python -m uvicorn backend.main:app --reload
   ```

5. **Open browser:**
   Navigate to http://localhost:8000

### Railway Deployment (with Supabase)

#### Initial Setup (One-Time)

1. **Create Supabase project** at https://supabase.com
   - Enable pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`
   - Note the connection string and API key

2. **Connect repository to Railway**
   ```bash
   railway login
   railway link  # Link to existing project or create new
   ```

3. **Set environment variables** (Railway Dashboard or CLI):
   ```bash
   railway variables set OPENAI_API_KEY=sk-...
   railway variables set ANTHROPIC_API_KEY=sk-ant-...
   railway variables set VOYAGE_API_KEY=pa-...
   railway variables set ADMIN_API_KEY=$(openssl rand -hex 32)
   railway variables set RAG_PIPELINE=enhanced
   railway variables set STORE_BACKEND=supabase
   railway variables set SUPABASE_URL=https://xxx.supabase.co
   railway variables set SUPABASE_KEY=eyJhbG...
   railway variables set SUPABASE_DB_URL=postgresql://postgres:xxx@db.xxx.supabase.co:5432/postgres
   ```

4. **Deploy**:
   ```bash
   railway up
   # or push to GitHub and let Railway auto-deploy
   ```

5. **Populate Supabase** (one-time):
   ```bash
   # Option A: Migrate existing data
   python scripts/migrate_to_supabase.py --create-index

   # Option B: Re-ingest from scratch
   python scripts/ingest_v2.py --target supabase --reset
   ```

#### Why Supabase is Better

With Supabase, there's no need for:
- Railway volumes
- Volume seeding scripts
- `init-data.sh` entrypoint

All data lives in the managed Supabase Postgres database, making deployments simpler and more reliable.

#### Updating Data

To update the indexed data:
```bash
# Re-ingest to Supabase
python scripts/ingest_v2.py --target supabase --reset
```

No container restarts or volume operations needed!

### Railway Deployment (Legacy - with Volumes)

<details>
<summary>Click to expand legacy volume-based deployment</summary>

#### Initial Setup (One-Time)

1. **Connect repository to Railway**
   ```bash
   railway login
   railway link  # Link to existing project or create new
   ```

2. **Set environment variables** (Railway Dashboard or CLI):
   ```bash
   railway variables set OPENAI_API_KEY=sk-...
   railway variables set ANTHROPIC_API_KEY=sk-ant-...
   railway variables set VOYAGE_API_KEY=pa-...
   railway variables set ADMIN_API_KEY=$(openssl rand -hex 32)
   railway variables set RAG_PIPELINE=enhanced
   railway variables set STORE_BACKEND=sqlite
   ```

3. **Create and attach volume**:
   ```bash
   # Create volume mounted at /app/data
   railway volume add --mount-path /app/data

   # Verify volume
   railway volume list
   ```

4. **Deploy**:
   ```bash
   railway up
   # or push to GitHub and let Railway auto-deploy
   ```

#### How Volume Seeding Works

The Dockerfile uses a two-directory approach:
- `/app/data-seed/` - Baked-in data from the Docker image
- `/app/data/` - Volume mount point (empty on first deploy)

The `scripts/init-data.sh` entrypoint:
1. Checks if `/app/data/` is empty
2. If empty, copies from `/app/data-seed/`
3. Starts the application

This ensures the volume is automatically seeded on first deploy.

#### Updating Data

To update the indexed data:
1. Re-run ingestion locally: `python scripts/ingest_v2.py --reset`
2. Commit and push (data-seed gets updated in image)
3. SSH into container and re-seed:
   ```bash
   railway ssh
   # Inside container:
   cp -r /app/data-seed/* /app/data/
   exit
   ```

</details>

#### Useful Commands

```bash
# Check project status
railway status

# View logs
railway logs

# SSH into running container
railway ssh

# Check volume usage
railway volume list

# Redeploy
railway redeploy
```

## Key Paths

| Path | Description |
|------|-------------|
| `~/code/wellness-librarian` | Project root |
| `~/Documents/wellness_evolution_community/vimeo_transcripts_rag` | Vimeo transcripts (members-only) |
| `~/Documents/wellness_evolution_community/youtube_transcripts` | YouTube transcripts (public) |
| `./data/chroma_db` | Vector database |

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Total Videos | 195 (76 Vimeo + 119 YouTube) |
| Total Chunks | 24,700 |
| Vimeo Chunks | 3,622 (members-only) |
| YouTube Chunks | 21,078 (public) |
| Categories | 17 |
| Estimated Embedding Cost | $0.12 |

## API Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/api/chat` | POST | Chat with librarian | - |
| `/api/search` | POST | Semantic search | - |
| `/api/sources` | GET | List categories | - |
| `/api/feedback` | POST | Submit thumbs up/down | - |
| `/api/recommend` | POST | Transcript → video recommendations | Admin |
| `/api/community-post` | POST | Generate WhatsApp community post | Admin |
| `/api/ingest` | POST | Add transcript | Admin |
| `/api/health` | GET | Health check | - |

## Design Decisions

1. **Multi-Source Support:** Source-agnostic fields (`video_id`, `video_url`) with auto-detection
2. **Access Levels:** YouTube=public, Vimeo=members_only for future gating
3. **VTT Parsing:** Timestamps preserved for future deep-linking
4. **Chunk Strategy:** 1000 chars, 200 overlap, semantic boundaries
5. **Python Version:** 3.12 for ChromaDB compatibility
6. **Deterministic IDs:** `{video_id}::chunk::{index}` for idempotency
7. **Frontend Stack:** No build step (Tailwind CDN + Alpine.js)
8. **WEC Brand Polish:** CSS-only decorative elements (inline SVG, gradients) - no external images to manage

### RAG Enhancement Design Decisions

9. **Parent-Child Architecture:** Search small chunks (250 tokens) for precision, return larger parents (500-2000 tokens) for context
10. **Token-Based Chunking:** Using tiktoken `cl100k_base` for accurate counts aligned with LLM context windows
11. **SQLite DocStore:** Simpler than putting large documents in vector store; efficient ID-based retrieval
12. **Voyage vs OpenAI:** Voyage embeddings optimized for retrieval; separate query/document embedding modes
13. **MMR Lambda=0.5:** Balanced relevance/diversity; tunable per use case
14. **Max 1 Chunk/Video:** Ensures maximum video diversity in results (changed from 2 after testing showed condition-specific videos still dominated general queries)
15. **Session-Based A/B:** Deterministic routing ensures consistent user experience within session

### Citation & Feedback Design Decisions

16. **Timestamp Mapping:** Accumulate VTT segment character counts to map chunk `start_char` → video timestamp
17. **Deep-Link Format:** YouTube `?t=` vs Vimeo `#t=s` respects each platform's URL conventions
18. **Numbered Citations:** `[1]`, `[2]` format chosen over footnote-style for inline readability
19. **Feedback in Docstore:** Same SQLite DB as parent chunks keeps deployment simple (single file)
20. **Glass Box Excerpts:** 500-char truncation balances visibility with UI space constraints
21. **Persistent Disclaimer:** Always-visible banner (not dismissible) for consistent legal compliance

### Critic Layer Design Decisions

22. **Haiku for Critic:** Fast model (~1s) keeps total latency acceptable; health verification doesn't need creative reasoning
23. **Binary Output Format:** "VERIFIED" or corrected text simplifies parsing; avoids critic hallucinating its own analysis
24. **600-char Excerpts:** Enough context for verification without bloating prompt; keeps critic cost low
25. **Fail-Open Design:** On critic error, original response passes through (avoids blocking user on transient failures)
26. **No Streaming Critic:** Buffering entire response would defeat streaming UX; accept tradeoff for streaming mode

### Conversation History Design Decisions

27. **Client-Side History:** Frontend owns conversation state; no server session storage needed. Keeps backend stateless.
28. **History → Dict Conversion:** Route layer converts Pydantic ChatMessage to plain dict; pipeline layer stays framework-agnostic
29. **RAG Context on Current Only:** History contains raw user/assistant messages; only current turn gets retrieval context. Avoids re-retrieving for past queries.

### Supabase Migration Design Decisions

30. **pgvector over ChromaDB:** Standard Postgres with vector extension is more portable and maintainable than specialized vector DB
31. **IVFFlat Indexing:** Uses inverted file index with sqrt(n) lists for optimal query performance on ~17k vectors
32. **Dual-Backend Architecture:** Config toggle (`store_backend`) allows gradual migration; local dev uses SQLite, production uses Supabase
33. **psycopg2-binary:** Binary wheels avoid needing libpq-dev in all environments; acceptable for production use
34. **Lazy Initialization:** Backend-specific stores only initialized when first accessed; prevents errors if config not set
35. **Stateless Containers:** With Supabase, all persistent data lives in managed DB; no Railway volumes needed

## Monthly Cost Estimate

| Service | Cost |
|---------|------|
| Railway (Hobby) | ~$5 |
| OpenAI Embeddings | ~$0.50 initial, ~$0.01 ongoing |
| Voyage Embeddings/Reranking | ~$1-3 (usage dependent) |
| Claude API (generation) | ~$5-20 (usage dependent) |
| Claude API (critic) | ~$1-5 (usage dependent) |
| **Total** | **~$15-35/month** |
