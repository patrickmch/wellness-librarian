# Wellness Librarian - Progress Tracking

## Current Status

**Phase:** Deployed to Production 🚀
**Last Updated:** 2026-02-10

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
