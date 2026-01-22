# Wellness Librarian - Progress Tracking

## Current Status

**Phase:** 6 of 6 - Deployment (Complete) + Multi-Source Support ✨
**Last Updated:** 2026-01-22

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

### Railway Deployment

1. Connect repository to Railway
2. Set environment variables:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `ADMIN_API_KEY` (generate a secure key)
3. Add volume mount for `/app/data/chroma_db`
4. Deploy!

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

## Monthly Cost Estimate

| Service | Cost |
|---------|------|
| Railway (Hobby) | ~$5 |
| OpenAI Embeddings | ~$0.50 initial, ~$0.01 ongoing |
| Claude API | ~$5-20 (usage dependent) |
| **Total** | **~$15-30/month** |
