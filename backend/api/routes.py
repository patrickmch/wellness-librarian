"""
API route handlers for the Wellness Librarian.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse

from backend.config import get_settings
from backend.api.models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceInfo,
    SourcesResponse,
    CategoryInfo,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    ErrorResponse,
    PipelineMetadata,
    FeedbackRequest,
    FeedbackResponse,
)
from backend.rag.docstore.sqlite_store import get_docstore
from backend.rag.pipelines import (
    generate_response,
    generate_response_async,
    generate_response_stream,
)
from backend.rag.retriever import retrieve, get_stats
from backend.rag.vectorstore import get_document_count
from backend.ingestion.pipeline import ingest_single

logger = logging.getLogger(__name__)

router = APIRouter()


def verify_admin_key(x_admin_key: Optional[str] = Header(None)) -> str:
    """Verify admin API key for protected endpoints."""
    settings = get_settings()
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")
    return x_admin_key


@router.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat(request: ChatRequest):
    """
    Chat with the Wellness Librarian.

    Send a question and receive an AI-generated response based on
    the video transcript library, with source citations.

    Optionally provide session_id for A/B test routing between
    legacy and enhanced pipelines.
    """
    try:
        # Convert ChatMessage models to dicts for the pipeline
        history = None
        if request.history:
            history = [{"role": h.role, "content": h.content} for h in request.history]

        if request.stream:
            # Return streaming response
            # For streaming, we use the async generator and return metadata separately
            stream, metadata = await generate_response_stream(
                user_message=request.message,
                category=request.category,
                session_id=request.session_id,
                history=history,
            )

            async def generate():
                async for chunk in stream:
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={
                    "X-Pipeline-Used": metadata.get("pipeline_name", "unknown"),
                },
            )

        # Non-streaming response
        result = await generate_response_async(
            user_message=request.message,
            category=request.category,
            session_id=request.session_id,
            history=history,
        )

        # Build pipeline metadata from result
        pipeline_meta = None
        if result.pipeline_metadata:
            pipeline_meta = PipelineMetadata(
                embedding_model=result.pipeline_metadata.get("embedding_model"),
                reranking_enabled=result.pipeline_metadata.get("reranking_enabled"),
                diversity_enabled=result.pipeline_metadata.get("diversity_enabled"),
                max_per_video=result.pipeline_metadata.get("max_per_video"),
                children_searched=result.pipeline_metadata.get("children_searched"),
                parents_expanded=result.pipeline_metadata.get("parents_expanded"),
                critic_enabled=result.pipeline_metadata.get("critic_enabled"),
                critic_corrected=result.pipeline_metadata.get("critic_corrected"),
            )

        return ChatResponse(
            response=result.response,
            sources=[
                SourceInfo(
                    title=s.get("title", ""),
                    category=s.get("category", ""),
                    video_url=s.get("video_url", ""),
                    duration=s.get("duration", ""),
                    video_id=s.get("video_id", ""),
                    source=s.get("source", ""),
                    start_time_seconds=s.get("start_time_seconds", 0),
                    excerpt=s.get("excerpt"),
                )
                for s in result.sources
            ],
            retrieval_count=result.retrieval_count,
            pipeline_used=result.pipeline_name,
            pipeline_metadata=pipeline_meta,
        )

    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse, responses={500: {"model": ErrorResponse}})
async def search(request: SearchRequest):
    """
    Perform semantic search across transcripts.

    Returns relevant transcript chunks ranked by similarity.
    """
    try:
        retrieval = retrieve(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
        )

        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    text=r.text,
                    score=r.score,
                    title=r.title,
                    category=r.category,
                    video_url=r.video_url,
                    chunk_index=r.metadata.get("chunk_index", 0),
                    total_chunks=r.metadata.get("total_chunks", 1),
                    source=r.source,
                )
                for r in retrieval.results
            ],
            total_results=retrieval.total_results,
        )

    except Exception as e:
        logger.exception("Error in search endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources", response_model=SourcesResponse, responses={500: {"model": ErrorResponse}})
async def list_sources():
    """
    List available video categories and statistics.

    Returns information about the indexed transcript library (v2 enhanced index).
    """
    try:
        # Use v2 docstore stats for accurate counts
        docstore = get_docstore()
        stats = docstore.get_stats()

        return SourcesResponse(
            total_videos=stats.get("unique_videos", 0),
            total_chunks=stats.get("total_parent_chunks", 0),
            categories=[
                CategoryInfo(
                    name=cat,
                    video_count=count,
                )
                for cat, count in sorted(stats.get("categories", {}).items())
            ],
        )

    except Exception as e:
        logger.exception("Error in sources endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ingest_transcript(
    request: IngestRequest,
    _: str = Depends(verify_admin_key),
):
    """
    Ingest a new transcript into the vector store.

    Requires admin API key in X-Admin-Key header.
    """
    try:
        settings = get_settings()
        filepath = Path(request.filepath)

        if not filepath.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {filepath}")

        stats = ingest_single(
            filepath=filepath,
            source_dir=settings.transcript_source_dir,
            force=request.force,
        )

        if stats.errors:
            return IngestResponse(
                success=False,
                message=f"Ingestion failed: {stats.errors[0]}",
                chunks_created=0,
                chunks_skipped=0,
            )

        return IngestResponse(
            success=True,
            message=f"Successfully ingested {filepath.name}",
            chunks_created=stats.chunks_created,
            chunks_skipped=stats.chunks_skipped,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in ingest endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and basic statistics.
    """
    try:
        chunk_count = get_document_count()

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            vector_store_chunks=chunk_count,
        )

    except Exception as e:
        logger.warning(f"Health check warning: {e}")
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            vector_store_chunks=0,
        )


@router.post("/feedback", response_model=FeedbackResponse, responses={500: {"model": ErrorResponse}})
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback (thumbs up/down) for a response.

    Stores feedback in SQLite for analytics and improvement tracking.
    """
    try:
        docstore = get_docstore()
        feedback_id = docstore.add_feedback(
            message_id=request.message_id,
            feedback_type=request.feedback_type,
            session_id=request.session_id,
            query=request.query,
            parent_ids=request.parent_ids,
        )

        return FeedbackResponse(
            status="ok",
            feedback_id=feedback_id,
        )

    except Exception as e:
        logger.exception("Error in feedback endpoint")
        raise HTTPException(status_code=500, detail=str(e))
