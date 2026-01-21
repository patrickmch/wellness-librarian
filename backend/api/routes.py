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
)
from backend.rag.generator import (
    generate_response,
    generate_response_async,
    generate_response_stream,
)
from backend.rag.retriever import retrieve, get_stats
from backend.rag.vectorstore import get_document_count
from backend.ingestion.pipeline import ingest_single
from backend.ingestion.loader import get_categories, get_transcript_count

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
    """
    try:
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in generate_response_stream(
                    user_message=request.message,
                    category=request.category,
                ):
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type="text/plain",
            )

        # Non-streaming response
        result = await generate_response_async(
            user_message=request.message,
            category=request.category,
        )

        return ChatResponse(
            response=result.response,
            sources=[
                SourceInfo(
                    title=s.get("title", ""),
                    category=s.get("category", ""),
                    vimeo_url=s.get("vimeo_url", ""),
                    duration=s.get("duration", ""),
                    vimeo_id=s.get("vimeo_id", ""),
                )
                for s in result.sources
            ],
            retrieval_count=result.retrieval_count,
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
                    vimeo_url=r.vimeo_url,
                    chunk_index=r.metadata.get("chunk_index", 0),
                    total_chunks=r.metadata.get("total_chunks", 1),
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

    Returns information about the indexed transcript library.
    """
    try:
        settings = get_settings()
        stats = get_stats()
        categories = get_categories(settings.transcript_source_dir)

        return SourcesResponse(
            total_videos=get_transcript_count(settings.transcript_source_dir),
            total_chunks=get_document_count(),
            categories=[
                CategoryInfo(
                    name=cat,
                    video_count=stats.get("categories", {}).get(cat, 0),
                )
                for cat in categories
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
