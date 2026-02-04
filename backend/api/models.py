"""
Pydantic models for API request/response schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field


# === Request Models ===

class ChatMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    category: Optional[str] = Field(None, description="Filter by category")
    stream: bool = Field(False, description="Enable streaming response")
    session_id: Optional[str] = Field(None, description="Session ID for A/B test routing")
    history: Optional[list[ChatMessage]] = Field(None, description="Previous conversation turns for context")


class SearchRequest(BaseModel):
    """Request body for search endpoint."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(8, ge=1, le=20, description="Number of results")
    category: Optional[str] = Field(None, description="Filter by category")


class IngestRequest(BaseModel):
    """Request body for ingest endpoint."""
    filepath: str = Field(..., description="Path to transcript file")
    force: bool = Field(False, description="Replace if exists")


class FeedbackRequest(BaseModel):
    """Request body for feedback endpoint."""
    message_id: str = Field(..., description="ID of the message being rated")
    feedback_type: str = Field(..., description="Feedback type: 'up' or 'down'")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    query: Optional[str] = Field(None, description="Original query text")
    parent_ids: Optional[list[str]] = Field(None, description="Parent chunk IDs used")


# === Response Models ===

class SourceInfo(BaseModel):
    """Information about a source video."""
    title: str
    category: str
    video_url: str
    duration: str
    video_id: str
    source: str = ""  # "youtube" or "vimeo"
    start_time_seconds: int = 0  # Video timestamp for deep-linking
    excerpt: Optional[str] = None  # Raw transcript excerpt for glass box


class PipelineMetadata(BaseModel):
    """Metadata about the pipeline used for the response."""
    embedding_model: Optional[str] = None
    reranking_enabled: Optional[bool] = None
    diversity_enabled: Optional[bool] = None
    max_per_video: Optional[int] = None
    children_searched: Optional[int] = None
    parents_expanded: Optional[int] = None
    critic_enabled: Optional[bool] = None
    critic_corrected: Optional[bool] = None


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    response: str
    sources: list[SourceInfo]
    retrieval_count: int
    pipeline_used: str = Field("legacy", description="Pipeline used: 'legacy' or 'enhanced'")
    pipeline_metadata: Optional[PipelineMetadata] = None


class SearchResult(BaseModel):
    """A single search result."""
    text: str
    score: float
    title: str
    category: str
    video_url: str
    chunk_index: int
    total_chunks: int
    source: str = ""  # "youtube" or "vimeo"


class SearchResponse(BaseModel):
    """Response body for search endpoint."""
    query: str
    results: list[SearchResult]
    total_results: int


class CategoryInfo(BaseModel):
    """Information about a category."""
    name: str
    video_count: int


class SourcesResponse(BaseModel):
    """Response body for sources endpoint."""
    total_videos: int
    total_chunks: int
    categories: list[CategoryInfo]


class IngestResponse(BaseModel):
    """Response body for ingest endpoint."""
    success: bool
    message: str
    chunks_created: int
    chunks_skipped: int


class HealthResponse(BaseModel):
    """Response body for health endpoint."""
    status: str
    version: str
    vector_store_chunks: int


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response body for feedback endpoint."""
    status: str
    feedback_id: str
