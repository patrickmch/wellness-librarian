"""
RAG pipeline implementations.

Provides pluggable RAG pipelines including:
- LegacyPipeline: Original implementation using OpenAI embeddings
- EnhancedPipeline: Parent-child retrieval with reranking
- ABTestRouter: Routes requests between pipelines for A/B testing
"""

from backend.rag.pipelines.base import RAGPipeline, PipelineResult
from backend.rag.pipelines.legacy import LegacyPipeline
from backend.rag.pipelines.enhanced import EnhancedPipeline
from backend.rag.pipelines.router import (
    ABTestRouter,
    get_router,
    generate_response,
    generate_response_async,
    generate_response_stream,
)

__all__ = [
    # Base
    "RAGPipeline",
    "PipelineResult",
    # Pipelines
    "LegacyPipeline",
    "EnhancedPipeline",
    # Router
    "ABTestRouter",
    "get_router",
    # Convenience functions
    "generate_response",
    "generate_response_async",
    "generate_response_stream",
]
