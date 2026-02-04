"""
A/B test router for RAG pipelines.

Routes requests between legacy and enhanced pipelines based on:
- Configuration (force one pipeline)
- Session-based deterministic routing (A/B test mode)
"""

import hashlib
import logging
from typing import AsyncIterator, Optional

from backend.config import get_settings
from backend.rag.pipelines.base import RAGPipeline, PipelineResult
from backend.rag.pipelines.legacy import LegacyPipeline
from backend.rag.pipelines.enhanced import EnhancedPipeline

logger = logging.getLogger(__name__)


class ABTestRouter(RAGPipeline):
    """
    A/B test router that routes requests to different pipelines.

    Routing modes:
    - "legacy": Always use legacy pipeline
    - "enhanced": Always use enhanced pipeline
    - "ab_test": Route based on session_id hash

    When in ab_test mode, routing is deterministic based on session_id
    so the same user gets the same pipeline throughout their session.
    """

    def __init__(
        self,
        legacy_pipeline: LegacyPipeline | None = None,
        enhanced_pipeline: EnhancedPipeline | None = None,
    ):
        """
        Initialize the router.

        Args:
            legacy_pipeline: Legacy pipeline instance (default: create new)
            enhanced_pipeline: Enhanced pipeline instance (default: create new)
        """
        self._legacy = legacy_pipeline
        self._enhanced = enhanced_pipeline

    def _get_legacy(self) -> LegacyPipeline:
        """Get or create legacy pipeline."""
        if self._legacy is None:
            self._legacy = LegacyPipeline()
        return self._legacy

    def _get_enhanced(self) -> EnhancedPipeline:
        """Get or create enhanced pipeline."""
        if self._enhanced is None:
            self._enhanced = EnhancedPipeline()
        return self._enhanced

    @property
    def name(self) -> str:
        return "router"

    def select_pipeline(self, session_id: str | None = None) -> RAGPipeline:
        """
        Select which pipeline to use.

        Args:
            session_id: Optional session ID for deterministic routing

        Returns:
            Selected pipeline instance
        """
        settings = get_settings()

        if settings.rag_pipeline == "legacy":
            return self._get_legacy()

        if settings.rag_pipeline == "enhanced":
            return self._get_enhanced()

        # A/B test mode - use session_id for deterministic routing
        if session_id:
            # Hash session_id to get deterministic 0-1 value
            hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
            ratio = (hash_value % 1000) / 1000.0

            if ratio < settings.ab_test_ratio:
                logger.debug(f"Routing session {session_id[:8]}... to enhanced pipeline")
                return self._get_enhanced()
            else:
                logger.debug(f"Routing session {session_id[:8]}... to legacy pipeline")
                return self._get_legacy()

        # No session_id - default to legacy for safety
        logger.debug("No session_id provided, defaulting to legacy pipeline")
        return self._get_legacy()

    def generate(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """
        Generate response using selected pipeline.

        Args:
            user_message: The user's question
            category: Optional category filter
            top_k: Number of results
            session_id: Session ID for A/B routing
            history: Previous conversation turns

        Returns:
            PipelineResult from selected pipeline
        """
        pipeline = self.select_pipeline(session_id)
        logger.info(f"[router] Selected {pipeline.name} pipeline for request")
        return pipeline.generate(user_message, category, top_k, history)

    async def generate_async(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """Generate response asynchronously using selected pipeline."""
        pipeline = self.select_pipeline(session_id)
        logger.info(f"[router] Selected {pipeline.name} pipeline for request")
        return await pipeline.generate_async(user_message, category, top_k, history)

    async def generate_stream(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> tuple[AsyncIterator[str], dict]:
        """Generate streaming response using selected pipeline."""
        pipeline = self.select_pipeline(session_id)
        logger.info(f"[router] Selected {pipeline.name} pipeline for streaming request")
        return await pipeline.generate_stream(user_message, category, top_k, history)

    def get_stats(self) -> dict:
        """Get statistics from both pipelines."""
        return {
            "router_mode": get_settings().rag_pipeline,
            "ab_test_ratio": get_settings().ab_test_ratio,
            "legacy": self._get_legacy().get_stats(),
            "enhanced": self._get_enhanced().get_stats(),
        }


# Module-level singleton
_router: ABTestRouter | None = None


def get_router() -> ABTestRouter:
    """Get the singleton router instance."""
    global _router
    if _router is None:
        _router = ABTestRouter()
    return _router


def generate_response(
    user_message: str,
    category: Optional[str] = None,
    top_k: Optional[int] = None,
    session_id: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> PipelineResult:
    """
    Generate a response using the configured pipeline.

    This is the main entry point for the RAG system.

    Args:
        user_message: The user's question
        category: Optional category filter
        top_k: Number of results
        session_id: Session ID for A/B routing
        history: Previous conversation turns

    Returns:
        PipelineResult with response and metadata
    """
    router = get_router()
    return router.generate(user_message, category, top_k, session_id, history)


async def generate_response_async(
    user_message: str,
    category: Optional[str] = None,
    top_k: Optional[int] = None,
    session_id: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> PipelineResult:
    """Generate a response asynchronously."""
    router = get_router()
    return await router.generate_async(user_message, category, top_k, session_id, history)


async def generate_response_stream(
    user_message: str,
    category: Optional[str] = None,
    top_k: Optional[int] = None,
    session_id: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> tuple[AsyncIterator[str], dict]:
    """Generate a streaming response."""
    router = get_router()
    return await router.generate_stream(user_message, category, top_k, session_id, history)
