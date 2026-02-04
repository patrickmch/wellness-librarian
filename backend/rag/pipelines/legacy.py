"""
Legacy RAG pipeline implementation.

Wraps the original retriever and generator functionality into the
RAGPipeline interface, enabling it to participate in A/B testing
while preserving the existing behavior exactly.
"""

import logging
from typing import AsyncIterator, Optional

from backend.config import get_settings
from backend.rag.retriever import retrieve, get_stats as retriever_get_stats
from backend.rag.generator import (
    build_context_prompt,
    get_anthropic_client,
    get_async_anthropic_client,
    GenerationResult,
    SYSTEM_PROMPT,
)
from backend.rag.pipelines.base import RAGPipeline, PipelineResult

logger = logging.getLogger(__name__)


class LegacyPipeline(RAGPipeline):
    """
    Legacy RAG pipeline using OpenAI embeddings and flat retrieval.

    This wraps the original implementation to maintain backward compatibility
    while conforming to the RAGPipeline interface for A/B testing.

    Pipeline steps:
    1. Embed query with OpenAI text-embedding-3-small
    2. Retrieve top-k chunks via ChromaDB cosine similarity
    3. Filter by minimum score threshold
    4. Assemble context prompt
    5. Generate response with Claude
    """

    @property
    def name(self) -> str:
        return "legacy"

    def _build_messages(
        self,
        user_prompt: str,
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Build messages list including conversation history."""
        messages = []
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """Generate a response using the legacy retrieval strategy."""
        settings = get_settings()
        top_k = top_k or settings.default_top_k

        # Retrieve relevant context using the original retriever
        retrieval = retrieve(
            query=user_message,
            top_k=top_k,
            category=category,
        )

        logger.info(
            f"[{self.name}] Retrieved {retrieval.total_results} chunks "
            f"for query: {user_message[:50]}..."
        )

        # Build prompt with context
        user_prompt = build_context_prompt(retrieval, user_message)

        # Build messages with history
        messages = self._build_messages(user_prompt, history)

        # Generate response with Claude
        client = get_anthropic_client()

        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_response_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        response_text = message.content[0].text

        return PipelineResult(
            response=response_text,
            sources=retrieval.get_sources(),
            retrieval_count=retrieval.total_results,
            model_used=settings.claude_model,
            pipeline_name=self.name,
            pipeline_metadata={
                "embedding_model": settings.embedding_model,
                "similarity_threshold": settings.similarity_threshold,
            },
        )

    async def generate_async(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """Generate a response asynchronously using the legacy strategy."""
        settings = get_settings()
        top_k = top_k or settings.default_top_k

        # Retrieve relevant context (sync for now - could be made async)
        retrieval = retrieve(
            query=user_message,
            top_k=top_k,
            category=category,
        )

        logger.info(
            f"[{self.name}] Retrieved {retrieval.total_results} chunks "
            f"for query: {user_message[:50]}..."
        )

        # Build prompt with context
        user_prompt = build_context_prompt(retrieval, user_message)

        # Build messages with history
        messages = self._build_messages(user_prompt, history)

        # Generate response with Claude
        client = get_async_anthropic_client()

        message = await client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_response_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        response_text = message.content[0].text

        return PipelineResult(
            response=response_text,
            sources=retrieval.get_sources(),
            retrieval_count=retrieval.total_results,
            model_used=settings.claude_model,
            pipeline_name=self.name,
            pipeline_metadata={
                "embedding_model": settings.embedding_model,
                "similarity_threshold": settings.similarity_threshold,
            },
        )

    async def generate_stream(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> tuple[AsyncIterator[str], dict]:
        """Generate a streaming response using the legacy strategy."""
        settings = get_settings()
        top_k = top_k or settings.default_top_k

        # Retrieve relevant context
        retrieval = retrieve(
            query=user_message,
            top_k=top_k,
            category=category,
        )

        logger.info(
            f"[{self.name}] Retrieved {retrieval.total_results} chunks "
            f"for query: {user_message[:50]}..."
        )

        # Build prompt with context
        user_prompt = build_context_prompt(retrieval, user_message)

        # Build messages with history
        messages = self._build_messages(user_prompt, history)

        # Prepare metadata (available immediately)
        metadata = {
            "sources": retrieval.get_sources(),
            "retrieval_count": retrieval.total_results,
            "model_used": settings.claude_model,
            "pipeline_name": self.name,
            "pipeline_metadata": {
                "embedding_model": settings.embedding_model,
                "similarity_threshold": settings.similarity_threshold,
            },
        }

        # Create streaming generator
        async def stream_response() -> AsyncIterator[str]:
            client = get_async_anthropic_client()
            async with client.messages.stream(
                model=settings.claude_model,
                max_tokens=settings.max_response_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        return stream_response(), metadata

    def get_stats(self) -> dict:
        """Get legacy pipeline statistics."""
        base_stats = retriever_get_stats()
        return {
            "pipeline": self.name,
            "embedding_provider": "openai",
            "embedding_model": get_settings().embedding_model,
            "reranking_enabled": False,
            "diversity_filtering": False,
            **base_stats,
        }


# Convenience function for backward compatibility with existing code
def get_legacy_pipeline() -> LegacyPipeline:
    """Get a LegacyPipeline instance."""
    return LegacyPipeline()
