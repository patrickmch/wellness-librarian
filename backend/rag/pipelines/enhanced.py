"""
Enhanced RAG pipeline with parent-child retrieval.

Uses:
- Voyage AI embeddings
- Parent-child chunking
- Diversity filtering (per-video dedup + MMR)
- Voyage reranking
"""

import logging
from typing import AsyncIterator, Optional

from backend.config import get_settings
from backend.rag.generator import (
    SYSTEM_PROMPT,
    get_anthropic_client,
    get_async_anthropic_client,
)
from backend.rag.retrieval.parent_child import (
    ParentChildRetriever,
    ParentRetrievalResponse,
)
from backend.rag.pipelines.base import RAGPipeline, PipelineResult
from backend.rag.pipelines.critic import verify_response, verify_response_async

logger = logging.getLogger(__name__)


def build_messages_with_history(
    user_prompt: str,
    history: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Build messages list including conversation history.

    Args:
        user_prompt: The current user prompt (with context)
        history: Previous conversation turns [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        List of messages for the LLM
    """
    messages = []

    # Add previous conversation turns
    if history:
        for turn in history:
            messages.append({
                "role": turn["role"],
                "content": turn["content"],
            })

    # Add current user message with context
    messages.append({"role": "user", "content": user_prompt})

    return messages


def build_enhanced_context_prompt(
    retrieval: ParentRetrievalResponse,
    user_message: str
) -> str:
    """
    Build the user prompt with retrieved parent context.

    Uses numbered citations [1], [2], etc. for deep-linking support.
    """
    if not retrieval.results:
        context_section = "No relevant video content was found for this query."
    else:
        context_parts = []
        for i, result in enumerate(retrieval.results, 1):
            context_parts.append(
                f"[{i}] From \"{result.title}\" ({result.category}):\n"
                f"{result.text}"
            )
        context_section = "\n\n---\n\n".join(context_parts)

    return f"""## Retrieved Video Transcripts

{context_section}

---

## User's Question

{user_message}

When citing information, use bracketed numbers [1], [2], etc. to reference the sources above.
Example: "She recommends 400mg of magnesium before bed [1]."
Always cite specific health claims, supplement dosages, and recommendations with their source number.
If multiple sources support a claim, cite all of them: [1][2]."""


class EnhancedPipeline(RAGPipeline):
    """
    Enhanced RAG pipeline with parent-child retrieval.

    Pipeline steps:
    1. Embed query with Voyage (query mode)
    2. Search child chunks in ChromaDB
    3. Apply diversity filters (per-video dedup, MMR)
    4. Expand to unique parent chunks from docstore
    5. Rerank parents with Voyage rerank-2
    6. Generate response with Claude

    Benefits over legacy:
    - Better precision from smaller search chunks
    - Better context from larger parent chunks
    - Reduced video domination from diversity filters
    - Improved relevance from reranking
    """

    def __init__(self, retriever: ParentChildRetriever | None = None):
        """
        Initialize enhanced pipeline.

        Args:
            retriever: ParentChildRetriever instance (default: create new)
        """
        self._retriever = retriever

    def _get_retriever(self) -> ParentChildRetriever:
        """Get or create retriever."""
        if self._retriever is None:
            self._retriever = ParentChildRetriever()
        return self._retriever

    @property
    def name(self) -> str:
        return "enhanced"

    def generate(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """Generate response using enhanced retrieval."""
        settings = get_settings()
        top_k = top_k or settings.rerank_top_n or settings.default_top_k

        # Retrieve using parent-child strategy
        retriever = self._get_retriever()
        retrieval = retriever.retrieve(
            query=user_message,
            final_top_k=top_k,
            category=category,
        )

        logger.info(
            f"[{self.name}] Retrieved {retrieval.total_results} parents "
            f"(searched {retrieval.children_searched} children, "
            f"expanded {retrieval.parents_expanded} parents) "
            f"for query: {user_message[:50]}..."
        )

        # Build prompt with context
        user_prompt = build_enhanced_context_prompt(retrieval, user_message)

        # Build messages with history
        messages = build_messages_with_history(user_prompt, history)

        # Generate response with Claude
        client = get_anthropic_client()

        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_response_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        response_text = message.content[0].text
        sources = retrieval.get_sources()

        # Apply critic verification if enabled
        critic_corrected = False
        if settings.enable_critic:
            response_text, critic_corrected = verify_response(response_text, sources)
            if critic_corrected:
                logger.info(f"[{self.name}] Response was corrected by critic")

        return PipelineResult(
            response=response_text,
            sources=sources,
            retrieval_count=retrieval.total_results,
            model_used=settings.claude_model,
            pipeline_name=self.name,
            pipeline_metadata={
                "embedding_model": settings.voyage_embedding_model,
                "reranking_enabled": settings.enable_reranking,
                "diversity_enabled": True,
                "max_per_video": settings.max_chunks_per_video,
                "children_searched": retrieval.children_searched,
                "parents_expanded": retrieval.parents_expanded,
                "critic_enabled": settings.enable_critic,
                "critic_corrected": critic_corrected,
            },
        )

    async def generate_async(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """Generate response asynchronously using enhanced retrieval."""
        settings = get_settings()
        top_k = top_k or settings.rerank_top_n or settings.default_top_k

        # Retrieve (sync for now - retriever doesn't have async yet)
        retriever = self._get_retriever()
        retrieval = retriever.retrieve(
            query=user_message,
            final_top_k=top_k,
            category=category,
        )

        logger.info(
            f"[{self.name}] Retrieved {retrieval.total_results} parents "
            f"(searched {retrieval.children_searched} children, "
            f"expanded {retrieval.parents_expanded} parents) "
            f"for query: {user_message[:50]}..."
        )

        # Build prompt with context
        user_prompt = build_enhanced_context_prompt(retrieval, user_message)

        # Build messages with history
        messages = build_messages_with_history(user_prompt, history)

        # Generate response with Claude
        client = get_async_anthropic_client()

        message = await client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_response_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        response_text = message.content[0].text
        sources = retrieval.get_sources()

        # Apply critic verification if enabled
        critic_corrected = False
        if settings.enable_critic:
            response_text, critic_corrected = await verify_response_async(response_text, sources)
            if critic_corrected:
                logger.info(f"[{self.name}] Response was corrected by critic")

        return PipelineResult(
            response=response_text,
            sources=sources,
            retrieval_count=retrieval.total_results,
            model_used=settings.claude_model,
            pipeline_name=self.name,
            pipeline_metadata={
                "embedding_model": settings.voyage_embedding_model,
                "reranking_enabled": settings.enable_reranking,
                "diversity_enabled": True,
                "max_per_video": settings.max_chunks_per_video,
                "children_searched": retrieval.children_searched,
                "parents_expanded": retrieval.parents_expanded,
                "critic_enabled": settings.enable_critic,
                "critic_corrected": critic_corrected,
            },
        )

    async def generate_stream(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> tuple[AsyncIterator[str], dict]:
        """Generate streaming response using enhanced retrieval."""
        settings = get_settings()
        top_k = top_k or settings.rerank_top_n or settings.default_top_k

        # Retrieve
        retriever = self._get_retriever()
        retrieval = retriever.retrieve(
            query=user_message,
            final_top_k=top_k,
            category=category,
        )

        logger.info(
            f"[{self.name}] Retrieved {retrieval.total_results} parents "
            f"(searched {retrieval.children_searched} children, "
            f"expanded {retrieval.parents_expanded} parents) "
            f"for query: {user_message[:50]}..."
        )

        # Build prompt with context
        user_prompt = build_enhanced_context_prompt(retrieval, user_message)

        # Build messages with history
        messages = build_messages_with_history(user_prompt, history)

        # Prepare metadata (available immediately)
        # Note: Critic verification is skipped for streaming responses
        metadata = {
            "sources": retrieval.get_sources(),
            "retrieval_count": retrieval.total_results,
            "model_used": settings.claude_model,
            "pipeline_name": self.name,
            "pipeline_metadata": {
                "embedding_model": settings.voyage_embedding_model,
                "reranking_enabled": settings.enable_reranking,
                "diversity_enabled": True,
                "max_per_video": settings.max_chunks_per_video,
                "children_searched": retrieval.children_searched,
                "parents_expanded": retrieval.parents_expanded,
                "critic_enabled": False,  # Skipped for streaming
                "critic_corrected": False,
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
        """Get enhanced pipeline statistics."""
        retriever = self._get_retriever()
        return retriever.get_stats()


# Convenience function
def get_enhanced_pipeline() -> EnhancedPipeline:
    """Get an EnhancedPipeline instance."""
    return EnhancedPipeline()
