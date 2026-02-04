"""
Abstract base class for RAG pipelines.

Defines the interface that all RAG pipelines must implement,
enabling pluggable retrieval strategies for A/B testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class PipelineResult:
    """
    Result from a RAG pipeline execution.

    Contains both the generated response and metadata about the pipeline
    used, enabling A/B test tracking.
    """

    response: str
    sources: list[dict]
    retrieval_count: int
    model_used: str
    pipeline_name: str  # For A/B tracking: "legacy" or "enhanced"
    pipeline_metadata: dict = field(default_factory=dict)  # Extra pipeline-specific info


class RAGPipeline(ABC):
    """
    Abstract base class for RAG pipelines.

    A pipeline encapsulates the full retrieval-augmented generation flow:
    1. Query processing
    2. Retrieval (with whatever strategy)
    3. Context assembly
    4. LLM generation

    Implementations can vary in retrieval strategy while maintaining
    a consistent interface for the API layer.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the pipeline identifier (e.g., 'legacy', 'enhanced')."""
        pass

    @abstractmethod
    def generate(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """
        Generate a response synchronously.

        Args:
            user_message: The user's question
            category: Optional category filter for retrieval
            top_k: Number of chunks to retrieve
            history: Previous conversation turns [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            PipelineResult with response and metadata
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """
        Generate a response asynchronously.

        Args:
            user_message: The user's question
            category: Optional category filter for retrieval
            top_k: Number of chunks to retrieve
            history: Previous conversation turns [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            PipelineResult with response and metadata
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        user_message: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> tuple[AsyncIterator[str], dict]:
        """
        Generate a streaming response.

        Args:
            user_message: The user's question
            category: Optional category filter for retrieval
            top_k: Number of chunks to retrieve
            history: Previous conversation turns [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            Tuple of:
            - AsyncIterator yielding response text chunks
            - Dict with sources and metadata (available immediately)
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Get pipeline statistics.

        Returns:
            Dict with pipeline-specific metrics
        """
        pass
