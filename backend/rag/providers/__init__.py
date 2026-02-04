"""
Embedding providers package.

Provides a pluggable interface for different embedding backends
(OpenAI, Voyage AI, etc.) with consistent APIs.
"""

from backend.rag.providers.base import EmbeddingProvider
from backend.rag.providers.openai_provider import OpenAIEmbeddingProvider
from backend.rag.providers.voyage_provider import VoyageEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "VoyageEmbeddingProvider",
]
